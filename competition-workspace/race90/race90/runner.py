from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import math
import os
import shutil
import sys
import threading
import time
import uuid
from .io import atomic_json
from .process import run_command
from .selection import select_best
from .remote import Runpod
from .deliverables import final_notebooks


def event(directory, kind, **fields):
    item = {"time": time.time(), "event": kind, **fields}
    with (Path(directory) / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(item, ensure_ascii=False), flush=True)


def local_stage(cfg, task, run_dir, stage, deadline, threads, device="cpu", cancel=None):
    output = Path(run_dir) / task["name"]
    payload = {"cfg": cfg, "task": task, "output": str(output), "stage": stage,
               "deadline": deadline, "threads": threads, "device": "cuda:0" if device != "cpu" else "cpu"}
    job = output / f"{stage}-payload.json"
    atomic_json(job, payload)
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(threads)
    env["CUDA_VISIBLE_DEVICES"] = "" if device == "cpu" else str(device)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent.parent), cfg["project_root"], env.get("PYTHONPATH", "")])
    return run_command([sys.executable, "-m", "race90", "_worker", str(job)], deadline,
                       log=output / f"{stage}.log", env=env, cwd=cfg["project_root"], cancel=cancel)


def snapshot(cfg, run_dir):
    state = {}
    for task in cfg["tasks"]:
        try:
            selected, info = select_best(cfg, task, Path(run_dir) / task["name"])
            state[task["name"]] = {"candidate": selected.name, "score": info["score"], "stage": info["stage"]}
        except ValueError as e:
            state[task["name"]] = {"error": str(e)}
    atomic_json(Path(run_dir) / "status.json", state)
    return state


def run(cfg, resume=None, deadline_epoch=None):
    started = time.time()
    if deadline_epoch is not None and not math.isfinite(float(deadline_epoch)):
        raise ValueError("Deadline must be finite")
    digest_input = {"tasks": cfg["tasks"], "seed": cfg["seed"], "data_root": cfg["data_root"],
                    "total_minutes": cfg["total_minutes"], "finalize_minutes": cfg["finalize_minutes"],
                    "time_management": cfg.get("time_management", {})}
    fingerprints = {}
    from .config import resolve_data
    for task in cfg["tasks"]:
        for key in ("train_csv", "test_csv", "sample_submission", "data_yaml", "notebook"):
            if task.get(key):
                path = resolve_data(cfg, task[key])
                stat = path.stat()
                fingerprints[str(path)] = [stat.st_size, stat.st_mtime_ns]
        if task.get("image_root"):
            for path in sorted(resolve_data(cfg, task["image_root"]).rglob("*")):
                if path.is_file():
                    stat = path.stat()
                    fingerprints[str(path)] = [stat.st_size, stat.st_mtime_ns]
    digest_input["input_metadata"] = fingerprints
    from .bundle import source_files
    digest_input["source_hashes"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for name, path in source_files(cfg).items()}
    config_digest = hashlib.sha256(json.dumps(digest_input, sort_keys=True, default=str).encode()).hexdigest()
    if resume:
        if deadline_epoch is not None:
            raise ValueError("--resume retains the original deadline; omit --deadline-epoch")
        run_dir = Path(resume).resolve()
        old = json.loads((run_dir / "run.json").read_text())
        if old.get("task_names") != [t["name"] for t in cfg["tasks"]]:
            raise ValueError("Resume task names differ from the original run")
        if old.get("config_digest") != config_digest:
            raise ValueError("Resume configuration or input file metadata changed; start a new run")
        end = old["deadline"]  # Resume never grants another 90 minutes.
    else:
        run_dir = Path(cfg["output_root"]) / (time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6])
        maximum_end = started + cfg["total_minutes"] * 60
        end = min(float(deadline_epoch), maximum_end) if deadline_epoch is not None else maximum_end
        run_dir.mkdir(parents=True)
        atomic_json(run_dir / "run.json", {"started": started, "deadline": end,
                                          "task_names": [t["name"] for t in cfg["tasks"]], "config_digest": config_digest})
    from .timing import timeline, time_status
    plan = timeline(cfg, end)
    cfg["_timeline"] = plan
    atomic_json(run_dir / "timeline.json", plan)
    time_status(run_dir, plan, "training")
    compute_end = plan["compute_deadline"]
    cfg["dependency_outputs"] = {}
    cancel = threading.Event()
    errors = []
    remote = None
    from .resources import cpu_capacity
    total_threads = min(int(cfg.get("cpu_threads") or cpu_capacity()), cpu_capacity())
    workers = min(int(cfg["cpu_workers"]), len(cfg["tasks"]), total_threads)
    threads = max(1, total_threads // (workers + int(cfg["runpod"].get("enabled", False))))
    event(run_dir, "started", deadline=end, compute_deadline=compute_end, cpu_workers=workers,
          threads_per_worker=threads, run_dir=str(run_dir))
    gpu_tasks = [t for t in cfg["tasks"] if t.get("improve", True) and
                 t.get("resource", "gpu" if t["kind"] in {"image", "yolo"} else "cpu") == "gpu"]
    # Remote GPU provisioning overlaps CPU baselines immediately.
    prep_pool = ThreadPoolExecutor(max_workers=1)
    ready = None
    if gpu_tasks and cfg["runpod"].get("enabled", False) and time.time() < compute_end:
        remote = Runpod(cfg, run_dir, compute_end, cancel)
        ready = prep_pool.submit(remote.prepare)
    try:
        if time.time() < compute_end:
            from .scheduling import run_stages
            run_stages(cfg, run_dir, compute_end, workers, threads, cancel, ready,
                       event, local_stage, snapshot)
    except KeyboardInterrupt:
        cancel.set()
        event(run_dir, "interrupted", reason="Finalizing completed candidates")
    finally:
        time_status(run_dir, plan, "retrieval")
        cancel.set()
        prep_pool.shutdown(wait=True, cancel_futures=True)
        if remote is not None:
            remote.cleanup()
    selected = {}
    deliver = run_dir / "deliverables"
    deliver.mkdir(exist_ok=True)
    for task in cfg["tasks"]:
        try:
            path, info = select_best(cfg, task, run_dir / task["name"])
            selected[task["name"]] = (path, info)
            cfg["dependency_outputs"][task["name"]] = str(path)
            shutil.copy2(path / "submission.csv", deliver / f"{task['name']}_submission.csv")
            shutil.copy2(run_dir / task["name"] / "selected.json", deliver / f"{task['name']}_result.json")
        except Exception as e:
            errors.append(str(e))
    time_status(run_dir, plan, "notebooks")
    notebooks = final_notebooks(cfg, run_dir, selected, plan["notebook_deadline"])
    for task in cfg["tasks"]:
        name = task["name"]
        if notebooks.get(name, {}).get("status") != "done":
            (deliver / f"{name}.solution.ipynb").unlink(missing_ok=True)
    checklist = {"online_upload_completed": False, "exam_deadline": end,
                 "deliverable_deadline": plan["deliverable_deadline"],
                 "next_action": "제공 Notebook에 풀이 셀과 실행 출력을 반영하고 CSV를 제출한 뒤 접수 여부를 확인하세요.",
                 "tasks": {t["name"]: {"provided_notebook": t.get("notebook"),
                           "csv": f"{t['name']}_submission.csv" if t["name"] in selected else None,
                           "notebook": notebooks.get(t["name"], {})} for t in cfg["tasks"]}}
    atomic_json(deliver / "submission_checklist.json", checklist)
    time_status(run_dir, plan, "packaging")
    packaged = False
    try:
        previous_pack = run_dir / "submission_pack.zip"
        if previous_pack.exists():
            os.replace(previous_pack, run_dir / "submission_pack.previous.zip")
        run_command([sys.executable, "-m", "race90", "_package", str(run_dir),
                     str(plan["deliverable_deadline"])], plan["deliverable_deadline"],
                    log=run_dir / "packaging.log", cwd=str(Path(__file__).parent.parent))
        packaged = True
    except Exception as exc:
        errors.append(f"Packaging failed; individual verified files remain in deliverables: {exc}")
    from .bundle import source_zip
    (run_dir / "source.zip").write_bytes(source_zip(cfg))
    report = {"run_dir": str(run_dir), "deadline": end, "finished_at": time.time(),
              "elapsed_seconds": time.time() - started,
              "tasks": {name: {**info, "candidate": path.name} for name, (path, info) in selected.items()},
              "notebooks": notebooks, "errors": errors, "packaged": packaged, "timeline": plan,
              "complete": len(selected) == len(cfg["tasks"]) and all(n["status"] == "done" for n in notebooks.values())}
    report["fallback_tasks"] = [name for name, (_, info) in selected.items() if info.get("score") is None]
    report["within_deadline"] = report["finished_at"] <= end
    report["handoff_on_time"] = report["finished_at"] <= plan["deliverable_deadline"]
    report["compact_notebooks"] = [name for name, n in notebooks.items() if n.get("variant", "").startswith("compact")]
    report["provided_notebooks_used"] = all(t.get("notebook") for t in cfg["tasks"])
    report["submission_ready"] = bool(report["complete"] and not report["fallback_tasks"] and packaged
                                      and report["provided_notebooks_used"]
                                      and all(n.get("executed") for n in notebooks.values()) and report["handoff_on_time"])
    report["online_upload_completed"] = False
    time_status(run_dir, plan, "ready_for_upload" if report["submission_ready"] else "needs_attention",
                submission_ready=report["submission_ready"], fallback_tasks=report["fallback_tasks"])
    atomic_json(run_dir / "final.json", report)
    event(run_dir, "finished", complete=report["complete"], run_dir=str(run_dir))
    return report
