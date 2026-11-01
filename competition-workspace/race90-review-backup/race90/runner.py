from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import hashlib
import os
import queue
import shutil
import sys
import threading
import time
import uuid
import zipfile
from .io import atomic_json, sha256
from .process import run_command
from .selection import select_best
from .remote import Runpod


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


def monitor_futures(futures, cfg, run_dir, cancel):
    pending = set(futures)
    while pending:
        try:
            done, pending = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
        except KeyboardInterrupt:
            cancel.set()  # Signal subprocesses before executor __exit__ waits.
            raise
        for future in done:
            task, stage = futures[future]
            try:
                future.result()
                event(run_dir, "stage_done", task=task, stage=stage)
            except Exception as e:
                event(run_dir, "stage_failed", task=task, stage=stage, error=str(e))
        snapshot(cfg, run_dir)


def final_notebooks(cfg, run_dir, selected, deadline):
    job = Path(run_dir) / "resolved-config.json"
    atomic_json(job, cfg)
    def one(task):
        path, _ = selected[task["name"]]
        output = Path(run_dir) / "deliverables" / f"{task['name']}.solution.ipynb"
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent.parent), cfg["project_root"], env.get("PYTHONPATH", "")])
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[name] = "1"
        run_command([sys.executable, "-m", "race90", "_notebook", str(job), task["name"],
                     str(path), str(output), str(deadline)], deadline,
                    log=Path(run_dir) / task["name"] / "notebook.log", env=env)
        return str(output)
    outcomes = {}
    with ThreadPoolExecutor(max_workers=min(3, len(cfg["tasks"]))) as pool:
        future_map = {pool.submit(one, task): task["name"] for task in cfg["tasks"] if task["name"] in selected}
        for future, name in future_map.items():
            try:
                outcomes[name] = {"status": "done", "path": future.result()}
            except Exception as e:
                outcomes[name] = {"status": "failed", "error": str(e)}
    atomic_json(Path(run_dir) / "notebooks.json", outcomes)
    return outcomes


def run(cfg, resume=None, deadline_epoch=None):
    started = time.time()
    digest_input = {"tasks": cfg["tasks"], "seed": cfg["seed"], "data_root": cfg["data_root"]}
    fingerprints = {}
    from .config import resolve_data
    for task in cfg["tasks"]:
        for key in ("train_csv", "test_csv", "sample_submission", "data_yaml"):
            if task.get(key):
                path = resolve_data(cfg, task[key])
                stat = path.stat()
                fingerprints[str(path)] = [stat.st_size, stat.st_mtime_ns]
    digest_input["input_metadata"] = fingerprints
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
        end = min(float(deadline_epoch), maximum_end) if deadline_epoch else maximum_end
        run_dir.mkdir(parents=True)
        atomic_json(run_dir / "run.json", {"started": started, "deadline": end,
                                          "task_names": [t["name"] for t in cfg["tasks"]], "config_digest": config_digest})
    compute_end = end - cfg["finalize_minutes"] * 60
    cancel = threading.Event()
    errors = []
    remote = None
    workers = min(int(cfg["cpu_workers"]), len(cfg["tasks"]))
    total_threads = int(cfg.get("cpu_threads", max(1, (os.cpu_count() or 4) - 1)))
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
            baselines_ready = {t["name"]: threading.Event() for t in cfg["tasks"]}
            task_done = {t["name"]: threading.Event() for t in cfg["tasks"]}
            failed_dependencies = set()
            dependency_lock = threading.Lock()
            def complete_task(task):
                try:
                    selected, _ = select_best(cfg, task, run_dir / task["name"])
                    with dependency_lock:
                        cfg["dependency_outputs"][task["name"]] = str(selected)
                except Exception:
                    with dependency_lock:
                        failed_dependencies.add(task["name"])
                task_done[task["name"]].set()
            def task_cpu(task):
                for parent in task.get("depends_on", []):
                    while not task_done[parent].wait(.2):
                        if time.time() >= compute_end or cancel.is_set():
                            baselines_ready[task["name"]].set()
                            complete_task(task)
                            return
                    if parent in failed_dependencies:
                        event(run_dir, "dependency_failed", task=task["name"], parent=parent)
                        baselines_ready[task["name"]].set()
                        complete_task(task)
                        return
                skip = False
                if resume:
                    try:
                        _, prior = select_best(cfg, task, run_dir / task["name"])
                        skip = prior.get("score") is not None
                    except ValueError:
                        pass
                try:
                    if not skip:
                        baseline_end = min(compute_end, time.time() + float(task.get("baseline_seconds", 600)))
                        local_stage(cfg, task, run_dir, "baseline", baseline_end, threads, cancel=cancel)
                    event(run_dir, "baseline_done", task=task["name"])
                except Exception as e:
                    event(run_dir, "baseline_failed", task=task["name"], error=str(e))
                finally:
                    baselines_ready[task["name"]].set()
                if task.get("improve", True) and task not in gpu_tasks and time.time() < compute_end and not cancel.is_set():
                    try:
                        local_stage(cfg, task, run_dir, "upgrade",
                                    min(compute_end, time.time() + task.get("upgrade_seconds", 600)), threads, cancel=cancel)
                    finally:
                        complete_task(task)
                elif task not in gpu_tasks:
                    complete_task(task)
            with ThreadPoolExecutor(max_workers=workers) as cpus, ThreadPoolExecutor(max_workers=1) as improvements:
                futures = {}
                for task in cfg["tasks"]:
                    futures[cpus.submit(task_cpu, task)] = (task["name"], "CPU workflow")

                def gpu_work():
                    backend = None
                    if ready is not None:
                        try:
                            backend = ready.result()
                        except Exception as e:
                            event(run_dir, "gpu_unavailable", error=str(e))
                    ids = list(range(int(cfg["runpod"].get("gpu_count", 1)))) if backend else cfg.get("local_gpu_ids", [])
                    if not ids:
                        if gpu_tasks:
                            event(run_dir, "gpu_skipped", reason="No ready Runpod GPU or configured local_gpu_ids")
                        for task in gpu_tasks:
                            while not baselines_ready[task["name"]].wait(.2):
                                if time.time() >= compute_end or cancel.is_set():
                                    break
                            complete_task(task)
                        return
                    tasks = sorted(gpu_tasks, key=lambda t: -t.get("priority", 0))
                    task_lock = threading.Lock()
                    def on_gpu(gpu):
                        while time.time() < compute_end and not cancel.is_set():
                            with task_lock:
                                if not tasks:
                                    return
                                available = next((t for t in tasks if baselines_ready[t["name"]].is_set()), None)
                                remaining_tasks = len(tasks)
                                if available is not None:
                                    task = available
                                    tasks.remove(task)
                            if available is None:
                                cancel.wait(.2)
                                continue
                            if any(p in failed_dependencies for p in task.get("depends_on", [])):
                                complete_task(task)
                                continue
                            # Fair time slicing: leave a share for each queued GPU task.
                            waves = max(1, (remaining_tasks + len(ids) - 1) // len(ids))
                            allowance = max(1, (compute_end - time.time()) / waves)
                            stage_end = min(compute_end, time.time() + min(allowance, task.get("upgrade_seconds", 3600)))
                            try:
                                if backend:
                                    backend.run_task(task, gpu, stage_end, max(1, int(cfg.get("gpu_cpu_threads", 2))))
                                else:
                                    local_stage(cfg, task, run_dir, "upgrade", stage_end, threads, str(gpu), cancel)
                                event(run_dir, "gpu_task_done", task=task["name"], gpu=gpu)
                            except Exception as e:
                                event(run_dir, "gpu_task_failed", task=task["name"], gpu=gpu, error=str(e))
                            finally:
                                complete_task(task)
                    with ThreadPoolExecutor(max_workers=len(ids)) as gpus:
                        jobs = [gpus.submit(on_gpu, gpu) for gpu in ids]
                        for job in jobs:
                            job.result()
                if gpu_tasks:
                    futures[improvements.submit(gpu_work)] = ("GPU queue", "upgrade")
                monitor_futures(futures, cfg, run_dir, cancel)
    except KeyboardInterrupt:
        cancel.set()
        event(run_dir, "interrupted", reason="Finalizing completed candidates")
    finally:
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
            shutil.copy2(path / "submission.csv", deliver / f"{task['name']}_submission.csv")
            shutil.copy2(run_dir / task["name"] / "selected.json", deliver / f"{task['name']}_result.json")
        except Exception as e:
            errors.append(str(e))
    notebooks = final_notebooks(cfg, run_dir, selected, end - 5)
    # Only artifacts and source, never competition data or credentials, enter the submission pack.
    with zipfile.ZipFile(run_dir / "submission_pack.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in deliver.iterdir():
            archive.write(path, path.name)
    from .bundle import source_zip
    (run_dir / "source.zip").write_bytes(source_zip(cfg))
    report = {"run_dir": str(run_dir), "deadline": end, "finished_at": time.time(),
              "elapsed_seconds": time.time() - started,
              "tasks": {name: {**info, "candidate": path.name} for name, (path, info) in selected.items()},
              "notebooks": notebooks, "errors": errors,
              "complete": len(selected) == len(cfg["tasks"]) and all(n["status"] == "done" for n in notebooks.values())}
    atomic_json(run_dir / "final.json", report)
    event(run_dir, "finished", complete=report["complete"], run_dir=str(run_dir))
    return report
