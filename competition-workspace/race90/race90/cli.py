import argparse
import json
from pathlib import Path
import sys
import time
from .config import load_config


def main(argv=None, start_time=None):
    invocation = time.time() if start_time is None else start_time
    parser = argparse.ArgumentParser(prog="race90")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "check", "plan"):
        p = commands.add_parser(name)
        p.add_argument("config")
        if name == "run":
            p.add_argument("--resume", help="Existing run directory; retains its original deadline")
            clock_args = p.add_mutually_exclusive_group()
            clock_args.add_argument("--deadline-epoch", type=float, help="Actual exam end time in UTC epoch seconds")
            clock_args.add_argument("--clock", help="Clock JSON created at exam start; includes reading/download time")
    p = commands.add_parser("start-clock", help="Start the exam clock before reading/downloading the questions")
    p.add_argument("--minutes", type=float, default=90)
    p.add_argument("--output", default="exam-clock.json")
    p = commands.add_parser("status")
    p.add_argument("run_dir")
    p = commands.add_parser("finalize", help="Ask an active local run to stop training and finalize now")
    p.add_argument("run_dir")
    p = commands.add_parser("_package")
    p.add_argument("run_dir")
    p.add_argument("deadline", type=float)
    p = commands.add_parser("_worker")
    p.add_argument("payload")
    p = commands.add_parser("_notebook")
    for name in ("config", "task", "candidate", "output", "deadline"):
        p.add_argument(name)
    p = commands.add_parser("_lease")
    for name in ("cli", "pod_id", "deadline", "cancelled"):
        p.add_argument(name)
    args = parser.parse_args(argv)
    if args.command == "start-clock":
        import math
        if not math.isfinite(args.minutes) or args.minutes <= 0:
            raise ValueError("Clock minutes must be finite and positive")
        path = Path(args.output).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Never silently reset a clock when the command/cell is run twice.
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"started_at": invocation, "deadline": invocation + args.minutes * 60}, stream, indent=2)
        print(str(path))
        return 0
    if args.command == "_package":
        from .deliverables import package_deliverables
        package_deliverables(args.run_dir, args.deadline)
        return 0
    if args.command == "finalize":
        from .io import atomic_json
        directory = Path(args.run_dir).resolve()
        if not (directory / "run.json").is_file():
            raise ValueError("Not a race90 run directory")
        if (directory / "final.json").exists():
            print("이미 최종화된 실행입니다. final.json과 deliverables를 확인하세요.")
            return 0
        atomic_json(directory / "finalize-request.json", {"requested_at": time.time()})
        print("실행 중인 하네스에 최종화를 요청했습니다. time-status.json에서 진행 상황을 확인하세요.")
        return 0
    if args.command == "_worker":
        from .worker import run_worker
        return run_worker(json.loads(Path(args.payload).read_text()))
    if args.command == "_notebook":
        from .notebook import export_notebook
        cfg = json.loads(Path(args.config).read_text())
        task = next(t for t in cfg["tasks"] if t["name"] == args.task)
        export_notebook(cfg, task, args.candidate, args.output, float(args.deadline))
        from .deliverables import verify_export
        verify_export(cfg, task, args.candidate, args.output)
        return 0
    if args.command == "_lease":
        from .remote import lease_watchdog
        return lease_watchdog(args.cli, args.pod_id, args.deadline, args.cancelled)
    if args.command == "status":
        path = Path(args.run_dir)
        for name in ("time-status.json", "status.json", "runpod.json", "notebooks.json", "final.json"):
            if (path / name).exists():
                state = json.loads((path / name).read_text())
                if name == "time-status.json":
                    state["exam_seconds_left"] = max(0, state["exam_deadline"] - time.time())
                    state["compute_seconds_left"] = max(0, state["compute_deadline"] - time.time())
                    state["handoff_seconds_left"] = max(0, state["deliverable_deadline"] - time.time())
                print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0
    cfg = load_config(args.config)
    if args.command == "plan":
        from .resources import cpu_capacity
        from .timing import timeline
        plan = timeline(cfg, cfg["total_minutes"] * 60)
        capacity = cpu_capacity()
        workers = min(cfg["cpu_workers"], len(cfg["tasks"]), int(cfg.get("cpu_threads") or capacity), capacity)
        # No auth, rental, or training happens in this command.
        print(json.dumps({"minutes": cfg["total_minutes"], "finalize_minutes": cfg["finalize_minutes"],
                          "stop_training_at_minute": plan["compute_deadline"] / 60,
                          "finish_notebooks_at_minute": plan["notebook_deadline"] / 60,
                          "start_manual_upload_at_minute": plan["deliverable_deadline"] / 60,
                          "data_root": cfg["data_root"], "cpu_workers": workers, "detected_cpu_capacity": capacity,
                          "runpod_enabled": cfg["runpod"].get("enabled", False),
                          "gpu_count": cfg["runpod"].get("gpu_count", 0),
                          "gpu_priorities": cfg["runpod"].get("gpu_priorities", []),
                          "max_gpu_hourly_usd": cfg["runpod"].get("max_gpu_hourly_usd"),
                          "tasks": [{"name": t["name"], "kind": t["kind"], "resource": t["resource"],
                                     "depends_on": t["depends_on"], "split": t["split"],
                                     "baseline_seconds": t.get("baseline_seconds", 600),
                                     "upgrade_seconds": t.get("upgrade_seconds", 600)} for t in cfg["tasks"]]},
                         indent=2, ensure_ascii=False))
        return 0
    if args.command == "check":
        from .data import audit
        from concurrent.futures import ThreadPoolExecutor
        output = Path(cfg["output_root"]) / "preflight"
        with ThreadPoolExecutor(max_workers=cfg["cpu_workers"]) as pool:
            jobs = [pool.submit(audit, cfg, t, output / t["name"]) for t in cfg["tasks"]]
            for job in jobs:
                report = job.result()
                print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    from .runner import run
    explicit_deadline = args.deadline_epoch
    if args.clock:
        explicit_deadline = float(json.loads(Path(args.clock).read_text())["deadline"])
    deadline = explicit_deadline if explicit_deadline is not None else invocation + cfg["total_minutes"] * 60
    report = run(cfg, args.resume, None if args.resume and explicit_deadline is None else deadline)
    print(json.dumps({"complete": report["complete"], "submission_ready": report["submission_ready"],
                      "fallback_tasks": report["fallback_tasks"], "run_dir": report["run_dir"]}, ensure_ascii=False))
    return 0 if report["submission_ready"] else 1
