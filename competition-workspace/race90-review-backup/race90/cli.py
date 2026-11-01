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
            p.add_argument("--deadline-epoch", type=float, help="Actual exam end time in UTC epoch seconds")
    p = commands.add_parser("status")
    p.add_argument("run_dir")
    p = commands.add_parser("_worker")
    p.add_argument("payload")
    p = commands.add_parser("_notebook")
    for name in ("config", "task", "candidate", "output", "deadline"):
        p.add_argument(name)
    p = commands.add_parser("_lease")
    for name in ("cli", "pod_id", "deadline", "cancelled"):
        p.add_argument(name)
    args = parser.parse_args(argv)
    if args.command == "_worker":
        from .worker import run_worker
        return run_worker(json.loads(Path(args.payload).read_text()))
    if args.command == "_notebook":
        from .notebook import export_notebook
        cfg = json.loads(Path(args.config).read_text())
        task = next(t for t in cfg["tasks"] if t["name"] == args.task)
        export_notebook(cfg, task, args.candidate, args.output, float(args.deadline))
        return 0
    if args.command == "_lease":
        from .remote import lease_watchdog
        return lease_watchdog(args.cli, args.pod_id, args.deadline, args.cancelled)
    if args.command == "status":
        path = Path(args.run_dir)
        for name in ("status.json", "runpod.json", "final.json"):
            if (path / name).exists():
                print((path / name).read_text())
        return 0
    cfg = load_config(args.config)
    if args.command == "plan":
        # No auth, rental, or training happens in this command.
        print(json.dumps({"minutes": cfg["total_minutes"], "finalize_minutes": cfg["finalize_minutes"],
                          "data_root": cfg["data_root"], "cpu_workers": cfg["cpu_workers"],
                          "runpod_enabled": cfg["runpod"].get("enabled", False),
                          "gpu_count": cfg["runpod"].get("gpu_count", 0),
                          "gpu_priorities": cfg["runpod"].get("gpu_priorities", []),
                          "max_gpu_hourly_usd": cfg["runpod"].get("max_gpu_hourly_usd"),
                          "tasks": [{"name": t["name"], "kind": t["kind"], "resource": t["resource"],
                                     "depends_on": t["depends_on"], "split": t["split"]} for t in cfg["tasks"]]},
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
    deadline = args.deadline_epoch or (invocation + cfg["total_minutes"] * 60)
    report = run(cfg, args.resume, None if args.resume and args.deadline_epoch is None else deadline)
    print(json.dumps({"complete": report["complete"], "run_dir": report["run_dir"]}, ensure_ascii=False))
    return 0 if report["complete"] else 1
