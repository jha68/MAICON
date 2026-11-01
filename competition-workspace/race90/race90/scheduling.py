"""Dispatch ready stages without parking workers on dependency waits."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import time
from .selection import select_best


def run_stages(cfg, run_dir, compute_end, workers, threads, cancel, ready,
               event, local_stage, snapshot):
    tasks = {t["name"]: t for t in cfg["tasks"]}
    state = {name: {"baseline": "pending", "upgrade": "pending" if t.get("improve", True) else "done"}
             for name, t in tasks.items()}
    done, failed, running = set(), set(), {}
    backend, gpu_ids = None, None

    def finish(name):
        try:
            path, _ = select_best(cfg, tasks[name], run_dir / name)
            cfg["dependency_outputs"][name] = str(path)
        except ValueError:
            failed.add(name)
        done.add(name)

    def duration(name, stage):
        return float(tasks[name].get(stage + "_seconds", 600))

    def work_left(name):
        return sum(duration(name, stage) for stage in ("baseline", "upgrade") if state[name][stage] != "done")

    def chain_after(name):
        children = [n for n, t in tasks.items() if name in t.get("depends_on", []) and n not in done]
        return max((work_left(n) + chain_after(n) for n in children), default=0)

    def stage_deadline(name, stage, resource):
        requested = duration(name, stage)
        # Leave proportional time for downstream tasks and queued work. A long
        # parent upgrade must not consume the entire shared compute deadline.
        chain = requested + chain_after(name)
        if stage == "baseline" and state[name]["upgrade"] != "done":
            chain += duration(name, "upgrade")
        queued = 0
        for n, t in tasks.items():
            for s in ("baseline", "upgrade"):
                r = "gpu" if s == "upgrade" and t.get("resource", "cpu") == "gpu" else "cpu"
                if r == resource and state[n][s] != "done":
                    queued += duration(n, s)
        slots = workers if resource == "cpu" else max(1, len(gpu_ids or []))
        allowance = (compute_end - time.time()) * requested / max(requested, chain, queued / slots)
        return min(compute_end, time.time() + min(requested, max(.1, allowance)))

    cpu_pool = ThreadPoolExecutor(max_workers=workers)
    gpu_pool = ThreadPoolExecutor(max_workers=max(1, int(cfg["runpod"].get("gpu_count", 1)),
                                                len(cfg.get("local_gpu_ids", []))))
    last_snapshot = 0
    warned = set()
    plan = cfg.get("_timeline")
    try:
        while len(done) < len(tasks) and time.time() < compute_end and not cancel.is_set():
            if (run_dir / "finalize-request.json").exists():
                event(run_dir, "finalization_requested", reason="Stop training and build current completed results")
                break
            if gpu_ids is None and (ready is None or ready.done()):
                if ready is not None:
                    try:
                        backend = ready.result()
                    except Exception as exc:
                        event(run_dir, "gpu_unavailable", error=str(exc))
                gpu_ids = (list(range(int(cfg["runpod"].get("gpu_count", 1)))) if backend
                           else list(cfg.get("local_gpu_ids", [])))
                if not gpu_ids:
                    for name, task in tasks.items():
                        if task.get("resource", "cpu") == "gpu":
                            state[name]["upgrade"] = "done"
                    if any(t.get("resource") == "gpu" for t in tasks.values()):
                        event(run_dir, "gpu_skipped", reason="No ready GPU; completed CPU baseline is retained")

            for future in list(running):
                if not future.done():
                    continue
                name, stage, resource, slot = running.pop(future)
                try:
                    future.result()
                    event(run_dir, "stage_done", task=name, stage=stage, resource=resource)
                except Exception as exc:
                    event(run_dir, "stage_failed", task=name, stage=stage, error=str(exc))
                state[name][stage] = "done"

            # Propagate failures before dispatch. No child occupies a worker
            # while waiting for a parent; independent baselines get first use.
            for name, task in tasks.items():
                if name in done:
                    continue
                parents = set(task.get("depends_on", []))
                if parents & failed:
                    failed.add(name)
                    done.add(name)
                    state[name] = {"baseline": "done", "upgrade": "done"}
                    event(run_dir, "dependency_failed", task=name, parents=sorted(parents & failed))
                elif all(s == "done" for s in state[name].values()):
                    finish(name)

            available = []
            for name, task in tasks.items():
                if name in done or not set(task.get("depends_on", [])) <= done:
                    continue
                for stage in ("baseline", "upgrade"):
                    if state[name][stage] != "pending":
                        continue
                    # GPU training can start as soon as the Pod is ready; a slow
                    # local baseline must not leave a paid GPU idle.
                    if stage == "upgrade" and task.get("resource", "cpu") != "gpu" and state[name]["baseline"] != "done":
                        continue
                    available.append((0 if stage == "baseline" else 1, -task.get("priority", 0), name, stage))
            for _, _, name, stage in sorted(available):
                task = tasks[name]
                resource = "gpu" if stage == "upgrade" and task.get("resource", "cpu") == "gpu" else "cpu"
                slot = None
                if resource == "cpu":
                    if sum(r[2] == "cpu" for r in running.values()) >= workers:
                        continue
                else:
                    used = {r[3] for r in running.values() if r[2] == "gpu"}
                    free = [i for i in (gpu_ids or []) if i not in used]
                    if not free:
                        continue
                    slot = free[0]
                # Reuse completed work on resume without losing the original clock.
                if stage == "baseline" and state[name]["upgrade"] != "running":
                    try:
                        _, info = select_best(cfg, task, run_dir / name)
                        if info.get("score") is not None:
                            state[name][stage] = "done"
                            continue
                    except ValueError:
                        pass
                deadline = stage_deadline(name, stage, resource)
                minimum = float(task.get("minimum_upgrade_seconds", cfg.get("time_management", {}).get(
                    "minimum_upgrade_seconds", min(60, cfg["total_minutes"] * 60 / 90))))
                if stage == "upgrade" and deadline - time.time() < minimum:
                    state[name][stage] = "done"
                    event(run_dir, "upgrade_skipped", task=name, reason="Insufficient time for a useful completed checkpoint")
                    continue
                payload_cfg = deepcopy(cfg)
                if resource == "gpu" and backend:
                    job = gpu_pool.submit(backend.run_task, task, slot, deadline,
                                          max(1, int(cfg.get("gpu_cpu_threads", 2))))
                else:
                    pool = cpu_pool if resource == "cpu" else gpu_pool
                    job = pool.submit(local_stage, payload_cfg, task, run_dir, stage, deadline,
                                      threads, device="cpu" if resource == "cpu" else str(slot), cancel=cancel)
                running[job] = (name, stage, resource, slot)
                state[name][stage] = "running"
                event(run_dir, "stage_started", task=name, stage=stage, resource=resource, deadline=deadline)

            if time.monotonic() - last_snapshot >= 5:
                current = snapshot(cfg, run_dir)
                if plan:
                    from .timing import time_status
                    time_status(run_dir, plan, "training", tasks=current)
                    for minutes in (30, 20, 10, 5):
                        if minutes < cfg["total_minutes"] and minutes not in warned and plan["exam_deadline"] - time.time() <= minutes * 60:
                            warned.add(minutes)
                            event(run_dir, "time_warning", remaining_minutes=minutes,
                                  action="최종화 시간 확보; 새 장시간 실험을 시작하지 마세요.")
                last_snapshot = time.monotonic()
            if running:
                wait(running, timeout=min(.2, max(.01, compute_end - time.time())), return_when=FIRST_COMPLETED)
            elif len(done) < len(tasks):
                cancel.wait(.05)
    finally:
        if plan:
            from .timing import time_status
            time_status(run_dir, plan, "retrieval")
        cancel.set()  # Set before waiting for subprocess-owning executor threads.
        cpu_pool.shutdown(wait=True, cancel_futures=True)
        gpu_pool.shutdown(wait=True, cancel_futures=True)
        snapshot(cfg, run_dir)
