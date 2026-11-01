"""One exam clock, with protected finalization and manual handoff windows."""
import time
from .io import atomic_json


def timeline(cfg, deadline):
    reserve = float(cfg["finalize_minutes"]) * 60
    options = cfg.get("time_management", {})
    upload = float(options.get("upload_minutes", reserve / 240)) * 60
    retrieval = float(options.get("retrieval_seconds", reserve * .10))
    packing = float(options.get("pack_seconds", reserve * .05))
    return {"exam_deadline": deadline, "compute_deadline": deadline - reserve,
            "retrieval_deadline": deadline - reserve + retrieval,
            "notebook_deadline": deadline - upload - packing,
            "deliverable_deadline": deadline - upload, "upload_seconds": upload,
            "retrieval_seconds": retrieval, "pack_seconds": packing}


def time_status(run_dir, plan, phase, **fields):
    now = time.time()
    state = {"phase": phase, "updated_at": now, **plan,
             "exam_seconds_left": max(0, plan["exam_deadline"] - now),
             "compute_seconds_left": max(0, plan["compute_deadline"] - now),
             "handoff_seconds_left": max(0, plan["deliverable_deadline"] - now), **fields}
    atomic_json(run_dir / "time-status.json", state)
    return state
