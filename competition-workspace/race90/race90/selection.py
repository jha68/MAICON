from pathlib import Path
import json
import shutil
import numpy as np
from .io import read_csv, read_saved_submission, validate_submission, atomic_json, sha256, save_csv
MINIMIZE = {"rmse", "mae", "log_loss"}


def select_best(cfg, task, task_dir):
    task_dir = Path(task_dir)
    template = read_csv(cfg, task["sample_submission"], task)
    candidates, rejected = [], []
    for marker in (task_dir / "candidates").glob("*/manifest.json"):
        try:
            info = json.loads(marker.read_text())
            candidate = marker.parent
            if info["task"] != task["name"] or info["metric"] != task["metric"]:
                raise ValueError("Candidate task/metric mismatch")
            for parent in task.get("depends_on", []):
                parent_dir = Path(cfg["dependency_outputs"][parent])
                parent_info = json.loads((parent_dir / "manifest.json").read_text())
                parent_id = parent_info.get("candidate_id", parent_dir.name)
                if info.get("dependencies", {}).get(parent) != parent_id:
                    raise ValueError("Candidate uses a different parent checkpoint; rerun this dependent task")
            score = info.get("score")
            if score is not None and not np.isfinite(score):
                raise ValueError("Invalid score")
            if sha256(candidate / "submission.csv") != info["sha256"]:
                raise ValueError("Candidate CSV hash mismatch")
            for artifact in info.get("artifacts", []):
                if Path(artifact).name != artifact or not (candidate / artifact).is_file():
                    raise ValueError("Missing/unsafe candidate artifact")
            frame = read_saved_submission(candidate / "submission.csv", template, task, info)
            validate_submission(frame, template, task)
            candidates.append((candidate, info, frame))
        except Exception as e:
            rejected.append({"candidate": marker.parent.name, "reason": str(e)})
    if not candidates:
        raise ValueError(f"{task['name']}: no valid completed candidate")
    scored = [c for c in candidates if c[1].get("score") is not None]
    split_hashes = {c[1]["details"].get("split_hash") for c in scored} - {None}
    if len(split_hashes) > 1:
        raise ValueError("Different validation splits cannot be compared")
    if scored:
        direction = -1 if task["metric"] in MINIMIZE else 1
        best = max(scored, key=lambda c: (direction * c[1]["score"],
                                         bool(c[1]["details"].get("refit_all_rows")), c[1]["created_at"]))
    else:
        best = max(candidates, key=lambda c: c[1]["created_at"])
    selected, info, frame = best
    save_csv(task_dir / "submission.csv", frame)
    atomic_json(task_dir / "selected.json", {**info, "candidate": selected.name,
                                             "fallback_only": info.get("score") is None,
                                             "rejected_candidates": rejected})
    return selected, info
