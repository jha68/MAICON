from dataclasses import dataclass
from pathlib import Path
import importlib
import json
import shutil
import time
import traceback
import uuid
import joblib
import numpy as np
from .io import atomic_json, save_csv, validate_submission, read_csv, sha256


@dataclass
class Context:
    cfg: dict
    task: dict
    output: Path
    stage: str
    deadline: float
    threads: int = 1
    device: str = "cpu"

    def seconds_left(self):
        return max(0.0, self.deadline - time.time())

    @property
    def dependencies(self):
        """Immutable selected candidate directories of declared parent tasks."""
        return {name: Path(self.cfg["dependency_outputs"][name]) for name in self.task.get("depends_on", [])}

    def publish(self, frame, score, details, model=None, artifacts=None, stage=None):
        template = read_csv(self.cfg, self.task["sample_submission"], self.task)
        validate_submission(frame, template, self.task)
        if score is not None and not np.isfinite(score):
            raise ValueError("Non-finite validation score")
        name = f"{stage or self.stage}-{uuid.uuid4().hex[:10]}"
        candidate = self.output / "candidates" / name
        candidate.mkdir(parents=True)
        if model is not None:
            joblib.dump(model, candidate / "model.joblib", compress=3)
        for filename, src in (artifacts or {}).items():
            if Path(filename).name != filename:
                raise ValueError("Artifact filenames must be simple")
            shutil.copy2(src, candidate / filename)
        save_csv(candidate / "submission.csv", frame)
        manifest = {"task": self.task["name"], "stage": stage or self.stage, "score": score,
                    "metric": self.task["metric"], "details": details, "created_at": time.time(),
                    "sha256": sha256(candidate / "submission.csv"), "rows": len(frame),
                    "artifacts": [p.name for p in candidate.iterdir() if p.name != "submission.csv"]}
        # Manifest is the commit marker: interrupted candidates cannot be selected.
        atomic_json(candidate / "manifest.json", manifest)
        print(json.dumps({"event": "candidate", "task": self.task["name"], "stage": manifest["stage"],
                          "score": score, "candidate": name}), flush=True)
        return candidate


def callable_from(spec):
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)


def run_worker(payload):
    ctx = Context(payload["cfg"], payload["task"], Path(payload["output"]), payload["stage"],
                  payload["deadline"], payload.get("threads", 1), payload.get("device", "cpu"))
    ctx.output.mkdir(parents=True, exist_ok=True)
    try:
        if ctx.task["kind"] == "custom":
            callable_from(ctx.task["adapter"])(ctx)
        else:
            from .models import publish_fallback, train_cpu
            from .data import audit
            if ctx.stage == "baseline":
                publish_fallback(ctx)
                audit(ctx.cfg, ctx.task, ctx.output)
            if ctx.task["kind"] == "yolo" and ctx.stage == "baseline":
                pass
            elif ctx.task["kind"] in {"image", "yolo"} and ctx.stage == "upgrade":
                from .vision import train_vision
                train_vision(ctx)
            elif ctx.stage == "upgrade" and ctx.device != "cpu" and ctx.task["kind"] in {"text", "timeseries"}:
                from .neural import train_text, train_sensor
                (train_text if ctx.task["kind"] == "text" else train_sensor)(ctx)
            else:
                train_cpu(ctx)
        atomic_json(ctx.output / f"{ctx.stage}-status.json", {"status": "done", "time": time.time()})
        return 0
    except Exception as e:
        atomic_json(ctx.output / f"{ctx.stage}-status.json", {"status": "failed", "error": str(e), "time": time.time()})
        traceback.print_exc()
        return 1
