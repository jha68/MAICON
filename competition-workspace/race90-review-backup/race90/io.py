from pathlib import Path
import hashlib
import json
import os
import uuid
import numpy as np
import pandas as pd
from .config import resolve_data


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_csv(cfg, path, task):
    return pd.read_csv(resolve_data(cfg, path), encoding=task.get("encoding", "utf-8"),
                       **task.get("read_csv_kwargs", {}))


def validate_submission(frame, template, task):
    id_col = task["id_column"]
    targets = task["prediction_columns"]
    if list(frame.columns) != list(template.columns):
        raise ValueError("Submission column names/order differ from sample_submission")
    if len(frame) != len(template):
        raise ValueError("Submission row count differs from sample_submission")
    if id_col not in template or template[id_col].isna().any() or template[id_col].duplicated().any():
        raise ValueError("Sample submission IDs must be non-null and unique")
    if not frame[id_col].equals(template[id_col]):
        raise ValueError("Submission IDs/order differ from sample_submission")
    for col in targets:
        if col not in frame:
            raise ValueError(f"Missing prediction column {col}")
        if frame[col].isna().any():
            raise ValueError(f"Null predictions in {col}")
        if not task.get("allow_empty_predictions", False) and frame[col].astype(str).eq("").any():
            raise ValueError(f"Empty predictions in {col}")
        if pd.api.types.is_numeric_dtype(frame[col]) and not np.isfinite(frame[col]).all():
            raise ValueError(f"Non-finite predictions in {col}")
        if task.get("prediction_type") == "probability":
            p = pd.to_numeric(frame[col], errors="raise")
            if not p.between(0, 1).all():
                raise ValueError("Probabilities must lie in [0, 1]")
        if task.get("allowed_labels") is not None:
            if not frame[col].isin(task["allowed_labels"]).all():
                raise ValueError("Predictions contain labels outside allowed_labels")
    if task.get("prediction_type") == "probability" and len(targets) > 1:
        if not np.allclose(frame[targets].sum(axis=1), 1, atol=1e-5):
            raise ValueError("Multiclass probabilities must sum to one")
    untouched = [c for c in template.columns if c not in targets]
    if not frame[untouched].equals(template[untouched]):
        raise ValueError("Non-prediction columns changed")
    return frame


def make_submission(cfg, task, ids, predictions):
    template = read_csv(cfg, task["sample_submission"], task)
    id_col = task["id_column"]
    pred = np.asarray(predictions)
    if pred.ndim == 1:
        pred = pred[:, None]
    if pred.shape != (len(ids), len(task["prediction_columns"])):
        raise ValueError(f"Prediction shape {pred.shape} doesn't match configured output")
    series = pd.Series(ids).reset_index(drop=True)
    if series.isna().any() or series.duplicated().any():
        raise ValueError("Test IDs must be non-null and unique after feature extraction")
    if set(series.tolist()) != set(template[id_col].tolist()):
        raise ValueError("Test IDs and sample submission IDs differ")
    indexed = pd.DataFrame(pred, columns=task["prediction_columns"], index=series)
    ordered = indexed.loc[template[id_col]].reset_index(drop=True)
    result = template.copy()
    for c in task["prediction_columns"]:
        result[c] = ordered[c].to_numpy()
    return validate_submission(result, template, task)


def save_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()
