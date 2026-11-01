import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from .config import resolve_data
from .io import read_csv, atomic_json


def sensor_features(df, task):
    ident, sensors = task["id_column"], task["sensor_columns"]
    if task.get("time_column"):
        df = df.sort_values([ident, task["time_column"]], kind="stable")
    grouped = df.groupby(ident, sort=False)
    values = df[sensors].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    # Per-recording features only: never cross sample boundaries or use labels.
    result = values.groupby(df[ident], sort=False).agg(["mean", "std", "min", "max", "median"])
    result.columns = [f"{sensor}_{stat}" for sensor, stat in result.columns]
    for sensor in sensors:
        g = values[sensor].groupby(df[ident], sort=False)
        result[f"{sensor}_rms"] = g.apply(lambda x: float(np.sqrt(np.nanmean(x.to_numpy() ** 2))))
        result[f"{sensor}_delta"] = g.last() - g.first()
        result[f"{sensor}_slope"] = g.apply(_slope)
    result["length"] = grouped.size()
    return result.reset_index()


def _slope(x):
    arr = x.to_numpy(dtype=float)
    valid = np.isfinite(arr)
    if valid.sum() < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(arr))[valid], arr[valid], 1)[0])


def image_path(cfg, task, value):
    return resolve_data(cfg, str(Path(task.get("image_root", ".")) / str(value)))


def image_features(cfg, task, df):
    rows = []
    for filename in df[task["image_column"]]:
        with Image.open(image_path(cfg, task, filename)) as im:
            arr = np.asarray(im.convert("RGB").resize((16, 16)), dtype=np.float32) / 255
        hist = np.concatenate([np.histogram(arr[..., c], bins=16, range=(0, 1), density=False)[0] / 256
                               for c in range(3)])
        rows.append(np.concatenate([arr.mean(axis=(0, 1)), arr.std(axis=(0, 1)), hist,
                                    arr[::4, ::4].ravel()]))
    return pd.DataFrame(rows)


def load_dataset(cfg, task):
    train = read_csv(cfg, task["train_csv"], task)
    test = read_csv(cfg, task["test_csv"], task)
    for feature in task.get("dependency_features", []):
        parent = Path(cfg["dependency_outputs"][feature["task"]])
        key = feature.get("on", task["id_column"])
        train_extra = pd.read_csv(parent / feature["train_file"])
        test_extra = pd.read_csv(parent / feature["test_file"])
        if train_extra[key].duplicated().any() or test_extra[key].duplicated().any():
            raise ValueError("Dependency features require exactly one row per join key")
        if set(train[key]) != set(train_extra[key]) or set(test[key]) != set(test_extra[key]):
            raise ValueError("Dependency feature IDs don't cover training/test data")
        columns = set(train_extra) - {key}
        if columns != set(test_extra) - {key} or columns & (set(train) | set(test)):
            raise ValueError("Dependency train/test feature columns must match and not overwrite existing columns")
        train = train.merge(train_extra, on=key, how="left", validate="many_to_one", sort=False)
        test = test.merge(test_extra, on=key, how="left", validate="many_to_one", sort=False)
    target, ident = task["target"], task["id_column"]
    if target not in train or train[target].isna().any():
        raise ValueError("Target missing or contains nulls")
    if task["kind"] == "timeseries":
        if (train.groupby(ident)[target].nunique() > 1).any():
            raise ValueError("Each sensor recording must have exactly one target")
        metadata = train.groupby(ident, sort=False).first().reset_index()
        tr, te = sensor_features(train, task), sensor_features(test, task)
        metadata = metadata.set_index(ident).loc[tr[ident]].reset_index()
        y = metadata[target]
        X, Xt = tr.drop(columns=ident), te.drop(columns=ident)
        ids = te[ident]
        train = metadata
    elif task["kind"] == "text":
        columns = task["text_columns"]
        X = train[columns].fillna("").astype(str).agg(" ".join, axis=1)
        Xt = test[columns].fillna("").astype(str).agg(" ".join, axis=1)
        y, ids = train[target], test[ident]
    elif task["kind"] == "image":
        X, Xt = image_features(cfg, task, train), image_features(cfg, task, test)
        y, ids = train[target], test[ident]
    else:
        ignored = set(task.get("drop_columns", [])) | {target, ident}
        ignored |= {v for v in (task["split"].get("group_column"), task["split"].get("time_column")) if v}
        columns = task.get("feature_columns") or [c for c in train.columns if c not in ignored]
        if target in columns or ident in columns:
            raise ValueError("Target/ID columns cannot be features")
        if not columns:
            raise ValueError("No feature columns")
        X, Xt, y, ids = train[columns], test[columns], train[target], test[ident]
    if train[ident].duplicated().any() or ids.duplicated().any():
        raise ValueError("Duplicate sample ID (use timeseries kind for long sensor tables)")
    return X, Xt, y.reset_index(drop=True), ids.reset_index(drop=True), train


def audit(cfg, task, output):
    report = {"task": task["name"], "kind": task["kind"], "warnings": []}
    tables = {}
    for key in ("train_csv", "test_csv", "sample_submission"):
        if task.get(key):
            df = read_csv(cfg, task[key], task)
            tables[key] = df
            report[key] = {"rows": len(df), "columns": list(df.columns),
                           "nulls": {k: int(v) for k, v in df.isna().sum().items()},
                           "dtypes": {k: str(v) for k, v in df.dtypes.items()}}
    ident = task["id_column"]
    sample, test = tables["sample_submission"], tables["test_csv"]
    if sample[ident].duplicated().any() or sample[ident].isna().any():
        raise ValueError("Invalid sample submission IDs")
    if set(sample[ident]) != set(test[ident]):
        raise ValueError("Sample/test ID mismatch")
    train = tables.get("train_csv")
    if train is not None:
        target = task.get("target")
        if target and target in train:
            if train[target].isna().any():
                raise ValueError("Null targets")
            report["target_counts"] = {str(k): int(v) for k, v in train[target].value_counts().head(100).items()}
        if ident in train and set(train[ident]) & set(test[ident]):
            report["warnings"].append("Train/test IDs overlap; verify problem semantics")
        if task["kind"] != "timeseries" and train[ident].duplicated().any():
            raise ValueError("Duplicate train IDs")
        if task["kind"] == "timeseries" and (train.groupby(ident)[target].nunique() > 1).any():
            raise ValueError("Conflicting targets within recording")
        if task["split"]["strategy"] == "time":
            column = task["split"].get("time_column")
            if not column or column not in train or train[column].isna().any():
                raise ValueError("Time validation needs a complete time_column")
        if task["split"]["strategy"] == "group":
            column = task["split"].get("group_column")
            if not column or column not in train or train[column].isna().any():
                raise ValueError("Group validation needs a complete group_column")
    if task["kind"] in {"image", "yolo"}:
        image_tables = [test] + ([train] if train is not None else [])
        checked = 0
        for df in image_tables:
            for value in df[task["image_column"]]:
                with Image.open(image_path(cfg, task, value)) as im:
                    im.verify()
                checked += 1
        report["verified_images"] = checked
    if task["kind"] == "yolo":
        from .vision import normalized_yolo_yaml
        _, counts = normalized_yolo_yaml(cfg, task, output)
        report["yolo_split_images"] = counts
    atomic_json(Path(output) / "audit.json", report)
    return report
