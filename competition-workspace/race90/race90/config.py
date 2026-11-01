from pathlib import Path
import math
import re
import yaml

KINDS = {"tabular", "text", "timeseries", "image", "yolo", "custom"}
CLASSIFICATION_METRICS = {"accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "f1", "roc_auc", "log_loss"}
REGRESSION_METRICS = {"rmse", "mae", "r2"}


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping")
    cfg.setdefault("total_minutes", 90)
    cfg.setdefault("finalize_minutes", 20)
    cfg.setdefault("cpu_workers", 3)
    cfg.setdefault("seed", 42)
    cfg.setdefault("data_root", "data")
    cfg.setdefault("output_root", "runs")
    cfg.setdefault("runpod", {"enabled": False})
    if any(k in cfg["runpod"] for k in ("api_key", "apiKey", "RUNPOD_API_KEY")):
        raise ValueError("Keep API keys in RUNPOD_API_KEY/environment, not the configuration file")
    cfg.setdefault("notebooks", {"execute": True, "embed_models": True})
    for key in ("data_root", "output_root"):
        p = Path(cfg[key]).expanduser()
        cfg[key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    cfg["project_root"] = str(path.parent)
    if not all(math.isfinite(float(cfg[k])) for k in ("total_minutes", "finalize_minutes")) or not 0 < cfg["finalize_minutes"] < cfg["total_minutes"]:
        raise ValueError("0 < finalize_minutes < total_minutes required")
    from .timing import timeline
    plan = timeline(cfg, 0)
    if any(not math.isfinite(plan[k]) or plan[k] < 0 for k in ("upload_seconds", "retrieval_seconds", "pack_seconds")):
        raise ValueError("Time management reserves must be finite and non-negative")
    if plan["retrieval_deadline"] >= plan["notebook_deadline"]:
        raise ValueError("Finalization reserve must leave time for Notebook execution after retrieval, packing and upload")
    minimum = float(cfg.get("time_management", {}).get("minimum_upgrade_seconds", 0))
    if not math.isfinite(minimum) or minimum < 0:
        raise ValueError("minimum_upgrade_seconds must be finite and non-negative")
    if int(cfg["cpu_workers"]) < 1:
        raise ValueError("cpu_workers must be positive")
    if cfg.get("cpu_threads") is not None and int(cfg["cpu_threads"]) < 1:
        raise ValueError("cpu_threads must be positive")
    if int(cfg.get("notebook_workers", 1)) < 1:
        raise ValueError("notebook_workers must be positive")
    if cfg["runpod"].get("enabled") and int(cfg["runpod"].get("gpu_count", 1)) < 1:
        raise ValueError("runpod.gpu_count must be positive")
    if cfg["notebooks"].get("mode", "restore") not in {"restore", "infer"}:
        raise ValueError("notebooks.mode must be restore or infer")
    if cfg["notebooks"].get("mode", "restore") == "infer" and not cfg["notebooks"].get("embed_models", True):
        raise ValueError("notebooks.mode=infer requires embed_models=true")
    tasks = cfg.get("tasks", [])
    if not tasks:
        raise ValueError("At least one task is required")
    seen = set()
    for t in tasks:
        name = t.get("name", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in seen:
            raise ValueError(f"Unique, simple task name required: {name!r}")
        seen.add(name)
        if t.get("kind") not in KINDS:
            raise ValueError(f"Unsupported task kind: {t.get('kind')}")
        for key in ("test_csv", "sample_submission", "id_column", "prediction_columns"):
            if not t.get(key):
                raise ValueError(f"{name}: {key} is required")
        if not isinstance(t["prediction_columns"], list) or t["id_column"] in t["prediction_columns"]:
            raise ValueError(f"{name}: prediction_columns must be a list excluding the ID column")
        if len(set(t["prediction_columns"])) != len(t["prediction_columns"]):
            raise ValueError(f"{name}: duplicate prediction_columns")
        if t["kind"] not in {"yolo", "custom"}:
            for key in ("train_csv", "target"):
                if not t.get(key):
                    raise ValueError(f"{name}: {key} is required")
        if t["kind"] == "custom" and not t.get("adapter"):
            raise ValueError(f"{name}: adapter module:function required")
        if t["kind"] == "text" and not t.get("text_columns"):
            raise ValueError(f"{name}: text_columns required")
        if t["kind"] == "timeseries" and not t.get("sensor_columns"):
            raise ValueError(f"{name}: sensor_columns required")
        if t["kind"] in {"image", "yolo"} and not t.get("image_column"):
            raise ValueError(f"{name}: image_column required")
        t.setdefault("objective", "classification")
        t.setdefault("metric", "map50" if t["kind"] == "yolo" else
                     ("accuracy" if t["objective"] == "classification" else "rmse"))
        if t["objective"] not in {"classification", "regression"}:
            raise ValueError(f"{name}: objective must be classification or regression")
        t.setdefault("prediction_type", "label")
        t.setdefault("resource", "gpu" if t["kind"] in {"image", "yolo"} else "cpu")
        if t["resource"] not in {"cpu", "gpu"}:
            raise ValueError(f"{name}: resource must be cpu or gpu")
        if t.get("cpu_estimator", "hist_gradient_boosting") not in {"hist_gradient_boosting", "extra_trees"}:
            raise ValueError(f"{name}: unknown cpu_estimator")
        if t.get("image_upgrade", "trees") not in {"trees", "embeddings"}:
            raise ValueError(f"{name}: image_upgrade must be trees or embeddings")
        t.setdefault("depends_on", [])
        if not isinstance(t["depends_on"], list):
            raise ValueError(f"{name}: depends_on must be a list")
        t.setdefault("split", {"strategy": "stratified", "fraction": 0.2})
        split = t["split"]
        if split.get("strategy") not in {"stratified", "random", "group", "time"}:
            raise ValueError(f"{name}: unknown split strategy")
        if not 0 < float(split.get("fraction", .2)) < 1:
            raise ValueError(f"{name}: split fraction must be in (0, 1)")
        for strategy, column in (("group", "group_column"), ("time", "time_column")):
            if split["strategy"] == strategy and not split.get(column):
                raise ValueError(f"{name}: {column} is required for {strategy} split")
        if int(split.get("gap", 0)) != float(split.get("gap", 0)) or int(split.get("gap", 0)) < 0:
            raise ValueError(f"{name}: split gap must be a non-negative integer")
        for key in ("baseline_seconds", "upgrade_seconds"):
            if key in t and (not math.isfinite(float(t[key])) or float(t[key]) <= 0):
                raise ValueError(f"{name}: {key} must be positive and finite")
        if t["kind"] not in {"yolo", "custom"}:
            metrics = CLASSIFICATION_METRICS if t["objective"] == "classification" else REGRESSION_METRICS
            if t["metric"] not in metrics:
                raise ValueError(f"{name}: unsupported metric for {t['objective']}: {t['metric']}")
        if t["prediction_type"] not in {"label", "probability"}:
            raise ValueError(f"{name}: prediction_type must be label or probability")
        if t["metric"] in {"roc_auc", "f1"} and "positive_class" not in t:
            raise ValueError(f"{name}: {t['metric']} requires positive_class (binary metric)")
        if t["prediction_type"] == "probability":
            if t["objective"] != "classification":
                raise ValueError(f"{name}: probability requires classification")
            columns = t["prediction_columns"]
            if len(columns) == 1 and "positive_class" not in t:
                raise ValueError(f"{name}: binary probability requires positive_class")
            if len(columns) > 1 and (len(t.get("class_order", [])) != len(columns) or
                                    len(set(t.get("class_order", []))) != len(columns)):
                raise ValueError(f"{name}: class_order must map each probability column exactly once")
        if t["kind"] not in {"custom", "yolo"} and t["prediction_type"] == "label" and len(t["prediction_columns"]) != 1:
            raise ValueError(f"{name}: built-in label/regression models require one prediction column")
        if t["kind"] == "yolo" and t["metric"] not in {"map50", "map50_95"}:
            raise ValueError(f"{name}: YOLO metric must be map50 or map50_95")
        if t["kind"] == "timeseries" and t["split"]["strategy"] not in {"group", "time"}:
            raise ValueError(f"{name}: choose group or time split explicitly for sensor data")
        if t["kind"] == "yolo" and not t.get("data_yaml"):
            raise ValueError(f"{name}: data_yaml with explicit train/val split required")
        # All data stays under one root, including remote transfer and Notebook replay.
        for key in ("train_csv", "test_csv", "sample_submission", "notebook", "image_root", "data_yaml"):
            if t.get(key):
                resolve_data(cfg, t[key])
    names = {t["name"] for t in tasks}
    for t in tasks:
        if set(t["depends_on"]) - names or t["name"] in t["depends_on"]:
            raise ValueError(f"{t['name']}: invalid dependency")
        if t.get("dependency_features") and t["kind"] != "tabular":
            raise ValueError("dependency_features is for tabular tasks; use ctx.dependencies in custom adapters for other types")
        for feature in t.get("dependency_features", []):
            if feature["task"] not in t["depends_on"]:
                raise ValueError("dependency_features must name a declared depends_on task")
            for key in ("train_file", "test_file"):
                if Path(feature[key]).name != feature[key]:
                    raise ValueError("Dependency feature files must have simple filenames")
    # Stable topological order ensures CPU queues never put children before parents.
    ordered, pending = [], list(tasks)
    while pending:
        available = next((t for t in pending if set(t["depends_on"]) <= {x["name"] for x in ordered}), None)
        if available is None:
            raise ValueError("Task dependency cycle detected")
        ordered.append(available)
        pending.remove(available)
    cfg["tasks"] = ordered
    cfg["dependency_outputs"] = {}
    return cfg


def resolve_data(cfg, value):
    root = Path(cfg["data_root"]).resolve()
    value = Path(value)
    if value.is_absolute():
        raise ValueError(f"Data paths must be relative to data_root: {value}")
    result = (root / value).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Data path escapes data_root: {value}")
    return result
