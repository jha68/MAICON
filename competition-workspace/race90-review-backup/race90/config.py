from pathlib import Path
import re
import yaml

KINDS = {"tabular", "text", "timeseries", "image", "yolo", "custom"}


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping")
    cfg.setdefault("total_minutes", 90)
    cfg.setdefault("finalize_minutes", 8)
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
    if not 0 < cfg["finalize_minutes"] < cfg["total_minutes"]:
        raise ValueError("0 < finalize_minutes < total_minutes required")
    if int(cfg["cpu_workers"]) < 1:
        raise ValueError("cpu_workers must be positive")
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
        if t["kind"] == "tabular" and t["resource"] == "gpu":
            raise ValueError(f"{name}: built-in tabular models use CPU; use custom for a GPU tabular model")
        t.setdefault("depends_on", [])
        if not isinstance(t["depends_on"], list):
            raise ValueError(f"{name}: depends_on must be a list")
        t.setdefault("split", {"strategy": "stratified", "fraction": 0.2})
        if t["kind"] == "yolo" and t["metric"] not in {"map50", "map50_95"}:
            raise ValueError(f"{name}: YOLO metric must be map50 or map50_95")
        if t["kind"] == "timeseries" and t["split"]["strategy"] == "stratified":
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
