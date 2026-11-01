from pathlib import Path
import hashlib
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, mean_squared_error, mean_absolute_error, log_loss)
from .data import load_dataset
from .io import read_csv, make_submission

MINIMIZE = {"rmse", "mae", "log_loss"}


def split_indices(task, y, metadata, seed):
    split = task["split"]
    strategy = split["strategy"]
    fraction = float(split.get("fraction", 0.2))
    if not 0 < fraction < 1 or len(y) < 5:
        raise ValueError("Validation requires at least 5 records and 0 < fraction < 1")
    idx = np.arange(len(y))
    if strategy == "group":
        groups = metadata[split["group_column"]]
        if groups.nunique() < 2 or groups.isna().any():
            raise ValueError("Group split needs at least two complete groups")
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(idx, y, groups))
        if set(groups.iloc[a]) & set(groups.iloc[b]):
            raise AssertionError("Validation group leakage")
    elif strategy == "time":
        stamps = metadata[split["time_column"]]
        if stamps.isna().any():
            raise ValueError("Null validation timestamps")
        if not pd.api.types.is_numeric_dtype(stamps):
            stamps = pd.to_datetime(stamps, errors="raise", utc=True)
        order = np.argsort(stamps.to_numpy(), kind="stable")
        n_valid = max(1, int(np.ceil(len(idx) * fraction)))
        cutoff = stamps.iloc[order[-n_valid]]
        b = idx[(stamps >= cutoff).to_numpy()]
        a = idx[(stamps < cutoff).to_numpy()]
        gap = int(split.get("gap", 0))
        a = a[np.argsort(stamps.iloc[a].to_numpy(), kind="stable")]
        if gap:
            a = a[:-gap]
    elif strategy in {"random", "stratified"}:
        stratify = y if strategy == "stratified" and task["objective"] == "classification" else None
        a, b = train_test_split(idx, test_size=fraction, random_state=seed, stratify=stratify)
    else:
        raise ValueError(f"Unknown validation strategy: {strategy}")
    if len(a) < 2 or not len(b):
        raise ValueError("Empty/tiny training or validation partition")
    if task["objective"] == "classification" and set(y.iloc[b]) - set(y.iloc[a]):
        raise ValueError("Validation contains unseen classes; correct the split or collect more data")
    fingerprint = hashlib.sha256(a.tobytes() + b.tobytes()).hexdigest()
    return a, b, fingerprint


def as_strings(x):
    frame = pd.DataFrame(x).astype(object)
    return frame.where(pd.notna(frame), "__missing__").astype(str).to_numpy()


def finite_values(x):
    frame = pd.DataFrame(x)
    return frame.replace([np.inf, -np.inf], np.nan)


def build_model(task, X, stage, threads, seed):
    classification = task["objective"] == "classification"
    if task["kind"] == "text":
        char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                               max_features=task.get("max_char_features", 50000), sublinear_tf=True)
        features = char
        if stage == "upgrade":
            word = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True)
            features = FeatureUnion([("char", char), ("word", word)])
        estimator = (LogisticRegression(C=task.get("C", 2.0), max_iter=250, random_state=seed)
                     if classification else Ridge(alpha=2.0, solver="lsqr"))
        return Pipeline([("features", features), ("model", estimator)])
    numeric = list(X.select_dtypes(include="number").columns)
    categorical = [c for c in X.columns if c not in numeric]
    prep = ColumnTransformer([
        ("numeric", Pipeline([("finite", FunctionTransformer(finite_values)),
                               ("impute", SimpleImputer(strategy="median", keep_empty_features=True))]), numeric),
        ("category", Pipeline([("string", FunctionTransformer(as_strings)),
                                ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ], sparse_threshold=1.0)
    kwargs = dict(n_estimators=64 if stage == "baseline" else int(task.get("trees", 256)),
                  n_jobs=threads, random_state=seed, min_samples_leaf=1)
    estimator = ExtraTreesClassifier(class_weight="balanced", **kwargs) if classification else ExtraTreesRegressor(**kwargs)
    return Pipeline([("preprocess", prep), ("model", estimator)])


def predictions(model, X, task):
    if task.get("prediction_type", "label") != "probability":
        return model.predict(X)
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    if len(task["prediction_columns"]) == 1:
        if "positive_class" not in task:
            raise ValueError("Set positive_class explicitly for a binary probability submission")
        return probs[:, classes.index(task["positive_class"])]
    order = task.get("class_order")
    if order is None or set(order) != set(classes):
        raise ValueError("class_order must explicitly map every class to probability columns")
    return probs[:, [classes.index(c) for c in order]]


def metric_score(model, X, y, task):
    metric = task["metric"]
    if metric == "roc_auc":
        return float(roc_auc_score((y == task["positive_class"]).astype(int),
                                  model.predict_proba(X)[:, list(model.classes_).index(task["positive_class"])]))
    if metric == "log_loss":
        return float(log_loss(y, model.predict_proba(X), labels=model.classes_))
    pred = model.predict(X)
    funcs = {"accuracy": accuracy_score, "balanced_accuracy": balanced_accuracy_score,
             "macro_f1": lambda a, b: f1_score(a, b, average="macro", zero_division=0),
             "weighted_f1": lambda a, b: f1_score(a, b, average="weighted", zero_division=0),
             "rmse": lambda a, b: np.sqrt(mean_squared_error(a, b)), "mae": mean_absolute_error}
    if metric not in funcs:
        raise ValueError(f"Unsupported metric {metric}")
    return float(funcs[metric](y, pred))


def publish_fallback(ctx):
    task, cfg = ctx.task, ctx.cfg
    test = read_csv(cfg, task["test_csv"], task)
    ids = test[task["id_column"]].drop_duplicates().reset_index(drop=True)
    if task["kind"] in {"yolo", "custom"}:
        if task["kind"] == "custom":
            return  # No invented solution for an unknown task.
        if not task.get("allow_empty_predictions", False):
            raise ValueError("YOLO requires explicit allow_empty_predictions for no-detection rows")
        frame = make_submission(cfg, task, ids, np.array([""] * len(ids), dtype=object))
        ctx.publish(frame, None, {"algorithm": "empty_detections", "feature_mode": "empty"}, stage="fallback")
        return
    train = read_csv(cfg, task["train_csv"], task)
    y = train[task["target"]]
    if task["kind"] == "timeseries":
        y = train.groupby(task["id_column"], sort=False)[task["target"]].first()
    if y.isna().any():
        raise ValueError("Null targets")
    model = DummyClassifier(strategy="prior") if task["objective"] == "classification" else DummyRegressor(strategy="mean")
    model.fit(np.zeros((len(y), 1)), y)
    frame = make_submission(cfg, task, ids, predictions(model, np.zeros((len(ids), 1)), task))
    ctx.publish(frame, None, {"algorithm": type(model).__name__, "feature_mode": "constant"},
                model=model, stage="fallback")


def train_cpu(ctx):
    X, Xt, y, ids, metadata = load_dataset(ctx.cfg, ctx.task)
    a, b, split_hash = split_indices(ctx.task, y, metadata, ctx.cfg["seed"])
    model = build_model(ctx.task, X, ctx.stage, ctx.threads, ctx.cfg["seed"])
    model.fit(X.iloc[a], y.iloc[a])
    score = metric_score(model, X.iloc[b], y.iloc[b], ctx.task)
    # Validation is measured before refitting on all available training records.
    model.fit(X, y)
    frame = make_submission(ctx.cfg, ctx.task, ids, predictions(model, Xt, ctx.task))
    ctx.publish(frame, score, {"algorithm": type(model[-1]).__name__, "feature_mode": "dataset",
                              "split_hash": split_hash, "train_rows": len(a), "valid_rows": len(b),
                              "refit_all_rows": True}, model=model)


def predict_checkpoint(cfg, task, candidate, device="cpu"):
    import json
    candidate = Path(candidate)
    info = json.loads((candidate / "manifest.json").read_text())
    mode = info["details"]["feature_mode"]
    test = read_csv(cfg, task["test_csv"], task)
    ids = test[task["id_column"]].drop_duplicates().reset_index(drop=True)
    if mode == "empty":
        return make_submission(cfg, task, ids, np.array([""] * len(ids), dtype=object))
    if mode in {"resnet", "yolo"}:
        from .vision import predict_vision_checkpoint
        return predict_vision_checkpoint(cfg, task, candidate, device)
    if mode in {"sensor_cnn", "transformer"}:
        from .neural import predict_neural_checkpoint
        return predict_neural_checkpoint(cfg, task, candidate, device)
    model = joblib.load(candidate / "model.joblib")
    if mode == "constant":
        Xt = np.zeros((len(ids), 1))
    else:
        _, Xt, _, ids, _ = load_dataset(cfg, task)
    return make_submission(cfg, task, ids, predictions(model, Xt, task))
