"""GPU CatBoost candidates with native checkpoints that also infer on CPU."""
import json
import time
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .data import load_dataset
from .io import atomic_json, make_submission
from .models import split_indices, metric_score, predictions, MINIMIZE


def prepare_features(frame, schema):
    frame = frame.copy()
    frame.columns = frame.columns.map(str)
    frame = frame[schema["columns"]].copy()
    for column in frame:
        if column in schema["categorical"]:
            frame[column] = frame[column].astype(object).where(frame[column].notna(), "__missing__").astype(str)
        else:
            frame[column] = frame[column].astype(float).replace([np.inf, -np.inf], np.nan)
    return frame


class CatBoostPredictor:
    def __init__(self, model, classes=None, threads=1):
        self.model, self.threads = model, threads
        if classes is not None:
            self.classes_ = np.asarray(classes)

    def predict(self, X):
        result = np.asarray(self.model.predict(X, task_type="CPU", thread_count=self.threads)).reshape(-1)
        return self.classes_[result.astype(int)] if hasattr(self, "classes_") else result

    def predict_proba(self, X):
        return self.model.predict_proba(X, task_type="CPU", thread_count=self.threads)


def train_tabular(ctx):
    from catboost import CatBoostClassifier, CatBoostRegressor
    X, Xt, y, ids, metadata = load_dataset(ctx.cfg, ctx.task)
    task = ctx.task
    a, b, split_hash = split_indices(task, y, metadata, ctx.cfg["seed"])
    classification = task["objective"] == "classification"
    encoder = LabelEncoder().fit(y) if classification else None
    labels = encoder.transform(y) if classification else y.to_numpy()
    schema = {"columns": [str(c) for c in X.columns],
              "categorical": [str(c) for c in X.columns if c not in X.select_dtypes(include="number").columns],
              "classes": encoder.classes_.tolist() if classification else None}
    X, Xt = prepare_features(X, schema), prepare_features(Xt, schema)
    schema_path, checkpoint = ctx.output / "schema-current.json", ctx.output / "catboost-current.cbm"
    atomic_json(schema_path, schema)
    model_class = CatBoostClassifier if classification else CatBoostRegressor
    kwargs = dict(task_type="GPU" if ctx.device.startswith("cuda") else "CPU",
                  thread_count=ctx.threads, random_seed=ctx.cfg["seed"],
                  depth=int(task.get("depth", 6)), learning_rate=float(task.get("learning_rate", .06)),
                  cat_features=schema["categorical"], verbose=False, allow_writing_files=False)
    if ctx.device.startswith("cuda"):
        kwargs["devices"] = "0"  # CUDA_VISIBLE_DEVICES already isolates the assigned GPU.
    best, best_model, best_seconds = None, None, 0
    for iterations in task.get("catboost_iterations", [100, 300, 800]):
        if best_model is not None and ctx.seconds_left() < best_seconds * iterations / best_model.tree_count_ * 1.5 + 10:
            break
        model = model_class(iterations=int(iterations), **kwargs)
        started = time.monotonic()
        model.fit(X.iloc[a], labels[a], eval_set=(X.iloc[b], labels[b]),
                  early_stopping_rounds=int(task.get("patience", 40)), use_best_model=True)
        elapsed = time.monotonic() - started
        predictor = CatBoostPredictor(model, schema["classes"], ctx.threads)
        score = metric_score(predictor, X.iloc[b], y.iloc[b], task)
        improved = best is None or (score < best if task["metric"] in MINIMIZE else score > best)
        if not improved:
            continue
        model.save_model(checkpoint)
        frame = make_submission(ctx.cfg, task, ids, predictions(predictor, Xt, task))
        details = {"algorithm": model_class.__name__, "feature_mode": "catboost", "split_hash": split_hash,
                   "train_rows": len(a), "valid_rows": len(b), "iterations": model.tree_count_, "refit_all_rows": False}
        ctx.publish(frame, score, details, artifacts={"model.cbm": checkpoint, "schema.json": schema_path})
        best, best_model, best_seconds = score, model, elapsed
    if best_model is None:
        raise RuntimeError("No CatBoost checkpoint completed")
    if task.get("refit", True) and ctx.seconds_left() > best_seconds * len(y) / len(a) * 1.5 + 15:
        model = model_class(iterations=best_model.tree_count_, **kwargs)
        model.fit(X, labels)
        model.save_model(checkpoint)
        frame = make_submission(ctx.cfg, task, ids, predictions(CatBoostPredictor(model, schema["classes"], ctx.threads), Xt, task))
        ctx.publish(frame, best, {**details, "refit_all_rows": True},
                    artifacts={"model.cbm": checkpoint, "schema.json": schema_path})


def predict_tabular_checkpoint(cfg, task, candidate):
    from catboost import CatBoostClassifier, CatBoostRegressor
    schema = json.loads((candidate / "schema.json").read_text())
    _, Xt, _, ids, _ = load_dataset(cfg, task)
    model = CatBoostClassifier() if schema["classes"] is not None else CatBoostRegressor()
    model.load_model(candidate / "model.cbm")
    predictor = CatBoostPredictor(model, schema["classes"])
    return make_submission(cfg, task, ids, predictions(predictor, prepare_features(Xt, schema), task))
