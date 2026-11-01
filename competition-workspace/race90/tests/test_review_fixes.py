from pathlib import Path
import importlib.util
import json
import threading
import time
import tarfile
import numpy as np
import pandas as pd
import pytest
import yaml
from test_harness import demo
from race90.config import load_config
from race90.io import make_submission, read_csv, atomic_json
from race90.models import train_cpu, predict_checkpoint
from race90.selection import select_best
from race90.worker import Context


def test_refit_failure_keeps_a_real_trained_checkpoint(demo, tmp_path, monkeypatch):
    import race90.models as models
    from sklearn.pipeline import Pipeline
    original = Pipeline.fit
    fits = []

    def fit(self, X, y, **kwargs):
        fits.append(len(X))
        if len(fits) == 2:
            raise TimeoutError("simulated deadline during full refit")
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", fit)
    task = demo["tasks"][2]
    ctx = Context(demo, task, tmp_path / "task", "baseline", time.time() + 120)
    with pytest.raises(TimeoutError):
        train_cpu(ctx)
    candidate, info = select_best(demo, task, ctx.output)
    assert info["score"] is not None
    assert not info["details"]["refit_all_rows"]
    assert len(predict_checkpoint(demo, task, candidate)) == 10


def test_refit_wins_a_tied_validation_score(demo, tmp_path):
    task = demo["tasks"][2]
    ctx = Context(demo, task, tmp_path / "task", "baseline", time.time() + 120)
    train_cpu(ctx)
    _, info = select_best(demo, task, ctx.output)
    assert info["details"]["refit_all_rows"]


def test_string_labels_and_leading_zero_ids_survive_saved_csv(demo, tmp_path):
    task = dict(demo["tasks"][2], allowed_labels=["001", "NA"],
                read_csv_kwargs={"dtype": {"id": "string"}})
    sample_path = Path(demo["data_root"]) / task["sample_submission"]
    sample = pd.read_csv(sample_path)
    sample["id"] = [f"00{i}" for i in range(len(sample))]
    sample.to_csv(sample_path, index=False)
    ctx = Context(demo, task, tmp_path / "task", "baseline", time.time() + 60)
    frame = make_submission(demo, task, sample.id, ["001", "NA"] * 5)
    ctx.publish(frame, .5, {"feature_mode": "custom"})
    candidate, _ = select_best(demo, task, ctx.output)
    saved = pd.read_csv(ctx.output / "submission.csv", dtype=str, keep_default_na=False)
    assert saved.label.tolist() == ["001", "NA"] * 5
    assert saved.id.tolist() == sample.id.tolist()


def test_shared_table_excludes_other_question_targets_and_train_only_columns(demo):
    from race90.data import load_dataset
    base = dict(demo["tasks"][2], kind="tabular")
    other = dict(base, name="other", target="second_label")
    demo["tasks"] = [base, other]
    root = Path(demo["data_root"])
    for key in ("train_csv", "test_csv"):
        path = root / base[key]
        frame = pd.read_csv(path)
        frame["second_label"] = np.arange(len(frame)) % 2
        if key == "train_csv":
            frame["train_only_notes"] = "do not learn"
        frame.to_csv(path, index=False)
    X, Xt, _, _, _ = load_dataset(demo, base)
    assert list(X.columns) == list(Xt.columns) == ["text"]


@pytest.mark.parametrize("patch, message", [
    ({"metric": "roc_auc"}, "positive_class"),
    ({"metric": "typo"}, "unsupported metric"),
    ({"split": {"strategy": "time", "time_column": "id", "gap": -1}}, "gap"),
    ({"prediction_columns": ["a", "b"], "prediction_type": "probability"}, "class_order"),
    ({"baseline_seconds": 0}, "positive and finite"),
])
def test_bad_settings_fail_before_training(demo, tmp_path, patch, message):
    demo["tasks"][2].update(patch)
    config = tmp_path / "bad.yaml"
    config.write_text(yaml.safe_dump(demo))
    with pytest.raises(ValueError, match=message):
        load_config(config)


def publish_dummy(cfg, task, output, stage, score=.5):
    ctx = Context(cfg, task, output, stage, time.time() + 60)
    ids = read_csv(cfg, task["test_csv"], task).id
    return ctx.publish(make_submission(cfg, task, ids, np.arange(len(ids)) % 2), score,
                       {"feature_mode": "custom"})


def test_gpu_starts_while_cpu_baseline_is_still_running(demo, monkeypatch):
    import race90.runner as runner
    gpu_started = threading.Event()
    original = demo["tasks"][2]
    demo["tasks"] = [dict(original, resource="gpu", improve=True)]
    demo["runpod"].update(enabled=True, gpu_count=1)
    demo["notebooks"]["execute"] = False
    outcomes = []

    class FakeRunpod:
        def __init__(self, cfg, run_dir, deadline, cancel):
            self.cfg, self.run_dir = cfg, Path(run_dir)
        def prepare(self):
            return self
        def run_task(self, task, gpu, deadline, threads):
            gpu_started.set()
            publish_dummy(self.cfg, task, self.run_dir / task["name"], "upgrade", .9)
        def cleanup(self):
            pass

    def baseline(cfg, task, run_dir, stage, deadline, threads, **kwargs):
        outcomes.append(gpu_started.wait(3))
        publish_dummy(cfg, task, Path(run_dir) / task["name"], stage)

    monkeypatch.setattr(runner, "Runpod", FakeRunpod)
    monkeypatch.setattr(runner, "local_stage", baseline)
    report = runner.run(demo)
    assert outcomes == [True]
    assert report["tasks"][original["name"]]["score"] == .9


def test_cpu_queue_gives_independent_baselines_priority_and_reserves_child_time(demo, monkeypatch):
    import race90.runner as runner
    original = demo["tasks"][2]
    demo["cpu_workers"] = 1
    demo["tasks"] = [dict(original, name=name, kind="tabular", resource="cpu", improve=improve,
                          baseline_seconds=600, upgrade_seconds=3600, depends_on=parents)
                     for name, improve, parents in [("parent", True, []), ("child", False, ["parent"]),
                                                     ("other", False, [])]]
    demo["notebooks"]["execute"] = False
    calls = []

    def stage(cfg, task, run_dir, stage, deadline, threads, **kwargs):
        calls.append((task["name"], stage, deadline))
        assert all(p in cfg["dependency_outputs"] for p in task["depends_on"])
        publish_dummy(cfg, task, Path(run_dir) / task["name"], stage)

    monkeypatch.setattr(runner, "local_stage", stage)
    report = runner.run(demo)
    order = [(name, stage) for name, stage, _ in calls]
    assert order.index(("other", "baseline")) < order.index(("parent", "upgrade"))
    assert order.index(("parent", "upgrade")) < order.index(("child", "baseline"))
    compute_end = report["deadline"] - demo["finalize_minutes"] * 60
    assert next(d for n, s, d in calls if (n, s) == ("parent", "upgrade")) < compute_end - 5


def test_remote_pack_only_contains_selected_committed_checkpoint(demo, tmp_path):
    from race90.remote import pack_result
    task = demo["tasks"][2]
    output = tmp_path / "results" / task["name"]
    publish_dummy(demo, task, output, "upgrade", .5)
    best = publish_dummy(demo, task, output, "upgrade", .9)
    (output / "large-optimizer.bin").write_bytes(b"unneeded")
    (output / "candidates" / "incomplete").mkdir()
    payload, archive = tmp_path / "payload.json", tmp_path / "result.tar"
    atomic_json(payload, {"cfg": demo, "task": task, "output": str(output)})
    pack_result(payload, archive)
    with tarfile.open(archive) as packed:
        names = packed.getnames()
    assert any(best.name in name for name in names)
    assert not any("optimizer" in name or "incomplete" in name for name in names)
    assert sum(name.endswith("manifest.json") for name in names) == 1


def test_changing_parent_checkpoint_invalidates_old_child(demo, tmp_path):
    base = demo["tasks"][2]
    parent = dict(base, name="parent")
    child = dict(base, name="child", depends_on=["parent"])
    first = publish_dummy(demo, parent, tmp_path / "parent", "baseline", .5)
    demo["dependency_outputs"] = {"parent": str(first)}
    publish_dummy(demo, child, tmp_path / "child", "baseline", .8)
    assert select_best(demo, child, tmp_path / "child")[1]["score"] == .8
    second = publish_dummy(demo, parent, tmp_path / "parent", "upgrade", .9)
    demo["dependency_outputs"]["parent"] = str(second)
    with pytest.raises(ValueError, match="no valid completed candidate"):
        select_best(demo, child, tmp_path / "child")


def test_retrieval_ignores_training_cancellation(demo, tmp_path, monkeypatch):
    from race90.remote import Runpod
    cancel = threading.Event()
    pod = Runpod(demo, tmp_path, time.time() + 90, cancel)
    pod.host = "127.0.0.1"
    retrieval = []

    def ssh(argv, **kwargs):
        if "timeout" in argv:
            cancel.set()
            raise TimeoutError("training stopped")
        if cancel.is_set():
            retrieval.append(kwargs.get("cancellable"))

    def copy(src, dst, **kwargs):
        if str(src).startswith("root@"):
            retrieval.append(kwargs.get("cancellable"))
            with tarfile.open(dst, "w"):
                pass

    monkeypatch.setattr(pod, "ssh", ssh)
    monkeypatch.setattr(pod, "copy", copy)
    with pytest.raises(TimeoutError, match="training stopped"):
        pod.run_task(demo["tasks"][1], 0, time.time() + 60, 1)
    assert retrieval == [False, False]


def test_remote_install_supports_externally_managed_python(demo, tmp_path, monkeypatch):
    from race90.remote import Runpod
    pod = Runpod(demo, tmp_path, time.time() + 90, threading.Event())
    pod.host = "127.0.0.1"
    created_venv = False
    installed = False

    def ssh(argv, **kwargs):
        nonlocal created_venv, installed
        if "venv" in argv:
            assert "--system-site-packages" in argv  # Keep the image's CUDA/PyTorch.
            created_venv = True
        if any("pip" in part and "install" in part for part in argv):
            if argv[0] == "python":
                raise RuntimeError("externally-managed-environment")
            assert created_venv and argv[0] == pod.remote_python
            installed = True
        if "PYTHONPATH=" + pod.remote_root in argv:
            assert pod.remote_python in argv

    monkeypatch.setattr(pod, "ssh", ssh)
    monkeypatch.setattr(pod, "copy", lambda *a, **kw: None)
    pod.upload_and_install(time.time() + 60)
    assert installed


@pytest.mark.skipif(importlib.util.find_spec("catboost") is None, reason="optional CatBoost missing")
@pytest.mark.parametrize("objective", ["classification", "regression"])
def test_catboost_native_checkpoint_infers_on_cpu(demo, tmp_path, objective):
    from race90.tabular_gpu import train_tabular
    task = dict(demo["tasks"][2], kind="tabular", objective=objective,
                metric="accuracy" if objective == "classification" else "rmse",
                catboost_iterations=[5, 10], depth=2)
    root = Path(demo["data_root"])
    for key in ("train_csv", "test_csv"):
        path = root / task[key]
        table = pd.read_csv(path)
        table["signal"] = np.arange(len(table)) % 2
        table["category"] = ["a" if i % 2 else None for i in range(len(table))]
        table.drop(columns="text").to_csv(path, index=False)
    ctx = Context(demo, task, tmp_path / "catboost", "upgrade", time.time() + 120)
    ctx.output.mkdir()
    train_tabular(ctx)
    candidate, info = select_best(demo, task, ctx.output)
    assert info["details"]["feature_mode"] == "catboost"
    assert len(predict_checkpoint(demo, task, candidate)) == 10


@pytest.mark.skipif(importlib.util.find_spec("torchvision") is None, reason="optional torchvision missing")
def test_frozen_image_embeddings_checkpoint_infers_on_cpu(demo, tmp_path):
    from race90.vision import train_image_embeddings
    task = dict(demo["tasks"][1], image_size=32, pretrained=False, batch_size=8)
    ctx = Context(demo, task, tmp_path / "embeddings", "upgrade", time.time() + 120)
    ctx.output.mkdir()
    train_image_embeddings(ctx)
    candidate, info = select_best(demo, task, ctx.output)
    assert info["details"]["feature_mode"] == "resnet_features"
    assert len(predict_checkpoint(demo, task, candidate)) == 8
