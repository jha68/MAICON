from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import io
import time
import numpy as np
import pandas as pd
import pytest
import yaml
import nbformat
from race90.config import load_config
from race90.io import make_submission, validate_submission, atomic_json
from race90.data import sensor_features
from race90.models import split_indices
from race90.process import run_command
from race90.worker import Context
from race90.selection import select_best
from race90.bundle import safe_extract_tar
from race90.remote import eligible_gpus


@pytest.fixture
def demo(tmp_path):
    script = Path(__file__).parents[1] / "scripts/make_demo.py"
    spec = importlib.util.spec_from_file_location("demo_builder", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return load_config(module.make_demo(tmp_path))


def test_submission_order_and_probability_checks(demo):
    task = demo["tasks"][2]
    ids = pd.Series(np.arange(1000, 1010))
    frame = make_submission(demo, task, ids, np.arange(10) % 2)
    assert frame.id.tolist() == list(range(1009, 999, -1))
    assert frame.label.tolist() == (np.arange(10) % 2)[::-1].tolist()
    template = frame.copy()
    with pytest.raises(ValueError, match="IDs/order"):
        validate_submission(frame.iloc[::-1].reset_index(drop=True), template, task)
    task = {**task, "prediction_type": "probability"}
    frame.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="Probabilities"):
        validate_submission(frame, template, task)
    frame["label"] = frame["label"].astype(float)
    frame.loc[0, "label"] = np.inf
    with pytest.raises(ValueError, match="Non-finite"):
        validate_submission(frame, template, {**task, "allow_empty_predictions": True})


def test_group_and_time_validation_do_not_leak():
    task = {"objective": "classification", "split": {"strategy": "group", "group_column": "person", "fraction": .25}}
    y = pd.Series([0, 1] * 8)
    meta = pd.DataFrame({"person": np.repeat(np.arange(4), 4), "time": np.repeat(np.arange(8), 2)})
    a, b, _ = split_indices(task, y, meta, 42)
    assert not set(meta.person.iloc[a]) & set(meta.person.iloc[b])
    task["split"] = {"strategy": "time", "time_column": "time", "fraction": .3, "gap": 1}
    a, b, _ = split_indices(task, y, meta, 42)
    assert meta.time.iloc[a].max() < meta.time.iloc[b].min()


def test_sensor_features_respect_recording_boundaries():
    df = pd.DataFrame({"id": [1, 1, 2, 2], "time": [1, 0, 1, 0], "sensor": [3., 1., 30., 10.]})
    out = sensor_features(df, {"id_column": "id", "sensor_columns": ["sensor"], "time_column": "time"}).set_index("id")
    assert out.loc[1, "sensor_mean"] == 2
    assert out.loc[2, "sensor_delta"] == 20


def test_selection_keeps_completed_candidate_when_later_candidate_is_incomplete(demo, tmp_path):
    task = demo["tasks"][2]
    ctx = Context(demo, task, tmp_path / "task", "baseline", time.time() + 60)
    frame = make_submission(demo, task, np.arange(1000, 1010), np.arange(10) % 2)
    chosen = ctx.publish(frame, .8, {"feature_mode": "custom", "split_hash": "same"})
    ctx.publish(frame, .7, {"feature_mode": "custom", "split_hash": "same"})
    incomplete = ctx.output / "candidates/upgrade-incomplete"
    incomplete.mkdir()
    frame.to_csv(incomplete / "submission.csv", index=False)
    assert select_best(demo, task, ctx.output)[0] == chosen
    bad = ctx.output / "candidates/bad"
    bad.mkdir()
    atomic_json(bad / "manifest.json", {"task": task["name"], "metric": task["metric"], "score": float("nan")})
    assert select_best(demo, task, ctx.output)[0] == chosen


def test_hard_timeout_kills_process_group(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    code = "import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
    started = time.time()
    with pytest.raises(TimeoutError):
        run_command([sys.executable, "-c", code, str(child_pid_file)], started + 1)
    assert time.time() - started < 4
    pid = int(child_pid_file.read_text())
    # A zombie is already terminated; it can remain until the container's PID 1 reaps it.
    stat = Path(f"/proc/{pid}/stat")
    for _ in range(20):
        if not stat.exists() or stat.read_text().split()[2] == "Z":
            break
        time.sleep(.02)
    assert not stat.exists() or stat.read_text().split()[2] == "Z"


def test_archive_traversal_rejected(tmp_path):
    bad = tmp_path / "bad.tar"
    with tarfile.open(bad, "w") as archive:
        member = tarfile.TarInfo("../escape")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract_tar(bad, tmp_path / "dest")


def test_live_catalog_price_cap_is_for_three_gpus():
    catalog = [{"gpuId": "NVIDIA A100-SXM4-80GB", "displayName": "A100 SXM", "securePricePerHr": 1.5}]
    assert eligible_gpus(catalog, [catalog[0]["gpuId"]], "SECURE", 3, 4) == []
    assert eligible_gpus(catalog, [catalog[0]["gpuId"]], "SECURE", 3, 5) == [(catalog[0]["gpuId"], 4.5)]


def test_dependency_cycle_rejected(demo, tmp_path):
    cfg = {k: v for k, v in demo.items() if k not in {"project_root", "dependency_outputs"}}
    cfg["tasks"][0]["depends_on"] = ["q3"]
    cfg["tasks"][2]["depends_on"] = ["q1"]
    path = tmp_path / "cycle.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="cycle"):
        load_config(path)


def test_three_questions_end_to_end_and_real_notebook_outputs(demo):
    from race90.runner import run
    demo["notebooks"]["mode"] = "infer"
    report = run(demo)
    assert report["complete"], report
    assert len(report["tasks"]) == 3
    assert all(t["score"] is not None for t in report["tasks"].values())
    for name in report["tasks"]:
        nb = nbformat.read(Path(report["run_dir"]) / "deliverables" / f"{name}.solution.ipynb", as_version=4)
        assert "starter cell" in nb.cells[1].source
        assert nb.cells[1].execution_count is None
        executed = [c for c in nb.cells if c.cell_type == "code" and c.execution_count is not None]
        assert len(executed) == 3
        assert any("저장 모델로 CPU 추론" in str(c.outputs) for c in executed)
        assert not any(o.output_type == "error" for c in executed for o in c.outputs)
    old = json.loads((Path(report["run_dir"]) / "run.json").read_text())
    demo["seed"] += 1
    with pytest.raises(ValueError, match="configuration"):
        run(demo, resume=report["run_dir"])
    demo["seed"] -= 1
    with pytest.raises(ValueError, match="original deadline"):
        run(demo, resume=report["run_dir"], deadline_epoch=old["deadline"] + 100)


def test_dependency_dag_and_three_gpu_slots(demo, monkeypatch):
    import threading
    import race90.runner as runner
    from race90.models import publish_fallback
    from race90.io import read_csv
    events, event_lock = [], threading.Lock()
    parent_dirs = {}
    demo["notebooks"]["execute"] = False
    demo["project_root"] = str(Path(__file__).parents[1])
    # All tasks use the same table but are independent GPU jobs in this scenario.
    original = demo["tasks"][2]
    demo["tasks"] = [{**original, "name": f"q{i}", "kind": "custom", "adapter": "custom_adapters.example:solve",
                      "resource": "gpu", "improve": True, "depends_on": []} for i in range(1, 4)]
    demo["runpod"].update(enabled=True, gpu_count=3)
    barrier = threading.Barrier(3, timeout=5)
    class FakeRunpod:
        def __init__(self, cfg, run_dir, deadline, cancel):
            self.cfg, self.run_dir = cfg, Path(run_dir)
        def prepare(self):
            return self
        def run_task(self, task, gpu, deadline, threads):
            with event_lock:
                events.append((task["name"], gpu, time.time()))
            barrier.wait()
            ctx = Context(self.cfg, task, self.run_dir / task["name"], "upgrade", deadline)
            ids = read_csv(self.cfg, task["test_csv"], task).id
            frame = make_submission(self.cfg, task, ids, np.arange(len(ids)) % 2)
            ctx.publish(frame, .9, {"feature_mode": "custom"})
        def cleanup(self):
            pass
    def cpu(cfg, task, run_dir, stage, deadline, threads, device="cpu", cancel=None):
        ctx = Context(cfg, task, Path(run_dir) / task["name"], stage, deadline)
        assert all(parent in cfg["dependency_outputs"] for parent in task["depends_on"])
        ids = read_csv(cfg, task["test_csv"], task).id
        ctx.publish(make_submission(cfg, task, ids, np.arange(len(ids)) % 2), .5, {"feature_mode": "custom"})
    monkeypatch.setattr(runner, "Runpod", FakeRunpod)
    monkeypatch.setattr(runner, "local_stage", cpu)
    report = runner.run(demo)
    assert report["complete"]
    assert {gpu for _, gpu, _ in events} == {0, 1, 2}
    assert max(t for _, _, t in events) - min(t for _, _, t in events) < 2
    # In a connected problem, q2 starts only after q1's selected output exists.
    demo["runpod"]["enabled"] = False
    for task in demo["tasks"]:
        task["resource"] = "cpu"
        task["improve"] = False
    demo["tasks"][1]["depends_on"] = ["q1"]
    demo["tasks"][2]["depends_on"] = ["q2"]
    report = runner.run(demo)
    assert report["complete"]


@pytest.mark.skipif(importlib.util.find_spec("torchvision") is None, reason="optional vision dependencies missing")
def test_resnet_checkpoint_can_infer_on_cpu(demo, tmp_path):
    from race90.vision import train_resnet
    from race90.models import predict_checkpoint
    task = dict(demo["tasks"][1], pretrained=False, image_size=32, batch_size=8, epochs=1, inference_reserve_seconds=1)
    ctx = Context(demo, task, tmp_path / "vision", "upgrade", time.time() + 90)
    ctx.output.mkdir()
    train_resnet(ctx)
    path, info = select_best(demo, task, ctx.output)
    assert info["details"]["feature_mode"] == "resnet"
    frame = predict_checkpoint(demo, task, path)
    assert len(frame) == 8


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional torch dependency missing")
def test_sensor_cnn_checkpoint_can_infer_on_cpu(demo, tmp_path):
    from race90.neural import train_sensor
    from race90.models import predict_checkpoint
    task = dict(demo["tasks"][0], epochs=1, batch_size=16, sequence_length=32, inference_reserve_seconds=1)
    ctx = Context(demo, task, tmp_path / "sensor", "upgrade", time.time() + 60)
    ctx.output.mkdir()
    train_sensor(ctx)
    path, info = select_best(demo, task, ctx.output)
    assert info["details"]["feature_mode"] == "sensor_cnn"
    assert len(predict_checkpoint(demo, task, path)) == 10


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="optional transformers missing")
def test_text_transformer_checkpoint_offline(demo, tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast
    from race90.neural import train_text
    from race90.models import predict_checkpoint
    tiny = tmp_path / "tiny_model"
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
             "좋아요": 5, "재미있고": 6, "만족합니다": 7, "별로예요": 8, "지루하고": 9, "실망합니다": 10, "후기": 11}
    tokenizer = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    fast.save_pretrained(tiny)
    model = BertForSequenceClassification(BertConfig(vocab_size=len(vocab), hidden_size=16,
                    num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, max_position_embeddings=128, num_labels=2))
    model.save_pretrained(tiny)
    task = dict(demo["tasks"][2], model_name=str(tiny), epochs=1, batch_size=16, max_length=32,
                inference_reserve_seconds=1)
    ctx = Context(demo, task, tmp_path / "text", "upgrade", time.time() + 90)
    ctx.output.mkdir()
    train_text(ctx)
    path, info = select_best(demo, task, ctx.output)
    assert info["details"]["feature_mode"] == "transformer"
    assert len(predict_checkpoint(demo, task, path)) == 10


def test_runpod_creation_has_three_gpu_lease_and_no_live_rental(demo, tmp_path, monkeypatch):
    import threading
    import race90.remote as remote
    captured = {}
    demo["runpod"].update(gpu_count=3, max_gpu_hourly_usd=8)
    pod = remote.Runpod(demo, tmp_path, time.time() + 60, threading.Event())
    monkeypatch.setattr(remote, "api_key", lambda: "fake-test-key")
    monkeypatch.setattr(pod, "ctl", lambda *a, **kw: [{"gpuId": "NVIDIA A100-SXM4-80GB", "securePricePerHr": 1.5}])
    def fake_graphql(query, variables):
        captured.update(variables["input"])
        return {"podFindAndDeployOnDemand": {"id": "fake-pod-id"}}
    monkeypatch.setattr(remote, "graphql", fake_graphql)
    monkeypatch.setattr(remote.subprocess, "Popen", lambda *a, **kw: None)
    pod.create(time.time() + 30)
    assert captured["gpuCount"] == 3
    assert captured["terminateAfter"]
    assert pod.owned and pod.pod_id == "fake-pod-id"


def test_failed_remote_epoch_still_attempts_retrieval(demo, tmp_path, monkeypatch):
    import threading
    from race90.remote import Runpod
    pod = Runpod(demo, tmp_path, time.time() + 90, threading.Event())
    pod.host = "127.0.0.1"
    calls = []
    def ssh(argv, **kwargs):
        calls.append(argv)
        if "timeout" in argv:
            raise RuntimeError("later epoch failed")
    def copy(src, dst, **kwargs):
        if str(src).startswith("root@"):
            with tarfile.open(dst, "w"):
                pass
    monkeypatch.setattr(pod, "ssh", ssh)
    monkeypatch.setattr(pod, "copy", copy)
    with pytest.raises(RuntimeError, match="later epoch"):
        pod.run_task(demo["tasks"][1], 0, time.time() + 60, 1)
    assert any(argv[0] == "tar" for argv in calls)


def test_yolo_dataset_paths_and_split_overlap(demo, tmp_path):
    from race90.vision import normalized_yolo_yaml
    root = Path(demo["data_root"]) / "q2"
    images = root / "images"
    (root / "train.txt").write_text("images/0.png\n")
    (root / "val.txt").write_text("images/1.png\n")
    config = root / "dataset.yaml"
    config.write_text(yaml.safe_dump({"path": ".", "train": "train.txt", "val": "val.txt", "names": ["a", "b"]}))
    task = {**demo["tasks"][1], "data_yaml": "q2/dataset.yaml"}
    runtime, counts = normalized_yolo_yaml(demo, task, tmp_path / "normalized")
    data = yaml.safe_load(runtime.read_text())
    assert data["path"] == str(root)
    assert counts == {"train": 1, "val": 1}
    assert Path(data["train"]).read_text().strip() == str(images / "0.png")
    (root / "val.txt").write_text("images/0.png\n")
    with pytest.raises(ValueError, match="overlap"):
        normalized_yolo_yaml(demo, task, tmp_path / "normalized")


def test_shared_parent_features_and_child_notebook_cpu_inference(demo, tmp_path):
    from race90.io import read_csv
    from race90.models import train_cpu, predict_checkpoint
    from race90.notebook import export_notebook
    demo["project_root"] = str(Path(__file__).parents[1])
    parent = dict(demo["tasks"][2], name="parent", kind="custom", adapter="custom_adapters.example:solve")
    child = dict(demo["tasks"][2], name="child", kind="tabular", depends_on=["parent"],
                 dependency_features=[{"task": "parent", "train_file": "train_features.csv", "test_file": "test_features.csv", "on": "id"}])
    demo["tasks"] = [parent, child]
    train = read_csv(demo, parent["train_csv"], parent)
    test = read_csv(demo, parent["test_csv"], parent)
    train_file, test_file = tmp_path / "train_extra.csv", tmp_path / "test_extra.csv"
    pd.DataFrame({"id": train.id, "length_feature": train.text.str.len()}).to_csv(train_file, index=False)
    pd.DataFrame({"id": test.id, "length_feature": test.text.str.len()}).to_csv(test_file, index=False)
    ctx = Context(demo, parent, tmp_path / "parent", "baseline", time.time() + 60)
    selected = ctx.publish(make_submission(demo, parent, test.id, np.arange(len(test)) % 2), .5,
                           {"feature_mode": "custom"}, artifacts={"train_features.csv": train_file, "test_features.csv": test_file})
    demo["dependency_outputs"] = {"parent": str(selected)}
    ctx = Context(demo, child, tmp_path / "child", "baseline", time.time() + 60)
    train_cpu(ctx)
    candidate, _ = select_best(demo, child, ctx.output)
    assert len(predict_checkpoint(demo, child, candidate)) == 10
    demo["notebooks"]["mode"] = "infer"
    output = tmp_path / "child.solution.ipynb"
    export_notebook(demo, child, candidate, output, time.time() + 60)
    nb = nbformat.read(output, as_version=4)
    assert nb.metadata.race90.executed_solution_cells == 3


def test_tabular_missing_categories_and_infinite_numeric_features():
    from race90.models import build_model
    X = pd.DataFrame({"value": [1., 2., np.inf, np.nan, -np.inf, 3.],
                      "cat": pd.Series(["a", "b", None, "a", None, "b"], dtype="string")})
    model = build_model({"kind": "tabular", "objective": "classification"}, X, "baseline", 1, 42)
    model.fit(X, [0, 1, 0, 0, 0, 1])
    assert len(model.predict(pd.DataFrame({"value": [np.inf], "cat": ["new"]}))) == 1
