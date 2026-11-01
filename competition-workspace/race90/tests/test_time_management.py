from pathlib import Path
import json
import threading
import time
import zipfile
import nbformat
import pytest
import yaml
from test_harness import demo
from race90.config import load_config
from race90.timing import timeline
from race90.cli import main


def test_90_minute_default_schedule_reserves_submission_time():
    cfg = load_config(Path(__file__).parents[1] / "config.example.yaml")
    plan = timeline(cfg, 90 * 60)
    assert plan["compute_deadline"] == 70 * 60
    assert plan["retrieval_deadline"] == 72 * 60
    assert plan["notebook_deadline"] == 84 * 60
    assert plan["deliverable_deadline"] == 85 * 60
    assert plan["exam_deadline"] == 90 * 60


def test_invalid_finalization_reserve_rejected(demo, tmp_path):
    demo["time_management"] = {"upload_minutes": 1}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(demo))
    with pytest.raises(ValueError, match="leave time"):
        load_config(path)


def test_exam_clock_cannot_be_silently_restarted(tmp_path):
    path = tmp_path / "clock.json"
    assert main(["start-clock", "--output", str(path)], start_time=1000) == 0
    assert json.loads(path.read_text())["deadline"] == 6400
    with pytest.raises(FileExistsError):
        main(["start-clock", "--output", str(path)], start_time=1500)
    assert json.loads(path.read_text())["deadline"] == 6400


def test_run_uses_exam_clock_including_time_already_spent(demo, tmp_path, monkeypatch):
    import race90.runner as runner
    path, clock = tmp_path / "config.yaml", tmp_path / "clock.json"
    path.write_text(yaml.safe_dump(demo))
    clock.write_text(json.dumps({"deadline": 12345}))
    seen = []
    def fake_run(cfg, resume, deadline):
        seen.append(deadline)
        return {"complete": True, "submission_ready": True, "fallback_tasks": [], "run_dir": "test"}
    monkeypatch.setattr(runner, "run", fake_run)
    assert main(["run", str(path), "--clock", str(clock)], start_time=12000) == 0
    assert seen == [12345]


def test_enhancement_failure_preserves_complete_compact_notebooks(demo, tmp_path, monkeypatch):
    import race90.deliverables as delivery
    selected = {t["name"]: (tmp_path / t["name"], {}) for t in demo["tasks"]}
    calls = []
    def attempt(cfg, task, candidate, run_dir, mode, deadline):
        calls.append((task["name"], mode))
        if mode == "full":
            raise TimeoutError("large model packaging timed out")
        assert cfg["notebooks"]["compact_export"]
        assert cfg["notebooks"]["mode"] == "restore"
        out = tmp_path / "staged" / task["name"]
        out.mkdir(parents=True, exist_ok=True)
        notebook, csv = out / f"{task['name']}.solution.ipynb", out / f"{task['name']}_submission.csv"
        notebook.write_text("verified compact copy")
        csv.write_text("id,label\n1,0\n")
        return notebook, csv
    monkeypatch.setattr(delivery, "export_attempt", attempt)
    result = delivery.final_notebooks(demo, tmp_path, selected, time.time() + 60)
    assert [mode for _, mode in calls[:3]] == ["compact"] * 3
    assert all(v["status"] == "done" and v["variant"] == "compact" for v in result.values())
    assert all("enhancement_error" in v for v in result.values())
    assert all(Path(v["path"]).read_text() == "verified compact copy" for v in result.values())


def test_slow_first_notebook_leaves_time_for_other_tasks_and_retry(demo, tmp_path, monkeypatch):
    import race90.deliverables as delivery
    now = [100.]
    monkeypatch.setattr(delivery.time, "time", lambda: now[0])
    calls = []
    def attempt(cfg, task, candidate, run_dir, mode, deadline):
        calls.append((task["name"], mode, now[0], deadline))
        assert deadline > now[0]
        if task["name"] == "q1" and mode == "compact":
            now[0] = deadline
            raise TimeoutError("slow first question")
        now[0] += 1
        out = tmp_path / "staged" / task["name"]
        out.mkdir(parents=True, exist_ok=True)
        notebook, csv = out / f"{task['name']}.solution.ipynb", out / f"{task['name']}_submission.csv"
        notebook.write_text(mode)
        csv.write_text("id,label\n1,0\n")
        return notebook, csv
    monkeypatch.setattr(delivery, "export_attempt", attempt)
    selected = {t["name"]: (tmp_path / t["name"], {}) for t in demo["tasks"]}
    result = delivery.final_notebooks(demo, tmp_path, selected, 220)
    assert [(name, mode) for name, mode, _, _ in calls[:4]] == [
        ("q1", "compact"), ("q2", "compact"), ("q3", "compact"), ("q1", "compact-retry")]
    assert all(v["status"] == "done" for v in result.values())


def test_unexecuted_notebook_cannot_pass_verification(demo, tmp_path):
    from race90.deliverables import verify_notebook
    task = demo["tasks"][0]
    original = nbformat.read(Path(demo["data_root"]) / task["notebook"], as_version=4)
    original.cells.extend([nbformat.v4.new_markdown_cell("Python 소스: test"), nbformat.v4.new_code_cell("print(1)")])
    original.metadata["race90"] = {"candidate": "test"}
    path = tmp_path / "incomplete.ipynb"
    nbformat.write(original, path)
    with pytest.raises(ValueError, match="not all executed"):
        verify_notebook(path, task, demo, Path("test"))


@pytest.mark.parametrize("minor", [0, 2, 4, 5])
def test_export_preserves_and_validates_older_notebook_formats(demo, tmp_path, minor, monkeypatch):
    from race90.notebook import export_notebook
    from race90.deliverables import verify_notebook
    from race90.io import read_csv
    from race90.worker import Context
    task = demo["tasks"][0]
    original_path = Path(demo["data_root"]) / task["notebook"]
    original = nbformat.read(original_path, as_version=4)
    original.nbformat_minor = minor
    if minor < 5:
        for cell in original.cells:
            cell.pop("id", None)
    nbformat.write(original, original_path)
    ctx = Context(demo, task, tmp_path / "task", "baseline", time.time() + 60)
    sample = read_csv(demo, task["sample_submission"], task)
    candidate = ctx.publish(sample, 1.0, {"feature_mode": "custom"})
    # Exercise real execution for the 4.2 format supplied in the practice data.
    demo["notebooks"].update(execute=minor == 2, mode="restore")
    monkeypatch.setenv("RACE90_DATA_ROOT", demo["data_root"])
    monkeypatch.setenv("RACE90_OUTPUT_DIR", str(tmp_path))
    output = tmp_path / "solution.ipynb"
    export_notebook(demo, task, candidate, output, time.time() + 60)
    verify_notebook(output, task, demo, candidate)
    saved = nbformat.read(output, as_version=4)
    assert saved.nbformat_minor == minor
    assert saved.cells[:len(original.cells)] == original.cells


def test_manual_finalize_does_not_start_new_training(demo, tmp_path):
    from race90.scheduling import run_stages
    from race90.io import atomic_json
    atomic_json(tmp_path / "run.json", {"deadline": time.time() + 60})
    assert main(["finalize", str(tmp_path)]) == 0
    calls, events = [], []
    run_stages(demo, tmp_path, time.time() + 60, 1, 1, threading.Event(), None,
               lambda directory, kind, **kw: events.append(kind), lambda *a, **k: calls.append(a),
               lambda *a: {})
    assert not calls
    assert "finalization_requested" in events


def test_upgrade_is_skipped_when_time_cannot_produce_checkpoint(demo, tmp_path):
    from race90.scheduling import run_stages
    from test_review_fixes import publish_dummy
    base = demo["tasks"][2]
    demo["tasks"] = [dict(base, resource="cpu", improve=True, minimum_upgrade_seconds=3600)]
    stages, events = [], []
    def local(cfg, task, root, stage, deadline, threads, **kwargs):
        stages.append(stage)
        publish_dummy(cfg, task, root / task["name"], stage)
    run_stages(demo, tmp_path, time.time() + 60, 1, 1, threading.Event(), None,
               lambda directory, kind, **kw: events.append(kind), local, lambda *a: {})
    assert stages == ["baseline"]
    assert "upgrade_skipped" in events


def test_packaging_timeout_keeps_previous_verified_zip(tmp_path):
    from race90.deliverables import package_deliverables
    deliver = tmp_path / "deliverables"
    deliver.mkdir()
    (deliver / "q1.csv").write_text("id,label\n1,0\n")
    package_deliverables(tmp_path, time.time() + 10)
    original = (tmp_path / "submission_pack.zip").read_bytes()
    with pytest.raises(TimeoutError):
        package_deliverables(tmp_path, time.time() - 1)
    assert (tmp_path / "submission_pack.zip").read_bytes() == original
    with zipfile.ZipFile(tmp_path / "submission_pack.zip") as archive:
        assert archive.testzip() is None
