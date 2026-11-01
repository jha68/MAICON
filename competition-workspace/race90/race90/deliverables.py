"""Build all submission notebooks first; enrich only after a verified copy exists."""
from copy import deepcopy
from pathlib import Path
import json
import os
import shutil
import sys
import time
import zipfile
import nbformat
import numpy as np
from .config import resolve_data
from .io import atomic_json, read_csv, read_saved_submission, validate_submission, sha256
from .process import run_command


def verify_notebook(path, task, cfg, candidate):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    original = nbformat.read(resolve_data(cfg, task["notebook"]), as_version=4) if task.get("notebook") else None
    offset = len(original.cells) if original else 0
    if original and nb.cells[:offset] != original.cells:
        raise ValueError("Original question cells or outputs changed")
    metadata = nb.metadata.get("race90", {})
    if metadata.get("candidate") != candidate.name:
        raise ValueError("Notebook candidate differs from selected model")
    cells = [c for c in nb.cells[offset:] if c.cell_type == "code"]
    if not cells or not any(c.cell_type == "markdown" and "Python 소스:" in c.source for c in nb.cells[offset:]):
        raise ValueError("Notebook is missing solution code")
    if cfg["notebooks"].get("execute", True):
        if any(c.execution_count is None for c in cells):
            raise ValueError("Notebook solution cells were not all executed")
        if any(o.output_type == "error" for c in cells for o in c.outputs):
            raise ValueError("Notebook contains a solution execution error")
        if not any(c.outputs for c in cells):
            raise ValueError("Notebook has no execution outputs")
    return metadata


def export_attempt(cfg, task, candidate, run_dir, mode, deadline):
    work = run_dir / "notebook-work" / task["name"] / mode
    work.mkdir(parents=True, exist_ok=True)
    output = work / f"{task['name']}.solution.ipynb"
    job = work / "config.json"
    atomic_json(job, cfg)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent.parent), cfg["project_root"], env.get("PYTHONPATH", "")])
    env["RACE90_DATA_ROOT"] = cfg["data_root"]
    env["RACE90_OUTPUT_DIR"] = str(work)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "1"
    run_command([sys.executable, "-m", "race90", "_notebook", str(job), task["name"],
                 str(candidate), str(output), str(deadline)], deadline,
                log=run_dir / task["name"] / f"notebook-{mode}.log", env=env)
    marker = json.loads((work / "verification.json").read_text())
    if marker.get("candidate") != candidate.name or marker.get("status") != "verified":
        raise ValueError("Notebook worker did not verify the selected candidate")
    return output, work / f"{task['name']}_submission.csv"


def verify_export(cfg, task, candidate, output):
    """Run expensive parsing/checks inside the deadline-controlled worker."""
    candidate, output = Path(candidate), Path(output)
    work = output.parent
    verify_notebook(output, task, cfg, candidate)
    csv = work / f"{task['name']}_submission.csv"
    if cfg["notebooks"].get("execute", True):
        template = read_csv(cfg, task["sample_submission"], task)
        manifest = json.loads((candidate / "manifest.json").read_text())
        frame = read_saved_submission(csv, template, task, manifest)
        validate_submission(frame, template, task)
        expected = read_saved_submission(candidate / "submission.csv", template, task, manifest)
        for column in task["prediction_columns"]:
            a, b = frame[column], expected[column]
            if task.get("prediction_type") == "probability" or task.get("objective") == "regression":
                matches = np.allclose(a.astype(float), b.astype(float), rtol=1e-5, atol=1e-7)
            else:
                matches = a.equals(b)
            if not matches:
                raise ValueError("Notebook predictions differ from the selected candidate")
    else:
        shutil.copy2(candidate / "submission.csv", csv)
    atomic_json(work / "verification.json", {"status": "verified", "candidate": candidate.name,
                                            "csv_sha256": sha256(csv)})


def final_notebooks(cfg, run_dir, selected, deadline):
    run_dir = Path(run_dir)
    deliver = run_dir / "deliverables"
    deliver.mkdir(exist_ok=True)
    tasks = [t for t in cfg["tasks"] if t["name"] in selected]
    outcomes = {}
    compact = deepcopy(cfg)
    compact["notebooks"].update(mode="restore", embed_models=False, compact_export=True)
    enrich = cfg["notebooks"].get("embed_models", True) or cfg["notebooks"].get("mode") == "infer"
    # Half of the available window belongs to the first complete set. If no
    # richer version was requested, use the entire window for the base set.
    first_end = time.time() + max(0, deadline - time.time()) * (.5 if enrich else 1)

    def attempt(task, settings, mode, attempt_end):
        name = task["name"]
        candidate, _ = selected[name]
        output, csv = export_attempt(settings, task, candidate, run_dir, mode, attempt_end)
        # Only validated staged files replace a previously verified deliverable.
        os.replace(csv, deliver / csv.name)
        os.replace(output, deliver / output.name)
        marker = output.parent / "verification.json"
        digest = json.loads(marker.read_text())["csv_sha256"] if marker.exists() else sha256(deliver / csv.name)
        outcomes[name] = {"status": "done", "path": str(deliver / output.name),
                          "variant": mode, "mode": settings["notebooks"].get("mode", "restore"),
                          "models_embedded": settings["notebooks"].get("embed_models", True),
                          "executed": settings["notebooks"].get("execute", True),
                          "csv_sha256": digest}
        atomic_json(run_dir / "notebooks.json", outcomes)

    for i, task in enumerate(tasks):
        attempt_end = min(first_end, time.time() + max(0, first_end - time.time()) / (len(tasks) - i))
        try:
            attempt(task, compact, "compact", attempt_end)
        except Exception as exc:
            outcomes[task["name"]] = {"status": "failed", "error": str(exc)}
    # Repair missing base files before spending time on optional model bundles.
    missing = [t for t in tasks if outcomes[t["name"]]["status"] != "done"]
    for i, task in enumerate(missing):
        attempt_end = min(deadline, time.time() + max(0, deadline - time.time()) / (len(missing) - i))
        try:
            attempt(task, compact, "compact-retry", attempt_end)
        except Exception as exc:
            outcomes[task["name"]] = {"status": "failed", "error": str(exc)}
    if enrich and all(o["status"] == "done" for o in outcomes.values()):
        for i, task in enumerate(tasks):
            attempt_end = min(deadline, time.time() + max(0, deadline - time.time()) / (len(tasks) - i))
            try:
                attempt(task, cfg, "full", attempt_end)
            except Exception as exc:
                outcomes[task["name"]]["enhancement_error"] = str(exc)
    atomic_json(run_dir / "notebooks.json", outcomes)
    return outcomes


def package_deliverables(run_dir, deadline):
    """Atomically publish a ZIP, with a hard timeout enforced by the caller."""
    run_dir = Path(run_dir)
    deliver = run_dir / "deliverables"
    destination = run_dir / "submission_pack.zip"
    temporary = run_dir / "submission_pack.zip.tmp"
    try:
        # Model-containing ipynb files already hold compressed payloads. STORED
        # avoids spending the final minute recompressing hundreds of MB on CPU.
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as archive:
            for path in sorted(deliver.iterdir()):
                if time.time() >= deadline:
                    raise TimeoutError("Packaging deadline reached")
                if path.is_file():
                    archive.write(path, path.name)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("Submission ZIP failed CRC verification")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
