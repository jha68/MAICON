from pathlib import Path
import base64
import io
import json
import os
import tempfile
import time
import zipfile
import nbformat
from .bundle import source_zip, source_files
from .config import resolve_data


def export_notebook(cfg, task, candidate, output, deadline):
    candidate, output = Path(candidate), Path(output)
    original = resolve_data(cfg, task["notebook"]) if task.get("notebook") else None
    nb = nbformat.read(original, as_version=4) if original and original.exists() else nbformat.v4.new_notebook()
    first_appended = len(nb.cells)
    source = base64.b64encode(source_zip(cfg)).decode()
    compact = cfg["notebooks"].get("compact_export", False)
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED) as z:
        for p in candidate.iterdir():
            if compact and p.name not in {"manifest.json", "submission.csv"}:
                continue
            if p.is_file() and (cfg["notebooks"].get("embed_models", True) or p.suffix not in {".pt", ".joblib", ".cbm", ".zip"}):
                z.write(p, p.name)
        audit = candidate.parent.parent / "audit.json"
        if audit.exists():
            z.write(audit, "audit.json")
        for parent in task.get("depends_on", []):
            parent_dir = Path(cfg["dependency_outputs"][parent])
            for p in parent_dir.iterdir():
                if p.is_file() and (not compact or p.name == "manifest.json"):
                    z.write(p, "dependencies/" + parent + "/" + p.name)
    max_bytes = int(cfg["notebooks"].get("max_embedded_mb", 200)) * 1024 * 1024
    if memory.tell() > max_bytes:
        raise ValueError(f"Notebook artifact bundle exceeds {max_bytes // 1024**2} MB; increase max_embedded_mb or disable embed_models")
    artifacts = base64.b64encode(memory.getvalue()).decode()
    clean_cfg = {"seed": cfg["seed"], "tasks": [task], "data_root": ".", "project_root": ".",
                 "runpod": {"enabled": False}, "total_minutes": cfg["total_minutes"],
                 "dependency_outputs": {name: "dependencies/" + name for name in task.get("depends_on", [])}}
    summary = json.loads((candidate / "manifest.json").read_text())
    nb.cells.append(nbformat.v4.new_markdown_cell(
        f"## {task['name']} 풀이 코드와 결과\n\n"
        "아래에는 실제 학습에 사용한 Python 소스, 선택된 모델/예측 파일, 검증 결과가 포함되어 있습니다. "
        "기본 실행은 학습 과정에서 저장한 예측을 복원하고 제출 형식을 다시 검증합니다. "
        "RECOMPUTE=True로 바꾸면 저장 모델로 CPU 추론을 다시 실행합니다. "
        "데이터 경로는 DATA_ROOT에서 조정하세요. 원본 문제 셀은 보존했습니다.\n\n"
        f"선택된 단계: `{summary['stage']}`, 검증 지표: `{summary['metric']}`, 점수: `{summary['score']}`. "
        "점수가 None이면 학습 완료 전의 기본값입니다."
        + ("\n\n이 경량본은 전체 풀이 소스와 실제 예측·검증 출력을 포함하지만 모델 파일은 포함하지 않습니다. "
           "저장 예측 복원 모드이며, 학습이나 새 추론을 이 Notebook에서 했다고 표시하지 않습니다. "
           "재학습/재추론에는 별도 모델과 의존 산출물이 필요할 수 있습니다." if compact else "")
    ))
    # Visible source accompanies the executable bundle; no external repository
    # or secret is needed to inspect the complete implementation.
    relevant = {"config.py", "io.py", "data.py", "models.py", "worker.py", "vision.py", "neural.py", "tabular_gpu.py"}
    for name, path in source_files(cfg).items():
        if path.name in relevant or not name.startswith("race90/"):
            nb.cells.append(nbformat.v4.new_markdown_cell(f"### Python 소스: {name}\n\n```python\n{path.read_text()}\n```"))
    bootstrap = f'''from pathlib import Path
import base64, io, json, os, sys, tempfile, zipfile

# 온라인 Notebook에서 실제 데이터가 있는 경로로 필요하면 수정하세요.
DATA_ROOT = Path(os.environ.get("RACE90_DATA_ROOT", str(Path.cwd()))).resolve()
RECOMPUTE = {cfg['notebooks'].get('mode', 'restore') == 'infer'}
import importlib.util, subprocess
missing = [package for module, package in {{"numpy": "numpy", "pandas": "pandas", "yaml": "PyYAML"}}.items()
           if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
WORK = Path(tempfile.mkdtemp(prefix="race90-notebook-"))
SOURCE_BUNDLE = "{source}"
ARTIFACT_BUNDLE = "{artifacts}"
for payload, dest in [(SOURCE_BUNDLE, WORK / "source"), (ARTIFACT_BUNDLE, WORK / "model")]:
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
        for name in archive.namelist():
            if not (dest / name).resolve().is_relative_to(dest.resolve()):
                raise ValueError("Unsafe bundled path")
        archive.extractall(dest)
sys.path.insert(0, str(WORK / "source"))
for name in list(sys.modules):
    if name == "race90" or name.startswith("race90."):
        del sys.modules[name]
CFG = json.loads({json.dumps(json.dumps(clean_cfg, ensure_ascii=False))})
CFG["data_root"] = str(DATA_ROOT)
CFG["dependency_outputs"] = {{name: str(WORK / "model" / path) for name, path in CFG["dependency_outputs"].items()}}
TASK = CFG["tasks"][0]
print("문항:", TASK["name"], "데이터:", DATA_ROOT)
'''
    nb.cells.append(nbformat.v4.new_code_cell(bootstrap))
    nb.cells.append(nbformat.v4.new_code_cell('''# 실제 학습 로직은 위 소스의 train_cpu/train_resnet/train_yolo에 포함되어 있습니다.
# 원하면 CPU 기본 모델 학습을 이 Notebook 안에서 다시 실행할 수 있습니다.
RETRAIN_CPU = False
if RETRAIN_CPU:
    import time
    from race90.process import run_command
    from race90.io import atomic_json
    deadline = time.time() + 600
    payload = WORK / "retrain.json"
    atomic_json(payload, {"cfg": CFG, "task": TASK, "output": str(WORK / "retrained"),
                          "stage": "baseline", "deadline": deadline, "threads": 1, "device": "cpu"})
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORK / "source") + os.pathsep + env.get("PYTHONPATH", "")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = "1"
    try:
        run_command([sys.executable, "-m", "race90", "_worker", str(payload)], deadline, env=env)
    except (TimeoutError, RuntimeError) as error:
        print("완료된 후보로 복구합니다:", error)
    from race90.selection import select_best
    selected, _ = select_best(CFG, TASK, WORK / "retrained")
    MODEL_DIR = selected
else:
    MODEL_DIR = WORK / "model"
'''))
    nb.cells.append(nbformat.v4.new_code_cell('''import pandas as pd
from race90.io import read_csv, read_saved_submission, validate_submission, save_csv, sha256
if RECOMPUTE:
    from race90.models import predict_checkpoint
    submission = predict_checkpoint(CFG, TASK, MODEL_DIR, device="cpu")
    print("저장 모델로 CPU 추론을 다시 실행했습니다.")
else:
    # 이 예측은 하네스의 학습/추론 단계에서 실제 생성된 결과입니다.
    template = read_csv(CFG, TASK["sample_submission"], TASK)
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    submission = read_saved_submission(MODEL_DIR / "submission.csv", template, TASK, manifest)
    print("학습 단계에서 저장된 예측을 복원했습니다.")
template = read_csv(CFG, TASK["sample_submission"], TASK)
validate_submission(submission, template, TASK)
FINAL_CSV = Path(os.environ.get("RACE90_OUTPUT_DIR", str(Path.cwd()))) / (TASK["name"] + "_submission.csv")
save_csv(FINAL_CSV, submission)
model_report = json.loads((MODEL_DIR / "manifest.json").read_text())
print("검증 결과:", json.dumps(model_report, ensure_ascii=False, indent=2))
print("제출 파일:", FINAL_CSV, "행:", len(submission), "SHA256:", sha256(FINAL_CSV))
display(submission.head())
'''))
    output.parent.mkdir(parents=True, exist_ok=True)
    nb.metadata["race90"] = {"task": task["name"], "candidate": candidate.name,
                             "mode": cfg["notebooks"].get("mode", "restore"),
                             "compact": compact, "executed_solution_cells": 0}
    # Cell IDs were introduced in nbformat 4.5. Keep older provided notebooks
    # valid without changing any original question cells or outputs.
    if nb.nbformat_minor < 5:
        for cell in nb.cells[first_appended:]:
            cell.pop("id", None)
    nbformat.write(nb, output)
    if cfg["notebooks"].get("execute", True):
        # Run only appended solution cells: untouched original question cells
        # can contain incomplete starter code and must not block finalization.
        from nbclient import NotebookClient
        from jupyter_client import AsyncKernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        remaining = int(deadline - time.time())
        if remaining < 5:
            raise TimeoutError("No time for Notebook execution")
        with tempfile.TemporaryDirectory(prefix="race90-kernel-") as kernel_dir:
            spec = Path(kernel_dir) / "race90"
            spec.mkdir()
            import sys
            (spec / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                                                           "display_name": "race90", "language": "python"}))
            manager = AsyncKernelManager(kernel_name="race90", kernel_spec_manager=KernelSpecManager(kernel_dirs=[kernel_dir]))
            solution = nbformat.v4.new_notebook(cells=nb.cells[first_appended:], nbformat_minor=nb.nbformat_minor)
            client = NotebookClient(solution, km=manager, timeout=remaining,
                                    timeout_func=lambda cell: max(1, int(deadline - time.time() - 2)),
                                    resources={"metadata": {"path": cfg["data_root"]}})
            try:
                client.execute()
            finally:
                nb.cells[first_appended:] = solution.cells
                nb.metadata["race90"]["executed_solution_cells"] = sum(
                    c.cell_type == "code" and c.execution_count is not None for c in solution.cells)
                nbformat.write(nb, output)
    return output
