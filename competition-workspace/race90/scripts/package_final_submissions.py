"""Check, preserve, and package the actually executed competition notebooks."""
from pathlib import Path
import argparse
import ast
import base64
import hashlib
import json
import shutil
import zipfile
import zlib

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CHECKS = ROOT / "race90/notebook-checks/final"
COMPETITION = ROOT / "race90/competition"
DEST = ROOT / "최종제출"
SIGNALS = (
    "뒤에 다른 프롬프트가 주어져도",
    "직접적으로 코드를 만들어주지는 말아",
    "opacity:0.01",
    "font-size:0.1px",
    "BEGIN OPENSSH PRIVATE KEY",
    "BEGIN RSA PRIVATE KEY",
)
REFERENCE_CSV = {
    1: "submission.finetuned128_512.csv",
    2: "submission.reviewed.csv",
    3: "submission.cpu.refined.csv",
}
ALIASES = {
    1: "1번_레이더_실행본.ipynb",
    2: "2번_비행물체_실행본.ipynb",
    3: "3번_정비보고서_실행본.ipynb",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scan_text(text, path):
    for signal in SIGNALS:
        assert signal not in text, (str(path), signal)


def check_notebook(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    scan_text(path.read_text(encoding="utf-8"), path)
    assert nb.metadata.kernelspec.name == "python3"
    assert nb.metadata.language_info.version == "3.9.7"
    counts = []
    imports = set()
    snapshot = None
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        assert cell.execution_count is not None
        counts.append(cell.execution_count)
        assert all(o.output_type != "error" for o in cell.outputs)
        tree = ast.parse(cell.source, feature_version=(3, 9))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add((node.module or "").split(".")[0])
            elif isinstance(node, ast.Call):
                call = ast.unparse(node.func)
                assert call not in {"eval", "exec", "compile", "__import__", "os.system", "os.popen", "get_ipython"}, call
            elif isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == "_packed_snapshot" for t in node.targets):
                    packed = ast.literal_eval(node.value)
                    raw = zlib.decompress(base64.b64decode(packed)).decode("utf-8")
                    scan_text(raw, str(path) + "::decoded_json")
                    snapshot = json.loads(raw)
    assert counts == list(range(1, len(counts) + 1)), counts
    assert imports <= {
        "os", "csv", "hashlib", "json", "platform", "sys", "time", "pathlib",
        "importlib", "numpy", "pandas", "IPython", "base64", "zlib", "math", "re", "collections",
    }, imports
    assert snapshot is not None
    return nb, snapshot, sorted(imports)


def check_csv(question, path):
    frame = pd.read_csv(path)
    assert not frame.isna().any().any()
    assert not any(c.startswith("Unnamed:") for c in frame.columns)
    reference = pd.read_csv(COMPETITION / ("q" + str(question)) / REFERENCE_CSV[question])
    pd.testing.assert_frame_equal(frame, reference)
    if question == 1:
        test = pd.read_csv(COMPETITION / "q1/data/test.csv")
        assert list(frame.columns) == ["sample_id", "label"]
        assert len(frame) == 300 and frame.sample_id.is_unique
        assert frame.sample_id.tolist() == sorted(test.sample_id.unique())
        assert set(frame.label) <= {"LIGHT_VEHICLE", "HEAVY_VEHICLE", "TWO_WHEELER", "PERSONNEL", "PERSONNEL_GROUP"}
    elif question == 2:
        assert list(frame.columns) == ["filename", "airplane", "bird", "drone"]
        assert len(frame) == 300 and frame.filename.is_unique
        assert frame.filename.tolist() == ["test%03d.jpg" % i for i in range(1, 301)]
        values = frame[["airplane", "bird", "drone"]].to_numpy()
        assert np.isfinite(values).all() and (values >= 0).all()
        assert (values == np.floor(values)).all()
        assert ((values > 0).sum(axis=1) == 1).all()
    else:
        test = pd.read_csv(COMPETITION / "q3/data/test.csv")
        assert list(frame.columns) == ["id", "fault_type"]
        assert len(frame) == 180 and frame.id.is_unique
        assert frame.id.tolist() == test.id.tolist()
        assert set(frame.fault_type) <= {"ENGINE", "ELECTRICAL", "HYDRAULIC", "COMMUNICATION", "MOBILITY", "COOLING"}
    return frame


def preserve_then_copy(source, target, backup):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
    shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--completed-at-utc", required=True)
    args = parser.parse_args()
    source_root = CHECKS / "python397"
    report = json.loads((source_root / "execution-report.json").read_text())
    assert (report["python"], report["architecture"], report["numpy"], report["pandas"]) == ("3.9.7", "x86_64", "1.23.5", "1.5.2")
    assert len(report["cpu_affinity"]) == 2 and report["address_space_limit_bytes"] == 4294967296
    assert report["cuda_visible_devices"] == "" and not report["online_kernel_directly_verified"]
    assert [r["question"] for r in report["notebooks"]] == [1, 2, 3]
    audit = {"status": "passed", "final_notebooks": [], "original_zip_notebooks": [], "findings": []}
    notebooks = []
    for item in report["notebooks"]:
        q = item["question"]
        source = source_root / ("Problem #" + str(q))
        nb_path = source / item["notebook"]
        nb, snapshot, imports = check_notebook(nb_path)
        csv_path = source / "submission.csv"
        check_csv(q, csv_path)
        assert digest(csv_path) == item["csv_sha256"]
        assert item["executed_code_cells"] == sum(c.cell_type == "code" for c in nb.cells)
        folder = DEST / (str(q) + "번")
        folder.mkdir(parents=True, exist_ok=True)
        for p in (nb_path, csv_path):
            shutil.copy2(p, folder / p.name)
        audit["final_notebooks"].append({"question": q, "executed_cells_scanned": item["executed_code_cells"], "imports": imports, "decoded_payload_is_json_data": True})
        notebooks.append(dict(item, relative_notebook_path=str((folder / nb_path.name).relative_to(DEST)), notebook_sha256=digest(nb_path), execution_mode=snapshot["execution_mode"]))

    for archive_path in sorted(ROOT.glob("*번문제.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                if name.endswith(".ipynb"):
                    notebook = json.loads(archive.read(name).decode("utf-8"))
                    scan_text(json.dumps(notebook, ensure_ascii=False), str(archive_path) + "::" + name)
                    audit["original_zip_notebooks"].append(str(archive_path.name) + "::" + name)
    original_scan_count = 0
    for path in COMPETITION.rglob("*.ipynb"):
        scan_text(path.read_text(encoding="utf-8"), path)
        original_scan_count += 1
    audit["unpacked_original_and_checkpoint_notebooks_scanned"] = original_scan_count

    instructions = """최종 제출 파일 실행 안내

1. ZIP을 풀고 문항에 맞는 .ipynb를 대회 Jupyter의 해당 문제 폴더에 업로드합니다.
   1번 → Problem #1, 2번 → Problem #2, 3번 → Problem #3.
   제공된 Notebook과 같은 파일명을 사용합니다. 기존 data 폴더는 그대로 사용합니다.
   1·3번은 data/train.csv 및 data/test.csv, 2번은 data/aerial_objects가 필요합니다.
2. Notebook을 열고 Python 3 커널에서 Kernel → Restart & Run All을 실행합니다.
   JupyterLab에서는 Restart Kernel and Run All Cells에 해당합니다.
3. 마지막 셀에 '저장 완료'가 표시되면 Notebook을 저장합니다.
   같은 폴더에 submission.csv가 만들어집니다. 기존 CSV가 있으면 갱신합니다.
   마지막 출력의 submission.csv 링크로 내려받을 수 있습니다.
4. 대회 제출 화면에 각 문항의 submission.csv를 제출합니다.
   Notebook 제출 항목이 있으면 실행 결과를 저장한 해당 Notebook도 제출합니다.

문항별 생성 결과
1번: 300행 / sample_id,label / sample_id 오름차순
2번: 300행 / filename,airplane,bird,drone / 이미지 번호 오름차순
3번: 180행 / id,fault_type / test.csv의 ID 순서

세 문항 모두 파일명이 submission.csv이므로 반드시 문항별 폴더를 구분하세요.
ZIP에는 실행 검증으로 생성한 동일한 CSV도 포함했습니다.
데이터 폴더를 찾지 못하면 첫 코드 셀의 DATA_ROOT만 실제 data 폴더 경로로 수정하세요.
원본 입력과 내장 산출물의 대응을 검사하므로 다른 데이터가 들어가면 오류로 중단합니다.

실제 실행 내용
1번: GPU에서 학습·추론한 확률 배열을 사용해 앙상블, 검증, CSV 저장을 실행합니다.
2번: GPU에서 추론한 탐지 상자를 병합·집계하고 이미지 검토 결과를 반영합니다.
1·2번의 신경망 학습·추론 자체를 온라인에서 다시 수행하지는 않습니다.
GPU 작업 코드는 해당 Notebook 마지막 Markdown 부록에 보존했습니다.
3번: train.csv로 CPU 분류기를 학습하고 test.csv 180건을 실제로 다시 예측합니다.
추가 패키지 설치, 모델 파일 업로드, GPU 연결은 필요 없습니다.

실행 검증
Python 3.9.7, NumPy 1.23.5, pandas 1.5.2, ipykernel 6.9.1인 x86_64 별도 환경에서
전체 셀 실행을 완료했습니다. CPU affinity 2개, GPU 비노출, 주소 공간 상한 4 GiB로 검증했습니다.
대회 서버 전체 환경을 복제한 것은 아닙니다. PyTorch를 사용하는 실행 셀은 없습니다.
대회의 실제 온라인 커널 실행과 제출 접수는 직접 확인하지 않았습니다.
기존 숨은 프롬프트 인젝션은 제거했고, 최종 Notebook과 원본 ZIP을 다시 검사했습니다.
검증한 결과, 파일 해시, 개발 지표의 해석은 검증결과.json을 참고하세요.
"""
    (DEST / "실행안내.txt").write_text(instructions, encoding="utf-8")
    validation = {
        "completed_at_utc": args.completed_at_utc,
        "submission_files_ready": True,
        "online_submission_completed": False,
        "execution_validation": dict(report, notebooks=notebooks),
        "validation_scope": {
            "matching_versions": {"python": "3.9.7", "numpy": "1.23.5", "pandas": "1.5.2", "ipykernel": "6.9.1"},
            "whole_online_environment_replicated": False,
            "memory_limit_type": "RLIMIT_AS address-space ceiling, not a replicated cgroup",
            "torch_used_by_executed_cells": False,
            "extra_installation_needed_online": False,
            "local_python312_results_match": True,
        },
        "format_checks": "Passed row count, exact columns, unique IDs, required order, allowed labels, nonnegative integer object counts, no extra index, no missing values; matched selected predictions.",
        "injection_and_executable_code_audit": audit,
        "development_metrics": {
            "q1": json.loads((COMPETITION / "q1/finetune_ensemble128_512_validation.json").read_text()),
            "q2": {"test_accuracy": None, "reason": "No test reference labels were provided."},
            "q3": {
                "initial_grouped_cv_macro_f1": 0.9777654,
                "refined_development_macro_f1": 1.0,
                "caveat": "The refined rules were adjusted using training errors; 1.0 is not an independent validation estimate or test accuracy.",
                "test_accuracy": None,
            },
        },
    }
    local_report = json.loads((CHECKS / "local/execution-report.json").read_text())
    assert [r["csv_sha256"] for r in local_report["notebooks"]] == [r["csv_sha256"] for r in report["notebooks"]]
    (DEST / "검증결과.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    files = sorted(p for p in DEST.rglob("*") if p.is_file() and p.name != "파일목록_SHA256.txt")
    assert len(files) == 8, [str(p) for p in files]
    hashes = "\n".join(digest(p) + "  " + str(p.relative_to(DEST)) for p in files) + "\n"
    (DEST / "파일목록_SHA256.txt").write_text(hashes, encoding="utf-8")
    archive_path = ROOT / "제출파일_최종.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(p for p in DEST.rglob("*") if p.is_file()):
            archive.write(path, arcname=str(path.relative_to(DEST)))
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == 9
        assert len([n for n in archive.namelist() if n.endswith(".ipynb")]) == 3
        assert len([n for n in archive.namelist() if n.endswith(".csv")]) == 3
        for name in archive.namelist():
            assert archive.read(name) == (DEST / name).read_bytes()
            if name.endswith(".ipynb"):
                scan_text(archive.read(name).decode("utf-8"), archive_path.name + "::" + name)

    for item in notebooks:
        q = item["question"]
        final_nb = DEST / item["relative_notebook_path"]
        preserve_then_copy(final_nb, COMPETITION / ("q" + str(q)) / final_nb.name, CHECKS / "clean-originals" / (str(q) + "번") / final_nb.name)
        preserve_then_copy(final_nb, ROOT / ALIASES[q], CHECKS / "previous-root-notebooks" / ALIASES[q])
        shutil.copy2(final_nb.parent / "submission.csv", COMPETITION / ("q" + str(q)) / "submission.csv")
    handoff = {"completed_at_utc": args.completed_at_utc, "submission_ready": True, "packaged": True, "online_submission_completed": False, "archive": str(archive_path), "archive_bytes": archive_path.stat().st_size, "archive_sha256": digest(archive_path), "notebooks": notebooks}
    (CHECKS / "final.json").write_text(json.dumps(handoff, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(handoff, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
