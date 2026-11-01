"""Build portable, self-contained notebooks from an immutable current-work snapshot.

Q1/Q2 reuse GPU prediction artifacts and recompute postprocessing. Q3 fits and
predicts with the current standard-library CPU classifier. Original work is read
only; each notebook is intended to run beside its competition data directory.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import textwrap
import zlib
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd


HARNESS = Path(__file__).resolve().parents[1]
WORKSPACE = HARNESS.parent
COMPETITION = HARNESS / "competition"
FILENAMES = {
    1: "1번_레이더_실행본.ipynb",
    2: "2번_비행물체_실행본.ipynb",
    3: "3번_정비보고서_실행본.ipynb",
}
SOURCE_BYTES = {}


def read(path):
    path = Path(path)
    if path not in SOURCE_BYTES:
        SOURCE_BYTES[path] = path.read_bytes()
    return SOURCE_BYTES[path]


def read_json(path):
    return json.loads(read(path))


def read_csv(path):
    return pd.read_csv(io.BytesIO(read(path)))


def text_digest(path):
    value = read(path).decode("utf-8-sig").replace("\r\n", "\n")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def md(source):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source):
    return nbformat.v4.new_code_cell(textwrap.dedent(source).strip())


def python_definitions(path, names):
    source = read(path).decode("utf-8")
    tree = ast.parse(source)
    return "\n\n".join(
        ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    )


def packed_cell(payload):
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded = base64.b64encode(zlib.compress(raw, 9)).decode("ascii")
    chunks = "\n".join("    " + repr(part) for part in textwrap.wrap(encoded, 100))
    cell = code(
        "# 이 데이터는 현재 작업에서 얻은 예측·검토 산출물입니다. 모델 가중치나 정답 데이터가 아닙니다.\n"
        "import base64\nimport zlib\n\n"
        "_packed_snapshot = (\n" + chunks + "\n)\n"
        "_snapshot_bytes = zlib.decompress(base64.b64decode(_packed_snapshot))\n"
        "assert hashlib.sha256(_snapshot_bytes).hexdigest() == " + repr(hashlib.sha256(raw).hexdigest()) + "\n"
        "snapshot = json.loads(_snapshot_bytes.decode('utf-8'))\n"
        "del _packed_snapshot, _snapshot_bytes\n"
        "print('산출물 스냅샷:', snapshot['created_at_utc'])\n"
        "print('실행 방식:', snapshot['execution_mode'])"
    )
    cell.metadata["jupyter"] = {"source_hidden": True}
    return cell


def setup_cell(question):
    return code(
        '''
        import os
        for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[_thread_variable] = "2"

        import csv
        import hashlib
        import json
        import platform
        import sys
        import time
        from pathlib import Path
        from importlib import metadata
        import numpy as np
        import pandas as pd
        from IPython.display import display, FileLink

        # 보통 수정할 필요가 없습니다. 자동 탐색 실패 시 data 폴더 경로를 직접 지정하세요.
        DATA_ROOT = None  # 예: Path("./data")
        OUTPUT_PATH = Path.cwd() / "submission.csv"
        started_at = time.perf_counter()
        print("Python:", platform.python_version(), "|", sys.executable)
        print("작업 폴더:", Path.cwd())
        print("NumPy:", np.__version__, "| pandas:", pd.__version__)
        for _package in ("torch", "scikit-learn"):
            try:
                print(_package + ":", metadata.version(_package))
            except metadata.PackageNotFoundError:
                print(_package + ": 미설치 (이 실행본에서 필수 아님)")
        print("추가 패키지 설치나 GPU 연결 없이 실행합니다.")
        '''
    )


def common_io_cell(question, image_task=False):
    common = '''
        def sha256_file(path):
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()

        def sha256_csv_text(path):
            # UTF-8 BOM과 줄바꿈 방식의 차이는 허용합니다.
            value = Path(path).read_bytes().decode("utf-8-sig").replace("\\r\\n", "\\n")
            return hashlib.sha256(value.encode("utf-8")).hexdigest()

        def data_candidates():
            if DATA_ROOT is not None:
                return [Path(DATA_ROOT).expanduser()]
            here = Path.cwd()
            return [here / "data", here,
                    here / ("Problem #" + str(QUESTION)) / "data",
                    here / ("q" + str(QUESTION)) / "data",
                    here / "race90" / "competition" / ("q" + str(QUESTION)) / "data"]

        def save_submission(frame):
            if frame.isna().any().any():
                raise ValueError("제출 결과에 결측값이 있습니다.")
            temporary = OUTPUT_PATH.with_name(OUTPUT_PATH.name + ".tmp")
            frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\\n")
            written = pd.read_csv(temporary, encoding="utf-8")
            pd.testing.assert_frame_equal(written, frame.reset_index(drop=True), check_dtype=False)
            temporary.replace(OUTPUT_PATH)
            print("저장 완료:", OUTPUT_PATH)
            print("행 수:", len(frame), "| 열:", list(frame.columns))
            print("SHA256:", sha256_file(OUTPUT_PATH))
            print("실행 시간: %.2f초" % (time.perf_counter() - started_at))
            display(frame.head(10))
            display(FileLink(str(OUTPUT_PATH.relative_to(Path.cwd()))))
    '''
    source = "QUESTION = " + str(question) + "\n\n" + textwrap.dedent(common).strip()
    if image_task:
        source += "\n\n" + textwrap.dedent('''
            data_root = None
            for candidate in data_candidates():
                folder = candidate / "aerial_objects"
                if folder.is_dir():
                    data_root = candidate.resolve()
                    break
            if data_root is None:
                raise FileNotFoundError("data/aerial_objects 폴더를 찾지 못했습니다. 이 노트북을 2번 문제 폴더에 넣거나 DATA_ROOT를 수정하세요.")
            image_root = data_root / "aerial_objects"
            image_paths = {p.name: p for p in image_root.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
            expected_files = set(snapshot["image_sha256"])
            if set(image_paths) != expected_files:
                raise ValueError("현재 이미지 목록과 내장 예측의 300개 파일 목록이 다릅니다. 데이터 폴더를 확인하세요.")
            changed = [name for name in sorted(expected_files) if sha256_file(image_paths[name]) != snapshot["image_sha256"][name]]
            if changed:
                raise ValueError("내장 예측을 계산한 이미지와 현재 파일 내용이 다릅니다: " + repr(changed[:10]))
            print("입력 확인:", image_root, "| 300개 이미지 이름·내용 일치")
        ''').strip()
    else:
        required_train = ["sample_id", "label", "t_ms"] if question == 1 else ["id", "report", "fault_type"]
        required_test = ["sample_id", "t_ms"] if question == 1 else ["id", "report"]
        source += "\n\nREQUIRED_TRAIN = " + repr(required_train) + "\nREQUIRED_TEST = " + repr(required_test) + "\n"
        source += textwrap.dedent('''
            data_root = None
            for candidate in data_candidates():
                if not all((candidate / name).is_file() for name in ("train.csv", "test.csv")):
                    continue
                train_columns = pd.read_csv(candidate / "train.csv", nrows=0, encoding="utf-8-sig").columns
                test_columns = pd.read_csv(candidate / "test.csv", nrows=0, encoding="utf-8-sig").columns
                if set(REQUIRED_TRAIN).issubset(train_columns) and set(REQUIRED_TEST).issubset(test_columns):
                    data_root = candidate.resolve()
                    break
            if data_root is None:
                raise FileNotFoundError("이 문항의 train.csv/test.csv를 찾지 못했습니다. 노트북을 해당 문제 폴더에 넣거나 DATA_ROOT를 수정하세요.")
            for name, expected_hash in snapshot["input_csv_sha256"].items():
                if sha256_csv_text(data_root / name) != expected_hash:
                    raise ValueError(name + " 내용이 현재 작업 스냅샷과 다릅니다. 올바른 문제 데이터인지 확인하세요.")
            train = pd.read_csv(data_root / "train.csv", encoding="utf-8-sig")
            test = pd.read_csv(data_root / "test.csv", encoding="utf-8-sig")
            print("입력 확인:", data_root, "| train", train.shape, "| test", test.shape)
        ''').strip()
    return code(source)


def original_markdown(question):
    path = next((COMPETITION / ("q" + str(question))).glob("*.ipynb"))
    original = read_json(path)
    cells = []
    for cell in original["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        source = "".join(cell["source"])
        if "패키지 설치 후 반드시 커널 재시작" in source or source.startswith("## 결과 저장"):
            continue
        # The original installer uninstalls core packages; the current execution
        # mode does not use it. Preserve the actual problem and output schemas.
        source = source.split("\n## 초기코드")[0].rstrip()
        cells.append(md(source))
    return cells


def header(question, timestamp, description):
    return md(f"""
    # {question}번 문제 — 현재 작업 실행본

    스냅샷 시각: **{timestamp}**. 최종 제출 확정본이 아니라, 이 시점의 작업 결과를 온라인 Jupyter에서 확인하기 위한 실행본입니다.

    이 파일을 **해당 문항의 원본 Notebook과 같은 폴더**에 업로드하고 **Kernel → Restart & Run All**을 실행하세요.
    기존 `data/` 폴더를 사용하며, 마지막 셀에서 현재 작업 폴더의 **`submission.csv`**를 생성합니다.
    기존 `submission.csv`가 있으면 이 결과로 갱신합니다. 문항마다 별도 폴더에서 실행하세요.
    데이터 위치가 다르면 첫 코드 셀의 `DATA_ROOT`만 수정하세요. 추가 산출물 파일 업로드나 패키지 설치는 필요 없습니다.

    {description}

    실행 대상은 사용자 제공 온라인 환경(Python 3.9.7, CPU 2개, RAM 4 GiB)입니다.
    **온라인 동일 환경 검증 미완료:** 저장된 출력은 로컬 Python 3.12 실행 결과입니다.
    Python 3.9 문법 검사와 로컬 전체 실행은 실제 온라인 커널 실행 확인과 구분합니다.
    아래 원래 문제 지문 뒤에 실행 셀이 있습니다.
    """)


def source_appendix(entries):
    sections = ["## 참고: 현재 GPU 작업 소스", "아래 소스는 이미 수행한 GPU 계산의 기록입니다. Markdown이므로 Run All에서 실행하지 않습니다. 온라인에서 모델을 재학습·재추론한 결과로 해석하지 않습니다."]
    for name, source in entries:
        sections.append("<details><summary>" + name + "</summary>\n\n```python\n" + source.rstrip() + "\n```\n\n</details>")
    return md("\n\n".join(sections))


def make_q1(timestamp):
    root = COMPETITION / "q1"
    validation = read_json(root / "finetune_ensemble_validation.json")
    names = validation.get("members", validation.get("models"))
    if not names:
        raise ValueError("Q1 validation metadata does not identify ensemble members")
    members = []
    for name in names:
        with np.load(io.BytesIO(read(root / (name + ".npz"))), allow_pickle=True) as archive:
            members.append({key: archive[key].tolist() for key in ("classes", "oof", "test")})
    features = read_csv(root / "features_physics_test.csv")
    labels = read_csv(root / "labels.csv")
    expected = read_csv(root / "submission.finetuned.csv")
    payload = {
        "created_at_utc": timestamp,
        "execution_mode": "GPU 저장 확률을 읽어 앙상블·검증·CSV 저장 (모델 재추론 아님)",
        "input_csv_sha256": {name: text_digest(root / "data" / name) for name in ("train.csv", "test.csv")},
        "test_ids": features.iloc[:, 0].tolist(),
        "train_ids": labels.iloc[:, 0].tolist(),
        "members": members,
        "weights": validation["weights"],
        "validation": validation,
        "reference_predictions": expected.to_dict("records"),
    }
    cells = [header(1, timestamp, f"**실행 내용:** GPU에서 학습·추론을 마친 TabICL 후보 {len(names)}개의 확률을 저장된 가중치 {validation['weights']}로 합산합니다. 현재 `submission.finetuned.csv`를 재현하며, 입력 데이터와 ID, 확률, 레이블, 정렬을 확인합니다. GPU 학습과 모델 추론 자체는 다시 실행하지 않습니다.")]
    cells += original_markdown(1)
    cells += [md("## 실행 1. 환경과 경로"), setup_cell(1), md("## 실행 2. 내장 예측 산출물 및 원본 데이터 확인"), packed_cell(payload), common_io_cell(1)]
    cells += [md("## 실행 3. 확률 앙상블과 개발 검증\n\n개발 OOF 정확도는 훈련 데이터에 대한 교차검증 지표이며 실제 테스트 점수가 아닙니다. 후보 선택에도 사용한 값이므로 최종 일반화 성능을 보장하지 않습니다."), code('''
        allowed = {"LIGHT_VEHICLE", "HEAVY_VEHICLE", "TWO_WHEELER", "PERSONNEL", "PERSONNEL_GROUP"}
        classes = np.asarray(snapshot["members"][0]["classes"])
        if len(classes) != 5 or set(classes) != allowed:
            raise ValueError("클래스 목록 오류")
        ids = snapshot["test_ids"]
        if len(ids) != 300 or len(set(ids)) != 300 or sorted(test["sample_id"].unique()) != sorted(ids):
            raise ValueError("테스트 sample_id 불일치")
        weights = np.asarray(snapshot["weights"], dtype=float)
        if len(weights) != len(snapshot["members"]) or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("앙상블 가중치 오류")
        probabilities = np.zeros((300, 5), dtype=float)
        oof_probabilities = np.zeros((len(snapshot["train_ids"]), 5), dtype=float)
        for weight, member in zip(weights, snapshot["members"]):
            if member["classes"] != classes.tolist():
                raise ValueError("모델별 클래스 순서 불일치")
            for key, accumulator in (("test", probabilities), ("oof", oof_probabilities)):
                values = np.asarray(member[key], dtype=float)
                if values.shape != accumulator.shape or not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError("예측 확률 배열 오류: " + key)
                if not np.allclose(values.sum(axis=1), 1.0, atol=1e-4):
                    raise ValueError("확률 합 오류: " + key)
                accumulator += weight * values
        df = pd.DataFrame({"sample_id": ids, "label": classes[probabilities.argmax(axis=1)]})
        df = df.sort_values("sample_id").reset_index(drop=True)
        expected = pd.DataFrame(snapshot["reference_predictions"]).sort_values("sample_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(df, expected)
        if train.groupby("sample_id")["label"].nunique().ne(1).any():
            raise ValueError("같은 훈련 sample_id에 여러 레이블이 있습니다.")
        truth = train.groupby("sample_id")["label"].first().reindex(snapshot["train_ids"])
        if truth.isna().any():
            raise ValueError("OOF 검증용 훈련 ID 불일치")
        oof_accuracy = float(np.mean(truth.to_numpy() == classes[oof_probabilities.argmax(axis=1)]))
        if not np.isclose(oof_accuracy, snapshot["validation"]["accuracy"]):
            raise ValueError("저장된 개발 검증 지표와 불일치")
        print("현재 예측 300건과 모두 일치")
        print("개발 OOF Accuracy: %.4f (테스트 정답 점수 아님)" % oof_accuracy)
        display(df["label"].value_counts().rename("예측 수"))
    '''), md("## 실행 4. 제출 CSV 저장"), code('''
        assert list(df.columns) == ["sample_id", "label"]
        assert len(df) == 300 and df["sample_id"].is_unique
        assert df["sample_id"].tolist() == sorted(test["sample_id"].unique())
        assert set(df["label"]).issubset(allowed)
        save_submission(df)
    ''')]
    cells.append(source_appendix([
        ("sample_id별 기본 특징 추출", python_definitions(HARNESS / "custom_adapters/competition.py", {"radar_features"})),
        ("추가 특징 추출", read(HARNESS / "custom_adapters/radar_improved.py").decode()),
        ("교차검증 및 TabICL 미세조정", read(HARNESS / "scripts/radar_finetune.py").decode()),
    ]))
    return cells, payload


def make_q2(timestamp):
    root = COMPETITION / "q2"
    source_names = {
        "clip": "clip_predictions.json",
        "drone_s": "drone_s_stable_predictions.json",
        "drone_m": "drone_m_stable_predictions.json",
        "base_s": "yolov8s_base_predictions.json",
        "base_x": "yolov8x_base_predictions.json",
        "tta": "tta_base_predictions.json",
    }
    payload = {
        "created_at_utc": timestamp,
        "execution_mode": "GPU 저장 탐지 결과의 병합·수량 집계·검토 반영 (이미지 모델 재추론 아님)",
        "image_sha256": {p.name: hashlib.sha256(read(p)).hexdigest() for p in sorted((root / "data/aerial_objects").glob("*.jpg"))},
        "predictions": {key: read_json(root / name) for key, name in source_names.items()},
        "review": read_json(root / "visual_review.json"),
        "reference_auto": read_csv(root / "submission.auto.csv").to_dict("records"),
        "reference_reviewed": read_csv(root / "submission.reviewed.csv").to_dict("records"),
    }
    cells = [header(2, timestamp, "**실행 내용:** GPU에서 얻은 YOLOv8 비행기·새 탐지, 전이학습 드론 탐지, CLIP 분류, 증강 추론 결과를 내장했습니다. 이 결과를 다시 병합·집계하고 현재 시각 검토 내용을 적용해 `submission.reviewed.csv`를 재현합니다. 이미지 신경망 자체는 다시 실행하지 않습니다. 원본의 패키지 삭제·설치 셀은 이 실행 방식에 필요하지 않아 제외했습니다.")]
    cells += original_markdown(2)
    cells += [md("## 실행 1. 환경과 경로"), setup_cell(2), md("## 실행 2. 내장 탐지 결과 및 300개 이미지 확인"), packed_cell(payload), common_io_cell(2, image_task=True)]
    functions = python_definitions(HARNESS / "scripts/vision_merge.py", {"overlap", "merge"})
    cells += [md("## 실행 3. 탐지 병합과 자동 수량 계산\n\n같은 객체에 겹치는 탐지 상자를 합치고 신뢰도 및 여러 모델의 일치 여부로 수량을 산출합니다."), code(functions), code('''
        sources = {name: {row["filename"]: row for row in rows} for name, rows in snapshot["predictions"].items()}
        for name, records in sources.items():
            if set(records) != set(image_paths):
                raise ValueError("모델별 이미지 목록 불일치: " + name)
        rows = []
        for filename in sorted(image_paths, key=lambda value: int(Path(value).stem[4:])):
            category = sources["clip"][filename]["category"]
            if category not in ("airplane", "bird", "drone"):
                raise ValueError("분류 결과 오류")
            if category == "drone":
                selected_sources = [("ds", sources["drone_s"][filename]), ("dm", sources["drone_m"][filename])]
                class_id = 0
            else:
                class_id = 4 if category == "airplane" else 14
                selected_sources = [("bs", sources["base_s"][filename]), ("bx", sources["base_x"][filename])]
                selected_sources.extend(("r" + str(view["rotation"]), view)
                                        for view in sources["tta"][filename]["views"]
                                        if category == "bird" or view["rotation"] == 0)
            boxes = []
            for name, result in selected_sources:
                if not len(result["boxes"]) == len(result["conf"]) == len(result["cls"]):
                    raise ValueError("탐지 배열 길이 불일치")
                for box, confidence, label in zip(result["boxes"], result["conf"], result["cls"]):
                    if label == class_id and confidence >= 0.15:
                        boxes.append((box, confidence, name))
            merged = merge(boxes)
            accepted = [box for box in merged if box["conf"] >= 0.55 or
                        (box["conf"] >= 0.3 and len(set(box["models"])) >= 2)]
            if not accepted and merged:
                accepted = [merged[0]]
            row = {"filename": filename, "airplane": 0, "bird": 0, "drone": 0}
            row[category] = max(1, len(accepted))
            rows.append(row)
        df_auto = pd.DataFrame(rows, columns=["filename", "airplane", "bird", "drone"])
        pd.testing.assert_frame_equal(df_auto, pd.DataFrame(snapshot["reference_auto"]), check_dtype=False)
        print("GPU 탐지 결과의 자동 집계 300건 재현 완료")
    '''), md("## 실행 4. 현재 검토 내용 반영\n\n아래 수정은 제공 이미지를 직접 확인하여 정한 예측입니다. 테스트 정답을 확인한 값이 아닙니다. `test094.jpg`는 현재 9개를 선택했으며 8개 가능성도 검토 기록에 남아 있습니다."), code('''
        df = df_auto.copy().set_index("filename")
        for filename, change in snapshot["review"]["confirmed_overrides"].items():
            if filename not in df.index:
                raise ValueError("검토 대상 이미지가 없습니다: " + filename)
            active = [category for category in ("airplane", "bird", "drone") if df.loc[filename, category] > 0]
            if len(active) != 1 or not isinstance(change["count"], int) or change["count"] < 1:
                raise ValueError("검토 수량 오류")
            df.loc[filename, active[0]] = change["count"]
        df = df.reset_index()
        df[["airplane", "bird", "drone"]] = df[["airplane", "bird", "drone"]].astype(int)
        expected = pd.DataFrame(snapshot["reference_reviewed"])
        pd.testing.assert_frame_equal(df, expected, check_dtype=False)
        print("현재 검토 예측 300건과 모두 일치")
        print("직접 검토 기록:", len(snapshot["review"]["confirmed_overrides"]), "개")
        print("불확실성 기록:", json.dumps(snapshot["review"].get("uncertain_reviewed_counts", {}), ensure_ascii=False))
        display(df[["airplane", "bird", "drone"]].sum().rename("예측 객체 수"))
    '''), md("## 실행 5. 제출 CSV 저장"), code('''
        count_columns = ["airplane", "bird", "drone"]
        assert list(df.columns) == ["filename"] + count_columns
        assert len(df) == 300 and df["filename"].is_unique
        assert df["filename"].tolist() == sorted(image_paths, key=lambda value: int(Path(value).stem[4:]))
        assert (df[count_columns].to_numpy() >= 0).all()
        assert ((df[count_columns] > 0).sum(axis=1) == 1).all()
        save_submission(df)
    ''')]
    cells.append(source_appendix([(name, read(HARNESS / "scripts" / name).decode()) for name in ("vision_train.py", "vision_base.py", "vision_clip.py", "vision_tta.py")]))
    return cells, payload


def make_q3(timestamp):
    root = COMPETITION / "q3"
    payload = {
        "created_at_utc": timestamp,
        "execution_mode": "현재 CPU 분류기를 train.csv로 학습한 뒤 test.csv 전체를 실제 재예측",
        "input_csv_sha256": {name: text_digest(root / "data" / name) for name in ("train.csv", "test.csv")},
        "reference_predictions": read_csv(root / "submission.cpu.refined.csv").to_dict("records"),
        "development_validation": read_json(root / "cpu_validation_refined.json"),
    }
    cells = [header(3, timestamp, "**실행 내용:** 현재 CPU 분류기 소스를 모두 포함했습니다. `train.csv`에서 주 증상별 문자 TF-IDF 검색기를 학습하고, 주 증상 규칙과 검색 결과로 `test.csv` 180건을 실제 재예측합니다. 내장된 기존 예측은 일치 여부를 확인하는 용도로만 사용합니다.")]
    cells += original_markdown(3)
    cells += [md("## 실행 1. 환경과 경로"), setup_cell(3), md("## 실행 2. 원본 데이터 및 기존 결과 확인"), packed_cell(payload), common_io_cell(3)]
    functions = "import math\nimport re\nfrom collections import Counter\n\n"
    functions += python_definitions(HARNESS / "custom_adapters/text_improved.py", {"core_report", "semantic_rule"})
    functions += "\n\n" + python_definitions(HARNESS / "custom_adapters/text_cpu.py", {"grams", "ReportClassifier"})
    cells += [md("## 실행 3. 주 증상 추출과 CPU 분류기\n\n다른 계통의 정상 점검 결과를 분리하여 주 증상을 추출합니다. 규칙으로 판별되지 않은 문장은 훈련 데이터의 주 증상과 문자 2~4-gram TF-IDF 유사도를 비교합니다."), code(functions), md("## 실행 4. 학습 및 180건 예측"), code('''
        allowed = {"ENGINE", "ELECTRICAL", "HYDRAULIC", "COMMUNICATION", "MOBILITY", "COOLING"}
        if len(train) != 540 or len(test) != 180:
            raise ValueError("훈련/테스트 행 수 불일치")
        if not train["id"].is_unique or not test["id"].is_unique:
            raise ValueError("입력 ID가 중복됩니다.")
        if train[["report", "fault_type"]].isna().any().any() or test["report"].isna().any():
            raise ValueError("입력 텍스트 또는 레이블 결측")
        if set(train["fault_type"]) != allowed:
            raise ValueError("훈련 레이블 오류")
        model = ReportClassifier().fit(train["report"].tolist(), train["fault_type"].tolist())
        predictions = model.predict(test["report"].tolist())
        df = pd.DataFrame({"id": test["id"].tolist(), "fault_type": predictions})
        expected = pd.DataFrame(snapshot["reference_predictions"])
        pd.testing.assert_frame_equal(df, expected)
        print("학습 완료: 훈련 보고서", len(train), "개 / 고유 주 증상", len(model.cores), "개")
        print("실제 CPU 재예측 180건이 현재 작업 결과와 모두 일치")
        print("기존 개발 점검 Macro F1:", snapshot["development_validation"]["macro_f1"])
        print("위 지표는 훈련 오류를 보고 규칙을 개선한 개발 점검 값이며, 독립 검증 또는 테스트 점수가 아닙니다.")
        display(df["fault_type"].value_counts().rename("예측 수"))
    '''), md("## 실행 5. 제출 CSV 저장"), code('''
        assert list(df.columns) == ["id", "fault_type"]
        assert len(df) == 180 and df["id"].is_unique
        assert df["id"].tolist() == test["id"].tolist()
        assert set(df["fault_type"]).issubset(allowed)
        save_submission(df)
    ''')]
    return cells, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-utc", required=True, help="UTC time obtained from the session clock")
    args = parser.parse_args()
    manifest = {"created_at_utc": args.snapshot_utc, "online_exact_environment_verified": False, "notebooks": []}
    for question, builder in ((1, make_q1), (2, make_q2), (3, make_q3)):
        cells, payload = builder(args.snapshot_utc)
        notebook = nbformat.v4.new_notebook(cells=cells)
        notebook.metadata.update({
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9.7"},
            "race90_snapshot": {"created_at_utc": args.snapshot_utc, "question": question, "execution_mode": payload["execution_mode"], "online_exact_environment_verified": False},
        })
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                ast.parse(cell.source, filename=f"q{question}_cell{i}", feature_version=(3, 9))
        nbformat.validate(notebook)
        destination = WORKSPACE / FILENAMES[question]
        nbformat.write(notebook, destination)
        manifest["notebooks"].append({"question": question, "path": str(destination), "execution_mode": payload["execution_mode"], "python39_syntax": "passed"})
        print(destination.name, destination.stat().st_size, "bytes", flush=True)
    manifest["snapshot_sources"] = {str(p.relative_to(HARNESS)): hashlib.sha256(content).hexdigest() for p, content in SOURCE_BYTES.items()}
    checks = HARNESS / "notebook-checks" / "current"
    checks.mkdir(parents=True, exist_ok=True)
    (checks / "build-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
