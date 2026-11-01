# -*- coding: utf-8 -*-
# 이 전체 코드를 대회 원본 Notebook의 코드 셀에 붙여넣어 실행할 수 있습니다.
# 함께 제공한 prediction_data_q1.json.gz 파일을 같은 문제 폴더에 업로드하세요.

# % 실행 단계 1
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

# % 실행 단계 2
# 별도 파일에는 예측 산출물과 검증 정보만 들어 있습니다. Python 코드를 실행하지 않습니다.
import gzip
PREDICTION_DATA_PATH = Path.cwd() / 'prediction_data_q1.json.gz'
if not PREDICTION_DATA_PATH.is_file():
    raise FileNotFoundError('예측 데이터 파일을 Notebook과 같은 폴더에 업로드하세요: ' + str(PREDICTION_DATA_PATH))
with gzip.open(PREDICTION_DATA_PATH, 'rb') as _handle:
    _snapshot_bytes = _handle.read()
if hashlib.sha256(_snapshot_bytes).hexdigest() != '3b1625f33c52385bf51a4eba8bf3eeae3980a6a55d29a428e12a14815554bc52':
    raise ValueError('예측 데이터 파일이 손상되었거나 다른 문항 파일입니다.')
snapshot = json.loads(_snapshot_bytes.decode('utf-8'))
del _snapshot_bytes
print('산출물 스냅샷:', snapshot['created_at_utc'])
print('실행 방식:', snapshot['execution_mode'])

# % 실행 단계 3
QUESTION = 1

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def sha256_csv_text(path):
    # UTF-8 BOM과 줄바꿈 방식의 차이는 허용합니다.
    value = Path(path).read_bytes().decode("utf-8-sig").replace("\r\n", "\n")
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
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    written = pd.read_csv(temporary, encoding="utf-8")
    pd.testing.assert_frame_equal(written, frame.reset_index(drop=True), check_dtype=False)
    temporary.replace(OUTPUT_PATH)
    print("저장 완료:", OUTPUT_PATH)
    print("행 수:", len(frame), "| 열:", list(frame.columns))
    print("SHA256:", sha256_file(OUTPUT_PATH))
    print("실행 시간: %.2f초" % (time.perf_counter() - started_at))
    display(frame.head(10))
    display(FileLink(str(OUTPUT_PATH.relative_to(Path.cwd()))))

REQUIRED_TRAIN = ['sample_id', 'label', 't_ms']
REQUIRED_TEST = ['sample_id', 't_ms']
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

# % 실행 단계 4
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

# % 실행 단계 5
assert list(df.columns) == ["sample_id", "label"]
assert len(df) == 300 and df["sample_id"].is_unique
assert df["sample_id"].tolist() == sorted(test["sample_id"].unique())
assert set(df["label"]).issubset(allowed)
save_submission(df)
