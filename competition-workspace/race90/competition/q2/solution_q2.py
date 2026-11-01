# -*- coding: utf-8 -*-
# 이 전체 코드를 대회 원본 Notebook의 코드 셀에 붙여넣어 실행할 수 있습니다.
# 함께 제공한 prediction_data_q2.json.gz 파일을 같은 문제 폴더에 업로드하세요.

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
PREDICTION_DATA_PATH = Path.cwd() / 'prediction_data_q2.json.gz'
if not PREDICTION_DATA_PATH.is_file():
    raise FileNotFoundError('예측 데이터 파일을 Notebook과 같은 폴더에 업로드하세요: ' + str(PREDICTION_DATA_PATH))
with gzip.open(PREDICTION_DATA_PATH, 'rb') as _handle:
    _snapshot_bytes = _handle.read()
if hashlib.sha256(_snapshot_bytes).hexdigest() != 'a3072ae71f6a5327cf264edeb5e7880723fedea4fea30aa4fe0ba3f9033cb928':
    raise ValueError('예측 데이터 파일이 손상되었거나 다른 문항 파일입니다.')
snapshot = json.loads(_snapshot_bytes.decode('utf-8'))
del _snapshot_bytes
print('산출물 스냅샷:', snapshot['created_at_utc'])
print('실행 방식:', snapshot['execution_mode'])

# % 실행 단계 3
QUESTION = 2

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

# % 실행 단계 4
def overlap(a,b):
 x1,y1=np.maximum(a[:2],b[:2]);x2,y2=np.minimum(a[2:],b[2:]);inter=max(x2-x1,0)*max(y2-y1,0)
 area_a=(a[2]-a[0])*(a[3]-a[1]);area_b=(b[2]-b[0])*(b[3]-b[1])
 return inter/max(area_a+area_b-inter,1e-6),inter/max(min(area_a,area_b),1e-6)

def merge(boxes):
 boxes=sorted(boxes,key=lambda x:x[1],reverse=True);out=[]
 for box,conf,model in boxes:
  match=None
  for j,obj in enumerate(out):
   iou,iom=overlap(np.array(box),np.array(obj['box']))
   if iou>.3 or iom>.68:match=j;break
  if match is None:out.append({'box':box,'conf':conf,'models':[model]})
  else:
   obj=out[match];obj['models'].append(model)
   # Preserve the broader box when a model detected only one part.
   if conf>.25:obj['box']=[min(box[0],obj['box'][0]),min(box[1],obj['box'][1]),max(box[2],obj['box'][2]),max(box[3],obj['box'][3])]
 return out

# % 실행 단계 5
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

# % 실행 단계 6
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

# % 실행 단계 7
count_columns = ["airplane", "bird", "drone"]
assert list(df.columns) == ["filename"] + count_columns
assert len(df) == 300 and df["filename"].is_unique
assert df["filename"].tolist() == sorted(image_paths, key=lambda value: int(Path(value).stem[4:]))
assert (df[count_columns].to_numpy() >= 0).all()
assert ((df[count_columns] > 0).sum(axis=1) == 1).all()
save_submission(df)
