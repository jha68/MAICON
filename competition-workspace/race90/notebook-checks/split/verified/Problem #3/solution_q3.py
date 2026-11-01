# -*- coding: utf-8 -*-
# 이 전체 코드를 대회 원본 Notebook의 코드 셀에 붙여넣어 실행할 수 있습니다.
# 함께 제공한 prediction_data_q3.json.gz 파일을 같은 문제 폴더에 업로드하세요.

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
PREDICTION_DATA_PATH = Path.cwd() / 'prediction_data_q3.json.gz'
if not PREDICTION_DATA_PATH.is_file():
    raise FileNotFoundError('예측 데이터 파일을 Notebook과 같은 폴더에 업로드하세요: ' + str(PREDICTION_DATA_PATH))
with gzip.open(PREDICTION_DATA_PATH, 'rb') as _handle:
    _snapshot_bytes = _handle.read()
if hashlib.sha256(_snapshot_bytes).hexdigest() != 'b5f1872c38a20538580c5e8e2337a6fd783fd958d876c3ae2c0694b7c853e640':
    raise ValueError('예측 데이터 파일이 손상되었거나 다른 문항 파일입니다.')
snapshot = json.loads(_snapshot_bytes.decode('utf-8'))
del _snapshot_bytes
print('산출물 스냅샷:', snapshot['created_at_utc'])
print('실행 방식:', snapshot['execution_mode'])

# % 실행 단계 3
QUESTION = 3

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

REQUIRED_TRAIN = ['id', 'report', 'fault_type']
REQUIRED_TEST = ['id', 'report']
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
import math
import re
from collections import Counter

def core_report(text):
    # Main observation follows one of the report's structural headers.
    headers = [r'운용자가\s*', r'정비반 보고:\s*', r'기능 시험 결과\s*',
               r'다음 사항을 확인함\.\s*',r'현장 정비 기록: .*?에서\s*',
               r'\[현상\]\s*',r'\[점검 메모\] .*?에서\s*',
               r'정비 메모 — .*?:\s*',r'운용자 인계(?:사항| 기록)\([^)]*\):\s*',
               r'(?:에 대한 )?재현 시험(?:에서| 결과,)\s*',
               r'점검 결과,\s*',r'운용 후 보고:\s*']
    found = []
    for pat in headers:
        m=re.search(pat,text)
        if m:found.append(m)
    if found:
        m=min(found,key=lambda x:x.start());text=text[m.end():]
    # A normal finding may be inserted immediately after the header.
    parts=text.split('.')
    while len(parts)>1 and re.search(r'가능성은 낮으며|특이사항이 없었다|정상으로 확인됐다|뒷받침하는 징후는 발견되지 않았다',parts[0]):
        parts=parts[1:]
    text='.'.join(parts).strip()
    endings = [r'이라고 보고함',r'15분 점검 중',r'정비 후 첫 운용에서',
               r'외기 온도는 정상 범위였음',r'증상은 2회 연속 재현됨',
               r'주변 환경 조건은 정상 범위였다',r'동일 현상이 재시험에서도',
               r'짧은 확인 운전에서도',r'동일 조건에서 두 차례',
               r'정지 후 다시 시작하면',r'\[대상\]',r'\[기타\]']
    text=re.split('|'.join(endings),text)[0]
    text=text.split('.')[0]
    return text.strip(' .')

def semantic_rule(text):
    """Domain-level component rules applied only to the primary observation."""
    s=core_report(text)
    if re.search(r'통신|송신|수신|송수신|신호|무선|안테나|네트워크|호출|채널|헤드셋|음성|통화|단말|메시지|자료 교환|데이터|외부 장비와의 연결',s):
        return 'COMMUNICATION'
    cooling = re.search(r'냉각|라디에이터|온도|과열|팬|풍량|뜨거|열 축적|열 관련|흡입구|공기 흡입부|외기 조건|내부 측정값|보조 탱크|보조 저장조|장시간.*보호 경보|연속 운전.*보호 경보',s)
    electrical_heat = re.search(r'(전원 케이블|전력선).*과열',s)
    if cooling and not electrical_heat:
        return 'COOLING'
    if re.search(r'바퀴|차륜|조향|브레이크|제동|타이어|휠|트레드|직진|방향 전환|차체|속도.*진동.*저속',s):
        return 'MOBILITY'
    if re.search(r'엔진|동력부|공회전|크랭킹|연소|점화|윤활|배기|회전수|가속.*회전|진동감.*가속|첫 기동|기계음|금속성 충격|노킹|회전이 급격히|부하 증가.*힘이 빠|오르막.*출력 부족',s):
        return 'ENGINE'
    if re.search(r'배터리|전압|전류|충전|축전|배선|전력|전기|발전|회로|계기|커넥터|단자|전원|표시등|표시 램프',s):
        return 'ELECTRICAL'
    if re.search(r'압력|실린더|펌프|피팅|작동유|유압|액추에이터|저압|연결부.*젖|저장 탱크|구동.*둔',s):
        return 'HYDRAULIC'
    return None

def grams(text):
    text=re.sub(r'\s+',' ',text)
    return Counter(text[i:i+n] for n in (2,3,4) for i in range(len(text)-n+1))

class ReportClassifier:
    def fit(self,reports,labels):
        unique={}
        for text,label in zip(reports,labels):
            core=core_report(text)
            if core in unique and unique[core]!=label:raise ValueError('Conflicting symptom labels')
            unique[core]=label
        self.cores=list(unique);self.labels=[unique[c] for c in self.cores]
        docs=[grams(c) for c in self.cores];df=Counter()
        for d in docs:df.update(d.keys())
        self.idf={g:math.log((1+len(docs))/(1+n))+1 for g,n in df.items()}
        self.vectors=[self.vector(d) for d in docs]
        return self
    def vector(self,counts):
        x={g:(1+math.log(n))*self.idf[g] for g,n in counts.items() if g in self.idf}
        norm=math.sqrt(sum(v*v for v in x.values())) or 1
        return {g:v/norm for g,v in x.items()}
    def predict_one(self,report):
        rule=semantic_rule(report)
        if rule:return rule
        x=self.vector(grams(core_report(report)))
        sims=[sum(v*vec.get(g,0) for g,v in x.items()) for vec in self.vectors]
        return self.labels[max(range(len(sims)),key=sims.__getitem__)]
    def predict(self,reports):return [self.predict_one(s) for s in reports]

# % 실행 단계 5
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

# % 실행 단계 6
assert list(df.columns) == ["id", "fault_type"]
assert len(df) == 180 and df["id"].is_unique
assert df["id"].tolist() == test["id"].tolist()
assert set(df["fault_type"]).issubset(allowed)
save_submission(df)
