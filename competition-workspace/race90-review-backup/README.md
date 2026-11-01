# race90 — 90분 AI 경진대회 실행 하네스

세 문항의 데이터 검사·CPU 기본 모델을 병렬 실행하는 동안 Runpod을 준비합니다. 준비된 A100 3장에는 문항별 GPU 작업을 한 개씩 배정합니다. 선행 문항의 결과가 필요한 경우에는 의존성이 끝난 작업만 실행합니다. 학습 마감은 82분이며 마지막 8분을 결과 회수·Notebook 실행·제출 파일 묶기에 남깁니다.

현재 설정은 **한 Pod에 A100 80GB 3장**입니다. A100 종류와 가격 상한은 `config.yaml`에서 바꿀 수 있습니다. 실제 GPU 재고와 3장 동시 배치 가능 여부는 실행 때 확인됩니다. API 키가 없거나 준비에 실패하면 저장된 CPU 결과로 마무리합니다.

## 대회 전에 한 번 준비

Python 3.10 이상, Linux/macOS, OpenSSH `ssh`/`scp`가 필요합니다. Windows에서는 WSL을 사용하세요.

```bash
cd race90
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'

# Runpod 공식 CLI 설치. 대회 전 버전을 확정하고 대회 중에는 업데이트하지 마세요.
curl -sSL https://cli.runpod.net | bash
runpodctl version
runpodctl doctor
runpodctl gpu list --include-unavailable
runpodctl template search pytorch

cp config.example.yaml config.yaml
```

키는 환경변수 `RUNPOD_API_KEY`로 설정하거나 `runpodctl doctor`로 저장하세요. 하네스는 키를 설정 파일·Notebook·제출 ZIP에 넣지 않습니다. 실제 대여는 `run` 명령을 실행할 때 시작됩니다. `plan`, `check`, 합성 데이터 데모는 대여하지 않습니다.

`max_gpu_hourly_usd`는 **3장 합계의 GPU 시간당 가격 상한**입니다. 스토리지 가격을 포함한 총 청구액 상한은 아닙니다. `existing_pod_id`를 지정하면 이미 준비한 Pod에 접속하므로 부팅 시간을 줄일 수 있습니다. 하네스가 만든 Pod는 결과 회수 후 삭제합니다. 사용자가 미리 만든 Pod는 남겨 둡니다.

최종 Notebook에서 GPU 모델을 CPU로 다시 추론하려면 해당 패키지도 미리 설치하세요. CPU 기본 모델만 사용할 때에는 필요 없습니다.

```bash
# CPU 재추론용 PyTorch; GPU 패키지를 무작정 설치하며 시간을 쓰지 않도록 분리했습니다.
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install 'transformers>=4.46,<6' safetensors sentencepiece ultralytics
```

## 대회 시작 후

1. 제공된 폴더와 원본 `.ipynb`를 `data/`에 넣습니다.
2. 지문과 예시 제출 파일을 읽고 `config.yaml`의 경로, 정답 컬럼, 지표, 데이터 유형, 검증 방법을 바꿉니다.
3. GPU 대여 설정과 가격 상한을 확인한 뒤 실행합니다.

```bash
.venv/bin/python -m race90 plan config.yaml
.venv/bin/python -m race90 run config.yaml
```

문제 이해·다운로드에 쓴 시간도 제한 시간에 포함된다면 실제 대회 종료 시각을 지정하세요. 기본 90분은 명령을 시작한 시점부터 계산합니다.

```bash
# 실제 종료 시각을 timezone이 포함된 datetime으로 변환한 epoch 초를 사용하세요.
.venv/bin/python -m race90 run config.yaml --deadline-epoch 실제_종료_epoch_초
```

`check config.yaml`은 학습 없이 데이터 검사만 병렬 실행합니다. 대회 중 `run` 직전에 같은 검사를 반복할 필요는 없습니다. `run` 자체가 검사합니다.

Jupyter에서 실행하려면 [notebooks/START_HERE.ipynb](notebooks/START_HERE.ipynb)를 여세요. 배경 실행·진행 상황 확인·최종 파일 찾기 셀이 들어 있습니다.

```bash
.venv/bin/python -m race90 status runs/실행폴더
.venv/bin/python -m race90 run config.yaml --resume runs/실행폴더
```

`--resume`은 기존 종료 시각을 유지합니다. 입력 파일의 크기/수정 시각이나 모델 설정이 바뀌면 재개를 거부합니다. 중단 시에는 완료된 CSV·체크포인트만 선택합니다. 기본값만 남은 문항은 `selected.json`의 `fallback_only`와 검증 점수 `null`로 식별합니다. 이것은 유효한 형식의 비상 제출본이며 학습된 모델과 구별됩니다.

종료 시각을 지정해도 `total_minutes`보다 더 오래 학습하도록 연장하지는 않습니다. 테스트에 사용한 코어 버전을 고정하려면 설치 전에 `pip install -r requirements-tested.txt`를 실행하세요. 구체적인 검증 환경은 `VALIDATION.json`에 있습니다.

## 데이터 공유와 문항 연결

**공유 데이터만 있는 독립 문항:** 같은 `shared/train.csv`, `shared/test.csv`, `shared/images` 경로를 여러 문항에서 사용하고 `depends_on: []`로 둡니다. 데이터는 Pod에 한 번만 전송되며 문항은 동시에 실행됩니다.

**선행 결과를 쓰는 연결 문항:** `depends_on`을 지정합니다. 예를 들어 q2는 q1 결과가 필요하고 q3는 독립이면 q1과 q3를 먼저 실행하고 q1이 마무리된 뒤 q2를 실행합니다. 선행 문항의 CPU/GPU 후보 중 최종 선택된 산출물을 고정한 뒤 후속 문항에 넘깁니다.

```yaml
tasks:
  - name: q1
    # 나머지 필수 필드...
    depends_on: []
  - name: q2
    # 나머지 필수 필드...
    depends_on: [q1]
    dependency_features:
      - task: q1
        train_file: train_features.csv
        test_file: test_features.csv
        on: id
  - name: q3
    # 나머지 필수 필드...
    depends_on: []
```

`dependency_features`는 표 데이터에 특징을 ID로 결합합니다. 선행 문항은 custom 어댑터에서 두 파일을 `ctx.publish(..., artifacts=...)`로 저장해야 합니다. 학습 타깃으로 만든 상위 모델 예측을 후속 학습 특징으로 쓸 때에는 학습 행의 예측을 **OOF**로 만들어야 합니다. 같은 학습 행을 이미 본 모델의 예측을 붙이면 검증 점수가 부풀 수 있습니다. 특징 파일은 ID별 한 행이어야 하며 학습/테스트 양쪽의 컬럼을 일치시킵니다.

더 복잡한 연결은 custom 어댑터의 `ctx.dependencies["q1"]`에서 체크포인트·CSV를 읽어 처리하세요. 순환 의존성, 존재하지 않는 문항, 상위 문항 결과 누락은 검사합니다. 코드 모듈은 설정 파일이 있는 폴더 안에 두어야 원격 실행과 Notebook에 함께 포함됩니다.

## 모델과 제출 형식

| 유형 | 먼저 확보하는 CPU 결과 | `resource: gpu` 개선 모델 |
|---|---|---|
| tabular | 결측/범주 처리 + ExtraTrees | 기본 제공은 CPU ExtraTrees 개선 |
| timeseries | 표본별 평균/표준편차/범위/RMS/변화량/기울기 + ExtraTrees | 재표본화한 센서 시퀀스의 1D CNN |
| text | 한글에 적용 가능한 문자 TF-IDF + LogisticRegression | 한국어 `klue/roberta-small` 분류 미세조정 |
| image | 색/공간 특징 + ExtraTrees | ResNet18 전이학습 |
| yolo | 명시적으로 허용한 빈 탐지 결과 | YOLO 학습 + 이미지별 탐지 CSV |
| custom | 사용자가 구현한 `ctx.stage == "baseline"` | 사용자가 구현한 `ctx.stage == "upgrade"` |

GPU 모델은 같은 검증 분할에서 기본 모델보다 점수가 좋을 때 선택됩니다. 표·텍스트·센서 CPU 모델은 검증 후 전체 학습 데이터로 다시 학습합니다. 신경망 모델은 검증 세트를 제외하고 학습한 가장 좋은 체크포인트를 사용합니다. 회귀는 표/텍스트/집계 센서의 CPU 모델에서 지원하며 신경망 회귀는 custom 어댑터로 확장합니다.

센서 CSV는 한 녹화 ID에 여러 시간 행이 있는 형태입니다. 각 녹화의 라벨은 일치해야 합니다. 이미 한 표본당 한 행의 특징이 제공되면 `kind: tabular`를 사용하세요. 미래를 예측하는 문제에는 `split.strategy: time`과 `split.time_column`, 필요하면 `gap`을 지정합니다. 피험자/장비를 구분해야 하면 `strategy: group`, `group_column`을 지정합니다. 시계열을 무작위로 섞는 설정은 기본 허용하지 않습니다.

```yaml
prediction_type: probability
prediction_columns: [positive_probability]
positive_class: 1
metric: roc_auc
```

다중 클래스 확률은 `class_order: [0, 1, 2]`와 동일 순서의 `prediction_columns`를 함께 지정합니다. CSV의 컬럼, 행 수, ID 집합·순서, 결측/무한값, 확률 범위·합계를 검사합니다. ID 앞의 0을 보존해야 하면 `read_csv_kwargs: {dtype: {id: string}}`를 설정하세요.

YOLO는 [examples/yolo.task.yaml](examples/yolo.task.yaml)을 사용하세요. `dataset.yaml`의 `path`는 YAML 파일 기준 상대 경로이고 `train`/`val`은 그 데이터 경로 기준 상대 경로입니다. txt 이미지 목록은 txt 파일 기준 상대 경로를 사용하세요. 하네스가 로컬/Pod에서 절대 경로로 변환하고 학습/검증 이미지 중복을 검사합니다. 기본 CSV는 이미지당 한 행에 `class confidence xmin ymin xmax ymax`를 픽셀 좌표로 이어 붙입니다. 대회 형식이 다르면 `exporter: custom_adapters.내모듈:내함수`를 구현합니다. 제출 형식을 지문 없이 자동 추정하지 않습니다.

GPU 한 장당 학습 프로세스 하나를 실행합니다. 기본적으로 한 모델을 세 GPU에 분산하는 DDP는 사용하지 않습니다. 독립 문항 3개를 각 GPU에 배정하고, 연결 문항이 대기하는 동안에는 실행 가능한 문항을 먼저 배정합니다. CPU 모델·파일 전송·원격 패키지 설치는 가능한 범위에서 겹쳐 실행합니다.

## 최종 산출물과 제공 Notebook

```text
runs/실행폴더/
  deliverables/
    q1_submission.csv
    q1.solution.ipynb
    q1_result.json
    ... q2, q3 ...
  submission_pack.zip
  source.zip
  final.json
  status.json
  events.jsonl
  runpod.json
  q1/candidates/...
```

`q*.solution.ipynb`는 원본 Notebook에 풀이를 추가한 파일입니다. 원본 문제 셀과 기존 출력은 보존하고 추가한 셀만 실제 실행합니다. 각 Notebook에는 읽을 수 있는 모델링 소스, 실행 가능한 코드 묶음, 저장된 예측/선택 모델, 검증 지표, 실제 CSV 검증 출력이 포함됩니다. **`mode: restore`는 하네스가 실제 생성한 예측을 복원합니다. Notebook에서 학습을 다시 했다고 표시하지 않습니다.**

온라인 환경에서 해당 문제 Notebook에 풀이 셀을 옮기거나 문제 파일을 이 결과 Notebook으로 교체하고, `DATA_ROOT`를 온라인 데이터 위치에 맞춰 추가된 셀을 다시 실행하세요. 생성 파일이 있다는 것만으로 온라인 제공 Notebook에 반영되는 것은 아닙니다.

Notebook에서 `RECOMPUTE=True`면 저장 모델을 CPU로 다시 추론하고, `RETRAIN_CPU=True`면 CPU 기본 모델을 다시 학습할 수 있습니다. 그때 필요한 scikit-learn/PyTorch/Transformers/Ultralytics 버전을 설치하세요. `mode: infer`를 설정하면 내보내기 단계부터 CPU 재추론을 실행해 검증합니다.

`embed_models: true`는 모델까지 Notebook에 넣어 외부 모델 파일 의존성을 줄입니다. 대형 모델 때문에 Notebook을 올리기 어렵다면 `false`로 바꿔 예측·코드·검증 결과만 넣으세요. 이 경우 저장 모델 재추론에는 별도 체크포인트를 온라인 환경에 옮겨야 합니다. `max_embedded_mb`를 넘기면 해당 Notebook 내보내기를 실패로 보고하므로 실습 때 업로드 크기를 확인하세요.

마지막에 `final.json`의 `complete`, 문항별 `stage`/`score`, `notebooks.json`의 상태를 확인한 뒤 CSV를 대회 사이트에 제출하세요. 제출 사이트/API 정보가 없으므로 사이트 업로드는 자동 실행하지 않습니다.

## 유료 자원 없이 전체 흐름 연습

```bash
.venv/bin/python scripts/make_demo.py --output demo
.venv/bin/python -m race90 run demo/demo.yaml
.venv/bin/python -m pytest -q
```

합성 센서·이미지·한국어 텍스트 세 문제를 만들고 CSV와 실행된 Notebook까지 생성합니다. 데모 검증 점수는 하네스 동작 확인용이며 실제 대회 성능을 뜻하지 않습니다. Runpod 대여·SSH·원격 CUDA 학습은 실제 계정과 GPU로 별도 연습해야 합니다. 준비시간 7분, 회수시간, 82분 학습 마감, 전체 90분 종료는 설정 가능합니다.

새 Pod 생성 시 API의 `terminateAfter`를 넣고, 독립 로컬 watchdog과 종료 시 삭제로 정리를 보완합니다. watchdog은 로컬 컴퓨터가 계속 켜져 있어야 동작합니다. 외부 서비스 오류로 삭제에 실패한 경우 `runpod.json`에 상태와 Pod ID가 남습니다. 기존 Pod에 연결한 경우 사용자가 정리합니다.

## 구현에 사용한 공식 자료

- [Runpod GPU 유형과 A100 식별자](https://docs.runpod.io/references/gpu-types)
- [Runpod CLI](https://github.com/runpod/runpodctl), [GraphQL 생성/종료 시각 필드](https://graphql-spec.runpod.io/)
- [scikit-learn 검증 분할](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Torchvision ResNet18 가중치](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
- [KLUE 한국어 모델](https://huggingface.co/klue/roberta-small), [Transformers 모델 저장/불러오기](https://huggingface.co/docs/transformers/main_classes/model)
- [Ultralytics 학습 콜백](https://docs.ultralytics.com/usage/callbacks/)
