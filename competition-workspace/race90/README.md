# race90 — 90분 AI 경진대회 실행 하네스

**새 채팅에서 시작할 때는 [AGENTS.md](AGENTS.md)를 먼저 읽으세요.** 실제 대회도 현재 연결된
Runpod Pod `vvpxznt3e1lvyn`의 A100 SXM 80GB 3장을 재사용합니다. `config.live.yaml`의
`existing_pod_id`와 실제 SSH 키를 유지하고 문항 설정을 지문에 맞게 바꾸세요.
기존 Pod에 접속하는 경로가 실제 연습에서 검증됐습니다. 시스템 Python 설치 차단, 구형 Notebook
호환성, CSV 형식, 자동 생성 API의 미해결 제한과 회귀 테스트를 위 지침에 기록했습니다.
현재 자동 삭제 예약은 **2026-10-10 16:30 KST**이므로 대회 시작 전에 남은 임대 시간을 확인하세요.

예선 온라인 환경은 사용자 제공 예상값으로 **Python 3.9.7(conda-forge), 논리 CPU 2개,
PyTorch 1.10.0+cpu, CUDA 없음**입니다. 실행 파일은 `/opt/conda/bin/python`, 예시 작업 폴더는
`/home/jovyan/work/Problem #1`입니다. [환경 기록](submission-environment.json)과
[제출 호환성 지침](AGENTS.md)을 확인하세요. 실제 학습에는 기존 Runpod GPU를 사용합니다.
**개발·분석·학습은 현재 로컬/Runpod 환경 기준이며, 예선 버전에 맞춰 낮추거나 모델·라이브러리 선택을
제한하지 않습니다.** 위 사양은 마지막에 제출 Notebook에서 실제 실행할 부분을 확인하는 참고 정보입니다.
하네스 자체의 Python >= 3.10 요구사항은 로컬/Runpod용이며, 온라인 Python 3.9에 그대로 설치하는
절차가 아닙니다. 동일한 온라인 환경에서의 Notebook 실행은 아직 검증하지 않았습니다.

**문제 분석·코딩·학습·추론은 로컬 CLI를 중심으로 진행하고, 마지막에 제공된 Notebook을 제출용으로 완성합니다.** 로컬에서는 빠른 비상 제출본을 확보하고, Runpod의 A100 3장에는 문항별 GPU 작업을 한 개씩 배정합니다. GPU는 준비되는 즉시 시작하며 로컬 CPU 기본 학습이 끝나기를 기다리지 않습니다. 선행 문항의 결과가 필요한 경우에는 의존성이 끝난 작업만 실행합니다.

90분 중 기본 학습 마감은 **70분**, 제출 파일 완성 목표는 **85분**입니다. 마지막 5분은 온라인 제공 Notebook 반영·CSV 업로드·접수 확인에 남깁니다. Notebook을 만드는 데 걸릴 시간까지 학습에 쓰지 않도록 별도 마감을 적용합니다.

제공 Jupyter에 GPU가 없다는 조건과 로컬/원격 GPU 사용은 구분합니다. 기본 예제는 원격 GPU 중심이며, 현재 작은 로컬 CPU에 맞춰 `cpu_workers: 1`, `cpu_threads: 2`, 문항별 `baseline_seconds: 120`으로 설정했습니다. 실제 CPU affinity와 컨테이너 quota보다 많은 스레드를 배정하지 않습니다. CPU 작업이 비슷하게 빠른 작은 표/텍스트 문제는 `resource: cpu`를 사용하세요.

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

Pod에서는 실행 폴더의 `.venv`에 추가 패키지를 설치합니다. 공식 이미지의 PyTorch·CUDA를 `--system-site-packages`로 공유하고, 작업 실행·CUDA 검사·결과 회수 모두 같은 Python을 사용합니다. 시스템 Python이 패키지 설치를 막는 이미지에서도 시스템 환경을 변경하지 않습니다.

```bash
# CPU 재추론용 PyTorch; GPU 패키지를 무작정 설치하며 시간을 쓰지 않도록 분리했습니다.
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install 'transformers>=4.46,<6' safetensors sentencepiece ultralytics catboost
```

## 대회 시작 후

문제 읽기와 다운로드를 시작하기 전에 로컬 터미널에서 전체 대회 시계를 시작하세요. 파일이 이미 있으면 덮어쓰지 않아 실수로 90분이 다시 부여되지 않습니다. 연습 때에는 다른 `--output` 경로를 사용하세요.

```bash
python -m race90 start-clock --minutes 90 --output exam-clock.json
```

1. 제공된 폴더와 원본 `.ipynb`를 `data/`에 넣습니다.
2. 지문과 예시 제출 파일을 읽고 `config.yaml`의 경로, 정답 컬럼, 지표, 데이터 유형, 검증 방법을 바꿉니다.
3. GPU 대여 설정과 가격 상한을 확인한 뒤 실행합니다.

```bash
.venv/bin/python -m race90 plan config.yaml
.venv/bin/python -m race90 run config.yaml --clock exam-clock.json
```

`--clock`은 문제 이해·다운로드에 이미 사용한 시간도 포함합니다. 공식 종료 시각을 알고 있으면 대신 `--deadline-epoch`을 사용할 수 있습니다. 두 옵션 모두 생략하면 명령 시작부터 90분으로 계산하므로 실전에서는 시계 파일 또는 공식 종료 시각을 지정하세요.

```bash
# 실제 종료 시각을 timezone이 포함된 datetime으로 변환한 epoch 초를 사용하세요.
.venv/bin/python -m race90 run config.yaml --deadline-epoch 실제_종료_epoch_초
```

`check config.yaml`은 학습 없이 데이터 검사만 병렬 실행합니다. 대회 중 `run` 직전에 같은 검사를 반복할 필요는 없습니다. `run` 자체가 검사합니다.

풀이 중에는 Jupyter를 켜 둘 필요가 없습니다. [notebooks/START_HERE.ipynb](notebooks/START_HERE.ipynb)는 원할 때 쓰는 보조 실행 화면이며, 기본 작업 경로는 로컬 CLI입니다.

```bash
.venv/bin/python -m race90 status runs/실행폴더
.venv/bin/python -m race90 run config.yaml --resume runs/실행폴더
# 현재 결과가 충분하거나 조기에 제출본을 확보하려면 별도 터미널에서:
.venv/bin/python -m race90 finalize runs/실행폴더
```

`finalize`는 실행 중인 하네스에 학습 중단과 최종화를 요청합니다. 완료된 후보를 보존하고 회수·Notebook 제작으로 전환합니다. 중단된 프로세스 자체를 재시작하는 명령은 아니며, 그 경우 `run --resume`을 사용하세요. 재개 시에는 기존 종료 시각을 쓰므로 `--clock`이나 `--deadline-epoch`을 추가하지 않습니다.

## 제출본을 위한 시간 관리

| 대회 시작 기준 | 처리 내용 |
|---|---|
| 0~70분 | 로컬 분석·전처리·CPU/GPU 학습·검증·예측 저장 |
| 70~72분 | 학습 강제 종료 후 완료된 원격 결과 회수·자원 정리 |
| 72~84분 | 제공 Notebook에 소스·선택 설정·실제 결과 추가, 풀이 셀 실행·검증 |
| 84~85분 | CSV·Notebook·문항별 안내를 ZIP으로 묶고 무결성 검사 |
| 85~90분 | 사용자가 온라인 Notebook 반영·CSV 제출·접수 확인 |

각 시간은 **최종 허용 시각**이며, 학습이 빨리 끝나면 기다리지 않고 다음 단계로 진행합니다. `finalize_minutes`는 마지막 20분 전체를 뜻하고 `time_management.upload_minutes`의 5분도 여기에 포함됩니다. 연결 문항의 남은 작업량을 고려해 학습 예산을 배분하며, 새 개선 작업에 120초도 줄 수 없으면 기본 설정에서는 그 작업을 생략합니다.

`timeline.json`에는 변경되지 않는 마감 시각, `time-status.json`에는 현재 단계와 남은 시간, `notebooks.json`에는 문항별 최종화 결과가 기록됩니다. `status` 명령은 현재 시각으로 남은 시간을 다시 계산합니다. 학습 중에는 종료 30·20·10·5분 전 구간 진입을 로그로 알립니다.

Notebook은 다음 순서로 완성합니다.

1. **세 문항 모두 경량 실행본 확보:** 원본 지문·기존 출력, 전체 풀이 소스와 설정, 선택한 실제 예측·검증 지표, 실행된 CSV 검증 출력을 포함합니다. 무거운 모델 파일은 이 단계에서 제외합니다.
2. **실패 문항 먼저 재시도:** 한 문항의 변환이 다른 문항 시간을 모두 쓰지 않도록 남은 예산을 나눕니다. 미완료 Notebook을 제출 폴더에 넣지 않습니다.
3. **남은 시간에 모델 포함/재추론본 보강:** 요청한 `embed_models`와 `mode: infer`를 적용합니다. 시간 초과·용량 초과·패키지 오류가 나면 이미 검증된 경량 실행본을 보존하며 `enhancement_error`에 이유를 남깁니다.

이 과정은 최종화 단계에서만 진행하며, 풀이 중 Notebook을 반복 제작하지 않습니다. 경량본은 저장된 실제 결과를 복원·검증한 파일이며 Notebook에서 학습이나 새 추론을 했다고 표시하지 않습니다. 모델 재추론에 필요한 바이너리는 경량본에 포함되지 않습니다.

파일은 임시 폴더에서 실행·검증한 뒤에만 `deliverables/`에 반영합니다. 원본 문제 셀 보존, 풀이 코드 존재, 풀이 셀 실행 여부, 오류 출력 유무, CSV 형식과 선택 모델 예측의 일치를 확인합니다. ZIP 생성은 별도 제한시간을 가진 프로세스에서 처리합니다. 원격 장애나 모든 학습 실패까지 막을 수는 없으므로, 학습된 후보가 없거나 파일 검증에 실패하면 성공으로 표시하지 않습니다.

`--resume`은 기존 종료 시각을 유지합니다. 입력 CSV·이미지·Notebook의 크기/수정 시각, 모델 설정, 소스 코드가 바뀌면 재개를 거부합니다. 중단 시에는 완료된 CSV·체크포인트만 선택합니다. 선행 문항의 선택 모델이 달라지면 이전 모델에 의존한 후속 후보는 재사용하지 않습니다. 기본값만 남은 문항은 `selected.json`의 `fallback_only`와 검증 점수 `null`로 식별합니다. 이것은 유효한 형식의 비상 제출본이며 학습된 모델과 구별됩니다.

종료 시각을 지정해도 `total_minutes`보다 더 오래 학습하도록 연장하지는 않습니다. 테스트에 사용한 코어 버전을 고정하려면 설치 전에 `pip install -r requirements-tested.txt`를 실행하세요. 구체적인 검증 환경은 `VALIDATION.json`에 있습니다.

## 데이터 공유와 문항 연결

**공유 데이터만 있는 독립 문항:** 같은 `shared/train.csv`, `shared/test.csv`, `shared/images` 경로를 여러 문항에서 사용하고 `depends_on: []`로 둡니다. 데이터는 Pod에 한 번만 전송되며 문항은 동시에 실행됩니다.

같은 학습 표에 여러 문항의 정답 컬럼이 있으면 자동 특징 선택에서 해당 컬럼을 모두 제외합니다. 테스트에 없는 학습 전용 컬럼도 제외합니다. 지문에서 허용한 다른 정답/정보를 특징으로 써야 하는 경우에는 `feature_columns`를 명시하세요.

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

OOF를 전체 데이터에서 미리 만드는 것만으로 후속 문항의 검증 누수가 모두 없어지지는 않습니다. 후속 holdout을 평가할 때는 그 holdout의 정답이 상위 모델이나 후속 학습 특징 생성에도 쓰이지 않도록 바깥 검증 분할을 먼저 고정하고 그 안에서 OOF를 만드세요. 하네스는 ID·컬럼·부모 후보의 일치 여부를 검사하지만, 사용자 특징의 OOF 생성 과정까지 자동 증명하지는 않습니다.

더 복잡한 연결은 custom 어댑터의 `ctx.dependencies["q1"]`에서 체크포인트·CSV를 읽어 처리하세요. 순환 의존성, 존재하지 않는 문항, 상위 문항 결과 누락은 검사합니다. 코드 모듈은 설정 파일이 있는 폴더 안에 두어야 원격 실행과 Notebook에 함께 포함됩니다.

대기 중인 문항은 CPU/GPU 슬롯을 점유하지 않습니다. CPU 큐는 실행 가능한 기본 모델부터 처리하고, 긴 선행 문항의 개선 작업에는 후속 문항의 예산을 고려한 상한을 적용합니다. 따라서 독립 문항과 연결 문항이 섞여 있어도 실행 가능한 작업이 먼저 진행됩니다. 이 시간 배분은 작업별 설정 예산을 이용한 추정이며, 데이터가 매우 크면 마감 전 학습 완료를 보장하지는 않습니다.

## 모델과 제출 형식

| 유형 | 먼저 확보하는 CPU 결과 | `resource: gpu` 개선 모델 |
|---|---|---|
| tabular | 결측/범주 처리 + ExtraTrees | CatBoost GPU 분류/회귀 |
| timeseries | 표본별 평균/표준편차/범위/RMS/변화량/기울기 + ExtraTrees | 재표본화한 센서 시퀀스의 1D CNN |
| text | 한글에 적용 가능한 문자 TF-IDF + LogisticRegression | 한국어 `klue/roberta-small` 분류 미세조정 |
| image | 색/공간 특징 + ExtraTrees | ResNet18 전이학습 |
| yolo | 명시적으로 허용한 빈 탐지 결과 | YOLO 학습 + 이미지별 탐지 CSV |
| custom | 사용자가 구현한 `ctx.stage == "baseline"` | 사용자가 구현한 `ctx.stage == "upgrade"` |

GPU 모델은 같은 검증 분할에서 점수가 좋은 후보를 선택합니다. CPU 모델과 CatBoost는 **검증을 마친 모델·테스트 예측을 먼저 저장한 뒤**, 예상 소요시간이 남으면 전체 학습 데이터로 재학습합니다. 재학습 중 중단돼도 앞서 저장한 모델이 남습니다. 검증 점수가 같은 경우 전체 데이터로 재학습한 후보를 우선합니다. 신경망 모델은 검증 세트를 제외하고 학습한 가장 좋은 체크포인트를 사용합니다. 회귀는 표/텍스트/집계 센서의 CPU 모델과 표 데이터 CatBoost에서 지원하며 신경망 회귀는 custom 어댑터로 확장합니다.

표 데이터 GPU 작업은 `kind: tabular`, `resource: gpu`로 지정합니다. `catboost_iterations: [100, 300, 800]` 순서로 검증하며 `model.cbm`을 저장합니다. CatBoost 패키지는 해당 작업이 있을 때 Pod에 설치합니다. GPU에서 학습한 모델도 최종 Notebook에서 CatBoost를 설치하면 CPU로 추론할 수 있습니다. 실제 GPU CatBoost 속도는 데이터 크기·범주 수에 따라 달라지므로 작은 표에는 CPU가 유리할 수 있습니다.

`resource: cpu`의 표/센서 개선 단계는 기본적으로 HistGradientBoosting을 시도합니다. `cpu_estimator: extra_trees`로 기존 방식도 선택할 수 있습니다. CPU 이미지 개선은 기본적으로 색·공간 특징을 사용하며, `image_upgrade: embeddings`를 명시하면 고정 ResNet18 특징을 한 번 추출해 선형 분류기를 학습합니다. 원격 GPU 예제의 ResNet18 전이학습 경로는 그대로 유지합니다.

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

원격 학습이 끝나면 가장 좋은 완료 후보와 필요한 산출물만 회수합니다. 모든 epoch 체크포인트나 YOLO optimizer 디렉터리를 내려받지 않습니다. 학습 취소/마감 신호가 들어와도 결과 회수는 작업당 최대 90초, 전체 회수 마감 이내에서 시도합니다. 회수와 정리는 마지막 20분에 포함됩니다.

## 최종 산출물과 제공 Notebook

```text
runs/실행폴더/
  deliverables/
    q1_submission.csv
    q1.solution.ipynb
    q1_result.json
    submission_checklist.json
    ... q2, q3 ...
  submission_pack.zip
  source.zip
  final.json
  status.json
  timeline.json
  time-status.json
  events.jsonl
  runpod.json
  q1/candidates/...
```

`q*.solution.ipynb`는 원본 Notebook에 풀이를 추가한 파일입니다. 원본 문제 셀과 기존 출력은 보존하고 추가한 셀만 실제 실행합니다. 각 Notebook에는 읽을 수 있는 모델링 소스, 실행 가능한 코드 묶음, 저장된 예측/선택 모델, 검증 지표, 실제 CSV 검증 출력이 포함됩니다. **`mode: restore`는 하네스가 실제 생성한 예측을 복원합니다. Notebook에서 학습을 다시 했다고 표시하지 않습니다.**

온라인 환경에서 해당 문제 Notebook에 풀이 셀을 옮기거나 문제 파일을 이 결과 Notebook으로 교체하고, `DATA_ROOT`를 온라인 데이터 위치에 맞춰 추가된 셀을 다시 실행하세요. 생성 파일이 있다는 것만으로 온라인 제공 Notebook에 반영되는 것은 아닙니다.

Notebook에서 `RECOMPUTE=True`면 저장 모델을 CPU로 다시 추론하고, `RETRAIN_CPU=True`면 CPU 기본 모델을 다시 학습할 수 있습니다. 그때 필요한 scikit-learn/PyTorch/Transformers/Ultralytics 버전을 설치하세요. `mode: infer`를 설정하면 내보내기 단계부터 CPU 재추론을 실행해 검증합니다.

`embed_models: true`는 시간과 용량이 허용될 때 모델까지 Notebook에 넣어 외부 모델 파일 의존성을 줄입니다. 대형 모델 때문에 Notebook을 올리기 어렵다면 `false`로 바꿔 예측·코드·검증 결과만 넣으세요. 이 경우 저장 모델 재추론에는 별도 체크포인트를 온라인 환경에 옮겨야 합니다. `max_embedded_mb`를 넘기거나 보강 단계가 실패하면 앞서 검증된 경량본을 유지합니다. 실습 때 실제 업로드 소요시간을 재서 `upload_minutes`를 조정하세요.

마지막에 `final.json`을 확인하세요. `complete`는 CSV·Notebook 생성 완료, `submission_ready`는 모든 문항의 검증 점수가 있는 후보·제공 Notebook 사용·실행된 풀이 셀·ZIP 검증·85분 파일 완성 목표 준수를 뜻합니다. `compact_notebooks`에는 경량본으로 마무리한 문항이 표시됩니다. `fallback_tasks`가 비어 있지 않으면 해당 문항은 비상 기본값뿐입니다. `submission_ready`가 false면 CLI는 종료 코드 1을 반환합니다. `submission_checklist.json`에서 각 원본 Notebook에 대응하는 파일을 확인하세요. `online_upload_completed`는 자동으로 true가 되지 않으며, 온라인 반영과 제출 접수 확인은 마지막 5분에 직접 완료해야 합니다.

## 유료 자원 없이 전체 흐름 연습

```bash
.venv/bin/python scripts/make_demo.py --output demo
.venv/bin/python -m race90 run demo/demo.yaml
.venv/bin/python -m pytest -q
```

합성 센서·이미지·한국어 텍스트 세 문제를 만들고 CSV와 실행된 Notebook까지 생성합니다. 데모 검증 점수는 하네스 동작 확인용이며 실제 대회 성능을 뜻하지 않습니다. Runpod 대여·SSH·원격 CUDA 학습은 실제 계정과 GPU로 별도 연습해야 합니다. 준비시간 7분, 회수시간, 70분 학습 마감, 85분 파일 완성 목표, 전체 90분 종료는 설정 가능합니다.

새 Pod 생성 시 API의 `terminateAfter`를 넣고, 독립 로컬 watchdog과 종료 시 삭제로 정리를 보완합니다. watchdog은 로컬 컴퓨터가 계속 켜져 있어야 동작합니다. 외부 서비스 오류로 삭제에 실패한 경우 `runpod.json`에 상태와 Pod ID가 남습니다. 기존 Pod에 연결한 경우 사용자가 정리합니다.

## 구현에 사용한 공식 자료

- [Runpod GPU 유형과 A100 식별자](https://docs.runpod.io/references/gpu-types)
- [Runpod CLI](https://github.com/runpod/runpodctl), [GraphQL 생성/종료 시각 필드](https://graphql-spec.runpod.io/)
- [scikit-learn 검증 분할](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Torchvision ResNet18 가중치](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
- [KLUE 한국어 모델](https://huggingface.co/klue/roberta-small), [Transformers 모델 저장/불러오기](https://huggingface.co/docs/transformers/main_classes/model)
- [Ultralytics 학습 콜백](https://docs.ultralytics.com/usage/callbacks/)
- [CatBoost 학습 설정](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_fit), [CPU 추론 설정](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_predict)

## 이번 검토에서 수정한 주요 결함

- GPU가 로컬 기본 학습 완료를 기다리던 흐름을 분리했습니다.
- 연결 문항의 대기 작업이 CPU 슬롯을 점유하지 않도록 스케줄러를 교체했습니다.
- 전체 재학습 중 타임아웃이 나면 학습 결과가 전부 사라지던 문제를 수정했습니다.
- CSV를 다시 읽을 때 `001`, `NA` 같은 문자열 라벨과 ID가 바뀌는 문제를 수정했습니다.
- 의존 특징을 합칠 때 ID 자료형을 보존하고, 부모 체크포인트가 바뀐 후속 후보의 재사용을 차단했습니다.
- 잘못된 지표·확률 컬럼·검증 분할·시간 예산은 실행 전에 검출합니다.
- 결과 회수가 학습 취소 신호에 함께 취소되는 문제와 불필요한 대형 파일 전송을 수정했습니다.
- Notebook 직접 재학습에도 프로세스 제한시간을 적용하고, 작은 CPU에서 Notebook 내보내기를 기본 한 개씩 실행합니다.

자동 테스트와 합성 데이터 검증 결과는 `VALIDATION.json`에 기록합니다. 실제 유료 Pod 대여, SSH 연결, CUDA 성능은 이번 검토에서 실행하지 않았습니다.
