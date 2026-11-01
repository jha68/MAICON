# 새 세션을 위한 대회 실행 지침

## 최신 인프라 상태 — 2026-10-10

사용자의 비용 중단 요청으로 Pod `vvpxznt3e1lvyn`을 종료·삭제했다.
Runpod 삭제 응답과 Pod 목록 0개를 확인했고, 연결된 별도 네트워크 볼륨은 없었다.
기존 삭제 감시는 해제했다. 아래의 기존 Pod 연결·재사용·삭제 예약 안내는 삭제 전 기록이다.
최종 제출 파일은 로컬에 보존되어 있으며, 새 대여 지시 없이 Pod를 생성하지 않는다.
확인 기록: `live/pod-shutdown.json`.

이 파일은 사용자와 확정한 작업 방식과 2026-10-10 실제 Runpod 연습에서 확인한 문제를 기록한다.
새 채팅에서 이 저장소를 사용하거나 수정하기 전에 적용한다. 최신 사용자 지시가 우선한다.

## 실제 대회에서 사용할 환경

**실제 대회도 현재 연결된 Runpod GPU를 재사용한다. 새 Pod 대여가 기본 동작이 아니다.**

| 항목 | 현재 설정 |
| --- | --- |
| 대회 시작 설정 | `config.live.yaml` |
| 기존 Pod | `vvpxznt3e1lvyn`, Secure Cloud, US-MD-1 |
| GPU | NVIDIA A100-SXM4-80GB × 3 |
| 검증한 이미지 | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| SSH 키 | `/config/.ssh/runpod_codex_ed25519` (내용 출력·복사 금지) |
| 마지막 확인 SSH | `root@154.54.102.25`, 포트 `14949` (실행 전에 CLI로 다시 조회) |
| 자동 삭제 예약 | 2026-10-10 16:30 KST / 07:30 UTC, 로컬 watchdog |

`config.live.yaml`의 `runpod.existing_pod_id`를 유지한다. `tasks`는 예시이므로 실제 지문과 데이터에
맞춰 교체한다. `config.example.yaml`의 기본 SSH 키 경로는 현재 실제 키와 다르므로 무작정 덮어쓰지 않는다.
Pod 실행 여부, GPU, 포트, 가격, 임대 시각은 변할 수 있다. 이전 성공 기록만으로 현재 접속을 보장하지 않는다.

현재 설치 Python은 `/config/workspace/.venv/bin/python`이다. 셸의 기본 Python을 임의로 사용하지 않는다.
Pod의 하네스 작업은 각 원격 작업 폴더의 `.venv/bin/python`으로 실행한다.

**사용자 확정: 실제 개발·분석·학습은 현재 로컬/Runpod 환경 기준이다.** 아래 예선 사양에 맞추기 위해
Python·PyTorch를 낮추거나 CPU/GPU 자원, 모델 선택, 개발 코드 문법, 라이브러리 사용을 제한하지 않는다.
예선 환경 복제·구버전 포팅을 작업 시작의 선행 조건으로 삼지 않는다. 제출 시 실제로 실행할 코드에 필요한
조정만 최종화 단계에서 처리하며, 로컬/Runpod 풀이 소스와 결과는 그대로 보존한다.

## 실제 온라인 Notebook 환경 — 사용자 제공 커널 진단 출력

2026-10-10 실제 대회 중 사용자가 커널에서 확인한 상세 출력으로 갱신했다.
**사용자 제공 실제 진단 결과이며, 에이전트가 해당 온라인 커널에서 직접 실행한 검증은 아니다.**
전체 버전과 출처는 `submission-environment.json`에 보존한다.

| 항목 | 사용자 제공 실제값 |
| --- | --- |
| Python | 3.9.7, conda-forge, Sep 29 2021 19:20:46, GCC 9.4.0 |
| Python 실행 파일 | `/opt/conda/bin/python` |
| OS | `Linux-6.1.158-180.294.amzn2023.x86_64-x86_64-with-glibc2.31` |
| 예시 작업 폴더 | `/home/jovyan/work/Problem #1` |
| 논리 CPU / affinity | 2 / 2, cgroup v2 cpu.max = `max 100000` |
| RAM | 표시 총량 7.56 GiB, 가용 6.07 GiB, **cgroup 상한 4 GiB (4294967296 bytes)** |
| PyTorch | `1.10.0+cpu` |
| PyTorch CUDA build / CUDA available | `None` / `False` |
| NumPy / pandas / SciPy | `1.23.5` / `1.5.2` / `1.9.3` |
| scikit-learn / joblib | `1.1.3` / `1.2.0` |
| torchvision / TensorFlow | `0.11.1+cpu` / `2.10.0` |
| Pillow / ipykernel | `9.2.0` / `6.9.1` |

`ultralytics`, `timm`, `xgboost`, `lightgbm`, `catboost`, `transformers`, `datasets`,
`accelerate`, `opencv-python`, `opencv-python-headless`는 미설치다.
`nvidia-smi` 실행 파일이 없고 PyTorch GPU 연산 검사는 CUDA 미지원으로 건너뛰었다.
최종 온라인 실행의 메모리 예산은 표시 총량이 아니라 **4 GiB cgroup 제한**으로 잡는다.
파일시스템 여유 공간으로 보고된 `8589929469.46 GiB`는 비정상적인 수치이므로
원문만 보존하고 실제 가용 용량의 근거로 쓰지 않는다.

이 환경은 **온라인 제출·Notebook 실행 환경**이다. 실제 학습/무거운 추론에는 앞서 지정한 기존
Runpod A100 3장을 계속 사용한다. Runpod 이미지를 이 구형 CPU 환경으로 교체하지 않는다.

### 최종 제출 시에만 적용할 호환성 원칙

- 하네스 설치 요구사항은 Python >= 3.10이며 현재 로컬/Runpod에서 실행한다.
  Python 3.9.7 온라인 커널에 `pip install -e .` 또는 `requirements-tested.txt`를 그대로 실행하지 않는다.
  최종 Notebook의 풀이 셀과 실제 필요한 의존성만 별도로 검증한다.
- 실제 온라인 Python 3.9 커널에서 실행할 셀에 한해 해당 문법·API 호환성을 확인한다.
  Notebook에 포함할 로컬/GPU 학습 소스 전체를 Python 3.9로 포팅하라는 뜻이 아니다.
  pandas/NumPy/scikit-learn/torchvision 등
  버전은 위 사용자 제공 진단을 기준으로 하며 **제공되지 않은 버전은 추측하지 않는다.** 필요 패키지는 해당 Python과
  의존성 조합을 확인한 뒤 최소한으로 설치한다. 대회 종료 직전에 전체 환경을 업그레이드하지 않는다.
- Runpod의 PyTorch 2.8/CUDA 12.8과 제출 환경의 PyTorch 1.10 CPU는 다른 환경이다.
  최신 모델 구조·Transformers/YOLO 의존성·`torch.load(..., weights_only=True)` 같은 API 경로·체크포인트
  호환성을 검증 없이 가정하지 않는다. GPU 학습 성공과 온라인 CPU 재추론 성공을 따로 기록한다.
  GPU 전용 API를 온라인에서 무조건 호출하지 않는다.
- 기본 `notebooks.mode: restore`는 저장한 실제 예측을 복원·검증하는 모드다. 이 모드에서도
  pandas/NumPy/PyYAML 등 필요한 의존성·CSV dtype 처리가 온라인 버전에서 동작하는지 확인한다.
  이를 온라인에서 모델 재학습·재추론한 것으로 설명하지 않는다. 실전 지문이 재추론/재학습을 요구하면
  그 요구를 우선하여 CPU 실행 가능 경로를 준비한다. 단순 집계는 전체 풀이를 직접 재실행한다.
- `mode: infer`, `RECOMPUTE=True`, `RETRAIN_CPU=True`는 로컬에서는 현재 환경에 맞춰 사용할 수 있다.
  이 옵션으로 온라인 커널에서 실제 재실행할 경우에만 그 환경에서의 동작을 별도로 확인한다.
  로컬의 Python 3.12 CPU 실행만으로 Python 3.9.7/PyTorch 1.10에서 검증됐다고 쓰지 않는다.
- 온라인 커널에서 수행하는 CPU 작업과 Notebook 재실행만 2개 논리 CPU 및 제출 마감에 맞춘다.
  로컬/Runpod는 각각 실제 가용 자원에 맞춰 스레드·배치·작업량을 정한다.
  온라인에서 장시간 재학습을 기본 제출 절차로 삼지 않는다.
- `/home/jovyan/work/Problem #1`은 한 문항의 예시다. 다른 문항까지 경로를 고정하지 않는다.
  Notebook에서는 `Path.cwd()`와 실제 데이터 폴더를 기준으로 하고, 공백과 `#`가 있는 경로를
  셸에 넘길 때는 인수별로 올바르게 인용한다. CSV는 지문이 지정한 위치·이름으로 저장한다.
- 같은 버전 환경에서 실제 실행하지 않았다면 “온라인 동일 환경 검증 미완료”라고 기록한다.
  Python 3.9 문법 검사, 로컬 실행, 실제 온라인 커널 실행은 서로 다른 검증이다.

## 대회 시작 순서와 시간 관리

1. 사용자에게 실제 대회 시작 지시를 받으면 문제 읽기·다운로드 전에 `start-clock --minutes 90`으로
   시계를 한 번 만든다. 기존 실전 시계가 있으면 재사용한다. 새 채팅·재접속 때문에 90분을 다시 부여하지 않는다.
2. 세 지문, 원본 Notebook, 제출 예시를 읽는다. 실제 파일 위치·인코딩·정답 정의·결측 처리·출력 스키마·
   문항 간 의존 관계를 확인한다. 모델 학습이 필요한지 먼저 판단한다.
   개발·학습은 현재 로컬/Runpod에서 바로 진행한다. 예선 환경에 맞추는 작업으로 풀이 시작을 지연하지 않는다.
   온라인 환경 정보는 제출 최종화 시 실제 실행할 셀을 검증하는 참고값으로 사용한다.
3. 기존 Pod를 `runpodctl pod get vvpxznt3e1lvyn --output json`으로 확인하고 현재 주소로 SSH 접속한다.
   GPU 메모리와 `torch.cuda.is_available()`/장치 개수, 작은 실제 CUDA 연산을 확인한다.
   패키지를 설치한 뒤에도 **작업에 사용하는 동일 Python**으로 검사한다.
4. `live/latest.txt`가 가리키는 상태 파일의 임대 종료·감시 PID와 실제 프로세스를 확인한다.
   대회 종료·결과 회수 전에 삭제되지 않는지 비교한다. 현재 예약은 로컬 감시이므로 서버 측 보장으로 설명하지 않는다.
   시간이 부족하면 즉시 사용자에게 알리고, 문제 분석 등 독립 작업은 계속한다. 승인 없이 예약을 바꾸지 않는다.
5. `config.live.yaml`을 바탕으로 실제 문제용 설정을 만든다. 기존 Pod ID·실제 SSH 키를 유지하고,
   파일 경로·타깃·검증 분할·의존 관계를 지문에 맞춘다. CPU 스레드는 실제 quota 이내로 둔다.
6. `python -m race90 run 실제설정.yaml --clock 실제시계.json`으로 실행한다.
   `plan`/`check`는 GPU 연산 검사가 아니다. GPU 업그레이드 로그와 회수된 후보를 확인해야 원격 성공이다.

기본 마감: 학습 70분, 결과 회수 72분, Notebook 완성 84분, ZIP 검증 85분, 온라인 반영·제출 확인 90분.
단, 이번 실전에서 사용자가 일정을 명시적으로 변경했다. `live/competition-schedule.json`에 따라
가장 최근 사용자의 지시는 남은 **85분**을 정확도 개선 60분, 제출 준비 25분으로 배분하는 것이다.
직전 95분이라는 안내는 사용자가 정정했다. 별도 10분 여유는 없다.
변경 시각은 2026-10-10 05:52:59 UTC이며, 정확도 개선은 06:52:59 UTC / 15:52:59 KST까지,
제출 준비와 전체 마감은 07:17:59 UTC / 16:17:59 KST이다. 실행 검증과 업로드 여유도
마지막 25분 안에 확보한다.
기존 시작 시각은 보존했고 직전 일정은 `live/exam-clock.before-60-25-revision.json`에 보관했다.
Pod 자동 삭제 예약 07:30 UTC는 변경하지 않는다. GPU 결과는 정확도 개선 단계 종료 전에 회수한다.
이후 사용자가 개선 작업 종료와 제출 파일 생성을 지시하여 **06:43:09 UTC에 제출 준비로 전환**했다.
**06:52:41 UTC 최종 파일 준비 완료**: `../제출파일_최종.zip`, `../최종제출/`.
원래 Notebook 파일명과 지문을 유지했고 전체 셀 실행 출력 및 문항별 CSV를 포함한다.
별도 x86_64 환경의 Python 3.9.7 / NumPy 1.23.5 / pandas 1.5.2 / ipykernel 6.9.1,
CPU affinity 2개 / GPU 비노출 / RLIMIT_AS 4 GiB 조건에서 3개 Notebook 모두 실행 성공했다.
1·2번은 저장한 실제 GPU 예측의 후처리, 3번은 실제 CPU 학습·예측이다. 온라인 커널과 제출 접수는 미확인이다.
최종 검증 기록은 `notebook-checks/final/final.json`이다. 예전 `competition-runs` ZIP을 최종 제출본으로 사용하지 않는다.
사용자 추가 요청 전에는 새 개선 실험이나 후보 변경을 시작하지 않는다. 마감 07:17:59와 Pod 삭제 07:30 UTC는 유지한다.
사용자는 제공한 ZIP·지문·환경 정보가 전달받은 자료의 전부라고 확인했다. 자료에 없는
추가 조건을 다시 문의하지 않는다. 특히 1번 테스트의 클래스별 개수는 명시되지 않았으므로
60개씩으로 강제 배분하지 않는다.
모든 문항의 유효 CSV를 일찍 확보하되 Notebook의 본격 제작·실행은 최종화 단계에서 한다.
완료 후보를 보존하고, 마감이 다가오면 `finalize RUN_DIR`로 최종화한다.
`final.json`의 `submission_ready`, `packaged`, `handoff_on_time`, 각 Notebook 실행 여부를 확인한다.
온라인 제공 Notebook 반영과 제출 접수는 별도 작업이다. 파일 생성만으로 온라인 제출 완료라고 말하지 않는다.

## 실제 연습에서 발견한 문제 — 다시 만들지 말 것

### 1. 공식 이미지의 시스템 Python 패키지 설치 차단 (수정됨)

`python -m pip install pandas`가 `externally-managed-environment`로 실패했고, 기본 이미지에는 pandas가 없었다.
`remote.py`는 이제 `python -m venv --system-site-packages <remote_root>/.venv`를 만든 뒤
`remote_python`으로 설치·CUDA 검사·작업·의존 파일 해제·결과 묶기를 수행한다.
기존 PyTorch/CUDA를 공유한다. 시스템 pip로 되돌리거나 불필요하게 PyTorch를 재설치하지 않는다.

회귀 테스트: `test_remote_install_supports_externally_managed_python`.

### 2. 구형 제공 Notebook의 셀 ID 호환성 (수정됨)

연습 Notebook은 nbformat 4.2였다. 새 셀의 `id` 필드를 그대로 추가하면
`NotebookValidationError: Additional properties are not allowed ('id' was unexpected)`가 발생했다.
`notebook.py`는 minor < 5에서 추가 셀의 ID를 제거하고 실행용 Notebook에도 같은 형식 버전을 적용한다.
원본 셀·출력을 유지한다. 새 셀을 추가할 때 최신 형식을 무조건 섞지 않는다.

회귀 테스트: `test_export_preserves_and_validates_older_notebook_formats` (4.0/4.2/4.4/4.5, 4.2 실제 실행 포함).

### 3. 문제 초기 코드는 완성된 제출 코드가 아니다

연습문제에는 Python/R 셀이 섞였고 `./data/`와 실제 파일 위치가 달랐다. Python 3 풀이를 선택하고
지문이 허용한 언어 선택에 따라 R 예제 셀을 제외한 별도 풀이본을 만들었다. 원본 파일은 보존한다.
제출 Notebook은 GPU 없는 온라인 환경에서도 전체 Python 셀을 재실행할 수 있는지 확인한다.
하네스의 추가 셀 실행 검증은 원본의 R·미완성 셀까지 실행해 준다는 뜻이 아니다.
단순 집계는 전체 풀이를 Notebook에서 재계산할 수 있게 한다.

연습 CSV는 `ISO-8859-1`, 제출 CSV는 UTF-8이었다. 인코딩을 모든 문제에 일괄 적용하지 않는다.
`to_csv()` 기본값은 불필요한 index 열을 추가한다. 지문과 샘플의 컬럼·행 순서를 확인하고
이번 문제처럼 index 열을 요구하지 않으면 `index=False`로 저장한다.

### 4. 규칙 기반 집계 문제를 학습 문제로 억지 변환하지 않는다

검증에 사용했던 연습문제는 pm2_5 결측 행을 제거한 뒤 16/36/76 경계로 집계하는 작업이었다.
학습/테스트 데이터와 정답 라벨이 필요한 기본 tabular 어댑터에 맞지 않는다.
문제별 어댑터를 작성해 `kind: custom`으로 실행했다. 해당 연습 소스는 사용자 요청으로 삭제했으므로
실전에서는 지문에 맞는 어댑터를 작성한다.
실제 제출 정답은 지문에 지정된 경계·순서를 따르고 외부의 최신 WHO 기준으로 임의 변경하지 않는다.
GPU와 독립 Boolean 조건 CPU 구현의 결과·CSV 해시가 일치하는지 확인했다.
경계값과 결측값을 테스트한다. pandas 3의 읽기 전용 NumPy 배열을 Torch로 넘길 때는 필요한 복사도 고려한다.

이 문제의 7,304개 유효 값은 로컬 CPU로도 약 0.5초에 처리됐다. 작은 집계 때문에 GPU 3장을 모두 쓰지 않는다.
GPU가 유리한 모델링은 기존 A100을 우선 사용한다. 사용자 정의 점수 1.0은 규칙 검사 통과이며 모델 정확도가 아니다.

### 5. Pod 생성 GraphQL 요청 차단 (자동 생성 경로는 미해결)

이 환경에서 직접 GraphQL 요청은 HTTP 403 `error code: 1010`으로 차단됐다.
리소스 목록으로 생성되지 않았음을 확인한 후 Runpod v2 MCP로 기존 Pod를 만들었다.
**현재 검증한 실전 경로는 `existing_pod_id`로 기존 Pod에 연결하는 방식이다.**
자동 생성 경로도 고쳐졌다고 주장하지 않는다. 생성 요청이 실패/시간 초과되면 실제 Pod 목록을 먼저 확인하고
중복 생성·과금을 피한다. 이번 대회는 기존 GPU 재사용이 사용자 지시이므로 이 생성 경로를 호출할 필요가 없다.

## 검증 기록

- 실제 GPU 하네스 성공 기록: `practice-runs/20261010-044209-34cb49/final.json`.
  기존 Pod 접속·가상환경 설치·GPU 집계·회수·실행 Notebook·ZIP까지 약 70초, `submission_ready: true`.
- 연습 데이터·풀이·설정 폴더는 사용자 요청으로 삭제했다. 위 실행 기록은 과거 검증 증거이며
  삭제된 입력으로 재실행하거나 실전 실행을 `--resume`하지 않는다.
  실전은 `config.live.yaml`을 바탕으로 실제 데이터와 90분 시계를 사용한다.
- 회귀 테스트: `/config/workspace/.venv/bin/python -m pytest -q` (저장소 폴더에서 실행), 2026-10-10 기준 50개 통과.
  실제 대회 시작 후 불필요하게 전체 테스트를 반복해 시간을 쓰지 않는다. 관련 코드가 바뀐 경우 필요한 검사를 수행한다.
- `VALIDATION.json`과 연습 검증 기록을 보되 단순 집계 성공을 이미지·텍스트·YOLO 학습까지 검증한 것으로 확대하지 않는다.

## 배포본

작업에 쓰는 코드는 이 저장소이며 `../race90-review-backup/`은 수정 전 백업이다.
`../race90-ready.zip`을 갱신할 때 이 `AGENTS.md`와 회귀 테스트를 포함한다.
`.env`, SSH 키, 실제 연결 설정, `live/`, 데이터·실행 산출물은 일반 하네스 배포 ZIP에서 제외한다.

## 최종 사용자 결정 및 최신 파일

CPU 재학습 대안 비교 후 사용자가 정확도 저하를 이유로 재학습을 취소하고 기존 계산 결과를 유지하라고 했다.
진행 중인 CPU 비교 학습은 중단했다. Notebook 열기 문제를 줄이기 위해 코드·예측 데이터를 분리했다.
2026-10-10 07:03:29 UTC 기준 `../최종제출/N번/`과 `../제출파일_최종.zip`이 최신 분리본이다.
각 폴더의 작은 .ipynb와 `prediction_data_qN.json.gz`를 함께 올려 실행한다. .gz는 풀지 않는다.
`solution_qN.py` 전체를 원본 Notebook의 빈 코드 셀에 붙여넣어 그 셀을 실행해도 된다.
또는 .py도 같은 폴더에 올리고 `%run ./solution_qN.py`를 실행한다. 기존 data 폴더는 필요하다.
실행 결과는 기존 확정 CSV 3개와 SHA256까지 동일하며, 검사 기록은 `notebook-checks/split/final.json`이다.
이전 큰 Notebook과 ZIP은 `notebook-checks/restore-version/`에 보존했다. 실제 온라인 제출 접수는 미확인이다.
