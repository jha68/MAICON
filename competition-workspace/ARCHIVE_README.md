# 대회 작업공간 전체 보관본

이 폴더는 `/config/workspace`에 보존되어 있던 대회 자료를 새 브랜치에 모은 보관본입니다. 기존 저장소의 MAICON 자료는 상위 폴더에 그대로 유지합니다.

## 주요 파일

- [1번 레이더 문제 풀이 전체 과정](1번_레이더_문제풀이_전체과정.md): 문제 해석, 특징 설계, 모델 비교, 시행착오, 최종 앙상블 및 제출 과정
- [최종 제출 폴더](최종제출/): 문항별 Notebook, 실행 코드, 예측 데이터, 제출 CSV
- [최종 제출 ZIP](제출파일_최종.zip)
- [race90](race90/): 대회 하네스, 학습·추론 소스, 원본 데이터, 실험 산출물, 실행·검증 기록
- [수정 전 백업](race90-review-backup/)
- [전체 파일 목록과 SHA256](WORKSPACE_ARCHIVE_MANIFEST.json)

원본 문제 ZIP, 루트의 실행 Notebook과 예측 데이터, 연습 및 실전 실행 로그도 함께 포함했습니다. 로컬 가상환경 `.venv`, Python·pytest 캐시, 설치 메타데이터 및 인증용 파일은 제외합니다. 제외한 경로는 전체 파일 목록의 `excluded` 항목에 기록되어 있습니다.

## 대용량 가중치 복원

`race90/competition/q1/models-selected.tgz`는 원본이 약 3.06GB로 GitHub의 일반 Git 단일 파일 제한을 넘습니다. 원본 바이트를 48MiB씩 나눠 `models-selected.tgz.parts/`에 보존했습니다. Git LFS나 추가 다운로드 없이 이 브랜치를 클론하면 모든 조각이 함께 내려옵니다.

이 폴더에서 다음을 실행하면 각 조각과 합친 파일의 SHA256을 검사한 뒤 원래 이름으로 복원합니다.

```bash
bash restore_large_files.sh
```

조각별 순서·크기·SHA256 및 원본 전체의 SHA256은 `models-selected.tgz.parts/parts-manifest.json`에도 있습니다. 복원에는 원본 크기만큼의 추가 디스크 공간이 필요합니다. 복원된 3GB 파일은 실수로 일반 Git에 추가하지 않도록 상위 `.gitignore`에 등록했습니다.

전체 보관 파일과 분할 원본을 검증하려면 Node.js가 있는 환경에서 다음을 실행합니다.

```bash
node verify_archive.mjs
```

## 보존 방식

일반 파일은 내용을 변경하지 않고 복사했습니다. Notebook 검증 폴더의 데이터 링크 9개는 다른 환경에서도 동작하도록 작업공간 내부의 절대경로를 상대경로로 바꿨고, 이전 링크와 새 링크를 파일 목록의 `symlinks`에 기록했습니다.

GPU 학습을 새로 실행하거나, 최종 후보·예측값·제출 CSV를 변경하지 않았습니다. 당시 Runpod는 종료·삭제된 상태이며 현재 실행 중인 GPU를 포함하는 보관본이 아닙니다.
