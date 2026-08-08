# python (AI 엔지니어 전환 학습 리포지토리)

백엔드 개발자 → AI 엔지니어 직무 전환을 위한 개인 학습 리포지토리. 주제별 폴더에 Jupyter 노트북/스크립트를 쌓아가는 형태이며, 별도 앱·서비스·빌드 파이프라인은 없다.

<!-- 이 파일은 지도다. 상세 설명을 늘리지 말고 포인터를 유지할 것. -->

## 디렉토리 지도

| 경로 | 역할 |
|------|------|
| `basis/` | 파이썬 기초 문법 Q&A 노트 |
| `hands_on_machine_learning/` | 핸즈온 머신러닝 교재 실습 |
| `kaggle/` | Kaggle 대회 풀이 |
| `leetcode/` | 알고리즘 문제 풀이 (날짜별 파일) |
| `leetgpu/` | CUDA GPU 프로그래밍 실습 (LeetGPU) |
| `pytorch_study/` | PyTorch 딥러닝 기초~CNN~RNN (가장 큰 폴더, `rnn/` 하위 포함) |
| `algorithm_solver/` | 알고리즘 문제 분류용 스크립트 (nltk/sklearn 기반) |
| `tensorflow_basic/` | TensorFlow 기초 예제 |
| `tutorial/` | Kaggle 튜토리얼 (Melbourne housing) |
| `statics/` | 노트북에서 참조하는 이미지·MNIST 원본 데이터 리소스 |
| `README.md` | Phase 1~10 학습 로드맵 + 체크리스트 (사람이 읽는 진행 현황판) |

각 폴더의 세부 내용은 폴더별 `README.md`를 참고할 것 (있는 폴더에 한함).

## 명령어

<!-- TODO(확인 필요): requirements.txt/pyproject.toml 등 의존성 매니페스트가 트래킹되어 있지 않음.
     .venv(Python 3.13)는 존재하지만 설치된 패키지 목록을 코드에서 확정할 수 없음.
     새 스크립트를 실행하기 전에는 필요한 패키지(import 문 기준: torch, sklearn, nltk, pandas, tensorflow 등)가
     .venv에 설치돼 있는지 먼저 확인할 것. -->

- 빌드/테스트/린트: 없음 (테스트 스위트, CI, 린터 미설정)
- 노트북 실행: Jupyter로 `.ipynb` 파일을 직접 열어 셀 단위 실행
- 스크립트 실행: `python3 <path>.py` (예: `python3 tutorial/basic_data_exploration.py`, 단 실행 위치의 상대경로에 유의 — 아래 참고)

## 핵심 규칙 · 하지 말 것

- `.gitignore`로 대용량 산출물(`hands_on_machine_learning/datasets`, `hands_on_machine_learning/images`, `pytorch_study/cifar10`, `pytorch_study/MNIST_data`, `pytorch_study/custom_data`, `pytorch_study/model`)이 제외되어 있음 — 이 경로들은 로컬 실행 시 자동 생성/다운로드되는 데이터이므로 커밋 대상이 아니다.
- 스크립트 안의 상대경로(`pd.read_csv('gpascore.csv')`, `melb_data.csv` 등)는 **해당 폴더를 cwd로 실행**하는 것을 전제로 작성됨. 리포 루트에서 바로 실행하면 파일을 못 찾는다.
- `leetcode/`의 파일명은 `YYYY_MM_DD` 형식(풀이한 날짜)을 따른다. 신규 항목 추가 시 [새 학습 항목 추가하기](docs/runbooks/새-학습-항목-추가.md) 참고.
- 루트 `README.md`의 Phase 체크리스트는 실제 진행 상황을 반영하는 문서이므로, 새 학습 결과물을 추가할 때 함께 갱신하는 편이 좋다(단, 매 커밋마다 강제되는 규칙은 아니며 과거 커밋에서도 종종 생략됨).

## 더 볼 문서 (필요할 때만)

- 반복 작업 절차: `docs/runbooks/`
- 사람용 학습 로드맵/현황: `README.md`
