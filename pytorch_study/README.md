# pytorch_study

PyTorch 딥러닝 학습 노트 — 텐서 기초부터 Transformer/CV 심화까지. 리포에서 가장 큰 폴더이며, `NN_phase/MM_topic/` 형태의 번호 매긴 폴더로 구성된다.

## 폴더 구성 규칙

각 주제 폴더(`MM_topic/`)는 다음 3파일 세트로 구성된다:

- `concept.md` — 이론 정리 (한 줄 요약, 핵심 개념, 원리/수식)
- `example.ipynb` — 최소 예제(교재 스타일 토이 데이터)
- `practice.ipynb` — Kaggle/LeetGPU 등 실제 데이터셋을 쓴 응용 실습 (첫 셀에 링크 + 문제 정의)
- (선택) `img/` — concept.md에서 참조하는 구조도/설명 이미지

새 주제를 추가할 때도 이 3파일 세트 패턴을 따른다.

## 테마(Phase) 목록

| 폴더 | 테마 | 루트 README Phase |
|------|------|------|
| `01_basics/` | 텐서 기초 | Phase 2 |
| `02_regression/` | 선형 회귀 | Phase 2 |
| `03_classification/` | 분류 (로지스틱/소프트맥스) | Phase 2 |
| `04_neural_network/` | 퍼셉트론·MLP·MNIST 입문 | Phase 2 |
| `05_training_techniques/` | ReLU·초기화·드롭아웃·배치정규화 | Phase 2 |
| `06_cnn/` | CNN·VGG·ResNet·EfficientNet | Phase 3 |
| `07_rnn/` | RNN·문자예측·시계열·LSTM/GRU·NLP | Phase 5 |
| `08_transformer/` | Attention·Transformer·BERT/GPT·RAG | Phase 6 |
| `09_cv_advanced/` | Object Detection·Segmentation·생성모델 | Phase 7 |
| `supplementary/` | 보강 학습 (A.핵심 메커니즘 · B.일반화·평가 · C.실전 운용) | — |

각 주제별 진도(✅/⬜)와 실습 링크는 루트 [README.md](/README.md#학습-현황)의 표에서 관리된다 — 여기서 중복 나열하지 않음.

## 기타 파일

- `self_qna.md` — ChatGPT/구글링 기반 셀프 Q&A 모음 (예: CNN vs Transformer)
- 리소스(모두 `.gitignore` 대상, 노트북 실행 시 자동 생성/다운로드): `MNIST_data/`, `cifar10/`, `custom_data/`, `model/`
- `images/` — concept.md/노트북에서 참조하는 이미지 (gitignore 대상 아님)
