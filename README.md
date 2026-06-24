# python

- 백엔드 개발자 → AI 엔지니어 직무 전환을 위한 학습 리포지토리

> `pytorch_study/`의 각 주제는 **개념(`concept.md`) · 예제(`example.ipynb`) · 실습(`practice.ipynb`)** 1세트로 구성됩니다.
> 실습 노트북 첫 셀에는 검증된 LeetGPU/Kaggle 링크와 문제 정의가 들어 있습니다.

---

## 학습 현황

### Phase 1. Python 기초 및 ML 입문

- [x] Python 기본 문법 학습 ([basis/lessons.md](/basis/lessons.md))
- [x] 핸즈온 머신러닝 2장 - End-to-End ML 프로젝트 ([california.ipynb](/hands_on_machine_learning/california.ipynb))
  - 데이터 수집, 탐색, 전처리, 모델 훈련, 평가
- [x] Kaggle - 타이타닉 competition ([titanic.py](/kaggle/titanic.py))

### Phase 2. 딥러닝 기초 (PyTorch) ✅

각 주제: `개념` = concept.md, 예제/실습 노트북은 같은 디렉토리에 위치.

**01_basics — 기초**

| 주제 | 개념 | 핵심 내용 | 실습 |
|---|---|---|---|
| 텐서 조작 | [concept](/pytorch_study/01_basics/01_tensor_manipulation/concept.md) | 생성·shape·브로드캐스팅, 요소곱 vs 행렬곱, view/reshape/squeeze, cat/stack, mean/max | [LeetGPU](https://leetgpu.com/challenges) |

**02_regression — 선형 회귀**

| 주제 | 개념 | 핵심 내용 | 실습 |
|---|---|---|---|
| 선형 회귀 | [concept](/pytorch_study/02_regression/01_linear_regression/concept.md) | `H=Wx+b`, MSE, SGD | [Kaggle: House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| 다변수 회귀 | [concept](/pytorch_study/02_regression/02_multivariable_regression/concept.md) | 행렬곱, `nn.Linear` | [Kaggle: House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| 경사하강법 | [concept](/pytorch_study/02_regression/03_gradient_descent/concept.md) | 미분, 학습률, `optim.SGD` | [LeetGPU](https://leetgpu.com/challenges) |
| 데이터 로딩 | [concept](/pytorch_study/02_regression/04_data_loading/concept.md) | `Dataset`/`DataLoader`, 미니배치 | [Kaggle: House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) · [Student Marks](https://www.kaggle.com/datasets/yasserh/student-marks-dataset) |

**03_classification — 분류**

| 주제 | 개념 | 핵심 내용 | 실습 |
|---|---|---|---|
| 로지스틱 회귀 | [concept](/pytorch_study/03_classification/01_logistic_regression/concept.md) | 이진분류, 시그모이드, BCE | [Kaggle: Titanic](https://www.kaggle.com/c/titanic) |
| 소프트맥스 분류 | [concept](/pytorch_study/03_classification/02_softmax_classification/concept.md) | 다중분류, 소프트맥스, 교차 엔트로피 | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) |

**04_neural_network — 신경망**

| 주제 | 개념 | 핵심 내용 | 실습 |
|---|---|---|---|
| 퍼셉트론 | [concept](/pytorch_study/04_neural_network/01_perceptron/concept.md) | 선형 분류기, XOR의 한계 | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) · [LeetGPU](https://leetgpu.com/challenges) |
| 다층 퍼셉트론 (MLP) | [concept](/pytorch_study/04_neural_network/02_multilayer_perceptron/concept.md) | 은닉층, 역전파(연쇄법칙), XOR 해결 | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) · [LeetGPU](https://leetgpu.com/challenges) |
| MNIST 입문 | [concept](/pytorch_study/04_neural_network/03_mnist_intro/concept.md) | epoch/batch/iteration, `CrossEntropyLoss` | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) |

**05_training_techniques — 학습 기법**

| 주제 | 개념 | 핵심 내용 | 실습 |
|---|---|---|---|
| ReLU / 활성화 | [concept](/pytorch_study/05_training_techniques/01_relu_activation/concept.md) | 기울기 소실, ReLU, optimizer 비교 | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) · [LeetGPU](https://leetgpu.com/challenges) |
| 가중치 초기화 | [concept](/pytorch_study/05_training_techniques/02_weight_initialization/concept.md) | Xavier, He | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) · [LeetGPU](https://leetgpu.com/challenges) |
| 드롭아웃 | [concept](/pytorch_study/05_training_techniques/03_dropout/concept.md) | 과적합 방지, train/eval 모드 | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) |
| 배치 정규화 | [concept](/pytorch_study/05_training_techniques/04_batch_normalization/concept.md) | internal covariate shift | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) |
| 학습 팁 | [concept](/pytorch_study/05_training_techniques/05_tips/concept.md) | 학습률, 전처리/정규화, 과적합 종합 | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) |

### Phase 3. CNN (Convolutional Neural Network) ✅

| 주제 | 개념 | 핵심 내용 | 실습 |
|---|---|---|---|
| 합성곱 연산 | [concept](/pytorch_study/06_cnn/01_convolution/concept.md) | 필터/스트라이드/패딩, 풀링 | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) · [LeetGPU](https://leetgpu.com/challenges) |
| MNIST CNN | [concept](/pytorch_study/06_cnn/02_mnist_cnn/concept.md) | CNN 레이어 구성, MNIST 분류 | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) · [LeetGPU](https://leetgpu.com/challenges) |
| VGG | [concept](/pytorch_study/06_cnn/03_vgg/concept.md) | 3×3 conv 스택 | [Kaggle: Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) |
| ResNet | [concept](/pytorch_study/06_cnn/04_resnet/concept.md) | 잔차 연결(residual), 깊은 신경망 | [Kaggle: Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) |
| ImageFolder | [concept](/pytorch_study/06_cnn/05_image_folder/concept.md) | 커스텀 이미지 데이터셋 | [Kaggle: Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) |
| Visdom 시각화 | [concept](/pytorch_study/06_cnn/06_visdom/concept.md) | 학습 과정 시각화 | [Visdom](https://github.com/fossasia/visdom) · [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) |

- [ ] EfficientNet 구현

### Phase 4. GPU 프로그래밍 (CUDA / LeetGPU)

- [x] Color Inversion ([color_inversion](/leetgpu/color_inversion.ipynb))
- [x] Matrix Transpose ([matrix_transpose](/leetgpu/matrix_transpose.ipynb))
- [x] Matrix Multiplication ([matrix_multiplication](/leetgpu/matrix_multiplication.ipynb))
- [ ] Tiled Matrix Multiplication (공유 메모리 최적화)
- [ ] Softmax CUDA 커널

### Phase 5. RNN / 시퀀스 모델 ✅(기초)

| 주제 | 개념 | 핵심 내용 | 실습 |
|---|---|---|---|
| RNN 기초/구조 | [concept](/pytorch_study/07_rnn/01_intro/concept.md) | 셀, 입출력 형태 | [Kaggle: Shakespeare](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays) |
| `nn.RNN` 사용 | [concept](/pytorch_study/07_rnn/02_basics/concept.md) | 시퀀스 차원 다루기 | [Kaggle: Shakespeare](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays) |
| 문자 예측 (hihello) | [concept](/pytorch_study/07_rnn/03_hihello/concept.md) | one-hot, 교차 엔트로피 | [Kaggle: Shakespeare](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays) |
| 긴 시퀀스 (char-RNN) | [concept](/pytorch_study/07_rnn/04_longseq/concept.md) | 윈도우, 긴 텍스트 | [Kaggle: Shakespeare](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays) |
| 시계열 예측 | [concept](/pytorch_study/07_rnn/05_time_series/concept.md) | LSTM, 정규화, 주가 예측 | [Kaggle: Store Sales](https://www.kaggle.com/c/store-sales-time-series-forecasting) · [US Stocks](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) |

- [ ] LSTM / GRU **내부 게이트 구조** 정리 (현재는 사용 위주) · bidirectional RNN
- [ ] `nn.Embedding` 기반 시퀀스 (one-hot → 임베딩)
- [ ] Seq2Seq & Attention 기초 (Transformer로 가는 다리)
- [ ] 자연어 처리 기초 (텍스트 분류, 감성 분석)

### Phase 6. Transformer & LLM

- [ ] Attention 메커니즘 이해 (self-attention, multi-head, positional encoding)
- [ ] Transformer 구조 구현
- [ ] BERT(인코더) / GPT(디코더) 구조 차이 이해
- [ ] HuggingFace Transformers 사용 (tokenizer, `AutoModel`, fine-tuning)
- [ ] LLM 파인튜닝 (LoRA / QLoRA / PEFT)
- [ ] RAG (Retrieval-Augmented Generation) 시스템 구축

### Phase 7. Computer Vision 심화

- [ ] Object Detection (YOLO, Faster R-CNN)
- [ ] Semantic Segmentation (U-Net, Mask R-CNN)
- [ ] 최신 백본 (EfficientNet, Vision Transformer)
- [ ] Image Generation (GAN, VAE, Diffusion Model 개념)

### Phase 8. MLOps & 모델 서빙

- [ ] 모델 저장 및 로딩 (ONNX, TorchScript)
- [ ] FastAPI로 모델 서빙
- [ ] Docker 컨테이너화
- [ ] MLflow 실험 관리
- [ ] DVC 데이터 버전 관리

### Phase 9. 클라우드 & AI 파이프라인

- [ ] AWS SageMaker 또는 GCP Vertex AI
- [ ] 클라우드를 이용한 모델 학습 파이프라인
- [ ] 모델 모니터링 및 운영

### Phase 10. 포트폴리오 & 실전

- [ ] Kaggle 대회 참여 (점수 개선)
- [ ] 개인 프로젝트 (백엔드 경험 + AI 결합)
- [ ] 논문 구현 1개 이상
- [ ] GitHub 포트폴리오 정리

---

## 🔧 보강 학습 (현재 토대 강화)

모델 구조 위주의 현재 커리큘럼에서 **빠져 있지만, 모델을 제대로 학습·평가·운용하려면 반드시 필요한** 기반 지식입니다.
Phase 2~5와 병행해 채우면 좋습니다.

### A. 학습의 핵심 메커니즘
- [ ] **Autograd 심화** — 계산 그래프, `backward()` 원리, `requires_grad`/`detach()`/`no_grad()`, `zero_grad()`가 필요한 이유
- [ ] **`nn.Module` 구조 심화** — `nn.Sequential`/`ModuleList`, 커스텀 레이어, `parameters()`/`named_parameters()`
- [ ] **Optimizer 심화** — Momentum, RMSprop, **Adam/AdamW**, weight decay(L2), `lr_scheduler`
- [ ] **손실 함수 정리** — MSE/MAE, `BCEWithLogitsLoss`, `CrossEntropyLoss`(로짓 입력) 선택 기준

### B. 일반화 & 평가
- [ ] **규제** — L1/L2, dropout, **early stopping**, data augmentation의 관계
- [ ] **평가 지표** — accuracy의 한계, precision/recall/F1, confusion matrix, ROC-AUC, RMSE/R²
- [ ] **데이터 분할 & 검증** — train/val/test, k-fold, 데이터 누수(leakage) 방지
- [ ] **전처리/증강** — `torchvision.transforms`, normalize, 이미지 augmentation, 불균형 데이터

### C. 실전 운용
- [ ] **GPU/디바이스 관리** — `.to(device)`, CPU/GPU 텐서 혼용 오류 디버깅
- [ ] **모델 저장/로드** — `state_dict`, checkpoint(옵티마이저 포함), 추론 시 `model.eval()`
- [ ] **재현성 & 모니터링** — seed 고정, **TensorBoard**(visdom 표준 대안)·`wandb`
- [ ] **전이학습** — `torchvision.models` 사전학습 모델, feature extractor vs fine-tuning (VGG/ResNet의 다음 단계)

---

## 디렉토리 구조

```
python/
├── basis/                        # Python 기초 문법
├── hands_on_machine_learning/    # 핸즈온 ML 교재 실습
├── kaggle/                       # Kaggle 대회 풀이
├── leetgpu/                      # CUDA GPU 프로그래밍
├── pytorch_study/                # PyTorch 딥러닝 (테마 > 주제 > concept/example/practice)
│   ├── 01_basics/                #   텐서 기초
│   ├── 02_regression/            #   선형 회귀
│   ├── 03_classification/        #   분류
│   ├── 04_neural_network/        #   퍼셉트론·MLP·MNIST
│   ├── 05_training_techniques/   #   ReLU·초기화·드롭아웃·배치정규화·팁
│   ├── 06_cnn/                   #   CNN·VGG·ResNet·데이터·시각화
│   ├── 07_rnn/                   #   RNN·문자예측·시계열
│   ├── self_qna.md               #   CNN/Transformer 셀프 Q&A
│   ├── images/  MNIST_data/      #   원본 이미지·데이터 캐시
└── algorithm_solver/             # 알고리즘 분류 도구
```

---

## 참고 자료

### 강의 / 교재 (링크 검증 완료)
- [PyTorch 공식 튜토리얼](https://docs.pytorch.org/tutorials/) — autograd·저장/로드·전이학습·배포 공식 예제 (보강 학습에 최적)
- [Dive into Deep Learning (d2l.ai)](https://d2l.ai/) — 수식+PyTorch 코드 무료 교재, 거의 모든 주제 포괄
- [Stanford CS231n](https://cs231n.github.io/) — CNN·컴퓨터 비전 명강의 노트
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) — Transformers·NLP 실습 코스
- [fast.ai Practical Deep Learning](https://course.fast.ai/) — 톱다운 실전 위주 코스

### 원본 커리큘럼 / 도구
- [핸즈온 머신러닝 3판](https://github.com/ageron/handson-ml3)
- [모두를 위한 딥러닝 (PyTorchZeroToAll)](https://github.com/hunkim/PyTorchZeroToAll)
- [LeetGPU](https://leetgpu.com)
- [HuggingFace 문서](https://huggingface.co/docs)
- [fast.ai](https://www.fast.ai)
