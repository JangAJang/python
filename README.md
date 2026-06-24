# python

- 백엔드 개발자 → AI 엔지니어 직무 전환을 위한 학습 리포지토리

> `pytorch_study/`의 각 주제는 **개념(`concept.md`) · 예제(`example.ipynb`) · 실습(`practice.ipynb`)** 1세트로 구성됩니다.
> 실습 노트북 첫 셀에는 검증된 LeetGPU/Kaggle 링크와 문제 정의가 들어 있습니다.
> 아래 표의 **진도** 열에서 ✅는 학습 완료, ⬜는 미완료를 의미합니다.

---

## 학습 현황

### Phase 1. Python 기초 및 ML 입문

| 진도 | 주제 | 개념 | 핵심 내용 | 실습 |
|:---:|---|---|---|---|
| ✅ | Python 기본 문법 | [lessons.md](/basis/lessons.md) | 기본 문법 학습 | — |
| ✅ | 핸즈온 ML 2장 | — | End-to-End ML 프로젝트 (수집·탐색·전처리·훈련·평가) | [california.ipynb](/hands_on_machine_learning/california.ipynb) |
| ✅ | Kaggle 타이타닉 | — | 이진분류 대회 풀이 | [titanic.py](/kaggle/titanic.py) · [Kaggle](https://www.kaggle.com/c/titanic) |

### Phase 2. 딥러닝 기초 (PyTorch)

**01_basics — 기초**

| 진도 | 주제 | 개념 | 핵심 내용 | 실습 |
|:---:|---|---|---|---|
| ✅ | 텐서 조작 | [concept](/pytorch_study/01_basics/01_tensor_manipulation/concept.md) | 생성·shape·브로드캐스팅, 요소곱 vs 행렬곱, view/reshape/squeeze, cat/stack, mean/max | [practice](/pytorch_study/01_basics/01_tensor_manipulation/practice.ipynb) · [LeetGPU: Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose) |

**02_regression — 선형 회귀**

| 진도 | 주제 | 개념 | 핵심 내용 | 실습 |
|:---:|---|---|---|---|
| ✅ | 선형 회귀 | [concept](/pytorch_study/02_regression/01_linear_regression/concept.md) | `H=Wx+b`, MSE, SGD | [practice](/pytorch_study/02_regression/01_linear_regression/practice.ipynb) · [Kaggle: House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| ✅ | 다변수 회귀 | [concept](/pytorch_study/02_regression/02_multivariable_regression/concept.md) | 행렬곱, `nn.Linear` | [practice](/pytorch_study/02_regression/02_multivariable_regression/practice.ipynb) · [Kaggle: House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| ✅ | 경사하강법 | [concept](/pytorch_study/02_regression/03_gradient_descent/concept.md) | 미분, 학습률, `optim.SGD` | [practice](/pytorch_study/02_regression/03_gradient_descent/practice.ipynb) · [LeetGPU: Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication) |
| ✅ | 데이터 로딩 | [concept](/pytorch_study/02_regression/04_data_loading/concept.md) | `Dataset`/`DataLoader`, 미니배치 | [practice](/pytorch_study/02_regression/04_data_loading/practice.ipynb) · [Kaggle: House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) · [Student Marks](https://www.kaggle.com/datasets/yasserh/student-marks-dataset) |

**03_classification — 분류**

| 진도 | 주제 | 개념 | 핵심 내용 | 실습 |
|:---:|---|---|---|---|
| ✅ | 로지스틱 회귀 | [concept](/pytorch_study/03_classification/01_logistic_regression/concept.md) | 이진분류, 시그모이드, BCE | [practice](/pytorch_study/03_classification/01_logistic_regression/practice.ipynb) · [Kaggle: Titanic](https://www.kaggle.com/c/titanic) |
| ✅ | 소프트맥스 분류 | [concept](/pytorch_study/03_classification/02_softmax_classification/concept.md) | 다중분류, 소프트맥스, 교차 엔트로피 | [practice](/pytorch_study/03_classification/02_softmax_classification/practice.ipynb) · [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) |

**04_neural_network — 신경망**

| 진도 | 주제 | 개념 | 핵심 내용 | 실습 |
|:---:|---|---|---|---|
| ✅ | 퍼셉트론 | [concept](/pytorch_study/04_neural_network/01_perceptron/concept.md) | 선형 분류기, XOR의 한계 | [practice](/pytorch_study/04_neural_network/01_perceptron/practice.ipynb) · [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) · [LeetGPU: Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication) |
| ✅ | 다층 퍼셉트론 (MLP) | [concept](/pytorch_study/04_neural_network/02_multilayer_perceptron/concept.md) | 은닉층, 역전파(연쇄법칙), XOR 해결 | [practice](/pytorch_study/04_neural_network/02_multilayer_perceptron/practice.ipynb) · [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) · [LeetGPU: Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication) |
| ✅ | MNIST 입문 | [concept](/pytorch_study/04_neural_network/03_mnist_intro/concept.md) | epoch/batch/iteration, `CrossEntropyLoss` | [practice](/pytorch_study/04_neural_network/03_mnist_intro/practice.ipynb) · [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) |

**05_training_techniques — 학습 기법**

| 진도 | 주제 | 개념 | 핵심 내용 | 실습 |
|:---:|---|---|---|---|
| ✅ | ReLU / 활성화 | [concept](/pytorch_study/05_training_techniques/01_relu_activation/concept.md) | 기울기 소실, ReLU, optimizer 비교 | [practice](/pytorch_study/05_training_techniques/01_relu_activation/practice.ipynb) · [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) · [LeetGPU: ReLU](https://leetgpu.com/challenges/relu) |
| ✅ | 가중치 초기화 | [concept](/pytorch_study/05_training_techniques/02_weight_initialization/concept.md) | Xavier, He | [practice](/pytorch_study/05_training_techniques/02_weight_initialization/practice.ipynb) · [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) · [LeetGPU: Leaky ReLU](https://leetgpu.com/challenges/leaky-relu) |
| ✅ | 드롭아웃 | [concept](/pytorch_study/05_training_techniques/03_dropout/concept.md) | 과적합 방지, train/eval 모드 | [practice](/pytorch_study/05_training_techniques/03_dropout/practice.ipynb) · [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) |
| ✅ | 배치 정규화 | [concept](/pytorch_study/05_training_techniques/04_batch_normalization/concept.md) | internal covariate shift | [practice](/pytorch_study/05_training_techniques/04_batch_normalization/practice.ipynb) · [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) |
| ✅ | 학습 팁 | [concept](/pytorch_study/05_training_techniques/05_tips/concept.md) | 학습률, 전처리/정규화, 과적합 종합 | [practice](/pytorch_study/05_training_techniques/05_tips/practice.ipynb) · [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) |

### Phase 3. CNN (Convolutional Neural Network)

| 진도 | 주제 | 개념 | 핵심 내용 | 실습 |
|:---:|---|---|---|---|
| ✅ | 합성곱 연산 | [concept](/pytorch_study/06_cnn/01_convolution/concept.md) | 필터/스트라이드/패딩, 풀링 | [practice](/pytorch_study/06_cnn/01_convolution/practice.ipynb) · [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) · [LeetGPU: 2D Convolution](https://leetgpu.com/challenges/2d-convolution) |
| ✅ | MNIST CNN | [concept](/pytorch_study/06_cnn/02_mnist_cnn/concept.md) | CNN 레이어 구성, MNIST 분류 | [practice](/pytorch_study/06_cnn/02_mnist_cnn/practice.ipynb) · [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) · [LeetGPU: 2D Convolution](https://leetgpu.com/challenges/2d-convolution) |
| ✅ | VGG | [concept](/pytorch_study/06_cnn/03_vgg/concept.md) | 3×3 conv 스택 | [practice](/pytorch_study/06_cnn/03_vgg/practice.ipynb) · [Kaggle: Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) |
| ✅ | ResNet | [concept](/pytorch_study/06_cnn/04_resnet/concept.md) | 잔차 연결(residual), 깊은 신경망 | [practice](/pytorch_study/06_cnn/04_resnet/practice.ipynb) · [Kaggle: Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) |
| ✅ | ImageFolder | [concept](/pytorch_study/06_cnn/05_image_folder/concept.md) | 커스텀 이미지 데이터셋 | [practice](/pytorch_study/06_cnn/05_image_folder/practice.ipynb) · [Kaggle: Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) |
| ✅ | Visdom 시각화 | [concept](/pytorch_study/06_cnn/06_visdom/concept.md) | 학습 과정 시각화 | [practice](/pytorch_study/06_cnn/06_visdom/practice.ipynb) · [Visdom](https://github.com/fossasia/visdom) · [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) |
| ⬜ | EfficientNet | [concept](/pytorch_study/06_cnn/07_efficientnet/concept.md) | compound scaling 기반 효율적 백본 | [Kaggle: Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) |

### Phase 4. GPU 프로그래밍 (CUDA / LeetGPU)

| 진도 | 주제 | 개념 | 핵심 내용 | 실습 |
|:---:|---|---|---|---|
| ✅ | Color Inversion | — | 픽셀 반전 커널 | [notebook](/leetgpu/color_inversion.ipynb) · [LeetGPU: Color Inversion](https://leetgpu.com/challenges/color-inversion) |
| ✅ | Matrix Transpose | — | 행렬 전치 커널 | [notebook](/leetgpu/matrix_transpose.ipynb) · [LeetGPU: Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose) |
| ✅ | Matrix Multiplication | — | 행렬곱 커널 | [notebook](/leetgpu/matrix_multiplication.ipynb) · [LeetGPU: Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication) |
| ⬜ | Tiled Matrix Multiplication | [concept](/leetgpu/tiled_matrix_multiplication.md) | 공유 메모리 최적화 (행렬곱 문제를 타일링으로 최적화) | [LeetGPU: Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication) |
| ⬜ | Softmax CUDA 커널 | [concept](/leetgpu/softmax.md) | softmax 커널 구현 (max 빼기, 병렬 reduction) | [LeetGPU: Softmax](https://leetgpu.com/challenges/softmax) |

### Phase 5. RNN / 시퀀스 모델

| 진도 | 주제 | 개념 | 핵심 내용 | 실습 |
|:---:|---|---|---|---|
| ✅ | RNN 기초/구조 | [concept](/pytorch_study/07_rnn/01_intro/concept.md) | 셀, 입출력 형태 | [practice](/pytorch_study/07_rnn/01_intro/practice.ipynb) · [Kaggle: Shakespeare](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays) |
| ✅ | `nn.RNN` 사용 | [concept](/pytorch_study/07_rnn/02_basics/concept.md) | 시퀀스 차원 다루기 | [practice](/pytorch_study/07_rnn/02_basics/practice.ipynb) · [Kaggle: Shakespeare](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays) |
| ✅ | 문자 예측 (hihello) | [concept](/pytorch_study/07_rnn/03_hihello/concept.md) | one-hot, 교차 엔트로피 | [practice](/pytorch_study/07_rnn/03_hihello/practice.ipynb) · [Kaggle: Shakespeare](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays) |
| ✅ | 긴 시퀀스 (char-RNN) | [concept](/pytorch_study/07_rnn/04_longseq/concept.md) | 윈도우, 긴 텍스트 | [practice](/pytorch_study/07_rnn/04_longseq/practice.ipynb) · [Kaggle: Shakespeare](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays) |
| ✅ | 시계열 예측 | [concept](/pytorch_study/07_rnn/05_time_series/concept.md) | LSTM, 정규화, 주가 예측 | [practice](/pytorch_study/07_rnn/05_time_series/practice.ipynb) · [Kaggle: Store Sales](https://www.kaggle.com/c/store-sales-time-series-forecasting) · [US Stocks](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) |
| ⬜ | LSTM / GRU 내부 게이트 구조 | [concept](/pytorch_study/07_rnn/06_lstm_gru/concept.md) | 게이트 동작, 장기 의존성, bidirectional RNN | [Kaggle: Shakespeare](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays) |
| ⬜ | `nn.Embedding` 기반 시퀀스 | [concept](/pytorch_study/07_rnn/07_embedding/concept.md) | one-hot → 임베딩 | [Kaggle: IMDB Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| ⬜ | Seq2Seq & Attention 기초 | [concept](/pytorch_study/07_rnn/08_seq2seq_attention/concept.md) | 인코더-디코더, Transformer로 가는 다리 | [Kaggle: Eng-Fr Translation](https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench) |
| ⬜ | 자연어 처리 기초 | [concept](/pytorch_study/07_rnn/09_nlp_basics/concept.md) | 텍스트 분류, 감성 분석 | [Kaggle: NLP Getting Started](https://www.kaggle.com/c/nlp-getting-started) |

### Phase 6. Transformer & LLM

| 진도 | 주제 | 개념 | 핵심 내용 | 실습 |
|:---:|---|---|---|---|
| ⬜ | Attention 메커니즘 | [concept](/pytorch_study/08_transformer/01_attention/concept.md) | self-attention, multi-head, positional encoding | [Kaggle: NLP Getting Started](https://www.kaggle.com/c/nlp-getting-started) |
| ⬜ | Transformer 구조 구현 | [concept](/pytorch_study/08_transformer/02_transformer/concept.md) | 인코더/디코더 블록 | [Kaggle: Eng-Fr Translation](https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench) |
| ⬜ | BERT / GPT 구조 차이 | [concept](/pytorch_study/08_transformer/03_bert_gpt/concept.md) | 인코더(BERT) vs 디코더(GPT) | [Kaggle: NLP Getting Started](https://www.kaggle.com/c/nlp-getting-started) |
| ⬜ | HuggingFace Transformers | [concept](/pytorch_study/08_transformer/04_huggingface/concept.md) | tokenizer, `AutoModel`, fine-tuning | [HF NLP Course](https://huggingface.co/learn/nlp-course) · [Kaggle: NLP Getting Started](https://www.kaggle.com/c/nlp-getting-started) |
| ⬜ | LLM 파인튜닝 | [concept](/pytorch_study/08_transformer/05_finetuning/concept.md) | LoRA / QLoRA / PEFT | [HF PEFT 문서](https://huggingface.co/docs/peft) |
| ⬜ | RAG 시스템 구축 | [concept](/pytorch_study/08_transformer/06_rag/concept.md) | 검색 증강 생성(임베딩+벡터DB+프롬프트) | [LangChain RAG 튜토리얼](https://python.langchain.com/docs/tutorials/rag/) |

### Phase 7. Computer Vision 심화

| 진도 | 주제 | 개념 | 핵심 내용 | 실습 |
|:---:|---|---|---|---|
| ⬜ | Object Detection | [concept](/pytorch_study/09_cv_advanced/01_object_detection/concept.md) | YOLO, Faster R-CNN | [Kaggle: Global Wheat Detection](https://www.kaggle.com/c/global-wheat-detection) |
| ⬜ | Semantic Segmentation | [concept](/pytorch_study/09_cv_advanced/02_segmentation/concept.md) | U-Net, Mask R-CNN | [Kaggle: Carvana Image Masking](https://www.kaggle.com/c/carvana-image-masking-challenge) |
| ⬜ | 최신 백본 | [concept](/pytorch_study/09_cv_advanced/03_backbones/concept.md) | EfficientNet, Vision Transformer(ViT) | [Kaggle: Cassava Leaf Disease](https://www.kaggle.com/c/cassava-leaf-disease-classification) |
| ⬜ | Image Generation | [concept](/pytorch_study/09_cv_advanced/04_generation/concept.md) | GAN, VAE, Diffusion Model | [Kaggle: GAN Getting Started](https://www.kaggle.com/c/gan-getting-started) |

### Phase 8. MLOps & 모델 서빙

| 진도 | 주제 | 개념 | 핵심 내용 | 실습 |
|:---:|---|---|---|---|
| ⬜ | 모델 저장 및 로딩 | — | ONNX, TorchScript | — |
| ⬜ | 모델 서빙 | — | FastAPI로 추론 API | — |
| ⬜ | 컨테이너화 | — | Docker | — |
| ⬜ | 실험 관리 | — | MLflow | — |
| ⬜ | 데이터 버전 관리 | — | DVC | — |

### Phase 9. 클라우드 & AI 파이프라인

| 진도 | 주제 | 개념 | 핵심 내용 | 실습 |
|:---:|---|---|---|---|
| ⬜ | 클라우드 ML 플랫폼 | — | AWS SageMaker / GCP Vertex AI | — |
| ⬜ | 학습 파이프라인 | — | 클라우드 기반 모델 학습 | — |
| ⬜ | 모델 운영 | — | 모니터링 및 운영 | — |

### Phase 10. 포트폴리오 & 실전

| 진도 | 주제 | 개념 | 핵심 내용 | 실습 |
|:---:|---|---|---|---|
| ⬜ | Kaggle 대회 참여 | — | 점수 개선 | — |
| ⬜ | 개인 프로젝트 | — | 백엔드 경험 + AI 결합 | — |
| ⬜ | 논문 구현 | — | 1개 이상 재현 | — |
| ⬜ | GitHub 포트폴리오 정리 | — | 결과물 정리 | — |

---

## 🔧 보강 학습 (현재 토대 강화)

모델 구조 위주의 현재 커리큘럼에서 **빠져 있지만, 모델을 제대로 학습·평가·운용하려면 반드시 필요한** 기반 지식입니다. Phase 2~5와 병행해 채우면 좋습니다.

### A. 학습의 핵심 메커니즘

| 진도 | 주제 | 개념 | 핵심 내용 | 실습 |
|:---:|---|---|---|---|
| ⬜ | Autograd 심화 | [concept](/pytorch_study/supplementary/A_core_mechanics/01_autograd/concept.md) | 계산 그래프, `backward()` 원리, `requires_grad`/`detach()`/`no_grad()`, `zero_grad()`의 이유 | [PyTorch 튜토리얼](https://docs.pytorch.org/tutorials/) |
| ⬜ | `nn.Module` 구조 심화 | [concept](/pytorch_study/supplementary/A_core_mechanics/02_nn_module/concept.md) | `nn.Sequential`/`ModuleList`, 커스텀 레이어, `parameters()`/`named_parameters()` | [PyTorch 튜토리얼](https://docs.pytorch.org/tutorials/) |
| ⬜ | Optimizer 심화 | [concept](/pytorch_study/supplementary/A_core_mechanics/03_optimizer/concept.md) | Momentum, RMSprop, Adam/AdamW, weight decay(L2), `lr_scheduler` | [d2l.ai](https://d2l.ai/) |
| ⬜ | 손실 함수 정리 | [concept](/pytorch_study/supplementary/A_core_mechanics/04_loss_functions/concept.md) | MSE/MAE, `BCEWithLogitsLoss`, `CrossEntropyLoss`(로짓 입력) 선택 기준 | [d2l.ai](https://d2l.ai/) |

### B. 일반화 & 평가

| 진도 | 주제 | 개념 | 핵심 내용 | 실습 |
|:---:|---|---|---|---|
| ⬜ | 규제(Regularization) | [concept](/pytorch_study/supplementary/B_generalization/01_regularization/concept.md) | L1/L2, dropout, early stopping, augmentation의 관계 | [d2l.ai](https://d2l.ai/) |
| ⬜ | 평가 지표 | [concept](/pytorch_study/supplementary/B_generalization/02_metrics/concept.md) | accuracy 한계, precision/recall/F1, confusion matrix, ROC-AUC, RMSE/R² | [scikit-learn: Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) |
| ⬜ | 데이터 분할 & 검증 | [concept](/pytorch_study/supplementary/B_generalization/03_data_split/concept.md) | train/val/test, k-fold, 데이터 누수(leakage) 방지 | [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) |
| ⬜ | 전처리/증강 | [concept](/pytorch_study/supplementary/B_generalization/04_preprocessing/concept.md) | `torchvision.transforms`, normalize, 이미지 augmentation, 불균형 데이터 | [PyTorch 튜토리얼](https://docs.pytorch.org/tutorials/) |

### C. 실전 운용

| 진도 | 주제 | 개념 | 핵심 내용 | 실습 |
|:---:|---|---|---|---|
| ⬜ | GPU/디바이스 관리 | [concept](/pytorch_study/supplementary/C_operations/01_device/concept.md) | `.to(device)`, CPU/GPU 텐서 혼용 오류 디버깅 | [PyTorch 튜토리얼](https://docs.pytorch.org/tutorials/) |
| ⬜ | 모델 저장/로드 | [concept](/pytorch_study/supplementary/C_operations/02_save_load/concept.md) | `state_dict`, checkpoint(옵티마이저 포함), 추론 시 `model.eval()` | [PyTorch 튜토리얼](https://docs.pytorch.org/tutorials/) |
| ⬜ | 재현성 & 모니터링 | [concept](/pytorch_study/supplementary/C_operations/03_reproducibility/concept.md) | seed 고정, TensorBoard(visdom 표준 대안), `wandb` | [PyTorch 튜토리얼](https://docs.pytorch.org/tutorials/) |
| ⬜ | 전이학습 | [concept](/pytorch_study/supplementary/C_operations/04_transfer_learning/concept.md) | `torchvision.models` 사전학습 모델, feature extractor vs fine-tuning | [CS231n](https://cs231n.github.io/) |

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
│   ├── 06_cnn/                   #   CNN·VGG·ResNet·EfficientNet·시각화
│   ├── 07_rnn/                   #   RNN·문자예측·시계열·LSTM/GRU·임베딩·seq2seq·NLP
│   ├── 08_transformer/           #   Attention·Transformer·BERT/GPT·HuggingFace·파인튜닝·RAG
│   ├── 09_cv_advanced/           #   Object Detection·Segmentation·백본(ViT)·생성모델
│   ├── supplementary/            #   보강 학습 (A.핵심 메커니즘 · B.일반화·평가 · C.실전 운용)
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
