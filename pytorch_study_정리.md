# PyTorch 학습 전체 정리

`pytorch_study/` 디렉토리의 모든 학습 내용을 **중간 테마 → 세부 주제 → (개념 / 예제 / 실습)** 구조로 재정리한 인덱스입니다.

각 세부 주제 디렉토리는 자체 완결형으로 구성되어 있습니다.

| 파일 | 역할 |
|---|---|
| `concept.md` | 개념 정리 (핵심 개념 · 수식 · PyTorch 구현 포인트 · 자주 하는 실수) |
| `example.ipynb` | 기존 학습 노트북 (예제, 이미지 경로는 각 디렉토리 `img/`로 로컬화) |
| `practice.ipynb` | 실습 노트북 (첫 셀에 검증된 LeetGPU/Kaggle 링크 + 문제 정의, 이후 코드 스캐폴드) |

---

## 학습 로드맵

```
01_basics              텐서 다루기 (PyTorch의 기본 자료구조)
   ↓
02_regression          선형 회귀 — 모델·비용함수·경사하강법·데이터 로딩
   ↓
03_classification       분류 — 로지스틱(이진) → 소프트맥스(다중)
   ↓
04_neural_network      신경망 — 퍼셉트론 → MLP → MNIST 입문
   ↓
05_training_techniques 학습 기법 — ReLU·초기화·드롭아웃·배치정규화·팁
   ↓
06_cnn                 합성곱 신경망 — Conv → CNN → VGG → ResNet → 데이터/시각화
   ↓
07_rnn                 순환 신경망 — 기초 → 문자예측 → 긴 시퀀스 → 시계열
```

---

## 01_basics — 기초

| 주제 | 개념 | 핵심 내용 | 실습 링크 |
|---|---|---|---|
| 01_tensor_manipulation | [concept](pytorch_study/01_basics/01_tensor_manipulation/concept.md) | 텐서 생성·shape·브로드캐스팅, 요소곱 vs 행렬곱, view/reshape/squeeze, cat/stack, mean/sum/max | [LeetGPU: Vector Add / Transpose / Matmul](https://leetgpu.com/challenges) |

## 02_regression — 선형 회귀

| 주제 | 개념 | 핵심 내용 | 실습 링크 |
|---|---|---|---|
| 01_linear_regression | [concept](pytorch_study/02_regression/01_linear_regression/concept.md) | 가설 `H=Wx+b`, MSE, SGD | [Kaggle: House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| 02_multivariable_regression | [concept](pytorch_study/02_regression/02_multivariable_regression/concept.md) | 다변수 회귀, 행렬곱, `nn.Linear` | [Kaggle: House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| 03_gradient_descent | [concept](pytorch_study/02_regression/03_gradient_descent/concept.md) | 경사하강법 원리, 미분, 학습률, `optim.SGD` | [LeetGPU Challenges](https://leetgpu.com/challenges) |
| 04_data_loading | [concept](pytorch_study/02_regression/04_data_loading/concept.md) | `Dataset`/`DataLoader`, 미니배치 | [Kaggle: House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) · [Student Marks](https://www.kaggle.com/datasets/yasserh/student-marks-dataset) |

## 03_classification — 분류

| 주제 | 개념 | 핵심 내용 | 실습 링크 |
|---|---|---|---|
| 01_logistic_regression | [concept](pytorch_study/03_classification/01_logistic_regression/concept.md) | 이진분류, 시그모이드, BCE | [Kaggle: Titanic](https://www.kaggle.com/c/titanic) |
| 02_softmax_classification | [concept](pytorch_study/03_classification/02_softmax_classification/concept.md) | 다중분류, 소프트맥스, 교차 엔트로피 | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) |

## 04_neural_network — 신경망

| 주제 | 개념 | 핵심 내용 | 실습 링크 |
|---|---|---|---|
| 01_perceptron | [concept](pytorch_study/04_neural_network/01_perceptron/concept.md) | 퍼셉트론, 선형 분류기, XOR의 한계 | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) · [LeetGPU](https://leetgpu.com/challenges) |
| 02_multilayer_perceptron | [concept](pytorch_study/04_neural_network/02_multilayer_perceptron/concept.md) | 은닉층, 역전파(연쇄법칙), XOR 해결 | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) · [LeetGPU](https://leetgpu.com/challenges) |
| 03_mnist_intro | [concept](pytorch_study/04_neural_network/03_mnist_intro/concept.md) | MNIST, epoch/batch/iteration, `CrossEntropyLoss` | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) |

## 05_training_techniques — 학습 기법

| 주제 | 개념 | 핵심 내용 | 실습 링크 |
|---|---|---|---|
| 01_relu_activation | [concept](pytorch_study/05_training_techniques/01_relu_activation/concept.md) | 기울기 소실, ReLU, optimizer 비교 | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) · [LeetGPU](https://leetgpu.com/challenges) |
| 02_weight_initialization | [concept](pytorch_study/05_training_techniques/02_weight_initialization/concept.md) | Xavier, He 초기화 | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) · [LeetGPU](https://leetgpu.com/challenges) |
| 03_dropout | [concept](pytorch_study/05_training_techniques/03_dropout/concept.md) | 드롭아웃, 과적합 방지, train/eval 모드 | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) |
| 04_batch_normalization | [concept](pytorch_study/05_training_techniques/04_batch_normalization/concept.md) | internal covariate shift, 배치정규화 | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) |
| 05_tips | [concept](pytorch_study/05_training_techniques/05_tips/concept.md) | 학습률, 데이터 전처리/정규화, 과적합 종합 | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) |

## 06_cnn — 합성곱 신경망

| 주제 | 개념 | 핵심 내용 | 실습 링크 |
|---|---|---|---|
| 01_convolution | [concept](pytorch_study/06_cnn/01_convolution/concept.md) | 합성곱 연산, 필터/스트라이드/패딩, 풀링 | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) · [LeetGPU](https://leetgpu.com/challenges) |
| 02_mnist_cnn | [concept](pytorch_study/06_cnn/02_mnist_cnn/concept.md) | CNN으로 MNIST 분류, 레이어 구성 | [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) · [LeetGPU](https://leetgpu.com/challenges) |
| 03_vgg | [concept](pytorch_study/06_cnn/03_vgg/concept.md) | VGG 아키텍처, 3×3 conv 스택 | [Kaggle: Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) |
| 04_resnet | [concept](pytorch_study/06_cnn/04_resnet/concept.md) | 잔차 연결(residual), 깊은 신경망 | [Kaggle: Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) |
| 05_image_folder | [concept](pytorch_study/06_cnn/05_image_folder/concept.md) | `ImageFolder`, 커스텀 이미지 데이터셋 | [Kaggle: Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) |
| 06_visdom | [concept](pytorch_study/06_cnn/06_visdom/concept.md) | Visdom으로 학습 과정 시각화 | [Visdom 공식 저장소](https://github.com/fossasia/visdom) · [Kaggle: Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) |

## 07_rnn — 순환 신경망

| 주제 | 개념 | 핵심 내용 | 실습 링크 |
|---|---|---|---|
| 01_intro | [concept](pytorch_study/07_rnn/01_intro/concept.md) | RNN 구조, 셀, 입출력 형태 | [Kaggle: Shakespeare Plays](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays) |
| 02_basics | [concept](pytorch_study/07_rnn/02_basics/concept.md) | `nn.RNN` 기본 사용, 시퀀스 차원 | [Kaggle: Shakespeare Plays](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays) |
| 03_hihello | [concept](pytorch_study/07_rnn/03_hihello/concept.md) | 문자 예측, one-hot, 교차 엔트로피 | [Kaggle: Shakespeare Plays](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays) |
| 04_longseq | [concept](pytorch_study/07_rnn/04_longseq/concept.md) | 긴 시퀀스, char-RNN, 윈도우 | [Kaggle: Shakespeare Plays](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays) |
| 05_time_series | [concept](pytorch_study/07_rnn/05_time_series/concept.md) | 시계열 예측(주가), 정규화, LSTM | [Kaggle: Store Sales](https://www.kaggle.com/c/store-sales-time-series-forecasting) · [US Stocks](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) |

---

## 부록

- `pytorch_study/self_qna.md` — CNN/Transformer를 활용한 문장 분류 등 셀프 Q&A 모음.
- `pytorch_study/images/` — 원본 이미지 보관소(각 주제 디렉토리의 `img/`로 복사되어 사용됨).
- `pytorch_study/MNIST_data/` — MNIST 데이터셋 캐시.

> 모든 실습 링크는 작성 시점에 실제 URL 존재 여부를 확인했습니다(LeetGPU 챌린지, Kaggle 대회/데이터셋).
> 일부 `concept.md`의 "자주 하는 실수" 항목에는 원본 노트북에서 발견된 코드 버그(연산자 우선순위, 이중 softmax, 변수명 오타 등)를 학습 포인트로 정리해 두었습니다.
