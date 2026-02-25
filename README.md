# python

- 백엔드 개발자 → AI 엔지니어 직무 전환을 위한 학습 리포지토리

---

## 학습 현황

### Phase 1. Python 기초 및 ML 입문

- [x] Python 기본 문법 학습 ([basis/lessons.md](/basis/lessons.md))
- [x] 핸즈온 머신러닝 2장 - End-to-End ML 프로젝트 ([california.ipynb](/hands_on_machine_learning/california.ipynb))
  - 데이터 수집, 탐색, 전처리, 모델 훈련, 평가
- [x] Kaggle - 타이타닉 competition ([titanic.py](/kaggle/titanic.py))

### Phase 2. 딥러닝 기초 (PyTorch)

- [x] 텐서 조작 ([tensor_manipulation](/pytorch_study/tensor_manipulation.ipynb))
- [x] 선형 회귀 ([linear_regression](/pytorch_study/linear_regression.ipynb))
- [x] 다변수 선형 회귀 ([multivariable_linear_regression](/pytorch_study/multivariable_linear_regression.ipynb))
- [x] 로지스틱 회귀 ([logistic_regression](/pytorch_study/logistic_regression.ipynb))
- [x] 소프트맥스 분류 ([softmax_classification](/pytorch_study/softmax_classification.ipynb))
- [x] 경사 하강법 ([gradient_descent](/pytorch_study/gradient_descent.ipynb))
- [x] 활성화 함수 - ReLU ([ReLU](/pytorch_study/ReLU.ipynb))
- [x] 퍼셉트론 ([perceptron](/pytorch_study/perceptron.ipynb))
- [x] 다층 퍼셉트론 MLP ([mult_layer_perceptron](/pytorch_study/mult_layer_perceptron.ipynb))
- [x] 배치 정규화 ([batch_normalization](/pytorch_study/batch_normalization.ipynb))
- [x] 드롭아웃 ([dropout](/pytorch_study/dropout.ipynb))
- [x] 가중치 초기화 ([weight_initialization](/pytorch_study/weight_initialization.ipynb))
- [x] 데이터 로딩 & DataLoader ([loading_data](/pytorch_study/loading_data.ipynb))
- [x] ImageFolder 사용법 ([image_folder](/pytorch_study/image_folder.ipynb))
- [x] 학습 팁 (Learning Rate Scheduler 등) ([tips](/pytorch_study/tips.ipynb))
- [x] 시각화 - Visdom ([visdom](/pytorch_study/visdom.ipynb))

### Phase 3. CNN (Convolutional Neural Network)

- [x] 합성곱 연산 이해 ([convolution](/pytorch_study/convolution.ipynb))
- [x] MNIST CNN 구현 ([mnist_cnn](/pytorch_study/mnist_cnn.ipynb))
- [x] MNIST 기초 ([mnist_intro](/pytorch_study/mnist_intro.ipynb))
- [x] VGG 네트워크 구현 ([vgg](/pytorch_study/vgg.ipynb))
- [ ] ResNet 구현
- [ ] EfficientNet 구현

### Phase 4. GPU 프로그래밍 (CUDA / LeetGPU)

- [x] Color Inversion ([color_inversion](/leetgpu/color_inversion.ipynb))
- [x] Matrix Transpose ([matrix_transpose](/leetgpu/matrix_transpose.ipynb))
- [x] Matrix Multiplication ([matrix_multiplication](/leetgpu/matrix_multiplication.ipynb))
- [ ] Tiled Matrix Multiplication (공유 메모리 최적화)
- [ ] Softmax CUDA 커널

### Phase 5. RNN / 시퀀스 모델

- [ ] RNN 기초
- [ ] LSTM / GRU
- [ ] 시계열 예측 실습
- [ ] 자연어 처리 기초 (텍스트 분류, 감성 분석)

### Phase 6. Transformer & LLM

- [ ] Attention 메커니즘 이해
- [ ] Transformer 구조 구현
- [ ] BERT / GPT 구조 이해
- [ ] HuggingFace Transformers 사용
- [ ] LLM 파인튜닝 (LoRA / QLoRA)
- [ ] RAG (Retrieval-Augmented Generation) 시스템 구축

### Phase 7. Computer Vision 심화

- [ ] Object Detection (YOLO, Faster R-CNN)
- [ ] Semantic Segmentation
- [ ] Image Generation (GAN, Diffusion Model 개념)

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

## 디렉토리 구조

```
python/
├── basis/                        # Python 기초 문법
├── hands_on_machine_learning/    # 핸즈온 ML 교재 실습
├── kaggle/                       # Kaggle 대회 풀이
├── leetgpu/                      # CUDA GPU 프로그래밍
├── pytorch_study/                # PyTorch 딥러닝 기초~CNN
└── algorithm_solver/             # 알고리즘 분류 도구
```

---

## 참고 자료

- [핸즈온 머신러닝 3판](https://github.com/ageron/handson-ml3)
- [모두를 위한 딥러닝 (PyTorchZeroToAll)](https://github.com/hunkim/PyTorchZeroToAll)
- [LeetGPU](https://leetgpu.com)
- [HuggingFace 문서](https://huggingface.co/docs)
- [fast.ai](https://www.fast.ai)
