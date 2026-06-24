# MNIST 데이터셋 소개 (MNIST Intro)

> 테마: 04_neural_network · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
손글씨 숫자(0~9) 이미지 데이터셋 MNIST를 torchvision으로 불러와, 784차원으로 펼친 입력을 단층 선형 + softmax(CrossEntropyLoss)로 10개 클래스 분류한다.

## 핵심 개념
- **MNIST**: handwritten digits dataset. 우편번호 인식 자동화를 위해 만들어졌다.
- 각 이미지는 **28 x 28 픽셀**, **1 channel grayscale**(밝기 정보만, 색 없음)이다.
  - channel = 색 정보. RGB = 3 channel, RGBA = 4 channel. 1 channel gray scale은 밝기만 가진다.
- 학습 60,000장 / 테스트 10,000장.
- **torchvision**: 유명 데이터셋, 모델 아키텍처, 이미지 변환기(transforms)를 제공하는 PyTorch 패키지.

### 학습 용어
- **epoch**: 전체 데이터를 한 번 다 학습하는 단위. (예: MNIST 60,000장을 전부 한 번 학습 = 1 epoch)
- **batch size**: 1 epoch 내 데이터를 한 번에 처리하기 힘들 때 나누는 묶음 크기.
- **iterations**: 1 epoch을 위해 사용된 batch의 수. `1 epoch = batch_size × iterations`.

## 원리 / 수식
- 28×28 이미지를 1차원 벡터 `784`로 펼친다(`view(-1, 28*28)`).
- 선형 변환: `logits = W·x + b`, `W`는 `(784, 10)`.
- **softmax**: `softmax(z)_i = e^{z_i} / Σ_j e^{z_j}` → 10개 클래스에 대한 확률 분포.
- **cross entropy** 손실로 정답 클래스의 확률을 최대화한다.
- 예측: `argmax(logits, dim=1)`로 가장 큰 점수의 클래스를 고른다.

## PyTorch 구현 포인트
- `torchvision.datasets.MNIST(root, train, transform=transforms.ToTensor(), download=True)` : 데이터 로드.
- `torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True)` : 미니배치 단위 공급.
- `torch.nn.Linear(784, 10)` : 입력 784 → 출력 10(클래스 수).
- `torch.nn.CrossEntropyLoss()` : **내부에서 softmax를 자동 계산**하므로 모델 마지막에 별도 softmax를 두지 않는다. 라벨은 one-hot이 아닌 정수 인덱스.
- `torch.optim.SGD(linear.parameters(), lr=0.1)`.
- 평가는 `with torch.no_grad():` 안에서 `argmax`로 정확도 측정.

```python
linear = torch.nn.Linear(784, 10, bias=True).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)  # softmax 내장
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)
X = X.view(-1, 28 * 28)  # (batch, 784)
```

## 자주 하는 실수 / 팁
- `CrossEntropyLoss`는 softmax를 내부에서 처리하므로 모델 출력에 softmax를 또 적용하면 안 된다(이중 적용 오류).
- 라벨 `Y`는 정수 클래스 인덱스여야 한다(one-hot 아님).
- 이미지를 `view(-1, 784)`로 펼치는 것을 잊으면 `Linear(784, 10)`과 차원이 맞지 않는다.
- 구버전 API `mnist_test.test_data`/`test_labels`는 deprecated → 최신은 `.data`/`.targets`.
- 단층 선형 분류기는 약 88~90% 정확도에 그친다. 더 높이려면 은닉층(MLP)이나 CNN이 필요하다.

## 예제 노트북 요약
- `example.ipynb`는 MNIST를 torchvision으로 다운로드/로드하고 DataLoader로 배치를 구성한다.
- `Linear(784,10) + CrossEntropyLoss(softmax 내장)`로 15 epoch 학습 후 테스트 정확도(약 0.888)를 출력하고, 무작위 샘플 하나를 예측·시각화한다.

## 더 보기
- 이전 단계: [`../02_multilayer_perceptron/concept.md`](../02_multilayer_perceptron/concept.md) — 은닉층과 역전파
- 분류 기초: [`../../03_classification/02_softmax_classification/concept.md`](../../03_classification/02_softmax_classification/concept.md)
