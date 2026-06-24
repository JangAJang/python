# 합성곱 (Convolution)

> 테마: 06_cnn · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
이미지 위에서 작은 필터(커널)를 stride만큼 이동시키며 겹치는 영역끼리 곱해 더하는 연산으로, CNN의 핵심 빌딩 블록이다.

## 핵심 개념
- **Convolution(합성곱)**: 입력 이미지 위에서 filter(kernel)를 stride만큼 이동시키며, 겹치는 부분의 값을 원소별로 곱한 뒤 모두 더한 값을 출력으로 반환하는 연산.
- **filter / kernel**: 작은 가중치 행렬(예: 3x3). 한 칸씩 이동하며 위치마다 하나의 출력값을 계산한다.
- **stride**: 필터가 한 번에 이동하는 칸 수. 클수록 출력 크기가 작아진다.
- **padding**: 입력 행렬의 상하좌우를 0으로 둘러싸는 것. 출력 크기를 유지하거나 가장자리 정보를 보존할 때 사용한다.
- **Pooling**: 특성 맵 크기를 줄이는 연산. Max Pooling은 영역 내 최댓값, Average Pooling은 영역 내 평균값을 반환한다. fully connected 연산을 대체하기 위해 average pooling을 쓰기도 한다.

## 원리 / 수식
- 출력 (0,0) 예시: 필터와 겹친 영역의 각 위치 값을 곱해 모두 더한다. `(1·1)+(2·0)+(3·1)+(0·1)+(1·1)+(5·0)+(1·1)+(0·0)+(2·1) = 8`.
- 뉴런(퍼셉트론) 관점에서는 필터에 매칭되는 입력값들이 뉴런에 입력되고, bias가 더해져 `출력 = 8 + bias` 형태가 된다.
- 출력 크기: `out = (in + 2·padding - kernel) / stride + 1`.
- **Convolution vs Cross-correlation**: 수학적 정의의 합성곱은 커널을 뒤집어 계산하지만, **딥러닝 프레임워크의 `nn.Conv2d`는 실제로는 Cross-correlation(상호상관)** 연산을 한다. Autocorrelation(자기상관)은 신호를 자기 자신과 비교해 주기성을 탐지하는 데 쓰인다.

## PyTorch 구현 포인트
- `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)` : 2D 합성곱 레이어.
- `nn.MaxPool2d(kernel_size)` : 최대 풀링.
- 입력 텐서 형태는 `(batch, channel, height, width)`.

```python
import torch, torch.nn as nn
x = torch.Tensor(1, 1, 28, 28)          # (N, C, H, W)
conv = nn.Conv2d(1, 5, 5)                # in=1, out=5, kernel=5x5
pool = nn.MaxPool2d(2)
out = conv(x)                            # -> [1, 5, 24, 24]
out2 = pool(out)                         # -> [1, 5, 12, 12]
```

## 자주 하는 실수 / 팁
- 입력은 반드시 4차원 `(N, C, H, W)`로 넣어야 한다. 채널 차원을 빠뜨리기 쉽다.
- `nn.Conv2d`가 수학적 합성곱이 아니라 cross-correlation임을 기억하면 커널 방향 혼동을 피할 수 있다.
- 출력 크기 공식을 미리 계산해야 이후 `nn.Linear`의 입력 차원을 맞출 수 있다.

## 예제 노트북 요약
- `example.ipynb`는 합성곱의 정의, stride/padding, 뉴런과의 관계, pooling을 그림과 함께 설명한다.
- `nn.Conv2d`와 `nn.MaxPool2d`를 28x28 입력에 적용해 출력 크기가 어떻게 변하는지 직접 확인하고, 합성곱·상호상관·자기상관의 차이를 정리한다.

## 더 보기
- 다음 단계: [`../02_mnist_cnn/concept.md`](../02_mnist_cnn/concept.md) — CNN으로 MNIST 분류
- 뉴런 기초: [`../../04_neural_network/01_perceptron/concept.md`](../../04_neural_network/01_perceptron/concept.md)
