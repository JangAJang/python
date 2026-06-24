# 다층 퍼셉트론 (Multi-layer Perceptron, MLP)

> 테마: 04_neural_network · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
은닉층(hidden layer)을 추가해 비선형 결정 경계를 만들 수 있는 신경망으로, 역전파(backpropagation)로 학습시켜 단층 퍼셉트론이 못 풀던 XOR을 해결한다.

## 핵심 개념
- AND/OR는 하나의 직선으로 분류 가능하지만 **XOR**은 단층 퍼셉트론(직선 하나)으로 분류 불가능하다.
- XOR은 두 입력이 서로 다르면 1, 같으면 0을 반환한다.
- **퍼셉트론 2개 이상(은닉층)** 을 쌓으면 XOR을 풀 수 있다.
- 단, 다층 구조를 어떻게 학습시킬지가 문제였고, 그 해법이 **역전파(Backpropagation)** 다.
- 층을 더 깊게(예: 2→10→10→10→1) 쌓아도 충분한 학습으로 XOR을 풀 수 있다(예제에서 확인).

## 원리 / 수식
### 역전파(Backpropagation)
- 출력 `O`와 실제 정답 `GT`의 손실(loss)을 최소화하도록 가중치 `w`를 조정하는 방법.
- 손실을 `w`에 대해 미분한 값(gradient)이 0이면 극값(최소/최대)에 해당 → 손실이 최소가 되는 `w`를 향해 이동한다.
- **연쇄 법칙(chain rule)** 으로 출력 쪽에서 입력 쪽으로 기울기를 거꾸로 전파한다.

### 예제로 보는 미분
- `f = wx + b`에서 `g = w·x`로 두면 `f = g + b`, `g = w·x`.
- 각 변수로 미분하면: `∂f/∂w = x`, `∂f/∂x = w`, `∂f/∂b = 1`, `∂f/∂g = 1`.
- `w=-2, x=5, b=3`일 때 **forward**: `f = -2·5 + 3 = -7`.
- **backward**: 위 미분식에 값을 대입 → `∂f/∂w = 5`, `∂f/∂x = -2`, `∂f/∂b = 1`.
- 아무리 복잡한 식이라도 각 노드의 단순 미분값(local gradient)을 곱해 나가면 전체 기울기를 얻는다.
- gradient는 "손실을 줄이려면 각 변수를 얼마나, 어느 방향으로 조정해야 하는가"를 알려준다.

## PyTorch 구현 포인트
- 은닉층을 쌓아 `Sequential`로 연결: `Linear → Sigmoid → Linear → Sigmoid → ...`.
- `loss.backward()` : 역전파를 자동 수행해 각 파라미터의 `.grad`를 채운다(autograd).
- `optimizer.step()` : 계산된 gradient와 학습률(lr)에 따라 가중치를 갱신한다.
- `optimizer.zero_grad()` : 이전 스텝의 gradient 누적을 0으로 초기화(필수).

```python
linear1 = torch.nn.Linear(2, 2, bias=True)
linear2 = torch.nn.Linear(2, 1, bias=True)
sigmoid = torch.nn.Sigmoid()
model = torch.nn.Sequential(linear1, sigmoid, linear2, sigmoid).to(device)
# 은닉층 덕분에 XOR 손실이 0.69 -> 0.001 수준까지 내려감
```

## 자주 하는 실수 / 팁
- `optimizer.zero_grad()`를 빼먹으면 gradient가 누적되어 학습이 망가진다.
- Sigmoid를 깊게 많이 쌓으면 기울기가 작아져(gradient vanishing) 학습이 느리게 시작될 수 있다(예제의 4층 모델은 6000스텝 부근에서야 급격히 손실이 떨어짐).
- XOR처럼 작은 문제는 lr을 크게(예: 1) 잡아도 학습이 잘 된다. 일반 문제에는 더 작은 lr이 안전하다.

## 예제 노트북 요약
- `example.ipynb`는 역전파 원리(연쇄 법칙, forward/backward 계산)를 설명한다.
- `Linear(2,2)+Sigmoid+Linear(2,1)+Sigmoid` 2층 MLP로 XOR을 학습해 손실을 0.001 수준까지 떨어뜨린다.
- 이어서 2→10→10→10→1의 더 깊은 MLP로도 XOR을 학습시켜 비교한다.

## 더 보기
- 이전 단계: [`../01_perceptron/concept.md`](../01_perceptron/concept.md) — 단층 퍼셉트론과 XOR의 한계
- 다음 단계: [`../03_mnist_intro/concept.md`](../03_mnist_intro/concept.md) — MNIST와 softmax 분류
