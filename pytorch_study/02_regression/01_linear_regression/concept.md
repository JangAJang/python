# 단순 선형회귀 (Linear Regression)

> 테마: 02_regression · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
하나의 입력 변수 x로부터 출력 y를 직선 `H(x) = Wx + b`로 예측하고, MSE를 최소화하도록 경사하강법(SGD)으로 W, b를 학습한다.

## 핵심 개념
- **가설(Hypothesis)**: `H(x) = Wx + b`. W는 가중치(weight), b는 편향(bias).
- **목표**: 데이터를 가장 잘 설명하는 직선의 W, b를 찾는 것.
- **비용함수(Cost / Loss)**: 선형회귀는 **MSE(Mean Squared Error, 오차제곱평균)** 를 사용한다.
- **학습**: 비용을 줄이는 방향으로 W, b를 반복 갱신(경사하강법, SGD).
- **requires_grad=True**: 해당 텐서를 학습 대상으로 지정하여 autograd가 gradient를 추적하게 한다.
- W, b를 0으로 초기화하면 초기 예측은 항상 0이다.

## 원리 / 수식
- 가설: $H(x) = Wx + b$
- 비용함수 (MSE):
$$\text{cost}(W, b) = \frac{1}{m} \sum_{i=1}^{m} \left( H(x^{(i)}) - y^{(i)} \right)^2$$
- 학습은 이 cost를 최소로 만드는 W, b를 찾는 것이며, 그 방법으로 경사하강법을 쓴다 (자세한 원리는 `../03_gradient_descent/concept.md`).

## PyTorch 구현 포인트
- 학습 파라미터 선언: `W = torch.zeros(1, requires_grad=True)`, `b = torch.zeros(1, requires_grad=True)`
- 가설: `hypothesis = x_train * W + b`
- 비용(MSE): `cost = torch.mean((hypothesis - y_train) ** 2)`
- 옵티마이저: `optimizer = torch.optim.SGD([W, b], lr=0.01)`
- 학습 루프 3단계:
  - `optimizer.zero_grad()` — 이전 단계 gradient 초기화 (PyTorch는 grad를 누적하므로 필수)
  - `cost.backward()` — 역전파로 gradient 계산
  - `optimizer.step()` — 계산된 gradient로 W, b 업데이트

```python
optimizer = torch.optim.SGD([W, b], lr=0.01)
for epoch in range(1000):
    hypothesis = x_train * W + b
    cost = torch.mean((hypothesis - y_train) ** 2)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
```

## 자주 하는 실수 / 팁
- **MSE 괄호 실수**: `torch.mean(hypothesis - y_train ** 2)` 는 틀린 코드다. 연산자 우선순위 때문에 `y_train ** 2` 만 먼저 계산된다. 반드시 `torch.mean((hypothesis - y_train) ** 2)` 처럼 오차 전체를 괄호로 묶고 제곱해야 한다. (예제 노트북 초반 셀에 이 실수가 있으니 학습 루프 셀의 올바른 형태를 참고할 것)
- `zero_grad()`를 빼먹으면 gradient가 누적되어 학습이 망가진다.
- 학습률(lr)이 너무 크면 발산하고, 너무 작으면 수렴이 느리다.

## 예제 노트북 요약
- `example.ipynb`는 공부 시간(x)과 점수(y)의 관계를 단순 선형회귀로 예측한다.
- `x_train=[[1],[2],[3]]`, `y_train=[[2],[4],[6]]` (즉 정답은 W=2, b=0)을 1000 epoch 동안 SGD로 학습하며 가설/MSE/옵티마이저 사용법을 보여준다.

## 더 보기
- 입력이 여러 개인 경우: [`../02_multivariable_regression/concept.md`](../02_multivariable_regression/concept.md)
- 경사하강법 원리: [`../03_gradient_descent/concept.md`](../03_gradient_descent/concept.md)
- 대용량 데이터 적재: [`../04_data_loading/concept.md`](../04_data_loading/concept.md)
