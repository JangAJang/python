# 경사하강법 (Gradient Descent)

> 테마: 02_regression · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
비용함수(MSE)를 W에 대해 미분한 기울기(gradient)의 반대 방향으로 W를 조금씩 이동시켜 비용이 최소가 되는 지점을 찾는 최적화 알고리즘이다.

## 핵심 개념
- 선형회귀의 비용함수 MSE는 W에 대해 **2차함수(아래로 볼록)** 형태이므로 최소점이 하나 존재한다.
- 그 최소점으로 가려면 현재 W에서 비용의 **기울기(gradient)** 를 구해, 기울기의 반대 방향으로 이동한다.
- **기울기의 절댓값이 클수록** 최소점에서 멀다는 뜻이므로 W를 크게 변화시키고, 최소점에 가까울수록(기울기≈0) 작게 변화한다.
- **학습률(lr, learning rate)**: 한 번에 이동하는 보폭. 너무 크면 발산/진동, 너무 작으면 수렴이 느리다.

## 원리 / 수식
- 단순화한 가설 $H(x) = Wx$ (b=0), 비용:
$$\text{cost}(W) = \frac{1}{m}\sum_{i=1}^{m}(Wx^{(i)} - y^{(i)})^2$$
- W에 대한 미분(기울기):
$$\frac{\partial\, \text{cost}}{\partial W} = \frac{2}{m}\sum_{i=1}^{m}(Wx^{(i)} - y^{(i)})\,x^{(i)}$$
- 갱신식:
$$W \leftarrow W - \alpha\,\frac{\partial\, \text{cost}}{\partial W}$$
  여기서 $\alpha$가 학습률(lr).
- 예제에서는 상수 2를 생략한 `torch.mean((hypothesis - y_train) * x_train)`을 gradient로 사용한다 (lr에 흡수되므로 동작에는 무방).

## PyTorch 구현 포인트
- **직접 구현 (수식으로)**:
```python
W = torch.zeros(1)
lr = 0.1
for epoch in range(101):
    hypothesis = x_train * W
    cost = torch.mean((hypothesis - y_train) ** 2)
    gradient = torch.mean((hypothesis - y_train) * x_train)  # 미분 직접 계산
    W -= lr * gradient                                       # 갱신
```
- **torch.optim 사용 (자동미분)**:
```python
W = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.SGD([W], lr=0.15)
for epoch in range(11):
    cost = torch.mean((x_train * W - y_train) ** 2)
    optimizer.zero_grad()   # grad 초기화
    cost.backward()         # 자동미분으로 gradient 계산
    optimizer.step()        # W -= lr * grad
```
- `backward()`가 위 수식의 미분을 자동으로 해 주는 것이 직접 구현과의 차이다.

## 자주 하는 실수 / 팁
- 직접 구현 시 `W`는 `requires_grad=False`(기본)로 두고 수식으로 직접 갱신한다. optim을 쓸 때는 `requires_grad=True`가 필요하다.
- lr이 적절하면 W가 정답(예제에선 1.0)으로 수렴한다. lr이 너무 크면 0.84↔1.06처럼 정답 주위를 진동하다 발산할 수 있다.
- 매 epoch `zero_grad()`를 호출하지 않으면 gradient가 누적된다.

## 예제 노트북 요약
- `example.ipynb`는 `x_train=y_train=[1,2,3]` (정답 W=1, b=0) 문제로 경사하강법을 학습한다.
- 먼저 미분 수식을 직접 계산해 `W -= lr*gradient`로 100 epoch 갱신해 W가 1.0으로 수렴함을 보이고, 이어 `torch.optim.SGD`로 같은 일을 자동화한다.

## 더 보기
- 경사하강법을 적용하는 기본 회귀: [`../01_linear_regression/concept.md`](../01_linear_regression/concept.md)
- 다변수 회귀에서의 학습: [`../02_multivariable_regression/concept.md`](../02_multivariable_regression/concept.md)
- 미니배치 경사하강법: [`../04_data_loading/concept.md`](../04_data_loading/concept.md)
