# 다변수 선형회귀 (Multivariable Linear Regression)

> 테마: 02_regression · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
입력 변수가 여러 개일 때 `H(x) = w1x1 + w2x2 + w3x3 + b`를 행렬곱(`matmul`)으로 한 번에 계산하고, `nn.Linear`로 W, b 관리를 자동화한다.

## 핵심 개념
- 단순 선형회귀는 입력 x가 하나였지만, 현실 데이터는 보통 여러 특성(feature)을 가진다.
- 가설: `H(x) = w1x1 + w2x2 + ... + wnxn + b`.
- 가중치를 일일이 손으로 선언/관리하는 대신 **행렬곱(matmul)** 으로 벡터화한다.
- `nn.Module` + `nn.Linear`를 쓰면 W, b 선언과 forward 계산이 자동화된다.
- 비용함수는 단순 선형회귀와 동일하게 MSE를 쓰며, `F.mse_loss`로 간결하게 표현 가능.

## 원리 / 수식
- 가설 (스칼라 형태): $H(x) = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$
- 행렬 형태 (샘플 m개, feature n개):
$$H(X) = X W + b, \quad X \in \mathbb{R}^{m \times n},\ W \in \mathbb{R}^{n \times 1}$$
- `matmul` 한 번으로 모든 샘플의 예측을 동시에 계산한다. feature가 늘어도 W의 행 차원만 바꾸면 된다.
- 비용함수 (MSE): $\text{cost} = \frac{1}{m}\sum (H(x^{(i)}) - y^{(i)})^2$

## PyTorch 구현 포인트
- 직접 구현:
```python
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
hypothesis = x_train.matmul(W) + b      # 행렬곱으로 벡터화
optimizer = torch.optim.SGD([W, b], lr=1e-5)
```
- `nn.Module`로 모듈화:
```python
class MultivariableLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)   # (입력차원, 출력차원)
    def forward(self, x):
        return self.linear(x)

model = MultivariableLinearRegression()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
cost = F.mse_loss(model(x_train), y_train)
```
- `nn.Linear(in_features, out_features)` 가 W, b를 내부에서 초기화/관리한다.
- `F.mse_loss(prediction, y_train)` — Cost Function을 라이브러리로 쓰면 다른 함수로 교체/디버깅이 쉽다.
- `tensor.detach()` — autograd 그래프에서 분리. 출력/저장용으로 쓴다. `tensor.detach().cpu().numpy()` 패턴: detach(그래프 분리) → cpu(GPU→CPU 이동) → numpy(배열 변환).

## 자주 하는 실수 / 팁
- feature가 큰 값(예: 점수 73~100)일 때 lr이 크면 발산한다. 예제는 `lr=1e-5`로 작게 잡는다. 실무에선 입력을 표준화/정규화하면 더 큰 lr을 쓸 수 있다.
- 예제 노트북 마지막 학습 루프에 변수 오타가 있다: `Hypothesis.squeeze()` (대문자 H)는 정의되지 않은 이름이라 실제로는 `hypothesis.squeeze()` (소문자)여야 한다. 출력의 hypothesis가 변하지 않는 것도 이 print 오류 때문.
- `W = torch.zeros((3,1))` 처럼 출력 차원(1)을 명시해 형태를 맞춰야 `matmul` 결과가 `(m,1)`이 된다.

## 예제 노트북 요약
- `example.ipynb`는 3개 시험 점수(x: 5x3)로 최종 점수(y: 5x1)를 예측한다.
- 먼저 `matmul`을 이용한 직접 구현으로 20 epoch 학습하고, 이후 동일 문제를 `nn.Linear` + `F.mse_loss`로 모듈화해 다시 학습한다.

## 더 보기
- 입력이 하나인 기본형: [`../01_linear_regression/concept.md`](../01_linear_regression/concept.md)
- 학습에 쓰이는 경사하강법: [`../03_gradient_descent/concept.md`](../03_gradient_descent/concept.md)
- 데이터가 많을 때의 적재: [`../04_data_loading/concept.md`](../04_data_loading/concept.md)
