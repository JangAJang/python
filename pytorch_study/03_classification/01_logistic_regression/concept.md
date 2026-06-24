# 로지스틱 회귀 (Logistic Regression)

> 테마: 03_classification · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
입력 특징을 받아 시그모이드로 0~1 사이 "확률"을 출력하고, 임계값(보통 0.5)으로 두 클래스를 나누는 이진 분류 모델이다.

## 핵심 개념
- **이진 분류(binary classification)**: 정답 레이블이 0 또는 1 두 가지뿐인 문제.
- **확률 예측**: 모델은 곧바로 0/1을 내놓지 않고 $P(y=1 \mid x)$ 를 예측한다. 이후 임계값으로 라벨을 결정한다.
- **시그모이드(sigmoid)**: 선형 결합 $z = W^\top x + b$ 의 결과(임의의 실수)를 0~1 범위로 눌러 확률처럼 해석 가능하게 만든다.
- **손실 함수**: 분류에서는 MSE 대신 **이진 교차 엔트로피(BCE, Binary Cross Entropy)** 를 쓴다. 예측 확률이 정답에서 멀어질수록 손실이 급격히 커진다.
- **학습**: 경사하강법으로 BCE 손실을 최소화하도록 $W, b$ 를 갱신한다.

## 원리 / 수식
- 가설(hypothesis):
$$H(x) = \sigma(W^\top x + b) = \frac{1}{1 + e^{-(W^\top x + b)}}$$
- 시그모이드 정의: $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ — $z$가 커지면 1에, 작아지면 0에 수렴.
- 비용 함수(BCE), 샘플 $m$개:
$$\text{cost}(W) = -\frac{1}{m} \sum_{i=1}^{m} \Big( y^{(i)} \log H(x^{(i)}) + (1 - y^{(i)}) \log(1 - H(x^{(i)})) \Big)$$
  - 정답이 1인데 예측이 0에 가까우면 $-\log(H)$ 가 매우 커지고, 정답이 0인데 예측이 1에 가까우면 $-\log(1-H)$ 가 매우 커진다.
- 가중치 갱신(경사하강법): $W := W - \alpha \dfrac{\partial\, \text{cost}(W)}{\partial W}$ ($\alpha$ = 학습률)

## PyTorch 구현 포인트
- 가설은 `torch.sigmoid(x_train.matmul(W) + b)` — 직접 `1/(1+torch.exp(-z))` 로 구현해도 동일하지만 `torch.sigmoid`를 쓰는 것이 안전·간결.
- 손실은 `F.binary_cross_entropy(hypothesis, y_train)` — 직접 `-(y*log(H) + (1-y)*log(1-H)).mean()` 구현과 같은 값.
- 학습 루프 표준 3단계: `optimizer.zero_grad()` → `cost.backward()` → `optimizer.step()`.
- 모델은 `nn.Module` 클래스로 캡슐화하는 것이 권장:
  ```python
  class BinaryClassifier(nn.Module):
      def __init__(self):
          super().__init__()
          self.linear = nn.Linear(2, 1)   # 입력 2차원 -> 출력 1
          self.sigmoid = nn.Sigmoid()
      def forward(self, x):
          return self.sigmoid(self.linear(x))
  ```
- 예측/정확도: `prediction = hypothesis >= 0.5` 로 라벨 결정 후 `(prediction.float() == y_train)` 의 평균으로 정확도 계산.

## 자주 하는 실수 / 팁
- `y_train` 은 `FloatTensor` 여야 한다(BCE는 0.0/1.0 실수 타깃 사용). 다중분류의 `LongTensor` 인덱스와 헷갈리지 말 것.
- `F.binary_cross_entropy` 는 입력으로 **시그모이드를 거친 확률**을 받는다. 시그모이드 전의 로짓을 넣으면 안 된다(수치 안정성이 필요하면 `F.binary_cross_entropy_with_logits` 사용).
- 학습률이 너무 크면(예제처럼 `lr=1`) 손실이 출렁일 수 있다. 정확도가 중간에 떨어졌다 회복하는 것은 정상적 현상일 수 있으나 발산하면 학습률을 낮춘다.
- `nn.Linear(in, out)` 의 가중치 형상은 입력 차원으로 결정된다. 예제는 입력 2차원이므로 $W=(2,1)$, $b=(1)$.

## 예제 노트북 요약
- `example.ipynb` 는 2차원 입력 6개 샘플로 시그모이드 가설과 BCE 손실을 **직접 수식으로 구현**한 뒤, 동일 결과를 `torch.sigmoid` / `F.binary_cross_entropy` 로 간단화한다.
- 이후 1000 에폭 학습으로 비용이 0.69에서 0.02까지 감소하는 과정을 보이고, 임계값 0.5 기반 예측과 정확도를 계산한다.
- 마지막으로 `nn.Module` 기반 `BinaryClassifier` 클래스로 동일 학습을 재구성하여 정확도 100%에 도달한다.

## 더 보기
- 다중 클래스로 확장: [`../02_softmax_classification/concept.md`](../02_softmax_classification/concept.md)
- 회귀(연속값 예측)와의 대비: [`../../02_regression/01_linear_regression/concept.md`](../../02_regression/01_linear_regression/concept.md)
