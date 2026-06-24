# 소프트맥스 분류 (Softmax Classification)

> 테마: 03_classification · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
여러 클래스에 대한 점수(로짓)를 소프트맥스로 확률 분포로 변환하고, 교차 엔트로피로 정답 분포와의 차이를 최소화하는 다중 분류 모델이다.

## 핵심 개념
- **다중 분류(multi-class classification)**: 정답이 3개 이상의 클래스 중 하나인 문제.
- **소프트맥스(softmax)**: 클래스 개수만큼의 실수 점수 벡터를 받아 합이 1인 확률 분포로 바꾼다. 가장 큰 점수가 가장 큰 확률을 받는다.
- **이산 확률 분포 + 원-핫(one-hot)**: 정답은 클래스 인덱스 하나이며, 이를 해당 위치만 1인 원-핫 벡터로 표현해 예측 확률 분포와 비교한다.
- **교차 엔트로피(cross entropy)**: 두 확률 분포(정답 분포 vs 예측 분포)의 차이를 측정하는 손실. 정답에 가까울수록 손실이 작아진다.
- **NLL / log_softmax**: `cross_entropy = log_softmax + nll_loss` 로 분해되며, 수치 안정성을 위해 보통 합쳐서 쓴다.

## 원리 / 수식
- 소프트맥스: $k$개 클래스, 점수 벡터 $z=(z_1,\dots,z_k)$ 에 대해
$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}, \qquad \sum_i \text{softmax}(z)_i = 1$$
- 교차 엔트로피 손실(정답 분포 $p$, 예측 분포 $q$):
$$\text{CE}(p, q) = -\sum_x p(x)\,\log q(x)$$
  정답이 원-핫이므로 사실상 정답 클래스의 예측 로그확률만 남는다: $-\log q(\text{정답})$.
- 분해 관계: $\text{cross\_entropy}(z, y) = \text{nll\_loss}(\text{log\_softmax}(z), y)$.
  - `nll` = negative log likelihood(음의 로그 가능도).

## PyTorch 구현 포인트
- 확률 분포: `F.softmax(z, dim=1)` — 행(샘플)마다 클래스 차원(`dim=1`)으로 정규화.
- 원-핫 만들기: `y_one_hot = torch.zeros_like(hyp); y_one_hot.scatter_(1, y.unsqueeze(1), 1)`.
- 손실은 직접 `(y_one_hot * -torch.log(softmax)).sum(dim=1).mean()` 으로 가능하지만, 보통 한 줄로:
  ```python
  cost = F.cross_entropy(z, y_train)   # z = 로짓(softmax 적용 전!), y = LongTensor 인덱스
  ```
- 모델은 마지막에 소프트맥스를 넣지 않고 **로짓을 그대로 출력**하는 `nn.Linear(in, num_classes)` 로 두고, 손실에서 `F.cross_entropy` 가 소프트맥스를 처리하게 한다.
  ```python
  class SoftmaxClassifierModel(nn.Module):
      def __init__(self):
          super().__init__()
          self.linear = nn.Linear(4, 3)   # 4차원 입력 -> 3 클래스
      def forward(self, x):
          return self.linear(x)            # 로짓 반환
  ```

## 자주 하는 실수 / 팁
- `F.cross_entropy` 에는 **소프트맥스를 적용하지 않은 로짓**을 넣어야 한다. 소프트맥스를 두 번 적용하면(예: 모델 출력에 softmax + cross_entropy) 학습이 느려지거나 잘못된다. 실제로 예제의 low-level 학습 셀은 `F.softmax(... )` 결과에 다시 `F.softmax`를 적용하는 형태라 비용 감소가 둔하다 — 권장 형태는 로짓에 바로 `cross_entropy` 를 적용하는 것.
- 타깃 `y_train` 은 원-핫이 아니라 **클래스 인덱스(`LongTensor`)** 여야 한다. `cross_entropy` 가 내부에서 처리한다.
- 이진 분류(BCE)와 달리 출력 차원은 클래스 수와 같아야 한다(`nn.Linear(in, num_classes)`).
- `dim` 인자 주의: 배치 차원이 0, 클래스 차원이 1이면 `softmax(dim=1)`.

## 예제 노트북 요약
- `example.ipynb` 는 먼저 `F.softmax` 로 점수 벡터를 확률 분포로 바꾸고 합이 1임을 확인한다.
- 이어서 교차 엔트로피를 원-핫 + `-log` 로 직접 구현하고, 같은 값을 `F.log_softmax` + `F.nll_loss`, 최종적으로 `F.cross_entropy` 로 단계적으로 단순화한다.
- 4차원 입력·3클래스 데이터로 1000 에폭 학습을 low-level / `F.cross_entropy` / `nn.Module` 세 방식으로 비교하며 비용 감소를 보인다.

## 더 보기
- 이진 분류(시그모이드·BCE)와의 대비: [`../01_logistic_regression/concept.md`](../01_logistic_regression/concept.md)
- 신경망으로의 확장: [`../../04_neural_network/`](../../04_neural_network/)
