# 학습 종합 팁 (Training Tips)

> 테마: 05_training_techniques · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
MLE 관점의 학습 의미, learning rate 조정, 데이터 전처리/정규화, 과적합 방지까지 DNN 학습의 실전 팁을 종합한다.

## 핵심 개념
- **Maximum Likelihood Estimation (MLE)**: 관측 데이터가 가장 그럴듯하게 나올 모델 파라미터를 찾는 것. $\theta_{MLE} = \arg\max_\theta L(\theta \mid x)$. 신경망 학습은 결국 데이터의 likelihood를 최대화(= negative log-likelihood, 즉 손실을 최소화)하는 파라미터 탐색이다.
- **Learning Rate**:
  - 너무 크면 발산(diverge)해 cost가 점점 커진다.
  - 너무 작으면 cost가 거의 줄지 않아 학습이 느리다.
  - 적절한 lr 선택이 수렴 속도와 안정성을 좌우한다.
- **데이터 전처리(Data Preprocessing)**: 입력 스케일이 제각각이면 학습이 어렵다. 표준화(standardization)로 계산이 수월해진다.
- **Overfitting(과적합)**: train에는 잘 맞지만 새 데이터에는 부정확. 방지법 → 데이터 늘리기, feature 줄이기, regularization.

## 원리 / 수식
- **MLE 예시(베르누이)**: 압정을 100번 던져 머리 27번 → $L(p) = p^{27}(1-p)^{73}$. 로그 취해 미분=0으로 풀면 $p = 0.27$. (로그를 씌우고 미분=0으로 두어 최적 파라미터를 구한다.)
- **표준화(Standardization)**: 각 feature를 평균 $\mu$, 표준편차 $\sigma$로 정규화.
  $$x_{norm} = \frac{x - \mu}{\sigma}$$
  데이터가 정규분포를 따른다고 가정하고 전처리한다.
- **Regularization 기법들**:
  - Early Stopping: validation loss가 더 안 줄면 멈춤.
  - Reducing Network Size: 망 크기 축소.
  - Weight Decay: 가중치 크기 제한.
  - Dropout: 뉴런 일부를 무작위로 끔.
  - Batch Normalization: 각 레이어 출력을 평균 0, 분산 1로.

## PyTorch 구현 포인트
- 손실/최적화: `F.cross_entropy`, `optim.SGD(model.parameters(), lr=...)`.
- 표준화:
```python
mu = x_train.mean(dim=0)
sigma = x_train.std(dim=0)
norm_x_train = (x_train - mu) / sigma
```
- DNN 학습 기본 절차:
  1. 망 설계(입력 차원 → 출력 클래스 수).
  2. 학습 후 과적합 체크 → 과적합 아니면 모델 키우고, 과적합이면 regularization 추가.
  3. 반복.

## 자주 하는 실수 / 팁
- train/test에 같은 $\mu, \sigma$(train 기준)를 적용해야 한다. test로 다시 fit하면 정보 누수.
- lr은 가장 먼저 튜닝할 하이퍼파라미터. 발산하면 낮추고, 너무 느리면 올린다.
- 과적합 신호: train cost는 계속 내려가는데 test/validation cost가 올라감.

## 예제 노트북 요약
- `example.ipynb`는 MLE 개념(베르누이 예제)으로 시작해, softmax 분류기로 learning rate가 크/작을 때의 발산·정체를 비교하고, 회귀 데이터에 표준화 전처리를 적용하는 과정을 보여준다.

## 더 보기
- 드롭아웃: [`../03_dropout/concept.md`](../03_dropout/concept.md)
- 배치 정규화: [`../04_batch_normalization/concept.md`](../04_batch_normalization/concept.md)
- ReLU/optimizer: [`../01_relu_activation/concept.md`](../01_relu_activation/concept.md)
