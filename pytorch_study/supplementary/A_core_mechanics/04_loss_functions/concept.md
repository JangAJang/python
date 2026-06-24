# 손실 함수 정리 (Loss Functions)

> 보강 학습 · A. 학습의 핵심 메커니즘 · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
회귀에는 MSE/MAE, 이진 분류에는 `BCEWithLogitsLoss`, 다중 분류에는 `CrossEntropyLoss`를 쓰며, 둘 다 logit을 입력으로 받는다는 점이 핵심이다.

## 핵심 개념
- **MSE(L2 손실)**: 오차 제곱의 평균. 큰 오차에 민감(이상치에 취약)하지만 미분이 매끄럽다.
- **MAE(L1 손실)**: 오차 절댓값의 평균. 이상치에 강건하나 0에서 미분 불연속.
- **`BCEWithLogitsLoss`**: 이진 분류용. 모델의 raw **logit**을 받아 내부에서 sigmoid + binary cross entropy를 한 번에 계산(수치 안정적).
- **`CrossEntropyLoss`**: 다중 분류용. 클래스별 **logit**을 받아 내부에서 log_softmax + NLL을 계산한다. 타깃은 클래스 인덱스(LongTensor).
- **logit vs 확률**: logit은 softmax/sigmoid 적용 *전*의 raw 점수. 위 두 손실에는 확률이 아니라 logit을 넣는다.

## 원리 / 수식
- MSE: $\frac{1}{n}\sum (y_i-\hat y_i)^2$, MAE: $\frac{1}{n}\sum |y_i-\hat y_i|$
- 이진 BCE: $-\frac{1}{n}\sum\big[y\log\sigma(z) + (1-y)\log(1-\sigma(z))\big]$, $z$는 logit.
- 다중 CE: $-\frac{1}{n}\sum \log\frac{e^{z_{y}}}{\sum_j e^{z_j}}$ — 정답 클래스 logit의 log_softmax.
- logit을 그대로 받아 log-sum-exp로 묶어 계산하면 sigmoid/softmax를 따로 한 뒤 log를 취할 때 생기는 overflow/underflow를 피한다(수치 안정성).

## PyTorch 구현 포인트
```python
# 회귀
mse = nn.MSELoss()(pred, target)        # nn.L1Loss()는 MAE

# 이진 분류: 출력 1차원 logit, target은 float(0/1)
bce = nn.BCEWithLogitsLoss()(logit, target.float())

# 다중 분류: 출력 (N, C) logit, target은 (N,) Long 인덱스
ce = nn.CrossEntropyLoss()(logits, target_idx)
```
- 모델 마지막에 sigmoid/softmax를 **넣지 말고** logit을 그대로 출력한다.

## 자주 하는 실수 / 팁
- `BCEWithLogitsLoss`에 이미 sigmoid를 통과한 확률을 넣으면 이중 적용으로 잘못된다(확률을 넣어야 하면 `BCELoss`).
- `CrossEntropyLoss`에 softmax 출력을 넣으면 안 된다 — logit이어야 한다. 타깃을 one-hot으로 주는 것도(인덱스가 원칙) 실수.
- 선택 기준: 연속값→MSE/MAE, 두 클래스→BCEWithLogits, 셋 이상 배타적 클래스→CrossEntropy.
- 클래스 불균형이면 `pos_weight`(BCE)·`weight`(CE) 인자로 보정한다.
- `CrossEntropyLoss`의 출력 차원은 클래스 수($C$)와 같아야 한다. 이진 분류라도 2클래스 CE로 풀 수 있지만, 보통 1차원 logit + BCEWithLogits가 간결하다.
- 이상치가 많은 회귀에서는 MSE와 MAE의 절충인 `nn.SmoothL1Loss`(Huber)를 고려한다.

## 더 보기
- 손실에 들어가는 logit을 만드는 nn.Module: [`../02_nn_module/concept.md`](../02_nn_module/concept.md)
- PyTorch 학습/손실 튜토리얼: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- d2l.ai Softmax Regression: https://d2l.ai/chapter_linear-classification/softmax-regression.html
