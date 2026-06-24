# 평가 지표 (Evaluation Metrics)

> 보강 학습 · B. 일반화 & 평가 · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
accuracy 하나로는 불균형 데이터를 평가할 수 없으므로 precision/recall/F1, confusion matrix, ROC-AUC(분류)와 RMSE/R²(회귀)를 목적에 맞게 함께 본다.

## 핵심 개념
- **accuracy의 한계**: 95:5 불균형에서 전부 다수 클래스로 찍어도 정확도 0.95가 나와 성능을 과대평가한다.
- **confusion matrix**: TP/FP/FN/TN을 표로 정리한 것. 대부분의 분류 지표가 여기서 파생된다.
- **precision(정밀도)**: 양성이라 예측한 것 중 실제 양성 비율. **recall(재현율)**: 실제 양성 중 잡아낸 비율.
- **F1**: precision과 recall의 조화평균. 둘의 균형을 본다.
- **ROC-AUC**: 임계값을 바꿔가며 그린 TPR-FPR 곡선 아래 면적. 1에 가까울수록 좋고, 임계값에 독립적.

## 원리 / 수식
- $\text{precision} = \dfrac{TP}{TP+FP},\quad \text{recall} = \dfrac{TP}{TP+FN}$
- $F_1 = 2\cdot\dfrac{\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}}$
- $\text{RMSE} = \sqrt{\frac{1}{n}\sum (y_i-\hat y_i)^2}$ (타깃과 같은 단위)
- $R^2 = 1 - \dfrac{\sum (y_i-\hat y_i)^2}{\sum (y_i-\bar y)^2}$ (1에 가까울수록 설명력 높음, 음수 가능)
- 다중 클래스의 precision/recall은 클래스별로 구한 뒤 macro(단순 평균)·weighted(빈도 가중)로 종합한다.

## PyTorch 구현 포인트
```python
pred = logits.argmax(dim=1)
acc = (pred == target).float().mean()

tp = ((pred == 1) & (target == 1)).sum().float()
fp = ((pred == 1) & (target == 0)).sum().float()
fn = ((pred == 0) & (target == 1)).sum().float()
precision = tp / (tp + fp + 1e-8)
recall    = tp / (tp + fn + 1e-8)
```
- ROC-AUC·confusion matrix는 보통 `sklearn.metrics`(`roc_auc_score`, `confusion_matrix`)로 계산한다.

## 자주 하는 실수 / 팁
- 불균형 데이터에서 accuracy만 보고하면 안 된다. recall/precision/F1을 함께 본다.
- precision↔recall은 임계값(threshold)에 따라 트레이드오프된다. 목적(놓치면 안 됨→recall, 오탐 비용 큼→precision)에 맞춰 고른다.
- ROC-AUC에는 클래스 예측이 아니라 **확률/점수**를 넣어야 한다.
- 회귀에서 R²는 음수가 될 수 있다(평균보다 못한 모델).
- 지표는 반드시 train이 아닌 validation/test에서 계산해야 일반화 성능을 반영한다.
- RMSE는 타깃과 같은 단위라 해석이 쉽고, MAE보다 큰 오차에 더 큰 페널티를 준다.

## 더 보기
- 지표를 계산할 데이터 분할: [`../03_data_split/concept.md`](../03_data_split/concept.md)
- d2l.ai Classification 기초: https://d2l.ai/chapter_linear-classification/classification.html
- PyTorch 빠른 시작(평가 루프): https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
