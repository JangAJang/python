# 데이터 분할 & 검증 (Data Splitting & Cross Validation)

> 보강 학습 · B. 일반화 & 평가 · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
모델을 정직하게 평가하려면 train/val/test를 분리하고, test는 마지막까지 건드리지 않으며, 전처리 통계는 train에서만 학습해 데이터 누수를 막아야 한다.

## 핵심 개념
- **train / validation / test**: train으로 학습, validation으로 하이퍼파라미터·early stopping 결정, test로 최종 성능을 *한 번* 보고한다.
- **k-fold cross validation**: 데이터를 k등분해 한 조각을 검증, 나머지로 학습하기를 k번 반복 후 평균. 데이터가 적을 때 평가 분산을 줄인다.
- **데이터 누수(data leakage)**: 평가에 쓸 정보가 학습에 스며드는 것. 성능을 비현실적으로 부풀린다.
- **stratified split**: 클래스 비율을 분할마다 유지해 불균형 데이터의 평가를 안정화한다.

## 원리 / 수식
- test를 튜닝에 쓰면 그 지표는 더 이상 일반화 성능이 아니다(test에 과적합).
- k-fold: 각 fold 점수 $s_i$의 평균 $\bar s = \frac{1}{k}\sum s_i$를 성능 추정치로 쓴다.
- 누수의 전형: 분할 *전* 전체 데이터로 스케일러를 fit → val/test 통계가 train에 새어든다.
- 데이터가 충분하면 단순 hold-out(고정 분할)으로 족하고, 적을수록 k-fold의 평균이 추정 분산을 줄여 가치가 커진다.

## PyTorch 구현 포인트
```python
from torch.utils.data import random_split
n = len(ds); n_val = int(0.2 * n)
train_ds, val_ds = random_split(ds, [n - n_val, n_val])

# 누수 방지: 스케일러는 train에만 fit, 나머지엔 transform만
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)    # fit 금지!
```
- k-fold 인덱스는 `sklearn.model_selection`의 `KFold`/`StratifiedKFold`로 만들고, fold마다 `Subset`/`DataLoader`를 구성한다.

## 자주 하는 실수 / 팁
- 스케일러·정규화·결측치 대치 등 모든 fit 단계는 train에서만 한다. val/test는 transform만.
- 시계열은 무작위 분할이 미래→과거 누수를 일으킨다. 시간 순서로 분할한다.
- test로 모델을 고르거나 멈추면 안 된다(그건 validation의 역할).
- 분할 전 중복 샘플·동일 개체(같은 환자/사용자)가 train과 test에 걸치지 않게 그룹 단위로 나눈다.
- 재현성을 위해 `random_split`에 `generator=torch.Generator().manual_seed(seed)`로 시드를 고정한다.
- 불균형 데이터는 단순 무작위 분할 시 소수 클래스가 한쪽에 쏠릴 수 있어 stratified 분할을 권장한다.

## 더 보기
- 분할된 데이터에 적용하는 전처리·증강: [`../04_preprocessing/concept.md`](../04_preprocessing/concept.md)
- d2l.ai Model Selection·Cross-Validation: https://d2l.ai/chapter_linear-regression/generalization.html
- PyTorch Dataset/DataLoader: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
