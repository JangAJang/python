# 규제 (Regularization)

> 보강 학습 · B. 일반화 & 평가 · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
규제는 모델이 train_set을 외우지 못하도록 제약·노이즈·중단을 가해 과적합을 억제하고 일반화 성능을 높이는 기법들의 총칭이다.

## 핵심 개념
- **L2(weight decay)**: 가중치 제곱합을 손실에 더해 큰 가중치를 억제, 더 매끄러운 함수를 선호하게 한다.
- **L1**: 가중치 절댓값 합을 더해 일부 가중치를 정확히 0으로 만들어 희소(sparse) 해를 유도(특징 선택 효과).
- **dropout**: 학습 중 뉴런을 확률 $p$로 무작위로 꺼서 특정 뉴런 의존을 줄이고 앙상블 효과를 낸다.
- **early stopping**: validation 손실이 더 이상 줄지 않으면 학습을 멈춰 train에 과도하게 맞춰지기 전에 정지한다.
- **data augmentation**: 데이터에 변형(flip/crop 등)을 가해 사실상 데이터를 늘려 모델이 불변성을 학습하게 한다.

## 원리 / 수식
- L2: $\mathcal L_{\text{total}} = \mathcal L + \frac{\lambda}{2}\|\theta\|_2^2$, L1: $\mathcal L + \lambda\|\theta\|_1$
- 공통 원리: 모델의 유효 자유도(effective capacity)를 낮춰 train·test 성능 격차(generalization gap)를 줄인다.
- dropout은 매 스텝 다른 부분망을 학습 → 가중치 평균화/앙상블로 해석된다.
- augmentation은 입력 공간에, dropout은 표현 공간에, weight decay는 파라미터 공간에 제약을 가하는 식으로 서로 보완한다.

## PyTorch 구현 포인트
```python
opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)  # L2

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                      nn.Dropout(p=0.5), nn.Linear(256, 10))

# early stopping (개요)
if val_loss < best: best, wait = val_loss, 0
else:
    wait += 1
    if wait >= patience: break
```
- L1은 내장 인자가 없어 손실에 `lambda * sum(p.abs().sum() for p in params)`를 직접 더한다.

## 자주 하는 실수 / 팁
- dropout은 학습 시에만 켠다. 평가 전 `model.eval()`을 빼먹으면 추론이 불안정해진다.
- weight decay를 너무 크게 주면 underfitting(과소적합)이 된다.
- early stopping은 반드시 **validation** 지표로 판단해야 한다(test로 멈추면 누수).
- augmentation은 train에만 적용하고, val/test에는 결정적(deterministic) 변환만 쓴다.
- 여러 규제를 한꺼번에 세게 걸면 underfitting이 되기 쉽다. 하나씩 추가하며 validation으로 효과를 확인한다.
- early stopping의 `patience`(개선 없이 기다릴 epoch 수)는 노이즈에 너무 민감하지 않게 적당히 둔다.

## 더 보기
- weight decay를 적용하는 옵티마이저: [`../../A_core_mechanics/03_optimizer/concept.md`](../../A_core_mechanics/03_optimizer/concept.md)
- d2l.ai Weight Decay: https://d2l.ai/chapter_linear-regression/weight-decay.html
- d2l.ai Dropout: https://d2l.ai/chapter_multilayer-perceptrons/dropout.html
