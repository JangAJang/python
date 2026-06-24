# 전이학습 (Transfer Learning)

> 보강 학습 · C. 실전 운용 · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
대규모 데이터로 사전학습된 모델의 가중치를 가져와, 적은 데이터로도 새 과제에 빠르게 적응시키는 기법.

## 핵심 개념
- **사전학습 모델 (pre-trained)**: ImageNet 등으로 미리 학습된 모델. `torchvision.models`가 다양한 백본을 weights와 함께 제공.
- **feature extractor 방식**: 사전학습 가중치를 freeze(`requires_grad=False`)하고, 마지막 분류 layer만 새 클래스 수에 맞게 교체·학습. 데이터가 적을 때 유리.
- **fine-tuning 방식**: 일부 또는 전체 레이어를 함께 재학습. 데이터가 충분하고 도메인 차이가 클 때 유리.
- **학습률 차등 (discriminative LR)**: 사전학습된 하위 레이어는 작은 LR, 새로 추가한 head는 큰 LR을 줘 기존 표현을 보존하면서 적응시킨다.

## 동작 방식
- CNN의 하위 레이어는 엣지·질감 같은 범용 특징을, 상위 레이어는 과제 특화 특징을 학습한다. 범용 특징은 새 과제에도 재사용 가능하다.
- freeze한 레이어는 gradient가 계산되지 않아 학습 속도·메모리에서 이득이 있다.

## PyTorch 구현 포인트
```python
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 1) feature extractor: 전체 freeze 후 마지막 layer 교체
for p in model.parameters():
    p.requires_grad = False
in_feat = model.fc.in_features
model.fc = nn.Linear(in_feat, num_classes)   # 새 layer는 requires_grad=True

# 2) 학습률 차등 (fine-tuning)
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-4},  # 백본
    {"params": model.fc.parameters(),     "lr": 1e-3},  # head
])
```
- 교체한 `fc`(또는 `classifier`)는 기본적으로 `requires_grad=True`라 freeze해도 학습된다.
- 입력은 사전학습 시와 같은 전처리(정규화 mean/std)를 맞춰야 한다.

## 자주 하는 실수 / 팁
- freeze 후 `model.fc`만 새로 만들었는데 optimizer에 `model.parameters()` 전체를 넘기면, freeze된 파라미터까지 대상에 들어가니 `requires_grad=True`인 것만 넘기는 게 깔끔하다.
- 사전학습 모델의 정규화 통계를 무시하고 임의 전처리를 쓰면 성능이 크게 떨어진다.
- fine-tuning 시 LR이 너무 크면 사전학습 지식이 망가진다(catastrophic forgetting).
- BatchNorm을 가진 백본을 freeze할 때는 `model.eval()`로 통계 갱신을 막을지 고려한다.

## 더 보기
- PyTorch 튜토리얼: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- torchvision 모델: https://docs.pytorch.org/vision/stable/models.html
- 선행: [`../../06_cnn/04_resnet/concept.md`](../../06_cnn/04_resnet/concept.md) — 백본으로 쓰이는 ResNet
