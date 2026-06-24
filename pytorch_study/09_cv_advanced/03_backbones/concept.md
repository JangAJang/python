# 최신 백본 (EfficientNet & Vision Transformer)

> 테마: 09_cv_advanced · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
EfficientNet은 CNN을 효율적으로 키우는 compound scaling을, ViT는 이미지를 patch 시퀀스로 보는 Transformer를 제시한 현대 백본이다.

## 핵심 개념
- **EfficientNet**: 깊이(depth)·너비(width)·해상도(resolution)를 하나의 계수 φ로 균형 있게 함께 키우는 **compound scaling**. MobileNet의 효율 블록(MBConv)을 기반으로 적은 연산으로 높은 정확도 달성.
- **ViT (Vision Transformer)**: 이미지를 고정 크기 patch(예: 16x16)로 잘라 각 patch를 벡터로 임베딩한 뒤 Transformer encoder에 입력.
- **patch embedding**: 각 patch를 flatten 후 Linear로 임베딩하고 위치 정보를 위해 **positional embedding**을 더한다.
- **[CLS] token**: 시퀀스 앞에 붙이는 학습 가능한 토큰. 최종 [CLS] 표현을 분류 head에 사용.
- **inductive bias**: CNN은 locality·translation equivariance라는 강한 사전 가정을 가지나, ViT는 이를 거의 갖지 않아 **대규모 데이터(또는 사전학습)**가 있어야 CNN을 능가한다.

## 원리 / 수식
- compound scaling: depth `d=α^φ`, width `w=β^φ`, resolution `r=γ^φ`, 제약 `α·β²·γ² ≈ 2`.
- ViT 입력 토큰 수 = `(H·W)/P²` + 1([CLS]). self-attention은 토큰 수에 대해 O(N²) 비용.
- CNN vs ViT: CNN은 적은 데이터에서도 잘 일반화, ViT는 데이터가 많을수록 표현력이 커지고 전역 문맥을 직접 모델링.

## PyTorch 구현 포인트
```python
from torchvision.models import efficientnet_b0, vit_b_16

cnn = efficientnet_b0(weights="DEFAULT")
vit = vit_b_16(weights="DEFAULT")     # patch=16, ImageNet 사전학습
logits = vit(img_batch)               # [N, 1000]
```
- 두 모델 모두 ImageNet 사전학습 가중치 제공. 전이학습 시 마지막 분류 head만 교체.
- ViT는 입력 해상도가 사전학습 설정(보통 224)과 맞아야 positional embedding이 일치한다.

## 자주 하는 실수 / 팁
- ViT를 소규모 데이터셋에서 from scratch로 학습하면 CNN보다 성능이 나쁘다 — 사전학습/증강 필수.
- EfficientNet은 해상도도 함께 키우는 모델이라 입력 크기를 모델 변형(b0~b7)에 맞춰야 한다.
- `weights="DEFAULT"`가 요구하는 전처리(normalize)를 `weights.transforms()`로 그대로 적용할 것.

## 더 보기
- 선행 개념: [`../../06_cnn/07_efficientnet/concept.md`](../../06_cnn/07_efficientnet/concept.md) — EfficientNet 상세
- 선행 개념: [`../../08_transformer/01_attention/concept.md`](../../08_transformer/01_attention/concept.md) — self-attention
