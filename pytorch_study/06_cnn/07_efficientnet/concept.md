# EfficientNet (Compound Scaling)

> 테마: 06_cnn · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
depth(층 수)·width(채널 수)·resolution(입력 해상도)을 단일 계수 φ로 균형 있게 동시에 스케일링하여, ResNet 등보다 훨씬 적은 파라미터로 더 높은 정확도를 달성한 CNN 아키텍처 계열이다.

## 핵심 개념
- **세 가지 스케일링 축**:
  - **depth**: 레이어를 더 깊게 쌓아 복잡한 특징을 학습.
  - **width**: 채널 수를 늘려 더 세밀한 특징을 표현.
  - **resolution**: 입력 이미지 해상도를 높여 더 작은 패턴까지 인식.
- **compound scaling**: 한 축만 키우면 금방 정확도가 포화된다. 세 축을 함께 키우되 비율을 고정해 균형을 맞추는 것이 핵심 아이디어.
- **baseline (B0)**: NAS(Neural Architecture Search)로 정확도/연산량을 함께 최적화해 찾은 작은 기준 모델. 여기에 compound scaling을 적용해 B1~B7로 키운다.
- **MBConv 블록**: MobileNetV2의 inverted residual + depthwise separable convolution 구조를 기본 빌딩 블록으로 사용한다.
- **Squeeze-and-Excitation(SE)**: 채널별 중요도를 학습해 가중치를 재조정하는 attention 모듈을 블록 내부에 포함.

## 원리 / 수식
- depth `d = α^φ`, width `w = β^φ`, resolution `r = γ^φ` 로 정의하고, `α·β²·γ² ≈ 2` 제약을 둔다.
- 제약의 직관: 해상도와 width를 2배로 키우면 FLOPs는 각각 제곱으로 늘기 때문에 `β², γ²` 항이 붙는다. φ가 1 늘 때 전체 연산량이 약 2배가 되도록 맞춘 것.
- `α, β, γ`는 작은 그리드 서치로 한 번만 찾고(B0 기준), 이후 φ만 키워 B1~B7을 만든다.
- inverted residual: `1x1 expand → 3x3 depthwise → 1x1 project`로 채널을 늘렸다가 줄이며, project 출력에 skip connection을 더한다.

## PyTorch 구현 포인트
```python
import torchvision.models as models
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
# 전이학습: 마지막 classifier만 교체
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
```
- torchvision은 `efficientnet_b0` ~ `efficientnet_b7`을 내장 제공한다.
- B0~B7로 갈수록 권장 입력 해상도가 커진다(B0 224 → B7 600 부근). 데이터 전처리 해상도를 모델에 맞춰야 한다.
- depthwise separable conv는 `nn.Conv2d(..., groups=in_channels)`(depthwise) + `1x1 Conv2d`(pointwise)로 구현된다.

## 자주 하는 실수 / 팁
- 한 축(예: 해상도)만 무리하게 키우면 정확도 향상이 빠르게 포화되고 연산만 늘어난다. 세 축을 함께 키워야 효율적이다.
- 모델 크기에 맞는 입력 해상도를 쓰지 않으면 사전학습 효과가 떨어진다.
- 파라미터 수가 적다고 학습/추론 메모리도 항상 적은 것은 아니다. 활성화 메모리는 해상도에 민감하다.
- 미세한 구조 변형(EfficientNetV2 등)은 학습 속도를 개선했으니, 단순 정확도 외에 학습 비용도 함께 고려하자.

## 더 보기
- 선행 개념: [`../04_resnet/concept.md`](../04_resnet/concept.md) — 잔차 연결과 깊은 망
- 선행 개념: [`../03_vgg/concept.md`](../03_vgg/concept.md) — 단순 깊은 CNN과 표준 구조
