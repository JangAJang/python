# VGG (VGGNet)

> 테마: 06_cnn · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
3x3 작은 합성곱만 반복해서 깊게 쌓은 단순하고 규칙적인 CNN 아키텍처로, features(합성곱부) → avgpool → classifier(FC부)의 표준 구조를 가진다.

## 핵심 개념
- **VGG의 철학**: 큰 커널 대신 **3x3 conv를 여러 번 쌓아** 같은 수용 영역(receptive field)을 더 적은 파라미터와 더 많은 비선형성으로 표현한다.
- **모델 변형**: 깊이에 따라 `vgg11, vgg13, vgg16, vgg19`가 있고, `_bn` 접미사는 BatchNorm을 추가한 버전이다.
- **구조 3단계**:
  - `features` : Conv 블록들의 묶음(`nn.Sequential`). 이미지에서 특징을 추출.
  - `avgpool` : `nn.AdaptiveAvgPool2d((7, 7))` 로 입력 크기에 상관없이 출력을 7x7로 고정.
  - `classifier` : `Linear(512·7·7, 4096) → ReLU → Dropout → Linear(4096,4096) → ReLU → Dropout → Linear(4096, num_classes)`.
- **pre-trained weights**: `model_urls`에 ImageNet으로 학습된 가중치 다운로드 URL이 정의되어 있어, 전이학습에 활용 가능하다.

## 원리 / 수식
- `forward`: `x → features(conv) → avgpool → x.view(N, -1) → classifier(FC) → logits`.
- `AdaptiveAvgPool2d`는 입력 해상도가 달라도 FC 입력 차원(512·7·7)을 일정하게 만들어 준다.
- 가중치 초기화: Conv는 He(kaiming) 초기화, BatchNorm은 weight=1/bias=0, Linear는 평균0·표준편차0.01 정규분포로 초기화한다.

## PyTorch 구현 포인트
- `class VGG(nn.Module)`로 `features`, `avgpool`, `classifier`를 조립.
- `nn.AdaptiveAvgPool2d((7,7))` : 가변 입력 대응.
- `nn.Dropout()` : FC에서 과적합 방지.
- `nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')` : Conv 초기화.
- 실제 사용 시 `torchvision.models.vgg16(weights=...)` 처럼 내장 모델을 바로 쓸 수도 있다.

```python
self.features = features                       # conv 블록
self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
self.classifier = nn.Sequential(
    nn.Linear(512*7*7, 4096), nn.ReLU(True), nn.Dropout(), ...)
```

## 자주 하는 실수 / 팁
- 예제의 `_initialize_weights`에는 첫 분기마다 `return`이 들어 있어 모든 모듈을 순회하지 못하는 점에 주의(학습용 코드의 한계). 실제로는 `continue`나 `elif`로 모든 레이어를 초기화해야 한다.
- VGG는 FC층 파라미터가 매우 커서 메모리를 많이 쓴다. 작은 데이터셋에는 그대로 쓰기보다 전이학습/특징 추출기로 쓰는 편이 좋다.
- 입력은 보통 `3x224x224`(RGB)를 가정한다.

## 예제 노트북 요약
- `example.ipynb`는 torchvision 스타일의 `VGG` 클래스를 직접 정의해 features/avgpool/classifier 구조와 가중치 초기화 로직을 보여준다.
- 깊이별 변형(vgg11~vgg19)과 pre-trained 가중치 URL 딕셔너리를 함께 정리한다.

## 더 보기
- 선행 개념: [`../02_mnist_cnn/concept.md`](../02_mnist_cnn/concept.md) — 기본 CNN
- 다음 단계: [`../04_resnet/concept.md`](../04_resnet/concept.md) — 잔차 연결로 더 깊게
- 전이학습 데이터: [`../05_image_folder/concept.md`](../05_image_folder/concept.md)
