# ResNet (Residual Network)

> 테마: 06_cnn · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
입력을 출력에 그대로 더하는 잔차 연결(skip connection)을 도입해, 매우 깊은 네트워크에서도 성능 저하와 기울기 소실 없이 학습이 가능하게 만든 CNN 아키텍처.

## 핵심 개념
- **깊은 망의 두 문제**:
  - **성능 저하(Degradation)**: 레이어를 더 쌓았는데 오히려 학습/테스트 오차가 증가. 과적합이 아니라 최적화 자체가 어려워지는 현상.
  - **기울기 소실(Vanishing Gradient)**: 역전파가 여러 레이어를 거치며 gradient가 점점 작아져 초기 레이어가 거의 학습되지 않음.
- **Residual Learning**: 완전한 함수 `H(x)`를 직접 학습하는 대신, 잔차 `F(x) = H(x) - x`만 학습하고 `H(x) = F(x) + x`로 복원한다.
- **Residual Block / Skip Connection**: `y = F(x, Wi) + x`. Conv 레이어 출력에 입력을 그대로 더한다.
  ```
  x ─► [Conv → BN → ReLU → Conv → BN] ─► + ─► ReLU
    └────────────────────────────────────┘ (identity)
  ```
- **깊이 변형**: ResNet-18/34는 `BasicBlock`, ResNet-50/101/152는 `BottleNeck`(1x1 → 3x3 → 1x1, expansion=4)을 사용한다.

## 원리 / 수식
- `y = F(x) + x` 이므로 역전파 시 `dL/dx` 경로에 항등 항(+1)이 항상 존재해 gradient가 소실되지 않고 전달된다.
- 불필요한 레이어는 `F(x) = 0`으로 학습되어 `y = x`(항등 사상)가 되므로, 깊게 쌓아도 손해가 없다.
- **downsample이 필요한 이유**: `out += identity`는 두 텐서의 shape이 같아야 한다.
  - `stride > 1` → 공간 크기(H, W) 불일치
  - 채널 수 변화(예: 64→128) → 채널(C) 불일치
  - 이 경우 `downsample = conv1x1(stride) + BatchNorm`으로 identity를 out에 맞춰 변환한다. shape이 맞으면 `downsample=None`.

## PyTorch 구현 포인트
- `conv3x3`, `conv1x1` 헬퍼로 반복 conv를 정의.
- `BasicBlock`(expansion=1), `BottleNeck`(expansion=4) 클래스로 블록 정의. `forward`에서 `out += identity` 후 `relu`.
- `ResNet._make_layer(block, planes, blocks, stride)` : 첫 블록만 stride/downsample 적용하고 나머지는 반복.
- `nn.AdaptiveAvgPool2d((1,1))` → `nn.Linear(512*expansion, num_classes)`.
- `model_zoo.load_url(model_urls[...])` : ImageNet pre-trained 가중치 로드.

```python
def forward(self, x):
    identity = x
    out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
    out = self.conv2(out); out = self.bn2(out)
    if self.downsample is not None:
        identity = self.downsample(x)
    out += identity
    return self.relu(out)
```

## 자주 하는 실수 / 팁
- `conv1x1`에 `padding=1`을 주면 1x1 conv임에도 공간 크기가 어긋날 수 있다(예제 코드의 미묘한 점). downsample의 출력 크기가 out과 정확히 일치하는지 항상 확인.
- `out += identity`는 ReLU **이전**에 더해야 한다. 순서를 바꾸면 잔차 학습 효과가 사라진다.
- 50층 이상은 BottleNeck을 써야 파라미터/연산이 감당 가능하다.
- 예제 코드에 일부 오타(`conv3x3(...)@`, `Bottleneck` vs `BottleNeck` 대소문자)가 있으니 직접 실행 시 수정 필요.

## 예제 노트북 요약
- `example.ipynb`는 torchvision 스타일로 `conv3x3/conv1x1`, `BasicBlock`, `BottleNeck`, `ResNet`을 직접 구현하고 resnet18~152 생성 함수를 정의한다.
- skip connection과 downsample의 동작 원리를 표/그림으로 설명하고, `resnet34()` 구조를 출력해 레이어 구성을 확인한다.

## 더 보기
- 선행 개념: [`../03_vgg/concept.md`](../03_vgg/concept.md) — 단순 깊은 CNN
- 배치 정규화: 잔차 블록 내부의 BatchNorm은 [`../../04_neural_network`](../../04_neural_network) 관련 자료 참고
- 다음 단계: [`../05_image_folder/concept.md`](../05_image_folder/concept.md) — 커스텀 데이터로 학습
