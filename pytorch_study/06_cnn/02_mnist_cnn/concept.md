# MNIST CNN (MNIST Convolutional Neural Network)

> 테마: 06_cnn · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
Conv → ReLU → MaxPool 블록을 쌓아 만든 CNN으로 MNIST 손글씨 숫자를 분류하며, 합성곱 기반 모델이 완전연결망보다 이미지에서 높은 정확도를 내는 것을 확인한다.

## 핵심 개념
- **딥러닝 학습 8단계**: ① 라이브러리 import ② GPU/seed 설정 ③ 하이퍼파라미터 설정 ④ 데이터셋 로드 ⑤ 모델 정의 ⑥ 손실함수·옵티마이저 선택 ⑦ 학습 루프 + loss 확인 ⑧ 성능 평가.
- **CNN 블록 구조**: `Conv2d → ReLU → MaxPool2d`를 하나의 레이어로 묶어(`nn.Sequential`) 여러 개 쌓는다.
- **특성 맵 크기 추적**: padding=1, kernel=3, stride=1 의 conv는 크기를 유지하고, MaxPool2d(2)는 절반으로 줄인다. 28 → (conv) 28 → (pool) 14 → 7 ... 식으로 줄어든다.
- **분류기(FC)**: 마지막 특성 맵을 `view`로 펼친 뒤 `nn.Linear`로 10개 클래스 점수를 낸다.
- **레이어 추가 실험**: 2개 레이어(약 99.69%)에서 3번째 conv 레이어 + FC를 추가하면 정확도가 더 오른다(약 99.96%).

## 원리 / 수식
- 2 레이어 모델: `28x28 → conv(1→32) → pool → 14x14 → conv(32→64) → pool → 7x7 → flatten(7·7·64) → Linear → 10`.
- 3 레이어 모델: conv를 하나 더 쌓아 `7→3`으로 줄이고 `3·3·128 → 625 → 10` 으로 FC를 2단으로 구성한다.
- 손실: `CrossEntropyLoss` (softmax + NLL을 한 번에 처리, logits를 그대로 입력).

## PyTorch 구현 포인트
- `dsets.MNIST(root, train, transform=transforms.ToTensor(), download=True)` : 데이터셋 로드.
- `nn.Sequential(nn.Conv2d(...), nn.ReLU(), nn.MaxPool2d(2))` : conv 블록.
- `torch.nn.init.xavier_uniform_(layer.weight)` : 가중치 초기화.
- `nn.CrossEntropyLoss()` + `torch.optim.Adam(model.parameters(), lr=0.001)`.
- 학습 루프: `optimizer.zero_grad()` → `model(X)` → `cost.backward()` → `optimizer.step()`.

```python
self.layer1 = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(), nn.MaxPool2d(2))
out = out.view(out.size(0), -1)   # flatten 후 FC
```

## 자주 하는 실수 / 팁
- flatten 후 `nn.Linear`의 입력 크기(`7*7*64` 등)를 특성 맵 크기와 정확히 맞춰야 한다. 틀리면 shape 에러가 난다.
- 학습 데이터로 학습하고 테스트 데이터로 평가해야 한다(예제는 시연을 위해 일부 단순화되어 있으니, 실제로는 `mnist_train`으로 학습할 것).
- 평가 시 `with torch.no_grad():`로 감싸 메모리/속도를 아끼고, `test_data`/`test_labels`는 최신 torchvision에서 `data`/`targets`로 이름이 바뀌었다.

## 예제 노트북 요약
- `example.ipynb`는 8단계 학습 흐름에 따라 2-레이어 CNN으로 MNIST를 분류해 약 99.69% 정확도를 얻는다.
- 이어서 conv 레이어와 FC를 추가한 3-레이어 모델로 성능이 더 향상되는지(약 99.96%) 비교한다.

## 더 보기
- 선행 개념: [`../01_convolution/concept.md`](../01_convolution/concept.md) — 합성곱/풀링 기초
- 다음 단계: [`../03_vgg/concept.md`](../03_vgg/concept.md) — 더 깊은 CNN 아키텍처
- 분류 기초: [`../../04_neural_network/03_mnist_intro/concept.md`](../../04_neural_network/03_mnist_intro/concept.md)
