# GPU/디바이스 관리 (Device Management)

> 보강 학습 · C. 실전 운용 · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
연산을 수행할 디바이스(CPU/GPU)를 `torch.device`로 지정하고, 연산에 참여하는 모델과 텐서를 모두 같은 device에 올려야 한다.

## 핵심 개념
- **`torch.device`**: 연산이 일어나는 위치를 나타내는 객체. `"cpu"`, `"cuda"`, `"cuda:0"`(GPU 인덱스 지정) 등으로 생성.
- **`.to(device)`**: 텐서/모델을 지정한 device로 이동. 텐서는 새 텐서를 반환하지만, `nn.Module`은 in-place로 이동한다.
- **device 일치 규칙**: 두 텐서를 연산하려면 같은 device에 있어야 한다. 모델 파라미터와 입력 텐서도 마찬가지.
- **`torch.cuda.is_available()`**: GPU 사용 가능 여부를 bool로 반환. 환경에 따라 CPU로 자동 폴백하는 코드를 작성할 때 사용.

## 동작 방식
- GPU 텐서는 CUDA 메모리에, CPU 텐서는 시스템 RAM에 저장된다. 둘은 메모리 공간이 달라 직접 연산할 수 없다.
- 디바이스 간 이동(`.to()`, `.cpu()`, `.cuda()`)은 메모리 복사를 동반하므로 비용이 있다. 불필요한 왕복은 피한다.
- 모델을 GPU로 옮기면 파라미터·버퍼가 모두 이동하지만, `forward`에 들어오는 입력은 자동 이동되지 않으므로 직접 옮겨야 한다.

## PyTorch 구현 포인트
```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MyModel().to(device)           # 모델 파라미터를 device로 이동 (in-place)
for x, y in loader:
    x, y = x.to(device), y.to(device)  # 입력/정답도 같은 device로
    pred = model(x)
    loss = criterion(pred, y)
```
- 결과를 numpy로 꺼낼 때는 `tensor.cpu().numpy()` (GPU 텐서는 `.numpy()` 직접 호출 불가).
- 새 텐서 생성 시 `torch.zeros(3, device=device)`처럼 device를 바로 지정하면 이동 비용을 아낀다.

## 자주 하는 실수 / 팁
- **"Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu"**: 모델은 GPU인데 입력이 CPU(또는 그 반대)일 때 발생. 에러 메시지의 두 device를 보고 어느 텐서를 옮길지 결정한다.
- 학습 루프에서 매 배치마다 `.to(device)`를 빠뜨리지 않도록 주의. 특히 `criterion`에 들어가는 정답 라벨.
- 새로 만든 가중치/상수 텐서(예: loss 계산용 weight)도 device로 옮겨야 한다.
- `model.to(device)`는 in-place지만 텐서의 `.to(device)`는 반환값을 받아야 적용된다(`x = x.to(device)`).

## 더 보기
- PyTorch 튜토리얼: https://docs.pytorch.org/tutorials/
- CUDA semantics: https://docs.pytorch.org/docs/stable/notes/cuda.html
- 다음: [`../02_save_load/concept.md`](../02_save_load/concept.md) — 저장/로드 시 `map_location`으로 device 다루기
