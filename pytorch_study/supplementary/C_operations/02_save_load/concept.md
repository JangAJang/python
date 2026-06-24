# 모델 저장/로드 (Save & Load)

> 보강 학습 · C. 실전 운용 · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
가중치 묶음인 `state_dict`만 저장하는 방식이 권장되며, 학습을 이어가려면 optimizer·epoch까지 묶은 checkpoint를 저장한다.

## 핵심 개념
- **`state_dict`**: 레이어별 학습 파라미터(weight/bias)와 버퍼를 담은 파이썬 dict. 모델 구조와 분리된 순수 가중치.
- **state_dict 저장 (권장)**: `torch.save(model.state_dict(), path)`. 모델 클래스 코드만 있으면 로드 가능해 이식성이 높다.
- **전체 모델 저장**: `torch.save(model, path)`. pickle로 클래스 정의·경로에 의존하므로 리팩터링에 취약하다.
- **checkpoint**: model state_dict + optimizer state_dict + epoch + loss 등을 dict로 묶어 저장. 학습 재개에 필요.

## 동작 방식
- 로드는 `load_state_dict()`가 키(파라미터 이름) 기준으로 매칭하므로, 먼저 동일 구조의 모델을 만들고 가중치를 주입한다.
- `torch.save`는 내부적으로 pickle을 사용한다. 전체 모델 저장은 저장 당시의 클래스/디렉터리 구조가 로드 시점에도 유지되어야 한다.

## PyTorch 구현 포인트
```python
# 저장 (권장 방식)
torch.save(model.state_dict(), "model.pth")

# 로드 (추론)
model = MyModel()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()                # dropout/BN을 추론 모드로 전환

# checkpoint (학습 재개)
torch.save({"epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss}, "ckpt.pth")
ckpt = torch.load("ckpt.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
```
- `map_location`: 저장된 device와 로드 환경이 다를 때 매핑(예: GPU에서 저장 → CPU에서 로드 시 `map_location="cpu"`).

## 자주 하는 실수 / 팁
- 추론 전 `model.eval()` 호출을 잊으면 dropout/BatchNorm이 학습 모드로 동작해 결과가 달라진다.
- `model.load_state_dict(torch.load(path))`처럼 **로드한 dict를 다시 load_state_dict에 넣어야** 한다. `model = torch.load(...)`와 혼동 금지.
- 학습 재개 시 optimizer state(모멘텀 등)를 함께 로드하지 않으면 학습이 매끄럽지 않다.
- 확장자(`.pth`/`.pt`)는 관례일 뿐 기능 차이는 없다.

## 더 보기
- PyTorch 튜토리얼: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
- 선행: [`../01_device/concept.md`](../01_device/concept.md) — device와 `map_location`
- 다음: [`../03_reproducibility/concept.md`](../03_reproducibility/concept.md) — 재현성과 모니터링
