# 재현성 & 모니터링 (Reproducibility & Monitoring)

> 보강 학습 · C. 실전 운용 · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
난수 seed와 cuDNN 설정을 고정해 실험을 재현 가능하게 만들고, TensorBoard·wandb로 학습 과정을 시각화·기록한다.

## 핵심 개념
- **seed 고정**: 가중치 초기화·데이터 셔플·dropout 등에 쓰이는 난수 생성기를 같은 시작값으로 맞춰 실행마다 동일한 결과를 얻는다.
- **여러 RNG**: PyTorch뿐 아니라 NumPy, 파이썬 `random`, (GPU 사용 시) CUDA RNG까지 모두 고정해야 한다.
- **cuDNN 결정성**: `cudnn.deterministic=True` + `cudnn.benchmark=False`로 비결정적 알고리즘 선택을 막는다(속도와 트레이드오프).
- **TensorBoard**: `SummaryWriter`로 scalar/이미지/그래프를 로그 파일에 기록하고 웹 UI로 확인.
- **wandb**: 클라우드 기반 실험 추적 도구. 하이퍼파라미터·메트릭·아티팩트를 자동 기록·비교.

## 동작 방식
- 딥러닝 학습은 난수에 크게 의존한다. seed를 고정하면 같은 코드·데이터·하드웨어에서 동일한 학습 궤적을 재현할 수 있다.
- 완전한 재현성은 동일한 라이브러리 버전·GPU 환경을 전제로 하며, 일부 연산은 결정성 설정에도 미세한 차이가 남을 수 있다.

## PyTorch 구현 포인트
```python
import torch, numpy as np, random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)        # 모든 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/exp1")
writer.add_scalar("Loss/train", loss.item(), global_step)
writer.close()    # 실행 후: tensorboard --logdir runs
```
- wandb 개요: `wandb.init(project="...")` → `wandb.log({"loss": loss})`로 메트릭 기록(설치·로그인 필요).

## 자주 하는 실수 / 팁
- PyTorch만 고정하고 NumPy·`random`을 빠뜨리면 데이터 전처리/증강 단계에서 비결정성이 남는다.
- `cudnn.benchmark=True`는 입력 크기가 고정일 때 속도를 높이지만 재현성을 깬다. 목적에 맞게 선택한다.
- `DataLoader`의 `num_workers>0`는 워커별 seed가 필요할 수 있다(`worker_init_fn` 또는 `generator` 지정).
- `SummaryWriter`를 `close()`하지 않으면 일부 로그가 flush되지 않을 수 있다.

## 더 보기
- PyTorch 튜토리얼: https://docs.pytorch.org/tutorials/
- 재현성 노트: https://docs.pytorch.org/docs/stable/notes/randomness.html
- 다음: [`../04_transfer_learning/concept.md`](../04_transfer_learning/concept.md) — 전이학습
