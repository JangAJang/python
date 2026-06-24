# Optimizer 심화 (Optimization Algorithms)

> 보강 학습 · A. 학습의 핵심 메커니즘 · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
gradient를 어떻게 가공해 파라미터를 갱신하느냐에 따라 SGD·Momentum·RMSprop·Adam이 나뉘며, weight decay와 lr_scheduler로 일반화와 수렴을 함께 다듬는다.

## 핵심 개념
- **SGD + Momentum**: 이전 갱신 방향을 관성처럼 누적해 진동을 줄이고 수렴을 가속한다.
- **RMSprop**: gradient 제곱의 이동평균으로 각 파라미터별 학습률을 적응적으로 조절한다.
- **Adam**: Momentum(1차 모멘트) + RMSprop(2차 모멘트)을 결합하고 bias 보정을 더한 적응형 옵티마이저. 기본값으로도 잘 동작한다.
- **Adam vs AdamW (weight decay)**: 표준 Adam의 weight decay는 gradient에 섞여 적응 스케일에 영향을 받는다. **AdamW**는 decay를 gradient와 분리(decoupled)해 갱신과 따로 적용한다.
- **lr_scheduler**: 학습이 진행되며 learning rate를 줄여 미세 수렴을 돕는다.

## 원리 / 수식
- Momentum: $v_t = \mu v_{t-1} + g_t,\quad \theta \leftarrow \theta - \alpha v_t$
- RMSprop: $s_t = \rho s_{t-1} + (1-\rho)g_t^2,\quad \theta \leftarrow \theta - \frac{\alpha}{\sqrt{s_t}+\epsilon} g_t$
- Adam: 1차 $m_t$, 2차 $v_t$ 모멘트를 bias 보정 후 $\theta \leftarrow \theta - \alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$
- weight decay(L2): 손실에 $\frac{\lambda}{2}\|\theta\|^2$를 더하는 것과 같아 갱신마다 $\theta$를 $\lambda$만큼 0쪽으로 당긴다. AdamW는 이 항을 adaptive scaling과 분리한다.
- CosineAnnealing: lr을 코사인 곡선으로 부드럽게 0에 가깝게 감소시켜 후반부 미세 수렴을 돕는다.

## PyTorch 구현 포인트
```python
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)  # decoupled

sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
for epoch in range(50):
    opt.zero_grad(); loss.backward(); opt.step()
    sched.step()        # 보통 epoch 단위로 호출
```
- `StepLR(opt, step_size=30, gamma=0.1)`은 30 epoch마다 lr을 1/10로 줄인다.

## 자주 하는 실수 / 팁
- L2 정규화가 필요하면 손실에 직접 더하지 말고 `weight_decay` 인자를 쓴다. Adam에서 진짜 weight decay를 원하면 AdamW.
- `scheduler.step()`을 `optimizer.step()`보다 먼저 부르면 lr이 한 스텝 어긋난다.
- Adam은 초반 수렴이 빠르지만 일반화는 잘 튜닝한 SGD+Momentum이 더 나을 때도 있다.
- bias·BatchNorm 파라미터에는 보통 weight decay를 적용하지 않는다(param group으로 분리).
- 새 옵티마이저는 `model.parameters()`를 넘겨 만든다. 모델을 GPU로 옮긴 *뒤*에 생성하는 편이 안전하다.
- `momentum`은 SGD 인자, `betas`는 Adam 계열 인자다 — 옵티마이저마다 받는 인자가 다르다.

## 더 보기
- gradient 계산 원리(autograd): [`../01_autograd/concept.md`](../01_autograd/concept.md)
- PyTorch Optimization 튜토리얼: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- d2l.ai Optimization Algorithms: https://d2l.ai/chapter_optimization/
