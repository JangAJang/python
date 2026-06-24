# Autograd 심화 (Automatic Differentiation)

> 보강 학습 · A. 학습의 핵심 메커니즘 · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
PyTorch는 연산을 수행하는 동시에 동적 계산 그래프를 만들고, `backward()`로 연쇄법칙을 자동 적용해 모든 leaf 텐서의 gradient를 구한다.

## 핵심 개념
- **동적 계산 그래프(define-by-run)**: 그래프를 미리 정의하지 않고 forward 연산을 실행하는 순간 그래프가 그려진다. 매 iteration마다 새로 생성되므로 if/for 같은 파이썬 제어문을 그대로 쓸 수 있다.
- **`requires_grad`**: 텐서에 이 플래그가 True이면 해당 텐서를 거치는 연산이 추적된다. 모델 파라미터(`nn.Parameter`)는 기본 True.
- **leaf vs non-leaf**: 사용자가 직접 만든 `requires_grad=True` 텐서가 leaf이며, gradient가 `.grad`에 쌓인다. 연산 결과(non-leaf)는 `grad_fn`을 갖고 그래프 중간 노드가 된다.
- **`detach()`**: 그래프에서 끊어낸 새 텐서를 반환한다(값 공유, 추적 안 함). gradient를 흘리고 싶지 않은 지점에 사용.
- **`torch.no_grad()`**: 블록 내부 연산의 그래프 추적을 끈다. 추론·평가에서 메모리/속도를 아낀다.

## 원리 / 수식
- 합성함수 $y = f(g(x))$의 미분은 연쇄법칙(chain rule):
  $$\frac{\partial y}{\partial x} = \frac{\partial y}{\partial g}\cdot\frac{\partial g}{\partial x}$$
- `loss.backward()`는 그래프를 출력에서 입력 방향으로 거슬러 가며(reverse-mode) 각 노드의 국소 미분을 곱해 누적, 최종적으로 각 leaf의 `.grad`를 채운다.
- 스칼라가 아닌 텐서에 `backward()`를 호출하려면 `gradient=` 인자(vector-Jacobian product의 벡터)가 필요하다.
- reverse-mode는 출력이 적고 입력(파라미터)이 많은 딥러닝 상황에서, 입력별로 다시 forward를 돌리는 것보다 훨씬 효율적이다.

## PyTorch 구현 포인트
```python
w = torch.tensor([1.0], requires_grad=True)
y = (w ** 2 + 3 * w).sum()   # 그래프 생성
y.backward()                 # 연쇄법칙으로 dy/dw 계산
print(w.grad)                # tensor([5.]) = 2*w + 3 at w=1

with torch.no_grad():        # 추론: 추적 끔
    pred = model(x)
emb = feat.detach()          # 그래프 분리
```
- `optimizer.zero_grad()`를 매 스텝 호출해야 한다. PyTorch는 `.grad`를 덮어쓰지 않고 **누적(+=)** 하기 때문이다.

## 자주 하는 실수 / 팁
- `zero_grad()`를 빠뜨리면 이전 스텝의 gradient가 더해져 학습이 망가진다(RNN의 의도적 누적은 예외).
- `backward()`는 그래프를 한 번 쓰고 해제한다. 재호출하려면 `retain_graph=True`.
- leaf 텐서를 in-place로 수정(`w += 1`)하면 그래프 추적이 깨질 수 있다. 갱신은 `no_grad` 블록이나 optimizer에 맡긴다.
- `.grad`는 leaf에만 채워진다. 중간 텐서 gradient가 필요하면 `retain_grad()`.
- 추론 코드를 `torch.no_grad()`로 감싸지 않으면 그래프가 계속 쌓여 메모리가 불필요하게 늘어난다.
- numpy 변환 등 gradient가 필요 없는 후처리 전에는 `detach()`로 그래프를 끊는다.

## 더 보기
- gradient를 실제로 사용하는 경사하강법: [`../03_optimizer/concept.md`](../03_optimizer/concept.md)
- PyTorch Autograd 튜토리얼: https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
- d2l.ai Automatic Differentiation: https://d2l.ai/chapter_preliminaries/autograd.html
