# nn.Module 구조 심화 (Building Models with nn.Module)

> 보강 학습 · A. 학습의 핵심 메커니즘 · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
`nn.Module`은 파라미터·서브모듈을 자동으로 등록·관리하는 모델의 기본 단위이며, `forward`만 정의하면 학습에 필요한 나머지를 PyTorch가 처리한다.

## 핵심 개념
- **`nn.Module`**: 모든 레이어와 모델의 상위 클래스. `__init__`에서 서브모듈을 attribute로 할당하면 자동 등록되고, `forward`에서 연산 흐름을 정의한다.
- **`nn.Sequential`**: 레이어를 순서대로 연결한 컨테이너. forward가 자동으로 순차 적용된다.
- **`nn.ModuleList`**: 리스트처럼 인덱싱/반복하는 모듈 모음. forward는 직접 작성해야 한다(가변 깊이·반복 구조에 유용).
- **`nn.ModuleDict`**: 이름(key)으로 서브모듈을 고르는 딕셔너리형 컨테이너(분기 구조에 유용).
- **parameter vs buffer**: `parameter`는 학습되는 텐서(gradient 대상), `buffer`는 학습되지 않지만 상태로 저장해야 하는 텐서(예: BatchNorm의 running_mean).

## 원리 / 수식
- 파이썬 리스트(`self.layers = [linear1, ...]`)에 담으면 등록이 안 되어 `parameters()`에 안 잡히고 `to(device)`·저장에서 누락된다. 그래서 `ModuleList`/`ModuleDict`가 필요하다.
- `state_dict()`에는 parameter와 buffer가 모두 포함되어 체크포인트로 저장/복원된다.
- `model.to(device)`·`model.eval()`은 등록된 모든 서브모듈에 재귀적으로 전파된다(등록되지 않은 텐서는 누락).

## PyTorch 구현 포인트
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
        self.register_buffer("running_mean", torch.zeros(10))  # buffer
    def forward(self, x):
        for layer in self.blocks:    # ModuleList는 forward 직접 작성
            x = torch.relu(layer(x))
        return x

for name, p in Net().named_parameters():
    print(name, p.shape)             # 등록된 학습 파라미터 순회
```
- 서브모듈 정의 전 `super().__init__()`를 반드시 먼저 호출한다.

## 자주 하는 실수 / 팁
- 모델 호출은 `model(x)`로 한다. `model.forward(x)`를 직접 부르면 hook 등이 생략된다.
- 레이어를 파이썬 list에 담으면 학습이 안 된다 → 반드시 `ModuleList`/`ModuleDict`.
- 학습되면 안 되지만 device 이동·저장이 필요한 상태는 `register_buffer`로 둔다.
- `parameters()`는 텐서만, `named_parameters()`는 (이름, 텐서) 쌍을 준다. 선택적 freeze 시 이름으로 거른다.
- 특정 레이어를 freeze하려면 `for p in layer.parameters(): p.requires_grad = False`로 gradient를 끈다.
- `nn.Sequential`은 forward 자동, `nn.ModuleList`는 forward 수동 — 분기·반복이 있으면 후자를 쓴다.

## 더 보기
- 파라미터를 갱신하는 옵티마이저: [`../03_optimizer/concept.md`](../03_optimizer/concept.md)
- PyTorch nn 빌딩 블록: https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
- d2l.ai Layers and Modules: https://d2l.ai/chapter_builders-guide/model-construction.html
