# 텐서 조작 (Tensor Manipulation)

> 테마: 01_basics · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
PyTorch 텐서를 생성하고, 모양(shape)을 자유자재로 바꾸며(view/reshape/squeeze/unsqueeze), 합치고(cat/stack), 연산(broadcasting/matmul/mean/sum/max)하는 기본 조작들을 정리한다.

## 핵심 개념
- **텐서(Tensor)**: 다차원 배열. `dim`(=rank, 차원 수), `shape`/`size`(차원별 원소 개수)로 구조를 표현한다. numpy의 `ndarray`와 거의 동일한 개념이지만 GPU 연산과 자동미분을 지원한다.
- **인덱싱/슬라이싱**: numpy와 동일하게 `t[0]`, `t[2:5]`, `t[:2]`, `t[-1]` 등을 사용한다.
- **브로드캐스팅(Broadcasting)**: 크기가 다른 텐서 간 사칙연산 시, PyTorch가 자동으로 차원을 맞춰 연산한다. 편리하지만 의도치 않은 모양으로도 "에러 없이" 동작해 디버깅이 어려운 휴먼 에러의 원인이 된다.
- **요소곱 vs 행렬곱**: `*`/`mul()`은 요소별 곱(element-wise), `@`/`matmul()`은 행렬곱(matrix multiplication)이다. 둘은 완전히 다른 결과를 낸다.
- **차원 축소 연산**: `mean`, `sum`, `max` 등은 `dim` 인자로 어느 축을 따라 줄일지 지정한다. `max(dim=...)`는 (최댓값, argmax 인덱스) 튜플을 반환한다.
- **모양 변경**: `view`/`reshape`(원소 수 유지하며 재구성), `squeeze`(크기 1인 차원 제거), `unsqueeze`(특정 위치에 크기 1 차원 추가).
- **타입 캐스팅**: `.float()`, `.long()` 등으로 dtype 변환. 정수형(Long)은 평균을 직접 구할 수 없다.
- **합치기**: `cat`(기존 차원 방향으로 이어붙임, 차원 수 유지), `stack`(새 차원을 만들어 쌓음, 차원 수 증가).
- **In-place 연산**: 함수명 뒤 `_`가 붙으면(`mul_`) 원본 텐서를 직접 수정한다.

## 원리 / 수식
- **브로드캐스팅 규칙**: 두 텐서의 shape을 뒤(오른쪽)부터 비교해, 각 축의 크기가 같거나 한쪽이 1이면 1인 쪽을 복제해 맞춘다.
  - `(1,2) + (1,)` → `(1,2)` 로 확장
  - `(1,2) + (2,1)` → `(2,2)` 로 확장 (양쪽 모두 늘어남)
- **행렬곱 차원 규칙**: `(m, k) @ (k, n) = (m, n)`. 안쪽 차원 `k`가 일치해야 한다.
- **view/reshape**: 총 원소 수는 보존된다. `(2,2,3)` → `view(-1, 3)` 이면 `(4,3)`. `-1`은 "나머지로 자동 계산"을 뜻한다 (2*2*3=12, 12/3=4).
- **cat vs stack**: `(2,2)`와 `(2,2)`를 `cat(dim=0)` → `(4,2)`, `cat(dim=1)` → `(2,4)`. 반면 `stack`은 새 축을 추가해 `(2,2,2)`를 만든다.
- **mean(dim)**: `dim=0`은 행 방향으로 줄여 열별 평균, `dim=1`(=`dim=-1`)은 열 방향으로 줄여 행별 평균.

## PyTorch 구현 포인트
- 생성: `torch.FloatTensor([...])`, `torch.LongTensor`, `torch.ByteTensor`, `torch.tensor(...)`
- 구조 확인: `t.dim()`, `t.shape`, `t.size()`
- 연산: `t.mean(dim=...)`, `t.sum(dim=...)`, `t.max(dim=...)` (→ `values`, `indices` 튜플)
- 모양: `t.view([-1, 3])`, `t.reshape(...)`, `t.squeeze()`, `t.unsqueeze(dim)`
- 합치기: `torch.cat([x, y], dim=0)`, `torch.stack([x, y, z], dim=0)`
- 캐스팅: `t.float()`, `t.long()`
- 같은 모양 텐서: `torch.ones_like(x)`, `torch.zeros_like(x)`
```python
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
m1 @ m2      # 행렬곱 -> (2,1): [[5],[11]]
m1 * m2      # 요소곱(브로드캐스팅) -> (2,2)
```

## 자주 하는 실수 / 팁
- `*`(요소곱)와 `@`(행렬곱)를 혼동하기 쉽다. 브로드캐스팅 때문에 `*`가 에러 없이 동작해 버그가 숨는다.
- `cat`은 합치는 축을 제외한 나머지 축 크기가 모두 같아야 한다. 예: `(2,2)`와 `(3,2)`는 `dim=0`으로는 가능하지만 `dim=1`로는 불가능.
- LongTensor에 `mean()`을 호출하면 dtype 추론 실패 에러가 난다. 먼저 `.float()`로 변환할 것.
- `view`/`reshape`는 원소 수가 보존되어야 한다. 맞지 않으면 에러.
- `max(dim=...)`의 반환은 튜플이다. `[0]`은 값, `[1]`은 argmax 인덱스.
- `squeeze()`는 크기 1인 차원을 모두 제거하므로, 의도치 않은 차원이 사라질 수 있다. 특정 차원만 제거하려면 `squeeze(dim)`을 쓴다.
- in-place(`_`) 연산은 메모리를 아끼지만 autograd 그래프를 깨뜨릴 수 있어 주의.

## 예제 노트북 요약
- `example.ipynb`는 numpy/PyTorch로 1D·2D 텐서를 생성하고 dim/shape/size를 확인한 뒤, 브로드캐스팅, 요소곱 vs 행렬곱, mean/sum/max(+argmax), view, squeeze/unsqueeze, 타입 캐스팅, cat/stack, ones_like/zeros_like, in-place 연산을 차례로 실습한다. 의도적으로 cat 차원 불일치 에러도 보여준다.

## 더 보기
- 다음 주제: 선형 회귀 등 기본 학습으로 진행하면 텐서 조작이 어떻게 모델 입력/출력에 쓰이는지 볼 수 있다.
