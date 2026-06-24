# PyTorch에서 RNN 기초 (RNN Basics in PyTorch)

> 테마: 07_rnn · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
`torch.nn.RNN`을 생성하고 입력 시퀀스를 넣으면, 전체 출력과 마지막 hidden state를 반환받는 가장 기본적인 사용법을 익힌다.

## 핵심 개념
- PyTorch에서 RNN은 `torch.nn.RNN(input_size, hidden_size)` 한 줄로 정의한다.
- `input_size`: 한 시점 입력 벡터의 차원 (예: one-hot이면 vocab 크기).
- `hidden_size`: hidden state(그리고 출력)의 차원.
- 호출하면 `outputs, _status = rnn(input_data)` 형태로 두 값을 받는다.
  - `outputs`: 모든 시점의 출력 시퀀스.
  - `_status`: 마지막 hidden state.

## 원리 / 수식
- 내부적으로 각 시점에서 `h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b)` 를 계산한다.
- 입력 텐서 shape은 `batch_first` 설정에 따라 결정된다.
  - 기본값(`batch_first=False`): `(seq_len, batch, input_size)`
  - `batch_first=True`: `(batch, seq_len, input_size)`

## PyTorch 구현 포인트
```python
import torch
rnn = torch.nn.RNN(input_size, hidden_size)
outputs, _status = rnn(input_data)
```
- 실제로 동작시키려면 `input_size`, `hidden_size`에 구체적인 정수를 넣고, `input_data`를 올바른 shape의 `FloatTensor`로 만들어야 한다.

## 자주 하는 실수 / 팁
- `input_size`, `hidden_size`를 정의하지 않고 변수명만 쓰면 `NameError`가 난다 (예제 노트북의 셀처럼).
- 입력 텐서의 차원 순서를 `batch_first` 설정과 일치시켜야 한다.
- one-hot 입력은 반드시 `FloatTensor`여야 한다 (정수 텐서 아님).

## 예제 노트북 요약
- `example.ipynb`는 `torch.nn.RNN`의 가장 기본적인 호출 형태(생성 → 입력 → outputs/hidden 반환)를 한눈에 보여준다.

## 더 보기
- 이전: [`../01_intro/concept.md`](../01_intro/concept.md) — RNN 개념
- 다음: [`../03_hihello/concept.md`](../03_hihello/concept.md) — hihello 문제로 실제 학습
