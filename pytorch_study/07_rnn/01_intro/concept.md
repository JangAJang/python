# RNN 소개 (Recurrent Neural Network Intro)

> 테마: 07_rnn · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
순서가 중요한 시계열/시퀀스 데이터를 위해, 이전 단계의 정보를 hidden state로 다음 단계에 전달하며 처리하는 신경망이다.

## 핵심 개념
- RNN은 시계열(순서가 있는) 데이터를 다루기 위해 만들어졌다. 같은 단어라도 문장 내 위치에 따라 의미(벡터)가 달라질 수 있다.
- 입력 데이터의 벡터만 저장하는 것이 아니라, 순서 정보(position)를 함께 다룬다.
- `x0`을 받아 `h0`을 만들고, 그 hidden state를 다음 셀로 전달한다. 따라서 n번째 출력은 n-1번째 입력의 영향을 받는다.
- 예: `HELLO`를 입력할 때, 두 개의 `L` 중 첫 번째 다음에는 `L`, 두 번째 다음에는 `O`가 오는지를 hidden state로 구분한다.
- 셀 `A`는 항상 하나다. 같은 가중치를 시간축으로 재사용(parameter sharing)한다.

## 원리 / 수식
- 셀 A는 이전 hidden state `h_{t-1}`와 이번 입력 `x_t`로 다음 hidden state `h_t`를 만든다.
- 기본 형태: `h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b)`
- 셀 내부 함수의 설계에 따라 vanilla RNN, LSTM, GRU 등으로 나뉜다.
- 셀 A가 복잡할수록 학습 능력(trainability)은 올라가지만 연산 자원이 더 필요하다.

## PyTorch 구현 포인트
- `torch.nn.RNN(input_size, hidden_size, batch_first=True)` 로 기본 RNN 셀을 만든다.
- `outputs, _status = rnn(input_data)` 형태로 (전체 시퀀스 출력, 마지막 hidden state)를 반환받는다.
- task 유형(one-to-many, many-to-one, many-to-many)에 따라 입력/출력 구성을 다르게 설계한다.

## 자주 하는 실수 / 팁
- `batch_first=True`를 쓰면 입력 shape이 `(batch, seq_len, input_size)`가 된다. 설정에 따라 차원 순서가 달라지므로 주의.
- RNN은 셀이 여러 개로 그려지지만 실제로는 하나의 셀(가중치)을 반복 사용한다는 점을 기억하자.

## 예제 노트북 요약
- `example.ipynb`는 RNN의 전체 구조, 셀 A 내부에서 일어나는 일, 그리고 task별 사용 형태(usage in RNN)를 그림과 함께 개념적으로 설명한다.

## 더 보기
- 다음: [`../02_basics/concept.md`](../02_basics/concept.md) — PyTorch에서 RNN 다루기
- 활용: [`../05_time_series/concept.md`](../05_time_series/concept.md) — 시계열 예측
