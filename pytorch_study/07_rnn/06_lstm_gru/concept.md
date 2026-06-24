# LSTM / GRU (게이트 기반 순환 신경망)

> 테마: 07_rnn · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
vanilla RNN의 vanishing gradient로 인한 장기 의존성 학습 실패 문제를, 정보의 흐름을 조절하는 게이트(gate)와 별도의 cell state로 해결한 순환 신경망이 LSTM과 그 경량화 버전 GRU다.

## 핵심 개념
- **vanilla RNN의 한계**: 시점이 멀어질수록 gradient가 곱해지며 0으로 수렴(vanishing)하거나 발산해, 먼 과거의 정보를 hidden state가 거의 기억하지 못한다.
- **LSTM의 cell state `c_t`**: hidden state와 별개로 정보를 "컨베이어 벨트"처럼 거의 그대로 흘려보내는 장기 기억 통로. 게이트가 여기에 정보를 더하거나 지운다.
- **LSTM의 세 게이트**:
  - **forget gate `f_t`**: cell state에서 무엇을 버릴지 결정.
  - **input gate `i_t`**: 새 후보 정보 중 무엇을 cell state에 더할지 결정.
  - **output gate `o_t`**: cell state 중 무엇을 hidden state로 내보낼지 결정.
- **GRU**: cell state 없이 게이트를 두 개로 줄인 경량 버전. **reset gate**(과거를 얼마나 무시할지)와 **update gate**(과거와 새 정보를 얼마나 섞을지)를 사용한다.
- **bidirectional RNN**: 정방향·역방향 두 RNN을 함께 돌려 각 시점이 과거와 미래 문맥을 모두 보게 한다.

## 원리 / 수식
- `f_t = σ(W_f·[h_{t-1}, x_t] + b_f)`
- `i_t = σ(W_i·[h_{t-1}, x_t] + b_i)`, 후보 `g_t = tanh(W_g·[h_{t-1}, x_t] + b_g)`
- `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t` (이전 기억 일부 유지 + 새 정보 추가)
- `o_t = σ(W_o·[h_{t-1}, x_t] + b_o)`, `h_t = o_t ⊙ tanh(c_t)`
- GRU: `z_t = σ(...)`(update), `r_t = σ(...)`(reset), `h_t = (1 - z_t)⊙h_{t-1} + z_t⊙tanh(W·[r_t⊙h_{t-1}, x_t])`
- 핵심 직관: `c_t = f_t ⊙ c_{t-1} + ...`의 덧셈 구조 덕분에 gradient가 곱셈으로만 줄어들지 않아 장기 의존성이 보존된다.

## PyTorch 구현 포인트
```python
lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
out, (h_n, c_n) = lstm(x)        # LSTM은 (h_n, c_n) 튜플을 반환
gru = nn.GRU(input_size, hidden_size, batch_first=True)
out, h_n = gru(x)                # GRU는 h_n 하나만 반환
```
- `nn.LSTM`은 `(output, (h_n, c_n))`을, `nn.GRU`는 `(output, h_n)`을 반환한다.
- `bidirectional=True`면 `output`의 마지막 차원이 `2*hidden_size`가 된다.
- `num_layers`로 셀을 수직으로 쌓을 수 있고, 그 사이에 `dropout`을 줄 수 있다.

## 자주 하는 실수 / 팁
- LSTM의 반환값을 RNN처럼 `out, h = lstm(x)`로 받으면 `(h_n, c_n)` 튜플이 통째로 `h`에 들어가 버린다. 언패킹에 주의.
- bidirectional 사용 시 다음 FC layer의 입력 차원을 `2*hidden_size`로 맞춰야 한다.
- bidirectional은 전체 시퀀스가 미리 주어진 분류/태깅에만 쓸 수 있고, 실시간 생성에는 부적합하다.
- 게이트의 σ(시그모이드)는 0~1 게이트 값을, tanh는 후보/출력 값을 만든다는 역할 구분을 기억하자.

## 더 보기
- 선행 개념: [`../01_intro/concept.md`](../01_intro/concept.md) — RNN과 hidden state
- 활용: [`../05_time_series/concept.md`](../05_time_series/concept.md) — LSTM 시계열 예측
