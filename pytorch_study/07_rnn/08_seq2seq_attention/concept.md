# Seq2Seq & Attention (인코더-디코더와 어텐션)

> 테마: 07_rnn · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
가변 길이 입력 시퀀스를 인코더가 압축하고 디코더가 다른 가변 길이 출력으로 풀어내는 encoder-decoder 구조에서, 고정 context vector의 병목을 attention으로 해소해 번역·요약 성능을 끌어올린 모델이다.

## 핵심 개념
- **encoder-decoder**: 인코더 RNN/LSTM이 입력을 읽어 마지막 hidden state(= context vector)를 만들고, 디코더 RNN이 이를 초기 상태로 받아 출력 시퀀스를 한 토큰씩 생성한다. 입력·출력 길이가 달라도 된다.
- **context vector 병목**: 입력 전체를 단일 고정 길이 벡터로 압축하면, 문장이 길어질수록 앞쪽 정보가 소실되어 성능이 급격히 떨어진다.
- **attention**: 디코더가 매 출력 시점마다 인코더의 모든 hidden state를 다시 보고, 관련 있는 부분에 더 높은 가중치를 주어 동적으로 context를 만든다.
- **Bahdanau(additive)**: 작은 신경망으로 score를 계산. 디코더의 이전 상태를 쓴다.
- **Luong(multiplicative)**: dot/general 형태의 곱으로 score를 계산해 더 단순·빠르다.
- **teacher forcing**: 학습 시 디코더 입력으로 모델의 예측 대신 실제 정답 토큰을 넣어 수렴을 빠르게 한다.

## 원리 / 수식
- score: `e_{t,i} = score(s_{t-1}, h_i)` (디코더 상태 `s_{t-1}`와 인코더 상태 `h_i`의 정합도)
  - dot: `s_{t-1}ᵀ h_i`, general: `s_{t-1}ᵀ W h_i`, additive: `vᵀ tanh(W[s_{t-1}; h_i])`
- 가중치: `α_{t,i} = softmax_i(e_{t,i})`
- context: `c_t = Σ_i α_{t,i} h_i` → 디코더가 `s_{t-1}, y_{t-1}, c_t`로 다음 토큰을 예측.
- 직관: attention은 입력의 어느 부분을 "지금 번역 중인지"를 정렬(alignment)하는 soft 검색이다.

## PyTorch 구현 포인트
```python
# score = dot-product attention 예
scores = torch.bmm(dec_hidden, enc_outputs.transpose(1, 2))  # (B, 1, T)
attn = torch.softmax(scores, dim=-1)
context = torch.bmm(attn, enc_outputs)                       # (B, 1, H)
```
- 인코더는 보통 `nn.LSTM(..., bidirectional=True)`로, 디코더는 단방향으로 구성한다.
- 학습 시 `teacher_forcing_ratio`로 정답 주입 여부를 확률적으로 섞는 것이 일반적이다.
- 생성(추론) 시에는 직전 예측을 다음 입력으로 넣는 auto-regressive 루프를 돈다.

## 자주 하는 실수 / 팁
- teacher forcing만으로 학습하면 추론 시 자기 예측을 입력받는 상황(exposure bias)에 약해진다. 비율을 조절하자.
- 패딩 위치의 score는 `-inf`로 마스킹한 뒤 softmax해야 패딩에 attention이 새지 않는다.
- attention은 RNN의 순차 처리를 없애지는 않는다. 이를 self-attention만으로 대체한 것이 Transformer다.

## 더 보기
- 다음 개념: [`../../08_transformer/01_attention/concept.md`](../../08_transformer/01_attention/concept.md) — self-attention과 Transformer
