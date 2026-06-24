# Attention 메커니즘 (Attention)

> 테마: 08_transformer · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
query와 key의 유사도로 가중치를 만들어 value를 가중합하는 연산이며, self-attention과 multi-head로 확장해 시퀀스 내 모든 위치 간 관계를 한 번에 모델링한다.

## 핵심 개념
- **Q / K / V**: 입력을 세 개의 선형 변환으로 사영한 query, key, value 행렬. "무엇을 찾는가(Q)", "무엇을 가졌는가(K)", "실제 내용(V)"으로 직관할 수 있다.
- **scaled dot-product attention**: Q와 K의 내적으로 점수를 구하고 softmax로 정규화한 가중치를 V에 적용한다.
- **self-attention**: Q, K, V를 모두 같은 입력 시퀀스에서 만든다. 각 토큰이 시퀀스 내 다른 모든 토큰을 참조한다.
- **multi-head attention**: attention을 여러 개의 head로 병렬 수행해 서로 다른 표현 부분공간을 학습한 뒤 concat한다.
- **positional encoding**: attention 자체는 순서를 모르므로 위치 정보를 임베딩에 더해 준다.

## 원리 / 수식
- `Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V`
- `√d_k`로 나누는 이유: d_k가 크면 내적 값의 분산이 커져 softmax가 포화(gradient 소실)되므로 스케일을 맞춘다.
- multi-head: `head_i = Attention(QWᵢQ, KWᵢK, VWᵢV)`, `MultiHead = Concat(head_1..head_h)·Wᴼ`.
- sinusoidal positional encoding: `PE(pos, 2i) = sin(pos / 10000^(2i/d))`, `PE(pos, 2i+1) = cos(pos / 10000^(2i/d))`.

## PyTorch 구현 포인트
```python
import torch, math
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)   # (..., L_q, L_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return attn @ v, attn

# 고수준 API
mha = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
out, attn_w = mha(query, key, value)   # self-attention이면 셋 다 동일 입력
```
- `nn.MultiheadAttention`은 `embed_dim`이 `num_heads`로 나누어떨어져야 한다(head별 차원 = embed_dim/num_heads).
- mask는 `-inf`로 채운 뒤 softmax하면 해당 위치 가중치가 0이 된다.
- `batch_first=True`를 주면 입력 shape이 `(batch, seq, dim)`이 된다(기본은 `(seq, batch, dim)`).

## 자주 하는 실수 / 팁
- 스케일링(`/√d_k`)을 빼먹으면 차원이 클 때 학습이 불안정해진다.
- padding mask와 look-ahead mask는 목적이 다르다. 둘을 혼동하지 말 것.
- positional encoding을 더하지 않으면 단어 순서를 바꿔도 출력이 같아진다(permutation invariant).
- multi-head의 head 수를 늘려도 총 파라미터 수는 비슷하다(차원을 쪼개는 것).

## 더 보기
- 선행 개념: [`../../07_rnn/08_seq2seq_attention/concept.md`](../../07_rnn/08_seq2seq_attention/concept.md) — seq2seq + attention
- 다음 단계: [`../02_transformer/concept.md`](../02_transformer/concept.md) — Transformer 전체 구조
- 논문: "Attention Is All You Need" (Vaswani et al., 2017)
