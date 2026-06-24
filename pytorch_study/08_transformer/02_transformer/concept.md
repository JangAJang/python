# Transformer 구조 (Transformer)

> 테마: 08_transformer · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
attention만으로 encoder-decoder를 구성한 아키텍처로, 각 블록은 multi-head attention + position-wise FFN을 residual 연결과 LayerNorm으로 감싸 쌓아 만든다.

## 핵심 개념
- **encoder 블록**: (self-attention) → (position-wise FFN), 각 sublayer는 residual + LayerNorm으로 감싼다.
- **decoder 블록**: (masked self-attention) → (cross-attention) → (FFN), 역시 각 sublayer마다 residual + LayerNorm.
- **masked self-attention**: 디코더는 autoregressive하므로 미래 토큰을 보지 못하게 look-ahead mask로 가린다.
- **cross-attention**: 디코더의 query가 encoder 출력을 key/value로 참조해 입력 시퀀스 정보를 가져온다.
- **position-wise FFN**: 각 위치에 독립적으로 적용되는 2-layer MLP (`Linear → ReLU → Linear`).
- **residual + LayerNorm**: `LayerNorm(x + Sublayer(x))` 형태로 깊은 스택의 학습을 안정화한다.

## 원리 / 수식
- sublayer 출력: `out = LayerNorm(x + Sublayer(x))` (post-norm). 최근 구현은 `x + Sublayer(LayerNorm(x))`(pre-norm)도 많이 쓴다.
- FFN: `FFN(x) = max(0, xW₁ + b₁)W₂ + b₂`, 보통 내부 차원 `d_ff = 4 · d_model`.
- RNN과 달리 시퀀스를 순차 처리하지 않아 병렬화가 가능하고, 임의 두 위치 간 경로 길이가 O(1)이다.
- 전체: 입력 임베딩 + positional encoding → N개 encoder 블록 → N개 decoder 블록 → 출력 projection + softmax.

## PyTorch 구현 포인트
```python
import torch.nn as nn
# 고수준: 전체 Transformer
model = nn.Transformer(d_model=512, nhead=8,
                       num_encoder_layers=6, num_decoder_layers=6,
                       dim_feedforward=2048, batch_first=True)

# 인코더만 쌓기
layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
encoder = nn.TransformerEncoder(layer, num_layers=6)

tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)  # look-ahead mask
```
- `nn.Transformer`는 `src_mask`, `tgt_mask`(look-ahead), `*_key_padding_mask`(padding)를 구분해 받는다.
- `generate_square_subsequent_mask`로 디코더용 상삼각 mask를 만든다.
- `d_model`은 `nhead`로 나누어떨어져야 한다.

## 자주 하는 실수 / 팁
- look-ahead mask(미래 가리기)와 key_padding_mask(패딩 가리기)를 동시에 올바르게 넣어야 한다.
- post-norm 구조는 깊어질수록 학습이 까다로울 수 있어 warmup 스케줄이 중요하다.
- cross-attention의 key/value는 encoder 출력, query는 decoder hidden임을 헷갈리지 말 것.
- positional encoding 없이는 순서 정보가 사라진다(attention 파일 참고).

## 더 보기
- 선행 개념: [`../01_attention/concept.md`](../01_attention/concept.md) — attention 메커니즘
- 다음 단계: [`../03_bert_gpt/concept.md`](../03_bert_gpt/concept.md) — BERT vs GPT
- 논문: "Attention Is All You Need" (Vaswani et al., 2017)
