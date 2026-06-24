# BERT vs GPT

> 테마: 08_transformer · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
BERT는 양방향 문맥을 보는 encoder-only 모델로 이해(understanding) 과제에, GPT는 단방향 autoregressive decoder-only 모델로 생성(generation) 과제에 강하다.

## 핵심 개념
- **BERT (Bidirectional Encoder Representations from Transformers)**: Transformer encoder만 쌓은 구조. 한 토큰이 좌우 양쪽 문맥을 모두 참조한다(bidirectional).
- **GPT (Generative Pre-trained Transformer)**: Transformer decoder만 쌓은 구조. masked self-attention으로 왼쪽 문맥만 보고 다음 토큰을 예측한다(단방향, autoregressive).
- **사전학습 목표 차이**: BERT는 masking 기반 복원, GPT는 다음 토큰 예측(language modeling).
- **용도**: BERT는 분류·NER·QA 등 인코딩 기반 과제, GPT는 텍스트 생성·대화·few-shot 추론.

## 원리 / 수식
- **BERT 사전학습**:
  - **MLM (Masked Language Modeling)**: 입력 토큰의 약 15%를 `[MASK]`로 가리고 원래 토큰을 맞춘다. 양방향 문맥을 쓸 수 있는 이유.
  - **NSP (Next Sentence Prediction)**: 두 문장이 연속인지 이진 분류. (이후 RoBERTa 등에서는 효과가 작아 제거되기도 함.)
- **GPT 사전학습**: autoregressive LM. `L = Σ log P(x_t | x_<t)` — 이전 토큰들로 다음 토큰 확률을 최대화한다.
- BERT는 양방향이라 생성에 직접 쓰기 어렵고, GPT는 단방향이라 미래 정보를 누설하지 않아 생성에 자연스럽다.

## PyTorch 구현 포인트
```python
from transformers import BertModel, GPT2LMHeadModel, AutoTokenizer
# BERT: 이해 과제 → 보통 [CLS] 표현을 분류기에 연결
bert = BertModel.from_pretrained("bert-base-uncased")
out = bert(**tok("I love NLP", return_tensors="pt"))
cls = out.last_hidden_state[:, 0]          # [CLS] 토큰 표현

# GPT: 생성 과제 → 다음 토큰을 autoregressive하게 디코딩
gpt = GPT2LMHeadModel.from_pretrained("gpt2")
gen = gpt.generate(**tok("Once upon a time", return_tensors="pt"), max_new_tokens=20)
```
- BERT는 `[CLS]`, `[SEP]` 특수 토큰을 쓰고, 분류 시 `[CLS]` 위치 표현을 활용한다.
- GPT 계열은 `generate()`로 디코딩하며 greedy/beam/sampling 등 전략을 고를 수 있다.

## 자주 하는 실수 / 팁
- BERT로 텍스트를 "생성"하려는 시도는 구조상 부적합하다. 생성은 GPT 계열을 쓰자.
- MLM의 `[MASK]`는 학습 때만 등장하고 추론(다운스트림)에는 없어 분포 차이가 생긴다(그래서 일부는 랜덤/원본 토큰으로 대체).
- "decoder-only"라고 cross-attention이 있는 게 아니다. GPT는 self-attention만 쓴다.

## 더 보기
- 선행 개념: [`../02_transformer/concept.md`](../02_transformer/concept.md) — Transformer encoder/decoder 구조
- 논문: BERT (Devlin et al., 2018), GPT (Radford et al., 2018)
