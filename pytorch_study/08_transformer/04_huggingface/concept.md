# HuggingFace Transformers

> 테마: 08_transformer · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
사전학습된 Transformer 모델과 토크나이저를 통일된 API(`AutoModel`, `AutoTokenizer`)로 불러와 추론·파인튜닝까지 손쉽게 수행하는 라이브러리.

## 핵심 개념
- **AutoTokenizer / AutoModel**: 모델 이름만 주면 알맞은 토크나이저·모델 클래스를 자동 선택해 로드한다.
- **pipeline**: 전처리→추론→후처리를 한 줄로 묶은 고수준 추론 API (분류·생성·QA 등 task 지정).
- **Trainer / TrainingArguments**: 학습 루프·평가·체크포인트·로깅을 표준화한 고수준 학습기.
- **task-specific head**: `...ForSequenceClassification`, `...ForCausalLM` 등 과제별 출력 head가 붙은 모델 클래스.

## 원리 / 수식
- **토크나이저 동작**: 텍스트 → subword 토큰(WordPiece/BPE/SentencePiece) → 정수 `input_ids` + `attention_mask` (+ `token_type_ids`).
  - 배치 처리 시 `padding`/`truncation`/`max_length`로 길이를 맞춘다.
  - subword 단위라 OOV(미등록 단어)를 조각으로 처리해 어휘 폭발을 막는다.
- **fine-tuning 워크플로**: 데이터셋 토큰화 → 사전학습 모델 + task head 로드 → `Trainer`로 학습 → 평가/저장.

## PyTorch 구현 포인트
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    Trainer, TrainingArguments

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

enc = tok(texts, padding=True, truncation=True, return_tensors="pt")

args = TrainingArguments(output_dir="out", per_device_train_batch_size=16,
                         num_train_epochs=3, eval_strategy="epoch")
trainer = Trainer(model=model, args=args,
                  train_dataset=train_ds, eval_dataset=eval_ds)
trainer.train()

# 빠른 추론
from transformers import pipeline
clf = pipeline("sentiment-analysis")
clf("I love this!")
```
- `from_pretrained`는 모델 가중치와 토크나이저 설정을 캐시에 받아 둔다.
- `num_labels`를 지정해야 분류 head가 알맞은 출력 차원으로 초기화된다.

## 자주 하는 실수 / 팁
- 모델과 토크나이저는 **반드시 같은 체크포인트**에서 로드해야 한다(어휘·특수 토큰 불일치 방지).
- `return_tensors="pt"`를 빼면 파이썬 리스트가 나와 모델에 바로 못 넣는다.
- `padding`/`truncation` 없이 길이가 제각각이면 배치 텐서를 만들 수 없다.
- 분류 head는 새로 초기화되므로 파인튜닝 없이 바로 쓰면 무의미한 출력이 나온다.

## 더 보기
- 선행 개념: [`../03_bert_gpt/concept.md`](../03_bert_gpt/concept.md) — BERT/GPT 구조
- 다음 단계: [`../05_finetuning/concept.md`](../05_finetuning/concept.md) — LLM 파인튜닝
- 외부 자료: https://huggingface.co/learn/nlp-course
