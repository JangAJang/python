# 자연어 처리 기초 (NLP Basics)

> 테마: 07_rnn · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
원시 텍스트를 토큰화·어휘 사전화하여 인덱스 시퀀스로 바꾸고, 임베딩 + RNN/LSTM 분류기로 텍스트 분류·감성 분석을 수행하는 표준 NLP 파이프라인을 정리한다.

## 핵심 개념
- **tokenization**: 문장을 단어/서브워드/문자 단위 토큰으로 쪼갠다. 공백 분리, 형태소 분석, BPE 등 방식이 있다.
- **vocabulary 구축**: 학습 코퍼스의 토큰을 빈도순으로 모아 `token → index` 사전을 만든다. 특수 토큰 `<pad>`, `<unk>`, (필요 시) `<sos>`, `<eos>`를 둔다.
- **수치화(numericalization)**: 각 문장을 사전에 따라 정수 인덱스 리스트로 변환한다. 미등록 단어는 `<unk>`로.
- **padding**: 배치 내 문장 길이가 다르므로 `<pad>`로 길이를 맞춘다.
- **pack_padded_sequence**: 패딩을 RNN이 실제 계산하지 않도록 묶어주는 도구. 연산을 줄이고, 마지막 유효 시점의 hidden을 정확히 얻게 한다.
- **텍스트 분류 / 감성 분석**: 문장을 입력받아 긍정/부정 등 라벨을 예측하는 many-to-one 분류 문제.

## 원리 / 수식
- 파이프라인: `text → tokens → indices → pad → Embedding → LSTM → 마지막 hidden → Linear → softmax`.
- 감성 분석은 보통 클래스가 2~수개인 분류이므로 손실은 `CrossEntropyLoss`(또는 이진은 `BCEWithLogitsLoss`).
- `pack_padded_sequence`는 길이 내림차순 정렬(또는 `enforce_sorted=False`)된 시퀀스를 받아, 패딩 시점의 무의미한 hidden 갱신을 건너뛴다.

## PyTorch 구현 포인트
```python
emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
lstm = nn.LSTM(embed_dim, hidden, batch_first=True)
fc = nn.Linear(hidden, num_classes)

x = emb(indices)                                   # (B, T, E)
packed = nn.utils.rnn.pack_padded_sequence(
    x, lengths, batch_first=True, enforce_sorted=False)
_, (h_n, _) = lstm(packed)
logits = fc(h_n[-1])                               # 마지막 layer의 hidden
```
- `pack_padded_sequence`의 `lengths`는 패딩 전 실제 길이여야 한다.
- 출력을 다시 펼치려면 `pad_packed_sequence`를 쓴다.
- `collate_fn`을 정의해 `DataLoader`에서 배치별 padding을 처리하는 것이 일반적이다.

## 자주 하는 실수 / 팁
- vocabulary는 반드시 학습 데이터로만 만들어야 한다. 검증/테스트 단어로 사전을 만들면 데이터 누수다.
- padding을 무시하지 않으면(`pack` 미사용, `padding_idx` 미설정) 패딩 토큰이 결과에 영향을 준다.
- 분류에서 hidden을 뽑을 때 패딩이 아닌 마지막 유효 시점을 써야 한다. `pack`을 쓰면 `h_n`이 이를 보장한다.
- `enforce_sorted=False`를 쓰면 길이 정렬을 직접 하지 않아도 된다(내부 정렬).

## 더 보기
- 선행 개념: [`../07_embedding/concept.md`](../07_embedding/concept.md) — nn.Embedding과 dense 표현
