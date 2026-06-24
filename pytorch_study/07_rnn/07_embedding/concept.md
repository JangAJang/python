# nn.Embedding (단어 임베딩)

> 테마: 07_rnn · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
단어를 고차원·희소한 one-hot 벡터 대신, 학습 가능한 lookup table로부터 꺼낸 저차원 dense 벡터로 표현해 RNN/LSTM 입력으로 쓰는 방법이다.

## 핵심 개념
- **one-hot의 한계**: 어휘 크기 `V`가 커지면 벡터 차원이 `V`로 폭발하고(고차원), 한 칸만 1인 희소(sparse) 표현이라 단어 간 의미적 유사성을 전혀 담지 못한다.
- **dense embedding**: 각 단어를 `embedding_dim` 차원의 실수 벡터로 표현. 비슷한 의미의 단어가 가까운 벡터가 되도록 학습된다.
- **lookup table**: `nn.Embedding`은 `(num_embeddings, embedding_dim)` 크기의 가중치 행렬이며, 입력은 단어 인덱스(LongTensor). 해당 인덱스의 행을 그대로 꺼내온다(미분 가능한 인덱싱).
- **padding_idx**: 패딩 토큰의 인덱스를 지정하면 그 행은 항상 0 벡터로 유지되고 학습에서 gradient를 받지 않는다.
- **사전학습 임베딩**: word2vec, GloVe 같은 대규모 코퍼스로 미리 학습된 벡터를 lookup table에 로드해 초기값으로 쓰면 적은 데이터에서도 성능이 좋아진다.

## 원리 / 수식
- 수학적으로 `embed(i) = W[i]` 로, one-hot 벡터 `e_i`와 `W`의 곱 `e_iᵀ W`와 동일하지만 실제로는 곱셈 없이 인덱싱으로 처리한다.
- `W`는 일반 파라미터처럼 backprop으로 갱신되어, 태스크에 맞는 의미 공간이 형성된다.
- one-hot은 모든 단어 거리가 같지만(`cos=0`), 학습된 임베딩은 `cos similarity`로 의미적 거리를 표현한다.

## PyTorch 구현 포인트
```python
embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=128, padding_idx=0)
x = torch.LongTensor([[1, 2, 4, 0]])   # 단어 인덱스 (0=pad)
out = embed(x)                          # shape: (1, 4, 128)

# 사전학습 벡터 로드
embed = nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=0)
```
- 입력은 반드시 정수 인덱스(`LongTensor`)이며 one-hot이 아니다.
- 출력 shape은 `(*input_shape, embedding_dim)` 으로, 그대로 `nn.LSTM` 등의 입력으로 연결된다.
- `from_pretrained(weights, freeze=True)`로 GloVe/word2vec를 로드하며, `freeze=False`면 fine-tuning된다.

## 자주 하는 실수 / 팁
- 인덱스가 `[0, num_embeddings)` 범위를 벗어나면 런타임 에러가 난다. `<unk>` 토큰을 두어 미등록 단어를 처리하자.
- `padding_idx`를 지정하지 않으면 패딩 토큰도 학습되어 시퀀스 길이에 따라 결과가 미묘하게 달라질 수 있다.
- 사전학습 임베딩의 차원과 `embedding_dim`이 일치해야 로드된다.
- one-hot을 직접 만들어 `nn.Linear`에 넣는 방식과 수학적으로 같지만, `nn.Embedding`이 메모리·속도 면에서 훨씬 효율적이다.

## 더 보기
- 선행 개념: [`../03_hihello/concept.md`](../03_hihello/concept.md) — one-hot과 char-RNN
