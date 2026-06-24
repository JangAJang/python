# RAG 시스템 (Retrieval-Augmented Generation)

> 테마: 08_transformer · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
질문과 관련된 외부 문서를 벡터 검색으로 찾아 프롬프트에 덧붙인 뒤 LLM이 답하게 하여, 환각(hallucination)을 줄이고 최신 지식을 반영하는 방식.

## 핵심 개념
- **retrieval-augmented generation**: 모델 파라미터에만 의존하지 않고, 검색한 근거 문서를 함께 넣어 생성한다.
- **chunking**: 긴 문서를 임베딩·검색 단위로 잘게 쪼갠다(보통 일정 토큰 길이 + overlap).
- **embedding 모델**: 텍스트를 의미 벡터로 변환해 유사도 비교가 가능하게 한다.
- **vector DB**: 임베딩 벡터를 저장하고 근접 이웃(ANN) 검색을 빠르게 수행하는 저장소(FAISS, Chroma 등).
- **3단계 파이프라인**: retrieve(검색) → augment(프롬프트에 근거 삽입) → generate(LLM 생성).

## 원리 / 수식
- 인덱싱(오프라인): 문서 → chunk → embedding → vector DB에 저장.
- 질의(온라인): query를 같은 embedding 모델로 벡터화 → cosine/내적 유사도 top-k chunk 검색.
- `cos_sim(q, d) = (q·d) / (‖q‖‖d‖)` 로 의미적 근접도를 측정한다.
- 프롬프트 구성: `"다음 문맥을 참고해 답하라:\n{retrieved_chunks}\n질문: {query}"`.
- 환각 완화: 모델이 기억이 아니라 제시된 근거를 바탕으로 답하게 하고, 출처를 인용할 수 있다. 최신성: DB만 갱신하면 재학습 없이 새 지식 반영.

## PyTorch 구현 포인트
```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")
emb = embedder.encode(chunks, normalize_embeddings=True)   # (N, dim)

index = faiss.IndexFlatIP(emb.shape[1])   # 내적 = 정규화 시 cosine
index.add(np.asarray(emb, dtype="float32"))

q = embedder.encode([query], normalize_embeddings=True)
scores, idx = index.search(np.asarray(q, dtype="float32"), k=3)
context = "\n".join(chunks[i] for i in idx[0])
prompt = f"다음 문맥을 참고해 답하라:\n{context}\n\n질문: {query}"
```
- 임베딩을 `normalize_embeddings=True`로 정규화하면 내적(`IndexFlatIP`)이 cosine 유사도가 된다.
- query와 문서는 **같은 embedding 모델**로 인코딩해야 의미 공간이 일치한다.

## 자주 하는 실수 / 팁
- chunk가 너무 크면 검색 정밀도↓, 너무 작으면 문맥 단절↑. overlap을 주어 경계 손실을 줄인다.
- 검색된 근거가 부실하면 LLM이 그래도 환각을 낼 수 있으니, 근거가 없으면 "모른다"고 답하도록 프롬프트로 유도한다.
- LLM의 context window를 넘지 않도록 top-k와 chunk 길이를 함께 조절한다.
- 임베딩 모델과 생성 모델은 별개다. 검색 품질은 embedding 모델이 좌우한다.

## 더 보기
- 선행 개념: [`../04_huggingface/concept.md`](../04_huggingface/concept.md) — 모델·토크나이저 로딩
- 관련: [`../05_finetuning/concept.md`](../05_finetuning/concept.md) — 파인튜닝과의 비교(지식 주입 vs 검색)
