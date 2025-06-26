# 셀프 Q&A 모음집

모든 내용은 채찍피티, 구글링을 통해 알아냄

### Q. CNN(Convolutional Neural Network)가 현대 NLP, LLM에 쓰일까? 

transformer가 나온 이후로는 쓰이지 않지만, 이전에는 CNN을 이용한 sentence classification이 사용되었다. 

#### CNN in sentence classification

[논문 링크](https://arxiv.org/abs/1408.5882)

논문의 내용은 다음과 같다. 
- 문장을 2차원 텐서로 간주해 CNN을 적용한다. 
    - 문장을 토크나이징한 후, 각 토큰을 `Word2Vec`나 `GloVe`와 같은 고정된 임베딩 벡터로 변환해, 2차원 행렬로 만든다. 
    - 임베딩 벡터 : 각 단어를 고정된 크기의 벡터로 변환한다. 
- 여기에 1D convolution을 적용한다. 
    - `n-gram` 필터로 max-pooling을 해, 전체 문장을 요약한 벡터를 생성한다. 
    - n-gram 필터 : n개의 연속된 단어 조합을 잡기 위한 필터이며, 필터의 크기는 n * 300이다. 이를 슬라이딩하며 특징을 추출한다. 
    - max-pooling : 필터를 문장 전체에 슬라이딩한 결과로, feature map이 생성된다. 여기에서 가장 강한 특징 하나를 추출하는 방식이다. (max값을 pool한다. 이를 대표값으로 활용할 수 있다)

전체 흐름을 요약하면, 다음과 같다. 
```
문장 → 단어 임베딩 → [단어 수 x 임베딩 차원 행렬]
      ↓
n-gram convolution (ex. 3-gram, 4-gram 필터 여러 개)
      ↓
각 필터에서 sliding → feature map 생성
      ↓
각 feature map에 대해 max pooling → 1값 추출
      ↓
(여러 필터 결과를 concat) → 최종 벡터
      ↓
Dense Layer + Softmax → 분류 결과
```

그렇다면 어떤 문제가 있어 CNN이 아닌 transformer를 사용하게 되었을까? 
- 문장에서 n-gram filter로 인해 고정된 크기 표현에 의존된다. 
- 동일 단어라도 문맥에 따라 의미가 변화하는 것을 인지하지 못한다. 

transformer를 이용하며 이런 단점이 상쇄될 수 있다. 

transformer가 뭐길래 더 나은 성능을 보이는가?

#### Transformer for sentence classification

이를 위해 트랜스포머에 대한 기본적인 구조를 알아야 한다. 
- 토크나이징(BERT tokenizer)
- 토큰 임베딩 + Positional Encoding 추가
- Transformer인코더
- 토큰 출력(CLS) -> 분류기(Feedforwrard + Softmax)

이렇게 생겼다고는 하는데, 모르는 단어가 너무 많다. 하나씩 더 찾아보자. 

- BERT(Bidirectional Encoder Representation from Transformer)
    - 구글이 만든 사전학습된 트랜스포머 모델
    - 문장의 양방향 문맥을 모두 이해할 수 있게 훈련됨
    - 문장 분류, 감성 분석, 개체명 인식 등 다양한 NLP 작업에 사용 가능

- BERT to Tokenizer
    - BERT 모델이 이해할 수 있는 단위로 토크나이징한다. 
    - CLS, SEP가 존재하며, CLS는 문장의 시작, SEP는 문장의 끝을 의미한다. 

- 토큰 임베딩
    - 각 토큰을 숫자벡터(보통 768차원)로 변환한다. 
    - 임베딩은 의미정보를 포함한다. 

- Positional Encoding
    - Transformer 모델은 순서 정보를 모르기 때문에, 위치 정보를 더해준다. 
    - 궁극적으로, `토큰 임베딩 + 위치 인코딩 = 입력 벡터`로 만들어 트랜스포머 모델에 입력한다

- Transformer 인코더
    - 문장 내 모든 단어들이 서로 어떤 영향을 주는지 계산한다. 
    - Self attention : 모든 단어가 다른 단어와 연결되고, 얼마나 중요한지 판단한다. 
    - 인코더를 통해 각 문장의 CLS 토큰을 대표 벡터로 사용하게 한다. 

- Feedforward Layer
    - CLS벡터를 클래스 개수만큼 차원 축소한다. 

- Softmax
    - Linear 결과를 확률값으로 바꿔준다. 