# 셀프 Q&A 모음집

모든 내용은 채찍피티, 구글링을 통해 알아냄

### Q. CNN(Convolutional Neural Network)가 현대 NLP, LLM에 쓰일까? 

transformer가 나온 이후로는 쓰이지 않지만, 이전에는 CNN을 이용한 sentence classification이 사용되었다. 

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