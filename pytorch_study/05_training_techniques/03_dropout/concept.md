# 드롭아웃 (Dropout)

> 테마: 05_training_techniques · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
학습 중 일부 뉴런을 무작위로 꺼서 특정 뉴런 의존성을 줄이고 과적합(overfitting)을 방지하는 정규화 기법.

## 핵심 개념
- **과적합(overfitting)**: 모델이 train_set에 과도하게 맞춰져 train error는 매우 낮지만 test error는 높아지는 상태.
- **과적합 해결 방법들**: 데이터 늘리기, feature 줄이기, regularization, 그리고 dropout.
- **드롭아웃 동작**: 각 학습 스텝마다 각 레이어의 노드를 확률 $p$로 무작위 제외하고, 남은 노드들로 출력을 구해 loss를 계산·역전파한다. 매 스텝마다 다른 부분망(sub-network)을 학습하는 효과가 있어 앙상블처럼 작동한다.

## 원리 / 수식
- 각 뉴런을 확률 $p$로 0으로 만든다(drop). 결과적으로 특정 뉴런 조합에 과의존하지 않게 되어 일반화 성능이 올라간다.
- **train vs eval 차이가 핵심**:
  - 학습 시(`model.train()`): dropout 활성화, 입력한 비율만큼 노드를 끈다.
  - 평가 시(`model.eval()`): dropout 비활성화, 전체 노드를 사용한다(스케일 보정은 PyTorch가 내부 처리).

## PyTorch 구현 포인트
- 레이어 정의: `torch.nn.Dropout(p=drop_prob)` (p는 끌 비율, 예: 0.3).
- 모델에 활성화 사이에 끼워 넣는다:
```python
model = nn.Sequential(linear1, relu, dropout,
                      linear2, relu, dropout,
                      linear3)
```
- 학습/평가 모드 전환을 반드시 명시:
```python
model.train()   # dropout ON
...
model.eval()    # dropout OFF (평가/추론)
with torch.no_grad():
    pred = model(X_test)
```

## 자주 하는 실수 / 팁
- 평가 시 `model.eval()`을 빼먹으면 dropout이 켜진 채로 추론되어 결과가 불안정해진다.
- p가 너무 크면 학습이 어려워지고, 너무 작으면 정규화 효과가 약하다(은닉층 0.3~0.5가 흔함).
- dropout + 가중치 초기화(Xavier/He)를 함께 쓰면 좋다(예제는 Xavier로 초기화).
- 배치 정규화와 dropout을 함께 쓸 때는 순서/상호작용에 주의.

## 예제 노트북 요약
- `example.ipynb`는 MNIST에서 5층 MLP(512 hidden)에 Xavier 초기화 + Dropout(p=0.3)을 적용해 학습한다. `model.train()`/`model.eval()` 전환을 보여주고 테스트 정확도 약 0.979를 얻는다.

## 더 보기
- 과적합/정규화 종합 팁: [`../05_tips/concept.md`](../05_tips/concept.md)
- 가중치 초기화: [`../02_weight_initialization/concept.md`](../02_weight_initialization/concept.md)
- 배치 정규화: [`../04_batch_normalization/concept.md`](../04_batch_normalization/concept.md)
