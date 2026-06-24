# 가중치 초기화 (Weight Initialization)

> 테마: 05_training_techniques · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
가중치를 어떻게 초기화하느냐가 학습 속도와 최종 성능을 좌우한다. Xavier/He 초기화로 레이어 간 분산을 적절히 유지한다.

## 핵심 개념
- **왜 중요한가**: 가중치가 기댓값에서 동떨어진 상태로 학습을 시작하면 학습 시간이 늘고 성능에도 영향을 준다.
- **0 초기화 금지**: 모든 가중치를 0으로 두면 back-propagation 시 모든 gradient가 0이 되어 학습이 진행되지 않는다(대칭성 문제).
- **RBM / DBN (역사적 배경)**: 과거 Hinton 등은 RBM(Restricted Boltzmann Machine)으로 레이어를 하나씩 사전 학습(pre-training)한 뒤 fine-tuning하는 방식(DBN)으로 좋은 초기값을 얻었다. 지금은 Xavier/He 같은 분석적 초기화가 표준이다.
- **Xavier / He**: 입력/출력 차원에 맞춰 분산을 조절하는 초기화법. Xavier는 sigmoid/tanh, He는 ReLU 계열에 적합.

## 원리 / 수식
- **Xavier initialization**: 입력/출력 노드 수를 고려해 분산을 맞춘다.
  - 정규분포: $W \sim N(0, \sigma^2)$
  - 균등분포: $W \sim U(a, b)$
  - sigmoid/tanh처럼 좌우 대칭 활성화에 적합.
- **He initialization**: ReLU가 음수를 0으로 죽이므로 분산을 더 키워 보정한다. ReLU/LeakyReLU에 적합.

## PyTorch 구현 포인트
- Xavier:
```python
torch.nn.init.xavier_uniform_(linear.weight)
torch.nn.init.xavier_normal_(linear.weight)
```
- He (Kaiming):
```python
torch.nn.init.kaiming_uniform_(linear.weight, mode='fan_in', nonlinearity='relu')
torch.nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
```
- 단순 분포 초기화:
```python
torch.nn.init.normal_(tensor, mean=0.0, std=0.05)
torch.nn.init.uniform_(tensor, a=-0.05, b=0.05)
```

## 자주 하는 실수 / 팁
- 활성화 함수와 초기화를 맞춰라: ReLU에는 He, sigmoid/tanh에는 Xavier.
- bias는 보통 0으로 초기화해도 무방(가중치와 달리 대칭성 문제 없음).
- 초기화만으로 기울기 소실을 완전히 막을 수는 없다 → 배치 정규화와 함께 쓰면 효과적.

## 예제 노트북 요약
- `example.ipynb`는 가중치 초기화가 왜 중요한지, RBM/DBN을 통한 사전학습 개념, 그리고 Xavier/He 초기화의 정규분포·균등분포 형태와 PyTorch/TF API를 정리한다.

## 더 보기
- ReLU 활성화: [`../01_relu_activation/concept.md`](../01_relu_activation/concept.md)
- 드롭아웃: [`../03_dropout/concept.md`](../03_dropout/concept.md)
- 배치 정규화: [`../04_batch_normalization/concept.md`](../04_batch_normalization/concept.md)
