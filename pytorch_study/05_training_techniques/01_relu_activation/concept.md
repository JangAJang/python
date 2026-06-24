# ReLU 활성화 함수 (ReLU Activation)

> 테마: 05_training_techniques · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
sigmoid의 기울기 소실(vanishing gradient) 문제를 ReLU 같은 활성화 함수로 완화하고, 다양한 optimizer로 학습 속도/안정성을 개선한다.

## 핵심 개념
- **활성화 함수(activation function)**: 선형 레이어 출력을 비선형으로 변환해 신경망이 비선형 패턴을 학습하게 해주는 함수.
- **sigmoid의 기울기 소실**: sigmoid는 입력이 0에서 멀어질수록 기울기가 0에 가까워진다. back-propagation 시 작은 기울기들이 여러 레이어에 걸쳐 곱해지면 gradient가 거의 0이 되어(vanishing gradient) 앞쪽 레이어가 학습되지 않는다. 레이어가 깊을수록 더 심각하다.
- **ReLU**: `f(x) = max(0, x)`. x > 0이면 기울기가 항상 1이라 양수 구간에서 기울기 소실이 없다. 계산이 단순하고 깊은 망에서 잘 동작한다.
- **dead neuron**: ReLU는 입력이 0 이하이면 기울기가 0이라, 어떤 뉴런이 계속 음수 입력만 받으면 학습되지 못하고 죽을 수 있다.

## 원리 / 수식
주요 활성화 함수 비교:
- **sigmoid**: $\sigma(x) = \dfrac{1}{1+e^{-x}}$ — 출력 (0,1), 기울기 소실 심함.
- **tanh**: $\tanh(x) = \dfrac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$ — 출력 (-1,1), 중심이 0이라 sigmoid보다 학습이 안정적이나 여전히 소실 발생.
- **ReLU**: $f(x) = \max(0, x)$ — 양수 구간 기울기 1, 음수 구간 0(dead neuron 위험).
- **Leaky ReLU**: $f(x) = \max(ax, x)$ ($a$는 작은 양수). 음수 구간에도 작은 기울기를 주어 dead neuron을 완화.

## PyTorch 구현 포인트
- 활성화 함수: `nn.ReLU()`, `nn.Sigmoid()`, `nn.Tanh()`, `nn.LeakyReLU(0.1)` 또는 함수형 `F.relu(x)`.
- optimizer는 `torch.optim`에 다양하게 제공:
  - `optim.SGD` — 가장 기본, lr 튜닝 필요.
  - `optim.Adagrad` / `optim.Adadelta` / `optim.RMSprop` — lr을 적응적으로 조정.
  - `optim.Adam` — 모멘텀 + RMSprop 결합, 가장 널리 쓰임, 튜닝 부담 적음.
```python
model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

## 자주 하는 실수 / 팁
- 깊은 망에서 sigmoid/tanh를 은닉층에 쓰면 학습이 거의 안 될 수 있다 → 은닉층은 보통 ReLU 계열.
- ReLU에서 dead neuron이 많이 생기면 LeakyReLU나 He 초기화로 완화.
- optimizer 선택과 learning rate는 함께 튜닝한다. 잘 모르겠으면 Adam이 무난한 출발점.

## 예제 노트북 요약
- `example.ipynb`는 `make_moons` 데이터로 SimpleNet을 만들고 Sigmoid/Tanh/ReLU/LeakyReLU의 학습 손실을 비교한다. 이어서 SGD/Adagrad/Adadelta/RMSprop/Adam/Adamax/ASGD 등 여러 optimizer의 validation loss를 비교 시각화한다.

## 더 보기
- 가중치 초기화: [`../02_weight_initialization/concept.md`](../02_weight_initialization/concept.md)
- 배치 정규화(기울기 소실 직접 해결): [`../04_batch_normalization/concept.md`](../04_batch_normalization/concept.md)
- 경사하강법 기초: [`../05_tips/concept.md`](../05_tips/concept.md)
