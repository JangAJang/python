# 퍼셉트론 (Perceptron)

> 테마: 04_neural_network · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
입력에 가중치를 곱해 합하고 활성화 함수를 통과시켜 출력을 내는 인공신경망의 가장 기본 단위로, 단층 퍼셉트론은 선형 분리만 가능해 XOR을 풀 수 없다.

## 핵심 개념
- **뉴런(Neuron)**: 동물 신경계의 뉴런 동작을 본떠 만든 모델. 입력 신호들이 들어와 임계값(threshold) 이상이면 신호를 전달한다.
- **퍼셉트론(Perceptron)**: 입력 `x`에 가중치 `w`를 곱해 모두 합하고 편향 `b`를 더한 뒤, 활성화 함수(예: sigmoid)를 거쳐 출력을 만드는 구조.
- 초창기 퍼셉트론은 **선형 분류기(linear classifier)** 로서 AND, OR 문제 해결에 사용되었다.
- **AND, OR**는 하나의 직선으로 두 클래스를 나눌 수 있어 선형 분리가 가능하다.
- **XOR**는 하나의 직선으로 나눌 수 없다(선형 분리 불가). 단층 퍼셉트론의 근본적 한계이며, 이를 풀려면 다층 퍼셉트론(MLP)이 필요하다.

## 원리 / 수식
- 출력: `output = activation( Σ (w_i · x_i) + b )`
- 활성화 함수로 sigmoid `σ(z) = 1 / (1 + e^{-z})` 를 사용하면 0~1 사이 확률 형태의 출력을 얻는다.
- XOR 진리표: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0. 이 4점은 한 개의 직선으로 0과 1을 분리할 수 없다.
- 그래서 단층 퍼셉트론으로 XOR을 학습시키면 손실(BCE)이 `ln2 ≈ 0.6931`에서 더 줄지 않고 멈춘다(예제에서 확인 가능).

## PyTorch 구현 포인트
- `torch.nn.Linear(in_features, out_features, bias=True)` : `Σ(w·x)+b` 계산.
- `torch.nn.Sigmoid()` : 활성화 함수.
- `torch.nn.Sequential(linear, sigmoid)` : 레이어를 순차로 묶음.
- `torch.nn.BCELoss()` : 이진 분류 손실(Binary Cross Entropy).
- `torch.optim.SGD(model.parameters(), lr=1)` : 경사하강법 최적화.
- 학습 루프: `optimizer.zero_grad()` → `hypothesis = model(X)` → `cost.backward()` → `optimizer.step()`.

```python
X = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]])
Y = torch.FloatTensor([[0],[1],[1],[0]])  # XOR
model = torch.nn.Sequential(torch.nn.Linear(2,1), torch.nn.Sigmoid())
# 단층으로는 cost가 0.6931에서 멈춤 → XOR 학습 실패
```

## 자주 하는 실수 / 팁
- 단층 퍼셉트론으로 XOR을 풀려다 손실이 줄지 않는다고 학습률/에폭만 늘리는 것은 무의미하다. 구조적 한계이므로 은닉층을 추가해야 한다.
- `BCELoss`는 입력이 0~1 확률이어야 하므로 마지막에 Sigmoid를 반드시 통과시켜야 한다.
- 결과 재현을 위해 `torch.manual_seed`로 시드를 고정하면 좋다.

## 예제 노트북 요약
- `example.ipynb`는 뉴런/퍼셉트론 개념을 설명하고, `nn.Linear + Sigmoid` 단층 모델로 XOR 데이터를 10000 스텝 학습한다.
- 손실이 `0.6931`에서 멈춰 단층 퍼셉트론이 XOR을 분류하지 못함을 직접 보여준다.

## 더 보기
- 다음 단계: [`../02_multilayer_perceptron/concept.md`](../02_multilayer_perceptron/concept.md) — 은닉층과 역전파로 XOR 해결
- 분류 기초: [`../../03_classification/01_logistic_regression/concept.md`](../../03_classification/01_logistic_regression/concept.md)
