# 데이터 적재 (Data Loading: Dataset & DataLoader)

> 테마: 02_regression · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
데이터가 많아 한 번에 메모리에 올릴 수 없을 때, `Dataset`으로 데이터를 추상화하고 `DataLoader`로 미니배치 단위로 잘라 학습(미니배치 경사하강법)한다.

## 핵심 개념
- 데이터가 많을수록 학습 표본이 많아 좋지만, 전체를 한 번에 메모리에 올리기 어렵고 느리다.
- **미니배치 경사하강법(Minibatch Gradient Descent)**: 전체 데이터를 균일한 크기의 묶음(minibatch)으로 나눠, 각 배치마다 cost를 계산하고 경사하강 한다.
  - 한 번 업데이트에 쓰는 데이터가 적어 **빠르다**.
  - 전체를 쓰지 않으니 cost가 매끄럽게 줄지 않고 **지진계처럼 흔들리며** 줄어든다.
- **Dataset**: 데이터 접근 방식을 정의하는 추상 클래스. `__len__`, `__getitem__`을 구현한다.
- **DataLoader**: Dataset을 받아 배치화/셔플/순회를 담당한다.

## 원리 / 수식
- 미니배치 경사하강법은 전체 m개 중 크기 B의 부분집합에 대해서만 기울기를 계산해 갱신한다.
$$W \leftarrow W - \alpha\,\frac{1}{B}\sum_{i \in \text{batch}} \nabla\,\text{cost}_i(W)$$
- 배치를 무작위로 섞으면(shuffle) 매 epoch 다른 순서로 학습해 편향을 줄인다. 셔플하지 않으면 매번 같은 순서 → 같은 학습 패턴이 반복된다.

## PyTorch 구현 포인트
- **Dataset 정의** — 세 메서드 필수:
```python
from torch.utils.data import Dataset

class CustomDataSet(Dataset):
    def __init__(self):
        self.x_data = [[73,80,75], ...]
        self.y_data = [[152], ...]
    def __len__(self):                 # 전체 샘플 수
        return len(self.x_data)
    def __getitem__(self, idx):        # idx번째 샘플 (x, y) 반환
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y
```
- **DataLoader 생성**:
```python
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```
  - `batch_size`: 미니배치 크기.
  - `shuffle=True`: 매 epoch 데이터 순서를 섞는다.
  - `len(dataloader)`: epoch당 미니배치의 개수.
- **학습 루프** — epoch 안에서 배치를 순회:
```python
for epoch in range(nb_epochs+1):
    for batch_idx, sample in enumerate(dataloader):
        x_train, y_train = sample
        prediction = model(x_train)
        cost = F.mse_loss(prediction, y_train)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
```
  - `enumerate(dataloader)`: 미니배치의 인덱스와 데이터를 함께 받아 순회.

## 자주 하는 실수 / 팁
- `__getitem__`은 인덱스 하나에 대한 (x, y)를 반환해야 하며, 텐서로 변환해 주는 게 안전하다.
- 학습 루프가 이중 for문(epoch × batch)이라는 점에 주의. cost가 배치마다 출력되어 흔들리는 게 정상이다.
- shuffle을 끄면 매 epoch 결과가 동일하게 재현되지만, 일반화 성능엔 보통 shuffle=True가 좋다.

## 예제 노트북 요약
- `example.ipynb`는 다변수 회귀와 같은 시험점수 데이터를 `Dataset`/`DataLoader`로 감싼다.
- `batch_size=2, shuffle=True`로 미니배치를 만들어 `nn.Linear(3,1)` 모델을 20 epoch 학습하며, 배치마다 cost가 출력(흔들림)되는 미니배치 경사하강법을 보여준다.

## 더 보기
- DataLoader로 적재하는 모델(다변수 회귀): [`../02_multivariable_regression/concept.md`](../02_multivariable_regression/concept.md)
- 경사하강법 원리: [`../03_gradient_descent/concept.md`](../03_gradient_descent/concept.md)
- 기본 선형회귀: [`../01_linear_regression/concept.md`](../01_linear_regression/concept.md)
