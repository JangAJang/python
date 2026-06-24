# 시계열 예측 (Time Series with LSTM)

> 테마: 07_rnn · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
주가 같은 시계열 데이터를 윈도우로 잘라 LSTM에 넣고, 마지막에 FC layer로 다음 종가(1차원 값)를 회귀 예측한다.

## 핵심 개념
- 주가 예측: 매 종가는 이전 종가들에 영향을 받는 전형적인 시계열 문제다.
- 8일차 종가를 직접 맞추려면 출력은 `dim=1`인 스칼라여야 한다.
- 하지만 hidden state까지 `dim=1`로 강제하면 모델 표현력이 떨어진다.
- 더 나은 방식: hidden state는 충분히 큰 차원(예: 10)으로 풍부하게 주고받게 하고, 마지막에 Fully Connected layer를 두어 종가(1개 값)로 차원 축소한다.

## 원리 / 수식
- 입력 feature: 시가/고가/저가/거래량/종가 등 5개 → `data_dim = 5`.
- 윈도우: `seq_length = 7` 일치 데이터로 8일차 종가 `[-1]`(마지막 컬럼)을 예측.
- Min-Max scaling: `(data - min) / (max - min + 1e-7)` 로 각 feature를 0~1로 정규화.
- 손실: 회귀이므로 `MSELoss`(평균제곱오차)를 쓴다.

## PyTorch 구현 포인트
```python
class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim, bias=True)
    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x[:, -1])   # 마지막 시점의 hidden만 사용 (many-to-one)
        return x
```
- `torch.nn.LSTM`을 사용 (vanilla RNN보다 장기 의존성에 강함).
- `x[:, -1]`로 시퀀스의 마지막 출력만 뽑아 FC에 넣는다 → many-to-one 구조.
- 손실: `torch.nn.MSELoss`, 옵티마이저: `Adam`.

## 자주 하는 실수 / 팁
- 데이터를 시간순으로 뒤집어야 할 수 있다 (`xy = xy[::-1]`). 원본 csv의 정렬 순서를 확인하자.
- train/test 분리 후 각각 스케일링하거나, 누수(leakage)를 막기 위해 train 기준으로 스케일링한다.
- test_set 구성 시 윈도우가 끊기지 않도록 `train_size - seq_length`부터 시작한다.
- 회귀이므로 cross entropy가 아니라 MSE를 쓴다.

## 예제 노트북 요약
- `example.ipynb`는 `data-02-stock_daily.csv`(이 디렉토리에 포함)를 로드해 7일 윈도우로 데이터셋을 만들고, 1층 LSTM + FC layer로 다음 종가를 예측한 뒤 실제값과 예측값을 그래프로 비교한다.

## 더 보기
- 이전: [`../04_longseq/concept.md`](../04_longseq/concept.md) — 긴 시퀀스 char-RNN
- 처음: [`../01_intro/concept.md`](../01_intro/concept.md) — RNN 개념
