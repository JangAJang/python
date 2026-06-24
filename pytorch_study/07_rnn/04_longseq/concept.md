# 긴 시퀀스 (Long Sequence Char-RNN)

> 테마: 07_rnn · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
긴 문장을 슬라이딩 윈도우로 잘라 여러 시퀀스 샘플로 만들고, 여러 층의 RNN + FC layer로 다음 글자를 예측한다.

## 핵심 개념
- 매우 긴 문자열은 한 번에 처리하기 어렵다. 고정 길이 윈도우(예: 10글자)를 한 글자씩 이동시키며 데이터셋을 만든다.
- 각 샘플: 입력 = `sentence[i:i+seq_length]`, 라벨 = `sentence[i+1:i+seq_length+1]` (한 칸 뒤로 밀린 글자들).
- 이렇게 하면 하나의 긴 문장에서 다수의 (입력, 라벨) 쌍이 생성된다 → batch 학습 가능.
- 모델 표현력을 높이기 위해 RNN을 여러 층으로 쌓고(`num_layers`), 마지막에 Fully Connected layer를 붙인다.

## 원리 / 수식
- 슬라이딩 윈도우: `for i in range(0, len(sentence) - seq_length)` 로 윈도우를 이동.
- one-hot: `x_one_hot = [np.eye(dic_size)[x] for x in x_data]`.
- 손실: 모든 시점·모든 샘플에 대해 cross entropy를 평균.

## PyTorch 구현 포인트
```python
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x
```
- `num_layers`로 RNN을 깊게 쌓는다(stacked RNN).
- RNN 출력을 FC layer에 통과시켜 최종 분류 logits를 만든다.
- 예측 문자열 복원 시, 첫 샘플은 전체를, 이후 샘플은 마지막 글자만 이어 붙인다.

## 자주 하는 실수 / 팁
- 윈도우 라벨은 입력보다 한 칸 뒤로 밀린 시퀀스라는 점을 헷갈리지 말 것.
- `outputs.view(-1, dic_size)` 와 `Y.view(-1)` 로 펼쳐 cross entropy에 넣는다.
- 층을 많이 쌓으면 표현력은 오르지만 학습이 느려지고 vanishing gradient가 생길 수 있다 → LSTM/GRU 고려.

## 예제 노트북 요약
- `example.ipynb`는 "if you want to build a ship..." 긴 문장을 길이 10 윈도우로 잘라 데이터셋을 만들고, 2층 RNN + FC로 다음 글자를 예측하는 char-RNN을 학습한다.

## 더 보기
- 이전: [`../03_hihello/concept.md`](../03_hihello/concept.md) — 짧은 시퀀스 예측
- 다음: [`../05_time_series/concept.md`](../05_time_series/concept.md) — 시계열(주가) 예측
