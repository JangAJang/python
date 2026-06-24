# hihello 문제 (Char-RNN: One-hot & Cross Entropy)

> 테마: 07_rnn · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
`hihello` 같은 짧은 문자열에서 '다음 글자'를 예측하도록 RNN을 학습시키며, one-hot 인코딩과 cross entropy loss를 익힌다.

## 핵심 개념
- hihello 문제: `h, i, h, e, l, l, o`를 받았을 때, 같은 `h` 다음에 `i`가 올지 `e`가 올지를 RNN이 hidden state로 구분하도록 학습한다.
- 이는 many-to-many 구조다. 입력 시퀀스의 각 시점마다 출력(다음 글자)이 나온다.
- One-hot encoding: 글자 집합을 유니크하게 뽑고(`['h','i','e','l','o']`), 각 글자를 해당 인덱스만 1인 벡터로 표현한다.

## 원리 / 수식
- 입력 인덱스 예: `x_data = [[0, 1, 0, 2, 3, 3]]` → 글자로는 `h, i, h, e, l, l`.
- 라벨 예: `y_data = [[1, 0, 2, 3, 3, 4]]` → 글자로는 `i, h, e, l, l, o` (한 칸 뒤로 밀린 다음 글자).
- Cross Entropy Loss: 분류 문제의 표준 손실. 모델이 낸 logits와 정답 클래스(글자 인덱스) 사이의 차이를 측정한다.

## PyTorch 구현 포인트
```python
rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), learning_rate)

outputs, _status = rnn(X)
loss = criterion(outputs.view(-1, input_size), Y.view(-1))
```
- `outputs.view(-1, input_size)`: (batch*seq_len, num_classes) 형태로 펼쳐 시점별 분류로 만든다.
- `Y.view(-1)`: 정답도 1차원으로 펼친다.
- 입력은 `FloatTensor`(one-hot), 정답은 `LongTensor`(클래스 인덱스).

## 자주 하는 실수 / 팁
- `CrossEntropyLoss`에는 softmax를 직접 적용하지 않은 raw logits를 넣어야 한다 (내부에서 log-softmax 수행).
- 정답 텐서는 one-hot이 아니라 클래스 인덱스(`LongTensor`)다.
- `view`로 펼칠 때 `input_size`(=클래스 수)와 차원을 맞추지 않으면 shape 에러가 난다.

## 예제 노트북 요약
- `example.ipynb`는 hihello와 `if you want you`(charseq) 두 예제로, one-hot 입력 → RNN → cross entropy로 다음 글자를 학습하는 전체 루프를 보여준다.

## 더 보기
- 이전: [`../02_basics/concept.md`](../02_basics/concept.md) — RNN 기초
- 다음: [`../04_longseq/concept.md`](../04_longseq/concept.md) — 긴 시퀀스 처리
