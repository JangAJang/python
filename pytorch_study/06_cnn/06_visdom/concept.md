# Visdom 시각화 (Visdom Visualization)

> 테마: 06_cnn · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
별도 서버에 텍스트·이미지·그래프를 실시간으로 전송해 보여주는 시각화 도구로, 학습 루프에서 loss 곡선을 라이브로 그릴 때 유용하다.

## 핵심 개념
- **서버 기반**: Visdom은 서버를 띄워 사용한다. 가상환경에서 `python -m visdom.server`로 실행하고, 기본 주소는 `http://localhost:8097`.
- **클라이언트**: `vis = visdom.Visdom()`으로 연결 후 메서드로 데이터를 전송.
- **주요 기능**:
  - `vis.text('Hello', env='main')` : 텍스트 창.
  - `vis.image(tensor)` : 단일 이미지(`C x H x W`).
  - `vis.images(tensor)` : 여러 이미지(`N x C x H x W`)를 그리드로.
  - `vis.line(Y, X, opts=...)` : 라인 플롯.
- **env(환경)**: 시각화를 논리적으로 묶는 작업 공간. `env='main'` 등으로 구분/관리한다.

## 원리 / 수식
- 각 메서드는 윈도우 식별자(예: `'window_...'`)를 반환한다. 이 식별자를 `win`으로 넘기면 같은 창을 갱신할 수 있다.
- **Line update**: `vis.line(Y, X, win=plt, update='append')` — `append`(이어 붙이기), `replace`(교체), `remove`(제거).
- X를 주지 않으면 0~1 사이로 자동 생성된다.
- **여러 선 그리기**: `Y`를 `(N, 2)` 형태로, `X`도 같은 모양으로 맞춰 `vis.line(Y=torch.rand(10,2), X=num)`.
- `opts=dict(title=..., legend=[...], showlegend=True)` 로 제목·범례 지정.

## PyTorch 구현 포인트
- 학습 루프에서 epoch별 평균 cost를 한 점씩 append 해 loss 곡선을 실시간으로 본다.

```python
import visdom
vis = visdom.Visdom()

def loss_tracker(loss_plot, loss_value, num):
    vis.line(X=num, Y=loss_value, win=loss_plot, update='append')

loss_plt = vis.line(Y=torch.Tensor(1).zero_(),
                    opts=dict(title='loss_tracker', legend=['loss'], showlegend=True))
# 학습 루프 안에서:
loss_tracker(loss_plt, torch.Tensor([avg_cost]), torch.Tensor([epoch]))
```
- 정리: `vis.close(env='main')`로 해당 env의 창을 모두 닫는다.

## 자주 하는 실수 / 팁
- **서버를 먼저 켜야 한다.** `python -m visdom.server`를 실행하지 않으면 `Visdom()` 연결에서 멈추거나 경고가 뜬다.
- `update='append'`로 갱신하려면 최초 `vis.line(...)`이 반환한 `win`을 반드시 넘겨야 한다.
- `vis.line`에 넘기는 `X`, `Y`는 텐서 모양이 서로 맞아야 한다(여러 선이면 둘 다 `(N, k)`).
- 그래프가 안 보이면 브라우저에서 올바른 `env`를 선택했는지 확인.

## 예제 노트북 요약
- `example.ipynb`는 Visdom의 text/image/images/line 기본 사용법을 MNIST·CIFAR10 샘플로 보여준다.
- line plot의 append 갱신, 다중 선, 제목/범례 옵션을 실습한 뒤, 앞서 만든 MNIST CNN 학습 루프에 `loss_tracker`를 연결해 epoch별 loss 곡선을 실시간으로 그린다.

## 더 보기
- 적용 대상 모델: [`../02_mnist_cnn/concept.md`](../02_mnist_cnn/concept.md) — 시각화를 붙일 CNN
- 커스텀 데이터 학습: [`../05_image_folder/concept.md`](../05_image_folder/concept.md)
- 공식 저장소: https://github.com/fossasia/visdom
