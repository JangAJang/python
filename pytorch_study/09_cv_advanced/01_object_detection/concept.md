# 객체 탐지 (Object Detection)

> 테마: 09_cv_advanced · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
이미지 속 객체의 위치(bbox)와 클래스를 동시에 예측하는 과제로, two-stage(정확도)와 one-stage(속도) 계열로 나뉜다.

## 핵심 개념
- **bbox (bounding box)**: 객체를 둘러싸는 사각형. 보통 `(x_min, y_min, x_max, y_max)` 또는 `(cx, cy, w, h)`로 표현.
- **IoU (Intersection over Union)**: 두 box의 교집합 면적 / 합집합 면적. 예측-정답 일치도를 0~1로 측정.
- **NMS (Non-Maximum Suppression)**: 같은 객체에 겹쳐 나온 box들 중 score가 가장 높은 것만 남기고, IoU가 임계값을 넘는 나머지를 제거.
- **anchor**: 미리 정의한 다양한 크기/비율의 기준 box. 모델은 anchor로부터의 offset을 회귀한다.
- **two-stage (Faster R-CNN)**: RPN(Region Proposal Network)이 후보 영역을 뽑고, RoI(Region of Interest) head가 분류+box 보정. 정확하지만 느림.
- **one-stage (YOLO / SSD)**: 영역 제안 없이 grid/anchor에서 한 번에 분류+box 예측. 빠르지만 작은 객체에 약할 수 있음.

## 원리 / 수식
- IoU = `area(A ∩ B) / area(A ∪ B)`. 학습 시 IoU로 anchor를 positive/negative로 라벨링.
- 손실 = 분류 손실(cross entropy/focal loss) + 위치 회귀 손실(smooth L1 등)의 합.
- **mAP (mean Average Precision)**: 각 클래스의 AP(precision-recall 곡선 아래 면적)를 평균. IoU 임계값별(예: mAP@0.5, mAP@[.5:.95])로 측정.

## PyTorch 구현 포인트
```python
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()
preds = model([img_tensor])           # img: [C,H,W], 0~1 float
# preds[0] -> {"boxes", "labels", "scores"}
boxes = torchvision.ops.nms(preds[0]["boxes"], preds[0]["scores"], iou_threshold=0.5)
```
- 입력은 `[0,1]` 범위 float 텐서 리스트. 학습 시에는 `targets`(boxes, labels)도 함께 전달.
- `torchvision.ops`에 `nms`, `box_iou` 등 유틸이 제공된다.

## 자주 하는 실수 / 팁
- box 좌표 포맷(xyxy vs xywh) 혼동 — torchvision은 기본 `xyxy`.
- score threshold와 NMS의 iou_threshold를 혼동하지 말 것(전자는 신뢰도 컷, 후자는 중복 제거 기준).
- mAP는 단순 정확도가 아니라 PR 곡선 기반이므로 클래스 불균형에 민감하다.

## 더 보기
- 선행 개념: [`../../06_cnn/04_resnet/concept.md`](../../06_cnn/04_resnet/concept.md) — 탐지 백본으로 쓰이는 ResNet
- 다음: [`../02_segmentation/concept.md`](../02_segmentation/concept.md) — 픽셀 단위 분할
