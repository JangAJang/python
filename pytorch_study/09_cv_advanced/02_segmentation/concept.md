# 분할 (Semantic / Instance Segmentation)

> 테마: 09_cv_advanced · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
이미지를 픽셀 단위로 분류하는 과제로, 같은 클래스를 묶는 semantic과 개별 객체까지 구분하는 instance로 나뉜다.

## 핵심 개념
- **semantic segmentation**: 모든 픽셀에 클래스 라벨 부여. 같은 클래스의 두 객체는 구분하지 않음(자동차 두 대 → 모두 "car").
- **instance segmentation**: 픽셀 분류 + 객체 인스턴스 구분(자동차 1, 자동차 2). 탐지 + 분할의 결합.
- **FCN (Fully Convolutional Network)**: FC layer를 conv로 대체해 임의 크기 입력의 dense 예측을 가능케 한 초기 모델.
- **U-Net**: encoder-decoder 구조. encoder가 다운샘플로 문맥을 압축, decoder가 업샘플로 해상도 복원. **skip connection**으로 같은 해상도의 encoder feature를 decoder에 이어 붙여 경계 디테일을 보존.
- **Mask R-CNN**: Faster R-CNN에 mask 분기를 추가해 RoI마다 binary mask를 예측하는 instance segmentation 모델.

## 원리 / 수식
- 출력은 `[N, num_classes, H, W]` 형태의 픽셀별 logit. semantic은 픽셀별 cross entropy로 학습.
- **Dice loss**: `1 - 2|P∩G| / (|P| + |G|)`. 클래스 불균형(작은 전경)에 강함.
- **IoU loss**: `1 - IoU`. 분할 품질을 직접 최적화.
- U-Net은 RoIAlign 없이 전체 이미지를 한 번에 처리하는 dense prediction.

## PyTorch 구현 포인트
```python
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.detection import maskrcnn_resnet50_fpn

seg = fcn_resnet50(weights="DEFAULT").eval()
out = seg(img_batch)["out"]          # [N, num_classes, H, W]
pred_mask = out.argmax(dim=1)        # 픽셀별 클래스

inst = maskrcnn_resnet50_fpn(weights="DEFAULT").eval()
# inst([img])[0] -> {"boxes","labels","scores","masks"}
```
- semantic 모델 출력은 dict의 `"out"` 키. argmax로 클래스 맵을 얻는다.
- Mask R-CNN의 `masks`는 `[N,1,H,W]` 확률 맵 → threshold(예: 0.5)로 이진화.

## 자주 하는 실수 / 팁
- semantic 출력에서 `argmax` 차원은 채널(`dim=1`)임을 확인.
- 마스크 해상도와 원본 입력 해상도가 다르면 `F.interpolate`로 맞춰야 한다.
- 전경 픽셀이 적으면 plain cross entropy보다 Dice/IoU loss가 안정적이다.

## 더 보기
- 선행 개념: [`../01_object_detection/concept.md`](../01_object_detection/concept.md) — Mask R-CNN의 토대
- 다음: [`../03_backbones/concept.md`](../03_backbones/concept.md) — 백본 발전
