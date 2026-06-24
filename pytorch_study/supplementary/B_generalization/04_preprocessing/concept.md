# 전처리 / 증강 (Preprocessing & Augmentation)

> 보강 학습 · B. 일반화 & 평가 · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
입력을 일관되게 정규화하고(train 통계 기준) 이미지에 무작위 변형을 더하며, 불균형 데이터는 weighted sampling이나 class weight로 보정한다.

## 핵심 개념
- **`torchvision.transforms`**: 이미지 전처리·증강을 함수 체인으로 구성하는 도구. `Compose`로 순서대로 묶는다.
- **normalize(mean/std)**: 채널별 평균을 빼고 표준편차로 나눠 입력 분포를 표준화한다. 학습 안정·수렴을 돕는다.
- **image augmentation**: RandomHorizontalFlip, RandomCrop, RandomRotation 등으로 데이터를 사실상 늘려 불변성을 학습시킨다.
- **불균형 보정**: `WeightedRandomSampler`로 소수 클래스를 더 자주 뽑거나, 손실에 `class weight`를 줘 소수 클래스 오류에 가중한다.

## 원리 / 수식
- 정규화: $x' = \dfrac{x - \mu}{\sigma}$ (채널별 $\mu, \sigma$). ToTensor가 [0,1]로 만든 뒤 Normalize가 표준화한다.
- augmentation은 데이터 분포를 키워(label-preserving transform) generalization gap을 줄인다.
- class weight: 손실에서 클래스 $c$ 항에 $w_c$(보통 빈도 역수)를 곱해 불균형을 상쇄한다.
- 정규화로 입력 스케일을 맞추면 gradient가 특정 차원에 치우치지 않아 학습률을 더 안정적으로 쓸 수 있다.

## PyTorch 구현 포인트
```python
from torchvision import transforms
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std =[0.247, 0.243, 0.261])])

# 불균형: 클래스 가중 손실 / 가중 샘플러
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
```
- val/test에는 무작위 증강 없이 `ToTensor` + 동일 `Normalize`만 쓴다.

## 자주 하는 실수 / 팁
- Normalize의 mean/std는 **train 데이터**에서 구한다(val/test도 같은 값으로 transform).
- `ToTensor` 다음에 `Normalize`를 둔다(텐서·[0,1] 스케일 이후 적용).
- 무작위 augmentation을 val/test에 적용하면 평가가 흔들린다.
- `WeightedRandomSampler`를 쓰면 DataLoader에 `shuffle=True`를 함께 줄 수 없다(샘플러가 순서를 정함).
- augmentation이 과하면 label을 망가뜨릴 수 있다(예: 숫자 이미지를 너무 회전). 데이터 특성에 맞게 강도를 조절한다.
- 표 형태(tabular) 데이터에는 transforms 대신 표준화/원-핫 인코딩 등 별도 전처리를 적용한다.

## 더 보기
- 누수 없이 train에만 fit하는 분할 원칙: [`../03_data_split/concept.md`](../03_data_split/concept.md)
- PyTorch transforms 튜토리얼: https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
- d2l.ai Image Augmentation: https://d2l.ai/chapter_computer-vision/image-augmentation.html
