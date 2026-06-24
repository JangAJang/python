# 이미지 폴더 데이터셋 (torchvision ImageFolder)

> 테마: 06_cnn · 예제: [`example.ipynb`](./example.ipynb) · 실습: [`practice.ipynb`](./practice.ipynb)

## 한 줄 요약
폴더 이름을 클래스 라벨로 자동 인식하는 `torchvision.datasets.ImageFolder`로 내 로컬 이미지를 손쉽게 데이터셋으로 만들어 CNN을 학습/평가한다.

## 핵심 개념
- **ImageFolder의 규칙**: `root/클래스명/이미지.jpg` 형태로 폴더를 구성하면, 하위 폴더 이름이 알파벳 순으로 정렬되어 라벨 0,1,2...로 자동 매핑된다. (예: `gray/` → 0, `red/` → 1)
- **transform**: 불러올 때 `transforms.Compose([...])`로 리사이즈/텐서 변환 등을 적용한다.
- **전처리 파이프라인**: 원본(`origin_data`)을 리사이즈해 저장 → 학습용(`train_data`) 폴더로 정리 → `ToTensor`로 다시 로드해 학습.
- **DataLoader**: `ImageFolder`를 `DataLoader`로 감싸 배치/셔플/병렬 로딩을 처리.
- **모델 저장/로드**: `torch.save(model.state_dict(), path)` / `model.load_state_dict(torch.load(path))`.

## 원리 / 수식
- `ImageFolder`는 `(image, label)` 튜플을 반환하는 `Dataset`이다. `__getitem__`/`enumerate`로 순회 가능.
- 입력 크기 `3 x 64 x 128` 기준 예제 CNN: `conv(3→6,5x5) → pool → conv(6→16,5x5) → pool → flatten(16·13·29) → Linear(→120) → Linear(→2)`.
- FC 입력 차원 `16*13*29`는 conv/pool을 거친 특성 맵 크기에서 계산된다. 입력 해상도가 바뀌면 이 값도 바뀐다.

## PyTorch 구현 포인트
- `torchvision.datasets.ImageFolder(root, transform)` : 폴더 기반 데이터셋.
- `transforms.Compose([transforms.Resize((64,128)), transforms.ToTensor()])`.
- `DataLoader(dataset, batch_size, shuffle=True, num_workers=2)`.
- `torch.save` / `torch.load` + `state_dict` 로 모델 가중치 저장·복원.

```python
trans = transforms.Compose([transforms.Resize((64,128)), transforms.ToTensor()])
train_data = torchvision.datasets.ImageFolder(root='custom_data/train_data', transform=trans)
loader = DataLoader(train_data, batch_size=8, shuffle=True)
```

## 자주 하는 실수 / 팁
- 클래스 폴더가 하나라도 비어 있거나 이미지가 아닌 파일이 섞이면 로딩 에러가 난다.
- 라벨 순서는 폴더 이름의 알파벳 정렬을 따른다. 의도한 매핑인지 `train_data.class_to_idx`로 확인.
- 학습/테스트 모두 동일한 `transform`(특히 `Resize`)을 적용해야 FC 입력 차원이 일치한다.
- `num_workers > 0`은 Windows/노트북 환경에서 오류를 낼 수 있으니 문제가 생기면 0으로 두자.
- flatten 차원(`16*13*29`)은 입력 해상도에 종속적이므로, 해상도를 바꾸면 반드시 다시 계산할 것.

## 예제 노트북 요약
- `example.ipynb`는 로컬 사진을 `ImageFolder`로 불러와 리사이즈 후 학습용 폴더로 저장하고, 직접 정의한 작은 CNN으로 2클래스(gray/red)를 분류한다.
- 학습 후 `state_dict`로 모델을 저장·재로드하고 test_data로 정확도(예제에서는 1.0)를 확인한다.

## 더 보기
- 선행 개념: [`../02_mnist_cnn/concept.md`](../02_mnist_cnn/concept.md) — CNN 학습 흐름
- 데이터 로딩 기초: [`../../02_regression/04_data_loading/concept.md`](../../02_regression/04_data_loading/concept.md)
- 학습 시각화: [`../06_visdom/concept.md`](../06_visdom/concept.md)
