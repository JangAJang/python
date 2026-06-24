# LLM 파인튜닝 (Fine-tuning & PEFT)

> 테마: 08_transformer · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
거대 모델 전체를 다시 학습하는 full fine-tuning 대신, 소수의 파라미터만 학습하는 PEFT(LoRA/QLoRA/adapter)로 메모리·비용을 크게 줄여 적응시키는 기법.

## 핵심 개념
- **full fine-tuning**: 모델의 모든 가중치를 업데이트. 성능은 좋지만 GPU 메모리·저장 비용이 모델 크기에 비례해 매우 크다.
- **PEFT (Parameter-Efficient Fine-Tuning)**: 대부분의 가중치는 동결(freeze)하고 소량의 추가 파라미터만 학습.
- **LoRA (Low-Rank Adaptation)**: 가중치 변화량을 두 저차원 행렬의 곱 `BA`로 근사해 학습한다.
- **QLoRA**: 기반 모델을 4bit로 양자화해 메모리에 올리고, 그 위에 LoRA 어댑터를 학습한다.
- **adapter**: 각 Transformer 블록 사이에 작은 병목 MLP를 끼워 그 부분만 학습한다.

## 원리 / 수식
- LoRA: 원 가중치 `W₀`는 동결하고 `W = W₀ + ΔW`, `ΔW = B·A` (A: r×d, B: d×r, rank `r ≪ d`).
  - 학습 파라미터는 `2·d·r`로 `d²` 대비 극적으로 작다. 보통 `α/r` 스케일을 곱한다.
  - 추론 시 `BA`를 `W₀`에 병합하면 추가 지연이 없다.
- QLoRA: 기반 모델 = 4bit(NF4) 양자화로 VRAM 절약, gradient는 LoRA 어댑터(고정밀)만 흐른다.
- 트레이드오프: full은 표현력↑·비용↑, PEFT는 비용↓·다중 태스크 어댑터 교체가 쉽다.

## PyTorch 구현 포인트
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# QLoRA: 4bit 양자화 로드
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype="bfloat16")
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                            quantization_config=bnb)

cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                 lora_dropout=0.05, task_type="CAUSAL_LM")
model = get_peft_model(base, cfg)
model.print_trainable_parameters()   # 전체 대비 학습 파라미터 비율 확인
```
- `target_modules`로 LoRA를 붙일 선형층(보통 attention의 q/v proj)을 지정한다.
- `r`(rank)과 `lora_alpha`가 핵심 하이퍼파라미터다.

## 자주 하는 실수 / 팁
- `r`을 너무 작게 잡으면 표현력 부족, 너무 크게 잡으면 PEFT의 이점이 사라진다.
- QLoRA에서 기반 모델은 4bit로 동결되어 학습되지 않는다. LoRA 어댑터만 저장하면 된다.
- adapter/LoRA 가중치는 base와 별도로 저장·공유할 수 있어 태스크별 교체가 쉽다.
- 양자화 학습에는 `bitsandbytes` 등 호환 라이브러리·CUDA 환경이 필요하다.

## 더 보기
- 선행 개념: [`../04_huggingface/concept.md`](../04_huggingface/concept.md) — HuggingFace 워크플로
- 다음 단계: [`../06_rag/concept.md`](../06_rag/concept.md) — RAG로 외부 지식 결합
- 외부 자료: https://huggingface.co/docs/peft
