# 이미지 생성 (GAN · VAE · Diffusion)

> 테마: 09_cv_advanced · 개념 정리 (실습은 리포지토리 README 참고)

## 한 줄 요약
노이즈나 잠재 변수로부터 사실적인 이미지를 만드는 생성 모델 세 계열(GAN, VAE, Diffusion)의 원리와 차이를 정리한다.

## 핵심 개념
- **GAN (Generative Adversarial Network)**: **generator**(노이즈 z → 가짜 이미지)와 **discriminator**(진짜/가짜 판별)가 경쟁하는 minimax 게임. G는 D를 속이도록, D는 잘 가려내도록 동시에 학습.
- **VAE (Variational Autoencoder)**: encoder가 입력을 **latent space**의 분포(μ, σ)로 인코딩, decoder가 latent를 샘플링해 복원. 연속적이고 매끄러운 잠재 공간을 학습.
- **Diffusion (DDPM)**: **forward** 과정에서 이미지에 점진적으로 가우시안 noise를 더해 순수 노이즈로 만들고, **reverse** 과정에서 신경망이 noise를 단계적으로 제거(denoising)하며 이미지를 복원/생성.

## 원리 / 수식
- GAN 목적: `min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]`. 균형점에서 G의 분포가 데이터 분포에 수렴.
- VAE **ELBO** = (재구성 항) − (KL 항): `E[log p(x|z)] − KL(q(z|x) ‖ p(z))`. 재구성 품질과 잠재 분포의 정규성을 동시에 최적화.
- Diffusion: forward `q(x_t|x_{t-1})`로 noise 추가, 모델은 각 step의 noise ε을 예측해 reverse `p(x_{t-1}|x_t)`를 학습. 손실은 대개 예측 noise의 MSE.

## PyTorch 구현 포인트
```python
# GAN의 한 스텝 (개념 골격)
d_loss = bce(D(real), 1) + bce(D(G(z).detach()), 0)   # D 업데이트
g_loss = bce(D(G(z)), 1)                              # G 업데이트
# VAE 재파라미터화 트릭
z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
```
- VAE는 샘플링을 미분 가능하게 만드는 **reparameterization trick**이 핵심.
- Diffusion 학습은 보통 임의 step `t`를 샘플링해 해당 noise를 예측하도록 한다.

## 자주 하는 실수 / 팁
- GAN: mode collapse(다양성 상실)와 학습 불안정 — D/G 균형, 학습률 조절이 중요. `G(z)`를 D 업데이트 시 `detach()`로 끊을 것.
- VAE 출력은 다소 흐릿(blurry)한 경향. KL 가중치가 너무 크면 posterior collapse 발생.
- Diffusion은 품질이 높지만 다단계 샘플링이라 추론이 느리다(스텝 수가 속도-품질 트레이드오프).

## 더 보기
- GAN 원논문: Goodfellow et al., 2014 — Generative Adversarial Nets
- DDPM: Ho et al., 2020 — Denoising Diffusion Probabilistic Models
- VAE: Kingma & Welling, 2013 — Auto-Encoding Variational Bayes
