---
title: "Denoising Diffusion Probabilistic Models"
authors: ["Jonathan Ho", "Ajay Jain", "Pieter Abbeel"]
year: 2020
arxiv: "2006.11239"
url: https://arxiv.org/abs/2006.11239
priority: Must-Read
read_on: 2026-08-22
tags: [paper, diffusion, vision, theory]
---
## The Core Idea

Take a photo. Add a tiny bit of Gaussian noise. Repeat 1000 times. You end up with pure static — no signal left. That destruction process is fixed, hand-designed, has zero learnable parameters, and every step is a simple Gaussian.

Now learn to run the film backwards. If each forward step added only a *little* noise, then each backward step is also (approximately) a Gaussian, so you can model it with a neural network that outputs a mean. Chain 1000 of those learned backward steps starting from pure static and you get a new image.

That idea is from Sohl-Dickstein et al. (2015). It did not produce good pictures. What this paper adds is the **parameterisation trick that makes it work**: instead of asking the network to predict the mean of the reverse step, ask it to predict *the noise that was added*. Then train it with plain mean squared error on that noise, throwing away the weighting factor that the proper variational bound says you should use.

$$L_{\text{simple}}(\theta) = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol\epsilon}\left[\|\boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon,\ t)\|^2\right]$$

That is the whole loss. A regression on noise. It turned CIFAR10 FID from the 25–31 range (score matching, NCSN) down to **3.17**, beating BigGAN (14.73) and matching StyleGAN2+ADA (3.26) — with no adversarial game, no discriminator, no [[Mode Collapse]].

The second contribution is conceptual: this $\epsilon$-prediction objective *is* denoising score matching at many noise levels, and the sampling loop *is* annealed Langevin dynamics. Two research lines that looked different — "latent variable model trained by variational inference" and "learn $\nabla_x \log p(x)$ and do Langevin sampling" — turn out to be the same thing written two ways. $\boldsymbol\epsilon_\theta$ is, up to a scale factor, the [[Derivative#Gradient|gradient]] of the log data density.

What it unlocks: a generative model that is **stable to train** (it is just supervised regression), covers modes (it optimises a likelihood bound, unlike a [[Generative Adversarial Networks|GAN]]), and scales to $256\times256$ images. Everything downstream — Stable Diffusion, DALL·E 2, Imagen — is this loss plus conditioning.

> [!NOTE] Diffusion model
> A latent variable model where the *encoder* is fixed noise-adding (no parameters) and only the *decoder* — the reverse chain — is learned. Unlike a [[Auto-Encoding Variational Bayes (VAE)|VAE]], the latents $\mathbf{x}_1,\dots,\mathbf{x}_T$ have the same dimension as the data, and the top latent $\mathbf{x}_T$ carries almost zero information about $\mathbf{x}_0$. ^diffusion-model

## The Methodology

**Forward process (fixed, no learning).** A [[Markov Property|Markov]] chain that adds noise:

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\!\left(\mathbf{x}_t;\ \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\ \beta_t \mathbf{I}\right)$$

The $\sqrt{1-\beta_t}$ shrink factor matters — it keeps the variance from exploding, so the network always sees inputs on a consistent scale. (NCSN omits this; the paper calls it out as one reason DDPM samples are better.)

The key algebraic gift: because Gaussians compose, you can jump to any timestep in one shot. With $\alpha_t := 1-\beta_t$ and $\bar\alpha_t := \prod_{s\le t}\alpha_s$:

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\ \sqrt{\bar\alpha_t}\mathbf{x}_0,\ (1-\bar\alpha_t)\mathbf{I}) \quad\Longleftrightarrow\quad \mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon$$

So training never has to simulate the chain. Pick a random $t$, jump straight there. This is the same reparameterisation move as in [[Auto-Encoding Variational Bayes (VAE)#^reparameterisation-trick|the VAE]].

**Reverse process (learned).** $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol\mu_\theta(\mathbf{x}_t,t), \sigma_t^2\mathbf{I})$. The variance is **not learned** — it is fixed to either $\beta_t$ or $\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$. Both worked about the same.

**The objective.** Start from the usual variational bound (an [[Auto-Encoding Variational Bayes (VAE)#^elbo|ELBO]]) and rewrite it as a sum of [[KL Divergence|KL divergences]]:

$$L = \underbrace{D_{\mathrm{KL}}(q(\mathbf{x}_T|\mathbf{x}_0)\,\|\,p(\mathbf{x}_T))}_{L_T} + \sum_{t>1}\underbrace{D_{\mathrm{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)\,\|\,p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0|\mathbf{x}_1)}_{L_0}$$

Every term compares two Gaussians, so every KL has a closed form — no high-variance Monte Carlo. $L_T$ is a constant (nothing learnable in $q$) and is about $10^{-5}$ bits/dim, so it drops out.

Each middle term reduces to a squared distance between means:

$$L_{t-1} = \mathbb{E}_q\left[\tfrac{1}{2\sigma_t^2}\|\tilde{\boldsymbol\mu}_t(\mathbf{x}_t,\mathbf{x}_0) - \boldsymbol\mu_\theta(\mathbf{x}_t,t)\|^2\right] + C$$

**The trick.** Substitute $\mathbf{x}_0 = \frac{1}{\sqrt{\bar\alpha_t}}(\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon)$ into the target mean. Everything cancels except $\boldsymbol\epsilon$. So instead of a network that outputs a mean, define

$$\boldsymbol\mu_\theta(\mathbf{x}_t,t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)\right)$$

and the loss becomes

$$\mathbb{E}\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\|\boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\mathbf{x}_t,t)\|^2\right]$$

Drop the ugly prefactor $\Rightarrow$ $L_{\text{simple}}$. This is not neutral: with the linear $\beta$ schedule the prefactor is *large* for small $t$, so dropping it **down-weights the easy near-clean denoising tasks** and makes the network spend capacity on heavily-noised inputs. The authors argue that reweighting is exactly why sample quality improves.

**Training loop (Algorithm 1).** Sample an image, sample $t \sim \text{Uniform}\{1..1000\}$, sample $\boldsymbol\epsilon\sim\mathcal N(0,I)$, form $\mathbf{x}_t$ in closed form, take one [[Backpropagation|gradient step]] on the MSE. That is it.

**Sampling loop (Algorithm 2).** $\mathbf{x}_T\sim\mathcal N(0,I)$, then for $t=T\dots1$:
$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)\right) + \sigma_t \mathbf{z}$$
1000 network evaluations per image. This looks exactly like Langevin dynamics — a gradient step toward high density plus injected noise — which is why it connects to [[Markov Chain Monte Carlo]] samplers.

**Architecture.** A U-Net built on PixelCNN++ / Wide ResNet, i.e. stacked [[Deep Residual Learning for Image Recognition (ResNet)#^residual-block|residual blocks]] with downsampling then upsampling and skip connections. Group normalisation instead of weight norm. **[[Causal Attention|Self-attention]] at the $16\times16$ feature map** (unmasked — it can look everywhere). Timestep $t$ enters via the [[Attention Is All You Need|Transformer]] sinusoidal position embedding, added into *every* residual block — not just at the output. One network, shared across all 1000 timesteps.

**Numbers that mattered.**

| Setting | Value |
|---|---|
| $T$ | 1000 (chosen without a sweep) |
| $\beta_t$ | linear, $10^{-4} \to 0.02$ |
| Optimiser | Adam, lr $2\times10^{-4}$ (dropped to $2\times10^{-5}$ at $256^2$ — unstable otherwise) |
| Batch | 128 (CIFAR), 64 (256²) |
| Dropout | 0.1 on CIFAR (swept over 0.1–0.4); **0** elsewhere |
| EMA decay | 0.9999 |
| Params | 35.7M (CIFAR), 114M (LSUN/CelebA), 256M (large LSUN Bedroom) |
| Data scaling | ints $\{0..255\}\to[-1,1]$ |
| Hardware | TPU v3-8; CIFAR trains in **10.6 hours** to 800k steps |

$L_0$ uses a discretised Gaussian decoder — integrate the Gaussian density over each $\frac{1}{255}$-wide pixel bin — so the bound is a genuine lossless codelength on discrete data.

## Ablation Studies and Experiments

**CIFAR10 headline (Table 1, unconditional):**

| Model | IS ↑ | FID ↓ | NLL bits/dim ↓ |
|---|---|---|---|
| NCSN (score matching) | 8.87 | 25.32 | — |
| SNGAN | 8.22 | 21.7 | — |
| StyleGAN2 + ADA | **9.74** | 3.26 | — |
| Sparse Transformer (autoregressive) | — | — | **2.80** |
| Gated PixelCNN | 4.60 | 65.93 | 3.03 |
| **Ours, true bound $L$** | 7.67 | 13.51 | **≤3.70** |
| **Ours, $L_{\text{simple}}$** | 9.46 | **3.17** | ≤3.75 |

Note the tension: the true variational bound gives the better likelihood (3.70 vs 3.75) but far worse samples (FID 13.51 vs 3.17). Optimising the thing you claim to optimise makes the pictures worse.

Also: beats *class-conditional* BigGAN (FID 14.73) while being unconditional. FID against the test set instead of train set is 5.24 — still competitive.

**The main ablation (Table 2)** — parameterisation × objective. Blank cells mean *training was unstable and samples were garbage*:

| Parameterisation | Objective | IS | FID |
|---|---|---|---|
| predict $\tilde{\boldsymbol\mu}$ | $L$, learned diagonal $\Sigma$ | 7.28 | 23.69 |
| predict $\tilde{\boldsymbol\mu}$ | $L$, fixed isotropic $\Sigma$ | 8.06 | 13.22 |
| predict $\tilde{\boldsymbol\mu}$ | unweighted MSE on $\tilde\mu$ | — | — |
| predict $\boldsymbol\epsilon$ | $L$, learned diagonal $\Sigma$ | — | — |
| predict $\boldsymbol\epsilon$ | $L$, fixed isotropic $\Sigma$ | 7.67 | 13.51 |
| **predict $\boldsymbol\epsilon$** | **$L_{\text{simple}}$** | **9.46** | **3.17** |

Three things this reveals:

1. **Neither ingredient works alone.** $\epsilon$-prediction with the full bound is *no better* than $\mu$-prediction with the full bound (13.51 vs 13.22 — a wash). $\mu$-prediction with unweighted MSE outright fails. The 4× FID improvement only appears when you combine $\epsilon$-prediction **and** the reweighting. The reweighting is the active ingredient; $\epsilon$-prediction is what makes the reweighting stable.
2. **Learning the reverse variance is actively harmful.** Diagonal $\Sigma_\theta$ made training unstable and samples worse. Freezing $\Sigma$ to a constant schedule beats learning it.
3. **Predicting $\mathbf{x}_0$ directly** (a third option, mentioned in §3.2) gave worse sample quality early on and was dropped — no numbers reported.

**LSUN $256\times256$ (Table 3, FID):** Bedroom 6.36 (4.90 with the 256M model), Church 7.89, Cat 19.75. ProgressiveGAN gets 8.34 / 6.42 / 37.52; StyleGAN2 gets 3.86 on Church. So DDPM beats ProgressiveGAN on Bedroom and Cat, loses to StyleGAN2 on Church.

**Rate–distortion / progressive coding (§4.3, Table 4).** Splitting the bound into rate ($L_1..L_T$ = 1.78 bits/dim) and distortion ($L_0$ = 1.97 bits/dim, RMSE 0.95 on a 0–255 scale). The curve is brutally lopsided: after receiving only **0.12 bits/dim** (the first 900 reverse steps' worth), RMSE is already down to 12.0 out of 255. The remaining **1.66 bits/dim** buys you the drop from RMSE 12 to RMSE 0.95 — i.e. more than half the total codelength describes distortions the eye cannot see. That is the paper's explanation for why the likelihood is mediocre while the samples are excellent: diffusion models have an inductive bias toward being great *lossy* compressors.

**Progressive generation (Fig. 6, 10).** Watching $\hat{\mathbf{x}}_0$ over the reverse chain: coarse structure first, fine detail last. Inception Score climbs from ~2 to ~9.5 over the 1000 steps; FID falls from ~300 to ~3.

**Latent structure (Fig. 7).** Freeze $\mathbf{x}_t$ and re-run the reverse chain several times. At $t=1000$ the samples are unrelated. At $t=750$ they already share gender, hair colour, eyewear, pose, expression — even though $\mathbf{x}_{750}$ looks like pure noise to a human.

**Interpolation (§4.4, Fig. 8/9).** Encode two images to $\mathbf{x}_t$, linearly mix, decode back. At $t=500$ you get smooth changes in pose, skin tone, hairstyle, expression, background — **but not eyewear** (glasses pop in and out discretely). At $t=0$ it degenerates to pixel-space blending; at $t=1000$ you just get a novel sample.

**Overfitting check.** Train/test likelihood gap ≤ 0.03 bits/dim. Nearest-neighbour visualisations in pixel and Inception feature space show it is not memorising. Without dropout on CIFAR, samples showed overfitting artefacts — so [[Regularization|regularisation]] mattered there.

## Worth Remembering

- **Likelihood is not the goal here and the authors say so.** 3.75 bits/dim loses to Sparse Transformer's 2.80. Diffusion models were, at this point, bad density estimators that made beautiful pictures. The paper's honest framing is that the good samples come from a *weighted* bound that is no longer a valid bound in the tight sense.
- **1000 forward passes per image.** Sampling 128 CelebA images takes 300 seconds on a TPU v3-8. Training is cheap (10.6 GPU-hours for CIFAR), inference is not. Every follow-up paper on fast samplers (DDIM, distillation, consistency models) exists because of this line.
- **The compression scheme is theatre, not a system.** Algorithms 3 and 4 need minimal random coding, which the appendix admits "is not tractable for high dimensional data." It is an interpretation of the bound, not a codec.
- **Diffusion as generalised autoregression (§4.3).** Rewrite the bound as $L = D_{\mathrm{KL}}(q(\mathbf{x}_T)\|p(\mathbf{x}_T)) + \sum_t D_{\mathrm{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t)\|p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)) + H(\mathbf{x}_0)$. Now set $T = $ number of pixels and make the "noise" be *masking out one coordinate at a time*. You have exactly trained an [[Auto-regressive models|autoregressive model]]. Gaussian diffusion is the same object with a "bit ordering" that no coordinate permutation can express — and crucially $T$ is decoupled from data dimension (1000 steps for a $256\times256\times3 = 196{,}608$-dim image). Shorter $T$ for speed, longer for expressiveness.
- **Why it beats NCSN (Appendix C), four concrete differences:** (1) U-Net + self-attention vs RefineNet, and $t$ conditioning injected into *every* block; (2) the $\sqrt{1-\beta_t}$ rescaling keeps input variance stable; (3) tiny $\beta_t$ and a forward process that fully destroys signal, so the prior matches the aggregate posterior and there is no distribution shift at sampling time; (4) sampler step sizes derived from $\beta_t$ rather than hand-tuned post-hoc — the sampler is *trained*, not bolted on.
- **Almost no hyperparameter search.** $T=1000$: no sweep. Learning rate: no sweep. Batch size: no sweep. EMA decay: no sweep. They swept dropout and the $\beta$ schedule shape on CIFAR and transferred everything else. That is a strong signal the method is robust, and worth remembering when someone claims diffusion is finicky.
- **Practical caveat:** the $256\times256$ models needed lr dropped 10× to $2\times10^{-5}$ to train stably. Scale changes the optimisation regime.
- **Open question the paper leaves:** the $L_{\text{simple}}$ weighting was found empirically, tied to this specific linear $\beta$ schedule. What is the *right* weighting? (Answered later by Improved DDPM's cosine schedule and by min-SNR weighting.)

## Links

Related: [[Auto-Encoding Variational Bayes (VAE)]] · [[KL Divergence]] · [[Generative Adversarial Networks]] · [[Mode Collapse]] · [[Markov Chain Monte Carlo]] · [[Markov Property]] · [[Hamiltonian Monte Carlo]] · [[Auto-regressive models]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Attention Is All You Need]] · [[Causal Attention]] · [[Cross Entropy]] · [[Random variable]] · [[Regularization]] · [[Loss, Objectives, and Business Alignment]]

New topics worth writing: Score matching and denoising score matching, Langevin dynamics, U-Net architecture, FID and Inception Score, Rate–distortion theory, DDIM and fast diffusion samplers, Classifier-free guidance, Group normalisation, Exponential moving average of weights, Nonequilibrium thermodynamics origins of diffusion models
