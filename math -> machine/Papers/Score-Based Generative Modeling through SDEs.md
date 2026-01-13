---
title: "Score-Based Generative Modeling through SDEs"
authors: ["Yang Song", "Jascha Sohl-Dickstein", "Diederik P. Kingma", "Abhishek Kumar", "Stefano Ermon", "Ben Poole"]
year: 2020
arxiv: "2011.13456"
url: https://arxiv.org/abs/2011.13456
priority: Must-Read
read_on: 2026-08-23
tags: [paper, diffusion, vision, theory]
---
## The Core Idea

Adding noise to a picture is easy. Removing it is generative modelling. This paper says: stop thinking of noising as a *finite ladder* of steps, and think of it as a *continuous flow in time* described by a stochastic differential equation (SDE).

> [!NOTE] Stochastic differential equation (SDE)
> A rule for how a point moves in tiny time steps, with a predictable push (the *drift*) plus random jitter (the *diffusion*). $\mathrm{d}\mathbf{x} = \mathbf{f}(\mathbf{x},t)\mathrm{d}t + g(t)\mathrm{d}\mathbf{w}$, where $\mathbf{w}$ is Brownian motion. ^ito-sde

The forward SDE takes a real image at $t=0$ and, by $t=T$, has smeared it into pure Gaussian noise. It has **no learnable parameters at all** — it is a fixed physical process you choose.

The magic is a 1982 result by Anderson: every diffusion process run backwards in time is also a diffusion process, and its equation is

$$\mathrm{d}\mathbf{x} = \big[\mathbf{f}(\mathbf{x},t) - g(t)^2\,\nabla_\mathbf{x}\log p_t(\mathbf{x})\big]\mathrm{d}t + g(t)\,\mathrm{d}\bar{\mathbf{w}}$$

The only unknown in that reverse equation is $\nabla_\mathbf{x}\log p_t(\mathbf{x})$ — the **score**.

> [!NOTE] Score
> The [[Derivative#Gradient|gradient]] of the log-density *with respect to the data*, not the parameters. It is a vector field that points "uphill" toward regions where data is more likely. Learn it, and you can walk noise back into data. ^score

So: train one neural net $\mathbf{s}_\theta(\mathbf{x},t)$ to predict the score at every noise level $t$, plug it into the reverse SDE, and solve that SDE numerically. That is the whole generative model.

Why this matters, and what it unlocks:

1. **Unification.** [[Denoising Diffusion Probabilistic Models|DDPM]] and Song & Ermon's score matching with Langevin dynamics (SMLD) turn out to be two *discretisations of two different SDEs*. They were never really different families.
2. **Any solver becomes a sampler.** Once it is an SDE, you can use Euler–Maruyama, Runge–Kutta, or a hybrid with MCMC. Sampling is no longer welded to training.
3. **An exact likelihood.** Every SDE has a matching deterministic ODE with the *same* marginal densities at every $t$. That ODE is a neural ODE, so you get exact log-likelihood by the change-of-variables formula — something DDPM could only bound with an ELBO.
4. **Conditioning without retraining.** Since $\nabla_\mathbf{x}\log p_t(\mathbf{x}\mid\mathbf{y}) = \nabla_\mathbf{x}\log p_t(\mathbf{x}) + \nabla_\mathbf{x}\log p_t(\mathbf{y}\mid\mathbf{x})$, you just *add* a second gradient to the score. This is the ancestor of classifier guidance.

Result: FID 2.20 and Inception Score 9.89 unconditional on CIFAR-10 (beating the best *conditional* GAN of the time), 2.99 bits/dim likelihood, and the first $1024\times1024$ images from a score model.

## The Methodology

### The two (three) SDEs

**Variance Exploding (VE)** — the continuous limit of SMLD. Noise is *added on top* of the image, so the variance grows without bound:
$$\mathrm{d}\mathbf{x} = \sqrt{\tfrac{\mathrm{d}[\sigma^2(t)]}{\mathrm{d}t}}\,\mathrm{d}\mathbf{w}$$
With the usual geometric noise schedule, $\sigma(t) = \sigma_{\min}(\sigma_{\max}/\sigma_{\min})^t$, $\sigma_{\min}=0.01$.

**Variance Preserving (VP)** — the continuous limit of DDPM. The image is shrunk toward zero *as* noise is added, so total variance stays at 1:
$$\mathrm{d}\mathbf{x} = -\tfrac{1}{2}\beta(t)\mathbf{x}\,\mathrm{d}t + \sqrt{\beta(t)}\,\mathrm{d}\mathbf{w}$$
with $\beta(t) = \bar\beta_{\min} + t(\bar\beta_{\max}-\bar\beta_{\min})$, $\bar\beta_{\min}=0.1$, $\bar\beta_{\max}=20$ to match Ho et al.

**sub-VP** — new here. Same drift as VP but with a shrunken diffusion term:
$$\mathrm{d}\mathbf{x} = -\tfrac{1}{2}\beta(t)\mathbf{x}\,\mathrm{d}t + \sqrt{\beta(t)\big(1-e^{-2\int_0^t\beta(s)\mathrm{d}s}\big)}\,\mathrm{d}\mathbf{w}$$
Its variance is provably $\preccurlyeq$ the VP variance at every $t$, but it still converges to $\mathcal{N}(\mathbf{0},\mathbf{I})$. It turns out to give the best likelihoods.

All three have **affine drift**, which is the whole reason they are usable: affine drift ⟹ the transition kernel $p_{0t}(\mathbf{x}(t)\mid\mathbf{x}(0))$ is Gaussian with a closed form. So you can jump straight from a clean image to time $t$ in one shot during training. No simulation loop.

### The loss

One weighted denoising-score-matching objective over continuous time:

$$\theta^* = \arg\min_\theta\ \mathbb{E}_t\Big\{\lambda(t)\,\mathbb{E}_{\mathbf{x}(0)}\mathbb{E}_{\mathbf{x}(t)\mid\mathbf{x}(0)}\big[\lVert \mathbf{s}_\theta(\mathbf{x}(t),t) - \nabla_{\mathbf{x}(t)}\log p_{0t}(\mathbf{x}(t)\mid\mathbf{x}(0))\rVert_2^2\big]\Big\}$$

In plain terms: sample a clean image, sample a random time $t$, add the exact Gaussian noise for that $t$, and ask the network to predict the score of that *known* Gaussian. Because the perturbation kernel is Gaussian with mean $\mu$ and variance $\sigma_t^2$, its score is just $-(\mathbf{x}(t)-\mu)/\sigma_t^2$ — i.e. a scaled copy of the noise you just added. This is exactly DDPM's noise-prediction loss in different clothes.

The weight is chosen as $\lambda(t) \propto 1/\mathbb{E}[\lVert\nabla\log p_{0t}\rVert^2]$, which reduces to $\sigma_i^2$ for SMLD and $(1-\alpha_i)$ for DDPM. It roughly equalises the loss magnitude across noise levels.

Practical detail that matters: you never integrate to $t=0$. Variance vanishes there and things blow up numerically. They clip to $t\in[\epsilon,1]$ with $\epsilon=10^{-5}$ for training/likelihood and $\epsilon=10^{-3}$ for VP sampling.

### Predictor–Corrector (PC) sampling

This is the practical contribution. A generic SDE solver only knows the equation. But *we also have the score itself*, which means we can run [[Markov Chain Monte Carlo|MCMC]] to fix up any drift in the distribution. So alternate:

- **Predictor**: one step of a numerical reverse-SDE solver — moves you from time $t_{i+1}$ to $t_i$.
- **Corrector**: $M$ steps of Langevin MCMC *at fixed $t_i$* — $\mathbf{x} \leftarrow \mathbf{x} + \epsilon\,\mathbf{s}_{\theta}(\mathbf{x},t_i) + \sqrt{2\epsilon}\,\mathbf{z}$ — nudging the sample back onto $p_{t_i}$.

The step size is set adaptively from a "signal-to-noise ratio" $r$: $\epsilon = 2(r\lVert\mathbf{z}\rVert_2/\lVert\mathbf{g}\rVert_2)^2$, with $r\approx0.16$ for VE and $r\approx0.01$ for VP on CIFAR-10.

This framing shows the old samplers were degenerate cases: SMLD's annealed Langevin is *corrector only* (identity predictor); DDPM's ancestral sampling is *predictor only* (identity corrector).

They also define a **reverse diffusion predictor**: discretise the reverse SDE using exactly the same scheme as the forward one. It is mechanical to derive for any new SDE, unlike ancestral sampling.

### The probability flow ODE

For any SDE there is a deterministic ODE with identical marginals at every $t$:

$$\mathrm{d}\mathbf{x} = \Big[\mathbf{f}(\mathbf{x},t) - \tfrac{1}{2}g(t)^2\nabla_\mathbf{x}\log p_t(\mathbf{x})\Big]\mathrm{d}t$$

Note the $\tfrac12$ instead of $1$, and no noise term. The derivation is a rewrite of the Fokker–Planck equation: you fold half the diffusion term into the drift, leaving a Liouville equation whose "SDE" has zero noise.

Consequences:
- **Exact likelihood**: $\log p_0(\mathbf{x}(0)) = \log p_T(\mathbf{x}(T)) + \int_0^T \nabla\cdot\tilde{\mathbf{f}}_\theta(\mathbf{x}(t),t)\,\mathrm{d}t$. The divergence is estimated unbiasedly with the Skilling–Hutchinson trick, $\nabla\cdot\tilde{\mathbf{f}} = \mathbb{E}_\epsilon[\epsilon^\mathsf{T}\nabla\tilde{\mathbf{f}}\epsilon]$, which is one [[Vector Jacobian Product|vector–Jacobian product]] via [[Pytorch Autograd|autograd]] — same cost as a forward pass.
- **Adaptive sampling**: they use `scipy.integrate.solve_ivp` RK45 with `atol=rtol=1e-5`. Loosening tolerance cuts function evaluations by **>90%** with no visible quality loss.
- **Uniquely identifiable encoding**: because the forward SDE has *no parameters*, two independently trained models (4 vs 8 residual blocks per resolution) map the same CIFAR-10 image to almost the same latent code, dimension by dimension. Not true of normalizing flows or [[Auto-Encoding Variational Bayes (VAE)|VAEs]].

### Architectures (NCSN++ / DDPM++)

Starting from the DDPM UNet, they swept five changes: FIR anti-aliased up/downsampling (StyleGAN-2), rescaling skip connections by $1/\sqrt{2}$, BigGAN-style [[Deep Residual Learning for Image Recognition (ResNet)|residual blocks]], 4 instead of 2 blocks per resolution, and progressive-growing input/output paths.

**NCSN++** (best for VE) uses all of them with "residual" input and no progressive output. **DDPM++** (best for VP) is the same *minus* FIR and minus progressive growing. To move to continuous $t$, they swap the discrete positional embedding for random Fourier feature embeddings (scale 16).

EMA rate is not shared: **0.999 for VE, 0.9999 for VP**. Getting this backwards hurts.

## Ablation Studies and Experiments

### Samplers (Table 1, CIFAR-10 FID, models trained with the original discrete losses)

| Predictor | VE P1000 | VE PC1000 | VP P1000 | VP PC1000 |
|---|---|---|---|---|
| ancestral | 4.98 | 3.62 | 3.24 | 3.21 |
| reverse diffusion | 4.79 | **3.60** | 3.21 | **3.18** |
| probability flow | 15.41 | 3.51 | 3.51 | 3.06 |
| corrector-only (C2000) | — | 20.43 | — | 19.06 |

What this says, and it is the most useful table in the paper:

- **Corrector-only is terrible.** 20.43 FID for VE, 19.06 for VP, with 2000 steps. Pure Langevin without a solver does not mix in reasonable time.
- **Reverse diffusion beats ancestral sampling everywhere**, but only slightly (4.79 vs 4.98; 3.21 vs 3.24). Its real value is that it is easy to derive for a new SDE.
- **Adding one corrector step per predictor step always helps.** PC1000 (2000 total evaluations) beats P2000 (also 2000 evaluations) in every column. So spending compute on MCMC correction is better than spending it on finer time discretisation.
- **Probability flow alone is bad but corrects beautifully.** 15.41 → 3.51 for VE; 3.51 → 3.06 for VP. The ODE has no noise to escape errors, so it needs the corrector most. With a corrector, it becomes the *best* VP sampler.
- The VE gains from PC are much larger than the VP gains (4.98→3.62 vs 3.24→3.21). VE is the one that needed fixing.

### Likelihood (Table 2, bits/dim on uniformly dequantised CIFAR-10, lower better)

| Model | NLL | FID (ODE) |
|---|---|---|
| Glow | 3.35 | — |
| Flow++ | 3.29 | — |
| DDPM ($L_\text{simple}$) ELBO | ≤ 3.75 | 3.17 |
| Same DDPM, exact via ODE | **3.28** | 3.37 |
| DDPM cont. (VP) | 3.21 | 3.69 |
| DDPM cont. (sub-VP) | 3.05 | 3.56 |
| DDPM++ cont. (deep, VP) | 3.13 | 3.08 |
| DDPM++ cont. (deep, sub-VP) | **2.99** | 2.92 |

Three things stack: (i) exact likelihood beats the ELBO by a lot on the *identical weights* — 3.28 vs ≤3.75, so DDPM was always better than it could prove; (ii) the continuous objective beats the discrete one (3.28 → 3.21); (iii) sub-VP beats VP by ~0.1–0.15 bits/dim **at every architecture**. And 2.99 bits/dim is a record set without any maximum-likelihood training.

### Sample quality (Table 3)

NCSN++ (VE, discrete loss) → FID 2.45. Continuous loss → 2.38. Double depth → **2.20, IS 9.89**. That beats StyleGAN2-ADA conditional (2.42) without labels.

DDPM++ (VP): 2.78 → 2.55 (continuous) → 2.41 (deep). sub-VP: 2.61 → 2.41.

**The clean split**: VE gives better FID, VP/sub-VP give better likelihood. There is no single winner; the paper explicitly says practitioners should try both.

### What did not work

- **Equalized learning rate** (from ProgressiveGAN/StyleGAN) was *harmful* early on and abandoned.
- **Progressive growing** helped on average but no single configuration was consistently best — the authors could not identify a winner.
- **NCSN++ transferred to VP is not optimal.** It ranked 4th of 144 configurations. The VP winner drops FIR and progressive growing entirely. Architecture choices do not transfer across SDEs.
- **Probability flow ODE sampling on VE is much worse than on VP**, especially in high dimensions. If you want fast ODE sampling, pick VP.
- **Ancestral-sampling discretisation on continuously-trained models hurt FID** near $t\to0$ because the DDPM discretisation does not match the continuous variance there. They switched to plain Euler–Maruyama.
- **The final denoising step is not optional.** Without one last Tweedie's-formula denoise, FID degrades badly. This, not the model, is a big part of why SMLD historically scored worse than DDPM — DDPM had the denoise step and SMLD did not.

## Worth Remembering

- **Sampling is still slow.** The authors admit it: hundreds to thousands of network evaluations against one forward pass for a [[Generative Adversarial Networks|GAN]]. They flag "combine stable score-based training with GAN-speed sampling" as the open problem — which is exactly what consistency models and distillation later attacked.
- **Hyperparameter sprawl.** The SDE choice, $\epsilon$ clipping, predictor type, corrector count, signal-to-noise ratio $r$ — all matter and all need tuning. $r$ was grid-searched at 0.01 resolution per configuration.
- The framework generalises: with a matrix-valued, state-dependent $\mathbf{G}(\mathbf{x},t)$ the reverse SDE picks up a $\nabla\cdot[\mathbf{G}\mathbf{G}^\mathsf{T}]$ term. And if $\mathbf{f}$ is not affine, you lose the closed-form kernel — but you can fall back on sliced score matching, which never needs it.
- **Colorization trick worth stealing**: greyscale conditioning couples all three RGB channels, so imputation does not directly apply. They apply a fixed orthogonal $3\times3$ matrix to decouple luminance into its own channel, impute in that space, then rotate back. Orthogonality means Brownian motion stays Brownian motion, so nothing else changes.
- The conditional score decomposition $\nabla\log p_t(\mathbf{x}\mid\mathbf{y}) = \nabla\log p_t(\mathbf{x}) + \nabla\log p_t(\mathbf{y}\mid\mathbf{x})$ needs a **noise-aware classifier** — trained on $\mathbf{x}(t)$ at all $t$, with a mixture of [[Cross Entropy|cross-entropy]] losses across time. A normal classifier will not do; its accuracy collapses as noise rises.
- Uniquely identifiable encoding is the quiet result here. Two different architectures agree dimension-by-dimension on the latent code. It suggests the latent space is a property of *the data*, not of the model — a very different situation from [[Auto-Encoding Variational Bayes (VAE)|VAE]] latents.
- Open question worth chasing: sub-VP wins on likelihood and VE wins on FID. Nobody in this paper explains *why*. That gap is where the later noise-schedule literature (EDM and friends) lives.

## Links

Related: [[Denoising Diffusion Probabilistic Models]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Generative Adversarial Networks]] · [[Markov Chain Monte Carlo]] · [[Hamiltonian Monte Carlo]] · [[Derivative]] · [[Vector Jacobian Product]] · [[Pytorch Autograd]] · [[KL Divergence]] · [[Cross Entropy]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Random variable]] · [[Markov Property]] · [[Fourier Series Decomposition]] · [[Mode Collapse]]

New topics worth writing: Langevin dynamics, Score matching (Hyvärinen / denoising / sliced), Fokker–Planck equation, Neural ODEs, Normalizing flows, Itô calculus and Brownian motion, Tweedie's formula, Skilling–Hutchinson trace estimator, FID and Inception Score, Classifier guidance, Euler–Maruyama and Runge–Kutta solvers, Exponential moving average of weights
