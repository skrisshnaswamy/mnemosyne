---
title: "Classifier-Free Diffusion Guidance"
authors: ["Jonathan Ho", "Tim Salimans"]
year: 2022
arxiv: "2207.12598"
url: https://arxiv.org/abs/2207.12598
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, diffusion, vision]
---
## The Core Idea

Diffusion models generate images by starting from pure noise and repeatedly denoising. When you condition on a class label ("malamute"), the raw samples are diverse but often mushy — many of them do not look strongly like the class. Other generative families have a knob for this: [[Generative Adversarial Networks|BigGAN]] has "truncation", flows have "low temperature sampling". Turn the knob and you get fewer, better, more stereotyped samples. Diffusion had no such knob. The obvious hacks (scale the model's score, shrink the noise added at each reverse step) just produce blurry garbage.

Dhariwal & Nichol's fix was **classifier guidance**: train a separate image classifier on *noisy* images, and at each sampling step push the image in the direction that makes the classifier more confident. It works, but it is ugly. You need a second model, and it must be trained on noise, so you cannot drop in a pretrained ResNet. Worse, the procedure is literally a gradient-based adversarial attack on a classifier — and the headline metrics (Inception Score, FID) are *themselves* computed with a classifier. So the whole thing smelled like metric gaming.

The trick here: **you do not need a classifier, because the diffusion model already contains one.** By Bayes' rule,

$$p(\mathbf{c}\mid\mathbf{z}) \propto \frac{p(\mathbf{z}\mid\mathbf{c})}{p(\mathbf{z})}$$

Take logs and gradients, and the "classifier gradient" becomes the *difference* between the conditional score and the unconditional score. Both are things a diffusion model can predict. So: train one network that can run either with the label or with a blank label, and at sampling time extrapolate away from the blank prediction and past the conditional one:

$$\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{z}_\lambda,\mathbf{c}) = (1+w)\,\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda,\mathbf{c}) - w\,\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda)$$

That is the whole paper. One line at training (randomly drop the label), one line at sampling. It unlocked every text-to-image model you have used — Stable Diffusion, Imagen, DALL·E 2 all run CFG, usually at $w \approx 6$–$8$ in the convention where the scale is $1+w$.

> [!NOTE] Classifier-free guidance ^classifier-free-guidance
> Sample with a linear extrapolation between the label-conditioned score and the label-free score. $w=0$ is the ordinary conditional model; larger $w$ pushes further from the unconditional prediction, raising fidelity and killing diversity.

The intuition the authors land on: guidance **lowers the unconditional likelihood while raising the conditional one**. The negative score term is the novel bit — you are actively steering *away* from "images in general" and toward "images that are unusually specific to this label".

## The Methodology

**The diffusion setup (continuous time, log-SNR parameterisation).** Instead of indexing time by $t$, they index by $\lambda = \log(\alpha_\lambda^2/\sigma_\lambda^2)$, the log signal-to-noise ratio. Forward corruption is

$$q(\mathbf{z}_\lambda\mid\mathbf{x}) = \mathcal{N}(\alpha_\lambda\mathbf{x},\,\sigma_\lambda^2\mathbf{I}),\qquad \alpha_\lambda^2 = \frac{1}{1+e^{-\lambda}},\ \ \sigma_\lambda^2 = 1-\alpha_\lambda^2$$

so $\lambda$ runs down as noise goes up. The model predicts the noise, $\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda,\mathbf{c})$, and the clean image is recovered as $\mathbf{x}_\theta = (\mathbf{z}_\lambda - \sigma_\lambda\boldsymbol{\epsilon}_\theta)/\alpha_\lambda$. Loss is plain [[Denoising Diffusion Probabilistic Models|DDPM]] noise-prediction:

$$\mathbb{E}_{\boldsymbol{\epsilon},\lambda}\!\left[\|\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda,\mathbf{c}) - \boldsymbol{\epsilon}\|_2^2\right]$$

This is denoising score matching at every noise level, so $\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda) \approx -\sigma_\lambda\nabla_{\mathbf{z}_\lambda}\log p(\mathbf{z}_\lambda)$ — see [[Score-Based Generative Modeling through SDEs]].

$\lambda$ is sampled as $\lambda = -2\log\tan(au+b)$ with $u\sim\mathrm{Unif}[0,1]$ — a truncated hyperbolic-secant distribution, the continuous version of the cosine schedule. Endpoints $\lambda_{\min}=-20$, $\lambda_{\max}=20$.

**Training (Algorithm 1).** Identical to a normal conditional diffusion model, plus one line: with probability $p_{\mathrm{uncond}}$, replace the class label $\mathbf{c}$ with a null token $\varnothing$. So the *same weights, same network* learn both $\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda,\mathbf{c})$ and $\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda) := \boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda,\varnothing)$. No extra parameters. They note you *could* train two separate models; joint training is chosen purely for simplicity.

**Sampling (Algorithm 2).** At each step: two forward passes, one with $\mathbf{c}$ and one with $\varnothing$; combine with Eq. 6; convert to an $\tilde{\mathbf{x}}_t$ estimate; take an ancestral step. The reverse-step variance is a log-space interpolation $(\tilde\sigma^2_{\lambda'|\lambda})^{1-v}(\sigma^2_{\lambda|\lambda'})^{v}$ with $v$ a fixed constant, not learned — $v=0.3$ for the 64×64 models, $v=0.2$ for 128×128.

**Data and scale.** Class-conditional ImageNet, area-downsampled to 64×64 and 128×128. They reuse ADM's architecture and hyperparameters exactly (tuned for classifier guidance, so possibly suboptimal here). 64×64 trained 400k steps; 128×128 trained 2.7M steps. FID/IS computed on 50,000 samples.

**A subtlety worth keeping.** Eq. 6 is *inspired by* an implicit classifier but is not the gradient of one. The networks are unconstrained, so $\boldsymbol{\epsilon}_\theta$ is not a conservative vector field — there is generally no scalar potential whose gradient it is. So the guided step direction is genuinely not a classifier gradient, which is exactly what defuses the "you are just adversarially attacking Inception" objection.

## Ablation Studies and Experiments

**Sweeping $w$ (the main result).** $w \in \{0, 0.1, \ldots, 4\}$. FID and IS trade off monotonically, exactly like BigGAN truncation.

ImageNet 64×64, $p_{\mathrm{uncond}}=0.1$:

| $w$ | FID ↓ | IS ↑ |
|---|---|---|
| 0.0 | 1.80 | 53.7 |
| **0.1** | **1.55** | 66.1 |
| 0.3 | 3.03 | 92.8 |
| 1.0 | 12.60 | 170.1 |
| 4.0 | 26.22 | **260.2** |

Baselines: ADM 2.07 FID; CDM 1.48 FID / 67.95 IS.

ImageNet 128×128, $T=256$:

| $w$ | FID ↓ | IS ↑ |
|---|---|---|
| 0.0 | 7.27 | 82.5 |
| **0.3** | **2.43** | 158.5 |
| 1.0 | 7.86 | 298.0 |
| 4.0 | 21.53 | 421.0 |

This beats ADM-G (2.97 FID), CDM (3.52), LOGAN (3.36), and BigGAN-deep (5.7 FID / 124.5 IS). At $w=4$, CFG beats BigGAN-deep at its max-IS truncation on *both* axes (21.5 vs 25 FID, 421 vs 253 IS).

Note the shape: **best FID needs only a whisper of guidance** ($w=0.1$–$0.3$), and FID gets much worse fast after that. IS keeps climbing. If you optimise for IS you are wrecking FID.

**Unconditional training probability $p_{\mathrm{uncond}}$.** Tested $\{0.1, 0.2, 0.5\}$ on 64×64. **$0.5$ is consistently worse across the entire frontier**; $0.1$ and $0.2$ are indistinguishable. Reading: you only need to spend a small slice of model capacity on the unconditional task to get a useful negative score. This echoes Dhariwal & Nichol finding that a *small* classifier suffices for classifier guidance.

**Number of sampling steps $T$.** $T \in \{128, 256, 1024\}$ on 128×128. $T=256$ and $T=1024$ are essentially identical (2.43 FID each at $w=0.3$); $T=128$ is worse (3.04). $T=256$ is the sweet spot. **But the honest comparison stings**: ADM-G uses ~256 steps with *one* diffusion forward pass each, while CFG needs *two* per step. The compute-matched comparison is CFG at $T=128$, which at 3.04 FID **loses** to ADM-G's 2.97. The authors say this plainly.

**What did not work / what the paper does not claim.** Naive truncation analogues for diffusion (scaling scores, shrinking reverse-process noise) fail — that is the motivating negative result, from prior work. High $p_{\mathrm{uncond}}$ fails. There is no theoretical guarantee: they cite Grandvalet & Bengio (discriminative classifiers usually beat generative-model-derived implicit ones) and Grünwald & Langford (Bayes-rule classifiers can be *inconsistent* under model misspecification). So the implicit classifier "should" be bad, and the fact that it works is empirical only.

## Worth Remembering

- **Saturated colours.** At high $w$ the samples come out visibly over-saturated (Fig. 3, 8). This is a real artefact people later fixed with dynamic thresholding (Imagen) and other rescaling tricks. If your CFG samples look like a bad Instagram filter, this is why.
- **Two forward passes per step.** The one real cost. The authors suggest, but do not try, injecting conditioning *late* in the network so the shared trunk runs once and only the head runs twice.
- **You can skip the unconditional model if the label space is tiny.** Since $\sum_\mathbf{c} p(\mathbf{x}\mid\mathbf{c})p(\mathbf{c}) = p(\mathbf{x})$, you can marginalise conditional scores to get the unconditional one — at the cost of one pass per class. Useless for text conditioning.
- **The diversity cost is a deployment problem, not just a metric.** The authors flag it: guidance actively suppresses underrepresented parts of the data distribution. This is [[Mode Collapse]] by choice rather than by accident, and it is the mechanism behind a lot of "all faces from this model look the same" complaints.
- **Convention warning.** This paper uses $\tilde\epsilon = (1+w)\epsilon_c - w\epsilon_u$, so $w=0$ means no guidance. Most codebases use $s = 1+w$, so their "guidance scale 7.5" is this paper's $w=6.5$. Do not confuse them.
- The "negative score term" idea — steer *away* from a distribution — generalises far beyond class labels. Negative prompts, safety guidance, and composable diffusion all fall out of the same move.
- Open question the paper leaves: can you get the fidelity gain without paying in diversity? Nobody has really answered this.

## Links
Related: [[Denoising Diffusion Probabilistic Models]] · [[Score-Based Generative Modeling through SDEs]] · [[Generative Adversarial Networks]] · [[Mode Collapse]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[A Tutorial on Energy-Based Learning]] · [[KL Divergence]] · [[Derivative#Gradient|gradient]]

New topics worth writing: Classifier guidance (ADM / Diffusion Models Beat GANs), Fréchet Inception Distance, Inception Score, Denoising score matching, Truncation trick in BigGAN, Dynamic thresholding and guidance artefacts, Latent Diffusion / Stable Diffusion, Negative prompting, Conservative vector fields and score parameterisation
