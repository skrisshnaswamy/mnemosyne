---
title: "VICReg"
authors: ["Adrien Bardes", "Jean Ponce", "Yann LeCun"]
year: 2021
arxiv: "2105.04906"
url: https://arxiv.org/abs/2105.04906
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, self-supervised, vision]
---
## The Core Idea

Self-supervised image learning usually works like this: take one photo, make two random crops/colour-jitters of it, push both through a network, and force the two output vectors to be close. This is the [[Bootstrap Your Own Latent (BYOL)#^collapse|collapse]] trap — the network can win instantly by outputting the same constant vector for every image. Every method before this one dodged collapse with some *indirect* trick: negative pairs ([[A Simple Framework for Contrastive Learning (SimCLR)|SimCLR]]), a memory bank ([[Momentum Contrast (MoCo)|MoCo]]), a momentum encoder (BYOL), a stop-gradient ([[Exploring Simple Siamese Representation Learning (SimSiam)|SimSiam]]), cluster balancing (SwAV), or a cross-correlation matrix pushed to identity ([[Barlow Twins]]).

VICReg's move is to stop being clever and just **write the anti-collapse condition directly into the loss**, as two explicit penalties on each branch's output, computed *separately* per branch:

1. **Variance** — every output dimension must have standard deviation of at least $\gamma = 1$ across the batch. If it shrinks, you pay. This kills the "all vectors identical" collapse.
2. **Covariance** — every *pair* of output dimensions must be uncorrelated across the batch. This kills *informational collapse*, where 8192 dimensions all encode the same one thing.

Plus the obvious third term, **invariance**: plain mean-squared distance between the two views' embeddings.

Why this matters more than the accuracy number (which just ties Barlow Twins): because the regulariser touches only **one branch at a time**, the two branches no longer need to be the same. Not the same weights, not the same architecture, not even the same *input modality*. Barlow Twins computes a cross-correlation *between* the branches, so it implicitly assumes both sides produce similarly-scaled statistics. VICReg does not. That is what unlocks image↔text and audio-waveform↔spectrogram joint embedding with one loss.

> [!NOTE] Informational collapse
> Not the trivial collapse where all outputs are the same vector. Here outputs *do* vary, but the dimensions are highly correlated, so a $d$-dimensional embedding carries far less than $d$ dimensions worth of information. Decorrelating the dimensions is a cheap proxy for maximising information content. ^informational-collapse

## The Methodology

**Architecture.** Two branches. Each is an *encoder* $f_\theta$ (ResNet-50, 2048-d output — this is what you keep) followed by an *expander* $h_\phi$ (3 fully-connected layers, all width 8192; BN + ReLU on the first two, linear third). The loss lives on the expander output $z$, not the representation $y$. The expander is thrown away after pretraining.

The expander does two jobs: (a) absorb the information by which the two views differ, so the encoder is not forced to be invariant to everything; (b) blow up the dimension non-linearly, so that killing *correlations* in the wide space actually kills *dependencies* in the narrow representation space.

**The loss.** For a batch $Z = [z_1,\dots,z_n]$ of $n$ vectors of dimension $d$, write $z^j$ for the $j$-th coordinate across the batch.

Variance, a hinge on the standard deviation:
$$v(Z) = \frac{1}{d}\sum_{j=1}^{d}\max\!\left(0,\ \gamma - \sqrt{\mathrm{Var}(z^j)+\epsilon}\right),\quad \gamma=1,\ \epsilon=10^{-4}$$

Using the **standard deviation and not the variance is load-bearing**. If you hinge on $\mathrm{Var}(x)$ directly, its gradient goes to zero exactly when $x$ approaches its mean — the collapse point becomes a flat region and the model falls into it. The square root keeps a strong gradient right where you need it.

Covariance, from Barlow Twins:
$$C(Z) = \frac{1}{n-1}\sum_{i=1}^{n}(z_i - \bar z)(z_i - \bar z)^{\top}, \qquad c(Z) = \frac{1}{d}\sum_{i\neq j}[C(Z)]_{i,j}^{2}$$

Invariance, plain MSE with **no $\ell_2$ normalisation**:
$$s(Z,Z') = \frac{1}{n}\sum_i \lVert z_i - z_i'\rVert_2^2$$

Total:
$$\ell(Z,Z') = \lambda\, s(Z,Z') + \mu\,[v(Z)+v(Z')] + \nu\,[c(Z)+c(Z')]$$

with $\lambda = \mu = 25$, $\nu = 1$ on ImageNet.

**Training.** ImageNet, 1000 epochs, LARS, batch size 2048, weight decay $10^{-6}$, base LR 0.2 scaled as $\text{batch}/256 \times 0.2$, cosine decay with 10 warmup epochs, final LR 0.002. Augmentations are the BYOL pipeline, symmetrised: random resized crop to 224, horizontal flip, colour jitter, grayscale $p{=}0.2$, Gaussian blur $p{=}0.5$, solarisation $p{=}0.1$.

Cost: 11h / 100 epochs on 32 V100s, 11.3 GB peak per GPU — same memory as Barlow Twins, less than BYOL's 14.6 GB.

## Ablation Studies and Experiments

**ImageNet linear probe (frozen ResNet-50, 1000 epochs):** VICReg 73.2% top-1 / 91.1% top-5. Exactly ties Barlow Twins (73.2 / 91.0). Below BYOL (74.3) and SwAV+multi-crop (75.3), above SimSiam (71.3), SwAV (71.8), SimCLR (69.3). Supervised is 76.5. Semi-supervised 1% / 10% labels: 54.8 / 69.5 top-1. Three random seeds spread <0.1% — very stable.

**Transfer (Table 2):** Places205 54.3, VOC07 86.6 mAP, iNat18 47.0 — all slightly better than Barlow Twins. But detection is *worse*: VOC07+12 82.4 AP50 and COCO 39.4 det / 36.4 seg, versus Barlow Twins 82.6 / 40.0 / 36.7 and SwAV 82.6 / 41.6 / 37.8. Classification gains, detection loses.

**Loss coefficient ablation (Table 7) — the money table:**

| Terms | $\lambda,\mu,\nu$ | Top-1 |
|---|---|---|
| Invariance only | 1, 0, 0 | collapse |
| Invariance + covariance | 25, 0, 1 | **collapse** |
| Invariance + variance | 1, 1, 0 | 57.5 |
| All three | 25, 25, 1 | 68.6 |

Read that second row carefully. **Covariance alone does not prevent collapse.** Decorrelation is meaningless if everything shrinks to zero — a constant vector has zero covariance too. Variance is what stops the shrink; covariance then does the work of spreading information across dimensions, and it is worth a further **+11.1 points** (57.5 → 68.6). Both are needed, and they do different things.

Also from Table 7: training is unstable if $\lambda \neq \mu$, or if $\nu > \mu$. $\lambda = \mu \gg \nu$ is the stable regime and the exact value barely matters (5,5,1 → 68.1; 50,50,1 → 68.3).

**Replacing the architectural tricks (Table 4, 100 epochs).** Starting from a bare symmetric encoder+expander with no tricks:

| Setup | No reg | +Var | +Var/Cov |
|---|---|---|---|
| SimSiam (SG+PR+BN) | 67.9 | 68.1 | 67.6 |
| SimSiam (SG+PR, no BN) | 35.1 | 67.3 | 67.1 |
| SimSiam (SG only) | collapse | 56.8 | 66.1 |
| VICReg (BN only) | collapse | 57.5 | 68.6 |
| BYOL (ME+SG+PR+BN) | 69.3 | 70.2 | 69.5 |

Three things fall out. (1) Once you have variance regularisation, adding a **predictor is redundant** — no significant change. (2) Without VR, both stop-gradient *and* predictor are required, and dropping BN craters SimSiam from 67.9 to 35.1, while VR rescues it to 67.3. So variance regularisation substitutes for the whole trick stack. (3) Adding VR to *unmodified* BYOL still gives +0.9%, and Appendix figures show BYOL/SimSiam representation std drifting down during training — i.e. **these methods are slowly collapsing, and VR halts it.** The BYOL gain shrinks to +0.2% at 1000 epochs, so it is mostly faster convergence, not a better ceiling.

**What did not work:**
- **Covariance regularisation combined with stop-gradient is hard to optimise.** In every SimSiam row, adding CR on top of VR makes things slightly *worse*. The authors show (Appendix D.8) that SG and CR are chasing the same objective — both drive the average correlation coefficient down — so they interfere. In BYOL the correlation does drop with CR, but accuracy still falls 0.7 points.
- **Normalisation hurts (Table 8).** Best config is std-normalising the expander *hidden* layers only: 68.6. Normalise the embedding too → 68.4. Drop hidden-layer BN → 67.4 (−1.2). $\ell_2$-normalising the embeddings onto the unit sphere (with $\gamma$ retargeted to $1/\sqrt d$) → **65.1, a 3.5-point loss.** Letting the covariance entries roam outside $[-1,1]$ apparently helps optimisation.
- Variance on $\mathrm{Var}$ instead of $\sqrt{\mathrm{Var}}$ → collapse (gradient vanishes at the collapse point).
- Regularising the *predictor* output instead of the expander output worked worse.

**Weight sharing (Table 5, 100 epochs).** This is the whole point of the paper:

| | Shared R50 | Unshared R50 | R50/R101 | R50/ViT-S |
|---|---|---|---|---|
| BYOL | 69.3 | ✗ | ✗ | ✗ |
| SimCLR | 64.4 | 63.1 | 63.9 | 63.5 |
| Barlow Twins | 68.7 | 64.2 | 65.3 | 63.9 |
| VICReg | 68.6 | **66.5** | **68.1** | **66.2** |

Barlow Twins drops 4.5 points when weights are untied; VICReg drops 2.1. With a [[An Image is Worth 16x16 Words (ViT)|ViT-S]] on one side and a ResNet on the other, VICReg beats Barlow Twins by 2.3. BYOL cannot run these at all.

**Multi-modal, MS-COCO 5K retrieval** (ResNet-152 for images, word-embedding + GRU for text, different $\nu$ per branch — which Barlow Twins structurally cannot do). Image-to-text R@1: VSE++ contrastive 30.3, Barlow Twins 31.4, **VICReg 33.6**. Text-to-image R@1: 41.3 / 42.9 / **45.2**.

**Audio, ESC-50** (1-D ResNet-18 on raw waveform ↔ 2-D ResNet-18 on mel spectrogram): supervised baseline 72.7, Barlow Twins 75.4, **VICReg 78.4**.

**Scale knobs.** Expander width matters a lot: 256-d → 55.9, 2048 → 65.1, 8192 → 68.6, 16384 → 68.8 (saturating). Batch size is not very sensitive: 128 → 67.3, 2048 → 68.6, 4096 → 67.8 — no negatives means no large-batch requirement. Representation dimension matters more than parameter count: Narrow-ResNet-50(×4) with a 2048-d representation gets 76.0, but full ResNet-50(×4) with 8192-d representation gets only 75.6, saturating vs ×2's 75.5.

## Worth Remembering

- **VICReg does not beat Barlow Twins on ImageNet. It ties it exactly.** If you only care about a single-modality ImageNet number, this paper buys you nothing. Buy it for modularity: independent branches, no shared weights, no predictor, no momentum encoder, no stop-gradient, no memory bank, no negatives, no batch-size floor, no normalisation of the output.
- The whole paper is an argument that BYOL's and SimSiam's mystery is not really a mystery — the tricks are *implicitly* doing variance preservation and decorrelation, just badly and slowly. Appendix D.8 shows this empirically: BYOL/SimSiam embeddings land at std exactly $1/\sqrt d$ (spread on the unit sphere), and their average correlation coefficient falls during training even with no explicit term. If you already run BYOL, adding the variance hinge with $\mu=1$ is a nearly free stability patch.
- Detection/segmentation transfer lags Barlow Twins and SwAV by 0.3–2.2 AP. Whatever the covariance-on-each-branch formulation gains in modularity, it does not gain in spatially-localised features.
- The $\gamma = 1$ target is arbitrary in absolute terms — it only matters relative to the scale the invariance term wants. This is why $\lambda = \mu$ is required for stability; you are balancing a "push apart to std 1" force against a "pull together" force.
- Practical caveat: batch statistics are computed **per GPU** in the reference pseudocode. With small per-device batches the variance and covariance estimates get noisy, and the covariance term is a $d \times d$ matrix at $d = 8192$ — that is a 67M-entry matmul per branch per step. The expander, not the encoder, is where the memory goes.
- Open question the paper only gestures at (Appendix D): *why* does decorrelating the 8192-d embedding decorrelate the 2048-d representation upstream? They call it "a non-trivial phenomenon" and measure it, but do not explain it. The direct descendant [[LeJEPA- Provable and Scalable Self-Supervised Learning|LeJEPA]] later attacks exactly this by specifying the target distribution rather than just its second moments.

## Links

Related: [[Barlow Twins]] · [[Bootstrap Your Own Latent (BYOL)]] · [[Exploring Simple Siamese Representation Learning (SimSiam)]] · [[A Simple Framework for Contrastive Learning (SimCLR)]] · [[Momentum Contrast (MoCo)]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Understanding Contrastive Learning through Alignment and Uniformity]] · [[LeJEPA- Provable and Scalable Self-Supervised Learning]] · [[Self-Supervised Learning from Images with I-JEPA]] · [[Whitening Sentence Representations]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[Batch Normalization]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Regularization]] · [[Representation Degeneration Problem in Training NLMs]]

New topics worth writing: LARS optimizer and layer-wise learning-rate scaling, W-MSE and Karhunen–Loève whitening for SSL, SwAV and Sinkhorn–Knopp cluster balancing, hinge loss, covariance matrix estimation from minibatches, VSE++ and image–text retrieval, ESC-50 audio classification, Mask R-CNN / Faster R-CNN transfer protocols
