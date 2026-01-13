---
title: "A Simple Framework for Contrastive Learning (SimCLR)"
authors: ["Ting Chen", "Simon Kornblith", "Mohammad Norouzi", "Geoffrey Hinton"]
year: 2020
arxiv: "2002.05709"
url: https://arxiv.org/abs/2002.05709
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, self-supervised, vision]
---
## The Core Idea

Train an image encoder with no labels by playing one game: take a photo, make two random distorted copies of it, and force the network to pick out "these two came from the same photo" among all the other photos in the batch. That is all. No memory bank of stored feature vectors, no hand-designed pretext puzzle (jigsaw, rotation, colourisation), no surgery on the network to restrict receptive fields.

The interesting claim is not that contrastive learning works — [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)|CPC]] and others already showed that. It is that **every complicated part of previous methods was replaceable by something simpler, and the wins came from three unglamorous knobs**:

1. **The augmentation policy defines the task.** Random crop + strong colour distortion is what makes the problem hard enough to be useful. Crop alone is not enough, because two crops of the same photo share a colour histogram — the network can cheat by matching colours. That is a textbook [[Shortcut Learning in Deep Neural Networks|shortcut]].
2. **Throw away the last two layers after training.** Put a small MLP "projection head" between the encoder and the loss, train through it, then delete it and use the layer *before* it. That single change is worth >10% top-1.
3. **Big batches and long training.** The batch *is* the negative set. Batch 8192 gives 16,382 negatives per positive, for free.

Result: a linear classifier on frozen SimCLR features hits **76.5% top-1 on ImageNet** (ResNet-50 4×), matching a supervised ResNet-50. Fine-tuned on **1% of labels**, 85.8% top-5 — better than AlexNet, which used 100× more labels.

Why it did not exist before: people assumed the hard part was architecture or the negative-sampling machinery. Chen et al. showed the hard part was augmentation design and where you attach the loss.

> [!NOTE] Contrastive learning
> Learn a representation by pulling matched pairs (two views of the same thing) together in embedding space and pushing unmatched pairs apart. No labels needed — the "label" is just "which source image did you come from". ^contrastive-learning

## The Methodology

Four pieces, in order.

**1. Stochastic augmentation $\mathcal{T}$.** For each image $\bm{x}$, draw two independent transforms $t, t' \sim \mathcal{T}$ giving views $\tilde{\bm{x}}_i = t(\bm{x})$, $\tilde{\bm{x}}_j = t'(\bm{x})$. Default policy is exactly three things:
- Inception-style random crop (area uniform in $[0.08, 1.0]$, aspect ratio $3/4$ to $4/3$), resized to 224×224, plus 50% horizontal flip.
- Colour distortion: brightness / contrast / saturation jitter with max delta $0.8s$, hue with $0.2s$, applied with prob 0.8; random grayscale with prob 0.2. Strength $s=1$ by default.
- Gaussian blur, applied 50% of the time, $\sigma \sim U[0.1, 2.0]$, kernel = 10% of image height.

**2. Base encoder $f$.** A plain [[Deep Residual Learning for Image Recognition (ResNet)|ResNet-50]]. $\bm{h}_i = f(\tilde{\bm{x}}_i)$ is the 2048-d output after global average pooling. This is the thing you keep.

**3. Projection head $g$.** A 2-layer MLP, $\bm{z}_i = W^{(2)}\sigma(W^{(1)}\bm{h}_i)$ with ReLU, output 128-d. This is the thing you throw away.

**4. NT-Xent loss.** For a minibatch of $N$ images you get $2N$ views. Every view is an anchor; its partner is the one positive, and the other $2(N-1)$ views are negatives. With $\mathrm{sim}(\bm{u},\bm{v}) = \bm{u}^\top\bm{v}/(\|\bm{u}\|\|\bm{v}\|)$ (cosine similarity):

$$\ell_{i,j} = -\log \frac{\exp(\mathrm{sim}(\bm{z}_i,\bm{z}_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]}\exp(\mathrm{sim}(\bm{z}_i,\bm{z}_k)/\tau)}$$

$$\mathcal{L} = \frac{1}{2N}\sum_{k=1}^{N}\big[\ell(2k-1, 2k) + \ell(2k, 2k-1)\big]$$

This is just a $2N$-way [[Cross Entropy|softmax cross-entropy]] where the correct class is "my other view". The $\tau$ is a [[Distilling the Knowledge in a Neural Network#^softmax-temperature|softmax temperature]]. Negatives are never sampled explicitly — the batch supplies them.

> [!NOTE] NT-Xent
> Normalised Temperature-scaled cross entropy. $\ell_2$-normalise the embeddings so similarity lives in $[-1,1]$, divide by $\tau$, then do softmax cross-entropy against the in-batch positive. Same shape as [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)#^infonce|InfoNCE]]. ^nt-xent

**Training setup.** Batch size 4096, 100 epochs by default (1000 epochs for headline numbers). LARS optimiser — plain SGD/[[Momentum|momentum]] with linear LR scaling goes unstable at these batch sizes. LR $= 0.3 \times \text{BatchSize}/256 = 4.8$, weight decay $10^{-6}$, 10 epochs linear warmup then cosine decay. 32–128 TPU cores.

**Global BN — a real trap.** [[Batch Normalization|BatchNorm]] statistics are normally computed per device. Positive pairs land on the same device, so the model can leak information through the local batch statistics and solve the task without learning anything. Fix: aggregate BN mean/variance across all devices. (MoCo shuffles examples across devices instead; CPC v2 uses [[Layer Normalization|LayerNorm]].)

**Evaluation.** Linear probe: freeze $f$, train a logistic regression on $\bm{h}$, report ImageNet top-1.

## Ablation Studies and Experiments

**Augmentations (Fig. 5).** They apply the target transform to *one* branch only (identity on the other) to remove the crop confound. Finding: **no single augmentation works**, even though the network solves the contrastive task nearly perfectly with one. The pairing that matters is **crop + colour distortion**. Figure 6 shows why — pixel-intensity histograms alone separate different images, so crop-only lets the net match on colour statistics.

**Augmentation strength (Table 1)**, ResNet-50 linear eval / supervised top-1:

| colour strength | 1/8 | 1/4 | 1/2 | 1 | 1 (+Blur) | AutoAug |
|---|---|---|---|---|---|---|
| SimCLR | 59.6 | 61.0 | 62.6 | 63.2 | **64.5** | 61.1 |
| Supervised | 77.0 | 76.7 | 76.5 | 75.7 | 75.4 | **77.1** |

Two things did **not** work: AutoAugment — a policy found by search *on supervised learning* — is worse (61.1) than plain crop + strong colour (64.5). And stronger colour actively *hurts* supervised training (77.0 → 75.4). Contrastive learning wants harsher augmentation than supervised learning does.

**Projection head (Fig. 8, Table 3).**
- No head: baseline. Linear head: +~7%. Nonlinear MLP head: another **+3%** on top of linear, **>10%** over none.
- Output dimension of $\bm{z}$ barely matters once a head exists.
- $\bm{h}$ (before head) beats $\bm{z}$ (after head) by **>10%**, even when the head is nonlinear.

Why? $\bm{z}$ is trained to be invariant to the augmentations, so $g$ deliberately destroys colour and orientation information — useful downstream, useless for the loss. They tested this by training an MLP to predict which transform was applied:

| predict | chance | from $\bm{h}$ | from $g(\bm{h})$ |
|---|---|---|---|
| colour vs grayscale | 80 | 99.3 | 97.4 |
| rotation | 25 | 67.6 | 25.6 |
| orig vs corrupted | 50 | 99.5 | 59.6 |
| orig vs Sobel | 50 | 96.6 | 56.3 |

Rotation info drops to chance after the head. The head is an information sink. (Appendix: the linear-head matrix $W$ is approximately low-rank; t-SNE shows classes better separated in $\bm{h}$ than $\bm{z}$.)

**Loss function (Table 4), linear eval top-1**, all with $\ell_2$ norm and tuned hyperparameters:

| Margin | NT-Logistic | Margin (semi-hard) | NT-Logistic (sh) | NT-Xent |
|---|---|---|---|---|
| 50.9 | 51.6 | 57.5 | 57.9 | **63.9** |

Semi-hard negative mining helps the alternatives by ~6 points and they are still ~6 points behind. The gradient table explains it: NT-Xent's softmax denominator automatically weights each negative by $\exp(\bm{u}^\top\bm{v}^-/\tau)/Z(\bm{u})$ — hard negatives get large gradients for free. Margin and logistic losses weight all negatives equally, so you must mine by hand.

**Normalisation and temperature (Table 5).**

| $\ell_2$? | $\tau$ | contrastive acc. | top-1 |
|---|---|---|---|
| Yes | 0.05 | 90.5 | 59.7 |
| Yes | **0.1** | 87.8 | **64.4** |
| Yes | 0.5 | 68.2 | 60.7 |
| Yes | 1 | 59.1 | 58.0 |
| No | 10 | 91.7 | 57.2 |
| No | 100 | 92.1 | 57.0 |

The key inversion: **without $\ell_2$ normalisation the contrastive task accuracy is *higher* (92.1) but the representation is *worse* (57.0)**. Solving the pretext task well is not the objective. Do not use it as a model-selection signal.

**Batch size and epochs (Fig. 9).** At 100 epochs, batch 8192 beats batch 256 by ~7 points. At 800+ epochs the gap mostly closes. So large batches buy *convergence speed*, not a better ceiling. Appendix B.1: performance saturates around batch 8192 (they went to 32K), but training longer (to 3200 epochs) keeps helping. Also, with LARS, **square-root** LR scaling ($0.075\sqrt{\text{BatchSize}}$) beats linear scaling for small batches and short runs — e.g. batch 256 / 100 epochs: 57.5 → 62.8.

**Model size (Fig. 7).** Bigger nets help both, but the gap to supervised *shrinks* with size: self-supervised ResNet-50 is 6.8% behind supervised, ResNet-50 (4×) only 1.8% behind. Unsupervised learning benefits more from scale.

**Headline comparisons.**

| Method | Arch | Params (M) | Top-1 |
|---|---|---|---|
| MoCo | ResNet-50 | 24 | 60.6 |
| CPC v2 | ResNet-50 | 24 | 63.8 |
| **SimCLR** | ResNet-50 | 24 | **69.3** |
| CPC v2 | ResNet-161 | 305 | 71.5 |
| MoCo | ResNet-50 (4×) | 375 | 68.6 |
| **SimCLR** | ResNet-50 (4×) | 375 | **76.5** |

Semi-supervised (top-5): with 1% of labels SimCLR ResNet-50 (4×) gets 85.8 vs CPC v2's 77.9 and a supervised baseline's 48.4.

Transfer to 12 datasets, ResNet-50 (4×) fine-tuned: SimCLR beats the supervised baseline on 5, loses on 2 (Pets, Flowers), ties on 5. With the *narrow* ResNet-50 the supervised baseline wins almost everywhere — the parity only appears at 4× width.

## Worth Remembering

- **The negative set is the batch.** That is why this needs TPU pods. MoCo's momentum queue exists precisely to get many negatives on modest hardware; SimCLR's answer is "just buy a bigger batch". This is the practical caveat if you want to reproduce it.
- **Global BN is not an optional detail.** Per-device BN lets the model cheat through batch statistics. This class of leak generalises: any time positives share a normalisation group, you have a shortcut.
- **Pretext task accuracy ≠ representation quality.** Table 5 is the cleanest demonstration in the paper. Removing $\ell_2$ norm makes the contrastive task easier to solve and the features worse.
- **The projection head trick spread everywhere.** BYOL, MoCo v2, SimSiam, and [[SimCSE- Simple Contrastive Learning of Sentence Embeddings|SimCSE]] all adopted it. The mechanism — the head absorbs augmentation-invariance so $\bm{h}$ can keep information the loss would otherwise delete — is worth internalising.
- Authors' honest note: "almost all individual components of our framework have appeared in previous work... the superiority is not explained by any single design choice, but by their composition." This is a careful-ablation paper, not an invention paper.
- They also flag that it is unclear whether contrastive success comes from mutual-information maximisation or just the form of the loss (citing Tschannen et al. 2019). Later work on [[Understanding Contrastive Learning through Alignment and Uniformity|alignment and uniformity]] and [[Understanding Dimensional Collapse in Contrastive Learning|dimensional collapse]] picked this up.
- Supervised models get nothing from longer training or stronger augmentation (Table B.3: ResNet-50 at 90 / 500 / 1000 epochs = 76.5 / 76.2 / 75.8). The asymmetry between the two regimes is real, not an artefact of an under-tuned baseline.
- Appendix: adding Sobel filtering, equalize/solarize, and motion blur to the policy pushes ResNet-50 to 70.0 and fine-tuning on *full* ImageNet to 80.4% top-1 for the 4× model vs 78.4% from scratch. So the augmentation search was not exhausted.
- Open question worth chasing: SimCLR needs negatives; BYOL (a few months later) removed them entirely. What does the negative set actually contribute beyond preventing collapse?

## Links
Related: [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[Understanding Contrastive Learning through Alignment and Uniformity]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[SimCSE- Simple Contrastive Learning of Sentence Embeddings]] · [[LeJEPA- Provable and Scalable Self-Supervised Learning]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Batch Normalization]] · [[Cross Entropy]] · [[Distilling the Knowledge in a Neural Network]] · [[Shortcut Learning in Deep Neural Networks]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Layer Normalization]]

New topics worth writing: LARS optimiser and large-batch learning-rate scaling, MoCo and the momentum encoder queue, BYOL / negative-free self-supervised learning, linear probe evaluation protocol, cosine learning rate schedule with warmup, AutoAugment and learned augmentation policies
