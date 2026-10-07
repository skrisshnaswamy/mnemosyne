---
title: "Group Normalization"
authors: ["Wu & He"]
year: 2018
arxiv: "1803.08494"
url: https://arxiv.org/abs/1803.08494
priority: Good-To-Read
read_on: 2026-09-25
tags: [paper, vision]
---
## The Core Idea

[[Batch Normalization]] works by averaging over the batch. That is its power and its bug. If a channel's mean and variance are computed over $N \times H \times W$ values and $N$ is small, those statistics are noisy garbage. ResNet-50 on ImageNet goes from 23.6% error at 32 images per GPU to **34.7% at 2 images per GPU** — the model falls apart, and nothing about the model changed except how many samples the normaliser got to look at.

That matters because a lot of computer vision cannot afford big batches. Detection and segmentation run at 1–2 high-resolution images per GPU. Video models with 3D convolutions trade clip length against batch size. In practice people just *froze* BN during fine-tuning — turning it into a fixed linear layer $y = \frac{\gamma}{\sigma}(x-\mu)+\beta$ using statistics baked in from ImageNet — which means no normalisation happens at all during the task you actually care about.

Group Normalization removes the batch from the equation entirely. Split a layer's $C$ channels into $G$ groups (default $G=32$). For each sample, for each group, compute one mean and one variance over that group's channels and all spatial positions. Normalise. Done.

The statistics now depend only on **one sample's own activations**. Batch size 32, 8, 2, or 1 — the arithmetic is identical. Measured: GN scores 24.1 / 24.2 / 24.0 / 24.2 / 24.1 % error at batch sizes 32 / 16 / 8 / 4 / 2. A flat line.

> [!NOTE] Group Normalization
> Normalise each sample independently, using the mean and variance of a *group* of channels (plus the spatial dimensions). No batch dimension is touched, so the train-time and test-time computations are the same function. ^group-norm

The motivation given is that channels are not independent. Classical features like SIFT and HOG are built as groups — a histogram of orientations — and are normalised *per group*. A conv filter and its horizontally-flipped twin should have similar response distributions; those two channels belong together. Grouping is a middle ground between "all channels share statistics" ([[Layer Normalization]]) and "every channel is on its own" (Instance Norm).

What it unlocks: you can train detectors **from scratch** with no ImageNet pre-training, you can raise video clip length without the batch-size penalty eating the gain, and you stop having to freeze normalisation when you fine-tune.

## The Methodology

All four normalisers are the same two lines of code with a different index set. For feature $x_i$ indexed by $i=(i_N, i_C, i_H, i_W)$:

$$\hat{x}_i = \frac{1}{\sigma_i}(x_i - \mu_i), \qquad \mu_i = \frac{1}{m}\sum_{k \in \mathcal{S}_i} x_k, \qquad \sigma_i = \sqrt{\frac{1}{m}\sum_{k \in \mathcal{S}_i}(x_k - \mu_i)^2 + \epsilon}$$

Everything hinges on $\mathcal{S}_i$ — which pixels get averaged together:

| Method | $\mathcal{S}_i$ | Averages over |
|---|---|---|
| BN | $\{k \mid k_C = i_C\}$ | $(N,H,W)$ — same channel, whole batch |
| LN | $\{k \mid k_N = i_N\}$ | $(C,H,W)$ — one sample, all channels |
| IN | $\{k \mid k_N = i_N, k_C = i_C\}$ | $(H,W)$ — one sample, one channel |
| **GN** | $\{k \mid k_N = i_N, \lfloor \tfrac{k_C}{C/G}\rfloor = \lfloor \tfrac{i_C}{C/G}\rfloor\}$ | $(H,W)$ + a block of $C/G$ channels |

The floor-division condition just means "$i$ and $k$ sit in the same contiguous block of channels."

All of them then apply the same learned per-channel affine:

$$y_i = \gamma \hat{x}_i + \beta$$

GN is exactly LN when $G=1$, and exactly IN when $G=C$. It is a dial between them.

**The implementation is a reshape.** Take `[N, C, H, W]`, reshape to `[N, G, C//G, H, W]`, take moments over axes `[2,3,4]`, normalise, reshape back, apply $\gamma,\beta$. Six lines.

**ImageNet setup.** ResNet-50, 8 GPUs, [[Delving Deep into Rectifiers (He init, PReLU)|He init]] for all convs, $\gamma$ initialised to 1 except the last norm layer in each residual block where it is 0 (so a fresh [[Deep Residual Learning for Image Recognition (ResNet)|residual block]] starts as the identity). Weight decay 1e-4 applied to everything including $\gamma,\beta$. 100 epochs, LR ÷10 at 30/60/90. When shrinking batch size they applied the [[Accurate, Large Minibatch SGD- Training ImageNet in 1 Hour|linear scaling rule]]: LR $=0.1N/32$.

**COCO setup.** Mask R-CNN in Detectron, 1 image/GPU × 8 GPUs. GN replaces frozen-BN (written BN\*) in the backbone, box head and mask head. One detail that mattered a lot: **weight decay 0 on $\gamma,\beta$ during fine-tuning**, otherwise detection results suffer. They also swapped the FPN box head from `2fc` to `4conv1fc` so there were conv layers for GN to normalise.

**Kinetics setup.** ResNet-50 I3D, pre-trained from ImageNet, normalisation extended from $(H,W)$ to $(T,H,W)$.

## Ablation Studies and Experiments

**ImageNet, ResNet-50, batch 32 per GPU** (the regime where BN is strongest):

| BN | LN | IN | GN |
|---|---|---|---|
| **23.6** | 25.3 | 28.4 | 24.1 |

GN loses by 0.5%. But GN has *lower training error* than BN. So GN optimises at least as well; what it lacks is BN's accidental [[Regularization|regularisation]] — the noise from random batch composition. That noise is a free regulariser you give up.

**Sensitivity to batch size** (ResNet-50 val error %):

| images/GPU | 32 | 16 | 8 | 4 | 2 |
|---|---|---|---|---|---|
| BN | 23.6 | 23.7 | 24.8 | 27.3 | 34.7 |
| GN | 24.1 | 24.2 | 24.0 | 24.2 | 24.1 |
| gap | +0.5 | +0.5 | −0.8 | −3.1 | **−10.6** |

Note the curiosity: BN at batch 2 (34.7%) is *worse than IN* (28.4%). At batch 1 the batch noise vanishes and BN degenerates into IN. So batch 2 is the worst of both worlds — no useful population estimate, maximum noise.

**How many groups?** (ResNet-50, batch 32)

Fixed $G$: 64→24.6, **32→24.1**, 16→24.6, 8→24.4, 4→24.6, 2→24.7, **1 (=LN)→25.3**.
Fixed channels-per-group: 64→24.4, 32→24.5, **16→24.2**, 8→24.3, 4→24.8, 2→25.6, **1 (=IN)→28.4**.

Read this carefully. Anything from 2 to 64 groups works within ~0.6%. The method is not sensitive to $G$. What *is* sensitive is the extremes. Going all the way to one channel per group (IN) costs 4.2 points; even **2 channels per group already recovers most of it (25.6 vs 28.4)**. The channel-grouping is doing real work — it is not just "avoid the batch dimension".

**Batch Renorm** — the prior fix for small batches. Tuned to $r_{\max}=1.5, d_{\max}=0.5$. At batch 4: BR 26.3, BN 27.3, GN 24.2. BR helps but is still batch-dependent, so it still degrades.

**Deeper.** ResNet-101 at batch 32: BN 22.0, GN 22.4. At batch 2: BN 31.9, GN 23.0.

**Is normalisation needed at all?** VGG-16, which trains fine without it: none 29.2, BN 28.0, **GN 27.6**. Feature distributions at conv5_3 look qualitatively the same under BN and GN, and both look very different from no-norm. Here GN actually *beats* BN — consistent with the idea that VGG benefits less from BN's regularisation, so GN's better optimisation wins.

**COCO, Mask R-CNN R50-C4:** BN\* 37.7 box / 32.8 mask → GN 38.8 / 33.6. Note GN's pre-trained ImageNet model was *worse* (24.1 vs 23.6) and it still transfers better, because BN\* is frozen and therefore inconsistent between pre-training and fine-tuning.

**COCO, FPN — the decomposition ablation:**

| backbone | box head | AP$^{bbox}$ | AP$^{mask}$ |
|---|---|---|---|
| BN\* | — | 38.6 | 34.2 |
| BN\* | GN | 39.5 | 34.4 |
| GN | GN | 40.0 | 34.8 |

Most of the gain (+0.9) comes from normalising the *head*, not the backbone. The backbone adds another +0.5. Full results with longer training: R50 GN 40.8 / 36.1 (vs BN\* 38.6 / 34.5 — **+2.2 box, +1.6 mask**); R101 GN 42.3 / 37.2.

**Training from scratch, no ImageNet:** R50 GN 39.5 box / 35.2 mask; R101 GN 41.0 / 36.4. Concurrent work with *synchronised* BN got 34.5 box. The from-scratch GN numbers are competitive with ImageNet-pretrained ones.

**Kinetics, I3D:**

| clip / batch | 32 / 8 | 32 / 4 | 64 / 4 |
|---|---|---|---|
| BN | **73.3** / 90.7 | 72.1 / 90.0 | 73.3 / 90.8 |
| GN | 73.0 / 90.6 | **72.8** / 90.6 | **74.5** / 91.7 |

The 64-frame column is the one that matters. Under BN, doubling clip length looks like it bought *nothing* (73.3 → 73.3) — but only because halving the batch to fit it in memory cost about as much as the longer clip gained. Under GN, the longer clip gives +1.7 top-1. **BN was hiding a real architectural gain behind its own batch-size penalty.**

### What did not work

- **Fine-tuning BN unfrozen at batch 2** for detection: roughly **−6 AP**. Unusable; that is why everyone freezes it.
- **BN in the detection box head**: ~**9 AP worse**. The 512 RoIs per image are sampled from the *same* image, so they are not i.i.d. — the batch statistics are meaningless. GN, being per-sample, is immune.
- **LN for detection**: 1.9 box AP worse than GN, and 0.8 worse than even frozen BN. Avoiding the batch dimension is not sufficient; you need the grouping.
- **IN in visual recognition**: 28.4% ImageNet error, 4.8 behind BN. Per-channel-only statistics throw away channel dependence.
- **Synchronised BN** (the alternative fix, computing statistics across GPUs): the authors argue it converts an algorithm problem into a hardware problem, needs GPU count proportional to BN's appetite, and blocks asynchronous solvers.

## Worth Remembering

- **GN is not free at large batch.** You pay ~0.5% on ImageNet at batch 32. That gap is BN's regularisation, not BN's optimisation — GN's training error is lower. The authors explicitly flag "GN + a suitable regulariser" as unexplored future work.
- **The batch-2 anomaly.** BN at batch 2 (34.7%) is worse than IN (28.4%). Small-but-not-one batches are the pathological case: enough samples to be noisy, not enough to be accurate.
- **Memory is the real prize.** Removing the batch-size floor frees up ~16× or more memory per GPU, which lets you train higher-capacity models that were previously bottlenecked. This is an architecture-search unlock disguised as a normalisation paper.
- **$G=32$ is a fine default and you probably should not tune it.** The ablation spans 2–64 groups within 0.6%.
- **Practical caveat for fine-tuning:** set weight decay to 0 on $\gamma$ and $\beta$. The paper calls this "important for good detection results". Easy to miss.
- **Honest limitation the authors state:** most state-of-the-art systems and their hyper-parameters were designed *around* BN. GN-based models may need a fresh hyper-parameter search to show their true numbers. The COCO "long schedule" result is evidence — GN kept improving past Detectron's default 180k iterations (to 270k) while BN\* did not.
- **Connections.** GN's relatives — [[Layer Normalization]] and Instance Norm — are the ones that stuck in sequence models and [[Generative Adversarial Networks|GANs]]. The authors suggest GN could replace them there, untested. In practice history went a specific way: transformers kept LayerNorm/RMSNorm, diffusion U-Nets adopted GroupNorm almost universally, and GN remains the default in detection/segmentation stacks.
- **Follow-up question worth chasing:** GN removes batch dependence, but the paper never disentangles *why* BN helps in the first place. [[How Does Batch Normalization Help Optimization]] argues it is loss-landscape smoothing, not "internal covariate shift" — and GN smooths in a batch-free way, which is quiet corroboration.

## Links

Related: [[Batch Normalization]] · [[Layer Normalization]] · [[How Does Batch Normalization Help Optimization]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Accurate, Large Minibatch SGD- Training ImageNet in 1 Hour]] · [[Regularization]] · [[Delving Deep into Rectifiers (He init, PReLU)]] · [[Distributed Training]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Fine-Tuning]] · [[U-Net]] · Training Dynamics

New topics worth writing: Instance Normalization, Batch Renormalization, Synchronized BatchNorm, RMSNorm, Mask R-CNN, Feature Pyramid Networks, I3D and 3D convolutions, Frozen-BN in transfer learning
