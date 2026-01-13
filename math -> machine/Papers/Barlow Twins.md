---
title: "Barlow Twins"
authors: ["Jure Zbontar", "Li Jing", "Ishan Misra", "Yann LeCun", "Stéphane Deny"]
year: 2021
arxiv: "2103.03230"
url: https://arxiv.org/abs/2103.03230
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, self-supervised, vision]
---
## The Core Idea

Self-supervised vision learners all share one trick: take an image, make two random distorted copies, push both through a network, and force the two output vectors to agree. The problem is that a network can cheat perfectly — output the same constant vector for every image. Agreement is 100%, information is zero. This is [[Bootstrap Your Own Latent (BYOL)#^collapse|collapse]].

Every method before this bought its way out of collapse with machinery:

- [[A Simple Framework for Contrastive Learning (SimCLR)|SimCLR]] pushes different images apart (negatives), which needs big batches.
- [[Momentum Contrast (MoCo)|MoCo]] keeps a queue of 60,000+ stored embeddings and a slow-moving copy of the encoder.
- [[Bootstrap Your Own Latent (BYOL)|BYOL]] and [[Exploring Simple Siamese Representation Learning (SimSiam)|SimSiam]] break the symmetry between the two branches with a predictor head and a [[Exploring Simple Siamese Representation Learning (SimSiam)#^stop-gradient|stop-gradient]] — and nobody at the time could say *why* that works.

Barlow Twins removes all of it. The insight: instead of stopping collapse with architecture, stop it with the **loss**, by looking at the wrong axis. Everyone else compares vectors *sample by sample*. Barlow Twins compares them *feature by feature*, across the batch.

Build the $D \times D$ cross-correlation matrix between the two branches' outputs. Element $(i,j)$ asks: "how correlated is feature $i$ of view A with feature $j$ of view B, over the batch?" Push that matrix towards the identity. The diagonal going to 1 means each feature survives distortion (invariance). The off-diagonal going to 0 means no two features say the same thing (redundancy reduction).

A collapsed constant output has zero variance, so its correlation matrix is undefined/degenerate — it can never be the identity. Collapse is ruled out **by construction**, not by a training trick.

> [!NOTE] Redundancy reduction principle ^redundancy-reduction
> H. Barlow's 1961 neuroscience hypothesis: the point of sensory processing is to recode redundant input into a *factorial code* — one where the components are statistically independent. The retina and cortex look like they do this. Barlow Twins is that principle written as a loss.

What it unlocks: works at batch size 256; no momentum encoder; no predictor; no negatives; and — the strange part — it keeps getting *better* as you widen the embedding to 16,384 dimensions, where SimCLR and BYOL saturate around 256.

## The Methodology

**The loss.** Let $Z^A, Z^B$ be the two $N \times D$ embedding batches, each column normalised to zero mean and unit standard deviation along the batch axis. Then

$$\mathcal{C}_{ij} \triangleq \frac{\sum_b z^A_{b,i} z^B_{b,j}}{\sqrt{\sum_b (z^A_{b,i})^2}\sqrt{\sum_b (z^B_{b,j})^2}}$$

$$\mathcal{L}_{BT} \triangleq \underbrace{\sum_i (1 - \mathcal{C}_{ii})^2}_{\text{invariance}} + \lambda \underbrace{\sum_i \sum_{j \neq i} \mathcal{C}_{ij}^2}_{\text{redundancy reduction}}$$

$b$ indexes samples in the batch, $i,j$ index feature dimensions. $\mathcal{C}$ is $D \times D$, entries in $[-1, 1]$. That is the whole method. The pseudocode is about eight lines.

Note the axis swap versus [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)#^infonce|InfoNCE]]: InfoNCE normalises each *sample vector* to the unit sphere and takes inner products **over features**; Barlow Twins normalises each *feature column* and takes inner products **over the batch**.

**Architecture.**
- Encoder: ResNet-50, no classification head, 2048 outputs. This is the "representation" used for downstream tasks.
- Projector: 3 linear layers, each 8192 wide. First two followed by [[Batch Normalization|BatchNorm]] + ReLU. The 8192-d output is the "embedding" that goes into the loss.
- Note the bottleneck: the ResNet only emits 2048 dims, yet an 8192-d projector output helps. Nobody fully explains this.

**Augmentations.** Same pipeline as BYOL: random crop → resize to $224 \times 224$ → horizontal flip → colour jitter → grayscale → Gaussian blur → solarisation. Crop and resize always; the rest with probability, and blur/solarise use *different* probabilities for the two views (asymmetric augmentation, not asymmetric architecture).

**Optimisation.** LARS, 1000 epochs, batch size 2048 (but 256 works). LR 0.2 for weights, 0.0048 for biases and BN parameters, scaled by `batch_size/256`. 10-epoch warmup, then cosine decay by a factor of 1000. Weight decay $1.5 \times 10^{-6}$; biases and BN params excluded from both LARS adaptation and decay. $\lambda = 5 \times 10^{-3}$. 32 V100s, ~124 hours.

**The information-theory story (Appendix A).** Start from the Information Bottleneck objective: keep information about the sample, throw away information about the distortion.

$$\mathcal{IB}_\theta \triangleq I(Z_\theta, Y) - \beta I(Z_\theta, X)$$

Since $f_\theta$ is deterministic, $H(Z_\theta \mid Y) = 0$, and it rearranges to $H(Z_\theta \mid X) + \frac{1-\beta}{\beta} H(Z_\theta)$. Estimating the [[Cross Entropy|entropy]] of a high-dimensional signal from one batch is hopeless, so assume $Z$ is Gaussian — then entropy is $\log |\mathcal{C}_Z|$. Be honest about the gap: they say optimising the log-determinant directly did **not** reach state of the art, so they swapped in the Frobenius-norm-of-off-diagonals proxy, which shares the same global optimum. So the IB derivation is a motivation, not a proof.

This Gaussian parametrisation is exactly why it beats InfoNCE on batch size and dimension. InfoNCE's contrastive term is a *non-parametric* entropy estimate — it suffers the curse of dimensionality and needs many samples. A Gaussian assumption needs far fewer.

## Ablation Studies and Experiments

All ablations use a 300-epoch model; that baseline gets **71.4% top-1** on ImageNet linear eval (the full 1000-epoch model gets 73.2%).

**Headline numbers, ResNet-50, ImageNet linear eval (top-1):**

| Method | Top-1 |
|---|---|
| Supervised | 76.5 |
| SimCLR | 69.3 |
| MoCo v2 | 71.1 |
| SimSiam | 71.3 |
| **Barlow Twins** | **73.2** |
| BYOL | 74.3 |
| SwAV | 75.3 |

Not best. Competitive, with far less machinery.

**Semi-supervised, 1% of ImageNet labels:** 55.0% top-1 — beats BYOL (53.2) and SwAV (53.9). This is the one place it is genuinely state of the art. At 10% it is 69.7, slightly behind SwAV's 70.2.

**Transfer:** Places-205 54.1, VOC07 86.2, iNat18 46.5 — roughly BYOL-level, above SimCLR and MoCo-v2. Detection: VOC07+12 $AP_{all}$ 56.8, COCO $AP^{bb}$ 39.2 — indistinguishable from MoCo-v2/SimSiam.

**Loss ablations (Table 5) — this is the interesting table:**

| Loss variant | Top-1 |
|---|---|
| Baseline | 71.4 |
| Only invariance (diagonal) | 57.3 |
| Only redundancy reduction (off-diagonal) | **0.1** |
| Normalise along feature dim instead | 69.8 |
| No BN in projector MLP | 71.2 |
| No BN **and** covariance instead of correlation | 53.4 |
| Cross-entropy with temperature | 63.3 |

Read it carefully:

- Off-diagonal alone is 0.1% — total garbage, as expected. Decorrelated features with no invariance are just noise.
- Diagonal alone gives 57.3%, *not* 0.1%. Pure collapse would be near-chance. So even the plain agreement term learns something before falling apart — presumably the projector's BatchNorm props it up.
- Dropping BatchNorm from the projector costs almost nothing (71.2 vs 71.4). But dropping BatchNorm **and** un-normalising along the batch (covariance not correlation) drops to 53.4. So the batch-axis normalisation is the load-bearing part, not the BN layers. This matters: BYOL was suspected of relying on BatchNorm to avoid collapse; Barlow Twins doesn't.
- The cross-entropy-with-temperature version, $\mathcal{L} = -\log\sum_i \exp(\mathcal{C}_{ii}/\tau) + \lambda \log \sum_i \sum_{j\neq i} \exp(\max(\mathcal{C}_{ij},0)/\tau)$, gets 63.3. Making it look more like InfoNCE hurts by 8 points.

**Batch size.** Nearly flat from 256 to 2048; best is 71.7 at 1024. SimCLR drops ~4 points at batch 256. Confirms the loss is doing something structurally different from InfoNCE.

**Embedding dimension.** Monotonically improving up to 16,384, no saturation observed. SimCLR and BYOL flatten early. They stopped only because of memory.

**BYOL with a bigger head — the fair-comparison check.** They widened BYOL's projector/predictor to see if the dimension effect was universal. It is not:

| BYOL projector / predictor | Top-1 |
|---|---|
| 4096-256 / 4096-256 (baseline) | 74.1 |
| 4096-4096-256 / 4096-4096-256 | 73.2 |
| 8192-8192-8192 / 8192-8192 (same as BT) | **72.3** |

BYOL gets *worse* with a Barlow-sized head. The high-dimension benefit belongs to the loss, not the architecture.

**Adding asymmetry — it hurts.** Bolting BYOL's tricks onto Barlow Twins:

| Setting | Top-1 |
|---|---|
| Baseline (symmetric) | 71.4 |
| + stop-gradient | 70.5 |
| + predictor | 70.2 |
| + both | 61.3 |

Both together cost 10 points. Once collapse is prevented by the loss, the anti-collapse machinery is pure damage.

**What did not work / limitations found:**
- Removing augmentations badly hurts, like SimCLR and unlike BYOL. Barlow Twins depends on the specific distortion set. The authors spin this as "better controlled invariances," which is a fair but generous reading.
- Optimising $\log|\mathcal{C}|$ directly (the honest IB objective) did not reach SoTA.
- IMAX (Becker & Hinton 1992), $\mathcal{L} = \log|\mathcal{C}_{(Z^A - Z^B)}| - \log|\mathcal{C}_{(Z^A + Z^B)}|$, is conceptually close. "Our attempts to make it work on ImageNet were not successful."
- $\lambda$ is not sensitive — a good sign for reproducibility.

## Worth Remembering

- **The one-sentence version:** compare features across the batch, not samples across features, and collapse becomes impossible for free.

> [!NOTE] Soft whitening ^soft-whitening
> Driving the off-diagonals of the correlation matrix to zero is a *soft* whitening constraint on the embedding. The concurrent W-MSE work does *hard* whitening (an actual Cholesky decomposition per batch) and only reaches 66.3% top-1. Soft beats hard here.

- The redundancy-reduction term relates to the [[Understanding Contrastive Learning through Alignment and Uniformity#^uniformity|uniformity]] term in contrastive learning — both fight for spread-out representations — but this attacks it as feature decorrelation, which is closer to fixing [[Understanding Dimensional Collapse in Contrastive Learning#^dimensional-collapse|dimensional collapse]] than to spreading points on a sphere. Note that "no two features correlated" is a strictly stronger structural demand than "points far apart"; it directly targets the collapse mode that InfoNCE tolerates.
- The paper says the cross-correlation matrix could equally be an **auto**-correlation matrix of one branch, with similar performance (unshown). If true, that means the "twin" structure only matters for the invariance term.
- The 2048-d ResNet bottleneck feeding an 8192-d projector that keeps improving is genuinely unexplained. The intrinsic dimension of what is learned cannot exceed 2048, yet the loss benefits from more room. Worth thinking about — probably the loss is easier to satisfy in an overcomplete space.
- Practical caveat: 8192-wide projectors and a $D \times D = 8192^2$ correlation matrix cost memory. The $D^2$ term is the reason they could not push past 16k.
- This is the direct ancestor of VICReg (variance-invariance-covariance) and of the LeCun-line work in [[LeJEPA- Provable and Scalable Self-Supervised Learning|LeJEPA]] and [[Self-Supervised Learning from Images with I-JEPA|I-JEPA]] — all of them replacing "how do we avoid collapse in practice" with "what statistical property should the embedding have."
- Follow-up question worth chasing: if the loss is a Gaussian-parametrised entropy proxy, what happens if you use a richer parametrisation? The determinant version failed, but maybe not for fundamental reasons.

## Links

Related: [[A Simple Framework for Contrastive Learning (SimCLR)]] · [[Bootstrap Your Own Latent (BYOL)]] · [[Exploring Simple Siamese Representation Learning (SimSiam)]] · [[Momentum Contrast (MoCo)]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[Understanding Contrastive Learning through Alignment and Uniformity]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Whitening Sentence Representations]] · [[Batch Normalization]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[LeJEPA- Provable and Scalable Self-Supervised Learning]] · [[Self-Supervised Learning from Images with I-JEPA]] · [[KL Divergence]] · [[Cross Entropy]] · [[Mode Collapse]]

New topics worth writing: Information Bottleneck principle, VICReg, LARS optimizer, W-MSE and hard whitening for SSL, SwAV and online clustering, Barlow's efficient coding hypothesis, cross-correlation vs cross-covariance matrices
