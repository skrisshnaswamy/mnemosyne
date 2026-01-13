---
title: "Understanding Contrastive Learning through Alignment and Uniformity"
authors: ["Tongzhou Wang", "Phillip Isola"]
year: 2020
arxiv: "2005.10242"
url: https://arxiv.org/abs/2005.10242
priority: Must-Read
read_on: 2026-08-24
tags: [paper, rl, self-supervised, vision, theory]
---
## The Core Idea

Contrastive learning works. Nobody could say *why*. The usual story — "it maximises mutual information between two views" — was already known to be broken: tighter bounds on mutual information often give *worse* representations (Tschannen et al., 2019).

This paper gives a cleaner answer. The popular contrastive loss, in the limit of infinitely many negatives, splits exactly into two terms. One term rewards **alignment**: two augmented copies of the same image should land in the same place. The other term rewards **uniformity**: all the features together should spread out evenly over the surface of the unit sphere.

That is the whole thing. Two properties, both measurable, both optimisable on their own.

> [!NOTE] Alignment
> Positive pairs (two augmentations of the same input) should map to nearby feature vectors. Formally $\mathbb{E}_{(x,y)\sim p_{\mathsf{pos}}}\|f(x)-f(y)\|_2^\alpha$ should be small. ^alignment

> [!NOTE] Uniformity
> The distribution of features on the unit hypersphere $\mathcal{S}^{m-1}$ should be as close to the uniform distribution as possible — this is the distribution that preserves the most information about the input. ^uniformity

Why this matters practically: the two metrics are one-liners in PyTorch, they need no negatives-queue and no softmax, and **training directly on them beats the contrastive loss itself**. On ImageNet-100, MoCo with the contrastive loss gets 72.80% top-1; the same setup with align+uniform gets 74.60%. On full ImageNet with MoCo v2, 67.5% → 67.69%.

The gap exists because Theorem 1 only holds as $M \to \infty$. In practice you have a finite number of negatives, so the contrastive loss is an *approximation* of align+uniform. If you want align+uniform, just ask for it.

## The Methodology

**Setup.** An encoder $f:\mathbb{R}^n \to \mathcal{S}^{m-1}$ maps inputs to $\ell_2$-normalised vectors — points on the unit sphere. Normalising is not decoration: without it the softmax inside the contrastive loss can be made arbitrarily sharp by just scaling all features up.

**The loss being analysed** (this is InfoNCE / SimCLR / MoCo form):

$$\mathcal{L}_{\mathsf{contrastive}}(f;\tau,M) = \mathbb{E}\left[-\log \frac{e^{f(x)^\mathsf{T}f(y)/\tau}}{e^{f(x)^\mathsf{T}f(y)/\tau} + \sum_{i=1}^{M} e^{f(x_i^-)^\mathsf{T}f(y)/\tau}}\right]$$

with $(x,y)$ a positive pair, $x_i^-$ drawn i.i.d. from the data, $\tau$ a temperature.

**Theorem 1 (the decomposition).** As $M \to \infty$, with $\log M$ subtracted off to keep things finite:

$$\lim_{M\to\infty}\mathcal{L}_{\mathsf{contrastive}} - \log M = -\frac{1}{\tau}\mathbb{E}_{p_{\mathsf{pos}}}\!\left[f(x)^\mathsf{T}f(y)\right] + \mathbb{E}_{x}\left[\log \mathbb{E}_{x^-}\left[e^{f(x^-)^\mathsf{T}f(x)/\tau}\right]\right]$$

- First term: minimised **iff** $f$ is perfectly aligned. (On the sphere $\|u-v\|_2^2 = 2 - 2u^\mathsf{T}v$, so maximising dot product *is* minimising distance.)
- Second term: if perfectly uniform encoders exist, they are exactly the minimisers.
- The error from truncating at finite $M$ decays as $\mathcal{O}(M^{-1/2})$.

**The two proposed metrics.**

Alignment is the obvious thing:
$$\mathcal{L}_{\mathsf{align}}(f;\alpha) = \mathbb{E}_{(x,y)\sim p_{\mathsf{pos}}}\left[\|f(x)-f(y)\|_2^\alpha\right], \quad \alpha > 0$$

Uniformity is the interesting one. They use the **Gaussian (RBF) potential** $G_t(u,v) = e^{-t\|u-v\|_2^2}$ and take the log of the average pairwise potential:
$$\mathcal{L}_{\mathsf{uniform}}(f;t) = \log \mathbb{E}_{x,y \overset{\text{i.i.d.}}{\sim} p_{\mathsf{data}}}\left[e^{-t\|f(x)-f(y)\|_2^2}\right]$$

Two propositions justify the choice. Proposition 1: the uniform measure $\sigma_d$ on the sphere is the **unique** minimiser of the expected pairwise Gaussian potential. Proposition 2: as the number of points $N \to \infty$, the $N$-point minimising configurations converge weak\* to $\sigma_d$. Both follow from the Gaussian kernel being *strictly positive definite* on the sphere.

Designing such an objective is not trivial. Average pairwise dot product, or average pairwise Euclidean distance, is minimised by *any* zero-mean distribution — including a two-point distribution $\frac{1}{2}\delta_u + \frac{1}{2}\delta_{-u}$. Two antipodal blobs are not uniform. The log has to sit in exactly the right place: push it all the way inside the inner integral and you are just minimising $\|\mathbb{E}[U]\|^2$, which is the degenerate objective above.

**The implementation** (this is the whole thing):

```python
def lalign(x, y, alpha=2):
    return (x - y).norm(dim=1).pow(alpha).mean()

def lunif(x, t=2):
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()

loss = lalign(x, y) + lam * (lunif(x) + lunif(y)) / 2
```

**Range of $\mathcal{L}_{\mathsf{uniform}}$** (worth knowing for debugging). It lives in $[-2t + \log{}_0F_1(;\tfrac{m}{2};t^2),\ 0]$. The upper bound $0$ is hit only by a fully collapsed encoder (everything maps to one point). The lower bound decreases with output dimension $m$ and converges to $-2t$. Intuition for $-2t$: in high dimensions every pair of unit vectors is roughly orthogonal, so $\|u-v\|_2 \approx \sqrt{2}$, giving $e^{-2t}$.

**Nice connection.** The second term of Theorem 1 is also a **resubstitution entropy estimator** of $H(f(x))$, using a von Mises–Fisher kernel density estimate with concentration $\kappa = \tau^{-1}$. So uniformity = maximising an estimate of feature entropy = maximising an estimate of $I(x; f(x))$. Compared to the InfoMax framing $I(f(x);f(y)) = H(f(x)) - H(f(x)|f(y))$: uniformity is the $H(f(x))$ part, but alignment is *stronger* than merely wanting small $H(f(x)|f(y))$.

**What they trained.** 304 STL-10 encoders (AlexNet-ish), 64 NYU-Depth-V2 encoders, 45 ImageNet-100 encoders (ResNet-50, MoCo), 108 BookCorpus encoders (bi-GRU, Quick-Thought Vectors), plus one full-ImageNet MoCo v2 run. Sweeps over loss weights, $\tau$, $\alpha \in \{1,2\}$, $t \in \{1,\dots,8\}$, batch size, embedding dim, epochs, LR, initialisation. SGD, momentum 0.9, weight decay $10^{-4}$, linearly scaled LR (0.12 per 256 batch).

## Ablation Studies and Experiments

**The metrics predict downstream performance.** Plot every trained encoder as a point with $x = \mathcal{L}_{\mathsf{align}}$, $y = \mathcal{L}_{\mathsf{uniform}}$, colour = validation accuracy. The good encoders sit in the bottom-left corner every single time — across STL-10 classification, NYU depth regression (MSE), ImageNet-100 linear probe, and BookCorpus sentence classification.

**Head-to-head numbers.**

| Setting | Best contrastive-only | Best align+uniform |
|---|---|---|
| STL-10, output+linear | 80.46% ($\tau{=}0.19$) | **81.15%** ($0.98\mathcal{L}_a + 0.96\mathcal{L}_u$, $t{=}2$) |
| STL-10, fc7+linear | 83.89% | **84.43%** |
| NYU-Depth conv5 MSE ↓ | 0.7024 | **0.7014** |
| ImageNet-100 top-1 (MoCo) | 72.80% ($\tau{=}0.07$) | **74.60%** ($3\mathcal{L}_a + \mathcal{L}_u$, $t{=}3$) |
| ImageNet top-1 (MoCo v2, 200ep) | 67.5% ± 0.1% | **67.69%** |
| BookCorpus MR | **77.51%** ($\tau{=}0.075$) | 73.76% |
| BookCorpus CR | **83.86%** ($\tau{=}0.05$) | 80.95% |

Note the text result: on BookCorpus, align+uniform **loses** to the contrastive loss. The paper reports this honestly. The correlation between metrics and accuracy still holds there; direct optimisation just does not win.

**Causal check (Figure 8).** Take an encoder trained with a deliberately bad temperature $\tau = 2.5$ and fine-tune it three ways:
- Optimise alignment only → alignment improves, uniformity degrades, **accuracy degrades**.
- Optimise uniformity only → uniformity improves, alignment degrades, **accuracy degrades**.
- Optimise both → both improve, **accuracy steadily rises**.

This is the strongest evidence in the paper. It rules out "these metrics just correlate with training progress".

**The weight trade-off (Figure 7).** Sweep the ratio of $\mathcal{L}_{\mathsf{align}}$ weight to $\mathcal{L}_{\mathsf{uniform}}$ weight. Accuracy is an inverted U. Weight alignment too heavily and you get **total collapse** — every input maps to the same vector, $\exp \mathcal{L}_{\mathsf{uniform}} = 1$, STL-10 accuracy drops to 10% (chance). This shows up all over the appendix tables: e.g. $0.88 \cdot \mathcal{L}_a + 0.12 \cdot \mathcal{L}_u$ collapses to 10.00%, while $0.85 \cdot \mathcal{L}_a + 0.15 \cdot \mathcal{L}_u$ also collapses, but $0.845/0.155$ trains fine at 74.99%. The boundary is sharp. As long as the ratio is below roughly 4, results are good and insensitive to the exact weights.

**Uniformity alone is nearly useless.** $\mathcal{L}_{\mathsf{uniform}}$ only: 20.50% on STL-10. $2\mathcal{L}_u$ only: 21.91%. Alignment alone: 10.00% (collapse). You need both.

**Temperature sensitivity of the baseline.** The STL-10 contrastive sweep shows a clear peak: $\tau = 0.19$ gives 80.46%, $\tau = 0.005$ gives 68.14%, $\tau = 5$ gives 55.75%. This is a real hyperparameter with a narrow good band — and part of the appeal of align+uniform is that the weight ratio is more forgiving than $\tau$.

**Batch size / dimension effects.** Batch size 2 → 19.31%; batch 16 → 74.68%; batch 768 → 80.76%. Output dimension 2 → ~29%; dimension 128 → ~80%; dimension 1024 with $0.25\mathcal{L}_a + \mathcal{L}_u$ → 83.03%. Larger batches give more pairs for the pairwise potential, which matters.

**MoCo-specific ablation.** With a memory queue, you can compute $\mathcal{L}_{\mathsf{uniform}}$ using only query-vs-queue distances (Eq. 21) or also including intra-batch distances (Eq. 22). Including intra-batch is very slightly better (73.74% vs 73.30% top-1 in the $2\mathcal{L}_a + \mathcal{L}_u$ setting) but the difference is small. Queue size 16384 vs 32768 barely matters.

**What did not work / instabilities.** With $\alpha = 1$ (L1 alignment) on BookCorpus, six configurations hit `NaN` during training. $\alpha = 2$ was stable throughout. On STL-10, $\mathcal{L}_a(\alpha{=}1)$ combined with heavy uniformity weight ($2\times$, $3\times$, $4\times$, $5\times$) gave flat 10.00% — collapse.

## Worth Remembering

- **The temperature and the $t$ in the Gaussian potential are the same knob in disguise.** The proof shows the second term of Theorem 1 relaxes to minimising the expected $G_{1/(2\tau)}$ potential. So $\tau = 0.5$ corresponds to $t = 1$. This is why $t = 2$ (i.e. $\tau = 0.25$) is a sane default.

- **You cannot have both perfectly.** If the data is a finite set and positives come from augmenting it, perfect alignment forces all augmentations of one image to a single point — which makes perfect uniformity impossible. The inverted-U in Figure 7 is this tension made visible.

- **Perfect uniformity is not always realisable** either — e.g. if the data manifold has lower dimension than $m-1$. It does exist when $n \ge m-1$ and $p_{\mathsf{data}}$ has bounded density.

- **The results transfer across method families.** MoCo has a momentum encoder and a queue; Quick-Thought Vectors uses two different encoders, no normalisation during training, and non-random minibatches. The align/uniform correlation with downstream accuracy holds in both. That is a strong hint the framing is general, not an artefact of the exact loss in Equation 1.

- **Practical caveat on the estimator.** The one-liner `lunif` uses `torch.pdist`, which excludes self-distances. This is $\hat{\mathcal{L}}^{(2)}$ in the appendix, and it does *not* respect the theoretical lower bound (it is not the expected potential of any distribution). For reasonably large batches the difference is negligible. If you want a non-negative loss, add an offset of $2t$; for low output dimension also add $-\log {}_0F_1(;\tfrac{m}{2};t^2)$, computable as `scipy.special.hyp0f1(m/2, t**2)`.

- **Weaker result for $M=1$.** Theorem 2 shows that even with a single negative, *if* perfectly aligned and uniform encoders exist, they are the exact minimisers. But this is conditional in a way Theorem 1 is not — Theorem 1 decomposes the loss into two independently-interpretable terms with no existence assumption.

- **Open question the authors flag.** Why is the unit hypersphere a nice feature space at all? The hand-wavy argument (Figure 2) is that well-clustered classes on a sphere form spherical caps, which are linearly separable from the rest — and linear separability is the standard probe. Nobody has made this rigorous.

- **Connection to your existing notes.** The uniformity story is the mirror image of the anisotropy problem in [[How Contextual are Contextualized Word Representations]] and [[Representation Degeneration Problem in Training NLMs]]: those papers observe that embeddings collapse into a narrow cone; this one gives a loss that provably prevents it. And [[LeJEPA- Provable and Scalable Self-Supervised Learning]] does the same thing with a different target distribution (isotropic Gaussian instead of uniform-on-sphere) and a different regulariser (SIGReg instead of Gaussian potential).

- **Follow-up worth chasing.** This decomposition became the standard lens for later work — dimensional collapse, BYOL/SimSiam (which have no negatives at all yet somehow avoid collapse), and the whole "how do non-contrastive methods get uniformity for free" line.

## Links

Related: [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[LeJEPA- Provable and Scalable Self-Supervised Learning]] · [[Representation Degeneration Problem in Training NLMs]] · [[How Contextual are Contextualized Word Representations]] · [[KL Divergence]] · [[Cross Entropy]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[Mode Collapse]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Regularization]] · [[Linear Projection]]

New topics worth writing: strictly positive definite kernels, weak\* convergence of measures, von Mises–Fisher distribution, Riesz s-energy and the Thomson problem, MoCo momentum encoder and memory queue, SimCLR, BYOL and non-contrastive SSL, dimensional collapse in self-supervised learning, kernel density estimation on manifolds, Berry–Esseen theorem
