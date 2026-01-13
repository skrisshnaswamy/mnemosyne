---
title: "Understanding Dimensional Collapse in Contrastive Learning"
authors: ["Li Jing", "Pascal Vincent", "Yann LeCun", "Yuandong Tian"]
year: 2021
arxiv: "2110.09348"
url: https://arxiv.org/abs/2110.09348
priority: Must-Read
read_on: 2026-08-24
tags: [paper, optimization, self-supervised, vision]
---
## The Core Idea

Contrastive learning stops the obvious failure: every image mapping to the same point. Negative pairs push different images apart, so the embeddings cannot all pile up in one spot. Everyone assumed that was enough.

It is not. The embeddings still squash into a **flat slab** inside the sphere. Many directions of the 128-dimensional embedding space carry essentially zero variance. Take a SimCLR model trained 100 epochs on ImageNet, compute the covariance matrix of the validation embeddings, and look at its singular values sorted and logged: a chunk of them fall off to zero. That is dimensional collapse, and it happens even with negatives.

> [!NOTE] Dimensional collapse
> Embeddings do not shrink to one point, but they only span a subspace of dimension $k < d$. The covariance matrix of the embeddings is low-rank. Measured by singular values of $C = \frac{1}{N}\sum_i (z_i - \bar z)(z_i - \bar z)^T$ dropping to zero. ^dimensional-collapse

The paper names two separate causes, and they are genuinely different mechanisms:

1. **Strong augmentation.** If the noise your augmentation adds along some direction is bigger than the real spread of the data along that direction, the network learns to kill that direction. This is a sane thing to do — that direction is pure noise — but it costs you dimensions.
2. **Implicit regularization.** Even when augmentation is mild and every direction is informative, a network with more than one layer *still* collapses dimensions. Gradient descent on stacked matrices quietly prefers low-rank solutions. This one has nothing to do with the data.

The payoff is an explanation for a piece of folklore. SimCLR's **projector** — the small MLP glued to the encoder whose output goes into the loss, and which you throw away afterwards — buys a big accuracy jump and nobody had a clean story for why. The theory says the projector's job is to absorb the collapse, so the representation underneath stays full-rank. And it says the projector's rotation part is redundant: it will align itself with the encoder anyway. So you can delete the projector entirely and just feed the loss the **first $d_0$ coordinates of the representation vector**. That is DirectCLR, and it beats SimCLR-with-a-linear-projector (62.7% vs 61.1% ImageNet linear probe).

## The Methodology

**Setup for the theory.** Input $\mathbf{x}$, augmentation is additive noise so the pair is $(\mathbf{x}_i, \mathbf{x}'_i)$. Network is linear: $\mathbf{z} = W\mathbf{x}$. Loss is [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)#^infonce|InfoNCE]]:

$$L=-\sum_{i}\log\frac{\exp(-|\mathbf{z}_{i}-\mathbf{z}_{i}^{\prime}|^{2}/2)}{\sum_{j\neq i}\exp(-|\mathbf{z}_{i}-\mathbf{z}_{j}|^{2}/2)+\exp(-|\mathbf{z}_{i}-\mathbf{z}_{i}^{\prime}|^{2}/2)}$$

Plain SGD, no momentum, no weight decay. Analysis is by **gradient flow** — [[Backpropagation|gradient descent]] with an infinitesimally small learning rate, so the weights follow a differential equation.

**Cause 1 — strong augmentation.** Chain rule gives $\dot W = WX$ where

$$X = \hat\Sigma_0 - \hat\Sigma_1$$

$\hat\Sigma_0 = \sum_{i,j}\alpha_{ij}(\mathbf{x}_i-\mathbf{x}_j)(\mathbf{x}_i-\mathbf{x}_j)^T$ is the (softmax-weighted) spread of the *data*. $\hat\Sigma_1 = \sum_i (1-\alpha_{ii})(\mathbf{x}'_i-\mathbf{x}_i)(\mathbf{x}'_i-\mathbf{x}_i)^T$ is the spread of the *augmentation noise*. The $\alpha_{ij}$ are the InfoNCE softmax weights.

Both are PSD, so $X$ is a tug-of-war. Solving $\dot W = WX$ with $X$ held fixed gives $W(t) = W(0)U\exp(\Lambda t)U^T$ where $\Lambda$ holds the eigenvalues of $X$. Any **negative** eigenvalue makes that factor decay to zero, so $W$ loses singular values along that eigendirection. Negative eigenvalue means augmentation variance exceeds data variance in that direction. Then $C = WCov(\mathbf{x})W^T$ is low-rank too.

Note the generality over the earlier Tian et al. result: this holds for multiple negatives, varying $\alpha_{ij}$, and finite batch $N$ — not just the population limit with one negative term.

**Cause 2 — implicit regularization.** Now two layers, no nonlinearity, $\mathbf{z} = W_2W_1\mathbf{x}$, and assume augmentation is *mild* so $X \succ 0$ (positive definite, no negative eigenvalues). A single layer would not collapse here. Two layers do.

Step one, **alignment**. SVD both: $W_1 = U_1S_1V_1^T$, $W_2 = U_2S_2V_2^T$. The alignment matrix is $A = V_2^TU_1$. From the flow, $\frac{d}{dt}(W_1W_1^T - W_2^TW_2) = 0$, and both Frobenius norms blow up (because $\mathrm{tr}(W_2W_1XW_1^TW_2^T) > 0$ when $X \succ 0$). So in the limit $W_1W_1^T = W_2^TW_2$, which with non-degenerate singular values forces $U_1 = V_2$, i.e. $A \to I$. **Adjacent layers rotate until they agree.**

Step two, once aligned, the singular values evolve as

$$\dot\sigma_1^k = \sigma_1^k(\sigma_2^k)^2\,(\mathbf{v}_1^{kT}X\mathbf{v}_1^k), \qquad \dot\sigma_2^k = \sigma_2^k(\sigma_1^k)^2\,(\mathbf{v}_1^{kT}X\mathbf{v}_1^k)$$

Each singular value's growth rate is **proportional to itself, cubed-ish**. $X \succ 0$ makes the quadratic form non-negative, so nothing shrinks — but small singular values grow far slower than large ones. Rich get richer. After training the smallest group is still tiny, so $W_2W_1$ is effectively low-rank, so the embedding covariance is low-rank. Same trick as [[Auto-regressive models|matrix factorization]]'s implicit bias toward nuclear-norm-minimal solutions.

**DirectCLR.** From the theory, two claims about a *linear* projector $W_2$:

- **Proposition 1 — it only needs to be diagonal.** Its orthogonal part $V_2$ is redundant, because the encoder's last layer $U_1$ is trainable and will rotate to satisfy $V_2^TU_1 = I$ anyway.
- **Proposition 2 — it only needs to be low-rank.** It converges to low rank regardless, so just set it there.

A fixed low-rank diagonal matrix is the same as *slicing*. So: take representation $\mathbf{r} \in \mathbb{R}^{2048}$ out of the ResNet-50, keep $\mathbf{z} = \mathbf{r}[0{:}d_0]$, normalise $\hat{\mathbf{z}} = \mathbf{z}/|\mathbf{z}|$, apply InfoNCE on that. No projector parameters at all.

$$L=\sum_{i}\log\frac{\exp(\hat{\mathbf{z}}_{i}\cdot\hat{\mathbf{z}}_{i}^{\prime})}{\sum_{j}\exp(\hat{\mathbf{z}}_{i}\cdot\hat{\mathbf{z}}_{j})}$$

**Why the other 2048 − $d_0$ dimensions are not garbage.** The gradient arriving at $\mathbf{r}$ is low-rank — only the first $d_0$ channels are non-zero. But it then passes backwards through the last nonlinear conv block, which mixes channels and makes it full rank. So the hidden layer $\mathbf{h}$ (also 2048 channels) gets gradient everywhere. On the forward pass, $\mathbf{h}$ is added straight into $\mathbf{r}$ by the [[Deep Residual Learning for Image Recognition (ResNet)#^residual-block|residual connection]]. So the whole of $\mathbf{r}$ is trained, indirectly.

**Training.** ResNet-50 encoder, 2048-d output. Standard SimCLR augmentation pipeline (random crop to 224×224, horizontal flip, colour jitter, grayscale, Gaussian blur, solarisation). LARS optimizer, 100 epochs, batch 4096 on 32 GPUs, learning rate 4.8, 10-epoch warmup then cosine decay.

## Ablation Studies and Experiments

**Main table — ImageNet linear probe, top-1, 100 epochs, ResNet-50:**

| Method | Projector | Acc |
|---|---|---|
| SimCLR | 2-layer nonlinear | **66.5** |
| SimCLR | 1-layer linear | 61.1 |
| SimCLR | none | 51.5 |
| DirectCLR | none | **62.7** |

DirectCLR beats the linear projector by 1.6 points and the no-projector baseline by 11.2. It does **not** beat the 2-layer nonlinear projector — 66.5 is still the number to hit.

**Projector ablations (Table 2), each testing one proposition:**

| Projector | diagonal | low-rank | Acc |
|---|---|---|---|
| none | | | 51.5 |
| orthogonal (all singular values fixed at 1) | | | 52.2 |
| trainable linear | | | 61.1 |
| trainable diagonal | ✓ | | 60.2 |
| fixed low-rank | | ✓ | 62.3 |
| fixed low-rank diagonal (= DirectCLR) | ✓ | ✓ | 62.7 |

Read it as three matched pairs. Orthogonal (52.2) ≈ none (51.5): a projector with no singular-value structure does nothing, so **the singular values are the whole story**. Trainable diagonal (60.2) ≈ trainable linear (61.1): **the rotation is redundant**, Proposition 1. Fixed low-rank (62.3) ≈ fixed low-rank diagonal (62.7): same, again. And low-rank is consistently the best column — Proposition 2.

**What did not work:**

- **Random dropout instead of a fixed slice.** Feed a *randomly chosen* $d_0$ subset of features to InfoNCE each iteration: **43.0%**, far worse than DirectCLR's 62.7 and worse even than no projector at all (51.5). The fixed slice is what lets the alignment effect settle; a shuffling mask never lets the encoder's $U_1$ converge to anything.
- **Linear probe on only the sliced sub-vector $\mathbf{z}$** gives 47.9%, versus 62.7% for the full $\mathbf{r}$. So the untouched tail really does carry a lot — but it is not sufficient alone, confirming the residual-connection story rather than contradicting it.
- **$d_0$ has an interior optimum.** $d_0 \to 0$ starves the loss of gradient signal. $d_0 \to 2048$ *is* SimCLR-with-no-projector, which collapses. Both ends drop.

**Toy simulations confirming the theory:**

- 1-layer linear, 16×16 weights, isotropic Gaussian data, augmentation covariance $\mathrm{blockdiag}(0, kI)$ with an 8×8 block. Raising $k$ kills singular values exactly in that block. Cause 1 verified.
- 2-layer linear, 16×16, non-degenerate initialisation. $|A| = |V_2^TU_1|$ converges to the identity. Singular values of both $W_1$ and $W_2$ separate, with the smallest group stuck near zero, and the embedding covariance goes low-rank. Cause 2 verified.
- **Depth amplifies it.** More layers → more collapsed dimensions. With $L=1$ there is no collapse, consistent with the theory. Inserting ReLU between layers gives the same collapse pattern as linear, so it is not an artefact of linearity.

## Worth Remembering

- **The authors' own disclaimer is important.** The theory explains a *linear* projector. It does **not** explain why a *nonlinear* projector works better (66.5 vs 62.7). And DirectCLR is not projector-free in spirit — it leans on the ResNet's last nonlinear conv block to do the projector's job. Read DirectCLR as "the linear projector is provably redundant," not "projectors are unnecessary."

- **Non-degenerate singular values are an assumption, not a fact.** With ordinary random init, singular values come in near-degenerate clusters and the SVD is not unique inside a cluster. The alignment matrix converges to *block-diagonal*, not identity, with one block per degenerate group. The toy experiments deliberately initialise to avoid this.

- **The two causes pull in opposite directions on augmentation strength.** Weaker augmentation avoids cause 1 but does nothing about cause 2. There is no augmentation setting that removes both. This is worth holding next to [[Understanding Contrastive Learning through Alignment and Uniformity]] — uniformity on the hypersphere is a *marginal* property and is perfectly compatible with a low-rank slab, which is exactly the gap this paper walks through.

- **Connects tightly to [[Representation Degeneration Problem in Training NLMs]] and [[How Contextual are Contextualized Word Representations#^anisotropy|anisotropy]].** Different fields, same disease: embeddings occupy a narrow cone or subspace instead of the full space. The mechanism here (implicit low-rank bias of stacked matrices under gradient flow) may be the more general explanation.

- **[[LeJEPA- Provable and Scalable Self-Supervised Learning]] attacks the same problem from the other side** — instead of explaining why collapse happens, it adds an explicit regulariser (SIGReg) pushing embeddings toward an isotropic Gaussian. Reading both together is the right move.

- **Practical caveat:** DirectCLR is cheap to try — it is literally a slice plus a normalise, and the released PyTorch code is at `facebookresearch/directclr`. But if you already run a 2-layer MLP projector you are leaving ~4 points on the table by switching. Its value is conceptual (and for cases where you cannot afford a projector head).

- **Diagnostic you should steal regardless of the method:** dump embeddings on a validation set, compute the covariance, plot $\log \sigma_k$ sorted. If the tail falls off a cliff you are burning capacity. This is a two-line health check for any joint-embedding model.

- **Open question:** cause 2 is a statement about *linear* stacks proved under gradient flow with no weight decay and no momentum. Real training has [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], [[Momentum|momentum]], LARS. Does explicit weight decay change the low-rank bias, and in which direction?

## Links

Related: [[Understanding Contrastive Learning through Alignment and Uniformity]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[Representation Degeneration Problem in Training NLMs]] · [[How Contextual are Contextualized Word Representations]] · [[LeJEPA- Provable and Scalable Self-Supervised Learning]] · [[SimCSE- Simple Contrastive Learning of Sentence Embeddings]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Backpropagation]] · [[Derivative]] · [[Regularization]] · [[Mode Collapse]] · [[Linear Projection]]

New topics worth writing: SimCLR, BYOL / SimSiam and non-contrastive collapse, Barlow Twins and redundancy reduction, VICReg, implicit regularization of gradient descent toward low-rank solutions, singular value decomposition dynamics under gradient flow, LARS optimizer, deep linear networks as a theory testbed
