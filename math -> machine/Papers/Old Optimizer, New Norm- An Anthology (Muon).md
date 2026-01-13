---
title: "Old Optimizer, New Norm: An Anthology (Muon)"
authors: ["Jeremy Bernstein", "Laker Newhouse"]
year: 2024
arxiv: "2409.20325"
url: https://arxiv.org/abs/2409.20325
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, optimization]
---
## The Core Idea

Every optimizer secretly answers one question: **"if I am allowed to move a certain distance, which direction do I go?"** The catch is that "distance" is not fixed. You have to pick a way to measure the size of a weight update. Pick a different ruler, get a different optimizer.

The formal object is **steepest descent under a norm**. You build a local model of the loss around the current weights $\bm{w}$:

$$\mathcal{L}(\bm{w}) + \nabla_{\bm{w}}\mathcal{L}(\bm{w})^\top \Delta\bm{w} + \frac{\lambda}{2}\|\Delta\bm{w}\|^2$$

and minimise it over $\Delta\bm{w}$. The linear term is the [[Derivative#Gradient|gradient]]. The quadratic term is a penalty for moving far — but "far" is measured by whatever norm $\|\cdot\|$ you chose *in advance*. $\lambda$ is called the **sharpness** and is also chosen in advance. Nothing here touches a [[Derivative#Hessian|Hessian]]. So this is a first-order method, not an approximate second-order one, even though it produces updates that look like preconditioning.

The central lemma is that this splits cleanly into two independent decisions:

$$\argmin_{\Delta\bm{w}}\left[\bm{g}^\top\Delta\bm{w} + \frac{\lambda}{2}\|\Delta\bm{w}\|^2\right] = -\underbrace{\frac{\|\bm{g}\|^\dagger}{\lambda}}_{\text{step size}}\cdot\underbrace{\argmax_{\|\bm{t}\|=1}\bm{g}^\top\bm{t}}_{\text{step direction}}$$

where $\|\cdot\|^\dagger$ is the **dual norm**. Direction comes from the norm. Magnitude comes from the dual norm of the gradient divided by sharpness.

> [!NOTE] Dual norm
> $\|\bm{g}\|^\dagger := \max_{\|\bm{t}\|=1}\bm{g}^\top\bm{t}$ — the biggest inner product the gradient can achieve against any unit-length vector. For $\ell_\infty$ the dual is $\ell_1$. For the spectral norm the dual is the nuclear norm (sum of singular values). ^dual-norm

The punchline of the three "stories": Adam, Shampoo and Prodigy — all three originally justified with convex analysis or second-order approximations — are each just steepest descent under a particular norm, once you switch off their exponential moving averages.

- **[[Adam- A Method for Stochastic Optimization|Adam]]** without EMA is $\operatorname{sign}(\bm{g})$, which is steepest descent under $\ell_\infty$.
- **Shampoo** without accumulation is $\bm{U}\bm{V}^\top$ (the gradient's SVD with all singular values set to 1), which is steepest descent under the **spectral norm**.
- **Prodigy** without EMA is sign descent again, with an automatic step-size warmup.

And the thing this unlocks: if the choice of norm is the real design knob, then you should choose norms **per layer, based on what the layer does**. An embedding matrix and a linear layer both live in $\mathbb{R}^{m\times n}$, but one maps one-hot vectors and one maps unit-RMS activations. They deserve different rulers. That reframing is what later became the **Muon** optimizer.

## The Methodology

There is no training run in this paper. The method *is* the derivations. Here they are, concretely.

### Adam → sign descent → max-of-max norm

Adam, stripped of bias correction and $\epsilon$:

$$\bm{m}_t = \beta_1 \bm{m}_{t-1} + (1-\beta_1)\bm{g}_t,\quad \bm{v}_t = \beta_2 \bm{v}_{t-1} + (1-\beta_2)\bm{g}_t^2,\quad \bm{w}_{t+1} = \bm{w}_t - \eta\, \bm{m}_t/\sqrt{\bm{v}_t}$$

Set $\beta_1=\beta_2=0$. Then $\bm{m}_t/\sqrt{\bm{v}_t} = \bm{g}_t/\sqrt{\bm{g}_t^2} = \operatorname{sign}(\bm{g}_t)$. Done. (Hinton already said RMSprop was "the mini-batch version of just using the sign of the gradient" in 2012; RPROP in 1993 did the same.)

Sign descent solves steepest descent under $\ell_\infty$:

$$\argmin_{\Delta\bm{w}}\left[\bm{g}^\top\Delta\bm{w} + \frac{\lambda}{2}\|\Delta\bm{w}\|_\infty^2\right] = -\frac{\|\bm{g}\|_1}{\lambda}\operatorname{sign}(\bm{g})$$

But *why* would the $\ell_\infty$ norm of the flattened weight vector have anything to do with a neural network? It throws away all layer structure. The authors' answer: it doesn't — there is a coincidence. **A max of a max is a max.** If $\bm{w}$ is all the weight matrices flattened together,

$$\|\bm{w}\|_\infty = \max_l \max_r \|\mathrm{row}_r(\bm{W}_l)\|_\infty = \max_l \|\bm{W}_l\|_{\ell_1\to\ell_\infty}$$

So the same sign update *also* solves a second, structure-aware problem: minimise $\sum_l \langle \bm{G}_l, \Delta\bm{W}_l\rangle + \frac{\lambda}{2}\max_l\|\Delta\bm{W}_l\|_{\ell_1\to\ell_\infty}^2$. The solution is layerwise sign descent with a **shared** step size $\eta = \frac{1}{\lambda}\sum_l \|\bm{G}_l\|_1$.

The real content: sign descent is doing **per-matrix gradient normalisation**. Every layer gets an update of the same size regardless of its gradient magnitude. That is likely the main reason Adam, Lion and signSGD beat plain SGD on large language models.

> [!NOTE] Induced operator norm
> Given input norm $\|\cdot\|_\alpha$ and output norm $\|\cdot\|_\beta$, $\|\bm{M}\|_{\alpha\to\beta} = \max_{\bm{x}} \frac{\|\bm{M}\bm{x}\|_\beta}{\|\bm{x}\|_\alpha}$ — the most a matrix can stretch a vector, measured with your chosen rulers on each side. Vary $\alpha,\beta$ and you get a whole family of matrix norms, hence a whole family of optimizers. ^induced-operator-norm

### Shampoo → orthogonalised gradient → spectral norm

Shampoo keeps two preconditioners and updates:

$$\bm{L}_t = \bm{L}_{t-1} + \bm{G}_t\bm{G}_t^\top,\quad \bm{R}_t = \bm{R}_{t-1} + \bm{G}_t^\top\bm{G}_t,\quad \bm{W}_{t+1} = \bm{W}_t - \eta\,\bm{L}_t^{-1/4}\bm{G}_t\bm{R}_t^{-1/4}$$

Kill the accumulation ($\bm{L}_t = \bm{G}_t\bm{G}_t^\top$, $\bm{R}_t = \bm{G}_t^\top\bm{G}_t$) and substitute the reduced SVD $\bm{G}_t = \bm{U}_t\bm{\Sigma}_t\bm{V}_t^\top$:

$$(\bm{G}\bm{G}^\top)^{-1/4}\,\bm{G}\,(\bm{G}^\top\bm{G})^{-1/4} = \bm{U}\bm{V}^\top$$

The singular values cancel out entirely. The update is a **semi-orthogonal matrix** — same singular vectors as the gradient, all singular values forced to 1. This is exactly parallel to $\bm{g}/\sqrt{\bm{g}^2} = \operatorname{sign}(\bm{g})$, one dimension up.

Two facts about $\bm{U}\bm{V}^\top$:

1. It is the closest semi-orthogonal matrix to $\bm{G}$ in Frobenius norm: $\argmin_{\bm{A}\bm{A}^\top=\bm{I}}\|\bm{A}-\bm{G}\|_F = \bm{U}\bm{V}^\top$. Unique iff $\bm{G}$ is full rank.
2. It solves steepest descent under the max-over-layers spectral norm, with step size $\eta = \frac{1}{\lambda}\sum_l \operatorname{tr}\bm{\Sigma}_l$ (sum of nuclear norms — the dual of the spectral norm).

Why is spectral norm the right ruler? Because loss bounds naturally come out in spectral norm. Simplest case, a linear predictor with square loss and inputs normalised to $\|\bm{x}_i\|_2 = \sqrt{d_{\text{in}}}$:

$$\mathcal{L}(\bm{W}+\Delta\bm{W}) \leq \mathcal{L}(\bm{W}) + \langle\nabla\mathcal{L},\Delta\bm{W}\rangle + \frac{1}{2}\cdot\frac{d_{\text{in}}}{d_{\text{out}}}\cdot\|\Delta\bm{W}\|_{\ell_2\to\ell_2}^2$$

The sharpness is *literally* $\lambda = d_{\text{in}}/d_{\text{out}}$ — a known constant, no Hessian needed. Minimising this upper bound gives you spectral descent exactly. That design pattern (bound the loss, then minimise the bound) is **majorization-minimization**, and it is fully first-order.

### Computing $\bm{U}\bm{V}^\top$ cheaply — the Newton–Schulz iteration

This is the part that made the method practical. Full SVD every step is too slow. Instead set $\bm{X}_0 = \bm{G}/\|\bm{G}\|_F$ and iterate:

$$\bm{X}_{t+1} = \tfrac{3}{2}\bm{X}_t - \tfrac{1}{2}\bm{X}_t\bm{X}_t^\top\bm{X}_t$$

This applies the scalar cubic $f(x) = \frac{3}{2}x - \frac{1}{2}x^3$ to every singular value at once. For $0 < x < \sqrt{3}$, iterating $f$ pushes $x$ toward $1$. So $\bm{X}_t \to \bm{U}\bm{V}^\top$ using only matmuls — no SVD, no eigendecomposition. Normalising by $\|\bm{G}\|_F$ is enough; the stricter spectral normalisation is not required.

More generally you can use any odd polynomial $\bm{X}_{t+1} = a\bm{X}_t + b\bm{X}_t\bm{X}_t^\top\bm{X}_t + c(\bm{X}_t\bm{X}_t^\top)^2\bm{X}_t + \dots$ and tune $a,b,c,\dots$ so $g(x) = ax + bx^3 + cx^5 + \dots$ approximates $\operatorname{sign}(x)$ on $(0,1]$. The authors say the coefficients "can be tuned graphically". (They later found the fixed-coefficient version is classical — Kovarik 1970, Björck & Bowie 1971.)

Other listed options: full SVD, randomised sketching, Newton iteration for inverse $p$th roots. Also a useful identity: $(\bm{G}\bm{G}^\top)^{-1/4}\bm{G}(\bm{G}^\top\bm{G})^{-1/4} = (\bm{G}\bm{G}^\top)^{-1/2}\bm{G} = \bm{G}(\bm{G}^\top\bm{G})^{-1/2}$, so you only need whichever of the two Gram matrices is smaller.

### Prodigy → sign descent with an automatic warmup

Prodigy's five-line update collapses, at $\beta_1=\beta_2=0$, to:

$$\eta_{t+1} = \max\left(\eta_t,\ \frac{\bm{g}_t^\top(\bm{w}_0 - \bm{w}_t)}{\|\bm{g}_t\|_1}\right),\qquad \bm{w}_{t+1} = \bm{w}_t - \eta_t\operatorname{sign}(\bm{g}_t)$$

Same norm as Adam ($\ell_\infty$), different step size rule. Rewrite the numerator using the angle $\theta$ between $\bm{g}_t$ and $\bm{w}_0-\bm{w}_t$:

$$\eta_{t+1} = \max\left(\eta_t,\ \frac{\|\bm{g}_t\|_2}{\|\bm{g}_t\|_1}\times\|\bm{w}_t-\bm{w}_0\|_2\times\cos\theta\right)$$

Assume (1) the gradient is dense so $\|\bm{g}\|_2/\|\bm{g}\|_1 \approx 1/\sqrt{n}$, and (2) we are still near init so $\cos\theta\approx 1$. Then $\eta_{t+1}\approx\max(\eta_t, \|\bm{w}_t-\bm{w}_0\|_{\text{RMS}})$ where $\|\cdot\|_{\text{RMS}} = \frac{1}{\sqrt n}\|\cdot\|_2$. Since a sign vector has unit RMS norm, the next step's size equals *all the progress made so far* — so the step size doubles each iteration. It grows exponentially until assumption (2) breaks.

> [!NOTE] Escape velocity
> The step size warmup stops when the weights have left the region where the loss is still well-approximated by its linearisation at $\bm{w}_0$. The signal is the directional derivative $(\bm{w}_1-\bm{w}_0)^\top\bm{g}_1$: at the optimal step size it vanishes (Cauchy, 1847). If it is still negative, you could have stepped further. No convexity assumed anywhere. ^escape-velocity

This is an online line search, and Prodigy is one arbitrary point in that space.

### The modular norm — the generalisation

Give layer $l$ its own norm $\|\cdot\|_l$ and a scalar weight $s_l$. Define the modular norm as $\max_l s_l\|\bm{W}_l\|_l$. Then:

$$\Delta\bm{W}_l = -\frac{\eta}{s_l}\cdot\argmax_{\|\bm{T}_l\|_l=1}\langle\bm{G}_l,\bm{T}_l\rangle,\qquad \eta = \frac{1}{\lambda}\sum_{k=1}^L\frac{1}{s_k}\|\bm{G}_k\|_k^\dagger$$

Each layer gets a direction from *its own* norm; a single global step size is a weighted sum of dual norms across layers. Set all $s_l=1$ and all norms to $\ell_1\to\ell_\infty$ → Adam. Set all norms to spectral → Shampoo.

Which norms are actually computable? Proposition 8:

$$\|\bm{M}\|_{\ell_1\to\ell_p} = \max_j\|\mathrm{col}_j(\bm{M})\|_p,\qquad \|\bm{M}\|_{\ell_p\to\ell_\infty} = \max_i\|\mathrm{row}_i(\bm{M})\|_{\frac{p}{p-1}}$$

Max column norm and max row norm. Trivially cheap.

The prescriptive advice:
- **Linear layers** map unit-RMS vectors to unit-RMS vectors → use the RMS→RMS induced norm, which is the spectral norm rescaled by $\sqrt{d_{\text{out}}/d_{\text{in}}}$.
- **Embedding layers** map one-hot vectors to unit-RMS vectors → use the $\ell_1\to$RMS norm, which is a rescaled $\ell_1\to\ell_2$ norm, i.e. **largest column $\ell_2$ norm**. Concretely: normalise each embedding row/column, do not orthogonalise.

Same weight shape, different job, different norm.

## Ablation Studies and Experiments

**There are none.** This is an anthology of propositions with proofs in Appendix B. No benchmark, no loss curve, no table of BLEU or perplexity. Table 1 is a taxonomy, not results. If you want empirical evidence you have to go to the cited work — Shampoo won the external tuning track of AlgoPerf 2024, and Zhao et al. 2024 found sign-like optimizers beat SGD on LLMs.

What the theory rules in and rules out is still informative:

**Where the equivalences break.**
- Every result requires **EMA switched off** ($\beta_1=\beta_2=0$). With EMA on, none of these are exactly steepest descent. The authors are explicit that the role of EMA is "perhaps still an open problem" — they hand-wave it as "smoothing out the algorithm" or making it robust to mini-batch noise. This is the biggest hole.
- The Shampoo↔spectral-descent equivalence needs $\bm{G}_l$ to be **full rank** for the solution to be unique. If $\bm{G}$ is rank-deficient, $\bm{U}\bm{V}^\top$ is one of many equally good answers — you can flip the sign on any zero-singular-value direction and get the same objective value. In practice gradient matrices from small batches are often low rank.
- The Prodigy story needs two assumptions that both eventually fail: dense gradients ($\|\bm{g}\|_2/\|\bm{g}\|_1\approx 1/\sqrt n$) and $\cos\theta\approx 1$. The second failing is the *mechanism* that stops the warmup, so that one is by design; the first one silently controls the conversion between $\ell_2$ and RMS and is just an approximation.

**What they criticise.**
- Prodigy's step size can only **increase, never decrease** — the authors call this possibly sub-optimal.
- Measuring the angle against $\bm{w}_t-\bm{w}_0$ (total displacement from init) rather than $\bm{w}_t - \bm{w}_{t-1}$ (last step) is an arbitrary choice.
- Relying on the ratio $\|\bm{g}\|_2/\|\bm{g}\|_1$ to implicitly do an $\ell_2\to$RMS conversion is indirect.

**The one empirical crumb.** They mention "preliminary experiments" where a step-size rule of the form

$$\eta_{t+1} = \eta_t \times (1 + \cos\theta)$$

worked well. No numbers, no setup, no baseline. Treat it as a hint, not a result.

**What the analysis reveals about which component matters.** For Adam, the working ingredient is not the second moment as a curvature estimate — it is per-matrix normalisation, which makes every layer take an equal-sized step. For Shampoo, the working ingredient is not the accumulated preconditioner — it is discarding the singular values of the gradient entirely. Both preconditioners turn out to be, at their core, normalisation schemes wearing a second-order costume.

## Worth Remembering

**This paper is where Muon comes from.** The name never appears in the text, but Keller Jordan is in the acknowledgements, and Muon = momentum + Newton–Schulz orthogonalisation of the gradient (Equation 36) applied to 2D parameters, with Adam kept for embeddings, scalars and the output head. That layer split is exactly the prescription in the epilogue. This paper is the theory; Muon is the artefact.

**The reframing to keep.** "Optimizer design = choose a norm + choose a step size." Once you accept that, a lot of accumulated folklore reorganises itself: gradient clipping, layerwise learning rates, per-layer LR scaling in muP, normalisation of updates — all become statements about which ruler you are using.

**Practical caveats if you want to use this.**
- Newton–Schulz needs input singular values in $(0,\sqrt3)$. Normalising by $\|\bm{G}\|_F$ guarantees this. Do not skip the normalisation.
- The iteration is all matmuls, so it runs well in low precision — see [[Mixed Precision training]] — but the cubic is numerically touchy near $x=0$; small singular values converge to 1 slowly, which is why tuned higher-degree polynomials are used in practice.
- The theory gives a step size ($\eta = \frac{1}{\lambda}\sum_l\operatorname{tr}\bm{\Sigma}_l$, i.e. sum of nuclear norms), but it depends on a sharpness $\lambda$ you do not know for a real deep network. Proposition 6 only computes $\lambda$ for a linear model with square loss. In practice you still tune the learning rate.
- The max-over-layers structure means a **single global step size** shared across all layers, with per-layer directions. That is a different design from per-layer adaptive LRs.

**Connections.** The "assign norms by layer role" idea is the same insight as the spectral condition for feature learning (Yang, Simon & Bernstein 2023) that underlies muP and learning-rate transfer across model scale — relevant to anything in [[Scaling Laws for Neural Language Models]] or [[Training Compute-Optimal Large Language Models (Chinchilla)]] territory, since transferring hyperparameters across scale is what makes those sweeps affordable. The majorization-minimization pattern also underlies Automatic Gradient Descent (Bernstein et al. 2023).

**Open questions this leaves.**
- What does EMA actually *do*, in norm language? Is momentum a change of norm, a variance reduction, or something else? (Compare [[Momentum]] and the weight-decay decoupling story in [[Decoupled Weight Decay Regularization (AdamW)]] — weight decay under a non-Euclidean norm is also unresolved here.)
- How do you pick the $s_l$ coefficients in the modular norm? The paper defines them and never specifies them.
- What norm should attention layers, LayerNorm gains, or convolutions get? Only linear and embedding layers are addressed.

## Links

Related: [[Adam- A Method for Stochastic Optimization]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Momentum]] · [[Derivative]] · [[Backpropagation]] · [[Fundamentals]] · [[Mixed Precision training]] · [[Layer Normalization]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[Regularization]] · [[Deep Learning]] · [[Loss, Objectives, and Business Alignment]]

New topics worth writing: Muon optimizer, Newton–Schulz orthogonalisation, Shampoo optimizer, Prodigy and parameter-free learning, modular norm, dual norms and Hölder duality, Schatten norms, majorization-minimization, signSGD and Lion, muP and the spectral condition for feature learning, AlgoPerf benchmark, line search methods
