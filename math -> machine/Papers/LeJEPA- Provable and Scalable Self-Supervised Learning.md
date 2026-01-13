---
title: "LeJEPA: Provable and Scalable Self-Supervised Learning"
authors: ["Randall Balestriero", "Yann LeCun"]
year: 2025
arxiv: "2511.08544"
url: https://arxiv.org/abs/2511.08544
priority: Must-Read
read_on: 2026-08-23
tags: [paper, self-supervised, vision, theory]
---
## The Core Idea

Self-supervised learning (SSL) for vision has a dirty secret: the loss that stops the network from cheating is a pile of tricks. Stop-gradients, teacher–student copies with exponential moving average schedules, whitening layers, prototype banks, register tokens. Nobody could say *why* any of it was needed, only that removing it made the encoder collapse — every image mapping to nearly the same vector.

This paper replaces the whole pile with two statements it can prove.

**Statement 1: there is a single best shape for the embedding cloud, and it is the isotropic Gaussian.** If you do not know what downstream task you will face, the distribution of $f_\theta(x)$ that minimises worst-case prediction risk is $\mathcal{N}(0, \sigma^2 I)$ — a round ball of points, equal variance in every direction, Gaussian along every axis. This is proved for linear probes, for radius-based k-NN probes, and for kernel (Nadaraya–Watson) probes.

**Statement 2: you can force that distribution cheaply, using 1-D statistics.** Testing "is this 1024-dimensional cloud a standard Gaussian?" directly is expensive and suffers the curse of dimensionality. Instead, project the embeddings onto a few hundred random unit directions, and on each 1-D projection run a classical normality test. The Cramér–Wold theorem says that if *all* 1-D projections match, the multivariate distributions match. That is SIGReg — Sketched Isotropic Gaussian Regularization.

Put the JEPA prediction loss and SIGReg together and you get **LeJEPA** (Latent-Euclidean JEPA): one hyperparameter $\lambda$, $\mathcal{O}(N)$ time and memory, ~50 lines of PyTorch, no stop-gradient, no teacher, no schedulers. ViT-H/14 pretrained on ImageNet-1k reaches **79%** frozen linear probe.

The unlock that matters most in practice: because there are no fragile heuristics, you can pretrain **in-domain on small weird datasets**. On Galaxy10 (11k galaxy images), a ResNet-34 trained from scratch with LeJEPA hits 78.17% frozen linear probe, beating DINOv3 ViT-S/16 transfer at 71.38%.

> [!NOTE] Isotropic Gaussian embeddings
> Embeddings whose covariance is $\sigma^2 I$ — same spread in every direction — and whose marginal along every direction is Gaussian. The paper proves this is the risk-minimising embedding distribution when the downstream task is unknown. ^isotropic-target

> [!NOTE] SIGReg
> Match a high-dimensional embedding distribution to a target by (a) projecting onto $M$ random unit directions, (b) running a univariate goodness-of-fit test on each projection, (c) averaging. Linear in batch size and dimension, fully differentiable, works over multi-GPU with one `all_reduce`. ^sigreg

## The Methodology

### Why isotropic Gaussian is optimal

**Linear probe.** Fit ridge regression $\hat\beta = \arg\min \|y - Z\beta\|_2^2 + \lambda\|\beta\|_2^2$ on frozen embeddings $Z$. Compare two embedding matrices with the same total energy: one with covariance eigenvalues $\{\lambda_k\}$ (anisotropic), one with all eigenvalues equal to $\frac{1}{K}\sum_k \lambda_k$.

- *Variance* (with $\lambda = 0$): $\mathrm{tr}(\mathrm{Var}(\hat\beta)) = \sigma^2 \sum_{k} 1/\lambda_k$. Since $x \mapsto 1/x$ is strictly convex, [[Derivative#Understanding **Curvature**|Jensen]] gives $\sum_k 1/\lambda_k > K / \bar\lambda$ whenever the eigenvalues differ. Squashed directions blow up the estimator's variance.
- *Bias* (with $\lambda > 0$): the ridge bias is $-\lambda(Z^\top Z + \lambda I)^{-1}\beta_{\text{true}}$. Pick $\beta_{\text{true}}$ aligned with the smallest eigenvector and the bias norm becomes $\frac{\lambda}{\lambda_{\min} + \lambda}\|\beta_{\text{true}}\|$, strictly worse than the isotropic $\frac{\lambda}{\bar\lambda + \lambda}\|\beta_{\text{true}}\|$. There is *always* a downstream task that punishes anisotropy.

**Nonlinear probes.** For radius-$r_0$ k-NN, the pointwise bias at query $q$ is
$$\mathrm{Bias}(q) = \frac{r_0^2}{K+2}\Big(\nabla\eta(q)^\top \nabla \log p_z(q) + \tfrac12 \Delta\eta(q)\Big) + o(r_0^2)$$
Averaging the square over queries, and assuming the unknown label function $\eta$ has isotropic mean-zero gradient, the only term that depends on the embedding density $p_z$ is the **Fisher information**
$$J(p) = \int \|\nabla \log p(x)\|^2\, p(x)\, dx$$
So: *minimise the Fisher information of your embedding density.* Two steps finish it. (i) By the matrix Cramér–Rao bound applied to the location family $p_\theta(x) = p(x-\theta)$, $J(p) \ge \mathrm{tr}(\Sigma^{-1})$, with equality **iff** $p$ is Gaussian (equality needs an affine score, which integrates to a Gaussian). (ii) $\mathrm{tr}(\Sigma^{-1}) = \sum_k 1/\lambda_k$ is minimised at equal eigenvalues under any scalar constraint (trace, determinant, Frobenius, spectral radius) — Cauchy–Schwarz or AM–GM. Hence: isotropic Gaussian, uniquely. The kernel-probe derivation gives the same answer with $J(p)$ appearing in the bias bound $\big(\tfrac{h^2\mu_2(K)}{2}\big)^2(2B^2 + 8L^2 J(p))$.

### SIGReg: the actual regulariser

Reformulate distribution matching as a hypothesis test $H_0: P_\theta = Q$ where $Q = \mathcal{N}(0, I)$. Testing this in $K$ dimensions costs at least $\mathcal{O}(N^2)$. So test along directions instead. For unit vector $a$, test $H_0(a): P_\theta^{(a)} = Q^{(a)}$ on the scalars $a^\top f_\theta(x_n)$. A hypersphere version of the Cramér–Wold theorem gives $\langle u, X\rangle \stackrel{d}{=} \langle u, Y\rangle\ \forall u \in \mathbb{S}^{K-1} \iff X \stackrel{d}{=} Y$, so directional tests are sufficient. The definition:
$$\mathrm{SIGReg}_T(\mathbb{A}, \{z_n\}) = \frac{1}{|\mathbb{A}|}\sum_{a \in \mathbb{A}} T\big(\{a^\top z_n\}_{n=1}^N\big)$$
Note: *average*, not max, so gradients are not sparse across directions.

**Which test $T$?** They compare three families and reject two:

- **Moment tests** (Jarque–Bera, extended to first four moments). Rejected. Theorem 3: matching finitely many moments does not pin down the distribution — you can always build two distinct discrete measures matching $K$ moments (a rank argument on the Vandermonde-style map). More moments would fix identifiability, but the gradient of the $k$-th moment scales as $O(k)$ in magnitude and the Monte Carlo variance as $O(k^2 m_{2(k-1)})$. Stability and identifiability cannot both be had.
- **CDF tests** (Cramér–von Mises, Anderson–Darling, Watson, Shapiro–Wilk). Rejected. They require sorting: $\mathcal{O}(N\log N)$, non-differentiable, and it destroys the embarrassingly parallel structure of multi-GPU SGD. Kolmogorov–Smirnov additionally uses $\ell_\infty$, which gives sparse gradients. Shapiro–Wilk was found unstable in practice.
- **Characteristic-function tests** — the winner. The Epps–Pulley statistic:
$$EP = N\int_{-\infty}^{\infty}\big|\hat\phi_X(t) - \phi(t)\big|^2 w(t)\, dt, \qquad \hat\phi_X(t) = \frac1N\sum_{j=1}^N e^{itX_j}$$
with Gaussian window $w(t) = e^{-t^2/\sigma^2}$ and target $\phi(t) = e^{-t^2/2}$. The empirical characteristic function is the [[Fourier Series Decomposition|Fourier transform]] of the empirical density — it is a plain **average of complex exponentials**, so it is differentiable and syncs across GPUs with one `all_reduce` on the mean.

Two properties make it work. **Boundedness** (Theorem 4): $\left|\partial EP/\partial z_i\right| \le 4\sigma^2/N$ and $\left|\partial^2 EP/\partial z_i^2\right| \le C\sqrt\pi\sigma^3/(2N)$, regardless of the input distribution. Compare moments, whose sample gradient is a polynomial $\frac{2}{N}\sum_r c_r r z_i^{r-1}$ that grows like $|z_i|^{k-1}$. **Small bias** (Theorem 6): the minibatch V-statistic satisfies
$$\mathbb{E}[\hat L_n(\theta)] = L(\theta) + \frac1N \int w_s(t)\big(1 - |\varphi_P(t)|^2\big)\,dt$$
so both loss and gradient carry only an $\mathcal{O}(1/N)$ bias — harmless even at batch size 16.

The integral is done by trapezoid quadrature with 17 knots on $t \in [-5,5]$.

### How many directions?

Two arguments beat the curse of dimensionality. **Smoothness** (Theorem 5): if the embedding density lies in the Sobolev space $H^\alpha$, the expected residual discrepancy over unseen directions decays as $|\mathbb{A}|^{-2\alpha/(K-1)}$, so $|\mathbb{A}| = O(K)$ suffices when $\alpha$ is large — and deep nets are smooth. **SGD**: directions are *resampled every step*, seeded by `global_step` so all GPUs agree. Empirically $|\mathbb{A}| = 16$ resampled beats thousands of fixed directions.

### The full loss

Views follow the DINO convention: $V_g$ global crops (224×224) plus $V_l$ local crops (96×96), $V = V_g + V_l$. The prediction term is "every view predicts the mean of the global views":
$$\mathcal{L}_{\text{pred}} = \frac{1}{V}\sum_{v'=1}^{V}\big\|\mu_n - z_{n,v'}\big\|_2^2, \qquad \mu_n = \frac{1}{V_g}\sum_{v=1}^{V_g} z_{n,v}$$
(this equals the all-pairs form $\frac{1}{V_g V}\sum_{v,v'}\|z_{n,v} - z_{n,v'}\|^2$ by expansion). Total:
$$\mathcal{L}_{\text{LeJEPA}} = \frac{\lambda}{V}\sum_{v=1}^{V}\mathrm{SIGReg}\big(\{z_{n,v}\}_{n=1}^B\big) + \frac{1-\lambda}{B}\sum_{n=1}^{B}\mathcal{L}_{\text{pred}}$$

Recipe that transfers everywhere: $\lambda = 0.05$, $V_g = 2$, $V_l = 8$, batch $\ge 128$, 1024 slices, 17 quadrature points on $[-5,5]$, AdamW with lr $\in \{5\text{e-}3, 5\text{e-}4\}$, wd $\in \{1\text{e-}1, 1\text{e-}2, 1\text{e-}5\}$, cosine schedule on lr only, off-the-shelf `timm` backbones.

## Ablation Studies and Experiments

**Hyperparameter stability (ViT-L/14, ImageNet-1k, 100 epochs, frozen linear probe).**

| Knob | Range | Top-1 |
|---|---|---|
| Batch size | 128 / 256 / 512 / 1024 | 72.20 / 74.15 / 74.72 / 74.07 |
| Integration domain | $[-1,1]$ / $[-3,3]$ / $[-5,5]$ | 72.1 / 74.2 / 74.2 |
| Quadrature points | 5 / 17 / 41 | ~74.2 in all cases |
| Slices $|\mathbb{A}|$ | 512 / 2048 | 74.2 / 74.8 |
| Register tokens | 0 / 1 / 2 / 4 / 8 | 75.14 → 75.23 (flat) |
| Projector dim | 64 / 128 / 256 / 512 / 1024 | 75.65 / 75.47 / 75.02 / 74.65 / 74.79 |

Nothing catastrophically collapses. Smaller projector heads are mildly better. Batch 128 still works, which is unusual for SSL.

**The one thing that actually breaks it: a single global view.** With $V_g = 1$ and 4 total views, accuracy is 53.06%; with $V_g = 2$ it jumps to 72.26%. Even at 10 views, $V_g=1$ only reaches 68.97 vs 74.06. The prediction target being an *average of two* global views is doing real work.

**Removing heuristics (ImageNet-100, 400 epochs, Table 4).** Predictor networks and teacher–student are usually load-bearing — take them out of BYOL/DINO and you get chance-level collapse. Under LeJEPA, no collapse. ResNet-50 with a 3-layer projector: 83.93% without predictor or SWA, 83.50% with SWA, 83.57% with predictor. For ViT-Small/8 the teacher-style stochastic weight averaging does help (80.77 → 83.63 for the 2-layer projector) but it is a bonus, not a requirement. Register tokens are unnecessary — the instabilities they were invented for came from a badly conditioned objective, not the architecture.

**Architecture agnosticism.** ~50 `timm` models under 20M params, 8 families, on ImageNet-10: all land between **91.5% and 95%** frozen linear probe. The models that are good supervised (ResNets, ViTs) are the good ones here; specialised models like EfficientNet lag.

**Loss predicts downstream accuracy.** This is the surprising practical result. In current JEPAs the training loss correlates weakly with probe accuracy and may not even fall monotonically. LeJEPA's loss has Spearman $\rho_s \approx 85\%$ with frozen linear probe accuracy across learning rates, weight decays, epochs and $\lambda$. Rescaling by the trade-off weight,
$$C^{(\alpha)} = \rho_s\!\left(\frac{\text{train\_loss}}{\lambda^{\alpha}},\ \text{test\_accuracy}\right)$$
with $\alpha \approx 0.4$ pushes it to **~99%**. That means label-free model selection and cross-validation.

**Scale.** ImageNet-1k online linear probe: ViT-L (0.3B) 77.1%, ConvNeXtV2-Huge (0.6B) 78.5%, ViT-H/14 79%. A 1.8B ViT-gigantic trains with a stable, smooth loss curve. Transfer to 8 datasets, all-shot average: LeJEPA ViT-L (304M, 100 epochs) **79.48** vs I-JEPA ViT-H (632M, 300 epochs) **78.50** — smaller model, 3× shorter schedule. I-JEPA+STOP gets 80.70 and I-JEPA pretrained on IN-22K for 900 epochs gets 80.82, so LeJEPA is not the top line here, but it is the best per unit of compute. LeJEPA wins the fine-grained tasks (DTD 78.30 vs 73.32, food101 82.05 vs 81.02, flowers102 91.21 vs 86.47) and loses on CIFAR-10/100.

**In-domain beats frontier transfer.** Galaxy10, frozen backbone, all data: LeJEPA ResNet-34 **78.17** vs DINOv3 ViT-S/16 71.38 vs DINOv2 Small 67.62. At 1-shot: 31.08 vs 30.17 vs 27.68. Full fine-tuning: 83.28 vs 81.60. On flowers102 (1020 training images) LeJEPA ResNeXt-26 from scratch hits 82.19, versus IJEPA-IN22K transfer at 85.76 — competitive from a thousand images.

**Emergent structure.** PCA of last-layer features cleanly separates foreground from background; thresholding [CLS] self-attention gives temporally coherent video object masks, without any segmentation labels. Same qualitative behaviour DINO reported, without DINO's machinery.

## Worth Remembering

- **VICReg is the degenerate case.** Set $T(\{x_n\}) = \mathrm{mean}(\{x_n\})^2 + (\mathrm{std}(\{x_n\}) - 1)^2$ and plug it into SIGReg — in the limit of many slices this enforces $\mathbb{E}[Z] = 0$ and $\mathrm{Cov}(Z) = I$, i.e. VICReg. The authors explicitly warn against it: matching only two moments is exactly the non-identifiability of Theorem 3, which is why VICReg has observed shortcut solutions.
- **Relatives elsewhere.** The slicing trick is the same one behind sliced score matching and the sliced Wasserstein distance. If you compute the Epps–Pulley integral exactly rather than by quadrature, each slice's loss *is* a kernel MMD — but at $\mathcal{O}(N^2)$. The quadrature is what buys linearity.
- **Cost.** On a V100, SIGReg forward+backward at $N=512$, $M=512$, 16 knots is 0.47 ms; at $N=8192$, $M=8192$ it is 8.7 ms. Cheap relative to a backbone step.
- **Admitted limitations.** The $\mathcal{O}(1/N)$ gradient bias is not removed — U-statistic debiasing or sample splitting would fix it, unexplored. The isotropy proof assumes an "isotropic gradient prior" on the unknown label function $\eta$ (mean-zero gradient, $\mathbb{E}[\nabla\eta\nabla\eta^\top] = \tau_g^2 I$); if you *do* know the downstream task, the argument no longer says isotropy is optimal. The cross term $\mathbb{E}[A(X)C(X)]$ in the k-NN bias is only zero under an extra uncorrelatedness assumption; otherwise it is $O(r_0^4)$ and merely does not dominate.
- **Vision only.** Every experiment is images. The theory is domain-free but nobody ran text, audio, or robotics.
- **Open question worth chasing.** The loss–accuracy correlation of 99% is the most useful engineering result in the paper. If it holds up, SSL pretraining becomes a normal optimisation problem with a validation signal, not a guessing game requiring labelled probes. Worth checking whether it survives longer schedules and larger data.
- **Practical caveat.** The $\lambda$ default of 0.05 shifts slightly with the number of views — Figure 8 shows peak accuracy at $\lambda \approx 0.01$ for 2 views, $\lambda \approx 0.02$ for 4 views, $\lambda \approx 0.05$ for 8 views. Scale it up as you add views.

## Links

Related: [[An Image is Worth 16x16 Words (ViT)]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Foundation Models]] · [[Mode Collapse]] · [[Regularization]] · [[Fourier Series Decomposition]] · [[KL Divergence]] · [[Derivative]] · [[Random variable]] · [[Uncertainty]] · [[Linear Projection]] · [[Regression Analysis]] · [[Backpropagation]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Loss, Objectives, and Business Alignment]] · [[Saliency]] · [[Distillation]]

New topics worth writing: Joint-Embedding Predictive Architectures, Fisher information, Cramér–Rao bound, Cramér–Wold theorem, characteristic functions, Epps–Pulley normality test, Maximum Mean Discrepancy, sliced Wasserstein distance, dimensional collapse in SSL, VICReg, DINO / DINOv2, I-JEPA, Sobolev smoothness, quasi-Monte Carlo and low-discrepancy sequences, Stochastic Weight Averaging, Spearman rank correlation
