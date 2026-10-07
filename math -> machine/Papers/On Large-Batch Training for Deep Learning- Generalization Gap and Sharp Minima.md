---
title: "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"
authors: ["Keskar et al."]
year: 2016
arxiv: "1609.04836"
url: https://arxiv.org/abs/1609.04836
priority: Good-To-Read
read_on: 2026-09-25
tags: [paper]
---
## The Core Idea

If you train the same network with big batches instead of small batches, it reaches the same low training loss but does worse on test data. Keskar et al. measured this gap — up to **5 percentage points** of test accuracy — and offered an explanation for *why*, not just a fix.

The explanation: small batches and large batches end up in **geometrically different places** in weight space. Large-batch training lands in a "sharp" minimum — a narrow pit where the loss shoots up if you nudge the weights a little. Small-batch training lands in a "flat" minimum — a wide basin where nudging the weights barely changes the loss.

Why sharpness would hurt test accuracy: the training loss surface and the test loss surface are slightly shifted copies of each other (you drew them from different samples). In a wide flat basin, a small horizontal shift costs you almost nothing. In a narrow pit, the same shift can drop you off a cliff. The paper's Figure 1 is exactly this picture.

> [!NOTE] Sharp vs flat minimum
> A **flat minimiser** is a point where the loss stays roughly constant over a large neighbourhood of the weights — the Hessian $\nabla^2 f(x)$ has many small eigenvalues. A **sharp minimiser** has many large positive eigenvalues, so the loss rises steeply nearby. ^sharp-vs-flat

The mechanism they propose for *how* batch size decides this: gradient noise. A small batch gives a noisy gradient estimate, and that noise keeps kicking the iterate out of narrow pits. It can only settle where the noise is not enough to eject it — i.e. a wide basin. A large batch gives an accurate gradient, so nothing kicks it out, and it converges to whatever pit is nearest the start.

Why this mattered in 2016: batch size is the main lever for [[Distributed Training|data parallelism]]. If you cannot raise the batch, you cannot use more GPUs, because the per-step work is too small to spread. So "large batch generalises worse" was a hard ceiling on parallel scaling.

## The Methodology

**The setup.** Six classifiers, all trained with [[Adam- A Method for Stochastic Optimization|ADAM]], cross-entropy loss, five random seeds each, no epoch budget — trained until the loss stopped improving.

| Name | Type | Data |
|---|---|---|
| $F_1$ | 5-layer MLP, 512 units, BN, ReLU | MNIST |
| $F_2$ | 7-layer MLP, 512 units, 1973 classes | TIMIT |
| $C_1$, $C_3$ | shallow conv (AlexNet-like) | CIFAR-10 / 100 |
| $C_2$, $C_4$ | deep conv (VGG-like) | CIFAR-10 / 100 |

- **Small batch (SB)** = 256 examples.
- **Large batch (LB)** = 10% of the whole training set. For CIFAR that is 5,000; for TIMIT, ~72,000.
- Crucially: **the learning rate was not re-tuned between the two regimes.** Remember this; it is the paper's weak point.

**Evidence 1 — parametric plots.** Take the SB solution $x_s^\star$ and the LB solution $x_\ell^\star$. Walk along the straight line between them and plot the loss:

$$f(\alpha x_\ell^\star + (1-\alpha) x_s^\star), \quad \alpha \in [-1, 2]$$

At $\alpha=0$ (SB) the curve is a wide shallow bowl. At $\alpha=1$ (LB) it is a narrow spike. They repeat it on a curved path, $f(\sin(\tfrac{\alpha\pi}{2})x_\ell^\star + \cos(\tfrac{\alpha\pi}{2})x_s^\star)$, and get the same picture.

**Evidence 2 — a sharpness number.** Computing the [[Derivative#Hessian|Hessian]] spectrum is far too expensive, so they define a proxy: how much can the loss climb inside a small box around the solution?

$$\phi_{x,f}(\epsilon, A) := \frac{\left(\max_{y \in \mathcal{C}_\epsilon} f(x + Ay)\right) - f(x)}{1 + f(x)} \times 100$$

In words: **maximise the loss in a tiny neighbourhood, and report the percentage rise.** Pieces:

- $A \in \mathbb{R}^{n \times p}$ is a random matrix. Setting $A = I_n$ searches the whole weight space; setting $p = 100$ searches a random 100-dimensional slice, which guards against the box happening to contain one freak direction.
- $\mathcal{C}_\epsilon$ is a box whose side in coordinate $i$ scales with $\epsilon(|(A^+x)_i| + 1)$ — so the probe is relative to the weight magnitudes, not absolute. This makes it comparable across networks of different scale.
- The inner maximisation is solved badly on purpose: **10 iterations of L-BFGS-B**, because each evaluation of $f$ is a full pass over the training set.
- $\epsilon \in \{10^{-3}, 5\times10^{-4}\}$.

For small $\epsilon$ and $A = I_n$, this number tracks the **largest eigenvalue** of the Hessian. With random $A$ it approximates a Ritz value of the Hessian projected onto $A$'s column space.

## Ablation Studies and Experiments

**The generalisation gap (Table 2).** Training accuracy is ~99% in both regimes everywhere. Test accuracy is not:

| | SB test | LB test | gap |
|---|---|---|---|
| $F_1$ MNIST | 98.03% | 97.81% | 0.2 |
| $F_2$ TIMIT | 64.02% | 59.45% | **4.6** |
| $C_1$ CIFAR-10 | 80.04% | 77.26% | 2.8 |
| $C_2$ CIFAR-10 | 89.24% | 87.26% | 2.0 |
| $C_3$ CIFAR-100 | 49.58% | 46.45% | 3.1 |
| $C_4$ CIFAR-100 | 63.08% | 57.81% | **5.3** |

**Not overfitting.** The test curves never peak and then decay — they rise and flatten in both regimes. So [[Regularization|early stopping]] cannot close this gap. That is an important negative: it rules out the boring explanation.

**Sharpness (Table 3, full space, $\epsilon = 10^{-3}$).** One to two **orders of magnitude** apart:

| | SB | LB |
|---|---|---|
| $F_1$ | $1.23 \pm 0.83$ | $205 \pm 70$ |
| $F_2$ | $1.39 \pm 0.02$ | $311 \pm 38$ |
| $C_1$ | $28.6 \pm 3.1$ | $707 \pm 43$ |
| $C_2$ | $8.7 \pm 1.3$ | $925 \pm 38$ |

The random-subspace version (Table 4) shows the same separation, e.g. $F_1$: $0.11$ vs $9.22$. So the sharpness is not one lucky direction.

**Sharpness is anisotropic.** Sampling around LB solutions, the loss rises steeply along only about **5% of directions**. The rest are flat. So a "sharp minimum" is not a cone — it is a flat plate with a few steep walls.

**The batch-size threshold (Figure 4).** Test accuracy is roughly flat as you raise batch size, then falls off a cliff — at about **500 for $C_1$** and **15,000 for $F_2$**. Sharpness rises across the same region and then plateaus. So there is a critical batch size, and it is problem-specific, not universal.

**The warm-start (piggyback) experiment — the most informative one.** Train with SB for 100 epochs, saving the weights after every epoch. From each of those 100 checkpoints, run LB training for 100 epochs. Then plot the final LB test accuracy against how many SB epochs it was warm-started from.

Result: warm-starting from the first few epochs gives no benefit — LB still ends up sharp and generalises badly. Past a certain number of SB epochs, the LB result suddenly matches SB accuracy, and its sharpness drops. Reading: SB spends its early epochs **exploring** and eventually finds a flat basin; once it is inside one, LB can happily converge within it. LB's problem is exploration, not convergence.

**Distance from initialisation.** $\|x_s^\star - x_0\| / \|x_\ell^\star - x_0\|$ is between **3 and 10**. Small batches travel far; large batches zoom in on what is nearby.

**Sharpness vs loss trajectory (Figure 6).** Early on (high loss) both regimes have similar sharpness. As loss falls, LB sharpness climbs fast; SB sharpness stays flat for a while then *decreases*. Again: exploration phase, then descent into a flat basin.

### What did not work

Three attempted fixes, all in Appendix E, all with the same verdict: **test accuracy improves a bit, sharpness stays high.**

1. **Aggressive [[Regularization|data augmentation]]** (horizontal flips, rotations up to $10°$, translations up to 0.2 of image size). LB now matches augmented SB — e.g. $C_2$: 90.26% LB vs 89.82% SB baseline; $C_4$ actually beats it, 65.88% vs 63.05%. But LB sharpness is still 468 ($C_2$, $\epsilon=10^{-3}$). The accuracy came from the regulariser, not from fixing the geometry.
2. **Conservative (proximal) training** — solve $x_{k+1} = \arg\min_x \frac{1}{|B_k|}\sum_{i\in B_k} f_i(x) + \frac{\lambda}{2}\|x - x_k\|^2$ with 3 ADAM steps and $\lambda = 10^{-3}$. Mostly *hurt*: $F_2$ 61.94% vs 64.02% SB, $C_3$ 45.98% vs 49.58%. Sharpness remained in the hundreds.
3. **Adversarial training** and **stability training** (Zheng et al. 2016) — "no generalization benefit". Test accuracy, sharpness and parametric plots all looked like the untouched baseline. This is a genuinely surprising null, since robustness to input perturbation and flatness in weight space sound like they should be linked (Shaham et al. prove an equivalence for solution-vs-data robustness).

True robust optimisation, $\min_x \max_{\|\Delta x\| \le \epsilon} f(x + \Delta x)$ — literally rolling an $\epsilon$-disc down the loss surface instead of a point — is the principled fix, but each step is a large-scale second-order cone program, so they never ran it.

## Worth Remembering

**The big caveat, and it is big: the learning rate was never re-tuned.** Within a year, [[Accurate, Large Minibatch SGD- Training ImageNet in 1 Hour|Goyal et al.]] trained ImageNet with batch 8,192 and *no* accuracy loss, using the **linear scaling rule** (scale LR with batch size) plus **gradual warmup**. That strongly suggests much of the gap Keskar et al. measured was an under-tuned-step-size artefact, not an inevitable property of large batches. Current best understanding: there *is* a critical batch size past which returns diminish, but it is far larger than this paper's experiments imply, and it moves with the optimiser and schedule.

**The second caveat: sharpness as defined here is not reparameterisation-invariant.** Dinh et al. (2017) showed you can rescale the weights of a ReLU network — leaving the function, and therefore the test error, completely unchanged — while making the Hessian arbitrarily sharp. So a large $\phi$ does not by itself imply bad generalisation. The metric here does normalise the box size by weight magnitude, which helps, but does not fully fix this.

**What still holds up.** The *empirical* pairing of large batch with high curvature and short distance from initialisation has been reproduced many times. And the flat-minima framing directly produced a family of methods: Entropy-SGD (Chaudhari et al., cited here), and later SAM (Sharpness-Aware Minimisation), which actually solves an approximation of that robust objective the authors said was too expensive.

**Practical caveats if you are scaling batch size today.**
- Scale the learning rate and add warmup before you conclude anything about generalisation. That is the null hypothesis.
- The warm-start result is an actionable recipe: small batch early (exploration), large batch later (convergence). This is the **dynamic sampling** idea the authors flag as the most promising remedy — grow the batch as training proceeds.
- Find your critical batch size empirically. It was 500 for a shallow CIFAR conv net and 15,000 for a TIMIT MLP — a 30× spread across two models in the same paper.
- Note the [[Distributed Training|scaling arithmetic]] in Appendix C: with small-batch parallel efficiency $f_s(P) = 0.2$ and $B_s/B_\ell = 0.1$, the large-batch method must converge in at most **half** the iterations to actually be faster. Large batch only wins if the step count does not blow up.

**Open questions the authors name, still largely open:** can you prove LB converges to sharp minima? What is the relative *density* of sharp vs flat minima? Can you design an architecture or initialisation that suits large batches?

## Links

Related: [[Accurate, Large Minibatch SGD- Training ImageNet in 1 Hour]] · [[Adam- A Method for Stochastic Optimization]] · [[Understanding Deep Learning Requires Rethinking Generalization]] · [[Reconciling Modern ML Practice and the Bias-Variance Trade-off]] · [[Deep Double Descent- Where Bigger Models and More Data Hurt]] · [[Distributed Training]] · [[Regularization]] · [[Derivative]] · [[GPU processing]] · [[Collective Communication]] · [[SGDR- Stochastic Gradient Descent with Warm Restarts]] · [[Cyclical Learning Rates for Training Neural Networks]] · [[Dropout- A Simple Way to Prevent Overfitting]] · [[Batch Normalization]] · [[Old Optimizer, New Norm- An Anthology (Muon)]] · [[The Lottery Ticket Hypothesis]] · [[Momentum]]

New topics worth writing: Sharpness-Aware Minimisation (SAM), critical batch size and gradient noise scale, Entropy-SGD, reparameterisation invariance of flatness measures (Dinh et al. 2017), loss landscape visualisation methods, dynamic batch-size schedules, L-BFGS-B, minimum description length and generalisation
