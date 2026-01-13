---
title: "Bootstrap Your Own Latent (BYOL)"
authors: ["Jean-Bastien Grill", "Florian Strub", "Florent Altché", "Corentin Tallec", "Pierre H. Richemond", "Elena Buchatskaya", "Carl Doersch", "Bernardo Avila Pires", "Zhaohan Daniel Guo", "Mohammad Gheshlaghi Azar", "Bilal Piot", "Koray Kavukcuoglu", "Rémi Munos", "Michal Valko"]
year: 2020
arxiv: "2006.07733"
url: https://arxiv.org/abs/2006.07733
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, rl, optimization, self-supervised, vision]
---
## The Core Idea

Before BYOL, the standard recipe for learning image features without labels was contrastive: take one image, make two random crops, pull their representations together, and push apart the representations of *different* images. Those "push apart" pairs — the negatives — were believed to be load-bearing. Without them, nothing stops the network from cheating: output the same constant vector for every image, and the "pull together" loss is perfectly zero. That failure is called collapse.

BYOL removes the negatives entirely and still does not collapse.

The setup is two copies of the same network. The **online** network (weights $\theta$) sees one augmented view. The **target** network (weights $\xi$) sees a second augmented view. The online network must *predict* the target network's output. The target network is never trained by gradients — its weights are just a slow exponential moving average of the online weights:

$$\xi \leftarrow \tau\xi + (1-\tau)\theta$$

Two asymmetries make this work, and both are necessary:

1. The online branch has an extra small MLP on top, the **predictor** $q_\theta$. The target branch does not. So the two branches are not the same function.
2. The gradient flows only into $\theta$. The target is a stop-gradient.

Why does this not collapse? The honest answer in the paper is "we don't fully know, but here is the intuition." The key observation is that BYOL's dynamics are **not gradient descent on any single loss** over $(\theta,\xi)$ jointly — the $\xi$ update is not $-\nabla_\xi \mathcal{L}$, it is an averaging step. So there is no reason for the pair to slide into the constant-output minimum. The authors compare this to [[Generative Adversarial Networks]], where there is also no joint loss being minimised.

The sharper argument: if the predictor were *optimal*, i.e. $q^\star(z_\theta)=\mathbb{E}[z'_\xi \mid z_\theta]$, then the gradient BYOL follows is the gradient of the **conditional variance** $\mathbb{E}[\sum_i \mathrm{Var}(z'_{\xi,i}\mid z_\theta)]$. Conditional variance never increases when you condition on *more* information: $\mathrm{Var}(X\mid Y,Z)\le \mathrm{Var}(X\mid Y)$. So throwing information away out of $z_\theta$ can never help the loss. Collapse is an equilibrium, but an unstable one. The role of the moving average is then reinterpreted: it keeps the target changing slowly enough that the predictor stays close to optimal.

What this unlocks: no memory bank, no huge batches to farm negatives from, and much less sensitivity to which augmentations you pick. ResNet-50 linear probe on ImageNet goes from SimCLR's 69.3% to **74.3%** top-1; a ResNet-200 (2×) reaches **79.6%**, beating the previous best (76.8%) with 30% fewer parameters.

> [!NOTE] Collapse
> The degenerate solution of a self-supervised objective where the encoder maps every input to the same (or a very low-rank) vector. The loss is minimised, the representation is useless. ^collapse

> [!NOTE] Bootstrapping (BYOL sense)
> Using your own current network's outputs as the regression target for training a better version of that same network. Borrowed from the target-network trick in Q-learning, not from statistical resampling. ^bootstrap-target

## The Methodology

**The two pipelines.** Given image $x$, sample two augmentation functions $t\sim\mathcal{T}$, $t'\sim\mathcal{T}'$ to get views $v=t(x)$, $v'=t'(x)$.

- Online: $y_\theta = f_\theta(v)$ → $z_\theta = g_\theta(y_\theta)$ → $q_\theta(z_\theta)$
- Target: $y'_\xi = f_\xi(v')$ → $z'_\xi = g_\xi(y'_\xi)$, then `stop_gradient`

**The loss.** $\ell_2$-normalise both vectors and take squared error, which is just negative cosine similarity up to constants:

$$\mathcal{L}_{\theta,\xi} = \left\|\overline{q_\theta}(z_\theta) - \overline{z}'_\xi\right\|_2^2 = 2 - 2\cdot\frac{\langle q_\theta(z_\theta), z'_\xi\rangle}{\|q_\theta(z_\theta)\|_2\,\|z'_\xi\|_2}$$

The loss is **symmetrised**: also feed $v'$ through the online branch and $v$ through the target branch, and add the two terms. Gradient is taken w.r.t. $\theta$ only.

**Architecture, concretely.**
- Encoder $f_\theta$: ResNet-50 v1, post-activation ([[Deep Residual Learning for Image Recognition (ResNet)]]). Representation $y$ = output of final average pool, dim 2048.
- Projector $g_\theta$: Linear(2048→4096) → [[Batch Normalization|BatchNorm]] → ReLU → Linear(4096→256). Note: **no** BatchNorm on the output, unlike [[A Simple Framework for Contrastive Learning (SimCLR)|SimCLR]].
- Predictor $q_\theta$: identical MLP shape (256→4096→256).
- At the end, throw away everything except $f_\theta$.

**Augmentations.** Exactly SimCLR's set: random resized crop to $224\times224$ (area 8%–100%, log-uniform aspect ratio 3/4–4/3), horizontal flip $p=0.5$, colour jitter $p=0.8$ (brightness/contrast 0.4, saturation 0.2, hue 0.1), grayscale $p=0.2$, Gaussian blur, solarisation. The two views use *asymmetric* probabilities: blur $p=1.0$ for $\mathcal{T}$ vs $p=0.1$ for $\mathcal{T}'$; solarise $p=0.0$ vs $p=0.2$.

**Optimisation.**
- LARS optimiser, cosine decay without restarts, 1000 epochs, 10-epoch warmup.
- Base LR $0.2$, scaled linearly: $\text{LR}=0.2\times\text{BatchSize}/256$.
- Weight decay $1.5\times10^{-6}$, excluding biases and BN parameters from both LARS adaptation and decay.
- Target decay $\tau$ starts at $\tau_{\text{base}}=0.996$ and is annealed **towards 1** over training: $\tau = 1-(1-\tau_{\text{base}})(\cos(\pi k/K)+1)/2$. So the target moves fast early and freezes late.
- Batch size 4096 over 512 TPU v3 cores; ~8 hours for ResNet-50.

A batch-512 recipe also exists (64 TPU cores, ~4 days) and still gets **73.7%** with base LR 0.4 and $\tau_{\text{base}}=0.9995$.

## Ablation Studies and Experiments

All ablations: 300 epochs, LR 0.3, batch 4096, $\tau_{\text{base}}=0.99$, 3 seeds. The 300-epoch BYOL baseline is **72.5%**; the reproduced SimCLR baseline is **67.9%**.

**The central ablation — what actually prevents collapse (Table 5b).** They write a family of losses interpolating BYOL and SimCLR, with $\beta$ weighting the negative-pairs term:

| Predictor | Target net | $\beta$ | Top-1 |
|---|---|---|---|
| ✓ | ✓ | 0 | **72.5** (= BYOL) |
| ✓ | ✓ | 1 | 70.9 |
| ✗ | ✓ | 1 | 70.7 |
| ✗ | ✗ | 1 | 69.4 (= SimCLR) |
| ✓ | ✗ | 1 | 69.1 |
| ✗ | ✓ | 0 | **0.3** |
| ✓ | ✗ | 0 | **0.2** |
| ✗ | ✗ | 0 | **0.1** |

Read the bottom three rows. With $\beta=0$ (no negatives), predictor *alone* collapses, target network *alone* collapses. Only **both together** works. This is the whole paper in one table.

Two side findings from the same table: (a) bolting a target network onto SimCLR alone buys +1.6 points, which reframes what [[Momentum Contrast (MoCo)|MoCo]]'s momentum encoder is doing — it is not just a cheap negative bank, it is a stabiliser; (b) adding the predictor to SimCLR barely matters.

**Adding negatives back to BYOL hurts** — 72.5 → 70.9 — but only because the temperature was left at SimCLR's optimum ($\alpha=0.1$). A full sweep over $(\alpha,\beta)$ (Table 18) shows you can match BYOL at 72.7 with $\beta>0$ if you retune $\alpha$ to $\ge 0.3$. So negatives are not harmful, just unnecessary.

**Target decay rate (Table 5a).**

| Target | $\tau_{\text{base}}$ | Top-1 |
|---|---|---|
| Frozen random network | 1 | 18.8 |
| EMA | 0.999 | 69.8 |
| EMA | 0.99 | **72.5** |
| EMA | 0.9 | 68.4 |
| Instant copy (stop-grad only) | 0 | 0.3 |

The frozen-random-target result is the seed of the whole idea: predicting a *random* frozen network gives 18.8%, while the random network itself gives 1.4%. Prediction alone bootstraps something out of nothing.

**Batch size.** BYOL is flat from 4096 down to 256 (72.5 → 71.8). SimCLR degrades much faster (67.9 → 64.3, with a ±2.1 spread). At 64 both crash, which the authors blame on BatchNorm misbehaving at tiny batch, not on the objective.

**Augmentation robustness.** Remove colour distortion: BYOL loses 9.1 points, SimCLR loses 22.2. Crop-only: BYOL 59.4% (−13.1), SimCLR 40.3% (−27.6). The explanation is nice — crops of one image share a colour histogram, so a contrastive task with crop-only can be *solved* by reading colour histograms, and the representation learns nothing else. BYOL has no discrimination task to shortcut; it is always pushed to encode whatever the target encoded. Compare [[Shortcut Learning in Deep Neural Networks]].

**Mean Teacher check.** Removing the predictor from BYOL gives an unsupervised Mean Teacher. It collapses. So the semi-supervised literature's teacher–student consistency loss only survives because a classification loss was grounding it.

**Removing the target network entirely (Appendix I).** You *can*, if the predictor stays near-optimal:
- Optimal linear predictor solved in closed form per batch, $q^\star=(Z_\theta^\top Z_\theta)^{-1}Z_\theta^\top Z'_\xi$: **52.5%**.
- Just raise the predictor's learning rate by $\lambda$: $\lambda=1$ → 5.5%, $\lambda=10$ → 66.6%, $\lambda=20$ → 66.3%.
- Raise predictor *and* projector LR together: ~25%, collapse-adjacent.

Table 22 makes the pattern crisp: performance is decent only when $\lambda_{\text{pred}} > \mu_{\text{proj}}$. This is the strongest empirical support for "the target net exists to keep the predictor near-optimal."

**Normalisation in the loss (Table 20).** $\ell_2$: 72.5. LayerNorm: 72.5. No normalisation: 67.4 (and the projection norm blows up to $\sim 3\times10^6$ and then plateaus — it does not diverge). BatchNorm: 65.3, the worst.

**Hyperparameter sensitivity.** Projector and predictor both depth 2 is best (depth 1 for the projector costs ~10 points: 61.9 vs 72.5). Projection dim is flat from 128 to 512; only dim 16 hurts (69.9). LR 0.2–0.4 all fine. **Setting weight decay to zero makes both BYOL and SimCLR diverge** — worth knowing.

**Downstream numbers.**
- Semi-supervised, ResNet-50, 1% labels: 53.2 top-1 vs SimCLR 48.3. 10%: 68.8 vs 65.6.
- Fine-tuned on 100% ImageNet: 77.7 vs a from-scratch baseline of 76.5. SimCLR initialisation gives *no* gain over random init here; BYOL does.
- Linear transfer to 12 datasets: beats SimCLR on all 12, beats a supervised ImageNet encoder on 7 of 12.
- VOC detection AP50 77.5 vs supervised 74.4 and SimCLR 75.2. VOC segmentation 76.3 mIoU vs 74.4 / 75.2. NYU depth pct<1.25: 84.6 vs 81.1 / 83.3.
- Pretrained on Places365 instead of ImageNet, BYOL still beats SimCLR on every transfer task, though it loses to ImageNet-pretrained BYOL everywhere except Places365 and SUN397.

**What did not work / limits.** Big architectures overfit during semi-supervised fine-tuning — ResNets with the same depth but a larger width multiplier score *worse*, with lower train loss and higher validation loss. Regularisation is needed there. And BYOL never beats the strongest supervised baseline (78.9% with MaxUp) on ResNet-50.

## Worth Remembering

- The authors are unusually honest that they have a **hypothesis**, not a proof. "We did not observe convergence to such equilibria in our experiments" is the actual claim about collapse. Later work (SimSiam) showed you can drop the EMA entirely if you keep the stop-gradient and predictor, which fits the "predictor near-optimality" story better than the "EMA is essential" story.
- The conditional-variance argument (Eq. 5) relies on the envelope theorem: at an optimal predictor, $\mathbb{E}[\partial L/\partial q \cdot \partial q^\star/\partial\theta]=0$, so the only surviving gradient path is through the predictor's *input*. That is exactly what BYOL's autograd computes. Neat, but it only holds *at* optimality — which is why keeping the predictor near-optimal keeps showing up in the ablations.
- **Practical caveat:** BatchNorm inside the projector was later shown by follow-up blog work to be doing implicit contrasting (it centres against the batch). This paper's own Table 20 shows BatchNorm *in the loss* hurts, but BatchNorm *in the projector MLP* is kept. Do not remove it casually.
- Weight decay is not optional. Zero weight decay diverges. This is a general lesson for self-supervised training, not a BYOL quirk.
- The evaluation protocol matters: swapping in pre-train augmentations plus logit clipping for the linear probe lifts BYOL from 74.3 → 74.8 without touching the encoder. Always check what protocol a reported number came from — compare [[On the Difficulty of Evaluating Baselines]].
- Admitted limitation: BYOL depends on hand-designed vision augmentations. Moving to audio, video or text needs a new augmentation set per modality, and the authors flag automating that search as the next step. [[Self-Supervised Learning from Images with I-JEPA]] and [[LeJEPA- Provable and Scalable Self-Supervised Learning]] are later attacks on the same problem from a different angle.
- The lineage is explicitly reinforcement learning: the slow target network comes from DQN and DDPG ([[Playing Atari with Deep Reinforcement Learning (DQN)]], [[Continuous control with deep reinforcement learning (DDPG)]]), and the direct ancestor is PBL, a representation-learning method for RL agents.

## Links

Related: [[A Simple Framework for Contrastive Learning (SimCLR)]] · [[Momentum Contrast (MoCo)]] · [[Self-Supervised Learning from Images with I-JEPA]] · [[LeJEPA- Provable and Scalable Self-Supervised Learning]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Understanding Contrastive Learning through Alignment and Uniformity]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Batch Normalization]] · [[Layer Normalization]] · [[Continuous control with deep reinforcement learning (DDPG)]] · [[Playing Atari with Deep Reinforcement Learning (DQN)]] · [[Generative Adversarial Networks]] · [[Shortcut Learning in Deep Neural Networks]] · [[Mode Collapse]] · [[Regularization]] · [[SimCSE- Simple Contrastive Learning of Sentence Embeddings]]

New topics worth writing: LARS optimizer, SimSiam, Mean Teacher and consistency regularisation, exponential moving average of weights (Polyak averaging), linear evaluation protocol, envelope theorem, conditional variance, DINO and self-distillation without labels
```
