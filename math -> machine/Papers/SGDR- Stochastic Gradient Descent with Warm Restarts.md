---
title: "SGDR: Stochastic Gradient Descent with Warm Restarts"
authors: ["Loshchilov & Hutter"]
year: 2016
arxiv: "1608.03983"
url: https://arxiv.org/abs/1608.03983
priority: Good-To-Read
read_on: 2026-09-25
tags: [paper, optimization, vision, scaling]
---
## The Core Idea

Training a deep network needs a learning-rate schedule, and in 2016 the standard one was a staircase: hold the rate constant, then divide it by a constant at a few hand-picked epochs. Zagoruyko & Komodakis's Wide ResNet results used exactly this — start at $0.1$, multiply by $0.2$ at epochs 60, 120 and 160, stop at 200.

Two things are wrong with a staircase. First, you must decide the total budget up front, because the drop points are fractions of it. Second, its **anytime performance** is bad — if you stop at epoch 90 you get something much worse than the final answer, so you cannot cheaply ask "is this architecture promising?"

SGDR replaces the staircase with a smooth cosine decay from a high rate down to (almost) zero, and then **jumps the learning rate back up to the top and does it again**. No reinitialisation of weights, no reset of momentum — the only thing that "restarts" is $\eta_t$. The parameter vector $x_t$ carries straight on.

> [!NOTE] Warm restart ^warm-restart
> Raise the learning rate back to a high value mid-training while keeping the current weights. "Warm" because the search continues from where it was; only the step size is reset. Contrast with a cold restart, which throws the weights away.

Why that helps is the honest weak spot of the paper, and the authors say so: they explicitly **do not** claim the restarts are escaping local minima or exploiting multimodality. What they measure is that each cosine cycle ends at a good point, so you have a usable model at epoch 10, 30, 70, 150, … rather than only at 200. You get a sequence of good models instead of one.

Two things fell out of that which mattered more than the speedup:

1. The cosine decay is now the **default schedule in deep learning**, restarts or not. On CIFAR-100 a single cosine run with no restart at all beat the tuned staircase.
2. Because each cycle ends in a different basin, the snapshots taken at the end of each cycle disagree with each other in useful ways. Averaging their softmax outputs gives you an ensemble for **zero extra compute** — this became "Snapshot Ensembles" (Huang et al.), and it is where the headline numbers come from.

## The Methodology

The whole method is one equation. Inside run $i$, at batch step $t$:

$$\eta_t = \eta^i_{min} + \tfrac{1}{2}\left(\eta^i_{max} - \eta^i_{min}\right)\left(1 + \cos\left(\frac{T_{cur}}{T_i}\pi\right)\right)$$

- $T_i$ — length of run $i$, in epochs.
- $T_{cur}$ — epochs since the last restart. Updated **every batch**, so it takes fractional values (0.1, 0.2, …) and the learning rate moves smoothly, not in per-epoch steps.
- At $T_{cur}=0$: $\cos 0 = 1$, so $\eta_t = \eta^i_{max}$.
- At $T_{cur}=T_i$: $\cos \pi = -1$, so $\eta_t = \eta^i_{min}$.

In the experiments $\eta_{min} = 0$ and $\eta_{max} = 0.05$, held fixed across all restarts to keep the hyperparameter count down.

**Restart scheduling.** Two knobs: $T_0$ (length of the first run) and $T_{mult}$ (multiply the run length by this at each restart).

- $T_{mult}=1$ → equal-length cycles. They tried $T_0 \in \{50, 100, 200\}$ inside a 200-epoch budget.
- $T_{mult}=2$ → doubling cycles. $T_0=1$ gives restarts at epochs 1, 3, 7, 15, 31, 63, 127; $T_0=10$ gives 10, 30, 70, 150. Doubling exists purely for anytime performance: you get a finished model very early, then progressively better ones.

**Which model do you report?** Not the latest $x_t$, because a restart temporarily wrecks performance. The recommended model is always the one at the *end* of the most recently completed run, where $\eta_t = \eta^i_{min}$. Nice practical consequence: no validation split needed to pick a checkpoint.

**Setup.** Wide ResNets (WRN-$d$-$k$: depth $d$, $k\times$ the filters of a standard [[Deep Residual Learning for Image Recognition (ResNet)|ResNet]]) on CIFAR-10 and CIFAR-100. WRN-28-10 is 36.5M params; WRN-28-20 is 145.8M. Plain SGD with [[Momentum|momentum]] $0.9$ (they used classical momentum, *not* Nesterov, unlike the WRN paper), weight decay $5\times10^{-4}$, batch size 128, 200 epochs. Data augmentation: horizontal flips and random crops from 4-pixel reflection padding. They subtracted the per-pixel mean and skipped the ZCA whitening the WRN paper used.

**Snapshot ensembling.** Run SGDR with $T_0=10, T_{mult}=2$. Save the weights at epochs 30, 70 and 150 — the three cycle ends. At test time average the softmax probabilities of those $M$ models with uniform weights. Do this across $N$ independent runs for $N \times M$ total members.

## Ablation Studies and Experiments

**Single model, WRN-28-10, median of 5 runs (test error %, CIFAR-10 / CIFAR-100):**

| Schedule | C-10 | C-100 |
|---|---|---|
| staircase, $\eta_0 = 0.1$ (reproduces WRN paper) | 4.24 | 20.33 |
| staircase, $\eta_0 = 0.05$ | 4.13 | 20.21 |
| $T_0=50, T_{mult}=1$ | 4.17 | 19.99 |
| $T_0=100, T_{mult}=1$ | 4.07 | 19.87 |
| $T_0=200, T_{mult}=1$ (**no restart at all**) | **3.86** | 19.98 |
| $T_0=1, T_{mult}=2$ | 4.09 | 19.74 |
| $T_0=10, T_{mult}=2$ | 4.03 | **19.58** |

**The most important row is the one with no restarts.** $T_0=200, T_{mult}=1$ is a single cosine decay over the whole budget, and it gives the best CIFAR-10 number in the table. So on final accuracy, the win is the *cosine shape*, not the restarting. The restarts buy anytime performance: the $T_{mult}=2$ variants reach ~4% on CIFAR-10 and ~20% on CIFAR-100 **2–4× earlier in wall-clock** than the staircase. Correspondingly, $T_0=200$ has the *worst* anytime curve of all the SGDR variants until the very last epochs.

**WRN-28-20 (145.8M params, median of 2):** best numbers are $T_0=200$ → 3.66 on CIFAR-10, and $T_0=10,T_{mult}=2$ → 18.70 on CIFAR-100. The wider net costs 3–4× the compute per epoch, but because SGDR is good early, they reached a *better* CIFAR-100 error on WRN-28-20 in 50 epochs (below 19%) than any schedule managed on WRN-28-10 in 200 (nothing beat 19.5%).

**Ensembles, WRN-28-10 with $T_0=10, T_{mult}=2$:**

| Config | C-10 | C-100 |
|---|---|---|
| $N=1, M=1$ (median of 16 runs) | 4.03 | 19.57 |
| $N=1, M=3$ — free ensemble | 3.51 | 17.75 |
| $N=3, M=3$ | 3.25 | 16.64 |
| $N=16, M=3$ | **3.14** | **16.21** |

Three findings hide in that table:

- $N=1, M=3$ matches $N=3, M=1$ — i.e. three snapshots from one run are worth three fully independent runs. A **3× speedup**, since extra snapshots cost nothing.
- Snapshot diversity beats run diversity at fixed member count: $N=3, M=3$ (9 members) beats $M=1$ with $N=18$ and even $N=21$.
- **What did not work:** taking $M=3$ snapshots at epochs 148, 149, 150 gave no improvement over a single model. Three consecutive checkpoints are the same model. The diversity comes specifically from the learning-rate cycling, which confirms Huang et al.'s claim rather than just restating it.

**Overfitting (Figure 7).** SGDR drives the training loss down faster than the staircase up to about epoch 120. After that, the staircase's test error starts *rising* on both CIFAR sets — it overfits. SGDR shows only very mild overfitting. Unexpected, and unexplained in the paper.

**EEG recordings** (14 subjects, ~1000 trials each, right/left hand and foot movement). The reference CNN pipeline's median error is around 9%. SGDR matches the tuned baseline's final error while having a better anytime curve and needing no pre-declared epoch budget. Snapshots cut error 1–2% within one run, 2–3% when pooling snapshots across two learning-rate settings.

**Downsampled ImageNet** (all 1000 classes at 32×32 — the dataset that later became ImageNet32). WRN-28-10, SGDR with $T_0=10, T_{mult}=2, \eta_{max}=0.01$: top-1 39.24%, top-5 17.17%. That roughly matches [[ImageNet Classification with Deep CNNs (AlexNet)|AlexNet]]'s 40.7% / 18.2% on full-resolution ImageNet with ~50× more pixels per image. Across four initial learning rates $\{0.05, 0.025, 0.01, 0.005\}$, SGDR won on anytime performance every time. The authors' reading: because the cosine sweeps the rate all the way down from $\eta_{max}$ to 0, SGDR **partly absorbs a bad choice of initial learning rate** — it visits every scale on the way down.

**Appendix check that matters.** Their augmentation code appended flipped images, making an "epoch" 100k examples, not 50k. Since every schedule is defined in epoch units, this changes what $T_0=50$ means. They re-ran WRN-28-1 with genuine 50k epochs and swept learning rates ($\{0.01, 0.025, 0.05, 0.1\}$ for the staircase, $\{0.025, 0.05, 0.1\}$ for SGDR). Conclusion held.

## Worth Remembering

**The paper's own hedge.** Section 5 states plainly: *"we do not claim that we observe any effect related to multi-modality."* The restart framing is borrowed from gradient-free optimisation (CMA-ES restarts, O'Donoghue & Candès's adaptive restarts for Nesterov schemes), but no mechanism is demonstrated here. Treat "restarts escape local minima" as folklore, not as a result from this paper. The demonstrated effects are: anytime performance, ensemble diversity, and less late-stage overfitting.

**Cosine decay outlived restarts.** Nearly every modern LLM and vision recipe uses linear warmup followed by cosine decay to a small floor — one cycle, no restart. That half of SGDR won completely. SGDR-with-restarts survives mainly in the [[Cyclical Learning Rates for Training Neural Networks|cyclical learning rate]] lineage and in snapshot ensembling.

**Two hyperparameters, not five.** With $T_{mult}=1$ and $\eta_{min}=0$ you specify only $\eta_{max}$ and the budget. Compare the staircase, which needs an initial rate, a decay factor, and a list of drop epochs.

**It coexists with, not replaces, warmup.** SGDR restarts jump *up* to a high rate instantly. That is the opposite of the [[Accurate, Large Minibatch SGD- Training ImageNet in 1 Hour|gradual warmup]] used for large-batch training, which ramps up slowly to avoid early divergence. Modern recipes do warmup first, then cosine; they do not warm-restart. Unresolved in this paper whether restarts are safe at large batch sizes.

**Related-authors note.** Loshchilov & Hutter went on to write [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], and that paper's motivation is exactly the loose end left here — SGDR works beautifully with SGD+momentum but interacts badly with [[Adam- A Method for Stochastic Optimization|Adam]]'s L2 regularisation, because coupled weight decay scales with the learning rate and therefore changes strength across a cosine cycle. The last line of the conclusion here ("future work should consider warm restarts for Adam") is literally the seed of AdamW.

**Practical caveats.**

- Define your schedule in **steps, not epochs**, if your dataset size or augmentation factor can change. The 50k/100k confusion in the appendix is exactly this bug.
- Never report the latest checkpoint under SGDR. Report the end of the last completed cycle. Mid-cycle models are much worse.
- $\eta_{min} = 0$ means the last few hundred steps of a cycle do essentially nothing. Many implementations use a small floor instead.
- Snapshot ensembles need $M\times$ the weight storage and $M\times$ the inference cost. Free to train, not free to serve — check against [[Distillation|distillation]] of the ensemble into one model.
- All results here are median-of-5 or median-of-2 with no significance tests, and the authors say the bolding is not statistically justified. The 3.86 vs 4.13 gap on CIFAR-10 is real but small relative to seed variance.

**Open question worth chasing.** Why does SGDR overfit less? Is it that the repeated high-learning-rate phases act as an implicit regulariser (noise injection knocking the weights out of sharp minima), or is it just that the recommended model is always taken at a cycle end rather than after a long low-rate grind? The paper measures the effect and does not explain it.

## Links

Related: [[Cyclical Learning Rates for Training Neural Networks]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Adam- A Method for Stochastic Optimization]] · [[Momentum]] · [[Accurate, Large Minibatch SGD- Training ImageNet in 1 Hour]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Regularization]] · [[GPU processing]] · [[Dropout- A Simple Way to Prevent Overfitting]] · [[Distillation]]

New topics worth writing: Cosine annealing schedule, Snapshot Ensembles, Wide Residual Networks, Learning-rate warmup, One-cycle policy, ImageNet32 / downsampled ImageNet, Anytime performance as an evaluation criterion
