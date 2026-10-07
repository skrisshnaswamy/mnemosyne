---
title: "Cyclical Learning Rates for Training Neural Networks"
authors: ["Leslie N. Smith"]
year: 2015
arxiv: "1506.01186"
url: https://arxiv.org/abs/1506.01186
priority: Good-To-Read
read_on: 2026-09-25
tags: [paper, optimization, vision, theory]
---
## The Core Idea

The learning rate is the single hyperparameter that most determines whether a network trains well. Before this paper the standard recipe was: guess a value, decay it monotonically, and run a grid of experiments to find the best guess. That grid is expensive.

Smith's claim is that you should stop searching for *the* learning rate and instead let it bounce up and down between two bounds for the whole of training. The value goes up linearly, comes back down linearly, repeat. No decay-only schedule, no adaptive per-weight machinery.

The surprising part is that **raising** the learning rate helps, even though it visibly hurts accuracy while it is happening. Test accuracy dips in the middle of each cycle and peaks at the end. Over a full run, the dips are worth it.

> [!NOTE] Cyclical learning rate (CLR)
> A schedule where the global learning rate oscillates linearly between a lower bound `base_lr` and an upper bound `max_lr`, instead of being fixed or monotonically decayed. Costs nothing extra to compute. ^cyclical-lr

Why it did not exist before: "conventional wisdom dictates that the learning rate should be a single value that monotonically decreases." Nobody had checked whether the increases were harmful in aggregate. They aren't.

Two mechanisms are offered, one hand-wavy and one practical:

1. **Saddle points, not local minima, are what slow training** (citing Dauphin et al.). Saddle plateaus have tiny [[Derivative#Gradient|gradients]]. A small learning rate crawls across them. Turning the rate up gets you across faster.
2. The boring but more convincing reason: if your bounds bracket the good value, then a good learning rate is being used *some* of the time, always, without you having to know which one it is.

The second contribution is arguably more useful than the first: the **LR range test**. One short run, learning rate increasing linearly, plot accuracy against learning rate, read the two bounds off the curve. This is where `lr_find` in fast.ai comes from.

On CIFAR-10 it gets the same 81.4% in 25,000 iterations that the tuned baseline needed 70,000 iterations to reach.

## The Methodology

**The triangular policy.** At iteration $t$, with `stepsize` = number of iterations in *half* a cycle:

$$\text{cycle} = \left\lfloor 1 + \frac{t}{2 \cdot \text{stepsize}} \right\rfloor$$
$$x = \left| \frac{t}{\text{stepsize}} - 2 \cdot \text{cycle} + 1 \right|$$
$$\eta_t = \eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot \max(0,\; 1 - x)$$

That is it. Four lines of Lua in the paper. $x$ is a triangle wave in $[0,1]$; $\max(0, 1-x)$ turns it into the up-down ramp. Plugged straight into ordinary SGD:

$$\theta^t = \theta^{t-1} - \eta_t \frac{\partial L}{\partial \theta}$$

**Three variants.**

| Policy | What changes |
|---|---|
| `triangular` | bounds stay fixed forever |
| `triangular2` | the gap $\eta_{\max} - \eta_{\min}$ is **halved at the end of every cycle** |
| `exp_range` | both bounds decay by $\gamma^{\,t}$ each iteration |

`triangular2` is the one most of the headline results use. It is CLR plus a decay envelope.

**Choosing `stepsize`.** One epoch = (training images) / (batch size). CIFAR-10: $50{,}000/100 = 500$ iterations. Set `stepsize` to **2–10 epochs' worth** of iterations. Results are "quite robust" to this — `stepsize = 8·epoch` was only slightly better than `2·epoch`. Run at least 3 cycles, preferably 4+. **Stop at the end of a cycle**, when the rate is at its minimum and accuracy peaks.

> [!NOTE] LR range test
> Run the model for a few epochs with the learning rate increasing linearly from very small to very large. Plot accuracy vs learning rate. Set `base_lr` to where accuracy *starts* rising; set `max_lr` to where the curve gets ragged or turns down. One run replaces a grid search. ^lr-range-test

For CIFAR-10 this gave `base_lr = 0.001`, `max_lr = 0.006`. For AlexNet, the curve said convergence starts near 0.006 and dies above 0.015, so `max_lr = 0.015`. For GoogLeNet, usable range 0.01–0.04, but above 0.025 it converges erratically, so `max_lr = 0.026`.

Fallback rule of thumb if you don't want to read the plot: the best rate is usually within a factor of two of the largest one that converges, so set `base_lr` to $\tfrac{1}{3}$ or $\tfrac{1}{4}$ of `max_lr`.

**The staged CIFAR-10 run** (Table 2) — note this is CLR *and* manual stage drops, not CLR alone:

| base_lr | max_lr | stepsize | start iter | end iter |
|---|---|---|---|---|
| 0.001 | 0.005 | 2,000 | 0 | 16,000 |
| 0.0001 | 0.0005 | 1,000 | 16,000 | 22,000 |
| 0.00001 | 0.00005 | 500 | 22,000 | 25,000 |

## Ablation Studies and Experiments

**The ablation that carries the paper: the `decay` policy.**

The obvious objection is "accuracy only climbs when the rate is falling, so the descent is doing all the work — the ascent is dead weight." Smith tested exactly that. `decay` starts at `max_lr = 0.007`, ramps *down* linearly to `base_lr = 0.001` over 4,000 iterations, then holds flat.

Result: **78.5%** at 25,000 iterations, versus **81.4%** for `triangular2` at the same 25,000 iterations. The down-ramp alone is worse than the baseline's 81.4%. So both directions matter.

**CIFAR-10, Caffe reference architecture (Table 1):**

| Policy | Iterations | Accuracy |
|---|---|---|
| `fixed` | 70,000 | 81.4% |
| `triangular2` | **25,000** | 81.4% |
| `decay` | 25,000 | 78.5% |
| `exp` | 70,000 | 79.1% |
| `exp_range` | 42,000 | **82.2%** |

**ImageNet (Table 1):**

| Net | Policy | Iterations | Accuracy |
|---|---|---|---|
| AlexNet | `fixed` | 400,000 | 58.0% |
| AlexNet | `triangular2` | 400,000 | 58.4% |
| AlexNet | `exp` | 300,000 | 56.0% |
| AlexNet | `exp` | 460,000 | 56.5% |
| AlexNet | `exp_range` | 300,000 | 56.5% |
| GoogLeNet | `fixed` | 420,000 | 63.0% |
| GoogLeNet | `triangular2` | 420,000 | **64.4%** |
| GoogLeNet | `exp` | 240,000 | 58.2% |
| GoogLeNet | `exp_range` | 240,000 | 60.2% |

Read this honestly. On [[ImageNet Classification with Deep CNNs (AlexNet)|AlexNet]] the gain is **+0.4%** and the iteration count is identical — because the Caffe baseline learning rate was already well tuned, which the range test itself confirmed. The +1.4% on GoogLeNet is larger precisely because there was no published baseline and `base_lr = 0.01` was a *guess*. **CLR's value scales with how badly tuned your baseline is.** That is the honest reading of the whole paper.

**Residual family, average of 5 runs (Table 4).** Baselines were run at fixed initial LRs of 0.1, 0.2, 0.3; CLR ranged over 0.1–0.3 with cycle length set to a tenth of the total epoch budget.

| Arch | CIFAR-10 | CIFAR-100 |
|---|---|---|
| ResNet best fixed | 93.3 (0.2) | 71.9 (0.3) |
| ResNet + CLR | **93.6** | **72.5** |
| Stochastic Depth best fixed | 94.6 (0.1) | 75.2 |
| SD + CLR | 94.5 | **75.4** |
| DenseNet best fixed | 94.5 | 75.3 (0.2) |
| DenseNet + CLR | **94.9** | **75.9** |

Margins of 0.2–0.6%, and Stochastic Depth on CIFAR-10 is a **loss** (94.5 vs 94.6). These are averages of 5 runs with no variance reported, so several of these differences are within noise. The defensible claim is "CLR matches or slightly beats the best fixed LR, without you having to find the best fixed LR."

**The biggest single win, and it's easy to miss.** The Caffe CIFAR-10 variant with sigmoid nonlinearities and [[Batch Normalization|batch normalisation]]: fixed LR gets **60.8%**, CLR gets **72.2%**. An 11.4-point gap. This is the one case where CLR isn't a tuning convenience — something about that architecture's loss surface makes the fixed rate genuinely bad.

**What did not work well: combining CLR with adaptive optimisers (Table 3).**

| Optimiser | `fixed`, 70k | `triangular`, 25k | `triangular`, 70k |
|---|---|---|---|
| Nesterov momentum | 82.1% | 81.3% | — |
| Adam | 81.4% | 79.8% | 81.1% |
| RMSprop | 75.2% | 72.8% | 75.1% |
| AdaGrad | 74.6% | **76.0%** | — |
| AdaDelta | 67.3% | 67.3% | — |

Nesterov and AdaGrad get the speedup: same or better accuracy in 25,000 iterations instead of 70,000. Adam and RMSprop **do not** — they still need the full 70,000 to match, and at 25,000 they are 1.6 and 2.4 points behind. AdaDelta is completely unmoved.

The paper's own summary: "When using adaptive learning rate methods, the benefits from CLR are sometimes reduced." Since Adam is what most people actually run, this is the ablation to remember. Adaptive methods already rescale per-weight step sizes, so a global cycle on top has less left to fix.

**Functional form doesn't matter.** Triangular, Welch (parabolic) and Hann (sinusoidal) windows "all produced equivalent results." Triangle was chosen for simplicity. So do not spend time on the shape of the wave.

## Worth Remembering

- **The mechanism is not established.** The saddle-point story is borrowed intuition, not measured. The paper explicitly ends with "we believe that a theoretical analysis would provide an improved understanding." Treat the *why* as an open question and the *what works* as an empirical result.

- **CLR is free; adaptive methods are not.** This is the framing Smith uses throughout — CLR is positioned as a competitor to AdaGrad/RMSprop/[[Adam- A Method for Stochastic Optimization|Adam]] that costs no extra memory or compute. In 2015 that mattered more than it does now.

- **The range test is the durable contribution.** Even if you never cycle, one linear-ramp run tells you your usable LR band for a new architecture or dataset. Smith's own advice: run the range test, then compare fixed-LR against CLR-over-that-range, and use whichever wins for the rest of your experiments.

- **Read the ImageNet numbers as "CLR removes the need to tune", not "CLR beats tuning."** The AlexNet baseline was already tuned and CLR only got +0.4%. That is the null-ish result inside the paper.

- **Peaks land at cycle ends.** If you evaluate mid-cycle you will see worse numbers and think it's broken. Always checkpoint and report at the minimum-LR point.

- **`triangular2` is not pure CLR.** Halving the amplitude each cycle is a decay schedule wearing a cycle costume. The pure `triangular` results (Table 3) are consistently a bit weaker than `triangular2`. Some of the win is the envelope.

- **Direct lineage.** Smith notes CLR "is likely most similar to the SGDR method" (Loshchilov & Hutter, cosine warm restarts), which appeared the same year. This paper's descendant is Smith's own one-cycle / super-convergence work, which keeps the ramp-up and runs exactly one giant cycle. Cosine-with-warmup schedules that everyone now uses for [[Deep Learning|deep nets]] are the surviving form of this family.

- **Untested:** recurrent architectures, anything non-vision. The paper says so.

- **Practical caveat for large-batch work:** CLR's bounds are coupled to batch size. If you change batch size you must re-run the range test — the linear scaling rule in [[Accurate, Large Minibatch SGD- Training ImageNet in 1 Hour]] is the relevant relationship, and warmup there is essentially the up-ramp of one cycle with the down-ramp removed.

## Links

Related: [[Momentum]] · [[Adam- A Method for Stochastic Optimization]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Accurate, Large Minibatch SGD- Training ImageNet in 1 Hour]] · [[Batch Normalization]] · [[How Does Batch Normalization Help Optimization]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Backpropagation]] · [[Bayesian Optimization]] · [[GPU processing]] · [[Regularization]] · [[Old Optimizer, New Norm- An Anthology (Muon)]]

New topics worth writing: SGDR and cosine warm restarts, the one-cycle policy and super-convergence, learning-rate warmup, saddle points and loss-surface geometry, learning-rate schedules as a family, the lr_find tool in fast.ai
