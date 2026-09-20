---
title: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
authors: ["Priya Goyal", "Piotr Dollár", "Ross Girshick", "Pieter Noordhuis", "Lukasz Wesolowski", "Aapo Kyrola", "Andrew Tulloch", "Yangqing Jia", "Kaiming He"]
year: 2017
arxiv: "1706.02677"
url: https://arxiv.org/abs/1706.02677
priority: Must-Read
read_on: 2026-09-17
tags: [paper, vision]
---
## The Core Idea

You want to train faster, so you split each [[Backpropagation|gradient]] step across 256 GPUs. Each GPU handles 32 images, so your minibatch is now 8192 images instead of 256. The question that had blocked this for years: does the model get worse?

The answer here is no — as long as you fix two things, and neither needs a hyper-parameter search.

**1. The linear scaling rule.** Multiply the batch size by $k$, multiply the learning rate by $k$. Nothing else changes. Weight decay, epochs, schedule, momentum all stay put.

**2. Gradual warmup.** For the first 5 epochs, ramp the learning rate linearly from the old small-batch value up to the new big one. Only then start the normal schedule.

With those, ResNet-50 on ImageNet trains in **1 hour on 256 Tesla P100s** at 23.74% ± 0.09 top-1 error, versus 23.60% ± 0.12 for the 256-batch baseline that takes 29 hours on 8 GPUs. The gap is inside the noise.

Why the rule should work, informally. Take $k$ small steps of size $n$ with learning rate $\eta$:

$$w_{t+k} = w_t - \eta\frac{1}{n}\sum_{j<k}\sum_{x\in\mathcal{B}_j}\nabla l(x, w_{t+j})$$

versus one big step over the union of those batches with rate $\hat\eta$:

$$\hat w_{t+1} = w_t - \hat\eta\frac{1}{kn}\sum_{j<k}\sum_{x\in\mathcal{B}_j}\nabla l(x, w_t)$$

The only difference is that the small-batch version evaluates gradients at $w_{t+j}$ (weights that have already moved) and the big one at $w_t$. **If** $\nabla l(x,w_t) \approx \nabla l(x,w_{t+j})$, then setting $\hat\eta = k\eta$ makes the two updates match. That assumption is exactly what breaks in the first few epochs, when the weights are moving fast — hence warmup.

> [!NOTE] Linear scaling rule
> When the minibatch size is multiplied by $k$, multiply the learning rate by $k$ and leave every other hyper-parameter alone. ^linear-scaling-rule

> [!NOTE] Gradual warmup
> Start at the *small-batch* learning rate $\eta$ and increase it by a fixed amount every iteration until it reaches $k\eta$ after ~5 epochs. Then resume the normal decay schedule. ^gradual-warmup

The bigger conceptual claim: large-batch training fails because of an **optimization** problem early in training, not a **generalization** problem. This directly contradicts Keskar et al.'s "sharp minima" story, which said large batches find flatter-but-worse solutions. Here, when the training curve is fixed, the validation error follows for free.

## The Methodology

**Setup.** ResNet-50 (the `fb.resnet.torch` variant, stride-2 on 3×3 layers), ImageNet-1k, 90 epochs regardless of batch size. Nesterov [[Momentum|momentum]] $m = 0.9$, weight decay $\lambda = 0.0001$ (not applied to the BN $\gamma,\beta$), [[Delving Deep into Rectifiers (He init, PReLU)|He initialisation]] for conv layers, final FC layer from $\mathcal{N}(0, 0.01^2)$.

Reference learning rate is $\eta = 0.1 \cdot \frac{kn}{256}$, cut by $10\times$ at epochs 30, 60, 80. With $k=8$ workers and $n=32$ images each you get the classic $\eta = 0.1$.

**Batch norm is where it gets subtle.** [[Batch Normalization|BN]] computes statistics across the batch, so a sample's loss depends on its batchmates. Change the batch size and you have literally changed the loss function you are minimising. The fix: **hold the per-worker batch size $n = 32$ fixed** and only grow $k$, the number of workers. Then each worker's batch is one independent draw from $X^n$ (the set of all size-$n$ subsets), the loss is unchanged, and the update becomes

$$\hat w_{t+1} = w_t - \hat\eta\frac{1}{k}\sum_{j<k}\nabla L(\mathcal{B}_j, w_t)$$

which is the same algebra as before with $\mathcal{B}$ playing the role of "one sample". Corollary: **do not sync BN statistics across workers.** That is not just a communication saving, it is required to keep the objective fixed.

**BN $\gamma$ init.** In each residual block, initialise the *last* BN layer's scale $\gamma = 0$ instead of 1. This makes the block output zero at init, so signal flows through the identity shortcut of the [[Deep Residual Learning for Image Recognition (ResNet)|ResNet]] and the network starts out effectively shallow.

### The four pitfalls (§3) — this is the most reusable part of the paper

**Remark 1 — scaling the loss is not the same as scaling the learning rate.** Weight decay is the gradient of an L2 term in the loss: $l(x,w) = \frac{\lambda}{2}\|w\|^2 + \varepsilon(x,w)$, so

$$w_{t+1} = w_t - \eta\lambda w_t - \eta\frac{1}{n}\sum_{x\in\mathcal{B}}\nabla\varepsilon(x,w_t)$$

Frameworks compute $\lambda w_t$ separately and add it after backprop. So multiplying the cross-entropy by $k$ scales only the second term, not the decay term. Same trap the [[Decoupled Weight Decay Regularization (AdamW)|AdamW]] paper documents from the other direction.

**Remark 2 — momentum correction.** Two equivalent-looking implementations:

$$u_{t+1} = m u_t + \tfrac{1}{n}\textstyle\sum\nabla l,\quad w_{t+1} = w_t - \eta u_{t+1}$$
$$v_{t+1} = m v_t + \eta\tfrac{1}{n}\textstyle\sum\nabla l,\quad w_{t+1} = w_t - v_{t+1}$$

They differ the moment $\eta$ changes, because $v$ has the old $\eta$ baked into its history. If you use the second form, rescale: $v_{t+1} = m\frac{\eta_{t+1}}{\eta_t}v_t + \eta_{t+1}\frac{1}{n}\sum\nabla l$. Matters most when $\eta_{t+1} \gg \eta_t$ — i.e. during warmup ramps.

**Remark 3 — normalise by the total batch $kn$, not per-worker $n$.** `allreduce` sums, it does not average. Loss layers already divide by local $n$, so you are missing a $1/k$. Fold it into the loss (only the loss gradient needs scaling, not the whole gradient vector) rather than scaling the gradient after aggregation.

**Remark 4 — one shuffle per epoch, split $k$ ways.** Not $k$ independent shuffles. Getting this wrong quietly changes the sampling distribution.

### Communication

Gradients are all-reduced layer by layer, overlapped with backprop, since layer $i$'s gradient is ready while layer $i-1$ is still computing. Three phases: intra-server reduce over 8 GPUs (NCCL for buffers ≥256 KB), inter-server all-reduce, then broadcast back.

For inter-server they compared **recursive halving-and-doubling** ($2\log_2 p$ steps) against the **ring/bucket** algorithm ($2(p-1)$ steps). Both move $2\frac{p-1}{p}b$ bytes. Halving/doubling won by **3× on 32 servers** because at these buffer sizes the problem is latency, not bandwidth. Non-power-of-two server counts use the binary blocks generalisation.

Bandwidth budget: ResNet-50 has ~25M params = 100 MB fp32; all-reduce moves ~2× that; backprop takes 120 ms; so ~1600 MB/s ≈ 12.8 Gbit/s, call it 15 with overhead. Their 50 Gbit **commodity Ethernet** is plenty. No InfiniBand needed.

## Ablation Studies and Experiments

All numbers are mean ± std over **5 independent runs**, with each run's error being the median of its last 5 epochs. The paper makes a point of this: ImageNet run-to-run std is ~0.1%, and single-trial results are not trustworthy.

**Warmup strategy, batch 8k, $\eta = 3.2$:**

| Setting | top-1 error |
|---|---|
| baseline, $kn=256$, $\eta=0.1$ | 23.60 ± 0.12 |
| no warmup | 24.84 ± 0.37 |
| **constant** warmup ($\eta=0.1$ for 5 epochs, then jump to 3.2) | 25.88 ± 0.56 |
| **gradual** warmup | **23.74 ± 0.09** |

Constant warmup is *worse than no warmup at all*. Error drops during the low-$\eta$ phase, then spikes the instant you jump to $3.2$, and never recovers. The sudden jump is the poison, not the high rate itself.

**Learning rate rules at 8k:**

| $\eta$ | top-1 error |
|---|---|
| $0.05\cdot 32$ | 24.27 ± 0.08 |
| $0.10\cdot 32 = 3.2$ (linear) | **23.74 ± 0.09** |
| $0.20\cdot 32$ | 24.05 ± 0.18 |
| $0.10$ (no scaling) | **41.67 ± 0.10** |
| $0.10\cdot\sqrt{32}$ (square-root rule) | **26.22 ± 0.03** |

The square-root rule — theoretically motivated by Krizhevsky as matching the reduction in gradient-estimator variance — loses 2.5 points. Not scaling at all loses 18 points. This is the strongest evidence in the paper.

**Batch size sweep (64 → 65536).** Validation error is flat from 64 through 8k, degrades from 16k (24.79 ± 0.27), and **diverges past 64k**. Training curves are the tell: whenever validation matches the baseline, the training curve matches too (after warmup); whenever validation is off, the training curve is visibly higher *for every epoch*. Practical consequence — you can diagnose a bad config from the training curve long before convergence.

**BN $\gamma = 0$ init:**

| $kn$ | $\gamma$-init | error |
|---|---|---|
| 256 | 1.0 | 23.84 ± 0.18 |
| 256 | 0.0 | 23.60 ± 0.12 |
| 8k | 1.0 | 24.11 ± 0.07 |
| 8k | 0.0 | 23.74 ± 0.09 |

Helps both, helps large batches slightly more. Another sign that large batch = fragile early optimisation.

**Does a bigger dataset raise the ceiling? No.** ImageNet-5k (6.8M images, 5× bigger) has a qualitatively identical elbow. 8k batch costs 0.26% (25.83 → 26.09). So you cannot buy a larger usable batch by adding data.

**ResNet-101.** 22.08 → 22.36 going from 256 to 8k. Slightly worse than ResNet-50's gap; 8k is probably also near this model's edge. 92.5 minutes on 256 GPUs.

**Transfer to detection.** ImageNet models pre-trained at batches 256…16k, each used to initialise Mask R-CNN on COCO (5 pre-trained models × 7 batch sizes = 35 detection runs). Box AP is 35.8–35.9 and mask AP 33.8–33.9 for everything up to **8k**; at 16k it drops to 35.1 / 33.2 — exactly tracking the ImageNet error. So no hidden generalisation damage; if classification error is matched, transfer is matched.

**Linear scaling inside Mask R-CNN itself.** Scaling 1→2→4→8 GPUs with $\eta$ from 0.0025 to 0.02 gives box AP 35.7, 35.7, 35.7, 35.6. Notable because Mask R-CNN has *different* effective batch sizes at different layers (2 images per GPU for the backbone, 512 RoIs each for the heads) and three different losses (softmax [[Cross Entropy|cross-entropy]], smooth-L1, per-pixel binomial CE). The rule survives all of it.

**Throughput.** Time per iteration rises only **12%** while batch grows 44× (256 → 11k). Time per epoch falls from ~16 minutes to 30 seconds on 352 GPUs. ~**90% scaling efficiency** from 8 to 256 GPUs.

**What did not work:** constant warmup, square-root scaling, fixed learning rate, batches ≥16k, and (implicitly) syncing BN across workers.

## Worth Remembering

- The headline is really two claims stacked. The engineering claim (1 hour, 90% efficiency) aged into standard practice. The scientific claim — **large-batch degradation is an optimisation failure, not a generalisation failure** — is the part still worth arguing about, and it directly contests Keskar et al. 2017.
- The window is $[64, 8192]$ for ResNet-50/ImageNet. The authors are honest that they have no theory for where the elbow sits, and that adding 5× data does not move it. They also warn: do not run *at* the breaking point, because sensitivity to $\eta$ increases there (at 8k, a 2× change in reference $\eta$ costs 0.3–0.5%; at 256 it costs almost nothing).
- Warmup escaped this paper entirely. It is now unconditional in transformer training — see the 4000-step warmup in [[Attention Is All You Need]] and every LLM recipe since — though there the justification is more about [[Adam- A Method for Stochastic Optimization|Adam]]'s second-moment estimate being garbage in the first few hundred steps than about the linear scaling argument here. Two different reasons, same fix.
- **The linear rule is SGD-specific.** Adam-family optimisers empirically prefer something closer to $\sqrt{k}$, and modern LLM practice tunes the peak rate directly or transfers it via μP ([[Let's Scale Step by Step- Compute-Efficient Hyperparameter Transfer for Large-Scale Mixture-of-Experts|μTransfer]]). Do not copy $\eta \propto k$ into an AdamW run and expect it to hold.
- The BN argument is the most under-appreciated bit. **Per-worker batch size $n$ is a property of the loss function, not of your parallelism.** If you go from 8 GPUs at $n=32$ to 4 GPUs at $n=64$ to keep the global batch the same, you have changed the objective and your numbers are not comparable. Same reason SyncBN changes results rather than just speeding them up. Compare with [[How Does Batch Normalization Help Optimization]] on why BN helps at all, and [[Layer Normalization]] for the batch-independent alternative that sidesteps the whole issue.
- The four remarks in §3 are free engineering advice and generalise beyond this paper. The weight-decay-vs-loss-scaling one in particular bites people who "scale the loss" for [[Mixed Precision Training|loss scaling]] in fp16 and forget that decay is added outside the backward pass.
- Reporting 5 runs with std is still unusual for ImageNet-scale work and quietly makes the whole paper credible — a 0.14% gap only means something once you know std is 0.1%. Compare the reproducibility concerns in [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] and [[On the Difficulty of Evaluating Baselines]].
- Practical caveat for anyone reusing this: 90% efficiency depended on overlapping all-reduce with backprop and on a model small enough (100 MB of params) that 50 Gbit Ethernet sufficed. For a model 100× larger you land in [[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models|ZeRO]] / [[Megatron-LM- Training Multi-Billion Parameter Models Using Model Parallelism|tensor-parallel]] territory and the arithmetic changes completely.
- Open question the paper leaves: what actually determines the elbow? Not dataset size. Possibly gradient noise scale — a later line of work that this paper set up but did not pursue.

## Links

Related: [[GPU processing]] · [[Batch Normalization]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Momentum]] · [[Delving Deep into Rectifiers (He init, PReLU)]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[How Does Batch Normalization Help Optimization]] · [[Regularization]] · [[Adam- A Method for Stochastic Optimization]] · [[Attention Is All You Need]] · [[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models]] · [[Megatron-LM- Training Multi-Billion Parameter Models Using Model Parallelism]] · [[Mixed Precision Training]] · [[ML Infrastructure]] · [[Cross Entropy]] · [[Understanding the difficulty of training deep feedforward networks (Xavier init)]] · [[Layer Normalization]]

New topics worth writing: Gradient noise scale and critical batch size, Sharp vs flat minima and the large-batch generalisation gap (Keskar et al.), All-reduce algorithms (ring vs recursive halving-doubling), Synchronous vs asynchronous SGD and stale gradients, LARS and layer-wise adaptive rate scaling, Learning-rate warmup for adaptive optimisers, Mask R-CNN
