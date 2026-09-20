---
title: "Dynamic Important Example Mining for Reinforcement Finetuning"
authors: ["Haoru Tan", "Sitong Wu", "Yanfeng Chen", "Shizhen Zhao", "Yang-Tian Sun", "Tianjia Liu", "Chirui Chang", "Shaofeng Zhang", "Samm Sun", "Xiuzhe Wu", "Ruobing Xie", "Xiaojuan Qi"]
year: 2026
arxiv: "2608.29252"
url: https://arxiv.org/abs/2608.29252
priority: Good-To-Read
read_on: 2026-09-05
tags: [paper, rl, optimization, theory]
---
## The Core Idea

In reinforcement fine-tuning (RFT) — running RL on top of a pretrained model, usually with a verifiable reward like "is the maths answer right" — you sample a batch of prompts, roll out answers, score them, and take a [[Derivative#Gradient|gradient]] step. Every sample in the batch gets equal say in that step.

People have tried to fix this by picking better data. Almost all of those methods score a sample **once**, before training: pick problems with high reward variance, or medium difficulty, or a "difficulty score" from an auxiliary value model. The complaint here is that a sample's worth is not a fixed property. Early in training an easy problem teaches a lot; three thousand steps later it teaches nothing. The policy is moving, so the data's value moves with it.

DIEM's answer: **stop guessing what a sample is worth and just measure it, at every step, using quantities you already computed.**

The measurement it wants is a leave-one-out quantity. Take the batch $\mathcal{B}_t$. Do one update using the whole batch, and separately do one update using the batch *minus* sample $z$. Compare how well each resulting policy scores on the batch. The difference is $z$'s true marginal contribution:

$$\mathcal{I}_t(z) = \mathcal{J}\big(\theta^{\text{update}}_{\mathcal{B}_t}, \mathcal{B}_t\big) - \mathcal{J}\big(\theta^{\text{update}}_{\mathcal{B}_t \setminus \{z\}}, \mathcal{B}_t\big)$$

This costs $|\mathcal{B}_t|$ extra full training steps per step. Useless in practice. The trick is that to first order it collapses into a dot product between two gradients you already have:

$$\hat{\mathcal{I}}_t(z) = \eta_t \big\langle \mathcal{G}^{(t)}_z,\ \mathcal{G}^{(t)}_{\mathcal{B}_t} \big\rangle$$

where $\mathcal{G}_z$ is the sample's own policy gradient and $\mathcal{G}_{\mathcal{B}_t}$ is the summed batch gradient. If a sample pulls the weights in roughly the same direction as the batch as a whole, it is helping. If it pulls sideways or backwards, it is noise, or actively harmful, *right now*.

> [!NOTE] Gradient-alignment importance
> A sample's value at step $t$ is the inner product of its gradient with the batch's aggregate gradient. Positive = the sample agrees with where the group is going. Negative = it fights the update. Costs one dot product, needs no extra model, no extra forward pass, and no assumption that you are near a minimum. ^gradient-alignment

The second half is what to *do* with those scores. Naively upweighting the high-scoring samples changes the size of the update, which is exactly the thing that destabilises RL. So the weights are chosen by a constrained problem: maximise total measured utility, subject to the reweighted gradient having the **same length** as the plain unweighted one. Direction changes; step size does not.

What this unlocks: data selection becomes a free, always-on part of the optimiser instead of a preprocessing pipeline. And the authors show the resulting weights drift from easy → hard over training on their own, so you get curriculum learning without hand-designing a curriculum.

## The Methodology

Base algorithm is GRPO ([[Proximal Policy Optimization Algorithms|PPO]]-style clipping, group-normalised advantage, no learned value function):

$$A(s,a_i) = \frac{r_i - \text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}$$

DIEM bolts two steps into the middle of each iteration.

**Step 1 — importance.** Compute a per-sample policy gradient $\mathcal{G}_z$ for each of the $N$ samples in the minibatch. Stack them into $\mathbf{G} \in \mathbb{R}^{N \times D}$ ($D$ = number of parameters). The batch gradient is $\mathbf{1}^\top \mathbf{G}$. The importance vector is $\mathbf{I} = \eta_t \mathbf{G}(\mathbf{1}^\top\mathbf{G})^\top$.

**Step 2 — reweighting.** Find weights $\mathbf{W} \in \mathbb{R}^N$:

$$\max_{\mathbf{W}} \ \mathbf{I}^\top \mathbf{W} \quad \text{s.t.} \quad \|\mathbf{W}^\top \mathbf{G}\|^2 = \|\mathbf{1}^\top \mathbf{G}\|^2$$

Objective: pile weight on useful samples. Constraint: the resulting update vector has exactly the norm it would have had with uniform weights.

With $\mathbf{P} = \mathbf{G}\mathbf{G}^\top \in \mathbb{R}^{N \times N}$ (the Gram matrix of per-sample gradients) and $C = \|\mathbf{1}^\top\mathbf{G}\|^2$, Lagrange multipliers give a closed form. The paper prints it as

$$\mathbf{W}^* = \frac{\mathbf{P}^{-1}\mathbf{I}}{\sqrt{C}}\sqrt{\mathbf{I}^\top \mathbf{P}^{-1}\mathbf{I}}$$

(Working it out myself I get the two square-root factors swapped — $\mathbf{W}^* = \sqrt{C}\,\mathbf{P}^{-1}\mathbf{I}\,/\sqrt{\mathbf{I}^\top\mathbf{P}^{-1}\mathbf{I}}$ — which is the version that actually satisfies the constraint. Likely a typo.)

The key efficiency point: $\mathbf{P}$ is $N \times N$ where $N$ is the *minibatch size* (32 here), not $D$. Inverting a 32×32 matrix is nothing next to an RFT step that takes minutes.

**Step 3 — clip and update.** $\mathbf{W}^*$ can contain negatives. Negative weight means "push away from this sample", which is not what you want when the score might just be estimation noise, so: $\mathbf{W}^* \leftarrow \max(0, \mathbf{W}^*)$. Then $\mathbf{G}_{\text{weighted}} = \mathbf{W}^{*\top}\mathbf{G}$ and $\theta_{t+1} = \theta_t + \eta_t \mathbf{G}_{\text{weighted}}$.

> [!NOTE] Norm-preserving reweighting
> Reweighting a batch changes *both* the direction and the magnitude of the update. DIEM's equality constraint pins the magnitude to whatever it would have been under uniform weights, so the only thing the weights are allowed to change is direction. This is why it does not need a learning-rate retune. ^norm-preserving-reweighting

**Error bound (Prop. 2).** If the log-likelihood is $\ell$-Lipschitz and advantages are bounded by $A_{\max}$:

$$\big|\mathcal{I}_t(z) - \hat{\mathcal{I}}_t(z)\big| \le \mathcal{O}\big(\eta_t \ell^2 + 2\eta_t \ell A_{\max}\big)$$

The error scales with the learning rate, which is $10^{-6}$ here. Notably it needs **no convexity** and **no near-stationarity**, unlike classic influence functions — which matters because RFT never sits near a minimum.

**Training setup.**
- *LLM*: Qwen3-1.7B/4B, Qwen2.5-3B/7B, veRL, 16×H200. GRPO with KL penalty and entropy bonus **off**, clip 0.2. Prompt batch 64, 8 rollouts per prompt, minibatch 32, micro-batch 8 (4 for 7B/8B). Prompt ≤1024 tok, response ≤2048 tok. Constant LR $10^{-6}$, no warmup. Data: 14,973 maths problems (7,500 MATH + 7,473 dapo-math).
- *VLM*: Qwen2.5-VL-7B (16×A100) and -32B (32 GPUs), 52K samples from MM-Eureka, global batch 128, LR $10^{-6}$.

## Ablation Studies and Experiments

**LLM maths (average over MATH-500, Gaokao23en, AMC-23, AIME24, AIME25):**

| Model | GRPO | HVS | LIMR | **DIEM** |
|---|---|---|---|---|
| Qwen3-1.7B | 31.36 | 31.72 | 32.20 | **33.10** |
| Qwen2.5-3B | 27.82 | 28.24 | 28.78 | **30.32** |
| Qwen3-4B | 37.30 | 38.14 | 39.42 | **40.66** |
| Qwen2.5-7B | 34.00 | 34.38 | 34.98 | **35.68** |

The gains concentrate on the hardest benchmarks. AIME25 on Qwen2.5-7B: 5.5 → 10.8 (+96% relative). AIME25 on Qwen3-1.7B: 3.4 → 5.5. But these are tiny 30-problem test sets, so a couple of extra correct answers moves the number a lot — treat the "96%" with suspicion.

**One clean loss:** AMC-23 on Qwen3-4B, DIEM 55.0 vs GRPO 58.5. The authors call it an outlier. It is the only cell where DIEM is beaten by plain GRPO.

**VLM (avg over MathVista, MathVerse, MathVision, MMStar, MMMU, AI2D):**

| Method | Qwen2.5-VL-7B | Qwen2.5-VL-32B |
|---|---|---|
| Base model | 58.2 | 63.8 |
| Vanilla RFT (GRPO) | 59.1 | 64.9 |
| LIMR (static) | 58.3 | 64.9 |
| HVS (static) | 59.5 | 63.9 |
| SPEED-RL (dynamic) | 60.0 | 65.6 |
| PCL (dynamic) | 58.8 | 65.3 |
| **DIEM** | **61.8** | **67.3** |

Worth staring at the baseline columns: on the 7B model, **LIMR (58.3) and PCL (58.8) are worse than doing nothing at all with vanilla RFT (59.1)**, and HVS at 63.9 is *below the untouched base model* on 32B. Existing data-selection methods are close to noise here. DIEM at 61.8 on 7B nudges past GPT-4o's 60.9 average, though GPT-5-nano (73.1 MathVista) is in a different league.

**Ablation (MathVerse, Qwen2.5-VL-32B, full model = 58.0).**

Swap the importance score for something else:

| Replacement | Score | Δ |
|---|---|---|
| DIEM score | 58.0 | — |
| Random values | 53.0 | −5.0 |
| Pass@k (SPEED-RL's signal) | 53.2 | −4.8 |
| Pass@k distance-to-median | 54.9 | −3.1 |
| PCL difficulty score | 52.1 | −5.9 |
| Difficulty distance-to-median | 53.6 | −4.4 |

Swap the reweighting step:

| Replacement | Score | Δ |
|---|---|---|
| DIEM constrained solve | 58.0 | — |
| NULL (no reweighting) | 55.4 | −2.6 |
| Softmax normalisation | 56.4 | −1.6 |

**What this reveals.** Two things. First, plugging *any* score into the constrained reweighting machinery is worse than not reweighting at all (52.1–54.9 vs the 55.4 NULL row) — the reweighting solve amplifies whatever signal you feed it, so a bad score is worse than no score. The alignment estimator is doing the real work. Second, the "distance to median" versions of the heuristics beat the raw versions by 1.5–1.7 points, which confirms the folk wisdom that medium-difficulty is what you want — but still lands 3+ points short.

**What did not work:** raw difficulty and raw pass rate as importance signals (both worse than random weights would suggest is possible), and softmax normalisation as a reweighting rule — softmax has no norm constraint, so it changes the effective step size, which costs 1.6 points.

**Speed.** Total wall clock for the same run: GRPO 70.3h, DIEM 71.2h (+1.28%), PCL 79.1h, SPEED-RL 94.6h, LIMR and HVS both 122.0h. The static methods are *slowest* because they need a full surrogate training pass over the corpus before RFT even starts.

**Curriculum visualisation (Fig. 3).** Bucket samples into Easy/Medium/Hard by Pass@k, track their mean DIEM weight. Early: Easy and Medium both weighted high. Later: Easy weight collapses, Hard weight climbs steadily (with oscillation). Nobody designed this schedule; it falls out of gradient alignment.

## Worth Remembering

**There is an algebraic problem with the method as written, and it is worth thinking about.** Note that $\mathbf{I} = \eta\,\mathbf{G}\mathbf{G}^\top\mathbf{1} = \eta\,\mathbf{P}\mathbf{1}$. Substituting into the closed form gives $\mathbf{P}^{-1}\mathbf{I} = \eta\mathbf{1}$, so $\mathbf{W}^* \propto \mathbf{1}$ — and the norm constraint then forces the proportionality constant to exactly 1. **Taken literally, the equations produce uniform weights and DIEM reduces to vanilla GRPO.** No clipping fires either, since all weights are positive. The empirical results say the implementation is clearly doing something (NULL-op costs 2.6 points), so the code must differ from the paper: my guess is a damped inverse $(\mathbf{P} + \epsilon\mathbf{I})^{-1}$ for numerical stability — the Gram matrix of billion-dimensional gradients is horribly ill-conditioned — which breaks the cancellation and leaves a mild reweighting proportional to the small eigen-directions of $\mathbf{P}$. If so, **the regularisation constant is the actual hyperparameter of the method, and it is never mentioned.** Read the code before trusting the derivation.

**Per-sample gradients are not free.** The paper repeatedly claims $\mathcal{G}_z$ and $\mathcal{G}_{\mathcal{B}_t}$ "are already calculated during the standard RFT backpropagation step". This is not true of ordinary [[Backpropagation]] — a normal backward pass hands you the *sum* over the batch, not the per-example gradients. Getting them needs per-sample gradient machinery (functorch/vmap, or a separate backward per micro-batch). With micro-batch size 8 and minibatch 32, my read is $N=4$ micro-batch gradients per step, not 32 per-sample ones. That would explain the 1.28% overhead, and it means "per-sample" reweighting is really "per-micro-batch" reweighting. Check this before you budget memory.

**Storage.** Even at $N=32$, holding 32 copies of a full gradient for a 32B model is impossible. The real implementation must project gradients down (random projection, last-layer-only, or LoRA-subspace) before forming $\mathbf{P}$. Not discussed in the paper.

**Honest strengths.** The 1.28% overhead figure is genuinely the headline — most data-selection work in this space costs 12%–74% extra wall clock, and half of it does not beat the do-nothing baseline. The error bound not requiring convexity or stationarity is the right theoretical move for RFT, where you are nowhere near a critical point.

**Limitations the authors do not raise.** All results are single runs, on 30-problem competition benchmarks (AIME24/25) where variance swamps 1–2 point differences. No seed variance is reported anywhere. The one adverse result (AMC-23 on Qwen3-4B, −3.5) is dismissed as an outlier without investigation. The KL penalty is disabled in the GRPO baseline, so we do not know how DIEM interacts with the reference-model constraint used in most real [[Training language models to follow instructions with human feedback|RLHF]] pipelines.

**Conceptual caveat on the signal itself.** Gradient alignment rewards samples that agree with the majority direction of the batch. That is a *conformity* measure. A genuinely novel hard example whose gradient points somewhere nobody else is pointing gets a low or negative score and is clipped to zero weight. The curriculum plot suggests this does not bite in practice — hard samples gain weight over time, presumably because once the batch is mostly hard problems, the majority direction *is* the hard direction — but the mechanism is self-reinforcing and could in principle collapse diversity. Worth checking whether entropy drops faster under DIEM.

**Follow-up questions.** Does the alignment score correlate with the true leave-one-out influence if you actually compute it on a toy setup? Would simply *dropping* the negative-alignment samples (a hard filter) capture most of the gain without the Gram inverse? And how does this interact with the clipping in GRPO — a sample already clipped contributes zero gradient, so it gets importance zero automatically, which means DIEM and PPO clipping are partly doing the same job.

## Links

Related: [[Proximal Policy Optimization Algorithms]] · Trust Region Policy Optimization · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[Training language models to follow instructions with human feedback]] · [[High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)]] · [[Backpropagation]] · [[Derivative]] · [[Adam- A Method for Stochastic Optimization]] · [[A Unified Approach to Interpreting Model Predictions (SHAP)]] · [[Fundamentals]] · [[Direct Preference Optimization (DPO)]]

New topics worth writing: GRPO (Group Relative Policy Optimization), influence functions and leave-one-out data attribution, curriculum learning, per-sample gradients and vmap in PyTorch, Gram matrix conditioning and damped inverses, Lagrange multipliers for constrained optimisation, DAPO / dynamic sampling policy optimization, gradient surgery and conflicting gradients (PCGrad), coreset selection
