---
title: "Let's Scale Step by Step: Compute-Efficient Hyperparameter Transfer for Large-Scale Mixture-of-Experts"
authors: ["Nayeon Kim", "Hojin Lee", "Yunju Bak", "Jaesun Park", "Boseop Kim"]
year: 2026
arxiv: "2608.20061"
url: https://arxiv.org/abs/2608.20061
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, transformers, optimization, theory, scaling]
---
## The Core Idea

Picking the learning rate for a giant model is expensive. The best value depends on **two** things at once: how big the model is, and how many tokens you train on. The usual way to find it is a 2D sweep — try many model sizes × many token budgets — and that costs a fortune before you have trained anything real.

This work splits that 2D grid into two cheap 1D problems.

1. **Model size axis → solved for free.** Use μP (Maximal Update Parameterization), which rescales initialisation and per-layer learning rates so that the *best* learning rate stops moving as you widen the model. Then a tiny model's best LR is also the huge model's best LR. This was known for dense transformers; the contribution here is making it work for a modern MoE stack — sparse experts, Multi-head Latent Attention, and the Muon optimizer — while the scaling path also increases the *number of experts*, not just width.
2. **Token budget axis → one cheap regression.** Train a small proxy model, measure the best LR at many token budgets, and fit a straight line in log-log space. Extrapolate that line out to 10 trillion tokens.

The second trick that makes step 2 affordable: instead of running a separate decayed run per token budget (one run = one usable data point), they run **one** undecayed run and take an exponential moving average (EMA) of the weights at intervals. EMA weights approximate what a learning-rate decay would have given you, so a single run yields dozens of "as if decayed" checkpoints at 10B-token intervals.

The payoff: the whole hyperparameter search cost 64.8 ZFLOPs; the real training run was ~98× that. They then pretrained a 155B-total / 17B-active MoE on 10T tokens with the predicted LR of $3.85\times10^{-4}$, and the loss curve had **no spikes**.

> [!NOTE] μ-Transfer
> Reparameterise a network so that the optimal learning rate is invariant to width. Tune on a small proxy, reuse the number on the big model, zero extra sweeps. ^mu-transfer

## The Methodology

### Step 1 — μP for an MoE with MLA and Muon

Parameters are bucketed by how many of their dimensions grow with width:

| Type | Examples | Init variance | LR multiplier |
|---|---|---|---|
| Vector-like (one expandable dim) | embeddings, biases, **expert FC2** | $\mathrm{fan\_in_{base}}/\mathrm{fan\_in}$ | $1$ |
| Matrix-like (two expandable dims) | attention, FFN, **router**, **expert FC1** | $\mathrm{fan\_in_{base}}/\mathrm{fan\_in}$ | $\mathrm{fan\_in_{base}}/\mathrm{fan\_in}$ |

Embeddings get init variance $0.04$ and LR multiplier $1$.

The MoE-specific call: expert FC2 counts as **vector-like**, because its effective input width is bounded by the fixed number of active experts and the fixed expert intermediate dimension. Router and expert FC1 are matrix-like.

Two more choices, following Wortsman et al.:
- LR scaling only on linear (matrix-like) hidden weights — nothing else needed.
- **Depth is frozen.** Depth scaling under μP is known to be unstable. Only width moves, and head *dimension* stays fixed while head *count* grows with $d_{\text{model}}$.

For [[RoFormer- Enhanced Transformer with Rotary Position Embedding|MLA]]-style latent attention: the low-rank query and KV projection dims are held constant across width scaling. Since those dims are the `fan_in` of the up-projections, the LR scaling factor for those matrices collapses to $1$ automatically.

**The MoE scaling path.** Going wider only is impractical past 100B params. Instead they scale width *and* total experts together, keeping active experts $k$ and the expert intermediate dim fixed. So $d_{\text{ff}} = k \times d_{\text{expert}}$ never changes. Their justification via the spectral-condition view of μP: adding more experts does not change any individual expert's fan-in or fan-out beyond what width scaling already did, so the same rule applies. The routed scaling factor $\lambda_{\text{routed}}$ is re-derived per size (2.428 → 2.448).

**Muon.** To reuse LRs tuned for AdamW, they match Muon's update RMS to AdamW's by multiplying the LR by $0.2\cdot\sqrt{\max(A,B)}$ for a matrix parameter of shape $A\times B$. Weight decay fixed at $0.1$, z-loss at $5\times10^{-6}$.

### Step 2 — the token-budget scaling law

Batch size is deliberately **excluded** from the transfer. Their reasoning: batch size is a systems knob you change for throughput, and the literature disagrees on whether optimal batch size depends on compute or on token count alone. So they pin it (8M → 32M tokens after 200B) and transfer only the LR.

The proxy (10.8B total / 3.3B active, $1/4$ the target width) trains ~500B tokens with a Warmup-Stable-Decay schedule, but the run is **cut off during the stable phase — no decay**. EMA of the weights stands in for decay:

$$\theta_{\text{EMA}}^{(t)} = \alpha\,\theta_{\text{EMA}}^{(t-1)} + (1-\alpha)\,\theta^{(t)},\quad \alpha = 0.6$$

EMA is applied every ~2B tokens; checkpoints every 10B tokens are used. With $\alpha=0.6$ at 2B-token steps, roughly the last 20B tokens retain >1% influence on the merged weights.

At each token budget $B$, sweep LRs, evaluate validation loss on a fixed batch, and fit a parabola in log-LR:

$$\mathcal{L}(\eta) = a(\log\eta)^2 + b(\log\eta) + c$$

The vertex gives $\eta^{*} = \exp(-b/2a)$. Then fit across budgets:

$$\log(\eta^{*}) = \beta\log(B) + \gamma$$

Only points after 255B tokens are used, so the batch-size jump at 200B does not contaminate the fit. $R^2 = 0.95$. Extrapolated to 10T tokens: $\eta^{*} = 3.85\times 10^{-4}$.

### The models

| Role | Total / Active | $n_{\text{layers}}$ | $d_{\text{model}}$ | $n_{\text{experts}}$ |
|---|---|---|---|---|
| μP validation base | 0.6B / 0.3B | 48 | 256 | 16 |
| μP validation $8\times$ | 30.7B / 3.6B | 48 | 2048 | 128 |
| Token-law proxy | 10.8B / 3.3B | 62 | 1024 | 32 |
| **Target** | **155B / 17B** | 62 | 4096 | 128 |

Target data mixture, Stage 1: 45% English / 12.5% Math-STEM / 27.5% Code / 15% Multilingual, switched at the 6T-token mark to 22.5 / 27.5 / 25 / 25.

## Ablation Studies and Experiments

**Does μP actually transfer? (Section 3.2)** Base MoE 0.6B, scaled to $2\times$ (2.2B), $4\times$ (8B), $8\times$ (30.7B), all 1.3B tokens, LR swept over $\{1,3,6\}\times10^{-4}$, $\{1,3,4,6\}\times10^{-3}$, $\{1,3\}\times10^{-2}$. Under **Standard Parameterization the optimal LR drifts** as width grows — the loss-vs-LR minimum walks left. Under μP all four curves put their minimum at the same LR. Appendix C repeats this for dense MLA models (0.24B → 8.02B) and gets the same answer, which was a needed check since MLA + Muon + μP had not been tested together before.

**Does the parabola fit hold across widths? (Section 3.3.1)** Proxy 5.6B/1.8B vs held-out $2\times$ 20.7B/3.8B, at 40/60/80/100B tokens. Four LRs are used to fit each parabola; the fifth point ($2\times10^{-3}$) is held out and **lands directly on the fitted curve**. Curvature *and* vertex location match between proxy and $2\times$ model at every token scale. The optimal LR drifts slightly down as tokens increase — which is the whole reason step 2 is needed.

**Held-out validation of the extrapolation (Appendix E).** They refit the log-log line using only budgets 255B–350B (11 points) and predict five unseen budgets near 500B:

| Budget | Predicted | Actual | Ratio |
|---|---|---|---|
| 462.7B | $9.57\times10^{-4}$ | $9.26\times10^{-4}$ | 1.033 |
| 482.5B | $9.49\times10^{-4}$ | $8.97\times10^{-4}$ | 1.058 |
| 502.3B | $9.41\times10^{-4}$ | $9.01\times10^{-4}$ | 1.045 |

Average error ≈ 4.4%. Note this is a $1.4\times$ extrapolation, not a $20\times$ one — the 10T claim rests on the line staying straight far past what was tested.

**The real run.** 10T tokens, 155B/17B. Loss curve monotone, no spikes. Benchmarks: MMLU, MMLU-Pro, BBH, Global-MMLU (Ko/Ja/Vi/Zh), MATH, GSM8K, MBPP, HumanEval. On compute ($6ND$, $N$ = active params) vs MMLU-Pro, the model sits on the Pareto frontier — above dots.llm1 and GLM-4.5-Air at equal or lower compute.

**What cost what.** Proxy runs: 64.8 ZFLOPs. Adding the $1.5\times$ and $2\times$ model-scale sweeps that a 2D approach would require: **+240.3 ZFLOPs**, ~4.7× more, all of it avoided by μP. Target training: ~98× the proxy total.

**What did not work / was avoided:**
- **Decay during proxy runs** — rejected twice over. Each decayed run gives one usable point, and premature decay biases the loss estimate. EMA replaced it.
- **Depth scaling** — not attempted; known to break μP transfer.
- **Batch size transfer** — explicitly abandoned as an unresolved question.
- **Regression points before 255B** — discarded because the batch-size jump at 200B changes the optimisation regime.
- **Expert routing bias in Stage 2 (Appendix F).** Three settings: inherit Stage-1 bias and keep updating; inherit and freeze; zero out entirely. Continuing to update gives the lowest MaxVio (load imbalance). Surprisingly, if you *must* freeze, **zeroing beats inheriting** — Stage-1 biases are tuned for the wrong data distribution. But all three give **near-identical loss curves**, so the bias term barely affects optimisation.
- **Sparsity axis is confounded.** They scale experts and width jointly, so they cannot say what the sparsity dimension does on its own. They admit this.

## Worth Remembering

- The honest limitation: they never verified the predicted $3.85\times10^{-4}$ *is* optimal at 10T tokens. That would need a full sweep at target scale, which is infeasible. Evidence is circumstantial — stable loss, competitive scores.
- The compute maths in Appendix E is a good sanity anchor: sweeping five LRs for 10T tokens on a model one-tenth the target size costs about **half** of the full 155B training run. Any "just sweep it" instinct dies there.
- EMA-as-decay-proxy leans on Zhang et al. (2024): constant LR + exponential weight averaging matches cosine decay in **large-batch** regimes, and the gap widens as batch shrinks. This run used 32M-token batches, so the assumption is safe here — it may not be for you. DeepSeek-V3 and OLMo 3 use the same trick.
- Routing balance and expert specialisation are **not** the same thing (Appendix G). MaxVio stays flat across depth, but domain-conditioned specialisation — measured as normalised mutual information $I(E;D)/H(D)$ and mean pairwise Jensen–Shannon divergence between per-domain routing distributions — rises sharply in the deepest MoE layers. Code specialises early; multilingual stays generic until the very last layers. Balanced load does not mean the experts are interchangeable.
- Practical caveat on MLA: because the low-rank projection dims are held fixed, the μP LR factor on their up-projections silently becomes $1$. If you widen those latent dims instead, this whole table changes.
- Open follow-up the authors flag: with top-$k$ routing, different experts see different effective batch sizes and different gradient noise, so a **per-expert learning rate** might help. They left it alone because routing drifts during training and the engineering is nasty.
- The first MoE layer in the large runs is replaced by a plain dense layer with intermediate dim $k \times d_{\text{expert}}$ — a small detail worth copying.

## Links

Related: [[Sparsely-Gated Mixture-of-Experts Layer]] · [[Old Optimizer, New Norm- An Anthology (Muon)]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[Adam- A Method for Stochastic Optimization]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Megatron-LM- Training Multi-Billion Parameter Models Using Model Parallelism]] · [[Attention Is All You Need]] · [[KL Divergence]] · [[Practical Bayesian Optimization of Machine Learning Algorithms]] · [[Foundation Models]] · [[Understanding the difficulty of training deep feedforward networks (Xavier init)]] · [[Delving Deep into Rectifiers (He init, PReLU)]]

New topics worth writing: Maximal Update Parameterization (μP), Multi-head Latent Attention, Warmup-Stable-Decay schedulers, exponential moving average of weights, MoE load balancing and MaxVio, auxiliary-loss-free load balancing, critical batch size, spectral condition for feature learning
