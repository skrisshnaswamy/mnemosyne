---
title: "Gated Recurrent Transformers: Expressive Depth through Recurrent Modulation"
authors: ["Amr Hegazy", "Amr Alanwar", "Mostafa Elhoushi"]
year: 2026
arxiv: "2608.15062"
url: https://arxiv.org/abs/2608.15062
priority: Good-To-Read
read_on: 2026-09-04
tags: [paper, transformers, llm, scaling]
---
## The Core Idea

A normal transformer buys depth with parameters. Every extra layer is a fresh block of weights, so a 36-layer model stores 36 blocks. Weight sharing is the obvious escape: use one block and run it many times. But shared weights have a known failure mode — the same function is applied to a hidden state that is very different at pass 1 and pass 8. One fixed transformation cannot be both "read the input" and "polish an almost-final answer", so quality collapses.

The **Gated Recurrent Transformer (GRT)** fixes this without adding weights: it makes the *input* to the shared block different at each pass, instead of making the block different. Three things change every step:

1. The block is fed a **projection of the current state concatenated with a frozen "prelude" representation of the original input**, so it always sees both "where we are" and "where we started".
2. Gaussian **noise** is resampled every step, so the model cannot memorise an exact step-by-step script.
3. A learned **per-element gate** decides, for each token and each hidden dimension separately, how much of the block's proposed output actually gets written into the residual stream.

The gate is a GRU-style update gate, but running over *depth* rather than *time*. A GRU reuses one weight matrix across sequence positions; GRT reuses one stack of transformer blocks across recurrence steps. Same trick, rotated 90 degrees.

What it unlocks: a 3-block model matches 12-layer GPT-2 Small at the same FLOPs with 64% fewer parameters. At large scale, 62% fewer parameters and 59% less peak decoding memory for +10% generation latency. And because recurrence depth is sampled randomly during training, one checkpoint gives you a continuous compute–quality dial at inference — stop after 3 of 6 loops and keep ~92% of accuracy, with no auxiliary early-exit loss.

> [!NOTE] Recurrent depth
> Applying the *same* weights repeatedly along the depth axis of a network, so effective depth grows while parameter count stays fixed. Contrast with recurrence over the sequence axis (RNN/LSTM/GRU). ^recurrent-depth

> [!NOTE] isoFLOPs vs isoParams
> Two ways to compare a shared-weight model to a dense one. **isoFLOPs**: match compute per forward pass, and see how many parameters you saved. **isoParams**: match parameter count, spend more compute by looping more, and see how much loss you gained. Kaplan et al. found shared models win on isoParams and lose on isoFLOPs; this paper attacks both. ^iso-regimes

## The Methodology

**Layout.** Blocks are split into three roles, written as $n_{\text{pre}}\texttt{+}n_{\text{rec}}\times R\texttt{+}n_{\text{coda}}$:

- **Prelude** ($n_{\text{pre}}$ blocks): run once on the embeddings to produce $\mathbf{h}^{(\mathrm{pre})}$. This is frozen for the rest of the forward pass and acts as a stable anchor.
- **Shared core** ($n_{\text{rec}}$ blocks): run $R$ times with identical weights.
- **Coda** ($n_{\text{coda}}$ blocks): run once, feeds the LM head.

A `2+5×4+2` config executes $2 + 5\cdot4 + 2 = 24$ block applications but stores weights for only $9$ blocks — the same compute as 24-layer GPT-2 Medium, 2.6× fewer unique blocks. Note: sharing *all* layers (the ALBERT approach) trained worse; keeping fixed encoders and decoders around a shared middle was more stable.

**The recurrence step.** At step $r$:

$$\tilde{\mathbf{h}}^{(r)} = W_{\mathrm{proj}}\bigl[\mathbf{h}^{(r-1)} + \epsilon_x,\; \mathbf{h}^{(\mathrm{pre})}\bigr]$$
$$\mathbf{o}^{(r)} = \mathcal{B}_{\text{shared}}\bigl(\tilde{\mathbf{h}}^{(r)}\bigr)$$
$$\mathbf{h}^{(r)} = \mathbf{g}^{(r)} \odot \mathbf{h}^{(r-1)} + \bigl(1 - \mathbf{g}^{(r)}\bigr) \odot \mathbf{o}^{(r)}$$

$[\cdot,\cdot]$ is concatenation along features, so $W_{\mathrm{proj}} \in \mathbb{R}^{d \times 2d}$ ([[Linear Projection]]). $\epsilon_x \sim \mathcal{N}(0, \sigma_x^2 I)$ with $\sigma_x = 0.1$. $\odot$ is elementwise product. Read the last line as: gate near 1 means "copy the old state through", gate near 0 means "overwrite with the block's proposal".

**The gate.**

$$\mathbf{g}^{(r)} = \sigma\!\left(f_{\mathbf{g}}\bigl([\mathrm{LN}(\mathbf{h}^{(r-1)}),\; \mathrm{LN}(\mathbf{h}^{(\mathrm{pre})})]\bigr)/\tau + \epsilon_g\right)$$

$f_{\mathbf{g}}$ is a two-layer MLP with SiLU activation and hidden width $d$. $\mathrm{LN}$ is [[Layer Normalization|layer norm]]. $\tau = 1.0$. $\epsilon_g \sim \mathcal{N}(0, 0.1^2)$ is per-scalar noise applied during training only, to stop the gate collapsing to a constant. The gate is $[0,1]^{T\times d}$ — one value per token *per dimension*, not per layer or per token.

**Gate bias init = +4.** This puts $\mathbf{g} \approx 0.98$ at the start, so at initialisation the residual stream passes through all $R$ recurrences almost untouched, and the model learns *which elements to overwrite* as training goes. This is the old LSTM forget-gate-bias trick (Gers et al. 2000, Jozefowicz et al. 2015) moved from time to depth ([[Long Short-Term Memory (Neural Computation)|LSTM]], [[Gated Activation]]).

**Depth sampling.** At each training step, sample $r \sim \mathrm{Uniform}\{1,\ldots,R\}$ and only run that many loops. Two effects: every exit point gets trained (so early exit works with no auxiliary losses and no gradient interference between exits), and it acts as stochastic depth [[Regularization|regularisation]].

**FLOPs accounting** per token at sequence length $S$:

$$\text{FLOPs} = (n_{\text{pre}} + n_{\text{rec}}R + n_{\text{coda}})(24d^2 + 4Sd) + R \cdot 10d^2$$

The $10d^2$ per step is the extra cost of $W_{\mathrm{proj}}$ and the gate MLP — small, and how they hit exact isoFLOPs parity with the dense baseline.

**Training.** nanoGPT backbone, GPT-2 BPE (50,257 vocab), $T = 1024$. [[Decoupled Weight Decay Regularization (AdamW)|AdamW]] with $\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay 0.1, grad clip 1.0, 2,000-step linear warmup then cosine from $6\times10^{-4}$ to $6\times10^{-5}$. [[Mixed Precision Training|bfloat16]]. 20,000 steps at ~491,520 tokens/step ≈ 9.8B tokens. 2×H200.

## Ablation Studies and Experiments

**Main table (validation loss, lower better).**

isoFLOPs — matched compute per forward pass:

| Scale | Dense | MoR | Poisson | Ouro | RRT | **GRT** | GRT params |
|---|---|---|---|---|---|---|---|
| Small (12L, 124M) | 3.15 | 3.30 | 3.23 | 3.19 | 3.14 | **3.14** | 35M |
| Medium (24L, 354M) | 2.84 | 3.02 | 2.97 | 2.93 | 2.95 | **2.89** | 127M |
| Large (36L, 774M) | 2.71 | 2.91 | 2.93 | 2.82 | 2.85 | **2.77** | 293M |

GRT beats every recurrent competitor at all three scales, and the margin over RRT grows with scale (tie at small, 0.06 at medium, 0.08 at large). The authors' explanation: RRT's per-recurrence LoRA adapters ([[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]]) are chosen at training time and applied identically to every input — the diversity is fixed in advance. GRT's gate conditions on the actual state it is about to update.

The dense baseline still leads at medium (0.05 nats) and large (0.06 nats) at this token budget. At small scale GRT actually wins across three seeds: $3.145 \pm 0.004$ vs $3.188 \pm 0.056$, with GRT's *worst* seed (3.148) below dense's *best* (3.154). Note the dense baseline's much larger seed variance — one bad seed at 3.253.

isoParams — matched parameters, more compute:

| Scale | Dense | GRT | GRT FLOPs |
|---|---|---|---|
| Small | 3.15 | **3.04** (`1+10×10+1`) | 15.64 G vs 1.84 G |
| Medium | 2.84 | **2.76** (`2+20×4+2`) | 26.4 G vs 7.35 G |
| Large | 2.71 | **2.65** (`3+30×6+3`) | 109.0 G vs 21.1 G |

At medium, heavy-tail Poisson actually edges GRT out (2.74 vs 2.76) — worth noting since the abstract does not.

**Downstream (lm-eval-harness, zero-shot, large scale).** isoFLOPs GRT averages 42.08 vs dense 42.05 at 37% of the parameters — a wash, but it wins on ARC-Challenge (+1.80) and BoolQ (+2.60) while losing HellaSwag (−1.59). isoParams GRT averages 44.15, +2.10 over dense, winning 8 of 9 tasks, with LAMBADA (Standard) +8.29 and HellaSwag +4.31. The one loss is BoolQ, −5.29.

**Component ablation (small, 20k steps).** Each row adds to the one above:

| Step | $\Delta$ val loss |
|---|---|
| Recurrence alone | **+0.107** (worse than dense) |
| + prelude/coda split | −0.035 |
| + state noise $\epsilon_x$ | −0.018 |
| + prelude re-injection into every step | −0.022 |
| + elementwise gate | **−0.048** |

Only the full stack falls below the dense line at 3.157. The gate is the single largest contributor. Naive weight sharing on its own is clearly *worse* than not sharing.

**A striking horizon effect.** The same ablation run on medium at only 2,000 steps flips the last two rows: prelude re-injection gives −0.198 and the gate only −0.115. The structural components pay off immediately; the gate is a learned mechanism whose value accrues over training. If you ablate at short horizons you will draw the wrong conclusion about your own architecture.

**Hyperparameter sensitivity — the "it did not matter much" results.** Gate temperature $\tau \in \{0.5, 1.0, 2.0\}$ gives 3.62 / 3.60 / 3.63. Gate bias swept over $\{-2, 0, +2, +4\}$ at full horizon spans only 0.019 nats, and non-monotonically ($+4 \to 3.141$, $+2 \to 3.152$, $0 \to 3.160$, $-2 \to 3.151$). State noise: removing it costs 0.018, doubling it costs 0.019 — a flat symmetric optimum. Good news for robustness; also means the exact +4 bias is not the magic.

**Things that did not work.**
- *Running more loops at inference than at training.* $R \in \{8, 10, 12\}$ on a model trained at $R=6$ gives 2.69 / 2.72 / 2.74 — monotonically *worse* than 2.68. The recurrence has converged; extra steps add noise, not refinement.
- *Forcing the gate open ($g=1$) at inference.* Loss 5.26, exactly the prelude-only output. If you never write, six loops are one loop.
- *Forcing the gate shut ($g=0$).* Loss 12.03. Catastrophic. The two bracket the trained model's 2.68 — the learned gate is doing genuine selective blending, not degenerating to either extreme.
- *Anchor ablation.* Replacing the frozen $\mathbf{h}^{(\mathrm{pre})}$ with zeros: 8.08. With the raw input embedding: 3.73. With the *drifting* previous hidden state $\mathbf{h}^{(r-1)}$ (gate still active): 3.38, still 0.70 nats worse than the fixed anchor. So the win is specifically from a **stable, non-drifting reference point**, not just from having a second input.
- *Naive KV caching.* Each recurrence step needs its own KV cache, so memory multiplies by $R$. At batch 1 decoding is weight-dominated and GRT-Medium still lands at 0.55× dense memory, but at batch 32 the cache dominates and the saving shrinks to 0.91×. Averaging K/V across steps recovers 0.39× *and* slightly improves HellaSwag (33.90 vs 33.65) — mild regularisation. Reusing only the *last* step's cache was worst.
- *torch.compile* failed on their hardware for all recurrent methods, so the training-time comparison against dense (3h18m vs GRT's 6h18m) is unfair to the recurrent side.

**Latency.** Compiled, GRT-Large is 3.67 ms/tok vs dense 3.33 (+10%) with 639 MB vs 1570 MB peak. In eager mode the overhead is +23% — so roughly half the gap is kernel-launch overhead from the extra elementwise ops, not arithmetic.

**Data budget.** At small scale GRT holds a better Pareto frontier at every token budget. At medium and large, dense leads at low data but the gap narrows as tokens grow, and at medium GRT overtakes once the budget is doubled. Suggests shared weights need more data to saturate their capacity ([[Training Compute-Optimal Large Language Models (Chinchilla)|Chinchilla]]-flavoured caveat: these comparisons are budget-dependent).

## Worth Remembering

**The mechanistic analysis is the most interesting part of the paper**, and it is all in the appendix.

*Loss is front-loaded.* Prelude-only loss is 5.29. One shared-block application drops it to 3.77 (−1.52 nats). Step 2 reaches 3.14. The remaining four steps buy only 0.46 nats down to 2.68. So **77% of the total improvement happens in the first two steps**. [[KL Divergence]] to the final output distribution falls from 2.61 (prelude) to 0.46 (step 2) to 0.014 (step 5) — the answer is basically committed by step 4.

*The gate goes write-heavy then copy-heavy.* Mean gate is ~0.82 at steps 2–3 (most open, most variable) and tightens to 0.87 by step 6. Effective openness — the ratio of the applied update norm to the proposal norm — declines monotonically from 0.182 to 0.066. Copy-saturated dimensions ($g > 0.95$) grow from 19.7% to 28.8%; write-saturated ($g < 0.05$) stay below $10^{-4}$. The model blends; it almost never fully overwrites.

*The block does not know it is being looped.* The raw proposal $\|\mathbf{o}^{(r)}\|_2$ is nearly constant across steps (93–99). All the decay in what actually enters the residual stream (17.6 → 6.5) comes from the gate closing. Same for attention: head roles are locked in by weight and stable across all six steps — 2 local/sharp heads, 3 previous-token heads, 7 broadcast, 86 general, out of 100. The block runs the same program every time; the gate decides how loud it plays.

*The projection is contrastive, not averaging.* The mean row-wise cosine between the $W_x$ half and the $W_h$ half of $W_{\mathrm{proj}}$ is **−0.189**, with only 0.16% of rows above cosine 0.5. The projection reads the *difference* between current state and anchor. Features unchanged since the prelude partially cancel; features that have moved get amplified. That is the clean mechanistic reason the fixed anchor helps: it is the reference for computing what changed.

*The gate's source shifts.* At step 1, gate logit variance splits 59% current state / 41% anchor. By step 6 it is 74% / 26%. This is baked into the weights: the leading singular value of $W_x$ is 9.87 vs 6.85 for $W_h$, a 44% gap that gets amplified as the state drifts from the anchor.

*Implicit compute routing, for free.* Sort tokens by post-prelude loss into deciles. Easiest decile gains 0.49 nats over six steps; hardest gains 5.02 — a **10× difference**, near-perfectly linear in initial difficulty ($R^2 = 0.998$). No router, no halting mechanism, no explicit difficulty signal — the elementwise gate produces per-position compute allocation as a side effect. This is a cheaper answer to the same question Mixture-of-Depths and Mixture-of-Recursions attack with routers ([[Sparsely-Gated Mixture-of-Experts Layer|conditional computation]]).

*A representational phase transition.* Centered Kernel Alignment — not in the vault, but: CKA between prelude and step 1 is 0.996 (nearly identical), then drops to ~0.33–0.65 against steps ≥2, and steps 2–6 form a tight cluster (pairwise ≥ 0.907). Cross-model against dense GPT-2 Large: prelude/step-1 peak at baseline layer 2, but *every* later step peaks at layer 11, and nothing aligns with layers >20 (CKA < 0.50). GRT compresses the computational path — it does not reproduce the dense model's deep-layer hierarchy. That may be exactly why HellaSwag drops in the isoFLOPs setting.

**Limitations the authors own.** $R$ is fixed at inference with no per-token halting. Gate bias and noise magnitudes may need re-tuning outside the GPT-2 family. The optimal sharing fraction (how many blocks in the prelude vs core vs coda) varies with scale and was not studied systematically — the three configs used (`1+1×10+1`, `2+5×4+2`, `1+5×6+5`) look hand-chosen.

**Practical caveats.** (1) The $R\times$ KV cache is the real deployment cost and it eats the parameter saving at large batch — budget for averaged-KV or [[GQA- Training Generalized Multi-Query Transformer Models|GQA]]-style sharing before you count the win. (2) Everything is at GPT-2 scale on ~9.8B tokens; the data-budget crossover means these results might reverse at modern token counts. (3) The extra elementwise ops cost real latency in eager mode; you need `torch.compile` to get the +10% number, and it did not compile on their hardware during training. (4) The "matches GPT-2 Small" claim rests on a small-scale seed comparison where the dense baseline had one bad run.

**Open question.** The paper argues 77% of the work happens in the first two steps and extending $R$ at inference hurts. So what is the "test-time compute dial" actually buying beyond step 4? The early-exit story is real (you can cheaply exit *early*), but the deep end saturates fast — this is not [[Chain-of-Thought Prompting Elicits Reasoning in LLMs|chain-of-thought]]-style unbounded thinking, it is a fixed budget with a soft floor.

## Links

Related: [[Attention Is All You Need]] · [[Long Short-Term Memory (Neural Computation)]] · [[Gated Activation]] · [[Layer Normalization]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Sparsely-Gated Mixture-of-Experts Layer]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Mixed Precision Training]] · [[GQA- Training Generalized Multi-Query Transformer Models]] · [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Dropout- A Simple Way to Prevent Overfitting]] · [[KL Divergence]] · [[Cross Entropy]] · [[Regularization]] · [[Linear Projection]] · [[Distilling the Knowledge in a Neural Network]]

New topics worth writing: ALBERT and cross-layer parameter sharing, Universal Transformers, Adaptive Computation Time, Mixture-of-Depths, Mixture-of-Recursions, Relaxed Recursive Transformers, Deep Equilibrium Models, Centered Kernel Alignment, effective rank of representations, early-exit inference and LayerSkip, Per-Layer Embeddings (Gemma 3n), stochastic depth, transformer FFN as key-value memory
