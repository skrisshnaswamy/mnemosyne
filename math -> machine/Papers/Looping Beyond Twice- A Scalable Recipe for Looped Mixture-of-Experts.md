---
title: "Looping Beyond Twice: A Scalable Recipe for Looped Mixture-of-Experts"
authors: ["Di He", "Pengxiang Li", "Da Chang", "Qingyan Meng", "Lu Yin", "Shiwei Liu"]
year: 2026
arxiv: "2610.01153"
url: https://arxiv.org/abs/2610.01153
priority: Good-To-Read
read_on: 2026-10-05
tags: [paper, transformers, llm, theory, scaling]
---
## The Core Idea

A **looped Transformer** reuses the same block of layers several times instead of stacking more unique layers. Six layers run three times gives you the depth of eighteen layers with the parameters of six. Depth becomes a third scaling knob, separate from parameter count and token count.

> [!NOTE] Looped Transformer ^looped-transformer
> One block of $M$ layers applied $H$ times in sequence. Effective depth is $MH$; parameter count stays at $M$ layers' worth. Attention and feed-forward weights are shared across every pass.

The problem: on **Mixture-of-Experts** models this stops working almost immediately. Every prior looped-MoE paper lands on two loops and stops. Three loops is usually worse than one. So the extra FLOPs you spend looping buy nothing, and the whole idea looks dead for sparse models.

This paper names the two reasons why, and both are specific to looping an MoE block.

**Reason one — the curse of depth, amplified.** Each residual update adds to the hidden state. Looping reuses the *same* block, so the updates it adds are correlated rather than independent. Variance in the residual stream grows pass after pass. The hidden state drifts away from anything the shared weights were trained to handle, and training blows up. In their 350M model, a plain 9-loop run reaches validation perplexity of **1313** — it diverges outright.

**Reason two — expert selection collapse.** This one is new and is the actual MoE-specific finding. The router is part of the shared block, so loop 1 and loop 7 see similar hidden states and route to the *same experts*. The paper measures this: cosine similarity between the per-loop expert-load distributions is high across all loop pairs, and each expert's load barely varies across loops. Loop 7 is not refining loop 6's work with different machinery — it is re-running almost the same computation. You pay nine times the FLOPs for roughly one block's worth of distinct computation.

> [!NOTE] Expert selection collapse ^expert-selection-collapse
> When a shared router is reused across loops, expert assignments stay nearly invariant from one loop to the next. Extra iterations add compute without adding *computational diversity*.

The fix, LOOM, follows one sentence: **each loop should contribute new computation while keeping the recurrent state stable.** Four pieces — two for stability, two for diversity. The payoff is that recurrence scales to 9–12 loops instead of 2. At 1.7B parameters on 60B tokens, nine loops drop validation perplexity from 9.62 to **7.77** and lift the seven-task zero-shot average from 42.4% to **47.7%**. Under matched FLOPs, five loops beat one: perplexity 18.36 → 16.54.

What it unlocks: a 1.7B MoE unrolled nine times behaves like a ~15B-parameter-deep stack while storing 1.7B of weights. That is a memory-for-compute trade, which matters when weights are the binding constraint.

## The Methodology

The backbone is a Llama-style pre-norm decoder — [[Grouped Query Attention|grouped-query attention]], [[RoPE]], sparse SwiGLU MoE layers in the DeepSeekMoE style with routed plus shared experts. Nothing exotic. The contribution is the four things bolted onto the loop.

Notation: $x$ is the token embedding, $h_\ell^t$ is the residual stream after layer $\ell$ of loop $t$, $h_0^1 = x$, and $h_M^H$ feeds the LM head. $M$ physical layers, $H$ loops. Attention weights and expert weights are shared across loops; **routers are not**.

### Piece 1 — residual scaling (stability)

Shrink every branch output before adding it to the residual stream:

$$\gamma = \frac{\lambda}{H\sqrt{M}}$$

$$\tilde h_\ell^t = h_{\ell-1}^t + \gamma\,\mathrm{Attn}_\ell(\mathrm{RMS}(h_{\ell-1}^t))$$

$$h_\ell^t = \tilde h_\ell^t + \gamma\,\mathrm{RMS}\big(\mathrm{MoE}_\ell^t(\mathrm{RMS}(\tilde h_\ell^t))\big)$$

Read the two factors separately. The $1/H$ bounds accumulation **across loops** — more loops, smaller each contribution. The $1/\sqrt{M}$ bounds variance growth **across physical layers**, which is the original curse-of-depth fix. $\lambda = 0.5$, fixed, not learned, same at every scale.

Note the extra `RMS` wrapped around the MoE output. That is a separate component and the ablation treats it separately — removing it costs more than removing the Looping Residual.

### Piece 2 — embedding re-injection (stability)

At the start of every loop $t \ge 2$, mix the original embedding back in:

$$g_t = \frac{\lambda}{t\sqrt{M}}, \qquad h_0^t = (1 - g_t)\,h_M^{t-1} + g_t\,x$$

Convex mixing, so the magnitude stays controlled. $g_t$ shrinks as $t$ grows — early loops get a strong anchor to the raw token, later loops are trusted to carry their own state. The recalibration of $g_t$ by $t$ (rather than a fixed gate) is the adaptation to looped MoE.

### Piece 3 — loop-specific routers (diversity)

Each layer gets a **separate router per loop**, $R_\ell^t$. Expert weights stay shared. Going from $H$ to $H+1$ loops adds router parameters only — a router is a $d \times E$ matrix, negligible next to the experts.

This is the direct cure for expert selection collapse. Figure 4 confirms it: cross-loop cosine similarity of expert-load distributions drops, and per-expert load variance across loops rises. Different loops genuinely touch different expert combinations.

Routers are sigmoid gates computed in float32, with a $z$-loss coefficient of $10^{-3}$, capacity factor 1.5, and a load-balance bias learning rate of $2\times10^{-3}$.

### Piece 4 — Looping Residual (diversity)

Diverse computation is useless if each loop overwrites the last. The Looping Residual is a fixed-decay **exponential moving average** of attention outputs, with two parallel memories.

Let $o_\ell^t = \gamma\,\mathrm{Attn}_\ell(\mathrm{RMS}(h_{\ell-1}^t))$. Each memory $A$ keeps an accumulator and a normaliser:

$$N_A \leftarrow \beta N_A + o_\ell^t, \qquad D_A \leftarrow \beta D_A + 1, \qquad r_A = \frac{N_A}{D_A}$$

with $\beta = 0.5$. Memory $H$ (global) persists across the whole recurrence. Memory $L$ (local) resets at the start of each loop. The attention residual update then becomes:

$$\tilde h_\ell^t = h_{\ell-1}^t + r_H + r_L$$

The direct attention add is **replaced**, not supplemented. Each memory is one tensor plus one scalar regardless of $H$ — no history buffer, elementwise updates only, so overhead is negligible.

### Training — segmented backpropagation

Split the $H$ loops into consecutive segments of at most $K=3$. Each segment gets its own language-modelling loss at its final loop. Gradients stay inside the segment; the hidden state and global EMA memory are passed forward with `stop_gradient`. Forward information flows through all $H$ loops; backward graphs stay three loops deep.

This is not just a memory trick — it is load-bearing for stability:

| $H$ | Peak memory (segmented) | Peak memory (full) | Train time at $H{=}12$ |
|---|---|---|---|
| 3 → 12 | ~12.6 GiB flat | 12.5 → 19.3 GiB | 12.6 h vs 26.1 h |

At $H=9$ on the 350M model, full backprop diverges: eval loss ~4.4, average accuracy ~34%. Segmented gets perplexity 14.80 and 41.1%.

### Scales and hyperparameters

| | ~100M | ~350M | ~1.7B |
|---|---|---|---|
| Physical layers $M$ | 6 | 10 | 15 |
| Hidden $d$ | 384 | 512 | 1280 |
| Q / KV heads | 6 / 3 | 8 / 4 | 20 / 10 |
| Expert width $I$ | 384 | 512 | 768 |
| Routed / shared experts | 24 / 2 | 30 / 2 | 30 / 2 |
| Top-$k$ | 6 (4+2) | 8 (6+2) | 8 (6+2) |
| Peak LR | $4\times10^{-4}$ | $2\times10^{-4}$ | $6\times10^{-5}$ |
| Tokens | ~5B | ~10B | ~60B |

Data is FineWeb-Edu, sequence length 1024, vocab 65,536. [[Decoupled Weight Decay Regularization (AdamW)|AdamW]] with $\beta_1=0.9$, $\beta_2=0.95$, weight decay 0.1, [[On the difficulty of training Recurrent Neural Networks|gradient clipping]] at 1.0, global batch 1024, warmup-stable-decay at 10/80/10 with a $0.1\times$ LR floor.

The detail worth noting: $\lambda = 0.5$, $\beta = 0.5$, $K = 3$ are **identical at all three scales**. Width and expert dimensions scale; the loop coefficients do not.

## Ablation Studies and Experiments

### Matched FLOPs — the experiment that decides whether this is useful

Backbone fixed at $M=10$, $d=512$, 80 routed experts, no shared experts, 700M parameters, 10B tokens. As $H$ grows, top-$k$ shrinks to hold compute roughly constant. Per-token cost in units of $d^2$:

$$f(H,k) = H(6 + 3k) = 3H(2+k)$$

(attention is $6d^2$: $3d^2$ for QKV, $2d^2$ for causal SDPA, $1d^2$ for the output projection; each SwiGLU expert is $3d^2$). The non-looped anchor is $f_1 = 84$. **These runs use full backprop, no segmentation** — so no extra LM-head or backward cost confounds the comparison.

| Loops | Eff. depth | top-$k$ | $f$ | Val. ppl | Avg. acc |
|---|---|---|---|---|---|
| 1× (baseline) | 10 | 26 | 84 | 18.36 | 38.84 |
| 2× | 20 | 12 | 84 | 17.37 | 39.00 |
| 3× | 30 | 8 | 90 | 16.91 | 39.34 |
| 4× | 40 | 5 | 84 | 16.58 | **39.53** |
| 5× | 50 | 4 | 90 | **16.54** | **39.53** |
| 6× | 60 | 3 | 90 | 16.57 | 39.50 |

The headline: at equal compute, four loops with top-5 routing beats one loop with top-26. Perplexity falls monotonically to loop 4–5, then flattens rather than collapsing. Prior work's two-loop ceiling is comfortably cleared.

Be honest about the magnitude though. Perplexity moves a real 1.8 points; average accuracy moves **0.69 points**. On seven zero-shot tasks that is near noise, and individual tasks go the wrong way — WinoGrande drops from 51.38 to 50.20 at loop 5, SIQA from 35.72 to 34.65. The perplexity signal is clean; the downstream signal at this scale is not.

### Unconstrained FLOPs — the three baselines are the interesting part

Fixed layout, top-$k$ held constant, so compute grows with $H$. Three partial baselines isolate what each stability piece does.

**100M, 5B tokens — validation perplexity:**

| Method | 3× | 6× | 9× | 12× |
|---|---|---|---|---|
| No tech. | 27.61 | 1311† | 1312† | 1587† |
| Embed inject only | 27.48 | 71.28† | 141.5† | 186.4† |
| Res. scale only | 28.99 | 34.67† | 33.78† | 35.64† |
| **LOOM** | 23.21 | 19.81 | 19.29 | **19.26** |

(† = unrecoverable loss spike. Baseline 1× is 26.26.)

**350M, 10B tokens — validation perplexity:**

| Method | 3× | 6× | 9× | 12× |
|---|---|---|---|---|
| No tech. | 23.75 | 1312† | 1313† | 1528† |
| Embed inject only | 42.76 | 803.6† | 780.6† | 864.2† |
| Res. scale only | 23.80 | 23.40 | 33.89† | 42.15† |
| **LOOM** | 18.27 | 15.35 | **14.80** | 14.86 |

(Baseline 1× is 20.07. LOOM 9× average accuracy 41.15% vs baseline 37.96%.)

**What did not work, stated plainly:**

- **Naive looping fails catastrophically.** At six loops and beyond, perplexity is in the 1300s. This is not degradation, it is divergence.
- **Embedding re-injection alone is worse than nothing.** At 350M, three loops gives 42.76 perplexity against 23.75 for the untouched loop. Injecting a fresh embedding into an already-exploding residual stream makes the explosion worse. The ordering matters: you must bound variance *before* anchoring helps.
- **Residual scaling alone delays the failure but does not prevent it.** It holds to six loops at 350M (23.40), then spikes at nine (33.89) and twelve (42.15).
- **Every single-technique baseline is worse than no looping at all**, even at three loops. There is no partial credit here. The combination is what works.

Read against Figure 3: residual scaling alone *does* flatten activation variance across both training steps and loop index — the variance diagnostic looks fixed — yet training loss stays above full LOOM. Controlled variance is necessary and not sufficient.

### Component ablation — 350M, 9 loops, step 5,000

| Variant | Val. ppl | Avg. acc |
|---|---|---|
| **LOOM** | **18.62** | **39.8** |
| w/o residual scaling | 24.63 | 38.6 |
| w/o embedding re-injection | 26.05 | 39.2 |
| w/o MoE RMSNorm | 26.50 | 39.0 |
| w/o Looping Residual | 19.25 | 39.3 |
| w/o routing refresh (shared router) | 19.83 | 39.2 |

The ordering is the useful content. The three **stability** components each cost 6–8 perplexity points when removed. The two **diversity** components cost 0.6 and 1.2. Stability is doing most of the work; diversity is the smaller, second-order win that turns "does not diverge" into "actually improves".

The MoE RMSNorm — a one-line change nowhere in the title or abstract — is the single most expensive thing to remove (+7.88 ppl). Worth filing away.

### Scaling to 1.7B, 60B tokens

| Loops | Eff. depth | Val. ppl | HellaSwag | Avg. acc |
|---|---|---|---|---|
| 1× | 15 | 9.62 | 37.1 | 42.4 |
| 3× | 45 | 8.94 | 39.1 | 43.9 |
| 6× | 90 | 7.91 | 46.6 | 46.7 |
| 9× | 135 | **7.77** | **48.8** | **47.7** |
| 12× | 180 | 7.84 | 48.6 | 47.5 |

Here the downstream numbers are no longer noise: +5.3 points average, +11.7 on HellaSwag, +7.5 on ARC-e. The gain from looping *grows* with model scale, which is the opposite of what you might fear. Peak shifts from 12 loops at 100M/350M to 9 loops at 1.7B — more physical layers, less recurrence needed.

## Worth Remembering

**The honest framing of the FLOPs result.** Unconstrained, LOOM is a large win. Matched, it is a modest perplexity win and a near-flat accuracy win at 700M. The paper is upfront about this and the 1.7B run has no FLOPs control at all — nine loops really is nine times the compute. So "recurrent depth is a compute-efficient scaling axis" is **current best understanding on perplexity, open on downstream quality at matched compute.** Nobody has run the 1.7B iso-FLOP version.

**Looping is a memory-for-compute trade, not a free lunch.** Nine loops of 1.7B stores 1.7B of weights and spends 15B worth of compute. That is good when weights are what you cannot fit — single-device serving, memory-bound deployment — and bad when compute is the binding constraint. Note also that the number of activated parameters per token is listed at ~0.63B for the 1.7B model, so the sparse-activation savings and the loop cost partly cancel.

**Segmented backprop is a general looped-model recipe, not a LOOM detail.** Memory flat across loop depth, roughly 2× faster at $H=12$, *and* better final quality than full backprop at every depth tested — including rescuing $H=9$ from divergence. If you are training any looped model, do this first. The reason it improves quality (and not merely cost) is not explained; truncating gradients to three loops presumably removes the same long-dependency pathology that makes deep [[Backpropagation#^exploding-gradients|BPTT]] hard.

**The two-loop ceiling in prior work was an engineering artefact, not a property of recurrence.** SMELT, Loop the Loopies and Nanbeige all independently landed on $H=2$. The honest reading of this paper is that they were all hitting the same two bugs. That is a useful prior: when several groups converge on the same suspiciously small hyperparameter, suspect a shared failure mode rather than a law of nature.

**Expert selection collapse is a new named failure and worth carrying into other contexts.** Any time you reuse a learned routing or selection decision across repeated passes over similar state, expect it to repeat itself. The cure — a fresh router per pass over shared parameters — is cheap and generalisable.

**Zero hyperparameter retuning across 17× scale.** $\lambda=0.5$, $\beta=0.5$, $K=3$ held from 100M to 1.7B. The $1/(H\sqrt{M})$ form is doing real work in absorbing the scale dependence, which is the same philosophy as [[Tensor Programs V- Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (muP)|μP]] applied to the loop axis rather than the width axis.

**Caveats for anyone wanting to use this:**
- Inference latency is $H\times$ the base model, serially. No parallelism recovers it. For a latency-sensitive product this is disqualifying.
- All training is on FineWeb-Edu at sequence length 1024. Nothing here tests long context, instruction tuning or anything post-training.
- $\lambda$, $\beta$ and $K$ are asserted, not swept. There is no sensitivity analysis.
- $\beta = 0.5$ means the EMA memory has a very short horizon — roughly a two-step effective window. Whether anything is genuinely being carried across *loops* rather than across the last couple of layers is untested.
- The Looping Residual replaces the attention residual entirely. That is an aggressive change to the identity path, and it is the component the ablation says matters least.

**Open questions.** Does the loop-depth optimum keep shrinking with model size — would a 7B model prefer 5 loops, and does the benefit vanish entirely by 70B? Does variable test-time loop count work, i.e. can you train at 9 and serve at 3 for easy prompts? And can the per-loop routers be shared-but-conditioned on $t$ (a loop embedding added to the router input) instead of fully separate, which would be cheaper and might generalise to unseen loop counts?

## Links

Related: [[Mixture of Experts]] · [[Sparsely-Gated Mixture-of-Experts Layer]] · [[Gated Recurrent Transformers- Expressive Depth through Recurrent Modulation]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Layer Normalization]] · [[On Layer Normalization in the Transformer Architecture]] · [[Backpropagation]] · [[Grouped Query Attention]] · [[RoPE]] · [[Gated Activation]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Tensor Programs V- Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (muP)]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[Perplexity]] · [[Test-Time Compute]] · [[Attention Is All You Need]]

New topics worth writing: Curse of depth in deep residual stacks, Universal Transformer, Recurrent depth / latent reasoning, Expert routing collapse, Segmented / truncated backpropagation through depth, Residual branch scaling, Input re-injection, ALBERT parameter sharing, FineWeb-Edu, Mix-LN, Chain-of-Experts, Fixed-point looped transformers
