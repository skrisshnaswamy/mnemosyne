---
title: "Block Sparse Attention with Log-Linear Complexity"
authors: ["Bohao Tang", "Zhen Qin", "Yuqi Pan", "Zheng Li", "Pengfei Liu"]
year: 2026
arxiv: "2609.31093"
url: https://arxiv.org/abs/2609.31093
priority: Good-To-Read
read_on: 2026-09-29
tags: [paper, transformers, llm, diffusion, theory]
---
## The Core Idea

Attention is expensive because every query looks at every key. Block sparse attention fixes half of that problem: chop the keys into blocks of 64, pick the best $K$ blocks per query, and only attend inside those. The attention step then costs $O(N)$ instead of $O(N^2)$.

But the *picking* step was still quadratic. To choose the best 8 blocks out of $N/64$, you scored all $N/64$ of them. For every one of the $N$ queries. That is $O(N^2/C)$ — the same quadratic wall, just divided by 64. At 256K tokens, selection becomes the bottleneck, not the attention.

PISA makes the selection itself sub-quadratic. The trick is a **pyramid**: build a tree over the key blocks by repeatedly averaging pairs, so you get a coarse-to-fine hierarchy with $O(\log N)$ levels. Then search the tree top-down. Start at the root, score its children, keep the best $K$, expand only those into their children, score again, keep $K$ again. Repeat until you hit the leaves.

At every level a query scores at most $gK$ candidates ($g=2$ children, $K=8$ kept). That is a constant. With $O(\log N)$ levels, one query costs $O(\log N)$ to route. All $N$ queries cost $O(N \log N)$. Decoding one token costs $O(\log N)$.

> [!NOTE] Pyramid Top-K selection
> Instead of scoring every key block against every query, build a tree of pooled key summaries and walk it top-down, keeping only the $K$ best branches at each level. Turns block selection from $O(N^2)$ into $O(N\log N)$. ^pyramid-topk

The second piece is *how* you score a candidate. Averaging the dot products inside a block is the obvious choice (that is what MoBA and mean-pooled BSA do) but it is wrong in a specific way: attention does not average logits, it exponentiates them. One very hot key inside a block should make the whole block attractive. Averaging washes that out. PISA scores with **LogSumExp** instead, which is the exact quantity attention mass is proportional to.

$$s_{t,i}^{(\ell)} = \log \sum_{r \in \mathrm{Ch}_{\ell-1}(i)} \exp\!\left(\frac{q_t^\top \bar{k}_r^{(\ell-1)}}{\sqrt{d}}\right)$$

Why this did not exist before: log-linear hierarchical selection existed in *training-free* form (HiP) and for diffusion transformers (LLSA), but every *trainable* sparse-attention method that got pretrained from scratch — [[Sparsely-Gated Mixture-of-Experts Layer|MoBA]]-style, NSA, HiLS — kept flat quadratic selection. PISA is the first trainable one that is $O(N\log N)$ in prefill *and* $O(\log N)$ in decode. What it unlocks is that selection stops being the thing you pay for at long context: at 256K tokens it is 9.95× faster than flat selection.

## The Methodology

### Building the pyramid

Level 0 is the raw keys, one per token. Level 1 groups $C = 64$ adjacent keys into a leaf block — the same blocks ordinary block sparse attention uses. Above that, every level merges $g = 2$ adjacent blocks:

$$M_{\ell+1} = \lceil M_\ell / g_\ell \rceil, \qquad g_0 = C = 64, \quad g_\ell = 2 \ \ (\ell \ge 1)$$

Each block carries a summary vector, computed by [[Embeddings|mean pooling]] its children:

$$\bar{k}_i^{(\ell+1)} = \frac{1}{|\mathrm{Ch}_\ell(i)|}\sum_{r \in \mathrm{Ch}_\ell(i)} \bar{k}_r^{(\ell)}$$

When the groups are equally sized all the way down, a block's summary is exactly the mean of every original key it covers. Queries are never pooled — they stay token-wise.

### Walking the tree down

For query $q_t$, start at the coarsest level $L$ with candidate set $\mathcal{A}_t^{(L)} = \{1\}$ (the root). At each level:

1. Score every candidate with LogSumExp over its children's summaries.
2. Keep the top $K$ → $\mathcal{I}_t^{(\ell)}$.
3. Expand: $\mathcal{A}_t^{(\ell-1)} = \bigcup_{i \in \mathcal{I}_t^{(\ell)}} \mathrm{Ch}_{\ell-1}(i)$.

Since $|\mathcal{I}_t^{(\ell)}| \le K$ and each block has $g = 2$ children, $|\mathcal{A}_t^{(\ell-1)}| \le gK = 16$. Bounded at every level, which is the whole reason the complexity works. If a level has $\le K$ candidates, everything is kept without scoring.

The final set $\mathcal{I}_t^{(1)}$ is the leaf blocks. Attention then reads the *original* keys and values from those blocks — the summaries are only ever used for routing.

One important asymmetry: at the leaf level the LSE is over 64 *real* keys, so a finished leaf gets its **exact** attention-mass score. Intermediate levels use child summaries, which is an approximation.

### Why LSE and not the mean

Appendix A does the [[Derivative|Taylor]] expansion. For logits $z_1 \dots z_m$ with mean $\bar z$ and variance $\mathrm{Var}(z)$:

$$\log \sum_j e^{z_j} = \log m + \bar z + \tfrac{1}{2}\mathrm{Var}(z) + \text{higher order}$$

This gives three methods, and the paper ships all three as ablations:
- **PISA-1**: $s = \bar z$ — first order. This *is* mean-pooled scoring.
- **PISA-2**: $s = \bar z + \tfrac12 \mathrm{Var}(z)$ — second order.
- **PISA**: full LSE, no truncation.

There is also a sandwich bound via Jensen. With $m_j$ the child mean logits and $F_q(B)$ the parent's normalised raw-key LSE:

$$\frac{1}{g}\sum_j m_j \;\le\; \log\Big(\tfrac{1}{g}\sum_j e^{m_j}\Big) \;\le\; F_q(B)$$

So LSE-over-child-means sits between the mean score and the true value — never worse than the mean as an estimate of the real block mass. It can exploit differences *between* children, but cannot see variation *inside* a child.

### Grouped-query handling

Following NSA, all query heads sharing a KV head ([[Grouped Query Attention|GQA]], here $G_Q = 16$) share one selected block set. Scores are summed across the group:

$$u_{h,t,i}^{(\ell)} = \sum_{h' \in \mathcal{H}(h)} s_{h',t,i}^{(\ell)}$$

Note this is a *sum of LSEs*, whereas the full-attention reference used in diagnostics averages normalised masses. The paper is honest that these two aggregations can rank blocks differently even with exact scores.

### The kernels

Written in [[GPU processing|Triton]], and this is where the paper earns its "hardware-aware" label. The design is different for prefill vs decode because the IO arithmetic flips.

**Prefill, Stage 1 — intermediate levels.** Parallelise over (query position, KV head). Load the query vectors once, then loop $\ell = L \to 2$ entirely inside the kernel. Candidate indices live in registers; nothing is written out between levels. No dense query–key score matrix is ever materialised.

**Prefill, Stage 2 — leaf scoring.** After Stage 1 each query has $\le gK$ leaf candidates. Group queries that want the *same* leaf block into a cluster, tile it into groups of $Q_{\text{tile}} = 4$, and score them together. Each program loads one $d \times C$ key tile and a $(Q_{\text{tile}}G_Q) \times d$ query tile.

The reason for splitting into two stages is pure IO. A single-stage kernel would pay $O(NgKCd)$ in leaf key traffic. Two stages pay:

$$O\!\left(NgKd\left(G_Q + \frac{C}{Q_{\text{tile}}}\right)\right)$$

which is cheaper when $G_Q + C/Q_{\text{tile}} < C$. With their numbers: $16 + 64/4 = 32 < 64$. Wins by 2×.

**Decode.** One query per step, so there is no cross-query key reuse to harvest — the two-stage split would only cost you a second query load and extra kernel launches. So decoding uses a **single fused kernel** running all the way to $\ell = 1$, reusing the query already in registers. The mean pyramid is cached and only the current leaf's summary plus its ancestor path get updated per token.

### Training and backprop

Selected block indices are **held fixed** during backward. Gradients flow through attention on the selected entries only, never through the discrete top-$K$ decision. This is the same non-differentiable-routing compromise every hard-routing method makes.

Causal masking: candidates starting after the query are dropped. First, previous, and current leaf blocks (and their ancestor paths) are force-retained, following the NSA implementation in Flash Linear Attention.

### Training setup

Three scales — 418M, 1.47B, 2.67B — all decoder-only, pre-RMSNorm, SiLU-gated FFN, no dropout, [[RoPE]] on the high-frequency half with base 10,000, GPT-2 BPE (50,257 vocab). 32 query heads / 2 KV heads throughout.

- 100B tokens pretraining at 4K sequence length, $K = 8$.
- Then 10B tokens continued pretraining at 16K, $K = 32$, RoPE base bumped 10,000 → 80,000.
- [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], $\beta = (0.9, 0.95)$, weight decay 0.1, peak LR $3\times10^{-4}$.
- Schedule: 1K linear warmup → 90K plateau → 9K square-root decay to $0.1\times$ peak.
- Gradient clip 1.0, bf16 params with FP32 reductions, FSDP.
- $C = 64$ everywhere.

## Ablation Studies and Experiments

### The headline: it is roughly a wash on quality

At 2.67B after 100B tokens:

| Method | Loss ↓ | Avg ppl ↓ | MC avg ↑ | Containment avg ↑ |
|---|---|---|---|---|
| Full attention | 2.2209 | 12.28 | 58.23 | **53.01** |
| NSA | **2.2098** | **11.83** | **58.46** | 50.20 |
| HiLS | 2.2047 | 11.93 | 58.13 | 51.62 |
| BSA | 2.2184 | 12.11 | 58.61 | 51.13 |
| PISA | 2.2151 | 12.10 | 58.10 | **52.14** |

PISA beats BSA on loss at all three scales, and has the best containment average among sparse methods at all three scales. Commonsense reasoning (BoolQ, PIQA, [[BERT- Pre-training of Deep Bidirectional Transformers|HellaSwag]], WinoGrande, ARC, OBQA, SIQA) is essentially indistinguishable across every method — differences are within noise.

Full attention still wins containment at every scale. Nobody has closed that gap.

### The scoring ablation — the interesting one

PISA vs PISA-1 vs PISA-2 keeps everything identical except the score function. Training loss at 2.67B: PISA 2.2151 < PISA-2 2.2152 < PISA-1 2.2179 < BSA 2.2184. Tiny margins, but the ordering is **consistent across all three scales** — full LSE beats second-order beats first-order beats mean-pooled BSA.

The block-selection diagnostic makes the mechanism visible. They replay all four selectors on identical queries and keys from a frozen 418M full-attention checkpoint, 100 FDA prompts, all 24 layers, $K = 8$:

| Selector | Recall@8 | Captured mass | Mass ratio |
|---|---|---|---|
| BSA | 85.91 | 85.67 | 98.42 |
| PISA-1 | 84.75 | 85.50 | 98.22 |
| PISA-2 | 88.24 | 86.13 | 99.02 |
| **PISA** | **90.95** | **86.47** | **99.46** |

PISA-1 is *worse* than BSA on recall despite both being mean-based — the hierarchy costs you something when the score is crude. Add the variance term (PISA-2) and you beat BSA. Use the full LSE and you gain 5 points of recall over BSA. So the hierarchy alone does not buy accuracy; **the hierarchy plus a better score does.**

Note the recall floor: three blocks are force-retained, so 100% overlap is guaranteed on 3/8 slots. Real discrimination happens on the remaining 5.

### Speed, which is the actual point

Prefill selection latency, $C = 64$, $K = 8$, 32 query heads / 2 KV heads:

| $N$ | BSA | PISA (Q4) | Speedup |
|---|---|---|---|
| 4K | **0.167 ms** | 0.701 ms | 0.24× |
| 16K | **1.46 ms** | 1.58 ms | 0.92× |
| 32K | 5.10 ms | **3.18 ms** | 1.60× |
| 64K | 19.14 ms | **6.70 ms** | 2.86× |
| 128K | 76.68 ms | **14.45 ms** | 5.31× |
| 256K | 312.96 ms | **31.44 ms** | 9.95× |

**BSA is faster below 32K.** The log-linear crossover is real and it is not early. If your context is 4K–16K, this whole method is a loss. The constants matter more than the asymptotics until you are well past 32K.

The $Q_{\text{tile}}$ ablation confirms the IO analysis: Q4 (4 queries per key tile) beats the per-query fused version by 1.35×, 1.30×, 1.33× at 64K/128K/256K — and is *slower* at 4K, where there is not enough work to amortise the grouping.

### Long context (RULER, 2.67B, after 16K CPT)

Averaged over four needle-in-a-haystack families at 1K–16K:

| Method | Avg |
|---|---|
| Full attention | 66.24 |
| NSA | 61.07 |
| **PISA-2** | **62.92** |
| PISA | 62.80 |
| PISA-1 | 61.69 |
| BSA | 54.99 |

The BSA → PISA jump of **+7.8 points** is the biggest single number in the paper. On `niah_single` at 16K, BSA collapses to 59.67 while PISA holds 71.27 and PISA-2 holds 72.40. Better block selection matters most exactly where it should — retrieval over long context.

### What did not work, or worked awkwardly

- **PISA loses to PISA-2 on RULER** (62.80 vs 62.92) despite winning the selection diagnostic and the training loss. Full LSE is not uniformly best downstream.
- **Containment gets *worse* after continued pretraining for PISA.** At 2.67B post-CPT, BSA hits 68.15 containment and PISA only 66.40 — a reversal of the pretraining result. PISA-2 is best at 68.41. The paper reports this and does not explain it.
- **All sparse methods lose to full attention on containment** at every pretraining scale. The gap at 418M is large: 45.15 vs 41.77.
- **Below 32K the method is strictly slower than what it replaces.** No caveat, just true.
- **Intermediate scoring is genuinely approximate.** The Jensen bound says LSE-over-summaries is closer to the truth than the mean, but it "cannot recover variation within each child" — and the paper explicitly disclaims that lower numerical error implies correct block *rankings*.
- **The grouped-head sum-of-LSEs is not the reference aggregation.** Even with exact per-head scores, summing LSEs across a GQA group can rank blocks differently from averaging masses. Acknowledged, not fixed.

## Worth Remembering

**The complexity claim is conditional.** $O(N\log N)$ holds for *fixed* $g$, $K$, block size, head counts and head dimension. If you scale $K$ with $N$ — and you probably want to, since they went $K=8 \to K=32$ for 16K context — the log-linear story weakens. Nobody tested $K$ scaling.

**Two different sparsity fixes, do not confuse them.** This is not [[FlashAttention- Fast and Memory-Efficient Exact Attention|FlashAttention]]. Flash is exact and fixes memory traffic; PISA is approximate and fixes which keys you look at. They compose — PISA's attention stage is still a tiled kernel.

**The decode cache has amortised cost.** The pyramid stores $O((N/C)d)$ per KV head and grows by a fixed fraction when full. Even if every expansion copies the whole buffer, total copying through length $N$ is $O((N/C)d\log N)$, so the $O(\log N)$ per-token average survives.

**Selection is non-differentiable and they did not try to fix it.** Indices frozen in backward. Contrast with SeerAttention and DashAttention, which learn gates or differentiable selection rules. Open question whether differentiable routing would help here — the whole appeal of PISA is that its router is cheap, and a differentiable router usually is not.

**Limitations the authors own:** compute constrained them to 2.67B and 100B tokens. Both matter, and relative gains over baselines are exactly the sort of thing that moves with scale.

**Follow-up questions worth holding:**
- Does the crossover point move if $C$ changes? They only ever tested $C = 64$.
- Why does PISA's containment advantage invert after CPT at 16K? Is the pyramid mis-calibrated when $K$ jumps 8 → 32?
- The pyramid summaries are mean-pooled, not learned. NSA uses *learned* compression and does better on perplexity. What happens if you put a learned projection at each pyramid level?
- The mean pyramid must be rebuilt or updated per layer per KV head. IndexCache and LongCat share indices across layers via distillation. That seems directly composable.

**Practical caveat:** if you want this, your bar is ≥32K context and you must be training from scratch or doing heavy CPT — it is not a training-free drop-in like HiP.

## Links

Related: [[Attention]] · [[Sparse Attention]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Flash Attention]] · [[Grouped Query Attention]] · [[GQA- Training Generalized Multi-Query Transformer Models]] · [[Long Context]] · [[KV Cache]] · [[Prefill and Decode]] · [[Attention Is All You Need]] · [[Multi-Head Attention]] · [[RoPE]] · [[Query, Key, and Value (QKV)]] · [[GPU processing]] · [[Linear Attention]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Cost and Latency]] · [[Multi-head Latent Attention]]

New topics worth writing: NSA (Native Sparse Attention), MoBA / Mixture of Block Attention, LogSumExp scoring and attention mass, Triton kernel design, RULER benchmark, hierarchical top-K routing, HiP hierarchically pruned attention, block-sparse kernel tiling, straight-through vs frozen-index routing
