---
title: "Triadic Linear Attention: Three-Dimensional Recurrent States for Long-Context Sequence Modeling"
authors: ["Oliver Sieberling", "Bharat Runwal", "David Jin", "Ryan Chin", "Rameswar Panda", "Yoon Kim"]
year: 2026
arxiv: "2609.36529"
url: https://arxiv.org/abs/2609.36529
priority: Must-Read
read_on: 2026-10-05
tags: [paper, transformers, llm, scaling]
---
## The Core Idea

Linear attention keeps a fixed-size memory. Instead of a growing cache of past keys and values, it keeps one matrix $\mathbf{S}_t$ and writes into it:

$$\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{k}_t\mathbf{v}_t^\top, \qquad \mathbf{o}_t = \mathbf{S}_t^\top\mathbf{q}_t$$

That is the whole mechanism. Write with an **outer product** of two vectors, read with one vector. See [[Linear Attention]].

The matrix has $d^2$ numbers in it. That is the hard ceiling on how much it can remember. With $d$-dimensional keys, you can only store $d$ associations cleanly — after that, keys stop being orthogonal and old values bleed into new reads. Every improvement of the last few years ([[Gated Activation|gating]], the delta rule, better feature maps) changed *how* the matrix is managed, not how big it is.

The trick here: **use three vectors in the outer product instead of two.**

$$\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{k}_t \otimes \mathbf{k}'_t \otimes \mathbf{v}_t, \qquad \mathbf{o}_t = \mathbf{S}_t \times_1 \mathbf{q}_t \times_2 \mathbf{q}'_t$$

The state is now a 3D block of numbers, $d \times E \times d$. You write with a key, a *second* key $\mathbf{k}'_t$ of size $E$, and a value. You read by contracting both key axes with two queries. Set $E=1$ and $\mathbf{k}'=\mathbf{q}'=1$ and you get ordinary linear attention back exactly.

> [!NOTE] Triadic linear attention
> A recurrent layer whose memory is a third-order tensor, written by the outer product of three input-dependent vectors (key, second key, value) and read by contracting both key axes with two queries. `^triadic-def`

**Why this matters, and why nobody did it:** the state grows $E$-fold, but the parameter count grows by only two small projections — the ones that make $\mathbf{k}'$ and $\mathbf{q}'$. At $E=8$, the state is 8× bigger and the model has 1.2% more parameters. Every other way of enlarging the state (wider heads, wider values, more heads) fattens the big projection matrices, so you have to shrink the MLP to stay inside a parameter budget, and you lose more than you gain.

The reason it did not exist before is mostly plumbing. A 3D state does not fit in a GPU's registers, and the naive implementation makes every attention inner product $E$ times more expensive. Both problems turn out to be avoidable (§ Methodology), and the kernels are the real contribution alongside the idea.

The theoretical backing is old. Smolensky's (1990) tensor product representations bind a "filler" to a "role" with an outer product and store everything superposed by addition; unbinding is a contraction. An $n$-th order tensor product can store $d^{n-1}$ associations exactly. So going from $n=2$ to $n=3$ moves capacity from $d$ to $d^2$. The correlation-matrix-memory literature knew this in the 1970s and 80s. Schlag and Schmidhuber (2018) built a third-order recurrent state for graph traversal. What is new is using the third order purely as a capacity knob inside a modern, gated, delta-rule, chunk-parallel layer, at 1.3B scale.

**What it unlocks:** at 1.3B parameters, Triadic Gated DeltaNet with $E=8$ predicts the next token on long books *better than a Transformer* out to 64k context — while holding a state smaller than the Transformer's KV cache past about 6k tokens.

## The Methodology

### The read, written out

$$\mathbf{o}_t = \sum_{s \le t} (\mathbf{q}_t^\top\mathbf{k}_s)(\mathbf{q}'^\top_t\mathbf{k}'_s)\,\mathbf{v}_s$$

Two gates on each stored value, multiplied. A value is retrieved only if *both* its keys match. That is the capacity win: you now have $d \times E$ distinct addresses instead of $d$.

The second key does not correspond to anything in the data. The model picks how to use it.

### Forgetting

Standard gated linear attention multiplies the state by a scalar $\alpha_t \in [0,1]$ before writing. Here each slice along the second-key axis gets its own gate:

$$\mathbf{S}_t = \mathbf{S}_{t-1} \times_2 \operatorname{diag}(\bm{\alpha}_t) + \mathbf{k}_t \otimes \mathbf{k}'_t \otimes \mathbf{v}_t$$

So within a slice decay is scalar; across slices it is channelwise. With $E=8$ you get eight independent memory timescales per head, and this turns out to matter (see the half-life result below). Cheap, because $E$ is small.

### The delta rule

The delta rule erases what is stored for a key before writing the new value. Here the "currently stored" value is the *two-key* readout:

$$\mathbf{S}_t = \mathbf{S}_{t-1} + \beta_t\,\mathbf{k}_t \otimes \mathbf{k}'_t \otimes \big(\mathbf{v}_t - \mathbf{S}_{t-1} \times_1 \mathbf{k}_t \times_2 \mathbf{k}'_t\big)$$

with $\beta_t \in (0,1)$ a learned write strength.

Flatten $\bm{\kappa}_t = \mathbf{k}_t \otimes \mathbf{k}'_t$ into one $d\!\cdot\!E$ vector and the whole thing collapses to something familiar:

$$\mathbf{S}_t = (\mathbf{I} - \beta_t\bm{\kappa}_t\bm{\kappa}_t^\top)\mathbf{D}_t\mathbf{S}_{t-1} + \beta_t\bm{\kappa}_t\mathbf{v}_t^\top$$

That is Gated DeltaNet with a $d\!\cdot\!E$-dimensional key. So you can also read the whole paper as: *a feature map that expands the concatenated key $[\mathbf{k},\mathbf{k}'] \in \mathbb{R}^{d+E}$ into the Kronecker product $\mathbf{k}\otimes\mathbf{k}' \in \mathbb{R}^{d\cdot E}$.*

### Making it trainable — the two engineering moves

Linear attention trains in parallel using the **chunkwise form**: split the sequence into chunks of $C$ positions, do everything inside a chunk as masked attention, and carry only the chunk-boundary state recurrently.

$$\mathbf{O} = \mathbf{Q}\mathbf{S}_{[i]} + \operatorname{tril}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}, \qquad \mathbf{S}_{[i+1]} = \mathbf{S}_{[i]} + \mathbf{K}^\top\mathbf{V}$$

Naively running this with the $d\!\cdot\!E$ joint key costs $E\times$ more in every inner product *and* $E\times$ more state, which no longer fits on chip. Two fixes.

**1. Separate the joint key.** Kronecker inner products factorise:

$$(\mathbf{q}^r \otimes \mathbf{q}'^r)^\top(\mathbf{k}^s \otimes \mathbf{k}'^s) = (\mathbf{q}^{r\top}\mathbf{k}^s)(\mathbf{q}'^{r\top}\mathbf{k}'^s)$$

So the within-chunk attention is ordinary $d$-dimensional linear attention whose causal mask is replaced by a $C\times C$ matrix:

$$(\mathbf{Q}\mathbf{K}^\top) \odot \mathbf{R}', \qquad \mathbf{R}' = \operatorname{tril}(\mathbf{Q}'\mathbf{K}'^\top)$$

Cost goes from $C^2(d\cdot E)$ to $C^2(d+E)$. The only extra work over plain linear attention is the $C^2 E$ to build $\mathbf{R}'$.

**2. Tile the state along the value axis.** Written as slices, the chunk update is

$$\mathbf{O} = \sum_{e=1}^{E} \operatorname{diag}(\mathbf{q}'_e)\mathbf{Q}\mathbf{S}_{[i],e} + (\mathbf{Q}\mathbf{K}^\top \odot \mathbf{R}')\mathbf{V}, \qquad \mathbf{S}_{[i+1],e} = \mathbf{S}_{[i],e} + \mathbf{K}^\top\operatorname{diag}(\mathbf{k}'_e)\mathbf{V}$$

Different columns of the value axis never interact. So each thread block owns 32 value columns, keeps all $E$ slices of *those columns* in registers for the entire sequence, and the full head state is never assembled anywhere. At $E=8$ a head's state is 512 KiB in FP32 — twice a Hopper SM's register file. One block of it is 128 KiB. That is the whole reason this is practical.

> [!NOTE] Why tiling the value axis works
> In linear attention the value dimension is an independent output channel: nothing in the recurrence mixes value columns. Slices of the second-key axis interact only through a final sum. So the state can be cut into column strips, each living on its own SM. `^value-axis-tiling`

### Numerical stability — a small but instructive detail

The per-slice gates are evaluated as $\exp$ of a *difference of accumulated log gates*, element by element, costing $C^2 E$ exponentials. There is a cheaper route: pivot every factor around a mid-chunk reference $m_e$, writing $\gamma^r_e/\gamma^s_e = (\gamma^r_e/m_e)(m_e/\gamma^s_e)$, which turns $\mathbf{R}$ into the lower triangle of a $C\times E$ matrix product and needs only $2CE$ exponentials.

They rejected it. The pivoted factors have *positive* exponents, which overflow FP32 above ~88.7. Clamp to $[-88,88]$ and you silently corrupt the near-diagonal entries of $\mathbf{R}$ once a slice decays by more than $e^{176}$ within a chunk. In an FP64 simulation of one chunk, relative output error is 0.05 at within-chunk decay $e^{-200}$ and **1.5** at $e^{-600}$. The expensive version's exponents are always non-positive, so it cannot overflow at any decay rate and needs no clamp at all.

### Architecture and training

- 400M (24 layers, $d_{\text{model}}=1024$, 8 heads) and 1.3B (24 layers, $d_{\text{model}}=2048$, 16 heads). Head dim $d=128$ everywhere.
- Two base mixers: **Gated DeltaNet (GDN)** and **sGLA** (= GDN with the delta erase term removed).
- $\mathbf{k}', \mathbf{q}'$ come from one shared projection GEMM with $q,k,v$, then one shared causal depthwise convolution, then **softplus**, then $\ell_2$-normalise. SiLU is applied only to $q,k,v$. So a triadic layer runs the same two GEMMs and one conv as a GDN layer, just wider.
- Pretrain 50 tokens/parameter ($2.5\times$ [[Training Compute-Optimal Large Language Models (Chinchilla)|Chinchilla]]-optimal): 20B tokens at 400M, 65B at 1.3B. FineWeb-Edu, 4k context.
- Then long-context extend to 64k on 5 tokens/parameter, mixing FineWeb-Edu + PG19 + scientific PDFs.
- [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], peak LR $3\times10^{-4}$ pretrain / $10^{-4}$ extension, weight decay 0.1, linear warmup then cosine to 10% of peak. Batch ~0.5M tokens (400M) / ~1M (1.3B).
- Chunk size $C=64$. Kernels in the CuTe DSL of CUTLASS, warp-specialised: at $E=8$, two warpgroups hold four state slices each, a third loads inputs via the Tensor Memory Accelerator and forms $\mathbf{U}$, a fourth forms the output. Chunk-boundary states are stored in BF16 on the forward pass so the backward does not recompute them. Reductions use a fixed order with no float atomics, so gradients are bitwise reproducible.
- Transformer baseline: [[RoPE]] (base 10k → 2M at extension), QK-norm, [[Grouped Query Attention|GQA]] with 8 query heads per KV head. Both QK-norm and the 2M base were chosen *because they strengthened the baseline*.

## Ablation Studies and Experiments

### Capacity in isolation (MQAR)

Before any language modelling, they strip everything away: two layers, four heads, $d=16$, the sequence mixer is literally the bare triadic outer product — no convolutions, no gating, no delta rule, no nonlinearity. Task is multi-query associative recall: $N$ key-value pairs, then the same $N$ keys in random order, predict each value. Queries are read-only (they never write to the state).

**Every doubling of $E$ shifts the accuracy curve right by roughly a doubling of $N$.** At $E=16$ the model stores ~16× as many associations as plain linear attention, for a **1.08×** increase in non-embedding parameters.

This is the cleanest result in the paper. It isolates the claim from everything else.

### The state-matched comparison — this is the main table

400M, three seeds, parameter-matched (MLP width shrunk to compensate where projections grow). GDN half:

| Model | State (MB) | Wiki ↓ | PG19 ≤4k ↓ | PG19 4–16k ↓ | PG19 16–64k ↓ | Recall ↑ |
|---|---|---|---|---|---|---|
| Transformer | 12.6/1k tok | 11.11 | 15.39 | 14.41 | 13.92 | 41.6 |
| GDN base | 6.3 | 11.25 | 15.03 | 14.38 | 14.15 | 26.2 |
| Larger heads ($d{=}256$) | 12.6 | 11.22 | 15.10 | 14.42 | 14.17 | 26.6 |
| Grouped values (2/key) | 12.6 | 11.18 | 15.05 | 14.37 | 14.12 | 27.1 |
| Wider values ($d_v{=}256$) | 12.6 | 11.20 | 15.07 | 14.39 | 14.14 | 27.4 |
| More heads (16) | 12.6 | 11.32 | 15.25 | 14.54 | 14.28 | 27.5 |
| **Triadic ($E{=}2$)** | 12.6 | **11.08** | **14.90** | **14.21** | **13.95** | **28.4** |
| Larger heads ($d{=}512$) | 25.2 | 11.25 | 15.21 | 14.50 | 14.24 | 28.5 |
| Grouped values (4/key) | 25.2 | 11.35 | 15.35 | 14.63 | 14.36 | 27.3 |
| Wider values ($d_v{=}512$) | 25.2 | 11.41 | 15.41 | 14.70 | 14.44 | 27.5 |
| **Triadic ($E{=}4$)** | 25.2 | **10.93** | **14.80** | **14.08** | **13.79** | **31.1** |

**The negative result is the important half.** At $2\times$ state, every alternative barely moves. At $4\times$ state, **all three alternatives get worse than the 1× baseline on every PG19 range.** Wider values and grouped values need so many parameters that the MLP has to be cut from width 2816 to **640** to stay in budget. You pay for state with capacity you needed more.

Triadic adds two small projections, keeps the full MLP, and improves monotonically. The sGLA half shows the same pattern.

### Scaling $E$

At 400M and 1.3B, with $E \in \{1,2,4,8\}$. GDN beats the Transformer on early tokens but falls behind around 10k context. $E=2$ pushes the crossover out considerably; $E=8$ beats the Transformer at **64k** — past the point where the Transformer's KV cache exceeds the recurrent state. Language modelling improves with $E$ but with diminishing returns. Recall gains are largest on **FDA** and **SWDE**, which require copying from a long document.

1.3B, $E=1 \to E=8$: Wiki 8.08 → 7.87, PG19 16–64k 10.20 → 9.94, recall **36.1 → 44.4**, zero-shot 59.6 → 60.9.

RULER needle-in-a-haystack shows the same shape: bigger state keeps retrieval accurate out to longer contexts.

### Does the extra state actually get used for distant context?

Fix the prediction target to the last 4k tokens of a ≥64k-token book, vary how much context precedes it. GDN **saturates by 16k**. Triadic GDN ($E=8$) keeps improving through 32k and gains consistently more from added context. So the capacity is being spent on long-range information, not on local smoothing.

Two more diagnostics, both informative:

- **Write strengths grow.** Mean delta $\beta$ across tokens/heads/layers goes from **0.31 (GDN) to 0.50 (Triadic $E{=}8$)**, with a broader distribution. With more room, the model commits harder to each write instead of hedging.
- **Timescales spread out.** Rank the 8 slices in each head by half-life (tokens until a stored contribution halves). Median half-life ranges from **0.20 tokens** for the shortest-lived slice to **1080 tokens** for the longest. GDN has one gate per head, median half-life **5.1** tokens. The second axis is being used as a *multi-timescale* memory, not just a bigger bucket.

### Upcycling a pretrained model

Take a trained $E=1$ model, copy its forget gates to all 8 slices, initialise the $\mathbf{k}',\mathbf{q}'$ projections and convs from scratch, then do the long-context extension as a triadic model.

| 1.3B | Recall ↑ | PG19 16–64k ↓ |
|---|---|---|
| GDN base | 36.1 | 10.20 |
| Triadic $E{=}8$ from scratch | 44.4 | 9.94 |
| Triadic $E{=}8$ upcycled | **42.0** | **10.00** |

Upcycling recovers roughly half to three-quarters of the from-scratch gain, at both scales and for both mixers. Suggestive of a training recipe: pretrain small-state on cheap short-context data (which is most of it and needs little state), then grow the state when you move to long sequences — mirroring how a KV cache naturally grows with context.

### Hybrid: enlarge the linear state or the KV cache?

3:1 GDN/GQA-8 hybrid at 400M, with NoPE. Compare doubling the KV cache (GQA-4) against quadrupling the linear state ($E{=}4$).

| Model | State @4k | State @64k | Wiki ↓ | PG19 4–64k ↓ | Recall ↑ | NIAH ↑ |
|---|---|---|---|---|---|---|
| 3:1 base | 17 MB | 206 MB | 10.69 | 13.50 | 43.7 | 51.5 |
| Larger KV (GQA-4) | 30 MB | 407 MB | 10.67 | 13.47 | **45.3** | 53.0 |
| Triadic state ($E{=}4$) | 31 MB | **220 MB** | **10.63** | **13.39** | 44.5 | **56.3** |

Same memory at 4k; past ~4.6k tokens the triadic hybrid uses less, and at 64k it uses **almost half**. It wins on perplexity at every range, zero-shot, and NIAH; it loses to GQA-4 on the recall suite by 0.8, within seed spread. Honest reporting.

### What did *not* work

**Making the two key dimensions similar.** At fixed joint key $d_k \cdot E = 1024$ and $8\times$ state, with one scalar gate per head:

| | Wiki ↓ | PG19 16–64k ↓ | Recall ↑ |
|---|---|---|---|
| $d_k{=}128, E{=}8$ | **10.80** | **13.70** | **33.4** |
| $d_k{=}64, E{=}16$ | 10.84 | 13.73 | 32.5 |
| $d_k{=}32, E{=}32$ | 10.90 | 13.78 | 32.5 |
| $d_k{=}32, E{=}32$, tied keys | 10.89 | 13.79 | 31.9 |

Perplexity degrades as the factorisation becomes balanced. The asymmetric choice (a big first key, a small second key) is the right one. They note the balanced factorisations shrink the total key/query projections, which could matter for MoE architectures — but they cost quality here.

**Signed second keys.** Activation on $\mathbf{k}',\mathbf{q}'$ after the conv, at $E=8$:

| Activation | Wiki ↓ | PG19 16–64k ↓ |
|---|---|---|
| Softplus | 10.86 | 13.77 |
| Sigmoid | 10.86 | 13.77 |
| SiLU | 10.98 | 13.84 |
| Linear (none) | 11.06 | 13.98 |

Non-negative beats signed, consistently. Proposed explanation: with signed entries a token can write to one slice positively and another negatively, and those contributions cancel when the second query sums over slices. The slices stop being independent address space.

**The cheap-but-unstable $\mathbf{R}$ construction** (see Methodology) — rejected on numerical grounds, with a quantified FP64 error study rather than a hand-wave.

### Speed

One block, forward + backward, batch 4, H100. Triadic uses their CuTe kernels; GDN uses FlashQLA TileLang kernels; Transformer uses FlexAttention.

- Overhead of Triadic over vanilla GDN: **9–11%** at $E{=}2$, **14–15%** at $E{=}4$, **28–30%** at $E{=}8$.
- Triadic overtakes the Transformer at 4k ($E{=}2,4$) or 8k ($E{=}8$). At $E{=}8$ it is 3% slower than the Transformer at 4k and **5.1× faster at 64k**.

## Worth Remembering

**The reframing worth keeping.** Triadic linear attention *is* linear attention with a Kronecker feature map on the key. That connects it straight back to the early feature-map line of work (Performer, Random Feature Attention, Based). The novelty is not the map — it is realising that a Kronecker map lets you expand the key dimension $E$-fold while the inner products factorise, so you only pay in state, not in FLOPs per inner product. Then tiling makes the state payable too.

**Limitations the authors state.** 28–30% training slowdown at $E=8$, even with hand-written kernels. Recall on some tasks still trails the Transformer — the Transformer gets 41.6 recall at 400M with a KV cache that grows without bound, while $E{=}8$ gets 33.1 with a fixed 50 MB. The construction is applied to only two mixers (GDN, sGLA), a small slice of the literature.

**The claim about test-time training is the most provocative line in the paper.** TTT-MLP, LaCT, Titans, TTT-E2E all use much larger states than a conventional linear attention block. Since a gradient step can itself be written as a linear attention update, the paper suggests some of their reported gains may simply be *state capacity*, not the learning-to-learn framing. That is an open question, and a good one to hold onto when reading the next TTT paper.

**Practical caveats if you wanted to use this.**
- Keep $E$ small and asymmetric. $E=8$ with $d_k=128$ is the sweet spot measured here; balanced factorisations lose.
- Use a non-negative activation on the second key and query. This is not cosmetic — linear (no activation) costs 0.26 Wiki perplexity.
- The kernels are the artefact. Without them you either hold a 512 KiB-per-head state (does not fit) or pay $E\times$ in every chunk inner product.
- If you have an existing trained GDN, upcycling gets you most of the way for the cost of one long-context extension run. That is a genuinely cheap upgrade path.
- In a hybrid stack, prefer growing the recurrent state over growing the KV cache: better quality and roughly half the memory at 64k.

**Connections.** The second-key axis learning a spread of half-lives (0.2 to 1080 tokens) is the same thing people wanted from multi-timescale RNNs and clockwork RNNs, arrived at by accident. It is also exactly the structure [[Linear Attention|hybrid]] architectures try to get by interleaving layer types — here one layer does it internally.

**Follow-up questions.**
- Does $n=4$ help, or does the $d^{n-1}$ capacity scaling stop paying once write bandwidth, not storage, is the bottleneck?
- The spread of half-lives emerged without being asked for. What happens if you *initialise* the slice gates to a geometric ladder of timescales?
- Does the $\beta \to 0.50$ shift mean the delta rule was previously being suppressed by interference, and if so is write strength a usable diagnostic for "this model's state is too small"?
- Can the upcycling story be pushed further — grow $E$ at every curriculum stage rather than once?

## Links

Related: [[Linear Attention]] · [[Attention Is All You Need]] · [[GQA- Training Generalized Multi-Query Transformer Models]] · [[Grouped Query Attention]] · [[KV Cache]] · [[Long Context]] · [[Fast Transformer Decoding- One Write-Head is All You Need (MQA)]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Flash Attention]] · [[State-Space Model]] · [[Long Short-Term Memory (Neural Computation)]] · [[Gated Activation]] · [[Query, Key, and Value (QKV)]] · [[Mixture of Experts]] · [[Sparse Attention]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[RoPE]] · [[Perplexity]] · [[Mixed Precision training]] · [[GPU processing]] · [[Embeddings]] · [[Test-Time Compute]]

New topics worth writing: Gated DeltaNet, DeltaNet and the delta rule in linear attention, tensor product representations (Smolensky binding/unbinding), chunkwise-parallel training of linear RNNs, MQAR and the recall-throughput tradeoff, RULER, correlation matrix memories and higher-order associative memory capacity, dense associative memory / modern Hopfield networks, test-time training as a sequence mixer, Mamba and selective state spaces, Mixture-of-Memories, product-key memory layers, CUTLASS CuTe and warp specialisation, Tensor Memory Accelerator, upcycling pretrained architectures, NoPE
