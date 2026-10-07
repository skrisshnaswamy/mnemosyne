---
title: "FlashPrefill V2: Block-Sparse Prefill Attention for Long-Context LLM Serving"
authors: ["Qihang Fan", "Huaibo Huang", "Zhiying Wu", "Bingning Wang", "Ran He"]
year: 2026
arxiv: "2608.19758"
url: https://arxiv.org/abs/2608.19758
priority: Low-Priority
read_on: 2026-10-01
tags: [paper, transformers, llm, theory]
---
## The Core Idea

Attention during the **prefill** stage — the pass where the model reads the whole prompt before writing anything — costs $O(L^2)$ in the prompt length $L$. At 128K tokens that dominates the time-to-first-token. The standard fix is **block-sparse attention**: cheaply guess which blocks of keys matter for each block of queries, compute only those, skip the rest.

FlashPrefill V1 (same authors) already did the guessing well. This paper is about the three things that stop a good sparse-attention idea from actually shipping.

**1. Skipping blocks throws away probability mass.** When you drop a key block, you drop its $e^{s_i}$ terms from both the numerator and the denominator of the softmax. At 4% density the dropped mass is no longer negligible, and accuracy falls off a cliff. The fix: instead of dropping a block, replace all $B$ of its tokens with *one* averaged token. Cheap, and it puts the mass back.

**2. The kernel was built on the wrong generation of hardware tricks.** V1 sat on FlashAttention-2. On Hopper GPUs (H100, H20) that leaves most of the chip idle — you need TMA async copies, warp specialisation and pipelined GEMMs (what [[FlashAttention- Fast and Memory-Efficient Exact Attention|FlashAttention]]-3/4 do) before sparsity shows up as wall-clock time rather than being eaten by kernel inefficiency.

**3. Research kernels assume one contiguous KV tensor.** Real servers use a [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)|paged KV cache]] and continuous batching. A kernel that cannot read through a page table cannot be a serving backend.

> [!NOTE] Mean correction
> For each pruned key block, use its mean key $\bar k_J$ and mean value $\bar v_J$ as a single surrogate token, weighted as if it were $|\mathcal B_J|$ tokens. This restores the discarded softmax mass to *both* the numerator and the denominator. ^mean-correction

What it unlocks: at 128K on an H20, $27.2\times$ faster attention than FlashAttention-2 in BF16, $47.3\times$ in FP8, $30.5\times$ against a properly-tuned FA3/4 dense kernel — with RULER accuracy within 1.8 points of full attention, and end-to-end time-to-first-token in SGLang down $4.8\times$.

## The Methodology

### Stage 1 — decide which blocks to compute

Split $K$ and $V$ into blocks of $B=128$ tokens. Represent each block by its mean key

$$\bar k_J = \frac{1}{B}\sum_{k_i \in \mathcal B_J} k_i.$$

Why a mean is a sane proxy: $\exp(q\cdot\bar k_J)$ is the *geometric* mean of the per-token scores, so by AM–GM it under-estimates the block's true contribution $\frac1B\sum_i \exp(q\cdot k_i)$. It is a lower bound, not an unbiased estimate — but because scores inside one block do not vary much, the *ordering* across blocks survives, and ordering is all selection needs.

For each query tile $I$ (128 rows) and key block $J$, a fused kernel keeps a running max and a running sum of exponentials, never materialising the $L \times (L/B)$ score matrix:

$$m_{I,J}=\max_{q_i \in I}(q_i\cdot\bar k_J), \qquad \mathcal S_{I,J}=\sum_{q_i\in I}\exp(q_i\cdot\bar k_J - m_{I,J}).$$

Memory drops from $O(L^2/B)$ to $O((L/B)^2)$. Normalise per row to get $\mathrm{Score}_{I,J}$.

Selection avoids any sort. No top-$k$, no cumulative sum — just a threshold relative to the row's own peak:

$$\mathrm{thresh}_I = \alpha \cdot \max_J \mathrm{Score}_{I,J}, \qquad \alpha = 0.1.$$

Keep every block above it, plus forced blocks: 256 sink tokens at the front, a 512-token local window, the recent/diagonal blocks. Survivors are compacted **in place** (safe, because the write pointer never overtakes the scan pointer) into a **CSR index** — offsets plus a sorted list of selected block ids.

Measured density: ~70% at 4K, falling to ~4.6–4.9% at 128K. That falling curve is where the speedup comes from.

### Stage 2 — mean correction inside the softmax

Plain sparse attention truncates both sums to the selected set $\mathcal S$:

$$O \approx \frac{\sum_{J\in\mathcal S}\sum_{i\in\mathcal B_J} e^{s_i} v_i}{\sum_{J\in\mathcal S}\sum_{i\in\mathcal B_J} e^{s_i}}.$$

Corrected, with $\bar s_J = q\cdot\bar k_J/\sqrt d$:

$$\hat O = \frac{\sum_{J\in\mathcal S}\sum_{i\in\mathcal B_J} e^{s_i} v_i + \sum_{J\notin\mathcal S}|\mathcal B_J| e^{\bar s_J}\bar v_J}{\sum_{J\in\mathcal S}\sum_{i\in\mathcal B_J} e^{s_i} + \sum_{J\notin\mathcal S}|\mathcal B_J| e^{\bar s_J}}.$$

**Why it works, in plain terms.** Write $\delta_i = s_i - \bar s_J$, the wobble of a token's logit around its block mean. Expanding $e^{s_i}$ around $\bar s_J$:

- the **denominator** surrogate is accurate to second order — the first-order term vanishes because $\sum_i \delta_i = 0$;
- the **numerator** surrogate leaves a first-order error, the within-block covariance $\overline{\delta v} = \frac{1}{|\mathcal B_J|}\sum_i \delta_i (v_i - \bar v_J)$.

So the output shift from correcting (rather than exactly computing) one pruned block is

$$\Delta_J = \frac{w_J}{D}(\mu_J - \bar v_J) + O\!\left(\frac{w_J}{D}\overline{\delta^2}\right),$$

with $w_J$ the block's mass and $D$ the total. Three things bound it. Max-based thresholding caps each pruned block's share at roughly $\alpha$. The covariance vanishes if logit wobble and value deviation are uncorrelated, and is bounded by $\delta_{\mathrm{rms}}\sigma_v$ otherwise. And against just *dropping* the block, where the error is $\frac{w_J}{D}(\mu_J - O)$, the ratio is

$$\frac{\|\Delta_J\|}{\|\Delta_J^{\mathrm{none}}\|} = \frac{\|\mu_J - \bar v_J\|}{\|\mu_J - O\|} \sim O(\delta_{\mathrm{rms}}),$$

because the numerator is an *intra*-block deviation while the denominator is an *inter*-block one. Correction cancels the zeroth order and leaves only the wobble.

**The implementation detail that makes it free.** A corrected block enters the kernel as one extra pipeline iteration whose logit is shifted by $+\log|\mathcal B_J|$ — so a single mean vector stands in for $B$ tokens in both numerator and denominator, reusing the same MMA pipeline and [[Flash Attention#^online-softmax|online-softmax]] rescaling. No extra kernel launch. A two-pointer scan over the selected list builds a bitmask `todo = valid & ¬selected`; chunks where everything was selected are skipped identically by producer and consumer, so the pipeline stays in lockstep with no added synchronisation. Only blocks fully visible to the whole query tile get corrected; diagonal-band blocks are already covered by the forced sink/window/recent selections.

### The kernel, rebuilt for Hopper

Built on CUTLASS/CuTe. The Hopper techniques themselves are FlashAttention-3/4's; the contribution is extending that execution model to block-sparse traversal, GQA, paging and FP8.

**PackGQA.** Under [[Grouped Query Attention|grouped-query attention]] with ratio $g = H_q/H_{kv}$, the naive mapping gives one thread block per (query head, tile) pair, so $g$ thread blocks each reload the same KV block. Instead reshape

$$Q \in \mathbb R^{L\times H_q\times d} \longrightarrow Q' \in \mathbb R^{(g\cdot L)\times H_{kv}\times d},$$

with row $r$ decoding to $\text{pos}(r)=\lfloor r/g\rfloor$, $h(r) = h_{kv}\cdot g + (r \bmod g)$ — one fast divmod. Three wins: every staged KV block is consumed by all rows of the tile, so sharing happens in shared memory and never depends on L2 luck; the sparse index is defined per *KV* head, shrinking index metadata by $g\times$; and tiles stay fully occupied for any $L$.

**Warp specialisation + pingpong.** One producer warpgroup issues async TMA / `cp.async` loads of K, V and correction chunks into a two-stage shared-memory pipeline. Two consumer warpgroups run `wgmma`. Registers are reallocated between roles: producer 24 (40 on the paged path), each consumer 240 (232 paged). Inside a consumer, three things overlap per step:

$$\underbrace{S^{(n)}=QK_{(n)}^\top}_{\text{GEMM}_0} \parallel \underbrace{O \mathrel{+}= P^{(n+1)}V_{(n+1)}}_{\text{GEMM}_1} \parallel \underbrace{\tilde P^{(n)} = \text{online-softmax}(S^{(n)})}_{\text{softmax}}$$

so the MMA units and the exponentiation/rescaling units stop stalling each other.

**FP8.** Per-tensor scales $c_q,c_k,c_v$; operands dequantised inside the online softmax:

$$S = c_q c_k(\tilde Q\tilde K^\top), \qquad \tilde P = \lfloor 2^8 \exp(S-m)\rceil_{\text{e4m3}} \in [0,256], \qquad O = c_v \frac{\sum_t \tilde P^{(t)}\tilde V^{(t)}}{\sum_t \tilde P^{(t)}}.$$

The $2^8$ offset spreads probabilities across e4m3's usable range; it cancels in the softmax ratio. The awkward part: FP8 `wgmma` demands a K-major B operand. If $\tilde V$ is column-major it already fits, and instead $\tilde P$ is shuffled into A-operand register layout with warp shuffles plus byte permutes, un-permuted in the epilogue. If row-major, the producer transposes in shared memory via `LDSM.T` → byte-permute → `STSM`, permuting columns to dodge bank conflicts and restoring order at the end. Neither path touches global memory.

**Index-driven traversal.** The CSR list is walked backwards, $n^{(t)} = \mathrm{idx}[\mathrm{cu}[g]+T-1-t]$, matching the dense kernel's descending schedule. Producer and consumer each compute the same sequence independently, so control flow is identical by construction — block skipping needs no synchronisation — and $n^{(t+1)}$ is prefetched while $n^{(t)}$ is still in flight.

**Load balancing.** Cost per tile is proportional to its selected-block count, which is $O(L)$ for the last tiles and $O(1)$ elsewhere under causal masking. The dense heuristic of disabling KV splitting once tiles fill the SMs would serialise that heavy tail. So splits are always admitted, and the CSR list is partitioned **by selected-block count, not position range**, so each split gets equal actual work. Requests are sorted device-side by descending causal workload.

**Paging.** $\mathrm{addr}(b,j) = \mathrm{page\_table}[b, \lfloor j/p\rfloor]\cdot p + (j \bmod p)$, resolved on the fly with `cp.async`, arbitrary page size. A varlen persistent scheduler enumerates $(b,h,I)$ work units from per-request $(L^q_b, L^{kv}_b)$ — which is exactly continuous batching.

**Fused scoring.** Scoring and thresholding are one pass. The kernel keeps a running tile max and block energies $E_J = \sum_{q_i\in I} 2^{s_{iJ}\log_2 e - M}$, bitcasts $E_J$ into the index buffer itself, then rescales and thresholds. No separate score buffer, no second GEMM.

### Overhead accounting

Per KV head, with retained density $\rho$:

$$C_{\text{idx}} = O\!\left(gL_q\frac{L}{B}d\right), \quad C_{\text{corr}} = O\!\left(gL_q\frac{(1-\rho)L}{B}d\right), \quad C_{\text{attn}} = O(gL_q\,\rho L\,2d),$$

so the auxiliary work relative to attention is $O\!\left(\frac{2-\rho}{2\rho B}\right)$ — about 19% at $\rho = 4\%$, $B=128$. That gap is exactly why end-to-end speedup trails operator speedup.

### Serving integration

Dropped into SGLang as an attention backend. Prefill (extend) uses the two-stage pipeline; **decode falls back to dense**, since one query token per step leaves nothing to sparsify at block granularity. Page-level index is derived from SGLang's token-level map by striding every $p$ entries. Metadata built once per forward step, shared by all layers. $\alpha$, sink and window sizes are server arguments. No changes to the model, the cache layout or the scheduler.

## Ablation Studies and Experiments

**Setup.** NVIDIA H20 (Hopper) — chosen as the most widely deployed inference GPU in the authors' environment. CUDA 12.9, PyTorch 2.9.1, CUTLASS 4.3, SGLang 0.5.10. Models: Llama-3.1-8B-Instruct, Qwen3-4B-Instruct-2507, Qwen3-30B-A3B-Instruct-2507. Benchmarks: RULER (controlled lengths 4K–128K) and LongBench (21 realistic tasks). Baselines: MInference, FlexPrefill, XAttention, FlashPrefill V1, FlashAttention-2, and — crucially — **an FA3/4-aligned dense kernel sharing the same warp-specialised pipeline**, so measured gains isolate sparsity from kernel engineering. One config for all models: $B=128$, 256 sinks, 512 window, $\alpha=0.1$.

**Operator speedup over FA2 at 128K, batch 4** (Qwen3-30B-A3B):

| Method | 4K | 32K | 128K |
|---|---|---|---|
| MInference | 0.11× | 0.83× | 2.76× |
| FlexPrefill | 0.12× | 2.31× | 5.43× |
| XAttention | 0.86× | 2.42× | 3.42× |
| FlashPrefill V1 | 1.22× | 6.92× | 18.67× |
| **V2 (BF16)** | **1.75×** | **8.21×** | **27.19×** |
| **V2 (FP8)** | **2.78×** | **14.93×** | **47.26×** |

Note the three baselines are *slower than FA2* at 4K — their index overhead swamps the savings. V2 is already ahead at 4K. Against the FA3/4 dense kernel at 128K: 17.5× (BF16), 30.5× (FP8).

**RULER accuracy.** Average over 4K–128K, Qwen3-30B-A3B: full attention 92.05, V2 91.76, V2-FP8 91.39, best sparse baseline (FlexPrefill) 91.09. On Llama-3.1-8B: full 88.82, V2 87.79. Worst case is 128K where density is under 5% — V2 stays within 1.8 points.

**LongBench.** V2 has the best average of every sparse method on all three models: 49.31 vs 48.25 (XAttention) on Llama, 46.96 vs 45.90 on Qwen3-4B, 50.73 vs 49.61 on Qwen3-30B-A3B — all within 0.9 of full attention. The gap concentrates on retrieval-sensitive tasks: English passage retrieval, V2 holds 96.0–99.5 while baselines fall to 91.0–97.0. Summarisation is essentially lossless, occasionally above full attention (SAMSum 44.76 vs 43.59 — noise, not a win).

**The mean-correction ablation** (RULER, Qwen3-4B, drop pruned blocks vs correct them):

| | 4K | 32K | 64K | 128K | Avg |
|---|---|---|---|---|---|
| Full attention | 94.82 | 88.86 | 82.81 | 71.22 | 87.06 |
| V2 w/o correction | 94.38 | 87.21 | 80.92 | 68.86 | 85.77 |
| V2 | 94.49 | 87.48 | 81.67 | 69.76 | 86.23 |
| V2-FP8 w/o correction | 93.78 | 85.04 | 76.86 | **62.78** | 83.49 |
| V2-FP8 | 94.13 | 87.47 | 80.59 | **68.97** | 85.82 |

Read the last two rows. In BF16 correction is worth 0.5 average / 0.9 at 128K — nice but not decisive. **In FP8 it is worth 2.3 average and 6.2 points at 128K.** The authors' explanation: quantisation compresses the score margins, so the blocks the max-based threshold discards carry a larger share of the softmax mass, and recovering them matters more. Without correction, FP8 at 128K is unusable.

**Threshold sweep** (64K, Qwen3-4B, FP8; full attention = 82.81):

| $\alpha$ | 0.2 | 0.1 | 0.05 | 0.025 | 0.0125 |
|---|---|---|---|---|---|
| Density | 5.2% | 9.2% | 14.1% | 19.8% | 23.6% |
| w/o correction | 74.68 | 76.86 | 77.16 | 77.41 | 78.02 |
| with correction | 80.12 | 80.59 | 80.67 | 80.72 | 80.75 |

The corrected curve is nearly flat from 23.6% down to 5.2% density — a 4.5× swing in compute for 0.6 points. The uncorrected curve is not flat, and the gap widens as density falls (2.7 → 5.4 points). That is the real result: **correction is what makes the sparsity knob usable, not just what recovers a point of accuracy.**

**Correction overhead** (64K): at most 12.2 ms in the dense limit, 3.6–4.5 ms at 90% sparsity. Relative cost 5–14% up to 50% sparsity, rising to 18% (BF16) / 27% (FP8) at 90% sparsity — i.e. the relative cost is worst exactly where you need it most, but in absolute terms it is a few milliseconds.

**End-to-end TTFT in SGLang** (TP=4, four H20s, decode fixed to FA3/4). Qwen3-30B-A3B at 128K, batch 16: dense 123.2 s → 36.2 s (BF16, 3.40×) → 25.5 s (FP8, 4.83×). Speedup is nearly batch-invariant from 1 to 16, so sparsity composes with continuous batching. At 4K, where density is ~70%, BF16 is at parity (1.01–1.09×) — the index stage eats the savings — while FP8 already gives 1.1–1.8×.

**Open-loop serving** (Poisson arrivals, mixed 4K–128K prompts, 64 output tokens). The dense backend is saturated at every rate: P50 TTFT 77–106 s, throughput pinned at 0.31–0.37 req/s regardless of arrival rate. V2 roughly doubles request throughput (0.70–0.76 req/s); FP8 reaches 0.88–1.34. The interesting second-order effect: **decode gets faster too** — P50 TPOT improves 1.6–2.4× (BF16) / 2.3–8.8× (FP8), because long dense prefills block the decode steps of concurrently running requests. Speedup is largest at the *lowest* arrival rate (5.1× P50 at 1 req/s), where shorter prefills shrink queueing dramatically.

### What did not work, or worked worse

**Chunked prefill is the honest weak spot.** Bounding tokens per scheduling step is standard in production to protect inter-token latency. It barely touches the dense backend but eats V2's margin, for two reasons: the index selection is re-run on every chunk, and the mandatory tail/diagonal blocks push the effective density of a short chunk up. At 8K chunks, Qwen3-30B-A3B's P50 TTFT speedup collapses from 2.2× to **1.2×**. 16K chunks recover most of it (1.8–2.0× BF16, 2.8–3.0× FP8). The authors' own recommendation: do not go below 8K chunks.

**Decode gets nothing.** One query token per step means there is no block-level structure to exploit; decode falls back to dense.

**Short contexts get little.** At 4K density is 70–76%, so BF16 is at parity with a good dense kernel. Sparsity here is not a free lunch, it is roughly a wash.

**Against the production kernel.** Compared with Tencent HPC-Ops' block-sparse attention (FP8 only, 64K), V2 is 6–7% faster at every sparsity level, with the two dense references comparable (123.7 vs 133.4 ms) — so the gain is in the sparse path, not the backbone. The design deltas: CSR index per KV head (3 MB at 64K) vs a padded dense mask per query head (8 MB at 64K, 32 MB at 128K, regardless of density); pingpong GEMM–softmax overlap vs serial; `cp.async` paging at arbitrary page size vs per-page TMA needing page size divisible by 128; count-balanced KV splits vs none; fused correction vs none.

## Worth Remembering

**The numbers that actually matter for a serving decision.** Operator speedup is 27–47× at 128K. End-to-end TTFT speedup is 2.1–4.8×. That gap is not marketing slippage — it is Amdahl's law: attention is only part of prefill FLOPs, and the index stage plus correction add up to ~19% at 4% density. Always ask which number a sparse-attention paper is quoting.

**Correction is a precision enabler more than an accuracy fix.** The 6.2-point recovery at 128K FP8 versus 0.9 in BF16 is the most actionable result here. If you plan to serve quantised, the mean-correction term is not optional. Mechanistically: FP8 squashes the score margins, so what the threshold discards is proportionally more of the softmax mass. Worth holding on to as a general pattern — approximation errors from sparsity and from [[Quantization|quantisation]] are not independent, they compound.

**Max-based thresholding earns its keep twice.** Once for speed (no sort, no cumulative sum, unlike top-$k$/top-$p$ methods), and once for the error bound — capping each pruned block's mass share at $\approx\alpha$ is precisely what makes $\rho_{\mathrm{pr}} = \sum_{J\notin\mathcal S} w_J/D$ small in Eq. 9. The selection rule and the error analysis are the same object.

**PackGQA generalises beyond this paper.** Reshaping $Q$ so all $g$ query heads of a KV group live in one tile makes group sharing a shared-memory guarantee rather than an L2 cache hope, and shrinks sparse index metadata $g\times$. That second benefit is why their index is 3 MB where a dense mask is 8 MB. Any GQA kernel with per-head metadata should consider this.

**Count-balanced KV splitting.** Under causal masking, the last query tiles have $O(L)$ selected blocks and the rest $O(1)$. Splitting by *position range* (the dense heuristic) leaves the heavy tail serialised. Splitting the CSR list by *selected-block count* equalises real work. This is a general lesson for any irregular-workload kernel, not an attention-specific one.

**Honest limitations the authors state.** Decode is untouched. Chunked prefill below 8K degrades the win sharply. Short contexts (≤4K) are near parity in BF16. Density is not controlled directly — you set $\alpha$ and the density falls out, which means capacity planning has to be empirical. And the correction is **zero-order only**: the first-order covariance term $\overline{\delta v}$ in the numerator is left on the table, which is presumably where a V3 would go.

**Not shown, and I would want it.** All timings are H20 only — no H100/H200/B200, and H20 is a compute-throttled part, so the compute-bound-to-memory-bound balance may differ elsewhere, which would change how much the FP8 path buys. Accuracy is instruct-model inference only; nothing about training with this pattern (contrast NSA, MoBA, InfLLM-v2, which make sparsity learnable). And the 128K RULER gap of 1.8 points is a benchmark average — no per-task breakdown of *which* retrievals the 4.6% density loses.

**Connections.** This is the sparse counterpart to [[FlashAttention- Fast and Memory-Efficient Exact Attention|FlashAttention]]'s exactness story: FlashAttention is exact and fights memory traffic, block-sparse attention is approximate and fights FLOPs, and [[Sparse Attention#^sparse-vs-flash|the two are routinely confused]]. The paging half is downstream of [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)|PagedAttention]]. The mean-surrogate trick has a sibling in Sol-Attn (video diffusion), which the authors cite. The sink-plus-window forced selection is the StreamingLLM [[Sparse Attention#^attention-sinks|attention sink]] result showing up as a hard constraint in a selection rule.

## Links

Related: [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Flash Attention]] · [[Sparse Attention]] · [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]] · [[KV Cache]] · [[Grouped Query Attention]] · [[GQA- Training Generalized Multi-Query Transformer Models]] · [[Fast Transformer Decoding- One Write-Head is All You Need (MQA)]] · [[Attention]] · [[Attention Is All You Need]] · [[Prefill and Decode]] · [[Long Context]] · [[Continuous Batching]] · [[Quantization]] · [[Cost and Latency]] · [[GPU processing]] · [[Lost in the Middle]] · [[Block Sparse Attention with Log-Linear Complexity]] · [[Multi-head Latent Attention]] · [[Linear Attention]]

New topics worth writing: online softmax as an algorithm, CUTLASS/CuTe and warp specialisation, Tensor Memory Accelerator (TMA) and async copy, FP8 e4m3/e5m2 formats and per-tensor scaling, chunked prefill and Sarathi-Serve, RULER benchmark, LongBench benchmark, MInference, FlexPrefill, XAttention, Native Sparse Attention (NSA), MoBA / mixture-of-block-attention, StreamingLLM and attention sinks, CSR sparse index formats on GPU, prefill/decode disaggregation (DistServe, Mooncake), SGLang internals, time-to-first-token and TPOT as serving metrics, open-loop vs closed-loop load testing
