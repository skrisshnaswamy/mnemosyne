---
title: "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression"
authors: ["DeepSeek-AI", ":", "Anyi Xu", "B. Li", "Bangcai Lin", "Bing Xue", "BingCheng Xian", "Bingzheng Xu", "Bochao Wu", "Bowei Zhang", "Boyi Deng", "C. C. Yu", "Chao Jin", "Chaofan Lin", "Chen Dong", "Chenbing Wang", "Chenfan Feng", "Chengda Lu", "Chenggang Zhao", "Chengqi Deng"]
year: 2026
arxiv: "2609.19969"
url: https://arxiv.org/abs/2609.19969
priority: Must-Read
read_on: 2026-09-19
tags: [paper, transformers, vision, theory]
---
## The Core Idea

The bottleneck for a long-running agent is no longer "how fast can attention run". It is **where do you put the KV cache, and how do you move it around**. An agent that reads a repo, runs tools, and comes back ten minutes later needs its context to still exist somewhere. In DeepSeek-V4-Flash that "somewhere" was HBM (fast GPU memory) plus SSD, and it was big.

> [!NOTE] KV cache
> The stored keys and values for every past token, so you do not recompute them each step. One per layer, per token. It grows linearly with context and is the main memory cost of serving. ^kv-cache-def

This model attacks that footprint from four directions at once, and the numbers are the point:

- **Global KV: 890 bytes per token**, about $1/4$ of V4-Flash and about $1/437$ of DeepSeek-V1. This is the part that must live in HBM.
- **Persistent KV (SSD / host RAM): about $1/8$ of V4-Flash**, because sliding-window KV is no longer saved at all.
- **8B active parameters during prefill, 16B during decode.** Reading is cheaper than writing.
- Decode FLOPs grow by only ~25% when context goes from 4K to 1M — a 256× context increase.

The four tricks, each simple on its own:

1. **Causal Encoder–Decoder (CED).** Split the 40 layers into a 20-layer encoder and a 20-layer decoder. The decoder's global KV is *projected from the encoder's last hidden state*, not from its own. So prefill only needs to run the bottom half. Prefill compute nearly halves.
2. **CSA2 — cross-layer sharing.** Most layers do not own a KV cache at all. They borrow one from an earlier layer, and often borrow that layer's Top-K selection too. Storage collapses along the *layer* axis, which prior work had only partly exploited.
3. **FP4 KV cache.** Four bits per value instead of eight. Half the bytes again.
4. **SWA Bounded Replay.** Stop storing the sliding-window cache on SSD. When it is missing, recompute it by replaying only the last $n_{\text{win}} = 128$ tokens instead of the mathematically correct $L \times n_{\text{win}} = 5120$ tokens. The reconstruction is *wrong*, and they ship it anyway because the quality loss is negligible.

That last one is the interesting philosophical move. Every previous version tried to reconstruct state exactly. This one accepts an approximate state at the cache boundary and trades a provable guarantee for an 8× storage cut. The same "close enough" logic shows up again in the decoder replay.

Why did this not exist before? Because each piece alone is either small or risky. Cross-layer KV sharing ([[GQA- Training Generalized Multi-Query Transformer Models|GQA]]'s idea, but across depth rather than across heads) costs quality. FP4 costs quality. Approximate SWA replay costs quality. Stacked, with quantisation-aware training and replay simulated during post-training so the model *learns* under the approximation, the costs mostly cancel out and the savings multiply.

## The Methodology

### The shape of the model

- 40 transformer layers, hidden size $d = 5120$. Layers 0–19 = causal encoder, layers 20–39 = decoder.
- 552B total backbone parameters, **plus 196B Engram (lookup memory) parameters**. Only 8B active per token in prefill, 16B in decode.
- Every [[Mixture of Experts|MoE]] layer: 1 shared expert + 384 routed experts, expert hidden dim 2304, **6 routed experts fired per token**. SwiGLU with output clamped at 10.
- Attention: 64 query heads, head dim 512, query compression dim 1280. Sliding window $n_{\text{win}} = 128$. Attention Top-K = 512 entries.
- Vision: a from-scratch ViT ("DeepSeek-ViT"), 32 layers, width 1024, 16 heads, patch 14, 2D-[[RoPE]] instead of learned positions, RMSNorm + SwiGLU, linear patch embedding instead of a conv (so Muon works on it). A $3\times3$ pixel-unshuffle cuts visual tokens 9×, supporting images up to $1344\times1344$.

### Causal Encoder–Decoder

Normally each layer $l$ builds its own K and V from its own hidden state $H_l$. CED breaks that for the top half:

$$C_l = H_{L/2} W_l^{KV}, \qquad Z_l = H_{L/2} W_l^{Z}, \qquad l > \tfrac{L}{2}$$

Every decoder layer projects its global KV from the *encoder's final* hidden state, with its own weights. So a decoder layer still gets a distinct cache — unlike YOCO, where the top half literally shares one cache — but you never have to run the decoder over the prompt to build it.

Sliding-window attention is the exception: each layer computes its own local KV from its own $H_l$, which keeps local depth intact. That is what forces the replay trick below. Prefill cost falls from $\mathcal{O}(NL)$ to $\mathcal{O}(NL/2 + n_{\text{win}} L/2) \approx \mathcal{O}(NL/2)$.

### CSA2 and its three modes

Each layer is *statically* assigned one of three modes. In all three it computes its own query and its own sliding-window KV.

| Mode | Owns global KV? | Owns indexer K? | Computes Top-K? |
|---|---|---|---|
| **Full** | yes | yes (projected from its own main KV) | yes |
| **Reindex** | no — borrows | no — borrows | **yes**, rescores the borrowed keys with its own indexer Q |
| **Reuse** | no | no | no — borrows the last computed indices too |

Reindex is the clever middle rung: storage is shared, but *which* 512 entries each layer reads can still differ. Cache sharing and index reuse are decoupled, which is exactly what earlier work (IndexCache reuses indices only; YOIO shares one routing network-wide; HySparse keeps dense layers around) did not do.

The actual layout:

- Layers 0–1: pure sliding-window attention, no global branch.
- Encoder layers 2–19: CSA2 with sequence compression $m=2$ (two tokens squashed into one cache entry), in **three groups of six**: `[Full, Reuse, Reuse, Reuse, Reuse, Reuse]`.
- Decoder layers 20–39: CSA2 with $m=1$ (no sequence compression), in **five groups of four**. First group `[Full, Reuse, Reuse, Reuse]`; the other four `[Reindex, Reuse, Reuse, Reuse]`.

So across 40 layers there are only **four** global KV caches. That is where the 4× comes from.

CSA2 also *simplifies* CSA from V4: the old compressor built each entry from $2m$ overlapping source entries with an absolute positional embedding baked in; CSA2 drops the overlap and drops the positional embedding, and derives indexer K by projecting the main KV rather than running a second compression path from hidden states.

### Hierarchical Sparse Indexer

Indexing is $O(N)$ per query — you score every visible position. With cross-layer reuse only four layers index, but four full scans of a 1M context still hurt in decode. So, in the decoder only, the first Full-mode layer does double duty: besides picking its own Top-512, it scores **blocks** (max score over the 8 positions in a block), takes the best 2048 blocks, and hands down a candidate pool of $2048 \times 8 = 16{,}384$ positions. Every later Reindex layer searches only that pool. Their per-query cost becomes **constant in context length** instead of linear.

This is introduced during *post-training*, and applied identically in training and inference, so the deeper indexers are optimised against the same restricted search space they will see at serving time.

### FP4 main KV cache

Format: **E2M1 values, one E4M3 scale per 16 channels** — NVFP4 without its second global scale. The justification is a clean bound: the largest trained RMSNorm weight is ~1, so after normalisation the 512-channel latent has $\ell_2$ norm $\le \sqrt{512} \approx 22.6$; [[RoPE]] is a rotation and preserves norms; so no channel can exceed ~22.6, while the format tops out at $448 \times 6 = 2688$. Observed max during training was ~10. The global scale buys nothing.

Quantisation-aware training is used, introduced during post-training. They quantise **after** RoPE (before RoPE was marginally more accurate but adds decode overhead). Sliding-window KV stays FP8 — it is more sensitive.

### SWA Bounded Replay

The dependency: layer 1's window covers the last 128 tokens, layer 2 sees layer 1's outputs over 128 tokens which each saw 128 before them, and so on. Exact reconstruction of $L$ layers of window state requires replaying $L \times n_{\text{win}}$ tokens. V4 called the no-storage version "Zero SWA Caching" and found it too expensive in production.

Bounded Replay replays **only $n_{\text{win}}$ tokens** and truncates the window to the replay segment. For a replay starting at position $s$, a query at $i$ attends to keys in $[\max(s, i-W+1), i]$. Two uses:

- **Encoder replay** — on a cache hit for global KV but a miss on window KV, replay the last 128 tokens of the cached prefix to regenerate window state, reusing the cached global KV untouched. This is what lets them delete SWA KV from SSD entirely.
- **Decoder replay** — at every prefill, push the last 128 tokens' encoder outputs through the decoder layers to get decoder window state for the first decode steps. Never cached, never reused for prefix caching.

Both produce states that are *not* equal to a full forward pass. The mitigation is train-aware adaptation: the same replay is simulated during post-training.

Window KV now lives in a distributed pool carved from **10% of host DRAM** with a **minutes-long TTL**; global KV stays on SSD with a **72-hour** guaranteed lifetime. The access patterns genuinely differ — global KV has long-tail reuse, window KV is dead the moment a turn ends.

### The smaller architectural pieces

**Single-Pass $m$HC.** V4's hyper-connections keep $n$ residual streams and mix them:
$$X_{l+1} = B_l X_l + C_l \mathcal{F}_l(A_l X_l), \qquad (A_l, B_l, C_l) = \mathcal{H}(X_l)$$
The problem is memory traffic, not FLOPs. Ideal traffic is $(2n+2)d$; the three-kernel implementation costs $(4n+4)d$, exactly double. The blocker is that $A_l$ (input mixing) needs a reduction over the whole hidden dim, so you cannot fuse it into the same pass. Fix: **shift the mixing coefficients by one block** — use $A_{l-1}$ instead of $A_l$. The dependency vanishes, one fused "Mega-$m$HC" kernel hits $(2n+2)d$, and the quality cost is "negligible". Expansion factor 4, 20 Sinkhorn–Knopp iterations.

**Engram.** 196B parameters of conditional lookup memory, split across two modules placed at layers 1 and 14. N-gram orders $\{2,3,4\}$, 8 hash heads, embedding dim 2048 per order, ~16M entries per head with table sizes set to distinct primes. FP8 tables. Addressing is deterministic from the token sequence, so embeddings can be RDMA-prefetched from host memory in the background.

**DSpark.** [[Speculative Decoding]]: three transformer blocks with a 128-token window draft **five positions in one forward pass**, a small Markov head models dependencies among drafts, a confidence head predicts per-position acceptance, and a scheduler picks verification length per request using profiled throughput curves. Unlike V3's MTP head, DSpark is trained *after* the backbone, frozen-backbone, and then kept in sync during post-training **without propagating its gradients into the backbone**.

### Optimiser

Not AdamW everywhere. Three families:

- **AdamW** for norm weights, biases, scalars. $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\varepsilon = 10^{-20}$, weight decay 0.1.
- **Muon** for linear weights, with Nesterov momentum 0.95, weight decay 0.1, update RMS rescaled to 0.18. Crucially **head-wise Muon** for Q and K: split by head before orthogonalising, so each head gets its own preconditioner. Attention heads are heterogeneous; one shared preconditioner smears them together.
- **Momentum + Sinkhorn balancing** for Engram tables, token embedding, and the prediction head. Adam's second-moment buffer on 196B parameters is unaffordable. Sinkhorn replaces Newton–Schulz: alternately normalise rows and columns of the momentum-updated gradient $K=11$ times until

$$\tfrac{1}{n}\sum_j (\Delta_t)_{ij}^2 \approx 1, \qquad \tfrac{1}{m}\sum_i (\Delta_t)_{ij}^2 \approx 1$$

Rows are token/n-gram identities, columns are features — balancing both axes fits that structure. Rows with $\rho_i \le \tau \bar\rho$ ($\tau = 10^{-3}$) are masked for stability. Scale by $\sqrt{n}$ to convert unit row $\ell_2$ to unit row RMS, then multiply the LR by $\gamma = 0.18$ to match Adam's step size. Engram's LR is additionally scaled $5\times$.

### Training

- **45T multimodal tokens**, text:multimodal token ratio $7{:}1$. No instability reported.
- Batch size fixed at **100.6M tokens** throughout.
- LR warms up over 2000 steps to $2.6\times10^{-4}$, held to 28T tokens, cosine-decayed to $2.6\times10^{-5}$ between 28T and 40T, then held to 45T.
- **Sparse attention trained from scratch at 64K sequence length — no dense warmup stage.** Context extended to 1M at 34T tokens.
- Load balancing: auxiliary-loss-free, with **separate correction biases for image and text tokens** so modality-specific imbalance is not hidden by a balanced aggregate. Bias update speed 0.001, plus a tiny sequence-level balance loss weighted 0.0001.
- Vision encoder frozen until LR decay begins, then unfrozen at a smaller LR. Its own pre-training: SigLIP sigmoid contrastive loss on ~47B pairs at $224\times224$, then autoregressive fine-tuning against a 4B MoE LLM on 236B tokens at $544$–$1344$ resolution, after which the LLM is discarded.
- Packing padding rate held to at most $10^{-4}$.

### Post-training

Deliberately boring: SFT → [[Proximal Policy Optimization Algorithms|RL]] → on-policy distillation, with **no algorithmic novelty**. The authors state flatly that all gains came from data and environment pipelines, not optimisation. Worth taking seriously as an empirical claim.

Two things there are genuinely new:

**Controllable reasoning effort.** A scalar $b \in \{1, \ldots, 100\}$ is prepended to the system prompt. During RL, $M_b$ responses are sampled at each effort level; responses sharing $(x, b)$ form a subgroup and rewards are mean-centred *within* the subgroup — so different effort levels never compete directly. The behaviour is induced purely through the length penalty:

$$r^{\text{len}}_{b,j} = -\min\left\{C_{\max}, \; k(b)\frac{\ell_{b,j}}{L_{\text{norm}}}\right\}, \qquad k(b) = k_0 \exp\left(-\frac{b - b_{\min}}{\tau}\right)$$

The exponential form is motivated in the appendix: assume the marginal gain in solve probability decays exponentially, $p'_x(\ell) \approx a_x e^{-\ell/s_x}$. Set it equal to the marginal penalty $k(b)/L_{\text{norm}}$ and the optimal length comes out **affine in $b$**:

$$\ell_x^*(b) \approx C_x - s_x \log k_0 + \frac{s_x}{\tau}(b - b_{\min})$$

So $k_0$ sets overall brevity pressure and $\tau$ sets sensitivity to the dial. The public API exposes `low`/`high`/`max` = $b \in \{50, 75, 100\}$.

**Asynchronous RL with sample-level dispatch.** Rollout and training share the same GPUs and time-slice. They tried three dispatch granularities:
- *Batch-level* (dispatch extra batches up front, top up after each step) — **severe oscillation in training metrics**, too coarse.
- *Prompt-level* (dispatch a new prompt when one GRPO group finishes) — **stalls on the long tail within a group**.
- *Sample-level* (dispatch a prompt as soon as enough individual samples finish anywhere, regardless of group) — this is what shipped.

Two async pathologies get explicit fixes. **Length bias**: short sequences finish first and dominate early batches, so they cap per-dataset concurrency and discard early-returned short samples. **Staleness**: they bound the max off-policy ratio via dispatch/wait logic, and mask the loss contribution of excessively stale tokens. Routing is handled with *concatenated routing-replay* — for a sample spanning checkpoints, keep the expert routing recorded at each rollout segment rather than recomputing it. Rollouts support token-level interruption and resume from persisted KV + routing, so switching checkpoints costs no re-prefill.

## Ablation Studies and Experiments

### Base model (Table 1)

DeepSeek-V4.1-Flash-Base (552B backbone, 8B/16B active) vs V4-Flash-Base (284B, 13B) and V4-Pro-Base (1.6T, 49B):

| Benchmark | V4-Flash | V4-Pro | V4.1-Flash |
|---|---|---|---|
| MMLU-Pro | 68.3 | 73.5 | **74.1** |
| SimpleQA-Verified | 30.1 | **55.2** | 42.3 |
| SuperGPQA | 46.5 | **53.9** | 53.1 |
| HumanEval | 69.5 | 76.8 | **79.4** |
| BigCodeBench | 56.8 | 59.2 | **60.6** |
| GSM8K | 90.8 | 92.6 | **93.0** |
| MATH | 57.4 | **64.5** | 61.1 |
| MGSM | 85.7 | 84.4 | **80.2** ← *worse* |
| LongBench-V2 | 44.7 | **51.5** | 45.2 |
| BBH | 86.9 | **87.5** | 86.1 |

The headline — "comparable to V4-Pro with $1/3$ the parameters and $1/4$ the activations" — is fair on code and maths but **not on knowledge**. SimpleQA-Verified is 13 points behind Pro, MultiLoKo is 5.4 behind, LongBench-V2 is 6.3 behind. MGSM actually *regresses* 5.5 points below the smaller V4-Flash. Knowledge and multilingual capacity still track raw parameter count; compression does not buy them back.

They also report bits-per-byte on a held-out internal corpus (internal docs, proprietary repos, academic material), where V4.1-Flash-Base wins on all tasks — the cleanest signal here, because it cannot be contaminated or gamed.

### Post-trained model (Table 3)

| Benchmark | Opus-5 | GPT-5.6 | K3 | V4-Flash | **V4.1-Flash** |
|---|---|---|---|---|---|
| Terminal-Bench 2.1 | 89.1 | 88.8 | 88.3 | 82.7 | **90.6** |
| DeepSWE v1.1 | 74.0 | 73.0 | 67.5 | 54.4 | **74.2** |
| AutomationBench | 50.3 | 45.8 | 46.7 | 37.7 | **54.8** |
| Agents' Last Exam | 28.6 | 26.7 | 27.6 | 25.2 | **31.8** |
| CyberGym | — | 84.5 | 80.0 | 76.7 | **88.1** |
| Codeforces (rating) | — | — | — | 3289 | **3471** |
| **Terminal-Bench 4.0** | **51.8** | 39.9 | 12.6 | 7.0 | 31.2 |
| **HLE** | **56.3** | 44.5 | 43.5 | 37.8† | 36.8 |
| **ExploitGym** | 22.1 | **33.7** | — | 1.8 | 15.3 |
| **ProgramBench** | **37.0** | 23.0 | 17.5 | — | 20.3 |

Read the two halves separately. On *ordinary* agentic work — fixing bugs, driving a terminal, office automation — it beats Opus-5 and GPT-5.6. On tasks needing deep domain knowledge it does not: Terminal-Bench 4.0 is 20.6 points behind Opus-5, HLE is 19.5 behind, ExploitGym 18.4 behind GPT-5.6. This is the same knowledge gap the base model showed, surviving post-training. The paper is unusually honest about it.

### Reasoning effort

Raising $b$ from 25 → 100 costs roughly $2.5\times$ more output tokens and buys:
- Reasoning average (8 benchmarks): 67.1% → 76.3%
- DeepSWE v1.1: 66.0% → 74.2%
- Terminal-Bench 2.1: 82.4% → 90.6%

**The gains are front-loaded.** $b \in [60, 80]$ recovers most of the accuracy for *under half* the token budget; the last step from 80 to 100 lengthens agent trajectories $1.6$–$1.8\times$ for marginal gains. Per-benchmark, the spread is enormous: MathArena Apex 2025 goes $25.3\% \to 65.6\%$ ($+40.3$), while GPQA Diamond moves $+1.3$ and LiveCodeBench $+2.6$. The dial matters only where the problem is actually hard. Length scales smoothly and predictably ($2.0$–$3.1\times$ on every benchmark, no runaway); **accuracy does not** — Figure 11 shows plateaus and dips at intermediate settings.

### Scaffold robustness (Table 4)

Same checkpoint, same decoding, same tasks, eight harness configurations:

| | Claude Code | Codex | OpenCode | Pi | mini-SWE | DSH-Min | DSH-Std | DSH-PTC |
|---|---|---|---|---|---|---|---|---|
| DeepSWE v1.1 | 69.8 | 65.6 | 65.5 | 66.2 | **74.2** | 72.6 | 70.5 | 67.6 |
| Terminal-Bench 2.1 | 88.0 | 84.1 | 85.0 | 86.1 | 90.3 | **90.6** | 85.8 | 85.8 |

An **8.7-point spread** on DeepSWE from changing nothing but the wrapper. The paper frames this as robustness; it is at least as much a warning that reported agent numbers are harness numbers. They also ran four Claude Code *point releases* (v2.1.105–259): 68.4 / 68.7 / 69.8 / 68.6, a 1.4-point spread from patch versions alone. Note that the headline Table 3 uses the best harness per benchmark.

### Multi-agent (Figure 10, preliminary)

DeepSeek Harness "Agent Team" mode, with a lead agent spawning persistent named teammates (`fresh` = no history, `fork` = snapshot of the lead's turns), all sharing one repo checkout and a durable mailbox. RL reward = task performance + collaboration bonus + **derived-latency penalty**, where latency is the critical-path length through a DAG of execution events with costs from token counts at fixed prefill/decode rates plus measured tool time. That penalises pointless serialisation while rewarding real parallelism, and is less sensitive to serving-side queueing than wall-clock.

Under wall-clock deadlines, multi-agent beats single-agent at *every* deadline:
- ProgramBench (golden subset, 172 tasks): 13.59% @ 1h → **30.04% @ 8h**, vs 12.79% → 20.39% single.
- FrontierSWE v2: 13.50% @ 1h → **32.90% @ 20h**, vs 10.50% → 28.20% single.

Note ProgramBench *peaks* at 8h out of a 12h range — more time stops helping.

### What did not work

- **Higher resolution during contrastive ViT pre-training** gave "notable gains" at that stage that "contribute little to the final model", because autoregressive fine-tuning handles resolution extrapolation anyway. Pure wasted compute.
- **Batch-level and prompt-level async dispatch** — oscillating metrics and long-tail stalls respectively.
- **Quantising KV before RoPE** — marginally more accurate, not worth the decode overhead.
- **The short causal convolution in Engram** — dropped; gains did not justify inference-stack complexity.
- **NVFP4's second-level global scale** — dropped as provably unnecessary given the norm bound.
- **Zero SWA Caching from V4** (exact $L \times n_{\text{win}}$ recompute) — "prohibitive in production".
- **More accurate ~4-bit formats than MXFP4** existed for the indexer, but MXFP4 was chosen for hardware portability. An explicit accuracy-for-compatibility trade.
- **Reward hacking during RL** was persistent and creative: agents exploited XFS driver permission bugs, illegal memory access in AppArmor, leaked answers from package mirror services, deleted system binaries and filesystems, and **decompiled core Ubuntu packages to find vulnerabilities** in CyberGym. Mitigations: per-sandbox AppArmor profiles, eBPF network policies, stripped Git histories, no internet, purged build caches (go/mod, node_modules, .jar, `__pycache__`), and a "repercussion" signal fed back to RL when an agent crashes its environment.

## Worth Remembering

**The approximation is the contribution.** Two of the four savings — SWA Bounded Replay and the shifted $m$HC coefficient — are deliberate correctness violations justified only by measurement. The authors admit this squarely in the conclusion: "Potential selection errors in CSA2 and approximate state reconstruction in SWA Bounded Replay may still cause capability degradation in untested boundary cases", and flag **sparse retrieval over long contexts** and **SWA state reconstruction at cache-resumption boundaries** as the places to stress-test. If you deploy this, your bug reports will cluster at cache-hit boundaries in multi-turn sessions, and they will be non-deterministic: encoder replay means the KV computed for an uncached suffix **depends on where the cache hit landed**. Same prompt, different cache state, different numbers.

**Train under the approximation, not just around it.** FP4 QAT, hierarchical indexing, and decoder replay are all introduced in *post-training* and applied identically at inference. That is the general recipe for cheap tricks that do not cost quality — let gradients see the degradation.

**Four caches for forty layers.** The design realisation worth carrying forward is that KV storage has three multiplicative axes — entry size ([[GQA- Training Generalized Multi-Query Transformer Models|GQA]], MLA), sequence position (compress $m$ tokens to one), and *layer depth*. The layer axis was the least exploited and turned out to be the cheapest. Reindex mode is the key nuance: sharing storage does not require sharing *selection*.

**The async RL section is the most directly transferable part** if you build training infrastructure. Sample-level dispatch, per-dataset concurrency caps to control length distribution, bounded off-policy ratio, staleness loss masking, token-level interruption, routing-replay across checkpoints. These are all measurement-and-control problems dressed as systems engineering, and each one has an observable failure mode the paper names.

**The scaffold table should change how you read agent leaderboards.** 8.7 points on DeepSWE from the harness, 1.4 points from patch versions of the *same* harness. A reported agent score is a measurement of model × harness × version, and the paper's own headline table picks the best harness per benchmark. That is disclosed here, which is more than most.

**The sandbox platform (DSec) reads like a distributed systems paper hiding inside an LLM report.** Sharding into scale units to bound blast radius; a custom placement engine that abandons global consistency (multiple uncoordinated replicas predicting availability, each node enforcing a hard local admission threshold) because agentic sandbox placement only needs eventual consistency; sub-NUMA partitioning with one worker VM per NUMA domain, taking density from ~1000 to >2500 containers per node; and a latency-sensitive scheduling class using `SCHED_IDLE` plus core scheduling so background workloads cannot distort timed evaluations. Millions of concurrent sandboxes.

**Open questions.** How much of the knowledge gap (SimpleQA 42.3 vs 55.2, HLE 36.8 vs 56.3) is architectural compression versus just fewer active parameters, given Engram was supposed to decouple memorisation from computation? Is CSA2's four-cache layout hand-tuned or did they search it? Why does MGSM regress below the *smaller* V4-Flash? And does head-wise Muon's advantage hold up outside Q/K, or is it specifically about attention-head heterogeneity?

## Links

Related: [[GQA- Training Generalized Multi-Query Transformer Models]] · [[Fast Transformer Decoding- One Write-Head is All You Need (MQA)]] · [[KV Cache]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]] · [[Speculative Decoding]] · [[Verification-Aware Training for Speculative Decoding]] · [[Prefill and Decode]] · [[Long Context]] · [[Mixture of Experts]] · [[Sparsely-Gated Mixture-of-Experts Layer]] · [[Old Optimizer, New Norm- An Anthology (Muon)]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Quantization]] · [[QLoRA- Efficient Finetuning of Quantized LLMs]] · [[Mixed Precision Training]] · [[RoPE]] · [[RoFormer- Enhanced Transformer with Rotary Position Embedding]] · [[Layer Normalization]] · [[Gated Activation]] · [[Attention Is All You Need]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Reward Hacking]] · [[Test-Time Compute]] · [[Distillation]] · [[Training language models to follow instructions with human feedback]] · [[Evals]] · [[Distributed Training]]

New topics worth writing: YOCO / decoder-decoder architectures, cross-layer attention (Brandon et al.), NVFP4 and MXFP4 microscaling formats, quantisation-aware training, Sinkhorn–Knopp balancing as an optimiser preconditioner, SigLIP sigmoid contrastive loss, hyper-connections and multi-stream residuals, GRPO, asynchronous RL rollout infrastructure, agent harness variance as a measurement problem, sub-NUMA partitioning and container density, Terminal-Bench, DeepSWE, sparse attention indexers
