---
title: "WhiteMatter: All-to-All Cross-Layer Connections via KV Source Mixing"
authors: ["Wenbo Zhang", "Xiang Ren"]
year: 2026
arxiv: "2608.18486"
url: https://arxiv.org/abs/2608.18486
priority: Good-To-Read
read_on: 2026-09-30
tags: [paper, transformers]
---
## The Core Idea

In a normal Transformer, layer 7 can only attend to keys and values that were also made at layer 7. Every layer sees the past through its own depth, and no other. That is a rule nobody chose on purpose — it exists because it makes training easy to parallelise. Each token's whole layer stack can be computed at once, because no layer ever needs a *deeper* layer's output for an *earlier* token.

WhiteMatter breaks that rule. Every layer can read past-token representations from **any** depth, and which depths it reads is decided per token by a small learned router.

The concrete mechanism: instead of each layer having its own K and V projection fed by its own hidden state, there is a shared **pool** of $k$ key-value channels. Each channel is built from a weighted sum of *all* $L$ hidden states of that past token, with weights that depend on the token's content. Layer $\ell$ then reads channel $j = \ell \bmod k$. If $k = L$, every layer gets its own private mixture. If $k = L/2$, pairs of layers share a channel — and the [[KV Cache]] is half the usual size.

Two things unlock at once:

1. **Quality.** With a full-size cache, the 16-layer model reaches perplexity comparable to a 24-layer vanilla Transformer — 50% more layers — on the same 8B training tokens.
2. **Memory.** With half the cache, it still *beats* the matched vanilla model: 5.9% lower held-out perplexity at small scale, 4.3% at 1.3B parameters, with 39.4% lower peak memory during decoding.

### Why it did not exist before

Two earlier attempts did feedback from deep layers to shallow ones and both hit a wall.

**Feedback Transformer** (Fan et al., 2021) squashes all of a token's layer states into *one* weighted sum, projects that into one KV, and every layer reads it. **LCKV** (Wu & Tu, 2024) builds one shared KV from the deepest layer's state. LCKV only works if you leave some "warmup" layers with their normal same-depth KV; applying feedback KV to *every* layer makes perplexity much worse.

WhiteMatter's diagnosis of why: both methods force a **single** layer-width summary to serve every reader. That summary has to carry every feature any layer might want. The paper's bet is that different layers want different depths — so give each reader its own mixture. The ablation confirms it hard: one shared mixture with 16 independent KV projections loses to WhiteMatter with $k=4$, which has a **quarter** the cache. More cache does not buy you what specialisation buys you.

### The second half of the paper: how do you train it?

Feedback creates a cycle. To build token 5's KV channels you need token 5's deep layers; but token 6's shallow layers want to read token 5's deep channels. During decoding this is fine — the past is already computed. During training and prefill it destroys the parallelism that makes Transformers trainable.

The fix is to notice that the exact cache is a **fixed point**:

$$\mathrm{KV} = \operatorname{Model}(X; \mathrm{KV})$$

Feed the cache in, get the same cache out. So you can *iterate* towards it. LCKV used Jacobi iteration — every pass recomputes all tokens in parallel, reading only the *previous* pass's cache. Correct, but slow: information moves one token-hop per full-sequence pass.

WhiteMatter uses **cyclic Gauss–Seidel iteration**. Split token positions into $g$ interleaved groups by $i \bmod g$, run the groups in order, and refresh the cache after each group. Token 5 in group 1 can read token 4's *current-pass* KV, because token 4 was in group 0. Updates propagate much further per pass. On a 4-layer reference model trained with exact autoregressive execution, cyclic with $g{=}16$ hits the target perplexity in 4 passes (7.32 ms/sequence) where Jacobi needs 53 passes (91.20 ms/sequence) — **12.5× faster**.

> [!NOTE] Cyclic Gauss–Seidel iteration
> Solving the feedback cache by splitting tokens into strided groups ($i \bmod g$), processing groups in sequence, and refreshing the cache after each group — so later groups read earlier groups' fresh values within the same pass. Contrast with Jacobi, where all tokens read only the previous pass. ^cyclic-gauss-seidel

## The Methodology

### The KV pool, for one token position $i$

Let $h_\ell[i] \in \mathbb{R}^D$ be the hidden state entering layer $\ell$ at token $i$. Three steps.

**Step 1 — mix $L$ states into $k$ channels.** The key and value branches are fully separate (own norms, own routers, own weights). Taking the key branch:

Each source state is RMS-normalised first, giving $\hat{h}^K_\ell[i]$. This "pre-mix norm" puts all $L$ layers on one scale and — importantly — stops magnitudes exploding as states recirculate through the feedback loop.

The router reads every $p$-th source layer (stride $p{=}2$ in the experiments) to keep its parameter count down, concatenated into $\xi^K[i]$. It emits weights over all $L$ layers:

$$\alpha^K_j[i] = W^{\alpha K}_j \xi^K[i] + b^{\alpha K}_j, \qquad \tilde{h}^K_j[i] = \sum_{\ell=0}^{L-1} \alpha^K_j[i][\ell]\, \hat{h}^K_\ell[i]$$

The weights are **signed**, not a softmax. So a channel can express *differences* between layers, not just averages.

**Step 2 — project.** A second RMSNorm, then the projection:

$$K_j[i] = W^K_j \,\mathrm{RMSNorm}^K_j(\tilde{h}^K_j[i]), \qquad V_j[i] = W^V_j\, \mathrm{RMSNorm}^V_j(\tilde{h}^V_j[i])$$

Per-channel key normalisation and [[RoPE]] are applied to the keys before caching. Values cache directly.

**Step 3 — select.** Layer $\ell$ reads channel $j = \ell \bmod k$. Fixed, not learned. The reason is bandwidth: if a layer could read *all* $k$ channels it would have to stream all of them from HBM, and the whole memory saving evaporates. One channel read per layer keeps the traffic identical to vanilla.

Router weights initialise at **zero**; biases initialise to select one sensible source (entry $0.25$ for the chosen layer, zero elsewhere). So the model starts out roughly behaving like a Transformer with a mild depth offset and learns the mixing from there.

### Strict causality

Token $i$ attends only to positions $s < i$ — not to itself. That is required: reading its own channels would need its own deep states, which need its own shallow states, which is the cycle. After token $i$'s layer stack finishes, its channels are built and appended for later tokens. A learned boundary token supplies cache slot zero (rotary position zero, always visible past the document mask).

This strictness is exactly what makes the fixed-point equation well-posed.

### Training loop

Per Algorithm 1: initialise the cache from token embeddings (treating the embedding as the token's state at every depth), then for $n$ passes, for $g$ groups, evaluate group $\mathcal{G}_q = \{i : i \bmod g = q\}$ in parallel with the current cache, collect layer inputs $H$, and overwrite $\mathrm{KV}[\mathcal{G}_q] \leftarrow \operatorname{Pool}(H)$.

Attention needs a strided causal mask, so they adapted FlashAttention-style tiling into a custom cyclic kernel.

**Truncated backprop.** Gradients flow only through the last $n_g \le n$ passes. Earlier passes run under no-grad purely to get near the fixed point. Main runs used **one no-grad cyclic pass then two gradient-carrying passes**, $g{=}8$. (LCKV's recipe by comparison: seven no-grad Jacobi passes then two with gradients.)

### Training setup

| | $D{=}512$ | $D{=}1792$ |
|---|---|---|
| Layers | 16 (also 24L vanilla) | 28 |
| Params | — | 1.326B WM / 1.351B vanilla |
| Intermediate width | 1536 | 5376 |
| Q / KV heads | 6 / 3 | 14 / 7 |
| Head dim | 96 | 128 |
| Tokens | 8B | 10B |
| Effective batch | 128 | 32 |
| Steps | 30,518 | 152,588 |

Qwen3-style decoders, from scratch on shuffled FineWeb-Edu, Qwen3 tokeniser (vocab 151,936), length-2048 packed sequences with within-document attention. [[Old Optimizer, New Norm- An Anthology (Muon)|Muon]] (momentum 0.95, five Newton–Schulz steps) for 2-D weight matrices, [[Decoupled Weight Decay Regularization (AdamW)|AdamW]] ($\beta_1{=}0.9$, $\beta_2{=}0.95$) for everything else. Both at peak LR $3\times10^{-4}$, 2% warmup, cosine to 10% of peak, weight decay 0.1. bfloat16 autocast with fp32 master weights, eight RTX A6000s, per-GPU gradient clip at 1.0 before DDP all-reduce.

## Ablation Studies and Experiments

### Headline quality, $D{=}512$, 8B tokens

| Model | WikiText PPL ↓ | LAMBADA PPL ↓ | Avg. acc ↑ | KV cache |
|---|---|---|---|---|
| Vanilla 16L | 49.34 | 127.47 | 47.21 | 1× |
| FusedKV | 48.35 | 92.33 | 48.34 | 1/2 |
| LCKV $w{=}4$ | 48.81 | 107.52 | 48.65 | 5/16 |
| LCKV $w{=}7$ | 49.02 | 102.97 | 48.69 | 1/2 |
| **WhiteMatter $k{=}8$** | **44.40** | **71.58** | **49.24** | 1/2 |
| **WhiteMatter $k{=}16$** | **43.28** | **60.73** | **49.89** | 1× |
| Vanilla 24L | 44.71 | 97.40 | 48.52 | 1.5× |

Avg. is the unweighted mean of 11 zero-shot scores (BLiMP, PIQA, HellaSwag, ARC-E/C, WinoGrande, OBQA, SciQ, ReCoRD, SQuAD-completion, LAMBADA acc).

Two readings worth holding onto. Full-cache WhiteMatter beats the **50% deeper** vanilla on both perplexities and on average accuracy, with fewer non-embedding parameters. And at equal half-cache, it beats both LCKV and FusedKV clearly — this is not just "KV sharing works".

The LAMBADA gap is the loudest single number: 127.47 → 60.73. That benchmark needs a long-range referent, which is exactly where reading other depths of the past should pay.

### At 1.3B, 10B tokens

| Model | WikiText PPL | LAMBADA PPL | Avg. acc | Held-out PPL |
|---|---|---|---|---|
| Vanilla 28L | 26.61 | 23.84 | 57.48 | 13.51 |
| WhiteMatter $k{=}14$ | 24.67 | 18.42 | 59.24 | 12.93 |

Half the cache, 1.76 points better average accuracy. The gain holds at scale, though 1.3B / 10B tokens is where the evidence stops.

Also reassuring: for every WhiteMatter and LCKV model, perplexity under **exact autoregressive** decoding differs by under 1% from perplexity under fixed-point iteration. The iteration is genuinely converging to the thing the model was meant to compute, not to some separate parallel-only regime.

### Runtime, 1.3B, batch 64, RTX A6000, compiled BF16

| | Throughput | Peak memory |
|---|---|---|
| Decode, vanilla | ~2,500 tok/s | 16.59 GiB |
| Decode, WhiteMatter | ~2,500 tok/s | 10.05 GiB (−39.4%) |
| Decode, LCKV | ~2,500 tok/s | 10.01 GiB |
| Decode, Feedback Transformer | ~2,500 tok/s | 3.86 GiB |
| Prefill, WhiteMatter | 31% of vanilla | 13.44 vs 21.21 GiB |

So: decoding is free, memory is much better, **prefill is 3× slower than vanilla**. It is 1.78× LCKV's prefill and 2.92× Feedback Transformer's, so it is the best of the feedback family, but it is not free. FLOPs say the same thing: at $D{=}512$ WhiteMatter is 0.99–1.03× vanilla's decode FLOPs but 2.32–2.50× training and 3.05–3.30× three-pass prefill (excluding the LM head).

### The ablations, and this is where the paper earns its claim

**Specialised mixtures are the mechanism, not cache size.** Replace the $k$ distinct mixtures with one shared mixture, keeping 16 independent KV projections. That model loses to WhiteMatter at $k{=}4$, which has **4× less cache**. This is the direct refutation of the "just make the summary wider" explanation — and it is what separates WhiteMatter from Feedback Transformer and LCKV.

**Channel count.** Sweeping $k \in \{1,2,4,8,12,16\}$: more channels generally better, with the biggest single jump from $k{=}1$ to $k{=}2$. Going from one shared mixture to just *two* is most of the win. Diminishing returns after that.

**Dynamic routing vs static.** Replace the content-dependent router with static learnable weights: +3.0% perplexity at $k{=}1$, +1.9% at $k{=}16$. Real but modest — the content-dependence matters most when you have few channels and each one has to do more work.

**Deep-to-shallow feedback is doing real work.** Restrict layer $\ell$ to mixtures of states $0,\dots,\ell$ only. That removes feedback entirely, so no iteration is needed and training is cheap — you get a pure feedforward cross-layer aggregation, like DenseFormer or MUDDFormer. It beats vanilla, but its perplexity is **4.1% worse** than full-cache WhiteMatter. So roughly: cross-layer mixing gets you part of the way, and the expensive backwards part gets you the rest.

**Training schedule matters enormously.** Sweeping $n_g \in \{1,2\}$ gradient passes, $n_\text{no-grad} \in \{1,2,4\}$, and schedule $\{\text{Jacobi}, C_4, C_8, C_{16}\}$ over two seeds on a 328M-token budget: the strongest schedule is **32% better perplexity** than the weakest. And there is a subtle stability result — models trained *far* from the fixed point **degrade** if you run more iterations at inference. Models trained close to it stay stable under further refinement, but may need more inference passes to reach their own best perplexity. Train near convergence or your inference-time knob becomes a liability.

### What did not work

- **Contiguous chunks instead of strided groups.** Splitting each pass into left-to-right contiguous blocks needs more passes and more time than cyclic groups at *every* group count tested. The reason is intuitive: [[Attention]] concentrates on nearby tokens, and cyclic assignment guarantees your immediate predecessor is in an earlier group, so its fresh value is available. Contiguous chunking gives fresh values to only the first token of each chunk.
- **More groups is not monotonically better.** Cyclic $g{=}16$ was the runtime winner, not $g{=}64$. More groups → fewer passes needed, but less parallel work per group. The largest $g$ does not give the lowest wall time.
- **Bigger cache alone** does not recover the benefit of specialised mixtures (above).

## Worth Remembering

**The generalisable idea is not the architecture, it is the fixed point.** Any architecture with a backwards dependency across positions can be written $\mathrm{KV} = \operatorname{Model}(X; \mathrm{KV})$ and solved iteratively. Once you see it as a linear-solve-shaped problem, the whole Jacobi/Gauss–Seidel literature is available to you. That is the transferable move here, and cyclic ordering is just a good preconditioner chosen because attention is local.

> [!NOTE] Cross-layer KV pool
> A shared set of $k$ key-value channels, each built as a signed, content-dependent weighted sum of *all* $L$ hidden states of a past token, with layer $\ell$ reading channel $\ell \bmod k$. Cache size becomes $k/L$ of vanilla's. ^cross-layer-kv-pool

**Where this bites in practice.** This is a decode-optimised architecture that pays at prefill. So it is right for long generations against short prompts — reasoning traces, agent loops — and wrong for RAG-style workloads where you stuff 30k tokens of retrieved context in and generate 200. The paper says as much in its motivation, citing agentic workloads and long reasoning traces as the regimes where decoding dominates.

**The memory win is real but read it carefully.** 39.4% lower *peak device memory* at batch 64, not 50% lower cache. The cache is half, but weights and activations are not. Feedback Transformer uses far less (3.86 GiB) because it shares one KV across all layers — it just also has bad quality and the worst prefill.

**Limitations the authors state plainly.** Iterative training and prefill remain more expensive than a standard Transformer, full stop. Quality evidence stops at 1.3B / 10B tokens — nowhere near a regime where you would trust it against a production model. The full-cache variant and the alternative KV-sharing baselines (LCKV, FusedKV) were only evaluated at $D{=}512$, so the 1.3B table is WhiteMatter vs vanilla only.

**A caveat the paper does not dwell on.** Inference now has a hyperparameter — how many iteration passes — and the schedule ablation shows quality is *non-monotonic* in it for under-converged models. That is a new operational footgun: the same weights give different quality depending on a serving config, and getting it wrong can make things actively worse rather than just slower.

**Connections.** The signed weights and the channel-mixing are close cousins of value-residual learning and MUDDFormer, which form separate mixtures of earlier representations for attention inputs — WhiteMatter's own feedforward-only ablation is essentially that family, and it loses by 4.1%. The concurrent Latent Recurrent Transformer proposes an interleaved schedule with a single refinement sweep but does not study multi-pass convergence, which is where most of this paper's schedule analysis lives. The fixed-point-by-iteration framing also rhymes with Jacobi decoding for [[Speculative Decoding|speculative decoding]], though here it is prefill rather than generation.

**Follow-up questions.** Does the router actually learn interpretable depth preferences, and do shallow layers reach further back than deep ones? What do the learned $\alpha$ patterns look like after training — sparse, or genuinely distributed? Does the 4.1% feedback premium hold at scale, or does it shrink once depth gives you more feedforward paths anyway? And can prefill be made cheaper by accepting a looser convergence threshold at a measured quality cost — the 1% threshold used throughout is a choice, not a law.

## Links

Related: [[KV Cache]] · [[Attention]] · [[Grouped Query Attention]] · [[GQA- Training Generalized Multi-Query Transformer Models]] · [[Fast Transformer Decoding- One Write-Head is All You Need (MQA)]] · [[Multi-head Latent Attention]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Flash Attention]] · [[RoPE]] · [[Attention Is All You Need]] · [[Prefill and Decode]] · [[Perplexity]] · [[Old Optimizer, New Norm- An Anthology (Muon)]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Layer Normalization]] · [[Speculative Decoding]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Linear Attention]] · [[Sparse Attention]] · [[Long Context]] · [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]]

New topics worth writing: Gauss–Seidel and Jacobi iteration for linear systems, Feedback Transformer, LCKV (Layer-Condensed KV Cache), cross-layer KV sharing, MUDDFormer and dynamic dense residual connections, value-residual learning, DenseFormer, hyper-connections, Jacobi decoding, fixed-point iteration in neural architectures, truncated backpropagation through iterative solvers, FineWeb-Edu, LAMBADA, BLiMP, ReCoRD
