---
title: "SAS: Simple Attention Sparsification via End-to-End Optimization of Context Ranking"
authors: ["Zhiwei Li", "Lei Zhu", "Hao Gu", "Xiang Hu", "Yan Wang", "Haitao Mi", "Sirui Han", "Leo Liang", "Zhijiang Guo"]
year: 2026
arxiv: "2609.13141"
url: https://arxiv.org/abs/2609.13141
priority: Good-To-Read
read_on: 2026-09-17
tags: [paper, transformers, llm, rl]
---
## The Core Idea

Long chats are slow because every new token has to look at every old token. If the context is $n$ tokens, generating the whole thing costs $O(n^2)$ attention work. The obvious fix: let each query look at only a small slice of the past — say 32 blocks of 64 tokens instead of all 8000 tokens. This is **sparse attention**. The hard part is not *skipping*; it is *choosing what to skip*.

The standard recipe bolts a small "selector" onto a pretrained model. The selector scores each past block, and a hard Top-$K$ picks the winners. The problem is that Top-$K$ is a step function. Its derivative is zero almost everywhere, so the [[Cross Entropy|language modelling loss]] cannot tell the selector "you picked wrong" through [[Backpropagation|backprop]]. Everyone works around this by training the selector to *imitate the dense model's attention weights*, layer by layer — a [[Distillation|distillation]] job.

The insight of SAS is that imitating dense attention is the wrong target. Two reasons:

1. **It is per-layer.** Layer 7 is taught to match layer 7. Nothing rewards the layers for *dividing the work* between them. If layer 7 and layer 8 both burn their budget on the same blocks, distillation is happy.
2. **It ignores $\mathbf{V}$.** Attention weights are only half of attention. A block can have a modest weight but a huge value vector, so it moves the output a lot. Matching weights never sees this.

So distillation produces a **ranking misalignment**: the selector ranks blocks by "where the dense model looked", not by "which blocks actually change the next-token prediction when I only get 32 of them".

> [!NOTE] Ranking misalignment
> Training a selector on a surrogate target (dense attention mass) makes it rank context by the wrong criterion. Under a tight budget, the limited slots go to blocks that were popular in the dense model rather than blocks that carry the prediction. ^ranking-misalignment

The fix is almost embarrassingly small. During training, take the selector's continuous score for each block, turn it into a positive gate $g_m$, and **add $\log g_m$ to the attention logits before the softmax**. Now the gate is a normal differentiable tensor sitting in the middle of attention. The language modelling loss flows back through it into the selector by plain backprop. No teacher, no auxiliary loss, no straight-through estimator. At inference you throw the gate away and just do hard Top-$K$ on the scores.

What it unlocks: the selector is now trained on the thing you actually care about. Gains are biggest where budget is tightest — at a 1024-token budget on Qwen3-14B, GPQA-Diamond goes from 45.64 (distilled) to **61.14** (end-to-end), against 65.25 for full attention.

## The Methodology

### Setup

Split the $n$ past tokens into contiguous blocks of $b = 64$ tokens. Call the always-kept current block $B_0$, and the $C$ earlier blocks $\mathcal{H} = \{B_1, \dots, B_C\}$. A lightweight selector $\mathcal{R}_\theta$ — they reuse the *AttnGate* module from SeerAttention-R unchanged, so the comparison is clean — scores the historical blocks:

$$\mathbf{s} = \mathcal{R}_\theta\!\left(\mathbf{q}, \{\mathbf{K}_{B_m}\}_{m=1}^{C}\right) \in \mathbb{R}^C$$

### The gated attention

$$\mathbf{g} = \operatorname{softmax}(\mathbf{s}), \qquad g_0 = 1$$
$$\mathcal{I} = \text{Top-}K(\mathbf{g}, K), \qquad \mathcal{S} = B_0 \cup \bigcup_{m \in \mathcal{I}} B_m$$
$$\mathbf{o}_{\text{SAS}} = \operatorname{softmax}\!\left(\mathbf{q}\mathbf{K}_{\mathcal{S}}^{\top} + \log \mathbf{g}_{\mathcal{S}}\right)\mathbf{V}_{\mathcal{S}}$$

The block gate is broadcast to every token in that block. The current block gets $\log g_0 = 0$, i.e. no bias. Because $\log g$ is added *inside* the softmax, it multiplies the attention probabilities after normalisation — a gate of $g_m$ scales block $m$'s share of the attention mass.

Loss is just next-token prediction. Backbone frozen; only $\theta$ trains.

### The four choices that make it work

This is the heart of the paper. The simple idea only works with the right details.

**1. Gate inside the softmax, not outside.** The alternative is $\operatorname{softmax}(\mathbf{q}\mathbf{K}^\top)(\mathbf{g} \odot \mathbf{V})$ — rescale the values afterwards. The gradients differ:

$$dg_m^{\text{inner}} = \sum_{i \in B_m} \frac{\tilde p_i}{g_m}\, d\mathbf{o}^\top(\mathbf{v}_i - \mathbf{o}), \qquad dg_m^{\text{outer}} = \sum_{i \in B_m} p_i\, d\mathbf{o}^\top \mathbf{v}_i$$

The outer version asks "is this block's value useful in absolute terms?". The inner version asks "is this block's value better than what I'm currently outputting?" — the $\mathbf{v}_i - \mathbf{o}$ term. That relative signal is what you need to *rank* blocks against each other, because attention mass is a fixed pie.

**2. Normalise the gates with softmax.** The alternatives: $\operatorname{sigmoid}(\mathbf{s})$ per block, or dumping raw $\mathbf{s}$ into the logits. With softmax, $\log g_m = s_m - \operatorname{LSE}(\mathbf{s})$. The shared $-\operatorname{LSE}(\mathbf{s})$ term does *not* cancel, because the current block has no such term — so it calibrates "how much attention goes to history at all" against the current block. It also makes the gates invariant to a global shift $\mathbf{s} \mapsto \mathbf{s} + c$, which removes a degenerate direction the selector could drift along.

Empirically (Fig. 3) sigmoid gates saturate toward 1 and raw logits collapse toward 0 with shrinking variance. Both end up as *ungated attention*: every block looks the same, no ranking is learned. Softmax is competitive — one block's gain is another's loss — so it cannot collapse this way.

**3. Keep the gates continuous.** The tempting alternative is a hard binary mask with a straight-through estimator: $\hat{\mathbf{g}} = \operatorname{stopgrad}(\mathbf{g}^{hard} - \mathbf{g}) + \mathbf{g}$. This blows up. With soft gates every block is in the normaliser, so $\tilde p_i^{soft} \le 1$. With hard gates the normaliser only sums over the selected set, and a *dropped* block gets

$$\tilde p_i^{hard} = \frac{\exp(z_i)}{\sum_{j \in \mathcal{S}} \exp(z_j)} \le \exp\!\left(z_i - \max_{j\in\mathcal{S}} z_j\right)$$

which is unbounded. Early in training the randomly initialised selector drops high-scoring blocks constantly, so $z_i > \max_{j \in \mathcal{S}} z_j$ happens often and the gradient (which is $\propto p_i$) explodes. This is a clean instance of [[Backpropagation#What goes wrong — and it all comes from that multiplication|exploding gradients]] coming from a bad normaliser, not from depth.

**4. Train on the sparse scope, not the full one.** Full scope means computing gated attention over *all* $C$ blocks during training, so every block gets its own gradient. Sparse scope only computes over $\mathcal{S}$. For an unselected block $m \notin \mathcal{I}$:

$$ds_m^{sparse} = -g_m \underbrace{\sum_{\ell \in \mathcal{I}} g_\ell\, dg_\ell}_{\text{only via selected blocks}}, \qquad ds_m^{full} = -g_m\left(-dg_m + \sum_{\ell=1}^{C} g_\ell\, dg_\ell\right)$$

Under sparse scope an unselected block's score moves only through the softmax normaliser — all unselected blocks get pushed the *same direction at once*, regardless of their own content (Fig. 4). That is a worse signal. But it converges to the same place and costs far less, so SAS uses sparse scope.

### Kernel

A naive implementation materialises the full $n \times n$ gated score matrix, which is fatal at 32K tokens. They wrote a Triton kernel in the [[FlashAttention- Fast and Memory-Efficient Exact Attention|FlashAttention]] style: during the streaming scan over KV tiles, add the block's $\log g$ to the tile's logits, mask non-selected blocks to $-\infty$, then do the usual online softmax update. Top-$K$ is encoded as a per-query threshold $\tau$ on the log gate, so no sort is needed, and a tile with zero selecting queries is skipped entirely. Backward accumulates the block gate gradient by summing score gradients within each selected block. Memory stays FlashAttention-level.

### Training recipe

- Backbones: Qwen3-4B / 8B / 14B, frozen.
- Data: 93.7K examples from OpenR1-Math-220k, one epoch, seq len 32,768, global batch 32.
- [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], lr 1e-3, cosine decay, block size 64.
- **One selector, trained on maths only, evaluated on everything** — reasoning, long-context, agentic. No per-task retraining.
- Serving: SGLang backend, dense prefill, sparse decode, FlashInfer sparse kernels, GQA with per-group selection.

## Ablation Studies and Experiments

### The design ablation (Table 1, Qwen3-4B, GPQA-Diamond, budget 2048, avg@16)

| Variant | 100 step | 1 epoch |
|---|---|---|
| Full attention (ceiling) | — | 56.1 |
| **Inner softmax gate, softmax act, soft, full scope** | 52.2 | 54.4 |
| Outer gate ($\mathbf{g} \odot \mathbf{V}$) | 38.9 | **41.6** |
| Sigmoid gate activation | 20.4 | **17.0** |
| Raw logit injection (no activation) | 19.9 | **18.8** |
| Hard Top-$K$ gate via STE | 42.4 | **46.0** |
| Sparse scope | 51.0 | 54.8 |
| Full scope + noise (`full*`) | 51.2 | 52.2 |

Read the failures, they are the story:

- **Sigmoid and raw logits get *worse* with training** (17.0 and 18.8, well below the 30-ish start). They also have the *lowest training loss* of all variants (Fig. 8). The model is minimising loss by making the gate meaningless — every block gets the same treatment, attention becomes effectively ungated and dense, and the loss is happy. Then you discretise at inference and it falls apart. **Training loss is not a valid proxy for ranking quality here.**
- **Outer gating plateaus at ~42.** Rescaling values after the softmax gives no relative signal.
- **STE is fast out of the gate (46.5 at 10 steps) but ends worse (46.0).** Lower training loss, much higher gradient norm — aggressive, unstable optimisation.
- Generation length tells the same story: the healthy variants converge to ~7,000-token traces; the broken ones sit at ~27,000, i.e. they are rambling and hitting the cap.
- Adding noise to scores to simulate full scope (`full*`) did essentially nothing (52.2 vs 54.8). Cheap tricks do not substitute for real gradients.

### Reasoning (Table 2)

At **budget 1024**, against SeerAttention-R (identical selector architecture, identical data, only the training signal differs):

| Model | MATH500 | GPQA-D |
|---|---|---|
| Qwen3-4B, SeerAttn-R → SAS | 84.67 → **90.65** | 39.84 → **50.41** |
| Qwen3-8B | 83.57 → **91.27** | 39.43 → **53.17** |
| Qwen3-14B | 86.12 → **92.93** | 45.64 → **61.14** |

At budget 2048 on Qwen3-4B, AIME24 goes 55.83 → **68.85** (+13.0); AIME25 45.16 → **56.38**. Training-free baselines are wrecked at this budget: Quest scores **0** on both AIME sets, StreamingLLM 29.7/15.9.

At budget 4096 SAS ties or beats full attention in places (AIME24 71.72 vs 71.25 on 4B; AIME25 63.41 vs 67.86 on 8B is still a gap). The advantage over distillation narrows a lot at 4096 — **when the budget is generous, ranking quality stops mattering much**.

### Transfer (Tables 3–5)

Selector trained on maths only:
- **LongBench**: ahead at nearly every budget, biggest gap on the longest bucket — Qwen3-14B at budget 2048, 8K+ inputs: 51.5 → **53.9**. At 4096, average 56.2 vs 56.6 full attention.
- **BFCL multi-turn**: +3.5 on 4B at budget 2048 (29.00 → 32.50); at 4096, 44.00 vs 44.50 full attention on 14B.
- **VitaBench** (realistic tool use): ahead on most metrics at 4096. Note SeerAttention-R actually *beats* SAS on the Delivery scenario at budget 2048 (32.2 vs 30.0) — not a clean sweep.

### Continued pretraining (Tables 6–7)

OLMo3-7B stage-1 base, backbone **and** selector trained jointly, ~50B tokens, seq len 8192. Average over 11 downstream tasks: SAS **43.28**, sliding-window baseline 43.24, HiLS-Attn-RoPE 41.68, dense base 43.88. LongBench average 30.0, tied best, beating the dense base (29.0). But CRUX (code reasoning) drops hard: 24.62 → **19.25**. This section is thin — one run, no ablations.

### Why it works (Section 6, the most interesting analysis)

They checked the obvious hypothesis — that SAS picks blocks with more attention mass — and **it is false**. SAS covers *less* per-layer attention mass than distillation, under both the raw $\exp(z_i)$ weighting and a value-aware $\exp(z_i + \log\|\mathbf{v}_i\|_2)$ weighting, in nearly every layer.

The actual mechanism: take the **union** of selected blocks across all layers and compare it to the full-attention oracle's union. SAS has consistently higher recall. So each layer hoards less mass, but the layers **cover different things**. Distillation's per-layer target is a local objective and produces redundant selections; end-to-end training lets the layers specialise.

Secondary evidence: at budget 4096 on Qwen3-4B, SAS produces shorter reasoning traces and truncates less often (max length 32,768), with the gap widest on AIME.

### Speed (Fig. 7)

Decode only; prefill is unchanged. Qwen3-4B, SGLang, single GPU, CUDA graphs, 512 timed tokens:
- Batch 1: near parity at 8K; **2.4× at 64K, 4.6× at 256K, 5.6× at 512K**.
- Batch 8: **~13× at 64K**.
- Budget size barely changes the speedup — the win comes from not reading the whole KV cache.
- Cost breakdown: attention compute is flat, but **Top-$K$ selection goes from 21% of the step at 8K to 90% at 512K**. Selector scoring also grows (it scans all block summaries). The selection stage, not attention, is the bottleneck at extreme context.

## Worth Remembering

**The honest limitation (Appendix B).** On RULER — needle-in-haystack retrieval — this falls over. Qwen3-14B at 128K: full attention 82.23, SAS 23.80, SeerAttention-R 14.95. SAS is roughly 60% better than the baseline and still catastrophically far from dense. The authors blame **pooled block summaries**: averaging 64 tokens into one vector destroys the sharp, localised signal a needle consists of. If your workload is retrieval over very long context, this line of work is not ready.

**The loss-is-not-the-metric trap.** The sigmoid and raw-logit variants achieved the *lowest training loss* and the *worst* downstream accuracy. The model found a way to make the gate uninformative, which is fine during training (attention becomes dense) and fatal at inference (attention becomes hard Top-$K$ on garbage scores). Any time your training forward pass differs from your inference forward pass, monitor the thing that differs — here, the variance of the selector logits — not the loss.

**Why softmax over sigmoid, in one line.** Softmax is competitive; sigmoid is not. Competition is what prevents the "everybody wins" degenerate solution. Same reason [[Sparsely-Gated Mixture-of-Experts Layer|MoE]] routers normalise across experts.

**Gate placement is a general lesson.** Multiplying $\mathbf{V}$ after the softmax cannot teach relative importance, because the probabilities are already fixed. If you want a learned signal to compete for a fixed budget, it must enter *before* the normaliser. Compare [[Gated Activation]], where gates modulate magnitudes rather than allocate a budget — different job, different placement.

**Controlled comparison, which is rare.** SAS and SeerAttention-R share the AttnGate architecture, the training data, the backbone, and the inference kernel. The *only* difference is the training signal. That makes the "end-to-end beats distillation" claim unusually clean — contrast with the sloppiness documented in [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|Are We Really Making Much Progress]].

**Practical caveats.**
- Prefill is untouched. If your workload is prefill-heavy (long prompt, short answer), this buys you very little.
- The Top-$K$ ranking cost eats the win past ~256K. You need a better selection kernel, not a better attention kernel, to go further.
- One selector transfers across maths → long-context → agentic, which is a genuinely useful property, but everything was trained on OpenR1-Math. Transfer to, say, code-heavy agents is untested (and the CRUX regression in the pretraining run is mildly worrying).
- The standard deviations on AIME are 6.5–7.6 points. Single AIME numbers here mean little; the 12–13 point gaps do.

**Open questions.** Does the cross-layer complementarity effect show up if you distil against a *global* rather than per-layer target? Could you keep continuous gates at inference time rather than discretising, trading a little speed for accuracy? And could the sparse-scope gradient problem be fixed cheaply by sampling a few unselected blocks per step to give them real gradients?

## Links

Related: [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Flash Attention]] · [[Attention Is All You Need]] · [[Causal Attention]] · [[Query, Key, and Value (QKV)]] · [[Sparsely-Gated Mixture-of-Experts Layer]] · [[Distilling the Knowledge in a Neural Network]] · [[Distillation]] · [[Backpropagation]] · [[Cross Entropy]] · [[GQA- Training Generalized Multi-Query Transformer Models]] · [[Fast Transformer Decoding- One Write-Head is All You Need (MQA)]] · [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Gated Activation]] · [[Train Short, Test Long (ALiBi)]] · [[Auto-regressive models]]

New topics worth writing: Straight-through estimator, Block-sparse attention, KV cache eviction and selection, Triton kernel programming, SeerAttention and learned attention gates, StreamingLLM and attention sinks, Quest query-aware sparsity, RULER benchmark, YaRN context extension, Gumbel-softmax and differentiable Top-K, Native Sparse Attention (NSA), MoBA mixture-of-block-attention
