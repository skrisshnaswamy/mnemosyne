---
title: "GQA: Training Generalized Multi-Query Transformer Models"
authors: ["Joshua Ainslie", "James Lee-Thorp", "Michiel de Jong", "Yury Zemlyanskiy", "Federico Lebrón", "Sumit Sanghai"]
year: 2023
arxiv: "2305.13245"
url: https://arxiv.org/abs/2305.13245
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, transformers, llm, optimization, theory]
---
## The Core Idea

When a transformer generates text one token at a time, the slow part is not maths. It is memory traffic. At every step the chip must read the model weights **and** the whole cache of past keys and values (the KV cache) out of high-bandwidth memory. Reading is slower than computing. So the KV cache size sets the speed.

[[Fast Transformer Decoding- One Write-Head is All You Need (MQA)|Multi-query attention]] fixed this by brute force: keep $H$ query heads, but only **one** key head and **one** value head shared by all of them. The KV cache shrinks by a factor of $H$. But you lose capacity, quality drops, and training gets unstable. Also, nobody wants to train a second model from scratch just to serve it faster.

Two ideas here, both simple:

1. **Uptraining.** You do not need to train an MQA model from scratch. Take an existing multi-head checkpoint, **mean-pool** the key and value projection matrices across heads into one, then continue pre-training for only $\alpha = 5\%$ of the original steps. That is enough for the model to adapt.

2. **Grouped-query attention (GQA).** Instead of 1 KV head or $H$ KV heads, use $G$ of them. Split the query heads into $G$ groups; every head in a group shares one key head and one value head. $G=1$ is MQA. $G=H$ is normal multi-head attention. Anything in between is a dial.

> [!NOTE] Grouped-query attention
> Query heads are partitioned into $G$ groups. Each group has its own single key/value head. KV cache shrinks by $H/G$ instead of $H$. ^grouped-query

Why does the middle of the dial matter? Because bigger models have more heads. Going all the way to one KV head is a *more* aggressive cut for a big model than for a small one — you throw away proportionally more capacity as you scale. GQA keeps the cut proportional. There is also a sharding argument: when you split a model across $N$ chips, the single MQA key/value head has to be **replicated** on every chip, so you pay for it $N$ times anyway. With $G=8$ groups and 8 partitions, each chip owns one group and nothing is wasted.

The result: T5-XXL with GQA-8 scores 47.1 average versus 47.2 for full multi-head, at 0.28 s/sample versus 1.51 s/sample. Roughly 5× faster, quality basically unchanged. This is why almost every modern open LLM (Llama 2 70B onward, Mistral, and successors) ships with GQA.

## The Methodology

**Checkpoint conversion.** For each group $g$ of query heads, build the shared key projection by averaging the original per-head key projections in that group:

$$W_K^{(g)} = \frac{1}{|g|}\sum_{h \in g} W_K^{(h)}$$

Same for values. Query and output projections are untouched. This is a [[Linear Projection|linear projection]] surgery on the weight matrices, not on activations.

**Uptraining.** Continue pre-training the converted checkpoint on the *same* data and *same* recipe as the original, for $\alpha$ fraction of the original steps. At $\alpha = 0.05$ this cost about 600 TPUv3 chip-days for T5-XXL.

**Where GQA is applied.** Decoder self-attention and cross-attention only. **Not** encoder self-attention — the encoder processes the whole input in parallel, so it is compute-bound, not memory-bandwidth-bound. There is no incremental cache to load.

**Setup.**
- Architecture: T5.1.1 ([[Exploring the Limits of Transfer Learning (T5)|T5]]), encoder-decoder, in JAX/Flax.
- Optimiser: Adafactor with T5's original hyperparameters and schedule.
- Fine-tuning: constant LR 0.001, batch 128, [[Dropout- A Simple Way to Prevent Overfitting|dropout]] 0.1 everywhere.
- Lengths: CNN/DM and WMT use 512 in / 256 out; other summarisation 2048 in / 512 out; TriviaQA 2048 in / 32 out.
- Greedy decoding. Timing on 8 TPUv4 chips, batch up to 32 per chip, parallelisation tuned per model.

**Data:** CNN/Daily Mail, arXiv, PubMed, MediaSum, Multi-News (summarisation, Rouge-1); WMT14 EN-DE (BLEU); TriviaQA (F1). GLUE-style classification is skipped because autoregressive decoding barely matters there.

## Ablation Studies and Experiments

**Main table (T5, dev sets):**

| Model | Time/sample (s) | Avg | CNN R1 | arXiv | PubMed | MediaSum | MultiNews | WMT BLEU | TriviaQA F1 |
|---|---|---|---|---|---|---|---|---|---|
| MHA-Large | 0.37 | 46.0 | 42.9 | 44.6 | 46.2 | 35.5 | 46.6 | 27.7 | 78.2 |
| MHA-XXL | 1.51 | 47.2 | 43.8 | 45.6 | 47.5 | 36.4 | 46.9 | 28.4 | 81.9 |
| MQA-XXL | 0.24 | 46.6 | 43.0 | 45.0 | 46.9 | 36.1 | 46.5 | 28.5 | 81.3 |
| GQA-8-XXL | 0.28 | 47.1 | 43.5 | 45.4 | 47.7 | 36.3 | 47.2 | 28.4 | 81.6 |

Read the two comparisons that matter. **MQA-XXL beats MHA-Large on quality (46.6 vs 46.0) while being faster (0.24 vs 0.37 s).** So the right move is not "shrink the model" — it is "keep the big model, shrink the KV cache." And **GQA-8 recovers almost all the gap to MHA-XXL (47.1 vs 47.2) for 0.04 s extra over MQA.**

**Ablation 1 — how to convert the checkpoint.** T5-Large uptrained to MQA at $\alpha=0.05$, scored on a 3-task subset:
- Mean-pool the heads: ~55.5 (best)
- Take just the first head: ~54.8
- Random re-init: ~54.4 (worst)

The ordering is exactly "how much of the pre-trained information survives." Random init throws it all away and even 5% uptraining does not recover it. This is the clearest signal in the paper that uptraining is *repair*, not *retraining*.

**Ablation 2 — how much uptraining.** Sweeping $\alpha$ from 0 to 0.1 for XXL:
- At $\alpha = 0$ (straight after conversion, no extra training), **GQA-8 is already reasonable; MQA is not usable.** Averaging 8 heads into 1 destroys something that averaging 8 heads into 8 groups does not.
- Both gain a lot by $\alpha = 0.05$.
- Diminishing returns past $\alpha = 0.10$.

**Ablation 3 — number of groups.** Sweeping $G \in \{1,4,8,16,32,64\}$ at 2048 in / 512 out. Going 1 → 8 costs only a modest slowdown; the curve then steepens sharply toward MHA. They picked $G=8$ as the knee. Note this is a *speed* curve, not a quality curve — the quality-vs-$G$ curve is not reported, which is a real gap.

**What did not work.**
- Training MQA T5-Large **from scratch**: frequent loss spikes during pre-training, and the models **diverged immediately** when fine-tuned on long-input tasks. This is in the appendix and is arguably the strongest practical argument for GQA.
- Uptrained MQA is more stable but still high-variance; for unstable tasks they had to average over 3 fine-tuning runs to report a number.
- Uptrained GQA was simply stable, so they never dug into the root cause of MQA's instability.
- Random initialisation of the new KV heads (above).

## Worth Remembering

- **The bottleneck framing is the real lesson.** Decoding is memory-bandwidth-bound, not FLOP-bound. KV cache grows with model dimension $d$, but parameters and FLOPs grow with $d^2$. So as models get bigger, the KV cache is *relatively* cheaper — which is exactly why the aggressive MQA cut is less necessary at scale and GQA's interpolation pays off. Same roofline logic that drives [[FlashAttention- Fast and Memory-Efficient Exact Attention|FlashAttention]], applied to a different tensor.

- **Uptraining as a general recipe.** The idea is borrowed from Sparse Upcycling (dense checkpoint → Mixture-of-Experts). The pattern is: change the architecture by a weight-space surgery that preserves as much information as possible, then pay 5% of pre-training compute to let the model settle. Worth stealing for other architecture swaps.

- **Limitations the authors state plainly.** They evaluate long-output generation with Rouge, which they call a flawed metric, so they are not sure the trade-off curve is right. They **never compared uptrained GQA-XXL to a GQA-XXL trained from scratch** — so the cost of uptraining vs. clean training is unknown. And everything is encoder-decoder T5; they only *predict* GQA helps decoder-only models more (correctly, as it turned out — decoder-only models have no cross-attention, so all attention is the memory-bound kind).

- **Practical caveat:** $G$ is usually chosen to match the tensor-parallel degree, so each shard holds exactly one KV group. If your serving topology changes, the "right" $G$ changes. See [[Megatron-LM- Training Multi-Billion Parameter Models Using Model Parallelism|Megatron-style tensor parallelism]] for why replication of a single head is wasteful.

- **Open question:** the paper gives a speed-vs-$G$ curve and a quality number for $G=8$ only. Nobody in this paper measured quality as a smooth function of $G$. If you are choosing $G$ for a new model, you are extrapolating from one point.

- Rabe independently implemented GQA in Flaxformer around the same time — this is a case of the idea being obvious in hindsight once you stare at the MQA/MHA gap.

## Links

Related: [[Fast Transformer Decoding- One Write-Head is All You Need (MQA)]] · [[Attention Is All You Need]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Flash Attention]] · [[Exploring the Limits of Transfer Learning (T5)]] · [[Megatron-LM- Training Multi-Billion Parameter Models Using Model Parallelism]] · [[Query, Key, and Value (QKV)]] · [[Causal Attention]] · [[Distilling the Knowledge in a Neural Network]] · [[Mixed Precision Training]] · [[Linear Projection]] · [[Seq2Seq models]]

New topics worth writing: KV cache, memory-bandwidth bound vs compute bound (roofline model), speculative decoding, sparse upcycling, post-training quantization (GPTQ / LLM.int8), Adafactor, Rouge and its failure modes, multi-head latent attention (MLA)
