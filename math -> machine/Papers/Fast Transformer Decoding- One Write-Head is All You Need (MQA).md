---
title: "Fast Transformer Decoding: One Write-Head is All You Need (MQA)"
authors: ["Shazeer"]
year: 2019
arxiv: "1911.02150"
url: https://arxiv.org/abs/1911.02150
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, transformers, llm]
---
## The Core Idea

Generating text one token at a time is slow, and the reason is not maths — it is memory traffic. At each step the model must read back the whole cache of keys and values from every past position, for every head. On a TPU or GPU, arithmetic is roughly 100× faster than memory bandwidth, so the chip spends its time waiting for numbers to arrive, not multiplying them.

The fix is almost embarrassingly small. In standard [[Attention Is All You Need#Multi-head attention|multi-head attention]] each of the $h$ heads has its own queries, its own keys, and its own values. **Multi-query attention keeps $h$ separate query projections but gives all heads one shared key projection and one shared value projection.** The cache you have to reload each step shrinks by a factor of $h$ — with $h=8$, that is 8× less to read.

> [!NOTE] Multi-query attention (MQA)
> Attention where the "read heads" (queries) stay multiple but the "write head" (the K/V that gets stored) is single. Cache size per layer goes from $b \cdot h \cdot m \cdot k$ down to $b \cdot m \cdot k$. ^multi-query

Why this did not exist before: during **training** nobody notices the problem. Training processes all $n$ positions in parallel, so the K/V tensors are loaded once and reused across many queries — the memory-to-arithmetic ratio is $O(1/k + 1/(bn))$, tiny. Only in **incremental decoding**, where positions must be produced one after another because each token feeds the next, does the ratio blow up to $\Theta(n/d + 1/b)$. When $n \approx d$ that ratio is around 1, meaning one byte moved per one multiply — catastrophic. MQA is a fix aimed squarely at a cost that only shows up at serving time.

What it unlocks: decoder step time on WMT14 EN-DE drops from **46 µs to 3.8 µs per token** — a 12× speedup — with BLEU essentially unchanged. This is the trick that later became grouped-query attention and is why modern LLM serving stacks can hold long KV caches in memory at all.

## The Methodology

**The arithmetic that motivates everything.** Assume $m = n$, $k = v = d/h$, $n \le d$.

Incremental multi-head decoding, across $n$ steps:
- Arithmetic: $\Theta(b n d^2)$
- Memory touched: $\Theta(b n^2 d + n d^2)$ — the first term is reloading $K$ and $V$ at every step, the second is reloading the projection matrices $P_q, P_k, P_v, P_o$.
- Ratio: $\Theta\!\left(\frac{n}{d} + \frac{1}{b}\right)$

The $1/b$ term is easy — just batch more sequences, memory permitting. The $n/d$ term is the villain. It comes from the K/V cache having size $b\,h\,m\,k = b n^2$.

Incremental multi-query decoding:
- Memory touched: $\Theta(b n d + b n^2 k + n d^2)$
- Ratio: $\Theta\!\left(\frac{1}{d} + \frac{n}{dh} + \frac{1}{b}\right)$

The $n/d$ became $n/(dh)$. That is the whole paper.

**The code change.** Literally: delete the letter `h` from the einsum equations wherever it indexes $K$, $V$, $P_k$, or $P_v$.

Multi-head:
```
K = einsum("bmd,hdk->bhmk", M, P_k)
logits = einsum("bhnk,bhmk->bhnm", Q, K)
O = einsum("bhnm,bhmv->bhnv", weights, V)
```
Multi-query:
```
K = einsum("bmd,dk->bmk", M, P_k)      # P_k is now [d,k], not [h,d,k]
logits = einsum("bhnk,bmk->bhnm", Q, K) # h broadcasts over the shared K
O = einsum("bhnm,bmv->bhnv", weights, V)
```
$P_q$ and $P_o$ keep their head dimension. So queries still ask $h$ different questions; they just all read the same book.

**Models trained.**

*Translation, WMT 2014 EN-DE.* Encoder-decoder [[Attention Is All You Need|Transformer]], 6 layers, $d_{model}=1024$, $d_{ff}=4096$, $h=8$, $d_k = d_v = 128$, learned positional embeddings, embedding weights tied to the output layer. 211M parameters. 100K steps ($\approx$ 20 epochs), batch of 128 examples, each 256 input + 256 target tokens (sentences concatenated to fill the length). 32-core TPUv3, ~2 hours per model.

*Language modelling, Billion-Word Benchmark.* Decoder-only, 6 layers, $d_{model}=1024$, $d_{ff}=8192$, $h=8$, $d_k = d_v = 128$. 192M params, 136K steps (10 epochs), 64K tokens per batch, ~3 hours on 32-core TPUv3.

**The key fairness detail.** Removing per-head K/V projections deletes parameters. To keep the comparison honest, the feed-forward hidden width is widened from 4096 to **5440** in the translation model (and 8192 → 9088 in the LM) so total parameter count is identical. Every variant in every table has the same parameter budget.

MQA replaces **all** attention layers — encoder self-attention, decoder self-attention, and encoder-decoder cross-attention — not just the decoder.

## Ablation Studies and Experiments

**The real competition is not multi-head vs multi-query.** It is multi-query vs the obvious cheaper alternatives: just use fewer heads, or make $d_k, d_v$ smaller. Those also shrink the cache. Do they hurt more?

WMT14 EN-DE, greedy decode, sacrebleu on wmt13 dev / wmt14 test:

| Attention | $h$ | $d_k,d_v$ | $d_{ff}$ | ln(PPL) dev | BLEU dev | BLEU test (beam 1 / 4) |
|---|---|---|---|---|---|---|
| multi-head | 8 | 128 | 4096 | 1.424 | 26.7 | 27.7 / 28.4 |
| **multi-query** | 8 | 128 | 5440 | 1.439 | 26.5 | 27.5 / **28.5** |
| multi-head local | 8 | 128 | 4096 | 1.427 | 26.6 | 27.5 / 28.3 |
| multi-query local | 8 | 128 | 5440 | 1.437 | 26.5 | 27.6 / 28.2 |
| multi-head | 1 | 128 | 6784 | 1.518 | 25.8 | — |
| multi-head | 2 | 64 | 6784 | 1.480 | 26.2 | 26.8 / 27.9 |
| multi-head | 4 | 32 | 6784 | 1.488 | 26.1 | — |
| multi-head | 8 | 16 | 6784 | 1.513 | 25.8 | — |

Read the bottom four rows. Every naive way of shrinking the cache costs **0.5–0.9 BLEU** and 0.05–0.09 nats of log-perplexity. MQA costs **0.2 BLEU** and 0.015 nats. And with beam-4 decoding MQA actually *edges out* the baseline at 28.5 vs 28.4 — almost certainly noise, but it means the degradation is inside the noise floor.

The lesson: **the head dimension matters for queries, not for keys and values.** Having 8 different ways of *asking* is what multi-head buys you. Having 8 different copies of the *stored content* buys you comparatively little.

Billion-Word LM, dev perplexity per word (lower better):

| Attention | $h$ | $d_k,d_v$ | $d_{ff}$ | PPL |
|---|---|---|---|---|
| multi-head | 8 | 128 | 8192 | 29.9 |
| **multi-query** | 8 | 128 | 9088 | 30.2 |
| multi-head | 1 | 128 | 9984 | 31.2 |
| multi-head | 2 | 64 | 9984 | 31.1 |
| multi-head | 4 | 32 | 9984 | 31.0 |
| multi-head | 8 | 16 | 9984 | 30.9 |

Same shape: MQA loses 0.3 PPL, the alternatives lose 1.0–1.3.

**Speed**, on one TPUv2 (8 cores). Batch of 1024 sequences (128/core), source length 128, target length 128. Values are µs per output token:

| Attention | Training | Inference (enc + dec) | Beam-4 (enc + dec) |
|---|---|---|---|
| multi-head | 13.2 | 1.7 + **46** | 2.0 + **203** |
| multi-query | 13.0 | 1.5 + **3.8** | 1.6 + **32** |
| multi-head local | 13.2 | 1.7 + 23 | 1.9 + 47 |
| multi-query local | 13.0 | 1.5 + 3.3 | 1.6 + 16 |

Raw numbers: baseline encoder 222ms, each decoder step 47ms. MQA encoder 195ms, each decoder step **3.9ms**. That is a **12× decoder speedup** greedy, **6.3×** with beam-4.

Note what *does not* change: **training time is identical** (13.2 vs 13.0 µs). MQA buys you nothing at train time, because training was never bandwidth-bound. And the encoder barely moves (1.7 → 1.5), because the encoder runs in parallel over positions too. All the win is in the incremental decoder.

**Local attention is orthogonal, and that is the point of those two extra rows.** Restricting decoder self-attention to the current position plus the previous 31 gets you 46 → 23 µs on its own. Stacking it with MQA gets 3.8 → 3.3. Multiplying the tricks together works; they attack different parts of the same $bn^2$ term. Local attention alone is a much weaker lever than MQA here.

**What did not work:** nothing dramatic is reported as a failure, but the fewer-heads and smaller-$d_k$ rows are exactly that — the intuitive baselines a reviewer would demand, and they all lose more quality for the same cache saving. The $h=1$ row (a true single head, 25.8 BLEU) is the clearest demonstration: collapsing queries too is 0.9 BLEU worse, while collapsing only K/V is 0.2 BLEU worse.

## Worth Remembering

- **This is a serving-time optimisation dressed as an architecture change.** If you only ever benchmark training throughput, MQA looks like it does nothing. The paper is a good reminder that arithmetic-op counts (FLOPs) are the wrong currency for autoregressive decoding — bytes moved is the currency. Same lesson as [[FlashAttention- Fast and Memory-Efficient Exact Attention|FlashAttention]], from the opposite direction: Flash fixes bandwidth during the parallel forward pass, MQA fixes it during the serial decode.

- **The parameter-matching via $d_{ff}$ widening is doing hidden work.** MQA-with-narrow-FFN was never tested. So the honest claim is "at equal parameters, MQA ≈ MHA", not "removing K/V heads is free". Some of the retained quality may be the fatter feed-forward layer compensating.

- **The gain scales with $h$.** The $n/(dh)$ term means a model with 32 or 64 heads saves far more than the 8-head model tested here. Modern LLMs have many more heads than 8, which is why MQA mattered far more in 2023 than the modest 2019 numbers suggest.

- **Quality loss is small but real and one-sided.** Every dev metric is slightly worse: 1.424 → 1.439 ln PPL, 29.9 → 30.2 PPL. The single beam-4 test win at 28.5 should not be over-read from one dev/test pair. Later work (grouped-query attention) found the loss grows with scale and gets worse for tasks needing fine-grained retrieval, which is why the industry settled on an intermediate: share K/V across *groups* of heads rather than all of them.

- **Training instability is not discussed.** Later practitioners found MQA models can be harder to train and often need retraining from scratch rather than conversion. Nothing here warns you.

- **Only 2 hours of training per model on 32 TPUv3 cores.** These are small, quickly-trained models by any modern standard. The conclusion "quality degradation is minor" is established at 211M parameters, not at 7B.

- Practical caveat if you implement it: the shared $K$ has shape `[b, m, k]`, so the logits einsum `bhnk,bmk->bhnm` broadcasts $K$ across heads. On hardware this can turn a clean batched matmul into a broadcast matmul — check your kernel actually exploits the smaller tensor rather than materialising $h$ copies.

- Open question worth chasing: is a single shared K/V enough because the [[Query, Key, and Value (QKV)|value]] content genuinely is head-independent, or because $P_o$ (which keeps its heads) can re-diversify the shared values on the way out? The paper never probes this.

## Links

Related: [[Attention Is All You Need]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Query, Key, and Value (QKV)]] · [[Causal Attention]] · [[Flash Attention]] · [[Auto-regressive models]] · [[Seq2Seq models]] · [[Linear Projection]] · [[Train Short, Test Long (ALiBi)]] · [[Megatron-LM- Training Multi-Billion Parameter Models Using Model Parallelism]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[Mixed Precision Training]]

New topics worth writing: Grouped-Query Attention (GQA), KV cache management and paged attention, arithmetic intensity / roofline model, memory-bandwidth-bound vs compute-bound kernels, local/sliding-window attention, speculative decoding, BLEU and sacrebleu evaluation protocol, einsum notation for tensor contractions
