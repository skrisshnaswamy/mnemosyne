---
title: "Train Short, Test Long (ALiBi)"
authors: ["Ofir Press", "Noah A. Smith", "Mike Lewis"]
year: 2021
arxiv: "2108.12409"
url: https://arxiv.org/abs/2108.12409
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, transformers, vision]
---
## The Core Idea

Transformers have no built-in sense of order. Something must tell the model which token came first. The original [[Attention Is All You Need|Transformer]] added sinusoidal position vectors to the word embeddings at the bottom of the network. That works — but only for sequence lengths the model actually saw during training. Feed such a model a longer input at test time and perplexity explodes. A WikiText-103 model trained on 512 tokens goes from 20.05 perplexity at 512 tokens to **406** at 15,512 tokens.

ALiBi ("Attention with Linear Biases") fixes this by deleting position embeddings entirely and instead **subtracting a penalty from each attention score that grows linearly with how far apart the query and key are**. A token 10 places back gets a penalty of $10m$; a token 100 places back gets $100m$. Each attention head uses a different slope $m$, fixed before training, never learned.

That is the whole method. A few lines of code, folded into the causal mask that already exists.

Why it matters: you can now **train short and test long**. Train on 1024 tokens, run inference on 2048, and get the same perplexity as a sinusoidal model that was trained on 2048 — while training 11% faster and using 11% less memory. Attention cost is quadratic in sequence length, so halving the training length is a large, real saving.

The second, quieter finding: even with no extrapolation at all, ALiBi beats sinusoidal on smaller corpora. On WikiText-103 with $L = L_{valid} = 3072$, ALiBi gets 17.60 vs sinusoidal's 18.67. The recency penalty is a useful [[An Image is Worth 16x16 Words (ViT)#^inductive-bias|inductive bias]] for language, not just an extrapolation trick.

> [!NOTE] Extrapolation (length) ^length-extrapolation
> A model's ability to keep performing well when the number of input tokens at validation time exceeds the number it was trained on. Distinct from generalising to new *data* — here the data distribution is the same, only the sequence is longer.

## The Methodology

Standard causal attention for query $i$ against the first $i$ keys:

$$\text{softmax}(\mathbf{q}_i \mathbf{K}^\top)$$

ALiBi:

$$\text{softmax}\big(\mathbf{q}_i \mathbf{K}^\top + m \cdot [-(i-1), \ldots, -2, -1, 0]\big)$$

The bias vector is a fixed ramp. The key right next to the query gets $0$. The key one step further back gets $-m$. The oldest key gets $-(i-1)m$. Nothing is learned, nothing is added to the values, and nothing is added at the bottom of the network — position information enters at **every layer**, in the keys/queries only.

**The slopes.** For $n$ heads, the slopes form a geometric sequence starting at $2^{-8/n}$ with ratio $2^{-8/n}$. For 8 heads: $\tfrac{1}{2}, \tfrac{1}{4}, \ldots, \tfrac{1}{256}$. For 16 heads, interpolate: $2^{-0.5}, 2^{-1}, 2^{-1.5}, \ldots, 2^{-8}$. Small-slope heads decay slowly and see far; big-slope heads are sharply local. This gives the model a spread of receptive-field sizes for free.

**Implementation.** In practice, transformer LMs already add an $L \times L$ causal mask (a matrix of $0$ and $-\infty$) to the pre-softmax scores. ALiBi just bakes the linear ramp into that mask. The mask grows from $L \times L$ to $n \times L \times L$ (one per head), costing up to 100 MB extra. Zero extra FLOPs, zero extra parameters. Measured speed difference vs sinusoidal: within 1% for training, 3% for inference.

**Baselines compared.** All were dropped into the same Baevski & Auli LM (16 layers, $d=1024$, 8 heads, FFN 4096, tied embeddings, 205 epochs, same seed):

- *Sinusoidal* — fixed sin/cos vectors added at input.
- *Rotary* ([[RoFormer- Enhanced Transformer with Rotary Position Embedding|RoPE]]) — rotates keys and queries by sinusoids at every layer. Position enters every layer, never the values. ALiBi copies these two design choices.
- *T5 bias* — a **learned** scalar added per query-key distance, shared across layers, with distant offsets bucketed together.

**Big run.** 1.3B params, 25 layers, 16 heads, $d=2048$, FFN 8192, one epoch = 50k updates on 128 V100s, over 461 GB of CC-100 + the [[RoBERTa- A Robustly Optimized BERT Pretraining Approach|RoBERTa]] corpus. Crucially: **the slopes were tuned once on WikiText-103 and reused unchanged** here and on BookCorpus.

## Ablation Studies and Experiments

**Extrapolation, WikiText-103, trained on $L=512$**, evaluated at growing $L_{valid}$ (perplexity):

| $L_{valid}$ | Sinusoidal | Rotary | T5 bias | ALiBi |
|---|---|---|---|---|
| 512 | 20.05 | 20.07 | 19.65 | 19.73 |
| 1012 | 43.54 | 21.37 | 18.79 | 18.73 |
| 3512 | 178.97 | 35.54 | 22.91 | **18.40** |
| 15512 | 406.01 | 79.25 | OOM | **18.31** |

Sinusoidal improves for about 20–50 extra tokens then collapses. Rotary buys ~200 extra tokens. T5 bias buys ~600–800 and genuinely extrapolates — but it is **at least twice as slow to train** (14.4k words/sec at $L=512$ vs sinusoidal's 28.5k at $L=1024$), so its extrapolation buys you no wall-clock win at all. That is the paper's sharpest argument: T5 bias already solved extrapolation; it just solved it uselessly.

**Train short, beat long.** ALiBi at $L=512$, evaluated at 3072, gets 18.40 — better, by a statistically significant margin (sinusoidal std dev 0.24), than the sinusoidal model *trained* on 3072 (18.67). And it trains 1.84× faster and fits on a much smaller GPU.

**1.3B scale.** ALbi $L=1024$ evaluated at 2048 gets 8.92 vs sinusoidal $L=2048$'s 9.01, using 3.1 GB less memory, reaching any given perplexity ~11% sooner. Note the honest asterisk: at this scale, with matched $L$, ALiBi ties rather than wins (8.84 vs 8.83 at $L=2048$). The recency prior's quality gain is a **low-resource** benefit; at 461 GB it washes out and only the efficiency remains.

**What did not work:**

- **Learned slopes.** They tried making $m$ trainable. Extrapolation got worse.
- **Multiplying** attention scores by a distance function (as in the Distance-Aware Transformer) instead of adding — degraded performance.
- Roughly ten hand-tried slope sets. The rule that emerged: slopes should live in $(0,1)$ and get denser near $0$. Sampling slopes from an exponential distribution sometimes worked but with high variance.

**The ablation that undercuts the story.** Appendix B is the honest part. Two explanations for why longer $L_{valid}$ helps: (1) the model genuinely uses the longer history, or (2) it merely dodges the **early token curse** — with non-overlapping chunks, the first tokens of each chunk have almost no context, and longer chunks mean a smaller fraction of such starved predictions.

They test this with sliding-window evaluation at stride $S=1$, which gives *every* prediction maximal context. If explanation (1) were right, ALiBi should keep improving. It does not — perplexity stays **flat**: $L=512$ model gives 17.98 / 17.92 / 18.20 / 18.28 / 18.30 at $L_{valid} = 512/1024/1536/2048/3072$. So explanation (2) is the operative one. **ALiBi is not really reading longer histories; it is failing gracefully.** (Sinusoidal, by contrast, still explodes: 18.35 → 204.42 → 360.12.)

> [!NOTE] Early token curse ^early-token-curse
> When you chop an evaluation set into non-overlapping chunks of length $L$, the predictions near the start of each chunk have almost no left context, which inflates perplexity. Longer chunks reduce the fraction of such predictions. Any "longer context helps" claim measured with non-overlapping inference is confounded by this.

## Worth Remembering

- **The mechanism, in one line:** attention scores get $-m \cdot \text{distance}$ before the softmax, so far-away tokens are exponentially down-weighted after exponentiation. Each head has its own decay rate.
- ALiBi's own analysis says it likely does *not* exploit context beyond $L$. Its value is that perplexity stays flat instead of exploding, so you get a cheap, safe long-context inference path without the very slow sliding window. Anyone reading ALiBi as "free unlimited context" is over-reading it.
- Peak performance sits at about $2L$, then plateaus. The authors' guess: at $L_{valid} = 2L$ under non-overlapping inference, half the predictions still have context lengths the model saw in training; past that, fewer than half do.
- The slopes transferred with **zero tuning** from WikiText-103 (103M tokens, Wikipedia) to BookCorpus (700M tokens, novels) to a 461 GB web corpus and a 1.3B-param model. Rare for a hyperparameter to be that robust — it is why ALiBi got adopted (BLOOM, MPT).
- Practical caveat: ALiBi's per-head mask is $n \times L \times L$, which interacts badly with fused kernels that assume a shared mask. [[FlashAttention- Fast and Memory-Efficient Exact Attention|FlashAttention]] needed an explicit ALiBi code path.
- Second caveat: the strong recency prior is right for language modelling but is a real constraint if you need a token to attend precisely to something 5000 positions back (retrieval, long-document QA). RoPE plus interpolation eventually won that ground.
- Results are perplexity-only. No downstream task evaluation at all — worth remembering before assuming the quality win carries.
- Comparison is causal-LM only; ALiBi for bidirectional encoders like [[BERT- Pre-training of Deep Bidirectional Transformers|BERT]] is untested here.

## Links

Related: [[Attention Is All You Need]] · [[RoFormer- Enhanced Transformer with Rotary Position Embedding]] · [[Exploring the Limits of Transfer Learning (T5)]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Causal Attention]] · [[Query, Key, and Value (QKV)]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[RoBERTa- A Robustly Optimized BERT Pretraining Approach]] · [[Auto-regressive models]] · [[Scaling Laws for Neural Language Models]] · [[An Image is Worth 16x16 Words (ViT)]]

New topics worth writing: Perplexity as a language-model metric, Relative vs absolute position encoding, Position interpolation for RoPE, Sliding-window vs non-overlapping evaluation, Transformer-XL and recurrence-based context extension, Long-context attention benchmarks
