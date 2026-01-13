---
title: "BERT4Rec: Sequential Recommendation with Bidirectional Transformer"
authors: ["Fei Sun", "Jun Liu", "Jian Wu", "Changhua Pei", "Xiao Lin", "Wenwu Ou", "Peng Jiang"]
year: 2019
arxiv: "1904.06690"
url: https://arxiv.org/abs/1904.06690
priority: Must-Read
read_on: 2026-08-22
tags: [paper, transformers, llm]
---
## The Core Idea

Sequential recommendation means: given the list of things a user clicked, watched, or bought in order, guess the next one. Before this paper, every strong model read that list **left to right only**. RNNs (GRU4Rec) walked forward one step at a time. SASRec used a Transformer with a [[Causal Attention|causal mask]], so position $t$ could only look at positions $\le t$.

The claim here is that left-to-right is the wrong prior for user behaviour. Two reasons:

1. **Weak representations.** If item $v_3$ can only see $v_1, v_2$, its hidden vector is starved. In a bidirectional model, $v_3$ also sees $v_4, v_5$ and becomes a much richer summary of "where this user is in their journey".
2. **The order is not real.** Text has grammar, so word order is rigid. A shopping history is not like that. You might have bought the phone case before the phone, or on the same day in an arbitrary order. Forcing a strict left-to-right causal story onto noisy behaviour data is an assumption the data does not support.

So: use a **bidirectional** Transformer encoder over the item sequence. But there is an obvious trap. If every position sees every other position, and you train by "predict the next item at every position", then position $t$ can see item $v_{t+1}$ — which is the answer. The model learns nothing. This is information leakage, and it is exactly why nobody had done this before.

The fix is borrowed wholesale from [[BERT- Pre-training of Deep Bidirectional Transformers|BERT]]: the **Cloze** objective. Randomly hide some items in the sequence with a `[mask]` token, and train the model to guess what they were from both sides. No leakage, because the answer is literally deleted from the input.

> [!NOTE] Cloze task
> Delete a random fraction of tokens in a sequence, replace with `[mask]`, and train the model to recover them from the surrounding left and right context. Named after a 1953 readability test. Identical to [[BERT- Pre-training of Deep Bidirectional Transformers#Objective 1: Masked LM|masked language modelling]]. ^cloze-task

There is a bonus. Left-to-right training gives you $n$ training targets from a length-$n$ sequence, and they are the same $n$ every epoch. Cloze masking is random, so across epochs you can draw up to $\binom{n}{k}$ different masked versions of the same sequence. Sparse recommendation data is starved of samples; this is free data augmentation.

The mismatch this creates: training predicts *middle* items, but at serving time you want the *next* item. Their patch is to glue a `[mask]` token onto the end of the sequence at test time and read the prediction from that position. They also mix in training samples where **only the last item is masked**, which acts like fine-tuning for the real task.

## The Methodology

Notation: users $\mathcal{U}$, items $\mathcal{V}$, and for user $u$ a chronological sequence $\mathcal{S}_u = [v_1^{(u)}, \dots, v_{n_u}^{(u)}]$. Goal is $p(v_{n_u+1}^{(u)} = v \mid \mathcal{S}_u)$.

**Architecture.** It is the [[Attention Is All You Need|Transformer]] encoder, essentially unchanged, with $L$ stacked layers.

Input embedding — item embedding plus a *learned* positional embedding (not sinusoidal; learned worked better):
$$\bm{h}_i^0 = \bm{v}_i + \bm{p}_i$$
Because $\bm{P} \in \mathbb{R}^{N \times d}$ is a fixed table, sequences longer than $N$ get truncated to the most recent $N$ items.

Each layer:
$$\bm{A}^{l-1} = \texttt{LN}\big(\bm{H}^{l-1} + \texttt{Dropout}(\texttt{MH}(\bm{H}^{l-1}))\big)$$
$$\bm{H}^{l} = \texttt{LN}\big(\bm{A}^{l-1} + \texttt{Dropout}(\texttt{PFFN}(\bm{A}^{l-1}))\big)$$

Multi-head attention is standard scaled dot-product ([[Query, Key, and Value (QKV)|Q, K, V]] all projected from the same $\bm{H}^l$, so it is [[Attention Is All You Need#^self-attention|self-attention]]):
$$\texttt{Attention}(\bm{Q},\bm{K},\bm{V}) = \mathrm{softmax}\left(\frac{\bm{Q}\bm{K}^\top}{\sqrt{d/h}}\right)\bm{V}$$
**No causal mask.** That is the whole architectural change from SASRec.

The feed-forward block is $d \to 4d \to d$ with GELU in between:
$$\texttt{FFN}(\bm{x}) = \texttt{GELU}(\bm{x}\bm{W}^{(1)} + \bm{b}^{(1)})\bm{W}^{(2)} + \bm{b}^{(2)}, \qquad \texttt{GELU}(x) = x\,\Phi(x)$$
where $\Phi$ is the standard normal CDF. Plus [[Deep Residual Learning for Image Recognition (ResNet)#^residual-block|residual connections]] and layer norm around both sub-layers.

**Output head.** Two-layer feed-forward with GELU, then a softmax over the whole item vocabulary using the *tied* input embedding matrix $\bm{E}$:
$$P(v) = \mathrm{softmax}\big(\texttt{GELU}(\bm{h}_t^L \bm{W}^P + \bm{b}^P)\bm{E}^\top + \bm{b}^O\big)$$
Tying input and output embeddings shrinks the model and reduces overfitting.

**Loss.** Plain [[Cross Entropy|negative log-likelihood]] averaged over the masked positions only:
$$\mathcal{L} = \frac{1}{|\mathcal{S}_u^m|}\sum_{v_m \in \mathcal{S}_u^m} -\log P(v_m = v_m^* \mid \mathcal{S}_u')$$

**What they dropped from BERT.** No next-sentence prediction, no segment embeddings — a user history is one sequence, there is no second sentence. And crucially, **no pre-training**. BERT pre-trains on a giant generic corpus then fine-tunes, because all English text shares grammar. Item catalogues do not share anything across domains, so BERT4Rec is trained end-to-end from scratch on each dataset.

**Training details.** Adam, learning rate 1e-4, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\ell_2$ weight decay 0.01, linear LR decay, gradient clipping at $\ell_2$ norm 5, batch size 256, one GTX 1080 Ti. Weights initialised from truncated normal in $[-0.02, 0.02]$. Default $L=2$, $h=2$, head dimension 32.

**Data.** Amazon Beauty (avg sequence 8.8), Steam (12.4), ML-1m (163.5), ML-20m (144.4). All ratings converted to implicit 1/0, users with fewer than 5 interactions dropped.

**Mask rate $\rho$ was tuned per dataset:** 0.6 for Beauty, 0.4 for Steam, 0.2 for both MovieLens sets. Max length $N = 50$ for Beauty/Steam, $N = 200$ for MovieLens.

**Evaluation.** Leave-one-out: last item is test, second-to-last is validation. The ground-truth item is ranked against **100 popularity-sampled negatives**. Metrics: HR@$k$, [[NDCG]]@$k$, MRR.

## Ablation Studies and Experiments

**Headline numbers (Table 2).** Best baseline is always SASRec. Selected NDCG@10:

| Dataset | GRU4Rec+ | Caser | SASRec | BERT4Rec | gain |
|---|---|---|---|---|---|
| Beauty | 0.1453 | 0.1360 | 0.1633 | **0.1862** | +14.0% |
| Steam | 0.2053 | 0.1484 | 0.2147 | **0.2261** | +5.3% |
| ML-1m | 0.4064 | 0.4268 | 0.4368 | **0.4818** | +10.3% |
| ML-20m | 0.4087 | 0.3062 | 0.4665 | **0.5340** | +14.5% |

Average across everything: +7.24% HR@10, +11.03% NDCG@10, +11.46% MRR over the best baseline. The most dramatic single number is HR@1 on ML-20m: 0.2544 → 0.3440, a 35% jump. Getting the top-1 slot right is where bidirectionality pays most.

**The critical ablation — is it bidirectionality, or is it the Cloze data augmentation?** They separate the two by running BERT4Rec with **exactly one mask per sequence**, which kills the $\binom{n}{k}$ sample-multiplication advantage and leaves only the architectural difference from SASRec ($d = 256$):

| Model | Beauty NDCG@10 | ML-1m NDCG@10 |
|---|---|---|
| SASRec | 0.1633 | 0.4368 |
| BERT4Rec (1 mask) | 0.1769 | 0.4696 |
| BERT4Rec (full) | 0.1862 | 0.4818 |

So bidirectionality alone accounts for roughly two-thirds of the gain, and the multi-mask objective adds the rest. Both components are real, neither is the whole story.

**Component ablations (Table 5, NDCG@10, $d=64$, $L{=}2$, $h{=}2$):**

| Variant | Beauty | Steam | ML-1m | ML-20m |
|---|---|---|---|---|
| default | 0.1832 | 0.2241 | 0.4759 | 0.4513 |
| w/o positional emb | 0.1741 | 0.2060 | **0.2155** ↓ | **0.2867** ↓ |
| w/o PFFN | 0.1803 | 0.2137 | 0.4544 | 0.4296 |
| w/o LayerNorm | 0.1642 ↓ | 0.2058 | 0.4334 | 0.4186 |
| w/o residual | 0.1619 ↓ | 0.2193 | 0.4643 | 0.4483 |
| w/o dropout | 0.1658 | 0.2185 | 0.4553 | 0.4471 |
| $L=1$ | 0.1782 | 0.2122 | 0.4412 | 0.4238 |
| $L=4$ | 0.1834 | 0.2279 | **0.4898** | **0.4732** |
| $h=8$ | 0.1823 | 0.2248 | 0.4743 | 0.4550 |

Readings:
- **Positional embeddings are load-bearing on long sequences.** ML-1m collapses from 0.4759 to 0.2155 — a 55% drop. Without them, every `[mask]` position has an identical representation, so the model is asked to predict different targets from the same vector. The problem is ill-posed. Short-sequence Beauty barely notices.
- **[[Regularization|Regularisers]] (LayerNorm, residual, dropout) matter most on small/sparse data.** Beauty loses 10%+ without LN or RC. On big ML-20m at the default depth they barely help — but rerun at $L=4$ and removing residuals costs ~10% NDCG. Depth is what makes residuals necessary.
- **Depth helps big data, hurts small data.** $L=4$ is best on ML-1m and ML-20m; on Beauty $L=3$ (0.1859) beats $L=4$ (0.1834), which is overfitting.
- **Heads follow the same split.** Beauty prefers $h=1$ (0.1853, better than the default!); ML-20m prefers more heads. More heads help capture long-distance dependencies, which only exist in long sequences.

**Mask proportion $\rho$ (Figure 4).** Non-monotonic, with a clear peak. $\rho = 0.1$ is worse than $\rho = 0.2$ everywhere (too little signal). Above 0.6 performance falls everywhere (too much guessing from too little context). Optimum tracks sequence length: Beauty 0.6, Steam 0.4, MovieLens 0.2. The intuition is absolute count, not fraction — $\rho = 0.6$ on ML-1m means predicting ~98 items per sequence, which is hopeless; on Beauty it means 5.

**Hidden size $d$ (Figure 3).** Performance saturates. Bigger is not better on sparse Beauty and Steam — overfitting. BERT4Rec beats every baseline even at small $d$, and $d \ge 64$ is enough.

**Max length $N$ (Table 4).** Beauty peaks at $N=20$ (0.1875) and slightly degrades at 50. ML-1m peaks around $N=200$. Longer windows bring information *and* noise. Throughput cost is steep: Beauty goes 5504 → 1441 samples/s going from $N=10$ to $N=50$; ML-1m goes 14255 → 1213 from $N=10$ to $N=400$. Attention is $\mathcal{O}(n^2 d)$ per layer.

**Attention visualisation (Figure 2).** Different heads specialise in direction — layer 1 head 1 looks left, head 2 looks right. Layer 2 concentrates on recent items, because it feeds the output. Some heads attend heavily to the `[mask]` token itself, which the authors guess is a way to route a sequence-level summary back down to item level. Most importantly, attention genuinely spreads to both sides, which is the direct evidence that the bidirectional capacity is being used.

## Worth Remembering

**The evaluation protocol is a serious caveat.** Ranking the true item against only 100 popularity-sampled negatives is standard in this literature but is now known to be unreliable — sampled-metric rankings do not always match full-catalogue rankings (Krichene & Rendle, KDD 2020). Compare with [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|the reproducibility critique]]: BERT4Rec in particular has been the subject of a well-known replication paper (Petrov & Macdonald, RecSys 2022) which found that the original numbers are hard to reproduce, that the reference implementation was **badly under-trained** relative to what it needs, and that a properly-trained SASRec with a full softmax cross-entropy loss closes much or all of the gap. Read the headline table with that in mind. The bidirectionality result may hold; the size of the margin probably does not.

**No pre-training is the honest structural difference from BERT.** The whole point of BERT is transfer from a huge generic corpus. Recommendation has no such shared substrate — item IDs in Amazon Beauty mean nothing in Steam. So BERT4Rec gets the architecture and the objective but not the thing that made BERT famous. It is essentially "a masked-LM-style encoder trained from scratch on one dataset". Later work (item-text-based recommenders, ID-agnostic models) attacked exactly this gap.

**The train/test mismatch is a real wart, papered over.** Training predicts interior items; serving predicts the future. Appending `[mask]` at the end plus mixing in last-item-only training samples is a heuristic patch, not a principled fix. The paper never quantifies how much the "only mask the last item" samples contribute — an ablation that is conspicuously absent.

**Practical rules of thumb that generalise:**
- Tune $\rho$ by *how many items get masked*, not by the fraction. Aim for a handful of targets per sequence, not a hundred.
- Positional embeddings are non-optional whenever sequences are long, and this is more severe here than in NLP because the masked-position representation is otherwise degenerate.
- On sparse catalogues, go shallow, few heads, small $d$, heavy dropout. On dense ones, go deep.

**Limitations the authors name:** item IDs only — no category, price, cast, or other side features. And no explicit user representation, so multi-session users are handled implicitly through their concatenated history.

**Open questions:** the $\mathcal{O}(n^2)$ cost is dismissed as "GPUs parallelise well", which is fine at $N=200$ but not for a real e-commerce history of thousands of events. [[Flash Attention]] or a linear-attention variant would be the obvious modern move. Also: the CBOW/Skip-Gram connection they draw is elegant — Cloze with one central mask, uniform attention, one layer, no positions *is* CBOW — which frames masked prediction as the general form of the word2vec objectives.

## Links

Related: [[Attention Is All You Need]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Causal Attention]] · [[Query, Key, and Value (QKV)]] · [[Recommender Systems - Evolution]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[NDCG]] · [[Cross Entropy]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Regularization]] · [[Foundational_RecSys_Ranking_Reading_Plan]] · [[Auto-regressive models]] · [[Seq2Seq models]] · [[Markov Chain Monte Carlo]] · [[Linear Projection]]

New topics worth writing: SASRec (Self-Attentive Sequential Recommendation), GELU activation, Layer Normalization, Leave-one-out evaluation with sampled negatives, Sampled metrics considered harmful (Krichene & Rendle), BERT4Rec replicability (Petrov & Macdonald 2022), GRU4Rec and session-based recommendation, Caser (convolutional sequence embedding), FPMC and factorized Markov chains, BPR pairwise ranking loss, CBOW and Skip-Gram, Weight tying in embedding layers, Mean Reciprocal Rank
