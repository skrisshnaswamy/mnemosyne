---
title: "Self-Attentive Sequential Recommendation (SASRec)"
authors: ["Wang-Cheng Kang", "Julian McAuley"]
year: 2018
arxiv: "1808.09781"
url: https://arxiv.org/abs/1808.09781
priority: Must-Read
read_on: 2026-08-25
tags: [paper, transformers, llm, vision]
---
## The Core Idea

Sequential recommendation asks: given the ordered list of things a user already touched, what do they touch next? Before 2018 there were two camps.

**Markov Chains (MC).** Assume the next item depends only on the last one (or last few). Very few parameters. Works surprisingly well when data is sparse — most users have only a handful of actions, so a simple model is all you can fit.

**RNNs.** Squash the whole history into a hidden state. In principle they see everything. In practice they need lots of dense data before they beat the simple models, and they are slow because step $t$ must wait for step $t-1$.

SASRec takes the [[Attention Is All You Need|Transformer]] decoder and points it at item sequences. The bet: [[Causal Attention|causal self-attention]] gets you both ends of the trade-off at once. It *can* look at all $n$ previous items (like an RNN), but because attention is a soft selection, it *chooses* to put nearly all its weight on the last one or two items when the data is sparse (like a Markov Chain). Nobody has to pick the Markov order $L$ up front — the model picks it, per dataset, per time step, and the attention heatmaps prove it does.

That is the whole insight, and the paper's most convincing figure is the one showing attention on *Beauty* (sparse, 7.6 actions/user) hugging the diagonal, while on *MovieLens-1M* (dense, 163.5 actions/user) it spreads far back.

> [!NOTE] Sequential recommendation
> Predict the next item from the *order* of past items. Distinct from temporal recommendation, which models actual timestamps and drift ("what do people watch at 4pm"). SASRec throws timestamps away and keeps only rank order. ^sequential-recommendation

It also runs ~11× faster per epoch than the CNN baseline and ~18× faster than the RNN one, because attention over a length-$n$ sequence is one big matmul instead of $n$ dependent steps.

## The Methodology

**Input.** Each user's sequence $\mathcal{S}^u$ is truncated or left-padded to fixed length $n$. Padding items get a constant zero embedding. Input is $(\mathcal{S}^u_1,\dots,\mathcal{S}^u_{|\mathcal{S}^u|-1})$, target is the same sequence shifted by one — exactly the [[Auto-regressive models|autoregressive]] setup of a language model, with items instead of tokens.

**Embedding layer.** Item embedding matrix $\mathbf{M} \in \mathbb{R}^{|\mathcal{I}|\times d}$, plus a **learned** positional embedding $\mathbf{P}\in\mathbb{R}^{n\times d}$:

$$\widehat{\mathbf{E}}_i = \mathbf{M}_{s_i} + \mathbf{P}_i$$

They tried the fixed sinusoidal positions from the original Transformer and it was *worse*. Dropout is applied on $\widehat{\mathbf{E}}$.

**Self-attention block.** Standard scaled dot-product, with [[Query, Key, and Value (QKV)|Q/K/V]] all coming from the same input via three [[Linear Projection|linear projections]] $\mathbf{W}^Q,\mathbf{W}^K,\mathbf{W}^V\in\mathbb{R}^{d\times d}$:

$$\mathbf{S} = \text{softmax}\!\left(\frac{\widehat{\mathbf{E}}\mathbf{W}^Q(\widehat{\mathbf{E}}\mathbf{W}^K)^T}{\sqrt{d}}\right)\widehat{\mathbf{E}}\mathbf{W}^V$$

Attention from query $i$ to key $j$ is **masked off for $j > i$** — position $t$ must not see the future, or the task is trivial.

**Feed-forward.** Attention is a weighted *sum*, so it is linear. A two-layer point-wise FFN adds the nonlinearity:

$$\mathbf{F}_i = \text{ReLU}(\mathbf{S}_i\mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)}$$

Applied independently per position, so no leakage.

**Stacking.** $b=2$ blocks by default. Each sublayer is wrapped **pre-norm** style:

$$g(x) = x + \text{Dropout}\big(g(\text{LayerNorm}(x))\big)$$

The [[Deep Residual Learning for Image Recognition (ResNet)#^residual-block|residual connection]] matters for a specific recommender reason: after two attention blocks the last item's embedding is smeared together with everything else, but the last item is the single strongest predictor. The skip path carries it straight to the top.

**Prediction.** Score of item $i$ at step $t$ is a dot product with the **shared** item embedding matrix:

$$r_{i,t} = \mathbf{F}^{(b)}_t \mathbf{M}_i^T$$

Reusing $\mathbf{M}$ instead of a second matrix $\mathbf{N}$ is a big win (see ablations). The usual worry — a symmetric inner product cannot express "$i$ often follows $j$ but not the reverse" — does not apply, because $\text{FFN}(\mathbf{M}_i)\mathbf{M}_j^T \neq \text{FFN}(\mathbf{M}_j)\mathbf{M}_i^T$. The nonlinearity supplies the asymmetry.

**Loss.** Binary [[Cross Entropy|cross-entropy]] with **one** sampled negative per position per epoch:

$$-\sum_{\mathcal{S}^u}\sum_{t}\Big[\log \sigma(r_{o_t,t}) + \sum_{j\notin\mathcal{S}^u}\log(1-\sigma(r_{j,t}))\Big]$$

Padded positions are skipped. Note this is a pointwise sigmoid loss over sampled negatives, close in spirit to [[Distributed Representations of Words and Phrases (negative sampling)|negative sampling]] and cousin to [[BPR- Bayesian Personalized Ranking from Implicit Feedback|BPR]]'s pairwise objective.

**Hyperparameters that mattered.** Adam, lr $0.001$, batch 128. Dropout $0.2$ for ML-1M, $0.5$ for the three sparse datasets. $n=200$ for ML-1M, $n=50$ elsewhere — roughly the mean actions per user. $d=50$; performance saturates around $d\geq 40$.

**Complexity.** Parameters $O(|\mathcal{I}|d + nd + d^2)$ — crucially **no per-user parameters**, unlike FPMC's $O(|\mathcal{U}|d + |\mathcal{I}|d)$. Time $O(n^2 d + nd^2)$, dominated by $n^2d$, but fully parallel. Max path length from any input to any output is $O(1)$ versus $O(n)$ for an [[Long Short-Term Memory (Neural Computation)|RNN]].

**Reductions.** Zero out the attention blocks, unshare the item embeddings, drop positions → you get Factorized Markov Chains. Add an explicit user embedding by concatenation → FPMC. Use one attention layer with *uniform* weights → FISM. So SASRec is literally "FISM with learned, position-aware, hierarchical weights".

## Ablation Studies and Experiments

**Datasets.** Amazon Beauty (0.4M actions, 7.6/user), Amazon Games (0.3M, 9.3/user), Steam (3.7M, 11.0/user — new dataset, crawled by the authors), MovieLens-1M (1.0M, 163.5/user). 5-core filtering. Leave-one-out split: last action test, second-to-last validation.

**Evaluation.** Hit@10 and [[NDCG|NDCG@10]], ranking the true item against **100 randomly sampled negatives**. (This sampled-negative protocol was later shown to be unreliable — see below.)

**Headline results (NDCG@10):**

| Dataset | FPMC | TransRec | GRU4Rec⁺ | Caser | **SASRec** |
|---|---|---|---|---|---|
| Beauty | 0.2891 | 0.3020 | 0.2556 | 0.2547 | **0.3219** |
| Games | 0.4680 | 0.4557 | 0.4759 | 0.3214 | **0.5360** |
| Steam | 0.5011 | 0.4852 | 0.5595 | 0.5381 | **0.6306** |
| ML-1M | 0.5176 | 0.3969 | 0.5513 | 0.5538 | **0.5905** |

Average gains: +6.9% Hit, +9.6% NDCG over the strongest baseline. The clean pattern in the baseline columns is the paper's motivating story made concrete: on sparse Beauty the *non-neural* TransRec (0.3020) beats every neural method; on dense ML-1M the neural Caser (0.5538) beats FPMC. SASRec wins both.

**Ablations (NDCG@10, $d=50$; default row is 0.3142 / 0.5360 / 0.6306 / 0.5905):**

| Variant | Beauty | Games | Steam | ML-1M |
|---|---|---|---|---|
| Remove positional emb. | **0.3183** | 0.5301 | 0.6036 | 0.5772 |
| Unshared item emb. | 0.2437 ↓ | 0.4266 ↓ | 0.4472 ↓ | 0.4557 ↓ |
| Remove residual conn. | 0.2591 ↓ | 0.4303 ↓ | 0.5693 | 0.5535 |
| Remove dropout | 0.2436 ↓ | 0.4375 ↓ | 0.5959 | 0.5801 |
| $b=0$ blocks | 0.2620 ↓ | 0.4745 ↓ | 0.5588 ↓ | 0.4830 ↓ |
| $b=1$ | 0.3066 | **0.5408** | 0.6202 | 0.5653 |
| $b=3$ | 0.3078 | 0.5312 | 0.6275 | **0.5931** |
| Multi-head (2 heads) | 0.3080 | 0.5311 | 0.6272 | 0.5885 |

What actually does the work:

- **Sharing the item embedding is the single biggest lever.** Unsharing costs 20–29% NDCG on every dataset. Two embedding tables overfit; one does not.
- **Residual connections and dropout are load-bearing on sparse data only.** Removing either costs ~18% on Beauty/Games but under 6% on Steam/ML-1M. Sparse data = overfitting risk = [[Regularization|regularisation]] matters.
- **Depth pays off only when data is dense.** One block is nearly as good as two on Beauty and actually *better* on Games; two blocks help on Steam and ML-1M.

**Things that did not work:**

1. **Fixed sinusoidal positional embeddings** — worse than learned ones. (Later work like [[RoFormer- Enhanced Transformer with Rotary Position Embedding|RoPE]] revisits this question for language.)
2. **Multi-head attention** — consistently and slightly *worse* than a single head, on all four datasets. Their explanation: $d=50$ is too small to carve into subspaces, unlike the Transformer's $d=512$.
3. **Explicit user embedding** ($r_{u,i,t}=(\mathbf{U}_u+\mathbf{F}^{(b)}_t)\mathbf{M}_i^T$) — no improvement. The sequence already identifies the user.
4. **Removing positional embeddings** — mostly hurts, but *helps* on Beauty (0.3183 vs 0.3142), the sparsest set. When sequences are ~8 items long, order carries little signal and the position table is just parameters to overfit.

**Efficiency (ML-1M, single GTX-1080 Ti).** 1.7 s/epoch versus Caser 19.1 s and GRU4Rec⁺ 30.7 s. Converges in ~350 s total.

**Scalability in $n$ (ML-1M).** NDCG@10 climbs 0.480 ($n=10$) → 0.587 ($n=200$) → 0.596 ($n=500$), then flattens at $n=600$ (0.595), since 99.8% of actions are covered by then. Training time grows roughly quadratically: 75 s at $n=10$, 1895 s at $n=600$.

**Attention visualisations.** Four heatmaps of average attention over the last 15 positions:
- Beauty layer 1 hugs the diagonal (recent items); ML-1M layer 1 spreads back. This is the adaptivity claim, shown directly.
- ML-1M *without* positional embeddings gives near-uniform attention — position embeddings are what make attention recency-biased.
- ML-1M layer 2 is more recency-focused than layer 1: the first block has already aggregated distant history, so the second does not need to reach back.
- Attention between 200 query movies and 200 key movies from Sci-Fi / Romance / Animation / Horror forms an approximately **block-diagonal** matrix. The model discovers genre similarity without ever seeing genre labels.

## Worth Remembering

- **The evaluation protocol is the paper's weakest point in hindsight.** Ranking against 100 sampled negatives inflates and distorts metrics; sampled-metric bias was documented later (Krichene & Rendle 2020), and the RecSys reproducibility literature ([[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]]) shows how easily tuned simple baselines close these gaps. Any comparison you run against these exact table numbers is on shaky ground unless you replicate the protocol.
- **Admitted limitations.** $O(n^2 d)$ means it cannot handle very long click streams. Their proposed fixes — restricted/local attention, or chunking sequences — are exactly what later industrial systems ([[Actions Speak Louder than Words- Generative Recommenders (HSTU)|HSTU]]) had to solve properly.
- **No side features.** Pure item-ID sequences. No dwell time, action type, device, or category. The conclusion flags this as future work; production systems (DIN, PinnerFormer, HSTU) all had to add it back.
- **One negative sample per position per epoch** is very cheap. It works here because each position in each sequence generates a training example, so the effective number of (positive, negative) pairs is large. Contrast with the full-softmax and [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)|logQ-corrected]] approaches used in retrieval at scale.
- **The multi-head result is a genuinely useful negative.** At small $d$, splitting into heads costs you. If you are running a recommender at $d=64$, do not assume 8 heads is free.
- **SASRec is left-to-right; [[BERT4Rec- Sequential Recommendation with Bidirectional Transformer|BERT4Rec]] is the masked-LM counterpart** published a year later, arguing bidirectional context helps. Worth reading the two together, and worth knowing that later replications found well-tuned SASRec often matches or beats BERT4Rec.
- **Practical caveat:** the ablation table's default Beauty number (0.3142) does not match the main results table (0.3219). Small, but a reminder that these numbers carry run-to-run variance the paper does not report error bars for.
- **Open question:** the residual connection is justified as "carry the last item's embedding to the top". That suggests an explicit last-item skip might work as well or better than a full residual stack on sparse data. Nobody tested it here.

## Links

Related: [[Attention Is All You Need]] · [[Causal Attention]] · [[Query, Key, and Value (QKV)]] · [[BERT4Rec- Sequential Recommendation with Bidirectional Transformer]] · [[Actions Speak Louder than Words- Generative Recommenders (HSTU)]] · [[BPR- Bayesian Personalized Ranking from Implicit Feedback]] · [[Collaborative Filtering for Implicit Feedback Datasets (ICDM)]] · [[Factorization Machines (ICDM)]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Deep Interest Network for CTR Prediction (DIN)]] · [[PinnerFormer- Sequence Modeling for User Representation at Pinterest]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[Cross Entropy]] · [[NDCG]] · [[Regularization]] · [[Recommender Systems - Evolution]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Auto-regressive models]] · [[RoFormer- Enhanced Transformer with Rotary Position Embedding]] · [[Foundational_RecSys_Ranking_Reading_Plan]]

New topics worth writing: Layer Normalization, Dropout, GRU4Rec and session-based recommendation, Caser and convolutional sequence embedding, FPMC and factorized Markov chains, FISM item similarity models, TransRec translation-based recommendation, sampled-metric bias in top-N evaluation, leave-one-out splitting for recommenders, weight tying in embeddings
