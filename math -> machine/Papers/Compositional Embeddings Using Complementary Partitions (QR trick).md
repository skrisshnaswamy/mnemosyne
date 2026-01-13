---
title: "Compositional Embeddings Using Complementary Partitions (QR trick)"
authors: ["Hao-Jun Michael Shi", "Dheevatsa Mudigere", "Maxim Naumov", "Jiyan Yang"]
year: 2020
arxiv: "1909.02107"
url: https://arxiv.org/abs/1909.02107
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, vision]
---
## The Core Idea

Recommendation models keep one big lookup table per categorical feature. One row per category, $D$ numbers wide. When a feature has $10^7$ categories and $D \approx 100$, that single table is gigabytes. In [[Deep Learning Recommendation Model (DLRM)]] these tables are the whole memory bill — the baseline model here has $5.4 \times 10^8$ parameters, almost all of it embeddings.

The standard fix is the **hashing trick**: pick a small table of $m$ rows and send category $i$ to row $i \bmod m$. Cheap, but brutal — if $m = |S|/4$, then four different categories share one row exactly. "Nike shoes" and "lawn mower" get the same vector, forever, and the model cannot tell them apart.

The insight here is embarrassingly simple once you see it. Division gives you *two* numbers, not one. If you keep both the remainder $i \bmod m$ **and** the quotient $i \setminus m$, you have thrown nothing away — the pair $(i \bmod m,\; i \setminus m)$ reconstructs $i$ exactly. So keep two small tables, look up one row in each, and combine them (element-wise multiply). Category A and category B may collide in the remainder table, and some other pair may collide in the quotient table, but **no two categories collide in both**. Every category still gets its own distinct final vector, and you never stored $|S|$ rows.

Memory goes from $O(|S|D)$ to $O\!\left(\left(m + \tfrac{|S|}{m}\right)D\right)$, which is minimised at $m = \sqrt{|S|}$, giving $O(\sqrt{|S|}\,D)$.

> [!NOTE] Quotient-remainder trick
> Store two embedding tables of size $m$ and $|S|/m$. Index the first by $i \bmod m$, the second by $i \setminus m$, and multiply the two vectors element-wise. Unique vector per category, square-root memory. ^qr-trick

The generalisation is the real contribution. Any family of set partitions works, as long as it is **complementary**.

> [!NOTE] Complementary partitions
> Partitions $P_1, \dots, P_k$ of the category set $S$ are complementary if for every pair $a \neq b$ there is at least one $i$ where $a$ and $b$ land in different buckets of $P_i$. Each partition becomes one embedding table; the buckets are the rows. ^complementary-partitions

With $k$ complementary partitions of roughly equal size, memory is $O(k|S|^{1/k}D)$.

Why did this not exist before? Prior compression work (deep compositional code learning, product-quantisation-style codebooks — see [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)]]) *learns* the codes that map a category to its sub-embeddings. Learning codes means storing them: $O(|S|D)$ again, which defeats the point when $|S| \gg D$. The trick here is that the code is **fixed and free** — it is arithmetic on the index, computed on the fly, zero storage, zero pre- or post-training steps.

## The Methodology

**The hashing trick, formally.** With table $\widetilde{\bm{W}} \in \mathbb{R}^{m \times D}$ and hash matrix $\bm{R} \in \mathbb{R}^{m \times |S|}$ where $\bm{R}_{i,j} = 1$ iff $j \bmod m = i$:

$$\bm{x}_{\text{emb}} = \widetilde{\bm{W}}^T \bm{R} \bm{e}_i$$

**The QR trick.** Add a second table $\bm{W}_2 \in \mathbb{R}^{(|S|/m) \times D}$ and a quotient matrix $\bm{Q}$ with $\bm{Q}_{i,j} = 1$ iff $j \setminus m = i$:

$$\bm{x}_{\text{emb}} = \bm{W}_1^T \bm{R} \bm{e}_i \;\odot\; \bm{W}_2^T \bm{Q} \bm{e}_i$$

In code it is four lines: compute `j = i % m`, `k = i // m`, two row lookups, one multiply.

**Four families of complementary partitions** they list (all proved in the appendix):

1. **Naive** — every category its own bucket. That is just the full table.
2. **Quotient-remainder** — the $k=2$ case above.
3. **Generalized quotient-remainder** — pick $m_1, \dots, m_k$ with $\prod_i m_i \geq |S|$, and use $P_j$ indexed by $\left(i \setminus M_j\right) \bmod m_j$ where $M_j = \prod_{i<j} m_i$. This is just writing $i$ in mixed radix.
4. **Chinese remainder** — pick pairwise coprime $m_1, \dots, m_k$ with $\prod m_i \geq |S|$ and use $i \bmod m_j$ for each. Uniqueness follows straight from the Chinese Remainder Theorem.

**Combining the $k$ vectors.** Three operation functions $\omega$ tested:

- **Concatenation**: $D = \sum_j D_j$. This is the only one with a proof — Theorem 1 says if the rows within each table are distinct, concatenation gives a distinct final vector per category.
- **Addition**: all $D_j = D$, sum them.
- **Element-wise multiplication**: all $D_j = D$, $\odot$ them.

Note the gap: multiplication and addition are *not* proved injective (two different pairs can multiply to the same vector), yet multiplication wins empirically.

A fourth option, **feature generation**, is the baseline where you skip combining entirely and just feed the $k$ small embeddings into the network as $k$ separate sparse features. Effective, but it adds parameters downstream and ignores the fact that all $k$ came from one original feature.

**Path-based compositional embeddings** (the more speculative variant). Use partition $P_1$ for a real embedding table, then use each *other* partition to select a **transformation** to apply:

$$\bm{x}_{\text{emb}} = \left(M_{k, p_k(x)} \circ \dots \circ M_{2, p_2(x)}\right)\!\left(\bm{W}\bm{e}_{p_1(x)}\right)$$

Each $M_{j,i}$ is a linear map or a small MLP, learned jointly. Every category walks a unique *path* of transforms. Downside: you now have non-embedding parameters entangled in the lookup, which makes training harder.

**Setup.** Criteo Ad Kaggle: ~45M rows, 7 days, 13 dense + 26 categorical features. Days 1–6 train, day 7 split into val/test. Dense features log-transformed. Two architectures:

- **DCN** ([[Deep & Cross Network for Ad Click Predictions]]): deep net 512–256–64, cross net 6 layers, $D = 16$.
- **DLRM**: bottom MLP 512–256–64, top MLP 512–256, $D = 16$.

Adagrad and AMSGrad at default hyperparameters, whichever gives better validation loss. Batch size 128, **single epoch**, no regularisation, 5 trials, loss is binary [[Cross Entropy]].

"Number of hash collisions" is their knob: 4 collisions means $m = |S|/4$, roughly a $4\times$ smaller model.

## Ablation Studies and Experiments

**Baseline behaviour (Figure 4).** At 4 collisions the QR trick's validation curve sits cleanly *between* the hashing trick and the full table. Exactly the interpolation you would hope for, and the standard deviation bands over 5 trials do not overlap with hashing.

**Operation sweep (Figure 5), 2–7 and 60 collisions.**

- **Element-wise multiplication wins overall.** It tracks the feature-generation baseline closely on DLRM — and feature generation costs an extra half-million parameters. On DCN it beats every other operation by a clear margin.
- At **4 collisions**, QR-mult is within **0.3%** of the full-table baseline on DCN and **0.7%** on DLRM.
- At **60 collisions** — a $\sim 15\times$ smaller model — QR-mult matches or beats the *hashing trick at 4 collisions*. That is the headline compression result.
- Performance decays roughly exponentially as parameters shrink, for every method.

**A hyperparameter interaction worth flagging:** AMSGrad significantly outperformed Adagrad *specifically when using the multiplication operation*. The optimiser choice and the composition operation are not independent. If you try $\odot$ with plain Adagrad you may conclude it does not work. (See [[Adam- A Method for Stochastic Optimization]] for why AMSGrad exists.)

**Thresholding ablation (Figures 6–7).** Categorical features vary wildly in cardinality; compressing a 20-category feature is pointless. They only compress tables larger than a threshold $\in \{1, 20, 200, 2000, 20000\}$, at 4 collisions. Here the story fragments:

- **Multiplication is best for DCN. Concatenation is best for DLRM.** The winning operation depends on the surrounding architecture, not on some intrinsic property of the operation.
- With thresholding, DLRM's gap to baseline improves from **0.7% → 0.5%** while still holding a $4\times$ reduction.

**What did not work: path-based embeddings.** Table 1, 4 collisions, single-hidden-layer MLP:

| Hidden | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| DCN test loss | 0.45263 | 0.45254 | **0.45252** | 0.4534 |
| DLRM test loss | 0.45349 | 0.45312 | **0.45306** | 0.45651 |

Hidden size 64 is the sweet spot, and 128 gets *worse* on both — too many extra parameters to learn on a single epoch. But crucially, **path-based never beat operation-based compositional embeddings**. The authors call these preliminary and say the idea deserves more work, but as shipped it is strictly worse and strictly more compute-heavy.

**What the ablations reveal.** The work is being done by *uniqueness of representation*, not by any clever structure. The partitions are arbitrary arithmetic on an index that has no semantic meaning — category 7 and category 8 are unrelated. All the QR trick buys is that no two categories are forced to share the same vector. That alone recovers most of the gap to the full table. The authors are honest that model quality "ought to depend on how closely the chosen partitions reflect intrinsic properties of the category set," and here they reflect nothing at all.

## Worth Remembering

- **Practical recipe the authors themselves recommend**: threshold (only compress large tables) + QR trick + element-wise multiplication. Try concatenation too, because the winner flipped between DCN and DLRM. This shipped into the open-source DLRM repo.
- **It composes with existing hashing.** In production you often already hash raw strings to indices online. QR replaces the *second* remainder step (the size reduction), not the first.
- **Admitted weakness: frequency-blind.** A category seen 10 million times and one seen twice get the same treatment. Real CTR data is brutally long-tailed. Frequency-aware or entropy-aware sizing (Naumov's work on embedding dimension) is orthogonal and probably complementary.
- **Admitted weakness: no learned structure.** Codebook-learning methods would in principle find better decompositions, but storing the codes costs $O(|S|D)$, which is the thing you were trying to avoid. This is the core tension.
- **Uniqueness is not the same as separability.** Concatenation is the only operation with a uniqueness proof, yet multiplication works better. Two different $(\bm{a}, \bm{b})$ pairs can have $\bm{a} \odot \bm{b}$ identical, and multiplication kills a dimension whenever either factor is near zero. The theory does not explain the empirics here.
- **Single-epoch training** is standard for Criteo but means these numbers say little about how compositional embeddings behave under long training, where the smaller tables might overfit differently.
- **Compute goes up.** Two lookups instead of one, plus a multiply, per categorical feature, times 26 features. Memory-bound systems will happily take that trade; latency-bound ones should measure.
- **Connection worth chasing**: this is a fixed, non-learned analogue of low-rank factorisation ([[LoRA- Low-Rank Adaptation of Large Language Models]]) and of Tensor Train embedding compression. The Khrulkov et al. Tensor Train work is a *specific* operation in this framework — worth reading side by side.
- **Open question**: could the partitions be built from real metadata instead of arithmetic? The paper's own car example (year / make / type) is exactly this, and would be complementary if the triple uniquely identifies a car. Nobody tested it because Criteo categories are anonymised hashes.

## Links

Related: [[Deep Learning Recommendation Model (DLRM)]] · [[Deep & Cross Network for Ad Click Predictions]] · [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)]] · [[Factorization Machines (ICDM)]] · [[Wide & Deep Learning for Recommender Systems]] · [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] · [[Ad Click Prediction- a View from the Trenches (KDD)]] · [[Matryoshka Representation Learning]] · [[Adam- A Method for Stochastic Optimization]] · [[Cross Entropy]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[Recommender Systems - Evolution]]

New topics worth writing: Feature hashing / the hashing trick (Weinberger et al.), Chinese Remainder Theorem, Tensor Train decomposition for embedding compression, Adagrad, AMSGrad and the Adam convergence failure, Deep compositional code learning, Criteo CTR benchmark, Embedding dimension selection from feature entropy
