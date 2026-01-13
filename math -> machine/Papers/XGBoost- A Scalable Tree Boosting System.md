---
title: "XGBoost: A Scalable Tree Boosting System"
authors: ["Tianqi Chen", "Carlos Guestrin"]
year: 2016
arxiv: "1603.02754"
url: https://arxiv.org/abs/1603.02754
priority: Must-Read
read_on: 2026-08-25
tags: [paper]
---
## The Core Idea

Gradient boosted trees already existed (Friedman, 2001). XGBoost is not a new learning algorithm so much as a **new implementation of an old one, engineered until it was ten times faster and could eat a terabyte of data on one desktop**. That sounds like an incremental contribution. It was not: the speed changed how people work. If a model trains in one minute instead of thirty, you try thirty ideas instead of one. In 2015, 17 of the 29 winning Kaggle solutions used it; the runner-up method, deep nets, appeared in 11.

Four ideas do the real work.

1. **A regularised, second-order objective.** Instead of fitting each new tree to the residual (first-order only), XGBoost writes a second-order Taylor expansion of *any* twice-differentiable loss and gets a closed form for the best leaf value and for the split gain. Tree structure is then chosen by an actual objective, not by a hand-picked impurity heuristic like Gini.
2. **Sparsity-aware splitting.** Every node stores a *default direction*. Missing values (and zeros from one-hot encoding) go that way, and which way is *learnt*. The algorithm only ever touches non-missing entries, so cost is linear in the number of non-zeros. On a one-hot-heavy dataset this was 50× faster than the naive version.
3. **A weighted quantile sketch with a proof.** For approximate split finding you need candidate thresholds at percentiles of the feature. But the second-order derivation says each row carries weight $h_i$, so you need percentiles of a *weighted* distribution. No such streaming sketch existed. They built one with `merge` and `prune` operations that keep the same $\epsilon$ guarantee as the classic unweighted GK sketch.
4. **Systems work nobody in ML had bothered with**: a pre-sorted column block layout, cache-aware prefetching, block compression, and sharding across disks.

The lasting lesson is the last one. The paper's real claim is that **cache misses and disk throughput are machine learning problems**, not someone else's problem.

## The Methodology

**The model.** $K$ additive regression trees (CART — trees with a real number, not a class, on each leaf):

$$\hat{y}_i = \sum_{k=1}^{K} f_k(\mathbf{x}_i), \qquad f_k(\mathbf{x}) = w_{q(\mathbf{x})}$$

$q$ maps a row to a leaf index, $w \in \mathbb{R}^T$ holds the $T$ leaf scores.

**The objective.**

$$\mathcal{L} = \sum_i l(\hat{y}_i, y_i) + \sum_k \Omega(f_k), \qquad \Omega(f) = \gamma T + \tfrac{1}{2}\lambda\|w\|^2$$

$\gamma$ charges you per leaf; $\lambda$ is L2 on the leaf values. Set both to zero and you are back at plain gradient boosting. This is [[Regularization]] baked into the split criterion rather than bolted on as post-hoc pruning.

**The second-order trick.** Trees cannot be optimised by [[Backpropagation]] — the parameters are a discrete structure. So build greedily, one tree at a time. At round $t$, expand the loss around the current prediction:

$$\tilde{\mathcal{L}}^{(t)} = \sum_{i=1}^{n}\left[g_i f_t(\mathbf{x}_i) + \tfrac{1}{2} h_i f_t^2(\mathbf{x}_i)\right] + \Omega(f_t)$$

with $g_i = \partial_{\hat{y}} l$ and $h_i = \partial^2_{\hat{y}} l$ — the [[Derivative#Gradient|gradient]] and the diagonal of the [[Derivative#Hessian|Hessian]], per row. Because the objective is now quadratic in the leaf value, group rows by leaf ($I_j$) and solve exactly:

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}, \qquad \tilde{\mathcal{L}}^{(t)}(q) = -\frac{1}{2}\sum_{j=1}^{T} \frac{\left(\sum_{i \in I_j} g_i\right)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T$$

> [!NOTE] Structure score
> $-\frac12 \sum_j G_j^2/(H_j+\lambda) + \gamma T$ is a quality score for a whole tree shape, derived from the loss rather than assumed. To evaluate a split you only need two sums per candidate node: $G = \sum g_i$ and $H = \sum h_i$. ^structure-score

The split gain is the score before minus the score after, less the per-leaf charge:

$$\mathcal{L}_{split} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$$

If this is negative, do not split. That is the pruning rule, and it falls out of the maths.

**Two more anti-overfit knobs.** *Shrinkage*: scale every new tree by $\eta$ (they use 0.1) — a learning rate for function space, leaving room for later trees. *Column subsampling*: borrowed from Random Forests, sample features per tree. Users reported it helped more than row subsampling, and it also speeds up the parallel split search.

**Split finding, exact.** Sort each feature once, scan left to right accumulating $G_L, H_L$, take $G_R = G - G_L$. Cost per level is one linear pass per column.

**Split finding, approximate.** Sorting all rows is impossible out-of-core or across machines. Propose $\approx 1/\epsilon$ candidate thresholds from feature percentiles, bucket the rows, sum $g$ and $h$ per bucket, pick the best bucket boundary. Two variants: *global* (propose once per tree) and *local* (re-propose after every split).

**Why the sketch must be weighted.** Rewrite the round-$t$ objective as

$$\sum_{i=1}^{n} \tfrac{1}{2} h_i \left(f_t(\mathbf{x}_i) - g_i/h_i\right)^2 + \Omega(f_t) + \text{const}$$

That is weighted least squares with targets $g_i/h_i$ and weights $h_i$. So the "percentiles" that matter are percentiles of the $h$-mass, not of row count. Their rank function is

$$r_k(z) = \frac{\sum_{(x,h)\in\mathcal{D}_k,\, x<z} h}{\sum_{(x,h)\in\mathcal{D}_k} h}$$

and they want candidates with $|r_k(s_{k,j}) - r_k(s_{k,j+1})| < \epsilon$. The appendix builds a summary storing $(x_i, \tilde r^-, \tilde r^+, \tilde\omega)$ — four numbers per retained point — where `merge` of an $\epsilon_1$- and an $\epsilon_2$-summary is $\max(\epsilon_1,\epsilon_2)$-approximate, and `prune` to $b+1$ points costs you $+1/b$ error. Those two guarantees are what make it distributable.

> [!NOTE] Default direction
> Each node stores which branch missing values take. The algorithm tries "all missing go left" and "all missing go right" by scanning the non-missing entries once ascending and once descending, and keeps whichever scores higher. Complexity becomes $O(\|\mathbf{x}\|_0)$, not $O(nm)$. ^default-direction

**Systems layer.**

- **Column block (CSC).** Store the data once, per-column, pre-sorted by feature value, with row indices. Sorting is the expensive part of tree learning and now it happens exactly once. Exact-greedy complexity drops from $O(Kd\|\mathbf{x}\|_0 \log n)$ to $O(Kd\|\mathbf{x}\|_0 + \|\mathbf{x}\|_0 \log n)$ — the $\log n$ becomes a one-time amortised cost. Approximate drops from $O(Kd\|\mathbf{x}\|_0 \log q)$ to $O(Kd\|\mathbf{x}\|_0 + \|\mathbf{x}\|_0 \log B)$. Columns are independent, so split finding parallelises across threads.
- **Cache-aware prefetch.** Scanning a sorted column means fetching $g_i, h_i$ by row index — random memory access. The naive loop has an immediate read/write dependency and stalls on cache misses. Fix: per-thread buffer, prefetch a mini-batch of gradient stats, then accumulate.
- **Block size $2^{16}$ rows.** Too small starves the threads; too big blows the CPU cache.
- **Out-of-core.** Blocks on disk, a prefetch thread per disk. Compress blocks by column (26–29% ratio) and decompress on the fly — trading CPU for disk bandwidth. Row indices stored as 16-bit offsets from the block start, which is exactly why $2^{16}$ per block.

**Common experimental settings:** max depth 8, shrinkage 0.1, no column subsampling unless stated.

## Ablation Studies and Experiments

**Datasets.** Allstate (10M rows, 4227 features, sparse from one-hot), Higgs (10M, 28 dense), Yahoo LTRC (473K, 700, ranking), Criteo (1.7B, 67, CTR, >1 TB in LibSVM).

**Exact greedy classification, Higgs-1M, 500 trees:**

| Method | sec/tree | Test AUC |
|---|---|---|
| XGBoost | 0.6841 | 0.8304 |
| XGBoost (colsample 0.5) | 0.6401 | 0.8245 |
| scikit-learn | 28.51 | 0.8302 |
| R gbm | 1.032 | 0.6224 |

Same accuracy as scikit-learn, ~42× faster. R's gbm is fast but expands only one branch, and pays 21 AUC points for it.

**Ranking, Yahoo LTRC, 500 trees:**

| Method | sec/tree | [[NDCG]]@10 |
|---|---|---|
| XGBoost | 0.826 | 0.7892 |
| XGBoost (colsample 0.5) | 0.506 | 0.7913 |
| pGBRT | 2.576 | 0.7915 |

**What did not work / what surprised them.**

- **Column subsampling is not free.** On Higgs it *hurt* — AUC 0.8304 → 0.8245. Their explanation: few features carry the signal, so greedily choosing among all of them pays. On Yahoo LTRC the same setting *helped* (0.7892 → 0.7913) and halved the time. So it is a per-dataset regularisation knob, not a default.
- **Global vs local proposals (Fig. 3, Higgs-10M).** Local proposal needs far fewer buckets, because it refines candidates after each split — this matters for deep trees. But global proposal with a small enough $\epsilon$ (more buckets) matches it. And critically, **the approximate quantile method matches exact greedy AUC at a reasonable $\epsilon$** — the approximation costs nothing in accuracy. So the whole scalability story does not trade away quality.
- **Sparsity-aware (Fig. 5, Allstate-10K):** >50× over the sparsity-blind version. The single biggest algorithmic speedup in the paper.
- **Cache prefetching (Fig. 7):** 2× on 10M-row datasets, and *nothing* on 1M-row datasets. The gradient stats fit in cache at 1M rows. A pure scale effect — easy to miss if you benchmark small.
- **Block size sweep (Fig. 9):** a genuine U-curve. $2^{16}$ wins; smaller means poor parallelisation, larger means cache misses.
- **Out-of-core (Fig. 11, Criteo):** the basic version dies at 200M examples (disk space). Compression gives 3×, two-disk sharding another 2×. There is a visible knee where the OS file cache runs out; the compressed+sharded version has a gentler knee and stays linear after it. Final system: 1.7B examples on one c3.8xlarge.
- **Distributed (Fig. 12, 32 m3.2xlarge on YARN):** >10× faster per iteration than Spark MLlib, 2.2× faster than tuned H2O (and H2O is much slower to load data, so end-to-end is worse). Spark and H2O are in-memory only and simply cannot finish the full dataset with those resources. Scaling with machine count is slightly *super*-linear — more machines means more aggregate file cache. Four machines suffice for the full 1.7B rows.

## Worth Remembering

- **The $h_i$ weighting is the subtle bit.** Most people know boosting fits residuals. Fewer notice that once you go second-order, rows are not equal — a row with tiny curvature $h_i$ barely constrains the leaf value, so binning by row-count percentile is the wrong binning. For squared loss $h_i = 1$ and this collapses to ordinary quantiles, which is why nobody hit the problem before.
- **Missing values are not imputed.** The default direction is learnt per node from the gain. This is why XGBoost tolerates raw sparse feature matrices where a linear model or a neural net would need careful filling. It also means "missing" is treated as informative — usually right, sometimes a leak.
- **The regularised objective is presented as a minor addition** ("we include for completeness") but $\gamma$ and $\lambda$ are what let people run depth-8 trees on noisy tabular data without catastrophe.
- **Practical caveats.** The exact greedy algorithm needs the whole dataset in one block in memory. `tree_method` choice (exact vs approx vs hist) is a memory decision, not an accuracy decision — the ablation says accuracy is roughly unchanged. High-cardinality categoricals are still your problem; the Criteo preprocessing here replaced 26 ID features with count and average-CTR statistics computed on an earlier ten days, which is exactly the leak-prone target-encoding trick you have to be careful with.
- **Connections.** The GBDT-as-feature-transform pipeline in [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] is the same family of model in production. The second-order + regularisation combination is what LightGBM and CatBoost later built on (with histogram binning and leaf-wise growth respectively). For tabular data, XGBoost still beats deep nets often enough that this 2016 paper remains a live baseline — an interesting counterweight to [[The Bitter Lesson (essay)]].
- **Open questions.** The $h_i$-weighted quantile sketch is claimed as generally useful beyond trees — that thread was never really picked up. Also: no comparison against a well-tuned Random Forest, and no study of how the ablation gains interact (is 50× sparsity plus 2× cache actually 100×?).

## Links

Related: [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] · [[Regularization]] · [[Derivative]] · [[Loss, Objectives, and Business Alignment]] · [[Cross Entropy]] · [[NDCG]] · [[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Recommender Systems - Evolution]] · [[Foundational_Reading_Plan]]

New topics worth writing: Gradient boosting (Friedman 2001), CART regression trees, Greenwald-Khanna quantile sketch, LightGBM histogram binning and GOSS, CatBoost ordered target statistics, Random Forest, LambdaMART, CSC sparse matrix format, out-of-core learning
