---
title: "LightGBM: A Highly Efficient Gradient Boosting Decision Tree (NeurIPS)"
authors: ["Ke et al."]
year: 2017
url: https://papers.nips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, theory]
---
## The Core Idea

Gradient boosted trees are slow for one boring reason: to pick where to split a node, you must scan every row, for every feature. Cost is $O(\#\text{data} \times \#\text{feature})$. If you want it faster, you must shrink one of those two numbers. LightGBM shrinks both, and does it in a way that provably barely changes the split you would have picked anyway.

**Shrink the rows (GOSS).** In [[XGBoost- A Scalable Tree Boosting System|GBDT]] there are no sample weights, so the old boosting sampling tricks (which reweight rows) do not apply. But there *is* a per-row [[Derivative#Gradient|gradient]], and a big absolute gradient means "the model is still wrong about this row". So: keep **all** the big-gradient rows, randomly sample a small slice of the small-gradient rows, and multiply that slice's gradients by a constant so the sums still look like the full dataset. You get most of the signal from a fraction of the rows.

**Shrink the columns (EFB).** Real wide datasets are sparse. One-hot columns are *mutually exclusive* — they are almost never nonzero at the same time. Two such columns can be jammed into one column by giving each its own range of histogram bins. The histogram you build from the bundle is the same as the two histograms you would have built separately. Cost becomes $O(\#\text{data} \times \#\text{bundle})$ with $\#\text{bundle} \ll \#\text{feature}$.

Neither trick existed before because (a) sampling for boosting had been studied only for AdaBoost, where weights exist, and (b) the histogram algorithm — unlike the pre-sorted algorithm — cannot skip zeros, since it must look up a bin index for every row regardless. EFB is the fix that finally lets histogram-based GBDT exploit sparsity.

Result: up to **21× faster per iteration** than the same code without these tricks, with AUC unchanged to the fourth decimal.

> [!NOTE] Histogram-based split finding
> Instead of sorting continuous feature values and trying every threshold, bucket each feature into ~255 discrete bins once, then accumulate gradient sums per bin. Building the histogram costs $O(\#\text{data} \times \#\text{feature})$; scanning it for the best split costs only $O(\#\text{bin} \times \#\text{feature})$. Building dominates. ^histogram-split

## The Methodology

**The gain being approximated.** At a node with data $O$, splitting feature $j$ at point $d$ gives variance gain

$$V_{j|O}(d) = \frac{1}{n_O}\left(\frac{\left(\sum_{\{x_i \in O:\, x_{ij} \le d\}} g_i\right)^2}{n^j_{l|O}(d)} + \frac{\left(\sum_{\{x_i \in O:\, x_{ij} > d\}} g_i\right)^2}{n^j_{r|O}(d)}\right)$$

where $g_i$ is the negative gradient of the loss at row $i$. Squared sum of gradients divided by count, on each side. Note it depends on rows only through their gradients — that is exactly why gradient magnitude is the right importance signal.

**GOSS, step by step (one boosting iteration):**

1. Predict with the current ensemble, compute gradients $g$ for all $n$ rows.
2. Sort rows by $|g_i|$, descending.
3. `topSet` = top $a \times n$ rows. Keep all of them.
4. `randSet` = a uniform random $b \times n$ rows drawn from the *remaining* $(1-a)n$.
5. Set weight $1$ for `topSet`, weight $\text{fact} = \frac{1-a}{b}$ for `randSet`.
6. Grow one tree on `topSet ∪ randSet` only, with those weights.

The estimated gain becomes

$$\tilde V_j(d) = \frac{1}{n}\left(\frac{\left(\sum_{x_i\in A_l} g_i + \frac{1-a}{b}\sum_{x_i\in B_l} g_i\right)^2}{n^j_l(d)} + \frac{\left(\sum_{x_i\in A_r} g_i + \frac{1-a}{b}\sum_{x_i\in B_r} g_i\right)^2}{n^j_r(d)}\right)$$

$A$ is the kept top set, $B$ the sampled small-gradient set. The factor $\frac{1-a}{b}$ rescales $B$'s gradient sum back up to what the whole discarded region $A^c$ would have contributed. Without it you shift the data distribution and break the model.

**The error bound.** With probability at least $1-\delta$,

$$E(d) = |\tilde V_j(d) - V_j(d)| \le C_{a,b}^2 \ln(1/\delta)\cdot\max\left\{\tfrac{1}{n^j_l(d)}, \tfrac{1}{n^j_r(d)}\right\} + 2DC_{a,b}\sqrt{\tfrac{\ln(1/\delta)}{n}}$$

with $C_{a,b} = \frac{1-a}{\sqrt b}\max_{x_i\in A^c}|g_i|$. Read it as: the error shrinks like $O(1/\sqrt n)$ as data grows, provided the split is not wildly unbalanced. Crucially, $C_{a,b}$ contains $\max_{x_i \in A^c}|g_i|$ — the largest gradient among the rows you *did not* keep. Because GOSS keeps the big ones, that max is small, so the bound is tight. Plain random sampling is the special case $a=0$, where that max is the global max, and the bound is loose.

**EFB, step by step (once, before training):**

*Which features to bundle (Alg. 3).* Build a graph: features are vertices, an edge joins two features that conflict (both nonzero on some row), weighted by conflict count. Finding the fewest bundles is exactly graph colouring, hence **NP-hard** (the proof: take each row of a graph's incidence matrix as a feature; a bundle is a colour class). So use a greedy: sort features by degree descending, walk the list, drop each feature into the first existing bundle whose total conflict stays $\le K$, else start a new bundle. Cost $O(\#\text{feature}^2)$. For millions of features, replace the degree sort with a cheaper proxy: sort by count of nonzero values, and skip building the graph entirely.

*How to merge (Alg. 4).* Offset the bins. Feature A occupies bins $[0,10)$, feature B occupies $[0,20)$; add offset 10 to B so it lives in $[10,30)$. The bundle has bin range $[0,30)$. For each row, whichever member feature is nonzero writes its offset bin; if all are zero the bundle is zero. The bundle's histogram is bit-identical to the separate histograms of the members.

Allowing a small conflict rate $\gamma > 0$ gives fewer bundles. Randomly polluting a $\gamma$ fraction of values costs at most $O([(1-\gamma)n]^{-2/3})$ in accuracy.

**Tree growth.** Leaf-wise (best-first), not level-wise — grow whichever leaf gives the largest gain, anywhere in the tree.

**Experimental settings.** $a=b=0.05$ for Allstate, KDD10, KDD12; $a=b=0.1$ for Flight Delay and LETOR. $\gamma=0$ in the headline EFB numbers. 16 threads, dual E5-2670 v3 (24 cores), 256 GB RAM.

## Ablation Studies and Experiments

Five datasets: Allstate (12M rows, 4228 features, sparse, AUC), Flight Delay (10M × 700, sparse, AUC), LETOR (2M × 136, **dense**, [[NDCG]]@10), KDD10 (19M × 29M, sparse), KDD12 (119M × 54M, sparse).

Baselines: `xgb_exa` (XGBoost pre-sorted), `xgb_his` (XGBoost histogram), `lgb_baseline` (LightGBM with *neither* trick), `EFB_only` (EFB but no GOSS).

**Seconds per iteration (Table 2):**

| | xgb_exa | xgb_his | lgb_baseline | EFB_only | LightGBM |
|---|---|---|---|---|---|
| Allstate | 10.85 | 2.63 | 6.07 | 0.71 | **0.28** |
| Flight Delay | 5.94 | 1.05 | 1.39 | 0.27 | **0.22** |
| LETOR | 5.55 | 0.63 | 0.49 | 0.46 | **0.31** |
| KDD10 | 108.27 | OOM | 39.85 | 6.33 | **2.85** |
| KDD12 | 191.99 | OOM | 168.26 | 20.23 | **12.67** |

Speedups over `lgb_baseline`: 21× (Allstate), 6× (Flight), 1.6× (LETOR), 14× (KDD10), 13× (KDD12). XGBoost's histogram mode ran out of memory on both KDD datasets.

**Which trick does the work?** Read the columns. `lgb_baseline` → `EFB_only` is the EFB contribution: 6.07 → 0.71 on Allstate (8.5×), 39.85 → 6.33 on KDD10 (6.3×). `EFB_only` → `LightGBM` is the GOSS contribution: roughly 2× everywhere. **EFB is doing most of the lifting on sparse data.** On LETOR — dense, only 136 features — EFB does almost nothing (0.49 → 0.46) and GOSS supplies essentially all of the modest 1.6×.

Note GOSS gives ~2×, not 10×, even at a 10% sampling rate. Prediction and gradient computation still run over the *full* dataset every iteration; only tree building sees the subset. Speedup is sublinear in sampling rate by construction.

Also worth noting: `lgb_baseline` already includes a nonzero-index table optimisation for sparse features, and EFB still beats it by 8×. Part of that is fewer columns; part is **cache locality** — bundled features sit contiguously in memory, so histogram building hits cache far more often.

**Accuracy (Table 3):** unchanged.

| | lgb_baseline | SGB | LightGBM |
|---|---|---|---|
| Allstate AUC | 0.6093 | 0.6064 ± 7e-4 | 0.6093 ± 9e-5 |
| Flight AUC | 0.7847 | 0.7780 ± 8e-4 | 0.7846 ± 4e-5 |
| LETOR NDCG@10 | 0.5277 | 0.5239 ± 6e-4 | 0.5275 ± 5e-4 |
| KDD10 AUC | 0.78735 | 0.7759 ± 3e-4 | 0.78732 ± 1e-4 |
| KDD12 AUC | 0.7049 | 0.6989 ± 8e-4 | 0.7051 ± 5e-5 |

**GOSS vs Stochastic Gradient Boosting, matched sampling rate (Table 4, LETOR NDCG@10):**

| ratio | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.35 | 0.4 |
|---|---|---|---|---|---|---|---|
| SGB | 0.5182 | 0.5216 | 0.5239 | 0.5249 | 0.5252 | 0.5263 | 0.5267 |
| GOSS | 0.5224 | 0.5256 | 0.5275 | 0.5284 | 0.5289 | 0.5293 | 0.5296 |

GOSS wins at every rate, and the gap widens as the rate shrinks — exactly what the bound predicts, since $C_{a,b}$ blows up for uniform sampling when $b$ is small. At ratio 0.2, GOSS already matches the full-data 0.5277; SGB needs more than 0.4 and still does not get there.

**What did not work / was rejected:**

- **Plain random subsampling (SGB, Friedman 2002).** Applicable to GBDT, but costs accuracy at every sampling rate. The authors call it "not a desirable choice".
- **AdaBoost-style weight-threshold sampling.** Cannot be ported — GBDT has no native instance weights at all.
- **Dropping small-gradient rows outright** (no rescaling). Changes the data distribution, hurts the model. The $\frac{1-a}{b}$ multiplier is not optional.
- **Feature reduction via PCA / projection pursuit.** Assumes features are redundant. The authors argue features are usually engineered to each contribute something unique, so throwing any away costs accuracy. EFB is lossless by contrast — nothing is discarded, only repacked.
- **Per-feature nonzero-index tables** as the sparsity fix. Implemented in LightGBM as a basic function, but it needs extra memory and must be maintained through the whole tree growth, and it does not give the cache-locality win. EFB dominates it (and the two compose — you can still use the table when bundles are themselves sparse).

## Worth Remembering

- **The speedup is dataset-shaped, not universal.** 21× on wide sparse Allstate; 1.6× on narrow dense LETOR. If your features are dense and few, EFB gives you nothing and you are left with GOSS's ~2×.
- **The exclusivity assumption is the whole game for EFB.** One-hot columns, bag-of-words, hashed categorical IDs — yes. Dense numeric telemetry — no. This is why LightGBM shines on the kind of high-cardinality-categorical data that fills [[Deep Learning Recommendation Model (DLRM)|recsys]] and [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)|ad CTR]] pipelines.
- **Open problem the authors name:** how to choose $a$ and $b$. They tuned them per dataset by hand and list optimal selection as future work. In practice `top_rate` / `other_rate` are rarely touched, and GOSS is not even the default in modern LightGBM (`boosting_type='gbdt'` uses all rows; you must set `boosting_type='goss'`).
- **Leaf-wise growth is a separate, confounded choice.** It is in `lgb_baseline` too, so the tables isolate GOSS/EFB correctly — but it is part of why LightGBM beats XGBoost overall, and it is the thing that makes LightGBM overfit small datasets without `num_leaves` tuning.
- **XGBoost histogram mode OOM'd on KDD10/KDD12.** Memory, not just speed, is the real win at 29M–54M features. EFB reduces the number of histograms you must hold.
- **The bound is about split *selection*, not final loss.** GOSS guarantees you pick nearly the same split point; it says nothing directly about the ensemble's generalisation. The authors decompose $E^{\text{GOSS}}_{\text{gen}} \le E^{\text{GOSS}} + E_{\text{gen}}$ and hand-wave that sampling may even help via [[Regularization|added diversity]] in the base learners.
- Connects naturally to [[Why do tree-based models still outperform deep learning on tabular data|why trees still win on tabular data]] — this paper is a large part of why the practical answer is "and they are fast too".

## Links

Related: [[XGBoost- A Scalable Tree Boosting System]] · [[Why do tree-based models still outperform deep learning on tabular data]] · [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] · [[NDCG]] · [[Derivative#Gradient|Gradient]] · [[Regularization]] · [[Deep Learning Recommendation Model (DLRM)]] · [[Foundational_Reading_Plan]]

New topics worth writing: Gradient Boosting Machines (Friedman 2001), Stochastic Gradient Boosting, Graph colouring and greedy approximation ratios, Leaf-wise vs level-wise tree growth, Cache locality in ML systems, AdaBoost
`
