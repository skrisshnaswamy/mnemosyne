---
title: "CatBoost: Unbiased Boosting with Categorical Features"
authors: ["Liudmila Prokhorenkova", "Gleb Gusev", "Aleksandr Vorobev", "Anna Veronika Dorogush", "Andrey Gulin"]
year: 2018
arxiv: "1706.09516"
url: https://arxiv.org/abs/1706.09516
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, theory]
---
## The Core Idea

Gradient boosting has a leak that nobody had named. At every step you compute the gradient (the "how wrong am I" signal) for each training example *using a model that was already trained on that same example*. So the model's prediction on a training point $x_k$ is not distributed like its prediction on a fresh test point $x$. The residual you fit next is therefore slightly wrong in a systematic direction. CatBoost calls this **prediction shift**, proves it is non-zero, and shows the bias shrinks like $1/n$ — so it hurts most on small data.

The same leak appears, much more violently, in **target encoding** for categorical features: replacing a category like `user_id=4711` with the average label of that category. If you include example $k$'s own label $y_k$ in that average, a single split on the encoded feature can perfectly separate the training set and learn nothing.

The fix for both is one trick: **the ordering principle**. Pretend the training data arrived in a random order over time. For each example, only use the examples *before* it. Nothing an example is scored with ever saw its own label.

> [!NOTE] Prediction shift
> The conditional distribution of the model's output on a training point, $F_{t-1}(x_k)\mid x_k$, differs from that on a test point, $F_{t-1}(x)\mid x$, because $y_k$ was used to build $F_{t-1}$. This biases every subsequent gradient step. ^prediction-shift

What it unlocks: you get out-of-fold-style honesty *without* throwing away data or paying a $k\times$ training cost. It is the same instinct as [[Double-Debiased Machine Learning for Treatment and Structural Parameters#Why sample splitting, separately|cross-fitting]] and [[Recursive Partitioning for Heterogeneous Causal Effects#The Methodology|honest estimation]] in causal inference, applied inside the boosting loop.

## The Methodology

**Target statistics (TS).** For categorical feature $i$ of example $k$, replace the category with a number estimating $\mathbb{E}(y \mid x^i = x^i_k)$, smoothed toward a prior $p$:

$$\hat{x}^i_k = \frac{\sum_{x_j \in \mathcal{D}_k} \mathbb{1}\{x^i_j = x^i_k\}\, y_j + a\,p}{\sum_{x_j \in \mathcal{D}_k} \mathbb{1}\{x^i_j = x^i_k\} + a}$$

Everything hinges on the choice of $\mathcal{D}_k$, the set you average over.

- **Greedy TS**: $\mathcal{D}_k = \mathcal{D}$ (everything, including $x_k$). Leaks.
- **Holdout TS**: split the data, compute TS on one half, train on the other. Honest but wastes data.
- **Leave-one-out**: $\mathcal{D}_k = \mathcal{D}\setminus x_k$. Still leaks! For a constant feature, $\hat{x}^i_k = \frac{n^+ - y_k + ap}{n-1+a}$, so a split at $t = \frac{n^+-0.5+ap}{n-1+a}$ perfectly classifies the training set.
- **Ordered TS** (CatBoost): draw a random permutation $\sigma$; set $\mathcal{D}_k = \{x_j : \sigma(j) < \sigma(k)\}$. At test time use the whole training set.

The two properties they demand: **P1** the distribution of $\hat{x}^i \mid y$ matches between train and test; **P2** all training data gets used. Only ordered TS satisfies both.

**Prediction shift, made precise.** Toy setup: two i.i.d. Bernoulli(1/2) features, $y = c_1 x^1 + c_2 x^2$, squared loss, two boosting rounds with depth-1 stumps, step size 1.

> [!NOTE] Theorem 1
> If independent datasets $\mathcal{D}_1, \mathcal{D}_2$ are used for the two stumps, $\mathbb{E}F_2(x) = f^*(x) + O(1/2^n)$ — unbiased.
> If the *same* dataset is used twice, $\mathbb{E}F_2(x) = f^*(x) - \frac{1}{n-1}c_2\!\left(x^2 - \tfrac12\right) + O(1/2^n)$. ^theorem-1

The bias is $\propto 1/n$ and $\propto c_2$ (the strength of the relationship the *second* tree has to learn). This is the first formal proof that the shift is non-zero, not just hand-waved.

**Ordered boosting (Algorithm 1).** Pick permutation $\sigma$. Maintain $n$ supporting models $M_1,\dots,M_n$, where $M_i$ is trained only on the first $i$ examples. To get the residual for example $j$, use $M_{j-1}$ — a model that has never seen $y_j$. Naively this costs $n\times$ memory and time.

**The practical version (Algorithm 2/3).** Two modes:

- **Plain** — ordinary GBDT, but with ordered TS baked in.
- **Ordered** — the real thing, made cheap.

Key implementation choices:

1. **Base learner is an oblivious decision tree**: every node at the same depth uses the *same* split. Balanced, less overfitting, and the whole tree evaluates as one bit-index lookup at serve time.
2. **$s+1$ permutations**, generated up front. $\sigma_1,\dots,\sigma_s$ are used for evaluating splits (a random one is drawn each iteration); $\sigma_0$ is used to compute the final leaf values $b_j$. Using one permutation makes early examples very high-variance; several average that out.
3. **The $\log n$ memory trick.** Instead of $O(sn^2)$ supporting predictions $M_{r,j}(i)$, only store $M'_{r,j}(i) := M_{r,2^j}(i)$ for $j = 1..\lceil\log_2 n\rceil$. Prefix lengths get rounded down to powers of two. Memory drops to $O(sn)$, and the asymptotic cost per iteration matches plain GBDT: $O(s\cdot n)$ for gradients, $O(|C|\cdot n)$ to build the tree.
4. **Split scoring by cosine similarity.** A candidate split $T_c$ is scored by $\cos(\Delta, G)$, where $G$ is the vector of gradients and $\Delta(i)$ is the leaf value example $i$ would get — computed by averaging gradients only over *preceding* examples in the same leaf. This is itself an ordered statistic, with gradients playing the role of targets.
5. **$\sigma_{cat} = \sigma_{boost}$.** The permutation used for target encoding must be the *same* one used for ordered boosting. Otherwise $M_{\sigma_{boost}(j)}$ is trained on TS features that were built using $y_j$, and the leak comes back through the side door.
6. **Feature combinations.** At each split, CatBoost concatenates every categorical feature (and combination) already used in the current tree with every categorical feature in the dataset, and converts the concatenation to a TS on the fly. Greedy, so it dodges the exponential blowup. This is how it captures `user_id × ad_topic` interactions — the same [[Wide & Deep Learning for Recommender Systems#The Core Idea|memorisation]] job that crossed features do in ad systems.
7. **Bayesian bootstrap** for row subsampling: weights $w_i = a^t_i$ multiply the gradients. They are explicit that [[Proximal Policy Optimization Algorithms|subsampling]]-style stochasticity à la Friedman only *alleviates* prediction shift, it does not remove it.

Tuning: 50 steps of TPE via Hyperopt (a [[Practical Bayesian Optimization of Machine Learning Algorithms|Bayesian optimisation]] method), 5-fold CV on 80% of data, tree count searched exhaustively in $[1, 5000]$.

## Ablation Studies and Experiments

Nine datasets: Adult, Amazon, Click Prediction, Epsilon, four KDD-Cup-2009 sets, Kick. Metrics: logloss (see [[Cross Entropy]]) and zero-one loss. **Important for fairness: XGBoost and LightGBM also got ordered TS as preprocessing** — the comparison is not CatBoost's encoding against one-hot.

**Versus baselines** (logloss / zero-one, relative increase for baselines):

| Dataset | CatBoost | LightGBM | XGBoost |
|---|---|---|---|
| Adult | 0.2695 / 0.1267 | +2.4% / +1.9% | +2.2% / +1.0% |
| Amazon | 0.1394 / 0.0442 | +17% / +21% | +17% / +21% |
| Epsilon | 0.2647 / 0.1086 | +1.5% / +4.1% | +11% / +12% |
| Internet | 0.2089 / 0.0937 | +6.8% / +8.6% | +7.9% / +8.0% |
| Kick | 0.2855 / 0.0949 | +3.5% / +4.4% | +3.2% / +4.1% |
| Upselling | 0.1662 / 0.0490 | +0.3% / +0.1% | +0.04% / +0.3% |

Wins everywhere, but the margin is *huge* on Amazon (high-cardinality categoricals) and *nothing* on Appetency/Churn/Upselling — those three are the ones where the improvement is **not** statistically significant.

**Which TS actually matters** (relative logloss increase vs ordered TS, all else fixed):

| | Greedy | Holdout | Leave-one-out |
|---|---|---|---|
| Amazon | +40% | +8.3% | +4.5% |
| Upselling | +57% | +1.6% | +3.9% |
| Internet | +33% | +2.6% | +27% |
| Adult | +1.1% | +2.1% | +5.5% |

This is the biggest lever in the whole paper. Note the flip: leave-one-out beats greedy usually, but is *worse* on Adult, because greedy TS hurts on low-frequency categories while leave-one-out hurts on high-frequency ones — and every Adult feature is high-frequency.

**Ordered vs Plain boosting** is a much smaller effect, and sometimes negative: Internet +3.9% logloss for Plain, Adult +1.1%, but Amazon **−0.6%**, Kick −0.2%, Churn −0.06%. Plain sometimes wins. The clean story comes from the subsampling curve (Figure 2): train both on randomly filtered fractions of each dataset, and Plain's relative loss degrades sharply as the fraction shrinks — up to +25% at 0.1% of the data. Exactly what Theorem 1's $1/n$ bias predicts.

**Feature combinations**: going from $c_{max}=1$ to $2$ improves logloss by 1.86% on average, up to 11.3% on one dataset. $1\to3$ gives 2.04%. Beyond 3, nothing.

**Number of permutations $s$**: $s=3$ gives 0.19%, $s=9$ gives 0.38%. Real but tiny.

**The honest ablation they buried in Appendix G.** They strip CatBoost down to a "raw setting" — Plain mode, no combinations, no random strength, categoricals pre-encoded as ordered TS for everyone. Averaged over 8 datasets, raw CatBoost is **0.2% worse than the baselines on logloss and 0.2% better on zero-one loss**. In other words: their GBDT engine is not better than XGBoost or LightGBM. Every point of the headline win comes from ordered TS, feature combinations, and ordered boosting.

**What did not work / was rejected:**
- **Leave-one-out TS** — intuitively safe, provably leaky (the constant-feature counterexample).
- **Iterated bagging** (Breiman) — Appendix E shows its out-of-bag residuals are *still* shifted, because model $F^t_k$ trained without $x_j$ still uses estimates $M^{t-1}_{j'}$ for $j' \in \mathcal{D}_k$, which depend on $(x_j, y_j)$. Plus it costs $K\times$.
- **Stochastic subsampling** — only heuristic relief, not a fix.
- **Mismatched permutations** ($\sigma_{boost} \neq \sigma_{cat}$) — Ordered still beats Plain by 0.5% logloss, but matching them gets you 0.6% logloss and 0.9% zero-one. The correspondence matters.
- **LightGBM's gradient statistics** for categoricals — dismissed on cost grounds (recompute per step, store category→node maps) and because LightGBM's own docs recommend converting high-cardinality categoricals to numeric anyway.

**Speed** (Epsilon, 8000 trees, 16 threads): CatBoost Plain 1.1 s/tree, LightGBM 1.1 s/tree, CatBoost Ordered 1.9 s/tree, XGBoost (hist) 3.9 s/tree. Ordered mode costs ~1.7×.

## Worth Remembering

- **The one-line takeaway for practice**: if you use target encoding anywhere — CTR models, [[Deep Interest Network for CTR Prediction (DIN)|recsys features]], anything with `user_id` — compute it from a prefix or an out-of-fold split, never from the full data. This is the mistake that quietly inflates offline metrics and vanishes online.
- **Ordered boosting is the weaker half of the paper.** The TS ablation shows swings of 40–57%; the boosting-mode ablation shows 0.1–4%, with sign flips. Use Ordered mode when your dataset is small (<40K rows, per their evidence); on large data Plain is as good and 1.7× faster.
- **Oblivious trees are a serving decision, not an accuracy decision.** They cost some expressiveness per tree and buy fast, branch-free inference. The raw-setting table confirms they do not hurt.
- **The $1/n$ bias scaling is the useful mental model.** Prediction shift is a small-sample pathology, like the [[Reconciling Modern ML Practice and the Bias-Variance Trade-off|bias–variance]] story generally. If you have 10M rows it is not your problem.
- **A caveat on the baselines.** The comparison uses Hyperopt-tuned XGBoost/LightGBM, which is more careful than most papers — but [[On the Difficulty of Evaluating Baselines|baseline tuning effort]] is asymmetric almost by construction when the authors own one library. Also, CatBoost's numbers in the paper differ slightly from the public benchmark repo, which they admit.
- **The prediction-shift analysis is proved for squared loss and depth-1 stumps only.** They assert the method applies elsewhere; the theorem does not cover it.
- Open question worth chasing: they say the ordering principle "is also effective for gradient statistics" (LightGBM's categorical handling) but never test it.
- Nice conceptual connection: ordered TS is exactly online learning's "only use the past" rule, imported into an offline batch setting by inventing a fake arrival time. Same trick as [[Deep Neural Networks for YouTube Recommendations (RecSys)#^example-age|holding out the future]] in recsys training.

## Links

Related: [[XGBoost- A Scalable Tree Boosting System]] · [[LightGBM- A Highly Efficient Gradient Boosting Decision Tree (NeurIPS)]] · [[Why do tree-based models still outperform deep learning on tabular data]] · [[Double-Debiased Machine Learning for Treatment and Structural Parameters]] · [[Recursive Partitioning for Heterogeneous Causal Effects]] · [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] · [[Factorization Machines (ICDM)]] · [[Wide & Deep Learning for Recommender Systems]] · [[Cross Entropy]] · [[Practical Bayesian Optimization of Machine Learning Algorithms]] · [[On the Difficulty of Evaluating Baselines]] · [[Regularization]]

New topics worth writing: Target encoding / mean encoding, Gradient boosting (Friedman 2001), Oblivious decision trees and decision tables, Out-of-bag estimation and iterated bagging, Bayesian bootstrap, One-hot encoding for high-cardinality features, Conditional shift vs covariate shift
