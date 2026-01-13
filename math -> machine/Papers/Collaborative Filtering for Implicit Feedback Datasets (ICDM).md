---
title: "Collaborative Filtering for Implicit Feedback Datasets (ICDM)"
authors: ["Hu", "Koren & Volinsky"]
year: 2008
url: http://yifanhu.net/PUB/cf.pdf
priority: Must-Read
read_on: 2026-08-25
tags: [paper, optimization]
---
## The Core Idea

Most recommender papers of that era assumed you had **explicit feedback** — star ratings, thumbs up/down. Then you only fit the cells you observed and treat everything else as missing. With **implicit feedback** — watch time, clicks, purchases — that recipe breaks, because everything you observe is positive. If you only fit the observed cells, the model learns "everything is good" and has nothing to rank against.

The trick here: split one raw number into **two** numbers with different jobs.

- $p_{ui}$ — did the user show *any* interaction? A binary preference, 1 or 0.
- $c_{ui}$ — how much do we *trust* that binary label? A confidence weight.

So watching a show 40 times does not mean the user likes it 40× more than a show watched once. It means we are much more sure that the "1" is real. And a zero does not mean dislike — it means "probably no, but we barely believe it," so it gets small weight, not zero weight.

> [!NOTE] Preference–confidence split
> Raw implicit count $r_{ui}$ is decomposed into a binary label $p_{ui} = \mathbb{1}[r_{ui} > 0]$ and a confidence $c_{ui} = 1 + \alpha r_{ui}$. The label says *what* to predict; the confidence says *how hard to push*. ^preference-confidence

The second contribution is the reason this became the default industrial algorithm for a decade: the loss now runs over **all** $m \times n$ user–item pairs (billions of terms), which kills stochastic gradient descent. But because the confidence matrix is "1 everywhere plus a sparse correction", the alternating least squares update can be computed in time linear in the number of *non-zero* observations, not in $m \times n$. You get to optimise a dense objective at sparse cost.

## The Methodology

**The loss.** Learn a user vector $x_u \in \mathbb{R}^f$ and item vector $y_i \in \mathbb{R}^f$ ($f$ = number of latent factors, 10–200), predicting $\hat p_{ui} = x_u^T y_i$:

$$\min_{x_\star, y_\star} \sum_{u,i} c_{ui}\left(p_{ui} - x_u^T y_i\right)^2 + \lambda\left(\sum_u \|x_u\|^2 + \sum_i \|y_i\|^2\right)$$

Note the sum is over *every* $(u,i)$ pair, not just observed ones. The $\lambda$ term is standard L2 [[Regularization]], tuned by cross-validation.

**Confidence functions.** Two variants:
$$c_{ui} = 1 + \alpha r_{ui} \qquad\text{or}\qquad c_{ui} = 1 + \alpha \log(1 + r_{ui}/\epsilon)$$
$\alpha = 40$ worked well in general. For TV data they used the log version with $\epsilon = 10^{-8}$, because watch counts spanned 0.1 to hundreds (a DVR left recording all day).

**Alternating Least Squares (ALS).** With $y$ fixed the loss is quadratic in $x$, so there is a closed-form global minimum, and vice versa. Take the [[Derivative#Gradient|gradient]], set to zero:

$$x_u = (Y^T C^u Y + \lambda I)^{-1} Y^T C^u p(u)$$
$$y_i = (X^T C^i X + \lambda I)^{-1} X^T C^i p(i)$$

where $C^u$ is the $n \times n$ diagonal matrix of that user's confidences.

**The scalability trick.** Computing $Y^T C^u Y$ naively is $O(f^2 n)$ per user — hopeless. Rewrite:
$$Y^T C^u Y = Y^T Y + Y^T (C^u - I) Y$$
$Y^T Y$ does not depend on $u$, so precompute it once per sweep in $O(f^2 n)$. And $C^u - I$ has only $n_u$ non-zeros (the items the user actually touched), since confidence is exactly 1 everywhere else. Likewise $C^u p(u)$ has $n_u$ non-zeros. So one user update costs $O(f^2 n_u + f^3)$, and a full sweep costs $O(f^2 N + f^3 m)$ where $N = \sum_u n_u$ is the number of non-zero observations. **Linear in the input.** About 10 alternating sweeps to converge.

**Explanations for free.** Latent factor models are usually opaque — the user's history is compressed into $x_u$ and you cannot point at which past action caused a recommendation. Substitute the closed form for $x_u$ back into the prediction. Let $W^u = (Y^T C^u Y + \lambda I)^{-1}$ and define a *user-specific* item similarity $s^u_{ij} = y_i^T W^u y_j$. Then:

$$\hat p_{ui} = \sum_{j: r_{uj} > 0} s^u_{ij}\, c_{uj}$$

The factor model collapses into a linear item–item neighbourhood model, where each past action contributes one term you can read off. You can even split the contribution into "how strong was the past action" ($c_{uj}$) and "how similar is it to the target" ($s^u_{ij}$). Costs $O(f^2 n_u + f^3)$ per explanation.

**Data.** Digital TV service, ~300k set-top boxes, ~17,000 programs over four weeks, ~32M non-zero $r_{ui}$. $r_{ui} = 0.7$ means 70% of the show watched; $r_{ui}=2$ means watched twice. Test set = the following single week, ~2M non-zeros.

**Two data hacks that mattered more than the model:**
1. **Remove re-watches from the test set.** Shows the user already watched in training are deleted from test. Predicting re-watching is trivially easy and is not what a recommender is for.
2. **Momentum down-weighting.** If a channel stays on for hours, later shows probably reflect a sleeping viewer, not preference. The $t$-th show after a channel tune is weighted by $\frac{e^{-(at-b)}}{1+e^{-(at-b)}}$ with $a=2, b=6$ — the third show gets halved, the fifth is cut by 99%.

## Ablation Studies and Experiments

**Metric.** Precision is unusable (you never know what a user *disliked*), so they use expected percentile rank:
$$\overline{\text{rank}} = \frac{\sum_{u,i} r^t_{ui} \cdot \text{rank}_{ui}}{\sum_{u,i} r^t_{ui}}$$
$\text{rank}_{ui}=0\%$ means top of the list. Random guessing gives 50%. Lower is better.

| Model | $\overline{\text{rank}}$ |
|---|---|
| Popularity (non-personalised) | 16.46% |
| Item-item neighbourhood (cosine, all items as neighbours) | 10.74% |
| Dense SVD on raw $r_{ui}$, no regularisation | worse than popularity |
| Dense SVD on raw $r_{ui}$, $\lambda_1=500$, $f=50$ | 13.63% |
| Dense SVD on raw $r_{ui}$, $f=100$ | 13.40% |
| Factorise binary $p_{ui}$ only, $\lambda_2=150$, $f=50$ | 10.72% |
| Factorise binary $p_{ui}$ only, $f=100$ | 10.49% |
| **Full model (preference + confidence), $f=50$** | **8.93%** |
| **Full model, $f=100$** | **8.56%** |
| **Full model, $f=200$** | **8.35%** |

This table *is* the ablation, and it is unusually clean:

- **Fitting raw counts as if they were ratings does not work.** 13.4% — worse than a simple neighbourhood model. Counts are confidence, not preference. Without regularisation it cannot even beat popularity.
- **Binarising alone gets you to neighbourhood parity** (10.49%), barely better than cosine item-item.
- **Adding confidence weights is what actually wins** — 10.49% → 8.56%, roughly a 2-point drop from a single change. The $c_{ui}$ term is doing the heavy lifting, not the factorisation.
- More factors monotonically help up to 200. Their advice: use the largest $f$ you can afford.

**Distribution of hits.** With $f=100$, a show actually watched in the test week lands in the top 1% of recommendations about 27% of the time, beating the neighbourhood curve everywhere.

**Where the model is weak (Figure 3).** Splitting test items into 15 popularity bins: accuracy varies enormously — popular shows are easy, tail shows are hard. The model plays it safe with familiar items. But splitting *users* by total watch time: performance is basically flat after the very first bin. Surprising — with explicit feedback, more history reliably means better predictions. The authors' guess: heavy TV accounts are shared households, so more data means a *more mixed* signal, not a cleaner one.

**Neighbourhood baseline tuning.** Two choices that were opposite to explicit-feedback practice: use *all* items as neighbours (not top-$k$), and use cosine rather than Pearson correlation. Worth remembering as a baseline-tuning lesson in the spirit of [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]].

**Explanation quality.** For one user: *So You Think You Can Dance* explained by *Hell's Kitchen*, *Access Hollywood*, *Judge Judy*; *Spider-Man* explained by *Batman*, *Superman*, *Pinky and The Brain*. The top 5 past shows account for only 35–40% of the score, so the recommendation is genuinely a broad sum, not one dominant neighbour.

## Worth Remembering

- **The zero-confidence uniformity is a deliberate compromise.** Every unobserved pair gets $c_{ui}=1$, which is exactly why $C^u - I$ is sparse and the whole thing scales. The authors admit a better model would give *different* confidences to different zeros — "not watched because it was on an obscure channel at 3am" vs "not watched because a favourite show was on at the same time" vs "genuinely uninterested." Making that flexible without wrecking the $O(f^2 N)$ cost was left open. This is where later work on exposure and [[Unbiased Learning-to-Rank with Biased Feedback|propensity]] picks up.
- **No negative sampling here.** Unlike [[BPR- Bayesian Personalized Ranking from Implicit Feedback]] (published the same year) or [[Distributed Representations of Words and Phrases (negative sampling)]], this uses *all* negatives with small weight rather than sampling a few. Pointwise weighted regression vs pairwise ranking — two different answers to the same "there are no explicit negatives" problem. Worth reading them side by side.
- **The closed-form update is why ALS beat SGD here.** Because the objective is dense, SGD would need billions of updates per epoch. Once you exploit the "identity plus sparse" structure, the exact solution is cheaper than the approximate one. Also trivially parallel: every $x_u$ is independent given $Y$.
- **The re-watch removal is the honest choice and it costs a lot of headline numbers.** Leaving previously-watched shows in the test set makes results look dramatically better (the black dotted line in their Figure 2). Any implicit-feedback benchmark that does not say whether repeats were removed is not comparable. Related: the [[Deep Neural Networks for YouTube Recommendations (RecSys)#^example-age|example age]] and held-out-labelling issues at YouTube.
- **The final paragraph is the most quietly important.** They say outright that predicting future behaviour is *not* the purpose of a recommender — the goal is to surface things the user would not otherwise have found, and there is no offline metric for that. Removing re-watches is their crude approximation of "discovery."
- **Practical note.** This algorithm is what `implicit` (Python) and Spark MLlib's `ALS(implicitPrefs=True)` implement. $\alpha$ and $\lambda$ interact strongly; $\alpha=40$ is a starting point, not a law. If your counts have a heavy tail, use the log confidence.
- **Open question worth chasing.** The explanation formula turns the factor model into a *user-specific* item-item model — the similarity $s^u_{ij}$ depends on $W^u$, so different users disagree about which items are similar. That is a genuinely interesting object and mostly ignored by the literature that followed.

## Links

Related: [[BPR- Bayesian Personalized Ranking from Implicit Feedback]] · [[Recommender Systems - Evolution]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Unbiased Learning-to-Rank with Biased Feedback]] · [[Regularization]] · [[Derivative#Gradient|gradient]] · [[Foundational_RecSys_Ranking_Reading_Plan]] · [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)]] · [[NDCG]] · [[Uncertainty]]

New topics worth writing: Alternating Least Squares, Matrix Factorization for Collaborative Filtering, Singular Value Decomposition, Weighted Regularized Matrix Factorization (WRMF), Exposure bias in implicit feedback, Percentile rank / mean percentile ranking metric, Cosine similarity, Item-item collaborative filtering
