---
title: "On Sampled Metrics for Item Recommendation (KDD)"
authors: ["Krichene & Rendle"]
year: 2020
url: https://dl.acm.org/doi/pdf/10.1145/3394486.3403226
priority: Must-Read
read_on: 2026-09-18
tags: [paper]
---
## The Core Idea

When you evaluate a recommender, you must find where the one held-out item lands among all $n$ items in the catalogue. That is expensive when $n$ is millions. So a lot of papers cheat: draw $m$ random irrelevant items (usually $m = 100$), rank the true item only against those, and compute Recall@10 or NDCG@10 on that tiny list.

This paper shows that cheat is not a noisy version of the truth. It is a *different metric wearing the same name*. And it breaks the one thing a metric exists for: if recommender A beats B on the sampled metric, that tells you almost nothing about whether A beats B on the real metric — **not even if you repeat the sampling infinitely many times**.

> [!NOTE] Consistency under sampling
> A metric $M$ is consistent if, for all pairs of recommenders, $\bar M(A) > \bar M(B) \iff \mathbb{E}[\bar M(\tilde A)] > \mathbb{E}[\bar M(\tilde B)]$, where $\tilde{\cdot}$ means ranks measured on the sample. Only AUC passes. Recall, Precision, [[NDCG]], Average Precision, Hit Ratio, MRR all fail. ^sampling-consistency

Why this happens, in one sentence: sampled ranks are squashed into $\{1, \dots, m+1\}$, and any top-heavy metric applied to a squashed rank stops being top-heavy. In the extreme $m = 1$, *every* metric becomes a linear function of the true rank $r$ — which is exactly AUC. So the whole zoo of ranking metrics collapses onto the one metric nobody in recsys actually wants, because AUC values a move from rank 101→100 as much as 2→1.

This is a bias problem, not a variance problem, and that makes it nastier. Repeat the sampling 100 times and you get a tight standard deviation. The tightness is real. The number it is tight around is wrong, and *how* wrong depends on which recommender you are measuring. You can be confidently, reproducibly, precisely wrong.

What it unlocks: a reason to stop sampling, plus three correction recipes for when you cannot.

## The Methodology

### Setup

One relevant item per instance (this is the standard leave-one-out protocol), true rank $r \in \{1, \dots, n\}$. The metrics simplify beautifully:

$$\mathrm{AUC}(r) = \frac{n-r}{n-1}, \quad \mathrm{Prec}_k(r) = \frac{\delta(r \le k)}{k}, \quad \mathrm{Recall}_k(r) = \delta(r \le k)$$
$$\mathrm{AP}(r) = \frac{1}{r}, \quad \mathrm{NDCG}(r) = \frac{1}{\log_2(r+1)}$$

With one relevant item, Reciprocal Rank $=$ AP, Hit Ratio $=$ Recall, Accuracy $=$ Recall@1.

### The rank distribution under sampling

This is the key derivation. Draw one irrelevant item uniformly. It outranks the relevant item with probability

$$p(j < r) = \frac{r-1}{n-1}$$

Draw $m$ of them with replacement and count how many outrank it. That is a binomial:

$$\tilde r \sim B\!\left(m, \frac{r-1}{n-1}\right) + 1$$

So $\tilde r$ lives in $\{1, \dots, m+1\}$ and $\mathbb{E}[\tilde r] = 1 + m\frac{r-1}{n-1}$. Every expected sampled metric is then $\mathbb{E}[M(\tilde r)] = \sum_{i=1}^{m+1} p(\tilde r = i) M(i)$. (Without replacement it is hypergeometric; the results barely change.)

### Why AUC survives and nothing else does

AUC is **linear in $r$**: $\mathrm{AUC}_n(r) = \text{const}_1 \cdot r + \text{const}_2$. Linearity plus $\mathbb{E}[\tilde r] = 1 + m\frac{r-1}{n-1}$ gives, exactly,

$$\mathbb{E}[\mathrm{AUC}_{m+1}(\tilde r)] = \mathrm{AUC}_{m+1}(\mathbb{E}[\tilde r]) = \frac{n-r}{n-1} = \mathrm{AUC}_n(r)$$

Unbiased, and independent of $m$. Every other metric is nonlinear, so Jensen bites and the expectation does not pass through.

For cut-off metrics you get a binomial CDF:
$$\mathbb{E}[\mathrm{Recall}_k(\tilde r)] = \mathrm{CDF}\!\left(k-1; m, \tfrac{r-1}{n-1}\right)$$
A step function has become a smooth sigmoid in $r$. The "top 10" question is no longer being asked.

For AP, closed form (for $r > 1$):
$$\mathbb{E}[\mathrm{AP}(\tilde r)] = \frac{1 - \left(\frac{n-r}{n-1}\right)^{m+1}}{(r-1)\frac{m+1}{n-1}} = \frac{1 - \mathrm{AUC}_n(r)^{m+1}}{r-1}\cdot\text{const}$$
Note the shape: sampled AP only looks like true AP ($\propto 1/(r-1)$) when $\mathrm{AUC}_n(r)^{m+1} \approx 0$. But a *good* recommender has AUC near 1, so that term stays near 1 and needs enormous $m$ to vanish. **Sampled metrics are worst exactly for the good recommenders.**

### The $m=1$ collapse

$\mathbb{E}[M(\tilde r)] = \frac{n-r}{n-1}(M(1) - M(2)) + M(2) = r\,\text{const}_1 + \text{const}_2$. Linear in $r$ for *any* $M$. So at $m=1$ no metric carries information the others do not, and all of them are affine reskins of exact AUC. Precision/Recall at $k \ge 2$ are literally constant — zero information.

### The corrections

The naive sampled metric is $\hat M(\tilde r) = M(\tilde r)$, which applies the metric to a rank that is systematically far too small. Three fixes, in order of effort:

**1. Rank estimate.** $\tilde r$ is a bad estimate of $r$, but $\frac{\tilde r - 1}{m}$ is an unbiased estimate of $p = \frac{r-1}{n-1}$. So scale it back up:
$$\hat M(\tilde r) = M\!\left(\left\lfloor 1 + \frac{(n-1)(\tilde r - 1)}{m} \right\rfloor\right)$$
One line of code. Still biased whenever $M$ is nonlinear (unbiased rank $\ne$ unbiased metric), but far better than nothing.

**2. Minimal-bias least squares.** Treat $\hat M$ as a free vector in $\mathbb{R}^{m+1}$ — one learned value per possible sampled rank — and fit it to minimise squared bias against the true metric:
$$\arg\min_{\hat M \in \mathbb{R}^{m+1}} \sum_{r=1}^{n} p(r)\left(\sum_{\tilde r} p(\tilde r \mid r)\hat M_{\tilde r} - M(r)\right)^2$$
Solution $\hat M = (A^\top A)^{-1}A^\top b$ with $A_{r,\tilde r} = \sqrt{p(r)}\,p(\tilde r \mid r)$ and $b_r = \sqrt{p(r)}M(r)$. $p(r)$ is a prior over true ranks — they use uniform, since the true rank distribution is recommender-dependent and unknown. Adding the monotonicity constraint $\hat M_{\tilde r} \ge \hat M_{\tilde r + 1}$ turns it into isotonic regression; that is **CLS** in the tables. Note $m+1 < n$ means the system is under-determined — you cannot get zero bias at every $r$.

**3. Bias–variance trade-off (BV $\gamma$).** Pure minimal-bias solutions oscillate wildly across $\tilde r$ (visible in Fig. 4) and have high variance. So regularise:
$$\arg\min_{\hat M} \sum_{r=1}^n p(r)\Big[(\mathbb{E}[\hat M_{\tilde r}\mid r] - M(r))^2 + \gamma \operatorname{Var}[\hat M_{\tilde r}\mid r]\Big]$$
with solution $\hat M = \big((1-\gamma)A^\top A + \gamma\,\mathrm{diag}(c)\big)^{-1}A^\top b$, $c_{\tilde r} = \sum_r p(r)p(\tilde r \mid r)$. $\gamma = 0$ recovers minimal bias; $\gamma = 1$ gives the posterior-mean estimator $\hat M_{\tilde r} = \sum_r p(r \mid \tilde r)M(r)$. Since you average over thousands of evaluation instances anyway, variance is cheap — small $\gamma$ wins.

### Experimental setup

Deliberately copied from the [[Neural Collaborative Filtering vs. Matrix Factorization Revisited|NCF]] protocol: binarised MovieLens 1M, leave-last-event-out, $m = 100$, 6040 test users, HR@10 and NDCG@10, plus AP and AUC. Three recommenders, anonymised to stop you arguing about the algorithms:

- **X** — [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)|matrix factorization]], $d=16$, $\lambda=10$, $\alpha=0.2$, trained with implicit ALS on the all-pairs squared loss $\sum_{(u,i)\in H}(\hat y - 1)^2 + \alpha\sum_{u,i}\hat y^2 + \lambda\|V\|_F^2$.
- **Y** — [[Amazon.com Recommendations- Item-to-Item Collaborative Filtering (IEEE Internet Computing)|item-based CF]], cosine similarity sharpened by exponent $q=3$, no kNN sparsification.
- **Z** — same, $q=1$, sparsified to $k'=10$ nearest neighbours.

Each sampling experiment is repeated 100 times to measure variance — something a real study doing this for speed would never do.

## Ablation Studies and Experiments

### The toy example (this is the whole paper in one table)

$n = 10{,}000$, five instances, $m = 99$ sampled negatives.

| | Predicted ranks | AUC | AP | NDCG | Recall@10 |
|---|---|---|---|---|---|
| A | 100, 100, 100, 100, 100 | 0.990 | **0.010** | 0.150 | 0.000 |
| B | 40, 40, 8437, 9266, 4482 | 0.555 | **0.010** | 0.122 | 0.000 |
| C | 212, 2, 743, 5342, 1548 | 0.843 | **0.101** | 0.208 | 0.200 |

Exact: C is best on every top-heavy metric, with 10× the AP of A and B.

Sampled ($\pm$ over 1000 repeats):

| | AUC | AP | NDCG | Recall@10 |
|---|---|---|---|---|
| A | 0.990±0.004 | **0.630±0.129** | 0.724±0.097 | 1.000±0.000 |
| B | 0.555±0.014 | 0.336±0.073 | 0.444±0.054 | 0.400±0.000 |
| C | 0.843±0.014 | **0.325±0.050** | 0.460±0.039 | 0.567±0.092 |

C went from best to worst on AP. A and B are tied on exact AP but A looks 2× better sampled. Recall@10 says A is *perfect* — it never once ranked an item in the true top 10. AUC is the only row that survives.

Sweeping $m$ (Fig. 2) is worse still: for AP you can get **any** ordering you like by choosing the sample size. $m < 50$ → A > C > B. $m \approx 200$ → A > B > C. $m \approx 500$ → C > A > B. Large $m$ → C > A ≈ B. Recall@10 needs $m \approx 5000$ out of $n = 10{,}000$ — a 50% sampling rate — before it becomes consistent. At that point you have saved nothing.

### The real experiment: MovieLens 1M

Rank distributions first (Fig. 5): Z is best in the top 10 but dumps the relevant item into the worst bucket for over 1600 of 6040 users. X is flat — 2310 items in the top 100, fewer than 300 in the bottom half. Y sits between. These are genuinely different shapes, which is what makes the comparison interesting.

Exact vs uncorrected sampled ($m=100$), all values ×100:

| Metric | | Exact | Sampled |
|---|---|---|---|
| Recall@10 | X | 7.60 | **66.19±0.25** |
| | Y | 8.84 | 56.51±0.22 |
| | Z | **9.42** | 54.20±0.22 |
| NDCG@10 | X | 3.76 | **39.21±0.20** |
| | Y | 4.59 | 34.82±0.16 |
| | Z | **4.79** | 35.34±0.16 |
| AP | X | 3.75 | **32.55±0.21** |
| | Y | 4.32 | 30.01±0.20 |
| | Z | **4.44** | 30.71±0.21 |
| AUC | X | 89.13 | 89.12±0.04 |
| | Y | 85.33 | 85.33±0.04 |
| | Z | 74.73 | 75.04±0.23 |

The ordering is **exactly reversed** for Recall. The worst recommender under every exact top-heavy metric becomes the best under every sampled one. And look at the standard deviations: ±0.25 on a value that is wrong by 58 points. AUC reproduces to two decimal places.

### Do the corrections work?

Table 4 is the real ablation: over 100 repeats, how often does each method get a pairwise ordering right? 100 = always correct, 0 = always wrong.

| Pair | Metric | Uncorr. | Rank Est. | CLS | BV 1 | **BV 0.1** | BV 0.01 | BV 0.001 |
|---|---|---|---|---|---|---|---|---|
| X vs Y | Recall | 0 | 31 | 31 | 11 | **93** | 91 | 78 |
| | NDCG | 0 | 31 | 31 | 15 | **93** | 88 | 76 |
| | AP | 0 | 24 | 31 | 0 | **68** | 79 | 66 |
| X vs Z | Recall | 0 | 100 | 100 | 100 | **100** | 95 | 86 |
| | AP | 0 | 100 | 100 | 61 | **99** | 92 | 81 |
| Y vs Z | Recall | 0 | 100 | 100 | 100 | **95** | 72 | 68 |
| | NDCG | 100 | 100 | 100 | 100 | 94 | 70 | 67 |
| all | AUC | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

What the ablation actually reveals:

- **Uncorrected is not merely noisy, it is anti-correlated.** Scores of 0, not 50. A coin flip would beat it.
- **$\gamma$ traces the bias–variance curve exactly as predicted.** $\gamma = 1$ (pure variance minimisation, the posterior mean) is *terrible* — it shrinks all three recommenders toward the prior mean and scores 0–15 on X vs Y. $\gamma \to 0.001$ (pure bias minimisation) has the right expectation but variance so large that single runs flip, dropping to 66–86. $\gamma = 0.1$ is the sweet spot at >90% on all but one comparison.
- **The hardest pair is X vs Y** — exact Recall 7.60 vs 8.84, a genuinely small gap. Corrections that trivially fix the wide gaps (X vs Z) still struggle here, which is precisely the regime where papers claim their wins.
- **Rank estimate is shockingly good for how trivial it is.** Strictly better than uncorrected on every single comparison. Its embarrassing side effect: under rank-estimate correction, Recall@10 and NDCG@10 become *numerically identical* (17.46 / 17.26 / 18.67 for X/Y/Z) — the correction has collapsed two metrics into one, which is itself a symptom of the underlying problem.
- **Sample size does not rescue you.** Uncorrected Recall@10 needs $m > 1000$ (a third of the catalogue) to order X and Y correctly even in expectation. BV 0.1 gets it right at $m = 60$.

### What did not work

- **Minimal-bias LS and CLS alone.** Unbiased in expectation, but the fitted $\hat M$ oscillates violently across $\tilde r$ (some values go *negative* for AP, Fig. 4) and single-run variance destroys the gain. Regularisation was mandatory.
- **$\gamma = 1$ (plain least squares / posterior mean).** Lowest variance, largest bias, worst ordering accuracy of any correction. On X vs Y AP it scored 0 — as bad as no correction at all.
- **Getting a correct prior $p(r)$.** The true rank distribution is recommender-dependent and unknown at evaluation time, so they use a uniform prior. That is a known weakness, not a solved problem.

## Worth Remembering

**The one-line rule:** if you sample negatives at evaluation time, you are not measuring Recall@10. You are measuring a smeared, near-linear function of rank that is closer to AUC than to the metric you named. Do not sample. If you must, apply the rank-estimate correction at minimum — it is one line and strictly dominates doing nothing.

**Low variance is the trap.** The tight error bars in Table 3 are real and meaningless. This is the cleanest illustration in the recsys literature that a reproducible number and a correct number are different things. If your evaluation has a bias that depends on the system under test, repeating it does not help. Compare with [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] and [[On the Difficulty of Evaluating Baselines]] — together these three explain most of why the deep-recsys literature of 2016–2019 does not replicate.

**Why the bias favours certain models.** Sampling rewards recommenders with a *flat* rank distribution (X: few disasters, few triumphs) and punishes ones that are sharp in the head but bad in the tail (Z). Against 100 random negatives, the relevant item usually beats all of them regardless of whether its true rank was 100 or 2 — so the metric stops rewarding head accuracy. Matrix factorization is often the flat one, which is at least suggestive about why [[Neural Collaborative Filtering vs. Matrix Factorization Revisited]] (Rendle again) found MF beating NCF once evaluation was done properly.

**Practical caveats:**
- Analysis assumes **exactly one relevant item** per instance. The multi-relevant case is handled by assuming observed ranks are independent, which they are not. Removing that assumption is left as open work.
- The corrections need $n$ and the binomial $p(\tilde r \mid r)$, so you must know your catalogue size. That is cheap.
- If you use corrections, **re-run with different negative-sampling seeds** and report that variance. You are trading bias for variance; you should be prepared to fail to find a significant difference. The authors are explicit that if a difference survives the inflated variance, that is *stronger* evidence than the uncorrected result ever was.
- The bias cannot be eliminated by any correction. Only by not sampling.

**Cost context.** Full evaluation on MovieLens 1M is 6040 users × ~3700 items — a single dense matmul. The sampling shortcut was never justified at academic scale; it was copied from paper to paper because a prior paper did it. Sampling negatives during *training* ([[Distributed Representations of Words and Phrases (negative sampling)|negative sampling]], [[Dense Passage Retrieval (DPR)|in-batch negatives]]) is a completely separate and well-founded practice — the confusion between the two is part of how this got normalised.

**Open question worth chasing:** everything here concerns offline metric estimation bias. The neighbouring bias — that logged feedback is missing-not-at-random — is handled by [[Recommendations as Treatments- Debiasing Learning and Evaluation]] and [[Unbiased Learning-to-Rank with Biased Feedback]]. A study that sampled negatives *and* ignored propensities is compounding two independent biases in the same number.

## Links

Related: [[NDCG]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Neural Collaborative Filtering vs. Matrix Factorization Revisited]] · [[On the Difficulty of Evaluating Baselines]] · [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[Unbiased Learning-to-Rank with Biased Feedback]] · [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)]] · [[Amazon.com Recommendations- Item-to-Item Collaborative Filtering (IEEE Internet Computing)]] · [[Collaborative Filtering for Implicit Feedback Datasets (ICDM)]] · [[BPR- Bayesian Personalized Ranking from Implicit Feedback]] · [[Self-Attentive Sequential Recommendation (SASRec)]] · [[BERT4Rec- Sequential Recommendation with Bidirectional Transformer]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[Dense Passage Retrieval (DPR)]] · [[Uncertainty]] · [[Random variable]] · [[Troubling Trends in Machine Learning Scholarship]]

New topics worth writing: Isotonic regression, Binomial and hypergeometric rank distributions, Bias–variance decomposition of an estimator, Leave-one-out evaluation protocols in recsys, Mean reciprocal rank, Jensen's inequality and nonlinear metrics, Plackett–Luce and rank distributions
