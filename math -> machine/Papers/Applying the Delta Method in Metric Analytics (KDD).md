---
title: "Applying the Delta Method in Metric Analytics (KDD)"
authors: ["Deng", "Knoblich & Lu"]
year: 2018
arxiv: "1803.06336"
url: https://arxiv.org/abs/1803.06336
priority: Good-To-Read
read_on: 2026-09-25
tags: [paper, theory]
---
## The Core Idea

Most online metrics are not simple averages. They are **ratios** of averages (clicks per page-view), **quantiles** (95th percentile page load time), or averages computed on units that are not the unit you randomised. The central limit theorem gives you error bars for a simple average of independent things and nothing else. So the industry reaches for heavy machinery: bootstrap, mixed effect models, kernel density estimation. All of them are slow, and some of them are quietly wrong.

The claim of this paper is that one 100-year-old tool — the **Delta method** — covers all four of these cases with a closed-form formula that is a handful of sums and sums-of-squares, which means it parallelises trivially and costs almost nothing.

> [!NOTE] Delta method
> If a statistic $T_n$ is asymptotically normal around $\theta$, then any smooth function of it, $\phi(T_n)$, is *also* asymptotically normal. Take a first-order Taylor expansion, $\phi(T_n)-\phi(\theta)\approx\phi'(\theta)(T_n-\theta)$, and the variance falls out: $\sqrt{n}\{\phi(T_n)-\phi(\theta)\}\to N(0,\phi'(\theta)^2)$. ^delta-method

The insight that makes the paper more than a textbook recap: **model the metric, not the data**. Instead of writing a probabilistic model for each user and fitting it, treat the metric as a function of a few sufficient statistics (sums, sums of squares, counts), which are cheap and distributable, and then push the normal approximation through that function. Big data becomes a small problem. The hard part is never the maths — it is spotting the right link function $\phi$ for your metric.

Why did it not already exist as a stated practice? It did, scattered in appendices. The paper's actual contribution is (a) making the recipe explicit for four common cases, (b) a new trick for quantile confidence intervals that avoids density estimation entirely, and (c) a data-augmentation trick that handles missing data without modelling the missingness.

## The Methodology

### 1. Percent change (ratio metrics)

You want a confidence interval for $\Delta\% = (\mu_y-\mu_x)/\mu_x$, estimated by $\bar Y/\bar X - 1$. The classic answer is **Fieller's interval**, a page-long formula that needs an extra nuisance parameter $g = n s_x^2 t^2_{\alpha/2}(r)/\bar X^2$.

The Delta method: set $\phi(x,y)=y/x$, expand, and you get

$$\frac{\bar Y}{\bar X}-\frac{\mu_y}{\mu_x}\approx\frac{1}{\mu_x}(\bar Y-\mu_y)-\frac{\mu_y}{\mu_x^2}(\bar X-\mu_x)$$

which is just the average of i.i.d. quantities $W_i = Y_i/\mu_x - \mu_y X_i/\mu_x^2$. The interval is

$$\underbrace{\frac{\bar Y}{\bar X}-1}_{\text{point estimate}} \pm \underbrace{\frac{z_{\alpha/2}}{\sqrt n\,\bar X}\sqrt{s_y^2-2\frac{\bar Y}{\bar X}s_{xy}+\frac{\bar Y^2}{\bar X^2}s_x^2}}_{\text{uncertainty}}$$

Three sample moments and you are done. Fieller's interval converges to exactly this as $n\to\infty$ (because $t_{\alpha/2}(r)\to z_{\alpha/2}$ and $g\to 0$).

Two optional refinements: an **Edgeworth correction** replacing $\pm z_{\alpha/2}$ with quantiles of a skew-corrected distribution, $F(t)=\Phi(t)-6n^{-1/2}\kappa_w(t^2-1)\phi(t)$, where $\kappa_w$ is the skewness of the $W_i$; and a **second-order bias correction** added to the point estimate, computed in code as $\bar Y s_x^2/(n\bar X^3) - s_{xy}/(n\bar X^2)$.

### 2. Cluster randomisation

You randomise by user but your metric is per-page-view. The observations are not independent — page-views inside a user are correlated.

The trick is to rewrite the metric so the i.i.d. units are the *randomisation* units. With $K$ clusters, cluster $i$ having $N_i$ observations and within-cluster sum $S_i=\sum_j Y_{ij}$:

$$\bar Y = \frac{\sum_i S_i / K}{\sum_i N_i / K} = \frac{\bar S}{\bar N}$$

That is a ratio of two averages of i.i.d. cluster-level quantities. Same formula as case 1:

$$\mathrm{Var}(\bar Y)\approx\frac{1}{K\mu_N^2}\left(\sigma_S^2 - 2\frac{\mu_S}{\mu_N}\sigma_{SN} + \frac{\mu_S^2}{\mu_N^2}\sigma_N^2\right)$$

A practical read-out from the formula: $\sigma_N^2$, the variance of **cluster size**, feeds directly into the metric's variance. Uneven cluster sizes cost you statistical power. Make clusters as equal-sized as you can.

### 3. Quantile metrics

The textbook result is $\sqrt n\{X_{\lfloor np\rfloor}-F^{-1}(p)\}\to N[0,\ \sigma^2/f\{F^{-1}(p)\}^2]$ with $\sigma^2=p(1-p)$. The killer is the denominator: you need the unknown density $f$ at the unknown quantile. In the tail the density decays toward zero, so a small bias in $\hat f$ becomes a large bias in the variance.

The fix is **outer confidence intervals**, which never estimate $f$. Let $Y_i = I\{X_i \le F^{-1}(p)\}$. Then $\sum Y_i$ counts observations below the quantile and $\bar Y \approx p$ with variance $\sigma^2/n$. Invert that into a confidence interval on the *rank*, not the value:

$$L,U = n\left(p \pm z_{\alpha/2}\,\sigma/\sqrt n\right)$$

then just fetch the order statistics $X_{(L)}$ and $X_{(U)}$. No density, no sorting the whole dataset — sketching algorithms narrow the range first.

The novel bit: if observations are **clustered**, the only change is $\sigma^2$, which you get from the cluster formula (6) instead of $p(1-p)$. And because the correction only touches the numerator, you can apply it *after* the fact. Two algorithms:

- **Pre-adjustment**: compute $\sigma$ first, then fetch the three ntiles. Two passes over the data.
- **Post-adjustment**: fetch all three ntiles at once using naive $p(1-p)$ ranks, then rescale $X_{(L)},X_{(U)}$ by the factor $\sigma/\sqrt{p(1-p)}$. One ntile stage. Shipped in Microsoft's ExP on Spark and SCOPE.

### 4. Missing data in within-subject studies

Tracking visits-per-user week over week, you need $\mathrm{Cov}(\bar X_t, \bar X_{t-1})$. But users appear in some weeks and not others, and missingness correlates with engagement — it is *not* missing at random. Using only complete cases is biased.

**Data augmentation**: for every user $i$ and period $t$, replace the scalar $X_{it}$ with the pair $(I_{it}, X_{it})$, where $I_{it}=1$ if the user appeared and $X_{it}=0$ when they did not. Now

$$\bar X_t = \frac{\sum_i X_{it}}{\sum_i I_{it}} = \frac{\bar X_t}{\bar I_t}$$

is a ratio again, and crucially **there is no missing data any more** — every user has a row in every period. The vector $(\bar I_t,\bar X_t,\bar I_{t'},\bar X_{t'})$ is jointly normal, you estimate its covariance $\Sigma$ directly, and push it through the Delta expansion.

For a cross-over design (group I gets treatment then control, group II the reverse), the mean vector is $\boldsymbol\mu(\boldsymbol\theta)=(\theta_1+\Delta,\ \theta_2,\ \theta_1,\ \theta_2+\Delta)$, and since it is linear in $\boldsymbol\theta$, generalised least squares gives $\hat{\boldsymbol\theta}=(M^T\Sigma_X^{-1}M)^{-1}M^T\Sigma_X^{-1}\mathbf X$ with $\mathrm{Var}(\hat{\boldsymbol\theta})=(M^T\Sigma_X^{-1}M)^{-1}$. The model being fitted is a four-parameter toy. All the work is in $\Sigma$, which is trivially distributed.

## Ablation Studies and Experiments

### Ratio intervals — Delta vs Fieller (Table 1, 10,000 replicates, nominal 95%)

| Model | $n$ | Fieller | Delta | Delta+BC | Edgeworth | Edgeworth+BC |
|---|---|---|---|---|---|---|
| Normal | 20 | 0.9563 | 0.9421 | 0.9422 | 0.9426 | 0.9426 |
| Bernoulli | 20 | 0.9539 | 0.9403 | 0.9490 | 0.9476 | 0.9521 |
| Poisson | 20 | 0.9400 | 0.9322 | 0.9370 | 0.9341 | 0.9396 |
| all | 200 | ≈0.950 | ≈0.949 | ≈0.949 | ≈0.950 | ≈0.950 |
| all | 2000 | ≈0.950 | ≈0.950 | ≈0.950 | ≈0.950 | ≈0.950 |

At $n\ge200$ every method is indistinguishable. Fieller only wins at $n\le50$, and the bias correction closes most of that gap (Bernoulli $n=20$: 0.9403 → 0.9490). **Verdict: the extra complexity of Fieller buys nothing at web scale.** Recommendation is Delta + bias correction.

### Cluster randomisation — Delta vs mixed effect model (Table 2)

1000 clusters of three sizes (Poisson means 2, 5, 30) with different per-cluster Bernoulli rates (0.3, 0.5, 0.8). True $\mathbb E(\bar Y)=0.667$, true SD 0.00895.

| Method | Estimate | Avg. SE |
|---|---|---|
| Naive (pretend i.i.d.) | 0.667 | 0.00522 |
| Mixed effect | **0.547** | 0.00956 |
| Delta method | 0.667 | 0.00908 |

This is the sharpest result in the paper. The naive estimator gets the point right but **understates the standard error by 42%** — you will ship false positives. The mixed effect model gets the variance right but the **point estimate is catastrophically biased** (0.547 vs 0.667). Why: the random-effects intercept $\alpha$ estimates the *average of per-cluster means*, weighting every cluster equally. The metric you actually care about weights by cluster size. These agree only if cluster sizes are equal or effects are homogeneous — neither is true in practice. Re-weighting the mixed effect estimates by $N_i/\sum N_i$ afterwards recovers 0.662, but then you have no easy variance for it.

The same logic applies to ATE, not just the mean: heterogeneous per-cluster treatment effects plus unequal cluster sizes ⇒ biased mixed-effect ATE.

### Quantile intervals — outer CI vs bootstrap (Table 3, 95th percentile, 10,000 runs)

| Distribution | $N_u$ | Bootstrap (1000 reps) | Outer CI (pre-adj) | Outer CI (post-adj) |
|---|---|---|---|---|
| Normal | 100 | 0.9039 | 0.9465 | 0.9369 |
| Normal | 1000 | 0.9500 | 0.9549 | 0.9506 |
| Normal | 10000 | 0.9500 | 0.9500 | 0.9482 |
| Log-normal | 100 | 0.8551 | 0.9198 | 0.9049 |
| Log-normal | 1000 | 0.9403 | 0.9474 | 0.9421 |
| Log-normal | 10000 | 0.9458 | 0.9482 | 0.9479 |

**Bootstrap loses, and loses badly at small $n$** — 0.8551 coverage on heavy-tailed data with 100 clusters, against 0.9198 for outer CI. And it was ~20× slower in their unoptimised simulation. Pre-adjustment is consistently slightly more conservative than post-adjustment; the gap widens as $n$ shrinks, because with unadjusted ranks the order statistics $X_{(L)}, X_{(U)}$ are more likely to coincide with the quantile value itself, which understates variance. Guidance: post-adjustment above 1000 clusters, pre-adjustment below if accuracy matters.

### Missing data — Delta vs mixed effect (Tables 4–5)

2000 users, cross-over design, user effect $u_i\sim N(10,3)$, treatment effect scaled by engagement level $l_i$, and probability of being missing $=1-\max(0.1,l_i)$. So heavy users get bigger effects *and* are less likely to be missing. True ATE 6.592.

| Method | Estimate | Avg. SE | True SD |
|---|---|---|---|
| Mixed effect | 7.129 | 0.1261 | 0.1295 |
| Delta method | 6.593 | 0.1568 | 0.1573 |

Both variance estimates are honest. But the mixed effect estimate is biased upward by 8%. The diagnosis (Table 5) is the interesting part:

| | Estimate | Var |
|---|---|---|
| Mixed effect, all data | 7.1290 | 0.00161 |
| Mixed effect, complete-data users only | 7.3876 | 0.00180 |
| Linear model, single-appearance users | 5.1174 | 0.01133 |
| Inverse-variance weighted average of the two | 7.0766 | 0.00155 |

The mixed effect model *assumes* one fixed treatment effect, so it pools the two subgroups with inverse-variance weights. The complete-data group has far lower variance (within-subject comparison cancels the user effect), so it dominates — and that group is disproportionately heavy users, who have the largest effects. **The mixed effect model's efficiency gain is exactly what makes it biased here.** It is not a bug in the fit; it is the homogeneity assumption doing the damage.

## Worth Remembering

- The mixed effect model is not a drop-in replacement for the Delta method. It estimates a **cluster-weighted** quantity; your metric is usually **observation-weighted**. Deng notes the Delta method is, in the Gaussian case, "the ultimate simplification of GEE's sandwich variance estimator after summarising data points into sufficient statistics" — but you can derive it in three lines instead of a chapter.

- **Cluster-size variance is a power lever.** $\sigma_N^2$ enters the variance formula directly. If you can make your randomisation clusters more uniform in size, do it — it is free sensitivity.

- The quantile trick generalises further than the paper shows: any statistic whose asymptotic variance factorises into (a nuisance term you cannot estimate) × (a term you can) can get a post-hoc rescaling of a distribution-free interval.

- **What is not claimed:** the paper is honest that the Delta method does not replace good design. Variance reduction ([[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)|CUPED]]), cluster configuration choice, and metric transformations are all still needed and are named as future work.

- Fieller's interval is genuinely better below $n\approx50$. If you are analysing a small holdout or a rare segment, do not blindly use the asymptotic formula.

- Practical caveat for the bias correction: the paper's text and its Algorithm 1 differ in the power of $n$ (text has $(n\bar X)^2$, pseudocode divides by $n$ once). Use the pseudocode version, which is what shipped: `bc = Ȳ/X̄³·s_x²/n − s_xy/(n·X̄²)`.

- Connection worth chasing: the ratio-of-averages structure here is exactly the shape of the self-normalised importance sampling estimator in [[Off-Policy Evaluation|OPE]]. The Delta method variance for $\bar S/\bar N$ is the same object as the variance of [[The Self-Normalized Estimator for Counterfactual Learning (SNIPS) (NeurIPS)|SNIPS]], where the denominator is a random sum of weights rather than a random count.

- Open question the paper leaves: what happens to the augmentation trick when missingness is *caused by* the treatment? The cross-over analysis assumes no carryover and no differential attrition; both are common in real products.

- Code released at `https://aka.ms/exp/deltamethod`.

## Links

Related: [[AB Testing]] · [[Controlled experiments on the web- survey and practical guide (DMKD)]] · [[Trustworthy Online Controlled Experiments- Five Puzzling Outcomes Explained (KDD)]] · [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)]] · [[Graph cluster randomization- network exposure to multiple universes (KDD)]] · [[Design and Analysis of Switchback Experiments]] · [[Always Valid Inference- Bringing Sequential Analysis to A-B Testing]] · [[The Self-Normalized Estimator for Counterfactual Learning (SNIPS) (NeurIPS)]] · [[Off-Policy Evaluation]] · [[Panel Regression]] · [[Monte Carlo Methods]] · [[Uncertainty]] · [[Derivative]] · [[Random variable]] · [[Recursive Partitioning for Heterogeneous Causal Effects]]

New topics worth writing: Delta method, Fieller's theorem, mixed effect models and random intercepts, generalized estimating equations (GEE) and sandwich variance, intra-cluster correlation coefficient, order statistics and distribution-free quantile intervals, Edgeworth expansion, kernel density estimation bias-variance tradeoff, missing-at-random vs missing-not-at-random, cross-over and repeated-measures designs, bag of little bootstraps
