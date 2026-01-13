---
title: "Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)"
authors: ["Deng", "Xu", "Kohavi & Walker"]
year: 2013
url: https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf
priority: Must-Read
read_on: 2026-08-22
tags: [paper, theory]
---
## The Core Idea

An A/B test compares an average in treatment against an average in control. The test is only as good as the **noise** in those averages. If queries-per-user has a huge spread — some people search twice a month, some search 200 times a day — then the difference between the two group means is buried under that spread, and you need enormous traffic to see a 0.5% effect.

The insight: **most of that spread has nothing to do with your experiment.** A heavy user was already a heavy user last month. That is a fact about the person, not about your treatment. So you can go and measure it *before* the experiment started, and subtract it out.

Formally, take a covariate $X$ measured in the pre-experiment window (e.g. the same metric, queries-per-user, from last week). Instead of estimating the mean of $Y$ directly, estimate

$$\hat{Y}_{cv} = \bar{Y} - \theta \bar{X} + \theta\, \mathbb{E}(X)$$

The last two terms cancel in expectation, so the estimate is still unbiased. But its variance is

$$\mathrm{var}(\hat{Y}_{cv}) = \mathrm{var}(\bar{Y})\,(1-\rho^2)$$

where $\rho = \mathrm{cor}(Y, X)$. Correlation of 0.7 kills half the variance.

Why this works in an experiment specifically, and this is the whole trick: normally control variates need you to *know* $\mathbb{E}(X)$, which you never do. But here you take the difference between two groups:

$$\Delta_{cv} = \hat{Y}^{(t)}_{cv} - \hat{Y}^{(c)}_{cv}$$

and since users were randomised, $\mathbb{E}(X^{(t)}) = \mathbb{E}(X^{(c)})$ in the pre-period — the treatment did not exist yet. The unknown $\mathbb{E}(X)$ terms cancel exactly. You never need to know it.

What this unlocks: at Bing, ~50% variance reduction. Same statistical power with **half the users, or half the duration**. Three named experiments gave 45%, 52%, 49% reduction with one week of experiment and one week of pre-period.

Why it did not exist before in this form: the standard answer was linear regression / ANCOVA, which assumes $\mathbb{E}(Y_i \mid Z_i, X_i) = \theta_0 + \delta Z_i + \theta^T X_i$ and equal residual variance — [[Regression Analysis|assumptions]] that are wrong for web metrics. CUPED needs no model of $Y$ at all. The reduction factor $1-\rho^2$ holds regardless of distribution shape.

> [!NOTE] CUPED ^cuped
> Controlled-experiment Using Pre-Experiment Data. Adjust the treatment–control delta by a covariate measured before the experiment began. Unbiased by construction (randomisation makes the covariate's group means equal), and variance drops by $1-\rho^2$.

## The Methodology

The baseline is the two-sample t-test. With metric $Y$:

$$t = \frac{\bar{Y}^{(t)} - \bar{Y}^{(c)}}{\sqrt{\mathrm{var}(\bar{Y}^{(t)} - \bar{Y}^{(c)})}}, \qquad \mathrm{var}(\Delta) = \mathrm{var}(\bar{Y}^{(t)}) + \mathrm{var}(\bar{Y}^{(c)})$$

Sample sizes are in the thousands, so the Central Limit Theorem handles normality. Shrink the denominator, get a bigger $t$, get more sensitivity.

The paper gives two routes to the same place.

**Route 1 — stratification.** Split users into $K$ groups (strata) by a pre-experiment variable, e.g. which browser they used. Weight $w_k$ = probability of falling in stratum $k$. Estimate

$$\hat{Y}_{strat} = \sum_{k=1}^{K} w_k \bar{Y}_k$$

The decomposition:

$$\mathrm{var}(\bar{Y}) = \underbrace{\sum_k \frac{w_k}{n}\sigma_k^2}_{\text{within-strata}} + \underbrace{\sum_k \frac{w_k}{n}(\mu_k - \mu)^2}_{\text{between-strata}} \;\ge\; \mathrm{var}(\hat{Y}_{strat})$$

Stratifying deletes the between-strata term. The analogy in the paper: children's heights vary a lot overall, but very little within a single age group. Problem: you need the weights $w_k$, which you have to estimate from users outside the experiment.

**Route 2 — control variates.** As above. Optimal coefficient:

$$\theta = \frac{\mathrm{cov}(Y, X)}{\mathrm{var}(X)}$$

which is exactly the OLS slope of centred $Y$ on centred $X$, giving $\mathrm{var}(\hat{Y}_{cv}) = \mathrm{var}(\bar{Y})(1 - R^2)$ — see [[R-Squared]]. No weights needed. Handles continuous covariates. **Appendix A proves that for categorical $X$, the two routes give numerically identical estimates**, so control variates is strictly the more general framing.

Nonlinear version is allowed too: $\hat{Y}_{cv} = \bar{Y} - \overline{f(X)} + \mathbb{E}(f(X))$, optimal $f(X) = \mathbb{E}(Y \mid X)$, i.e. the regression function. They do not use it in production.

**Practical rules.**

- **Pick $X$ = the same metric from the pre-period.** Queries-per-user in the experiment ← queries-per-user in the previous 1–2 weeks. Highest correlation by far.
- **One $\theta$ for both arms.** If you fit $\theta$ separately for treatment and control the estimator becomes biased. Estimate it from the pooled population.
- **Missing pre-period data.** New users, infrequent users, cookie churn — many experiment users have no pre-period row. Fix: add a binary covariate "did this user appear in the pre-period", then set the missing $X$ values to any constant. This is equivalent to first stratifying into "seen before" / "not seen before", and that stratification is itself an extra source of variance reduction.
- **Non-user metrics.** CTR = total clicks / total pageviews. Randomisation unit (user) ≠ analysis unit (page). You need the **delta method** — linearise the ratio by Taylor expansion, then compute $\theta$ from the multivariate normal covariance matrix $\Sigma$ of $(\bar{Y}, \bar{N}, \bar{X}, \bar{M})$ where $N, M$ are pageview counts:
$$\theta = \frac{\beta_1^T \Sigma \beta_2}{\beta_2^T \Sigma \beta_2}, \quad \beta_1 = (1/\mu_N, -\mu_Y/\mu_N^2, 0, 0)^T,\; \beta_2 = (0,0,1/\mu_M, -\mu_X/\mu_M^2)^T$$
This also opens up page-level covariates (e.g. timestamp of a pageview) rather than only user-level ones.

**The one hard constraint:** $X$ must be unaffected by the treatment. Pre-experiment data guarantees this. But so does anything fixed before a user *triggers* the feature — e.g. the day-of-week a user first entered the experiment. That matters for low-trigger-rate experiments.

## Ablation Studies and Experiments

**The headline experiment — the Bing slowdown test.** They deliberately added 250 ms of server delay. Ran two weeks on a small slice of traffic; the CTR effect came out *borderline* significant, p just under 0.05. A much larger rerun confirmed the effect was real (p = 2e-13). Applying CUPED with 2-week pre-period CTR as covariate: **the delta was significant from day 1**, and the whole p-value curve sat below the 0.05 line. Even running CUPED on only half the users still beat the vanilla t-test on all users.

**Covariate ablation (3-week A/A test, queries-per-user).** An A/A test is control vs. control, so the true effect is known to be 0 — clean ground for measuring variance.

| Covariate | Variance reduction |
|---|---|
| Entry-day (first day user appears), categorical | ~9–10% after 2 weeks |
| Queries-per-user, 1-week pre-period | > 45% |
| Both combined | only +2–3% over pre-period alone |

The combination result is the interesting one. Entry-day adds almost nothing once you have the pre-period metric — the partial correlation between entry-day and the outcome, given the pre-period metric, is low. **Stacking covariates is mostly wasted effort if you already have the same-metric lag.**

**Pre-period length and experiment length.** Two forces pull against each other:

- **Correlation** rises with longer pre-period and longer experiment — more data, better signal-to-noise for cumulative metrics.
- **Coverage** (fraction of experiment users who also appear in the pre-period) *falls* as the experiment runs longer. Frequent visitors show up early; later arrivals are new or churned-cookie users with no pre-period row. Longer pre-period raises coverage.

Net effect: the reduction curve for a fixed pre-period is **not monotonic** — it peaks around 2 weeks of experiment then slowly decays. With a 2-week pre-period, reduction stays near 50% across a wide range of experiment durations. Recommendation: **1–2 weeks of pre-period.** Too short → poor matching; too long → weaker correlation with the experiment window.

**Where it did not work.** **Revenue-per-user: under 5% variance reduction.** Revenue in the pre-period barely correlates with revenue in the experiment period — it is spiky and rare per user. This is the single most important negative result in the paper, because revenue is usually the metric you care most about. CUPED works well when the metric has a stable per-user level (queries, clicks, visits) and badly when it is a rare event.

**The failure that produces a wrong sign.** They tried using an *in-experiment* covariate: Distinct-Queries-per-user (queries with consecutive duplicates removed). It correlates with queries-per-user almost perfectly, so variance collapsed and confidence intervals got beautifully narrow. On an experiment known to *increase* queries-per-user, CUPED reported a **negative** delta with the CI entirely below zero. Reason: the treatment raised DQ-per-user too, so $\mathbb{E}(X^{(t)}) \ne \mathbb{E}(X^{(c)})$, and the covariate "adjusted away" the very effect being measured. Same trap in the load-time example: use clicks as the covariate in a page-speed experiment and you erase the speed benefit.

## Worth Remembering

- **The condition is the whole method.** $\mathbb{E}(X^{(t)}) = \mathbb{E}(X^{(c)})$. Break it and you do not get a slightly worse answer — you get a confidently wrong answer with a tight confidence interval, which is the worst possible failure mode. Post-treatment covariates are poison.
- **Sensitivity scales badly.** Detecting a 0.5% delta instead of a 5% delta needs $10^2 = 100\times$ the users. That is why a 50% variance cut is worth so much — it is equivalent to doubling traffic.
- **Ideal metric profile:** the value differs a lot between light and heavy users, and a user's level is stable over time. Queries-per-user is the paradigm case.
- CUPED is a purely *analysis-time* change. Nothing about assignment, logging or the experiment design changes. That is why it slots into an existing platform cheaply — and why it is now standard at basically every experimentation platform.
- **The stated follow-up:** move covariate information into the *assignment* step rather than the analysis step (covariate-aware randomisation), plus automated selection of multiple covariates from a library. The first became a research line of its own.
- Note the honest framing on cookie churn: users are identified by cookies, cookies get cleared, so "no pre-period data" is not a rare edge case. The missing-data indicator trick is not an afterthought.
- Connects cleanly to the [[Counterfactual Reasoning and Learning Systems]] world: same goal (estimate a causal effect with less noise), different lever (variance reduction after randomisation, rather than reweighting logged data).
- A practical caveat for the modern reader: the linear form is just one choice. Nothing stops you from fitting $\mathbb{E}(Y\mid X)$ with a gradient-boosted model or a network on rich pre-period features — that is the CUPED++/ML-CUPED line — but the same bias condition still binds absolutely.

## Links
Related: [[Counterfactual Reasoning and Learning Systems]] · [[Regression Analysis]] · [[R-Squared]] · [[Uncertainty]] · [[Random variable]] · [[Panel Regression]] · [[experimentation_question_bank]] · [[Principal Data Scientist — Experimentation & Causal Inference Interview Question Bank]] · [[Decision Sciences]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]]

New topics worth writing: Control variates, Stratified sampling, Delta method for ratio metrics, Statistical power and sample size, Two-sample t-test, ANCOVA, A/A testing, Variance reduction in Monte Carlo, Sequential testing and peeking, Post-treatment bias
