---
title: "Recursive Partitioning for Heterogeneous Causal Effects"
authors: ["Susan Athey", "Guido Imbens"]
year: 2015
arxiv: "1504.01132"
url: https://arxiv.org/abs/1504.01132
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, theory]
---
## The Core Idea

A randomised experiment tells you the *average* effect of a treatment. It does not tell you **who** the treatment helps. The obvious move is to hunt for subgroups — "the drug works better for young women" — but if you go hunting in the same data you then report p-values from, you have p-hacked. Classical practice fixes this by forcing you to write down the subgroups *before* seeing the data (a pre-analysis plan). That is safe but blind.

This paper makes the hunt legitimate. Two ideas do the work.

**Idea 1 — honesty.** Split your data in two. Use sample A to *decide* the subgroups (grow the tree). Use sample B, which the tree-growing never touched, to *estimate* the treatment effect inside each subgroup. Because sample B is independent of the partition, the leaf estimates are unbiased and ordinary confidence intervals are valid — no sparsity assumption, no bound on the number of covariates, no multiple-testing correction. You can have 20 covariates and 500 people and still get honest 90% intervals.

> [!NOTE] Honest estimation
> An estimator is *honest* if it does not use the same data to choose the model structure and to fit the parameters given that structure. Contrast with *adaptive* (standard CART), which reuses one sample for both and therefore inherits the selection bias of the search. ^honest-estimation

**Idea 2 — the tree must be told to look for effect heterogeneity, not outcome variation.** Standard regression trees split to predict $Y$ well. Here we want to predict $\tau(x)$ well. That is hard because of what Holland called the fundamental problem of causal inference: for any one person you observe either $Y_i(1)$ or $Y_i(0)$, never both, so the "ground truth" label $\tau_i = Y_i(1)-Y_i(0)$ is **never observed**. There is no residual to minimise and nothing obvious to cross-validate against. The trick is that although $\tau_i$ is unobservable, *unbiased estimators of the mean squared error in $\tau$* are constructible, and those are enough to drive both splitting and cross-validation.

The third, subtler contribution: **anticipating honesty changes the splitting rule itself**. If you know leaf means will be re-estimated later on fresh data, you know bias will be gone. So the split criterion should stop worrying about bias and instead trade off *sharper leaves* against *noisier leaf estimates from having fewer people in them*. That variance penalty is written into the objective explicitly.

What it unlocks: you can take any old RCT off the shelf, data-mine it for subgroups, and publish confidence intervals you are allowed to believe.

## The Methodology

**Setup.** $N$ units, potential outcomes $(Y_i(0), Y_i(1))$, binary treatment $W_i$, observed $Y_i^{\rm obs} = Y_i(W_i)$, covariates $X_i \in \mathbb{R}^K$. Assume unconfoundedness, $W_i \perp\!\!\!\perp (Y_i(0),Y_i(1)) \mid X_i$ — in a randomised trial this holds for free. Let $p = \Pr(W_i=1)$.

> [!NOTE] CATE
> The conditional average treatment effect $\tau(x) \equiv \mathbb{E}[Y_i(1)-Y_i(0) \mid X_i = x]$. Most of causal inference chases the single number $\mathbb{E}[Y_i(1)-Y_i(0)]$; this paper chases the whole function, approximated as a step function over a partition. ^cate

A tree is just a partition $\Pi = \{\ell_1,\dots\}$ of covariate space. Inside a leaf, the estimate is the plain difference of means:
$$\hat\tau(x;\mathcal{S},\Pi) = \hat\mu(1,x;\mathcal S,\Pi) - \hat\mu(0,x;\mathcal S,\Pi)$$
where each $\hat\mu$ is the average observed outcome among treated (resp. control) units of that leaf in sample $\mathcal S$.

**Three samples.** $\mathcal S^{\rm tr}$ grows and prunes the tree. $\mathcal S^{\rm est}$ estimates leaf effects. $\mathcal S^{\rm te}$ evaluates. In experiments $N^{\rm tr} = N^{\rm est}$.

**The honest objective, prediction case first.** Expand the expected MSE and use $\mathbb{E}_{\mathcal S}[\hat\mu] = \mu$:
$$-{\rm EMSE}(\Pi) = \mathbb{E}_{X}[\mu^2(X;\Pi)] - \mathbb{E}_{\mathcal S^{\rm est},X}[\mathbb{V}(\hat\mu(X;\mathcal S^{\rm est},\Pi))]$$
Read it plainly: **reward spread between leaf means, punish noise within leaves.** Both terms can be estimated from the training sample alone, given only the *size* $N^{\rm est}$. The variance term is estimated from within-leaf sample variances $S^2_{\mathcal S^{\rm tr}}(\ell)$ divided by leaf counts; the first term uses $\hat\mu^2$ debiased by its own variance. With $N^{\rm tr} = N^{\rm est}$ the estimator collapses to
$$\widehat{\rm EMSE}(\mathcal S^{\rm tr},\Pi) = \frac{1}{N^{\rm tr}}\sum_{i\in\mathcal S^{\rm tr}} \hat\mu^2(X_i;\mathcal S^{\rm tr},\Pi) \;-\; \frac{2}{N^{\rm tr}}\sum_{\ell\in\Pi} S^2_{\mathcal S^{\rm tr}}(\ell)$$
Standard CART uses only the first term. In the *prediction* setting the two terms turn out to be proportional, so this correction barely changes which splits get chosen — a useful negative result.

**The causal version.** Same expansion, $\tau$ in place of $\mu$:
$$\widehat{\rm EMSE}_\tau(\mathcal S^{\rm tr},\Pi) = \frac{1}{N^{\rm tr}}\sum_{i\in\mathcal S^{\rm tr}} \hat\tau^2(X_i;\mathcal S^{\rm tr},\Pi) - \frac{2}{N^{\rm tr}}\sum_{\ell\in\Pi}\left(\frac{S^2_{\mathcal S^{\rm tr}_{\rm treat}}(\ell)}{p} + \frac{S^2_{\mathcal S^{\rm tr}_{\rm control}}(\ell)}{1-p}\right)$$
Now the two terms are **not** proportional, so the variance penalty genuinely matters. Concretely: a split can lower the variance of $\hat\tau$ even when both children have identical treatment effects, simply because a covariate that drives the *level* of $Y$ makes each side more homogeneous. The honest criterion notices this; the adaptive one does not.

**Honest cross-validation.** Even the honest split criterion overstates fit as trees deepen, because early splits already grouped similar outcomes together, so within-leaf sample variances in $\mathcal S^{\rm tr}$ are artificially small. Fix: evaluate the *same* $\widehat{\rm EMSE}_\tau$ formula on the held-out fold, $\widehat{\rm EMSE}_\tau(\mathcal S^{\rm tr,cv},\Pi)$ — note the leaf effects are re-estimated on the CV fold, not carried over from the fitting fold, because that mirrors what will happen at estimation time.

**The four estimators compared** (each in adaptive "-A" and honest "-H" flavour, eight total):

1. **CT (Causal Tree)** — the above. Splits on $\hat\tau^2$ minus the variance penalty.
2. **TOT (Transformed Outcome Tree)** — build a pseudo-label whose conditional mean *is* the treatment effect, $\mathbb{E}[Y^*_i\mid X_i=x] = \tau(x)$, then run stock CART. (The paper's printed formula $Y^*=(Y_i-W_i)/(p(1-p))$ is a typo; the intended object is $Y_i(W_i-p)/(p(1-p))$, i.e. $Y_i/p$ if treated, $-Y_i/(1-p)$ if control.) Dead simple, but wasteful: the leaf mean of $Y^*$ equals $\hat\tau$ only if the treated fraction in that leaf is exactly $p$.
3. **F (Fit-based)** — split on ordinary goodness-of-fit of $Y$, but fit a two-parameter model (intercept + treatment dummy) in each leaf. From Zeileis et al.'s model-based recursive partitioning. Failure mode: two splits that improve fit equally look equally good, even if only one of them creates heterogeneity in $\tau$.
4. **TS (Squared t-statistic)** — split to maximise $T^2 = N(\bar Y_L - \bar Y_R)^2 / (S^2/N_L + S^2/N_R)$ on the treatment effect difference (Su et al.). Failure mode: puts *zero* value on splits that improve fit.

The paper shows CT sits between F and TS. For a single binary split,
$$\widehat{\rm EMSE}_\tau(\mathcal S,\Pi_N)-\widehat{\rm EMSE}_\tau(\mathcal S,\Pi_S) = \frac{(T^2-4)(\tilde S^2 - F/N) + 2\tilde S^2}{p(1-p)}$$
i.e. dominated by $T^2$ (like TS) but with the fit improvement $F$ folded in (like F). That is the whole design argument in one line.

**Implementation details that mattered.** Built on R's `rpart`. Minimum leaf size $n_m = 25$ *treated* **and** 25 *control*, so a treatment effect is always computable. **5** cross-validation folds instead of the usual 10 — with 25-per-arm leaves, tenth-sized folds can end up with zero treated units in a leaf. And a "bucketing" trick: within each leaf, sort treated and control separately, group into buckets of $b=4$, and only consider bucket boundaries as split points. Without this, moving the split point by one observation moves *one* unit from *one* arm, which jerks $\hat\tau$ around wildly whenever a covariate strongly drives the outcome level — causing spurious splits on exactly the covariates you should ignore.

## Ablation Studies and Experiments

Simulation only, three designs, $p=0.5$, $Y_i(w) = \eta(X_i) + \tfrac12(2w-1)\kappa(X_i) + \epsilon_i$, $\epsilon \sim \mathcal N(0, 0.01)$, $X \sim \mathcal N(0,I)$, test set $N^{\rm te} = 6000$.

- **Design 1:** $K=2$, $\eta = \tfrac12 x_1 + x_2$, $\kappa = \tfrac12 x_1$. No noise covariates.
- **Design 2:** $K=10$, $\kappa = \sum_{k=1}^{2}\mathbb 1\{x_k>0\}x_k$.
- **Design 3:** $K=20$, $\kappa = \sum_{k=1}^{4}\mathbb 1\{x_k>0\}x_k$.

The indicator matters: those covariates move $\eta$ over their whole range but move $\kappa$ only when positive. So criteria that chase outcome fit will split in the wrong places.

**Confidence interval coverage at nominal 90% — the headline.**

| | D1 (500/1000) | D2 | D3 |
|---|---|---|---|
| CT-A (adaptive) | 0.85 / 0.85 | 0.81 / 0.83 | 0.80 / 0.81 |
| TOT-A | 0.83 / 0.86 | 0.83 / 0.83 | **0.74** / 0.79 |
| CT-H (honest) | 0.90 / 0.90 | 0.90 / 0.89 | 0.90 / 0.90 |
| all -H methods | 0.89–0.91 | 0.90–0.92 | 0.89–0.90 |

Every honest variant hits nominal. Every adaptive variant undercovers, worst in the hardest design (0.74). This is the point of the paper and it is clean.

**Which splitting criterion actually finds heterogeneity** (MSE$_\tau$ relative to CT-H, lower is better, honest versions):

| | D1 500/1000 | D2 | D3 |
|---|---|---|---|
| TOT-H | 1.77 / 2.12 | 1.03 / 1.04 | 1.03 / 1.05 |
| F-H | 1.93 / 1.54 | 1.69 / 2.07 | 1.63 / 2.08 |
| TS-H | 1.01 / 1.02 | 1.06 / 0.99 | 1.24 / 1.38 |
| CT-H | 1.00 | 1.00 | 1.00 |

Readings:
- **F is the clear loser everywhere**, 1.5–2.1× worse. It splits on covariates that move $\eta$ but not $\kappa$. It also builds by far the deepest trees (13.2 leaves at $N=1000$ vs ~7.5 for CT) — deep trees on the wrong variables. F would only look good in a design where $\eta = \kappa$.
- **TOT collapses in Design 1** (1.77–2.12×) because conditional outcome variance is tiny (0.01) and TOT ignores the realised $W_i$, so its noise dominates. TOT also prunes very aggressively — 2.1–3.4 leaves — because that extra noise makes splits look bad.
- **TS matches CT in Design 1** (where $x_1$ enters $\eta$ and $\kappa$ identically, so the two criteria agree) and **degrades in Design 3** (1.24 → 1.38 as $N$ grows). The gap *widens* with sample size: bigger trees mean more chances for the criteria to disagree.
- Honest versions build **deeper** trees than adaptive ones (CT-H 11.2 leaves vs CT-A 2.5 in D2 at $N=1000$). The honest criterion has removed the overfitting-through-bias channel, so the only brake left is leaf variance, and it can afford to go deeper.

**The cost of honesty.** Comparing adaptive-with-1000-observations against honest-with-500+500, the MSE$_\tau$ ratio (adaptive ÷ honest) for CT is 0.91 / 0.93 / 0.76 across designs. So **adaptive with double the data does fit better** — sample splitting is not free. But at *equal training size*, honesty wins outright: Appendix Table A1, Design 1, $N=500$: CT-A = 0.104, CT-H = 0.087. Design 2: 1.418 → 1.149. Design 3: 3.324 → 3.033. Removing selection bias buys back a good chunk of what splitting costs. F-H/F-A = 0.50 in Design 1 is the outlier where adaptive looks twice as good; F is barely affected by honesty because it splits on outcome fit anyway.

**A feasible evaluation criterion.** Since MSE$_\tau$ is unobservable in real life, they also report $MSE^{\rm TOT} = \frac{1}{N^{\rm te}}\sum (Y^*_i - \hat\tau_i)^2$, which is unbiased for it (up to a constant) but noisy. It ranks the estimators identically except in one cell (Design 3, $N=500$), though the gaps look much smaller — e.g. F-H is 1.12–1.16× CT-H instead of 1.63–2.08×. Practical warning: this feasible criterion is directionally right but badly compresses differences.

## Worth Remembering

- **What honesty is really buying is not accuracy, it is inference.** The simulations say so plainly: fit gets slightly worse, coverage goes from ~0.80 to exactly 0.90. If you only want point predictions of $\tau(x)$, sample splitting may not pay. If you want to *report* a subgroup effect with an interval, it is the whole game.
- **The variance-penalty term only matters for causal trees.** In plain prediction it is proportional to the fit term and changes nothing. This is a nice, precise statement of why you cannot just bolt CART onto causal inference.
- **Partitions, not personalisation.** The authors are explicit that a tree is chosen partly for interpretability and deployability — "settings where decision rules must be remembered, applied or interpreted by human beings or computers with limited processing power," e.g. clinical guidelines, or a lookup table that cuts latency in an online personalisation system. The follow-up, Wager & Athey's causal forests (ref [28], cited here as "http://arxiv.org/pdf/1510.04342"), drops interpretability for accuracy and gets asymptotic normality for fully personalised estimates.
- **Observational data.** Everything extends under unconfoundedness by propensity-score weighting inside the leaves, with weights renormalised within each leaf *and* within each arm. Trimming extreme propensity scores (Crump et al.) is recommended. The additional regularity conditions for asymptotic normality under weighting carry over unchanged, since the estimation stage is on an independent sample. See [[Recommendations as Treatments- Debiasing Learning and Evaluation]] and [[Unbiased Learning-to-Rank with Biased Feedback]] for the same weighting machinery in recsys.
- **SUTVA is assumed and will break on networks.** No interference between units. Marketplace or social-graph experiments violate this.
- **Caveats for a practitioner.** Everything is simulated; no real dataset in the paper. The variance term needs a within-leaf sample variance, which is why leaves are forced to ≥25 per arm and folds are cut to 5 — small-sample plumbing that will bite you if you port the idea naively. The bucketing hack ($b=4$) "improved goodness of fit on average for the simulations we considered, although it can in principle make things worse" — an honest admission that it is a heuristic.
- **Connection worth chasing:** honest sample splitting here is the same instinct as cross-fitting in [[Double-Debiased Machine Learning for Treatment and Structural Parameters#Why sample splitting, separately|double machine learning]] — use one fold to select/fit nuisance structure, another to estimate the parameter you care about, so the selection noise does not contaminate the inference. Different mechanism, same disease being treated.
- **Open question:** the trade-off between $N^{\rm tr}$ and $N^{\rm est}$ is fixed at 50/50 throughout and never tuned. Nothing says that is optimal.

## Links

Related: [[Double-Debiased Machine Learning for Treatment and Structural Parameters]] · [[Doubly Robust Policy Evaluation and Learning]] · [[Counterfactual Reasoning and Learning Systems]] · [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[Unbiased Learning-to-Rank with Biased Feedback]] · [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)]] · [[Counterfactual Risk Minimization]] · [[XGBoost- A Scalable Tree Boosting System]] · [[Why do tree-based models still outperform deep learning on tabular data]] · [[Regularization]] · [[Uncertainty]] · [[Principal Data Scientist — Experimentation & Causal Inference Interview Question Bank]] · [[experimentation_question_bank]]

New topics worth writing: Causal forests (Wager & Athey), CART / recursive partitioning, Rubin causal model and potential outcomes, Propensity score, Unconfoundedness, SUTVA, Conditional average treatment effect, Sample splitting and cross-fitting, Pre-analysis plans and multiple hypothesis testing, Model-based recursive partitioning (Zeileis), Targeted learning (van der Laan)
