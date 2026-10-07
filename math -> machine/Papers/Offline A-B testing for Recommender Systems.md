---
title: "Offline A/B testing for Recommender Systems"
authors: ["Gilotte et al."]
year: 2018
arxiv: "1801.07030"
url: https://arxiv.org/abs/1801.07030
priority: Good-To-Read
read_on: 2026-09-25
tags: [paper, rl, theory]
---
## The Core Idea

You want to know whether a new recommender policy $\pi_t$ beats the one in production $\pi_p$, but you only have logs from $\pi_p$. The standard answer is [[Off-Policy Evaluation|importance sampling]]: reweight each logged reward by $w = \pi_t(a|x)/\pi_p(a|x)$. In ranking this blows up, because the action is a whole top-$K$ list — there are $K!\binom{M}{K}$ of them — so the weights have enormous variance. The usual fix is to clip the weights at some cap $c$. Clipping kills variance but introduces bias, and here is the uncomfortable part: **the bias of capped importance sampling is only small when the new policy performs *badly* on exactly the actions it favours most.** That is the opposite of the case you care about.

The measured version of that problem at Criteo: to get variance small enough to detect a 1% uplift you need $c \lesssim 10^2$; to get the worst-case bias bound small enough you need $c \gtrsim 10^{23}$. There is no cap that satisfies both. The bias-variance dial is broken.

The move in this paper: **stop trying to bound the bias in the worst case, and instead model it** — and model it *locally*, per context, not globally.

The already-known global version is self-normalisation (NCIS): divide by the sum of capped weights instead of by $n$. That implicitly assumes the reward on the clipped-away region equals the reward on the region you kept, averaged over everything. The new contribution is to make that same assumption **within each context group** (PieceNCIS) or **within each single context** (PointNCIS). Because the reward depends much more on who the user is than on which banner you showed them, normalising per-context removes the dominant part of the bias.

> [!NOTE] Normalised Capped Importance Sampling (NCIS) ^ncis
> Capped importance sampling divided by the average capped weight, not by $n$. It patches the clipping bias by assuming the clipped region earns the same average reward as the un-clipped region. Consistent only if reward and clipping are uncorrelated. ^ncis-def

## The Methodology

**Setup.** One log $\mathcal{S}_n = \{(x_i, a_i, r_i)\}$ from the production policy. $x$ is a display (user context + eligible items), $a$ is a top-$K$ ranking, $r \in [0, r_{\max}]$ is a click-based reward. Both policies are stochastic distributions over rankings (Plackett–Luce-style), so $\pi_p(a|x)$ was logged and $\pi_t(a|x)$ can be computed. The target is the uplift

$$\Delta\mathcal{R}(\pi_p, \pi_t) = \mathbb{E}_{\pi_t}[R] - \mathbb{E}_{\pi_p}[R].$$

The second term is a plain average over the log. Only the first needs an estimator.

**The baseline ladder.**

$$\hat{\mathcal{R}}^{\rm IS} = \frac1n\sum w(a,x)\,r, \qquad w = \frac{\pi_t(a|x)}{\pi_p(a|x)}$$

$$\hat{\mathcal{R}}^{\rm maxCIS} = \frac1n\sum \min(w, c)\,r, \qquad \hat{\mathcal{R}}^{\rm zeroCIS} = \frac1n\sum \mathbf{1}_{w<c}\,w\,r$$

Write $\bar w$ for either capped weight. The exact bias of capping is

$$\mathbb{E}_{\pi_t}[R] = \underbrace{\mathbb{E}_{\pi_p}[\bar W R]}_{\text{what CIS estimates}} + \underbrace{\mathbb{E}_{\pi_t}\!\left[R\tfrac{W-\bar W}{W}\,\middle|\,W>c\right]\mathbb{P}_{\pi_t}(W>c)}_{\text{bias}}$$

and the only available bound is $0 \le \mathcal{B}^{\rm CIS} \le r_{\max}(1 - \mathbb{P}(W \le c))$ — tight only if reward is low where weights are high.

**NCIS and its bias.** Self-normalising gives, asymptotically,

$$\mathcal{R}^{\rm NCIS} = \frac{\mathbb{E}_{\pi_t}[\bar W R / W]}{\mathbb{E}_{\pi_t}[\bar W / W]}, \qquad \mathcal{B}^{\rm NCIS} = -\frac{\operatorname{Cov}_{\pi_t}\!\left(R, \tfrac{\bar W}{W}\right)}{\mathbb{E}_{\pi_t}\!\left[\tfrac{\bar W}{W}\right]}.$$

So NCIS is unbiased exactly when reward and *how much you got clipped* are uncorrelated. In zero-capping form the assumption reads plainly: $\mathbb{E}_{\pi_t}[R \mid W>c] \approx \mathbb{E}_{\pi_t}[R \mid W<c]$.

**Why that fails.** Condition on the context and the covariance splits in two:

$$\mathcal{B}^{\rm NCIS} = -\frac{\operatorname{Cov}_{\pi_t}\!\left(\mathbb{E}[R|X],\ \mathbb{E}\!\left[\tfrac{\bar W}{W}\middle|X\right]\right) + \mathbb{E}_{\pi_t}\!\left[\operatorname{Cov}\!\left(R, \tfrac{\bar W}{W}\middle|X\right)\right]}{\mathbb{E}_{\pi_t}\!\left[\tfrac{\bar W}{W}\right]}$$

The first term is *between* contexts, the second *within*. The authors argue the first dominates, because the recommendation is an unsolicited nudge while the context already contains the user's intent — reward correlates with intent far more than with the banner.

Their toy counter-example makes this concrete. 10% registered customers (reward 10 under $\pi_p$, 12 under $\pi_t$, $\mathbb{E}[\bar W/W] = 0.7$), 90% unknown (reward 1 under both, $\mathbb{E}[\bar W/W] = 1$). True $\mathbb{E}_{\pi_t}[R] = 2.1$, $\mathbb{E}_{\pi_p}[R] = 1.9$ — a real win. NCIS says 1.8. You reject a genuine improvement.

**PieceNCIS (piecewise constant bias model).** Partition contexts into groups $\mathcal{G}$, run NCIS inside each, recombine:

$$\hat{\mathcal{R}}^{\rm PieceNCIS} = \sum_{g \in \mathcal{G}} \alpha_g \, \hat{\mathcal{R}}|_g^{\rm NCIS}, \qquad \alpha_g = \frac{1}{n}\sum \mathbf{1}_{x \in g}$$

The partition must not depend on $\pi_t$. Hand-crafted feature splits work but are arbitrary, so instead they fit a value model $V(x)$ predicting expected reward from context on held-out data, then bucket log-uniformly: $\mathcal{G} = \{V^{-1}([b^k, b^{k+1}])\}$. Two benefits — within a bucket reward barely depends on $x$, and bucket sizes stay large enough to estimate a ratio. Cost: you must train and maintain a separate value model.

**PointNCIS (pointwise bias model).** Push it to one group per context:

$$\mathbb{E}_{\pi_t}[R|X=x] \approx \frac{\mathbb{E}_{\pi_t}[R\bar W/W \mid X=x]}{\mathbb{E}_{\pi_t}[\bar W/W \mid X=x]}$$

You cannot form that ratio empirically — almost no two log rows share the same $x$. But the **denominator only involves the two policies, not the reward**, and you can sample rankings from $\pi_t$ freely. So estimate $1/\mathbb{E}_{\pi_t}[\bar W/W \mid X=x]$ by Monte Carlo. A plain reciprocal-of-average is biased, so they use the Midzuno–Sen rejection sampling scheme to get an *unbiased* estimate $\hat{IP}_c(x)$:

1. draw $u \sim \mathrm{Unif}(0,1)$ and $w_1 \sim \pi_t$ repeatedly until $w_1 < u$;
2. draw $w_2, \dots, w_n \sim \pi_t$;
3. return $n / \left(\tfrac{\bar w_1}{w_1} + \dots + \tfrac{\bar w_n}{w_n}\right)$.

Then

$$\hat{\mathcal{R}}^{\rm PointNCIS} = \frac1n \sum_{(x,a,r) \in \mathcal{S}_n} \hat{IP}_c(x)\,\bar w(a,x)\,r.$$

> [!NOTE] Pointwise bias correction ^pointncis
> Per-context self-normalisation, where the normaliser is estimated by sampling from the target policy rather than from the log. Works because the denominator $\mathbb{E}_{\pi_t}[\bar W/W \mid X=x]$ needs no reward data. ^pointncis-def

**The failure mode they patch.** If the two policies barely overlap at some $x$, the denominator gets tiny and the *effective* weight $\bar w / \mathbb{E}_{\pi_t}[\bar W/W|x]$ can far exceed the cap $c$ — variance returns. Lemma A.3 shows that with **max** capping you can always shrink the internal cap $\tilde c$ so the effective weight stays under $c$ (as $\tilde c \to 0$ the effective weight $\to 1$). Lemma A.4 gives a counter-example proving this is **impossible with zero capping**. So PointNCIS requires max capping.

## Ablation Studies and Experiments

**Data.** 39 real online A/B tests on Criteo's commercial recommender, a few hundred billion recommendations total. Click-based reward, so $\mathbb{E}[R|X,A] \approx 10^{-3}$ and $R \in \{0,1\}$. Ground truth = the online uplift. Offline estimate is computed on the *control* arm only (logging = $\pi_p$), one comparison per test so the comparisons stay independent. Cap fixed at $c = 100$.

**What they refused to run, and why.** This is the most useful negative part of the paper.

- **Plain IS and NIS**: variance so high the confidence interval on $\Delta\hat{R}$ never excludes zero. Every decision would be "neutral". Discarded.
- **[[Doubly Robust Policy Evaluation and Learning|Doubly robust]]**: two separate reasons. First, learning $\bar r(a,x)$ over $K!\binom{M}{K}$ rankings is hopeless. Second and worse — when the reward is Bernoulli with $p \approx 10^{-3}$, a reward model predicting $0.001$ simply cannot correlate with a $0/1$ outcome, so the control variate removes almost no variance and DR collapses back onto IS. **DR is a bad fit for extremely rare, high-variance rewards.** And capped DR degenerates: as $c \to 0$ the optimal policy under the estimator is just the argmax of the reward model.
- **Zero vs max capping**: empirically near-identical (the offending weights are enormous relative to $c$), so only max capping is reported.

**The main table.** Correlation with online uplift, precision of the positive/neutral/negative decision, false-negative rate, and mean confidence-interval width relative to CIS:

| | Correlation | Precision | FNR | CI size |
|---|---|---|---|---|
| CIS | $-0.15 \pm 0.35$ | $0.28 \pm 0.10$ | $0.64 \pm 0.11$ | — |
| NCIS | $0.24 \pm 0.18$ | $0.47 \pm 0.11$ | $0.33 \pm 0.11$ | 1.1 |
| PieceNCIS | $0.36 \pm 0.22$ | $0.53 \pm 0.11$ | $0.28 \pm 0.08$ | 1.5 |
| PointNCIS | $0.49 \pm 0.13$ | $0.56 \pm 0.13$ | $0.16 \pm 0.09$ | 0.8 |

**What the ablations actually reveal.**

- CIS has *negative* correlation. Not noise — it systematically under-estimates, because clipping always throws reward away. With more genuine winners than losers in the dataset, systematic under-estimation reads as anti-correlation. FNR 0.64: it rejects two thirds of real improvements.
- **Global bias modelling buys most of the false-negative reduction.** $0.64 \to 0.33$ just from self-normalising. Cheapest win in the paper.
- **Local bias modelling buys the rest.** Precision $0.47 \to 0.53 \to 0.56$, FNR $0.33 \to 0.28 \to 0.16$. Inspecting the tests where decisions flipped, several were structurally like the registered-vs-unknown-customer counter-example — the mechanism is the one they predicted, not a coincidence.
- **PointNCIS is also the tightest.** CI size 0.8 vs PieceNCIS 1.5. So the finer model is not paying for bias with variance, which is the opposite of the naive expectation.
- **The honest caveat on the last step.** PointNCIS beats PieceNCIS clearly on correlation (0.49 vs 0.36) but precision is nearly the same (0.56 vs 0.53). The authors say the correlation gain may just be a better *magnitude* estimate rather than better *decisions*.
- **Compute.** NCIS and PieceNCIS need every row of the log, including the overwhelming majority with $r=0$, because the normaliser sums $\bar w$ over everything. CIS and PointNCIS only need rows with non-zero reward. With clicks at $\sim 10^{-3}$ that is a three-orders-of-magnitude difference in data read.

**Framing that drove the whole evaluation.** They deliberately treat false negatives as much worse than false positives. A false positive costs you one online A/B test. A false negative is a real improvement that never gets tested, and the cost compounds forever. Metrics were chosen around that asymmetry, which is why FNR is reported at all.

## Worth Remembering

- The headline diagnostic — *variance wants $c < 10^2$, the bias bound wants $c > 10^{23}$* — is worth reusing. Before choosing a cap, plot both curves against $c$ and see whether the window is empty. If it is, you need a bias *model*, not a better cap.
- Even the best estimator here gets precision 0.56 and correlation 0.49. Offline A/B testing on a production recommender is a **screening filter**, not a replacement for the online test. It stops you shipping disasters and helps you prioritise; it does not tell you the uplift.
- The key structural insight generalises past recommendation: **reward correlates with context far more than with action**. That is why per-context normalisation beats global normalisation. It is also why [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)|CUPED]] works, and the same intuition sits behind local/conditional control variates everywhere.
- PointNCIS needs to *sample from* $\pi_t$ at evaluation time, many times per logged context. That is a real engineering requirement — the candidate policy has to be callable, not just scoreable. And you need the logged propensity $\pi_p(a|x)$ stored exactly, the usual prerequisite.
- Max capping is not interchangeable with zero capping here. Lemma A.4 kills zero capping for PointNCIS. Small detail, easy to get wrong in an implementation.
- Unaddressed limitations: the isolation assumption (units independent) is assumed throughout; there is no treatment of interference between users. The value model for PieceNCIS needs a separate held-out dataset and can drift. And the per-context approximation still carries the *within*-context covariance term, which they assume is small and never measure.
- Their own proposed follow-up: several policies run each week, so mix them into a logging distribution that sits closer to whatever you want to test next. That is a deliberate exploration-design idea rather than an estimator idea — closer in spirit to [[Contextual Bandit|logging-policy design]] than to reweighting.
- Note where this sits historically. [[Counterfactual Reasoning and Learning Systems|Bottou & Peters 2013]] introduced clipping and *bounded* the bias; [[The Self-Normalized Estimator for Counterfactual Learning (SNIPS) (NeurIPS)|SNIPS]] introduced self-normalisation for learning; this paper *models* the bias conditionally. Later work like [[Optimal and Adaptive Off-policy Evaluation in Contextual Bandits (SWITCH)|SWITCH]] attacks the same variance wall by switching between IS and a reward model per-region, and [[Off-policy evaluation for slate recommendation|the pseudo-inverse estimator]] attacks it by exploiting slate structure instead.

## Links

Related: [[Off-Policy Evaluation]] · [[The Self-Normalized Estimator for Counterfactual Learning (SNIPS) (NeurIPS)]] · [[Counterfactual Reasoning and Learning Systems]] · [[Doubly Robust Policy Evaluation and Learning]] · [[Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms]] · [[Optimal and Adaptive Off-policy Evaluation in Contextual Bandits (SWITCH)]] · [[Off-policy evaluation for slate recommendation]] · [[Counterfactual Risk Minimization]] · [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[AB Testing]] · [[Controlled experiments on the web- survey and practical guide (DMKD)]] · [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)]] · [[Contextual Bandit]] · [[On-Policy vs Off-Policy]] · [[Monte Carlo Methods]] · [[NDCG]] · [[Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning (MAGIC)]]

New topics worth writing: Midzuno–Sen and Lahiri unbiased ratio sampling, Plackett–Luce and Thurstonian ranking policies, empirical Bernstein bounds, false-negative-weighted decision metrics for offline screening, logging-policy design and exploration budgets, effective sample size for importance weights
