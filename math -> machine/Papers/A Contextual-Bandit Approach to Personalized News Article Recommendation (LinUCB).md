---
title: "A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)"
authors: ["Li et al."]
year: 2010
arxiv: "1003.0146"
url: https://arxiv.org/abs/1003.0146
priority: Must-Read
read_on: 2026-08-25
tags: [paper, theory]
---
## The Core Idea

Picking which news article to show a user is not a supervised learning problem. You only ever see the click (or no-click) for the article you actually showed. The other three articles you could have shown stay silent forever. So the system must **choose in order to learn**, and every choice made purely to learn costs you clicks today.

That is the exploration/exploitation trade-off, and before this paper the well-understood tools for it (UCB1, $\epsilon$-greedy) were **context-free**: they treated an article as a single unknown click-through rate, the same for everybody. Useless for personalisation, because a 40-year-old woman in Ohio and a teenage boy have very different opinions about the same iPod article.

The contribution is showing that if you assume the expected payoff of an arm is **linear in a feature vector describing the user–article pair**, then the confidence interval around your prediction has a *closed form*. No sampling, no simulation, no Bayesian machinery. It is just

$$\sqrt{\mathbf{x}_{t,a}^{\top}\mathbf{A}_a^{-1}\mathbf{x}_{t,a}}$$

where $\mathbf{A}_a = \mathbf{D}_a^\top \mathbf{D}_a + \mathbf{I}_d$ is the ridge-regression Gram matrix for that arm. This is the standard-error formula from [[Regression Analysis|linear regression]], reused as an exploration bonus. That is the whole trick, and it turns UCB from a per-arm counter into a per-arm *model*.

> [!NOTE] Contextual bandit
> At each step $t$ you see a context $\mathbf{x}_{t,a}$ for each available arm $a$, pick one arm, and observe the payoff for **only that arm**. Goal: minimise regret $R_{\mathsf{A}}(T) = \mathbf{E}[\sum_t r_{t,a_t^*}] - \mathbf{E}[\sum_t r_{t,a_t}]$ against always picking the best arm. Unlike a [[Markov Decision Process]], your action does not change the next state — one step, then reset. ^contextual-bandit

The second contribution is arguably the more useful one for an engineer: a **provably unbiased way to evaluate any bandit policy offline** from logged random-traffic data, with a five-line algorithm. Before this you either ran the policy live (expensive, slow, risky) or built a simulator (biased in ways nobody can bound).

## The Methodology

### LinUCB, disjoint version

Assume each arm $a$ has its own hidden weight vector $\boldsymbol{\theta}_a^*$ and

$$\mathbf{E}[r_{t,a} \mid \mathbf{x}_{t,a}] = \mathbf{x}_{t,a}^\top \boldsymbol{\theta}_a^*.$$

"Disjoint" means no weights are shared between arms — article A's clicks teach you nothing about article B.

Per arm, keep two things: $\mathbf{A}_a \in \mathbb{R}^{d\times d}$ (init to $\mathbf{I}_d$) and $\mathbf{b}_a \in \mathbb{R}^d$ (init to zero). Each round:

1. $\hat{\boldsymbol{\theta}}_a \leftarrow \mathbf{A}_a^{-1}\mathbf{b}_a$ — this is the ridge solution, and the identity-matrix initialisation *is* the ridge penalty. [[Regularization]] falls out for free from "I have not seen data yet, so assume nothing."
2. Score every arm: $p_{t,a} \leftarrow \hat{\boldsymbol{\theta}}_a^\top \mathbf{x}_{t,a} + \alpha\sqrt{\mathbf{x}_{t,a}^\top \mathbf{A}_a^{-1}\mathbf{x}_{t,a}}$.
3. Show $\arg\max_a p_{t,a}$, observe $r_t$.
4. $\mathbf{A}_{a_t} \mathrel{+}= \mathbf{x}\mathbf{x}^\top$, $\mathbf{b}_{a_t} \mathrel{+}= r_t\mathbf{x}$.

Only one hyperparameter, $\alpha$. Theory says $\alpha = 1 + \sqrt{\ln(2/\delta)/2}$, but the authors say this is conservatively large and tune it empirically.

Three ways to read the bonus term, all the same quantity:
- **Frequentist**: the width of the confidence interval on $\mathbf{x}^\top\hat{\boldsymbol{\theta}}_a$.
- **Bayesian**: the ridge posterior is Gaussian with covariance $\mathbf{A}_a^{-1}$, so $\mathbf{x}^\top\mathbf{A}_a^{-1}\mathbf{x}$ is the predictive variance and the bonus is one predictive standard deviation. Connects directly to [[Uncertainty]] and [[Beliefs]].
- **Information-theoretic**: the entropy of the posterior drops by $\tfrac12\ln(1 + \mathbf{x}^\top\mathbf{A}_a^{-1}\mathbf{x})$ if you pick this arm. So LinUCB is trading off predicted reward against **how much you learn**.

Cost: $O(d^2)$ to update, $O(d^3)$ if you invert every step — but you can cache $\mathbf{A}_a^{-1}$ and refresh it periodically. Linear in number of arms. Handles arms appearing and disappearing (news articles do this hourly) because a new arm just starts at $\mathbf{I}_d, \mathbf{0}$. Regret bound $\tilde{O}(\sqrt{KdT})$, matching the best known.

### LinUCB, hybrid version

Real systems want features shared across all arms ("this user likes politics"). So:

$$\mathbf{E}[r_{t,a}\mid \mathbf{x}_{t,a}] = \mathbf{z}_{t,a}^\top\boldsymbol{\beta}^* + \mathbf{x}_{t,a}^\top\boldsymbol{\theta}_a^*$$

with $\boldsymbol{\beta}^*$ global. Now the arms' confidence intervals are *coupled*, so the clean disjoint formula breaks. Algorithm 2 recovers it with block matrix inversion. You maintain $\mathbf{A}_0, \mathbf{b}_0$ (global, $k\times k$) plus $\mathbf{A}_a, \mathbf{B}_a, \mathbf{b}_a$ per arm, and the confidence term becomes

$$s_{t,a} = \mathbf{z}^\top\mathbf{A}_0^{-1}\mathbf{z} - 2\mathbf{z}^\top\mathbf{A}_0^{-1}\mathbf{B}_a^\top\mathbf{A}_a^{-1}\mathbf{x} + \mathbf{x}^\top\mathbf{A}_a^{-1}\mathbf{x} + \mathbf{x}^\top\mathbf{A}_a^{-1}\mathbf{B}_a\mathbf{A}_0^{-1}\mathbf{B}_a^\top\mathbf{A}_a^{-1}\mathbf{x}.$$

Still $O(d^2 + k^2)$ per trial with cached inverses. The point of the shared term is **transfer**: clicks on one article inform your estimate for others.

### The offline evaluator (Algorithm 3)

This is the bit worth stealing. You have logged data where the production system chose arms **uniformly at random**. To evaluate any policy $\pi$:

> Walk the log. For each event, ask $\pi$ what it would pick. If $\pi$ picks the same arm the logger picked, keep the event — add it to $\pi$'s history, add the reward. Otherwise throw it away entirely and move on, changing nothing.

Because the logger was uniform, each event survives with probability exactly $1/K$, independent of everything. So the surviving stream has the same distribution as the real world. Theorem 1: every history $h_T$ has *identical probability* under the evaluator and under running $\pi$ live. Not approximately — identically. So $R_T/T$ is an unbiased estimate of $\pi$'s value. Cost: you need $KT$ logged events in expectation to simulate $T$ real steps.

> [!NOTE] Replay / rejection-sampling evaluation
> Unbiased off-policy evaluation by discarding every logged event where the candidate policy disagrees with the logging policy. Requires the logger to be randomised and the events i.i.d. Generalises to non-uniform loggers via [[Doubly Robust Policy Evaluation and Learning|importance-weighted]] rejection sampling, at the cost of throwing away more data. ^replay-evaluation

### Data and features

Yahoo! Front Page Today Module, "Featured" tab, F1 story position only (to avoid position bias). May 2009 random bucket: 4.7M events on May 01 for tuning, ~36M events May 03–09 for evaluation (33M+ headline number).

Raw features: >1000 binary user features (gender, 10 age buckets, ~200 geo, ~1000 behavioural categories, kept only if support $\geq 0.1$), ~100 article features (URL categories, editor tags). Normalised to unit length, plus a constant-1 feature: 1193 user dims, 83 article dims.

Then aggressive dimensionality reduction, because storing big vectors online is expensive:
1. Fit a **bilinear** logistic regression $\phi_u^\top \mathbf{W}\phi_a$ on Sept 2008 random traffic.
2. Project users: $\psi_u = \phi_u^\top\mathbf{W}$ — component $i$ is "how much this user likes article category $i$". A learned [[Linear Projection]].
3. K-means into **5 user clusters**. Final user feature = 5 soft memberships (Gaussian kernel, normalised to sum to 1) + constant = **6 dims**.

So $\mathbf{x}_{t,a} \in \mathbb{R}^6$ (user only, disjoint), and $\mathbf{z}_{t,a} \in \mathbb{R}^{36}$ = outer product of 6-dim user and 6-dim article features (the interaction term, shared).

Two metrics, both from Algorithm 3. **Learning bucket** CTR = the small slice where exploration happens; high CTR here means fast learning / low regret. **Deployment bucket** CTR = the big slice served greedily from current estimates; this is what the business cares about. All CTRs reported relative to the random policy.

## Ablation Studies and Experiments

Baselines: `random` (=1.0 by definition), `ε-greedy`, `ucb` (UCB1 with $c_{t,a}=\alpha/\sqrt{n_{t,a}}$), `omniscient` (best fixed article chosen in hindsight, CTR **1.615** — the ceiling for anything non-personalised), warm-start variants, and per-segment variants running one copy of the bandit inside each of the 5 user clusters.

Headline table, 100% data, deploy / learn:

| algorithm | deploy | learn | lift vs ε-greedy (deploy) |
|---|---|---|---|
| ε-greedy | 1.596 | 1.326 | — |
| ucb | 1.594 | 1.569 | 0% (but +18.3% learn) |
| ε-greedy (seg) | 1.742 | 1.446 | 9.1% |
| ucb (seg) | 1.781 | 1.677 | 11.6% |
| ε-greedy (disjoint) | 1.769 | 1.309 | 10.8% |
| **linucb (disjoint)** | **1.795** | 1.647 | **12.5%** |
| ε-greedy (hybrid) | 1.739 | 1.521 | 9.0% |
| linucb (hybrid) | 1.730 | 1.663 | 8.4% |

Everything with features beats `omniscient` (1.615). That is the cleanest statement in the paper: personalisation buys you more than perfect hindsight knowledge of which single article is globally best.

**Guided beats unguided exploration.** UCB-style methods win in the learning bucket by a wide margin at every data level, and win in deployment too. `ε-greedy` explores by picking uniformly at random; UCB explores by picking the arm it is most *uncertain* about. Same exploration budget, better spent.

**Data sparsity is where hybrid earns its keep.** They subsampled the data the algorithms learn from (30%, 20%, 10%, 5%, 1%) while still evaluating on everything. At 1%:

- `linucb (hybrid)` deploy = **1.482**, learn = **1.446** — a 20.1% / 27.0% lift, its *best* relative showing.
- `linucb (disjoint)` deploy = 1.382, learn = 1.197.
- `ucb` = 1.354 / 1.220.

So at full data the shared-parameter model is not worth it (8.4% vs 12.5%), but as data thins out the transfer across arms becomes the dominant effect. Sensible: with 1% of the traffic, per-arm ridge regressions have almost no data each, and the global $\boldsymbol{\beta}$ is the only thing being estimated well.

### What did not work

- **Warm start was unstable and got dropped.** They pre-trained a bilinear LR on Sept 2008 traffic and used it as an offset on the context-free CTR estimate. Both warm-start variants beat `omniscient` on tuning data, but performance was erratic. Specifically `ucb (warm)` helped in the learning bucket but not in deployment — "since `ucb` relies on a confidence interval for exploration, it is hard to correct the initialisation bias introduced by warm start." The confidence interval shrinks based on counts, not on whether the prior was wrong, so a bad prior stays baked in. They did not run warm start on the evaluation data at all.
- **`linucb (disjoint)` was statistically indistinguishable from `ucb (seg)`** — the trivial baseline of "run a separate context-free UCB inside each of 5 hard user clusters." The authors diagnose this honestly: their 6-dim features are *soft* cluster memberships, and 85% of users had a max membership above 0.5, 40% above 0.8. So the "soft" features were effectively one-hot already. Linear personalisation with degenerate features degenerates to segmentation. They suggest PCA-style features with more diverse components would separate the two.
- **Inverted-U in the tuning curves.** Both $\epsilon$ and $\alpha$ have a sweet spot in the learning bucket: too small = under-explore, never find the good articles; too large = over-explore, waste impressions. LinUCB is not tuning-free.
- $\epsilon$-greedy (seg) actually went *negative* in the learning bucket at 20% data ($-12\%$) and 10% ($-3.1\%$) — unguided exploration inside small segments is worse than not segmenting at all.

## Worth Remembering

- **The exploration bonus is a standard error.** If you already know [[Regression Analysis]], you already know LinUCB. $\mathbf{A}_a^{-1}$ is the inverse Gram matrix; directions of feature space you have seen little of have large $\mathbf{x}^\top\mathbf{A}_a^{-1}\mathbf{x}$ and get explored. This is much better targeted than "flip a coin with probability $\epsilon$."
- **Assumption you are buying:** expected reward is *linear* in features and the noise is conditionally independent. Real CTR is not linear. In practice people bolt this on top of a neural net's last layer (neural-linear) precisely because the closed form only survives in the last layer.
- **The offline evaluator needs uniform-random logged traffic.** This is the practical blocker. You must be running a random exploration bucket already, and you burn $K$ logged events per simulated step. With $K = 20$ articles and a need for 1M evaluation steps, that is 20M logged random impressions. Also: it only evaluates policies, it does not train them off-policy. Compare with [[Counterfactual Reasoning and Learning Systems]] and [[Unbiased Learning-to-Rank with Biased Feedback]], which use [[Doubly Robust Policy Evaluation and Learning|propensity weighting]] instead of rejection, keeping all data at the cost of variance.
- **Cold start is the real motivation.** Collaborative filtering cannot rank an article published 20 minutes ago. Content features plus a bandit can. This is a different answer to the same problem [[Deep Neural Networks for YouTube Recommendations (RecSys)|YouTube's example-age feature]] attacks.
- **Interleaved learn/deploy buckets** is a nice production pattern worth copying: a small slice pays the exploration cost, the large slice greedily consumes the estimates, and you measure both. Related to how [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)|online experiment infrastructure]] is normally split.
- Authors' own open questions: arms as complex objects (a *ranking* of pages, not one item), non-stationarity (user interests drift, and nothing here decays old data — $\mathbf{A}_a$ only ever grows), and comparison to Banditron.
- Note the feature engineering is doing a lot of work and is entirely pre-deep-learning: bilinear LR → projection → k-means → 6 dims. A [[The Bitter Lesson (essay)|bitter-lesson]] reading would predict that the hand-built 5-cluster segmentation is exactly the part that got replaced later.

## Links

Related: [[Doubly Robust Policy Evaluation and Learning]] · [[Counterfactual Reasoning and Learning Systems]] · [[Unbiased Learning-to-Rank with Biased Feedback]] · [[Regression Analysis]] · [[Regularization]] · [[Uncertainty]] · [[Decision Sciences]] · [[Markov Decision Process]] · [[Recommender Systems - Evolution]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] · [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)]] · [[Linear Projection]] · [[Beliefs]]

New topics worth writing: UCB1 and the Lai–Robbins lower bound, Thompson Sampling, Ridge regression and the Gram matrix, Regret bounds, Off-policy evaluation with importance sampling, Neural-linear bandits, Cold start in recommenders, Bilinear models for user-item interaction, EXP3 and adversarial bandits, Epoch-greedy
