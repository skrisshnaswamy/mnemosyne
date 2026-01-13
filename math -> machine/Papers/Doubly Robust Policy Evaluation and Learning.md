---
title: "Doubly Robust Policy Evaluation and Learning"
authors: ["Dudík", "Langford & Li"]
year: 2011
arxiv: "1103.4601"
url: https://arxiv.org/abs/1103.4601
priority: Must-Read
read_on: 2026-08-25
tags: [paper, rl, theory]
---
## The Core Idea

You have a log of past decisions: for each user we saw context $x$, we showed action $a$, and we got reward $r_a$. We never learn what would have happened for the other actions. Now someone hands you a new policy $\pi$ and asks "what would this have earned?" That is **offline policy evaluation**.

Before this paper the field had two tools, and both were broken in opposite ways.

1. **Direct Method (DM):** fit a model $\hat{\varrho}_a(x)$ that predicts reward from context and action, then just ask the model what $\pi$ would have got. Low variance, but if your reward model is wrong you are confidently wrong forever. Bias never shrinks with more data.
2. **Inverse Propensity Score (IPS):** keep only the logged rows where the old policy happened to pick the same action as $\pi$, and re-weight each by $1/\hat p(a\mid x)$ to undo the old policy's preferences. Nearly unbiased if you know the logging probabilities — and you usually do, because you wrote the logging system. But when $\hat p$ is small the weights explode and the estimate is noise.

The trick here is to **use both at once, so that either one being right is enough**. Take the reward model's prediction as a baseline, then use IPS only to correct the *residual* — the part of the reward the model got wrong:

$$\hat V^{\pi}_{\text{DR}}=\frac{1}{|S|}\sum_{(x,a,r_a)\in S}\left[\frac{\big(r_a-\hat{\varrho}_a(x)\big)\,\mathbf{I}(\pi(x)=a)}{\hat p(a\mid x,h)}+\hat{\varrho}_{\pi(x)}(x)\right]$$

Two things happen. If the propensities $\hat p$ are right, the correction term has mean zero error and the estimator is unbiased no matter how bad $\hat\varrho$ is. If instead $\hat\varrho$ is right, the numerator $r_a-\hat\varrho_a(x)$ is zero on average, the noisy weighted term vanishes, and you are left with the (correct) DM estimate. You need **either** model, not both.

> [!NOTE] Doubly robust estimator ^doubly-robust
> An estimator built from two nuisance models — a reward model and a propensity model — that stays unbiased if *at least one* of them is correct. The variance term that IPS pays is multiplied by the reward model's error, so a merely *decent* reward model shrinks the noise even when neither model is exactly right.

Doubly robust estimation was old statistics (Cassel 1976, Robins 1994). What was new: applying it to **contextual bandit policy evaluation and learning**, and — importantly — analysing it *without* assuming either model is correct. Earlier statistics papers were asymptotic and assumed one model was right; this one gives finite-sample bias and variance as smooth functions of how wrong both models are.

## The Methodology

**Setting.** A contextual bandit. World draws $(x,\vec r)\sim D$ with $\vec r\in[0,1]^{\mathcal A}$, $k$ actions. Logging policy picks $a\sim p(a\mid x,h)$ where $h$ is history. Only $r_a$ is revealed. Target: $V^\pi=\mathbf{E}_{(x,\vec r)\sim D}[r_{\pi(x)}]$. This is a one-step [[Markov Decision Process|MDP]] — no state transitions, just credit assignment across unseen actions.

**Two error terms.** Define the additive error of the reward model and the multiplicative error of the propensity model:

$$\Delta(a,x)=\hat{\varrho}_a(x)-\varrho_a(x),\qquad \delta(a,x,h)=1-\frac{p(a\mid x,h)}{\hat p(a\mid x,h)}$$

$\Delta=0$ means perfect reward model; $\delta=0$ means perfect propensities.

**Theorem 1 (bias).** For a stationary logging policy:

$$|\mathbf{E}[\hat V^\pi_{\text{DR}}]-V^\pi| = |\mathbf{E}_x[\Delta\delta]|$$

Compare: $|\mathbf{E}[\hat V^\pi_{\text{DM}}]-V^\pi|=|\mathbf{E}_x[\Delta]|$ and $|\mathbf{E}[\hat V^\pi_{\text{IPS}}]-V^\pi|=|\mathbf{E}_x[\varrho_{\pi(x)}\delta]|$. The DR bias is a *product* of two errors. If both are small-but-nonzero, say each 10%, DR's bias is 1% while DM's is 10%. IPS is literally the special case $\hat\varrho\equiv 0$.

**Theorem 2 (variance).**

$$\mathbf{Var}[\hat V^\pi_{\text{DR}}]=\frac{1}{|S|}\left(\mathbf{E}[\varepsilon^2]+\mathbf{Var}_x[\varrho_{\pi(x)}+\Delta\delta]+\mathbf{E}_x\left[\frac{1-p}{p}\Delta^2(1-\delta)^2\right]\right)$$

with $\varepsilon=(r_a-\varrho_a)\mathbf{I}/\hat p$. Three parts: reward noise, context noise, and the **importance-weighting penalty**. The IPS version is identical except the third term has $\varrho_{\pi(x)}^2$ where DR has $\Delta^2$. That is the whole story of why DR is quieter: IPS pays $\frac{1-p}{p}\varrho^2$, DR pays $\frac{1-p}{p}\Delta^2$. As long as your reward model beats predicting zero, $\Delta^2 < \varrho^2$ and the $1/p$ blow-up is damped. DM's variance is just $\frac{1}{|S|}\mathbf{Var}_x[\varrho_{\pi(x)}+\Delta]$ — tiny, which is exactly why people keep reaching for it despite the bias.

**Practical note in the analysis:** $\hat p$ and $\hat\varrho$ are assumed fitted on data *independent* of $S$ (sample splitting). Fitting the reward model on the same rows you evaluate on breaks the guarantee.

**Turning classification into a bandit.** To get public data with known propensities, take a $k$-class dataset, convert label $c$ into a loss vector $l_a=\mathbf{I}(a\neq c)$, then pick $a\sim\text{unif}(1..k)$ and reveal only $l_a$. So $p(a\mid x)\equiv 1/k$, exactly known. Nine UCI datasets, $k$ from 4 to 26, sizes 214 to 20,000.

**Reward model.** Deliberately weak: per-action linear model $\hat l(x,a)=w_a\cdot x$ fit by least-squares ridge regression ([[Regularization|ridge]] = L2-penalised [[Regression Analysis|linear regression]]). During policy optimisation, $w_a$ is fit only on rows where the logged action equals $a$.

**Policy optimisation loop.** Split 70/30. Bandit-ify the training set. Use IPS or DR to *impute* the whole loss vector for each training example. Feed the completed cost-sensitive dataset to one of two learners: Direct Loss Minimization (a perceptron-style update, $\theta_{a_1}\!\leftarrow\!\theta_{a_1}+\eta x$, $\theta_{a_2}\!\leftarrow\!\theta_{a_2}-\eta x$, $\eta=t^{-0.3}/2$, $\epsilon=0.1$, 20 random restarts) or a Filter Tree with J48 decision trees at the nodes. 30 repeats.

## Ablation Studies and Experiments

**Policy evaluation (500 repeats, bias and RMSE of the estimated classification error):**

| dataset | bias DM | bias IPS | bias DR | rmse DM | rmse IPS | rmse DR |
|---|---|---|---|---|---|---|
| ecoli | 0.129 | 0.004 | 0.002 | 0.129 | 0.137 | **0.101** |
| glass | 0.147 | 0.003 | 0.001 | 0.147 | 0.194 | **0.142** |
| letter | 0.213 | 0.000 | 0.001 | 0.213 | 0.049 | **0.030** |
| optdigits | 0.175 | 0.000 | 0.000 | 0.175 | 0.023 | 0.023 |
| page-blocks | 0.063 | 0.000 | 0.000 | 0.063 | 0.012 | **0.011** |
| pendigits | 0.208 | 0.000 | 0.000 | 0.208 | **0.015** | 0.016 |
| vehicle | 0.281 | 0.000 | 0.001 | 0.281 | 0.062 | **0.058** |
| yeast | 0.193 | 0.006 | 0.007 | 0.193 | 0.099 | **0.076** |

The linear reward model is genuinely bad — DM's bias is 0.06 to 0.28 on an error scale of $[0,1]$. Yet DR still cuts IPS's RMSE by ~40% on letter and ~25% on yeast. **What did not help:** on optdigits DR ties IPS, and on pendigits DR is *marginally worse* (0.016 vs 0.015), and on yeast DR's bias is slightly *higher* than IPS (0.007 vs 0.006). When the reward model is near-useless and propensities are exact, DR has nothing to add and can add a hair of noise.

**Policy optimisation (classification error, 30 runs):**

| dataset | IPS+DLM | DR+DLM | IPS+FT | DR+FT | Offset Tree |
|---|---|---|---|---|---|
| ecoli | 0.529 | **0.289** | 0.466 | 0.326 | 0.340 |
| letter | 0.930 | 0.607 | 0.939 | **0.472** | 0.584 |
| optdigits | 0.644 | **0.090** | 0.840 | 0.178 | 0.325 |
| pendigits | 0.536 | 0.127 | 0.731 | **0.096** | 0.150 |
| satimage | 0.402 | **0.171** | 0.693 | 0.186 | 0.210 |
| page-blocks | 0.089 | 0.083 | 0.370 | 0.053 | **0.045** |
| yeast | 0.730 | **0.529** | 0.811 | 0.591 | 0.590 |

This is the headline. On optdigits, DLM trained on IPS-imputed losses gets **64.4%** error — barely better than guessing among 10 classes — while DR-imputed losses give **9.0%**. Pendigits: 53.6% → 12.7%. The gap is far bigger than in evaluation, because a learner *amplifies* noisy loss estimates: a few blown-up importance weights drag the whole decision boundary.

Against **Offset Tree**, a method purpose-built for partial-label learning: DR+FilterTree beats it on 7 of 9 datasets (letter 0.472 vs 0.584, optdigits 0.178 vs 0.325). Offset Tree only wins on page-blocks (0.045 vs 0.053). Note the authors' own caveat — Filter Tree and Offset Tree share a representation, so *that* comparison is a fair test of the estimator; DLM vs trees is not.

**Generality check:** DR improved both a gradient-descent learner (DLM) and a tree-induction learner (Filter Tree). It is an estimator swap, not an algorithm.

**Real-data experiment: covariate shift as a 2-armed bandit.** 3,854,689 browser cookies from a Yahoo portal, March 2010, ~5000 sparse binary features each, target = average visits per cookie (ground truth **23.8**). Sampling was made deliberately non-uniform: project features onto the first principal component, and set $p_i=\min\{\mathcal N(x_i\cdot\bar x),1\}$ for a normal $\mathcal N$ centred off to one side. So $a_i\in\{0,1\}$ is "was this cookie sampled", and evaluating the constant policy $\pi(x)\equiv 1$ recovers the true mean. Subsample fractions $f\in\{0.0001,\dots,0.05\}$, 100 repeats. DR beat IPS on RMSE at every data size, by **10–20%, averaging 13.6%**, with the biggest gains at small $n$. Decomposing, the gain was all variance, not bias — exactly what Theorem 2 predicts.

**A failed tweak buried in the appendix:** for DLM they tried picking the best of 20 random restarts using a held-out validation set instead of training loss. No benefit.

## Worth Remembering

- **DR is a strict upgrade you can drop in.** Wherever you already compute IPS, you can add a reward model and get the DR form for almost no extra cost. Setting $\hat\varrho\equiv 0$ recovers IPS exactly, so it is impossible to do structurally worse — you only lose if your reward model is *anti*-correlated with truth.
- **"Doubly robust" is about bias, but the practical win is variance.** In the experiments propensities were exactly known ($1/k$), so IPS was already unbiased and the robustness guarantee bought nothing. Every improvement came from the $\Delta^2$ vs $\varrho^2$ swap in the variance. That is the pattern in real systems too: you usually *do* know your logging probabilities, and the reward model is there as a control variate.
- **The reward model does not need to be good.** A per-action ridge regression with 0.2 absolute bias still cut RMSE substantially. Do not wait for a great reward model.
- **Sample splitting matters.** The theory assumes $\hat\varrho,\hat p$ are independent of the evaluation set. Fit them on a separate fold or use cross-fitting; otherwise the residuals $r_a-\hat\varrho_a(x)$ are optimistically small and you re-import DM's bias.
- **Limitation: one step only.** This is contextual bandits, not sequential [[Markov Decision Process|RL]]. Extending DR across a trajectory multiplies importance weights over timesteps and the variance returns with a vengeance — that is the whole later literature on doubly robust off-policy evaluation for RL.
- **Limitation: needs $p>0$ everywhere $\pi$ acts.** If the logging policy never took an action, no amount of robustness recovers it; you fall back entirely on the reward model, unverifiably.
- The DR estimator is unbounded — a single row with $\hat p=10^{-4}$ and a large residual can dominate. The paper does not clip or self-normalise. In production people add weight capping or the self-normalised (Hájek) variant, trading a little bias for a lot of variance.
- Connection worth holding: the same "predict a baseline, correct only the residual" shape is a **control variate**, and it is the same idea as the advantage function in [[Proximal Policy Optimization Algorithms|policy gradient]] methods and as covariate adjustment in [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)|CUPED]]. Subtract what you can predict, only pay variance on what you cannot.
- Open question the authors flag: build an Offset Tree variant that uses a *learned* reward model instead of a crude constant offset.

## Links

Related: [[Unbiased Learning-to-Rank with Biased Feedback]] · [[Counterfactual Reasoning and Learning Systems]] · [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)]] · [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)]] · [[Markov Decision Process]] · [[Decision Sciences]] · [[Regression Analysis]] · [[Regularization]] · [[Uncertainty]] · [[Random variable]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Entire Space Multi-Task Model (ESMM)]] · [[Proximal Policy Optimization Algorithms]]

New topics worth writing: Contextual bandits, Off-policy evaluation, Horvitz-Thompson estimator, Control variates, Covariate shift, Self-normalised importance sampling, Cross-fitting and Neyman orthogonality, Offset Tree, Cost-sensitive multiclass classification, Filter Tree reduction
