---
title: "The Self-Normalized Estimator for Counterfactual Learning (SNIPS) (NeurIPS)"
authors: ["Swaminathan & Joachims"]
year: 2015
url: https://www.cs.cornell.edu/people/tj/publications/swaminathan_joachims_15d.pdf
priority: Good-To-Read
read_on: 2026-09-24
tags: [paper, optimization, vision, theory]
---
## The Core Idea

When you learn a policy from logged bandit data, the usual scoring rule is the inverse-propensity-score (IPS) estimate of risk. This paper shows that rule is broken as a *training objective*, not just noisy — and the fix is one line of arithmetic.

The IPS risk estimate of a new policy $h$ on logged data is

$$\hat R(h) = \frac{1}{n}\sum_{i=1}^{n} \delta_i \frac{h(y_i \mid x_i)}{p_i}$$

where $\delta_i$ is the observed loss, $p_i = h_0(y_i \mid x_i)$ is the probability the *logging* policy gave the action it actually took, and lower loss is better. It is unbiased. It is also degenerate when you minimise it over a rich hypothesis class.

Why. The ratio $h(y_i\mid x_i)/p_i$ is a free multiplier that the optimiser controls. If losses are negative, the optimiser can drive $\hat R$ arbitrarily low by *piling probability onto the logged actions* — never mind whether they were good. If losses are positive, it can drive $\hat R$ to exactly $0$ by putting probability on actions the log never contains — never mind whether they are good either. Either way the learner is gaming the propensity ratio, not the loss. The authors name this **propensity overfitting**, and it has no analogue in supervised learning, where an overfitted training loss is at least bounded by the range of the loss function.

Two concrete symptoms of the same disease:
- $\hat R(h)$ can fall **outside** the range of $\delta$. With $\delta \in [-2,-1]$ and $k$ actions, a memorising hypothesis scores $\hat R = -k$, arbitrarily below the true optimum $-2$.
- $\hat R$ is **not equivariant**: adding a constant $C$ to every loss shifts the true risk by $C$, but does *not* shift the finite-sample estimate by $C$. So the winner of the optimisation depends on where you put zero. That is an absurd property for an objective.

The fix is the **self-normalised** estimator: divide by the sum of the weights instead of by $n$.

$$\hat R_{SN}(h) = \frac{\sum_{i=1}^n \delta_i \frac{h(y_i\mid x_i)}{p_i}}{\sum_{i=1}^n \frac{h(y_i\mid x_i)}{p_i}}$$

> [!NOTE] Self-normalised IPS (SNIPS)
> Replace the $1/n$ denominator of the IPS estimator with the empirical sum of importance weights. The result is a weighted average of the observed losses, so it is bounded inside the range of $\delta$, is equivariant to translating $\delta$, and cannot be gamed by inflating or deflating the weights. It is biased at finite $n$ but strongly consistent. ^snips

The reason it works: the sum of weights $\hat S(h) = \frac1n\sum_i h(y_i\mid x_i)/p_i$ has known expectation $1$ for *every* $h$. That is a free diagnostic — if a learned policy reports $\hat S = 0.000$, it has fled the data; if it reports $\hat S = 5.35$, it has piled onto it. Rather than test for this after the fact, the paper folds the known quantity in as a **multiplicative control variate**, which is exactly the division above. The estimate becomes a convex combination of the $\delta_i$ you actually observed, and a convex combination cannot leave the interval.

What it unlocks: counterfactual learning over high-capacity hypothesis spaces stops being a lottery on sign conventions. Before this, the standard practice of "shift your losses to be non-negative" could silently produce a policy worse than random.

## The Methodology

**Setting.** Batch learning from logged bandit feedback (BLBF). You have a log $D = \{(x_i, y_i, \delta_i, p_i)\}$ from a stationary logging policy $h_0$ with full support over the action space. You want a new policy minimising $R(h) = \mathbb{E}_{x}\mathbb{E}_{y\sim h(x)}[\delta(x,y)]$. See [[Off-Policy Evaluation]] and [[Counterfactual Risk Minimization]] for the framing.

**The objective.** Norm-POEM keeps the CRM recipe — minimise the risk estimate *plus* a data-dependent variance penalty — and swaps the estimator:

$$\hat h = \arg\min_{h\in\mathcal{H}} \left\{ \hat R_{SN}(h) + \lambda\sqrt{\widehat{\mathrm{Var}}(\hat R_{SN}(h))} \right\}$$

The variance penalty comes from the generalisation bound in the original CRM paper, which is an empirical-Bernstein argument: the true risk is bounded by the estimate plus a term scaling with $\sqrt{\widehat{\mathrm{Var}}/n}$. Penalising variance is penalising distance from $h_0$ — a leash, the same idea as the KL term in [[PPO]] or [[RLHF]], but derived from a bound rather than chosen.

**Variance of a ratio estimator.** Using the delta method / normal approximation:

$$\widehat{\mathrm{Var}}(\hat R_{SN}(h)) = \frac{\sum_i (\delta_i - \hat R_{SN}(h))^2 \left(\frac{h(y_i\mid x_i)}{p_i}\right)^2}{\left(\sum_i \frac{h(y_i\mid x_i)}{p_i}\right)^2}$$

Note this is a *global* quantity — every term depends on $\hat R_{SN}(h)$, which depends on every sample. That kills stochastic gradient descent.

**Hypothesis space.** Stochastic linear rules in the exponential family over a joint feature map $\phi(x,y)$:

$$h_w(y\mid x) = \frac{\exp(w\cdot\phi(x,y))}{Z(x)}$$

i.e. a CRF. Gradients are tractable whenever the partition function $Z(x)$ is.

**Optimisation.** Non-convex; optimised with L-BFGS over the full batch, not SGD. This is forced by the ratio structure, and the authors were braced for it to be slow — it was not (see below).

**Hyperparameters.** Two, the same as POEM: $M$, a cap on the importance weights (needed so the true variance of $\hat R_{SN}$ exists at all), and $\lambda$, the variance-regularisation strength. Both tuned by counterfactual evaluation on a 25% held-out slice of $D$.

**Data.** The supervised-to-bandit trick: take four multi-label LibSVM datasets (Scene, Yeast, TMC, LYRL), train a CRF on a random 5% of the supervised labels to serve as $h_0$, then replay the training inputs 4 times, sampling a label from $h_0$ each time and revealing only the Hamming loss between the sampled label and the true label. The full-information CRF trained on all labels is the skyline. Loss is measured as expected Hamming loss on the held-out supervised test set; lower is better.

## Ablation Studies and Experiments

**Headline (Table 1, test Hamming loss, mean of 10 runs, lower better):**

| | Scene | Yeast | TMC | LYRL |
|---|---|---|---|---|
| $h_0$ (logging policy) | 1.511 | 5.577 | 3.442 | 1.459 |
| POEM (IPS) | 1.200 | 4.520 | 2.152 | 0.914 |
| **Norm-POEM (SNIPS)** | **1.045** | **3.876** | **2.072** | **0.799** |
| CRF skyline (full info) | 0.657 | 2.830 | 1.187 | 0.222 |

Significant on all four (one-tailed paired $t$-test, $\alpha = 0.05$).

**The ablation that actually proves the thesis (Table 2).** They train each method twice — once with losses shifted to be all positive, once all negative — and report both the learned $\hat S(\hat h)$ (should be $1$) and the test loss.

| | $\hat S$ Scene | $\hat S$ Yeast | $\hat S$ TMC | $\hat S$ LYRL |
|---|---|---|---|---|
| POEM, $\delta > 0$ | 0.274 | 0.028 | **0.000** | 0.175 |
| POEM, $\delta < 0$ | 1.782 | 5.352 | 2.802 | 1.230 |
| Norm-POEM, $\delta > 0$ | 0.981 | 0.840 | 0.941 | 0.945 |
| Norm-POEM, $\delta < 0$ | 0.981 | 0.821 | 0.938 | 0.945 |

Exactly the predicted pattern. Positive losses → POEM's learned policy runs *away* from the logged data ($\hat S \to 0$). Negative losses → it runs *onto* it ($\hat S \gg 1$). Norm-POEM sits near 1 either way, and gives identical results under translation, as equivariance demands.

The corresponding test losses are the damning part. POEM with positive losses: Scene 2.059, Yeast 5.441, LYRL 2.399, and **TMC 17.305** — worse than random guessing, and worse than the logging policy it was supposed to improve on. Norm-POEM: 1.058 / 3.881 / 2.079 / 0.799, essentially unchanged from the negative-loss run.

**Is the variance regulariser still needed?** Reasonable hypothesis: maybe SNIPS alone is enough and you can drop $\lambda$. No. Norm-IPS (unregularised) vs Norm-POEM: Scene 1.072 → 1.045, Yeast 3.905 → 3.876, LYRL 0.806 → 0.799, and TMC **3.609 → 2.072**. Significant on Scene, TMC and LYRL. The two mechanisms fix different failures — self-normalisation kills propensity overfitting, variance regularisation still handles ordinary high-variance weights.

**Sample-size curve (Yeast).** Sweeping ReplayCount from $2^0$ to $2^8$, Norm-POEM is below POEM at every sample size. Both converge towards the CRF skyline, but slowly, because $h_0$ is thin-tailed — rare actions have tiny $p_i$ and contribute almost nothing.

**The surprise: it is faster, not slower (Table 4, seconds).**

| | Scene | Yeast | TMC | LYRL |
|---|---|---|---|---|
| POEM | 78.69 | 98.65 | 716.51 | 617.30 |
| Norm-POEM | 7.28 | 10.15 | 227.88 | 142.50 |
| CRF | 4.94 | 3.43 | 89.24 | 72.34 |

Per-iteration cost is *higher* for Norm-POEM, but it needs far fewer iterations. The diagnosis given: POEM chases large $\|w\|$, trying to push probability towards 1 on every training point with negative loss — an objective with no bottom, so the optimiser keeps walking. Norm-POEM's objective only cares about each instance's loss *relative to the others in the sample*, so it settles at a much smaller $\|w\|$. Runtime ends up within a small factor of full-information CRF training.

**What did not work / limits.** Naively shifting losses to be non-negative — the standard hygiene move — is precisely what triggers the worst POEM failure. And the ratio form of the variance estimate blocks stochastic optimisation entirely, so this recipe as written is full-batch only.

## Worth Remembering

- The single cheapest diagnostic in the whole paper: **compute $\hat S(h) = \frac1n\sum_i h(y_i\mid x_i)/p_i$ on any policy you learned off-policy.** It should be near 1. If it is 0.0 or 5.4, your learner is gaming propensities and the reported risk is fiction. This costs nothing and applies to any IPS-based pipeline, including the ones in [[Unbiased Learning-to-Rank with Biased Feedback]] and [[Recommendations as Treatments- Debiasing Learning and Evaluation]].
- SNIPS is **biased at finite $n$** and unbiasedness was the whole selling point of IPS. The trade is deliberate: a biased estimator with bounded range beats an unbiased one that an optimiser can drive to $-\infty$. Bias you can reason about; degeneracy you cannot. This is an instance of the general [[Reward Hacking]] pattern — the optimiser is an adversary against your measurement.
- The bias is $O(1/n)$-ish and vanishes asymptotically (Theorem 2, strong consistency via the law of large numbers on numerator and denominator separately).
- Self-normalisation does **not** remove the need for weight clipping. You still need $M$ to guarantee the estimator's true variance exists, which is required for the normal approximation behind the variance formula.
- Everything here assumes **full support**: $h_0(y\mid x) > 0$ for all actions, and logged propensities $p_i$ recorded at decision time. No propensities, no method. Same precondition as [[Doubly Robust Policy Evaluation and Learning]] and the replay evaluator in [[Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms]].
- Control variates are the general tool here. The additive version gives you regression and doubly-robust estimators; this paper uses the **multiplicative** version. Worth holding both in mind as two moves on the same [[Monte Carlo Methods]] variance problem.
- Practical caveat: the experiments are small multi-label classification with a 5%-trained logging policy. Not a production recommender with millions of actions and a near-deterministic logger. The thin-tail slow-convergence note in Figure 1 is honest about that.
- Follow-up question worth chasing: SNIPS is the default self-normalised baseline in later off-policy work (BanditNet, Open Bandit Pipeline). Does self-normalisation interact well or badly with doubly-robust estimators, which already reduce variance through a different mechanism?

## Links
Related: [[Off-Policy Evaluation]] · [[Counterfactual Risk Minimization]] · [[Doubly Robust Policy Evaluation and Learning]] · [[Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms]] · [[Counterfactual Reasoning and Learning Systems]] · [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[Unbiased Learning-to-Rank with Biased Feedback]] · [[Open Bandit Dataset and Pipeline- Towards Realistic and Reproducible Off-Policy Evaluation]] · [[Contextual Bandit]] · [[On-Policy vs Off-Policy]] · [[Offline RL]] · [[Monte Carlo Methods]] · [[Regularization]] · [[Reward Hacking]] · [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]]

New topics worth writing: Importance sampling and its variance, Control variates (additive vs multiplicative), Equivariance of estimators, Empirical Bernstein bounds, Weight clipping / truncated importance sampling, Effective sample size for importance weights, BanditNet
</document>
