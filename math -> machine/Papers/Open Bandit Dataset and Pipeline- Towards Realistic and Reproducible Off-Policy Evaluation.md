---
title: "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation"
authors: ["Yuta Saito", "Shunsuke Aihara", "Megumi Matsutani", "Yusuke Narita"]
year: 2020
arxiv: "2008.07146"
url: https://arxiv.org/abs/2008.07146
priority: Must-Read
read_on: 2026-09-17
tags: [paper, rl, theory]
---
## The Core Idea

Off-policy evaluation (OPE) asks: *how well would a policy I have not deployed have done?* You only have logs from the policy that was actually running. The field had a decade of estimator theory — [[Doubly Robust Policy Evaluation and Learning|DR]], [[Counterfactual Risk Minimization|SNIPW]], Switch, shrinkage — but almost no honest way to test any of it.

The reason is subtle and worth holding on to. To score an OPE estimator you need the **true** value of the evaluation policy $V(\pi_e)$. In a simulator you know it by construction, but the simulator may not resemble anything real. On a real logged dataset you do not know it at all, because the evaluation policy was never run. So researchers benchmarked on synthetic environments or on multiclass classification data pretending to be bandit data, and every paper used its own setup.

The unlock here is almost embarrassingly simple: **log two policies at once**. ZOZOTOWN (a large Japanese fashion e-commerce site) ran an A/B test where each user impression was randomly assigned to either uniform Random or Bernoulli Thompson Sampling. That gives two logged datasets from the same platform, same week, same users. Now you can take the Random log as $\mathcal{D}^{(b)}$, estimate the value of Bernoulli TS from it, and compare against the **on-policy average click rate actually observed in the Bernoulli TS log**. That observed average is the ground truth. Swap the roles and you get a second test.

> [!NOTE] Off-policy evaluation ground truth
> With logs from two policies on the same platform, the empirical reward mean of policy B's log *is* $V(\pi_B)$, so it can score any estimator that tries to guess $V(\pi_B)$ from policy A's log. One log alone can never do this. ^ope-ground-truth

They also released the **exact policy code** — including the Beta priors used in production for Bernoulli TS — which nobody had done before. Without that you cannot replay the evaluation policy to get $\pi_e(a\mid x)$, so you cannot compute importance weights, so no OPE. The code release is not a nicety; it is a load-bearing part of the contribution.

The second deliverable is **Open Bandit Pipeline** (`obp`), a scikit-learn-style Python package with a `dataset`, `policy` and `ope` module, so that everyone's benchmark runs the same way.

## The Methodology

**The setting.** Standard contextual bandit. Context $x$ (user features), discrete action $a$ (a fashion item), reward $r\in[0,r_{\max}]$ (click). A policy $\pi(a\mid x)$ is a distribution over actions. Logs come from the behaviour policy:

$$\{(x_i,a_i,r_i)\}_{i=1}^n \sim \prod_{i=1}^n p(x_i)\,\pi_b(a_i\mid x_i)\,p(r_i\mid x_i,a_i)$$

The target is the policy value

$$V(\pi_e) := \mathbb{E}_{(x,a,r)\sim p(x)\pi_e(a\mid x)p(r\mid x,a)}[r]$$

**The three base estimators.** Write $w(x,a) := \pi_e(a\mid x)/\pi_b(a\mid x)$ for the importance weight, $\hat q(x,a)$ for a learned reward model, and $\hat q(x,\pi) := \mathbb{E}_{a\sim\pi}[\hat q(x,a)]$.

- Direct Method: $\hat V_{\mathrm{DM}} = \mathbb{E}_n[\hat q(x_i,\pi_e)]$. Low variance, but inconsistent if $\hat q$ is wrong — and you cannot measure how wrong it is from the data.
- IPW: $\hat V_{\mathrm{IPW}} = \mathbb{E}_n[w(x_i,a_i)\,r_i]$. Unbiased when $\pi_b$ is known, but blows up in variance when the two policies disagree.
- DR: $\hat V_{\mathrm{DR}} = \mathbb{E}_n[\hat q(x_i,\pi_e) + w(x_i,a_i)(r_i - \hat q(x_i,a_i))]$. Uses $\hat q$ as a control variate; consistent if *either* the weights or the reward model are right.

**The variance-control family** they also benchmark:

- SNIPW: $\hat V_{\mathrm{SNIPW}} = \dfrac{\mathbb{E}_n[w\,r]}{\mathbb{E}_n[w]}$ — divide by the sum of weights, which bounds the estimate inside the reward support. SNDR is the same trick inside DR.
- Switch-DR: use the DR correction term only where $w \le \tau$, fall back to DM elsewhere. $\tau=0$ is DM, $\tau\to\infty$ is DR.
- **DRos** (DR with optimistic shrinkage, Su et al. 2020): replace $w$ with a shrunk weight
$$\hat w(x,a;\lambda) := \frac{\lambda}{w^2(x,a)+\lambda}\,w(x,a)$$
$\lambda=0$ gives DM, $\lambda\to\infty$ gives DR. The shrinkage is chosen to minimise a bound on MSE.

**Automatic hyperparameter tuning** (also from Su et al.) picks $\theta \in \{\tau,\lambda\}$ by minimising an estimated MSE:
$$\hat\theta \in \arg\min_\theta\; \overline{\mathrm{Bias}}(\theta;\mathcal{D})^2 + \hat{\mathbb{V}}(\theta;\mathcal{D})$$
with the bias upper bound $\overline{\mathrm{Bias}} = |\mathbb{E}_n[(\hat w - w)(r_i - \hat q)]| + \sqrt{\tfrac{2\mathbb{E}[w^2]\log(2/\delta)}{n}} + \tfrac{2w_{\max}\log(2/\delta)}{3n}$, with $\delta = 0.05$.

**The data.** 7 days, late November 2019, ~26M impressions over three campaigns:

| Campaign | Policy | #Data | #Items | CTR |
|---|---|---|---|---|
| ALL | Random | 1,374,327 | 80 | 0.35% |
| ALL | Bernoulli TS | 12,168,084 | 80 | 0.50% |
| Men's | Random | 452,949 | 34 | 0.51% |
| Men's | Bernoulli TS | 4,077,727 | 34 | 0.67% |
| Women's | Random | 864,585 | 46 | 0.48% |
| Women's | Bernoulli TS | 7,765,497 | 46 | 0.64% |

Each row has: timestamp, item id, **position** (1/2/3 — three slots in the UI), click indicator, **the true action choice probability** `action_prob`, hashed user features (age, gender), user–item affinity from past click counts, and item features (price, brand, category). CC BY 4.0, non-commercial research.

The three-slot interface is handled by a click-model assumption: **reward depends only on the item and its position**, not on the other two items shown. That is what lets the standard single-action contextual bandit maths apply at all.

**The evaluation protocol.**

1. Compute on-policy ground truth $V_{\mathrm{on}}(\pi_e) = \frac{1}{n^{(e)}}\sum_i r^{(e)}_i$ from the evaluation policy's own log.
2. Bootstrap a sample of size $n$ from the behaviour log, run the estimator, record squared error $(\hat V(\pi_e;\mathcal{D}^{(b,*)}) - V_{\mathrm{on}}(\pi_e))^2$.
3. Repeat $T=200$ times; report RMSE.

$\hat q$ is gradient boosting (`HistGradientBoostingClassifier`, lr 0.01, 100 iters, depth 5) with **2-fold cross-fitting** — train $\hat q$ on one half, evaluate on the other — to stop overfitting from leaking bias into the estimate. Same idea as in [[Double-Debiased Machine Learning for Treatment and Structural Parameters#cross-fitting|cross-fitting]].

## Ablation Studies and Experiments

**Main table, Bernoulli TS → Random, $n=300{,}000$** (RMSE $\times 10^3$, lower better; relative to the best in parentheses):

| Estimator | ALL | Men's | Women's |
|---|---|---|---|
| IPW | 0.493 (1.56) | 0.789 (1.72) | 0.776 (1.38) |
| SNIPW | 0.507 (1.60) | 0.644 (1.40) | 0.804 (1.43) |
| DM | 1.026 (3.24) | 0.773 (1.69) | 0.816 (1.46) |
| DR | 0.482 (1.53) | 0.613 (1.34) | 0.803 (1.43) |
| SNDR | 0.482 (1.53) | 0.659 (1.44) | 0.791 (1.41) |
| Switch-DR | 0.482 (1.53) | 0.613 (1.34) | 0.803 (1.43) |
| **DRos** | **0.316 (1.00)** | **0.459 (1.00)** | **0.561 (1.00)** |

DRos wins everywhere, by 30–60%. Switch-DR is numerically identical to DR — its tuned $\tau$ landed high enough that essentially nothing got switched off.

**Reverse the direction and the picture changes.** Random → Bernoulli TS, ALL campaign: SNIPW is best (0.468), DM is worst by a mile (1.499), DRos is 0.508 — mid-pack. On the *Women's* campaign in this direction, **DM is the single best estimator** (0.584) and every weighted estimator is 2.6–2.8× worse. There is no universal winner. The weights are much nastier when the behaviour policy is uniform and the evaluation policy is a concentrated TS.

**The hyperparameter ablation is the most damning result.** Sweeping $\lambda$ for DRos on the ALL campaign:

| $\lambda$ | Random → BTS | BTS → Random |
|---|---|---|
| 1 | 1.384 | 0.963 |
| 100 | 0.778 | 0.498 |
| 1,000 | 0.482 | **0.245** |
| 5,000 | **0.476** | 0.270 |
| 10,000 | 0.476 | 0.323 |
| **tuning** | **0.476** | 0.323 |

The automatic tuner always reaches for a large $\lambda$ — it weights bias heavily, since large $\lambda$ means low bias and high variance. That is right for Random → BTS (it lands on the optimum). It is **wrong** for BTS → Random, where the optimum is $\lambda=1{,}000$ and the tuner picks a setting 32% worse. The tuner has a systematic preference, not an adaptive one.

**Sample size flips the ranking.** At $n=10{,}000$ vs $n=300{,}000$, BTS → Random:

| | ALL small | ALL large | Men's small | Men's large | Women's small | Women's large |
|---|---|---|---|---|---|---|
| IPW | 1.899 | 0.493 | 3.683 | 0.789 | 3.156 | 0.776 |
| DM | **0.797** | 1.026 | **3.041** | 0.773 | **2.665** | 0.816 |
| DRos | 0.765 | **0.316** | 3.727 | **0.459** | 3.051 | **0.561** |

In the small-sample regime DM — the biased, model-based estimator — beats the weighted ones, because at 10k samples variance dominates bias. At 300k it loses. **The best estimator is a function of your sample size**, which means an estimator choice validated on a big log does not transfer to a small one.

**What actually does the work in DR-family estimators: the reward model.** Same estimators, same data, swapping logistic regression for gradient boosting as $\hat q$ (ALL, BTS → Random, $n=300$k, RMSE $\times10^3$):

| Estimator | LR | GB | Improvement |
|---|---|---|---|
| DM | 1.597 | 1.026 | 64.2% |
| DR | 2.250 | 0.482 | 78.5% |
| SNDR | 2.255 | 0.482 | 78.7% |
| Switch-DR | 2.250 | 0.482 | 78.5% |
| DRos | 2.065 | 0.316 | 84.7% |

This is the ablation people skip and shouldn't. Picking DRos over DR buys you ~34%. Picking a better $\hat q$ buys you ~80%. The measured quality gap is real: relative cross-entropy of $\hat q$ is 0.295 (GB) vs 0.109 (LR) for Random → BTS on ALL. **Your regression model matters more than your estimator.**

**What did not work / what is missing:**
- Switch-DR never separated from DR under tuning — the hyperparameter search did not find a regime where switching helped.
- The automatic tuner failed in one of two directions, as above.
- Only two policies exist in the dataset, so you cannot benchmark **policy selection** or **ranking** of many candidate policies — which is the thing practitioners actually do. DOPE (Fu et al. 2021) does this, but only in simulation.

## Worth Remembering

- The authors flag the **overfitting-to-one-dataset** risk explicitly: OBD is currently *the only* public dataset that supports OPE benchmarking, so the field can collectively tune itself to ZOZOTOWN.
- The position assumption ("reward depends only on item and position") is the load-bearing simplification and they admit it is probably false — an attractive item in slot 1 changes the click probability of slot 2. Slate-action estimators (independent IPS, reward-interaction IPS) are implemented in `obp` but not benchmarked here. That comparison is the obvious follow-up.
- Compute is real: $T=200$ bootstrap iterations at $n=300{,}000$ on the ALL campaign took **750 minutes** on a MacBook Pro. Budget for it.
- Bernoulli TS gets 1.43× the CTR of Random on the ALL campaign (0.50% vs 0.35%). So the dataset also documents a genuine bandit win in production, not just an evaluation artefact. Note this is confounded by which campaign — Random on Men's (0.51%) beats Random on ALL (0.35%).
- The action distribution plots show the real mechanism of OPE difficulty: Random is flat over 80 items, Bernoulli TS is concentrated on a few. Importance weights $\pi_e/\pi_b$ are therefore extreme in both directions, and which direction you go changes which failure mode bites you.
- Practical takeaway if you deploy this: **log the propensities**. `action_prob` in the schema is what makes the whole dataset usable. A production system that logs only the chosen action has thrown away the ability to do any of this.
- The headline research gap the paper opens is not a new estimator. It is **estimator selection and hyperparameter tuning without ground truth** — because in a real application you never have $V_{\mathrm{on}}(\pi_e)$, which is the whole premise. Every result here depends on ground truth the practitioner will not have.
- Connects naturally to [[Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms|replay evaluation]] (the other way to use logged data, requiring a uniform-random log) and to [[Recommendations as Treatments- Debiasing Learning and Evaluation|propensity-weighted recommendation]].

## Links
Related: [[Doubly Robust Policy Evaluation and Learning]] · [[Counterfactual Risk Minimization]] · [[Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms]] · [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]] · [[A Tutorial on Thompson Sampling]] · [[An Empirical Evaluation of Thompson Sampling (NeurIPS)]] · [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[Unbiased Learning-to-Rank with Biased Feedback]] · [[Double-Debiased Machine Learning for Treatment and Structural Parameters]] · [[Counterfactual Reasoning and Learning Systems]] · [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[On the Difficulty of Evaluating Baselines]] · [[Uncertainty]]

New topics worth writing: DR with Optimistic Shrinkage, Switch estimator, Self-normalised importance weighting, OPE estimator selection without ground truth, Click models for position bias, Slate-action off-policy evaluation, Bootstrap for estimator variance, Relative cross-entropy as a CTR model metric
