---
aliases:
  - OPE
  - Counterfactual Evaluation
  - Counterfactual Estimation
  - Inverse Propensity Scoring
  - Inverse Propensity Weighting
  - IPS
  - IPW
  - SNIPS
  - Doubly Robust
  - Doubly Robust Estimator
  - Propensity Score
  - Propensity
  - Logged Bandit Feedback
tags:
  - bandits
  - experimentation
  - evaluation
  - causal-inference
  - decision-sciences
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Estimate how a **new** policy would have performed using logs collected by the **old** one — by re-weighting each logged reward by *how much more (or less) often the new policy would have made that choice*.
> **Metaphor:** Correcting a biased poll. If young voters were over-sampled 3×, count each of their answers ⅓ as much.
> **Where it bites:** Evaluating recommenders, rankers and ad policies offline. The reason *"our offline metric went up and the A/B test went down"* happens — and the reason to **log propensities**.

---
A news site has two kinds of reader, in equal numbers, and two kinds of article:

| | shown **sport** | shown **finance** |
|---|---|---|
| **sport fan** clicks | 30% | 5% |
| **finance fan** clicks | 5% | 20% |

The live system is sensible: it shows people what matches their taste **97%** of the time.

Someone proposes a radical new policy: *"finance articles get higher ad rates — just show finance to **everyone**."* Before risking an [[AB Testing|A/B test]], you check the logs. Across every impression where a finance article was shown, the click rate is a healthy **19.5%**.

~={blue}So the new policy should get about 19.5%. …Shouldn't it?=~

---
# Correcting the poll 🗳️

It would get **12.5%**. Half the audience are sport fans who click finance 5% of the time: $\frac{1}{2}(5\%) + \frac{1}{2}(20\%) = 12.5\%$.

The logs said 19.5% because of **who was shown finance**. The old policy showed it almost exclusively to people who like it. That number isn't "how finance articles perform" — it's "how finance articles perform *on the audience the old policy hand-picked for them*."

It's a biased poll. And you fix a biased poll by **re-weighting**. If a group was under-sampled, count each of its members more. Here: a sport fan being shown finance was *rare* under the old policy (3% of the time). Under the new policy it would happen 100% of the time. So each such logged row should count for

$$w = \frac{\pi_{\text{new}}(a \mid x)}{\pi_{\text{old}}(a \mid x)} = \frac{1.0}{0.03} \approx 33 \times$$

and each finance-fan-shown-finance row for $\frac{1.0}{0.97} \approx 1.03\times$. Average the re-weighted rewards:

$$\hat{V}_{\text{IPS}} = \frac{1}{n}\sum_i \frac{\pi_{\text{new}}(a_i \mid x_i)}{\pi_{\text{old}}(a_i \mid x_i)}\; r_i$$

![[ope_ips_vs_naive.png]]
> [!TIP] Reading the chart
> 600 simulated logs of 5,000 visits each. The **naive** estimate (red) is tightly clustered — around **0.195**. Precisely, repeatably *wrong*. **IPS** (blue) is centred on the truth, **0.125** — but spread almost twice as wide (±0.014 vs ±0.008), because it leans on a handful of rows each counted 33 times.

> [!NOTE] Off-policy evaluation
> Estimating the expected reward of a target policy from data logged under a different (behaviour) policy. **Inverse propensity scoring (IPS)** re-weights each observation by the ratio of the two policies' probabilities of the logged action. It's the [[On-Policy vs Off-Policy#The correction — importance sampling|importance-sampling correction]], applied to bandit logs. ^ope-def

> [!SUCCESS] Core idea
> ~={pink}The naive estimate is confidently wrong; IPS is noisily right.=~ Logged data isn't a sample of *the world* — it's a sample of **the world as filtered by the old policy's choices**. Any metric computed on it inherits that filter unless you divide it back out. ^confidently-wrong-vs-noisily-right

---
# What it needs — and this is the practical part

> [!WARNING] Two conditions, both non-negotiable
> 1. **The propensity must be logged.** $\pi_{\text{old}}(a \mid x)$ — the probability with which the shown item was chosen — recorded **at serving time**. You generally cannot reconstruct it later: the model has been retrained, the features have changed, the randomisation seed is gone. [[Contextual Bandit#^log-the-propensity|Log it. Always.]]
> 2. **The old policy must have explored.** If it *never* showed finance to a sport fan, that weight is $1/0$ — there's simply no data about what would happen, and no estimator can conjure it. It's an [[Observability#^plumbing-not-tuning|is-the-information-even-there]] problem. A purely deterministic production system produces logs that are **useless** for evaluating anything different. ^ope-requirements

This is the hard-nosed business case for keeping a little randomisation in production: it isn't only [[Exploration vs Exploitation|exploration]], it's what makes every *future* offline evaluation possible.

---
# Taming the variance

IPS is unbiased, but a few rows with weight 33 — or 3,000 — can dominate the whole estimate.

| Estimator | Idea | Trade |
|---|---|---|
| **IPS** | re-weight by $\pi_\text{new}/\pi_\text{old}$ | unbiased · high variance |
| **Clipped IPS** | cap the weights at some maximum | a little bias · much less variance |
| **SNIPS** (self-normalised) | divide by the *sum* of weights, not by $n$ | tiny bias · more stable · invariant to shifting the rewards |
| **Direct method** | fit a reward model $\hat{r}(x,a)$, score the new policy on it | low variance · **biased wherever the model is wrong** — which is exactly where the old policy rarely went |
| **Doubly robust** | direct method **+** an IPS correction on its residuals | right if *either* the reward model *or* the propensities are right |

→ [[Doubly Robust Policy Evaluation and Learning]] · [[Counterfactual Risk Minimization]] · [[Open Bandit Dataset and Pipeline- Towards Realistic and Reproducible Off-Policy Evaluation]]

---
# Where this shows up 🔗

- **"Offline metrics up, A/B test flat."** The most common version of this whole note. Your offline [[NDCG]] was computed on items the *old* ranker chose to show. → [[Recommendations as Treatments- Debiasing Learning and Evaluation]], [[Unbiased Learning-to-Rank with Biased Feedback]]
- **Position bias** in ranking: things shown at the top get clicked because they're at the top. Same disease, same cure.
- **[[AB Testing|A/B test]] logs** are the best OPE data you can own — propensities are exactly 0.5 and everything was explored.
- **[[Offline RL]]** is the multi-step version. There the weights *multiply* along the trajectory, so the variance explodes with horizon — which is why long-horizon OPE is still hard.
- **[[System Identification#^closed-loop-confounding|Identifying a plant from closed-loop data]]** is the control engineer's version of the same trap.
- The foundational paper is in the vault: [[Counterfactual Reasoning and Learning Systems]].

> [!WARNING] "We validated it offline on last month's logs"
> Ask: *logs of whose decisions?* If the new model is scored on data the old model selected, and nobody re-weighted, the result mostly measures **how much the new model agrees with the old one**. ^validated-on-whose-logs

---
> [!SUCCESS] If you remember one thing
> Logs record what happened **under the old policy's choices**. ~={pink}To ask "what if we'd chosen differently?", divide out how the choices were made — which you can only do if you wrote the probabilities down at the time.=~

---
# ⁉️
One more member of the bandit family — for when each "pull" is a training run costing six GPU-hours, the arms form a continuum, and nearby arms give similar results.

→ [[Bayesian Optimization]]
