---
aliases:
  - Bayesian Optimisation
  - BayesOpt
  - Bayes Opt
  - Acquisition Function
  - Expected Improvement
  - Gaussian Process Optimization
  - GP-UCB
  - Surrogate Model
  - Hyperparameter Optimization
  - Hyperparameter Optimisation
  - Hyperparameter Tuning
  - Successive Halving
  - Hyperband
tags:
  - bandits
  - bayesian
  - training
  - decision-sciences
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** When every trial is **expensive**, fit a cheap model of *"score vs settings"* that also knows **where it's unsure** — then run the next trial wherever looks most **promising or unknown**.
> **Metaphor:** Prospecting. Each drill hole costs a fortune, so the geologist keeps a map of estimated richness *and* of how little she knows, and drills where those two add up best.
> **Where it bites:** Hyperparameter tuning, PID gains, A/B-testing a continuous knob, drug and materials design — anywhere you get ~20 tries, not 20,000.

---
You're tuning a learning rate. Each training run takes **six GPU-hours**. You have budget for twenty.

So far you've done five, scattered across the range:

| log₁₀(lr) | −4.8 | −4.0 | −2.4 | −1.7 | −1.1 |
|---|---|---|---|---|---|
| val loss | 0.69 | 0.50 | **0.27** | 0.65 | 0.86 |

Best so far: $10^{-2.4}$. You have fifteen runs left.

~={blue}Where do you put the sixth? Right next to −2.4 to refine it? Or in that big unexplored gap between −4.0 and −2.4, where *anything* could be hiding?=~

---
# Prospecting ⛏️

That's [[Exploration vs Exploitation]] again — but with a twist that makes it far more tractable than slot machines: **the arms are related.** A learning rate of $10^{-2.5}$ will behave very like $10^{-2.4}$. So one trial teaches you about a whole *neighbourhood*.

Bayesian optimisation exploits that with two pieces:

**1 · A surrogate model** — a cheap stand-in for the expensive function. Usually a **Gaussian process**, which returns *two* things for any setting: a **best guess** and an **uncertainty**. It's pinned tight at the points you've measured and balloons in the gaps.

**2 · An acquisition function** — a rule that turns (guess, uncertainty) into a single *"how worthwhile is a trial here?"* score. You run the next trial at its maximum.

![[bayesopt_gp_ei.png]]
> [!TIP] Reading the chart
> **Top:** black dots are the five runs. The blue line is the surrogate's best guess; the band is its doubt — zero at each dot, widest in the gap. The dashed grey line is the truth, which the optimiser never sees. **Bottom:** *expected improvement* peaks at **lr = 10⁻²·⁹⁴** — in the gap, leaning toward the current best. The true optimum is at $10^{-2.71}$. Neither "refine next to −2.4" nor "dead centre of the gap" — it's the principled blend of both.

> [!NOTE] Bayesian optimisation
> Sequential, model-based optimisation of an expensive black-box function: maintain a probabilistic surrogate of the objective, choose the next evaluation by maximising an acquisition function, evaluate, update, repeat. → [[Practical Bayesian Optimization of Machine Learning Algorithms]] ^bayesopt-def

> [!SUCCESS] Core idea
> ~={pink}Spend cheap compute deciding where to spend expensive compute.=~ It's a [[Multi-Armed Bandit|bandit]] with infinitely many arms that *share information*, and every acquisition function is just a different recipe for mixing **"looks good"** with **"don't know yet"**. ^cheap-compute-for-expensive

---
# The acquisition functions are the bandit algorithms

| Acquisition | Rule | Its bandit twin |
|---|---|---|
| **Expected Improvement (EI)** | how much do I expect to *beat my current best* by, averaged over my uncertainty? | the workhorse default |
| **GP-UCB** | guess + $\beta \times$ uncertainty (flip the sign when minimising) | [[Upper Confidence Bound]] → [[Gaussian Process Optimization in the Bandit Setting (GP-UCB)]] |
| **Thompson** | draw one plausible function from the surrogate; optimise *that* | [[Thompson Sampling]] — and trivially parallel |
| **Probability of Improvement** | chance of beating the best at all | too greedy; rarely used alone |

---
# What people actually use

| Method | Idea | Reach for it when |
|---|---|---|
| **Grid search** | try every combination | ≤ 2 hyperparameters. It's the [[Curse of Dimensionality]] in a lab coat |
| **Random search** | sample configurations at random | the baseline to beat. Far better than grid, because usually only a couple of hyperparameters matter |
| **Bayesian optimisation (GP)** | the above | trials are **expensive**, ≲ 20 dimensions, mostly sequential |
| **TPE** | model "good" and "bad" configurations as two densities | Optuna's default; copes with categorical and conditional parameters |
| **Successive halving / Hyperband / ASHA** | start many configurations on a small budget, keep the best half, double their budget, repeat | trials are **cheap to start and early results are predictive**; lots of parallel workers |
| **Population-based training** | evolve hyperparameters *during* training | you want schedules rather than constants |

> [!TIP] Successive halving is a bandit too
> It's **best-arm identification** where each "pull" is *a few more epochs* rather than a fresh trial → [[Regret#Two kinds of regret — and they want different algorithms|simple regret]]. It saves compute by killing losers early — and its blind spot is the slow starter that would have won in the end (a low learning rate always looks bad at epoch 2).

---
# Practicalities

- **Search on a log scale** for learning rates, weight decay, regularisation strength. The chart's x-axis is $\log_{10}$ for a reason.
- **The objective is noisy.** Two runs with different seeds differ; tell the surrogate so, or it will chase noise.
- **It's sequential by nature.** Twenty workers and a strictly one-at-a-time optimiser is a waste; use batch / Thompson variants, or switch to ASHA.
- **It fades above ~20 dimensions** — GPs need data to cover the space, and [[Curse of Dimensionality|there's a lot of space]].
- **You're tuning against validation data.** Tune hard enough and you overfit *it* — same selection effect as [[Q-Learning#^maximisation-bias|maximisation bias]]. Keep a test set the tuner never sees.

> [!WARNING] "Bayesian optimisation always beats random search"
> With **cheap** trials, **many** dimensions or **lots of parallel compute**, random search or ASHA is often as good or better — and much simpler to run. BO earns its keep in one specific regime: ~={red}few, expensive, sequential trials in a modest number of dimensions.=~ That happens to describe tuning a large model rather well. ^bo-regime

---
> [!SUCCESS] If you remember one thing
> **A model of the score *and* of your ignorance, plus a rule that trades them off.** ~={pink}When each try costs hours, thinking hard about where to try next is the cheapest compute you'll ever spend.=~

---
# ⁉️
Look at what that surrogate did: from five measurements it produced an estimate for *every* learning rate — without trying them all. Generalising from the visited to the unvisited is precisely the move that rescues RL from its table.

→ [[Function Approximation]]
