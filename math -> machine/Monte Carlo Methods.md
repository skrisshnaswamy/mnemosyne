---
aliases:
  - Monte Carlo
  - Monte Carlo Method
  - Monte Carlo Estimation
  - Monte Carlo Simulation
  - Monte Carlo Return
  - Monte Carlo Rollouts
  - MC Methods
tags:
  - reinforcement-learning
  - probability
  - decision-sciences
  - fundamentals
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Can't compute an expectation? **Sample it and take the average.** The error shrinks like $1/\sqrt{N}$ — no matter how many dimensions the problem has.
> **Metaphor:** Throwing darts at a square with a circle drawn in it, and counting.
> **Where it bites:** Estimating a state's value by averaging real returns, rollouts in game search, simulation of anything — and the 100×-more-samples-for-10×-less-error bill.

---
Here's a square, with a circle drawn snugly inside it. You have a bucket of darts and **no formula for the area of a circle**.

You throw 2,000 darts at the square, blindfolded, so they land anywhere with equal chance. **1,540** of them land inside the circle.

~={blue}What's $\pi$?=~

---
# Darts 🎯

The circle takes up $\pi/4$ of the square. So the *fraction* of darts inside should be about $\pi/4$:

$$\pi \approx 4 \times \frac{1540}{2000} = \mathbf{3.080}$$

Not great — but you did no geometry at all. You replaced a calculation with **counting**.

![[monte_carlo_pi.png]]
> [!TIP] Reading the chart
> **Right:** five separate runs, all funnelling toward $\pi$ inside the orange $\pm 2$ standard-error band. With 1,000 darts the standard error is **0.052**; with 100,000 it's **0.0052**. ~={red}A hundred times the darts bought ten times the accuracy.=~ That's the $1/\sqrt{N}$ law, and it's both the method's great strength and its great cost.

> [!NOTE] Monte Carlo method
> Estimate a quantity that's defined as an expectation by drawing random samples and averaging: $\;\mathbb{E}[f(X)] \approx \frac{1}{N}\sum_i f(x_i)$. ^monte-carlo-def

> [!SUCCESS] Core idea
> ~={pink}The error depends on the *variance* of what you're averaging and on $N$ — and not on the dimension of the space.=~ A 1,000-dimensional integral costs the same number of samples as a 1-dimensional one with the same variance. That single property is why Monte Carlo is the standard escape from the [[Curse of Dimensionality]] — "[[Markov Chain Monte Carlo#^dont-compute-sample|don't compute, sample]]". ^error-is-dimension-free

---
# In RL: the value of a state is an average you can just *take*

A [[Value Function|value]] is an expectation — *the expected [[Discount Factor|return]] from here*. With no model you can't compute it. So:

1. Play a whole episode.
2. For each state you visited, note the return you actually collected **from that point on**.
3. Average those, over many episodes.

The shop manager in [[Decision Sciences#The leap — you can skip Table A entirely|the chapter]] did exactly this. Four times they hit `Low`, ordered stock, and cleared £38, £45, £37, £41:

$$\hat{Q}(\text{Low}, \text{order}) = \frac{38 + 45 + 37 + 41}{4} = £40.25$$

No transition probabilities. No Bellman equation. Just experience, averaged.

And you don't need to store the list. Keep a running mean:

$$\text{new average} = \text{old average} + \frac{1}{n}\,(\text{new sample} - \text{old average})$$

Hold on to that shape — *old estimate, plus a fraction of the surprise*. Swap $\frac{1}{n}$ for a constant $\alpha$ and it's the update rule of nearly everything that follows.

| Monte Carlo value estimation | |
|---|---|
| ✅ **Needs no model** | only sampled episodes |
| ✅ **Unbiased** | the target is a *real* return, not an estimate |
| ✅ Doesn't care whether the state is [[Markov Property\|Markov]] | it never looks at the next state's value |
| ❌ **High variance** | a return is the sum of many random steps |
| ❌ **Must wait for the episode to end** | useless for tasks that never end, slow for long ones |
| ❌ Credit is smeared across the whole episode | see [[Credit Assignment]] |

---
# The same trick, all over the vault

| Where | What's being sampled |
|---|---|
| [[Markov Chain Monte Carlo]] | a posterior you can't normalise — when you can't even draw samples *directly* |
| [[Particle Filter]] | a belief over a hidden state |
| [[Monte Carlo Tree Search]] | the outcome of a game, by playing it out at random from a position |
| [[Thompson Sampling]] | one plausible world, from your belief about the arms |
| [[Policy Gradient]] | the gradient of expected return — estimated from sampled trajectories |
| [[GRPO]] | a baseline — the average score of 8 sampled answers to the *same* prompt |
| Mini-batch SGD | the gradient over the full dataset, estimated from 32 examples → [[Random variable]] |

> [!WARNING] Two things Monte Carlo is not
> **It isn't "guessing".** It's an estimator with a known, shrinking error bar — you can say exactly how wrong it's likely to be. **And it isn't cheap to make precise.** One more decimal place costs **100×** the samples. When a Monte Carlo estimate is too noisy, the productive fix is almost never "more samples" — it's **reducing the variance** of what you're averaging: subtract a baseline ([[Policy Gradient#^baseline]]), bootstrap from a value estimate ([[Temporal Difference Learning]]), or sample more cleverly. ^variance-not-samples

---
> [!SUCCESS] If you remember one thing
> If it's an expectation, you can **sample it and average**. ~={pink}The error falls as $1/\sqrt{N}$ whatever the dimension — so the game is never "more samples", it's "less variance per sample".=~

---
# ⁉️
Monte Carlo has one deeply annoying limitation: it learns nothing until the final whistle. You have to finish the episode before a single estimate moves. But you often know you've messed up *long* before the end. Can you learn **during** the game?

→ [[Temporal Difference Learning]]
