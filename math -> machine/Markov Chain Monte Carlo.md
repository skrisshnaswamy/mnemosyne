---
aliases:
  - MCMC
tags:
  - MCMC
  - probability
  - bayesian
link: https://arxiv.org/pdf/1909.12313
keywords: Metropolis-Hastings
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** When you can't *compute* a distribution, take a **random walk** that visits each region in proportion to its probability — then just use the samples.
> **Metaphor:** Mapping a mountain range in fog by walking it, preferring uphill but sometimes stepping down. Where you spend your time *is* the map.
> **Where it bites:** Bayesian posteriors, anything where the normalising constant is intractable. Diagnostics: burn-in, $\hat{R}$, effective sample size.

---
# The problem — the denominator is impossible

Recall Bayes' rule from [[Beliefs#The update rule|Beliefs]]:

$$P(\theta \mid D) = \frac{P(D \mid \theta) \, P(\theta)}{P(D)}$$

The numerator is usually easy — you can evaluate the likelihood and the prior for any given $\theta$.

The denominator is $P(D) = \int P(D \mid \theta)P(\theta)\, d\theta$: an integral over **every possible value of every parameter**. In 2 dimensions you could grid it. In 50 dimensions, a grid with 10 points per axis needs $10^{50}$ evaluations. That's not "slow", that's ~={red}never, on any hardware, ever.=~

So the posterior exists, and you can compare any two points on it — but you cannot *normalise* it, and therefore cannot sample from it directly.

> [!SUCCESS] The escape hatch
> You almost never actually need the distribution itself. You need **expectations** from it — a mean, a variance, a credible interval, a prediction.
>
> And expectations can be approximated by **averaging over samples**. So: stop trying to compute the distribution. ~={blue}Find a way to draw samples from it instead.=~ ^dont-compute-sample

That's the **Monte Carlo** half — approximate a hard integral by averaging random draws. The **Markov Chain** half is the trick that gets you those draws.

---
# The walk in the fog 🌫️

You're on a mountain range in thick fog. You can't see the landscape. But wherever you stand, you can measure the **altitude right here** — and you can measure it at a spot one step away.

Goal: spend time in each region **proportional to its altitude**, so that a log of where you walked becomes a map of the range.

The rule (**Metropolis-Hastings**) is remarkably simple:

1. **Propose** a random step from where you are.
2. If the proposed spot is **higher** → always move there. ⬆️
3. If it's **lower** → move there anyway, but only *sometimes* — with probability equal to the ratio $\frac{\text{new height}}{\text{old height}}$. ⬇️
4. Otherwise stay put (and record your current spot **again**).
5. Repeat, logging every position.

Run it long enough and the histogram of visited positions converges to the true shape of the landscape.

> [!NOTE] Why the ratio is the whole trick
> Step 3 uses a **ratio** of two densities — and the intractable $P(D)$ appears in both numerator and denominator, so it **cancels**. You never need the constant you couldn't compute. That single observation is what makes Bayesian inference practical. ^ratio-cancels-the-constant

> [!WARNING] Why accept downhill moves at all?
> If you only ever went uphill you'd climb the nearest peak and be stuck there forever — that's optimisation, not sampling. You'd report "the answer is exactly this" with no sense of spread.
>
> You're not looking for the summit. You're trying to **characterise the whole range**. Downhill moves are what let you explore, cross valleys, and find other peaks. ^why-downhill

The "Markov chain" part is just the [[Markov Property]]: each step depends only on where you currently are.

---
# The practical realities

MCMC works, but it needs babysitting. These four things come up in every conversation about it:

**Burn-in.** You start somewhere arbitrary, probably a low-probability region. Those first samples are junk — throw away the first 500–1000.

**Autocorrelation.** Consecutive samples are *neighbours*, not independent draws. 10,000 correlated samples might carry the information of 200 independent ones. That's the **Effective Sample Size (ESS)**, and it's the number you should actually report.

**Convergence.** Has the chain found the whole distribution, or is it stuck in one mode? Standard check: run several chains from different starting points and compare them — **$\hat{R}$ (R-hat)**. If it's above ~1.01, your chains disagree and you should not trust the result.

**The step size dilemma.** Steps too small → you crawl, and everything is autocorrelated. Steps too large → almost every proposal is rejected and you stand still. Both look like "running", neither makes progress. 😖

> [!WARNING] The real killer — random walks in high dimensions
> In 50+ dimensions, a randomly proposed direction is almost certainly *wrong*. Nearly all the posterior's mass sits in a thin shell (the "typical set"), and a blind random step almost always leaves it and gets rejected.
>
> Metropolis-Hastings degrades from "slow" to **"useless"** as dimension grows. This is the specific failure that motivated everything that came after. ^curse-of-dimensionality-mcmc

---
# ⁉️
The problem is that our walker is blind — it proposes directions at random and mostly wastes them. But we can usually compute the **[[Derivative#Gradient|gradient]]** of the log-density. Why not let the walker feel the slope under its feet?

→ [[Hamiltonian Monte Carlo]]
