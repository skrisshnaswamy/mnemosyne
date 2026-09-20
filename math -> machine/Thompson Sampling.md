---
aliases:
  - Posterior Sampling
  - Probability Matching
  - Beta-Bernoulli Bandit
  - Bayesian Bandit
  - Thompson
tags:
  - bandits
  - bayesian
  - decision-sciences
  - probability
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Keep a **belief** about each option. Each round, **draw one random sample from every belief, play whichever sample is highest**, observe, update. Uncertain options get tried *because* their samples swing wide.
> **Metaphor:** Each morning, imagine one plausible version of the world — then act as if it's the real one.
> **Where it bites:** The default bandit algorithm in industry — ads, recommendations, experiment platforms. Copes with delayed and batched feedback; needs almost no tuning.

---
Three email subject lines. You've sent a few of each:

| Subject | Sent | Opened | Open rate so far |
|---|---|---|---|
| A | 100 | 30 | 30% |
| B | 10 | 3 | 30% |
| C | 100 | 22 | 22% |

A and B are tied on **30%**. Which should the next email use?

A pure exploiter flips a coin between them. But you *know* they're not equally well understood: A's 30% rests on a hundred sends, B's on ten. B's true rate could easily be 15% — or 50%.

[[Upper Confidence Bound|UCB]] would handle this with a confidence-interval formula. ~={blue}Is there a way to get the same behaviour without deriving any formula — just from the beliefs themselves?=~

---
# Imagine a world, act in it 🎲

Yes — and it's almost embarrassingly simple.

Hold a [[Beliefs|belief]] about each subject line's true open rate. For an open/didn't-open outcome, the natural one is a **Beta** distribution: $\text{Beta}(1 + \text{opens},\; 1 + \text{non-opens})$.

- A: $\text{Beta}(31, 71)$ — mean 0.30, tight: ±0.045
- B: $\text{Beta}(4, 8)$ — mean 0.33, **wide: ±0.13**

Now, each round:

1. **Draw one random number from each belief.** A might give 0.29. B might give 0.47 — or 0.18; it swings a lot.
2. **Send whichever subject line drew highest.**
3. Observe open / no open. **Update that belief** (add one to the right count).

That's the whole algorithm. No $\varepsilon$, no bonus term, no schedule.

> [!NOTE] Thompson sampling
> At each round, sample a parameter from the posterior of every arm, act greedily with respect to the *samples*, and update the posterior with the outcome. Equivalently: **play each arm with exactly the probability that it's the best one**, given what you've seen ("probability matching"). William Thompson, 1933. ^thompson-def

> [!SUCCESS] Core idea
> ~={pink}Wide beliefs get tried; narrow ones settle — and the exploration rate tunes itself.=~ B wins the draw often *because* its belief is wide. Every time it's tried, the belief narrows. If it's really good it keeps winning; if it isn't, its samples stop reaching the top and it fades out. Nobody set a schedule — [[Uncertainty#^epistemic|the uncertainty itself]] is the schedule. ^exploration-tunes-itself

---
# Watching the beliefs sharpen

![[thompson_posteriors.png]]
> [!TIP] Reading the chart
> Three machines paying 30% / 50% / 70% (dotted lines). **After 10 pulls** the beliefs are broad smears and the pulls are spread 2 / 2 / 6. **After 1,000**, the best machine has had **950** pulls and its belief is a needle on 0.70. The 30% machine got **14** pulls in total — and look how *wide* its belief still is. ~={blue}It never learned that arm precisely. It only learned enough to be sure it isn't the best.=~ That's the efficiency.

In [[Regret|the regret experiment]] this came out at **18** after 10,000 pulls — against 98 for UCB1 and 210 for ε-greedy.

---
# Why it became the industry default

| Property | Why it matters in production |
|---|---|
| **Randomised** | feedback is usually **delayed and batched** — you serve 10,000 requests before any reward comes back. Deterministic [[Upper Confidence Bound\|UCB]] sends the *entire batch* to one arm; Thompson naturally spreads it |
| **Trivial to implement** | two counters per arm and a Beta sampler |
| **Almost nothing to tune** | the prior, and it washes out fast |
| **Works with any model you can sample from** | a Bayesian logistic regression, a bootstrapped ensemble, dropout at inference → contextual and deep versions for free → [[Contextual Bandit]] |
| **Strong in practice** | regularly matches or beats UCB in empirical studies → [[An Empirical Evaluation of Thompson Sampling (NeurIPS)]], [[A Tutorial on Thompson Sampling]] |
| **Handles drift** | discount old counts and the beliefs re-widen on their own |

---
# The same move, elsewhere 🔗

- [[Particle Filter]] *holds* a belief as a cloud of samples. Thompson *acts* on a single one.
- [[Sampling Parameters|Sampling a token]] instead of taking the argmax is the same instinct: when you're genuinely unsure, don't commit to the mode.
- [[Bayesian Optimization]] has a Thompson variant: sample one plausible *function*, then optimise that.
- In full RL it's **posterior sampling RL** (PSRL): sample one plausible MDP, solve it, act on it for an episode.

> [!WARNING] "It explores by adding noise"
> ε-greedy adds noise — uniformly, blindly, forever. Thompson's randomness is **shaped by what you know**: lots of it where you're ignorant, almost none where you're sure, and it shrinks on its own as evidence arrives. That distinction — *undirected* vs *uncertainty-directed* exploration — is the whole reason its regret curve bends and ε-greedy's doesn't. ^noise-vs-directed

> [!TIP] One thing to watch
> If your belief is **mis-specified** — you assumed independent arms and they aren't, or a fixed rate and it's drifting — Thompson will be confidently wrong in exactly the way your model is. The algorithm is only as calibrated as the posterior you hand it.

---
> [!SUCCESS] If you remember one thing
> **Sample from each belief. Play the best sample. Update.** ~={pink}Uncertainty does the exploring for you — and switches itself off once you know enough.=~

---
# ⁉️
All of this assumed there is *one* best subject line for everyone. But what if line A works on new customers and line B works on loyal ones — so that the "best arm" depends on who's asking?

→ [[Contextual Bandit]]
