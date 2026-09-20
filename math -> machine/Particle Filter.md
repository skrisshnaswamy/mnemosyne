---
aliases:
  - Particle Filters
  - Sequential Monte Carlo
  - SMC
  - Bootstrap Filter
  - Monte Carlo Localization
  - Monte Carlo Localisation
  - Resampling
  - Particle Degeneracy
tags:
  - estimation
  - decision-sciences
  - probability
  - bayesian
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Represent the belief as **thousands of guesses**. Move them all, score each against the sensor, then clone the good ones and drop the bad. No bell curve required.
> **Metaphor:** A search party on a moor. Every time a clue comes in, searchers in the implausible spots get reassigned next to those in the plausible ones.
> **Where it bites:** Robot localisation, tracking through clutter, any belief with **more than one bump** — and it dies in high dimensions.

---
A robot vacuum wakes up in a long corridor with **three identical doors**. It has a map. It has no idea where on the map it is.

Its sensor says: *"I'm next to a door."*

Good — that rules out most of the corridor. But which door? It genuinely could be any of the three. Its honest belief is *"here, **or** there, **or** over there"* — three separate bumps.

Try writing that as a [[Kalman Filter|mean and a variance]]. The "average" position is somewhere in the middle of a wall between doors, which is the one place it *definitely isn't*. ~={blue}So how do you carry around a belief with three humps?=~

---
# The search party 🔦

Stop trying to describe the belief with a formula. Just **keep a big bag of guesses**.

Scatter 2,000 imaginary robots — *particles* — evenly along the corridor. Each one is a hypothesis: *"maybe I'm here."* Then run the same two beats as any [[Bayes Filter]]:

1. **Predict — move them all.** The real robot drives 20 m, so every particle drives 20 m too, each with a bit of random wobble (your wheels slip; you don't know exactly how far you went).
2. **Update — score them.** For each particle ask: *"if I really were here, how likely is the sensor reading I just got?"* Particles next to a door score high when the sensor says DOOR. The rest score near zero.
3. **Resample — reassign the searchers.** Draw a fresh set of 2,000, picking each old particle with probability proportional to its score. High scorers get cloned several times. Low scorers vanish.

![[particle_filter_corridor.png]]
> [!TIP] Reading the chart
> **1:** no idea — only 7% of particles are near the truth (▲). **2:** "DOOR" → three equal piles, 29% at the right one. It *cannot* know more yet, and the belief honestly says so. **3:** drive 20 m — all three piles shift. **4:** "DOOR" again. Only the pile that started at the first door has arrived at *another* door. The other two are now standing next to blank wall, and die. **82%** of particles are within 3 m of the truth.

Nothing clever happened at any step. It never "worked out" which door it started at — ~={blue}the wrong hypotheses just failed to survive the evidence.=~

> [!NOTE] Particle filter
> A [[Bayes Filter]] that represents the belief as a set of weighted samples, updated by **propagate → weight by likelihood → resample**. Also called **Sequential Monte Carlo**. It can represent any distribution, at the price of needing many samples. ^particle-filter-def

> [!SUCCESS] Core idea
> ~={pink}Don't compute the belief — *populate* it.=~ Where the particles are dense, the probability is high. That's the same move as [[Markov Chain Monte Carlo#^dont-compute-sample|MCMC]] and [[Monte Carlo Methods]]: when the maths is intractable, replace it with a crowd of samples and count. ^populate-the-belief

---
# What goes wrong

| Problem | What it looks like | The usual fix |
|---|---|---|
| **Degeneracy** | after a few updates one particle holds ~all the weight; the other 1,999 are dead weight | that's what **resampling** is for |
| **Impoverishment** | resample too eagerly and you get 2,000 *copies of the same three* guesses — diversity gone | add jitter after resampling; only resample when the effective sample size drops |
| **Particle deprivation** | no particle happens to be near the truth, so nothing can ever be cloned there | inject a few random particles every step |
| **The kidnapped robot** | someone picks the robot up and moves it; every particle is now wrong and confident | same — random injection |

> [!WARNING] "Just use more particles"
> In 1-D, 2,000 particles is lavish. But the number you need grows **exponentially** with the dimension of the state — to cover a space you need particles along *every* axis at once. By ~10 dimensions it's hopeless without exploiting structure. That's the [[Curse of Dimensionality]], and it's why high-dimensional tracking still uses an [[Extended Kalman Filter|EKF/UKF]] where it can. ^particles-and-dimensions

---
# When to reach for it

| Your belief is… | Use |
|---|---|
| one bump, linear world | [[Kalman Filter]] |
| one bump, mildly curved world | [[Extended Kalman Filter]] |
| a few discrete possibilities | [[Hidden Markov Model]] |
| **several bumps, or a strange shape, in low dimensions** | **particle filter** |

> [!TIP] The same trick, elsewhere
> [[Thompson Sampling]] acts on **one sample** drawn from a belief. A particle filter *is* the belief, held as samples. And "propagate, score, keep the best, repeat" is also the skeleton of beam search and of evolutionary methods — a population pruned by evidence.

---
> [!SUCCESS] If you remember one thing
> A cloud of guesses, moved forward and pruned by each new piece of evidence. ~={pink}It can hold "here *or* there" — which no mean-and-variance ever can=~ — and it pays for that freedom in samples.

---
# ⁉️
The robot's position was continuous. But plenty of hidden states aren't a position at all — they're a **label**: raining or dry, bull market or bear, user browsing or buying. With only a handful of possibilities you don't need samples; you can keep an exact probability for each.

→ [[Hidden Markov Model]]
