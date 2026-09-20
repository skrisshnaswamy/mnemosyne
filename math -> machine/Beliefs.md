---
aliases:
  - Belief
  - Belief State
  - Posterior
tags:
  - fundamentals
  - probability
  - bayesian
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A belief isn't a guess — it's a **whole distribution** over what might be true, and it gets *updated* by evidence rather than replaced.
> **Metaphor:** A detective's suspect board. Not one name — every suspect, each with a confidence, all shifting as clues arrive.
> **Where it bites:** POMDPs, Thompson sampling, the [[Kalman Filter]], and anywhere the state is hidden and you only get noisy hints about it.

---
# A guess vs. a belief

Ask a normal model "what's the temperature outside?" and it says **21°C**. One number. A point estimate.

Ask something that holds a *belief* and you get a different kind of answer:

> "Probably around 21. Could plausibly be 19 or 23. Almost certainly not 5 or 40."

That second answer is a **distribution**, not a number. And it carries strictly more information — it contains the guess (21 is still the peak) *plus* how much you should trust it.

> [!NOTE] Belief
> Your current state of knowledge about something you **cannot directly observe**, held as a full probability distribution over all the values it might take. Not "what it is" — ~={blue}"what it might be, and how strongly I lean toward each option."=~ ^belief-def

That "cannot directly observe" is the important qualifier. You don't need a belief about things you can just *look at*. Beliefs exist because the world hides things from you and only leaks noisy hints. 🕵️

---
# The detective's board 🕵️

Picture the classic corkboard with photos and red string.

At the start of a case the detective has **six suspects**, and no real reason to prefer any. That's the **prior** — what you think before the evidence arrives. Roughly flat, six equally-sized pins.

Then a clue lands: *the culprit was left-handed.*

Now watch carefully, because this is the whole mechanic. The detective does **not** throw away the board and write down one name. They **reweight** it. Two suspects were left-handed — their pins get bigger. The other four don't get removed, they get *smaller*. Still on the board. Still possible, because maybe the witness was wrong.

That reweighted board is the **posterior** — what you believe *after* the evidence.

And here's the part people miss: **tomorrow's prior is today's posterior.** The next clue arrives and reweights the board again, starting from where it already is. The belief is a running thing that never resets. ^belief-is-running

> [!SUCCESS] Core idea
> Evidence doesn't **replace** a belief. It **reshapes** it.
> $$\text{prior} \;\xrightarrow{\;\text{evidence}\;}\; \text{posterior} \;\xrightarrow{\;\text{more evidence}\;}\; \text{better posterior}$$

---
# The update rule

The maths for "reweight the board" is Bayes' rule, and it's less scary than it looks:

$$\underbrace{P(H \mid E)}_{\text{new belief}} \;=\; \underbrace{P(H)}_{\text{old belief}} \times \underbrace{\frac{P(E \mid H)}{P(E)}}_{\text{how surprising this evidence is, if } H \text{ were true}}$$

Read it as a sentence rather than symbols:

> **New belief = old belief × how well this hypothesis explains what I just saw.**

Take one suspect. Old belief: $1/6$. Now, *if they were guilty*, how likely was it that we'd find a left-handed clue? Very likely (they're left-handed) — so that fraction is big, and their pin grows. For a right-handed suspect, that fraction is tiny, and their pin shrinks.

The denominator $P(E)$ is just bookkeeping — it makes everything add back up to 1 so the board stays a valid distribution.

> [!TIP] Two things the formula quietly tells you
> 1. **A strong prior is hard to move.** If your old belief was near zero, multiplying by anything finite keeps it near zero. This is why you need *overwhelming* evidence to overturn a firmly held belief — and it's the same maths as a strong [[Regularization#The deeper way to see it: a regularizer is a prior|regularization prior]] refusing to budge for small data.
> 2. **Surprising evidence moves you most.** If the evidence was expected under *every* hypothesis, that ratio is ~1 for everyone, everything scales equally, and your belief barely changes. Evidence that fails to discriminate teaches you nothing. ^surprising-evidence

---
# You've already built one of these

Go back to the [[Kalman Filter]] note and re-read it with this vocabulary. It's the same machine:

| Kalman filter language | Belief language |
|---|---|
| The physics prediction | The **prior** — what I expect before looking |
| The noisy radar measurement | The **evidence** |
| The **Kalman Gain** | How much this evidence should move me |
| The updated estimate + its uncertainty | The **posterior** |
| Feed that estimate into the next timestep | Today's posterior becomes tomorrow's prior |

> [!SUCCESS] The Kalman filter *is* a belief tracker
> It's Bayes' rule, run every timestep, with the simplifying assumption that every distribution involved is a **Gaussian**. That assumption is exactly why it collapses to clean arithmetic instead of an intractable integral — a Gaussian belief needs only two numbers (mean and variance), so "updating the board" becomes updating two numbers. ^kalman-is-bayes

That's the general pattern, actually. Beliefs are conceptually simple and computationally brutal — the honest update requires an integral over every possible world. So every practical method is a different bargain about how to cheat:
- **Assume it's Gaussian** → [[Kalman Filter]] (exact, fast, only works if the world cooperates)
- **Approximate with samples** → [[Markov Chain Monte Carlo]], [[Hamiltonian Monte Carlo]] (general, slow)
- **Approximate with a simpler distribution** → variational inference, minimising [[KL Divergence]] to the truth

---
# Belief states: when the state itself is hidden

This is where beliefs stop being a statistics topic and become the core of a whole class of problems.

A [[Markov Decision Process|MDP]] assumes you can **see the state**. You're in state $s$, you pick an action, you land in state $s'$. Fine.

But most real problems aren't like that. A robot doesn't know where it is — it knows what its camera saw. A doctor doesn't know the disease — they know the symptoms. A recommender doesn't know what you want — it knows what you clicked.

The state is **hidden**. You only get noisy observations of it. That's a **POMDP** (Partially Observable MDP).

And here's the beautiful trick that makes POMDPs tractable at all:

> [!SUCCESS] The belief state trick
> You can't act on the true state — you can't see it. So instead, ~={pink}**treat your belief itself as the state.**=~
> A POMDP over hidden states becomes an ordinary MDP over *beliefs*. You've traded a small discrete space you can't observe for a big continuous one you fully can — because you always know exactly what you currently believe. ^belief-state-trick

This is also what makes multi-agent problems so hard. In a [[Dec-POMDP]], each agent has its own belief, *and* has to maintain beliefs about the other agents' beliefs — which are themselves changing as those agents learn. That recursion ("what do I think they think I think?") is the thing that makes [[Multi-Agent Reinforcement Learning|MARL]] genuinely difficult, not just RL with more robots.

---
# Acting on a belief

If you're holding a whole distribution, how do you choose one action?

**Option 1 — collapse it and act.** Take the mean, act as if it's true. Simple, and it throws away the uncertainty, which means you'll never explore.

**Option 2 — be optimistic.** Act on the *best plausible* value in your belief (UCB). Bold where you're unsure.

**Option 3 — sample from it.** Draw one plausible world from your belief and act as if that's reality. This is **Thompson sampling**, and it's lovely because exploration comes out for free: a wide belief produces wildly varying samples, so you naturally try different things; a narrow belief produces near-identical samples, so you settle down. ~={blue}The exploration rate self-tunes as you learn.=~

> [!TIP] Why this connects back
> All three are answers to the *epistemic* half of [[Uncertainty]]. Option 1 ignores it, options 2 and 3 exploit it. That's why "belief" and "uncertainty" keep showing up in the same sentence — uncertainty is the *width* of the belief, and the width is what tells you whether to gather more information or commit.

---
> [!SUCCESS] If you remember one thing
> A belief is a **distribution that survives contact with evidence** by reshaping instead of resetting. Point estimates forget; beliefs accumulate.

---
# ⁉️
Once you're carrying a belief and acting on it, you're making decisions under uncertainty over time — which is exactly the problem [[Markov Decision Process|MDPs]] and the wider story in [[Decision Sciences]] were built to formalise.
