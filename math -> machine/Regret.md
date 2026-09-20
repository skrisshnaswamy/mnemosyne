---
aliases:
  - Cumulative Regret
  - Regret Bound
  - Sublinear Regret
  - Linear Regret
  - Logarithmic Regret
  - Regret Minimisation
  - Regret Minimization
  - Simple Regret
  - No-Regret Learning
tags:
  - bandits
  - reinforcement-learning
  - decision-sciences
  - metrics
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** The reward you **missed out on** by not knowing the best action from the start. What matters is its *shape over time*: **linear** means you never stop paying; **logarithmic** means you learn.
> **Metaphor:** Tuition fees. Everyone pays something to learn. A good learner's bill flattens out.
> **Where it bites:** Comparing exploration strategies, reading a bandit paper, and the argument for a bandit over an A/B test.

---
Three ad banners. Their true click-through rates are **4%, 5% and 6%** — but you don't know that; you only see clicks.

You run them for 10,000 impressions and collect **540 clicks**.

Is that good?

You can't say from the number alone. What you *can* say is what an all-knowing marketer would have done: show the 6% banner every single time and collect **600**.

$$600 - 540 = \mathbf{60} \text{ clicks.}$$

~={blue}That gap has a name — and the interesting question isn't how big it is after 10,000 impressions. It's what it does over the *next* 10,000.=~

---
# Tuition fees 🎓

> [!NOTE] Regret
> The difference between the reward an oracle would have earned by always playing the best action, and what you actually earned:
> $$\text{Regret}(T) = T\mu^* - \sum_{t=1}^{T} \mathbb{E}[r_t]$$
> Each pull of an arm that's worse than the best by a gap $\Delta$ adds $\Delta$ to the bill. ^regret-def

You can't get it to zero — you didn't know the answer at the start, so *some* tuition is unavoidable. The question is whether you **keep** paying.

Here are four strategies on three slot machines paying out 30%, 50% and 70% of the time (bigger gaps than the banners, so the curves separate quickly):

![[regret_curves.png]]
> [!TIP] Reading the chart
> After 10,000 pulls: **pure greed 1,048 · ε-greedy 210 · UCB1 98 · Thompson sampling 18.** But the *shapes* are the lesson. Greedy and ε-greedy are **straight lines** — they pay the same fee every round, forever. UCB and Thompson **bend over and go flat** — the fee per round is heading to zero.

> [!SUCCESS] Core idea
> ~={pink}Linear regret means you never learned. Sub-linear regret means you did.=~ If regret grows like $\log T$, then regret *per round* $\to 0$ — you converge on the best arm and stop paying. That's the bar a good exploration strategy has to clear, and $O(\log T)$ is provably the best possible (Lai & Robbins, 1985). ^linear-vs-log

---
# Why each one has the shape it has

| Strategy | Shape | Why |
|---|---|---|
| **Greedy** | **linear** | in some fraction of runs it locks onto the wrong arm in the first few pulls and *never checks again* |
| **ε-greedy, fixed ε** | **linear** | it never stops exploring: 10% of pulls are random *forever*, including onto arms it knows are bad. Slope ≈ $\varepsilon \times$ average gap |
| **ε-greedy, decaying ε** | can be log | only if the decay schedule happens to match the (unknown) gaps |
| **[[Upper Confidence Bound\|UCB]]** | **logarithmic** | each bad arm gets tried just often enough to be *confident* it's bad — about $\frac{\log T}{\Delta^2}$ times — then left alone |
| **[[Thompson Sampling]]** | **logarithmic** | same guarantee, usually a much smaller constant in practice |

And where the pulls actually went:

![[bandit_pull_allocation.png]]

> [!WARNING] A guarantee about shape is not a guarantee about now
> Look again at **1,000 pulls**: ε-greedy's regret is **30** and UCB1's is **46**. *ε-greedy is winning.* UCB1's bound is about the long run; its constant is large, and at short horizons a simple heuristic can beat it. If you'll only ever have a few hundred decisions, "asymptotically optimal" may be the wrong thing to optimise for. ~={red}Always ask what $T$ actually is.=~ ^shape-not-constant

---
# Two kinds of regret — and they want different algorithms

| | **Cumulative regret** | **Simple regret** |
|---|---|---|
| Counts | every sub-optimal pull *along the way* | only the quality of your **final** pick |
| You care when | the experiment *is* the product — every user you show a bad variant to is a real loss | the experiment is a means to a one-off decision |
| Best approach | bandit — shift traffic to the winner as you learn | explore **evenly**; you want to be sure at the end |
| Example | headlines, ads, recommendations | choosing which model to ship → [[AB Testing]] |

This is the cleanest way to settle "bandit or A/B test?": *which regret are you being charged?*

---
# The bigger picture

Regret is how the **bandit** literature scores algorithms. Full RL mostly reports *sample complexity* ("how many episodes to reach near-optimal") instead — a related idea, harder to pin down, because in an [[Markov Decision Process|MDP]] a bad action can also put you somewhere bad.

**"No-regret learning"** in game theory uses the same word for a related guarantee: average regret against the best *fixed* action in hindsight goes to zero — even against an adversary.

> [!WARNING] Regret isn't about feeling bad, and zero isn't the goal
> It's an accounting identity against an oracle who knew the answer all along. You're *supposed* to have some. An algorithm with zero regret on your benchmark either got lucky or peeked. ^zero-regret-is-suspicious

---
> [!SUCCESS] If you remember one thing
> Don't ask how much regret — ask **what shape**. ~={pink}A straight line means it's still paying for ignorance; a curve that flattens means it learned.=~

---
# ⁉️
Every example in this note had something in common: pulling a lever didn't change anything about the *next* round. No state, no consequences — just "which one's best?". That's the simplest decision problem there is, and it has a name.

→ [[Multi-Armed Bandit]]
