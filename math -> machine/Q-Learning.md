---
aliases:
  - Q Learning
  - Q-learning
  - Q-Table
  - Tabular Q-Learning
  - Watkins Q-Learning
  - Maximisation Bias
  - Maximization Bias
  - Double Q-Learning
tags:
  - reinforcement-learning
  - decision-sciences
  - fundamentals
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Keep a table of *"how good is doing **this**, from **here**?"* and nudge each entry toward `reward + γ × (best entry in the next state)`. No model needed — and it learns the **optimal** policy even while behaving sub-optimally.
> **Metaphor:** The shop manager's notebook: one line per (situation, decision), scribbled over a little after every day's trading.
> **Where it bites:** The root of DQN and most value-based deep RL. Also: the $\max$ makes it off-policy, *and* makes it over-optimistic.

---
The shop manager from [[Decision Sciences#Two tables 📋|the chapter]] keeps a notebook. One row per stock level, one column per decision:

| | do nothing | order | discount |
|---|---|---|---|
| **Low** | 12 | **40** | 8 |
| **Medium** | 45 | 30 | **50** |
| **High** | 55 | 20 | **62** |

Today stock is `Low`. They **order**. It costs 4, and tomorrow they find themselves in `Medium`.

~={blue}Using only what's on this page, how should the "Low / order" entry — currently 40 — change?=~

---
# The notebook 📒

What did ordering *turn out* to be worth? You paid 4, and you landed somewhere whose best option is worth 50:

$$\text{target} = -4 + 0.9 \times \underbrace{\max(45, 30, 50)}_{50} = 41$$

The notebook said 40. Reality suggests 41. Move a little of the way ($\alpha = 0.1$):

$$Q(\text{Low}, \text{order}) \leftarrow 40 + 0.1 \times (41 - 40) = \mathbf{40.1}$$

That's the entire algorithm.

$$Q(s,a) \;\leftarrow\; Q(s,a) + \alpha\,\big[\, r + \gamma \max_{a'} Q(s',a') - Q(s,a) \,\big]$$

It's [[Temporal Difference Learning|TD learning]] applied to **action** values, with a $\max$ in the target.

> [!NOTE] Q-learning
> A model-free, off-policy TD control algorithm (Watkins, 1989). It learns the optimal action-value function $Q^*$ directly by bootstrapping from the **greedy** value of the next state, regardless of which action the agent actually takes next. ^q-learning-def

**Why $Q$ and not $V$?** With only $V$, choosing an action means asking "where would each action take me?" — which needs the model. With $Q$ you just take the [[Markov Decision Process#max vs argmax 🏙️|argmax]] of a row. **$Q$ is the value function you can act on without knowing how the world works.**

> [!SUCCESS] Core idea
> ~={pink}The target uses the *best* next action, not the one you actually take.=~ So the notebook fills in with the values of **perfect play** while the manager is still experimenting, making mistakes, trying the discount on a whim. Behaviour and learning are decoupled — that's what **off-policy** means, and it's why Q-learning can learn from old data, other people's data, or a replay buffer. → [[On-Policy vs Off-Policy]] ^learns-optimal-while-exploring

It provably converges to $Q^*$ — given that every (state, action) pair keeps being tried and the step size decays. The first condition is the expensive one → [[Exploration vs Exploitation]].

---
# The $\max$ has a second, nastier effect 🪤

Suppose a state has 10 actions and the **true value of every one is 0**. Your estimates are noisy — each is the truth plus some random error. What does $\max_a Q$ come out as?

Not zero. The max picks whichever estimate happened to be *most over-optimistic*.

![[maximisation_bias.png]]
> [!TIP] Reading the chart
> One honest noisy estimate averages **0.00**. The max over ten of them averages **+1.54** — *every true value is zero*. And that inflated number becomes the *target* for the previous state, which inflates it too… the optimism propagates backwards through the whole table.

> [!WARNING] Maximisation bias
> $\mathbb{E}[\max_a \hat{Q}] \geq \max_a \mathbb{E}[\hat{Q}]$. Using the same noisy numbers both to **choose** the best action and to **score** it is systematically over-optimistic. ~={red}The more actions and the noisier the estimates, the worse it gets.=~
>
> **The fix — Double Q-learning:** keep two independent tables. Let one *pick* the action and the other *score* it. The green curve: mean **0.00**. This idea carries straight into deep RL as **Double DQN** → [[DQN]]. ^maximisation-bias

(It's the same trap as picking your best model on the validation set and then reporting that validation score — or [[Evals|benchmark-chasing]]. *Selecting* on a noisy number and *reporting* that number are jobs for two different datasets.)

---
# Where it runs out

> [!WARNING] A table only knows where it has been
> One cell per (state, action). Chess has ~$10^{47}$ positions; after ten moves you're somewhere no one has ever been, staring at an empty row. The table can't generalise from a position to a *similar* one — that's the [[Curse of Dimensionality]]. Replacing the table with a network is [[Function Approximation]]; doing it *stably* is [[DQN]]. ^tables-dont-generalise

And it only handles **discrete** actions — $\max_a$ over a continuous steering angle isn't something you can enumerate → [[SAC]].

| | Q-learning |
|---|---|
| Needs a model? | no |
| Learns | $Q^*$ — optimal action values |
| On / off-policy | **off** |
| Works with | small, discrete state & action spaces |
| Famous weakness | over-estimation; no generalisation |

---
> [!SUCCESS] If you remember one thing
> **$Q(s,a) \leftarrow$ reward + $\gamma \max Q(\text{next})$.** ~={pink}The max is what lets it learn the best policy while following a worse one — and the same max is what makes it over-optimistic.=~

---
# ⁉️
"It learns about the greedy policy, whatever it's actually doing." That sounds like a pure win. It isn't always — there's a famous little gridworld where it makes Q-learning walk along the edge of a cliff.

→ [[On-Policy vs Off-Policy]]
