---
aliases:
  - POMDPs
  - Partially Observable Markov Decision Process
  - Partially Observable MDP
  - Partial Observability
  - Belief MDP
  - Belief-State MDP
  - Value of Information
  - Tiger Problem
tags:
  - reinforcement-learning
  - decision-sciences
  - estimation
  - bayesian
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** An [[Markov Decision Process|MDP]] where you **can't see the state** — only noisy hints. The fix is to act on your **belief** about the state, which turns it back into an MDP (over beliefs).
> **Metaphor:** Two doors. A tiger behind one, treasure behind the other. You can't look — but you can listen.
> **Where it bites:** Robots with imperfect sensors, poker, medical diagnosis, dialogue — and why DQN stacks four frames and agents need memory.

---
Two doors. Behind one, a **tiger** (−100). Behind the other, **treasure** (+10). You don't know which is which.

You may **open** either door. Or you may **listen** (costs 1): you'll hear a growl from the correct side **85%** of the time, and from the wrong side 15%.

You listen. *Growl on the left.*

So the tiger is probably left — 85% probably. The "obvious" move is to open the right-hand door and take the treasure.

~={blue}Before you do: work out what that move is worth.=~

---
# Two doors 🐯

![[tiger_problem_two_doors.png]]

$$0.85 \times (+10) + 0.15 \times (-100) = 8.5 - 15 = \mathbf{-6.5}$$

You're 85% sure, and opening the door **loses money on average**. The downside is ten times the upside, so "probably right" isn't nearly good enough.

Listen again. Another growl on the left. Two independent hints in agreement:

$$\frac{0.85^2}{0.85^2 + 0.15^2} = \frac{0.7225}{0.745} = 0.970 \qquad\Rightarrow\qquad 0.97 \times 10 - 0.03 \times 100 = \mathbf{+6.7}$$

*Now* open it. The break-even belief is $\frac{100}{110} = 0.909$ — and one growl doesn't get you there.

![[tiger_belief_value.png]]
> [!TIP] Reading the chart
> **Left:** each consistent growl sharpens the belief — 0.50 → 0.85 → 0.97 → 0.995. **Right:** the payoff of opening is a straight line in your belief, and it only crosses zero at 0.909. Below that, the best action is one that earns *nothing* and costs 1: **go and get more information.**

> [!NOTE] POMDP
> A **P**artially **O**bservable MDP: the usual states, actions, transitions and rewards — *plus* an observation model $P(o \mid s)$. The agent never receives $s$, only $o$. ^pomdp-def

> [!SUCCESS] Core idea
> ~={pink}When you can't see the state, your **belief** becomes the state.=~ The belief "97% left" is fully known to you, it updates by a fixed rule ([[Bayes Filter|Bayes]]), and it's all that matters for deciding. So a POMDP *is* an MDP — over beliefs instead of world-states. That's the [[Beliefs#^belief-state-trick|belief-state trick]]. ^belief-is-the-state

---
# Two consequences people don't expect

**1 · Information has a price tag — and sometimes it's worth paying.** "Listen" never pays out directly. Its entire value is that it makes *later* decisions better. In a fully observed MDP there's no such thing as an information-gathering action; in a POMDP they're often the best move on the board. This is **value of information**, and it's [[Exploration vs Exploitation|exploration]] in its purest form.

**2 · "Act on your best guess" is wrong.**

> [!WARNING] Certainty equivalence fails here
> At 85% the *most likely* state says "tiger left → open right" — expected value −6.5. Treating your best estimate as if it were the truth throws away the one thing that mattered: **how sure you are**. It works for linear-Gaussian control ([[LQR#^separation-principle|the separation principle]]) and almost nowhere else. ~={red}Decide on the whole belief, not its peak.=~ ^act-on-belief-not-peak

---
# Why it's hard, and what people actually do

The belief-MDP has a **continuous** state (a probability vector) even if the world has two states. Exact solutions exist only for toy problems. In practice:

| Trick | The idea | Seen in |
|---|---|---|
| **Frame stacking** | feed the last $k$ observations — a cheap way to make the input Markov | [[DQN]]: 4 frames, because one frame has no **velocity** in it ([[Observability]]) |
| **Recurrent policy** | let an RNN/LSTM carry a hidden state — a *learned*, approximate belief | DRQN, most robotics policies |
| **Explicit filter + planner** | track the belief with a [[Kalman Filter\|Kalman]] or [[Particle Filter\|particle]] filter, then plan on it | classical robotics, [[LQR\|LQG]] |
| **Context as memory** | keep the whole history in the prompt | LLM agents → [[Memory]], [[Context Window]] |

> [!TIP] The design question to ask first
> *"Is my agent's input actually a **state** — or just an **observation**?"* If yesterday's information would change today's best action and it isn't in the input, you have a POMDP, and no amount of training on single observations will fix it. It's a [[Markov Property#^markov-is-about-state-design|state-design]] problem before it's a learning problem.

---
# The family

| | State visible? | Agents |
|---|---|---|
| [[Markov Decision Process\|MDP]] | ✅ | one |
| **POMDP** | ❌ | one |
| [[Dec-POMDP]] | ❌ | **many**, each with its own keyhole → [[Multi-Agent Reinforcement Learning]] |

And the filtering side of the house — [[State-Space Model]], [[Hidden Markov Model]], [[Kalman Filter]] — is exactly a POMDP **with the actions and rewards removed**. Estimation answers "where am I?"; a POMDP adds "…and what should I do about not being sure?"

---
> [!SUCCESS] If you remember one thing
> You don't act on the world, you act on **what you believe about the world**. ~={pink}And sometimes the most valuable action is the one whose only payoff is a better belief.=~

---
# ⁉️
That completes the grammar: states, actions, rewards, discounting, values, policies, hidden or not. And every method so far has needed someone to hand over the transition probabilities and the rewards in advance. Nobody ever does.

→ [[Reinforcement Learning]]
