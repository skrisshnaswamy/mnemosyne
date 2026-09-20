---
aliases:
  - Policies
  - Deterministic Policy
  - Stochastic Policy
  - Optimal Policy
  - Greedy Policy
  - Policy Network
  - Control Law
  - Decision Rule
tags:
  - reinforcement-learning
  - decision-sciences
  - fundamentals
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A **rule that maps every situation to an action** — not a fixed list of moves. It's the thing RL is actually solving for.
> **Metaphor:** A satnav versus a printed list of directions. Miss one turn and the list is scrap paper; the satnav just says *"in 200 metres, turn left"* from wherever you now are.
> **Where it bites:** Everything in RL outputs one. And an LLM *is* one: context in, distribution over next tokens out.

---
Two ways to get to a wedding in a town you've never visited.

**Option A** — your friend texts you directions: *"Left out of the station. Second right. Straight over the roundabout. Third left."*

**Option B** — a satnav.

You walk out of the station and the road on the left is closed for roadworks.

Option A is now worthless. Not slightly worse — *worthless*. Every remaining instruction assumed you'd taken that first left. Option B doesn't even notice there was a problem; it looks at where you are and tells you what to do from there.

~={blue}What does the satnav have that the list doesn't?=~

---
# The satnav 🧭

![[plan_vs_policy_satnav.png]]

An answer for **every place you might end up** — not just the places you were *supposed* to be.

The list is a **plan**: a sequence of actions, fixed in advance. It's [[Feedback Loop#^open-closed-def|open-loop]], and it works only if the world behaves exactly as predicted. The satnav is a **policy**: a rule that takes *the current situation* and returns an action. It's closed-loop by construction.

And in any world with randomness — where [[Markov Decision Process#^actions-choose-the-distribution|your action only chooses the odds]], not the outcome — you *will* get knocked off the expected path. So a plan was never going to be enough.

> [!NOTE] Policy
> A mapping from states to actions, written $\pi$. **Deterministic:** $a = \pi(s)$ — one action per state. **Stochastic:** $\pi(a \mid s)$ — a probability distribution over actions in each state. ^policy-def

For the shop from [[Markov Decision Process]], a whole policy fits in three lines:

| Stock | Action |
|---|---|
| `Low` | order more |
| `Medium` | do nothing |
| `High` | discount |

> [!SUCCESS] Core idea
> ~={pink}A plan answers "what do I do next?". A policy answers "what do I do *from anywhere*?"=~ Value functions, Q-tables, reward models — they're all scaffolding. The policy is the deliverable. And note its *type*: [[Markov Decision Process#^policy-is-a-mapping|it returns an action, not a score]]. ^policy-core

---
# Why would you ever want a *random* policy?

Deterministic sounds strictly better — why roll dice when you know the best move? Four reasons, and they come up constantly:

| Reason | Example |
|---|---|
| **Exploration** | if you always do what currently looks best, you never find out you were wrong → [[Exploration vs Exploitation]] |
| **Adversaries** | rock–paper–scissors: *any* deterministic policy loses to someone who's noticed it. The only safe policy is ⅓, ⅓, ⅓ |
| **Aliased states** | two corridors look identical to your sensors but need opposite actions. A deterministic rule gets one of them wrong *every* time; a coin-flip gets both right half the time → [[POMDP]] |
| **Smoothness** | nudge a probability and the outcome changes a little; flip an argmax and it jumps. Gradients need the former → [[Policy Gradient]] |

---
# How you get one

| Route | How | Examples |
|---|---|---|
| **Through values** | learn $Q(s,a)$, then act **greedily**: $\pi(s) = \arg\max_a Q(s,a)$. [[Markov Decision Process#max vs argmax 🏙️\|Once you have Q, the policy is free]] | [[Q-Learning]], [[DQN]] |
| **Directly** | parameterise $\pi_\theta$ and push $\theta$ toward actions that turned out well | [[Policy Gradient]], [[PPO]] |
| **Both** | a policy *and* a value estimate, each helping the other | [[Actor-Critic]], [[SAC]] |
| **By planning** | don't store a policy at all — compute the action fresh each step | [[Model Predictive Control]], [[Monte Carlo Tree Search]] |
| **By copying** | supervised learning on an expert's state → action pairs | [[Imitation Learning]] |

And how it's *stored* climbs the same ladder as everything else: a **table** (tiny problems) → a **linear rule** ($u = -Kx$ — that's exactly what [[LQR]] produces) → a **neural network** that outputs a softmax over discrete actions, or a mean and spread for continuous ones.

> [!TIP] A language model is a policy 🤖
> State = the context so far. Action = the next token. $\pi(a \mid s)$ = the softmax over the vocabulary. That isn't an analogy — it's why [[RLHF]], [[PPO]] and [[GRPO]] can be applied to LLMs *at all*. [[Sampling Parameters|Temperature]] is just how stochastic you let the policy be at run-time, and [[Markov Property#^llm-is-markov|the context window is what makes the state Markov]].

---
# Two pairs of words you'll hear

- **Behaviour policy vs target policy.** The one that *generated the data* vs the one you're *trying to learn about*. When they differ, you're learning off-policy → [[On-Policy vs Off-Policy]].
- **Optimal policy $\pi^*$.** The one with the highest value from every state. In any finite MDP one always exists, and there's always a *deterministic* one among the best. (The randomness above is needed for learning, for opponents and for partial observability — not for optimality in a fully observed MDP.)

> [!WARNING] Policy ≠ plan
> When someone says "the agent learned a plan", check what they mean. A **plan** is a list of actions and breaks at the first surprise. A **policy** is a function and doesn't. Most of what makes RL hard — and useful — is that it learns the second thing. ^policy-is-not-a-plan

---
> [!SUCCESS] If you remember one thing
> A policy is a **function from situations to actions**. ~={pink}You don't get to choose which situations you end up in, so you'd better have an answer for all of them.=~

---
# ⁉️
A policy is judged by what it earns. Which pushes the real question back one step: **who decides what's worth earning** — and how do you say it in a way a machine can't misread?

→ [[Reward Function]]
