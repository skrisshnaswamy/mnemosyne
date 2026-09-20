---
aliases:
  - SAC (algorithm)
  - Soft Actor Critic
  - Maximum Entropy RL
  - Maximum Entropy Reinforcement Learning
  - Max-Ent RL
  - MaxEnt RL
  - DDPG
  - Deep Deterministic Policy Gradient
  - TD3
  - Twin Delayed DDPG
  - Continuous Control
  - Entropy Regularization
  - Entropy Regularisation
  - Entropy Bonus
tags:
  - deep-rl
  - reinforcement-learning
  - decision-sciences
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** The off-policy actor–critic for **continuous actions**. Its twist: the agent is rewarded for **staying as random as it can while still succeeding** — maximum-entropy RL.
> **Metaphor:** A hiker who, when three paths look about equally good, doesn't pick a favourite — and so never forgets that the other two exist.
> **Where it bites:** Robotics and continuous control, where samples are expensive. The family tree is DDPG → TD3 → SAC.

---
A robot arm with seven joints. An action is **seven real numbers** — a torque for each joint.

[[DQN]] chooses actions by computing $Q$ for *every* action and taking the biggest. Here there are infinitely many. You can't list them. You can't even grid them — ten levels per joint is $10^7$ actions, and that's a coarse grid ([[Curse of Dimensionality]]).

[[PPO]] handles continuous actions fine — but it's on-policy, and throws every batch away. On a physical robot, each second of data is wear, time and someone standing by with an emergency stop.

~={blue}How do you get DQN's data efficiency — replay buffer and all — when you can't take a $\max$ over the actions?=~

---
# Step one — learn the argmax (DDPG)

If you can't *search* for the best action, **train a network to output it**.

- A **critic** $Q(s, a)$ — takes a state *and* an action, scores the pair.
- An **actor** $\mu(s)$ — outputs one action. Train it by asking the critic: *"which way should I nudge my output to raise your score?"* — i.e. follow $\nabla_a Q$.

That's **DDPG** — "DQN for continuous actions": replay buffer, target networks, the lot. → [[Continuous control with deep reinforcement learning (DDPG)]]. And it's famously brittle, for two reasons. The actor is *hunting for the critic's highest point* — so it finds every spot where the critic is **wrong in the optimistic direction** ([[Q-Learning#^maximisation-bias|maximisation bias]], weaponised). And a deterministic actor explores only by bolted-on noise.

**TD3** patches the first: **two** critics, use the **minimum** of the pair (pessimism), update the actor less often, smooth the target.

---
# Step two — keep your options open (SAC) 🥾

SAC keeps the twin critics and changes something more fundamental: **the objective itself.**

$$J = \mathbb{E}\Big[\sum_t \; r_t \;+\; \alpha \cdot \underbrace{\mathcal{H}\big(\pi(\cdot \mid s_t)\big)}_{\text{entropy: how random the policy still is}}\Big]$$

The agent is paid for reward **and** for being unpredictable. The optimal policy is no longer "always the best action" — it's *"each action in proportion to how good it is"*:

$$\pi(a \mid s) \;\propto\; \exp\!\big(Q(s,a) / \alpha\big)$$

That's a softmax over $Q$ with **temperature** $\alpha$.

![[sac_temperature.png]]
> [!TIP] Reading the chart
> Five actions; $a_1$ and $a_2$ are almost equally good (1.0 vs 0.9). At $\alpha = 0.05$ the policy is nearly greedy — 88% / 12%. At $\alpha = 0.3$ it's **56% / 40%**: it keeps *both* good options alive and ignores the bad ones. At $\alpha = 5$ it's almost uniform, and useless. A hard $\max$ would commit 100% to $a_1$ on the strength of a 0.1 difference that may well be noise.

> [!NOTE] Soft Actor-Critic
> An off-policy actor–critic that maximises expected return **plus policy entropy**, with a stochastic (Gaussian) actor, twin $Q$-critics (taking the minimum), target networks, a replay buffer, and an automatically-tuned temperature $\alpha$. → [[Soft Actor-Critic]] (the paper) ^sac-def

> [!SUCCESS] Core idea
> ~={pink}Don't commit harder than the evidence justifies.=~ The entropy bonus buys **exploration for free** (the policy stays random wherever it hasn't found a reason not to be), **robustness** (it knows more than one way to do the task, so a perturbation doesn't strand it), and a **smoother optimisation problem**. And $\alpha$ is tuned automatically — "keep at least this much entropy" — which removes the most painful knob. ^dont-overcommit

---
# The family

| | Actions | On/off-policy | Policy | Sample efficiency | Stability |
|---|---|---|---|---|---|
| [[DQN]] | discrete | off | greedy on $Q$ | good | OK |
| [[PPO]] | either | **on** | stochastic | poor | **good** |
| **DDPG** | continuous | off | deterministic + noise | good | poor |
| **TD3** | continuous | off | deterministic + noise | good | better |
| **SAC** | continuous | off | **stochastic, max-entropy** | **good** | **good** |

Rule of thumb: **simulation is cheap → [[PPO]]** (robust, parallelises, forgiving). **Real-world samples are expensive → SAC** (every transition is reused many times). The order-of-magnitude gap in sample efficiency is the whole reason.

> [!TIP] It's the same temperature 🌡️
> $\pi \propto \exp(Q/\alpha)$ is a softmax with a temperature — *exactly* the knob in [[Sampling Parameters]]. Low → commit to the top choice. High → spread out. And it's the same Boltzmann form that turns up in [[Preference Learning|Bradley–Terry]] and in [[Inverse Reinforcement Learning|max-entropy IRL]]. Whenever you see $\exp(\text{score}/T)$, someone is refusing to commit harder than the scores justify. Also see [[PPO]]'s entropy bonus — the same idea in a milder dose.

> [!WARNING] "The entropy bonus is just an exploration trick"
> It changes **what's being optimised**. A max-entropy agent's *final* policy is deliberately stochastic — it isn't noise that gets annealed away. If you need a deterministic controller at deployment you take the mean of the Gaussian, and you should check that it still performs. It also means reward **scale** matters: double all the rewards and you've halved the effective temperature. ^entropy-changes-the-objective

Still an active area — the vault has [[WarpSAC- Towards the Pinnacle of Scalable Off-policy RL by Rethinking Exploration and Exploitation]].

---
> [!SUCCESS] If you remember one thing
> Can't take a max over continuous actions → **learn an actor to do it**. Actor exploits the critic's errors → **two critics, take the min**. ~={pink}And pay the policy to stay random, so it never commits harder than it has evidence for.=~

---
# ⁉️
Everything in this section has been a **reflex**: a network is trained, and at run time it reacts instantly. There's a completely different way to be good at something — *stop, and think about this particular position.*

→ [[Monte Carlo Tree Search]]
