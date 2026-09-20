---
aliases:
  - Proximal Policy Optimization
  - Proximal Policy Optimisation
  - PPO-Clip
  - Clipped Surrogate Objective
  - TRPO
  - Trust Region Policy Optimization
  - Trust Region Policy Optimisation
  - Trust Region
tags:
  - deep-rl
  - reinforcement-learning
  - decision-sciences
  - alignment
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A [[Policy Gradient|policy gradient]] that **refuses to move the policy too far in one update** — by clipping the new/old probability ratio to about ±20%, so there's no reward for overshooting.
> **Metaphor:** A short leash. Walk in the right direction, but only a step or two before you stop and look around again.
> **Where it bites:** The default deep-RL algorithm for a decade — robotics, game agents, OpenAI Five — and the "RL" inside RLHF.

---
In supervised learning, a bad gradient step is an inconvenience. The loss goes up, the next batch arrives — *the same distribution as before* — and you recover.

Now a robot learning to walk with a [[Policy Gradient]]. One noisy batch suggests a big change. You take it. The new policy falls flat on its face.

And here's the difference: **the policy generates its own data.** Every trajectory the robot now collects is of *a robot lying on the floor*. There's no information in there about how to walk. The next update is computed from garbage, and the one after that.

~={blue}One bad step, and the run never recovers. How do you take useful steps without ever taking that one?=~

---
# The leash 🦮

Don't let the new policy get far from the old one in a single update.

**TRPO** (2015) made that a hard constraint: *maximise improvement, subject to* $\text{KL}(\pi_\text{old} \,\|\, \pi_\text{new}) \leq \delta$ — a **trust region**. It works, and it needs second-order optimisation that's painful to implement. → [[Trust Region Policy Optimization (TRPO)]]

**PPO** (2017) gets nearly the same effect with a trick you can write in one line. Define the probability ratio for an action you took:

$$r(\theta) = \frac{\pi_\text{new}(a \mid s)}{\pi_\text{old}(a \mid s)} \qquad \text{(starts at 1)}$$

The plain objective is $r \cdot A$ — *make good actions (advantage $A > 0$) more likely, bad ones less*. PPO **clips** it:

$$L = \min\big(\; r A,\;\; \text{clip}(r,\, 1-\varepsilon,\, 1+\varepsilon)\, A \;\big) \qquad \varepsilon = 0.2$$

![[ppo_clip.png]]
> [!TIP] Reading the chart
> **Left — the action was good.** Pushing its probability up earns more objective… until the ratio hits **1.2**, where the line goes **flat**. Past +20% there is *nothing more to gain*, so the gradient is zero and the optimiser stops pushing. **Right — the action was bad.** Same in mirror image: flat below **0.8**. The `min` keeps one escape hatch open: if you've moved the *wrong* way, the full penalty still applies, so you can always undo a mistake.

In plain words: ~={blue}*"If it was good, make it more likely — but by no more than about 20% this round. If it was bad, less likely — by no more than about 20%."*=~

> [!NOTE] PPO
> Proximal Policy Optimization: an on-policy actor–critic that maximises a **clipped surrogate objective**, removing the incentive to change any action's probability by more than a factor of $1 \pm \varepsilon$ per update. → [[Proximal Policy Optimization Algorithms]] ^ppo-def

> [!SUCCESS] Core idea
> ~={pink}Cap the gain of the loop.=~ A policy that learns from its own data is a [[Feedback Loop|feedback loop]], and an over-large step is exactly the "strong reaction to noisy information" that makes loops [[Stability|unstable]]. PPO's clip is a gain limiter — and because it makes each batch safe to reuse, you get several gradient steps out of data you'd otherwise use once. ^cap-the-gain

---
# The recipe

1. Run the current policy; collect a batch of trajectories.
2. Compute advantages with the critic — **GAE**, $\gamma = 0.99$, $\lambda = 0.95$ → [[Actor-Critic]].
3. Do **several epochs** of minibatch SGD on the clipped objective over *that same batch*. (This is the part plain policy gradient can't do — after one step the data is off-policy. The ratio $r$ **is** the [[On-Policy vs Off-Policy#The correction — importance sampling|importance-sampling correction]], and the clip keeps it near 1.)
4. Throw the batch away. Go to 1.

Plus, in the loss: a value-function term (to train the critic) and a small **entropy bonus** (to stop the policy collapsing to one action too early → [[Exploration vs Exploitation]]).

| | [[Policy Gradient\|REINFORCE]] | TRPO | **PPO** |
|---|---|---|---|
| Step-size control | none | hard KL constraint | **clipped ratio** |
| Optimiser | first-order | second-order (conjugate gradient) | **first-order — plain Adam** |
| Reuses each batch | ❌ | ❌ | ✅ a few epochs |
| Implementation | trivial | painful | moderate |
| Robustness | poor | good | good |

---
# PPO inside RLHF 🤖

The LLM is the **actor**. A separate **value model** is the critic. The **reward model** supplies the score at the end of each response. And there are **two leashes**, which are easy to confuse:

| Leash | Tethers the policy to | Why |
|---|---|---|
| **The PPO clip** | *itself, one update ago* | stable optimisation |
| **The KL penalty** in the reward | the frozen **reference model** (the SFT model) | stops it drifting somewhere the reward model can't be trusted → [[RLHF#^kl-leash\|the KL leash]], [[Reward Hacking]], [[KL Divergence]] |

That's **four** large models in memory — policy, reference, reward, value — which is the cost that [[DPO]] and [[GRPO]] each set out to cut. → [[Training language models to follow instructions with human feedback]]

> [!WARNING] The clip is not a guarantee
> It removes the *incentive* to move further — it doesn't *prevent* it. Several epochs of Adam on the same batch can carry ratios well outside [0.8, 1.2]; implementations monitor the actual KL and stop early. And PPO is notorious for **implementation details that matter more than the algorithm**: advantage normalisation, value clipping, learning-rate annealing, orthogonal init, observation normalisation. ~={red}Two "PPO" codebases can differ by a factor of two in final score.=~ Start from a reference implementation. ^clip-is-not-a-guarantee

---
> [!SUCCESS] If you remember one thing
> **Improve the policy, but never by much at once.** ~={pink}When a learner collects its own data, the size of the step is a stability question, not just a speed question.=~

---
# ⁉️
PPO is on-policy: every batch is used for a few epochs and then discarded. For a real robot, where each second of experience is expensive, that's hard to stomach. Is there an off-policy method that keeps a replay buffer *and* handles continuous actions?

→ [[SAC]]
