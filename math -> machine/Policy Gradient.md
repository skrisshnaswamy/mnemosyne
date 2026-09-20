---
aliases:
  - Policy Gradients
  - Policy Gradient Methods
  - REINFORCE
  - Policy Gradient Theorem
  - Score Function Estimator
  - Log-Derivative Trick
  - Likelihood Ratio Gradient
  - Policy Optimization
  - Policy Optimisation
tags:
  - reinforcement-learning
  - decision-sciences
  - training
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Skip the value table. **Make the actions that led to good outcomes more likely, and the ones that led to bad outcomes less likely** — in proportion to *how* good or bad.
> **Metaphor:** Practising a tennis serve. Nobody tells you the correct motion; you keep more of whatever you did on the serves that went in.
> **Where it bites:** Continuous actions, stochastic policies, and **non-differentiable rewards** — which is exactly why it's the engine under RLHF, PPO and GRPO.

---
You're learning to serve at tennis. There's no coach — just you, a basket of balls, and a line on the far side of the net.

Every serve is a bit different: toss slightly higher, wrist a little later, more or less knee bend. Some go in. Most don't.

You can't compute *"the derivative of where the ball landed with respect to my elbow"*. The flight of the ball is a black box to you.

~={blue}And yet you get better. What's the rule you're actually following?=~

---
# The serve 🎾

Something like: *"That one went in. Whatever I just did — do a bit more of that."* And: *"That one hit the back fence. Less of that."* And, crucially, scaled: an ace reinforces the motion more than a serve that only just crept in.

Write your policy as $\pi_\theta(a \mid s)$ — a network with weights $\theta$ producing a distribution over actions. The rule becomes:

$$\theta \;\leftarrow\; \theta + \alpha \cdot \underbrace{\nabla_\theta \log \pi_\theta(a \mid s)}_{\text{"which way makes what I did more likely"}} \cdot \underbrace{G}_{\text{how well it turned out}}$$

That's **REINFORCE** (Williams, 1992 — [[Simple Statistical Gradient-Following Algorithms (REINFORCE)|the paper's in the vault]]). Sample an episode, and for every action taken, push its log-probability **up** if the return was good and **down** if it was bad.

> [!NOTE] Policy gradient
> A family of methods that parameterise the policy directly and ascend the gradient of expected return: $\nabla J(\theta) = \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a \mid s) \cdot (\text{how good that turned out})\big]$. The expectation is estimated by [[Monte Carlo Methods|sampling]] trajectories from the current policy — so it's [[On-Policy vs Off-Policy|on-policy]]. ^policy-gradient-def

> [!SUCCESS] Core idea
> ~={pink}The gradient never passes through the environment or the reward.=~ You only ever differentiate $\log \pi$ — your *own* network. The reward is just a number you multiply by. So the reward can be anything: a game score, a human's thumbs-up, whether the unit tests passed. That single property is why policy gradients, and not backprop-through-the-task, are how language models are trained on human preferences. ^reward-is-a-black-box

---
# The problem — it's incredibly noisy

Try it on three slot machines that pay out about **10, 10.5 and 11**. Every pull returns roughly +10. So *every* action gets pushed **up** — the good arm a little more than the others. The signal you care about (a difference of 0.5) is buried under a constant +10 that carries no information whatsoever.

Now subtract the **average reward so far** before multiplying. A pull that returns 11 when you usually get 10.4 becomes **+0.6**; a pull that returns 10 becomes **−0.4**. Same data. Same gradient direction on average.

![[reinforce_baseline.png]]
> [!TIP] Reading the chart
> After 3,000 pulls, the version with **no baseline** picks the best arm **53%** of the time. With a baseline: **99%**. Nothing else changed.

> [!NOTE] Baseline
> Any quantity $b(s)$ that doesn't depend on the action can be subtracted from the return without changing the *expected* gradient — it only changes the variance. The natural choice is $b(s) = V(s)$, which turns the multiplier into the **[[Value Function|advantage]]**: *"was this better than usual from here?"* ^baseline

> [!WARNING] "Subtracting a baseline biases the gradient"
> It doesn't — that's the beautiful part. In expectation the baseline term is exactly zero ($\mathbb{E}[\nabla \log \pi \cdot b] = b \cdot \nabla \sum_a \pi(a|s) = b \cdot \nabla 1 = 0$). It removes noise for free. ~={red}A policy gradient without a baseline is almost never the right call.=~ ^baseline-is-unbiased

---
# When to reach for it

| | Value-based ([[Q-Learning]], [[DQN]]) | Policy gradient |
|---|---|---|
| Continuous actions | ❌ can't $\arg\max$ over a continuum | ✅ output a mean and a spread |
| Stochastic policies | awkward | ✅ native — see [[Policy#Why would you ever want a *random* policy?\|why you'd want one]] |
| Optimises | a proxy (Bellman error) | **the thing you care about**, directly |
| Sample efficiency | better (off-policy, replay) | worse (on-policy — data is used once) |
| Variance | lower | **high** |
| Behaviour of updates | can change the greedy action abruptly | smooth — small change in $\theta$, small change in behaviour |

---
# The thread to the rest of the vault 🧵

- Add a learned $V(s)$ as the baseline, updated by TD → **[[Actor-Critic]]**.
- Stop the policy lurching too far on one noisy batch → **[[PPO]]**.
- Use the *mean reward of a group of samples for the same prompt* as the baseline, and skip the critic entirely → **[[GRPO]]**.
- Set the "reward" to 1 on expert demonstrations and you get… ordinary maximum-likelihood training. **Supervised fine-tuning is a policy gradient where every example is declared equally good** → [[Imitation Learning]], [[Cross Entropy]].

> [!TIP] The trick has a name
> $\nabla_\theta\, \mathbb{E}_{x \sim p_\theta}[f(x)] = \mathbb{E}\big[f(x)\, \nabla_\theta \log p_\theta(x)\big]$ is the **log-derivative trick** (or *score-function estimator*). It turns "differentiate an expectation over a distribution I'm changing" into "average something I can sample". It appears well beyond RL — anywhere you need a gradient through a sampling step.

---
> [!SUCCESS] If you remember one thing
> **Do more of what worked, in proportion to how much better than usual it was.** ~={pink}You differentiate your own policy, never the world — which is why the reward is allowed to be a black box.=~

---
# ⁉️
That baseline — "how good is it *usually* from here?" — is a value function. If we're going to need one anyway, we may as well learn it properly and let it do more than reduce noise.

→ [[Actor-Critic]]
