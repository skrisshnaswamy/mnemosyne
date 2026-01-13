---
title: "Continuous control with deep reinforcement learning (DDPG)"
authors: ["Timothy P. Lillicrap", "Jonathan J. Hunt", "Alexander Pritzel", "Nicolas Heess", "Tom Erez", "Yuval Tassa", "David Silver", "Daan Wierstra"]
year: 2015
arxiv: "1509.02971"
url: https://arxiv.org/abs/1509.02971
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, rl]
---
## The Core Idea

[[Playing Atari with Deep Reinforcement Learning (DQN)|DQN]] works by asking, at every step, "which action has the highest $Q$ value?" That question is cheap when you have 18 joystick buttons. It is impossible when the action is a vector of 7 real-valued joint torques — you would have to solve an optimisation problem inside every single timestep.

The fix here is to stop searching for the best action and instead **train a second network to output it**. Two networks:

- a **critic** $Q(s,a\mid\theta^Q)$, which scores a state–action pair, trained exactly like DQN;
- an **actor** $\mu(s\mid\theta^\mu)$, which spits out one specific real-valued action vector, trained to make the critic's score go up.

The clever part is how the actor learns. The critic is a differentiable function of the action. So you can ask [[Pytorch Autograd|autograd]] for $\nabla_a Q(s,a)$ — "which way should I nudge this torque to get more reward?" — and then push that signal back through the actor with the [[Backpropagation|chain rule]]. No sampling, no likelihood ratio, no high-variance [[Simple Statistical Gradient-Following Algorithms (REINFORCE)|REINFORCE]] estimator. Just a gradient flowing from the critic straight into the actor's weights.

> [!NOTE] Deterministic policy gradient
> For a policy that outputs one action rather than a distribution, the gradient of expected return is $\nabla_{\theta^\mu} J = \mathbb{E}_{s\sim\rho^\beta}\big[\nabla_a Q(s,a)|_{a=\mu(s)}\,\nabla_{\theta^\mu}\mu(s)\big]$. Because the outer expectation is only over *states*, not actions, it can be estimated from off-policy data. ^deterministic-policy-gradient

That gradient was already proved by Silver et al. (2014) — DPG. But plugging in deep networks made it diverge. The contribution of this paper is the set of stabilisers borrowed from DQN that make it actually train: a big [[Playing Atari with Deep Reinforcement Learning (DQN)#^experience-replay|replay buffer]], **soft** target networks, [[Batch Normalization|batch normalisation]] on the inputs, and temporally correlated exploration noise. With one fixed set of hyperparameters, the same code solves 20+ MuJoCo physics tasks, and often from raw 64×64 pixels.

## The Methodology

**The critic loss.** Standard [[Markov Decision Process|MDP]] temporal-difference regression. Sample a minibatch of $N$ transitions $(s_i,a_i,r_i,s_{i+1})$ from the buffer, form the bootstrap target with the *target* copies of both networks:

$$y_i = r_i + \gamma\, Q'\big(s_{i+1},\, \mu'(s_{i+1}\mid\theta^{\mu'}) \mid \theta^{Q'}\big)$$

$$L(\theta^Q) = \frac{1}{N}\sum_i \big(y_i - Q(s_i,a_i\mid\theta^Q)\big)^2$$

Note there is no $\max$ anywhere. The target policy network $\mu'$ *is* the argmax, learned rather than searched.

**The actor loss.** Not really a loss — a gradient ascent step:

$$\nabla_{\theta^\mu} J \approx \frac{1}{N}\sum_i \nabla_a Q(s,a\mid\theta^Q)\big|_{s=s_i,\,a=\mu(s_i)} \; \nabla_{\theta^\mu}\mu(s\mid\theta^\mu)\big|_{s_i}$$

In code this is `loss = -Q(s, mu(s)).mean()` with the critic's weights frozen.

**Soft target updates.** DQN copies weights wholesale every 10k steps. Here the targets creep instead:

$$\theta' \leftarrow \tau\theta + (1-\tau)\theta', \qquad \tau = 0.001$$

Both actor and critic have a target twin. The paper is explicit that you need *both* — a target critic alone was not enough to stop divergence. This slows value propagation but turns the moving-target problem into something much closer to plain supervised regression.

**Exploration.** Because $\mu$ is deterministic, exploration is bolted on from outside: $\mu'(s_t) = \mu(s_t) + \mathcal{N}_t$. They use an **Ornstein–Uhlenbeck** process ($\theta=0.15$, $\sigma=0.2$) rather than white noise. OU noise is a mean-reverting random walk — it wanders in one direction for a while. In systems with inertia, independent per-step noise just cancels out and the arm jitters in place; correlated noise actually pushes the limb somewhere new.

**Batch norm.** Joint angles are in radians, velocities in rad/s, positions in metres. Different scales per dimension per environment. [[Batch Normalization|BatchNorm]] on the state input and on all actor layers (and all critic layers *before* the action is injected) removes the need to hand-tune feature scaling per task — which is what makes "one hyperparameter set for 20 tasks" possible.

**Architecture and hyperparameters.**

| | value |
|---|---|
| Optimiser | Adam, lr $10^{-4}$ actor, $10^{-3}$ critic |
| Critic weight decay | $L_2$, $10^{-2}$ |
| $\gamma$ | 0.99 |
| $\tau$ | 0.001 |
| Replay buffer | $10^6$ transitions |
| Minibatch | 64 (low-dim), 16 (pixels) |
| Low-dim net | 2 hidden layers, 400 → 300 units, ReLU (~130k params) |
| Pixel net | 3 conv layers, 32 filters each, no pooling, then 2×200 FC (~430k params) |
| Actor output | $\tanh$, to bound actions |
| Final layer init | uniform $[-3\times10^{-3}, 3\times10^{-3}]$ (low-d) so initial $Q$ and $\mu$ sit near zero |
| Other layers | uniform $[-1/\sqrt{f}, 1/\sqrt{f}]$, $f$ = fan-in |

One design detail worth keeping: **the action is not fed into the critic at layer 1.** It enters at the second hidden layer, after the state has already been processed. In the pixel case the action joins only at the fully connected layers.

**Pixels.** Action repeat of 3 (simulate 3 steps per agent step), stack the 3 RGB renderings → 9 feature maps at 64×64, scaled to $[0,1]$. The stacking is what lets the network infer velocity from position frames.

## Ablation Studies and Experiments

Benchmarks: ~30 MuJoCo tasks (cartpole swing-up, multi-link cartpoles, reacher, gripper, cheetah, hopper, walker2d, a hyq quadruped) plus Torcs driving. Every task run in both low-dim and pixel form, 5 seeds, ≤2.5M environment steps.

**Normalisation of scores.** 0 = uniform-random action policy. 1 = **iLQG**, a model-predictive trajectory optimiser with full access to the true simulator dynamics and their derivatives. So a score above 1 means the model-free learner beat a planner that was allowed to cheat.

Selected numbers (average over 5 seeds / best seed):

| task | low-dim | pixels | plain DPG baseline |
|---|---|---|---|
| cartpole | 0.844 / 1.115 | 0.482 / 1.138 | 0.244 / 0.755 |
| cartpoleBalance | 0.951 / 1.000 | 0.335 / 0.996 | **−0.468** / 0.528 |
| cheetah | 0.903 / 1.206 | 0.457 / 0.792 | −0.008 / 0.425 |
| hardCheetah | 1.311 / 1.990 | 1.204 / 1.431 | −0.031 / 1.411 |
| walker2d | 0.705 / 1.573 | 0.944 / 1.476 | 0.393 / 1.397 |
| hopper | 0.676 / 0.936 | 0.112 / 0.924 | 0.078 / 0.917 |
| pendulum | 0.946 / 1.021 | 0.663 / 1.055 | 0.099 / 0.951 |

The `cntrl` column is the original DPG with a replay buffer and batch norm but **no target networks**. Its averages hover around zero — often worse than random. Its *best* seeds are sometimes fine (hardCheetah 1.411). That gap between average and best is the whole story: without target networks the algorithm is a lottery.

**The component ablation (Figure 2).** Four variants compared: DPG + batchnorm only; DPG + target networks only; both; both from pixels. Conclusion stated flatly by the authors — *target networks are crucial*. Batch norm alone does not save you. Both are needed to do well across *all* tasks; batch norm's job is generality across environments with different unit scales, not raw performance on any single one.

**Value calibration (Figure 3).** They plot predicted $Q$ against actual observed return on test episodes. On pendulum and cartpole the estimates sit on the unity line with no systematic bias — notable given [[A Distributional Perspective on Reinforcement Learning|Q-learning's]] known tendency to over-estimate. On harder tasks the estimates degrade badly, **yet the policies are still good**. Useful lesson: a miscalibrated critic can still give a correct *gradient direction* in $a$.

**What did not work / surprises.**

- Naive deep DPG diverges. That is the negative result the whole paper is built around.
- Target critic alone was insufficient; a target actor was also required for stable $y_i$.
- **Torcs is a failure case.** Mean scores are large and negative (−393 low-dim, −402 pixels) while best seeds reach ~1840–1876. Some replicas learn to drive a lap, others never learn anything. They also had to change the noise process for Torcs — the one place the "same hyperparameters everywhere" claim breaks.
- Discretising the action space, the obvious alternative, is dismissed on combinatorial grounds: a 7-DoF arm with the crudest 3-way split per joint gives $3^7 = 2187$ actions, and finer control makes it worse.
- Learning from pixels was sometimes *as fast* as from joint angles (e.g. walker2d pixels 0.944 > low-dim 0.705 on average). The authors guess this is action repeat simplifying the problem, plus conv layers producing an easily separable state representation.

## Worth Remembering

- **Sample efficiency vs DQN.** Most tasks solved in under 2.5M steps — roughly 20× fewer than DQN needs on Atari. Being off-policy with a replay buffer plus getting a real gradient through the critic buys a lot over one-hot value iteration.
- **Exploration is a separate knob.** Because DDPG is off-policy, you can change the noise process without touching the learning rule. That is a genuine engineering convenience and also where most practitioner pain lives — OU noise has since largely been replaced by plain Gaussian noise in follow-ups like TD3 and [[Soft Actor-Critic|SAC]].
- **The honest limitations.** No convergence guarantee once you use nonlinear approximators. Still needs many episodes. Compared with TRPO (which the authors call "significantly less data efficient") and guided policy search (more data efficient but needs full-state locally-linear dynamics), DDPG sits in a middle spot: simple, general, brittle.
- **DDPG is the deterministic limit of SVG(0)**, and the same stabilisation tricks transfer to stochastic policies via the [[Auto-Encoding Variational Bayes (VAE)#^reparameterisation-trick|reparameterisation trick]]. That connection is the direct line to SAC.
- **Practical caveats if you implement this.** (a) The near-zero final-layer init matters — a big initial $Q$ makes the actor chase noise. (b) $L_2$ decay on the critic only, not the actor. (c) Freeze critic params when stepping the actor or you will corrupt the value function. (d) The variance across seeds in Table 1 is enormous; reporting a single run is meaningless, which is a theme [[On the Difficulty of Evaluating Baselines]] would recognise.
- **Open question worth chasing:** the Q-overestimation seen on hard tasks is exactly what TD3's twin critics and delayed policy updates were later built to fix. Read this paper knowing its failure modes were the research agenda for the next four years.

## Links

Related: [[Playing Atari with Deep Reinforcement Learning (DQN)]] · [[Soft Actor-Critic]] · [[Proximal Policy Optimization Algorithms]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[Markov Decision Process]] · [[Batch Normalization]] · [[Adam- A Method for Stochastic Optimization]] · [[Backpropagation]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] · [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)]] · [[Derivative#Gradient|gradient]]

New topics worth writing: Deterministic Policy Gradient theorem, Ornstein–Uhlenbeck process, TD3 (twin delayed DDPG), iLQG and model-predictive control, Q-value overestimation and Double Q-learning, Actor-critic methods, Off-policy vs on-policy learning
