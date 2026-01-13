---
title: "Proximal Policy Optimization Algorithms"
authors: ["John Schulman", "Filip Wolski", "Prafulla Dhariwal", "Alec Radford", "Oleg Klimov"]
year: 2017
arxiv: "1707.06347"
url: https://arxiv.org/abs/1707.06347
priority: Must-Read
read_on: 2026-08-23
tags: [paper, llm, rl, optimization, vision, theory]
---
## The Core Idea

Policy gradient methods have a nasty habit: you collect a batch of experience, take **one** gradient step, and throw the data away. If you try to take ten steps on the same batch, the policy drifts far from the one that generated the data, the gradient estimate becomes garbage, and the run collapses.

TRPO (Trust Region Policy Optimization) fixed this by adding a hard constraint — "the new policy may not be more than $\delta$ [[KL Divergence|KL]] away from the old one" — and solving the constrained problem with conjugate gradient and a Fisher-matrix (second-order) approximation. It works, but it is fiddly: you need Hessian-vector products, it breaks when the policy and value network share weights, and it breaks with dropout or other noise in the network.

PPO's trick is to throw away the constraint machinery and bake the trust region **into the loss itself**, using nothing but a `min` and a `clip`. Define the probability ratio

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$$

which is 1 at the start of every update. Then optimise

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat{A}_t\big)\Big]$$

with $\epsilon = 0.2$. That's it. It is a first-order objective you can hand to Adam, and you can now safely run 3–10 epochs of minibatch SGD over the same batch of experience.

Why the `min` matters. The clip alone would flatten the objective outside $[1-\epsilon, 1+\epsilon]$ in both directions. Taking the minimum of clipped and unclipped makes the objective a **pessimistic lower bound**: the gradient is killed only when moving further would *improve* the surrogate, and is kept alive when moving further makes it *worse*. Concretely, if the advantage $\hat{A}_t > 0$ (this action was better than average), the objective stops rewarding you once $r_t > 1+\epsilon$ — no more incentive to keep pumping up that action's probability. If $\hat{A}_t < 0$, the objective stops once $r_t < 1-\epsilon$. But if the ratio has already blown past the band in the *bad* direction, the unclipped term is smaller and it is selected, so the gradient pulls you back.

> [!NOTE] Probability ratio
> $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_\text{old}}(a_t|s_t)$ — how much more (or less) likely the current policy is to take the action that was actually taken. It is the importance-sampling weight that lets you reuse old data. $r_t=1$ means no change. ^probability-ratio

> [!NOTE] Clipped surrogate objective
> A loss that is identical to the vanilla importance-weighted policy gradient to first order around $\theta_\text{old}$, but goes flat once the policy has moved more than $\epsilon$ in the direction that would help. A trust region implemented as an if-statement instead of a constrained solve. ^clipped-surrogate

What it unlocks: a method with TRPO's stability that is "few lines of code change to a vanilla policy gradient implementation", works with shared policy/value trunks, works with recurrent nets, and parallelises across actors. This is why PPO became the default RL algorithm for a decade, including as the optimiser inside [[Training language models to follow instructions with human feedback|RLHF]].

## The Methodology

**The full objective.** When policy and value function share parameters you cannot just optimise $L^{CLIP}$, because the value head needs its own supervision. The combined per-timestep loss (maximised):

$$L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t\big[L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t)\big]$$

- $L_t^{VF} = (V_\theta(s_t) - V_t^{targ})^2$, a plain squared error on the value head.
- $S$ is the entropy of the policy's action distribution — an exploration bonus that stops the policy collapsing to a deterministic one too early.
- Atari settings: $c_1 = 1$, $c_2 = 0.01$.

**Advantage estimation.** They use truncated Generalized Advantage Estimation (GAE), which runs the policy for only $T$ steps (much shorter than an episode) and needs no lookahead past $T$:

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$\delta_t$ is the one-step TD error. $\lambda = 1$ recovers the plain discounted-return-minus-baseline estimator; $\lambda = 0.95$ is used everywhere in the paper. $\gamma = 0.99$.

**The training loop** (Algorithm 1, actor–critic style):

1. $N$ parallel actors each run the *frozen* policy $\pi_{\theta_\text{old}}$ for $T$ timesteps in the environment.
2. Compute $\hat{A}_1 \ldots \hat{A}_T$ for each actor with GAE.
3. Pool the $NT$ samples. Run $K$ epochs of minibatch Adam on $L^{CLIP+VF+S}$, minibatch size $M \le NT$.
4. Set $\theta_\text{old} \leftarrow \theta$. Repeat.

Note step 3 is where the whole contribution lives — reusing a batch $K$ times is exactly what plain policy gradient cannot do.

**Hyperparameters that mattered.**

| | MuJoCo (1M steps) | Roboschool humanoid | Atari |
|---|---|---|---|
| Horizon $T$ | 2048 | 512 | 128 |
| Adam stepsize | $3\times10^{-4}$ | KL-adaptive | $2.5\times10^{-4}\cdot\alpha$ |
| Epochs $K$ | 10 | 15 | 3 |
| Minibatch | 64 | 4096 | $32\times8$ |
| Actors $N$ | 1 | 32 / 128 | 8 |
| $\epsilon$ | 0.2 | — | $0.1\cdot\alpha$ |

$\alpha$ anneals linearly from 1 to 0 over training — so on Atari both the learning rate *and the clip width* shrink to zero. Atari gets only 3 epochs versus MuJoCo's 10.

**Policy network.** MuJoCo: an MLP with two hidden layers of 64 units and `tanh` activations, outputting the mean of a Gaussian over actions with a separately-learned standard deviation. No parameter sharing with the value net (so $c_1$ was irrelevant), no entropy bonus. Atari: the same CNN as A3C.

**The alternative they also describe: adaptive KL penalty.** Instead of clipping, optimise

$$L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t\big[r_t(\theta)\hat{A}_t - \beta\,\text{KL}[\pi_{\theta_\text{old}}, \pi_\theta]\big]$$

then after the update measure $d = \hat{\mathbb{E}}_t[\text{KL}]$ and adjust: if $d < d_\text{targ}/1.5$, halve $\beta$; if $d > d_\text{targ}\times1.5$, double $\beta$. The initial $\beta$ barely matters because it self-corrects within a few updates. This is included as a baseline, and it loses.

## Ablation Studies and Experiments

**Ablation 1 — which surrogate objective (Table 1).** Seven MuJoCo tasks (HalfCheetah, Hopper, InvertedDoublePendulum, InvertedPendulum, Reacher, Swimmer, Walker2d), 3 seeds each, 1M timesteps. Scores normalised so random policy = 0 and best result = 1, averaged over all 21 runs.

| Setting | Normalised score |
|---|---|
| **No clipping or penalty** | **−0.39** |
| Clipping $\epsilon = 0.1$ | 0.76 |
| **Clipping $\epsilon = 0.2$** | **0.82** |
| Clipping $\epsilon = 0.3$ | 0.70 |
| Adaptive KL $d_\text{targ}=0.003$ | 0.68 |
| Adaptive KL $d_\text{targ}=0.01$ | 0.74 |
| Adaptive KL $d_\text{targ}=0.03$ | 0.71 |
| Fixed KL $\beta = 0.3$ | 0.62 |
| Fixed KL $\beta = 1$ | 0.71 |
| Fixed KL $\beta = 3$ | 0.72 |
| Fixed KL $\beta = 10$ | 0.69 |

Read this carefully — it is the whole paper. Removing the clip does not merely hurt; it makes the score **negative**, meaning on HalfCheetah the final policy is worse than the untrained random one. Multiple epochs on an unconstrained importance-weighted objective is actively destructive. Every KL variant works (0.62–0.74) but none beats clipping at 0.82, and the *best* KL setting is barely better than the *worst* clip setting.

**What did not work:**

- **Multiple epochs on plain $L^{PG} = \hat{\mathbb{E}}_t[\log\pi_\theta(a_t|s_t)\hat{A}_t]$.** Not shown in tables, but described as "similar or worse than the no clipping or penalty setting" — i.e. also destructive.
- **Clipping in log space** instead of on the ratio. Tried, "no better".
- **A fixed KL penalty coefficient $\beta$.** The authors state plainly that no single $\beta$ works across problems, or even across the *course of learning on one problem*, because the geometry of the policy changes. This is exactly why TRPO used a hard constraint despite the theory recommending a penalty.
- **Unclipped $L^{CPI}$** (the conservative-policy-iteration objective) alone — the same thing as "no clipping or penalty".

**Figure 2 as evidence.** They interpolate linearly between $\theta_\text{old}$ and the post-PPO-update $\theta$ on Hopper-v1 and plot $L^{CPI}$, the clipped term, and $L^{CLIP}$. $L^{CPI}$ keeps rising as you extrapolate; $L^{CLIP}$ peaks at roughly KL $= 0.02$ and then falls. The clip is genuinely acting as a soft trust region, and the peak lands where TRPO's constraint would have put you anyway.

**Comparison 2 — continuous control (Figure 3).** PPO vs tuned A2C, A2C+trust region, cross-entropy method (CEM), vanilla PG with adaptive stepsize, and TRPO, on the same 7 MuJoCo tasks for 1M timesteps. PPO wins on almost all of them, including beating TRPO — which is the point, since PPO is strictly simpler.

**Showcase — Roboschool 3D humanoid.** Three tasks: plain forward running; Flagrun (target teleports every 200 steps); FlagrunHarder (robot is pelted with cubes and must get up off the floor). Trains to ~4000 reward on Humanoid over 50M steps, ~2500 on Flagrun and ~3000 on FlagrunHarder over 100M. The log-stdev of the action distribution is linearly annealed from $-0.7$ to $-1.6$ — the policy is manually made more deterministic over time.

**Comparison 3 — Atari, 49 games, 40M frames (Table 2).** Against tuned A2C and ACER (the latter uses a replay buffer, so it is far from simple).

| Metric | A2C | ACER | PPO | Tie |
|---|---|---|---|---|
| Avg reward over *all* training (favours fast learning) | 1 | 18 | **30** | 0 |
| Avg reward over *last 100 episodes* (favours final score) | 1 | **28** | 19 | 1 |

The honest reading: PPO is clearly more **sample efficient** than both, and crushes A2C outright. But ACER, with its replay buffer, reaches a better final policy on more games. PPO's argument on Atari is simplicity plus learning speed, not final performance. Individual results are lumpy — PPO gets 9928 on Kangaroo vs A2C's 45 and ACER's 50, and 758 on Enduro where both others score literally 0, but only 1590 on BeamRider vs A2C's 3031.

## Worth Remembering

- **The clip is not a KL constraint and does not bound KL.** It bounds the per-sample ratio, which is a much cruder, per-timestep proxy. A single update can still move the policy arbitrarily far in KL if enough samples push the same direction. In practice people add an explicit KL early-stopping check on top of PPO — including in RLHF, where the KL-to-reference penalty is a separate term serving a different purpose.
- **$\epsilon$ is not monotone.** 0.1 → 0.76, 0.2 → 0.82, 0.3 → 0.70. Too tight is as bad as too loose. And on Atari they used 0.1 annealed to 0, not 0.2 — the "right" $\epsilon$ is domain-dependent even in the original paper.
- **Epoch count $K$ is the real knob nobody talks about.** MuJoCo uses 10, Atari uses 3. More epochs = more data reuse = more sample efficiency, but also more policy drift per batch, and the clip is your only defence. This is the tradeoff dial.
- **The gradient is zero in the clipped region, but only for that sample.** Individual timesteps drop out of the update once they leave the band; the batch keeps learning from the ones still inside. It is a per-sample gate, not a global step-size rule.
- **The paper never proves anything.** TRPO had a monotonic improvement guarantee. PPO explicitly abandons that theory and offers an "approximately maximised" objective justified by Table 1. It is an empirical result dressed in the vocabulary of a theoretical one — the word "pessimistic lower bound" refers to $L^{CLIP} \le L^{CPI}$, not to a bound on true policy performance.
- **Why it beat DQN's family for continuous control.** [[Playing Atari with Deep Reinforcement Learning (DQN)|DQN]] needs a $\max$ over actions, which is intractable in continuous action spaces; footnote 1 in the paper makes exactly this point. PPO outputs a Gaussian and sidesteps it entirely.
- **Practical caveat:** reproductions of PPO differ wildly depending on implementation details the paper does not mention — advantage normalisation per minibatch, observation normalisation, value-loss clipping, orthogonal initialisation, gradient clipping. Later work ("Implementation Matters in Deep Policy Gradients") showed these details contribute as much of the gain as the clipped objective does. If you reimplement from the paper alone you will probably underperform the reference code.
- Concurrent work (Heess et al. 2017, "Emergence of Locomotion Behaviours in Rich Environments") used the **adaptive KL** variant, not the clipped one, for 3D locomotion — so the "losing" variant was in real use immediately.

## Links

Related: [[Markov Decision Process]] · [[KL Divergence]] · [[Playing Atari with Deep Reinforcement Learning (DQN)]] · [[Training language models to follow instructions with human feedback]] · [[Backpropagation]] · [[Momentum]] · [[Derivative#Hessian|Hessian]] · [[Decision Sciences]] · [[Multi-Agent Reinforcement Learning]] · [[Regularization]] · [[Uncertainty]] · [[Foundational_Reading_Plan]]

New topics worth writing: Policy gradient theorem and REINFORCE, Trust Region Policy Optimization (TRPO), Generalized Advantage Estimation (GAE), Advantage function and baselines, Importance sampling, Actor-critic methods, A3C/A2C, Entropy regularisation in RL, Natural gradient and Fisher information, Conjugate gradient method, GRPO and critic-free policy optimisation
