---
title: "Soft Actor-Critic"
authors: ["Haarnoja et al."]
year: 2018
arxiv: "1801.01290"
url: https://arxiv.org/abs/1801.01290
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, llm, rl, optimization]
---
## The Core Idea

Standard reinforcement learning asks: **maximise total reward**. Soft Actor-Critic (SAC) asks a different question: **maximise total reward *and* stay as random as you can while doing it**.

The objective becomes

$$J(\pi)=\sum_{t=0}^{T}\mathbb{E}_{(\mathbf{s}_t,\mathbf{a}_t)\sim\rho_\pi}\big[r(\mathbf{s}_t,\mathbf{a}_t)+\alpha\,\mathcal{H}\big(\pi(\cdot\mid\mathbf{s}_t)\big)\big]$$

where $\mathcal{H}(\pi(\cdot\mid s)) = -\mathbb{E}_{a\sim\pi}[\log \pi(a\mid s)]$ is the entropy of the policy at that state — how spread out its action choices are. $\alpha$ is a temperature knob. As $\alpha\to 0$ you get ordinary RL back.

> [!NOTE] Maximum entropy RL
> Reward the agent not just for doing well, but for keeping its action distribution wide. The optimal policy becomes $\pi(a\mid s)\propto\exp(Q(s,a)/\alpha)$ — a Boltzmann distribution over Q-values rather than an argmax. Ties between equally good actions get equal probability mass instead of an arbitrary winner. ^max-entropy-rl

Why this matters, and why it did not exist in this form before. In 2018 you had two families of continuous-control algorithms and both were unsatisfying:

- **On-policy** methods ([[Proximal Policy Optimization Algorithms|PPO]], TRPO, A3C) are stable but throw away every sample after one gradient step. Millions of environment steps for a walking robot.
- **Off-policy** methods (DDPG) reuse a replay buffer and are sample-efficient, but DDPG's actor is *deterministic* and its only exploration is hand-injected noise. The deterministic actor and the Q-function chase each other into bad fixed points. DDPG is famously brittle — change the seed, get a different outcome.

Maximum-entropy RL existed already (soft Q-learning, Ziebart's inverse RL), but it had been cast as **Q-learning**. In continuous action spaces that forces you to sample from $\exp(Q(s,\cdot))$, an unnormalised density over a continuous space, which needs a whole approximate-inference machine bolted on.

SAC's contribution is the missing quadrant: an **off-policy actor-critic** in the maximum-entropy framework. The actor is a stochastic policy trained with [[Backpropagation|backprop]] through the Q-network, not an approximate sampler. And crucially the entropy bonus is not a regulariser added on top — it is *inside* the Bellman backup, so the critic values states that lead to high-entropy futures.

What it unlocks: DDPG-level sample efficiency with PPO-level (better, actually) stability, on tasks like 21-dimensional Humanoid where DDPG scores essentially zero.

## The Methodology

**Setup.** Infinite-horizon [[Markov Decision Process|MDP]] $(\mathcal{S},\mathcal{A},p,r)$, continuous states and actions, discount $\gamma$.

### The soft Bellman equations

Two definitions carry the whole method. The soft state value:

$$V(\mathbf{s}_t)=\mathbb{E}_{\mathbf{a}_t\sim\pi}\big[Q(\mathbf{s}_t,\mathbf{a}_t)-\log\pi(\mathbf{a}_t\mid\mathbf{s}_t)\big]$$

That $-\log\pi$ term *is* the entropy bonus (its expectation is $\mathcal{H}$). And the backup operator:

$$\mathcal{T}^\pi Q(\mathbf{s}_t,\mathbf{a}_t)\triangleq r(\mathbf{s}_t,\mathbf{a}_t)+\gamma\,\mathbb{E}_{\mathbf{s}_{t+1}\sim p}[V(\mathbf{s}_{t+1})]$$

Repeatedly applying $\mathcal{T}^\pi$ converges to the soft Q-value of $\pi$ (Lemma 1) — the proof just folds entropy into the reward and cites standard policy-evaluation convergence.

### Policy improvement as a KL projection

The theoretically optimal update is "set $\pi \propto \exp(Q)$". But you want a tractable policy (a Gaussian, say), so you *project* onto a family $\Pi$ using [[KL Divergence]]:

$$\pi_{\mathrm{new}}=\arg\min_{\pi'\in\Pi}\;D_{\mathrm{KL}}\!\left(\pi'(\cdot\mid\mathbf{s}_t)\;\Big\|\;\frac{\exp(Q^{\pi_{\mathrm{old}}}(\mathbf{s}_t,\cdot))}{Z^{\pi_{\mathrm{old}}}(\mathbf{s}_t)}\right)$$

The partition function $Z$ is intractable, but it depends only on the state, not on $\phi$, so it vanishes from the gradient. That is the trick that makes the whole thing implementable.

Lemma 2 says this projection never makes things worse: $Q^{\pi_{\mathrm{new}}}\ge Q^{\pi_{\mathrm{old}}}$ everywhere. Theorem 1 chains this into convergence to the best policy *within $\Pi$* — a stronger claim than soft Q-learning, whose convergence depended on how well its sampler network approximated the true posterior.

### The three (four) networks

Practical SAC parameterises:

- $V_\psi(\mathbf{s})$ — state value, plus a target copy $V_{\bar\psi}$
- $Q_{\theta_1}, Q_{\theta_2}$ — two independent Q-functions
- $\pi_\phi(\mathbf{a}\mid\mathbf{s})$ — Gaussian, mean and log-std from a network

**Value loss** (regress $V$ onto the soft value implied by $Q$ and $\pi$):

$$J_V(\psi)=\mathbb{E}_{\mathbf{s}_t\sim\mathcal{D}}\Big[\tfrac12\big(V_\psi(\mathbf{s}_t)-\mathbb{E}_{\mathbf{a}_t\sim\pi_\phi}[Q_\theta(\mathbf{s}_t,\mathbf{a}_t)-\log\pi_\phi(\mathbf{a}_t\mid\mathbf{s}_t)]\big)^2\Big]$$

States come from the replay buffer $\mathcal{D}$; **actions come from the current policy**, not the buffer. One sample gives an unbiased gradient.

**Q loss** (soft Bellman residual):

$$J_Q(\theta)=\mathbb{E}_{(\mathbf{s}_t,\mathbf{a}_t)\sim\mathcal{D}}\Big[\tfrac12\big(Q_\theta(\mathbf{s}_t,\mathbf{a}_t)-r(\mathbf{s}_t,\mathbf{a}_t)-\gamma V_{\bar\psi}(\mathbf{s}_{t+1})\big)^2\Big]$$

Here both state and action come from the buffer — fully off-policy, exactly like [[Playing Atari with Deep Reinforcement Learning (DQN)|DQN]]'s use of [[Playing Atari with Deep Reinforcement Learning (DQN)#^experience-replay|experience replay]].

**Policy loss.** You could use a likelihood-ratio estimator ([[Simple Statistical Gradient-Following Algorithms (REINFORCE)|REINFORCE]]), but the target density here *is* a differentiable neural net, so instead they use the [[Auto-Encoding Variational Bayes (VAE)#^reparameterisation-trick|reparameterisation trick]]. Write $\mathbf{a}_t=f_\phi(\epsilon_t;\mathbf{s}_t)$ with $\epsilon_t\sim\mathcal{N}(0,I)$, and the KL objective becomes

$$J_\pi(\phi)=\mathbb{E}_{\mathbf{s}_t\sim\mathcal{D},\,\epsilon_t\sim\mathcal{N}}\big[\log\pi_\phi(f_\phi(\epsilon_t;\mathbf{s}_t)\mid\mathbf{s}_t)-Q_\theta(\mathbf{s}_t,f_\phi(\epsilon_t;\mathbf{s}_t))\big]$$

Gradient flows through $f_\phi$ into $Q$ — much lower variance than score-function estimators. This generalises DDPG's deterministic policy gradient to any stochastic policy you can sample and evaluate.

### Two details that matter in code

**Twin Q-functions.** Train $Q_{\theta_1},Q_{\theta_2}$ independently, then use $\min(Q_{\theta_1},Q_{\theta_2})$ inside the value and policy gradients. This is the double-Q / TD3 trick against optimistic bias in the max. SAC *can* learn Humanoid with one Q-function, but two are "significantly" faster on hard tasks.

**Squashed Gaussian.** Actions must be bounded, so sample $\mathbf{u}\sim\mathcal{N}$ and set $\mathbf{a}=\tanh(\mathbf{u})$. The change-of-variables correction to the log-density is

$$\log\pi(\mathbf{a}\mid\mathbf{s})=\log\mu(\mathbf{u}\mid\mathbf{s})-\sum_{i=1}^{D}\log\big(1-\tanh^2(u_i)\big)$$

Forget that Jacobian term and your entropy estimate is wrong.

### Training loop

Per iteration: take **one** environment step, push the transition to $\mathcal{D}$, then take one gradient step on each of $\psi$, $\theta_1$, $\theta_2$, $\phi$, then Polyak-update the target: $\bar\psi\leftarrow\tau\psi+(1-\tau)\bar\psi$.

Hyperparameters: [[Adam- A Method for Stochastic Optimization|Adam]], lr $3\cdot10^{-4}$, $\gamma=0.99$, buffer $10^6$, 2 hidden layers × 256 units, ReLU, batch 256, $\tau=0.005$.

## Ablation Studies and Experiments

**Benchmarks.** OpenAI Gym MuJoCo (Hopper-v1, Walker2d-v1, HalfCheetah-v1, Ant-v1, Humanoid-v1) plus the harder rllab Humanoid (21 action dims). Five seeds each, evaluation rollout every 1000 env steps; plots show mean with min–max shading.

**Baselines.** DDPG, PPO, soft Q-learning (SQL, also given twin Q's to be fair), TD3 (concurrent work, author's own code), and Trust-PCL in the appendix.

**Results.** Comparable to baselines on easy tasks; wins by a wide margin on hard ones, in both learning speed and final return. Specifics:

- **DDPG makes no progress at all** on Ant-v1, Humanoid-v1, and Humanoid (rllab). This matches prior reports (Gu et al., Duan et al.).
- **PPO is much slower**, because it needs huge batches to be stable in high dimensions.
- **SQL learns everything but is slower and plateaus lower** — the max-entropy objective alone is not the win; the actor-critic formulation is doing real work.
- **Trust-PCL fails to solve most tasks** in the given step budget.

### Stochastic vs deterministic (the key ablation)

They build a deterministic SAC: drop the entropy terms from both the critic and actor updates, drop the value network, use fixed Gaussian exploration noise, hard target updates. This is basically DDPG-with-two-Q-functions.

On Humanoid (rllab), the five seeds of deterministic SAC **fan out wildly** — some learn, some do not. SAC's five seeds sit almost on top of each other. This is the paper's central empirical claim: *stochasticity plus entropy maximisation is what buys stability across seeds*, not the twin Q's, not the target network.

Interestingly, in Appendix E the deterministic ablation performs *comparably to SAC on all tasks except Humanoid (rllab)* in average return. So the entropy term only clearly pays off when the task is hard enough that the hyperparameter basin gets narrow.

### Reward scale — the one hyperparameter that bites

Since $\alpha$ is folded into the reward by scaling, **reward scale = inverse temperature**. Figure 3(b) on Ant-v1:

- Too small → policy stays near-uniform, never exploits the reward, badly degraded return.
- Too large → fast early progress, then the policy collapses to near-deterministic and gets stuck in a poor local optimum from lack of exploration.
- Right value → best of both.

Tuned per environment: 5 for Hopper/Walker2d/HalfCheetah/Ant, 10 for Humanoid (rllab), **20 for Humanoid-v1**. The authors say this is the *only* hyperparameter that needed tuning. (This is exactly the pain that SAC-v2's automatic temperature tuning later removed.)

### Target smoothing $\tau$

Large $\tau$ (fast-moving target) → instability. Small $\tau$ → slow learning. But the usable range is wide, and 0.005 worked for every task. Not a sensitive knob.

### Evaluation policy

Evaluating with the sampled stochastic action gives lower return than evaluating with the **mean** of the Gaussian. Makes sense — training optimises reward *plus* entropy, but the plots report reward alone, and the mean action is an MAP approximation. Worth remembering that SAC's training curves and its deployed-policy performance are two different quantities.

### The hard-target-update variant

Copying $\psi$ into $\bar\psi$ every 1000 gradient steps instead of Polyak averaging also works, and benefits from taking >1 gradient step per env step (4 for non-humanoids) — better performance, higher compute cost.

## Worth Remembering

- **The two theorems only hold for $|\mathcal{A}|<\infty$ and the tabular case.** The convergence proof is for soft policy iteration in a finite action space; the deep version is an approximation with no guarantee. Read the theory as motivation, not as a bound.

- **The separate $V_\psi$ network is admitted to be unnecessary in principle** — $V$ is determined by $Q$ and $\pi$ via Equation 3, and can be estimated unbiasedly from a single action sample. They keep it because it "stabilises training". SAC-v2 (2019) drops it entirely and the algorithm is fine. So this is a piece of the architecture that turned out not to be load-bearing.

- **The entropy shows up in two places and does two different jobs.** In the policy update it stops the variance from collapsing to zero prematurely. In the value update it inflates the value of state regions that permit high-entropy behaviour — i.e. it makes the critic itself prefer to keep options open. That second effect is the one Q-learning-flavoured entropy methods share, but the first is unique to having a real actor.

- **Reward scaling is a temperature in disguise.** If you drop SAC into a new environment and it either never learns or learns then collapses, check the reward magnitude first. This is also why SAC does not transfer hyperparameters across environments with different reward units.

- **Off-policy is what buys sample efficiency; entropy is what buys robustness.** Two orthogonal ingredients. The paper's own ablation shows you can get most of the performance without entropy on easy tasks — but you also get DDPG's seed lottery.

- **Connection to control-as-inference.** The keyword list literally says "control as inference". $\pi\propto\exp(Q/\alpha)$ is the posterior over actions if you treat optimality as a binary observed variable; the KL projection is variational inference, structurally the same move as the [[Auto-Encoding Variational Bayes (VAE)#^elbo|ELBO]]. If you find SAC's objective arbitrary, this is the frame that makes it inevitable.

- **Follow-ups worth knowing:** SAC-v2 adds automatic $\alpha$ tuning by constraining the average entropy to a target (typically $-\dim(\mathcal{A})$) and removes $V_\psi$. SAC is also the standard model-free comparison point for model-based methods like [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)|Dreamer]], and the base algorithm many [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives|offline RL]] methods (CQL, BEAR) build a constraint on top of.

- **Practical caveat:** one gradient step per environment step is the default, but SAC scales well with more (the update-to-data ratio). Later work (REDQ) pushes this to 20 with more Q-networks. If you have a slow simulator, that trade is available.

## Links

Related: [[Proximal Policy Optimization Algorithms]] · [[Playing Atari with Deep Reinforcement Learning (DQN)]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[KL Divergence]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Markov Decision Process]] · [[Adam- A Method for Stochastic Optimization]] · [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)]] · [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] · [[A Distributional Perspective on Reinforcement Learning]] · [[Uncertainty]] · [[Backpropagation]]

New topics worth writing: DDPG and the deterministic policy gradient, TD3 and clipped double Q-learning, Control as inference / the probabilistic graphical model view of RL, Soft Q-learning and energy-based policies, Entropy (information-theoretic), Polyak averaging and target networks, Automatic temperature tuning in SAC-v2, Update-to-data ratio and replay ratio in off-policy RL, Maximum entropy inverse reinforcement learning
