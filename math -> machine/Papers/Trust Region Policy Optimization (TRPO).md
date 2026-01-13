---
title: "Trust Region Policy Optimization (TRPO)"
authors: ["John Schulman", "Sergey Levine", "Philipp Moritz", "Michael I. Jordan", "Pieter Abbeel"]
year: 2015
arxiv: "1502.05477"
url: https://arxiv.org/abs/1502.05477
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, llm, rl, vision, theory]
---
## The Core Idea

Policy gradient methods have one nasty failure mode: you take a step in parameter space, and the policy gets *worse* — sometimes catastrophically, unrecoverably worse. Unlike supervised learning, a bad step is not just a bad batch. The policy generates the next batch of data. A collapsed policy collects garbage data and never recovers.

The reason is that the step size in **parameter** space tells you nothing about the size of the change in **behaviour** space. Move $\theta$ by $0.01$ and a Gaussian policy with a small standard deviation might completely swap which actions it prefers.

TRPO's insight: measure the step in behaviour space instead, using [[KL Divergence]] between the old and new action distributions, and put a **hard constraint** on it. Then solve the constrained problem exactly-ish at every iteration.

What makes the paper more than a heuristic is that it starts from a theorem. Kakade & Langford (2002) had a monotonic improvement guarantee, but only for policies formed by *mixing* the old policy with a new one — useless for neural nets. TRPO extends the guarantee to **any** stochastic policy, with the mixing coefficient $\alpha$ replaced by a divergence between the two policies:

$$\eta(\tilde\pi) \;\ge\; L_\pi(\tilde\pi) - \frac{4\epsilon\gamma}{(1-\gamma)^2}\, D^{\max}_{\mathrm{KL}}(\pi,\tilde\pi), \qquad \epsilon = \max_{s,a}|A_\pi(s,a)|$$

That is a **lower bound** on the true return. Maximise the bound, and the true return cannot go down. This is a minorization–maximization (MM) algorithm, the same family as EM.

> [!NOTE] Surrogate objective $L_\pi$
> The true return of a new policy depends on which states the *new* policy visits, which you cannot sample without running it. $L_\pi(\tilde\pi)$ cheats by keeping the **old** policy's state visitation frequencies $\rho_\pi$ and only swapping the action distribution. It matches $\eta$ in value and in [[Derivative#Gradient|gradient]] at $\theta = \theta_{\text{old}}$, so it is a good local model. ^surrogate-objective

What this unlocked: for the first time, a single general-purpose policy search method learned swimming, hopping and walking gaits from scratch with plain MLPs and a naive reward, *and* played Atari from pixels — with essentially no per-task tuning. Before this, locomotion needed hand-engineered low-dimensional policy classes. TRPO is also the direct ancestor of [[Proximal Policy Optimization Algorithms|PPO]], which replaces the expensive constrained solve with a clipped ratio, and PPO is what powers [[Training language models to follow instructions with human feedback|RLHF]].

## The Methodology

Standard discounted [[Markov Decision Process]] $(\mathcal{S},\mathcal{A},P,r,\rho_0,\gamma)$. Advantage $A_\pi(s,a) = Q_\pi(s,a) - V_\pi(s)$.

**Step 1 — from theory to a usable problem.** The theoretical penalty coefficient $C = 4\epsilon\gamma/(1-\gamma)^2$ is enormous (with $\gamma=0.99$ that is $\sim 40{,}000\epsilon$), so the steps it allows are microscopic. Practical fix: throw away the penalty, use a hard constraint with a tunable budget $\delta$.

Second problem: the theory constrains the KL at *every* state ($D^{\max}_{\mathrm{KL}}$), which is infinitely many constraints. Practical fix: constrain the **average** KL over visited states.

$$\max_\theta \; \mathbb{E}_{s\sim\rho_{\theta_{\text{old}}},\, a\sim q}\!\left[\frac{\pi_\theta(a|s)}{q(a|s)} Q_{\theta_{\text{old}}}(s,a)\right] \quad \text{s.t.} \quad \mathbb{E}_{s\sim\rho_{\theta_{\text{old}}}}\big[D_{\mathrm{KL}}(\pi_{\theta_{\text{old}}}(\cdot|s)\,\|\,\pi_\theta(\cdot|s))\big] \le \delta$$

Note the importance-sampling ratio $\pi_\theta/q$ — this is exactly the ratio PPO later clips. Note also they use $Q$ rather than $A$; it only shifts the objective by a constant.

**Step 2 — getting $Q$ estimates.** Two schemes.

*Single path*: roll out trajectories with $\pi_{\theta_{\text{old}}}$, so $q = \pi_{\theta_{\text{old}}}$, and set $\hat{Q}(s_t,a_t)$ to the discounted sum of rewards from $t$ onward. This is what everyone actually uses. Works on real hardware.

*Vine*: roll out "trunk" trajectories, pick $N$ states as a rollout set, and from each state fire off $K$ separate actions with a fresh rollout after each. Uses **common random numbers** (same RNG seed for all $K$ branches) so the $Q$ differences between actions have far lower variance. Per state:

$$L_n(\theta) = \frac{\sum_{k=1}^{K} \frac{\pi_\theta(a_{n,k}|s_n)}{\pi_{\theta_{\text{old}}}(a_{n,k}|s_n)}\hat{Q}(s_n,a_{n,k})}{\sum_{k=1}^{K} \frac{\pi_\theta(a_{n,k}|s_n)}{\pi_{\theta_{\text{old}}}(a_{n,k}|s_n)}}$$

The self-normalising denominator removes the need for a baseline. Vine gives much better advantage estimates per $Q$-sample but needs far more simulator calls and, fatally, **needs the ability to reset the simulator to an arbitrary state**. Simulation only.

**Step 3 — solving the constrained problem.** Linearise the objective, quadratically approximate the constraint. The Hessian of the average KL is the **Fisher information matrix** $A$, so:

$$\max_\theta \; g^\top(\theta - \theta_{\text{old}}) \quad \text{s.t.} \quad \tfrac{1}{2}(\theta-\theta_{\text{old}})^\top A (\theta - \theta_{\text{old}}) \le \delta$$

Solution direction $s = A^{-1}g$. With 30k+ parameters you cannot form $A$, so they run **conjugate gradient** using only Fisher–vector products $y \mapsto Ay$. $k=10$ CG iterations; more did not help.

The Fisher–vector product is computed as $J^\top M J y$, where $J = \partial\mu_a(x)/\partial\theta_i$ is the [[Vector Jacobian Product|Jacobian]] of the network output $\mu$ (the distribution parameters), and $M$ is the Fisher of the distribution in terms of $\mu$ — a closed form for Gaussians and categoricals. Multiplying by $J^\top$ is just [[Backpropagation]]. Crucially they use the **analytic Hessian of the KL**, not the outer product of sampled log-prob gradients (the "empirical FIM"), because it integrates over actions and does not require storing per-sample gradients.

Cost trick: the Fisher is only a metric, so compute it on a **10% subsample** of the batch. Then 10 CG iterations cost about as much as one gradient.

**Step 4 — step length and line search.** From $\delta = \tfrac{1}{2}\beta^2 s^\top A s$ you get the max step $\beta = \sqrt{2\delta/(s^\top A s)}$. Then **backtrack**: shrink $\beta$ exponentially until the *true* (nonlinear) surrogate improves and the *true* KL is still under $\delta$. This line search is not optional — see below.

**Policies.** Continuous: MLP outputs the mean of a diagonal Gaussian; log-std is a separate state-independent parameter vector. Discrete (Atari): factored categorical, softmax per factor.

**Hyperparameters.** $\delta = 0.01$ everywhere. $\gamma = 0.99$. Swimmer: 10-D state, 364 policy params, 50K sim steps/iter, 30 hidden units. Hopper: 12-D, 4806 params, 1M steps/iter, 50 hidden. Walker: 20-D, 8206 params, 1M steps/iter. 200 policy iterations. Atari: two conv layers, 16 channels, stride 2, then one FC layer of 20 units → 33,500 params; 500 iterations, ~30 hours on 16 cores.

## Ablation Studies and Experiments

**MuJoCo locomotion (swimmer / hopper / walker), 5 seeds each.** Competitors: single-path TRPO, vine TRPO, CEM, CMA, natural gradient, empirical FIM, max KL.

- **Both TRPO variants solved all three tasks** and got the best final returns. No hand-engineered policy class, no balance priors.
- **Natural gradient failed on hopper and walker.** It is the *same* update direction $A^{-1}g$ — the only difference is a fixed step size (fixed Lagrange multiplier) instead of enforcing $\bar D_{\mathrm{KL}} \le \delta$ at every update. They swept the step size in factors of three and took the best. It still could not produce forward locomotion; it learned to stand. **This is the ablation that isolates the paper's actual contribution**: the constraint, not the natural-gradient metric, is what makes it work.
- **Empirical FIM** (outer product of sampled gradients instead of the analytic KL Hessian) improved at a similar rate. So the analytic FIM is a *computational* win, not an accuracy win.
- **Max KL** (the theoretically correct $D^{\max}_{\mathrm{KL}}$, tractable only on cart-pole) learned somewhat *slower* than the average-KL version, because the constraint is more restrictive. It reaches similar quality. So the average-KL heuristic is a good, slightly-more-aggressive stand-in.
- **CEM and CMA** — gradient-free — did fine on cart-pole (6 parameters) and collapsed on the larger problems. Sample complexity of derivative-free search scales badly with dimension. This is the paper's argument for gradients over black-box search.

**Atari, 7 games from the ALE, one run each, same architecture and hyperparameters for all games.**

| | B.Rider | Breakout | Enduro | Pong | Q*bert | Seaquest | S.Invaders |
|---|---|---|---|---|---|---|---|
| Random | 354 | 1.2 | 0 | −20.4 | 157 | 110 | 179 |
| Human | 7456 | 31.0 | 368 | −3.0 | 18900 | 28010 | 3690 |
| DQN | 4092 | 168.0 | 470 | 20.0 | 1952 | 1705 | 581 |
| UCC-I (MCTS+supervised) | 5702 | 380 | 741 | 21 | 20025 | 2995 | 692 |
| TRPO single path | 1425.2 | 10.8 | 534.6 | **20.9** | 1973.5 | 1908.6 | 568.4 |
| TRPO vine | 859.5 | 34.2 | 430.8 | 20.9 | **7732.5** | 788.4 | 450.2 |

Honest reading: TRPO **loses to DQN on most games**. It wins on Enduro and Q*bert (vine), ties on Pong, and is badly beaten on Beam Rider and Breakout. The claim is generality, not dominance — the same algorithm that learned to walk also plays Atari respectably. They ran each game once and admit they could not produce error bars because of time constraints, and that run-to-run variance is substantial.

**Vine vs single path.** Vine wins big on Q*bert (7732 vs 1973) and Breakout, loses on Beam Rider and Seaquest. On locomotion both work. Vine costs far more simulator calls for the same number of $Q$ estimates — Walker vine takes 40 min/setting vs 100 min single path but with 400K vs 100K sim steps per iteration on Atari.

**What did not work / what breaks:**

- **The theoretical penalty coefficient $C$.** Using it makes steps "prohibitively small". The entire theory is used as motivation, then abandoned for a tuned $\delta$.
- **Fixed penalty coefficient in general.** "Empirically, it is hard to robustly choose the penalty coefficient" — hence the hard constraint.
- **Dropping the line search.** Without it, "the algorithm occasionally computes large steps that cause a catastrophic degradation of performance." The quadratic approximation of the KL is only locally valid; the backtracking check on the true KL is a safety net you cannot remove.
- The theory **ignores advantage estimation error** entirely. In practice $\hat{A}$ is noisy Monte Carlo, so the monotonic guarantee is not real. The paper says it "tends to give" monotonic improvement.

## Worth Remembering

- The three approximations that separate the theorem from the code: (1) hard constraint replaces the penalty, (2) average KL replaces max KL, (3) sampled advantages replace exact ones. Each one breaks the guarantee. TRPO is a *theoretically motivated* algorithm, not a theoretically guaranteed one.
- The unifying view in Section 7 is worth internalising. Change only the constraint on the step and you recover different classic algorithms: an $\ell_2$ constraint $\tfrac{1}{2}\|\theta - \theta_{\text{old}}\|^2 \le \delta$ gives the **vanilla policy gradient** ([[Simple Statistical Gradient-Following Algorithms (REINFORCE)|REINFORCE]]); a quadratic KL constraint with a fixed multiplier gives the **natural policy gradient**; no constraint at all gives **policy iteration**. Same objective $L$, different geometry. This is the same "which norm are you descending in?" framing as [[Old Optimizer, New Norm- An Anthology (Muon)|Muon]].
- Practical caveat: TRPO is a pain to implement. Conjugate gradient, Fisher-vector products, backtracking line search, no minibatching in the usual sense. This is exactly why PPO — which gets ~the same behaviour from a clipped ratio and plain [[Adam- A Method for Stochastic Optimization|Adam]] — displaced it almost entirely. If you are reading TRPO today, read it to understand *why* PPO's clip exists.
- TRPO is **on-policy** and sample-hungry: Walker used 1M simulator steps per iteration × 200 iterations. Contrast with off-policy methods like [[Continuous control with deep reinforcement learning (DDPG)|DDPG]] and [[Soft Actor-Critic]] that reuse a replay buffer. The authors flag reducing sample complexity via model learning as the route to real robots.
- The vine estimator's requirement — reset the simulator to an arbitrary state — is the single biggest practical limitation, and it is why vine essentially disappeared from the literature.
- No learned value function baseline here; $Q$ is raw Monte Carlo. GAE (Schulman et al., 2015) came next and is what people actually pair with TRPO/PPO.
- Note they use the *forward* KL direction $D_{\mathrm{KL}}(\pi_{\text{old}} \| \pi_{\text{new}})$ in the constraint, whose Hessian at $\theta_{\text{old}}$ is the Fisher matrix. The choice matters for what the quadratic approximation looks like.

## Links

Related: [[Proximal Policy Optimization Algorithms]] · [[Markov Decision Process]] · [[KL Divergence]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[Playing Atari with Deep Reinforcement Learning (DQN)]] · [[Soft Actor-Critic]] · [[Continuous control with deep reinforcement learning (DDPG)]] · [[Training language models to follow instructions with human feedback]] · [[Old Optimizer, New Norm- An Anthology (Muon)]] · [[Derivative]] · [[Vector Jacobian Product]] · [[Backpropagation]] · [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]]

New topics worth writing: Natural policy gradient, Fisher information matrix, Conjugate gradient method, Generalized Advantage Estimation (GAE), Minorization-maximization (MM) algorithms, Total variation distance and Pinsker's inequality, Trust region methods in numerical optimisation, Common random numbers for variance reduction, Importance sampling and self-normalised estimators, MuJoCo and the Arcade Learning Environment
