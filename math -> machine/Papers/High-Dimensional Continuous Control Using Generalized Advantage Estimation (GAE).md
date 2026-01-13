---
title: "High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)"
authors: ["John Schulman", "Philipp Moritz", "Sergey Levine", "Michael Jordan", "Pieter Abbeel"]
year: 2015
arxiv: "1506.02438"
url: https://arxiv.org/abs/1506.02438
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, rl]
---
## The Core Idea

Policy gradient methods need a number for every action taken: "was this action better or worse than what I usually do here?" That number is the **advantage**. The whole difficulty of policy gradient RL is that you never see the advantage — you have to estimate it, and every way of estimating it is either noisy or wrong.

The two extremes were already known:

- Use the **actual future rewards** minus a baseline. Unbiased, but the noise is enormous, because the reward you collect 500 steps from now depends on 500 other random actions, not on the one you are trying to judge.
- Use a **one-step TD residual**, $r_t + \gamma V(s_{t+1}) - V(s_t)$. Almost no noise, but if your learned value function $V$ is wrong — and it always is — this is biased, and bias does not go away with more data. It can converge to a bad policy forever.

GAE says: do not choose. Take the exponentially weighted average of *all* the $k$-step estimators, with weight $\lambda^{k-1}$. After a telescoping-sum collapse, this whole infinite mixture becomes one line of code:

$$\hat{A}^{\mathrm{GAE}(\gamma,\lambda)}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \, \delta^V_{t+l}, \qquad \delta^V_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

A discounted sum of TD residuals. $\lambda = 0$ gives the one-step estimator. $\lambda = 1$ gives Monte Carlo returns minus baseline. Anything in between is a dial between bias and variance. This is exactly the TD($\lambda$) construction, but applied to the *advantage* instead of the value.

> [!NOTE] Generalized Advantage Estimation ^gae
> The advantage estimate $\hat{A}_t = \sum_l (\gamma\lambda)^l \delta^V_{t+l}$. A geometric blend of $k$-step advantage estimators. $\lambda$ controls how much you trust the learned value function versus the raw rewards. ^gae

The second insight — and this is the part people forget — is that $\gamma$ and $\lambda$ are *not* the same knob wearing two hats. The paper deliberately sets up the problem as **undiscounted**: the true objective is $\sum_t r_t$. The discount $\gamma$ is demoted from "part of the problem" to "a variance reduction hyperparameter". Consequence:

- $\gamma < 1$ introduces bias **always**, even with a perfect value function, because you are literally throwing away rewards more than $\approx 1/(1-\gamma)$ steps out.
- $\lambda < 1$ introduces bias **only when $V$ is wrong**. If $V = V^{\pi,\gamma}$ exactly, every $\lambda$ is unbiased.

That asymmetry is why the empirically best $\lambda$ (~0.95–0.98) sits lower than the best $\gamma$ (~0.99–0.995).

What this unlocked: GAE plus [[Trust Region Policy Optimization (TRPO)|TRPO]] was the first generic, model-free method to learn 3D bipedal running and standing-up from raw joint state to raw joint torques, with a plain [[Deep Learning|MLP]] policy and no hand-designed gait. It is now the default advantage estimator inside [[Proximal Policy Optimization Algorithms|PPO]].

## The Methodology

**The estimator itself.** Define the TD residual $\delta^V_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ where $V$ is your current learned value network. The $k$-step advantage estimate is

$$\hat{A}^{(k)}_t = \sum_{l=0}^{k-1}\gamma^l \delta^V_{t+l} = -V(s_t) + r_t + \gamma r_{t+1} + \dots + \gamma^{k} V(s_{t+k})$$

which is "$k$ real rewards, then bootstrap, minus the baseline". As $k$ grows the bias shrinks (the wrong $V(s_{t+k})$ term is crushed by $\gamma^k$) and the variance grows. GAE is $(1-\lambda)\left(\hat{A}^{(1)} + \lambda \hat{A}^{(2)} + \lambda^2\hat{A}^{(3)} + \dots\right)$, which telescopes to the one-line formula above.

**The formal correctness condition.** They define an estimator as **$\gamma$-just** if plugging it into $\mathbb{E}[\hat{A}_t \nabla_\theta \log \pi_\theta(a_t|s_t)]$ gives you the true discounted policy gradient $g^\gamma$. Proposition 1: $\hat{A}_t$ is $\gamma$-just if it splits as $Q_t - b_t$ where $Q_t$ is an unbiased estimate of $Q^{\pi,\gamma}(s_t,a_t)$ and $b_t$ is *any* function of things that happened before $a_t$. The baseline term vanishes because $\mathbb{E}[\nabla_\theta \log \pi_\theta(a_t|s_t)] = 0$ — the same trick that justifies baselines in [[Simple Statistical Gradient-Following Algorithms (REINFORCE)|REINFORCE]].

**The reward-shaping reading.** This is the elegant part. Ng et al.'s potential shaping replaces $r$ with $\tilde r(s,a,s') = r + \gamma\Phi(s') - \Phi(s)$, which provably leaves the advantage function unchanged. Set $\Phi = V$. Then $\tilde r$ *is* $\delta^V$. And GAE is exactly the $\gamma\lambda$-discounted return of the shaped MDP:

$$\sum_{l=0}^\infty (\gamma\lambda)^l \tilde r_{t+l} = \hat{A}^{\mathrm{GAE}(\gamma,\lambda)}_t$$

So GAE = "reshape rewards with $V$ so the credit for an action arrives immediately, then apply a *steeper* discount $\gamma\lambda$ to cut off the leftover long-delay noise."

They formalise "credit arrives when" with the **response function** $\chi(l; s_t,a_t) = \mathbb{E}[r_{t+l}|s_t,a_t] - \mathbb{E}[r_{t+l}|s_t]$ — how much action $a_t$ shifts the reward $l$ steps later. Note $A^{\pi,\gamma}(s,a) = \sum_l \gamma^l \chi(l;s,a)$. If $\Phi$ were the *exact* $V^{\pi,\gamma}$, the shaped response function is nonzero only at $l=0$: all credit becomes immediate. A good approximate $V$ partially achieves this, which is why the aggressive $\gamma\lambda$ discount costs so little.

> [!NOTE] Response function ^response-function
> $\chi(l;s_t,a_t)$ = how much taking action $a_t$ changes the expected reward $l$ steps later. It decomposes the advantage across time and makes "long-range credit assignment" a measurable quantity. Using $\gamma<1$ means dropping all terms with $l \gg 1/(1-\gamma)$. ^response-function

**Fitting the value function.** Naive least squares $\min_\phi \sum_n \|V_\phi(s_n) - \hat V_n\|^2$ against Monte Carlo returns $\hat V_t = \sum_l \gamma^l r_{t+l}$ overfits the current batch. Instead they run a trust-region version: compute $\sigma^2 = \frac1N\sum_n \|V_{\phi_{old}}(s_n) - \hat V_n\|^2$, then minimise the same squared error subject to

$$\frac{1}{N}\sum_n \frac{\|V_\phi(s_n) - V_{\phi_{old}}(s_n)\|^2}{2\sigma^2} \le \epsilon$$

which is the average [[KL Divergence]] between old and new $V$ if you read $V_\phi(s)$ as the mean of a Gaussian with variance $\sigma^2$. Solved by linearising the objective, quadraticising the constraint, and running conjugate gradient with Hessian-vector products, where $H = \frac1N\sum_n j_n j_n^\top$, $j_n = \nabla_\phi V_\phi(s_n)$ — the Gauss-Newton approximation, which up to $\sigma^2$ is the Fisher information matrix. Same machinery as the TRPO policy step.

**Training loop.**
```
for i = 0, 1, 2, ...:
    roll out π_θi for N timesteps
    δ_t = r_t + γ V_φi(s_{t+1}) - V_φi(s_t)      # using OLD value net
    Â_t = Σ_l (γλ)^l δ_{t+l}
    θ_{i+1} ← TRPO step using Â
    φ_{i+1} ← trust-region regression step
```

**One ordering detail that matters a lot:** the policy is updated using $V_{\phi_i}$, the value function from *before* this iteration's fit. If you fit $V$ first and then compute advantages, you inject extra bias. Pathological case: perfectly overfit $V$ makes every $\delta_t = 0$ and the policy gradient becomes exactly zero.

**Setup.** [[Attention Is All You Need|No transformers here]] — a plain feedforward net, 3 hidden layers of 100/50/25 tanh units, linear output, same architecture for policy and value (value has 1 scalar output). >$10^4$ parameters each. MuJoCo physics. Humanoid: 33 state dims, 10 actuated DOF. Quadruped: 29 state dims, 8 DOF. Timestep 0.01s, episodes capped at 2000 steps. Batch: 50k timesteps for biped locomotion, 200k for quadruped and standing-up. Rewards are hand-written, e.g. biped locomotion is $v_{\text{fwd}} - 10^{-5}\|u\|^2 - 10^{-5}\|f_{\text{impact}}\|^2 + 0.2$ — forward velocity, small torque and impact penalties, plus a constant alive bonus. That constant matters: without it, the quadratic penalties make the optimal policy "fall over immediately to end the episode."

## Ablation Studies and Experiments

The comparison is deliberately narrow: TRPO is held fixed, only $\gamma$ and $\lambda$ vary. They are not benchmarking against other algorithms; the previous TRPO paper already did that.

**Cart-pole** (linear policy, 20-unit value net, 20 trajectories/batch, 21 random seeds). Best results at intermediate values on both axes: $\gamma \in [0.96, 0.99]$, $\lambda \in [0.92, 0.99]$. The learning-curve sweep at fixed $\gamma=0.99$ shows fastest improvement at $\lambda \in [0.92, 0.98]$. Both $\lambda=0$ and $\lambda=1$ are worse.

**3D bipedal locomotion** (9 seeds, ~2 hours per trial on 16 cores). Same story: best at $\gamma \in [0.99, 0.995]$, $\lambda \in [0.96, 0.99]$. After 1000 iterations they get a fast, smooth, stable gait. Total simulated experience: $0.01 \text{s} \times 50000 \times 1000 = 5.8$ days of real time — the paper's argument that this is nearly feasible on physical robots if you could reset them safely.

**Quadruped and biped-standing** (5 seeds each, ~4 hours per trial on 32 cores). Limited sweep: $\gamma = 0.995$ fixed, $\lambda \in \{0, 0.96\}$, plus a **"No VF"** condition using a time-dependent baseline (just the average return at each timestep across the batch, no state dependence).

- Quadruped: $\lambda = 0.96$ with a value function wins.
- Standing up: the value function always helps, but $\lambda = 0.96$ and $\lambda = 1$ are roughly tied.

**What did not work:**

- **$\lambda = 0$** — the pure one-step TD estimator. "Excessive bias and poor performance." This is the single clearest negative result, and it is the reason the whole paper exists. Note the honest caveat in the discussion: concurrent work ([[Continuous control with deep reinforcement learning (DDPG)|DDPG]], SVG) *does* make one-step methods work, but on much lower-dimensional problems.
- **A parameterised $Q$-function instead of $V$.** Appendix A.2 gives two reasons they avoided it: $V$ takes a lower-dimensional input and is easier to learn, and more importantly $\hat A = Q(s,a) - V(s)$ gives you *only* the high-bias estimator — there is no $\lambda$ dial to interpolate toward low bias. Given how badly $\lambda=0$ performed, they expect $Q$-based advantages to suffer similarly.
- **No value function at all** (time-dependent baseline). Consistently worse; on the standing-up task the value function helped in every configuration.
- **$\lambda = 1$** — unbiased but too noisy. Never the winner except in the tie on standing-up.
- Fitting $V$ before computing advantages — flagged as a bias source, not run as an experiment but argued from the degenerate case.

What the ablations reveal: the win comes from *both* pieces together. A learned state-value baseline is necessary (No VF is worse) but not sufficient (with $\lambda=0$ you get a value function and still fail). The $\lambda$ interpolation is doing real work.

## Worth Remembering

- **The analysis is explicitly informal.** The authors say so in the discussion: "an intuitive but informal analysis." There is no bias-variance theorem here, just Proposition 1 (which only covers the $\lambda=1$ unbiased case) and good intuition. The formula itself was already in Kimura & Kobayashi (1998) and Wawrzyński (2009); the contribution is the framing, the reward-shaping interpretation, and making it work in a batch trust-region setting.

- **The undiscounted framing is a genuine conceptual move.** Most RL papers treat $\gamma$ as given by the problem. Here the objective is $\sum_t r_t$ and $\gamma$ is a hyperparameter you tune for variance. This also means the reported "discounted policy gradient" $g^\gamma$ is itself already a biased approximation of the thing you actually want, before $\lambda$ enters at all. Two layers of bias, stacked deliberately.

- **Practical defaults that stuck:** $\gamma \approx 0.99$, $\lambda \approx 0.95$. These are still the PPO defaults a decade later. If you only remember one thing: $\lambda$ lower than $\gamma$.

- **Implementation gotcha.** The infinite sum is computed backwards in one pass: `A[t] = δ[t] + γλ·A[t+1]`, with `A[T] = 0` at episode end. For truncated (not terminated) episodes you must bootstrap, otherwise you silently treat "ran out of time" as "died."

- **Open questions the authors flag.** (1) Adaptive/automatic tuning of $\gamma,\lambda$. (2) The relationship between value-function fitting error and policy-gradient error is unknown — squared error on returns is a proxy, not the thing you care about; they suggest Bellman error or projected Bellman error as candidates. (3) Sharing a trunk between policy and value networks — now completely standard practice, but in 2015 they listed it as an open problem with no convergence guarantees.

- **Connection to compatible features** (Appendix A.1): if you project any $\gamma$-just $\hat A_t$ onto the span of $\nabla_{\theta_i}\log\pi_\theta(a_t|s_t)$ by least squares, the solution *is* the natural policy gradient. Compatible-features theory tells you nothing about exploiting temporal structure, which is why it is orthogonal to GAE rather than competing with it.

- The trust-region value fit is more machinery than most people use today; the common modern substitute is a clipped value loss plus a few epochs of [[Adam- A Method for Stochastic Optimization|Adam]]. The *purpose* is the same — do not let $V$ chase this batch too hard.

## Links

Related: [[Proximal Policy Optimization Algorithms]] · [[Trust Region Policy Optimization (TRPO)]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[Markov Decision Process]] · [[KL Divergence]] · [[Continuous control with deep reinforcement learning (DDPG)]] · [[Soft Actor-Critic]] · [[Playing Atari with Deep Reinforcement Learning (DQN)]] · [[A Distributional Perspective on Reinforcement Learning]] · [[Derivative#Hessian|Hessian]] · [[Uncertainty]]

New topics worth writing: TD(λ) and eligibility traces, Potential-based reward shaping (Ng et al. 1999), Natural policy gradient and the Fisher information matrix, Conjugate gradient with Hessian-vector products, Actor-critic methods, MuJoCo and continuous control benchmarks, Bias-variance tradeoff in gradient estimators
