---
title: "Conservative Q-Learning for Offline RL"
authors: ["Aviral Kumar", "Aurick Zhou", "George Tucker", "Sergey Levine"]
year: 2020
arxiv: "2006.04779"
url: https://arxiv.org/abs/2006.04779
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, llm, rl, theory]
---
## The Core Idea

Offline RL means learning a policy from a fixed dataset with no new interaction. The killer problem is simple. Q-learning computes a target $r + \gamma \max_{a'} Q(s', a')$. That max looks at actions the dataset never contains. The network's guess for those actions is arbitrary. If the guess is too high, the policy chases it, and there is no environment to correct the mistake. Errors compound through the [[Markov Decision Process|Bellman]] backup and the values blow up — sometimes to $10^{12}$.

Everyone before this attacked it from the **policy** side: constrain the learned policy to stay near the behaviour policy (BEAR, BRAC, BCQ). That needs an estimate of the behaviour policy, which is itself hard, especially when the data came from many different sources.

CQL attacks it from the **Q-function** side. Add one term to the standard Bellman loss that *pushes down* Q-values on actions the policy wants, and *pushes up* Q-values on actions actually in the data. Do this hard enough and the learned Q-function becomes a **lower bound** on the true value. A policy trained against a pessimistic critic cannot be fooled by a hallucinated high value — the worst case is that it is too timid.

The clever refinement: a naive penalty gives a lower bound at *every* $(s,a)$ pair, which is far too pessimistic. But policy evaluation and improvement only ever need $\mathbb{E}_{a \sim \pi}[Q(s,a)]$ — the *expected* value. So CQL only guarantees the lower bound on that expectation, and buys back a much tighter estimate. This is the whole trick.

> [!NOTE] Conservative Q-learning
> Regularise the critic so that $\mathbb{E}_{a\sim\pi}[\hat{Q}(s,a)] \le V^\pi(s)$, instead of constraining the policy. No behaviour-policy model needed. ^conservative-q

Practical payoff: ~20 lines on top of [[Soft Actor-Critic]] or QR-DQN, no extra networks, and 2–5× returns on hard offline benchmarks.

## The Methodology

**The naive version (Equation 1).** Standard Bellman error plus a term that minimises Q-values under some distribution $\mu(a|s)$, with states drawn from the data:

$$\hat{Q}^{k+1} \leftarrow \arg\min_Q \; \alpha\, \mathbb{E}_{s\sim\mathcal{D},\, a\sim\mu}[Q(s,a)] + \tfrac{1}{2}\mathbb{E}_{s,a,s'\sim\mathcal{D}}\big[(Q - \hat{\mathcal{B}}^\pi \hat{Q}^k)^2\big]$$

States come from $\mathcal{D}$ only, because the Bellman backup never queries out-of-distribution *states* — only out-of-distribution *actions*. In the tabular case, setting the derivative to zero gives an exact update:

$$\hat{Q}^{k+1}(s,a) = \hat{\mathcal{B}}^\pi \hat{Q}^k(s,a) - \alpha \frac{\mu(a|s)}{\hat\pi_\beta(a|s)}$$

Every entry is pushed down. So $\hat{Q}^\pi \le Q^\pi$ pointwise (Theorem 3.1). Correct, but crude.

**The tighter version (Equation 2).** Add a *maximisation* term under the data's own action distribution $\hat\pi_\beta$:

$$\arg\min_Q \; \alpha\Big(\mathbb{E}_{s\sim\mathcal{D},a\sim\mu}[Q] - \mathbb{E}_{s\sim\mathcal{D},a\sim\hat\pi_\beta}[Q]\Big) + \tfrac{1}{2}\mathbb{E}_\mathcal{D}\big[(Q - \hat{\mathcal{B}}^\pi\hat{Q}^k)^2\big]$$

Now the tabular update subtracts $\alpha\big(\tfrac{\mu(a|s)}{\pi_\beta(a|s)} - 1\big)$, which is *negative* wherever $\mu < \pi_\beta$ — so some Q-values go **up**. No pointwise bound survives. But when $\mu = \pi$, the expected value still lower-bounds, because the penalty on the value is

$$D_{\text{CQL}}(s) = \sum_a \pi(a|s)\Big(\tfrac{\pi(a|s)}{\pi_\beta(a|s)} - 1\Big) = \sum_a \frac{(\pi(a|s)-\pi_\beta(a|s))^2}{\pi_\beta(a|s)} \ge 0$$

a chi-square-like divergence, zero only when $\pi = \pi_\beta$. That algebraic identity is the heart of Theorem 3.2.

**Making it an algorithm (Equation 3).** We do not want to run full policy evaluation for each policy iterate. Instead make $\mu$ itself the maximiser of the current Q-function, giving a min-max problem with a regulariser $\mathcal{R}(\mu)$:

$$\min_Q \max_\mu \; \alpha\big(\mathbb{E}_{\mu}[Q] - \mathbb{E}_{\hat\pi_\beta}[Q]\big) + \tfrac12\mathbb{E}_\mathcal{D}[(Q - \hat{\mathcal{B}}^{\pi_k}\hat{Q}^k)^2] + \mathcal{R}(\mu)$$

**CQL($\mathcal{H}$)**: set $\mathcal{R}$ = entropy of $\mu$. The inner max solves in closed form to $\mu \propto \exp(Q)$, and plugging back gives a log-sum-exp:

$$\min_Q \; \alpha\, \mathbb{E}_{s\sim\mathcal{D}}\Big[\log\textstyle\sum_a \exp Q(s,a) - \mathbb{E}_{a\sim\hat\pi_\beta}[Q(s,a)]\Big] + \tfrac12 \mathbb{E}_\mathcal{D}\big[(Q - \hat{\mathcal{B}}^{\pi_k}\hat{Q}^k)^2\big]$$

This is a soft-max minus the data mean. Beautifully simple: **push down the soft-maximum over all actions, push up the actions you actually saw.**

**CQL($\rho$)**: set $\mathcal{R} = -D_{\mathrm{KL}}(\mu \Vert \rho)$ with $\rho = \hat\pi^{k-1}$, the previous policy. The log-sum-exp is replaced by an exponentially weighted average over samples from the old policy. Needed when the action space is big and the log-sum-exp estimate is too noisy.

**Two theoretical properties beyond the bound.**

*Gap-expanding* (Theorem 3.4): the learned gap between in-distribution and out-of-distribution Q-values is **larger** than the true gap. So the greedy policy is implicitly pulled back toward the data. This is what stops OOD actions without any explicit policy constraint.

*Safe policy improvement* (Theorems 3.5–3.6): CQL is exactly maximising $J(\pi, \hat M) - \tfrac{\alpha}{1-\gamma}\mathbb{E}[D_{\text{CQL}}(\pi,\hat\pi_\beta)]$ in the empirical MDP $\hat M$, and this gives $J(\pi^*, M) \ge J(\hat\pi_\beta, M) - \zeta$ with high probability. $\zeta$ shrinks as $1/\sqrt{|\mathcal{D}(s)|}$.

**Implementation.**
- Continuous: on top of SAC. Twin Q-functions, soft target updates, all SAC defaults kept.
- Discrete: on top of QR-DQN.
- Policy LR = **3e-5** (Q-function LR 3e-4). Deliberately slow — Theorem 3.3 requires $D_{\mathrm{TV}}(\hat\pi^{k+1}, \pi_{\hat Q^k}) \le \varepsilon$ small, i.e. the policy must not race ahead of the critic.
- $\log\sum_a \exp Q$ in continuous spaces is estimated with 10 importance samples from $\text{Unif}(a)$ and 10 from $\pi$.
- $\alpha$ tuned by **Lagrangian dual descent** against a budget $\tau$: if the measured Q-gap is below $\tau$, $\alpha \to 0$; if above, $\alpha$ grows. $\tau = 10.0$ for gym MuJoCo, $\tau = 5.0$ for Adroit and Kitchen. Atari used fixed $\alpha \in \{0.5, 1.0, 4.0\}$.
- 1M gradient steps, 4 seeds.

## Ablation Studies and Experiments

**D4RL gym MuJoCo (Table 1, normalised return).** On single-policy datasets CQL roughly ties the best baseline. On *mixed* datasets it wins big:

| Task | BC | BEAR | BRAC-v | CQL($\mathcal{H}$) |
|---|---|---|---|---|
| hopper-medium | 29.0 | 47.6 | 32.3 | **58.0** |
| walker2d-medium-expert | 11.3 | 10.8 | 0.9 | **98.7** |
| hopper-mixed | 11.8 | 25.3 | 0.8 | **48.6** |
| walker2d-random-expert | 0.7 | 1.9 | 2.7 | **91.1** |
| hopper-random-expert | 10.1 | 10.1 | 11.1 | **110.5** |
| halfcheetah-random-expert | 1.3 | 24.6 | 2.2 | **92.5** |

The pattern is the whole story: **policy-constraint methods collapse on multi-modal data.** They try to stay close to a behaviour policy that is a mixture of a random policy and an expert — so "close to the data" means "average of good and terrible". CQL has no such object to fit.

**AntMaze (Table 2).** Requires stitching sub-optimal trajectories. On medium and large mazes every baseline scores **exactly 0.0**. CQL($\mathcal{H}$) gets 61.2 / 53.7 / 15.8 / 14.9. This is the most striking result in the paper.

**Adroit, 24-DoF hand, human demonstrations.** BC is the strongest baseline; every prior offline RL method loses to it. CQL is the only method that beats BC: pen-human 34.4 (BC) → 55.8 (CQL($\rho$)); door-human 0.5 → 9.9.

**Franka Kitchen.** CQL 43.8 / 49.8 / 51.0 vs BC 33.8 / 33.8 / 47.5. BEAR and BRAC are near zero on two of three.

**Atari, image inputs.** With only 1% of a DQN replay buffer: Q*bert 383.6 (QR-DQN) → **14012.0** (CQL), a 36× gap. Breakout 7.9 → 61.1. With 10% data the gap narrows a lot and CQL actually *loses* on Asterix (156.3 vs 189.2 for QR-DQN). Conservatism pays most when data is scarce.

**Does it actually lower-bound? (Table 4)** They measure predicted value minus true return.

| Task | CQL($\mathcal{H}$) | CQL (Eq. 1) | Ensemble(2) | Ensemble(20) | BEAR |
|---|---|---|---|---|---|
| hopper-medium-expert | **−43.2** | −151.4 | 3.71e6 | 2.41e4 | 65.9 |
| hopper-medium | **−7.5** | −156.7 | 2.60e13 | 8.85e5 | 4.3 |

Three things at once. (a) CQL is the only method with a negative gap — it genuinely lower-bounds. (b) Ensemble-minimum, the standard trick for taming overestimation, fails *catastrophically* offline — errors of $10^{13}$. Even 20 ensemble members do not help. (c) Equation 1 (pointwise bound) underestimates by ~20× more than Equation 2. The maximisation term is doing real work.

**What did not work / what the ablations reveal:**

- **Dropping the data-maximisation term** (Appendix G, Table 6): hopper-mixed 1563 → 865, hopper-medium 1866 → 1028. The tighter bound is not a theoretical nicety.
- **Other choices of $\nu$ for the maximisation term** (Theorem D.3): they prove that $\nu = \pi_\beta$ is the *only* distribution for which the lower bound holds for every possible $\pi$. Solving the concave-convex max-min gives $\nu^* = \pi_\beta$ with optimal penalty value exactly 0; anything else admits a $\pi$ that makes the penalty negative. So the data distribution is necessary, not just convenient.
- **$\tau = 2.0$** blew up: $\alpha$ shot into the millions and Q-values crashed to $-10^6$. $\tau = 10.0$ on human-demo data failed the other way, letting Q-values diverge past $+10^6$. The budget is the sensitive knob.
- **CQL($\mathcal{H}$) vs CQL($\rho$)** (Table 5): $\mathcal{H}$ wins on MuJoCo (halfcheetah-medium-expert 7234 vs 3996). $\rho$ wins on high-dimensional Adroit, where log-sum-exp importance weights have too much variance. $\rho$ also trained more stably with no sudden performance drops.
- **Fixed $\alpha=5$ vs Lagrange** (Table 7): comparable on MuJoCo, but on AntMaze the Lagrange version is 0.53 vs 0.21 (medium-diverse) and 0.15 vs 0.02 (large-play).
- **Direct measurement of gap-expansion** (Appendix B): they plot $\hat\Delta^k = \mathbb{E}_\mathcal{D}[\max_{a' \sim \text{Unif}} \hat Q(s,a') - \hat Q(s,a)]$. For CQL this is negative. For BEAR it is positive **and grows during training**, even on hopper-*expert* — and BEAR then "unlearns", with returns collapsing late in training. Policy constraints stop the backup from querying bad actions but do nothing to stop function-approximation coupling from putting high values there.

## Worth Remembering

- **The failure mode is asymmetric.** Overestimation of an OOD action is self-reinforcing (policy → that action → higher target → higher value). Underestimation is self-correcting. CQL exploits this asymmetry; ensembles do not.
- **The penalty is a divergence in disguise.** $D_{\text{CQL}} = \sum_a (\pi - \pi_\beta)^2/\pi_\beta$ is $\chi^2$-like. So CQL *is* a behaviour-regularised method — it just never builds an explicit model of $\pi_\beta$, which is exactly why it survives multi-modal data.
- **Required $\alpha$ shrinks like $1/\sqrt{|\mathcal{D}(s,a)|}$.** With infinite data any $\alpha > 0$ suffices; with the exact Bellman operator $\alpha = 0$ works. Conservatism is a finite-sample correction, not a permanent tax.
- **Admitted limitations.** No lower-bound proof for deep nets — only tabular, linear, and NTK-linearised regimes (Theorems D.1, D.2), and the NTK case assumes a *single* gradient step per iteration, not full minimisation. And there is no early-stopping criterion: offline RL has no validation error, so "how many gradient steps" remains guesswork. They just used 1M everywhere.
- **The slow policy LR is not a detail.** Theorem 3.3's condition is that underestimation (a) must exceed the drift term (b) bounded by $D_{\mathrm{TV}}(\pi_{\hat Q^k}, \hat\pi^{k+1}) \cdot \max_a \tfrac{\pi_{\hat Q^k}}{\hat\pi_\beta}$. Move the policy fast and the lower bound is no longer guaranteed.
- **Practical caveat:** the log-sum-exp estimator degrades in high dimensions. If your action space is large, use CQL($\rho$) or the approximate max-backup variant (sample 10 actions from $\pi$ at $s'$, take the max instead of the expectation) which they note helps on Kitchen and AntMaze.
- Connects naturally to [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]], where CQL is "Route 4 — conservative Q-learning". The uncertainty-based routes are exactly what Table 4 shows failing.
- A follow-up question worth chasing: CQL underestimates by a *fixed* amount everywhere in the support. A count-dependent $\alpha$ (Appendix E sketches $\alpha/n(s,a)$) would make it uncertainty-aware, and would recover the $1/\sqrt{n}$ shape of optimism-based bounds, flipped in sign.

## Links

Related: [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] · [[Soft Actor-Critic]] · [[Playing Atari with Deep Reinforcement Learning (DQN)]] · [[Markov Decision Process]] · [[A Distributional Perspective on Reinforcement Learning]] · [[Continuous control with deep reinforcement learning (DDPG)]] · [[KL Divergence]] · [[Uncertainty]] · [[Proximal Policy Optimization Algorithms]] · [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)]]

New topics worth writing: BEAR / bootstrapping error reduction, BCQ, D4RL benchmark design, Neural Tangent Kernel, safe policy improvement and SPIBB, chi-square divergence, distributionally robust optimization, QR-DQN, Lagrangian dual gradient descent for constrained losses, IQL and expectile regression as the successor to CQL
