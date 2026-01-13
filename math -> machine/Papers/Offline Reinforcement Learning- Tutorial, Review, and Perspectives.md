---
title: "Offline Reinforcement Learning: Tutorial, Review, and Perspectives"
authors: ["Sergey Levine", "Aviral Kumar", "George Tucker", "Justin Fu"]
year: 2020
arxiv: "2005.01643"
url: https://arxiv.org/abs/2005.01643
priority: Must-Read
read_on: 2026-08-25
tags: [paper, llm, rl, theory]
---
## The Core Idea

Reinforcement learning is normally an *active* loop: try something, see what happens, update, try again. That loop is what blocks RL from most real problems — you cannot let a half-trained policy prescribe drugs, drive a car, or talk to customers. **Offline RL** removes the loop entirely. You are handed a fixed pile of logged transitions $\mathcal{D} = \{(\mathbf{s}_t^i, \mathbf{a}_t^i, \mathbf{s}_{t+1}^i, r_t^i)\}$, collected by some unknown *behaviour policy* $\pi_\beta$, and you must produce the best policy you can. No new data. Ever. Then you deploy.

The promise is exactly the promise that made supervised learning work: turn a big dataset into a powerful artefact. Instead of a pattern recogniser, you get a **decision-making engine**.

The reason this did not already exist is subtle, and it is the whole point of the tutorial. Standard "off-policy" algorithms (DQN, actor-critic with a replay buffer) *look* like they should work — they already learn from data collected by old policies. But they always kept collecting a trickle of fresh data, and that trickle was quietly doing enormous work: it corrected the algorithm's own delusions. Take it away and they break.

The failure has a name: **distributional shift**, and it bites in a specific place. The Bellman target is

$$Q^\pi(\mathbf{s}_t,\mathbf{a}_t) = r(\mathbf{s}_t,\mathbf{a}_t) + \gamma \, \mathbb{E}_{\mathbf{s}_{t+1}\sim T,\ \mathbf{a}_{t+1}\sim \pi}\big[Q^\pi(\mathbf{s}_{t+1},\mathbf{a}_{t+1})\big].$$

The next action $\mathbf{a}_{t+1}$ comes from the *learned* policy $\pi$, not from the data. So the Q-function is asked "how good is this action?" about actions it has never seen. Neural networks answer such questions confidently and wrongly. And then the policy is explicitly optimised to $\max_\pi \mathbb{E}_{\mathbf{a}\sim\pi}[Q(\mathbf{s},\mathbf{a})]$ — it *hunts* for the actions where the Q-function is most wrongly optimistic. Online, the agent tries the action, gets a bad reward, and the error is deleted. Offline, the error is copied into the next target, and the next, and the Q-values blow up.

> [!NOTE] Action distribution shift
> The Q-function is only ever *trained* on states and actions from $d^{\pi_\beta}$. During training it is only ever *evaluated* on states from $d^{\pi_\beta}$ — but on actions from $\pi$. States are fine; **actions are the leak**. Fixing offline RL is mostly about plugging that one hole. ^action-distribution-shift

Framed more deeply: offline RL is a counterfactual question — "what would have happened had we acted differently?" — which sits outside the i.i.d. assumption that all of supervised learning rests on. That connects it directly to [[Counterfactual Reasoning and Learning Systems]].

## The Methodology

The paper is a review, so the "method" is a taxonomy. Here is the map, with the actual maths.

**Setup.** An [[Markov Decision Process]] $\mathcal{M}=(\mathcal{S},\mathcal{A},T,d_0,r,\gamma)$, objective $J(\pi)=\mathbb{E}_{\tau\sim p_\pi}[\sum_t \gamma^t r(\mathbf{s}_t,\mathbf{a}_t)]$. $d^{\pi}(\mathbf{s})$ is the state visitation frequency. Everything rests on the [[Markov Property]].

**A warning shot from imitation learning.** Even in the easiest imaginable offline case — you are *given* the optimal action label $\mathbf{a}^\star$ at every state in $\mathcal{D}$ — plain supervised behavioural cloning with generalisation error $\epsilon$ gives a mistake bound of $\ell(\pi)\le C + H^2\epsilon$, quadratic in the horizon $H$. If you are allowed to collect on-policy states and label those (DAgger), the bound is $C + H\epsilon$, linear. The extra factor of $H$ is purely the cost of drifting off-distribution: one mistake takes you somewhere unfamiliar, and unfamiliar means all further guarantees are void for the rest of the episode.

### Route 1 — importance sampling

Estimate $J(\pi_\theta)$ from $\pi_\beta$'s trajectories by reweighting:

$$J(\pi_\theta) = \mathbb{E}_{\tau\sim\pi_\beta}\Big[\Big(\prod_{t=0}^{H}\tfrac{\pi_\theta(\mathbf{a}_t|\mathbf{s}_t)}{\pi_\beta(\mathbf{a}_t|\mathbf{s}_t)}\Big)\sum_t \gamma^t r_t\Big].$$

That product of $H$ ratios is a variance bomb — it grows exponentially in horizon. Fixes, in order of increasing sophistication:

- **Self-normalising** the weights (weighted IS): biased, much lower variance, still consistent.
- **Per-decision IS**: drop the future weights, since $r_t$ cannot depend on actions after $t$.
- **Doubly robust**: use a model-based $\hat{Q}^\pi$ as a control variate; unbiased if *either* $\pi_\beta$ is known *or* the model is right. Same trick as [[Doubly Robust Policy Evaluation and Learning]].
- **Marginalised IS**: skip per-action ratios entirely and estimate the state-marginal ratio $\rho^\pi(\mathbf{s}) = d^{\pi}(\mathbf{s})/d^{\pi_\beta}(\mathbf{s})$ directly. This satisfies a *forward* Bellman equation and can be fit by TD updates, adversarial saddle-point objectives ($\min_\rho \max_f L(\rho,f)^2$), or convex-duality tricks (DualDICE / AlgaeDICE), which flip it into a *backward* Bellman equation on a value-like function $\nu$ and recover $\rho^\pi(\mathbf{s},\mathbf{a}) = \nu^*(\mathbf{s},\mathbf{a}) - \tilde{\mathcal{B}}^\pi\nu^*(\mathbf{s},\mathbf{a})$.

There is also the cheap approximation: just use the dataset's state distribution and pretend it is the policy's. The **off-policy policy gradient**

$$\nabla_\theta J_{\pi_\beta}(\pi_\theta) \approx \mathbb{E}_{\mathbf{s}\sim d^{\pi_\beta},\ \mathbf{a}\sim\pi_\theta}\big[Q^{\pi_\theta}(\mathbf{s},\mathbf{a})\nabla_\theta\log\pi_\theta(\mathbf{a}|\mathbf{s})\big]$$

is biased but empirically tolerable, and underlies DDPG-style algorithms.

### Route 2 — dynamic programming with a policy constraint

This is where most modern offline RL lives. Alternate the two steps of policy iteration, but bolt a constraint onto the actor:

$$\hat{Q}^\pi_{k+1} \leftarrow \arg\min_Q \mathbb{E}_{\mathcal{D}}\Big[\big(Q(\mathbf{s},\mathbf{a}) - (r + \gamma\,\mathbb{E}_{\mathbf{a}'\sim\pi_k}[\hat{Q}^\pi_k(\mathbf{s}',\mathbf{a}')])\big)^2\Big]$$
$$\pi_{k+1} \leftarrow \arg\max_\pi \mathbb{E}_{\mathbf{s}\sim\mathcal{D}}\big[\mathbb{E}_{\mathbf{a}\sim\pi}[\hat{Q}^\pi_{k+1}(\mathbf{s},\mathbf{a})]\big]\quad \text{s.t. } D(\pi,\pi_\beta)\le\epsilon.$$

The **policy penalty** variant instead folds the divergence into the reward, $\bar{r}(\mathbf{s},\mathbf{a}) = r(\mathbf{s},\mathbf{a}) - \alpha D(\pi(\cdot|\mathbf{s}),\pi_\beta(\cdot|\mathbf{s}))$, so the policy also avoids *going somewhere* it would later be forced to deviate.

Choices of $D$:

- **Explicit $f$-divergence**, $D_f = \mathbb{E}_{\mathbf{a}\sim\pi}[f(\pi(\mathbf{a}|\mathbf{s})/\pi_\beta(\mathbf{a}|\mathbf{s}))]$. With $D_{\mathrm{KL}}$ this is the control-as-inference / max-entropy formulation: add $\alpha\log\pi_\beta(\mathbf{a}|\mathbf{s})$ to the reward and let the entropy bonus absorb $-\alpha\log\pi$. See [[KL Divergence]]. Downside: you must *fit* $\pi_\beta$ by behavioural cloning first.
- **Implicit KL** (AWR, AWAC, ABM). Solve for the constrained optimum in closed form and then project it back into your policy class:
  $$\bar\pi_{k+1}(\mathbf{a}|\mathbf{s}) \propto \pi_\beta(\mathbf{a}|\mathbf{s})\exp\!\big(\tfrac{1}{\alpha}Q^\pi_k(\mathbf{s},\mathbf{a})\big),\qquad \pi_{k+1}\leftarrow\arg\min_\pi D_{\mathrm{KL}}(\bar\pi_{k+1},\pi).$$
  In practice: weight the dataset actions by $\exp(Q/\alpha)$ and do weighted supervised regression. **No explicit behaviour model needed** — the weights come from samples you already have. $\alpha$ is the Lagrange multiplier.
- **Integral probability metrics**, $D_\Phi = \sup_{\phi\in\Phi}|\mathbb{E}_{\pi}[\phi] - \mathbb{E}_{\pi_\beta}[\phi]|$. Unit-ball-in-an-RKHS gives MMD (used by BEAR); unit-Lipschitz gives Wasserstein-1.

### Route 3 — uncertainty estimation

Learn a distribution over Q-functions $\mathcal{P}_\mathcal{D}(Q^\pi)$ (bootstrap ensemble, or a parametric Gaussian) and improve the policy against a *pessimistic* estimate:

$$\pi_{k+1}\leftarrow\arg\max_\pi \mathbb{E}_{\mathbf{s}\sim\mathcal{D}}\Big[\mathbb{E}_{\mathbf{a}\sim\pi}\big[\mathbb{E}_{Q\sim\mathcal{P}_\mathcal{D}}[Q(\mathbf{s},\mathbf{a})] - \alpha\,\text{Unc}(\mathcal{P}_\mathcal{D}(Q^\pi))\big]\Big].$$

This is exploration's optimism, run backwards. See [[Uncertainty]].

### Route 4 — conservative Q-learning

Skip the policy model and the ensemble; regularise the Q-function directly. Add $\alpha\,\mathcal{C}(B,\phi)$ to the Bellman error. The base version pushes *down* on Q-values under an adversarially chosen $\mu$:

$$\mathcal{C}_{\text{CQL}_0}(B,\phi) = \mathbb{E}_{\mathbf{s}\sim B,\ \mathbf{a}\sim\mu}[Q_\phi(\mathbf{s},\mathbf{a})],\qquad \mu = \arg\max_\mu \mathbb{E}[\mathbb{E}_\mu[Q_\phi] + \mathcal{H}(\mu)].$$

The solution is $\mu(\mathbf{a}|\mathbf{s})\propto\exp(Q(\mathbf{s},\mathbf{a}))$, so the penalty becomes $\mathbb{E}_{\mathbf{s}\sim B}[\log\sum_\mathbf{a}\exp Q_\phi(\mathbf{s},\mathbf{a})]$ — a log-sum-exp dominated by whichever action currently has the highest Q-value. The Bellman error anchors the in-distribution actions, so the penalty mostly crushes the out-of-distribution spikes. For a suitable $\alpha$ this provably gives a **pointwise lower bound** on the true $Q$.

The practical version adds a maximisation term for actions actually in the batch:

$$\mathcal{C}_{\text{CQL}_1}(B,\phi) = \mathbb{E}_{\mathbf{s}\sim B,\mathbf{a}\sim\mu}[Q_\phi(\mathbf{s},\mathbf{a})] - \mathbb{E}_{(\mathbf{s},\mathbf{a})\sim B}[Q_\phi(\mathbf{s},\mathbf{a})].$$

This is zero on average when $\mu=\pi_\beta$. It only lower-bounds in *expectation* under $\pi$, not pointwise — but it stops the excessive pessimism.

### Route 5 — model-based

Fit $T_\psi(\mathbf{s}_{t+1}|\mathbf{s}_t,\mathbf{a}_t)$ by supervised learning, then plan or train a policy inside it. The failure mode is **model exploitation**: the planner finds the states where the model hallucinates high reward. The bound (from MBPO), with model TV-error $\epsilon_m$ and policy TV-divergence $\epsilon_\pi$:

$$J(\pi) \ge J_\psi(\pi) - \left[\frac{2\gamma r_{\max}(\epsilon_m + 2\epsilon_\pi)}{(1-\gamma)^2} + \frac{4 r_{\max}\epsilon_\pi}{1-\gamma}\right].$$

Second term: policy drift. First term: model error, which itself scales with $\epsilon_\pi$ because a more divergent policy pushes the model further from its training data. MOReL and MOPO make this conservative by penalising the reward, $\tilde{r} = r - \lambda u(\mathbf{s},\mathbf{a})$, where $u$ is an *error oracle* upper-bounding the model's divergence from truth; MOReL instead sends the agent to a low-reward absorbing state when $u$ crosses a threshold.

## Ablation Studies and Experiments

This is a survey, so the evidence is assembled from the cited literature. The load-bearing results:

**The unlearning curve (Kumar et al. 2019, Figure 2).** SAC run purely offline on HalfCheetah-v2. Return rises, then collapses. Learned Q-values, plotted on a log scale, explode. Crucially: **increasing the dataset size $n$ does not fix it.** The curve *looks* like overfitting but is not — it is target-value contamination compounding through the Bellman backup. This single figure is the paper's best argument that offline RL is a distinct problem and not "supervised learning with more data".

**Distribution constraints fail where support constraints succeed (Figure 3, 1D lineworld).** The task: get from S to G. The behaviour policy strongly prefers *left* at every state. A distribution constraint (KL) forces the learned policy to keep resembling that leftward density, so it never reliably reaches the goal. A support constraint — which only forbids actions $\pi_\beta$ assigns near-zero mass to, and is indifferent to the actual densities — recovers the optimal policy.

The clean thought experiment for the same point: if $\pi_\beta$ is **uniformly random**, offline RL should be trivially easy — there are no out-of-distribution actions at all, and even unconstrained actor-critic works fine. But a KL constraint would forbid the learned policy from becoming concentrated, forcing it to stay near-uniform and therefore near-useless. **KL is the wrong constraint in principle.** MMD is preferred not because it is a support metric asymptotically (it is not — it also converges to a distribution divergence) but because in the *finite-sample* regime it is computed kernel-wise from samples, ignores densities, and empirically behaves like a support constraint.

**$\text{CQL}_0$ vs $\text{CQL}_1$.** The pure lower-bound penalty is provably conservative and empirically *too* conservative — it underestimates so heavily that performance suffers. Adding the in-batch maximisation term ($\text{CQL}_1$) sacrifices the pointwise guarantee for an expectation-level guarantee and wins in practice. This is the paper's clearest "the theoretically cleanest version is not the one that works".

**Uncertainty methods underperform policy constraints and conservative value functions.** Surprising, because uncertainty estimation is the workhorse of exploration ([[A Tutorial on Thompson Sampling]], bootstrapped DQN). The explanation is a bar-height argument: exploration only needs the uncertainty set to *contain* good behaviour, and can tolerate it containing lots of junk too. Offline RL needs the set to genuinely quantify how much you can trust the Q-function. That is a far stricter requirement, and modern deep nets do not deliver it. Too tight → over-conservatism, no learning. Too loose → the policy exploits out-of-distribution actions again.

**Model-based methods work offline with almost no modification.** Yu et al. (2020) report that plain MBPO gets reasonable performance on standard offline benchmarks while naïve SAC learns nothing. Model-based methods already carry uncertainty machinery (ensembles, Bayesian NNs) to prevent model exploitation, and that machinery happens to be exactly what offline RL needs.

**What does not work / what is admitted broken:**
- Naïve importance sampling in the fully offline setting. The $\prod_t$ variance blowup means IS is only viable when $\pi$ stays *very* close to $\pi_\beta$ — which is fine in classic off-policy RL, where fresh data keeps arriving, and useless offline.
- Marginalised IS avoids the product, but still needs dynamic programming to estimate $\rho^\pi$, which reintroduces out-of-distribution action queries. You have moved the problem, not solved it.
- Fitting $\pi_\beta$ with a unimodal Gaussian when the true behaviour is multi-modal. The fitted model averages the modes, so constraining against it does *not* prevent out-of-distribution actions — it may actively point at the gap between modes.
- Evaluating offline RL on the full replay buffer of a converged online run. Nearly every early paper did this. It is a bad benchmark, because the point of offline RL is to beat the best behaviour in the data, and here the data already contains near-optimal behaviour.

**Benchmarking.** The D4RL suite exists to test the actual capability of interest: **compositional stitching**. If the data shows how to get from state 1 to state 2, and separately from 2 to 3, a good offline RL method should discover 1→3, which may score far higher than any single trajectory in the dataset. Maze2D is the explicit test. This is the honest answer to "can offline RL exceed the data?" — it cannot invent better *actions*, but it can recombine sub-sequences, and with function approximation possibly do so on a *subset* of state variables.

## Worth Remembering

- **The horizon penalty is not fixable.** Kidambi et al. (2020) prove a lower bound: quadratic scaling in the horizon, $1/(1-\gamma)^2 \approx H^2$, is unavoidable in the worst case for *any* offline RL method. This is not a weakness of current algorithms.
- **The tension is unresolved and central.** Strong constraints control error accumulation but cap how much better than $\pi_\beta$ you can get. Weak constraints let errors compound. The paper flags "dynamically modulating the degree of conservatism" as an open problem, and names *excessive pessimism* as possibly the main thing holding back benchmark numbers.
- **State distribution still leaks in, silently.** Even with a perfect action constraint, the Q-function is fit under $d^{\pi_\beta}(\mathbf{s})$. Function approximation ties together states with very different densities, so a rare state can inherit a wrong value from a common one. Online RL self-corrects this ("corrective feedback"); offline RL has no mechanism at all.
- **Support ≠ distribution.** Internalise this. An effective constraint should stop the policy leaving the set of actions that have mass in the data, while letting it concentrate freely on a *subset* of them. Most off-the-shelf divergences do the wrong thing.
- **QT-Opt caveat.** 500,000 logged robotic grasps trained a working vision-based grasping policy offline — but online fine-tuning still improved it *considerably*. Practically: treat offline RL as a pre-training step where you can, not a replacement for interaction.
- **Open theory question:** is model-based offline RL better than model-free dynamic programming *even in principle*? Under linear function approximation, model-based updates and fitted value iteration produce identical iterates. Nobody knows whether that equivalence survives non-linear function approximation. Both are prediction problems — one predicts returns, one predicts states.
- **Healthcare data has a nasty structural bias.** Minor cases rarely get treatment, so a naïve agent can conclude that *any* drug causes death, because drugs are only given to people who are already dying. A textbook confounding failure dressed as an RL problem.
- **Recommender/advertising applications almost always collapse to contextual bandits** (see [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]]), dropping the sequential structure. Cheap and usually fine, but wrong whenever actions have temporal dependence.
- **The closing argument is a [[The Bitter Lesson (essay)]] argument.** Vision and NLP progressed as much through datasets as through algorithms. Online RL structurally cannot benefit from that, because it insists on generating its own data. Offline RL is the bridge — and, the authors suggest, the reason sim-to-real transfer is so popular in robotics is precisely that offline RL does not yet work well enough.

## Links

Related: [[Markov Decision Process]] · [[Markov Property]] · [[Playing Atari with Deep Reinforcement Learning (DQN)]] · [[Proximal Policy Optimization Algorithms]] · [[Doubly Robust Policy Evaluation and Learning]] · [[Counterfactual Reasoning and Learning Systems]] · [[Unbiased Learning-to-Rank with Biased Feedback]] · [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]] · [[A Tutorial on Thompson Sampling]] · [[KL Divergence]] · [[Uncertainty]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)]] · [[A Distributional Perspective on Reinforcement Learning]] · [[Generative Adversarial Networks]] · [[Regularization]] · [[Decision Sciences]] · [[The Bitter Lesson (essay)]]

New topics worth writing: Behavioural cloning and DAgger, Conservative Q-Learning (CQL), BEAR and MMD support constraints, Advantage-Weighted Regression / AWAC, D4RL benchmark suite, f-divergences and their variational dual form, Integral probability metrics (MMD, Wasserstein-1), Marginalised importance sampling and DualDICE, MOPO and MOReL, Model-Based Policy Optimization (MBPO), Least-Squares Policy Iteration (LSPI), Soft Actor-Critic, Bootstrap ensembles for epistemic uncertainty, Control as inference / linearly-solvable MDPs, Off-policy evaluation, Deadly triad in RL
