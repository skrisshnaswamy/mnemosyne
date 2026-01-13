---
title: "A Distributional Perspective on Reinforcement Learning"
authors: ["Marc G. Bellemare", "Will Dabney", "Rémi Munos"]
year: 2017
arxiv: "1707.06887"
url: https://arxiv.org/abs/1707.06887
priority: Must-Read
read_on: 2026-08-23
tags: [paper, llm, rl]
---
## The Core Idea

Standard reinforcement learning learns one number per state-action pair: the **expected** return $Q(x,a)$. This paper says: learn the whole **distribution** of the return instead. Call it $Z(x,a)$ — a random variable whose mean is $Q(x,a)$.

The reason this is possible at all is that Bellman's equation still works when you strip out the expectations:

$$Z(x,a) \overset{D}{=} R(x,a) + \gamma\, Z(X', A')$$

where $\overset{D}{=}$ means "has the same distribution as". Three random things combine: the reward $R$, the random next state-action $(X',A')$, and the return distribution at that next pair. You shrink the next distribution by $\gamma$, shift it by the reward, and mix over transitions. That is the whole recursion.

> [!NOTE] Value distribution ^value-distribution
> The full probability distribution of the discounted sum of future rewards from $(x,a)$, not just its mean. It captures *intrinsic* randomness of the environment and policy — not uncertainty about what the model doesn't know.

Why this is not obvious, and why it had not been done properly before:

1. Distributional Bellman equations are old (Jaquette 1973, Sobel 1982), but people used them only for niche goals — risk-sensitive control, variance penalties, Bayesian [[Uncertainty|uncertainty]] modelling. Nobody used them as the *main* learning target.
2. Earlier attempts approximated $Z$ with a Gaussian. That throws away the multimodality, which is exactly the useful part.
3. There is a theory problem: the distributional operator is only well-behaved for a **fixed** policy. In control (the greedy case) it is provably *not* a contraction in any distributional metric.

The surprise: even though the agent still acts greedily on the **mean** ($\epsilon$-greedy over $\mathbb{E}Z$), learning the distribution makes learning dramatically better. Nothing about the behaviour policy changed. The gain is entirely from the learning signal being richer and the loss being nicer. On 57 Atari games, C51 hits **mean 701% / median 178%** of human, beating Prioritized Dueling DQN (592% / 172%), Dueling (373% / 151%), Double DQN (307% / 118%), and plain [[Playing Atari with Deep Reinforcement Learning (DQN)|DQN]] (228% / 79%).

## The Methodology

**The theory half.**

Define the metric that makes everything work. For two CDFs $F, G$, the **Wasserstein-$p$** distance is

$$d_p(F,G) = \left(\int_0^1 |F^{-1}(u) - G^{-1}(u)|^p\, du\right)^{1/p}$$

> [!NOTE] Wasserstein metric ^wasserstein
> The cost of moving one pile of probability mass onto another, measured along the value axis. Unlike [[KL Divergence]] it cares about *how far apart* the outcomes are, and it stays finite when the two distributions have disjoint support.

Take the sup over state-actions: $\bar d_p(Z_1, Z_2) := \sup_{x,a} d_p(Z_1(x,a), Z_2(x,a))$.

*Lemma 3 (policy evaluation):* $\mathcal{T}^\pi Z(x,a) \overset{D}{:=} R(x,a) + \gamma P^\pi Z(x,a)$ is a $\gamma$-contraction in $\bar d_p$. Proof is three lines using the properties $d_p(aU,aV) \le |a| d_p(U,V)$ and $d_p(A+U, A+V) \le d_p(U,V)$. So by Banach fixed point, repeated application converges to $Z^\pi$, and *all moments* converge exponentially fast.

The metric choice matters: the same operator is **not** a contraction in total variation, KL divergence, or Kolmogorov distance. Variance is a $\gamma^2$-contraction (Sobel); higher centred moments are not contractions but still converge exponentially.

*Control is a mess.* Define $\mathcal{T}Z = \mathcal{T}^\pi Z$ for any $\pi$ greedy w.r.t. $\mathbb{E}Z$. Then:
- Lemma 4: the *mean* $\mathbb{E}Z_k \to Q^*$ at rate $\gamma^k$, as usual.
- Proposition 1: $\mathcal{T}$ is **not a contraction**. Two-state counterexample: from $x_2$, action $a_1$ gives $0$, action $a_2$ gives $\epsilon \pm 1$. Start with $Z$ that is wrong only at $(x_2,a_2)$, off by $2\epsilon$. Greedy selection flips to $a_1$, and after one step the distance is $\approx 1 \gg 2\epsilon$. Distance grew.
- Proposition 2: some optimality operators have no fixed point at all (tie-breaking oscillates forever).
- Theorem 1: the best you get is pointwise convergence to $\mathcal{Z}^{**}$, the set of **nonstationary** optimal value distributions — distributions from *sequences* of optimal policies, which need not equal any single policy's distribution.

**The algorithm half — C51.**

Represent $Z$ as a categorical distribution on $N$ fixed **atoms** (support points), evenly spaced:

$$z_i = V_{\min} + i\,\Delta z, \quad \Delta z = \frac{V_{\max}-V_{\min}}{N-1}, \quad i = 0,\dots,N-1$$

The network outputs logits $\theta_i(x,a)$; probabilities are a softmax over the $N$ atoms, per action. $V_{\max} = -V_{\min} = 10$, $N=51$ (hence "C51").

The problem: after the Bellman update, $\hat{\mathcal{T}}z_j = r + \gamma z_j$ no longer lands on the atom grid. Supports are disjoint, so you cannot compare distributions directly. Minimising Wasserstein would be the principled fix — but you cannot do it from sampled transitions (Prop. 5: the expected sample Wasserstein loss is an *upper bound* on the true loss, strictly so, and its gradient is biased).

So instead: **project**. Shift each atom, clip to $[V_{\min}, V_{\max}]$, and split its probability mass linearly between the two nearest grid atoms:

$$(\Phi\hat{\mathcal{T}}Z_\theta(x,a))_i = \sum_{j=0}^{N-1}\left[1 - \frac{\big|[\hat{\mathcal{T}}z_j]^{V_{\max}}_{V_{\min}} - z_i\big|}{\Delta z}\right]_0^1 p_j(x', \pi(x'))$$

Then the loss is plain **cross-entropy** against this projected target:

$$\mathcal{L}_{x,a}(\theta) = -\sum_i m_i \log p_i(x_t,a_t), \qquad m = \Phi\hat{\mathcal{T}}Z_{\tilde\theta}(x,a)$$

which is the cross-entropy term of $D_{\mathrm{KL}}(\Phi\hat{\mathcal{T}}Z_{\tilde\theta} \,\|\, Z_\theta)$. The Bellman update has become **multiclass classification** over 51 buckets. See [[Cross Entropy]] and [[KL Divergence]].

**Training loop.** Identical to DQN otherwise: same convnet, same replay buffer, same target network $\tilde\theta$, same $\alpha = 0.00025$. Differences: output layer is $|\mathcal{A}| \times 51$ instead of $|\mathcal{A}|$; squared TD error replaced by $\mathcal{L}_{x,a}$; **Adam instead of RMSProp**, with $\epsilon_{\text{adam}} = 0.01/32$ (tuned over $\{1, 0.1, 0.01, 0.001, 0.0001\}/L$). Greedy action is $\arg\max_a \sum_i z_i p_i(x,a)$ — the mean. Training $\epsilon = 0.01$, evaluation $\epsilon = 0.001$. Terminal transitions use $\gamma_t = 0$.

## Ablation Studies and Experiments

**Number of atoms** (5 training games, $\epsilon=0.05$, moving average over 5M frames):
- Too few atoms → poor behaviour. **More atoms always helped** — it did not saturate, which the authors did not expect given fixed network capacity.
- The 51-atom version beat DQN on **all five** training games; state-of-the-art on Seaquest.
- The degenerate $N=2$ **Bernoulli algorithm** (a single parameter, essentially just tracking the mean squashed into $[0,1]$) still beat DQN on 3 of 5 games and was notably more robust on Asterix. So even a tiny amount of distributional structure helps — but the bulk of the gain is at high $N$.

**Full 57-game table (best evaluation during training):**

| | Mean | Median | > human | > DQN |
|---|---|---|---|---|
| DQN | 228% | 79% | 24 | 0 |
| Double DQN | 307% | 118% | 33 | 43 |
| Dueling | 373% | 151% | 37 | 50 |
| Prioritized | 434% | 124% | 39 | 48 |
| Pr. Dueling | 592% | 172% | 39 | 44 |
| **C51** | **701%** | **178%** | **40** | **50** |

Standouts: Seaquest 266,434 (vs 50,254 for Dueling), Video Pinball 949,604, Breakout 748, Private Eye 15,095 (vs 206), Venture 1,520 (vs 497). The sparse-reward wins (Venture, Private Eye) suggest distributions **propagate rare events better** than a mean does — a rare big reward shows up as a mode instead of being diluted into an average.

Where C51 loses: Fishing Derby (8.9 vs 46.4), Star Gunner (49,095 vs 125,117), Up and Down (15,612 vs 44,939), Skiing, Ice Hockey. Montezuma's Revenge is still 0.0 — exploration is untouched by this.

**Robustness.** With ALE's sticky-action stochasticity ($p=0.25$ chance the agent's action is rejected), C51 still gains **+126% mean / +21.5% median** over DQN on a DQN-normalised scale.

**Speed.** Within 50M frames C51 beat a *fully trained* (200M frame) DQN on 45 of 57 games.

**What did not work:**
- **Directly minimising the Wasserstein loss by SGD.** This is the big negative result. In the CliffWalk experiment they compared supervised targets vs sampled one-step Bellman targets, under the Wasserstein loss and the categorical projection. The categorical algorithm minimised $d_1(Z^\pi, Z_\theta)$ well in *both* regimes. Sample-based Wasserstein converged to a fixed point that badly missed the true Monte Carlo distribution, and hit different values on repeat runs — local minima, worse with fewer atoms. Prop. 5 gives the reason: the expected sample loss upper-bounds the true loss strictly, so gradients are biased.
- **KL divergence between continuous densities.** Early experiments failed, "in part because the KL divergence is insensitive to the values of its outcomes" — KL does not know that atom 3 and atom 4 are close together, but the projection step does.
- **Gaussian approximations** (prior work) — the whole point is preserving multimodality, so a unimodal family defeats it.
- Interesting asymmetry: clipping the support to $[V_{\min}, V_{\max}]$ *helps* C51 (it makes all extreme returns equivalent, a form of [[Regularization|regularisation]] / inductive bias). Doing the analogous **value clipping in DQN significantly degrades performance in most games.**

## Worth Remembering

**Why does it work if you still act on the mean?** The authors are honest that they are speculating. Four proposed reasons:

1. **Reduced chattering.** The optimality operator is unstable (Prop. 1–3); combined with function approximation this makes greedy policies oscillate. The gradient-based categorical update effectively *averages* the competing distributions, folding the chattering into the approximate solution rather than letting it swing the policy.
2. **State aliasing.** Even in deterministic Atari, an agent that cannot see everything faces effective randomness. In Pong the exact reward timing depends on unobservable internal registers; C51's prediction shows **two modes** across five consecutive frames, one for "reward already came" and one for "not yet". Since the state has no memory of past rewards, the agent literally cannot collapse the modes — and the distributional target is stable where a mean target would flicker.
3. **A richer set of predictions** — auxiliary tasks whose accuracy is tightly coupled to performance, unlike bolt-on auxiliary losses.
4. **Well-behaved optimisation.** Cross-entropy over a categorical is just an easier landscape than a squared TD error.

The Space Invaders picture is the clearest intuition: three actions (the ones involving button press) lead to firing the laser too early and eventually dying, and their distributions place real mass on 0 (terminal). A mean would blend "I die" and "I survive" into one number that corresponds to *no actual outcome*. The distribution keeps them separate.

**Surprises.**
- Distributions were often close to Gaussian-shaped, not spiky, despite ALE being deterministic. Attributed to discretising the diffusion induced by $\gamma$.
- The **erratum** matters as a methodology lesson: the camera-ready reported 1010% mean; the real figure is 701%. The bug was Atlantis episodes running past the standard 30-minute cap, giving 3.7M score (22,824% human-normalised) instead of 841,075 (5,199%). One game moved the *mean over 57 games* by 300 points. Median was unaffected at 178%. Report medians.

**Caveats if you use this.**
- $V_{\min}, V_{\max}$ are hyperparameters you must set per domain, and they are not innocent — they change the optimisation problem. Rewards outside the range are silently clipped.
- Theory covers policy evaluation only. Control has no convergence guarantee to a *stationary* optimal value distribution.
- Action selection still uses only the mean. The authors explicitly leave "the many ways an agent could select actions on the basis of the full distribution" as future work — risk-sensitive policies, optimism, etc.
- The follow-up (Bellemare et al. 2017, Cramér distance) explains the biased-Wasserstein-gradient problem properly; QR-DQN and IQN later replaced the fixed-atom grid with learned quantiles, which fixes the $V_{\min}/V_{\max}$ hack. C51 is also one of the six components of Rainbow.

**Open question they raise:** could stochastic policies (à la conservative policy iteration, Kakade & Langford 2002) restore stability to the distributional control operator?

## Links

Related: [[Playing Atari with Deep Reinforcement Learning (DQN)]] · [[Markov Decision Process]] · [[KL Divergence]] · [[Cross Entropy]] · [[Random variable]] · [[Uncertainty]] · [[Decision Sciences]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)]] · [[Proximal Policy Optimization Algorithms]] · [[Regularization]]

New topics worth writing: Wasserstein distance and optimal transport, Contraction mappings and the Banach fixed-point theorem, Quantile Regression DQN (QR-DQN) and IQN, Rainbow DQN, Cramér distance, Risk-sensitive reinforcement learning, Arcade Learning Environment and sticky actions, Auxiliary tasks in RL (UNREAL)
