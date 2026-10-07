---
title: "PACT: From Credit Assignment to Critic Alignment"
authors: ["Jiayan Fu", "Hang Xu", "Yong Zhang", "Zhaokai Luo", "Yao Hu", "Dongyan Zhao", "Mu Chuan"]
year: 2026
arxiv: "2609.26355"
url: https://arxiv.org/abs/2609.26355
priority: Must-Read
read_on: 2026-09-24
tags: [paper, llm, rl, theory]
---
## The Core Idea

Reinforcement learning on language models has a granularity mismatch. You get **one number at the end** — the answer was right, the tests passed — but you update **every token** in a trajectory that might be 64k tokens long. How much of that final number belongs to token 4,312? That is the credit assignment problem, and until now nobody had written down what "credit" even *means* mathematically. Every algorithm just picked a proxy: GAE picked TD residuals, GRPO picked the group mean, on-policy distillation picked a teacher's log-ratio.

This paper does the Shannon move: instead of proposing another proxy, it writes down three properties any sane credit rule must have, and proves **exactly one rule satisfies all three**.

The three properties, in plain words:

1. **Completeness.** The credits must add up to the whole surprise. $\sum_i C_i = R - \mathbb{E}[R\mid \mathcal{F}_0]$. Nothing is left unexplained, nothing is invented.
2. **Prefix Consistency.** Credit already given to a prefix cannot be rewritten by things that happen later. If two trajectories share the first $i$ tokens, they must have the same credit-so-far.
3. **Neutrality.** Before you generate a token, its expected credit is zero: $\mathbb{E}[C_i \mid \mathcal{F}_{i-1}] = 0$. No token gets a systematic bonus or tax that was predictable in advance.

The unique answer is a one-liner:

$$C_i = V_i - V_{i-1} = \mathbb{E}[R \mid \mathcal{F}_i] - \mathbb{E}[R \mid \mathcal{F}_{i-1}]$$

Credit for a token is **how much the token changed your best guess about the final reward**. Nothing else. And $\{C_i\}$ is a martingale difference sequence, which is where all the downstream results come from.

> [!NOTE] Token-level credit
> The credit of token $i$ is the jump in the conditional expectation of the final reward when that token is revealed: $C_i = V_i - V_{i-1}$, with $V_i = \mathbb{E}[R\mid\mathcal{F}_i]$. It is the *unique* assignment satisfying Completeness, Prefix Consistency and Neutrality. ^unique-credit

Why this unlocks things: once you have the definition, you can ask of any existing algorithm "is your signal this, in expectation?" The answer is yes for [[Simple Statistical Gradient-Following Algorithms (REINFORCE)|RLOO]], yes for on-policy distillation with an ideal teacher, and *partially* for [[High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)|GAE]] — and the place GAE leaks is exactly where the authors build their fix.

The practical upshot is **PACT**: train the critic *after* the actor, with an importance-sampling correction so the critic's values belong to the policy you just made, and train it with binary cross-entropy rather than squared error. On four maths benchmarks it gets 72.87% average vs 64.07% for [[GRPO]] and 59.71% for [[PPO]].

## The Methodology

### Setup and notation

A trajectory is $Y = (q, T_1, O_1, \ldots, T_\tau, O_\tau)$: prompt, then alternating generated tokens $T_i$ and environment observations $O_i$ (tool output, test results, user reply). Plain text generation is the special case where every $O_i = \varnothing$. The reward $R = \mathcal{R}(Y)$ arrives only at the end. $\mathcal{F}_i$ is everything known after token $i$ and its observation.

This is the standard token-level [[Markov Decision Process|MDP]] view, but the paper's point is that the Markov framing alone says nothing about attribution — you need the three extra conditions.

Coarser granularities fall out by summing: turn-level credit for a segment from token $a_k$ to $b_k$ is just $V_{b_k} - V_{a_k - 1}$. Same theorem, aggregated.

### Consequence 1 — the distillation teacher is a critic in disguise

In [[Distillation|on-policy distillation]] (OPD), you sample from the student and give each token the signal $A_t^{\mathrm{OPD}} = \log \frac{q_t(T_t)}{p_t(T_t)}$, teacher over student.

Define an **ideal teacher** as the [[KL Divergence|KL]]-regularised improvement of the current policy:

$$q_t^\star = \arg\max_{q} \left\{ \mathbb{E}_{a\sim q}[Q_t^\pi(a)] - \beta D_{\mathrm{KL}}(q \| p_t) \right\}$$

which has the closed form $q_t^\star(a) \propto p_t(a)\exp(Q_t^\pi(a)/\beta)$. Take logs: $\log \frac{q_t^\star(a)}{p_t(a)} = \frac{1}{\beta}Q_t^\pi(a) - \log \mathcal{Z}_t$. The $\log \mathcal{Z}_t$ term is prefix-constant, so it dies against the score-function identity $\mathbb{E}[\nabla_\theta \log \pi_\theta(T_t\mid\mathcal{F}_{t-1})] = 0$ — the same trick as a [[Policy Gradient#baseline|baseline]]. What is left:

$$G_t^{\mathrm{OPD}}(q_t^\star) = \tfrac{1}{\beta}\,\mathbb{E}_\pi[Z_t C_t^\pi \mid \mathcal{F}_{t-1}]$$

The OPD gradient *is* the credit-weighted policy gradient, up to the scale $1/\beta$. So a teacher model is a critic that happens to be stored as a distribution over tokens instead of a value head. This reframes the "critic-free methods won" narrative: OPD never dropped the critic, it just hid it.

### Consequence 2 — RLOO's coarse signal is not biased, only noisy

[[Simple Statistical Gradient-Following Algorithms (REINFORCE)|RLOO]] gives every token in response $i$ the same scalar $R_i - \bar{R}_{-i}$, where $\bar{R}_{-i}$ is the mean reward of the other $G-1$ samples for that prompt. That is response-level, and credit is token-level. Does it matter?

In expectation, no:

$$\mathbb{E}[Z_t (R_i - \bar{R}_{-i})] = \mathbb{E}[Z_t C_t]$$

The proof is two lines of martingale bookkeeping. For $k < t$, $C_k$ is already known at $\mathcal{F}_{t-1}$ so it kills against the score. For $k > t$, $Z_t$ is known at $\mathcal{F}_{k-1}$ so $C_k$'s zero-mean property kills it. Only $C_t$ survives. The leave-one-out baseline is independent of the current trajectory, so it contributes nothing.

But — and this is the motivation for the whole paper — **equal in expectation is not equal in variance**. $V_t$ is the minimum-mean-squared-error predictor of $R$ given $\mathcal{F}_t$; the RLOO baseline keeps the raw randomness of a handful of sampled outcomes. In long trajectories that noise swamps the local signal you are trying to read.

### Consequence 3 — credit is approximately sparse, and that is bad news for GAE

Normalise $R \in [0,1]$ (a positive affine map, does not change the optimal policy). Because the $C_i$ are orthogonal martingale differences summing to $R - V_0$:

$$\mathbb{E}\Big[\sum_{i=1}^\tau C_i^2 \,\Big|\, \mathcal{F}_0\Big] = \mathrm{Var}(R\mid\mathcal{F}_0) \le \tfrac14$$

and by Chebyshev-style counting, the expected number of tokens with $|C_i| > \epsilon$ is at most $\frac{1}{4\epsilon^2}$ — **independent of how long the response is**. A 64k-token trajectory does not get 64k meaningful decisions. It gets a handful of big ones and a sea of near-zero ones.

Now put critic error $\widehat{V}_i = V_i + \varepsilon_i$ into GAE with $\gamma = 1$. The TD residual becomes $\widehat{\delta}_i = C_i + \varepsilon_i - \varepsilon_{i-1}$, and the advantage decomposes as:

$$\widehat{A}_t^\lambda = \sum_{i=t}^{\tau}\lambda^{i-t}C_i \;-\; \varepsilon_{t-1} \;+\; (1-\lambda)\sum_{i=t}^{\tau-1}\lambda^{i-t}\varepsilon_i$$

The last term is pure critic noise, and it only exists when $\lambda < 1$. Combine with sparsity: the true credits are tiny almost everywhere, so a noise term of comparable size *dominates*. At $\lambda = 1$ the term vanishes exactly and you are left with $\widehat{A}_t^1 = R - \widehat{V}_{t-1}$ — one value error, not a weighted pile of them.

This explains the folklore that $\lambda = 1$ beats $\lambda = 0.95$ in LLM RL (DeepSeek-R1 reported this). PACT uses $\lambda = 1$.

### PACT itself

Two changes to [[PPO]].

**1. Binary cross-entropy critic.** Since $R \in [0,1]$, so is $V_i$. Parameterise $\widehat{V}_{\phi,i} = \sigma(z_{\phi,i})$ and train with soft BCE:

$$\mathcal{L}_{\mathrm{BCE}} = \mathbb{E}\big[-R\log\widehat{V}_{\phi,i} - (1-R)\log(1-\widehat{V}_{\phi,i})\big]$$

For $R\in[0,1]$, BCE and MSE have the *same* minimiser, $V_i^\pi$ — so this changes the optimisation geometry, not the target. (See [[Cross Entropy]]. The precedent is Farebrother et al.'s "Stop Regressing" result in deep RL.) The logit gradient is the clean $\widehat{V}_{\phi,t-1} - R$, and the appendix proves $\mathbb{E}[(v-m)^2] \le \tfrac12 \mathbb{E}[\ell(v) - \ell(m)]$, so driving BCE to its floor drives value error to zero in mean square.

**2. Actor-then-Critic with importance correction.** This is the real contribution. In standard PPO the critic that computes advantages for batch $\mathcal{D}_k \sim \pi_k$ was trained on $\mathcal{D}_{k-1} \sim \pi_{k-1}$. It is always one policy behind, and since credit is a *difference* of two values, a lag that is small in absolute value can be large relative to $C_t$.

PACT reorders the loop:

1. Roll out with $\pi_k$, record old log-probs $\ell^{\mathrm{old}}$.
2. Compute advantages with the current (stale) critic, do all $B$ actor steps → $\pi_{k+1}$.
3. **One extra forward pass** of $\pi_{k+1}$ over the same rollouts to get $\ell^{\mathrm{new}}$.
4. Ratio $\rho_t = \exp(\ell^{\mathrm{new}}_t - \ell^{\mathrm{old}}_t)$, detached. Mask tokens where $\rho \notin [0, 6]$.
5. Train the critic with target $Y_t = \rho_t R$ instead of $R$.

The justification is a [[Off-Policy Evaluation|change of measure]]: the exact correction for the *new* policy's value is $V_{t-1}^\pi = \mathbb{E}_\mu[I_t R \mid \mathcal{F}_{t-1}]$ where $I_t = \prod_{k=t}^{\tau} \rho_k$ is the whole continuation ratio. That product explodes over 64k tokens, so PACT truncates it to the single current-token ratio. The appendix is honest about what this is: $\rho_t R$ gives you the value of the **hybrid policy** that samples token $t$ from $\pi$ and everything after from $\mu$. It is the one-step truncation of the first-order expansion, accurate to $O_L(\varepsilon)$ when all $|\rho_k - 1| \le \varepsilon$. Critically, BCE recovers the *conditional mean* of whatever you feed it, so individual targets $\rho_t R$ falling outside $[0,1]$ do not break the minimiser.

Cost: one forward pass per iteration. No extra rollouts.

### Training setup

- Maths: Qwen3.5-4B, 3,200 problems from DAPO-Math-17k (biased towards ones the base model fails), OpenCode agent harness with a Python sandbox, reward = final answer correct.
- Coding: Qwen3.6-35B-A3B, OpenSWE environments, Codex agent via Harbor, reward from task verifier.
- 512 trajectories per rollout round, optimisation batch 128 → 4 minibatches. GRPO uses 8 samples × 64 prompts.
- 128k context, 64k max generation per turn. Actor LR $1\mathrm{e}{-6}$, critic LR $5\mathrm{e}{-6}$.
- Framework: Dressage on top of slime.

## Ablation Studies and Experiments

**Maths, Avg@16 across four benchmarks (Qwen3.5-4B):**

| Method | AIME 2025 | AIME 2026 | BeyondAIME | HMMT Nov 25 | Avg |
|---|---|---|---|---|---|
| Base | 46.67 | 51.25 | 28.63 | 37.50 | 41.01 |
| PPO ($\lambda{=}0.95$) | 32.33 | 29.38 | 17.86 | 25.67 | **26.31** |
| SAO | 51.25 | 63.33 | 36.63 | 53.33 | 51.14 |
| PPO ($\lambda{=}1.0$) | 66.04 | 73.96 | 40.31 | 58.54 | 59.71 |
| GRPO (clip-higher 0.28) | 76.50 | 74.78 | 41.81 | 63.19 | 64.07 |
| PACT w/o IS | 76.04 | 82.50 | 51.38 | 61.04 | 67.74 |
| **PACT** | **83.12** | **85.21** | **51.69** | **71.46** | **72.87** |

**SWE-bench Verified, pass@1 (Qwen3.6-35B-A3B):** base 60.8 → SAO 63.6 → PPO($\lambda{=}1$) 65.0 → GRPO 65.4 → **PACT 67.4**.

**The thing that did not work: PPO with $\lambda = 0.95$.** It ended at 26.31%, *below the 41.01% base model* — it collapsed during training. That is the single most informative number in the paper, because it is the prediction the theory makes: at $\lambda < 1$, intermediate critic errors enter the advantage, and against approximately-sparse credit they dominate. Switching one hyperparameter to $\lambda = 1$ moves you from catastrophic to 59.71%.

**Importance sampling ablation.** "PACT w/o IS" keeps everything — Actor-then-Critic ordering, BCE critic, actor-side settings — and only removes the $\rho_t$ correction on critic targets. 67.74 → 72.87, **+5.13 points**, improving on all four benchmarks, and with visibly more stable training reward curves. So the reordering plus BCE buys ~3.7 points over GRPO, and the correction itself buys another ~5.

**BCE vs MSE critic.** Isolated properly: freeze the rollout policy, train both critics from the same init on the same on-policy data. The BCE critic reaches lower BCE loss, lower *MSE* loss, and higher value separation $\Delta_\pm$ (mean predicted value on successful trajectories minus failed ones) — at both 4B and 35B scale. It converges substantially faster. Worth noting: this is a *pretraining* ablation at fixed policy, not an ablation inside the RL loop, so the contribution of BCE to the final 72.87 is not separately measured.

**Secondary evidence for sparsity.** During PACT training, response length grows while the mean $|\widehat{V}_t - \widehat{V}_{t-1}|$ falls — consistent with "more tokens, same total credit budget, so each one carries less."

## Worth Remembering

**The uniqueness result is conditional, and the authors say so.** Change the axioms, get a different credit. They use the parallel-postulate analogy: swap it and you get a consistent non-Euclidean geometry. The three conditions are shown to be logically independent — Appendix C.2.3 builds a two-step Rademacher example where dropping any one admits a different assignment. Dropping Completeness lets you assign zero everywhere. Dropping Prefix Consistency lets future noise rewrite past credit. Dropping Neutrality lets you shuffle credit between positions with predictable compensating terms.

**This credit is statistical, not causal.** $C_i = V_i - V_{i-1}$ is a conditional expectation. It does not tell you what would have happened had you emitted a different token. That is a different object (see Counterfactual Credit Assignment, COCOA, Hindsight Credit Assignment in their related work) and the authors flag the gap explicitly.

**They define credit but do not solve estimating it.** The whole PACT contribution is making the critic *less wrong about which policy it describes*. Getting $V_i$ accurate enough that $V_i - V_{i-1}$ is meaningful in a 64k-token trajectory remains open.

**Read the SAO number with suspicion.** SAO at 51.14 on maths is barely above the 41.01 base and far below GRPO, yet it is competitive on SWE-bench (63.6 vs 65.4). SAO is an asynchronous method being run in what looks like a synchronous comparison, and a baseline that underperforms this badly in one setting and not the other usually indicates a tuning problem rather than a method problem. See [[On the Difficulty of Evaluating Baselines]] and [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] — the pattern is well documented.

**No seeds, no error bars.** Avg@16 smooths sampling noise within an evaluation but says nothing about run-to-run variance in training. AIME-scale benchmarks are 30 problems; 83.12 vs 76.50 is roughly two problems.

**Practical caveats if you want to use PACT:**
- You need a critic. This is an [[Actor-Critic]] method, so you pay the memory and the second optimiser, which is exactly what [[GRPO]] was designed to avoid. On a 35B MoE that is not free.
- Your reward must be bounded and normalised to $[0,1]$ for the BCE parameterisation to make sense. Fine for verifiable rewards; awkward for a learned [[Preference Learning|reward model]] with unbounded scores.
- The masking bounds matter: they drop critic-loss tokens with $\rho \notin [0,6]$, and use actor-side DIS in $[0.7, 6.0]$ for maths, PPO clipping for coding. Those are not obviously transferable.
- The $\rho_t R$ target is a *one-step* truncation. If your actor takes large steps per iteration (many minibatch passes, high LR), $\varepsilon$ is not small and the $O_L(\varepsilon)$ guarantee degrades.

**The conceptual payoff worth keeping.** Three unrelated-looking families — teacher distillation, group-relative baselines, learned value heads — are all estimating the same quantity $V_i - V_{i-1}$ with different machinery and different variance. That is a genuinely useful way to hold the [[Reinforcement Learning|RL]]-for-LLMs landscape in your head, and it suggests the right question about any new algorithm is not "does it have a critic?" but "what is its variance when estimating the value jump?"

**Open question this leaves.** If credit is approximately sparse — at most $1/(4\epsilon^2)$ tokens with $|C_i| > \epsilon$, regardless of length — can you *find* those tokens cheaply and only spend estimation effort there? Nothing in PACT exploits the sparsity; it just uses it to argue against $\lambda < 1$.

## Links

Related: [[Credit Assignment]] · [[PPO]] · [[GRPO]] · [[Actor-Critic]] · [[High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)]] · [[Proximal Policy Optimization Algorithms]] · [[Value Function]] · [[Policy Gradient]] · [[Temporal Difference Learning]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[On-Policy vs Off-Policy]] · [[Off-Policy Evaluation]] · [[Distillation]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[KL Divergence]] · [[Cross Entropy]] · [[Markov Decision Process]] · [[Reward Function]] · [[RLHF]] · [[Bellman Equation]] · [[DRACO- Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training]] · [[On the Difficulty of Evaluating Baselines]]

New topics worth writing: Martingale difference sequences, Doob decomposition, RUDDER and return redistribution, Hindsight Credit Assignment, Counterfactual Credit Assignment (COCOA), VC-PPO and VAPO value pretraining, classification-based value functions ("Stop Regressing"), SAO / single-rollout asynchronous RL, truncated importance sampling for long sequences, SWE-bench Verified as a benchmark
</br>
