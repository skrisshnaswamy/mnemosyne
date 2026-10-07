---
title: "Bellman Policy Optimization"
authors: ["Song et al."]
year: 2026
arxiv: "2609.15987"
url: https://arxiv.org/abs/2609.15987
priority: Good-To-Read
read_on: 2026-09-27
tags: [paper, llm, rl, optimization, theory]
---
## The Core Idea

GRPO-style RL for reasoning models multiplies each token's gradient by an **importance-sampling ratio** $r_t = \pi(y_t|s_t)/\mu(y_t|s_t)$ — how much more likely the token is under the current policy $\pi$ than under the rollout policy $\mu$ that actually generated it. That ratio is inherited from [[PPO]], and nobody re-derived it for the specific case of "one reward at the end of a long autoregressive generation".

This paper does re-derive it, and gets a different weight:

$$\omega_t = \frac{1 + \epsilon - \mu(y_t|s_t)}{1 + \epsilon - \pi(y_t|s_t)}$$

A ratio of **complementary** probabilities — one minus the token probability — instead of the probabilities themselves. Everything else in the loss is GRPO. Swapping $r_t \to \min\{\omega_t, C\}$ takes AIME 2024–26 average accuracy on Qwen3-30B-A3B-Base from **39.5% to 50.5%**.

Why this weight and not the other one? The derivation starts from **Policy Mirror Descent** (PMD), which has a known closed-form optimal update:

$$\pi^+(y_t|s_t) \;=\; \frac{\mu(y_t|s_t)\exp\!\big(\eta A^\mu(s_t, y_t)\big)}{Z_\mu(s_t)}$$

You cannot run this directly on a language model, because it needs the [[Value Function|advantage]] $A^\mu$ at *every intermediate token position*. That means training a critic — extra memory, extra compute, and critics are known to be inaccurate on reasoning tasks (VinePPO's finding).

The trick: because rewards arrive only at the end and token transitions are deterministic, the per-token advantage is exactly a difference of values, $A^\mu(s_t,y_t) = V^\mu(s_{t+1}) - V^\mu(s_t)$. Sum along a response and the middle terms cancel:

$$\sum_{t=1}^{|y|} A^\mu(s_t, y_t) = V^\mu(s_{|y|+1}) - V^\mu(s_1) = R(x,y) - V^\mu(x)$$

The verifier gives you $R(x,y)$. The prompt-level value $V^\mu(x)$ is just the mean reward over the sampled group. **Every intermediate value vanishes.** So you can write down a single trajectory-level condition that the PMD optimum must satisfy, with no critic anywhere, and they prove that condition pins down the *same* policy PMD would have produced.

> [!NOTE] Mismatch-correction weight
> The scalar $\omega_t = (1+\epsilon-\mu_t)/(1+\epsilon-\pi_t)$ that multiplies $\nabla \log \pi(y_t|s_t)$ in BPO, replacing the importance-sampling ratio. It corrects for the gap between the rollout engine's policy and the training policy. ^mismatch-correction-weight

## The Methodology

### The setup, in MDP terms

State $s_t = (x, y_{<t})$ — prompt plus tokens generated so far. Action = next token. Transition is deterministic (append the token). Reward is zero everywhere except the terminal state, where a verifier gives $R(x,y) \in [-1,1]$. This is a standard [[Markov Decision Process]] framing of RLVR.

Two policies matter and they are not the same object:
- $\mu$ — the **rollout policy**, whatever the inference engine (vLLM/SGLang) actually sampled from.
- $\pi$ — the **current policy** being updated. It drifts from $\mu$ across the 8 optimizer steps per rollout batch, and differs from it even at step 0 because of numerical differences between the rollout and training engines.

### Step 1 — PMD with advantages

Instantiate PMD at each state:

$$\max_{\pi(\cdot|s_t)} \;\; \mathbb{E}_{y_t \sim \pi}\big[A^\mu(s_t,y_t)\big] - \tfrac{1}{\eta} D_{\mathrm{KL}}\big(\pi(\cdot|s_t)\,\|\,\mu(\cdot|s_t)\big)$$

Maximise advantage, pay a [[KL Divergence|KL]] penalty for moving away from $\mu$. Using $A^\mu$ rather than $Q^\mu$ changes nothing, since $V^\mu(s_t)$ does not depend on the action.

### Step 2 — kill the partition function

Take logs of the closed-form solution:

$$\log \pi^+(y_t|s_t) - \log\mu(y_t|s_t) - \eta A^\mu(s_t,y_t) + \log Z_\mu(s_t) = 0$$

Now take the expectation of this over $y_t \sim \mu$. The advantage term vanishes, because $\sum_{y}\mu(y|s)A^\mu(s,y) = 0$ by definition. What is left is a neat identity:

$$\log Z_\mu(s_t) = D_{\mathrm{KL}}\big(\mu(\cdot|s_t)\,\|\,\pi^+(\cdot|s_t)\big)$$

The awkward state-dependent normaliser *is* a reverse KL. This is the same reparameterisation-by-likelihood-ratio move that [[DPO]] uses to eliminate the reward model.

### Step 3 — telescope, and get a squared residual

Substitute back, sum over the response, apply the telescoping identity. Define the **trajectory residual**:

$$\delta(x,y;\pi,\mu) = \eta\big(R(x,y) - V^\mu(x)\big) - \sum_{t=1}^{|y|}\Big(\log\frac{\pi(y_t|s_t)}{\mu(y_t|s_t)} + D_{\mathrm{KL}}\big(\mu(\cdot|s_t)\,\|\,\pi(\cdot|s_t)\big)\Big)$$

The PMD optimum makes $\delta = 0$ for every sampled response. So minimise the expected square:

$$\min_{\pi} \;\; \mathbb{E}_{x\sim\mathcal{D},\,y\sim\mathbb{P}_\mu(\cdot|x)}\Big[\phi(x)\,\tfrac{\delta(x,y;\pi,\mu)^2}{2\eta}\Big]$$

$\phi(x)$ is any positive per-prompt weight.

### Theorem 1 and the martingale argument

Sufficiency is easy: $\pi^+$ gives $\delta = 0$, and the objective is non-negative, so the minimum is zero.

Necessity is the interesting half. Why can't some *other* policy hit $\delta = 0$ by having positive and negative per-token errors that cancel along the response? Define the per-token residual $d_\pi(s_t,y_t)$. For **any** feasible $\pi$, its conditional expectation under $\mu$ is exactly zero — again because $\sum_y \mu A^\mu = 0$ and the reverse-KL term is definitionally the expectation of the log-ratio. So the running sum of per-token residuals is a **martingale** under the rollout distribution. If the terminal value is zero almost surely and the horizon is bounded by $T_{\max}$, then $M_n = \mathbb{E}[M_{T_{\max}}|\mathcal{F}_n] = 0$ for every $n$, so every increment is zero. Cancellation is impossible. Every token-level condition must hold individually on states reachable under $\mu$.

That is the whole theoretical content: **a trajectory-level loss enforces a token-level condition, for free, because the residuals are a martingale.**

### Step 4 — four approximations to get a shippable loss

The theory gives a squared trajectory objective. The practical loss is reached by throwing most of it away:

**(a) Linearise.** Expand around $\pi = \mu$, where $\delta(x,y;\mu,\mu) = \eta(R - V^\mu(x))$. This replaces the factor $\delta/\eta$ in the gradient with $R(x,y^i) - V^\mu(x)$.

**(b) Estimate from the group.** $V^\mu(x) \approx \texttt{mean}(\{R_j\}_{j=1}^G)$, and choose $\phi(x) = 1/\texttt{std}(\{R_j\})$. The two together are *exactly* the [[GRPO]] group-normalised advantage $\hat A^i$. The free weighting function was chosen to reproduce GRPO's advantage.

**(c) Binary KL.** The full reverse KL needs logits over the whole vocabulary. Replace it with the **binary KL** (from DPPO): collapse the vocabulary into $\{y_t\}$ vs everything else, giving two Bernoullis. This is a lower bound on the true KL. The payoff is an exact gradient identity:

$$\nabla\Big(\log\pi(y_t|s_t) + D^{\mathrm{bin}}_{\mathrm{KL}}\big(\mu\,\|\,\pi; y_t\big)\Big) = \frac{1 - \mu(y_t|s_t)}{1 - \pi(y_t|s_t)}\,\nabla\log\pi(y_t|s_t)$$

Two lines of algebra (use $\nabla p = p\nabla\log p$). The complementary ratio drops out of the derivative of the KL term. That is where $\omega$ comes from — **it is not a heuristic, it is the gradient of the binary reverse KL.**

**(d) Smooth, cap, mask.** $(1-\mu)/(1-\pi)$ explodes as $\pi \to 1$, so add $\epsilon$ to both numerator and denominator. Cap at $C$. Apply GRPO's asymmetric clipping mask, with $\omega$ in place of $r$.

Final loss:

$$\mathcal{L}^{\mathrm{BPO}} = -\hat A^i\, M^i_t \,\min\{\texttt{sg}(\omega^i_t), C\}\, \log\pi(y^i_t|x,y^i_{<t})$$

`sg` is stop-gradient. The weight is a plain scalar multiplier — this is a one-line diff from GRPO.

### What the weight actually does differently

Worth running numbers, because the *behaviour* is the opposite of importance sampling. With $\epsilon = 0.1$:

| $\mu_t$ | $\pi_t$ | $r_t = \pi/\mu$ | $\omega_t$ |
|---|---|---|---|
| 0.001 | 0.01 | **10.0** | 1.008 |
| 0.5 | 0.9 | 1.8 | **3.0** |
| 0.9 | 0.5 | 0.56 | 0.33 |
| 0.5 | 0.5 | 1.0 | 1.0 |

Both weights equal 1 when the policies agree, and both move the same direction. But importance sampling blows up on **rare tokens** (the classic variance problem — tiny denominator), which is why GRPO needs clipping and CISPO needs capping. BPO's weight is almost flat there and instead becomes large on tokens the current policy has become **confident** about. Maximum possible value is $(1+\epsilon)/\epsilon = 11$, hence the cap $C = 3$.

### Training configuration

Qwen3-30B-A3B-Base on the English subset of DAPO-Math-17k. 256 prompts per rollout batch, $G = 16$ responses each, so 4096 responses split into 8 minibatches of 512 → 8 optimizer updates per training step. 400 training steps = 3200 updates. Max response 16384 tokens. AdamW, constant LR $10^{-6}$, $\beta_2 = 0.98$, weight decay 0.1, grad clip 1.0. **No KL penalty, no entropy bonus.** Temperature 1.0, top-$p$ 1.0, no top-$k$. Loss aggregation `seq-mean-token-mean`. All methods use rollout-router replay (R3) to stop MoE router mismatch between engines. BPO: $\epsilon = 0.1$, $C = 3.0$.

## Ablation Studies and Experiments

Controlled: only the policy loss changes, everything else identical. Metric is Avg@32 (32 samples per question, averaged correctness) as an estimate of Pass@1, at the best checkpoint by mean of the three benchmarks.

| Method | AIME24 | AIME25 | AIME26 | Avg. |
|---|---|---|---|---|
| GRPO-ClipHigher | 45.6 | 34.8 | 38.0 | 39.5 |
| GSPO | 50.3 | 35.5 | 44.6 | 43.5 |
| DPPO | 55.8 | 39.2 | 44.2 | 46.4 |
| CISPO | 52.7 | 39.0 | 50.4 | 47.4 |
| **BPO** | **57.4** | **41.0** | **53.0** | **50.5** |

BPO wins on all three benchmarks. Gains: +11.0 over GRPO-ClipHigher, +7.0 GSPO, +4.1 DPPO, +3.1 CISPO. At the end of 400 steps (not peak) BPO is at 49.4% vs DPPO's 45.5%, so the peak is not a lucky checkpoint spike.

Baselines were given reasonable configs, not strawmen: GRPO-ClipHigher uses DAPO's $[0.8, 1.28]$; CISPO's cap is 3.0 (same as BPO's); DPPO uses binary total variation with $\delta = 0.1$ from TMax; GSPO uses $[1-3\!\times\!10^{-3}, 1+5\!\times\!10^{-3}]$ from the RL-scaling paper.

### Hyperparameter ablations — Qwen3-4B-Base, 1000 steps

Smaller model, 128 prompts × 16 responses, 4 minibatches, 8192 max length.

$\epsilon$ sweep with $C = 3.0$: **25.8 / 25.4 / 25.5 / 24.1** for $\epsilon \in \{0.05, 0.1, 0.2, 0.3\}$. GRPO-ClipHigher baseline: **20.5**.

$C$ sweep with $\epsilon = 0.1$: **25.3 / 25.4 / 25.8** for $C \in \{2.0, 3.0, 4.0\}$ — a spread of 0.5 points.

The honest reading: BPO is insensitive to both knobs across a 4× range, and beats the baseline at all seven settings tested. $\epsilon = 0.3$ is mildly worse (too much smoothing flattens $\omega$ towards 1, i.e. towards no correction at all). Notably the 4B gap (+4.9) is much smaller than the 30B gap (+11.0).

### What is *not* ablated, and this is the gap

There is **no ablation isolating which of the four approximations matters.** Specifically missing:
- BPO's weight with GRPO-style clipping vs BPO's weight *without* the cap $C$ — so we cannot tell whether the cap is load-bearing or cosmetic.
- The un-smoothed weight $(1-\mu)/(1-\pi)$ at all, which is the actual quantity the theory produces.
- The squared-residual objective *before* linearisation. Every result uses the linearised form, which is GRPO-plus-a-different-weight. The theorem is about an objective nobody trained on.
- No seed variance on any number. The main table is single-run.

So the empirical claim that survives is narrow and useful: **replacing $\pi/\mu$ with the smoothed complementary ratio is worth 3–11 points on AIME.** The claim that the PMD equivalence theorem *explains* that is unproven.

## Worth Remembering

**"Critic-free" here means no value network, not per-token credit.** All tokens in a response still share one scalar advantage $\hat A^i$. The Bellman telescoping does not recover per-token credit — it *destroys* it, deliberately, so that only the terminal reward and prompt-level baseline survive. If you want actual [[Credit Assignment]] within a response, this paper does not give it to you; see [[PACT- From Credit Assignment to Critic Alignment]] for the other direction.

**The whole theory reduces to one gradient identity.** After linearisation and group estimation, BPO is GRPO. The single thing PMD + Bellman + binary KL buys you is the formula for $\omega$. Worth knowing because it means you can adopt BPO without buying the framework — it's a scalar change in your loss function.

**The variance story is inverted, and that may be the real mechanism.** GSPO, CISPO and DPPO are all fixes for the same disease: token-level importance ratios explode on low-probability tokens, because $\mu$ sits in a denominator. $\omega$ has $1-\pi$ in the denominator instead, so rare tokens contribute a weight near 1 and the variance simply is not there. My current best guess is that BPO's gain is mostly this, and that the PMD derivation is a principled route to a variance-reduced weight rather than the cause of the improvement. Open question, not established.

**The weight up-weights confidence, which deserves suspicion.** When $\pi \to 1$ on a token and the response got positive reward, $\omega$ hits the cap and you push hard on a token that is already nearly certain. That gradient does almost nothing (the logit is saturated) but it does consume the clipping budget. Conversely for negative advantage the mask kicks in at $\omega < 1-\epsilon_{\text{low}}$. Whether this is a feature of the reverse-KL $D(\mu\|\pi)$ term — which penalises $\pi$ for *dropping* mass $\mu$ put somewhere — or an artefact of the binary approximation, the paper does not say.

**Reverse KL, not forward.** The derivation produces $D_{\mathrm{KL}}(\mu\|\pi)$, the mode-seeking direction with $\mu$ in front. This falls out of the partition-function identity, not from a design choice. Relevant if you are tracking the [[KL Divergence#Forward KL vs Reverse KL|forward/reverse]] distinction across RLHF methods.

**Practical caveats for adoption.** One model family (Qwen3), one domain (competition maths), terminal binary rewards only. The derivation *requires* deterministic transitions and zero intermediate reward — it will not transfer as-is to agentic settings with tool feedback mid-trajectory, or to any process reward model. R3 router replay was on for every run, so this is untested without it on MoE. $\epsilon$ and $C$ are both safe defaults; do not spend a sweep on them.

**The rollout–training mismatch framing.** The paper calls $\omega$ a "mismatch-correction weight", positioning it alongside the FP16 paper and the rollout-training-mismatch line of work. That is the right way to hold it: this is not an exploration or credit-assignment improvement, it is a **numerical-stability** improvement for the gap between your inference engine and your trainer.

## Links

Related: [[GRPO]] · [[PPO]] · [[Proximal Policy Optimization Algorithms]] · [[Policy Gradient]] · [[Bellman Equation]] · [[Value Function]] · [[KL Divergence]] · [[DPO]] · [[Credit Assignment]] · [[On-Policy vs Off-Policy]] · [[PACT- From Credit Assignment to Critic Alignment]] · [[Markov Decision Process]] · [[Reward Function]] · [[Off-Policy Evaluation]] · [[Temporal Difference Learning]]

New topics worth writing: Policy Mirror Descent, regularised MDPs (Geist et al.), binary KL divergence and trust-region surrogates, GSPO, CISPO, DPPO, rollout–training mismatch in RL infrastructure, VinePPO and critic accuracy on reasoning tasks, martingale arguments for trajectory-to-token objective equivalence
