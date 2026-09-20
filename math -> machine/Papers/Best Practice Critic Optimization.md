---
title: "Best Practice Critic Optimization"
authors: ["Qi et al."]
year: 2026
arxiv: "2608.23566"
url: https://arxiv.org/abs/2608.23566
priority: Good-To-Read
read_on: 2026-09-02
tags: [paper, llm, rl, theory]
---
## The Core Idea

Reinforcement learning on language models needs a way to say "this token was good, that token was bad". Two families do it differently.

**Group-based** (GRPO, Dr. GRPO): sample 16 answers to the same question, see which scored higher than average, and give every token in a good answer the same positive credit. No value network needed. But you pay 16 rollouts per prompt, and every token in a 20,000-token answer gets the same score — even the ones that did nothing.

**Critic-based** (PPO): train a second network, the critic, that reads a half-finished answer and predicts "how likely is this to end up correct?". Then you only need **one** rollout per prompt, and you get a different number for each token. In theory this is strictly better. In practice people found it kept blowing up, so the field mostly abandoned it.

This paper is a debugging report. The authors take critic-based training, break it on purpose in a tiny controlled setting, and fix five separate things. The fixes are individually boring and collectively decisive:

1. Clip by absolute probability change, not by ratio (DPPO).
2. Squash the critic's output into the range the reward can actually take. A linear head predicting "0.7 correct" can also predict "−4.2 correct", which is nonsense.
3. Train the critic on the **actual observed reward**, not on a target that contains the critic's own old prediction.
4. Do **not** divide advantages by their batch standard deviation.
5. Scale the GAE $\lambda$ with response length, so the final reward has the same influence in a 1,000-token answer as in a 20,000-token one.

Plus one genuinely new idea. The critic is thrown away after training — it never runs at deployment. So you can **feed it things the policy is not allowed to see**: the reference answer, the official worked solution, the grading rubric. This is the multi-agent RL trick of "centralised training, decentralised execution" applied to LLM RL. It does not change the true value function (that information is already fixed by the prompt) but it makes the function much easier for a finite network to approximate.

Result: one rollout per prompt matches or beats 16-rollout Dr. GRPO on maths, at 1.5B and at 30B-A3B.

> [!NOTE] Privileged critic input
> Information given only to the value network during training, hidden from the policy. Valid because the critic is discarded before deployment, and because the information is a deterministic function of the prompt so it does not change $V^\mu$. ^privileged-critic

## The Methodology

**Setup.** Prompt $x$, response $y = (y_1,\dots,y_T)$ generated token by token. State $s_t = (x, y_{<t})$, action is the next token. Reward is outcome-only: $r_t = 0$ for $t < T$, and $r_T = R(x,y)$ (typically 1 if the maths answer is right, 0 otherwise). Discount $\gamma = 1$.

**Policy objective — DPPO instead of PPO.** Standard [[Proximal Policy Optimization Algorithms|PPO]] clips the ratio $\rho_t(\theta) = \pi_\theta(y_t|s_t)/\mu(y_t|s_t)$ to $[1-\epsilon, 1+\epsilon]$. With a 150k-token vocabulary this is unfair: a token with probability 0.9 can move by 0.18 in absolute terms, while a token with probability 0.0001 can only move by 0.00002. DPPO replaces $\epsilon$ with $\epsilon/\mu(y_t|s_t)$:

$$\mathcal{L}_{\mathrm{DPPO}} = \mathbb{E}_t\!\left[\min\!\left(\rho_t \widehat{A}_t,\ \mathrm{clip}\!\left(\rho_t, 1-\tfrac{\epsilon}{\mu(y_t|s_t)}, 1+\tfrac{\epsilon}{\mu(y_t|s_t)}\right)\widehat{A}_t\right)\right]$$

Which is exactly the constraint $|\pi_\theta(y_t|s_t) - \mu(y_t|s_t)| \le \epsilon$. Every token gets the same absolute probability budget.

**Bounded value head.** Let $z_\phi(s_t)$ be the raw linear head output. Squash it:

$$V_\phi(s_t) = R_{\min} + (R_{\max}-R_{\min})\left(\frac{1}{2} + \frac{1}{\pi}\arctan\big(z_\phi(s_t)\big)\right)$$

The expectation of a bounded random variable must lie inside the same bounds, so any prediction outside $[R_{\min}, R_{\max}]$ is provably wrong. `arctan` maps to the open interval and saturates smoothly at the ends. For binary rewards $R_{\min}=0, R_{\max}=1$.

**Decoupled $\lambda$: two different GAE parameters.** Recall [[High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)|GAE]]: $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, and $\widehat{A}_t^{\mathrm{GAE}(\lambda)} = \sum_{l} (\gamma\lambda)^l \delta_{t+l}$.

Most implementations build the critic target as $\widehat{V}_t(\lambda) = \widehat{A}_t^{\mathrm{GAE}(\lambda)} + V_{\phi_{\mathrm{old}}}(s_t)$ using the *same* $\lambda$ as the policy. When $\lambda < 1$ this target contains the old critic's own output — the critic is partly regressing onto itself.

BPCO uses $\lambda_V = 1$ for the critic. With $\gamma=1$ and outcome-only reward the sum telescopes completely:

$$\widehat{V}_t = \widehat{A}_t^{\mathrm{GAE}(1)} + V_{\phi_{\mathrm{old}}}(s_t) = R(x,y)$$

The critic regresses every prefix of the response directly onto the final observed reward. That is an unbiased Monte Carlo sample of $V^\mu(s_t)$. Meanwhile the *policy* keeps $\lambda_\pi < 1$ for variance reduction.

**Length-adaptive GAE (LA-GAE).** With fixed $\lambda_\pi$, the weight on the terminal reward for a token at position $t$ is $\lambda_\pi^{T-t}$. At $\lambda_\pi = 0.99$ and $T-t = 10{,}000$ that weight is effectively zero — early tokens are driven entirely by critic bootstrap error. Fix: make $\lambda$ depend on the response length $L$:

$$\lambda_\pi(L) = 1 - \frac{1}{\alpha L}$$

Then $\left(1 - \frac{1}{\alpha L}\right)^{L} \approx \exp(-1/\alpha)$, which does not depend on $L$. Every response, short or long, gives the terminal reward the same influence on its first token. They use $\alpha = 0.4$, so the weight is $e^{-2.5} \approx 0.082$.

**No advantage normalization.** The usual trick replaces $\widehat{A}_t$ with $(\widehat{A}_t - \bar{A})/\sigma_A$ per batch. BPCO deletes this. Two arguments:
- As the policy nears optimal, real advantages shrink toward zero and updates *should* fade out. Dividing by a tiny $\sigma_A$ resurrects pure estimation noise as a unit-scale training signal. The updates never stop.
- Subtracting $\bar{A}$ can flip the sign of a genuinely good action whose advantage happens to be below the batch mean, which pushes the policy away from it.

**Privileged critic.** The critic estimates $V^\mu_\phi(s_t, q(x))$ where $q(x)$ is reward-defining side information. Variants: `+Ans` (reference answer), `+Sol` (official worked solution), `+Rubrics`. The policy input is untouched.

**Training details.** Base recipe on `verl`. 1,024 trajectories per iteration, minibatch 256, one epoch → four optimizer steps per iteration. Policy LR $10^{-6}$, critic LR $10^{-5}$. Critic warm-up: 15 iterations in the large runs, none needed in the sanity test. One rollout per prompt.

## Ablation Studies and Experiments

**The sanity test is the best part of the paper.** Take DeepSeek-R1-Distill-Qwen-1.5B and 1,460 maths problems it can *already solve*. A working recipe must reach ~100% training reward. Failing here is an optimization bug, not a capacity or reward-signal problem. Held-out metric: AIME 2025 avg@32.

Each step keeps the previous ones:

| Step | Change | Result |
|---|---|---|
| 1 | PPO → DPPO, $\lambda=1$ | PPO's reward **collapses** after rising. DPPO is stable. |
| 1b | DPPO with $\lambda = 0.99$ | Unstable again. Kept as a deliberate stress test. |
| 2 | Bounded value head | Linear head predicts values outside $[0,1]$; training reward is erratic. Bounded head reaches ~1.0. |
| 3 | $\lambda_V = 1$ (MC target) | Stabler reward, faster convergence. |
| 4 | Remove advantage norm | Same training reward, **better** AIME, advantage range stays small and flat. |
| 5 | Privileged answer to critic | Faster reward, higher explained variance, AIME rises faster but **peaks earlier**. |
| 6 | LA-GAE, $\alpha=0.4$ | Best trade-off between speed and AIME decline. |

**The explained-variance trap (step 3).** They track $\mathrm{EV} = 1 - \mathrm{Var}(\widehat{V}_t - V_\phi(s_t))/\mathrm{Var}(\widehat{V}_t)$. With the bootstrapped target at $\lambda = 0.99$, EV rockets to nearly 1 while policy training falls apart. The critic is trivially predicting a target that is mostly its own old output. **EV against a bootstrapped target is a meaningless metric.** Measured against the observed reward instead, it becomes informative. This is a practical caveat worth carrying around.

**The $\lambda_\pi$ trade-off (step 6).** Fixed $\lambda_\pi = 0.99$ fits training fastest but AIME peaks and then declines sharply. $\lambda_\pi = 1$ avoids the decline but optimizes slowly. LA-GAE at $\alpha = 0.4$ sits between them.

**Larger dataset (§4.1).** DeepScaleR, 40.3K problems, 1.5B model, 24k max generation. BPCO beats both the critic baseline (which already has decoupled GAE + MC target + LA-GAE, and differs from BPCO *only* by the unbounded head and the advantage normalization) and Dr. GRPO with $G=16$. Explained variance is consistently higher.

Ablations off BPCO+Ans:
- Removing the value bound: slower training reward, lower AIME. Still matters at scale.
- Reintroducing advantage normalization: advantage magnitude grows over training again, though less dramatically than in the sanity test. Performance loss is only modest here — because training has not converged, so $\sigma_A$ has not yet collapsed. The authors still recommend removing it as a default.
- Privileged info at this scale: **helps clearly**. Reference answer gives faster training, higher EV, better AIME. Official solution gives a modest gain even though only 7.3K of 40.3K problems have one. The overfitting risk from the sanity test does not materialise when the dataset is big enough.

**Larger models (§4.2).** Qwen3-30B-A3B-Base and Qwen3-30B-A3B (MoE) on DAPO-Math-17K, 12k max length. On the instruct model, the critic baseline **stops improving AIME after ~100 steps** — the old recipe simply breaks at this scale. BPCO keeps going and reaches substantially higher AIME. Versus the group baseline: BPCO wins on Qwen3-30B-A3B, ties on the Base model.

**Rubric rewards (§4.3).** Qwen3-4B-Base on OpenRubrics, judged by a frozen Qwen3-4B-Instruct-2507 against a per-prompt rubric. Both BPCO variants learn faster than group and critic baselines, though the group baseline eventually catches up. The critic baseline ends slightly lower with low EV.

**What did not work:**
- **Privileged rubric gave no policy benefit here**, despite raising explained variance. The authors guess the task is too easy for a better critic to matter. So: a better critic does not automatically mean a better policy.
- **Privileged info overfits in the small-data sanity test** — validation peaks earlier than without it.
- **Fixed $\lambda_\pi < 1$ on long responses** — fast training, worse generalisation.
- **PPO clipping itself** — collapses in the sanity test where DPPO does not.
- **Critic warm-up in the small-data setting** — no benefit observed.

## Worth Remembering

- The headline economic claim: **one rollout per prompt matching 16.** If the critic's forward/backward cost is less than 15 extra generations, this is a large compute win. But the authors admit the trajectory-matched comparison "does not capture" the critic's extra compute and memory — you are carrying a second full-size model. Verify the accounting yourself before believing the efficiency story.
- The advantage-normalization argument is the one to internalise generally. Dividing by $\sigma_A$ means your update magnitude is scale-free *by construction*, so it can never tell you "you are done". Near the optimum you are amplifying noise. This applies well outside LLMs.
- Bounding the value head is nearly free and provably correct whenever the reward range is known. Odd that it took this long.
- The "centralised training, decentralised execution" import from [[Multi-Agent Reinforcement Learning]] is the most transferable idea. Anything the reward function sees but the policy does not — unit tests, ground-truth labels, grading criteria — is fair game for the critic.
- Limitations the authors state plainly: only maths and rubric rewards tested; a known reward range is required (rules out unbounded reward models unless you squash them); privileged variants need evaluator information you may not have; extra compute unaccounted for.
- Open question: does BPCO's per-token advantage actually assign credit *sensibly*, or is it just a variance-reduced version of the same outcome signal? The paper shows training stability and benchmark numbers but never inspects which tokens get high advantage.
- Note the citation dates (2026, arXiv IDs like 2607/2608). Several of the building blocks — DPPO, SAO, DeepScaleR — are themselves very recent and mostly from the same group.

## Links

Related: [[Proximal Policy Optimization Algorithms]] · [[High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)]] · [[Trust Region Policy Optimization (TRPO)]] · [[Training language models to follow instructions with human feedback]] · [[Multi-Agent Reinforcement Learning]] · [[Value Function Factorization]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[Soft Actor-Critic]] · [[Conservative Q-Learning for Offline RL]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Sparsely-Gated Mixture-of-Experts Layer]] · [[Markov Decision Process]] · [[Uncertainty]]

New topics worth writing: GRPO and Dr. GRPO, DPPO / trust regions in probability space, explained variance as a critic diagnostic, centralized training with decentralized execution, VAPO and VC-PPO, rubric-as-reward RL, verl / HybridFlow training stack, single-rollout asynchronous RL (SAO)
