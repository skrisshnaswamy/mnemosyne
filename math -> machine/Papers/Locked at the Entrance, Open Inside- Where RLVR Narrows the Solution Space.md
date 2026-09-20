---
title: "Locked at the Entrance, Open Inside: Where RLVR Narrows the Solution Space"
authors: ["Qiancheng Zhou", "Ruizhe Li"]
year: 2026
arxiv: "2608.29188"
url: https://arxiv.org/abs/2608.29188
priority: Good-To-Read
read_on: 2026-09-16
tags: [paper, llm, rl]
---
## The Core Idea

When you train a language model with RLVR — reinforcement learning where the reward is just "did the checker say the answer is right?" — two things happen at once. Single-sample accuracy (pass@1) goes up a lot. And the model stops producing *different* solutions. Sample it 320 times and you get the same answer path over and over. That kills test-time scaling: repeated sampling, self-consistency, and search all need varied candidates to work.

Everyone knew breadth collapses. Nobody knew **where inside a single reasoning chain** it collapses. That is the question this paper answers, and the answer is clean:

> [!NOTE] Access vs Execution
> A model solving a problem by route $b$ must do two things: **access** — actually start down route $b$; and **execute** — finish the arithmetic correctly once started. Formally
> $$\pi_\theta(\text{solve via } b \mid x) = \underbrace{\pi_\theta(B=b\mid x)}_{\text{access}} \cdot \underbrace{\pi_\theta(\text{valid completion}\mid x, B=b)}_{\text{execution}}$$
> RLVR destroys access. Execution actually gets *better*. ^access-vs-execution

The catchphrase: **breadth is lost at the door, not inside the room.** The trained model still knows how to solve the problem five different ways. It has simply stopped opening four of the five doors.

Why could nobody measure this before? Because prior work measured diversity by clustering sampled traces or counting distinct final answers. If the model never samples a route, clustering cannot see that the route existed. You need a **ground-truth list of every valid solution** to notice a missing one. That is why the authors pick Countdown, where a symbolic solver can enumerate the complete solution set exactly.

What it unlocks: a targeted fix. If breadth dies at the first decision, then prompting tricks and temperature bumps should fail (they do), and interventions that reshape the *early* computational state should work (they do). Blending late-checkpoint layers 20–28 with step-50 weights gives **+37% solution coverage at no loss in pass@1**.

## The Methodology

**The task.** Countdown: given numbers like $\{4,5,7,9\}$ and target $24$, build an arithmetic expression using each number exactly once. A solver enumerates every valid expression, canonicalised so that $a+b$ and $b+a$ count as one.

**Entrance families.** Partition the solution set by the *first operand and first operator*. For $\{4,5,7,9\}\to 24$ the families are things like `5−`, `7×`, `4×`, `9−`. Across 150 test problems the solver finds 379 feasible families, ~2.53 per problem. A family fixes only the opening move; everything downstream is free. That is exactly what makes it a clean access/execution split.

**Coverage.** With $n$ rollouts $Y_{1:n}$:
$$\mathrm{Cov}(x, Y_{1:n}) = \frac{|\{s \in \mathcal{S}(x) \mid s \text{ is the canonical form of some } Y_i\}|}{|\mathcal{S}(x)|}$$
averaged over problems. $\mathcal{S}(x)$ is the full solver-enumerated solution set — the denominator is *real*, not empirical.

**Two training runs, deliberately independent.**
- PPO on Qwen2.5-3B via TinyZero. 11 actor checkpoints, steps 25→275. Rollout group size $G=1$, no actor KL penalty, actor LR $10^{-6}$, critic $10^{-5}$, GAE with $\gamma=\lambda=1.0$.
- A public GRPO checkpoint series (Qwen2.5-3B-Instruct, TRL GRPO). Evaluated, not trained. The 50k-example training superset is filtered out by semantic key, leaving 135 clean problems.

Evaluation: $T=0.7$, top-$p=0.9$, 320 rollouts per problem, 256-token cap (PPO) or 1024 (GRPO).

**Three measurement instruments.**

1. **Teacher-forced NLL phase attribution.** Take a solver-written valid solution. Score it token-by-token under an early and a late checkpoint. Split the sequence at the first arithmetic operator. Compare how much the negative log-likelihood per token rose in the "entrance" segment versus the "execution" segment. Teacher forcing means the model never has to *generate* anything, so you can score even badly-formatted early checkpoints.

2. **Entrance clamping.** Append a solver-built prefix to the prompt. Three levels: nothing; **minimal entrance** (just `Let me try: 4 *`); **completed first calculation** (`4 * 7 = 28`). Then let the model finish freely. Measure the rate of valid completions *inside the designated family*. This is an interventional quantity, written $E_t^{\mathrm{do}}(b\mid x)$.

3. **A predictability gate.** Critical control. Does the prefix leak downstream information? They test whether the rest of the expression can be read off the prefix. Entrance-only prefixes score 0 (no leak) on all 150 problems. Prefixes cut from the model's own successful reasoning score 1 (massive leak) on all 150 — which is exactly why the paper refuses to use them for identification.

**Why RLVR should do this, analytically.** Let $q_b$ be access to family $b$ and $\mu_b$ its conditional success rate. For the surrogate $J_x = \sum_b q_b \mu_b$ with $q = \mathrm{softmax}(u)$:
$$\frac{\partial J_x}{\partial u_b} = q_b(\mu_b - \bar\mu)$$
The gradient is **multiplied by current access**. A family that gets sampled rarely receives a tiny gradient, so it stays rare, so it gets an even tinier gradient. Self-reinforcing contraction, baked into [[Proximal Policy Optimization Algorithms|PPO]]-family on-policy updates.

**Generalising off Countdown.** Real math benchmarks have no enumerable solution set, so they build proxies: the **first calculation** (earliest parsed arithmetic relation) and its categorical entropy; the **distinct-trace rate** over canonicalised calculation sequences. Six benchmarks: GSM8K (500), MATH500 (500), Minerva (272), OlympiadBench (500), AMC23 (40), AIME24 (30). 64 samples each at $T=0.6$, top-$p=0.95$, 16k token cap.

## Ablation Studies and Experiments

**The tradeoff itself.**

| Checkpoint | pass@1 | pass@64 | pass@256 | Coverage@320 |
|---|---|---|---|---|
| PPO base | 0.001 | 0.076 | 0.228 | 0.097 |
| PPO step 50 | 0.051 | 0.500 | 0.596 | **0.337** |
| PPO step 150 | 0.323 | 0.441 | 0.457 | 0.152 |
| PPO step 275 | 0.285 | 0.376 | 0.385 | **0.111** |
| GRPO step 25 | 0.121 | 0.633 | 0.750 | 0.467 |
| GRPO step 450 | 0.429 | 0.570 | 0.621 | 0.268 |

pass@1 goes up 50× under PPO while coverage falls 67%. Note pass@64 and pass@256 *decline* after step 50 — the model gets worse at the thing you actually use test-time compute for. Bootstrap CIs for step-50 coverage $[0.277, 0.397]$ and step-275 $[0.079, 0.145]$ are disjoint.

**The survivorship control — the most important sanity check.** Maybe coverage falls because the model forgot the hard problems. No. On the 58 problems solved at *both* step 50 and step 275, coverage still falls 0.564 → 0.286. Only 31% of the individual solution leaves found at step 50 are ever regenerated at step 275. And no problem missed at step 50 gets newly unlocked at step 275. Push to 2,048 samples and 33 lost problems stay lost. RLVR does not widen reachability; it concentrates mass on a shrinking subset.

**Localisation, the headline number.**

| Diagnostic | PPO | GRPO |
|---|---|---|
| NLL increase/token: entrance vs execution | $+4.50$ vs $+0.28$ (**16×**) | $+8.08$ vs $+0.71$ (**11×**) |
| Designated completion: no prefix → minimal entrance | $0.018 \to 0.212$ | $0.104 \to 0.188$ |
| Minimal-entrance completion: early → late | $0.113 \to 0.212$ | $0.131 \to 0.188$ |

Read the last row carefully. Conditional execution **improves** during training ($+0.107$, CI $[0.073, 0.143]$ on matched problem/family keys) while access collapses. Token-level attribution puts the single biggest likelihood drop on the operand token that selects the family — $-3$ positions before the operator, NLL jumps $8.602 \to 16.108$.

**Extinct families are still competent.** Take 28 families with ≥0.05 access at step 50 and **zero** access across 320 free samples at step 275. Clamp the minimal entrance: designated completion is **0.529** at step 275, higher than 0.475 at step 50. The routes are not broken. They are unvisited.

**The prefix ladder — what is the minimum useful nudge?** (retry scaffold, step 275)

| Cue | Completion |
|---|---|
| Generic retry text | 0.015 |
| First number alone | 0.015 |
| Generic plan | 0.023 |
| Misleading plan | 0.004 |
| **Completed local calculation** | **0.211** |

The useful intervention sits in a narrow band: fix one concrete arithmetic action, leave everything else open. A bare label does nothing. A plan does nothing. An *operand plus operator* does everything.

**Interventions, 100 held-out problems, 64 samples, paired against step-275 control (coverage 0.111, pass@1 0.278).**

| Intervention | pass@1 | Δ coverage |
|---|---|---|
| Prompt "give a different method" | 0.282 | $+0.014$ $[0.000, 0.038]$ |
| **Forced alternative operator** | 0.286 | $\mathbf{-0.007}$ $[-0.017, 0.002]$ |
| **Answer-only logit mixing** | 0.269 | $\mathbf{-0.006}$ $[-0.015, 0.000]$ |
| High temperature ($T=1.0$) | 0.264 | $+0.016$ $[0.007, 0.027]$ |
| Reasoning-phase logit mixing | 0.277 | $+0.013$ $[0.004, 0.026]$ |
| **Layer interpolation (20–28 ↔ step 50)** | **0.305** | $\mathbf{+0.041}$ $[0.020, 0.069]$ |
| Checkpoint sampling (32/32 split) | 0.161 | $+0.100$ $[0.059, 0.147]$ |

**What did not work, and why it is informative:**
- *Forced operator override* moves coverage **negative**. This is the sharpest ablation in the paper. Clamping the operator *token* without supplying the operand state does nothing — so "diversity" is not about which symbol gets emitted, it is about which computational state the model is in.
- *Answer-only logit mixing* (blending early-checkpoint logits only inside `<answer>` tags) does nothing. Confirms breadth is not a formatting or output-layer phenomenon.
- *Prompt diversification* moves $+0.014$. Surface prompting cannot steer a collapsed policy.
- *Temperature* is the one surface method that widens support, but it costs pass@1 (0.278 → 0.264), and a full $5\times2\times2$ decoding sweep never reaches step-50 coverage. Best is 0.194 at $T=2.0$, where pass@1 falls to 0.238.
- *Checkpoint sampling* buys the most coverage but drags pass@1 down to 0.161 — you are half-sampling a weak model.

Layer interpolation is the only free lunch: more breadth *and* higher pass@1.

**Test-time entrance allocation.** If you have no old checkpoints, spend your token budget uniformly across solver-identified feasible entrances instead of free resampling. At step 275: $+0.077$ coverage $[0.048, 0.109]$, $+0.430$ distinct families $[0.297, 0.570]$. At step 50 the same trick gives **nothing** ($-0.001$). Structured allocation only pays once the policy has collapsed — exactly what the gradient argument predicts.

**Scaling to real math.** Qwen2.5 base vs SimpleRL, macro over six benchmarks:

| Model | pass@1 | pass@64 | First-calc $H$ | Distinct-trace |
|---|---|---|---|---|
| Base 7B | 0.379 | 0.754 | 1.063 | 0.780 |
| SimpleRL 7B | 0.510 | **0.760** | 0.744 | 0.444 |
| Base 14B | 0.377 | 0.774 | 1.207 | — |
| SimpleRL 14B | 0.546 | **0.774** | 0.711 | — |

pass@64 is flat at both scales. All the gain is pass@1. First-calculation entropy drops 30% (7B) and 41% (14B). On GSM8K correct-only traces, entropy collapses $0.894 \to 0.285$. At 256 samples on GSM8K, SimpleRL produces **2.20** distinct first calculations per problem versus **8.98** for base.

**Same-trace difference-in-differences.** Score identical reference traces under both policies, split at the first complete calculation:
$$\mathrm{DiD}_d = (C_{\text{base},d} - E_{\text{base},d}) - (C_{\text{RLVR},d} - E_{\text{RLVR},d})$$
Positive on all six benchmarks with CIs above zero (GSM8K 0.313, MATH500 0.337, Minerva 0.193). Robust to a **blinded LLM resegmentation** of 19,160 traces — so it is not an artifact of the regex-based boundary parser.

**Execution becomes a second bottleneck on long horizons.** In Countdown, an entrance leaves 6–10 tokens to go. On GSM8K it leaves hundreds. Supplying base-discovered first calculations that RLVR never samples, the late policy still completes at 0.850 (GSM8K) and 0.713 (MATH500). But a depth-controlled handoff on Countdown gives a depth×late interaction of $\gamma = -0.0152$ $[-0.0224, -0.0080]$: the late policy's execution advantage shrinks as more downstream reasoning remains.

**The collapse is not inevitable.** This is the constructive half.

| Model | pass@1 | pass@64 | First-calc $H$ | Distinct-trace |
|---|---|---|---|---|
| Qwen SimpleRL 7B (direct RLVR) | 0.510 | 0.760 | 0.744 | 0.444 |
| Qwen Distill 7B (R1-distill) | 0.526 | 0.795 | **1.240** | 0.871 |
| OLMo-3 7B SFT | 0.447 | 0.781 | 1.189 | 0.770 |
| OLMo-3 7B DPO | 0.508 | 0.784 | 1.269 | 0.786 |
| OLMo-3 7B RLVR | **0.613** | **0.828** | **1.137** | 0.763 |

The staged SFT → [[Direct Preference Optimization (DPO)|DPO]] → RLVR ladder raises pass@1 from 0.447 to 0.613 *and* pass@64 from 0.781 to 0.828, while first-calculation entropy stays near the SFT baseline. Direct RLVR on a base model is what wrecks breadth.

**Multi-solution SFT, dose–response.** Fine-tune Qwen2.5-3B-Instruct on solver-enumerated demonstrations, $k$ solutions per problem, LoRA rank 32, $\alpha=64$:

| $k$ | pass@1 | pass@64 | Coverage |
|---|---|---|---|
| GRPO step 450 | 0.429 | 0.570 | 0.268 |
| 1 | 0.120 | 0.737 | 0.471 |
| 2 | 0.221 | 0.806 | 0.551 |
| 4 | 0.333 | 0.855 | 0.709 |
| 8 | 0.293 | 0.836 | 0.766 |

Strict monotone in $k$. Even $k=1$ keeps nearly double the coverage of late GRPO. But: continue GRPO for 500 steps from the $k=4$ checkpoint and rollout entropy falls to 0.085 nats. **The collapse re-emerges from a high-diversity start.** SFT does not immunise you.

## Worth Remembering

- The single most useful number to carry: **supplying only an operand and an operator lifts completion in never-visited families from 0.018 to 0.212**, while a generic hint or a plan lifts it to 0.023. The information content of the useful cue is essentially one arithmetic state.

- Layer interpolation on blocks **20–28** (late layers of a 3B model) is what recovers breadth. Early layers are not implicated. This is a concrete, cheap intervention — you already have the checkpoints.

- The gradient argument $\partial J_x/\partial u_b = q_b(\mu_b - \bar\mu)$ generalises well beyond Countdown. Any on-policy method that samples from the current policy has this multiplicative-by-current-probability property. It is the same rich-get-richer force behind [[Mode Collapse]] in other settings.

- Admitted limitations: exhaustive enumeration ties the *clean* analysis to Countdown. The PPO run uses $G=1$ and **no actor KL penalty** — normally the [[KL Divergence|KL]] leash is the standard tool against distribution collapse, so this run is an unusually favourable setting for observing narrowing. The GRPO run is evaluated, not trained by the authors. On real math the "first calculation" is a heuristic proxy, not a solver-defined entrance.

- $E_t^{\mathrm{do}}$ is interventional — it measures completion after you *force* an entrance, not after the policy chooses one. Their own observational Shapley decomposition on free rollouts assigns 34.5% of the change in solve probability to access and 65.5% to improved conditional execution. Both stories are true: execution genuinely improves, and *that* is where pass@1 gains come from; access collapse is where breadth goes.

- Practical caveat for anyone running [[Proximal Policy Optimization Algorithms|PPO]]/GRPO on reasoning: track first-calculation entropy or distinct-trace rate alongside pass@1. pass@64 saturating while pass@1 climbs is the warning sign. At 7B/14B it is *flat* — you are paying for accuracy purely with breadth.

- The R1-distill result is quietly the most interesting: highest pass@1 (0.526) *and* highest first-calculation entropy (1.240) among 7B models. High-capacity distilled initialisation plus RLVR appears to escape the tradeoff. Connects to [[Distillation]] and [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]].

- The zero-access designation is statistically solid: of 337 families absent from 512 samples, 336 are absent from an independent second 512, none above 0.02 access, 95% Clopper–Pearson upper bound 0.0058. These families really are extinct under free sampling.

- Open question: does the entrance framing survive when the "entrance" is a strategic choice (which lemma, which substitution) rather than an arithmetic op? The GSM8K case study (forward algebra vs backward deduction) suggests yes, but it is one problem.

## Links

Related: [[Proximal Policy Optimization Algorithms]] · [[Training language models to follow instructions with human feedback]] · [[Direct Preference Optimization (DPO)]] · [[Mode Collapse]] · [[KL Divergence]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Distillation]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[Auto-regressive models]] · [[Cross Entropy]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[The Lottery Ticket Hypothesis]] · [[Is Next-Chunk Reasoning RL Really Better than SFT- Revisiting Training Strategies under no-CoT Data]] · [[Dynamic Important Example Mining for Reinforcement Finetuning]]

New topics worth writing: GRPO (Group Relative Policy Optimization), RLVR, pass@k estimator, test-time scaling / repeated sampling, self-consistency decoding, weight-space interpolation and model soups, entropy collapse in policy optimization, forking tokens / high-entropy minority tokens, temporal checkpoint sampling, Countdown / TinyZero benchmark, problem-cluster bootstrap, TOST equivalence testing, Clopper–Pearson intervals, Shapley decomposition of probability changes
