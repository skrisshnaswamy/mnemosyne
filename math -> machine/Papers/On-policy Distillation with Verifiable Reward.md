---
title: "On-policy Distillation with Verifiable Reward"
authors: ["Lin et al."]
year: 2026
arxiv: "2608.24696"
url: https://arxiv.org/abs/2608.24696
priority: Good-To-Read
read_on: 2026-09-02
tags: [paper, llm, rl]
---
## The Core Idea

Two ways to post-train a reasoning model, each broken in a different way.

**RLVR** (reinforcement learning with verifiable rewards): let the model write a full answer, check the final answer with a rule, give $+1$ if right and $0$ or $-1$ if wrong. One number for a 8000-token trajectory. Great signal, almost no information about *which* token was the mistake.

**On-policy distillation (OPD)**: let the student write the answer, then at every token ask a teacher model "how likely was this token under you?" and push the student toward the teacher. Dense — a signal at every position. But it never checks whether the answer was actually right. The student's ceiling is the teacher.

The insight of this paper is that **sampled-token OPD is already secretly a policy-gradient method with a badly-signed reward**, and fixing the sign is one line of code.

Write the sampled-token OPD loss (a one-sample estimate of reverse KL):

$$\mathcal{L}_{\text{OPD}}^{\text{sample}} = \sum_{t}\log\frac{\pi_\theta(o_t\mid q,o_{<t})}{\pi_T(o_t\mid q,o_{<t})}$$

The log-ratio is stop-gradient'd (treated as a constant). So its gradient is

$$\nabla_\theta\mathcal{L} = \sum_t \log\frac{\pi_\theta}{\pi_T}\cdot\nabla_\theta\log\pi_\theta(o_t)$$

Compare to the REINFORCE gradient $-R\sum_t\nabla_\theta\log\pi_\theta(o_t)$. Same shape. Match coefficients and you get an **implicit token reward**

$$R_{\text{OPD}}(o_t) = \log\frac{\pi_T(o_t)}{\pi_\theta(o_t)}$$

Now the problem is obvious. This reward is positive whenever the teacher is more confident than the student, negative otherwise — and that has *nothing to do with whether the answer was correct*. Every mainstream RL recipe follows the rule: tokens on a correct trajectory get non-negative advantage, tokens on a wrong trajectory get non-positive advantage. OPD breaks that rule about half the time.

The fix is a ReLU gate keyed on the verifier:

$$R_{\text{OPDVR}}(o_t) = R\cdot\text{ReLU}\!\left(R\cdot\log\frac{\pi_T(o_t)}{\pi_\theta(o_t)}\right),\qquad R\in\{+1,-1\}$$

That is it. No new hyperparameter, no weighting coefficient between two losses, no heuristic switch. It turns OPD into a legitimate RLVR algorithm, which means you can drop it into GRPO, PPO or DAPO wherever the advantage goes.

> [!NOTE] Sampled-token OPD ^sampled-token-opd
> Instead of computing the full KL over the whole vocabulary at each position (expensive: vocab is ~150k), take the single token the student actually sampled and use $\log(\pi_\theta/\pi_T)$ on that token as a stop-gradient reward. An unbiased one-sample estimate of the reverse-KL gradient.

> [!NOTE] Verifier-conflicting token ^conflicting-token
> A token where the OPD update direction fights the task reward. Two kinds: (I) correct trajectory but $\pi_\theta > \pi_T$ — OPD pushes a *correct* token's probability *down* to match a less confident teacher; (II) wrong trajectory but $\pi_T > \pi_\theta$ — OPD pushes a token on a *wrong* answer *up*. OPDVR zeroes both.

The slogan: **the teacher sets the magnitude, the verifier sets the direction.** Reinforce confident-and-correct behaviour, suppress overconfident mistakes.

## The Methodology

**The gate, spelled out.** For a trajectory with verifier reward $R$ and per-token log-ratio $r_t=\log(\pi_T/\pi_\theta)$:

- Correct trajectory ($R=+1$): reward is $\max(0, r_t)$. Keep tokens where the teacher was *more* confident than the student — those are the places the student can still learn. Zero out the tokens where the student was already more confident than the teacher.
- Wrong trajectory ($R=-1$): reward is $-\max(0, -r_t)$. Keep and punish tokens where the *student* was more confident than the teacher — overconfident errors. Zero out the ones where the teacher agreed, since the teacher's endorsement did not lead anywhere good here.

Loss:
$$\mathcal{L}_{\text{OPDVR}}(\theta) = -\sum_{t=1}^{|o|} R_{\text{OPDVR}}(o_t)\cdot\log\pi_\theta(o_t\mid q,o_{<t})$$

**As a mask.** Split the OPD gradient into two halves:
$$g_{\text{OPD}} = \underbrace{[r_t]_+\nabla_\theta\log\pi_\theta}_{\text{A: push up}} - \underbrace{[-r_t]_+\nabla_\theta\log\pi_\theta}_{\text{B: pull down}}$$
OPDVR keeps only Term A when $R=+1$ and only $-$Term B when $R=-1$. So it is exactly OPD with the harmful component deleted. The appendix proves the trivial-but-clarifying fact that $\langle \Delta_{\text{OPDVR}}, \Delta_{\text{RLVR}}\rangle \ge 0$ always, while $\langle \Delta_{\text{OPD}}, \Delta_{\text{RLVR}}\rangle = r_t R\|u_t\|^2$ which goes negative whenever $r_tR<0$.

**GRPD (Group Relative Policy Distillation).** Since OPDVR is now a real RLVR method, swap the crude binary $R$ for a GRPO group-relative advantage. Sample $G=8$ responses per prompt, compute $\hat A_{i,t} = (R_i - \text{mean})/\text{std}$, and gate on its sign:

$$R_{\text{GRPD}}(o_{i,t}) = \text{sign}(\hat A_{i,t})\cdot\text{ReLU}\!\left(\text{sign}(\hat A_{i,t})\cdot\log\frac{\pi_T(o_{i,t})}{\pi_\theta(o_{i,t})}\right)$$

with the usual GRPO normalisation $\frac{1}{G}\sum_i\frac{1}{|o_i|}\sum_t$. Note the magnitude of $\hat A$ is discarded — only its sign is used, and the teacher log-ratio supplies the size.

**Setups.**
- *Same architecture*: student Qwen3-4B-nonthinking; teacher is Qwen3-4B trained with GRPO on DeepMath (57k filtered samples, difficulty $\ge 6$).
- *Cross architecture*: student Qwen3-1.7B-base; teacher Qwen3-4B-base fine-tuned with GRPO for 3 epochs on DAPO-Math-17k. 3 epochs of distillation.
- Benchmarks: AIME24, AIME25, AMC, MATH500, Minerva, OlympiadBench. Metric avg@16.
- Hyperparameters: LR $1\times10^{-6}$, batch 256, PPO mini-batch 256, max response 8192, max prompt 1024, rollout temperature 1.0, eval temperature 0.7 / top-$p$ 0.95. Verl framework, RTX 5090s.

## Ablation Studies and Experiments

**Same architecture (Qwen3-4B ← Qwen3-4B-RL), avg@16:**

| Method | AIME24 | AIME25 | AMC | MATH500 | Minerva | Olympiad | Avg |
|---|---|---|---|---|---|---|---|
| Student | 24.0 | 15.8 | 60.8 | 80.9 | 27.6 | 42.9 | 42.0 |
| Teacher | 36.0 | 29.0 | 65.9 | 87.0 | 35.4 | 49.3 | 50.4 |
| Sampled-token OPD | 34.2 | 26.0 | 63.1 | **85.5** | 31.6 | 46.5 | 47.8 |
| Top-64 OPD | 34.6 | 23.5 | 62.0 | 85.0 | 32.2 | 46.8 | 47.4 |
| **OPDVR** | **36.9** | **28.1** | **64.8** | 84.7 | **33.2** | **47.0** | **49.1** |

$+1.3$ average over OPD, and it beats the teacher on AIME24 (36.9 vs 36.0). Note it *loses* to OPD on MATH500 (84.7 vs 85.5) — the easiest benchmark in the set.

**Cross architecture (1.7B ← 4B):** OPDVR avg 22.8 vs 20.9 for sampled-token OPD and 21.7 for top-64. Biggest single jump is AMC 30.3 vs 24.8 ($+5.5$). Everything here is far below the teacher's 30.9 — a 1.7B student cannot close the gap in 3 epochs.

**GRPD (on DAPO-Math-17k, deliberately *different* data from the teacher's DeepMath training set):**

| Method | AIME24 | AIME25 | AMC | MATH500 | Minerva | Olympiad | Avg |
|---|---|---|---|---|---|---|---|
| GRPO | 28.3 | 20.8 | 62.3 | 83.9 | 28.9 | 44.6 | 44.8 |
| OPD | 32.0 | 31.7 | 65.6 | 85.4 | 28.9 | 46.6 | 48.4 |
| **GRPD** | **34.8** | 31.7 | **67.0** | **85.6** | **30.5** | **47.0** | **49.4** |

GRPD beats plain GRPO by $+6.5$ on AIME24 and $+10.9$ on AIME25 — a large gap, which mostly says that dense teacher signal is worth a lot when the reward is sparse. It beats OPD on five of six (ties on AIME25). Interesting: OPD here scores 31.7 on AIME25, *above* the teacher's 29.0.

**The ablation that carries the argument: inverse gating.** Flip the gate. Keep exactly the tokens OPDVR throws away (student-more-confident on correct trajectories, teacher-more-confident on wrong ones), discard the ones it keeps. Masking ratio and everything else identical.

| Method | Avg |
|---|---|
| OPDVR | 49.1 |
| OPD | 47.8 |
| Inverse-gated | 44.6 |

The ordering OPDVR > OPD > Inverse-gated holds on **all six** benchmarks and holds monotonically through training on the training-set accuracy reward curve. This rules out "the gain is just from throwing away half the tokens" — throwing away the *other* half is strictly worse than throwing away none.

The second, honest finding here: inverse-gated still improves over the raw student (44.6 vs 42.0). So even sign-scrambled teacher guidance carries useful information. The gate is not doing all the work; it is recovering a decent chunk of the loss.

**Training dynamics.** They tracked entropy, response length, and the fraction of tokens the gate zeroes.

- Entropy and length are *not* stable signals. Same-arch: entropy drifts up 0.33 → 0.40, response length balloons $\sim$1.6k → 6.7k tokens. Cross-arch: entropy *collapses* from $\sim$2.0, length stays flat. These depend on the teacher–student pair, not the objective.
- The zero-gated ratio *is* stable: $\approx$0.48–0.50 for the 4B student, $\approx$0.40–0.44 for the 1.7B one, throughout training. It never degenerates to 0 or 1.

That last number is the striking one. **Roughly half of all sampled tokens in plain OPD are pushing against the verifier**, and this stays true from the first step to the last. Standard OPD is spending half its gradient budget fighting itself.

**What did not work / was not tried:** top-64 OPD is roughly a wash with sampled-token OPD (47.4 vs 47.8 same-arch, 21.7 vs 20.9 cross-arch) — more vocabulary coverage buys nothing here, and OPDVR only exists for the sampled-token form. There is no ablation on how to gate *full-vocabulary* OPD; the RLVR reinterpretation depends on the one-sample form.

## Worth Remembering

- The whole method is `R * relu(R * (logp_teacher - logp_student))` in place of `logp_teacher - logp_student`. If you already run sampled-token OPD, this is a one-line change with a verifier bolted on.
- The appendix has a clean toy argument for why OPDVR can *beat the teacher*, which OPD structurally cannot. One decision point, teacher puts $p<\tfrac12$ on the correct token, student already puts $q_0>p$. OPD's optimum is $q=p$ (reverse KL is minimised at exact match) so it drags the student *down* to teacher level: $J_{\text{OPD}} = 2p-1 = J_{\text{teacher}}$. Under OPDVR both gates fire zero — the update vanishes and the student stays at $q_0$, giving $J = 2q_0 - 1 > 2p-1$. The gate acts as a ratchet: teacher pulls you up but never pulls you down.
- Corollary worth internalising: **OPDVR does not simply preserve the teacher's distribution.** It deliberately refuses to match it where matching would hurt. The reverse-KL interpretation is gone; this is no longer minimising a divergence.
- Caveat: you need a teacher that runs alongside training (forward passes on every rollout token) *and* a verifier. So this costs OPD's compute plus RLVR's verification. Only for domains with automatic checkers — maths, code.
- The teacher in both settings was itself produced by GRPO on the same/similar data. So this is closer to self-distillation from a stronger checkpoint than to distillation from a genuinely different model family. Whether the gate helps when the teacher has a different tokenizer or a very different distribution is untested.
- MATH500 regressing while AIME improves is a pattern worth watching: gating away "the student is already confident and right" tokens means the model stops rehearsing easy behaviour it has mastered. Possibly fine, possibly a slow forgetting risk over long runs.
- Open question: only the *sign* of $\hat A_{i,t}$ is used in GRPD. Magnitude information from the group-relative advantage is thrown away and replaced by the log-ratio. No ablation on whether keeping both (e.g. $|\hat A|\cdot\text{ReLU}(\dots)$) would help.
- Related-work landscape is crowded — a dozen 2026 papers combining OPD and RLVR via weighting, sign-based switching, or sample filtering. The selling point here is purely that there is no extra hyperparameter to tune.

## Links

Related: [[Distillation]] · [[Distilling the Knowledge in a Neural Network]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[SecOPD- Mitigating Adaptive Prompt Injections by On-Policy Distillation]] · [[KL Divergence]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[Proximal Policy Optimization Algorithms]] · [[Trust Region Policy Optimization (TRPO)]] · [[High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)]] · [[Training language models to follow instructions with human feedback]] · [[Direct Preference Optimization (DPO)]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Cross Entropy]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Markov Decision Process]] · [[Uncertainty]]

New topics worth writing: RLVR (reinforcement learning with verifiable rewards), GRPO / group-relative advantage estimation, DAPO, reverse KL vs forward KL in distillation, credit assignment in long-horizon LLM rollouts, stop-gradient rewards and the log-derivative trick, DeepSeek-R1 training recipe, Verl RL framework, AIME/AMC math benchmarks and avg@k evaluation
