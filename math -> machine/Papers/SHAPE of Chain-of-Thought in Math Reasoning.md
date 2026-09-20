---
title: "SHAPE of Chain-of-Thought in Math Reasoning"
authors: ["Jonghyun Song", "Sangjun Song", "Minjae Oh", "Haesung Pyun", "Sungsik Lee", "Yohan Jo"]
year: 2026
arxiv: "2608.28600"
url: https://arxiv.org/abs/2608.28600
priority: Good-To-Read
read_on: 2026-09-05
tags: [paper, llm, rl]
---
## The Core Idea

Everyone measures chain-of-thought (CoT) by whether the final number is right, or by crude proxies: how long the trace is, how many times it says "wait", whether it looks like "planning" or "verification". None of those say anything *mathematical*. They tell you the model paused; they do not tell you the model decided to work backwards from the goal, or gave up on algebra and started plugging in numbers.

SHAPE borrows two ideas from **mathematics education research** — a field that has spent 80 years watching students solve problems out loud — and applies them to LLM traces.

> [!NOTE] Heuristic ^heuristic
> A purposeful mathematical *action*: simplify the problem, introduce a variable, work backward from the answer, argue by contradiction, try small cases, check the result. Pólya's vocabulary from *How to Solve It* (1945), consolidated here into 11 families.

> [!NOTE] Semantic space ^semantic-space
> The *interpretation* the solver is currently operating under — which objects, goals, and constraints they think they are dealing with. "This is a system of linear equations" is one space. "This is a counting problem, let me enumerate" is a different space. The solver rarely says which one they are in; you infer it from the actions.

So a CoT trace becomes two aligned sequences: what actions were taken, and under which framing. From that you can ask questions nobody could ask before — does the model *commit* to one interpretation and drill down, or does it thrash between five framings without finishing any?

Three findings fall out. (1) Heuristic frequencies predict correctness better than length or "wait"-counting. (2) Correct traces stay inside **fewer** semantic spaces; wrong traces scatter and revisit. (3) RL post-training does not teach new heuristics — it squeezes the model into a narrow subset of the heuristics the base model already had, a strategy-level version of [[Mode Collapse|mode-seeking]].

And then the payoff: just *listing the 11 heuristics in the rollout prompt* during RL training raises accuracy substantially. No new reward, no new data.

## The Methodology

**The annotation pipeline** (three stages, all LLM-driven):

1. **Content-unit segmentation.** Split the CoT into the largest contiguous spans that carry one coherent mathematical move. 8,334 sentences collapsed into 1,598 units in their gold set.
2. **Heuristic tagging.** Multi-label. A unit can be both "try small cases" (H9a) and "verify by a different derivation" (H11b). Units with no strategic content get non-heuristic labels N1–N4 (repetition, routine computation, off-topic, final answer). Note the distinction is *functional*: a line full of algebra is N2 if the plan was already fixed and it is just grinding.
3. **Semantic-space tracking.** A small state machine. Only units carrying a *representation-changing* heuristic (H1, H2, H3, H5, H8, H11) can trigger a change. The annotator model sees the unit plus a memory buffer $\mathcal{M}$ of spaces seen so far and outputs one of `New` / `Return` / `Maintain`; on `Return` it also picks which past space ID. Everything else inherits the current space.

**The gold set.** 48 traces (6 MATH-Perturb problems × original+hard × 4 models). Four authors, one of them a maths-education graduate researcher, annotated and argued to consensus. Candidate annotators were then scored against it: Grok-4.1-Fast got 76.98 weighted F1 / 65.04 macro F1; Qwen3.5-27B got 70.44 / 61.36 and is the open-weight workhorse used for most experiments. Per-class Cohen's kappa runs 0.42 (H2, reinterpretation) to 0.67 (H7, contradiction) — respectable for interpretive coding, not great.

**The metrics.** Let $\mathbf{S} = (s_1,\dots,s_T)$ be the space label per unit and $H_t$ the heuristic set for unit $t$. Two distributions over heuristic *mass*:

- **Space distribution** $q(i) = \sum_t \mathbf{1}[s_t = i]\,|H_t| \big/ \sum_t |H_t|$ — how much effort landed in each distinct space, merging revisits.
- **Segment distribution** $p(k)$ — same thing but per *contiguous run*, so a space visited three times counts as three segments.

Each is summarised by its **effective number**, the exponential of its [[Cross Entropy|entropy]] (a Hill number — "how many spaces are meaningfully in play"):

$$N_{\text{space}}^{\text{eff}} = \exp\left(-\sum_i q(i)\log q(i)\right), \qquad N_{\text{trans}}^{\text{eff}} = \exp\left(-\sum_k p(k)\log p(k)\right) - 1$$

The **transition ratio** $\rho = N_{\text{trans}}^{\text{eff}} / N_{\text{space}}^{\text{eff}}$ is transitions per space — high $\rho$ with low $N_{\text{space}}^{\text{eff}}$ means the model is bouncing between two framings over and over. Finally the **heuristic frequency distribution** $u(h)$, the share of each heuristic across the trace.

Worked example from Figure 1: $\mathbf{S}=(1,2,1)$ with heuristic counts $(1,2,1)$ gives $q=(1/2,1/2)$, $N_{\text{space}}^{\text{eff}}=2$; $p=(1/4,1/2,1/4)$ gives $N_{\text{trans}}^{\text{eff}}\approx 1.83$.

**Heuristic-Augmented GRPO.** Qwen3-1.7B-Base, trained with GRPO on the MATH train split, 200 steps, ~3 hours on 2×B200. Batch 32, 4 rollouts per prompt, AdamW, lr $1\times10^{-6}$, 10 warmup steps, max response 2048 tokens, temperature 1.0. Two variants that differ **only in the rollout prompt**:

- *Plan-GRPO*: asks for a `[Plan]` block with Goal and Sub-goals before solving.
- *HA-Plan-GRPO*: same plan block, but the prompt also lists the 11 heuristics in plain English ("Work Backward from the Goal", "Wishful Thinking: Simplify Temporarily", …) and asks each step to name the heuristic it is using and why. The prompt explicitly says it is not a checklist and not to force irrelevant heuristics in.

Same reward, same verifier, same optimiser, same *evaluation* prompt. Only the training rollout prompt differs.

## Ablation Studies and Experiments

**Does heuristic vocabulary carry correctness signal?** Logistic regression ([[Regression Analysis|logistic regression]], $\ell_1$ or $\ell_2$ chosen by inner CV), 5-fold stratified, on 100 Omni-MATH problems × 15 models. AUROC:

| Features | AUROC | # feat |
|---|---|---|
| CoT length | 0.504 ± 0.03 | 1 |
| Length + reasoning-token count/proportion | 0.503 ± 0.03 | 3 |
| Self-revision markers ("wait", "aha") | 0.618 ± 0.03 | 3 |
| ThinkARM episode ratios | 0.618 ± 0.02 | 8 |
| SHAPE heuristics (H1–H11) | 0.653 ± 0.02 | 11 |
| SHAPE heuristics + non-heuristic | **0.664 ± 0.02** | 12 |

Length is literally chance. Adding reasoning-token statistics makes it slightly *worse*. This is the cleanest negative result in the paper: the single most-cited CoT feature carries no correctness signal at all here.

**Structure of correct vs incorrect traces** (Table 2, 15 models). Across almost every model, incorrect traces have higher $N_{\text{space}}^{\text{eff}}$, higher $N_{\text{trans}}^{\text{eff}}$, and higher $\rho$. QwQ-32B is the starkest: correct traces $N_{\text{space}}^{\text{eff}}=1.74$, $\rho=0.32$; incorrect $2.74$ and $0.60$. DeepSeek-R1: $1.72/0.34$ correct vs $2.47/0.51$ incorrect. Reasoning models as a class sit higher on both axes ($N_{\text{space}}^{\text{eff}}$ 1.81–2.53, $\rho$ 0.40–0.51) than instruction-tuned ones (1.37–1.71, 0.19–0.32) — long thinking is not just longer, it traverses differently.

The one **exception is GPT-4o**, where correct traces have *higher* $N_{\text{space}}^{\text{eff}}$ (1.68) than incorrect (1.40), and it has the worst accuracy (.19) in the non-reasoning group. Worth noting: the direction is not universal.

**Perturbation experiment** (MATH-Perturb, 115 problems; *simple* perturbation keeps the solution method, *hard* perturbation changes what method is needed). Pass@1 collapses on hard: Qwen3-32B .92→.76, Nemotron-Cascade-8B .92→.71, Olmo-3-7B-Think-RLVR .97→.76. The interesting part is what the traces do. Jensen–Shannon divergence of heuristic distributions (a symmetric cousin of [[KL Divergence]]) is reliably larger for hard than simple ($\approx.24$–$.25$ vs $.19$–$.21$, one-sided Wilcoxon $p<.05$). $\Delta N_{\text{space}}^{\text{eff}}$ goes *negative* for simple (−.00 to −.06) and *positive* for hard (+.07 to +.14), and $\Delta\rho$ likewise (+.03 to +.11).

So the model is **not** blindly replaying a memorised solution. It notices something changed, opens more framings, and switches more — it just never commits to a productive one. Appendix E rules out the "this only happens late, after it gets stuck" explanation: truncate to the first 5 units and hard-vs-simple JSD is already .19–.21 vs .15–.17, still significant. The divergence starts immediately.

**Does RL narrow the strategy space?** Using Density and Coverage (Naeem et al. 2020) in heuristic-frequency space, $k=3$ nearest neighbours, cosine, base model as reference, successful trajectories only:

| Base → Post-trained | Density | Coverage |
|---|---|---|
| Qwen3-1.7B-Base → GRPO | 1.220 | 0.871 |
| Olmo-3-7B → Think-RL-Zero | 1.250 | 0.707 |
| Olmo-3-7B → Think-RLVR | 1.032 | 0.531 |
| Olmo-3-7B → Qwen3-1.7B-Base (control) | 0.520 | 0.437 |

Density > 1 means post-trained traces pile into the dense core of the base distribution. Coverage < 1 means whole regions of the base's heuristic repertoire are abandoned. The unrelated-models control has both low — so the pattern is genuinely a base→child relationship, not generic overlap. A PCA projection (Figure 3) shows Think-RLVR sitting on the base's peak with the entire left tail empty.

This is the heuristic-level version of "RLVR sharpens rather than expands", which prior work argued from surface-form or answer diversity. Here it is at the level of *what mathematical move the model reaches for*.

**Heuristic-augmented RL** (Qwen3-1.7B-Base, MATH-Perturb test):

| Model | Orig Avg@64 | Orig Pass@64 | Simple Avg@64 | Hard Avg@64 | Hard Pass@64 |
|---|---|---|---|---|---|
| Base | 23.54 | 77.40 | 23.10 | 11.84 | 57.39 |
| + Plan-GRPO | 30.00 | 80.00 | 29.86 | 14.52 | 61.74 |
| + HA-Plan-GRPO | **36.80** | 80.00 | **35.80** | **17.72** | **62.61** |

Avg@64 jumps by ~7 points over Plan-GRPO on every split. But look at Pass@64: 80.00 vs 80.00 on original, 79.13 vs 78.26 on simple. Pass@64 barely moves. The heuristic prompt makes the model *reliably* find solutions it could already occasionally find — it raises the average without meaningfully expanding the reachable set. Which is exactly the "RL sharpens, does not expand" story the paper itself diagnosed one section earlier. The authors call this "preliminary" and do not comment on it.

## Worth Remembering

- **Length is a dead feature.** AUROC 0.504. If you are using CoT length as a correctness proxy or a reward shaping term, this is evidence against it. Self-revision markers do carry signal (0.618) with only three features — a very cheap baseline.
- **"More exploration" is a failure signal, not a success signal.** Incorrect traces open more semantic spaces and bounce between them harder. The transition ratio $\rho$ is arguably the single most useful number here: it isolates *thrashing* from *breadth*. Low $N_{\text{space}}^{\text{eff}}$ with high $\rho$ = alternating between two framings, neither committed to. Plausibly a cheap online overthinking detector.
- **The whole framework rests on an LLM judge.** Grok-4.1-Fast hits 76.98 weighted F1 against a 48-trace consensus gold set. Macro F1 is 65. Rare classes (H7 contradiction, H10 working backward) have few gold instances and unstable estimates. The semantic-space tracker has **no gold standard at all** — the authors say so plainly and only "calibrate the prompt by iterative manual review". Every space-level result inherits that.
- **Interpretation is genuinely contested.** Their own borderline example (Appendix B) has one content unit legitimately tagged H9a *and* H11b *and* H5, resolved by keeping all three. That is honest, but it means heuristic frequency vectors are soft and annotator-dependent.
- **48 traces, 6 problems, 4 models** is the entire validation set. Everything downstream is scaled from that.
- **The RL win is prompt-only.** Same reward, same data, same optimiser. If it replicates, it says that a lot of what post-training struggles to install can be handed over as vocabulary in the rollout prompt — cf. how [[Chain-of-Thought Prompting Elicits Reasoning in LLMs|CoT prompting]] unlocked behaviour already latent in the weights. But it was tested on one 1.7B base model, 200 steps, 2048-token responses. That is a small experiment.
- **Open question the paper leaves.** If RL narrows the heuristic distribution, and narrowing is bad, could you regularise *toward* base heuristic diversity — a [[KL Divergence|KL]] penalty in heuristic-frequency space rather than token space? The diagnosis points there and the paper does not go.
- **Scope.** Maths only. Semantic spaces are inferred from the *stated* trace, so all the CoT-faithfulness caveats apply — if the verbalised reasoning is post-hoc decoration, SHAPE is annotating the decoration.

## Links

Related: [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Mode Collapse]] · [[KL Divergence]] · [[Cross Entropy]] · [[Is Next-Chunk Reasoning RL Really Better than SFT- Revisiting Training Strategies under no-CoT Data]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[Training language models to follow instructions with human feedback]] · [[Proximal Policy Optimization Algorithms]] · [[TTPO- Test-Time Policy Optimization]] · [[Thinking in a Low-Resource Language- What SFT Builds, What RL Fixes, What Accuracy Cannot See]] · [[Regression Analysis]] · [[In Context Learning]]

New topics worth writing: GRPO (Group Relative Policy Optimization), RLVR (reinforcement learning with verifiable rewards), Hill numbers and effective number of species, Jensen–Shannon divergence, Density and Coverage metrics for generative models, MATH-Perturb, Pólya's How to Solve It heuristics, CoT faithfulness, Cohen's kappa and interpretive coding reliability
