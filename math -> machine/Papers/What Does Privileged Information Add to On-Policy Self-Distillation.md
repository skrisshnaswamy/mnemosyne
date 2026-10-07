---
title: "What Does Privileged Information Add to On-Policy Self-Distillation?"
authors: ["Zhang et al."]
year: 2026
arxiv: "2609.20612"
url: https://arxiv.org/abs/2609.20612
priority: Good-To-Read
read_on: 2026-09-27
tags: [paper, llm, vision]
---
## The Core Idea

On-policy self-distillation (OPSD) trains a model using a frozen copy of *itself* as the teacher. The trick that makes it interesting: the teacher gets to see the worked solution, the student does not. The teacher scores the student's own attempts, and the student learns to match those scores. That extra thing the teacher sees — a hint, a sketch, a full solution — is called **privileged information**.

> [!NOTE] Privileged information (PI)
> Information available only at training time, never at inference. The classic framing is Vapnik's: a teacher who sees more can shape a student that sees less. ^privileged-information

The obvious next question, and the one nobody had answered cleanly: **does giving the teacher more of the solution actually help the student more?** A one-line hint versus a full step-by-step trace — is the trace better?

The answer here is mostly **no**, and the reason why is the real contribution.

The paper isolates the effect by building a dataset where **six different views of the reasoning all end in the same verified answer**. So the answer is held fixed; only the *amount and shape of reasoning* varies. Then every view is compared against a **reference-free control**: same setup, same teacher, but no solution shown at all.

The result: for Qwen3-1.7B, the reference-free control captures **most of the gain**. Reference-free gives $+1.80$ points in-domain at step 100; the best view, `Clean Solution`, gives $+3.10$. Even a control that shows the teacher **a different problem's answer** gives $+1.67$.

So where does the improvement come from, if not from the solution?

From a second asymmetry nobody was controlling for. In the standard OPSD recipe, the teacher answers **with thinking enabled** (long chain of thought) while the student's rollouts are generated **with thinking disabled** (short, direct answers). Even with zero privileged information, the teacher is a different distribution from the student. Learning from that gap updates weights that are **shared by both inference modes**. The student answers short questions during training, and its *thinking-enabled* reasoning improves as a side effect.

> [!NOTE] Cross-mode self-distillation
> A thinking-enabled teacher scoring thinking-disabled student prefixes. Same weights, two modes, so the loss is non-zero and informative even without a reference. This is the engine most OPSD gains are actually running on. ^cross-mode-distillation

The supporting evidence is the clincher: gains concentrate on problems the base model **never** solved in four direct-response tries but solves 62% of the time with thinking on. The capability was already there. OPSD improved *access* to it.

The second finding, which matters more for anyone building this: **the reference's value depends on how the student answers during training, not on how complete the solution is**. Keep the problems, the references, the teacher and the evaluation all fixed, and swap short direct-response rollouts for long thinking-enabled rollouts — gains become losses. In both model families. $-7.60$ points at step 50, pooled.

The takeaway the authors state plainly: judge privileged information by what it adds *to the student*, not by how much of the solution it hands the teacher.

## The Methodology

### The loss

Student $\pi_\theta$ and teacher $\pi_{\bar\theta}$ start from the same backbone. The teacher is frozen. The student generates $y = (y_1,\dots,y_L)$ from the problem alone under rollout mode $m_S$.

At prefix $s_t = (x, y_{<t})$:

- Student: $p^S_t(a) = \pi_\theta(a \mid x, y_{<t}; m_S)$
- Teacher: $p^T_{v,t}(a) = \pi_{\bar\theta}(a \mid x, z_v(x), y_{<t}; m_T, \tau)$

where $z_v(x)$ is the privileged reference under view $v$, $m_T$ the teacher's reasoning mode, $\tau$ the scoring temperature.

The objective:

$$\mathcal{L}_v(\theta; m_S) = \mathbb{E}_{x, y \sim \pi_\theta(\cdot \mid x; m_S)}\left[\frac{1}{|I(y)|}\sum_{t \in I(y)} D_{\mathrm{gKL}}\left(p^T_{v,t} \,\|\, p^S_t\right)\right]$$

$D_{\mathrm{gKL}}$ is teacher-to-student forward [[KL Divergence|KL]] with **each vocabulary term clipped from above at $0.05$** before summing. Teacher distribution and sampled completion are held fixed when differentiating — this is an [[On-Policy vs Off-Policy|on-policy]] scheme where the student's own samples define the states, but the gradient does not flow through sampling.

Note the implementation detail the authors flag: the released OPSD code uses the generalised divergence at $\beta = 0$ with a $0.05$ clip, not the $\beta = 0.5$ the original paper describes. They followed the code.

Forward KL here means the teacher is $P$ and the student is $Q$, so the pressure is **mode-covering** — see [[KL Divergence#Forward KL — $D_{KL}(P \parallel Q)$, "**mode-covering**"|forward KL]].

The key structural point: with identical prompt, thinking mode, initial weights and temperature, a reference-free teacher would *be* the student and the loss would be zero at step 0. **Privileged information supplies the initial discrepancy.** But in the reference-free control here the modes differ, so there is still a non-zero starting gap. That control is therefore not "no teacher signal" — it is cross-mode distillation.

### AMPLE-Math

5,319 maths problems from OpenThoughts-114k. Six views per problem, one shared verified answer. Mean rendered length in Qwen3 tokens:

| View | Tokens | What it is |
|---|---|---|
| `Answer Only` | 13 | no reasoning body at all |
| `Gist` | 76 | one paragraph, the central method |
| `Key Points` | 284 | ordered steps |
| `Clean Solution` | 526 | polished worked solution |
| `Summary` | 841 | narrative retelling |
| `Full Trace` | 4,916 | the complete original reasoning trace |

Every view ends in the same canonical answer section. Rendered as `<think>` body `</think><answer>` answer `</answer>`, exactly one boxed answer.

`Gist`, `Key Points` and `Summary` are generated from the intact trace by Qwen3.6-35B-A3B-FP8, up to five seeded attempts at temperatures $0.0, 0.3, 0.6, 0.9, 1.2$, then surface-validated and checked for **source fidelity** by the same model under a separate prompt — no unsupported claims, no backward-constructed reasoning, no confidence upgrades. The honest caveat the authors state: generator and reviewer are the same model, so the review may share the generator's blind spots.

A problem only enters the released set if all six views exist, all pass both checks, and body lengths satisfy `Answer Only` $<$ `Gist` $<$ `Key Points` $<$ `Summary` $<$ `Full Trace`. `Clean Solution` sits off that spine. 5,319 of 5,480 candidates survive; $5{,}319 \times 6 = 31{,}914$ records.

### The splits — and why they are model-relative

Difficulty is annotated two ways.

*Absolute*: 8–16 problem-only samples per question from three evaluator models, then a **binomial Rasch model** fitting evaluator abilities and item difficulties jointly, with a zero-centred $L_2$ penalty of $0.5$ so all-correct and all-wrong items stay finite.

*Model-relative*, and this is the one used for splits: four unprivileged direct-response samples per problem from the **frozen student base** define three bands — 3–4 correct, 1–2 correct, 0 correct. A zero-success problem is only eligible if at least one structured-PI teacher view gets it right. Equal quotas give 1,536 / 192 / 384 for train / dev / test.

SmolLM3 reuses the dev and test **problem IDs** but rebuilds its training split from its own direct-response outcomes.

### Training

- Qwen3-1.7B, [[LoRA]] adapters, $r = 64$, $\alpha = 128$, dropout $0$, on Q/K/V/O/gate/up/down projections.
- Effective batch 32, LR $5 \times 10^{-6}$, [[Decoupled Weight Decay Regularization (AdamW)|AdamW]] with $\beta_2 = 0.999$.
- Rollout and loss temperatures both 1.1. 100 optimizer steps, checkpoints at 0,1,2,5,10,15,20,25,50,75,100.
- Direct-response rollouts capped at 1,024 tokens, **all supervised**. Thinking-enabled rollouts run to 20,764 tokens but keep the **same first-1,024-token loss window**. That mismatch becomes an experiment later.
- SmolLM3-3B: same recipe, 24,536-token thinking rollout cap, 40,960 training context, still a 1,024-token loss window.
- Teacher is always the frozen thinking-enabled backbone with the student adapter switched off. Student rollouts and all evaluation get no PI.

### Evaluation

Avg@4 in-domain: four thinking-enabled samples per problem, temperature 1.0, top-$p = 0.95$, top-$k = 20$, 16,384-token initial budget. **Every capped response is regenerated from scratch** with the same seed at up to 32,512 tokens — the authors call the graded result *effective correctness*. External: 12 samples per problem on AIME 2024, AIME 2025, HMMT Feb 2025 (30 problems each), 38,912 generated tokens.

The inferential unit is the **problem**. 95% intervals from 10,000 problem-cluster [[Monte Carlo Methods|bootstrap]] resamples, sharing one resampled index array across compared conditions so contrasts are paired. Holm correction applied within stated families. They are careful to distinguish "interval excludes zero, unadjusted" from "survives Holm", and they say outright that intervals including zero do not establish equivalence.

### The profiling instrument

Before training, they measure what the teacher's scores *push on*. For a completion $y$, over the longest prefix $J(y)$ that fits every teacher context:

$$\Delta_v(y) = \frac{1}{|J(y)|}\sum_{t \in J(y)}\left[\log p^T_{v,t}(y_t) - \log p^S_t(y_t)\right]$$

Three statistics fall out:

**Correctness alignment** $C_v$ — within each problem that has both correct and wrong samples, mean shift on correct minus mean shift on wrong, then average problems equally:

$$C_v(x) = \frac{\sum_{y \in \mathcal{Y}^+_x}\Delta_v(y)}{|\mathcal{Y}^+_x|} - \frac{\sum_{y \in \mathcal{Y}^-_x}\Delta_v(y)}{|\mathcal{Y}^-_x|}$$

Positive means the teacher raises likelihood more on right answers than wrong ones.

**Correction pressure** $F_v$ — the mean shift at sampled reconsideration tokens (`wait`, `but`, `however`, `maybe`, `actually`, `recheck`, `check`, `verify`, `reconsider`, `mistake`, `instead`, matched by substring after lowercasing and stripping tokenizer space markers). Negative means the teacher is pushing those tokens *down*.

**Temporal KL allocation** — share of unclipped full-vocabulary $D_{\mathrm{KL}}(p^T_{v,t} \| p^S_t)$ in each quarter of the scored span.

Scale: 512 problems, four fixed no-PI trajectories each, 18,467,995 distinct token positions, seven contexts (six views plus no-PI), 129,275,965 scored token–condition evaluations.

## Ablation Studies and Experiments

### More reasoning does not buy more gain

Qwen3-1.7B, direct-response training, thinking-enabled evaluation, in-domain $\Delta$Avg@4 at step 100:

| Reference | PI tokens | $\Delta$ vs base | $\Delta$ vs reference-free |
|---|---|---|---|
| Reference-free | 0 | $+1.80\,[0.20, 3.47]$ | — |
| Wrong answer | 13 | $+1.67\,[0.07, 3.32]$ | — |
| `Answer Only` | 13 | $+2.41\,[0.86, 3.99]$ | $+0.35\,[-1.04, 1.71]$ |
| `Gist` | 76 | $+2.15\,[0.26, 4.04]$ | $+0.35\,[-1.13, 1.80]$ |
| `Key Points` | 284 | $+2.80\,[0.72, 4.88]$ | $+1.00\,[-0.48, 2.45]$ |
| `Clean Solution` | 526 | $+3.10\,[1.54, 4.73]$ | $+1.78\,[0.24, 3.30]$ |
| `Summary` | 841 | $+2.54\,[0.71, 4.43]$ | $+0.74\,[-0.69, 2.17]$ |
| `Full Trace` | 4,916 | $+2.38\,[0.91, 3.86]$ | $+0.61\,[-0.87, 2.11]$ |

Reference tokens span a factor of 378. The gains span about 1.3 points and do not order by length. `Full Trace` — 4,916 tokens of complete reasoning — beats reference-free by $0.6$ points with the interval straddling zero.

`Clean Solution` is the only view with any signal: three-seed advantage over reference-free of $+1.30\,[0.20, 2.41]$ at step 100. **That excludes zero unadjusted ($p = 0.024$) but does not survive Holm across the six views ($p = 0.15$).** The authors report it as modest, view-specific evidence, not a finding. The seed-level Welch check agrees: $+1.30\,[0.27, 2.33]$.

Externally (pooled Avg@12 over 90 problems, step 50), **no view's difference from reference-free lies above zero**. `Full Trace` is the only one with an interval excluding zero and it is on the wrong side: $-1.78\,[-3.43, -0.21]$, which does not survive correction either.

### Where the gains live

Step 100, Qwen, split by how many of four base direct-response tries were right:

| Configuration | 0/4 correct | 1–2/4 | 3–4/4 |
|---|---|---|---|
| Reference-free | $+3.97$ | $+0.98$ | $+0.46$ |
| `Answer Only` | $+4.69$ | $+2.34$ | $+0.20$ |
| `Clean Solution` | $+7.03$ | $+1.89$ | $+0.39$ |
| `Full Trace` | $+5.66$ | $+1.51$ | $-0.05$ |
| **Frozen base, thinking on** | **61.91%** | 83.98% | 94.73% |

Read the bottom row. The "never solved" group is never-solved *directly* — with thinking enabled the base already gets 62% of them. That is where all the movement is, with and without a reference. `Clean Solution`'s extra benefit is also largest here: $+3.06\,[0.46, 5.79]$ over reference-free.

The separate validation on the original OPSD training data, evaluated externally, tells the same story from a different angle. Grouping the 90 benchmark problems by the base's 12 thinking-enabled samples: never-solved $+0.62$, intermittently solved $+10.03$, always-solved $-0.60$. **The gain is entirely in the "sometimes gets it" band.** That is what improved *access* looks like, not new capability.

Also from that validation, and worth pausing on: pooled Majority@12 rose $7.78$ points while **Pass@12 fell $7.78$ points**. The model got more reliable at what it could already sometimes do, and lost coverage. Compare [[GRPO#^sharpening-not-expanding|sharpening, not expanding]].

### SmolLM3 disagrees — usefully

SmolLM3-3B is where a reference does earn its place. Thinking-enabled Avg@4 gains over base (three seeds):

| Student | Step 50 | Step 100 |
|---|---|---|
| Reference-free | $+0.59\,[-1.06, 2.26]$ | $-5.25\,[-7.20, -3.32]$ |
| `Answer Only` | $+0.87\,[-0.74, 2.50]$ | $-5.03\,[-6.94, -3.17]$ |
| `Full Trace` | $+2.58\,[1.06, 4.14]$ | $-2.19\,[-4.08, -0.37]$ |

`Full Trace` minus reference-free: $+2.00\,[0.82, 3.17]$ at step 50, $+3.06\,[1.71, 4.41]$ at step 100.

Two distinct contributions in one column. At step 50 the reference **adds**. By step 100 every configuration is below base, and the reference **loses less**. A gain-over-base table cannot tell those apart.

The failure mode behind step 100 is visible in the completion diagnostics. Reference-free and `Answer Only` direct-response no-answer rates — *after* regeneration — go from 20.2% and 22.0% at step 50 to 42.0% and 39.4% at step 100. The students stop producing a boxed answer. `Full Trace` holds at 23.5%. So part of what the dense reference buys is **format stability**, not reasoning.

### Same student, different evaluation mode, opposite ranking

This is the sharpest inconsistency in the paper.

Qwen step 100, `Answer Only` minus `Full Trace`, four seeds:

| Evaluation | Contrast |
|---|---|
| Thinking-enabled | $+0.03\,[-0.91, 0.98]$ |
| Direct-response | $+5.32\,[3.47, 7.16]$ |
| **Interaction** | $+5.29\,[3.26, 7.32]$ |

Reference-free minus `Full Trace` under direct-response evaluation: $+10.81$.

SmolLM3 flips even harder — step-50 `Full Trace` advantage is $+2.00$ with thinking on and $-10.63$ answering directly.

And the direct-response winners win partly by **writing more**. Grading only the first 4,096 saved tokens of the same stored responses reverses the ordering:

| Prefix graded | `Answer Only` $-$ `Full Trace` | Reference-free $-$ `Full Trace` |
|---|---|---|
| 4K | $-3.01\,[-4.85, -1.22]$ | $-3.99\,[-6.26, -1.78]$ |
| 8K | $+3.29\,[1.46, 5.06]$ | $+3.86\,[1.75, 5.98]$ |
| 16K | $+5.09\,[3.26, 6.92]$ | $+9.51\,[7.41, 11.62]$ |
| Full | $+5.32\,[3.47, 7.16]$ | $+10.81\,[8.60, 13.00]$ |

Reference choice changes **how the student answers**, not just how often it is right. Any conclusion about which reference is best is conditional on the evaluation mode *and* the length budget.

### The trajectory reversal — the headline negative result

Keep the problems, references, teacher, loss window and evaluation fixed. Only swap direct-response rollouts for thinking-enabled rollouts during training. Thinking-enabled evaluation throughout, Qwen:

| View | Step | DR-trained | TH-trained | Gap |
|---|---|---|---|---|
| `Answer Only` | 50 | 83.59 | 75.20 | $-8.40\,[-11.00, -5.92]$ |
| `Clean Solution` | 50 | 84.64 | 77.54 | $-7.10\,[-9.38, -4.95]$ |
| `Full Trace` | 50 | 82.42 | 75.13 | $-7.29\,[-9.44, -5.14]$ |
| Pooled | 50 | 83.55 | 75.95 | $-7.60\,[-9.31, -5.95]$ |
| Pooled | 100 | 82.92 | 75.78 | $-7.14\,[-8.66, -5.64]$ |

Base is 80.21%. **All nine gaps negative**; only `Clean Solution` at step 25 includes zero. SmolLM3 replicates at $-6.84\,[-8.50, -5.18]$ (step 50, two seeds). Externally, Qwen `Full Trace` at step 50: direct-response-trained $+2.87$ over base, thinking-enabled-trained $-6.57$.

The external result gets a proper robustness check, which I found the most credible part of the paper. The thinking-trained student hits the 38,912-token cap in 79 of 1,080 samples versus 2 for the direct-trained one. Excluding capped samples: gap still $-8.16\,[-12.16, -4.25]$. Scoring **every capped sample as correct** (none was): $-2.13\,[-6.85, +2.69]$, which finally includes zero. So truncation accounts for some of it, but the effect survives the honest exclusion.

Caveat the authors state: switching to thinking rollouts changes reasoning mode, horizon, *and* the fraction of the response covered by the 1,024-token loss window, all at once. Three things move together. Which leads to the next experiment.

### Teacher profiles — the diagnosis, which then mostly fails to act

**Correctness alignment is not a property of the reference.** $C_v$ in nats/token:

| View | Base, thinking-on prefixes | Base, direct prefixes | Own step-50 direct prefixes |
|---|---|---|---|
| `Answer Only` | $+0.0065$ | $+0.0041$ | $+0.0013$ |
| `Clean Solution` | $-0.0005$ | $+0.0093$ | $-0.0003$ |
| `Full Trace` | $-0.0123$ | $+0.0131$ | $+0.0099$ |

`Full Trace` flips sign between prefix modes — on direct-response prefixes it favours correct answers, on thinking-enabled prefixes it favours wrong ones. After direct-response training, five views collapse toward zero; `Full Trace` keeps about three-quarters of its value.

Two honesty notes the authors add: denser references induce larger shifts overall, which raw $C_v$ does not separate from correctness *selectivity*; and restricting thinking-mode profiles to their first 1,024 tokens weakens the differences. Also, the teacher assigns *lower* likelihood than the student to both correct and wrong responses on average — $C_v$ is a difference between two negative numbers, not evidence the teacher likes correct answers.

**The reconsideration hypothesis does not discriminate.** Kaur et al. proposed suppressed self-checking as the cause of thinking-model degradation. Here, **every view lowers reconsideration-marker probability in both prefix modes**. Same sign everywhere. It cannot explain two configurations with opposite transfer outcomes.

Then the intervention. Exclude or downweight ($\times 0.1$) loss at sampled marker positions, direct-response training, step 100:

| Intervention | $\Delta$ vs matched | $\Delta$ markers used |
|---|---|---|
| `Full Trace` exclude | $-0.65\,[-2.41, 1.17]$ | $+0.4\%$ |
| `Full Trace` downweight | $-0.78\,[-2.34, 0.78]$ | $-0.3\%$ |
| `Clean Solution` exclude | $-0.72\,[-2.21, 0.78]$ | $+0.3\%$ |
| `Clean Solution` downweight | $-0.72\,[-2.34, 0.91]$ | $+1.1\%$ |

**Nothing moves.** Not accuracy, not marker usage. Removing a penalty on a token is not the same as encouraging the behaviour that token stands for. Clean negative result.

### The one that would have been a paper on its own

Diagnostic KL is concentrated in the **first quarter** of the scored span. The inherited 1,024-token loss window covers a whole direct-response rollout but only the opening of a 20,000-token thinking rollout. Obvious fix: widen the window. `First-4K` extends to 4,096 tokens; `Distributed-1K` places four 256-token windows at fractions $0.125, 0.375, 0.625, 0.875$ of the reasoning span.

Both beat development-selected `Early-1K` by about **four points**.

Except they stop at step 25, and `Early-1K` was selected at step 50. Raw Avg@4:

| Support | Step 25 | Step 50 | Step 100 |
|---|---|---|---|
| `Early-1K` | 78.26 | 75.13 | 72.79 |
| `First-4K` | 79.23 | 74.28 | 71.22 |
| `Distributed-1K` | 79.23 | 75.65 | 71.48 |

At matched step 25 the advantage is **under one point**, and every paired same-step interval includes zero. All three configurations are simply best at step 25 and decline from there.

The four-point "gain from better loss coverage" was **four points of early stopping**. Citing Dodge et al. and Bouthillier et al. on exactly this hazard.

### What *did* depend on reference content

Reference-free gains do not make the reference text irrelevant. Replace `Clean Solution` or `Key Points` with a **length-matched view from a different problem**, including that problem's answer:

| Intervention | $\Delta$ vs matched genuine view |
|---|---|
| `Clean Solution` → other problem | $\mathbf{-1.95}\,[-3.78, -0.20]$ |
| `Key Points` → other problem | $\mathbf{-2.15}\,[-3.97, -0.39]$ |

Both survive within-family Holm. So relevance matters roughly two points, while *completeness* is worth approximately nothing. The signal in a reference is that it is about **this** problem, not that it spells out the answer.

Truncating `Full Trace` to shorter lengths (keeping the opening, dropping later reasoning and the answer section) costs nothing under thinking-enabled evaluation — $-0.20$ to $-1.11$, all intervals spanning zero — but **gains 8–11 points under direct-response evaluation**, with mode interactions of $+8.9$ to $+11.9$. Compression shifts *which mode* benefits.

Prompt template swaps to the original OPSD wording: `Clean Solution` $+0.13$, `Full Trace` $-1.63$. Both include zero.

### The toy model that explains why profiles cannot predict transfer

Appendix D.4 has a two-problem construction worth its own note. Two teacher contexts $A$ and $B$ with correct-answer probabilities $(0.54, 0.51)$ and $(0.51, 0.54)$. **Identical** expected accuracy ($0.525$), **identical** mean teacher–student KL, **identical** correctness alignment ($C_A = C_B \approx 0.100$).

The student shares one parameter across both problems: $p_\theta(1 \mid x_1) = \sigma(\theta)$, $p_\theta(1 \mid x_2) = \sigma(-2\theta)$.

One gradient step of size $\eta$: accuracy changes by $-0.00125\eta$ under $A$ and $+0.004375\eta$ under $B$. **Opposite signs.** All vocabulary terms sit below the $0.05$ clip near init, so the clipped loss behaves the same.

The cause is [[Backpropagation|shared parameters]]: fitting $x_1$ moves $x_2$ the other way, with twice the sensitivity. No aggregate token-level diagnostic sees that. It is a clean argument for why "profile the teacher, then fix the loss" is not a valid research loop on its own.

## Worth Remembering

**Reference-free is not a null control.** This is the single easiest thing to misread in any OPSD paper. If the teacher thinks and the student does not, removing privileged information leaves a real training signal. Whenever you see an OPSD ablation reporting "gains without a reference", check whether the modes were matched. If they were not, the baseline is doing work.

**A privileged reference seems to buy access, not capability.** Every piece of evidence points at redistributing probability mass over solution paths the model already had: gains concentrate on sometimes-solved problems, Majority@12 up 7.78 while Pass@12 down 7.78, and the near-total insensitivity to reference length. The same shape as [[GRPO]] and [[Locked at the Entrance, Open Inside- Where RLVR Narrows the Solution Space]].

**Relevance ≈ 2 points, completeness ≈ 0.** Length-matched other-problem references cost about two points; a 4,916-token full trace beats a 13-token bare answer by nothing measurable. If you were planning to spend money curating rich rationales for a PI teacher, this is an argument to spend it on coverage instead.

**Design the reference and the rollout mode together.** The reversal result is the paper's most transferable claim: identical reference, identical teacher, gain becomes a 7-point loss when rollouts change. Reference choice is not separable from the student's training trajectory. The authors' own suggested follow-up is references that **adapt to the student's evolving attempts** rather than being fixed throughout training.

**Match your checkpoints before believing any loss-design result.** Four points of "better supervision placement" evaporated to under one point at matched steps. Development selection across $\{25, 50, 75, 100\}$ silently smuggled in an early-stopping advantage. Cheap discipline, large effect.

**Your evaluation mode determines your ranking.** Not "affects" — determines. The SmolLM3 `Full Trace` verdict swings from $+2.00$ to $-10.63$ depending only on whether thinking is on. Same weights. Report both, or state the mode in the claim.

**Grade prefixes, not just final answers.** The 4K/8K/16K/full table shows that "better" configurations partly just write longer. This is a re-usable diagnostic and it costs nothing — decode stored token IDs at several cutoffs and re-grade.

**Statistical practice worth copying.** Problem as the inferential unit, 10,000 problem-cluster bootstraps with a shared resample array so contrasts are paired, explicit separation of "unadjusted interval excludes zero" from "survives Holm", seed-level $t$ intervals reported *alongside* problem bootstraps because they condition on different things, and the flat statement that intervals including zero do not establish equivalence. Most papers do about two of these six.

### Limitations, stated by the authors

- **Short-run [[LoRA]] only.** 100 optimizer steps, $r = 64$, adapters not full fine-tuning, two small models (1.7B and 3B). Whether any of this survives full-parameter training at scale is untested.
- **Mathematics with a verifiable boxed answer.** No claim beyond that.
- **Dataset self-review.** Generator and fidelity reviewer are the same model, so shared blind spots are possible. Fidelity review preserves the source's reasoning and uncertainty; it does not certify the proof.
- **Training data overlap.** 202 of 384 test problems (52.6%) also appear in the released OPSD training data, because both draw on OpenThoughts. Only the original-data implementation check trains on those, and it is evaluated externally. All AMPLE-Math comparisons use disjoint splits. External benchmarks cleared a shingle scan — max 8-gram containment $0.25$, from shared contest phrasing.
- **Marker detection is lexical.** Substring matching against eleven words finds sampled marker tokens, not all semantically corrective continuations.
- **Paired cross-mode profiles use a different population** from the per-mode averages: 116 eligible problems for thinking-mode alignment, 200 for direct, only 48 in the intersection.

### Open questions

Does the cross-mode story hold if you **match** teacher and student modes and add PI back? That would isolate the reference cleanly, and it is the obvious missing cell in this design.

What happens past 100 steps? Both families are *declining* by step 100 in several configurations. The whole paper lives inside a regime where the models are about to get worse.

Is the SmolLM3 `Full Trace` advantage actually about reasoning, or about not forgetting how to emit `\boxed{}`? The no-answer-rate table (23.5% versus 42.0%) makes format preservation a live alternative explanation for its entire margin.

## Links

Related: [[Distilling the Knowledge in a Neural Network]] · [[Distillation]] · [[KL Divergence]] · [[LoRA]] · [[GRPO]] · [[Chain of Thought]] · [[Locked at the Entrance, Open Inside- Where RLVR Narrows the Solution Space]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[On-policy Distillation with Verifiable Reward]] · [[Negative Self-Distillation- Learning to Reason by Avoiding Flaws]] · [[RetireOPD- Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning]] · [[Mind2Dialogue- Training Human-Aware Language Models by Simulating User Mental States]] · [[Best Practice Critic Optimization]] · [[Test-Time Compute]] · [[Evals]] · [[On-Policy vs Off-Policy]] · [[Monte Carlo Methods]] · [[Backpropagation]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Cyclical Learning Rates for Training Neural Networks]] · [[Troubling Trends in Machine Learning Scholarship]] · [[On the Difficulty of Evaluating Baselines]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]]

New topics worth writing: Learning Using Privileged Information (Vapnik–Vashist), on-policy self-distillation, Rasch item-response models for dataset difficulty, cluster bootstrap and the problem as inferential unit, Holm correction in ML evaluation, checkpoint selection as a confounder in ablation studies, thinking-mode vs direct-response inference in reasoning models, Pass@k vs Majority@k as distinct capability measures
