---
title: "On-Policy Parameter Update Direction Underlies Generalization in LLM Post-Training"
authors: ["Shufan Shen", "Zhongni Hou", "Junshu Sun", "Yufei Zhang", "Wei Lin", "Guojun Yin", "Qingming Huang", "Shuhui Wang"]
year: 2026
arxiv: "2609.36659"
url: https://arxiv.org/abs/2609.36659
priority: Good-To-Read
read_on: 2026-10-05
tags: [paper, llm, rl, optimization]
---
## The Core Idea

Two ways to post-train a language model, and everyone knows which one generalises better.

**SFT** (supervised fine-tuning): show the model a correct answer written by a teacher, and push up the probability of exactly those tokens. Cheap, simple, and it tends to memorise.

**On-policy** (GRPO, on-policy distillation): let the model write its own answer, score it, then push up or down depending on the score. Expensive — you have to generate text every step — but it generalises.

The new claim: the whole generalisation gap can be carried by **one bit per parameter**.

Take a model before on-policy training, $\bm{\theta}_{\mathrm{base}}$, and after, $\bm{\theta}_{\mathrm{on}}$. Record only the sign of the change for each weight:

$$\bm{v} = \text{sign}(\bm{\theta}_{\mathrm{on}} - \bm{\theta}_{\mathrm{base}}) \in \{-1, 0, 1\}^d$$

Now throw away the on-policy training entirely. Go back to plain SFT on fixed teacher trajectories, but every step, delete any gradient component that would move a weight *against* its sign in $\bm{v}$. That is the whole method, called **OPSFT**.

It works. On Qwen3-8B trained on DeepMath, mean accuracy over AIME24/25 and HMMT-Feb/Nov: vanilla SFT 38.41, GRPO 40.31, OPSFT **41.67** — and OPSFT took 8.9 hours against GRPO's 19.3.

> [!NOTE] On-policy update direction
> The per-parameter sign vector $\bm{v}$ of the cumulative weight change produced by on-policy training. Not a subspace, not a mask — one trit per weight. It is sufficient to transfer on-policy generalisation to SFT. ^on-policy-direction

Why this did not exist before: prior work on on-policy weight updates looked at *where* the weights move — the sparse set of coordinates that change at all — and treated it as a curiosity of the optimiser. This paper's sharpest result is that location alone is useless. Constraining SFT to the same coordinates GRPO touched, with no sign constraint, scores **34.07** on Qwen3-4B — *worse* than vanilla SFT at 34.22. Add the sign and it jumps to **41.15**. The information is in the sign, not the support.

What it unlocks is a reordering of the post-training pipeline. The expensive, rollout-heavy part becomes a short *direction-finding* phase (50 GRPO steps), after which you can run cheap SFT forever and still generalise. And it gives you a way to keep teaching a model that has already been through [[GRPO]] without wrecking what it learnt.

It also directly contradicts the slogan "SFT memorises, RL generalises". SFT memorises *when it is free to move in any direction*. Pin the directions and it generalises.

## The Methodology

### Why the two paradigms move differently

Write $s_{\bm\theta}(x,\tau) = \nabla_{\bm\theta}\log \pi_{\bm\theta}(\tau \mid x)$ — the [[Policy Gradient|score]] of a trajectory $\tau$ for prompt $x$. The useful fact is that it averages to zero under the model's own distribution:

$$\mathbb{E}_{\tau\sim\pi_{\bm\theta}(\cdot\mid x)}\big[s_{\bm\theta}(x,\tau)\big] = \nabla_{\bm\theta}\sum_\tau \pi_{\bm\theta}(\tau\mid x) = \nabla_{\bm\theta} 1 = \bm{0}$$

SFT's gradient is just a score averaged over a *fixed* teacher distribution:

$$g_{\mathrm{sft}}(\bm\theta) = -\mathbb{E}_x\big[\mathbb{E}_{\tau\sim\pi_{\mathrm{teacher}}}[s_{\bm\theta}(x,\tau)]\big]$$

The on-policy gradient weights the score by an advantage, $g_{\mathrm{on}} = \mathbb{E}[A_{\bm\theta}(x,\tau)\,s_{\bm\theta}(x,\tau)]$. Because the unweighted score has zero mean, this collapses to a **covariance**:

$$g_{\mathrm{on}}(\bm\theta) = \mathbb{E}_x\big[\operatorname{Cov}_{\tau\sim\pi_{\bm\theta}}[A_{\bm\theta}(x,\tau),\, s_{\bm\theta}(x,\tau)]\big]$$

That is the whole theoretical argument. SFT follows a *mean* against a frozen target. On-policy follows a *covariance* against a target that moves, because both $\pi_{\bm\theta}$ and the advantage distribution $A_{\bm\theta}$ shift as the model learns. The frozen target gives a stable direction; the moving one does not.

Measured, with Qwen3-1.7B on DeepMath, as cosine similarity of weight changes between training stages:

| | interval updates | cumulative updates |
|---|---|---|
| SFT | positively correlated | $\approx 1.0$ |
| GRPO / OPD | $\approx$ orthogonal | $\approx 0.5$ |

SFT walks in a straight line. On-policy keeps turning. That constant turning is what motivates treating the *final* cumulative direction as the thing worth copying — it is the endpoint of a search SFT never performs.

### The constraint, exactly

Given SFT gradient $\bm g$, the parameter moves by roughly $-\bm g$. Keep only the components whose induced movement agrees with $\bm v$:

$$\bm g_s = \mathbb{I}\big(\text{sign}(-\bm g) = \bm v\big) \odot \bm g$$

Hand $\bm g_s$ to the optimiser. Components that disagree get zeroed — not flipped, not shrunk, dropped.

One detail that matters in practice: **AdamW can violate the constraint even after the gradient is clean.** [[Momentum]] carries old steps forward, and decoupled weight decay (see [[Decoupled Weight Decay Regularization (AdamW)]]) pulls every weight toward zero regardless of sign. So OPSFT enforces the constraint **twice**:

1. **Before the step** — mask gradient entries that disagree with $\bm v$ (and entries outside the on-policy support $M$).
2. **After the step** — recompute cumulative displacement $\bm\theta - \bm\theta_0$; any coordinate whose *total* displacement now sits on the wrong side of $\bm v$ gets reset to $\bm\theta_0$.

Stage 2 is a hard projection on the accumulated travel, not the step. Without it the optimiser's own machinery leaks the constraint away.

### Training setup

- **Models.** Qwen3-1.7B / 4B / 8B, DeepSeek-R1-Distill-Llama-3-8B.
- **Direction-finding.** GRPO, reward 1.0 for a correct final answer (math) or all unit tests passing (code), else 0.0. Batch 128, 8 rollouts per prompt, LR $1\times10^{-6}$, KL coefficient **0.0** (no [[KL Divergence|KL]] leash), 150 steps max, 16,384-token responses.
- **SFT data.** Teacher is Qwen3-30B-A3B-Instruct-2507, one completion per prompt at temperature 0.6 / top-$p$ 0.95, **kept only if verified correct** by `math_verify` or by actually executing the code against reference tests. Final: 44,810 math examples, 9,585 code examples.
- **SFT optimisation.** Cross-entropy, FP32, FSDP2 across 8× H20. Global batch 64, AdamW ($\beta_1=0.9$, $\beta_2=0.95$), weight decay 0.01, grad clip 1.0, **constant LR with no warmup**, LR $1\times10^{-7}$ for the 700-step runs and $1\times10^{-6}$ for the fast runs, 700 steps, seed 42.
- **Evaluation.** AIME24, AIME25, HMMT25-Feb, HMMT25-Nov; 8 samples per problem, temperature 1.0, top-$p$ 1.0. Code: HumanEval+, MBPP+, LiveCodeBench v6, 4 samples per problem.

## Ablation Studies and Experiments

### The headline table (DeepMath, mean over four math benchmarks)

| Model | GRPO | SFT | DFT | OPSFT |
|---|---|---|---|---|
| Qwen3-1.7B | 12.09 (4.9h) | 13.15 (3.5h) | 13.16 (3.9h) | **15.11 (2.3h)** |
| Qwen3-4B | 38.96 (16.5h) | 34.22 (8.1h) | 34.69 (8.6h) | **40.11 (6.4h)** |
| Qwen3-8B | 40.31 (19.3h) | 38.41 (13.2h) | 38.97 (13.9h) | **41.67 (8.9h)** |
| R1-Distill-Llama-8B | 25.21 (18.2h) | 18.02 (13.6h) | 19.90 (14.1h) | **26.56 (8.1h)** |

OPSFT's time includes the 50 GRPO steps used to find the direction plus 100 OPSFT steps. So: beats 100-step GRPO using a direction found at step 50, in half the wall-clock.

Code, Qwen3-4B on Eurus: GRPO 57.19 (20.3h), SFT 54.30 (9.8h), OPSFT **58.07** (9.5h).

### The ablation that carries the paper

Qwen3-4B, DeepMath, four-way:

| Location constraint | Direction constraint | Mean |
|---|---|---|
| Random | — | 30.84 |
| On-policy | — | 34.07 |
| On-policy | **Random** | 27.71 |
| On-policy | On-policy | **41.15** |

Read it in order. Random masking hurts (30.84 vs SFT's 34.22) — so masking is not a free [[Regularization|regulariser]]. On-policy *locations* recover roughly vanilla SFT (34.07). On-policy locations with *random* signs is the worst of everything (27.71) — this is the control that rules out "it's just sparsity". Only the real signs produce the jump.

### What did not work

- **Location-only constraints.** Worse than vanilla SFT on Qwen3-4B (34.07 vs 34.22). The sparse-subnetwork story from prior work is real but not the mechanism.
- **Plain SFT on an already-GRPO'd model.** This is the most useful negative result. Take Qwen3-4B after 100 GRPO steps (38.96 mean) and continue with ordinary SFT on verified high-quality trajectories: it drops to **36.25**. Qwen3-8B: 40.31 → 37.50. R1-Distill-Llama-8B collapses, 25.21 → **16.36**. Code is brutal: Qwen3-4B 57.19 → 48.71. Good data is not enough; unconstrained SFT undoes on-policy learning. OPSFT on the same data goes *up* instead — 38.96 → 41.36, 40.31 → 42.39, 25.21 → 26.15, 57.19 → 60.05. Compare [[Fine-Tuning#The failure modes 🪤|catastrophic forgetting]].
- **Cross-domain direction reuse.** Using a direction found on Eurus (code) to run OPSFT on DeepMath (math) generalises poorly. Directions are domain-specific.
- **BF16.** Works, but at reduced strength — see below.

### Where the direction comes from

Directions found later in GRPO training are better, with **diminishing returns**: the gain from step-50 → step-100 directions exceeds the gain from step-100 → step-150. Fifty steps is already most of the signal.

Cross-dataset reuse *within* a domain is fine, and the numbers are odd in an interesting way. GRPO on DAPO and GRPO on DeepMath produce directions with cosine similarity only $\approx 0.2$, yet OPSFT on DeepMath using the **DAPO** direction slightly *beats* OPSFT using DeepMath's own. So there are multiple distinct directions in the same domain that all support generalisation. Update *location* overlap, by contrast, is $\approx 0.4$ whether datasets share a domain or not — another sign that location carries no domain information.

### Precision, and how sparse this gets

Qwen3-8B, fraction of parameters actually changed:

| Precision | Method | Params updated | Mean |
|---|---|---|---|
| FP32 | GRPO | 9.499% | 42.01 |
| BF16 | SFT | 2.702% | 38.33 |
| BF16 | OPSFT | **0.408%** | 40.94 |
| FP32 | SFT | 89.286% | 38.41 |
| FP32 | OPSFT | 9.454% | **42.81** |

Two things. First, BF16 SFT is *accidentally* sparse — small gradients round to zero under BF16's limited mantissa (see [[Mixed Precision training]]), so only 2.7% of weights move at all. Second, BF16 OPSFT beats BF16 SFT while touching 0.408% of the weights — a 24.6% relative gain over base. And the FP32 OPSFT update density (9.454%) lands almost exactly on GRPO's (9.499%), which is a quiet but neat corroboration.

### What the constrained model actually looks like inside

- **Spectrum.** Vanilla SFT concentrates its update in the top few singular values and rotates the weight matrices' principal singular vectors a lot. GRPO spreads singular values more evenly, has higher stable rank, and rotates little. **OPSFT matches GRPO on all three.** So the "small spectral shift" that earlier papers attributed to on-policy training is a *consequence* of the direction, not of rollouts.
- **Matrix-level direction.** OPSFT's *parameter-level* sign constraint causes its *matrix-level* direction to differ sharply between early and late training — the constraint redirects the trajectory once, then OPSFT walks happily along the new direction.
- **Reasoning behaviour.** Response lengths are similar for SFT, OPSFT and GRPO, so this is not "think longer". But OPSFT's accuracy standard deviation across samples rises toward GRPO's, and its per-1k-word rate of case-splitting cues (*if*, *otherwise*, *consider a case*) and logical cues (*therefore*, *because*) moves toward GRPO's too.
- **Generality across on-policy paradigms.** Qwen3-1.7B with a direction from **OPD** (on-policy distillation, teacher Qwen3-30B-A3B) instead of GRPO: base 8.59, OPD 16.75, SFT 13.83, OPSFT-location 13.48, OPSFT-direction **16.71**. Same pattern, different paradigm.
- **Out of domain.** Trained on math only, Qwen3-8B mean over IFEval / ARC / HaluEval / HellaSwag / WinoGrande / PIQA: base 75.85, SFT 75.54 (a *drop*), GRPO 76.20, OPSFT **76.39**.

## Worth Remembering

**The practical recipe.** Run GRPO for ~50 steps. Snapshot $\text{sign}(\bm\theta_{\mathrm{on}} - \bm\theta_{\mathrm{base}})$. Then run SFT with that sign mask applied to gradients *and* to cumulative displacement after every optimiser step. You get on-policy-grade generalisation at SFT cost, and you can keep feeding it new verified trajectories indefinitely.

**The limitation the authors state.** Every SFT trajectory here came from a teacher (Qwen3-30B-A3B) that is substantially stronger than the student, and was verified correct before use. On-policy training generates its own trajectories, which may be better matched to the student in diversity and difficulty. There is no good measure yet for "is this trajectory suitable for this student", so the paper cannot separate "the direction did it" from "the direction plus unusually good data did it". Treat the result as **current best understanding**, not settled.

**A caveat they do not raise.** Qwen3-1.7B is the one model where vanilla SFT *beats* GRPO (13.15 vs 12.09). At that scale the GRPO baseline is weak, so "OPSFT ≥ GRPO" is a lower bar than it looks. The 4B/8B and R1-Distill results are the load-bearing ones.

**Open question worth chasing.** If multiple near-orthogonal directions (cosine $\approx 0.2$) all support generalisation, what property do they share? Not a shared subspace, not a shared support. Something about the *per-coordinate sign pattern* encodes task-relevant structure, and nobody knows what. That is the obvious next paper.

**Connections.** This is gradient projection — mechanically close to GaLore-style methods — but the projection comes from a *different training run* rather than from decomposing the SFT gradient. It is also a parameter-localisation method where, unusually, the localisation signal is *learnt by RL* rather than picked by a sensitivity heuristic. And it is a counterexample to reading [[Reward Hacking|Goodhart]]-style "RL generalises because of the reward signal": here the reward signal's entire contribution is compressed into a sign vector and then discarded.

**Three numbers to keep.** 0.408% of parameters changed, 24.6% relative gain over base (BF16, Qwen3-8B). 34.07 for location-only vs 41.15 for location-plus-direction (Qwen3-4B). 25.21 → 16.36, the R1-Distill collapse from plain SFT on a post-trained model.

## Links

Related: [[GRPO]] · [[Policy Gradient]] · [[Fine-Tuning]] · [[On-Policy vs Off-Policy]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Momentum]] · [[Mixed Precision training]] · [[Regularization]] · [[Distillation]] · [[On-Policy or Off-Policy Learning- A Systematic Study of Distillation Dynamics]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[Do We Really Need KL Divergence for On-Policy Distillation of Large Language Models]] · [[1% of Tokens Can Be Enough- On Gradient Estimation in On-Policy Distillation]] · [[Drift-Constrained Optimization- Only Direction Matters in Fine-Tuning Instruct Models]] · [[The Lottery Ticket Hypothesis]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Training language models to follow instructions with human feedback]] · [[Backpropagation]]

New topics worth writing: Projected gradient descent in LLM post-training, GaLore and gradient low-rank projection, Sign-based parameter update masks, Sparse subnetwork localisation under RL, DFT (reward-rectified SFT), DeepMath-103K, Stable rank of weight updates, Trajectory quality metrics for distillation
