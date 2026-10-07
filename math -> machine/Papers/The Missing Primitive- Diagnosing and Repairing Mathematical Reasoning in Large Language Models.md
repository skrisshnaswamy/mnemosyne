---
title: "The Missing Primitive: Diagnosing and Repairing Mathematical Reasoning in Large Language Models"
authors: ["Shuo Xing", "Zilin Dai", "Chengyuan Qian", "Fangzhou Lin", "Wenjing Chen", "Ping He", "Pan Lu", "Alvaro Velasquez", "Mohit Bansal", "Zhengzhong Tu"]
year: 2026
arxiv: "2610.02191"
url: https://arxiv.org/abs/2610.02191
priority: Good-To-Read
read_on: 2026-10-06
tags: [paper, llm, vision, theory]
---
## The Core Idea

A model can solve a hard maths problem and still not *see* why the problem is solvable. This paper separates those two things and gives the first one a name.

> [!NOTE] Mathematical Primitive
> The one compact, non-procedural observation that explains **why** a problem can be solved — an invariant, a hidden structure, a reduction, a theorem's applicability condition, a witness object. It must be short, specific to that problem, and explanatory. It is **not** a proof, not a calculation, and not generic advice like "use induction". ^mathematical-primitive

The trick is that "use induction" is not a primitive, but "the configuration is secretly a Cayley graph" is. A primitive names the *property* that makes a method apply, not the method.

Once you have that object, you can ask four separate questions of a model instead of one. Call the problem $x$, the primitive $p$, the solution $y$, the model $\pi$:

- **Discovery** — $\pi(x) \to \hat{p}$. Can it find the key idea from the problem alone?
- **Generation** — $\pi(x) \to \hat{y}$. Can it just solve it? (This is normal final-answer accuracy.)
- **Digestion** — $\pi(x, y) \to \hat{p}$. Given a correct solution, can it name the idea that organises it?
- **Execution** — $\pi(x, p) \to \hat{y}$. Given the idea for free, can it finish?

This did not exist before because the whole field grades maths reasoning on the final answer, or on whether a proof type-checks. Both are outcome metrics. They cannot tell you whether a failure was "didn't have the idea" or "had the idea and fumbled the algebra" — and those two failures want completely different fixes.

What it unlocks is a diagnosis with three parts:

1. **Equal accuracy hides very different models.** Qwen3.6-27B and gpt-5.4-mini score within 2 points on Generation (52.75 vs 50.55) but differ by **over 30 points** on Discovery (24.73 vs 59.34).
2. **Handing over the primitive unlocks capacity that was already there.** Every one of 12 models gains between **+17.58 and +29.67** points on Execution versus Generation.
3. **Discovery is the bottleneck.** **83.6%** of all Generation failures happen on problems where Discovery also failed.

And then the practical payoff: feed the primitive to a *teacher* during post-training (never to the student at inference), and the student's unassisted solving improves by 2.42–4.48 average points, where plain [[Fine-Tuning|SFT]] and standard on-policy distillation often make things *worse*.

## The Methodology

### The benchmark: `Prim`

Built from Humanity's Last Exam, maths, text-only, free-form, `exactMatch`-scored. They start from the community-cleaned `HLE-Verified` release, keep the `Gold` and `Revision` subsets, sample 200 problems, drop 18 that are ill-posed or unverifiable → **182 problems**.

Primitive annotation is a two-step pipeline:
1. GPT-5.4-High writes a draft primitive, conditioned on problem + reference answer + gold rationale.
2. Three humans with graduate maths training check it independently.

The humans found real rot in the source data: **15** of the kept problems had wrong official answers (13 maths errors, 2 corrupted strings). Fixing them flipped 11 of the strongest model's predictions from wrong to right, and **zero** in the other direction — a nice internal consistency check on the corrections.

They also audited for answer leakage into the primitive (since primitives were written with the solution in hand). Four primitives leaked the answer string; those were rewritten by hand.

### Scoring a predicted primitive

Three judgements, then one deterministic formula.

- $V \in \{0,1\}$ — is this a genuine primitive at all, or is it vacuous (generic advice / bare technique name / pure computation)?
- $\sigma_{\text{gate}} \in \{0, \tfrac12, 1\}$ — does it name the right **what**, the essential structure?
- $\sigma_{\text{mech}} \in \{0, \tfrac12, 1\}$ — does it explain **how** that structure unlocks the solution?

$$\mathrm{Score} = V \cdot \sigma_{\text{gate}} \cdot (0.6 + 0.4\,\sigma_{\text{mech}}), \qquad \mathrm{PrimitiveAcc} = \mathbf{1}[\mathrm{Score} \geq 0.8]$$

Read it plainly: validity and getting the structure right are **necessary** — either one at zero zeroes the whole thing. Mechanism quality only refines the score, moving it between 0.6 and 1.0 of the gate value. With $\tau = 0.8$, you need $V=1$, $\sigma_{\text{gate}}=1$, and $\sigma_{\text{mech}} \geq \tfrac12$ to be marked correct. Score takes only seven values: $\{0, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00\}$.

Because "the same idea" looks different in different kinds of maths, each problem is routed into one of three **families** before scoring:

| Family | What the gate matches against | Example types |
|---|---|---|
| **Recast** | The target setting $Y$ you move the problem into | reformulation, hidden structure, reduction, duality, theorem applicability |
| **Witness** | A specific object you build or single out | construction, extremal object, invariant/monotone quantity |
| **Argument** | The load-bearing claim the proof rests on | obstruction/contradiction, strengthened induction hypothesis |

For **Argument**, saying "by contradiction" is explicitly a gate *mismatch* — you must name the specific obstruction.

Judging is GPT-5.4-High at temperature 0. Validated against 200 human-labelled (model, problem) pairs stratified over the 12 models: **95.5% agreement, Cohen's $\kappa = 0.91$**, 9 disagreements. They also checked the single-gold-primitive risk — a valid alternative route being marked wrong — and found it in exactly 1 of 100 audited incorrect cases, and 1 of the 32 incorrect cases for the strongest model.

### `Absorb`: the post-training method

Start from standard **on-policy self-distillation** (OPSD). Teacher and student are the *same base model* $\pi_0$. The student generates its own rollout. At each step $t$, both are conditioned on the **student's own prefix** $\hat{y}_{<t}$, but the teacher additionally sees the reference solution $y$:

$$\mathcal{L}_{\mathrm{OPSD}} = \frac{1}{T}\sum_{t=1}^{T} D\big(\pi_S(\cdot \mid \hat{y}_{<t}, x),\ \pi_T(\cdot \mid \hat{y}_{<t}, x, y)\big)$$

`Absorb` changes two things.

**Change 1 — swap the privileged information.** The teacher sees the *primitive* $p$, not the full solution $y$. The reasoning: the full solution also prescribes execution steps the student can already do, which is wasted (and damaging) supervision. The primitive supplies only the missing structural bit.

**Change 2 — bounded override.** Two design principles:
- When the privileged teacher *prefers* a token more than the student does, transfer that in full — it is information the primitive genuinely contributed.
- When the teacher *strongly suppresses* a token the student likes, cap how hard that pushes. Large disagreement may just mean the teacher saw something the student cannot see, and crushing the student there teaches it nothing it could reproduce at inference.

Implemented as reverse [[KL Divergence|KL]] with a one-sided clamp, over the teacher's top-$K$ support $\mathcal{S}_t$, with both distributions renormalised over that support ($\bar{\pi}$):

$$\mathcal{L}_{\textsc{Absorb}} = \frac{1}{T}\sum_{t=1}^{T}\sum_{v \in \mathcal{S}_t} \min\left\{ \bar{\pi}_S(v \mid \hat{y}_{<t}, x)\log\frac{\bar{\pi}_S(v \mid \hat{y}_{<t}, x)}{\bar{\pi}_T(v \mid \hat{y}_{<t}, x, p)},\ \tau \right\}$$

The clamp bites only on the positive (suppression) side of the per-token term. Crucially, bounding suppression does **not** freeze a bad student preference — positive pull toward better alternatives stays fully active, so probability mass drains away from the bad token over training anyway.

> [!NOTE] Privileged information
> Information given to the teacher that the student will never have at inference time. Here: the primitive. The student is trained to *behave as if* it had seen the primitive, without ever being asked to produce one. ^privileged-information

### Training data and hyperparameters

- **709 Mathematics PhD qualifying-exam problems**, 1991–2026, parsed to LaTeX. Sources: Oregon 332, UW Madison 131, Harvard 126, Berkeley 120. Domains: Algebra 318, Analysis 206, Topology 97, Geometry 82, Number Theory 5, Logic 1.
- Each paired with a human-written proof, **median 118 words**, plus a one-sentence primitive extracted by GPT-5.4-High.
- Chosen because they are open-ended proofs with no verifiable short answer — which rules out RLVR and forces text-supervised post-training.
- Rollouts: 1 per problem, temp 1.0, top-$p$ 0.95, top-$k$ 20, 4096 completion tokens, 6144 context.
- Teacher = same base model with the adapter disabled.
- $K = 128$, clamp $\tau = 0.06$, applied to all completion tokens including truncated rollouts.
- [[LoRA]] rank 64, $\alpha = 128$, on all attention and MLP projections (including Qwen3.5's linear-attention projections).
- Fused [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], no weight decay, LR $5\times10^{-6}$, linear schedule, **no warmup**, grad clip 0.1, effective batch 32, bf16, seed 42.
- **One epoch = 22–23 optimizer steps.** vLLM co-located, resynced to the student after every step.

## Ablation Studies and Experiments

### The diagnosis — 12 models, four dimensions

| Model | Discovery | Generation | Digestion (Δ vs Disc.) | Execution (Δ vs Gen.) |
|---|---|---|---|---|
| gpt-5.4 | 82.42 | 67.03 | 100.00 (+17.58) | 86.81 (+19.78) |
| gpt-5.4-mini | 59.34 | 50.55 | 97.80 (+38.46) | 71.98 (+21.43) |
| gpt-5.4-nano | 41.76 | 44.51 | 97.25 (+55.49) | 68.68 (+24.17) |
| gpt-oss-20B | 34.62 | 43.41 | 91.21 (+56.59) | 60.99 (+17.58) |
| Qwen3.6-27B | 24.73 | 52.75 | 92.31 (+67.58) | 78.57 (+25.82) |
| Qwen3.5-27B | 28.57 | 47.80 | 93.41 (+64.84) | 71.98 (+24.18) |
| Qwen3.5-9B | 13.19 | 38.46 | 79.67 (+66.48) | 61.54 (+23.08) |
| Qwen3.5-4B | 6.59 | 26.37 | 70.88 (+64.29) | 56.04 (+29.67) |
| R1-0528-8B | 6.59 | 17.03 | 68.68 (+62.09) | 39.56 (+22.53) |
| R1-Distill-32B | 6.04 | 18.13 | 65.38 (+59.34) | 43.41 (+25.28) |
| R1-Distill-14B | 4.40 | 15.93 | 67.03 (+62.63) | 39.01 (+23.08) |
| R1-Distill-7B | 7.14 | 13.19 | 48.90 (+41.76) | 32.42 (+19.23) |

High reasoning effort where available, 120k token budget per problem.

**The Digestion–Discovery gap is the single most striking number in the paper.** Qwen3.6-27B goes from 24.73% to **92.31%** once it is shown a correct solution. R1-Distill-32B goes from 6.04% to 65.38%. Models can *recognise* the key idea when it is already instantiated in a valid argument, and largely cannot *find* it beforehand. Retrospective recognition ≫ prospective discovery.

**Two distinct capability profiles.** The joint decomposition over (Discovery, Generation) outcomes splits the families:

- OpenAI models are **primitive-forward**: lots of $D^+G^-$ (had the idea, failed the execution). Qualitative review of gpt-5.4's $D^+G^-$ cases finds two causes — missing technical knowledge (a lemma or theorem condition the model simply doesn't have) and plain procedural slips (algebra errors, missed edge cases, unresolved gaps).
- Qwen is **grind-first**: lots of $D^-G^+$ (no clean idea, still solved it). In most of these, the decisive structural idea *does* appear in the reasoning trace, typically about a quarter of the way in, after a first attempt that didn't use it. Its cold Discovery answer usually captures a partial version of the idea but misses the decisive piece. Combined with Qwen's longer traces, this reads as structure emerging *during* extended search rather than being isolated up front.
- The R1 distills cluster in joint $D^-G^-$ failure.

### Is the "rescue" just task decomposition?

This is the ablation that earns the whole concept. Four kinds of help, same models:

| Model | Generation | Self-generated primitive | Teacher Plan | Teacher Primitive | Gold Primitive |
|---|---|---|---|---|---|
| gpt-5.4-mini | 50.55 | 48.90 (**−1.65**) | 60.44 (+9.89) | 63.74 (+13.19) | 71.98 (+21.43) |
| gpt-5.4-nano | 44.51 | 46.15 (+1.64) | 54.40 (+9.89) | 60.99 (+16.48) | 68.68 (+24.17) |
| Qwen3.6-27B | 52.75 | 41.21 (**−11.54**) | 56.04 (+3.29) | 64.84 (+12.09) | 78.57 (+25.82) |

Three things fall out:

**Self-generated primitives are worse than nothing.** Forcing a discover-then-execute split, using the model's *own* primitive, costs Qwen3.6-27B **11.54 points**. Across all 12 models in the appendix, 9 of 11 non-teacher cases go *down*; Qwen3.5-9B loses 15.38. So the gain is not from decomposition — it is from the *content* of the primitive.

**A Teacher Plan is strictly weaker than a Teacher Primitive.** Both generated by the same teacher (gpt-5.4) from the problem alone. The plan is a 3–6 step procedural outline; the primitive is the structural observation. The primitive wins by 3.3 (gpt-5.4-mini), 6.6 (nano), and 8.8 (Qwen3.6-27B) points. The rescue is not generic scaffolding.

**Digestion predicts rescuability.** Split initially-failed problems by whether the model can recover the primitive from the reference solution. For Qwen3.5-27B/9B/4B, digestible problems are rescued 17.6–28.6 points more often than undigestible ones. The link is weaker for the R1 distills, so the two capabilities are related but not the same thing.

### Where the failures live

Decomposing every Generation failure by Discovery ($D$) and Execution ($E$):

- **83.6% of all failures are in the $D^-$ regime.**
- A large chunk are **discovery-limited** ($D^-E^+$): the model can't find the idea, but solves the problem once given it. Qwen3.6-27B: 48.8% of its 86 failures. Qwen3.5-9B: 44.6% of 112.
- Strong models shift toward $D^+$: gpt-5.4 has 53.3% of its 60 failures as $D^+E^+$ — *utilization-limited*, it has both the idea and the ability and still misses.
- Weak models collapse into $D^-E^-$: R1-Distill-14B at 68.0%, R1-Distill-7B at 68.4%.

### Which failures can post-training repair?

709 qualifying-exam problems, Qwen3.5 at 27B/9B/4B. 341 base failures across the three; 294 with failed Discovery, split evenly 147/147.

| Base failure type | SFT | OPSD | `Absorb` |
|---|---|---|---|
| $D^-E^+$ discovery-limited | 20.4% | 21.1% | **23.8%** |
| $D^-E^-$ capability-limited | 6.8% | 8.2% | 10.2% |
| $D^+E^-$ execution-limited | 11.1% | **0.0%** | 18.5% |
| $D^+E^+$ utilization-limited | 35.0% | 30.0% | 45.0% |

Discovery-limited failures repair at roughly **3×** the rate of capability-limited ones, and this ordering holds at every model scale. Note OPSD repairing **zero** of the execution-limited cases.

### Main results

| Model | Prim Discovery | Prim Generation | HLE Math | HMMT25 | Omni-MATH | Avg |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | 6.59 | 26.37 | 26.70 | 76.67 | 78.00 | 51.94 |
| + SFT | 5.49 | 25.27 | 27.36 | 83.33 | 79.33 | 53.82 (+1.89) |
| + OPSD | 8.79 | 30.22 | 29.58 | 83.33 | 79.33 | 55.62 (+3.68) |
| + **`Absorb`** | 9.89 | 31.32 | 31.02 | 83.33 | 80.00 | **56.42 (+4.48)** |
| Qwen3.5-9B | 13.19 | 38.46 | 35.60 | 90.00 | 78.67 | 60.68 |
| + SFT | 12.64 | **32.42 (−6.04)** | 34.55 | 90.00 | 80.67 | 59.41 (−1.27) |
| + OPSD | 10.99 | **33.52 (−4.95)** | 35.08 | 90.00 | 81.33 | 59.98 (−0.70) |
| + **`Absorb`** | 14.29 | 43.96 (+5.49) | 38.35 | 93.33 | 82.00 | **64.41 (+3.73)** |
| Qwen3.5-27B | 28.57 | 47.80 | 42.67 | 96.67 | 87.33 | 68.62 |
| + SFT | 28.02 | 48.35 | 43.59 | 96.67 | 86.67 | 68.82 (+0.20) |
| + OPSD | 28.57 | **44.51 (−3.30)** | 42.80 | 96.67 | 88.00 | 67.99 (−0.62) |
| + **`Absorb`** | 28.02 | 48.90 | 46.60 (+3.93) | 100.00 | 88.67 | **71.04 (+2.42)** |

HMMT25 and Omni-MATH are pass@4 (they're too hard otherwise). Avg is over the four Generation scores only.

Two things to notice. First, **SFT and OPSD regress** at 9B (−6.04 and −4.95 Generation) and OPSD regresses at 27B (−3.30). That explains the modest net gains in the repair table — repairs are being paid for with new breakages on problems the model already solved. Second, `Absorb`'s gains land on *solving*, not on explicit Discovery: at 9B, Generation +5.49 but Discovery only +1.10; at 27B, Generation +1.10 and HLE Math +3.93 while Discovery goes **down** 0.55. The primitive's guidance shows up inside the reasoning, not as better primitive articulation.

### What did not work — the useful half

All on Qwen3.5-9B, base Generation 38.46 / Discovery 13.19.

**Solution privilege instead of primitive privilege.** Same objective, same rollouts, teacher sees the full reference proof: **33.52 (−4.95)**. So the gain isn't "privileged conditioning helps" — the full solution is actively harmful here.

**Three transfer mechanisms that all failed:**

| Variant | Generation | Discovery |
|---|---|---|
| rKL, no cap | 31.87 (−6.59) | 10.44 |
| fKL, capped | 36.81 (−1.65) | 12.09 |
| Divergence control (drop high-KL tokens) | 34.62 (−3.85) | 12.09 |
| Entropy control | 31.87 (−6.59) | 14.29 |
| **Bounded override (`Absorb`)** | **43.96 (+5.49)** | 14.29 |

Unbounded reverse KL with the primitive loses 6.59 points — **privilege leakage**: the teacher drags the student toward choices only reachable with the primitive in context. Forward KL with the same cap also loses, so the cap alone isn't the mechanism; the *asymmetry* matters.

The two "control" variants are the sharpest negative results. **Divergence control** masks tokens whose KL exceeds the 95th percentile ($\tau_D = 0.05$) — that drops only ~5% of tokens but ~86% of the total divergence mass, and recovers only to 34.62, still below base. **Entropy control** then tries to be cleverer: keep high-divergence tokens where the *student* is uncertain ($H_t \geq 0.86$), on the theory that uncertain positions are where genuine primitive guidance lives. It masks *fewer* tokens and does *worse* — right back to 31.87.

The lesson: neither disagreement magnitude nor student confidence cleanly separates helpful guidance from harmful override at the token level. `Absorb` wins because it never makes a keep-or-drop decision about a token at all — it keeps supervision everywhere and bounds it *within* each distribution.

**Primitive as rollout target instead of privilege.** Train the student to generate the primitive:

| Rollout target / objective | Generation | Discovery |
|---|---|---|
| `Absorb` (solution rollout, primitive privilege) | 43.96 | 14.29 |
| Primitive rollout, rKL no cap | 32.97 | 13.19 |
| Primitive rollout, fKL capped | 33.52 | 9.34 |
| Primitive rollout, rKL capped (same objective) | 39.01 | 12.09 |

Holding the objective fixed and changing *only* what the student generates costs 4.95 points. The primitive is worth more as a lens on the teacher than as a thing the student is asked to produce.

**Primitive SFT is catastrophic.** Direct supervision on the primitive: Generation **28.02 (−10.44)**, Discovery **2.75 (−10.44)**. It destroys the very thing it was supervising. Solution SFT: 32.42 (−6.04).

**TOP-D, adapted.** A policy-gradient on-policy distillation method with a bounded proximal reward $\tilde{r}_t = \log(\alpha\rho_t + 1 - \alpha)$, $\alpha = 0.1$, $G=4$ rollouts, group-normalised advantages ([[GRPO]]-style). Given the *same* primitive privilege and the same corpus: 31.87 (−6.59), Discovery 9.34. The privileged signal alone buys nothing without the right transfer.

**Dual-channel** — their own more elaborate alternative, split into an early-route channel (reverse KL on the first 512 tokens, since rescue seemed to act through early route choice) and an adjudication channel at student self-doubt forks ("Wait", "But", "However", "Actually"), where the teacher only ranks student-generated candidates via base-corrected preferences $q_T = \mathrm{softmax}((z_T - z_0)/0.2)$ with loss $D_{\mathrm{KL}}(q_T \| q_S)$ weighted 0.3. Result: **29.67 (−8.79)**, the worst distillation variant. Caveat: it used a bigger 1,692-problem pool, full-length primitives and 8k completions, so it is not a clean single-variable ablation.

## Worth Remembering

**The absolute accuracies look suspiciously high for HLE.** gpt-5.4 at 67% Generation is far above its usual full-HLE number. The authors flag this: `Prim` is a curated scorable subset, and they deliberately removed items with broken answers, gappy rationales and awkward formatting. Do not compare these numbers to published HLE leaderboards.

**The source benchmark was dirtier than expected.** 15 of 182 retained problems had wrong official answers, even after starting from a *verified* HLE release. That's ~8%. Worth internalising as a general prior about frontier maths benchmarks.

**The headline gains are small in absolute terms.** 22–23 optimizer steps on 709 problems, one epoch, LoRA rank 64. +2.42 to +4.48 average points. The more durable contribution is the diagnostic frame and the pile of negative results, not the delta.

**Discovery barely improves — and that's the point, but it's also a limitation.** `Absorb` internalises primitive-*guided reasoning*, not primitive-*finding*. At 27B, Discovery actually drops 0.55 while Generation and HLE Math rise. So the dominant bottleneck the paper identifies is not the bottleneck the method fixes. It fixes the downstream consequence.

**Privilege leakage is the central failure mode of this whole family.** Uncapped reverse KL with primitive privilege is *worse than doing nothing*. If you try privileged-teacher distillation, the transfer mechanism is not a detail — it is the method. See also [[What Does Privileged Information Add to On-Policy Self-Distillation]] and [[Do We Really Need KL Divergence for On-Policy Distillation of Large Language Models]].

**The single-gold-primitive design is a known soft spot.** Each problem has one annotated primitive. Audits suggest valid alternative routes are mis-scored ~1% of the time, which is reassuring but is measured on a sample and depends on the judge agreeing with the annotators. Multi-route gold annotations would be the obvious extension.

**No RLVR, by construction.** Qualifying-exam proofs have no short verifiable answer, so outcome-verified RL is off the table and everything here is text-supervised. That's a reasonable design choice, but it means the comparison is SFT vs OPSD vs `Absorb` — not `Absorb` vs a properly tuned verifiable-reward pipeline.

**Practical caveat for anyone reusing this:** the primitive annotations cost a frontier model plus three graduate-trained humans per item. For the 709-problem training corpus, primitives came from GPT-5.4-High *unreviewed*. So the training-side signal is model-generated and unvalidated; only the 182 benchmark primitives got human review.

**Questions I'd want answered.** Does the Digestion→Discovery gap shrink with scale, or is it structural? gpt-5.4 hits 100% Digestion and 82% Discovery — a 17.6-point gap, versus 67.6 for Qwen3.6-27B. That trend suggests scale closes it, but with four OpenAI points it's hard to call. Second: the authors suggest separating structural search from procedural execution as an *inference-time* compute allocation strategy — but their own self-generated-primitive ablation shows naive discover-then-execute makes things worse. What would a search procedure over primitives need to look like to beat that?

**Connections.** The teacher/student-same-base-model setup is the OPSD line ([[On-Policy or Off-Policy Learning- A Systematic Study of Distillation Dynamics]], [[Learning from Teacher Continuations at Student States]]). The one-sided clamp is structurally the same move as the clip in [[PPO]] — bound how far one update can push — and the reverse-KL mode-seeking choice connects to [[KL Divergence#Reverse KL — $D_{KL}(Q \parallel P)$, "**mode-seeking**"|reverse KL]]. The "model has the capability but can't access it" story rhymes with [[Locked at the Entrance, Open Inside- Where RLVR Narrows the Solution Space]] and [[Base Models Can Reason By Taking a Cue From Training Data]].

## Links
Related: [[What Does Privileged Information Add to On-Policy Self-Distillation]] · [[On-Policy or Off-Policy Learning- A Systematic Study of Distillation Dynamics]] · [[Do We Really Need KL Divergence for On-Policy Distillation of Large Language Models]] · [[Learning from Teacher Continuations at Student States]] · [[KL Divergence]] · [[LoRA]] · [[Fine-Tuning]] · [[Chain of Thought]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Distillation]] · [[Locked at the Entrance, Open Inside- Where RLVR Narrows the Solution Space]] · [[Base Models Can Reason By Taking a Cue From Training Data]] · [[Evals]] · [[GRPO]] · [[PPO]] · [[Test-Time Compute]] · [[Mind2Dialogue- Training Human-Aware Language Models by Simulating User Mental States]] · [[Negative Self-Distillation- Learning to Reason by Avoiding Flaws]]

New topics worth writing: Privilege leakage in distillation, Structural vs procedural reasoning evaluation, Benchmark answer-key errors and verified releases, Humanity's Last Exam, One-sided clamped divergences as transfer controls, Failure-mode decomposition as an evaluation design pattern, Mathematical proof-sketch supervision, Judge validation with Cohen's kappa
