---
title: "MotorMind: Scaffolding General Vision Language Models for Zero-Shot Robot Manipulation"
authors: ["Bingxuan Li", "Siqi Song", "Yizhuo Wu", "Jiarui Yao", "Tong Zhang", "Huan Zhang"]
year: 2026
arxiv: "2609.38078"
url: https://arxiv.org/abs/2609.38078
priority: Good-To-Read
read_on: 2026-10-05
tags: [paper, llm, vision, theory]
---
## The Core Idea

A robot arm can be driven by a plain vision-language model (VLM) — no robot-specific training at all — if you give it the right *size* of action to think about and let it watch itself while it moves.

Today's robot policies are **VLA models** (vision-language-action: image + sentence in, motor commands out). They work well on tasks they were trained on and fall apart off-distribution. The zero-shot numbers in this paper are brutal: $\pi_{0.5}$, MolmoAct2, OpenVLA-OFT and GR00T N1.5 all score **0.0%** on LIBERO-PRO without fine-tuning. They also cannot inherit progress from general VLMs, because turning a VLM into a VLA means retraining it on robot data.

The other camp, "agentic robotics", keeps a general VLM for planning but bolts on a pile of machinery: segmentation models (SAM3), learned action experts, skill libraries, motion planners, code generators. The VLM's own spatial reasoning never gets to drive.

MotorMind strips all of that out. The VLM is the only learned component. What it outputs is not joint angles and not Python code, but **mid-level actions** in the robot's base frame: `move forward 12.20 mm`, `rotate yaw_left 15°`, `close gripper`. A dumb deterministic controller turns these into motion and reports back what actually happened.

> [!NOTE] Mid-level action representation
> An action vocabulary coarse enough that a language model can reason about it (named directions, millimetres, degrees) but fine enough that a non-learned controller can execute it exactly. It sits between low-level torques and high-level skills like "grasp the mug". ^mid-level-action

Two findings make this work, and the second is the real contribution.

**First**, they measured *which* decisions a VLM is actually good at. Three questions a manipulation policy must answer over and over: what next, did that help, am I done? Across six models, "what next" is always the worst. Qwen3.8-Flash-Next scores 36.25% on action selection, 55.00% on progress, 65.00% on completion. So: propose *short* action batches, because they will often be wrong, and make them cheap to revise.

**Second**, revision has to overlap with motion. A sequential observe–think–act loop cannot stop a bad action, because it is busy executing it. MotorMind runs a **Monitor** on a background thread that can cancel pending commands at the next action boundary. If the gripper closed on the wrong object, the queued "place in bowl" never runs.

What this unlocks: **66.7%** on LIBERO-PRO base suites and **53.8%** under perturbation, against **13.3%** and **19.2%** for the best zero-shot baseline (CaP-X at 10 loops). 53.8% under perturbation beats fine-tuned OpenVLA-OFT's 51.2%. On a real xArm6, **95%** average success with no adaptation. And swapping Qwen for GPT-6 Sol lifts base success to **83.3%** — the harness is a free rider on VLM progress.

## The Methodology

### The diagnostic that shaped the design

240 questions built from LIBERO expert demonstrations, 80 per capability. Each question shows a fixed scene camera plus a wrist camera at times $t_0$ and $t_1$, a local subgoal, and a success criterion.

| Model | Action | Progress | Completion | Overall | Latency (ms) |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next-FP8 | 36.25 | 55.00 | 65.00 | 52.08 | **276** |
| HY-Embodied-0.5 MoT-2B | 20.00 | 50.00 | 47.50 | 39.17 | 402 |
| HY-Embodied-VLM-1.0 A3B | 18.75 | 43.75 | 50.00 | 37.50 | 592 |
| Cosmos3-Nano | 18.75 | 50.00 | 62.50 | 43.75 | 3035 |
| GLM-5.3-Flash | 37.50 | 52.50 | 58.75 | 49.58 | 403 |
| GPT-6 Astra | **60.00** | **76.25** | **82.50** | **72.92** | 8724 |

The breakdown by primitive type is the sharpest result. Qwen gets **0/19 rotation questions right**. GLM gets 3/19. Only GPT-6 Astra manages rotations (12/19, 63.16%). Rotational reasoning in a base frame is essentially absent from smaller VLMs.

Labels came from privileged simulator state the model never sees: end-effector poses, joint states, expert controls, simulator predicates, outcomes of perturbed rollouts. Rotation ground truth used relative orientation

$$\Delta R = R_1 R_0^{-1}$$

converted to a rotation vector in the base frame. Negative progress examples were made two ways — pair a real expert transition with a plausible-but-wrong subgoal, or restore an expert state and run a physically simulated perturbed action, keeping only those that genuinely fail to advance.

Qwen3.8-Flash-Next was picked as the backbone: 52.08% overall at 276 ms, versus 72.92% at 8724 ms. A 32× latency penalty for 21 accuracy points is a bad trade when you need frequent feedback.

### Task formulation

Instruction $\ell$. At decision cycle $t$ the harness sees $o_t = (\mathcal{I}_t, r_t)$ — camera images and measured robot state (tool pose, gripper state). The Planner emits an ordered subgoal list $\mathcal{G} = (g_1, \dots, g_M)$, each carrying an intended state change, a target description, and a success criterion. "Take the object" needs evidence it is *held*; "carry it" needs arrival *and* retained grasp.

### The action batch

Per cycle the Executor returns a structured proposal: `assessment`, `done`, optional `command`, `expect`, `confidence`. The command holds

$$\mathcal{A}_t = (a_{t,1}, \dots, a_{t,K_t}), \qquad a_{t,k} = (\tau_{t,k}, \boldsymbol{\eta}_{t,k})$$

with $\tau$ the action type and $\boldsymbol{\eta}$ its parameters. Three core primitives plus two auxiliaries:

- `move` — one direction word or base axis, distance in **millimetres**
- `rotate` — direction or axis, angle in **degrees**
- `gripper` — `open`/`close`, optional width
- `wait` — positive duration; `home` — return to rest

Base-frame convention, spelled out in the prompt: forward $=+X$, backward $=-X$, left $=+Y$, right $=-Y$, up $=+Z$, down $=-Z$. Rotations by right-hand rule: `yaw_left` $=+Z$, `pitch_up` $=-Y$, `roll_ccw` $=+X$.

A subtlety worth noting: the prompts carefully separate *two* reference frames. Left/right in the **instruction** means left/right in the fixed external camera image. Left/right in the **action output** means base-frame $\pm Y$. The prompt explicitly warns that a target on image-right does not imply the action `right`. Schema errors are bounced back to the proposer for correction before anything executes.

### Five roles, one frozen model

The same VLM is called five ways. Separation is enforced by *what context each call gets* and *what its output schema permits*.

| Role | Sees | Can emit |
|---|---|---|
| **Planner** | instruction, memory note, evidence | subgoals + success criteria |
| **Executor** | one subgoal, images, robot state, recent cycles | action batch, done signal |
| **Monitor** | running subgoal, fresh observations | alerts, STOP — **no actions** |
| **Verifier** | before/after scene, Executor's claim | success verdict |
| **Memory** | plan, last note, motion log | compact note — **never an action** |

The Monitor and Memory schemas have no action field *by construction*, so they cannot silently override a motion in flight. The Verifier prompt is explicit that it "is not that robot and is not obliged to agree with it."

A deterministic **Controller** sits underneath: validates each command, checks later actions in a batch against fresh observations, resolves parameters into Cartesian motion, and records what the robot *measured*. So the next proposal sees achieved motion, not intended motion. An intended movement is never treated as an accomplished one.

### Grounding without SAM

No segmentation model. The VLM localises targets by naming labelled image-space bounding boxes. Compatible fixed-camera views are triangulated; a close wrist view can refine the estimate. Target offsets, uncertainty, and view disagreement are handed to the Executor **in the same base-frame direction words its actions use** — so grounding output and action input speak one language.

### Asynchronous scheduling

What stays sequential: observe → localise → build context → propose → execute batch → assess outcome. Each step genuinely depends on the last.

What runs concurrently:

- **Monitor thread** — checks updated observations periodically and after motion steps. Informational alerts become evidence. A `STOP` alert cancels the running batch; the Controller finishes reaching the next action boundary, then discards pending commands. The Monitor never picks the fix.
- **Memory writer** — after each outcome assessment, a summary is requested on a background writer. The next subgoal starts without waiting. Planner and Verifier read the *latest completed* note; pending requests are consolidated so the writer folds in everything since its last accepted summary.

Stale-state hygiene: when an attempt ends, its monitoring stops without waiting for an in-flight model call, and any late response from that attempt is thrown away. A fresh attempt gets fresh monitoring context.

An interruption is a *reason to reassess*, not a verdict. After one, the harness verifies, then advances, retries, or sends unresolved requirements back to the Planner. The Verifier can be short-circuited by directly measured facts — a lost grasp or a measured release settles an attempt with no model call.

## Ablation Studies and Experiments

### LIBERO-PRO, zero-shot

Three suites (Goal, Spatial, Object), 10 tasks each, five conditions: Base, Language, Object, Position Swap, Task — 200 configurations.

| Method | Base avg | Perturb avg | Base time (s) | Time score |
|---|---|---|---|---|
| $\pi_{0.5}$ zero-shot | 0.0 | 0.0 | 8.3 | 0.0 |
| MolmoAct2 zero-shot | 0.0 | 0.0 | 9.4 | 0.0 |
| OpenVLA-OFT zero-shot | 0.0 | 0.8 | 34.7 | 0.0 |
| GR00T N1.5 zero-shot | 0.0 | 0.0 | 6.3 | 0.0 |
| CaP-X, 10 loops | 13.3 | 19.2 | 346.4 | 2.3 |
| VoLoAgent (zero-shot VLA) | 0.0 | 0.0 | 500.0 | 0.0 |
| **MotorMind (Qwen3.8-Flash-Next)** | **66.7** | **53.8** | 223.4 | **17.91** |

Per-suite base: Goal 45.0, Spatial 75.0, Object 80.0.

The time score is $\mathrm{TimeScore} = 60s / \bar{T}$ in percentage points per minute, with $s$ success on 0–100 and $\bar{T}$ mean wall time. MotorMind's 17.91 pp/min is ~8× CaP-X's 2.3 and ~6× Harness VLA's best. Read it alongside success — a fast failure scores zero, and the all-zero VLA rows at 6–9 s make that point.

Against **fine-tuned** policies, the perturbation column is where it gets interesting. $\pi_{0.5}$ fine-tuned: 98.3% base, 63.3% perturbed — and only **36.7%** on Position, **23.3%** on Task. MotorMind gets 51.7% on Position and 58.3% on Task without ever seeing a LIBERO demonstration. Fine-tuned OpenVLA-OFT averages 51.2% perturbed; MotorMind 53.8%, a 2.6-point edge.

GR00T N1.5 fine-tuned is a curiosity: 95.0% on Spatial, **0.0%** on Goal and Object. The checkpoint used was `gr00t-n1.5-libero-spatial-posttrain` across all families — a clean illustration of how narrow VLA fine-tuning is.

### Adaptive tasks — where the async design pays

30 online tasks, 600 s budget. Simulation advances on wall-clock time during planning: at 20 Hz, one sim step per 0.05 s of thinking. So latency costs you scene evolution; you cannot think for free.

| Method | Dynamic Reasoning (10) | Scene Shift (10) | Dynamic Manip. (5) | Prompt Shift (5) |
|---|---|---|---|---|
| $\pi_{0.5}$ | 0% | 70% | 0% | 20% |
| GR00T N1.5 | 0% | 10% | 0% | 0% |
| MolmoAct2 | 20% | 50% | 40% | 0% |
| CaP-X | 30% | 20% | 20% | **60%** |
| **MotorMind** | **70%** | **90%** | **80%** | **60%** |

The conveyor tasks (1.5 mm/s, single pass — no second chance) are the clearest: 80% vs 40% best baseline. Dynamic Reasoning asks for targets by property, exclusion, spatial relation at initial arrangement, or cumulative temporal count ("the second distinct food item over the episode"). 70% vs 30%.

Prompt Shift is the one tie. CaP-X also hits 60%, which makes sense — regenerating a program is a natural response to a rewritten instruction.

### Backbone swap

| Suite | Qwen3.8-Flash-Next | GPT-6 Sol (medium) |
|---|---|---|
| Goal | 50% | 70% |
| Spatial | 70% | 100% |
| Object | 80% | 100% |
| **Avg** | **66.7%** | **83.3%** |

Spatial wall time rises 199.0 s → 370.4 s. The architecture is unchanged. This is the paper's strongest claim: VLM progress converts to robot capability with no redesign and no robot data.

(Note the single-seed Qwen numbers here — 50/70/80 — differ slightly from the main table's 45/75/80.)

### Component ablations

| Removed | Success | $\Delta$ |
|---|---|---|
| Nothing (full) | 66.7% | — |
| Verifier | 60.0% | $-6.7$ |
| Replanning | 36.7% | $-30.0$ |
| Planner | **0.0%** | $-66.7$ |

Replanning is the load-bearing piece among the correction mechanisms — 30 points. Verification adds a smaller but real 6.7. Without the Planner the system does nothing at all; subgoals with explicit success criteria are the scaffolding everything else hangs from.

The authors flag the obvious trap: ablated variants have shorter wall times, so their time scores look better. The no-planner variant's time score is meaningless next to 0.0% success.

### What did not work

**More time hurts.** The execution-budget sweep is the most surprising table in the paper:

| Budget | Goal | Spatial | Object | Avg | Time (s) | Time score |
|---|---|---|---|---|---|---|
| 300 s | 40% | 70% | 90% | **66.7%** | **159.8** | **25.03** |
| 450 s (default) | 50% | 70% | 80% | 66.7% | 210.0 | 19.05 |
| 900 s | 40% | 60% | 50% | **50.0%** | 320.0 | 9.37 |
| 3600 s | 50% | 60% | 70% | 60.0% | 296.4 | 12.14 |

Non-monotonic. 300 s matches the default's success with 23.9% less wall time. 900 s *loses 16.7 points*. Extra attempts actively make things worse — more opportunities to disturb a scene past recovery. Knowing when to stop is part of the harness, not an afterthought.

**Rotations are a dead zone for small VLMs.** 0/19 for Qwen. Any task needing deliberate reorientation is out of reach for the cheap backbone.

**Planner-free operation is impossible.** 0.0%.

### Failure taxonomy

Excluding malformed output, failures sort into grounding, completion, planning, progress-checking, and failure-monitoring.

- **Grounding** is the largest share overall, worst under Semantic, Object and Position perturbations — picking the wrong object or wrong target location.
- **Premature completion** ("false done": declaring success while the goal predicate is false) is the other big bucket, worst under Task and Object perturbations.

Both shrink as the backbone gets stronger, which is the authors' argument that these are VLM bottlenecks rather than harness bottlenecks.

## Worth Remembering

**The honesty is unusual.** The paper repeatedly refuses to over-claim from its own diagnostic: "these percentages describe performance on the diagnostic rather than equivalent measures of intrinsic difficulty"; "the diagnostic alone does not establish that repeated checks eliminate these errors"; "they do not isolate control abstraction as the cause." The adaptive suites are called "stress tests rather than precise quantitative evidence." Worth copying as a writing standard.

**Baseline fairness, mostly.** All $\pi_{0.5}$ uses — direct, inside Harness VLA, inside VoLo — share the same `libero_base` checkpoint. Harness VLA's exploration memory is frozen at 5 or 10 attempts and labelled as such. But MolmoAct2's "base weight" configuration uses LIBERO-derived normalisation and robot metadata, which the authors admit "is not free of LIBERO-derived interface information." Not a pure zero-shot row.

**Measured feedback beats model self-report.** The design principle to steal: directly measured outcomes (a lost grasp, a measured release) can settle an attempt *before* any model verdict. Trust the sensor over the generated token wherever you have a sensor. Compare the `false done` failure mode — the model's own completion claim is exactly the signal you cannot trust.

**Role separation by output schema, not by instruction.** The Monitor cannot propose an action because its JSON schema has no action field. This is a much stronger guarantee than a prompt saying "do not propose actions". Transferable to any multi-role LLM system.

**Real robot, no adaptation.** xArm6 + RealSense D455, standard SDK. Direct Perception 97%/100% (bowl/box), Human Perturbation 93%/90%. Semantic Understanding: 80%, 100%, 100%, 60% at 5 trials each. Small $n$ — 5 trials per semantic task means a single failure moves the number 20 points. The case study is good though: the gripper approaches the white bowl instead of the blue cube, the Monitor flags the mismatch, the Executor re-localises from a fresh observation and finishes correctly.

**Honest limitations.**
- Everything here is pick-and-place and table-top. No contact-rich work, no tool use, no bimanual.
- Latency is intrinsic. 223 s per LIBERO episode against ~8 s for a direct VLA. The GPT-6 Sol upgrade buys 17 points for ~2× the time.
- Grounding without a segmentation model works, but grounding errors are the top failure cause. The SAM-free claim is architectural cleanliness bought with accuracy.
- The 300 s result hints the tuning surface is non-obvious. Budget is not a "set it high and forget" knob.
- Single-seed numbers in the backbone comparison; the small per-suite deltas there should not be read too closely.

**Concurrent work.** Show-Harness shares the thesis. Differences, per the authors: Show-Harness is primarily sequential, formulates control as tool selection over predefined Python plugins, uses predefined or rule-selected motion increments, and recovers *reactively* after failure. MotorMind monitors concurrently, decomposes decisions across five roles, has the VLM reason motion magnitudes directly, and interrupts *proactively*. No head-to-head experiment — stated as a scope decision.

**Follow-up questions.**
1. Does the Monitor's STOP firing rate correlate with task success? The ablations remove replanning and verification but never the Monitor alone, which is odd given it is the paper's headline mechanism.
2. If rotation accuracy is ~0% for Qwen, how many of the 33% base failures are rotation-shaped? Would constraining the vocabulary to translation + gripper improve things?
3. The millimetre magnitudes are VLM-reasoned. How sensitive is success to systematic over- or under-shoot, and would a calibration layer on the Controller help more than a bigger model?
4. Budget non-monotonicity: is 900 s worse because of scene disturbance, memory-note drift, or accumulated grounding error? The paper does not separate these.

## Links

Related: [[Transferring the Intelligence of VLMs to Robotic Control]] · [[Grounded Action Model- 3D Grounding as a Foundation for Robotics]] · [[In-Context Robot Learning with VLM Agents]] · [[Think Like a World Model, Act Like a VLA- Distilling World-Model Representations into Compact Robot Policies]] · [[EXIMO- VLM Guided Exploration of VLA Policies]] · [[Breaking the Vision-Action Shortcut- Latent Interface Training for Generalizable Robotics Foundation Models]] · [[Coding Agents for Generalized Task and Motion Planning Problems]] · [[An Empirical Study of Harness Design for Coding Agents]] · [[WHALE- A Simple Recipe for Joint Harness-Weight Optimization]] · [[RRSI- Regularized Recursive Self-Improvement of Agent Harnesses]] · [[Agentic Workflows]] · [[Planning and Decomposition]] · [[Reflection]] · [[Tool Use]] · [[Memory]] · [[Human in the Loop]] · [[Agent State and Checkpointing]] · [[Multi-Agent LLM Systems]] · [[Structured Output]] · [[Agent Evaluation]] · [[Grounding]] · [[Imitation Learning]] · [[Feedback Loop]] · [[Model Predictive Control]] · [[Observability]] · [[Diffusion Policy]] · [[RoboFollow- Unveiling the Instruction Following Mirage in Embodied Agents]] · [[PolicyGuide- From Guarding One Action to Guiding the Whole Workflow for Policy-Compliant LLM Agents]]

New topics worth writing: LIBERO and LIBERO-PRO as a manipulation benchmark, vision-language-action models as a family, mid-level vs low-level action abstraction in robot control, asynchronous monitoring and interruption in agent harnesses, base-frame vs image-frame spatial reference confusion in VLMs, embodied question-answering diagnostics, time-normalised success metrics, execution-budget non-monotonicity, false-completion as an agent failure mode, embodiment transfer without fine-tuning, xArm6 and RealSense deployment practicalities
