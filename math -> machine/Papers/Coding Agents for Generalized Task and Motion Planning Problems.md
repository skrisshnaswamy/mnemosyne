---
title: "Coding Agents for Generalized Task and Motion Planning Problems"
authors: ["Merler et al."]
year: 2026
arxiv: "2609.30233"
url: https://arxiv.org/abs/2609.30233
priority: Good-To-Read
read_on: 2026-09-28
tags: [paper, llm, rl, vision]
---
## The Core Idea

Task and motion planning (TAMP) is the problem of deciding *what* to do and *how to move* at the same time. Pick which block to move (discrete), and also find a collision-free arm trajectory that actually does it (continuous). The two halves are coupled: the discrete choice you make determines whether any motion at all is feasible.

Classic TAMP systems solve this with a lot of hand-built machinery — symbolic predicates, operators, samplers for continuous parameters, motion planners. **Generalized TAMP** tries to reuse effort across problem instances, usually by learning a sampler or a feasibility predictor, but that still needs the same TAMP scaffolding underneath.

The claim here: **you do not need any of it.** Hand a modern coding agent a text description of the task and a `reset`/`step` simulator client, give it $20 of model usage, and let it write a plain Python program. Freeze that program. Evaluate on 100 unseen instances with no LLM in the loop at all. The programs beat hand-engineered TAMP planners — 82% mean success for Claude Code (Opus 5) and 95% for Codex (GPT-6 Astra) against 47% for the planner, on the 16 environments where a planner exists.

Why this was not obvious: there was genuine evidence pointing the other way. LLMs are great at software benchmarks and classical symbolic planning, but when prior work asked LLMs to make the *geometric* decisions inside a TAMP system — pick a grasp pose, pick a placement — they did badly, even with the geometry written into the prompt. So the question was whether coding skill transfers to physical reasoning.

The answer seems to be yes, but through a specific route: **the agent does not reason about geometry in its head; it writes experiments and measures.** One run built a kinematic model of a Kinova arm from its own memory, then grasped a cube as a marker (the state exposes object positions but not the gripper position), moved the arm around, and fitted six mount/grasp-offset parameters — cutting prediction error from 38.9 mm to 1.8 mm. That fitted model then served as its inverse-kinematics solver. That is [[System Identification]] done by an agent, unprompted.

> [!NOTE] Generalized TAMP
> Finding a single reusable solution that works across many instances of the same task family (different object counts, poses, geometries), instead of replanning from scratch for each one. ^generalized-tamp

> [!NOTE] Programmatic policy
> The output here is a Python class with `reset()` and `get_action(state)`. It can carry internal state between steps, so the policy is really $a_t = \pi(h_t)$ over the history $h_t = (s_0, a_0, \dots, s_t)$, not just the current state. ^programmatic-policy

## The Methodology

**Setting.** Each environment is a finite-horizon goal-directed MDP $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \rho, H \rangle$ — see [[Markov Decision Process]]. States are fully observed and object-centric: every typed object maps to a feature vector (pose, geometry, joint angles, velocity). Reward is sparse — it fires on goal achievement. $\rho$ samples instances that differ in object count and layout, so no fixed action sequence can work.

**Synthesis loop.** The agent gets:

- a task description $d$ (environment name, observation and action spaces, goal, a note that object counts vary),
- a simulator *client* exposing `reset` ($\rho$), `step` ($P$), and a render-to-image helper,
- the evaluation timeout $\tau$ it will face,
- a $20 model-usage budget.

It gets **no** predicates, operators, samplers or skills. The main setting also gives it a bare sandbox — Python, NumPy, SciPy, nothing else — so it cannot lean on an existing motion-planning library. The real simulator implementation runs *outside* the container on a server; the agent only holds a client. Everything runs in Docker with no network, and they red-teamed the isolation (attempts to read env source, import forbidden libs, reach the host). Relevant to [[Sandboxing]].

Within the budget the agent decides everything: what to probe, what tests to write, when to rewrite. It is asked to `git commit` before each test, which gives a replayable history of how the program evolved — a nice trick for [[Observability and Tracing]] of an agent run.

**Evaluation.** Program frozen. 100 held-out instances per environment, seeds drawn randomly and verified afterwards to be unused during synthesis. 60 s wall-clock timeout per instance (matching KinDER's protocol). Metric: success rate, plus per-instance *policy computation time* (planning and action selection only, excluding simulator stepping and excluding synthesis).

**Scale.** 28 environments × 7 synthesis methods × 5 runs × 100 instances = 980 programs, 98,000 episodes.

**Environments.** KinDER's 25 environments in four families — Kinematic2D, Dynamic2D, Kinematic3D, Dynamic3D — plus three PDDLStream domains (Packing, Blocked, Rovers) where LLMs had previously been shown to fail. Kinematic means no dynamics; dynamic means contact, friction and velocity matter, so you need sweeping, pouring, tossing. 19 environments have variants with more objects than the original benchmark tested.

**Agents compared.**

| Name in the paper | Backend |
|---|---|
| *Opus* | Claude Code, Opus 5 (high) |
| *Sol* | Codex, GPT-5.6 Sol (medium) |
| *Astra* | Codex, GPT-6 Astra (high) |

**Baselines.**

1. **TAMP planners** — the benchmark's own hand-written predicates, operators, samplers and motion planners. Replans per instance. Available for 16 of 28 environments (PDDLStream itself for the three PDDLStream domains).
2. **LLMGenPlan** — a re-implementation of Silver et al.'s generalized planning method, run on Opus 5 with chain-of-thought and thinking disabled. It gets the **full environment source code** and the same $20 budget, but no tools, no filesystem, no ability to run code. A fixed harness evaluates each program and returns canned feedback: an exception plus traceback, an invalid action, or an unsolved instance with its seed. This isolates "being agentic" from "being a good LLM".
3. **One-shot** — LLMGenPlan's first program, no refinement.
4. **+ source** — Opus and Astra with the environment source code inside the container. They can import the real IK solver, call the motion planner, read the success check, and set arbitrary states (i.e. generative access to $P$).

## Ablation Studies and Experiments

**Headline, on the 16 environments with a planner:** planner 47%, Sol 56%, Opus 82%, Astra 95%. Agents beat the planner in 15/16 environments for Astra, 12/16 for Opus, 9/16 for Sol.

**Per-family means (main setting, all 28 environments):**

| | Kin2D | Dyn2D | Kin3D | Dyn3D | PDDLStream |
|---|---|---|---|---|---|
| Astra | 99% | 97% | 93% | 65% | ~100% |
| Opus | 96% | 92% | 90% | 45% | 78% |

**The agentic ablation is the cleanest result in the paper.** LLMGenPlan uses the *same model* (Opus 5), the *same budget*, and gets *more* information (full source code), yet averages 28% across 28 environments. Opus + source beats it in **27 of 28**. One-shot is near-zero almost everywhere. So the gain is not "big model writes good code" — it is the ability to choose experiments, run them, and revise. Compare [[An Empirical Study of Harness Design for Coding Agents]] and [[Reflection]].

**Source access helps, but less than being agentic.** Opus 74% → 84%; Astra 86% → 95%. Two distinct mechanisms:
- *Information.* Without source, the agent infers the goal from sparse reward and rendered images, and sometimes gets it wrong. In SortClutteredBlocks, main-setting Astra programs assign cubes to bins by their order in the state vector — works for 4 cubes, fails on every 20-cube instance. With source they read the true target bin: 43% → 95%.
- *Reuse.* Loading the robot model for IK, calling the real motion planner and collision checks, running a private simulator to test actions. One Opus + source run used internal collision data to work out that the *gripper*, not the held part, was hitting the rack, and changed the grasp for clearance.

**Source access makes programs slower.** On the 15 environments where all four configurations hit 100% in at least one run:

| Config | ms per action (mean, [min–max] over env means) |
|---|---|
| Astra | 1.3 [0.03–6.0] |
| Opus | 11.7 [0.04–109] |
| Opus + source | 42.5 [0.06–382] |
| Astra + source | 50.4 [0.05–477] |

The reason is instructive: with source, programs call the environment's own planner at decision time. Without it, the agent is forced to *distil* what it learned into self-contained code. Denial of a crutch produces a faster artefact.

**Style difference between agents.** Astra writes compact closed-form control — hardcoded joint poses, analytic IK. Opus more often precomputes large candidate sets and searches over them at decision time with retry/fallback logic. Hence the 9× speed gap.

**Versus the planner on compute.** On the 14 planner-available environments with multiple object counts: Opus 2.1 s, Astra 0.5 s, planner 29 s per instance. And as object count grows the planner's success *falls* while the agents' holds.

**Astra > Opus is mostly consistency, not ceiling.** Astra is better in 20/28 environments, but in 11 of those the *best* Opus program is at least as good as the best Astra program. Shelf: Opus runs range 0%–100% (the worst never figures out which shelf; another puts one cube right and leaves the rest on the floor). Every Astra run was tested up to 8 cubes and scores ≥98%.

**Interaction visibly changes the program.** Traceable through the git history:
- BalanceBeam, a Sol run: moving small-block placement targets closer to the beam centre, 6% → 64%.
- Blocked, a Sol run: one commit adding a "fall back to the distant unobstructed block" branch, 15% → 56%, with 41 newly solved instances and *zero* regressions. Final 83% — solving 80/80 instances that have an alternative block, but only 3/20 that do not. Astra combined both behaviours (reliable awkward-orientation grasp *plus* fallback) for 100%.

**Unexpected strategies** — the qualitative evidence they lean on hardest, because these do not appear in any published solution and are therefore hard to explain as memorised training data:
- Dynamic2D ScoopPour: rotate the tool and regrasp from the left, scooping far more balls per pass.
- SweepIntoDrawer: ignore the provided sweeper entirely and push cubes one at a time with the gripper.
- Non-prehensile manoeuvres and uses of environment layout generally.

**What did not work.** Dynamic 3D tasks needing sweeping or pouring of many small objects remain largely unsolved in the main setting:

| Environment | Sol | Opus | Astra | Opus+src | Astra+src |
|---|---|---|---|---|---|
| SweepSimple | 0.00 | 0.00 | 0.03 | 0.07 | 0.44 |
| Dyn3D ScoopPour | 0.00 | 0.09 | 0.14 | 0.38 | 0.42 |
| SweepIntoDrawer | 0.00 | 0.00 | 0.72 | 0.57 | 0.86 |
| ConstrainedCupboard | 0.00 | 0.08 | 0.29 | 0.47 | 1.00 |

Even with source code, SweepSimple tops out at 44%. Contact-rich many-object dynamics is where the approach actually breaks.

Also: Kinematic3D Packing exposed a real strategy failure in Opus. Its best program places right triangles at fixed hand-tuned positions and has no mechanism at all for equilateral ones. Three Astra programs instead search over position *and* rotation for every part and solve everything. Hand-tuned constants are the agentic equivalent of overfitting.

## Worth Remembering

- **The evaluation is genuinely LLM-free.** No model call at test time, no retrieval, no adaptation. Whatever the agent learned had to be compiled into Python. That is a much stronger claim than "an LLM agent solved the task", and it is why the compute-per-action numbers are meaningful.

- **Contamination.** The authors admit training data is undisclosed and prior exposure to benchmark code cannot be ruled out. Their defence is the synthesis logs (visible probing, calibration, dozens of revisions) and the novel strategies. Reasonable, not conclusive. **Current best understanding**, not established fact.

- **The setting is friendly.** Fully observed, object-centric states. No perception, no language grounding, no sim-to-real. Simulator access during synthesis is assumed. Under perception uncertainty none of this is tested. If you were porting this idea, the "write a program that probes the environment" loop needs a probe-able environment.

- **$20 per environment.** That is the whole synthesis cost. Against the human-weeks of engineering predicates, operators and samplers for a TAMP domain, this is the actual headline for a practitioner.

- **Generalisable pattern beyond robotics.** Agent + simulator + frozen program + held-out evaluation is a template. It is very close in spirit to the harness-synthesis line — [[Harness-Zero- Harness Distillation via Agent-as-Harness]], [[WHALE- A Simple Recipe for Joint Harness-Weight Optimization]], [[Recursive self-improvement of AI research agents]] — except the artefact being optimised is a *policy* rather than a *harness*.

- **The git-commit-before-each-test protocol is worth stealing.** It gave them a per-commit held-out success curve, which is how they could attribute a +41-instance jump to one specific fallback branch. Most agent studies cannot say which edit did the work.

- **Caveat on variance.** Five runs per method, and the run-to-run spread is huge (Shelf: 0%–100% for Opus). Means alone would mislead; the [min–max] columns are doing real work. Any production use of this would need best-of-$n$ selection on a validation set, which is essentially what the +source Astra configuration achieves by being more consistent.

- **Open question it raises:** the agent's advantage here is *experimentation*, not physical intuition. Would a much weaker model with the same probing loop close most of the gap? Sol (GPT-5.6, medium) at 56% versus Astra at 95% suggests model strength still matters a lot, but nobody ran the "weak model, unlimited probing budget" cell.

## Links

Related: [[Markov Decision Process]] · [[Agentic Workflows]] · [[An Empirical Study of Harness Design for Coding Agents]] · [[System Identification]] · [[Planning and Decomposition]] · [[Sandboxing]] · [[Tool Use]] · [[Reflection]] · [[Reward Function]] · [[Imitation Learning]] · [[Diffusion Policy]] · [[Model Predictive Control]] · [[Monte Carlo Tree Search]] · [[Observability and Tracing]] · [[Test-Time Compute]] · [[Evals]] · [[Harness-Zero- Harness Distillation via Agent-as-Harness]] · [[Recursive self-improvement of AI research agents]] · [[The Bitter Lesson (essay)]] · [[Transferring the Intelligence of VLMs to Robotic Control]] · [[Grounded Action Model- 3D Grounding as a Foundation for Robotics]] · [[In-Context Robot Learning with VLM Agents]]

New topics worth writing: Task and Motion Planning, PDDLStream, Program synthesis as policy representation, Inverse kinematics, Non-prehensile manipulation, KinDER benchmark, Code-as-Policies
