---
title: "FACET: Preserving Source Intent and Executable State in Terminal Task Synthesis"
authors: ["Kou Shi", "Zun Wang", "Qisheng Su", "Shiting Huang", "Ziao Zhang", "Zhen Fang", "Qingnan Ren", "Jin Liu", "Yu Zeng", "Yiming Zhao", "Lin Chen", "Zehui Chen", "Feng Zhao"]
year: 2026
arxiv: "2608.18580"
url: https://arxiv.org/abs/2608.18580
priority: Low-Priority
read_on: 2026-09-29
tags: [paper, vision]
---
## The Core Idea

A "terminal task" for training an agent is not one thing. It is a bundle of four things that must agree with each other:

1. an **instruction** ("clean this CSV and produce a report at `/out/report.md`"),
2. an **environment** — a Docker image with the files, packages and services already in place,
3. a **reference solution** — a shell script that actually does the job,
4. a **verifier** — a `pytest` suite that checks the final state.

If any one of these disagrees with the others, the task is garbage. The instruction names `data/input.csv` but the environment only built `data/raw.csv`. The solution assumes `pandas` is installed and it isn't. The verifier checks for a database table the task never creates. The task is either impossible or graded wrongly, and either way you have poisoned your training data.

FACET's claim is that this disagreement happens because the four artifacts are generated **from text passed between stages**, not from a real running machine. Each generator sees a written specification and independently imagines what the filesystem looks like. They imagine differently.

The fix is blunt and obvious in hindsight: **build the container first, then generate everything else while looking at the actual container**.

> [!NOTE] Executable-state grounding
> Build and boot the environment before writing the instruction, solution or verifier. Expose the realised container state $e_0$ — real file paths, real schemas, real installed package versions, real open ports — as read-only context to every later generator. Any late change to the environment (a renamed file, a different port) is automatically seen by all of them, because they read the machine, not a document. ^executable-state-grounding

The second idea is about the *input* side. FACET builds tasks from ~71K scraped "agent skills" (portable procedural how-to packages from OpenClaw, ClawHub, GitHub). The naive move is: sample a few related skills → ask a model to write a task. That throws away almost everything. You get a shallow workflow that uses the single most obvious capability of each skill and forgets the dependencies between them.

> [!NOTE] Source-intent preservation
> Rich source material encodes goals, tool dependencies, input/output contracts, intermediate states and procedural constraints. Each generation stage compresses. Compress twice and you have a one-line task description. FACET inserts an explicit *reconstruction* stage that expands the skills back out into a full scenario before any compression happens. ^source-intent-preservation

What it unlocks: tasks with **22.77 executable checks each** on average, roughly 1.4× the next densest dataset and 7× some others. And 1.2K trajectories from those tasks are enough to move Qwen3.5-27B from 40.82 → 47.57 on Terminal-Bench 2.1.

## The Methodology

### The formal object

A task bundle is

$$\mathcal{T} = (\mathcal{I}, \mathcal{E}, \mathcal{S}, \mathcal{V}, \mathcal{M})$$

instruction, environment spec, solution, verifier, metadata. Booting gives $e_0 = \mathrm{Init}(\mathcal{E})$; running the solution gives $e_T = \mathrm{Run}(\mathcal{S}, e_0)$ or failure $\bot$. Write $B(\mathcal{E})$ for "the image builds" and $\nu_\mathcal{V}(e)$ for "the verifier passes on state $e$". Accept the task only when

$$\mathcal{A}(\mathcal{T}) = B(\mathcal{E}) \wedge \neg\nu_\mathcal{V}(e_0) \wedge (e_T \neq \bot) \wedge \nu_\mathcal{V}(e_T)$$

Read that in plain words, left to right: it builds; the verifier **fails** on the untouched starting state (so the task isn't already solved — this is the non-triviality clause and it is the one people forget); the reference solution runs without crashing; the verifier passes afterwards.

### Stage 1 — getting the raw material

Scrape skill packages. Drop anything unsafe, anything needing private websites or non-public resources, anything unreadable or duplicated. 71,341 skills survive, spread over 5 top-level families (AI/agents 21.3%, software/systems 21.1%, data/analysis 17.4%, documents/productivity 15.8%, multimedia 24.4%) and 34 fine-grained categories.

Each skill is normalised into a record: description, required tools, inputs, outputs, procedural steps, provenance.

Then the combination step. An extraction agent guesses, for each skill, what *situations* it would be used in — user goals, likely initial state, desired final state. Those scenario hypotheses get [[Embeddings|embedded]], and nearest-neighbour retrieval finds skills whose hypotheses look similar. Grouping them gives candidate pairs $p_c = (c, X_c)$: a scenario plus a set of skills.

A model judge keeps only combinations that are relevant, complementary, non-redundant, and runnable as a terminal workflow:

$$\mathcal{P} = \{p_c \mid J(p_c) = 1\}$$

### Stage 2 — scenario reconstruction

This is the anti-compression stage. The pipeline is

$$p_c \xrightarrow{\text{reconstruct}} D_c \xrightarrow{\text{synthesise}} C \xrightarrow{\text{build refs}} (R_S, R_I)$$

Five reconstruction modules run in sequence:

- **Skill analysis** — for each skill, pin down capabilities, tools, inputs, outputs, preconditions, observable effects.
- **Scenario exploration** — propose concrete settings where these skills serve one shared user objective.
- **Association and filtering** — throw out scenarios that merely place unrelated operations side by side.
- **Evolution and recovery** — the important one. Arrange the capabilities into a *workflow*, and recover the cross-skill dependencies: which intermediate artifact does step 3 need from step 1, what state transition connects them.
- **Information expansion** — add concrete resources, formats, constraints, observable success conditions.

The result is described along five separate axes, deliberately kept apart so nothing gets lost in a single prose blob:

$$D_c = \{d_{\mathrm{goal}}, d_{\mathrm{context}}, d_{\mathrm{capability}}, d_{\mathrm{state}}, d_{\mathrm{io\text{-}tool}}\}$$

goal (objective + deliverables) · context (setting, motivation) · capability (each skill's role and its relation to the others) · state (initial, intermediate, final) · inputs-outputs-tools (files, formats, schemas, paths, services, dependencies).

A model fuses the five into one natural-language scenario $C$. From $C$ come two references, **in this order**:

$$R_S = f_S(C), \qquad R_I = f_I(C, R_S)$$

Solution reference first — setup actions, workflow, intermediate artifacts, state transitions. Instruction reference second, generated from both $C$ and $R_S$ — goal, inputs, outputs, deliverables, constraints. A consistency checker verifies the two agree on the initial state and the target, that every instruction requirement is backed by a solution step, and that every required effect is observable in the final state.

### Stage 3 — build the machine, then write about it

**Environment construction.** Generating a whole file-rich environment in one model response does not work, so planning and materialisation are split. The agent first emits a *manifest* (directories, files, services, dependencies, expected properties), then materialises it inside a restricted base image — with network access, shell and Python available, so it can download real public resources, reshape them, or procedurally generate text and binary assets.

Two details worth copying:

- Downloaded resources are **localised into the build context**, so the finished task needs no network at evaluation time.
- Fixtures are deliberately *perturbed and augmented* — extra records, extra metadata fields, distractor entries, cross-file relations — to stop the task being a template with three rows of toy data. This is what makes a 22-check verifier possible.

Build, run init checks. Compiler errors, missing packages, malformed fixtures, failed downloads, dead services go back to the environment agent. **At most 3 repair iterations.** Repair is conditioned on the failure trace *and* the specification $Z = (C, R_S, R_I)$ — specifically so the agent can't quietly delete a requirement just to get a green build.

**Then artifact generation, forward order:**

1. Observe $e_0$ (the realised state — a read-only record of files, dirs, schemas, deps, services).
2. Instruction $I$ from $R_I$ and $e_0$.
3. Solution $S$ from $I$, $R_S$, $e_0$.
4. Execute $S$ to get $e_T$.
5. Verifier $V$ from $I$, $R_S$, $e_0$ **and** $e_T$.

The verifier checks behaviour and final state, not exact commands, so an agent that solves the task a different way still passes. Nondeterministic values (timestamps, generated IDs, irrelevant ordering) are normalised.

**Validation and targeted repair.** Package in Harbor format: `environment/`, `solution/`, `tests/`, `instruction.md`, `task.toml`. Then the four checks of $\mathcal{A}(\mathcal{T})$, each in a *fresh* container — baseline validation (verifier must return reward 0 on the untouched state) and oracle validation (solution then verifier, must return reward 1) never share a container.

On failure, a constrained router reads the trace and blames exactly one artifact:

| Failure | Blamed |
|---|---|
| Build / init fails | environment |
| Solution crashes | solution |
| Test collection error, bad assertion | verifier |
| Instruction refers to something not there | instruction *or* environment, per source |

Only that component is regenerated. **At most 5 task-level repair rounds**, full Docker lifecycle each time. Tasks are never accepted out of an incrementally-patched debug container.

### Training

Terminus-2 scaffold driven by DeepSeek-V4-Pro rolls out on ~6K validated tasks. 1,200 fully successful trajectories become the SFT set. Full-parameter fine-tuning of Qwen3.5-4B/9B/27B with LLaMA-Factory, BF16 + ZeRO-3 (see [[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models]]), 3 epochs, effective batch 64, LR $1\times10^{-5}$ cosine with 0.1 warmup ratio, 32,768-token sequences, 8× H200.

## Ablation Studies and Experiments

### Main result

Terminal-Bench 2.1, Terminus-2 scaffold, 3 attempts per task, mean pass rate:

| Model | Base | +FACET | Δ |
|---|---|---|---|
| Qwen3.5-4B | 17.60 | 24.72 | **+7.12** (+40.5% rel.) |
| Qwen3.5-9B | 27.34 | 35.58 | **+8.24** |
| Qwen3.5-27B | 40.82 | 47.57 | **+6.75** |

Reference points under the same setting: Qwen3.5-397B-A17B scores 49.06, Kimi-K2.6 (1T) 59.93, DeepSeek-V4-Pro-Preview (1.6T) 73.03. So the fine-tuned 27B lands 1.49 points below a model ~15× its size.

1.2K trajectories. That is the headline for anyone building a data pipeline: the gain came from task quality, not volume.

### Dataset comparison — and the deliberately bad-looking number

| Dataset | #Tasks | Tests/task | P@1 | P@3 | Turns |
|---|---|---|---|---|---|
| Nemotron-Terminal | 15K | 6.18 | 40.67 | 48.00 | 6.12 |
| Endless-Terminals | 2,492 | 5.51 | 83.00 | 87.00 | 4.53 |
| Terminal-Lego | 15K | 16.60 | 47.00 | 49.00 | 5.77 |
| TerminalWorld | 1,530 | 3.98 | 57.00 | 82.00 | 11.94 |
| Tmax | 15K | 3.29 | 80.00 | 86.00 | 11.14 |
| **FACET** | **6,078** | **22.77** | **27.00** | **35.00** | 11.86 |

FACET has the *worst* pass rate by a wide margin. That is the argued-for outcome, not an embarrassment: success is conjunctive over all checks, so 22.77 checks is a much narrower gate than 3.29. Endless-Terminals at 83.00 P@1 with 5.51 checks is close to saturated — it cannot discriminate between a good agent and a very good one.

Caveat the authors state plainly: the trajectory columns, task columns and the 100-task P@1/P@3 samples are drawn **independently**. These are dataset-level characteristics, not measurements on identical instances. Only the scaffold (Terminus-2), solver (DeepSeek-V4-Pro) and seed (42) are held fixed.

### Generation order — the cleanest ablation in the paper

Same 100 scenario–skill pairs, three orders after the environment is built:

- **Forward** (FACET): $I \to S \to V$
- **Reverse**: $I \to V \to S$ — verifier written from the textual spec, before any solution exists
- **Joint**: all three in one model call

| Scheme | Reached validation | Initially valid | Final yield |
|---|---|---|---|
| Forward | 99 | 46 (46.5%) | 83/100 |
| Reverse | 91 | 22 (24.2%) | 63/100 |
| Joint | 96 | 36 (37.5%) | 65/100 |

The *failure composition* is the interesting part:

- **Reverse**: 56.5% of failures are cross-artifact contract mismatch. Writing the verifier before you have seen a working solution means you assert against a state nobody has produced.
- **Joint**: contract mismatch drops to 13.3% — one call, one consistent story — but errors move to fixture/schema/path grounding (38.3%) and infrastructure/dependency failures (21.7%). Doing it all at once keeps the artifacts consistent with *each other* and inconsistent with the *machine*.
- **Forward**: highest validity, most evenly spread failures, 37.7% contract mismatch.

Paired sign test on the 88 pairs reaching validation in all three schemes: Forward beats Reverse on 29 pairs, loses on 9, $p = 0.0017$ two-sided. Forward vs Joint is 27 to 18, $p = 0.233$ — **not significant**. The authors say so. So the well-supported claim is narrow: *generate the solution before the verifier*. Whether staging beats one big call is unresolved.

Two more honesty notes: Forward got 5 repair rounds, Reverse and Joint got 3, so the final-yield column is a comparison of pipeline configurations, not repair efficiency. And negative-discrimination (does a partial solution correctly fail?) had wildly uneven coverage — 95/96 for Joint, 7/99 for Forward — so those numbers are descriptive only.

Cost: Joint is 1 call, 2.80 min/task. Forward 3 calls, 4.14 min. Reverse 3 calls, 5.19 min.

### End-to-end pipeline ablation, 500 shared skill pairs

| Pipeline | Packages | Validated | Yield | P@1 | P@3 | Avg. cmds |
|---|---|---|---|---|---|---|
| Baseline (no reconstruction) | 437 | 78 | 15.6% | 80.8% | 85.9% | 12.8 |
| TW (TerminalWorld repro) | 449 | 139 | 27.8% | 58.8% | 64.0% | 17.0 |
| FACET | 395 | **350** | **70.0%** | 25.1% | 33.1% | 21.5 |

The gap between "Packages" and "Validated" is the whole thesis in one column. Baseline produces *more* complete file bundles (437 vs 395) and validates 78 of them. Writing all the right files is easy; making them agree is not.

Of FACET's 350, only **182 passed first validation** — 168 were recovered by targeted repair. So roughly half the final dataset exists because of the repair loop.

And the yield did not come from easier tasks: FACET's validated tasks have the lowest pass rates and the most commands per rollout.

Construction funnel over 7,852 seeds: 84.56% initial environment build success → +874 recovered by env repair → 95.70% successful environments → 7,446 reach validation → **38.35% first-pass valid** → +3,222 recovered by task repair → 6,078 final (81.63%). 58 candidates were dropped pre-validation for weak verifiers, ambiguous deliverables, unresolved external dependencies, or workflows with trivial bypasses.

### What the check-level analysis reveals

Across teacher rollouts, **89.40% of individual checks pass but only 20.94% of rollouts fully succeed.** Among failures, **54.00% fail only one or two checks** — the agent built the main artifact, ran the workflow, then got one field value wrong or skipped a secondary deliverable.

This is [[Credit Assignment|credit assignment]] at its worst. Binary task reward throws away the information that the trajectory was 95% right. The authors are careful not to promote check-pass-rate to a metric: checks differ in granularity and importance, one root error can trip several assertions, and one assertion can encode the whole task.

Also measured, on the 1,270 successful trajectories (15,075 turns, 39,136 commands): `cat`, `python3`, `ls` are 69.5% of all commands; top ten are 88.4%; 84.7% of commands are observations. 95.6% of trajectories open with an observation-only turn. After an action-only turn, 53.1% of transitions return to observation vs 28.7% continuing to act. A legible observe–act–verify loop — though the authors note they never compared against *failed* trajectories, so this characterises success without explaining it.

## Worth Remembering

**The transferable principle is ordering, not architecture.** There is no new model, no new loss. The result is: realise state before you describe it, and produce the solution before the thing that grades the solution. That second point generalises well beyond terminals — it is the same reason [[Evals]] written before you have seen a working output tend to test the wrong thing.

**Build the environment first is an expensive commitment.** Every task costs a Docker build, up to 3 environment repairs, up to 5 task repairs, each a full container lifecycle. Forward costs 4.14 min of generation latency on top. You are trading compute for validity at maybe 4:1 versus Joint, for a yield advantage of 83 vs 65 per 100 — and that comparison is confounded by unequal repair budgets.

**Repair is not a detail; it is half the dataset.** 168 of 350 tasks in the ablation and 3,222 of 6,078 in the main run came from repair. If you copy this, copy the *targeted* router — regenerating the whole bundle on any failure discards valid components and probably loops.

**The non-triviality clause $\neg\nu_\mathcal{V}(e_0)$ is the check people skip.** Without it you accept tasks whose verifier passes on an untouched container. Your agent gets reward for nothing, and you will not notice until the eval numbers look suspiciously good.

**Low pass rates are a feature, and this cuts against most dataset papers.** Datasets at 80%+ P@1 have almost no headroom left to teach with. Related to [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] in spirit: the easy-benchmark trap is not specific to recsys.

**Limitations the authors state themselves**, which is more than most:
- Forward vs Joint is not statistically separated ($p = 0.233$).
- Unequal repair budgets across schemes.
- Negative-discrimination coverage is 7/99 vs 95/96 — the numbers are not comparable.
- The TW reproduction needed adapters for skill-pair input and Harbor packaging, so it may differ from the original system.
- Pipelines retain different subsets of inputs, so their P@1 differences are descriptive, not causal.
- The command parser cannot recover dynamically generated shell operations.
- Skill-tag pass rates: 7 tags per task means the groups overlap heavily and the estimates are not independent.

**Practical caveats if you wanted to use this.** All numbers ride on a specific scaffold (Terminus-2), a specific solver (DeepSeek-V4-Pro), and a fixed 32,768-token context — and the FACET models were *evaluated* at 32,768 while baselines used their official maximums, which is not a like-for-like context budget. Trajectory turn counts are also scaffold-dependent; "11.86 turns" is not a portable difficulty measure. The task corpus tilts towards whatever the 71K scraped skills cover, which is heavy on file manipulation and document/data wrangling and thin on anything needing live network access at eval time (deliberately, since resources are localised).

**Follow-up questions.**
- The 54%-fail-one-or-two-checks finding is a direct invitation to dense reward. Partial credit over verifier checks would give a far less noisy signal than binary outcome — see the framing in [[DRACO- Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training]]. Why did they not try it?
- Only successful trajectories were used for SFT. The 4,796 failures, most of them near-misses, are thrown away. That is a lot of signal on the table for anything preference-based or for [[Offline RL]].
- The authors flag RL as future work. With dense executable checks and a shaped return, these tasks look like a decent [[Reward Function|reward function]] — with the usual [[Reward Hacking]] worry that the agent learns to satisfy 22 assertions rather than do the job.
- Does the density actually cause the transfer, or is it the 11.86-turn length? Both correlate and neither was isolated.

## Links

Related: [[Evals]] · [[Agent Evaluation]] · [[Agentic Workflows]] · [[Credit Assignment]] · [[Reward Function]] · [[Reward Hacking]] · [[Sandboxing]] · [[Tool Use]] · [[Fine-Tuning]] · [[Instruction Tuning]] · [[An Empirical Study of Harness Design for Coding Agents]] · [[DRACO- Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training]] · [[EnvHarness- Awakening Static Worlds for Agent Learning]] · [[AgentMercury- Your Agent Can Synthesize Verifiable Environments for Business Scenarios at scale]] · [[ScienceIDE- Turning World's Scientific Codebase into Agent Learnable Environments]] · [[What Makes Good Agentic Data- An ACE Lens on Data Generation for LLM Agents]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models]] · [[Embeddings]] · [[Offline RL]]

New topics worth writing: Terminal-Bench, Harbor task format, agent skill packages as a data source, executable verifiers and conjunctive success criteria, synthetic environment construction, dense vs sparse verifier design, targeted artifact repair loops, cross-artifact consistency in synthetic data pipelines, teacher-rollout filtering for SFT, exact sign test for paired pipeline comparison
