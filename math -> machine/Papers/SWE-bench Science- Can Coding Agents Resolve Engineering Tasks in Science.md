---
title: "SWE-bench Science: Can Coding Agents Resolve Engineering Tasks in Science?"
authors: ["Zhipeng Xu", "Jiahao Lu", "Yining Zheng", "Yuxin Wang", "Xipeng Qiu"]
year: 2026
arxiv: "2608.19799"
url: https://arxiv.org/abs/2608.19799
priority: Low-Priority
read_on: 2026-10-05
tags: [paper, theory]
---
## The Core Idea

Scientific software is not just software. When a physics simulation or a chemistry parser has a bug, the broken thing is not a button in a web app — it is the number a paper reports. A patch that makes the visible test go green can still corrupt the science underneath.

**SWE-bench Science** is a benchmark built around that gap. 119 repository-level repair tasks, pulled from 98 real GitHub repositories, across 20 scientific fields (chemistry 24 tasks, materials science 16, biology 13, biomedical engineering 12, physics 11, and a long tail down to one task each for statistics, geography, nuclear science).

The design trick that makes the benchmark say something new is **two separate test sets**:

- **Public tests** live in the agent's workspace. The agent can run them, read them, debug against them.
- **Private tests** are mounted only *after* the agent submits its patch, in a separate container. They probe whether the *scientific* contract holds — different parameter scales, mirrored coordinates, reversed atom ordering, boundary regimes, equivalent representations of the same physical object.

The headline number is the gap between them. The best agent configuration (Claude Code driving Claude-Opus-5 at max reasoning) scores **96.64% on public tests but only 47.90% Pass@1** on private ones. Another configuration (Claude Code + DeepSeek-V4-Pro) scores a perfect **100% public** and **42.02% Pass@1**. Passing everything you can see tells you almost nothing about whether the repair is correct.

> [!NOTE] Scientific contract
> The invariants a piece of scientific code must respect regardless of input — energy conservation, unit consistency, symmetry under relabelling, correct limiting behaviour. The benchmark's private tests are written to check these, not to check program behaviour. A patch can satisfy behaviour and violate the contract. ^scientific-contract

Why this did not exist before: earlier scientific coding benchmarks test either self-contained function writing (SciCode: 80 tasks, 16 domains, no repos) or research-workflow execution (SUPER, ScienceAgentBench). The one closest predecessor, AInsteinBench, covers 6 domains across 6 repositories. SWE-bench Science trades depth per domain for **breadth**: 20 domains, 98 distinct repos.

What it unlocks: because every failure is hand-audited, you can ask *why* agents fail, not just *how often*. That produces a four-way taxonomy of failure that is the most portable part of the paper.

## The Methodology

### What the agent sees

Four fields, frozen before evaluation:

1. **Repository snapshot** — the exact state just before the real bug was fixed. Git history, remotes, future changelogs, build caches and anything linking to the solution are stripped out.
2. **Problem statement** — written before the test author and patch author saw each other's work, to stop the statement leaking the fix.
3. **Required scientific context** $c_i^{\mathrm{req}}$ — the minimum definitions and constraints needed for the task to be well-posed. Held fixed in *every* experimental condition.
4. **Public tests.**

Everything else is evaluator-only: private tests, the reference patch, alternative valid patches, localisation hints, difficulty labels, contamination tier. None of it is reachable from the workspace, environment variables, logs or metadata.

### The scale of the thing

| Quantity | Min | Mean | Max |
|---|---|---|---|
| Input code (non-empty lines) | 174 | 80,600 | 2,029,051 |
| Reference patch, lines added | 1 | 117.81 | 1,035 |
| Reference patch, lines deleted | 0 | 44.53 | 458 |

So this is not a one-line-fix benchmark. The median task requires reading a genuinely large codebase and writing roughly a hundred lines.

### Three task paradigms

The 119 tasks split into three kinds, each built differently on purpose.

**Issue-driven (52 tasks, 43.7%).** Start from a confirmed historical bug. Roll the code back to just before the fixing commit. Build a minimal reproducible example. Then *narrow the symptom* — tell the agent only something coarse like "results differ across two computational paths", never where the bug is. Private tests vary data scale, boundary parameters and input ordering, so a patch that fits the public script does not pass.

**Expert-exploratory (49 tasks, 41.2%).** These do *not* start from an issue. The authors pick a scientific scenario (molecular representation consistency, measurement-chain bias, medical-image coordinate alignment), find a discrepancy worth investigating, then build a runnable workflow that *shows* the anomaly while hiding where in the source it comes from. The agent must do controlled comparisons itself. Private tests change physical topologies, coordinate orderings and parameter scales to check whether the agent inferred the *mechanism* or just patched the instance.

**Engineering-integration (18 tasks, 15.1%).** A capability gap spanning the whole call chain — data loading, parameter interpretation, intermediate representation, operator assembly, numerical solve. The real package structure and neighbouring modules are preserved, so the agent must navigate across files. Private tests include alternative execution paths, state-reset tests and inter-module contract tests.

### Construction protocol

All raw tasks go through four stages the paper calls the **Chain-of-Evidence Protocol**:

1. Sample issues, PRs, commits and literature from candidate repos; discard trivial fixes, unstable dependency environments, heavy solution leakage, and projects overlapping existing samples.
2. Freeze the snapshot $\mathcal{S}_{\mathrm{bug}}$ and reproduce the anomaly in an isolated container — confirming the failure comes from algorithm semantics, not environment noise.
3. Abstract public materials: public repo, coarse phenomenon description, reproduction script, background docs. No patch location, no hidden assertions.
4. Build hidden validators from semantic equivalence and boundary conditions, *including reverse checks* for hard-coded answers, heuristic pseudo-fixes and incomplete repairs.

### Metrics

Per task–attempt:

- **PublicScore** / **PrivateScore** — mean score over applicable public / private test cases.
- **Fail2Pass** — fraction of previously-failing private tests that now pass. Progress on the repair.
- **Pass2Pass** — fraction of previously-passing private tests still passing. Regression check. Set to 1 if that set is empty.
- **Pass@1** — binary, and strict: 1 only if *every* applicable private test passes.

## Ablation Studies and Experiments

### Main results, 119 common tasks

| LLM | Harness | Public | Private | F2P | P2P | **Pass@1** | Issue | Expert | Eng. |
|---|---|---|---|---|---|---|---|---|---|
| GPT-5.6-sol (max) | Codex | 98.32 | **75.57** | **69.98** | 96.30 | 40.34 | 34.62 | 46.94 | 38.89 |
| Claude-Opus-5 (max) | Claude Code | 96.64 | 75.11 | 68.60 | 97.37 | **47.90** | **38.46** | **65.31** | 27.78 |
| DeepSeek-V4-Pro (max) | Claude Code | **100.00** | 73.16 | 65.77 | 96.58 | 42.02 | 26.92 | 57.14 | **44.44** |
| Kimi-K3 (max) | Kimi Code | 98.32 | 66.34 | 57.55 | 94.94 | 35.29 | 25.00 | 44.90 | 38.89 |
| GLM-5.2 (max) | Codex | 94.12 | 63.61 | 53.81 | **97.53** | 31.93 | 17.31 | 46.94 | 33.33 |
| Nex N2 | Codex | 93.28 | 61.89 | 51.09 | 94.92 | 24.37 | 11.54 | 36.73 | 27.78 |
| DeepSeek-V4-flash (max) | Claude Code | 98.32 | 61.41 | 52.34 | 95.74 | 23.53 | 19.23 | 26.53 | 27.78 |
| Qwen3.5-397B | Codex | 96.64 | 51.79 | 38.33 | 95.16 | 14.29 | 5.77 | 24.49 | 11.11 |

What this table actually says:

- **No model wins everything.** Best private score (GPT-5.6-sol), best Pass2Pass (GLM-5.2), best engineering-integration (DeepSeek-V4-Pro) and best overall Pass@1 (Claude-Opus-5) are four different configurations.
- **Pass2Pass is saturated** — 94.9% to 97.5% for everyone. Agents are not breaking what already worked. All the difficulty is in Fail2Pass.
- **Partial credit is much easier than complete credit.** GPT-5.6-sol repairs 69.98% of failing private tests but fully solves only 40.34% of tasks. Being 70% right about the physics is not being right.
- **Engineering-integration is the hardest category for the best model.** Claude-Opus-5 is top on Issue-driven (38.46%) and Expert-exploratory (65.31%) but drops to 27.78% on Engineering-integration. Cross-module reasoning is a different skill from local repair.
- Token spend does not explain the ordering. Claude-Opus-5 gets the top Pass@1 on a moderate budget; GPT-5.6-sol scores lower with far shorter outputs; DeepSeek-V4-Pro uses more input tokens for middling results. Among models under 400B parameters, Nex-N2 has the best score-per-token.

### The failure taxonomy

Every unsuccessful attempt was manually assigned to exactly one of four mechanisms.

> [!NOTE] The four failure mechanisms
> **Knowledge/abstraction deficit** — the repair is built on the wrong scientific object or mathematical definition.
> **Misguided exploration / surface repair** — the patch chases the visible symptom or the public metric, never tracing back to an independent oracle.
> **Incomplete coverage / system integration** — one module is fixed correctly but its interactions, data flow or shared invariants are not preserved.
> **Scientific-knowledge generalisation failure** — the observed case is handled; the same principle is not extended to equivalent representations, boundary regimes or other variants. ^four-failure-mechanisms

| LLM | Total errors | Knowledge | Exploration | Integration | Generalisation |
|---|---|---|---|---|---|
| GPT-5.6-sol | 71 | 18 | 13 | 21 | 19 |
| Claude-Opus-5 | **58** (+4 runtime) | 24 | **2** | 21 | 11 |
| DeepSeek-V4-Pro | 69 | **15** | 12 | **19** | 23 |
| Kimi-K3 | 77 | 20 | 14 | 22 | 21 |
| GLM-5.2 | 81 | 20 | 14 | 24 | 23 |
| Nex N2 | 90 | 31 | 14 | 23 | 22 |
| DeepSeek-V4-flash | 91 | 23 | 14 | 48 | **6** |
| Qwen3.5-397B | 102 | 26 | 15 | 33 | 28 |

The interesting column is **Exploration**: Claude-Opus-5 has 2, everybody else has 12–15. Whatever the Claude Code harness is doing — more disciplined investigation before editing — it nearly eliminates surface-level patching. Yet Claude-Opus-5 has the *most* knowledge/abstraction errors (24). Disciplined search does not supply missing physics.

DeepSeek-V4-flash shows the mirror image: only 6 generalisation failures but 48 integration failures — it understands the principle and fails to wire it through the system.

### The scientific-knowledge ablation — the result that does not go the expected way

91 of the 119 tasks allow the explicit scientific guidance to be peeled off while leaving everything else identical: same snapshot, same environment, same reproduction entry points, same private validators, same $c_i^{\mathrm{req}}$.

What gets removed: scientific rationales, equations and assumptions, expected properties, domain diagnoses, scientifically motivated repair strategies, upstream repairs, paper excerpts, expert guidance. What stays: executable context, task objective, observable symptoms, input data, minimal interface docs, and all repository-intrinsic cues (code structure, interfaces, traces, tests, half-finished implementations) — because stripping those would change the software task itself.

| Config | Public | Private | **Pass@1** | Input tok | Output tok |
|---|---|---|---|---|---|
| GPT-5.6-sol, **with** sci info | 97.80 | 74.06 | **31.87** | 3.70M | 40.25k |
| GPT-5.6-sol, **without** | 96.70 | 73.23 | **36.26** | 3.86M | 43.75k |
| DeepSeek-V4-flash, **with** | 100.00 | 62.53 | **23.08** | 7.40M | 159.26k |
| DeepSeek-V4-flash, **without** | 98.90 | 61.21 | **16.48** | 4.84M | 128.43k |

Giving the stronger model the science **hurt it**: Pass@1 fell from 36.26% to 31.87%, even though public and private *mean* scores went slightly up and tokens went slightly down. Giving the weaker model the science **helped**: 16.48% → 23.08%, but at 53% more input tokens.

The task-level overlap explains how:

- GPT-5.6-sol: 21 tasks pass either way, **8 pass only with** the guidance, **12 pass only without** it.
- DeepSeek-V4-flash: 12 both ways, **9 only with**, **3 only without**.

So the information is not noise — it genuinely unlocks 8–9 tasks in each case by supplying constraints the code symptoms cannot express (limiting cases, coordinate consistency, independent observables, which interface is authoritative). But for the stronger model it *loses* 12 tasks to **anchoring**: the agent accepts the supplied explanation, stops scoping independently, and skips the validation it would otherwise have run. See [[Reward Hacking]] for the same shape — an optimiser latching onto the stated proxy instead of the goal.

The practical reading: scientific knowledge helps in proportion to how badly the agent needs it, and hurts in proportion to how willing the agent is to substitute it for execution.

## Worth Remembering

**The public/private gap is the finding.** 96.64% public versus 47.90% private, from the same run. If you build an agent evaluation where the agent can see the grader, you are measuring something close to nothing. This is the same lesson as [[Evals]]`#^judge-rules` and [[Agent Evaluation]]`#^grade-the-route-too`, now with a hard number attached.

**Pass2Pass being near-saturated is quietly informative.** Frontier agents have stopped causing obvious regressions. The remaining problem is entirely "did you understand the thing", not "did you break something adjacent".

**Partial credit is dangerous in science specifically.** 70% Fail2Pass means most of the broken assertions now pass. In a simulation, a patch that fixes most invariants and violates one is arguably *worse* than no patch, because it looks fixed.

**Appendix B is a resource in itself.** All 119 tasks are listed with upstream repo, issue/PR number, knowledge scope and a one-line statement of the actual scientific contract — e.g. task 001: rotational symmetry of a transition-state graph must be set by real linear-segment topology, so relabelling atoms or reversing the reaction cannot change the observation. That is 119 worked examples of what "scientific contract" means concretely.

**Limitations the authors state:** domain counts are thin (six domains have one task each), so cross-domain comparisons are unreliable. And the knowledge analysis is preliminary — they measure *whether* guidance helps, not *how* an agent consumes it.

**Caveats they do not state:**

- Model names in this paper (GPT-5.6-sol, Claude-Opus-5, Kimi-K3, DeepSeek-V4-Pro, GLM-5.2, Nex N2, Qwen3.5-397B) do not correspond to anything I can verify as released. Treat the specific rankings as internal to this paper, not as a public leaderboard claim. The *structure* of the result — the public/private gap, the failure taxonomy, the anchoring effect — is the transferable part.
- The ablation is two models, one run each, with no significance test. The authors say so explicitly: "paired descriptive differences... do not establish statistical significance or a causal effect." Nine tasks flipping one way and three the other is well inside noise for $n=91$.
- Only one harness per model in the main table, so model and harness are confounded throughout. Claude-Opus-5's near-zero exploration errors may be Claude Code, not Claude. [[An Empirical Study of Harness Design for Coding Agents]] is the right companion read for untangling that.
- Contamination tier is recorded as evaluator-only metadata but never reported in the results. Most of these are real historical fixes in public repos, so some leakage into pretraining is near-certain. [[Schrödinger's Code Repository- Have LLMs Learned SWE-bench or Memorized It]] is the method you would want applied here.

**Open question worth chasing:** the ablation shows that *supplying* knowledge is not the same as *grounding* it. The obvious next experiment is to require the agent to derive a check from the supplied principle and run it, rather than letting the principle be read and believed. That turns scientific context from a claim into an oracle.

## Links

Related: [[Evals]] · [[Agent Evaluation]] · [[An Empirical Study of Harness Design for Coding Agents]] · [[SWE Refactor Bench- Can Coding Agents Complete a Long-Horizon, Whole-Repository Stack Migration]] · [[Schrödinger's Code Repository- Have LLMs Learned SWE-bench or Memorized It]] · [[ScienceIDE- Turning World's Scientific Codebase into Agent Learnable Environments]] · [[Reward Hacking]] · [[Agentic Workflows]] · [[Planning and Decomposition]] · [[Sandboxing]] · [[Observability and Tracing]] · [[On the Difficulty of Evaluating Baselines]] · [[Troubling Trends in Machine Learning Scholarship]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]]

New topics worth writing: held-out test design for agent benchmarks, anchoring bias in tool-using agents, scientific invariants as test oracles, SWE-bench task construction protocols, benchmark contamination tiers, partial-credit metrics for repair tasks (Fail2Pass / Pass2Pass)
