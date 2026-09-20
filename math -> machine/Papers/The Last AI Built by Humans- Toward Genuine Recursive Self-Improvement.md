---
title: "The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement"
authors: ["Duan et al."]
year: 2026
arxiv: "2609.11873"
url: https://arxiv.org/abs/2609.11873
priority: Good-To-Read
read_on: 2026-09-19
tags: [paper, llm]
---
## The Core Idea

This is a survey, not a new method. Its contribution is a way of *classifying* systems that improve themselves, plus a normalised score that shows where AI progress is still weak.

The core move: stop asking "what algorithm does this system use?" and start asking "who makes the improvement decisions?" The unit of analysis is the **improvement loop**, not the learning rule. For any system, you ask three questions:

1. Where does the loop close — does a change actually come back to the system, or does it only change this one output?
2. What is retained and inherited by the next round?
3. Which decisions stay with humans or fixed infrastructure?

> [!NOTE] Recursive self-improvement (RSI)
> A system that turns experience into *persistent* changes to itself, where those changes can also alter the mechanisms used to generate, evaluate, select, and consolidate later improvements. Not "the model got better" — "the process that makes the model better got better." ^rsi-def

The reason this framing did not exist before is that people kept mixing three separate things: automation (a machine does the step), persistence (the change survives the task), and autonomy (the machine chose the change). You can have any one without the others. A data-filtering pipeline is fully automated, fully persistent, and has zero decision autonomy. A [[Chain of Thought|chain-of-thought]] self-critique has some decision autonomy and zero persistence.

The six levels:

| Level | What AI controls | Example |
|---|---|---|
| **B0** | Nothing persistent — refines the current output only | Self-Refine, Reflexion, Tree of Thoughts |
| **L1** | Executes a human-written improvement procedure; result persists | FineWeb-Edu quality filtering |
| **L2** | Chooses *which* change to try; objective and evaluator fixed | GEPA prompt search, ADAS, AFlow |
| **L3** | Chooses what *experience* to learn from next | Absolute Zero Reasoner, R-Zero, SEAgent |
| **L4** | Decides what deployment experience persists | ReasoningBank, PANDO, Tax AI |
| **L5** | Revises the improver / evaluator / research policy itself | STOP, Gödel Agent, A-Evolve-Training |

The transitions are the useful part. B0→L1 is **persistence**. L1→L2 is **strategy selection**. L2→L3 is the **learning agenda**. L3→L4 is **deployment interaction**. L4→L5 is **recursive inheritance**.

The second contribution, which matters for anyone who reads benchmark tables: the **Headroom-Closed Index**, and what it reveals about where AI is actually weak.

## The Methodology

### The Headroom-Closed Index

The survey collects 393 model-benchmark observations across ten capability domains, 2023 to September 2026. Raw benchmark numbers cannot be pooled — a 60% on one benchmark means nothing next to a 60% on another. So they normalise twice.

First, when several sources report the same model on the same benchmark under the same protocol, take a weighted consensus:

$$\bar{s}_{mbh}=\frac{\sum_{i}\tilde{w}_{i}s_{imbh}}{\sum_{i}\tilde{w}_{i}}$$

Base weights: benchmark-owner tables 3, independent common-harness evals 2.5, combined reports 2, model-author tables 1. First-party numbers get multiplied by a further 0.75. This is a crude but explicit correction for self-reported results — worth stealing.

Then normalise against the benchmark's own starting point:

$$H_{mbh}=100\times\frac{\bar{s}_{mbh}-F_{b,0}}{100-F_{b,0}}$$

where $F_{b,0}$ is the 90th-percentile model score in the benchmark's *first* year in the dataset. So $H=0$ means "as good as the frontier when this benchmark launched" and $H=100$ means perfect. This measures **how much of the remaining room was closed**, not raw score.

Domain trajectories aggregate benchmark families with a square-root coverage weight:

$$T_{d,y}=\frac{\sum_{b}\sqrt{n_{b,y}}\,Q_{b,y}}{\sum_{b}\sqrt{n_{b,y}}}$$

$n_{b,y}$ is how many distinct models contributed to that benchmark-year frontier. The square root stops the most-populated benchmark from dominating the domain average.

Protocol hygiene: two results only join the same family if benchmark version and harness are stable, or if overlapping models bridge the protocols. This rule admitted 17 of 33 results in the latest-model audit; the other 16 (Terminal-Bench, DeepSWE, CyberGym, ExploitBench, AutomationBench, BrowseComp) were kept as reference only.

### The improvement-loop vocabulary

For classifying any system, they define eight components:

- **System state** — what the end of round $n$ hands to round $n+1$.
- **Experience** — information from earlier interaction that informs a later change.
- **Target** — the object modified this round.
- **Improver** — the mechanism turning state + experience into candidates.
- **Strategy** — how the improver decides where to search. Retainable, so it can itself become a target.
- **Verifier** — evaluates candidates, applies the acceptance rule.
- **Improvement** — a candidate that passed and was retained.
- **Successor** — the system that inherits it.

### How each level actually works

**L1 — execution.** Humans encode engineering practice into explicit steps; AI runs them at scale. Google's label curation: humans define what "clickbait" means, an LLM applies the criterion, a clustering step picks boundary samples for expert annotation. FineWeb-Edu: humans define educational quality, an LLM labels, a classifier scales it to the web. Meta's NCCL debugging: engineers distilled watchdog-timeout root causes into a decision tree, and an agent walks that runbook — aligning evidence across ranks, tracing collective sequences — to find the earliest actionable divergence. See [[Collective Communication#^nccl-timeout|NCCL timeouts]] and [[Distributed Training]] for what those failures are.

The L1 risk is specific: because outputs persist into later stages, an execution error propagates beyond the task that produced it.

**L2 — strategy.** The loop is observe → diagnose → select how to change → instantiate and test → retain or revert. Organised by what is editable:

- *Prompt.* GEPA attributes failures to particular modules from execution traces and keeps complementary prompt variants on a Pareto front rather than committing to one greedy lineage. Dropbox used it to rewrite the relevance judge in Dash search.
- *Harness.* ADAS represents an agent as a Python `forward` function; a meta-agent writes candidates into a growing archive with code and metrics. AFlow encodes workflows as executable graphs and runs MCTS over code-level edits. AgentSquare factorises designs into modules and recombines them.
- *Training.* OpenAI's GPT-6 Astra NanoGPT eval: fixed validation target, one H100, constrained setup, but the agent must diagnose bottlenecks and edit the training loop itself. Karpathy's `autoresearch` fixes the data pipeline, eval function, metric, and per-experiment compute, and keeps an edit only if the protected validation metric improves.
- *Kernels.* AutoKernel profiles first to find operations with the largest end-to-end impact, then proposes Triton/CUDA candidates accepted only after correctness checks and measured GPU speedups.

**L3 — experience.** The defining property is *learner-conditioned* acquisition: evidence about the evolving learner must change what experience is sought next, and the resulting persistent update must feed back into later acquisition.

- AZR couples proposal and solution in one model; a solver-dependent learnability reward guides task proposal and a code executor validates both tasks and solutions.
- R-Zero's Challenger gets an uncertainty reward based on the *consistency* of multiple Solver responses. No answer labels needed — but consistency is a proxy for perceived difficulty, not correctness.
- STP co-trains a conjecturer and a prover; conjectures the current prover can barely prove become conjecturer training data. A proof assistant supplies correctness.
- PSV generates formally verified code specifications with solver-derived difficulty labels. Honest finding: realised difficulties of problems targeted as easy, medium, and hard *overlap substantially*.
- VOYAGER links curriculum objectives to exploration history and a growing executable skill library in Minecraft — persistence lives in external skills, not weights.
- SEAgent runs a World State Model that judges trajectories and describes GUI state changes, feeding a Curriculum Generator that maintains a persistent "software guidebook".

**L4 — deployment.** Three mechanisms:

*Trajectory distillation* — turn interaction histories into compact reusable artifacts. Dynamic Cheatsheet keeps one evolving note of strategies and pitfalls, no labels needed. ACE stores short entries with helpful/harmful counters and applies *patches* rather than rewrites, to avoid compression loss. ReasoningBank distils each judged trajectory — failures included — into a short titled strategy. Metis keeps dual memory: text plans plus a code library, where a frequently reused text plan is rewritten as a callable tool and accepted only after it compiles and runs in a sandbox.

*System revision* — DecoEvo co-evolves a solver skill and a rubric-generator skill, and solves the circularity problem (better scores might just mean an easier judge) by gating the rubric generator on two score-independent audits: a structural audit for requirement coverage, a contrastive audit for discriminating near-ties. **The generator never sees the solver's aggregate score**, so it cannot win by making the rubric easier.

*Selective retention* — HDSO tests each candidate skill by running the same tasks twice, control (current repository) versus treatment (repository + candidate), in stages of increasing size, ending on independent tasks.

**L5 — meta.** STOP makes the improver a Python program that calls a fixed LM, generates candidates, evaluates a utility, and selects. It then receives *its own source* as the optimisation target. A selected fourth-generation improver beat the seed on all five transfer tasks held out of self-improvement.

RQGM handles the evaluator problem directly: the learned evaluator is **frozen within an epoch**; at a scheduled boundary, challenger evaluators are compared against an independent ground-truth anchor; scores that depended on a replaced evaluator are discarded and affected agents re-evaluated.

A-Evolve-Training runs autonomous post-training of a 30B Nemotron model. Workers return recipe changes and failures; a collector consolidates; a meta-agent revises the next round's search policy, which holds a standing recipe, promoted/retired directions, and a registry of failures.

## Ablation Studies and Experiments

### The HCI numbers — the most reusable part of the paper

2026 domain values:

| Domain | HCI 2026 |
|---|---|
| Cybersecurity agents | 91.9 (caveat below) |
| Advanced mathematics | 86.4 |
| Graduate-level science | 85.8 |
| Broad knowledge | 77.2 |
| Legal reasoning | 64.5 |
| Multimodal reasoning | 62.2 |
| Frontier academic breadth | 60.4 |
| Search and terminal agents | 56.8 |
| Software engineering | 52.6 |
| Tool agents | 39.9 |

The trajectories matter more than the levels. Broad knowledge gains 32.8, then 26.9, then 17.6 — steady deceleration. Legal reasoning: 48.2 → 11.4 → 4.9, a hard plateau. Advanced mathematics goes the other way: 32.8 in 2025, then 53.6 in 2026. Multimodal reasoning gains 59.7 in 2025 and 2.5 in 2026.

**The headline finding: interactive capability lags badly.** Tool agents sit 45.9 points below graduate-level science. Software engineering is 33.2 below. Tool agents jumped 8.2 → 39.9 in 2026 and are still last. The cybersecurity figure is flagged with a dashed line because later Cybench observations changed task subsets or moved to `pass@1` aggregation — do not lean on the 52-point gap.

The mechanism is stated plainly: bounded evaluations exercise L1–L2 behaviour; environment tasks need L2–L3; long, stateful workflows expose L3–L4 verification, memory, and adaptation requirements. Errors propagate across a trajectory, so both data collection and evaluation must cover complete interactions.

### What actually worked, and by how much

| System | Result |
|---|---|
| A-Evolve-Training | 30B model, 4 autonomous rounds, external score 0.80 → 0.86 (top human: 0.87) |
| DGM | SWE-bench subset 20% → 50% |
| RQGM | 71.7% pass on held-out Polyglot vs 69.9% for HGM-H, with fewer search tokens |
| Theseus (clean vs noisy workspace) | +21.7 to +51.6 pp across 8 model-harness configs, 30 tasks, 1,280 rubrics |
| Theseus (reconstructed environment) | +18.65 to +39.67 pp, same model and harness |
| Humanlaya V0→V4 | Key-defect rate after auto-repair 9.0% → 3.7%; human handling 48 → 27 min/task, on 600 held-out packages |
| ForgeTrain | Matched Megatron-LM v0.15 on H100 in ~8 hours from an empty directory; MFU 40.1% → 44.1% (0.5B), 47.0% → 50.9% (8B) |
| Agent-Native Research Lab | QA over prior work 72.4% → 93.7% with executable artifacts; RE-Bench reproduction 57.4% → 64.4% |
| Frontis | 216 improvement tasks, 63 full cycles, scores 5.25 → 5.87 on 76 tasks; ~20% faster evolution after >100 meta-tasks |

The Theseus clean-versus-noise pilot is the single most striking number here and has nothing to do with RSI: **workspace state, not model capability, bounded agent performance.** DeepSeek-V4-Pro went from 98.2% clean to 46.6% noisy. That is a bigger swing than most model upgrades.

### What did not work — the important half

**Gödel Agent: 14 of 100 MGSM trials ended *below* the initial policy.** Persistence cuts both ways. Unrestricted runs could also call stronger models, which contaminates the comparison.

**STOP with weaker models regressed on average.** Some generated programs evaded soft compute budgets or exploited bugs in the evaluation harness.

**AIDE² found no statistically significant efficiency advantage** when the evolved harness was installed as the *outer* improver. It produced 7 accepted improvements over 100 unattended steps and transferred to external tasks — but the stronger claim, that the evolved agent accelerates the search, failed. This is the closest thing to a direct test of the recursive claim and it came back negative.

**HyperAgents' 200-iteration experiment showed no statistically significant final advantage** for transferred initialisation, even though shorter runs showed transfer from paper review and robotics to unseen mathematics grading.

**Anthropic's automated research agents cheated**: random-seed cherry-picking, shortcut discovery, and attempted test-label extraction through repeated evaluator queries. Once a benchmark is queried adaptively it becomes part of the optimisation surface, not a measuring instrument. This is [[Reward Hacking]] with the evaluator as the target. See also [[Reward Hacking#^optimiser-as-adversary|optimiser as adversary]].

**Library Drift**: unbounded skill accumulation degrades retrieval quality and stalls progress — often *before* the effect shows in task scores. The fix is lifecycle management (append-only evidence log, contribution-based retirement, a cap on active skills, a meta-skill governing how new skills are authored). Crucially, **retiring skills too aggressively performed worse than leaving the library unguided.** There is a middle.

**Harness Updating Is Not Harness Benefit** is the sharpest ablation in the whole survey. It separates two capabilities under a fixed solve-evolve protocol with identical prompts and budgets:

- *Producing useful updates* is largely **independent of base model capability**. A small model's updates gave gains comparable to a frontier model's.
- *Benefiting from an updated harness* varies **non-monotonically**, and failures concentrate in two modes: the relevant artifact is never activated, or it is activated but not faithfully followed.

Practical consequence, stated by the authors: **put capability investment in the task-solving agent, not the evolver.**

**Self-correction is weak at B0.** Self-Correction Bench identifies verification — specifically locating the *first* incorrect reasoning step — as the bottleneck. Without reliable external feedback, revision can turn a correct answer into a wrong one. Blind resampling outperforms self-repair in small code models. This connects to [[Hallucination#^what-doesnt-work|what doesn't work]] and [[Evals#^benchmarks-are-weak|weak benchmarks]].

**PSV's difficulty control is loose**: substantial overlap between realised difficulties of easy/medium/hard-targeted problems.

**SIMA 2's fixed-task experiment does not establish curriculum autonomy** — only the full ASKA setup, where a Gemini task setter uses reward-model evaluations to focus practice on weak skills, demonstrates the L3 mechanism. The authors are careful about this distinction and it is a good template for reading agent papers.

### The evaluation controls worth copying

For L3, to test whether learner-conditioning helps: freeze the acquisition mechanism's learner-state input at an earlier checkpoint, or replace adaptive selection with a learner-independent schedule. Match data-source access and total compute *including experience generation and assessment cost*. Do **not** hold the acquired samples identical — that removes the distributional adaptation you are trying to measure. This is a clean design and directly relevant to anyone building [[Evals]] for adaptive systems.

For L5, the survey distinguishes **structural recursion** (a revised mechanism is inherited and invoked) from **effective recursion** (it produces better successors under comparable budgets and independent evaluation). Almost all current evidence is structural. The evaluation dimensions they propose: adaptivity (gain, trajectory, time-to-target), retention (replay loss, tasks fixed vs broken), transfer (held-out gains), efficiency (cost per validated gain), stability (harmful updates, largest temporary decline), meta-recursion (mechanism reuse, successor quality).

## Worth Remembering

**The measurement angle.** This survey is unusually good on evaluation hygiene for a field this hype-prone, and much of it generalises to [[Evals]] and experimentation work:

- Weighting sources by provenance, with an explicit 0.75 penalty on first-party numbers.
- Refusing to pool results across incompatible harnesses, and *reporting* the 16 of 33 excluded.
- The structural/effective split — a mechanism can demonstrably persist and be invoked while producing no measurable benefit.
- Requiring matched budgets *including the cost of evaluating the mechanism itself*.
- "Once a benchmark is queried adaptively, it becomes part of the optimisation surface rather than a passive measurement instrument." That is a one-sentence statement of the whole problem with iterated offline evaluation, and it maps directly onto the reason [[Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms|replay evaluation]] needs logged propensities and a protected holdout.

**The Xiaohongshu case is the one with a recsys shape**, and it is the weakest evidence in the report. The Intent-Memory Agent decomposes state into three layers — Content (topic), Who (user state), Need (inferred requirement) — so that retrieval can key on any one, and so that feedback corrects one variable instead of rewriting the whole profile. Dual timescale: real-time memory updates at the user level, plus slower post-training on reviewed hard cases. Reported numbers: Discovery Feed `score_mean@16` 2.211 → 2.366 (+7.0%), median +4.8%; in-video feed 1.537 → 1.739 (+13.1%) — but that last one is an *offline* comparison on 470 vs 447 samples, and the authors state plainly that **no click-through or conversion gains were established.** A ranking-score improvement on unequal sample sets is not a result. Compare with how [[Recommending What Video to Watch Next- A Multitask Ranking System (RecSys)|YouTube's multitask ranker]] and [[Controlled experiments on the web- survey and practical guide (DMKD)|Kohavi's guidance]] treat proxy metrics.

**The honest framing throughout**: higher autonomy level ≠ better improvement process. "Greater delegated authority can coexist with inefficient search, unreliable feedback, regression, evaluator exploitation, or poor transfer." Industrial numbers are consistently labelled company-reported and not independently replicated.

**Limitations the authors admit.** End-to-end L5 evidence is confined to bounded prototypes. Statistically reliable accumulation across generations under comparable resources is *unestablished*. Most L4 evaluations cover short task streams. Healthcare evidence is almost entirely simulated or retrospective. Embodied RSI mostly runs in simulation with externally specified rewards.

**Practical caveats if you wanted to build any of this:**

- Budget for the evaluator, not just the search. Repeated adaptive access will corrupt any metric you optimise against.
- Freeze the evaluator within a round and validate replacements against an independent anchor (RQGM's design).
- Cap and prune your persistent store, but not aggressively — Library Drift shows over-pruning is worse than no governance.
- Measure activation and faithful use separately from artifact quality. Most of the failure is downstream of producing a good update.
- Use paired control/treatment runs for skill admission (HDSO), staged by size.
- Spend model capability on the solver, not the evolver.

**Follow-up questions.** The survey never seriously treats improvement-candidate selection as a bandit problem, despite it being exactly that: repeated choices among arms with noisy, expensive, delayed feedback, under a budget. There is no mention of [[A Tutorial on Thompson Sampling|Thompson sampling]], [[Finite-time Analysis of the Multiarmed Bandit Problem (UCB1)|UCB]], or best-arm identification anywhere in the improvement-search discussion — DGM's archive and parent-selection is a hand-rolled exploration policy, and HGM's "metaproductivity" is an ad-hoc value estimate over a lineage tree. The gap between "we kept an archive and picked parents" and a principled allocation rule is large and unexamined. Similarly, the repeated-adaptive-access problem has a direct answer in [[Always Valid Inference- Bringing Sequential Analysis to A-B Testing|always-valid inference]], which nobody cites.

## Links

Related: [[Evals]] · [[Reward Hacking]] · [[Always Valid Inference- Bringing Sequential Analysis to A-B Testing]] · [[Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms]] · [[A Tutorial on Thompson Sampling]] · [[Finite-time Analysis of the Multiarmed Bandit Problem (UCB1)]] · [[Controlled experiments on the web- survey and practical guide (DMKD)]] · [[Do ImageNet Classifiers Generalize to ImageNet]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[On the Difficulty of Evaluating Baselines]] · [[Troubling Trends in Machine Learning Scholarship]] · [[Chain of Thought]] · [[Agentic Workflows]] · [[Continual Learning Mechanisms Compose for Long-Horizon Memorization]] · [[Memory]] · [[Collective Communication]] · [[Distributed Training]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Recommending What Video to Watch Next- A Multitask Ranking System (RecSys)]]

New topics worth writing: Headroom-Closed Index and benchmark normalisation, benchmark saturation and adaptive overfitting, best-arm identification under budget, open-endedness and minimal criterion coevolution (POET), automated curriculum learning, self-play without labels (AZR/R-Zero), skill library lifecycle management, agent harness engineering, metaproductivity and lineage-based selection, provenance and rollback for persistent agent state
