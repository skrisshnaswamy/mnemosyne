---
title: "Rufus-Air: An Open LLM Post-Training Recipe"
authors: ["Chang et al."]
year: 2026
arxiv: "2609.29421"
url: https://arxiv.org/abs/2609.29421
priority: Good-To-Read
read_on: 2026-09-28
tags: [paper, llm, rl, vision, theory]
---
## The Core Idea

Most "how we post-trained our model" sections read like a system card: a list of stage names and a leaderboard. This report is the opposite — a full, stage-by-stage account of turning a public base checkpoint into a competitive chat/agent model, with the data, the reward code, the hyperparameters, the infrastructure, and the things that failed.

The model is **GLM-4.5-Air-Base**, a [[Mixture of Experts|mixture-of-experts]] model with 106B total and 12B active parameters. The pipeline is eight stages, run strictly in series, each starting from the checkpoint the previous one produced:

$$\text{SFT} \to \text{Reasoning RL} \to \text{Coding RL} \to \text{IF RL} \to \text{General Agent} \to \text{Coding Agent} \to \text{Search Agent} \to \text{RLHF}$$

The one genuinely transferable idea is the **ordering rule**. Stages are not sorted by capability, and not quite by "how hard is the reward to compute". They are sorted by **how easy the reward is to cheat**.

> [!NOTE] Reward-hackability ordering
> Run stages whose reward is a deterministic verifier first. Run stages whose reward is a learned model or an LLM judge last. Every stage updates the same weights, so a gameable reward that runs early is under optimisation pressure for the whole rest of the pipeline. Running it last bounds that exposure. ^reward-hackability-ordering

The test is not the *format* of the reward. Instruction-Following RL uses a rubric-based LLM judge — softer than a unit test — but it runs fourth, before the agent stages whose rewards are execution tests. Why? Because instruction following is close to what the policy already does, so the judge has very little slack to be exploited. [[RLHF]], where a learned [[Preference Learning|Bradley–Terry]] reward scores open-ended writing, is where [[Reward Hacking|hacking]] is genuinely dangerous, so it goes last.

The second reusable idea: **every RL stage filters prompts by learnability**, i.e. drops anything the current policy always gets right (no gradient under a group-mean baseline) and usually anything it never gets right (no signal at all). Because the policy improves but the filter window stays fixed, dead prompts keep drifting into the productive band. You get a curriculum for free without writing one.

Everything else is engineering that the authors argue *is* the recipe, not an implementation detail — the multi-turn rollout must return exact token IDs, the sandbox must survive a 100-tool-call episode, the MoE router must be replayed.

The scale is deliberately modest: RL stages ran on 8–32 nodes of 8×H200. That is reachable by a lab, not just a frontier company.

## The Methodology

### Stage 0 — SFT, treated as capability-building, not warm-up

9.01M samples, 44.5B raw tokens, 66.7M turns. After masking system, user, and tool-observation turns, **27.0B assistant tokens** carry loss. All data is from 17 public datasets, used as released. No new human annotation, no in-house teacher regenerating responses.

The sample/token split is worth internalising:

| Category | % samples | % train tokens |
|---|---|---|
| General Agent | 39.6 | 14.2 |
| General Chat | 17.7 | 12.2 |
| STEM | 15.1 | 8.8 |
| Math | 12.7 | **29.4** |
| Code | 8.8 | 15.7 |
| Coding Agent | 6.1 | **19.7** |

Math and Coding Agent are 18.8% of samples but 49.1% of supervision. If you budget your mix by row count you are lying to yourself.

Training: 3 epochs, 512 GPUs, batch 4096 sequences, [[Decoupled Weight Decay Regularization (AdamW)|AdamW]] ($\beta_1{=}0.9$, $\beta_2{=}0.95$), weight decay 0.1, clip 1.0, 100-step linear warmup to $5\times10^{-5}$ then cosine to $5\times10^{-6}$, 128K context after packing. ~13 days, ~832 node-days — the single largest cost in the whole recipe. The peak LR came from a small-batch sweep: $3\mathrm{e}{-4}$ and $5\mathrm{e}{-4}$ were unstable, $\le 1\mathrm{e}{-4}$ gave monotone loss.

The training-dynamics finding here is the one I'd carry away:

> Loss falls $0.84 \to 0.42$ (end of epoch 1) $\to 0.38 \to 0.34$, stepping down at each epoch boundary. Held-out AIME/GPQA/IFEval scores **flatten inside epoch 1** and afterwards move only within noise.

Lower loss on repeated data buys nothing. They ship checkpoint **3799**, taken inside the plateau, not the last step.

Decontamination is unusually thorough: sampled word-level 8-gram matching against ten benchmarks (drop a sample if ≥50% of an item's sampled phrases hit), plus for math/knowledge an *exhaustive* 8-gram pass and a dense-retrieval pass with manual review of high-cosine hits. Net: 4,187 samples removed, 3,321 of them from one synthetic terminal corpus overlapping Terminal-Bench. They kept 11,874 SWE-bench "matches" that traced to generic `pytest` scaffolding — false positives, judged benign.

The SFT-only checkpoint already beats the *vendor's fully post-trained* GLM-4.5-Air on IFEval (88.3 vs 83.0), IFBench (57.8 vs 33.6), AIME 25 (90.8 vs 84.2) and AIME 26 (90.0 vs 86.5), trailing only on GPQA (68.2 vs 73.9). That gap is exactly what the RL stages then go after.

### Stage 1 — Reasoning RL

121,161 verifiable single-turn prompts: 47.7% math (Math-Verify on canonicalised `\boxed{}` answers), 35.7% science (fuzzy string match), 16.7% puzzles from Enigmata and ReasoningGym (generated Python checkers). Puzzles are only 17% of prompts but 55% of prompt tokens — serialised grids are long.

Two offline filters gate entry:
- **Correctness**: drop any prompt where a strong teacher (GPT-OSS-120B) never gets positive reward. Evidence the prompt is solvable at all.
- **Learnability**: drop prompts solved >0.8 of the time (too easy) and prompts never solved (currently unlearnable).

Objective is **GSPO**, a [[Policy Gradient|policy-gradient]] method that defines the importance ratio and clipping at the *sequence* level rather than the token level — chosen because it is more stable on long MoE rollouts. Clip range is very tight: $\varepsilon = 10^{-3}$, $\varepsilon_{\text{high}} = 2\times10^{-3}$. [[KL Divergence|KL]] and entropy coefficients are **zero**. AdamW, constant $\text{lr}=10^{-6}$. Each rollout: 256 prompts × 16 samples at temperature 1.0, 30K-token budget, feeding 2 optimiser steps (global batch 2,048).

Two practical controls:
1. **Length penalty relative to the shortest correct rollout in the group** — zero at or below that baseline, growing linearly to a cap. Partly a preference against [[Chain of Thought|CoT]] bloat, partly economics: verbosity cuts throughput and raises truncation.
2. **Online dynamic filtering** (à la DAPO): over-sample $4\times$ to 1,024 prompts, keep only groups with mean reward $\bar r \in (0, 0.8]$. This is the automatic curriculum.

Truncated samples are masked from the loss; truncation grows from ~3% to 15–20% as responses lengthen.

### Stage 2 — Coding RL

29,405 Python problems (50% EvolveCoder synthetic with adversarially evolved tests, 32% Nemotron contest problems, rest Dolci/ADR). Two verification modes: `assert` unit tests for named functions, stdin/stdout comparison otherwise.

Reward: extract the last fenced ```python block, execute in a sandbox, **binary** — 1 if all tests pass. Tests are sub-sampled to at most 50 with a **SHA-256-seeded deterministic choice**, so the same problem gets the same test subset across every rollout and the reward does not drift with sampler randomness. No format penalty needed; fenced-block compliance was already 98–100% at step 0.

Difficulty filter is "drop-all-pass": 4 warm-up samples, remove anything solved 4/4. Unlike Reasoning RL they *keep* never-solved problems, and they do not re-filter during training.

GSPO again, 128 prompts × 64 samples (8,192 sequences) → 8 optimiser steps, 256 GPUs.

The stage's real lesson is a budget story. At a 64K response cap, 11–28% of samples per rollout truncated and were loss-masked, so a large slice of each batch contributed no gradient. At step 23 they raised the cap to 128K; truncation fell below 0.1% and both reward (0.34 → 0.42) and LiveCodeBench (74.5 → 75.9) resumed climbing.

### Stage 3 — Instruction-Following RL

Two synthetic datasets, trained jointly:

- **Multi-constraint, 14K single-turn.** Hand-written single-constraint seeds, augmented by Qwen3-235B-A22B into instructions with 2–6 atomic constraints, each paired with a generated Python checker. Example: "exactly three paragraphs, each under 40 words, the word 'therefore' exactly once."
- **Multi-turn, 13K conversations** (avg 6.1 turns, 2.7 rubrics). Generated *adversarially*: a teacher LLM plays the user and actively tries to break the assistant — layering instructions, revising earlier requirements, planting distractors. Several open models rotate as the assistant so the data isn't one model's style. Some carry a system prompt and train instruction hierarchy (obey the system prompt even when the user pushes back).

Reward: **all-or-nothing rubric**. Each instance carries a list of rubrics; code-verifiable ones are Python functions, the rest go to an LLM judge. Positive reward only if *every* rubric passes. [[GRPO]], 256 prompts × 16, 16K budget, $\text{lr}=1.5\times10^{-6}$, 90 steps.

The design rule they emphasise: **rubrics describe only what the response *must* satisfy**. No "optional" or "nice-to-have" rubrics.

### Stages 4–6 — The three agent stages

All three use [[GRPO]] with DAPO-style dynamic sampling (drop all-pass and all-fail groups) and **Rollout Routing Replay (R3)**, which caches which experts the rollout engine routed each token to and replays those decisions in the training forward pass. Without it, rollout and trainer can pick different experts for the same token even at identical precision, adding a spurious log-probability gap on top of any precision mismatch.

**General Agent.** AgentWorldModel: 10K single-server [[Model Context Protocol|MCP]] tasks over 1K synthetic environments, filtered to ~2K. Each environment is a scenario database plus an MCP server with ~35 tools that read and write it. Reward is the task's own code verifier applied to the *final server state* — binary, no step shaping, no judge. They deliberately dropped AWM's LLM-judge reward (the correctness filter handles environment imperfections instead) and its format reward (they register every tool schema in the system prompt, so there's no discovery step to enforce). 48 prompts × 64 samples, 32K budget, 16 nodes, 3,072 rollouts per step.

**Coding Agent.** ~4K tasks from Endless-Terminal, SETA-Env and Scale-SWE, all wrapped in the Harbor format (`instruction.md` + `Dockerfile` + `tests/test.sh`). The tool surface is **one tool**: `execute_command`, returning stdout, stderr, exit code. Reward is the task's test suite, binary. Rollouts that die for infrastructure reasons (sandbox boot failure, verifier timeout) are tagged **`aborted`** and removed from their group *before* advantages are computed, so infra noise doesn't poison the group baseline. That detail is easy to get wrong and silently biases every advantage in the group.

**Search Agent.** Three tools: web search, page scrape (with a summariser for oversized pages), and a network-less Python interpreter. Up to 100 tool calls in training, 200 at eval, 128K trajectory budget.

Data filtering here is the most interesting in the paper. Starting from 36,614 MiroVerse QA pairs:
1. **Drop anything answerable with tools disabled** (8 rollouts). These reward parametric recall, not research. Removes ~25%.
2. **Rollout 8× with tools, judge each, keep only $0 < c < 4$.** 15.5% were never solved, 31.4% always solved — at both extremes the group-relative advantage is exactly zero. Final set: ~2.2K questions.

Reward is a Qwen3-32B judge on $[0,1]$ with partial credit, plus guards: positive reward requires ≥2 successful tool calls, malformed calls scale reward down to $0.5\times$, no final answer costs $-0.1$. An earlier unconditional bonus for emitting a boxed answer was **removed after the policy learned to collect it with tool-free guesses** — a clean, small example of [[Reward Hacking]].

The advantage estimator matters because the judge is noisy. They use **mean-only** group advantages, $\widehat A_i = r_i - \text{mean}(\{r_j\})$, dropping the standard-deviation division. The argument is arithmetic: for $k$ successes out of $G$ separated from failures by gap $\Delta$, the std-normalised advantage of a success is

$$\sqrt{\frac{(G-k)(G-1)}{kG}}$$

which does not depend on $\Delta$ at all. At $G=16$, a lone success with a full $1.0$ gap and a lone success with a $0.1$ partial-credit gap **both get $+3.75$**, about $3.9\times$ the advantage in a balanced group. Since the learnability band deliberately keeps 1-in-8 prompts, such groups are everywhere, and normalisation amplifies grading noise to the scale of a real solve. They keep GRPO's per-sequence length normalisation though, because on 30–100K-token trajectories a constant normaliser lets the longest trajectory dominate the batch gradient.

They also keep a small KL anchor ($10^{-3}$) to the stage's init: without it entropy drifted 0.72 → 1.12; with it, 0.72 → 0.94 alongside rising validation.

### Stage 7 — RLHF

"RLHF" here means on-policy RL against an **open, off-the-shelf reward model** (Skywork-Reward-V2-Qwen3-8B), not against fresh human labels. On-policy rather than [[DPO]] so the signal acts on the policy's own rollouts. Objective: raw reward-model score under a linear length penalty that only bites when the raw reward is positive. GRPO, 768 prompts × 5 samples.

Prompts come from HH-RLHF only (75,815 after a validity filter). The chosen/rejected pairs are used *only* for offline dataset analysis; training uses the prompts and the reward model.

### Infrastructure worth copying

- **Token-in/token-out rollouts.** The SGLang backend takes prompt token IDs and returns generated token IDs plus per-token log-probs, loss masks and routed-expert assignments. The environment appends each turn's tokens as produced, never re-rendering the conversation to text and re-[[Tokenization|tokenising]]. This kills three silent bugs: retokenisation drift, repeated chat-template application rewriting earlier turns, and training on tool calls that were repaired outside the sampled trajectory.
- **FP8 rollout, BF16 training.** First three transformer layers stay BF16; the rest are FP8 for rollout. The parameter update is fully BF16. Truncated importance sampling absorbs the residual mismatch. See [[Mixed Precision training]].
- **[[Sandboxing|Sandboxes]] are Firecracker microVMs**, not containers — self-hosted E2B on bare-metal EC2. Every task image is baked into a snapshot ahead of time, so sandbox creation is a resume in seconds. Agent commands are **not idempotent**, so retries fire only on failures that prove execution never started.
- **Ray** distributes environment actors across nodes so CPU-heavy tool execution and reward computation don't contend with each other.

## Ablation Studies and Experiments

### Stage-by-stage deltas, each measured against the checkpoint it started from

| Stage | Benchmark | Before → After | $\Delta$ |
|---|---|---|---|
| Reasoning RL | GPQA | 68.2 → 73.5 | **+5.3** |
| Reasoning RL | AIME 25 / 26 | 90.8 → 88.0 / 90.0 → 87.4 | −2.8 / −2.6 |
| Coding RL | LiveCodeBench v6 pass@1 | 68.6 → 75.9 | **+7.3** |
| Coding RL | LiveCodeBench v6 pass@8 | 85.1 → 87.4 | +2.3 |
| IF RL | Multi-challenge | 31.1 → 55.8 | **+24.7** |
| IF RL | AdvancedIF | 45.5 → 62.9 | +17.4 |
| IF RL | IFBench | 63.8 → 77.8 | +14.0 |
| IF RL | IFEval | 90.5 → 94.5 | +4.0 |
| General Agent | Tau2-Retail | 74.0 → 83.8 | **+9.8** |
| General Agent | MCP-Atlas | 35.0 → 42.8 | +7.8 |
| Coding Agent | SWE-bench Verified | 65.6 → 67.8 | +2.2 |
| Coding Agent | Terminal-Bench 2.1 | 38.8 → 40.2 | +1.4 |
| Search Agent | Seal-0 | 48.6 → 54.0 | +5.4 |
| Search Agent | HLE-Verified | 47.7 → 51.1 | +3.4 |
| Search Agent | BrowseComp | 34.2 → 37.2 | +3.0 |
| RLHF | Arena-Hard v2 CW | 38.6 → 53.0 | **+14.4** |
| RLHF | Arena-Hard v2 HP | 83.1 → 89.1 | +6.0 |

Final Rufus-Air beats the vendor's GLM-4.5-Air release on **every** reported benchmark except Arena-Hard v2 Creative Writing (53.0 vs 60.3). The biggest gaps against same-base baselines are in instruction following (IFBench 76.9 vs 33.6), Tau2-Telecom (93.0 vs 32.7), and the search rows (HLE-Verified 51.1 vs 20.2).

Note the pattern: the recipe moves most where it spends the most training signal. On competition maths and GPQA — one stage's worth of effort — it sits near the pack.

### What did not work

**Optional rubrics cause length bias.** In IF RL, scoping rubrics to *necessary* conditions made the final response shrink from ~5.4K to ~3.6K tokens over training, because padding sometimes actively breaks a constraint. When they instead attached rubrics to optional content too (~10 rubrics per instance), the policy acquired a pronounced length bias, emitting long redundant answers trying to cover as many rubrics as possible. Same method, opposite behaviour, purely from rubric scope.

**Unconditional format bonuses get farmed.** Search Agent's early "boxed answer" bonus was collected by tool-free guesses. Removed.

**Std-normalised advantages lost to mean-only.** A paired ablation differing only in that flag: mean-only went 55.8 → 64.8 on held-out, normalised went 49.7 → 58.7, and the normalised run's entropy rose faster with no validation payoff.

**Two of three RLHF prompt sets were unusable.** Unfiltered HH-RLHF degraded coding and instruction following. HelpSteer3, despite the largest chosen/rejected score separation (9.60 vs 2.95), had prompts the reward model could not score reliably — reward rose slowly while science reasoning and instruction following regressed. Instruction-following and agentic prompts were outright unsuitable because the reward model can't assess complex constraint satisfaction. They settled on filtered HH-RLHF.

**Coding Agent gains did not survive.** SWE-bench Verified went 65.6 → 67.8 in the stage, but the shipped checkpoint reads 65.6 — right back to where the stage started. Terminal-Bench went the other way, 40.2 → 42.7. Neither was measured after Search Agent, so they honestly say they cannot attribute it.

**Search depth stops paying.** On BrowseComp, accuracy is 88% for trajectories finishing within 25 tool calls, 64% at 25–50, and **below 30% beyond 50**. Wrong answers average 92 iterations against 43 for correct ones. About 10% of trajectories end by exhausting the budget, essentially none of them correct. Depth past ~50 is a stuck-search signal, not progress.

But within the training run, the *shape* of long searches changed. Tool calls per episode fell then climbed. The fall was compression everywhere (winners −4 calls, losers −7 calls). The climb was a new tail of *productive* long searches: trajectories over 45 calls nearly tripled (49 → 133) and their win share rose 24% → 43%, while winners' text kept shrinking (42K → 33K tokens median).

### Measurement caveats that are arguably the most reusable part

**The user simulator is part of the benchmark.** Tau2-Bench scores move by up to **23.7 points** on the same model just by swapping the simulated customer (GLM-4.5-Air retail: 57.0 under Sonnet 4, 80.7 under Sonnet 5). Sonnet 5 gives the highest score in all twelve model–domain pairs. Rankings are more stable than levels, but the airline leader flips between simulators. If you are comparing Tau2 numbers across papers without matching simulators, you are comparing noise.

**[[Context Engineering|Context management]] is worth more than most method changes.** They compared 12 strategies across 4 families; an external-memory scratchpad (record findings after each research round, keep only the current step in context) won. Disabling it drops BrowseComp by **10.7 points** on the same checkpoint — larger than any other single choice in the stage. They also flag that frontier labs report BrowseComp with "discard-all" context handling, which restarts from the original query whenever context fills — effectively repeated independent attempts, which they classify as [[Test-Time Compute|test-time compute scaling]] rather than trained search competence.

**Baseline numbers are not comparable.** Appendix E documents the provenance of every externally-sourced cell. Qwen3.5-122B's AIME 25 is reported as 92.92, 90.36, and 89.48 by three different sources. GPT-OSS-120B's Multi-challenge is 58.29 in NVIDIA's report and 45.34 on the benchmark owner's leaderboard.

## Worth Remembering

- **The SFT loss curve and the eval curve decouple after one epoch.** Loss kept stepping down at epoch boundaries (0.42 → 0.38 → 0.34); held-out scores flattened. They shipped a mid-plateau checkpoint. This is the cleanest small demonstration in the paper that training loss is not the thing you care about.

- **Three of the eight stages use binary, deterministic rewards with no shaping at all.** No step-level credit, no partial credit, no judge. The complexity lives in prompt selection and in the environment, not in the [[Reward Function|reward]]. Compare to [[DRACO- Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training|DRACO]], which goes the other way.

- **The std-normalisation argument generalises beyond this paper.** Any group-relative estimator that divides by the within-group standard deviation throws away the *magnitude* of the reward gap. If your reward is noisy or gives partial credit, and your prompt filter deliberately keeps rare-success groups, that division amplifies grading noise to the size of a genuine win. Worth remembering whenever you see [[GRPO]] applied to a judge-scored task.

- **Learnability filtering is off-policy-evaluation-adjacent.** Deciding which prompts to train on is a value-of-information question: a prompt with $\bar r \in \{0, 1\}$ has zero expected gradient, so sampling it is pure waste. The online variant (over-sample $4\times$, keep $\bar r \in (0, 0.8]$) is a cheap bandit-flavoured selector over a fixed prompt pool. See also [[Dynamic Important Example Mining for Reinforcement Finetuning]].

- **Agentic RL has a cost structure outside the GPU bill.** The sandbox service is sized for ~10,000 concurrent microVMs at ~$10K/month. One pass over BrowseComp's 1,266 questions at ~100 tool calls each makes on the order of $10^5$ external API calls, a few hundred dollars at list prices ($2/1K Serper searches, $0.05/1M Jina Reader tokens) — and *every training rollout is additionally scored by an LLM judge*. Budget these stages separately, not by per-stage average.

- **Limitations the authors admit.** The stage order is "one that worked", not one shown optimal — it was inherited from earlier experiments and constrained by compute. Coding Agent trained to its budget, not to convergence. No single benchmark suite was tracked across all eight stages, so between-stage movement often cannot be attributed. They did not try the obvious alternative of training domain experts in parallel and merging by multi-teacher on-policy [[Distillation|distillation]].

- **Practical caveat if you want to use this.** The recipe depends on the base model's native `<think>` chat format and on chat-template consistency held fixed from SFT all the way through the agent stages. Change the template mid-pipeline and your verifiers, judges, and tool-call parsers all start disagreeing with each other silently.

## Links

Related: [[GRPO]] · [[RLHF]] · [[PPO]] · [[Reward Hacking]] · [[Reward Function]] · [[Fine-Tuning]] · [[Instruction Tuning]] · [[Mixture of Experts]] · [[Mixed Precision training]] · [[Quantization]] · [[Tool Use]] · [[Model Context Protocol]] · [[Sandboxing]] · [[Agentic Workflows]] · [[Agent Evaluation]] · [[Evals]] · [[Context Engineering]] · [[Preference Learning]] · [[DPO]] · [[Policy Gradient]] · [[On-Policy vs Off-Policy]] · [[KL Divergence]] · [[Tokenization]] · [[Test-Time Compute]] · [[Distillation]] · [[Training language models to follow instructions with human feedback]] · [[Direct Preference Optimization (DPO)]] · [[Small Language Models as Judges for Rubric-Based Reinforcement Learning]] · [[ImpossibleRubrics- Stress-Testing Generated Rubrics as Reward Signals]] · [[Dynamic Important Example Mining for Reinforcement Finetuning]] · [[Towards Full Pipeline FP8 Reinforcement Learning for LLMs]] · [[Megatron-LM- Training Multi-Billion Parameter Models Using Model Parallelism]] · [[Decoupled Weight Decay Regularization (AdamW)]]

New topics worth writing: Group Sequence Policy Optimization (GSPO), Rollout Routing Replay for MoE RL, DAPO dynamic sampling, learnability filtering and LILO, Dr. GRPO and advantage normalisation, Firecracker microVM sandboxes for agent RL, Tau2-Bench user-simulator sensitivity, Arena-Hard v2, benchmark decontamination pipelines, Harbor task format, AgentWorldModel synthetic MCP environments
