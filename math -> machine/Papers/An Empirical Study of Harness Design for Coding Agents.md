---
title: "An Empirical Study of Harness Design for Coding Agents"
authors: ["Fan et al."]
year: 2026
arxiv: "2609.20804"
url: https://arxiv.org/abs/2609.20804
priority: Good-To-Read
read_on: 2026-09-26
tags: [paper, llm]
---
## The Core Idea

A **coding harness** is the software wrapper around a model that turns it into an agent: the loop that calls the model, the tools it can call, the plan it keeps, and the rule for what to throw away when the conversation gets too long. Almost every paper before this one shipped a whole harness (SWE-Agent, OpenHands, Agentless) and reported one number. So when Claude-Opus does better in OpenHands and Claude-Sonnet does better in SWE-Agent, nobody can say *why*.

This work takes the harness apart. One fixed ReAct loop, three swappable parts: **planning**, **action space**, **context management**. Everything else — permissions, post-edit linting, stuck detection — is frozen so it cannot confound anything. 176 matched settings: 4 models × 2 benchmarks × (5 context strategies × 4 context budgets + 2 extra ablations).

The headline result is that **no component is universally good**. Each one's value flips sign depending on the model's strength, the task type, or how tight the context budget is.

> [!NOTE] Coding harness
> The non-model software layer of a coding agent: control loop, tool schemas, planning scaffold, and context-compaction policy. Holding the model fixed and swapping harnesses changes success rates by tens of points. ^coding-harness

The four conditional findings:

1. Context management is mostly an **overflow insurance policy**. At 32k it is worth ~36 points on SWE-Bench; at 128k, ~3 points. Its benefit tracks exactly the rate at which unmanaged runs die from a full window.
2. Cheap rule-based deletion **before** expensive LLM summarisation (their "T4") is the best cost profile. Making deleted text recoverable is machinery models essentially never touch.
3. Planning is an **accuracy crutch for weak models** and a **cost saver for strong ones** — it stops weak models from quitting early, and stops strong models from over-verifying.
4. Predefined file tools help models that cannot drive `bash`; `bash`-only is cheaper *and* more accurate for models that can.

What it unlocks: a modular test bed where you can ask "does this new harness trick survive a different model size and a different context budget?" — which is the question the field was not asking.

## The Methodology

### The loop

Fixed [[Agentic Workflows#🖼️ The ReAct loop — it is just a while-loop with a model in it|ReAct]] loop. Each turn = reason → action → observation appended to history $H$. Built on [[LangGraph]]; benchmark containers driven through Harbor. Max 300 steps per task.

### Component 1 — Planning

A persistent to-do list the model owns.

- System prompt tells it: for any task with ~3+ steps, the **first** action must be `update_plan`. Keep exactly one item `in_progress`.
- First turn gets a reminder if no plan exists yet.
- Every later turn, the current plan is **re-injected** into the model input as a `<system-reminder>` — and crucially it is *not* stored in conversation history. So the plan is always fresh and never duplicated. This is the same trick as a [[System Prompt]]: config re-sent, not accumulated.
- Planning-off removes the prompt, the reminders, the injection, and the tool.

### Component 2 — Action space

Two conditions.

**Full tool set** (8 tools + auxiliaries): `read_file`, `write_file`, `edit_file`, `list_files`, `glob_files`, `grep_text`, `bash`, `web_fetch`. Plus `update_plan` (if planning on) and `recall_event` (if T2/T4).

**bash-only**: the predefined file/search/web tools are deleted from the registry. `bash` remains. Auxiliaries stay.

This is not a clean single-variable change and the authors say so. The full tool set also brings:
- **read-before-write enforcement** — you cannot `edit_file` a file you have not fully read this session, checked via content hash
- **file-state tracking** — the harness knows what the agent has seen
- **automatic post-edit diagnostics** — after a Python edit, the harness runs `ruff`/`pyflakes` and staples the findings onto the tool result

None of that happens if you edit through `bash`. So the ablation measures *the whole interface*, including its safety rails.

No web search — SWE-Bench tasks come from public GitHub issues and search would leak the ground-truth pull request.

### Component 3 — Context management

Three composable mechanisms:

- **M1 Elision** — replace the body of a stale tool observation with a stub: `[tool output elided: 412 lines / 18k chars...]`. Original discarded.
- **M2 Recall** — same elision, but the original is written to the filesystem and a `recall_event(id)` tool can read it back. Elision becomes reversible (lossless).
- **M3 Summarisation** — fold the oldest messages into a running natural-language summary. Produced by a **separate, tool-free call to the same model under evaluation**, with a fixed-heading prompt (Goal / Files touched / Done / Pending / Errors & fixes / Current state / Next step). The summary replaces the events it covers.

Five tiers:

| Tier | M1 elide | M2 recall | M3 summarise |
|---|---|---|---|
| T0 | ✗ | ✗ | ✗ |
| T1 | ✓ | ✗ | ✗ |
| T2 | ✓ | ✓ | ✗ |
| T3 | ✗ | ✗ | ✓ |
| T4 | ✓ | ✓ | ✓ |

T0 has no compaction: when history exceeds the window, the run dies with an error.

**Two thresholds** in T4. Let $B_1 < B_2$ be a soft and a hard token threshold, set at $0.6$ and $0.85$ of the usable window. The [[System Prompt|preamble]] (system prompt + task) and a recent window (budget $0.3$ of the window, floor of two turns) stay **verbatim**. Only the middle region $M$ is touched.

Per turn:

$$\text{tokens}(H) \ge B_1 \;\Rightarrow\; \text{elide bulky observations in } M \text{ (M1), store originals (M2)}$$
$$\text{tokens}(H) \ge B_2 \;\Rightarrow\; \text{additionally summarise oldest events in } M \text{ (M3)}$$

T1–T3 have only one action each, so they fire at $B_2$ only. T4 is the only staged policy: elide early and cheaply, summarise only if that was not enough.

### Fixed substrate (held constant everywhere)

- **Safety**: path guard rejecting escapes from project root including via symlinks; read-before-write with content hashing; allow/ask/deny permission layer. Tool errors are returned *as observations*, never raised — a failed action never kills the loop.
- **Post-edit diagnostics**: `ruff`/`pyflakes`/syntax-only fallback on edited Python files, appended to the tool result.
- **Stuck detection**: a streak = consecutive calls with identical tool name *and* byte-identical arguments. At 5 identical calls (or 5 identical failing calls) inject a one-time "change approach" reminder. At 8 identical failing calls, terminate the run early rather than burning to the 300-step budget.

### Models, data, pricing

Nemotron-3 at 30B / 120B / 550B (a within-family capability axis) plus Mistral-Medium-3.5-128B (cross-family check). Served locally with SGLang, BF16, temperature 0, top-$p$ 0.95, 16,384 output tokens per turn. Tool results truncated to 24k chars. Up to 8 read-only tools in parallel per step.

Benchmarks: SWE-Bench Verified (500 human-verified GitHub issues, Python only) and Terminal-Bench 2.1 (89 end-to-end command-line tasks).

OpenRouter pricing per 1M in/out tokens: $0.05/$0.20 (30B), $0.08/$0.45 (120B), $0.50/$2.20 (550B), $1.50/$7.50 (Mistral).

Significance: two-sided exact **McNemar test** on task-paired outcomes within three comparison families, Benjamini–Hochberg FDR control at 0.05.

> [!NOTE] McNemar test
> The paired test for two binary classifiers on the same items. Only the **disagreements** count: of tasks where exactly one config succeeded, is the split further from 50/50 than chance? The right test here because both harnesses face identical tasks. ^mcnemar

## Ablation Studies and Experiments

### Context management: it is an overflow fix, not an intelligence fix

Averaged over models, the gap between managed tiers (T1–T4) and T0:

| Window | SWE-Bench gap (pp) | Terminal-Bench gap (pp) |
|---|---|---|
| 32k | **+35.7** | +9.5 |
| 64k | +15.9 | +7.5 |
| 96k | +5.5 | +4.8 |
| 128k | +2.7 | +2.8 |

And the mechanism, plainly: T0's **window-overflow failure rate** falls from 78.7% → 8.7% on SWE-Bench and 61.0% → 12.1% on Terminal-Bench across those same budgets. **All managed tiers overflow on exactly zero tasks at every budget.** The gap is the overflow rate.

The single most vivid number: Nemotron-3-550B at 32k scores **6.4%** under T0 and **58.4%** under T3. A 52-point swing from a compaction rule.

### T4 is the cost winner, not the accuracy winner

Accuracy across T1–T4 is broadly a wash. Cost is not. T4 has the lowest mean cost per task at all four window budgets, and the lowest peak-context ratio (peak tokens ÷ nominal window) at all four.

Why: at 32k, T1 and T2 trajectories still hit approximately the *full* window before acting; T3 and T4 stay well below it. And T4 calls the expensive mechanism less — fewer M3 summarisation calls than T3 at every budget, because early cheap elision already reclaimed the tokens. Concrete: 550B at 32k, T3 costs $1.25 vs T1's $2.57 for a higher score; T4 $1.45.

The lesson generalises past this paper: **put the free compaction before the paid one.**

### The negative result that matters: recall (M2) is dead machinery

T1 vs T2 differ *only* in whether `recall_event` exists. Over 32 matched comparisons: T2 wins 15, loses 14, ties 3. Equal-weight mean difference **$-0.36$ pp** ($+0.40$ on SWE-Bench, $-1.12$ on Terminal-Bench).

Usage numbers:
- 36 of 64 recall-enabled settings (56.3%) **never call it once**
- median invocation rate is **zero**
- mean calls/task: $0.540$ at 32k → $0.069$ (64k) → $0.011$ (96k) → $0.007$ (128k)
- all 16 planning/action-space settings at T4/128k: **zero calls**
- usage is concentrated almost entirely in the *weakest* model under the *tightest* budget
- the heaviest user — 30B, Terminal-Bench, 32k, T2, at 4.326 calls/task — scores **3.37 points worse than T1**, which just throws the data away

So lossless context management sounds obviously better and is not, because it depends on the model choosing to retrieve, and models do not. This is the same failure mode as [[Agentic RAG]]: giving the agent a retrieval option is not the same as it retrieving well.

### Planning flips role with capability

At T4/128k, full tools, planning off → on:

| Model | SWE-Bench SR | SWE-Bench cost | TB SR | TB cost |
|---|---|---|---|---|
| Nemotron-3 30B | 13.6 → **25.2** (+11.6) | $0.02 → $0.09 | 9.0 → 13.5 | +75% |
| Nemotron-3 120B | 46.6 → 44.0 | $0.25 → $0.34 | 28.1 → 28.1 | −26% |
| Nemotron-3 550B | 67.8 → 65.8 (−2.0) | $3.31 → $2.33 (**−30%**) | 46.1 → 44.9 | −3.6% |
| Mistral-3.5 | 69.0 → 68.6 (−0.4) | $4.65 → $3.14 (**−32%**) | 39.3 → 37.1 | −40% |

Execution deltas explain it. For 30B, planning multiplies turns by **+293%** and tool calls by **+474%** on SWE-Bench. For 550B and Mistral it *cuts* turns by ~24% and ~23%.

The trajectory analysis nails the mechanism:

- **30B without planning dies at turn 5** (median), and **68.6% of runs terminate without ever editing a file**, 58.4% stuck in the Localize phase. With planning: 27.8% and 10.4%. Planning does not make it smarter; it stops it from giving up.
- **550B and Mistral with planning stop verifying.** Median trajectory 108 → 74 turns (550B) and 68 → 53 (Mistral), and the phase decomposition puts nearly all of the reduction in **Verify**, not Localize or Fix. The without-edit rate barely moves (both under 3%). Planning fixes their *stopping rule*.

> [!NOTE] Stopping behaviour as the real cost lever
> For a strong model, most wasted spend on long-horizon coding is post-edit re-verification, not exploration. A scaffold that says "the plan is complete" is a cheap termination signal. ^stopping-behaviour

### Action space: `bash` is a capability test

At T4/128k, planning on, full tools vs bash-only:

| Model | SWE-Bench SR | TB SR | Cost change |
|---|---|---|---|
| 30B | **25.2 → 10.2** | 13.5 → 3.4 | cheaper, useless |
| 120B | 44.0 → 42.4 | 28.1 → 23.6 | flat |
| 550B | 65.8 → **69.4** | 44.9 → **50.6** | SWE −53%, TB −30% |
| Mistral-3.5 | **68.6 → 45.4** | 37.1 → **43.8** | cheaper both |

The 30B failure is not subtle and is worth internalising: it emits tool calls for tools **that do not exist in the bash-only registry**, because those call patterns were baked in during training. The harness cannot resolve them, and **66% of bash-only Terminal-Bench trajectories terminate on such out-of-interface emissions**. Median trajectory collapses from 71 to 15 turns. The predefined tool set was not helping it think — it was matching its learned action vocabulary.

The 550B result is the opposite. bash-only issues **32% fewer calls** on SWE-Bench and 24% fewer on Terminal-Bench, because it bundles operations into composite shell commands. Median Terminal-Bench trajectory 47 → 31 actions, while the **Write-code share rises 16% → 27%** — a shorter run with more of it spent writing code.

**Mistral is the interesting case: the answer flips by benchmark.** Full tools is +23.2 points on SWE-Bench but −6.7 on Terminal-Bench. Figure 7(c) explains it: with all tools available, Mistral routes **71.9% of Terminal-Bench workspace actions through `bash`** but only **40.4% on SWE-Bench**. Terminal-Bench is shell-centric, so the predefined tools are just competing noise. SWE-Bench needs structured read/search/edit. And the failure is pre-repair: **32.8% of Mistral's bash-only SWE-Bench runs end without editing any file**, vs **1.2%** with full tools; the share of unresolved runs never reaching the correct file rises from 16.0% to 41.4%.

### Action granularity — the fine-grained evidence

| Model | Re-patches/task (tools → bash) | Median largest edit (lines) | TB create-or-replace share |
|---|---|---|---|
| 30B | 3.3 → 0.4 | 26 → 13 | 28% → 64% |
| 120B | 2.8 → 2.2 | 22 → 24 | 39% → 76% |
| 550B | 4.6 → 1.5 | 18 → **54** | 51% → 76% |
| Mistral | 3.0 → 1.3 | 87 → 68 | 28% → 57% |

Predefined tools make each *individual* action simpler and safer, but they pay for that by needing more actions and more incremental repair cycles. `edit_file` with its read-before-write gate produces 4.6 re-patches per task for 550B; `bash` produces 1.5 with a 3× larger typical edit.

### Failure-stage attribution

An LLM judge assigns each *unresolved* SWE-Bench run to the earliest broken stage (file localisation → line localisation → patch → verification). The distribution shifts with capability:

- 30B: **56.3%** of failures are file localisation (T4). Rises to 73.8% with planning off, 76.6% with bash-only.
- 550B / Mistral: majority fail at **patch implementation** (64.9% / 60.3%) — they find the right place and write the wrong fix.
- Across context tiers at 128k the distribution barely moves, confirming tiers change *how many* runs fail, not *where*.

Judge validation: 3 human annotators, 200 trajectories, non-overlapping splits, **15,610 labelled units**. Aggregate raw agreement ≈ 94.2%, weighted mean Cohen's $\kappa = 0.929$. SWE-Bench action labels $\kappa$ 0.881–0.984; Terminal-Bench actions 0.758–0.965; failure stage 0.813–1.000. This is unusually careful validation for a [[Evals|LLM-as-judge]] pipeline and worth copying.

### What context management does *not* do

At 128k, T0–T4 leave trajectory shape almost unchanged: 30B median 39–42 turns, 550B median 70–74, re-patch counts within 2, without-edit rate within 4 pp for three of four models. Context management **extends** trajectories; it does not change what the agent does inside them. Which is exactly why its benefit vanishes when the window stops binding.

## Worth Remembering

**The one-line practical rule.** Pick harness components for your model, budget and task type — none of them is a default. Concretely: tight window → managed context, staged (elide then summarise); weak model → planning on, predefined tools; strong model → planning on for the cost saving, and test bash-only, especially on shell-centric work.

**Limitations the authors own up to:**

- Planning and action space are ablated **only** at T4/128k. No full factorial — compute limits. Whether the crossovers hold at 32k is unknown.
- **One run per task.** Terminal-Bench has 89 tasks, so most Terminal-Bench contrasts do not pass the paired McNemar test. Those conclusions rest on *consistent direction across models and budgets*, not on individual significance. Read them as suggestive.
- The action-space intervention is **bundled**: tool availability + interface prompts + file-state tracking + read-before-write + post-edit diagnostics all change together. You cannot attribute the effect to tool count or action granularity alone.
- Planning is one prompt and one `update_plan` mechanism. Not a claim about planning as a reasoning strategy — see [[Chain of Thought]] and [[Planning and Decomposition]] for the separate question.
- SWE-Bench Verified is **Python only**. Model size is an imperfect proxy for capability; Mistral's benchmark-dependent preference proves it.

**Surprises worth carrying:**

- Summarisation-only (T3) at 32k beats every other tier for 550B (58.4%) and is *cheaper* than elision-only. Compressing history into a summary can be better than keeping stale raw text.
- The **summariser is the model under evaluation**, called tool-free. So the compaction quality co-varies with model strength — an uncontrolled confound in any "does compaction help weak models?" question, and a possible partial explanation for why the 30B benefits less from T3/T4 than 550B does.
- Planning *lowers* accuracy slightly for the two strongest models on both benchmarks. Scaffolds can cost you a little ceiling.
- Post-edit diagnostics only fire through the predefined tools. If you run bash-only you silently lose your linter feedback loop — plausibly part of why bash-only re-patches less (no diagnostics prompting a second pass) and part of why Mistral collapses on SWE-Bench.

**Practical caveats if you wanted to build this:**

- The read-before-write gate needs a **content hash** to detect external modification, and a *partial* read must not satisfy it. Otherwise the model overwrites files it half-looked-at.
- Stuck detection must match on tool name **and byte-identical arguments** — a paginated read at a new offset is progress, not a loop.
- Return tool errors as observations, never as exceptions. An agent that can read its own error recovers; a crashed loop cannot.
- Keep the plan out of history and re-inject it. Otherwise you accumulate N stale copies of a to-do list, which is both a token bill and a [[Lost in the Middle]] hazard.

**Follow-up questions:**

- Does a *harness-side* retrieval policy — the harness decides what to recall, not the model — rescue M2? The failure was agency, not storage.
- The overflow story suggests context management ≈ [[Prompt Caching]]-friendly truncation. What happens to cost when the preamble is cached and the middle region churns?
- Would training a model on the target tool schema (the 30B's real problem) erase the action-space finding entirely?
- Does the planning crossover point track a measurable property — instruction-following score, self-termination rate — rather than parameter count?

## Links

Related: [[Agentic Workflows]] · [[Context Engineering]] · [[Context Window]] · [[Memory]] · [[Tool Use]] · [[Planning and Decomposition]] · [[Evals]] · [[Agent Evaluation]] · [[Agent State and Checkpointing]] · [[Lost in the Middle]] · [[Agentic RAG]] · [[Sandboxing]] · [[LangGraph]] · [[Observability and Tracing]] · [[Cost and Latency]] · [[System Prompt]] · [[Structured Output]] · [[Prompt Caching]] · [[Agent Frameworks]] · [[Multi-Agent LLM Systems]] · [[Human in the Loop]] · [[Reflection]] · [[SWE Refactor Bench- Can Coding Agents Complete a Long-Horizon, Whole-Repository Stack Migration]] · [[WHALE- A Simple Recipe for Joint Harness-Weight Optimization]] · [[The Handoff Tax- Continuing Non-Native Trajectories in LLM Agents]] · [[Using Grounded Theory for Agent Behavior Analysis at Scale]] · [[LatentPress- Context Compression Beyond Text and Vision]]

New topics worth writing: McNemar's test for paired binary outcomes, Benjamini–Hochberg FDR control, SWE-Bench Verified, Terminal-Bench, context compaction policies (elide vs summarise vs offload), trajectory phase annotation for agent analysis, read-before-write enforcement as an agent safety gate, stuck-detection and loop-breaking heuristics, agent-computer interface design, tool-schema training mismatch, agent termination and stopping rules, action granularity in coding agents
