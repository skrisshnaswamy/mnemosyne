---
title: "Schrödinger's Code Repository: Have LLMs Learned SWE-bench or Memorized It?"
authors: ["Chen et al."]
year: 2026
arxiv: "2609.27891"
url: https://arxiv.org/abs/2609.27891
priority: Good-To-Read
read_on: 2026-09-28
tags: [paper, llm]
---
## The Core Idea

SWE-bench is built from famous open-source repos — Django, sympy, scikit-learn. Those repos were in the training data. So when an agent fixes a Django bug, you cannot tell whether it *reasoned* about the code or *remembered* the code.

The usual fix is to collect newer bugs the model has not seen. That works, but new bugs are scarce and skew toward whatever issues happen to be open right now.

The trick here is different: keep the same bug, but change what the repo *looks like*. Rename everything. Shuffle the order of functions. Rewrite the implementation near the fix into equivalent code. Reword the issue text. The program still runs the same way and the same tests still fail, so the task is identical — but every familiar surface cue is gone.

> [!NOTE] SchrodingerRepo ^schrodinger-repo
> Treat the repository *presentation* as a random variable that is only sampled when the agent enters the environment. Feed in a different seed and you get a different, deterministic, semantically identical view of the same repo. The benchmark is no longer a fixed artefact that can leak.

Why this did not exist before: earlier perturbation work (RepoMirage, PoorCodeSumEval) produced *one* obfuscated copy. That copy then becomes a new static artefact that can itself be trained on. Seeding the transformation is what makes it re-usable forever.

What it unlocks: a controlled knob. The underlying task, the execution environment and the pass/fail tests are all held fixed. Any drop in score is attributable to presentation alone.

The headline: Pass@1 on SWE-bench Verified falls **6.0–14.4 points** across four models, and input tokens rise by up to **2.5×**.

## The Methodology

### The leakage check that motivates it

Before building anything, they ran a human-in-the-loop probe. Show the model only the instance ID and a few broad sentences from the issue. No files, no patch, no tests. Ask what it thinks. A human compares the output against hidden reference info and grades the recall into four buckets: nothing useful, correct file/symbol named, correct repair logic, or actual patch/test lines.

Result: **>65% of SWE-bench Verified instances** showed clear leakage evidence, and **>18%** were recalled down to patch or test level.

### Four transformation levels

**Level 1 — problem statement reconstruction.** An LLM rewrites the issue text: reorder information, paraphrase, drop incidental identifiers. A second LLM verifies no task-defining constraint was lost, and triggers a refinement if so. A human engineer signs off at the end.

**Level 2 — namespace remapping.** This is the one that matters. Walk the AST and collect every identifier the repo owns: classes, functions, module vars, imports. Strictly filter out Python builtins, keywords, and third-party symbols — touching those would break execution. Split what remains into subword tokens and remap each token under a seed. `QuerySet` → split into `Query` + `Set` → `Ledger` + `Suite` → `LedgerSuite`. The `Query → Ledger` mapping is then reused everywhere `Query` appears, so names stay internally consistent. Naming conventions are preserved: CamelCase stays CamelCase, `snake_case` stays snake, `UPPER_CASE` stays upper, dotted module paths stay dotted.

Crucially the repo on disk is **not** changed. The mapping is a bidirectional translator sitting between agent and environment: observations get translated into the fake namespace on the way out, agent commands get translated back into real names before execution. The command head of a bash line is left alone; only namespace-bearing arguments and embedded code get rewritten. A session notebook logs every substitution actually used, and reverse translation is restricted to those.

**Level 3 — intra-file reordering.** This one *does* change files on disk. Within a file or class body, find contiguous runs of reorderable definitions (top-level funcs/classes, methods). Non-reorderable statements act as anchors that partition the runs, so global layout survives. Build a dependency graph $G$ over the units in a run: an edge from $u$ to $v$ when $v$ needs a name $u$ defines at *definition time* — decorators, default arg values, type annotations, class bases, class-body expressions. Then sample a random topological order of $G$. If only one valid order exists, leave it alone.

**Level 4 — functionality-preserving rewrite.** Find the code region tied to the repair, then have a constrained rewriting agent emit a unified diff that restates that implementation in a behaviourally equivalent but visibly different form. The goal is explicitly *not* to fix the bug. This attacks memorised implementation patterns near the fix site.

### Keeping it honest

For Levels 3 and 4 a transformed repo $V$ is kept only if

$$\mathrm{Pass\_to\_Pass}(V) = 1 \quad\text{and}\quad \mathrm{Fail\_to\_Pass}(V) = 0$$

First condition: nothing that worked broke. Second condition: the target bug is still broken. Together they guarantee it is still the same unresolved instance. 100 random instances per level were also hand-inspected.

### Scoring

The submission is not the agent's raw patch text. It is reconstructed by diffing final repo state against original repo state, file by file:

$$\mathrm{final\_submission} = \bigoplus_{f \in C_B} \mathrm{Diff}(b_f, r'_f)$$

where $C_B$ is the set of changed files, $b_f$ the original contents, $r'_f$ the final contents. So grading always happens in original SWE-bench coordinates regardless of what the agent saw.

### Setup

Scaffold is `mini-swe-agent`, default config, temperature 0, max 250 actions per instance. Models: GPT-5.4-mini, GPT 5.1, DeepSeek-v4-Flash, Gemini-3.1-Flash-Lite. Levels 3 and 4 are restricted to golden-patch-related files/regions for cost reasons. Three seeded views per instance per level, results averaged.

## Ablation Studies and Experiments

### SWE-bench Verified, all four levels on

| Model | Baseline Pass@1 | Full | Δ | Actions | Input tokens |
|---|---|---|---|---|---|
| GPT 5.1 | 44.6% | 36.2% | −8.4 | +61.5% | +161% |
| GPT-5.4-mini | 46.8% | 35.6% | −11.2 | +76.2% | +255% |
| DeepSeek-v4-Flash | 72.8% | 66.8% | −6.0 | +116% | +254% |
| Gemini-3.1-Flash-Lite | 56.7% | 42.3% | −14.4 | +81.1% | +136% |

All drops significant at $p<0.05$.

### Which level does the work — this is the interesting part

**Level 1 does essentially nothing.** GPT 5.1 and GPT-5.4-mini: 0.0 point change. DeepSeek: −2.0. Rewording the issue does not remove the advantage. That is a genuinely useful negative result — the leakage is not in the issue text's phrasing.

**Level 2 does almost all of it.** −7.4 / −6.4 / −6.0 points, $p<0.01$ everywhere. Actions +63.3% / +32.6% / +112.4%. Level 2 alone ≈ the full setting for GPT 5.1 (37.2% vs 36.2%) and for DeepSeek (66.8% vs 66.8%).

**Levels 3 and 4 are mild.** 0.8–3.4 and 1.2–2.8 points. They still raise cost, and they still contribute in combination — GPT-5.4-mini goes from −6.4 (Level 2 alone) to −11.2 (all four), so the levels are not fully redundant.

Read plainly: **the memorised cue is names.**

### Where the extra actions go

Actions were bucketed into `navigate`, `search`, `read`, `probe`, `edit`, `test`. Of the *additional* actions in the full setting, exploration (first four) accounts for **83.6%** for DeepSeek and **81.6%** for GPT-5.4-mini. Editing and testing get only ~16.5%.

The two models explore differently. DeepSeek leans on `probe` (31.4% of extra actions) — it runs things to interrogate behaviour. GPT-5.4-mini leans on `read` (41.8%) and `search` (33.1%), with `probe` at only 3.7% — static inspection.

That difference predicts robustness. DeepSeek explores hard by default (46 actions baseline vs GPT-5.4-mini's 11) and loses the least accuracy (−6.0) while paying the most in tokens. GPT-5.4-mini is cheap and brittle. **Exploration intensity buys representation robustness.**

### SWE-QA transfer (144 instances, 3 repos)

GPT-5.4-mini: score 70.35 → 65.71 (−4.64), actions +18.15%, input tokens +34.65%. DeepSeek: 72.97 → 72.42 (−0.75) but actions +43.02%, input tokens +59.10%. Same pattern — Level 2 dominates, Level 1 is noise. Levels 3/4 matter less here, because QA needs locating and reading but not localise-edit-test-iterate.

### The control experiment (RQ4) — the most important table

If the transformations just made the tasks harder, the story collapses. So they ran the full setting on 110 SWE-rebench instances created *after* GPT-5.4-mini's release — instances the model cannot have memorised.

Pass@1: **17.27% → 17.27%.** Exactly unchanged. Actions +8.15%, input tokens +22.01%.

So on unfamiliar repos, transformation costs you a bit of exploration but zero accuracy. On familiar repos it costs 6–14 points. The difference *is* the memorisation. They also note the cost increase is larger in long-standing popular repos.

### Case study

`django__django-11999`. Original view: the agent greps `get_.*_display` under `django/db/models/`, hits the right file immediately, done in **37 actions**. Transformed view: the path is `working_repository/storage_engine/object_models/entries/__init__.py` and the method is `render_%s_label`. The agent lists directories, enumerates Python files, searches source and tests, then keeps checking registration logic, tests, config and a reproduction script to convince itself the renamed thing is the display method. **217 actions** — same final patch.

## Worth Remembering

- The single cleanest takeaway: **Level 2 ≈ the whole effect.** If you only have budget for one perturbation, rename the repo's identifiers consistently. Rewording issue text is wasted effort.
- The RQ4 held-out control is what turns this from "we made a harder benchmark" into "we measured memorisation." Copy that design pattern — always include a population where the effect *should not* appear.
- Watch out for the units. A "6.0–14.4 percentage point drop" on a 72.8% baseline is ~8% relative; on 46.8% it is ~24% relative. Gemini's −14.4 was also measured on the 300 most-leaked instances, not the full 500, so it is not directly comparable to the other three rows.
- Accuracy alone under-reports the damage. DeepSeek loses 0.75 points on SWE-QA but burns 43% more actions and 59% more input tokens. If you evaluate agents on score only, you miss that the agent is now doing far more work for the same answer. Cost is a first-class metric here.
- Level 2 never touches disk — it is a translation layer over observations and commands. Elegant, but it is why the authors flag internal-validity risk: it only covers command-line interaction. An agent with an IDE API, language server, or AST navigation tool would need each of those interfaces wrapped too, or the real names leak straight through.
- Levels 3 and 4 were only applied to golden-patch-related files, for cost. So the measured effect of those levels is a lower bound on what repo-wide transformation would do.
- Everything is Python and open-source. Unknown whether the naming-cue dependence holds for Java, Go, or private monorepos with different conventions.
- Level 4's rewriting agent is an LLM asked to produce equivalent-but-different code. The $\mathrm{Pass\_to\_Pass}/\mathrm{Fail\_to\_Pass}$ gate catches behavioural breakage, but there is no guarantee the rewrite is *hard* rather than cosmetic. That may be why Level 4 is the weakest level.
- Open question: does this generalise to *training*? If naming cues are what agents lean on, seeded namespace remapping could work as a data augmentation for training more robust coding agents, not just as an evaluation harness.
- Connection worth drawing: this is the same shape of problem as the recsys replication crisis — strong reported numbers that turn out to be artefacts of a fixed evaluation setup rather than real capability.

## Links

Related: [[Agent Evaluation]] · [[Evals]] · [[An Empirical Study of Harness Design for Coding Agents]] · [[SWE Refactor Bench- Can Coding Agents Complete a Long-Horizon, Whole-Repository Stack Migration]] · [[Shortcut Learning in Deep Neural Networks]] · [[Do ImageNet Classifiers Generalize to ImageNet]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Towards Quantifying Benchmark Optimization in ASR Models]] · [[Agentic Workflows]] · [[Observability and Tracing]] · [[Cost and Latency]] · [[Reward Hacking]]

New topics worth writing: data contamination and benchmark leakage, semantics-preserving program transformation, SWE-bench, seeded evaluation harnesses, AST-based code refactoring, agent action taxonomies, exploration cost as an evaluation metric
