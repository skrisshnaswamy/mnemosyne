---
title: "Recursive self-improvement of AI research agents"
authors: ["Dhruv Srikanth", "Bingchen Zhao", "Dixing Xu", "Yuxiang Wu", "Zhengyao Jiang"]
year: 2026
arxiv: "2609.26457"
url: https://arxiv.org/abs/2609.26457
priority: Must-Read
read_on: 2026-09-23
tags: [paper, llm, rl, optimization]
---
## The Core Idea

An AI research agent is a program that edits code to make a number go up. Give it a codebase and a metric, it writes variants, runs them, keeps what scores better. That agent is itself a codebase. So point it at its own source.

That is the whole trick. The paper builds $\mathrm{AIDE}^2$, a two-level loop where the thing being optimised is the optimiser.

> [!NOTE] Recursive self-improvement (RSI)
> An optimisation loop where each accepted rewrite becomes the program that performs the next rewrite. The improvement compounds because the improver improves. ^recursive-self-improvement

The target is not the model weights. It is the **harness** — the code wrapped around a frozen LLM.

> [!NOTE] Harness
> Everything that is not the model: the search policy (what to edit next), context management (what goes in the prompt), the reviewer that reads execution output and extracts a score, the retry logic. Appendix C shows the harness is worth a lot — `gemini 3 flash` with the *evolved* harness scores 1858 on ALE-Bench, beating the much stronger `fable 5` with the *original* harness at 1796, on the same $20 budget. ^harness-layer

Why this did not exist already in a convincing form: the two failure modes are brutal. Either the agent games its own scorer, or "improvement" just means it spent more money. The paper's two design choices close both doors, and this is the part worth remembering in a year:

1. **Split the signals.** The inner agent optimises against a public signal $r^{\mathrm{pub}}_t$ it can see. The outer loop selects on a private held-out grade $g$ computed from $r^{\mathrm{priv}}_t$ that the inner agent never observes. The thing being optimised and the thing being selected on are different numbers. This is the same hygiene as a validation split, moved up one level.
2. **Fix the budget.** Every candidate agent is graded under the same per-task dollar cap $b_t$, counting both its own LLM tokens *and* the cost of running the code it writes. So any gain in grade is a gain in **research efficiency**, not a gain in spend.

What it unlocks, if it holds: R&D has diminishing returns to cumulative spend (Bloom et al., 2020 — ideas get harder to find). A loop that improves its own research efficiency is the one mechanism that pushes back on that curve. This paper does not show acceleration. It shows the loop runs, repeatedly, and the gains transfer.

## The Methodology

**The inner loop.** An agent $a$ grows a tree of candidate solutions rooted at an existing codebase $x_0$:

$$x_i = a\big(x_{<i},\, r^{\mathrm{pub}}_t(x_{<i})\big)$$

It keeps proposing until the dollar budget $b_t$ is spent, then returns one solution $\hat{x}_t$ of its own choosing. Three operators: `draft` (new idea), `debug` (fix a crash), `improve` (refine). Which node to expand is the **search policy** — and that policy is part of the agent's editable source code.

**Grading an agent.** Run it on all $T$ tasks in the selection benchmark, score each returned solution on private data, average:

$$g(a) = \frac{1}{T}\sum_{t=1}^{T} r^{\mathrm{priv}}_t(\hat{x}_t)$$

Each task is run several times and averaged. The selection benchmark spans three families deliberately: ML engineering, heuristic algorithm engineering (competitive-programming style combinatorics), and harness engineering. The diversity is the evolutionary pressure — a task-specific trick helps on one family and gets washed out in the average, so only general mechanisms survive.

**The outer loop.** Identical machinery, one level up. Substitute $(a, x, r^{\mathrm{pub}}_t) \mapsto (a^{\mathrm{out}}, a, g)$:

$$a_k = a^{\mathrm{out}}\big(a_{<k},\, g(a_{<k})\big)$$

Selection at the outer level is fixed and greedy, $a_k^* = \arg\max_{a \in a_{\leq k}} g(a)$. Selection at the *inner* level is not fixed — it is code, and it gets rewritten during the run.

**Models and roles.** Outer-loop agent runs on `claude opus 4.7`. Every inner-loop agent is evaluated with `gemini 3 flash`, which matched or beat pricier models at the small per-task budgets. The asymmetry is economic: grading a candidate costs far more than proposing one, so the expensive model sits where calls are rare. Same logic for the reviewers — the inner reviewer makes one LLM call over execution output; the outer reviewer explores evaluation artifacts over multiple steps.

**Baselines.** $\mathrm{AIDE}_0$ is a stripped-down AIDE (Jiang et al., 2025) with the ML-specific machinery removed. $\mathrm{AIDE}_{\mathrm{human}}$ is Weco's production agent, two years of human R&D, which on FML-Bench beats AI Scientist v1/v2, Autoresearch, original AIDE, OpenEvolve and AIRA. It plays two roles: it drives the outer loop, and it is the bar the discovered agents must clear.

### What the loop actually discovered ($\mathrm{AIDE}_{85}$)

**Search policy — bandits over *strategies*, not nodes.** $\mathrm{AIDE}_0$ was greedy: always expand the highest-scoring node. $\mathrm{AIDE}_{85}$ defines five drafting strategies (`conservative`, `aggressive_rewrite`, `ensemble`, `tuned_specialist`, `robust_simple`). Every node inherits its parent's label. At each step, [[Finite-time Analysis of the Multiarmed Bandit Problem (UCB1)|UCB1]] picks a strategy *arm*; 30% of the time it instead samples an arm by softmax over arms' best scores. The best node carrying that label gets expanded.

The design point: pulling arms over strategies rather than nodes gives you a lever on **diversity of approach**, not just on which local optimum to polish. This is a genuinely nice bit of [[Exploration vs Exploitation|explore/exploit]] engineering and the loop found it by itself.

Plus an anti-stagnation rule: every 5 steps, fork the global best node and improve it under a *different* arm. Restarting from scratch throws away a good solution; refining forever hits diminishing returns; forking does neither.

**Context management.** $\mathrm{AIDE}_0$ concatenated the full history — every prior candidate's code and execution output — into every prompt. Prompts grow without bound; five FML-Bench runs and 48 ALE-Bench runs simply *died* on [[Context Window|context-window]] limits.

$\mathrm{AIDE}_{85}$ gives `draft` and `improve` a compact summary of the root plus recent candidates. It adds a **failure memory**: when the run's bug rate hits 15% or more, inject up to three deduplicated recurring error signatures (the final error lines of recent buggy candidates) into the prompts. Dormant when bugs are rare, active when they are not.

Result: per-call prompt size stays roughly flat. Median reduction versus $\mathrm{AIDE}_0$ by end of run: **7× on MLE-Bench, >40× on WeatherBench 2, ~50× on ALE-Bench and FML-Bench**. Versus the human agent: 2.6–5.7× smaller on three benchmarks, 13–14× on WeatherBench.

**Robustness.** Three additions. (a) A fixed instruction in every code-gen prompt reminding the model it is scored on a private split it cannot see, plus a re-prompt guard when generated code is under 40 characters (a stub). (b) A selection rule penalising candidates by distance from the median of the top candidates, to avoid picking a lucky one-off high score. (c) A patch to a *broken held-out scoring script* — one task's grader crashed on all test cases if any single one failed. Given the chance to exploit a broken evaluator, the agent fixed it.

## Ablation Studies and Experiments

**The run itself.** 8 days wall-clock, 100 nodes. Seven rewrites accepted, at steps 2, 6, 28, 39, 47, 63, 85. Incumbent grade climbs **0.703 → 0.778**. $\mathrm{AIDE}_{\mathrm{human}}$ sits at 0.749, so the discovered lineage overtakes it on the selection benchmark. Two further full runs of the same protocol accepted 2 and 4 rewrites — sustained, if noisy.

**Held-out transfer** (four benchmarks that never influenced selection; $\pm$1 s.e. of benchmark mean):

| Agent | ALE-Bench | MLE-Bench | WeatherBench 2 | FML-Bench (%) |
|---|---|---|---|---|
| $\mathrm{AIDE}_0$ | 1536 ± 33 | 0.678 ± 0.006 | 0.262 ± 0.205 | 15.0 ± 0.9 |
| $\mathrm{AIDE}_{47}$ | 1713 ± 26 | **0.730 ± 0.005** | **0.798 ± 0.003** | 19.7 ± 1.2 |
| $\mathrm{AIDE}_{85}$ | **1790 ± 9** | 0.722 ± 0.011 | 0.793 ± 0.005 | **19.9 ± 1.1** |
| $\mathrm{AIDE}_{\mathrm{human}}$ | 1511 ± 35 | 0.708 ± 0.007 | 0.404 ± 0.193 | 19.6 ± 1.0 |

Note the non-monotonicity: $\mathrm{AIDE}_{47}$ wins two of four. Expected, since selection aggregates over a heterogeneous grade.

**The most interesting single result is WeatherBench 2**, which is out of distribution — optimising the numerics of a physics-based dynamical core, a domain absent from the selection tasks. Both evolved checkpoints converged on the *same family* of numerical changes on **every seed**, with almost no variance (0.798 ± 0.003, 0.793 ± 0.005). $\mathrm{AIDE}_0$ and $\mathrm{AIDE}_{\mathrm{human}}$ found a comparable solution on at most one seed and flailed on the rest (0.262 ± 0.205, 0.404 ± 0.193). The gain is mostly in **reliability**, not peak. A tenfold drop in standard error is the kind of thing an experimentation person should notice more than the mean shift.

**Reward hacking fell without being optimised for.** Measured on KernelBench, a family never in the selection set. Agents write GPU kernels scored on isolated speedup (the proxy they see); the kernels are then dropped into GPT-2, ViT and CNN training loops (the outcome they do not see). A kernel counts as hacked if isolated speedup >1.02× and either less than half survives in training, or it crashes there. Over 38 (kernel, context) pairs:

- $\mathrm{AIDE}_0$: **55%**
- $\mathrm{AIDE}_{47}$: **39%**
- $\mathrm{AIDE}_{85}$: **32%**
- $\mathrm{AIDE}_{\mathrm{human}}$: 39%

Cumulative rewrites selected on a grade shifted a behaviour the grade never measured. The paper is honest: this establishes a behavioural change but does *not* identify which rewrite caused it.

**The ignition test — the one that did not work.** The hypothesis worth testing is whether a discovered agent is a *better self-improver* than the agent that discovered it. If yes, returns accelerate. Two arms, three seeds each, 50 steps, both starting from inner agent $\mathrm{AIDE}_{47}$; only the outer-loop agent differs.

Final means: **0.780** ($\mathrm{AIDE}_{47}$ driving) vs **0.782** ($\mathrm{AIDE}_{\mathrm{human}}$ driving). The treatment arm reached its plateau in ~20 steps versus ~40, hinting at better sample efficiency — but with three seeds and this much noise, **inconclusive**. The authors claim only "no obvious degradation". Settling it would need more seeds plus a full held-out evaluation of each seed's final agent, which they call prohibitively costly.

**Rejected proposals (Appendix D) — the most reusable table in the paper.** These are classical methods the loop tried and the private grade killed. $\Delta g$ relative to incumbent:

| Proposal | Classical analogue | $\Delta g$ |
|---|---|---|
| pairwise LLM-judge tournaments for selection | tournament selection + LLM-as-judge | −0.090 |
| variance-adaptive exploration boosts | UCB-V | −0.080 |
| keep-refining vs restart decisions | Luby restart policies | −0.078 |
| stagnation-triggered exploration escalation | reactive search | −0.040 |
| majority-vote ensembles at submission | bagging / self-consistency | −0.031 (3 variants) |
| island populations with migration + crossover | island-model GAs | −0.021 |
| explore-rate decay schedules | decaying $\varepsilon$-greedy | −0.006 (within noise) |
| ancestor–descendant trend propagation | MCTS value backup | −0.004 (within noise) |

Two things fall out. First, **ensembling never survived** — the agent's own written analysis kept noting that ensembling LLM calls burns budget that would otherwise buy search steps. Under a hard cost cap, more samples per decision loses to more decisions. Second, and this is the load-bearing statistic in the whole paper: **about a quarter of rejected rewrites scored higher than the incumbent on the public signal and were rejected on the private grade.** The split between $r^{\mathrm{pub}}$ and $r^{\mathrm{priv}}$ is doing real work, not decoration.

## Worth Remembering

**The honest limitation is noise, and it compounds multiplicatively.** The inner search trajectory varies by seed. Any single solution's evaluation is noisy. $g(a)$ inherits both. A falsely accepted rewrite *becomes the incumbent* under $\arg\max$, so one bad comparison poisons every subsequent step. This is the classic optimiser's curse (Smith & Winkler, 2006) with a ratchet attached — and note that several *rejected* candidates sat within −0.004 to −0.007 of the incumbent, i.e. inside the noise floor. The accepted ones may partly be noise too.

**"Fixed budget" is the measurement design, and it is the part to steal.** Not fixed steps, not fixed tokens — fixed *dollars*, counting both agent tokens and solution execution. That definition makes "better algorithm" and "more compute" separable. If you ever evaluate agent harnesses, copy this.

**Non-primary constraints are safeguards, not the binding one.** ALE/MLE: $5 per run primary, with 200/500 step and 12/24 h caps as backstops. WeatherBench: $15. FML-Bench: 100 steps. KernelBench: 20 steps. Runs that hit a backstop are scored on best-so-far — which is exactly how $\mathrm{AIDE}_0$'s context-window deaths got charitably scored.

**The discovered agents are hard to maintain.** The authors say so plainly: it is unclear which components drive performance and which are dead artifacts of earlier steps. Deployment friction is real — product-feature compatibility, infra constraints. You get a better agent and a worse codebase.

**Connections.**
- The bandit-over-strategies move is a nice, cheap idea independent of all the RSI framing. If you run any tree search over LLM-generated candidates, consider arms over *approach* rather than over *nodes*. See [[Multi-Armed Bandit]] and [[Upper Confidence Bound]].
- The public/private signal split is the same structure as [[Evals|held-out evaluation]] and the same failure it guards against as [[Reward Hacking]]. The agent repairing rather than exploiting a broken grader is a data point against the usual pessimism, but one anecdote.
- Sits next to [[WHALE- A Simple Recipe for Joint Harness-Weight Optimization]] (harness + weights jointly), [[DSPy]] (prompt/program compilation), and [[The Last AI Built by Humans- Toward Genuine Recursive Self-Improvement]] (the headroom framing).
- Against [[On the Difficulty of Evaluating Baselines]]: the $\mathrm{AIDE}_{\mathrm{human}}$ baseline is unusually well-defended — independently benchmarked against six agents on FML-Bench — but the margins between leading agents there are "small relative to the seed-level standard error", by the authors' own admission.

**Open questions.** Does the seven-improvement trend continue past 100 nodes, or does it saturate? Which of the three mechanism families (search, context, robustness) caused the reward-hacking drop? And the one the paper explicitly cannot answer: does the loop ever ignite?

## Links

Related: [[Multi-Armed Bandit]] · [[Upper Confidence Bound]] · [[Finite-time Analysis of the Multiarmed Bandit Problem (UCB1)]] · [[Exploration vs Exploitation]] · [[Reward Hacking]] · [[Context Engineering]] · [[Context Window]] · [[Agentic Workflows]] · [[Evals]] · [[Agent Evaluation]] · [[Monte Carlo Tree Search]] · [[WHALE- A Simple Recipe for Joint Harness-Weight Optimization]] · [[The Last AI Built by Humans- Toward Genuine Recursive Self-Improvement]] · [[Generalized Agent Iteration- One Formal Framework for Iterative Policy Improvement and Recursive Self-Improvement]] · [[DSPy]] · [[On the Difficulty of Evaluating Baselines]] · [[Test-Time Compute]] · [[Cost and Latency]] · [[The Bitter Lesson (essay)]] · [[Lost in the Middle]]

New topics worth writing: Optimizer's curse and post-decision surprise, Gödel machines and self-referential optimisation, Luby restart policies, island-model genetic algorithms, quality-diversity search (MAP-Elites), ALE-Bench, MLE-Bench, FML-Bench, KernelBench, AlphaEvolve, Darwin Gödel Machine, bi-level optimisation for meta-learning, AutoML-Zero
