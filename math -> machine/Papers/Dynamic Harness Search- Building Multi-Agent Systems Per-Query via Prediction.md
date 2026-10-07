---
title: "Dynamic Harness Search: Building Multi-Agent Systems Per-Query via Prediction"
authors: ["Som Sagar", "Shasha Li", "Hejie Cui", "Ransalu Senanayake", "Sercan Ö. Arık"]
year: 2026
arxiv: "2610.04137"
url: https://arxiv.org/abs/2610.04137
priority: Good-To-Read
read_on: 2026-10-06
tags: [paper, llm]
---
## The Core Idea

A "harness" is everything around the model: which agents exist, what each is told to do, which tools each may call, and who passes results to whom. SHIFT's claim is that the *right* harness changes from one query to the next, and that you can pick it per query without paying to try the options.

Two queries over the same document corpus: "what revenue did the report state?" needs one reader. "Reconcile this budget workbook and return a corrected file" needs a planner, a solver with Python, and a verifier loop. Run the big harness on the lookup and you burn tokens and latency for nothing. Run the small one on the workbook and you fail.

The problem that blocked per-query choice: **the value of a design choice is only visible after you execute it.** A verifier helps only if the solver's output is checkable. A tool helps only if the agent actually uses it. So components interact, and their worth depends on the query *and* on the rest of the harness. Measuring that meant running candidates — which is exactly the cost you were trying to avoid.

> [!NOTE] Harness
> The full specification of a multi-agent system: agent roles, their instructions ("directives"), their tool permissions, and the edges between them. ^harness-spec

SHIFT's move is to **train a predictor so search never has to execute.** A small local model (Gemma 4 E2B, LoRA-tuned) learns two things from real executions during training:

1. a **policy** over harness-building actions — which component to add next,
2. a **value** — a predicted utility that trades task success against execution cost.

At inference, [[Monte Carlo Tree Search|MCTS]] builds a harness for the query using only forward passes of this small model. Then **one** harness is executed. Construction takes under a second on one H100; executing an OfficeQA harness costs one to eight million executor tokens. The asymmetry is the whole point.

What it unlocks: on six benchmarks with a shared Gemini 3.5 Flash executor, 79.9% mean accuracy against 72.7% for the best of 17 baselines. A cheaper mode gets 74.5% — still above every baseline — at 32% fewer execution tokens than that baseline.

The lineage is [[Mastering Chess and Shogi by Self-Play (AlphaZero)|AlphaZero]], almost exactly: known deterministic transitions, a policy–value net, visit counts as policy targets, and search that calls the net instead of the world.

## The Methodology

### Harnesses as graphs

A harness state $s$ is a directed graph $G = (\mathcal{V}, \mathcal{E})$ with a start node $v_0$. Each agent $v$ has a role $\rho_v$, an ordered list of directives $D_v$, and permitted tools $T_v \subseteq \mathcal{K}_d$ where $\mathcal{K}_d$ is what domain $d$ offers. An edge $(u,v)$ pipes $u$'s output into $v$. Feedback edges let a verifier send work back.

Every graph in the space is **runnable**. That is deliberate: search can compare a half-built harness against a finished one on equal terms.

### The action space

40 actions, grouped three ways:

| Group | Count | Examples |
|---|---|---|
| Structural | 10 | add Planner / Researcher / Reasoner / Critic / Verifier / Aggregator; add feedback edge from Verifier or Critic; grant all tools; **stop** |
| Directives | 15 | "Think step-by-step before generating code" (Coder); "Explicitly check units (millions vs billions)" (Critic) |
| Tool grants | 15 | give Coder web search / file reader / calculator; give Verifier Python execution |

Each action changes exactly one thing, so a path through the tree *is* a readable record of design decisions.

Transitions are deterministic: $s_{t+1} = M(s_t, a_t)$. The feasible set is

$$\mathcal{A}(s) = \{a \in \mathcal{A}_d : M(s,a) \in \mathcal{S}_d,\ M(s,a) \neq s\} \cup \{a_{\mathrm{stop}}\}$$

— must keep it executable, must actually change something, plus stop. Directives and tool grants need their target agent to exist; tools must be in $\mathcal{K}_d$.

Every search starts from the same minimal seed $s_0$: **one Coder, no tools.** All structure must be earned. Horizon $H = 10$ actions. Respecting dependencies, that's roughly $1.5 \times 10^7$ distinct harnesses.

### The architect

Input (2,048 tokens, truncated from the middle): benchmark description, available tools, the query, the feasible action list, and a text serialisation of the current graph. The final-token hidden state $h_\theta(q,s)$ feeds two linear heads.

$$\pi_\theta(a \mid q,s) = \frac{\mathbf{1}[a \in \mathcal{A}(s)]\exp(\ell_{\theta,a}(q,s)/T_\pi)}{\sum_{b \in \mathcal{A}(s)}\exp(\ell_{\theta,b}(q,s)/T_\pi)}, \qquad V_\theta(q,s) = \sigma(\hat{v}_\theta(q,s))$$

Masking infeasible actions to zero probability matters — the policy only ever scores actions that can be taken. $T_\pi = 1$.

Only [[LoRA]] adapters (rank 16, scale 32, dropout 0.05, on attention Q/K/V/O) and the two heads are trained. The Gemma backbone is frozen.

### Search

Standard PUCT descent from $s_0$, $B = 64$ simulations:

$$a^\star = \arg\max_{a \in U(s)}\left[\widetilde{Q}(s,a) + c_{\mathrm{puct}}\,P(a \mid s)\frac{\sqrt{\max(1, \widetilde{N}(s))}}{1 + \widetilde{N}(s,a)}\right]$$

with $c_{\mathrm{puct}} = 2.5$. On first visit to a node, the top $K_{\mathrm{exp}} = 10$ feasible actions plus $a_{\mathrm{stop}}$ form the expanded set $U(s)$, and $\pi_\theta$ is renormalised over $U(s)$ to give the prior $P$.

Two implementation details worth keeping:

**Batched leaves with virtual loss.** Eight leaves are evaluated per architect forward pass, or the GPU idles. Without a correction all eight descents follow the same branch, so a pending leaf counts as a visit of value zero along its whole path, pushing other descents elsewhere. Tildes in the formula are these adjusted counts.

**Optimistic-but-not-too-optimistic init.** An untried action gets $\widetilde{Q}(s,a) = \mathrm{clip}(Q(s) - 0.1, 0, 1)$ — slightly *below* its parent. That keeps search on policy-favoured actions until the alternatives are measured.

The executor is never called inside this loop.

### The reward — where cost enters

This is the part that makes the method about more than accuracy. For an executed harness with task score $u \in [0,1]$:

$$R = \alpha u - b - \sum_x \gamma_x r_x, \qquad z = \frac{\mathrm{clip}(R, -1, 1) + 1}{2}$$

with $\alpha = 1.5$, $b = 0.5$. Five penalties, each in $[0,1]$:

$$r_{\mathrm{tok}} = \frac{t}{t + \rho_t}, \quad r_{\mathrm{lat}} = \frac{\ell}{\ell + \rho_\ell}, \quad r_{\mathrm{tool}} = \min\!\left(1, \tfrac{n_{\mathrm{tool}}}{6}\right), \quad r_{\mathrm{agent}} = \min\!\left(1, \tfrac{|\mathcal{V}| - 1}{7}\right)$$

plus $r_{\mathrm{timeout}} \in \{0,1\}$. Weights: 0.15 tokens, 0.10 latency, 0.10 tools, 0.08 agents, 0.15 timeout.

The reference scales $\rho_t, \rho_\ell$ are per-benchmark — 30k tokens / 25s for GSM8K, 1.5M tokens / 400s for OfficeQA. The saturating form puts wildly different cost regimes on one scale while staying sensitive past the reference.

Run the numbers: penalties sum to at most 0.58, so a success scores $\geq 0.71$ and a failure $\leq 0.25$. **Success always dominates cost; cost only breaks ties among successes.** That ordering is why the value head learns "cheapest thing that works" rather than "cheapest thing".

Note $r_{\mathrm{tool}}$ counts distinct tool *types granted*, not calls — it charges for capability, not usage. Tools essential to a benchmark (file reader on OfficeQA) are exempt.

### Training

Per training query: search from $s_0$ with Dirichlet(0.35) noise on root priors (mixing weight 0.25), sample a path in proportion to visit counts, then execute the harness at its end **plus up to $K_{\mathrm{exec}} = 4$ other distinct harnesses** from the most-visited branches.

Targets:
- **Policy**: visit shares along the sampled path, $p_i(a) = N(s_i,a)/\sum_{b \in U(s_i)} N(s_i,b)$.
- **Value**: the measured utility $z_i$, and only for states that were actually executed. Intermediate states never inherit another harness's outcome.

$$\mathcal{L}_{\mathrm{online}} = \frac{1}{|\mathcal{I}|}\Big[\sum_{i \in \mathcal{I}_\pi} w_i\big(\mathrm{CE}(p_i, \pi_i^U) - \beta\,\mathcal{H}(\pi_i)\big) + \lambda_v \sum_{i \in \mathcal{I}_V} w_i\,\mathrm{BCE}(z_i, V_i)\Big]$$

Cross-entropy uses $\pi_i^U$, renormalised over expanded actions, so actions search never looked at are not treated as negative evidence. $\beta = 0.01$, $\lambda_v = 1$. Examples come from a prioritised replay buffer (256 queries, exponents 0.6/0.4) favouring mispredicted values, with importance weights $w_i$ undoing the sampling bias.

At each epoch's end, a replay phase adds a **ranking loss** on pairs executed on the *same* query:

$$\mathcal{L}_{\mathrm{rank}} = \frac{1}{|\mathcal{J}|}\sum_{(i,j) \in \mathcal{J}} \max\{0,\ m - \mathrm{sgn}(z_i - z_j)(\hat{v}_i - \hat{v}_j)\}$$

margin $m = 0.05$, minimum utility gap $\delta = 0.05$, weight $\lambda_r = 1$, on pre-sigmoid logits. This is why they execute several harnesses per query: it manufactures same-query pairs. The value head does not need calibrated absolute utility, it needs to *rank* candidates for one query — and SHIFT-value lives entirely off that ranking.

AdamW, lr $10^{-4}$, weight decay 0.01, cosine to $10^{-5}$, grad clip 1.0, 4 epochs, minibatch 8.

### The two inference modes

- **SHIFT-search**: execute the harness at the end of the most-visited path.
- **SHIFT-value**: pull up to 5 distinct harnesses from the most-visited branches and execute the one with highest $V_\theta$. If that value is below 0.55, search again with 192 extra simulations and choose across both pools.

## Ablation Studies and Experiments

Six benchmarks, 9,193 test tasks, shared Gemini 3.5 Flash executor, three runs each, 17 baselines.

| Method | GSM8K | HotpotQA | MBPP | Sheet | OfficeQA | GAIA | Mean | Cost |
|---|---|---|---|---|---|---|---|---|
| Direct | 97.7 | 67.9 | 93.6 | 0.0 | 8.9 | 4.6 | 45.5 | 1.0 |
| ReAct | 98.2 | 69.1 | 97.1 | 84.4 | 16.3 | 33.3 | 66.4 | 12.3 |
| AgentVerse | 97.3 | 68.2 | 96.1 | 87.8 | 4.9 | 51.0 | 67.6 | 36.8 |
| DSPy-GEPA | 97.7 | 69.6 | 95.3 | 95.3 | 11.4 | 29.4 | 66.5 | 2.5 |
| GPTSwarm | 97.9 | 64.1 | 95.7 | 83.9 | 9.8 | 32.7 | 64.0 | 58.6 |
| DyLAN | 98.1 | 64.7 | 97.0 | 88.1 | 12.2 | 32.0 | 65.3 | 43.7 |
| **Trace** (best baseline) | 98.4 | 69.9 | 96.6 | 88.6 | 29.3 | 53.6 | 72.7 | 40.8 |
| **SHIFT-value** | 97.7 | 67.5 | 96.1 | 81.7 | **39.0** | **64.7** | 74.5 | 27.7 |
| **SHIFT-search** | 98.2 | 68.7 | 96.6 | 87.8 | **59.3** | **68.6** | **79.9** | 84.3 |

Cost is the geometric mean of executor tokens relative to Direct.

**Where the gains come from, and where they don't.** On GSM8K, HotpotQA and MBPP, SHIFT does not win — it ties. 16, 13 and 6 of the 17 baselines land within 2% of the leader on those three. The executor, not the harness, is the ceiling. All the headroom is on OfficeQA (+30.0 over Trace) and GAIA (+15.0), where every baseline sits below 30% and 54%.

The authors state the condition plainly: **search pays off when harness structure strongly affects success *and* no existing design has already found the right structure.** Both must hold.

Question-level, not driven by outliers: on OfficeQA, SHIFT-search beats Trace on 24 questions and loses on 4 ($p = 2\times10^{-4}$, paired sign test); on GAIA, 19 vs 7 ($p = 0.03$). Paired task bootstrap on the mean gain: 7.2%, 95% CI [4.42, 9.88]. Dropping OfficeQA entirely, SHIFT-search still leads (84.0 vs 81.4).

### The structure-sensitivity premise

The whole argument rests on "the right amount of structure varies." Training-run data, grouping executions by structure level:

| Benchmark | 1 agent | ≥4 agents | no tools | all tools | 0 directives | ≥2 directives |
|---|---|---|---|---|---|---|
| GSM8K | 98.2 | 98.4 | 98.4 | 97.8 | 98.0 | 97.9 |
| HotpotQA | 75.6 | 84.3* | 72.4 | 83.7* | 79.6 | 83.1 |
| SpreadsheetBench | 55.3 | 82.8* | 44.8 | 87.3* | 65.4 | 83.0* |
| GAIA | 40.0 | 73.9* | 35.3 | 66.9* | 50.0 | 72.2* |
| OfficeQA | 9.4 | 59.3* | 16.0 | 46.8* | 26.2 | 54.6* |

(* Fisher exact $p < 0.05$.) GSM8K: structure buys nothing and costs ~19× the tokens. OfficeQA: agents buy 50 points. And *which* axis matters differs within a benchmark — tools dominate on SpreadsheetBench, agents on OfficeQA.

Caveat the authors flag themselves: search chose which harnesses to run, so this is **observational**, not an intervention on a fixed harness.

### Action-space restriction — the joint-choice claim

500 HotpotQA questions, trained architect, actions restricted at inference, no retraining:

| Allowed | Accuracy | Δ vs full (95% CI) |
|---|---|---|
| All actions | 69.4 | — |
| Tool grants only | 67.9 | −1.5 [−3.1, −0.1] |
| Directives only | 60.3 | −9.1 [−12.0, −6.4] |

Both restricted variants keep the single Coder and cannot add agents. On a retrieval-heavy task tools recover most of it, but only the full space wins. Up to 9.1 points comes from choosing structure, instructions and tools *together*.

### Decoding ablation — what is search actually buying?

Same trained architect, five ways to decode a harness from it:

| Mode | Mean | Cost |
|---|---|---|
| SHIFT-greedy (policy argmax) | 79.8 | 120.2 |
| SHIFT-policy-veto | 79.1 | 115.0 |
| **SHIFT-value-greedy** (1-step value hill-climb) | **60.2** | **5.3** |
| SHIFT-search | 79.9 | 84.3 |
| SHIFT-value | 74.5 | 27.7 |

This is the most informative table in the paper.

**Search does not buy accuracy. It buys cost.** Greedy construction matches search at 79.8 vs 79.9 — because the policy was *trained to reproduce search's visit counts*, so it already inherits search's quality. What search adds at inference is lookahead on a utility that charges for tokens: 30% fewer executor tokens than greedy, 27% fewer than policy-veto. Verified per-question, not just in aggregate: search is cheaper on 60–98% of questions depending on benchmark; on OfficeQA tokens drop 46% while tool calls barely move, so it is not just "fewer tool calls".

**What did not work: value without lookahead.** Value-greedy is by far the cheapest (5.3× Direct) and collapses to 60.2% — nearly 20 points down. The reason is mechanical and worth remembering: an agent whose benefit only appears *after* later actions (a Verifier is useless until the Solver has tools) scores badly one step ahead. The value head is a ranker over complete-ish harnesses, not a local gradient.

### Is the value head actually a good ranker?

Pools of 4–5 candidate harnesses per question from several construction strategies, every candidate executed twice (Gemma 3 27B executor), scored by $V_\theta$ *before* any outcome is seen. 400 HotpotQA + 200 GSM8K questions.

| Selection rule | HotpotQA | GSM8K |
|---|---|---|
| Lowest $V_\theta$ | 35.6 | 86.1 |
| Uniform (in expectation) | 39.1 | 89.7 |
| Second-highest $V_\theta$ | 37.7 | 89.9 |
| **Highest $V_\theta$** | **49.6** | **92.6** |

+10.5 over uniform on HotpotQA (CI [7.5, 13.6]), +2.9 on GSM8K (CI [0.8, 5.0]) — and the chosen harness uses 9.5k tokens vs 19.5k (HotpotQA) and 1.8k vs 8.8k (GSM8K). More accurate *and* half to a fifth the cost. The top pick beats the second-ranked by 11.9 points on HotpotQA. The gap widens as $k$ grows: 39.1 → 42.7 → 46.1 → 48.9 for $k = 1..4$, while uniform is flat by construction.

**What did not work: fixed structural recipes.** Always picking the most agents, most tools, or most directives is *worse than random*. Picking fewest directives barely beats random. Picking fewest agents helps on HotpotQA and dies on OfficeQA (9.4 vs 59.3). There is no static rule; that is the paper's negative result in favour of learning.

### Transfer

GSM8K-trained architect applied to MATH-500 unchanged, Llama 4 Scout executor:

| Config | Lv 1–3 | Lv 4–5 | All | Tokens/problem |
|---|---|---|---|---|
| Fixed single Coder | 92.4 | 71.0 | 81.2 | 1,064 |
| Uniform $\pi$, constant $V$ | 92.0 | 72.9 | 82.0 | 6,283 |
| Learned $\pi$, constant $V$ | 91.6 | 74.0 | 82.4 | 7,614 |
| Uniform $\pi$, learned $V$ | 91.6 | 71.8 | 81.2 | 5,585 |
| **SHIFT (both learned)** | 92.4 | **75.6** | **83.6** | 8,049 |

Gains land only on levels 4–5 (+4.6, CI [0.4, 9.2]). Both heads contribute; learned value *alone* with a uniform policy is no better than the fixed Coder. Consistent with the decoding ablation — the value head needs a sensible proposal distribution to rank.

Also note: SHIFT uses 7.6× the tokens of a single Coder for 2.4 points. On easy math this is a bad trade, and the per-level breakdown shows exactly why.

### Training dynamics

Policy loss falls 35–52% over four epochs on every benchmark. Composition shifts measurably: by the last epoch Planners appear in most harnesses everywhere, Researchers are dropped on MBPP/SpreadsheetBench/OfficeQA, Reasoners become common on GAIA.

## Worth Remembering

**Three baselines got a zero they never earned.** CoT, Self-Consistency and Plan-and-Solve score 0.0 on SpreadsheetBench and were *not run* — the task requires writing a modified workbook to disk, so text-only methods cannot pass by construction. The zero still enters their mean, and their cost column excludes SpreadsheetBench. The table is honest about this (the ‡ marker), but it does inflate the headline gap. Removing SpreadsheetBench: SHIFT-search 78.3, SHIFT-value 73.0, Trace 69.6 — ranking holds.

**The training cost is real and per-benchmark.** Four epochs, every training question searched once, up to five harnesses executed per tree. That's 1,092 executions on OfficeQA up to ~9,400 on HotpotQA. In executor tokens, training one architect ≈ answering 440 (OfficeQA) to 6,200 (HotpotQA) test questions with SHIFT-search. Amortised over a deployment, fine. For a one-off job, you lose. The authors name this as the main limitation: **one architect per benchmark and per executor.**

**Architect overhead is genuinely negligible.** 0.73–0.90s median per harness at 64 simulations on one H100 NVL; 10–14ms per simulation, so cost scales linearly (2.6–3.6s at 256). Against one to eight million executor tokens, this is free. The whole method is an arbitrage on that ratio — and it only exists because the predictor is small.

**The reward design is the transferable trick.** Setting $\alpha = 1.5$, $b = 0.5$ and capping penalties at 0.58 guarantees success $\geq 0.71 >$ failure $\leq 0.25$. This is a lexicographic ordering enforced through a scalar. If you are building any cost-aware utility for policy selection, copy this: check the penalty sum can never flip the ordering of the primary objective. Otherwise you get a [[Reward Hacking|reward-hacked]] optimiser that prefers cheap failures.

**SHIFT-value is the practically interesting variant.** It beats every baseline in mean accuracy at 27.7× Direct vs Trace's 40.8× — and the per-query plots show why: one or two agents on GSM8K and MBPP, two or three on HotpotQA, four-plus on SpreadsheetBench and GAIA. On OfficeQA its choices are bimodal (one agent or three-four), which means the cost distribution is bimodal too. That's correct behaviour, not instability.

**Connection to [[Off-Policy Evaluation]] and [[Bayesian Optimization]].** The value head is a learned surrogate for an expensive measured outcome, used to rank candidates offline before committing to one execution. That is the same shape as offline policy selection — and the same honest difficulty: $V_\theta$ is trained only on harnesses that search chose to execute, so it is evaluated on its own behaviour policy's support. The structure-sensitivity table has the identical confound, which the authors flag. **Open question:** how badly does $V_\theta$ degrade on harnesses far from the training distribution? The MATH-500 transfer result is mildly reassuring; the uniform-$\pi$-learned-$V$ row (81.2, no gain) is mildly alarming.

**Follow-up questions.**
- The value target comes from a *single* execution per harness. On stochastic executors the label is a coin flip, not a utility. The authors explicitly list "value estimation using outcomes averaged over executions" as future work.
- Policy targets are visit counts from search by the *current* architect — a moving target, like [[DQN|DQN]]'s. Falling loss means the policy anticipates its own search better, which is not the same as the search being better.
- Nothing is routed across executors. A harness tuned for Gemini 3.5 Flash is not a harness tuned for a weaker model, and the Gemma 3 27B / Llama 4 Scout experiments use different architects or different tasks.

**Practical caveat for anyone adopting this.** The 40-action library, the role list, the directive strings and the per-benchmark reference scales $\rho_t, \rho_\ell$ are all hand-written. SHIFT learns to *compose* a human-designed library, it does not invent components. Your gains will be bounded by whether the right component is in your library — which is a different and less tractable problem.

**Comparison worth keeping straight.** Prompt optimisers ([[LoRA|—]] no, see: DSPy/MIPROv2/GEPA) tune instructions inside a fixed program. Workflow optimisers (AFlow, GPTSwarm, Trace) find one reusable structure offline. Routers pick among predefined models or workflows. AgentSquare uses an LLM performance predictor but still executes newly evolved modules. SHIFT is the only one doing per-query construction with explicit lookahead over all three component types, evaluated by a *locally trained* predictor.

## Links

Related: [[Mastering Chess and Shogi by Self-Play (AlphaZero)|AlphaZero]] · [[Monte Carlo Tree Search]] · [[Agentic Workflows]] · [[Multi-Agent LLM Systems]] · [[Agent Frameworks]] · [[DSPy]] · [[Planning and Decomposition]] · [[Tool Use]] · [[Test-Time Compute]] · [[Value Function]] · [[Policy]] · [[LoRA]] · [[Experience Replay]] · [[Reward Function]] · [[Reward Hacking]] · [[Off-Policy Evaluation]] · [[Exploration vs Exploitation]] · [[Upper Confidence Bound]] · [[Cost and Latency]] · [[An Empirical Study of Harness Design for Coding Agents]] · [[WHALE- A Simple Recipe for Joint Harness-Weight Optimization]] · [[RRSI- Regularized Recursive Self-Improvement of Agent Harnesses]] · [[Agent Evaluation]] · [[Evals]]

New topics worth writing: PUCT and virtual loss in parallel MCTS, learned surrogates for expensive evaluation, prompt optimisation (DSPy / MIPROv2 / GEPA), LLM routing and model selection, AgentSquare and modular agent search, agentic supernets (MaAS), lexicographic reward construction, amortised inference, OfficeQA Pro, GAIA, SpreadsheetBench
