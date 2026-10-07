---
title: "RRSI: Regularized Recursive Self-Improvement of Agent Harnesses"
authors: ["Peng Xia", "Rujun Han", "Zifeng Wang", "Yanfei Chen", "Yufan Zhuang", "Yoonho Lee", "Chengsong Huang", "Han Yu", "Zhongying CuiZhu", "Yifei Ming", "Huaxiu Yao", "Burak Gokturk", "Tomas Pfister", "Chen-Yu Lee"]
year: 2026
arxiv: "2609.24972"
url: https://arxiv.org/abs/2609.24972
priority: Good-To-Read
read_on: 2026-09-27
tags: [paper, transformers, llm]
---
## The Core Idea

An LLM agent is two things: the frozen model weights, and the **harness** around them — the system prompt, the loop that decides when to plan or act or stop, the tool descriptions, the memory files, the context-compaction rules. Most recent agent progress came from hand-editing the harness, not from new weights.

The obvious next step is to automate that: let an LLM read failed trajectories, propose harness edits, keep the edits that raise a benchmark score, repeat. That is recursive self-improvement at the system level — the agent rewrites its own scaffolding.

The problem this paper names: **that loop overfits, and it overfits in a way that looks like success.**

The score you optimise comes from a finite "evolve set" of tasks, measured with noisy stochastic rollouts, and you look at it again and again across rounds. This is adaptive data analysis, the same trap as tuning on your test set. The search happily finds edits that raise the evolve score without installing any reusable mechanism.

The paper breaks the failure into three named behaviours:

- **Benchmark-specific fitting** — the edit hardcodes task names, entity names, or answers into the prompt.
- **Noise chasing** — a candidate wins because its rollouts got lucky, and that lucky win becomes permanent harness state forever after.
- **Complexity accumulation** — edits pile on more prompt text, more tools, more retries. The score creeps up because you are spending more test-time compute, not because the agent got smarter.

The numbers make this concrete. Unregularized evolution scores **92.8** on the split it evolves against — the best of any method tested — and **40.3** out of distribution, which is within one point of the un-evolved starting harness at 39.7. It also burns **3.80 million** policy tokens per trial against the base harness's 1.56 million. All that apparent self-improvement bought nothing transferable and tripled the bill.

> [!NOTE] Harness
> Everything wrapped around a frozen LLM: prompts, control flow, tool interfaces, memory, skill files, context management. An agent is $A = (\pi, H)$ with policy $\pi$ frozen and harness $H$ as the only optimisation variable. ^harness-def

> [!NOTE] Adaptive overfitting in harness evolution
> The candidates proposed at round $t$ depend on measurements taken from the same finite task set in rounds $0 \dots t-1$. Each round is another look at the same data, so the evolve-set score stops being an unbiased estimate of anything. ^adaptive-overfitting

**The fix — and this is the actual contribution — is to leave the edit space completely open and regularize the *search trajectory* through it instead.** Any harness component may still be added, rewritten or deleted. What gets constrained is (a) how many things one round may change, (b) where the search spends its attention, and (c) which measured improvements are allowed to become permanent.

The framing is deliberately borrowed from classical regularisation, and the mapping is by *role*, not by mathematics:

| Classical | RRSI mechanism | What it limits |
|---|---|---|
| $L_0$ cardinality | annealed edit budget | how many edits one candidate may bundle |
| Lasso / $L_1$ | structural pruning | components that never earned their keep get deleted |
| Ridge / $L_2$ | cost-aware acceptance | growth in total token footprint |
| entropy bonus | structured exploration | collapse onto one narrow edit family |

RRSI posts the **smallest** evolve-set gain of any evolved harness (+1.1 on Harvey LAB, where unregularized evolution gets +3.4) and the **only** out-of-distribution average that clears the base harness by more than a point: 43.6 vs 39.7. That trade — give up evolve-set score, keep transfer — is the whole design.

## The Methodology

### The loop being fixed

Standard harness evolution, at round $t$:

$$\mathcal{H}_t = \{H_t^{(1)},\dots,H_t^{(m_t)}\} \sim P_0(\cdot \mid H_t, \mathcal{F}_t), \qquad H_{t+1} = \arg\max_{H' \in \mathcal{H}_t \cup \{H_t\}} \hat{S}(H'; \mathcal{D}_{\text{evolve}})$$

Run the current harness on the evolve set, summarise the failures into feedback $\mathcal{F}_t$, have a proposer LLM write candidate harnesses, score them all, take the best. The empirical score over $k$ trials per task is

$$\hat{S}(H) = \frac{1}{k|\mathcal{D}_{\text{evolve}}|}\sum_{x}\sum_{j=1}^{k} r(x, \tau_x^{(j)}), \qquad \hat{C}(H) = \frac{1}{k|\mathcal{D}_{\text{evolve}}|}\sum_{x}\sum_{j=1}^{k} c(\tau_x^{(j)})$$

where $c(\tau)$ is policy tokens consumed. RRSI keeps this exact shape and inserts constraints on both halves:

$$\mathcal{H}_t \sim P_{\text{reg}}(\cdot \mid H_t, \mathcal{F}_t, \mathcal{L}_t, b_t, \mathcal{E}_t, \mathcal{B}_t), \qquad H_{t+1} = \arg\max_{H' \in \mathcal{H}_t \cap \mathcal{A}_t} \hat{S}(H')$$

with $H_{t+1} = H_t$ when nothing is admissible. The new symbols: $\mathcal{L}_t$ is the edit history, $b_t$ the edit budget, $\mathcal{E}_t$ exploration directives, $\mathcal{B}_t$ pruning targets, $\mathcal{A}_t$ the admissible set.

### Proposal side, piece 1 — the annealed edit budget

A candidate applies a subset of a drafted edit pool, written as an indicator vector $z_t \in \{0,1\}^{|E_t|}$. The constraint is a hard cap on how many are switched on:

$$\|z_t\|_0 \le b_t, \qquad b_t = \left\lceil b_{\min} + (b_{\max}-b_{\min})\cdot\tfrac{1}{2}\big(1 + \cos(\pi t/T)\big)\right\rceil$$

A cosine schedule from $b_{\max}$ down to $b_{\min} = 1$. Coding uses $b_{\max}=4$ over $T=20$ rounds; agentic workspace $b_{\max}=3$; engineering design $b_{\max}=4$ over $T=40$.

Two reasons this matters. A candidate bundling eight unrelated changes has high effective capacity — it can fit more quirks of the current feedback. And when it does help, you cannot tell *which* change helped. Early rounds are allowed to bundle so coordinated mechanisms can be discovered; late rounds are one edit at a time, so every measurement is attributable.

The analogy is to update sparsity, not parameter sparsity — the edit pool is redrawn each round, and nothing optimises an $L_0$-penalised objective.

### Proposal side, piece 2 — evidence-aware credit assignment

Every atomic edit in every evaluated candidate is logged:

$$\mathcal{L}_t = \{(t_i, \ell_i, h_i, d_i, \Delta S_i, \Delta C_i, a_i) : i \le n_t\}$$

component touched, hypothesis tested, source diff, measured score change, measured cost change, and $a_i = 1$ only if that candidate actually won its round. Bundled edits all inherit their candidate's single measurement — which is precisely why attribution sharpens as $b_t$ anneals to 1.

The proposer conditions on this whole history. Rejected mechanisms stay on the record as negative evidence, so the search does not re-test hypotheses it already falsified. Every re-test is another adaptive look at the same finite set for no new information.

### Proposal side, piece 3 — structured exploration

The editable component vocabulary is fixed and explicit:

$$\mathcal{K} = \{\texttt{prompt}, \texttt{control\_flow}, \texttt{config}, \texttt{output\_plumbing}, \texttt{context\_mgmt}, \texttt{client\_tool}, \texttt{skill}, \texttt{memory}, \texttt{subagent}\}$$

A stall flag fires when progress over the last $w=3$ rounds sits inside the noise band:

$$\sigma_t = \mathbb{1}[\hat{S}_t - \hat{S}_{t-w} \le \delta], \qquad \mathcal{U}_t = \mathcal{K} \setminus \mathcal{T}_t$$

$\mathcal{T}_t$ is the set of components already touched by some measured edit, so $\mathcal{U}_t$ is the untouched ones. During a stall, $m_{\text{draft}} = 1$ candidate slot is reserved for $\mathcal{U}_t$. This is the diagnosis for a proposer that has collapsed onto rewriting prompts forever while never touching control flow — a role like the entropy bonus in [[Soft Actor-Critic]].

### Selection side, piece 1 — leakage screening, before evaluation

A critic LLM reads each candidate diff and rejects anything encoding task names, entity names, task-specific constants, answers, or evolve-benchmark-specific branching. It also kills inert machinery — code added that never runs.

The ordering is the point. **Screen before you score.** A leaking candidate that is allowed to run gets an inflated evolve-set score, and that inflated score then sits in $\mathcal{L}_t$ as positive evidence pulling later rounds towards more of the same. Generic prompt or tool-description improvements survive the screen; only benchmark-specific content is targeted.

### Selection side, piece 2 — the noise-adjusted floor

Before evolution starts, the unchanged base harness is run repeatedly to estimate an empirical noise band $\delta$. Then with $S^\star$ the best evolve score seen so far, every candidate must clear

$$\hat{S}(H') \ge S^\star - \delta$$

This blocks the slow downhill walk: a chain of regressions each individually small enough to pass as noise, summing to a worse harness. In real units $\delta$ is tiny — 3 passes out of 178 trials on coding, 60 criteria out of ~14,100 on Harvey LAB, 5 out of 244 on engineering design.

### Selection side, piece 3 — cost-aware acceptance

Define, relative to the incumbent:

$$\Delta S = \hat{S}(H') - \hat{S}(H_t), \qquad \Delta C = \frac{\hat{C}(H') - \hat{C}(H_t)}{\hat{C}(H_t)}$$

When the gain genuinely clears the noise band, $\Delta S > \delta$, the candidate must satisfy

$$\Delta C \le \beta_0 + \beta_1 \Delta S$$

$\beta_0$ is the cost rise tolerated for a negligible gain (0.10 on coding and agentic, 0.15 on engineering). $\beta_1$ is how much extra cost a bigger gain buys — 44.5 on coding, which in readable units is a 25% token allowance per additional task passed. Both are chosen on the evolve set and frozen for all transfer runs.

**The branch that does not appear in the main text.** When $\Delta S \le \delta$ — a measured change inside the noise — equation (7) is *not* used. Instead:

$$w_s \Delta S - w_c \Delta C + w_n \nu_t(H') > 0$$

with $\nu_t(H')$ counting structural component types ($\texttt{client\_tool}$, $\texttt{skill}$, $\texttt{memory}$, $\texttt{subagent}$) that the candidate touches and that have *never* appeared in a winning edit before. Inside the uncertainty band, a score bump earns you nothing on its own; you must earn credit by cutting cost or by trying genuinely new structure. The coding instance sets $w_s = 0$, so a within-noise score rise is worth literally zero there.

### Selection side, piece 4 — structural pruning

Track the best recent measured gain per component over a pruning window $n_{\text{prune}}$ (4–5 rounds):

$$g_t(\ell) = \max\{\Delta S_i : \ell_i = \ell,\ t - t_i \le n_{\text{prune}}\}, \qquad \mathcal{B}_t = \{\ell \in \mathcal{T}_t : g_t(\ell) \le 0\}$$

Components in $\mathcal{B}_t$ have been exercised and produced nothing positive. They are handed to the proposer as **deletion targets**. The budget sparsifies each *update*; pruning sparsifies the *retained harness*. A plain score-maximising search has no incentive whatsoever to remove anything, which is exactly how complexity accumulates.

### Domain guards

Engineering design adds two non-compensatory checks: reject if valid-output rate drops more than 0.03, or no-submission rate rises more than 0.02. A pass-rate gain cannot be paid for with broken execution. Coding and agentic workspace use no extra guard.

### Setup

Policy frozen at Claude Opus 4.8 throughout. Proposer, failure analyst and leakage critic are also Claude Opus 4.8. Base harnesses: Terminus-2 for coding; a [[Agentic Workflows|ReAct]] loop over an [[Model Context Protocol|MCP]] tool gateway with a dynamic toolbelt and ReSum-style context management for the other two. $k=2$ trials per task (4 on engineering design).

## Ablation Studies and Experiments

### Eight benchmarks, three domains, evolve on one and run unchanged on the rest

- **Coding** — evolve on Terminal-Bench 2.1 (89 containerised shell tasks, graded by hidden unit tests). Transfer to SWE-bench Verified.
- **Agentic workspace** — evolve on 120 tasks of Harvey LAB (legal work, 25 practice areas, per-criterion rubric, ~14,000 criterion verdicts per full eval). Transfer to a pristine 40-task in-distribution held-out split, plus JobBench, GDPval and APEX-Agents out of distribution.
- **Engineering design** — evolve on 61 EngDesign tasks. Transfer to Frontier-Eng (47 tasks, Medal Score).

### Headline numbers

| Split | Result |
|---|---|
| Terminal-Bench 2.1 (evolve) | 74.2 → 80.2, **+6.0** |
| SWE-bench Verified (OOD) | 82.0 → 83.8, **+1.8** |
| EngDesign (evolve) | **+4.9** |
| Frontier-Eng (OOD) | **+4.3** Medal, +24.3% relative |
| Harvey LAB (evolve) | 89.4 → 90.5, **+1.1** |
| Harvey LAB (ID held-out) | 86.9 → 89.2, **+2.3** |
| JobBench / GDPval / APEX-Agents | **+3.5 to +4.7**, 7.2%–13.1% relative |

**No held-out split regresses anywhere.** That is the specific failure a memorising harness produces, and it does not appear.

### The comparison that carries the paper

Four prior methods, same $H_0$, same policy, same evolve split, same candidate budget (Harvey LAB):

| Method | Evolve | ID held-out | JobBench | GDPval | APEX |
|---|---|---|---|---|---|
| $H_0$ (no evolution) | 89.4 | 86.9 | 36.0 | 48.8 | 34.2 |
| Meta-Harness | **93.0** | 89.2 | 37.1 | 49.1 | 35.7 |
| AHE | 90.7 | 88.7 | 37.2 | 47.2 | 33.1 |
| TTHE | 91.1 | 88.5 | 35.2 | 47.0 | 31.7 |
| HarnessX | 91.8 | 89.1 | 36.3 | 48.5 | 34.3 |
| **RRSI** | 90.5 | **89.2** | **40.7** | **52.3** | **37.9** |

Read the first column, then the last three. **The ranking inverts.** Meta-Harness wins the evolve split by 2.5 points over RRSI and adds 0.9 to the OOD average. HarnessX lands on the base harness. AHE and TTHE finish *below* the harness they started from — TTHE by 1.7 points, meaning its evolution was actively destructive once the benchmark changed. RRSI has the smallest evolve gain of any evolved method and the only OOD average above 40 (43.6 vs 39.7 for $H_0$).

In-distribution held-out barely separates anything (88.5–89.2 for every method). The gap only opens when the task descriptions, tool interfaces and verifiers change.

### The ablation

| Variant | Evolve | ID held-out | OOD avg | Tokens/trial (M) |
|---|---|---|---|---|
| $H_0$ | 89.4 | 86.9 | 39.7 | 1.56 |
| Unregularized evolution | **92.8** | 88.9 | 40.3 | 3.80 |
| w/o proposal regularizers | 90.7 | 88.8 | 41.9 | 2.69 |
| w/o acceptance regularizers | 91.5 | 88.7 | 41.0 | 3.59 |
| **RRSI** | 90.5 | **89.2** | **43.6** | 2.42 |

Every row is the same inverse relationship: **more evolve score, less transfer.**

Dropping acceptance constraints costs 2.6 OOD points and adds half again as many tokens (2.42 → 3.59M) while gaining 1.0 on evolve. The selection rule is where the token bloat is stopped.

Dropping proposal constraints costs only 0.2 on evolve but **1.7 out of distribution**. This is the more interesting half: even when nothing bad is *accepted*, steering where the search *looks* changes what it finds. The budget, the history and the exploration reserve do real work on their own.

Dropping both is worst: highest evolve score of any arm (92.8), OOD average within a point of no evolution at all, at 3.80M tokens.

### Policy robustness

Evolution re-run independently with a different backbone family:

| Policy | Terminal-Bench (evolve) | SWE-bench Verified (OOD) |
|---|---|---|
| Claude Opus 4.8 | 74.2 → 80.2 (**+6.0**) | 82.0 → 83.8 (**+1.8**) |
| Gemini 3.5 Flash | 64.6 → 78.7 (**+14.1**) | 76.8 → 79.0 (**+2.2**) |

The weaker policy gains far more on the evolve set — it starts further from the ceiling. Both transfer.

### Cross-model transfer — the strongest single result

Take the final harness evolved with Gemini 3.5 Flash. Run it **unchanged** on Gemini 3.1 Flash Lite, a much weaker model that never appeared in the search.

| Evaluation policy | $H_0$ | RRSI | $\Delta$ |
|---|---|---|---|
| Gemini 3.5 Flash (search policy) | 64.6 | 78.7 | +14.1 |
| Gemini 3.1 Flash Lite (unseen) | 11.2 | 14.6 | **+3.4** |

+3.4 points from a base of 11.2 is a 30.4% relative gain. A harness is a program, not weights, so a mechanism that only helps the policy it was searched against is an artefact of that policy. This says the mechanisms are not tied to a capability level.

### Cost

RRSI: **2.42M** policy tokens per trial, **26.3** steps. Prior methods: 27.3–34.6 steps, up to **3.82M** tokens (AHE — 58% more than RRSI, for 4.4 fewer OOD points). All four baselines sit in the region RRSI strictly dominates: more tokens, worse transfer.

But $H_0$ is still the cheapest at **1.56M** tokens and 21.2 steps. **Evolution does buy part of its gain with test-time compute.** No evolved harness escapes that. The budget only decides how much you pay.

### Judge-mediation is not the explanation

Harvey LAB, JobBench and GDPval are all graded by LLM judges, so a harness could in principle learn to write the way a judge likes rather than to work better. EngDesign and Frontier-Eng close that route — each task is graded by its own frozen simulator or testbench, deterministically. The gains survive there unchanged (+4.9 evolve, +4.3 Medal transfer). That also removes judge variance from the measurement entirely.

### What did not work — from the case study

Four logged decisions, which show the rules biting:

- **Coding R0-A, accepted.** Bounded pre-completion verification audit plus non-blocking polling guidance for long jobs. +3.93 points. A broad early-round bundle survives *because* the gain clearly clears $\delta$.
- **Coding R0-B, rejected by the cost rule.** Nearly the same idea — verification reminder plus long-running-work guidance — but +1.69 points for +26.1% cost. Inside the noise band with a big bill. Superficially identical to the accepted one; killed on the arithmetic.
- **Coding R8-B, rejected by the floor.** Pins the original task instruction into the completion gate so the policy re-reads the literal spec before submitting. Sounds sensible. −2.81 points, and **−13.6% cost could not save it**. Cheaper does not buy below-floor performance.
- **Engineering R2, accepted.** A bounded recovery hint for the recurring `workdir must be an existing directory` tool error. 122/244 → 128/244 passes for +1.6% tokens. Small, task-agnostic, cheap — exactly the shape the regularisers are hunting for.

## Worth Remembering

**The central lesson generalises past harnesses.** Any time you propose and select against the same finite noisy set repeatedly, the score you are maximising stops estimating what you want. This is [[Reward Hacking#The bounty|Goodhart]] with the optimiser being an LLM writing code, and it is the same disease as tuning on a test set — just with far more expressive edits available. The reference for the theory is Dwork et al. on generalisation in adaptive data analysis.

**The diagnostic to steal.** If your self-improving system posts big gains on the split it is scored on and near-zero on anything else, you have measured the search's ability to fit that split, not a capability. Report both columns or the number means nothing.

**Calibrate $\delta$ before you start.** Running the unchanged base harness repeatedly to estimate the noise band is cheap and is the load-bearing prerequisite for everything on the selection side. Without it you cannot tell a real gain from a lucky rollout, and lucky rollouts become permanent harness state.

**Screening order is not cosmetic.** Reject leaking candidates *before* evaluation. Once a leaky candidate has an inflated score in the history, the proposer treats that score as evidence and follows it.

**Pruning has no natural counterpart in score-only search.** Nothing in $\arg\max \hat{S}$ ever wants to delete a component. If you build an evolution loop, the delete path has to be engineered in deliberately or your harness will monotonically grow.

**The token caveat is honest and important.** RRSI is the lightest *evolved* harness but still runs 55% more tokens per trial than the un-evolved baseline (2.42M vs 1.56M). Part of every evolution gain is bought with [[Test-Time Compute|test-time compute]]. The $\beta_0, \beta_1$ rule bounds the purchase; it does not eliminate it. If you cared purely about cost-per-task, $H_0$ wins.

**Hyperparameter honesty.** $\delta$, $(b_{\min}, b_{\max})$, $w$, $m_{\text{draft}}$, $n_{\text{prune}}$, $(\beta_0, \beta_1)$, $(w_s, w_c, w_n)$ are all chosen on the evolve environment only, never on held-out or OOD. Good discipline — but that is nine knobs, and the authors concede effectiveness may depend on feedback quality and search budget.

**The within-band rule is where the real design taste lives**, and it is buried in Appendix C.3. Setting $w_s = 0$ for coding means a within-noise score improvement is worth *nothing* — a candidate must earn admission by reducing cost or by touching structural machinery never tried before. Most descriptions of this paper will skip that. It is arguably the sharpest single anti-noise-chasing decision in the whole method.

**Limitations the authors state.** Backbone weights stay frozen, so nothing here speaks to co-evolving the harness and the model. Still needs a finite evolve set. Validation is across three domains and a handful of backbones, not across substantially different agent architectures or tool ecosystems, and not across genuinely long-running self-improvement.

**Open questions worth chasing.** The concurrent work is the interesting part of the related-work section: HarnessCompass targets generalisation as an *explicit search objective* rather than as a constraint on the trajectory, and Luo et al. replace greedy selection with a diversity-preserving archive. Both are orthogonal to RRSI and could compose with it. Also: the noise-adjusted floor and the cost rule are hand-set thresholds — the natural upgrade is to treat candidate selection as an [[Off-Policy Evaluation|off-policy evaluation]] problem, or as a [[Multi-Armed Bandit|bandit]] over candidate harnesses with proper confidence bounds instead of a single $\delta$.

**Connections into the vault.** This is [[Regularization]] applied to a search over programs rather than over weights, and the $L_0$/$L_1$/$L_2$ mapping is by role only — the authors are careful to say so, and are right to be. The overfitting story is the same shape as [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|weak-baseline overfitting in recsys]] and [[Do ImageNet Classifiers Generalize to ImageNet|ImageNet test-set reuse]]. The exploration reserve plays the role that entropy regularisation plays in [[Soft Actor-Critic]]. The attribution problem — which of eight bundled edits caused the gain — is [[Credit Assignment]] with a very small, very expensive sample.

## Links

Related: [[Regularization]] · [[Reward Hacking]] · [[Agentic Workflows]] · [[Evals]] · [[Test-Time Compute]] · [[Credit Assignment]] · [[Exploration vs Exploitation]] · [[Multi-Armed Bandit]] · [[Off-Policy Evaluation]] · [[Do ImageNet Classifiers Generalize to ImageNet]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[On the Difficulty of Evaluating Baselines]] · [[Soft Actor-Critic]] · [[Model Context Protocol]] · [[Context Engineering]] · [[Recursive self-improvement of AI research agents]] · [[The Last AI Built by Humans- Toward Genuine Recursive Self-Improvement]] · [[WHALE- A Simple Recipe for Joint Harness-Weight Optimization]] · [[An Empirical Study of Harness Design for Coding Agents]] · [[Troubling Trends in Machine Learning Scholarship]]

New topics worth writing: Adaptive data analysis and holdout reuse (Dwork et al.), Harness evolution, Recursive self-improvement at the system level, Terminal-Bench, SWE-bench Verified, LLM-as-judge rubric grading, Noise-adjusted acceptance thresholds, Cost-aware model selection, $L_0$ regularisation and learned sparsity, Cross-model harness transfer, HarnessCompass, Quality-diversity archives for program search
