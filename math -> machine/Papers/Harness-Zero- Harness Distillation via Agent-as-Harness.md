---
title: "Harness-Zero: Harness Distillation via Agent-as-Harness"
authors: ["Ye et al."]
year: 2026
arxiv: "2609.24974"
url: https://arxiv.org/abs/2609.24974
priority: Good-To-Read
read_on: 2026-09-27
tags: [paper, llm, vision]
---
## The Core Idea

A modern LLM agent is two things: the **model weights**, and the **harness** — the code wrapped around the model that gives it tools, trims its context, blocks dangerous actions, and injects skill documents and notes. Harness engineering works. It also does not travel. If you tune a harness for spreadsheet work, those gains only exist while that harness is bolted on. Since the best harness differs per domain, per task and per model, a general agent either accepts one mediocre shared harness or maintains a zoo of specialised ones plus a router.

> [!NOTE] Agent harness distillation ^harness-distillation
> Use an optimised harness only at **training time**, as a source of supervision, and push the behaviours it induces into the model's parameters — so at deployment you can delete the harness and keep the gains under one fixed, minimal harness.

The obstacle is not the fine-tuning. It is that the optimised harness $h^\star$ and the deployment harness $h$ speak different languages. $h^\star$ might hand the model a `smiles_check` tool; $h$ has only a single `execute` bash call. A trajectory recorded under $h^\star$ is full of tool calls the student will never be able to make. Fine-tuning on it teaches the model to hallucinate a tool that does not exist — the paper measures this directly, and it makes things **worse** (12% → 3%).

The trick: **agent-as-harness**. Instead of running the student inside the specialised harness, run the student inside the *minimal* harness, and put a second agent — the *harnessing agent* — at the student's mouth. Before each student response is executed, the harnessing agent reads it, consults a private, adapted copy of $h^\star$, and either passes it through or rewrites it into the smallest correction that is **still a legal action under $h$**. Every accepted response is therefore something the student could have said. Fine-tune on those, and the harness behaviour goes into the weights.

Two results make it interesting:

1. Even with **no training at all**, this beats running the specialised harness directly, on frontier models (81.1% vs 78.1% average over six benchmark×model settings).
2. After distillation, with $h^\star$ removed, the 9B student goes from **23.3% → 44.3%** macro-average task success — beating the 41.7% it gets with $h^\star$ still attached.

The last point is the one to remember. The distilled model under a bare harness is better than the base model with the good harness. The scaffold was a crutch; the corrections were the lesson.

If you already know [[Imitation Learning]], this is DAgger in agent clothing: an expert relabels actions **on the student's own state distribution**, which is exactly the fix for [[Imitation Learning#^errors-compound|compounding error]]. The novelty is where the expert's knowledge comes from — not a human, not a bigger model, but a piece of optimised scaffolding code, translated on the fly.

## The Methodology

Three symbols carry the whole paper:

- $h$ — the **target harness**. Fixed, minimal, mini-SWE-agent style. One system prompt, one tool: `execute`, which runs one bash command in a fresh shell (so state only persists through the filesystem). No tool call in a response = final answer, trial over. A `agent "<task>"` CLI allows one level of subagent delegation, so bash is expressive enough to emulate most harness features.
- $h^\star$ — the **evolved student-side harness**. Built on the DeepAgents abstraction: tools, middleware, skills, memory.
- $\mathcal{K}$ — the **private reference harness**. $h^\star$ rewritten for a reviewer instead of a doer.

### Stage 1 — evolve, then adapt

$$h^\star = \operatorname{Evolve}(\mathcal{D}_{\mathrm{train}}), \qquad \mathcal{K} = \operatorname{Adapt}(h^\star; h)$$

Evolution is a three-round loop. Run the bare student on the training split. A swarm of up to 10 analysis subagents reads the failing trajectories, classifies failure modes, and proposes *reusable* changes. A main agent consolidates them into the shared harness. Re-run on only the still-failing tasks. Solved tasks leave the pool. A hard rule enforced in the evolution skill: anything task-specific (a cell address, a filename, a one-off step) is forbidden; if a failure is not shared by a class of tasks, you abandon the task. This is the anti-overfitting guard, and it matters because the evolved harness is later the only source of supervision.

The adaptation changes the **audience and enforcement point** of each component while preserving its intent:

| In $h^\star$ | In $\mathcal{K}$ |
|---|---|
| **Tool** — a utility in the student's action space | **Action recipe** — instructions for writing an equivalent bash/Python command the student *can* run |
| **Middleware** — inspects state, blocks the loop | **Review middleware** — detects the same condition from the visible trajectory, privately warns the reviewer |
| **Skill** — "do it this way" | **Review guidance** — "spot the missing step, and how to insert it" |
| **Memory** — recorded past failures | **Failure patterns** — each failure becomes a review condition + targeted intervention |

The worked example in the paper is the clearest illustration. SpreadsheetBench tasks often pre-fill a few example cells inside the answer range — those cells *are* the spec, and overwriting them is a common failure. In $h^\star$, a middleware reaches into the sandbox at finish time, diffs input vs output workbooks over the answer range, and rejects the finish. The harnessing agent **cannot** do that, deliberately: it has no private channel into the student's environment. So $\mathcal{K}$ instead (a) checks whether the *visible* trajectory contains evidence of such a comparison, and (b) if not, supplies the recipe for writing a `prefilled_diff.py` script as the replacement response. The check then runs inside the student's own trajectory, and its output becomes a student-visible observation — so it is learnable.

### Stage 2 — agent-as-harness rollout

Baseline (**code-as-harness**) is just:
$$y_t \sim \pi_{\theta_0}^{h^\star}(\cdot \mid c_t), \qquad c_{t+1} = T_{h^\star}(c_t, y_t)$$

**Agent-as-harness** instead:
$$
\begin{aligned}
y_t &\sim \pi_S(\cdot \mid c_t) := \pi_{\theta_0}^{h}(\cdot \mid c_t) \\
(d_t, z_t) &\sim \pi_H(\cdot \mid c_t, y_t, \mathcal{K}, r_{<t}) \\
\tilde{y}_t &= \begin{cases} y_t & d_t = \textsc{pass} \\ z_t & d_t = \textsc{replace}\end{cases} \\
c_{t+1} &= T_h(c_t, \tilde{y}_t)
\end{aligned}
$$

$c_t$ is the student's visible context. $r_{<t}$ is the harnessing agent's **private** history. Only $\tilde{y}_t$ and its resulting observation enter the student trajectory; the rejected proposal and the review discussion never do.

The constraints on the harnessing agent are the engineering substance:

- **No privileged access** beyond reading $\mathcal{K}$. No hidden answers, no verifier feedback, no peeking into the sandbox. Any new evidence must be obtained by proposing a legal action under $h$.
- **Smallest coherent correction.** Pass sound proposals. Pass harmless inefficiency, stylistic differences, and *recoverable* mistakes — self-recovery is itself valuable training behaviour.
- **First-person voice.** The replacement's reasoning must read as the student's own inner monologue catching its own mistake, never as advice addressed to the student.
- **Replace budget.** 5 replacements per trial on USPTO; 5/3/1 tried on SpreadsheetBench and AppWorld. Stated in the prompt so the reviewer can allocate.
- One session spans one full trial; the reviewer sees incremental updates and keeps context. Exactly one `submit_review` call per update, with `decision`, `replacement`, `components_used`, `reason`. Malformed submissions get re-prompted up to 3 times, then the trial fails.

### Stage 3 — SFT

Ordinary next-token [[Cross Entropy]] over accepted responses only:
$$\hat{\theta} = \arg\min_{\theta} - \sum_{\tau \in \mathcal{D}_{\mathrm{review}}} \sum_{t=1}^{T_\tau} \log \pi_\theta^h(\tilde{y}_t \mid c_t)$$

Two hygiene steps that are easy to skip and would leak:

- **Structural/privacy filter.** Drop any trajectory containing an absolute `/components/` path, the candidate-file path, the review tool name, the `components_used` metadata key, or the review middleware's internal name.
- **Reasoning mask.** A replacement is written inside the review context, so its reasoning sometimes slips into reviewer voice ("the proposal is missing…"). Those token spans are masked from the loss, matched by patterns on words like *proposal*, *draft*, *the student's command*, and review verbs.

Retained data: 487 trajectories (SpreadsheetBench), 282 (AppWorld), 500 (USPTO). [[LoRA]] SFT on Qwen3.5-9B, rank 32, $\alpha=32$, batch 8, 65,536-token sequences, 2 epochs, linear schedule with 5% warmup to $2\times10^{-4}$ then decay to $10^{-6}$, seed 42. One model per domain. Harnessing agent = GPT-5.6 Sol. Harness evolution run by Kimi K3.

At deployment: $y_t \sim \pi_{\hat\theta}^h(\cdot\mid c_t)$. No $h^\star$, no $\mathcal{K}$, no reviewer.

## Ablation Studies and Experiments

Three domains, all in the Harbor container framework: **SpreadsheetBench Verified** (300 train / 100 test, pass@1), **AppWorld** (147 train / 168 `test_normal` tasks grouped into 56 scenarios, scenario goal completion), **USPTO Retrosynthesis** (500 train balanced over 10 reaction classes / 100 test, pass@1).

### Training-free: does agent-as-harness beat code-as-harness?

Same model plays student and reviewer, so no stronger-teacher confound. $h^\star$ was evolved on Qwen3.5-9B and reused for the frontier models.

| | $h$ (bare) | $h^\star$ (meta-harness) | $(h,\varnothing)$ review, empty $\mathcal{K}$ | $(h,\mathcal{K})$ |
|---|---|---|---|---|
| Average over 6 settings | 68.6 | 78.1 | 69.2 | **81.1** |
| Relative to $h$ | — | +22.5% | +1.0% | +27.6% |

The empty-$\mathcal{K}$ column is the key control: **review by itself buys +0.6 points.** Almost all the gain is the adapted harness content, not the act of having a second agent look at things. Biggest single win: DeepSeek-V4-Pro on USPTO, 31.0 → 60.0. Biggest non-event: GPT-5.6 Sol on AppWorld, already at 96.4, unmoved.

### Distillation

| | SpreadsheetBench | AppWorld | USPTO | Avg |
|---|---|---|---|---|
| Base, $h$ | 31.0 | 26.8 | 12.0 | 23.3 |
| Base, $h^\star$ | 39.0 | 48.2 | 38.0 | 41.7 |
| Base, DeepAgents | 35.0 | 19.6 | 7.0 | 20.5 |
| Base, Claude Code | 31.0 | 10.7 | 6.0 | 15.9 |
| **Harness-Zero, $h$** | **44.0** | **58.9** | 30.0 | **44.3** |

Two things stand out. First, **generic strong harnesses actively hurt a 9B model** — Claude Code drops AppWorld from 26.8 to 10.7. Big tool suites and long context are a tax the small model cannot pay, and the domains need *specific* tooling anyway.

Second, the split between what distils and what does not. Spreadsheet and AppWorld harnesses mostly encode **procedure** — inspect, edit narrowly, verify — and the distilled model beats $h^\star$ on both. USPTO's harness also carries **knowledge**: reaction priors, candidate enumeration logic, executable SMILES validation. There the distilled model triples the base (12 → 30) but still trails $h^\star$ (38). Procedure internalises; domain knowledge largely does not, at this data scale.

### The ablation that carries the paper (USPTO)

All conditions: same 500 collection tasks, same recipe, 2-epoch checkpoint, evaluated under $h$.

| Trajectory source | Collection success | Test pass@1 |
|---|---|---|
| Base, untrained | — | 12.0 |
| Teacher (GPT-5.6 Sol) rollout under $h$ | 52.0 | 12.0 |
| Teacher under $h^\star$ | 62.0 | **3.0** |
| Student under $h^\star$ | 39.4 | 12.0 |
| Review with **empty** $\mathcal{K}$ | 44.2 | 11.0 |
| Review with **oracle answer** | 98.6 | 15.0 |
| **Harness-Zero ($\mathcal{K}$)** | 59.4 | **30.0** |

What did **not** work, and why it is the interesting half:

- **Plain distillation from a much stronger model: zero gain.** 52% success trajectories, 12.0 after SFT. The teacher's responses are not reachable from the student's states.
- **Trajectories recorded under $h^\star$: actively harmful** (3.0). Diagnosis given: the distilled model repeatedly attempts harness tool calls that do not exist under $h$, and burns out the turn limit. This is the [[Fine-Tuning|action-space mismatch]] that agent-as-harness exists to fix.
- **Oracle answers: best collection success, near-worst distillation.** 98.6% of collection tasks solved, 15.0 test. Answer access produces shortcut corrections that do not generalise. **Collection success does not predict distillation value** — arguably the single most transferable lesson here.
- **Empty $\mathcal{K}$: 11.0.** Generic review is worthless as supervision, matching the training-free result.

So the effective ingredient is narrow and specific: *procedural* guidance, applied to the *student's own* trajectory, expressed in the *student's* action space.

### Behavioural recovery

They turn each enabled component of $h^\star$ into a deterministic trajectory detector (semantic, not string-match: AppWorld pagination counts whether done via the `aw.pages` helper or an explicit `page_index` loop). Support tasks = test tasks where the base model shows the behaviour under $h^\star$ but **never** under $h$, with a 10-task minimum. Duplicate detectors are merged. Result: 18 patterns on SpreadsheetBench, 6 on USPTO, 4 on AppWorld.

**82.3% average recovery across 28 patterns.** By construction the base model scores 0% on every one, so this measures behaviour that distillation *adds*. Perfect recovery (100%) on things like "parse product with RDKit first", "validate with RDKit", "canonicalize SMILES", "inspect before editing", "reload saved workbook", "avoid fragile coordinates". Weakest: AppWorld "retrieve complete pages" (40%) and "read back mutations" (50%) — multi-step disciplines that need the model to remember across turns.

## Worth Remembering

**The honest limitations, in the authors' own words.**

- **It needs a capable reviewer.** Appendix F extends USPTO down two capability tiers. The gap $(h,\mathcal{K}) - h^\star$ tracks model strength: +1.0 (GPT-5.6 Sol), +4.0 (DeepSeek-V4-Pro), −1.0 (V4-Flash), **−12.0** (Qwen3.6-35B-A3B). On the weakest model the reviewer replaces 66% of steps and its edits are net harmful; even empty-$\mathcal{K}$ review scores *below* the bare harness (12.0 vs 16.0). A weak reviewer is worse than no reviewer.
- **Collection is 2.4× slower** on USPTO (237.2s vs 100.1s per trial), because every proposal needs an extra model call that ingests the trajectory prefix plus $\mathcal{K}$. This cost is training-time only and vanishes at deployment.
- **The harness does not disappear, it shrinks.** Some mechanisms — notably context management, which is a property of the loop rather than of any single response — are not expressible as a student response at all. The claim is narrowing what the harness must provide, not eliminating it.
- Deep domain knowledge resists SFT at this scale.

**Surprises worth internalising.**

- The distilled small model under a bare harness beats itself under the good harness (44.3 vs 41.7). Scaffolding, absorbed, is better than scaffolding, attached.
- Making the data-collection policy *succeed more* can make the resulting student *worse*. Oracle answers, 98.6% → 15.0.
- Deliberately **denying** the reviewer a private channel into the environment is what makes the behaviour learnable. If the reviewer can check the workbook silently, the check never appears in the trajectory, so the student never learns to do it.

**Connections.** The DAgger framing is the cleanest mental model — see [[Imitation Learning]]. On the privileged-guidance side this sits beside [[What Does Privileged Information Add to On-Policy Self-Distillation]], [[Mind2Dialogue- Training Human-Aware Language Models by Simulating User Mental States]] and [[Best Practice Critic Optimization]]; on harness co-evolution, beside [[WHALE- A Simple Recipe for Joint Harness-Weight Optimization]], [[RRSI- Regularized Recursive Self-Improvement of Agent Harnesses]] and [[Recursive self-improvement of AI research agents]]. The cited multi-harness RL study (Le et al.) is the contrary evidence to keep in mind: changing the evaluation harness moved results more than the training method did, and cross-harness feedback did not transfer to a held-out minimal ReAct harness.

**If you wanted to use this.** Three things are load-bearing and easy to get wrong: (1) the privacy/structural filter and the reviewer-voice reasoning mask — without them you train on leaked review text; (2) the evolution rule banning task-specific content, which is your only defence against baking the training set into the "generic" harness; (3) the intervention budget, which converts "correct everything" into "correct selectively". Everything else is standard LoRA SFT.

**Open questions the authors raise.** The reviewer is implicitly solving a counterfactual prediction problem — *would executing this proposal lead somewhere worse?* — which is an agent world-model problem. Reviewing every step is wasteful; proxy signals could gate it. And each `replace` gives you a rejected and a preferred response **at the same state**, which is exactly a preference pair that response-level SFT throws away; [[DPO]]-style training on those is the obvious next move.

## Links

Related: [[Imitation Learning]] · [[Distillation]] · [[Distilling the Knowledge in a Neural Network]] · [[Fine-Tuning]] · [[LoRA]] · [[Agentic Workflows]] · [[Agent Frameworks]] · [[Tool Use]] · [[Context Engineering]] · [[WHALE- A Simple Recipe for Joint Harness-Weight Optimization]] · [[RRSI- Regularized Recursive Self-Improvement of Agent Harnesses]] · [[Recursive self-improvement of AI research agents]] · [[An Empirical Study of Harness Design for Coding Agents]] · [[What Does Privileged Information Add to On-Policy Self-Distillation]] · [[Best Practice Critic Optimization]] · [[Mind2Dialogue- Training Human-Aware Language Models by Simulating User Mental States]] · [[On-policy Distillation with Verifiable Reward]] · [[The Handoff Tax- Continuing Non-Native Trajectories in LLM Agents]] · [[DPO]] · [[Cross Entropy]] · [[Memory]] · [[Human in the Loop]] · [[Guardrails]] · [[Agent Evaluation]]

New topics worth writing: DAgger and expert relabelling on the learner's state distribution, agent harness as an object of optimisation, action-space mismatch in behaviour cloning, SpreadsheetBench, AppWorld, USPTO retrosynthesis as an agent benchmark, behavioural-pattern detectors for trajectory auditing, review-budget allocation in supervisor agents
