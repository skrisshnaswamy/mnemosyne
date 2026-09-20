---
title: "The Handoff Tax: Continuing Non-Native Trajectories in LLM Agents"
authors: ["Ganz et al."]
year: 2026
arxiv: "2608.24358"
url: https://arxiv.org/abs/2608.24358
priority: Good-To-Read
read_on: 2026-09-04
tags: [paper, llm]
---
## The Core Idea

You are running a coding agent. Halfway through a long task, you switch models — either up to a stronger, pricier model because the cheap one is stuck (**escalation**), or down to a cheap model because the hard thinking is done (**downshift**). Every coding agent product has a `/model` command for exactly this.

The new model now has to continue a conversation it did not write. It inherits someone else's reasoning, someone else's dead ends, someone else's tool-call habits. The paper calls this a **non-native trajectory**, and measures what it costs.

The headline: escalation is a bad deal. Passing the full cheap-model transcript to the expensive model recovers **less than half** of the quality gap between the two models, while costing several times more than just running the cheap model alone. For the Claude pair it is worse than that — paying for the whole cheap run and *then* restarting the expensive model from scratch is both cheaper and more accurate than continuing. They name this penalty the **handoff tax**.

The interesting asymmetry is what fixes it. In escalation, *deleting* the cheap model's transcript (but keeping the files it edited on disk) helps a lot — quality recovery jumps from 47% to 64% on Claude, 36% to 84% on GPT. In downshift, deleting the strong model's transcript *hurts* — recovery drops from 50% to 28% on Claude. So a weak model's reasoning is a burden on a strong receiver, and a strong model's reasoning is a crutch a weak receiver genuinely needs.

> [!NOTE] Handoff tax
> The combined quality shortfall and cost premium you pay when one model continues a trajectory produced by a different model, rather than starting fresh or finishing alone. ^handoff-tax

> [!NOTE] Escalation vs downshift
> Escalation = start with the low-cost/low-capability model (LC), switch to the high-cost/high-capability one (HC). Downshift = the reverse. LC/HC are always a *pair within one family*, e.g. Haiku 4.5 → Opus 4.7. ^escalation-downshift

Why this did not exist before: prior work on cost saving is about **routing** and **cascades** — pick which model handles a request, or retry with a bigger one. Those decide *who acts next*. Nobody had studied *what the next actor is handed*. In a long-horizon coding agent the receiver inherits two things — the conversation and the actual edited repository on disk — and those two turn out to be separable, with opposite value.

## The Methodology

**Setup.** SWE-bench Verified, all 500 real GitHub issues, graded by executable tests. One scaffold throughout: `mini-swe-agent`, same prompts, same Bash tool, same Docker container. Two model pairs:

- Claude: Haiku 4.5 (LC) → Opus 4.7 (HC). HC costs ~2× LC per task.
- GPT-5.6: Luna (LC) → Sol (HC). HC costs ~8× LC per task.

Handoff is implemented by swapping the model inside one continuous agent loop. The Docker container is never reset, so **the working tree — every file the first model edited — always survives the switch**. Only the conversation changes.

**The four interfaces.** Let $\mathcal{T}_{1:K}$ be the trajectory (messages, tool calls, observations) up to step $K$, and $\mathcal{W}_K$ the working tree. All four keep $\mathcal{W}_K$:

| Interface | What the receiver gets |
|---|---|
| **Raw** | $\mathcal{T}_{1:K}$ verbatim. No message says the model changed. This is what products do today. |
| **Compact$_\text{pre}$** | The *sender* reads $\mathcal{T}_{1:K}$ and writes a plain-text summary. Receiver sees only the summary. |
| **Compact$_\text{suf}$** | The *receiver* reads $\mathcal{T}_{1:K}$ and summarises it itself, then continues from its own summary. |
| **Traj-drop** | Nothing. System prompt + original task + one static line: "A previous agent worked on this task. Any file changes it made are still in the working tree." |

Both compaction variants pay for one extra summarisation call, charged to the handoff.

**When to switch.** Fixed step counts are not comparable — an easy task and a hard task have very different trajectory lengths. So switch points are **percentiles of the prefix model's own termination-step distribution, computed separately per difficulty bucket** (easy / medium / hard). They sweep $p \in \{5, 10, 15, 25, 35, 45, 50\}$. Concretely: Haiku on a medium task, p25 = step 51; Opus on the same bucket, p25 = step 16. Opus finishes in far fewer steps, so its percentiles are much earlier in absolute terms.

**Fair comparison.** If the prefix model finishes before step $K$, no handoff happens and that run tells you nothing. So every comparison is on the **switched subset** — the intersection of tasks that actually switched under *all four* interfaces, with the LC-only and HC-only baselines recomputed on that same subset.

**Two normalised metrics.** With $R_m(K)$ = pass rate and $C_m(K)$ = mean dollar cost of strategy $m$:

$$\mathrm{QRec}(m,K) = 100\,\frac{R_m(K) - R_{\text{LC}}(K)}{R_{\text{HC}}(K) - R_{\text{LC}}(K)}$$

$$\mathrm{CSRet}(m,K) = 100\,\frac{C_{\text{HC}}(K) - C_m(K)}{C_{\text{HC}}(K) - C_{\text{LC}}(K)}$$

Quality Recovery is 0 at LC quality, 100 at HC quality. Cost-Savings Retention is 100 at LC cost, 0 at HC cost. **Negative CSRet means you spent more than just running HC alone.** These two numbers carry the whole paper.

**Two restart controls** for escalation, built by stitching together already-measured runs rather than executing anew. Both discard the LC trajectory *and* its edits, restarting HC from the pristine repo, but you still pay for the LC work:
- *Abort + HC fresh*: pay LC through step $K$, then a full HC run.
- *LC-full + HC-full*: pay a complete LC run, then a full HC run.

Both get HC-only quality by construction, so the only question is cost.

**Scale.** 58 configurations per family (2 baselines + 7 switch points × 4 interfaces × 2 directions). 58,000 agent runs, 2 million API calls, 36 billion tokens.

## Ablation Studies and Experiments

**Escalation, averaged over all seven switch points:**

| Strategy | Claude Pass / Cost / QRec / CSRet | GPT Pass / Cost / QRec / CSRet |
|---|---|---|
| LC-only | 60.7% / \$0.40 / 0 / 100 | 58.7% / \$0.06 / 0 / 100 |
| HC-only | 79.2% / \$0.72 / 100 / 0 | 83.7% / \$0.47 / 100 / 0 |
| Abort + HC fresh | 79.2% / \$0.90 / 100 / −58 | 83.7% / \$0.51 / 100 / −11 |
| LC-full + HC-full | 79.2% / \$1.12 / 100 / −130 | 83.7% / \$0.53 / 100 / −15 |
| **Raw** | 69.2% / **\$1.61** / 47 / **−285** | 67.5% / \$0.36 / 36 / 26 |
| Compact$_\text{pre}$ | 71.8% / \$0.75 / 60 / **−11** | 68.8% / \$0.27 / 40 / **49** |
| Compact$_\text{suf}$ | 69.6% / \$0.98 / 49 / −82 | 68.8% / \$0.43 / 40 / 10 |
| Traj-drop | 72.4% / \$0.81 / **64** / −30 | 79.7% / \$0.50 / **84** / −8 |

The Claude Raw row is the paper's most striking number. \$1.61 versus \$0.72 for just running Opus. It is **strictly dominated** — both restart controls cost less (\$0.90, \$1.12) *and* solve more tasks. Continuing Haiku's transcript is worse than throwing it away after paying for it.

**Downshift, same averaging:**

| Strategy | Claude Pass / Cost / QRec / CSRet | GPT Pass / Cost / QRec / CSRet |
|---|---|---|
| LC-only | 54.6% / \$0.41 / 0 / 100 | 63.6% / \$0.05 / 0 / 100 |
| HC-only | 75.8% / \$0.85 / 100 / 0 | 85.8% / \$0.47 / 100 / 0 |
| **Raw** | 65.6% / \$0.51 / 50 / **80** | 81.0% / \$0.41 / **79** / **14** |
| Compact$_\text{pre}$ | 66.8% / \$0.52 / **56** / 78 | 79.8% / \$0.43 / 72 / 10 |
| Compact$_\text{suf}$ | 63.7% / \$0.53 / 42 / 73 | 80.4% / \$0.42 / 75 / 13 |
| Traj-drop | 60.9% / \$0.59 / 28 / 59 | 75.4% / \$0.43 / 53 / 10 |

Downshift is the good deal. Claude keeps 80% of the cost savings and half the quality. GPT keeps 79% of the quality for well under HC cost. And Traj-drop — the *best* escalation interface — is the *worst and most expensive* downshift interface in both families. That reversal is the paper.

**The cost mechanism.** They decompose post-handoff cost into (steps taken) × (cost per step):
- Raw escalation: each HC step costs **2.2× more** than under Compact$_\text{pre}$ (1.6× for GPT), with a similar number of steps. The premium is expensive calls — the strong model is re-reading a huge cheap-model transcript on every turn, and prompt caching does not save you.
- Traj-drop downshift: **1.6× more LC steps** than Compact$_\text{pre}$ (2.0× for GPT), at similar cost per step. The premium is rework — the weak model rediscovers what the strong model already figured out.

Same tax, two completely different physical causes.

**Difficulty conditioning (Claude).** On easy tasks every escalation interface is terrible: Raw has CSRet of −982%. On *hard* tasks all three reduced-context interfaces flip to cheaper-than-HC (CSRet +12% to +42%) while recovering 65–74% of the gap. Raw never makes this transition (still −86%). Caveat the authors flag themselves: only ~24 tasks per hard cell, treat as exploratory.

### What did not work

- **Just telling the model.** They tried Raw + a single injected message: "The preceding assistant messages contain a previous agent's work... The previous agent used a different model." Quality nudged from 48 → 55 QRec, but cost got *worse* (−285 → −300 CSRet). Disclosure alone does not fix the tax. The problem is not that the model is confused about authorship; it is that the context itself is expensive and misleading.
- **Compact$_\text{suf}$** (receiver writes its own summary) is consistently the weakest compaction variant — it still pays to read the full non-native trajectory, so it gets neither the cost saving nor a cleaner summary. On Claude escalation it costs \$0.98 vs \$0.75 for Compact$_\text{pre}$.
- **Who authors the summary is not a cross-family finding.** Compact$_\text{pre}$ beats Compact$_\text{suf}$ by +2.2pp on Claude escalation (CI [+0.7, +3.7]) but by −0.0pp on GPT (CI [−2.4, +2.4]). The authors explicitly decline to claim it.
- **Traj-drop downshift on easy Claude tasks goes negative**: QRec −28%, i.e. *worse than the cheap model alone*, despite inheriting Opus's file edits. Being handed a half-finished repo with no explanation is actively harmful.

**Statistics.** Task-clustered bootstrap, 10,000 resamples, pointwise 95% CIs on pairwise pass-rate differences. The central reversal survives: Traj-drop − Raw is +3.1 [+1.3, +5.0] for Claude escalation and −4.7 [−7.3, −2.1] for Claude downshift; GPT is +12.2 and −5.6, both excluding zero.

### Beyond coding

They vary one orthogonal thing: **when task-relevant information arrives**. Raw handoffs only.

- **Lost in Conversation** (535 tasks, requirements revealed one shard per turn). Here the ordering flips: escalation gets QRec 86 / CSRet 36, downshift only 31 / 53 (Claude). Reason: the task is not fully specified until the last shard, so whoever holds the final turns is the model that actually solves it. Put HC there.
- **BrowseComp** (200 browsing questions, GPT pair). Escalation nearly closes the quality gap (QRec 95.8) — inherited search results are genuinely useful, cutting the HC continuation from 33.4 to 30.4 steps. But CSRet is still −30%, worse than Abort + fresh HC at −8.5%. Downshift again gives the useful middle: 56.7 QRec at 76.8 CSRet.

So escalation *can* recover quality when the inherited state is evidence rather than reasoning. It still does not save money.

## Worth Remembering

- **The practical rule.** Downshifting mid-run is a genuinely good deal and you should pass the full trajectory. Escalating mid-run mostly is not — and if you do it, strip the cheap model's transcript and keep only its file edits. If you want minimum cost, use sender-written compaction; if you want maximum quality, drop the trajectory entirely.
- **The dual framing.** HC trajectories *guide* an LC receiver; LC trajectories *burden* an HC receiver. The weak model does not have the ability to profitably question the strong model's reasoning, so it follows it and benefits. The strong model gets anchored on wrong turns it would never have taken, and pays token cost to read them.
- **CSRet negative is the number to watch.** It means "you would have been better off starting with the expensive model in the first place." Claude Raw escalation at −285% is a genuinely shocking production result given this is the default behaviour in shipping products.
- The two model pairs have very different economics (2× vs 8× HC/LC cost ratio) and this changes the *cost* conclusions while leaving the *quality* conclusions intact. Do not port the dollar numbers to your own pair without re-measuring.
- **Limitations the authors state plainly:** one benchmark for the main study; two model pairs; a *single* run per task-condition, so no within-task variance estimate (they mitigate with matched subsets and clustered bootstrap); switch points fixed in advance rather than triggered by an adaptive policy; hard-difficulty cells only ~24 tasks; all dollar figures depend on provider pricing and cache-hit rates.
- **Open question they point at.** Handoff design should be optimised *jointly* with routing and timing. Current routers assume the full trajectory just carries forward. It does not, and the right amount to carry depends on which direction you are going.
- Connects naturally to context compaction in agent frameworks and to [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)|KV-cache reuse]] — Raw handoff invalidates the sender's cache entirely, which is part of why per-step cost explodes.

## Links

Related: [[In Context Learning]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]] · [[Distilling the Knowledge in a Neural Network]] · [[SWE Refactor Bench- Can Coding Agents Complete a Long-Horizon, Whole-Repository Stack Migration]] · [[Prefix Sliding for efficient test-time scaling]] · [[On the Difficulty of Evaluating Baselines]] · [[The Embedder's Dilemma- LLMs Are Better, but at What Cost]]

New topics worth writing: LLM cascades and cost-aware routing, SWE-bench Verified, agent scaffolds (mini-swe-agent / SWE-agent), context compaction in long-horizon agents, ReAct tool-use loops, task-clustered bootstrap confidence intervals, prompt caching economics
