---
title: "SkillEvo: Self-Renewing Evolution Gradients from Multi-Turn Interaction Feedback"
authors: ["Qianxi Yan", "Chunrong Chen", "Jiuzhou Zhao", "Min Zhang", "Yongzhou Xu", "Xiaochuan Xu"]
year: 2026
arxiv: "2608.13120"
url: https://arxiv.org/abs/2608.13120
priority: Low-Priority
read_on: 2026-09-30
tags: [paper, llm, vision]
---
## The Core Idea

An "Agent Skill" is just a folder of Markdown: one `SKILL.md` routing file plus reference files holding domain knowledge. The agent loads it and answers from it. Today those folders are written by hand, or generated once by an LLM, and then they rot.

Prior work tried to close the loop: run the skill against test questions, look at failures, rewrite the skill. The problem this paper identifies is that **the feedback runs out**. If your test is a single question and a single answer, you only ever see the gaps that are visible in the user's opening sentence. Patch those in round one, and round two has nothing left to learn from. Their numbers show exactly this: single-turn-driven evolution goes 30.0 → 58.9 → 64.5 → 65.7 → 66.4 task success rate. Flat after round two.

The fix is to make the evaluator a **multi-turn simulated user**. Because the user asks follow-ups, patching a shallow gap lets the conversation get further, which exposes a *deeper* gap that was previously hidden behind the shallow one. So each round of repair both consumes feedback and manufactures new feedback. They call this a self-renewing "evolution gradient" — a metaphor, there are no real gradients here, it is text-in/text-out revision like [[DSPy]] or TextGrad. Same curve, different feedback source: 30.0 → 59.4 → 71.3 → 77.9 → **81.8**.

> [!NOTE] Evolution gradient
> A usable failure signal that tells the next round of editing what to change. It "decays" when the evaluation can no longer produce new failures. The paper's claim is that the bottleneck on skill self-evolution is neither the editor's ability nor the number of rounds, but whether the evaluator keeps producing these. ^evolution-gradient

The second half of the paper is about what goes wrong when you let an LLM rewrite a knowledge base four times in a row. Existing work gates each revision on one number: did the score go up? A scalar can reject a bad candidate, but it cannot tell you *what* broke. A skill is a directed graph — routing table pointing at reference files, files cross-referencing each other — and the three things that actually degrade are invisible in score space: files bloat, references dangle, and concrete facts ("200 GB, resets at 00:00 on 28 July") get softened into "refer to the official documentation". So they add a separate governance layer that diagnoses and repairs those directly.

## The Methodology

The loop, per round:

**Scenario Synthesizer.** Take a real support ticket that was escalated to a human. Pull out four things: an *intent agenda* (the user's requests, each labelled `key` or `minor`), *behaviour facts* (what the user already tried and saw), an *emotion trajectory*, and the *human reference solution*. The first three become the simulated user's persona. The reference solution goes only to the judge, never to the user — that is the answer-leakage guard.

**User Agent.** A model role-plays the user for up to 10 turns against the agent loaded with skill $S_t$. It is heavily constrained: reveal information one detail at a time, refuse tasks a real user cannot do (no log reading, no SSH), never request a human handoff. Every turn it emits an XML block with an `<agenda_check>` tag listing which agenda topics it just raised.

**Intent state machine.** Reads those `agenda_check` tags and tracks, per intent, *raised?* and *substantively answered?*. The dialogue may only terminate normally once every intent is both raised and answered; otherwise it terminates as abandonment. This is what stops the simulator from wandering off or quitting early.

**Dual-sided orthogonal evaluation.** The key trick for making the feedback trustworthy: score the simulator and the agent separately, so a failure caused by a bad simulation is not blamed on the skill.

Simulator side — intent coverage:
$$c_U = \frac{|\mathcal{K}_{\text{asked}}|}{|\mathcal{K}|}$$
If $c_U < 1$, the sample is thrown out of the agent's denominator entirely and labelled Evaluation Noise.

Agent side — exposed-intent accuracy, judged per intent against the human reference:
$$s_C = h\cdot\frac{\sum_i w_i a_i}{\sum_i w_i},\qquad w_i = \begin{cases}\alpha=0.7 & \text{key}\\ 1-\alpha=0.3 & \text{minor}\end{cases}$$
$h$ is a hard gate: if the right skill was not even routed to, $h=0$ and the sample fails outright. Note $s_C$ is deliberately *not* end-to-end resolution — it only grades intents that actually got raised.

> [!NOTE] Dual-sided orthogonal evaluation
> Grade the test harness and the system under test on separate axes, so you can tell "my agent is bad" apart from "my simulated user never asked the question". Without this, simulator distortion silently becomes training signal. ^dual-sided-evaluation

**Verifier.** An LLM judge scores each dialogue $0$–$100$ against the human reference. Pass needs $\ge 60$ *and* no missing key condition. Contradicting the human's rule caps you at 59. Style, verbosity and extra information are explicitly not grounds for deduction. Spot-checked against domain experts: >90% agreement.

**Collective Attribution.** Failures are sorted into three buckets by *who can fix it*:
- `knowledge_gap` — the human stated a stable fact the bot missed or got wrong. Repairable.
- `capability_limit` — permissions, missing tools, clumsy delivery, infrastructure faults.
- `eval_noise` — judge false negative, or a distorted scenario.

Only `knowledge_gap` gets written back. This is the piece that stops unrepairable signals from being mis-encoded as knowledge (which is the mechanism behind document bloat in prior work). Then all `knowledge_gap` cases in a round are merged by semantic similarity, so the editor fixes the common cause rather than each instance.

**Skill Optimizer.** Edits with two boundaries. The *evidence boundary*: only patch gaps present in the merged signal, invent nothing. The *reference boundary*: every edit is anchored to the production baseline $S_0$ so new knowledge cannot overwrite stable old facts. Three modes — `evolve` (first round), `fix` (after a failed inspection), `refine` (driven by the evaluation report, and told to induce 2–4 root causes rather than patch tickets one at a time).

**Skill Governor.** Two constraints.

*Fact consistency*, a **hard** constraint: $\operatorname{Facts}(S_t) \supseteq \operatorname{Facts}(S_0) \cap \mathcal{S}_{\text{stable}}$. Checked against **two anchors**, and this is the nice detail — diffing against $S_0$ finds facts lost cumulatively over all rounds; diffing against $S_{t-1}$ finds errors introduced *this* round. One anchor cannot tell them apart, so you cannot tell the editor whether to restore or to correct. A violation rejects the candidate and triggers same-round repair, handing the editor both diffs: deleted lines from $S_0\to S_t$ say what to restore, changed lines from $S_{t-1}\to S_t$ say what not to revert.

*Structural consistency*, a **soft** constraint: detect knowledge bloat, reference breakage (dangling refs, orphan files), and factual over-generalisation. Rather than rejecting, it emits recommendations (merge sections, consolidate scattered tail notes, split files over 700 lines) which get merged with next round's knowledge gaps. So structure degradation dissolves along the iteration rather than blocking it.

**Setup.** Tencent Cloud production support. 6 service categories, 9 skills, 98 reference files, 2,000 tickets — all of them tickets that were escalated to humans, i.e. the existing skills' *failure set*, which is why the original-skill baseline is only 30.0. Tickets per skill are ordered chronologically and split into quarters: first three quarters drive evolution, last quarter is held out for reporting only. 4 outer rounds, up to 3 edit iterations per round. Editor is `deepseek-v4-pro`; user simulator, verifier, attributor and governor are all `minimax-m3` — enforcing Generator ≠ Evaluator so a model never reviews its own edits.

## Ablation Studies and Experiments

Headline, evaluation-set TSR (%), rounds 1–4:

| Method | Init | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| Original skill | 30.0 | — | — | — | — |
| Self-Reflection (no eval) | 30.0 | 59.2 | 58.7 | 57.4 | 58.8 |
| Single-turn QA (SkillForge) | 30.0 | 58.9 | 64.5 | 65.7 | 66.4 |
| **SkillEvo** | 30.0 | 59.4 | 71.3 | 77.9 | **81.8** |

Three tiers, three behaviours. Self-reflection has no evaluation, so no gradient *and* no gate — it oscillates around its first-round level. Single-turn QA has a decaying gradient but keeps its gate, so it plateaus rather than falling. Only the multi-turn version keeps climbing. All three jump hugely in round one, because round one is just "patch the obvious holes".

**Ablation:**

| Variant | TSR |
|---|---|
| Full | 81.8 |
| (a) swap multi-turn → single-turn QA | 66.4 |
| (b) remove governance layer | 78.6 |

Variant (a) is the important one. Holding attribution, revision and governance fixed and changing *only the feedback source* reproduces the single-turn baseline exactly (66.4). So the whole 15.4-point lead is attributable to the feedback modality, not to any of the surrounding machinery.

Variant (b) only costs 3.2 points — and the authors are straight about this: governance is not there to raise the score. Its job shows up in the two degradation metrics.

**Cross-round regression rate**, the fraction of previously-passing tickets that break:
$$\mathrm{RegR}(r) = \frac{|\{t: s_{r-1}(t)\ge 60 \wedge s_r(t) < 60\}|}{|\{t: s_{r-1}(t)\ge 60\}|}$$
28.2% → 24.4% → 21.1% across the three transitions. Falling, but note the level: **roughly a fifth of working tickets break on every revision**. That is not a small number, and it is the clearest honest signal in the paper about how fragile text-based self-editing still is.

**Knowledge bloat**, line growth over baseline:
$$\mathrm{Bloat}(S_t) = \frac{\text{lines}(S_t) - \text{lines}(S_0)}{\text{lines}(S_0)}$$
+2.8% with governance, **+16.2% without** — about six times worse, and growing round on round with no dissolution mechanism. This is the direct answer to SkillForge's append-only policy. The framing they draw from it is the strongest claim in the paper: TSR rises 51.8 points while document size barely moves, so the gain comes from *correcting* existing knowledge, not from adding text.

**Trustworthiness of the simulator side:** intent coverage $c_U = 98.9\%$; exposed-intent accuracy $s_C = 71.1\%$. To rule out "the state machine just forces a checklist walk", they had two domain experts blind-compare 200 simulated dialogues against their real tickets on intent expression, information-reveal pace and emotion trajectory — 95.3% agreement. That is a good control and more than most simulation papers do.

**The case study (Appendix D) is the best argument in the paper.** A COS traffic-package renewal question. The original skill states the rule *exactly backwards* ("renewal takes effect immediately, quota is added at once"; truth is "renewal extends validity, fresh quota at the reset date"). The simulated user does not abandon — the answer was plausible and self-consistent — so they build on the false premise and ask about service suspension, which the agent answers *correctly*. A single-turn test would likely never surface this. The attributor pulls three facts, the editor appends them to one section of one file, and the re-run scores 92. Their line: **wrong knowledge is more deceptive than absent knowledge**, and only multi-turn interaction reaches it.

**What did not work / was excluded:** single-turn paradigms like GEPA, TextGrad, SkillOpt, Trace2Skill and SkillCAT are argued to be inapplicable rather than beaten — they assume automatically verifiable single-turn tasks and cannot express layered intent exposure or dialogue-level judgment. So SkillForge is used as the sole stand-in for that whole tier. No sensitivity sweep was run on $\alpha$, the pass threshold or the early-stopping criteria; they inherit the deployed pipeline's values.

## Worth Remembering

**Read the per-round table carefully.** For the two methods with evaluation feedback, the reported round-$r$ number is *the best dev-selected version up to round $r$*, which makes those curves monotone by construction. Self-Reflection has no gate so it reports its current version and oscillates. The SkillEvo-vs-single-turn comparison (the 15.4 points) is apples-to-apples; the 23.0-point margin over self-reflection is partly a selection artefact.

**The judge is inside the loop and also reports the result.** `minimax-m3` is the verifier, the user simulator, the attributor *and* the governor. They carefully enforce editor ≠ evaluator, but not simulator ≠ judge, so the same model writes the exam and marks it. And TSR — the headline metric — is that same verifier's output. Some of the 51.8-point gain is plausibly the editor learning to satisfy the judge rather than the user. Classic [[Reward Hacking|Goodhart]] exposure; the >90% expert agreement check mitigates it but does not remove it. See also [[Evals#LLM-as-judge, done properly ⚖️|judge rules]].

**The 30.0 baseline is not a general capability number.** Every ticket in the dataset is one the existing skills already failed on. Improvement on a curated failure set is much easier to obtain than improvement on the full traffic distribution, and the paper does not report what happens to tickets the original skills already handled.

**No error bars, one run, no significance testing.** With RegR sitting at 21–28% per round, round-to-round variance is likely material. Also unreproducible externally: the ticket data cannot be released for privacy and commercial reasons, and both model names (`deepseek-v4-pro`, `minimax-m3`) are not ones I can verify exist.

**The bits worth stealing even if you never build a skill system:**
- *Dual-anchor diffing.* When you have a system that edits itself repeatedly, diff against both the original and the previous step. One anchor cannot distinguish accumulated loss from a fresh mistake, so it cannot tell you what to do.
- *Attribution before repair.* Classify failures by *who can fix this* before feeding them anywhere. Their `knowledge_gap / capability_limit / eval_noise` split is a direct fix for the bloat failure mode — unrepairable signals get written in as knowledge if you do not screen them.
- *Separate the harness metric from the system metric.* $c_U$ for the simulator, $s_C$ for the agent. Any offline evaluation harness needs this, and most do not have it.
- *Soft vs hard constraints.* Facts are hard (reject and repair). Structure is soft (recommend, and let the next round absorb it). Rejecting on structure would stall the loop.

**Honest limitation the authors state:** nothing ships without human confirmation. The loop terminates at a reviewed candidate, not an automatic rollout — they call this a necessary condition, not an optional safeguard. Which means the throughput win is in *drafting* the revision, not in removing the human. That is a smaller claim than the framing suggests, and the right one. See [[Human in the Loop]].

**Open questions.** Does the multi-turn gradient also decay, just later? Four rounds is not enough to see the ceiling, and rounds 3→4 already slowed (77.9 → 81.8). What happens if the simulator and judge are different families? And can the RegR of ~20% per round be driven down, or is that the floor for LLM-edited knowledge bases?

## Links
Related: [[Evals]] · [[Agent Evaluation]] · [[Reflection]] · [[RAG]] · [[Context Engineering]] · [[DSPy]] · [[Reward Hacking]] · [[Human in the Loop]] · [[Credit Assignment]] · [[Chain-of-Experience for Continual LLM Improvement]] · [[Harness-Zero- Harness Distillation via Agent-as-Harness]] · [[RRSI- Regularized Recursive Self-Improvement of Agent Harnesses]] · [[WHALE- A Simple Recipe for Joint Harness-Weight Optimization]] · [[Using Grounded Theory for Agent Behavior Analysis at Scale]] · [[ImpossibleRubrics- Stress-Testing Generated Rubrics as Reward Signals]] · [[Observability and Tracing]] · [[Grounding]] · [[Hallucination]]

New topics worth writing: Agent Skills as a knowledge carrier, user simulation for dialogue evaluation (τ-bench, SAGE, ECom-Bench), TextGrad and GEPA (textual "gradients" for prompt/program optimisation), SkillForge, failure attribution in multi-agent systems, offline dialogue evaluation harness design, knowledge-base drift and regression testing for RAG corpora
