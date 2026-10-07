---
title: "PolicyGuide: From Guarding One Action to Guiding the Whole Workflow for Policy-Compliant LLM Agents"
authors: ["Seongjae Kang", "Taehyung Yu", "Sung Ju Hwang"]
year: 2026
arxiv: "2608.19861"
url: https://arxiv.org/abs/2608.19861
priority: Low-Priority
read_on: 2026-10-01
tags: [paper, llm, theory]
---
## The Core Idea

A customer-service LLM agent that acts on your account has to follow a company policy. Two different things can go wrong. It can do a **forbidden thing** — change a ticket that is not eligible. Or it can do an **allowed thing the wrong way** — change the ticket without first checking who you are, without checking eligibility, without asking you to confirm.

The existing guard tools only catch the first kind. They sit in front of the dangerous call (the "mutating tool call" — a tool that writes to the database) and say pass or block. That is an *action-local* check. By the time the agent reaches `update_reservation_flights`, the three skipped steps already happened. The guard can block, but blocking is a late, blunt signal, and the agent then has to guess the procedure from an error message.

There is a second literature — workflow agents, SOP agents — that walks an agent through a prescribed procedure. But those systems *are* the agent. They optimise for finishing the workflow, not for stopping a general agent from misbehaving.

PolicyGuide's move: **make the workflow an external monitor, and give it memory.**

Compile each domain's plain-English policy into a graph of required steps, once, offline. At runtime a separate verifier LLM holds that graph, remembers where each of the user's open requests sits in it, and before every agent reply it walks the graph from the remembered position, stops at the first step that is not yet satisfied, and hands the agent one concrete instruction: *ask for the user id*, or *refuse, the fare is Basic Economy*.

> [!NOTE] Action-local vs workflow-level guarding
> An action-local guard fires only when the agent proposes a guarded action. A workflow-level guard fires at fixed points in the conversation regardless of what the agent is doing, so it can catch deviations that never touch a guarded action at all. ^action-vs-workflow-guard

Why this matters is an empirical fact about the policies, not a philosophical claim. The authors hand-classified every atomic requirement in the three [τ²-bench](https://arxiv.org/abs/2506.07982) policy documents:

| Domain | argument-level | process-level (flat) | workflow-level (ordered) | % ordered |
|---|---|---|---|---|
| Airline | 14 | 27 | 2 | 4.7% |
| Retail | 0 | 27 | 1 | 3.6% |
| Telecom | 1 | 22 | 27 | 54.0% |

Telecom's tech-support manual is 20/21 ordered. Its procedures are *diagnose → tell the user to toggle something on their phone → verify it worked*. The fix happens on the user's device. **There is often no agent-side mutating call for an action guard to intercept at all.** That is the hole, and it is why PolicyGuide's telecom number moves so far.

What it unlocks: the same compiled policy graph works with GPT 5.4, Claude Sonnet 4.6 and Gemini 2.5 Pro without re-authoring, because the graph lives outside the agent.

## The Methodology

Two halves: an offline compiler and an online verifier.

### Offline — policy text to workflow graph

A six-stage pipeline, run once per domain, frozen afterwards.

1. Pull out the tool specs; mark which tools mutate. Exclude user-device actions.
2. Derive request types, shared procedures, ordered subflows; audit policy coverage.
3. Review that plan.
4. Generate each subflow, schema-validate it, one repair retry.
5. Wire the intake spine + classifier + subflows together.
6. Validate: schema, tool inventory, every mutating tool has an authorization node, edge arity, reachability. Prune unused subflows.

Node types:

- `entry` / `exit` — structure
- `agent_action` — agent does something that is not a tool call
- `user_input` — a user reply is required
- `tool_call` — read-only tool
- `tool_authorization` — the choke point for one mutating tool
- `decision` — a branch
- `subflow` — invoke a shared procedure

Every node carries an **explicit satisfying condition** in natural language, which the runtime verifier judges against the conversation. Subflows are inlined at load time, so the runtime traverses one flat graph. Node ids become paths: `identify_user.load_profile`, `book_flow.authorize_book`.

The pattern for any write: `authorize_<tool>` (trapezoid) → `verify_<tool>`. Everything the policy requires before the write must sit *upstream* of the authorization node. The verify node afterwards demands a successful `TOOL_RESULT` — this is how post-action checks like "always verify the bill became PAID" get represented.

Graph sizes: airline 158 nodes / 11 authorization nodes, retail 104 / 7, telecom 127 / 5.

### Online — the verifier

One call per firing:

$$V_\phi(\pi, \mathcal{T}, G, H, S) = (d, \widehat{\mathcal{R}})$$

- $\pi$ = raw policy text, $\mathcal{T}$ = tool specs, $G$ = frozen graph, $H$ = conversation history, $S$ = code-owned request state
- $d$ = merged remediation (what the agent should do next), $\widehat{\mathcal{R}}$ = updated request records

**When it fires.** Before the agent responds to each user turn. Plus once more if the agent tries a mutating call that the current graph state has not authorized. Tool-result turns are skipped and folded into the next firing's conversation delta, so each call still sees the whole trajectory.

**What it does, in three steps** (Algorithm 1):

1. **Reconcile.** Match the requests visible in the conversation against the tracked ones in $S$. A new request opens at the graph entry. A continuing one keeps its recorded node. Abandoned requests are dropped, duplicates merged.
2. **Traverse.** For each open request, walk from its recorded node. At each node, judge the satisfying condition against $H$. Satisfied → step to the successor whose edge matches. Not satisfied → **stop**. One generation can advance several nodes.
3. **Decide.** The first unsatisfied node's required action becomes that request's remediation. Terminal node → request done. Merge remediations across requests, return them.

**The evidence rule** is the load-bearing bit of the prompt. Facts and eligibility count *only* when a tool result confirms them — not when the user asserts them, not when the agent says them. The user's own choices and consent count from their message. A tool result that contradicts a user claim wins. Every argument the verifier puts in an authorizing instruction must be recorded in `grounded_values` with a source: which tool result, which record, or `'user message'`.

**Code owns state, not the model.** The runtime rejects unknown node ids, filters the verifier's authorization output against the enumerated mutating-tool inventory, rebuilds the enabled tool set, and persists each request's position. The merged remediation is injected as a guidance message before the agent acts.

**Intervention is advisory, not hard.** The first unauthorized mutating call in each user-turn region is intercepted before it runs and triggers a corrective verifier firing. Then the gate **disarms for one immediate retry** — otherwise a wrong verifier judgement would deadlock the run. Everything else is steered by text, not gated.

Verifier runs at temperature 0, model-paired with the agent. The prompt is a byte-stable cached prefix (policy + graph + tools + judging rules + output contract) plus the varying conversation.

### The theory, briefly

Represent an interaction as an event sequence. $P_G$ is the set of valid prefixes — partial sequences that can still be completed compliantly. A **reachable first deviation** is a valid prefix $\tau$ plus a next agent event $e$ with $\tau e \notin P_G$.

**Theorem 1:** an ideal binding verifier preserves validity throughout every execution *iff* its firing schedule covers every reachable first deviation.

The point is timing. $P_G$ is prefix-closed, so once $\tau e \notin P_G$, no later action can make it valid again. An action-triggered guard that fires later can block the next write, but it cannot undo the procedural violation that already happened.

**Corollary 1:** an action-triggered schedule covers exactly the deviations whose event is in the guarded action class $A$. If any first deviation has $e \notin A$ — say, telling the user to reboot before diagnosing — that guard cannot help, even if the guard itself is perfect.

**Corollary 2** is honest about the shipped system: firing at user-turn boundaries plus one intercept does *not* cover everything. A deviation that happens between two firings escapes.

### Two ablation variants

- **PolicyGuide-Raw** — same verifier model, same schedule, same remediation channel, but $G$ is replaced by the raw policy text. No graph position persists. Isolates the compiled graph.
- **PolicyGuide-Self** — the frozen graph goes in the *agent's* system prompt; no external verifier, no code-owned state, no per-turn remediation, no intercept. Isolates external tracking (as a bundle).

## Ablation Studies and Experiments

Benchmark: τ²-bench base splits — Airline 50 tasks (24 policy-violation / 26 mutation), Retail 114 (10 PV / 104 Mut), Telecom 114 (43 PV / 71 Mut). PV = the agent must refuse. Mut = the agent must complete the change correctly. Success needs both the final database state and the natural-language assertions to hold.

$\text{Pass}^k = \binom{c}{k} / \binom{n}{k}$ averaged over tasks, where $c$ of $n=4$ trials passed. $\text{Pass}^4$ is "all four trials passed" — a reliability metric.

### Main results, GPT 5.4 agent and verifier

$\text{Pass}^4$, overall:

| System | Airline | Retail | Telecom |
|---|---|---|---|
| ReAct (no guard) | 0.460 | 0.596 | 0.193 |
| ToolGuard | 0.520 | — | — |
| PolicyGuard | 0.580 | 0.360 | 0.202 |
| **PolicyGuide** | **0.620** | **0.614** | **0.614** |

Mean across domains: 0.42 → 0.62.

Telecom is the story: **0.19 → 0.61**, and on the Mut slice **0.042 → 0.549**, a 13× move. That is exactly where 54% of requirements are ordered.

Retail is the counter-story and the authors say so: PolicyGuide's edge over ReAct is +0.018, **not significant** (bootstrap CI $[-0.070, +0.105]$). Retail is 96% process-level but almost 0% ordered — flat gates that a conversation-aware pass/block guard already covers.

The most interesting negative in the table is **PolicyGuard on Retail: 0.360 vs ReAct's 0.596.** Its PV goes to 0.900 but Mut drops to 0.308. A block-only guard trades completion for refusal. PolicyGuide keeps Mut at 0.587 while lifting PV to 0.900 — because it tells the agent what to do, not just what not to do.

Pooled stratified McNemar over all three domains: vs ReAct $Z = +5.92$, $p < 10^{-8}$; vs PolicyGuard $Z = +7.24$, $p < 10^{-12}$.

### What did not work: giving the agent the graph

**PolicyGuide-Self** puts the frozen graph in the actor's prompt and removes everything else. $\text{Pass}^4$ on the Mut slice:

| Domain | ReAct | Self | Raw | Full |
|---|---|---|---|---|
| Airline | 0.192 | **0.154** | 0.192 | 0.346 |
| Retail | 0.556 | **0.306** | 0.556 | 0.694 |
| Telecom | 0.053 | 0.053 | 0.053 | 0.684 |

Self never beats ReAct on Mut, and on Retail it is 25 points *worse*. Handing a model its own procedure is not enough. Something external has to track position and hand back one step at a time. (Caveat the authors flag: Self removes several components at once, so this tests the external stack as a bundle, not state persistence alone.)

### What the graph itself buys

**PolicyGuide-Raw** keeps everything except the compiled graph. Gap to full PolicyGuide, overall $\text{Pass}^4$:

- Airline **+0.100**
- Retail **+0.150**
- Telecom **+0.325**

The gap tracks the workflow-level fraction almost exactly. The graph is doing structural work — letting the verifier resume a long diagnostic chain — rather than just being a better-formatted policy.

Note Telecom-Raw's Mut is 0.053, identical to ReAct and Self. Without the graph, telecom mutation tasks are unimprovable by this verifier. That is a clean result.

### Matched workflow-controller baseline

FlowAgent, on the 40-task Telecom test split, with the *same* frozen graph deterministically compiled into PDL (no LLM re-authoring):

| System | Control | $\text{Pass}^4$ |
|---|---|---|
| ReAct | actor only | 0.250 |
| PolicyGuard | action-local | 0.325 |
| FlowAgent | PDL inside actor + API controllers | 0.350 |
| PolicyGuide | external graph verifier | **0.675** |

Same procedural information. The difference is who holds it.

### Cross-agent transfer, Airline

The GPT 5.4-authored graph, reused unchanged. $\text{Pass}^4$ overall:

| Agent | ReAct | PolicyGuard | PolicyGuide |
|---|---|---|---|
| GPT 5.4 | 0.460 | 0.580 | 0.620 |
| Claude Sonnet 4.6 | 0.720 | 0.780 | 0.780 |
| Gemini 2.5 Pro | 0.480 | 0.600 | 0.680 |

On Claude it ties PolicyGuard. On Gemini the Mut slice goes 0.231 → 0.462 while PV drops 1.000 → 0.917 — a completion gain paid for with slightly less strict final blocking. **Transfer across workflow-*author* models is untested.**

### Adversarial: CRAFT red-teaming

Persuasive users inject false eligibility claims to force a forbidden change. 20 airline attack tasks. Per-trial attack success rate (lower is safer):

- ReAct 0.200
- PolicyGuard 0.125
- **PolicyGuide 0.087** — 91.3% of attacks prevented

Lowest ASR@$k$ at every $k$. The mechanism is the evidence rule: a workflow prerequisite can only be satisfied by a tool result, so a confident user claim cannot unlock the gate.

### Procedural compliance, not just outcomes

τ²-bench scores final database state. It does not check whether the steps happened in order. So the authors hand-wrote a Telecom rubric from the raw policy: identification, diagnose-before-intervene, consent, correction order, final verification.

| System | Step-TCR | Trace-TCR | Process-valid rate |
|---|---|---|---|
| ReAct | 86.4 | 35.4 | 17.5 |
| PolicyGuard | 85.7 | 23.9 | **13.1** |
| PolicyGuide | **94.5** | **63.4** | **56.2** |

Process-valid = passes the outcome test *and* the rubric. Note PolicyGuard is **below** ReAct here. An action guard can produce a correct final state through a procedurally invalid route, and the standard metric cannot see it.

### The audit that found nothing

Call-NMR checks whether a successful mutation was missing a required earlier read. Airline: PolicyGuide 15.6%, ReAct 25.4%, PolicyGuard 32.5%. Retail: a tie at ~34.7%. Telecom: **0.0% for all three systems** — a ceiling effect. The oracle only knows agent-side reads, and telecom's requirements are conversational and user-device. The authors report it as a non-identification result rather than hiding it. Good practice.

### Cost

| Domain | verifier calls/task | prompt tok/call | cached | output tok/call | $/task |
|---|---|---|---|---|---|
| Airline | 7.56 | 32,360 | 88.1% | 2,478 | 0.40 |
| Retail | 7.42 | 22,803 | 85.8% | 2,179 | 0.34 |
| Telecom | 11.47 | 28,518 | 86.5% | 2,186 | 0.56 |

Caching kills most of the input cost, but output is 67–71% of spend because the verifier writes a long structured audit every turn. Wall-clock is **5.45–5.78× ReAct** (airline 36.4s → 210.1s per task).

## Worth Remembering

**The honest framing of the whole result.** Gains are largest where ordered requirements concentrate and vanish where they do not. That is a *conditional* claim about when to reach for this, and the authors built a 50-requirement hand classification to establish it rather than just reporting three numbers. That appendix is more reusable than the method.

**Retail is the domain to think about if you are deciding whether to build this.** 96% process-level, 3.6% ordered, +0.018 not significant. If your policy is a conjunction of flat gates checkable at the write, a dialogue-aware pass/block guard is enough and five times cheaper.

**The completion/refusal tension is real and measurable.** PolicyGuard's Retail PV goes up while Mut collapses; Gemini's PV goes down while Mut doubles. Any guard sits somewhere on that curve. Report both slices or the headline number is meaningless.

**Dual control is the structural argument.** In telecom the fix runs on the user's phone. There is no agent-side call to intercept. This is not a τ²-bench quirk — any domain where the resolution involves instructing a human has the same property, and action-local guarding is architecturally blind to it.

**Limitations the authors admit.**

- Advisory, not binding. The intercept is one-shot and disarms on retry, so Theorem 1's guarantee explicitly does not apply to the evaluated system. Verifier exceptions **fail open**.
- One frozen GPT 5.4-authored graph per domain. Deliberate, for fair comparison, but author-side generalisation is untested.
- The Telecom trace rubric is author-designed, uses text matching, and has **no second-annotator agreement estimate**.
- CRAFT is airline-only, 20 tasks, non-adaptive attacks. The Retail release did not reproduce cleanly and they say so.
- Three English customer-service domains, simulated users, $n=4$.
- Verifier calls see the full conversation and tool results, so privacy, retention and access-control requirements extend to verifier logs.

**Connections.** The reconcile step is [[Hidden Markov Model|dialogue-state tracking]] in all but name — persist a discrete position, update it from observations. The verifier-as-monitor framing is the classical reference monitor (Anderson 1972, Schneider 2000) with an LLM doing the predicate evaluation, which is also exactly where the guarantee breaks: each node judgement is probabilistic. The offline-compile / online-execute split rhymes with what [[DSPy]] does to prompts — the artefact is generated once and frozen, and the runtime is a fixed interpreter over it.

**Follow-up questions.**
- The verifier is the same model family as the agent. Would a much smaller verifier hold? Output tokens are 70% of cost, so a 4B verifier emitting a terser audit is the obvious ablation and it is not run.
- Firing schedule is a dial, not a constant. Corollary 2 says coverage is incomplete; nobody measured $\text{Pass}^4$ against calls-per-task.
- Does the graph survive policy drift? Real policies change weekly and the pipeline is six LLM stages plus manual verification.

**Practical caveats if you wanted to use this.** $0.40/conversation and 5.5× latency is fine for a back-office review queue and probably not fine for live chat. The graph must be reviewed by whoever owns the policy — the pipeline can both drop a real rule and invent one that is not there (the validator flagged `disable_roaming` as unauthorized, and manual review confirmed the graph was right and the environment was over-exposed). And the evidence rule is the part worth stealing even if you take nothing else: *a policy gate is never satisfied by the user's word.*

## Links

Related: [[Agentic Workflows]] · [[Guardrails]] · [[Human in the Loop]] · [[Tool Use]] · [[Prompt Injection]] · [[Jailbreak]] · [[Agent Evaluation]] · [[Evals]] · [[Observability and Tracing]] · [[Agent State and Checkpointing]] · [[Planning and Decomposition]] · [[Hidden Markov Model]] · [[Structured Output]] · [[Prompt Caching]] · [[Multi-Agent LLM Systems]] · [[LangGraph]] · [[DSPy]] · [[Cost and Latency]] · [[An Empirical Study of Harness Design for Coding Agents]] · [[WHALE- A Simple Recipe for Joint Harness-Weight Optimization]] · [[Guardrails]]

New topics worth writing: tau2-bench, reference monitor and enforceable security policies, runtime verification and LTL monitoring, dialogue state tracking, Pass^k as a reliability metric, standard operating procedure agents, red-teaming policy-adherent agents (CRAFT), McNemar's test for paired system comparison, prefix-closed languages and trace validity
