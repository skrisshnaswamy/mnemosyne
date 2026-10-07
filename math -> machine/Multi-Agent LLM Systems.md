---
aliases:
  - Orchestrator-Worker
  - Agent Handoff
  - Multi-Agent Orchestration
  - Agent Debate
  - Sub-agents
tags:
  - llm
  - agents
  - architecture
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Splitting work across several agents buys you **parallelism and clean context windows** — and costs you everything that doesn't survive a handoff, which is more than you think.
> **Metaphor:** A relay race where the runners don't pass the baton. They pass a note describing it, and each note is shorter than the last.
> **Where it bites:** "Specialised agents" are the same model in a different costume. The thing you actually bought was a fresh context window — so buy it deliberately.

---
You build a research crew. Four agents, clean roles:

```
Researcher  →  Analyst  →  Writer  →  Editor
```

The demo is beautiful. Two weeks later a report goes out saying *"revenue grew 12% in constant currency and 12% as reported."* Both can't be true.

The trace shows exactly where it happened:

```
Researcher found : "revenue grew 12% YoY excluding FX; reported growth 4%"
→ handed over   : "revenue grew 12% YoY"
Analyst wrote   : "strong double-digit revenue growth"
→ handed over   : "growth is strong"
Writer produced : "revenue grew 12% in constant currency and 12% as reported"
```

No agent hallucinated. Each one faithfully summarised what it was given. ~={blue}The qualifier died at the first handoff, and everything downstream was honest reasoning over a fact that had quietly changed meaning.=~

Now count: how many such qualifiers are in a real task?

---
# The relay with no baton

A relay race works because the baton is **the same physical object** all the way round. Runner 3 doesn't receive a description of the baton. She receives the baton.

![[multi_agent_relay_note.png]]

An agent handoff is the other thing. Agent 1 finishes with 40,000 tokens of context and hands agent 2 a 600-token message. That message is a **lossy compression chosen by agent 1**, who didn't know which detail agent 4 was going to need.

Put a number on it. Say a task depends on 8 specific facts and each handoff preserves any given fact with probability 0.94 — a *generous* summariser. Then:

```
1 handoff   →  0.94⁸  =  61% of runs still have all 8
3 handoffs  →  23%
6 handoffs  →   5%
```

Same shape as the [[Agentic Workflows#^compounding-error|compounding error]] multiplication, on a different quantity. And unlike a failed step, ~={red}a lost fact produces no error. It produces a fluent, confident, wrong report.=~

> [!SUCCESS] Core idea
> A handoff is a **lossy compression of the task**, and the compressor doesn't know what the decompressor will need. ~={pink}Share state; don't pass summaries.=~ Every agent reading and writing one structured object loses nothing, because nothing is ever re-described. ^handoff-is-lossy

> [!NOTE] Multi-agent system
> Several LLM loops, each with its own context window, prompt and tool set, coordinating on one task. The design decisions are: **who decides what runs next** (a fixed graph, an orchestrator, or peer handoffs), and **what crosses the boundary** (a message, or a shared state object). ^multi-agent-def

---
# The patterns, with a verdict

| Pattern | Shape | Good for | Verdict |
|---|---|---|---|
| **Orchestrator–worker** | One planner spawns workers, collects results | Independent sub-tasks: search 6 sources at once | **The one that earns its keep** 🥇 |
| **Sequential handoff** | A → B → C, each passing a message | Genuinely staged pipelines | Usually a [[Agentic Workflows\|workflow]] in disguise. Write the workflow |
| **Swarm / peer handoff** | Any agent can hand to any other | Customer routing (billing → tech → refunds) | Fine with **shared** transcript; lossy without |
| **Debate / critic** | Two agents argue; a judge picks | Ambiguous judgement calls | Rarely beats one good rubric. See [[Reflection]] |
| **Hierarchical** | Managers of managers | Nothing you have | ~={red}Never.=~ Compounding loss, cubed |
| **Blackboard** | All agents read/write one shared state | Long tasks with artefacts | Underrated. This is what [[LangGraph]] gives you |

> [!WARNING] Specialisation is a costume, not a capability
> "A researcher agent, an analyst agent and a writer agent" sounds like a team of experts. It is **one model with three different system prompts**. The weights are identical; none of them is better at analysis than the others.
>
> What you actually bought is real, but it isn't expertise — it's **context isolation**: a sub-agent can burn 40k tokens exploring and return 500, so the main loop never has to hold the mess. Buy that on purpose, and stop expecting the org chart to add IQ. ^specialisation-is-a-costume

---
# When one agent with more tools wins

Most of the time. The split has to pay for a handoff, an extra system prompt, extra tool schemas, and a second chance to lose the plot. It pays only when one of these is true:

- **Genuine parallelism** — six independent lookups, six workers, one round trip instead of six. This is the big one; see the 3.3× in [[Planning and Decomposition]].
- **Context isolation** — the sub-task generates enormous intermediate output that the parent must never see.
- **Different authority** — the worker that can `DROP TABLE` runs with different credentials from the one that reads the web. A boundary you're going to *enforce* is a real boundary. See [[Sandboxing]].
- **Different models** — a cheap model for extraction, an expensive one for the final judgement. That's routing, and it lives in [[Model Gateway]].

If none of those holds, ~={blue}add the tool to the agent you already have.=~

> [!TIP] Two rules that remove most multi-agent bugs
> 1. **One shared state object, not messages.** Workers write typed fields into it; the orchestrator reads fields. A fact written once is never re-described, so it can't degrade. The relay gets its baton back.
> 2. **Hand over evidence, not conclusions.** If a worker must summarise, make it return `{claim, source_id, verbatim_quote}` rather than prose. The qualifier in the opening scene survives a quote; it does not survive a paraphrase. 📎

> [!WARNING] Agent-to-agent messages are untrusted input
> Agent B reads whatever agent A wrote, in the same channel as its own instructions. If A's context was poisoned by a fetched web page, A's message is now the delivery vehicle — and B has no way to tell A's summary from B's system prompt. This is [[Prompt Injection]] with an extra hop and **no user in the room to notice**. Treat every inter-agent message as tainted data and keep per-agent permissions minimal. See [[Guardrails]]. ^agent-messages-untrusted

Related: [[Multi-Agent Reinforcement Learning]] and [[Dec-POMDP]] for the formal version of "several actors, partial views, one objective" — the credit-assignment problem there is the same one you'll have when a five-agent run produces a bad report and you need to know which agent caused it. Also [[The Handoff Tax- Continuing Non-Native Trajectories in LLM Agents]].

---

---
![[multi_agent_handoff_loss.png]]
> [!TIP] Reading the chart
> The grey line at the top is a shared state object: nothing is ever re-summarised, so nothing is ever lost. Every curve below it is the price of a conversation between agents — and even a 98%-faithful summariser is down to 38% by the sixth handoff.

---
> [!SUCCESS] If you remember one thing
> ~={pink}Share state; don't pass summaries.=~ Every handoff is a lossy compression chosen by an agent who doesn't know what the next one will need — and a lost fact doesn't raise an error, it produces a fluent, confident, wrong report.

---
# ⁉️
Some steps shouldn't be handed to *any* agent. Sending the email, moving the money, merging the PR — those need a person to say yes. Which means the loop has to be able to stop dead, wait, and then carry on as if nothing happened.

→ [[Human in the Loop]]
