---
aliases:
  - Context Compaction
  - Context Management
  - Context Pruning
  - Context Curation
tags:
  - llm
  - agents
  - context
  - engineering
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Prompt engineering writes the instruction **once**; context engineering decides, **every single turn**, which finite set of tokens the model is allowed to see.
> **Metaphor:** Re-packing a rucksack at every camp on a long hike. Fixed volume, and the thing you drop at camp 3 is the thing you need at camp 7.
> **Where it bites:** Turn 40 of an agent run. The instructions still work; the window they live in no longer does.

---
An agent is 40 turns into a task. Here's what is actually in its 128k window:

```
system prompt                 1,200 tokens   ← still fine
tool schemas (14 tools)       2,800
the user's original request     150           ← buried 95k tokens back
turn 1–40 thoughts + calls    7,200
turn 1–40 tool results       88,600           ← 89% of everything
                            ────────
                             99,950 tokens, and turn 41 adds 2,400 more
```

At $3 per million input tokens, **turn 40 alone costs $0.30** and turn 41 costs more. At this rate the window is full at turn 52 and the run simply stops.

But the bill and the wall aren't the interesting part. The interesting part is that ~={blue}the model is now reading 88,600 tokens of tool output to find the 150 tokens that said what the job was.=~

Nobody wrote a bad prompt. So what exactly is the thing that went wrong?

---
# The rucksack

You're eight days into a hike. Fixed pack, fixed volume, and at every camp you unpack and re-pack it.

Day 1, everything goes in: it's all potentially useful. By day 3 the pack contains a damp map of a valley you've left, four empty ration wrappers, the guidebook page for a section you've walked, and — somewhere at the bottom — the permit you'll need at the gate on day 8.

![[context_engineering_rucksack.png]]

Re-packing isn't an optimisation you do when the pack is full. It's **the job**, done at every camp, and it has exactly three moves: *keep it*, *throw it out*, or **write it down somewhere else and carry the note**.

Notice the third one, because it's the one people forget. The guidebook page you'll need on day 8 doesn't have to be in the pack. It has to be *retrievable* on day 8.

> [!NOTE] Context engineering
> The per-turn discipline of deciding what occupies the model's finite window: what is **loaded** (instructions, tools, retrieved documents), what is **kept** from earlier turns, what is **compressed**, what is **evicted to external storage**, and in **what order** it all sits. Prompt engineering is a subset of it — the part about the instruction itself. ^context-eng-def

> [!SUCCESS] Core idea
> The window is not a buffer that fills up; it's a **budget you re-allocate every turn**, and the allocation is a *product decision*. ~={pink}Prompt engineering asks "what should I say?" Context engineering asks "what is in the room when I say it?"=~ Past a handful of turns, the second question dominates the first. ^context-is-a-budget

---
# What actually occupies the window, and the lever for each

| Occupant | Grows with | The lever |
|---|---|---|
| System prompt | Nothing | Write it once. Keep it **prefix-stable** — see [[Prompt Caching]] |
| Tool schemas | Number of tools | ~200 tokens each. 14 tools ≈ 2,800 tokens **every turn**. Load a subset per phase |
| Retrieved documents | Top-$k$ × chunk size | [[Reranking]] to 3–5. See [[Chunking]] |
| Conversation history | Turns | Rolling summary, or a sliding window |
| **Tool results** | Turns × verbosity | ~={red}The one that actually kills you.=~ Truncate at the boundary, not in the model |
| The plan / TODO | Nothing, if you rewrite it | Keep **one** current copy. Never append plan revisions |

Tool results dominate because nobody budgets them. A `list_files` on a real repo, an unpaginated API response, a stack trace, a 200-row SQL result — each is 2–8k tokens, arrives 40 times, and is 95% irrelevant 3 turns later.

> [!TIP] Truncate at the tool boundary, not in the prompt
> Cap every tool's return in the **executor**: head/tail the output, paginate, or write the full thing to a file and return `{"rows": 4210, "preview": [...], "file": "/tmp/q7.json"}`. The agent can re-read the file if it needs to. This converts ~={blue}an unbounded, expensive, lossy context problem into a bounded, cheap, exact file problem=~ — the same move [[Memory|the scratchpad]] makes. 📝

---
# The four strategies, and when each is wrong

| Strategy | What it does | Loses | Use when |
|---|---|---|---|
| **Sliding window** | Keep the last $k$ turns verbatim | The original goal, silently | Chat. **Never** for agents |
| **Summarise / compact** | Every $k$ turns, an LLM call rewrites history into a brief | Detail, irreversibly, and it can hallucinate the summary | Long agent runs — the default |
| **Truncate tool results** | Cap each observation at ingest | Nothing you needed | **Always.** Do this first |
| **Externalise + retrieve** | Write to files/store, pull back on demand | Nothing; costs a retrieval | Tasks with many artefacts |

The numbers from the simulation below: appending everything reaches **100k tokens at turn 40** and hits a 128k window at **turn 52**. Compacting every 10 turns holds the same run at **12k tokens** at turn 40 — turn 40 costs **$0.036 instead of $0.30**, an 8× difference on identical work.

> [!WARNING] A bigger window is not the fix, and this is the misconception worth killing
> The instinct is: 1M context, problem solved. Three things say otherwise.
> 1. **The deadline moves; it doesn't go away.** At 2,400 tokens a turn, 1M buys you turn 415 instead of turn 52. Long-running agents blow through it.
> 2. **You pay for every token, every turn.** The same run at 1M context costs roughly 8× more, for *worse* answers.
> 3. **Quality degrades long before capacity does.** Accuracy is U-shaped in position ([[Lost in the Middle]]) and the effect gets worse as the context grows — the phenomenon people call **context rot**. A full 1M window is not a well-read 1M window. See [[Long Context]].
>
> ~={red}Capacity and comprehension are different curves, and the second one bends first.=~ ^bigger-window-is-not-the-fix

**Compaction has a failure mode of its own**, and it's nasty: the summarising call is an LLM call, so it can drop the one constraint that mattered ("the client said no PostgreSQL") and the agent then confidently violates it with no trace of why. Mitigation: never summarise the **goal** or **hard constraints** — pin them verbatim at the top of every compaction, and summarise only the middle.

> [!TIP] The ordering is free and you get it wrong by default
> Stable, cacheable, important → **top**. Volatile, bulky, disposable → **middle**. The current question and the current instruction → **bottom**, immediately before generation. Most frameworks append everything in arrival order, which puts your freshest tool dump exactly where attention is strongest and your goal exactly where it is weakest. 🎯

Related: [[Context Window]] for what the window *is*, [[Memory]] for the three storage tiers this feeds, [[Agentic Workflows]] for the loop that fills it, and [[Cost and Latency]] for what the re-sent tokens cost.

---

---
![[context_growth_compaction.png]]
> [!TIP] Reading the chart
> The red line is what every agent does on day one, and it has a deadline written into it. The green sawtooth is the same run, same tools, same task — it just re-packs the rucksack every ten turns.

---
> [!SUCCESS] If you remember one thing
> The window is a budget you re-allocate every turn, not a buffer that fills up. ~={pink}A bigger window moves the deadline; it never removes it.=~

---
# ⁉️
Compaction throws detail away. Some of it needs to survive the run entirely — a user's preferences, what was decided last month, the fact that this customer is on enterprise. That isn't a window problem at all.

→ [[Memory]]
