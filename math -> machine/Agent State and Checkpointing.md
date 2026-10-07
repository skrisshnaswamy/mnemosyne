---
aliases:
  - Durable Execution
  - Agent State
  - Checkpointer
  - Resumability
  - Agent Threads
tags:
  - llm
  - agents
  - engineering
  - infrastructure
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Treat the run's state as a **first-class object you persist after every step**, so a crash at step 14 of 20 resumes at step 15 — and the transcript becomes a *view* of that state, not the state itself.
> **Metaphor:** Save points in a long game. The save doesn't store the pixels; it stores your position, your inventory, and which doors are already unlocked.
> **Where it bites:** The demo runs in one process for 40 seconds. Production runs for four hours across three deploys and a pod eviction.

---
A migration agent, 20 steps. Step 14 finished: it wrote **4.2 million rows** into the new table and the run was nearly done.

At step 15 the pod was evicted for a node upgrade.

You restart it. It begins at step 1.

It re-reads the schema, re-derives the mapping, and at step 14 it writes 4.2 million rows into a table that already has 4.2 million rows. Now you have a data problem *and* a migration problem.

Put the cost in numbers. At ~£0.04 per step, with a crash landing uniformly anywhere in a 20-step run:

```
no checkpoints        → 9.5 steps re-executed on average, £0.39 burnt per crash
checkpoint every step → 0 steps re-executed
```

The money is the least of it. ~={red}The re-execution is the outage.=~ And notice — nothing about the model was wrong. Nothing about the prompt was wrong.

So what, precisely, did this system fail to have?

---
# The save point

A 40-hour game. You're six hours into chapter 3 when the power goes.

![[agent_state_save_point.png]]

If the game saves at the **end of each chapter**, you lose six hours. If it saves at every door, you lose ninety seconds. Same crash, entirely different day.

Now the part that matters. When the game saves, what does it write? Not the pixels on screen, not the animation frame. It writes:

```
position:        chapter 3, east tower
inventory:       [brass key, 3 potions]
flags:           {bridge_repaired: true, guard_bribed: true}
quest:           "find the archivist" — step 2 of 4
```

**Position, possessions, and what's already irreversibly happened in the world.** The bridge is repaired. Reloading must not un-repair it, and must not make you repair it again.

That last line is the whole idea, and it's what the migration agent didn't have.

> [!NOTE] Agent state
> The typed object that fully determines what happens next: the message history, the scratchpad and artefacts, the current step or node, any pending tool call, **and the record of side effects already committed to the outside world**. A **checkpointer** persists it after every step, keyed by a thread id. ^agent-state-def

> [!SUCCESS] Core idea
> The transcript is **not** the state — it's a rendering of it. Replaying a transcript replays the *conversation*; it does not tell you that 4.2 million rows are already written. ~={pink}State is what happened; the transcript is what was said about it.=~ Persist the first and derive the second. ^state-not-transcript

---
# What to put in it

| Field | Why it's there | What breaks without it |
|---|---|---|
| `thread_id` | The unit of resumption | You can't find the run |
| `messages` | Model context for the next call | Amnesia |
| `step` / `node` | Where to restart | Restarts at the top |
| `scratchpad` / artefacts | The work product so far | Redoes the thinking |
| `pending_call` | A tool call emitted but not yet confirmed | Double-executes or drops it |
| `effects[]` | `{tool, idempotency_key, result}` for everything already committed | **Re-sends the email** |
| `budget` | Steps, tokens and money spent | Resets the limits on every resume |

The `effects` table is the one nobody builds first and everybody builds second. It's what makes a resume **idempotent**: before executing, check whether this exact call, with this key, already ran. If it did, return the stored result and move on.

> [!TIP] Idempotency keys are cheap and you will need them anyway
> Derive a key deterministically from `(thread_id, step, tool, arguments)` and pass it to every side-effecting tool — most real APIs (Stripe, SendGrid, GitHub) accept one natively. Then "did this already happen?" is a lookup rather than an archaeology project. ~={blue}A retry you can't distinguish from a first attempt is not a retry; it's a second action.=~ 🔑
> The same discipline appears again in [[Retries and Fallbacks]], for the same reason.

---
# The ladder of durability 🪜

| Level | What it is | Survives | Cost |
|---|---|---|---|
| **In-memory loop** | A `while` in a request handler | Nothing | Zero. Fine for a demo |
| **Transcript in a DB** | Save the messages, reconstruct on load | Process restart | Low. **Re-runs side effects** |
| **Checkpointer per step** | Typed state written after every node | Crash, deploy, [[Human in the Loop\|4-hour pause]] | A table. **The default you want** 🥇 |
| **Durable execution engine** | Event-sourced workflow with replay (Temporal-style) | Everything, including partial failure inside a step | An extra system. For multi-day, money-moving work |

A checkpointer buys you four things that look unrelated and are the same feature:

1. **Crash recovery** — resume at step 15.
2. **Human interrupts** — pause for four hours, resume from the identical point.
3. **Time travel** — load the checkpoint from step 9, change one thing, run forward. This is how you debug a non-deterministic system: not by reading logs, but by **re-entering the run**.
4. **Forking** — run two different continuations from step 9 and compare. That's an eval harness for free — see [[Agent Evaluation]].

> [!WARNING] Checkpointing the messages is not checkpointing the state
> The common half-build: persist the message list, resume by replaying it. The model then re-emits the tool call it emitted before, and your executor happily runs it again. ~={red}You have restored the conversation and re-performed the actions.=~
>
> The fix is the `effects` ledger above, written **before** the side effect, confirmed after — the same write-ahead-log discipline a database uses, for the same reason. ^replay-reruns-effects

> [!TIP] The checkpoint interval is a real dial
> Persisting after every step is usually right: a state object is kilobytes and the write is microseconds against a model call of seconds. Checkpointing every 5 steps re-executes **2.0 steps** per crash on average; every 10 steps, **4.5**; never, **9.5**. Unless your state is huge, take the every-step end and stop thinking about it. 💾

Related: [[Human in the Loop]] (the feature that *requires* this), [[LangGraph]] (whose checkpointer is the reason people pick it), [[Observability and Tracing]] for seeing which step a run died on, and [[Beliefs]] / [[State-Space Model]] for the older idea that a system's future depends only on a sufficient summary of its past — [[Markov Property|exactly what a good state object is]].

---

---
![[checkpoint_interval_redo.png]]
> [!TIP] Reading the chart
> This is not a curve about reliability — it's a curve about *waste*. The crash rate is the same at every point on the x-axis. All you're choosing is how much work you agree to do twice.

---
> [!SUCCESS] If you remember one thing
> ~={pink}The transcript is what was said; the state is what happened.=~ Persist the second, derive the first, and keep a ledger of the side effects already committed to the world.

---
# ⁉️
Resumable, checkpointed, approved by a human — and at step 12 the agent decides the cleanest way to reformat those files is a shell script it just wrote. Where, exactly, does that run?

→ [[Sandboxing]]
