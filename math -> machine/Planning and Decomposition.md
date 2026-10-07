---
aliases:
  - Task Decomposition
  - Plan-and-Execute
  - Agent Planning
  - Task Graph
tags:
  - llm
  - agents
  - architecture
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Write the whole route down before you leave, or decide at each junction — and the right answer is **neither**: plan a horizon, execute one step, re-plan.
> **Metaphor:** A driver with a printed route and a windscreen. The route is worth having; obeying it after the road closes is not.
> **Where it bites:** Planning doesn't make an agent smarter. It makes runs **shorter and parallel** — and that's the only reason it helps.

---
Give an agent a real job: *"migrate our billing service off the legacy Stripe API."*

**Run A — decide at each junction.** It reads a file, greps, edits, runs tests, reads the failure, edits again. 34 tool calls. On call 19 it discovers the `subscriptions` table has no `price_id` column — an assumption it silently baked in at call 4. Fifteen calls of work, wrong.

**Run B — plan first.** It writes a 9-step plan in one call, then executes. Step 3 fails for the same reason. But now the plan *says* step 4 depends on step 3, so it stops there instead of building on sand.

Now the arithmetic that governs both. At 95% reliability per step:

```
34 steps  →  0.95³⁴  =  17% chance the whole run is right
 9 steps  →  0.95⁹   =  63%
```

Same model. Same tools. Same task. ~={blue}The single largest lever on whether an agent run succeeds is how many steps it takes.=~

So: is the answer always to plan?

---
# The route and the windscreen

Two drivers, same city, same destination.

**The first** prints the route before leaving. Twelve instructions on a sheet. She never looks up — at instruction 7 she turns into a road that has been closed for resurfacing since Tuesday, and the sheet has nothing to say about it.

**The second** takes no route at all and decides at every junction: *which of these looks like it goes the right way?* He never hits a closed road he can't handle. He also spends forty minutes in a residential grid going gently north-east.

![[planning_route_vs_junction.png]]

The plan's value was never that it was correct. It was that it **committed to an order**, so the second driver's forty minutes of local wandering never happened. And its cost was that it committed to an order *before seeing the road*.

What both drivers actually want is a windscreen and a plan: look as far ahead as you can see, commit to the next move, and redraw when the view changes.

> [!SUCCESS] Core idea
> Planning adds **no information**. Everything in the plan was already derivable at step 0. What it adds is ~={pink}commitment, ordering, and the ability to notice that step 4 depended on step 3=~ — which is worth a great deal precisely because it shortens the run and exposes what can run in parallel. ^planning-adds-no-information

#### 🎯 The callback
That "look ahead, commit to one step, re-plan" shape is not new. It is exactly [[Model Predictive Control]] — plan over a receding horizon, execute the first action only, throw the rest of the plan away, re-plan from the new state. The control literature settled this argument decades ago, and for the same reason: *the model of the world is wrong, so a plan's value decays with how far into it you are.*

> [!NOTE] The three shapes
> - **Plan-then-execute** — one planning call, then execute the list. Cheap, auditable, brittle.
> - **Interleaved (ReAct)** — no plan; decide after each observation. Adaptive, long, drifts. See [[Agentic Workflows]].
> - **Plan, execute, re-plan** — plan a horizon, execute one or two steps, re-plan on new information or on failure. ~={blue}The default worth starting from.=~ ^planning-shapes

---
# The task graph — where the real win is

A plan written as a numbered list is a **sequence**. A plan written as a graph is a **dependency structure**, and that changes what's possible:

```
1. read schema            ─┐
2. list Stripe webhooks   ─┼─→  4. write migration  →  5. run tests  →  6. open PR
3. find price mappings    ─┘
```

Steps 1–3 don't depend on each other. A sequence runs them in 3 round trips; a graph runs them in **1**. On the measured tool-latency distribution in [[Cost and Latency]], four independent calls take **9.2s sequentially and 2.8s in parallel** — 3.3×, for free, purely because the plan said they were independent.

And the depth of the graph, not the number of nodes, is what the reliability multiplication acts on. Six nodes in a chain is $0.95^6 = 74\%$; six nodes with three in parallel is depth 4, $0.95^4 = 81\%$.

> [!TIP] Decompose until each step is verifiable
> The right size for a sub-task isn't "small". It's **"a machine can tell me whether this worked."** A step whose success you can only judge by reading it is a step you cannot checkpoint, retry, or grade. If you can't write the check, the step is still too big. ✅

---
# When planning ahead is the wrong move

| Situation | Plan first? | Why |
|---|---|---|
| Steps are independent and known | ✅ **Yes** — a graph | Parallelism, and depth < length |
| Long task, expensive steps | ✅ Yes, with re-planning | Catch contradictions before spending |
| Each step's *existence* depends on the last result | ❌ No | You'd be planning fiction |
| Exploring an unfamiliar codebase or API | ❌ No | The plan encodes assumptions you haven't tested |
| Cheap, fast, reversible steps | ❌ No | Planning costs more than the wandering |
| Anything with a human approval gate | ✅ Yes | The plan **is** the thing the human approves — [[Human in the Loop]] |

> [!WARNING] The plan becomes the most dangerous thing in the context
> Two failure modes, both common:
>
> **Plan lock-in.** The plan is in the window, it reads like an instruction, and the model follows it past the point where it stopped being true. A plan and a closed road, and the model picks the plan. Fix: re-planning must be an explicit, *cheap*, permitted action — and the prompt must say the plan is a hypothesis.
>
> **Plan accumulation.** Revision 1, revision 2 and revision 3 all sit in the context, and the model averages them. ~={red}Keep exactly one plan and overwrite it=~ — in the state object, not the transcript. See [[Context Engineering]] and [[Agent State and Checkpointing]]. ^plan-lock-in

Related: [[Monte Carlo Tree Search]] (planning when you *can* simulate the outcome — LLM agents usually can't), [[Value and Policy Iteration]] for what planning means when you have a model, and [[Test-Time Compute]] for the "think longer" axis this sits on.

---

---
![[agent_compounding_reliability.png]]
> [!TIP] Reading the chart
> The red curve is why nobody's 20-step demo survived the quarter. The green curve is the same 95%-per-step model with a check every five steps that catches 80% of failures — **70% instead of 36%**. Shortening the run and checking it are the same lever pulled twice.

---
> [!SUCCESS] If you remember one thing
> Planning adds no information — it shortens the run and exposes what can happen at once. ~={pink}Step count is the only lever in an agent that acts quadratically.=~

---
# ⁉️
A check every five steps assumes something can *do* the checking. The cheapest thing to reach for is the model itself — ask it to review its own work. That turns out to be a much weaker move than it looks.

→ [[Reflection]]
