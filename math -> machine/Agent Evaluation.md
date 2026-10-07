---
aliases:
  - Trajectory Evaluation
  - Agent Evals
  - Trajectory Grading
  - Tool Choice Accuracy
tags:
  - llm
  - agents
  - evaluation
  - metrics
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Grading the final answer of an agent is grading a maths exam on the number at the bottom — you need the **trajectory**, the **tool choices**, and the **cost** as first-class scores.
> **Metaphor:** Two students both wrote 42. One derived it; one guessed. Mark the answer only and you cannot tell — and next week's question is different.
> **Where it bites:** A 30-case agent suite cannot detect a 10-point regression. Most agent suites have about 30 cases.

---
Two runs of the same agent, same question: *"what did we spend on cloud in Q2?"*

```
Run A   4 steps   £0.02    query_warehouse(...) → answered, cited the query
Run B  19 steps   £0.61    guessed · searched the web · found a 2024 figure ·
                           doubted it · searched again · found the invoice ·
                           answered the same number
```

Both final answers are correct. A final-answer grader scores them **1.0 and 1.0**.

Now: which one do you want in production? Which one still works when the web result changes? Which one's p99 is 4 seconds and which one's is 90? Which one costs £61 per hundred users?

~={blue}Your eval said these systems were identical. They differ by 30× in cost and by everything in reliability.=~

This note builds on [[Evals]] — fixed inputs, a grader, a tracked history. What changes for an agent is *what the grader can see*.

---
# Marking the working, not the answer

An exam that only marks the final number has a specific pathology, and every teacher knows it: it cannot distinguish understanding from luck, so it cannot **predict** next week's performance.

![[agent_eval_answer_vs_working.png]]

Show your working and four new things become markable:

1. **Did they use the right method?** ← *tool choice*
2. **In a sensible order?** ← *trajectory*
3. **Was each line valid given the one above?** ← *per-step*
4. **Did it take four lines or forty?** ← *efficiency*

And there's a fifth, which is the one that decides whether you can ship: with working shown, a wrong answer tells you **where** it went wrong. A wrong number tells you nothing.

> [!SUCCESS] Core idea
> A single final-answer score compresses a 19-step run into one bit, and ~={pink}throws away the only information that would let you fix it or predict it.=~ Grade the destination *and* the route *and* the fare — three numbers, and the first one alone is the least informative of the three. ^grade-the-route-too

---
# The four things to grade

| Level | What it asks | Grader | Cost |
|---|---|---|---|
| **Outcome** | Is the final answer right? | Exact match, or a judge with a rubric | £ |
| **Tool choice** | Right tool, right arguments? | **Exact match — it's a classification problem** ✅ | Free 🥇 |
| **Trajectory** | Did the required steps happen, in a workable order? | Set/sequence comparison against a reference | Free |
| **Efficiency** | Steps, tokens, cost, wall clock | Arithmetic | Free |

**Tool choice is the cheapest high-value metric in this whole area and almost nobody tracks it.** "Given this question, which tool should the first call be?" is a labelled classification problem over your own traces. You can build 200 cases in an afternoon, grade them with `==`, and it catches the most common real regression there is: *you added a fifteenth tool and selection accuracy fell for the other fourteen.* See [[Tool Use]].

**Trajectory matching** needs one decision, and exact-sequence matching is the wrong one — there are usually several valid routes. The useful variants:

```
in-order subset  : did [search, read, cite] appear in this order?  ← the default
any-order set    : were all required tools called at all?
precision/recall : over tool calls — recall = did it do what it needed,
                   precision = how much did it do that it didn't need
forbidden set    : did it call anything it must not have? (writes, deletes)
```

That last line is a **safety** eval, not a quality one, and it should be a hard gate: any run that calls a forbidden tool fails, whatever its answer was.

---
# The statistics, which are worse for agents 📉

From the Wilson intervals below, at a true pass rate of 80%:

```
n =  30  →  80% ± 13.9 points   [63, 90]
n = 100  →  80% ±  7.8 points   [71, 87]
n = 400  →  80% ±  3.9 points   [76, 84]
```

A 30-case suite genuinely cannot tell 70% from 88%. Teams ship on that difference every week.

And agents have a **second** variance source: the same case run twice gives different trajectories. So you need $n$ cases *and* $k$ repeats, and the repeats matter most where the pass rate is near the middle:

> [!TIP] Budget the suite properly
> - **~100–200 cases minimum** for a headline pass rate you'd act on. ~={red}30 is a smoke test, not a measurement.=~
> - **$k = 3$ repeats** per case, and report **pass@1 and pass@k separately** — pass@3 that's much higher than pass@1 means your agent is *lucky*, not capable, and that gap is the number to drive down.
> - **Tier it**: 20 cases on every commit (2 minutes), the full suite nightly, the expensive judge weekly.
> - Compare versions **paired** — same cases, same seeds, look at the per-case deltas. A paired comparison detects a real change at a fraction of the sample size an unpaired one needs. 📊

---
# Where the cases come from

> [!NOTE] Production traces are the only sustainable source
> Every run you already have in [[Observability and Tracing|tracing]] is a real input with a real trajectory and a real cost. The pipeline is boring and it works:
> ```
> user thumbs-down / human gate rejected / guardrail blocked / cost outlier
>      → triage weekly → label the correct trajectory → add to the suite
> ```
> A case written by an engineer imagining a user is worth perhaps a tenth of a case a user actually produced. And the failures cluster: the same six shapes account for most of it. ^cases-from-traces

Four categories that are almost always missing, and each maps to a failure you've already had:

| Category | Example case | Catches |
|---|---|---|
| **Tool errors** | The search tool returns a 500 | Doom loops, swallowed errors |
| **Ambiguity** | "How much did we spend?" (which period?) | Agents that guess instead of asking |
| **Out of scope** | "Book me a flight" to a support bot | Over-answering; the [[Evals#^eval-case-sources\|missing refusal cases]] |
| **Adversarial** | A retrieved doc containing "ignore previous instructions" | [[Prompt Injection]] — run these on every release |

> [!WARNING] Don't grade the reasoning prose
> The obvious move is to have a judge score the agent's *stated* thinking. Resist it: the [[Chain of Thought#^cot-unfaithful|written reasoning may not be why it acted]], so you'd be optimising a post-hoc narrative. A model that learns to write convincing rationales for bad tool calls is a [[Reward Hacking|worse]] system that scores better.
>
> Grade **what it did** (the calls, the arguments, the result) and **what it produced**. Those are observable. ~={red}The reasoning is a story about the action, not the action.=~ ^dont-grade-the-narrative

> [!TIP] Put efficiency in the headline, not the footnote
> Report cost and steps **beside** accuracy, always: `82% pass · 6.1 steps · £0.09 · 11 s p95`. The moment accuracy is the only visible number, every change that trades 4× the cost for one point gets shipped, and a quarter later nobody can explain the invoice. *An eval that doesn't show cost will select for expensive agents.* 💸

Related: [[Evals]] (the foundations — graders, judges, benchmarks), [[Observability and Tracing]], [[Agent State and Checkpointing]] (fork a checkpoint to A/B a single decision), [[Planning and Decomposition]], [[Cost and Latency]], [[DRACO- Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training]], [[What Makes Good Agentic Data- An ACE Lens on Data Generation for LLM Agents]].

---

---
![[eval_sample_size_ci.png]]
> [!TIP] Reading the chart
> The suite you have is probably at the red dot. To call a five-point improvement you need to be past the green one — and for an agent, with repeats on top of that.

---
> [!SUCCESS] If you remember one thing
> Grade the destination, the route **and** the fare. ~={pink}An eval that doesn't show cost will quietly select for expensive agents=~ — and thirty cases cannot tell 70% from 88%.

---
# ⁉️
Every eval above assumes you can tell a right answer from a wrong one when you see it. The hardest failure in this whole field is the one that defeats exactly that assumption — output that is fluent, well-formed, correctly cited, and untrue.

→ [[Hallucination]]
