---
aliases:
  - Self-Critique
  - Self-Correction
  - Self-Refine
  - Critic Loop
tags:
  - llm
  - agents
  - failure-mode
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Asking a model to review its own answer is a **channel, not a signal** — it only improves things by as much as whatever is behind it, and behind it is usually nothing.
> **Metaphor:** Proofreading your own essay. You read what you meant to write; a stranger reads what's on the page.
> **Where it bites:** "Add a reflection step" is the most commonly recommended and least effective agent fix in circulation.

---
A model writes this:

```python
def last_n(xs, n):
    return xs[len(xs) - n + 1:]
```

Ask it: *"Review your answer carefully. Is it correct?"*

> *"Yes — the function slices from `len(xs) - n + 1` to the end, returning the final `n` elements. This is correct."*

Now paste in one line of output instead:

```
>>> last_n([1,2,3,4,5], 2)
[5]        # expected [4, 5]
```

> *"The slice is off by one. It should be `xs[len(xs) - n:]`."*

Instantly, confidently, correctly. Same model, same weights, same context — the only thing that changed was **five characters of ground truth** arriving from outside.

So what was the "review carefully" prompt actually doing?

---
# Proofreading your own essay

You've written three pages and you read them back for typos. You will miss things, and not randomly: you'll miss the ones where you know what the sentence was supposed to say, because your eyes supply the intended word over the written one.

Hand the same page to someone else and they find it in ten seconds. Not because they're a better reader — because they don't have your intention in their head.

![[reflection_proofreading_own_essay.png]]

A model reviewing its own output is in exactly the first position. The generation that produced the bug and the critique that's supposed to catch it come from the **same distribution, with the same blind spots, conditioned on the same context**. If it had known the slice was wrong, it would have written it right.

> [!SUCCESS] Core idea
> Reflection doesn't produce information; it *routes* information back into the context. ~={pink}The gain is set entirely by the quality of the signal you feed it=~ — a test result, a schema validation, a compiler, a second model with different evidence. With no external signal, you're paying two extra calls to have the model re-read its own confidence. ^reflection-is-a-channel

> [!NOTE] Reflection
> A loop in which the output of a generation step is fed back as input to a critique step, whose verdict drives a revision. Three parts, and only one of them is usually examined: the **generator**, the **critic** (what does it see that the generator didn't?), and the **revision** (can it act on the critique?). ^reflection-def

---
# The arithmetic, which settles the argument 🧮

One round of critique-and-revise changes accuracy by exactly this much:

```
new accuracy = accuracy
             + (wrong answers the critic catches, and the revision fixes)
             − (right answers the critic falsely flags, and the revision breaks)
```

In symbols, with $p$ = accuracy, $d$ = P(critic flags a wrong answer), $a$ = P(critic flags a right answer), $f$ = P(a revision lands):

$$p' = p + (1-p)\,d\,f - p\,a\,f$$

**Self-critique.** Start at $p = 0.55$. The model catches about a third of its own errors ($d = 0.33$) and second-guesses a fifth of its correct ones ($a = 0.22$), revisions land 70% of the time:

$$0.55 + (0.45)(0.33)(0.70) - (0.55)(0.22)(0.70) = 0.55 + 0.104 - 0.085 = \mathbf{0.569}$$

**Under two points.** For roughly 2.5× the tokens.

**An external verifier** — run the tests. Now $d = 0.96$, $a = 0.02$:

$$0.55 + (0.45)(0.96)(0.70) - (0.55)(0.02)(0.70) = 0.55 + 0.302 - 0.008 = \mathbf{0.845}$$

~={red}Thirty points, from the same loop.=~ The loop was never the variable. $d$ and $a$ were.

> [!WARNING] The false-alarm term is why reflection can make things *worse*
> Look at the subtraction. If the critic flags correct answers at a decent rate and the model is already good, $p\,a\,f$ exceeds $(1-p)\,d\,f$ and accuracy **falls**. This is not hypothetical — it's the standard outcome when you bolt self-critique onto a task the model was already 85% good at. A confident critic with no evidence is a random-number generator with a veto. ^reflection-can-hurt

---
# Where the signal can come from, ranked

| Signal | $d$ (catches errors) | Cost | Verdict |
|---|---|---|---|
| **Unit tests / compiler / type checker** | ~0.95+ | Free ✅ | **The gold standard.** This is why coding agents work |
| **Schema / JSON validation** | ~1.0 on format | Free ✅ | Always on. See [[Structured Output]] |
| **A real execution** — run the SQL, hit the endpoint | ~0.9 | Cheap | Under-used outside code |
| **Retrieved evidence** — check each claim against a source | ~0.7 | £ | [[Grounding]]; the best option for prose |
| **A second model with *different context*** | ~0.5 | £ | Works because the blind spots differ, not because there are two |
| **Same model, same context, "check your work"** | ~0.3 | £ | ~={red}Barely moves. This is what most "reflection" is=~ |

Notice the pattern: everything above the line is a **verifier**, and the asymmetry it exploits is the same one behind [[Test-Time Compute]] — *checking is cheaper than producing.* Where that asymmetry exists, reflection is transformative. Where it doesn't, reflection is theatre.

> [!TIP] The one diagnostic
> Before adding a reflection step, answer this: **what will the critic see that the generator did not?** If the honest answer is "nothing, but it'll be looking more carefully" — ~={blue}delete the step and spend the tokens on retrieval, a test, or a better prompt instead.=~ 🔍

**Two things that genuinely help without a hard verifier:**
- **Fresh eyes.** Run the critique in a *separate* call with only the output and the requirements — not the generation transcript. Stripping the reasoning that produced the error removes some of the shared blind spot. It raises $d$ and lowers $a$.
- **A rubric.** "Check it" is a vibe. "Check: does every numeric claim appear verbatim in the sources? Is every required field present? Is any instruction unaddressed?" is a checklist, and checklists find things. The same discipline as [[Evals#^judge-rules|LLM-as-judge rubrics]].

And a hard cap: **two rounds**. The chart below shows the curve is flat after that — you're re-litigating, not improving.

Related: [[Agentic Workflows]] for the loop, [[Agent Evaluation]] for grading the whole trajectory, and [[Chain of Thought#^cot-unfaithful|why the stated reasoning may not be the real reasoning]].

---

---
![[reflection_vs_verifier.png]]
> [!TIP] Reading the chart
> Both lines are "add a reflection step." The red one is the version that gets recommended; it buys **1.6 points** in the first round and then flattens. The green one is the same loop with a test suite behind it: **55% → 84%** in one round.

---
> [!SUCCESS] If you remember one thing
> Before adding a critique step, answer one question: **what will the critic see that the generator did not?** ~={pink}If the honest answer is "nothing", you've built a loop, not a check.=~

---
# ⁉️
The cheapest external signal for anything factual isn't a test — it's a document. Which means the loop should be able to *go and get one*, mid-task, as an action it chooses, rather than being handed five chunks up front and told to cope.

→ [[Agentic RAG]]
