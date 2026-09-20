---
aliases:
  - Positional Bias
  - Needle in a Haystack
  - Context Rot
  - U-shaped attention
tags:
  - llm
  - context
  - failure-mode
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Accuracy is **U-shaped** in position — a fact at the start or end of a long context is found; the same fact in the middle is often missed.
> **Metaphor:** A long meeting. You remember the opening and the last five minutes. The 40 minutes in between are gone.
> **Where it bites:** Ordering your [[RAG]] chunks is a free accuracy lever. And "1M context" is a capacity claim, not a comprehension one.

---
Run this experiment yourself; it takes ten minutes.

Take 20 documents. Put a specific fact in **exactly one** of them — *"the internal build server is named Corvid."* Fill the [[Context Window]] with all 20 and ask: *"What is the internal build server named?"*

Now vary **only which position** the fact-bearing document sits in.

```
position  1  of 20  →  94% correct
position  5  of 20  →  73%
position 10  of 20  →  ~68%   ← the floor
position 15  of 20  →  76%
position 20  of 20  →  91%
```

Identical information. Identical prompt. Identical token count. The **only** variable is where in the list it sat — and accuracy swings by 26 points.

Worse: sometimes the model with 20 documents does worse than the model with **just the one relevant document**. More context made it dumber.

Why would position matter at all? [[Causal Attention|Attention]] can reach any token in the window.

---
# The long meeting

You sat through a two-hour meeting. A week later, what do you actually have?

The **opening** — the agenda, the framing, why you were all there. And the **last five minutes** — the decisions, the actions, the thing said as everyone stood up.

The 90 minutes in the middle? A haze. Not because you were absent, and not because it was unimportant. Attention degrades in the middle of long sequences, and the ends are privileged — the start because it framed everything after, the end because it's freshest.

Psychologists call these **primacy** and **recency**. Language models exhibit both, and for structurally similar reasons.

> [!NOTE] Lost in the Middle
> The empirical finding that a model's ability to use information from its context varies with **position**, following a U-shaped curve: strong at the beginning, strong at the end, weakest in the middle. It gets worse as context grows. ^lost-in-middle-def

---
# Why it happens (and why it's not a bug you can patch)

Three contributing causes, and the third is the one worth internalising:

**1. Training distribution.** Documents put their thesis at the top and their conclusion at the bottom. Instructions come first; the question comes last. The model learned that ~={blue}the ends are where the load-bearing content lives=~ — because in its training data, they were.

**2. Position encoding decay.** [[RoPE]]-style encodings induce a mild long-range decay, so very distant tokens are systematically slightly down-weighted. Context-extension tricks (interpolation) stretch positions the model was never trained on, blurring mid-range resolution further.

**3. Softmax is a fixed budget.** Attention weights must sum to 1. With 10 candidate passages, the strongest gets serious weight. With 200, the ~={red}same total attention is spread across 200 competitors=~ — every individual signal is diluted, so a mid-list passage that was merely *good* gets drowned by aggregate noise. This is why **adding more retrieved chunks can reduce accuracy**, which is genuinely counter-intuitive until you see it as a fixed budget being divided.

> [!SUCCESS] Core idea
> Context is ~={pink}not a flat array where every slot is equally readable=~. It's a gradient of attention, with the ends privileged. So *what* you put in the window is only half the decision — **where you put it** is the other half, and it's free to change. ^context-is-not-flat

---
# What to actually do 🛠️

> [!TIP] Concrete, cheap fixes — in order of value
> 1. **Send fewer, better chunks.** [[Reranking]] to 3–5 beats stuffing 20. The best fix for a middle is not having one.
> 2. **Put the best chunk at an end.** Many teams order by relevance *ascending*, so the strongest passage sits immediately before the question. Measure both directions on your data; it's a one-line change worth several points. 🥇
> 3. **Repeat the instruction at the end** of a long context. Cheap insurance against your own [[System Prompt]] getting diluted.
> 4. **Ask for citations.** Forcing the model to name which document it used pulls attention across the whole set and makes misses visible. See [[Grounding]].
> 5. **Iterate instead of stuffing** — [[Agentic Workflows|agentic]] retrieval over several small windows beats one enormous one.

> [!WARNING] "Needle in a haystack" green means very little
> The classic NIAH test hides one verbatim sentence and asks for it back. Models ace it at 1M tokens — and the marketing follows. But it tests ~={red}exact-match lookup of a conspicuous, out-of-place string=~, which is the easiest possible case.
>
> Real work needs *aggregation* ("how many of these contracts auto-renew?"), *comparison across distant passages*, and *reasoning over paraphrase*. On those, measured performance falls off long before the advertised limit. Treat NIAH as a smoke test, not evidence — and see [[Long Context]]. ^niah-is-weak

---

---
![[lost_in_the_middle.png]]
> [!TIP] Reading the chart
> Put the thing that matters at the start or the end. The middle is where facts go to die.


# ⁉️
So the window is finite, unevenly read, and wiped between requests. If you want a system that genuinely remembers a user across weeks, the window cannot be where that lives.

→ [[Memory]]
