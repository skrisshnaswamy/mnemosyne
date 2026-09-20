---
aliases:
  - Context Length
  - Context
  - Context Limit
tags:
  - llm
  - inference
  - fundamentals
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** The **total** number of tokens the model can hold at once — your prompt, the docs you pasted, the history, *and* the answer it's about to write, all sharing one budget.
> **Metaphor:** A whiteboard in a meeting room. Finite surface. To write something new when it's full, you must erase something old.
> **Where it bites:** Long chats degrading, RAG chunks silently truncated, and the `maximum context length exceeded` error at 2am.

---
You're building a support bot. The numbers look fine on paper:

```
System prompt               600 tokens
Retrieved knowledge-base docs   6,000
Conversation so far (12 turns)  5,200
User's new question               150
                             --------
                             11,950
```

Model limit: 128,000. You have tons of room. Ship it.

Three weeks later a customer has a long thread, support pasted in a 40-page PDF, and the bot answers a question from *turn three* while ignoring the thing asked 30 seconds ago. Then it hard-errors.

What actually ran out?

---
# The whiteboard

Think of the model as a very sharp consultant in a room with **one whiteboard and no memory whatsoever.**

Every time you ask something, you don't continue a conversation — ~={blue}you re-brief them from scratch.=~ Everything they're allowed to know must be on that board right now: the instructions, the reference docs, everything said so far, and there must be blank space left for them to *write the answer*.

Two things follow immediately, and both surprise people:

1. The board is **shared between input and output**. Fill it to the brim with context and there's no room left to write. That's why `max_tokens` and prompt length trade against each other.
2. The consultant is **stateless**. There is no "memory" of turn 3 — turn 3 is only known because it's still written on the board. Erase it and it never happened. ^context-is-everything

> [!NOTE] Context window
> The maximum number of tokens a model can attend over in a single forward pass — prompt **plus** generated completion. It is not storage and not memory; it's the entire universe the model can perceive at that instant. ^context-window-def

> [!SUCCESS] Core idea
> A chatbot has no memory. The API is stateless. ~={pink}The illusion of a conversation is created by re-sending the entire transcript on every single turn.=~ That's also why turn 20 costs far more than turn 1.

---
# So why can't we just make the board enormous?

Two walls, and they're different walls — people conflate them constantly.

**Wall 1 — attention is quadratic.** Every token attends to every other token, so doubling length **quadruples** the attention compute. 4× the tokens = 16× the work. This one hits *prefill*. ([[Flash Attention]] and [[Long Context]] are the responses.)

**Wall 2 — the [[KV Cache]] is linear but huge.** Every token in the window must keep its keys and values in GPU memory for the whole generation. That grows linearly with length *and* with batch size, and it's frequently ~={red}bigger than the model weights themselves=~.

> [!WARNING] Advertised ≠ usable
> A model shipping "1M context" means it will *accept* 1M tokens. It does not mean it *uses* them well — accuracy on a fact buried at 70% depth can fall off a cliff. See [[Lost in the Middle]]. It also doesn't mean you can afford it: cost scales with tokens, and latency with it.

---
# What you actually do about it

Four strategies, and the choice is an architecture decision, not a config tweak:

| Strategy | What it does | Cost |
|---|---|---|
| **Truncate** | Drop the oldest turns | Free, and lossy in the worst way — you drop the setup |
| **Summarise (rolling)** | Compress old turns into a paragraph with the model itself | An extra LLM call; details get smoothed away |
| **Retrieve** | Store everything outside, pull back only what's relevant now — [[RAG]] | Retrieval infrastructure, plus retrieval errors |
| **Externalise state** | Keep facts in a structured store, re-inject as needed — [[Memory]] | Engineering; but it's the only one that actually scales |

> [!TIP] The budget discipline
> Write your context budget as a table before you build, with a hard number per slot: system prompt, tools, retrieved docs, history, **reserved output**. When you blow the budget, you now know *which* slot to cut. Doing this after the fact is how people end up truncating the system prompt by accident. 🧮

And note: **prompt caching** makes the stable front of the window (system prompt, tool definitions, long pasted doc) far cheaper on repeat calls — but only if it's a *prefix* and only if it's byte-identical. Put the volatile stuff last. This is a direct consequence of how [[Prefill and Decode|prefill]] works.

---

---
![[context_budget.png]]
> [!TIP] Reading the chart
> By the time the user's question arrives, most of the window is already gone.


# ⁉️
Notice that filling the board and writing on it are two completely different physical operations — one is a bulk read, the other is a slow word-by-word crawl. They bottleneck on different hardware, and almost every serving trick exploits the difference.

→ [[Prefill and Decode]]
