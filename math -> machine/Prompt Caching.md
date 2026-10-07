---
aliases:
  - Prefix Caching
  - Context Caching
  - Cache Breakpoint
tags:
  - llm
  - inference
  - performance
  - engineering
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** The provider keeps the **KV cache of a prefix you've already sent**, so re-sending it costs a fraction of the price and almost none of the prefill time — as long as the prefix is byte-identical.
> **Metaphor:** A tower of blocks. You can reuse any height of it, but change a block near the bottom and everything above it has to be rebuilt.
> **Where it bites:** Prefix stability becomes a **design constraint on your prompt layout**. Put the volatile part last, or pay 4×.

---
Two system prompts. Find the difference:

```
A:  You are a support assistant for Acme Ltd.
    Current time: 2026-09-20T14:03:11Z
    <4,000 tokens of policy and tool definitions>

B:  You are a support assistant for Acme Ltd.
    <4,000 tokens of policy and tool definitions>
    Current time: 2026-09-20T14:03:11Z
```

Same tokens. Same information. Same order of magnitude of everything.

Over a 20-step agent run, prompt A costs **$0.80** and prompt B costs **$0.21**.

One line moved, ~={red}3.8× the bill.=~ And when you look at the invoice, nothing says "timestamp". It just says input tokens.

Why would the *position* of a line change what it costs?

---
# The tower

[[KV Cache|The KV cache]] is built strictly left to right. Token 400's key and value depend on tokens 1–399, and on nothing after — that's [[Causal Attention]], and it's a hard structural fact.

![[prompt_caching_tower.png]]

So think of the cache as a tower of blocks. Block 400 rests on 399 blocks below it.

Now: **any prefix of the tower is reusable**. If the next request starts with the same first 4,000 tokens, those 4,000 blocks are already stacked — the provider keeps them and starts building at block 4,001.

And: **change one block near the bottom and everything above it falls.** A timestamp at token 12 doesn't invalidate twelve tokens. It invalidates the entire tower, because every block above it was computed on top of the old one.

That is the whole mechanism, and everything practical follows from it.

> [!NOTE] Prompt caching
> The provider stores the computed key/value tensors for a prefix of your prompt, keyed by its exact content, for a short TTL. A later request whose prompt **begins with the identical bytes** skips prefill for that span and pays a reduced "cache read" rate. Variants: **automatic** (the provider finds the longest match — nothing to configure) and **explicit** (you mark **cache breakpoints** and control what's kept). ^prompt-cache-def

> [!SUCCESS] Core idea
> This is the *only* optimisation in this whole area where ~={pink}the winning move is to make your prompt longer but more stable, rather than shorter.=~ A 4,000-token fixed preamble that's cached beats a 2,000-token one that changes every request. Stability, not size. ^stability-beats-size

---
# The arithmetic 💸

Rough prices, and the shape is what matters more than the digits:

```
ordinary input tokens      1.0×
cache WRITE (first time)   ~1.25×      you pay a premium to store it
cache READ (hits)          ~0.1×       explicit caching
                           ~0.5×       automatic caching
```

So the break-even is almost immediate. You pay 1.25× once and 0.1× thereafter; a prefix reused **twice** has already paid for itself:

$$1.25 + 0.1 = 1.35 \quad\text{vs}\quad 1.0 + 1.0 = 2.0$$

Push the 20-step agent run through it — 4,150 tokens of system prompt and tool schemas, plus a transcript that grows by ~880 tokens a step:

| Strategy | Cost of the run | Saving |
|---|---|---|
| No cache (or a timestamp at the top) | **$0.805** | — |
| Cache the system prompt + tool definitions | **$0.602** | 25% |
| Move the breakpoint **down the transcript** each turn | **$0.212** | ~={pink}74%=~ |

The jump from the second row to the third is the one people miss. Caching the static preamble is obvious and worth a quarter. Caching **the conversation so far** — placing the breakpoint after the last completed turn, so every turn's growing history is also a reusable prefix — is worth three times as much, because in a long run the transcript dwarfs the preamble.

And the latency follows the money: prefill of 18k tokens is ~640 ms, and a cache hit turns most of that into a lookup. TTFT improves by roughly the fraction you cached.

---
# The layout rule, which is the whole practical takeaway

```
┌─ STABLE, identical on every request ───────────────┐
│  system prompt                                     │
│  tool definitions                                  │  ← cache breakpoint
│  long-lived documents, style guides, few-shot set  │
├─ SEMI-STABLE, grows by append only ────────────────┤
│  conversation history / agent transcript           │  ← rolling breakpoint
├─ VOLATILE, changes every request ──────────────────┤
│  current time, request id, retrieved chunks        │
│  the user's message                                │
└────────────────────────────────────────────────────┘
```

> [!WARNING] The things that silently cost you the cache
> Each of these looks harmless and each one drops your hit rate to zero:
> - **A timestamp or request id near the top** 🕐 — the original sin
> - **Non-deterministic serialisation** — a `dict` of tool schemas dumped in a different key order, a `set` iterated, a float formatted differently
> - **Retrieved chunks placed before the system prompt**, or in relevance order that shuffles
> - **Dynamically selected few-shot examples** — a different demo per request rebuilds the tower
> - **Per-user personalisation at the top** rather than the bottom
> - **A load balancer that spreads your requests across replicas** — on self-hosted vLLM the cache is *per node*, so round-robin routing destroys a hit rate that prefix-aware sticky routing would preserve. ~={red}Your cache hit rate can be a load-balancer config bug.=~ ^cache-killers

> [!TIP] Measure it, because it is invisible otherwise
> Every provider returns cache-read and cache-write token counts in the usage block. Log them, and put **cache hit rate** on the same dashboard as cost — see [[Observability and Tracing]]. A hit rate that quietly falls from 85% to 12% after a deploy is one of the few production regressions that shows up *only* as a bigger invoice, four weeks later. 📊

**The TTL is short** — minutes, not hours — and refreshes on each hit. Which means caching pays for bursty, conversational and agentic traffic (many requests sharing a prefix in a short window) and pays nothing for a cron job that runs the same prompt once an hour. Know which one you have before you plan around it.

**Self-hosted** is the same idea under a different name: vLLM's automatic prefix caching hashes blocks of the [[KV Cache]] and reuses matching ones across requests. Same mechanism, same rules, and you own the eviction policy.

Related: [[KV Cache]] (what is actually being stored), [[Prefill and Decode]] (the phase being skipped), [[Context Engineering]] (which decides what the prefix *is*), [[Cost and Latency]], [[Model Gateway]].

---

---
![[prompt_cache_savings.png]]
> [!TIP] Reading the chart
> All three lines are the same task with the same model and the same words. The only variable is where the volatile text sits — and it's the difference between 80 cents and 21 cents on a single run.

---
> [!SUCCESS] If you remember one thing
> ~={pink}Put the volatile part last.=~ The cache is a tower built strictly left to right: change a block near the bottom and every block above it is rebuilt at full price.

---
# ⁉️
That run cost 21 cents and completed. The interesting question is the one where it *didn't* — where the provider returned a 529, your client tried again, and again, and the incident report afterwards showed the bill tripled during an outage.

→ [[Retries and Fallbacks]]
