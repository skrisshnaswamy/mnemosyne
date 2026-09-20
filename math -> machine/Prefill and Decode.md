---
aliases:
  - Prefill
  - Decode
  - TTFT
  - Time To First Token
  - Prefill/Decode
tags:
  - llm
  - inference
  - performance
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Inference is **two different jobs**: reading the prompt (one big parallel matmul, compute-bound) and writing the answer (one token at a time, memory-bound).
> **Metaphor:** A chef reading the whole recipe in one glance, then plating one dish at a time.
> **Where it bites:** TTFT vs tokens/sec are different SLAs with different fixes. Optimising the wrong phase gets you nothing.

---
You benchmark your model. Two requests:

- **Request A** — 8,000-token document, "summarise in one line." Output: 15 tokens.
- **Request B** — 12-token question, "write me a 1,500-word essay."

Request A: first token appears after **900ms**, then it's done almost instantly.
Request B: first token appears after **40ms**, then it takes **30 seconds** to finish.

Request A processed ~8,000 tokens in under a second. Request B produced 1,500 tokens in thirty. Per token, ~={blue}reading was roughly 400× faster than writing.=~

Same model. Same GPU. Why?

---
# The chef

A chef gets a recipe card and an order for 50 covers.

**Reading the card** is a single glance. Ingredients, steps, quantities — they're all there at once, and nothing about step 7 stops him reading step 2. He can absorb the whole thing in parallel.

**Cooking** is the opposite. He can't plate dish 12 before dish 11 — and crucially, he keeps walking back to the walk-in fridge for every single dish. The knife work is fast. The **walk** is what takes the time.

That's the whole distinction.

> [!NOTE] Prefill
> Processing the entire input prompt in **one forward pass**. All $n$ tokens go through the network simultaneously as a big matrix. The GPU's arithmetic units are saturated — this phase is ~={blue}compute-bound=~. Cost scales with prompt length (and quadratically in attention). Determines **Time To First Token (TTFT)**. ^prefill-def

> [!NOTE] Decode
> Generating the output **one token at a time**. Each step processes a *single* token, but to do it you must read the **entire model's weights** out of HBM into the compute units. One token of arithmetic, gigabytes of memory traffic — this phase is ~={red}memory-bandwidth-bound=~. Determines **tokens per second (TPS)** / inter-token latency. ^decode-def

---
# Why decode is so wasteful

Here's the number that makes it click. Take a 70B model in 16-bit: **140 GB of weights**.

To generate **one** token, you must stream all 140 GB from HBM through the compute units. An H100 has ~3.3 TB/s of bandwidth, so:

$$\text{time per token} \ge \frac{140\ \text{GB}}{3300\ \text{GB/s}} \approx 42\ \text{ms}$$

That's a hard floor of ~24 tokens/sec for a *single* request, no matter how fast your GPU can multiply. The arithmetic itself is trivial — one token times the weights. ~={red}The GPU sits idle >90% of the time waiting for memory.=~

> [!SUCCESS] Core idea
> Prefill is **rich in parallel work and starved of nothing**. Decode has **almost no work but must move the entire model** to do it. Every serving optimisation you'll meet is an answer to one of those two facts — and ~={pink}most of them attack decode, by finding *more work* to do per byte moved.=~ ^two-phases

Look at how neatly the techniques sort:

| Technique | Which phase | The trick |
|---|---|---|
| [[KV Cache]] | decode | Don't recompute the past |
| [[Continuous Batching]] | decode | Amortise one weight-read over many requests |
| [[Speculative Decoding]] | decode | Verify many tokens per weight-read |
| [[Quantization]] | decode | Fewer bytes to move |
| [[Grouped Query Attention]] | decode | Shrink the cache so batches fit |
| [[Flash Attention]] | prefill | Stop round-tripping the $n\times n$ matrix |
| **Chunked prefill** | both | Slice a long prefill so it doesn't stall everyone's decode |
| **Prompt caching** | prefill | Reuse the KV of an identical prefix |

---
# The scheduling headache 🚦

On one GPU, a long prefill and everyone's decode steps compete. If a user pastes 100k tokens, that prefill hogs the device — and **every other user's stream visibly stutters**. This is the single most common cause of "latency is fine in testing, jittery in production."

The fix is **chunked prefill**: break the big prefill into slices and interleave them with decode steps, trading a little TTFT for smooth token streaming everywhere else.

> [!TIP] Which SLA are you failing?
> - **TTFT bad** → it's prefill. Shorten the prompt, cache the prefix, chunk it, check whether [[Flash Attention]] is on.
> - **TPS bad** → it's decode. Quantize, raise batch size, use a smaller model or [[Speculative Decoding]].
> Reporting a single "latency" number hides which one is broken. Always split them. ⏱️

See [[GPU processing]] for why HBM bandwidth — not FLOPs — is the number that decides all of this.

---

---
![[prefill_vs_decode.png]]
> [!TIP] Reading the chart
> One phase saturates the GPU. The other leaves it idle, waiting on memory. Same request.


# ⁉️
Decode generates one token by attending to *everything before it*. Recomputing the whole history's keys and values at every step would be insane. So we don't.

→ [[KV Cache]]
