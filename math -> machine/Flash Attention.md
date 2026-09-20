---
aliases:
  - FlashAttention
tags:
  - transformers
  - attention
  - performance
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Same attention maths, **exact same answer** — it just never writes the giant $n \times n$ attention matrix to slow memory.
> **Metaphor:** Cooking from ingredients on the counter instead of walking to the pantry for every single item.
> **Where it bites:** It's why long context is affordable. Note it is **exact**, not an approximation — a common thing to get wrong in conversation.

---
A super-efficient, optimized algorithm for calculating the attention mechanism. It's much faster and uses far less memory than the standard implementation by being clever about how it accesses the GPU's memory.

---
# The bottleneck isn't the maths

The intuition almost everyone has is wrong, so it's worth correcting properly.

You'd assume attention is slow because $n^2$ dot products is a lot of arithmetic. It isn't. Modern GPUs are *absurdly* good at arithmetic. Attention is slow because of **moving data around**.

Recall from [[GPU processing]] that a GPU's memory is a hierarchy:

| | Size | Speed |
|---|---|---|
| **SRAM** (on-chip, per SM) | ~20 MB | ~19 TB/s ⚡ |
| **HBM** (the "VRAM" on the spec sheet) | 40–80 GB | ~1.5–3 TB/s 🐢 |

SRAM is roughly **10× faster** and thousands of times smaller. Every trip to HBM is expensive.

Now look at what standard attention does for a sequence of 8,000 tokens:
1. Compute $S = QK^T$ → an $8000 \times 8000$ matrix → **write to HBM** (~256 MB in fp32, per head)
2. Read it back → apply the [[Causal Attention|mask]] → write it back
3. Read it back → softmax → write it back
4. Read it back → multiply by $V$ → write the output

That matrix gets dragged across the slow bus **four separate times**. The GPU's compute units spend most of that period idle, waiting.

> [!SUCCESS] Core idea
> Attention is **memory-bandwidth-bound**, not compute-bound. The fix isn't fewer calculations — it's ~={blue}fewer round trips to slow memory.=~ ^memory-bound

---
# The fix — never build the matrix

Flash Attention's insight: that $n \times n$ matrix is an **intermediate**. Nobody wants it. It exists only on the way to the output. So — don't materialise it.

Instead, chop Q, K and V into **tiles** small enough to fit in SRAM. For each pair of tiles, load them once, do the *entire* computation — scores, mask, softmax, multiply by V — while everything stays on-chip, and write out only the small running result.

The $n \times n$ matrix is computed **in pieces that never simultaneously exist**. 🧩

This is called **kernel fusion**: four separate operations that each did their own HBM round trip become one operation that does one.

> [!NOTE] The hard part — online softmax
> There's an obvious objection: softmax needs to normalise over the **whole** row, and you're only looking at one tile of it. How can you normalise by a sum you haven't finished computing?
>
> The answer is a **running rescale**. Keep a running max and a running sum as you sweep across tiles, and when a new tile arrives with a bigger max, retroactively correct what you've accumulated so far by a scaling factor. Mathematically identical to the full softmax, computed in one streaming pass. ^online-softmax

> [!WARNING] It is exact, not approximate
> Worth being firm about, because it gets confused with sparse or linear attention. Flash Attention produces **bit-for-bit the same output** (up to floating-point reordering) as standard attention. Sparse/linear attention approximates by *skipping* interactions. Flash computes all of them — it just computes them somewhere faster. ^exact-not-approximate

And for the backward pass it uses **recomputation**: rather than storing the attention matrix for [[Backpropagation|backprop]], it throws it away and recomputes tiles on demand. Counterintuitive — extra FLOPs — but since we're bandwidth-bound, recomputing on-chip is *cheaper* than fetching from HBM. Same bargain as gradient checkpointing in [[Pytorch Autograd#3. Backward OOMs even though forward was fine|autograd]].

---
# What it buys you

| | Standard | Flash |
|---|---|---|
| **Memory** | $O(n^2)$ | $O(n)$ ✅ |
| **HBM traffic** | $O(n^2 d)$ | $O(n^2 d^2 / M)$ |
| **Exact?** | yes | yes |
| **Speedup** | — | ~2–4× typical, more at long context |

The memory row is the one that changed things. Dropping from quadratic to **linear** memory is what made 32k, 128k and million-token contexts practically possible. The speedup was nice; the memory scaling was the unlock. 🔓

**FlashAttention-2** improved the work partitioning across GPU warps; **FlashAttention-3** exploits Hopper-specific hardware (async copies, FP8). In PyTorch you mostly get it for free via `F.scaled_dot_product_attention`, which dispatches to a Flash kernel when your shapes and dtype qualify — and silently falls back to the slow path when they don't, which is worth checking if you expected a speedup and didn't get one.

> [!TIP] The transferable lesson
> Flash Attention is the flagship example of an **IO-aware algorithm** — one designed around the memory hierarchy rather than the operation count. If a kernel feels slower than its FLOP count suggests, ask how many times data crossed the slow bus. That question generalises well beyond attention. 🎯

---
### Context from Kidan
Switching to Flash Attention produced a similar drop in resource usage to the [[Mixed Precision training|BFLOAT16]] gain on the Vanilla Encoder — consistent with both fixes attacking the same underlying bottleneck: bytes moved, not maths done.

---
# ⁉️
Flash Attention makes attention cheap in memory. The other half of that fight is making every number itself smaller → [[Mixed Precision training]].
