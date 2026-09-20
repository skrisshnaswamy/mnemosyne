---
aliases:
  - AMP
  - BF16
  - FP16
tags:
  - training
  - performance
  - gpu
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Do the **maths** in 16-bit (fast, half the memory) but keep a **master copy** of the weights in 32-bit (so tiny updates don't vanish).
> **Metaphor:** Doing your rough working in shorthand, but writing the running total in full — so you never lose the pennies.
> **Where it bites:** ~2× throughput and ~half the memory, nearly free. Also the #1 source of `NaN` losses. **bf16 vs fp16** is the distinction to have ready.

---
### FP32 vs BFLOAT16 (Numerical Precision)

- `FP32` (Floating Point 32) is the standard, highly precise way computers store numbers.
- `BFLOAT16` (Brain Float 16) is a "half-precision" format that uses less memory and is much faster on modern GPUs, with a tiny loss in precision that often doesn't hurt model performance.

---
# What the bits are actually for

A floating-point number splits its bits into two jobs, and the split is the whole story:

- **Exponent** → **range**. How big or small can it get before overflowing to `inf` or underflowing to `0`?
- **Mantissa** → **precision**. How many significant digits?

| Format | Exponent | Mantissa | Range | Precision |
|---|---|---|---|---|
| **FP32** | 8 | 23 | ~$10^{\pm 38}$ | ~7 digits |
| **FP16** | **5** | 10 | ~$10^{\pm 5}$ ⚠️ | ~3 digits |
| **BF16** | **8** ✅ | 7 | ~$10^{\pm 38}$ ✅ | ~2 digits |

Look at what BF16 did. It kept **FP32's full exponent** and sacrificed mantissa bits instead.

> [!SUCCESS] Core idea
> That's the key design decision, and it reflects a real fact about deep learning: ~={blue}neural networks need **range** far more than they need **precision**.=~
>
> A gradient of $10^{-8}$ still points the right way even if you only know 2 of its digits. But if $10^{-8}$ **underflows to exactly zero**, the signal is gone permanently. FP16 loses gradients; BF16 keeps them (fuzzily). ^range-beats-precision

This is why BF16 has become the default. It converts to and from FP32 by simply truncating bits — same exponent range, so no overflow, no scaling gymnastics.

---
# Why you can't just use 16-bit everywhere

Two failures make pure 16-bit training break:

**1. Gradient underflow (FP16 only).** Small gradients fall below $\sim 6 \times 10^{-8}$ and become exactly zero. Whole layers silently stop learning.
> **Fix — loss scaling.** Multiply the loss by a large constant (say 1024) before `.backward()`, so every gradient is scaled up out of the danger zone; divide it back out before the optimizer step. Modern implementations do this *dynamically*: push the scale up until an `inf` appears, back off, skip that step, continue. It's the main reason FP16 needs a `GradScaler` and BF16 doesn't. ✅

**2. The vanishing update.** This one applies to *both* formats and is more fundamental.

A weight is $0.7$. Its update is $0.0000001$. In 16-bit, $0.7$ has roughly 3 significant digits — so $0.7 + 0.0000001$ rounds back to exactly $0.7$. **The update does nothing.** Repeat for 10,000 steps and the model has not moved at all.
> **Fix — the FP32 master copy.** Keep the authoritative weights in FP32. Cast a 16-bit copy for the forward/backward pass, but apply the optimizer update to the FP32 master. Small updates accumulate there properly. This is the "mixed" in mixed precision. ^master-weights

> [!NOTE] So what's actually in each precision
> - **FP16/BF16** — activations, gradients, the matmuls and convolutions (the bulk of compute and memory)
> - **FP32** — master weights, optimizer state (Adam's $m$ and $v$), loss accumulation, softmax, and normalisation-layer statistics
>
> Reductions stay in FP32 because summing thousands of small numbers in 16-bit loses badly to rounding.

---
# Why it's actually fast

Two separate wins, and people usually only credit the first:

1. **Tensor Cores** — dedicated hardware on NVIDIA GPUs from Volta onward that does 16-bit matrix multiply-accumulate. Several times the FP32 throughput. This only applies to **matmul-shaped** work.
2. **Bandwidth** — every tensor is half the bytes, so you move half as much data. Since most training is [[Flash Attention#The bottleneck isn't the maths|memory-bandwidth-bound]], this is often the *bigger* win.

> [!WARNING] Why you sometimes get no speedup
> Tensor Cores want dimensions that are **multiples of 8** (16 for some formats). A hidden size of 4096 flies; 4095 falls back to the slow path with no warning. If you enabled AMP and gained nothing, check your shapes first. 📏

```python
scaler = torch.cuda.amp.GradScaler()          # not needed for bf16
for x, y in loader:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = criterion(model(x), y)
    scaler.scale(loss).backward()
    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
```

`autocast` maintains a per-operation allowlist — matmuls run in 16-bit, softmax and normalisations stay FP32. You rarely need to override it.

> [!TIP] Choosing
> **BF16** on Ampere (A100) or newer — no loss scaling, far more forgiving. **FP16** only on older cards (V100, T4) that lack BF16 support, and accept the `GradScaler`. **FP8** on Hopper/Blackwell for the largest runs, where per-tensor scaling becomes a real engineering exercise. See [[ML Infrastructure]].

---
### Kidan's experiments
- The Vanilla Encoder gets a massive **120% throughput increase** by switching to `BFLOAT16`.
- The old CM2 encoder couldn't take advantage of it for some reason.
	- But then by switching to [[Flash Attention]], we noticed similar drop in resource usages.

> [!TIP] Likely explanation for the CM2 gap
> Two candidates, both checkable:
> 1. **Shapes not multiples of 8** — the fallback is silent, and it's the most common cause of "AMP did nothing".
> 2. **Not matmul-bound.** BF16 accelerates matmuls. If CM2's time went on the $n \times n$ attention matrix's memory traffic rather than arithmetic, there was little for Tensor Cores to speed up. That the [[Flash Attention]] switch produced a *similar* gain is strong evidence for this one — Flash attacks exactly that bandwidth bottleneck, and the two fixes landing in the same place suggests they were both hitting the same wall. ^kidan-bf16-gap

---
# ⁉️
Precision is one lever on training speed. The rest of the picture — memory hierarchy, interconnect, which card, what's actually bottlenecking you — is in [[GPU processing]] and [[ML Infrastructure]].
