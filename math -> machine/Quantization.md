---
aliases:
  - Quantisation
  - INT8
  - INT4
  - GPTQ
  - AWQ
  - GGUF
tags:
  - llm
  - inference
  - performance
  - compression
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Store weights in **4 or 8 bits** instead of 16 — a 70B model goes from 140 GB to ~35 GB, and decode speeds up because there are fewer bytes to move.
> **Metaphor:** Rounding every price on the menu to the nearest 50p. The bill is nearly the same; the menu fits on a postcard.
> **Where it bites:** Fitting a big model on a small GPU. Also the source of "it got dumber and I can't say why."

---
A 70B model at 16-bit is **140 GB**. Your A100 has 80.

You could shard it across two GPUs and pay for two GPUs plus the interconnect. Or you could ask a cheeky question: how much of those 16 bits is *doing anything*?

Print a random slice of a trained weight matrix:

```
0.0142, -0.0097,  0.0331, -0.0018,  0.0205, ...
```

They're small, they're clustered near zero, and **there are billions of them**. Does the difference between `0.014211` and `0.014217` change the next token?

Almost never. So why spend 16 bits saying it?

---
# The menu

A restaurant prints its menu with every price to the penny: £12.47, £8.03, £15.99.

Round everything to the nearest 50p: £12.50, £8.00, £16.00. The menu is now far simpler. A 40-cover night's takings are off by maybe a pound. Nobody notices.

But now try it on the **wine list**, where one bottle is £4,000. Round *that* to the nearest 50p and it's fine — but if you insisted on using the *same fixed scale* across your whole till, calibrated to fit £4,000 in, your coffee prices would all round to £0. ~={red}One outlier ruins the scale for everything else.=~

Hold that thought — it turns out to be the entire difficulty of quantizing an LLM.

> [!NOTE] Quantization
> Mapping high-precision floats to a small set of integer levels, plus a **scale** (and optionally a zero-point) to map back:
> $$w \approx s \cdot q, \qquad q \in \{-8,\dots,7\}\ \text{for int4}$$
> The scale is shared across a *group* — per tensor, per channel, or per block of 64/128 weights. **Smaller groups = better accuracy, slightly more overhead.** ^quantization-def

---
# The outlier problem, for real

In real LLMs, a small number of activation channels carry values **100× larger** than the rest — and they're the *important* ones. Naive per-tensor int8 crushes everything else to zero around them and the model collapses.

Every modern method is a different answer to that:

| Method | Idea | Where |
|---|---|---|
| **LLM.int8()** | Detect outlier channels, keep **those** in fp16, rest in int8 | Mixed |
| **GPTQ** | Quantize column by column, **compensating** remaining weights for the error already made (second-order) | Weights only |
| **AWQ** | Find the ~1% of weights that matter (by *activation* magnitude) and scale them up before quantizing | Weights only ✅ |
| **SmoothQuant** | Mathematically shift the outlier difficulty from activations into weights | W8A8 |
| **GGUF / k-quants** | Block-wise scales, mixed bit-widths per tensor, CPU-friendly | llama.cpp 💻 |

> [!SUCCESS] Core idea
> ~={pink}Not all weights are equal.=~ Quantization stopped being "round everything" and became "spend your precious bits where the activations say they matter." That single reframing is what got int4 from unusable to near-lossless. ^outliers-matter

---
# Two axes people mix up 🧭

**Weights vs activations.** *Weight-only* (W4A16: 4-bit weights, 16-bit maths) is the common case — it targets [[Prefill and Decode#^decode-def|memory bandwidth]], which is decode's bottleneck, and it's much easier because weights are static and can be analysed offline. Activations change per input, so *W8A8* is harder but also speeds up compute-bound **prefill**.

**PTQ vs QAT.** *Post-Training Quantization* takes a finished model and a few hundred calibration samples — minutes of work. *Quantization-Aware Training* simulates the rounding during training so the model learns around it — much better at 2–3 bits, much more expensive.

> [!WARNING] What it actually costs you
> Benchmarks quote perplexity deltas of ~1% and people hear "free." What degrades first is rarely captured there: ~={red}long-context recall, multi-step arithmetic, rare languages, and instruction-following at the tail.=~ Quantize, then run *your* [[Evals]] — not a leaderboard's. ^quantization-cost

| Precision | 70B size | Verdict |
|---|---|---|
| FP16 | 140 GB | Reference |
| INT8 | 70 GB | Effectively lossless |
| INT4 (AWQ/GPTQ) | ~35 GB | Small, usually acceptable loss ✅ |
| INT3 / INT2 | ~26 / 18 GB | Visible degradation without QAT |

Related: [[Mixed Precision training]] is the *training*-time cousin (and the reason BF16 keeps range over precision), [[QLoRA- Efficient Finetuning of Quantized LLMs]] fine-tunes *on top of* a frozen 4-bit base, and [[Distillation]] is the alternative way to get small. See also [[Why Gated DeltaNet Survives 4-Bit Quantization- NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM]].

---

---
![[quantization_levels.png]]
> [!TIP] Reading the chart
> Quantization doesn't shrink the numbers — it coarsens the grid they must land on.


# ⁉️
Now the model fits and it's fast. But given the same prompt twice you get two different answers — and there's a set of knobs deciding exactly how different.

→ [[Sampling Parameters]]
