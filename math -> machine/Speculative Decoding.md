---
aliases:
  - Speculative Sampling
  - Draft Model
  - Medusa
tags:
  - llm
  - inference
  - performance
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A cheap model **guesses** the next few tokens; the big model **checks all of them in one forward pass** and keeps the longest correct prefix.
> **Metaphor:** A junior drafts the paragraph, the partner reads it in one pass and strikes through from the first wrong word.
> **Where it bites:** 2–3× faster decode with **mathematically identical** output. Costs extra FLOPs and extra memory.

---
Here's the wasteful fact from [[Prefill and Decode]] again: to produce **one** token, decode must stream the entire 140 GB of weights out of HBM. The arithmetic is trivial; the walk to the fridge is everything.

But now look at prefill. It pushes 8,000 tokens through in a single pass — one walk, 8,000 tokens of work.

So the machine is perfectly capable of ~={blue}scoring many tokens for the price of one weight-read.=~ During decode we just don't have many tokens to score, because we don't know what they are yet.

Unless… we guess?

---
# The junior and the partner

A partner at a law firm bills a fortune per hour. A junior is cheap.

Old way: the partner writes the paragraph one word at a time. Expensive per word.

New way: the **junior drafts five words**. The partner reads the draft **in one glance** and checks: "would I have written *this* word here? and *this* one?" She accepts the run up to the first word she'd have chosen differently, replaces that one with her own, and hands it back.

Here's the bit that matters: reading five words takes her *the same single glance* as reading one. She was going to open the file anyway.

If the junior is decent, four or five words get accepted per glance. If the junior is useless, one word gets accepted — and you've lost only the junior's (cheap) time.

> [!NOTE] Speculative decoding
> A small **draft model** autoregressively proposes $k$ tokens. The large **target model** scores all $k{+}1$ positions in **one** forward pass. A rejection-sampling test accepts the longest prefix the target would plausibly have produced, resamples the first rejected position from a corrected distribution, and continues. ^spec-decoding-def

> [!SUCCESS] Core idea
> ~={pink}Verification is parallel; generation is serial.=~ Speculative decoding converts serial generation into parallel verification and pays for the guesses with FLOPs you weren't using anyway — because decode leaves the GPU's arithmetic units ~90% idle. ^verify-vs-generate

---
# The part people get wrong 🎯

> [!WARNING] It is **not** an approximation
> The most common misconception: "so it's faster but slightly worse?" **No.** The accept/reject rule (rejection sampling with a corrected residual distribution) makes the output distribution ~={blue}provably identical=~ to running the target model alone. Same temperature, same everything. You are not trading quality for speed — you're trading *spare FLOPs* for speed. ^spec-is-exact

The speedup is governed by the **acceptance rate** $\alpha$ — how often the draft's token is one the target would have picked:

$$\text{expected tokens per pass} = \frac{1 - \alpha^{k+1}}{1 - \alpha}$$

With $\alpha = 0.8$ and $k = 4$ you get ~3.4 tokens per target pass. With $\alpha = 0.3$ you get ~1.4 — and after the draft model's own cost, you may be **slower than baseline**.

| Flavour | Where the draft comes from | Notes |
|---|---|---|
| **Draft model** | A small model from the same family (7B drafting for 70B) | Needs matching [[Tokenization\|tokenizer]]; extra weights in VRAM |
| **Self-speculation (Medusa)** | Extra prediction heads bolted onto the target itself | No second model; needs a little training |
| **n-gram / prompt lookup** | Copy from the prompt itself | Free, and brilliant for code edits and RAG summarisation where output echoes input 🧠 |
| **EAGLE / feature-level** | Draft in feature space, not token space | Higher $\alpha$, currently the strongest |

→ [[Verification-Aware Training for Speculative Decoding]]

---
# When it helps and when it doesn't

> [!TIP] The batch-size interaction — the one most people miss
> Speculation spends **idle FLOPs**. At batch size 1, the GPU is starved of work and idle FLOPs are abundant → big win. Under heavy [[Continuous Batching]], those FLOPs are ~={red}already spoken for=~, so speculation now *competes* with real requests and can reduce total throughput.
>
> Rule of thumb: **speculate for latency at low concurrency; batch for throughput at high concurrency.** Good servers switch strategy dynamically based on queue depth. ⚖️

Also: $\alpha$ is highest on predictable text (boilerplate code, structured output, quoting retrieved documents) and lowest on creative text at high [[Sampling Parameters|temperature]]. Measure $\alpha$ on *your* traffic before believing any published speedup.

---

---
![[speculative_decoding.png]]
> [!TIP] Reading the chart
> Below roughly 50% acceptance you are doing extra work for nothing. The draft model has to actually be good.


# ⁉️
Every trick so far moves the same bytes more cleverly. The blunt alternative is to make the bytes themselves smaller.

→ [[Quantization]]
