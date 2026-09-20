---
aliases:
  - Temperature
  - Top-k
  - Top-p
  - Nucleus Sampling
  - Repetition Penalty
  - Decoding Strategy
tags:
  - llm
  - inference
  - generation
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** The model outputs a **distribution**, never a word — temperature reshapes it, top-k/top-p truncate it, and then you roll a die.
> **Metaphor:** A DJ with a crate of records. Temperature is how adventurous she is; top-p is how many crates she'll even look in.
> **Where it bites:** "Set temperature to 0 for determinism" is wrong in two separate ways. And repetition penalty quietly breaks code and JSON.

---
The model has processed your prompt. Its final layer outputs one number per vocabulary entry — the **logits**. Softmax turns them into probabilities:

```
" Paris"     0.71
" the"       0.11
" France"    0.06
" a"         0.03
...
" banana"    0.0000004
```

Now — what does the API return? It has to emit **one** token.

The naive answer is "the top one, obviously." Do that for every token and you get **greedy decoding**. Try it on a story and watch what happens: it loops. `The man walked to the store. The man walked to the store.` Perfectly high-probability, completely dead.

Why would always picking the most likely word produce *worse* text than sometimes picking a less likely one?

---
# The DJ

A DJ has thousands of records. If she always plays the single most popular track, she plays the same song all night. Real human sets weave in the less obvious record — and *that's* what makes it sound alive.

But she doesn't pick uniformly at random either, or you'd get a polka in the middle of a house set.

So she has two dials:
- **How adventurous am I feeling?** (temperature)
- **Which crates am I even willing to reach into?** (top-k / top-p)

Two genuinely different controls — one reshapes the odds, the other bans the long tail outright.

---
# Temperature 🌡️

Divide the logits by $T$ *before* softmax:

$$p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

- $T < 1$ → gaps widen, the distribution gets **peaky**, safer and more repetitive
- $T = 1$ → the model's own beliefs, untouched
- $T > 1$ → gaps shrink toward uniform, wilder, and eventually incoherent

> [!WARNING] "Temperature 0" — two misconceptions at once
> **(a)** $T=0$ is a division by zero. Implementations *special-case* it to mean greedy argmax. It isn't a temperature value at all.
> **(b)** Greedy is **not** reproducible in practice. Batched GPU matmuls are non-deterministic in floating-point reduction order, and under [[Continuous Batching]] your batch composition changes request to request — so near-tied logits can flip. ~={red}Temperature 0 gets you *stable-ish*, not *identical*.=~ For real reproducibility you need a fixed seed **and** a fixed batching regime, which you don't control on a hosted API. ^temp-zero-myth

---
# Truncation: top-k and top-p

Both throw away the tail before sampling; they disagree on *how much* tail.

**Top-k** — keep the $k$ most likely tokens, renormalise, sample. Fixed count.
**Top-p (nucleus)** — sort by probability, keep the smallest set whose cumulative probability ≥ $p$. **Adaptive count.**

The adaptivity is the whole point. Consider two positions:

| Context | Distribution | top-k=50 does | top-p=0.9 does |
|---|---|---|---|
| `The capital of France is` | one spike at 0.98 | keeps 50 candidates, 49 of them nonsense 😬 | keeps **1** ✅ |
| `He opened the door and saw a` | flat over hundreds | keeps 50 — too few | keeps ~200 ✅ |

> [!SUCCESS] Core idea
> ~={pink}Top-p adapts its cutoff to the model's own confidence.=~ Where the model is sure, it takes one option; where genuinely open, it opens up. That's why it beat top-k and became the default. ^top-p-adaptive

**Min-p** is the newer variant: keep tokens with $p_i \ge p_{\min} \cdot p_{\max}$ — a floor *relative to the best token*. It's more robust at high temperature, which is why it's popular with local models.

---
# Repetition controls 🔁

Three different mechanisms that people use interchangeably and shouldn't:

| Knob | Mechanism | Danger |
|---|---|---|
| **Repetition penalty** | *Divides* the logit of any already-seen token (multiplicative) | ~={red}Catastrophic for code and JSON=~ — `}`, `def`, `return`, `self` are *supposed* to repeat |
| **Frequency penalty** | Subtracts proportional to how many times seen | Gentler, scales with actual overuse |
| **Presence penalty** | Flat subtraction if seen at all | Pushes toward new topics |

> [!TIP] Presets worth memorising
> - **Factual Q&A / extraction / [[Structured Output]]** — `T≈0` (greedy), no penalties
> - **Code** — `T≈0.1–0.2`, **no repetition penalty** 🚨
> - **General chat** — `T≈0.7`, `top_p≈0.9`
> - **Creative writing** — `T≈1.0`, `top_p≈0.95`, small presence penalty
> Change **one** of temperature or top-p, not both. Fighting two truncations at once makes behaviour impossible to reason about. 🎛️

> [!NOTE] Beam search — why it's absent
> Beam search keeps $b$ partial sequences and maximises total sequence probability. It's excellent for translation and summarisation, where there's roughly one right answer, and it's *bad* for open-ended chat — maximising likelihood is exactly what produced the looping text above. It also fights streaming and [[Continuous Batching]]. Hence: sampling everywhere. ^why-not-beam

Related: [[Mode Collapse]] is the same "collapse onto the safe output" failure in generative models generally, and low temperature is one route to [[Hallucination|confidently flat wrongness]].

---

---
![[sampling_temperature.png]]
> [!TIP] Reading the chart
> The model's logits never change. Temperature only changes how sharply you read them.


# ⁉️
All of this reshapes a distribution the model already had. So how good was that distribution in the first place? There's a single number for it.

→ [[Perplexity]]
