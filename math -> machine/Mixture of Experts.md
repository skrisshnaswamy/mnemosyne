---
aliases:
  - MoE
  - Sparse MoE
  - Router
  - Sparse Mixture of Experts
tags:
  - llm
  - architecture
  - performance
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Replace one big feed-forward block with $N$ smaller ones and a **router** that activates only 2 per token — total parameters go up, compute per token doesn't.
> **Metaphor:** A hospital. You don't see all forty consultants; triage sends you to the two you need.
> **Where it bites:** Cheap FLOPs, ~={red}expensive VRAM=~ — you must hold every expert in memory even though you use 5% of them. It's a serving problem, not a training win.

---
Scaling says bigger is better, and [[Training Compute-Optimal Large Language Models (Chinchilla)|Chinchilla]] tells you how to spend the budget. But both cost and latency scale with parameters, because every token passes through **every** weight. A 1T dense model is unservable.

Now go and look at what the feed-forward block is doing. It's ~2/3 of the parameters, and its job is roughly "knowledge lookup and transformation."

Here's the question: when the token being processed is `def`, in Python code — do the weights that encode 16th-century poetry contribute anything?

Almost certainly not. They're multiplied anyway. Every token pays for every capability the model has, ~={blue}whether or not that capability is relevant.=~

So can you pay only for the parts you use?

---
# The hospital

A small clinic has one very experienced GP who has read everything — cardiology, dermatology, orthopaedics. Every patient sees him. He's good, but he's a bottleneck, and making him better means one person learning even more.

A large hospital works differently. Forty consultants, each deeply specialised. A patient arrives, **triage** takes thirty seconds and routes them to the two relevant specialists.

The hospital holds vastly more expertise than the clinic — and ~={blue}any individual patient's visit takes the same amount of time as seeing the GP.=~ Total capacity scaled; per-patient cost didn't.

What you *do* pay for: every consultant still needs an office, a salary, and to be physically present, even on days nobody needs them. Hold that thought — it's the entire MoE trade.

> [!NOTE] Mixture of Experts
> Replace the FFN in (some) transformer layers with $N$ parallel FFNs ("experts") plus a small **router** (gating network). For each *token*, the router scores all experts and sends it to the top-$k$ (usually $k=2$). Outputs are combined weighted by the gate scores. ^moe-def

> [!SUCCESS] Core idea
> ~={pink}Decouple total parameters from active parameters.=~ Mixtral 8×7B has ~47B total but activates ~13B per token — it costs like a 13B model at inference and knows far more than one. That decoupling is the only way anyone gets to trillion-parameter models that can actually be served. ^total-vs-active

→ [[Sparsely-Gated Mixture-of-Experts Layer]], and [[Let's Scale Step by Step- Compute-Efficient Hyperparameter Transfer for Large-Scale Mixture-of-Experts]].

---
# The routing problem 🚦

Routing is the whole difficulty, and it has a failure mode that feels obvious in hindsight:

Nothing in the loss says "use all the experts." If expert 3 is slightly better early on, it gets more tokens, so it trains more, so it gets better, so it gets *more* tokens. Within a few thousand steps, ~={red}two experts do everything and the other six are dead weight.=~ You're paying 47B of memory for a 13B model. **Router collapse.**

The fix is an **auxiliary load-balancing loss** added to the training objective, penalising uneven expert utilisation — a deliberate pressure toward spreading tokens out, fighting the rich-get-richer dynamic. (Newer work does it loss-free, by biasing the router's scores directly.)

You also need **capacity factors**: each expert gets a fixed buffer per batch so the hardware can be statically allocated. Overflow tokens get **dropped** — they skip the FFN entirely and pass through on the residual. Yes, really.

> [!TIP] What do experts actually specialise in?
> Not "the medicine expert" and "the French expert" — that intuition is wrong and worth correcting. Measured routing correlates much more with ~={blue}surface-level and syntactic features=~: punctuation, whitespace, numbers, code tokens, particular word classes. Routing is **per token, per layer**, so a single sentence is shredded across many experts. It's less "consult the specialist" than "a very fine-grained learned sparsity pattern." ^expert-specialisation

---
# The trade, honestly 📊

| | Dense 13B | MoE 8×7B (47B, top-2) |
|---|---|---|
| Active params/token | 13B | ~13B |
| **VRAM needed** | 26 GB | ~={red}94 GB=~ 💸 |
| Quality | 13B-ish | ~70B-ish 🥇 |
| Training cost | baseline | Similar FLOPs, better loss per FLOP |
| Batch-1 latency | good | good |
| **High-throughput serving** | simple | ~={red}Hard=~ — see below |

> [!WARNING] MoE is a memory and communication problem
> Two things bite in production:
>
> **1. VRAM.** All experts must be resident. You get dense-70B *memory* costs with dense-13B *compute*. If you're GPU-poor, MoE is the wrong trade — you wanted [[Quantization]] or [[Distillation]].
>
> **2. Batching breaks.** With one sequence, top-2 of 8 is fine. With a batch of 64 tokens routed independently, you're touching **all** experts anyway — so under [[Continuous Batching]] the sparsity advantage erodes, and experts sharded across GPUs turn every layer into an all-to-all network exchange. Expert-parallel serving is a genuinely hard infrastructure problem. See [[ML Infrastructure]] and [[Megatron-LM- Training Multi-Billion Parameter Models Using Model Parallelism]]. ^moe-serving-cost

Modern refinements: **fine-grained experts** (many more, smaller), plus **shared experts** that every token always visits (DeepSeek's design) — so common capability isn't redundantly duplicated across all the specialists.

Related: [[Gated Activation]] (the same gating idea, at the neuron level), [[Foundation Models]].

---

---
![[moe_active_params.png]]
> [!TIP] Reading the chart
> You still have to hold all 47B in memory. You only pay compute for 13B of it.


# ⁉️
Sparsity buys parameters cheaply. But attention has no parameters to sparsify — its cost is positional and quadratic. Which raises a question we've dodged all the way through: how does a model know what *order* the tokens came in at all?

→ [[RoPE]]
