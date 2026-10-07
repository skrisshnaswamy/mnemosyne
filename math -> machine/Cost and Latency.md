---
aliases:
  - Token Economics
  - LLM Cost
  - Cost per Request
  - Latency Budget
tags:
  - llm
  - performance
  - engineering
  - metrics
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** An agent run's bill is **input tokens you already sent**, re-sent; its clock is **decode**. Those are two different levers and most teams pull the wrong one for each.
> **Metaphor:** A phone line where, before every sentence you speak, you must read the whole conversation back from the beginning — and you're billed by the word.
> **Where it bites:** Input grows *quadratically* in steps. Double the steps and you quadruple the bill.

---
The invoice says £42,000. The price sheet says $3 per million input tokens and $15 per million output.

So: what did **one** agent run cost, and on what?

Here's the honest accounting for a 20-step run — 1,200-token system prompt, 2,800 tokens of tool schemas, ~180 tokens generated and ~700 tokens of tool output per step:

```
input tokens billed   250,200      →  $0.751   (93% of the cost)
output tokens billed    3,600      →  $0.054   ( 7% of the cost)
                                      ───────
                                       $0.805 per run
```

Now the part that should be uncomfortable. The *unique* text in that run — the system prompt, the tools, the question, and all twenty tool results — is **18,150 tokens**.

$$\frac{250{,}200 - 18{,}150}{250{,}200} = \textbf{93\% of your input bill is words you had already sent.}$$

And output tokens, at **five times the unit price**, are 7% of the total. ~={blue}Everyone's instinct is to make the answers shorter. The answers were never the problem.=~

---
# The phone line

Picture a phone call with a strange rule: before each new sentence, you must read the entire conversation back from the beginning. Aloud. And you're billed by the word.

![[cost_latency_phone_line.png]]

Sentence 1 is cheap. Sentence 20 means reading nineteen sentences first.

That's not a metaphor for the API; it *is* the API. The [[LLM Engineering#^stateless|model is stateless]] — a conversation is an illusion produced by re-sending the transcript — so the tokens you're billed for on step $s$ include everything from steps $1 \dots s-1$.

Which makes the total a sum, and the sum has a shape:

$$\text{input tokens} = \underbrace{S \times B}_{\text{preamble, } S \text{ times}} \;+\; \underbrace{\frac{g \, S(S-1)}{2}}_{\text{the transcript, re-sent}}$$

with $B$ = the fixed preamble, $g$ = tokens added per step, $S$ = steps. Put the run's numbers in:

$$20 \times 4{,}150 \;+\; \frac{880 \times 20 \times 19}{2} \;=\; 83{,}000 + 167{,}200 = \mathbf{250{,}200}$$

> [!SUCCESS] Core idea
> The second term is **quadratic in the number of steps**. ~={pink}Twenty steps cost four times what ten steps cost, not twice.=~ Every other lever in this note is linear; this one is the only quadratic, which is why "fewer steps" beats everything else you can do. ^cost-is-quadratic-in-steps

---
# The clock is a different story ⏱️

Money lives in input. Time lives in **output**. One step of that same run:

```
network + queue        0.12 s
prefill (18k context)  0.64 s   ← cacheable
decode (180 tokens)    3.27 s   ← 59% of the step
tool execution         0.81 s
re-prefill next step   0.68 s
                       ──────
                       5.52 s  →  a 20-step run = 1.8 minutes
```

Decode owns the clock because it's memory-bandwidth-bound and strictly sequential — one token at a time, the whole model read from HBM for each ([[Prefill and Decode]]). The tool call everybody blames is 15%.

So the two levers are almost perfectly crossed:

| | Effect on **cost** | Effect on **latency** |
|---|---|---|
| Shorten the **output** | 7% of the bill — negligible | **Large** — it's 59% of the clock |
| Shorten / cache the **input** | **Large** — it's 93% of the bill | Small (prefill is 12%) |
| Remove a **step** | **Quadratic** 🥇 | Linear |

---
# The levers, with measured numbers

Every figure below is from the simulations in this area's notes, not a guess:

| Lever | Cost effect | Latency effect | Effort |
|---|---|---|---|
| **[[Prompt Caching\|Cache the prefix]]** with a rolling breakpoint | $0.805 → **$0.212** (−74%) | TTFT −0.5 s/step | Config 🥇 |
| **[[Context Engineering\|Compact]] every 10 turns** | turn-40 input $0.30 → **$0.036** | Less prefill | A day |
| **[[Planning and Decomposition\|Fewer steps]]** (20 → 12) | −64% (quadratic) | −44% | Prompt + plan |
| **Parallel tool calls** | 0 | 4 calls: **9.2 s → 2.8 s** | An afternoon |
| **[[Model Gateway\|Route]] the easy 30% to a cheap model** | −24% at −0.9 accuracy pts | Faster on those | Config |
| **Truncate tool results** | −40%+ of growth | −prefill | An hour |
| Shorter system prompt | Tiny once cached | ~0 | Don't bother |
| Shorter answers | ~={red}−7% at most=~ | Large | Product decision |

> [!WARNING] The three numbers people quote that mean nothing on their own
> **"Cost per 1M tokens."** Nobody buys tokens; they buy *answered questions*. The unit that matters is **cost per resolved task** — a model at 3× the price that halves the step count is cheaper.
> **"Average latency."** Streaming makes the mean meaningless. Report **TTFT** and **tokens/sec** separately ([[Streaming]]), and report p95 rather than p50, because an agent's distribution has a long tail by construction.
> **"Tokens used."** Split it four ways or it tells you nothing: *cached input · uncached input · output · thinking tokens*. They differ in price by up to 50×. ^meaningless-metrics

> [!TIP] The per-request budget — write it down before you build 📋
> Pick the numbers first and let them constrain the design, the way a latency budget does in any other system:
> ```
> target:   ≤ $0.05 and ≤ 8 s per resolved task
> implies:  ≤ 16k input tokens per call  (at $3/M)
>           ≤ 6 steps                     (quadratic! this is the binding one)
>           ≥ 70% cache hit rate
>           parallel tool calls mandatory
> ```
> Then instrument against it from day one — [[Observability and Tracing]]. ~={blue}A budget discovered from an invoice is a post-mortem; a budget chosen up front is a design constraint.=~

**One more thing that surprises people:** reasoning/thinking tokens are billed as *output*, at output prices, and are usually invisible in the response. A model that "thinks" for 2,000 tokens before writing a 200-token answer has a bill ~10× the one you'd estimate from what you can see. Log `reasoning_tokens` explicitly or your cost model is wrong by an order of magnitude. See [[Test-Time Compute]].

Related: [[Prompt Caching]], [[Context Engineering]], [[Model Gateway]], [[Retries and Fallbacks]], [[Prefill and Decode]], [[KV Cache]], [[Streaming]].

---

---
![[agent_run_token_spend.png]]
> [!TIP] Reading the chart
> The left panel is the quadratic, drawn: each bar is one step, and almost all of it is the bars to its left. The right panel is the same run split into what you *had* to send and what you sent *again*.

![[agent_step_latency_waterfall.png]]
> [!TIP] Reading the chart
> Teams optimise the orange block because it's the one with an external dependency. It is 15% of the step. The red block is 59% and it's the one you shorten by generating fewer tokens.

![[parallel_vs_sequential_tools.png]]
> [!TIP] Reading the chart
> The gap is entirely model round-trips, not tool time. Issuing four independent calls in one turn instead of four turns is a config change in most frameworks and it's worth 3.3×.

---
> [!SUCCESS] If you remember one thing
> ~={pink}93% of an agent's input bill is words it already sent, and it grows with the square of the step count.=~ Money lives in input; time lives in decode. Don't pull the wrong lever for either.

---
# ⁉️
Every number in this note came from measurement. On a system where the same input produces a different trajectory, a different cost and a different latency every time, "measurement" can't mean log lines — it has to mean something that can reconstruct an entire run.

→ [[Observability and Tracing]]
