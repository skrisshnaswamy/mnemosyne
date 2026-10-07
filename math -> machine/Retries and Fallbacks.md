---
aliases:
  - Exponential Backoff
  - Fallback Model
  - Idempotency
  - Circuit Breaker
tags:
  - llm
  - engineering
  - infrastructure
  - failure-mode
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A retry converts a **failure** into **latency and money**, so the only real questions are *which errors are worth converting* and *when to stop converting them*.
> **Metaphor:** Knocking on a door. Knock again by all means — but if they're busy, your knocking is part of why.
> **Where it bites:** Retries multiply load at exactly the moment the upstream is weakest. That's how a slow provider becomes an outage.

---
14:10. A provider starts timing out on **20%** of calls. Your client retries up to three times. Here's what the dashboards say, measured:

```
                         no retry        retry up to 3×
requests that fail        19.5%            0.90%      ← the win
p50 latency               2.1 s            2.3 s      ← unchanged
p99 latency              30.0 s           64.8 s      ← doubled
calls billed per request   1.00             1.24      ← +24% spend
```

The retry did its job: **19.5% → 0.9%**. It also took your p99 from thirty seconds to sixty-five, and added a quarter to the bill *for every request in the system*, including the healthy ones.

Two teams look at that table and draw opposite conclusions. ~={blue}Which column you care about is a product decision, and if nobody makes it explicitly, the default library setting makes it for you.=~

---
# Knocking on the door

You knock. Nothing. What do you do?

Knock again straight away — and again, and again. If nobody's in, you've wasted your knuckles. But if they're in and **busy**, you've just made them more busy: every knock is an interruption competing with the thing that would have got them to the door.

![[retry_knocking_on_the_door.png]]

So you wait a bit, then longer, then longer still. That's **exponential backoff**, and it's a statement about the other side, not about you: *the longer this has gone on, the more likely it is that hurrying makes it worse.*

Now the part people skip. Four hundred people are knocking on four hundred doors after a power cut, and the power comes back. If everyone's "wait 4 seconds" starts at the same instant, everyone knocks at the same instant, the system falls over, and everyone waits 8 seconds — together. That's the **thundering herd**, and the fix is a coin flip: `sleep = random(0, backoff)` — **jitter**.

And eventually: stop knocking and try the other door. That's a **fallback**, and it's a different decision entirely.

> [!SUCCESS] Core idea
> A retry doesn't create capacity; it **re-queues your request at the back**, plus interest. ~={pink}It's the right move when the failure is transient and independent, and it is the wrong move — actively harmful — when the failure is capacity.=~ Telling those apart is what the status code is for. ^retry-is-a-bet

---
# Which errors to retry — the table that removes most bugs

| Signal | Retry? | Why |
|---|---|---|
| `429 Too Many Requests` | ✅ — **honour `Retry-After`** | You are over a limit. Backoff is the protocol |
| `500 / 502 / 503 / 529` | ✅ with backoff | Transient server-side |
| Connection reset, DNS blip | ✅ | Network |
| **Read timeout** | ⚠️ Only if idempotent | The work may have *succeeded*. See below |
| `400 / 422` bad request | ❌ **Never** | The prompt is malformed. It'll be malformed again |
| `401 / 403` | ❌ Never | Credentials. Retrying just logs more failures |
| `404` model not found | ❌ Never → **fallback** | Wrong model name or region |
| Content filter / refusal | ❌ Never | Deterministic. Retrying is [[Jailbreak\|jailbreaking your own system]] |
| **Schema validation failed** | ✅ *once*, with the error fed back | Not a network retry — it's [[Reflection\|a critique loop]] with a real verifier |

That last row is worth separating: retrying a *validation* failure with the parser's error message in the prompt is one of the highest-yield retries you can do, and it has nothing to do with the network. Retrying a `400` because "retry everything" was easier to write is pure waste — you pay for it and it can never succeed.

---
# The policy, written out 🧰

```python
MAX_ATTEMPTS = 3
DEADLINE     = 45          # seconds for the WHOLE operation — the real limit
BASE, CAP    = 0.5, 8.0

for attempt in range(MAX_ATTEMPTS):
    if time.monotonic() > deadline: break
    try:
        return call(model, messages, idempotency_key=key, timeout=per_try)
    except Retryable as e:
        if attempt == MAX_ATTEMPTS - 1: raise
        wait = e.retry_after or random.uniform(0, min(CAP, BASE * 2**attempt))
        time.sleep(wait)                      # full jitter
```

Four things in there are load-bearing:

1. **A deadline, not just a count.** Three attempts at a 30-second timeout is a 90-second worst case, and nobody chose 90 seconds. The deadline is the number your users experience; the attempt count is an implementation detail beneath it.
2. **Full jitter**, not "backoff ± 10%". Uniform over the whole interval is what actually de-synchronises a herd.
3. **`Retry-After` wins** over your own formula. The server knows more than you do.
4. **An idempotency key.** A timed-out request may have *completed* — the model generated, the tool fired, the email went. Without a key your retry is a second action. Same discipline as [[Agent State and Checkpointing#^replay-reruns-effects|the effects ledger]]. 🔑

> [!WARNING] The malignant case, which the numbers above understate
> The measured +24% assumes your retries don't cause the failures. When the upstream is capacity-limited, they do: errors rise → everyone retries → offered load rises 2–3× → more errors → more retries. That's **congestion collapse**, and it's how a provider having a bad ten minutes becomes your two-hour incident.
>
> Two things stop it, and you need both:
> - **A circuit breaker.** After $N$ consecutive failures on a route, stop calling it for $T$ seconds and fail fast (or go straight to the fallback). Probe with a single request before reopening.
> - **A retry budget.** Cap *total* retries at ~10% of total requests. Under a broad outage that turns 3× amplification into 1.1× — the single most effective line in the whole policy. ~={red}Per-request retry limits do nothing about system-wide amplification; only a global budget does.=~ ^retry-budget

---
# Fallbacks — a different decision with a different risk

| Rung | Change | Preserves quality? | Watch out for |
|---|---|---|---|
| Same model, **other region/cloud** | Infrastructure only | ✅ Identical | Different rate limits, slightly different versions |
| Same model, **other provider** (Bedrock/Vertex/Azure) | Hosting | ✅ Near-identical | Feature gaps: tool-call format, caching, JSON mode |
| **Different model** | ~={red}Behaviour=~ | ❌ Not automatically | Must be on your [[Evals\|eval set]] *before* the incident |
| **Degraded response** | Product | ❌ | Cached answer, template, or an honest "try again shortly" |

The third rung is where teams get hurt. A fallback model that has never been evaluated is an **untested code path that only executes during an incident** — the worst possible time to discover that your prompt relies on a JSON mode it doesn't have. Run your eval set against the whole ladder on every release, and wire the ladder in the [[Model Gateway]] so it's config rather than fourteen copies of this loop.

> [!TIP] Three rules worth memorising
> - **Retry the network; never retry a decision.** A refusal, a 400 and a content filter are decisions.
> - **Budget the deadline, not the attempts.** Users feel seconds.
> - **You cannot retry a [[Streaming|stream]] after the first byte.** Decide transparently before then; after that it's a visible "regenerating", which is a product choice, not a library one. 🔁

Related: [[Model Gateway]], [[Cost and Latency]], [[Observability and Tracing]], [[Agent State and Checkpointing]], [[Streaming]].

---

---
![[retry_storm_cost.png]]
> [!TIP] Reading the chart
> The left panel is the trade in one picture: the orange line is what a failure costs you, and the red line is what *not accepting* that failure costs everyone else. The right panel is the invoice — and it's charged against your whole traffic, not just the requests that failed.

---
> [!SUCCESS] If you remember one thing
> A retry converts a failure into latency and money. ~={pink}Retry the network; never retry a decision=~ — and cap retries globally, because a per-request limit does nothing about system-wide amplification.

---
# ⁉️
Tokens re-sent, calls retried, models routed, caches hit or missed. Every one of those is a line on the same bill, and almost nobody can say which line is the big one. It's worth doing the arithmetic once, properly.

→ [[Cost and Latency]]
