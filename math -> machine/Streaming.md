---
aliases:
  - Token Streaming
  - SSE
  - Server-Sent Events
  - Streaming Responses
tags:
  - llm
  - inference
  - engineering
  - performance
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Send each token as it's produced — the total time is **identical**, but the wait the user experiences collapses from the whole generation to the **time to first token**.
> **Metaphor:** Reading over someone's shoulder as they write. They write faster than you read, so after the first line you never wait again.
> **Where it bites:** Streaming breaks validation, guardrails, error handling and retries — every one of which assumed it had the whole answer before anyone saw any of it.

---
Two deployments. Same model, same prompt, same GPU. A 600-token answer takes **11.6 seconds** to generate in both.

```
Deployment A:  blank screen for 11.6s, then the whole answer
Deployment B:  first word at 0.65s, then it writes itself out over 10.9s
```

Users score B as *"much faster"*. Support tickets about slowness drop. Nothing got faster — the last token still lands at 11.6 seconds in both.

Now the number that explains why it isn't a trick. A person reads at roughly **4.5 tokens per second**. The model writes at **55**.

```
600 tokens to read  → ~133 seconds of reading
600 tokens to write →  ~11 seconds of writing
```

~={blue}The writer is twelve times faster than the reader.=~ So once the first word is on screen, the user is never waiting for the model again — they're the bottleneck.

That turns one long wait into one short wait. But what did it cost?

---
# Over the writer's shoulder

Someone is writing you a letter. Two ways to receive it.

**Posted.** They finish, seal it, post it, and you read it whole. Clean: you can check the signature before you read a word, and if they made a mistake on page 2 they tore it up and you never saw it.

**Over the shoulder.** You read as they write. You're into the second paragraph before they've finished the third.

![[streaming_over_the_shoulder.png]]

The second is far nicer to sit through, and it gives up three things you didn't notice you had:

1. You can't **check the whole thing first** — you've read the opening before you know how it ends.
2. If they change their mind on page 2, **you already read page 1**. Retracting it is a conversation, not a delete.
3. If they stop writing halfway, you're holding **half a letter** with no note saying why.

Those three are, exactly, what streaming breaks.

> [!NOTE] Server-Sent Events
> The transport for nearly all token streaming: a plain HTTP response with `Content-Type: text/event-stream`, held open, one `data:` line per chunk.
> ```
> data: {"choices":[{"delta":{"content":"The"}}]}
> data: {"choices":[{"delta":{"content":" refund"}}]}
> data: {"choices":[{"delta":{"content":" window"}}]}
> data: [DONE]
> ```
> One-directional, no handshake, works through ordinary HTTP infrastructure, reconnects on its own. A WebSocket is bidirectional and heavier — you need it when the *client* also sends mid-stream. ^sse-def

> [!SUCCESS] Core idea
> Streaming changes **perceived** latency, not latency. ~={pink}The number the user feels is TTFT; the number your dashboard shows is total duration; those are different SLAs and only one of them is on your graph.=~ Measure and alert on TTFT separately — see [[Prefill and Decode]], because TTFT *is* prefill. ^ttft-is-what-users-feel

---
# What breaks, and what to do about each

| What breaks | Why | Fix |
|---|---|---|
| **JSON validation** | You can't parse an object until it closes | Partial-JSON parser; or stream only the prose field and buffer the rest. See [[Structured Output]] |
| **Tool calls** | Arguments arrive as *string fragments across chunks*, indexed by call | Accumulate per `index`, parse once at `finish_reason` |
| **Output [[Guardrails\|guardrails]]** | You've shown 200 tokens before the check fails | Buffer the first $N$ tokens, or check in ~250-token windows and redact forward |
| **Errors** | `HTTP 200` was sent at token 1; the failure arrives **inside the body** | Clients must parse an error *event*, not a status code. ~={red}Missing this looks like truncation=~ |
| **[[Retries and Fallbacks\|Retries]]** | Bytes are already on the wire | Only transparently retryable **before** the first token. After that: a visible "regenerating" |
| **Proxies** | `proxy_buffering on` / a CDN / gzip silently collects the whole response | `X-Accel-Buffering: no`, disable buffering *and* compression for this route |
| **Usage / cost** | Token counts come in the final chunk, sometimes only on request | Ask for usage in the stream, and handle its absence |

The proxy row is the one that eats an afternoon. Everything is correct end-to-end in dev, you deploy, and streaming is gone — because a load balancer in the middle is politely waiting for the response to finish. ~={red}Test streaming through the real ingress, never against the service directly.=~

> [!TIP] The 30-second diagnostic
> ```bash
> curl -N -s https://your-api/chat -d '{"stream":true,...}' | ts
> ```
> If the lines arrive in one burst at the end, something between you and the model is buffering. Work outwards: app → ingress → CDN. 🔎

---
# Streaming an *agent* is a different problem

For a single call you stream **tokens**. For an [[Agentic Workflows|agent]] the user is waiting through a 110-second run with tool calls in it, and a stream of tokens from the *final* answer starts 105 seconds in. Useless.

What to stream instead, roughly in order of value:

```
step started    → "Searching the policy index…"
tool call       → search_policies(product="Enterprise Plus")
tool result     → "4 documents found"      (a summary, never the payload)
step finished   → ✅
final answer    → token by token
```

This is what a framework means by "stream modes" — [[LangGraph]] separates `values` (state after each node), `updates` (what each node changed) and `messages` (tokens). Pick per surface: a UI wants updates + tokens; a log wants values.

> [!WARNING] Progress is not the same as honesty
> Streaming step names makes a slow agent *feel* responsive, and that's the trap: it also makes a **stuck** agent feel responsive. A loop that searches, fails, and rephrases forever emits a beautiful stream of activity. Always ship a visible **step counter and budget** ("step 7 of 20"), and a stop button that actually cancels the underlying request rather than closing the connection and leaving the run billing in the background. ~={red}A cancelled stream that doesn't cancel the generation is a silent cost leak.=~ ^cancel-must-cancel

> [!TIP] Where TTFT actually goes 🕐
> ```
> auth + routing + gateway      40 ms
> retrieval (if any)           300 ms   ← usually the biggest slice
> prefill of 18k tokens        640 ms   ← halved by [[Prompt Caching]]
> first token emitted            ~1 s
> ```
> Two levers dominate: shorten or cache the prefix, and overlap retrieval with anything else you can. If TTFT is bad and the prompt is long, it's prefill — and prefill has a cache.

Related: [[Prefill and Decode]], [[Sampling Parameters]], [[Cost and Latency]], [[Observability and Tracing]].

---

---
![[streaming_perceived_wait.png]]
> [!TIP] Reading the chart
> The gap between the two lines is pure perception — no token arrives any earlier in absolute terms. Notice the red line is a *slope*: the longer the answer, the more streaming is worth, which is why it matters most exactly where it's hardest to validate.

---
> [!SUCCESS] If you remember one thing
> ~={pink}Streaming changes the wait, not the work.=~ TTFT is the number users feel — and everything that assumed it had the whole answer before anyone saw any of it now needs rewriting.

---
# ⁉️
That 640 ms of prefill is the same 18,000 tokens on every turn of every run, and the GPU recomputes it from scratch each time. It does not have to.

→ [[Prompt Caching]]
