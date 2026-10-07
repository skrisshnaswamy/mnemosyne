---
aliases:
  - LiteLLM
  - OpenRouter
  - LLM Proxy
  - LLM Gateway
  - Model Routing
tags:
  - llm
  - engineering
  - infrastructure
  - tooling
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** One endpoint in front of every provider — so routing, fallbacks, keys, spend caps, caching and logging are **configured once** instead of re-implemented in fourteen services.
> **Metaphor:** The building's fuse box and meter. Every circuit comes through one board: sub-metered, breakered, and switchable to the generator without rewiring a single room.
> **Where it bites:** The first time a provider 503s for forty minutes, or the first time finance asks what the support bot cost in August.

---
Count what's actually in your estate:

```
14 services · 4 providers · 9 API keys across 6 secret stores
```

Two things happen in the same month.

**Tuesday, 14:10.** A provider returns 503 for forty minutes. Six of the fourteen services are down. Six teams each discover it independently, and four of them ship a fallback that afternoon — four different fallbacks, four different retry policies, one of which [[Retries and Fallbacks|triples the load]] on the provider that's already struggling.

**Month end.** Finance asks what the support bot cost in August. The invoice is one number per provider. The bot shares a key with two other services. ~={red}Nobody can answer, and nobody can be given a budget they can't measure.=~

Neither of these is an AI problem. Both are the shape of a problem that already has a standard answer in every other part of your infrastructure.

---
# The fuse box

A building doesn't give each room its own connection to the grid. Everything comes through one board in a cupboard, and that board does four jobs nobody thinks about:

![[model_gateway_fuse_box.png]]

- **One supply, many circuits.** Rooms don't know or care which substation is feeding them.
- **Sub-meters.** You can see that the kitchen drew 40% of the building's power, without asking the kitchen.
- **Breakers.** A circuit that draws too much is cut off — *that circuit*, not the building.
- **A changeover switch.** Mains fails, the generator takes over, and no room is rewired.

Every one of those maps onto a line item in the incident above. And notice the property that makes the board worth having: ~={blue}it is the one place where a decision about supply can be made without touching anything that consumes it.=~

> [!NOTE] Model gateway
> A service (or library) exposing a **single, usually OpenAI-compatible API** in front of many model providers. It holds the provider credentials, issues **virtual keys** per team or per service, applies **routing** and **fallback** rules, enforces **rate limits and budgets**, optionally **caches**, and **logs every request** with tokens and cost attributed to the caller. LiteLLM and OpenRouter are the two common shapes — self-hosted and hosted. ^gateway-def

> [!SUCCESS] Core idea
> Which model serves a request is an **operational decision**, not an application decision. ~={pink}Move it out of the code and it becomes a config change at 14:10 on a Tuesday instead of fourteen deploys.=~ ^model-choice-is-config

---
# Fallbacks: the arithmetic that justifies the whole thing 🧮

Say your primary provider is up 99.5% of the time — about **3.6 hours of downtime a month**. Add a second provider at 99.5%, and assume the failures are roughly independent (different companies, different data centres):

$$1 - (0.005 \times 0.005) = 0.999975 \;\Rightarrow\; \approx 1.1 \text{ minutes a month}$$

**3.6 hours → 1 minute**, from one configuration block:

```yaml
router:
  - model: chat-default
    primary:  [anthropic/claude-sonnet, openai/gpt-4o]     # load balanced
    fallbacks: [bedrock/claude-sonnet, azure/gpt-4o]        # different clouds
    timeout: 25s
    num_retries: 2
```

Two caveats that matter. The independence assumption breaks if both routes are the *same model on the same cloud* — a Bedrock fallback for an Anthropic primary is real diversification; two regions of one provider is less. And fallbacks are only useful if the prompt works on both models, which is an [[Evals|eval]] question you must answer *before* the incident, not during it.

---
# Routing: where the money is

The other job is sending each request to the cheapest model that can do it. From the simulation at the bottom of this note:

```
always the cheap model   $1.20 per 1,000 requests   76.0% accuracy
always the big model    $18.00 per 1,000            92.9%
cheap first, escalate
  the hardest 30%        $6.53 per 1,000            86.7%   ← 2.8× cheaper
  the hardest 70%       $13.65 per 1,000            92.0%   ← 24% cheaper,
                                                              0.9 points down
```

That last row is the one to remember: ~={blue}you can give back a single accuracy point and keep a quarter of the bill=~ — *provided* the escalation signal is any good. The whole curve collapses to a straight line if you escalate at random, so the interesting engineering is the trigger, not the routing.

| Escalation trigger | Signal quality | Notes |
|---|---|---|
| **Task type** (classify → cheap; draft → big) | Good | Static, free, do this first 🥇 |
| **Input length / complexity heuristic** | Fair | Cheap, blunt |
| **Cheap model's own confidence / refusal** | Fair–good | Free-ish; it *underestimates* its failures |
| **A verifier failed** (schema, tests) | **Excellent** | Only where verification exists — see [[Reflection]] |
| **A small classifier trained on your logs** | Best | Needs the logs the gateway is already collecting |

---
# What belongs in the gateway — and what doesn't

| In the gateway ✅ | In your application ❌ |
|---|---|
| Provider credentials, virtual keys | Prompts |
| Routing, load balancing, fallbacks | Which *tools* exist |
| Rate limits, per-team budgets, hard caps | Conversation state |
| Retries, timeouts, circuit breaking | The agent loop |
| Exact-prefix response caching | Business logic |
| Token/cost logging per caller | [[Guardrails]] that need domain context |

> [!WARNING] Two ways a gateway makes things worse
> **1. It's now a single point of failure on every request.** A proxy that dies takes down what it was protecting. Run it with several replicas, health-check the *upstreams* not just the proxy, keep the added latency where it belongs (single-digit milliseconds against a 3-second call — measure it), and make sure the client can fall back to a direct provider call if the gateway is unreachable. Some teams run it in **library mode** in-process for exactly this reason.
>
> **2. Semantic caching.** Exact-prefix caching is safe and boring — same bytes, same answer ([[Prompt Caching]]). *Semantic* caching returns a cached answer for a *similar* question, and "what's our refund window for **Enterprise**?" is similar to "what's our refund window for **Basic**?". ~={red}You will serve the confidently wrong answer, fast, and it will be invisible in the logs because there was no model call to inspect.=~ Leave it off unless you have an eval that specifically hunts for it. ^gateway-risks

> [!TIP] Turn it on for the data even if you want none of the features
> Point everything at a gateway on day one, with no routing rules at all. You immediately get: cost per service, cost per user, p50/p99 per model, token distributions, and a complete request log to build [[Evals|eval sets]] and [[Agent Evaluation|regression sets]] from. That log is the input to every optimisation in [[Cost and Latency]] — and you cannot reconstruct August once August is over. 📊

Related: [[Retries and Fallbacks]], [[Cost and Latency]], [[Observability and Tracing]], [[Prompt Caching]], [[Agent Frameworks]].

---

---
![[model_routing_cost_curve.png]]
> [!TIP] Reading the chart
> The blue curve is the menu the gateway gives you; the two dots are the only two options you have without one. The star is the point most products should sit at, and it's a config change rather than a code change.

---
> [!SUCCESS] If you remember one thing
> ~={pink}Which model serves a request is an operational decision, not an application one.=~ Move it into config and a bad Tuesday becomes a config change instead of fourteen deploys.

---
# ⁉️
The request has been routed, the model is generating, and the user is looking at a blank box. There's no reason to make them wait for the last token before showing them the first one — and the moment you don't, several things quietly break.

→ [[Streaming]]
