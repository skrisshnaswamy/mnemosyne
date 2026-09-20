---
aliases:
  - Inference-Time Compute
  - Test Time Scaling
  - Reasoning Models
  - Thinking Budget
tags:
  - llm
  - reasoning
  - inference
  - scaling
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Buy capability with **inference tokens** instead of training FLOPs — think longer, sample more, verify, and pick.
> **Metaphor:** A chess player given 3 hours instead of 3 seconds. Same player, far better moves.
> **Where it bites:** It moved the cost from a one-off training bill to a **per-request** bill. Your unit economics change shape.

---
For a decade the scaling story had one shape: want a better model, make it bigger and train it on more data. [[Scaling Laws for Neural Language Models]], then [[Training Compute-Optimal Large Language Models (Chinchilla)]] refining the ratio. Capability was something you **bought once, at training time**, and inference was the cheap part you tried to make cheaper.

Then a 70B model started beating a 400B model on competition maths — by taking 30 seconds per question instead of 2.

Hold on. The weights are fixed. Nothing was learned during those 30 seconds. So what did the extra time *buy*?

---
# Blitz vs classical

A grandmaster playing **blitz** (3 seconds a move) and playing **classical** (3 hours a move) is the same grandmaster with the same knowledge.

In blitz she plays her strongest intuition — the first move that looks right. In classical she does something else entirely:

- **explores** several candidate moves rather than one
- **plays each one forward** and sees where it lands
- **notices** that her favourite loses a rook in six
- **backtracks** and takes the second-best line

Her *knowledge* didn't change. Her **search over what she knows** did. And she plays hundreds of Elo points stronger.

That's the whole idea. [[Chain of Thought]] was the first click of this dial; this is turning it all the way up.

> [!NOTE] Test-time compute
> Spending additional computation at **inference** to improve a single answer, rather than additional computation at training to improve the model. Three families:
> 1. **Think longer** — extended internal reasoning before answering (o1/o3-style "thinking" traces)
> 2. **Sample wider** — $N$ independent attempts, then aggregate (best-of-N, [[Chain of Thought|self-consistency]] majority vote)
> 3. **Search and verify** — generate candidates, score them with a verifier or reward model, expand the promising ones (tree search, beam over reasoning steps) ^ttc-def

> [!SUCCESS] Core idea
> ~={pink}Capability is no longer a single number attached to a checkpoint — it's a function of how much you're willing to spend on the request.=~ The same model is "weaker" or "stronger" depending on its thinking budget, which is a genuinely new thing to have in a systems design conversation. ^capability-is-a-dial

---
# Why generating many and picking one works ✅

It rests on one asymmetry, and the asymmetry is the whole engine:

> **Verifying a solution is easier than producing one.**

Factoring a number is hard; checking the factors is multiplication. Writing a correct program is hard; running the tests is cheap. So if you can generate 64 candidates and **verify** them, you convert "be right first time" into "be right *once* in 64" — which is a far lower bar.

This is why test-time compute works brilliantly where verification is **cheap and objective**, and much less well where it isn't:

| Domain | Verifier | Gains |
|---|---|---|
| Competition maths | The answer checks out | Enormous 🥇 |
| Code | Unit tests run | Enormous 🥇 |
| Formal proofs | A proof checker | Enormous |
| Multi-hop retrieval QA | Citation checking, partially | Moderate |
| Essay quality, strategy, taste | ~={red}No verifier=~ | Modest — you're voting on vibes |

---
# The consequences you'll have to live with ⚖️

> [!WARNING] The cost curve inverted
> Training cost is capex, paid once, amortised over every request. Test-time compute is **opex, paid per request, per user, forever.** A "thinking" answer can cost 10–100× a normal one and take 30× as long.
>
> So the design question stops being "which model?" and becomes ~={blue}"which requests deserve to think?"=~ Route: cheap model for the easy 90%, thinking model for the hard 10%. That routing decision is now a core piece of product engineering, not an optimisation. ^ttc-economics

> [!TIP] Practical shapes
> - **Budget-aware prompting** — many APIs expose an explicit thinking-token budget. Treat it as a per-endpoint config with a real number, not a default.
> - **Best-of-N with a real verifier** — if you have tests, a linter, a schema, a SQL `EXPLAIN`: use them. Far better than a judge model.
> - **Self-consistency** — no verifier? Sample 5 chains and majority-vote. Cheap, works, hard to beat for the effort.
> - **Reasoning traces are billed and usually hidden** — you pay for tokens you never see. Watch this on your invoice. 💸

Related: [[Prefix Sliding for efficient test-time scaling]], [[TTPO- Test-Time Policy Optimization]], [[τ_0-VLA- a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation]]. And note the loop back to training: traces that *worked* become [[Fine-Tuning]] data, so search at inference becomes knowledge in the weights — the [[Distillation]] of reasoning.

---

---
![[test_time_compute.png]]
> [!TIP] Reading the chart
> Sampling more only helps if something can tell you which sample was right.


# ⁉️
Thinking longer produces better content. It also produces *more* content — prose, caveats, restatements. If a program has to consume it, prose is a liability.

→ [[Structured Output]]
