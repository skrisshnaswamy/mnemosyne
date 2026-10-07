---
aliases:
  - Long Context Models
  - Context Extension
  - Million Token Context
tags:
  - llm
  - architecture
  - context
  - performance
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Three separate walls — **quadratic** attention compute, **linear** [[KV Cache]] memory, and **positions the model never trained on** — each with its own fix.
> **Metaphor:** Extending a dinner table. Longer top, more chairs, *and* nobody at the far end can hear the conversation.
> **Where it bites:** "1M context" is a capacity claim. Usable context is measured, not advertised — see [[Lost in the Middle]].

---
A vendor announces **2 million tokens** of context. A competitor says 10 million. Somebody in the meeting asks whether this means [[RAG]] is dead — just paste the whole corpus in.

Before answering, price one request at 2M tokens on a 70B model:

```
Prefill attention: (2M)² pairwise scores, per head, per layer
KV cache:          ~2M tokens × ~0.8 MB/1k tokens ≈ 1.6 TB   (MHA, fp16)
Latency to TTFT:   minutes
Cost per request:  £££ before a single output token
```

Meanwhile a [[Reranking|reranked]] RAG call feeds it 4,000 tokens, returns in 900ms, costs a fraction of a penny, and **cites its sources**.

So: what did "2 million" actually buy?

---
# The dinner table

You've got eight for dinner and a table for six. Easy: put the extension leaf in.

Do it again. And again. Now the table seats forty and three separate things have gone wrong, and they're **not the same problem**:

1. **Conversation got quadratic.** Six people can all talk to each other. Forty can't — the number of pairs exploded, and the room is just noise. *(Attention compute.)*
2. **You ran out of room.** Forty chairs, forty place settings — the physical stuff scales linearly and you've filled the house. *(KV cache memory.)*
3. **The far end can't hear.** Someone at seat 38 is technically at the table and is functionally absent. *(Positional degradation and [[Lost in the Middle]].)*

Three walls. People discuss "long context" as one problem and then wonder why one fix doesn't help.

---
# Wall 1 — attention is $O(n^2)$ compute

Hits **prefill**. 4× the tokens = 16× the attention work.

| Fix | Idea | Cost |
|---|---|---|
| [[Flash Attention]] | Never materialise the $n\times n$ matrix; tile through SRAM | ~={blue}Exact, no quality loss=~ ✅ — still $O(n^2)$ FLOPs, but memory-linear and ~5–10× faster |
| **Sliding window** (Mistral) | Each token attends to the last $w$ only; stacked layers give an effective receptive field | Approximate |
| **Sparse / dilated** (Longformer) | Local window + a few global tokens | Approximate |
| **Linear attention / SSMs** (Mamba) | $O(n)$ via a recurrent state | Different architecture; weaker exact recall |
| **Chunked prefill** | Slice it so it doesn't stall other users | Scheduling, not maths |

> [!NOTE] Flash Attention is not an approximation
> Worth being precise, because it's constantly miscategorised: Flash Attention computes **exactly** the same result as standard attention. It's an IO-aware *kernel* — same maths, fewer HBM round-trips. Sliding-window and sparse attention genuinely change the maths. ^flash-is-exact

---
# Wall 2 — the KV cache is $O(n)$ memory

Hits **decode**, and this is usually the binding constraint in production. See [[KV Cache#^concurrency-formula|the concurrency formula]]. Fixes: [[Grouped Query Attention]] (4–8×), KV [[Quantization]] (2–4×), **PagedAttention** (kills fragmentation), sliding-window eviction, and cross-layer KV sharing.

Even with all of them, 1M tokens × many users is enormous. ~={red}Advertised context and *concurrent* advertised context are very different products.=~

---
# Wall 3 — the model was trained on 4k

Positions beyond training are meaningless to [[RoPE]] unless you intervene — position interpolation, NTK scaling, YaRN, or raising `rope_theta` and continuing pre-training. See [[RoPE#^rope-extrapolation|RoPE extrapolation]].

But even when the *mechanics* work, the **behaviour** degrades: attention dilutes across more candidates, and mid-context recall sags into the U-shape of [[Lost in the Middle]].

> [!SUCCESS] Core idea
> ~={pink}A model *accepting* $n$ tokens and a model *using* $n$ tokens are different claims.=~ Context length is a **specification**; effective context is an **empirical measurement** on your task. Never quote the first when someone is asking about the second. ^advertised-vs-effective

---
# So: long context or RAG? 🥊

The honest answer is *both, for different jobs* — and the boundary is about where the information lives.

| | **Long context** | **[[RAG]]** |
|---|---|---|
| Corpus size | Fits in the window | Unbounded ✅ |
| Cost per query | High, scales with length | Low, flat |
| Latency | Slow prefill | Fast |
| Freshness | Re-send everything | Edit one row ✅ |
| Attribution | Weak | Citations ✅ |
| Cross-document reasoning | **Excellent** ✅ | Limited by retrieval |
| Access control | All or nothing | Per-document ✅ |
| Best at | *One* big thing: a codebase, a contract, a long meeting | *Many* things: a knowledge base, a corpus |

> [!TIP] The pattern that actually wins
> **Retrieve broadly, then think long.** Use retrieval to get from 10M tokens down to 100k of plausibly-relevant material, then let a long-context model reason across *all* of it at once — which is exactly the cross-document reasoning RAG's top-5 chunks can't do.
>
> And **cache the prefix**: if the same 80k-token document is queried repeatedly, prompt caching turns an expensive prefill into a cheap one. Long context is at its best when the *same* context is reused. 💡

Related: [[LatentPress- Context Compression Beyond Text and Vision]], [[SAS- Simple Attention Sparsification via End-to-End Optimization of Context Ranking]], [[Select, Compress, Reinvest- A Controlled Study of Visual-Token Allocation in Long-Video MLLMs]].

---

---
![[attn_cost_quadratic.png]]
> [!TIP] Reading the chart
> Quadratic memory is what made long context unaffordable. Making it linear is what made it a product.


# ⁉️
Everything up to here makes the model a better *reader*. It still can't check today's price, run a query, or send an email. It can only produce text.

→ [[Tool Use]]

*End of the [[Attention]] path — the four attacks on the $n^2$ bill (move fewer bytes, store less, compute fewer pairs, drop the softmax) all land here. ↩ back to* [[Attention]]
