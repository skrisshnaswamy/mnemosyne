---
aliases:
  - KV Caching
  - Key-Value Cache
  - PagedAttention
tags:
  - llm
  - inference
  - performance
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Cache every past token's **K** and **V** so each new token does $O(n)$ work instead of $O(n^2)$ — and that cache is the thing that eats your GPU memory.
> **Metaphor:** A detective's case board. Each new lead gets pinned up once; you never re-interview the old witnesses.
> **Where it bites:** It's *the* limit on concurrent users. Batch size, max context, and cost per request are all really questions about the KV cache.

---
You're generating a sentence, token by token. You've produced 500 tokens so far, and you're about to produce 501.

To produce it, the model runs [[Query, Key, and Value (QKV)|attention]]: token 501's **query** is compared against the **keys** of all 500 previous tokens, and blends their **values**.

So compute the keys and values for tokens 1…500. Now produce token 502 — compute keys and values for tokens 1…501. Then 1…502.

Notice something about token 7's key: it's the same number every single time. Token 7's key depends only on token 7 and the tokens before it — and those are ~={blue}frozen, because generation only ever appends.=~ ([[Causal Attention]] is exactly the reason nothing later can change it.)

So why would you ever compute it twice?

---
# The case board

A detective works a case. Each witness interviewed gets a card pinned to the board: **who they are** (the key) and **what they said** (the value).

A new lead comes in. He doesn't re-interview all 47 witnesses — he stands in front of the board, scans the cards, and finds which ones connect. Then he pins the new card up and moves on.

That's the KV cache. Interview once, pin forever.

> [!NOTE] KV Cache
> Storage for the **key** and **value** vectors of every token processed so far, per layer, per attention head. At each decode step you compute K and V for the *one new token*, append them to the cache, and attend over the whole thing. ^kv-cache-def

> [!TIP] Why K and V but not Q?
> The query is "what am I looking for **right now**." It's used once, this step, and then it's meaningless — the next token has its own question. Keys and values are "who I am" and "what I offer" — *permanent* properties other tokens will want to consult forever. 🔍 ^why-not-q

> [!SUCCESS] Core idea
> Without the cache, generating $n$ tokens costs $O(n^2)$ — you'd redo all history at every step. With it, each step is $O(n)$ and generation is $O(n^2)$ total *reads* but no recomputation. ~={pink}You trade compute for memory. And then memory becomes the entire problem.=~ ^cache-tradeoff

---
# The bill 💸

$$\text{KV bytes} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{seq len} \times \text{batch} \times \text{bytes per value}$$

The leading $2$ is K and V. Put real numbers in — Llama-2-13B (40 layers, 40 heads, $d_{head}$=128), fp16, one user, 8k context:

$$2 \times 40 \times 40 \times 128 \times 8192 \times 2 \approx 6.7\ \text{GB}$$

**For one user.** The weights are 26 GB. On an 80 GB A100 you have ~54 GB spare → about **8 concurrent users** before you're out of memory.

> [!WARNING] This is the real capacity limit
> Not FLOPs. Not the model size. ~={red}Your max concurrency is (VRAM − weights) ÷ KV-per-request.=~ Every serving decision — batch size, max context you advertise, GPU you buy — falls out of this one division. ^concurrency-formula

---
# How the cache gets attacked

| Technique | What it does | Typical saving |
|---|---|---|
| [[Grouped Query Attention\|GQA / MQA]] | Many query heads share one K/V head | 4–8× 🥇 |
| **KV quantization** | Store the cache in int8/fp8 — see [[Quantization]] | 2–4× |
| **PagedAttention** | Stop pre-allocating contiguous blocks | ~2× effective |
| **Sliding window / eviction** | Only keep the last $w$ tokens | Bounded, but lossy |
| **Prefix / prompt caching** | *Share* one cache across requests with the same prefix | Huge for fixed system prompts |

**PagedAttention** deserves the extra sentence, because the insight is not about attention at all. Naively you must reserve the cache for the *maximum* length up front — a request that might run to 4,096 tokens but stops at 200 reserves all 4,096. Most of your VRAM is holding empty space. vLLM's fix is to borrow **virtual memory paging** from operating systems: allocate the cache in small fixed blocks, keep a block table per sequence, allocate on demand. Fragmentation drops from ~60–80% waste to a few percent, and identical prefixes can *share* physical blocks copy-on-write.

→ [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]]

> [!TIP] The one-line diagnostic
> If throughput collapses when users send long prompts, or you keep hitting OOM at a batch size that "should" fit — it's not the model, it's the cache. Compute the formula above before blaming anything else. 🧮

See [[ML Infrastructure]] and [[GPU processing]] for where this sits in the memory hierarchy.

---

---
![[kv_cache_growth.png]]
> [!TIP] Reading the chart
> Every concurrent user carries their own cache. This — not the model weights — is usually what decides how many people you can serve at once.


# ⁉️
If the cache is what's capping you, the most direct fix is to make it structurally smaller — not by compressing it, but by having fewer K and V heads in the first place.

→ [[Grouped Query Attention]]
