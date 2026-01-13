---
title: "Efficient Memory Management for LLM Serving with PagedAttention (vLLM)"
authors: ["Woosuk Kwon", "Zhuohan Li", "Siyuan Zhuang", "Ying Sheng", "Lianmin Zheng", "Cody Hao Yu", "Joseph E. Gonzalez", "Hao Zhang", "Ion Stoica"]
year: 2023
arxiv: "2309.06180"
url: https://arxiv.org/abs/2309.06180
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, transformers, llm]
---
## The Core Idea

When you serve a large language model, most of the GPU memory is not doing anything useful. That is the whole finding.

Generation is one token at a time. To avoid recomputing everything, you keep the key and value vectors of every past token in memory — the **KV cache**. For a 13B OPT model, one token's KV cache costs 800 KB: $2 \times 5120 \times 40 \times 2$ bytes (key+value, hidden size, layers, FP16). A 2048-token sequence is 1.6 GB. On a 40 GB A100 with 26 GB of weights, you have ~12 GB left, which is about 15,700 token slots total. Whoever uses those slots best wins, because throughput is set by how many requests you can batch, and batch size is set by memory.

The problem: PyTorch wants tensors in one contiguous block. So earlier serving systems reserved a contiguous chunk sized to the request's *maximum possible* length (2048), even if the answer turned out to be 25 tokens. Three kinds of waste follow:

- **Reserved** — slots that will eventually be used, but sit idle now, blocking other requests.
- **Internal fragmentation** — slots inside the chunk that are never used, because the output was short.
- **External fragmentation** — gaps between chunks of different sizes, from the buddy allocator.

Measured: only **20.4% – 38.2%** of KV cache memory in existing systems holds actual token state.

The trick is old and borrowed. Operating systems solved exactly this in 1962 with **virtual memory and paging**. Cut memory into fixed-size pages. Let a program see contiguous *logical* addresses while the *physical* pages are scattered. Allocate physical pages only when needed.

> [!NOTE] PagedAttention
> An attention kernel that reads keys and values from fixed-size blocks scattered in non-contiguous GPU memory, using a block table to find them. Because a sequence no longer needs one contiguous slab, memory can be allocated one block at a time and shared block-by-block between sequences. ^paged-attention

The mapping is: **blocks are pages, tokens are bytes, requests are processes.** Fixed block size kills external fragmentation entirely. Small blocks (default 16 tokens) cap internal fragmentation at under one block per sequence. And because the block table is just a level of indirection, two sequences can point at the *same* physical block — so a shared prompt, or a shared beam-search prefix, is stored once.

What it unlocks: 2–4× throughput over Orca and FasterTransformer at the same latency, with *zero* change to model outputs. Nothing about the maths changes. Only where the bytes live.

## The Methodology

**The block-wise attention.** Standard attention, for query at position $i$:

$$a_{ij} = \frac{\exp(q_i^\top k_j/\sqrt{d})}{\sum_{t=1}^{i}\exp(q_i^\top k_t/\sqrt{d})}, \qquad o_i = \sum_{j=1}^{i} a_{ij} v_j$$

Now group the keys into blocks of $B$ tokens, $K_j = (k_{(j-1)B+1},\dots,k_{jB})$, same for values. The same computation, regrouped:

$$A_{ij} = \frac{\exp(q_i^\top K_j/\sqrt{d})}{\sum_{t=1}^{\lceil i/B\rceil}\exp(q_i^\top K_t\mathbf{1}/\sqrt{d})}, \qquad o_i = \sum_{j=1}^{\lceil i/B\rceil} V_j A_{ij}^\top$$

The kernel fetches one block, does its dot products, moves to the next block. The blocks need not be neighbours in memory. This is algebraically identical to normal [[Attention Is All You Need#Scaled dot-product attention|scaled dot-product attention]] — it is a memory layout change, not a new maths.

**The system (vLLM).** A centralised scheduler plus distributed GPU workers.

- A **block engine** on each GPU grabs one big contiguous chunk of DRAM up front and slices it into physical KV blocks. It does the same on CPU RAM, for swapping.
- A **KV block manager** holds a **block table** per request: logical block index → physical block index, plus a `#filled` counter for the last block.
- Each step: the scheduler picks which sequences to run, allocates any new physical blocks needed, then broadcasts *(input token ids, block tables)* to all workers. Workers run the model, read KV via the block table, write new KV into the assigned physical slots, and send sampled tokens back.

**Walkthrough.** A 7-token prompt with block size 4. Prefill maps logical blocks 0 and 1 to physical blocks 7 and 1. Block 0 is full (4 tokens), block 1 has 3 filled and 1 free. First decode step writes into that free slot and bumps `#filled` to 4. Second decode step finds block 1 full, so it allocates physical block 3 and adds the mapping. Waste at any moment is at most one partially-filled block per sequence.

**Sharing, via reference counts and copy-on-write.** Same idea as `fork()` in an OS.

- *Parallel sampling* (n samples, one prompt): both sequences' logical blocks point at the same physical prompt blocks, refcount 2. When sample A1 wants to write to a shared block, vLLM sees refcount > 1, allocates a fresh block, copies the old one over, drops the refcount to 1. When A2 writes later, refcount is already 1, so it writes in place. Only the *last* block is ever copied.
- *Beam search* with width $k$: candidates share the prompt and all common history. When a beam candidate dies, its blocks' refcounts drop; blocks reaching 0 are freed. Old systems had to physically copy a candidate's whole KV cache when beams forked — vLLM copies one block at most.
- *Shared prefix* (a system prompt / few-shot examples): the provider pre-computes those blocks once and every request maps to them, marking the last block copy-on-write. The prefill only runs on the user's own text.

The API for all of this is three operations: `fork`, `append`, `free`.

**Scheduling and preemption.** FCFS, earliest arrival served first, latest preempted first. Eviction is **all-or-nothing per sequence** — you cannot run a sequence with half its KV cache, so evicting half a sequence is pointless. Beam candidates of one request are **gang-scheduled** as a sequence group, preempted together, since they share blocks. Two recovery options:

- **Swapping** — copy evicted blocks to CPU RAM. Bounded: swapped bytes never exceed the GPU KV region. After preempting, vLLM stops admitting new requests until the preempted ones finish.
- **Recomputation** — throw the blocks away and recompute later. Cheaper than it sounds: the already-generated tokens get concatenated onto the original prompt, so the whole thing recomputes in *one* parallel prefill pass, not token by token.

**Distributed.** Megatron-LM style [[Megatron-LM- Training Multi-Billion Parameter Models Using Model Parallelism#|tensor parallelism]], attention split across heads. Key observation: every shard processes the same token positions, so **one** KV manager in the central scheduler serves all workers. They share the same block table; each worker just stores its own heads' slice of each physical block. Workers all-reduce among themselves and never coordinate on memory.

**Implementation.** 8.5K lines Python (scheduler, block manager) + 2K lines C++/CUDA. Three custom kernels: (1) fused reshape-and-block-write for new KV; (2) fused block-read + attention, one GPU warp per block for coalesced access; (3) fused block copy that batches many copy-on-write copies into one launch instead of many `cudaMemcpyAsync` calls.

## Ablation Studies and Experiments

**Setup.** OPT-13B (1×A100-40GB), OPT-66B (4×A100), OPT-175B (8×A100-80GB), LLaMA-13B. Workloads synthesised from ShareGPT (real ChatGPT conversations, long — 8.4× longer inputs and 5.8× longer outputs than Alpaca) and Alpaca (short instructions). Arrivals Poisson at varying rates. Metric: **normalised latency** = mean end-to-end latency ÷ output length. You plot it against request rate and read off where the curve explodes.

**Baselines.** FasterTransformer (latency-optimised, coarse dynamic batching) and three versions of Orca, which does iteration-level scheduling but contiguous allocation:
- Orca (Oracle) — magically knows the true output length. Not achievable in practice; an upper bound.
- Orca (Pow2) — over-reserves by up to 2×.
- Orca (Max) — always reserves 2048.

**Basic sampling.** vLLM sustains **1.7×–2.7×** higher request rate than Orca (Oracle), **2.7×–8×** over Orca (Max), and up to **22×** over FasterTransformer. Mechanism confirmed directly: on OPT-13B/ShareGPT, vLLM has **2.2×** more requests in flight than Orca (Oracle), **4.3×** more than Orca (Max).

**Where the gain shrinks.** On OPT-175B + Alpaca, vLLM's edge over Orca (Oracle) and Orca (Pow2) mostly vanishes. 8×A100-80GB leaves 264 GB for KV cache (60.1K slots) and Alpaca sequences are short — so even a wasteful allocator batches plenty. The system becomes **compute-bound, not memory-bound**, and vLLM has nothing left to fix. This is the honest boundary of the paper.

**Sharing.** Beam search gains most, because sharing grows with beam width. On OPT-13B/Alpaca the advantage over Orca (Oracle) goes from **1.3×** (basic sampling) to **2.3×** (beam width 6). Measured memory saved by sharing:

| | Alpaca | ShareGPT |
|---|---|---|
| Parallel sampling | 6.1–9.8% | 16.2–30.5% |
| Beam search | 37.6–55.2% | 44.3–66.3% |

Longer prompts → more to share → bigger win.

**Shared prefix** (LLaMA-13B, WMT16 EN→DE): 1.67× over Orca (Oracle) with a one-shot prefix (80 tokens), **3.58×** with a five-shot prefix (341 tokens).

**Chatbot** (ShareGPT history concatenated, prompts truncated to 1024 tokens): 2× over all three Orca variants. All three behave the same here because buddy allocation rounds 1024-token prompts up identically regardless of output-length prediction.

**The cost, admitted plainly.** The PagedAttention kernel is **20–26% slower** than FasterTransformer's attention kernel, from block-table lookups, extra branches, and variable-length handling. This does *not* kill the win, because attention is one operator among many (Linear layers dominate FLOPs) and the batch-size gain swamps it. It is a real trade: slower kernel, far better memory.

**Block size ablation.** Too small → you cannot saturate GPU parallelism reading KV. Too large → internal fragmentation grows and sharing gets coarser. On ShareGPT, 16–128 all work. On Alpaca, 16 and 32 are fine and **larger sizes degrade badly**, because sequences are shorter than the block. Default is **16** — the value that is safe on both. This ablation is the reason the default exists.

**Swapping vs recomputation.** Swapping is terrible at small block sizes: many tiny CPU↔GPU transfers, so PCIe bandwidth goes unused. Recomputation cost is flat in block size (it never touches KV blocks). So recomputation wins for small blocks, swapping wins for large ones; at 16–64 they are comparable end-to-end. Recomputation overhead never exceeds 20% of swapping's latency.

## Worth Remembering

- **Nothing about model quality changes.** This is pure systems work. Same logits, same outputs, 2–4× more of them per second. That combination is rare and is why vLLM became the default open-source serving engine.
- **The complementarity with Orca is stated explicitly.** Orca's iteration-level scheduling (drop finished requests, add new ones after every token, no padding) attacks *scheduling*; PagedAttention attacks *memory*. vLLM does both. The authors note that finer-grained scheduling actually makes memory management *harder*, which makes paging more necessary, not less.
- **The limitation the authors volunteer:** paging is right for LLM serving specifically because (a) output length is unknown so allocation must be dynamic, and (b) the workload is memory-bound. Neither holds for DNN training (static shapes, plan ahead) or for serving small non-LLM models (compute-bound). Applying vLLM's indirection there would *hurt*, from memory-indirection overhead.
- **Two places vLLM deviates from a real OS**, and both are wins from knowing the application: all-or-nothing eviction (an OS cannot know which pages a process needs together; vLLM knows a sequence needs all of its blocks), and recomputation as a recovery path (an OS cannot regenerate a swapped-out page from scratch; vLLM can, cheaply, in one parallel prefill).
- **Complementary to [[FlashAttention- Fast and Memory-Efficient Exact Attention|FlashAttention]], not competing.** FlashAttention reduces HBM↔SRAM traffic *inside* one attention call; PagedAttention changes *where the KV cache lives across calls*. vLLM in fact uses a conventional FlashAttention-style kernel for the prefill phase and PagedAttention only for decode.
- **Practical caveat:** the swapping design halts admission of new requests once anything is preempted, until all preempted work drains. That is a simple policy with obvious tail-latency implications under bursty load.
- The 800 KB-per-token arithmetic is worth internalising. KV cache scales with $2 \times d_{\text{model}} \times n_{\text{layers}} \times \text{bytes}$ per token, which is why [[Fast Transformer Decoding- One Write-Head is All You Need (MQA)|MQA]] and [[GQA- Training Generalized Multi-Query Transformer Models|GQA]] — which shrink that constant by sharing key/value heads — are the natural next lever after paging.
- Open question the paper leaves: vLLM deliberately does *not* keep KV cache between chatbot turns ("doing this would occupy the space for other requests"). Prefix caching across turns is exactly what later versions added — the block table makes it nearly free.

## Links

Related: [[Attention Is All You Need]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Fast Transformer Decoding- One Write-Head is All You Need (MQA)]] · [[GQA- Training Generalized Multi-Query Transformer Models]] · [[Megatron-LM- Training Multi-Billion Parameter Models Using Model Parallelism]] · [[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models]] · [[Mixed Precision Training]] · [[Causal Attention]] · [[Query, Key, and Value (QKV)]] · [[Auto-regressive models]] · [[Seq2Seq models]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[Prefix Sliding for efficient test-time scaling]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]]

New topics worth writing: KV cache, virtual memory and paging, copy-on-write, beam search, continuous batching / iteration-level scheduling, prefix caching, memory fragmentation, CUDA kernel fusion, roofline and memory-bound vs compute-bound workloads, Orca serving system
