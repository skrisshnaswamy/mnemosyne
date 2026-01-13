---
title: "Prefix Sliding for efficient test-time scaling"
authors: ["Niklas Muennighoff", "Zhengyang Wang", "Zeyi Chen", "Weijia Shi", "Binyuan Hui", "John Yang", "Dapeng Jiang", "Mika Senghaas", "Fares Obeid", "Johannes Hagemann", "Sami Jaghouar", "Ludwig Schmidt", "Percy Liang", "Jason Wei", "Andrew Y. Ng", "Luke Zettlemoyer", "Yejin Choi", "Mike Lewis"]
year: 2026
arxiv: "2608.26070"
url: https://arxiv.org/abs/2608.26070
priority: Must-Read
read_on: 2026-08-28
tags: [paper, transformers, llm, rl, theory]
---
## The Core Idea

When a reasoning model thinks for a long time, every new token it writes must attend to every token it has already written. That is [[Causal Attention|causal attention]] with a full KV cache. The cost per new token grows without bound. Thinking for 200,000 tokens becomes absurdly slow and memory-hungry.

The observation here is blunt: **most of those old thinking tokens stop mattering.** If the model computes `42 + 84 = 126` on line 300, then by line 5,000 the scratch work is dead weight — only `126` mattered, and it has already been carried forward in the text. A heatmap of post-softmax attention probabilities in Qwen3-1.7B on an AIME trace (Figure 2) shows the shape clearly: huge mass on the first four tokens (the classic attention-sink effect), elevated mass on the rest of the prompt and on the `<think>` delimiter, then a long flat valley across the middle of the reasoning trace, then a sharp spike at the most recent tokens.

So: keep the two ends, throw away the middle. **Prefix Sliding** retains (a) the prefix — system instruction, tool definitions, the actual problem — and (b) a sliding window of the last $W$ tokens. Everything in between is evicted from the KV cache.

> [!NOTE] Prefix Sliding
> Attention is restricted to $\text{prefix} \cup \text{last } W \text{ tokens}$. Memory is capped at $|\text{prefix}| + W$ regardless of how long the model reasons, so cost per new token becomes **constant** in the limit. ^prefix-sliding

Why this is not just "sliding window attention", which has existed since Longformer (2020): a plain sliding window eventually slides past the prompt itself. The model forgets what problem it is solving and which tools it has. Keeping the prefix as a block of permanent global tokens is the whole difference, and on long traces it is the difference between working and not working.

The unlock: **bounded cost per token is a prerequisite for models that reason for hours or days.** Any method whose per-token cost keeps growing — full attention, sparse attention, Linformer, Ring Attention — cannot get there, no matter how good the constant factor. And unlike RNNs, SSMs, or Mamba, which are also bounded, Prefix Sliding needs no new architecture and no pretraining. You can bolt it onto Qwen3 today.

## The Methodology

**Inference, no training.** Model: Qwen3-1.7B, served with vLLM and [[FlashAttention- Fast and Memory-Efficient Exact Attention|FlashAttention]]. Window sizes swept: 512 → 16384.

Position embeddings need a decision, because the model uses [[RoFormer- Enhanced Transformer with Rotary Position Embedding|RoPE]] and tokens have been evicted. Two options:

- **Reset PE** — renumber the surviving tokens as if they were contiguous. Requires re-applying rotations to cached keys.
- **Continue PE** — leave the original absolute positions alone, so there is a numeric "hole" between the prefix and the window. Lets you reuse cached representations verbatim, so it is cheaper.

They use Continue PE. Appendix D finds the accuracy gap on AIME25 to be within standard error, so the cheap option is free.

**The kernel.** Naively you would compute full attention and mask. Instead they write a Hopper FlashAttention kernel with two-level filtering:

- *Intra-tile masking* — for tiles that straddle the boundary of the allowed region, apply an elementwise mask so only valid $(q,k)$ pairs enter the softmax. Keeps the tiling strategy of FlashAttention untouched, so the [[FlashAttention- Fast and Memory-Efficient Exact Attention#^io-aware|IO-awareness]] survives.
- *Inter-tile skipping* — restructure the producer–consumer pipeline to iterate over **two disjoint block ranges** (prefix blocks, then window blocks) and never load the tiles in between.

Result: essentially the same throughput as a vanilla sliding-window kernel, slightly slower only because the prefix is extra memory.

**Training with RL.** The point of training is that you no longer have to truncate and discard over-long rollouts (the usual DAPO-style hack). But backpropagating through a 100,000-token generation blows up trainer memory. Two fixes, both exploiting the fact that a sliding window has a **limited receptive field**:

Stacking $L$ layers of window $W$ has a theoretical receptive field of $W \times L$, but because of information bottlenecks it is empirically closer to $1.5 \times W$. So to get accurate gradients for a block of $W$ tokens, you only need to feed the trainer the prefix plus roughly $1.5W$ preceding tokens.

- **Chunked backprop** — walk the chain in chunks, accumulate gradients, near-equivalent to full backprop.
- **Truncated backprop** — only backprop the final chunk. This is what they use.

Concretely, with $W = 2048$ and a 100,000-token trace: send the last **8192** tokens to the trainer (a $4\times$ multiplier). Use the first 6144 purely as context and compute the token-level RL loss only on the final 2048. In code this is just a **loss mask** — zero the loss for the first 6144 tokens and let [[Pytorch Autograd|autograd]] do the rest. No custom backward pass.

Continue PE is used in the trainer too. Reset PE is a nightmare there because of teacher forcing: every token would have seen a different combination of preceding positions, so you cannot batch it.

**RL setup.** [[Proximal Policy Optimization Algorithms|GRPO]] via `trl` (synchronous) and `prime-rl` (asynchronous). Async is preferred: with rollouts of wildly different lengths, synchronous batching leaves GPUs idle waiting for the one 100K-token generation. Learning rate and other hyperparameters were **not** tuned, and fixed across comparisons.

**Data.** Math problems pooled from SkyWork and s1 (which itself draws on NuminaMATH, MATH, OlympiadBench, GPQA, and others), decontaminated against test sets. Filtered on three criteria:
- *Guessability* — drop anything a small model gets right in 8 tries **without thinking**.
- *Verifiability* — drop questions containing words like "How" or "Explain" that have no objective answer.
- *Difficulty* — drop what weak models always solve and what strong models never solve.

**Evaluation.** GPQA, MATH500, AIME25. avg@64 (64 samples averaged), temperature 0.6, top-p 0.95. Budget forcing to hit specific thinking budgets. The headline efficiency metric is **average thinking seconds per sample**, not FLOPs or token counts, because those miss memory differences between methods.

## Ablations Studies and Experiments

**The main table (Table 1, Qwen3-1.7B, no training):**

| Window | AIME25 avg@64 | avg len | GPQA | MATH500 | tok/s @32K | tok/s @128K |
|---|---|---|---|---|---|---|
| 2048 | 27.7 | 47643 | 35.9 | 89.8 | 8973 | 8737 |
| 4096 | 33.9 | 29943 | 37.0 | 91.5 | 5479 | 5224 |
| 8192 | **35.8** | 19373 | 38.0 | 91.4 | 3291 | 2788 |
| 16384 | 35.3 | 19872 | **38.2** | 91.5 | 2441 | 1420 |
| Full attention | 34.2 | 19158 | 37.6 | **91.7** | 1477 | 448 |

Read the last column carefully. At a 128K sequence length, full attention runs at **448 tok/s**; Prefix Sliding with a 4096 window runs at **5224 tok/s** — nearly 12×. And window 8192 *beats* full attention on AIME25 (35.8 vs 34.2) while running 6× faster.

**The crucial honest framing:** the paper states outright that Prefix Sliding wins *not because each token is better*, but because it can generate **more tokens in the same wall-clock thinking time**. Look at avg len for window 2048 on AIME25: 47,643 tokens versus full attention's 19,158. It is a worse thinker per token; it just gets far more thinking done per second. The quality-per-token drop at small windows is real — window 2048 scores 27.7 versus full attention's 34.2.

**Throughput curve (Figure 6).** Tok/s for Prefix Sliding and vanilla sliding window *drop* at first, then flatten around 5,000. The initial drop is the **sliding-window warm-up phase** — while generated tokens are fewer than $W$, you are still doing full attention. Once the window fills, the cost per token is flat forever. Full attention just keeps degrading.

**Truncated backprop numerics (Figure 8).** They measure [[KL Divergence|KL divergence]] between the generator's and the trainer's per-token log-probs on the last 2048 tokens (the ones that get gradient). Window = 2048, max sequence 16384, varying how many trailing tokens go to the trainer:

- 2K (just the window): KL > 0.1 — bad, as predicted by receptive-field reasoning.
- 4K ($2\times$): drops sharply.
- 8K ($4\times$): slightly better again. **This is what they ship.**
- 16K ($8\times$, i.e. everything): about the same as 8K.

Residual KL never hits zero, because the generator uses their custom FlashAttention kernel and the trainer uses FlexAttention — tiny numerical differences. The $4\times$ multiplier also holds for larger windows: Appendix E trains DeepSeek-R1-Distill-Qwen-7B with $W = 8192$, $4\times$ multiplier, and matches full-attention full-backprop performance.

**Head-to-head against alternatives (Figure 9, AIME25, max gen 262144, local window 4096 for everything but full attention):**

- **Last $k$** — generate until context is full, delete everything except the last $k$ tokens, restart. This is "Markovian Thinking / Delethink" and is what many agent frameworks do with turn history. Sweep in Table 2 (2048 context, one pass, avg@64): $k$=64 → MATH500 58.2 / AIME25 3.2; $k$=128 → 59.7/3.5; **$k$=256 → 60.8/4.2**; $k$=512 → 60.3/4.2; $k$=1024 → 54.6/2.4. Non-monotonic — large $k$ hurts because you reprocess too much, small $k$ hurts because you cut useful recent reasoning mid-sentence. Chosen: $k=256$. Failure mode caught in Figure 21: the carried-over 256 tokens are cut mid-token, the model reads garbage, and **regenerates the entire derivation from scratch**.
- **Summary (compaction)** — summarise the context and restart with prompt + summary. This is what Opus 4.6, GPT-5.4 and Cursor Composer do. Implemented as a tool call, model summarises itself, max summary 256 tokens, force-inserted if the model has not called it by the time the window is nearly full. Prompt ablation (Table 3, AIME25 accuracy / coverage@64): tool-only prompt → 23.2 / 33.3; **adding an example of using the tool and consuming context → 26.4 / 53.3**; further explaining the context → 25.8 / 46.7. So the example helps a lot, extra explanation does not. Failure mode in Figure 20: the model receives a perfectly good summary containing $r = 12+6\sqrt{3}$ and $s = 12-6\sqrt{3}$ — literally the answer — and **restarts the whole problem from first principles anyway.**
- **Vanilla sliding window** — flattens out early. Once the window slides past the prompt, the model no longer knows what it is solving.

**What did not work:**

- Inserting the summary *inside* the model's thinking (rather than as a separate `<context>` block) made things worse — likely far out of distribution for the model.
- The "explain the context more" summary prompt (prompt 3) was worse than the simpler example-based one.
- $k = 1024$ for last-$k$ was clearly worse than $k = 256$.
- Backpropagating only the window itself ($1\times$) gives KL > 0.1 — unusable gradients.
- $H_2O$ (heavy-hitter KV eviction) could not be fairly compared because it does not integrate with FlashAttention/vLLM. The authors note it is *backward*-looking (keep what got attention so far) while Prefix Sliding is *forward*-looking (keep what we know will be needed later, like a tool definition that sits unused for 10,000 tokens). They speculate the combination would beat either.

Also note: the summary baseline was given a slight advantage — its summary length was not subtracted from the second-turn token budget, so it holds marginally more in memory than Prefix Sliding or last-$k$. It still lost.

## Worth Remembering

**Where it breaks — LiveCodeBench.** Needs a window of at least 16384 to match full attention (Figure 11). Inspecting samples reveals why: the model starts writing a function, then thinks for thousands of tokens *inside code comments*, and by the time it resumes coding, the beginning of its own function has slid out of the window. This is a genuine information-loss failure, not an artifact. The authors' claim — untested — is that RL training with Prefix Sliding would teach the model to stop commenting so much. Their other suggestion is a mechanism to *append* tokens from the window into the prefix, which would be a much more interesting method.

**Where it is pointless — short tasks.** HealthBench averages 2086 tokens per sample. With $W = 2048$ the window almost never slides, so Prefix Sliding ≈ full attention and there is no speedup (Figure 12). All the benefit lives in the tail.

**Prefill is not helped.** Prefix Sliding does nothing about the cost of the prompt itself. A gigantic prefix (a whole book pasted in) is still gigantic. The suggested fix is to put long inputs in a *file* and give the model the path, so it reads incrementally.

**The agentic problem is unsolved.** If the model `cat`s a large file or reads a web page, the tool output can flood the entire window and evict the reasoning. Worse, if the content is larger than $W$, the model *strictly cannot* read all of it. Same issue for multi-turn: do new user messages go into the prefix, or do they eventually slide away? No answer given. Suggested mitigations: train the model to read in chunks (`head` not `cat`), or add guardrails truncating oversized tool outputs.

**The taxonomy is the durable idea.** *Bounded* methods have asymptotically constant cost per new token (RNNs like RWKV, SSMs like Mamba, Transformer-XL, Infini-attention, Prefix Sliding). *Unbounded* methods cost more per token forever, even if sub-quadratic (Sparse Transformers, Reformer, Linformer, Ring Attention, [[Attention Is All You Need|vanilla attention]]). You can trade space for time to make one bounded, but not both. If you want a model that thinks for a week, you need a bounded method — everything else is a constant-factor delay of the same wall.

Figure 10 adds a nice detail: last-$k$ and summary *are* bounded, but with a **sawtooth** cost profile — a spike at every chunk boundary when tokens are reprocessed or a summary is generated. That makes GPU utilisation hard to plan. Prefix Sliding is flat.

**Practical caveats if you want to use it:**
- Pick $W$ per task family. 4096 is a reasonable default for math; code needs 16384+.
- Continue PE is fine, use it, it is cheaper.
- If you RL-train: truncated backprop with a $4\times$ window multiplier, implemented as a loss mask. Use async RL or you will burn GPU time waiting on long rollouts.
- Expect a *worse* model per token and a much *better* model per second. If your budget is measured in tokens, this loses. If it is measured in seconds or dollars, it wins.

**Open question the paper leaves:** they only tested up to 7B parameters. Larger models may have different attention-decay profiles — an attention sink is a small-model-friendly artifact, and it is not obvious the "valley" in the middle of the trace stays as empty at 70B.

## Links

Related: [[Attention Is All You Need]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Flash Attention]] · [[Causal Attention]] · [[Query, Key, and Value (QKV)]] · [[RoFormer- Enhanced Transformer with Rotary Position Embedding]] · [[Proximal Policy Optimization Algorithms]] · [[KL Divergence]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Training language models to follow instructions with human feedback]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[Scaling Laws for Neural Language Models]] · [[Long Short-Term Memory (Neural Computation)]] · [[Pytorch Autograd]] · [[Backpropagation]] · [[Chain-of-Experience for Continual LLM Improvement]]

New topics worth writing: Attention sinks and StreamingLLM, KV cache eviction (H2O, SnapKV, PyramidKV), Test-time scaling taxonomy (sequential vs parallel), GRPO, Mamba and state-space models, RWKV, Transformer-XL and recurrent memory, Context compaction in agent frameworks, Asynchronous RL for LLMs, Longformer / BigBird global tokens, Receptive field of stacked sliding-window attention, Budget forcing
