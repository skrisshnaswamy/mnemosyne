---
title: "The Other Half of the Memory Wall: Serving 35B MoEs from SSD with Trained Routing Prediction"
authors: ["Lin et al."]
year: 2026
arxiv: "2609.18063"
url: https://arxiv.org/abs/2609.18063
priority: Good-To-Read
read_on: 2026-09-26
tags: [paper, transformers, theory, scaling]
---
## The Core Idea

A mixture-of-experts model is sparse in *compute*, not in *storage*. A 35B MoE with 256 experts per layer touches only ~3B parameters for each token it generates — but all 19.5 GB of 4-bit weights still have to be *somewhere*. On a 24 GB desktop that "somewhere" is the whole machine.

Edge0's answer: keep the expert weights on the SSD and read them in as needed. Peak memory then tracks the *active* expert set, not the parameter count. The 35B model runs in 2.9 GiB of allocator memory instead of 18.2 GiB.

The obvious problem is that naive on-demand reading is slow, and the reason is a dependency, not bandwidth. To know which experts layer $N{+}1$ needs, you need layer $N$'s output. So the read cannot begin until the moment it is needed, and disk latency gets serialised into every layer of every token.

> [!NOTE] Prerouter
> A small trained head, owned by layer $N$, that looks at layer $N$'s hidden state at token $t$ and predicts **layer $N{+}1$'s expert choice at token $t{+}1$**. Because it fires a whole token early, the SSD read for those experts overlaps the current forward pass. ^prerouter

The trick that makes this clean is refusing to treat the prediction as a guess. At decode time the prerouter's logits **replace** the real router's logits. The experts that were staged are, by definition, exactly the experts that get used. Nothing is dropped, no fallback load is needed, no coverage-vs-quality knob exists at runtime.

This is the contrast with Pre-gated MoE (Hwang et al., ISCA 2024), which picks layer $N{+}1$'s experts *within the same token*, right after layer $N$'s attention. Edge0 measured that schedule and it loses: the per-layer sync plus head evaluation costs 30–100 ms per step, more than the load it hides, and every same-token variant they tried came in *below* a plain LRU cache. One token of lead moves head evaluation off the per-layer critical path — a single flush predicts all 32 staged layers at once.

The approximation is not free; it is just **paid in training instead of at inference**. A LoRA adapter is distilled from the fp16 teacher while the student runs with prerouter routing active, so the adapter compensates for int4 quantisation *and* the swapped-out router together.

What it unlocks: 20.4 tok/s for a 35B-class MoE on a Mac mini M4 Pro, inside 3 GiB, within ~3.9 points of the fp16 base on five benchmarks. The fully-resident baseline on the same box gets 3.9 tok/s.

## The Methodology

Three parts: a streaming executor, the prerouter, and an unmerged recovery adapter.

### Streaming the experts

The 35B tier is 40 layers × 256 experts. At int4 (affine, group-64), one expert is 1.77 MB, one layer's experts are 453 MB, and the routed experts total 18 GB of a 19.5 GB checkpoint.

Expert weights live on disk as per-layer stacked safetensors, `mmap`-ed. Byte ranges are read on demand; the OS page cache holds whatever is hot.

Four executor paths, all computing the same thing:

| Path | What it does | Used for |
|---|---|---|
| `exact` | dedup, build bundle, stack, quantised gather | correctness reference |
| `staged` | fixed-slot double buffer; routing indices mapped through a slot table with `take`, never leaving the GPU | decode |
| `hot` | LRU-resident hot experts per layer; misses fall through to `exact` | the PowerInfer-style split |
| `whole-layer` | all 256 experts in 9 direct reads (gate/up/down × weight/scale/bias) | prefill |

Every path computes

$$\mathrm{down}\big(\mathrm{silu}(\mathrm{gate}(x)) \cdot \mathrm{up}(x)\big)$$

with the same quantised gather kernel, and each is checked element-wise against the dequantised reference (relative L2 under 1%; measured residual ≈0.24%, which is just the kernel's bf16 internals). That contract is what lets the engine switch paths per layer and per phase without changing outputs.

One systems detail worth keeping: `incr_stack`. Rebuilding the nine stacked tensors per layer per step is expensive, so the staged path holds persistent "sticky slot" tensors and rewrites only the experts that actually changed. Worth **+34% decode** on its own in a separate A/B.

### The prerouter head

Per layer, tiny:

$$\text{fc}_1 \to \text{erf-gelu} \to \text{fc}_2 \;+\; \ell(x)$$

where $\ell$ is a linear residual path **warm-started from the next layer's actual router weight**. So training starts from "apply layer $N{+}1$'s router directly to layer $N$'s hidden state" and the MLP only has to learn the correction. (Its default init is zero.)

Input features — hidden state concatenated with two top-$k$ one-hot vectors: the experts *this* layer routed to at this token, and at the previous token.

- 35B: $2048 + 2{\times}256 = 2560$ input dim, hidden width 512, 33 heads (owned by layers 6–38), predictions consumed at layers 7–38, so 32 of 40 layers stream from prediction.
- 8B: $1536 + 2{\times}128$, 16 heads (owners 7–22), consumed at 8–23.

Heads are fp16 — fp32 costs measurable time for no accuracy gain.

The predicted logits then go through *the same* routing function as the original model. Both families are implemented verbatim:

Softmax-top-k (Qwen3.6):
$$g = \mathrm{softmax}(\ell), \quad \mathrm{inds} = \text{top-}k(g, K), \quad w = g[\mathrm{inds}] \Big/ \sum_{\mathrm{inds}} g$$

Sigmoid-group (DeepSeek-V3 / Ling style): $\sigma = \mathrm{sigmoid}(\ell)$; each group scored by the sum of its top two $\sigma$; keep the best $G$ groups, mask the rest to $-\infty$; top-$k$ among survivors; weights are the *raw* sigmoid values, renormalised and scaled by $s$. The 8B tier uses $n{=}8$ groups, $G{=}4$, $s{=}2.5$, $K{=}8$. Selection uses the biased score, weights use the raw sigmoid — the DeepSeek distinction.

> [!NOTE] Feature drift
> At training, a head's one-hot input records what the *base router* selected. At decode it records what the *head itself* predicted, because prediction replaced routing. The heads are never retrained for that shift. The recovery LoRA absorbs it, because the LoRA is trained on the deployed path. ^feature-drift

### Recovery LoRA, served unmerged

$r{=}16$, $\alpha{=}32$, attached to attention, linear-attention and shared-expert projections — **not** the routed experts, whose weights must stay streamable and swappable. 42 MB of adapter.

At serve time:
$$y = W_{\text{int4}}(x) + \tfrac{\alpha}{r} BAx$$

computed as a parallel delta at a sum node, with the 4-bit base bytes untouched.

### Training: three phases, base frozen throughout

All phases run on the **dequantised bf16 reconstruction of the 4-bit deployment checkpoint** — train on what you serve.

1. **Distil the heads.** Only the prerouter heads are trainable. Loss imitates the next layer's true router. The heads learn routing, not text.
2. **SFT on the student path.** Full forward with prerouter routing active, cross-entropy against ~2M rows of teacher-generated text. LoRA is the trainable surface.
3. **On-policy distillation.** The Phase-2 checkpoint generates; the original fp16 base scores those tokens as teacher. Objective is **reverse KL** (mode-seeking, so the student is not forced to cover the teacher's whole support) on the teacher's top-$k$ tokens, with the mass outside top-$k$ carried by a tail term rather than discarded. Converges on ~200k rows, a tenth of Phase 2.

The order was not optional in their runs: heads first, because the SFT signal otherwise drowns the tiny head gradients.

## Ablation Studies and Experiments

### Quality (OpenCompass, identical settings, no timing)

| Benchmark | edge0-35b (int4) | Qwen3.6-35B-A3B (fp16) | edge0-8b (int4) | Ling 3.0 tiny (fp16) |
|---|---|---|---|---|
| AIME 2026 | 86.6 | 92.7 | 63.3 | 73.3 |
| HumanEval | 90.9 | 95.1 | 91.5 | 92.7 |
| GPQA-Diamond | 79.8 | 81.8 | 70.7 | 71.2 |
| MMLU-Pro | 81.0 | 84.6 | **70.1** | 65.8 |
| IFBench | 57.9 | 61.7 | 53.9 | 60.6 |
| **Average** | **79.2** | 83.2 | **69.9** | 72.7 |

Mean per-benchmark gap: 3.9 points (35B), 2.8 points (8B). The 8B tier *beats* its teacher on MMLU-Pro by 4.3 — a hint that the distillation is doing real work, not just damage control.

### Serving profiles (Mac mini M4 Pro, 24 GB)

| | edge0-35b | edge0-8b |
|---|---|---|
| Layers × experts | 40 × 256 + shared | 24 × 128 + shared |
| Active params/token | ≈3B | ≈1.2B |
| Routing width $K$ | 4 | 8 |
| Checkpoint on disk | 19.5 GB | 4.5 GB |
| Decode | 20.4 tok/s | 28.0 tok/s |
| Prefill cold/warm | 113 / 140 tok/s | 500 / 1102 tok/s |
| Peak active memory | 2.9 GiB | 1.5 GiB |

Baseline: vanilla mlx-lm with all 19.5 GB resident decodes at **3.9 tok/s occupying 18.2 GiB**. Streaming from SSD is 5× faster *and* uses a sixth of the memory — because the resident path leaves nothing of the 24 GB for KV cache and the OS, so the system pages.

### The prerouter A/B (MacBook M2, 16 GB, 18.4 GiB checkpoint — weights do not fit)

Same session, rotated arm order, both arms replaying the same sampled token sequence, 3-round medians.

| $K$ | on-demand | prerouter | gain |
|---|---|---|---|
| 2 | 4.8 | 8.6 | +80% |
| 4 | 3.5 | 6.4 | +82% |
| 8 | 1.8 | 3.3 | +84% |

Main-thread time blocked on expert loads: 154.9 → 46.5 ms/step at $K{=}2$; 244.0 → 101.9 at $K{=}4$; 575.0 → 211.6 at $K{=}8$.

**Nothing is saturated while this happens.** Disk read is at most 12% of the step, the process uses about one core of eight, GPU sits at 35–41%. The thing being removed is serialised load *latency*, full stop.

### What the ablations actually reveal

**It is not prediction quality — it is conservation.** Both arms move within a few percent of the same bytes: 125.1 vs 126.9 MiB/step at $K{=}8$, 29.5 vs 30.4 at $K{=}2$. Only at $K{=}4$ does the prerouter read 16% more (58.9 vs 50.9). Moving a read earlier can hide it; it cannot delete it. The win is entirely the cold-read time currently exposed on the critical path — which means **the size of the win is a property of the storage tier, not of the head.** Fast NVMe or a hot cache shrinks it.

**Granularity is the visible mechanism.** Per-load cost fits

$$1.17\,\text{ms} + 1.33\,\text{ms} \times (\text{cold fraction})$$

reproducing every measured per-load cost within 0.13 ms. On-demand: 99–399 loads/step at ~0.3 MiB and 20% cold. Prerouter: 19–90 loads/step at ~1.4 MiB and 84–98% cold. Fewer, larger, colder loads — cheap per load, expensive per byte moved early.

**The price is unreclaimable residency.** Prefetched experts must be resident, and MLX allocations are not reclaimable. Unreclaimable allocation rises from 1.72/1.73/1.79 GiB (on-demand, $K{=}2,4,8$) to 2.33/2.60/3.22 GiB. At $K{=}8$ the prerouter holds 1.43 GiB more, and the page cache loses 1.15 GiB — four fifths of it. That trade only pays where memory is tight *and* storage is slow.

**Reuse bounds everything.** Adjacent tokens agree on only about **a quarter** of a layer's expert set. Most prefetched experts are never read again: issued, paid for, unused. Storage speed sets how much is available to win; reuse sets how much the prediction actually collects.

**Width is the one knob that moves speed and memory together.** Same weights, same session, one cache budget: narrowing $K{=}8 \to K{=}4$ takes decode from 3.3 to 6.4 tok/s and lowers peak memory. Because the base is frozen, retraining for a new width is a two-file swap, which is why the 35B ships at $K{=}4$ despite the base model's $K{=}8$.

### What did not work

- **Same-token pre-gating** (the Pre-gated MoE schedule). Per-layer sync plus head evaluation costs 30–100 ms/step — more than the load it hides. *Every* same-token variant measured fell below a plain LRU baseline. The lead has to be a full token.
- **Merging the LoRA into the int4 base.** LoRA deltas have RMS $\sim 10^{-3}$, below the 4-bit group step. Requantising erases most of the effect: **34% survives on an attention projection, 2% on a dense projection, 18% at the logits level.** Not an implementation bug — arithmetic.
- **Distillation-only (heads, no SFT).** With student routing active, the model produces repetitive, collapsed text. Heads plus SFT produce coherent output at identical speed and memory. Phase 2 is what makes the approximation usable.
- **Wrong phase order.** SFT before head distillation drowns the tiny head gradients.
- **fp32 heads.** Measurable time cost, no accuracy benefit.
- **Chasing router agreement harder.** Cross-token prediction from the previous token's hidden state is information-limited. The objective is good text under student routing, not high top-$k$ overlap with the base router.
- **Staging without a prediction source.** It either drops experts (output degrades) or pins a hot set that churns. "Prediction chooses, staging loads" — one mechanism, not two.

## Worth Remembering

**The framing is the contribution.** The field spent years compressing the *dynamic* half of memory — MLA, sparse attention, linear-state models, paged KV. The *static* half, tens of gigabytes of weights, got one answer: put it in a datacenter. Edge0's claim is that weight placement is a *storage tier* decision that nobody local had actually made.

**Prior offloading work moved the footprint; it did not shrink it.** llama.cpp's `--cpu-moe`, PowerInfer's hot/cold neuron split, Mixtral-offloading and MoE-Infinity's popularity caches, FlexGen's disk hierarchy — the weights still occupy tens of gigabytes, and the ones that reach disk do so *without knowing which experts the next step needs*. That last clause is the whole paper.

**Limitations the authors state plainly:**

- **One request at a time, FIFO.** Batching changes the expert working set in ways none of these per-request profiles model. This is a single-user desktop engine, not a serving system.
- **The remaining bottleneck is CPU, not storage.** 44 ms/step of *graph building* across 40 layers is the floor. No storage-side optimisation touches it. Fixing it needs kernel-level graph amortisation or a smaller model.
- **Two levers left unpulled.** The stager holds both a bundle and a stacked tensor view of the same weights — dropping one returns ~0.45 GiB at $K{=}8$. And the incremental stack fill runs synchronously on the main thread, costing 62.5 ms/step at $K{=}8$, for no reason anyone has to wait for.
- **Quality loss concentrates in long-chain reasoning.** AIME drops 6.1 points (35B) and 10.0 (8B). Everything else on the 8B is within 6.7. Int4 plus routing replacement is most visible where errors compound over many steps.
- **One backend (MLX).** The CUDA slot is "architecture, not code."

**Measurement hygiene worth copying.** Single-shot benchmarks on this hardware carry **±40% run-to-run spread**, and up to **2.3× across sessions** on the 16 GB box. Their response: every arm in its own process, arm order rotated each round, both arms replaying the same sampled token sequence, matched cache budgets, page-cache warmup, medians over repeats, and **no cross-session number used as evidence anywhere**. If you have ever tried to A/B an inference change on a laptop, this is the protocol.

**Open questions I would want answered:**

- The heads are trained against base-router labels but deployed on their own predictions — classic exposure bias, the same shape as teacher forcing in [[Auto-regressive models]]. They explicitly decline to retrain for the shift and let the LoRA absorb it. Would iterating head training on the student's own routing distribution (a DAgger-style loop) buy anything, or is the information limit binding first?
- The ~25% adjacent-token expert overlap is the ceiling on prefetch value. Does that figure hold across domains? Reasoning traces might route more stably than chat.
- $K{=}4$ at serve time when the base trained at $K{=}8$: how much of the AIME loss is quantisation, how much is routing replacement, and how much is just halving the routing width? The paper never separates the three.

**Practical caveat.** The +80–84% figure comes from a machine where the checkpoint *does not fit*, so every step faults from SSD. On a box with headroom, or with fast NVMe, the prerouter gain shrinks and its residency cost stays. The mechanism is regime-specific by construction, and the authors say so.

## Links

Related: [[Mixture of Experts]] · [[Quantization]] · [[LoRA]] · [[QLoRA- Efficient Finetuning of Quantized LLMs]] · [[Distillation]] · [[On-policy Distillation with Verifiable Reward]] · [[KL Divergence]] · [[Prefill and Decode]] · [[KV Cache]] · [[Multi-head Latent Attention]] · [[Sparse Attention]] · [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]] · [[Sparsely-Gated Mixture-of-Experts Layer]] · [[Cost and Latency]] · [[Speculative Decoding]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Distilling the Knowledge in a Neural Network]] · [[Continuous Batching]]

New topics worth writing: Memory wall and arithmetic intensity (roofline/ridge point), Expert offloading and hot/cold caching (PowerInfer, MoE-Infinity, FlexGen), Pre-gated MoE, GPTQ and AWQ post-training quantization, mmap and OS page cache as an ML memory tier, Reverse-KL on-policy distillation with tail correction, Benchmarking on consumer hardware (rotated same-session A/B protocol), MLX
