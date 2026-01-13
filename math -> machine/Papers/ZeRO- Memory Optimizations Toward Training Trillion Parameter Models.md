---
title: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
authors: ["Samyam Rajbhandari", "Jeff Rasley", "Olatunji Ruwase", "Yuxiong He"]
year: 2019
arxiv: "1910.02054"
url: https://arxiv.org/abs/1910.02054
priority: Must-Read
read_on: 2026-08-25
tags: [paper, llm, optimization]
---
## The Core Idea

Training a big model on many GPUs used to force an ugly choice.

**Data parallelism (DP)** is fast and easy: give every GPU a full copy of the model, feed each a different slice of the batch, average the [[Derivative#Gradient|gradients]]. But every GPU holds the *same* copy of everything. With 64 GPUs you are storing 64 identical copies of the optimiser state. Memory per GPU does not shrink at all, so adding GPUs never lets you train a bigger model.

**Model parallelism (MP)** cuts each layer across GPUs, so memory does shrink. But now GPUs must talk to each other *inside* every layer. That is fine inside one machine with fast NVLink; across machines it collapses. The authors measured a 40B model with Megatron-LM across two DGX-2 nodes at **~5 TFlops per V100** — under 5% of peak.

ZeRO's insight is embarrassingly simple once said out loud: **the redundancy in data parallelism is pure waste, and you can delete it without changing the computation at all.** Instead of every GPU storing all the optimiser states, gradients and parameters, each GPU stores only $1/N_d$ of them. When a GPU needs a piece it does not own, it fetches it just in time with a collective communication op, uses it, and throws it away.

The reason this is nearly free is the second insight: **not all state is needed all the time.** Layer 7's parameters only matter during layer 7's forward and backward pass. So you can stream them in and out. And the total bytes moved barely changes, because a standard DP all-reduce is *already* implemented as a reduce-scatter followed by an all-gather — ZeRO just splits those two halves apart and puts an optimiser step in between.

What it unlocks: memory per GPU drops **linearly with the number of GPUs**. A trillion-parameter model needs ~16 TB of model state; on 1024 GPUs that is 16 GB each, which fits on a 32 GB V100. Model size is no longer capped by one device's memory. And it needs **zero model refactoring** — you wrap a `torch.nn.Module` and go.

> [!NOTE] Zero Redundancy Optimizer ^zero
> Data-parallel training where optimiser states, gradients and parameters are *partitioned* across the $N_d$ workers rather than replicated. Missing shards are fetched on demand via all-gather/broadcast and discarded after use. Mathematically identical to standard DP — same gradients, same updates, same convergence.

## The Methodology

### Where the memory actually goes

Take a model with $\Psi$ parameters, trained with Adam in [[Mixed Precision training|mixed precision]]. Per GPU you store:

| Tensor | Precision | Bytes |
|---|---|---|
| parameters | fp16 | $2\Psi$ |
| gradients | fp16 | $2\Psi$ |
| fp32 master parameters | fp32 | $4\Psi$ |
| Adam momentum | fp32 | $4\Psi$ |
| Adam variance | fp32 | $4\Psi$ |

Total $= 2\Psi + 2\Psi + K\Psi = 16\Psi$ bytes, where $K = 12$ is the "optimiser multiplier". GPT-2 at 1.5B params therefore needs **24 GB of model state** even though the fp16 weights are only 3 GB. That is why a 32 GB GPU chokes on it.

Everything else — activations, temp buffers, fragmentation — the paper calls **residual state**.

### ZeRO-DP: three cumulative stages

**Stage 1 — $P_{os}$, partition optimiser states.** Split the fp32 master weights, momentum and variance into $N_d$ chunks. GPU $i$ owns chunk $i$ and only updates chunk $i$ of the parameters. At the end of the step, an all-gather collects the updated fp16 parameters back to everyone.

$$\text{mem} = 4\Psi + \frac{K\Psi}{N_d} \approx 4\Psi \quad \text{(4x reduction)}$$

Communication: still $2\Psi$. Identical to baseline DP.

**Stage 2 — $P_{os+g}$, also partition gradients.** GPU $i$ only ever updates parameter chunk $i$, so it only needs the *averaged* gradient for chunk $i$. As each layer's gradients appear during [[Backpropagation|backprop]], reduce them onto their owner GPU and free the rest. That is a **reduce-scatter**, not an all-reduce. Gradients are bucketed so the reduction happens on large contiguous blocks (overlapping communication with compute, same trick as NVIDIA AMP).

$$\text{mem} = 2\Psi + \frac{14\Psi}{N_d} \approx 2\Psi \quad \text{(8x reduction)}$$

Communication: reduce-scatter ($\Psi$) + all-gather ($\Psi$) $= 2\Psi$. **Still identical to baseline DP.** This is the free lunch.

**Stage 3 — $P_{os+g+p}$, also partition parameters.** Each GPU stores only its $\Psi/N_d$ slice of the fp16 weights. During the forward pass, before computing partition $j$, the GPU owning $j$ broadcasts those weights to everyone; after the layer finishes, they are discarded. Repeat in reverse for backward.

$$\text{mem} = \frac{16\Psi}{N_d}$$

Communication: $\Psi$ (forward all-gather) $+ \Psi$ (backward all-gather) $+ \Psi$ (reduce-scatter) $= 3\Psi$, i.e. **1.5x baseline DP** for an $N_d$-fold memory cut.

Concrete numbers from Table 1, a 7.5B model at $N_d = 64$: 120 GB (plain DP) → 31.4 GB ($P_{os}$) → 16.6 GB ($P_{os+g}$) → **1.88 GB** ($P_{os+g+p}$).

### ZeRO-R: the residual state

**$P_a$ — partitioned activation checkpointing.** Model parallelism partitions weights but *replicates activations*: if you split a linear layer vertically across two GPUs, both need the whole input. ZeRO-R stores activation checkpoints partitioned across the MP group and all-gathers them only right before recomputation. For a 100B model, batch 32, seq 1024, MP=16: activation checkpoints go from **33 GB → ~2 GB per GPU**. Extra communication is one all-gather of $\text{seq\_len} \times \text{hidden}$ per transformer block against Megatron's existing $12 \times \text{seq\_len} \times \text{hidden}$ — **under 10% overhead**.

**$P_{a+cpu}$** pushes those partitions to CPU memory, taking activation memory to ~zero. Viable only because large transformers have arithmetic intensity $\geq 10\text{K}$ (compute per iteration ÷ checkpoint bytes), so the PCIe transfer hides behind compute.

**$C_B$ — constant-size buffers.** Libraries fuse all gradients into one flat buffer before an all-reduce, because bandwidth improves with message size. But that buffer scales with the model: a 3B model needs a 12 GB fp32 fused buffer. ZeRO caps the buffer at a fixed size — big enough for good bandwidth, small enough not to matter.

**$M_D$ — memory defragmentation.** During forward, checkpointed activations are long-lived and recomputed ones short-lived; during backward, parameter gradients are long-lived and activation gradients short-lived. Interleaving lifetimes shreds the heap. The authors saw OOM with **over 30% of memory still free**. Fix: pre-allocate contiguous chunks for checkpoints and gradients and copy into them as they are produced. Also speeds up the allocator's search.

### What they actually shipped

**ZeRO-100B** = $P_{os+g}$ + all of ZeRO-R. Stage 3 was left for future work. PyTorch, 400 V100s (25 DGX-2 nodes), 800 Gbps internode. Models are GPT-2-style transformers, hidden size up to 8192, up to 212 layers.

## Ablation Studies and Experiments

**Baselines:** PyTorch DDP for no-MP runs; Megatron-LM (NVIDIA open source, Sept 2019) for MP runs. ZeRO runs combine its DP with Megatron's MP.

**Throughput vs model size (Figure 2).** ZeRO-100B holds **>38 TFlops/GPU, ~15 PetaFlops aggregate** (over 30% of peak) across 8B–100B params. Megatron alone degrades hard past 40B, because MP must cross node boundaries and bandwidth drops from 300 GB/s (NVSwitch) to 12.5 GB/s (Infiniband EDR). Max speedup **10x**. Max efficient model size: **170B vs ~40B** — over 8x.

Note the baseline was given an *advantage*: it ran on 256 or 384 GPUs (powers of 2 needed for MP) versus ZeRO's 400, so it had less DP communication. For the 170B baseline, DP degree was 1 — literally zero DP communication. ZeRO still won on per-GPU throughput.

**Super-linear scaling (Figure 3).** A 60B model from 64 → 400 GPUs: throughput per GPU *rises*, so doubling GPUs more than doubles total throughput. Reason: $P_{os+g}$ shrinks per-GPU memory as $N_d$ grows, so you can fit a bigger per-GPU batch, which raises arithmetic intensity. Per-GPU batch went 16 (64 GPUs) → 64 (400 GPUs).

**Usability (Figure 5).** With **no MP at all** on 128 GPUs, ZeRO trains up to **13B parameters at >40 TFlops/GPU**. Plain PyTorch DDP tops out at **1.4B at under 20 TFlops/GPU**. 13B is larger than T5-11B, the biggest published model at the time.

**The configuration ablation (Table 3, C1–C5), MP=16, fixed batch:**

| Config | ZeRO-DP | ZeRO-R | Max model |
|---|---|---|---|
| C1 | $P_{os}$ | $C_B + M_D$ | 40B |
| C2 | $P_{os}$ | $+ P_a$ | 60B |
| C3 | $P_{os+g}$ | $C_B + M_D$ | — |
| C4 | $P_{os+g}$ | $+ P_a$ | 140B |
| C5 | $P_{os+g}$ | $+ P_{a+cpu}$ | 150B |

What this reveals: the 40B → 60B jump is **activation partitioning** (16x cut in activation memory from MP degree 16). The 60B → 140B jump is **gradient partitioning** halving model-state memory versus $P_{os}$. The 140B → 150B jump is CPU offload of activations. So at MP=16 with big batches, activations and gradients are both first-order — you need both, and neither alone gets you there.

**Model-size prediction check (Table 2).** Theory said $P_{os}$ should reach 7.6B (MP=1, 64 GPUs) up to 121.6B (MP=16, 1024 GPUs). Measured: 6.2B and 100B. Close enough that the memory analysis is a usable planning tool, not just arithmetic.

**What did not work / cost something:**

- **$P_{a+cpu}$ (C5) is usually a throughput loss.** For the 60B model, C5 has lower memory than C4 but *worse* performance, because you pay for moving activations over PCIe both ways. It only pays off when the model is so large it will not run otherwise (170B in Figure 8) or when the achievable batch without it is tiny. The system turns it on adaptively, not always.
- **ZeRO-100B's own throughput dips past 100B params** — not enough memory left for large batches, so arithmetic intensity falls.
- **Stage 3 ($P_{os+g+p}$) was never actually run.** All the trillion-parameter claims are memory analysis, not measurement. Shipped later in DeepSpeed.
- **Memory savings from C2 → C3 are not monotone in an obvious way** — whether $P_g$ or $P_a$ helps more depends on whether activation memory or model-state memory dominates for your particular shape.

**Turing-NLG, 17B params.** Trained end to end on ZeRO-100B at a sustained **41.4 TFlops/GPU**, reaching **WebText-103 perplexity 10.21** over 300K iterations — SOTA at the time, beating Megatron-LM's 8.3B model.

## Worth Remembering

**The compute wall is separate from the memory wall.** The authors are honest: fitting 1T params on 1024 GPUs is solved by ZeRO, but *training* it is not. A 1T model is ~3000x the compute per sample of BERT-Large. BERT-Large takes 67 minutes on 1024 DGX-2H GPUs, so 1T would take **~140 days** assuming the same sample count and sequence length — and both grow with model size, so realistically over a year. You need an exaflop machine. This is exactly the trade-off that [[Training Compute-Optimal Large Language Models (Chinchilla)|Chinchilla]] would later reframe: memory-limited thinking made people build models too big for their token budget.

**ZeRO does not change the maths.** Unlike memory-efficient optimisers such as Adafactor, which coarsen the second-moment statistics and can perturb convergence, ZeRO is bit-identical to standard DP + Adam. It is a pure systems win. That is rare and worth internalising as a design principle: *first look for redundancy you can delete before you look for approximations you can tolerate.*

**MP is still useful, just not for the reason you think.** After ZeRO, you do not need MP to fit a model. You want it when (a) activation memory is the bottleneck and $P_a$ needs an MP group to partition across, or (b) the aggregate batch size from pure DP gets so large that convergence suffers. Combining gives up to $N_d \times N_m$ memory reduction. The recommended shape is MP *within* a DGX-2 node (16-way, fast NVSwitch) and DP *across* nodes.

**Communication complementarity.** $P_a$ costs ~10% more MP traffic but lets you raise the batch size by up to the MP degree (16x), and DP communication volume is inversely proportional to batch size. So $P_a$ can cut DP traffic by an order of magnitude. Two memory optimisations that also happen to fix each other's communication.

**Practical caveats.** MP degree constrains your shape — hidden size must divide by attention heads and by MP degree, heads must divide by MP degree. Also note the memory-fragmentation result: if you are OOMing with 30% free, the problem may be contiguity, not capacity, and `torch.cuda.memory_reserved()` vs `memory_allocated()` will tell you.

**Follow-ups worth chasing.** ZeRO-Offload and ZeRO-Infinity extend the CPU/NVMe offload direction. FSDP in PyTorch is essentially stage 3 upstreamed. The stage-3 broadcast-then-discard pattern is also what makes [[Flash Attention]]-style memory thinking natural: streaming state through fast memory rather than resident storage.

## Links

Related: [[Mixed Precision training]] · [[Backpropagation]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[Attention Is All You Need]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[Flash Attention]] · [[Momentum]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Deep Learning]] · [[The Bitter Lesson (essay)]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Foundational_Reading_Plan]]

New topics worth writing: Data parallelism vs model parallelism vs pipeline parallelism, Collective communication primitives (all-reduce, reduce-scatter, all-gather, ring algorithms), Activation checkpointing / gradient recomputation, Adam optimiser state and memory cost, DeepSpeed and FSDP, Megatron-LM tensor parallelism, GPipe and PipeDream, Arithmetic intensity and the roofline model, CUDA memory fragmentation and caching allocators, ZeRO-Offload and ZeRO-Infinity, Turing-NLG
