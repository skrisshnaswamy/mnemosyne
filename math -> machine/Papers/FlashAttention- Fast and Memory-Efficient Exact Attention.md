---
title: "FlashAttention: Fast and Memory-Efficient Exact Attention"
authors: ["Tri Dao", "Daniel Y. Fu", "Stefano Ermon", "Atri Rudra", "Christopher Ré"]
year: 2022
arxiv: "2205.14135"
url: https://arxiv.org/abs/2205.14135
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, transformers, llm]
---
## The Core Idea

Attention is slow on long sequences. Everyone knew that. Everyone assumed the fix was to do **fewer arithmetic operations** — that is what Reformer, Linformer, Performer and friends all did. They were mostly wrong about where the time goes.

The real bottleneck is **memory traffic**. A GPU has two kinds of memory: a big slow one (HBM, 40–80 GB on an A100, ~1.5–2.0 TB/s) and a tiny fast one (SRAM, 192 KB per streaming multiprocessor, ~19 TB/s — about 10× faster). Standard attention writes the full $N \times N$ score matrix $\mathbf{S}$ to HBM, reads it back to softmax it, writes $\mathbf{P}$ back, then reads it again to multiply by $\mathbf{V}$. For GPT-2 medium at $N=1024$, that is **40.3 GB of HBM reads/writes** for only 66.6 GFLOPs of maths. The chip spends its life waiting on memory.

FlashAttention never writes $\mathbf{S}$ or $\mathbf{P}$ to HBM at all. It computes attention in blocks that fit in SRAM, and keeps a running softmax that is corrected as each new block arrives. Result for the same GPT-2 medium layer: 75.2 GFLOPs (*more* FLOPs, because of recomputation), **4.4 GB of HBM traffic**, and runtime drops from 41.7 ms to 7.3 ms. Same numbers out — this is **exact** attention, not an approximation.

> [!NOTE] IO-aware algorithm
> An algorithm designed by counting reads and writes between levels of the memory hierarchy, not by counting arithmetic operations. The relevant ratio is *arithmetic intensity* — FLOPs per byte moved. ^io-aware

What it unlocks: memory now grows **linearly** in $N$ instead of quadratically. Context length 64K becomes possible on a single A100. It also explains why a decade of "efficient attention" papers reduced FLOPs but never got a wall-clock win — they were optimising the wrong quantity.

## The Methodology

Standard attention, with $\mathbf{Q},\mathbf{K},\mathbf{V} \in \mathbb{R}^{N \times d}$:

$$\mathbf{S}=\mathbf{Q}\mathbf{K}^\top, \quad \mathbf{P}=\mathrm{softmax}(\mathbf{S}), \quad \mathbf{O}=\mathbf{P}\mathbf{V}$$

The obstacle to blocking this is the softmax: the denominator of row $i$ sums over *all* columns, so you seemingly need the whole row before you can normalise anything. (See [[Cross Entropy]] for the softmax itself.)

### The tiling trick — online softmax

Numerically stable softmax of a vector $x$ uses the row max $m(x) = \max_i x_i$:

$$f(x) = [e^{x_1 - m(x)} \dots e^{x_B - m(x)}], \quad \ell(x) = \sum_i f(x)_i, \quad \mathrm{softmax}(x) = \frac{f(x)}{\ell(x)}$$

Split $x$ into two halves $x^{(1)}, x^{(2)}$. Then the statistics merge exactly:

$$m(x) = \max(m(x^{(1)}), m(x^{(2)}))$$
$$\ell(x) = e^{m(x^{(1)}) - m(x)}\ell(x^{(1)}) + e^{m(x^{(2)}) - m(x)}\ell(x^{(2)})$$

So if you carry two extra scalars per row — the running max $m$ and the running sum $\ell$ — you can process $\mathbf{K}$ and $\mathbf{V}$ one block at a time and rescale what you already have. That is the whole algorithmic content.

### The loop

Block sizes: $B_c = \lceil M/4d \rceil$ (for $\mathbf{K},\mathbf{V}$ blocks), $B_r = \min(\lceil M/4d\rceil, d)$ (for $\mathbf{Q}$ blocks), where $M$ is SRAM size. The $4d$ is because four things ($\mathbf{K}_j$, $\mathbf{V}_j$, $\mathbf{Q}_i$, $\mathbf{O}_i$) plus the $B_r \times B_c$ score tile must all fit.

- **Outer loop** over $j = 1 \dots T_c$: load $\mathbf{K}_j, \mathbf{V}_j$ into SRAM.
- **Inner loop** over $i = 1 \dots T_r$: load $\mathbf{Q}_i, \mathbf{O}_i, \ell_i, m_i$.
  - On chip: $\mathbf{S}_{ij} = \tau \mathbf{Q}_i \mathbf{K}_j^\top$, apply mask, take $\tilde m_{ij} = \mathrm{rowmax}(\mathbf{S}_{ij})$, $\tilde{\mathbf{P}}_{ij} = \exp(\mathbf{S}_{ij} - \tilde m_{ij})$, $\tilde\ell_{ij} = \mathrm{rowsum}(\tilde{\mathbf{P}}_{ij})$.
  - Update $m_i^{\text{new}} = \max(m_i, \tilde m_{ij})$ and $\ell_i^{\text{new}}$ by the merge rule above.
  - **Rescale and accumulate the output in place**:
$$\mathbf{O}_i \leftarrow \mathrm{diag}(\ell_i^{\text{new}})^{-1}\left(\mathrm{diag}(\ell_i)e^{m_i - m_i^{\text{new}}}\mathbf{O}_i + e^{\tilde m_{ij} - m_i^{\text{new}}}\tilde{\mathbf{P}}_{ij}^{\text{dropped}}\mathbf{V}_j\right)$$

Only $\mathbf{O}$, $\ell$, $m$ ever touch HBM. Extra memory beyond inputs/outputs is $O(N)$.

Note the sneaky detail: dropout is applied **inside** the kernel, and the RNG state $\mathcal{R}$ is saved rather than the $N \times N$ dropout mask.

### Recomputation instead of storage

The backward pass normally needs $\mathbf{S}$ and $\mathbf{P}$. Instead, FlashAttention saves only $\mathbf{O}$, $\ell$, $m$, $\mathcal{R}$, and rebuilds $\mathbf{P}_{ij} = \mathrm{diag}(\ell_i)^{-1}\exp(\mathbf{S}_{ij}^{\text{masked}} - m_i)$ on chip. This is selective gradient checkpointing — but unlike usual checkpointing, it is **faster**, not just smaller, because the recompute is cheaper than the HBM round trip. (See [[Backpropagation]] and [[Vector Jacobian Product]].)

Two clean bits of algebra make the backward pass work in $O(N)$ memory. The softmax Jacobian is $\mathrm{diag}(y) - yy^\top$, giving

$$dS_{ij} = P_{ij}(dP_{ij} - D_i), \qquad D_i = P_{i:}^\top dP_{i:}$$

and — the trick — $D_i = do_i^\top o_i$. So instead of reducing over a length-$N$ row, you take a dot product of two length-$d$ vectors you already have in SRAM.

### The IO complexity result

> [!NOTE] IO complexity of attention
> Standard: $\Theta(Nd + N^2)$ HBM accesses. FlashAttention: $\Theta(N^2 d^2 M^{-1})$. With $d \in [64,128]$ and $M \approx 100$KB, $d^2 \ll M$, so FlashAttention moves up to $9\times$ less data. ^flash-io-complexity

Intuition for the bound: each $\mathbf{K},\mathbf{V}$ element is loaded once; you make $T_c = \Theta(Nd/M)$ passes over $\mathbf{Q}$ and $\mathbf{O}$, each pass moving $\Theta(Nd)$ elements.

**Proposition 3** gives a matching lower bound: no exact attention algorithm can do $o(N^2d^2M^{-1})$ HBM accesses for *all* $M \in [d, Nd]$. Proof is one line — at $M = \Theta(Nd)$ that would mean $o(Nd)$ accesses, but the inputs alone are $Nd$ elements sitting in HBM.

### Block-sparse variant

Given a block mask $\mathbf{M} \in \{0,1\}^{N/B_r \times N/B_c}$, just skip the zero blocks. IO complexity becomes $\Theta(Nd + N^2d^2M^{-1}s)$ where $s$ is the nonzero fraction. They use a **fixed butterfly sparsity pattern** (never learned) — a pattern known to be able to express arbitrary structured matrices.

### Implementation

One fused CUDA kernel. Built on top of Nvidia Apex's FMHA code. Supports head dims 16/32/64/128, all Turing and Ampere GPUs.

## Ablations Studies and Experiments

**Is memory traffic really the cause?** Two direct checks. (1) FlashAttention does *more* FLOPs than standard attention (75.2 vs 66.6 GFLOPs) yet is 5.7× faster — the only thing that changed is HBM traffic (40.3 → 4.4 GB). (2) Sweeping block size $B_c$: bigger blocks → fewer HBM accesses → faster, **up to about 256**, after which runtime is bottlenecked by arithmetic and larger blocks don't fit in SRAM anyway. That plateau is the clean evidence that the IO story is the whole story until it isn't.

**Training speed.**

| Model | Baseline | FlashAttention |
|---|---|---|
| BERT-large (seq 512), to 72.0% MLM acc | 20.0 ± 1.5 min (Nvidia MLPerf 1.1 record) | **17.4 ± 1.4 min** (15% faster) |
| GPT-2 small, OpenWebText, ppl 18.2 | 9.5 d (HuggingFace) / 4.7 d (Megatron) | **2.7 d** (3.5× / 2.0×) |
| GPT-2 medium, ppl 14.3 | 21.0 d (HF) / 11.5 d (Megatron) | **6.9 d** (3.0× / 1.8×) |
| LRA (seq 1K–4K) | — | 2.4× (block-sparse: 2.8×) |

Perplexity is identical to baselines, and the validation curves lie on top of each other — worth noting because a fused FP16 kernel with rescaling *could* have been numerically worse. It isn't.

**Longer context is free quality.** GPT-2 small at 4K context with FlashAttention trains in 3.6 days versus Megatron's 4.7 days at 1K context — 30% faster with 4× the context — and gets ppl 17.5 vs 18.2 (0.7 better).

Long-document classification (micro-$F_1$), fine-tuning RoBERTa with repeated positional embeddings:

| seq len | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|
| MIMIC-III | 52.8 | 50.7 | 51.7 | 54.6 | 56.4 | **57.1** |
| ECtHR | 72.2 | 74.3 | 77.1 | 78.6 | **80.7** | 79.2 |

Note MIMIC *dips* at 1024 before recovering — the authors guess a distribution shift in document length for specialised medical text. ECtHR peaks at 8K and drops at 16K.

**Path-X / Path-256.** Classify whether two dots in a 128×128 (or 256×256) black-and-white image are connected by a path, fed one pixel at a time — sequence length 16K and 64K. Every prior Transformer either OOM'd or scored chance. FlashAttention gets **61.4%** on Path-X; block-sparse gets **63.1%** on Path-256. They pretrain on Path-64, upsample the positional embeddings spatially, then fine-tune.

**LRA accuracy (Table 3)** — this is the ablation that quietly demolishes the approximate-attention field:

| Model | Avg | Speedup |
|---|---|---|
| Transformer | 59.3 | — |
| FlashAttention | **59.8** | 2.4× |
| Block-sparse FlashAttention | 59.6 | 2.8× |
| Linformer | 54.9 | 2.5× |
| Linear Attention | 59.6 | 2.3× |
| Performer | 58.9 | 1.8× |
| Reformer | 57.6 | 1.3× |

Approximate methods give up accuracy for speedups that exact FlashAttention matches or beats.

**What did not work / where it loses.**
- **The backward pass is slower than Apex FMHA** at short sequences: 0.20 vs 0.17 ms at $N=128$. Recomputation costs you when there was never much memory pressure. Overall FlashAttention is 4% *slower* than FMHA at $N=128$, 8% faster at 256, 5% faster at 512.
- **Approximate methods do eventually win on raw speed.** Crossover happens somewhere between $N=512$ and $N=1024$. At $N=65536$, dense FlashAttention forward+backward is 9341 ms; block-sparse is 64 ms; Linformer is faster than dense too. Dense FlashAttention is still $O(N^2)$ in compute — it just has a much better constant.
- **Head dimension 128 gives less speedup** than $d=64$, because blocks must shrink to fit SRAM. Still up to 3× with a causal mask (half the blocks masked out entirely).
- **Hardware-dependent.** RTX 3090 sees *more* speedup (2.5–4.5×) than A100, because its HBM bandwidth is worse (~900 GB/s vs 1.5 TB/s) so the memory bottleneck is tighter. T4 sees *less*, because its SRAM is smaller so blocks shrink — exactly what $\Theta(N^2d^2M^{-1})$ predicts.
- Memory at $N=65536$: FlashAttention 13,376 MB versus PyTorch OOM'ing at 8192. Up to 20× more memory-efficient than exact baselines, 2× better than Linformer.

## Worth Remembering

**Versus Rabe & Staats (2021)**, who had already shown attention needs only $O(\log n)$ memory. Three differences the authors are careful about: (i) that work targets *peak memory*, this one targets *memory accesses* — hence Rabe & Staats is roughly the same speed as standard attention while FlashAttention is 2–4× faster; (ii) they combine per-block temporary outputs at the end ($K$ copies for $K$ blocks), FlashAttention updates $\mathbf{O}$ in place (one copy); (iii) they use generic gradient checkpointing in the backward, FlashAttention derives the backward analytically and only recomputes $\mathbf{S}, \mathbf{P}$, not the temporary outputs.

**Limitations the authors admit.** Each new attention variant needs a hand-written CUDA kernel — significant engineering effort, and kernels may not transfer across GPU architectures. They want a Halide-style compiler for this. Also, the optimality result is single-GPU only; multi-GPU adds another memory level (other GPUs' HBM) that isn't analysed.

**The generalisable lesson.** Most Transformer operations are memory-bound: elementwise ops (activation, dropout) and reductions (softmax, [[How Does Batch Normalization Help Optimization|batch norm]], layer norm). Only large matmuls are compute-bound. Every layer touches HBM. The IO-aware framing should apply well beyond attention — the authors flag sparse MLP layers and kernel methods (kernel matrices are also functions of low-rank inputs, so the same load-and-recompute trick applies; the KeOps library is the existing example).

**Practical caveats if you want to use it.**
- The speedup is *bigger* with dropout and masking, because kernel fusion avoids separate passes for those too.
- Numerically it's fine in FP16 — the max-subtraction is done per block and merged exactly. See [[Mixed Precision training]].
- Don't expect wins below $N \approx 256$. If your sequences are short, the fused FMHA-style kernel is already close to optimal.
- The block-sparse pattern is fixed at butterfly, not learned. It's a "fixed lottery ticket" for attention, and it performs nearly as well as dense on LRA.

**Follow-up questions.** How does this interact with [[RoFormer- Enhanced Transformer with Rotary Position Embedding|rotary embeddings]], which modify $\mathbf{Q},\mathbf{K}$ before the score computation (answer: fine, applied before the kernel)? What about inference with a KV cache, where $\mathbf{Q}$ has one row — the tiling structure is quite different, and this is what FlashDecoding later addresses. And the lower bound in Proposition 3 is only over the full range of $M$; a parameterised lower bound in terms of $M$ is left open.

## Links

Related: [[Attention Is All You Need]] · [[Flash Attention]] · [[Query, Key, and Value (QKV)]] · [[Causal Attention]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models]] · [[Mixed Precision training]] · [[Backpropagation]] · [[Cross Entropy]] · [[Vector Jacobian Product]] · [[Scaling Laws for Neural Language Models]] · [[An Image is Worth 16x16 Words (ViT)]] · [[The Bitter Lesson (essay)]] · [[Decoupled Weight Decay Regularization (AdamW)]]

New topics worth writing: GPU memory hierarchy (HBM vs SRAM vs registers), arithmetic intensity and the Roofline model, kernel fusion, gradient checkpointing, online softmax (Milakov & Gimelshein), Long Range Arena benchmark, butterfly matrices and structured sparsity, Linformer / Performer / Reformer, FlashAttention-2 and FlashDecoding, CUDA kernel programming for ML
