---
title: "Megatron-LM: Training Multi-Billion Parameter Models Using Model Parallelism"
authors: ["Mohammad Shoeybi", "Mostofa Patwary", "Raul Puri", "Patrick LeGresley", "Jared Casper", "Bryan Catanzaro"]
year: 2019
arxiv: "1909.08053"
url: https://arxiv.org/abs/1909.08053
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, transformers, llm]
---
## The Core Idea

A transformer with 8 billion parameters does not fit on one GPU. With [[Adam- A Method for Stochastic Optimization|Adam]] you need roughly 16 bytes per parameter (fp16 weights, fp32 master copy, two moment buffers), so 8B parameters is ~128 GB of state before you store a single activation. A V100 has 32 GB.

The trick here is **intra-layer model parallelism**: cut each individual weight matrix into pieces, put the pieces on different GPUs, and choose *where* to cut so that the pieces almost never need to talk to each other. Not "layer 1–10 on GPU 0, layer 11–20 on GPU 1" (that is pipeline parallelism, which has bubbles), and not a new compiler like Mesh-TensorFlow. Just a handful of `all-reduce` calls dropped into a normal PyTorch [[Attention Is All You Need|transformer]].

The insight is that matrix multiplication splits two ways, and only one of them survives a nonlinearity. For $Y = \text{GeLU}(XA)$ you can either split $A$ by rows (and $X$ by columns), which forces you to sum the partial products *before* the GeLU — a synchronisation point — or split $A$ by **columns**, which lets each GPU apply GeLU to its own slice independently. Pick column-splitting for the first matrix, row-splitting for the second, and the two matmuls of the MLP fuse into one communication-free region. The same reasoning applies to attention, where the heads are already independent, so you hand each GPU its own subset of heads.

Result: **two all-reduces in the forward pass and two in the backward pass per transformer layer**, and nothing else. That is the whole method. It unlocked 8.3B-parameter GPT-2 and 3.9B-parameter BERT in 2019, and the tensor-parallel code in essentially every large-model training stack today descends from it.

> [!NOTE] Tensor (intra-layer) parallelism
> Splitting a single weight matrix across devices so each device holds a slice and computes a partial output. Contrast with data parallelism (split the batch) and pipeline parallelism (split the layers). ^tensor-parallel

## The Methodology

**The MLP block.** The block is $Y = \text{GeLU}(XA)$ then $Z = YB$. Split
$$A = [A_1, A_2], \qquad B = \begin{bmatrix}B_1 \\ B_2\end{bmatrix}$$
GPU $i$ computes $Y_i = \text{GeLU}(XA_i)$ from the full input $X$ — no communication, because GeLU is elementwise and $\text{GeLU}(XA_1)$ is genuinely the first half of the answer. Then GPU $i$ computes $Y_iB_i$, and the two partial results are summed with one all-reduce. The rejected alternative — row-splitting $A$ — fails because $\text{GeLU}(X_1A_1 + X_2A_2) \neq \text{GeLU}(X_1A_1) + \text{GeLU}(X_2A_2)$.

**The attention block.** The $Q$, $K$, $V$ projections ([[Query, Key, and Value (QKV)|QKV]]) are column-split so each GPU owns whole heads. Each head's softmax attention runs entirely locally. The output projection is row-split and consumes the local head outputs directly. One all-reduce at the end.

**The `f` / `g` operators.** These are the only new code. They are conjugates:

- `f`: identity forward, all-reduce backward (scatters the input into the parallel region).
- `g`: all-reduce forward, identity backward (gathers the output).

In PyTorch this is a five-line `torch.autograd.Function`. That is the "no compiler needed" claim — see [[Backpropagation]] and [[Pytorch Autograd]] for why a custom backward is all you need.

**The embedding, which is the fiddly bit.** Input and output embeddings are tied and have shape $H \times v$ with $v \approx 50{,}257$. They split $E = [E_1, E_2]$ along vocabulary. The naive output path computes logits $[XE_1, XE_2]$ and all-gathers them — that moves $b \times s \times v$ numbers, which is enormous. Instead they **fuse the parallel GEMM with the [[Cross Entropy|cross-entropy]] loss**, so each GPU computes its share of the log-sum-exp and only $b \times s$ scalar losses cross the wire. Vocabulary is padded from 50,257 to 51,200 so the per-GPU slice is a multiple of 128 (tensor-core-friendly GEMM shapes).

**What is duplicated rather than split.** [[Layer Normalization|LayerNorm]] parameters, [[Dropout- A Simple Way to Prevent Overfitting|dropout]], and residual adds are *replicated* on every GPU. Recomputing them is cheaper than communicating them. Each model-parallel worker optimises its own parameter slice, so no optimiser state is exchanged.

**Random numbers.** Dropout outside the parallel region must be *identical* across workers (same residual stream), dropout *inside* the attention block must be *different* (real randomness across heads). So they keep two RNGs: one seeded identically everywhere, one seeded per-rank.

**Hybrid with data parallelism.** 8 GPUs inside a DGX-2H server (300 GB/s NVSwitch) form a model-parallel group; the same rank across 64 servers forms a data-parallel group that all-reduces gradients over InfiniBand. $8 \times 64 = 512$ GPUs.

**Training recipe.**
- [[Mixed Precision training|Mixed precision]] fp16 with dynamic loss scaling.
- Init $W \sim \mathcal{N}(0, 0.02)$; weights immediately before residual adds scaled by $1/\sqrt{2N}$ with $N$ = number of layers.
- [[Decoupled Weight Decay Regularization (AdamW)|Adam with decoupled weight decay]] $\lambda = 0.01$, global grad-norm clip 1.0, dropout 0.1.
- Activation checkpointing after every transformer layer (recompute activations in the backward pass instead of storing them).
- GPT-2: seq len 1024, batch 512, 300k iterations, LR 1.5e-4, 3k warmup, single-cycle cosine decay to 1e-5.
- BERT: batch 1024, LR 1e-4, 10k warmup, linear decay over 2M iterations, sentence-order prediction instead of next-sentence prediction, whole-word n-gram masking.
- Data: Wikipedia + CC-Stories + RealNews + OpenWebText, deduplicated with LSH at Jaccard 0.7, documents under 128 tokens dropped → **174 GB** of text.

**The BERT fix.** Scaling [[BERT- Pre-training of Deep Bidirectional Transformers|BERT]] past 336M parameters was known to *degrade*. The cause is the post-LayerNorm block ordering. Megatron moves the LayerNorm to the *input* of each sub-block, so the residual path is a clean identity from input to output (pre-LN). This alone removed the instability and gave lower training loss. It is the same argument as [[Deep Residual Learning for Image Recognition (ResNet)|ResNet]]'s clean skip path.

## Ablation Studies and Experiments

**Scaling (weak scaling, ~1B parameters per GPU).** Baseline: a 1.2B model on one V100 sustaining **39 TFLOPs**, 30% of theoretical peak — a genuinely strong baseline, not a strawman.

| Params | Hidden | Heads | Layers | MP GPUs | MP+DP GPUs |
|---|---|---|---|---|---|
| 1.2B | 1536 | 16 | 40 | 1 | 64 |
| 2.5B | 1920 | 20 | 54 | 2 | 128 |
| 4.2B | 2304 | 24 | 64 | 4 | 256 |
| 8.3B | 3072 | 32 | 72 | 8 | 512 |

8-way model parallel: **77% of linear scaling**. Adding 64-way data parallelism (512 GPUs): **74%**, sustaining **15.1 PetaFLOPs** across the application.

**Strong scaling (fixed 1.2B model, fixed batch of 8).** Speedup 1.0 → 1.64 → 2.34 → 2.98 on 1/2/4/8 GPUs. Clearly sub-linear: more GPUs means smaller per-GPU GEMMs, and communication plus memory bandwidth start to dominate. Tensor parallelism is a memory tool first, a speed tool second.

**The attention-head ablation — the sharpest one.** Fix the 8.3B model at 8-way parallelism and vary head count:

| Heads | Hidden per head | Scaling efficiency |
|---|---|---|
| 16 | 192 | 82% |
| 24 | 128 | 80% |
| 32 | 96 | 77% |

More heads is *worse* for throughput, because each head's GEMMs shrink and the softmax element count grows. Head count is a systems hyperparameter, not just a modelling one. (This tension is what [[FlashAttention- Fast and Memory-Efficient Exact Attention|FlashAttention]] later attacks from the memory side.)

**GPT-2 zero-shot results.**

| Model | WikiText103 PPL ↓ | LAMBADA acc ↑ |
|---|---|---|
| 355M | 19.31 | 45.18% |
| 2.5B | 12.76 | 61.73% |
| 8.3B | **10.81** | **66.51%** |
| Previous SOTA | 15.79 | 63.24% |

Validation perplexity for 8.3B reached 9.27. Larger models converge *faster in iterations*, not just to a better endpoint — the same observation that drives [[Scaling Laws for Neural Language Models|scaling laws]].

**BERT downstream (Table 5).** Megatron-3.9B, trained on *half* the tokens of RoBERTa and a third of ALBERT:

| Model | MNLI m/mm | QQP | SQuAD 1.1 F1/EM | SQuAD 2.0 F1/EM | RACE |
|---|---|---|---|---|---|
| RoBERTa | 90.2/90.2 | 92.2 | 94.6/88.9 | 89.4/86.5 | 83.2 |
| ALBERT | 90.8 | 92.2 | 94.8/89.3 | 90.2/87.4 | 86.5 |
| Megatron-336M | 89.7/90.0 | 92.3 | 94.2/88.0 | 88.1/84.8 | 83.0 |
| Megatron-1.3B | 90.9/91.0 | 92.6 | 94.9/89.1 | 90.2/87.1 | 87.3 |
| Megatron-3.9B | **91.4/91.4** | **92.7** | **95.5/90.0** | **91.2/88.5** | **89.5** |

Ensemble of 5 × 3.9B gets RACE 90.9% (previous ensemble SOTA 89.4%). Held-out perplexity falls monotonically 1.58 → 1.30 → 1.16 with size.

**What did not work:**

1. **Row-splitting the first MLP matrix.** Mathematically fine, but forces an all-reduce before the GeLU, doubling communication per block. Rejected on the whiteboard, not in an experiment.
2. **All-gathering logits before the loss.** $b \times s \times v$ elements is ruinous at $v = 51{,}200$. Fusing with cross-entropy cuts it to $b \times s$.
3. **The original post-LayerNorm BERT block.** Trains fine at 336M, becomes unstable and gets *worse* loss at 752M. ALBERT's answer was parameter sharing (which caps capacity); Megatron's answer was moving the LayerNorm, which does not.
4. **Weak scaling by growing the batch.** Explicitly rejected as a scaling strategy — large batches hurt convergence (Keskar et al.), so they scale the *model* instead.
5. **Pushing past 8-way tensor parallelism.** Not attempted, and the authors say why: past 16B parameters you exceed a single DGX-2H node, and all-reduce over InfiniBand instead of NVSwitch would wreck efficiency. They call for hybrid intra-layer + pipeline + inter-node parallelism — which is exactly what Megatron-LM v2/v3 became.

## Worth Remembering

- **Tensor parallelism is bandwidth-hungry and node-local.** Four all-reduces per layer per step over the *full activation tensor* is only tolerable at NVSwitch speeds (300 GB/s intra-node vs 100 GB/s inter-node here). This is the reason production stacks use tensor parallelism *inside* a node and pipeline or data parallelism *across* nodes.
- **It is orthogonal to [[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models|ZeRO]].** ZeRO shards optimiser state across data-parallel ranks without changing the math of a layer; Megatron shards the layer itself. Modern training uses both plus pipelining ("3D parallelism").
- **Pre-LN was discovered here as a systems fix, not a theory result.** The paper claims to be the first to report that this reordering enables larger BERTs. Every transformer you train today is pre-LN.
- **Contamination check.** WikiText103 test has ≤10.8% 8-gram overlap with the training corpus — but the WikiText103 *train* set already overlaps its own test set by 9.09%, so the marginal leakage is small. Worth copying this habit.
- **Perplexity accounting is subtle.** They normalise cross-entropy by the *original word-level* token count $T_o = 245{,}566$, not their subword count $T = 270{,}329$, and use sliding-window "overlapping evaluation" with stride 32 because a transformer only sees $P(t \mid t-w:t-1)$ with $w = 1024$. Compare perplexities across papers only if the tokenisation and window protocol match.
- **The honest limitation on cost.** One epoch of the 8.3B model takes ~2.1 days on 512 V100s. Nobody was going to iterate on architecture at that scale — which is partly why [[Training Compute-Optimal Large Language Models (Chinchilla)|Chinchilla]] mattered so much three years later.
- **Bitter-lesson exhibit A.** No new architecture, no clever inductive bias — just more parameters, more data, and the engineering to make it fit. See [[The Bitter Lesson (essay)]].
- Future work they flagged: [[Distilling the Knowledge in a Neural Network|distilling]] small students from these teachers, and other model families ([[Exploring the Limits of Transfer Learning (T5)|T5]], XLNet). Turing-NLG (17B) was built on this code within months, and [[Language Models are Few-Shot Learners (GPT-3)|GPT-3]] followed.

## Links
Related: [[Attention Is All You Need]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[Layer Normalization]] · [[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models]] · [[Mixed Precision training]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Adam- A Method for Stochastic Optimization]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Cross Entropy]] · [[Query, Key, and Value (QKV)]] · [[Backpropagation]] · [[Pytorch Autograd]] · [[RoBERTa- A Robustly Optimized BERT Pretraining Approach]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[The Bitter Lesson (essay)]] · [[Distilling the Knowledge in a Neural Network]]

New topics worth writing: All-reduce and NCCL collectives, Pipeline parallelism (GPipe / PipeDream), Activation checkpointing, Pre-LN vs Post-LN transformers, GeLU activation, 3D parallelism, LAMBADA and cloze evaluation, Weak vs strong scaling, Locality-sensitive hashing for deduplication
