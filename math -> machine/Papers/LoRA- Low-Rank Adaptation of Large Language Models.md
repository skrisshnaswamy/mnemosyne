---
title: "LoRA: Low-Rank Adaptation of Large Language Models"
authors: ["Edward J. Hu", "Yelong Shen", "Phillip Wallis", "Zeyuan Allen-Zhu", "Yuanzhi Li", "Shean Wang", "Lu Wang", "Weizhu Chen"]
year: 2021
arxiv: "2106.09685"
url: https://arxiv.org/abs/2106.09685
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, transformers, llm, scaling]
---
## The Core Idea

Fine-tuning a big model means making a full copy of it. GPT-3 has 175 billion weights; a fine-tuned copy for each task is 350 GB of checkpoint each. That is a deployment problem, not an inconvenience.

The insight: **the *change* you make to the weights during fine-tuning is low-rank**, even though the weights themselves are full-rank. If $W_0$ is a pre-trained weight matrix of shape $d \times k$, and fine-tuning would push it to $W_0 + \Delta W$, then $\Delta W$ can be written as a product of two skinny matrices $BA$ where $B$ is $d \times r$ and $A$ is $r \times k$, with $r$ tiny — as small as **1 or 2**, even when $d = 12{,}288$.

> [!NOTE] Rank
> The rank of a matrix is the number of genuinely independent directions it can move a vector in. A $12288 \times 12288$ matrix can have rank up to 12,288. Forcing it to rank 2 means it only stretches space along 2 directions. ^rank

So instead of learning $\Delta W$ (151 million numbers per matrix), you learn $A$ and $B$ (49 thousand numbers at $r=2$). Freeze $W_0$ entirely. Train only $A$ and $B$.

Why this had not been done: low-rank factorisation of weights was old news — people compressed networks with it, or trained networks under a rank constraint from scratch. Nobody applied a low-rank constraint to *the update of a frozen model for downstream adaptation*. The enabling evidence came from Aghajanyan et al. (2020), who showed pre-trained language models have low "intrinsic dimension" — you can fine-tune them well inside a small random subspace.

What it unlocks, concretely:

- **Checkpoint size: 350 GB → 35 MB** for GPT-3 at $r=4$ on query and value matrices. A 10,000× reduction.
- **Training VRAM: 1.2 TB → 350 GB**, because Adam's momentum and variance buffers only need to exist for the tiny matrices.
- **25% faster training** on GPT-3, since you skip gradients for 99.99% of parameters.
- **Zero extra inference latency.** This is the part that beat adapters. Because $\Delta W = BA$ has the same shape as $W_0$, at deploy time you just compute $W = W_0 + BA$ once and ship a normal model. No new layers, no new depth.
- **Task switching is subtraction.** Swap $BA$ for $B'A'$ on a shared frozen base.

## The Methodology

**The forward pass.** For a layer that used to compute $h = W_0 x$:

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} BA x$$

$W_0$ frozen. $A$ and $B$ trainable. Note $BAx$ runs in **parallel** with $W_0 x$, not sequentially after it — that is the whole difference from an adapter, which sits in the middle of the residual stream and must be waited on.

**Initialisation.** $A$ gets random Gaussian entries, $B$ gets **all zeros**. So $\Delta W = BA = 0$ at step 0 and the model starts as exactly the pre-trained model. If both were random you would corrupt the model before training began; if both were zero the [[Derivative#Gradient|gradient]] would be zero forever and nothing would move.

**The $\alpha$ scaling.** $\Delta W x$ is scaled by $\alpha / r$. Under Adam, tuning $\alpha$ is roughly equivalent to tuning the learning rate, so the authors set $\alpha$ to the first $r$ they tried and never tuned it again. The point of dividing by $r$ is that you can change $r$ without re-tuning the learning rate.

**Where they put it.** A Transformer block has four attention matrices — $W_q, W_k, W_v, W_o$ (see [[Query, Key, and Value (QKV)]]) — and two MLP matrices. They applied LoRA **only to attention weights**, and in most runs only to $W_q$ and $W_v$. The MLP, LayerNorms and biases stay frozen. They treat $W_q$ as one $d_{model} \times d_{model}$ matrix rather than slicing per head.

Trainable parameter count: $|\Theta| = 2 \times \hat{L}_{LoRA} \times d_{model} \times r$, where $\hat{L}_{LoRA}$ is how many matrices you attach to.

**Training objective.** Unchanged — the standard [[Auto-regressive models|autoregressive]] language modelling loss, just optimised over $\Theta$ instead of all of $\Phi$:

$$\max_{\Theta} \sum_{(x,y)\in\mathcal{Z}} \sum_{t=1}^{|y|} \log p_{\Phi_0 + \Delta\Phi(\Theta)}(y_t \mid x, y_{<t})$$

**Hyperparameters that mattered.** [[Decoupled Weight Decay Regularization (AdamW)|AdamW]] throughout, linear LR decay. GPT-3: 2 epochs, batch 128, weight decay 0.1, LR 2e-4 (vs 5e-6 for full fine-tuning — LoRA wants a much larger learning rate). GPT-2: $r_q = r_v = 4$, $\alpha = 32$, LR 2e-4, 5 epochs. RoBERTa: $r_q = r_v = 8$, LR 2e-4 to 5e-4.

**Why it generalises full fine-tuning.** If you apply LoRA to every matrix and set $r$ to the full rank of $W$, you recover the expressiveness of full fine-tuning. As you add parameters, LoRA converges toward training the original model. Adapters converge toward "an MLP bolted on"; prefix methods converge toward "a model that can't read long inputs".

## Ablation Studies and Experiments

**GLUE, RoBERTa and DeBERTa (Table 2).** Averages across 8 tasks:

| Model | Method | Trainable | Avg |
|---|---|---|---|
| RoBERTa-base | full FT | 125M | 86.4 |
| RoBERTa-base | AdapterD | 0.9M | 85.4 |
| RoBERTa-base | **LoRA** | **0.3M** | **87.2** |
| RoBERTa-large | full FT | 355M | 88.9 |
| RoBERTa-large | **LoRA** | **0.8M** | **89.0** |
| DeBERTa-XXL | full FT | 1500M | 91.1 |
| DeBERTa-XXL | **LoRA** | **4.7M** | **91.3** |

The RTE column is the loud one: LoRA 86.6 vs adapter 71.5 at similar budget on base.

**E2E NLG, GPT-2 medium (Table 3).** BLEU: full FT 68.2 (355M params), Adapter-L 68.9 (11M), PreLayer 69.7 (0.35M), **LoRA 70.4 (0.35M)**. Same story on DART (LoRA 47.1 BLEU vs FT 46.2) and WebNLG.

**GPT-3 175B (Table 4).**

| Method | Params | WikiSQL | MNLI-m | SAMSum R1/R2/RL |
|---|---|---|---|---|
| Fine-Tune | 175,256M | 73.8 | 89.5 | 52.0/28.0/44.5 |
| BitFit | 14.2M | 71.3 | 91.0 | 51.3/27.4/43.5 |
| PreEmbed | 3.2M | 63.1 | 88.6 | 48.3/24.2/40.5 |
| PreLayer | 20.2M | 70.1 | 89.5 | 50.8/27.3/43.5 |
| AdapterH | 40.1M | 73.2 | 91.5 | 53.2/29.0/45.1 |
| **LoRA** | **4.7M** | **73.4** | **91.7** | **53.8/29.8/45.9** |

LoRA beats full fine-tuning with 0.003% of the trainable parameters.

**The latency measurement that justifies the design (Table 1).** GPT-2 medium forward pass, ms:

| Batch / SeqLen | LoRA (=FT) | AdapterL | AdapterH |
|---|---|---|---|
| 32 / 512 | 1449.4 | +2.2% | +3.0% |
| 16 / 256 | 338.0 | +5.0% | +8.4% |
| **1 / 128** | **19.8** | **+20.7%** | **+30.3%** |

Adapters look free when batches are big — the GPU has idle parallelism to soak up the extra FLOPs. At batch size 1, which is exactly online serving, you eat 20–30%. The adapter layers must run *sequentially* after the block; there is no way around it. Worse under model sharding, where the extra depth means more `AllReduce`/`Broadcast` syncs.

### Which weights to adapt (Table 5)

Fixed budget of 18M params on GPT-3, 96 layers. WikiSQL accuracy:

| $W_q$ | $W_k$ | $W_v$ | $W_o$ | $W_q,W_k$ | $W_q,W_v$ | all four |
|---|---|---|---|---|---|---|
| $r{=}8$: 70.4 | 70.0 | 73.0 | 73.2 | $r{=}4$: 71.4 | **73.7** | $r{=}2$: 73.7 |

**Spreading a small rank over more matrices beats concentrating a large rank on one.** $W_q$ alone at $r=8$ is 3 points worse than $\{W_q, W_v\}$ at $r=4$.

### How small can $r$ go (Table 6)

WikiSQL:

| Weights | $r{=}1$ | $r{=}2$ | $r{=}4$ | $r{=}8$ | $r{=}64$ |
|---|---|---|---|---|---|
| $W_q$ | 68.8 | 69.6 | 70.5 | 70.4 | 70.0 |
| $W_q,W_v$ | **73.4** | 73.3 | 73.7 | 73.8 | 73.5 |
| all four | 74.1 | 73.7 | 74.0 | 74.0 | 73.9 |

Rank **one** is enough. Going to 64 buys nothing — the curve is flat, not rising. On GPT-2 medium (Table 18) the optimum is a bit higher, around $r=4$ to $16$, then flat out to $r=1024$.

### Why rank 1 works — the subspace analysis

They took the learned $A_{r=8}$ and $A_{r=64}$ from the same base model, did SVD on each, and measured overlap of the top-$i$ and top-$j$ singular directions using a normalised Grassmann similarity:

$$\phi(A_{r=8}, A_{r=64}, i, j) = \frac{\|U^{i\top}_{A_{r=8}} U^{j}_{A_{r=64}}\|_F^2}{\min(i,j)} \in [0,1]$$

Result: the **top singular direction** overlaps strongly ($\phi > 0.5$ for the 1-dimensional shared subspace). Everything below the top few directions does not overlap at all. Two runs with different random seeds at $r=64$ share only their top handful of directions. Conclusion: the extra 60 directions in $r=64$ are noise accumulated during training, not signal. That is *why* $r=1$ suffices.

### What $\Delta W$ actually does to $W$ (Table 7)

Project $W_q$ onto the subspace of $\Delta W_q$ via $U^\top W_q V^\top$, where $U, V$ are $\Delta W_q$'s singular vectors. Layer 48 of GPT-3, $\|W_q\|_F = 61.95$:

| | $\Delta W_q$ basis | $W_q$'s own top-$r$ basis | random basis |
|---|---|---|---|
| $r=4$ | 0.32 | 21.67 | 0.02 |
| $r=64$ | 1.90 | 37.71 | 0.33 |

Read it this way. $\Delta W$'s directions are far more aligned with $W$ than random (0.32 vs 0.02), so LoRA is not inventing new features. But those directions are **not** the top directions of $W$ (0.32 vs 21.67), so it is not just rescaling what the model already emphasises. And $\|\Delta W_q\|_F = 6.91$ against a projection of 0.32 gives an **amplification factor of ~21.5**. LoRA finds ~4 feature directions per layer that pre-training learned but left quiet, and turns them up 20×. At $r=64$ the amplification drops to ~2 — more evidence the useful rank is genuinely small.

### What did not work

- **Prefix-embedding and prefix-layer tuning do not scale monotonically.** PreEmbed on WikiSQL: 55.9 → 58.7 → 60.6 → 63.1 → **55.9** as you go from 32 to 512 prefix tokens. PreLayer peaks at 70.1 (20.2M) then falls to 64.9 (76.1M). Adding capacity makes them *worse*. Guess: more special tokens push the input distribution further from what pre-training saw. Also every prefix token you spend is sequence length you cannot use for the actual task.
- **Prefix methods collapse in the low-data regime (Table 16).** MNLI with only 100 examples: fine-tune 60.2, LoRA 63.8, PrefixLayer 48.3, PrefixEmbed **37.6** — barely above the 33.3% random-chance floor for 3-way classification.
- **LoRA + prefix-layer tuning (LoRA+PL) is worse than LoRA alone** (72.9 vs 73.8 on WikiSQL) *despite more parameters*. Prefix-layer tuning is very sensitive to learning rate, and that sensitivity contaminates the optimisation of the LoRA weights.
- **LoRA + prefix-embedding (LoRA+PE) does help** on WikiSQL — 76.2 vs LoRA's 73.8 — so the two are somewhat orthogonal. On MNLI it did not help, likely because LoRA was already near ceiling.
- **Larger rank does not help.** $r=64$ on $\{W_q,W_v\}$: 73.5, *below* $r=8$'s 73.8.

## Worth Remembering

**The limitation the authors admit.** If you merge $BA$ into $W_0$ to get zero latency, you cannot batch requests for different tasks in one forward pass — the merged weights belong to one task. You can keep them unmerged and route per-sample, but then you pay the extra matmul. In practice most serving stacks now do exactly this (unmerged, batched multi-LoRA), so the "no latency" claim is a design *option*, not a free lunch.

**They never tested MLP or LayerNorm.** Explicitly left to future work. Later practice (QLoRA and friends) found applying LoRA to *all* linear layers including the MLP does help, so this ablation is genuinely incomplete rather than settled.

**Few-shot is not a substitute for weight updates (Appendix A).** GPT-3 few-shot on MNLI-m: 40.6. Fine-tuned: 89.5. If you have a few thousand labelled examples, [[In Context Learning|in-context learning]] leaves a huge amount on the table.

**The intuition to keep.** Pre-training learns a huge basis of features. A downstream task does not need new features — it needs *a handful of existing but under-weighted features turned up loudly*. Four directions, amplified 20×. That is a strong claim about what adaptation actually is, and it is the reason the method works at all, not a side observation.

**Open question the paper flags.** If $\Delta W$ is rank-deficient, is $W$ itself rank-deficient? Nobody checked. Connects naturally to [[Understanding Dimensional Collapse in Contrastive Learning|dimensional collapse]] and to the anisotropy literature.

**Practical caveats.** Use a learning rate ~40× larger than full fine-tuning (2e-4 vs 5e-6 for GPT-3). Set $\alpha$ once and forget it — but remember the effective scale is $\alpha/r$, so if you copy someone's $\alpha$ and change $r$ you have silently changed the learning rate. And the choice of *which* matrices matters more than the choice of $r$: two matrices at $r=4$ beats one at $r=8$.

## Links

Related: [[Language Models are Few-Shot Learners (GPT-3)]] · [[Attention Is All You Need]] · [[Query, Key, and Value (QKV)]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Linear Projection]] · [[In Context Learning]] · [[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models]] · [[Mixed Precision training]] · [[Foundation Models]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Distillation]] · [[Regularization]] · [[Fundamentals]]

New topics worth writing: Singular Value Decomposition, Intrinsic dimension of objective landscapes, Adapter layers (Houlsby), Prefix-tuning and prompt-tuning, QLoRA and 4-bit quantised adaptation, Grassmann distance and subspace similarity, BitFit, Parameter-efficient fine-tuning (PEFT) survey, Multi-LoRA serving
