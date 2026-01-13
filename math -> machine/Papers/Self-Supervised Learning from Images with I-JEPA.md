---
title: "Self-Supervised Learning from Images with I-JEPA"
authors: ["Mahmoud Assran", "Quentin Duval", "Ishan Misra", "Piotr Bojanowski", "Pascal Vincent", "Michael Rabbat", "Yann LeCun", "Nicolas Ballas"]
year: 2023
arxiv: "2301.08243"
url: https://arxiv.org/abs/2301.08243
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, transformers, optimization, self-supervised, vision]
---
## The Core Idea

Two families dominated self-supervised vision before this paper, and both had a flaw.

**Family 1 — invariance methods** (SimCLR, DINO, BYOL, iBOT). Take one image, make two crops, jitter the colours, force the encoder to output the same vector for both. This works well but the "same-ness" is hand-designed. You are telling the model *by hand* that colour does not matter, that scale does not matter. Those biases are wrong for some tasks — counting objects and estimating depth genuinely depend on scale and position. And you cannot port "random colour jitter" to audio.

**Family 2 — masked reconstruction** (MAE, BEiT). Hide patches, rebuild the pixels. No hand-crafted priors, ports to any modality. But you get low-level features. MAE's frozen features get 77.2% on ImageNet linear probe after 1600 epochs; DINO gets 80.1% in 300.

I-JEPA sits between them. Keep the masking (no hand-crafted augmentations), but **do not predict pixels — predict the embeddings of the hidden patches**, where the embeddings come from a slowly-updated copy of the encoder itself.

Why that matters: pixels contain enormous amounts of detail nobody needs. The exact grain of the grass, the JPEG noise, the shade of the sky. A pixel-reconstruction loss spends capacity on all of it. If your target is instead a *learned representation*, the target itself is free to throw that detail away. The model only has to predict what is predictable.

> [!NOTE] Joint-Embedding Predictive Architecture (JEPA) ^jepa
> Encode $x$, encode $y$, then predict $y$'s **embedding** from $x$'s embedding, conditioned on a variable $z$ that says *which part of $y$* you are predicting. The loss lives in embedding space, not pixel space. Contrast with a joint-embedding architecture (match two embeddings, no predictor) and a generative architecture (rebuild the raw signal).

The catch that makes this non-trivial: if both sides are learned, the model can cheat by outputting a constant vector everywhere. Loss zero, representation useless. This is representation collapse. I-JEPA avoids it with an asymmetry — the target encoder is not trained by gradient descent at all, it is an [exponential moving average] of the context encoder (the same trick BYOL and DINO use).

The unlock: a ViT-Huge/14 trained on 16 A100s in under 72 hours (<1200 GPU-hours) that beats MAE on linear probing, beats DINO/iBOT on depth prediction and counting, and needs no augmentation pipeline at all.

## The Methodology

Three networks, all [[An Image is Worth 16x16 Words (ViT)|Vision Transformers]].

- **Context encoder** $f_\theta$ — a standard ViT. Trained by gradients.
- **Target encoder** $f_{\bar\theta}$ — identical architecture, weights are an EMA of $\theta$. No gradients flow through it.
- **Predictor** $g_\phi$ — a *narrow* ViT. Embedding dim fixed at 384 regardless of encoder size; depth 6 for ViT-B, 12 for ViT-L/H, 16 for ViT-G. Trained by gradients.

No `[cls]` token anywhere. Global representations come from average-pooling patch outputs.

**Step 1 — targets.** Feed the **whole, unmasked** image $y$ through the target encoder. Get patch representations $s_y = \{s_{y_1}, \dots, s_{y_N}\}$. *Then* sample $M = 4$ blocks from these outputs. Scale in $(0.15, 0.2)$ of the image, aspect ratio in $(0.75, 1.5)$. Blocks may overlap.

This ordering is the point. You mask the **output** of the target encoder, not its input. Each target patch representation therefore has seen the entire image through self-attention, so it carries context, so it is semantic. Masking the input instead would give you four isolated crops encoded in isolation.

**Step 2 — context.** Sample one block $x$, scale $(0.85, 1.0)$, square. Then **delete any patches that overlap any of the four target blocks**. What survives is a large but Swiss-cheesed region — spatially spread out (informative) but sparse (cheap to encode). Feed only these visible patches through $f_\theta$, MAE-style.

**Step 3 — predict.** For each target block $i$, run the predictor on the context output plus one mask token per patch you want:
$$\hat{s}_y(i) = g_\phi\!\left(s_x, \{m_j\}_{j \in B_i}\right)$$
Mask tokens are a single shared learnable vector plus a positional embedding. The positional embedding is the entire conditioning signal — it is the $z$ in the JEPA diagram, telling the predictor *where* it is being asked to hallucinate. The predictor is run $M=4$ separate times, once per block.

**Step 4 — loss.** Plain average $L_2$:
$$\frac{1}{M}\sum_{i=1}^{M} \sum_{j \in B_i} \lVert \hat{s}_{y_j} - s_{y_j} \rVert_2^2$$

Gradients update $\theta$ and $\phi$. Then $\bar\theta \leftarrow \tau \bar\theta + (1-\tau)\theta$.

**Hyperparameters that mattered.**
- Optimiser [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], batch 2048.
- LR warmed $10^{-4} \to 10^{-3}$ over 15 epochs, cosine decay to $10^{-6}$.
- Weight decay ramped linearly $0.04 \to 0.4$ across training.
- EMA momentum $0.996 \to 1.0$ linearly. At the end the target encoder is frozen.
- Masks are sampled in the data-loader collator. All context masks on one GPU are forced to the same size so batching works; same for target masks.
- No mixup, no cutmix, no random erasing, no drop-path during pretraining.

## Ablation Studies and Experiments

**ImageNet-1K linear probe** (freeze encoder, train linear classifier on full labels):

| Method | Arch | Epochs | Top-1 |
|---|---|---|---|
| MAE | ViT-H/14 | 1600 | 77.2 |
| data2vec | ViT-L/16 | 1600 | 77.3 |
| CAE | ViT-L/16 | 1600 | 78.1 |
| **I-JEPA** | ViT-H/14 | **300** | **79.3** |
| **I-JEPA** | ViT-H/16₄₄₈ | 300 | **81.1** |
| DINO (uses augs) | ViT-B/8 | 300 | 80.1 |
| iBOT (uses augs) | ViT-L/16 | 250 | 81.0 |

So the augmentation-free method finally caught the augmentation-based ones — but only at 448px resolution and huge scale.

**ImageNet-1% semi-supervised** (~13 labelled images per class): MAE ViT-H/14 gets 71.5 after 1600 epochs; I-JEPA ViT-H/14 gets 73.3 after 300. At 448px, 77.3 — beating MSN's 75.7, which does use augmentations.

**Transfer, linear probe**: CIFAR100 87.5 vs MAE's 77.3 and DINO's 84.9. iNat18 47.6 vs MAE's 32.9 — but iBOT still wins at 57.3. Fine-grained species classification remains the weak spot.

**Low-level tasks (Clevr)** — this is the most interesting result. Depth prediction (Clevr/Dist): I-JEPA 72.4, DINO **53.4**, iBOT 62.8. Object counting: I-JEPA 86.7, DINO 86.6, MAE 90.5. The view-invariance methods have been trained to *ignore* scale and position, so they are crippled on tasks that need exactly that information. I-JEPA has no such blind spot. MAE, the most low-level method, still wins at counting.

**Compute**: per-iteration, I-JEPA is ~7% *slower* than MAE (you pay for the target encoder forward pass). But it converges in roughly $5\times$ fewer iterations, so it wins overall. ViT-H/14 with I-JEPA costs less than ViT-S/16 with iBOT.

### The ablations that matter

**Predicting in representation space vs pixel space** — the whole thesis, one table:

| Target | Arch | Epochs | IN-1% Top-1 |
|---|---|---|---|
| Target-encoder output | ViT-L/16 | 500 | **66.9** |
| Pixels | ViT-L/16 | 800 | **40.7** |

A 26-point collapse. Same architecture, same masking, just swap the target. This is the single most convincing number in the paper.

**Masking the output vs the input of the target encoder** — ViT-H/16, 300 epochs, IN-1%: output masking 67.3, input masking 56.1. Confirms that target patches must be contextualised by full-image attention.

**Masking strategy** (ViT-B/16, 300 epochs, IN-1%):

| Strategy | Top-1 |
|---|---|
| multi-block (4 targets @ 0.15–0.2, context 0.85–1.0 minus overlap) | **54.2** |
| block (one target @ 0.6, context = complement) | 20.2 |
| random patches (0.6) | 17.6 |
| rasterized (1 quadrant → 3 quadrants) | **15.5** |

The masking scheme is doing enormous work. Standard MAE-style random masking gives 17.6 vs 54.2. Note the ordering: the quadrant scheme, which is the most "natural" spatial split, is the *worst*.

**Target block scale** — sweet spot is narrow:

| Target scale | Top-1 |
|---|---|
| (0.075, 0.2) | 19.2 |
| (0.1, 0.2) | 39.2 |
| (0.125, 0.2) | 42.4 |
| **(0.15, 0.2)** | **54.2** |
| (0.2, 0.25) | 38.9 |
| (0.2, 0.3) | 33.6 |

Too small → targets are texture patches, not objects. Too large → after removing overlap, the context is starved. This is a genuinely fragile hyperparameter.

**Number of targets**: 1 → 9.0, 2 → 22.0, 3 → 48.5, 4 → 54.2. With a single target the task is nearly degenerate.

**Context scale**: (0.4,1.0) → 31.2, (0.65,1.0) → 47.1, (0.75,1.0) → 49.3, (0.85,1.0) → 54.2. Monotone. Bigger informative context is always better.

**Predictor depth**: ViT-L/16, 500 epochs — depth 6 gives 64.0, depth 12 gives 66.9.

**Predictor width**: a bottleneck helps. ViT-L encoder is 1024-wide; a 384-wide predictor gives 70.7 on IN-1%, a 1024-wide predictor gives 68.4. Forcing the predictor through a narrow channel stops it from memorising the target space.

### What did not work / trade-offs

- **Pixel targets.** Catastrophic, as above.
- **Rasterized quadrant masking.** Worse than random patches.
- **Widening the predictor to match the encoder.** Hurt by 2.3 points.
- **Weight decay is a genuine trade-off, not a free win.** Ramping $0.04 \to 0.4$ gives 77.8 linear-eval / 69.4 fine-tuned IN-1%. A fixed $0.05$ gives 76.4 linear / **70.7** fine-tuned. You cannot have both. They chose the linear-eval-friendly setting for the paper.
- **ViT-G/16 on IN-22K** improved semantic tasks (iNat18 50.5 → 55.3) but *hurt* low-level ones (Clevr/Count 88.6 → 86.7, Clevr/Dist 75.0 → 73.0) versus ViT-H/14. The bigger patch size (16 vs 14) coarsens spatial resolution and that costs you on dense tasks. Scale is not uniformly good here.
- **Full ImageNet fine-tuning is a tie, not a win.** I-JEPA ViT-H/16₄₄₈ gets 87.1 vs MAE ViT-H/14₄₄₈ at 87.8 — I-JEPA loses, though with $5.3\times$ fewer epochs. The advantage of I-JEPA is in *frozen* features, not in fine-tuned ceilings.

## Worth Remembering

**The RCDM visualisations are the best qualitative evidence.** They train a diffusion decoder to map a frozen representation back to pixels, then sample it many times. What is *constant* across samples is what the representation encodes; what *varies* is what it discarded. Feeding the predictor's output for a masked region, the samples consistently show a bird's back or a car roof in the right pose — but with varying fur texture and totally different backgrounds. So the predictor genuinely learned "there is an object part here, at this pose", and genuinely threw away background and texture.

The comparison against MSN (Figure 8) is sharper: MSN samples vary in object pose, object scale, and *number of instances*. Invariance training destroyed that information on purpose. I-JEPA kept it. This is exactly why I-JEPA wins on Clevr/Dist by 19 points.

**Caveats for using it.**
- The masking hyperparameters are the method. Target scale $(0.15, 0.2)$ is not a detail — move it to $(0.2, 0.3)$ and you lose 20 points. Anyone porting JEPA to a new modality has to re-derive the equivalent of "sufficiently large, sufficiently many, sufficiently informative context".
- Collapse prevention rests entirely on the EMA asymmetry, with no explicit regulariser. There is no theoretical guarantee here — it is the same empirically-load-bearing trick as BYOL. LeJEPA later attacks exactly this gap by adding a provable isotropic-Gaussian regulariser.
- No `[cls]` token means you must average-pool, and the eval recipe had to be adapted from VISSL defaults. Comparisons to `[cls]`-based methods carry a small asterisk.
- Only 4 mask tokens' worth of prediction per image, but the predictor runs 4 separate forward passes. That is 4× predictor cost per step, hidden by the predictor being narrow.

**Open questions.** Does the EMA target encoder actually converge to something stable, or is it a slowly-drifting target the context encoder chases forever? Why does a *narrow* predictor help — is it an information bottleneck argument, or just regularisation? And does the compute advantage survive at data scales beyond IN-22K, where MAE's simpler objective might catch up?

**Lineage.** This is LeCun's "A Path Towards Autonomous Machine Intelligence" position paper made concrete for one modality. The line continues through V-JEPA (video) and LeJEPA.

## Links
Related: [[An Image is Worth 16x16 Words (ViT)]] · [[A Simple Framework for Contrastive Learning (SimCLR)]] · [[LeJEPA- Provable and Scalable Self-Supervised Learning]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Understanding Contrastive Learning through Alignment and Uniformity]] · [[Attention Is All You Need]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Denoising Diffusion Probabilistic Models]] · [[Shortcut Learning in Deep Neural Networks]] · [[Distillation]] · [[Foundation Models]]

New topics worth writing: Masked Autoencoders (MAE), data2vec, DINO and self-distillation with no labels, BYOL and EMA target networks, Energy-Based Models, representation collapse, Barlow Twins / VICReg redundancy reduction, RCDM representation visualisation, V-JEPA
