---
title: "FreeFlow: A Bias-free Hierarchical Transformer for Optical Flow Estimation"
authors: ["Vladislav Bargatin", "Alexander Yakovenko", "Khaled Abud", "Dmitriy Vatolin"]
year: 2026
arxiv: "2609.11486"
url: https://arxiv.org/abs/2609.11486
priority: Good-To-Read
read_on: 2026-09-17
tags: [paper, transformers, vision, theory]
---
## The Core Idea

Optical flow means: for every pixel in frame 1, where did it move to in frame 2? A dense per-pixel motion field.

For twenty years, the field has solved this by baking the physics of matching straight into the architecture. The standard parts list:

- **Correlation (cost) volume** — explicitly compute the dot product of every pixel's feature in frame 1 against every candidate pixel in frame 2, then look up "how well does this match?" as a tensor.
- **Warping** — take frame 2's features, bend them by the current flow guess, and compare to frame 1. If the guess is right, they line up.
- **Iterative refinement** — a small recurrent unit (a GRU) that nudges the flow estimate 8–32 times, RAFT-style.
- **Convex upsampling** — a learned, flow-aware way to go from 1/8 resolution back to full resolution.

FreeFlow throws away **all four**. What is left is a plain transformer: two shared-weight encoders, a decoder that alternates self-attention and cross-attention between the two views, and a three-layer convolution head. One forward pass. No recurrence. And it wins — Sintel Clean EPE 0.68 (previous best 0.79), Sintel Final 1.48 (previous best 1.65), KITTI-15 Fl-all 3.23, Spring 1px 3.192.

Why did nobody do this earlier? Two reasons, and the paper's real contribution is the second.

1. **Data.** A bias-free model must learn matching from supervision instead of getting it for free from architecture. That needs a pretraining task which teaches dense correspondence. Cross-view completion (mask most of one image, reconstruct it using the other) does exactly this, and it only became available recently via CroCo.
2. **Resolution.** Flow needs high resolution — thin structures, sharp motion boundaries, large displacements at 1080p. Full self-attention over a 1080p image at 8×8 patches is 32,400 tokens; the attention matrix is a billion entries. Everyone else dodged this by **tiling** (cut the image into pieces, run each separately, average the seams) or by **late fusion** (process scales independently, merge only in the decoder). Both starve the model of information flow *across* tiles during feature extraction, so big motions that span a tile boundary are simply not representable.

The unlock is what they call **Dense Feature Fusion**: every layer mixes local, cross-tile, and global information, so the coarse and fine signals reinforce each other all the way through the network, not just at the end.

> [!NOTE] Inductive bias
> Structure you hard-wire into an architecture because you know something about the task. A correlation volume *is* the belief "flow is about local feature similarity", made into a tensor. It saves data and compute, but it also caps what the model can express, and it is expensive engineering to make fast. Compare [[An Image is Worth 16x16 Words (ViT)#^inductive-bias|ViT dropping convolution's locality prior]]. ^flow-inductive-bias

The broader claim is a small [[The Bitter Lesson (essay)#^bitter-lesson|Bitter Lesson]] datapoint: hand-crafted task structure got beaten by a generic architecture plus the right pretraining, in a domain where the hand-crafted structure was genuinely good.

## The Methodology

**Shape of the thing.** Two images $I_1, I_2 \in \mathbb{R}^{H \times W \times 3}$. Patchify both with $P = 8$ (note: not 16 — this matters, see ablations) into $N = \frac{H}{8} \cdot \frac{W}{8}$ tokens of width $D$. Run both through a **Siamese** (shared-weight) encoder:

$$F^1 = \text{Encoder}(X_1), \qquad F^2 = \text{Encoder}(X_2)$$

Then a decoder whose blocks do self-attention over the current tokens *plus* cross-attention with $F^1$ as [[Query, Key, and Value (QKV)|queries]] and $F^2$ as keys/values:

$$Z = \text{Decoder}(F^1, F^2)$$

Reshape $Z$ to $\frac{H}{8} \times \frac{W}{8} \times D$, then one head maps it to $\mathbb{R}^{H \times W \times 5}$. This is the DUSt3R / CroCo / MASt3R skeleton, nothing exotic.

**The three attention blocks.** Every encoder and decoder layer runs these in a fixed order — Win, then Swin, then Global:

1. **Window (Win).** Split the token map into a $4 \times 4$ grid of 16 non-overlapping windows, each $\frac{H}{32} \times \frac{W}{32}$ tokens. Attend inside each window only. Cheap, gives fine local detail.
2. **Shifted-Window (Swin).** Shift the whole token map by half a window — $\lfloor W/64 \rfloor$ horizontally, $\lfloor H/64 \rfloor$ vertically — repartition into the same $4 \times 4$ grid, attend, shift back. Tokens that were on opposite sides of a boundary are now in one window, so information crosses tile borders. Straight from Swin Transformer.
3. **Global.** Downsample $2\times$ with a stride-2 convolution, do *full* attention at the reduced resolution, upsample $2\times$ with a stride-2 transposed convolution. Normalisation goes *after* the residual connection here.

The arithmetic that makes this work: attention is quadratic in token count, so halving each spatial dimension cuts tokens by $4\times$ and attention cost by $16\times$. That puts the Global block's cost in the same ballpark as a Win block at full resolution. All three are ordinary attention ops, so they run on off-the-shelf kernels like [[FlashAttention- Fast and Memory-Efficient Exact Attention|FlashAttention]] — unlike correlation-volume indexing or warping, which need bespoke CUDA to be fast.

Important decoder detail: the shifted-window partition is shifted *identically in both frames*, so corresponding regions stay co-located across the pair. Otherwise cross-attention would be comparing misaligned chunks.

**Position.** [[RoFormer- Enhanced Transformer with Rotary Position Embedding|RoPE]], no learned position table.

**Attention scale factor.** Instead of the usual $1/\sqrt{D}$, they multiply logits by $\log(N)/\sqrt{D}$:

$$\text{Attn}(Q,K,V) = \text{Softmax}\!\left(\frac{\log(H/8 \times W/8)}{\sqrt{D}} \, QK^\top\right) V$$

Softmax over more tokens spreads probability thinner; scaling logits up with $\log N$ keeps the distribution comparably peaked, so a model trained on small crops still behaves sanely at 1080p. This is a [[Train Short, Test Long (ALiBi)#^length-extrapolation|length-extrapolation]] fix in image form. Prior work normalises the factor to equal 1 at the training token count; FreeFlow uses the raw unnormalised version and reports it works fine.

**The head.** Three layers, that is it. $3\times3$ conv to $4D$ channels → $1\times1$ conv to 4096 channels → transposed conv with kernel and stride 8 to get back to $H \times W$. Output is 5 channels: 2 for flow, 3 for the uncertainty parameters. Flow channels get multiplied by 8 (the patch size) to land in pixel units. No DPT head, no multi-layer feature aggregation, no convex upsampling.

**Loss.** Mixture-of-Laplace from SEA-RAFT. Per flow coordinate:

$$\text{MixLap}(\mu_{gt}; \alpha,\beta,\mu) = -\log\!\left(\frac{\alpha}{2} e^{-|\mu_{gt}-\mu|} + \frac{1-\alpha}{2e^{\beta}} e^{-\frac{|\mu_{gt}-\mu|}{e^{\beta}}}\right)$$

Two Laplace distributions: a sharp one (scale 1) and a wide one (scale $e^\beta$, $\beta$ clamped to $[0,10]$), mixed by $\alpha$, which comes from a softmax over two head outputs. The model predicts its own error scale. When a pixel is occluded or ambiguous, it can put mass on the wide component and stop being punished for a large residual — the [[Uncertainty#^aleatoric|aleatoric uncertainty]] escape hatch that stops occlusions from dominating the [[Backpropagation#^what-backprop-computes|gradient]]. Averaged over both $x$ and $y$ and all valid pixels.

**Training, in stages.**

| Stage | Data | Crop / budget | LR | WD | Batch | Steps |
|---|---|---|---|---|---|---|
| Pretrain (cross-view completion) | ARKitScenes + MegaDepth + 3DStreetView, 3.7M pairs | 224×224 | 8e-4 | 5e-2 | 2048 | 346k |
| TaTSKH | TartanAir .23 / Sintel .25 / Things .24 / KITTI .09 / HD1K .19 | 10,880 tokens | 4e-5 | 1e-2 | 32 | 450k |
| TaTSKH-hq | same | 32,640 tokens | 1e-5 | 1e-5 | 32 | 90k |
| Sintel-ft | Sintel | 872×2048 | 1e-5 | 1e-5 | 32 | 12.5k |
| KITTI-ft | KITTI | 750×2484 | 1e-5 | 1e-5 | 32 | 2.5k |
| Spring-ft | Spring | 1080×1920 | 1e-5 | 1e-5 | 32 | 60k |

[[Decoupled Weight Decay Regularization (AdamW)|AdamW]] with $\beta_1 = 0.9$, $\beta_2 = 0.95$. Linear warmup then cosine decay for pretraining; linear warmup then linear decay for flow fine-tuning.

> [!NOTE] Cross-view completion
> The [[BERT- Pre-training of Deep Bidirectional Transformers#^masked-language-model|masked-LM]] idea for image pairs. Mask most patches of view 1 and reconstruct them *conditioned on view 2*, which shows the same scene from another angle. You cannot fill the hole from local texture — you must find the matching region in the other view. So the pretext task directly teaches dense correspondence, which is the entire downstream problem. ^cross-view-completion

One deviation from CroCo: CroCo drops masked tokens from the encoder and only inserts the learned $e_{\text{mask}}$ token at the *decoder* input. FreeFlow is hierarchical and uses windowed attention, so token positions must stay on a fixed grid — they insert $e_{\text{mask}}$ at the **encoder** input instead, objective otherwise unchanged.

**Variable-resolution training.** Rather than a fixed crop, they fix a *token budget* per minibatch. Per sample: pick one dimension at random, sample its value, then set the other dimension as large as the budget permits. Result is rectangular crops of many aspect ratios with bounded compute. Batch size is 1 sample per GPU, so the implementation is trivial. Frames are upsampled $2\times$ during fine-tuning (from MEMFOF) to match the motion magnitude distribution you actually see at FullHD.

Three sizes, co-scaling depth, width and heads per standard ViT practice:

| | Width (enc+dec) | Layers (enc+dec) | Heads | Params |
|---|---|---|---|---|
| FreeFlow-S | 256+256 | 12+24 | 4+4 | 35M |
| FreeFlow-M | 384+384 | 18+36 | 6+6 | 102M |
| FreeFlow-L | 512+512 | 24+48 | 8+8 | 231M |

Largest model: 4–5 days pretraining plus ~3 days fine-tuning on 32 GPUs.

## Ablation Studies and Experiments

**Spring** (1080p, real-world, 1px outlier rate is the headline metric; memory and time measured on an RTX 3090 with AMP):

| Method | Mem (GB) | Time (ms) | Params | 1px ↓ | EPE ↓ | Fl ↓ | WAUC ↑ |
|---|---|---|---|---|---|---|---|
| RAFT | 7.97 | 406 | 5.3M | 6.790 | 1.476 | 3.198 | 90.92 |
| FlowFormer | 1.90 | ~2084 | 16.2M | 6.510 | 0.723 | 2.384 | 91.68 |
| SEA-RAFT (M) | 8.12 | 198 | 19.7M | 3.686 | 0.363 | 1.347 | 94.53 |
| CroCo-Flow | 2.73 | 3266 | 447M | 4.565 | 0.498 | 1.508 | 93.66 |
| Win-Win | ~3.82 | ~305 | 230M | 5.371 | 0.475 | 1.621 | 92.72 |
| WAFT-DINOv3-a2 | 18.96 | 408 | 56.5M | **3.182** | 0.325 | 1.246 | 95.05 |
| MEMFOF (multi-frame) | 1.90 | 262 | 75.8M | 3.289 | 0.355 | 1.238 | 95.19 |
| ARFlow (multi-frame) | — | — | 76.5M | 3.265 | 0.353 | 1.212 | **95.28** |
| **FreeFlow-S** | **1.02** | 144 | 34.6M | 5.087 | 0.533 | 1.452 | 90.20 |
| **FreeFlow-M** | 1.66 | 325 | 102M | 3.392 | 0.346 | 1.171 | 94.92 |
| **FreeFlow-L** | 2.58 | 231 | 231M | 3.192 | **0.278** | **1.048** | 95.24 |

Best EPE and best Fl outright; second on 1px by 0.01. Against WAFT-DAv2-a2 that is 9% better EPE and 14% lower Fl — at **one eighth the inference memory** (2.58 GB vs 20.58 GB). CroCo-Flow needs 3.3 seconds per 1080p frame because of dense tiling; FreeFlow-L needs 231 ms because it runs 1080p natively.

**Sintel and KITTI-15:**

| Method | Sintel Clean ↓ | Sintel Final ↓ | KITTI Fl-all ↓ |
|---|---|---|---|
| RAFT | 1.61 | 2.86 | 5.10 |
| FlowFormer++ | 1.07 | 1.94 | 4.52 |
| SEA-RAFT (L) | 1.31 | 2.60 | 4.30 |
| CroCo-Flow | 1.09 | 2.44 | 3.64 |
| WAFT-DAv2-a2 | 0.94 | 2.33 | 3.31 |
| GeoViT | 0.79 | 1.88 | 3.79 |
| VideoFlow-MOF (multi-frame) | 0.99 | 1.65 | 3.65 |
| ARFlow (multi-frame) | 0.96 | 1.79 | 2.85 |
| **FreeFlow-S** | 1.03 | 1.99 | 4.06 |
| **FreeFlow-M** | 0.80 | 1.77 | 3.33 |
| **FreeFlow-L** | **0.68** | **1.48** | 3.23 |

FreeFlow-L is first on both Sintel splits. On Final it beats VideoFlow-MOF, which gets to see *five* frames, using only two. On KITTI it beats every non-stereo two-frame method; only multi-frame ARFlow (2.85) is ahead. FreeFlow-M at 102M params already beats GeoViT at 377M on Clean.

**Ablation 1 — which attention block does the work?** (scaled-down config: 8+8 layers, width 256, 8 heads; Spring sub-validation on scenes 0045/0047)

| Mask ratio | Win | Swin | Global | 1px ↓ | EPE ↓ |
|---|---|---|---|---|---|
| 0.9 | ✗ | ✗ | ✓ | 1.133 | 0.229 |
| 0.9 | ✓ | ✗ | ✓ | 0.801 | 0.190 |
| 0.9 | ✓ | ✓ | ✗ | 0.659 | 0.167 |
| 0.9 | ✓ | ✓ | ✓ | 0.688 | 0.170 |
| 0.925 | ✓ | ✓ | ✓ | 0.685 | 0.166 |
| 0.95 | ✓ | ✓ | ✗ | 0.658 | 0.166 |
| **0.95** | **✓** | **✓** | **✓** | **0.624** | **0.157** |
| 0.975 | ✓ | ✓ | ✓ | 0.656 | 0.174 |

Three readings:

- **Global attention alone is terrible** (1px 1.133). Downsampled global context cannot resolve fine motion. Local processing is doing most of the heavy lifting.
- **Swin is the one you cannot remove.** Dropping it (row 2) costs 0.801 vs 0.688 — a clear, consistent hit. This is the mechanism that gets information across tile borders, and it is the architectural core of the paper.
- **Global's value depends on the masking ratio.** At 0.9, removing Global actually *helps* slightly (0.659 vs 0.688). At 0.95, removing it hurts (0.658 vs 0.624). So pretraining difficulty and architecture interact — the authors' own conclusion is that masking ratio is a hyperparameter you must tune *jointly* with the model. Worth noting for anyone who assumes pretraining recipes transfer.
- 0.975 is too hard and everything degrades. 0.95 is the sweet spot, 9.3% better 1px than CroCo's default 0.9. Their explanation: FreeFlow uses 8×8 patches where CroCo uses 16×16, so any masked patch is physically closer to visible content, and the task needs more masking to stay hard.

Critically, the **qualitative** picture disagrees with the metrics on Global attention. Even where sub-val numbers barely move, removing Global produces visible large-scale motion inconsistencies. A reminder that averaged pixel metrics hide structured failures.

**Ablation 2 — is it the architecture or just the training recipe?** They swapped the FreeFlow blocks for vanilla ViT blocks, keeping everything else (this is essentially CroCo-Flow / Win-Win minus the DPT head):

| Type | Layers | Time (ms) | Params | 1px ↓ | EPE ↓ |
|---|---|---|---|---|---|
| ViT | 4 | 278+13 | 15.4M | 1.021 | 0.217 |
| ViT | 6 | 415+13 | 19.1M | 0.809 | 0.216 |
| ViT | 12 | 829+13 | 30.1M | 0.698 | 0.192 |
| FreeFlow-S | 4 | 131+13 | 34.6M | 0.709 | 0.181 |
| FreeFlow-S+ | 8 | 256+13 | 60.9M | 0.624 | 0.157 |

A 12-layer ViT at 829 ms roughly matches a 4-layer FreeFlow at 131 ms — **6× slower for equal or worse quality**. So the local–global design, not the CroCo pretraining plus training recipe alone, is what buys the quality-per-millisecond.

**Ablation 3 — put the biases back in.** They bolted GeoViT-style iterative warping (with GeoViT's ConvGRU) onto FreeFlow during fine-tuning only:

| Config | Iters | Time (ms) | Params | 1px ↓ | EPE ↓ |
|---|---|---|---|---|---|
| FreeFlow-S, 4 layers | — | 131+13 | 34.6M | 0.709 | 0.181 |
| + warping, 4 layers | 1 | 131+40 | 33.1M | 0.660 | 0.165 |
| + warping, 2 layers | 2 | 131+62 | 20.0M | 0.719 | 0.171 |
| + warping, 4 layers | 2 | 262+53 | 33.1M | **0.583** | **0.148** |

This is the honest bit and it does not support the title's strongest reading. **Adding the bias back makes it better** — 0.583 vs 0.709 at matched parameters. The authors' framing: the inductive bias offloads task knowledge into an explicit operator, raising effective capacity. Their claim narrows to "biases are not *necessary* for SOTA", which is true, rather than "biases are useless", which is false. You buy generality and speed with the bias-free design; you pay some accuracy.

**Ablation 4 — patch size.** FreeFlow-S (8×8, width 256) versus a 16×16 variant widened to 768 and 12 heads to match inference time:

| Patch | Width | Mem (GB) | Time (ms) | Params | 1px ↓ | EPE ↓ |
|---|---|---|---|---|---|---|
| 8×8 | 256 | 1.02 | 144 | 34.6M | 0.709 | 0.181 |
| 16×16 | 768 | 2.91 | 120 | 278.9M | 0.798 | 0.174 |

16×16 wins EPE by a hair, loses 1px, and costs **8× the parameters and 3× the memory**. Small patches win on cost.

**Ablation 5 — late fusion (the thing this paper is arguing against).** They built "CroCo-Pro", a DepthPro-style late-fusion baseline: CroCo encoder–decoder backbone, multi-scale pyramid features merged in a DPT decoder. It produces sharp local detail but **globally incoherent** flow. The model leans on the finest-scale tiles and under-uses coarse features, because with fusion happening only at the end there is no path for coarse context to influence full-resolution features. This is the empirical justification for Dense Feature Fusion.

**What did not work — zero-shot.** The weakest result in the paper. Fine-tuning only on TartanAir + FlyingThings3D and evaluating on Sintel/KITTI training sets:

| Method | Sintel Clean | Sintel Final | KITTI Fl-all |
|---|---|---|---|
| GeoViT (Kinetics-400 pretrain, 59M frames) | 0.69 | 1.78 | 11.5 |
| WAFT-DINOv3-a2 (LVD-1689M, 1.7B frames) | 1.28 | 2.56 | 12.9 |
| CroCo-Flow | 1.28 | 2.58 | — |
| FreeFlow-S | 0.91 | 3.16 | 10.4 |
| FreeFlow-M | 1.01 | 3.12 | 14.6 |
| FreeFlow-L | 1.04 | 2.30 | 12.9 |

**Bigger is worse** on Sintel Clean (S 0.91 → L 1.04) and KITTI (S 10.4 → M 14.6). Scaling stops helping and sometimes reverses. The authors attribute it to a data-limited regime: with no flow-specific bias, transfer is bounded by how much motion variety the training signal covers, and extra capacity just fits the training distribution harder. This is the exact price of removing inductive bias — it does not show up on in-distribution benchmarks, only here.

A useful side finding from the same table: WAFT with 1.7 *billion* frames of **monocular** pretraining transfers worse than GeoViT with 59M frames of **video**. Scale of pretraining does not substitute for the right pretraining signal; motion and multi-view data matter for a motion task.

## Worth Remembering

- **The bias-free claim is weaker than the title.** Their own Table 8 shows adding GeoViT warping back improves 1px from 0.709 to 0.583. The defensible claim is: biases are optional, and dropping them buys speed, memory, generality and clean scaling.
- **Memory is the quiet headline.** FreeFlow-L: 2.58 GB at 1080p versus WAFT's 18.96–20.58 GB for comparable accuracy. That is the difference between running on a consumer GPU and not. It comes from never materialising a correlation volume, which is quadratic in pixels.
- **Everything is standard attention.** No custom kernels. FreeFlow inherits every optimisation the transformer ecosystem produces for free — FlashAttention, better compilers, newer hardware paths. Correlation-volume indexing and warping each need their own hand-tuned CUDA to be competitive. This compounding advantage is arguably worth more long-term than the benchmark numbers.
- **Masking ratio and architecture are coupled**, and the coupling flips signs. At 0.9 masking, Global attention is neutral-to-harmful; at 0.95 it is essential. Do not inherit a pretraining hyperparameter from a paper with a different patch size.
- **Metrics missed a real failure.** Removing Global attention barely moves sub-val 1px/EPE but produces obviously wrong flow in the pictures. Pixel-averaged errors are dominated by the many easy pixels; structured, low-frequency errors hide in them. A close cousin of [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|the evaluation-methodology worries]] in other fields — the aggregate number is not the model.
- **Scaling is clean in-distribution, broken out-of-distribution.** S→M→L improves monotonically on Sintel/KITTI/Spring but is non-monotonic zero-shot. If you care about deployment on unseen domains, benchmark scaling curves are lying to you.
- **Uncertainty is free and load-bearing.** The 5-channel head predicts $\alpha, \beta$ alongside flow. The mixture-of-Laplace loss lets the model declare "I cannot see this pixel" on occlusions instead of receiving a huge gradient. If you are building anything downstream, that confidence field is usable and it costs 3 channels.
- **Variable-resolution training with a token budget** is a generally reusable trick, not flow-specific. Fixed crops waste compute on padding and narrow the motion distribution the model ever sees. Fixing the token count instead and sampling aspect ratios costs nothing and is trivial at batch-size-1-per-GPU.
- **Open questions.** Does the zero-shot collapse go away with a Kinetics-scale video pretraining corpus, given GeoViT's result suggests it might? Does the architecture transfer to stereo, depth, and point tracking — all the same dense-correspondence family — since nothing in it is about flow? And what happens with the warping variant scaled to L, which they did not try?

## Links

Related: [[An Image is Worth 16x16 Words (ViT)]] · [[Attention Is All You Need]] · [[Query, Key, and Value (QKV)]] · [[RoFormer- Enhanced Transformer with Rotary Position Embedding]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Flash Attention]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Self-Supervised Learning from Images with I-JEPA]] · [[Train Short, Test Long (ALiBi)]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Uncertainty]] · [[The Bitter Lesson (essay)]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Layer Normalization]] · [[Accurate, Large Minibatch SGD- Training ImageNet in 1 Hour]] · [[Foundation Models]]

New topics worth writing: Optical flow estimation, Swin Transformer and shifted-window attention, Correlation/cost volumes, RAFT and iterative refinement, Cross-view completion pretraining (CroCo), DUSt3R and binocular dense prediction, Siamese encoders, Mixture-of-Laplace loss and heteroscedastic regression, Dense Prediction Transformer (DPT) heads, Hierarchical vision backbones (Hiera, DepthPro), Zero-shot transfer vs inductive bias trade-off
