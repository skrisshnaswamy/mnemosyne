---
title: "Video DeltaNet: A Video-Native Hybrid Attention for Livestream Video Generation"
authors: ["Haocheng Xi", "Yiming Xie", "Hexu Zhao", "Yiwen Zhang", "Michael Liu", "Thomas Creavin", "Kurt Keutzer", "Xiuyu Li", "Zhaoyang Lv", "Chenfeng Xu", "Haiwen Feng"]
year: 2026
arxiv: "2609.20744"
url: https://arxiv.org/abs/2609.20744
priority: Good-To-Read
read_on: 2026-09-27
tags: [paper, transformers, llm, diffusion, vision, theory]
---
## The Core Idea

Video diffusion models spend most of their time on attention. In the MiniMax H3 model the authors profile, softmax attention is **over 85% of the denoiser's runtime**. The reason is the usual one: attention compares every token to every other token, so cost grows with the square of sequence length. A 14.3-second 768p video is 102 latent frames × 48 × 84 tokens ≈ 411k video tokens. That is a very big square.

The obvious fix is [[Linear Attention|linear attention]] — squash all past context into a fixed-size memory matrix $S$, so cost grows linearly. Language models do this. But swapping it in wholesale for video loses quality, because a fixed-size state cannot hold the fine detail that [[Attention|softmax]] keeps.

Video DeltaNet (VDN) is a **hybrid**: keep exact softmax for the interactions that need detail, use linear memory for everything far away. Two specific ideas make it work.

**1. Split attention by *temporal role*, not by a generic sparsity pattern.** Nearby frames get real token-to-token softmax (texture, edges, short motion need it). The first and last latent frames become **global anchors** — every frame sees them, and they see everything. Everything else — distant past and distant future — goes into two linear memories, one scanning forwards, one backwards.

**2. Update the linear memory once per *frame*, not once per token.** This is the actual novelty, and it's called Video Delta Attention (VDA).

> [!NOTE] Video Delta Attention
> A delta-rule memory update where all $U$ spatial tokens of one video frame are written into the recurrent state *jointly*, by solving a small least-squares problem, so overlapping tokens do not fight each other. ^video-delta-attention

Why this did not exist before: delta-rule linear attention (DeltaNet, Gated DeltaNet, Kimi Linear) was built for [[Auto-regressive models|autoregressive]] language decoding, where exactly one token arrives per step. Video diffusion has no such order — all the patches of a frame exist at once. You can force an arbitrary raster order, or you can write them all in parallel using the same frozen old state. The parallel version is what SANA-WM does, and it has a flaw: two patches that point in similar key directions each write a correction that ignores the other, so they double-count or conflict.

VDA fixes that by asking the whole frame to agree on one new state. What it unlocks: a linear branch that is stable without hand-tuned scaling factors, and quality that survives the swap. Eight-step VDN-H3 **matches or slightly beats** the 50-step dense baseline on five video-quality metrics, at **14.5× lower denoising latency** (6.70 s for 14.3 s of 768p video on 8×B200).

## The Methodology

### The softmax branch: window plus anchors

The local window follows the video tokenizer, not a round number. H3's [[VQ-VAE|VAE]] decodes 5 latent frames as one temporal chunk, so each query chunk attends to itself plus the chunk before and the chunk after — a **15-frame bidirectional window**. The point is not to cut the softmax boundary through the middle of a tokenizer chunk.

On top of that, **two boundary anchors with four-way connectivity**: every frame attends to all tokens of frame 0 and frame $F{-}1$, and those two frames attend to the whole sequence. Only two rows and two columns of the attention matrix are dense. This is cheap and it is exactly what image-to-video and first-last-frame-to-video conditioning needs, since the conditioning images *are* those frames.

Measured density: at 102 latent frames, softmax covers **20.0%** of the video-video pairs. At 42 frames it is 42.1%. The longer the clip, the bigger the win.

### The linear branch: two scans

For query frame $t$, the forward state $S_t^{\rightarrow}$ summarises frames *before* the softmax window, the reverse state $S_t^{\leftarrow}$ summarises frames *after* it. Anchors are excluded (softmax already sees them). The two regions are disjoint, so you can just add the readouts:

$$o_t^L = S_t^{\rightarrow} q_t + S_t^{\leftarrow} q_t$$

Neat trick for the text prompt: summarise all text tokens into a state $S_T$, then initialise **both** scans with $S_T/2$. Adding the two readouts counts the prompt exactly once:

$$S_0^{\rightarrow} = S_0^{\leftarrow} = \tfrac{1}{2}S_T, \qquad (S_0^{\rightarrow} + S_0^{\leftarrow})q = S_T q$$

At readout time the scan state is gathered from just outside the window boundary, and the accumulated channel-wise decay across the skipped span is applied ("the decay bridge") so the memory lines up with frame $t$.

### The frame-wise delta rule, derived

Standard token-wise gated delta rule, with state $S_{t-1} \in \mathbb{R}^{d_v \times d_k}$:

$$\bar{S}_t = S_{t-1}\operatorname{Diag}(\alpha_t), \qquad S_t = \bar{S}_t + \beta_t(v_t - \bar{S}_t k_t)k_t^\top$$

$\alpha_t$ is the decay gate (how much old memory survives), $\beta_t$ the write gate (how hard to erase-and-write).

The parallel-frame version (SANA-WM) computes every patch's correction against the same frozen $\bar{S}_t$. Stack the frame's keys and values as $K_t, V_t \in \mathbb{R}^{U \times d}$ and define two frame statistics:

$$A_t = K_t^\top \operatorname{Diag}(\beta_t) K_t \quad\text{(key correlations)}, \qquad B_t = V_t^\top \operatorname{Diag}(\beta_t) K_t \quad\text{(value writes)}$$

which gives the additive update $S_t^{\text{batch}} = \bar{S}_t(I - A_t) + B_t$.

VDA instead defines the new state as the minimiser of a joint objective:

$$S_t = \arg\min_S \ \tfrac{1}{2}\|S - \bar{S}_t\|_F^2 + \tfrac{1}{2}\sum_{u=1}^{U}\beta_{t,u}\|S k_{t,u} - v_{t,u}\|_2^2$$

In words: *stay close to the decayed old memory, while fitting all of this frame's key→value associations at once.* Differentiate, set to zero, and you get a normal equation with a closed form:

$$S_t(I + A_t) = \bar{S}_t + B_t \qquad\Longrightarrow\qquad S_t = (\bar{S}_t + B_t)(I + A_t)^{-1}$$

The inverse is $d_k \times d_k$ — small, computed in the key-channel space.

The difference from the batched rule is one character. Batched: every residual uses $\bar{S}_t$. VDA: every residual uses the *shared new* $S_t$.

$$S_t = \bar{S}_t + \sum_u \beta_{t,u}(v_{t,u} - S_t k_{t,u})k_{t,u}^\top$$

### Why the inverse buys stability for free

**Proposition 1.** With $\beta_{t,u} \ge 0$ and $0 \le \alpha_{t,j} \le 1$, the inherited-state transition $M_t = \operatorname{Diag}(\alpha_t)(I + A_t)^{-1}$ has $\|M_t\|_2 \le 1$. Old state can never be amplified.

The proof is short: $A_t$ is a Gram matrix, so it's positive semi-definite, so its eigenvalues $\lambda_i \ge 0$, so $(I+A_t)^{-1}$ has eigenvalues $1/(1+\lambda_i) \in (0,1]$. Multiply by a diagonal of numbers in $[0,1]$ and the spectral norm stays $\le 1$.

Contrast the additive rule with $\alpha_t = \mathbf{1}$: the factor is $I - A_t$, whose eigenvalues $1 - \lambda_i$ leave $[-1,1]$ as soon as any $\lambda_i > 2$. With unit keys, $\lambda_{\max}(A_t) \le \operatorname{tr}(A_t) = \sum_u \beta_{t,u} \le U$, and aligned keys approach that bound. So SANA-WM has to scale keys by $1/\sqrt{U}$ to force stability. **VDA gets it from the algebra.**

The inverse also adapts to key geometry. If all $U$ patches repeat the same unit key $k$ with gate $\beta$, the readout is

$$S_t k = \frac{1}{1+U\beta}s + \frac{U\beta}{1+U\beta}\bar{v}$$

Repeated keys get a **saturating** weight $U\beta/(1+U\beta)$; orthogonal keys each get $\beta/(1+\beta)$. Fixed $1/U$ scaling instead gives repeated-key consensus weight $\beta$ and each orthogonal direction only $\beta/U$ — it penalises diversity to control redundancy. VDA measures the redundancy instead of assuming it.

### Combining the branches

Both branches read from the **same pretrained QKV projections**. Then they diverge:

- Softmax branch: keeps H3's QK norm and [[RoPE|rotary embeddings]].
- Linear branch: its own feature map — depthwise $5\times5$ spatial + 5-tap temporal convolution on K and V, then SiLU, then L2-normalise Q and K. **No rotary** in the linear branch.

Because restricting softmax to a window concentrates its probability mass over fewer keys, its output scale changes. So each branch gets its own [[Gated Activation|sigmoid gate]], and the linear readout also gets RMS-normalised:

$$\widetilde{O}^S = G^S \odot O^S, \qquad \widetilde{O}^L = G^L \odot \operatorname{RMSNorm}(O^L)$$

And crucially, **separate output projections**, so each branch can write into different directions of the residual stream:

$$Y = \widetilde{O}^S W_O^S + \widetilde{O}^L W_O^L$$

The feed-forward sublayer is untouched. Note also: the hybrid is applied to **video↔video** interactions only. Anything involving text or audio tokens stays full softmax.

### Training: three stages then distillation

They do not train a foundation model. They convert a frozen pretrained one. Base weights stay frozen throughout A1, A2 and B.

| Stage | Steps | Trained | Grad clip |
|---|---|---|---|
| A1 per-layer | 200 | one linear branch, in isolation, from frozen pretrained activations | 0.1 per layer |
| A2 end-to-end | 500 | all linear branches assembled together | 1.0 global |
| B LoRA co-adapt | 2,000 | linear pathway + softmax gates + [[LoRA]] rank-64 on Q,K,V,O | 1.0 global |

Softmax gates are pinned at **0.99** during A1 and A2 — i.e. the model starts as almost pure softmax and the linear branch has to earn its way in. A1 exists so the initialisation signal doesn't have to travel through a deep stack of simultaneously-changing blocks; A2 exists because per-block training cannot see composition errors.

AdamW throughout, peak LR $10^{-4}$ decaying to $5\times10^{-6}$, zero weight decay, warmup then cosine. Following Chimera, parameters are split into LR groups by scale: large fan-in matrices at $1.0\times$, small vector parameters at $5.0\times$. In Stage B the linear branch uses $0.4\times$ the LoRA learning rate.

Then **[[Distillation|distillation]] from 50 steps to 8**, using a DMD2 objective with the GAN term removed. The student is initialised from a community Turbo-LoRA and distilled against *VDN-H3's own* 50-step sampler — deliberately, so step reduction is isolated from architecture conversion. Generator, real-score and fake-score models share one FSDP backbone through separate adapters; three fake-score updates per generator update; 250 generator steps.

**Data:** 10,015 clips at 1344×768, 345 frames at 24 fps. Everything pre-encoded and cached — video latents $(24,102,48,84)$, stereo audio latents $(2,32,575)$, Qwen3-VL text embeddings — so the VAEs and text encoder are out of the training loop entirely.

### Inference engineering

This half is where the wall-clock number actually comes from.

- **Four fused Triton kernels.** VDA-Prep (conv + SiLU + L2 norm + layout), VDA-Stats (builds $A_t, B_t$ in one pass), VDA-Gather (boundary states + decay bridge), VDA-Epilogue (RMSNorm + gate + layout).
- **Chunk-wise scans.** VDA gives an affine transition per frame, but readout only happens at VAE chunk boundaries. So compose each 5-frame chunk into one $S_{\text{out}} = S_{\text{in}}M_{\text{chunk}} + J_{\text{chunk}}$ and scan the shorter chunk sequence. Cuts scan depth ~5×.
- **Custom inverse.** Batched Cholesky needs several launches and round-trips through memory. They wrote one CUDA kernel doing blocked Gauss–Jordan in registers, emitting $M_t$ and $J_t$ directly. Inverse and state updates stay **FP32**.
- Window softmax packs queries by visible-key pattern so FlashAttention's varlen API works without a global mask. VDA is sharded by head ([[Megatron-LM- Training Multi-Billion Parameter Models Using Model Parallelism|Ulysses]]-style) and runs on a side CUDA stream alongside window softmax. MXFP8 for the wide GEMMs. AdaLN modulation precomputed outside the block loop.

## Ablation Studies and Experiments

**Setup.** 103 prompts from a fixed third-party set, all models at the same resolution and duration, no per-model prompt selection. Baselines: Dense H3 at 50 NFEs (full attention), FastH3 at 4 NFEs (the fast-model comparison). Metrics: EvalCrafter VQA$_A$/VQA$_T$, Q-Align, FAST-VQA, DOVER++, plus RAFT mean optical-flow magnitude as a *motion diagnostic* (not a quality score), FIRM-Video for instruction following / perceptual quality / world coherence, and PSNR/SSIM/LPIPS at the two conditioning frames for first-last-frame-to-video.

**Quality, 8-step VDN-H3 vs 50-step Dense H3:**

| | VDN-H3 (8 NFE) | Dense H3 (50 NFE) | FastH3 (4 NFE) |
|---|---|---|---|
| 5 no-reference quality metrics | **+0.06 to +1.00** over dense | baseline | −2.70 to −12.74 |
| RAFT flow (px) | 11.71 | 11.55 | 9.19 |
| FIRM Instruction Following | 2.25 | 2.25 | lower |
| FIRM Perceptual Quality | 4.45 | 4.40 | lower |
| FIRM World Coherence | 1.77 | 1.84 | lower |
| FL2VA PSNR | 28.67 | 28.85 | 27.59 |
| FL2VA SSIM | 0.826 | 0.833 | 0.785 |
| FL2VA LPIPS ↓ | 0.1156 | 0.1044 | 0.1834 |

Two things to read here. First, 8-step VDN-H3 beating 50-step dense on no-reference quality metrics is **not** evidence the architecture is better — distillation sharpens outputs, and no-reference quality scorers reward sharpness. See [[Evaluating Generative Models]]. Second, the RAFT number is the more interesting one: FastH3 drops to 9.19 px, meaning aggressive step reduction **kills motion**. VDN-H3 keeps it. Motion magnitude is a good canary for whether a fast sampler is quietly producing near-static video.

**The honest comparison is in Appendix A.6**, at matched 50 NFEs, before any distillation. There, Stage-B VDN-H3 is "broadly on par" with Dense H3 across aesthetic, technical, learned-quality and FIRM scores — but **two endpoint-fidelity measures show small degradations**. That is the real cost of the architecture swap, and it is not in the main results table.

**Efficiency, backbone only (one transformer evaluation, same sequence length and NFE):**

| | Dense H3 | VDN-H3 | Speedup |
|---|---|---|---|
| 1× H200 | 35.35 s | 11.16 s | 3.2× |
| 1× B200 | 16.00 s | 6.16 s | 2.6× |

**The scaling result, which is the important one.** As latent frames go 42 → 72 → 87 → 102:

| Latent frames | 42 | 72 | 87 | 102 |
|---|---|---|---|---|
| Softmax density | 42.08% | 26.82% | 22.85% | 19.98% |
| Attention speedup (B200) | 1.7× | 2.3× | 2.7× | 3.0× |
| Attention speedup (H200) | 2.0× | 3.0× | 3.5× | 4.0× |

The window and anchors are fixed width, so softmax density falls as $O(1/F)$ while the linear branch stays linear. The method gets *better* the longer the clip — which is the whole point, and also the reason to be careful when reading the headline number, since it is quoted at the longest length tested.

**The full deployment path (B200), where the 14.5× comes from:**

| Config | GPUs | NFE | s/NFE | Latency | Gain | Cumulative |
|---|---|---|---|---|---|---|
| Dense H3 | 1 | 50 | 15.99 | 799.6 s | – | 1.0× |
| VDN-H3 | 1 | 50 | 6.16 | 307.9 s | 2.6× | 2.6× |
| + few-step | 1 | 8 | 6.16 | 49.3 s | 6.3× | 16.2× |
| + distributed | 8 | 8 | 0.85 | **6.70 s** | 7.4× | 119.3× |

Read this table carefully. The **architecture** buys 2.6×. **Step distillation** buys 6.3×. **Eight GPUs** buys 7.4×. The abstract's 14.5× is the 8-GPU VDN-H3 against the 8-GPU dense 50-step baseline; the 119.3× column is against a *single*-GPU dense baseline and is not a like-for-like number. H200 is the same story with bigger multiples (3.2× / 6.3× / 7.1×, final 12.5 s).

**Kernel ablations (102 latent frames):**

| Kernel | H200 | B200 |
|---|---|---|
| VDA-Prep | 18.0 → 1.6 ms | 17.1 → 3.4 ms |
| VDA-Stats | 2.1× | 2.9× |
| VDA-Gather | 7.2× | 7.5× |
| VDA-Epilogue | 7.3× | 9.1× |
| Fused inverse vs Cholesky | 7.8 → 1.7 ms | 6.3 → 1.3 ms |
| Window softmax | 112.6 → 112.1 ms | 69.5 → 56.1 ms |

Two negative-ish results worth noting:

- **Window softmax optimisation does essentially nothing on H200** (112.6 → 112.1 ms). The query-packing trick only pays on B200. And note window softmax is still ~112 ms vs single-digit ms for all the VDA kernels — softmax is *still* the dominant cost even after being cut to 20% density.
- **Chunk-wise scans are a loss on a single GPU.** Composing frame transitions adds overhead when all 56 heads sit on one device. The trick only becomes a win *after head sharding*: with 7 heads per rank in 8-GPU inference, scan latency drops 4.6 → 1.1 ms (H200), 3.0 → 0.8 ms (B200). This is a real architecture/parallelism interaction — the optimisation is conditional on the deployment shape.

## Worth Remembering

**The recipe is a conversion, not a pretrain.** Total adaptation is 200 + 500 + 2,000 = 2,700 steps on 10k clips, plus 250 generator steps of distillation. The frozen-backbone-plus-staged-alignment pattern (align each new branch locally → assemble → [[LoRA]] co-adapt) is the transferable part, and it is not specific to video.

**Gate initialisation at 0.99 is the safety mechanism.** The new pathway starts contributing almost nothing, so pretrained capability is preserved while the linear branch learns to be useful. Compare to [[LoRA]]'s zero-init of the $B$ matrix — same principle, different place.

**Separate output projections are load-bearing.** Softmax and linear write into different residual-stream directions. This is a stronger form of what most hybrids do (a single shared projection after a weighted sum), and it means the two branches don't have to agree about what the output space means.

**The $(I+A_t)^{-1}$ trick generalises.** Any time you write $U$ correlated items into a fixed-size memory in parallel, "solve the joint least-squares" is strictly better-behaved than "sum independent corrections and then scale by $1/\sqrt{U}$". You get non-expansiveness without a hyperparameter, and you get correlation-awareness for free. The cost is a $d_k \times d_k$ inverse per frame per head, which they measured at 1.3–1.7 ms fused.

**Limitations the paper admits or implies:**

- At matched 50 NFEs, endpoint fidelity degrades slightly. The main-table story depends on distillation compensating.
- If a window covers the whole clip, the linear path is bypassed entirely and it falls back to dense attention. Short clips get no benefit — at 42 latent frames the speedup is only 1.7×.
- FL2VA numbers for FastH3 v2 at 8 NFEs were simply **not available**, so that comparison is incomplete.
- Significance flags in the figures compare each variant against Dense-50 only. They do **not** establish that VDN-H3 beats FastH3 significantly. The paper says this explicitly; it is easy to misread the asterisks.
- The linear branch drops rotary position encoding. Position information inside distant context comes only from the decay gates and the temporal convolution.

**Practical caveats if you wanted to use it:** the window is tied to the VAE chunk size (5), so a different tokenizer means re-deriving it. FP32 is mandatory for the recurrent states and the inverse — the paper keeps decay-$\alpha$ modules in FP32 even during training, and gradients reduce in FP32 while parameters gather in bf16. And the chunk-wise scan is a pessimisation unless you are head-sharding.

**Follow-up questions.** How does quality degrade past 102 latent frames — the fixed state has to hold more and more? Does the anchor pattern still work for videos with a hard scene cut, where frame 0 and frame $F{-}1$ are from different scenes? And is the frame-wise objective in Eq. (4) the right one, or would a weighted version (trusting recent memory more per-channel) do better?

## Links
Related: [[Linear Attention]] · [[Attention]] · [[Video Diffusion]] · [[Diffusion Transformer]] · [[Sparse Attention]] · [[Diffusion Models]] · [[Diffusion Sampling]] · [[Distillation]] · [[LoRA]] · [[Flash Attention]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Gated Activation]] · [[RoPE]] · [[Latent Diffusion]] · [[VQ-VAE]] · [[U-Net]] · [[Consistency Models]] · [[Evaluating Generative Models]] · [[Mixed Precision training]] · [[Why Gated DeltaNet Survives 4-Bit Quantization- NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM]] · [[Grouped Query Attention]] · [[Distributed Training]] · [[GPU processing]]

New topics worth writing: Delta rule linear attention (DeltaNet / Gated DeltaNet), Distribution Matching Distillation (DMD / DMD2), Ulysses sequence parallelism, Mamba and selective state-space models, Triton kernel fusion, Video quality metrics (DOVER++, FAST-VQA, Q-Align), RAFT optical flow, SGLang serving, Sherman–Morrison and small-matrix inverses in recurrent memory
