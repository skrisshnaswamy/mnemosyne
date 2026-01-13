---
title: "UniSpace: Unified Visual Representation and Scalable Multimodal Modeling"
authors: ["Jinbo Yan", "Limeng Qiao", "Jie Qin", "Junyan He", "Feize Wu", "Guanglu Wan"]
year: 2026
arxiv: "2608.08676"
url: https://arxiv.org/abs/2608.08676
priority: Good-To-Read
read_on: 2026-08-30
tags: [paper, transformers, vision]
---
## The Core Idea

Take a pretrained vision encoder like SigLIP2 or DINOv2 — a [[An Image is Worth 16x16 Words (ViT)|ViT]] trained to produce *semantic* image tokens. Those tokens are great for "what is in this picture". They are terrible for "redraw this picture pixel by pixel". Everyone assumed the detail was destroyed somewhere inside the transformer blocks, so the fix was always to train a second, separate encoder — a VAE — and carry two visual representations around.

The finding here: **the transformer blocks are not the problem. The patch embedding is.**

The diagnostic is one clean experiment. Take SigLIP2. Rip out its trained patch embedding — the little linear layer that turns each $16\times16$ pixel square into a vector — and replace it with a *random* linear projection. Freeze everything else. Train identical reconstruction probes on the features at each depth.

- Right after the patch embedding, both versions are equally recoverable: PSNR $39.29$ (pretrained) vs $39.68$ (random). No surprise, it is a nearly invertible linear map.
- After all the frozen blocks, the pretrained path gives PSNR $20.96$. The random path gives $24.66$. **+3.70 dB, with byte-identical transformer weights.**

Breaking the input encoding *improves* pixel recovery. That only makes sense if the blocks are capable of carrying detail and are being *steered away* from doing so. A ViT has no narrow bottleneck — token width stays high all the way through — so information can ride along the residual stream. But the pretrained patch embedding hands the blocks exactly the patterns they were optimised to abstract away, and layer by layer they suppress the semantically irrelevant variation. Feed them something they were never tuned on, and the low-level variation survives.

> [!NOTE] Patch Reparameterization
> Keep the original patch embedding $P_s$ frozen for semantics. Add a *second*, trainable patch embedding $P_r$ (initialised from $P_s$) that feeds the *same frozen blocks*. You get two token streams from one backbone: $T_s$ for meaning, $T_r$ for detail. ^patch-reparam

Random init is not a solution — it destroys the semantics. But it proves the ceiling is not where people thought it was. What this unlocks: a single frozen visual space that does understanding, reconstruction, generation, and editing. No VAE. The 8B model built on it (UniSpace) is the first MoT-style unified model with exactly **one** visual representation space instead of two.

## The Methodology

### Building the unified token

Two forward passes through the same frozen blocks $F_\phi$:

$$T_s = F_\phi(P_s(I)), \qquad T_r = F_\phi(P_r(I))$$

$P_s$ and $F_\phi$ are frozen forever. Only $P_r$ trains.

Then squeeze the detail stream and glue it on sideways:

$$\widetilde{T}_r = W_r T_r, \qquad T_u = \operatorname{Concat}(T_s,\, \widetilde{T}_r)$$

$W_r$ is a learned [[Linear Projection|linear projection]] on the channel axis, $d \to d_r$ with $d_r = 128$. So PR-SigLIP2 and PR-DINOv2 carry $768 + 128 = 896$ channels per token; PR-Qwen-ViT carries $1152 + 128 = 1280$.

The concatenation is the design choice that matters. They tried fusing $T_s$ and $T_r$ with an MLP into one entangled vector and it broke generation badly (see ablations). Keeping the two blocks of channels **explicitly separate and separately addressable** is what lets the generative loss be weighted per-component.

### Training the decoder

A ViT decoder $D_\psi$ (28 layers, width 1152, FFN 4096, 16 heads — ViT-XL scale) maps $T_u \to \hat{I}$. Loss is L2 + LPIPS perceptual, following the RAE recipe. Only $P_r$, $W_r$, $D_\psi$ update. Then a second phase: freeze the encoder side, train the decoder for 20 more epochs with L2 + LPIPS + a [[Generative Adversarial Networks|GAN]] discriminator loss for sharpness. [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], lr $2\times10^{-4}$, $\beta=(0.9,0.95)$, weight decay 0, batch 512, EMA 0.9978.

A nice extra step for the ImageNet numbers: **decoder calibration**. Sample latents from an early DiT checkpoint — i.e. *imperfect, generated* latents — and fine-tune only the decoder on those, so the decoder learns to cope with the errors the generator actually makes.

### Balanced flow matching

For generation they use conditional flow matching (a [[Score-Based Generative Modeling through SDEs|continuous-time]] cousin of [[Denoising Diffusion Probabilistic Models|diffusion]]). Straight-line path from noise to data:

$$Z_t = (1-t)Z_0 + tZ_1, \qquad V_t = \frac{\mathrm{d}Z_t}{\mathrm{d}t} = Z_1 - Z_0$$

with $Z_1 = T_u$, $Z_0 \sim \mathcal{N}(0,\mathbf{I})$, $t \sim \mathcal{U}[0,1]$. The network $v_\theta(Z_t, t, c)$ predicts the velocity.

A plain MSE over all 896 channels would silently weight the two parts by their dimension count and scale — 768 semantic channels would drown 128 detail channels. So split the velocity, normalise each half by its own size, and weight explicitly:

$$\mathcal{L}_{\mathrm{BFM}} = \mathbb{E}\left[(1-\lambda_r)\frac{\lVert\widehat{V}_t^{s}-V_t^{s}\rVert_F^2}{Na} + \lambda_r\frac{\lVert\widehat{V}_t^{r}-V_t^{r}\rVert_F^2}{Nb}\right]$$

$\lambda_r = 0.75$: three quarters of the objective weight goes to the 128 detail channels. Sampling is a 50-step Euler ODE solve from $t=0$ to $1$, then decode.

### UniSpace, the 8B system

Backbone: Qwen3-8B, decoder-only. Architecture: Mixture-of-Transformer-Experts, copied from BAGEL. Two experts — one for understanding, one for generation — with **hard routing by token type**. Text tokens and clean reference-image tokens go to the understanding expert; noised target visual tokens go to the generation expert. Self-attention at every layer is shared, so the two sides see each other with no bottleneck. This is not [[Sparsely-Gated Mixture-of-Experts Layer|learned-gate MoE]] — routing is deterministic on modality.

The one difference from BAGEL: BAGEL feeds SigLIP2 tokens to understanding and FLUX-VAE latents to generation — **two visual spaces**. UniSpace feeds $T_u$ to everything — **one space**, and the tokenizer stays frozen for all of training.

Three tasks, same tokens, different arrangement:
- **Understanding**: image as $T_u$, predict text with next-token [[Cross Entropy|cross-entropy]] $\mathcal{L}_{\mathrm{NTP}}$.
- **T2I**: target image as $T_u$, noise it along the flow path, predict velocity conditioned on text.
- **Editing**: reference image as $T_u^{\mathrm{ref}}$ into the understanding expert, plus instruction; noised target in the same $T_u$ space into the generation expert. Same decoder.

$$\mathcal{L} = \mathbb{E}_{(x,\tau)}\big[\mathbf{1}_{\tau=\mathrm{und}}\mathcal{L}_{\mathrm{NTP}} + \mathbf{1}_{\tau=\mathrm{t2i}}\mathcal{L}_{\mathrm{vis}}(c_{\mathrm{t2i}}) + \mathbf{1}_{\tau=\mathrm{edit}}\mathcal{L}_{\mathrm{vis}}(c_{\mathrm{edit}})\big]$$

**Curriculum** (resolution ramps, tasks get added):

| | Stage 1 | Stage 2 | Stage 3 | SFT |
|---|---|---|---|---|
| Res | 256 | 512 | 1024 | 1024 |
| Tasks | T2I + Und | + Edit | same | T2I+Edit+VLM |
| Ratio | 10:1 | 10:3:1 | 10:3:1 | 10:3:2 |
| Steps | 170K | 115K | 60K | 12K |
| Tokens | 110B | 196B | 151B | 15B |
| Peak lr | $2\!\times\!10^{-4}$ | $1\!\times\!10^{-4}$ | $4\!\times\!10^{-5}$ | $2\!\times\!10^{-5}$ |
| Seq len | 3,072 | 8,192 | 12,288 | 13,000 |

~510M sample instances, ~470B multimodal tokens, 256 Ascend 910B NPUs, ~142K NPU-hours total. [[Mixed Precision Training|BF16]], FSDP with `SHARD_GRAD_OP`, activation checkpointing, gradient clip 1.0, **no EMA**, optimizer and LR schedule reset at each stage.

**Packed training**: many examples concatenated into one sequence until the token budget fills (73–79% utilisation). Block-wise attention masks stop examples seeing each other; position indices reset per example; text uses causal attention, visual tokens use full attention within their own sample. 10% dropout on text conditioning for [[Classifier-Free Diffusion Guidance|classifier-free guidance]], 10% on instructions.

## Ablation Studies and Experiments

### Reconstruction — the headline win

ImageNet-1K val, $256\times256$. Compared against RAE, which uses the *same* frozen backbones with a normal patch embedding:

| Tokenizer | PSNR ↑ | SSIM ↑ | rFID ↓ |
|---|---|---|---|
| RAE (SigLIP2-B) | 19.35 | 0.49 | 0.53 |
| **PR-SigLIP2** | **29.64** | **0.87** | **0.18** |
| RAE (DINOv2-B) | 18.86 | 0.48 | 0.57 |
| **PR-DINOv2** | **30.84** | **0.90** | **0.14** |
| RAEv2 (DINOv3-**L**, K=7) | 22.57 | 0.63 | 0.29 |
| FLUX-VAE | 32.74 | 0.92 | 0.18 |
| SD-VAE 3 | 31.29 | 0.87 | 0.20 |
| VA-VAE | 27.96 | 0.79 | 0.28 |

rFID drops 66% (SigLIP2) and 75% (DINOv2) against matched-backbone RAE. PR-DINOv2 beats FLUX-VAE and SD-VAE 3 on rFID while *also* still doing semantics — and it beats RAEv2 while using a smaller Base backbone against RAEv2's Large.

### Understanding — did not break

LLaVA-v1.5 protocol, Vicuna-7B, frozen encoder, full $T_u$ into the projector. Average over POPE/GQA/TextVQA/MMVet/MMBench/MME:

- SigLIP2-B baseline **63.39** → PR-SigLIP2 **64.37**
- Qwen-ViT baseline **68.29** → PR-Qwen-ViT **68.94**

Slightly *up*, not down. Which makes sense — $T_s$ is bit-identical to the original, and $\widetilde{T}_r$ is 128 extra channels of free information.

### Generation — the honest weak spot

ImageNet $256\times256$, 839M DiT with a DDT head:

| | tokenizer rFID ↓ | gFID no CFG ↓ | gFID w/ CFG ↓ |
|---|---|---|---|
| RAE (DINOv2-B) | 0.57 | 1.51 | 1.13 |
| RAEv2 (DINOv3-L) | 0.29 | 1.65 | 1.06 |
| VTP-L | 0.36 | 1.85 | 1.11 |
| **PR-DINOv2** | **0.14** | 2.10 | 1.87 |
| **PR-SigLIP2** | 0.18 | 4.42 | 2.80 |

**PR is not the best generator.** It moves the operating point: far better reconstruction, somewhat worse gFID. The paper is upfront that this is a trade-off, not a free lunch. PR-SigLIP2 in particular is a lot worse at generation than PR-DINOv2 — the SigLIP2 latent space seems harder for a flow prior to model.

### The failed design: entangled fusion

The most instructive negative result. They built the "obvious" version: MLP-merge $Z_m = M([Z_s; Z_r])$ into one entangled latent, aligned to both.

On *real encoded* latents it looks perfect: zero-shot accuracy **78.53** (SigLIP baseline 79.10), PSNR **33.83**, rFID **0.069** — better than the final factorized design on every metric.

Then train a DiT on it and decode the *generated* latent:
- with the high-fidelity decoder $D_r$: **FID 120.9** — total failure
- with the semantic decoder $D_s$: **FID 8.07**

The generator learned the semantic structure and completely missed the detail directions. Why? They measured $\rho_s = \mathrm{Var}(M(Z_s,0))/\mathrm{Var}(M(Z_s,Z_r)) \approx 95\%$ — the semantic input explains 95% of the merged output's variance. Detail lives in a tiny, uncontrollable sliver of the space.

> [!NOTE] Representation quality ≠ generative modelability
> A latent can support perfect reconstruction *from real encodings* and still be un-generatable. If the information you need occupies a low-variance direction inside an entangled space, the flow prior will never find it. Explicit channel separation is what makes it addressable. ^modelability-gap

### $\lambda_r$ sweep (Table 12)

Same 768+128 representation throughout, only the loss weight varies:

| $\lambda_r$ | 0.25 | 0.60 | **0.75** | 0.82 | 0.86 | 0.92 |
|---|---|---|---|---|---|---|
| FID ↓ | 8.99 | 7.42 | **7.10** | 7.24 | 7.13 | 7.39 |

A middle optimum, not monotone. Under-weight the detail channels and the decoder gets garbage; over-weight them and you stop modelling the semantic scaffold that makes generation learnable at all. 0.6–0.86 is a wide flat basin, so it is not fragile.

### Compression sweep (Table 13)

$d_r = 128$ vs no compression ($d_r = 768$):

| $d_r$ | PSNR | rFID | FID @20ep | FID @40ep |
|---|---|---|---|---|
| 128 | 29.79 | 0.163 | **9.95** | **6.92** |
| 768 | **33.61** | **0.085** | 15.51 | 11.07 |

Uncompressed is clearly better at reconstruction and clearly worse at generation. $W_r$ is not there to help reconstruction — it is there to shrink the distribution the generator has to learn. Same lesson as the entanglement result, from the other side.

### UniSpace system-level

**Editing (ImgEdit, 9 categories, GPT-judged):** UniSpace 8B → **4.28** overall. BAGEL 7B → 3.20. SenseNova-U1 8B → 3.90. Emu3.5 32B → 4.41. Qwen-Image-Edit-2511 20B → 4.51. So it beats every comparable-scale unified model by a wide margin and lands near 4× larger models. Weakest category by far: **Hybrid, 2.70** (multi-instruction compositional edits) — worse than BAGEL's peers and its own other categories.

**Editing (GEdit):** EN overall 7.41, CN 7.38, bilingual avg 7.39. Beats BAGEL (6.52/6.50). Semantic Consistency is strong (**8.29** EN, best in table) but Perceptual Quality is weak (**7.06** vs SenseNova 7.49, Qwen-Image-Edit-2511 8.20). It follows instructions faithfully; it just does not make the prettiest pixels.

**T2I:** GenEval **0.84** overall (SenseNova-U1 0.91, Qwen-Image 0.87) — weak on counting (0.69) and colours (0.88), strong on position (0.83). OneIG-Bench bilingual avg **0.547**, edging out Emu3.5 32B (0.546) and SenseNova-U1 (0.542); best style score in the table on both languages (0.467 EN / 0.455 CN). DPG-Bench **86.49**, beating BAGEL 85.07 and Show-o2 86.14, with the **best relation score of any model listed, 94.97**, but weak Global (84.80).

## Worth Remembering

- **The mechanism claim is causal, not correlational.** One-variable intervention, everything else frozen, identical probes. That is why the +3.70 dB matters more than any benchmark table in the paper.
- **Frozen backbone + added parallel path** is the same shape as [[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]] and as adapter tuning, but applied at the *input parameterization* rather than inside the weights. The claim generalises: what a frozen network can express depends on how you encode the input into it. Worth holding onto.
- **Where does the detail actually live?** The authors say the residual stream carries it — the ViT never narrows, so input-dependent variation can survive. The pretrained $P_s$ puts the input into a region the blocks were trained to collapse; $P_r$ learns to put it somewhere they leave alone. This is a hypothesis with a strong existence proof, not a mechanistic account. Nobody probed *which* directions or *which* layers.
- **Limitation the authors state plainly:** because the encoder is shared and frozen, you cannot fine-tune it on understanding data without risking the generative side. That architectural constraint caps how good the understanding pathway can get. They also decline to report UniSpace's own VLM benchmark scores, pointing instead at the controlled Table 4 — read that as an admission that the 8B system's understanding lags dedicated VLMs.
- **Practical caveat for anyone reusing this:** if you want the best ImageNet gFID, RAE/RAEv2 still win. PR is the right choice when you need reconstruction fidelity *and* semantics from one frozen encoder — i.e. editing, where you must preserve untouched regions pixel-for-pixel.
- **The 128-channel number is a hyperparameter nobody swept properly.** They compared 128 vs 768. Nothing at 256 or 64. The "compression helps generation" claim is a two-point line.
- All UniSpace training data is internal and unreleased. The tokenizer results are on ImageNet-1K and reproducible; the system results are not.
- Connects nicely to [[Self-Supervised Learning from Images with I-JEPA|I-JEPA]] and the whole semantic-vs-pixel-representation tension, and to [[Distilling the Knowledge in a Neural Network|distillation]] approaches like UniFlow that get the same goal by *teaching* a new encoder instead of *rewiring* an old one.

## Links

Related: [[An Image is Worth 16x16 Words (ViT)]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Denoising Diffusion Probabilistic Models]] · [[Score-Based Generative Modeling through SDEs]] · [[Classifier-Free Diffusion Guidance]] · [[Sparsely-Gated Mixture-of-Experts Layer]] · [[Attention Is All You Need]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Cross Entropy]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Mixed Precision Training]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Generative Adversarial Networks]] · [[Self-Supervised Learning from Images with I-JEPA]] · [[Linear Projection]] · [[Distilling the Knowledge in a Neural Network]] · [[Foundation Models]]

New topics worth writing: Flow Matching and rectified flow, SigLIP and CLIP contrastive image-text pretraining, DINOv2 self-distillation, Representation Autoencoders (RAE), FID / rFID / gFID as metrics, LPIPS perceptual loss, Mixture-of-Transformer-Experts and modality-routed architectures, Diffusion Transformers (DiT), packed sequence training with block attention masks, unified multimodal models (BAGEL, Emu3.5, SenseNova-U1), instruction-based image editing benchmarks (ImgEdit, GEdit, GenEval, DPG-Bench)
