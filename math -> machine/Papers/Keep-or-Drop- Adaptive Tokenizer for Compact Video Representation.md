---
title: "Keep-or-Drop? Adaptive Tokenizer for Compact Video Representation"
authors: ["Yeonkyeong Lee", "Hyunsung Go", "Jongmin Kim", "Sewoong Lim", "Donghoon Lee"]
year: 2026
arxiv: "2608.24293"
url: https://arxiv.org/abs/2608.24293
priority: Good-To-Read
read_on: 2026-09-05
tags: [paper, transformers, diffusion, vision, theory]
---
## The Core Idea

Video diffusion models do not work on pixels. They work on a compressed code made by a [[Auto-Encoding Variational Bayes (VAE)|VAE]]. The usual VAE compresses by a **fixed** amount: a 256×256×16 clip always becomes the same number of tokens, no matter what is in it. A clip of a blank white wall costs the same as a clip of a crowded street. That is obviously wasteful, because video is full of repetition — most patches look like the patch next to them, and almost every patch looks like the same patch one frame earlier.

KATok makes the compression **content-dependent**. Each latent token gets a small network that outputs two numbers: keep or drop. The choice is made differentiable with a [[Auto-Encoding Variational Bayes (VAE)#^reparameterisation-trick|Gumbel–Softmax]] relaxation, so gradients flow through the decision. An $\ell_1$ penalty on the keep-probabilities pushes the model to drop as many tokens as it can get away with. The number of tokens is therefore **an output of training, not a hyperparameter**.

The important distinction the authors draw:

> [!NOTE] Flexible vs adaptive tokenization
> *Flexible* means the user can pick a token budget (ElasticTok, FlexTok, One-D-Piece). Finding the right budget for a given clip needs a binary search at inference time. *Adaptive* means the model itself decides the budget from the content, in one forward pass, with no search. ^flexible-vs-adaptive

The second idea is the one that makes it usable downstream. If you drop tokens, the surviving tokens sit at scattered, irregular positions in the 3D grid. A diffusion model trained on this sparse set has to generate *both* what the tokens contain *and* where they go — otherwise you get **content–position misalignment**: content generated for a location that does not match its coordinates, producing broken boundaries and flicker. KATok fixes this with a tiny separate model that generates the occupancy mask first, then conditions the content generator on those coordinates.

Result: 366 tokens per clip instead of OmniTokenizer's 5,120 (a 14× reduction), *better* reconstruction (PSNR 31.24 vs 28.10), and *better* generation (gFVD 61.53 vs 100.00 on UCF-101), with 6.9× faster training to a given quality.

## The Methodology

### The tokenizer

A video $X \in \mathbb{R}^{T \times H \times W \times C}$ is patchified into $N$ patches with a **linear** projection (patch size $16^2 \times 8$ — so a $256^2 \times 16$ clip gives $16 \times 16 \times 2 = 512$ patches).

Encoder: 12-layer self-attention transformer, ViT-B scale (hidden 768, 16 heads, MLP ratio 4). It uses 3D [[RoFormer- Enhanced Transformer with Rotary Position Embedding|RoPE]] and 2 **register tokens** (always kept, act as global anchors). Total model 344M params.

For each token the encoder output $e_i$ feeds two heads:

- a Gaussian head, $(\mu_i, \sigma_i) = f_\theta(e_i)$, latent dim 64, sampled with $\hat z_i = \mu_i + \sigma_i \odot \epsilon_i$;
- a keep/drop head, $\alpha_i = g_\theta(e_i) \in \mathbb{R}^2$.

Training-time mask:
$$[\tilde m_i,\, 1-\tilde m_i] = \mathrm{GumbelSoftmax}(\alpha_i;\tau)$$
with $\tau$ annealed $2.0 \to 0.1$ over 10K steps, **never hardened** (`hard=False`). Masks below $0.01$ are clamped to $0$. Gated latent: $z_i = \tilde m_i \cdot \hat z_i$. At inference the mask is hard: $m_i = \mathbb{I}(\alpha_{i0} \ge \alpha_{i1})$.

**Soft attention masking.** Zeroing the latent is not enough — the decoder still attends to the slot. So the same soft mask biases the decoder's attention logits:
$$A_{ij} \leftarrow A_{ij} + b_j, \qquad b_j = \log(\tilde m_j + \varepsilon)$$
A dropped token gets $\log \varepsilon \to -\infty$ and is invisible to attention; a half-kept token is half-audible. To keep [[FlashAttention- Fast and Memory-Efficient Exact Attention|FlashAttention]] usable they avoid materialising an $N \times N$ mask with a **KV-bias trick**: append a channel, $q_i \leftarrow [q_i, 1]$ and $k_j \leftarrow [k_j, \sqrt{d}\, b_j]$, so the extra dot product reproduces the bias exactly.

**Sparsity loss.**
$$\mathcal{L}_{\text{sparse}} = \mathbb{E}_X\Big[\textstyle\sum_{i=1}^N \tilde m_i(X)\Big] \approx \mathbb{E}_X[N_{\mathrm{eff}}(X)]$$
Weight $\lambda_{\text{sparse}} = 0.01$, switched on only after 5K steps and annealed to full by 20K. Turning it on from step 0 does not give the model time to learn to reconstruct first.

**Decoder.** FLUX-style *double-stream* blocks, 18 layers. One stream carries $M$ learnable query tokens with 3D RoPE; the other carries the latent tokens **with no positional encoding at all** (all-zero position indices). The queries pull content out of the latents through cross-attention.

> [!NOTE] Asymmetric coarse-to-fine decoding
> Encoder patch $16^2\times 8$ (512 tokens), decoder query grid $8^2\times 4$ (4,096 queries). Only the query count grows; the latent set stays small. You pay the fine-grid cost once in the decoder, never in the latent, so diffusion still runs on a tiny sequence. ^asymmetric-decoding

### Full loss

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{KL}}\mathcal{L}_{\text{KL}} + \lambda_{\text{sparse}}\mathcal{L}_{\text{sparse}} + \lambda_{\text{adv}}\mathcal{L}_{\text{adv}} + \lambda_{\text{align}}\mathcal{L}_{\text{align}}$$

- $\mathcal{L}_{\text{recon}}$: $\ell_1$ (weight 1.0) + **Video-LPIPS** (weight 0.1) — perceptual distance computed from an S3D spatio-temporal network instead of the usual image-only LPIPS.
- $\mathcal{L}_{\text{KL}}$: per-token [[KL Divergence]] to $\mathcal{N}(0,I)$, **weighted by $\tilde m_i$** (dropped tokens are not regularised). Weight $10^{-7}$.
- $\mathcal{L}_{\text{align}}$: cosine dissimilarity between a projection of the latents and features from a frozen vJEPA-2 ViT-H encoder (weight 0.5). Crucially the mask values are **detached** here, so alignment cannot fight the sparsity loss by pushing masks to 1.
- Latent noise augmentation: $\tilde z_i = (1-\eta)z_i + \eta\epsilon$, $\eta \sim \mathcal{U}(0, 0.2)$.
- $\mathcal{L}_{\text{adv}}$: stage-3 only, non-saturating [[Generative Adversarial Networks|GAN]] loss, discriminator initialised from the pretrained encoder, with approximate R1/R2 regularisation ($\gamma=0.25$).

**Three stages**, batch 256, [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], fp32, on 8×H200 nodes: (1) 210K steps at $256^2\times 16$; (2) 30K steps multi-resolution (up to $512^2\times16$, $256^2\times 64$); (3) 50K steps GAN fine-tune at LR $10^{-6}$ for the generator.

### Diffusion on sparse tokens

Backbone is SiT-XL (686M, depth 28, hidden 1152), trained with **flow matching**: interpolate $z_t = (1-t)z_0 + t z_1$ with $z_1 \sim \mathcal{N}(0,I)$ and regress the velocity,
$$\mathcal{L}_{\text{content}} = \mathbb{E}\big[\|v_\phi(z_t,t) - (z_1 - z_0)\|^2\big]$$
Because token counts differ per sample, all tokens in a batch are packed into one sequence and processed with `varlen` FlashAttention — no padding, no mask.

Conditioning via AdaLN-Zero: timestep + class label (10% CFG dropout) + **token-length embedding** (10% dropout). That last one is what enables the token count to become a control knob at sampling time.

**Variant A — joint content–position.** Concatenate the raw normalised coordinates $(t,h,w)$ to each token, 64 → 67 channels, and add a mirror flow-matching loss on the position channels:
$$\mathcal{L}_{\text{pos}} = \mathbb{E}\big[\|v_\phi(\rho_t,t) - (\rho_1-\rho_0)\|^2\big], \quad \lambda_{\text{pos}}=1$$
The two groups get **decoupled noise schedules**: at inference, timesteps are shifted by $t' = \sigma(\mathrm{logit}(t)\cdot\sigma_s + \mu_s)$ with $(\mu_s,\sigma_s) = (0,1)$ for content and $(-2.0, 0.3)$ for position, and the velocity is rescaled by $\mathrm{d}t'/\mathrm{d}t$ per group. The position shift makes layout resolve *early*, before content detail. This works, but is sensitive to the schedule hyperparameters.

**Variant B — cascaded mask prior (the default).** A separate 8.3M SiT (hidden 192, depth 12) generates a binary occupancy map over the fixed $2\times16\times16 = 512$ grid, targets in $\{-1,+1\}$, trained with the same flow-matching loss at $\lambda_{\text{prior}} = 0.1$, jointly with the content model but in its own parameter group. At inference: sample the mask, take the **top-$k$** cells where $k$ is the desired token count, feed those grid-centre coordinates through a 2-layer SiLU MLP to get positional embeddings, then run the content model conditioned on them. During training the content model sees ground-truth positions.

Sampling uses `dopri5` (atol $10^{-6}$, rtol $10^{-3}$) with CFG scale 4.0.

## Ablation Studies and Experiments

### Reconstruction (Panda-70M val, ~5K clips)

| Method | Res | #Tokens | Ch | Comp.↑ | PSNR↑ | LPIPS↓ | SSIM↑ | rFVD↓ |
|---|---|---|---|---|---|---|---|---|
| Omni-VAE | $256^2\times17$ | 5120 | 8 | 96 | 28.10 | 0.05 | 0.88 | 7.84 |
| Elastic-KL | $256^2\times16$ | 3845.6 | 8 | 102 | 30.52 | 0.06 | 0.91 | 12.37 |
| **KATok** | $256^2\times16$ | **366.2** | 64 | **134.2** | **31.24** | **0.04** | **0.94** | **5.12** |
| Omni-VAE | $512^2\times33$ | 36864 | 8 | 96 | 24.07 | 0.06 | 0.80 | 16.85 |
| **KATok** | $512^2\times32$ | **1554.2** | 64 | **253.0** | **33.23** | 0.05 | **0.95** | **6.40** |

The compression ratio here is channel-aware, $\frac{HWT\cdot 3}{\#\text{tokens}\times\text{channels}}$, which is fair since KATok uses 64 latent channels against everyone else's 8.

The interesting curve is **Fig. 4**: as resolution grows, KATok's compression ratio *rises* (134 → 253) while OmniTokenizer's stays flat and ElasticTok's barely moves. More pixels means more redundancy, and only an adaptive scheme cashes that in. A solid-white clip reconstructs from **28 tokens**.

### What drives token allocation

Correlating $N_{\mathrm{eff}}$ against Shannon-entropy measures of clip complexity: **temporal** entropy $r = 0.865$, **spatial** entropy $r = 0.618$, joint $r = 0.877$. So the selector is mostly a motion detector, not a texture detector. A complex but static scene gets few tokens despite high spatial entropy.

Dataset means at $256^2\times16$ (max 512): SkyTimelapse 316.9, UCF-101 368.2, Kinetics-600 399.2 — exactly the ordering you would guess from how much stuff moves.

### VAE ablations (stage-1, 100K steps)

| Config | PSNR↑ | LPIPS↓ | SSIM↑ | Tokens |
|---|---|---|---|---|
| Full | 30.85 | 0.07 | 0.93 | 365.6 |
| − latent reg. | 31.26 | 0.07 | 0.94 | 361.7 |
| − asymmetric decoding | 29.61 | 0.11 | 0.92 | 377.8 |
| − Video-LPIPS | 29.28 | 0.11 | 0.91 | 404.0 |
| − register tokens | 29.17 | 0.12 | 0.91 | 410.2 |
| − soft attention mask | **19.00** | **0.51** | 0.54 | **2.0** |
| − Gumbel–Softmax | **18.91** | **0.52** | 0.53 | **2.0** |

Two things **collapse training entirely**, leaving only the two register tokens alive:

1. **Removing the soft attention mask.** Zeroing the latent value alone gives the decoder no gradient signal distinguishing "dropped" from "happens to be near zero", and the sparsity loss wins outright.
2. **Removing Gumbel–Softmax.** Without the differentiable sampling relaxation, the selector cannot learn which tokens are worth keeping and just drops everything.

These are not "helps a bit" components. They are the whole mechanism.

Note also that **latent regularisation makes reconstruction slightly worse** (31.26 → 30.85 PSNR) and they keep it anyway — see the generation ablation.

### Sparsity weight sweep (70K steps, symmetric patches)

| $\lambda_{\text{sparse}}$ | 0.005 | 0.010 | 0.020 | 0.050 | 0.100 |
|---|---|---|---|---|---|
| PSNR↑ | 28.67 | 28.78 | 22.46 | 21.14 | 19.62 |
| rFVD↓ | 50.16 | 50.38 | 462.27 | 680.45 | 863.51 |
| Tokens | 401.5 | 372.3 | **34.9** | 16.2 | 6.9 |

A genuine **phase transition** between 0.01 and 0.02: token count falls off a cliff from 372 to 35 and quality collapses with it. Cheap redundancy gets removed first; past a threshold the pressure eats load-bearing tokens. They use 0.01. They admit they never ablated the Gumbel temperature $\tau$.

### Generation (gFVD↓, 5K samples, 100K steps, same SiT-XL for all)

| Model | Sky | UCF | Kinetics |
|---|---|---|---|
| Omni-VAE (5120 tok) | 23.28 | 100.00 | 206.58 |
| Elastic-KL (3845 tok) | 95.53 | 712.56 | — |
| KATok-Joint | 21.36 | 73.16 | 193.78 |
| **KATok-Cascaded (366 tok)** | 23.19 | **61.53** | **160.84** |

Generation design ablation on UCF-101, tokenizer fixed:

| Variant | gFVD↓ |
|---|---|
| Naïve (content only) | 95.69 |
| Joint content+position | 73.16 |
| Cascaded mask prior | **61.53** |

Naïve sparse flow matching already beats dense OmniTokenizer, but suffers visible misalignment (Fig. 9, red boxes). Handling position explicitly is worth ~35 gFVD.

And the ablation that justifies keeping a regulariser that hurts reconstruction: with the stage-1 tokenizer, **removing latent regularisation moves gFVD from 101.23 to 161.23**. Reconstruction PSNR and downstream generation quality point in opposite directions here. This echoes the well-known reconstruction-vs-generation tension in latent diffusion.

### Speed

On 8×H200, batch 256, UCF-101, 200K steps: Cascaded reaches 49.34 gFVD vs OmniTokenizer's 82.31, and hits 73.81 at **80K steps** — beating OmniTokenizer's *final* number, a **6.9× wall-clock speedup**. Against ElasticTok: 203.51 at 30K steps vs 571.41 at 200K, ~46.7×. Throughput: 15.71 videos/s vs 4.91 (Omni), 4.24 (Elastic), 5.01 (Joint) — the Joint variant is slow because 67 channels and dual schedules cost more per step.

### Mask prior quality and robustness

Treating masks as videos and computing FVD on them: ground truth 53.08, **predicted 91.80**, 1%-flipped GT 139.58, random 15,405. The prior is *closer to real* than a 1% corruption of the real thing.

Error propagation into generation: flipping 0/1/2/5/10% of mask cells gives gFVD 69.58 / 72.63 / 81.26 / 137.74 / 264.77. Graceful up to ~2%, then it hurts. The prior costs 1.2% of parameters and 4.72% of sampling time.

### What did not work

- **Naïve sparse diffusion** — content–position misalignment, unstable boundaries and temporal flicker.
- **Joint generation** is dominated by cascaded on both quality and speed, and needs hand-tuning of two noise schedules. Kept only as an ablation.
- **Extreme sparsity at generation time** — asking for fewer than 3 tokens (outside the training regime) degenerates into near-uniform single-colour clips.
- **Failure mode of the tokenizer**: clips with high-frequency texture *and* large motion. Fast-moving fine detail reconstructs blurry, PSNR drops to ~19–20.
- ElasticTok's binary search made it impractical to use in the training loop at all; they had to precompute latents on non-overlapping 16-frame clips.

## Worth Remembering

**Emergent controllability.** Because token count is a conditioning signal during training, at sampling time you just change the length of the initial noise sequence. 200 tokens → simpler, low-motion videos; 400 tokens → dynamic, detailed videos. No retraining, no extra conditioning. This falls out of the statistical correlation between token count and scene complexity, so it is a *bias* on the sample distribution, not a precise control.

**The images vs video comparison is the cleanest evidence for the mechanism.** On ImageNet with $32^2$ patches the model keeps essentially all 64 tokens (large patches carry unique information). Drop to $16^2$ patches and it keeps 229 of 256 — neighbouring small patches overlap in information, so some become droppable. In video, the temporal axis adds a whole extra dimension of redundancy, and the same machinery drops 30% of tokens at $256^2\times16$ and much more at $512^2\times32$. Compression ratios: images 48× / 27×, video 134× / 253×. **The sparsifier is a redundancy meter, and video has far more redundancy than images.**

**Caveats if you wanted to use this.**
- The two collapse modes are unforgiving. Soft attention masking and Gumbel–Softmax are both load-bearing; the sparsity loss must be delayed (5K steps) and annealed (to 20K), or the model finds the trivial solution of keeping only register tokens.
- 64 latent channels vs the usual 8. Comparisons on raw token count flatter KATok; the channel-aware compression ratio is the honest metric and they use it.
- Everything is trained in **fp32**. No [[Mixed Precision Training|mixed precision]] reported, which for a 344M VAE at batch 256 is expensive — possibly because the Gumbel logits and $\log(\tilde m + \varepsilon)$ bias are numerically delicate.
- Adaptive token counts mean variable-length sequences everywhere downstream. They handle it with packing + varlen FlashAttention; any pipeline assuming fixed shapes needs rework.
- Only ElasticTok is a true video baseline; the VQ-based adaptive tokenizers (EVATok, AdapTok) are compared at $128^2$ in the supplement with numbers taken from their papers, not rerun.

**Open questions.** The Gumbel temperature schedule was never ablated. The masks are never hardened during training (`hard=False`), so there is a train/test gap between soft masks and the hard threshold at inference that nobody measures. And the mask prior is trained jointly with the content model on ground-truth masks — a classic exposure-mismatch setup, which the 1%-flip robustness table partly but not fully addresses.

## Links

Related: [[Auto-Encoding Variational Bayes (VAE)]] · [[Denoising Diffusion Probabilistic Models]] · [[Score-Based Generative Modeling through SDEs]] · [[An Image is Worth 16x16 Words (ViT)]] · [[RoFormer- Enhanced Transformer with Rotary Position Embedding]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[KL Divergence]] · [[Generative Adversarial Networks]] · [[Classifier-Free Diffusion Guidance]] · [[Self-Supervised Learning from Images with I-JEPA]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Sparsely-Gated Mixture-of-Experts Layer]] · [[Recommender Systems with Generative Retrieval (TIGER)]] · [[Mastering Diverse Domains through World Models (DreamerV3)]]

New topics worth writing: Gumbel-Softmax and the concrete distribution, Flow matching and rectified flow, FVD (Fréchet Video Distance), LPIPS and perceptual losses, Token pruning and merging in ViTs (DynamicViT, ToMe), Register tokens in vision transformers, Straight-through estimators, Latent diffusion models, SiT and scalable interpolant transformers, Reconstruction-vs-generation tension in latent spaces
