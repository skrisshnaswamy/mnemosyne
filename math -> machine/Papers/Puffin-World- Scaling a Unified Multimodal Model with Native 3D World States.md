---
title: "Puffin-World: Scaling a Unified Multimodal Model with Native 3D World States"
authors: ["Liao et al."]
year: 2026
arxiv: "2609.04196"
url: https://arxiv.org/abs/2609.04196
priority: Good-To-Read
read_on: 2026-09-14
tags: [paper, transformers, llm, diffusion, vision]
---
## The Core Idea

Most video or multi-view generators know where the camera moved **relative to** the first frame. None of them know which way is **down**.

That sounds like a small gap. It is not. If a model only knows relative motion, then the same trajectory — "rotate 30° to the left" — can mean two completely different things depending on whether the first frame was already tilted. The model has no anchor to the real world. Over a long trajectory the horizon slowly drifts, buildings lean, the ground stops being flat. The model is generating a plausible *sequence of pixels* without ever committing to a physical frame.

Puffin-World's move: make gravity a first-class input, and get it from the model's own perception.

The model predicts three **native world states** for every view it generates:

- **physics** — the gravity field and latitude map (where is "down", where is the horizon)
- **geometry** — a depth map
- **appearance** — the RGB image

And it does all three in one network that also *perceives* those states from a photo. That closes a loop nobody had closed: look at one real photo → figure out its absolute roll, pitch, and field of view → carry that gravity direction forward through the whole imagined trajectory → generate future frames that stay upright.

Two concrete pieces make this work.

> [!NOTE] Omni-Camera representation
> A dense, per-pixel 9-channel camera condition. Channels 1–3 are **absolute**: a per-pixel up-vector plus latitude angle, both defined against gravity. Channels 4–9 are **relative**: the standard ray map (ray origin + ray direction). One representation covers both "where am I in the world" and "how did I move since the last frame". ^omni-camera

> [!NOTE] Physics propagation
> The model perceives gravity $\mathbf{g}_0$ in the reference frame, then rotates it into every future camera frame using the known relative rotation: $\mathbf{g}_t = \mathbf{R}^{\mathrm{rel}}_{t\leftarrow 0}\mathbf{g}_0$. Every frame in the trajectory now shares one gravity-anchored world frame. ^physics-propagation

Why this did not exist before: the training data did not exist. Public 3D datasets are shot by hand-held cameras, so roll lives in $[-5°, 5°]$ and pitch in $[-10°, 10°]$. A model trained on that has never seen a camera rolled 40°, so it cannot learn to keep the horizon straight under one. Puffin-16M is the fix — panoramas rendered into perspective views with roll and pitch sampled from $[-45°, 45°]$ and yaw from $[0°, 360°)$, with exact ground-truth camera parameters for free because the rendering is synthetic.

What it unlocks: **self-calibrated world exploration**. The model looks at a crooked photo, notices the gravity misalignment, predicts a corrective camera action, and imagines the corrected view — all inside one model, no external calibration tool.

## The Methodology

### The three-way unification

The whole thing is one backbone doing three jobs. What changes between tasks is only the *input composition*, plus a 4-channel **role mask** $\mathbf{m} \in \mathbb{R}^{4 \times H \times W}$ that tells each view what it is: generation target, conditioning reference, image-conditioned input, or geometry view.

- **Representation**: RGB *and* depth go through the same frozen VAE into 16-channel latents. All camera motion, from in-place spin to large translation, goes through Omni-Camera.
- **Modality**: one vision encoder + LLM produces autoregressive text (for perception) *and*, via 64 learnable queries and a 6-layer transformer connector, conditioning vectors for the diffusion generator.
- **Task**: determined entirely by which inputs you supply.

### Physics perception (the understanding branch)

Given one image, predict roll $\phi$, pitch $\theta$, vertical FoV $\alpha$, and a scene description. Framed as **language modelling**, not regression: a geometry-aligned vision encoder (C-RADIO) feeds features through an MLP projector into the LLM, which writes out a structured spatial analysis first and the numbers second, trained with next-token [[Cross Entropy|cross-entropy]]. The reasoning-then-numbers order matters — the estimate rests on scene-level cues (horizon, vertical structures) rather than low-level texture.

### Injecting the camera into the diffusion model

This is the main departure from the earlier Puffin paper. Puffin VAE-encoded a 3-channel perspective field and cross-attended it in. That caps you at 3 channels and does not extend to trajectories.

Instead, a light condition-fusion module $\mathcal{F}$ maps $[\mathbf{C}; \mathbf{m}]$ straight into the latent space and **adds** it to the noisy latent before patching:

$$\mathbf{h}^{0} = \mathcal{P}\big(\mathbf{z} + \mathcal{F}([\mathbf{C};\mathbf{m}])\big)$$

$\mathcal{F}$ is initialised so its contribution starts at zero — the pretrained generator behaves identically on step 1. The camera features are then **re-injected** at a sparse set of blocks $\mathcal{L}$:

$$\mathbf{h}^{l} \leftarrow \mathbf{h}^{l} + \mathbf{W}_l\,\mathcal{P}\big(\mathcal{F}([\mathbf{C};\mathbf{m}])\big), \quad l \in \mathcal{L}$$

Because the injection is pixel-aligned and additive, every latent token is grounded in its own exact camera geometry. Near-zero parameter cost.

### Multi-view: one attention sequence

Instead of denoising one image, denoise $T = 8$ views in one joint-attention sequence. $K \in \{1,2,3\}$ reference views are clean latents (and also go through the vision encoder + LLM for semantic conditioning); the other $T-K$ are noised and denoised together. Each view gets a **view-axis index** with 1D [[RoFormer- Enhanced Transformer with Rotary Position Embedding|rotary positional embedding]] along that axis. Cross-view consistency emerges from shared attention — there is no explicit consistency loss.

### Depth as an RGB image

Rather than build a depth encoder, depth is mapped into colour via an invertible **3D Hilbert-curve** colour mapping, then encoded by the same frozen VAE. Three benefits: it is bounded (fits the VAE's expected input range), it is invertible (decode back to real depth), and the nonlinear curve spends more colour range on near surfaces where accuracy matters.

The depth latent is appended as an extra token block sharing its RGB counterpart's Omni-Camera condition and view index, distinguished only by the geometry channel of $\mathbf{m}$.

### The loss

Flow matching, per view:

$$\mathcal{L}_{\mathrm{fm}}(\mathcal{V}) = \mathbb{E}_{t,\bm{\epsilon}}\left[\frac{1}{|\mathcal{V}|}\sum_{v\in\mathcal{V}}\big\|\mathbf{v}_\theta(\mathbf{z}^v_t, t, \mathbf{c}^v) - (\bm{\epsilon}^v - \mathbf{z}^v_0)\big\|_2^2\right]$$

with $\mathbf{z}^v_t = (1-\sigma_t)\mathbf{z}^v_0 + \sigma_t\bm{\epsilon}^v$.

Crucially, appearance and geometry are **not pooled into one average**. The saturated Hilbert colours would swamp the converged appearance model. They are computed separately and combined with a ramp:

$$\mathcal{L} = \mathcal{L}_{\mathrm{fm}}(\mathcal{V}_{\mathrm{rgb}}) + \omega(\tau)\,\mathcal{L}_{\mathrm{fm}}(\mathcal{V}_{\mathrm{geo}}), \qquad \omega(\tau) = \omega_{\max}\min\!\left(1, \tfrac{\tau}{\tau_0}\right)$$

with $\omega_{\max} = 1$ and $\tau_0 = 3000$ iterations. Normalising each modality by its own view count keeps the appearance gradient magnitude stable; the ramp from zero stops early depth gradients from wrecking the shared output layers.

Two more safeguards protect the appearance pathway:
1. **Asymmetric attention** — appearance and text queries *cannot* see geometry tokens; geometry tokens see everything. Geometry gets full scene context, but cannot leak into the RGB stream.
2. A **zero-initialised geometry-modality embedding** after patch embedding, so at init the model is functionally identical to the pure multi-view generator.

### Long-horizon rollout

Trained on fixed 8-view chunks, but real exploration is arbitrary length. Generate a chunk, carry the last generated view forward as the next chunk's reference — **in latent space**, using the denoised latent, not a re-encoded pixel image. This avoids compounding VAE encode–decode artefacts at every chunk boundary. Physics propagation is applied across the *whole* sequence, not per-chunk, so all chunks share one gravity frame.

### Four training stages

| | Stage I | Stage II | Stage III | Stage IV |
|---|---|---|---|---|
| LR | $1\times10^{-4}$ | $2\times10^{-5}$ | $5\times10^{-5}$ | $2\times10^{-5}$ |
| Batch | 512 | 512 | 256 | 128 |
| LLM / vision enc. | frozen | trainable | frozen | frozen |
| Diffusion | frozen | trainable | trainable | trainable |
| Focus | align | SFT single-view | cross-view gen | + geometry |

[[Decoupled Weight Decay Regularization (AdamW)|AdamW]], cosine schedule, weight decay 0.05, betas (0.9, 0.95). In Stage II the vision encoder's gradients are scaled by 0.1 to preserve pretrained features.

Stage III mixes sources with fixed repeat factors $3{:}2{:}1{:}1$ and keeps single-view Puffin-Cam-15M batches as **rehearsal** so Stage-II camera controllability is not forgotten. Absolute camera fields for all training frames are pre-computed offline using the Stage-II model itself — physics propagation costs nothing at training time.

[[Classifier-Free Diffusion Guidance|CFG]]: text + Omni-Camera dropped jointly with $p=0.1$; the absolute perspective field is *additionally* dropped with $p=0.15$ while keeping the ray map, which both reduces over-reliance on absolute cues and enables component-wise guidance. Inference uses 50 steps, CFG 4.5 (single-view) / 2.0 (3D).

### Models and data

- **Base**: C-RADIOv3-H + Qwen2.5-7B-Instruct + SD3.5-Medium (2.5B)
- **Pro**: C-RADIOv4-H + Qwen2.5-1.5B-Instruct + SD3.5-Large (8.1B) — note the LLM *shrinks* and the diffusion model grows
- **Caption**: C-RADIOv3-H + Qwen3.5-0.8B, an annotation-only expert

**Puffin-16M** = Puffin-Cam-15M (15M image–text–camera triplets from 900K panoramas, up from 200K in Puffin-4M, at 7 aspect ratios) + Puffin-Traj-1M (1M trajectories: look-down, look-up, clockwise, counter-clockwise, full 360° sweeps). Panoramas were gravity-corrected via line segmentation and vanishing-point estimation. Stanford2D3D was deliberately excluded from source panoramas to keep that benchmark honest.

They also ran their own model over **28 public datasets / ~44.5M images** to add absolute camera labels, and used Depth Anything 3 to densify sparse depth in ScanNet and DL3DV (aligning predictions to the real sparse measurements to keep metric scale).

## Ablation Studies and Experiments

### Camera-to-world understanding (Table 3)

Four benchmarks, median error and AUC at 1°/5°/10°. Best median error everywhere, best AUC on most cells.

**Stanford2D3D** (vs GeoCalib, the strong specialist):
| | Roll err | Pitch err | FoV err |
|---|---|---|---|
| GeoCalib | 0.40° | 0.93° | 3.21° |
| Puffin-World | **0.29°** | **0.53°** | **1.62°** |

Pitch AUC@5° goes 74.8 → 88.8. FoV AUC@5° goes 40.0 → 61.1.

**TartanAir**, the hardest one (largest rotational spread, roll std 15.1°):
| | Roll | Pitch | FoV |
|---|---|---|---|
| GeoCalib | 0.43° | 1.49° | 4.90° |
| Puffin (prior work) | 0.40° | 0.95° | 7.48° |
| Puffin-World | **0.31°** | **0.67°** | **2.34°** |

The FoV number is the loud one: the earlier Puffin was *worse than GeoCalib* at FoV on TartanAir (7.48 vs 4.90); scaling model + data flips it to 2.34. Puffin-World also beats **AnyCalib**, a model built specifically for intrinsics, on most benchmarks.

### Camera-controllable generation (Table 4, Puffin-Cam-Bench, 600 pairs)

They measure controllability by running their *own* understanding branch on the generated image and comparing the recovered perspective field against the requested one.

| Model | Gravity mean err | FID |
|---|---|---|
| GPT Image2 | 28.83° | 99.36 |
| Nano Banana 2 | 28.00° | 90.11 |
| Qwen-Image2-Pro | 28.57° | 98.51 |
| FLUX.2-dev | 28.31° | 97.06 |
| Z-Image | 29.41° | 98.33 |
| PreciseCam | 17.07° | 90.89 |
| Puffin | 4.92° | 80.29 |
| **Puffin-World** | **1.32°** | **75.93** |

Every frontier general-purpose generator sits at ~28°, i.e. essentially *no* camera control — they make a nice picture and ignore the requested viewpoint. The gap between "beautiful" and "geometrically obedient" is enormous and not closing on its own.

### 3D world generation (Table 5)

**RealEstate10K** (50 held-out clips, conventional hand-held motion): PSNR 17.22 / SSIM 0.595 / LPIPS 0.318, narrowly beating MVGenMaster (17.11 / 0.591 / 0.348) — which had *trained on* RealEstate10K.

**Puffin-Traj-Bench** (100 clips, big rotations) is where it separates:

| | PSNR | LPIPS | Roll err | Pitch err | FoV err |
|---|---|---|---|---|---|
| MotionCtrl | 12.53 | 0.604 | 3.20° | 2.15° | 10.04° |
| CameraCtrl | 14.28 | 0.532 | 11.79° | 10.81° | 45.84° |
| ViewCrafter | 14.52 | 0.510 | 1.92° | 2.89° | 6.43° |
| SEVA | 17.94 | 0.307 | 1.35° | 1.76° | 12.59° |
| MVGenMaster | 13.64 | 0.589 | 9.24° | 9.43° | 26.98° |
| **Puffin-World** | **18.00** | **0.288** | **0.80°** | **1.10°** | **2.96°** |

CameraCtrl's 45.84° FoV error is a total collapse — it has no idea what focal length it rendered. SSIM is the one metric Puffin-World does not win (0.613 vs SEVA's 0.614), an honest tie.

### The physics-propagation ablation (Table 6)

The only clean controlled experiment in the paper. Same model, PP on vs off, on rotation-only trajectories — chosen because pure rotation is the most sensitive probe: it changes camera orientation relative to gravity directly, so any inconsistency in the physical frame shows up immediately.

| Motion | PSNR (base → +PP) | SSIM | LPIPS | Roll err | Pitch err |
|---|---|---|---|---|---|
| Roll | 22.97 → **24.02** | 0.69 → 0.74 | 0.12 → 0.10 | 2.35 → 2.04 | 6.73 → 6.31 |
| Pitch | 18.53 → **19.18** | 0.60 → 0.65 | 0.26 → 0.24 | 2.17 → 2.03 | 5.60 → 5.28 |
| Yaw | 18.12 → **18.42** | 0.59 → 0.62 | 0.30 → 0.30 | 1.31 → 1.22 | 3.10 → 2.67 |
| Avg | 19.87 → **20.54** | 0.62 → 0.67 | 0.23 → 0.21 | 1.94 → 1.76 | 5.14 → 4.75 |

The gains are consistent but modest — +0.67 PSNR on average. The interpretable part: **roll and pitch gain most, yaw gains least on appearance**, exactly as the theory predicts, because yaw rotates *around* the gravity axis and so barely changes the relationship to gravity. Yaw still improves on camera metrics (3.10 → 2.67 pitch error), i.e. PP suppresses spurious horizon tilt even when the appearance barely moves.

### Design choices that were rejected

- **VAE-encoding the camera into a separate branch and cross-attending** (the earlier Puffin design) — abandoned because 3 channels cannot hold a holistic camera representation, and the mechanism does not extend to trajectories.
- **Pooling appearance and geometry into one averaged loss** — the saturated Hilbert-curve colours would corrupt the already-converged appearance model. Hence the separate, ramped, per-modality-normalised objective.
- **Symmetric attention over geometry tokens** — would let geometry features leak into the appearance stream; hence the asymmetric mask.
- **Re-encoding pixels at chunk boundaries** during long rollouts — compounding VAE artefacts; latent-space handoff instead.
- **Dropping Stanford2D3D from the panorama source set** so its benchmark numbers are not contaminated.

## Worth Remembering

**The Pro variant shrinks the LLM.** Base uses Qwen2.5-**7B** with SD3.5-Medium; Pro uses Qwen2.5-**1.5B** with SD3.5-Large. Compute was moved from the language side to the diffusion side. The paper does not benchmark Base against Pro head to head, so the trade-off is asserted rather than shown — and it is a real gap, since which variant produced which table is never stated.

**Self-evaluation is a caveat.** Camera controllability in Table 4 and Table 5 is measured using *Puffin-World's own* understanding branch. It is applied uniformly to all methods, which is fair, but a systematic bias in that estimator would flatter the model that shares its inductive biases. Table 3 shows the estimator is genuinely state of the art, which mitigates but does not eliminate the concern.

**The Hilbert-curve depth trick is portable.** Any time you want a pretrained image VAE to carry a scalar field, an invertible colour mapping with a nonlinear allocation of range is a cheap alternative to training a new tokenizer. No interface changes, no new decoder.

**Zero-init + loss ramp is the pattern for grafting a new modality onto a converged model.** Three separate uses here: $\mathcal{F}$ starts near zero, the geometry-modality embedding is zero-initialised, and $\omega(\tau)$ ramps from 0. The general principle — make the new component the identity at $\tau=0$ — echoes [[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]]'s zero-initialised $B$ matrix and [[Deep Residual Learning for Image Recognition (ResNet)|residual]] thinking.

**The dataset annotation is arguably the most reusable artefact.** 28 public datasets, ~44.5M images, now carrying absolute roll/pitch/FoV. Their statistics table is a nice piece of dataset archaeology: roll clusters tightly at 0° everywhere (level-camera bias), pitch skews negative almost universally (people look down), and EgoObjects has mean pitch $-24.5°$ (head-mounted cameras looking at hands). Conventional vision corpora massively under-sample tilted and wide-angle viewpoints — worth knowing if you ever wonder why a model fails on a rotated photo.

**Honest limitations the authors name:** static scenes only, no dynamics, no objects that move; physics is limited to gravity and latitude (no mass, friction, contact); horizons are still finite even with the sliding window.

**Open questions.** Does physics propagation still hold up when the reference-view perception is wrong? Equation 7 propagates $\mathbf{g}_0$ exactly, so a bad initial estimate becomes a systematically wrong world frame for the whole trajectory — and the paper never studies sensitivity to that error. Also: the CFG dropout of the absolute field with $p=0.15$ suggests the model *can* over-rely on absolute cues; how much does that regulariser matter? Unmeasured.

## Links

Related: [[Denoising Diffusion Probabilistic Models]] · [[Classifier-Free Diffusion Guidance]] · [[Score-Based Generative Modeling through SDEs]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[GameWAM- A World Action Model for Video Games]] · [[Hydra-0- Action Flow for Generalist World Modeling and Control]] · [[Latent Action as Intention Enables Efficient Future Imagination for World Action Models]] · [[An Image is Worth 16x16 Words (ViT)]] · [[RoFormer- Enhanced Transformer with Rotary Position Embedding]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Attention Is All You Need]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Causal Attention]] · [[Exploring the Limits of Transfer Learning (T5)]] · [[GigaBrain-0.7- Scaling Embodied Foundation Models to Emergent Capabilities with a Three-System Architecture]] · [[PAWBench- How Far Are We from Probabilistically Aligned World Modeling]]

New topics worth writing: Flow matching, Rectified flow, Perspective fields and gravity estimation, Plücker ray embeddings, Camera intrinsics and extrinsics, Novel view synthesis, Monocular depth estimation, Multimodal diffusion transformer (MMDiT), Hilbert curve space-filling encodings, Fréchet Inception Distance, LPIPS, Structure-from-motion, Panorama-to-perspective rendering
