---
aliases:
  - Latent Diffusion Model
  - Latent Diffusion Models
  - LDM
  - Stable Diffusion
  - SDXL
  - SD3
  - Flux
  - Perceptual Compression
  - Text-to-Image Pipeline
tags:
  - generative-models
  - diffusion
  - architecture
  - performance
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Don't run diffusion on pixels. **Compress the image ~48× with an autoencoder, run diffusion in that small latent space, and decode once at the end.**
> **Metaphor:** An architect works on the blueprint, not on the building. Get the blueprint right; render the bricks last.
> **Where it bites:** This *is* Stable Diffusion (and SDXL, SD3, Flux, and most video models). It's why image generation runs on a consumer GPU — and why fine text and faces sometimes come out mangled.

---
A 512×512 colour image is **786,432** numbers.

A diffusion model's denoiser must take all of them in and put all of them out — on **every one of its 30-odd sampling steps**, and on every training step. In 2021 the best pixel-space models needed hundreds of GPU-days to train, and many seconds per image on a data-centre GPU to sample.

Now, which of those 786,432 numbers carries what? A photo of a beach: *where the horizon is, that there's a person, that it's sunset* — that's the **meaning**, and it takes very few numbers to state. The exact arrangement of every grain of sand is most of the *data*, and none of the meaning.

~={blue}Why make an expensive generative model spend nearly all its effort on sand?=~

---
# Work on the blueprint 📐

![[latent_diffusion_pipeline.png]]

Split the job in two, and give each half to the tool that's good at it:

**Stage 1 — perceptual compression.** Train an [[Autoencoder|autoencoder]] — a KL-regularised [[Variational Autoencoder|VAE]] (or a [[VQ-VAE|VQ-GAN]]) with perceptual and adversarial losses so its reconstructions stay *sharp*. It shrinks each side by 8× and keeps 4 channels: 512×512×3 → **64×64×4**. Train once, then **freeze**.

**Stage 2 — diffusion in latent space.** Encode the whole training set. Train an ordinary [[Diffusion Models|diffusion model]] on the latents. Nothing else about the recipe changes — same [[Denoising Objective|loss]], same [[Diffusion Sampling|samplers]], same [[Classifier-Free Guidance|guidance]].

**Generate:** noise in latent space → N denoising steps → **one** pass through the VAE decoder → pixels.

![[latent_vs_pixel_cost.png]]
> [!TIP] Reading the chart
> 786,432 numbers become **16,384** — a **48×** reduction in what the denoiser touches on every step. If the denoiser uses attention with one token per position, the saving is quadratic: 64² positions instead of 512² is **4,096× less attention compute**. That's the difference between "a data centre" and "a laptop".

> [!NOTE] Latent diffusion model
> A diffusion model trained in the latent space of a pretrained, frozen autoencoder rather than in pixel space (Rombach et al., 2022 — *"High-Resolution Image Synthesis with Latent Diffusion Models"*). **Stable Diffusion** is its open-weights text-to-image instance. ^ldm-def

> [!SUCCESS] Core idea
> ~={pink}Separate *perceptual* compression from *semantic* generation.=~ The autoencoder throws away imperceptible high-frequency detail and restores plausible texture on the way out — cheap, and it's good at it. The diffusion model gets a small, smooth space in which only the things that matter — layout, objects, style — are left to decide. Each network does the job it's suited to. ^two-stage-split

---
# The whole text-to-image pipeline

| # | Component | Trained? | Job |
|---|---|---|---|
| 1 | **Text encoder** — [[CLIP]] and/or T5 | frozen | prompt → a sequence of vectors |
| 2 | **Noise** | — | a random 64×64×4 latent. The **seed** |
| 3 | **Denoiser** — [[U-Net]] or [[Diffusion Transformer\|DiT]] | **the model** | predicts noise/velocity, conditioned on the text and the timestep → [[Conditional Generation]] |
| 4 | **Sampler** + [[Classifier-Free Guidance\|CFG]] | no — chosen at inference | runs step 3 in a loop, 20–50 times → [[Diffusion Sampling]] |
| 5 | **VAE decoder** | frozen | final latent → 512×512 pixels, once |

Swap-ability is the point. Change the sampler, the step count, the guidance scale, the seed — no retraining. Add a [[LoRA]] or a ControlNet — the VAE and text encoder don't care → [[Controlling Diffusion]].

---
# The generations, at a glance

| Model | Year | Denoiser | Objective | Text encoder | Latent |
|---|---|---|---|---|---|
| **SD 1.x** | 2022 | [[U-Net]], ~0.9B | ε-prediction diffusion | CLIP ViT-L | 4-ch, 8× |
| **SD 2.x** | 2022 | U-Net | v-prediction | OpenCLIP | 4-ch |
| **SDXL** | 2023 | bigger U-Net, 2.6B + a refiner | ε-prediction | two CLIPs | 4-ch |
| **SD3** | 2024 | **MM-DiT** (transformer) | **[[Rectified Flow\|rectified flow]]** | 2× CLIP + T5 | **16-ch** |
| **Flux** | 2024 | DiT, 12B | rectified flow | CLIP + T5 | 16-ch |

Three trends in one table: **U-Net → transformer** ([[Diffusion Transformer]]), **diffusion → [[Flow Matching|flow matching]]**, and **more latent channels**.

---
# What the compression costs you

> [!WARNING] The VAE sets a ceiling nothing downstream can raise
> The diffusion model can only produce latents; the decoder has the last word on pixels. Whatever the autoencoder can't reconstruct, the system can't generate. With a 4-channel, 8× VAE that means **small text, distant faces, fine patterns, thin lines** — each 8×8 pixel block is summarised by 4 numbers. Many of Stable Diffusion's notorious artefacts are *decoder* artefacts. This is exactly why SD3 and Flux moved to **16 channels**: less compression, better detail, more compute. ^vae-is-the-ceiling

> [!TIP] Practical consequences
> - **Wrong-VAE bugs.** A washed-out or oversaturated result is often a VAE mismatch, or fp16 overflow in the decoder — not the diffusion model.
> - **The latent isn't semantic like a GAN's $z$.** It's a *spatial* grid — a small, abstract image. That's why [[Controlling Diffusion|inpainting and img2img]] work naturally: you can mask and blend regions of the latent.
> - **Resolution is tied to training.** Generate far from the trained size and you get duplicated subjects — the denoiser's receptive field and positional cues were learned at one scale.
> - **Noise schedule must scale with resolution** → [[Forward Diffusion Process#The schedule — where the model spends its effort|shifted schedules]].

> [!TIP] The shape that transfers
> "Compress with one model, do the expensive modelling in the compressed space, decode at the end" is everywhere: [[VQ-VAE|tokenizers]] + a transformer for images and audio; world models that plan in a learned latent ([[Model-Based vs Model-Free RL|Dreamer]]); and, loosely, an LLM working on [[Tokenization|tokens]] rather than bytes. It's also [[Video Diffusion]]'s only hope — video latents compress in *time* as well as space.

---
> [!SUCCESS] If you remember one thing
> **Autoencoder for the pixels, diffusion for the meaning.** ~={pink}A 48× smaller space is what made diffusion affordable — and the autoencoder's fidelity is the hard ceiling on what it can draw.=~

---
# ⁉️
Row 3 of that pipeline — the denoiser — is where all the parameters and all the compute live. It has an unusual job: take in an image-shaped thing, put out an image-shaped thing of the *same size*, while understanding both the global composition and the finest detail. For years, one architecture owned that job.

→ [[U-Net]]
