---
aliases:
  - UNet
  - U Net
  - Unet
  - Encoder-Decoder with Skip Connections
  - Denoising U-Net
  - Denoiser Backbone
tags:
  - generative-models
  - deep-learning
  - architecture
  - diffusion
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** An hourglass — shrink the image step by step to understand **what** it is, then grow it back step by step to decide **where** everything goes — with **bridges** carrying the fine detail straight across from each shrinking stage to its matching growing stage.
> **Metaphor:** Editing a photo by zooming out to judge the composition, zooming in to fix pixels — while keeping the full-resolution original open in another window.
> **Where it bites:** The denoiser in DDPM, Imagen, Stable Diffusion 1–2 and SDXL. Also most of medical imaging and segmentation. Now being displaced by transformers.

---
A diffusion denoiser has an awkward job description. **Input:** a noisy image. **Output:** an image *of exactly the same size* (the predicted noise).

And to get that output right, it needs two kinds of knowledge at once:

- **Global** — *this is a face; the eyes go here; the light comes from the left.* You can only see that by stepping back.
- **Local** — *this exact pixel is an eyelash edge.* You can only get that by looking closely.

A plain CNN at full resolution has a tiny receptive field — it never sees the whole face. An [[Autoencoder|autoencoder]]-style bottleneck sees the whole face — and throws the eyelash away.

~={blue}How do you build one network that steps all the way back *and* keeps every pixel in view?=~

---
# Zoom out, zoom in — and keep the original open 🔍

![[unet_architecture.png]]

1. **Down path (encoder).** Convolve, then halve the resolution. Repeat 3–4 times: 64×64 → 32 → 16 → 8. Each level has fewer positions, more channels, a wider view. By the bottom, every position "sees" the entire image.
2. **Bottleneck.** The smallest, most abstract grid. *What* is in the picture.
3. **Up path (decoder).** Upsample, convolve. 8 → 16 → 32 → 64. *Where* things go.
4. **Skip connections — the point of the whole design.** At each level, the down-path's feature map is **copied across and concatenated** onto the up-path's at the same resolution.

Without step 4, the decoder would have to reinvent every edge from an 8×8 summary — and you'd get autoencoder blur. With it, the decoder gets the abstract plan from below **and** the original fine detail from the side.

> [!NOTE] U-Net
> A fully-convolutional encoder–decoder with skip connections between layers of equal spatial resolution (Ronneberger, Fischer & Brox, 2015 — for biomedical image segmentation). Drawn as a "U": down the left, up the right, bridges across. ^unet-def

> [!SUCCESS] Core idea
> ~={pink}Multi-scale processing, with a shortcut for detail at every scale.=~ The bottleneck carries *semantics*; the skips carry *pixels*. It's the natural architecture for any image-in → same-size-image-out task — segmentation, denoising, super-resolution, depth — which is exactly what a diffusion step is. And the skips are [[Deep Learning#CNNs — exploit locality|ResNet's idea]] applied across the U: a short path for both information *and* gradients. ^multi-scale-with-shortcuts

---
# What diffusion bolted on

The 2015 U-Net just segmented cells. The diffusion version is the same skeleton with three additions:

| Addition | Why | How |
|---|---|---|
| **Timestep conditioning** | the same pixels mean different things at different noise levels | sinusoidal embedding of $t$ → MLP → added to (or used to scale/shift) the features in **every** residual block → [[Conditional Generation#Three ways to get the condition in\|modulation]] |
| **Self-attention** | convolutions are local; "the left eye should match the right eye" is not | attention layers at the **low-resolution** levels (16×16, 8×8) — where the $O(n^2)$ cost is affordable → [[Query, Key, and Value (QKV)]] |
| **Cross-attention to text** | the prompt has to get in somewhere | at those same levels: image features are queries, prompt tokens are keys/values |

So a Stable Diffusion U-Net is really a **hybrid**: ResNet blocks for local texture, transformer blocks at the coarse scales for global structure and text.

---
# Why it was the right choice — and why it's being replaced

| | **U-Net** | **[[Diffusion Transformer\|DiT]]** |
|---|---|---|
| Inductive bias | strong — locality, translation equivariance, multi-scale, built in | almost none — learns it all from data |
| Data efficiency | **good** — works at modest scale | needs a lot |
| Compute per image | efficient: most work happens at low resolution | attention over all patches at every layer |
| Scaling behaviour | gains flatten; architecture has many hand-tuned choices (channels per level, where to put attention…) | **clean, predictable scaling** with parameters and compute |
| Flexible resolutions / aspect ratios / video | awkward | natural — it's just a longer token sequence |
| Shares tooling with LLMs | no | **yes** — same kernels, same parallelism, same [[Flash Attention]] |

> [!TIP] It's [[The Bitter Lesson (essay)|the bitter lesson]], again
> The U-Net encodes good human ideas about images. At small scale those ideas are a gift. At large scale they're a ceiling: a general architecture that scales cleanly overtakes a clever one that doesn't. The same story as CNN → [[An Image is Worth 16x16 Words (ViT)|ViT]] in recognition, five years earlier. The U-Net's **long skip connections** were the one idea worth keeping — U-ViT and several DiT variants put them back.

---
# Practical notes 🛠️

> [!WARNING] Things that bite
> - **Input size must divide by $2^{\text{levels}}$.** Three downsamplings → height and width must be multiples of 8 (and ×8 again for the VAE → multiples of 64 in pixel terms). Otherwise the skip tensors don't line up.
> - **Attention dominates memory at high resolution.** Attention at the 64×64 level is 4,096 tokens → a 16M-entry matrix per head. This is where [[Flash Attention]] and attention slicing earn their keep.
> - **Most of a LoRA's effect lives in the attention layers** — that's where they're usually attached → [[Controlling Diffusion]], [[LoRA]].
> - **The skips are what [[Controlling Diffusion|ControlNet]] hijacks** — it feeds its control signal into the decoder through them. ^unet-gotchas

And it isn't going away elsewhere: segmentation, medical imaging, depth, weather models, audio source separation — anywhere the output is a dense map aligned with the input and data is limited, the U-Net is still the default.

---
> [!SUCCESS] If you remember one thing
> **Down for *what*, up for *where*, bridges for *detail*.** ~={pink}The perfect shape for image-in, image-out — and a bundle of built-in assumptions that a plain transformer, given enough data, does better without.=~

---
# ⁉️
Strip out the convolutions, the pyramid and the hand-placed attention. Cut the latent into patches, call them tokens, and hand them to the same kind of transformer that runs language models. Does that really work better?

→ [[Diffusion Transformer]]
