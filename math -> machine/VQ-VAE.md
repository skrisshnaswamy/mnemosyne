---
aliases:
  - VQVAE
  - Vector Quantized VAE
  - Vector-Quantised VAE
  - Vector Quantization
  - Vector Quantisation
  - Codebook
  - Image Tokenizer
  - Visual Tokenizer
  - VQGAN
  - Discrete Latents
  - Straight-Through Estimator
tags:
  - generative-models
  - deep-learning
  - tokenization
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** An autoencoder whose bottleneck must be built from a **fixed codebook** — every patch of the image is replaced by the ID of its nearest codebook entry. An image becomes a grid of integers: **tokens**.
> **Metaphor:** A mosaic. You can make any picture you like — but only from the 8,192 tile designs in the box.
> **Where it bites:** Image, audio and video **tokenizers**. It's how you feed pictures to a transformer — DALL·E 1, Parti, MaskGIT, every "unified multimodal" model.

---
Language models are the most successful generative models ever built, and they have one hard requirement: the data must be a **sequence of discrete tokens** from a finite vocabulary → [[Tokenization]].

Text complies. An image doesn't. A pixel is three continuous numbers, and a 256×256 image has 65,536 of them — far too long a sequence, and with no "vocabulary" in sight.

~={blue}How would you turn a photograph into a few hundred integers — in a way that lets you turn those integers back into the photograph?=~

---
# The mosaic 🧩

![[vq_vae_codebook.png]]

A mosaic artist can depict anything, but owns a fixed box of tile designs. Each spot on the wall gets *the tile that best matches what should be there*. The finished picture can be written down as a grid of tile numbers — and anyone with the same box can rebuild it.

1. **Encode.** A CNN shrinks the image to a small grid — say 32×32 — of continuous vectors.
2. **Quantise.** Replace each vector with the **nearest entry in a learned codebook** of $K$ vectors (512 … 16,384). Keep only its *index*.
3. **Decode.** Look the indices back up, and a decoder CNN rebuilds the image.

A 256×256×3 image (196,608 numbers) → a 32×32 grid of integers (**1,024 tokens**) from a vocabulary of, say, 8,192.

> [!NOTE] VQ-VAE
> **V**ector-**Q**uantised VAE: an autoencoder with a discrete bottleneck, where each encoder output is snapped to its nearest neighbour in a learned codebook. The codebook indices are a discrete latent representation — **image tokens**. (van den Oord et al., 2017.) ^vqvae-def

> [!SUCCESS] Core idea
> ~={pink}It's BPE for pixels.=~ A learned, finite vocabulary of visual "words", so that an image becomes a sentence — and then *any* sequence model can generate images, because it's just predicting the next token. Two-stage generation: **stage 1** learn the tokenizer; **stage 2** learn a prior over token grids ([[Auto-regressive models|autoregressive]], masked, or [[Discrete Diffusion|discrete diffusion]]). ^bpe-for-pixels

---
# The training problem: you can't differentiate "nearest"

Picking the nearest codebook entry is an $\arg\min$ — a hard, discrete choice. Its gradient is zero almost everywhere. [[Backpropagation]] stops dead at the bottleneck.

> [!NOTE] The straight-through estimator
> On the **forward** pass, quantise properly. On the **backward** pass, *pretend the quantisation wasn't there* — copy the decoder's gradient straight across to the encoder, as if the step were the identity. Crude, biased, and it works. ^straight-through

The loss has three parts: **reconstruct** the image; pull each **codebook** vector toward the encoder outputs assigned to it; and a **commitment** term that stops the encoder's outputs drifting away from the codebook.

> [!WARNING] Codebook collapse
> A common failure: only a few hundred of the 8,192 entries ever get used; the rest are dead. It's a rich-get-richer loop — an entry that's never chosen never moves, so it never becomes the nearest. Fixes: restart dead codes from live encoder outputs, EMA codebook updates, lower-dimensional codes, or lookup-free quantisers (**FSQ**, LFQ). ~={red}Always check codebook utilisation.=~ ^codebook-collapse

---
# VQGAN — the version people actually use

A plain VQ-VAE is trained with a pixel-wise squared error — which, as ever, gives **blurry** reconstructions ([[Variational Autoencoder#^why-vaes-blur|same reason as a VAE]]). **VQGAN** adds a perceptual loss and an adversarial critic ([[Generative Adverserial Network#^learned-loss|a learned loss]]), so the decoder produces *crisp* textures even at 16× compression. Nearly every modern image tokenizer is a descendant.

---
# Continuous vs discrete latents — the fork

| | Continuous latent ([[Variational Autoencoder\|VAE]]) | Discrete latent (**VQ**) |
|---|---|---|
| What the image becomes | a small grid of floats | a grid of **integers** |
| Natural generator on top | [[Diffusion Models\|diffusion]] / [[Flow Matching\|flow]] | [[Auto-regressive models\|autoregressive]] transformer, masked modelling, [[Discrete Diffusion]] |
| Reconstruction fidelity | higher | lower — quantisation loses information |
| Shares a vocabulary with text? | no | **yes** — one transformer, one token stream, text *and* images |
| Examples | Stable Diffusion, Flux | DALL·E 1, Parti, MaskGIT, Chameleon-style models |

That second-to-last row is why the discrete route keeps coming back: if images are tokens, a single LLM can read and write them. It's an active area — the vault has [[Studying Image Tokenizers as Visual Languages in Unified Multimodal Models]] and, for video, [[Keep-or-Drop- Adaptive Tokenizer for Compact Video Representation]].

> [!TIP] The same trick, elsewhere
> - **Audio:** SoundStream / EnCodec use *residual* VQ — quantise, then quantise the leftover error, several times. That's how speech becomes tokens for audio LMs.
> - **Recommenders:** *semantic IDs* are residual-VQ codes for items — see [[Recommender Systems - Evolution#RQ-VAE - Vector Quantization|RQ-VAE in the RecSys note]]. Same idea: replace an arbitrary ID with a short code that *means* something.
> - **Vector search:** product quantisation compresses embeddings the same way → [[Vector Database]].

---
> [!SUCCESS] If you remember one thing
> **Snap every patch to the nearest entry in a learned codebook → the image is now a grid of integers.** ~={pink}That's what lets a language model generate pictures — and the rest is making the snapping trainable and the codebook fully used.=~

---
# ⁉️
VAEs and their relatives are stable and cover the data — but their samples are soft. In 2014 someone proposed throwing away likelihood entirely and training the generator against an opponent instead.

→ [[Generative Adverserial Network]]
