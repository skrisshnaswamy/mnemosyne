---
aliases:
  - Autoencoders
  - Auto-encoder
  - Auto-Encoder
  - AE
  - Encoder-Decoder Bottleneck
  - Bottleneck
  - Denoising Autoencoder
  - Reconstruction Loss
  - Masked Autoencoder
tags:
  - generative-models
  - deep-learning
  - fundamentals
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Train a network to **reproduce its own input** — but force it through a **narrow middle**. To get through, it has to discover what actually matters about the data.
> **Metaphor:** Describing a photo over the phone in twenty words, to someone who then has to redraw it.
> **Where it bites:** Compression, anomaly detection, pre-training — and the front and back end of every latent-diffusion system. It is **not** a generative model on its own, and the reason why is the whole motivation for the VAE.

---
Here's a strange training objective: *take an image in, output the same image.*

A network could do that trivially — pass every pixel straight through. Loss zero. Nothing learned.

Now add one constraint. Somewhere in the middle, the network must squeeze the image — 196,608 numbers — through a layer that's only **64 numbers wide**. Then rebuild the full image from those 64.

~={blue}It can't copy any more. What does it have to do instead?=~

---
# Twenty words down the phone 📞

![[autoencoder_bottleneck_hourglass.png]]

You're describing a photograph to a friend who'll redraw it, and you get twenty words. You won't spend them on *"pixel 4,012 is slightly beige"*. You'll say: *woman, forties, smiling, facing left, outdoors, evening light.* The constraint **forces** you to find what matters.

- The **encoder** squeezes $x$ down to a short code $z$.
- The **bottleneck** is where the squeeze happens.
- The **decoder** rebuilds $\hat{x}$ from $z$ alone.
- The loss is just *how different is $\hat{x}$ from $x$?* — the **reconstruction loss**.

> [!NOTE] Autoencoder
> A network trained to reconstruct its input through a constrained intermediate representation: $\hat{x} = \text{dec}(\text{enc}(x))$, minimising $\|x - \hat{x}\|^2$. No labels needed — the input is its own target. ^autoencoder-def

> [!SUCCESS] Core idea
> ~={pink}Compression is understanding.=~ To rebuild a face from 64 numbers, those 64 numbers had better be things like pose and lighting — nobody told the network that, the bottleneck made it inevitable. It's a learned, nonlinear PCA — and the first concrete way to get the slider settings that [[Latent Variable Models]] asked for. ^compression-is-understanding

---
# The family

| Variant | The constraint | Used for |
|---|---|---|
| **Undercomplete** | a narrow bottleneck | compression, features |
| **Denoising** | corrupt the input, reconstruct the *clean* version | robust features — and, conceptually, ~={blue}**the direct ancestor of diffusion**=~: a diffusion model is a denoising autoencoder trained at *every* noise level → [[Denoising Objective]] |
| **Sparse** | wide code, but most units must be off | interpretability (sparse autoencoders on LLM activations) |
| **Masked (MAE)** | hide 75% of the patches, reconstruct them | self-supervised pre-training for vision — BERT's trick, for images |
| **[[Variational Autoencoder\|Variational]]** | the code must look like $\mathcal{N}(0, I)$ | **generation** |
| **[[VQ-VAE\|Vector-quantised]]** | the code must come from a fixed codebook | image **tokenizers** |

Practical uses of the plain one: **anomaly detection** (train on normal data; anything it reconstructs badly is unusual), **dimensionality reduction**, **denoising**, and — the big one today — as the **compressor** in [[Latent Diffusion]].

---
# Why you can't generate with it

This is the point of the note. You've trained a lovely autoencoder on faces. You want a *new* face. Natural plan: pick a random $z$, run the decoder.

You get garbage. Why?

![[vae_latent_holes.png]]
> [!TIP] Reading the chart (left panel)
> Nothing in the loss said *where* codes should live or *how* they should be arranged. So the encoder drops them wherever is convenient — tight islands, in arbitrary places, with vast empty sea between. A random $z$ almost always lands in the sea, somewhere the decoder has **never been asked to decode**.

> [!WARNING] An autoencoder is a compressor, not a generator
> It learns a great map from *real image* → *code* → *same image*. It learns nothing about **which codes are valid**. There's no prior to sample from, no guarantee neighbouring codes decode to similar images, no guarantee the space between two codes means anything. ~={red}It has a latent space, but not one you can sample.=~ ^ae-is-not-generative

Fixing that needs two things: a **known distribution** the codes are forced to follow, and **smoothness**, so that every point near a valid code is also valid. Both come from one change.

---
> [!SUCCESS] If you remember one thing
> **Squeeze, then rebuild** — the bottleneck makes the network find what matters. ~={pink}But a plain autoencoder only knows how to decode codes it has seen; to *generate*, the latent space itself has to be given a shape.=~

---
# ⁉️
What if the encoder didn't output a *point*, but a little **cloud** — "somewhere around here" — and you penalised the clouds for straying from a standard bell curve?

→ [[Variational Autoencoder]]
