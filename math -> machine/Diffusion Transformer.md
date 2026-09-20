---
aliases:
  - DiT
  - Diffusion Transformers
  - MM-DiT
  - MMDiT
  - Multimodal Diffusion Transformer
  - Patchify
  - Image Patches
  - adaLN-Zero
  - U-ViT
  - Scalable Diffusion Models with Transformers
tags:
  - generative-models
  - diffusion
  - transformers
  - architecture
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Cut the (latent) image into **patches**, treat each patch as a **token**, and denoise with a plain **transformer** — the timestep and class injected through the layer norms, the text through attention.
> **Metaphor:** A jigsaw where every piece can talk to every other piece, at every layer, from the start.
> **Where it bites:** SD3, Flux, Sora, and essentially every frontier image and video model since 2023. It's the point where image generation started riding the same scaling curve — and the same infrastructure — as LLMs.

---
The [[U-Net]] is full of good ideas about images: locality, a resolution pyramid, skip connections, attention only where it's affordable. Each is a human decision — and each is a knob someone had to tune.

Meanwhile, in language, the lesson of the previous five years was blunt: take the *plainest* architecture that scales, and scale it → [[Foundation Models#Scale is not a detail — it's the mechanism|scale is the mechanism]].

In recognition, [[An Image is Worth 16x16 Words (ViT)|ViT]] had already shown a transformer with no convolutions beats CNNs — *given enough data*.

~={blue}Does the same hold for generation? If you delete every image-specific idea from the denoiser, does it get worse — or does it finally start to scale?=~

---
# The talking jigsaw 🧩

![[dit_block.png]]

1. **Patchify.** Take the 64×64×4 latent from [[Latent Diffusion]]. Cut it into 2×2 patches → a sequence of **1,024 tokens**. Linearly project each; add positional embeddings.
2. **Transformer blocks.** Standard ones — self-attention + MLP. Every patch attends to every other patch **at every layer**. No pyramid, no downsampling.
3. **Unpatchify.** Project each token back to a 2×2×4 patch; reassemble; that's the predicted noise (or velocity).

The only non-standard part is how the conditions get in:

> [!NOTE] adaLN-Zero
> The timestep and class/pooled-text embedding are fed to a small MLP that outputs a **scale and shift for every LayerNorm** in the block, plus a gate on each residual branch — [[Conditional Generation#Three ways to get the condition in|modulation]]. The gates are **initialised to zero**, so every block starts life as the identity function. Peebles & Xie found this beat cross-attention and in-context conditioning, and the zero-init makes deep models train stably. ^adaln-zero

> [!NOTE] Diffusion Transformer (DiT)
> A diffusion denoiser built from a standard transformer operating on a sequence of latent patches (Peebles & Xie, 2023 — *"Scalable Diffusion Models with Transformers"*). ^dit-def

> [!SUCCESS] Core idea
> ~={pink}The result of the paper isn't an architecture — it's a scaling law.=~ Sample quality (FID) improved *smoothly and predictably* with the transformer's compute: more layers, wider layers, **smaller patches** (= more tokens). No plateau. That's the property the U-Net lacked, and once image generation had it, it inherited a decade of LLM engineering — [[Flash Attention]], [[Distributed Training|tensor and sequence parallelism]], [[Mixed Precision training|bf16]], even [[Mixture of Experts|MoE]]. ^dit-is-a-scaling-result

---
# Patch size is the compute dial

Halve the patch side and you quadruple the tokens — and attention cost grows with the *square* of that.

| Latent 64×64, patch… | Tokens | Relative attention cost | Quality |
|---|---|---|---|
| 8×8 | 64 | 1× | poor |
| 4×4 | 256 | 16× | OK |
| **2×2** | **1,024** | **256×** | best — the usual choice |

Same parameter count in every row; wildly different FLOPs. ~={blue}In a DiT, quality tracks **compute spent per image**, not parameter count=~ — which is also why resolution is so expensive: a 1024² image is 4,096 tokens, and video multiplies that by the number of frames.

---
# MM-DiT — how text gets in (SD3, Flux)

DiT-the-paper was class-conditional. For text there were two obvious options — cross-attention (as in the U-Net), or something more symmetric:

| | Cross-attention | **MM-DiT — joint attention** |
|---|---|---|
| Text tokens | fixed; the image *reads* them | concatenated with image tokens into **one sequence** |
| Information flow | text → image only | **both ways** — text representations are updated too |
| Weights | shared | **separate** weight sets per modality (text and image are statistically very different), but a **shared** attention operation |
| Effect | fine | better prompt following, typography, composition |

Add [[Flow Matching|rectified-flow]] training, a 16-channel VAE, a T5 encoder alongside [[CLIP]], and RoPE-style positions for flexible aspect ratios → [[RoPE]], and you have the SD3 / Flux recipe.

---
# The same trade as ViT

> [!WARNING] A transformer knows nothing about images
> No locality, no translation equivariance, no notion that neighbouring patches are related — only positional embeddings, and whatever it learns. With little data, a DiT **loses** to a U-Net. Its advantage appears only at scale, and it's paid for in compute: full attention across all patches at every layer is far more expensive than a U-Net doing most of its work at 8×8. ~={red}It isn't a better architecture for images. It's a better architecture for *scale*.=~ → [[The Bitter Lesson (essay)]] ^dit-needs-scale

> [!TIP] Why video made this inevitable
> A clip is a 3-D grid: time × height × width. "Patchify" generalises with no new ideas — cut **spacetime patches**, get a (long) token sequence, run the same transformer. Different resolutions, durations and aspect ratios are just different sequence lengths. A U-Net needs surgery for each of those. → [[Video Diffusion]]

And it closes a loop in the vault: the denoiser is now *the same object* as an LLM's backbone — the differences are that attention is **bidirectional** (no [[Causal Attention|causal mask]] — every patch sees every patch), the output is a continuous vector per token rather than a distribution over a vocabulary, and the network is called in a **loop over noise levels** rather than a loop over positions.

---
> [!SUCCESS] If you remember one thing
> **Patches as tokens, a plain transformer, conditions through the layer norms.** ~={pink}Worse than a U-Net when small, better when large — and its real gift was putting image generation on the LLM scaling curve and the LLM toolchain.=~

---
# ⁉️
A prompt gives you a say in *what* appears. It gives you almost none in *where*, in what *pose*, with which *exact face*, or how to change *just that corner* of an existing photo. Words are a blunt instrument for spatial intent.

→ [[Controlling Diffusion]]
