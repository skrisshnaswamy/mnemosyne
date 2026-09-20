---
aliases:
  - Video Diffusion Models
  - Video Generation
  - Text-to-Video
  - Spacetime Patches
  - Temporal Consistency
  - Temporal Attention
  - Sora
  - Video World Models
  - Generative World Models
  - Diffusion Forcing
tags:
  - generative-models
  - diffusion
  - multimodal
  - world-models
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Treat a clip as **one 3-D block** (time × height × width), compress it in space *and* time, cut it into **spacetime patches**, and denoise the whole block together — so the frames are generated **jointly** and therefore agree.
> **Metaphor:** A flip-book drawn by one artist in one sitting — not a hundred artists each handed one page.
> **Where it bites:** Sora, Veo, Movie Gen, Runway. And "world models": once a video model can be *conditioned on actions*, it's a learned simulator.

---
You have an excellent image generator. You want a four-second clip — 96 frames.

Obvious plan: prompt it 96 times. *"A corgi running on a beach, frame 1"… "frame 2"…*

Play the result. The corgi's markings change every frame. The background flickers. The sun jumps around the sky. Each frame is beautiful. The video is unwatchable.

~={blue}Every frame is a good sample. What is it that the 96 frames *as a set* are missing — and where would it have to come from?=~

---
# One artist, one sitting 📖

**Consistency** — and it can only come from the frames being generated *together*. Sampled independently, each frame is a separate draw from "all plausible beach-corgi images". A video isn't 96 draws from that distribution. It's **one** draw from the distribution over *whole clips* — in which frame 17 is almost entirely determined by frame 16.

So model the whole block.

1. **Compress in space and time.** A video [[Latent Diffusion|VAE]] shrinks each frame ~8× per side *and* merges every ~4 frames into one latent "frame". Without this the token count is hopeless.
2. **Spacetime patches.** Cut the latent block into small 3-D bricks. Each brick is a token → [[Diffusion Transformer#^dit-def|DiT]].
3. **Denoise all bricks jointly.** Attention runs across space *and* across time, so the patch holding the corgi's ear in frame 40 can see the same ear in frame 39.

> [!NOTE] Video diffusion model
> A diffusion (or [[Flow Matching|flow-matching]]) model over a spatio-temporal tensor. The denoiser — a 3-D [[U-Net]] or, now, a transformer over spacetime patches — generates all frames of a clip **jointly**, conditioned on text, an image, or previous frames. ^video-diffusion-def

> [!SUCCESS] Core idea
> ~={pink}Temporal consistency isn't a post-process — it's the joint distribution.=~ Nothing in the loss says "keep the dog the same dog". The model learns it because in real video, neighbouring frames are nearly identical, and denoising a block is only easy if you exploit that. Object permanence, continuity of motion, rough physics — all show up as *side effects of compressing video well*. ^consistency-is-the-joint

---
# The brutal arithmetic

| | Tokens |
|---|---|
| One 512×512 image, in latent patches | ~1,024 |
| 5 s of 720p at 24 fps, after 8× spatial + 4× temporal compression and 2×2 patching | **~100,000+** |

Attention is quadratic: 100× the tokens is **10,000×** the attention compute. Every design decision in video generation is a response to that number:

| Trick | Idea |
|---|---|
| **Factorised attention** | alternate *spatial* attention (within a frame) and *temporal* attention (same position across frames) instead of full 3-D attention |
| **Aggressive temporal compression** | the VAE merges frames — most are near-duplicates of their neighbours anyway |
| **Cascades** | generate low-res, low-fps; then super-resolve and interpolate |
| **Image pre-training** | treat an image as a 1-frame video and train on both — images are plentiful, clean and cheap |
| **[[Flash Attention]], sequence parallelism** | the same tools as [[Long Context\|long-context LLMs]] — it *is* a long-context problem |
| **Few-step distillation** | → [[Consistency Models]] — even more valuable here than for images |

---
# Long videos — the autoregressive hybrid

A joint block has a fixed length. For a minute-long video you generate a chunk, then **condition the next chunk on the last few frames** ([[Conditional Generation]]), and roll forward.

> [!WARNING] Rolling forward reintroduces compounding error
> Each chunk is conditioned on the model's *own* previous output — small flaws accumulate, identities drift, scenes slowly morph. It's [[Imitation Learning#^errors-compound|exposure bias]] again, the same disease as autoregressive text and behaviour cloning. **Diffusion forcing** and related methods train with a *different noise level per frame* so the model learns to continue from imperfect context.

---
# From video to world model 🌍

Condition not just on text, but on **actions** — *"the agent presses left"*, *"the car steers 5° right"* — and ask for the next frames.

You've built a **learned simulator**: a model of $P(\text{next observation} \mid \text{history}, \text{action})$. That's the *model* in [[Model-Based vs Model-Free RL|model-based RL]] — learned from pixels. An agent can practise inside it ([[Model-Based vs Model-Free RL#Dyna — just do both|Dyna]], at scale), or plan through it → [[Model Predictive Control]].

The vault has a cluster of papers here: [[Mastering Diverse Domains through World Models (DreamerV3)]], [[Game2World Engine- Unlocking In-the-Wild Gameplay Videos for World Model Training]], [[PAWBench- How Far Are We from Probabilistically Aligned World Modeling]], [[Hydra-0- Action Flow for Generalist World Modeling and Control]].

> [!WARNING] "It learned physics"
> It learned **what physics usually looks like on camera**. That's a lot — and it isn't the same thing. Video models still produce objects that pass through each other, liquids that don't conserve volume, a glass that shatters before it lands. They model *appearance statistics*, not conserved quantities, and they have no mechanism to be *consistent* about something that leaves the frame and comes back — a [[State-Space Model|hidden state]] they never explicitly track. Plausible is not correct → the same gap as [[Hallucination]]. ^plausible-is-not-physics

> [!TIP] Whether pixels are even the right target is contested
> One camp says: generating every pixel of the future wastes nearly all the model's capacity on unpredictable detail — leaves, water, noise — that doesn't matter for understanding or acting. Predict the future **in representation space** instead → [[JEPA]].

Evaluation is its own problem: **FVD** ([[Evaluating Generative Models|FID for video]]) inherits all of FID's flaws and adds insensitivity to *temporal* errors.

---
> [!SUCCESS] If you remember one thing
> **Generate the clip as one block, so the frames are one sample, not many.** ~={pink}The idea is a small step from image diffusion; the cost is a quadratic blow-up in tokens — and conditioning on actions turns a video generator into a simulator.=~

---
# ⁉️
A world model predicts what you'll *see*. Turn it around: could the same machinery generate what you should *do* — a robot's next second of motion, as a sample from a learned distribution over good trajectories?

→ [[Diffusion Policy]]
