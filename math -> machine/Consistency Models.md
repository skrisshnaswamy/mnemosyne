---
aliases:
  - Consistency Model
  - Consistency Distillation
  - Consistency Training
  - Latent Consistency Models
  - LCM
  - LCM-LoRA
  - Progressive Distillation
  - Diffusion Distillation
  - Few-Step Generation
  - One-Step Generation
  - Adversarial Diffusion Distillation
  - SDXL Turbo
  - Distribution Matching Distillation
tags:
  - generative-models
  - diffusion
  - inference
  - performance
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Train a network that maps **any point on a sampling trajectory straight to that trajectory's end**. The training signal: *two neighbouring points on the same path must give the same answer.* Result: 1–4 step generation.
> **Metaphor:** A water slide. Wherever you are on it, there's exactly one pool you'll land in — so learn to name the pool from any point on the slide.
> **Where it bites:** LCM, LCM-LoRA, SDXL-Turbo, Lightning, "Schnell" — every real-time image generator. It's how diffusion's one great weakness, speed, got fixed.

---
A diffusion model's deterministic sampler ([[Score SDE#The twin: a deterministic ODE|the probability-flow ODE]]) has a property worth staring at: **each starting noise leads to exactly one image**, along one fixed curve.

Sampling means walking that curve in 20–50 small steps, each a full pass through a multi-billion-parameter network.

But think about what's actually being computed. Pick any point on the curve — the start, the middle, 90% of the way along. All of them lead to the *same* final image. The walk is just an expensive way of finding out which one.

~={blue}What if a network could look at any point on the curve and simply tell you where the curve ends?=~

---
# The water slide 🛝

![[consistency_model_jump_to_end.png]]

Wherever you are on a water slide, the pool you'll end up in is already determined. A **consistency model** $f_\theta(x_t, t)$ learns to name the pool:

$$f_\theta(x_t, t) = x_0 \quad \text{for every point } (x_t, t) \text{ on the same trajectory}$$

Generation is then: draw noise, call $f_\theta$ **once**. Done.

How do you train that without knowing the answers? You don't need them. You only need the network to **agree with itself**:

1. Take a real image, noise it to level $t_{n+1}$ → point $A$.
2. Take **one** ODE step back toward the data (using a pretrained diffusion model as teacher) → point $B$ at level $t_n$. $A$ and $B$ are on the same slide.
3. Loss: $\;\big\| f_\theta(A, t_{n+1}) - f_{\theta^-}(B, t_n) \big\|$ — *make the two predictions match.*
4. Anchor the end: at $t \approx 0$ the function must be the identity, $f(x, 0) = x$.

Agreement between neighbours, propagated along the whole curve from an anchored end, forces every point to predict the true end-point.

> [!NOTE] Consistency model
> A network trained to be **self-consistent** along probability-flow ODE trajectories, so that it maps any noisy point directly to the trajectory's origin. Trained by **consistency distillation** (from a teacher diffusion model) or **consistency training** (from scratch). Song, Dhariwal, Chen & Sutskever, 2023. ^consistency-def

> [!SUCCESS] Core idea
> ~={pink}Don't learn the step. Learn the destination.=~ A diffusion model is a velocity — *which way now?* A consistency model is the **integral** of that velocity — *where does this end up?* If the shape feels familiar, it should: a target computed by a slowly-updated copy of yourself ($\theta^-$), one step further along, is **bootstrapping** — the same structure as [[Temporal Difference Learning|TD learning]] and [[DQN#^make-it-hold-still|DQN's target network]]. "My prediction here should equal my prediction one step later." ^learn-the-destination

---
# One step — or a few

One-step samples are fast but a little soft. The fix keeps the speed:

> **Multistep consistency sampling:** jump to the end → add *some* noise back (to a lower level than before) → jump again. Two to four rounds.

Each jump corrects the previous one's errors. **4 steps** is the usual sweet spot — roughly 10× fewer network calls than a standard sampler, for nearly the same quality.

---
# The family of "make diffusion fast by distillation"

| Method | Idea | Steps |
|---|---|---|
| **Progressive distillation** | train a student to do in **1** step what the teacher does in **2**; repeat: 1024 → 512 → … → 4 | 4–8 |
| **Consistency models / LCM** | self-consistency along the trajectory | 1–4 |
| **LCM-LoRA** | the consistency "speed-up" packaged as a small [[LoRA]] — drop it onto *any* fine-tune of the same base model | 4–8 |
| **Adversarial distillation** (SDXL-Turbo, Lightning) | add a [[Generative Adverserial Network\|GAN]] critic so 1-step outputs stay **sharp** | 1–4 |
| **Distribution matching** (DMD) | match the student's output *distribution* to the teacher's, using both models' [[Score Function\|scores]] | 1 |
| **[[Rectified Flow\|Reflow]]** | straighten the path so one Euler step is enough | 1–8 |

Notice the GAN coming back in through the side door. Its weakness was always *training from scratch* — unstable, mode-dropping. Used as a **finishing loss** on a student that's already been shown the whole distribution by a diffusion teacher, it contributes exactly what it's good at — crispness — without being asked to do what it's bad at.

---
# What you give up

> [!WARNING] Speed isn't free
> - **Diversity drops.** Distilled few-step models produce visibly less varied outputs for the same prompt — the student concentrates on the teacher's most typical results → [[Mode Collapse]].
> - **[[Classifier-Free Guidance|Guidance]] gets baked in.** Most distilled models are trained at a fixed guidance scale; the CFG dial stops working (or must be set ≈ 1). You lose a control you were used to having.
> - **A little quality.** Fine detail and prompt adherence usually trail the full multi-step teacher.
> - **It's a second training job** on top of an already-trained model, needing the teacher in the loop.
> - **Some downstream tools break** — anything relying on a long trajectory (certain editing and inversion methods) has less to work with. ^distillation-tradeoffs

> [!TIP] It's the image-side twin of LLM inference tricks
> [[Distillation]] into a faster student, [[Speculative Decoding]] (cheap draft, expensive verify), [[Quantization]] — different mechanics, same economics: generation is a **loop around a big network**, so the wins are *fewer iterations* or *cheaper iterations*. And it's the mirror image of [[Test-Time Compute]]: there you *spend* more inference compute for quality; here you *amortise* it into training to save inference.

---
> [!SUCCESS] If you remember one thing
> **Every point on a sampling path leads to one image — so train a network to name that image from anywhere on the path.** ~={pink}That collapses 50 steps into 1–4, at some cost in diversity and control.=~

---
# ⁉️
Everything so far generates *an* image — whatever the dice produce. Nobody wants a random image. They want *"a corgi wearing a top hat, watercolour"*. How do you get a say in what comes out?

→ [[Conditional Generation]]
