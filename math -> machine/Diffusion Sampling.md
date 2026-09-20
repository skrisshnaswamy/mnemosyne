---
aliases:
  - Reverse Process
  - Reverse Diffusion
  - Diffusion Sampler
  - Diffusion Samplers
  - Ancestral Sampling
  - DDIM
  - Denoising Diffusion Implicit Models
  - DPM-Solver
  - Euler Sampler
  - Sampling Steps
  - NFE
  - Predictor-Corrector
tags:
  - generative-models
  - diffusion
  - inference
  - performance
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Start from pure noise and loop: *ask the network what the clean image probably is → step a little of the way there → (optionally) add some fresh noise back*. The **sampler** is how you take those steps, and it decides whether you need 1,000 of them or 20.
> **Metaphor:** A sculptor working in passes — rough block, then form, then features, then polish. Never one cut.
> **Where it bites:** The "sampler" and "steps" dropdowns in every image tool — *Euler a, DDIM, DPM++ 2M Karras* — and the cost of inference, which is **steps × one full network pass**.

---
You've trained a [[Denoising Objective|denoiser]]. Hand it pure static and ask for the clean image in one shot.

You get a grey-brown blur — the average of every image it ever saw. It isn't broken: from pure static, *"the average of everything"* genuinely is the best single guess.

~={blue}So the network's one-shot answer is useless. How do you get a sharp image out of a model whose every individual guess is a blurry average?=~

---
# Working in passes 🗿

Don't take its guess. Take a **small step toward** its guess.

From static, the guess is mush — but it's mush that leans very slightly somewhere. Move 1% of the way. Now you hold something a hair less random, and from *here* the network's guess is slightly more specific. Move a little more. Each step commits to a bit more structure, and that commitment sharpens the next guess.

```python
x = randn(shape)                                  # pure static
for t in reversed(timesteps):                     # 1000 … 0, or a 20-step subset
    eps = model(x, t)                             # "which part of this is noise?"
    x0_hat = (x - sigma[t] * eps) / alpha[t]      # implied guess at the clean image (blurry early on)
    x = step_toward(x0_hat, x, t)                 # the sampler: move a little, maybe re-add noise
```

![[reverse_diffusion_2d.png]]

> [!NOTE] Sampling
> Numerically integrating the learned reverse process from $t = T$ (noise) to $t = 0$ (data). Each step costs one forward pass of the network — a **number of function evaluations (NFE)**. The rule used to take each step is the **sampler** (or solver, or scheduler). ^sampling-def

> [!SUCCESS] Core idea
> ~={pink}Blurry guesses, iterated, make a sharp image.=~ Composition gets fixed in the first few (high-noise) steps; detail in the last ones ([[Forward Diffusion Process#The schedule — where the model spends its effort|coarse-to-fine]]). And the key practical fact: **the sampler is chosen at inference time.** Same trained weights, any sampler, any step count — no retraining. ^iterate-blurry-guesses

---
# Two families of sampler

![[sde_vs_ode_paths.png]]
> [!TIP] Reading the chart
> Same start points, same model, same final distribution. **Left — stochastic:** after each denoising step, a bit of fresh noise is put back. Paths jitter and wander. **Right — deterministic:** no noise re-injected. Paths are smooth curves, and ~={blue}a given starting noise **always** produces the same image.=~

| | **Stochastic** (DDPM / ancestral, "Euler a") | **Deterministic** (DDIM, Euler, DPM-Solver) |
|---|---|---|
| Re-injects noise each step | yes | **no** |
| Mathematically | a reverse-time **SDE** | an **ODE** — the *probability-flow* ODE → [[Score SDE]] |
| Steps needed | many (hundreds → 1,000) | **few (10–50)** |
| Same seed → same image? | only with the same step count | **yes** — the noise *is* the image's ID |
| Self-corrects errors | ✅ fresh noise washes them out | ❌ errors accumulate |
| Enables | — | **inversion** (image → its noise), smooth latent **interpolation**, editing |

**DDIM** (Song, Meng & Ermon, 2020) was the breakthrough: the *same trained DDPM model*, sampled deterministically, gives good images in ~50 steps instead of 1,000. A 20× speed-up with no retraining.

---
# How few steps can you get away with?

Measured on a toy target, with the *exact* score so the only error is the sampler's:

![[sampling_steps_vs_error.png]]
> [!TIP] Reading the chart
> Distance from the true distribution vs number of network evaluations (lower is better; the dotted line is "as good as real samples"). The **stochastic** sampler is hopeless below ~16 steps (error 6.4 at one step, still 0.25 at eight). The **deterministic ODE** is ~3× better at every small budget (0.08 at eight). Everyone converges by ~64. The green line — [[Rectified Flow|straight paths]] — is at the floor **in one step**. ~={blue}How curved the path is decides how many steps you need.=~

Why curvature matters: a sampler takes straight-line steps. On a curved path, a big straight step cuts the corner and lands in the wrong place. Three ways to fight that:

| Strategy | Idea | Examples |
|---|---|---|
| **Better solvers** | use the *history* of previous steps to anticipate the curve (higher-order) | DPM-Solver++, UniPC, Heun — 10–25 steps |
| **Better step placement** | spend steps where the path bends most | "Karras" sigmas, EDM |
| **Straighter paths** | change training so there's less curve to follow | [[Flow Matching]], [[Rectified Flow]] — 4–30 steps |
| **Learn the shortcut** | distil the whole trajectory into a jump | [[Consistency Models]], LCM, Turbo — **1–4 steps** |

---
# The decoder ring for sampler names 🔎

| You see | It means |
|---|---|
| **Euler** | first-order ODE solver. Simple, deterministic, surprisingly good |
| **Euler a** | *ancestral* — Euler + fresh noise each step. Stochastic; never quite "converges" as you add steps |
| **DDIM** | the original deterministic sampler; ≈ Euler in the right variables |
| **DPM++ 2M** | second-order **m**ultistep — reuses the previous evaluation. The usual best default |
| **DPM++ SDE** | a stochastic, higher-order variant |
| **Heun** | second-order, **two** network calls per step |
| **…Karras** | not a sampler — a *step-spacing* schedule |
| **LCM / Turbo / Lightning** | not samplers — *distilled models* that need 1–8 steps → [[Consistency Models]] |

> [!WARNING] "More steps = better"
> Only up to a point, and only for deterministic samplers. Past ~30–50 steps a good solver has converged and extra steps buy nothing. With *ancestral* samplers the image keeps **changing** as you add steps, because you're adding different noise — it doesn't converge to anything. And with [[Classifier-Free Guidance|guidance]] on, every step costs **two** network passes, not one. ^more-steps-isnt-better

> [!TIP] Cost, in one line
> Latency ≈ **steps × (1 or 2 passes for guidance) × cost of one pass.** A 30-step guided sample is 60 forward passes of a multi-billion-parameter network. Compare an LLM, where [[Prefill and Decode|decode]] is one pass per token — diffusion's "tokens" are its steps, but each pass processes the *whole image* in parallel, so it's compute-bound rather than memory-bound.

---
> [!SUCCESS] If you remember one thing
> **Step a little toward a blurry guess; repeat.** ~={pink}Deterministic samplers need far fewer steps than stochastic ones, and the number you need is set by how *curved* the path from noise to data is=~ — which is the thread that leads to flow matching.

---
# ⁉️
DDPM and score matching gave us stochastic samplers; DDIM gave us a deterministic one that mysteriously produces the *same distribution*. There's a single framework in which all of them are the same object viewed from different angles.

→ [[Score SDE]]
