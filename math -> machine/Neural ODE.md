---
aliases:
  - Neural ODEs
  - Neural Ordinary Differential Equation
  - Continuous Normalizing Flow
  - Continuous Normalizing Flows
  - CNF
  - Velocity Field
  - Vector Field
  - ODE Solver
  - Adjoint Method
  - FFJORD
tags:
  - generative-models
  - deep-learning
  - diffusion
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Instead of a stack of layers, let a network define a **velocity** — "a point at $x$ at time $t$ should move *this* way" — and get the output by **integrating** that velocity over time. A flow with infinitely many, infinitely thin layers.
> **Metaphor:** A map of river currents. Drop a leaf anywhere and the map tells you where it drifts to.
> **Where it bites:** The object that flow matching trains and that diffusion's fast samplers solve. Also the reason "number of sampling steps" is really "how accurately do I solve an ODE".

---
A residual network computes $h_{k+1} = h_k + f_k(h_k)$ — *"take where you are, and add a small correction."* → [[Deep Learning#CNNs — exploit locality|ResNets]].

Look at that with a physicist's eyes. *Position now = position before + something.* That's one step of **Euler's method** for solving a differential equation, with a step size of 1:

$$h_{k+1} = h_k + \Delta t \cdot f(h_k, t_k)$$

~={blue}So a ResNet is a crude numerical solution of some ODE. What happens if you take that literally — shrink the step to zero and let the *equation* be the model?=~

---
# The map of currents 🍃

You get one network, $v_\theta(x, t)$, that doesn't transform its input at all. It answers a different question: **"if a point is at $x$ at time $t$, which way and how fast should it be moving?"**

$$\frac{dx}{dt} = v_\theta(x, t)$$

To compute the output, drop your input in at $t = 0$ and **follow the current** to $t = 1$, using any off-the-shelf ODE solver.

> [!NOTE] Neural ODE
> A model whose forward pass is the solution of an ODE $\dot{x} = v_\theta(x,t)$ with a neural-network velocity field (Chen et al., 2018). Depth becomes continuous; the number of "layers" is whatever the solver decides to use. ^neural-ode-def

Three properties fall out immediately, and they're exactly the ones generative modelling wanted:

| Property | Why | Why it matters |
|---|---|---|
| **Invertible, automatically** | run time backwards: integrate $-v$ from 1 to 0 | [[Normalizing Flows]] had to *handcuff* the architecture to get this. Here $v_\theta$ can be **any** network |
| **Paths never cross** | an ODE with a smooth velocity has a unique solution through each point | the map noise → data is one-to-one: each noise vector is the ID of exactly one image |
| **Density is trackable** | $\frac{d}{dt}\log p(x_t) = -\nabla \cdot v_\theta$ — the *instantaneous* change of variables | exact likelihoods, without an $O(d^3)$ determinant |

Use it as a generative model — noise at $t=0$, data at $t=1$ — and it's a **continuous normalizing flow (CNF)**.

> [!SUCCESS] Core idea
> ~={pink}Don't learn the map from noise to data. Learn the *velocity* that carries one to the other.=~ A velocity field is a far gentler thing to learn than a one-shot transformation — it only ever has to say "move a little this way" — and invertibility comes for free. It's the same trade diffusion made: many easy local moves instead of one impossible global one. ^learn-the-velocity

---
# Why it didn't take over in 2018

The original training recipe was **maximum likelihood**: to compute the loss for *one* training image you must solve the ODE all the way through, tracking the density — and then backpropagate through the solver (or use the *adjoint method*, which solves a second ODE backwards to save memory).

> [!WARNING] Simulation-based training doesn't scale
> Dozens to hundreds of network evaluations **per training example, per gradient step**. And as training proceeds, the learned flow tends to get *stiffer* and needs ever more solver steps. Fine for small tabular problems; hopeless for images. CNFs sat on the shelf for years for exactly this reason. ^simulation-is-expensive

Compare diffusion's [[Denoising Objective|training loop]]: **one** forward pass per example, no simulation, a regression target you can compute in closed form. That gap is the whole story.

---
# What rescued it

Two realisations, arriving from opposite directions:

1. **Diffusion already *is* one.** The deterministic sampler of a diffusion model is an ODE ([[Score SDE#The twin: a deterministic ODE|the probability-flow ODE]]) — so every diffusion model was secretly a CNF, trained by a cheap, simulation-free method.
2. **So train CNFs that way on purpose.** If you *prescribe* the path each point should follow from noise to data, the correct velocity along it is known in closed form — and you can regress onto it directly. No solver in the training loop at all. → [[Flow Matching]]

---
# Sampling = solving the ODE

This reframes a very practical question. "How many steps?" is just **"how accurately am I integrating?"**

| Solver | Network calls per step | Note |
|---|---|---|
| **Euler** | 1 | first-order; fine if the path is nearly straight |
| **Heun / midpoint** | 2 | second-order; fewer, more expensive steps |
| **RK4** | 4 | rarely worth it for generation |
| **Adaptive** (Dormand–Prince) | varies | chooses its own step size; good for likelihoods |
| **Multistep** (DPM-Solver++) | 1 | reuses *previous* evaluations — the usual winner → [[Diffusion Sampling]] |

And it makes the key geometric point obvious: ~={blue}**the straighter the trajectories, the fewer steps any solver needs.**=~ A perfectly straight, constant-speed path is integrated *exactly* by a single Euler step. That is the entire motivation for [[Rectified Flow]].

> [!TIP] The same idea, elsewhere
> - **Control:** $\dot{x} = f(x, u)$ *is* a [[State-Space Model]]; a neural ODE is a learned, autonomous one — and [[System Identification]] of continuous dynamics is fitting one.
> - **Sequence models:** S4 / Mamba discretise a linear ODE → [[Transfer Function]].
> - **Physics-informed models** and irregularly-sampled time series, where "depth = time" is literal.

---
> [!SUCCESS] If you remember one thing
> **A network that outputs a velocity; integrate it to get from noise to data.** ~={pink}Free invertibility, any architecture — originally crippled by needing to simulate the ODE during training, and rescued by a way of training it without doing so.=~

---
# ⁉️
So: can you train a velocity field with a plain regression loss, never once solving the ODE during training — and, since you get to *choose* the path from noise to data, choose the simplest one imaginable?

→ [[Flow Matching]]
