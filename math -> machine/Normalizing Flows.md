---
aliases:
  - Normalizing Flow
  - Normalising Flows
  - Normalising Flow
  - Flow-Based Models
  - Change of Variables
  - Invertible Neural Network
  - RealNVP
  - Glow
  - Coupling Layer
tags:
  - generative-models
  - probability
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Warp simple noise into data with a chain of **invertible** transformations, and keep track of how much each one **stretched space** — that bookkeeping gives you the *exact* density of any data point.
> **Metaphor:** A rubber sheet with ink dots evenly spread on it. Stretch a region and the dots there thin out; squeeze it and they crowd together.
> **Where it bites:** Exact likelihoods (RealNVP, Glow), and — more importantly now — the **ancestor of flow matching**: same idea, with the invertibility obtained for free from an ODE.

---
Take a number $z$ drawn from a standard bell curve. Pass it through the function

$$x = z + 1.6\,\tanh(2z)$$

Near $z = 0$ that function is **steep** — its slope is 4.2. Far from zero it's nearly flat-ish — slope close to 1.

~={blue}What does the distribution of $x$ look like? And — without sampling a single point — what is its exact density at $x = 0$?=~

---
# The rubber sheet 🧵

![[normalizing_flow_rubber_sheet.png]]

Picture the ink dots. Where the function is steep, neighbouring values of $z$ get flung far apart in $x$ — the dots **thin out**, density falls. Where it's flat, they stay close, and density holds up.

The middle of the bell curve — which used to be the *densest* part — gets stretched by a factor of 4.2 and becomes a **valley**. Two humps appear either side.

![[normalizing_flow_change_of_variables.png]]
> [!TIP] Reading the chart
> One-hump noise in, two-hump distribution out. The green curve isn't a fit — it's the **exact** density, computed as $p_z(z) \div |f'(z)|$. At the centre: $0.399 \div 4.2 = $ **0.095**. The 40,000 grey samples agree with it.

> [!NOTE] Change of variables
> If $x = f(z)$ with $f$ invertible, then $\;p_x(x) = p_z(z)\,\Big|\det \dfrac{\partial f}{\partial z}\Big|^{-1}$. The determinant of the [[Derivative#Jacobian|Jacobian]] is *how much volume got stretched*. Stack $K$ such layers and the log-densities just add: $\log p(x) = \log p(z) - \sum_k \log|\det J_k|$. ^change-of-variables

> [!SUCCESS] Core idea
> ~={pink}Probability is conserved, so density = original density ÷ how much you stretched.=~ Because $f$ is invertible, every image has exactly **one** code — the awful integral from [[Latent Variable Models]] disappears, and [[Maximum Likelihood]] can be done *exactly*. Same network gives you sampling (run it forwards) and likelihood (run it backwards). ^flows-core

---
# The price: the architecture is in handcuffs

For this to work, every layer must be (a) **invertible** and (b) have a **cheap Jacobian determinant**. A general $d \times d$ determinant costs $O(d^3)$ — for a million-pixel image, unthinkable.

The standard trick is the **coupling layer** (RealNVP): split the vector in half; leave the first half alone; shift and scale the second half by amounts *computed from the first half*.

- Trivially invertible — you still have the first half, so you can recompute the shift and undo it.
- The Jacobian is triangular, so its determinant is just the product of the scales. $O(d)$.
- The network that *computes* the shift and scale can be anything at all — it never needs inverting.

| | ✅ | ❌ |
|---|---|---|
| Likelihood | **exact** — rare and valuable | — |
| Sampling | one fast pass | — |
| Latent dimension | — | **must equal the data dimension.** No compression; a 1-megapixel image needs a 3-million-dim latent |
| Expressiveness | — | each layer can only do a constrained warp → you need *many*, and sample quality lagged GANs and diffusion |
| Topology | — | an invertible map can't tear space — turning one blob into two *separate* blobs needs an infinitely steep stretch |

That last row is visible in the chart: the two humps are still joined by a thin bridge. It can never quite reach zero.

---
# Why you still need to know this

> [!TIP] Flows didn't lose. They changed shape.
> Instead of stacking discrete invertible layers, let the warp happen **continuously in time**, driven by a velocity field: $\frac{dx}{dt} = v_\theta(x, t)$. The solution of an ODE is *automatically* invertible — just run time backwards — so the architectural handcuffs come off; $v_\theta$ can be any network. That's a **continuous normalizing flow** → [[Neural ODE]]. Train it without ever simulating the ODE → [[Flow Matching]], which is what drives Stable Diffusion 3 and Flux.
>
> So the lineage runs: *normalizing flow → continuous flow → flow matching.* And a diffusion model's deterministic sampler turns out to be one too → [[Score SDE]].

> [!WARNING] "Normalizing" has nothing to do with batch-norm
> It means the flow maps a complicated data distribution back to a **normal** (Gaussian) one — *normalising the distribution*. "Flow" is the chain of transformations the density flows through.

---
> [!SUCCESS] If you remember one thing
> **Invertible warp + keep the books on the stretching = exact density.** ~={pink}The discrete version was too constrained to win — the continuous version is how today's best image models work.=~

---
# ⁉️
Flows buy an exact density by shackling the architecture. Go to the other extreme: let the network be *anything*, have it output a single "how plausible is this?" number, and simply give up on making the probabilities add up to one. What can you still do?

→ [[Energy-Based Models]]
