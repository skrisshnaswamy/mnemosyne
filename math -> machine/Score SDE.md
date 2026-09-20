---
aliases:
  - Score-Based SDE
  - Stochastic Differential Equation
  - SDE
  - Reverse-Time SDE
  - Reverse SDE
  - Probability Flow ODE
  - PF-ODE
  - VP-SDE
  - VE-SDE
  - Continuous-Time Diffusion
  - Fokker-Planck Equation
tags:
  - generative-models
  - diffusion
  - probability
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Let the number of noise steps go to infinity and diffusion becomes a **stochastic differential equation**. It has an exact **reverse-time SDE** — which needs only the [[Score Function|score]] — *and* a deterministic **ODE** twin that produces the very same distribution.
> **Metaphor:** A river with turbulence. Forward: drop leaves in and watch them scatter. Reverse: run the river backwards — and the only extra thing you need to know is, at each spot, *which way the crowd of leaves is densest*.
> **Where it bites:** It's the Rosetta stone. DDPM, score-matching models, DDIM and flow matching are all this one framework with different coefficients.

---
By 2020 there were two lines of work that looked suspiciously alike:

- **DDPM** — a discrete chain of 1,000 noising steps; train a network to predict noise.
- **Score-based models** — a ladder of noise scales; train a network to estimate scores; sample with [[Langevin Dynamics|annealed Langevin]].

Different maths, different notation, different communities. Both slow. And then **DDIM** appeared: a *deterministic* sampler for DDPM, with no randomness at all, that somehow produced the same distribution.

~={blue}Two stochastic methods and one deterministic one, all giving the same samples. What's the single object they're all approximations of?=~

---
# The river 🌊

Make the noise steps smaller and more numerous until they blur into continuous time. The forward process becomes:

$$dx = \underbrace{f(x, t)\,dt}_{\text{drift: the current}} + \underbrace{g(t)\,dw}_{\text{diffusion: the turbulence}}$$

Read it as a river. A leaf at position $x$ is carried by the **current** $f$ and buffeted by **turbulence** of strength $g$. Drop a tidy cluster of leaves in upstream (your data); far downstream they're scattered into a featureless cloud (noise).

Now the remarkable fact (Anderson, 1982): **this can be run backwards, exactly**, as another SDE —

$$dx = \big[\, f(x,t) - g(t)^2\, \underbrace{\nabla_x \log p_t(x)}_{\text{the score}} \,\big]\,dt + g(t)\,d\bar{w}$$

Everything in it is known — $f$ and $g$ are *your own design choices* — except one term: the score of the noised data at time $t$. Which is precisely what a denoising network learns.

> [!NOTE] Score SDE
> A framework (Song et al., 2021) that defines diffusion as a continuous-time SDE, generates by solving the corresponding **reverse-time SDE**, and needs only a time-dependent score model $s_\theta(x, t) \approx \nabla_x \log p_t(x)$ to do so. → [[Score-Based Generative Modeling through SDEs]] ^score-sde-def

> [!SUCCESS] Core idea
> ~={pink}Design the forward noising however you like; the reverse is then *determined*, and the only thing you have to learn is the score.=~ Training objective, noise schedule and sampler come apart into three independent choices. That decoupling is why you can train once and then swap samplers freely. ^one-unknown

---
# The twin: a deterministic ODE

Here's the part that explains DDIM. For every such SDE there is an **ordinary** differential equation — no randomness — whose solutions have *exactly the same distribution at every time $t$*:

$$\frac{dx}{dt} = f(x,t) - \tfrac{1}{2}\, g(t)^2\, \nabla_x \log p_t(x) \qquad \text{the \textbf{probability-flow ODE}}$$

Same score, half the coefficient, no noise term.

![[sde_vs_ode_paths.png]]
> [!TIP] Reading the chart
> Individual leaves follow totally different paths — jittery on the left, smooth on the right — but **the cloud as a whole is identical at every moment**. The SDE describes what happens to *one particle*. The ODE describes how the *density* flows. Both are "the" reverse of the forward process.

| | **Reverse SDE** | **Probability-flow ODE** |
|---|---|---|
| Randomness while sampling | yes | **none** |
| Discrete-time cousin | DDPM ancestral sampling | **DDIM** |
| Solver | Euler–Maruyama, predictor–corrector | any ODE solver — Euler, Heun, **DPM-Solver** |
| Steps needed | many | **few** → [[Diffusion Sampling]] |
| Noise ↔ image | many-to-many | **one-to-one and invertible** |
| Exact likelihood | no | **yes** — via change of variables |

That last pair of rows should ring a bell. A smooth, invertible map from noise to data, with a computable likelihood…

> [!TIP] A diffusion model *is* a normalizing flow
> …it's a [[Normalizing Flows|normalizing flow]]. A *continuous* one ([[Neural ODE]]) whose velocity field is $f - \tfrac12 g^2 s_\theta$. ~={blue}Diffusion is a particular way of *training* a continuous flow=~ — by denoising, rather than by maximum likelihood. Ask *"what if I trained that velocity field directly, and chose a simpler path?"* and you arrive at [[Flow Matching]].

---
# The Rosetta table

| Model | Drift $f$ | Noise $g$ | What happens to the data |
|---|---|---|---|
| **VP-SDE** = DDPM | $-\tfrac12 \beta(t)\, x$ | $\sqrt{\beta(t)}$ | shrinks toward 0 **and** noise grows; total variance ≈ 1 |
| **VE-SDE** = Song & Ermon's NCSN | $0$ | $\sqrt{d\sigma^2/dt}$ | image untouched; noise **explodes** |
| **sub-VP** | VP, with less noise | — | better likelihoods |
| **[[Flow Matching]]** (as an ODE) | straight-line velocity | 0 | a linear crossfade — no SDE needed at all |

**Predictor–corrector sampling** falls out naturally: *predict* with one reverse-SDE step, then *correct* with a few [[Langevin Dynamics]] steps at the new noise level — which nudges the sample back onto the right distribution.

> [!NOTE] Jargon: Fokker–Planck
> The PDE describing how the *density* $p_t(x)$ evolves under an SDE. You rarely solve it; it's the theorem that guarantees the SDE and the ODE share their marginals. The [[Transfer Function|control-theory]] reader will recognise the flavour: one equation for a trajectory, another for the distribution of trajectories — see also the predict step of a [[Bayes Filter]], where uncertainty spreads under process noise.

> [!WARNING] You don't need SDEs to *use* diffusion
> This framework is for **understanding** and for **designing** samplers and schedules. Day-to-day you train with the three-line loop in [[Denoising Objective]] and pick a sampler from a menu. But when a paper writes $dx = f\,dt + g\,dw$, this note is the decoder: *$f$ and $g$ are the noise schedule; the score is the network; the ODE is the fast sampler.* ^sde-is-for-understanding

---
> [!SUCCESS] If you remember one thing
> **Forward: an SDE you design. Reverse: fixed by the score.** ~={pink}And every diffusion SDE has a deterministic ODE twin with the same distribution — which makes a diffusion model a continuous normalizing flow, and opens the door to training that flow directly.=~

---
# ⁉️
"A network that outputs a velocity; you integrate it to move a point from A to B" is an idea that stands on its own — it predates diffusion, and it's the piece flow matching is built from.

→ [[Neural ODE]]
