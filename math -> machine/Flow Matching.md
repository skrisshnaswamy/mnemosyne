---
aliases:
  - Flow-Matching
  - Conditional Flow Matching
  - CFM
  - Flow Matching Objective
  - Stochastic Interpolants
  - Velocity Prediction
  - Flow Models
  - Flow-Based Generative Models
tags:
  - generative-models
  - diffusion
  - training
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Pair each noise sample with a data sample, **draw a straight line between them**, and train a network to predict the **velocity** along that line. Generate by integrating that velocity from noise to data.
> **Metaphor:** A motorway instead of a mountain road. Same two towns — but you chose the route, so you chose a straight one.
> **Where it bites:** Stable Diffusion 3, Flux, and most recent image/video/audio models. It's "diffusion, simplified" — and the thing people usually mean when they half-remember it as *"flow control"*.

---
A [[Diffusion Models|diffusion model]] gets from noise to data along a path *dictated by its noise schedule* — and as [[Diffusion Sampling#How few steps can you get away with?|the sampling experiments]] showed, that path is curved, so you need many small steps to follow it.

But the [[Score SDE]] view made something clear: the deterministic sampler is just an ODE — a velocity field carrying noise to data ([[Neural ODE]]). The specific curvy path isn't sacred. It's a by-product of how the noise was added.

~={blue}If all you need is *some* velocity field that carries noise to data — why not pick the path first, and make it the simplest one possible?=~

---
# The motorway 🛣️

![[flow_matching_motorway_vs_mountain_road.png]]

Take one noise sample $x_0$ and one real image $x_1$. The simplest path between them is a straight line, travelled at constant speed:

$$x_t = (1 - t)\,x_0 + t\,x_1 \qquad t \in [0, 1]$$

Along that line the velocity is *constant*, and you know it exactly: $\;x_1 - x_0$.

So train a network to predict it:

```python
x1 = next(data)                       # real images
x0 = randn_like(x1)                   # noise
t  = rand(batch)                      # a random point along the road
xt = (1 - t) * x0 + t * x1            # where you'd be at time t
loss = mse(model(xt, t), x1 - x0)     # "which way, and how fast, should this point be moving?"
```

Compare it with [[Denoising Objective|diffusion's loop]]. Same shape — sample, blend, regress, MSE. No $\bar\alpha_t$, no $\beta$ schedule, no SDE. **To sample:** start at $x_0 \sim \mathcal{N}(0,I)$ and integrate $\dot{x} = v_\theta(x,t)$ from 0 to 1 with any ODE solver.

> [!NOTE] Flow matching
> Train a [[Neural ODE|continuous normalizing flow]] by regressing its velocity field $v_\theta(x_t, t)$ onto the known velocity of a *prescribed* conditional path between noise and data — **without simulating the ODE during training**. (Lipman et al.; Liu et al.; Albergo & Vanden-Eijnden — all 2022–23.) ^flow-matching-def

---
# The subtle bit — why regressing on one line at a time works

Here's what should bother you. At a given point $x_t$, *many different* (noise, image) pairs pass through — their lines cross there. One says "go up-left", another "go right". The target is contradictory.

It's fine, and it's the same trick as [[Score Function#^denoising-is-score|denoising score matching]]. A network trained with **squared error** on contradictory targets learns their **average**. And the average velocity of everything passing through $x_t$ is *exactly* the velocity field that transports the whole noise distribution onto the whole data distribution.

> [!SUCCESS] Core idea
> ~={pink}You never need the true global flow — regress on simple per-pair lines, and the conditional average does the rest.=~ Individual training lines cross all over the place; the *learned* field can't (an ODE's paths never cross — [[Neural ODE]]), so it bends to route traffic around. That's also why the result, though built from straight lines, is **not itself perfectly straight**. ^regress-on-conditional-paths

---
# So how much straighter is it, really?

Here's the honest picture, with exact velocity fields and no learned network:

![[flow_matching_paths.png]]
> [!TIP] Reading the chart
> **Left — diffusion's ODE:** nothing happens for the first 60% of the journey, then a sudden swerve as the noise level finally drops enough for the modes to separate. **Middle — flow matching:** the decision is spread over the whole journey, gentler and earlier — but still curved, because the training lines cross. **Right — [[Rectified Flow|rectified flow]]:** dead straight.

And measured ([[Diffusion Sampling#How few steps can you get away with?|same experiment as before]]): at **2 steps**, flow matching's error is **0.66** against the diffusion ODE's **0.81**; by 8 steps the diffusion ODE is actually slightly *ahead* (0.08 vs 0.14).

> [!WARNING] Flow matching is not automatically "few-step"
> This gets overstated constantly. With random noise–data pairing, plain flow matching is **comparable** to a well-solved diffusion ODE in step count. Its real advantages are elsewhere — see the table. The dramatic step reductions come from *straightening the paths afterwards* → [[Rectified Flow]], or from distillation → [[Consistency Models]]. ^fm-is-not-automatically-fast

| | [[Diffusion Models\|Diffusion]] | **Flow matching** |
|---|---|---|
| Path from noise to data | set by a noise schedule; curved | **a straight-line interpolation you chose** |
| Network predicts | noise ε (or v, or x₀) | **velocity** $x_1 - x_0$ |
| Sampler | SDE *or* ODE | ODE |
| Hyperparameters | $\beta$ schedule, parameterisation, loss weighting — all entangled | essentially: how to sample $t$ |
| End points | must be Gaussian noise ↔ data | **any two distributions** — image ↔ image, low-res ↔ high-res |
| Theory | SDEs, score matching, ELBO | "regress a velocity" |
| Training cost | one pass per example | one pass per example |

> [!TIP] They're closer than the marketing suggests
> With Gaussian noise, flow matching's linear path **is** a diffusion process with a particular schedule ($\alpha_t = t$, $\sigma_t = 1 - t$), and velocity prediction is a linear recombination of ε and x₀ — cousin of [[Denoising Objective#Three parameterisations — one network, three dialects|v-prediction]]. Convert the units and you can sample an FM model with a diffusion sampler and vice-versa. ~={blue}Flow matching is best understood as the *clean restatement* of diffusion, not a rival to it=~ — which is why everything built for diffusion ([[Classifier-Free Guidance]], [[Latent Diffusion]], [[Diffusion Transformer|DiT]]) carries straight over.

---
# What's left to choose

- **How to sample $t$.** Uniform wastes effort at the easy ends. SD3 uses a *logit-normal* distribution, concentrating training on the middle of the road where the velocity is hardest to predict.
- **How to pair $x_0$ with $x_1$.** Random pairing → crossing lines → curvature. Pair them *cleverly* (mini-batch optimal transport, or the model's own outputs) and the lines stop crossing → [[Rectified Flow]].
- **Resolution-dependent time shifting** — spend more of the trajectory at high noise for large images, for the same reason as [[Forward Diffusion Process#The schedule — where the model spends its effort|shifted noise schedules]].

> [!TIP] "Flow control" disambiguation
> If you came here looking for *flow control*: in generative modelling the terms are **normalizing flows** (invertible layers, exact likelihood), **continuous normalizing flows** (the ODE version), **flow matching** (how to train those cheaply — this note), and **rectified flow** (how to straighten them). "Flow control" itself is a networking / fluid-dynamics term and isn't used here.

---
> [!SUCCESS] If you remember one thing
> **Draw a straight line from noise to data; teach a network the velocity along it; integrate.** ~={pink}It's diffusion with the scaffolding removed — simpler to train and reason about, freer in what it can connect — and the door to genuinely few-step generation once the paths are straightened.=~

---
# ⁉️
The training lines are straight. The learned flow isn't — because the lines *cross*, and a flow can't. So what if you re-paired noise and data so that the lines never cross in the first place?

→ [[Rectified Flow]]
