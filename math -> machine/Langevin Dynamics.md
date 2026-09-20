---
aliases:
  - Langevin
  - Langevin Sampling
  - Langevin MCMC
  - Langevin Monte Carlo
  - Unadjusted Langevin Algorithm
  - ULA
  - MALA
  - Annealed Langevin Dynamics
  - Stochastic Gradient Langevin Dynamics
  - SGLD
tags:
  - generative-models
  - probability
  - diffusion
  - MCMC
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** To sample from $p$ using only its [[Score Function|score]]: **take a small step along the score, add a carefully sized dose of Gaussian noise, repeat.** Drift pulls you toward the data; noise keeps you from collapsing onto the peak.
> **Metaphor:** A marble rolling downhill on a table that's being gently shaken.
> **Where it bites:** The sampler underneath score-based models, the "corrector" in diffusion samplers, and the bridge between *optimisation* and *sampling*.

---
You have the score — an arrow at every point, pointing toward more probable data.

The obvious thing to do with it is **gradient ascent**: start at a random point, step along the arrow, repeat until you stop moving.

Run that from a thousand random starts and look at where they end up.

~={blue}All thousand are sitting on exactly the same few points — the very tops of the peaks. Is that a *sample* from the distribution?=~

---
# The shaken table 🎱

No. It's the **mode** — the single most typical face, a thousand times. A sample from "faces" should be a *different* face each time, spread out the way real faces are.

Put a marble on a bumpy table and it rolls to the bottom of the nearest dip and stops. Now **shake the table**. It still spends most of its time in the dips — but it jiggles around inside them, occasionally hops over a ridge, and over time visits each region *in proportion to how deep and wide it is*.

$$x_{k+1} = x_k + \underbrace{\varepsilon\, \nabla_x \log p(x_k)}_{\text{drift: toward the data}} + \underbrace{\sqrt{2\varepsilon}\; z_k}_{\text{noise: don't collapse}} \qquad z_k \sim \mathcal{N}(0, I)$$

> [!NOTE] Langevin dynamics
> A Markov chain (and the SDE it discretises) whose stationary distribution is $p(x)$, using only the score $\nabla \log p$. With step size $\varepsilon \to 0$ and enough steps, $x_k$ is an exact sample from $p$. ^langevin-def

![[langevin_sampling_steps.png]]

> [!SUCCESS] Core idea
> ~={pink}Optimisation finds the peak. Sampling explores the whole mountain. The only difference is the noise.=~ And the amount isn't arbitrary: $\sqrt{2\varepsilon}$ is *exactly* what balances the drift so that the chain settles into $p$ — no more, no less. Too little and you collapse toward the modes; too much and you're just diffusing at random. ^noise-is-the-difference

---
# Where it sits among the samplers you know

| | Uses | Proposal | Note |
|---|---|---|---|
| [[Markov Chain Monte Carlo\|Random-walk Metropolis]] | density *ratios* | blind | slow — mostly rejected |
| **Langevin (ULA)** | **the score** | a step uphill + noise | no accept/reject; slightly biased for finite $\varepsilon$ |
| **MALA** | score + density ratio | Langevin step, then Metropolis accept/reject | exact; needs the (unnormalised) density |
| [[Hamiltonian Monte Carlo]] | score + momentum | a long glide | far better mixing; [[Hamiltonian Monte Carlo#^hmc-needs-gradients\|same need for gradients]] |

It's the fog-walker from [[Markov Chain Monte Carlo#The walk in the fog 🌫️|the MCMC note]] — given a compass.

> [!TIP] Turn the noise off and you're back at gradient descent
> Replace $\log p$ with a negative loss and drop the noise term: that's gradient descent. Keep a little noise and it's **SGLD** — *stochastic gradient Langevin dynamics* — which turns an optimiser into an approximate sampler from the Bayesian posterior over weights. Mini-batch noise in ordinary SGD does something loosely similar by accident, which is one story for why SGD finds solutions that generalise → [[Uncertainty]], [[Regularization]].

---
# Why plain Langevin isn't enough for images

> [!WARNING] It mixes terribly between separated modes
> To get from one cluster to another the marble must cross a region of near-zero probability — where the score is tiny or points *back*. With well-separated modes that essentially never happens, so each particle stays in the basin it started in. In [[Score Function#The problem that led to diffusion|the experiment above]] the three clusters came out at **29 / 33 / 38%** when the truth was **20 / 30 / 50%** — after 2,000 steps. Right places, wrong proportions. ^langevin-mixes-badly

Real image distributions are the worst case: countless modes, separated by vast deserts of static.

**The fix — annealing.** Start with the data blurred by *enormous* noise, so everything is one connected blob and the marble roams freely. Run Langevin for a while. Reduce the noise a notch. Run again. Repeat down to almost none. Each level hands the next a starting point that's already in the right region, in the right proportion.

> [!TIP] And that is a diffusion sampler
> "Follow the score of a progressively less-noisy distribution" is precisely what the reverse process of a [[Diffusion Models|diffusion model]] does. The modern view ([[Score SDE]]) folds the noise schedule and the Langevin steps into one reverse-time SDE. Langevin survives inside it as the optional **corrector** step in predictor–corrector samplers → [[Diffusion Sampling]].

---
# The same equation in other clothes

- **Physics**, where it started (1908): a pollen grain in water — friction, a force, and random molecular kicks.
- **Finance:** drift + volatility. An SDE is an SDE.
- **Control:** it's a [[State-Space Model|state-space]] system driven by process noise — and the score plays the role of a restoring force.

---
> [!SUCCESS] If you remember one thing
> **Step along the score + $\sqrt{2\varepsilon}$ of noise.** ~={pink}Drift finds the data; noise keeps the samples diverse. And because it can't hop between distant modes, you anneal the noise level from huge to tiny — which is diffusion.=~

---
# ⁉️
We've been circling it for three notes. Put the pieces together — a ladder of noise levels, a network that denoises at every rung, and a walk down the ladder from static to data — and give it its name.

→ [[Diffusion Models]]
