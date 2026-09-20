---
aliases:
  - Energy-Based Model
  - Energy Based Models
  - Energy Based Model
  - EBM
  - EBMs
  - Energy Function
  - Partition Function
  - Boltzmann Distribution
  - Boltzmann Machine
  - Contrastive Divergence
  - Unnormalised Density
tags:
  - generative-models
  - probability
  - fundamentals
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Let a network output one number — an **energy**: low for plausible data, high for nonsense. Probability is $e^{-E}$ divided by a normalising constant $Z$ that **you can never compute**. Everything about EBMs is a way of living without $Z$.
> **Metaphor:** A landscape. Real data are marbles resting in the valleys; the model's job is to dig the valleys in the right places.
> **Where it bites:** The conceptual bridge into diffusion — *the score is the slope of this landscape, and the slope doesn't need $Z$*. Also: softmax, contrastive learning, reward models and JEPA are all energy-based in disguise.

---
[[Normalizing Flows]] get an exact probability by putting the network in handcuffs. Try the opposite deal.

Let the network be **anything you like**. It takes an image and returns a single number: *how implausible is this?* Call it the **energy**. A photo of a cat → 0.3. Static → 9.0.

No invertibility. No special layers. No latent variable. Total architectural freedom.

To turn energies into probabilities there's an obvious recipe — lower energy should mean exponentially more probable:

$$p(x) = \frac{e^{-E(x)}}{Z} \qquad\qquad Z = \int e^{-E(x)}\, dx$$

~={blue}Look at that $Z$. To compute it you must evaluate the network on **every possible image** and add them up. What can you still do with a model whose probabilities you can never actually calculate?=~

---
# The landscape ⛰️

![[ebm_landscape_marbles.png]]

More than you'd think — because most useful questions are **comparisons**, and in a comparison $Z$ cancels.

- *Is A more plausible than B?* → $\frac{p(A)}{p(B)} = e^{E(B) - E(A)}$. **No $Z$.**
- *From here, which direction is more plausible?* → the slope of the landscape, $-\nabla_x E(x)$. **No $Z$** — it's a constant, and the derivative of a constant is zero.

![[ebm_energy_vs_density.png]]
> [!TIP] Reading the chart
> **Top:** the energy — any function at all. Two valleys, the left one deeper. **Middle:** the density it implies. In one dimension you *can* integrate it: $Z = 2.94$. In a million dimensions, forget it. **Bottom:** the slope, $-dE/dx$. It points toward the nearest valley from everywhere — and **$Z$ appears nowhere in it.** ~={blue}That bottom panel is the entire reason diffusion models exist.=~

> [!NOTE] Energy-based model
> A model that defines an unnormalised density through a scalar energy function, $p_\theta(x) \propto \exp(-E_\theta(x))$. The normaliser $Z_\theta$ — the **partition function** — is intractable in high dimensions. The form comes from statistical physics: the **Boltzmann distribution**. ^ebm-def

> [!SUCCESS] Core idea
> ~={pink}You trade a tractable probability for total freedom of architecture — and then discover that the **slope** of log-probability is all you needed anyway.=~ That slope is the **score**: $\nabla_x \log p(x) = -\nabla_x E(x)$. Learn it directly and you never meet $Z$ → [[Score Function]]. ^ebm-core

---
# Why training one is miserable

[[Maximum Likelihood]] on an EBM gives a gradient with two halves:

$$\nabla_\theta \log p(x) \;=\; \underbrace{-\nabla_\theta E(x_\text{data})}_{\text{push energy DOWN on real data}} \;+\; \underbrace{\mathbb{E}_{x' \sim p_\theta}\big[\nabla_\theta E(x')\big]}_{\text{push energy UP on the model's own samples}}$$

The first half is easy. The second needs **samples from the model itself** — and the only general way to sample an EBM is [[Markov Chain Monte Carlo|MCMC]]: start somewhere, take noisy steps downhill, wait a long time. *Inside every training step.*

> [!WARNING] Pushing down is not enough
> If you only lower the energy on real data, the network's cheapest solution is to make the energy low **everywhere** — a flat landscape "explains" everything equally. The second term is what raises the ground *between* the valleys. Every EBM training method is a different answer to **"where do I push up?"** — and that's exactly the question [[A Tutorial on Energy-Based Learning|LeCun's tutorial]] is organised around. ^push-up-somewhere

| Method | Where it pushes energy up |
|---|---|
| **Contrastive divergence** (Hinton) | a few MCMC steps away from a *real* data point — cheap, biased |
| **Persistent chains** | on long-running MCMC samples kept between steps |
| **Noise-contrastive estimation** | on samples from a known noise distribution — turns it into classification |
| **[[Score Function\|Score matching]]** | nowhere explicitly — match the *slope* instead, and $Z$ never appears |
| **Contrastive learning** (InfoNCE) | on the *other items in the batch* → [[CLIP]] |
| **Regularised / non-contrastive** | nowhere — limit how much low-energy volume can exist → [[JEPA]] |

---
# You already use energy-based models 🔗

Once you know the shape $\dfrac{e^{\text{score}}}{\sum e^{\text{score}}}$, you see it everywhere:

| Where | The "energy" | How $Z$ is handled |
|---|---|---|
| **Softmax classifier** | −logit | summed over a few thousand classes — *tractable*, so nobody calls it an EBM |
| **An LLM's next token** | −logit | summed over the vocabulary → [[Sampling Parameters]]; temperature is literally the physics temperature |
| **[[Preference Learning\|Bradley–Terry]]** reward model | −reward | only differences are used, so it cancels |
| **[[SAC]]**'s policy | $-Q(s,a)/\alpha$ | sampled, never normalised |
| **[[CLIP]]** | −similarity(image, text) | the batch stands in for "everything else" |
| **Hopfield networks, Boltzmann machines** | the originals | MCMC — which is why they faded |

> [!TIP] The general lesson
> **Normalising over a small discrete set is trivial. Normalising over a continuous high-dimensional space is impossible.** That single fact is why classification was solved long before image generation, and it's the obstacle every family in [[Generative Models]] is designed around.

---
> [!SUCCESS] If you remember one thing
> $p(x) \propto e^{-E(x)}$: **valleys where the data is.** You can't compute the constant — ~={pink}but ratios and slopes don't need it, and the slope is enough to generate.=~

---
# ⁉️
Energy-based models keep all the freedom and none of the convenience. Before following the slope idea through to diffusion, there's a gentler branch of the family that *does* use a latent code and is trivially easy to train — starting with a network whose only job is to copy its input.

→ [[Autoencoder]]
