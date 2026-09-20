---
aliases:
  - VAE
  - VAEs
  - Variational Autoencoders
  - Variational Auto-Encoder
  - ELBO
  - Evidence Lower Bound
  - Reparameterization Trick
  - Reparameterisation Trick
  - Variational Inference
  - Amortized Inference
  - Posterior Collapse
  - Beta-VAE
tags:
  - generative-models
  - bayesian
  - probability
  - deep-learning
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** An [[Autoencoder]] whose encoder outputs a **fuzzy cloud** instead of a point, with a penalty that pulls every cloud toward $\mathcal{N}(0, I)$. The result is a latent space with **no holes** — so you can sample from it.
> **Metaphor:** Islands in an empty sea, versus one continuous country where every spot on the map is somewhere real.
> **Where it bites:** The compressor inside Stable Diffusion, the ELBO that diffusion's loss is derived from, the reparameterisation trick — and the reason VAE samples are **blurry**.

---
The plain [[Autoencoder]] failed as a generator for two reasons: its codes sat wherever they liked, and the space between them meant nothing.

So make two changes.

**One.** The encoder no longer says *"this image is the point (1.2, −0.7)."* It says *"this image is **somewhere around** (1.2, −0.7), give or take 0.3"* — a mean **and** a spread. To decode, you draw a random point from that little cloud.

**Two.** Add a penalty for every cloud that strays from a standard bell curve centred at zero.

~={blue}Think about what the decoder now experiences during training. What is it being forced to learn that it wasn't before?=~

---
# One continuous country 🗺️

![[vae_latent_holes.png]]

It never sees the same code twice for the same image — the code jitters. So it *has* to decode **every point in the neighbourhood** to roughly that image. Neighbourhoods of similar images overlap, so the decoder must blend smoothly between them. **The gaps fill in.**

And the pull toward $\mathcal{N}(0, I)$ herds all the clouds into one known region. After training, drawing $z \sim \mathcal{N}(0, I)$ lands you somewhere the decoder has practised.

![[vae_architecture.png]]

> [!NOTE] Variational autoencoder
> A [[Latent Variable Models|latent variable model]] with a learned approximate posterior. The **encoder** $q_\phi(z \mid x)$ outputs a Gaussian (mean, variance). The **decoder** $p_\theta(x \mid z)$ reconstructs. Both are trained by maximising the **ELBO**. → [[Auto-Encoding Variational Bayes (VAE)]] ^vae-def

---
# The loss — two forces in tension

$$\underbrace{\log p(x)}_{\text{what you want}} \;\geq\; \underbrace{\mathbb{E}_{q(z \mid x)}\big[\log p(x \mid z)\big]}_{\text{reconstruct well}} \;-\; \underbrace{\text{KL}\big(q(z \mid x) \,\|\, p(z)\big)}_{\text{stay close to the prior}}$$

The right-hand side is the **Evidence Lower BOund**. You can't compute $\log p(x)$ — [[Latent Variable Models#The catch — that integral|that integral]] — but you *can* compute a floor under it, and push the floor up.

| Term | Wants | Left alone, it would… |
|---|---|---|
| **Reconstruction** | each image to have its own sharp, distinctive code | recreate the plain autoencoder — islands |
| **[[KL Divergence\|KL]] to the prior** | every cloud to be identical to $\mathcal{N}(0,I)$ | erase all information — the decoder ignores $z$ |

> [!SUCCESS] Core idea
> ~={pink}The reconstruction term says "be informative". The KL term says "be organised". A usable latent space is the truce between them.=~ Turn the KL weight up (**β-VAE**) and you get tidier, more *disentangled* sliders and blurrier pictures; turn it down and it's the other way round. ^elbo-tension

---
# The reparameterisation trick

There's a snag. The network contains a step that says *"draw a random sample"* — and you can't backpropagate through a dice roll.

The fix is to move the randomness **outside** the network:

$$z = \mu + \sigma \cdot \varepsilon \qquad \varepsilon \sim \mathcal{N}(0, 1)$$

Now $\varepsilon$ is just an *input*, like the image. $z$ is a plain deterministic function of $\mu$ and $\sigma$, and gradients flow straight through it.

> [!TIP] You'll see this exact move again
> Diffusion's forward process is written the same way: $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\varepsilon$ — *signal plus scaled noise, with the noise as an explicit input* → [[Forward Diffusion Process]]. And when a sampling step **can't** be reparameterised (discrete choices — an action, a token), you fall back on the [[Policy Gradient#^policy-gradient-def|log-derivative trick]] instead. Those are the two ways to get a gradient through randomness.

---
# Why VAE samples are blurry

> [!WARNING] Two separate causes — both worth knowing
> 1. **The objective is mode-covering.** The ELBO is a [[Maximum Likelihood#^covering-vs-seeking|maximum-likelihood]] bound, so the model must put mass on *everything* plausible. Faced with several sharp possibilities for a region, a Gaussian decoder with a squared-error loss outputs their **average** — and the average of many sharp images is a blurry one.
> 2. **The clouds overlap.** One $z$ has to serve several slightly different images, so the decoder hedges.
>
> [[Generative Adverserial Network|GANs]] attacked this by dropping likelihood altogether. [[Diffusion Models]] attacked it by never asking one network pass to go from code to image in a single leap. ^why-vaes-blur

> [!WARNING] Posterior collapse
> Pair a VAE with a *very* powerful decoder (an autoregressive one, say) and it can discover that the cheapest way to zero the KL term is to make every cloud exactly $\mathcal{N}(0,I)$ — and **ignore $z$ completely**, modelling the data with the decoder alone. The latent carries no information. Fixes: anneal the KL weight up from zero, "free bits", a weaker decoder.

---
# Where it lives now

Rarely as the final generator. Constantly as the **compressor**:

- **[[Latent Diffusion]]** — a VAE shrinks a 512×512 image to a 64×64×4 latent; diffusion runs *there*; the VAE decoder renders the result. When a Stable Diffusion image has mangled fine text or odd skin texture, that's often **this** component's doing.
- **[[VQ-VAE]]** — the discrete cousin, which turns images into tokens.
- **World models** — Dreamer learns its dynamics in a VAE-style latent → [[Model-Based vs Model-Free RL]].
- **Theory** — a diffusion model *is* a hierarchical VAE with a thousand layers and a fixed encoder; DDPM's loss is this ELBO, simplified → [[Diffusion Models]].

---
> [!SUCCESS] If you remember one thing
> **Encode to a cloud, not a point — and pull the clouds toward a bell curve.** ~={pink}That fills the holes, which is what makes the latent space sampleable; and the averaging it implies is what makes the samples blurry.=~

---
# ⁉️
Clouds in a continuous space. But the most successful generative models of all — language models — work on **discrete tokens** drawn from a vocabulary. Could an image be turned into a sequence of tokens from a fixed codebook, so the same machinery applies?

→ [[VQ-VAE]]
