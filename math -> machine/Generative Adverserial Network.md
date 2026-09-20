---
aliases:
  - GAN
  - Generative Adversarial Network
tags:
  - generative-models
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Two networks fighting — a **forger** making fakes and a **detective** spotting them — each getting better because the other does.
> **Metaphor:** Counterfeiter vs. bank inspector. Neither improves alone; the arms race is the training signal.
> **Where it bites:** Unstable training, [[Mode Collapse]], and the reason diffusion largely replaced them. Still the cleanest example of a *learned* loss function.

> [!NOTE] Spelling
> Filename keeps the original spelling for link stability. Correct spelling is **adversarial** — the alias handles it. ⚠️

---
# The idea

Every generative model faces the same awkward question: **what's the loss?**

For classification it's obvious — you have a right answer. But if you're generating a photo of a face, there is no single correct output. Use pixel-wise error against a reference and you get blurry mush, because the safest way to minimise average pixel error is to output the *average of all plausible faces*.

Goodfellow's 2014 answer was audacious: ~={blue}**don't design the loss. Train a second network to be the loss.**=~

> [!NOTE] The two players
> - **Generator ($G$)** — takes random noise, outputs a fake sample. The forger. 🎨
> - **Discriminator ($D$)** — takes a sample, outputs "real or fake". The detective. 🔍
>
> $D$ is trained to catch fakes. $G$ is trained to fool $D$. They share no objective — one's gain is the other's loss. ^gan-two-players

$$\min_G \max_D \; \mathbb{E}_{x \sim \text{data}}[\log D(x)] + \mathbb{E}_{z \sim \text{noise}}[\log(1 - D(G(z)))]$$

The forger starts producing noise; the detective easily spots it. But *how* it spots it is information, and that information flows back through [[Backpropagation|backprop]] into the forger. The forger improves. Now the detective must get subtler. Repeat. 🥊

> [!SUCCESS] Core idea
> The loss function **improves during training**. A fixed loss can be gamed once and forever; an adversarial one adapts to whatever cheat the generator just found. That's the genuinely new idea, and it long outlived GANs themselves — it's the same shape as a reward model in RLHF. ^learned-loss

At equilibrium the generator's distribution matches the data and the discriminator is reduced to guessing (outputting 0.5 everywhere), because there is genuinely nothing left to distinguish.

---
# Why they're so hard to train

The elegance costs you stability, and it's worth knowing exactly why:

> [!WARNING] The four standard failures
> 1. **[[Mode Collapse]]** — the generator finds one output that fools the discriminator and produces only that. Nothing in the objective rewards variety.
> 2. **Vanishing gradients** — if the discriminator gets *too* good too fast, it rejects everything with total confidence, $\log(1-D(G(z)))$ saturates, and the generator receives **no usable gradient**. Being beaten too badly teaches you nothing. (Standard fix: the "non-saturating" generator loss.)
> 3. **Failure to converge** — it's a *minimax* game, not a minimisation. There's no loss going down that you can watch. The pair can orbit an equilibrium forever without settling.
> 4. **Balance** — if either network outpaces the other, training stalls. People resort to tricks like updating $D$ $k$ times per $G$ step. It's fiddly and dataset-specific. ^gan-failures

> [!TIP] The thing that catches people out
> **You cannot read GAN losses like normal losses.** A falling generator loss might mean better fakes — or a weakening discriminator. Both losses can look perfect while output is garbage. You evaluate GANs by *looking at samples* and by FID, never by the loss curve. 📉

**WGAN** was the major fix: replace the [[KL Divergence]]-flavoured objective with **Wasserstein** distance, which stays informative even when the two distributions barely overlap — so the generator keeps getting gradient instead of being shut out.

---
# Where they ended up

For about six years GANs were the state of the art in image generation — StyleGAN faces, CycleGAN style transfer, pix2pix, super-resolution.

Then **diffusion** models took over, and the reason is directly the list above. Diffusion trains on a stable, likelihood-style objective with no adversary, no balancing act, and a mode-**covering** loss that doesn't invite collapse. Slower to sample, dramatically easier to train, better coverage.

But the core idea outlived the architecture:
- **Reward models in RLHF** are a learned loss trained against a policy — the same shape, and they suffer the analogous failures (reward hacking ≈ finding the discriminator's blind spot, which is why there's a [[KL Divergence#1. As a leash — RLHF and PPO 🦮|KL leash]]).
- **Adversarial training** for robustness, and adversarial examples generally.
- GANs remain competitive where **single-step** generation matters — real-time, on-device — since they need one forward pass where diffusion needs many.

---
# ⁉️
GANs failed by collapsing onto a narrow output — → [[Mode Collapse]]. The alternative family, which sidesteps it by optimising a mode-covering objective, is diffusion; and the general question of what "good" means for a generative model runs through [[KL Divergence]]. (The map of all the families: [[Generative Models]].)
