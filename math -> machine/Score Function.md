---
aliases:
  - Stein Score
  - Score Matching
  - Denoising Score Matching
  - Score-Based Models
  - Score-Based Generative Models
  - Score Network
  - Tweedie's Formula
  - Noise Conditional Score Network
  - NCSN
tags:
  - generative-models
  - probability
  - diffusion
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** The **score** is $\nabla_x \log p(x)$ — at every point in data space, an arrow pointing toward *"more like real data"*. It never involves the normalising constant, and you can learn it just by **training a network to remove noise**.
> **Metaphor:** A compass in thick fog that always points uphill. You can't see the mountain, but you can always tell which way is up.
> **Where it bites:** This is what a diffusion model's network actually learns. "Predict the noise" and "estimate the score" are the same thing, up to a scale factor.

---
[[Energy-Based Models]] ended on a cliff-hanger: you can't compute the probability of an image, because of $Z$ — but the **slope** of the log-probability doesn't contain $Z$ at all.

$$\log p(x) = -E(x) - \underbrace{\log Z}_{\text{a constant}} \qquad\Longrightarrow\qquad \nabla_x \log p(x) = -\nabla_x E(x)$$

So forget the height of the landscape. Suppose all you had was the slope — at every point, an arrow saying *"real data is more that way."*

~={blue}Is that enough to generate a sample? And how on earth would you learn those arrows, when you don't know $p$ in the first place?=~

---
# The compass in the fog 🧭

![[score_field_mixture.png]]

**Is it enough?** Yes. Drop yourself anywhere — in the middle of pure static — and keep stepping the way the arrow points. You'll climb into a region of high probability: something that looks like data. (Add a little randomness as you go and you get *proper* samples → [[Langevin Dynamics]].)

> [!NOTE] Score function
> $s(x) = \nabla_x \log p(x)$ — a vector field with the same dimension as the data. It points in the direction in which log-density increases fastest, and its size says how steeply. Because $\log Z$ is constant in $x$, the score is **independent of the normalising constant**. ^score-def

**How do you learn it?** Naively you'd regress a network $s_\theta(x)$ onto the true score — which you don't have. Here's the trick that made the whole field work (Vincent, 2011):

1. Take a real data point $x$.
2. Add Gaussian noise: $\tilde{x} = x + \sigma\varepsilon$.
3. Ask the network: *from $\tilde{x}$, which way is back to the clean data?*

The right answer for that one noisy sample is obvious: $-\varepsilon/\sigma$ — *straight back along the noise you just added.* And it turns out that a network trained on millions of these, minimising squared error, converges to the **true score of the noised data distribution**. No $Z$, no MCMC, no density — just a regression.

> [!SUCCESS] Core idea
> ~={pink}Learning to denoise **is** learning the score.=~ The best guess for "where did this noisy point come from?" points toward where the data is dense — and that direction is exactly $\nabla \log p$. (Formally, **Tweedie's formula**: $\mathbb{E}[x \mid \tilde{x}] = \tilde{x} + \sigma^2 \nabla \log p_\sigma(\tilde{x})$ — *denoised estimate = noisy point + a step along the score*.) ^denoising-is-score

---
# The problem that led to diffusion

Try it with a single small noise level and generation **fails**. Two reasons, both about empty space:

> [!WARNING] Where there's no data, the score is unknown — and that's almost everywhere
> 1. **The arrows are garbage far from the data.** Real images occupy a thin sliver of pixel space ([[Latent Variable Models#^manifold-core|the manifold hypothesis]]). With a tiny $\sigma$, training only ever visits points right next to that sliver. Start sampling from random static and you're somewhere the network has never seen.
> 2. **The score can't see mode proportions.** Two clusters separated by empty space: the arrows near each cluster are *identical* whether the split is 50/50 or 99/1 — the slope of $\log p$ within a bump doesn't depend on the bump's weight. ^score-blind-spots

The second one isn't theoretical. Here it is, with the **exact** score of a three-cluster mixture whose true weights are 20% / 30% / 50%:

![[langevin_sampling_steps.png]]
> [!TIP] Reading the chart
> Following the score turns noise into three clean clusters — **in the right places**. But the shares come out **29% / 33% / 38%**, not 20 / 30 / 50. Each particle fell into whichever basin it started nearest; nothing told it the basins differ in importance.

**The fix (Song & Ermon, 2019) is the whole idea of diffusion in embryo:** use **many noise levels**, from enormous to tiny, and train *one* network conditioned on $\sigma$.

- **Huge noise** smears the data into a single broad blob that fills all of space — arrows are meaningful *everywhere*, and the blob's shape reflects the clusters' true weights.
- **Shrink the noise gradually**, following the score at each level. Samples are first herded to roughly the right region in the right proportions, then progressively sharpened.

That's **annealed Langevin dynamics**, and it is a diffusion model in all but name → [[Diffusion Models]], unified in [[Score SDE]].

---
# Three names for one network

| You'll read | The network outputs | Relation |
|---|---|---|
| **Score** $s_\theta(x_t, t)$ | the arrow | — |
| **Noise prediction** $\varepsilon_\theta(x_t, t)$ | the noise that was added | $s = -\varepsilon / \sigma_t$ |
| **Denoised image** $\hat{x}_0$ | the clean guess | $\hat{x}_0 = (x_t - \sigma_t \varepsilon)/\alpha_t$ |

Same information, rescaled → [[Denoising Objective]]. And when someone says **"guidance"**, they mean *adding another arrow to this one* → [[Classifier-Free Guidance]].

> [!WARNING] A name clash
> In classical statistics "the score" is $\nabla_{\theta} \log p(x; \theta)$ — the gradient with respect to the **parameters** (Fisher information is its variance). Here it's the gradient with respect to the **data** $x$ — strictly the *Stein* score. Same word, different derivative. ^score-name-clash

---
> [!SUCCESS] If you remember one thing
> The score is **an arrow toward the data, at every point in space** — needing no normalising constant, learnable by denoising. ~={pink}Its one weakness — it's blind wherever there's no data — is cured by adding noise at many scales, and that cure is diffusion.=~

---
# ⁉️
"Keep stepping the way the arrow points" — but pure uphill steps all pile up on the *peaks*. To get genuine samples, spread out the way the data is, you need to add exactly the right amount of randomness at every step.

→ [[Langevin Dynamics]]
