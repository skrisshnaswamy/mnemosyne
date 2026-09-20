---
aliases:
  - Forward Process
  - Forward Diffusion
  - Noising Process
  - Noise Schedule
  - Noise Schedules
  - Variance Schedule
  - Beta Schedule
  - Cosine Schedule
  - Signal-to-Noise Ratio
  - SNR
  - Variance Preserving
  - Variance Exploding
tags:
  - generative-models
  - diffusion
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A fixed recipe that blends an image with Gaussian noise: $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$. No learning. And because Gaussians add up, you can **jump to any noise level in one line** — no need to simulate the steps before it.
> **Metaphor:** A mixing-desk crossfade between two tracks — *the image* and *static* — with $t$ as the fader.
> **Where it bites:** The **noise schedule** is one of the most consequential hyperparameters in diffusion. "Cosine schedule", "SNR", "zero terminal SNR", "VP vs VE" all live here.

---
A [[Diffusion Models|diffusion model]] needs training pairs: *(a noisy image at level $t$, the noise that was added)*.

The definition says the noise is added in a thousand small steps: $x_0 \to x_1 \to x_2 \to \cdots \to x_{1000}$.

So to make one training example at $t = 700$, you'd run 700 noising steps. For a batch of 256 images, each at its own random $t$…

~={blue}That would make training hopelessly slow. It isn't. What's the shortcut?=~

---
# The crossfade 🎚️

Each step scales the image down a touch and adds a touch of fresh Gaussian noise:

$$x_t = \sqrt{1 - \beta_t}\; x_{t-1} + \sqrt{\beta_t}\; \varepsilon_t$$

Now the key fact: **a Gaussian plus a Gaussian is a Gaussian.** Seven hundred small independent doses of noise are indistinguishable from *one* dose of the right size. So all the steps collapse into a single line:

$$x_t = \underbrace{\sqrt{\bar\alpha_t}}_{\text{how much image survives}} x_0 \;+\; \underbrace{\sqrt{1 - \bar\alpha_t}}_{\text{how much noise}}\; \varepsilon \qquad\qquad \bar\alpha_t = \prod_{s \le t} (1 - \beta_s)$$

It's a crossfade. $\bar\alpha_t$ is the fader: 1 at the start (all image), 0 at the end (all static). The squares of the two weights always sum to 1, so the overall "volume" stays constant — that's why this version is called **variance-preserving**.

> [!NOTE] Forward process
> The fixed Markov chain $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1},\; \beta_t I)$, with closed-form marginal $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0,\; (1-\bar\alpha_t) I)$. The sequence $\beta_1 \ldots \beta_T$ is the **noise schedule**. ^forward-process-def

> [!SUCCESS] Core idea
> ~={pink}Training never simulates the chain.=~ Pick an image, pick a random $t$, draw one $\varepsilon$, apply the formula — one line, any noise level, instantly. It's the [[Variational Autoencoder#The reparameterisation trick|reparameterisation trick]] again: *signal plus scaled noise, with the noise as an explicit input*. This shortcut is what makes diffusion training as cheap per step as ordinary supervised learning. ^jump-to-any-t

![[forward_diffusion_density.png]]
> [!TIP] Reading the chart
> Two sharp bumps of data on the left; one featureless $\mathcal{N}(0,1)$ blob on the right. The orange lines are individual data points being jostled. The fraction of original **signal** surviving is 0.72 at $t = 0.25$, **0.28** at $t = 0.5$, 0.06 at $t = 0.75$ and 0.007 at the end. By the right-hand edge, ~={blue}every starting point looks the same=~ — which is the point: a distribution you can sample from with one call to `randn`.

---
# The schedule — where the model spends its effort

How fast should the fader move? It matters more than you'd guess, because the network's capacity is shared across *all* noise levels and you sample $t$ uniformly.

![[noise_schedules.png]]
> [!TIP] Reading the chart
> **Linear β** (the original DDPM) destroys the image fast: by step **543** of 1000, less than 5% survives — so *nearly half of training* is spent on inputs that are indistinguishable from static. At step 500 only **8%** is left. **Cosine** (Nichol & Dhariwal, 2021) is gentler: 5% isn't reached until step **856**, and at step 500 **49%** remains. More of the budget lands on the informative middle, where the image is recognisable but damaged.

The cleanest way to think about any schedule is the **signal-to-noise ratio**, $\text{SNR}(t) = \bar\alpha_t / (1 - \bar\alpha_t)$. A schedule *is* a curve of log-SNR against time; everything else is parameterisation.

| Noise level | What the network must do | What it learns |
|---|---|---|
| **High** (low SNR) | hallucinate global structure from almost nothing | layout, composition, *what is this a picture of* |
| **Middle** | the hard part — recognisable but badly damaged | objects, shapes, semantics |
| **Low** (high SNR) | polish | texture, edges, fine detail |

> [!TIP] That table explains a lot of downstream behaviour
> Coarse-to-fine isn't designed in — it falls out. It's why **the first few sampling steps fix the composition** and the last ones only sharpen; why img2img at "strength 0.3" keeps the layout (it only re-runs the low-noise end); and why high-resolution images need *more* noise to be destroyed to the same degree (neighbouring pixels are redundant, so noise averages out) — hence resolution-dependent "shifted" schedules in SD3-era models → [[Latent Diffusion]], [[Controlling Diffusion]].

---
# Gotchas

> [!WARNING] The "zero terminal SNR" bug
> Many popular schedules stop *just short* of pure noise — at the last step a whisper of the image, including its overall brightness, still leaks through. Training sees that leak; sampling starts from *true* noise, which has none. The mismatch is why early Stable Diffusion struggled to produce genuinely dark or genuinely bright images — everything came out mid-grey on average. Fix: rescale the schedule so $\bar\alpha_T = 0$ exactly. ~={red}Train and sample from the same end-point.=~ ^zero-terminal-snr

> [!WARNING] VP vs VE — two conventions, one idea
> **Variance-preserving** (DDPM): shrink the image *and* add noise; total variance stays ~1. **Variance-exploding** (Song & Ermon's line): leave the image alone and add ever-larger noise. Different bookkeeping, same log-SNR curve, interchangeable in theory → [[Score SDE]]. And [[Flow Matching]] uses a third: a *straight-line* crossfade, $x_t = (1-t)\,x_0 + t\,\varepsilon$.

---
> [!SUCCESS] If you remember one thing
> $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$ — **a crossfade you can jump into at any point.** ~={pink}The schedule decides which noise levels the network practises most, and therefore what it ends up good at.=~

---
# ⁉️
So training examples are free and instant. What exactly should the network be asked to predict from a noisy image — the clean image? The noise? Something else? It sounds like a detail. It's the difference between a loss that works and one that doesn't.

→ [[Denoising Objective]]
