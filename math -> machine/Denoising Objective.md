---
aliases:
  - Diffusion Loss
  - Diffusion Training Objective
  - Noise Prediction
  - Epsilon Prediction
  - ε-prediction
  - v-prediction
  - x0-prediction
  - Simple Loss
  - L_simple
  - Loss Weighting
  - Min-SNR Weighting
tags:
  - generative-models
  - diffusion
  - training
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Take a real image, noise it to a random level, and train the network to **predict the noise you added** — plain mean-squared error. That's the entire training loop of a diffusion model.
> **Metaphor:** A picture restorer who's handed paintings at every stage of grime, and asked each time: *"which part of this is the dirt?"*
> **Where it bites:** ε- vs x₀- vs v-prediction, loss weighting across noise levels — and the fact that **training loss tells you almost nothing about image quality**.

---
You can make unlimited training pairs for free ([[Forward Diffusion Process]]): a clean image $x_0$, a noise sample $\varepsilon$, a level $t$, and the blend $x_t$.

The network will see $x_t$ and $t$. Now — **what should it output?**

The obvious answer is *"the clean image $x_0$."* That's what you want in the end, after all.

~={blue}But think about $t = 999$, where $x_t$ is essentially pure static. What is the "correct" clean image then — and what will a network trained with squared error actually produce?=~

---
# Which part is the dirt? 🖼️

At $t = 999$ the input carries no information, so the best squared-error answer is **the average of every image in the dataset** — a grey-brown smear. The network learns "output mush" for a large share of its training, and that's an ugly target with a huge dynamic range across noise levels.

DDPM's move: ask for the **noise** instead.

$$\mathcal{L}_\text{simple} = \mathbb{E}_{x_0,\, \varepsilon,\, t}\;\big\|\, \varepsilon - \varepsilon_\theta(x_t,\, t) \,\big\|^2$$

The target is always a unit Gaussian — same scale at every $t$, no matter what the image is.

```python
x0   = next(data)                                  # a batch of real images
t    = randint(0, T, size=batch)                   # a random noise level for each
eps  = randn_like(x0)                              # the noise we are about to add
xt   = sqrt(abar[t]) * x0 + sqrt(1 - abar[t]) * eps   # jump straight to level t
loss = mse(model(xt, t), eps)                      # "which part is the dirt?"
```

That's it. No adversary, no MCMC, no intractable constant, no special architecture.

> [!NOTE] The denoising objective
> A regression: given $x_t$ and $t$, predict $\varepsilon$ (or an equivalent target). It is the variational bound (ELBO) of the diffusion model with the per-timestep weights dropped — Ho et al. found the *simplified* version gave better samples than the principled one. ^denoising-objective-def

> [!SUCCESS] Core idea
> ~={pink}Generative modelling, reduced to supervised regression with free labels.=~ And it's secretly [[Score Function#^denoising-is-score|score matching]]: the noise that was added points *away* from the data, so predicting it is estimating the score, $\nabla \log p_t(x_t) = -\varepsilon / \sqrt{1 - \bar\alpha_t}$. The stability of diffusion training comes from exactly this: a fixed target, a convex loss, no moving opponent — everything [[Generative Adverserial Network#^gan-failures|GAN training]] isn't. ^regression-with-free-labels

---
# Three parameterisations — one network, three dialects

Since $x_t = \alpha_t x_0 + \sigma_t \varepsilon$, knowing any one of these gives you the others:

| Predict | Target | Good at | Bad at |
|---|---|---|---|
| **ε** (noise) | $\varepsilon$ | low noise — fine detail | **high noise**: as $x_t \to$ pure noise, predicting ε is trivial (*copy the input*) and says nothing about the image |
| **x₀** (clean image) | $x_0$ | high noise — global structure | **low noise**: trivial (*copy the input*); early steps are blurry averages |
| **v** (velocity) | $\alpha_t \varepsilon - \sigma_t x_0$ | balanced — never degenerate at either end | slightly less intuitive |

> [!TIP] Why v-prediction exists
> Each of ε and x₀ becomes a no-op at one end of the noise range. **v** is a rotation between them that's informative everywhere. It's required for few-step distillation and for [[Forward Diffusion Process#^zero-terminal-snr|zero-terminal-SNR]] schedules — where ε-prediction at the last step is literally undefined — and it's one step away from what [[Flow Matching]] predicts. SD 2.x and most video models use it.

---
# Weighting — the hidden hyperparameter

Every noise level contributes to the loss. *How much should each count?*

- The **principled ELBO** weights heavily toward low noise — it cares about likelihood, i.e. imperceptible detail.
- **$\mathcal{L}_\text{simple}$** uses equal weight on ε, which (after converting units) **up-weights high noise** — it cares about structure, which is what humans see.
- **Min-SNR**, **P2**, **EDM's** weighting — deliberate choices of the same curve.

That's the real reason the "simplified" loss beat the correct one: it's a better match to perception, not to likelihood → [[Maximum Likelihood#^likelihood-is-not-quality-images|likelihood is not quality]].

---
# What you need to know in practice 🛠️

> [!WARNING] The loss curve will lie to you
> Diffusion training loss drops fast, then sits almost flat for the rest of training — *while sample quality keeps improving for hundreds of thousands of steps*. The loss is an average over all noise levels, dominated by levels that are easy or irreducibly noisy; the improvements that matter are a small part of it. ~={red}Never judge a diffusion run by its loss.=~ Generate fixed-seed samples at intervals, and track [[Evaluating Generative Models|FID]] or similar. ^loss-curve-lies

> [!TIP] Details that matter more than you'd expect
> - **EMA weights.** Sample from an exponential moving average of the weights, not the raw ones — it's routinely the difference between good and mediocre. (Same "slow reference copy" idea as [[DQN#^make-it-hold-still|target networks]].)
> - **Timestep sampling.** Uniform $t$ is the default; sampling more from the middle (logit-normal, as in SD3) helps.
> - **Conditioning dropout.** Randomly blank the prompt ~10% of the time during training — that's what makes [[Classifier-Free Guidance]] possible later.
> - **The network must know $t$.** The same image patch means different things at different noise levels, so $t$ is embedded and injected everywhere → [[U-Net]], [[Diffusion Transformer]].

And the restorer's lineage is direct: this is a [[Autoencoder#The family|denoising autoencoder]], trained at every noise level at once, and then **used iteratively** — which the 2008 version never was.

---
> [!SUCCESS] If you remember one thing
> **Noise a real image, predict the noise, MSE.** ~={pink}Everything sophisticated about diffusion is in how you *use* that denoiser — the training loop itself is about the simplest in deep generative modelling.=~

---
# ⁉️
You now own a network that, shown a noisy image, can point at the noise. But a single pass on pure static gives you mush. How do you turn a denoiser into a *generator* — and how few steps can you get away with?

→ [[Diffusion Sampling]]
