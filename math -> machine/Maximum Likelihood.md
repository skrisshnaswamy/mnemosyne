---
aliases:
  - Maximum Likelihood Estimation
  - Likelihood
  - Log-Likelihood
  - Negative Log-Likelihood
  - NLL
  - Density Estimation
  - Generative vs Discriminative
tags:
  - generative-models
  - fundamentals
  - probability
  - training
---

> [!ABSTRACT] 🧠 Recall
> **In one line:** Pick the parameters under which **the data you actually saw would have been least surprising**. Almost every generative model is trained this way — or is a workaround for when it can't be.
> **Metaphor:** Sliding a bell curve left and right over your data until the data looks as *unremarkable* as possible.
> **Where it bites:** It's cross-entropy, it's next-token prediction, it's the VAE's ELBO and diffusion's loss. And it explains *why* likelihood-trained models are blurry rather than collapsed.

---
You have 30 numbers. They came from *somewhere* — some process you can't see. You'd like to build a machine that produces **more numbers like them**.

You decide the machine will be a bell curve of width 1. One knob left to set: where's its centre, $\mu$?

Try $\mu = 1$. Most of your data sits out in that curve's thin right-hand tail — if the machine were really centred at 1, you'd have been *astonished* to see this data. Try $\mu = 5$. Same problem, other side.

~={blue}So what's the rule you're using to say one setting is better than another?=~

---
# Least surprised 🎯

*"Under this setting, how probable was the data I actually got?"* — and you want that as **high** as possible.

Multiply the probability the curve assigns to each point. (Products of 30 small numbers underflow, so take logs and add.) That total is the **log-likelihood**, and you slide $\mu$ until it peaks.

![[mle_gaussian_fit.png]]
> [!TIP] Reading the chart
> At $\mu = 1$ the log-likelihood is **−110**. At the peak it's **−47** — the data is $e^{63}$ times less surprising. And the peak lands at **3.06, exactly the sample mean**. "Take the average" was maximum likelihood all along.

> [!NOTE] Maximum likelihood
> Choose $\theta$ to maximise $\sum_i \log p_\theta(x_i)$ — the log-probability the model assigns to the observed data. Equivalently, minimise the **negative log-likelihood (NLL)**. ^mle-def

> [!SUCCESS] Core idea
> ~={pink}A generative model is a probability distribution with knobs; training is turning the knobs until your data stops being a surprise.=~ Minimising NLL is *identical* to minimising [[Cross Entropy]] against the data, which is identical to minimising the **forward** [[KL Divergence]] from data to model ([[Cross Entropy#^ce-vs-kl|why]]). Next-token prediction in an LLM is this, one token at a time → [[Auto-regressive models#^ar-factorisation]]. ^mle-core

---
# Generative vs discriminative — the fork this whole area hangs from

| | **Discriminative** | **Generative** |
|---|---|---|
| Learns | $p(y \mid x)$ — *given a photo, is it a cat?* | $p(x)$ or $p(x \mid y)$ — *what do cat photos look like?* |
| Output | a label, a score | **a new $x$** |
| Must model | only the boundary between classes | **everything** about the data — texture, lighting, whiskers |
| Difficulty | lower | far higher: $x$ is a million-dimensional object |

Everything downstream of [[Generative Models]] is on the right-hand side.

---
# The consequence that shapes the whole field

Give a model that's *too simple* a dataset with two separate clumps. It can't fit both exactly. How does maximum likelihood choose to be wrong?

![[kl_mode_covering_vs_seeking.png]]
> [!TIP] Reading the chart
> **Orange — maximum likelihood (forward KL).** Missing a data point costs $\log 0 = -\infty$, so it *must* put mass on both clumps. It stretches to σ = 2.6 and ends up putting most of its mass **in the empty middle**. Samples from it are things that never occur: the average of a cat and a dog. **Blue — reverse KL.** Penalised only for putting mass where there's no data, it picks **one** clump and ignores the other entirely.

> [!WARNING] This is why likelihood models blur and GANs collapse
> **Mode-covering** (maximum likelihood → VAEs, flows, early diffusion): never misses anything, pays for it with **blurry, averaged** samples. **Mode-seeking** (adversarial training → [[Generative Adverserial Network|GANs]]): razor-sharp samples, at the risk of **dropping whole categories** → [[Mode Collapse]]. It isn't a bug in either — it's [[KL Divergence#Forward KL vs Reverse KL|which direction of KL]] they optimise. The history of generative models is largely the search for *sharp **and** diverse*. ^covering-vs-seeking

---
# Why it isn't the end of the story

To use maximum likelihood you must be able to **compute $p_\theta(x)$** — a properly normalised density over images. That's astonishingly hard. Each family makes a different sacrifice:

| Family                                   | How it gets a likelihood                          | What it gives up                |
| ---------------------------------------- | ------------------------------------------------- | ------------------------------- |
| [[Auto-regressive models]]               | chain rule: a product of easy 1-D conditionals    | speed — one element at a time   |
| [[Normalizing Flows]]                    | invertible maps + change of variables — exact     | architecture must be invertible |
| [[Variational Autoencoder]]              | a **lower bound** (ELBO)                          | it's only a bound; blurry       |
| [[Energy-Based Models]]                  | unnormalised — skip the constant                  | can't evaluate or sample easily |
| [[Generative Adverserial Network\|GANs]] | **none at all** — a critic replaces it            | stability, coverage             |
| [[Diffusion Models]]                     | a bound that collapses into *"predict the noise"* | many sampling steps             |

> [!WARNING] Likelihood is not quality
> A model can assign high likelihood and produce ugly samples, or the reverse — likelihood in a million dimensions is dominated by imperceptible pixel-level detail. See [[Auto-regressive models#^likelihood-is-not-quality|the same point for language]], and [[Evaluating Generative Models]] for what people measure instead. ^likelihood-is-not-quality-images

---
> [!SUCCESS] If you remember one thing
> **Make the data unsurprising.** ~={pink}Every generative family is either a way of computing that likelihood, a way of bounding it, or a way of dodging it=~ — and the choice decides whether the model's failures look blurry or collapsed.

---
# ⁉️
A bell curve has one knob. A photograph of a face has a million pixels — but it doesn't have a million *degrees of freedom*: pose, lighting, age, expression… a few dozen causes explain most of it. How do you build a model around causes you never get to observe?

→ [[Latent Variable Models]]
