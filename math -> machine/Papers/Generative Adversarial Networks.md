---
title: "Generative Adversarial Networks"
authors: ["Ian J. Goodfellow", "Jean Pouget-Abadie", "Mehdi Mirza", "Bing Xu", "David Warde-Farley", "Sherjil Ozair", "Aaron Courville", "Yoshua Bengio"]
year: 2014
arxiv: "1406.2661"
url: https://arxiv.org/abs/1406.2661
priority: Must-Read
read_on: 2026-08-22
tags: [paper, self-supervised, vision]
---
## The Core Idea

Before this paper, training a generative model — a model that can produce new data that looks like the training data — meant writing down a probability density $p(x)$ and then trying to make it big on the real data. That is [maximum likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation), and it hurts. Any density has to sum to one, so you need a normalising constant (the "partition function"). For deep models like Boltzmann machines that constant is an intractable sum over every possible configuration. People approximated it with [[Markov Chain Monte Carlo]], which is slow, mixes badly between modes, and forces the learned distribution to be blurry (a chain cannot hop between sharp, isolated modes).

The trick here: **never write down a density at all**. Instead define a *sampler*. Take noise $z$ from something easy (uniform, Gaussian), push it through a neural net $G$, and call $G(z)$ your sample. This implicitly defines a distribution $p_g$ but you can never evaluate it — and that turns out to be fine.

The problem then becomes: how do you train a thing when you cannot compute its likelihood? Answer: **use a second network as the loss function**. A discriminator $D$ is trained to tell real data apart from $G$'s output. $G$ is trained to fool $D$. Two networks, opposite goals, one game. The counterfeiter and the police.

Why this is beautiful rather than just cute: when $D$ is trained to optimality, the game's objective for $G$ is exactly

$$C(G) = -\log 4 + 2 \cdot \mathrm{JSD}(p_{\text{data}} \,\|\, p_g)$$

the Jensen–Shannon divergence between the real and fake distributions. So the adversary is not a hack — it is a *learned, differentiable estimate of a divergence*. Minimising it drives $p_g \to p_{\text{data}}$, at which point $D(x) = \frac{1}{2}$ everywhere because it genuinely cannot tell.

What it unlocks: sampling is a single forward pass, no Markov chain, no burn-in, no approximate inference net. Gradients come from plain [[Backpropagation]]. Any differentiable function is a legal generator. And because there is no mixing requirement, the model can represent razor-sharp distributions.

> [!NOTE] Implicit generative model
> A model you can sample from but whose density you cannot evaluate. It is defined by the sampling procedure (noise → network) rather than by a formula for $p(x)$. ^implicit-generative-model

## The Methodology

**The two networks.**
- $G(z; \theta_g)$: an MLP. Input is noise $z \sim p_z(z)$. Output lives in data space (a 28×28 image, say). Mix of ReLU and sigmoid activations. Noise is fed only to the bottom layer — no stochasticity in the hidden layers, even though the theory allows it.
- $D(x; \theta_d)$: an MLP outputting one scalar in $[0,1]$ — "probability this came from the real data". Uses maxout activations and dropout.

**The objective.** A minimax game with value function

$$\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

Read it as two halves. $D$ wants $\log D(x)$ big on real data and $\log(1-D(G(z)))$ big on fakes — that is just binary [[Cross Entropy]] for a real-vs-fake classifier. $G$ wants the second term small.

**The training loop** (Algorithm 1):

```
for each iteration:
    for k steps:
        sample m noise vectors z, m real examples x
        ascend  ∇_θd  (1/m) Σ [ log D(x) + log(1 - D(G(z))) ]
    sample m noise vectors z
    descend ∇_θg  (1/m) Σ log(1 - D(G(z)))
```

They used $k=1$ — the cheapest option. The point of the inner loop is to keep $D$ near its optimum for the current $G$; if $G$ moves slowly, one step suffices. This is the same spirit as persistent contrastive divergence, where you carry Markov chain state between updates instead of re-burning it in. Updates use [[Momentum]].

**The non-saturating fix — the most practically important line in the paper.** Early in training, $G$ is terrible, $D$ rejects everything with confidence, so $D(G(z)) \approx 0$ and $\log(1 - D(G(z)))$ flattens out. The [[Derivative#Gradient|gradient]] vanishes exactly when $G$ needs help most. So in practice they flip it: instead of minimising $\log(1 - D(G(z)))$, **maximise $\log D(G(z))$**. Same fixed point, much stronger early gradients. Almost every GAN since uses this form.

**The theory, in three steps.**

1. *Optimal discriminator.* Hold $G$ fixed. The objective is $\int_x p_{\text{data}}(x)\log D(x) + p_g(x)\log(1-D(x))\,dx$. Pointwise, $a\log y + b\log(1-y)$ peaks at $y = \frac{a}{a+b}$. So
$$D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$
A pure density-ratio estimator. This is the whole reason GANs work — $D$ secretly computes how much more likely $x$ is under the data than under the model.

2. *Global optimum.* Plug $D^*$ back in, add and subtract $\log 4$, and you get $C(G) = -\log 4 + \mathrm{KL}(p_{\text{data}} \| \bar{p}) + \mathrm{KL}(p_g \| \bar{p})$ with $\bar{p} = \frac{p_{\text{data}}+p_g}{2}$, which is $2 \cdot \mathrm{JSD}$. See [[KL Divergence]]. JSD is $\ge 0$ and zero only when the two distributions match, so the unique minimum is $p_g = p_{\text{data}}$ with value $-\log 4 \approx -1.386$.

3. *Convergence.* If you optimise in the space of *densities* $p_g$ rather than parameters $\theta_g$, $\sup_D U(p_g, D)$ is convex in $p_g$ with a unique optimum, and subgradient descent converges. The authors are honest that this proof does not transfer: with an MLP you optimise $\theta_g$, and that parameter space has many critical points. They fall back on "MLPs work well in practice despite no guarantees".

## Ablation Studies and Experiments

There are, frankly, almost no ablations. This is a 2014 conference paper announcing an idea, not a systematic study.

**Datasets.** MNIST, Toronto Face Database (TFD), CIFAR-10.

**Metric.** They cannot compute likelihood, so they fit a Gaussian Parzen window to 
generated samples (a kernel density estimate, $\sigma$ chosen by cross-validation on the validation set) and report test log-likelihood under that.

| Model | MNIST | TFD |
|---|---|---|
| DBN | $138 \pm 2$ | $1909 \pm 66$ |
| Stacked CAE | $121 \pm 1.6$ | $\mathbf{2110 \pm 50}$ |
| Deep GSN | $214 \pm 1.1$ | $1890 \pm 29$ |
| **Adversarial nets** | $\mathbf{225 \pm 2}$ | $\mathbf{2057 \pm 26}$ |

Best on MNIST, statistically tied for best on TFD. The authors immediately undercut their own metric: Parzen estimates have high variance and behave badly in high dimensions. They say plainly it "is the best method available to our knowledge" — an admission that generative model evaluation was unsolved.

**Qualitative.** Figure 2 shows samples with the nearest training image beside each one, to prove the model is not memorising. Two CIFAR-10 variants: a fully connected one (bad) and one with a convolutional $D$ and a "deconvolutional" $G$ (better) — the seed of DCGAN two years later. Figure 3 interpolates linearly in $z$ space and the digits morph smoothly, evidence the latent space is learned structure, not a lookup table.

**What did not work / what they warn about.**
- The original $\log(1-D(G(z)))$ generator loss saturates and does not train. This is a real failed thing, fixed by the non-saturating swap.
- **The "Helvetica scenario"** — their name for what we now call [[Mode Collapse]]. If $G$ gets too many updates without $D$ catching up, $G$ collapses many $z$ values onto the same $x$. It has found one output $D$ currently likes and pumps all its probability mass there. The only defence offered is "keep $D$ synchronised with $G$", which is not an algorithm.
- Noise injected only at the input layer, not intermediate layers, despite theory permitting it. Read as: it did not help.
- No explicit $p_g(x)$, so no likelihood, no density evaluation, no principled model comparison.

## Worth Remembering

- **The discriminator is a density ratio estimator.** $D^*(x) = \frac{p_{\text{data}}}{p_{\text{data}}+p_g}$. Every later "learned critic" idea — reward models in [[Training language models to follow instructions with human feedback|RLHF]], contrastive losses, noise-contrastive estimation — is a variant of "train a classifier and use its logit as a signal". A GAN is NCE where the noise distribution is *learned and keeps improving*, which is why GANs do not stall the way NCE does once the model gets roughly right.

- **The JSD result is fragile.** It holds only for the *optimal* $D$. With $k=1$ and a finite MLP, $D$ is never optimal, so you are not really minimising JSD — you are doing something the theory does not cover. Also, if $p_g$ and $p_{\text{data}}$ have disjoint supports (very likely early on for images on a low-dimensional manifold), JSD is a constant $\log 2$ and its gradient is zero. Wasserstein GAN later attacked exactly this hole.

- **No convergence guarantee in parameter space.** Simultaneous gradient descent on a minimax objective is not gradient descent on anything. It can cycle. GAN training instability is not a bug in your code; it is in the problem definition.

- **Sibling paper.** [[Auto-Encoding Variational Bayes (VAE)]] came out the same year and solves the same problem — train a deep generator by backprop — with the opposite philosophy: keep an explicit likelihood bound (the ELBO) and pay for it with blur. GANs give up the likelihood and get sharpness. That trade defined generative modelling for the next six years.

- **The future-work list is a research programme.** Item 1 is conditional GANs (add $c$ to both $G$ and $D$) → pix2pix, StyleGAN. Item 2 is a learned inference net → BiGAN/ALI. Item 4 is semi-supervised learning from $D$'s features → a whole 2016 literature. All five items got papers.

- **Practical caveat if you were to build one today:** the "keep $D$ and $G$ balanced" advice is real and painful. Loss values tell you almost nothing — a $D$ loss near $\log 2 \approx 0.693$ means the game is balanced, but a $G$ loss going down can just mean $D$ got worse. You have to look at samples. And the Parzen-window metric here should not be reused; use FID.

- Small thing worth noticing: the generator never sees a training example directly. Gradients reach it only through $D$. The authors speculate this is a statistical advantage — no input component can be copied straight into $G$'s weights, so memorisation is structurally harder.

## Links

Related: [[Generative Adverserial Network]] · [[Mode Collapse]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[KL Divergence]] · [[Cross Entropy]] · [[Backpropagation]] · [[Markov Chain Monte Carlo]] · [[Momentum]] · [[Deep Learning]] · [[Random variable]] · [[Uncertainty]] · [[Regularization]]

New topics worth writing: Jensen-Shannon divergence, Wasserstein GAN and the Earth Mover distance, DCGAN, Conditional GAN, Fréchet Inception Distance, Noise-Contrastive Estimation, Density ratio estimation, Restricted Boltzmann Machines and the partition function, Maxout activation, Nash equilibrium and minimax games, Parzen window density estimation
