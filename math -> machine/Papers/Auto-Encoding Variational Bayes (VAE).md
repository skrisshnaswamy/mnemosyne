---
title: "Auto-Encoding Variational Bayes (VAE)"
authors: ["Diederik P Kingma", "Max Welling"]
year: 2013
arxiv: "1312.6114"
url: https://arxiv.org/abs/1312.6114
priority: Must-Read
read_on: 2026-08-22
tags: [paper, diffusion, vision, theory]
---
## The Core Idea

You want a model that can **generate** data — faces, digits — by first drawing a hidden code $\mathbf{z}$ from a simple prior, then decoding it into an image. Training such a model by maximum likelihood means computing

$$p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{z})\,p_\theta(\mathbf{x}\mid\mathbf{z})\,d\mathbf{z}$$

and that integral is hopeless the moment the decoder is a neural net. You also cannot compute the posterior $p_\theta(\mathbf{z}\mid\mathbf{x})$ — "which code produced this image?" — so the EM algorithm is out too.

The standard escape is **variational inference**: give up on the true posterior, fit an easier distribution $q_\phi(\mathbf{z}\mid\mathbf{x})$ to it, and maximise a lower bound on $\log p_\theta(\mathbf{x})$ instead. The problem in 2013 was that this bound is an *expectation over a distribution whose parameters you are optimising*. Taking a [[Derivative#Gradient|gradient]] through a sampling step is not something [[Backpropagation]] knows how to do. The available estimator (the score-function / REINFORCE trick) works but its variance is so large it is useless in practice.

Two moves fix this.

**Move 1 — the reparameterisation trick.** Instead of sampling $\mathbf{z}\sim\mathcal{N}(\mu,\sigma^2)$, sample noise $\epsilon\sim\mathcal{N}(0,1)$ *outside* the model and compute

$$\mathbf{z} = \mu + \sigma\odot\epsilon.$$

Same distribution. But now $\mathbf{z}$ is a deterministic, differentiable function of $\mu,\sigma$, and the randomness sits in a leaf node that has no parameters. Gradients flow straight through. The sampling stopped being an obstacle and became just another layer.

**Move 2 — amortised inference.** Do not run an optimisation loop per datapoint to find its posterior. Train one neural network (the *recognition model*, or encoder) that maps any $\mathbf{x}$ to the parameters $(\mu,\sigma)$ of its approximate posterior in a single forward pass. Cost per datapoint drops to one forward pass; the encoder generalises to unseen data for free.

Put the two together and the whole thing looks exactly like an autoencoder with a noisy bottleneck, trainable end to end by minibatch SGD. That is the **variational auto-encoder**.

> [!NOTE] Reparameterisation trick
> Rewrite a sample from $q_\phi$ as $\mathbf{z}=g_\phi(\epsilon,\mathbf{x})$ where $\epsilon$ comes from a fixed, parameter-free noise distribution. Turns $\nabla_\phi \mathbb{E}_{q_\phi}[f(\mathbf{z})]$ into $\mathbb{E}_{p(\epsilon)}[\nabla_\phi f(g_\phi(\epsilon,\mathbf{x}))]$, which is just backprop. ^reparameterisation-trick

## The Methodology

**The bound.** Start from an identity. For any $q_\phi$,

$$\log p_\theta(\mathbf{x}^{(i)}) = D_{KL}\!\left(q_\phi(\mathbf{z}\mid\mathbf{x}^{(i)}) \,\|\, p_\theta(\mathbf{z}\mid\mathbf{x}^{(i)})\right) + \mathcal{L}(\theta,\phi;\mathbf{x}^{(i)}).$$

[[KL Divergence]] is never negative, so $\mathcal{L}$ is a lower bound on the log-likelihood. Two readings of $\mathcal{L}$ matter:

$$\mathcal{L} = \mathbb{E}_{q_\phi}\!\left[\log p_\theta(\mathbf{x},\mathbf{z}) - \log q_\phi(\mathbf{z}\mid\mathbf{x})\right]$$

$$\mathcal{L} = \underbrace{-D_{KL}\!\left(q_\phi(\mathbf{z}\mid\mathbf{x}) \,\|\, p_\theta(\mathbf{z})\right)}_{\text{regulariser}} + \underbrace{\mathbb{E}_{q_\phi}\!\left[\log p_\theta(\mathbf{x}\mid\mathbf{z})\right]}_{\text{negative reconstruction error}}$$

The second form is the useful one. Maximising it pushes the encoder's output distribution towards the prior *and* asks the decoder to rebuild the input. Note the [[Regularization]] term is not a hyperparameter you tune — it falls out of the maths, unlike the hand-tuned penalties in denoising or contractive autoencoders.

> [!NOTE] Evidence Lower Bound (ELBO)
> $\mathcal{L}(\theta,\phi;\mathbf{x})\le \log p_\theta(\mathbf{x})$. The gap is exactly the KL between your approximate posterior and the true one. Tightening the bound and fitting the data are the same optimisation. ^elbo

**Two estimators.**

- $\tilde{\mathcal{L}}^A$ — the generic one. Sample $\mathbf{z}^{(l)}=g_\phi(\epsilon^{(l)},\mathbf{x})$ and average $\log p_\theta(\mathbf{x},\mathbf{z}^{(l)}) - \log q_\phi(\mathbf{z}^{(l)}\mid\mathbf{x})$. Works for any $q$ and $p$.
- $\tilde{\mathcal{L}}^B$ — when the KL term has a closed form (Gaussian $q$, Gaussian prior), compute it analytically and only Monte-Carlo the reconstruction term. **Lower variance**, and this is what they use.

**The concrete VAE.** Prior $p(\mathbf{z})=\mathcal{N}(\mathbf{0},\mathbf{I})$, no parameters. Encoder is a one-hidden-layer MLP with $\tanh$, outputting $\mu$ and $\log\sigma^2$ (predicting the log-variance keeps $\sigma^2$ positive and the scale well-behaved). Decoder is another one-hidden-layer $\tanh$ MLP, with Bernoulli outputs for binary MNIST (so the reconstruction term is exactly a [[Cross Entropy]] over pixels) or Gaussian outputs for the continuous Frey Face data.

With diagonal Gaussian $q$ and standard-normal prior, the whole per-datapoint objective is:

$$\mathcal{L} \simeq \frac{1}{2}\sum_{j=1}^{J}\left(1+\log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right) + \frac{1}{L}\sum_{l=1}^{L}\log p_\theta(\mathbf{x}\mid\mathbf{z}^{(l)}),\qquad \mathbf{z}^{(l)}=\mu+\sigma\odot\epsilon^{(l)}.$$

That first sum is the analytic Gaussian KL. $J$ is the latent dimension. Everything is differentiable; hand it to [[Pytorch Autograd]] and you are done.

**Training details that mattered.**
- Minibatch $M=100$, and **$L=1$ sample per datapoint**. One noise draw per image is enough as long as the batch is big — the minibatch itself averages away the variance.
- Adagrad, global step size picked from $\{0.01, 0.02, 0.1\}$ on early training performance.
- Weight init $\mathcal{N}(0,0.01)$; small weight decay corresponding to a $\mathcal{N}(0,\mathbf{I})$ prior on $\theta$, so the whole thing is approximate MAP on the generative parameters.
- 500 hidden units for MNIST, 200 for Frey Face (smaller dataset, less capacity to avoid overfitting).
- Frey Face decoder means squeezed into $(0,1)$ with a sigmoid.

**When does the trick apply?** Three recipes for building $g_\phi$: (1) invertible CDF — Exponential, Cauchy, Logistic, Gumbel, Weibull; (2) location-scale families — Gaussian, Laplace, Student's t, Uniform; (3) composition — Log-Normal as $\exp$ of a Gaussian, Gamma as a sum of Exponentials, Dirichlet from Gammas. Discrete latents are *not* covered — that is the trick's hard boundary.

## Ablation Studies and Experiments

**Baseline 1 — wake-sleep** (Hinton et al. 1995), the only other online method for this model class. Same encoder architecture, same optimiser. Measured by the ELBO per datapoint on MNIST and Frey Face, at latent dimensions $N_\mathbf{z} \in \{3,5,10,20,200\}$. AEVB converged **considerably faster and to a better bound in every single setting**. Wake-sleep's weakness is structural: it optimises two objectives that together do not correspond to any bound on the marginal likelihood, so it has no single quantity that is guaranteed to improve.

**Baseline 2 — Monte Carlo EM with a Hybrid Monte Carlo sampler** (10 leapfrog steps, step size auto-tuned to 90% acceptance, then 5 weight updates per sample). Compared on estimated marginal likelihood, MNIST, with 3 latent dimensions and 100 hidden units. MCEM is competitive on a *small* training set but it is not an online algorithm — it needs an expensive sampling loop per datapoint, so it cannot be run on full MNIST at all. This is the practical argument for amortisation: [[Markov Chain Monte Carlo]] per datapoint does not scale.

**The surprising ablation — latent dimensionality.** Going from 2 to 5 to 10 to 20 to 200 latent dimensions did **not** cause overfitting. Normally more capacity in a bottleneck means memorisation. Here the $-D_{KL}(q\|p)$ term does the work: unused latent dimensions get collapsed onto the prior and contribute nothing, so extra capacity is automatically switched off. The bound regularises itself.

**What did not work / was rejected.**
- The naive score-function gradient estimator $\mathbb{E}[f(\mathbf{z})\nabla_\phi \log q_\phi(\mathbf{z})]$ — correct in expectation, but variance so high it is "impractical for our purposes". This is the whole motivation for the paper.
- Mean-field VB — needs analytic expectations under $q$, which do not exist once the likelihood is a nonlinear net.
- The marginal-likelihood estimator in Appendix D (HMC samples + a fitted density, then a harmonic-mean-style ratio) becomes **unreliable above ~5 latent dimensions**. That is why the marginal likelihood comparison is stuck at $J=3$ while the ELBO comparison goes to 200.

**Qualitative results.** With a 2D latent space they sweep a grid over the unit square, push it through the inverse Gaussian CDF to get $\mathbf{z}$ values, and decode. The result is a smooth manifold of digits and of faces — stroke style and digit identity vary continuously, face pose and expression vary continuously. Latent space is not a lookup table; nearby codes decode to similar images.

## Worth Remembering

- **Speed context.** ~20–40 minutes per million training samples on an Intel Xeon at an effective 40 GFLOPS. Single hidden layer, CPU, 2013. The idea is cheap.
- **The name is two things.** *SGVB* is the gradient estimator (general — works for any continuous latent variable model, including full Bayesian inference over global parameters, derived in Appendix F). *AEVB* is the algorithm that adds an amortised recognition network. *VAE* is the specific case where both are MLPs. People say "VAE" for all three.
- **Independent co-discovery.** Rezende, Mohamed & Wierstra (2014) derived the same trick as "stochastic backpropagation" at essentially the same time. When two groups find it in the same year, the idea was ripe.
- **Discrete latents are the blocker.** Nothing here handles categorical $\mathbf{z}$ — that had to wait for Gumbel-Softmax / Concrete relaxations. Wake-sleep, for all its weaknesses, does handle discrete variables.
- **The bound is loose and you never know by how much.** You are reporting $\mathcal{L}$, not $\log p(\mathbf{x})$, and the gap is $D_{KL}(q\|p_\theta(\mathbf{z}\mid\mathbf{x}))$, which you cannot compute. A better ELBO could mean a better model *or* a better encoder. Comparing models by ELBO across papers is dangerous.
- **The diagonal-Gaussian $q$ is a real restriction.** The paper assumes the true posterior is "approximately Gaussian with approximately diagonal covariance". When it isn't, the bound stays loose no matter how long you train. Normalising flows exist precisely to relax this.
- **Practical failure mode not discussed here:** posterior collapse. With a powerful autoregressive decoder, the model can drive $D_{KL}\to 0$, ignore $\mathbf{z}$ entirely, and still get a decent bound. Fine with a weak MLP decoder; a serious problem later.
- **Contrast with [[Generative Adverserial Network]]** (one year later): VAEs optimise an explicit likelihood bound, so training is stable and you get an encoder, but samples are blurry because the Gaussian/Bernoulli decoder averages over uncertainty. GANs get sharp samples with no likelihood and no encoder, and suffer [[Mode Collapse]]. The blur is a direct consequence of the pixel-wise likelihood, not of the variational machinery.
- The reparameterisation trick escaped generative modelling entirely — it is the standard low-variance estimator wherever you need gradients through a sampling step, including stochastic policies and Bayesian neural nets.

## Links
Related: [[KL Divergence]] · [[Cross Entropy]] · [[Backpropagation]] · [[Pytorch Autograd]] · [[Regularization]] · [[Markov Chain Monte Carlo]] · [[Hamiltonian Monte Carlo]] · [[Generative Adverserial Network]] · [[Mode Collapse]] · [[Random variable]] · [[Uncertainty]] · [[Momentum]] · [[Deep Learning]]

New topics worth writing: Variational Inference, Evidence Lower Bound, Amortised Inference, Score-Function / REINFORCE Estimator, Posterior Collapse, Normalising Flows, Gumbel-Softmax, Wake-Sleep Algorithm, Adagrad, Expectation-Maximisation
