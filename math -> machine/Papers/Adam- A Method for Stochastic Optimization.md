---
title: "Adam: A Method for Stochastic Optimization"
authors: ["Diederik P. Kingma", "Jimmy Ba"]
year: 2014
arxiv: "1412.6980"
url: https://arxiv.org/abs/1412.6980
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, optimization, theory]
---
## The Core Idea

Plain [[Backpropagation|gradient]] descent uses one learning rate for every weight. That is a bad deal when different weights see gradients of wildly different sizes — a rare word's embedding gets a gradient once every ten thousand steps, a bias in the first layer gets one every step. Two earlier fixes existed:

- **AdaGrad**: divide each weight's step by the square root of the *sum of all past squared gradients*. Great for sparse features. But that sum only ever grows, so the effective learning rate decays to zero and learning stalls.
- **RMSProp**: same idea but use a *moving average* of squared gradients instead of a sum, so it never dies. Good for objectives that change over time. But it has no momentum baked in, and it starts up wrong.

Adam is the obvious-in-hindsight merge: keep a running average of the gradient itself (momentum, the "first moment") **and** a running average of the squared gradient (the scale, the "second raw moment"), then divide one by the square root of the other. The name is *adaptive moment estimation*.

The genuinely new piece — the bit that is not just "RMSProp plus momentum" — is **initialisation bias correction**. Both running averages start at zero. With $\beta_2 = 0.999$, after 10 steps the second-moment estimate is still roughly $1/1000$th of where it should be, because it is an average dominated by the zero you seeded it with. Dividing by a number that is far too small gives a gigantic step. Adam divides each estimate by $1 - \beta^t$ to undo exactly this. This is why you can safely use $\beta_2$ very close to 1 (needed for sparse gradients) without the first few hundred steps blowing up.

The second thing worth carrying around is the **scale invariance**. Multiply every gradient by $c$: the first moment scales by $c$, the second by $c^2$, the square root by $c$, and they cancel. So the size of the step Adam takes does not depend on the size of the loss, the size of the gradients, or the scale of your data. The step size is approximately bounded by the hyperparameter $\alpha$ itself, which means $\alpha$ has an interpretation you can reason about: it is roughly how far in parameter space you are willing to move per step — a **trust region**.

> [!NOTE] Adaptive moment estimation
> Track two exponential moving averages per parameter: the mean of the gradient ($m_t$) and the mean of the squared gradient ($v_t$). Step in the direction $m_t / \sqrt{v_t}$. The ratio is a per-parameter signal-to-noise measure: large when the gradient consistently points one way, near zero when it is just noise. ^adaptive-moment

## The Methodology

The whole algorithm, per timestep $t$, with all operations element-wise:

$$g_t = \nabla_\theta f_t(\theta_{t-1})$$
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}$$

with $m_0 = v_0 = 0$. Defaults: $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

Memory cost: two extra float vectors the size of your parameters. That is it. No curvature matrix, no per-minibatch storage.

**Where the bias correction comes from.** Unroll the second-moment recursion:

$$v_t = (1-\beta_2)\sum_{i=1}^{t} \beta_2^{t-i} g_i^2$$

Take expectations. If the true second moment $\mathbb{E}[g^2]$ is roughly constant over the window, you can pull it out:

$$\mathbb{E}[v_t] = \mathbb{E}[g_t^2](1-\beta_2^t) + \zeta$$

where $\zeta \approx 0$ when the objective is stationary. The factor $(1-\beta_2^t)$ is purely an artefact of seeding with zero. Divide it out. Same derivation for $m_t$ with $\beta_1$.

**Why the step is bounded.** With $\epsilon = 0$ the effective step is $\Delta_t = \alpha \cdot \hat m_t / \sqrt{\hat v_t}$. Since $|\mathbb{E}[g]| / \sqrt{\mathbb{E}[g^2]} \le 1$ (mean over root-mean-square), typically $|\hat m_t / \sqrt{\hat v_t}| \lesssim 1$, so $|\Delta_t| \lesssim \alpha$. The only exception is extreme sparsity — a gradient that was zero at every step except this one — where the bound loosens to $\alpha (1-\beta_1)/\sqrt{1-\beta_2}$.

**Automatic annealing.** Near an optimum the gradient mean shrinks toward zero but the gradient variance does not. So $\hat m_t / \sqrt{\hat v_t}$ — what the authors loosely call the signal-to-noise ratio — falls, and steps naturally get smaller. You get a decaying schedule for free, without writing one.

**Relation to the ancestors, precisely.** Set $\beta_1 = 0$, let $1-\beta_2$ become infinitesimal (so $\hat v_t \to t^{-1}\sum_i g_i^2$), and replace $\alpha$ with $\alpha t^{-1/2}$; you get exactly AdaGrad's $\theta_t - \alpha g_t / \sqrt{\sum_i g_i^2}$. Note this equivalence **only** holds *with* bias correction — without it, $\beta_2 \to 1$ gives infinite bias and infinite steps. Drop the bias correction terms entirely and you have RMSProp-with-momentum, except RMSProp applies momentum to the already-rescaled gradient, whereas Adam rescales the already-averaged gradient.

$\hat v_t$ is an approximation to the diagonal of the Fisher information matrix, so Adam is a cheap, conservative cousin of natural gradient descent — conservative because it preconditions with the square *root* of the inverse diagonal, not the inverse itself.

**Convergence theory.** In the online convex setting, with $\alpha_t = \alpha/\sqrt{t}$, $\beta_{1,t} = \beta_1 \lambda^{t-1}$ (momentum decayed toward zero), bounded gradients and bounded parameter distance, and $\beta_1^2/\sqrt{\beta_2} < 1$, the regret $R(T) = \sum_t [f_t(\theta_t) - f_t(\theta^*)]$ is $O(\sqrt{T})$, so average regret $R(T)/T = O(1/\sqrt{T}) \to 0$. For sparse features the bound sharpens to $O(\log d \sqrt{T})$ versus $O(\sqrt{dT})$ for non-adaptive methods. Note the theory needs decaying $\beta_1$ and a $1/\sqrt{t}$ learning rate — neither of which anybody uses in practice.

**AdaMax.** Replace the $L^2$ norm of past gradients with the $L^\infty$ norm. Taking $p \to \infty$ in $v_t = \beta_2^p v_{t-1} + (1-\beta_2^p)|g_t|^p$ collapses to a beautifully simple recursion:

$$u_t = \max(\beta_2 \cdot u_{t-1}, |g_t|), \qquad \theta_t = \theta_{t-1} - \frac{\alpha}{1-\beta_1^t}\cdot\frac{m_t}{u_t}$$

No bias correction needed on $u_t$ (a max of zero and something is that something), and the step bound is exactly $|\Delta_t| \le \alpha$. Default $\alpha = 0.002$.

## Ablation Studies and Experiments

All comparisons use identical parameter initialisation and a dense grid search over each optimiser's own hyperparameters, reporting each one's best. Minibatch size 128 throughout.

**Logistic regression, MNIST** (784-dim pixels, L2-regularised, $\alpha_t = \alpha/\sqrt{t}$ to match theory). Adam ≈ SGD+Nesterov, both faster than AdaGrad.

**Logistic regression, IMDB bag-of-words** (10k most frequent words, extremely sparse; 50% dropout on the BoW features). Here AdaGrad beats SGD+Nesterov by a large margin, and Adam matches AdaGrad. This is the sparse-gradient claim confirmed: the second-moment rescaling is what lets you learn rare features, and Adam inherits it.

**MLP on MNIST**, two hidden layers of 1000 ReLU units.
- *Deterministic objective* (cross-entropy + L2, no dropout): compared against SFO, a minibatch quasi-Newton method. Adam is faster per iteration count **and** per wall-clock second. SFO is 5–10× slower per iteration because it maintains curvature, and its memory grows linearly with the number of minibatch partitions — infeasible on a GPU.
- *With dropout*: SFO assumes deterministic subfunctions and **failed to converge entirely**. Adam beat AdaGrad, RMSProp, SGD+Nesterov and AdaDelta.

**CNN on CIFAR-10**, `c64-c64-c128-1000`: three stages of 5×5 conv + 3×3 max-pool (stride 2), then a 1000-unit ReLU layer, whitened inputs, dropout on input and FC layers.

This is the most instructive result, and it is partly a **negative** one. In the first three epochs Adam and AdaGrad both drop the cost fast. Over 45 epochs, Adam and SGD+Nesterov converge *far* faster than AdaGrad, which lags badly. The diagnosis: in a CNN, $\hat v_t$ collapses toward zero after a few epochs and the update becomes dominated by $\epsilon$. The second moment is simply a poor description of the loss geometry here, unlike in fully-connected nets. What is actually doing the work in CNNs is the **first** moment — averaging away minibatch variance. So on convnets Adam's win over SGD+momentum is only marginal; its practical advantage is that it sets a sensible per-layer learning-rate scale automatically, instead of you hand-picking a smaller rate for the conv layers as people did with SGD.

**The bias-correction ablation (Section 6.4)** — the sharpest experiment in the paper. Train a [[Auto-Encoding Variational Bayes (VAE)|VAE]] (one hidden layer, 500 softplus units, 50-dim spherical Gaussian latent) and sweep $\beta_1 \in [0, 0.9]$, $\beta_2 \in \{0.99, 0.999, 0.9999\}$, $\log_{10}\alpha \in [-5,\dots,-1]$, with and without the $1/(1-\beta^t)$ terms. Removing them gives you RMSProp-with-momentum.

Result: with $\beta_2$ close to 1 and no bias correction, training is unstable — especially in the first few epochs — and often diverges. With correction, the *best* results across the whole sweep come from small $(1-\beta_2)$. So bias correction is exactly what unlocks the sparse-gradient-friendly setting. The gap widens toward the end of training, when gradients become sparser as hidden units specialise. **Adam was equal to or better than RMSProp at every hyperparameter setting tested.**

## Worth Remembering

- The theory and the practice disagree and everyone follows the practice. The $O(\sqrt{T})$ regret bound requires $\alpha_t \propto 1/\sqrt{t}$ and $\beta_{1,t}$ decaying to zero. Nobody does this. The bound is also known to be flawed — the later "On the Convergence of Adam and Beyond" (AMSGrad, ICLR 2018 best paper) found an error in the proof and constructed convex problems where Adam does not converge.
- **The $\epsilon$ is not cosmetic.** The CIFAR result shows $\hat v_t \to 0$ in convnets so the update is $\epsilon$-dominated. In practice $\epsilon$ acts as a floor on the denominator and quietly turns Adam back into something momentum-like. If you ever wonder why $\epsilon$ tuning matters for some architectures, this is why.
- **Weight decay is broken in Adam.** The paper uses "L2 weight decay" as a term in the loss. Because the L2 gradient then gets divided by $\sqrt{\hat v_t}$ like everything else, weights with big gradients get decayed less. This is the whole subject of [[Decoupled Weight Decay Regularization (AdamW)|AdamW]] — read that immediately after this.
- The $\alpha$-as-trust-region argument is the practical reason Adam needs so little tuning. Because the step is scale-invariant and bounded by $\alpha$, and you usually know roughly how far your good parameters live from your initialisation, you can often guess the right order of magnitude of $\alpha$ without a sweep. $10^{-3}$ became a default for a reason.
- Adam does *not* fix bad conditioning between parameters in the way a true second-order method would. $\hat v_t$ is a diagonal approximation to the Fisher — it captures per-parameter scale, not interactions. See [[Derivative#Hessian|the Hessian]] for what is actually being thrown away.
- **Section 7.2, the forgotten feature:** the last iterate of any stochastic method is noisy. One extra line, $\bar\theta_t = \beta_2 \bar\theta_{t-1} + (1-\beta_2)\theta_t$ with the same $1/(1-\beta_2^t)$ correction, gives you an exponential-moving-average of the *parameters*, which usually generalises better. This is EMA-of-weights, which was rediscovered and is now standard in diffusion model training.
- AdaMax is almost never used, which is a mild shame — it has a genuinely exact step bound $|\Delta_t| \le \alpha$ and needs no second-moment bias correction. If you have gradients with occasional huge outliers, the max-based denominator decays them away geometrically rather than letting one spike poison a moving average for hundreds of steps.
- Author ordering was decided by a coin flip over a Google Hangout, which is the correct way to handle equal contribution.

## Links

Related: [[Backpropagation]] · [[Momentum]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Derivative]] · [[Deep Learning]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Batch Normalization]] · [[How Does Batch Normalization Help Optimization]] · [[Regularization]] · [[Cross Entropy]] · [[Loss, Objectives, and Business Alignment]] · [[MLE_L4_Question_Bank]] · [[Mixed Precision training]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Distributed Representations of Words and Phrases (negative sampling)]]

New topics worth writing: AdaGrad, RMSProp, AMSGrad and the Adam convergence flaw, Natural gradient descent and the Fisher information matrix, Nesterov accelerated gradient, Polyak–Ruppert averaging and weight EMA, Online convex optimisation and regret bounds, AdaDelta, Learning rate warmup (and why Adam needs it at scale)
