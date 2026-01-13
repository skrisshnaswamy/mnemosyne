---
title: "Understanding the difficulty of training deep feedforward networks (Xavier init)"
authors: ["Glorot & Bengio"]
year: 2010
url: https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, diffusion]
---
## The Core Idea

Before 2006, deep networks trained by plain [[Backpropagation]] from random weights just did not work. The field's fix was unsupervised pre-training — stack RBMs or denoising autoencoders, then fine-tune. This paper asks a different question: *why* does the plain version fail? Instead of inventing a new algorithm, Glorot and Bengio just **watched** the network. They plotted the activations and the gradients, layer by layer, epoch by epoch.

Two findings come out of that watching.

**One: the sigmoid is the villain.** With sigmoid units and small random weights, the top hidden layer slams to 0 (its saturated end) within a few epochs and stays there. Gradients cannot pass through a saturated sigmoid, so the layers below learn nothing. The 5-layer sigmoid net never escapes. The reason is that sigmoid's output mean is not 0. The softmax output layer learns its biases fast, then wants $Wh \to 0$; it gets there by pushing $h \to 0$. For `tanh`, $h=0$ is the *linear* middle of the curve — gradients flow fine. For sigmoid, $h=0$ is the *dead* end of the curve.

**Two: the initialisation scale must depend on layer width.** If you write down the variance of the signal going forward and the variance of the gradient going backward, each layer multiplies them by $n \cdot \text{Var}[W]$. If that factor is not 1, the product over $d$ layers either explodes or vanishes geometrically. The common heuristic of the day, $W \sim U[-1/\sqrt{n},\, 1/\sqrt{n}]$, gives $n\text{Var}[W] = 1/3$ — so the backward signal shrinks by a factor of 3 per layer. Fixing that one constant is the whole "Xavier init" contribution.

> [!NOTE] Xavier / normalized initialization
> Draw weights uniformly with a spread set by the **sum** of fan-in and fan-out: $W \sim U\left[-\frac{\sqrt6}{\sqrt{n_j + n_{j+1}}},\ \frac{\sqrt6}{\sqrt{n_j+n_{j+1}}}\right]$, giving $\text{Var}[W] = \frac{2}{n_j+n_{j+1}}$. Chosen so that both activation variance and gradient variance stay roughly constant across depth. ^xavier-init

What it unlocks: a big chunk of the gap between "plain supervised deep net" and "deep net with unsupervised pre-training" disappears once initialisation is fixed. The magic of pre-training was, partly, just being in a sane scale regime.

## The Methodology

**The derivation.** Assume the activation $f$ is symmetric with $f'(0)=1$ (true for `tanh`, softsign), assume you sit in the linear regime at init, and assume weights are independent. Let $z^i$ be layer $i$'s activations and $s^i = z^i W^i + b^i$ its pre-activations. Then

$$\text{Var}[z^i] = \text{Var}[x] \prod_{i'=0}^{i-1} n_{i'}\,\text{Var}[W^{i'}]$$

$$\text{Var}\!\left[\frac{\partial C}{\partial s^i}\right] = \text{Var}\!\left[\frac{\partial C}{\partial s^d}\right] \prod_{i'=i}^{d} n_{i'+1}\,\text{Var}[W^{i'}]$$

Wanting the forward variance constant gives $n_i \text{Var}[W^i] = 1$. Wanting the backward variance constant gives $n_{i+1}\text{Var}[W^i] = 1$. You cannot have both unless layers are equal width, so take the harmonic-ish compromise $\text{Var}[W^i] = \frac{2}{n_i + n_{i+1}}$. For a uniform distribution on $[-a,a]$, $\text{Var} = a^2/3$, which is where the $\sqrt6$ comes from.

A neat corollary from eq. 14: the **weight** gradient variance is the same at every layer even when the **back-propagated** gradient is vanishing. The two shrinking/growing products cancel. So "my weight gradients look fine across layers" is not evidence that your signal propagation is healthy — a trap worth remembering.

**Setup.** Feedforward nets, 1–5 hidden layers, 1000 units per layer, softmax output, negative log-likelihood loss (see [[Cross Entropy]]). Plain SGD, minibatch of 10, $\theta \leftarrow \theta - \epsilon g$, no momentum, no weight decay. Learning rate and depth tuned separately per model on validation error after 5M updates. Biases initialised to 0.

**Activations compared.** Sigmoid $1/(1+e^{-x})$, $\tanh(x)$, and softsign $x/(1+|x|)$. Softsign is `tanh`-shaped but its tails decay polynomially instead of exponentially, so it approaches $\pm1$ much more gently.

**Data.** Shapeset-3×2 (synthetic 32×32 shape images, 9 classes, infinite stream — chosen so results reflect optimisation, not overfitting), MNIST, CIFAR-10, and a small 37×37 grayscale ImageNet subset.

**Diagnostic.** Track mean/std/98th-percentile/histograms of activations at each layer over 300 fixed test examples, across training. Also track singular values of the layer Jacobian $J^i = \partial z^{i+1}/\partial z^i$ — the target is average singular value near 1.

## Ablation Studies and Experiments

Final test error, 5 hidden layers, "N" = normalized (Xavier) init:

| Activation | Shapeset | MNIST | CIFAR-10 | Small-ImageNet |
|---|---|---|---|---|
| Softsign | 16.27 | 1.64 | 55.78 | 69.14 |
| Softsign N | 16.06 | 1.72 | 53.8 | 68.13 |
| Tanh | 27.15 | 1.76 | 55.9 | 70.58 |
| **Tanh N** | **15.60** | **1.64** | **52.92** | **68.57** |
| Sigmoid | 82.61 | 2.21 | 57.28 | 70.66 |

The headline: `tanh` on Shapeset goes from **27.15% → 15.60%** error purely by changing the initialisation constant. Nothing else. Sigmoid at 82.61% on a 9-class task is barely above chance-ish behaviour — it is functionally broken. RBF SVM baseline on 100k Shapeset examples: 59.47% error; 5-layer tanh-N: 50.47% on the same set.

**Jacobian singular values.** Standard init gives average ratio ≈ **0.5** per layer; Xavier gives ≈ **0.8**. Closer to 1 = better.

**The saturation dynamics (Figures 2–4).** Sigmoid: top hidden layer crashes to 0 immediately. At depth 4 it *escapes* around epoch 100 — and exactly as the top layer desaturates, the first hidden layer starts saturating. At depth 5 it never escapes. So the plateaus you see in training curves have a mechanical explanation: units sitting in saturation, slowly crawling out. `tanh` with standard init saturates layer 1 first, then 2, then 3 — a wave moving upward, which the authors admit they cannot explain. Softsign saturates all layers together and more gently, and its final activation histogram peaks around $\pm0.6$–$0.8$ — the "knee", where the unit is genuinely non-linear *and* gradients still flow. `tanh`'s histogram piles up at $-1$, $0$, $+1$: either dead or linear.

**Loss function ablation.** Cross-entropy vs. squared error, plotted as a surface over two weights: the quadratic cost has visibly more severe plateaus. Old news by 2010 (Solla et al., 1988) but they re-stress it.

**Gradient flow during training (Figures 7, 9).** With standard init, backprop gradients shrink going downward at initialisation — but this trend **reverses quickly** once training starts, and later the lower layers get the *larger* gradients. Xavier keeps weight-gradient variance matched across layers throughout training, which is probably the real benefit (matched magnitudes = better conditioning, fewer per-layer learning-rate problems).

**What did not work as well as hoped:**
- Second-order methods alone (diagonal-Hessian learning rates à la LeCun's *Efficient Backprop*, and gradient-variance-based rates) applied to standard-init tanh **improved things but did not reach Xavier's result**. Combining both was better still — the interpretation being that a good init removes the *between-layer* discrepancy so the Hessian estimate can spend itself on *between-unit* discrepancies.
- Xavier init helps softsign only marginally (16.27 → 16.06 on Shapeset; on MNIST it is slightly *worse*, 1.64 → 1.72). Softsign is already robust to init because its non-linearity is gentle. The init fix matters most exactly where the activation is fragile.
- Sigmoid is not rescued by anything here. The paper's advice is simply: don't.

## Worth Remembering

- The derivation assumes **linear regime, independent weights, equal input variances**. All three break the moment training starts. The authors say this explicitly: "we cannot use simple variance calculations in our theoretical analysis because the weights values are not anymore independent." Xavier is an *initialisation* argument, not a training-time guarantee. That gap is what [[Batch Normalization]] later attacks directly, and what [[How Does Batch Normalization Help Optimization]] reinterprets as smoothing.
- The factor-of-2 story: Xavier assumes $f'(0)=1$ and symmetry. ReLU (see [[ImageNet Classification with Deep CNNs (AlexNet)|AlexNet]]) kills half the units, so its variance-preserving constant is $2/n$, not $2/(n_{in}+n_{out})$ — that is He initialisation, published five years later. Using Xavier with ReLU underscales by roughly $\sqrt2$ per layer.
- **The trap worth internalising**: constant weight-gradient magnitude across layers coexists with vanishing back-propagated gradients (eq. 14 explains why). Do not use weight-gradient norms as your only health check. Look at activations and at $\partial C/\partial s^i$.
- Unsupervised pre-training's advantage shrank a lot once init was fixed. This is an early instance of a recurring pattern — an elaborate mechanism turns out to be doing something a one-line scale fix also does. Relates to the skepticism in [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] and [[On the Difficulty of Evaluating Baselines]].
- Softsign never caught on, but its ablation is the interesting one: it says the *shape of the saturation tail* matters as much as the symmetry. Polynomial tails give you a wide band of "non-linear but still differentiable" operating points. Modern smooth activations (GELU, SiLU) rediscover the same intuition.
- The connection the authors flag to [[Long Short-Term Memory (Neural Computation)|recurrent nets]]: an unrolled RNN is a very deep net with *shared* weights, so the same $\prod n\text{Var}[W]$ product is exactly the vanishing-gradient problem of Bengio et al. 1994.
- Practical caveat: the fix targets a **multiplicative** per-layer effect. It buys you depth, not stability under a bad learning rate. Everything here was plain SGD with no [[Momentum]], no [[Regularization]], no residuals. [[Deep Residual Learning for Image Recognition (ResNet)|Residual connections]] later made the depth question mostly moot by giving gradients a path with Jacobian exactly 1.

## Links

Related: [[Backpropagation]] · [[Deep Learning]] · [[Cross Entropy]] · [[Batch Normalization]] · [[How Does Batch Normalization Help Optimization]] · [[Layer Normalization]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Long Short-Term Memory (Neural Computation)]] · [[Gradient-Based Learning Applied to Document Recognition (LeNet)]] · [[Derivative#Jacobian|Jacobian]] · [[Derivative#Hessian|Hessian]] · [[Vector Jacobian Product]] · [[Momentum]] · [[Gated Activation]]

New topics worth writing: He initialization, Vanishing and exploding gradients, Saturating activation functions, Softsign, Denoising autoencoders, Greedy layer-wise pre-training, Singular value decomposition and signal propagation, GELU and SiLU, Dynamical isometry and orthogonal initialization
