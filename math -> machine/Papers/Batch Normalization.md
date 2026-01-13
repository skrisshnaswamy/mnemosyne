---
title: "Batch Normalization"
authors: ["Sergey Ioffe", "Christian Szegedy"]
year: 2015
arxiv: "1502.03167"
url: https://arxiv.org/abs/1502.03167
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, optimization, vision, theory]
---
## The Core Idea

Deep networks are trained layer by layer with [[Backpropagation|backprop]], but every layer's input is produced by the layers below it. When those lower layers update, the *distribution* of numbers arriving at layer 7 shifts — new mean, new spread. Layer 7 then has to spend gradient steps just re-adapting to the new scale of its inputs instead of learning something useful. Ioffe and Szegedy name this **internal covariate shift** and attack it directly.

> [!NOTE] Internal covariate shift
> The change in the distribution of a hidden layer's inputs during training, caused by the parameters of all layers below it changing. ^internal-covariate-shift-bn

The fix sounds obvious — normalise each layer's inputs to zero mean and unit variance, the same thing everyone already did to the *input* data. The reason it did not exist before is a subtle trap the paper spells out. If you normalise as a separate step *outside* the gradient computation, the optimiser does not know the normalisation is happening. Concretely: take a layer $x = u + b$ and normalise $\hat{x} = x - E[x]$. Gradient descent updates $b \leftarrow b + \Delta b$. But then

$$u + (b + \Delta b) - E[u + (b+\Delta b)] = u + b - E[u+b]$$

The output is *unchanged*. The loss is unchanged. So the gradient keeps pushing $b$ in the same direction forever, and $b$ grows without bound until the model blows up. The authors saw this happen.

So the real idea is: **make normalisation a differentiable layer inside the network, and backprop through the mean and variance too.** The two extra tricks that make it practical:

1. Use **mini-batch statistics**, not whole-dataset statistics. The batch mean and variance are functions of the current batch, so they sit naturally in the computation graph.
2. Normalise each feature **independently** (per-scalar), not full whitening. Full whitening needs the covariance matrix and its inverse square root — expensive, and singular anyway when batch size < number of activations.

Then add a learned scale $\gamma$ and shift $\beta$ per feature, so the layer can *undo* the normalisation if that turns out to be optimal. Without this, normalising a sigmoid's input would trap it in the linear part of the curve and cost you representation power.

What it unlocks: 14× fewer training steps to reach the same ImageNet accuracy, learning rates 30× higher without divergence, sigmoid networks that actually train, and a free regularisation effect that let them delete Dropout.

## The Methodology

**The BN transform.** For one scalar activation $x$ over a mini-batch $\mathcal{B} = \{x_1 \dots x_m\}$:

$$\mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^m x_i \qquad \sigma^2_\mathcal{B} = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_\mathcal{B})^2$$

$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}} \qquad y_i = \gamma \hat{x}_i + \beta$$

$\epsilon$ is a small constant for numerical stability. $\gamma, \beta$ are learned per feature, trained by the same SGD as everything else. Setting $\gamma = \sqrt{\text{Var}[x]}$, $\beta = E[x]$ recovers the identity, so capacity is preserved.

**The backward pass.** This is the part that matters — the gradient flows through $\mu_\mathcal{B}$ and $\sigma^2_\mathcal{B}$, not around them:

$$\frac{\partial \ell}{\partial \sigma^2_\mathcal{B}} = \sum_i \frac{\partial \ell}{\partial \hat{x}_i}(x_i - \mu_\mathcal{B})\cdot \frac{-1}{2}(\sigma^2_\mathcal{B}+\epsilon)^{-3/2}$$

$$\frac{\partial \ell}{\partial x_i} = \frac{\partial \ell}{\partial \hat{x}_i}\frac{1}{\sqrt{\sigma^2_\mathcal{B}+\epsilon}} + \frac{\partial \ell}{\partial \sigma^2_\mathcal{B}}\frac{2(x_i - \mu_\mathcal{B})}{m} + \frac{\partial \ell}{\partial \mu_\mathcal{B}}\frac{1}{m}$$

$$\frac{\partial \ell}{\partial \gamma} = \sum_i \frac{\partial \ell}{\partial y_i}\hat{x}_i \qquad \frac{\partial \ell}{\partial \beta} = \sum_i \frac{\partial \ell}{\partial y_i}$$

Note the consequence: the output for example $i$ depends on the *other examples in the batch*. BN is not a per-example function during training.

**Where to put it.** Immediately *before* the nonlinearity, on $x = Wu + b$, not on the layer input $u$. Reason: $u$ is the output of a previous nonlinearity (e.g. ReLU), so its distribution is sparse and lopsided; fixing its first two moments does not fix its shape. $Wu + b$ is a sum of many terms, so it is closer to Gaussian, and matching mean/variance actually stabilises it.

The bias $b$ becomes redundant — mean subtraction cancels it — so drop it. The layer becomes $z = g(\text{BN}(Wu))$.

**Convolutional layers.** To respect the convolutional property (same normalisation at every spatial location), pool the statistics across batch *and* space. For batch size $m$ and feature maps of size $p \times q$, the effective batch is $m' = m \cdot pq$. One $(\gamma, \beta)$ pair **per feature map**, not per activation.

**Inference.** Batch statistics are undesirable at test time — you want a deterministic function of one input. So freeze population statistics collected over training mini-batches:

$$E[x] \leftarrow E_\mathcal{B}[\mu_\mathcal{B}] \qquad \text{Var}[x] \leftarrow \frac{m}{m-1}E_\mathcal{B}[\sigma^2_\mathcal{B}]$$

(the $\frac{m}{m-1}$ is the unbiased variance correction). At inference BN collapses to a single affine map that can be folded into the preceding weights:

$$y = \frac{\gamma}{\sqrt{\text{Var}[x]+\epsilon}}\cdot x + \left(\beta - \frac{\gamma E[x]}{\sqrt{\text{Var}[x]+\epsilon}}\right)$$

**Why higher learning rates work.** BN is scale-invariant in the weights. For a scalar $a$, $\text{BN}(Wu) = \text{BN}((aW)u)$, and:

$$\frac{\partial \text{BN}((aW)u)}{\partial u} = \frac{\partial \text{BN}(Wu)}{\partial u}, \qquad \frac{\partial \text{BN}((aW)u)}{\partial (aW)} = \frac{1}{a}\cdot\frac{\partial \text{BN}(Wu)}{\partial W}$$

So blowing up the weights does not blow up the gradient flowing backwards, and *bigger weights get smaller gradients* — a self-stabilising loop. The authors also conjecture (assuming Gaussian, uncorrelated activations and local linearity $F(\hat{x}) \approx J\hat{x}$) that $JJ^T = I$, so the layer [[Derivative#Jacobian|Jacobian]] has singular values near 1 and gradient magnitude is preserved. They admit this argument is loose.

**Training setup.** [[ImageNet Classification with Deep CNNs (AlexNet)|ImageNet]] LSVRC2012. A modified Inception: $5\times5$ convs replaced by two stacked $3\times3$, three $28\times28$ inception modules instead of two, 13.6M parameters, no fully-connected layers except the softmax. SGD with [[Momentum|momentum]], mini-batch size 32, distributed training. Baseline learning rate 0.0015.

**Changes you must make alongside BN** (just dropping BN in is not enough):
- Learning rate up 5× (0.0075) or 30× (0.045)
- Remove Dropout entirely
- $L_2$ weight decay reduced 5×
- Learning-rate decay 6× faster (the net trains faster, so decay must keep up)
- Remove Local Response Normalization
- Shuffle harder — within-shard shuffling so the same examples do not co-occur in a batch, worth ~1% validation accuracy
- Weaker photometric distortion, since each example is seen fewer times

## Ablation Studies and Experiments

**MNIST sanity check.** 3 hidden layers, 100 sigmoid units each, [[Cross Entropy|cross-entropy]] loss, 50000 steps, batch 60. BN version reaches higher test accuracy. More interesting is the diagnostic plot: they track the {15, 50, 85}th percentiles of a sigmoid's input over training. Without BN the distribution wanders in both mean and variance; with BN it is nearly flat. This is the paper's only direct evidence that internal covariate shift is the mechanism.

**ImageNet, single network** — steps to reach Inception's 72.2%, and max accuracy:

| Model | Steps to 72.2% | Max accuracy |
|---|---|---|
| Inception | $31.0\times10^6$ | 72.2% |
| BN-Baseline (BN only, same LR) | $13.3\times10^6$ | 72.7% |
| BN-x5 (LR 0.0075 + all mods) | $2.1\times10^6$ | 73.0% |
| BN-x30 (LR 0.045) | $2.7\times10^6$ | **74.8%** |
| BN-x5-Sigmoid | never | 69.8% |

What this actually tells you:
- BN *alone* buys 2.3× fewer steps. The 14× headline needs the higher learning rate and the other changes on top. The learning rate is doing most of the work; BN is what makes that learning rate survivable.
- **The same 5× learning rate on plain Inception drove the parameters to machine infinity.** That is the cleanest ablation in the paper.
- BN-x30 is *slower early* than BN-x5 but ends 1.8 points higher. Big LR trades early progress for a better final basin.
- Plain Inception with sigmoid instead of ReLU **never beats chance (1/1000)**. With BN, sigmoid gets to 69.8%. BN makes a saturating nonlinearity trainable at depth — that is a qualitative, not incremental, result.

**Ensemble.** 6 BN-x30 variants (some with 5–10% Dropout vs 40% originally, some with increased initial conv weights, some with per-activation rather than per-feature-map BN on the last hidden layers), averaged class probabilities, 144 crops. Top-5 validation error **4.9%**, test error 4.82%, beating the previous best of 4.94% and exceeding the reported human-rater estimate. Single BN-Inception with multicrop: 21.99% top-1 / 5.82% top-5.

**What did not work / what got dropped:**
- Normalising outside the gradient step — models blew up, $b$ grew unboundedly.
- Full whitening (decorrelation) — too expensive, needs $\text{Cov}[x]^{-1/2}$, and singular when $m <$ number of activations.
- Normalising the layer input $u$ instead of the pre-activation $Wu+b$ — rejected because post-nonlinearity distributions are sparse and non-Gaussian, so moment-matching does not stabilise them.
- Per-example or per-location statistics (Lyu & Simoncelli style) — discards the absolute scale of activations and changes what the network can represent.
- Local Response Normalization became useless.
- Dropout became mostly useless — BN's batch noise already regularises.

## Worth Remembering

- **The regularisation is an accident of the batch dependence.** The same example gets a slightly different normalisation depending on who else is in its batch. That is stochastic noise injected into the forward pass — exactly Dropout's mechanism by a different route. This is why the shuffling experiment gave +1%: more batch randomness, more regularisation. It is also why BN degrades at small batch sizes, which the paper does not test.
- **The stated mechanism is probably wrong.** [[How Does Batch Normalization Help Optimization]] (Santurkar et al. 2018) showed you can inject *deliberate* distributional noise after BN — deliberately restoring internal covariate shift — and training is still fast. Their claim is that BN smooths the loss landscape ($\beta$-smoothness), not that it fixes covariate shift. Read this note as: the *technique* is correct and enormous, the *explanation* is folklore.
- Train/inference asymmetry is a real engineering hazard. Forgetting to switch to eval mode (frozen population statistics) is one of the most common bugs in practice. Distribution shift between the moving-average statistics and your actual test data silently degrades accuracy.
- BN makes the loss non-decomposable across examples. This breaks things you might take for granted: per-example gradients, some contrastive setups, and reproducibility across different batch compositions.
- The bias term becomes dead weight. Any `Conv2d` followed by `BatchNorm2d` should have `bias=False`.
- BN unlocked the depth regime that [[Deep Residual Learning for Image Recognition (ResNet)|ResNet]] built on months later. The scale-invariance property is also why BN + weight decay interact in strange ways — see [[Decoupled Weight Decay Regularization (AdamW)]].
- The authors flag RNNs as future work, since "internal covariate shift and vanishing/exploding gradients may be especially severe" there. BN did not transfer well to sequences — the statistics depend on timestep and sequence length. Layer normalisation replaced it, which is what [[Attention Is All You Need|Transformers]] use.
- At inference BN folds entirely into the preceding weights, so it costs nothing at serving time. Free win.

## Links

Related: [[How Does Batch Normalization Help Optimization]] · [[Backpropagation]] · [[Deep Learning]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Regularization]] · [[Momentum]] · [[Derivative]] · [[Cross Entropy]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Vector Jacobian Product]] · [[Pytorch Autograd]] · [[Attention Is All You Need]]

New topics worth writing: Layer Normalization, Dropout, Covariate shift and domain adaptation, Whitening and ZCA, Group Normalization, Weight standardization, Xavier/He initialization, Inception / GoogLeNet architecture, Local Response Normalization
