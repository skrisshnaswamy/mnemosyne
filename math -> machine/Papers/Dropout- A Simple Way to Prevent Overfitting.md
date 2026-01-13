---
title: "Dropout: A Simple Way to Prevent Overfitting"
authors: ["Srivastava et al."]
year: 2014
url: https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, diffusion, vision]
---
## The Core Idea

A big neural net with lots of parameters memorises its training set. The classic fix is to train many different nets and average their predictions — an ensemble. But training 100 big nets is expensive, and running all 100 at test time is worse.

Dropout gets an ensemble for free. During training, on **every single training example**, you randomly delete each unit (and all its wires) with some probability. A net with $n$ units contains $2^n$ possible "thinned" sub-networks. Each training example trains one randomly chosen sub-network. All the sub-networks share the same weight matrix, so you are never storing more than one model.

At test time you do not sample anything. You use the full net once, with every outgoing weight multiplied by $p$ (the keep probability). That single scaled net approximates the geometric-mean prediction of all $2^n$ thinned nets. The scaling makes the expected activation match: if a unit fires with value $y$ but is only present $p$ of the time, its expected contribution during training is $py$, so at test time you send $py$ deterministically.

> [!NOTE] Co-adaptation ^co-adaptation
> When a hidden unit only produces something useful *because* three specific other units are there to clean up its mistakes. The group works on training data and falls apart on new data. Dropout kills this: a unit cannot rely on any particular partner being present, so it must learn a feature that is useful on its own.

The authors' analogy is sexual reproduction. Asexual reproduction copies a whole co-adapted gene set intact. Sex smashes gene sets apart every generation, so genes that survive are the ones that work with *random* other genes. Robustness beats fine-tuned teamwork when conditions change — and test data is a change in conditions.

Why it did not exist earlier: adding noise to *inputs* was known (denoising autoencoders, ~5% noise). The new bits are (a) noise in the **hidden** layers, (b) the interpretation as model averaging, and (c) the weight-scaling trick that makes 50% noise usable instead of 5%. This is what made it practical to train nets with 65M parameters on 60K examples without them collapsing into memorisation.

## The Methodology

Standard forward pass for layer $l$:

$$z^{(l+1)}_i = w^{(l+1)}_i y^{(l)} + b^{(l+1)}_i, \qquad y^{(l+1)}_i = f(z^{(l+1)}_i)$$

With dropout, you insert one extra line — a mask of independent [[Random variable|Bernoulli variables]]:

$$r^{(l)}_j \sim \text{Bernoulli}(p), \qquad \tilde{y}^{(l)} = r^{(l)} * y^{(l)}$$

$$z^{(l+1)}_i = w^{(l+1)}_i \tilde{y}^{(l)} + b^{(l+1)}_i, \qquad y^{(l+1)}_i = f(z^{(l+1)}_i)$$

where $*$ is element-wise product. At test time $W^{(l)}_{\text{test}} = pW^{(l)}$, no mask.

**Training loop.** Ordinary SGD. For each training case in a mini-batch, sample a fresh mask, forward and [[Backpropagation|backprop]] through *that* thinned net only. A parameter not used by a given case contributes gradient zero for that case. Gradients averaged over the mini-batch as usual. Nothing else changes.

**The three hyperparameters that actually matter**, and they are coupled:

1. **Keep probability $p$.** $p=0.5$ for hidden units, $p=0.8$ for inputs. For images/speech frames the input is real-valued and you can afford to drop less.
2. **Max-norm constraint.** After each update, project the incoming weight vector of each hidden unit back onto a ball: enforce $\|w\|_2 \le c$, with $c \in [3,4]$. This is separate from dropout but the paper insists the two together are much better than dropout alone. The reason: max-norm lets you use a giant learning rate without weights exploding, and the dropout noise then explores weight space that a small learning rate would never reach.
3. **Learning rate and [[Momentum|momentum]].** Dropout gradients are noisy and partly cancel, so a dropout net wants **10–100×** the learning rate of the equivalent plain net, and momentum of 0.95–0.99 instead of the usual 0.9.

**Network size heuristic.** If $n$ hidden units is right without dropout, use at least $n/p$ with dropout — you only have $pn$ units present in expectation.

**With pretraining.** If you pretrain with RBMs/autoencoders and then fine-tune with dropout, scale the pretrained weights *up* by $1/p$ first. And use a small fine-tuning learning rate — at the normal large dropout learning rate, the stochasticity wiped out the pretrained information entirely.

**Gaussian dropout (Section 10).** Instead of a Bernoulli mask, multiply each activation by $r' \sim \mathcal{N}(1, \sigma^2)$. Set $\sigma^2 = (1-p)/p$ and the multiplier has the same mean (1) and variance as the $1/p$-scaled Bernoulli version. Because the mean is already 1, **no test-time rescaling is needed at all**. This is where the modern "inverted dropout" convention comes from: scale by $1/p$ at training time, leave test time alone.

**Marginalising the noise (Section 9).** For plain linear regression with dropout on the inputs, you can integrate the noise out in closed form. Minimising $\mathbb{E}_R\left[\|y - (R * X)w\|^2\right]$ gives

$$\|y - pXw\|^2 + p(1-p)\|\Gamma w\|^2, \qquad \Gamma = (\text{diag}(X^\top X))^{1/2}$$

Absorbing $p$ into $\tilde{w} = pw$:

$$\|y - X\tilde{w}\|^2 + \frac{1-p}{p}\|\Gamma\tilde{w}\|^2$$

So input dropout on [[Regression Analysis|linear regression]] *is* ridge regression, with the penalty on each weight scaled by the standard deviation of that input dimension — features that vary a lot get squeezed harder. And the regularisation strength is exactly $\frac{1-p}{p}$, which goes to zero as $p \to 1$. For deep nets no such closed form exists.

## Ablation Studies and Experiments

**MNIST, permutation-invariant setting** (Table 2, test error %):

| Setup | Error |
|---|---|
| Standard net, logistic, 2×800 (Simard 2003) | 1.60 |
| Dropout, logistic, 3×1024 | 1.35 |
| Dropout, [[ImageNet Classification with Deep CNNs (AlexNet)#^relu\|ReLU]], 3×1024 | 1.25 |
| + max-norm, 3×1024 | 1.06 |
| + max-norm, 2×8192 | **0.95** |
| DBM pretrain + dropout finetune, 500-500-2000 | **0.79** |

That 2×8192 net has >65M parameters trained on 60K examples and does not need early stopping. This is a direct data point for the [[Understanding Deep Learning Requires Rethinking Generalization|capacity-vs-generalisation]] puzzle.

**Against other regularisers** (Table 9, same 784-1024-1024-2048-10 ReLU net, all hyperparameters tuned on validation):

| Method | Error |
|---|---|
| L2 | 1.62 |
| L2 + L1 late in training | 1.60 |
| L2 + KL-sparsity | 1.55 |
| Max-norm alone | 1.35 |
| Dropout + L2 | 1.25 |
| **Dropout + max-norm** | **1.05** |

Note max-norm alone already beats L2. The combination is where the win lives.

**SVHN** — the ablation that surprised people. ConvNet + max-pooling: 3.95%. Add dropout to fully-connected layers only: 3.02%. Add dropout to the **convolutional** layers too: **2.55%**. Everyone assumed conv layers have too few parameters to overfit, so dropout there would be pointless. It helps anyway — because it injects noisy inputs into the fully-connected layers above, which do overfit. (Human performance: 2.0%.)

**CIFAR-10 / CIFAR-100**, no data augmentation: 15.60 / 43.48 hand-tuned → 14.32 / 41.26 with FC dropout → **12.61 / 37.20** with dropout everywhere. The CIFAR-100 gain of 6.3 points is the largest in the paper.

**ImageNet.** ILSVRC-2010 top-5 goes from 25.7% (SIFT + Fisher vectors) to 17.0%. ILSVRC-2012 winner: 16.4% top-5 test with 5 averaged ConvNets, versus ~26% for the best hand-engineered vision pipeline.

**TIMIT speech.** 6-layer net 23.4% → 21.8% phone error. DBN-pretrained 4-layer 22.7% → **19.7%**. An 8-layer pretrained net also lands at 19.7% — dropout let the 4-layer net match the 8-layer net.

**Reuters-RCV1 text.** 31.05% → 29.62%. Tiny. The authors attribute this to 200K training documents making overfitting a non-problem.

**vs Bayesian neural nets** (alternative splicing, Code Quality, higher better): plain NN with early stopping 440, dropout NN **567**, full Bayesian NN **623**. Dropout loses to proper Bayesian model averaging, as it should — dropout weights every sub-model equally, Bayes weights by posterior. But dropout got there with a 1000-unit net, no PCA, and orders of magnitude less compute.

### Things that did not work, and what the ablations reveal

- **Dropout does not help on tiny datasets.** At 100 and 500 MNIST examples, the dropout and no-dropout curves are identical (Figure 10). The model can memorise even through the noise. The gain rises with dataset size, peaks, then declines again as overfitting stops being the bottleneck. There is a "sweet spot" of data size, not a monotone benefit.
- **Very small $p$ underfits.** With the architecture held fixed at 784-2048-2048-2048-10, small $p$ raises *training* error too, not just test error — that is underfitting, not regularisation. Test error is flat for $0.4 \le p \le 0.8$ and rises sharply near $p=1$. If instead you hold $pn$ constant (grow the layer as you drop more), the damage at small $p$ largely disappears: error at $p=0.1$ drops from 2.7% to 1.7%. So $p$ and layer width must be tuned together.
- **Deeper is not always better under dropout.** 3-layer nets with 4096/8192 units did *worse* than 2-layer nets at the same dropout level; raising dropout helped but never enough to overtake.
- **Naive fine-tuning of pretrained weights destroyed them** at normal dropout learning rates.
- **Max-norm gave no improvement on TIMIT**, unlike everywhere else.
- **Preprocessing on SVHN**: global/local contrast normalisation and ZCA whitening gave no noticeable improvement over just zero-mean unit-variance RGB.
- **Cost:** dropout nets take **2–3× longer** to train, because the gradient you compute is never the gradient of the architecture you will actually deploy.

**Is weight scaling a good approximation?** They compared the $p$-scaled single net against true Monte Carlo averaging over $k$ sampled thinned nets (Figure 11). At $k \approx 50$ the MC average catches up; beyond that MC is very slightly better but within one standard deviation. So the cheap deterministic approximation is worth roughly 50 forward passes. **Bernoulli vs Gaussian** (Table 10): MNIST 1.08 vs 0.95, CIFAR-10 12.6 vs 12.5. Gaussian is equal or slightly better — the authors note it has maximum entropy given the same mean and variance, while Bernoulli has minimum, and both extremes work.

**What is actually doing the work?** Two diagnostics. (1) First-layer features of a 256-unit ReLU autoencoder: without dropout the filters are unstructured noise that only make sense jointly; with $p=0.5$ they become clean edge, stroke and spot detectors — visible evidence that co-adaptation broke. Both autoencoders had similar reconstruction error, so the loss did not reveal this. (2) Dropout makes activations **sparse for free**, with no sparsity penalty anywhere: mean hidden activation falls from ~2.0 to ~0.7, and the activation histogram develops a sharp spike at zero.

## Worth Remembering

- **Robustness across architectures.** Figure 4 trains six architectures (2–4 layers, 1024–2048 units) with *identical* hyperparameters including $p$. The with-dropout and without-dropout trajectories form two cleanly separated clusters. Dropout wins everywhere without per-architecture tuning — unusual, and part of why it spread so fast.
- **The size of the win tracks how much overfitting there was.** CIFAR-100 (60K images, 100 classes): 6.3 points. Reuters (400K docs, 50 classes): 1.4 points. If your model is not overfitting, dropout buys you almost nothing and costs you 2–3× training time.
- **Modern practice has drifted.** Almost nobody uses max-norm any more, and the paper is emphatic that dropout + max-norm + huge LR + high momentum is the package. If you bolt dropout onto an [[Adam- A Method for Stochastic Optimization|Adam]]-trained net with a normal learning rate, you are not running the recipe that produced these numbers. Likewise, [[Batch Normalization]] arrived a year later and largely displaced dropout in conv nets — and the two interact badly, since BN's train/test statistics mismatch compounds dropout's.
- **The "inverted dropout" in every framework today is the Section 10 reparameterisation**, not the Section 4 formulation. Scale by $1/p$ during training; test time is untouched.
- **Dropout is not free ensembling in the strict sense.** The weight-scaling trick approximates a *geometric* mean of sub-model predictions, and only exactly so for a single linear-softmax layer. For deep nets it is a heuristic that happens to work — the Monte Carlo comparison is the empirical justification, not a proof.
- **Sampling at test time gives you uncertainty.** The paper does the $k$-sample Monte Carlo average purely to validate the approximation, but the same procedure is what later became MC-dropout for [[Uncertainty|uncertainty estimation]] — the spread across samples, not just the mean.
- Open question the authors flag: for anything more complex than linear regression, nobody knows the deterministic regulariser that dropout is implicitly applying. That is why the gradient is noisy and training is slow.

## Links

Related: [[Regularization]] · [[Backpropagation]] · [[Deep Learning]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Batch Normalization]] · [[Momentum]] · [[Understanding Deep Learning Requires Rethinking Generalization]] · [[Reconciling Modern ML Practice and the Bias-Variance Trade-off]] · [[Delving Deep into Rectifiers (He init, PReLU)]] · [[Gradient-Based Learning Applied to Document Recognition (LeNet)]] · [[Practical Bayesian Optimization of Machine Learning Algorithms]] · [[Random variable]] · [[Regression Analysis]] · [[Uncertainty]] · [[KL Divergence]] · [[Loss, Objectives, and Business Alignment]] · [[Distilling the Knowledge in a Neural Network]]

New topics worth writing: Max-norm regularization, Ensemble methods and model averaging, Denoising autoencoders, Restricted Boltzmann Machines and Contrastive Divergence, MC-dropout for predictive uncertainty, Maxout units, Ridge regression, DropConnect and structured dropout variants, Stochastic depth
