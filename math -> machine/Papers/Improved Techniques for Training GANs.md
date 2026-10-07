---
title: "Improved Techniques for Training GANs"
authors: ["Salimans et al."]
year: 2016
arxiv: "1606.03498"
url: https://arxiv.org/abs/1606.03498
priority: Good-To-Read
read_on: 2026-09-25
tags: [paper, optimization, vision]
---
## The Core Idea

A [[Generative Adverserial Network|GAN]] is not being optimised. It is being *played*. Two networks each push down their own cost, and the thing you want — a Nash equilibrium, where neither player can improve by moving alone — is not what gradient descent looks for. The classic counter-example the paper gives: one player minimises $xy$ over $x$, the other minimises $-xy$ over $y$. Gradient descent on both at once enters a stable orbit around $(0,0)$ and never lands. GAN training is that, in millions of dimensions.

This paper does not fix that. It gives five hand-motivated tricks that make the orbit smaller in practice, plus one evaluation metric, plus one reframing of what the discriminator is for.

Two things here actually mattered historically:

**1. The Inception score.** Before this, "is my generative model good?" was answered by squinting at a grid of samples. The paper proposes a number that correlates with human judgement:

$$\text{IS} = \exp\big(\mathbb{E}_{x}\,\mathrm{KL}(p(y\mid x)\,\|\,p(y))\big)$$

where $p(y\mid x)$ is a pretrained Inception classifier's label distribution for a generated image $x$, and $p(y)$ is the average of that over all generated images. The logic: a good image is *confidently one object* (low entropy $p(y|x)$), and a good *set* of images covers many objects (high entropy $p(y)$). [[KL Divergence|KL]] measures the gap between the two, and it is large exactly when both hold. This is the direct ancestor of FID — see [[Evaluating Generative Models#^fid-def|FID]], which replaced it about two years later.

> [!NOTE] Inception score
> Sharpness times diversity, measured through a frozen ImageNet classifier. Needs ~50k samples, because half of what it measures is diversity. ^inception-score

**2. The discriminator and a classifier are the same network.** If you want to classify into $K$ classes, just add a $(K{+}1)$-th class meaning "fake". Then real-vs-fake supervision and label supervision train *one* set of weights. This turned GANs from a picture-making toy into the best semi-supervised learner of 2016 — 8.11% error on SVHN with 1000 labels, down from 16.61%.

## The Methodology

Five techniques. Each is one or two lines of code.

**Feature matching.** The generator's usual job is "maximise $D$'s output". That lets it overfit whatever the current discriminator happens to believe, which is what drives the oscillation. Replace it with: match the *average intermediate activations* of the discriminator.

$$\big\| \mathbb{E}_{x\sim p_\text{data}} \mathbf{f}(x) - \mathbb{E}_{z} \mathbf{f}(G(z)) \big\|_2^2$$

$\mathbf{f}$ is some hidden layer of $D$. $D$ is still trained normally; it is used only to *pick which statistics are worth matching*, since the features it learns are by construction the ones that separate real from fake. There is still a fixed point at $p_\text{model}=p_\text{data}$, so nothing is broken in principle. In practice it stabilises training but makes uglier pictures.

**Minibatch discrimination.** [[Mode Collapse]] happens because $D$ sees each example alone. It has no way to say "these 64 outputs are all the same image". So give it one. Take features $\mathbf{f}(x_i)\in\mathbb{R}^A$, multiply by a learned tensor $T\in\mathbb{R}^{A\times B\times C}$ to get $M_i \in \mathbb{R}^{B\times C}$, then measure how close sample $i$ is to every other sample in the batch:

$$c_b(x_i,x_j)=\exp(-\|M_{i,b}-M_{j,b}\|_{L_1}), \qquad o(x_i)_b = \sum_{j=1}^{n} c_b(x_i,x_j)$$

Concatenate $o(x_i)\in\mathbb{R}^B$ onto $\mathbf{f}(x_i)$ and carry on. $D$ still outputs one number per example, but now it has side information about crowding. Computed separately for the real batch and the fake batch.

**Historical averaging.** Add $\left\|\theta - \tfrac{1}{t}\sum_{i=1}^{t}\theta[i]\right\|^2$ to both players' costs — a pull towards the running average of your own past parameters. Cheap to maintain online. Inspired by *fictitious play*, a classical method for finding equilibria. On toy minimax games where plain gradient descent orbits forever, this converges.

**One-sided label smoothing.** If you smooth the positive target to $\alpha$ and the negative to $\beta$, the optimal discriminator is

$$D(x)=\frac{\alpha\,p_\text{data}(x) + \beta\,p_\text{model}(x)}{p_\text{data}(x)+p_\text{model}(x)}$$

That $\beta\,p_\text{model}$ in the numerator is poison: in a region where there is no real data but lots of fake data, $D$ is happy, so the generator gets no push to move away. So smooth *only* the real label ($\alpha=0.9$), leave fake at exactly 0. See [[Regularization]] for plain label smoothing.

**Virtual batch normalization.** [[Batch Normalization]] makes one sample's output depend on the other samples in its batch — visible in generators as whole batches of correlated images. VBN normalises each example using statistics from a fixed *reference batch* chosen once at the start of training, plus the example itself. Costs two forward passes, so it is used in the generator only.

**The semi-supervised loss.** With a $(K{+}1)$-way softmax, the total [[Cross Entropy|cross-entropy]] splits cleanly:

$$L = \underbrace{-\mathbb{E}_{x,y\sim p_\text{data}} \log p(y\mid x, y<K{+}1)}_{L_\text{supervised}} + \underbrace{L_\text{unsupervised}}_{\text{the ordinary GAN value}}$$

Substituting $D(x) = 1 - p(y{=}K{+}1\mid x)$ turns the second term into exactly the original GAN objective. Neat detail: the $K{+}1$ logits are over-parameterised — subtracting any $f(x)$ from all of them leaves the softmax unchanged — so you can just fix $l_{K+1}(x)=0$. Then $L_\text{supervised}$ is your ordinary $K$-class loss and the discriminator is
$$D(x)=\frac{Z(x)}{Z(x)+1}, \qquad Z(x)=\sum_{k=1}^{K}\exp[l_k(x)].$$
No extra head needed.

**Setup.** MNIST: 5 hidden layers, weight normalisation, Gaussian noise added to each discriminator layer. CIFAR-10/SVHN: 9-layer conv discriminator with dropout + weight norm; 4-layer conv generator with batch norm. ImageNet at $128\times128$, 1000 classes, multi-GPU TensorFlow fork of DCGAN.

## Ablation Studies and Experiments

**Semi-supervised classification.** Errors, averaged over 10 label subsets.

| Dataset | Labels | Prior best | This paper | Ensemble of 10 |
|---|---|---|---|---|
| MNIST (errors /10k) | 100 | 96 ± 2 (Aux. DGM) | 93 ± 6.5 | 86 ± 5.6 |
| MNIST | 200 | — | 90 ± 4.2 | 81 ± 4.3 |
| CIFAR-10 (% err) | 4000 | 19.58 (CatGAN) | 18.63 ± 2.32 | 15.59 ± 0.47 |
| SVHN (% err) | 1000 | 16.61 (Skip DGM) | **8.11 ± 1.3** | 5.88 ± 1.0 |

With only 20 MNIST labels it falls apart: 1677 ± 452 errors, variance larger than the mean.

**Inception score ablations on CIFAR-10** (50k samples; higher better). This table is the useful one.

| Config | Score |
|---|---|
| Real data | 11.24 ± .12 |
| All techniques | 8.09 ± .07 |
| − VBN, + plain BN | 7.54 |
| − labels, + historical averaging | 6.86 |
| − label smoothing | 6.83 |
| − labels (no HA to compensate) | 4.36 |
| − minibatch features | **3.87** |

Removing minibatch discrimination is worse than removing the labels entirely, and historical averaging cannot rescue it. Minibatch discrimination is doing most of the work for image quality. VBN barely matters on CIFAR (8.09 → 7.54) but the authors say it is important on ImageNet.

**Human evaluation.** MNIST samples with minibatch discrimination: annotators got 52.4% right out of 2000 votes (50% = coin flip). CIFAR-10: 78.7% correct, i.e. 21.3% human error rate — but the authors themselves scored >95%, so MTurk workers were probably not trying very hard. Showing annotators feedback on their mistakes made them much better and the scores much worse; the metric depends on how motivated your raters are.

**Validation of the metric:** filter to the top 1% of samples by Inception score and MTurk accuracy drops 78.7% → 71.4%. So the score tracks something humans see.

**What did not work.**

- **Feature matching makes bad-looking samples.** On MNIST, samples generated during feature-matching semi-supervised training are visibly fake. Minibatch discrimination fixes the pictures.
- **But minibatch discrimination "does not work at all" for semi-supervised learning.** The exact phrase in the paper. The two goals — good samples and a good classifier — want different generator objectives, and the authors say they do not understand why.
- **Vanilla DCGAN on $128\times128$ ImageNet** learns colour and texture but no objects. With the new tricks it learns fur, eyes and noses — and then assembles them into animals with wrong anatomy.
- **Do not optimise the Inception score directly.** Doing so produces adversarial examples: images that fool Inception and look like nothing.

## Worth Remembering

The whole paper is heuristic and says so. "The contributions made in this work are of a practical nature; we hope to develop a more rigorous theoretical understanding in future work." No convergence guarantee is offered for any of the five tricks.

The mode-collapse story is worth internalising because it is a *mechanism*, not a vibe: when collapse is imminent, $D$'s gradients point the same way for many nearby points, $D$ has no coordination across examples, so every output races toward the single point $D$ currently likes best. After collapse, $D$ learns to reject that point, but gradient descent cannot pull identical outputs apart — it just shoves the one point around forever. Compare the collapse analyses in [[Bootstrap Your Own Latent (BYOL)]] and [[Understanding Dimensional Collapse in Contrastive Learning]]: different cause, same symptom of an objective with no term that rewards spread.

The "labels improve image quality" finding (§5.1) is the sleeper result. Asking $D$ to name the object forces it to build an internal representation weighted towards the features humans use to recognise things — which is exactly what the Inception score measures. The authors call it transfer learning and leave it there. This line runs straight to class-conditional GANs and to [[Classifier-Free Diffusion Guidance]].

Caveats if you ever touched this:

- Inception score is **blind to your training set**. It never looks at real data. Memorise the dataset and you score perfectly. FID at least compares two distributions. Both are Goodhart-able — see [[Reward Hacking#^goodharts-law|Goodhart's law]] and [[Evaluating Generative Models#^fid-goodhart|FID Goodhart]].
- Inception score depends on the class vocabulary of ImageNet. On faces or on any domain Inception does not know, it is meaningless.
- Minibatch discrimination makes the discriminator's output depend on batch composition, so evaluation is batch-dependent too. VBN was invented to escape exactly this problem in the generator, and then minibatch discrimination reintroduces it deliberately in the discriminator. The two techniques are philosophically opposed and the paper does not comment on it.
- Variance is huge. ±2.32% on CIFAR-10 with 4000 labels, ±452 errors on MNIST with 20 labels. Anything measured on one seed here is noise.

Open question worth chasing: why does feature matching help the classifier while minibatch discrimination helps the pictures? The paper flags this as unexplained. A plausible reading is that a generator producing *slightly-off* samples gives the classifier harder negatives near the data manifold, while a generator producing perfect samples gives it nothing to learn from — the same intuition behind hard-negative mining in [[Approximate Nearest Neighbor Negative Contrastive Learning (ANCE)]].

## Links
Related: [[Generative Adverserial Network]] · [[Generative Adversarial Networks]] · [[Mode Collapse]] · [[Evaluating Generative Models]] · [[Batch Normalization]] · [[Cross Entropy]] · [[KL Divergence]] · [[Regularization]] · [[Reward Hacking]] · [[Bootstrap Your Own Latent (BYOL)]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Classifier-Free Diffusion Guidance]] · [[Generative Models]]

New topics worth writing: Inception Score, Nash equilibrium and games in ML training, semi-supervised learning, fictitious play, weight normalization, virtual batch normalization, label smoothing as adversarial robustness, DCGAN
