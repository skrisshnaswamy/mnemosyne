---
title: "Understanding Deep Learning Requires Rethinking Generalization"
authors: ["Chiyuan Zhang", "Samy Bengio", "Moritz Hardt", "Benjamin Recht", "Oriol Vinyals"]
year: 2016
arxiv: "1611.03530"
url: https://arxiv.org/abs/1611.03530
priority: Must-Read
read_on: 2026-08-25
tags: [paper, optimization, vision, theory]
---
## The Core Idea

Take CIFAR-10. Throw away the real labels. Replace every label with a random class. Now there is literally no pattern to learn — a picture of a cat might be labelled "truck", and a different cat "ship". Train a standard Inception net on it with the *same* hyperparameters you would normally use.

It reaches **100% training accuracy**.

That single fact breaks the standard story of why deep nets generalise. The story went: big models can overfit, so something must be shrinking their effective capacity — the architecture's built-in assumptions, or weight decay, or dropout — and that shrinking is why test error stays close to train error. But if a network can memorise 50,000 random labels, then its effective capacity is *at least* "memorise the whole dataset". No capacity-based bound can be doing the work.

The argument is sharp because randomising labels is **only a change to the data**. Model, size, optimiser, learning-rate schedule, regularisers — all identical. Yet generalisation error jumps from ~11% to ~90% (chance on 10 classes). So any explanation of generalisation that never looks at the labels cannot possibly explain the gap.

That kills three classical tools at once:

- **VC dimension / Rademacher complexity** measure "can this model class fit random labels?". The experiments say: yes, essentially perfectly. So empirical Rademacher complexity $\hat{R}_n(\mathcal{H}) \approx 1$, the trivial upper bound. Useless.
- **Uniform stability** is a property of the *algorithm* alone, independent of what the labels are. Same algorithm, same stability, wildly different generalisation. Useless here too.

> [!NOTE] Effective capacity
> Not "how many parameters does the model have" but "what is the largest set of arbitrary labellings this model + this optimiser can actually fit in practice". The randomisation test measures it directly. ^effective-capacity

The unlock: generalisation in deep learning is not explained by the hypothesis class being small. It has to come from some interaction of data, architecture and optimiser that we do not yet have a formal name for. This paper is mostly a demolition job — it says loudly what the answer *is not*, and the field spent the next decade on what it might be.

## The Methodology

Four kinds of experiment, all deliberately simple.

**1. Randomisation tests.** Six data conditions on CIFAR-10:

- *True labels* — baseline.
- *Partially corrupted labels* — with probability $p$, replace a label by a uniform random class. Sweeps $p$ from 0 to 1.
- *Random labels* — all labels random.
- *Shuffled pixels* — pick one random pixel permutation, apply the **same** permutation to every image (train and test). Destroys locality, keeps the dataset self-consistent.
- *Random pixels* — a **different** random permutation per image.
- *Gaussian* — throw the images away entirely; sample every pixel from a Gaussian matched to the dataset's mean and variance.

Models: a small Inception (1.65M params, 28×28 inputs, batch norm throughout), a small AlexNet (1.39M), MLP 3×512 (1.74M), MLP 1×512 (1.21M). All ReLU. All roughly 1.2–1.7M parameters against 50,000 training images — 25–35× more parameters than data points. On ImageNet: full Inception V3 with 1.28M images and 1000 classes.

Training: plain SGD with [[Momentum|momentum]] 0.9, initial LR 0.1 (Inception) or 0.01 (AlexNet/MLP), decay 0.95 per epoch. **No hyperparameter tuning when switching to random labels.** That is the point — nothing was special-cased.

**2. Regulariser toggles.** Turn [[Regularization|explicit regularisers]] on and off independently and read the test accuracy:

- Data augmentation (random crop, brightness/saturation/hue jitter, flips, ±25° rotation)
- Weight decay ($\ell_2$ on weights — see [[Decoupled Weight Decay Regularization (AdamW)]] for why this is subtler than it looks)
- Dropout (ImageNet Inception V3 only)
- Batch norm (they build an "Inception w/o BatchNorm" variant by deleting every BN layer)
- Early stopping (tracked as the running-best test accuracy during training)

**3. A finite-sample expressivity theorem.** Existing expressivity results are *population level*: which functions over the whole input domain can a network represent. The authors argue that is the wrong question. What matters is: can it fit *these $n$ points*?

**Theorem 1.** There is a two-layer ReLU network with $2n + d$ weights that can represent any function on any sample of $n$ points in $d$ dimensions.

The proof is three lines of linear algebra. Build

$$c(x) = \sum_{j=1}^{n} w_j \max\{\langle a, x\rangle - b_j,\ 0\}$$

Pick a random direction $a$ so the projections $x_i = \langle a, z_i\rangle$ are all distinct, then pick thresholds $b_j$ that *interleave* them: $b_1 < x_1 < b_2 < x_2 < \cdots < b_n < x_n$. Then the matrix $A = [\max\{x_i - b_j, 0\}]_{ij}$ is lower triangular with strictly positive diagonal, hence invertible (smallest eigenvalue $\min_i x_i - b_i$). Solve $Aw = y$ for the output weights. Done. This beats the previous $O(dn)$ construction of Livni et al. Trading width for depth gives a depth-$k$ net of width $O(n/k)$ with $O(n+d)$ weights.

The moral: **any** overparameterised network of the size people actually use can, in principle, shatter its training set. Overparameterisation is not exotic; it is the default.

**4. Linear models as a sanity check.** For [[Regression Analysis|linear]] models with $d \ge n$, the system $Xw = y$ has infinitely many exact solutions. Do they generalise equally? Curvature does not distinguish them — the Hessian

$$\nabla^2 \frac{1}{n}\sum_i \text{loss}(w^\top x_i, y_i) = \frac{1}{n} X^\top \mathrm{diag}(\beta) X$$

does not depend on $w$ at all. So "flat minima are better" is vacuous in the linear case.

But SGD does distinguish them. The update is $w_{t+1} = w_t - \eta_t e_t x_{i_t}$, so if $w_0 = 0$ every iterate stays in the span of the data: $w = X^\top \alpha$. Combine with $Xw = y$ and you get

$$XX^\top \alpha = y$$

which has a unique solution — and that solution is exactly the **minimum $\ell_2$-norm** interpolant. SGD is silently regularising. Note this equation only touches dot products $x_i^\top x_j$, so you have just re-derived the kernel trick sideways.

> [!NOTE] Implicit regularisation
> The optimiser, not the loss, picks which of the many zero-training-error solutions you land on. Nothing in the objective says "prefer small norm"; SGD-from-zero does it for free by construction. ^implicit-regularisation

## Ablation Studies and Experiments

**Fitting the noise (CIFAR-10, Table 1, all regularisers off):**

| model | params | condition | train acc | test acc |
|---|---|---|---|---|
| Inception | 1.65M | true labels, crop + wd | 100.0 | 89.05 |
| Inception | | true, crop, no wd | 100.0 | 89.31 |
| Inception | | true, no crop, wd | 100.0 | 86.03 |
| Inception | | true, nothing | 100.0 | 85.75 |
| Inception | | **random labels** | **100.0** | **9.78** |
| Inception w/o BN | 1.65M | true, wd | 100.0 | 83.00 |
| Inception w/o BN | | true, nothing | 100.0 | 82.00 |
| AlexNet | 1.39M | true, crop + wd | 99.90 | 81.22 |
| AlexNet | | true, nothing | 100.0 | 76.07 |
| AlexNet | | **random labels** | 99.82 | 9.86 |
| MLP 3×512 | 1.74M | true, nothing | 100.0 | 52.39 |
| MLP 3×512 | | **random labels** | 100.0 | 10.48 |
| MLP 1×512 | 1.21M | **random labels** | 99.34 | 10.61 |

Even a **single-hidden-layer MLP** memorises 50,000 random CIFAR-10 labels to 99.3%.

**ImageNet (Table 2), Inception V3, random labels:** 95.20% top-1 *training* accuracy on a million random labels from 1000 classes, with all regularisers off. With dropout + weight decay on, still ~91% top-1 train. Test accuracy 0.09–0.12%, i.e. chance. No hyperparameter tuning at all.

**Random pixels and Gaussian noise:** convnets still hit zero training error. Strikingly, *random pixels and Gaussian inputs converge **faster** than random labels* on real images. Intuition: random inputs are further apart in input space than two real cats, so it is easier to carve out an arbitrary label assignment for them. Real images that look alike but must get different labels are the hard case.

**Corruption sweep (Figures 1b, 1c):** training accuracy is 100% at every corruption level. Time-to-overfit grows smoothly by a factor of ~1× to ~4× as corruption goes 0→1. Test error rises smoothly from ~11% to 90% (chance). So the network is not doing one thing or the other — it learns whatever real signal survives *and* brute-force memorises the noisy remainder, simultaneously.

**ImageNet regulariser ablation (true labels):**

| aug | dropout | wd | top-1 train | top-1 test | top-5 test |
|---|---|---|---|---|---|
| yes | yes | yes | 92.18 | 77.84 | 93.92 |
| yes | no | no | 92.33 | **72.95** | 90.43 |
| no | no | yes | 90.60 | 67.18 (best 72.57) | 86.44 |
| no | no | no | 99.53 | **59.80** (best 63.16) | 80.38 |

Read this carefully. Removing *everything* costs 18 points of top-1. That is a lot — regularisation is not useless. But 59.80% top-1 is nowhere near chance (0.1%), and 80.38% top-5 is close to the 83.6% that won ILSVRC 2012. **Data augmentation alone recovers 13 of those 18 points** (59.80 → 72.95), far more than weight decay or dropout. Encoding known symmetries of the data beats generic norm penalties.

**What did not work / what the ablations kill:**

- *Weight decay as a barrier to memorisation.* Mostly it is not one. With default coefficients (Table 4), Inception still fits random labels to 100%, MLP 3×512 to 100%, MLP 1×512 to 99.21%. **AlexNet with weight decay is the one exception — it fails to converge on random labels.** The authors do not explain this and it is the single loose thread in the paper.
- *Data augmentation as a barrier.* Also not one. Bump the weight-decay factor from 0.95 to 0.999 and train longer, and Inception overfits random labels even with random cropping (99.93%) and full augmentation (99.28%). Augmentation just makes it take longer, because it inflates the effective training set.
- *Early stopping as a universal explanation.* It helps on ImageNet when other regularisers are off (67.18 → 72.57, 59.80 → 63.16 best-so-far). On **CIFAR-10 there is no benefit at all**. So early stopping is not a general mechanism.
- *Batch norm as a regulariser.* [[Deep Residual Learning for Image Recognition (ResNet)|Modern nets]] lean on it, and it does stabilise the learning curves, but removing it from Inception costs only 3–4 points (86.03 → 83.00). Not the source of generalisation.
- *Curvature / flat minima.* Explicitly ruled out for linear models — the Hessian is the same at every global optimum and degenerate at all of them.
- *Minimum-norm as a predictor.* Their own most honest negative result. On MNIST, the minimum-$\ell_2$-norm kernel solution has norm ≈ 220 and 1.2% test error; with Gabor wavelet preprocessing the norm **rises** to ≈ 390 while test error **halves** to 0.6%. Higher norm, better generalisation. So "SGD finds small norm, small norm generalises" is not the story either.

**Kernels with zero regularisation (Table 3):** solving $XX^\top\alpha = y$ exactly — interpolating the training labels perfectly, no penalty term — gives:

| data | preprocessing | test error |
|---|---|---|
| MNIST | none | 1.2% |
| MNIST | Gabor filters | 0.6% |
| CIFAR-10 | Gaussian kernel on raw pixels | 46% |
| CIFAR-10 | random conv-net (32,000 random filters) | 17% |

Adding $\ell_2$ to the CIFAR-10 random-conv model moves 17% → 15%. On MNIST, regularisation **does not help at all**. Convex models that interpolate exactly can generalise fine. The MNIST kernel matrix is 30GB; a LAPACK call on a 24-core, 256GB workstation solves it in under 3 minutes.

## Worth Remembering

- **Optimisation ease and generalisation are decoupled.** Fitting random labels is *easy* — same LR schedule, converges fast once it starts, only a constant-factor slowdown. So whatever makes deep nets easy to optimise is not what makes them generalise. Two separate mysteries, often conflated.
- **The initial delay is real but boring.** On random labels the loss plateaus at first, because labels are uncorrelated with inputs so gradients point everywhere. Since the random labels are *fixed and consistent across epochs*, the network eventually latches on. If you re-randomised labels every epoch it could never fit.
- **This directly rebuts Hardt, Recht & Singer (2016)**, whose SGD generalisation bound goes through uniform stability. The authors point out uniform stability cannot see labels, so it cannot separate the true-label and random-label runs — and empirically, many-epoch SGD on neural nets is simply *not* uniformly stable. A weaker, data-dependent stability notion is needed.
- **Finite-sample expressivity is the right lens.** Population-level universal approximation theorems need sample sizes polynomial in input dimension and exponential in depth to transfer — hopeless in practice. Asking "can it fit these $n$ points" gives a two-line proof and a far more relevant answer.
- **Practical caveat: the randomisation test is a cheap capacity probe you can run yourself.** Corrupt $p$ of your labels, train, and watch the train/test gap. If your model fits noise trivially, no amount of weight-decay tuning will save you; you need more data, better augmentation, or a better [[An Image is Worth 16x16 Words (ViT)#The Core Idea|inductive bias]]. Also note the corollary: if your training loss goes to zero on a noisy real-world label set, you have memorised the label noise.
- **The honest conclusion is a confession.** "We argue that we have yet to discover a precise formal measure under which these enormous models are simple." No mechanism is offered. That vacuum drove the whole subsequent literature on double descent, neural tangent kernels, benign overfitting, and norm-based margin bounds.
- **Open questions this leaves.** Why does AlexNet + weight decay uniquely refuse to fit random labels? Why do random-pixel inputs converge faster than random labels? If minimum norm is not predictive, what functional of the SGD trajectory is? Does the same picture hold for [[Attention Is All You Need|transformers]] and [[Scaling Laws for Neural Language Models|models at scale]] where you never complete a second epoch, so memorisation has no chance to kick in?
- **Connection worth chasing:** [[The Bitter Lesson (essay)]] and this paper agree from opposite directions — hand-designed structure (here, explicit regularisers) matters much less than people assume; scale, data and generic learning do the heavy lifting. And note that data augmentation, the *one* intervention that gave a large gain, is precisely the one that injects real knowledge about the world rather than a generic penalty.

## Links

Related: [[Regularization]] · [[Deep Learning]] · [[Backpropagation]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Scaling Laws for Neural Language Models]] · [[Why do tree-based models still outperform deep learning on tabular data]] · [[The Bitter Lesson (essay)]] · [[Derivative#Hessian|Hessian]] · [[Momentum]] · [[Uncertainty]] · [[Practical Bayesian Optimization of Machine Learning Algorithms]]

New topics worth writing: VC dimension, Rademacher complexity, uniform stability, double descent, benign overfitting, neural tangent kernel, batch normalization, early stopping, the kernel trick and representer theorem, memorisation vs generalisation, flat minima hypothesis, universal approximation theorem
