---
title: "Reconciling Modern ML Practice and the Bias-Variance Trade-off"
authors: ["Mikhail Belkin", "Daniel Hsu", "Siyuan Ma", "Soumik Mandal"]
year: 2018
arxiv: "1812.11118"
url: https://arxiv.org/abs/1812.11118
priority: Must-Read
read_on: 2026-08-25
tags: [paper]
---
## The Core Idea

The textbook story says: make a model bigger and at some point it starts memorising the training data instead of learning the pattern, so test error goes up. Draw it and you get a U — error falls, hits a sweet spot, then rises. That U comes from the bias-variance trade-off, and it has been the justification for "don't use a model with more parameters than you have data points" for thirty years.

Modern practice ignores this completely. People train networks with far more parameters than data points, drive training error to exactly zero, and get excellent test error anyway. [[Understanding Deep Learning Requires Rethinking Generalization|Zhang et al.]] had already shown these networks can fit pure random labels, so it is not that they *cannot* memorise — they can, and yet on real labels they generalise.

This paper draws the missing piece of the picture. Keep increasing capacity past the point where the model can exactly fit the training set, and test error **falls again**, often below the old U-shaped sweet spot. Plotting error against capacity gives two descents with a spike in between.

> [!NOTE] Double descent
> Test risk as a function of model capacity: the classical U up to the interpolation threshold, a sharp peak at the threshold, then a second descent that keeps going down as capacity grows. The U is not wrong, it is just the left half of the curve. ^double-descent

> [!NOTE] Interpolation threshold
> The capacity at which the model class is exactly big enough to fit every training point with zero error. For $N$ parameters and $n$ examples with $K$ outputs, this is roughly $N = n \cdot K$. This is where test error is *worst*. ^interpolation-threshold

The proposed mechanism is the useful part. Past the threshold there are infinitely many functions that fit the training data perfectly, so "zero training error" no longer picks out one model. What picks the model is the **inductive bias** of the fitting procedure — here, "among all perfect fits, take the one with the smallest norm" (smallest = smoothest). A bigger function class contains *more* perfect fits, so the smallest-norm one available gets smaller and smoother. Capacity buys you a better Occam's-razor solution, not more overfitting.

At the threshold itself you have exactly one function that fits, and you have no choice: it is forced to contort wildly through every point, including the noise. That is the spike.

## The Methodology

No new algorithm. The contribution is a set of careful sweeps over model size, with the norm of the learned solution plotted alongside test error.

**Setting.** Squared loss everywhere, even for classification. Multi-class labels become one-hot vectors and each output gets its own squared loss. No regularisation (with one small exception below), no early stopping — both would hide the peak.

**Random Fourier Features (RFF)** — the cleanest case. A two-layer network where the first layer is frozen at random:

$$h(x) = \sum_{k=1}^{N} a_k\, e^{\sqrt{-1}\langle v_k, x\rangle}, \qquad v_k \sim \mathcal{N}(0, \sigma^{-2}I)$$

Only the $a_k$ are learned, by least squares. Bandwidth $\sigma$ = 5 for MNIST/SVHN/CIFAR-10, 0.1 for 20-Newsgroups, 16 for TIMIT. When $N > n$ the least-squares solution is not unique, and they take the **minimum $\ell_2$ norm** one — a computable stand-in for the RKHS norm $\|h\|_{\mathcal{H}_\infty}$. As $N \to \infty$ this class approaches the Gaussian-kernel RKHS, and $h_{n,\infty}$ (the plain kernel machine) is the true minimum-norm interpolator.

> [!NOTE] Minimum-norm interpolation
> Of all functions that pass exactly through the training points, pick the one with smallest function-space norm — the smoothest. This is the inductive bias the whole paper leans on. For kernels you can get it three different ways: solve the linear system, run SGD from zero initialisation, or take a Gaussian process posterior mean. All three give the same function. ^minimum-norm-interpolation

**Random ReLU features.** Same setup with $\phi(x;v) = \max(\langle v,x\rangle, 0)$, $v$ uniform on the unit sphere.

**Fully connected nets.** One hidden layer of $H$ units, so $(d{+}1)H + (H{+}1)K$ parameters. SGD with momentum 0.95. Two tricks mattered:
- Below the threshold, **weight reuse**: initialise the $H_2$-unit net's first $H_1$ hidden units from the already-trained $H_1$-unit net, fill the rest with $\mathcal{N}(0, 0.01)$ noise. Without this, non-convexity means SGD sometimes fails to drive training error down as size grows, and the curve turns into noise.
- Above the threshold, plain random init is fine — getting to zero training loss is easy there.
- Step size decayed 10% every 500 epochs below threshold, fixed above. Cap of 6000 epochs.

**Trees.** Capacity is indexed in a hybrid way: first grow $N_{\mathsf{leaf}}^{\mathsf{max}}$ (leaves per tree) up to the threshold, then, past it, grow $N_{\mathsf{tree}}$ (number of interpolating trees averaged). Bootstrap sampling is turned off in one variant so a *single* tree can interpolate. Also tested with $L_2$-boosting: shrinkage 0.85 (deliberately low so it interpolates fast), trees capped at 10 leaves, then average several such forests to go past the threshold.

**Theory (Appendix A).** In the noiseless case, with $x_i$ uniform on a compact cube and $y_i = h^*(x_i)$, any interpolating $h$ satisfies with high probability
$$\sup_{x\in\Omega}|h(x)-h^*(x)| < A e^{-B(n/\log n)^{1/d}}\left(\|h^*\|_{\mathcal{H}_\infty}+\|h\|_{\mathcal{H}_\infty}\right).$$
The bound is proportional to the norm of your interpolant, so the minimum-norm one gets the tightest bound. Note the $(\cdot)^{1/d}$ in the exponent — this decays terribly in high dimension, and it says nothing about noisy labels.

## Ablation Studies and Experiments

Datasets: MNIST, CIFAR-10, SVHN, TIMIT (440-d speech, 48 classes), 20-Newsgroups (bag of words summed into GloVe embeddings, 100-d). Subsampled to $n = 10^4$ for RFF, $n = 4\cdot10^3$ for the neural nets. Five repeats averaged for nets and ensembles; RFF was so consistent they report one run.

**RFF on MNIST** ($n = 10^4$, 10 classes, threshold at $N = 10^4$). Classic U up to $N=n$. The model with *exactly* enough features to interpolate is the worst on the whole plot — the paper says flatly it has "no predictive ability for classification". Push $N$ past $10^4$ and accuracy improves sharply, overtaking the bottom of the U. The kernel machine $h_{n,\infty}$ beats every finite-$N$ model. The $\ell_2$ norm curve is the tell: it peaks at exactly $N = n$ and decreases monotonically afterwards, tracking test risk. Same shape on CIFAR-10, 20-Newsgroups, TIMIT, SVHN.

**The one-dimensional picture (Fig. 3).** Ten data points, Random ReLU features. With $N = 40$ features the fit is a jagged piecewise-linear mess, coefficient norm $\approx 695$. With $N = 4000$ features the fit through the *same ten points* looks visibly smooth, norm $\approx 159$. Both interpolate. More parameters bought a simpler function. This single figure carries the whole argument.

**Neural nets.** Same qualitative curve, with the peak landing at $n \cdot K$ — confirming that multi-class problems need parameters multiplied by the number of classes to interpolate.

**Trees.** Double descent appears for random forests on MNIST and on regression tasks, with and without bootstrap, and for $L_2$-boosting. Under squared loss the pre-threshold overfitting is strong and obvious; under zero-one loss the U-shaped part is much less apparent.

**What did not work / what hides the effect:**
- **Random initialisation without weight reuse** (Fig. 9c): variance blows up, training risk stops decreasing monotonically with size, and the curve gets ugly — though double descent is still discernible. Because ERM on a net is non-convex, "bigger class" does not guarantee "lower training risk" when your optimiser is a local search.
- **High shrinkage in boosting** (0.1 instead of 0.85): the double descent curve "becomes less apparent" because shrinkage regularises. They had to deliberately weaken regularisation to see the effect.
- **Ridge regularisation** of $4\cdot10^{-6}$ was needed for Random ReLU on SVHN purely for numerical stability near the threshold — near $N=n$ the design matrix is horribly conditioned.
- Early stopping, any [[Regularization|regularisation]], and smoothing all flatten or erase the peak. The authors argue this is *why* the phenomenon was missed for decades: classical statistics used fixed small feature sets, non-parametric statistics always regularised, and RFF was only ever used in the $N \ll n$ regime because that is where it is computationally cheaper than a kernel.

## Worth Remembering

- The peak lives in a **narrow band** of capacities. If you sweep model sizes coarsely you will just see "bigger is better" and never notice there is a cliff you jumped over. This is a real practical hazard when you tune width.
- The definition of capacity here is **number of parameters**, which the authors admit is crude. It works for these controlled sweeps; it is not a general capacity measure.
- A sobering estimate: ImageNet has $\sim10^6$ examples and $\sim10^3$ classes, so interpolation would need $\sim10^9$ parameters — larger than most ImageNet models of the day. The authors conclude that for large datasets **the classical U-shaped regime is still the right mental model**. That framing aged interestingly given [[Scaling Laws for Neural Language Models|scaling laws]] and [[Training Compute-Optimal Large Language Models (Chinchilla)|Chinchilla]], which sit in a different corner of the same space.
- Theorem 1 only covers noiseless labels and degrades as $e^{-B(n/\log n)^{1/d}}$ — essentially vacuous in high $d$. The empirical claim that minimum-norm interpolation survives heavy label noise rests on cited experiments, not on this bound.
- Over-parameterisation has a second, separate benefit the paper flags: past the threshold the optimisation landscape is easier and SGD reliably reaches global minima of the training risk. Models left of the peak and right of the peak may need different tuning instincts entirely.
- The tree result is the most surprising one. It says double descent is not about gradient descent or neural nets at all — averaging fully-grown interpolating trees is a *different* route to the same smoothing inductive bias. Compare against [[XGBoost- A Scalable Tree Boosting System|XGBoost]], where shrinkage and depth limits are standard, i.e. where practitioners deliberately stay left of the threshold.
- Open question worth chasing: which quantity actually controls the second descent when you cannot compute a norm? For deep nets, "SGD finds a small-norm solution" is conjecture backed by partial evidence, not a theorem.

## Links
Related: [[Understanding Deep Learning Requires Rethinking Generalization]] · [[Regularization]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Why do tree-based models still outperform deep learning on tabular data]] · [[XGBoost- A Scalable Tree Boosting System]] · [[Scaling Laws for Neural Language Models]] · [[Fourier Series Decomposition]] · [[Deep Learning]] · [[Momentum]] · [[Regression Analysis]] · [[Uncertainty]]

New topics worth writing: Bias-variance decomposition, Random Fourier Features, Reproducing Kernel Hilbert Space, Implicit regularisation of SGD, Margin theory, Random forests, Gradient boosting shrinkage, Neural tangent kernel, Benign overfitting
```
