---
title: "How Does Batch Normalization Help Optimization?"
authors: ["Shibani Santurkar", "Dimitris Tsipras", "Andrew Ilyas", "Aleksander Madry"]
year: 2018
arxiv: "1805.11604"
url: https://arxiv.org/abs/1805.11604
priority: Must-Read
read_on: 2026-08-25
tags: [paper, optimization]
---
## The Core Idea

Batch Normalization (BatchNorm) works. Everyone uses it. The reason everyone *gave* for why it works — "it reduces internal covariate shift" — is wrong.

> [!NOTE] Internal covariate shift (ICS)
> The original story: when you update layer 3, the inputs that layer 5 sees change their distribution. Layer 5 keeps chasing a moving target, so training is slow. BatchNorm was sold as a fix: pin the mean and variance of each layer's inputs, and the target stops moving. ^internal-covariate-shift

This paper kills that story with three blows:

1. **Deliberately re-introducing covariate shift does not hurt.** Inject time-varying, non-zero-mean, non-unit-variance noise *after* every BatchNorm layer. The distributions become wildly unstable — worse than a network with no BatchNorm at all. Training performance barely changes. It still beats the unnormalised network.
2. **BatchNorm may not even reduce ICS**, once you define ICS in a way that actually matters for optimisation (gradient change, not distribution change). Measured that way, BatchNorm networks often have *more* ICS.
3. **The real mechanism is smoothing.** BatchNorm reparameterises the problem so the loss surface has smaller [[Derivative#Gradient|gradients]] (better Lipschitz constant) *and* gradients that change more slowly as you move (better $\beta$-smoothness). So the gradient you compute at a point stays a good guide for a long step. That is why you can use big learning rates and why training is robust to initialisation.

> [!NOTE] $\beta$-smoothness
> A function is $\beta$-smooth if its gradient is Lipschitz: $\|\nabla f(x) - \nabla f(y)\| \le \beta\|x-y\|$. Small $\beta$ means "the gradient does not swing around wildly nearby", which means a gradient step of size $\sim 1/\beta$ is safe. ^beta-smoothness

The unlock: normalisation is not about distributions at all, it is about the geometry of the optimisation landscape. And since it is about geometry, **BatchNorm is not special**. Normalising by the average $\ell_1$ or $\ell_\infty$ norm works just as well — sometimes better — even though those schemes give you no control over mean and variance and produce non-Gaussian activation histograms.

## The Methodology

Two models, both of which clearly benefit from BatchNorm:

- **VGG on CIFAR-10.** Plain SGD, 15,000 steps, batch size 128, fixed learning rate 0.1, Glorot init, no data augmentation. Test accuracy 83% with BatchNorm, 80% without (92% / 88% with augmentation).
- **25-layer deep linear network (DLN)** on synthetic Gaussian data. Minimise $\|W_1 \cdots W_{25} x - Ax\|_2^2$, with $A$ a random $10\times10$ Gaussian matrix. Full-batch gradient descent, 10,000 steps, learning rate $10^{-6}$. Linear means no non-linearity effects; full-batch means no gradient noise. Both confounds removed, and BatchNorm still helps a lot.

**The "noisy BatchNorm" experiment.** After each BatchNorm layer, perturb every activation $a_{i,j}$ (sample $i$, unit $j$) by
$$a^t_{i,j} \leftarrow s^t_{i,j} \cdot a_{i,j} + m^t_{i,j}$$
where per layer and timestep they draw $\mu^t \sim U(-0.5, 0.5)$ and $\sigma^t \sim U(1, 1.25)$, then per element $m^t_{i,j} \sim U(\mu^t - 0.1, \mu^t + 0.1)$ and $s^t_{i,j} \sim \mathcal{N}(\sigma^t, 0.1)$. The noise distribution itself changes every step, so every unit sees a different input distribution at every step — maximal covariate shift by construction.

**A better definition of ICS.** Since training is a first-order method, measure the thing first-order methods care about: how much does layer $i$'s gradient change when the layers *before* it get updated?

$$G_{t,i} = \nabla_{W_i^{(t)}}\mathcal{L}(W_1^{(t)},\dots,W_k^{(t)}; x^{(t)}, y^{(t)})$$
$$G'_{t,i} = \nabla_{W_i^{(t)}}\mathcal{L}(W_1^{(t+1)},\dots,W_{i-1}^{(t+1)}, W_i^{(t)},\dots,W_k^{(t)}; x^{(t)}, y^{(t)})$$

ICS is $\|G_{t,i} - G'_{t,i}\|_2$ (and they also report the cosine angle). Same batch, same later layers — only the preceding layers moved. This isolates the cross-layer dependency exactly.

**Landscape probes.** At each training step, take the current gradient direction and walk along it, measuring:
- (a) the range of loss values you hit — measures loss Lipschitzness,
- (b) the $\ell_2$ distance between the gradient at the start and the gradients along the way — "gradient predictiveness",
- (c) the max ratio of gradient change to distance moved — "effective $\beta$-smoothness".

Step lengths span $[1/2, 4]\times$ the learning rate for VGG, $[1/100, 30]\times$ for the DLN. This is measurement, not training.

**The theory.** Insert one BatchNorm after an arbitrary fully-connected layer $W$. Let $y = Wx$, $\hat y$ = whitened $y$, $z = \gamma \hat y + \beta$, batch size $m$, per-unit std $\sigma_j$. Treat $\gamma, \beta$ as constants. The key result on the loss gradient:

$$\|\nabla_{y_j}\widehat{\mathcal{L}}\|^2 \le \frac{\gamma^2}{\sigma_j^2}\left(\|\nabla_{y_j}\mathcal{L}\|^2 - \frac{1}{m}\langle \mathbf{1}, \nabla_{y_j}\mathcal{L}\rangle^2 - \frac{1}{m}\langle \nabla_{y_j}\mathcal{L}, \hat y_j\rangle^2\right)$$

Read it as: BatchNorm's gradient is the original gradient, *minus* its mean component, *minus* its component along the normalised activations, all scaled by $\gamma/\sigma_j$. Both subtracted terms are typically large — the mean term grows quadratically in dimension, and a gradient is rarely uncorrelated with the variable it is taken with respect to. This is an **additive** reduction; it survives even if you set $\gamma = \sigma_j$ so the scaling cancels.

For smoothness, the quadratic form of the [[Derivative#Hessian|Hessian]] along the gradient direction gets the same treatment:

$$(\nabla_{y_j}\widehat{\mathcal{L}})^\top \bm{\widehat H} (\nabla_{y_j}\widehat{\mathcal{L}}) \le \frac{\gamma^2}{\sigma^2}(\cdot)^\top \bm{H}_{jj}(\cdot) - \frac{\gamma}{m\sigma^2}\langle \hat g_j, \hat y_j\rangle \|\nabla_{y_j}\widehat{\mathcal{L}}\|^2$$

The two conditions for this to be a genuine improvement are mild: $\bm{H}_{jj} \succeq 0$ (true locally when you have piecewise-linear activations like [[ImageNet Classification with Deep CNNs (AlexNet)#^relu|ReLU]] and a convex final loss like [[Cross Entropy|softmax cross-entropy]]), and $\langle \hat y_j, \hat g_j \rangle > 0$ (true when the negative gradient points toward the minimum).

Two smaller results. **Observation 4.3:** for any $W$ there is a BatchNorm config $(W, \gamma, \beta)$ with $\gamma = \sigma_j$ giving identical activations — so all minima of the original landscape are preserved, and this is a genuine reparameterisation, not a rescale. **Lemma 4.5:** because BatchNorm is scale-invariant ($BN(aWx) = BN(Wx)$), the *whole ray* $kW^*$ is optimal, so the nearest optimum from any init $W_0$ is closer by exactly $\|W^*\|^2(1-k)^2$ where $k = \langle W^*, W_0\rangle / \|W^*\|^2$.

## Ablation Studies and Experiments

**Distributional stability is not the mechanism.** Figure 1(c): plotting a random activation's distribution over training, standard and BatchNorm VGG look barely different, despite a large gap in training loss and accuracy. Figure 2: noisy-BatchNorm nearly matches plain BatchNorm and comfortably beats the standard network, while having visibly *less* stable distributions than the standard network. The control that matters — **adding the same noise to a standard network stops it training entirely.** So the noise is genuinely large; BatchNorm is absorbing it.

**BatchNorm does not reduce gradient-level ICS.** In the DLN this is stark: the standard network has almost no ICS for the entire run (cosine angle near 1), while in the BatchNorm network $G$ and $G'$ are nearly uncorrelated. The BatchNorm network still wins on loss by a wide margin. In VGG, ICS is similar or worse with BatchNorm. (The apparent stabilisation late in the BatchNorm VGG run is just convergence, not shift reduction.)

**Smoothness is the mechanism.** Gradient predictiveness differs by close to **two orders of magnitude** early in training. Loss variation along the gradient direction is huge for the vanilla net and narrow for BatchNorm. Effective $\beta$-smoothness is consistently better. Note the honest caveat: they cap VGG step lengths at $0.4\times$ the gradient because beyond that the standard network simply diverges — BatchNorm keeps smoothing well past that point.

**BatchNorm is not special — the $\ell_p$ ablation.** They fix the first moment and then divide by the average $\ell_p$ norm (computed *before* mean-shifting) for $p = 1, 2, \infty$. Results:
- All three match or beat BatchNorm on training performance.
- On the DLN, **$\ell_1$-normalisation beats BatchNorm.**
- All three produce the same landscape smoothing.
- Their activation histograms are visibly non-Gaussian, and they induce *larger* distributional shift than the unnormalised network.

That last bullet is the ablation that does the real work: it decouples "control the moments" from "smooth the landscape", and only the second one predicts performance. The authors' own conclusion is blunt — BatchNorm's success "might be somewhat serendipitous", and the design space of normalisers deserves a principled search.

**What didn't work / wasn't shown:** the paper does not demonstrate that smoothing *causes* the generalisation gain, only the optimisation gain. They speculate that smoothing pushes training toward flat minima (Hochreiter & Schmidhuber; Keskar et al.) and leave it open.

## Worth Remembering

- The whole theory is for **one BatchNorm layer after one linear layer**, with $\gamma$ and $\beta$ held constant, and the bounds are on the loss with respect to *activations*, translated to weights only as worst-case minimax bounds over $\|X\| \le \lambda$. It is a local, single-layer story, not an end-to-end guarantee.
- The bounds are *upper* bounds. They show BatchNorm's gradient norm cannot exceed the reduced quantity; they do not prove the vanilla gradient is always worse in practice.
- The $\gamma/\sigma_j$ scaling matters empirically ($\sigma_j$ tends to be large), but the paper's point is that even with that factor neutralised, the additive projections still buy you smoothness. Do not confuse "BatchNorm shrinks gradients" with "BatchNorm rescales gradients".
- Practical read for an engineer: if BatchNorm is awkward for you (small batches, RNNs, distributed training), the paper's own evidence says you should not feel you are losing something magical about moment control. LayerNorm, GroupNorm, WeightNorm, or a hand-rolled $\ell_1$ normaliser plausibly buy the same smoothing. The [[RoFormer- Enhanced Transformer with Rotary Position Embedding|Transformer]] line went to LayerNorm for exactly this kind of reason.
- Connects nicely to why [[Deep Residual Learning for Image Recognition (ResNet)|residual connections]] help — Li et al.'s loss-landscape visualisations show the same "fewer kinks" effect, and Balduzzi et al.'s shattered-gradients paper argues the same failure mode from a different angle.
- Open question the paper raises and does not answer: is there a normalisation scheme *designed* to maximise smoothness, rather than one that stumbled into it? Nobody has really run that search.
- Historical note: this is one of the papers that came out of the "Back when we were kids" / alchemy critique (Rahimi & Recht, NIPS 2017 Test of Time) — its whole posture is "we all repeat an explanation nobody checked".

## Links

Related: [[Backpropagation]] · [[Derivative#Hessian|Hessian]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Understanding Deep Learning Requires Rethinking Generalization]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Momentum]] · [[Regularization]] · [[Cross Entropy]] · [[Deep Learning]] · [[Vector Jacobian Product]] · [[Mixed Precision training]]

New topics worth writing: Batch Normalization, Layer Normalization, Group Normalization, Weight Normalization, Lipschitz continuity, Flat minima and generalisation, Loss landscape visualisation, Shattered gradients, Glorot/Xavier initialisation, Scale invariance in neural networks
