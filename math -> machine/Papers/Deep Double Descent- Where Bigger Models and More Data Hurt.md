---
title: "Deep Double Descent: Where Bigger Models and More Data Hurt"
authors: ["Preetum Nakkiran", "Gal Kaplun", "Yamini Bansal", "Tristan Yang", "Boaz Barak", "Ilya Sutskever"]
year: 2019
arxiv: "1912.02292"
url: https://arxiv.org/abs/1912.02292
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, transformers, optimization, vision, scaling]
---
## The Core Idea

Old statistics says: make a model bigger and at some point it starts to memorise the training set, so test error goes up. A U-shaped curve. Modern practice says: bigger is always better. Both camps are right, but only about half the curve each.

The full picture has **two** descents. As you grow the model, test error goes down (classical regime), then **up** to a sharp peak, then **down again** and keeps going down. The peak sits exactly where the model becomes just barely able to fit the training data perfectly — the **interpolation threshold**. The idea of double descent came from Belkin et al. (2018) — see [[Reconciling Modern ML Practice and the Bias-Variance Trade-off]]. This paper's contribution is two things on top of that.

**First: it is not about parameter count.** The authors define a complexity measure that depends on the *training procedure*, not only the architecture.

> [!NOTE] Effective Model Complexity (EMC)
> The largest number of training samples $n$ for which a training procedure $\mathcal{T}$ still gets near-zero *training* error, on average. Formally $\mathrm{EMC}_{\mathcal{D},\epsilon}(\mathcal{T}) := \max\{n \mid \mathbb{E}_{S\sim\mathcal{D}^n}[\mathrm{Error}_S(\mathcal{T}(S))] \le \epsilon\}$, with $\epsilon = 0.1$ in practice. It depends on the data distribution, the architecture, *and* the optimiser and how long you train. ^effective-model-complexity

The hypothesis: whenever $\mathrm{EMC} \approx n$, weird things happen. Below that ($\mathrm{EMC} \ll n$), more complexity helps. Above that ($\mathrm{EMC} \gg n$), more complexity also helps. In the narrow band in between, more complexity can *hurt*.

**Second: because EMC includes training time, you get two brand-new phenomena.**

- **Epoch-wise double descent.** Fix the model. Just train longer. Test error goes down, up, then down again — for a *fixed* architecture. Training longer can undo overfitting. This had not been reported before.
- **Sample-wise non-monotonicity.** Adding training data pushes the peak to the right (you now need a bigger model to interpolate $n$ samples). If your model sits in the wrong place, that shift means **more data makes you worse**. They show a Transformer where going from 4k to 18k training sentences (4.5×) *increases* test perplexity.

What this unlocks practically: a warning label. If your model is *just barely* fitting the training set, tiny changes — a bit more width, a bit more weight decay, more data, another epoch — can move test error in an unpredictable direction.

## The Methodology

This is an empirical paper. No new algorithm; a very large grid of experiments.

**Architectures.** All three families are scaled by one width knob $k$:

- **ResNet18** (preactivation, 4 blocks) with layer widths $[k, 2k, 4k, 8k]$. Standard ResNet18 is $k=64$. See [[Deep Residual Learning for Image Recognition (ResNet)]].
- **5-layer CNN**: four Conv–[[Batch Normalization|BatchNorm]]–ReLU–MaxPool layers, widths $[k,2k,4k,8k]$, then one fully-connected layer. At $k=64$ it has 1.56M params and gets >90% on CIFAR-10 with augmentation.
- **Transformer**: the 6-layer encoder–decoder from [[Attention Is All You Need]] (fairseq), scaled by $d_{\text{model}}$ with $d_{\text{ff}} = 4 d_{\text{model}}$.

**Data.** CIFAR-10, CIFAR-100, IWSLT'14 German→English (160k sentences), WMT'14 English→French (subsampled to 200k).

**Loss.** [[Cross Entropy]] for images. For translation, per-token negative log-likelihood under the autoregressive factorisation $p_M(y|x) = \prod_i p_M(y_i \mid y_{<i}, x)$, with 10% label smoothing and no dropout.

**Optimisers.** Adam at a constant LR of $10^{-4}$ for 4K epochs, or SGD with an inverse-square-root schedule $\gamma(t) = \gamma_0 / \sqrt{1 + \lfloor t/512 \rfloor}$, $\gamma_0 = 0.1$, no momentum, 500K gradient steps. Batch size 128 everywhere. Default PyTorch init. No weight decay or dropout unless the experiment is specifically about them. They chose these because they need no per-experiment tuning.

**Label noise.** With probability $p$, a training label is replaced by a uniformly random wrong label. Sampled **once**, not per epoch — so the noisy label is a fixed, consistent target the model can memorise. Augmented copies of a sample keep the same (noisy) label. Test error is reported against the *clean* test set in most figures.

**The key visualisation.** A 2D heatmap of test error over (model size × training epoch). Model-wise double descent is a horizontal slice at the far right; epoch-wise double descent is a vertical slice at a large width. The ridge of high error runs diagonally along the interpolation threshold. Chris Olah suggested this plot, and it is what revealed epoch-wise double descent.

## Ablation Studies and Experiments

**Model-wise.** The peak appears for ResNets and CNNs on CIFAR-10/100, and for Transformers on IWSLT'14 and WMT'14. Every intervention that makes the training set *harder to fit* — adding label noise, adding data augmentation, adding more training samples — moves the peak to a **larger** model size. This is the single most convincing piece of evidence that the peak is genuinely tied to the interpolation threshold and not an artefact of a particular width.

**Label noise is an amplifier, not the cause.** With 0% noise on CIFAR-10 you get a *plateau* near the threshold; add 10–20% noise and the plateau becomes a sharp peak. But there are clean settings with a real peak: ResNet18 on CIFAR-100 with no noise, 5-layer CNN on CIFAR-100 with no noise, and Transformers on translation (which never had injected noise). The authors argue label noise is a proxy for **model mis-specification** — even pseudorandom (deterministic, invertible) noise would produce the same curve without changing the Bayes-optimal classifier.

**Epoch-wise.** Three regimes for a fixed dataset:
- Small model: test error falls monotonically with epochs. Never interpolates.
- Medium model: classical U. Early stopping is genuinely correct here.
- Large model: down, up, down. In the 10%-noise plots, the *final* test error is **lower than the first minimum**. Early stopping would have cost you accuracy.

Robust across Adam, SGD, and SGD+momentum (0.9), and across constant / inverse-sqrt / dynamic-drop learning-rate schedules. This rules out "it is an artefact of one optimiser".

**Sample-wise.** For 5-layer CNNs on CIFAR-10 with 20% noise, there is a band of model widths where training on **4× more data does not improve test error at all** — the two effects (curve gets lower, curve shifts right) cancel. For Transformers on IWSLT'14, 4k → 18k samples makes some model sizes strictly *worse*.

**What did not work / what kills the effect:**

- **Optimal early stopping mostly erases everything.** If you stop before reaching zero train error, EMC never reaches $n$, so no double descent. In the language-model plots (Figs 23, 24) with optimal early stopping, more samples is always better and test error is much lower overall. The authors never observed "more data hurts" under optimal early stopping — but they also cannot rule it out. One exception survives: ResNet18 on CIFAR-100 with no label noise still shows model-wise double descent *even with* optimal early stopping.
- **Rademacher complexity and VC dimension cannot explain this.** Both depend only on the model family and the input distribution, not on the label distribution or the training procedure. But adding label noise *moves the peak*, and epoch-wise double descent exists at all — so neither measure can locate the peak. This is the actual argument for why EMC needed inventing.
- **Ensembling helps most exactly at the peak.** A plurality vote over 5 independently-trained models (each with its own independent 15% label noise draw) closes most of the gap in the critical regime and helps much less elsewhere. This supports the mechanism: at the threshold there is essentially *one* interpolating solution, and it is highly sensitive to noise. Averaging over the noise fixes it.
- **Weight decay is another EMC axis.** Sweeping $\lambda$ from 0 to 0.1 on ResNet18/CIFAR-10/20% noise gives a qualitatively identical double-descent curve, with regularisation strength playing the role of model size. (They note using one initial LR across all $\lambda$ caused training instability and noisy plots.) Connects to [[Regularization]] and [[Decoupled Weight Decay Regularization (AdamW)]].
- **Adversarial training shows it too**, with a prominent *robust* test-error peak, even with no label noise (ResNet18, 25k CIFAR-10 samples, 10-step PGD at $\epsilon = 0.5$ and $1.0$).
- **Random Fourier Features reproduce it** (Appendix D): a 2-layer net with $e^{-ix}$ activation, first layer frozen at $\mathcal{N}(0, 1/d)$ init, second layer trained with MSE. Here $\mathrm{EMC} = d$ exactly, and the error ridge lies precisely along $n = d$. Both model-wise and sample-wise double descent appear. So this is not a deep-learning-specific pathology.

## Worth Remembering

**The intuition for the peak.** At the interpolation threshold, there is effectively only *one* function that fits the training data. It is forced to bend itself around every noisy or mis-specified label, and that destroys its global structure. Past the threshold, there are *many* interpolating solutions, and SGD's [[Understanding Deep Learning Requires Rethinking Generalization|implicit bias]] picks one that absorbs the noise locally while staying sensible elsewhere. This is theoretically justified for linear models (Bartlett et al., Mei & Montanari, Hastie et al.); for deep nets it remains a conjecture.

**Honest limitations the authors state.** $\epsilon = 0.1$ is a heuristic with no principled justification. "Sufficiently smaller" and "sufficiently larger" than $n$ are not defined. The width of the critical interval depends on the distribution and training procedure in ways they do not understand. The mechanism in deep networks is an open question.

**Practical caveats for someone using this.**
- If your model is just barely fitting the training set, treat small changes as dangerous. Widening slightly, tuning weight decay, or adding data can all move you along the ridge in a direction you did not intend.
- "Test error started rising, stop training" is not always right. For sufficiently over-parameterised models it may fall again to a *better* minimum.
- Practical takeaway if you want to avoid the mess entirely: be clearly over-parameterised, or use good early stopping. The dangerous zone is the middle.

**Connections.** The observation that any model with enough capacity can fit random labels comes from [[Understanding Deep Learning Requires Rethinking Generalization]] — that paper set up the puzzle, this one maps the shape of it. The "bigger is better" empirical practice this reconciles is the same one measured quantitatively in [[Scaling Laws for Neural Language Models]] and [[Training Compute-Optimal Large Language Models (Chinchilla)]] — note that the compute-optimal question ("given $n$, how big?") sits uncomfortably close to the critical regime this paper warns about. The idea that SGD learns functions of increasing complexity over training (Nakkiran et al. 2019) is the seed of epoch-wise double descent.

**Open questions.** Can "more data hurts" happen under optimal early stopping? Nobody knows a reason it cannot. Is there a principled way to *measure* EMC cheaply for a real training run, rather than by binary-searching over dataset sizes? And does the peak still exist for modern LLM-scale training, where you make roughly one pass over the data and never approach zero train error?

## Links

Related: [[Reconciling Modern ML Practice and the Bias-Variance Trade-off]] · [[Understanding Deep Learning Requires Rethinking Generalization]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Attention Is All You Need]] · [[Scaling Laws for Neural Language Models]] · [[Regularization]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Adam- A Method for Stochastic Optimization]] · [[Batch Normalization]] · [[Cross Entropy]] · [[Dropout- A Simple Way to Prevent Overfitting]] · [[Shortcut Learning in Deep Neural Networks]]

New topics worth writing: Effective Model Complexity, Interpolation threshold, Rademacher complexity, VC dimension, Benign overfitting in linear regression, Random Fourier Features, Label smoothing, Early stopping, Model mis-specification, Jamming transition in neural networks
