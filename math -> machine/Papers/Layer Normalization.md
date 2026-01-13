---
title: "Layer Normalization"
authors: ["Jimmy Lei Ba", "Jamie Ryan Kiros", "Geoffrey E. Hinton"]
year: 2016
arxiv: "1607.06450"
url: https://arxiv.org/abs/1607.06450
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, transformers]
---
## The Core Idea

[[Batch Normalization]] normalises **across the batch**: for each neuron, take its pre-activation value over all examples in the mini-batch, subtract the batch mean, divide by the batch standard deviation. That works well in feed-forward nets but breaks in three places:

1. **Recurrent nets.** Sequences have different lengths. Each time step sees a different distribution of activations, so you need separate statistics per time step. If a test sentence is longer than anything you saw in training, you have no statistics for those steps.
2. **Small batches.** With batch size 4 the mean and variance estimates are garbage. With batch size 1 they are undefined.
3. **Train/test mismatch.** BatchNorm has to keep running averages and does different arithmetic at test time than at training time.

Layer normalisation flips the axis. Instead of normalising one neuron across many examples, normalise **many neurons across one example**. Take the vector of summed inputs to a layer for a single training case, compute its mean and standard deviation over the $H$ hidden units, and standardise with those.

$$\mu^{l}=\frac{1}{H}\sum_{i=1}^{H}a_i^{l}\qquad \sigma^{l}=\sqrt{\frac{1}{H}\sum_{i=1}^{H}\left(a_i^{l}-\mu^{l}\right)^{2}}$$

Every unit in the layer shares the same $\mu$ and $\sigma$; every training case gets its own. That single change removes all three problems at once. No batch dependency, so batch size 1 is fine. No time-step dependency, so RNNs are trivial. And the computation at test time is *identical* to training — nothing to store, nothing to average.

> [!NOTE] Layer normalisation
> Standardise the pre-activation vector of one example using statistics computed over the hidden units of that layer, then apply a learned per-unit gain $g_i$ and bias $b_i$. Statistics never cross examples. ^layer-norm

The unlock: this is the normalisation that lives inside every [[Attention Is All You Need|Transformer]] block today. Its property of not needing a batch is exactly what makes it work for variable-length sequences and for the tiny effective batches you get in distributed training.

## The Methodology

**The operation.** For a layer with pre-activations $\mathbb{a}$ (the output of $W\mathbb{x}$, a [[Linear Projection|linear projection]], before the non-linearity):

$$\mathbb{h}= f\!\left[\frac{\mathbb{g}}{\sigma}\odot(\mathbb{a}-\mu)+\mathbb{b}\right]$$

$\mathbb{g}$ (gain) and $\mathbb{b}$ (bias) are learned vectors of length $H$. Default init: gains to $1$, biases to $0$.

**In an RNN.** Compute $\mathbb{a}^t = W_{hh}h^{t-1} + W_{xh}\mathbb{x}^t$, then normalise using $\mu^t,\sigma^t$ from that time step's own $H$ units. One set of $\mathbb{g},\mathbb{b}$ shared across all time steps. Because the layer is now invariant to rescaling the whole summed-input vector, the average magnitude of the recurrent state cannot drift upward or downward step after step — which is the usual source of exploding or vanishing gradients in [[Long Short-Term Memory (Neural Computation)|LSTMs]].

**In an LSTM specifically** (appendix), they normalise three things separately, not the concatenated sum:

$$(\mathbf{f}_t,\mathbf{i}_t,\mathbf{o}_t,\mathbf{g}_t)^\top = LN(\mathbf{W}_h\mathbf{h}_{t-1};\alpha_1,\beta_1) + LN(\mathbf{W}_x\mathbf{x}_t;\alpha_2,\beta_2) + b$$
$$\mathbf{c}_t = \sigma(\mathbf{f}_t)\odot\mathbf{c}_{t-1} + \sigma(\mathbf{i}_t)\odot\tanh(\mathbf{g}_t)$$
$$\mathbf{h}_t = \sigma(\mathbf{o}_t)\odot\tanh\big(LN(\mathbf{c}_t;\alpha_3,\beta_3)\big)$$

Note: the recurrent contribution and the input contribution get their own LN, and the cell state gets a third. The cell state is *not* normalised inside the recurrence — the [[Long Short-Term Memory (Neural Computation)#^constant-error-carrousel|constant error carrousel]] stays additive and clean; LN is applied only when reading $\mathbf{c}_t$ out.

**The invariance table** (Section 5.1) is the theoretical heart:

| | weight matrix rescale | weight matrix re-centre | single weight vector rescale | dataset rescale | dataset re-centre | single case rescale |
|---|---|---|---|---|---|---|
| BatchNorm | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ |
| WeightNorm | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| LayerNorm | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ |

The two cells that matter. LayerNorm is invariant to **re-centring the whole weight matrix**: if $W' = \delta W + \mathbf{1}\gamma^\top$, then $\mathbb{h}' = \mathbb{h}$ exactly, because the shift $\gamma^\top\mathbb{x}$ is identical for every unit and gets absorbed by subtracting $\mu$. And it is invariant to **rescaling a single training case**: scale $\mathbb{x}$ by $\delta$ and both $\mu$ and $\sigma$ scale by $\delta$, so the ratio is unchanged. BatchNorm has neither of those; LayerNorm gives up per-neuron weight-vector rescaling invariance in exchange.

**Why it stabilises learning (Section 5.2).** They analyse a generalised linear model and compute the Fisher information matrix — the [[KL Divergence|KL]]-based Riemannian metric $ds^2 \approx \tfrac12\delta^\top F(\theta)\delta$ that measures how much a parameter step changes the model's output distribution. The block along $w_i$ picks up a factor $g_i/\sigma_i$. If $\|w_i\|$ doubles, $\sigma_i$ doubles too, so the curvature along that direction halves. Large weights therefore get an automatically smaller effective step size — an implicit learning-rate decay and "early stopping" effect that falls out of the normalisation, not from the optimiser. Separately, learning the gain $g$ under LN or BN depends only on the size of the prediction error, whereas in a plain GLM it depends on $\|\mathbb{x}\|$ — so gain learning is robust to input scale.

## Ablation Studies and Experiments

Six tasks, five of them recurrent. Optimiser was [[Adam- A Method for Stochastic Optimization|Adam]] throughout.

**Order embeddings (MSCOCO image–caption retrieval, GRU encoder + VGG image features).** LN reaches its best validation model in **60% of the baseline's iterations**. Test set, caption retrieval R@1: 46.6 → **48.5**; image retrieval R@1: 37.8 → **38.9**. Mean rank 7.9 → 7.6.

**Attentive reader on CNN QA corpus** — the direct comparison against recurrent BatchNorm (Cooijmans et al. 2016). LN trains faster *and* converges to a better validation result than both the plain baseline and both BN variants.

**The one real ablation, and it is a good one.** Cooijmans et al. found recurrent BN only works if you initialise the BN gain to **0.1**; at 1.0 it degrades badly. Ba et al. ran LN with both initialisations and found **1.0 clearly better than 0.1**. So LN is not sensitive to gain initialisation the way recurrent BN is. That is a real practical difference — one fewer fragile hyperparameter.

**Skip-thought vectors** (2400-d sentence encoder, BookCorpus). After 1M iterations: SICK Pearson $r$ 0.842 → **0.854**, SICK MSE 0.298 → **0.277**, MR 77.3 → **79.5**, CR 81.8 → **82.6**, SUBJ 92.6 → **93.4**, MPQA 87.9 → **89.0**. Train a month (~1.7M iters) and it improves further on all but one task. They worried the 2400-d layer would make LN slow per iteration; with CNMeM there was no significant wall-clock difference.

**DRAW on binarised MNIST** (64 glimpses, 256 LSTM units). Converges roughly **twice as fast**. Final test variational bound after 200 epochs: 82.36 → **82.09 nats**. So the gain here is almost entirely speed, not quality.

**Handwriting generation, IAM-OnDB.** The stress test: sequences ~700 steps long, mini-batch size **8**, three layers of 400 LSTM cells, 3.7M weights, 20 bivariate Gaussian mixture outputs. LN reaches comparable log-likelihood much faster. This is the regime where BatchNorm simply cannot compete — small batch, very long sequence.

**Permutation-invariant MNIST, 784-1000-1000-10 feed-forward.** LN applied to hidden layers only, **not** the final softmax layer — because the logits' scale *is* the model's confidence, and normalising it away would be wrong. At batch size 128 LN converges faster than BN applied everywhere; at batch size **4** the gap widens sharply, since BN's variance estimate falls apart.

**What did not work: convolutional nets.** Stated plainly in Section 6.7. LN beats no-normalisation but **BatchNorm beats LN** on ConvNets. The reason given is the key insight: LN assumes every unit in a layer contributes similarly, so sharing one $\mu,\sigma$ across them is sensible. In a [[ImageNet Classification with Deep CNNs (AlexNet)#^convolutional-layer|convolutional layer]] that assumption fails — units whose receptive field sits near the image border are rarely activated and have statistics quite unlike units in the centre. Averaging them together pollutes the statistics. The authors say "further research is needed" and leave it.

## Worth Remembering

- The whole thing is one axis swap. BatchNorm normalises down the batch dimension; LayerNorm normalises across the feature dimension. Everything else — learned gain and bias, placement before the non-linearity — is the same.
- **Placement matters and the paper is explicit about it.** LN is applied to the summed inputs $a$, *after* the weight multiply, *before* the non-linearity. If you instead normalise the input before the weights, you lose the weight re-scaling and re-centring invariance entirely.
- **Do not normalise the output logits.** Prediction confidence is encoded in logit magnitude; LN would erase it.
- The Fisher-matrix argument is the most under-quoted part. It says normalisation gives you an *implicit* learning-rate schedule tied to weight norm, independent of whatever [[Momentum|momentum]] or adaptive scheme you use. Connects to the same question asked in [[How Does Batch Normalization Help Optimization]] — is the benefit distribution-fixing or optimisation-landscape-smoothing? Both papers land on "it is about the geometry", not the covariate-shift story from the original BN paper.
- The paper's framing is still "reduce covariate shift", which later work largely rejected as the explanation. Treat the motivation section as historical and the invariance/geometry sections as the real content.
- Two properties LN gives up versus BN: it is **not** invariant to rescaling a single neuron's weight vector, and it has **no regularising noise** — BN's stochasticity from batch statistics acts as a mild regulariser, LN's statistics are deterministic given the input. So LN models may need more explicit [[Regularization|regularisation]].
- History note: the ConvNet failure is why GroupNorm, InstanceNorm and friends were later invented — they interpolate between "one statistic per neuron" and "one statistic per layer". And LN's real payoff arrived a year later in the Transformer, where the feature dimension is the natural axis anyway.
- Practical caveat: LN adds a reduction over the feature dimension in both forward and backward pass. At 2400 dimensions the authors saw no slowdown given a good memory allocator, but on modern hardware LN is memory-bandwidth bound and is one of the things fused kernels ([[FlashAttention- Fast and Memory-Efficient Exact Attention|FlashAttention]]-style thinking) target.
- Open question worth chasing: the paper keeps the mean subtraction. RMSNorm later showed you can drop $\mu$ entirely and divide only by the root-mean-square, losing almost nothing — which suggests the re-centring invariance the authors prove in Eq. 6 is not where the benefit actually lives.

## Links
Related: [[Batch Normalization]] · [[How Does Batch Normalization Help Optimization]] · [[Long Short-Term Memory (Neural Computation)]] · [[Attention Is All You Need]] · [[Adam- A Method for Stochastic Optimization]] · [[KL Divergence]] · [[Backpropagation]] · [[Linear Projection]] · [[Derivative#Hessian|Hessian]] · [[Seq2Seq models]] · [[Regularization]]

New topics worth writing: RMSNorm, Group Normalization, Weight Normalization, Fisher information matrix, natural gradient descent, Pre-LN vs Post-LN Transformer placement, recurrent batch normalization, skip-thought vectors
—
