---
title: "On Layer Normalization in the Transformer Architecture"
authors: ["Xiong et al."]
year: 2020
arxiv: "2002.04745"
url: https://arxiv.org/abs/2002.04745
priority: Good-To-Read
read_on: 2026-09-25
tags: [paper, transformers, theory]
---
## The Core Idea

Training the original Transformer needs a **learning-rate warm-up**: start at almost zero learning rate and ramp up over a few thousand steps. Skip it and training collapses. Nobody knew why. This paper gives the reason, and the reason is the *position of the layer normalisation*.

> [!NOTE] Post-LN vs Pre-LN
> **Post-LN** (the original [[Attention Is All You Need]] design) is `x → sublayer → add x → LayerNorm`. The normalisation sits *between* residual blocks, on the main path. **Pre-LN** is `x → LayerNorm → sublayer → add x`. The normalisation sits *inside* the branch, and the residual path from input to output is completely clean. Pre-LN also needs one extra final LayerNorm before the output head. ^post-vs-pre-ln

The mechanism, in one line: **[[Layer Normalization|LayerNorm]] divides gradients by the size of its input, so how big the hidden states grow with depth decides how big the gradients are.**

- In Post-LN, every block ends in a LayerNorm, so the hidden state norm is reset to $\sqrt{d}$ at every layer. It does not grow with depth $L$. So the gradient at the last layer is $\mathcal{O}(d\sqrt{\ln d})$ — **independent of depth**, and large.
- In Pre-LN, the residual stream is never normalised, so its squared norm grows roughly linearly with layer index: $\mathbb{E}\|x_l\|_2^2$ is between $(1+\tfrac{l}{2})d$ and $(1+\tfrac{3l}{2})d$. The final LayerNorm therefore divides by something of size $\sqrt{L}$, and the last-layer gradient is $\mathcal{O}\!\left(d\sqrt{\tfrac{\ln d}{L}}\right)$ — **shrinking with depth**.

So at initialisation, Post-LN hands you huge gradients near the output. Apply a big learning rate to those and the first few steps blow the model apart. Warm-up is a patch: take tiny steps until the gradients settle down.

What this unlocks: if you use Pre-LN, **you can delete the warm-up stage entirely**. That removes two hyperparameters ($T_\text{warmup}$ and, in practice, much of the sensitivity to $\text{lr}_\text{max}$), and training converges faster — about 40% fewer updates to the same BERT validation loss.

## The Methodology

**The theory setup.** [[Understanding the difficulty of training deep feedforward networks (Xavier init)|Xavier initialisation]]: each weight drawn from $N(0, \tfrac{2}{n_{in}+n_{out}})$, biases zero, LayerNorm $\gamma=1,\beta=0$. Three simplifications to make the maths tractable:

1. Single head, all projection matrices $d\times d$.
2. $W^Q$ and $W^K$ initialised to **zero**. Then $QK^\top = 0$, softmax is uniform, and self-attention collapses to a plain average: $\frac{1}{n}\sum_j x_{l,j}W^{V,l}$.
3. Input embeddings are Gaussian too (reasonable — they are word + positional embeddings, both Gaussian-initialised).

They then work in the mean-field style: assume the hidden-state norms concentrate around their expectations.

> [!NOTE] $(\epsilon,\delta)$-bounded
> A non-negative random variable $Z$ is $(\epsilon,\delta)$-bounded if with probability $\ge 1-\delta$, $\frac{Z-\mathbb{E}Z}{\mathbb{E}Z}\le\epsilon$. For a $d$-dimensional standard Gaussian $Y$, $\|Y\|_2^2$ is $(\epsilon,\delta)$-bounded with $\delta=\exp(-d\epsilon^2/8)$. Plainly: the norm does not wander far from its mean. ^eps-delta-bounded

**The three lemmas that build the result.**

- **Lemma 1.** If $X\sim N(0,\sigma^2 I_d)$ then $\mathbb{E}\|\mathrm{ReLU}(X)\|_2^2 = \tfrac{1}{2}\sigma^2 d$. (ReLU kills half the mass.)
- **Lemma 2 (the heart).** At init, Post-LN: $\mathbb{E}\|x^{post,5}_{l,i}\|_2^2 = \tfrac{3}{2}d$ for every $l$ — constant. Pre-LN: $(1+\tfrac{l}{2})d \le \mathbb{E}\|x^{pre}_{l,i}\|_2^2 \le (1+\tfrac{3l}{2})d$ — grows linearly in depth. The growth comes from the fact that each residual branch adds roughly $\tfrac{d}{2}$ (FFN) or up to $d$ (attention) of independent energy that never gets renormalised.
- **Lemma 3.** $\|\mathbf{J}_{LN}(x)\|_2 = \mathcal{O}\!\left(\frac{\sqrt{d}}{\|x\|_2}\right)$. The Jacobian of LayerNorm is $\frac{\sqrt d}{\|y\|_2}\left(I - \frac{y^\top y}{\|y\|_2^2}\right)\left(I - \tfrac{1}{d}\mathbf{1}^\top\mathbf{1}\right)$; both projection matrices have eigenvalues 0 or 1, so the scale is entirely set by $1/\|x\|_2$.

Put together: gradient magnitude $\propto \|\mathbf{J}_{LN}\| \propto 1/\|\text{input to that LayerNorm}\|$. Post-LN's input norm is $\Theta(\sqrt d)$ regardless of $L$; Pre-LN's is $\Theta(\sqrt{Ld})$. That is **Theorem 1**.

**Extension to lower layers (appendix, less rigorous).** Chaining the per-layer Jacobians, Post-LN picks up a factor $\mathbb{E}\|\partial x_{j+1}/\partial x_j^{post,5}\| \approx \sqrt{d/\tfrac{3}{2}d} = \sqrt{2/3}$ at every layer. So the gradient at layer $l$ carries $\mathcal{O}\big((2/3)^{(L-l)/2}\big)$ — **gradients shrink exponentially as you move away from the output**. Post-LN is therefore *both* too big at the top and too small at the bottom. In Pre-LN, once $l$ is large the LayerNorm Jacobians are $\mathcal{O}(1/\sqrt j)$, the block Jacobians are close to identity, and the gradient norm is roughly the same at every layer.

**Training setup.** IWSLT14 De-En (153K pairs, 10K joint [[Neural Machine Translation of Rare Words with Subword Units|BPE]] vocab, 6-6 encoder-decoder, $d=512$, FFN 1024, 4 heads, label smoothing 0.1, dropout 0.1, 4096-token batches). WMT14 En-De (4.5M pairs, 37K BPE, Transformer *base*, 8192 tokens/GPU on 16 P40s). BERT *base* (12 layers, $d=768$, 12 heads) on Wikipedia + a self-crawled BookCorpus, ~3.4B words, 32 P40s.

Pre-LN runs have **no warm-up at all**: IWSLT starts at $5e^{-4}$ and decays $\times 0.1$ at epoch 8; WMT starts at $7e^{-4}$ or $1.5e^{-3}$ and decays at epoch 6 into an inverse-sqrt schedule; BERT starts at $3e^{-4}$ with linear decay. Post-LN baselines use $T_\text{warmup}=4000$ (translation) or 10k (BERT).

## Ablation Studies and Experiments

**Is warm-up actually essential for Post-LN?** (IWSLT14 De-En)

| Optimiser | Warm-up | BLEU |
|---|---|---|
| Adam | yes ($T=4000$) | ~34 |
| Adam | no | **8.45** |
| SGD | yes | (poor but trains) |
| SGD | no | ~0 after 15 epochs |

The SGD result is the important one. [Liu et al. 2019 / RAdam] argued warm-up exists to tame the variance of Adam's adaptive learning rate. But **SGD has no adaptive learning rate and still needs warm-up** — so the explanation cannot be Adam-specific.

**Is it sensitive to $T_\text{warmup}$?** Badly. With $T_\text{warmup}=500$: BLEU 31.16 at $\text{lr}_\text{max}=5e^{-4}$, and **2.77** at $1e^{-3}$. Halving a schedule hyperparameter can cost you the whole model.

**Do the gradient predictions hold empirically?**
- Measured hidden-state norms are $(0.1, 0.125)$-bounded — the concentration assumption is fine in practice.
- Gradient norm of $W^{2,L}$ across 6-6 / 8-8 / 10-10 / 12-12 / 14-14 models: Post-LN stays flat at ~1.6 as depth grows; Pre-LN decreases. Matches Theorem 1.
- Per-layer gradient norms in a 6-6 model: Post-LN grows with layer index (exactly the $(2/3)^{(L-l)/2}$ decay running backwards); Pre-LN is nearly flat across layers.
- After warm-up finishes, Post-LN gradients *are* small — so the problem really is an initialisation-time problem, not a permanent one.

**Main results with warm-up removed from Pre-LN.**

| Task | Result |
|---|---|
| IWSLT14 De-En | Pre-LN, no warm-up: BLEU ~34, val loss ~4 — matches Post-LN + warm-up. Pre-LN's 9th checkpoint ≈ Post-LN's 15th. |
| WMT14 En-De | Same pattern; Pre-LN converges faster at equal $\text{lr}_\text{max}$. |
| BERT pre-training | Pre-LN reaches the same validation loss (1.69) about 200k updates earlier than Post-LN's 500k — the authors quote a **40% speed-up**. Since $T_\text{warmup}$ was only 10k, the saving is *not* just the skipped warm-up steps: Pre-LN genuinely tolerates a larger learning rate. |
| GLUE MRPC / RTE | Pre-LN checkpoints also converge faster downstream. |

**Things that did not work.**

- **Post-LN BERT at $3e^{-4}$ diverged**, even with warm-up. It only trained at $1e^{-4}$. Pre-LN ran at $3e^{-4}$ with no warm-up.
- **RAdam is not a substitute for moving the LayerNorm.** RAdam does let Post-LN train without warm-up, but applied to Pre-LN it is indistinguishable from plain Adam. The authors' phrasing: the change of LN position "dominates" the change of optimiser.
- **A permanently tiny learning rate is only a half-fix.** Training Post-LN from scratch at a fixed $1e^{-4}$ (no warm-up) reaches val loss ~4.3 in 27 epochs — much better than the blown-up large-LR run, but still clearly worse than the warm-up baseline, and far slower. Small steps dodge the instability; they do not buy you the final quality.

The ablations pin the cause cleanly: it is not the optimiser, not the schedule shape, and not the batch size. It is **the size of the gradient at the top of a Post-LN stack at step zero**, and that is set by where you put the LayerNorm.

## Worth Remembering

- **This is why almost every modern LLM is Pre-LN.** GPT-2 onwards, LLaMA, most decoder-only stacks: Pre-LN (usually [[Layer Normalization|RMSNorm]]) plus a final norm. The paper is one of the main reasons.
- **The honest caveat the field learned later:** Pre-LN trains easily but often reaches slightly *worse* final quality than a well-tuned Post-LN at the same depth, because the residual stream keeps growing and later layers contribute proportionally less. That is the flip side of the exact same $\sqrt{L}$ normalisation that makes it stable. Post-LN deep-model work (DeepNet, Admin) exists precisely to get Post-LN quality with Pre-LN stability. This paper does not measure that gap.
- **The theory is at initialisation only.** It says nothing about mid-training dynamics. It happens to be enough, because warm-up is an initialisation-time patch — but do not over-extend it.
- **The zero-init of $W^Q, W^K$ is a real simplification.** Uniform attention is not what a trained (or even a Xavier-initialised) model does. The empirical checks are what make the argument convincing, not the proof alone.
- **Warm-up for [[Accurate, Large Minibatch SGD- Training ImageNet in 1 Hour|large-batch SGD]] is a different phenomenon.** There, warm-up compensates for a learning rate scaled up linearly with batch size. Here, batch size is unremarkable and warm-up is fixing an architectural gradient imbalance. Two problems, same patch — worth keeping separate in your head. See [[GPU processing]].
- **Practical recipe if you are starting a Transformer from scratch:** Pre-LN, no warm-up, a peak LR roughly 3× what Post-LN tolerated, then linear or inverse-sqrt decay. If you must use Post-LN (e.g. fine-tuning [[BERT- Pre-training of Deep Bidirectional Transformers|BERT]]), keep the warm-up and treat $T_\text{warmup}$ as a real hyperparameter you have to tune.
- **Follow-up questions:** does the same $1/\|x\|$ argument explain why RMSNorm behaves similarly? What does the residual-stream growth do to [[Quantization]] and activation ranges at 70B+ scale? And does the $(2/3)^{(L-l)/2}$ backward decay show up as measurable underuse of early layers in trained Post-LN models?

## Links
Related: [[Attention Is All You Need]] · [[Layer Normalization]] · [[Batch Normalization]] · [[How Does Batch Normalization Help Optimization]] · [[Group Normalization]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Understanding the difficulty of training deep feedforward networks (Xavier init)]] · [[Delving Deep into Rectifiers (He init, PReLU)]] · [[Adam- A Method for Stochastic Optimization]] · [[Accurate, Large Minibatch SGD- Training ImageNet in 1 Hour]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Backpropagation]] · [[GPU processing]] · [[SGDR- Stochastic Gradient Descent with Warm Restarts]] · [[Cyclical Learning Rates for Training Neural Networks]] · [[On the difficulty of training Recurrent Neural Networks]]

New topics worth writing: Pre-LN vs Post-LN trade-off in deep Transformers (DeepNet, Admin), RMSNorm, RAdam and adaptive-learning-rate variance, mean-field theory of neural network initialisation, Fixup initialisation (normalisation-free residual nets), learning-rate warm-up as a general diagnostic
