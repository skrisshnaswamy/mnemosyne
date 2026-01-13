---
title: "Delving Deep into Rectifiers (He init, PReLU)"
authors: ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"]
year: 2015
arxiv: "1502.01852"
url: https://arxiv.org/abs/1502.01852
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, vision]
---
## The Core Idea

Two small changes to rectified networks, both cheap, both about the same fact: **ReLU throws away half the signal, and nobody had done the algebra for that.**

1. **PReLU.** The negative side of ReLU is not fixed at zero. Give it a slope $a_i$ and *learn* it by [[Backpropagation|backprop]] like any other weight. One extra number per channel. Leaky ReLU had already tried a fixed $a=0.01$ and found it did nothing; learning $a$ is what makes it work.

2. **He initialisation.** [[Understanding the difficulty of training deep feedforward networks (Xavier init)|Xavier init]] derives its variance rule assuming the network is **linear**. It is not. A ReLU zeroes about half of its inputs, so it halves the variance of the signal at every layer. Xavier therefore shrinks the signal by $1/\sqrt{2}$ per layer. Over $L$ layers that is $2^{-L/2}$ — harmless at 10 layers, fatal at 30. Fix the factor of two and the problem disappears.

The unlock is concrete: a 30-layer plain conv net **converges from scratch** with He init and **completely stalls** with Xavier (gradients monitored and confirmed to be vanishing). Before this, VGG had to train an 8-layer model first and use it to seed the deeper ones; GoogLeNet bolted auxiliary classifiers onto intermediate layers. Both are crutches for a bad initial variance. He init removed the crutch.

Result: 4.94% top-5 on ImageNet 2012 test, versus GoogLeNet's 6.66% and the reported human number of 5.1%. First published claim of beating a human on this benchmark.

> [!NOTE] He initialisation
> Draw weights from $\mathcal{N}(0, 2/n_l)$ where $n_l = k^2 c$ is the fan-in of the layer (or $\hat n_l = k^2 d$, the fan-out, for the backward version). The $2$ compensates for ReLU killing half the units. ^he-init

> [!NOTE] PReLU
> $f(y_i) = \max(0, y_i) + a_i \min(0, y_i)$, with $a_i$ learned per channel. $a_i=0$ gives ReLU, $a_i=1$ gives the identity. ^prelu

## The Methodology

### PReLU, exactly

$$f(y_i)=\begin{cases} y_i & y_i > 0\\ a_i y_i & y_i \le 0\end{cases}$$

$i$ indexes the **channel**, so a conv layer with 256 filters gets 256 extra scalars. There is also a *channel-shared* variant with one $a$ per layer — 13 extra parameters for the whole 14-layer net.

Gradient, straight from the chain rule:
$$\frac{\partial \mathcal{E}}{\partial a_i} = \sum_{y_i} \frac{\partial \mathcal{E}}{\partial f(y_i)} \cdot \frac{\partial f(y_i)}{\partial a_i}, \qquad \frac{\partial f(y_i)}{\partial a_i}=\begin{cases}0 & y_i>0\\ y_i & y_i \le 0\end{cases}$$
summed over every spatial position in the feature map.

Three training details that matter:
- Updated with [[Momentum|momentum]]: $\Delta a_i := \mu \Delta a_i + \epsilon \frac{\partial \mathcal{E}}{\partial a_i}$.
- **No weight decay on $a_i$.** L2 pulls $a_i \to 0$, which is exactly reverting to ReLU. This is the same trap that [[Decoupled Weight Decay Regularization (AdamW)|AdamW]] worries about from another angle: decay applied blindly to every parameter is not free.
- **No range constraint** on $a_i$, so the activation is allowed to become non-monotonic. Initialised at $a_i = 0.25$. In practice learned values stayed below 1 anyway.

### The initialisation derivation

For a conv layer $\mathbf{y}_l = W_l \mathbf{x}_l + \mathbf{b}_l$, with $n_l = k^2 c$ connections per output, weights zero-mean and i.i.d., independent of the inputs:
$$\mathrm{Var}[y_l] = n_l \,\mathrm{Var}[w_l]\, E[x_l^2]$$

The pivot is that $E[x_l^2] \ne \mathrm{Var}[x_l]$, because $x_l = \max(0, y_{l-1})$ **does not have zero mean**. This is precisely where Glorot's linear assumption breaks. If $w_{l-1}$ is symmetric about zero and $b_{l-1}=0$, then $y_{l-1}$ is symmetric about zero, so
$$E[x_l^2] = \tfrac12 \mathrm{Var}[y_{l-1}]$$

Chaining $L$ layers:
$$\mathrm{Var}[y_L] = \mathrm{Var}[y_1]\prod_{l=2}^{L} \tfrac12 n_l \mathrm{Var}[w_l]$$

Set each factor to 1:
$$\boxed{\tfrac12 n_l \mathrm{Var}[w_l] = 1 \;\Rightarrow\; \text{std} = \sqrt{2/n_l}}$$

The backward pass gives the same shape. Since $\Delta y_l = f'(y_l)\Delta x_{l+1}$ and $f'$ is 0 or 1 with equal probability, $E[(\Delta y_l)^2] = \tfrac12 \mathrm{Var}[\Delta x_{l+1}]$, so $\tfrac12 \hat n_l \mathrm{Var}[w_l]=1$ with $\hat n_l = k^2 d_l$ (fan-out). **Either condition alone is enough** — if you fix the backward scaling, the forward product comes out to $c_2/d_L$, a constant, not something that shrinks exponentially. The paper uses the backward form, Eqn. 14.

For PReLU the half becomes:
$$\tfrac12(1+a^2)\, n_l \mathrm{Var}[w_l] = 1$$
so $a=0$ recovers He, $a=1$ recovers Xavier.

**Why 0.01 std fails, with numbers.** For VGG model B, the correct stds are 0.059 / 0.042 / 0.029 / 0.021 for layers with 64 / 128 / 256 / 512 filters. Using 0.01 everywhere, the gradient reaching conv2 from conv10 has std $1/(5.9 \times 4.2^2 \times 2.9^2 \times 2.1^4) \approx 1/(1.7\times10^4)$ of what it should be.

**Practical caveat they admit:** He init roughly *preserves* input variance to the last layer. If your input is in $[-128,128]$ and unnormalised, the softmax overflows. Their hack: use std 0.01 for the first two fc layers and 0.001 for the last — deliberately smaller than $\sqrt{2/4096}$ — to absorb the image scale.

### Architectures

Model A is VGG-19 rearranged: 7×7 stride-2 first layer, three conv layers moved off the 224/112 maps onto the 56/28/14 maps (same FLOPs, faster wall-clock because big feature maps are slow per FLOP), and spatial pyramid pooling with $\{7,3,2,1\}$ bins = 63 bins before fc1. B = A + 3 conv layers (22 layers). C = B but wider (384/768/896 filters), 2.3× the compute.

Training: 224 crops from images with shorter side jittered in $[256,512]$ **from epoch 0** (VGG only jittered during fine-tuning), per-pixel mean subtracted, horizontal flip on half the samples, colour altering. Weight decay 0.0005, momentum 0.9, [[Regularization|dropout]] 50% on the first two fc layers, batch 128, LR $10^{-2}\to10^{-3}\to10^{-4}$ on plateau, ~80 epochs. 3–4 weeks on 4 K20s (A/B) or 8 K40s (C). Data-parallel conv layers, fc layers on a single GPU (cheap, not worth parallelising): 3.8× on 4 GPUs, 6.0× on 8.

## Ablation Studies and Experiments

**PReLU vs ReLU, 14-layer model, ImageNet 10-view (top-1 / top-5):**

| | top-1 | top-5 |
|---|---|---|
| ReLU | 33.82 | 13.34 |
| PReLU, channel-shared | 32.71 | 12.87 |
| PReLU, channel-wise | 32.64 | 12.75 |

The interesting bit: channel-**shared** — 13 extra parameters total — recovers almost the whole 1.2% gain. The win is not capacity, it is the *shape* of the nonlinearity.

**What the learned $a_i$ look like.** conv1 learns $a \approx 0.6$–$0.68$ — nearly linear. Deeper conv layers drop to $0.12$–$0.20$, fc layers to $0.03$–$0.07$. So the network chooses to stay near-linear early (keep both the positive and negative response of the Gabor-like edge filters, when you only have 64 of them) and become sharply nonlinear late. Nice, interpretable, and something you get for free.

**PReLU on the big model A, dense multi-scale testing:** top-1 24.02 → 22.97, top-5 6.51 → 6.28. Best single scale is 384, the middle of the jitter range.

**Init, the headline ablation.** On a 22-layer model, both Xavier and He converge; He just starts dropping error earlier. On the **30-layer** model (27 conv + 3 fc), He converges and Xavier *completely stalls*, with confirmed diminishing gradients, and does not recover with more epochs.

**Accuracy from init alone: essentially nothing.** On the 14-layer ReLU model, Xavier gives 33.90/13.44 and He gives 33.82/13.34. The authors say plainly they observed no clear superiority on accuracy. He init buys you **trainability at depth**, not points.

### What did not work

- **Extreme depth did not help.** The 30-layer model they could finally train gets 38.56/16.59 — much *worse* than the 14-layer model's 33.82/13.34. Adding 3–9 conv layers on top of model B degraded both train and test error in the first 20 epochs. This is the degradation problem, and it is why the same authors wrote [[Deep Residual Learning for Image Recognition (ResNet)|ResNet]] the following year. This paper explicitly gives up on depth and goes **wider** instead.
- **Depth 19 vs 22 is a wash:** A+PReLU 22.97/6.28 vs B+PReLU 22.85/6.27. Width is what moves the number: C+PReLU 21.59/**5.71**.
- Fixed Leaky ReLU ($a=0.01$, from Maas et al.) had negligible effect on accuracy — the point is learning $a$, not just avoiding zero gradients.
- Model A is **not claimed to be a better architecture** than VGG-19. With less scale augmentation the two were comparable. A is just faster: 2.6s/batch vs 3.0s.

**Final tables.** Single model, val: MSRA A+ReLU 6.51 already beats VGG-19's 7.1 (they credit end-to-end training rather than stagewise pre-training). Best single model C+PReLU 5.71 — better than every previous *multi*-model result. Six-model ensemble on test: **4.94%**, vs GoogLeNet 6.66% (26% relative) and Baidu 5.98%.

## Worth Remembering

- **The "human 5.1%" number is one annotator**, trained on the val set, given an interface with 13 example images per class, evaluated on a random 1500-image subset. The authors are careful: beating it does not mean machine vision beats human vision. Human errors are mostly fine-grained species confusion and class unawareness — exactly what a CNN is good at. The model still fails on "spotlight" (38% top-5 error) and "restaurant" (36%), which need context and world knowledge.
- Per-class breakdown: zero top-5 error on 113 classes. Versus their own ILSVRC14 entry, error dropped in 824 classes, unchanged in 127, **increased in 49**.
- The ensemble is unbalanced — only one model of architecture C, the rest considerably worse. They guess fewer, stronger models would do better.
- **Practical:** `torch.nn.init.kaiming_normal_(w, mode='fan_in'|'fan_out', nonlinearity='relu')` is Eqn. 10 / Eqn. 14. `mode` picks which of the two conditions you enforce; the paper says either works. If you use PReLU or LeakyReLU, pass `a` so the $\tfrac12(1+a^2)$ correction applies.
- **Do not weight-decay your PReLU slopes.** Frameworks decay everything by default; you have to exclude them explicitly, same as you exclude [[Layer Normalization|LayerNorm]] gains and biases.
- Historically this is the moment right before [[Batch Normalization|BatchNorm]] (same year) made careful init much less critical for moderately deep nets — and right before ResNet made *very* deep nets work for a different reason. He init still matters: it is the default for conv/MLP stacks, and normalisation-free architectures depend on it entirely.
- Open question worth chasing: the learned $a_i$ decreasing with depth says the network *wants* to be linear early. Does that intuition survive in transformers, where the early-layer nonlinearity story is different?

## Links

Related: [[Understanding the difficulty of training deep feedforward networks (Xavier init)]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Batch Normalization]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Backpropagation]] · [[Momentum]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Gradient-Based Learning Applied to Document Recognition (LeNet)]] · [[Deep Learning]] · [[Regularization]] · [[Random variable]]

New topics worth writing: Leaky ReLU and the activation-function zoo (ELU, GELU, Swish), Spatial Pyramid Pooling, VGG, GoogLeNet / Inception, the degradation problem in plain deep nets, fan-in vs fan-out initialisation modes, ILSVRC and the human-baseline methodology, data-parallel vs model-parallel training
Note file: `He Init and PReLU - Delving Deep into Rectifiers.md`
