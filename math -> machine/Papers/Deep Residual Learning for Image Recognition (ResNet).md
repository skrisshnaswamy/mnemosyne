---
title: "Deep Residual Learning for Image Recognition (ResNet)"
authors: ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"]
year: 2015
arxiv: "1512.03385"
url: https://arxiv.org/abs/1512.03385
priority: Must-Read
read_on: 2026-08-22
tags: [paper, transformers, optimization, vision]
---
## The Core Idea

Before this paper, everyone believed "deeper is better" — but only up to a point. Past ~20 layers, stacking more layers made things **worse**, and not because of overfitting. A 56-layer plain convolutional network had *higher training error* than a 20-layer one on CIFAR-10. Higher **training** error means the network could not even memorise the data it was shown. That is an optimisation failure, not a generalisation failure.

The authors call this the **degradation problem**, and they point out how strange it is. Take a trained 20-layer network. Add 36 more layers that each just copy their input to their output (an identity map). You now have a 56-layer network with *exactly* the same training error as the 20-layer one. So a solution at least that good provably exists. [[Backpropagation|Gradient descent]] just cannot find it.

The fix is a re-parameterisation, not a new layer type. Instead of asking a stack of layers to output the function you want, $\mathcal{H}(\mathbf{x})$, ask it to output only the *difference* from its input:

$$\mathcal{F}(\mathbf{x}) := \mathcal{H}(\mathbf{x}) - \mathbf{x}, \qquad \mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

Same expressive power. Wildly different optimisation landscape. If the best thing a block can do is nothing, the network now only has to push its weights toward **zero** — which SGD is very good at — instead of learning to reproduce the identity function through two nonlinear layers, which it is apparently very bad at.

> [!NOTE] Residual block
> A few layers whose output is added back to their own input. The layers learn a *correction* to the signal rather than the signal itself. Costs no extra parameters and no extra compute — just an element-wise add. ^residual-block

The key subtlety: this is **not** primarily a vanishing-gradient fix. The authors explicitly rule that out. Their plain baselines already use batch normalisation, and they checked that backward gradients had healthy norms. Signals were not vanishing in either direction. The plain nets just converged unusably slowly — the authors conjecture "exponentially low convergence rates" and say the real reason is future work. So residual learning is a **preconditioning** trick: it puts the optimiser's starting point near a good solution, in the same spirit as multigrid solvers for PDEs, where each level solves for the residual between a coarse and a fine scale.

What it unlocked: depth stopped being a scaling bottleneck. 152 layers on ImageNet, 1202 layers on CIFAR-10 with training error under 0.1%. Every deep architecture since — including the [[Attention Is All You Need|Transformer]] — has skip connections around every sub-block, and it is because of this paper.

## The Methodology

**The block.** For a two-layer block, $\mathcal{F} = W_2\,\sigma(W_1\mathbf{x})$ where $\sigma$ is [[ImageNet Classification with Deep CNNs (AlexNet)#^relu|ReLU]]. Add the shortcut, *then* apply the second ReLU: $\sigma(\mathcal{F}(\mathbf{x}) + \mathbf{x})$. Order matters — the addition happens before the final nonlinearity.

Note the paper says a **one-layer** residual block, $\mathbf{y} = W_1\mathbf{x} + \mathbf{x}$, showed no advantage. It collapses to a plain linear layer. You need at least two layers inside for the residual framing to buy anything.

**When dimensions change.** Channels double and spatial size halves at three points in the net, so $\mathbf{x}$ and $\mathcal{F}(\mathbf{x})$ no longer have the same shape. Three options:
- **(A)** identity shortcut with zero-padding on the new channels — zero extra parameters
- **(B)** a $1\times1$ convolution ([[Linear Projection|linear projection]] $W_s$) only where dimensions change, identity everywhere else
- **(C)** $1\times1$ projections on *every* shortcut

**The backbone.** Built on VGG's design rules: mostly $3\times3$ convolutions, same filter count for the same feature-map size, double the filters when the map halves. Downsampling by stride-2 convolutions, no pooling in the middle. Ends with global average pooling and one 1000-way fully-connected softmax layer. No hidden FC layers at all — this is why ResNet-34 is 3.6 GFLOPs versus VGG-19's 19.6 GFLOPs, only 18% of the cost.

**The bottleneck block** (for 50/101/152 layers). Three layers instead of two: $1\times1$ to squeeze channels down, $3\times3$ to do the actual work in the narrow space, $1\times1$ to expand back up. E.g. 256 → 64 → 64 → 256. Same time cost as a two-layer $3\times3$ block, but three layers deep. Here identity shortcuts are not just cheap but *essential*: the shortcut connects the two wide 256-d ends, so replacing it with a projection doubles both model size and time.

Depths from Table 1: ResNet-50 = 3.8 GFLOPs, ResNet-101 = 7.6, ResNet-152 = 11.3. All still cheaper than VGG-16 (15.3).

**Training (ImageNet).** SGD, batch 256, [[Momentum|momentum]] 0.9, weight decay $10^{-4}$, LR starts at 0.1 and ÷10 on plateau, up to $60\times10^4$ iterations. Batch norm after every conv and before the activation. He initialisation. **No dropout.** Scale augmentation (shorter side sampled from $[256, 480]$), random $224\times224$ crops, horizontal flips, per-pixel mean subtraction, AlexNet-style colour augmentation.

**Training (CIFAR-10).** $6n+2$ layers, filters $\{16,32,64\}$, option A shortcuts throughout so plain and residual nets have *identical* parameter counts. Batch 128, LR 0.1 ÷10 at 32k and 48k iters, stop at 64k.

## Ablation Studies and Experiments

**The degradation problem, demonstrated** (Table 2, ImageNet top-1 error, 10-crop):

| | plain | ResNet |
|---|---|---|
| 18 layers | 27.94 | 27.88 |
| 34 layers | 28.54 | **25.03** |

Read this table carefully — it is the whole paper. Going 18 → 34 makes the plain net *worse* (27.94 → 28.54). Going 18 → 34 makes the ResNet *better* by 2.8 points. And at 34 layers, the ResNet beats the plain net by 3.5 points with **zero extra parameters**. The 34-layer plain net's *training* curve sits above the 18-layer one for the entire run.

At 18 layers the two are basically tied (27.94 vs 27.88). Residual learning does not help a shallow net's final accuracy — it just converges faster early on. The benefit is purely a deep-network phenomenon.

**Shortcut type** (Table 3, top-1 / top-5):
- ResNet-34 A (zero-pad): 25.03 / 7.76
- ResNet-34 B (project only on dim change): 24.52 / 7.46
- ResNet-34 C (project everywhere): 24.19 / 7.40

C is best but only by 0.84 points over A, at the cost of thirteen extra projection matrices. The authors' reading: the gap A→B exists because zero-padded channels carry no residual learning at all; the gap B→C is just extra parameters. **Projections are not what makes residual learning work.** They dropped C everywhere else.

**Depth scaling** (Table 3, top-1): ResNet-34 → 24.19, ResNet-50 → 22.85, ResNet-101 → 21.75, ResNet-152 → 21.43. Monotone improvement, no degradation. Single-model ResNet-152 got 4.49% top-5 on val — better than every previous *ensemble*. The six-model ensemble hit **3.57% top-5 on the test set** and won ILSVRC 2015 (GoogLeNet's 2014 winner was 6.66%).

**CIFAR-10** (Table 6): ResNet-20 8.75%, -32 7.51%, -44 7.17%, -56 6.97%, **-110 6.43%** with only 1.7M parameters. Highway Networks at 19 layers with 2.3M params managed 7.54%. So ResNet-110 beats it with fewer parameters and 5× the depth.

**What did not work:**
- **ResNet-1202** trained fine — training error below 0.1%, no optimisation trouble at all — but test error was **7.93%**, worse than the 110-layer net's 6.43%. Authors attribute this to overfitting: 19.4M parameters on 50k images with no dropout or maxout. So residual learning solves optimisation, not [[Regularization|regularisation]].
- **ResNet-110 would not start converging at LR 0.1.** They had to warm up at 0.01 for ~400 iterations until training error dropped below 80%, then switch back to 0.1. An early, ad-hoc instance of LR warm-up.
- **Single-layer residual blocks** gave no advantage.

**The response-magnitude analysis** (Fig. 7) is the ablation that reveals the mechanism. They measure the standard deviation of each $3\times3$ layer's output (after BN, before the nonlinearity). ResNet layer responses are systematically *smaller* than plain-net responses, and **the deeper the ResNet, the smaller they get** (ResNet-110 < ResNet-56 < ResNet-20). Each layer in a deep ResNet nudges the signal only slightly. This is direct evidence for the hypothesis: the optimal function is closer to identity than to zero, and residual blocks learn small perturbations around identity.

**Transfer** (with Faster R-CNN, swapping VGG-16 for ResNet-101, everything else identical): PASCAL VOC 2007 mAP 73.2 → 76.4; COCO mAP@[.5,.95] **21.2 → 27.2**, a 28% relative gain purely from better features. ImageNet localisation top-5 error went from VGG's 25.3% to 9.0% — a 64% relative error reduction. First place in ImageNet classification, detection, localisation, COCO detection and COCO segmentation, all in one year.

## Worth Remembering

- **The authors do not claim to know why plain nets fail.** They rule out vanishing gradients with evidence, then write "we conjecture that the deep plain nets may have exponentially low convergence rates" and "the reason for such optimization difficulties will be studied in the future." A landmark paper resting on a phenomenon it cannot explain. Later work (loss-landscape visualisation, identity-mapping analysis) filled some of this in.
- **The "constructed solution" argument is the intellectual core** and it generalises far beyond vision. Whenever a bigger model underperforms a smaller one on *training* loss, and the bigger model provably contains the smaller one as a special case, you have an optimisation bug, not a capacity bug. Re-parameterise so the trivial solution is easy to reach.
- Residual learning is orthogonal to gating. Highway Networks (concurrent) used learned, data-dependent gates that can *close* and block the signal. ResNet's shortcuts are never closed — information always flows. Highway never demonstrated gains past ~30 layers; ResNet went to 1202. Parameter-free beat learned here.
- **Practical caveat for detection fine-tuning:** they froze the BN layers after ImageNet pre-training, recomputing statistics once and then treating BN as a fixed affine transform. Reason given was memory in Faster R-CNN training, but this became standard practice for small-batch fine-tuning generally, since BN statistics are garbage at batch size 1–2.
- The paper's own regularisation stance is "deep and thin architectures by design" — no dropout, no maxout. That works until ResNet-1202. If you want to go absurdly deep on small data, you still need real regularisation.
- Note the connection to [[Distillation|distillation]]-adjacent work: FitNets (19 layers, 2.5M params, 8.39%) was the prior "how do we train thin deep nets" answer and needed hint-based supervision from a teacher. ResNet beat it with a structural change and no teacher at all.
- Open question worth chasing: the response-magnitude result suggests very deep ResNets are doing something like an iterative refinement / unrolled ODE. That reading became the basis for Neural ODEs and for the "ResNets are ensembles of shallow paths" line of work.

## Links

Related: [[Deep Learning]] · [[Backpropagation]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Attention Is All You Need]] · [[Momentum]] · [[Regularization]] · [[Distillation]] · [[Linear Projection]] · [[Derivative]] · [[MLE_L4_Question_Bank]] · [[Senior Machine Learning Engineer — Interview Question Bank]] · [[Foundational_Reading_Plan]]

New topics worth writing: Batch Normalization, VGG and the 3x3-convolution design rules, Highway Networks and gated shortcuts, Faster R-CNN, He (Kaiming) initialisation, Learning-rate warm-up, Identity Mappings in Deep Residual Networks (pre-activation ResNet v2), Neural ODEs, Degradation vs. overfitting as diagnostic categories, Global average pooling
