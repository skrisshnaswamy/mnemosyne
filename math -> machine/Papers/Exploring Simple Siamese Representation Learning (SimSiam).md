---
title: "Exploring Simple Siamese Representation Learning (SimSiam)"
authors: ["Xinlei Chen", "Kaiming He"]
year: 2020
arxiv: "2011.10566"
url: https://arxiv.org/abs/2011.10566
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, self-supervised, vision]
---
## The Core Idea

Self-supervised image learning had a standard recipe: take one image, make two random crops/colour-jitters of it, push their embeddings together. The obvious failure mode is **collapse** — the network outputs the same constant vector for every image, similarity is perfect, loss is minimal, representation is useless.

Everyone had a device to stop this. [[A Simple Framework for Contrastive Learning (SimCLR)|SimCLR]] pushes *other* images apart (negative pairs), which needs huge batches. [[Momentum Contrast (MoCo)|MoCo]] keeps a queue of negatives plus a slowly-updated momentum encoder. SwAV forces a balanced cluster assignment. [[Bootstrap Your Own Latent (BYOL)|BYOL]] dropped negatives but kept a momentum encoder, and its authors reported that removing the momentum encoder gives 0.3% accuracy — total collapse.

SimSiam removes **all of it**. No negatives. No momentum encoder. No large batch. Two views, one shared-weight encoder, a small prediction head on one side, and one `.detach()` on the other side. That gets 67.7% ImageNet linear-probe accuracy at 100 epochs — beating SimCLR, MoCo v2, BYOL and SwAV at the same epoch budget.

The finding underneath: the thing preventing collapse was never the negatives, never the momentum encoder, never batch norm. It was **stop-gradient**. Momentum encoders always carry a stop-gradient with them (they are not updated by gradients), so BYOL's ablation confused the two. Strip the momentum away but keep the detach, and the model is fine.

> [!NOTE] Stop-gradient
> Treat a tensor as a constant during [[Backpropagation|backprop]] — it still contributes to the loss value, but no [[Derivative#Gradient|gradient]] flows back through it. In PyTorch, `z.detach()`. ^stop-gradient

Why this matters conceptually: the paper argues stop-gradient is not a hack but the *signature of a different optimisation problem*. If you are alternating between two sets of variables (like k-means alternates between centres and assignments), then when you optimise one set the other is frozen — a stop-gradient by definition. SimSiam is one SGD step of that alternation.

And the broader claim: since all five methods differ wildly but all work, the shared **Siamese structure** — weight sharing across two views — is likely the real source of the success. It is an inductive bias for invariance, in the same way [[ImageNet Classification with Deep CNNs (AlexNet)#^convolutional-layer|convolution]] is a weight-sharing inductive bias for translation invariance.

## The Methodology

**Architecture.** One image $x$, two random augmentations $x_1, x_2$.

- Encoder $f$ = ResNet-50 backbone + 3-layer projection MLP. Hidden width 2048, output $d = 2048$. Batch norm on every fc layer *including the output* fc; no ReLU on the output.
- Predictor $h$ = 2-layer MLP, $2048 \to 512 \to 2048$. A **bottleneck**: hidden is 1/4 of output. BN on the hidden layer, none on the output.
- Weights of $f$ are fully shared between the two branches. $h$ is applied to one side only.

**Loss.** Negative cosine similarity,

$$\mathcal{D}(p_1, z_2) = -\frac{p_1}{\lVert p_1\rVert_2}\cdot\frac{z_2}{\lVert z_2\rVert_2}$$

where $p_1 = h(f(x_1))$ and $z_2 = f(x_2)$. This equals mean-squared error on $\ell_2$-normalised vectors up to a factor of 2. Symmetrised, with the detach:

$$\mathcal{L} = \tfrac12\mathcal{D}(p_1, \texttt{stopgrad}(z_2)) + \tfrac12\mathcal{D}(p_2, \texttt{stopgrad}(z_1))$$

Minimum possible value is $-1$ — which is exactly what a collapsed model achieves.

Note what the detach does *not* do: the encoder on $x_2$ still gets gradients, just from the $p_2$ term, not the $z_2$ term.

**Training.** Plain SGD. No LARS. Base $lr = 0.05$, scaled linearly as $lr \times \text{BatchSize}/256$, cosine decay. Weight decay 1e-4 applied to *everything* including BN scales and biases (unlike SimCLR/BYOL, which exclude them). Momentum 0.9. Default batch 512, fits on 8 GPUs. Synchronised BN across devices. 100 epochs for ablations. Augmentations are the standard SimCLR set: RandomResizedCrop (scale 0.2–1.0), horizontal flip, ColorJitter (0.4/0.4/0.4/0.1, p=0.8), grayscale (p=0.2), Gaussian blur.

Whole thing is ~10 lines of PyTorch.

**The EM hypothesis.** Consider a loss with an extra set of variables $\eta$, one vector $\eta_x$ per image in the dataset:

$$\mathcal{L}(\theta, \eta) = \mathbb{E}_{x,\mathcal{T}}\big[\lVert \mathcal{F}_\theta(\mathcal{T}(x)) - \eta_x \rVert_2^2\big]$$

Minimise over both. This is structurally k-means: $\theta$ plays the role of cluster centres, $\eta_x$ the assignment for sample $x$. Solve by alternation:

$$\theta^t \leftarrow \arg\min_\theta \mathcal{L}(\theta, \eta^{t-1}), \qquad \eta^t \leftarrow \arg\min_\eta \mathcal{L}(\theta^t, \eta)$$

The first step is SGD on $\theta$ with $\eta$ frozen — **that is the stop-gradient**. The second step has a closed form under squared error:

$$\eta_x^t \leftarrow \mathbb{E}_{\mathcal{T}}\big[\mathcal{F}_{\theta^t}(\mathcal{T}(x))\big]$$

i.e. $\eta_x$ is the *average* embedding of image $x$ over all augmentations. SimSiam approximates this by sampling one augmentation and dropping the expectation, and by doing one SGD step per alternation.

**Where the predictor comes in.** Dropping $\mathbb{E}_\mathcal{T}$ is a crude approximation. The optimal $h$ satisfies $h(z_1) = \mathbb{E}_z[z_2] = \mathbb{E}_\mathcal{T}[f(\mathcal{T}(x))]$ — exactly the expectation that was thrown away. So the predictor's job is to *learn to predict the augmentation-average*, with the sampling of $\mathcal{T}$ spread implicitly across epochs.

**Where symmetrisation comes in.** It is just denser sampling of $\mathcal{T}$ — an extra pair per image. Helps accuracy, irrelevant to collapse.

## Ablation Studies and Experiments

All ImageNet linear-probe accuracy, 100-epoch pre-training unless stated.

**Stop-gradient (Figure 2) — the headline.**

| | acc. |
|---|---|
| with stop-grad | 67.7 ± 0.1 (5 trials) |
| without | **0.1** (chance) |

Everything else identical. Without the detach, the loss crashes to $-1$ almost immediately. The diagnostic: per-channel std of the $\ell_2$-normalised output $z/\lVert z\rVert$. If outputs are a constant vector, std is 0 — that is what happens. With stop-grad, std sits near $1/\sqrt{d}$, which is what you get for a zero-mean isotropic Gaussian scattered on the unit hypersphere. Nice, cheap collapse monitor to steal.

**Predictor $h$ (Table 1).**

| variant | acc. |
|---|---|
| baseline ($h$ with cosine-decayed lr) | 67.7 |
| (a) no $h$ (identity) | 0.1 — collapses |
| (b) $h$ fixed at random init | 1.5 — does *not* collapse, just fails to converge |
| (c) $h$ with constant lr, no decay | **68.1** |

(a) is explainable: with the symmetric loss and $h$ = identity, the gradient direction is identical to the no-stop-grad case, scaled by 1/2. So stop-gradient becomes a no-op and you collapse. (b) is a different failure — loss stays high, no collapse; $h$ must *learn*. (c) is the surprise: never decaying the predictor's learning rate beats decaying it, because $h$ should chase the latest representations rather than converge early. They use constant-lr $h$ for all later ablations.

**Batch size (Table 2).**

| 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| 66.1 | 67.3 | 68.1 | 68.1 | 68.0 | 67.9 | 64.0 |

Flat from 256 to 2048 with plain SGD. Batch 64 still works. The 4096 drop is a known SGD-at-large-batch problem, not a collapse — LARS would probably fix it. Contrast with SimCLR and SwAV, which *need* 4096.

**Batch norm (Table 3).**

| BN config | acc. |
|---|---|
| (a) none in MLP heads | 34.6 — bad, but no collapse |
| (b) hidden layers only | 67.4 |
| (c) + BN on projection output (default) | 68.1 |
| (d) + BN on prediction output | unstable, loss oscillates |

So BN helps optimisation, exactly as in supervised training, and does nothing for collapse prevention — the stop-grad experiment had identical BN on both sides. Also: the learnable affine scale/offset in $f$'s output BN can be disabled with no loss (68.2%). This directly rebuts the folk theory of the time that BN was secretly acting as an implicit contrastive term.

**Similarity function.** Swap cosine for cross-entropy, $\mathcal{D}(p_1,z_2) = -\texttt{softmax}(z_2)\cdot\log\texttt{softmax}(p_1)$ (softmax over the $d$ channels, so $d$ "pseudo-categories"). Gets 63.2% vs 68.1%. Worse, but no collapse — so collapse prevention is not a property of cosine similarity.

**Symmetrisation.**

| sym. | asym. | asym. 2× |
|---|---|---|
| 68.1 | 64.8 | 67.3 |

Asymmetric works fine. Sampling two pairs per image in the asymmetric version ("2×") mostly closes the gap, confirming symmetrisation is just denser sampling.

**Output dimension.** 256/512/1024/2048 → 65.3/67.2/67.5/68.1. Saturates at 2048, later than SimCLR/MoCo/BYOL which saturate at 256–512. The predictor bottleneck (hidden = 1/4 of output) matters: making hidden equal to output makes training unstable or fails in some variants.

**Proof-of-concept for the EM story.** Multi-step alternation: pre-compute and cache $\eta_x$ for all images, then do $k$ SGD steps on $\theta$ before refreshing.

| 1-step (= SimSiam) | 10-step | 100-step | 1-epoch |
|---|---|---|---|
| 68.1 | 68.7 | **68.9** | 67.0 |

All work; the medium intervals are *better* than SimSiam. Strong support that alternating optimisation is the right frame.

Second proof-of-concept, and the cleanest one: **remove the predictor entirely**, but maintain $\eta_x$ as a moving average, $\eta_x^t \leftarrow m\eta_x^{t-1} + (1-m)\mathcal{F}_{\theta^t}(\mathcal{T}'(x))$ with $m = 0.8$ (essentially a memory bank). This is another way to approximate $\mathbb{E}_\mathcal{T}[\cdot]$. Result: **55.0% without $h$**, versus 0.1% if you remove $h$ without the moving average. The predictor and the augmentation-expectation are doing the same job.

**Things that did NOT work — the reverse ablations.** Bolting SimSiam's tricks onto other methods:

| SimCLR | + predictor | + predictor & stop-grad |
|---|---|---|
| 66.5 | 66.4 | 66.0 |

| SwAV | + predictor | remove stop-grad |
|---|---|---|
| 66.5 | 65.2 | **NaN** (diverges) |

The predictor is not a free accuracy booster — it is specific to SimSiam's underlying problem. And SwAV cannot lose its stop-gradient either, because clustering is inherently alternating.

**Main comparison (Table 4, ResNet-50, two 224² views, all reproduced by the authors, several *better* than the original papers).**

| method | batch | neg? | mom-enc? | 100 ep | 200 | 400 | 800 |
|---|---|---|---|---|---|---|---|
| SimCLR+ | 4096 | ✓ | | 66.5 | 68.3 | 69.8 | 70.4 |
| MoCo v2+ | 256 | ✓ | ✓ | 67.4 | 69.9 | 71.0 | 72.2 |
| BYOL | 4096 | | ✓ | 66.5 | 70.6 | 73.2 | **74.3** |
| SwAV+ | 4096 | | | 66.5 | 69.1 | 70.7 | 71.8 |
| **SimSiam** | 256 | | | **68.1** | 70.0 | 70.8 | 71.3 |

SimSiam wins at 100 epochs and beats SimCLR everywhere, but it **scales worst with training length** — BYOL overtakes it by 3 points at 800 epochs.

**Transfer (Table 5, 200-epoch pre-training).** VOC 07+12 detection: SimSiam-optimal AP 57.0, matching MoCo v2's 57.0 and beating ImageNet-supervised 53.5. COCO detection AP 39.2, again tying MoCo v2 and beating supervised 38.2. Every one of these self-supervised methods matches or beats supervised pre-training on transfer. Note the recipe change: $lr=0.5, wd=$ 1e-5 gives the same ImageNet accuracy but better transfer everywhere — a reminder that linear-probe accuracy does not rank transfer quality.

**CIFAR-10**, ResNet-18, 800 epochs: SimSiam 91.8% vs SimCLR 91.1%.

## Worth Remembering

- **The authors do not claim to explain why collapse is prevented.** The EM hypothesis explains *what problem is being solved* and why stop-gradient appears, not why the trajectory avoids the constant solution. Their hand-wave: $\eta$ is initialised from a random network, so it is not constant; and because gradients w.r.t. $\eta$ are never computed jointly across all images, the optimiser cannot easily steer everything to one point. This is an open question that spawned a literature (DirectPred, the eigenspace-alignment analyses).
- **Collapsing solutions genuinely exist** for this loss and architecture. Predictor, BN and $\ell_2$-normalisation together are *not* enough. Architecture alone does not save you.
- **Initialisation is fragile.** Default PyTorch fc init $\mathcal{U}(-\sqrt{k},\sqrt{k})$ with $k = 1/\text{in\_channels}$ works; a fixed std of 0.01 may not converge. Also, last-BN scale initialised to 0 in each [[Deep Residual Learning for Image Recognition (ResNet)#^residual-block|residual block]].
- **The std-of-normalised-output monitor** (should sit near $1/\sqrt{d}$) plus a kNN probe on validation are the two cheap diagnostics to run during any Siamese SSL training. Loss going to its minimum is a *bad* sign here.
- **Practical caveat: SimSiam plateaus.** If you have compute for 800+ epochs, BYOL's momentum encoder buys you real accuracy. SimSiam's advantage is the short-schedule, small-batch, single-8-GPU regime.
- The momentum encoder is reframed here as *one way to solve the $\eta$ sub-problem smoothly over time*. Other optimisers for $\eta$ are open territory. This makes [[Bootstrap Your Own Latent (BYOL)|BYOL]], SwAV and SimSiam three points in one design space rather than three unrelated tricks.
- Connects nicely to the alignment/uniformity view — SimSiam optimises alignment with no explicit uniformity term at all, yet the outputs still spread over the sphere. That is the puzzle.
- The predictor with a **non-decayed learning rate** beating the decayed one is a transferable trick: any auxiliary head whose job is to track a moving target probably should not have its lr annealed.

## Links

Related: [[Bootstrap Your Own Latent (BYOL)]] · [[A Simple Framework for Contrastive Learning (SimCLR)]] · [[Momentum Contrast (MoCo)]] · [[Understanding Contrastive Learning through Alignment and Uniformity]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Self-Supervised Learning from Images with I-JEPA]] · [[LeJEPA- Provable and Scalable Self-Supervised Learning]] · [[Batch Normalization]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[Backpropagation]] · [[Mode Collapse]] · [[Momentum]] · [[Pytorch Autograd]]

New topics worth writing: Siamese networks, Expectation–Maximization, k-means as alternating optimisation, Sinkhorn–Knopp / SwAV online clustering, LARS optimizer, linear-probe evaluation protocol, memory bank / instance discrimination, DirectPred and the theory of why SimSiam avoids collapse
