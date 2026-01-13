---
title: "An Image is Worth 16x16 Words (ViT)"
authors: ["Alexey Dosovitskiy", "Lucas Beyer", "Alexander Kolesnikov", "Dirk Weissenborn", "Xiaohua Zhai", "Thomas Unterthiner", "Mostafa Dehghani", "Matthias Minderer", "Georg Heigold", "Sylvain Gelly", "Jakob Uszkoreit", "Neil Houlsby"]
year: 2020
arxiv: "2010.11929"
url: https://arxiv.org/abs/2010.11929
priority: Must-Read
read_on: 2026-08-22
tags: [paper, transformers, llm, vision]
---
## The Core Idea

Take the Transformer from NLP. Do not change it. Feed it an image.

The trick is how you turn a picture into a sequence of "words". Cut the image into a grid of square patches — say $16\times16$ pixels each. Flatten each patch into a vector of numbers. Multiply by one learned matrix to get a $D$-dimensional embedding. Now you have a sequence of ~200 vectors, exactly the shape a [[Attention Is All You Need|Transformer]] encoder eats. Add a position embedding, prepend a `[class]` token like [[BERT- Pre-training of Deep Bidirectional Transformers|BERT]], run 12–32 encoder layers, classify from the class token's output.

That is the whole model. No convolutions anywhere.

Why did nobody do this before? Two reasons. First, attention costs $O(N^2)$ in sequence length, and if $N$ = number of pixels ($224^2 = 50{,}176$) it is hopeless. Patching fixes this: $224/16 = 14$, so $N = 196$. Cheap. Second, and more important, people had tried and it did not work. On ImageNet alone, ViT loses to a [[Deep Residual Learning for Image Recognition (ResNet)|ResNet]] of the same size. The paper's real finding is that this failure is a *data* problem, not an *architecture* problem.

> [!NOTE] Inductive bias
> A built-in assumption an architecture makes about the world. A CNN assumes that nearby pixels belong together (locality) and that a cat is a cat wherever it sits in the frame (translation equivariance). These assumptions are free knowledge — they save the model from learning it from data. ^inductive-bias

A CNN's inductive bias is a head start. It is also a ceiling. ViT has almost none — only the patch-cutting step knows anything about 2D. So ViT learns slower from small data, but it keeps learning long after the ResNet has stopped. Cross the data threshold and ViT wins.

The crossover happens somewhere between 14M and 300M images. On JFT-300M, ViT-H/14 hits **88.55%** ImageNet top-1 versus BiT-L's 87.54% — while using **2.5k TPUv3-core-days instead of 9.9k**. Four times less compute, better accuracy.

The unlock: vision no longer needs its own architecture. The same block, the same code, the same scaling infrastructure now works on pixels and text, which is what made CLIP, multimodal LLMs and vision [[Foundation Models|foundation models]] practical.

## The Methodology

**Patchify.** Image $\mathbf{x} \in \mathbb{R}^{H\times W\times C}$ → $N = HW/P^2$ flattened patches $\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 C)}$. With $224\times224$ RGB and $P=16$: each patch is $16\cdot16\cdot3 = 768$ numbers, and $N=196$.

**Embed.** One shared [[Linear Projection|linear projection]] $\mathbf{E} \in \mathbb{R}^{(P^2 C)\times D}$ maps each patch to $D$ dims. Prepend a learnable class token, add learnable 1D position embeddings:

$$\mathbf{z}_0 = [\mathbf{x}_{\text{class}};\, \mathbf{x}_p^1\mathbf{E};\, \cdots;\, \mathbf{x}_p^N\mathbf{E}] + \mathbf{E}_{pos}$$

Note: this projection is mathematically identical to a $16\times16$ [[ImageNet Classification with Deep CNNs (AlexNet)#^convolutional-layer|convolution]] with stride 16. That is the one and only convolution in the model, and it happens once.

**Encode.** $L$ pre-norm blocks. LayerNorm *before* each sub-block, [[Deep Residual Learning for Image Recognition (ResNet)#^residual-block|residual]] after:

$$\mathbf{z}'_\ell = \mathrm{MSA}(\mathrm{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}$$
$$\mathbf{z}_\ell = \mathrm{MLP}(\mathrm{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell$$

MSA is standard multi-head [[Query, Key, and Value (QKV)|QKV]] attention, $A = \mathrm{softmax}(\mathbf{q}\mathbf{k}^\top/\sqrt{D_h})$, no masking — every patch sees every patch, unlike [[Causal Attention|causal attention]] in GPT. The MLP is two layers with GELU.

**Classify.** $\mathbf{y} = \mathrm{LN}(\mathbf{z}_L^0)$, the class token's final state. MLP head with one hidden layer during pre-training; a single linear layer at fine-tune time.

**Sizes** (borrowed straight from BERT):

| | Layers | $D$ | MLP | Heads | Params |
|---|---|---|---|---|---|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

Notation `ViT-L/16` = Large, patch size 16. Smaller patch → longer sequence → quadratically more compute.

**Pre-training.** Adam ($\beta_1=0.9$, $\beta_2=0.999$), batch 4096, weight decay **0.1** (unusually high, and it mattered), 10k step linear warmup then linear decay, resolution 224. JFT-300M for 7 or 14 epochs; ImageNet-21k for 30–90 epochs. Supervised classification loss — plain [[Cross Entropy|cross entropy]] over 18k or 21k classes.

**Fine-tuning.** Throw away the head, attach a zero-initialised $D\times K$ linear layer. Switch to SGD with [[Momentum|momentum]] 0.9, batch 512, no weight decay, grad clip at norm 1, cosine decay, and — importantly — **higher resolution (384, or 512/518 for the headline numbers)**. Same patch size at higher resolution means more patches, so the pre-trained position embeddings no longer line up. Fix: 2D-interpolate them onto the new grid. This interpolation and the patching step are the only places 2D structure is hand-injected.

**Hybrids.** Instead of raw patches, feed a ResNet50's feature map into ViT with "patch size 1". Tested as a middle ground.

## Ablation Studies and Experiments

**Headline (Table 2).** All pre-trained on JFT-300M unless noted.

| | ViT-H/14 | ViT-L/16 | ViT-L/16 (I21k) | BiT-L (R152x4) | Noisy Student |
|---|---|---|---|---|---|
| ImageNet | **88.55** | 87.76 | 85.30 | 87.54 | 88.4 |
| ImageNet-ReaL | **90.72** | 90.54 | 88.62 | 90.54 | 90.55 |
| CIFAR-100 | **94.55** | 93.90 | 93.25 | 93.51 | – |
| VTAB (19 tasks) | **77.63** | 76.28 | 72.72 | 76.29 | – |
| TPUv3-core-days | 2.5k | **0.68k** | 0.23k | 9.9k | 12.3k |

ViT-L/16 beats BiT-L on every task at **1/14th the compute**. The ImageNet-21k model (public data, 8-core TPUv3 for ~30 days) is the practically reproducible one.

**The data-scale ablation — the paper's real argument (Figure 3/4, Table 5).** Same models, three pre-training sets:

- Pre-trained on ImageNet-1k: ViT-B/16 gets 77.91 on ImageNet, ViT-L/16 gets **76.53**. The bigger model is *worse*. BiT ResNets beat both.
- Pre-trained on ImageNet-21k: B/16 → 83.97, L/16 → 85.15. Big model now slightly ahead.
- Pre-trained on JFT-300M: B/16 → 84.15, L/16 → **87.12**. Big model clearly ahead, and ViT overtakes BiT.

Second cut: random JFT subsets of 9M / 30M / 90M / 300M, no extra regularisation, same hyperparameters, early stopping, few-shot linear probe. ViT-B/32 (slightly cheaper than ResNet50) is *much worse* at 9M and *better* at 90M+. Same story for ResNet152x2 vs ViT-L/16. So the crossover is a property of the model, not of tuned [[Regularization|regularisation]].

**Compute-scaling study (Figure 5).** 7 ResNets, 6 ViTs, 5 hybrids, all on JFT, plotted against pre-training exaFLOPs. Three findings:
1. ViT needs **2–4× less compute** than ResNet for equal transfer accuracy.
2. Hybrids beat pure ViT at small budgets, and **the advantage vanishes at large budgets**. The authors call this surprising — you would expect convolutional feature extraction to help at any size. It does not.
3. ViT shows no saturation in the range tested — consistent with the [[Scaling Laws for Neural Language Models#^power-law|power-law]] picture from language.

**What did not work:**

- **Global average pooling head "performed very poorly"** at first — but this was a red herring. The gap is entirely explained by needing a different learning rate. With the right LR, GAP and the `[class]` token are equivalent. A nice warning about attributing architecture wins to architecture.
- **Fancier position embeddings did nothing.** ViT-B/16, ImageNet 5-shot linear: no positional info **0.614**; 1D **0.642**; 2D **0.640**; relative **0.640**. Injecting them at every layer, or sharing across layers: also ~0.64. So *having* position matters a lot; *how* you encode it does not. Reason: at $14\times14$ patch resolution, spatial relations are easy to learn either way. Confirmed by inspection — the learned 1D embeddings develop row/column structure and cosine-similarity that decays with 2D distance, i.e. they rediscover the grid on their own.
- **Axial attention** (attend along rows, then columns) improved Axial-ViT-B/32 and B/16 over plain ViT-B in accuracy, but cost more compute per block (two attention ops + two MLPs). AxialResNet looked fine on FLOPs but was "extremely slow on TPUs" — a good reminder that FLOPs ≠ wall clock.
- **Self-supervised masked patch prediction** (mask 50% of patches, 80/10/10 replace-random-keep as in BERT, predict 3-bit mean colour → 512 classes) gave ViT-B/16 **79.9%** on ImageNet: +2% over from-scratch, but **4% behind supervised pre-training**. Variants tried: predict a $4\times4$ downsampled patch (works), L2 regression on the full patch (slightly worse), 15% mask rate as in BERT (slightly worse). Returns diminished after 100k steps — so this did *not* need JFT-scale data, and it did *not* close the gap. The [[BERT- Pre-training of Deep Bidirectional Transformers#^masked-language-model|MLM]] magic did not transfer directly. (MAE later fixed this.)
- **Shape ablation (Figure 8).** From an 8-layer, $D=1024$, patch-32 base: scaling **depth** helps most, visible to 64 layers though diminishing after 16. Scaling **width** helps least. Shrinking patch size (longer sequence) gives robust gains **with zero extra parameters** — evidence that compute predicts performance better than parameter count.
- **Adam beats SGD for pre-training ResNets** here (ResNet50 average over 5 datasets: 89.33 vs 88.79; ResNet152x2: 94.01 vs 93.72). Contrary to standard practice, and needed for a fair baseline.

**Attention behaviour (Figure 7 right, 11).** Mean attention distance ≈ receptive field. In the lowest layers, some heads already span most of the image (global mixing is genuinely used) while other heads stay tightly local — and this local behaviour largely disappears in hybrids, suggesting those heads are doing the job of early convolutions. Distance grows monotonically with depth; by the second half of the network nearly all heads are global. Attention-rollout maps land on the semantically relevant object.

## Worth Remembering

- **The one-line takeaway: "large scale training trumps inductive bias."** But note the threshold. ImageNet-1k (1.3M) is not enough. ImageNet-21k (14M) is roughly break-even. JFT-300M is where ViT wins. If you have 10k images and no pre-trained checkpoint, use a CNN.
- Nobody trains ViT from scratch. In practice you fine-tune a checkpoint, and then the data-hunger caveat evaporates. DeiT (2021) later showed you *can* train ViT on ImageNet-1k alone with heavy augmentation and [[Distillation|distillation]] from a CNN teacher.
- **ViT is more memory-efficient than ResNet at equal accuracy** (Figure 12) — larger per-core batch sizes, comparable wall-clock throughput. The feared bi-quadratic blowup with image size barely shows up except for the largest models at the largest resolutions. Later, [[Flash Attention]] made the quadratic term much less painful still.
- Weight decay **0.1** during pre-training and **0** during fine-tuning. High WD for transfer was a Big Transfer finding the authors carried over and found important for all models, ViT and ResNet alike.
- The GAP-vs-class-token episode is the most reusable methodological lesson: an architecture change that looks catastrophic may just want a different learning rate. Always re-sweep before concluding.
- The hybrid result cuts against intuition. Convolutional pre-processing is a crutch that helps small models and holds back nothing at scale — but also helps nothing at scale.
- Limitations the authors name: only classification (detection and segmentation untested — DETR hinted it would work, Swin and ViTDet later confirmed), self-supervision still far behind supervised, and no sign of saturation so "further scaling would likely lead to improved performance" (ViT-22B, 2023, delivered).
- Practical caveat: JFT-300M is proprietary. The reproducible path is ImageNet-21k → ViT-L/16 → 85.30 ImageNet, which is still very strong and ~0.23k TPU-core-days.
- Open question worth chasing: the patch embedding throws away all sub-patch structure with a single linear map. Its top principal components look like Gabor-ish basis functions — i.e. it reinvents the first layer of a CNN. Why does one layer of that suffice?

## Links

Related: [[Attention Is All You Need]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[Query, Key, and Value (QKV)]] · [[Linear Projection]] · [[Cross Entropy]] · [[Regularization]] · [[Flash Attention]] · [[Distillation]] · [[Foundation Models]] · [[Causal Attention]] · [[Momentum]]

New topics worth writing: Inductive bias in architectures, Layer Normalization and pre-norm vs post-norm, GELU activation, Group Normalization and weight standardisation, Big Transfer (BiT), VTAB benchmark, DeiT and data-efficient ViT training, Masked Autoencoders (MAE), Axial attention, Attention rollout, Transfer learning at higher resolution, Adam vs SGD for vision models
