---
title: "ImageNet Classification with Deep CNNs (AlexNet)"
authors: ["Krizhevsky et al."]
year: 2012
url: https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
priority: Must-Read
read_on: 2026-08-22
tags: [paper, vision]
---
## The Core Idea

Before 2012, image recognition worked like this: a human designed a feature extractor (SIFT, Fisher Vectors, sparse coding), and a shallow classifier sat on top. Neural nets that learned features from raw pixels existed since the 1980s, but nobody had made one big enough to matter. Two things had just changed: ImageNet gave you 1.2 million labelled photos instead of tens of thousands, and GPUs got fast enough to push a 60-million-parameter net through them.

AlexNet is the proof that if you scale a convolutional network up and actually train it end to end on raw RGB pixels, it crushes hand-designed pipelines. Top-5 error on ILSVRC-2012 went from **26.2%** (second place, Fisher Vector based) to **15.3%**. On ILSVRC-2010 the jump was **28.2% → 17.0%** top-5, and **47.1% → 37.5%** top-1. That is not an increment. That is a different regime.

The paper has no single new mathematical idea. Its contribution is an *engineering recipe* that made a deep net trainable at that size:

1. **ReLU** instead of `tanh` — trains ~6× faster to the same training error.
2. **Two GPUs**, model split across them, because 3 GB of memory was not enough.
3. **Dropout** in the big fully-connected layers, because 60M parameters overfit 1.2M images badly.
4. **Data augmentation** that is free (done on CPU while the GPU eats the previous batch).

Each of these is a small trick. Together they turn "impossible to train" into "five days on two gaming cards".

> [!NOTE] Convolutional layer
> A layer that slides a small filter (e.g. $11\times11\times3$) across the image and computes a dot product at each position. The same weights are reused at every location, so you get far fewer parameters than a fully-connected layer, and you bake in two assumptions about images: nearby pixels are related (locality), and a pattern means the same thing wherever it appears (translation stationarity). ^convolutional-layer

> [!NOTE] ReLU
> $f(x)=\max(0,x)$. Called *non-saturating* because its [[Derivative|derivative]] is 1 for all positive input, so the [[Backpropagation|backprop]] signal does not shrink. `tanh` and the sigmoid flatten out at both ends, their derivative goes to ~0, and learning stalls. ^relu

## The Methodology

**Data.** ILSVRC-2010/2012: 1.2M training images, 50k validation, 150k test, 1000 classes. Variable-resolution photos are rescaled so the short side is 256, then centre-cropped to $256\times256$. The only other preprocessing is subtracting the per-pixel mean over the training set. No SIFT, no edges, no hand-features — raw centred RGB goes in.

**Architecture.** Eight learned layers: five conv, three fully-connected, ending in a 1000-way softmax. Input is a $224\times224\times3$ crop.

| Layer | Kernels | Kernel size | Notes |
|---|---|---|---|
| conv1 | 96 | $11\times11\times3$, stride 4 | + LRN + max-pool |
| conv2 | 256 | $5\times5\times48$ | + LRN + max-pool |
| conv3 | 384 | $3\times3\times256$ | connects across both GPUs |
| conv4 | 384 | $3\times3\times192$ | same-GPU only |
| conv5 | 256 | $3\times3\times192$ | + max-pool |
| fc6 | 4096 | — | dropout |
| fc7 | 4096 | — | dropout |
| fc8 | 1000 | — | softmax |

Neuron counts per layer: 253,440 – 186,624 – 64,896 – 64,896 – 43,264 – 4096 – 4096 – 1000. Total 60M parameters, ~650k neurons. Note the kernel depths of 48 and 192 — those are *half* of 96 and 384, because the model is literally cut in two across the GPUs.

**The two-GPU split.** Half the kernels live on each GTX 580. Layers 2, 4, 5 only read from kernel maps on their own GPU; layer 3 reads from both. This is a hyperparameter you cross-validate: you tune how much cross-GPU chatter you can afford relative to compute. A nice side effect — GPU 1 spontaneously learned colour-agnostic edge/orientation filters, GPU 2 learned colour blobs, every single run.

**Objective.** Multinomial logistic regression, i.e. maximise the average log-probability of the correct label. That is plain [[Cross Entropy|cross-entropy]] with a softmax.

**Local Response Normalization (LRN).** After the ReLU in conv1 and conv2, divide each activation by a function of its neighbours *across kernels* at the same spatial position:

$$b^i_{x,y} = a^i_{x,y}\Big/\Big(k + \alpha \sum_{j=\max(0,\,i-n/2)}^{\min(N-1,\,i+n/2)} (a^j_{x,y})^2\Big)^{\beta}$$

with $k=2$, $n=5$, $\alpha=10^{-4}$, $\beta=0.75$. It is "brightness normalisation" — no mean subtraction. The idea is lateral inhibition: kernels that fire hard suppress their neighbours, creating competition. (This trick did not survive history; BatchNorm replaced it.)

**Overlapping max-pooling.** Pooling grid spaced $s=2$ pixels apart, each unit summarising a $z\times z = 3\times3$ window. Since $s < z$, windows overlap. Traditional pooling uses $s=z$.

**Dropout.** In fc6 and fc7, each hidden unit's output is zeroed with probability 0.5 during training. Dropped units contribute nothing to the forward pass or to backprop, so each minibatch effectively trains a different random sub-network that shares weights with all the others. At test time all units are on and outputs are multiplied by 0.5 — an approximation to the geometric mean over the exponentially many sub-networks. This kills *co-adaptation*: a neuron cannot rely on a specific partner being present, so it must learn a feature that is useful alongside many random subsets.

**Data augmentation** (two kinds, both computationally free — generated in Python on CPU while the GPU trains on the previous batch):

1. Random $224\times224$ crops from the $256\times256$ image, plus horizontal flips. Nominally multiplies the dataset by 2048 (though the samples are heavily correlated). At test time: 5 crops (4 corners + centre) × 2 flips = 10 patches, softmax averaged.
2. **PCA colour jitter.** Run PCA on all RGB pixel values in the training set. To each image add $[\mathbf{p}_1,\mathbf{p}_2,\mathbf{p}_3][\alpha_1\lambda_1, \alpha_2\lambda_2, \alpha_3\lambda_3]^T$ where $\mathbf{p}_i, \lambda_i$ are the eigenvectors/eigenvalues of the $3\times3$ RGB covariance matrix and $\alpha_i \sim \mathcal{N}(0, 0.1^2)$, drawn once per image per epoch. This simulates changes in illumination colour and intensity, which should not change what the object *is*.

**Optimisation.** SGD, batch size 128, [[Momentum|momentum]] 0.9, weight decay 0.0005:

$$v_{i+1} := 0.9 \cdot v_i - 0.0005 \cdot \epsilon \cdot w_i - \epsilon \left\langle \frac{\partial L}{\partial w}\Big|_{w_i} \right\rangle_{D_i}$$
$$w_{i+1} := w_i + v_{i+1}$$

Weights initialised from $\mathcal{N}(0, 0.01^2)$. Biases initialised to **1** in conv2, conv4, conv5 and the fc hidden layers, to 0 elsewhere — this feeds ReLUs positive input early so they are not dead at the start. Learning rate starts at 0.01, same for all layers, divided by 10 by hand whenever validation error plateaued; three drops total. ~90 epochs, 5–6 days on two GTX 580 3GB cards.

## Ablation Studies and Experiments

**Headline results.**

| Dataset | Method | Top-1 | Top-5 |
|---|---|---|---|
| ILSVRC-2010 | Sparse coding (2010 winner) | 47.1% | 28.2% |
| ILSVRC-2010 | SIFT + Fisher Vectors | 45.7% | 25.7% |
| ILSVRC-2010 | **AlexNet** | **37.5%** | **17.0%** |
| ILSVRC-2012 | SIFT + FVs (2nd place) | — | 26.2% (test) |
| ILSVRC-2012 | 1 CNN | 40.7% | 18.2% (val) |
| ILSVRC-2012 | 5 CNNs ensembled | 38.1% | 16.4% |
| ILSVRC-2012 | 1 CNN, pre-trained on full 15M-image ImageNet, fine-tuned | 39.0% | 16.6% |
| ILSVRC-2012 | **7 CNNs (5 + 2 pre-trained)** | **36.7%** | **15.3% (test)** |

Also ImageNet Fall 2009 (10,184 classes, 8.9M images, half train / half test): 67.4% / 40.9% versus the previous best of 78.1% / 60.9%.

**Component ablations** (each measured as the error increase when removed):

| Component removed | Δ Top-1 | Δ Top-5 |
|---|---|---|
| Two-GPU (vs. half-width single-GPU net) | +1.7% | +1.2% |
| Local Response Normalization | +1.4% | +1.2% |
| Overlapping pooling ($3\times3/2$ → $2\times2/2$) | +0.4% | +0.3% |
| PCA colour jitter | +1.0% | — |
| Ten-patch test-time averaging | +1.5% | +1.3% |

**Depth is doing the work.** Remove *any single* convolutional layer and top-1 drops by about 2% — despite each conv layer holding under 1% of the model's parameters. So the gain is not from parameter count, it is from composition of layers. This is the sentence in the paper that shaped the next decade.

**What was needed but easy to miss:**
- Weight decay of 0.0005 was *not* just a regulariser — it reduced **training** error. Without it the model learned worse, not just generalised worse.
- Without random-crop augmentation the net "suffers substantial overfitting" and they would have been forced to use a much smaller network.
- Without dropout, "substantial overfitting". Cost: dropout roughly doubles iterations to convergence.

**What did not work / was left on the table:**
- No unsupervised pre-training at all, though they expected it would help. (History says it did not, once labels were plentiful.)
- The one-GPU baseline comparison is admitted to be *biased against* the two-GPU net: they did not halve the final conv layer or the fc layers in the baseline, so the "half-size" net actually has more parameters than half. The real gain from splitting is therefore larger than 1.7%.
- ReLU speed evidence is on CIFAR-10 with a 4-layer net, not on ImageNet — 6× faster to 25% training error, no regularisation, learning rates tuned separately per net. The magnitude varies with architecture.

**Qualitative probes.** Take the 4096-dim fc7 activation as an image embedding and find nearest neighbours by Euclidean distance. Retrieved images share *semantics*, not pixels — dogs and elephants in wildly different poses come back together, and they are not close in raw L2. This is the first clear demonstration that a supervised classifier's penultimate layer is a general-purpose visual embedding, which is the whole basis of transfer learning in vision.

## Worth Remembering

- **The recipe survived; the details did not.** ReLU, dropout, augmentation, SGD+momentum, GPU training — all still standard. LRN was quietly dropped by everyone after BatchNorm. The two-GPU split was a memory hack, not a principle; VGG and ResNet went back to a single dense stack. Grouped convolutions (ResNeXt, MobileNet) later rediscovered the split as a deliberate design.
- **Model parallelism for the wrong reason.** They split across GPUs because 3 GB was too small. Modern model parallelism is for the same reason at a bigger scale.
- **The whole thing was compute-bound, and they knew it.** "Our results can be improved simply by waiting for faster GPUs and bigger datasets." That prediction is the informal ancestor of [[Scaling Laws for Neural Language Models]] and [[Training Compute-Optimal Large Language Models (Chinchilla)]].
- **Dropout as cheap ensembling.** Averaging many models works but is too expensive for a 6-day net. Dropout gets you the ensemble for about 2× training cost. It is a form of [[Regularization|regularisation]] that acts by breaking co-adaptation, not by shrinking weights.
- **Bias init to 1 for ReLU layers.** A small, forgotten detail — but if you initialise biases at 0 with symmetric weight init, a chunk of your ReLUs start dead and never recover. Modern practice fixes this with better weight init (He init) instead.
- **Test-time augmentation was worth 1.3% top-5.** Ten crops averaged. Cheap, and people still forget it.
- **Manual learning rate schedule.** Divide by 10 when validation error stops improving, three times. No cosine schedule, no warmup. The "step decay on plateau" heuristic starts here.
- **Practical caveat:** the "60M parameters" is dominated by fc6 (which takes the flattened conv5 output into 4096 units). That is where all the overfitting risk lives, and that is exactly where dropout is applied. Later architectures replaced this with global average pooling and got the same accuracy with 10× fewer parameters.
- **Open question the authors flag:** they wanted to move to video, where temporal structure gives extra signal. Took several more years.

## Links
Related: [[Deep Learning]] · [[Backpropagation]] · [[Momentum]] · [[Regularization]] · [[Cross Entropy]] · [[Derivative]] · [[Distillation]] · [[Saliency]] · [[Scaling Laws for Neural Language Models]] · [[Foundation Models]] · [[Mixed Precision training]] · [[Linear Projection]]

New topics worth writing: Convolutional neural networks, Dropout, ReLU and activation saturation, Batch Normalization, Local Response Normalization, Max pooling, Data augmentation, Weight decay vs L2 regularization, Transfer learning in vision, ImageNet and ILSVRC, He initialization, Global average pooling, Model parallelism, Test-time augmentation
