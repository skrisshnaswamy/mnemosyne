---
title: "Matryoshka Representation Learning"
authors: ["Aditya Kusupati", "Gantavya Bhatt", "Aniket Rege", "Matthew Wallingford", "Aditya Sinha", "Vivek Ramanujan", "William Howard-Snyder", "Kaifeng Chen", "Sham Kakade", "Prateek Jain", "Ali Farhadi"]
year: 2022
arxiv: "2205.13147"
url: https://arxiv.org/abs/2205.13147
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, llm, optimization, self-supervised, vision]
---
## The Core Idea

A trained network gives you one embedding of a fixed size — say 2048 numbers for ResNet50. That size is a guess made at training time. Some downstream jobs (a cheap first-pass search over a billion items) would be happy with 16 numbers. Some (fine-grained re-ranking) want all 2048. Today you either train many models of different widths, or you compress after the fact with something like PCA, and both are bad: many models means many forward passes and many copies of your database; PCA loses a lot of accuracy.

Matryoshka Representation Learning (MRL) makes **one** embedding where the first $m$ numbers are already a good standalone embedding, for many choices of $m$. Take the first 8 dimensions and throw the rest away — you get roughly the accuracy of a model that was trained from scratch to output 8 dimensions. Take the first 512 and you get the 512-dim model. Same vector, same forward pass, no extra cost at inference. Like Russian nesting dolls.

> [!NOTE] Matryoshka Representation ^matryoshka-representation
> A $d$-dimensional vector $z$ such that for every $m$ in a chosen set $\mathcal{M}$, the truncation $z_{1:m}$ is itself a useful, transferable representation. Truncation is free — it is a slice, not a projection.

Why did this not exist? Gradient descent has no reason to concentrate information in the early coordinates. Left alone, a network smears information evenly across all $d$ dimensions, so the first 8 of 2048 are junk (Table 2: 1-NN accuracy of the first 8 dims of a normal ResNet50 is **2.36%**, versus **62.19%** for MRL). The trick is embarrassingly simple: put a loss on each prefix during training.

What it unlocks is **adaptive deployment**. You store one database. At query time you decide how many dimensions to read, per query, based on how hard that query is. On ImageNet-1K this gives the same retrieval quality as full 2048-dim search at $128\times$ fewer FLOPs and $14\times$ less wall-clock time.

## The Methodology

The whole method is one equation. Let $F(\cdot\,;\theta_F)$ be your backbone producing $z \in \mathbb{R}^d$. Pick a set of nesting sizes $\mathcal{M}$, roughly $\log d$ of them by repeated halving. For ResNet50, $d = 2048$ and

$$\mathcal{M} = \{8, 16, 32, 64, 128, 256, 512, 1024, 2048\}.$$

For ViT-B/16 and BERT, $d = 768$ and $\mathcal{M} = \{12, 24, 48, 96, 192, 384, 768\}$.

Attach a separate linear classifier $\mathbf{W}^{(m)} \in \mathbb{R}^{L\times m}$ to each prefix and sum the losses:

$$\min_{\{\mathbf{W}^{(m)}\},\,\theta_F} \frac{1}{N}\sum_{i}\sum_{m\in\mathcal{M}} c_m \cdot \mathcal{L}\!\left(\mathbf{W}^{(m)} \cdot F(x_i;\theta_F)_{1:m}\,;\,y_i\right)$$

with $\mathcal{L}$ the usual softmax [[Cross Entropy]] loss and all importance weights $c_m = 1$. That is it. Ordinary [[Backpropagation]], ordinary SGD, no new hyperparameters, no architecture change beyond swapping the final `fc` layer for a list of `nn.Linear` heads.

**MRL–E (efficient variant).** Instead of $|\mathcal{M}|$ separate heads, tie the weights: $\mathbf{W}^{(m)} = \mathbf{W}_{1:m}$, one shared $\mathbf{W}\in\mathbb{R}^{L\times d}$ sliced column-wise. Halves the classifier memory, which matters when $L$ is in the millions. Costs about 1% top-1 accuracy at small $m$. For masked language modelling MRL *reduces* to MRL–E automatically, because BERT ties the input embedding matrix to the output classifier.

**Adaptations.** For contrastive setups (ALIGN, CLIP-style), apply the nesting to *both* sides of the pair. If the pipeline L2-normalises the embedding, normalise each prefix independently — the norm of $z_{1:m}$ is not the norm of $z$.

**Training details.** ResNet50 on ImageNet-1K via the FFCV pipeline, 40 epochs, cyclic LR peaking at 0.475, batch 256/GPU on 2×A100, SGD with [[Momentum]] 0.9 and weight decay 1e-4. **No hyperparameter search** — they reused the baseline's settings exactly. ViT-B/16 on JFT-300M: 300K steps, batch 128, Adafactor, 8×8 TPU pod. ALIGN: 1M steps. BERT-Base on Wikipedia+BooksCorpus: 450K steps, [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], LR 1e-4.

### The two deployment tricks

**Adaptive Classification (MRL–AC).** A cascade over prefixes. Start at $m=8$, look at the max softmax probability $p_m$; if $p_m \geq t_m^*$ accept the prediction, else move to $m=16$, and so on. Thresholds $t_m^*$ are found by grid search (100 points in $[0,1]$) on a 10K-image holdout, picking the smallest threshold that maximises accuracy. Crucially this needs **no extra forward passes** — all prefixes come from one pass, unlike normal model cascades.

**Adaptive Retrieval (AR).** Two stages. Shortlist $K=200$ neighbours using a cheap prefix $D_s$ (e.g. 16 dims). Re-rank that shortlist with a big prefix $D_r$ (e.g. 2048). Shortlisting is the expensive part — it scales $O(dN)$ exact, or $O(d\log N)$ with HNSW. Re-ranking 200 items at 2048 dims is 400 KFLOPs, i.e. free.

**Funnel Retrieval.** A cascade version of AR that removes the need to hand-pick $D_s, D_r$. At each step, double the dimension and halve the shortlist. On ImageNet-1K: shortlist $200 \to 100 \to 50 \to 25 \to 10$ paired with dims $16 \to 32 \to 64 \to 128 \to 256 \to 2048$.

## Ablation Studies and Experiments

**Baselines, all on ResNet50 / ImageNet-1K:** FF-$k$ (independently trained fixed-feature model of width $k$ — nine separate trainings), SVD on the FF-2048 classifier, random feature selection with a linear probe, JL random projection, and [[An Image is Worth 16x16 Words (ViT)|slimmable networks]] (a sub-network method).

**Linear probe top-1 (Table 1):**

| dims | Rand. LP | SVD | Slimmable | FF | MRL | MRL–E |
|---|---|---|---|---|---|---|
| 8 | 4.56 | 2.34 | 0.42 | 65.29 | **66.63** | 56.66 |
| 16 | 11.29 | 7.17 | 0.96 | 72.85 | **73.53** | 71.94 |
| 128 | 65.70 | 67.24 | 14.15 | 75.29 | **76.30** | 75.80 |
| 2048 | 76.87 | — | 76.26 | 76.87 | 76.80 | 76.51 |

MRL matches or beats nine independently trained models with one training run. Post-hoc compression and slimmable nets collapse below 256 dims — slimmable nets are catastrophic (0.42% at 8 dims) because they were never trained below 25% width and become unstable there.

**1-NN accuracy (Table 2)**, which is the honest test of representation quality since no new classifier is fit: MRL beats FF by ~3.3 points at 8 dims (62.19 vs 58.93) and is level from 128 dims up.

**The interpolation result — this is the most surprising bit.** MRL only optimises $|\mathcal{M}| \approx \log d$ prefixes, but accuracy at *unoptimised* intermediate sizes (12, 24, 48, 96, 192, 384, 768, 1536) lands smoothly on the curve. On ALIGN, 1-NN top-1 at the never-trained 128 dims is 66.63%, sitting neatly between the trained 96 (65.72) and 192 (67.00). This is what makes the method practical at web scale — you can slice anywhere.

**Adaptive classification.** MRL–AC hits **76.30%** top-1 with an *expected* dimensionality of ~37, versus 512 dims for the FF baseline at the same accuracy — a **14× smaller** representation. Even the pessimistic accounting (summing cumulative dims $8+16+32+\dots$ because each head is separate) gives expected size 62, still 8.2× smaller.

**Adaptive retrieval (Figure 8, Tables 11–12).** On ImageNet-1K, $D_s=16 \to D_r=2048$ gives mAP@10 of 65.27 versus 65.20 for full single-shot 2048-dim retrieval — **128× fewer FLOPs, 14× faster wall-clock** (HNSW32, same hardware). Every $(D_s, D_r)$ combination lies above the Pareto frontier of single-shot retrieval. On the harder ImageNet-4K (4.2M database, 4202 classes) you need $D_s=64$, giving 32× theoretical / 6× real speedup.

**Cross-modal / scale.** ALIGN-MRL beats plain ALIGN at almost every dimension, and hugely at small ones: 12-dim k-NN top-1 goes **11.90 → 43.57**. JFT-ViT at 12 dims: **27.07 → 53.61**. BERT-MRL MLM validation accuracy stays within 0.5% of BERT-FF at every size, with zero hyperparameter tuning.

### What did not work

- **Uniform spacing of $\mathcal{M}$.** Training with $\{8, 212, 416, \dots, 2048\}$ instead of log spacing hurt the low end badly: 1-NN at 8 dims drops **62.19 → 58.44**, at 16 dims **67.91 → 61.11**. Accuracy saturates logarithmically in dimension, so uniform granularities waste capacity on the top end where nothing is gained. Uniform also costs $O(L m^2)$ classifier memory versus $O(L \cdot 2\log d)$.
- **Going below 8 dimensions.** MRL-4 gives 27.25% top-1 at 4 dims — unusable — and slightly *drags down* the higher dims because the joint loss now includes a very hard sub-problem. 8 is an empirical sweet spot.
- **Fine-tuning only the last layer** to retrofit nesting onto a pretrained FF-2048 model. Just the `fc` layer gives 5.15% at 8 dims. You have to unfreeze non-linear conv layers: unfreezing through `4.2 conv2, conv3, fc` gets 8 dims to 54.78%, and unfreezing all of block 4.2 gets 60.02% — still 6.6 points short of end-to-end MRL (66.63%). Retrofitting works, but only partially, and needs real non-linearity in the tuned portion.
- **MRL–E at very low dims.** 56.66% at 8 dims versus MRL's 66.63%. Weight tying is a real cost at the bottom of the funnel.
- **Loss reweighting.** Setting $c_8 = 2$ (MRL-8boost) raises 8-dim top-1 by ~3 points (66.63 → 69.53) and *also* helps 16–256 dims, at a cost of ≤0.1% at 512–2048. The authors flag optimal $c_m$ as unresolved future work — they shipped $c_m = 1$ everywhere.
- **Shortlist length saturates.** On ImageNet-1K, mAP@10 peaks at $K=200$ and stops improving. On ImageNet-4K it keeps improving to $K=2048$ (FAISS's max), because the database is bigger and slightly out-of-distribution.

## Worth Remembering

**Model disagreement across dimensions is real and exploitable.** Only 244 of 1000 ImageNet classes improve monotonically with dimension; 49 classes get *worse* before getting better. If you had a perfect oracle routing each image to its best prefix, top-1 would be **81.5%** instead of 76.9% — a 4.6 point headroom. 18.46% of the validation set is wrong at *every* dimension. The threshold cascade recovers only a fraction of that gap; learning a better router is open.

**Low dimensions encode coarse semantics, not noise.** On a 31-way superclass version of ImageNet, 8 dimensions already gets 85.57% versus 90.21% at 2048 — a 4.6 point gap, versus a ~10 point gap on the 1000-way task. Tight bottlenecks capture the WordNet hierarchy. Grad-CAM shows the 8-dim model failing gracefully: it confuses a snake for the wrong snake species, or gets distracted by other objects in a cluttered scene. This is a nice diagnostic tool in its own right.

**Robustness is preserved, sometimes improved.** On ImageNet-V2/R/A/Sketch, MRL matches FF and gains 0.6% on ImageNet-A (a 20% relative gain on that hard set). MRL retrieval on ImageNetV2 beats FF at *every* dimension by up to 3 mAP@10.

**Few-shot: fewer shots need fewer dimensions.** On 1000-way ImageNetV2, 1-shot accuracy at 32 dims (41.40%) equals 2048 dims (41.16%). On the FLUID long-tail benchmark, MRL beats FF by ~2% on novel-tail classes; the gap between 64 and 2048 dims is 1% on pretrain-head classes but 6.7 points on novel-tail. Hard, data-poor tasks are the ones that actually need capacity.

**Practical caveats.** Storage overhead is 8MB for MRL (the extra heads), zero for MRL–E. The 14× wall-clock number is HNSW32 on a 24-core Xeon; the HNSW index itself is ~10.6GB for 2048-dim ImageNet-1K versus 10.2GB exact, so ANN indices are a memory tax on top. All of this composes with product quantization and other vector-compression tricks — MRL is orthogonal to them, not a replacement. No error bars anywhere; the authors say the experiments were too expensive to repeat.

**Connections.** The closest prior work is nested dropout (Rippel et al. 2014), which enforces ordering over $O(d)$ dimensions in autoencoders; MRL only optimises $O(\log d)$ and relies on interpolation to fill in the rest, which is what makes it affordable at web scale. This is now the standard trick behind OpenAI's `text-embedding-3` shortening and Nomic/Gemini variable-dimension embeddings.

## Links

Related: [[Deep Residual Learning for Image Recognition (ResNet)|ResNet]] · [[An Image is Worth 16x16 Words (ViT)]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Cross Entropy]] · [[Distillation]] · [[Distilling the Knowledge in a Neural Network]] · [[Whitening Sentence Representations]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Sentence-BERT]] · [[Embedding-based Retrieval in Facebook Search]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)]] · [[A Simple Framework for Contrastive Learning (SimCLR)]] · [[Representation Degeneration Problem in Training NLMs]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Momentum]] · [[Backpropagation]] · [[Recommender Systems - Evolution]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Scaling Laws for Neural Language Models]]

New topics worth writing: HNSW and approximate nearest neighbour search, FAISS, product quantization, nested dropout and ordered representations, slimmable neural networks and once-for-all training, model cascades and early exit, ALIGN, Grad-CAM, mAP@k as a retrieval metric, intrinsic dimensionality of representations, maximum softmax probability as a confidence signal
