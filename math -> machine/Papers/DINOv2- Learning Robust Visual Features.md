---
title: "DINOv2: Learning Robust Visual Features"
authors: ["Maxime Oquab", "Timothée Darcet", "Théo Moutakanni", "Huy Vo", "Marc Szafraniec", "Vasil Khalidov", "Pierre Fernandez", "Daniel Haziza", "Francisco Massa", "Alaaeldin El-Nouby", "Mahmoud Assran", "Nicolas Ballas", "Wojciech Galuba", "Russell Howes", "Po-Yao Huang", "Shang-Wen Li", "Ishan Misra", "Michael Rabbat", "Vasu Sharma", "Gabriel Synnaeve"]
year: 2023
arxiv: "2304.07193"
url: https://arxiv.org/abs/2304.07193
priority: Low-Priority
read_on: 2026-09-30
tags: [paper, llm, self-supervised, vision, theory]
---
## The Core Idea

Before this paper, if you wanted one image model whose features worked everywhere without extra training, you used CLIP-style text supervision. Pair each image with its caption, learn to match them. That works, but the caption is a bottleneck: it says "a dog on grass", and everything the words leave out — exact boundaries, depth, which patch belongs to which object part — is not in the training signal.

DINOv2 shows you can drop text entirely. Train only on images, self-supervised, and get frozen features that beat OpenCLIP on most benchmarks at both the image level (classification) and the pixel level (segmentation, depth).

Why this did not exist before: [[A Simple Framework for Contrastive Learning (SimCLR)|self-supervised]] image methods were almost all developed and tuned on ImageNet-1k, a small clean dataset. The few attempts to scale past it used *uncurated* web images and the feature quality dropped. So the field's belief was "self-supervision needs curation, and curation needs humans, so it cannot scale."

Two things unlock it:

1. **Automatic curation.** Build a 142M-image curated dataset with no labels, no captions, no metadata — just visual similarity retrieval against a pool of seed datasets. Copied from how text corpora are filtered, but with image embeddings instead of a language model.
2. **An engineering rewrite** that makes [[Self-Supervised Learning from Images with I-JEPA|joint-embedding]]-style training 2× faster on a third of the memory, which is what makes a 1.1B-parameter ViT trainable at all.

> [!NOTE] Frozen features
> Features you extract once from a network whose weights you never touch again. All downstream adaptation happens in a small head — usually a single linear layer — trained on top. The claim "our features are good" means "a linear layer on top of them is enough". ^frozen-features

The headline claim to keep: a ViT-g/14 hits **86.5%** linear-probe top-1 on ImageNet-1k, vs 86.2% for OpenCLIP ViT-G/14 and 82.3% for the previous best self-supervised model (iBOT ViT-L/16). Fine-tuning the backbone only adds +2.0%, so **fine-tuning becomes optional**. That is the actual product.

## The Methodology

### Building LVD-142M without labels

Start with 1.2B unique images scraped from public web crawl data (URLs pulled from `<img>` tags, NSFW filtered, faces blurred).

Then:

- **Self-deduplicate.** Embed every image with a copy-detection network (Pizzi et al. 2022), find 64 nearest neighbours by cosine similarity, build a k-NN graph keeping edges with similarity > 0.6, take connected components, keep one image per component. 1.2B → 1.1B.
- **Deduplicate against every test set** used in the paper, this time with a stricter threshold > 0.45, discarding the whole duplicate component. 1.1B → 744M. This step is why the results are believable.
- **Retrieve.** Embed with a self-supervised ViT-H/16 pretrained on ImageNet-22k. For each "seed" curated dataset (ImageNet-22k, ImageNet-1k train, Google Landmarks, plus ~20 fine-grained/segmentation/depth sets), pull close images out of the 744M pool.

Two retrieval modes, chosen by seed size:

- **Sample-based** (seed > 1M images): take $N$ nearest neighbours per seed image. $N=4$ for Google Landmarks and ImageNet-22k, $N=32$ for one big ImageNet-22k pass. Larger $N$ looked fine to the eye but caused more *collisions* — the same pool image being the neighbour of many queries — so $N=4$ is the compromise.
- **Cluster-based** (small seeds like DTD's 1,880 images): k-means the whole uncurated pool into 100,000 clusters, then for each cluster containing >3 seed images, take 10,000 pool images from it. Cap any single dataset's contribution at 1M images so nothing dominates.

Total: 142,109,386 images. The whole pipeline runs on 20 nodes × 8 V100s in under two days, using [[Efficient and robust approximate nearest neighbor search using HNSW|approximate nearest neighbour]] search via Faiss with [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)|product-quantized]] inverted indices.

The rebalancing point is the subtle one. Web images are wildly imbalanced — a naive sample is mostly stock photos and selfies. Clustering first, then sampling per cluster, flattens that.

### The training objective

A student ViT and a teacher ViT. The teacher is not a separate model: it is an [[Momentum|exponential moving average]] of the student weights, momentum on a cosine schedule from 0.994 → 1.0. Total loss is a sum of two [[Cross Entropy|cross-entropy]] terms plus a regulariser.

**Term 1 — DINO loss (image level).** Take two large crops (224×224) and several small crops (98×98) of the same image. Each network's `[CLS]` token goes through an MLP "head" producing $K$ = 128,000 prototype scores. Student scores get a softmax → $p_s$. Teacher scores get a softmax then a centering step → $p_t$. Then

$$\mathcal{L}_{DINO} = -\sum p_t \log p_s$$

Plain words: make the student's guess about "which of 128k abstract categories is this crop" match the teacher's guess about a *different* crop of the same image.

**Term 2 — iBOT loss (patch level).** Mask some input patches for the student only. The teacher sees everything. Push the student's prediction at each masked position toward the teacher's output at that same position:

$$\mathcal{L}_{iBOT} = -\sum_i p_{ti} \log p_{si}$$

where $i$ indexes masked patches. This is [[BERT- Pre-training of Deep Bidirectional Transformers|masked language modelling]] moved to pixels, except the targets are the teacher's live features, not a fixed vocabulary.

**Term 3 — KoLeo regulariser.** From the Kozachenko–Leonenko entropy estimator. $\ell_2$-normalise features, then for each of $n$ vectors in the batch let $d_{n,i} = \min_{j\neq i}\|x_i - x_j\|$ be the distance to its nearest neighbour in the batch:

$$\mathcal{L}_{\mathrm{koleo}} = -\frac{1}{n}\sum_{i=1}^{n}\log(d_{n,i})$$

Minimising this maximises the log of nearest-neighbour distances, i.e. it pushes every feature away from its closest neighbour and spreads the batch out over the sphere. Weight 0.1, applied to the first global crop's class tokens, computed per-GPU without cross-GPU communication. This is the anti-[[Understanding Dimensional Collapse in Contrastive Learning|collapse]] term, and it is the same job the uniformity term does in [[Understanding Contrastive Learning through Alignment and Uniformity|contrastive losses]].

**Teacher centering.** Instead of DINO's moving-average centering, use SwAV's Sinkhorn–Knopp batch normalisation — 3 iterations of alternately normalising rows and columns of the score matrix so it becomes roughly doubly stochastic. Student still just gets a softmax.

**Untied heads.** iBOT's own ablation said sharing the DINO and iBOT projection heads was better. At this scale the opposite holds, so two separate heads.

**Resolution ramp.** Train at 224×224 the whole way, then run the last 10k iterations at 518×518 with all schedules compressed and a lower base learning rate.

### Making it trainable

This is a third of the paper and the part most worth stealing.

- **Custom [[FlashAttention- Fast and Memory-Efficient Exact Attention|FlashAttention]].** The GPUs want head dim a multiple of 64, and total embedding dim a multiple of 256. So ViT-g is reshaped from Zhai et al.'s 1408-dim/16-head (88 per head) to **1536-dim/24-head (64 per head)**, 40 blocks, 1.1B params. No measurable accuracy change, meaningfully faster.
- **Sequence packing.** The 224 crops and the 98 crops become token sequences of different lengths, so they cannot batch together. Fix: concatenate them into one long sequence and apply a block-diagonal [[Causal Attention|attention mask]] so no crop can attend to another. Mathematically identical to separate forward passes, one kernel launch instead of many. Borrowed from NLP (Krell et al. 2022).
- **Efficient stochastic depth.** Normal stochastic depth computes the residual branch and then multiplies by zero. Here, shuffle the $B$ samples along the batch dim and *slice* the first $(1-d)B$ — so the dropped samples are never computed. At $d = 40\%$ that is a real ~40% saving in that block. Fused kernels, shipped in xFormers.
- **[[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models|FSDP]].** [[Decoupled Weight Decay Regularization (AdamW)|AdamW]] needs four fp32 replicas — student, teacher, first moment, second moment — so 16 GB for a 1B model. Shard all of it across GPUs. The second win is communication: weights are stored in fp32 as the optimiser requires, but broadcast and gradient-reduced in **fp16** for the backbone (fp32 for the MLP heads only, to avoid instability). That is ~50% less cross-GPU traffic than [[Distributed Training|DDP]]'s fp32 all-reduce.

Net: **2× faster, 1/3 the memory** vs the iBOT reference implementation, which is what buys batch size 3072 and 625k iterations.

### Distillation for the small models

ViT-S/B/L are *not* trained from scratch. They are [[Distilling the Knowledge in a Neural Network|distilled]] from the ViT-g using the same training loop with four changes: the teacher is the frozen ViT-g (no EMA on it), keep a separate EMA of the student and ship *that* as the final model, drop masking and stochastic depth, and apply the iBOT loss on both global crops. Loss terms themselves are unchanged.

Hyperparameters: 625k iterations everywhere, AdamW, LayerScale init 1e-5, weight decay cosine 0.04 → 0.2, 100k-iteration [[Accurate, Large Minibatch SGD- Training ImageNet in 1 Hour|LR warmup]], fp16. From-scratch models: drop rate 0.4, LR 3.5e-4, batch 3072, SwiGLU FFN. Distilled models: drop rate 0, LR 1e-3, batch 2048, plain MLP FFN.

## Ablation Studies and Experiments

### The recipe, added one piece at a time

ViT-L on ImageNet-22k. They optimise for **k-NN** accuracy, on the reasoning that linear-probe accuracy is lower-bounded by k-NN, so k-NN is the harder and more honest signal.

| Step | k-NN | linear |
|---|---|---|
| iBOT (reported) | 72.9 | 82.3 |
| their reproduction | 74.5 | 83.2 |
| + LayerScale, stochastic depth | 75.4 | **82.0** ↓1.2 |
| + 128k prototypes | 76.6 | 81.9 ↓0.1 |
| + KoLeo | **78.9** ↑2.3 | 82.5 |
| + SwiGLU FFN | 78.7 ↓0.2 | 83.1 |
| + patch size 14 | 78.9 | 83.5 |
| + teacher momentum 0.994 | 79.4 | 83.6 |
| + tweaked warmup | 80.5 ↑1.1 | 83.8 |
| + batch size 3072 | **81.7** ↑1.2 | **84.7** ↑0.9 |
| + Sinkhorn–Knopp | 81.7 = | 84.7 = |
| + untied heads (= DINOv2) | 82.0 | 84.5 ↓0.2 |

What this actually says:

- **LayerScale + 40% stochastic depth cost 1.2% linear accuracy.** They kept them anyway, because without them the loss goes NaN. That is a stability purchase, not a quality one, and everything after it depends on the run not dying.
- KoLeo (+2.3 k-NN) and batch size 3072 (+1.2 k-NN, +0.9 linear) are the two biggest single jumps.
- **Sinkhorn–Knopp changed nothing.** 81.7 → 81.7, 84.7 → 84.7. It is in the method and it is free, but it is not doing work here.
- Untied heads is +0.3 k-NN and −0.2 linear. Essentially noise, despite being presented as a correction of iBOT's finding.

### Which loss term earns its place

ViT-g, shorter-than-final training. Numbers are ImageNet-1k linear / ImageNet-A / ADE20k mIoU / Oxford-M mAP.

KoLeo off → on: 85.3→85.8, 70.6→72.8, 47.2→47.1, **55.6→63.9**.

So KoLeo is almost entirely a *retrieval* fix, worth +8.3 mAP on Oxford-M, and it costs nothing elsewhere. That is exactly what you would predict from a term whose whole job is spreading features apart so nearest-neighbour distances are meaningful.

MIM (the iBOT term) off → on: 85.3→85.8, 72.0→72.8, **44.2→47.1**, 64.3→63.9.

So MIM is almost entirely a *dense prediction* fix, +2.9 mIoU on segmentation, and it slightly *hurts* retrieval. The two extra terms are doing two different, clean, non-overlapping jobs.

### Data source

ViT-g, equal iterations, no high-res phase.

| Data | INet-1k | Im-A | ADE-20k | Oxford-M | iNat2018 | iNat2021 | Places205 |
|---|---|---|---|---|---|---|---|
| INet-22k | 85.9 | 73.5 | 46.6 | 62.5 | 81.1 | 85.6 | 67.0 |
| INet-22k ∖ INet-1k | 85.3 | 70.3 | 46.2 | 58.7 | 80.1 | 85.1 | 66.5 |
| Uncurated 142M | 83.3 | 59.4 | **48.5** | 54.3 | 68.0 | 76.4 | 67.2 |
| LVD-142M | 85.8 | **73.9** | 47.7 | **64.6** | **82.3** | **86.4** | **67.6** |

The uncurated row is the interesting one. It is catastrophic on fine-grained tasks (iNat2018 68.0 vs 82.3, a 14-point hole) and on retrieval — but it is the **best row on ADE-20k segmentation** (48.5). Raw diverse web images teach good local texture and layout; they do not teach categories. Curation buys semantics, not pixels.

LVD-142M beats ImageNet-22k on everything except ImageNet-1k itself, where it is 0.1 behind. And it wins on iNaturalist and Places205, which were *never used* as retrieval seeds — so the diversity generalises to unseen domains rather than just interpolating the seeds.

### Model size × data size

The crossover is the finding: at small model sizes ImageNet-22k and LVD-142M are comparable, and the gap opens as models grow. A ViT-g on LVD-142M matches a ViT-g on ImageNet-22k on ImageNet-1k while beating it everywhere else. More data only pays once the model is big enough to use it — the same shape as [[Scaling Laws for Neural Language Models|scaling law]] arguments, and consistent with [[Training Compute-Optimal Large Language Models (Chinchilla)|Chinchilla]]-style joint scaling.

### Distillation beats from-scratch, every time

ViT-L/14 on 12 benchmarks, distilled from ViT-g vs trained from scratch (teacher ViT-g in brackets):

| | INet-1k | Segm. | Depth ↓ | Classif. | Finegr. | Retriev. | ARSketch | Video |
|---|---|---|---|---|---|---|---|---|
| ViT-g scratch | 86.5 | 73.4 | 1.00 | 92.1 | 78.3 | 75.2 | 77.0 | 69.3 |
| ViT-L scratch | 84.5 | 72.2 | 1.10 | 90.2 | 75.8 | 71.3 | 69.5 | 67.3 |
| ViT-L **distilled** | 86.3 | 73.3 | 1.08 | 91.2 | 77.6 | **76.3** | 74.5 | 67.5 |

Distilled wins on all 12. Note retrieval: the distilled ViT-L (76.3) **beats its own teacher** (75.2). Student-exceeds-teacher, which normally signals the EMA-of-student trick acting as a regulariser rather than pure knowledge transfer.

### Resolution

ViT-L/16 on ImageNet-1k, three runs: fixed 224, fixed 416, and 224 then 416 for the last 10k iterations. Fixed-416 is best across eval resolutions but costs ~3× the compute of fixed-224. The 224→416 run is "almost as good" for a fraction of the cost. Hence the short high-res phase in the final recipe — a deliberate purchase of ~95% of the benefit for ~5% of the extra spend.

### Headline results

**ImageNet-1k linear probe.** ViT-g/14 86.5 val, 89.6 ReaL, 78.4 V2. OpenCLIP ViT-G/14: 86.2 / 89.4 / 77.2. EVA-CLIP ViT-g/14: 86.4 / 89.3 / 77.4. The gap on V2 (+1.1 over EVA-CLIP) is larger than on val, which they read as better generalisation — worth taking seriously given [[Do ImageNet Classifiers Generalize to ImageNet|how V2 was built]]. Beats iBOT ViT-L/16 by +4.2. k-NN: 83.5 vs iBOT's 72.9.

**Fine-tuning.** 86.5 → 88.5 at res 224; 86.7 → 88.9 at 448. Only +2.0/+2.2, using Touvron et al.'s pipeline with no hyperparameter tuning. Absolute SOTA at the time was 91.1.

**Robustness** (linear probe, no fine-tuning). vs iBOT: +29.6 on ImageNet-A (75.9 vs 41.5... actually 71.3 for ViT-L, 75.9 for ViT-g), +22.1 on ImageNet-R, +23.0 on Sketch. Beats OpenCLIP-G on ImageNet-A (75.9 vs 63.8) but **loses on R (78.8 vs 87.8) and Sketch (62.5 vs 66.4)**. Text supervision still helps on renditions and drawings — plausibly because captions are style-invariant by nature.

**Instance retrieval.** This is the blowout. Oxford-Hard mAP: DINOv2 ViT-L 54.0, OpenCLIP-G 19.7, iBOT 12.7. +34 over the weakly-supervised model, +41 over self-supervised. Captions simply do not contain "which specific building is this".

**Segmentation, frozen + linear.** ADE20k 49.0 mIoU linear, 53.0 with multiscale — which matches fully fine-tuned MAE with an UperNet decoder (53.6) using a single linear layer. Pascal VOC 86.2 vs SOTA 89.0. Plugging the frozen backbone into a ViT-Adapter + Mask2Former (66% of weights frozen) gives **60.2** on ADE20k vs SOTA 62.9, trained in 28 hours on 16 V100s.

**Depth.** NYUd RMSE 0.279 with a DPT head on the frozen backbone, vs the 0.330 reference SOTA. NYUd→SUN-RGBD zero-shot transfer 0.338 vs 0.421 reference. Note iBOT ViT-L (0.417 lin.1) beats OpenCLIP ViT-G (0.541 lin.1) on NYUd — the authors read this as caption supervision failing to encode geometry at all.

**Video, with no video training.** Kinetics-400 78.4 (OpenCLIP-G 78.3), UCF-101 91.2 (90.7), Something-Something v2 38.3 vs 35.8. SSv2 needs temporal understanding, so beating a text model there by 2.5 with 8 averaged frames is the surprise.

**Fine-grained, 12-dataset SimCLR suite.** Average 92.1 vs OpenCLIP-G's 91.9. Losses concentrated on SUN397 (−5.3) and Stanford Cars (−4.7) — both datasets where class names are literally text ("2012 BMW 3 Series Sedan"), so text supervision has an unfair advantage.

### What did not work

- Sinkhorn–Knopp: zero measured effect.
- Untied heads: within noise, and directly contradicts iBOT's ablation without a clean explanation beyond "at scale the opposite is true".
- Uncurated data: the whole reason the pipeline exists, and it fails hardest exactly where you would want a general model to be strong (fine-grained recognition).
- LayerScale + stochastic depth: a net *loss* on linear accuracy, kept only to prevent NaNs.
- Text-supervised models still win on ImageNet-R, Sketch, SUN397, Cars and Places205.

## Worth Remembering

**The one-line takeaway.** "Curation" turns out not to require humans. It requires a similarity metric and a clustering step. Everything else in the paper is engineering in service of that.

**The two extra loss terms are surgically separable.** KoLeo → retrieval (+8.3 mAP, nothing elsewhere). MIM → dense tasks (+2.9 mIoU, slightly negative for retrieval). If you are building on this and only care about one axis, you know which term to keep and which to drop. This is unusually clean for an ablation.

**Circular bootstrap, worth flagging.** The retrieval model that builds LVD-142M is itself a self-supervised ViT-H/16 pretrained on ImageNet-22k. So the dataset is shaped by ImageNet's notion of visual similarity. The authors show the result generalises to unseen domains (iNaturalist, Places205), which is reassuring, but the inductive bias of ImageNet is in the data pipeline, not just the seeds.

**Test-set deduplication at 0.45 similarity.** They discard the whole duplicate *component*, not just the matched image. This is the detail that makes the retrieval numbers trustworthy, and most papers building web-scale corpora do not do it. Copy the practice.

**The engineering is the reusable half.** Sequence packing with block-diagonal masks, compute-skipping stochastic depth, and fp16 comms under fp32-sharded FSDP are all portable to any multi-crop or variable-length training job. 2× speed / 1/3 memory is not a footnote — it is the difference between the experiment happening and not happening.

**Emergent parts, not trained for.** PCA on patch features: the first component thresholded at zero separates foreground from background, unsupervised. Components 2–4 correspond to object *parts*, and they match across different instances of the same category, and across style changes (photo vs drawing) and large pose changes. Patch matching across images links a plane's wing to a bird's wing. Nobody supervised any of this.

**Limitations the authors own.**
- Geographic bias is real and large. Dollar Street: 89.7% in Europe vs **74.0% in Africa** — a 15.7-point gap. Low income 67.4 vs high income 90.5 — a **31.7-point gap**. Better than SEERv2 on every bucket, but still badly skewed toward wealthy Western households.
- The gender/skintone/age audit finds no harmful-label pattern (Non-Human and Crime both ~0.0% across all groups), but "Possibly-Human" fires far more for men (52.2/48.1) than women (15.8/17.2), driven by the `Beard` class. They explicitly say a deeper audit may find more.
- Fine-tuning was run with no hyperparameter search, so +2.0% is a floor, not a measurement.

**Carbon.** ViT-g pretraining = 22,016 A100-hours = 9.7 MWh = 3.7 tCO₂eq (US grid, PUE 1.1). Whole project 0.5k–1k tCO₂eq, ~200k GPU-days. Reproducing OpenCLIP ViT-G would be 118.9 MWh, ~10× more — though that trains a text encoder too, so it is not apples to apples. If you only need vision features, self-supervision is the cheaper carbon option.

**Practical caveats if you want to use this.**
- Patch size 14, so a 518×518 input gives a 37×37 token grid. Plan your dense heads around that, not around 16.
- The linear-probe protocol includes a grid search over LR (13 values), 1 vs 4 output layers, and whether to concatenate average-pooled patch tokens with `[CLS]`. Reported numbers are the best on the validation set. Cheap to run — one backbone forward feeds all classifiers — but it *is* a search, so compare fairly.
- Depth and segmentation heads benefit from concatenating 4 layers, not just the last. For ViT-g those are layers {10, 20, 30, 40}.
- The small models are distilled, not independently trained. If you retrain on your own data you need the ViT-g first, or you lose the distillation gain (ViT-L scratch is 1.8 points behind distilled on ImageNet-1k).

**Open questions.** Why Sinkhorn–Knopp does nothing here when it mattered in SwAV. Why untied heads flip sign with scale. Whether the uncurated-data win on ADE20k means a *mixture* of curated and uncurated would beat either — they never tried it. And the stated follow-up: feeding these frozen features into a language model as if they were tokens, which is roughly the path the field then took.

## Links

Related: [[Self-Supervised Learning from Images with I-JEPA]] · [[JEPA]] · [[Bootstrap Your Own Latent (BYOL)]] · [[A Simple Framework for Contrastive Learning (SimCLR)]] · [[Momentum Contrast (MoCo)]] · [[VICReg]] · [[Barlow Twins]] · [[Exploring Simple Siamese Representation Learning (SimSiam)]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Understanding Contrastive Learning through Alignment and Uniformity]] · [[Mode Collapse]] · [[An Image is Worth 16x16 Words (ViT)]] · [[CLIP]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Distilling the Knowledge in a Neural Network]] · [[Distillation]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models]] · [[Distributed Training]] · [[Mixed Precision Training]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Accurate, Large Minibatch SGD- Training ImageNet in 1 Hour]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[Foundation Models]] · [[Embeddings]] · [[Efficient and robust approximate nearest neighbor search using HNSW]] · [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)]] · [[Vector Database]] · [[Do ImageNet Classifiers Generalize to ImageNet]] · [[Shortcut Learning in Deep Neural Networks]] · [[Regularization]] · [[Fine-Tuning]] · [[Gated Activation]] · [[U-Net]] · [[The Bitter Lesson (essay)]]

New topics worth writing: automatic data curation pipelines, KoLeo regulariser and Kozachenko–Leonenko entropy estimation, Sinkhorn–Knopp normalisation, sequence packing with block-diagonal attention masks, stochastic depth, LayerScale, copy-detection embeddings (SSCD), monocular depth estimation evaluation protocols, DPT decoder, ViT-Adapter, Mask2Former, semantic segmentation mIoU, instance-level recognition benchmarks (Oxford/Paris/Met/AmsterTime), geographic fairness evaluation (Dollar Street), carbon accounting for model training, masked autoencoders (MAE), SwAV, iBOT, FlexiViT and variable-resolution training
