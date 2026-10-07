---
title: "SPK: Eliciting Structured Prior Knowledge for Interpretable Out-of-Distribution Detection in Real-Time Object Detection"
authors: ["Changshun Wu", "Weicheng He", "Xiaowei Huang", "Saddek Bensalem"]
year: 2026
arxiv: "2608.19080"
url: https://arxiv.org/abs/2608.19080
priority: Low-Priority
read_on: 2026-10-05
tags: [paper, transformers, llm, vision, theory]
---
## The Core Idea

An object detector trained on 20 classes will happily draw a box around a kangaroo and call it a horse, with 0.9 confidence. That is an **out-of-distribution (OoD) hallucination**: a confident box on something the model was never taught.

> [!NOTE] OoD hallucination (in detection) ^ood-hallucination
> A detection whose box and class look normal, but whose object belongs to no training category — or is not an object at all. The detector has no mechanism to say "I don't know", so it picks the nearest known label.

Everyone's fix so far works on the detector's raw feature vectors. You take the 896-dimensional region feature, and either build a cleverer score on top of it ([[Deep Learning|KNN]], Mahalanobis, energy) or fine-tune the detector so those features separate better.

This paper does something different. It asks: **what does the detector already know, implicitly, that decides whether a box is real or hallucinated?** Then it decodes that knowledge out into five numbers you can read.

The five numbers are:

$$\mathbf{z}^{\mathrm{SPK}} = [\,s^{\mathrm{id}},\; s^{\mathrm{prox}},\; s^{\mathrm{bg}},\; r^{\mathrm{geo}},\; d^{\mathrm{ctx}}\,] \in \mathbb{R}^5$$

- $s^{\mathrm{id}}$ — how strongly this box shows parts that belong to the predicted class (a horse's mane, muzzle, hoof).
- $s^{\mathrm{prox}}$ — how strongly it shows parts of *near-miss* classes (things that look like horses but aren't).
- $s^{\mathrm{bg}}$ — how strongly it shows pure texture, no object parts at all.
- $r^{\mathrm{geo}}$ — box area divided by image area.
- $d^{\mathrm{ctx}}$ — how unlike the training images this whole image is.

Five dimensions. Down from 896. And it is *better*: plug the same off-the-shelf OoD algorithm into these five numbers instead of the raw features and FPR95 drops from 48.20 to 19.64 (KNN, YOLO, PASCAL-VOC Near-OoD). FPR95 is the share of OoD things you let through while keeping 95% of real detections.

Why did this not exist before? Two reasons. First, you need part-level labels — "this 7×7 cell is wing, that one is beak" — and nobody annotates those by hand at scale. Open-vocabulary grounding (OWLv2) plus [[Attention|segmentation]] (SAM 2) now makes that automatic. Second, you need to know *which* wrong samples to show the model. The paper's answer: hallucinations are not random. They come from exactly two places — **proximal OoD** (looks like a known class) and **background-only** (texture that trips objectness). So you mine those two kinds deliberately and use them as a diagnosis, not as outliers to reject.

What it unlocks: the rejection is now *readable*. A rejected kangaroo shows high $s^{\mathrm{prox}}$, low $s^{\mathrm{id}}$. A rejected patch of gravel shows high $s^{\mathrm{bg}}$. You can point at the dimension that fired.

## The Methodology

Nothing in the detector is touched. It stays frozen. Everything is bolted on.

### Step 1 — mine the two hallucination sources

**Proximal OoD.** For each in-distribution (ID) class, prompt GPT-5 for non-ID categories that look or mean something similar. Pull 1,000 annotated images per class from Objects365 matching those categories. Objects365 is used on purpose — the test benchmarks come from Open Images V7, so there is no leakage. Then run the target detector over them and **keep only the images that actually produce a hallucinated ID box.** That filter is the whole point: these are samples the detector demonstrably gets wrong.

**Background-only.** 1,000s of texture images from DTD (Describable Textures Dataset). DTD has no object labels, so first use YOLOE-11-L to throw out any image that contains an ID object. Then the same filter: keep only what hallucinates.

### Step 2 — annotate parts automatically

This is the pipeline that makes the whole thing feasible:

1. Prompt GPT-5 for a small vocabulary of *visually localisable* parts per class. For `bird`: beak, eye, wing, torso, feet.
2. OWLv2 grounds each part word inside the RoI crop → a part bounding box.
3. SAM 2 turns each box into a pixel mask.
4. **Keep the mask only if ≥70% of its pixels sit inside SAM 2's object mask.** This kills background bleed.
5. Project the surviving mask into RoI coordinates and rasterise to a binary $7\times 7$ grid.
6. Parts with no surviving mask get an all-zero target — which matters later.

Quality check against baselines, concept-mIoU / Recall@0.5:

| Annotator | mIoU | R@0.5 |
|---|---|---|
| **This tool** | **40.8** | **38.4** |
| VLPart | 37.0 | 36.8 |
| Grounded SAM | 32.3 | 30.7 |

### Step 3 — the semantic head

For each prediction $p_i = (\mathbf{b}_i, y_i)$, pool an RoI feature $\mathbf{F}_i$ at $7\times7$:

| Detector | Source | $\mathbf{F}_i$ shape |
|---|---|---|
| YOLO | detect neck, 3 scales | $896\times7\times7$ |
| RT-DETR | hybrid-encoder neck, 3 scales | $768\times7\times7$ |
| Faster R-CNN | Detectron2 `box_pooler` | $256\times7\times7$ |

For YOLO and RT-DETR, RoIAlign is run per scale and the results concatenated on channels. RT-DETR deliberately uses encoder features, not decoder queries — queries have no canonical spatial grid inside the box.

One head **per ID class**, because "wing" is meaningless for a bottle. Each head: $1\times1$ conv down to 256 channels → 2 residual blocks (3×3 conv, GroupNorm, GELU, Dropout2d 0.1) → $1\times1$ conv to $N_{y_i}$ concepts. Output is $\mathbf{A}_i \in [0,1]^{N_{y_i} \times 7 \times 7}$.

The concept vocabulary for class $y_i$ is split into three groups: ID parts, proximal-class parts, background concepts. At inference each channel is pooled with log-sum-exp ($\tau = 0.5$) to one scalar $a_i^c$, and each group takes its max:

$$s_i^k = \max_{c:\,\pi(c)=k} a_i^c, \qquad k \in \{\mathrm{id}, \mathrm{prox}, \mathrm{bg}\}$$

### The loss — three terms

$$\mathcal{L}_{\mathrm{SPK}} = \mathcal{L}_{\mathrm{concept}} + \lambda_g \mathcal{L}_{\mathrm{suppress}} + \lambda_s \mathcal{L}_{\mathrm{group}}$$

with $\lambda_g = 0.25$, $\lambda_s = 0.75$.

**1. Concept reconstruction** — Dice loss between activation and the $7\times7$ mask:

$$\mathcal{L}_{\mathrm{concept}} = \mathbb{E}_{p_i}\!\left[\mathrm{DiceLoss}(\mathbf{A}_i, \mathbf{M}_i)\right]$$

Dice, not [[Cross Entropy|cross-entropy]], because the masks are tiny and very unbalanced — a beak is 2 cells out of 49.

**2. Spurious suppression** — punish concepts that should be absent. If a bird's feet are occluded, the feet channel's target is empty, so "feet" joins the absent set $\mathcal{C}_i^-$. With $U_i$ the union of all annotated-positive cells:

$$\mathcal{L}_{\mathrm{suppress}} = \mathbb{E}_{p_i}\!\left[\frac{1}{|\mathcal{C}_i^-|}\sum_{c \in \mathcal{C}_i^-} \frac{\langle U_i,\, -\log(1 - A_{i,c})\rangle}{\|U_i\|_1}\right]$$

In words: inside the cells where *something* was annotated, the absent concepts must stay near zero. Average over absent concepts, normalise by how many cells were positive.

**3. Group discrimination** — the dominant group must match the sample's true source. Softmax cross-entropy over the three group scores $s_i^k$ against a one-hot label $\mathbf{q}_i$ (ID sample → id, proximal sample → prox, texture sample → bg):

$$\mathcal{L}_{\mathrm{group}} = -\mathbb{E}_{p_i}\sum_{k} q_i^k \log \frac{\exp(s_i^k)}{\sum_{k'}\exp(s_i^{k'})}$$

Training: AdamW, lr $2\times10^{-4}$, weight decay $5\times10^{-4}$, batch 2000, ≤80 epochs, early stopping patience 10, 10% val split, WeightedRandomSampler. All 10 BDD class heads: ~1 hour on one A100-40GB.

### Step 4 — the two cheap priors

**Geometric** — just relative box area, scale-invariant:

$$r_i = \frac{\mathrm{Area}(b_i)}{\mathrm{Area}(I_{\mathrm{ctx}}(p_i))}$$

**Contextual** — treat the whole image as one RoI, take channel-wise spatial mean and std of the neck features, concatenate, $\ell_2$-normalise. YOLO: 512-d. RT-DETR: 4096-d. Faster R-CNN: 2048-d. Build **class-specific** banks from ID training images, then:

$$d_{\mathrm{ctx}}(\mathbf{v}) = 1 - \frac{1}{k}\sum_{\mathbf{r} \in \mathcal{N}_k(\mathbf{v},\, \mathcal{V}_{\mathrm{ID}}^{\hat y})} \cos(\mathbf{v}, \mathbf{r}), \qquad k = 5$$

No external encoder — deliberately, since the claim is about priors *inside* the detector.

### Step 5 — the actual OoD decision

Fit anything on the 5-d vectors. They try MDS, BAM, KNN, and **Isolation Forest**, which wins. iForest is a few hundred random trees; it isolates a point by how few random splits it takes to cut off. Inference cost: 0.05 ms.

Total overhead: 2.72 ms/image on an A4000-8GB, 26.8% over the detector's 10.15 ms. Semantic heads are 0.17 ms; the rest is pulling RoI features out of the same forward pass.

## Ablation Studies and Experiments

Benchmarks: PASCAL-VOC and BDD-100K as ID. Near-OoD (visually close) and Far-OoD. Detectors: YOLOv10 (one-stage), Faster R-CNN (two-stage), RT-DETR (transformer). Protocol is the *calibrated* benchmark from Wu et al. 2026, which removed test contamination from the older setup.

### Same algorithm, different representation

FPR95, lower better. Each SPK row is the identical algorithm, fed 5 numbers instead of ~900:

| | YOLO VOC Near | YOLO VOC Far | YOLO BDD Near | YOLO BDD Far |
|---|---|---|---|---|
| MDS | 57.67 | 69.47 | 68.42 | 82.35 |
| **SPK-MDS** | **14.99** | **17.28** | **23.35** | **6.46** |
| BAM | 45.36 | 43.72 | 49.63 | 52.18 |
| **SPK-BAM** | **21.96** | **21.73** | **16.75** | **3.07** |
| KNN | 48.20 | 39.50 | 41.95 | 45.24 |
| **SPK-KNN** | **19.64** | **17.99** | **13.47** | **1.17** |
| iForest | 70.27 | 67.82 | 60.42 | 65.23 |
| **SPK-iForest** | **14.25** | **11.84** | **9.86** | **0.70** |

Look at iForest: 70.27 → 14.25. The algorithm is unchanged. **The gain is entirely the representation.** That is the paper's real claim, and it holds on Faster R-CNN (iForest 75.43 → 13.92 on VOC Near) and RT-DETR (79.52 → 15.48) too.

AUROC tells the same story: SPK-iForest hits 96.31 / 95.43 on VOC and 98.10 / 99.81 on BDD with YOLO.

### Beating a method that retrains the detector

The real test. Proximal-OoD (Wu et al. 2026) fine-tunes the detector's objectness *and* runs KNN on top. SPK touches nothing. Raw count of surviving hallucinations, Near/Far:

| Detector | Method | VOC | BDD |
|---|---|---|---|
| YOLO | Original | 946 / 440 | 701 / 666 |
| | Proximal-OoD | **134** / 60 | 80 / 47 |
| | SPK | 135 / **52** | **69** / **5** |
| Faster R-CNN | Original | 2150 / 1335 | 2576 / 1634 |
| | Proximal-OoD | 710 / 253 | 207 / 167 |
| | SPK | **299** / **140** | **60** / **25** |
| RT-DETR | Original | 2311 / 1589 | 3145 / 1220 |
| | Proximal-OoD | 386 / 470 | 525 / 240 |
| | SPK | **358** / **275** | **359** / **113** |

Faster R-CNN on BDD: 207 → 60 Near-OoD hallucinations. A frozen detector plus 5 numbers beats a fine-tuned detector.

### Loss ablation — which term earns its keep

YOLO, FPR95, averaged over VOC Near/Far and BDD Near/Far:

| Dice | Suppress | Group | Average |
|---|---|---|---|
| ✗ | ✓ | ✓ | 20.80 |
| ✓ | ✓ | ✗ | 17.44 |
| ✓ | ✗ | ✓ | 15.85 |
| ✓ | ✓ | ✓ | **9.16** |

Reading: **Dice is load-bearing** — removing it is the worst single hit (20.80). Group is next (17.44). Suppress is the smallest contributor on its own (15.85) but still worth 6.7 points. And no partial combination gets close to 9.16 — the three terms are not redundant with each other. Spatial grounding, group discrimination, and specificity are three different requirements.

Same ordering on Faster R-CNN and RT-DETR.

### Prior ablation — and the one honest negative

YOLO, FPR95:

| Priors | VOC Near/Far | BDD Near/Far | Avg |
|---|---|---|---|
| Semantic | 15.43 / 13.28 | 30.37 / 25.58 | 21.17 |
| + Geometric | **13.23** / **11.50** | 20.88 / 3.84 | 12.36 |
| + Contextual | 14.25 / 11.84 | **9.86** / **0.70** | **9.16** |

This is the most interesting table in the paper, because **adding the contextual prior makes PASCAL-VOC slightly worse** (13.23 → 14.25 Near, 11.50 → 11.84 Far). The authors say so plainly. Their explanation: VOC test images already look like VOC training images, so image-level context carries no signal and just adds noise to the 5-d vector.

On BDD, the same prior is transformative: 20.88 → 9.86 Near, 3.84 → 0.70 Far. BDD is driving scenes with real scene-level variety, so "is this whole image unlike anything I trained on" actually means something.

**Practical read: the contextual dimension is dataset-dependent and should be validated, not assumed.** On Faster R-CNN BDD the same step is 14.52 → 2.31, so the effect is about the dataset, not the architecture.

Note too that semantic-alone is already strong on VOC (15.43) but weak on BDD (30.37). Part concepts carry most of the weight when objects are large and centred; they carry less when objects are small, distant traffic participants.

### Does the semantic head actually learn parts?

Group-classification accuracy (argmax over the three group scores), YOLO on VOC:

| Split | Accuracy |
|---|---|
| ID training | 96.5% |
| Proximal OoD | 81.0% |
| Background | 77.0% |

The 96.5 → 81.0 / 77.0 gap is a fit gap and the authors flag it. The heads are much more certain on ID than on the mined hard cases.

Qualitatively, the wing / torso / foot activation maps land on the right anatomy. UMAP plots show ID and OoD overlapping heavily in the detector's classification logits but separating in SPK space.

### The uncalibrated benchmark, for comparability

To compare against methods with no public code or architecture-specific designs, they also run the older (contaminated) protocol: Deformable-DETR and Faster R-CNN on VOC/BDD, with MS-COCO and OpenImages as OoD.

Deformable-DETR, ID = BDD, OoD = OpenImages, FPR95: SIREN-KNN 47.28, SAFE 21.10, UNO-Adapter 3.80, **SPK 0.37**, SPK (DINO ViT) **0.00**.

But on **VOC Near** with Deformable-DETR, plain SPK loses: FPR95 52.32 vs UNO-Adapter's 32.61. Swapping the detector-internal contextual prior for a DINO ViT embedding fixes it (28.55) — which is honest, and also an admission that the detector-intrinsic image representation is the weak link on VOC. That is the same story the prior ablation told.

SPK variants win 10 of 12 cases; second in the remaining one.

## Worth Remembering

**The headline result is about representation, not algorithms.** Isolation Forest went from the *worst* method on raw features (70.27) to the best on SPK features (14.25). If you take one thing: when your OoD detector underperforms, suspect the feature space before the scoring function.

**The hard-sample filter is doing quiet work.** They don't just use proximal/background images — they keep only the ones that *already* fool this specific detector. That makes the training set detector-relative. It also means **the semantic heads do not transfer between detectors**; you remine and retrain per model.

**One head per class.** 20 classes on VOC means 20 heads. That is fine at 10–20 classes (an hour on one A100 for BDD's 10). It is a real problem at COCO's 80 or at open-vocabulary scale, and the paper doesn't address it.

**The pipeline leans on three large pretrained models** — GPT-5 for concept vocabularies and proximal class lists, OWLv2 for grounding, SAM 2 for masks. "Requires only lightweight manual verification" is doing some work in that sentence. And the annotation quality is 40.8 mIoU: better than VLPart, but not good.

**$7\times7$ is very coarse.** A bird's eye in a 49-cell grid is one cell, maybe zero. This keeps training cheap and avoids needing pixel segmentation at train time, but it caps how fine the part evidence can get.

**The group-accuracy drop (96.5% ID → 77% background) suggests the heads are closer to overfit than the main tables imply.** The OoD numbers are still excellent, so the 5-d vector is apparently robust to imperfect per-concept decoding — but it is worth watching if you reproduce this.

**Limitations the authors name:** only two hallucination sources are modelled. Occlusion, motion blur, unusual viewpoints, adversarial patterns — none are covered. They frame the framework as extensible to more sources, which is a promise rather than a result.

**Practical caveats if you want to use this:**
- Validate the contextual prior per dataset. It hurt VOC. Dropping to 4 dimensions may be the right call.
- RT-DETR: pull hybrid-encoder features, not decoder queries. Queries have no spatial grid in the box.
- 26.8% latency overhead at 2.72 ms on an A4000. Most of that is RoI extraction, not the heads (0.17 ms) or iForest (0.05 ms).
- The $\ge$70% object-mask-overlap threshold on part masks is the main thing standing between you and garbage annotations. Keep it.

**Connections.** The three group scores are a tiny concept bottleneck: $s^{\mathrm{id}}$, $s^{\mathrm{prox}}$, $s^{\mathrm{bg}}$ are human-named variables the decision must route through. The contextual prior is textbook [[Embeddings|embedding]]-space KNN — exactly the move in deep-nearest-neighbour OoD, applied at image level with class-conditional banks. [[Uncertainty|Epistemic uncertainty]] is what the whole thing is estimating; the "confident on things it never saw" failure is the canonical epistemic-uncertainty symptom. And the mining step is near-[[Distributed Representations of Words and Phrases (negative sampling)|hard-negative mining]]: the filter keeps only samples the current model gets wrong, which is the same instinct as [[Approximate Nearest Neighbor Negative Contrastive Learning (ANCE)|ANCE]].

**Open questions.** Does the 5-d space extend to more sources without becoming a 50-d space and losing its whole advantage? Can one shared head with class conditioning replace $N$ per-class heads? Why does suppress help least in isolation but still add 6.7 points when combined — is it regularising, or fixing a specific failure? And the authors' own framing — that latent priors could be elicited *before* the hallucination is generated rather than filtered after — is still entirely unexplored.

## Links

Related: [[Uncertainty]] · [[Embeddings]] · [[Cross Entropy]] · [[Attention]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[Approximate Nearest Neighbor Negative Contrastive Learning (ANCE)]] · [[Hallucination]] · [[Shortcut Learning in Deep Neural Networks]] · [[Evals]] · [[Curse of Dimensionality]]

New topics worth writing: Out-of-distribution detection, Object detection architectures (YOLO / Faster R-CNN / DETR), Isolation Forest, Concept bottleneck models, Dice loss, RoIAlign, Open-vocabulary grounding (OWLv2, SAM 2), Mahalanobis distance OoD scoring, FPR95 and AUROC as OoD metrics, Part-based models and adversarial robustness, Benchmark contamination in OoD evaluation
