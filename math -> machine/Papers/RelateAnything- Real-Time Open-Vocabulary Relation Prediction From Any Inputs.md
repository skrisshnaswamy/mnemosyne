---
title: "RelateAnything: Real-Time Open-Vocabulary Relation Prediction From Any Inputs"
authors: ["Maëlic Neau"]
year: 2026
arxiv: "2609.12552"
url: https://arxiv.org/abs/2609.12552
priority: Good-To-Read
read_on: 2026-09-22
tags: [paper, rl, self-supervised, vision]
---
## The Core Idea

Object detection stopped owning its label list. You hand [[CLIP]]-style detectors a list of class names at test time and they find those things. Segment-anything models go further and return regions with no names at all. In both cases the **taxonomy left the model and became an input**.

Relation prediction never made that jump. Scene-graph models still predict from a frozen list of 50 predicates (VG150) or 56 (PSG), and their relation head is wired to take the *object class labels* as input. That single design choice locks the model to one detector and one label space forever.

This paper moves relation prediction across that line. **RelateAnything** is a 53M-parameter model that takes three separable inputs:

1. pixels,
2. a set of boxes or masks from *any* source,
3. a list of predicate strings, supplied at inference.

Object class labels are never an input at any stage. So you can swap the detector — YOLO-World, YOLOE, FastSAM, a human annotator — without retraining. And the predicate vocabulary is a **bank of frozen text embeddings** with no learned layer on top, so changing the vocabulary is literally swapping one matrix.

> [!NOTE] Vocabulary-as-input
> If any learned weight transforms the text side, that weight is undefined for a string you supply later. Keeping the text path completely frozen is what makes the vocabulary swappable. Scores are just cosine similarities between a visual pair embedding and a text embedding. ^vocabulary-as-input

Three things had blocked this, and only one of them was a modelling problem.

**Supervision did not exist.** Visual Genome's annotators wrote 36,549 distinct predicate strings; the community kept the 50 most frequent and threw away exactly the part an open-vocabulary model is supposed to serve. Existing machine-generated relation corpora either project onto a fixed class list or never check the annotator at all.

**The architecture could not use it.** A head conditioned on object labels inherits a label space. A head that encodes the whole vocabulary as one caption is capped at roughly 150 strings by the text encoder's context.

**The measurement could not see it.** This is the deepest one. Scene-graph recall scores you against a benchmark whose predicate list *is also its training list*. A model that answers with a better, unlisted string is marked wrong. The paper shows this concretely: **a lookup table over ground-truth object category pairs, which never sees a pixel, beats the trained model by 15% on the metric that orders leaderboards** (68.4 vs 57.7 top-1 on VG150) while losing by 86% on the per-predicate average (18.9 vs 35.1).

Two more things had to be fixed to train at ten thousand predicates. First, supervision becomes **positive-unlabeled**: a pair annotated `riding` is also, silently, `sitting on`, and punishing the unstated one teaches the model to suppress correct answers. Second, and less obvious, the **target text space is broken**: a contrastive text encoder puts `above` and `below` at cosine 0.95 and `to the left of` / `to the right of` at 0.99 — indistinguishable from its synonym pairs at 0.96. No visual model regressing onto those directions can separate them, no matter how good it is.

> [!NOTE] Antonym collapse
> Contrastive text encoders see antonyms in near-identical contexts, so they embed them almost identically. Relations are the one task where this is fatal, because direction *is* the label. Mean-direction removal ([[Whitening Sentence Representations]]-style) does not fix it. ^antonym-collapse

## The Methodology

### Visual path

A [[An Image is Worth 16x16 Words (ViT)|DINOv3 ViT-S/16+]] backbone at $448\times448$. Patch features are read at three depths ($-6$, $-3$, $-1$), each layer-normalised on its own tap, then fused with learned weights. Without the per-tap norm the fusion is swamped by the last layer's larger activation scale.

Per-object features come from **pooling, not cropping**. A query is built from a Fourier position encoding of box $b_i$, and that query cross-attends over *all* patch tokens:

$$v_i = \mathrm{Attn}(q(b_i), F, F)$$

So an object's vector can already contain the context it interacts with. See [[Cross Attention]] — the box is the query, the image is the key/value.

The backbone is fully fine-tuned at a learning rate $8\times$ lower than the head's. That beat both a frozen backbone and [[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]] adapters at this scale.

### Pair representation

For an ordered pair $(i,j)$:

$$x_{ij} = [\,v_i;\ v_j;\ v_{\cup(i,j)};\ v_{\cap(i,j)};\ \mathrm{MLP}(g_{ij})\,]\,W_{\mathrm{proj}}, \qquad x_{ij}\in\mathbb{R}^{512}$$

- $v_\cup$ pools the union box.
- $v_\cap$ pools the **contact region** — the intersection if boxes overlap, the *gap between them* if they do not.
- $g_{ij}\in\mathbb{R}^{19}$ is scale-invariant geometry.

Then a small transformer refines the sampled pairs: self-attention *across pairs* (so one pair can support another), cross-attention to scene tokens, plus tokens encoding the pair's own box corners.

### Deformable scene read

A final additive stage lets a pair look outside its own boxes. It predicts sampling offsets around four anchors (subject, object, union, contact) — 8 heads × 4 points each — and reads the feature map there through a **zero-initialised gate**. Training therefore starts from the model *without* the stage, and the gate's own gradient decides whether it gets used.

This is the bit that lets `parked on` be settled by the road surface and `hanging from` by the attachment point above the subject. Measured on trained models, the subject anchor carries 43–48% of the attention mass and the union anchor only 4–5% (its box is nearly the whole image, so half its reads get clamped at the border).

Two implementation details that mattered: offsets are **unbounded** (the anchors are a prior, not a constraint), and sampling points are initialised on rings of distinct angles per head — a variant that zero-initialised the offset predictor put every point of an anchor in one place, where identical gradients cannot pull them apart.

### Two branches, one gate

Spatial and semantic relations need different evidence, so each pair gets two embeddings and the mix is chosen *per predicate*:

$$\ell_{\mathrm{pred}}(i,j,p) = \tau\big[\alpha_p \cos(z^{\mathrm{spa}}_{ij}, e_p) + (1-\alpha_p)\cos(z^{\mathrm{sem}}_{ij}, e_p)\big] + \beta$$

Crucially $\alpha_p = g(e_p)$ — the gate reads **only the text embedding**, so it is defined for strings the model has never seen. It gets no supervision of its own, and it comes out strongly bimodal: `behind` 1.000, `below` 0.9998, `above` 0.970 at the geometry end; `carrying` 0.0004, `parked on` 0.0008, `riding` 0.012 at the appearance end; contact predicates in the middle (`holding` 0.82, `on` 0.64, `wearing` 0.22). Median $\alpha_p$ is 0.002; 12% of predicates exceed 0.5.

Side effect: one forward pass gives you two graphs. Score against the spatial predicates for a layout graph, against the semantic ones for a content graph.

### Pair sampling

An image with $N$ objects has $N(N-1)$ ordered pairs. A two-stage sampler keeps 400 by geometric plausibility, then 128 by a learned relatedness score. It retains **99.79% of annotated positives**; scoring everything costs $1.02\times$ and buys nothing measurable.

The learned stage is trained to predict which pairs *carry annotations*, so it is explicitly learning annotation propensity. Its logit enters the final score **additively**, which means you can delete it at evaluation and measure what it was doing.

$$s = \sigma\big(a(\ell_{\mathrm{pred}} + w\,\ell_{\mathrm{pair}}) + b\big)$$

### The loss

The main term is a batch-local InfoNCE ([[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]]) over predicate slots, with two departures.

**The positive is a synonym group, not a string.** The annotated predicate's synonym group is aggregated as a weighted mean, with weights from an isotonic fit of "probability of synonymy" against text-space cosine on 1,177 lexical synonym pairs.

**Negatives are discounted by how likely they are to be true.** For a pair annotated $p$, the denominator logit of negative $q$ becomes

$$\ell_{\mathrm{pred}}(i,j,q) + \log\big(1 - \hat{P}(q \mid p)\big)$$

so a predicate that is almost certainly also true contributes almost nothing. $\hat{P}(q\mid p)$ is fitted on the 706k box pairs in the training mixture carrying more than one annotation, from co-annotation counts, text cosine and frequency, with the Elkan–Noto correction for the fact that an unstated predicate is unlabeled rather than false.

> [!NOTE] Positive-unlabeled supervision
> With 50 predicates, "not annotated ⇒ negative" is roughly right. With 10,000 it is systematically wrong. This is the same missing-not-at-random problem as implicit feedback in recommenders — see [[Collaborative Filtering for Implicit Feedback Datasets (ICDM)]] and [[Recommendations as Treatments- Debiasing Learning and Evaluation]]. ^pu-supervision

Two exemptions are essential: annotated predicates are never discounted, and **directional inverses always get full weight**. The signal separating `above` from `below` is the one that must never be softened.

Four auxiliary losses: a per-cell sigmoid term (each pair–predicate cell as an independent binary problem, negatives mass-matched to positives), a **swap hinge**

$$\mathcal{L}_{\mathrm{swap}} = \max\big(0,\ m - c(i,j,p) + c(j,i,p)\big), \quad m = 0.05$$

weighted by each predicate's estimated probability of being directional, background suppression at a small weight, and an object–text grounding term on the pooled features. Object categories touch training *only* through the sampler's relatedness loss and that grounding term — never the forward pass.

### Fixing the text encoder

A 512-d student is distilled from the dino.txt teacher over the full CLIP BPE vocabulary, so no predicate string can fall out of vocabulary. Six terms: relational distillation (matching the pairwise-cosine matrix), an absolute anchor, neighbourhood preservation (row-wise KL over batch similarities), antonym repulsion (a hinge pushing known inverses below a margin), synonym cohesion, and a softer hinge for modified spatial forms. Teacher targets get their three leading principal directions removed first, or every off-diagonal cosine sits near 0.80 and the relational term is dominated by that bulk.

Results (see [[Distilling the Knowledge in a Neural Network]] for the general shape):

| | teacher (2048-d) | student (512-d) |
|---|---|---|
| synonym cosine | 0.96 | 0.71 |
| inverse cosine | 0.92 | **0.09** |
| AUC syn vs inv | 0.854 | **0.989** |
| effective dim | 87 | 44 |
| hubness | 8.1 | 1.3 |

Antonym repulsion is the term a frozen teacher cannot supply. **Neighbourhood preservation is what stops the student cheating**: without it a student reaches an equally good AUC (0.980) but collapses to an effective dimension of 24 instead of 44.

### RA-4M, the corpus

474,413 images, 4,282,531 relations, 10,102 free-text predicates, 9.03 relations/image, generated in 104 GPU-hours. It re-annotates MegaSG's images and reuses its boxes verbatim, so the comparison isolates the annotation schema.

The annotator is an open-weight 26B MoE VLM (4B active) served with vLLM, shown the image with a **coloured numbered dot at the centre of each box**. Grounding is therefore an *input* to the annotator, not something inferred from its text output. Three passes:

1. **Semantic** — open vocabulary, spatial predicates explicitly banned.
2. **Grow** — one additive round to fill sparse graphs. Contributes most of the density and most of the errors.
3. **Spatial** — open prompt asking for depth/contact/proximity ahead of left/right; proposals not reducible to one of 21 layout forms are dropped.

Then every proposal passes a **deterministic geometric gate**, which rejects 11.3% of raw candidates. The design principle: *a gate fires only when box geometry logically contradicts the predicate.* Contact predicates need touching boxes; containment needs $|b_s \cap b_o| \ge 0.5|b_s|$; proximity needs a gap $\le 0.5\times$ the larger box diagonal; left/right and above/below get their **roles swapped** to match the boxes rather than rejected.

> [!NOTE] Verify only what geometry constrains
> A gate that also guessed on gaze or attention would trade a known coverage gap for an unknown error rate. Unconstrained predicates pass through unchecked and are *counted as residual risk*. About 22% of `looking at` annotations have disjoint boxes and cannot be verified. ^geometric-gate

Because the annotator prefers the left/upper object as subject, every verified left/right, above/below and front/behind relation is restated from the other endpoint with probability $\tfrac12$. That leaves the corpus **direction-balanced**, a property no source corpus has and which the swap hinge depends on.

Against the source annotations on the same images and boxes: $1.7\times$ denser (9.03 vs 5.29 rel/img), $107\times$ the vocabulary, 1.4 nats more predicate entropy, mean object degree 1.88 → 3.20, share of images forming one connected graph 70.9% → 82.6%. The head predicates change too — `near`/`on` at 24.3%/23.2% become balanced spatial predicates with nothing above 8%.

Synonyms are **not** collapsed. `holding`, `grasping`, `carrying` and 535 relatives stay distinct strings, because a model whose deployment vocabulary is arbitrary must be trained on targets that populate the text space densely. Canonical groups live only inside the loss and the gates.

Leakage control is worth noting: Visual Genome keys images by VG id and MegaSG by zero-padded COCO id, so a naive filename intersection returns zero and is *wrong*. Mapping through an explicit 51,498-entry VG↔COCO table plus perceptual near-duplicate search removed 8,687 corpus images and 32,456 raw VG images.

### Training mixture and recipe

Two "towers" are reported. The **zero-shot tower** is RA-4M + leakage-filtered raw Visual Genome (with all free-text strings, not VG150's 50). The **released tower** adds HICO-DET train at 21.0% of images — which is only a 5.0% *relation* share, because HICO-DET annotates 1.94 rel/img against RA-4M's 9.02. Mixture fractions are applied per image but the loss is per relation, so only the relation share describes the supervision.

The predicate bank is 19,103 strings: RA-4M's 10,102 plus 9,001 taken from three corpora whose *relations* are never trained on. Since nothing on the text side is learned, the bank is the answer space, not the supervision, and widening it is free.

Recipe: 12 epochs, batch 32 × 4 A100 (5h04m total), head lr $4\times10^{-4}$, backbone lr $5\times10^{-5}$, dropout 0.2, EMA 0.9998, loss weights InfoNCE 0.5 / sigmoid 0.25 / swap 0.5 / background 0.05 / grounding 0.1, 512 InfoNCE negatives per anchor.

Multi-scale training samples resolution from seven rungs in $[0.5, 1.5]\times 448$ — it works as a **regulariser** (cuts the train–val gap by 60%), not a resolution trick. **No geometric augmentation**: a horizontal flip inverts every left/right relation, and the corpus deliberately contains those in balanced quantity.

### OV-SGG-Bench

Six axes, every one scored **cross-dataset** (no benchmark contributes a training image), combined by a harmonic mean so imbalance is punished.

| axis | what it measures | how it fails alone |
|---|---|---|
| A1 transfer | closed-vocab recall, 4 sources, GT boxes | satisfiable by corpus match |
| A2 precision | Haystack, federated AP vs **adjudicated negatives** | invariant to uniform score depression |
| A3 open vocab | all 19,103 strings, synonym-tolerant matching | rewards head collapse |
| A4 deployment | detection mode on a *shared* detector, vs measured pair-recall ceiling | bounded by the detector |
| A5 graph quality | information in relations a VLM judge accepts | judge prefers short repetitive graphs |
| A6 spatial | SpatialSense, balanced adversarial true/false, chance 50% | a boxes-only baseline scores 68.8 |

A2 and A6 are the two **prior-blind** axes. Every claim about relational competence rests on those.

## Ablation Studies and Experiments

### The headline table

All cross-dataset, ground-truth boxes, graph-constrained, one evaluator. The baseline **receives ground-truth object labels**; RelateAnything receives none.

| benchmark | OvSGTR R@50 / mR@50 / rare | RelateAnything R@50 / mR@50 / rare |
|---|---|---|
| VG150 | 39.9 / 10.4 / 0.0 | **53.3 / 28.2 / 42.7** |
| PSG | 28.7 / 8.8 / 1.7 | **40.1 / 30.6 / 23.9** |
| IndoorVG | 48.1 / 12.8 / 4.0 | **52.7 / 29.5 / 21.6** |
| HICO-DET (zero-shot tower) | 34.0 / 4.5 / 0.2 | **35.3 / 12.7 / 4.3** |

Mean recall is $2.3$–$3.5\times$ the baseline; rare-bucket recall is $5$–$21\times$ (undefined on VG150, where the baseline scores exactly 0.0). Micro recall only $1.04$–$1.40\times$ — which is exactly what the prior-matching account predicts, since micro recall is dominated by the head predicates a frequency table can supply.

Composite: **40.1 vs 11.8**. Withholding any one axis leaves the ordering unchanged (ratios 2.0–3.7×).

### The measurement findings — the most transferable part

**Frequency table beats the model on the leaderboard metric.** Neural Motifs' `freq` baseline (most frequent training predicate for a ⟨subject, object⟩ category pair, **no pixels**) vs the trained model, per-edge top-1:

| | micro (freq / ours) | macro (freq / ours) |
|---|---|---|
| VG150 | 68.4 / 57.7 (−15.6%) | 18.9 / **35.1** (+85.7%) |
| PSG | 50.9 / 43.3 (−15.0%) | 20.7 / **31.6** (+52.4%) |
| IndoorVG | 67.9 / 57.3 (−15.5%) | 29.8 / **38.4** (+28.9%) |

Not a claim that vision is unnecessary: the edge-by-edge join shows the model correct where `freq` is wrong on 6.9–12.8% of edges. The narrower conclusion is that *a metric a pixel-free lookup table can win does not measure relation understanding, and it is the metric that orders leaderboards.* Cousin results: [[On Sampled Metrics for Item Recommendation (KDD)]], [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]].

**Shared triplet mass.** The fraction of a training corpus's relation instances whose ⟨subject category, predicate, object category⟩ *triple* the benchmark also annotates. The baseline's fine-tuning corpus shares **90.9%** of its mass with VG150; this paper's mixture shares **12.8%**. On PSG it is 8.6% vs 10.7% — and there the baseline's margin evaporates. The statistic is predictive: an arm trained with more raw Visual Genome hit **54.3 R@50 on VG150**, the best zero-shot figure the author is aware of, *while being the worst model they trained on every tail metric*.

**The matcher gives duplicate credit.** The standard SGDet matcher (inherited from Neural Motifs, still used by current open-vocabulary methods) enforces no assignment constraint, so a ground-truth object covered by $d$ detections offers $d$ chances to recover the same relation — and on VG150 each recovered GT box is covered by **2.99 detections** on average. Object detection has forbidden this since COCO. Restoring one-to-one assignment costs the published model **12.0% of R@50** (2.45 points) — larger than the 2.39 points gained by upgrading Swin-T → Swin-B, which is a separate leaderboard entry.

**The detector operating point is an unreported free parameter.** Detection-mode recall is capped by *pair recall*, which is quadratic in object recall. Holding detector weights fixed and moving only confidence threshold and box cap moves the ceiling by **12–49% relative**; on VG150 that is 19 points, against a **4.3-point spread between all published methods**. It can even invert comparisons: GroundingDINO looks stronger than YOLO (ceiling 0.82 vs 0.64) purely because it applies no threshold and keeps 97 boxes/image vs 27 — at a matched budget YOLO reaches 0.83.

**The matcher for free-text systems is a third convention with an oversized effect.** One set of generations per model, read up to three ways:

| model | matcher | R@50 | mR@50 | pair cov. |
|---|---|---|---|---|
| RelateAnything | exact | 13.6 | 13.0 | 99.7 |
| RelateAnything | A3 synonym | **36.8** | **31.3** | 99.7 |
| ROBIN-3B | exact | 26.2 | 20.0 | 45.6 |
| ROBIN-3B | A3 synonym | 37.9 | 25.1 | 65.6 |
| Qwen3-VL-32B | A3 | 15.6 | 5.2 | 35.5 |
| InternVL3.5-8B | A3 | 9.6 | 4.5 | 23.1 |

Synonym matching is worth $+11.7$ R@50 to ROBIN and $+23.2$ to RelateAnything, and **flips the mean-recall ordering** of the two purpose-built systems. A single row is a choice of scorer, not a measurement of a model.

### What is doing the work

**Label dependence.** Both models on the *same* boxes and predicted labels:

| | self-lookup % | $H(p\mid c_s, c_o)$ | distinct predicates used |
|---|---|---|---|
| OvSGTR Swin-T | 87.7 | 0.41 | 20 / 50 |
| OvSGTR + MegaSG | 89.9 | 0.37 | 31 / 50 |
| RelateAnything | 69.7 | 1.19 | **50 / 50** |

Shuffling the object labels leaves the baseline's output statistics **unchanged** (85% → 85% self-lookup), which means its predicate is a function of the labels, not the image.

**Lesion ladder.** Removing the (optional, gated) object→text channel costs **0.1% of micro accuracy** on every source. Exact variance decomposition of the semantic logit: **pair context 87–93%, subject 0.1%, object 0.0%**. The gate on that path, initialised at 0.1, is driven to 0.014–0.044 by training. Removing box geometry costs 6–24%; scoring against a shuffled image costs 44–68% micro and 85–91% macro.

**Relation supervision does not create relational features.** The fine-tuned backbone is genuinely different from the pre-trained one (linear [[Understanding Deep Learning Requires Rethinking Generalization|CKA]] 0.58 at the last state; two runs under *different* recipes agree at 0.94, so the task sets the destination). But what changes is *object* structure:

- class selectivity 0.242 → 0.281
- relation selectivity 0.133 → 0.138 (flat)
- $P(\text{partner} > \text{same-class object})$ 0.469 → **0.385**, against chance 0.500 — i.e. fine-tuning on relations made the dense features *worse* at telling a relation partner from an arbitrary object of the same class

The interaction region does emerge (contact advantage 0.356 → 0.438, and the feature visibly shifts from "whole person" to "hand + bird"). But cross-image 1-NN retrieval from that feature gets the **object class** at 0.761 (chance 0.068) and the **verb** at 0.269 — *below* the 0.424 majority-class chance, at every depth, before and after fine-tuning. The region encodes *which object is handled*, not *what is being done to it*.

Conclusion: there is no shared representation to justify a shared backbone, and there is a measurable cost to the shared input.

### What did not work

**Sparse annotation is worse than none.** An Open Images V6 extension (62,589 images, 4.33 rel/img vs RA-4M's 9.03) at a deliberately amplified 25% relation share cost **−6% on the composite and −40.2% on HICO-DET tail recall**. Its unannotated pairs enter training as false negatives under the PU structure. The threshold at which this bites is set by the loss, not the data.

**In-domain measurement lies by about $5\times$.** Doubling the corpus: $+15\%$ in-domain micro recall and $+56\%$ in-domain macro; under transfer, $+2.5$–$3.8\%$ and $+9.9$–$11.9\%$. Worse, one change **flips sign**: removing an automatically derived left/right excess improved in-domain metrics at every epoch ($+2.0\%$ micro, $+27.0\%$ macro) and *reduced* transfer macro recall by up to $5.9\%$. Extrapolated transfer gain from another 500k annotated images was about one point absolute, so they did not generate them.

**Distribution alignment does not explain transfer.** Dropping those edges raised training mass on PSG's 56 predicates from 27.3% to 39.4% and cut left/right from 29.5% to 7.4% — and transfer macro recall still fell 3.8 points.

**Capacity does not help.** ViT-S/16+ → ViT-B/16 is $+114\%$ parameters, $1.33\times$ memory, $1.22\times$ wall-clock, for **$-0.1\%$ composite**. On both ladders the *smallest* towers are best on the spatial axis. What scaling changes is *where the evidence comes from*: dependence on box geometry falls with size ($-35.2\% \to -9.8\%$ on VG150) while dependence on pixels rises ($-53.0\% \to -58.1\%$) — a substitution, not an addition, which rules out "bigger models are just more robust".

**Corpus generation: verify-before-assert backfired.** Asking the VLM to justify each relation before emitting it raised precision by **collapsing semantic predicate diversity from 404 types to 135**. The deterministic gates reach the same precision at no diversity cost. Conversely a purely geometric spatial layer was free but monotone: 6 surface forms, 58% left/right, almost no depth; propose-then-verify gives 20 forms at 32.5% depth.

**A long list of modelling ideas that went nowhere:**

- Stochastic depth cut the train–val gap $8.8\times$ and moved benchmarks by ~zero. *The generalisation gap and the transfer gap are different quantities.*
- WiSE-FT interpolation toward the pre-trained backbone: monotone loss at every weight. The transfer cost is not backbone drift.
- Model soups over seeds **halved A1** — a new seed re-randomises the head, so the arms are in mismatched basins.
- Masks as training input: $-18\%$ under box input. Train on boxes, accept masks at inference ($+6.6\%$).
- Head width 768: $-2.7\%$. The predicate bank has effective rank 43.7, so 12M head parameters already suffice.
- CSLS hubness correction at the synonym matcher: negative — **the cosine threshold it was calibrated against accepted 0.6% of true synonyms in the new embedding space and zeroed the axis.** Every similarity threshold is calibrated to one space.
- Per-predicate calibration: fits in-domain, does not transfer. Two global Platt parameters do.
- CUDA-graph capture: $3.45\times$ faster and **silently incorrect** — graph compilation moved evaluation metrics by $40\times$ the noise floor. No reported number uses compiled inference.
- Exact-string InfoNCE positives improve within-group ranking but cost rare-predicate transfer.
- Letterboxing corrects a real $5.2°$ angle error and moves the projective spatial axis by $+0.1\%$. The model does not use precise angles.

**One prior violation no axis could catch.** Under a *gated* variant of background suppression, the model asserted `wearing` between **disjoint boxes at 11.04%** — $22\times$ more often than its own training data. Three explanations were tested and rejected (vocabulary gap: no, 85% of type errors involve body parts absent from the ontology; graph reasoning: no, 97% of fan-in is the same garment attributed to several boxes of one person; spatial shortcut: no, entity features identify body parts correctly 66.9% of the time). Removing the gate took it to **0.00%** at no cost elsewhere. None of A1–A3 could see it, because the benchmarks never annotate those pairs in either direction.

### Cost

| | params | boxes/img | A40 batch-1 p50 | FPS |
|---|---|---|---|---|
| OvSGTR Swin-T | 177M | 98 | 194.0 ms | 5.1 |
| RelateAnything + YOLO-World | 231M | 20 | **25.0 ms** | **40.0** |

$7.8\times$ faster end-to-end *despite more parameters* — batch-1 cost is not parameter-bound. Honest caveat from the authors: the baseline runs tf32 at 800/1333 scoring ~9,500 pairs; this runs bf16 at 448 px scoring ≤128 sampled pairs. Box budget is the largest of the three differences.

At batch 1 the relation head is **dispatch-bound**: three towers spanning $2.5\times$ in FLOPs all cost 19–20 ms on an A40. The ordering only appears at batch 32 (201 / 188 / 130 img/s). Scoring 19,103 strings instead of 50 costs $<1$ ms at batch 1 and 11–22% of batched throughput. `torch.compile` is the single largest win (1.5–1.8×). An fp16 CPU build hits 7 FPS on 8 threads with 0.955 top-1 agreement.

Against closed-set real-time specialists (REACT, REACT++) trained *per benchmark*: they win micro recall everywhere they were trained; RelateAnything wins macro everywhere but PSG, and its F1@K beats REACT on all three — zero-shot, from an arbitrary vocabulary, with no object labels.

## Worth Remembering

**The reporting checklist is the cheapest thing to steal.** Each item below is nearly free and changed a conclusion in this paper:

1. **Shared triplet mass** between your training corpus and the benchmark, printed beside every recall number.
2. Whether the matcher enforces **one-to-one** ground-truth-to-detection assignment.
3. The detector's **confidence threshold and box cap**, with the resulting pair-recall ceiling.
4. Whether recall is **graph-constrained** (the difference is 11.6 R@50 on VG150, 19.0 on PSG).
5. At least one axis scored against **adjudicated negatives**.

**The propensity term has a sign that depends on the benchmark family.** Keeping the learned pair-relatedness logit *improves* recall on annotation-derived benchmarks and *degrades* truth judgement on SpatialSense's adjudicated negatives — every projective predicate gains $+0.03$ to $+0.05$ AUC when you delete it. "Recall against sparse annotation" and "truth judgement against adjudicated negatives" are different quantities. Same flavour as [[Unbiased Learning-to-Rank with Biased Feedback]].

**Counted, not argued:** at the max-F1 threshold on PSG test, of 39,516 emitted edges, 11.1% match an annotated triple, 10.6% fall on an annotated pair with a different predicate, and **78.3% fall on pairs PSG never annotated in either direction**. Only the middle group is a predicate error in the ordinary sense. Scored as "every unlabeled pair is a false positive" the same predictions give AP 0.035; scored against explicit negatives they give fAP 0.56.

**The LLM-judge lesson generalises beyond this paper.** A judge shown one relation at a time cannot perceive redundancy, because redundancy is a property of the *distribution* a relation sits in. Asked to rate informativeness directly, the judge scored the baseline's `on` relations at 1.05/3 against 0.72 for its other predicates — exactly backwards. The fix is to *measure* informativeness as surprisal under a shared reference rather than elicit it:

$$I = \sum_{r\,:\,\mathrm{judge}(r)=\mathrm{true}} -\log_2 p_{\mathrm{ref}}(\mathrm{pred}(r))$$

Truth alone rewards `on` (2.31 bits); surprisal alone rewards random rare predicates (`riding`, 7.04 bits) which the judge rejects. Neither factor can be inflated by repetition. See [[Evals]] for the judge-design rules this is an instance of.

**The authors report the verdicts that go against them.** The same judge, shown whole graphs at deployed length, **prefers the baseline's** (49 of 58 decisive comparisons), and shown single relations accepts slightly more of the baseline's (4.33 vs 4.07 per image). Their honest reading: per-relation precision is a genuine shortcoming; a deployment that cannot tolerate a false edge should prefer the shorter graph.

**Admitted limitations, in the authors' own words:**

- **No human audit of RA-4M.** Every precision claim about the corpus is *structural* — the gate rejects 11.3%, a rejection is a true negative up to box error, unconstrained predicates are counted rather than checked. A human-read sample is "the measurement a reader should expect before relying on 4.3M machine-generated annotations, and the one to add first."
- **Predicate strings are not held out.** Images, object distribution and annotation style are unseen; every benchmark predicate occurs in the training vocabulary. Only the OvR-SGG leaderboard row holds concepts out, and only for VG150's 15 novel predicates.
- **Several conclusions rest on one seed**, against a 1.3% noise floor.
- **The decoupling result is scope-limited**: nothing in this recipe provides a patch-level relational objective, so "relation supervision does not create relational features" means *this* supervision, not any.
- **A6 remains the weakest axis** — 69.0 macro AUC where the benchmark's own boxes-only baseline (which never sees the image) gets 68.8.

**The leaderboard postscript is instructive.** On OvR-SGG, 15 of 50 VG150 predicates are "novel" — but `on`, `of` and `in` are among them and account for **93% of the novel mass**. So "novel recall" is, to within a few points, recall on three of the most frequent strings in Visual Genome. Retraining with those 15 and their synonym groups removed costs $-5.5$ Base+Novel and $-12.3$ Novel R@50, which is a direct measurement of how much "novel performance" is supervision on `on`. On mean recall over the same predictions the held-out model beats the baseline by nearly $4\times$ (7.2 vs 1.9).

**Follow-up questions worth chasing:**

- Does the PU discount $\hat{P}(q\mid p)$ transfer, or is it fitted to one corpus's co-annotation habits? It is estimated from 706k multiply-annotated pairs *in the training mixture* — the same kind of corpus-bound statistic the paper criticises elsewhere.
- The "shared triplet mass" statistic is basically a train/test contamination measure for label distributions rather than examples. Does an analogue exist for ranking and recommendation benchmarks?
- Multi-label emission: HICO-DET annotates several verbs per pair while the model emits one, so `holding` absorbs `wielding`/`carrying`/`reading` and the specific verb ranked second counts as a miss. That is an emission-rule fix, not a capacity fix.
- The detector, not the relation model, is now the binding constraint — pair recall is quadratic in object recall, and under class-aware matching only 46% (PSG) and 19% (IndoorVG) of ground-truth-box performance survives.

## Links

Related: [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[CLIP]] · [[Collaborative Filtering for Implicit Feedback Datasets (ICDM)]] · [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[Unbiased Learning-to-Rank with Biased Feedback]] · [[On Sampled Metrics for Item Recommendation (KDD)]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[On the Difficulty of Evaluating Baselines]] · [[Do ImageNet Classifiers Generalize to ImageNet]] · [[Towards Quantifying Benchmark Optimization in ASR Models]] · [[Shortcut Learning in Deep Neural Networks]] · [[Distilling the Knowledge in a Neural Network]] · [[Whitening Sentence Representations]] · [[Representation Degeneration Problem in Training NLMs]] · [[How Contextual are Contextualized Word Representations]] · [[Efficient Estimation of Word Representations (word2vec)]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Cross Attention]] · [[Attention Is All You Need]] · [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)]] · [[Dense Passage Retrieval (DPR)]] · [[Evals]] · [[Embeddings]] · [[Reward Hacking]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Troubling Trends in Machine Learning Scholarship]] · [[LoRA- Low-Rank Adaptation of Large Language Models]]

New topics worth writing: Scene graph generation, Positive-unlabeled learning, Federated annotation and LVIS-style evaluation, Deformable attention, Open-vocabulary object detection, Platt scaling and probability calibration, Segment Anything and promptable segmentation, Surprisal as an evaluation signal, Cross-dataset transfer protocols, Annotation propensity and missing-not-at-random labels, Hubness in embedding spaces, DINOv3 and self-supervised visual backbones
