---
title: "Do ImageNet Classifiers Generalize to ImageNet?"
authors: ["Benjamin Recht", "Rebecca Roelofs", "Ludwig Schmidt", "Vaishaal Shankar"]
year: 2019
arxiv: "1902.10811"
url: https://arxiv.org/abs/1902.10811
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, vision]
---
## The Core Idea

Ten years of image classification progress was measured on two test sets: the CIFAR-10 test set (10,000 images) and the ImageNet validation set (50,000 images). Everyone tuned on those same images. The obvious fear is **adaptive overfitting** — the community collectively "trained on the test set" by picking whatever architecture happened to score well, so the reported numbers are inflated and real-world accuracy has plateaued.

The authors did the one experiment that settles this: they went back to the *original data sources* (Tiny Images for CIFAR-10, Flickr for ImageNet) and rebuilt fresh test sets, following the original collection and labelling recipes as closely as they could. Then they ran a decade of models on the new sets without ever having looked at the models during collection.

Two findings, and the second is the interesting one.

1. **Everything drops.** 3–15% on CIFAR-10, 11–14% top-1 on ImageNet. The best ImageNet model fell from 83% to 72% top-1, and 96% to 90% top-5. A 6% top-5 drop is roughly five years of ILSVRC progress erased.

2. **The ranking is preserved, and the slope is greater than 1.** Plot new accuracy against original accuracy and you get a straight line:

$$\text{acc}_{\text{new}} = 1.69\cdot\text{acc}_{\text{orig}} - 72.7\% \quad \text{(CIFAR-10)}$$
$$\text{acc}_{\text{new}} = 1.11\cdot\text{acc}_{\text{orig}} - 20.2\% \quad \text{(ImageNet top-1)}$$

Slope $>1$ means each point you gain on the old test set buys you *more* than a point on fresh data. That is the exact opposite of what adaptive overfitting predicts. If the community had been overfitting, later models would show *diminishing* returns — the line would flatten. It does not.

So the drop is not adaptivity. It is a **distribution gap**: the new images are genuinely a bit harder, and models are shockingly sensitive to that. The paper decomposes the gap formally:

$$L_S - L_{S'} = \underbrace{(L_S - L_{\mathcal{D}})}_{\text{adaptivity}} + \underbrace{(L_{\mathcal{D}} - L_{\mathcal{D}'})}_{\text{distribution}} + \underbrace{(L_{\mathcal{D}'} - L_{S'})}_{\text{generalisation}}$$

Sampling noise (third term) is ruled out: with 10,000 images a 95% Clopper–Pearson interval is at most $\pm 1\%$. Adaptivity (first term) is ruled out by the slope. That leaves the middle term.

> [!NOTE] Distribution gap ^distribution-gap
> The systematic accuracy loss from moving between two data distributions that were *meant* to be the same. Not random noise, not test-set reuse — the second dataset is simply built of slightly different stuff, and models notice even when humans do not.

What this unlocks: a reusable protocol for measuring robustness, and a warning. If a benign, well-intentioned replication of your own benchmark costs you 11 accuracy points, claims about human-level performance and about deployment readiness need heavy discounting.

## The Methodology

**CIFAR-10 replication.** Every CIFAR-10 image has an exact $\ell_2$-distance-0 duplicate in Tiny Images, so the authors recovered the original Tiny Images *keyword* for each image. The top 25 keywords per class cover >95% of the dataset, and the sub-class structure matters a lot — the most common `airplane` keyword is `stealth_bomber`, not a civilian jet; `fire_truck` images are mostly red. They then sampled new images matching that per-keyword distribution.

They even replicated the *labour split*: Author A played the student annotators (picking candidates using the original instruction sheet), Author B played the researchers (reviewing and rejecting). Final step: $\ell_2$ and SSIM nearest-neighbour search to strip near-duplicates within the new set and against all of original CIFAR-10 (train and test). Result: 2,000 images.

**ImageNet replication.** They targeted the *validation* set, not the test set, because validation comes from a single source (Flickr), has public labels, and is what papers report on.

- Flickr queries used WordNet synonyms per class, date range 11 July 2012 – 11 July 2013 (just after ImageNet's collection window, to avoid a depleted pool and hard-to-detect near-duplicates of training images).
- Results were sorted by **upload date**, not by "relevance" or "interestingness" — those Flickr rankings may themselves be driven by ImageNet-trained models, which would bias the new set toward images current classifiers already recognise.
- 208,145 raw candidates.
- MTurk filtering with a UI copied from ImageNet's: a 48-image grid per class, the same instructions ("ignore occlusions, clutter, text; avoid drawings and paintings").
- Near-duplicate removal against everything using three metrics: raw-pixel $\ell_2$, $\ell_2$ on VGG fc7 features, and SSIM.

> [!NOTE] Selection frequency ^selection-frequency
> The fraction of MTurk workers who ticked an image as belonging to its target class. It acts as a proxy for image quality/clarity, and it is the single knob that determines how hard the resulting test set is.

The crucial trick: **they seeded at least six original validation images into every MTurk task** (three from the target class, three from a nearby WordNet class), URLs obfuscated and resized to blend in. This gave them a selection-frequency distribution for the *original* images, measured by *their* worker population in 2018 — the only way to calibrate against a labeller pool that no longer exists.

**Three sampling strategies**, all from the same candidate pool, all manually verified correct:

- `MatchedFrequency` — for each class, bin the annotated original validation images into five selection-frequency buckets ($[0,0.2)$ … $[0.8,1]$), then sample 10 new images per class matching that per-class histogram. Average selection frequency 0.73 (original validation: 0.71).
- `Threshold0.7` — 10 images per class with selection frequency $\geq 0.7$. Average 0.85.
- `TopImages` — the 10 highest-frequency images per class. Average 0.93.

Each set is 10,000 images. The authors then hand-reviewed every image over **33 iterations**, removing wrong labels, drawings, blurry or occluded images, and things visibly off-distribution (they caught a batch of *statues* of great white sharks). Critically, no model was run on any of this until the dataset was frozen.

Testbed: 34 CIFAR-10 models (cuda-convnet through AutoAugment PyramidNet, plus random-features baselines) and 67 ImageNet models (AlexNet through PNASNet, plus Fisher Vector models). Pre-trained weights from public repos wherever possible — which conveniently makes the models independent of the new data.

## Ablation Studies and Experiments

**Headline numbers, `MatchedFrequency`:**

| Model | Orig top-1 | New top-1 | Gap |
|---|---|---|---|
| pnasnet_large_tf | 82.9 | 72.2 | 10.7 |
| resnet152 | 78.3 | 67.0 | 11.3 |
| densenet161 | 77.1 | 65.3 | 11.8 |
| vgg19_bn | 74.2 | 61.9 | 12.3 |
| alexnet | 56.5 | 44.0 | 12.5 |
| fv_64k (Fisher Vector) | 35.1 | 24.1 | 11.0 |

CIFAR-10: AutoAugment PyramidNet 98.4 → 95.5 (gap 2.9); ResNet-110 93.5 → 85.2 (gap 8.3); 256k random features 85.6 → 73.1 (gap 12.5). Rank changes are tiny — mostly 0, occasionally $\pm$3.

**The single most informative result — sampling strategy is everything:**

| Strategy | Avg selection freq. | Δ top-1 | Δ top-5 |
|---|---|---|---|
| `MatchedFrequency` | 0.73 | **−11.8%** | −8.2% |
| `Threshold0.7` | 0.85 | −3.2% | −1.2% |
| `TopImages` | 0.93 | **+2.1%** | +1.8% |

Same source, same pipeline, same manual correctness check, all with average selection frequency above 0.7 — and the accuracy swings by 14% top-1. For scale: excluding AlexNet and the SqueezeNets, the *entire spread of every convnet ever tested* is about 14% top-1. The choice of annotation threshold matters as much as a decade of architecture research.

Stratifying `MatchedFrequency` by frequency bin (Fig. 15) shows images in the $[0.4, 0.6)$ bucket cost ~20% accuracy relative to the whole new set and ~30% relative to the original validation set — even though every one of those images was hand-verified to have the correct label.

**Things that did not work — the hypotheses they killed:**

- **Statistical error.** New CIFAR-10 CI is $\pm1.2\%$; original is $\pm0.6\%$. Nowhere near an 8% gap.
- **Near-duplicate removal.** They found ~800 images in the *original* CIFAR-10 test set that they would have rejected as near-duplicates of training images (8% of the test set), on which models score ~100%. Solving $0.93 = 0.92\cdot\text{acc}_{\text{true}} + 0.08\cdot 1.0$ gives $\text{acc}_{\text{true}} \approx 0.924$. That explains **0.6%**, not 8%.
- **Hyperparameter re-tuning.** A full grid over learning rate $\{0.0125 \ldots 0.8\}$, dropout scale $\{0.5 \ldots 1.75\}$, weight decay $\{5\times10^{-5} \ldots 5\times10^{-3}\}$ on `vgg16_keras`. Best improvement on the new set: 85.3% → **85.8%**. Nothing.
- **Cross-validation.** Five class-balanced folds of CIFAR-10 gave 93.5–94.0 for `vgg_15_BN_64` (original test: 93.6) versus 84.9 on the new set. On ImageNet, five ResNet-50 folds carved out of the training set gave 92.55–92.62 top-5 (original validation: 92.5) versus 84.7 new. Cross-validation is completely blind to this failure.
- **Training on half the new test set.** Adding ~1,000 new-test images to CIFAR-10 training and testing on the held-out half: 85.1% and 85.4%, versus 84.9% without. Essentially no transfer — either the shift is tiny, or the model cannot learn it.
- **Training a discriminator.** ResNet-32/110 trained to classify "original test image vs. new test image" reached at best **53.1%**, with confidence intervals overlapping 50%. A convnet cannot even tell the two sets apart, yet loses 8 points of accuracy on one of them.
- **Class imbalance.** The new CIFAR-10 set was 8% `ship` instead of 10%. An exactly balanced rebuild moved weak models by ~0.3% and the best model not at all.
- **Human accuracy.** Nine grad students labelled the hardest ~5% of images from each set (500 original, 115 new). Average difference: **−0.8%** — humans do slightly *better* on the new images. Four participants scored higher on the original, five on the new. The images are harder for models, not for people.
- **Their own reviewing.** Reviewing *raised* ResNet-50 accuracy by ~4% on `MatchedFrequency` (it removes bad images). And the linear fit is already present in *revision 1*, before any reviewing. So the manual pass created neither the drop nor the line.
- **Adversarial robustness.** The standard $\ell_\infty$-robust CIFAR-10 model from Madry et al. lands almost exactly on the same line as everyone else. Adversarial robustness buys nothing here.

## Worth Remembering

**The probit model.** A straight line in raw accuracy is a bad fit over a 60-point range. Applying $\Phi^{-1}$ (inverse Gaussian CDF) to both axes gives an excellent fit across *all* 67 models including Fisher Vectors:

$$\Phi^{-1}(\alpha_{\text{new}}) = u\cdot\Phi^{-1}(\alpha_{\text{orig}}) + v$$

The authors sketch a generative story. Give each image a difficulty $\tau \sim \mathcal{N}(\mu, \sigma^2)$ and each model a skill $s_j$, with $P(\text{correct}) = \Phi(s_j - \tau)$. Then

$$\alpha_{j,\mu,\sigma} = \mathop{\mathbb{E}}_{\tau}[\Phi(s_j - \tau)] = \Phi\!\left(\frac{s_j - \mu}{\sqrt{\sigma^2+1}}\right)$$

so on the probit scale accuracy is just $(s_j - \mu)/\sqrt{\sigma^2+1}$, and two test sets with parameters $(\mu_1,\sigma_1)$, $(\mu_2,\sigma_2)$ are related linearly with $u = \sqrt{\sigma_1^2+1}/\sqrt{\sigma_2^2+1}$ and $v = (\mu_1-\mu_2)/\sqrt{\sigma_2^2+1}$. A single scalar "difficulty" axis plus a single scalar "skill" per model reproduces the entire picture. The authors explicitly say this is illustrative, not the true mechanism.

**What they admit they cannot rule out.** A *constant* adaptivity penalty applied equally to all models would be invisible in their plots — you'd still see no diminishing returns. Detecting that needs a genuinely i.i.d. held-out set that nobody has. Two candidate reasons adaptive overfitting did not bite: Blum & Hardt's Ladder mechanism (the community implicitly limits its interaction with the test set), and a limited effective model class (only SGD-trained architectures that also fit the training set ever get evaluated, so uniform convergence may hold over that small set).

**They cannot say what makes the new images hard.** The discriminator failed, the humans did not struggle, and no succinct description of the shift emerged. Candidate hypotheses left open: object size, sepia/B&W filters, unusual poses. Notably, Fig. 15 shows they *could* have built a legitimate test set with far lower accuracies — still real Flickr photos, still correctly labelled, still selected by >50% of workers.

**Contrast with related work.** This is not Torralba & Efros's cross-dataset transfer (different datasets, big shifts expected) and it is not [[Shortcut Learning in Deep Neural Networks]]-style corruption benchmarks like ImageNet-C, where a *deliberate, specified* perturbation (Gaussian noise, synthetic snow) is applied to existing images. Here nothing was perturbed. The images are new photographs from the same well, and that alone costs 11 points.

**Practical caveats.**
- Do not trust a single test set to certify a model. Cross-validation will not catch this — it was measured and it did not.
- Rankings survive. If you are comparing architectures, the old benchmark is still telling you the truth about *ordering*. It is lying about the *level*.
- Model selection is still worth doing. Slope $>1$ means progress on the leaderboard is real progress, and then some.
- The annotation threshold is a first-class hyperparameter of your benchmark. Reporting per-image annotator agreement should be standard.

**Their concrete suggestions for future datasets:** release the dataset-*construction code*, not just the data; run a standardised competence test on your annotators and publish it; keep a "super hold-out" test set hidden for several years so adaptive overfitting can be measured directly later; prefer tasks with fewer, cleaner classes but more visual variety, since ImageNet's 1,000 fine-grained classes force reliance on high-agreement (i.e. easy) images; and consider expert annotation for test sets specifically, since they are small.

**Follow-up questions.** Does the linear/probit relationship hold for NLP benchmarks, or for Kaggle leaderboards where adaptive overfitting *should* appear? Does the slope stay $>1$ as models approach 100% (the `TopImages` top-5 slope was 0.88 — possibly a ceiling effect, possibly real overfitting)? And does anything close the gap: heavier augmentation, robust optimisation, or just harvesting more hard images?

## Links

Related: [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Understanding Deep Learning Requires Rethinking Generalization]] · [[Shortcut Learning in Deep Neural Networks]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[On the Difficulty of Evaluating Baselines]] · [[Towards Quantifying Benchmark Optimization in ASR Models]] · [[Reconciling Modern ML Practice and the Bias-Variance Trade-off]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Delving Deep into Rectifiers (He init, PReLU)]] · [[Random variable]] · [[Uncertainty]] · [[The Bitter Lesson (essay)]]

New topics worth writing: Clopper-Pearson confidence intervals, the Ladder mechanism for leaderboards, ImageNet-C and common corruption benchmarks, probit regression, distribution shift taxonomy (covariate vs. concept vs. natural shift), effective model class and uniform convergence, structural similarity (SSIM), dataset bias and the Torralba–Efros name-that-dataset game
