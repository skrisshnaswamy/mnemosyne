---
title: "Symbolic Discovery of Optimization Algorithms (Lion)"
authors: ["Xiangning Chen", "Chen Liang", "Da Huang", "Esteban Real", "Kaiyuan Wang", "Yao Liu", "Hieu Pham", "Xuanyi Dong", "Thang Luong", "Cho-Jui Hsieh", "Yifeng Lu", "Quoc V. Le"]
year: 2023
arxiv: "2302.06675"
url: https://arxiv.org/abs/2302.06675
priority: Low-Priority
read_on: 2026-09-30
tags: [paper, transformers, llm, optimization, self-supervised, diffusion, vision]
---
## The Core Idea

Instead of a human inventing an optimizer, let a computer **search for the program** that is the optimizer.

The search space is literal code. A candidate optimizer is a short function `train(w, g, m, v, lr)` that returns an `update` to subtract from the weights, plus whatever it wants to keep in two slots of memory. The search is allowed to write any sequence of statements from 45 maths functions. The starting point of the search is AdamW written out as such a program.

What came out is **Lion** — *Evo**L**ved S**i**gn M**o**me**n**tum*. Two lines:

$$c_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad \theta_t = \theta_{t-1} - \eta_t\big(\mathrm{sign}(c_t) + \lambda\theta_{t-1}\big)$$
$$m_t = \beta_2 m_{t-1} + (1-\beta_2) g_t$$

with $\beta_1 = 0.9$, $\beta_2 = 0.99$.

Two things are strange about it, and both matter.

**One: every parameter moves by exactly the same amount.** The `sign` throws away magnitude entirely. The update is $\pm 1$ elementwise (before weight decay). This is not an [[Adam- A Method for Stochastic Optimization|adaptive]] method at all — there is no per-parameter step size, no second moment, no $\epsilon$. That contradicts a decade of received wisdom that adaptivity is what makes Transformers trainable.

**Two: the momentum used for the step is not the momentum that is stored.** Adam and SGD keep one running average and use it. Lion keeps a slow average ($\beta_2 = 0.99$, about a 100-step window) but *steps* along a blend of that average with the fresh gradient ($\beta_1 = 0.9$). So it remembers roughly 10× longer than usual while still weighting the current gradient heavily. This decoupling is the piece no earlier optimizer search could have found — earlier searches ([Bello et al. 2017](https://arxiv.org/abs/1709.07417)) fixed the operands to "gradient and momentum" and searched only a small expression tree on top, so they could never change *how momentum is tracked*.

> [!NOTE] Sign update
> The step direction is $\mathrm{sign}$ of a momentum blend, so its magnitude is uniform across all coordinates. Consequence: the update has a much larger norm than Adam's or SGD's, so the learning rate must be 3–10× smaller and the decoupled weight decay 3–10× larger to keep $lr \cdot \lambda$ constant. ^sign-update

What it unlocks, practically: **one optimizer state instead of two.** No second moment means half the optimizer memory. ViT-B/16 at batch 4096 needs 16 TPU v4 chips under AdamW and 8 under Lion. Runtime is also 2–15% faster per step because there is less arithmetic.

And the headline accuracy: 88.3% zero-shot ImageNet with BASIC-L (up from 85.7% with Adafactor), which was +2.0% over the prior state of the art.

## The Methodology

### The search space

A program is a list of assignment statements over n-dimensional arrays, Python/JAX-flavoured. Inputs: weight `w`, gradient `g`, learning-rate schedule value `lr`, and two zero-initialised extra slots `v1`, `v2`. Output: `update`, plus the new slots. Same signature as AdamW, so any discovered algorithm has memory footprint ≤ AdamW by construction.

45 functions available: unary maths (`sqrt`, `sign`, `tanh`, `arcsin`, `cosh`, …), binary (`+`, `*`, `power`, `maximum`, …), plus a few optimizer-flavoured ones — `norm`, `global_norm`, `dot`, `cosine_sim`, `clip_by_global_norm`, and `interp(x, y, a)` = `(1-a)*x + a*y`. Conditionals, loops and user-defined functions were tried and did not help, so they were dropped.

Three mutations: insert a random statement, delete a statement, or change one argument of a statement. Constants are mutated either by resampling from $\mathcal{N}(0,1)$ or multiplying by $2^a$, $a \sim \mathcal{N}(0,1)$. Those constants *are* the hyperparameters — learning rate, weight decay, $\beta$s all emerge as mutated numbers.

Redundant statements are allowed to survive, deliberately. A mutation only touches one line, so dead code is the scaffolding for a future multi-step change.

The space is infinite. Roughly $n_p = n_f^{\,l} \, n_v^{\,n_a \cdot l}$ programs for length $l$. Worse, good programs are vanishingly rare: a random search over **2 million** programs on the cheap proxy task never beat AdamW.

### The search itself

Regularized evolution. Population 1000, tournament size 2: pick 2 at random, the better one is the parent, copy-and-mutate to make a child, add the child, evict the oldest.

Two departures from textbook evolution:

- **Warm start** from AdamW rather than random programs.
- **Two kinds of restart.** Restart from AdamW again (parallel runs → different local optima, encourages exploration). And restart from the best-so-far when fitness plateaus at ~300K programs (exploitation). The second restart visibly lifts the plateau.

Fitness variance between runs is high, which is *why* the restarts are needed, not a nuisance.

### Abstract execution — the piece that makes it affordable

Before a program is ever run on real data, it is "executed" symbolically, with inputs and functions swapped for stand-in values. Three jobs:

1. **Type/shape inference.** Propagate shapes statement by statement; reject programs with mismatches. Without this the search drowns in invalid programs and makes no progress at all.
2. **Functional hash.** Assign each input and function a hash, combine them along the dataflow, hash the outputs. Two programs with different surface text but the same computation get the same hash. Cache hit rate reaches **89.1 ± 0.6%** → ~10× less compute.
3. **Dependency tracking.** Each variable carries the set of statements it depends on. Anything the outputs do not depend on is dead. **69.8 ± 1.9%** of statements are dead by the end → programs are ~3× shorter to read.

Cost is negligible next to real evaluation.

### Proxy tasks and the generalisation gap

Proxy for vision: a 3-layer, 96-hidden, 3-head ViT on 10% of ImageNet, 30K steps, batch 64, 64×64 images. Proxy for language: 2-layer, 128-hidden Transformer on LM1B, 20K steps, batch 64, sequence 32. Each evaluation: ≤20 min on one TPU v2 chip. Fitness = validation accuracy or perplexity.

Each search run: 100 TPU v2 chips, ~72h, 200–300K programs generated but only 20–30K actually evaluated (thanks to the cache). Total across five runs plus the restart round: **~3000 TPU v2 days**.

The target tasks are $>10^4\times$ larger. So there is a second selection stage:

- **Funnel selection.** Programs that pass on proxy A go to meta-validation task B (10× bigger), survivors go to task C (100× bigger). Filters out anything that only worked small.
- **Meta-overfitting as a signal.** Search fitness keeps climbing while meta-validation declines. Across 50 runs, the runs that meta-overfit *later* produced programs that generalised better — a usable heuristic for which run to trust.

### From raw program to Lion

The raw discovered program has 23 statements. After dead-code removal it becomes 13. Then manual simplification:

- `cosh(update)` written into `m` is deleted — `m` gets overwritten next iteration anyway.
- `arcsin(g)` and `clip(g, lr)` deleted — no measurable quality drop.
- Three statements — `m2 = m*m`, `abs_m = sqrt(m2)`, `update = m/abs_m` — are exactly `sign(m)`.
- Two `interp`s with factors ~0.9 and ~1.109 collapse into one EMA with factor ~0.99, so the second slot `v` never needs to be stored.
- Bias correction is dropped: it scales the update but does not change its direction, and `sign` discards scale.

### Why it might generalise

Two arguments, both empirical.

The `sign` injects noise into the update, which acts as regularisation. ViT-B/16 on ImageNet: Lion reaches **higher training loss** than AdamW ($L_{train}$ 0.75 vs 0.61) but **+1.96%** validation accuracy.

And the minimum is flatter. Measuring $L^{\mathcal{N}}_{train} = \mathbb{E}_{\epsilon\sim\mathcal{N}}[L_{train}(w+\epsilon)]$ over 1000 random perturbations: **1.37** for Lion vs **3.74** for AdamW. Lower training error, much worse under perturbation — the classic [[On Large-Batch Training for Deep Learning- Generalization Gap and Sharp Minima|sharp vs flat]] picture.

## Ablation Studies and Experiments

### Image classification from scratch (ImageNet)

| Model | AdamW | Lion | Δ |
|---|---|---|---|
| ResNet-50 (SGD 76.22) | 76.34 | 76.45 | +0.11 |
| Mixer-S/16 | 69.26 | 69.92 | +0.66 |
| Mixer-B/16 | 68.12 | 70.11 | **+1.99** |
| ViT-S/16 | 76.12 | 76.70 | +0.58 |
| ViT-B/16 | 75.48 | 77.44 | **+1.96** |
| ViT-S/16 + RandAug/Mixup | 78.89 | 79.46 | +0.57 |
| ViT-B/16 + RandAug/Mixup | 80.12 | 80.77 | +0.65 |
| CoAtNet-3 + aug | 84.45 | 84.87 | +0.42 |

The pattern: gain grows with capacity, grows with fewer inductive biases (Mixer > ViT > ResNet), and **shrinks when strong augmentation is on**. Both augmentation and `sign` are doing regularisation, so they partly substitute.

### Pre-training compute savings (JFT-300M → ImageNet)

ViT-L/16 trained with Lion matches ViT-H/14 trained with AdamW on ImageNet and ImageNet-V2 at **3× less pre-training compute**; on ImageNet ReaL, **5×**. A ViT-L/16 trained with AdamW for 4M steps still loses to the same model trained with Lion for 1M steps.

After fine-tuning at higher resolution, ViT-L/16 with Lion: 88.50 ImageNet / 81.13 V2 / 58.80 A / 72.49 R, vs AdamW's 87.72 / 79.80 / 52.72 / 66.95. Note the gaps on the hard robustness sets — **+6.08 on ImageNet-A**, +5.54 on R. Scaling to JFT-3B, ViT-g/14 with Lion (1.04B params) beats the published ViT-G/14 with Adafactor (1.88B params).

### Vision-language contrastive

LiT zero-shot ImageNet: +1.10 (B/32-B), +1.13 (B/16-B), +0.66 (g/14-L). Retrieval on Flickr30K improves more at Recall@1 than Recall@10 (+1.70 vs +0.60 for image→text), so the top hit is genuinely better rather than the list being reshuffled.

BASIC-L, Lion applied to *both* the vision-tower pre-training and the contrastive stage: **88.3%** zero-shot (from 85.7%), 91.1% fine-tuned. Consistent on V2, A, R, Sketch, ObjectNet.

### Diffusion

ImageNet unconditional, improved [[U-Net]], DDPM 1K steps, no guidance. At 256×256, Lion hits AdamW's final FID at 440K steps — **2.3× fewer iterations** — and finishes at FID **4.1 vs 4.7**. The advantage grows with resolution.

### Language

Wiki-40B and PG-19, Transformers 110M/336M/731M: Lion gives 1.6× and 1.5× speedup at medium size, rising to **2× on PG-19 at large size**. Masked LM on C4: 4.18 vs 4.25 (small), 3.42 vs 3.54 (medium), 3.18 vs 3.25 (large) — real but small.

Large-scale autoregressive (1.6T-token internal corpus, 1.1B–7.5B params, 300B tokens): **no perplexity difference at all throughout training.** Yet one-shot downstream scores improve — NLG exact-match +1.0/+0.9/+0.6 at 1.1B/2.1B/7.5B, NLU +0.7/+0.6/+0.4. The 7.5B baseline at 300B tokens already beats 8B PaLM at 780B tokens, so the baseline is honest.

T5 fine-tuning on GLUE: Lion wins 10/12, 12/12, 10/12 scores at Base, Large, 11B.

### Against other optimizers

ViT-S/16 and ViT-B/16 on ImageNet with augmentation, all with $lr$ and $\lambda$ tuned:

| | AdamW | RAdam | NAdam | AdaBelief | AMSGrad | PowerSign | AddSign | Lion |
|---|---|---|---|---|---|---|---|---|
| ViT-S/16 | 78.89 | 78.59 | 78.91 | 78.71 | 79.01 | 77.36 | 77.37 | **79.46** |
| ViT-B/16 | 80.12 | 80.26 | 80.32 | 80.29 | 79.85 | 78.95 | 78.50 | **80.77** |

No winner among the five adaptive baselines — AMSGrad is best on S/16 and worst on B/16. Their learning curves are near-identical; Lion's is visibly different and faster. The two earlier AutoML optimizers (PowerSign, AddSign) are the worst two, which is the honest reference point for "discovered optimizers don't transfer".

### The ablation that justifies two $\beta$s

Collapse Lion to a single momentum: `m = interp(g, m, β); update = sign(m)`.

| | AdamW | Ablation$_{0.9}$ | Ablation$_{0.99}$ | Lion |
|---|---|---|---|---|
| ViT-S/16 | 78.89 | 78.23 | 78.19 | **79.46** |
| ViT-B/16 | 80.12 | 79.54 | 79.90 | **80.77** |

Both single-$\beta$ variants lose to *every* baseline, including plain AdamW. So `sign` alone is not the trick — signSGD, which is essentially this, was reported to underperform on ConvNets. **The decoupling of "what is stored" from "what is stepped along" is load-bearing.** Same conclusion on PG-19 at all three model sizes.

### Batch size

ViT-B/16, fixed 300 epochs, varying batch. Optimal batch for AdamW is **256**; for Lion it is **4096**. At batch 32K (only 11K steps) Lion gets **77.9% vs 75.4%** — a 2.5% gap. Below batch 64 Lion is no longer reliably better.

### What did not work

- **Random search**: 2M programs, best still worse than AdamW.
- **Hyperparameter-only evolution** (mutate constants of AdamW), given 4× the compute: beaten clearly.
- **Conditionals, loops, function definitions** in the search space: no improvement, removed.
- **No type/shape filtering**: search cannot make progress, drowns in invalid programs.
- **`arcsin` and `clip` in the discovered program**: removable with no quality loss — i.e. the search produced decoration.
- **Imagen base 64×64 text-to-image**: no clear improvement. Only the super-resolution stage benefits.
- **Large-scale autoregressive LM perplexity**: flat. The authors say plainly that perplexity is arguably the more reliable metric than in-context benchmarks, so this is a genuine null.
- **Masked LM on C4**: differences small.
- **ResNets**: essentially a tie with SGD and AdamW.

The unifying story for the nulls: when the dataset is **massive and high-quality**, optimizer choice stops mattering. Regularisation is worth little when you are nowhere near overfitting.

## Worth Remembering

**Tuning rules, concretely.** Lion's $lr$ should be 3–10× *smaller* than AdamW's, and $\lambda$ 3–10× *larger*, because effective decay is $lr \cdot \lambda$. Scale the initial, peak and final learning rate by the same ratio. Do not touch the schedule shape or clipping. Worked examples:

| Task | Lion | AdamW/Adafactor |
|---|---|---|
| ViT-B/16 ImageNet + aug | $lr=1e{-4}$, $\lambda=10.0$ | $lr=1e{-3}$, $\lambda=1.0$ |
| Diffusion | $lr=3e{-5}$, $\lambda=0.1$ | $lr=3e{-4}$, $\lambda=0.01$ |
| 7.5B LM | $lr=1e{-4}$, $\lambda=0.01$ | $lr=1e{-3}$, $\lambda=0.001$ |

For language tasks both optimizers benefit from lowering $\beta_2$ — Lion uses $\beta_1=0.95, \beta_2=0.98$ there. Smaller $\beta_2$ = shorter memory = more training stability.

**Robustness to hyperparameters.** The $lr \times \lambda$ heatmap for Lion is flatter than AdamW's. That matters more than peak numbers for whether anyone actually adopts an optimizer.

**Lion still needs bf16 momentum**, which is expensive at giant scale. The authors suggest factorising the momentum (à la Adafactor) as future work.

**Limitations the authors own.** The search space, despite the effort, is still biased towards first-order methods that look like Adam — there are no primitives for second-order methods (Shampoo, K-FAC). Simplification from raw program to Lion required a human. The search cost (~3000 TPU v2 days) rules out casual replication.

**Other programs came out of the same machinery** and are in Appendix D. Shrink the proxy dataset and you get programs with *better regularisation* — one computes `dot(g, w)` and uses it as a dynamic weight-decay coefficient. Stop the search early and you get near-AdamW variants, including one equivalent to AdaGrad and one that tracks the second moment of $(g - m)$, i.e. AdaBelief rediscovered. Task-specialised optimizer search is an obvious open direction.

**The connection worth drawing.** Lion sits in a family with [[Old Optimizer, New Norm- An Anthology (Muon)|Muon]]: both replace Adam's per-coordinate scaling with a *normalised* direction — Lion via elementwise sign (steepest descent under the $\ell_\infty$ norm), Muon via orthogonalisation (steepest descent under the spectral norm). Lion is arguably the empirical evidence that made that theoretical framing worth pursuing.

**Questions it leaves open.** Why does 10× longer momentum memory plus a 0.9 blend beat a single average? No theory given — it's a search result. Does the batch-size preference come from the noise added by `sign` needing averaging out, and if so can you predict the crossover? And since the gains vanish on huge clean datasets, is Lion a regulariser dressed as an optimizer?

It is deployed in production on Google's search-ads CTR model, which is the most useful single fact about whether it survives contact with real systems.

## Links

Related: [[Adam- A Method for Stochastic Optimization]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Momentum]] · [[On Large-Batch Training for Deep Learning- Generalization Gap and Sharp Minima]] · [[Accurate, Large Minibatch SGD- Training ImageNet in 1 Hour]] · [[Regularization]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Denoising Diffusion Probabilistic Models]] · [[CLIP]] · [[Old Optimizer, New Norm- An Anthology (Muon)]] · [[Cyclical Learning Rates for Training Neural Networks]] · [[Mixed Precision Training]] · [[The Bitter Lesson (essay)]] · [[Exploring the Limits of Transfer Learning (T5)]] · [[Tensor Programs V- Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (muP)]]

New topics worth writing: signSGD and sign-based optimization, Regularized evolution / genetic programming, AutoML-Zero, Learning to optimize (L2O), Adafactor and factored second moments, Sharpness-aware minimization, Loss landscape flatness measurement, Symbolic program search and functional hashing, Neural optimizer search, Steepest descent under non-Euclidean norms
