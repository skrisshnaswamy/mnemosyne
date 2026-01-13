---
title: "Revisiting Deep Learning Models for Tabular Data (FT-Transformer)"
authors: ["Yury Gorishniy", "Ivan Rubachev", "Valentin Khrulkov", "Artem Babenko"]
year: 2021
arxiv: "2106.11959"
url: https://arxiv.org/abs/2106.11959
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, transformers, vision]
---
## The Core Idea

Tabular data — spreadsheets, rows of mixed numbers and categories — is the one place where deep learning kept losing to gradient boosted trees. Dozens of papers claimed to fix this, but each used its own datasets, its own splits, and its own tuning budget. So nobody knew which model was actually good.

This paper does two things. First, it runs everyone on the same eleven datasets with the same protocol (same splits, same Optuna budget, 15 seeds each). Second, it plants two flags for what a *baseline* should be:

1. **A ResNet for tables.** Just [[Deep Residual Learning for Image Recognition (ResNet)|residual blocks]] of linear layers with [[Batch Normalization|BatchNorm]] and dropout. Nothing clever. Once tuned, no existing "novel" tabular architecture consistently beats it. That is embarrassing for the field and useful for you.
2. **FT-Transformer.** Turn *every* feature — each number, each category — into its own embedding vector, then run a plain [[Attention Is All You Need|Transformer]] over that little sequence of features, with a `[CLS]` token for the prediction. Sequence length = number of columns. That is it.

The interesting finding is not "Transformer wins on average". It is *where* it wins. Plot the datasets by whether GBDT or ResNet does better. ResNet is good on the "DL-friendly" half and bad on the GBDT-friendly half. FT-Transformer is good on **both**. It gains almost nothing over ResNet where ResNet was already fine, and gains a lot exactly where ResNet fails. So it is a more *universal* model, not a uniformly stronger one.

And the honest conclusion: tuned CatBoost/XGBoost still win outright on California Housing, Adult and Yahoo, and the gaps are big enough that "DL has surpassed GBDT" is false. But GBDT is unusable on multiclass problems with many classes (ALOI has 1000 classes — tuning never finished).

> [!NOTE] Feature tokenizer
> Every column becomes a $d$-dimensional vector. A numeric column $j$ multiplies its scalar value by a learned vector; a categorical column looks up a row in an embedding table. Both add a learned per-column bias. This turns a table row into a sequence of $k$ tokens, which is what lets you run self-attention over *features*. ^feature-tokenizer

## The Methodology

**ResNet (the baseline they resurrect).**

$$\texttt{ResNetBlock}(x) = x + \texttt{Dropout}(\texttt{Linear}(\texttt{Dropout}(\texttt{ReLU}(\texttt{Linear}(\texttt{BatchNorm}(x))))))$$

The input goes through one `Linear` first, then $N$ of these blocks, then `Linear(ReLU(BatchNorm(·)))` for the prediction. The design rule that mattered: keep the main path clean — all normalisation lives *inside* the residual branch, never on the skip path. They tried the computer-vision block layout and the Transformer-style block layout; the Transformer-style one was equal or better.

**FT-Transformer.** Two parts.

*Part 1 — Feature Tokenizer.* For feature $j$, the token is

$$T_j = b_j + f_j(x_j) \in \mathbb{R}^d$$

- numeric: $T^{(num)}_j = b^{(num)}_j + x^{(num)}_j \cdot W^{(num)}_j$ — element-wise scalar times a learned vector.
- categorical: $T^{(cat)}_j = b^{(cat)}_j + e_j^\top W^{(cat)}_j$ — an ordinary [[Distributed Representations of Words and Phrases (negative sampling)|embedding lookup]].

Stack them into $T \in \mathbb{R}^{k \times d}$ where $k$ = number of columns.

*Part 2 — Transformer.* Prepend a `[CLS]` token (borrowed straight from [[BERT- Pre-training of Deep Bidirectional Transformers|BERT]]), run $L$ layers of standard MHSA + FFN, then

$$\hat{y} = \texttt{Linear}(\texttt{ReLU}(\texttt{LayerNorm}(T^{\texttt{[CLS]}}_L)))$$

Details that they say actually mattered:
- **PreNorm** ([[Layer Normalization|LayerNorm]] at the start of each residual branch), chosen because PostNorm needs warmup or fancy init to train, and they wanted one training recipe for all models. They note PostNorm might give better final numbers if you are willing to babysit it.
- **Remove the first LayerNorm of the first block entirely.** Necessary for good performance in the PreNorm setup.
- ReGLU activation in the FFN (a [[Gated Activation|gated]] variant), though they admit ReLU was about the same in preliminary runs.
- 8 heads, never tuned.

**Training protocol (identical for everything).** [[Cross Entropy|Cross-entropy]] for classification, MSE for regression. [[Decoupled Weight Decay Regularization (AdamW)|AdamW]] for all models except TabNet/GrowNet (Adam, per their originals). **No learning-rate schedule, no warmup, no pretraining, no augmentation, no distillation** — deliberately, so that differences come from architecture only. Early stop with patience 16. Numerical features go through scikit-learn's quantile transform by default (standardisation for Helena/ALOI, raw for Epsilon where preprocessing *hurt*). Regression targets standardised. 100 Optuna (TPE) iterations per model per dataset, 15 seeds on the winner, plus ensembles of 3×5 models.

**Default FT-Transformer** (an "educated guess", barely tuned): 3 layers, $d=192$, 8 heads, ReGLU with FFN factor $4/3$, attention dropout 0.2, FFN dropout 0.1, residual dropout 0.0, Kaiming init, lr $1\times10^{-4}$, weight decay $1\times10^{-5}$ (none on the tokenizer, LayerNorms and biases). ~929K params at 100 numeric features.

## Ablation Studies and Experiments

Eleven datasets: California Housing, Adult, Helena, Jannis, Higgs-small, ALOI, Epsilon, Year, Covertype, Yahoo, Microsoft. Metrics are RMSE or accuracy.

**Single models, average rank across all 11 (lower = better):**

| model | rank |
|---|---|
| TabNet | 7.5 |
| SNN | 6.4 |
| AutoInt | 5.7 |
| GrowNet | 5.7 |
| MLP | 4.8 |
| DCN V2 | 4.7 |
| NODE | 3.9 |
| ResNet | 3.3 |
| **FT-Transformer** | **1.8** |

Read that carefully: **a tuned MLP outranks TabNet, SNN, AutoInt and GrowNet.** The whole "explicit multiplicative interactions" line (DCN V2) lands at 4.7, basically tied with MLP. NODE is the only prior model that is genuinely competitive, and it loses to ResNet on six datasets while being far larger and ensemble-shaped internally.

**Ensembles vs GBDT (Table 4).** Ensembles of default (untuned!) FT-Transformer beat ensembles of default XGBoost and CatBoost everywhere except California Housing and Adult. Once GBDT is tuned, the picture splits:

- GBDT clearly wins: California Housing (CatBoost 0.423 vs FT-T 0.448 RMSE), Adult (0.874 vs 0.860), Yahoo (XGBoost 0.732 vs 0.747).
- FT-Transformer clearly wins: Helena (0.398 vs 0.388), Jannis (0.739 vs 0.727), Covertype (0.973 vs 0.968), Epsilon (0.8984 vs 0.8898), ALOI (0.967 vs *untunable*).

**The synthetic experiment (§5.1) — the most illuminating result.** They fix 100 Gaussian features and build a target that interpolates:

$$y = \alpha \cdot f_{GBDT}(x) + (1-\alpha)\cdot f_{DL}(x)$$

$f_{GBDT}$ = average of 30 randomly built decision trees (random split feature, random threshold $\sim\mathcal{N}(0,1)$, depth ≤ 10, 100 nodes). $f_{DL}$ = a randomly initialised, frozen 3-hidden-layer MLP. Sweep $\alpha$ from 0 to 1. At $\alpha=0$, ResNet and FT-Transformer tie and both beat CatBoost. As $\alpha\to1$, **ResNet's RMSE blows up while FT-Transformer stays flat and competitive**. So there is a real, identifiable function class — tree-like, piecewise-constant, axis-aligned targets — that self-attention over feature tokens approximates and a plain residual MLP does not.

**Architecture ablation (Table 5).** Two questions, both answered by the same table.

| | CA ↓ | HE ↑ | JA ↑ | CO ↑ | MI ↓ |
|---|---|---|---|---|---|
| AutoInt | 0.474 | 0.372 | 0.721 | 0.934 | 0.750 |
| FT-T w/o feature biases | 0.470 | 0.381 | 0.724 | 0.964 | 0.751 |
| FT-T | **0.459** | **0.391** | **0.732** | **0.970** | **0.746** |

AutoInt also tokenises features and applies self-attention — so the gain is *not* "attention on tabular features". It is the vanilla Transformer backbone plus the `[CLS]` inference plus the biases. And the **per-feature bias $b_j$ is not decoration**: dropping it costs 1.0 points of accuracy on Helena and 0.8 on Jannis. Without the bias, a numeric token is just $x_j W_j$, so all objects with $x_j$ near zero collapse to nearly the same token and the direction is fully determined by the column — the bias is what gives each feature its own anchor point in embedding space.

**Tuning-budget ablation (Table 11).** The worry: FT-Transformer got 100 Optuna *iterations* like everyone else, but each iteration is slower, so it consumed more wall-clock. Under equal *time* budgets on three datasets, FT-Transformer still wins among DL models — and it gets there fast. At 15 minutes on Higgs it has done **2** Optuna trials (Optuna's first 10 trials are random sampling) and already scores 0.727, matching the fully-tuned ResNet. Meanwhile giving MLP/ResNet/XGBoost 6 hours instead of 1 buys them essentially nothing.

**Attention maps as feature importance (§5.3).** Average the `[CLS]` attention distribution over heads, layers and samples. Rank-correlation with permutation-test importances: 0.81–0.92 across datasets, roughly matching Integrated Gradients — and beating IG badly on Year (0.92 vs 0.50) and Microsoft (0.86 vs 0.56). Cost: one forward pass, versus $n_{features}+1$ for the permutation test. Cheap and decent.

**What did not work / was left out:**
- Four extra datasets (Bank, Kick, MiniBooNE, Click) are reported in the appendix as *non-informative* — every model lands within noise. Worth knowing which public tabular benchmarks cannot discriminate.
- Quantile preprocessing **hurt** on Epsilon; they used raw features there.
- CatBoost had to be tuned on GPU (CPU too slow) but *evaluated* on CPU, because for that library version CPU inference gave meaningfully better metrics.
- NODE could not be tuned on Helena (100 classes) or ALOI (1000 classes) — the minimal non-default config has 600M+ parameters on Helena.
- XGBoost/CatBoost could not be tuned on ALOI at all.
- The "stacked" vs "parallel" DCN V2 variants made no clear difference.

## Worth Remembering

**Cost.** FT-Transformer is 1.3×–3.5× slower to train than ResNet on most datasets, and **13.8× slower on Yahoo** (699 features). MHSA is quadratic in the number of *columns*, so wide tables are the failure mode. On Epsilon (2000 features) they had to fall back to Linformer-style low-rank attention with headwise sharing and projection dim 128. The authors explicitly flag CO₂ cost and suggest [[Distilling the Knowledge in a Neural Network|distilling]] FT-Transformer into something smaller for inference. [[Flash Attention]] / efficient-attention variants are the obvious modern patch.

**The practical recipe this gives you.** If you have a tabular problem: run tuned CatBoost, and run *default* FT-Transformer (no tuning — the out-of-the-box config ensembles almost as well as the tuned one). If they disagree a lot, your target is probably tree-like or not. If you are doing anything multimodal, or have many classes, FT-Transformer is the safe deep option.

**Methodological point that generalises well beyond tables.** The reason all these architectures looked good in their own papers is that nobody tuned the baseline. This is exactly the failure mode in [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] and [[On the Difficulty of Evaluating Baselines]] and [[RoBERTa- A Robustly Optimized BERT Pretraining Approach]] — an undertrained baseline makes any new idea look like progress. Here 100 Optuna trials on an MLP is enough to beat four "novel tabular DL architectures".

**Open questions.** They deliberately excluded pretraining, LR schedules, warmup and augmentation, to isolate the architecture. So there is no claim about what FT-Transformer does with self-supervised pretraining on the unlabelled table — which is where a Transformer would be expected to pay off most. Also: the benchmark is admitted to be "slightly biased towards DL-friendly problems", and the authors' own advice is that future tabular DL work should target the datasets where GBDT still wins (California Housing, Adult, Yahoo) rather than reporting averages. Note also this paper is by the same Yandex group that produced [[CatBoost- Unbiased Boosting with Categorical Features|CatBoost]] and NODE, which makes the "GBDT still wins sometimes" conclusion more credible, not less.

Compare with [[Why do tree-based models still outperform deep learning on tabular data]], which comes at the same question from the other side and argues the culprit is rotational invariance of MLPs plus uninformative features — a story that fits the $f_{GBDT}$ synthetic result here rather neatly.

## Links

Related: [[Attention Is All You Need]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Layer Normalization]] · [[Batch Normalization]] · [[CatBoost- Unbiased Boosting with Categorical Features]] · [[XGBoost- A Scalable Tree Boosting System]] · [[LightGBM- A Highly Efficient Gradient Boosting Decision Tree (NeurIPS)]] · [[Why do tree-based models still outperform deep learning on tabular data]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Dropout- A Simple Way to Prevent Overfitting]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Practical Bayesian Optimization of Machine Learning Algorithms]] · [[On the Difficulty of Evaluating Baselines]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Deep Learning Recommendation Model (DLRM)]] · [[Delving Deep into Rectifiers (He init, PReLU)]] · [[Gated Activation]] · [[Distilling the Knowledge in a Neural Network]] · [[Flash Attention]]

New topics worth writing: Optuna and Tree-Structured Parzen Estimator, quantile transform preprocessing, NODE (neural oblivious decision ensembles), TabNet, DCN V2 / feature crossing, self-normalising networks (SELU), Integrated Gradients, permutation feature importance, Linformer and low-rank attention, PreNorm vs PostNorm Transformers, deep ensembles
