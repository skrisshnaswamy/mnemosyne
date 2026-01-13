---
title: "Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)"
authors: ["He et al."]
year: 2014
url: http://quinonero.net/Publications/predicting-clicks-facebook.pdf
priority: Must-Read
read_on: 2026-08-22
tags: [paper, transformers]
---
## The Core Idea

Facebook has ~750M daily users and ~1M advertisers. Every page visit triggers a request for ads, and the last stage of the ranking cascade must output a well-calibrated probability that this user clicks this ad. Calibrated matters because the ad auction multiplies predicted CTR by the bid — a ranking that is correctly ordered but numerically wrong under-delivers or over-delivers ads.

The trick the paper is famous for: **use gradient-boosted decision trees as a feature transformer, not as the final predictor.** Train a boosted tree ensemble. Then throw away the leaf *values*. For each tree, record only *which leaf* the example fell into, and one-hot encode that leaf index. Concatenate over all trees to get a sparse binary vector. Feed that vector to a plain [[Regression Analysis|logistic regression]] trained online with SGD.

Why this is clever: a linear model over raw features cannot express "user is on mobile AND ad is from a retail advertiser AND ad CTR last week > 2%". You would have to hand-build those crosses. A path from root to leaf in a decision tree *is* exactly such a conjunction rule, and boosting discovers useful ones automatically by greedily minimising loss. So the trees do **supervised feature engineering** — they convert real-valued and categorical inputs into a compact binary "which rules fired" vector — and the linear layer just learns a weight per rule.

The second half of the insight is a systems one. Trees are slow to train (a few hundred trees over hundreds of millions of rows can take >24 hours on one core) but the world changes daily. So split the model by *update frequency*: retrain the trees in batch every day or two, and train the linear layer **online, continuously**, from a live stream of impressions joined to clicks. Freshness where it's cheap, expressiveness where it's expensive.

> [!NOTE] GBDT feature transform
> Treat each tree in a boosted ensemble as a categorical feature whose value is the index of the leaf the example reaches. 1-of-K encode it. Two trees, one with 3 leaves and one with 2 leaves, example lands in leaf 2 and leaf 1 → `[0,1,0,1,0]`. ^gbdt-feature-transform

## The Methodology

**Metric — Normalized Entropy (NE).** This is the paper's other lasting contribution. Raw log loss is not comparable across datasets, because if the background click rate is very close to 0 it is trivially easy to get a low log loss. So divide the model's average log loss by the log loss of a constant predictor that always outputs the background CTR $p$:

$$\text{NE} = \frac{-\frac{1}{N}\sum_{i=1}^{N}\left(\frac{1+y_i}{2}\log p_i + \frac{1-y_i}{2}\log(1-p_i)\right)}{-\left(p\log p + (1-p)\log(1-p)\right)}$$

with $y_i \in \{-1,+1\}$. Lower is better. The numerator is [[Cross Entropy|cross-entropy]] of the model, the denominator is the entropy of the base rate. Relative Information Gain is just $\text{RIG} = 1 - \text{NE}$.

> [!NOTE] Normalized Entropy
> Log loss divided by the log loss of the "always predict base rate" model. Makes loss numbers comparable across datasets with wildly different click rates. Unlike AUC, it punishes miscalibration: over-predicting by 2× and then applying a 0.5 global fix improves NE but leaves AUC unchanged. ^normalized-entropy

**Calibration** is reported alongside: ratio of average predicted CTR to observed empirical CTR. Want 1.0.

**Trees.** Friedman's Gradient Boosting Machine with classic L2-TreeBoost — each new tree fits the residual of the ones before it. Capped at 12 leaves per tree, up to ~600 trees in the production-ish configuration.

**Linear layer.** After the transform, an impression is a structured vector $x = (e_{i_1},\dots,e_{i_n})$ of unit vectors picking out active categorical values. The score is

$$s(y,x,w) = y \cdot w^\top x = y\sum_{j=1}^{n} w_{j,i_j}$$

Logistic regression: $p(y|x,w) = \text{sigmoid}(s)$, updated per example by SGD:

$$w_{i_j} \leftarrow w_{i_j} + y\cdot \eta_{i_j}\cdot g(s), \qquad g(s) = \tfrac{y(y+1)}{2} - y\,\text{sigmoid}(s)$$

They compare five learning-rate schedules (see ablations). The winner is **per-coordinate**, i.e. AdaGrad-style:

$$\eta_{t,i} = \frac{\alpha}{\beta + \sqrt{\sum_{j=1}^{t}\nabla_{j,i}^2}}, \qquad \alpha = 0.1,\ \beta = 1.0$$

Each feature gets its own step size that shrinks with how much gradient it has already seen. Rates are floored at $10^{-5}$ so continuous learning never fully stops.

**Baseline competitor: BOPR** (Bayesian Online Probit Regression, from Microsoft Bing). Instead of a point weight, keep a Gaussian $\mathcal{N}(\mu_k, \sigma_k^2)$ per weight and update by expectation propagation with moment matching:

$$\mu_{i_j} \leftarrow \mu_{i_j} + y\cdot\frac{\sigma^2_{i_j}}{\Sigma}\cdot v\!\left(\frac{s(y,x,\mu)}{\Sigma}\right), \quad \sigma^2_{i_j} \leftarrow \sigma^2_{i_j}\left[1 - \frac{\sigma^2_{i_j}}{\Sigma^2}w\!\left(\frac{s(y,x,\mu)}{\Sigma}\right)\right]$$

with $\Sigma^2 = \beta^2 + \sum_j \sigma^2_{i_j}$. Note the shape: the mean update *is* per-coordinate SGD where the step size is set by the posterior variance — [[Uncertainty|uncertainty]] about a weight acts as its learning rate. High uncertainty → big step.

**The online joiner.** Infrastructure that builds the live training stream. Clicks are explicit; "no click" is not — there's no no-click button. So an impression is labelled negative if no click arrives within a fixed waiting window. Implementation is a distributed stream-to-stream join on `request ID`, using a **HashQueue**: a FIFO queue as the time buffer plus a hash map for O(1) lookup to attach a click to a buffered impression. Only after the window expires is the labelled example emitted to the trainer, which publishes fresh models back to the ranker.

The window length is a real tradeoff. Too long → stale data and huge memory for buffered impressions. Too short → late clicks get lost and are mislabelled negative, so empirical CTR is biased *downwards*. In practice they got the bias down to fractions of a percent, and it can be measured and corrected.

They also flag a failure mode worth internalising: if the click stream goes stale from an infra bug, the joiner emits near-zero CTR data, the online learner immediately learns to predict ~0, ad expected values collapse, and the system stops showing ads. Fix: anomaly detection that auto-disconnects the trainer from the joiner when the data distribution jumps. This is the same class of problem as feedback loops in [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]].

**Downsampling and recalibration.** Negative downsampling at rate $w$ distorts the predicted probability. To get back to real space:

$$q = \frac{p}{p + (1-p)/w}$$

e.g. base CTR 0.1%, downsample negatives at $w=0.01$ → empirical CTR in training becomes ~10%, and this formula maps predictions back.

## Ablation Studies and Experiments

Data: one arbitrary week of Q4 2013 Facebook ads, held fixed across all experiments. Train on one day, test on the next.

**The headline (Table 1), NE relative to trees-only:**

| Model | NE |
|---|---|
| LR + Trees | **96.58%** |
| LR only | 99.43% |
| Trees only | 100% (ref) |

A 3.4% relative NE drop. For calibration of expectations: a typical feature engineering experiment moves NE by a couple of tenths of a percent. The striking part is that LR alone and trees alone are *comparable* — neither is the winner. The lift is entirely from the combination.

**Data freshness (Fig 2).** Train on day $d$, test on days $d{+}1 \dots d{+}6$. Accuracy decays monotonically for both trees-only and LR+trees. Going from weekly to daily retraining buys ~1% NE. This is what justifies the whole online-learning architecture.

**Learning rate schedules (Fig 3).** Five options, all tuned by grid search:

| Schedule | Result |
|---|---|
| Per-coordinate $\alpha/(\beta+\sqrt{\sum \nabla^2})$ | **best** |
| Per-weight sqrt $\alpha/\sqrt{n_{t,i}}$ | slightly worse |
| Constant $\alpha$ | slightly worse |
| Global $\alpha/\sqrt{t}$ | much worse |
| Per-weight $\alpha/n_{t,i}$ | worst, ~5% NE above best |

The failure diagnosis is the useful part. **Global** fails because features appear at wildly different frequencies — a rare feature's rate is decayed by everyone else's updates and dies before that feature ever converges. **Per-weight $1/n$** fixes the frequency imbalance but decays far too fast for *every* feature, so training terminates early at a sub-optimal point. Only $1/\sqrt{\cdot}$ decay per coordinate gets both right. Compare against how [[Momentum|momentum]] and adaptive methods handle the same problem.

**LR vs BOPR (Table 3).** BOPR NE = 99.82%, LR = 100%. Essentially a tie — unsurprising given the update equations are structurally the same. LR wins on cost: half the model size (one weight per feature vs a mean and a variance), better cache locality, one inner product at prediction instead of two. BOPR wins on giving a full predictive distribution over click probability, which you can use for explore/exploit (e.g. Thompson sampling).

**Number of trees (Fig 5).** Swept 1 → 2000, max 12 leaves. Diminishing returns: nearly all the NE gain arrives in the first 500 trees; the last 1000 trees buy <0.1%. And "submodel 2" **regresses** past 1000 trees — overfitting, because its training set is 4× smaller than submodels 0 and 1. So more trees is not monotonically safe when data is thin.

**Feature count (Figs 6–7).** Using *Boosting Feature Importance* — the summed squared-error reduction attributable to a feature across all split points in all trees — the distribution is brutally skewed: top 10 features carry ~half the total importance; the bottom 300 features carry <1% combined. Retraining with only the top 10/20/50/100/200 features shows the same diminishing-returns curve. You can cut features aggressively and lose only a little.

**Historical vs contextual features (Table 4, Figs 8–9).** Historical = accumulated behaviour (ad's CTR last week, user's average CTR). Contextual = right-now state (time of day, device, current page). NE relative to contextual-only:

| Features | NE |
|---|---|
| All | 95.65% |
| Historical only | 96.32% |
| Contextual only | 100% (ref) |

Dropping contextual features costs <1%. Dropping historical features costs 4.5%. The top 10 features by importance are **all** historical; of the top 20, only 2 are contextual — despite historical being ~75% of the feature set overall, so this is more than a raw-count effect. And Fig 9 shows contextual-only models decay much faster with staleness, which fits: long-run behaviour aggregates are stable, "what's trending today" is not.

**Uniform subsampling (Fig 10).** Rates $\{0.001, 0.01, 0.1, 0.5, 1\}$. More data is better, with diminishing returns. At **10% of data, NE is only 1% worse** and calibration is unharmed. A cheap 10× training cost cut.

**Negative downsampling (Fig 11).** Rates $\{0.1, 0.01, 0.001, 0.0001\}$ swept; the optimum is **0.025**. This one genuinely matters — it is not a flat curve, and both too much and too little downsampling hurt. Requires the recalibration formula afterwards.

## Worth Remembering

- **The two "small" knobs really are small.** Freshness, learning-rate schedule, and sampling each move NE by around 1%. Model architecture (GBDT+LR) moves it 3.4%, and having historical features moves it 4.5%. The paper's own conclusion is: get the features right, then the model, and only then tune. At Facebook scale a 1% NE gain is still worth money, but the ordering is the lesson.
- **Contextual features are for cold start.** The authors flag this explicitly: for a brand new user or a brand new ad there *is* no history, and contextual features are the only thing you have. So you cannot conclude "drop contextual features" from the 1% number.
- **The GBDT+LR pattern became a standard industry baseline** and is the direct ancestor of the deep CTR line — Wide & Deep, DeepFM, and eventually attention-based user modelling like [[Deep Interest Network for CTR Prediction (DIN)]]. The trees are doing by search what those later models do by learned embeddings and crosses. Context in [[Recommender Systems - Evolution]].
- **AUC vs NE is a real design decision.** They argue explicitly for a calibration-aware metric because the downstream auction consumes the *probability*, not the rank. If your downstream system only consumes the ranking, [[NDCG]] or AUC is the right target instead.
- **Nothing is reproducible here.** They cannot reveal the feature list, the number of impressions, or absolute NE — every number is relative. This is the exact class of paper that [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] would flag, and it's worth holding both papers in mind together: this one is honest engineering reporting, not a benchmark claim.
- **What they did not try:** neural nets. In 2014 the practical choice for sparse high-cardinality ad features was still linear models plus clever feature engineering. Also absent: any treatment of position bias or the fact that the training data is generated by the model's own past decisions — a selection problem handled directly in [[Counterfactual Reasoning and Learning Systems]].
- **Practical caveat for reuse:** the leaf-index encoding means the linear layer's feature space is *defined by* the current tree ensemble. Retrain the trees and every leaf ID changes meaning, so the online-learned weights are invalidated and the linear layer must be re-warmed. The paper does not discuss how they handle this handoff, and it is the hardest part of actually shipping the architecture.
- **Follow-up question:** with per-coordinate rates and a continuously-running online learner, does the effective learning rate eventually go to zero for common features and freeze the model? The $10^{-5}$ floor is the band-aid, but this is a known long-run drift problem for AdaGrad-style rules.

## Links

Related: [[Cross Entropy]] · [[KL Divergence]] · [[Regression Analysis]] · [[Deep Interest Network for CTR Prediction (DIN)]] · [[Recommender Systems - Evolution]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Counterfactual Reasoning and Learning Systems]] · [[Momentum]] · [[Uncertainty]] · [[Regularization]] · [[NDCG]] · [[Foundational_RecSys_Ranking_Reading_Plan]]

New topics worth writing: Gradient Boosted Decision Trees, AdaGrad, Model calibration, Class imbalance and negative downsampling, Expectation Propagation, Thompson sampling, Wide & Deep, Feature importance in tree ensembles, Stream-to-stream joins
