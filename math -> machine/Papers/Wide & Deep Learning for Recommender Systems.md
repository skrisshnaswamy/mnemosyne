---
title: "Wide & Deep Learning for Recommender Systems"
authors: ["Heng-Tze Cheng", "Levent Koc", "Jeremiah Harmsen", "Tal Shaked", "Tushar Chandra", "Hrishi Aradhye", "Glen Anderson", "Greg Corrado", "Wei Chai", "Mustafa Ispir", "Rohan Anil", "Zakaria Haque", "Lichan Hong", "Vihan Jain", "Xiaobing Liu", "Hemal Shah"]
year: 2016
arxiv: "1606.07792"
url: https://arxiv.org/abs/1606.07792
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, optimization, vision]
---
## The Core Idea

Two kinds of model fail in opposite directions, and this paper welds them together into one thing that is trained end to end.

A **linear model on crossed binary features** memorises. If you build the feature `AND(user_installed_app=netflix, impression_app=pandora)` and give it a weight, the model has learned one specific rule from history. It is cheap, it is interpretable, and it can hold "exception rules" — pairs that break the general pattern — with a handful of parameters. But it is blind to any pair it never saw in training. That feature is exactly zero for a new combination, forever.

A **deep net over embeddings** generalises. Each sparse ID gets a dense vector, so any user-item pair produces *some* score, even an unseen one. The cost is the opposite failure: when the user-item matrix is sparse and high-rank — niche apps, users with narrow taste — dense embeddings produce nonzero scores for pairs that genuinely have no relationship. It over-generalises and recommends things that are vaguely related but wrong.

> [!NOTE] Memorisation vs generalisation
> Memorisation = exploiting co-occurrences that actually happened in the logs. Generalisation = transitive guessing about combinations that never happened. Recommenders need both: memorisation keeps results topical, generalisation keeps them diverse. ^memorisation-generalisation

The trick is that the two components are **jointly trained**, not ensembled. One shared logistic loss, gradients flow back into both halves at the same time. This matters more than it sounds. In an ensemble each model must be independently good, so the wide part has to be a full-size, heavily feature-engineered logistic regression. Under joint training the wide part only has to *cover the deep part's mistakes*, so a small number of cross-product features is enough. The deep part learns "the wide side already handles the memorised exceptions, I will spend my capacity elsewhere."

What it unlocked: this is the template for basically every industrial [[Recommender Systems - Evolution|ranking model]] after 2016 — a memorisation branch plus a generalisation branch summed in logit space. DeepFM, DCN, xDeepFM are all "replace the wide side with something you do not have to hand-engineer."

## The Methodology

**The wide side.** A plain generalised linear model $y = \mathbf{w}^T\mathbf{x} + b$ over one-hot sparse features plus **cross-product transformations**:

$$\phi_k(\mathbf{x}) = \prod_{i=1}^{d} x_i^{c_{ki}}, \qquad c_{ki} \in \{0,1\}$$

$c_{ki}=1$ means feature $i$ takes part in cross $k$. For binary inputs this is just an AND: $\phi_k$ is 1 only if every constituent feature is 1. That product is where the nonlinearity in an otherwise linear model comes from.

In the production model the wide side used exactly **one** cross: installed apps × impression app.

**The deep side.** Every categorical feature gets a **32-dimensional embedding**, randomly initialised, learned by [[Backpropagation|backprop]] against the final loss. All embeddings are concatenated with the normalised dense features into a vector of roughly **1200 dimensions**, then pushed through 3 hidden layers:

$$a^{(l+1)} = f\left(W^{(l)}a^{(l)} + b^{(l)}\right)$$

with $f$ = [[ImageNet Classification with Deep CNNs (AlexNet)#^relu|ReLU]]. The figure in the paper uses 1024 → 512 → 256.

**The join.** The two outputs are added *in log-odds space*, then squashed:

$$P(Y=1 \mid \mathbf{x}) = \sigma\!\left(\mathbf{w}_{wide}^T[\mathbf{x}, \phi(\mathbf{x})] + \mathbf{w}_{deep}^T a^{(l_f)} + b\right)$$

One [[Cross Entropy|logistic loss]] on top. Label $y = 1$ if the impressed app was installed ("app acquisition"), 0 otherwise. One training example per impression.

**Two optimisers, one model.** This is a detail people forget. The wide part is trained with **FTRL + $L_1$** — FTRL produces genuinely sparse weight vectors, which you want when the cross-product feature space is enormous. The deep part is trained with **AdaGrad**. Same backward pass, different update rules on the two parameter groups. (Compare with [[Adam- A Method for Stochastic Optimization]] / [[Decoupled Weight Decay Regularization (AdamW)]] — same idea that the optimiser must match the geometry of the parameters.)

**Feature preprocessing.**
- Categorical strings → integer IDs via a vocabulary built from strings seen more than $n$ times. Rare strings are dropped.
- Continuous features are **quantile-normalised**, not standardised: map $x$ to its CDF $P(X \le x)$, chop into $n_q$ quantiles, and a value in bucket $i$ becomes $\frac{i-1}{n_q-1}$. Everything lands in $[0,1]$ and heavy tails stop mattering.

**Training scale and ops.** Over **500 billion examples**. Retraining from scratch on every new data batch was too slow, so they **warm-start**: initialise the new model with the previous model's embeddings and linear weights. Before a model goes live it gets a dry run and an automatic quality check against the previous model — a guard against the kind of silent failure described in [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]].

**Serving.** The retrieval stage hands the ranker a short candidate list (the standard [[Deep Neural Networks for YouTube Recommendations (RecSys)#^two-stage-funnel|two-stage funnel]]); Wide & Deep only does ranking. Peak load is **over 10 million apps scored per second** with a ~10 ms budget.

## Ablation Studies and Experiments

A 3-week live A/B test on Google Play, three arms at 1% of traffic each. The control was *not* a weak baseline — it was the existing, heavily tuned, wide-only logistic regression with rich hand-built crosses. All arms got the same feature set.

| Model | Offline AUC | Online acquisition gain |
|---|---|---|
| Wide only (control) | 0.726 | 0% |
| Deep only | 0.722 | +2.9% |
| Wide & Deep | 0.728 | +3.9% |

The interesting number is the gap between the two columns. Offline AUC moves by **0.002** — noise, by most standards, and the deep-only model is actually *worse* offline than the control while being +2.9% better online. Anyone judging these models on a holdout set would have shipped the wrong one.

The authors' explanation: offline data has frozen impressions and frozen labels, so you can only ever be scored on what the old policy chose to show. Online, a model that blends memorisation with generalisation surfaces genuinely new items, gets new user responses, and compounds. This is the same feedback-loop problem that [[Recommendations as Treatments- Debiasing Learning and Evaluation|propensity-based debiasing]] and [[Counterfactual Reasoning and Learning Systems]] attack head-on, and it is a strong argument for [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)#^replay-evaluation|replay-style offline evaluators]] over plain AUC.

**Serving latency ablation.** Scoring all candidates as one batch of 200 on one thread: 31 ms — over budget. Splitting into smaller batches run in parallel:

| Batch size | Threads | Latency |
|---|---|---|
| 200 | 1 | 31 ms |
| 100 | 2 | 17 ms |
| 50 | 4 | 14 ms |

**What is actually doing the work.** The wide component contributes **+1% acquisition on top of deep-only**, using a single cross-product feature. Most of the lift (+2.9%) comes from going deep at all. So the honest reading is: embeddings + MLP is the big win; the wide branch is a small, cheap correction that patches over-generalisation. It is not a 50/50 partnership.

**What is missing from the paper.** No ablation on embedding dimension, depth, number of crosses, or the choice of two separate optimisers. No comparison against [[Factorization Machines (ICDM)|factorization machines]] or [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)|GBDT + LR]], despite both being named as inspiration. No result reported for a wide+deep *ensemble* — so the paper's central architectural claim, that joint training beats ensembling, is argued but never measured.

## Worth Remembering

- **Joint training changes what each half needs to be.** The claim is not "two models are better than one." It is "if you train them together, the linear half can be tiny, because it only needs to fix residual errors." That is why one cross-product feature was enough.

- **Summing in logit space** is the mechanically important bit. Both branches predict log-odds; adding them is multiplying probabilities' odds. The gradient of the shared loss with respect to each branch's output is the same scalar $(\hat{y} - y)$, so the two halves compete for the same residual.

- **A 0.002 AUC difference produced a 3.9% business metric difference.** Keep this one. Offline metrics on logged data measure your ability to reproduce the old policy, not your ability to make good recommendations. Related scepticism: [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]], [[On the Difficulty of Evaluating Baselines]].

- **The baseline was strong.** A "highly-optimized wide-only logistic regression with rich cross-product feature transformations" is the real thing, tuned by Google for years. +3.9% against that is a much better result than +3.9% against a fresh scikit-learn fit.

- **Practical caveats if you build this.** The cross-product space explodes combinatorially; the $L_1$-regularised FTRL is what keeps it tractable, and dropping it will bite you. Vocabularies drop rare strings, so genuinely long-tail IDs never get an embedding at all — the exact regime where the deep half was supposed to help. Warm-starting from the previous model means embedding drift and staleness accumulate silently across retrains.

- **Open question the paper leaves.** The wide side still needs a human to pick which crosses to build. Everything after this (DCN, DeepFM) is an attempt to learn the crosses automatically. Worth asking whether that ever actually beat "one well-chosen hand-built cross" in a live test.

- The related-work section is unusually honest about lineage: [[Factorization Machines (ICDM)|FMs]] factorise pairwise interactions as a dot product of embeddings; this paper replaces the dot product with an MLP so the interactions can be highly nonlinear. The direct sparse-feature-to-output connection also echoes [[Deep Residual Learning for Image Recognition (ResNet)|residual shortcuts]] — a path that skips the deep stack.

## Links

Related: [[Recommender Systems - Evolution]] · [[Factorization Machines (ICDM)]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] · [[Deep Interest Network for CTR Prediction (DIN)]] · [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)]] · [[Collaborative Filtering for Implicit Feedback Datasets (ICDM)]] · [[Cross Entropy]] · [[Backpropagation]] · [[Regularization]] · [[Adam- A Method for Stochastic Optimization]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Regression Analysis]]

New topics worth writing: FTRL-Proximal, AdaGrad, Cross-product feature transformation, DeepFM, Deep & Cross Network (DCN), AUC / ROC, Quantile normalisation, Warm-starting model training, Online vs offline metric divergence, Logistic regression at scale
