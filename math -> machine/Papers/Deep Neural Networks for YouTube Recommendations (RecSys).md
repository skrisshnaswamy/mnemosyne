---
title: "Deep Neural Networks for YouTube Recommendations (RecSys)"
authors: ["Covington", "Adams & Sargin"]
year: 2016
url: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf
priority: Must-Read
read_on: 2026-08-24
tags: [paper]
---
## The Core Idea

Recommending from a corpus of millions of videos in tens of milliseconds is really two problems, not one. This paper is the canonical write-up of splitting them:

1. **Candidate generation** — from millions of videos, pick a few hundred that this user might plausibly like. Broad, cheap, high precision on "is this vaguely relevant".
2. **Ranking** — from those few hundred, score each one carefully with a rich feature set and sort. Expensive per item, but only a few hundred items.

The trick that makes stage 1 work is reframing recommendation as **extreme multiclass classification**: "which video will this user watch next, out of a million classes?" The softmax over videos means each video gets an output embedding $v_j$, and the network's job is to produce a user embedding $u$ such that $u \cdot v_j$ is big for the video actually watched. At serving time you throw the softmax away and just do approximate nearest neighbour in dot-product space. Training is a classifier; serving is a retrieval index. That decoupling is the whole design.

> [!NOTE] Two-stage retrieval-then-rank funnel
> Millions → (candidate generation, dot-product ANN) → hundreds → (ranking, full DNN per item) → dozens shown. Also lets you blend candidates from other, non-learned sources into the same ranker. ^two-stage-funnel

Two further ideas carry more weight than the architecture:

- **The label you pick matters more than the model.** They predict the user's *next* watch, using only actions strictly before it — not a randomly held-out watch. Random hold-out leaks the future and ignores that co-watching is asymmetric (people watch episode 1 then 2, not the reverse).
- **Optimise for watch time, not clicks.** Ranking by click-through rate promotes clickbait. They hack ordinary logistic regression into a watch-time regressor by weighting positive examples by their watch time.

Why it did not exist before: matrix factorisation (the predecessor system at YouTube, trained under rank loss) can only consume user–item IDs. A deep net is a **non-linear generalisation of factorisation** that can eat arbitrary extra features — search tokens, geography, age, device — and model their interactions. See [[Recommender Systems - Evolution]] for where this sits historically.

## The Methodology

### Candidate generation

The model is
$$P(w_t = i \mid U, C) = \frac{e^{v_i u}}{\sum_{j \in V} e^{v_j u}}$$
where $u \in \mathbb{R}^N$ is the network's output for (user, context) and $v_j$ is the output embedding of video $j$. Trained with [[Cross Entropy]] loss.

Inputs, all concatenated into one wide first layer:

- **Watch history**: bag of up to 50 recent video IDs → 256-dim embeddings → **averaged**. Averaging beat sum and component-wise max. Vocabulary 1M videos. Directly inspired by continuous-bag-of-words ([[Efficient Estimation of Word Representations (word2vec)]]).
- **Search history**: each query split into unigrams and bigrams, 1M-token vocab, 256-dim, averaged. Deliberately *unordered* — see below.
- **Demographics**: geographic region and device embedded; gender, logged-in state, age fed as raw reals normalised to $[0,1]$.
- **Example age**: how old the training example is.

Then a "tower" of fully connected ReLU layers, halving in width: e.g. 1024 → 512 → 256. Embeddings are trained jointly with everything else by ordinary [[Backpropagation]].

**Sampled softmax.** A million-way softmax is unaffordable, so they sample several thousand negative classes from the background distribution and correct with importance weighting — over 100× faster than full softmax. Same family of trick as [[Distributed Representations of Words and Phrases (negative sampling)|negative sampling]]. Hierarchical softmax was tried and was *worse*: tree nodes force you to discriminate between sets of unrelated classes, which is a harder problem than the one you care about.

**Serving.** Calibrated probabilities are not needed for a top-N list, so the softmax denominator is dropped and scoring becomes nearest-neighbour search over $u \cdot v_j$. A/B results were "not particularly sensitive" to which ANN library was used.

> [!NOTE] Example age
> A feature equal to $t_{\max} - t_N$: how far into the past this training example sits. Fed during training, set to **zero (or slightly negative) at serving**, meaning "predict as if right now". Without it, the model predicts the average popularity over the several-week training window and systematically under-recommends fresh uploads. With it, the predicted class probability tracks the real, spiky, time-dependent popularity curve of a video after upload. ^example-age

**Label and context hygiene** — the part that is hard-won and easy to get wrong:

- Train on **all** YouTube watches, including ones embedded on other sites, not just watches on recommendations we served. Otherwise the system only ever exploits its own past output.
- **Fixed number of training examples per user**, so heavy users do not dominate the loss.
- **Deliberately withhold information.** If the model sees the user's last search was "taylor swift" as an ordered signal, it learns to reproduce the search results page on the homepage — which performs terribly. Bagging the query tokens hides the label's origin.
- **Rollback, do not hold out.** Pick a random watch, use only actions before it as input, predict that watch. Never let the model see the future.

### Ranking

Same style of network, but scores one (user, impression) pair at a time and sorts. Hundreds of features, roughly half categorical and half continuous.

**Categorical features.** Embedding dimension grows roughly like $\log(\text{cardinality})$. Huge ID spaces are truncated to the top N by frequency in clicked impressions; out-of-vocabulary → the zero embedding. **Shared embeddings across features in the same ID space**: one global video-ID embedding table used by the impression video, the last watched video, the seed video, etc. — better generalisation, faster training, less memory. Each feature is still fed into the net separately so higher layers can specialise. Most parameters live here: 1M IDs × 32 dims is ~7× the parameters of a 2048-wide fully connected layer.

**Continuous features.** Networks are sensitive to input scale, so each feature $x$ is mapped through its own empirical CDF, $\tilde{x} = \int_{-\infty}^{x} df$, giving something uniform on $[0,1)$. The quantiles are computed in one pass before training and interpolated linearly. Then they feed $\tilde{x}$, $\tilde{x}^2$ and $\sqrt{\tilde{x}}$ — cheap super- and sub-linear basis functions.

**The features that matter most** are those describing the user's own prior interaction with this item or similar items: how many videos from this channel has the user watched, when did they last watch this topic. Also: how many times was this video already impressed to this user and not clicked — this creates churn, so reloading the page does not return the identical list. Candidate-generation signals (which source nominated this video, what score did it give) are passed through as ranking features.

> [!NOTE] Weighted logistic regression for watch time
> Train binary logistic regression on clicked/not-clicked, but weight each **positive** example by its observed watch time $T_i$ and each negative by 1. The learned odds become $\frac{\sum T_i}{N - k}$ ($N$ examples, $k$ positives), which for small click rate $P$ is $\approx E[T](1+P) \approx E[T]$. At inference, use $e^x$ as the final activation to read out those odds directly as expected watch time. ^weighted-logistic

## Ablation Studies and Experiments

**Candidate generation — depth × features (Figure 6, holdout MAP %).** Vocab 1M videos + 1M search tokens, 256-dim embeddings, bag size 50 each.

| Depth | Layers |
|---|---|
| 0 | linear map to 256 (≈ the old matrix-factorisation system) |
| 1 | 256 ReLU |
| 2 | 512 → 256 |
| 3 | 1024 → 512 → 256 |
| 4 | 2048 → 1024 → 512 → 256 |

Four feature sets were swept across all depths: watches only; + searches; + example age; all features. MAP climbs from roughly 2–3% at depth 0 to roughly 12–14% at depth 4 for the full feature set. The two axes are **complementary**: extra features only pay off once there is enough depth to model their interactions. Depth 0 with all features is barely better than depth 0 with watches only. Width and depth were increased until benefit flattened and convergence got hard.

**Ranking — hidden layers (Table 1, next-day holdout, weighted per-user loss).** The metric: take a clicked and an unclicked impression from the same page; if the model scores the unclicked one higher, count the clicked one's watch time as *mispredicted*. Report mispredicted watch time / total watch time. Lower is better.

| Hidden layers | Loss |
|---|---|
| None | 41.6% |
| 256 ReLU | 36.9% |
| 512 ReLU | 36.7% |
| 1024 ReLU | 35.8% |
| 512 → 256 | 35.2% |
| 1024 → 512 | 34.7% |
| **1024 → 512 → 256** | **34.6%** |

Note the diminishing returns: going from one to three layers buys 2.3 points, but the last layer buys only 0.1. 1024→512→256 was chosen because it fit the serving CPU budget.

**What did not work / what the ablations expose:**

- **Removing the powers** $\tilde{x}^2, \sqrt{\tilde{x}}$ from continuous features: loss up **0.2%**. Small but real.
- **Weighting positives and negatives equally** (i.e. plain CTR prediction instead of watch-time weighting): loss up **4.1%**. This is by far the largest single effect measured — the objective is doing more work than the architecture.
- **Hierarchical softmax**: could not reach comparable accuracy to sampled softmax.
- **Predicting a randomly held-out watch** instead of the next watch: worse on live A/B.
- **Reproducing the search results page** as homepage recommendations (what an unrestricted model learns to do): "performs very poorly".
- **Training only on watches that came from our own recommendations**: biases the system towards exploitation, new content cannot surface.
- **Sum / component-wise max** for pooling watch embeddings: averaging beat both.
- The example-age feature improved offline holdout precision *and* dramatically increased watch time on recently uploaded videos in live A/B.

## Worth Remembering

- **Offline metrics and live A/B often disagree.** The authors say it twice. The surrogate-problem choices (which label, which context) had "outsized importance on performance in A/B testing but [are] very difficult to measure with offline experiments." Anyone reading [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] should note that even Google treats offline numbers as directional only.
- **"There is more art than science in selecting the surrogate problem."** The single most transferable lesson. Predicting ratings ≠ good recommendations; predicting clicks ≠ good recommendations.
- **Deliberately crippling the model is a feature.** Discarding search-query order is throwing away information on purpose, so the model cannot overfit the structure of the site instead of learning preference.
- **Deep learning did not remove feature engineering.** They still spend "considerable engineering resources" building features, and explicitly say raw data does not go straight into a feedforward net. The most valuable features are hand-built summaries of the user's past interaction with this item and similar items — matching [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]].
- Serving up-to-the-second impression and watch history is called "an engineering feat onto itself" and is skipped entirely. That is the hard part of reproducing this.
- Scale: ~1 billion parameters, hundreds of billions of training examples, built on TensorFlow.
- **Limitations they do not dwell on:** no sequence model at all — watch history is a bag, order is thrown away. That gap is exactly what [[BERT4Rec- Sequential Recommendation with Bidirectional Transformer]] and later [[Actions Speak Louder than Words- Generative Recommenders (HSTU)]] fill. Also, optimising watch time has its own well-known failure mode (rabbit-holing), and there is nothing here about diversity or calibration — see [[Calibrated Recommendations (RecSys)]].
- The sampled softmax has no logQ correction here; [[PinnerFormer- Sequence Modeling for User Representation at Pinterest]] shows what that costs.
- Open question worth chasing: the example-age feature is a hand-built fix for distribution shift. Is there a principled version? Setting it to "slightly negative" at serving is a magic knob.

## Links

Related: [[Recommender Systems - Evolution]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[Efficient Estimation of Word Representations (word2vec)]] · [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] · [[BERT4Rec- Sequential Recommendation with Bidirectional Transformer]] · [[Actions Speak Louder than Words- Generative Recommenders (HSTU)]] · [[PinnerFormer- Sequence Modeling for User Representation at Pinterest]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Calibrated Recommendations (RecSys)]] · [[Cross Entropy]] · [[Backpropagation]] · [[BPR- Bayesian Personalized Ranking from Implicit Feedback]] · [[Deep Interest Network for CTR Prediction (DIN)]] · [[Entire Space Multi-Task Model (ESMM)]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Real-time Personalization using Embeddings for Search Ranking at Airbnb (KDD, best paper)]] · [[NDCG]] · [[Regression Analysis]]

New topics worth writing: Sampled softmax and importance weighting, Approximate nearest neighbour search (HNSW/ScaNN), Quantile normalisation of features, Position and presentation bias in ranking, Surrogate objectives in recommender systems, Two-tower retrieval models, Extreme multiclass classification
