---
title: "Session-based Recommendations with RNNs (GRU4Rec)"
authors: ["Balázs Hidasi", "Alexandros Karatzoglou", "Linas Baltrunas", "Domonkos Tikk"]
year: 2015
arxiv: "1511.06939"
url: https://arxiv.org/abs/1511.06939
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, transformers]
---
## The Core Idea

Many websites do not know who you are. No login, no stable cookie, no long history. All they have is the handful of clicks you made in the last ten minutes. That is a **session**. Classic recommenders break here: [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)|matrix factorization]] needs a user vector, and a brand-new anonymous session has none. So industry fell back on [[Amazon.com Recommendations- Item-to-Item Collaborative Filtering (IEEE Internet Computing)|item-to-item similarity]]: "you clicked X, here are things often clicked with X." That works, but it throws away everything except the *last* click.

The idea here: treat a session as a **sequence** and run a recurrent neural network over it. Input at step $t$ is the item just clicked; output is a score for every item in the catalogue, meaning "how likely is this the next click". The hidden state carries the whole session so far, so click 5 is influenced by clicks 1–4, not just click 4.

> [!NOTE] Session-based recommendation
> Predicting the next item from the current anonymous browsing session only. No user identity, no history across sessions. Each session is treated as independent. ^session-based

That much is obvious once you know RNNs exist. What makes the paper useful is the three engineering fixes needed to make an RNN survive the recommender setting, where the output vocabulary is 37k–330k items and the data is tens of millions of clicks:

1. **Session-parallel mini-batches** — because sessions have wildly different lengths (2 events to hundreds) and you must not chop them up.
2. **Sampling the output** — you cannot score 330k items on every step. Use the *other items in the mini-batch* as negatives, which is free and is automatically popularity-weighted.
3. **A pairwise ranking loss** (BPR, and a new one called TOP1) — because pointwise losses like [[Cross Entropy|cross-entropy]] were numerically unstable here.

This is the paper that started deep learning in sequential recommendation. [[Self-Attentive Sequential Recommendation (SASRec)|SASRec]], [[BERT4Rec- Sequential Recommendation with Bidirectional Transformer|BERT4Rec]], and eventually [[Actions Speak Louder than Words- Generative Recommenders (HSTU)|HSTU]] are all descendants.

## The Methodology

**The unit.** A GRU (Gated Recurrent Unit). It is an RNN cell with two [[Gated Activation|gates]] that decide how much of the old state to keep, which is how it dodges vanishing gradients (see [[Long Short-Term Memory (Neural Computation)|LSTM]] for the same idea with more gates):

$$\mathbf{h}_t = (1-\mathbf{z}_t)\,\mathbf{h}_{t-1} + \mathbf{z}_t\,\hat{\mathbf{h}}_t$$
$$\mathbf{z}_t = \sigma(W_z\mathbf{x}_t + U_z\mathbf{h}_{t-1}), \quad \mathbf{r}_t = \sigma(W_r\mathbf{x}_t + U_r\mathbf{h}_{t-1})$$
$$\hat{\mathbf{h}}_t = \tanh\!\big(W\mathbf{x}_t + U(\mathbf{r}_t \odot \mathbf{h}_{t-1})\big)$$

The update gate $\mathbf{z}_t$ is a soft switch between "keep the old state" and "take the new candidate". The reset gate $\mathbf{r}_t$ lets the cell forget the past when computing the candidate.

**Input.** 1-of-N (one-hot) over the item set. Length = number of items. They also tried a weighted sum of past one-hots with older events discounted, then normalised. And they tried a learned [[Linear Projection|embedding]] layer. **One-hot on the last item alone won.**

**Architecture.** One GRU layer, optional feed-forward layers after it, output layer scoring all (sampled) items. Output activation: $\tanh$. Optionally the input is also wired directly into deeper GRU layers, which helped when they used multiple layers — but multiple layers did not help anyway.

### Session-parallel mini-batches

NLP usually slides a window over a sentence and stacks fragments. Wrong here. Instead: line up $X$ sessions side by side. Mini-batch 1 = the 1st event of each of the $X$ sessions, target = the 2nd event of each. Mini-batch 2 = the 2nd events, target = the 3rd. When a session ends, drop it, slot in the next unused session, and **reset that row's hidden state to zero**. So each row of the batch is one live session, and rows finish at different times.

### Sampling on the output

Scoring every item every step costs $O(\text{items} \times \text{events})$ — unusable. So score only the positive plus a sample of negatives.

Which negatives? Their argument: a missing interaction usually means "user never saw it", but for a *popular* item, missing more likely means "user saw it and didn't want it". So negatives should be sampled **in proportion to popularity**. The trick: use the target items of the *other examples in the same mini-batch* as your negatives. No sampling code, no extra lookups, pure matrix ops — and since an item appears in the batch in proportion to how often it is clicked, it is popularity-proportional for free. This is the same idea as [[Dense Passage Retrieval (DPR)#^in-batch-negatives|in-batch negatives]], arriving five years earlier.

> [!NOTE] In-batch popularity sampling
> Reuse the other examples' targets in a mini-batch as negative samples. Free, cache-friendly, and implicitly popularity-weighted because frequent items land in batches more often. ^in-batch-popularity

### Ranking losses

Both are pairwise: compare the score of the true next item $i$ against each negative $j$, at session position $s$, over $N_S$ samples.

**BPR** (from [[BPR- Bayesian Personalized Ranking from Implicit Feedback|BPR]], averaged over several negatives instead of one):
$$L_s = -\frac{1}{N_S}\sum_{j=1}^{N_S} \log \sigma(\hat r_{s,i} - \hat r_{s,j})$$

**TOP1** (new in this paper). Start from the relative rank of the true item, $\frac{1}{N_S}\sum_j I\{\hat r_{s,j} > \hat r_{s,i}\}$, and replace the indicator with a sigmoid so it is differentiable. Problem: minimising that alone lets scores drift upward without bound, because an item that is positive in one row is negative in another. Fix: add a term pushing negative scores toward zero, kept on the same scale as the rank term.
$$L_s = \frac{1}{N_S}\sum_{j=1}^{N_S}\Big[\sigma(\hat r_{s,j} - \hat r_{s,i}) + \sigma(\hat r_{s,j}^2)\Big]$$
The $\sigma(\hat r_{s,j}^2)$ piece is a [[Regularization|regulariser]] on the negative scores, not on the weights.

### Training setup

- Optimiser: **Adagrad**, which beat rmsprop. (This predates [[Adam- A Method for Stochastic Optimization|Adam]] being the default.)
- Hidden units: 100 or 1000.
- Hyperparameters: 100 random search points per dataset per loss, then per-parameter tuning on a validation split, then retrain on train+validation.
- Best settings differed by loss. RSC15/TOP1: batch 50, [[Dropout- A Simple Way to Prevent Overfitting|dropout]] 0.5, lr 0.01, [[Momentum|momentum]] 0. RSC15/BPR: batch 50, dropout 0.2, lr 0.05, momentum 0.2. RSC15/cross-entropy needed batch **500** and dropout **0**. VIDEO/TOP1: batch 50, dropout 0.4, lr 0.05.
- Weights initialised uniform in $[-x, x]$ with $x$ depending on matrix shape (see [[Understanding the difficulty of training deep feedforward networks (Xavier init)|Xavier init]]).
- A few hours on one GTX Titan X. The 100-unit version trains on CPU in acceptable time — which matters, because recommenders retrain often as new items arrive.

### Data

- **RSC15** (RecSys Challenge 2015 e-commerce clicks): ~6 months train = 7,966,257 sessions, 31,637,239 clicks, 37,483 items. Test = the next single day, 15,324 sessions / 71,222 events. Sessions of length 1 removed. Test clicks on items unseen in train removed. **Splits are by session, never mid-session.**
- **VIDEO** (an OTT video service, watch events past a duration threshold): ~3M sessions, ~13M watches, 330k videos. Test = last day, ~37k sessions / ~180k events. Very long sessions filtered as likely bots. Caveat the authors flag: this platform already showed item-to-item recommendations on screen, so the logged behaviour is contaminated by the incumbent algorithm.

**Evaluation.** Feed session events one at a time, check the rank of the actual next item. Reset hidden state between sessions. Metrics: **recall@20** (is the true item in the top 20) and **MRR@20** (mean reciprocal rank, zero if rank > 20). On RSC15 all 37,483 items are ranked; on VIDEO only against the 30,000 most popular, which they defend as standard practice since rare items score low anyway.

## Ablation Studies and Experiments

**Baselines** (Table 1):

| Baseline | RSC15 R@20 | RSC15 MRR@20 | VIDEO R@20 | VIDEO MRR@20 |
|---|---|---|---|---|
| POP (global popularity) | 0.0050 | 0.0012 | 0.0499 | 0.0117 |
| S-POP (popularity within session) | 0.2672 | 0.1775 | 0.1301 | 0.0863 |
| Item-KNN (cosine co-occurrence) | **0.5065** | **0.2048** | **0.5508** | **0.3381** |
| BPR-MF (session vector = mean of item vectors) | 0.2574 | 0.0618 | 0.0692 | 0.0374 |

Item-KNN crushes everything else. Matrix factorization, forced into this setting by averaging item factors, is *worse than session popularity*. That is the whole motivation in one row.

**Main results** (Table 3, % gain over Item-KNN):

| Loss / units | RSC15 R@20 | RSC15 MRR@20 | VIDEO R@20 | VIDEO MRR@20 |
|---|---|---|---|---|
| TOP1 100 | 0.5853 (+15.6%) | 0.2305 (+12.6%) | 0.6141 (+11.5%) | 0.3511 (+3.8%) |
| BPR 100 | 0.6069 (+19.8%) | 0.2407 (+17.5%) | 0.5999 (+8.9%) | 0.3260 (**−3.6%**) |
| Cross-entropy 100 | 0.6074 (+19.9%) | 0.2430 (+18.7%) | 0.6372 (+15.7%) | 0.3720 (+10.0%) |
| **TOP1 1000** | 0.6206 (+22.5%) | **0.2693 (+31.5%)** | **0.6624 (+20.3%)** | **0.3891 (+15.1%)** |
| BPR 1000 | **0.6322 (+24.8%)** | 0.2467 (+20.5%) | 0.6311 (+14.6%) | 0.3136 (**−7.2%**) |
| Cross-entropy 1000 | 0.5777 (+14.1%) | 0.2153 (+5.2%) | — (diverged) | — |

TOP1 with 1000 units is the recommendation: ~20–30% over the best baseline. Note BPR *loses* to Item-KNN on VIDEO MRR at both sizes — it gets the item into the top 20 but ranks it badly.

**What did not work:**

- **Pointwise losses.** Cross-entropy and direct MRR optimisation were numerically unstable even with regularisation. Out of 100 random hyperparameter runs, cross-entropy gave **only 10 stable networks on RSC15 and 6 on VIDEO**. Cross-entropy at 1000 units on VIDEO diverged entirely. Diagnosis: pointwise loss pushes the positive score up hard but barely pushes negatives down, so scores blow up. This is exactly what the $\sigma(\hat r^2)$ term in TOP1 exists to prevent. (Notably, cross-entropy at 100 units is actually the *best* number in the table on VIDEO — it just cannot be trained reliably. Later work, GRU4Rec+, revisited this and fixed the sampling to make cross-entropy work.)
- **More layers.** Stacking GRU layers made things worse on training loss *and* both test metrics, every time. Their guess: sessions are short, so there is no multi-timescale structure to model. They admit they do not really know why.
- **Item embeddings.** Slightly worse than plain one-hot. Surprising, since embeddings are the default everywhere else.
- **Feeding the whole session history at the input** (weighted sum of past one-hots) gave no gain over just the previous item — the GRU state already carries it.
- **Extra feed-forward layers after the GRU.** No help.
- **Other cells.** Vanilla RNN and LSTM both underperformed GRU.
- **rmsprop** lost to Adagrad.

**What did help:** widening the single GRU layer (100 → 1000 units), and $\tanh$ on the output layer.

## Worth Remembering

- The single biggest practical lesson is not "use an RNN" — it is that the **loss and the negative sampling determine whether the thing trains at all**. The architecture ablations are all negative results; the loss ablation is where the paper lives.
- The in-batch-negatives trick appears here for efficiency and popularity-weighting, with no correction for the sampling bias. Later work shows this bias matters and fixes it explicitly — see [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)|logQ correction]] and [[Approximate Nearest Neighbor Negative Contrastive Learning (ANCE)|ANCE]] for harder negatives.
- "Deeper is worse" aged badly. [[Self-Attentive Sequential Recommendation (SASRec)|SASRec]] and [[BERT4Rec- Sequential Recommendation with Bidirectional Transformer|BERT4Rec]] use stacked [[Attention Is All You Need|self-attention]] blocks and beat this. The finding was probably about GRUs plus 2015 optimisation practice, not about depth.
- "Embeddings did not help" also aged badly, and is suspicious. With 330k items and one-hot input, the input weight matrix $W$ *is* an embedding table — the difference is only whether you factor it. Something in the tuning likely favoured the unfactored form.
- **Reproducibility caveat.** This paper is one of the headline cases in [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|Dacrema et al.]] and related session-rec replication work: carefully tuned neighbourhood methods (a session-aware KNN, not the plain item-KNN used here) often match or beat GRU4Rec. Their Item-KNN baseline uses only the last click. A session-aware KNN is a much stronger opponent. Read the +20% with that in mind, and see [[On the Difficulty of Evaluating Baselines]].
- The VIDEO dataset is feedback-contaminated: item-to-item recommendations were shown on screen during collection, so the "true next item" is partly the incumbent algorithm's choice. Any model that mimics item-to-item behaviour gets an unearned boost. Compare the [[Recommendations as Treatments- Debiasing Learning and Evaluation|logging-policy bias]] problem.
- Evaluation ranks against only the top 30k items on VIDEO. Common, defensible, but it inflates recall relative to full-catalogue ranking. Do not compare these numbers to papers that rank everything.
- Practical note if you implement it: the hidden-state reset when a session ends inside a parallel mini-batch is the bug people get wrong. Every row of the batch has its own lifetime; you must zero that row's $\mathbf{h}$ and nothing else.
- Open question the authors leave: they wanted to feed item *content* (thumbnails, video, text) rather than IDs. That thread became content-aware and multimodal recommenders.

## Links

Related: [[BPR- Bayesian Personalized Ranking from Implicit Feedback]] · [[Self-Attentive Sequential Recommendation (SASRec)]] · [[BERT4Rec- Sequential Recommendation with Bidirectional Transformer]] · [[Amazon.com Recommendations- Item-to-Item Collaborative Filtering (IEEE Internet Computing)]] · [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)]] · [[Long Short-Term Memory (Neural Computation)]] · [[Gated Activation]] · [[Dense Passage Retrieval (DPR)]] · [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Recommender Systems - Evolution]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[Cross Entropy]] · [[NDCG]] · [[Collaborative Filtering for Implicit Feedback Datasets (ICDM)]] · [[Actions Speak Louder than Words- Generative Recommenders (HSTU)]]

New topics worth writing: Gated Recurrent Unit, Recurrent Neural Networks, Mean Reciprocal Rank, Adagrad, Truncated backpropagation through time, Session-aware KNN baselines, GRU4Rec+ and sampled softmax with additional negatives, Learning-to-rank (pointwise vs pairwise vs listwise)
