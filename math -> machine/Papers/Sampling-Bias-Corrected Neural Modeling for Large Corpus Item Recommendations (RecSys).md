---
title: "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)"
authors: ["Yi et al."]
year: 2019
url: https://raw.githubusercontent.com/tangxyw/RecSysPapers/main/Match/%5B2019%5D%5BGoogle%5D%20Sampling-Bias-Corrected%20Neural%20Modeling%20for%20Large%20Corpus%20Item%20Recommendations.pdf
priority: Must-Read
read_on: 2026-08-25
tags: [paper, llm, rl, self-supervised, theory]
---
## The Core Idea

When you train a two-tower retrieval model on a batch of $B$ user–item pairs, the cheapest way to get negatives is to reuse the other items *in the same batch*. This is free: their embeddings are already computed. But it introduces a silent bug. Items get into a batch in proportion to how often they appear in the log, and item popularity follows a power law. So a mega-popular video shows up as a negative in almost every batch and gets pushed down again and again, while a tail video almost never appears and is never pushed down. The in-batch softmax is not the full softmax — it is the full softmax with popular items systematically over-penalised.

The fix is a $\log Q$ correction: subtract the log of the item's sampling probability from its logit before the softmax.

$$s^c(x_i, y_j) = s(x_i, y_j) - \log(p_j)$$

This is an old idea from importance-sampled softmax in language models (Bengio & Senécal). The new part is that in a real recommender you **cannot precompute $p_j$**. The item vocabulary is not fixed — YouTube uploads new videos every second, popularity shifts daily, and training is a never-ending stream, not a pass over a static corpus. So the contribution is an online, distributed, hash-based estimator of $p_j$ that runs alongside SGD, adapts when the distribution shifts, and needs no item vocabulary.

> [!NOTE] logQ correction ^logq-correction
> Subtracting $\log p_j$ from the logit of item $j$ turns a biased sampled-softmax into an (approximately) unbiased estimate of the full softmax. Popular items were being punished for being *sampled* often, not for being *wrong* often; the correction removes exactly that.

The second idea, which is more of a trick than a theorem: instead of tracking "how often does item $y$ appear", track "how many training steps pass between two consecutive appearances of $y$". Call that gap $\delta$. Then $p = 1/\delta$. The gap is easy to estimate online with one moving average, and the global step counter is already synchronised across workers by the parameter servers — so the estimator is distributed for free.

## The Methodology

**The model.** Two towers, $u(x,\theta)$ for {user, context} and $v(y,\theta)$ for {item}, both mapping into $\mathbb{R}^k$. Score is a dot product $s(x,y) = \langle u(x,\theta), v(y,\theta)\rangle$. This is a [[Sentence-BERT#^bi-encoder|bi-encoder]] applied to recommendation: the item tower can be run offline over the whole corpus, indexed, and served with approximate maximum inner product search.

**The loss.** Weighted softmax [[Cross Entropy|cross-entropy]] over the batch:

$$P_B^c(y_i|x_i;\theta) = \frac{e^{s^c(x_i,y_i)}}{e^{s^c(x_i,y_i)} + \sum_{j \in [B], j \neq i} e^{s^c(x_i,y_j)}}, \qquad L_B(\theta) = -\frac{1}{B}\sum_{i \in [B]} r_i \log P_B^c(y_i|x_i;\theta)$$

$r_i$ is a **reward**, not just a 0/1 label. For YouTube, $r_i = 0$ for a click with almost no watch time, $r_i = 1$ for a fully watched video. It weights the example in the loss. Live experiments used $r_i$ tied to the actual engagement metric they wanted to move.

**Normalisation and temperature.** Both embeddings are L2-normalised, then divided by a temperature:
$$s(x,y) = \langle u(x,\theta), v(y,\theta)\rangle / \tau$$
They report normalisation "always" improved quality and stability, and $\tau$ has to be tuned hard. Same [[Distilling the Knowledge in a Neural Network#^softmax-temperature|temperature]] knob you see everywhere in contrastive learning.

**The frequency estimator (Algorithm 2).** Two arrays $A, B$ of size $H$, one hash function $h$ mapping item id to $[H]$.
- $A[h(y)]$ = the last global step at which $y$ was seen.
- $B[h(y)]$ = the running estimate of $\delta$, the gap between hits.

When $y$ appears at step $t$:
$$B[h(y)] \leftarrow (1-\alpha)\,B[h(y)] + \alpha\,(t - A[h(y)]), \qquad A[h(y)] \leftarrow t$$
At inference, $\hat p = 1/B[h(y)]$.

That update *is* SGD with fixed learning rate $\alpha$ on the squared error to the mean of $\Delta$. Proposition 4.1 gives:
$$\mathbb{E}[\delta_t] - \delta = (1-\alpha)^t \delta_0 - (1-\alpha)^t \delta$$
so the bias decays geometrically, and initialising $\delta_0 = \delta/(1-\alpha)$ would make it unbiased at every step. Variance is bounded by $(1-\alpha)^{2t}(\delta_0 - \delta)^2 + \alpha\,\mathbb{E}[(\Delta - \delta)^2]$. The second term never goes away — it is the standard fixed-learning-rate floor, the same bias–variance trade you get with any exponential moving average. Big $\alpha$: adapts fast, noisy. Small $\alpha$: smooth, slow to react.

**Hash collisions (Algorithm 3).** Two items sharing a bucket makes the bucket look like it is hit twice as often, so $\delta$ is *under*-estimated and $p$ *over*-estimated. Fix borrowed from count-min sketch: run $m$ independent hash functions with $m$ array pairs, and at inference take
$$\hat p = 1/\max_i \{B_i[h_i(y)]\}$$
The max over gaps is the min over probabilities — pick the bucket least polluted by collisions.

**YouTube system.**
- Query tower gets seed-video features (the video currently being watched) plus a bag-of-words average of the user's last $k$ watched video-id embeddings. Item tower gets candidate video features.
- Categorical features (video id, channel id, topics) get embedding tables; multi-valued sparse features become a weighted sum of embeddings; out-of-vocabulary ids go into a fixed set of **hash buckets** with learned embeddings, which is what lets brand-new uploads get a representation.
- Embeddings are shared between query and candidate towers where the feature exists in both. Non-shared embeddings were tried and gave no meaningful gain.
- Three-layer MLPs, hidden sizes [1024, 512, 128], both towers. Adagrad, lr 0.2, batch size 8192. Frequency estimator: $H = 50\text{M}$, $m=1$, $\alpha = 0.01$.
- **Sequential training**: the trainer walks days of logs oldest to newest, then waits for tomorrow's data. It never stops. The streaming estimator is what makes this possible — a static $p_j$ table would rot.
- Index rebuilt every few hours over ~10M videos, covering >90% of training examples. Pipeline: candidate example generation → embedding inference (right tower) → build a tree + quantised-hash index → stitch the query tower onto the index and export one SavedModel.

## Ablation Studies and Experiments

**Simulation.** Items drawn with $q_i \propto i^2$ over $M=1000$ items, batch 128, then the distribution is flipped to $q_i \propto (M-1-i)^2$ at step 10,000. Error is rescaled L1 between estimated and true probabilities.
- Learning rate $\alpha$: all settings converge; higher $\alpha$ recovers faster from the distribution flip but plateaus at a higher error floor. Exactly what Proposition 4.1 predicts.
- Multiple hashes: with the *total* number of buckets held constant, $m = 1, 2, 4$ gives monotonically lower error. So splitting your memory across more hash functions beats one big array. Cheap win.

**Wikipedia link prediction.** 5.3M pages, 430M links, predict destination page from source page. Two towers, [512, 128] ReLU, shared input embeddings, batch 1024, Adagrad lr 0.01, 10M steps, $m=1$, $H=40\text{M}$, $\alpha=0.01$.

| Method | R@10 | R@50 | R@100 | R@300 |
|---|---|---|---|---|
| mse-gramian | 0.0432 | 0.1326 | 0.2027 | 0.3530 |
| plain-sfx $\tau{=}0.05$ | 0.0579 | 0.2259 | 0.3573 | 0.5931 |
| plain-sfx $\tau{=}0.07$ | 0.0643 | 0.2423 | 0.3746 | 0.5991 |
| plain-sfx $\tau{=}0.14$ | 0.0614 | 0.2216 | 0.3341 | 0.5200 |
| **correct-sfx $\tau{=}0.05$** | 0.0987 | **0.3202** | **0.4835** | **0.7413** |
| correct-sfx $\tau{=}0.07$ | **0.1065** | 0.3079 | 0.4664 | 0.7234 |
| correct-sfx $\tau{=}0.14$ | 0.0807 | 0.2411 | 0.3519 | 0.5529 |

Recall@10 nearly doubles (0.0643 → 0.1065). Note the interaction: the *best* temperature moves once you apply the correction, and correct-sfx at a bad temperature ($\tau=0.14$, R@50 = 0.2411) is barely better than plain-sfx at a good one (0.2423). **The correction only pays off if you retune $\tau$.** That is the sharpest lesson in the table.

**YouTube offline**, retrieving clicked videos from a 10M index, $r_i = 1$ for all clicks (they admit they simplified the reward because there is no clean offline metric for a continuous reward). Evaluated after a 15-day catch-up phase, averaged over the following 7 days to wash out weekly seasonality.

| Method | R@5 | R@10 | R@30 | R@50 |
|---|---|---|---|---|
| mse-gramian | 0.0554 | 0.0768 | 0.1149 | 0.1338 |
| plain-sfx $\tau{=}0.05$ | 0.2069 | 0.2728 | 0.3964 | 0.4586 |
| correct-sfx $\tau{=}0.05$ | **0.2150** | **0.2960** | **0.4537** | **0.5322** |

Gains grow with $K$: +4% relative at R@5, +16% at R@50. Makes sense — the correction mostly changes where tail items land, and tail items live deep in the ranking.

**Live A/B on YouTube**, adding this as an extra nominator alongside the existing production retrieval:
- plain-sfx: **+0.20%** engagement
- correct-sfx: **+0.37%** engagement

Both against production control. Nearly double the lift from one line of arithmetic in the loss.

**What did not work / was rejected:**
- **MSE + Gramian regulariser** (the standard implicit-feedback loss: squared error on observed pairs plus a term pulling all unobserved pairs to 0) is crushed everywhere. On YouTube R@50 it gets 0.1338 vs 0.4586 for even uncorrected softmax — roughly 3.4× worse. Pointwise losses are simply the wrong tool for retrieval.
- **Non-shared embeddings** between query and candidate towers: tried, no significant quality improvement, so they kept sharing.
- **Adaptive sampling** (sampling negatives proportional to the model's own output distribution, which theory says is optimal) is not applicable here — with a deep item tower and content features you cannot cheaply score arbitrary candidates, so they stick with in-batch negatives and correct after the fact.
- **Hierarchical / tree softmax** rejected: needs a predefined tree over categorical attributes, incompatible with arbitrary content features.

## Worth Remembering

- The whole method is: track gaps between hits in a hash table, invert to get $p$, subtract $\log p$ from logits. It is maybe 30 lines of code and it doubled the A/B lift.
- **This is the sampled-softmax bias-correction paper that the recommender world cites for logQ.** [[PinnerFormer- Sequence Modeling for User Representation at Pinterest#^logq-correction|PinnerFormer]] uses this directly. If you are training any two-tower retrieval model with in-batch negatives, this is the default you should be starting from.
- The under/over-estimation asymmetry is worth internalising: hash collisions make a bucket look *hotter* than it is, so $\delta$ is under-estimated, so $p$ is over-estimated, so you over-correct and *over*-promote colliding items. Taking the max over $m$ estimates of $\delta$ is the conservative direction.
- Nothing here handles the "false negative" problem — if a popular item genuinely is a good recommendation for user $i$ but happens to be user $j$'s positive in the same batch, it is still treated as a negative for $i$. The logQ correction softens this statistically but does not fix individual cases.
- The reward $r_i$ as example weight is an underrated detail. It is how you make a retrieval model optimise for watch time rather than clicks without changing the architecture. Offline metrics cannot see it, which is precisely why offline recall and the live engagement number can disagree.
- Practical caveats: (1) $\tau$ must be retuned after adding the correction; (2) $\alpha$ trades adaptation speed against estimator noise and there is a permanent variance floor; (3) the estimator is tied to the *global step*, so it only works if your workers share a synchronised step counter — this is a parameter-server-era design and would need rethinking under fully synchronous data-parallel training; (4) $H = 50\text{M}$ buckets is a real memory cost you carry through training.
- Open question the paper does not answer: does the correction still help when your batch size is small enough that the popularity skew inside a batch is dominated by variance rather than by the power law? All their batches are 1024–8192.
- Connects cleanly to the general contrastive-learning story — this is [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)#^infonce|InfoNCE]] with a corrected partition function, and the popularity bias it removes is the same force that drives [[Understanding Contrastive Learning through Alignment and Uniformity#^uniformity|non-uniform]] embedding geometry.

## Links

Related: [[PinnerFormer- Sequence Modeling for User Representation at Pinterest]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[Cross Entropy]] · [[Sentence-BERT]] · [[Distilling the Knowledge in a Neural Network]] · [[BPR- Bayesian Personalized Ranking from Implicit Feedback]] · [[Recommender Systems - Evolution]] · [[Real-time Personalization using Embeddings for Search Ranking at Airbnb (KDD, best paper)]] · [[Understanding Contrastive Learning through Alignment and Uniformity]] · [[Foundational_RecSys_Ranking_Reading_Plan]] · [[Random variable]]

New topics worth writing: Sampled softmax and importance sampling, Count-min sketch, Approximate maximum inner product search (MIPS), Product quantization, Two-tower retrieval models, Adagrad, Exponential moving average estimators, Popularity bias in recommenders, Parameter server training
