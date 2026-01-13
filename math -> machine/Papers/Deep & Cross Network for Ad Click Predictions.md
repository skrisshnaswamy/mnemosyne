---
title: "Deep & Cross Network for Ad Click Predictions"
authors: ["Ruoxi Wang", "Bin Fu", "Gang Fu", "Mingliang Wang"]
year: 2017
arxiv: "1708.05123"
url: https://arxiv.org/abs/1708.05123
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, vision, theory]
---
## The Core Idea

Click prediction lives and dies on **cross features** — combinations like `country=USA AND banner_position=1 AND device=mobile`. A plain linear model cannot invent these; a human has to hand-write them. [[Wide & Deep Learning for Recommender Systems|Wide & Deep]] made the hand-writing official: you pick the crosses, the wide side memorises them, the deep side generalises. That works, but picking crosses is an exponential search problem with no good algorithm.

A plain deep network (MLP) does learn interactions, but *implicitly* — smeared through matrix multiplies and ReLUs. Nobody can point at a weight and say "that is the USA × mobile term". Worse, an MLP is surprisingly bad at cheaply representing simple multiplicative terms like $x_i x_j$; it needs a lot of neurons to approximate a product.

The trick here is a tiny network layer whose whole job is multiplication, and which stacks so the **polynomial degree goes up by exactly one per layer**. After $l$ cross layers you have every monomial $x_1^{\alpha_1}\cdots x_d^{\alpha_d}$ up to degree $l+1$, each with its own coefficient — and you paid only $2d$ parameters per layer. Not $O(d^2)$, not $O(d^n)$. Linear in the input width.

That is the unlock: explicit, bounded-degree feature crossing, automatic, essentially free. You bolt it in parallel next to a normal MLP, concatenate the two outputs, and train jointly. On Criteo it beats a tuned MLP while using **40% of the memory**.

> [!NOTE] Cross layer
> A layer of the form $\mathbf{x}_{l+1} = \mathbf{x}_0\mathbf{x}_l^\top \mathbf{w}_l + \mathbf{b}_l + \mathbf{x}_l$. It multiplies the *original* input by the current state, projects back down to $d$ dimensions, and adds a residual connection. Each layer raises the highest polynomial degree by one. ^cross-layer

## The Methodology

**Shape of the model.** One embedding-and-stacking layer feeds *two parallel* towers — the cross network and a normal MLP. Their outputs are concatenated and pushed through one logistic unit.

**Embedding and stacking.** Categorical features go through a lookup, $\mathbf{x}_{\text{embed},i} = W_{\text{embed},i}\,\mathbf{x}_i$, with embedding width set by a rule of thumb:
$$n_e = 6 \times (\text{cardinality})^{1/4}$$
Real-valued features get a log transform, then everything is concatenated into one vector
$$\mathbf{x}_0 = [\mathbf{x}_{\text{embed},1}^\top, \ldots, \mathbf{x}_{\text{embed},k}^\top, \mathbf{x}_{\text{dense}}^\top]$$
On Criteo (13 integer + 26 categorical features) this gives $d = 1026$.

**The cross network.** Each layer:
$$\mathbf{x}_{l+1} = \mathbf{x}_0 \mathbf{x}_l^\top \mathbf{w}_l + \mathbf{b}_l + \mathbf{x}_l$$
Read the first term right-to-left and the cost collapses. $\mathbf{x}_l^\top \mathbf{w}_l$ is a **scalar**. So you never build the $d \times d$ outer product $\mathbf{x}_0\mathbf{x}_l^\top$ — you just scale $\mathbf{x}_0$. That is the rank-one property doing all the work: $O(d)$ time, $O(d)$ memory, $2d$ parameters per layer, $2 d L_c$ total.

Conceptually the layer *does* form all $d^2$ pairwise products $x_i \tilde x_j$ and *does* project them back to $d$ dimensions — but the projection matrix is forced to be block-diagonal with the same $\mathbf{w}$ in every block, so you only store $d$ numbers instead of $d^3$.

The $+\mathbf{x}_l$ is a residual connection in the sense of [[Deep Residual Learning for Image Recognition (ResNet)|ResNet]]: $f$ fits the residual $\mathbf{x}_{l+1} - \mathbf{x}_l$.

**Why the degree grows.** Theorem 3.1, proved by induction in the appendix: with $g_l(\mathbf{x}_0) = \mathbf{x}_l^\top \mathbf{w}_l$,
$$g_l(\mathbf{x}_0) = \sum_{p=1}^{l+1}\ \sum_{|\mathbf{i}|=p}\ \prod_{j=0}^{l}(\mathbf{x}_0^\top \mathbf{w}_j)^{i_j}$$
Expand that and you get every monomial of degree $1$ through $l+1$, with coefficient $c_{\bm\alpha}$ a distinct product-sum of the $\mathbf{w}$ entries. A degree-$n$ polynomial in $d$ variables has $O(d^n)$ coefficients; the cross network reproduces the *set of terms* with $O(d)$ parameters, at the price that the coefficients are tied to each other rather than free.

**Relation to [[Factorization Machines (ICDM)|FMs]].** An FM gives feature $x_i$ a vector $\mathbf{v}_i$ and sets the weight of $x_i x_j$ to $\langle \mathbf{v}_i, \mathbf{v}_j\rangle$. DCN gives $x_i$ a set of scalars $\{w_k^{(i)}\}_{k=0}^{l}$ and the weight of $x_i x_j$ is a product of entries drawn from $x_i$'s set and $x_j$'s set. Same parameter-sharing spirit — which is why it generalises to pairs that never co-occur in training — but FM stops at degree 2, and DCN goes to degree $l+1$ with parameters still growing only linearly (higher-order FMs blow up).

**The deep tower.** Standard MLP, $\mathbf{h}_{l+1} = \mathrm{ReLU}(W_l \mathbf{h}_l + \mathbf{b}_l)$, with $d\,m + m + (m^2+m)(L_d-1)$ parameters — quadratic in width, which is the whole memory contrast.

**Combination and loss.**
$$p = \sigma\!\left([\mathbf{x}_{L_1}^\top, \mathbf{h}_{L_2}^\top]\,\mathbf{w}_{\text{logits}}\right)$$
$$\text{loss} = -\frac{1}{N}\sum_{i=1}^{N} y_i \log p_i + (1-y_i)\log(1-p_i) + \lambda \sum_l \|\mathbf{w}_l\|^2$$
Plain [[Cross Entropy|log loss]] plus L2. Both towers are trained **jointly**, so each is aware of the other during optimisation.

**Training details that mattered.** TensorFlow. [[Adam- A Method for Stochastic Optimization|Adam]], batch size 512. [[Batch Normalization]] applied to the deep tower only. Gradient clip norm 100. Learning rate grid-searched from 0.0001 to 0.001 in steps of 0.0001. Early stopping at 150,000 steps — past that it overfits. Data: 11 GB of Criteo logs, ~41M records over 7 days; days 1–6 train, day 7 split into equal validation and test halves.

## Ablation Studies and Experiments

**Criteo, best test logloss.** On this dataset a 0.001 improvement is considered commercially meaningful.

| Model | Logloss |
|---|---|
| **DCN** | **0.4419** |
| Deep Crossing | 0.4425 |
| DNN (DCN with 0 cross layers) | 0.4428 |
| FM | 0.4464 |
| LR (42 hand-picked crosses) | 0.4474 |

Over 10 independent runs at the best setting: DCN $0.4422 \pm 9\times10^{-5}$, DNN $0.4430 \pm 3.7\times10^{-4}$, Deep Crossing $0.4430 \pm 4.3\times10^{-4}$. Note DCN's variance is 4× smaller too.

Best DCN config: **2 deep layers of width 1024 and 6 cross layers**. Best DNN: 5 deep layers of 1024. The winning DCN has the *deepest* cross stack in the search range, which the authors read as evidence that high-degree interactions genuinely help.

**Memory efficiency — the headline ablation.** Parameters needed to hit a target logloss (embedding table excluded, identical in both):

| Target logloss | 0.4430 | 0.4460 | 0.4470 | 0.4480 |
|---|---|---|---|---|
| DNN | $3.2\times10^6$ | $1.5\times10^5$ | $1.5\times10^5$ | $7.8\times10^4$ |
| DCN | $7.9\times10^5$ | $7.3\times10^4$ | $3.7\times10^4$ | $3.7\times10^4$ |

Roughly an order of magnitude at the tight end.

**Fixed budget, best achievable loss:**

| #Params | $5\times10^4$ | $1\times10^5$ | $4\times10^5$ | $1.1\times10^6$ | $2.5\times10^6$ |
|---|---|---|---|---|---|
| DNN | 0.4480 | 0.4471 | 0.4439 | 0.4433 | 0.4431 |
| DCN | 0.4465 | 0.4453 | 0.4432 | 0.4426 | 0.4423 |

The gap narrows as the DNN gets huge but never closes — the authors argue the cross network captures something a big MLP structurally struggles with.

**Matched-architecture sweep.** For every combination of 2–5 deep layers × 32–1024 nodes, DCN beat the same-shape DNN. Differences ranged from $-0.28\times10^{-2}$ (2 layers, 32 nodes) to $-0.00\times10^{-2}$ (5 layers, 256 nodes). Consistency across all 24 cells is the argument that this is not initialisation luck.

**Where more cross layers stop helping.** Going from 0 → 1 cross layer gives a clear drop in validation logloss for every setting tested. Beyond that it splits: some configurations keep improving, others **fluctuate or get slightly worse**. So the degree-$l{+}1$ terms are not universally useful, and $L_c$ is a real hyperparameter, not a "more is better" dial.

**Non-CTR, dense data.** Forest covertype (581k × 54): DCN 0.9740 test accuracy with 8 cross layers of size 54 + 6 deep layers of 292, versus 0.9737 for both DNN (7 × 292) and Deep Crossing — a win, but a tiny one, bought mainly in memory. Higgs (11M × 28): DCN logloss 0.4494 (4 cross + 4 deep of 209) vs DNN 0.4506 (10 deep of 196), at **half** the memory. On these the raw input vector was fed straight into the cross network, no embedding.

**What did not work.** [[Regularization|L2]] and [[Dropout- A Simple Way to Prevent Overfitting|dropout]] were *not* effective — the only regulariser they kept was early stopping. And they **skipped the Wide & Deep comparison entirely**, admitting there is no known good way to select the cross features W&D needs, so the comparison would be arbitrary. That is an honest dodge, but it means the paper never actually beats its closest rival head to head.

## Worth Remembering

- **The rank-one catch the paper does not mention.** Ignore biases and unroll: $\mathbf{x}_1 = \mathbf{x}_0(\mathbf{x}_0^\top\mathbf{w}_0) + \mathbf{x}_0 = \alpha_1 \mathbf{x}_0$ for a scalar $\alpha_1$. By induction $\mathbf{x}_l = \alpha_l \mathbf{x}_0$. So the cross tower's output is always a **scalar multiple of the input vector** — it lives on a one-dimensional ray. All that polynomial richness is squeezed into a single learned scalar. This is exactly what DCN-V2 (2020) fixes by replacing $\mathbf{w}_l \in \mathbb{R}^d$ with a matrix $W_l \in \mathbb{R}^{d\times d}$ (and a low-rank version for cost). If you are implementing this today, implement V2.
- **The comparison is a bit rigged in DCN's favour on memory.** Excluding the embedding table from parameter counts is fair for the cross-vs-deep argument, but in a production system the embedding tables dominate totally — see [[Deep Learning Recommendation Model (DLRM)|DLRM]]. The "40% of the memory" figure is 40% of the non-embedding memory.
- **0.001 logloss is the stated bar for significance on Criteo**, and DCN beats DNN by 0.0009 on best-run and 0.0008 on 10-run mean. It is *right at* the threshold the authors set. The memory argument is stronger than the accuracy argument.
- The cross network sees the *concatenated embedding vector*, not fields. So a "cross" here is between individual embedding *dimensions*, not between semantic fields like `country × device`. The explicit polynomial story is cleaner in the maths than in the semantics.
- Deep Crossing (0.4425) is essentially a ResNet over stacked embeddings and lands very close to DCN. Residual structure alone gets you most of the way; the explicit multiplication buys a small extra margin plus the memory.
- Follow-up questions: does the benefit survive when the deep tower is very large (Table 3 suggests shrinking gains)? Does the degree bound matter, or would degree 2 suffice — nobody ablated "cross layers = 2 forever"? And why did L2 and dropout fail here when they work elsewhere in CTR?

## Links

Related: [[Wide & Deep Learning for Recommender Systems]] · [[Factorization Machines (ICDM)]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Deep Learning Recommendation Model (DLRM)]] · [[Deep Interest Network for CTR Prediction (DIN)]] · [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] · [[Ad Click Prediction- a View from the Trenches (KDD)]] · [[Batch Normalization]] · [[Adam- A Method for Stochastic Optimization]] · [[Cross Entropy]] · [[Regularization]] · [[Dropout- A Simple Way to Prevent Overfitting]] · [[Recommender Systems - Evolution]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]]

New topics worth writing: DCN-V2 and low-rank cross layers, Field-aware Factorization Machines (FFM), Deep Crossing, xDeepFM and the Compressed Interaction Network, Weierstrass approximation theorem, Criteo Display Ads benchmark, embedding dimension heuristics for high-cardinality categoricals
