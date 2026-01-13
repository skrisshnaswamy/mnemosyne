---
title: "Search-based User Interest Modeling (SIM)"
authors: ["Pi Qi", "Xiaoqiang Zhu", "Guorui Zhou", "Yujing Zhang", "Zhe Wang", "Lejian Ren", "Ying Fan", "Kun Gai"]
year: 2020
arxiv: "2006.05639"
url: https://arxiv.org/abs/2006.05639
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, transformers]
---
## The Core Idea

A user on Taobao may have clicked 50,000 things over their life. That history is valuable — it tells you they buy camera gear, or that they bought a crib eight months ago. But no CTR model can run attention over 50,000 items in 30 milliseconds.

The previous best answer, MIMN, squeezed the whole history into a **fixed-size memory matrix** that gets updated with each new click. The memory is built without knowing what ad you are about to score. That works up to ~1000 behaviours. Past that, the memory turns to mush: everything the user ever did is averaged into the same few slots, and the signal for any one candidate item drowns in noise.

The insight of SIM is boring and correct: **do not compress the history, search it.** Before you do any expensive modelling, throw away the 99% of behaviours that have nothing to do with the ad you are scoring. If the candidate ad is a tennis racket, retrieve only the sports-equipment clicks. That leaves a couple of hundred items, which is a length [[Deep Interest Network for CTR Prediction (DIN)]]-style attention can already handle comfortably.

So the pipeline is two cascaded stages:

1. **General Search Unit (GSU)** — cheap, sub-linear, filters $T \approx 54{,}000$ behaviours down to top-$K \approx 100$–200 relevant to the candidate.
2. **Exact Search Unit (ESU)** — expensive multi-head attention over those $K$ survivors.

> [!NOTE] Cascaded search paradigm
> Instead of learning one compressed user vector, retrieve a *candidate-conditional* sub-sequence first, then model it precisely. Retrieval buys you scale; attention buys you accuracy. ^cascaded-search

What this unlocks: maximum behaviour length goes from 1000 (MIMN) to **54,000** — 54× — while adding only **5 ms** of latency over MIMN. Deployed on Alibaba display advertising since 2019, giving **+7.1% CTR** and **+4.4% RPM** over MIMN in a month-long A/B test.

## The Methodology

Behaviour list $\mathbf{B} = [\mathbf{b}_1; \dots; \mathbf{b}_T]$. For each behaviour, compute a relevance score $r_i$ against the candidate item $a$, keep top-$K$.

**Two flavours of GSU:**

$$r_i = \begin{cases} \mathrm{Sign}(C_i = C_a) & \text{hard-search} \\ (W_b \mathbf{e}_i) \odot (W_a \mathbf{e}_a)^\top & \text{soft-search} \end{cases}$$

- **Hard-search** has no parameters at all. Keep the behaviour if its category id matches the candidate's category id. That is the whole rule.
- **Soft-search** is a learned inner product in embedding space. To get top-$K$ over tens of thousands of items in sub-linear time they use ALSH (asymmetric LSH for maximum inner product search) — the same family of trick as [[Efficient and robust approximate nearest neighbor search using HNSW]].

A subtlety on soft-search: you cannot reuse the embeddings trained on short-term behaviour, because long-term and short-term behaviour distributions differ. So soft-search gets its **own auxiliary CTR task** on long-term data. It pools behaviours by their relevance scores,

$$\mathbf{U}_r = \sum_{i=1}^{T} r_i \mathbf{e}_i,$$

concatenates $\mathbf{U}_r$ with $\mathbf{e}_a$, and feeds an MLP with a click label. If $T$ is too big even for this, sample a random sub-sequence — it has the same distribution.

**ESU.** Takes the $K$ survivors $\mathbf{B}^*$. Each survivor also gets a **time-interval embedding**: $\Delta_j$ = days between behaviour $j$ and the candidate impression, bucketed and embedded. Concatenate: $\mathbf{z}_j = \mathrm{concat}(\mathbf{e}_j^*, \mathbf{e}_j^t)$. Then multi-head target attention:

$$\mathbf{att}^i_{score} = \mathrm{Softmax}(W_{bi}\mathbf{z}_b \odot W_{ai}\mathbf{e}_a), \qquad \mathbf{head}_i = \mathbf{att}^i_{score}\,\mathbf{z}_b$$

$U_{lt} = \mathrm{concat}(head_1, \dots, head_q)$ goes into the MLP alongside short-term behaviour features and everything else. This is [[Attention Is All You Need|multi-head attention]] with the candidate item as the single query.

**Joint loss.** Both stages train together:

$$Loss = \alpha\,Loss_{GSU} + \beta\,Loss_{ESU}$$

with $\alpha = \beta = 1$ for soft-search, and $\alpha = 0$ for hard-search (nothing to train — it is non-parametric).

**Hyperparameters:** [[Adam- A Method for Stochastic Optimization|Adam]], learning rate 0.001 with exponential decay, FCN $200 \times 80 \times 2$, embedding dimension **4** (yes, four), metric AUC.

**Serving.** The killer engineering piece is the **User Behavior Tree (UBT)** — a Key-Key-Value index built offline and refreshed daily:

$$\texttt{user\_id} \rightarrow \texttt{category\_id} \rightarrow [\text{items}]$$

22 TB, distributed. At request time the GSU is a hash lookup, not a computation, so its latency is essentially free. A request scores hundreds of candidate ads but those ads span fewer than 20 distinct categories, so you do at most 20 lookups. Each retrieved sub-sequence is truncated to 200 (usually under 150 anyway). ESU attention is optimised with deep kernel fusion.

## Ablation Studies and Experiments

**Public datasets** (Amazon Books: max length 100, split 10 short / 90 long. Taobao: max length 500, split 100 short / 400 long).

| Model | Taobao AUC | Amazon AUC |
|---|---|---|
| DIN (short-term only) | 0.9214 | 0.7276 |
| Avg-Pooling Long DIN | 0.9281 | 0.7280 |
| MIMN | 0.9278 | 0.7396 |
| SIM (soft) | 0.9416 | **0.7510** |
| SIM (soft) + time info | **0.9501** | — |

Note MIMN is *worse than average-pooling* on Taobao. Compressing into memory buys nothing there.

**The two-stage ablation** — the important table. What happens if you only do stage one?

| Operation | Taobao | Amazon |
|---|---|---|
| Avg-pool, no search | 0.9281 | 0.7280 |
| Only stage 1 (hard) + avg-pool | 0.9330 | 0.7365 |
| Only stage 1 (soft) + avg-pool | 0.9357 | 0.7342 |
| SIM (hard), both stages | 0.9332 | 0.7413 |
| SIM (soft), both stages | 0.9416 | 0.7510 |

Read it this way: **most of the gain on Taobao comes from filtering, not from the attention.** Hard filter alone: 0.9281 → 0.9330. Adding full ESU attention on top of the hard filter: 0.9330 → 0.9332, essentially nothing. Soft-search is where the second stage pays: 0.9357 → 0.9416. On Amazon the second stage helps in both cases (+0.005 hard, +0.017 soft).

The honest summary is that **denoising is doing most of the work**. Long behaviour sequences contain massive noise, and simply deleting the irrelevant part recovers most of the value.

**Industrial dataset** (0.29B users, 12.2B instances, 49 days train / 1 day test, 180-day long-term window, 14-day short-term window, >30% of samples have >10,000 behaviours):

| Model | AUC |
|---|---|
| DIEN | 0.6452 |
| MIMN | 0.6541 |
| SIM (hard) | 0.6604 |
| SIM (hard) + time info | 0.6624 |
| SIM (soft) | 0.6625 |

**What they did not ship: soft-search.** It wins, but by 0.0021 AUC over plain hard-search — and by 0.0001 over hard-search with time info. Meanwhile it needs online ANN search instead of an offline hash table. They measured the overlap: the behaviours hard-search keeps cover **75%** of what soft-search keeps. On e-commerce, same-category ≈ similar item, so the learned similarity mostly rediscovers the category taxonomy. They chose hard-search.

**Is the gain really from long-term interest?** They invented a diagnostic metric, $d_{category}$ — days since the user's last click in the candidate's category ($-1$ if never). Compared to DIEN online:

| Model | avg $d_{category}$ | $p(d_{category} > -1)$ |
|---|---|---|
| DIEN | 11.2 | 0.91 |
| SIM | **13.3** | **0.94** |

Below 14 days the two distributions are identical (both models see the 14-day short-term features). Above 14 days SIM takes a visibly larger share. So yes — the lift comes from genuinely long-range interests, not from doing short-term better.

**Latency.** DIEN maxes out at 200 QPS throughput. MIMN and SIM both handle much more; SIM with 10,000+ un-truncated behaviours costs only **+5 ms** versus MIMN with behaviours truncated to 1000.

## Worth Remembering

- **The limitation the authors state:** the search unit shares one formula and one set of parameters across all users. A tennis racket retrieves sports clicks for everyone. They want per-user models that organise a lifelong history "with respect to personal conscious" — which is a research programme, not a fix.
- **Hard-search works because of a data property, not a principle.** In e-commerce, category ids are a good, human-curated similarity function. In a domain without a clean taxonomy (news, video, short-form feed), hard-search is likely much weaker and soft-search's 75%-overlap number would drop. Do not port the hard-search decision without checking your taxonomy.
- **The offline index is the actual product.** 22 TB, rebuilt daily. That means the long-term history is stale by up to a day, which is fine because long-term interests move slowly; the last 14 days are handled by the separate short-term features. This split is the design, not an accident — see [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] for what maintaining a 22 TB daily-refresh index costs you.
- **Embedding dimension 4.** For an industrial CTR model with 0.6B items. Sparse-feature models really are that different from language models — compare the sizing discussions in [[Deep Learning Recommendation Model (DLRM)]].
- **Retrieve-then-rerank, applied inside a single model.** Stage one is cheap recall, stage two is expensive precision. Exactly the funnel structure of [[Deep Neural Networks for YouTube Recommendations (RecSys)]], but folded into the user-representation step instead of the item-candidate step. Same shape as [[ColBERT- Efficient and Effective Passage Search via Late Interaction]] pushing cheap retrieval ahead of expensive interaction.
- **Open question:** the joint loss $\alpha Loss_{GSU} + \beta Loss_{ESU}$ never gets a sweep. With $\alpha = \beta = 1$ for soft-search we do not know whether the auxiliary GSU loss is helping the shared embeddings or fighting the main task.
- Time-interval embeddings gave a real, cheap win (+0.0085 AUC on Taobao, +0.0020 on industrial). If a behaviour is 5 months old, tell the model so. Related to but simpler than the temporal modelling in [[Deep Interest Evolution Network (DIEN)]].

## Links

Related: [[Deep Interest Network for CTR Prediction (DIN)]] · [[Deep Interest Evolution Network (DIEN)]] · [[Attention Is All You Need]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Self-Attentive Sequential Recommendation (SASRec)]] · [[Behavior Sequence Transformer for E-commerce (BST)]] · [[Efficient and robust approximate nearest neighbor search using HNSW]] · [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)]] · [[ColBERT- Efficient and Effective Passage Search via Late Interaction]] · [[Deep Learning Recommendation Model (DLRM)]] · [[Actions Speak Louder than Words- Generative Recommenders (HSTU)]] · [[PinnerFormer- Sequence Modeling for User Representation at Pinterest]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Cross Entropy]] · [[Adam- A Method for Stochastic Optimization]]

New topics worth writing: MIMN and the User Interest Center serving design, Asymmetric LSH for maximum inner product search, target attention vs self-attention in CTR models, lifelong user behaviour indexing, RPM as an advertising metric, deep kernel fusion for attention serving
```
