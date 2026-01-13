---
title: "Monolith: Real Time Recommendation System With Collisionless Embedding Table"
authors: ["Zhuoran Liu", "Leqi Zou", "Xuan Zou", "Caihua Wang", "Biao Zhang", "Da Tang", "Bolin Zhu", "Yijie Zhu", "Peng Wu", "Ke Wang", "Youlong Cheng"]
year: 2022
arxiv: "2209.07663"
url: https://arxiv.org/abs/2209.07663
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, theory]
---
## The Core Idea

Two things break when you run a recommender at TikTok scale with stock TensorFlow.

**One: the embedding table is the wrong shape.** TensorFlow's `Variable` is a fixed-size dense block. But the set of user IDs and video IDs in a feed app is huge and grows every day. The standard fix is the **hash trick**: squash the ID space down with a hash function and accept that two IDs share one embedding row. Monolith's argument is that this quietly costs you accuracy, and the cost gets worse over time. The assumption behind the hash trick — that IDs are spread evenly, so collisions are rare and harmless — is false. IDs are long-tail. A popular video appears millions of times; most appear ten times. When a hot ID collides with another hot ID, you have blended two very different things into one vector. So they replaced the table with a real key-value hash map (Cuckoo hashing) that never collides, and added eviction rules to stop it eating all the memory.

**Two: training and serving are separated by hours.** User taste moves fast — this is **concept drift**, the data distribution shifting under you. A model trained on yesterday's log is already stale. Monolith closes the loop: the training parameter servers push updated embeddings to the serving parameter servers every *minute*.

> [!NOTE] Concept drift
> The statistical relationship between features and labels changes over time. Not a bug in your model — a property of the world. In recommendation it is the dominant effect. ^concept-drift

The unlocking insight is a trade. A terabyte-scale model cannot be shipped whole every minute. But in any one minute, only a tiny handful of embedding rows actually changed, and the dense network weights barely moved at all. So sync the sparse part often and the dense part daily, and accept that the served model is internally *inconsistent* — new embeddings, old dense weights. It works. They also traded away reliability: snapshot the parameter servers once a day instead of constantly, and eat the rare loss.

## The Methodology

**Overall shape.** Standard TensorFlow Worker–Parameter Server. Workers do forward and backward passes; PS machines hold parameters and apply [[Backpropagation|gradient]] updates. Parameters split into two kinds:

- **Dense** — the weights of the neural network.
- **Sparse** — the embedding tables for categorical features.

Both live on the PS. Both are part of the TensorFlow graph.

**The collisionless HashTable.** A new TensorFlow *resource op*, sitting where `Variable` normally would, so lookups and updates are native TF ops and the rest of the stack does not need to know. Under the hood: **Cuckoo hashing**.

> [!NOTE] Cuckoo hashing
> Keep two tables $T_0, T_1$ with two different hash functions $h_0, h_1$. To insert $A$, try slot $h_0(A)$ in $T_0$. If $B$ already sits there, kick $B$ out and re-insert $B$ into $T_1$ at $h_1(B)$, which may kick out someone else, and so on. Stop when everything settles, or rehash if you find a cycle. Lookup and delete are worst-case $O(1)$; insert is expected amortised $O(1)$. ^cuckoo-hashing

Because insertion never overwrites another key, every ID gets its own embedding. The table grows on demand.

**Keeping memory finite.** Two heuristics, both from watching production:

1. **Admission filtering.** An ID must appear more than a threshold number of times before it earns a row. The threshold is a per-model hyperparameter. A probabilistic filter cuts memory further. Rationale: rare IDs are underfit anyway, so their embeddings are noise.
2. **Expiry.** Each ID is timestamped and evicted after a tunable period of inactivity. Dead users, stale videos. The expiry window is set per embedding table, so features that need long memory can keep it.

**The streaming pipeline.** Two Kafka queues — one logging user actions (click, like), one logging the features that were served. An Apache Flink job, the **online joiner**, matches them up and emits training examples into a third Kafka queue.

Three real problems the joiner handles:

- *Actions arrive out of order.* Each request carries a unique key so features and the later action can be paired.
- *Actions arrive very late.* A user might buy days after seeing the item. Keeping all features in RAM does not fit, so features that age past a threshold spill to an on-disk key-value store. The joiner checks the in-memory cache first, disk on a miss.
- *Negatives massively outnumber positives.* They downsample negatives, which shifts the learned distribution toward predicting positive. At serving time they apply a **log-odds correction** so the deployed model is still an unbiased estimator of the true rate.

The example queue is consumed twice: online training workers read it directly; a dump job also writes it to HDFS, and batch training reads from there later. Batch training is a single pass over the data — one epoch, no reshuffling — and is used mainly when the architecture changes and you need to rebuild from history.

**Parameter synchronisation.** The core engineering trick. Three observations drive it:

1. Sparse parameters dominate model size.
2. In a short window, only a small subset of IDs get touched.
3. Dense variables move *slowly*, because momentum-based optimisers ([[Adam- A Method for Stochastic Optimization|Adam]] and friends) average dense gradients over the whole giant dataset, while any single embedding row sees updates from only a few examples.

So: maintain a hash set of **touched keys** since the last sync. Every minute, push only those rows to the serving PS. Push dense weights once a day, scheduled at low-traffic hours.

The bandwidth maths they give: if 100,000 IDs change in a minute and embedding dim is 1024 (so 4 KB per row at fp32), that is $4\text{KB} \times 100{,}000 \approx 400\text{MB}$ per minute. Manageable.

The consequence is a **version skew** — served sparse embeddings are minutes old, served dense weights are up to a day old. They report no visible loss from this.

**Fault tolerance.** Snapshot every training PS once per day. On failure, restore from the snapshot and lose up to a day of updates on that shard.

## Ablation Studies and Experiments

**Does collision actually hurt? (MovieLens ml-25m, DeepFM)**

Setup: 25M ratings, ~162K users, ~62K movies. Ratings converted to binary — $\geq 3.5$ is positive — to mimic production click signals. Model is [[Factorization Machines (ICDM)|DeepFM]] (an FM component plus a dense tower). Metric is AUC.

The collision condition is created by MD5-hashing IDs into a smaller space:

| | User IDs | Movie IDs |
|---|---|---|
| Before hashing | 162,541 | 59,047 |
| After hashing | 149,970 | 57,361 |
| Collision rate | 7.73% | 2.86% |

The collisionless model has higher AUC **from the first epoch** and converges higher. Note the collision rate here is small — under 8% — and it still costs you.

Also worth noting: the collisionless table gives you far more parameters on sparser data, which should raise an overfitting alarm. It does not overfit after convergence. (Compare [[Deep Double Descent- Where Bigger Models and More Data Hurt]] and [[Understanding Deep Learning Requires Rethinking Generalization]] for why extra capacity is not automatically fatal.)

**Production model.** ~1000 embedding tables, very uneven sizes, original ID space $2^{48}$. The baseline uses a **QR-style decomposition** to shrink it — exactly the trick in [[Compositional Embeddings Using Complementary Partitions (QR trick)]]:

$$ID_r = ID \bmod 2^{24}, \quad ID_q = ID \div 2^{24}, \quad E = E_r + E_q$$

This cuts table size from $2^{48}$ to $2^{25}$. Measured by online AUC on live traffic, the collisionless version wins, and stays ahead across days as the distribution drifts.

**Does sync frequency matter? (Criteo Display Ads, DeepFM)**

7 days of chronologically ordered click data. Split: 5 days batch training, 2 days simulated online training. The 2 days are cut into $N$ chronological shards. The simulation loop is: sync train params to serve params → evaluate the serving model on the *next unseen* shard → train on that shard → repeat. $N = 10, 50, 100$ maps to sync intervals of roughly 5 hr, 1 hr, 30 min.

| Sync interval | Avg AUC (online) | Avg AUC (batch) |
|---|---|---|
| 5 hr | $79.66 \pm 0.020$ | $79.42 \pm 0.026$ |
| 1 hr | $79.78 \pm 0.005$ | $79.44 \pm 0.030$ |
| **30 min** | $\mathbf{79.80 \pm 0.008}$ | $79.43 \pm 0.025$ |

Two readings. Online beats batch at every interval. And shorter interval monotonically beats longer — though the returns shrink fast: 5hr→1hr buys 0.12 AUC, 1hr→30min buys 0.02. The batch column barely moves, which is the control: the gains are from freshness, not from the extra training steps.

**Live A/B on a production Ads model**, online training vs batch training, relative online AUC improvement per day:

| Day | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| AUC improvement % | 14.44 | 16.87 | 17.07 | 14.03 | 18.08 | 16.40 | 15.20 |

These numbers are stated as percentage improvement, and they are enormous compared to the Criteo simulation. Read them with care — the paper does not define the baseline precisely.

**The surprising negative result: snapshot frequency does not matter.** They expected minute-level parameter sync to demand frequent snapshots. They stretched the snapshot interval out to **one full day** and saw nearly no quality loss. Their reasoning: with a 0.01% daily PS failure rate and 1000 shards, one shard dies roughly every 10 days, losing one day of updates on that shard. With 15M DAU spread evenly, that is one day of feedback from ~15,000 users every 10 days — 0.01% of DAU. And the dense variables, which move slowly anyway, lose 1/1000th of one day's update. Negligible. So they cut snapshot frequency hard and saved the compute.

**What the ablations really show.** The two contributions are independent and both survive on their own. Collisionlessness is the model-quality lever; sync frequency is the freshness lever. The third, quieter finding is that *consistency is not the thing you need to protect* — neither version consistency between sparse and dense, nor snapshot consistency against failure. Freshness is.

## Worth Remembering

- The dense/sparse asymmetry is the load-bearing observation. Dense weights move slowly because momentum averages over the entire firehose of data; a single embedding row sees gradient from a handful of examples per batch. This is why you can sync them at wildly different rates and why the resulting skew is harmless. This asymmetry shows up everywhere in [[Deep Learning Recommendation Model (DLRM)]]-style architectures.
- **Batch training is one epoch only.** Standard for production recommenders — you have more data than you can pass over twice, and passing twice over old data is worse value than one pass over new data.
- The paper does not report the *memory cost* of collisionlessness versus the hash trick head-on. It claims "roughly similar memory usage" after filtering and expiry, but no table. That is the number you would want before adopting this.
- Admission thresholds and expiry windows are per-table hyperparameters. That is a lot of knobs, tuned by hand, on 1000 tables. The paper is silent on how.
- The log-odds correction after negative sampling is a small detail with big consequences — it is what keeps the model calibrated when you throw away most of your negatives. Related in spirit to the logQ correction in [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)]] and to [[Recommendations as Treatments- Debiasing Learning and Evaluation]].
- The MovieLens collision experiment is small and the deck is a bit stacked — a public dataset with 160K users is not where the hash trick hurts most. The production result is the more convincing evidence, but it is unreproducible.
- Positioning against prior work: XDL and Kraken also use native key-value tables with eviction. Monolith's claim to distinctness is that its `HashTable` is a plain TensorFlow op rather than a bolt-on software or hardware layer — easier to port, easier for TF to optimise around, which matters because this shipped as a B2B product (BytePlus Recommend).
- Open question: does collisionlessness still pay once you have very good long-tail handling elsewhere — frequency filtering, or [[Mixed Dimension Embeddings]] that give rare IDs tiny vectors? The two ideas overlap.

## Links

Related: [[Deep Learning Recommendation Model (DLRM)]] · [[Compositional Embeddings Using Complementary Partitions (QR trick)]] · [[Mixed Dimension Embeddings]] · [[Factorization Machines (ICDM)]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)]] · [[Ad Click Prediction- a View from the Trenches (KDD)]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] · [[Recommender Systems - Evolution]] · [[Actions Speak Louder than Words- Generative Recommenders (HSTU)]]

New topics worth writing: Cuckoo hashing, Concept drift and online learning, DeepFM, Parameter server architectures, Apache Flink and Kafka for ML pipelines, Log-odds correction for negative sampling, Feature eviction and admission policies
