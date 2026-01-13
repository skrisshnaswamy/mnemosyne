---
title: "Graph Convolutional Neural Networks for Web-Scale Recommender Systems (PinSage)"
authors: ["Rex Ying", "Ruining He", "Kaifeng Chen", "Pong Eksombatchai", "William L. Hamilton", "Jure Leskovec"]
year: 2018
arxiv: "1806.01973"
url: https://arxiv.org/abs/1806.01973
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, optimization, vision, scaling]
---
## The Core Idea

Graph convolutional networks (GCNs) learn an embedding for a node by repeatedly mixing in features from its neighbours. Before this paper, every GCN assumed you could hold the whole graph — the full adjacency structure — in memory and multiply feature matrices by powers of the graph Laplacian. That works on a citation graph with 10,000 nodes. Pinterest's graph has 3 billion nodes and 18 billion edges. Roughly 10,000× bigger than anything a GCN had touched.

PinSage makes GCNs work at that scale by never looking at the whole graph. Three changes carry the weight:

1. **Neighbourhoods are sampled by short random walks, not defined by hops.** A node's "neighbours" are the top $T$ nodes most often visited by random walks starting there. This fixes the size of the computation ($T=50$ works best) and, as a bonus, gives each neighbour an *importance weight* — its normalised visit count. Weighting the aggregation by those counts ("importance pooling") is worth a 46% relative jump in hit-rate.
2. **Hard negatives on a curriculum.** With 2 billion items, random negatives are trivially easy to beat. They mine negatives that are *somewhat* related (Personalized PageRank rank 2000–5000) and add one more per epoch. Worth ~12%.
3. **Engineering that actually ships.** A producer–consumer loop where CPUs sample walks and gather features for batch $n{+}1$ while GPUs train on batch $n$ (halves training time), plus a MapReduce pipeline that computes embeddings for all 3 billion pins in under 24 hours without recomputing shared sub-neighbourhoods.

What it unlocks: because the model learns *how to aggregate* rather than learning a lookup table per node, the parameter count is independent of graph size, and you can train on a 20% subgraph and then run inference on the full, growing graph — including pins never seen in training. That is the inductive property, and it is the whole reason this is deployable.

> [!NOTE] Importance-based neighbourhood
> Instead of "all nodes within $k$ hops", define $\mathcal{N}(u)$ as the $T$ nodes with the highest $L_1$-normalised random-walk visit count from $u$. Fixed memory footprint, plus a free importance score for weighted pooling. ^importance-neighborhood

## The Methodology

**The graph.** Bipartite: pins $\mathcal{I}$ (2B) and boards $\mathcal{C}$ (1B), edges = "this pin is on this board" (18B). Each pin $u$ has a feature vector $x_u$: a 4096-d visual embedding (fc6 of a VGG-16 classifier), a 256-d text-annotation embedding (word2vec-style, context = other annotations on the same pin), and the log degree of the node. Concatenated.

**One convolution layer** (Algorithm 1). Given node $u$ with current vector $h_u$, neighbours $\mathcal{N}(u)$ with weights $\alpha$:

$$n_u = \gamma\big(\{\mathrm{ReLU}(Q h_v + q) \mid v \in \mathcal{N}(u)\},\ \alpha\big)$$
$$z_u^{\text{new}} = \mathrm{ReLU}\big(W \cdot \mathrm{concat}(z_u, n_u) + w\big)$$
$$z_u^{\text{new}} \leftarrow z_u^{\text{new}} / \|z_u^{\text{new}}\|_2$$

Three details that matter. $\gamma$ is a **weighted mean** with weights = normalised visit counts (this is importance pooling). The self-vector is **concatenated**, not averaged in as in Kipf & Welling — the authors report "significant" gains from this. And the $L_2$ normalisation both stabilises training and makes downstream approximate nearest-neighbour search cheaper.

Parameters $Q, q, W, w$ are shared across all nodes but distinct per layer. Hidden width $m = 2048$, output width $d = 1024$, depth $K = 2$. After the last layer, a small MLP: $z_u = G_2 \cdot \mathrm{ReLU}(G_1 h_u^{(K)} + g)$.

**Minibatch construction** (Algorithm 2). Start from the target set $\mathcal{M}$, expand backwards $K$ times to collect every node whose vector you will need, then run the $K$ convolutions forward. The trick that makes it fast on GPU is **re-indexing**: build a small subgraph $G'$ containing only these nodes plus a small feature matrix ordered to match $G'$'s indices, ship both to GPU once at the start of the iteration. After that, zero CPU–GPU chatter during the convolutions.

**Loss.** Max-margin ranking on positive pairs $(q, i) \in \mathcal{L}$:

$$J(z_q, z_i) = \mathbb{E}_{n_k \sim P_n(q)} \max\{0,\ z_q \cdot z_{n_k} - z_q \cdot z_i + \Delta\}$$

Positives come from engagement logs: a user interacted with pin $i$ right after pin $q$. 1.2 billion such pairs.

**Negatives.** 500 random negatives are **shared across the whole minibatch** — a big compute saving, and empirically no worse than per-example sampling. On top of that, 6 hard negatives per pin, drawn from PPR ranks 2000–5000 w.r.t. $q$. Hard negatives alone double the epochs to convergence, so they use a curriculum: epoch 1 has none, epoch $n$ has $n-1$ hard negatives. Total training signal: 7.5 billion examples.

**Scale and optimisation.** Batch sizes 512–4096 across 16 Tesla K80s, multi-tower synchronous SGD. Linear-scaling-rule warmup over the first epoch, then exponential decay (Goyal et al.'s recipe). 500 GB RAM, Linux HugePages (2 MB pages instead of 4 KB) so feature lookup does not thrash the TLB. Training data 18 TB; output embeddings 4 TB.

**Inference by MapReduce.** Naively running Algorithm 2 per node recomputes shared neighbours over and over. Instead: job 1 projects every pin through $Q$ once; job 2 joins those projections onto the boards they sit on and pools to get board embeddings; repeat for layer 2. Each latent vector is computed exactly once. 3 billion embeddings in under 24 hours on 378 AWS d2.8xlarge nodes.

**Serving.** LSH for approximate KNN, then a two-level retrieval with the Weak-AND operator. Embeddings are precomputed and sat in a database, so serving is a lookup.

## Ablation Studies and Experiments

Task: given query pin $q$ from a held-out engagement pair $(q,i)$, retrieve top-500 neighbours from a 5-million-pin sample. **Hit-rate** = fraction of queries where $i$ made the top 500. **MRR** is scaled: $\frac{1}{n}\sum \frac{1}{\lceil R_{i,q}/100 \rceil}$, so rank 1000 vs 2000 is still distinguishable.

| Method | Hit-rate | MRR |
|---|---|---|
| Visual (VGG fc6 NN) | 17% | 0.23 |
| Annotation (text NN) | 14% | 0.19 |
| Combined (2-layer MLP on both) | 27% | 0.37 |
| max-pooling (GraphSAGE-best, no hard neg) | 39% | 0.37 |
| mean-pooling | 41% | 0.51 |
| mean-pooling-xent | 29% | 0.35 |
| mean-pooling-hard | 46% | 0.56 |
| **PinSage** (importance pooling + hard neg) | **67%** | **0.59** |

Reading the ladder: graph structure at all is the biggest single win (27% → 41%). Hard negatives add 5 points (41 → 46). Importance pooling adds 21 points on top (46 → 67) — the largest single algorithmic contribution, and the reason random-walk-defined neighbourhoods matter beyond just bounding memory.

**What did not work.**
- **Cross-entropy loss** (the GraphSAGE objective) was *worse* than max-margin at this scale: 29% vs 41% hit-rate, MRR 0.35 vs 0.51. A 12-point regression from swapping only the loss.
- **Max-pooling** underperformed mean-pooling (39% vs 41%, and MRR 0.37 vs 0.51 — a big MRR gap).
- **Kipf & Welling-style GCN variants** "performed significantly worse in development tests" and were dropped without reported numbers.
- **Averaging the self-vector into the neighbourhood** (instead of concatenating) lost significant performance.
- **More training data stopped helping.** A 300M-item subgraph matched full-graph hit-rate, at 6× less runtime.

**Neighbourhood size $T$** (Table 4): 10 → 60% HR / 20h; 20 → 63% / 33h; 50 → 67% / 78h. Clear diminishing returns; they took 50.

**Batch size** (Table 3): 512 → 63.9h total, 1024 → 53.2h, 2048 → **48.8h**, 4096 → 68.4h. Per-iteration cost grows roughly linearly but iterations-to-converge fall, and 4096 stops converging faster. 2048 is the sweet spot.

**Embedding spread.** Kurtosis of pairwise cosine similarity: PinSage 0.43, visual 1.20, annotation 2.49. Lower kurtosis = distances are spread out, not all bunched at one value. Two consequences: more "resolution" to rank items, and lower LSH collision rate, so serving is faster.

**User study.** Head-to-head, users pick the more related recommendation. Among decided cases PinSage wins ~56.5% vs Visual, 72.5% vs Annotation, 60.0% vs Combined, 62.4% vs Pixie. Note the huge "draw" fractions (46–58%) — most pairs are indistinguishable to humans.

**Production A/B.** Homefeed recommendations, metric = repin rate (fraction of shown pins a user saves to a board). 10–30% lift over Annotation and Visual embedding baselines. The abstract's "30% to 100%" refers to engagement across various settings more broadly.

## Worth Remembering

- **Pixie is the baseline that stings.** Pixie is pure random walk, no learning, no features — and PinSage only wins 62.4% of decided head-to-heads against it. The qualitative failure modes are instructive: visual embeddings confused plants with food and logging photos with war photos (same style, wrong semantics); Pixie got the *category* right ("plants") but not the best items within it. PinSage is the combination.
- The offline numbers are much more dramatic than the online ones. 150% relative hit-rate improvement offline; 10–30% repin lift in production. Standard gap — worth calibrating expectations against when reading any retrieval paper.
- **The 46% importance-pooling gain is the load-bearing claim** and it is not separately isolated from other changes in Table 1; PinSage differs from mean-pooling-hard in the pooling function only, so 46% → 67% is the cleanest read on it.
- **Shared negatives across the batch cost nothing.** 500 negatives shared by every example in the minibatch performed the same as independent sampling, at a fraction of the embedding compute. Cheap trick, generalises well beyond graphs — cf. [[Dense Passage Retrieval (DPR)#^in-batch-negatives|in-batch negatives]].
- **Limitation the authors do not dwell on:** no user embeddings. This is item–item. Homefeed recommendation is done by finding pins near the user's *most recently pinned item* — a crude personalisation. Compare [[PinnerFormer- Sequence Modeling for User Representation at Pinterest]], which is Pinterest's later answer to exactly this.
- $K=2$ only. Two hops on a bipartite pin–board graph means pin → board → pin. Deeper was not tried or not reported; on graphs this dense, three hops probably covers most of the graph anyway.
- Hard-negative mining requires an existing PPR system (Pixie). If you do not already have fast personalised random walks in production, that dependency is non-trivial.
- Practical caveat: 500 GB RAM to hold graph + features, 16 GPUs, a 378-node Hadoop cluster for inference. The *algorithm* is scalable; the *deployment* is not cheap.

## Links

Related: [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[PinnerFormer- Sequence Modeling for User Representation at Pinterest]] · [[Real-time Personalization using Embeddings for Search Ranking at Airbnb (KDD, best paper)]] · [[Approximate Nearest Neighbor Negative Contrastive Learning (ANCE)]] · [[Dense Passage Retrieval (DPR)]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[BPR- Bayesian Personalized Ranking from Implicit Feedback]] · [[Efficient and robust approximate nearest neighbor search using HNSW]] · [[Embedding-based Retrieval in Facebook Search]] · [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)]] · [[Recommender Systems - Evolution]] · [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)]] · [[Multi-Interest Network with Dynamic Routing (MIND)]] · [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)]]

New topics worth writing: GraphSAGE (inductive graph representation learning), Graph Convolutional Networks (Kipf & Welling), Personalized PageRank, Pixie random-walk recommender, curriculum learning, locality-sensitive hashing, Weak-AND retrieval operator, max-margin / triplet ranking loss, linear scaling rule for large-batch SGD, MapReduce, kurtosis as an embedding-quality diagnostic
</result>
