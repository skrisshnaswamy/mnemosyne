---
title: "Deep Learning Recommendation Model (DLRM)"
authors: ["Maxim Naumov", "Dheevatsa Mudigere", "Hao-Jun Michael Shi", "Jianyu Huang", "Narayanan Sundaraman", "Jongsoo Park", "Xiaodong Wang", "Udit Gupta", "Carole-Jean Wu", "Alisson G. Azzolini", "Dmytro Dzhulgakov", "Andrey Mallevich", "Ilia Cherniavskii", "Yinghai Lu", "Raghuraman Krishnamoorthi", "Ansha Yu", "Volodymyr Kondratenko", "Stephanie Pereira", "Xianjie Chen", "Wenlin Chen"]
year: 2019
arxiv: "1906.00091"
url: https://arxiv.org/abs/1906.00091
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, transformers, theory]
---
## The Core Idea

DLRM is Facebook's production click-prediction model, written down plainly and open-sourced. The architecture itself is not exotic — the contribution is the *statement*: here is what a real, industrial recommender looks like, here is why it is a strange beast for hardware, and here is a benchmark you can run.

The design fuses two lineages that had been separate.

1. **Recommender systems**: users and items get [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)#|latent vectors]], and a dot product between them predicts a rating.
2. **Predictive analytics / CTR**: you have a pile of features, some numeric, some categorical, and you want $P(\text{click})$.

DLRM says: treat *every* categorical feature (user id, ad id, country, device) as its own embedding table. Push the numeric features through a small MLP so they come out the same width as an embedding. Now you have a bag of $d$-dimensional vectors that all live in the same space. Take the dot product of **every pair**. That is exactly what a [[Factorization Machines (ICDM)|factorization machine]] does, but with learned dense features included. Concatenate those pairwise scores with the dense vector, run one more MLP, sigmoid, done.

The bet being made: **second-order interactions are enough**. Deep & Cross and xDeepFM build explicit third-, fourth-, higher-order crosses. DLRM argues the extra compute and memory is not worth it.

The second, more lasting contribution is systems. A DLRM is not like a CNN or a Transformer. The embedding tables hold billions of parameters — multiple GB each — but each forward pass touches only a handful of rows. The MLPs are tiny in memory but expensive in FLOPs. So the model is **memory-bandwidth bound in one half and compute bound in the other**, and you cannot parallelise it with one strategy. That observation drove a decade of recommender hardware work.

> [!NOTE] Sparse vs dense features ^sparse-dense
> **Dense (continuous)** features are numbers: age, price, count of past clicks. **Sparse (categorical)** features are IDs from a huge vocabulary, fed in as one-hot or multi-hot vectors. Sparse features are why recommenders have billions of parameters — one embedding row per possible ID.

## The Methodology

**Embedding lookup.** An embedding table is $W \in \mathbb{R}^{m \times d}$ with $m$ categories and dimension $d$. A lookup is a one-hot vector times the table:

$$\bm{w}_i^T = \bm{e}_i^T W$$

For multi-hot features (a user's list of liked pages), you take a weighted sum of rows. A whole mini-batch of $t$ lookups is one sparse matrix product $S = A^T W$ where $A = [\bm{a}_1, \dots, \bm{a}_t]$. In code this is `nn.EmbeddingBag` (PyTorch) or `SparseLengthSum` (Caffe2).

**Bottom MLP.** The $n_{\text{dense}}$ continuous features go through an MLP whose *output width equals the embedding dimension $d$*. This is the key plumbing detail — it makes the dense features just another vector in the pile, so the dot products are well-defined.

$$\hat{y} = W_k\sigma(W_{k-1}\sigma(\dots\sigma(W_1\bm{x} + \bm{b}_1)\dots) + \bm{b}_{k-1}) + \bm{b}_k$$

**Interaction layer.** Stack the $n_{\text{cat}}$ embedding vectors plus the one dense vector into a matrix, compute the full Gram matrix (all pairwise dot products), and keep the strictly upper triangle. This is the FM term:

$$\hat{y} = b + \bm{w}^T\bm{x} + \bm{x}^T\,\texttt{upper}(VV^T)\,\bm{x}$$

`bmm` / `BatchMatMul` in the frameworks.

**Top MLP.** Concatenate the triangle of dot products with the raw dense-MLP output (a skip connection for the dense path), run another MLP, then sigmoid. Loss is binary cross entropy — see [[Cross Entropy]].

**The parallelism scheme.** This is the part worth remembering.

- **Embeddings → model parallel.** Each device owns some tables. Replicating them (data parallel) is impossible; they do not fit.
- **MLPs → data parallel.** Small weights, big compute. Each device gets a slice of the mini-batch, gradients merged with `allreduce`.

Between the two halves you need an **all-to-all** shuffle. After lookup, device $j$ holds embeddings for *its tables* for *all* samples in the batch. But the top MLP on device $j$ needs *all tables* for *its slice* of samples. So you transpose across the network — they call it a **butterfly shuffle**. Neither PyTorch nor Caffe2 supported hybrid model+data parallelism in 2019, so they wrote it by hand with explicit copies.

> [!NOTE] Hybrid model/data parallelism ^hybrid-parallel
> Different parts of the same model use different parallel strategies, joined by an all-to-all collective. Forced by DLRM's split personality: huge-memory/low-compute embeddings, low-memory/high-compute MLPs. This is why recommender training clusters look nothing like LLM clusters. See [[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models]] for the contrasting approach when *all* parameters are dense.

**Data.** Three generators ship with the code: (1) purely random, for hardware benchmarking; (2) **synthetic traces** that reproduce the *reuse-distance distribution* of a real trace — you profile the stack distance between repeated accesses to an embedding row, then resample it. This gives realistic cache hit/miss behaviour without releasing private data; (3) public Criteo Kaggle (45M samples, 7 days) and Criteo Terabyte (24 days), both with 13 dense + 26 categorical features. Dense features get $\log(1+x)$; missing values map to index 0.

## Ablation Studies and Experiments

This is a thin experimental paper. It is a benchmark release, not a results paper, and the authors are honest about it.

**Accuracy vs Deep & Cross Network (DCN)**, Criteo Kaggle, one epoch, no regularisation, both sized to ~540M parameters:

- DLRM: bottom MLP $512 \to 256 \to 64$, top MLP $512 \to 256$, embedding dim 16.
- DCN: six cross layers, deep net $512 \to 256$.

DLRM gets "slightly higher" training and validation accuracy under both SGD and Adagrad. **No exact numbers are printed** — only accuracy curves in Figure 5, and the paper explicitly says it did not tune hyperparameters and ran DCN "as-is". Treat this as a sanity check, not a win. It is precisely the kind of comparison [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|Dacrema et al.]] and [[On the Difficulty of Evaluating Baselines]] warn about.

**Operator profiling**, sample model with 8 categorical features (1M rows each, dim 64) and 512 dense features, 2048K samples in 1K mini-batches, on Big Basin (dual Xeon 6138 + 8× V100 16GB):

| Device | Time |
|---|---|
| CPU (Caffe2) | ~256 s |
| GPU (Caffe2) | ~62 s |

The breakdown is the finding: embedding lookups and fully-connected layers dominate. On **CPU**, the FC layers eat a large slice. On **GPU**, FC layers are "almost negligible" — the GPU chews through the matmuls instantly and you are left staring at the embedding gather, which is random memory access, not arithmetic. That single sentence is the reason recommender inference hardware chases bandwidth and capacity rather than FLOPs.

**What was not shown / what is missing:** no ablation on embedding dimension, no ablation on whether the pairwise dot product beats replacing it with an MLP (the [[Recommender Systems - Evolution#Neural Matrix Factorization (NeuMF)|NCF]] choice), no test of the central claim that higher-order interactions are unnecessary, no scaling study of the parallelism scheme (deferred to "forthcoming work"), no Terabyte-dataset numbers. The argument against higher-order crosses is made rhetorically, not empirically.

## Worth Remembering

- **The dot product is a design choice with a cost model.** DLRM treats each embedding vector as one indivisible unit representing one category, so a pair of features gives you *one* scalar. Deep & Cross treats each *element* of each vector as a crossable unit, so dimensionality explodes. That is the whole architectural argument in one sentence.

- **Embedding dimension must match across all tables**, and the bottom MLP's output must match it too. Otherwise no dot products. This constrains you: a feature with 10 categories and a feature with 100M categories both get dimension $d$. [[Matryoshka Representation Learning]] and mixed-dimension embedding work later attacked exactly this.

- **The synthetic trace generator is underrated.** Recording the stack-distance distribution and resampling it lets you benchmark cache behaviour on a proxy for private data. Useful trick whenever you want to publish a systems benchmark without publishing the data.

- **Practical caveats for using it:** the paper's own numbers use fp32 and explicit copies for the all-to-all. Production DLRM has since moved to lower-precision embeddings, row-wise Adagrad, and NCCL-based all-to-all. Do not read the perf numbers as a ceiling. Also, embeddings are typically trained with sparse-gradient optimisers (row-wise Adagrad), not vanilla [[Adam- A Method for Stochastic Optimization|Adam]] — only the rows you touched get updated.

- **What DLRM does not do:** no sequence modelling, no attention over user history. It is a bag-of-features model. [[Deep Interest Network for CTR Prediction (DIN)]], [[Self-Attentive Sequential Recommendation (SASRec)]], and later [[Actions Speak Louder than Words- Generative Recommenders (HSTU)|HSTU]] all attack that gap — HSTU explicitly frames itself as replacing the DLRM feature-engineering paradigm with a generative sequence model.

- **Open question the paper raises but leaves open:** is second order actually enough? The claim is asserted, not measured. Given the model has ~540M parameters mostly locked in embeddings, the interaction layer is a tiny fraction of the compute, so "not worth the cost" is a weaker argument than it first sounds.

- DLRM became an **MLPerf training benchmark**, which is arguably its largest practical impact — it standardised what "recommender workload" means for hardware vendors. Which then means the [[Towards Quantifying Benchmark Optimization in ASR Models|benchmark-optimisation]] concern applies: hardware got tuned to this specific shape.

## Links

Related: [[Factorization Machines (ICDM)]] · [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)]] · [[Wide & Deep Learning for Recommender Systems]] · [[Deep Interest Network for CTR Prediction (DIN)]] · [[Actions Speak Louder than Words- Generative Recommenders (HSTU)]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Recommender Systems - Evolution]] · [[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models]] · [[Cross Entropy]] · [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] · [[Deep Learning]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Distributed Representations of Words and Phrases (negative sampling)]]

New topics worth writing: Deep & Cross Network (DCN), xDeepFM, Adagrad and row-wise Adagrad for sparse embeddings, all-to-all collectives and butterfly shuffle, MLPerf, Criteo CTR datasets, embedding table sharding and mixed-dimension embeddings, stack distance / reuse distance profiling, Neural Collaborative Filtering (NCF)
