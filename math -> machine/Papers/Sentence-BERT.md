---
title: "Sentence-BERT"
authors: ["Nils Reimers", "Iryna Gurevych"]
year: 2019
arxiv: "1908.10084"
url: https://arxiv.org/abs/1908.10084
priority: Must-Read
read_on: 2026-08-24
tags: [paper, transformers]
---
## The Core Idea

[[BERT- Pre-training of Deep Bidirectional Transformers|BERT]] is great at judging whether two sentences mean the same thing — but only if you feed it **both sentences at once**, glued together with a `[SEP]` token. That is a *cross-encoder*. It works, and it is useless at scale.

The reason is combinatorics. To find the most similar pair among 10,000 sentences you must run $n(n-1)/2 = 49{,}995{,}000$ forward passes. On a V100 that is roughly 65 hours. Clustering, search, dedup — anything with $O(n^2)$ comparisons — is off the table.

The obvious fix is to give each sentence its own fixed vector, then compare vectors with cosine similarity. People tried this by averaging BERT's token outputs, or by taking the `[CLS]` vector. **It does not work.** Averaged BERT embeddings score 54.81 average Spearman on seven STS benchmarks; the `[CLS]` vector scores 29.19. Averaged GloVe word vectors — a bag of static, context-free embeddings from 2014 — score 61.32. Raw BERT is *worse than GloVe* at this.

The fix here: take pretrained BERT, wrap two copies with tied weights (a *siamese network*), add a mean-pooling layer, and fine-tune the whole thing on natural language inference pairs so that the pooled vectors land in a space where cosine similarity actually means semantic similarity. That is SBERT. Average STS jumps to 74.89.

> [!NOTE] Cross-encoder vs bi-encoder
> A **cross-encoder** takes a *pair* as input and outputs one score — accurate, but $O(n^2)$ and no reusable vectors. A **bi-encoder** encodes each item independently into a vector; comparison is then just a dot product, so you can precompute, index, and search. SBERT is the bi-encoder version of BERT. ^bi-encoder

The unlock is entirely practical: 10,000 sentences → 10,000 embeddings (~5 seconds) → one cosine-similarity matrix (~0.01 seconds). 65 hours becomes 5 seconds. With an ANN index like FAISS, finding the nearest of 40 million Quora questions goes from 50 hours to milliseconds.

The deeper lesson: **pretrained representations are not automatically metric spaces.** BERT was trained with masked-token prediction and a classification head sitting on top; nothing in that objective ever asked the vector geometry to be meaningful under cosine distance. If you want a metric, you must train for a metric.

## The Methodology

**Architecture.** BERT (or RoBERTa) → pooling layer → fixed-size sentence vector $u$. Two branches, **tied weights** — literally the same network run twice. Three pooling options tested: `MEAN` (average all output token vectors), `MAX` (element-wise max over time), `CLS` (take the first token). Default is `MEAN`.

Three training objectives depending on what labels you have.

**1. Classification objective** (used for NLI data). Encode both sentences to $u, v$. Build a feature vector by concatenating $u$, $v$, and the element-wise absolute difference $|u-v|$, then a single linear layer into $k$ classes:

$$o = \text{softmax}\!\left(W_t\,(u, v, |u-v|)\right), \quad W_t \in \mathbb{R}^{3n \times k}$$

Trained with [[Cross Entropy|cross-entropy loss]]. Note $W_t$ is thrown away at inference — its only job is to shape the encoder.

**2. Regression objective** (used for STS-style graded labels). Compute $\cos(u,v)$ directly, train with mean-squared error against the gold score. No extra parameters at all.

**3. Triplet objective** (used for the Wikipedia sections task). Anchor $a$, positive $p$, negative $n$:

$$\max\left(\|s_a - s_p\| - \|s_a - s_n\| + \epsilon,\; 0\right)$$

with Euclidean distance and margin $\epsilon = 1$. Push the positive at least $\epsilon$ closer than the negative; once satisfied, zero loss.

**Training data and hyperparameters.** SNLI (570k pairs labelled *entailment* / *contradiction* / *neutral*) plus MultiNLI (430k pairs, mixed genres). One epoch. Batch size 16, [[Decoupled Weight Decay Regularization (AdamW)|Adam]] at learning rate $2\mathrm{e}{-5}$, linear warm-up over the first 10% of data. `MEAN` pooling.

That is it. Under 20 minutes of fine-tuning on top of an existing checkpoint — versus InferSent and Universal Sentence Encoder, which trained sentence encoders from random initialisation.

**Inference-time speed trick.** *Smart batching*: sort sentences by length and group similar-length ones into the same minibatch, so you pad to the longest element in the batch rather than the longest in the corpus. Gives +89% throughput on CPU, +48% on GPU.

## Ablation Studies and Experiments

**Unsupervised STS** (no STS training data at all; Spearman $\rho \times 100$ between cosine similarity and gold labels, averaged over STS12–16, STSb, SICK-R):

| Model | Avg. |
|---|---|
| BERT `CLS`-vector | 29.19 |
| Avg. BERT embeddings | 54.81 |
| Avg. GloVe | 61.32 |
| InferSent (GloVe BiLSTM) | 65.01 |
| Universal Sentence Encoder | 71.22 |
| SBERT-NLI-base | 74.89 |
| SBERT-NLI-large | 76.55 |
| SRoBERTa-NLI-large | 76.68 |

+11.7 over InferSent, +5.5 over USE. Swapping BERT for RoBERTa buys almost nothing (74.89 → 74.21 at base size). Manhattan and Euclidean distance gave roughly the same numbers as cosine.

**Supervised STS (STSb test).** Here the cross-encoder still wins. BERT-NLI-STSb-large hits **88.77**, SBERT-NLI-STSb-large only **86.10**. That is the price of the bi-encoder: you cannot do word-by-word cross-attention between the two sentences. Two-stage training (NLI first, then STSb) helped SBERT by 1–2 points and the cross-encoder by 3–4.

**Argument Facet Similarity — the honest failure.** 6,000 argument pairs on gun control, gay marriage, death penalty. With 10-fold cross-validation SBERT-base gets $\rho = 74.13$ vs BERT's 74.84 — basically tied. But in the **cross-topic** setup (train on two topics, test on the held-out third), SBERT drops to 50.65 while BERT holds 57.23. A ~7-point gap. The authors' explanation is the right one: BERT can directly attend word-to-word across the pair, whereas SBERT must map a sentence from an *unseen topic* into a space where matching claims-and-reasoning land nearby. Two topics of training data is not enough for that.

**Wikipedia sections triplets** (1.8M training triplets, 222,957 test; accuracy = is positive closer than negative?): SBERT-base 0.8042, SBERT-large 0.8078, against Dor et al.'s BiLSTM triplet network at 0.74.

**SentEval** (freeze the embeddings, fit logistic regression on top, 7 classification tasks): SBERT-NLI-large averages 87.69 vs InferSent 85.59 and USE 85.10. Best on 5 of 7. Big gains specifically on the sentiment tasks (MR, CR, SST).

**The most interesting single observation.** On SentEval, average BERT embeddings score **84.94** — beating GloVe's 81.52. On STS with cosine similarity they score **54.81** — losing badly to GloVe. Same vectors, opposite verdicts. Why? SentEval fits a logistic regression on top, which can reweight dimensions; cosine similarity weights all dimensions equally. So raw BERT vectors *do* contain the information, it is just distributed in a way cosine cannot read. This is the same phenomenon later work names anisotropy — see [[How Contextual are Contextualized Word Representations]] and [[Representation Degeneration Problem in Training NLMs]].

**Ablation table** (Spearman on STSb dev):

| | NLI (classification) | STSb (regression) |
|---|---|---|
| `MEAN` | 80.78 | 87.44 |
| `MAX` | 79.07 | **69.92** |
| `CLS` | 79.80 | 86.62 |

| Concatenation | NLI |
|---|---|
| $(u,v)$ | 66.04 |
| $(\|u-v\|)$ | 69.78 |
| $(u*v)$ | 70.54 |
| $(\|u-v\|, u*v)$ | 78.37 |
| $(u,v,u*v)$ | 77.44 |
| $(u,v,\|u-v\|)$ | **80.78** |
| $(u,v,\|u-v\|,u*v)$ | 80.44 |

What this reveals:

- **Pooling barely matters for classification training** (79.07–80.78, ~1.7 points) but matters enormously for regression training, where `MAX` collapses to 69.92. Interesting because InferSent found `MAX` *better* for its BiLSTM. Transformer outputs and LSTM outputs are not interchangeable.
- **The concatenation scheme is the big lever** — 66.04 to 80.78, a 14-point spread. The element-wise difference $|u-v|$ is the critical piece. Without it, $(u,v)$ alone gives 66.04. Intuition: $|u-v|$ is literally the per-dimension distance, so the softmax classifier is forced to make "similar" mean "small coordinate-wise gap", which is exactly what cosine will later measure.
- **What did not work:** adding $u*v$ on top of $(u,v,|u-v|)$ *hurt* slightly, 80.78 → 80.44. Both InferSent and USE use $(u,v,|u-v|,u*v)$; this architecture does not want it.
- **RoBERTa did not help.** Better on supervised tasks, a wash for sentence embeddings.
- **XLNet was tried and was generally worse than BERT.** One sentence in the related-work section, no table.

**Speed** (sentences/second, V100):

| | CPU | GPU |
|---|---|---|
| Avg. GloVe | 6469 | — |
| InferSent | 137 | 1876 |
| USE | 67 | 1318 |
| SBERT-base | 44 | 1378 |
| SBERT-base + smart batching | 83 | 2042 |

On CPU, InferSent's single BiLSTM beats SBERT's 12 transformer layers by ~65%. On GPU the transformer's parallelism flips it — SBERT is 9% faster than InferSent, 55% faster than USE.

## Worth Remembering

- **This is the paper behind `sentence-transformers`.** Almost every RAG pipeline, semantic search index, and embedding-based retriever in production descends from this recipe. Also the direct ancestor of the "encode independently, compare with dot product" pattern in [[Recommender Systems - Evolution|two-tower retrieval]].

- **The cross-encoder is still more accurate.** 88.77 vs 86.10 on supervised STSb, and a 7-point gap on cross-topic AFS. The standard production answer is *retrieve then rerank*: SBERT-style bi-encoder pulls the top ~100 candidates cheaply, a cross-encoder reranks them. Do not read this paper as "bi-encoders replaced cross-encoders."

- **NLI labels are a strange but effective supervision signal for similarity.** Entailment/contradiction/neutral is not a similarity score. It works because contradiction pairs are lexically similar but semantically opposed, forcing the encoder to encode meaning rather than word overlap. Later work (SimCSE) showed you can push this further by treating entailment pairs as positives and contradictions as *hard negatives* in an [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)|InfoNCE]] loss.

- **The classification head $W_t$ is disposable scaffolding.** You train with it and delete it. The gradient signal it produces is the entire point — the head itself is not the model. Same shape as the projection heads used in modern contrastive learning.

- **The $|u-v|$ ablation is the reusable insight.** If you want cosine similarity to be meaningful downstream, put an explicit distance-like feature into the training objective. Do not hope the geometry emerges from a generic classification loss on $(u,v)$ — 66.04 says it will not.

- **Limitation the authors admit up front:** SBERT's embeddings are not meant for transfer learning. If you have a labelled downstream task, full BERT fine-tuning updates all layers and beats frozen embeddings. SentEval numbers are a sanity check, not a use case.

- **Caveat on evaluation:** the authors deliberately report Spearman rank correlation, not Pearson, citing their own prior work that Pearson is badly suited to STS. They also train with 10 random seeds and report standard deviations — note BERT-STSb-large's $\pm 0.81$, which is large enough that some headline gaps in this literature are noise.

- **Open question:** the AFS cross-topic result suggests bi-encoders generalise poorly to unseen domains when similarity is defined by fine-grained reasoning rather than surface semantics. Modern embedding models attack this with massive multi-task training data, not architecture changes. Consistent with [[The Bitter Lesson (essay)|the bitter lesson]].

## Links

Related: [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Attention Is All You Need]] · [[How Contextual are Contextualized Word Representations]] · [[Representation Degeneration Problem in Training NLMs]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[Understanding Contrastive Learning through Alignment and Uniformity]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[Efficient Estimation of Word Representations (word2vec)]] · [[Cross Entropy]] · [[Long Short-Term Memory (Neural Computation)]] · [[Recommender Systems - Evolution]] · [[Real-time Personalization using Embeddings for Search Ranking at Airbnb (KDD, best paper)]] · [[Linear Projection]]

New topics worth writing: Siamese networks and weight tying, Triplet loss and margin selection, Cosine similarity as a metric on learned embeddings, InferSent, Universal Sentence Encoder, SentEval, Natural Language Inference (SNLI/MultiNLI), FAISS and approximate nearest neighbour search, Retrieve-then-rerank pipelines, Poly-encoders, SimCSE, Spearman vs Pearson correlation for similarity evaluation, Smart batching / length-bucketed padding
