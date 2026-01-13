---
title: "ColBERT: Efficient and Effective Passage Search via Late Interaction"
authors: ["Khattab & Zaharia"]
year: 2020
arxiv: "2004.12832"
url: https://arxiv.org/abs/2004.12832
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, transformers]
---
## The Core Idea

Before this paper you had two bad options for neural search.

Option one: the **cross-encoder**. Glue the query and the document into one string, push it through [[BERT- Pre-training of Deep Bidirectional Transformers|BERT]], read a relevance score off the `[CLS]` token. Very accurate. Also insane: to rank 1000 candidate passages you run BERT 1000 times, at query time, per query. That is 10.7 seconds on a V100 and 97 *tera*-FLOPs.

Option two: the **bi-encoder** (what [[Sentence-BERT]] and [[Dense Passage Retrieval (DPR)|DPR]] do). Squash the query into one vector, squash the document into one vector, take a dot product. Documents get encoded offline, so query time is milliseconds. But squashing a whole passage into 768 numbers throws away most of the word-level matching signal, and quality drops.

ColBERT's trick: **do not squash, and do not join early**. Encode the query and the document *separately*, but keep **one vector per token** instead of one vector per text. Then the interaction between them is a tiny, cheap, parameter-free operation done at the very end.

> [!NOTE] Late interaction
> Query and document are encoded independently (so documents can be pre-computed and stored), but each is kept as a *bag of token vectors*. Relevance is a cheap function over those two bags, computed at query time. You get fine-grained term matching *and* offline document encoding. ^late-interaction

The cheap function is **MaxSim**: for each query token, find its best-matching document token, and add up those best scores.

$$S_{q,d} = \sum_{i \in [|E_q|]} \max_{j \in [|E_d|]} E_{q_i} \cdot E_{d_j}^{T}$$

That is it. No parameters in the scoring step at all.

Two things this unlocks. First, re-ranking becomes 170× faster and 13,900× cheaper in FLOPs than BERT-base re-ranking, at 34.9 vs 36.0 MRR@10 — nearly free quality. Second, and more interesting: MaxSim is **pruning-friendly**. A max over dot products is exactly what an approximate nearest-neighbour index (faiss) computes. So you can throw every document token vector in the whole 8.8M-passage collection into one index and do *end-to-end retrieval*, not just re-ranking. You are no longer trapped inside whatever BM25 handed you.

## The Methodology

**One shared BERT, two prefixes.** The same BERT weights encode both queries and documents. They are told apart by a special token prepended right after `[CLS]`: `[Q]` for queries, `[D]` for documents. The `[Q]`/`[D]` embeddings are new parameters trained from scratch.

**Query encoder.** Tokenize with WordPiece. Pad or truncate to a *fixed* $N_q = 32$ tokens. The padding is not ignored — it is done with BERT's `[mask]` token, and the output vectors at those mask positions are kept and used in the score.

> [!NOTE] Query augmentation
> Padding the query with `[mask]` tokens so BERT produces extra query-side vectors. These act as a soft, learned query expansion: the model can put whatever terms it thinks are implied by the query into those slots, or use them to re-weight existing terms. Removing this hurts a lot. ^query-augmentation

**Document encoder.** Same BERT, prefix `[D]`, no mask padding. After encoding, vectors for punctuation tokens are **thrown away** using a fixed list — fewer vectors to store, and punctuation embeddings are assumed useless.

**The dimension squeeze.** Both encoders end with a linear layer (no activation) mapping BERT's 768 dims down to $m = 128$. Then L2-normalise every vector, so dot product = cosine similarity, in $[-1, 1]$. The 128 is purely a storage and PCIe-bandwidth decision — it barely affects query encoding speed but it is everything for how big your index is.

$$E_q := \texttt{Normalize}(\texttt{Linear}(\texttt{BERT}(\text{``}[Q]\,q_0 \dots q_l \,\#\#\dots\#\text{''})))$$
$$E_d := \texttt{Filter}(\texttt{Normalize}(\texttt{Linear}(\texttt{BERT}(\text{``}[D]\,d_0 \dots d_n\text{''}))))$$

**Loss.** Triples $\langle q, d^+, d^- \rangle$. Score each document on its own, then pairwise softmax [[Cross Entropy|cross-entropy]] over the two scores — i.e. push $S_{q,d^+}$ above $S_{q,d^-}$. Same spirit as [[BPR- Bayesian Personalized Ranking from Implicit Feedback|BPR]]. Trained with [[Adam- A Method for Stochastic Optimization|Adam]], learning rate $3\times10^{-6}$, batch size 32, 200k steps on MS MARCO.

**Offline indexing.** Run every document through $f_D$ once, save the vectors. Three throughput tricks that mattered: (1) multi-GPU batching, (2) sort documents by length in groups of $B = 100{,}000$ and batch $b = 128$ similar-length docs together, so padding waste is small (a "BucketIterator"), (3) parallelise WordPiece tokenization across CPU cores — it was a non-trivial share of wall time. Result: 8.8M passages indexed in ~3 hours on 4 GPUs.

**Re-ranking path (§3.5).** Load the $k = 1000$ document matrices, pad, stack into a 3-D tensor, ship to GPU, one batched dot product against $E_q$, max-pool over document tokens, sum over query tokens, sort. Note where the time actually goes: of 61 ms total, only **13 ms** is query encoding + interaction. The rest is gathering and transferring embeddings from CPU RAM to GPU.

**End-to-end path (§3.6).** All document token vectors from the whole collection go into a faiss **IVFPQ** index — space split into $P = 2000$ k-means cells, each vector split into $s = 16$ sub-vectors quantised to one byte each, similarity computed in the compressed domain. At query time: fire all 32 query vectors at the index in parallel, each retrieves $k' = 1000$ nearest document vectors, map each hit back to its source document, union them into $K$ unique candidate docs, then exactly re-rank those $K$ with full MaxSim using 16-bit stored vectors. Note they switch the similarity to squared L2 here, purely because faiss was faster at it.

## Ablation Studies and Experiments

**Re-ranking on MS MARCO** (re-rank official BM25 top-1000, MRR@10 on Dev):

| Model | MRR@10 | Latency | FLOPs/query |
|---|---|---|---|
| BM25 (official) | 16.7 | – | – |
| KNRM | 19.8 | 3 ms | 592M |
| Duet | 24.3 | 22 ms | 159B |
| fastText+ConvKNRM | 29.0 | 28 ms | 78B |
| BERT-base (Nogueira & Cho) | 34.7 | 10,700 ms | 97T |
| BERT-base (authors' training) | 36.0 | 10,700 ms | 97T |
| BERT-large | 36.5 | 32,900 ms | 340T |
| **ColBERT (over BERT-base)** | **34.9** | **61 ms** | **7B** |

ColBERT matches the original BERT-base re-ranker and loses ~1.6 points to BERT-large, at 1/540th the latency of BERT-large.

**Scaling with $k$.** ColBERT's advantage grows with re-ranking depth, because the query is encoded once no matter how many documents you score. FLOPs gap vs BERT-base: 180× at $k=10$, 13,900× at $k=1000$, 23,000× at $k=2000$.

**End-to-end retrieval from all 8.8M passages:**

| Model | MRR@10 (Dev) | Latency | R@50 | R@200 | R@1000 |
|---|---|---|---|---|---|
| BM25 (Anserini) | 18.7 | 62 ms | 59.2 | 73.8 | 85.7 |
| doc2query | 21.5 | 85 ms | 64.4 | 77.9 | 89.1 |
| DeepCT | 24.3 | ~62 ms | 69 | 82 | 91 |
| docTTTTTquery (T5) | 27.7 | 87 ms | 75.6 | 86.9 | 94.7 |
| ColBERT-L2 (re-rank BM25) | 34.8 | – | 75.3 | 80.5 | 81.4 |
| **ColBERT-L2 (end-to-end)** | **36.0** | 458 ms | **82.9** | **92.3** | **96.8** |

The striking row-pair: the *same model* scores 34.8 when re-ranking BM25 and 36.0 when retrieving end-to-end. BM25's recall ceiling (81.4 @1000) was silently capping the re-ranker. ColBERT's Recall@50 alone (82.9) beats BM25's Recall@1000.

**TREC CAR** (MAP): BM25 15.3, doc2query 18.1, DeepCT 24.6, BM25+BERT-base 31.0, **BM25+ColBERT 31.3**, BM25+BERT-large 33.5. Same story.

**The ablations (Figure 5, MS MARCO Dev).** To keep training affordable they used a 5-layer BERT for all ablation arms; the 5-layer full ColBERT [D] is the fair reference, and full 12-layer ColBERT [E] is 34.9.

- **[A] No late interaction.** Single vector per query and per document, from `[CLS]`, projected up to 4096 dims (= $32 \times 128$, matching ColBERT's total budget), scored by inner product. **Considerably worse.** So it is not "more numbers"— it is that the numbers are *addressable per token*. Note also that inner product beat cosine for this single-vector variant.
- **[B] Average instead of max.** Replace MaxSim with mean similarity over document tokens. Clearly worse. Query terms need to latch onto *one specific* document term; averaging drowns the signal in the rest of the passage. This is also the ablation that justifies the design on efficiency grounds, since average-sim is not prunable by an ANN index.
- **[C] No query augmentation.** Drop the `[mask]` padding. Noticeably lower MRR@10. The mask tokens are doing real work, not just shape-fixing.
- **End-to-end vs re-rank.** Retrieving directly lifts MRR@10 too, not just recall — it surfaces top-10 documents that BM25 never put in its top-1000 at all.

**Space vs quality (Table 4):**

| Setting | dim $m$ | bytes/dim | index size | MRR@10 |
|---|---|---|---|---|
| Re-rank cosine | 128 | 4 | 286 GiB | 34.9 |
| End-to-end L2 | 128 | 2 | 154 GiB | 36.0 |
| Re-rank cosine | 48 | 4 | 54 GiB | 34.4 |
| Re-rank cosine | 24 | 2 | 27 GiB | 33.9 |

Shrinking the index by 10× (286 → 27 GiB) costs 1 point of MRR@10. The representation is very robust to dimension reduction — a hint of the same redundancy that [[Matryoshka Representation Learning]] later exploits deliberately.

## Worth Remembering

- **The storage bill is the real cost.** ColBERT stores one vector per *token*, not per document. 286 GiB for 8.8M short passages at 128-dim float32. A bi-encoder would need ~4 GiB. You trade disk and RAM for FLOPs. Later work (ColBERTv2, PLAID) attacks exactly this with residual compression and centroid pruning.
- **The bottleneck at query time is not compute, it is memory movement.** 13 ms of the 61 ms is actual neural work; the rest is gathering and PCIe-transferring document embeddings. Anyone deploying this should think about layout and locality first, model size second.
- **End-to-end retrieval is a two-stage approximation.** Stage one (faiss) is approximate and can miss; stage two exactly re-scores only the $K$ survivors. So recall is bounded by whatever the ANN index surfaced. The paper does not measure how much recall the approximation itself costs.
- **Latency honesty.** ColBERT end-to-end is 458 ms vs BM25's 62 ms — 7× slower than lexical search, even if 20× faster than a BERT re-ranker. It is not free.
- **Asymmetric metric choice is a pragmatic hack.** Cosine for re-ranking, squared L2 for end-to-end, chosen because faiss happened to be faster on L2. Slightly uncomfortable that the similarity function is picked by index engineering, and the two give different MRR@10.
- **Baseline latency comparisons are generous to ColBERT.** For the non-BERT baselines they excluded CPU-side text preprocessing; for ColBERT they included the full pipeline. They flag this openly.
- The `[mask]`-padding trick is unusual and worth stealing: you are using a token the model was pre-trained to fill in as a *learned free slot*. It works because of [[BERT- Pre-training of Deep Bidirectional Transformers#Objective 1: Masked LM|masked language modelling]] pre-training — BERT already knows how to imagine what belongs at a mask.
- Open question the paper leaves: no result on whether MaxSim's per-token structure is interpretable (which query token matched which document token) or whether it survives out-of-domain. The BEIR benchmark later showed ColBERT generalises unusually well — better than single-vector [[Dense Passage Retrieval (DPR)|DPR]] — which is arguably its most important property and was not known at the time.
- Contrast with [[Distilling the Knowledge in a Neural Network|distillation]] and pruning approaches to BERT cost: those give maybe 2–10× speedups with quality loss. Re-architecting for the task gave 170×. Generic compression is weaker than task-shaped design.

## Links
Related: [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Dense Passage Retrieval (DPR)]] · [[Sentence-BERT]] · [[Attention Is All You Need]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Embedding-based Retrieval in Facebook Search]] · [[BPR- Bayesian Personalized Ranking from Implicit Feedback]] · [[Cross Entropy]] · [[Adam- A Method for Stochastic Optimization]] · [[Matryoshka Representation Learning]] · [[Exploring the Limits of Transfer Learning (T5)]] · [[NDCG]] · [[Distilling the Knowledge in a Neural Network]] · [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)]]

New topics worth writing: BM25 and lexical retrieval, MRR@k, faiss and IVFPQ, product quantization, approximate nearest neighbour search, doc2query / docTTTTTquery document expansion, DeepCT, cross-encoder vs bi-encoder, ColBERTv2 and PLAID, BEIR out-of-domain retrieval benchmark, MS MARCO dataset, WordPiece tokenization
```
