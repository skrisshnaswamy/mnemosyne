---
title: "Dense Passage Retrieval (DPR)"
authors: ["Vladimir Karpukhin", "Barlas Oğuz", "Sewon Min", "Patrick Lewis", "Ledell Wu", "Sergey Edunov", "Danqi Chen", "Wen-tau Yih"]
year: 2020
arxiv: "2004.04906"
url: https://arxiv.org/abs/2004.04906
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, transformers, self-supervised]
---
## The Core Idea

Before this paper, if you wanted to find the 20 Wikipedia passages most likely to answer a question, you used BM25 — a keyword-matching score over an inverted index. Everyone "knew" that learned dense vectors could not beat it unless you did some elaborate extra pre-training first. DPR shows that belief was wrong, and that the reason it looked true was that people were training the encoders badly, not that dense retrieval was weak.

The system itself is boring on purpose: two separate [[BERT- Pre-training of Deep Bidirectional Transformers|BERT]] models, one that turns a question into a 768-dim vector, one that turns a passage into a 768-dim vector, and a plain dot product as the relevance score. Twenty-one million Wikipedia passages get encoded once, dumped into a FAISS index, and searched by maximum inner product. Nothing clever in the architecture.

The whole contribution is the training recipe, and it is one sentence: **score every question in the batch against every passage in the batch, and treat the other questions' gold passages as negatives.** That "in-batch negatives" trick turns a batch of $B$ pairs into $B^2$ scored pairs, and it is worth about 6 points of top-5 accuracy over the same model trained with the same number of negatives drawn from outside the batch. Add exactly one hard negative per question (a BM25 passage that matches the words but does not contain the answer) and you get another 9 points.

> [!NOTE] In-batch negatives
> With $B$ questions and their $B$ gold passages in one batch, build $\mathbf{S} = \mathbf{Q}\mathbf{P}^\top$, a $B\times B$ score matrix. The diagonal is the positives; everything else is a negative. Free negatives, free compute reuse, and more of them as you grow the batch. ^in-batch-negatives

The headline numbers: top-20 retrieval accuracy on Natural Questions goes from **59.1% (BM25) to 78.4% (DPR)**. Top-5 goes **42.9% → 65.2%**. End-to-end exact-match QA on NQ goes from ORQA's 33.3 to **41.5**, beating a system that needed an expensive inverse-cloze pre-training stage. And DPR beats BM25 after fine-tuning on only **1,000** question–passage pairs.

What it unlocks: retrieval becomes a differentiable, learnable component you can train on a laptop-sized dataset. This is the retriever that RAG, FiD, and most of the modern retrieval-augmented stack were built on top of.

## The Methodology

**Corpus.** English Wikipedia (Dec 2018 dump), cleaned with the DrQA preprocessor (tables, info-boxes, lists, disambiguation pages dropped), then chopped into disjoint 100-word blocks. That gives **21,015,324 passages**. Each passage text is prefixed with its article title plus a `[SEP]` token — a small detail that gives the encoder the entity name for free.

**Encoders.** Two *independent* BERT-base-uncased models. No weight sharing. Take the `[CLS]` output, so $d = 768$.

$$\mathrm{sim}(q,p) = E_Q(q)^\top E_P(p)$$

The score must be *decomposable* — question and passage cannot see each other — otherwise you could not pre-compute the 21M passage vectors. That is the whole reason a [[Sentence-BERT#The Core Idea|bi-encoder]] is used instead of a cross-attention scorer.

**Loss.** Negative log-likelihood of the positive passage against $n$ negatives — this is a softmax over similarity scores, i.e. plain [[Cross Entropy|cross-entropy]] where the "classes" are candidate passages:

$$L = -\log \frac{e^{\mathrm{sim}(q_i, p_i^+)}}{e^{\mathrm{sim}(q_i, p_i^+)} + \sum_{j=1}^{n} e^{\mathrm{sim}(q_i, p_{i,j}^-)}}$$

No temperature parameter, unlike [[A Simple Framework for Contrastive Learning (SimCLR)|SimCLR]]'s NT-Xent. Same shape as [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)|InfoNCE]].

**Where the negatives come from.** Three sources considered:
1. *Random* — any passage from the 21M.
2. *BM25* — top BM25 hits that match the question words but do **not** contain the answer string. These are the hard ones.
3. *Gold* — the positive passage of a *different* question in the training set.

The final recipe: gold passages from the same mini-batch (in-batch) **plus one shared BM25 hard negative per question**. With batch size $B=128$, that BM25 negative from each question is also broadcast as a negative for every other question in the batch, so each question sees $127 + 128 = 255$ negatives.

**Positives.** For TREC, WebQuestions and TriviaQA there is no gold passage, only an answer string, so they take the highest-ranked BM25 passage that contains the answer (distant supervision). Questions with no answer-containing passage in the top 100 are thrown away. For NQ and SQuAD they match the original gold paragraph into the 100-word passage pool. Table 1 shows the attrition: NQ goes 79,168 → 58,880 usable training questions.

**Hyperparameters that mattered.** Batch size 128, LR $10^{-5}$ with [[Adam- A Method for Stochastic Optimization|Adam]], linear schedule with warm-up, [[Dropout- A Simple Way to Prevent Overfitting|dropout]] 0.1, 40 epochs on big datasets and 100 on small ones. Eight 32GB GPUs.

**The reader.** A separate BERT-base takes the top $k$ passages (up to 100) and produces three things per passage: a start-token distribution, an end-token distribution, and a passage-selection logit from the `[CLS]` vector.

$$P_{\text{start},i}(s) = \mathrm{softmax}(\mathbf{P}_i \mathbf{w}_{\text{start}})_s,\quad P_{\text{selected}}(i) = \mathrm{softmax}(\hat{\mathbf{P}}^\top \mathbf{w}_{\text{selected}})_i$$

Span score is $P_{\text{start},i}(s) \times P_{\text{end},i}(t)$. The reader trains on $\tilde{m}=24$ passages per question (1 positive, 23 negatives sampled from the retriever's top 100), maximising the marginal log-likelihood of all correct spans. Here cross-attention *is* allowed, because you are only reranking 100 candidates, not 21 million.

**Serving cost.** FAISS in-memory index does **995 questions/sec** returning top-100; BM25/Lucene does 23.7 q/s per thread. But building the index is expensive: 8.8 GPU-hours (on 8 GPUs) to embed 21M passages, plus 8.5 hours to build the FAISS index, versus ~30 minutes to build the Lucene inverted index.

## Ablation Studies and Experiments

**Table 3 — the money table** (NQ dev, top-$k$ retrieval accuracy). IB = in-batch.

| Negatives | #N | IB | Top-5 | Top-20 | Top-100 |
|---|---|---|---|---|---|
| Random | 7 | ✗ | 47.0 | 64.3 | 77.8 |
| BM25 | 7 | ✗ | 50.0 | 63.3 | 74.8 |
| Gold | 7 | ✗ | 42.6 | 63.1 | 78.3 |
| Gold | 7 | ✓ | 51.1 | 69.1 | 80.8 |
| Gold | 31 | ✓ | 52.1 | 70.8 | 82.1 |
| Gold | 127 | ✓ | 55.8 | 73.0 | 83.1 |
| Gold + 1 BM25 | 31+32 | ✓ | 65.0 | 77.3 | 84.4 |
| Gold + 2 BM25 | 31+64 | ✓ | 64.5 | 76.4 | 84.0 |
| Gold + 1 BM25 | 127+128 | ✓ | **65.8** | **78.0** | **84.9** |

Read the rows carefully, because they say something non-obvious:

- **Without in-batch training, the choice of negative barely matters** for $k \geq 20$. Random, BM25, gold — all land in a 63–64 band at top-20. This is why earlier work concluded dense retrieval was mediocre; they were on the flat part of the curve.
- **Same 7 gold negatives, but drawn from inside the batch: 63.1 → 69.1 top-20.** The only difference is *where* the negatives come from and that the loss now scores every pair. More gradient signal per step from the same forward passes.
- **Accuracy rises monotonically with batch size** (7 → 31 → 127 negatives: 51.1 → 52.1 → 55.8 top-5). Bigger batches are strictly better here, exactly as in [[Momentum Contrast (MoCo)|MoCo]]-style contrastive work.
- **One hard BM25 negative is worth more than everything else combined.** At 31 in-batch negatives, adding one BM25 negative jumps top-5 from 52.1 to 65.0 — a 13-point gain. That single component is doing most of the work.
- **A second BM25 negative makes it slightly worse** (65.0 → 64.5). Two hard negatives is over-egging it.

**Sample efficiency.** DPR trained on 1,000 NQ examples already beats BM25. Going 1k → 59k keeps improving accuracy but with diminishing slope.

**Main retrieval results (Table 2, top-20 / top-100 on test).**

| | NQ | TriviaQA | WQ | TREC | SQuAD |
|---|---|---|---|---|---|
| BM25 | 59.1 / 73.7 | 66.9 / 76.7 | 55.0 / 71.1 | 70.9 / 84.1 | **68.8 / 80.0** |
| DPR (single) | **78.4 / 85.4** | **79.4 / 85.0** | 73.2 / 81.4 | 79.8 / 89.1 | 63.2 / 77.2 |
| DPR (multi) | 79.4 / 86.0 | 78.8 / 84.7 | **75.0 / 82.9** | **89.1 / 93.9** | 51.6 / 67.6 |

**End-to-end QA, exact match (Table 4).** NQ: BM25+reader 32.6, ORQA 33.3, REALM$_{\text{News}}$ 40.4, **DPR 41.5**. TriviaQA: **56.8** (57.9 with BM25+DPR). TREC in the multi setting: **49.4** vs ORQA's 30.1. Higher retrieval precision does translate into higher QA accuracy — the paper's second stated contribution, and worth confirming since it is not automatic.

### What did not work

- **SQuAD.** DPR *loses* to BM25 (63.2 vs 68.8 top-20 single, 51.6 multi). Two reasons the authors give: annotators wrote questions while looking at the passage, so lexical overlap is huge and BM25 is handed an advantage; and the data comes from only ~500 Wikipedia articles, so the training distribution is badly skewed. SQuAD is excluded from all multi-dataset training for this reason.
- **Joint retriever+reader training** (Appendix D). They froze the passage encoder, backpropped the combined loss into the question encoder, used 1,600 passages per question for the retriever loss. Result: **39.8 EM on NQ dev — the same as the plain pipeline**. Complexity for nothing.
- **Cosine similarity** is worse than dot product. L2 is roughly equal to dot product (43.5 vs 44.9 top-1). So the geometry choice does not matter much, and the paper picks dot product for simplicity.
- **Triplet loss** instead of NLL: 41.6 vs 44.9 top-1 with dot product. Slightly worse, not catastrophic. Pairwise ranking losses like [[BPR- Bayesian Personalized Ranking from Implicit Feedback|BPR]]'s do not buy you anything here.
- **Distant supervision for positives** costs almost nothing: gold contexts 78.1 top-20, best-BM25-passage-containing-answer 77.1. About 1 point. Good news if your dataset only has answer strings.
- **Two BM25 hard negatives** slightly hurt versus one.

**Cross-dataset generalisation.** Train on NQ only, test on WQ/TREC without fine-tuning: 69.9 / 86.3 top-20, versus 75.0 / 89.1 for the fine-tuned model and 55.0 / 70.9 for BM25. It loses 3–5 points but still crushes the sparse baseline.

## Worth Remembering

- **BM25 and DPR fail differently, and the hybrid is sometimes best.** Appendix C makes this concrete. For *"What is the body of water between England and Ireland?"* BM25 returns a British Cycling article (the words "England", "Ireland", "body" all appear) while DPR returns the Irish Sea article with zero lexical overlap on "body of water". Reverse case: *"Who plays Thoros of Myr in Game of Thrones?"* — BM25 nails it because "Thoros of Myr" is a rare selective phrase; DPR retrieves a random Norwegian actor. Dense vectors are bad at rare proper nouns. The BM25+DPR hybrid reranks the union of both top-2000 lists with $\text{BM25}(q,p) + 1.1 \cdot \mathrm{sim}(q,p)$ and wins on TREC (85.2 vs 79.8 top-20) and SQuAD.
- **The "you need special pre-training" story was a red herring.** ORQA's inverse cloze task and REALM's asynchronous re-indexing are both expensive, and DPR beats both on NQ and TriviaQA with plain supervised fine-tuning of a stock BERT. The authors' guess: extra pre-training only helps when your target training set is tiny. Compare to how [[RoBERTa- A Robustly Optimized BERT Pretraining Approach|RoBERTa]] found BERT was just undertrained — same genre of result, where careful training beats architectural novelty.
- **Index-build cost is the real operational tax.** 8.5 hours to build a FAISS index over 21M vectors on one server versus 30 minutes for Lucene. Every time you retrain the passage encoder you re-embed and rebuild everything. This is precisely why REALM's asynchronous re-indexing exists and why DPR freezes the passage encoder in its joint-training ablation.
- **The reader consumes up to 100 passages in a single batch on one 32GB GPU**, latency ~20ms — basically the same as one passage. But $k=10$ only costs 0.7 EM on NQ (40.8 vs 41.5), so you can go much cheaper.
- **Follow-ups the paper already anticipates.** ANCE (Xiong et al. 2020) replaces the static BM25 hard negatives with negatives mined by the *previous iteration's* retriever, which improves on DPR — the obvious next move once you see that hard negatives are doing all the work. FiD (Izacard & Grave) and RAG (Lewis et al.) bolt generative readers onto DPR. ColBERT keeps token-level vectors and a late-interaction operator instead of one `[CLS]` vector per passage.
- **Practical caveat for reuse:** the two encoders are untied, so you must serve both. And the passage encoder saw title-prefixed 100-word chunks — feed it anything else at inference and you are off-distribution.
- Open question worth chasing: DPR compresses a 100-word passage into one 768-dim vector. That is a hard information bottleneck, and it is exactly what the "rare salient phrase" failures are. How much of the gap is capacity and how much is the training objective?

## Links

Related: [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Sentence-BERT]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[A Simple Framework for Contrastive Learning (SimCLR)]] · [[Momentum Contrast (MoCo)]] · [[Embedding-based Retrieval in Facebook Search]] · [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)]] · [[Cross Entropy]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[BPR- Bayesian Personalized Ranking from Implicit Feedback]] · [[Recommender Systems - Evolution]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Adam- A Method for Stochastic Optimization]] · [[RoBERTa- A Robustly Optimized BERT Pretraining Approach]] · [[The Embedder's Dilemma- LLMs Are Better, but at What Cost]] · [[Matryoshka Representation Learning]]

New topics worth writing: BM25 and TF-IDF, Inverted index, FAISS and approximate nearest neighbour search, Maximum inner product search (MIPS), Open-domain question answering, Retrieval-Augmented Generation (RAG), Fusion-in-Decoder (FiD), ColBERT and late interaction, ORQA and the inverse cloze task, REALM, ANCE and iterative hard negative mining, Natural Questions benchmark, Distant supervision, Exact match metric, Metric learning
