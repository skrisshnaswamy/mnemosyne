---
title: "Approximate Nearest Neighbor Negative Contrastive Learning (ANCE)"
authors: ["Xiong et al."]
year: 2020
arxiv: "2007.00808"
url: https://arxiv.org/abs/2007.00808
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, self-supervised, theory]
---
## The Core Idea

Dense retrieval means you encode a query and a document into vectors, and retrieve by nearest neighbour search instead of word overlap. The obvious problem: to train it, you need negative examples — documents that are *not* relevant. But the model has to beat *every* document in the corpus, and the corpus is millions of documents. So you sample.

Everyone before ANCE sampled negatives **locally**: other documents inside the same mini-batch (in-batch negatives, as in [[Dense Passage Retrieval (DPR)]]), or documents that BM25 ranked highly. ANCE's claim is that local negatives are nearly worthless, and it gives a gradient-based argument for *why*.

The argument in one line: a negative that the model already scores near zero produces a loss near zero, which produces a **gradient near zero**, which moves the weights not at all. Batch-mates are random documents. Random documents are trivially irrelevant. So most of your training signal is zero.

Two conditions make this bite in retrieval specifically. Let $b$ be batch size, $C$ the corpus, and $D^{-*}$ the set of genuinely hard negatives. Then $b \ll |C|$ and $|D^{-*}| \ll |C|$. The chance a random batch contains a hard negative is roughly

$$p = \frac{b\,|D^{-*}|}{|C|^2} \approx 0.$$

Measured empirically: the overlap between in-batch negatives and the top-100 hardest negatives is **exactly zero**. BM25 negatives get 15%. ANCE negatives start at 63% and reach 100% by construction.

The fix is blunt. Use the model *being trained* to search the *whole corpus* for its own hardest negatives, via an approximate nearest neighbour (ANN) index. This is expensive — re-encoding the corpus takes 10 hours — so you do it asynchronously and stale, refreshing every 10k batches.

What it unlocks: a plain two-tower BERT with dot product ([[Sentence-BERT|bi-encoder]] style) reaches nearly the accuracy of a full BM25 → BERT cross-encoder cascade, at **100× lower latency** (11.6 ms vs 1.42 s). That kills the then-standard belief that you need term-level interaction modelling to do search well.

> [!NOTE] ANCE negative
> A negative training document retrieved from the *entire corpus* by the current version of the model itself, using an ANN index that is refreshed periodically. Approximates the theoretically optimal importance-sampling distribution $p^*_{d^-} \propto \|\nabla_{\theta_t} l(d^+, d^-)\|_2$. ^ance-negative

## The Methodology

**The model.** BERT-Siamese / dual encoder. One shared encoder $g(\cdot;\theta)$ for both query and document, initialised from RoBERTa base ([[RoBERTa- A Robustly Optimized BERT Pretraining Approach]]). Take the `[CLS]` token from the last layer, push through a $768 \times 768$ projection ([[Linear Projection]]), then a [[Layer Normalization|layer norm]]. Score is dot product:

$$f(q,d) = \text{sim}\big(g(q;\theta), g(d;\theta)\big).$$

Loss is negative log likelihood over one positive and the sampled negatives.

**The theory that motivates the sampling.** With importance sampling, one SGD step is

$$\theta_{t+1} = \theta_t - \eta \frac{1}{N p_{d^-}} \nabla_{\theta_t} l(d^+, d^-),$$

where the $1/(Np_{d^-})$ keeps the estimator unbiased. Expanding the expected movement toward the optimum $\theta^*$ gives

$$\mathbb{E}\Delta^t = 2\eta\,\mathbb{E}(g_{d^-})^T(\theta_t - \theta^*) - \eta^2\,\mathbb{E}(g_{d^-})^T\mathbb{E}(g_{d^-}) - \eta^2\,\text{Tr}\big(\mathcal{V}(g_{d^-})\big).$$

Only the last term depends on the sampling distribution. Minimise the gradient variance and you converge faster. The minimiser is well known: sample proportional to per-instance [[Derivative#Gradient|gradient]] norm. And for MLPs the norm is bounded by the last-layer gradient,

$$\|\nabla_{\theta_t} l\|_2 \le L\rho \,\|\nabla_{\phi_L} l\|_2,$$

so for BCE or hinge loss, $l \to 0 \Rightarrow \|\nabla_{\theta_t} l\|_2 \to 0$. Zero loss, zero learning. ANCE picks the highest-loss negatives available, so it approximates the oracle distribution.

**The training loop.** Two processes running in parallel on a 1:1 GPU split:

- **Trainer.** Ordinary [[Backpropagation|backprop]] on triples $(q, d^+, d^-)$, where $d^-$ is uniformly sampled from the top-200 ANN results for $q$, minus the known positives. Uses the index built from the *previous* checkpoint $f_{k-1}$.
- **Inferencer.** Takes the latest checkpoint $f_k$, re-encodes the entire corpus, rebuilds the Faiss `IndexFlatIP` index, and hands it to the Trainer. Then repeats.

So the negatives are always slightly stale — this is the "asynchronous gap". Index rebuild itself is cheap (10 s); the corpus re-encode is the cost (10 h, 4.5 ms/doc).

**Documents longer than 512 tokens.** Two settings from Dai & Callan. *FirstP*: just use the first 512 tokens. *MaxP*: split into up to four 512-token passages, embed each, and take the max score — which ANN search supports natively, since you just index all the passages.

**Hyperparameters that mattered.** LAMB optimizer. LR 5e-6 for documents, 1e-6 for passages, linear warmup then decay after 5000 steps. Batch size 8 with gradient accumulation 2 across 4 GPUs. Index refresh every 10k batches. Converges in ~10 epochs, 1–2 h each. ANCE is *warm-started* — from BM25 negatives on TREC DL, from released DPR checkpoints on OpenQA. Cold-starting from scratch is not what they did.

## Ablation Studies and Experiments

**TREC 2019 Deep Learning Track.** The negative-construction ablation is the whole paper. Same architecture, same data, only the negatives change (document retrieval NDCG@10 / passage retrieval NDCG@10 / MARCO Dev MRR@10):

| Negatives | Doc retrieval | Passage retrieval | MARCO MRR@10 |
|---|---|---|---|
| BM25 (sparse baseline) | 0.519 | 0.506 | 0.240 |
| Rand Neg (in-batch random) | 0.543 | 0.552 | 0.261 |
| NCE Neg (hardest in-batch) | 0.542 | 0.539 | 0.256 |
| BM25 Neg | 0.529 | 0.591 | 0.299 |
| DPR (BM25 + Rand) | 0.557 | 0.600 | 0.311 |
| **ANCE (FirstP)** | **0.615** | **0.648** | **0.330** |
| ANCE (MaxP) | **0.628** | – | – |

Read the failures carefully. In *document* retrieval, every non-ANCE dense method **loses to plain BM25** (0.519). BM25 Neg actually makes document retrieval *worse* than random negatives (0.529 vs 0.543) — training on BM25's mistakes teaches the model to imitate BM25. NCE Neg — taking the hardest negative in the batch, the [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)|InfoNCE]] / [[A Simple Framework for Contrastive Learning (SimCLR)|SimCLR]] recipe — is statistically indistinguishable from random negatives. Hardest-in-a-batch-of-8 is still trivially easy.

**BM25 warm-up ablation.** Warming up with BM25 negatives then switching to Rand/NCE helps a little on documents (0.543 → 0.566) but never approaches ANCE. Warm-up is not the magic; the global negatives are.

**Rerank-vs-retrieve gap.** ANCE has the *smallest* gap between its reranking score (0.641 doc) and its retrieval score (0.615 doc) of any dense model. Others drop hard (Rand Neg: 0.615 → 0.543). That gap is exactly the price of training on negatives that do not look like what you meet at test time.

**OpenQA (answer coverage @20/100).** Single-task NQ: DPR 78.4/85.4 → ANCE **81.9/87.5**. TriviaQA: 79.4/85.0 → 80.3/85.3. Swapping DPR's retriever for ANCE and keeping the *same* reader lifts end answer accuracy: NQ 44.1 (RAG-Token) → **46.0**; TQA 56.8 (DPR Reader) → **57.5**.

**Production search engine.** +18.4% at 250M docs / 768-dim / exact KNN; +14.2% at 8B docs / 64-dim; +15.5% with approximate search at 8B. The gain survives aggressive dimension reduction and approximate search.

**Gradient norm measurement (Figure 4).** The direct test of the theory. Local negatives drive training loss to near zero and gradient norms to near zero at every BERT layer group. ANCE keeps loss high and gradient norms **orders of magnitude larger**. This is the ablation that says the mechanism is the one claimed, not something incidental about hard examples.

**Asynchronous gap ablation (A.3).** High learning rate (1e-5) or slow refresh (every 20k batches) makes training loss and test NDCG *oscillate* — the stale index drags the representation into bad local optima. Refreshing every 5k batches is smooth but needs 2× the GPUs on the Inferencer. 1:1 Trainer:Inferencer with a small LR is the sweet spot.

**What did not work.**
- Cosine similarity instead of dot product, and BCE / hinge loss instead of NLL: gradient norms on local negatives got *even smaller* (consistent with the theory), but retrieval accuracy was "not much better". The negative sampling, not the loss shape, is the binding constraint.
- Sampling from top-500 or top-1000 rather than top-200 hurt (MARCO MRR 0.33 → 0.31). Too deep and the negatives get easy again.
- LR 2e-6 on passages dropped MRR from 0.33 to 0.29. Divergence early in training was the common failure mode; they explored barely any hyperparameters because each run is expensive.

## Worth Remembering

- **Recall numbers on TREC DL are not trustworthy for dense models.** The TREC label pool was built from top-10 results of *sparse* systems only. ANCE overlaps with BM25's top-100 by at most 25%, giving a 13–15% "hole rate" (fraction of top-10 results with no label at all). Unlabelled counts as irrelevant. BM25 Neg has a 28% hole rate. MARCO's own labels come from Bing, so MRR there is more honest.

- **Only 25% overlap with BM25** is itself the headline finding for anyone building retrieval. Dense retrieval is not a better BM25; it finds different documents. Hybrid systems make sense for that reason.

- **The failure modes are semantic, not catastrophic** (A.5). ANCE loses 13 of 43 TREC queries. It retrieves "how long to hold a yoga pose" for "how long to hold bow in yoga" — related, not relevant. It fails on "active margin" because the pretrained encoder does not know it is a geology term, not a finance one. Contrast with the pre-ANCE fear that dense retrieval would return total nonsense; it does not.

- **The cost is real and it is the Inferencer.** 10 hours to re-encode the corpus per refresh, and you burn half your GPUs on it during training. Negative construction is 72 ms/batch vs 19 ms for backprop — 4× more time spent finding negatives than learning from them. Anyone reproducing this should budget for that.

- **This is [[Momentum Contrast (MoCo)|MoCo]]'s idea taken to its limit.** MoCo decouples the negative pool from the batch with a momentum queue; the memory-bank line of work freezes negative representations so more fit. ANCE says: stop growing the pool, just use the whole corpus and eat the staleness. The asynchronous refresh is borrowed from REALM.

- **The claimed death of interaction models is overstated in hindsight.** ANCE "nearly matches" the BERT reranker (doc: 0.641 rerank vs 0.646 for BERT Reranker; passage: 0.677 vs 0.742 — that passage gap is not small). [[ColBERT- Efficient and Effective Passage Search via Late Interaction|ColBERT]]-style late interaction went on to occupy the middle ground.

- **Open question:** the model mines negatives for itself, so it can only find negatives it currently thinks are plausible. Does this produce a self-confirming blind spot for document types the initial checkpoint scores low? The BM25 warm-up may be doing quiet work here that the paper does not isolate.

## Links
Related: [[Dense Passage Retrieval (DPR)]] · [[ColBERT- Efficient and Effective Passage Search via Late Interaction]] · [[Sentence-BERT]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[Momentum Contrast (MoCo)]] · [[A Simple Framework for Contrastive Learning (SimCLR)]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[Embedding-based Retrieval in Facebook Search]] · [[RoBERTa- A Robustly Optimized BERT Pretraining Approach]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)]] · [[Layer Normalization]] · [[Backpropagation]] · [[Derivative#Gradient|gradient]] · [[Recommender Systems - Evolution]]

New topics worth writing: Importance sampling for variance reduction in SGD, BM25 and inverted indexes, Faiss and ANN index structures (HNSW, IVF-PQ), TREC pooling and hole rate, LAMB optimizer, REALM, RAG (retrieval-augmented generation), MS MARCO benchmark, hard negative mining in metric learning, hybrid sparse-dense retrieval
