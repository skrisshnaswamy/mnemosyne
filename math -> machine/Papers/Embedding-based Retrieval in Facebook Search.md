---
title: "Embedding-based Retrieval in Facebook Search"
authors: ["Jui-Ting Huang", "Ashish Sharma", "Shuying Sun", "Li Xia", "David Zhang", "Philip Pronin", "Janani Padmanabhan", "Giuseppe Ottaviano", "Linjun Yang"]
year: 2020
arxiv: "2006.11632"
url: https://arxiv.org/abs/2006.11632
priority: Must-Read
read_on: 2026-08-25
tags: [paper]
---
## The Core Idea

Facebook Search used to be a Boolean keyword matcher. A document was a bag of terms (`text:john`, `location:seattle`), a query was a Boolean expression over terms, and retrieval was set intersection. That fails when the words do not literally match: "kacis creations" will never find the page *Kasie's creations*.

The fix is **embedding-based retrieval (EBR)**: turn both the query and the document into dense vectors, and make retrieval a nearest-neighbour search. That idea was already old by 2020. The two things that are actually new here:

1. **Unified embedding.** In web search the query is text. On Facebook the query is text *plus who is asking plus where they are*. Thousands of profiles are called "John Smith"; the right one is the searcher's friend. So the "query" side of the two-tower model eats the query string, the searcher's location, their social-graph embedding, and session context. It is not a text-embedding problem at all.
2. **Embeddings inside the inverted index, not beside it.** The obvious build is: run an ANN service, run the old Boolean engine, merge the two candidate lists. They tried that shape and rejected it — expensive, two indices to maintain, and the two lists overlap heavily so you pay twice for the same documents. Instead they quantised each document embedding into *a term* (its coarse cluster ID) *plus a payload* (the quantised residual) and added an `(nn <key> :radius <r>)` operator to the existing Boolean query language. Now nearest-neighbour search is just another Boolean primitive that can be `and`-ed and `or`-ed with term matches.

That second point is the durable lesson. Because `(nn)` composes with `(term ...)`, you get **constrained** ANN for free: search the embedding space *only within* Seattle-or-Menlo-Park. That both cuts cost and raises precision, and you inherit realtime index updates, query planning and multi-hop queries from the engine you already had.

> [!NOTE] Embedding-based retrieval (EBR)
> Replacing the recall/candidate-generation stage of a search engine with approximate nearest-neighbour search over learned dense vectors, so that documents can be retrieved without sharing any literal term with the query. ^ebr

> [!NOTE] Unified embedding
> A two-tower model where the "query" tower encodes the search *request* — text, searcher identity, location, social graph, context — rather than the query string alone. ^unified-embedding

## The Methodology

**Objective.** Retrieval is framed as recall optimisation. For query with target set $T$ of size $N$ and top-$K$ returned results,

$$\text{recall@}K = \frac{\sum_{i=1}^{K} \mathbb{1}[d_i \in T]}{N}.$$

Recall is not differentiable, so they approximate it with **triplet loss** over $(q, d_+, d_-)$:

$$L = \sum_{i=1}^{N} \max\!\big(0,\; D(q^{(i)}, d^{(i)}_+) - D(q^{(i)}, d^{(i)}_-) + m\big)$$

with $D(u,v) = 1 - \cos(u,v)$ and $m$ a margin. **Margin matters a lot** — tuning it moved KNN recall by 5–10%, and the best value differed wildly across verticals.

The neat argument for why random negatives approximate recall: if you draw $n$ negatives per positive, you are training the model to be right at rank 1 out of a pool of size $n$. If the real index has $N$ documents, you are roughly optimising $\text{recall@}K$ with $K \approx N/n$. Fewer negatives → bigger effective $K$ → more recall-flavoured model. This is a genuinely useful mental model for tuning negative counts, and it connects to [[Distributed Representations of Words and Phrases (negative sampling)|negative sampling]] and [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)|in-batch negatives]].

**Architecture.** Two encoders, $E_Q = f(Q)$ and $E_D = g(D)$, separate networks by default (optionally sharing some parameters), scored by cosine. A [[Sentence-BERT#|bi-encoder]] in the classic sense. Most input features are high-cardinality categoricals (one-hot or multi-hot); each gets an embedding lookup table, and multi-hot features are collapsed by a weighted sum of their embeddings before entering the tower.

**Features.**
- *Text*: character n-grams beat word n-grams (small vocabulary → better-trained table; robust to typos and out-of-vocabulary content). Adding word n-grams **on top of** character trigrams still gives +1.5% recall, even though query word-trigram cardinality is 352M and needs hashing (with collisions).
- *Location*: searcher city/region/country/language on the query side; admin-tagged group location on the document side. Table 2 is the vivid demo — for a Louisville searcher, "equipment for sale" goes from generic global groups to *Kentucky Farm Equipment for sale*.
- *Social*: a **separately trained** graph embedding of users and entities, fed in as a feature. They did not learn the graph inside the retrieval model.

**Training data.** Positives = clicks. Negatives = uniform random documents from the index.

**Serving.** Faiss for quantisation. Coarse quantisation (k-means, IVF or IMI) picks the cluster; **product quantisation** compresses the residual. At index time, document → one term (cluster) + one payload (PQ code). At query time, `(nn)` is rewritten into an `or` over the `nprobe` closest cluster terms, then payloads are decoded to check the radius. **Radius mode beat top-$K$ mode** in production, because a radius constraint composes with the rest of the Boolean expression, whereas top-$K$ forces a scan of the whole index before the other filters apply.

The two towers are split at serving time: query tower runs online in a realtime inference service; document tower runs as a nightly Spark batch job, writing embeddings into the forward index and quantised codes into the inverted index.

**Query and index selection.** EBR is *not* triggered for every query. Skipped: navigational queries where the user is re-finding something they already clicked, and queries whose intent is off-distribution for the model. The index is also pruned — only monthly-active users, recent events, popular pages/groups.

**Later-stage optimisation.** New candidates from EBR confuse rankers that were trained on the old retrieval distribution. Two fixes: (a) push the query-document **cosine similarity** down the funnel as a ranking feature — they tried Hadamard product and raw embeddings too, and plain cosine won consistently; (b) a **human-rating feedback loop** — log what EBR newly surfaced, have raters label relevance, retrain the relevance filter on those labels so it can kill EBR's junk without killing its wins.

## Ablation Studies and Experiments

Offline metric: recall@K by exact KNN over the full index, averaged over 10,000 sampled search sessions. Everything below is offline recall unless noted.

**Unified vs text embedding.** +18% recall for events search, +16% for groups. Feature-by-feature on groups (text baseline): location features **+2.20%**, then social graph embeddings **+1.77%** on top.

**Negatives — the big surprise.** Training with *non-click impressions* (shown but not clicked, from the same session) as negatives was **55% absolute worse** in recall than random negatives for people search. This is counterintuitive if you come from ranking, where non-click impressions are the obvious negative. The explanation: those negatives are all hard cases that partially match the query, but the real index is 99.99% documents that do not match at all. Training only on hard negatives destroys the model's picture of the actual retrieval space.

**Positives.** Clicks and impressions were *equally* good at equal data volume. Augmenting clicks with impressions gave **no additional gain** — so more data did not help here either. Worth noting as a rare "scale did not help" data point.

**Online hard negative mining.** For each query in a mini-batch, use the other queries' positive documents as the candidate pool and pick the highest-scoring one(s) as negatives. Gains: **+8.38%** recall people search, **+7%** groups, **+5.33%** events. Best setting is **at most two hard negatives per positive** — more than two *regresses* quality.

**Offline hard negative mining.** Generate top-$K$ for each query with ANN, select negatives, retrain, iterate. Findings:
- Training on hard negatives **alone** does *not* beat random negatives. Analysis showed the "hard" model overweighted social/location features and got *worse* at text matching.
- **Do not use the hardest examples.** Sampling from ranks **101–500** gave the best recall.
- **Mix easy and hard**, and the optimal ratio is extreme: easy:hard = **100:1**, still improving up to that point.
- **Transfer learning direction matters and is asymmetric.** Hard model → easy model helped. Easy → hard did not.
- Running ANN on a single random shard is enough to mine effective negatives, since you only want *semi-hard* ones anyway.

**Hard positive mining.** Mine likely targets from user activity logs in *failed* search sessions — results production never retrieved. This data is **4% of the click data volume** but trains a model to **the same recall**. Combining hard positives with impressions improves further.

**Ensembles.**
- *Weighted concatenation*: concatenate L2-normalised per-model vectors, with weights $\alpha_i$ applied to the query side only, so that $\cos(E_Q, E_D) = S_w(Q,D) / (\sqrt{\sum \alpha_i^2}\sqrt{n})$ — i.e. a weighted sum of per-model cosines is recoverable as one plain cosine over one longer vector, so serving is unchanged.
- The best *offline* second-stage model was the non-click-impression model: **+4.39%** recall. But it **degraded badly under quantisation**, so the online benefit largely evaporated. The shipped choice was a relatively easy model ensembled with a tuned offline-HNM model — worse offline, significantly better online.
- *Cascade model*: run stage 2 on stage 1's output. Non-click-impression model was a poor cascade stage, and its gain **shrank as the rerank depth grew**. Offline-HNM model gave **+3.4%**, which makes sense — its training triplets were literally built from stage-1 output.
- A second cascade that worked: **text embedding first to pre-select text-matching candidates, then unified embedding to rerank.** Unified embedding had shifted attention toward social/location and introduced *new text-match failures*; this ordering repairs that and gave significant online gains.

**ANN tuning (Table 3, 128-d, 1-recall@10, ≈0.58% of index scanned).**

| Scheme | 1-recall@10 |
|---|---|
| PCA,PQ16 | 51.62% |
| PQ16 | 67.54% |
| OPQ16,PQ16 | 70.51–74.29% |
| PQ32 | 74.27% |
| PQ64 | 74.81% |
| Flat (no quantisation) | 74.81% |

Lessons: **always try OPQ** (it beats PCA clearly); pick `pq_bytes` $= d/4$ because accuracy saturates past that (PQ32 ≈ PQ64 ≈ Flat at $d=128$); compare methods at equal **number of documents scanned**, not equal `num_cluster`/`nprobe`, because IMI clusters are badly imbalanced — roughly half hold only a few points; and **re-tune ANN parameters whenever the training task changes non-trivially**, because a model that is better in exact KNN can be worse after quantisation.

## Worth Remembering

- The single most transferable finding: **hardness is a knob with an interior optimum, in three places at once** — margin $m$, negative rank window (101–500, not top-1), and easy:hard ratio (100:1). The retrieval model has to model the *whole* input distribution, which is overwhelmingly trivial non-matches, so a diet of pure hard negatives distorts it. Compare [[Understanding Contrastive Learning through Alignment and Uniformity|alignment vs uniformity]] — hard negatives buy alignment/precision, easy negatives preserve coverage of the space.
- **Offline recall and online gain diverge, and the culprit is quantisation.** The non-click-impression ensemble was best offline and disappointing online. Any offline evaluation of an EBR model should run through the actual quantised index.
- Adding social/location features *created* text-match regressions. There is no free lunch: pushing a single embedding to model more things makes it worse at the thing it was already good at. Their answer was a cascade, not a bigger model.
- The system-design lesson generalises well beyond Facebook: express ANN in terms of primitives your existing engine already has (a term + a payload) rather than standing up a parallel vector service. You inherit realtime updates, filtering, and query planning, and you get filtered ANN for free — a thing that is still awkward in many vector databases today.
- **Limitations the authors admit**: this is only step one; the text tower is small n-gram machinery, not [[BERT- Pre-training of Deep Bidirectional Transformers|BERT]]; models are per-vertical rather than universal; and heavy end-to-end tuning across the whole ranking stack was needed to actually realise the retrieval wins. If your ranker was not built for the new candidates, EBR can look neutral or negative online.
- Practical caveat: EBR is deliberately *not* triggered on every query. Blanket rollout would over-trigger, cost capacity and add junk. The trigger policy is part of the system, not an afterthought.
- Open questions: they never report how much of the online win comes from the raw retrieval gain versus the human-rating feedback loop filtering EBR's mistakes. Also, hard-positive mining giving equal recall on 4% of the data is a striking data-efficiency result that deserves more than the two sentences it gets.

## Links

Related: [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)]] · [[Real-time Personalization using Embeddings for Search Ranking at Airbnb (KDD, best paper)]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[Sentence-BERT]] · [[BPR- Bayesian Personalized Ranking from Implicit Feedback]] · [[Understanding Contrastive Learning through Alignment and Uniformity]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[Recommender Systems - Evolution]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[PinnerFormer- Sequence Modeling for User Representation at Pinterest]] · [[Deep Interest Network for CTR Prediction (DIN)]]

New topics worth writing: Triplet loss and margin selection, Product quantization and OPQ, Inverted file index (IVF) vs inverted multi-index (IMI), Faiss, Approximate nearest neighbour search, Hard negative mining, Two-tower retrieval, Character n-gram text representation, Graph embeddings, Filtered / constrained ANN search
