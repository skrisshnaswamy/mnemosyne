---
title: "Amazon.com Recommendations: Item-to-Item Collaborative Filtering (IEEE Internet Computing)"
authors: ["Linden", "Smith & York"]
year: 2003
url: https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper]
---
## The Core Idea

Recommendation in 2003 meant: find people like you, then show what they bought. That is **user-to-user collaborative filtering**. It has a fatal cost problem. To recommend for one shopper you must compare their purchase history against every other shopper — 29 million of them at Amazon — and you must do it in under half a second while the page loads.

The trick here is to flip the axis. Instead of asking "which customers are similar to this customer?", ask "which *items* are similar to this item?". Item similarity is stable. Two books that get bought together stay bought together for weeks. Customer similarity is not stable — every click changes a customer vector.

So you can precompute the whole item–item similarity table **offline**, overnight, in a batch job. At request time you do almost nothing: look up the handful of items in this user's history, pull their neighbours from a table, merge, rank, return. The online cost depends only on how many items *this one user* touched — not on the number of customers $M$, not on the catalogue size $N$.

That is the whole insight, and it is a systems insight more than a modelling one. The unlock: personalisation that responds *instantly* to a new event (you just added a book to your cart → the cart page changes now), works from **two or three** items of history instead of needing a rich profile, and never degrades quality by sampling or clustering customers away.

> [!NOTE] Item-to-item collaborative filtering
> Recommend by matching each item the user touched to similar items, where "similar" means "bought by the same customers". The expensive part — the item×item similarity table — is computed offline; the online step is a table lookup. ^item-to-item

## The Methodology

**The customer vector (the old way, for contrast).** A customer is a vector in $\mathbb{R}^N$, one dimension per catalogue item. Positive entries for purchased or well-rated items, negative for badly-rated ones. Components are scaled by **inverse frequency** — divide by the number of customers who bought that item — so that *Harry Potter* stops dominating every neighbourhood and rare items carry signal. (Same motivation as IDF in search, same motivation as the subsampling in [[Distributed Representations of Words and Phrases (negative sampling)]].) Similarity is the cosine:

$$\text{sim}(\vec{A},\vec{B}) = \cos(\vec{A},\vec{B}) = \frac{\vec{A}\cdot\vec{B}}{\|\vec{A}\|\,\|\vec{B}\|}$$

**The item vector (the new way).** Same cosine, transposed. Each *item* is a vector in $\mathbb{R}^M$, one dimension per **customer**, non-zero where that customer bought the item. Two items are similar if the same people bought both.

**Building the table.** The naive way — loop over all $\binom{N}{2}$ item pairs — is hopeless, and mostly wasted, because the overwhelming majority of pairs share zero customers. Instead they walk the co-purchase graph:

```
for each item I1 in catalog:
    for each customer C who purchased I1:
        for each item I2 purchased by C:
            record co-purchase (I1, I2)
    for each I2 seen:
        compute sim(I1, I2)
```

You only ever touch pairs that actually co-occur. Worst case is $O(N^2 M)$, but because the average customer bought very few things, real-world cost is about $O(NM)$. Further speedup: **sample** the customers of best-selling titles — a book bought by two million people does not need all two million to estimate its neighbours — which they report costs "little reduction in quality".

**Serving.** Take the user's purchased and rated items. For each, fetch its similar-items row. Aggregate the rows (union the candidates, sum or otherwise combine the similarity scores), drop things the user already owns, sort, return the top few. Cost is $O(|\text{user history}| \times k)$. Nothing scales with $M$ or $N$.

**Where it shipped.** The homepage "Your Recommendations" module, most product pages, email campaigns, and the shopping-cart page — cart recommendations conditioned on what is in the cart right now, described as "impulse items at the supermarket checkout, but targeted."

## Ablation Studies and Experiments

Be blunt: **there are none.** This is a four-page industry report in *IEEE Internet Computing*, not a research paper. It contains zero tables, zero benchmark numbers, and no offline metric. The only quantitative claims in the entire text are:

- Amazon has "more than 29 million customers and several million catalog items."
- The latency budget is "no more than half a second."
- Click-through and conversion "vastly exceed" untargeted content such as banner ads and top-seller lists — with no figures attached.
- MovieLens (35k users × 3k items) and EachMovie (4k × 1.6k) are called out as being **three orders of magnitude** smaller than the production problem. This is the paper's sharpest empirical point, and it is a point about everyone else's evaluation, not their own.

What the paper *does* do carefully is a complexity argument against the three alternatives:

| Approach | Offline work | Online cost | Failure mode |
|---|---|---|---|
| User-user CF | ~none | $\approx O(M+N)$ | Breaks at 10M users / 1M items |
| Cluster models | heavy (clustering) | $O(\#\text{segments})$ | Quality is poor — you are matched to a *segment*, not to your actual nearest neighbours |
| Search / content-based | index build | fine for short histories | Recommendations are too generic ("best-selling dramas") or too narrow ("everything by this author"); breaks for users with thousands of purchases |
| Item-item CF | $\approx O(NM)$ table build | $O(\text{user history})$ | (see caveats) |

**What did not work — the fixes they rejected.** All the standard scaling patches for user-user CF are listed and shot down one by one, and this is the most useful part of the paper:

- *Sample the customers* → your sampled neighbours are less similar to the user → worse recommendations.
- *Discard low-activity customers* → those users get nothing.
- *Partition the item space by category* → recommendations can never cross a category boundary, which kills discovery.
- *Discard very popular or very unpopular items* → dropped items can never be recommended, and customers who only bought those items get nothing at all.
- *Dimensionality reduction (clustering, PCA) on the item space* → same effect as discarding low-frequency items, it wipes out the tail.
- *Dimensionality reduction on the customer space* → this **is** clustering, so it inherits the cluster-model quality problem.
- *More, finer segments to fix cluster quality* → online user→segment classification becomes as expensive as just doing user-user CF, so you gain nothing.

The implicit ablation, then: the component doing the work is not the similarity function (plain cosine, nothing clever) — it is the **decision about what to precompute**. Item similarity is cacheable because it changes slowly; user similarity is not.

## Worth Remembering

- **Cold start is nearly solved on the user side.** Two or three items produce decent recommendations, because you are not trying to estimate a user's position among 29 million people — you are just looking up neighbours of the things they touched. Cold start on the *item* side is untouched and remains hard: a brand-new item has no co-purchases, so it has no row in the table.
- **The offline/online split is the durable lesson**, and it survives into everything since. It is the same shape as the retrieval/ranking funnel in [[Deep Neural Networks for YouTube Recommendations (RecSys)]], and the same shape as the modern two-tower setup where item embeddings are precomputed and indexed with [[Efficient and robust approximate nearest neighbor search using HNSW]] or [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)]].
- **The similar-items table is just a sparse, non-learned nearest-neighbour index over items.** Later work replaces it with learned item vectors — [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)]], [[Collaborative Filtering for Implicit Feedback Datasets (ICDM)]], [[Factorization Machines (ICDM)]], and eventually sequence models like [[Self-Attentive Sequential Recommendation (SASRec)]]. The serving *shape* stays identical.
- **This algorithm is a fearsome baseline.** [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] found that well-tuned item-kNN methods beat most published neural recommenders. If you build something new, tune this properly first, or your comparison is worthless.
- **Practical caveat: co-purchase similarity is popularity-contaminated.** Cosine on binary purchase vectors still leans toward blockbusters; the inverse-frequency weighting mentioned for the user-vector version is the fix, and you should apply the analogous correction on the item side. Compare the explicit treatment in [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)]].
- **Second caveat: it recommends substitutes, not complements.** "People who bought this camera also bought this camera" is a bad cart recommendation. The paper does not distinguish the two, and separating them is real production work.
- **Third caveat: pure co-purchase is a feedback loop.** The table is built from purchases that were themselves driven by last week's recommendations. The paper never mentions this. See [[Recommendations as Treatments- Debiasing Learning and Evaluation]] and [[Counterfactual Reasoning and Learning Systems]].
- The core method is US Patent 6,266,649 (Linden, Jacobi, Benson, 2001), and the closest academic sibling is Sarwar et al., "Item-Based Collaborative Filtering Recommendation Algorithms," WWW 2001 — which *does* have the numbers this paper lacks.
- The 2003 closing prediction — that offline retailers would adopt this for coupons and postal mail — was correct and understated.

## Links

Related: [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)]] · [[Collaborative Filtering for Implicit Feedback Datasets (ICDM)]] · [[Recommender Systems - Evolution]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Efficient and robust approximate nearest neighbor search using HNSW]] · [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)]] · [[Self-Attentive Sequential Recommendation (SASRec)]] · [[BPR- Bayesian Personalized Ranking from Implicit Feedback]] · [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[Real-time Personalization using Embeddings for Search Ranking at Airbnb (KDD, best paper)]] · [[Foundational_RecSys_Ranking_Reading_Plan]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]]

New topics worth writing: Item-based kNN recommenders, Cosine similarity, Inverse document frequency (IDF), Cold start problem, Sparse matrix representations, Substitutes vs complements in retail recommendation, Feedback loops in deployed recommenders, Offline–online architecture split in ML serving
