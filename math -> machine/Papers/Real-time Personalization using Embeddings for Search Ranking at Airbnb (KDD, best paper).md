---
title: "Real-time Personalization using Embeddings for Search Ranking at Airbnb (KDD, best paper)"
authors: ["Grbovic & Cheng"]
year: 2018
url: https://www.kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb
priority: Must-Read
read_on: 2026-08-23
tags: [paper]
---
## The Core Idea

Take word2vec, but the "words" are Airbnb home listings and the "sentences" are user click sessions. Listings that get clicked near each other in the same browsing session end up with similar vectors. Then use cosine similarity between those vectors as **features inside a ranking model**, computed live at request time.

Three things make this more than "we ran word2vec on clicks":

1. **Real-time, not precomputed.** Most embedding deployments of the era built offline lookup tables: user → items, item → items. Airbnb loaded all 4.5 million 32-dimensional listing vectors into the memory of the search machines, kept a rolling 2-week history of each user's clicks/skips/wishlists in Kafka, and computed the similarity between candidate listings and that history *during the search request*. So a user who clicks two treehouses sees more treehouses on the very next search.

2. **The marketplace is two-sided, so negatives are real.** A host can reject a guest. That is an explicit negative signal that no pure "clicks" model has. They push rejected (user type, listing type) pairs apart in the embedding space.

3. **Bookings are too sparse to embed per-user or per-listing.** People travel 1–2 times a year. A listing might have 3 bookings ever. Solution: stop embedding IDs, embed *types* — hand-built buckets over metadata (country, room type, price bucket, review count, capacity...). Many listings map to one `listing_type`. Many users map to one `user_type`. Now every type has thousands of observations, and cold-start users get a vector for free because their type is computable from profile fields alone.

Why it did not exist before: the pieces (skip-gram, [[NDCG]]-style learning-to-rank) were all standard. The contribution is the set of domain hacks that make embeddings survive contact with a sparse, two-sided, geographically clustered marketplace, plus the engineering to serve them in the request path.

> [!NOTE] Type embedding
> Instead of learning one vector per entity ID, define a many-to-one rule mapping from metadata to a discrete "type", and learn one vector per type. Trades resolution for data density, and solves cold start by construction. ^type-embedding

---

## The Methodology

### Part 1 — Listing embeddings (short-term, from clicks)

A session $s = (l_1, \dots, l_M)$ is an unbroken run of listing clicks by one user. New session starts after 30 minutes of inactivity. Clicks under 30 seconds on the listing page are dropped as accidental. Sessions of length 1 dropped.

Base objective is plain skip-gram with negative sampling:

$$\arg\max_\theta \sum_{(l,c)\in D_p} \log \frac{1}{1+e^{-v'_c v_l}} + \sum_{(l,c)\in D_n} \log \frac{1}{1+e^{v'_c v_l}}$$

$D_p$ = (centre listing, listing clicked within window $m$). $D_n$ = (centre listing, random listing from the whole 4.5M vocabulary). Trained by stochastic gradient *ascent*.

Two modifications:

**Booked listing as global context.** If a session ends in a booking $l_b$, add a term that is always on, no matter where the window sits:

$$+ \log \frac{1}{1+e^{-v'_{l_b} v_l}}$$

So as the window slides from the first click to the booking, every listing in the session is pulled toward the thing that was actually booked. Booked sessions were also oversampled **5×** in training data.

**Market negatives (the congregated-search fix).** People search inside one city. So $D_p$ is nearly all same-city pairs, while random negatives $D_n$ are nearly all different-city. The model can get low loss by just learning "which city is this", and within-city similarity stays bad. Fix: add $D_{mn}$, negatives sampled *from the same market as the centre listing*:

$$+ \sum_{(l,mn)\in D_{mn}} \log \frac{1}{1+e^{v'_{mn} v_l}}$$

**Cold start.** New listing with no clicks: find the 3 nearest listings within 10 miles that share listing type and price bucket, average their vectors. Covers 98% of new listings.

**Training details.** $d = 32$ (chosen for RAM on the search boxes, not just accuracy). Window $m=5$. 10 epochs. 800M click sessions. Modified word2vec C code. MapReduce with 300 mappers feeding a single multithreaded reducer. Orchestrated by Airflow. **Retrained from scratch daily** on a sliding multi-month window — incremental continuation gave worse offline numbers. Vectors jump around day to day, which is fine because only cosine similarity is consumed downstream.

### Part 2 — User-type & listing-type embeddings (long-term, from bookings)

Booking sessions are sequences of `(user_type, listing_type)` tuples ordered in time, per user. 50M booking sessions, 500K user types, 500K listing types, $d=32$, $m=5$.

The trick that puts users and listings in **one shared space**: interleave them in the same sequence. The sliding window sometimes centres on a user type, sometimes on a listing type, and each is updated to predict its neighbours — which include the other kind. So $\cos(v_{ut}, v_{lt})$ is meaningful.

Note the user type is recomputed *up to each booking*, so the same person contributes several different user-type vectors over their life as their price point and review stats drift.

**Rejections as explicit negatives.** Collect $D_{reject}$: pairs where a host rejected this user type, focusing on cases where the rejection was followed by a successful booking elsewhere by the same user. Add a repelling term:

$$+ \sum_{(ut,lt)\in D_{reject}} \log \frac{1}{1+e^{v'_{lt} v_{ut}}}$$

This encodes host taste: listing types that tolerate empty profiles and low guest ratings sit closer to those user types.

No market negatives here — booking sessions naturally span cities.

### Part 3 — Feeding it to the ranker

The ranker is a **GBDT with Lambda Rank modification**, framed as pairwise regression on utilities. Labels come from waiting 1 week after each search:

| outcome | $y$ |
|---|---|
| booked | 1 |
| host contacted, no booking | 0.25 |
| clicked | 0.01 |
| viewed only | 0 |
| **host rejected the guest** | **−0.4** |

Only search sessions containing at least one booking are kept, truncated at the last clicked result. ~100 features. Trained on the most recent 30 days. Offline metric is [[NDCG]] on a 20% hold-out.

Eight new embedding features (Table 6), computed online. Each uses a real-time 2-week user history set: clicked $H_c$, long-clicked (>60s) $H_{lc}$, skipped $H_s$ (a listing passed over in favour of a lower-ranked click), wishlisted $H_w$, inquired $H_i$, booked $H_b$.

For each set, split by market, average the vectors inside each market to get a centroid, then take the **max** cosine similarity across markets:

$$\text{EmbClickSim}(l_i, H_c) = \max_{m \in M} \cos\Big(v_{l_i}, \sum_{l_h \in m,\, l_h \in H_c} v_{l_h}\Big)$$

The max-over-markets matters: a user browsing both NY and LA should not have their intent blurred into a single meaningless average vector.

Plus `EmbLastLongClickSim` (similarity to the single most recent long click) and `UserTypeListingTypeSim` $= \cos(v_{ut}, v_{lt})$.

---

## Ablation Studies and Experiments

### Offline embedding comparison (Figure 6)

Setup: given a user's most recent click, rank the candidate listings by cosine similarity to it, and record where the eventually-booked listing lands. Averaged over clicks going back 17 steps from the booking. Lower rank = better.

Three variants compared:
- `d32` — plain skip-gram
- `d32 book` — + booked listing as global context
- `d32 book + neg` — + same-market negatives

**`d32 book + neg` wins.** Both additions help, and the market-negative fix is the one specific to this domain. The plot also shows the production ranker gets stronger as clicks accumulate (it has memorisation features), so embedding similarity is most valuable **early in the funnel**, when there is little history.

### Similar Listings carousel (online A/B)

Baseline was calling the full search ranker for the same location, then filtering on availability, price, listing type. Replaced with online k-NN in embedding space, restricted to same market and available for the requested dates.

- **+21% CTR** on the carousel (+23% with dates entered, +20% dateless)
- **+4.9%** in guests who booked the listing they found via the carousel

### Search ranking (offline, Table 8)

Same model, same data, with vs. without embedding features:

| Metric | Lift |
|---|---|
| DCU −0.4 (rejections) | +0.31% |
| DCU 0.01 (clicks) | +1.48% |
| DCU 0.25 (contacts) | +1.95% |
| DCU 1 (bookings) | **+2.58%** |
| **NDCU** | **+2.27%** |

Bookings ranked higher with essentially no increase in rejections ranking higher — the rejection number being flat is the point, not an accident.

### Which feature is actually doing the work (Table 7)

| Feature | Coverage | Importance rank (of 104) |
|---|---|---|
| EmbClickSim | 76.2% | **5** |
| EmbSkipSim | 78.6% | **8** |
| EmbLastLongClickSim | 48.3% | 11 |
| EmbInqSim | 20.6% | 12 |
| EmbLongClickSim | 51.1% | 20 |
| UserTypeListingTypeSim | 86.1% | 22 |
| EmbWishlistSim | 36.5% | 47 |
| EmbBookSim | 8.1% | 46 |

Five of eight land in the top 20. The big finding: **skips are nearly as informative as clicks** (rank 8 vs rank 5). Negative in-session signal is cheap, high-coverage and underused.

The long-term feature `UserTypeListingTypeSim` (rank 22, from booking-session training) beats the short-term `EmbBookSim` (rank 46, from click-session training on 2 weeks of bookings) — evidence that training on booking sequences, not click sequences, is the right move for long-horizon taste.

### Sanity checks that shaped decisions

- k-means on the embeddings recovers geography (100 clean clusters in California), which they then used to redraw their internal market definitions.
- Cosine similarity is higher within listing type (Entire Home ↔ Entire Home 0.895) than across (Entire Home ↔ Shared Room 0.848), and higher within price bucket than across. Price and room type are recoverable from metadata anyway; the interesting claim is that *style and architecture* — houseboats, treehouses, castles — also cluster, which no hand-made feature captures.
- Partial dependency plots confirmed the GBDT uses the features in the intended direction: high `EmbClickSim` → higher score, high `EmbSkipSim` → **lower** score, high `UserTypeListingTypeSim` → higher score.

### Things that did not work

- **Incremental training** of embeddings day over day: worse than retraining from scratch daily.
- **Plain skip-gram without market negatives**: learned good cross-city separation but sub-optimal within-city similarity, which is the only kind that matters when the user has already picked a city.
- **Per-`listing_id` booking embeddings**: abandoned. Booking data is too small, most users have exactly 1 booking (session length 1, unusable), most listings have fewer than the 5–10 occurrences needed for a stable vector, and preferences drift between bookings anyway.

---

## Worth Remembering

- **The negative-sampling distribution is a modelling choice, not boilerplate.** This is the most transferable lesson. If your positives are drawn from a narrow slice (one city, one category, one query intent) and your negatives are uniform over the whole vocabulary, the model solves the easy discrimination and never learns the fine-grained one you care about. Sample hard negatives from the same slice. This idea recurs everywhere — see the sampling discussion in [[PinnerFormer- Sequence Modeling for User Representation at Pinterest]].

- **Embeddings as features, not as the model.** The vectors never rank anything directly in search. They produce eight scalar similarities that a GBDT consumes alongside ~96 hand-crafted features. This is a much lower-risk deployment than replacing the ranker, and it lets the tree model decide how much to trust them. Compare with [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]], where trees feed a linear model — same "learned representation as input to a simpler learner" pattern.

- **The −0.4 rejection label is the two-sided marketplace showing up in the loss.** Most ranking papers have only positive utilities of varying strength. Here one class of outcome is actively harmful and gets negative utility, and the same signal is injected a second time into the embedding space via $D_{reject}$.

- **32 dimensions was chosen by RAM budget.** Vectors for 4.5M listings live sharded across the search machines so similarity can be computed in the request path. Serving constraints picked the hyperparameter.

- **Retraining daily from scratch is safe here only because nothing downstream consumes raw coordinates.** Cosine similarity is invariant to the arbitrary rotation/relabelling that a fresh random init produces, so the GBDT's learned thresholds on those features stay valid. If you ever feed the raw vector into a downstream model, this trick breaks. A quiet instance of the entanglement problem in [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]].

- **The type mappings are hand-designed buckets** (Table 3 and 4 — country, room type, 8 price buckets, review count bands, capacity, ...). Buckets were chosen "in a data-driven manner to maximise coverage", but they are still human-specified. This is the paper's biggest maintenance liability and the thing a modern approach would learn end-to-end.

- **Limitation the authors are honest about:** the offline evaluation only measures whether cosine similarity to the last click ranks the booked listing well. That is a proxy, not the objective. They still ran a full online A/B before shipping, and later ran a **back test removing the features**, which produced negative bookings — the cleanest possible confirmation that the feature was carrying real weight and not just correlated with an existing signal.

- **Follow-up questions:** the max-over-market-centroids aggregation is crude — a mean of a set of vectors is a weak summary compared to attention over the sequence, which is exactly what [[Deep Interest Network for CTR Prediction (DIN)]] and later [[BERT4Rec- Sequential Recommendation with Bidirectional Transformer]] do. How much of the 2.27% NDCU comes from the embeddings versus just from *having any real-time in-session signal at all*? An ablation with a naive count-based in-session feature would have answered that, and is not present.

---

## Links

Related: [[Recommender Systems - Evolution]] · [[Deep Interest Network for CTR Prediction (DIN)]] · [[PinnerFormer- Sequence Modeling for User Representation at Pinterest]] · [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] · [[NDCG]] · [[BERT4Rec- Sequential Recommendation with Bidirectional Transformer]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Foundational_RecSys_Ranking_Reading_Plan]] · [[Linear Projection]]

New topics worth writing: word2vec / skip-gram with negative sampling, LambdaRank and LambdaMART, Gradient Boosted Decision Trees, hard negative mining, two-sided marketplace objectives, cold-start strategies, item2vec, node2vec / DeepWalk, partial dependency plots, Kafka for real-time feature stores
