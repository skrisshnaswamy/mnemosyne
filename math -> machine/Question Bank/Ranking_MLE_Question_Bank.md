## Part 28 — The Search & Ranking Stack, End to End

### Q170. Walk me through how a production search and ranking stack works.

The organising principle is a **compute budget per candidate**. I have a fixed latency budget — call it 200-300ms at p99 for the whole request — and millions of items. I can't afford a heavy model on all of them, and I don't need one, because most items are obviously irrelevant. So I build a funnel where each stage narrows the set and spends proportionally more compute per surviving candidate.

The stack I've worked with had four stages:

**1. Retrieval / candidate generation.** Millions of merchants down to roughly a thousand. Cheap per item — a dot product against an ANN index, plus some non-model channels. Budget: maybe 20-30ms.
>[!TIP] In a non-keyword personalized homepage, there may not be a literal **text query** like `"vegetarian restaurants"`.
>Instead, the retrieval system can construct a **request/context representation** from the user and the current context.
>
So you can think of:
> **`user + context → query/context embedding`**
>
and then:
> **`query or context embedding vs. merchant embeddings`**
>
>using something like a dot product.
>

**2. Coarse ranking (pre-ranking).** A thousand down to about a hundred. A lightweight model, restricted feature set, no expensive cross-features. Budget: ~20ms for a thousand scores.

**3. Fine ranking.** A hundred down to a final ordering, of which the top twenty are what actually renders. This is where the real model lives — full feature set, user behaviour sequence, multi-task heads. Budget: ~50-80ms for a hundred candidates, which is where most of the model budget goes.

**4. Re-ranking / blending.** Business logic on the final list: diversity so you don't show eight burger places in a row, deduplication, sponsored placement blending, freshness or new-merchant boosts, hard business rules. Usually not a learned model, or a light one.

The rough arithmetic: each stage cuts the candidate count by about 10× and increases per-candidate compute by about 10-100×, so **total compute per stage is roughly constant.** That's the design constraint, and it's why the funnel has the shape it does rather than being one big model.

**The thing I'd emphasise, because it's what people miss: each stage optimises a different objective.**

Retrieval is graded on **recall** — did the good stuff survive? Order doesn't matter at all, because a ranker comes after. Fine ranking is graded on **ordering quality** — NDCG, or calibrated probabilities. Coarse ranking sits in between and this is the subtle one: its job isn't to predict clicks, it's to **not discard anything the fine ranker would have ranked highly.** So the right target for a coarse ranker is often the *fine ranker's scores*, not user labels. You distil the expensive model into the cheap one. That framing surprises people and it's the single most useful thing to say about the pre-ranking stage.

**On pagination**, which is a real design decision: in the system I worked with, retrieval results were cached per query-session, and each scroll re-ran coarse and fine ranking but not retrieval. The reasoning is that retrieval is the expensive, high-variance stage and the candidate pool doesn't change within a session — but the *context* does. By the time the user scrolls, you know what they've already been shown and didn't tap, so re-ranking with updated impression context is worth the compute. The trade-off is that you've now capped session-level diversity at whatever the initial retrieval produced: if the first thousand candidates were narrow, no amount of scrolling escapes that. Whether that's acceptable depends on how deep users actually scroll, and it's worth measuring rather than assuming.

**The failure modes that come from the cascade itself**, as distinct from any single model:

- **The recall ceiling.** Anything retrieval drops is unrecoverable. If retrieval recall@1000 is 85%, your entire system is capped at 85% regardless of how good the ranker is. So I'd always measure each stage's recall against the final ground truth separately — otherwise you spend six months improving a ranker when the loss is upstream.

- **Cascade training/serving mismatch.** Each stage is served candidates produced by the previous stage, but is usually *trained* on a different distribution — often random or in-batch negatives. So the fine ranker trains on easy negatives and serves on the hundred hardest items in the catalog, which is exactly the distribution it saw least. The fix is to train each stage on the actual output distribution of the stage above it, which means logging candidate sets at every boundary.

- **Selection bias compounding down the funnel.** You only observe labels for items that survived every stage and got impressed. Retrain on that, and each generation of the model reinforces the previous generation's choices. This is the feedback loop problem, and in a cascade it compounds at every level.

- **Ownership seams.** In most orgs retrieval and ranking are different teams, and the interface between them — what's in the candidate set, what features travel with it — is where bugs live and where nobody owns the end-to-end metric.

↪ **Your hook:** you have the architecture; lead with the compute-budget framing rather than the stage list, since that's what shows you understand *why* it's built that way. And have your actual numbers ready — candidate counts, latency budget, QPS.

---

### Q171. Go deeper on retrieval. How does the candidate generation stage actually work?

The first thing I'd say is that **retrieval is usually not one model — it's a union of several parallel channels**, each with its own quota, merged and deduplicated. That surprises people who assume it's just a two-tower embedding lookup.

A typical channel mix for a marketplace:

- **Embedding retrieval (two-tower)** — the semantic/personalised workhorse. Usually the largest quota.

- **Item-to-item collaborative filtering** — "users who ordered from X also ordered from Y," precomputed offline. Cheap, high precision, and it catches co-occurrence patterns embeddings miss.

- **Geo/proximity** — in food delivery this is a hard constraint, not a signal. Nothing outside the delivery radius can be a candidate at all.

- **Keyword / BM25** — for typed queries. Essential because dense retrieval fails on exact names, brand terms, and specific dish names (Q120).

- **Re-order / history** — the user's previous merchants. Very high precision, and if you don't have this channel your embedding model has to rediscover it, badly.

- **Popularity and trending**, often geo-scoped — the fallback that guarantees a non-empty result for a cold user.

- **Category or intent-based** — from a query classifier.

- **Exploration channel** — a small quota of deliberately under-exposed or new merchants, which is how you break the cold-start feedback loop (Q92, Q55).

**Why multi-channel rather than one better model:** each channel has different and largely uncorrelated failure modes. Embeddings fail on exact match and on brand-new items; keyword fails on paraphrase; CF fails on cold items; popularity fails on personalisation. Union coverage beats any single channel's coverage, and the quotas give you a direct, interpretable lever — "raise the exploration quota to 3%" is a conversation you can have with a PM, unlike "adjust the temperature in the contrastive loss."

**On the two-tower model itself:** a query tower encoding user and context, an item tower encoding the merchant, both producing vectors in a shared space, scored by dot product or cosine. The reason it's structured that way is precomputation — the item tower doesn't depend on the request, so you embed the whole catalog offline and load it into an ANN index. At serve time it's one small forward pass plus a sublinear index lookup. The cost is that the towers can't interact, so any query-item interaction has to be compressed into two independent vectors before they meet (Q116). That's a real expressiveness limit, and it's precisely why a ranker follows.

**Training decisions that actually matter:**

- **In-batch negatives with a sampled softmax**, because you can't compute a softmax over millions of items. Batch size *is* your negative count, so quality scales with it — and cross-GPU gathering of negatives is nearly free in a distributed setup.

- **The logQ correction.** With negatives sampled from real traffic, popular merchants appear as negatives far more often and get systematically pushed down. Subtracting `log Q(item)` from each logit — where Q is the item's sampling probability, estimated from a streaming frequency counter — makes the sampled softmax an unbiased estimate of the full one. Without it your retriever quietly under-ranks the head of the catalog. This is standard in production and it's the detail that signals you've actually built one of these.

- **Hard negative mining** — retrieve top-k with the current model, exclude known positives, use the rest. Bigger quality lever than any architecture change. But watch for false negatives: mined "hard negatives" from an unlabelled catalog are often unlabelled positives, and you're then training the model that a correct answer is wrong (Q66).

- **The structural gap:** at training time the model discriminates against a few thousand batch negatives; at serving time it competes against the entire catalog. It has never been asked to distinguish most of them. This is why in-batch metrics look great and full-catalog recall is worse, and it's why evaluation must be against the **full catalog**, not the batch.

**Metrics:** recall@k against whatever the user eventually engaged with — ordered from, or clicked, depending on what you're optimising. Not NDCG; order is irrelevant here. And I'd measure **model recall and ANN recall separately**, because your true recall is the product of the two. If the index returns 92% recall@1000, you've lost 8% before ranking runs, and that's an index-tuning problem, not a model problem. People conflate these constantly.

**Operational issues worth naming:**

- **Freshness.** A new merchant isn't retrievable until it's in the index. If the index rebuilds nightly, a merchant onboarded at 9am is invisible all day. Usually you need a real-time path — a small incremental index searched in parallel with the main one.

- **Filtering vs retrieval.** If you retrieve top-1000 by relevance and *then* filter by open-now and delivery radius, you can be left with a handful of results at 11pm. Filters have to be pushed into the index (most vector DBs support this now) or you over-retrieve heavily. This is a very common real-world bug (Q121).

- **Version coupling.** Query tower and item index must be the same model version. Serving a v2 query tower against a v1 index produces well-formed numbers and garbage results. That deserves an explicit assertion at load time, not a runbook entry.

---

### Q172. Now the ranking stage. What's actually different between coarse and fine ranking, and how does the fine ranker work?

**The difference is a feature and compute budget, and it shows up mostly in what features each can afford.**

**The coarse ranker** scores ~1000 candidates in ~20ms. That budget rules out most of what makes a ranker good:

- No expensive user-item cross features — anything requiring a per-candidate lookup or join is too slow at 1000×.

- No long behaviour-sequence attention, because running a transformer over the user's history *per candidate* is the expensive thing.

- Often a vector-product or lightly-interactive architecture, so most of the computation can be done once per request rather than once per candidate. Some designs keep it deliberately close to two-tower with a small interaction layer on top.

**And its training target should usually be the fine ranker's output, not user labels.** Its job is to not lose anything the fine ranker would have liked, so you distil: train the coarse model to reproduce the fine model's ranking on logged candidate sets. This aligns the two stages directly, and it's much more sample-efficient than learning from sparse clicks. It also means when the fine ranker improves, you re-distil rather than retrain from scratch. If someone asks how you'd improve pre-ranking, "align it with the stage it feeds" is a stronger answer than any architecture suggestion.

The metric for the coarse ranker follows from this: **recall@100 of the fine ranker's top-K**, not click AUC.

**The fine ranker** scores ~100 candidates and can afford real machinery:

- **Full feature set.** User features, merchant features, dense cross features (this user's historical order rate with this cuisine, at this hour), query-merchant relevance, real-time context (ETA, current merchant load, weather), and counters at multiple time windows.

- **Behaviour sequence modelling.** This is where DIN-style attention pays off: attend over the user's order history *weighted by relevance to the candidate merchant*, so the user representation is query-dependent rather than a fixed vector. You can only afford this because there are 100 candidates, not a million — which is a nice illustration of why the cascade exists.

- **Multi-task heads.** Typically p(click), p(order | click), maybe p(cancel) and expected basket value. Usually MMoE or PLE so tasks share a representation without fighting over it (Q78).

**Combining the heads** is where calibration becomes load-bearing. The final score is something like `p(click) × p(order|click) × value^α`, and multiplying uncalibrated probabilities is meaningless. So the fine ranker is usually trained **pointwise with cross-entropy** specifically to preserve calibrated probabilities, and the ordering falls out of the combination formula. That's a real trade — a listwise loss would give better pure NDCG — but you need the probabilities for the business formula, and often for auction or budget logic too. The exponents in that formula are business parameters, tuned online, not learned.

**Two data problems that dominate ranking in practice:**

1. **Position bias.** Your training labels come from logged impressions where position drove clicks. Train naively and you learn to reproduce the previous ranker. The production fix is a **shallow position tower**: feed position as an input to a separate small tower, and set it to a constant at serving time. The position tower absorbs the positional effect; the main tower learns relevance. Cheap, needs no randomisation, and it's what the YouTube multi-task paper describes.

2. **Delayed conversion labels.** An order confirms in minutes but a cancellation or refund lands hours later. So recent training data is systematically incomplete, and if you train on it naively you teach the model that recent items don't convert. Handle it with an explicit attribution window plus delayed-feedback modelling. Related: if you train a conversion model only on *clicked* items but serve it on all impressions, that's sample selection bias — the ESMM formulation addresses it by modelling over the entire impression space.

**Evaluation.** Offline: AUC and calibration for the pointwise heads, NDCG for the ordering, with position debiasing applied to the labels. But I'd be direct that **the offline-online gap is the hard part** — offline you rerank a logged candidate set, online your candidates come from a live retrieval stage, and the two differ. The thing I'd actually invest in is measuring how well offline predictions track online results over many experiments, because an offline metric that doesn't predict online outcomes is worse than useless — it's actively misleading and it burns team time (Q76).

**And the re-ranking layer after fine ranking** is worth mentioning because it's where a lot of user-visible quality lives: diversity (don't show one cuisine repeatedly — MMR or a slate-level objective), dedup across chains, sponsored blending with a relevance floor, and new-merchant exposure guarantees. Mostly rules and light heuristics, and it's often where the biggest complaints come from, because a technically well-ranked list can still look terrible.

↪ **Your hook:** the coarse-ranker-as-distillation-target point is the one to lead with if they probe the two-stage design, because it explains *why* two ranking stages rather than one, and most candidates can only describe that there are two.