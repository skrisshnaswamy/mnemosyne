---
title: "Multi-Interest Network with Dynamic Routing (MIND)"
authors: ["Chao Li", "Zhiyuan Liu", "Mengmeng Wu", "Yuchi Xu", "Pipei Huang", "Huan Zhao", "Guoliang Kang", "Qiwei Chen", "Wei Li", "Dik Lun Lee"]
year: 2019
arxiv: "1904.08030"
url: https://arxiv.org/abs/1904.08030
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, transformers]
---
## The Core Idea

One user, one vector is a lie. A person on Tmall buys headphones, snacks, handbags and coats in the same week. If you squash all of that into a single embedding, you get an average — a point in space that sits between four interests and is close to none of them. Retrieval from that point returns mush.

MIND gives each user $K$ vectors instead of one. Each vector is one interest. Retrieval runs $K$ separate nearest-neighbour lookups and merges the results, so the headphone interest pulls headphones and the coat interest pulls coats, with no interference between them.

The reason this did not already exist in the **matching stage** (the first funnel step that pulls ~1000 candidates out of a billion items) is a serving constraint. [[Deep Interest Network for CTR Prediction (DIN)|DIN]] already handled interest diversity by re-weighting the user's history *per candidate item*. That works, but it means you must recompute the user vector once for every item you score. Fine for 1000 items in ranking; impossible for a billion items in matching. MIND's trick is to make the user representation **item-independent** — computed once, cached, then used with plain approximate nearest neighbour search ([[Efficient and robust approximate nearest neighbor search using HNSW|ANN]] / FAISS). Diversity moves from the item level to the interest level, and that is what makes it survivable at billion scale.

The mechanism for splitting history into interests is **dynamic routing** borrowed from capsule networks. Think of it as a soft k-means run inside the forward pass: behaviours are data points, interests are cluster centroids, and the routing iterations assign behaviours to centroids and re-estimate centroids, three times.

> [!NOTE] Matching vs ranking
> Matching (candidate generation) retrieves ~1000 items from ~1e9 with a cheap, item-independent user vector plus ANN. Ranking scores those 1000 with an expensive model that can look at each item. Anything that needs the candidate item to compute the user representation is locked out of matching. ^matching-stage

> [!NOTE] Capsule
> A neuron whose output is a vector, not a scalar. Direction encodes *what property* it found; length encodes *how confident* it is that the property is present. Here: direction = which interest, length = does this user actually have this interest. ^capsule

## The Methodology

**Inputs.** Three groups of categorical IDs: user profile $\mathcal{P}_u$ (gender, age...), user behaviour $\mathcal{I}_u$ (the items clicked), and the label item $\mathcal{F}_i$ (the item to predict). All are embedded. Profile embeddings are concatenated into $\vec{p}_u$. The label item's own ID plus its side IDs (brand, shop, category — these help cold-start items) are **average-pooled** into $\vec{e}_i$. Behaviour item embeddings form the set $\mathbf{E}_u = \{\vec e_j\}$.

**B2I dynamic routing (the multi-interest extractor).** Behaviour embeddings are the low-level capsules, interests are the high-level capsules. Routing logit:

$$b_{ij} = \vec{u}_j^\top \mathbf{S}\, \vec{e}_i, \qquad i \in \mathcal{I}_u,\ j \in \{1,\dots,K\}$$

$$w_{ij} = \frac{\exp b_{ij}}{\sum_k \exp b_{ik}}, \qquad \vec z_j = \sum_{i \in \mathcal{I}_u} w_{ij}\, \mathbf{S}\, \vec e_i$$

$$\vec u_j = \mathrm{squash}(\vec z_j) = \frac{\lVert \vec z_j\rVert^2}{1+\lVert \vec z_j\rVert^2}\cdot\frac{\vec z_j}{\lVert \vec z_j\rVert}$$

then $b_{ij} \mathrel{+}= \vec u_j^\top \mathbf{S} \vec e_i$, repeat $r=3$ times. Squash keeps direction and compresses length into $[0,1)$ so length reads as a probability.

Three changes from Sabour et al.'s original routing, and each is there for a reason:

1. **One shared bilinear matrix $\mathbf{S} \in \mathbb{R}^{d\times d}$** instead of a separate $\mathbf{S}_{ij}$ per capsule pair. Two reasons: behaviour sequences have variable length (dozens to hundreds), so per-pair matrices do not generalise; and per-pair matrices would map interests into different vector spaces, whereas retrieval needs all $K$ interest vectors and all item vectors living in one space.
2. **Random init of routing logits**, $b_{ij} \sim \mathcal{N}(0,\sigma^2)$. With a shared $\mathbf{S}$, zero-init makes every interest capsule identical at step 1, and identical capsules stay identical forever — a degenerate fixed point. Random init breaks the symmetry, exactly like random k-means centroid seeds.
3. **Per-user interest count**: $K'_u = \max(1, \min(K, \log_2|\mathcal{I}_u|))$. A user with 8 clicks gets 3 interests, not 7. Saves compute and memory.

**Head.** The $K'_u$ interest capsules are concatenated with $\vec p_u$ and pushed through a few ReLU layers, giving $\mathbf{V}_u = (\vec v_u^1,\dots,\vec v_u^K) \in \mathbb{R}^{d\times K}$.

**Label-aware attention (training only).** You have $K$ user vectors but one label item. Which vector should the loss push toward the label? Let the label choose. Scaled-dot-product [[Attention Is All You Need|attention]] with the label as query and the interests as both keys and values:

$$\vec v_u = \mathbf{V}_u\, \mathrm{softmax}\!\left(\mathrm{pow}(\mathbf{V}_u^\top \vec e_i,\, p)\right)$$

$p$ sharpens the distribution. $p=0$ is a uniform average (ignores the label). $p\to\infty$ is hard attention — only the single interest closest to the label gets gradient. They found hard attention converges fastest and works best.

**Loss.** Softmax over the whole item catalogue,

$$\Pr(i\mid u) = \frac{\exp(\vec v_u^\top \vec e_i)}{\sum_{j\in\mathcal{I}}\exp(\vec v_u^\top \vec e_j)}, \qquad L = \sum_{(u,i)\in\mathcal{D}} \log \Pr(i\mid u)$$

The denominator is over billions of items, so they use **sampled softmax** (same trick as [[Deep Neural Networks for YouTube Recommendations (RecSys)|YouTube DNN]]). Optimiser is [[Adam- A Method for Stochastic Optimization|Adam]].

**Serving.** Drop the label-aware attention layer — it needs a label, and at serving there is none. The rest of the network is $f_{user}$. Feed in behaviours + profile, get $K'_u$ vectors, run ANN on each, merge and sort candidates by

$$f_{score}(\mathbf{V}_u, \vec e_i) = \max_{1\le k\le K} \vec e_i^\top \vec v_u^k$$

Max, not sum: an item only needs to match *one* interest well. Whole retrieval pipeline finishes in **under 15 ms**. Trained on **100 GPUs in 8 hours**, model refreshed daily so new products get exposure. Because the user vector is a function of the behaviour sequence, a new click immediately changes the vectors — real-time personalisation without retraining.

## Ablation Studies and Experiments

**Data.** Amazon Books (351K users, 394K items, 6.3M samples, ≥10 interactions filter both sides) and TmallData (2M users, 935K items, 6377 categories, 50.9M samples, 10 days). Split 19:1. Task is next-item prediction; metric is HitRate@N.

**Main table** (relative to YouTube DNN):

| Dataset | Metric | WALS | YouTube DNN | MaxMF-$K$ | MIND-1 | MIND-$K$ |
|---|---|---|---|---|---|---|
| Amazon ($K$=3, $d$=36) | HR@10 | 0.0144 | 0.0231 | 0.0285 | 0.0273 | **0.0309** (+33.8%) |
| | HR@50 | 0.0553 | 0.0746 | 0.0862 | 0.0978 | **0.1101** (+47.6%) |
| | HR@100 | 0.0907 | 0.1143 | 0.1304 | 0.1459 | **0.1631** (+42.7%) |
| Tmall ($K$=5, $d$=64) | HR@10 | 0.0372 | 0.0589 | 0.0628 | 0.0720 | **0.0972** (+65.0%) |
| | HR@50 | 0.0831 | 0.1256 | 0.1820 | 0.1512 | **0.2080** (+65.6%) |
| | HR@100 | 0.1126 | 0.1648 | 0.2567 | 0.1930 | **0.2699** (+63.8%) |

Three things this table separates cleanly:

- **MIND-1-interest vs YouTube DNN** isolates the routing mechanism from the multi-vector idea. With $K=1$ MIND *is* YouTube DNN except that history is pooled by dynamic routing instead of averaging. It still gains +27.7% HR@100 on Amazon. So iterative soft-weighted pooling beats mean pooling even with a single output vector.
- **MIND-$K$ vs MIND-1** isolates the multi-vector idea. Amazon HR@100 0.1459 → 0.1631; Tmall 0.1930 → 0.2699. Much bigger on Tmall, which matches the story: Tmall users are more diverse, and the best $K$ is 5 there versus 3 on Amazon.
- **MIND-$K$ vs MaxMF-$K$** isolates *how* you get multiple vectors. Both use $K$ user vectors; MaxMF gets them by nonlinear latent factorisation. MIND wins everywhere, credited to (a) routing being an actual clustering procedure and (b) label-aware attention routing gradient to the right interest.

WALS (matrix factorisation) loses to everything, 20–37% below YouTube DNN.

**Hyperparameter probes (Amazon).**
- $\sigma$ for routing-logit init tried at 0.1, 1, 5 — the three curves essentially overlap. The model is insensitive to it. They ship $\sigma=1$.
- $p$ in label-aware attention: **$p=0$ is much worse than everything else.** This is the sharpest negative result in the paper. At $p=0$ every interest gets equal weight, the combined vector is a plain average, and the label has no say — you have destroyed the multi-interest structure at training time. Performance rises monotonically with $p$ and peaks at hard attention ($p\to\infty$).

**Online A/B at Tmall homepage, one week**, all arms feeding the same ranker, 1000 candidates each. Baselines: item-based CF (the incumbent serving most traffic) and YouTube DNN. Findings:
- Item-based CF beats YouTube DNN, and even beats MIND with a single interest. Years of tuning on a simple method is a real baseline.
- CTR climbs monotonically as interests go 1 → 5, peaks at 5, and **7 interests gives essentially nothing over 5**. Optimal is 5–7 for Tmall.
- **The dynamic interest number $K'_u = \log_2|\mathcal{I}_u|$ gave no CTR gain** over fixed $K=7$. It is a pure cost optimisation, not a quality one. Honest reporting.

**Interpretability check.** Heatmaps of coupling coefficients $w_{ij}$: for a user who bought headphones, snacks, handbags and clothes, each behaviour class peaks on a distinct interest capsule. For a user who only buys clothes, the routing does not idle — it splits at finer grain into sweaters, overcoats, down jackets. The clustering adapts its granularity to how concentrated the user is. Item-distribution plots show MIND's per-interest recalls are tightly correlated with that interest, while YouTube DNN's single recall list is spread thin across categories with lower similarity to the behaviour history.

## Worth Remembering

- **The real constraint that shapes this design is serving cost, not accuracy.** DIN already solved diversity better in principle. MIND is the version that fits into an ANN index. Whenever you read a matching-stage paper, ask first "can the user vector be computed without the candidate item?" — that question decides the architecture.
- **Hard attention at training, max at serving.** Both say the same thing: an interest is judged by its best match, never by its average. Softening either one hurts.
- **Zero-init symmetry collapse** is a general trap, not a capsule quirk. Shared weights + identical initial states = permanently identical outputs. Same failure family as identical-weight init in an MLP, and the same fix as random k-means seeding.
- The $\max$ scoring function means the $K$ vectors are never combined at retrieval. So $K$ multiplies your ANN query cost by $K$. The $\log_2|\mathcal{I}_u|$ rule exists purely to claw that back for light users.
- **Limitations the authors admit:** no time information in the behaviour sequence at all — order and recency are discarded, which is a big gap versus [[Self-Attentive Sequential Recommendation (SASRec)|SASRec]]-style models. They also flag the routing initialisation as crude and want a k-means++ analogue.
- The offline evaluation is a single random held-out item per user, not a temporal split. Read the numbers with the scepticism from [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|Dacrema et al.]] in mind — and note the online result where a well-tuned item-based CF beat a deep model.
- No ablation on the number of routing iterations $r$; they just take $r=3$ from the capsule paper.
- Open question: MIND's interests are learned per forward pass and are not stable across time or comparable across users. There is no guarantee that "interest 2" means the same thing for two users, or for one user on two days. That makes them hard to debug or use downstream as features.

## Links

Related: [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Deep Interest Network for CTR Prediction (DIN)]] · [[Deep Interest Evolution Network (DIEN)]] · [[Self-Attentive Sequential Recommendation (SASRec)]] · [[Behavior Sequence Transformer for E-commerce (BST)]] · [[PinnerFormer- Sequence Modeling for User Representation at Pinterest]] · [[Search-based User Interest Modeling (SIM)]] · [[Attention Is All You Need]] · [[Efficient and robust approximate nearest neighbor search using HNSW]] · [[Amazon.com Recommendations- Item-to-Item Collaborative Filtering (IEEE Internet Computing)]] · [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)]] · [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)]] · [[Adam- A Method for Stochastic Optimization]] · [[Recommender Systems - Evolution]] · [[Real-time Personalization using Embeddings for Search Ranking at Airbnb (KDD, best paper)]]

New topics worth writing: Capsule networks and dynamic routing (Sabour et al. 2017), Sampled softmax and its bias correction, K-means and soft clustering as a differentiable layer, ComiRec (multi-interest follow-up with self-attentive routing), Hit Rate vs Recall@K in matching-stage evaluation, FAISS billion-scale similarity search
