---
title: "Efficient and robust approximate nearest neighbor search using HNSW"
authors: ["Malkov & Yashunin"]
year: 2016
arxiv: "1603.09320"
url: https://arxiv.org/abs/1603.09320
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper]
---
## The Core Idea

You have 100 million vectors. Someone hands you a query vector and wants the 10 closest ones. Comparing against all 100 million is too slow. You need an index.

Before this paper the two families were **trees** (k-d trees, VP-trees, ball trees) and **hashing** (LSH). Both degrade badly as dimension grows — past roughly 20 dimensions a k-d tree ends up visiting most of the data anyway. A third family, **proximity graphs**, worked better: build a graph where each vector is a node linked to some of its near neighbours, then answer a query by *greedy walking* — jump to whichever neighbour is closer to the query, repeat until no neighbour improves. The problem was getting *to* the right region of the graph in the first place. Earlier graph methods (NSW) did this by starting from a random node and relying on a few accidental long-range links, which gave a walk length that scaled polylogarithmically at best, and they often needed a separate coarse index bolted on top.

The trick here is to **separate links by length scale into layers**, exactly like a skip list separates pointers by stride. Every vector gets a random maximum layer $l$ drawn from an exponentially decaying distribution. Layer 0 holds every vector and has short, local links. Layer 1 holds a small random sample, so *its* links are long-range by construction. Layer 2 holds a sample of that, so its links are longer still. Search starts at the single top-layer entry point, greedily walks that sparse graph until it is stuck, drops to the next layer down using the found node as the new entry point, and repeats. Each layer covers distance at one characteristic scale.

Because the number of nodes shrinks by a constant factor per layer, and because the expected number of hops inside any one layer is a constant (not growing with $N$), the total work is $O(\log N)$ distance computations. That is the headline result: a fully graph-based index, no auxiliary structure, logarithmic scaling, and in practice several times faster than every open-source baseline at the same recall.

The second contribution is subtler and matters more on real data: a **heuristic for choosing which neighbours to keep**. Naively keeping the $M$ nearest ones breaks on clustered data — a tight cluster's members all link only to each other, the graph splits into disconnected islands, and the greedy walk can never leave. The heuristic deliberately keeps some *farther* links that bridge to other regions.

> [!NOTE] Proximity graph search
> An index where data points are nodes and edges connect nearby points. A query is answered by greedy descent: from the current node, move to whichever neighbour is closest to the query; stop when nothing improves. Correctness is approximate — you can get stuck in a local minimum. ^proximity-graph

## The Methodology

**Layer assignment.** When inserting element $q$, draw its top layer as

$$l = \lfloor -\ln(\text{unif}(0,1)) \cdot m_L \rfloor$$

so $l$ is geometric with mean $m_L$. The paper shows the optimal choice is

$$m_L = \frac{1}{\ln M}$$

which makes the expected overlap between consecutive layers exactly one link's worth of information — the same reasoning as a skip list with fan-out $M$. An element appearing at layer $l$ is present in *all* layers $0 \ldots l$.

**Insertion (Algorithm 1).**
1. From the current global entry point, greedy-search down through layers $L_{\max} \ldots l+1$ with `ef = 1` — a plain one-node greedy walk, just to get close.
2. From layer $l$ down to 0: run the beam search below with `ef = efConstruction`, take the returned candidates, pick $M$ of them as neighbours via the selection heuristic, and add **bidirectional** links.
3. After adding a back-link, if a node's degree exceeds $M_{\max}$, re-run the selection heuristic on its neighbour list and prune. Layer 0 uses $M_{\max 0} = 2M$; higher layers use $M_{\max} = M$.
4. If $l > L_{\max}$, this element becomes the new global entry point.

**Search inside one layer (Algorithm 2)** is a best-first beam search. Keep a dynamic candidate list of size `ef`, a visited set, and a result heap. Pop the closest unexplored candidate, expand its neighbours, insert any that beat the current worst result. Stop when the closest candidate is farther than the worst result. `ef = 1` is pure greedy; larger `ef` trades speed for recall. At query time this single knob, `efSearch`, sweeps the whole speed/recall curve — you do **not** rebuild the index to move along it.

**The neighbour selection heuristic (Algorithm 4).** Given a candidate set sorted by distance to the new element $q$, walk it nearest-first and accept candidate $e$ only if

$$d(e, q) < d(e, r) \quad \text{for every } r \text{ already accepted.}$$

In words: keep $e$ only if $q$ is closer to $e$ than any already-kept neighbour is. This means you never keep two neighbours that sit in the same direction — the second one is "shadowed" by the first, and you can reach it by hopping through the first anyway. The effect is to approximate the relative-neighbourhood / Delaunay-like structure, and crucially it produces links that **cross between clusters**, keeping the graph connected. Two optional flags let you top up with rejected candidates if you did not fill $M$ slots.

**Parameters that matter.**
- $M$ — links per node, useful range 5–48. Small $M$ is better for low-dimensional or low-recall work; large $M$ for high-dimensional, high-recall work. Memory is roughly $M \times 8{-}10$ bytes per element on top of the vectors.
- $efConstruction$ — build-time beam width. Raise it until recall on a held-out sample stops improving; that is the saturation point.
- $M_{\max 0} = 2M$ — the bottom layer gets twice the degree budget, which measurably helps.

**Complexity.** Search is $O(\log N)$ distance evaluations; construction is $O(N \log N)$ and is trivially parallel (insertions are independent apart from a lock on the entry point). Memory for the hierarchy above layer 0 is negligible — the expected fraction of elements in layer $\ge 1$ is $1/M$-ish.

> [!NOTE] Length-scale separation
> The reason the hierarchy works: an edge in a graph built over a $1/M^k$ random sample of the data is on average $M^{k/d}$ times longer than an edge in the full graph. Searching top-down means you always take the longest useful step first, so the number of steps per layer stays constant instead of growing with $N$. ^length-scale-separation

## Ablation Studies and Experiments

**Baselines.** Compared against FLANN (randomised k-d forest / k-means tree), Annoy (random projection trees), FALCONN (multi-probe LSH), VP-tree, NAPP, and the flat NSW predecessor, on SIFT1M (128-d), GIST1M (960-d), GloVe (100-d), CoPhIR, MNIST, DEEP1B subsets, and synthetic random vectors of varying dimension. Metric is recall vs. queries per second.

**Headline.** HNSW is typically **several times faster** at the same recall than the best of the baselines, and the gap widens at high recall (0.95+) where tree and LSH methods collapse. On a 200M-point DEEP1B subset it reached ~0.95 recall@10 in a few milliseconds per query on a single core, with the index fitting in RAM. The scaling experiments confirm the $\log N$ prediction: query time grows linearly with $\log N$ across two orders of magnitude of dataset size, while flat NSW grows faster.

**What the hierarchy actually buys — and where it does not.** The most honest ablation in the paper: comparing HNSW against the same algorithm with the hierarchy removed (i.e. flat NSW-style search on layer 0 only). The speedup from the hierarchy is **large in low dimensions and at low recall**, and it **shrinks towards nothing in high dimensions at high recall**. Reason: at high `ef` in high-dimensional space, the beam search is already exploring a wide front and the initial descent is a small fraction of total work. So if your use case is 768-d embeddings at 0.99 recall, most of the win is coming from the graph and the heuristic, not the layers. The layers are still worth keeping because they cost almost nothing in memory.

**What did not work — naive neighbour selection.** On synthetic clustered data (random isolated clusters), keeping simply the $M$ nearest candidates as neighbours causes the graph to fragment. Recall saturates at a low ceiling no matter how large you make `ef`, because the greedy walk is trapped inside one cluster's component. The heuristic fixes this completely. On uniformly distributed data the two selection rules are nearly identical at low recall; the heuristic only pulls ahead in the high-recall regime and on clustered data. This is the component doing the real robustness work.

**Other findings.**
- $M_{\max 0} = M$ (instead of $2M$) is measurably worse.
- Raising `efConstruction` past saturation only wastes build time; it does not hurt query quality.
- The structure works for arbitrary metric spaces, not just $L_2$ — it was tested with cosine and non-metric distances too, unlike LSH which needs a hash family per metric.
- Adding elements is cheap and incremental. Deletion is not supported natively — you can only mark elements as deleted and filter them from results.

## Worth Remembering

- **No deletes, no cheap updates.** The graph is append-only. Production systems (FAISS, Weaviate, Qdrant) handle deletion by tombstoning and periodic full rebuilds. Plan for this.
- **Memory is the real cost.** HNSW stores the full vectors plus $\sim M$ 4-byte links per node per layer. For a billion float32 768-d vectors that is terabytes. This is why production stacks combine HNSW with product quantization (IVF-PQ + HNSW coarse quantizer) — the paper does not compress anything.
- **One knob at query time.** `efSearch` gives a continuous speed/recall dial on a *fixed* index. Tree and LSH methods usually require rebuilding to move along their curve. This is a huge operational advantage and a big part of why HNSW won in practice.
- **Distributed version is easy** because it mirrors a skip list: the top layers are tiny and can be replicated, layer 0 sharded.
- **Connection to the embedding papers in your vault.** Every dense-retrieval system — [[Dense Passage Retrieval (DPR)]], [[Approximate Nearest Neighbor Negative Contrastive Learning (ANCE)]], [[Embedding-based Retrieval in Facebook Search]] — assumes an ANN index underneath, and this is usually it. ANCE in particular *refreshes* an HNSW/FAISS index mid-training to mine hard negatives, so index build time becomes part of the training loop cost.
- **Anisotropic embeddings hurt ANN.** If your vectors all cluster into a narrow cone (the problem in [[How Contextual are Contextualized Word Representations]] and [[Representation Degeneration Problem in Training NLMs]]), distances compress and the greedy walk gets less informative signal per hop. Whitening or normalising before indexing is a real practical lever.
- **Open question the paper leaves.** There is no theoretical guarantee on recall — the whole thing is empirical plus a heuristic argument for $O(\log N)$. Worst-case adversarial data can still trap the greedy walk.

## Links

Related: [[Approximate Nearest Neighbor Negative Contrastive Learning (ANCE)]] · [[Dense Passage Retrieval (DPR)]] · [[ColBERT- Efficient and Effective Passage Search via Late Interaction]] · [[Embedding-based Retrieval in Facebook Search]] · [[Sentence-BERT]] · [[Matryoshka Representation Learning]] · [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Real-time Personalization using Embeddings for Search Ranking at Airbnb (KDD, best paper)]] · [[Recommender Systems - Evolution]] · [[Whitening Sentence Representations]] · [[How Contextual are Contextualized Word Representations]]

New topics worth writing: Skip lists, Navigable small-world networks (Kleinberg), Delaunay and relative neighbourhood graphs, Product quantization and IVF-PQ, Locality-sensitive hashing, Curse of dimensionality in metric search, FAISS, Recall@k for retrieval evaluation, k-d trees and random projection forests
