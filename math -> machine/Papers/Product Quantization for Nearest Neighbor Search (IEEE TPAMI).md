---
title: "Product Quantization for Nearest Neighbor Search (IEEE TPAMI)"
authors: ["Jégou", "Douze & Schmid"]
year: 2011
url: https://inria.hal.science/inria-00514462v2/document
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper]
---
## The Core Idea

You want to search a billion vectors by Euclidean distance, but you cannot hold a billion float vectors in RAM. So compress each vector into a short code — say 64 bits — and compute distances *directly from the codes*, without decompressing.

The obvious way to compress is k-means: learn $k$ centroids, store the index of the nearest one. To get a 64-bit code you need $k = 2^{64}$ centroids. That is impossible — you cannot train it, you cannot store it, there is not enough memory on Earth for $2^{64} \times 128$ floats.

**Product quantization (PQ)** fixes this by chopping the vector into $m$ pieces and running a *separate small* k-means on each piece. With $m = 8$ pieces and $k^* = 256$ centroids per piece, each piece costs 8 bits, the whole code is 64 bits, and the effective codebook size is $256^8 = 2^{64}$. But you only ever store $8 \times 256 = 2048$ small centroids. A giant codebook built as a Cartesian product of tiny ones.

> [!NOTE] Product quantizer
> Split $x \in \mathbb{R}^D$ into $m$ subvectors of dimension $D^* = D/m$. Quantize each subvector with its own k-means codebook of size $k^*$. The full code is the concatenation of $m$ indices; the implied codebook is $\mathcal{C} = \mathcal{C}_1 \times \dots \times \mathcal{C}_m$ with $k = (k^*)^m$ centroids. ^product-quantizer

The second idea, which matters as much: **do not compress the query.** Earlier short-code methods (spectral hashing, Hamming embedding) turn both query and database vector into binary codes and compare with Hamming distance. That throws away information about the query for no reason — the query is a single vector you have in full. Instead, compute the exact distance from the *real* query subvector to each of the 256 centroids of each subquantizer, cache those $m \times k^*$ numbers in a lookup table, and then every database code costs $m$ table lookups plus $m$ additions. Same cost as a Hamming distance, much lower error.

This is the ancestor of FAISS, of every "IVF-PQ" index, and of the compressed vector databases people run today. It sits in the same design space as [[Efficient and robust approximate nearest neighbor search using HNSW|HNSW]] but solves the *memory* problem rather than the *traversal* problem, and the two are routinely combined.

## The Methodology

**Notation.** A quantizer $q$ maps $x \in \mathbb{R}^D$ to a centroid $c_i$ from codebook $\mathcal{C}$ of size $k$. Quality is mean squared error

$$\mathrm{MSE}(q) = \mathbb{E}_X\!\left[d\big(q(x), x\big)^2\right].$$

k-means (Lloyd) gives a local optimum satisfying the two Lloyd conditions: assign to nearest centroid, and set each centroid to the mean of its cell.

**Product quantizer.** Split $x$ into $u_1(x), \dots, u_m(x)$, each of dimension $D^* = D/m$. Train $m$ separate k-means, each with $k^*$ centroids, on the corresponding slices of the training data. Because the subspaces are orthogonal, the errors just add:

$$\mathrm{MSE}(q) = \sum_j \mathrm{MSE}(q_j).$$

Cost comparison (Table I), for total codebook size $k$:

| | codebook memory | assignment cost |
|---|---|---|
| k-means | $kD$ | $kD$ |
| product k-means | $mk^*D^* = k^{1/m}D$ | $k^{1/m}D$ |

For $k=2^{64}$, $D=128$, $m=8$: $2^{64}\cdot128$ floats becomes $2048 \cdot 16 = 32{,}768$ floats.

**Two ways to compute a distance.**

*Symmetric (SDC)* — encode both, compare centroids:
$$\hat d(x,y) = d\big(q(x), q(y)\big) = \sqrt{\sum_j d\big(q_j(x), q_j(y)\big)^2}$$
Each term is read from a precomputed $k^* \times k^*$ table per subquantizer.

*Asymmetric (ADC)* — encode only the database vector:
$$\tilde d(x,y) = d\big(x, q(y)\big) = \sqrt{\sum_j d\big(u_j(x), q_j(u_j(y))\big)^2}$$
Here the tables are built at query time: $m \times k^*$ squared distances from the query's subvectors to every subcentroid. That is $k^*D$ flops, independent of database size $n$. Then scanning the database is $nm$ additions.

Square roots are never taken — ranking by squared distance is identical.

> [!NOTE] Asymmetric distance computation (ADC)
> Compare an *uncompressed* query against *compressed* database codes via a per-query lookup table. Costs the same as symmetric comparison but halves the effective quantization noise, because only one of the two vectors is approximated. ^adc

**Error bound.** By the triangle inequality, $\big(d(x,y) - d(x,q(y))\big)^2 \le d(y,q(y))^2$, so the mean squared *distance* error is bounded by the mean squared *reconstruction* error:
$$\mathrm{MSDE}(q) \le \mathrm{MSE}(q).$$
For SDC the bound is $2\,\mathrm{MSE}(q)$. So minimising quantizer distortion directly controls distance accuracy — a clean justification for just running k-means.

**Bias correction (which they end up not recommending).** $\tilde d$ systematically *underestimates* the true distance. The unbiased version adds the per-cell distortion:
$$\tilde e(x,y) = \tilde d(x,y)^2 + \sum_j \xi_j(y),$$
where $\xi_j(y)$ is the average squared distortion of the cell that $u_j(y)$ landed in, stored in a lookup table. Measured on SIFT: bias drops from $-0.044$ to $0.002$, but variance rises from $0.00146$ to $0.00155$. See "what did not work".

**IVFADC — avoiding the exhaustive scan.** Even at $m$ additions per vector, scanning two billion codes per query is too slow. So put a coarse quantizer $q_c$ in front — plain k-means with $k'$ centroids ($k' = 1024$ to $10^6$) — and build an inverted file: list $L_i$ holds every $y$ with $q_c(y) = c_i$.

The trick: PQ does **not** encode $y$. It encodes the **residual**
$$r(y) = y - q_c(y),$$
so the reconstruction is $\ddot y = q_c(y) + q_p(y - q_c(y))$. Residuals have much less energy than the raw vectors, so the same code length buys more precision. The coarse index is the "most significant bits", the PQ code the "least significant bits".

One single product quantizer is trained on residuals pooled across *all* Voronoi cells. Per-cell quantizers would be better but would need $k' \times D \times k^*$ floats — intractable.

Each inverted-list entry is:

| field | bits |
|---|---|
| identifier | 8–32 |
| PQ code | $m\lceil \log_2 k^*\rceil$ |

*Query.* Assign $x$ to its $w$ nearest coarse centroids (multiple assignment, because $x$ and its true NN often fall in different cells). For each such cell, form the residual $r(x) = x - q_c(x)$, build the $m \times k^*$ lookup table for that residual, scan the list summing $m$ looked-up values, and keep the top $K$ in a fixed-capacity max-heap. Only $\approx n w / k'$ entries get touched. Multiple assignment is applied to queries only, never to database vectors (that would multiply memory).

**Data.** SIFT: 100k learn / 1M database / 10k queries, $D=128$. GIST: 100k learn / 1,000,991 database / 500 queries, $D=960$. Metric is recall@$R$ — fraction of queries whose true nearest neighbour appears in the top $R$.

## Ablation Studies and Experiments

**How to spend your bits: few big subquantizers beat many small ones.** For a fixed code length $l = m\log_2 k^*$, distortion is lower with small $m$ and large $k^*$. At the extreme $m=1$ this is just plain k-means (best distortion, but $k^*$ cannot exceed ~$2^{16}$ before the centroid table blows out of cache). $k^* = 256$, $m = 8$ is the recommended sweet spot — and 8 bits per subcode is byte-aligned, which is why every PQ implementation since uses it.

Interestingly, MSE *underestimates* how well large-$m$ configurations actually retrieve. Ranking quality and reconstruction quality are not the same thing.

**ADC beats SDC, for free.** On SIFT with $m=8$, ADC at $k^*=64$ (48-bit codes) matches SDC at $k^*=256$ (64-bit codes). Same query-time cost. SDC's only advantage is that the query itself can be stored as a code — rarely useful.

**64-bit codes, SIFT, recall@R (Fig. 8):** IVFADC > ADC > SDC > HE ≈ spectral hashing. To hit the same recall as spectral hashing, ADC returns roughly an order of magnitude fewer candidates. On GIST the gap is even larger.

**IVFADC parameters.** Longer codes are wasted if $w$ is too small — a true neighbour that lands outside the $w$ scanned cells is lost forever, and no amount of code precision recovers it. For a fixed scan fraction $\approx w/k'$, larger $k'$ is better: $(k'{=}8192, w{=}8)$ beats $(k'{=}1024, w{=}1)$; $(8192, 64)$ beats $(1024, 8)$.

**Timings, GIST 1M, 64-bit codes, single core (Table V):**

| method | time (ms) | codes compared | recall@100 |
|---|---|---|---|
| SDC | 16.8 | 1,000,991 | 0.446 |
| ADC | 17.2 | 1,000,991 | 0.652 |
| Spectral hashing | 22.7 | 1,000,991 | 0.132 |
| IVFADC $k'{=}1024,w{=}1$ | 1.5 | 1,947 | 0.308 |
| IVFADC $k'{=}1024,w{=}8$ | 8.8 | 27,818 | 0.682 |
| IVFADC $k'{=}1024,w{=}64$ | 65.9 | 101,158 | 0.744 |
| IVFADC $k'{=}8192,w{=}8$ | 10.2 | 2,709 | 0.516 |

Note IVFADC with $w=8$ beats full exhaustive ADC on recall *and* is twice as fast — because encoding residuals is more accurate than encoding raw vectors.

**Which dimensions go in which group matters a lot (Table IV, recall@100, $k^*{=}256$).**

| | SIFT $m{=}4$ | SIFT $m{=}8$ | GIST $m{=}8$ |
|---|---|---|---|
| natural order | 0.593 | 0.921 | 0.338 |
| random order | 0.501 | 0.859 | 0.286 |
| "structured" | 0.640 | 0.905 | **0.652** |

Random permutation of dimensions before splitting *hurts everywhere*. For GIST, grouping dimensions by index mod 8 (so bins of the same gradient orientation get quantized together) nearly doubles recall, 0.338 → 0.652. For SIFT the natural order (spatially adjacent patch cells together) is already near-best. **Lesson: the split must respect the structure of the descriptor.** This is a genuine limitation — you need prior knowledge, and for something like bag-of-features embeddings you have none. They suggest minimum sum-squared residue co-clustering as an automatic fix but do not run it.

**vs FLANN (Fig. 10).** With a re-ranking stage (IVFADC returns a shortlist of $R$, then exact L2 rescoring), IVFADC dominates FLANN across most of the 1-recall@1 vs time curve. Index memory: **under 25 MB vs FLANN's 250+ MB**. Caveat the authors state plainly: re-ranking requires the full vectors in RAM, so this comparison is not the regime PQ is built for.

**Two billion SIFT vectors (Fig. 11).** IVFADC vs HE, same 20,000-word coarse codebook, $w=1$, 64 bits. IVFADC's extra quantization step shows as overhead at 10M vectors, but by 1G the list-scanning dominates and the two are near-identical in time — 8 table lookups either way. Floating-point PQ lookups are not meaningfully slower than binary Hamming XOR-popcount.

**Image retrieval, Holidays + up to 1M distractors (Fig. 12).** At 1M distractors and 64 bits, mAP goes from 0.497 (HE) to 0.517 (IVFADC).

### What did not work

- **Bias correction.** The unbiased estimator $\tilde e$ removes the bias but raises variance, and worse: the correction term $\sum_j \xi_j(y)$ tends to be large exactly for the nearest neighbours, so it *penalises vectors sitting in rare, high-distortion cells*. Retrieval got worse on average. Use it only if you need the distance value itself (for $\varepsilon$-radius search or Lowe's ratio test), not for ranking.
- **Random rotation before splitting.** Theory says multiplying by a random orthogonal matrix equalises energy across subvectors, which should help when $k^*$ is fixed. In practice it destroys the correlation between adjacent components and hurts (random order row above). *This is exactly what OPQ later fixed by learning the rotation instead of randomising it.*
- **Per-cell product quantizers.** Would be more accurate than one shared PQ on pooled residuals, but $k'$ codebooks is memory-intractable. Never attempted.
- **Lattice and scalar quantizers.** Dismissed: lattices are only good for uniform distributions, and real descriptors are not uniform. Cited evidence that they do significantly worse than k-means for indexing.
- **Hierarchical k-means (HKM).** Fixes learning cost but not the fundamental memory ceiling on the number of centroids.

## Worth Remembering

- The whole method is Euclidean-only. Spectral hashing and RBM codes can be trained for arbitrary metrics; PQ cannot. The authors acknowledge this explicitly.
- The $m$ additions per candidate are the entire inner loop. Modern implementations (FAISS's "fast scan") turn this into SIMD byte shuffles, which is why $k^*=256$ persisted — a 256-entry table fits an AVX register pair.
- The bound $\mathrm{MSDE}(q) \le \mathrm{MSE}(q)$ is the one line to keep. It means "train the quantizer to minimise reconstruction error" is provably the right proxy for "get distances right". No task-specific loss needed.
- The residual trick is doing real work, not just bookkeeping: IVFADC with $w{=}8$ beats exhaustive ADC on recall (0.682 vs 0.652) *while scanning 3% of the database*.
- Per-vector estimated quantization error can also be used to decide *which* candidates deserve exact re-ranking, instead of just taking a fixed top-$R$. Nobody seems to do this.
- Memory arithmetic worth internalising: 1B 128-d float SIFT vectors = 512 GB. At 64-bit PQ codes plus a 32-bit ID = 12 bytes = 12 GB. That factor of ~43 is the whole point.
- The identifier field is pure inverted-file overhead. For image search you can store an *image* ID rather than a descriptor ID (20 bits for a million images), and delta+Huffman coding of sorted IDs gets you to ~11 bits.
- The dimension-grouping ablation is the weakest joint. It is a hand-tuned, dataset-specific choice that changes recall by 2×. Optimized Product Quantization (OPQ, 2013) learns the rotation matrix jointly with the codebooks and removes this hand-tuning entirely — read it right after this.
- Open question: PQ on modern learned embeddings (BERT/CLIP outputs) where dimensions have no interpretable structure. Given [[How Contextual are Contextualized Word Representations|anisotropy]] and [[Understanding Dimensional Collapse in Contrastive Learning|dimensional collapse]] in such spaces, energy is very unevenly spread across directions, which is precisely the situation a fixed uniform split handles badly.

## Links

Related: [[Efficient and robust approximate nearest neighbor search using HNSW]] · [[Dense Passage Retrieval (DPR)]] · [[ColBERT- Efficient and Effective Passage Search via Late Interaction]] · [[Embedding-based Retrieval in Facebook Search]] · [[Matryoshka Representation Learning]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[Whitening Sentence Representations]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Mixed Precision training]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Sentence-BERT]]

New topics worth writing: Optimized Product Quantization (OPQ), FAISS index taxonomy (IVF, PQ, HNSW-PQ, IVF-PQ), Locality-Sensitive Hashing (E2LSH), Spectral Hashing, Lloyd's algorithm and k-means convergence, rate-distortion theory and vector quantization, lattice quantizers and the Leech lattice, SIFT and GIST descriptors, inverted file structures and posting-list compression, residual vector quantization (RVQ), scalar quantization for embeddings (int8/binary)
