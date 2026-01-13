---
title: "Recommender Systems with Generative Retrieval (TIGER)"
authors: ["Shashank Rajput", "Nikhil Mehta", "Anima Singh", "Raghunandan H. Keshavan", "Trung Vu", "Lukasz Heldt", "Lichan Hong", "Yi Tay", "Vinh Q. Tran", "Jonah Samost", "Maciej Kula", "Ed H. Chi", "Maheswaran Sathiamoorthy"]
year: 2023
arxiv: "2305.05065"
url: https://arxiv.org/abs/2305.05065
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, transformers]
---
## The Core Idea

Retrieval in a recommender usually works like this: turn the user into a vector, turn every item into a vector, then use approximate nearest-neighbour search to find the closest items. You need an index holding one vector per item. TIGER throws that away.

Instead, every item gets a short **code** — a tuple of 4 small integers like $(7, 1, 4, 0)$ — and a [[Attention Is All You Need|Transformer]] encoder-decoder is trained to *generate* the next item's code, one integer at a time, like translating a sentence. No index. No per-item embedding. The model's own weights are the index.

The trick that makes it work is that the codes are not random. They come from squeezing the item's text description (title, price, brand, category) through a pre-trained sentence encoder and then quantising that vector in a coarse-to-fine way. So the first integer means something like "hair products", the second narrows it to "hair styling", the third narrows further. Two similar items share a prefix.

> [!NOTE] Semantic ID
> A short tuple of discrete codewords derived from an item's *content*, arranged so that the first codeword is coarse and later ones are fine. Semantically close items overlap in their leading codewords. ^semantic-id

Why this did not exist before: prior generative retrieval work (DSI for documents) used hierarchical k-means or raw string IDs, and prior recommender work used **atomic random IDs** — item 4,271,882 is just an arbitrary row in a giant embedding table, carrying zero information. With a random ID, a brand-new item is a fresh, untrained row; the model literally cannot say anything about it. With a Semantic ID, a new item inherits the meaning of its codewords from every similar item the model already saw.

What it unlocks:
1. **Cold start for free.** A never-seen item still gets a Semantic ID from its text, and the model can emit that ID.
2. **Memory that does not grow with the catalogue.** You store $1024$ codeword embeddings ($256 \times 4$), not $N$ item embeddings. The number of representable items is the *product* of codebook sizes — $256^4 \approx 4$ trillion.
3. **Tunable diversity.** Sample the decoder with temperature. Raising temperature on the *first* codeword jumps categories; on the third, it stays inside a category.

## The Methodology

Two stages, trained separately.

### Stage 1 — Making Semantic IDs with RQ-VAE

Take an item's text features, glue them into a sentence, push through frozen **Sentence-T5** to get $\bm{x} \in \mathbb{R}^{768}$.

Then train a **Residual-Quantised VAE**. An MLP encoder $\mathcal{E}$ (layers $512 \to 256 \to 128$, ReLU) maps $\bm{x}$ to a latent $\bm{z} \in \mathbb{R}^{32}$. Now quantise it in levels:

- $\bm{r}_0 := \bm{z}$.
- At level $d$ there is a codebook $\mathcal{C}_d = \{\bm{e}_k\}_{k=1}^{K}$ with $K = 256$ vectors of dimension 32.
- Pick the nearest codebook vector: $c_d = \arg\min_k \|\bm{r}_d - \bm{e}_k\|$.
- Subtract it and pass the leftover down: $\bm{r}_{d+1} := \bm{r}_d - \bm{e}_{c_d}$.

Repeat $m = 3$ times. The three indices $(c_0, c_1, c_2)$ are the Semantic ID. The reconstruction is $\widehat{\bm{z}} := \sum_{d=0}^{m-1} \bm{e}_{c_d}$, decoded back to $\widehat{\bm{x}}$.

The loss is

$$\mathcal{L}(\bm{x}) = \underbrace{\|\bm{x} - \widehat{\bm{x}}\|^2}_{\mathcal{L}_{\text{recon}}} + \underbrace{\sum_{d=0}^{m-1}\Big(\|\text{sg}[\bm{r}_d] - \bm{e}_{c_d}\|^2 + \beta\,\|\bm{r}_d - \text{sg}[\bm{e}_{c_d}]\|^2\Big)}_{\mathcal{L}_{\text{rqvae}}}$$

with $\beta = 0.25$ and $\text{sg}$ the stop-gradient. The first term inside the sum drags the codebook vector toward the residual; the second (the "commitment" term) drags the encoder toward the codebook. This is the standard VQ-VAE machinery, applied to residuals.

Two details that matter:
- **Separate codebook per level**, not one big $mK$ codebook. Residual norms shrink as you go deeper, so each level needs its own scale.
- **k-means init** on the first training batch, otherwise the codebook collapses — most inputs snap onto a handful of vectors. They train 20k epochs to get $\geq 80\%$ codebook usage. Optimiser: Adagrad, lr $0.4$, batch 1024.

**Collisions.** Two items can land on the same $(c_0,c_1,c_2)$. Fix: append a fourth integer that just counts duplicates. $(12,24,52)$ becomes $(12,24,52,0)$ and $(12,24,52,1)$. Everything gets a 4th codeword, even singletons (which get 0). A lookup table (item ID ↔ Semantic ID) is built once and frozen — about $64N$ bits, trivial next to embedding tables.

### Stage 2 — The sequence-to-sequence recommender

Sort each user's interactions by time. Flatten the whole history into one long token stream:

$$(c_{1,0},\dots,c_{1,3},\ c_{2,0},\dots,c_{2,3},\ \dots,\ c_{n,0},\dots,c_{n,3})$$

Prepend a **user token**. Raw user IDs are hashed into 2000 buckets with the hashing trick — enough to personalise, small enough to fit in the vocabulary. Total vocabulary: $256 \times 4 = 1024$ item codeword tokens plus 2000 user tokens.

Train a T5X encoder-decoder to emit the 4 codewords of item $n+1$. Standard next-token [[Cross Entropy|cross-entropy]].

Hyperparameters: 4 encoder + 4 decoder layers, 6 attention heads of dim 64, MLP dim 1024, model dim 128, ReLU, dropout 0.1. **~13M parameters total.** 200k steps for Beauty and Sports, 100k for Toys (smaller). Batch 256. lr $0.01$ for 10k steps then inverse-square-root decay. History truncated to 20 items.

At inference: **beam search** over the 4 decoding positions, then map each generated tuple back to an item through the lookup table.

## Ablation Studies and Experiments

Three Amazon Product Reviews categories (Beauty, Sports & Outdoors, Toys & Games), 12k–18k items, 19k–36k users, mean sequence length ~8.6. Leave-one-out: last item test, second-to-last validation. Metrics: Recall@K and [[NDCG]]@K for $K=5,10$.

### Main result (Table 1)

Against [[Session-based Recommendations with RNNs (GRU4Rec)|GRU4Rec]], Caser, HGN, [[Self-Attentive Sequential Recommendation (SASRec)|SASRec]], [[BERT4Rec- Sequential Recommendation with Bidirectional Transformer|BERT4Rec]], FDSA, S³-Rec, and P5:

| Dataset | Best baseline NDCG@5 | TIGER NDCG@5 | Gain |
|---|---|---|---|
| Beauty | 0.0249 (SASRec) | **0.0321** | +29.0% |
| Toys & Games | 0.0306 (SASRec) | **0.0371** | +21.2% |
| Sports & Outdoors | 0.0161 (S³-Rec) | **0.0181** | +12.6% |

Recall@10 gains are much smaller (+0.15% on Beauty, +1.7% on Toys). So TIGER's win is mostly about **ordering the top of the list correctly**, not about finding more of the right items deeper down. Worth internalising: it is an NDCG win more than a Recall win.

Seed variance (3 runs) is tight: Beauty NDCG@5 $= 0.0309 \pm 0.00062$. Note this mean is slightly below the headline 0.0321 in Table 1.

### The ablation that carries the paper (Table 2)

Same architecture, same everything, only the item IDs change. Beauty NDCG@5:

| ID scheme | Beauty NDCG@5 | Beauty Recall@5 |
|---|---|---|
| Random ID (4 codewords, uniform from 255) | 0.0205 | 0.0296 |
| LSH Semantic ID ($h=8$ hyperplanes, $m=4$) | 0.0259 | 0.0379 |
| RQ-VAE Semantic ID | **0.0321** | **0.0454** |

This is the whole story. Random IDs already beat most baselines on Beauty and Toys — the generative seq2seq framing itself is worth something. But **semantics roughly halves the remaining gap**, and *learned* quantisation (RQ-VAE) beats *random-projection* quantisation ([[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)|LSH/SimHash]]) by another clear margin. So: framing helps, semantics helps more, and a non-linear learned quantiser helps more still.

### Hierarchy is real, and visible

With a toy configuration (codebook sizes 4, 16, 256), $c_1$ separates coarse categories — $c_1 = 3$ is nearly all "Hair", $c_1 = 1$ is Makeup/Skin. Fixing $c_1$ and sweeping $c_2$ splits into fine-grained sub-categories. This is not a claim, it is a plotted category distribution.

### Cold start

Remove 5% of test items from training entirely ("unseen"). Generate their Semantic IDs after RQ-VAE training. At decode time, a predicted $(c_1,c_2,c_3,c_4)$ retrieves the matching seen item, *plus* any unseen item sharing the first three codewords. A knob $\epsilon$ caps the fraction of unseen items allowed in the top-K. Baseline is Semantic_KNN (nearest neighbour in the Sentence-T5 space). TIGER beats it at every $K$ with $\epsilon = 0.1$, and beats it for all $\epsilon \geq 0.1$.

### Diversity via temperature

Entropy of the ground-truth category distribution over the top-10 predictions, Beauty:

| Temperature | Entropy@10 | Entropy@50 |
|---|---|---|
| 1.0 | 0.76 | 1.70 |
| 1.5 | 1.14 | 2.06 |
| 2.0 | 1.38 | 2.28 |

At $T=1.0$ a "Hair Styling Products" query returns only hair styling. At $T=2.0$ it also pulls hair tools and skin-face items.

### What did *not* work, or barely moved

- **VQ-VAE instead of RQ-VAE**: retrieval quality was *similar*, but you lose the coarse-to-fine hierarchy, and with it the temperature-diversity control and clean prefix matching. The hierarchy is the point, not the accuracy.
- **Hierarchical k-means** (the DSI approach): rejected because semantic meaning is not preserved *between* clusters.
- **Depth barely matters.** 3 / 4 / 5 layers give Beauty NDCG@5 of 0.03062 / 0.0321 / 0.03206. Noise-level. This is not a scaling result.
- **User ID barely matters.** Without user token: NDCG@5 $= 0.0302$. With: $0.0321$. A ~6% relative bump. The sequence carries almost all the signal.
- **Code shape barely matters.** 6 codewords of cardinality 64 performs about the same as 4 of 256. Longer IDs just lengthen the input sequence and cost more compute.
- **P5 baseline had a data leak.** P5's preprocessing renumbers items $1, 2, 3, \dots$ *before* the train/test split, so many sequences look like $a, a{+}1, a{+}2, \dots$ — and P5's SentencePiece tokenizer shares subwords between the train prefix and the held-out item. The authors reassigned random integer IDs and re-ran. P5's numbers in Table 1 are the fixed version; their own reimplementation ("P5-ours", Table 7) does better (Beauty NDCG@5 0.025 vs 0.0107) but still loses badly to TIGER. Good example of how ID assignment can silently leak.

### Invalid IDs

The decoder can emit a tuple that maps to no item. Space is $256^4 \approx 4$ trillion; only 10k–20k are valid. Yet invalid rate is **0.1%–1.6% for top-10** and 0.3%–6% for top-20. Fix in practice: widen the beam and filter. Proposed but not implemented: prefix matching, i.e. if the full tuple is invalid, fall back to items sharing $(c_1, c_2, c_3)$ — which is literally "the model predicted a category, not an item."

## Worth Remembering

**The honest limitation the authors state: inference is more expensive.** Beam search over 4 autoregressive steps versus one [[Efficient and robust approximate nearest neighbor search using HNSW|ANN]] lookup. They explicitly say efficiency was not their goal. If you deploy this at web scale, this is your first problem.

**Memory is the real win, and it is big.** Traditional systems store $Nd$ floats for item embeddings. TIGER stores $1024d$. At $N$ in the billions — which is the regime that motivates the whole paper — that ratio is enormous. Compare [[Compositional Embeddings Using Complementary Partitions (QR trick)|the QR trick]] and [[Mixed Dimension Embeddings]], which attack the same embedding-table problem with hashing rather than semantics. TIGER's argument is that semantically meaningful compression beats random compression, and Table 2 (LSH vs RQ-VAE) is the direct evidence.

**Scalability probe (Table 10):** train the RQ-VAE on all three datasets combined, then use those Semantic IDs on Beauty alone. NDCG@5 drops 0.0321 → 0.0368... actually Recall@5 drops 0.0454 → 0.04355. Small degradation. One data point, not a scaling law, but reassuring.

**Datasets are tiny.** 12k–18k items, mean sequence length under 9. These are the standard academic Amazon benchmarks and they are known to be shaky — see [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] and [[On the Difficulty of Evaluating Baselines]]. Most baseline numbers here are copied from S³-Rec's public results, not re-tuned. The RQ-VAE runs for 20k epochs on 12k items, which is a lot of training on a small corpus.

**Connections.** RQ-VAE's loss is VQ-VAE's loss applied recursively to residuals — the same [[Auto-Encoding Variational Bayes (VAE)|autoencoder]] skeleton with a discrete bottleneck. The generative-retrieval framing is DSI (Transformer Memory as a Differentiable Search Index) transplanted from documents to items. The content encoder is [[Sentence-BERT|a sentence bi-encoder]] (Sentence-T5) used frozen. The hierarchical prefix structure has the same flavour as coarse-to-fine [[Matryoshka Representation Learning|Matryoshka embeddings]], though the mechanism differs completely. The follow-up by Singh et al. (arXiv 2306.08121) shows Semantic IDs also help *ranking*, not just retrieval.

**Open questions worth chasing.**
- Can the RQ-VAE and the seq2seq model be trained jointly rather than in two frozen stages? The codes are currently optimised for reconstruction, not for recommendation.
- Prefix-matching fallback for invalid IDs is left as future work and would likely lift Recall further.
- The user token is nearly useless here (2000 hash buckets, +6% NDCG). Real personalisation almost certainly needs something richer.
- Beam search over 4 steps to retrieve top-K — how does the invalid-ID rate scale when $N$ goes from $10^4$ to $10^9$ and the valid set gets denser?

## Links

Related: [[Self-Attentive Sequential Recommendation (SASRec)]] · [[BERT4Rec- Sequential Recommendation with Bidirectional Transformer]] · [[Session-based Recommendations with RNNs (GRU4Rec)]] · [[Attention Is All You Need]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)]] · [[Efficient and robust approximate nearest neighbor search using HNSW]] · [[Compositional Embeddings Using Complementary Partitions (QR trick)]] · [[Mixed Dimension Embeddings]] · [[Sentence-BERT]] · [[NDCG]] · [[Exploring the Limits of Transfer Learning (T5)]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Actions Speak Louder than Words- Generative Recommenders (HSTU)]] · [[Recommender Systems - Evolution]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Cross Entropy]] · [[Matryoshka Representation Learning]]

New topics worth writing: RQ-VAE (residual quantisation), VQ-VAE and the straight-through/commitment loss, codebook collapse, Differentiable Search Index (DSI), generative retrieval for documents, Locality Sensitive Hashing / SimHash, beam search decoding, the hashing trick, Sentence-T5, cold-start recommendation, recommendation diversity metrics
