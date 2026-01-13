---
title: "Neural Collaborative Filtering vs. Matrix Factorization Revisited"
authors: ["Steffen Rendle", "Walid Krichene", "Li Zhang", "John Anderson"]
year: 2020
arxiv: "2005.09683"
url: https://arxiv.org/abs/2005.09683
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, vision]
---
## The Core Idea

Recommender models turn a user into a vector $\mathbf{p}$ and an item into a vector $\mathbf{q}$, then squash the pair into one number: "how much does this user like this item?" For decades that squashing function was the dot product $\langle \mathbf{p}, \mathbf{q}\rangle$ — that is [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)|matrix factorization]].

The 2017 Neural Collaborative Filtering (NCF) paper said: why hand-pick the dot product? Glue the two vectors together into one long vector $[\mathbf{p}, \mathbf{q}]$ and feed it to a small [[Deep Learning#^mlp|MLP]]. An MLP is a *universal approximator* — with enough hidden units it can express any continuous function, including the dot product. So it should be at least as good. NCF became the default baseline in hundreds of follow-up papers.

This paper says: no. Two claims, both concrete.

1. **The comparison was rigged by a weak baseline.** Tune the plain dot product properly — same data splits, same loss, same negative sampling, same evaluation code as NCF — and it beats the MLP on almost every dataset, metric, and embedding size. On Movielens, HR@10 goes $0.7093 \to 0.7294$; NDCG@10 $0.4349 \to 0.4523$. On Pinterest, $0.8777 \to 0.8895$ and $0.5576 \to 0.5794$.

2. **"Universal approximator" is a statement about *representing*, not *learning*.** In a clean synthetic experiment where the true label literally *is* a dot product, an MLP trained directly on that target still cannot fit it accurately once $d$ gets past a few dozen. Capability in the limit is not the same as learnability from finite data.

> [!NOTE] Inductive bias beats expressiveness
> A bigger function class contains the answer but also contains a billion wrong answers. The dot product is a *structural prior*: it hard-codes multiplicative interaction between matched coordinates. An MLP has to discover that structure from data, and discovering it is expensive. Same reason convolutions beat MLPs on images. ^inductive-bias-over-capacity

What this unlocks practically: dot products support **maximum inner product search**, so you can retrieve the top items from millions in a few milliseconds. There is no sublinear retrieval algorithm for an arbitrary MLP scorer. So the MLP is both worse *and* unservable.

## The Methodology

**The three similarity functions being compared.** All of them take the same free-parameter embeddings $\mathbf{p}, \mathbf{q} \in \mathbb{R}^d$. Only $\phi$ differs.

Dot product with biases (the baseline):
$$\phi^{\text{dot}}(\mathbf{p},\mathbf{q}) = b + p_1 + q_1 + \langle \mathbf{p}_{[2..d]}, \mathbf{q}_{[2..d]}\rangle$$
The first coordinate of each embedding is stolen to act as a user bias and item bias. This adds no expressive power (a dot product could learn it) but it is a better inductive bias.

MLP:
$$\phi^{\text{MLP}}(\mathbf{p},\mathbf{q}) = \mathbf{f}_{W_l,\mathbf{b}_l}(\ldots \mathbf{f}_{W_1,\mathbf{b}_1}([\mathbf{p},\mathbf{q}])\ldots), \quad \mathbf{f}_{W,\mathbf{b}}(\mathbf{x}) = \sigma(W\mathbf{x}+\mathbf{b})$$
Three hidden layers of sizes $[4h, 2h, h]$, ReLU.

NeuMF: split the embedding, send the first $j$ coordinates through the MLP and the rest through "generalized MF":
$$\phi^{\text{GMF}}(\mathbf{p},\mathbf{q}) = \sigma\!\left(\sum_{f=1}^d w_f p_f q_f\right), \qquad \phi^{\text{NeuMF}} = \phi^{\text{MLP}} + \phi^{\text{GMF}}$$

**Training the MF baseline.** Implicit feedback (only positives exist). For each positive $(u,i)$, draw $m$ uniform random negatives, re-drawn every epoch. Binary logistic loss with L2:
$$\ell(u,i,y) = -y\ln\sigma(\phi) - (1-y)\ln(1-\sigma(\phi)) + \lambda(\|\mathbf{p}_u\| + \|\mathbf{q}_i\|)$$
Plain SGD. No batching, no [[Momentum|momentum]], no [[Adam- A Method for Stochastic Optimization|Adam]]. Embeddings initialised $\mathcal{N}(0, 0.1^2)$.

**The hyperparameter search — this is where the whole result lives.** They built a *tuning split* that mirrors the real split: hold out each user's last training item as a tuning-test item. Everything, including the stopping epoch, is picked on that tuning split, never the test set. Coarse grid: $\eta \in \{0.001, 0.003, 0.01\}$, $m \in \{4,8,16\}$, $\lambda = 0$ first, then $\lambda \in \{0.001, 0.003, 0.01\}$ around the winners. Budget 256 epochs.

Final: Movielens $\eta = 0.002$, $m = 8$, $\lambda = 0.005$, 256 epochs. Pinterest $\eta = 0.007$, $m = 10$, $\lambda = 0.01$, 256 epochs. Final numbers averaged over 8 runs.

**The apples-to-apples fix.** NCF reported results by "predictive factor" $k$ = size of the last MLP hidden layer. But for the 3-layer MLP, $k$ corresponds to embedding dimension $d = 2k$; for NeuMF, $d = 3k$ (an MLP branch with $d=2k$ plus a GMF branch with $d=k$). Predictive factor is arbitrary — add more layers and it shrinks with nothing else changing. So all NCF numbers were re-plotted against true $d$, multiplying by 2 or 3.

**The synthetic dot-product-learning experiment.** Sample $\mathbf{p},\mathbf{q} \sim \mathcal{N}(0, \sigma_{\text{emb}}^2 I)$, label $y = \langle\mathbf{p},\mathbf{q}\rangle + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma_{\text{label}}^2)$. Fix $M$ user vectors and $N$ item vectors, draw $100M$ pairs, 90/10 train/test. Then a *second* test set with completely **fresh** embeddings never seen in training — does the learned similarity generalise to new points in embedding space?

Train the NCF-shaped MLP ($[4h,2h,h]$, ReLU, [[Adam- A Method for Stochastic Optimization|Adam]]) with $h \in \{d/2, d, 2d\}$. At $h=d$ the parameter count is roughly $18d^2$ — 1,152 params at $d=8$, 1.18M at $d=256$.

The noise scale is calibrated to the Netflix Prize so the errors mean something: $\sigma_{\text{label}} = 0.85$ (roughly the best achievable Netflix RMSE), and $\sigma_{\text{emb}}$ set so the always-predict-zero model gets RMSE 1.13 (Netflix's trivial baseline). Metric reported = MLP RMSE minus dot-product RMSE. On Netflix, 0.01 RMSE is a *huge* gap (took the community a year); 0.001 is a normal publishable increment.

## Ablation Studies and Experiments

**Main table ($d = 192$, baselines from the Dacrema et al. reproducibility study):**

| Method | ML HR@10 | ML NDCG@10 | Pin HR@10 | Pin NDCG@10 |
|---|---|---|---|---|
| Popularity | 0.4535 | 0.2543 | 0.2740 | 0.1409 |
| SLIM | 0.7162 | 0.4468 | 0.8679 | 0.5601 |
| iALS | 0.7111 | 0.4383 | 0.8762 | 0.5590 |
| MLP+GMF (NeuMF) | 0.7093 | 0.4349 | 0.8777 | 0.5576 |
| **This paper's MF** | **0.7294** | **0.4523** | **0.8895** | **0.5794** |

The tuned dot product wins everything. Note NeuMF is not even the best *non*-neural-beating method — SLIM and iALS, both from 2008–2011, already match or beat it.

**The cherry-picking asymmetry.** Dacrema et al. showed NCF's reported MLP/NeuMF numbers were taken at the *best iteration selected on the test set*. So the MLP numbers in Figure 2 are optimistically biased upward, while the MF numbers are honestly selected on a validation split. The dot product wins **despite** giving the opponent that advantage.

**What NeuMF's ablation actually reveals.** Two NeuMF variants:
- Trained end-to-end from scratch (green curve): barely better than pure MLP, far worse than MF. So "add a dot-product branch to your MLP" does not rescue the MLP.
- Pretrain MLP and GMF separately, then fine-tune the combination (red curve): better than green, but still worse than plain MF except at one point (HR on Movielens, $d=192$). The authors read this as plain **ensembling** — two models averaged do better than one — not as evidence that learned similarity helps. Ensembling MF with almost anything would probably give a similar bump.

**Why GMF underperformed in the original paper — a genuinely nice piece of analysis.** GMF adds weights $\mathbf{w}$ to the dot product. Looks harmless. But if you L2-regularise $P$ and $Q$ and *not* $\mathbf{w}$, the model is scale-invariant: rescaling $(P/a, Q/a, a^2\mathbf{w})$ leaves every prediction unchanged, and
$$L(P,Q,\mathbf{w},\lambda) = L\!\left(\tfrac{1}{a}P, \tfrac{1}{a}Q, a^2\mathbf{w}, a^2\lambda\right)$$
So minimising with regularisation $\lambda$ is the same problem as minimising with any $\tilde\lambda$, via $a = \sqrt{\tilde\lambda/\lambda}$. **Regularisation does literally nothing.** Worse: unless $\lambda = 0$, the optimiser drives $\|P\|, \|Q\| \to 0$ and $\|\mathbf{w}\| \to \infty$, which is numerically unstable. The tell in the original paper: GMF's quality did *not* improve with larger $d$ — the signature of an unregularised model. And $\mathbf{w}$ adds zero expressiveness anyway, since it absorbs into $P$ or $Q$.

**The synthetic experiment — what did not work.** With enough data and wide enough hidden layers, an MLP *can* eventually approximate a dot product, and it generalises to fresh embeddings too. But the sample cost scales badly: empirically about $O(d/\epsilon)^\alpha$ samples for $1 \le \alpha \le 2$, matching the theory bound of $O(d^4/\epsilon^2)$ steps for learning a degree-2 polynomial.

The damning number: **at $d = 128$ with 128,000 users, the approximation error is still above 0.02** — double what counts as a "very significant" difference on Netflix-scale problems. And this is the MLP trained *directly and only* on the dot product target, with clean data, no confounding recommendation task. If it cannot learn a dot product under those conditions, it will not discover one inside a real recommender.

## Worth Remembering

- **The serving argument may matter more than the accuracy argument.** Scoring $n$ items costs $O(dn)$ for a dot product versus $O(d^2 n)$ for the MLP. Both are hopeless at $n \sim 10^6$ done naively — but the dot product reduces to maximum inner product search, which has sublinear approximate algorithms ([[Efficient and robust approximate nearest neighbor search using HNSW|HNSW]], [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)|product quantization]]). No such trick exists for an arbitrary MLP scorer. If recommendations depend on query-time context (a sequential recommender over the last $L$ items), you cannot precompute, so this is fatal.

- **This is not an anti-deep-learning paper, and the authors are explicit about it.** Almost every strong DNN uses a dot product at the output: a softmax classifier computes $Q\mathbf{f}(\mathbf{x}) = [\langle\mathbf{p},\mathbf{q}_i\rangle]_{i=1}^n$ — that *is* the model class studied here, with $\mathbf{p} = \mathbf{f}(\mathbf{x})$ and classes as items. [[Deep Residual Learning for Image Recognition (ResNet)|ResNet]], [[BERT- Pre-training of Deep Bidirectional Transformers|BERT]], [[Efficient Estimation of Word Representations (word2vec)|word2vec]] all do this. And [[Attention Is All You Need|attention]] is a dot product between internal embeddings. Use deep nets to *build* the embeddings; use a dot product to *combine* them.

- **Limitations the authors concede.** They only revisit two datasets, Movielens 1M and Pinterest, both small and binarised. They admit MLP/NeuMF might themselves be undertuned and could improve, and that results could differ on other data. Their claim is narrow and honest: the NCF experiments *provide no evidence* that learned similarity beats a dot product — not that it never can.

- **The methodological moral is broader than recommenders.** This is the same lesson as [[On the Difficulty of Evaluating Baselines]] and [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]]: a new method's reported gain is often the gap between a tuned proposal and an untuned baseline. The fix is boring — build a proper validation split, tune the baseline with the same effort, select the stopping epoch honestly.

- **"Adding parameters to a simple model is not always a good idea."** GMF is the clean case study: it added $\mathbf{w}$, gained no expressiveness, and silently destroyed regularisation. Worth checking any "generalisation" of a model for scale invariances that neuter your penalty terms.

- **Open question for the reader:** the follow-up ConvNCF replaces the MLP with an outer product $\mathbf{p}\mathbf{q}^\top$ fed to a CNN. The dot product is the trace of that matrix, so finding it is trivial for the architecture — a much better inductive bias. But it was evaluated on different data, so nobody knows if it beats a tuned MF, and it has the same serving problem.

## Links
Related: [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)]] · [[Collaborative Filtering for Implicit Feedback Datasets (ICDM)]] · [[On the Difficulty of Evaluating Baselines]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Deep Learning]] · [[Regularization]] · [[Efficient and robust approximate nearest neighbor search using HNSW]] · [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[NDCG]] · [[Attention Is All You Need]] · [[Recommender Systems - Evolution]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Sentence-BERT]] · [[Troubling Trends in Machine Learning Scholarship]] · [[The Bitter Lesson (essay)]]

New topics worth writing: Maximum Inner Product Search (MIPS), Universal Approximation Theorem, Neural Collaborative Filtering (NCF/NeuMF), SLIM (Sparse Linear Methods), iALS, Scale invariance and broken regularisation, Two-tower retrieval architectures, Learning polynomials with neural networks (sample complexity)
