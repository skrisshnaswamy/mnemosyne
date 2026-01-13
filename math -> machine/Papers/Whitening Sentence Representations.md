---
title: "Whitening Sentence Representations"
authors: ["Jianlin Su", "Jiarun Cao", "Weijie Liu", "Yangyiwen Ou"]
year: 2021
arxiv: "2103.15316"
url: https://arxiv.org/abs/2103.15316
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, optimization]
---
## The Core Idea

Take BERT's sentence vectors — the ones everybody gets by averaging token vectors — and do nothing clever at all. Just centre them and multiply by a matrix that makes their covariance the identity. That is **whitening**, a trick from classical statistics older than deep learning. It matches or beats BERT-flow, which needed a whole normalising-flow network trained with gradient descent, and it takes about ten lines of NumPy.

The argument for *why* this should work is the interesting part. When you compare two sentence embeddings you almost always use cosine similarity:

$$\cos(x,y)=\frac{\sum_i x_i y_i}{\sqrt{\sum_i x_i^2}\sqrt{\sum_i y_i^2}}$$

That formula is not the definition of an angle. It is the *coordinate formula* for an angle, and it is only correct if the coordinates are written in a **standard orthogonal basis** — axes that are mutually perpendicular and all of unit length. BERT never promised its 768 hidden dimensions were such a basis. They are correlated with each other and have wildly different variances. So when you plug BERT vectors into that formula, you are computing an angle in the wrong geometry, and the number you get is not the angle you wanted.

This connects directly to [[How Contextual are Contextualized Word Representations|anisotropy]]: BERT vectors all crowd into a narrow cone, average cosine similarity near 0.99. Anisotropy is the *symptom*; "your basis is not orthonormal" is the authors' proposed *diagnosis*. If a cloud of vectors is isotropic — spread evenly in every direction — you can pretend the basis is orthonormal and the cosine formula is honest again.

> [!NOTE] Whitening
> A linear map that shifts a set of vectors to zero mean and rescales/rotates them so their covariance matrix becomes the identity $I$. Every direction then has equal variance and no two directions are correlated. ^whitening

The bonus, which turned out to matter as much as the accuracy gain: whitening hands you the eigenvalues for free, sorted. Throw away the small ones and you get [[Fundamentals|PCA]]-style dimensionality reduction in the same step. 768 dims → 256 dims, *and* the score goes up. Smaller index, faster retrieval, better numbers.

## The Methodology

You have $N$ sentence embeddings $\{x_i\}_{i=1}^N$, written as **row** vectors. Apply

$$\tilde{x}_i = (x_i - \mu)W$$

Mean is easy:

$$\mu = \frac{1}{N}\sum_{i=1}^{N} x_i$$

For $W$, start from the covariance (row-vector convention, so the transpose is on the left):

$$\Sigma = \frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^T(x_i-\mu)$$

After the transform the covariance is $\tilde{\Sigma} = W^T \Sigma W$, and we demand $W^T\Sigma W = I$. Rearranging gives $\Sigma = (W^{-1})^T W^{-1}$. Since $\Sigma$ is symmetric positive definite, SVD gives

$$\Sigma = U\Lambda U^T$$

with $U$ orthogonal and $\Lambda$ diagonal with positive entries. Set $W^{-1} = \sqrt{\Lambda}\,U^T$, which yields the solution:

$$W = U\sqrt{\Lambda^{-1}}$$

Read it right to left: $U^T$ rotates into the eigen-directions, $\sqrt{\Lambda^{-1}}$ divides each of those directions by its standard deviation so all variances become 1.

**Dimensionality reduction (Whitening-$k$).** SVD already returns $\Lambda$ sorted descending. A tiny diagonal entry means that direction barely varies across the corpus — nearly a constant, carrying no signal but still polluting the cosine sum. So keep only the first $k$ columns:

$$W = \left(U\sqrt{\Lambda^{-1}}\right)[:,\;:k]$$

That is the whole algorithm. No gradients, no [[Backpropagation]], no training loop.

**Streaming.** $\mu$ and $\Sigma$ can be accumulated online:

$$\mu_{n+1}=\frac{n}{n+1}\mu_n+\frac{1}{n+1}x_{n+1},\qquad \Sigma_{n+1}=\frac{n}{n+1}\Sigma_n+\frac{1}{n+1}(x_{n+1}-\mu)^T(x_{n+1}-\mu)$$

So memory is $O(d^2)$ regardless of corpus size (the paper says $O(1)$, meaning constant in $N$), time is $O(N)$. You never hold the corpus in RAM.

**Embedding recipe.** Default is `first-last-avg`: average the token vectors of the *first* and *last* [[BERT- Pre-training of Deep Bidirectional Transformers|BERT]] layers. The authors report this is stably better than averaging only the last layer.

**Two fitting regimes**, copied from BERT-flow so the comparison is fair:
- `-whitening(NLI)` — $\mu, W$ estimated on the NLI corpus (SNLI + MNLI).
- `-whitening(target)` — estimated on the target STS dataset itself: all sentences from train/dev/test, **labels excluded**. Unsupervised, but it does see the test *inputs*.

Evaluated on STS-12 through STS-16, STS-B, and SICK-R. Metric is Spearman rank correlation ($\rho \times 100$) between predicted cosine and the human 0–5 score.

## Ablation Studies and Experiments

**Baseline floor.** Raw `BERT CLS-vector` is catastrophic: 16.50 on STS-B, worse than averaged GloVe (58.02). Averaged BERT gets 46.35. `BERT_base-first-last-avg` gets 59.04.

**Without NLI supervision (Table 1), BERT_base:**

| Method | STS-B | STS-12 | STS-13 | STS-14 | STS-15 | STS-16 | SICK-R |
|---|---|---|---|---|---|---|---|
| first-last-avg | 59.04 | 57.86 | 61.97 | 62.49 | 70.96 | 69.76 | 63.75 |
| flow (NLI) | 58.56 | 59.54 | 64.69 | 64.66 | 72.92 | 71.84 | 65.44 |
| **whitening (NLI)** | **68.19** | 61.69 | 65.70 | 66.02 | **75.11** | **73.11** | 63.6 |
| flow (target) | 70.72 | 63.48 | 72.14 | 68.42 | 73.77 | 75.37 | 63.11 |
| **whitening-256 (target)** | **71.43** | **63.89** | **73.76** | 69.08 | 74.59 | 74.40 | 62.2 |

The STS-B jump from flow(NLI) 58.56 → whitening(NLI) 68.19 is nearly ten points, from a closed-form matrix multiply.

**With NLI supervision (Table 2), on top of [[Sentence-BERT|SBERT]].** This is the more surprising result: whitening still helps a model already fine-tuned on labelled NLI pairs. `SBERT_large-NLI-whitening-384(target)` reaches 82.22 on STS-B vs 79.16 for the plain first-last-avg baseline and 81.18 for SBERT-flow. `SBERT_base-NLI-whitening-256(NLI)` gets 82.73 on STS-15 vs 79.65 baseline.

**Dimensionality is not just free compression — it is doing real work.** Compare whitening(target) at full 768 vs whitening-256(target) on BERT_base: STS-13 goes 73.02 → 73.76, STS-12 goes 63.62 → 63.89. Cutting a third of the dimensions *improves* the score. Figure 1 shows the same curve shape across tasks: performance rises as $k$ drops, peaks around one third of the original dimension, then falls. On BERT_base-whitening(NLI) / SICK-R, $k=109$ scores 66.52, beating BERT-flow(NLI)'s 65.44 by 1.08 — at 109 dimensions instead of 768. So the low-variance directions were actively harmful, not merely useless. This is essentially the same story as [[Understanding Dimensional Collapse in Contrastive Learning|dimensional collapse]] read in reverse.

**What did not work.**
- **SICK-R is the consistent loser.** Almost every whitening row on SICK-R has a red ↓ against the flow baseline. BERT_base-whitening(target) drops SICK-R to 60.6 from a 63.75 baseline — whitening actively *hurt* it. The authors' defence is only that their vector is smaller.
- **Full-dimension whitening on BERT_large(target) is a clean failure**: every single one of the seven columns is ↓ against BERT_large-flow(target). STS-B 72.14 vs 72.26, STS-16 72.52 vs 77.63 — a five-point loss. Adding the dimensionality cut to 384 rescues most of it (STS-B 72.48, STS-13 74.60, STS-14 69.64 all now ↑) but STS-16 still trails at 75.90 vs 77.63. **Reading: on the large model, the dimensionality reduction, not the whitening, is the component carrying the win.** Whitening alone can amplify tiny noisy directions by dividing by a near-zero $\sqrt{\lambda}$; truncation is what stops that.
- The optimal $k$ moves with model size — 256 for base (768-dim), 384 for large (1024-dim). It is an empirical hyperparameter with no principled selection rule offered.

## Worth Remembering

- **The `(target)` setting is a soft form of transduction.** Fitting $\mu$ and $W$ on the target dataset's own sentences means the method sees the test *distribution* before predicting. No labels are used, so it is legitimately unsupervised, but you cannot reproduce these numbers on a genuinely cold, unseen query. In production you would fit on your corpus once and hope the query distribution matches. The `(NLI)` rows are the honest transfer numbers, and they are lower.
- **A whitening matrix is just a [[Linear Projection|linear projection]] — so fold it into the last layer.** $\tilde{x} = (x-\mu)W$ can be baked into the model's final weights at export time. Zero inference cost, and downstream vector search gets 3× smaller indices.
- **The theoretical argument is a heuristic, not a proof.** "Isotropic ⟹ the basis is standard orthonormal ⟹ the cosine formula is valid" is intuition dressed as derivation. Isotropy of a point cloud does not logically fix the basis. But the empirical result stands regardless, and the framing is a genuinely useful way to think about why cosine similarity misbehaves.
- Sits between two lines of work: [[Representation Degeneration Problem in Training NLMs]] tried to fix anisotropy *during* training with a regulariser; this fixes it *afterwards* with linear algebra. Compare with [[Understanding Contrastive Learning through Alignment and Uniformity|uniformity]] — contrastive losses push toward the same isotropic goal via gradient descent. [[SimCSE- Simple Contrastive Learning of Sentence Embeddings|SimCSE]] arrived shortly after and beat whitening by actually fine-tuning; whitening remains the strongest thing you can do with *zero* training.
- Prior art the paper credits and quietly generalises: Arora et al. (2017) removed the top principal direction; Mu & Viswanath's "all-but-the-top" removed the mean and a few dominant directions. Whitening does not remove the big directions, it *equalises* them — and then removes the small ones.
- Practical caveat: $\Sigma$ must be well-conditioned. Tiny eigenvalues get $1/\sqrt{\lambda}$ blow-up. If you skip the truncation you may make things worse, which is exactly what the BERT_large(target) row shows.

## Links

Related: [[How Contextual are Contextualized Word Representations]] · [[Representation Degeneration Problem in Training NLMs]] · [[Sentence-BERT]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[SimCSE- Simple Contrastive Learning of Sentence Embeddings]] · [[Understanding Contrastive Learning through Alignment and Uniformity]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Linear Projection]] · [[Fundamentals]] · [[Embedding-based Retrieval in Facebook Search]] · [[RoBERTa- A Robustly Optimized BERT Pretraining Approach]] · [[Efficient Estimation of Word Representations (word2vec)]]

New topics worth writing: Whitening transform, Principal Component Analysis, Normalising flows, BERT-flow, Covariance matrix and SVD, Spearman rank correlation, Semantic Textual Similarity benchmarks, Isotropy of embedding spaces, Cosine similarity and choice of basis
</result>
