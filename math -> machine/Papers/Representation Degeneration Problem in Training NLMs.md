---
title: "Representation Degeneration Problem in Training NLMs"
authors: ["Jun Gao", "Di He", "Xu Tan", "Tao Qin", "Liwei Wang", "Tie-Yan Liu"]
year: 2019
arxiv: "1907.12009"
url: https://arxiv.org/abs/1907.12009
priority: Must-Read
read_on: 2026-08-24
tags: [paper, transformers, llm, theory]
---
## The Core Idea

Train a language model or a translation model with the usual next-word loss, tie the input embedding matrix to the output softmax matrix, and something ugly happens to the learned word vectors: they all end up crammed into a **narrow cone**. Almost every pair of word vectors has *positive* cosine similarity. The vectors do not spread out around the origin — they huddle on one side of it.

The authors call this the **representation degeneration problem**. It is a training pathology, not a modelling choice, and it had gone unnoticed because people looked at BLEU and perplexity, not at the geometry of the embedding table.

> [!NOTE] Representation degeneration
> When a generation model is trained by likelihood maximisation with weight tying, most word embeddings collapse into a narrow convex cone, so all embeddings are positively correlated and the matrix is effectively low-rank. ^representation-degeneration

Why this is bad, in two roles at once (weight tying gives the matrix a dual job):

- **As softmax weights.** Class vectors that all point in similar directions cannot separate hidden states by a large margin. Compare a normal MNIST classifier, where the 10 rows of the last layer point in 10 well-spread directions.
- **As input embeddings.** If every word vector is close to every other, the vectors cannot encode diverse meanings. [[Efficient Estimation of Word Representations (word2vec)|word2vec]] and GloVe vectors do *not* look like this — they spread around the origin.

The mechanism is simple once stated. For any hidden state $h_i$, the [[Cross Entropy|cross-entropy]] gradient pushes the ground-truth word's vector *toward* $h_i$ and pushes **every other word's vector away from** $h_i$. Word frequencies follow Zipf's law: over 90% of tokens in WMT14 En-De appear with frequency below $10^{-4}$. Even a common word like "is" appears in only ~1% of positions. So for nearly every word, nearly every gradient step is a *push-away* step. All those rare words get shoved in roughly the same direction — the direction that is negative against most hidden states — and pile up together.

The fix is one extra term in the loss that pushes embedding directions apart. It costs $\Theta(N)$ time, adds zero parameters, and buys 2.0 perplexity on WikiText-2 and +1.08 BLEU on WMT14 En→De.

## The Methodology

**The setup being analysed.** Standard [[Auto-regressive models|autoregressive]] generation: hidden state $h_i$ (size 512 in base [[Attention Is All You Need|Transformer]]), vocabulary size $N$, and

$$P(Y_i = y_i \mid h_i) = \frac{\exp(\langle h_i, w_{y_i}\rangle)}{\sum_{l=1}^{N}\exp(\langle h_i, w_l\rangle)}$$

where $w_l$ is both the embedding of word $l$ and row $l$ of the softmax matrix.

**Step 1 — the extreme case: a word that never appears.** Fix everything except $w_N$, and assume word $N$ never occurs. Maximising likelihood reduces to

$$\min_{w_N} \frac{1}{M}\sum_{i=1}^{M}\log\big(\exp(\langle h_i, w_N\rangle) + C_i\big), \qquad C_i = \sum_{l=1}^{N-1}\exp(\langle h_i, w_l\rangle)$$

> [!NOTE] Uniformly negative direction
> A vector $v$ with $\langle v, h_i\rangle < 0$ for **every** hidden state $h_i$ in the data. ^uniformly-negative-direction

*Theorem 1:* if such a $v$ exists, the optimum is $w_N^* = \lim_{k\to\infty} k\cdot v$ — the embedding runs off to infinity along $v$. And the set of such $v$ is convex, i.e. it *is* a cone. So the degenerate solution is not an accident of optimisation; it is the actual minimiser.

*Theorem 2:* a uniformly negative direction exists **iff the convex hull of the hidden states does not contain the origin**. (Proof by separating hyperplane: if the hull misses the origin, some plane through the origin separates it, and the plane's normal works.)

*Appendix C — why this condition is essentially always true in practice.* With layer normalisation, $h' = g \odot \frac{h - \mathbf{1}\mu}{\sigma} + b$. Note $\mathbf{1}^T \frac{h-\mathbf{1}\mu}{\sigma} = 0$ by construction. Working through, the origin can be in the convex hull only if $\mathbf{1}^T \frac{b}{g} = 0$ exactly — one scalar equation on learned scale/bias vectors, which unconstrained gradient training will never satisfy. So with layer norm, the cone condition holds almost surely.

**Step 2 — real rare words, not imaginary ones.** Split the loss for word $w_N$ into two parts: $A_{w_N}$ = sentences that do **not** contain it (here it behaves exactly like a non-appeared token, and the loss is convex in $w_N$), and $B_{w_N}$ = sentences that do (messy, non-convex). Total loss is a $(1-\epsilon)/\epsilon$ mixture, with $\epsilon$ = probability a sentence contains the word — tiny for rare words.

*Theorem 3:* if $f$ is $\alpha$-strongly convex, $g$ has $\mathbf{H}(g) \succ -\beta I$ and $|g| < B$, and $\epsilon < \frac{\alpha}{\alpha+\beta}$, then the optimum of $(1-\epsilon)f + \epsilon g$ sits within

$$\|x^* - x^*_\epsilon\|_2^2 \le \frac{4\epsilon B}{\alpha - \epsilon(\alpha+\beta)}$$

of the optimum of $f$ alone. Plain reading: a rare word's embedding lands very close to where the "never appears" solution would put it. And since two rare words $w$ and $w'$ share almost all of their $A$-sets, their two convex losses are nearly identical — so their learned embeddings end up **near each other**. That is the huddling.

**The fix — MLE with Cosine Regularization (MLE-CosReg).** Widen the cone's aperture by directly penalising pairwise cosine similarity. With $\hat{w} = w/\|w\|$:

$$L = L_{\text{MLE}} + \gamma \frac{1}{N^2}\sum_{i}^{N}\sum_{j\neq i}^{N} \hat{w_i}^T\hat{w_j}$$

**Why it works, spectrally.** Write $R = \text{Sum}(\hat{W}\hat{W}^T) - N$. Since $\hat W \hat W^T$ is positive semi-definite with all-ones diagonal, its eigenvalues are non-negative and sum to $N$ (a constant). Merikoski's theorem: for a non-negative matrix, the spectral radius $\le \text{Sum}(A)$. Because in the degenerate regime *all* cosine similarities are positive, $\hat W \hat W^T$ is a non-negative matrix, so minimising $R$ minimises an upper bound on the top eigenvalue. Total is fixed, so shrinking the biggest eigenvalue pushes mass into the smaller ones — a flatter spectrum, a higher effective rank, a more expressive embedding table.

**Cost is linear, not quadratic.** The naive double sum is $O(N^2)$, but

$$\sum_i\sum_{j\neq i}\hat w_i^T \hat w_j = \Big\|\sum_i^N \hat w_i\Big\|_2^2 - N$$

so you just sum the normalised vectors once: $\Theta(N)$.

**Training setups.**
- *Language modelling:* WikiText-2, ~30k vocab. AWD-LSTM ([[Long Short-Term Memory (Neural Computation)|LSTM]]) exactly as Merity et al.: 3 layers, 1150 hidden units, embedding size 400, same dropout, Averaged SGD, 24M params. Optional neural continuous cache pointer on top.
- *Translation:* WMT14 En-De, 4.5M pairs, BPE with ~37k shared subword units. Base Transformer (6+6 layers, $d=512$) and Big Transformer, Adam, all defaults from the original paper.
- $\gamma = 1$ everywhere. The authors report it is "not very sensitive".

## Ablation Studies and Experiments

**Language modelling — WikiText-2 perplexity (lower better), all 24M params:**

| Model | Valid | Test |
|---|---|---|
| 2-layer skip-connection LSTM (tied) | 69.1 | 65.9 |
| AWD-LSTM, no finetune | 69.1 | 66.0 |
| **MLE-CosReg, no finetune** | **68.2** | **65.2** |
| AWD-LSTM, with finetune | 68.6 | 65.8 |
| **MLE-CosReg, with finetune** | **67.1** | **64.1** |
| AWD-LSTM + cache pointer | 53.8 | 52.0 |
| **MLE-CosReg + cache pointer** | **51.7** | **50.0** |

Gains of 0.8 / 1.7 / 2.0 test perplexity. Note the gain **grows** as the base model gets stronger — the cache-pointer setting benefits most.

**Machine translation — BLEU:**

| En→De | BLEU | De→En | BLEU |
|---|---|---|---|
| ConvS2S | 25.16 | DSL | 20.81 |
| Base Transformer | 27.30 | Dual-NMT | 22.14 |
| **Base + MLE-CosReg** | **28.38** | ConvS2S (own impl.) | 29.61 |
| Big Transformer | 28.40 | Base Transformer (own impl.) | 31.00 |
| **Big + MLE-CosReg** | **28.94** | **Base + MLE-CosReg** | **31.93** |

+1.08 En→De, +0.93 De→En, +0.54 on Big. **Base + CosReg (28.38) nearly matches Big Transformer (28.40) with a third of the parameters.** No architecture change, no extra parameters, no hyperparameter retuning — the delta is attributable to the single regulariser.

**Does the geometry actually change?** Two diagnostics:

1. **2D SVD projection (Figure 2a).** The vanilla Transformer's embeddings form a narrow cone off to one side; with CosReg they spread roughly uniformly around the origin. The authors checked that the rank-2 approximation is honest — the remaining singular values were much smaller than the top two, which is *itself* evidence of degeneration.
2. **Singular value spectrum (Figure 2b), normalised so the largest = 1.** Vanilla: a couple of singular values dominate and the rest fall off a cliff. CosReg: the spectrum is far flatter. This is the direct confirmation of the eigenvalue argument in the method.

**What the analysis rules out as the cause.** The degeneration is *not* about the architecture — the authors reproduced it in an LSTM-based NMT model (Wu et al. 2016) too. It is *not* about softmax classification in general — the MNIST classifier's category embeddings are well spread. It is *not* about word embeddings in general — word2vec and GloVe are fine (consistent with Mu et al. 2018's all-but-the-top findings). What is left: the combination of **likelihood maximisation + weight tying + a heavy-tailed frequency distribution + layer norm ensuring the hidden-state hull misses the origin**.

**Honest gaps.** There is no $\gamma$ sweep table, no ablation of tied vs untied embeddings under CosReg, and no experiment isolating layer norm (e.g. removing it to see if degeneration weakens), even though the theory says layer norm is the enabling condition. The claim "$\gamma$ is not very sensitive" is asserted, not shown.

## Worth Remembering

- **The gradient story is the thing to keep.** In any softmax over a large vocabulary, each step gives one positive pull and $N-1$ negative pushes. With Zipfian frequencies, rare tokens receive almost only pushes. This is the same asymmetry that motivated [[Distributed Representations of Words and Phrases (negative sampling)|negative sampling]] — but there the negatives are *sampled by frequency*, which changes the dynamics. It is a reasonable hypothesis that this is part of why word2vec vectors do not degenerate.
- **Weight tying is a double-edged sword.** Inan et al. and Press & Wolf showed tying reduces parameters and helps generalisation, so it is standard in [[Attention Is All You Need|Transformers]] and AWD-LSTM. This paper is the counterweight: tying forces one matrix to satisfy two conflicting geometric demands, and the softmax role wins.
- **This is the ancestor of the "anisotropy" literature.** Everything later written about BERT/GPT embedding spaces being anisotropic, and fixes like whitening, BERT-flow and SimCSE, traces back here. Worth reading alongside [[BERT- Pre-training of Deep Bidirectional Transformers|BERT]] — the phenomenon shows up in contextual hidden states too, not just the static embedding table.
- **The regulariser is cheap enough that there is no excuse not to try it.** One norm of one summed vector per step, $\Theta(N)$. Drop-in for any tied-embedding [[Seq2Seq models|seq2seq]] or LM training loop, $\gamma = 1$ as a starting point.
- **Caveat: it is a uniformity prior, and uniformity is not always right.** Pushing *all* pairs apart also pushes apart genuinely synonymous words. On these benchmarks it helps, but on a task where fine-grained semantic clustering matters you would want to check. The authors themselves say "there may exist some better regularization terms" and point at FRAGE (frequency-agnostic word representations, same group) as a complement.
- **The Big-Transformer gain is smaller (0.54 vs 1.08).** One reading: larger models partly escape degeneration on their own, and the regulariser's value shrinks with scale. Not tested, but a natural follow-up.
- **Open question worth chasing:** the theory pins the condition on the convex hull of hidden states missing the origin, caused by layer norm's learned bias $b$. Would a bias-free or zero-centred final normalisation remove the pathology at its root, with no regulariser at all?

## Links

Related: [[Attention Is All You Need]] · [[Long Short-Term Memory (Neural Computation)]] · [[Cross Entropy]] · [[Efficient Estimation of Word Representations (word2vec)]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[A Neural Probabilistic Language Model (JMLR)]] · [[Regularization]] · [[Linear Projection]] · [[Auto-regressive models]] · [[Seq2Seq models]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Derivative#Hessian|Hessian]] · [[Fundamentals#Linear Algebra|Linear Algebra]]

New topics worth writing: Weight tying (input/output embedding sharing), Layer normalization, Anisotropy in contextual embeddings, Softmax bottleneck (Yang et al. 2018), Zipf's law and long-tail vocabularies, Strong convexity and perturbation bounds, Singular value spectrum as a measure of effective rank, All-but-the-top post-processing, FRAGE / frequency-agnostic embeddings, Byte-pair encoding
