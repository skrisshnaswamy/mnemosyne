---
title: "A Neural Probabilistic Language Model (JMLR)"
authors: ["Bengio et al."]
year: 2003
url: https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
priority: Must-Read
read_on: 2026-08-24
tags: [paper, llm]
---
## The Core Idea

Before this paper, a language model was a giant lookup table. You counted how often "the cat is" was followed by "walking", and that count *was* the model. This is the n-gram approach, and it has a fatal flaw the authors call the **curse of dimensionality**: with a vocabulary of 100,000 words and a 10-word window, there are $100000^{10} - 1 \approx 10^{50}$ possible sequences. Almost every sentence you will ever be tested on was never seen in training. Words are discrete symbols, so "cat" and "dog" are as far apart as "cat" and "helicopter" — the table has no way to know they behave alike.

The fix: stop treating words as symbols. Give each word a short real-valued vector (30, 60, or 100 numbers) and make the probability function a smooth function of those vectors. Then learn the vectors and the probability function **at the same time**, by [[Backpropagation|backprop]] on next-word prediction.

Why this beats counting: the probability function is smooth, so if "dog" ends up near "cat" in vector space, then seeing *"The cat is walking in the bedroom"* automatically raises the probability of *"A dog was running in a room"* and a combinatorial number of other unseen sentences. One training sentence teaches the model about exponentially many neighbours. Counting tables can only generalise by gluing together short overlapping fragments they have literally seen.

> [!NOTE] Distributed representation
> A word is represented by a *pattern of activity across many units*, not by one dedicated slot. Similarity is a distance in $\mathbb{R}^m$, so information shares across similar words automatically. Contrast with class-based n-grams, which assign each word to one discrete cluster — a hard partition instead of a continuous space. ^distributed-representation

This is the paper that word2vec, GloVe, and every embedding layer in every modern model descend from. It is also, structurally, a tiny [[Improving Language Understanding by Generative Pre-Training (GPT-1)|decoder-only]] language model: fixed context window, [[Cross Entropy|cross-entropy]] on the next token, learned token embeddings.

## The Methodology

The model factorises a sentence the standard way, using the [[Markov Property|Markov]] assumption that only the last $n-1$ words matter:

$$\hat{P}(w_1^T) = \prod_{t=1}^{T} \hat{P}(w_t \mid w_{t-1}, \dots, w_{t-n+1})$$

Two pieces:

**1. The lookup table $C$.** A $|V| \times m$ matrix of free parameters. Row $i$ is the feature vector $C(i) \in \mathbb{R}^m$ for word $i$. This is literally the modern embedding matrix. The *same* $C$ is used for every position in the context — parameter sharing across time.

**2. The network $g$.** Concatenate the context embeddings into one long vector:

$$x = (C(w_{t-1}), C(w_{t-2}), \dots, C(w_{t-n+1})) \in \mathbb{R}^{(n-1)m}$$

then compute unnormalised scores (logits) with one $\tanh$ hidden layer plus an optional **direct connection** — a straight [[Linear Projection|linear map]] from $x$ to the output, skipping the hidden layer:

$$y = b + Wx + U\tanh(d + Hx)$$

$$\hat{P}(w_t \mid \text{context}) = \frac{e^{y_{w_t}}}{\sum_i e^{y_i}}$$

Shapes: $U$ is $|V| \times h$, $W$ is $|V| \times (n-1)m$, $H$ is $h \times (n-1)m$, $b$ is $|V|$, $d$ is $h$. Set $W = 0$ to drop the direct connections. Total parameters $\approx |V|(nm + h)$ — **linear** in vocabulary and in context length $n$, not exponential. That is the whole point.

Note there are effectively two hidden layers, but the first one ($C$) has no nonlinearity, because a nonlinearity there would add nothing.

**Objective.** Maximise penalised average log-likelihood by plain stochastic gradient *ascent*:

$$L = \frac{1}{T}\sum_t \log f(w_t, \dots, w_{t-n+1}; \theta) + R(\theta)$$

$R$ is weight decay (squared L2, like ridge — see [[Regularization]]) applied to $W$, $H$, $U$ and $C$ but **not** to the biases. Update: $\theta \leftarrow \theta + \varepsilon \frac{\partial \log \hat{P}}{\partial \theta}$, one word at a time. Only the rows of $C$ for the words in the current window get touched — a sparse gradient.

**Hyperparameters that mattered.** Initial learning rate $\varepsilon_0 = 10^{-3}$, decayed as $\varepsilon_t = \varepsilon_0 / (1 + rt)$ with $r = 10^{-8}$ and $t$ = number of updates. Weight decay $10^{-4}$ on Brown, $10^{-5}$ on AP News. Embeddings initialised randomly. Early stopping on the validation set (only needed on Brown).

**Data.**
- *Brown*: 1,181,041 words. 800k train / 200k valid / 181k test. Words with frequency $\le 3$ merged into one symbol → $|V| = 16{,}383$.
- *AP News 1995–96*: 14M train / 963k valid / 963k test. 148,721 distinct words squashed to $|V| = 17{,}964$ by lowercasing, mapping numbers and proper nouns to special symbols.

**The compute problem and how they solved it.** Unlike an n-gram, you cannot get one probability without computing all $|V|$ logits, because of the softmax denominator. They measured that the output layer is **99.7%** of the arithmetic. So they parallelised *across output units*, not across data: each of 40 CPUs owns a block of the vocabulary, computes its partial softmax sum, and an MPI `Allreduce` shares (a) the sum $S$, (b) the max logit $Q$ for numerically safe exponentials $e^{y_j - Q}$, and (c) the gradients $\partial L/\partial x$ and $\partial L/\partial a$. Every CPU redundantly recomputes the cheap embedding and hidden-layer forward pass. Communication overhead was only 1/15 of wall-clock — near-linear speedup.

They also tried a data-parallel version on shared memory. The locked version was **extremely slow** — processors spent most cycles waiting for write locks. Going fully asynchronous (processors clobber each other's updates) worked fine; the lost updates acted as small noise and did not slow convergence. This is Hogwild, fourteen years early.

**Mixture with a trigram.** Final reported numbers often average the neural output with an interpolated trigram, usually 50/50.

## Ablation Studies and Experiments

Metric is **perplexity** = geometric mean of $1/\hat{P}(w_t \mid \text{context})$, i.e. $\exp$ of average negative log-likelihood. Lower is better. Baselines: deleted-interpolation trigram, modified Kneser-Ney back-off (a strong smoothing method), and class-based n-grams — all from SRILM.

**Brown (Table 1), test perplexity:**

| Model | $n$ | $h$ | $m$ | direct | mix | test |
|---|---|---|---|---|---|---|
| MLP1 | 5 | 50 | 60 | yes | no | 268 |
| MLP2 | 5 | 50 | 60 | yes | yes | **257** |
| MLP3 | 5 | **0** | 60 | yes | no | 310 |
| MLP5 | 5 | 50 | 30 | yes | no | 279 |
| MLP7 | **3** | 50 | 30 | yes | no | 293 |
| MLP9 | 5 | 100 | 30 | **no** | no | 276 |
| MLP10 | 5 | 100 | 30 | no | yes | **252** |
| Deleted interpolation | 3 | | | | | 336 |
| Kneser-Ney back-off | 3/4/5 | | | | | 323 / 321 / 321 |
| class-based back-off | 3 | | 500 | | | 312 |

Best neural = 252 vs best n-gram = 312. **24% better** than the best n-gram, 33% better than the deleted-interpolation trigram.

**AP News (Table 2):** MLP10 ($n=6$, $h=60$, $m=100$, direct, mixed) = **109** test perplexity vs Kneser-Ney 5-gram = 117. Only ~8% better, and only 5 epochs were run (three weeks on 40 CPUs); no overfitting had appeared yet on validation.

**What the ablations reveal:**

- **Hidden units matter.** MLP1 (h=50) 268 vs MLP3 (h=0) 310. Removing the nonlinearity costs 42 perplexity. Embeddings alone are not enough.
- **Longer context helps the neural net but not the n-grams.** MLP7 (n=3) 293 → MLP1-family (n=5) 279 at matched $m=30$. Meanwhile Kneser-Ney is flat at 323/321/321 going from trigram to 5-gram — the count table cannot use the extra words because of data sparsity. This is the cleanest evidence for the central claim.
- **Mixing with the trigram always helps**, by roughly 10–15 perplexity in every pair (268→257, 310→272, 279→259, 293→270, 276→252). The two model families make errors in *different places*.
- **Direct connections: verdict inconclusive, leaning "drop them".** Without $W$ (MLP9/10) the model reached slightly lower perplexity but took **twice as long** to converge (20 epochs vs 10). The authors' reading: direct connections add capacity and let the linear part of the map be learned fast; removing them forces a tight bottleneck through the hidden layer, which may generalise better on a small corpus.
- **Class-based n-grams are sensitive and non-monotone in class count**: 150→348, 200→354, 500→**326**, 1000→335, 2000→343 (validation). No smooth trend; you must tune it.

**What did not work:**

- **Fixed, pre-computed embeddings.** They tried initialising word features from the first principal components of the word co-occurrence matrix — essentially LSI — and holding them fixed. This was *unsuccessful*. Learning the representation jointly with the probability function is what matters. Worth sitting with: the 2003 answer to "just use pretrained vectors" was "that was worse".
- **Lock-based data-parallel training.** Unusably slow.
- **Class-based n-grams on AP News** did not help the baselines at all.

## Worth Remembering

- **The bottleneck was, and for a long time remained, the softmax.** 99.7% of FLOPs. The paper's own listed future work is a preview of the next decade: hierarchical softmax over a word-class tree (a $|V|/\log|V|$ speedup), sampling only a subset of output words, and importance sampling (Bengio & Senécal 2003, 100× speedup). Same problem later solved by negative sampling in word2vec.

- **Section 5.1 quietly invents a second architecture.** An energy-based variant where the *output* word also gets an embedding fed into $x$:
  $$E(w_{t-n+1},\dots,w_t) = v \cdot \tanh(d + Hx) + \sum_{i=0}^{n-1} b_{w_{t-i}}$$
  with $\hat{P}(w_t \mid \cdot) \propto e^{-E}$. In the main model the output layer wastes most of the parameters and learns no similarity structure between *target* words — this fixes that. It is a product-of-experts / maximum-entropy model where the basis functions are learned rather than greedily selected. Modern weight-tied embedding layers are the descendant.

- **Out-of-vocabulary trick.** For an unseen word $j$ in context, initialise its embedding as the probability-weighted average of everything the model expected there: $C(j) \leftarrow \sum_{i \in V} C(i)\hat{P}(i \mid w_{t-n+1}^{t-1})$. Elegant, and rarely reimplemented.

- **Limitations the authors own.** Polysemy is unhandled — one word gets exactly one point in space, so "bank" is a blur of two meanings. Fixed context window (they suggest time-delay or recurrent nets to reach paragraph scale, which is exactly what happened next — see [[Long Short-Term Memory (Neural Computation)|LSTM]] language models, then [[Attention Is All You Need|Transformers]]). They also never got to interpret the learned features; they suspected $m = 2$ would be visualisable but meaningful structure would need much bigger corpora. It did.

- **The theoretical wobble they flag and then shrug off:** with weight decay on $W$ and $H$ but not on $C$, the network could in principle drive $W, H \to 0$ while $C$ blows up. Did not happen with SGD in practice.

- **Practical caveat for a modern reader:** the reported wins are *relative to n-grams on tiny corpora*. 14M words, 5 epochs, three weeks on 40 CPUs. The idea scaled; these particular numbers are historical. What survives is the claim in the conclusion: replacing conditional probability *tables* with smooth distributed representations shifts the cost from exponential memory to linear memory plus more compute — and compute is the thing that got cheap.

- **The honest framing of the result:** the neural model does not beat the trigram by being better at everything. It beats it in different places, which is why a flat 50/50 mixture reliably wins. In 2003 the neural net was a complement to counting, not a replacement.

## Links

Related: [[Backpropagation]] · [[Cross Entropy]] · [[Regularization]] · [[Attention Is All You Need]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[Long Short-Term Memory (Neural Computation)]] · [[Linear Projection]] · [[Markov Property]] · [[Deep Learning]] · [[Auto-regressive models]] · [[Seq2Seq models]] · [[Scaling Laws for Neural Language Models]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Foundational_Reading_Plan]]

New topics worth writing: Perplexity, Curse of dimensionality, Word embeddings and word2vec, Hierarchical softmax and negative sampling, Kneser-Ney smoothing, N-gram language models, Hogwild asynchronous SGD, Energy-based models, Products of experts, Latent Semantic Indexing, Weight tying in language models
