---
title: "Distributed Representations of Words and Phrases (negative sampling)"
authors: ["Mikolov et al."]
year: 2013
arxiv: "1310.4546"
url: https://arxiv.org/abs/1310.4546
priority: Must-Read
read_on: 2026-08-24
tags: [paper, self-supervised, vision]
---
## The Core Idea

The first word2vec paper ([[Efficient Estimation of Word Representations (word2vec)]]) showed skip-gram works. This one makes it fast and better, with three cheap tricks.

The bottleneck is the softmax. To say "given the word *France*, how likely is *Paris* nearby?" you must normalise over the whole vocabulary — 692,000 words. Every single training step. That is one dot product per word in the vocabulary, per example. Hopeless.

The trick: **stop asking for a probability distribution at all.** You do not need a normalised model of language. You only want good vectors. So replace the multi-class question with a pile of yes/no questions:

- "Is (*France*, *Paris*) a real pair from the corpus?" → say yes.
- "Is (*France*, *refrigerator*) a real pair?" → say no. Draw a handful of fake partners at random.

That is **negative sampling**. Each step touches $k+1$ output vectors instead of $W$. With $k=5$ that is 6 dot products instead of 692,000. This is a five-orders-of-magnitude cut in cost per example, and it is what let them train on 33 billion words in a day on one machine.

> [!NOTE] Negative sampling
> Turn "predict which word out of $W$" into "$k+1$ independent logistic-regression yes/no calls": one real (word, context) pair labelled positive, $k$ randomly drawn pairs labelled negative. No normalisation constant, no full-vocabulary sum. ^negative-sampling

Two other contributions ride along. **Subsampling** throws away most occurrences of "the", "a", "in" — they are frequent and carry almost no signal, and dropping them both speeds training 2–10× *and* improves the rare-word vectors. **Phrases**: run a bigram-scoring pass over the corpus, glue "New York Times" into a single token, and now the model can learn that a newspaper is not a city plus a globe.

The larger point that outlived the paper: negative sampling is now the default way to train any embedding model with a huge output space — recommenders, retrieval, contrastive vision. Every two-tower retrieval system is doing this. See [[Real-time Personalization using Embeddings for Search Ranking at Airbnb (KDD, best paper)]], which is literally skip-gram with listings instead of words.

## The Methodology

**Skip-gram objective.** Slide a window over the text. For each centre word $w_t$, try to predict each neighbour within $\pm c$ positions:

$$\frac{1}{T}\sum_{t=1}^{T}\ \sum_{-c\le j\le c,\ j\ne 0}\log p(w_{t+j}\mid w_t)$$

Every word has **two** vectors: an "input" vector $v_w$ used when it is the centre word, and an "output" vector $v'_w$ used when it is the context word. The naive softmax is

$$p(w_O\mid w_I)=\frac{\exp({v'_{w_O}}^\top v_{w_I})}{\sum_{w=1}^{W}\exp({v'_{w}}^\top v_{w_I})}$$

and the denominator is the problem. Note there are no hidden layers, no nonlinearity — just two embedding tables and a dot product. That linearity is why the vectors end up with linear analogy structure.

**Negative sampling replaces each $\log p(w_O \mid w_I)$ term with:**

$$\log \sigma({v'_{w_O}}^\top v_{w_I}) \;+\; \sum_{i=1}^{k}\mathbb{E}_{w_i\sim P_n(w)}\big[\log \sigma(-{v'_{w_i}}^\top v_{w_I})\big]$$

where $\sigma(x)=1/(1+e^{-x})$. Read it plainly: push the real pair's dot product up, push $k$ random pairs' dot products down. It is $k+1$ binary [[Cross Entropy|cross-entropy]] terms. Gradients flow to only $k+2$ vectors, so the update is sparse and cheap.

$k = 5$–$20$ for small corpora, $k = 2$–$5$ for large ones.

**The noise distribution matters.** They sample negatives from the unigram frequency raised to the $3/4$ power:

$$P_n(w) = \frac{U(w)^{3/4}}{Z}$$

This "significantly outperformed" both the plain unigram $U(w)$ and the uniform distribution, on every task they tried. Intuition: raising to $3/4$ flattens the distribution — rare words get sampled as negatives more often than their raw frequency suggests, frequent words less. Pure unigram would make "the" the negative almost every time (useless, you learn nothing new); pure uniform would make the negatives all be junk rare words (too easy to reject). The $3/4$ is a heuristic middle, never derived.

**Relation to NCE.** Noise Contrastive Estimation also does data-vs-noise logistic regression, but it needs the *numerical probabilities* $P_n(w)$ in the loss, and it provably approximates the softmax. Negative sampling drops the probability term and keeps only samples. It no longer approximates any softmax. The authors' argument: we do not care, we want vectors, not a language model.

**Hierarchical softmax (the alternative baseline).** Build a binary Huffman tree with words as leaves — frequent words get short codes. Predicting a word = a sequence of left/right binary decisions down the path:

$$p(w \mid w_I) = \prod_{j=1}^{L(w)-1}\sigma\big(\llbracket n(w,j+1)=\mathrm{ch}(n(w,j))\rrbracket \cdot {v'_{n(w,j)}}^\top v_{w_I}\big)$$

Cost is $\approx \log_2 W$ instead of $W$, and it stays a proper normalised distribution ($\sum_w p(w\mid w_I)=1$). Here $v'$ lives on the *inner nodes*, not on words.

**Subsampling of frequent words.** Discard each token $w_i$ with probability

$$P(w_i)=1-\sqrt{\frac{t}{f(w_i)}}$$

with $t \approx 10^{-5}$ and $f$ the word's corpus frequency. Words below the threshold are never dropped; above it, dropping gets aggressive but the frequency *ranking* is preserved. Chosen heuristically — the authors say so outright.

**Phrase detection.** Score every bigram:

$$\mathrm{score}(w_i,w_j)=\frac{\mathrm{count}(w_iw_j)-\delta}{\mathrm{count}(w_i)\times\mathrm{count}(w_j)}$$

The $\delta$ discount stops rare-word pairs from scoring high on a single lucky co-occurrence. Bigrams over a threshold become single tokens. Run 2–4 passes with a decreasing threshold so longer phrases ("San Jose Mercury News") accrete from shorter ones.

**Setup.** 1B-word internal Google news corpus, drop words seen fewer than 5 times → 692K vocabulary, 300 dimensions, window 5. The best phrase model: 33B words, hierarchical softmax, 1000 dimensions, whole sentence as context, one day of training.

## Ablation Studies and Experiments

**Word analogies** (Table 1, 300d, 1B words). Solve "Germany : Berlin :: France : ?" by finding the nearest vector by cosine distance to $\mathrm{vec}(\text{Berlin}) - \mathrm{vec}(\text{Germany}) + \mathrm{vec}(\text{France})$.

| Method | Time (min) | Syntactic | Semantic | Total |
|---|---|---|---|---|
| NEG-5 | 38 | 63 | 54 | 59 |
| NEG-15 | 97 | 63 | 58 | 61 |
| HS-Huffman | 41 | 53 | 40 | 47 |
| NCE-5 | 38 | 60 | 45 | 53 |
| *with $10^{-5}$ subsampling* | | | | |
| NEG-5 | **14** | 61 | 58 | 60 |
| NEG-15 | 36 | 61 | 61 | **61** |
| HS-Huffman | 21 | 52 | 59 | 55 |

Reads: negative sampling beats hierarchical softmax by 12–14 points, and edges out NCE by 6 points despite being the *simpler* thing. Subsampling gives NEG-5 the same accuracy in 14 minutes instead of 38 — 2.7× faster, free. HS gains 8 points from subsampling; it needed it more.

**Phrase analogies** (Table 3, 3218 examples, e.g. "Montreal : Montreal Canadiens :: Toronto : ?").

| Method | No subsampling | $10^{-5}$ subsampling |
|---|---|---|
| NEG-5 | 24 | 27 |
| NEG-15 | 27 | 42 |
| HS-Huffman | 19 | **47** |

**This is the interesting reversal.** On words, NEG wins and HS loses. On phrases with subsampling, HS goes from worst (19) to best (47), overtaking NEG-15 by 5 points. The paper calls this "surprising" and does not explain it. Plausible reason: phrases are rare tokens, and HS's tree structure shares parameters across the path, so rare leaves still get gradient from their ancestors — whereas under NEG a rare phrase's output vector is only updated when that phrase actually shows up. Practical takeaway: **the right sampling scheme depends on how heavy your tail is.**

Also note $k$ matters a lot more for phrases: NEG-5 → NEG-15 jumps 27 → 42 with subsampling.

**Scale.** 6B words → 66%. 33B words → 72%. More data is the single biggest lever, which is the [[The Bitter Lesson (essay)|bitter lesson]] arriving early.

**Qualitative nearest neighbours** (Table 4) agree: HS + subsampling gives better neighbours for rare phrases ("Alan Bean" → "moonwalker" under HS, vs "Rebbeca Naomi" under NEG-15).

**What did not work / was rejected:**
- Plain unigram and uniform noise distributions — both clearly worse than $U(w)^{3/4}$, on every task including language modelling (results not shown).
- Training on all n-grams as tokens — "too memory intensive", hence the greedy bigram scoring instead.
- NCE — works, but the extra machinery (needing the noise probabilities) buys nothing here; NEG-5 matched its speed at 53 → 59 total accuracy.

**Against prior work** (Table 6): Collobert & Weston 50d took 2 months, Turian 200d a few weeks, Mnih 100d 7 days. Skip-Phrase 1000d took **1 day** on 30B words, and its neighbours for "Havel" are "Vaclav Havel / president Vaclav Havel / Velvet Revolution" versus Collobert's "plauen / dzerzhinsky / osterreich". Two to three orders of magnitude more data, in less wall-clock time.

## Worth Remembering

**Additive compositionality.** $\mathrm{vec}(\text{Russian}) + \mathrm{vec}(\text{river}) \approx \mathrm{vec}(\text{Volga River})$; $\mathrm{vec}(\text{German}) + \mathrm{vec}(\text{airlines}) \approx \mathrm{vec}(\text{Lufthansa})$. The explanation the authors give is genuinely nice: a word vector sits in the *logit* of the softmax, so it is a log-probability-shaped object over contexts. Adding two vectors ≈ multiplying two context distributions ≈ a soft AND. Words that both "Russian" and "river" would predict get high scores; everything else dies.

**Limitations the authors admit.**
- The subsampling formula is a heuristic with no justification beyond "it worked".
- The phrase scoring is deliberately not compared against the existing phrase-extraction literature — out of scope.
- "The choice of training algorithm and hyper-parameter selection is a task specific decision." No universal recipe. The four levers that mattered most: architecture, vector size, subsampling rate, window size.
- Negative sampling gives you no valid probabilities. You cannot use these scores as a language model.

**Practical caveats.**
- Two embedding tables. Most people throw away $v'$ and use $v$, but that is a choice, not a law; summing or concatenating them sometimes helps.
- $k$ and the $3/4$ exponent are the two knobs people forget to tune when they port negative sampling to a new domain. The $3/4$ is tuned for Zipfian word frequencies — if your item popularity distribution differs, retune it.
- Sampled softmax in recommenders needs a **logQ correction** to debias the popularity-based sampling; word2vec does not do this and does not need to for representation quality, but you do if you want calibrated scores. Compare [[PinnerFormer- Sequence Modeling for User Representation at Pinterest]].

**Connections.** Later work (Levy & Goldberg) proved skip-gram with negative sampling is implicitly factorising a shifted pointwise-mutual-information matrix — so this is matrix factorisation in disguise, which links it back to the classical [[Recommender Systems - Evolution|recsys]] line. The lineage forward: the same "two towers, dot product, sampled negatives" recipe becomes retrieval, then contrastive learning, then CLIP. The softmax bottleneck this paper dodges is the same one that [[Attention Is All You Need|transformer]] LMs still pay in full at the output layer.

**Open question worth chasing:** why exactly does hierarchical softmax overtake negative sampling on rare phrases? The parameter-sharing story is a guess. If it holds, it argues for tree-structured or hierarchical output layers whenever your label distribution has a heavy tail — which is every recommender ever built.

## Links

Related: [[Efficient Estimation of Word Representations (word2vec)]] · [[A Neural Probabilistic Language Model (JMLR)]] · [[Cross Entropy]] · [[KL Divergence]] · [[Real-time Personalization using Embeddings for Search Ranking at Airbnb (KDD, best paper)]] · [[PinnerFormer- Sequence Modeling for User Representation at Pinterest]] · [[Recommender Systems - Evolution]] · [[Backpropagation]] · [[The Bitter Lesson (essay)]] · [[Linear Projection]] · [[BERT- Pre-training of Deep Bidirectional Transformers]]

New topics worth writing: Noise Contrastive Estimation, Hierarchical softmax and Huffman coding, Sampled softmax and logQ correction, Pointwise Mutual Information, Contrastive learning, Two-tower retrieval, Zipf's law and heavy-tailed vocabularies
