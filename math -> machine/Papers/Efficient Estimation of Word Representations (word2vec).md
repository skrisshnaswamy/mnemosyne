---
title: "Efficient Estimation of Word Representations (word2vec)"
authors: ["Mikolov et al."]
year: 2013
arxiv: "1301.3781"
url: https://arxiv.org/abs/1301.3781
priority: Must-Read
read_on: 2026-08-24
tags: [paper, llm, vision]
---
## The Core Idea

Before this paper, if you wanted word vectors you trained a language model — a neural net that predicts the next word — and grabbed the input embedding matrix as a side effect. That was [[A Neural Probabilistic Language Model (JMLR)|Bengio's NNLM]]. It worked, but it was slow, because the model had a big non-linear hidden layer in the middle. Cost per training example was dominated by $N \times D \times H$ (context words × vector size × hidden units). So people trained on ~30–300M words with 50–100 dimensional vectors and stopped there.

The insight here is blunt: **throw away the hidden layer.** The hidden layer is what makes a neural net powerful for *language modelling*, but you do not need language modelling. You need a cheap objective that forces words appearing in similar contexts to get similar vectors. A plain log-linear classifier — embedding lookup, sum, softmax — is enough. It is a much worse language model. It is a much better word-vector factory, because it is ~100× cheaper, so you can feed it 6 billion words and 1000 dimensions instead of 300M words and 100 dimensions.

This is [[The Bitter Lesson (essay)|the bitter lesson]] in miniature: simpler model + far more data beats cleverer model + less data. CBOW at 1000 dims on 6B words hits 63.7% total accuracy on their analogy set; the NNLM at 100 dims on the same 6B words hits 50.8% and takes 14 days × 180 cores instead of 2 days × 140 cores.

The second contribution is an **evaluation trick that made the field measurable**. Instead of eyeballing nearest neighbours ("France is close to Italy, nice"), they built 19,544 analogy questions of the form *a is to b as c is to ?*, answered by

$$X = \text{vec}(b) - \text{vec}(a) + \text{vec}(c)$$

and taking the nearest word to $X$ by cosine distance (excluding $a,b,c$). Exact match only — synonyms count as wrong. This turned "are these vectors good?" into a number, and revealed that the vector space has **linear structure**: the offset between *king* and *queen* is roughly the same direction as the offset between *man* and *woman*, and the offset between *Paris* and *France* roughly equals *Rome* minus *Italy*.

> [!NOTE] Linear regularity
> The property that semantic and syntactic relations show up as roughly constant *difference vectors* in embedding space, so analogies can be solved with addition and subtraction. Not designed in — it falls out of training. ^linear-regularity

## The Methodology

Two architectures, both with no hidden layer, both trained with SGD and [[Backpropagation|backprop]].

**CBOW (Continuous Bag-of-Words).** Take 4 words before and 4 words after the target. Look up each one's vector in a shared embedding matrix. **Average them** into one vector. Feed that to a softmax classifier that must predict the middle word. "Bag of words" because averaging destroys the order — the model has no idea which context word came first. Cost per example:

$$Q = N \times D + D \times \log_2(V)$$

**Skip-gram.** The mirror image. Take the current word's vector as input, and predict each surrounding word. For each training word, sample $R$ uniformly from $\{1,\dots,C\}$ (they use $C=10$), then use $R$ words before and $R$ after as targets — $2R$ separate classifications. Sampling $R$ this way is a cheap trick to **downweight distant words**: a word 8 positions away is only used when $R \geq 8$, so it appears in fewer training examples than a word 1 position away. Cost:

$$Q = C \times (D + D \times \log_2(V))$$

Skip-gram is roughly $C$ times more expensive than CBOW per word — 3 days vs 1 day for the same 783M-word run.

**Hierarchical softmax.** A full softmax over a 1M-word vocabulary is $D \times V$ per example — the whole cost of the model. They replace it with a **Huffman binary tree** over the vocabulary: each word is a leaf, and predicting a word means making a chain of binary decisions down the tree. Cost drops from $V$ to about $\log_2 V$. Huffman coding (short codes for frequent words) makes it better still — the expected depth is $\log_2(\text{unigram perplexity}(V))$, roughly a further 2× saving at $V = 10^6$. This matters here far more than it did for the NNLM, where the hidden layer was the bottleneck anyway.

**Training details that mattered:**
- Google News corpus, ~6B tokens, vocabulary capped at the 1M most frequent words.
- Starting learning rate 0.025, decayed **linearly to zero** over training.
- Early experiments: 3 epochs. Later experiments: **1 epoch**, because one pass over 2× the data beats 3 passes over 1× the data — same or better accuracy, less time.
- Large runs used DistBelief with 50–100 asynchronous model replicas and [[Momentum|Adagrad]]-style adaptive learning rates, sharing gradients through a central parameter server.

## Ablation Studies and Experiments

**Dimension vs data (CBOW, 30k vocab subset), Table 2.** Both axes matter, and you must scale them together:

| dim \ words | 24M | 98M | 391M | 783M |
|---|---|---|---|---|
| 50 | 13.4 | 18.6 | 22.5 | 23.2 |
| 300 | 23.2 | 35.3 | 43.7 | 45.9 |
| 600 | 24.0 | 36.5 | 46.6 | 50.4 |

Read the rows: at 50 dims, going from 24M to 783M words buys you 10 points and then flattens. At 600 dims the same data increase buys 26 points. This is the actual takeaway — the then-common practice of training 50-dim vectors on huge corpora was leaving most of the gain on the table. A proto-[[Scaling Laws for Neural Language Models|scaling law]], stated informally.

**Architecture comparison at 640 dims, same 320M-word LDC data (Table 3):**

| Model | Semantic | Syntactic | MSR syntactic |
|---|---|---|---|
| RNNLM | 9 | 36 | 35 |
| NNLM | 23 | 53 | 47 |
| CBOW | 24 | **64** | **61** |
| Skip-gram | **55** | 59 | 56 |

The split is the interesting bit. **CBOW wins on syntax, Skip-gram wins massively on semantics** (55 vs 24). Plausible reason: CBOW averages a tight local window, which encodes grammatical role well; Skip-gram sees each context word as a separate example over a wider window, which captures topical/semantic association. RNNLM vectors are bad at semantics (9%) despite the RNN being the stronger language model — good LM ≠ good embeddings.

**Against public vectors (Table 4).** Collobert–Weston 50-dim on 660M words: 11.0% total. Turian 200-dim on 37M: 1.8%. Huang 50-dim on 990M: 12.3%. Skip-gram 300-dim on 783M: **53.3%**. The gap is not subtle. Note their own NNLM at 100 dims on 6B words gets 50.8% — so a big chunk of the win is data scale, not architecture. The architecture is what *permits* the data scale.

**Epochs (Table 5).** 3-epoch Skip-gram 300d/783M: 53.3% in 3 days. 1-epoch Skip-gram 300d/1.6B: 53.8% in 2 days. More fresh data beats re-reading old data, and is faster.

**Best models (Table 6, DistBelief, 6B words):** CBOW 1000d = 63.7% (2 days × 140 cores); Skip-gram 1000d = 65.6% (2.5 days × 125 cores). NNLM at 1000 dims was not even attempted — too slow.

**What did not work / limits:**
- **Sentence Completion (MSR, 1040 items).** Skip-gram alone scores 48.0%, *worse* than plain average-LSA similarity (49) and well below the RNNLM ensemble (55.4). Only combining Skip-gram scores with RNNLM scores gives a new SOTA 58.9%. So these vectors are not a language model, and pretending they are one loses.
- Exact-match scoring means the ceiling is below 100%. The models see no morphology, so *swimming → swam* has to be memorised from co-occurrence alone. Even on the hand-picked good examples in Table 8, exact-match accuracy is only ~60%.
- Averaging **ten** example pairs to define the relation vector, instead of one, adds ~10 absolute points. So single-pair analogy arithmetic is noisier than the headline suggests — the direction estimate is the weak link.

## Worth Remembering

- The paper *never actually writes the loss down explicitly*. It is [[Cross Entropy|cross-entropy]] of the true word under a hierarchical softmax, with a single [[Linear Projection|linear projection]] and no non-linearity. The full objective, plus negative sampling and subsampling of frequent words, arrives in the NIPS 2013 follow-up ("Distributed Representations of Words and Phrases"). If you have implemented word2vec, you almost certainly implemented the *sequel*, not this paper — negative sampling is not here.
- The follow-up note says the released single-machine C++ code hit **billions of words per hour**, far faster than anything reported in the paper's own tables. The tables understate the method.
- Two embedding matrices exist (input and output/tree-node); people use the input one. Nobody explains why that is the right choice.
- Averaging context vectors (CBOW) is the ancestor of every "mean-pool your token embeddings" baseline. It is also exactly what [[Attention Is All You Need|attention]] later replaced — attention is a *learned weighted* average of context instead of a uniform one, and it keeps position information that CBOW throws away.
- The syntax/semantics split by architecture is a useful practical rule: if your downstream task is POS-ish or morphological, CBOW; if it is topical or retrieval-ish, Skip-gram.
- Everything here is a **static** embedding — one vector per word type, so *bank* (river) and *bank* (money) collapse into one point. Fixing that is the whole motivation for [[BERT- Pre-training of Deep Bidirectional Transformers|BERT]]-style contextual embeddings. Note the direct lineage though: CBOW's "predict the middle word from both sides" is [[BERT- Pre-training of Deep Bidirectional Transformers#Objective 1: Masked LM|masked language modelling]] with a bag of words instead of a transformer.
- Practical caveat: the analogy benchmark is a proxy, not a goal. The authors say only that they "believe" usefulness correlates with it. Later work found the analogy task can be partly gamed by nearest-neighbour effects, since you exclude the three input words and the answer is often just near $c$.
- The recipe — CBOW on a sequence of tokens — is why item2vec, prod2vec and the [[Real-time Personalization using Embeddings for Search Ranking at Airbnb (KDD, best paper)|Airbnb listing embeddings]] exist. Swap "sentence" for "user session" and the method transfers unchanged.

## Links

Related: [[A Neural Probabilistic Language Model (JMLR)]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Real-time Personalization using Embeddings for Search Ranking at Airbnb (KDD, best paper)]] · [[Scaling Laws for Neural Language Models]] · [[The Bitter Lesson (essay)]] · [[Cross Entropy]] · [[Linear Projection]] · [[Backpropagation]] · [[Attention Is All You Need]] · [[Long Short-Term Memory (Neural Computation)]] · [[Recommender Systems - Evolution]] · [[Foundation Models]]

New topics worth writing: Hierarchical softmax and Huffman-tree output layers, Negative sampling / noise-contrastive estimation, Adagrad, Latent Semantic Analysis, Cosine similarity, Static vs contextual embeddings, Word analogy benchmarks and their flaws, DistBelief and asynchronous SGD, item2vec / prod2vec, GloVe and matrix-factorisation views of word2vec
