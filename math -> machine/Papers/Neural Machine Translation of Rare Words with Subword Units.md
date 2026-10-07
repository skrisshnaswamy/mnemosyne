---
title: Neural Machine Translation of Rare Words with Subword Units
authors:
  - Sennrich et al.
year: 2016
arxiv: "1508.07909"
url: https://arxiv.org/abs/1508.07909
priority: Must-Read
read_on: 2026-09-24
tags:
  - paper
  - transformers
  - llm
  - bpe-paper
---
## The Core Idea

Translation has no fixed word list. New names, new compounds, new typos appear forever. But a 2015-era neural translator had a fixed output vocabulary of 30k–50k words, because the softmax over the vocabulary cost time and memory proportional to its size. Everything outside the list became one symbol: `UNK`.

The patch everyone used was a back-off dictionary. Produce `UNK`, then look up the aligned source word in a separate word-to-word dictionary and paste in a translation, or just copy the source word across. Two assumptions hide in that: that source and target words line up one-to-one, and that copying is a sensible answer. Both fail. German *Abwasserbehandlungsanlage* is one word for English "sewage water treatment plant" — no one-to-one alignment exists. And copying "Mirzayeva" into a Russian sentence gives you Latin letters in a Cyrillic text.

The fix: stop working with words. Chop rare words into pieces the network has seen before, and let the network do the whole job — no dictionary, no `UNK`, no back-off path at all.

Why pieces work is a claim about language, not about networks. Many rare words are *transparent*: you can translate them without ever having seen them, from their parts. In a hand-count of 100 rare German tokens the authors found 56 compounds, 21 names, 6 loanwords, 5 transparent affixations, 1 number, 1 code identifier. Compounds translate part by part. Names transliterate letter group by letter group. Loanwords (*claustrophobia* → *Klaustrophobie*) follow regular character rules.

The second contribution is the piece-finding algorithm: **byte pair encoding**, lifted from a 1994 data-compression paper and pointed at text segmentation.

> [!NOTE] Byte pair encoding (BPE)
> Start with every word written out as single characters. Repeatedly find the most frequent adjacent symbol pair in the corpus and glue it into one new symbol. Stop after $k$ merges. Final vocabulary size = number of distinct characters + $k$. ^byte-pair-encoding

BPE is the right shape for this problem for one reason: it is **open-vocabulary with a fixed vocabulary size**. Any string whatsoever can be encoded, because the fallback is always individual characters, which are all in the vocabulary. But the size of the symbol table is a knob you set, so the softmax stays cheap. Character-level models are also open-vocabulary, but they blow up sequence length (550m tokens vs 100m for words on the German data), and long sequences hurt an RNN badly — information has to travel further. BPE sits in between at 112m tokens with 63k symbol types.

What it unlocks, beyond this paper: this is the tokenizer under [[Tokenization|nearly every language model since]]. GPT, RoBERTa, everything. A compression algorithm from a C programming magazine became the front door of modern NLP.

## The Methodology

**The translation model is not new.** It is Bahdanau-style encoder-decoder with [[Attention|attention]]: a bidirectional GRU encoder producing annotation vectors $h_j$ (forward and backward states concatenated), and a GRU decoder that predicts $y_i$ from its hidden state $s_i$, the previous word $y_{i-1}$, and a context vector $c_i = \sum_j \alpha_{ij} h_j$. The alignment weights $\alpha_{ij}$ come from a small feedforward net trained jointly by [[Backpropagation|backprop]]. See [[Seq2Seq models]] for the family. The paper changes **what a token is**, nothing else.

That is the point worth holding onto: the whole gain comes from the input/output alphabet, not the architecture.

**Learning the merges.**

1. Split the corpus into words. Write each word as space-separated characters plus an end-of-word marker `·`. Keep a frequency count per word type — you work on the type dictionary, not the running text, which makes this cheap.
2. Count every adjacent symbol pair, weighted by word frequency.
3. Take the most frequent pair `(A, B)`. Replace every occurrence with the merged symbol `AB`. Record the merge.
4. Repeat $k$ times.

Pairs are never merged across word boundaries, so a symbol can never span two words.

The toy example from the paper, on the dictionary `{low:5, lower:2, newest:6, widest:3}`, learns in order: `r·→r·`, `l o→lo`, `lo w→low`, `e r·→er·`.

**Applying the merges at test time.** Split the new word into characters, then replay the learned merge list in the order it was learned. Unseen word `lower` becomes `low er·`. This always terminates and always succeeds — worst case you get single characters.

The end-of-word marker `·` is doing real work. It lets you undo the segmentation after translation and recover the original whitespace, and it distinguishes `er` inside a word from `er` finishing one.

> [!NOTE] Only one hyperparameter
> BPE has exactly one dial: the number of merge operations $k$. Final vocabulary $= |\text{characters}| + k$. Small $k$ → near-character model, long sequences. Large $k$ → whole frequent words become single symbols, short sequences, bigger softmax. ^bpe-one-knob

**Joint BPE.** Two options. Learn separate merge tables for source and target, or learn one table on the union of both vocabularies. The joint version wastes some vocabulary but makes segmentation *consistent across languages*, so the network can learn "this source piece maps to that target piece". For English→Russian the alphabets differ, so they transliterate Russian into Latin with ISO-9, learn joint merges, then transliterate the merge rules back into Cyrillic.

**Training setup.** WMT15. English→German: 4.2m sentence pairs, ~100m tokens. English→Russian: 2.6m pairs, ~50m tokens. Tokenised and truecased with Moses. Dev set newstest2013, test on newstest2014/2015.

Hidden layer 1000, embedding layer 620. Adadelta, minibatch 80, reshuffled each epoch. Train ~7 days, take the last 4 checkpoints (saved every 12h), continue each for 12h with the embedding layer frozen. Two independent runs per setting, one with [[On the difficulty of training Recurrent Neural Networks|gradient clipping]] at 5.0, one at 1.0 — the 1.0 runs generally gave better single models. Report the best-on-dev single model, plus an ensemble of all 8. Beam size 12, probabilities length-normalised.

**Metrics.** BLEU (`mteval-v13a.pl`), chrF3 (character n-gram F-score, recall-biased, correlates better with humans for translation *out of* English), and — the one that matters here — **unigram $F_1$**, the harmonic mean of clipped unigram precision and recall, reported separately for all words, rare words (outside the top 50k in training), and OOVs (never seen in training).

## Ablation Studies and Experiments

**Segmentation statistics first** (German side, 100m tokens, `#UNK` measured on newstest2013):

| segmentation | # tokens | # types | # UNK |
|---|---|---|---|
| none (words) | 100m | 1,750,000 | 1079 |
| characters | 550m | 3,000 | 0 |
| character bigrams | 306m | 20,000 | 34 |
| character trigrams | 214m | 120,000 | 59 |
| compound splitting | 102m | 1,100,000 | 643 |
| Morfessor | 109m | 544,000 | 237 |
| hyphenation | 186m | 404,000 | 230 |
| **BPE (59.5k merges)** | **112m** | **63,000** | **0** |
| **BPE joint (89.5k merges)** | **111m** | **82,000** | **32** |
| char bigrams + 50k shortlist | 129m | 69,000 | 34 |

The linguistically motivated segmenters — compound splitting, Morfessor, hyphenation — **all failed the actual requirement**. They cut the vocabulary only moderately (still 400k–1.1m types) and still leave hundreds of unknown tokens. They are conservative by design, because they were built for phrase-based SMT where being wrong about a split costs you. Here you need aggression.

**English→German, newstest2015:**

| system | BLEU (single) | BLEU (ens-8) | chrF3 (ens-8) | $F_1$ all | $F_1$ rare | $F_1$ OOV |
|---|---|---|---|---|---|---|
| syntax-based SMT | 24.4 | – | 55.3 | 59.1 | 46.0 | 37.7 |
| WUnk (no back-off) | 20.6 | 22.8 | 48.9 | 56.7 | 20.4 | 0.0 |
| WDict (back-off dict) | 22.0 | 24.2 | 52.4 | 58.1 | 36.8 | 36.8 |
| C2-50k (char bigram + shortlist) | **22.8** | **25.3** | 53.5 | 58.4 | 40.5 | 30.9 |
| BPE-60k | 21.5 | 24.5 | 53.9 | 58.4 | 40.9 | 29.3 |
| BPE-J90k (joint) | 22.8 | 24.7 | **54.1** | **58.5** | **41.8** | 33.6 |

**English→Russian, newstest2015:**

| system | BLEU (ens-8) | chrF3 (ens-8) | $F_1$ rare | $F_1$ OOV |
|---|---|---|---|---|
| phrase-based SMT | 24.3 | 53.8 | 31.3 | 16.5 |
| WUnk | 22.4 | 49.9 | 25.2 | 0.0 |
| WDict | 22.8 | 51.0 | 26.5 | 6.6 |
| C2-50k | **24.1** | 51.6 | 27.8 | 17.4 |
| BPE-60k | 23.6 | 52.7 | 29.7 | 15.6 |
| BPE-J90k | **24.1** | **53.0** | **29.7** | **18.3** |

Headline gains over WDict: **+1.1 BLEU EN→DE, +1.3 BLEU EN→RU**; chrF3 +0.6 to +2.0.

**The result that actually diagnoses the mechanism.** Look at EN→DE OOV $F_1$: WDict scores **36.8**, better than BPE-60k's 29.3. The back-off dictionary wins on unseen words for this language pair — because most OOVs are names, English and German share an alphabet, and copying is simply correct. Now look at EN→RU OOV: WDict collapses to **6.6** while BPE-J90k gets **18.3**. Copying Latin letters into Russian is useless; the subword model actually transliterates. So the subword advantage is largest exactly where copying is not an option. This is the cleanest evidence in the paper that the network is learning a character-level *mapping*, not memorising pairs.

**Where the real win is: rare words, not OOVs.** OOVs are a small slice (EN→DE: $n=1168$ of 44,085 test words). Rare-but-seen words are bigger ($n=2900$) and that is where all subword systems beat WDict cleanly: 36.8 → 41.8.

**The vocabulary-size ablation (Figure 2, the most interesting experiment).** They add `C2-3/500k` — character bigrams but with a 500k word shortlist, i.e. same target vocabulary as WDict. Plot unigram $F_1$ against training-set frequency rank. Below rank 50k, every system is identical (same representation, same scores). Between rank 50k and 500k, `C2-3/500k` **degrades steadily** while `C2-50k` — which represents that whole band as subwords — stays flat. At rank 500k, `C2-3/500k` switches to subwords and its performance *recovers*.

Read that again: **shrinking the vocabulary improved accuracy.** Rank 50k corresponds to frequency 60 in training; rank 500k to frequency 2. A word seen twice cannot get a useful embedding. Its constituent subwords have been seen thousands of times. Sparsity, not capacity, was the binding constraint. This is the finding that kills the "just use a bigger vocabulary" approach on its own terms.

**Joint vs separate BPE, measured.** EN→DE OOV precision/recall: BPE-60k gets 32.4% / 26.6%; BPE-J90k gets 38.6% / 29.8%. Both improve. The diagnosed cause is segmentation inconsistency across languages. Concrete failure: BPE-60k splits `Mirz|ayeva` on the English side but `Мир|за|ева` on the Russian, so the network cannot learn a piece-to-piece map. For `rakfisk` it produces `пра|ф|иск` — an inserted `п`, a dropped `к` — traced to training pairs like `p|rak|ri|ti → пра|крит|и`, from which the network wrongly learned `rak → пра`. Joint BPE segments it as `рак|ф|иска`, correct.

**Precision/recall trade, hidden by $F_1$.** WDict has high OOV precision (60.6%) but low recall (26.5%) — it only emits copies, and copies are usually right. C2-50k produces the most OOV words, precision drops to 29.1%, but recall rises to 33.0%. The subword models take more shots.

**Things that did not work or disappointed:**
- Unigram (pure character) representation "performed poorly in preliminary experiments"; they reported bigrams instead.
- The classic morphological segmenters (Morfessor, compound splitting, hyphenation) — not aggressive enough, do not close the vocabulary.
- BPE-60k's single-model BLEU on EN→DE is **21.5**, worse than WDict's 22.0. The BLEU story is not clean at the single-model level.
- Oversplitting still sometimes works: `research → Fo|rs|ch|un|g` still yields a correct translation, so the segmentation being linguistically wrong is not fatal.
- Genuine sparsity is not fixed. `asinine` should be German *dumm*; the subword models produce *Asinin-Situation* — a plausible-looking fake loanword. Pieces let you spell anything, including things that are wrong.
- Best BLEU (25.3, C2-50k) and best chrF3 (54.1, BPE-J90k) disagree on which system won. BLEU is precision-biased, chrF3 recall-biased.
- Against the syntax-based SMT baseline: better on BLEU, **worse on chrF3** (54.1 vs 55.3). EN→RU still trails phrase-based SMT slightly on ens-8.

## Worth Remembering

**The honest framing of the result.** Rare and unseen words are only 9–11% of the test sets. The authors say outright that BLEU and chrF3 probably *understate* the improvement, because rare words carry a lot of a sentence's meaning but few of its n-grams. Conversely, if you only read the BLEU column you would conclude the effect is about one point. The unigram-$F_1$-by-frequency-rank plot is the real evidence, and the paper's argument depends on you accepting that metric.

**Variance is admitted.** Up to 1 BLEU spread between models on dev. They pick best-on-dev out of 8, which they acknowledge is a stabilising hack, and say controlling for randomness "deserves further attention". Worth remembering when reading any 2015–2017 NMT number — see [[On the Difficulty of Evaluating Baselines]] for the general disease.

**Vocabulary size was chosen arbitrarily.** 59.5k merges, 89.5k joint — picked to match prior work's vocabulary sizes, not optimised. The authors flag learning it automatically as future work. Nobody really did; 32k/50k became convention by imitation.

**What BPE does not give you.** The merges are chosen by corpus frequency alone. They have no idea about morphemes. `Forsch|ungsinstitu|ten` is not a linguistic decomposition. It works anyway, which is itself a [[The Bitter Lesson (essay)|bitter-lesson]]-shaped result: the frequency-counting algorithm beat the linguistically informed segmenters at the thing linguists designed theirs to do.

**Caveats if you are using BPE today.** It is greedy and deterministic — one segmentation per word, no probability over segmentations (later work adds this). The vocabulary is frozen at training time; a domain shift means your text gets shredded into characters and sequences get long. It handles numbers badly, since digit pairs merge by frequency with no notion of place value. And BPE on bytes rather than characters (the modern default) removes the character-set assumption entirely.

**Follow-up questions worth chasing.** The paper's suggestion of *bilingually informed* segmentation — pieces chosen to be alignable across languages — is the natural extension of the joint-BPE result, and joint BPE is only a crude version of it. The constraint they note is real: at test time you do not have the target text, so segmentation must be decidable from the source alone.

**Connection to the reader's line of work.** The frequency-rank plot is the same story as the long tail in recommenders: items seen twice cannot support an embedding, and the fix is to represent them compositionally from pieces that are seen often. [[Compositional Embeddings Using Complementary Partitions (QR trick)|The QR trick]] and [[Recommender Systems with Generative Retrieval (TIGER)|semantic IDs]] are literally BPE's idea transplanted onto item IDs — replace a sparse atomic ID with a short sequence of shared, well-estimated sub-symbols. TIGER's RQ-VAE codes are a learned version of the same move.

## Links

Related: [[Tokenization]] · [[Attention]] · [[Seq2Seq models]] · [[Embeddings]] · [[Recommender Systems with Generative Retrieval (TIGER)]] · [[Compositional Embeddings Using Complementary Partitions (QR trick)]] · [[Efficient Estimation of Word Representations (word2vec)]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[On the difficulty of training Recurrent Neural Networks]] · [[The Bitter Lesson (essay)]] · [[Long Short-Term Memory (Neural Computation)]] · [[Attention Is All You Need]] · [[Curse of Dimensionality]]

New topics worth writing: Byte Pair Encoding, BLEU and chrF, open-vocabulary modelling, transliteration, SentencePiece and unigram LM tokenisation, subword regularisation, out-of-vocabulary handling, long-tail sparsity in embedding tables
