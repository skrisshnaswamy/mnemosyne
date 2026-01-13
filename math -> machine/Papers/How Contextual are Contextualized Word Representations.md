---
title: "How Contextual are Contextualized Word Representations?"
authors: ["Ethayarajh"]
year: 2019
arxiv: "1909.00512"
url: https://arxiv.org/abs/1909.00512
priority: Must-Read
read_on: 2026-08-24
tags: [paper, llm]
---
## The Core Idea

Everyone had switched from static word vectors (one vector per word, forever) to contextual ones (a fresh vector per word per sentence). Nobody had measured *how much* the vector actually changes with context. This paper measures it, geometrically, layer by layer, for ELMo, BERT, and GPT-2.

Two findings matter.

**First: the vectors are squashed into a narrow cone.** Take two random words, from random sentences, and compute the cosine similarity of their vectors. If the space were directionally uniform, you would get ~0. Instead you get big positive numbers — around 0.6 in GPT-2's middle layers, and *almost 1.0* in GPT-2's last layer. Two unrelated words point in nearly the same direction. This is called **anisotropy**, and it gets worse the higher you go.

This wrecks naive interpretation. If a word's vectors across contexts have cosine similarity 0.95, that sounds "not contextual at all" — but if random word pairs already sit at 0.99, then 0.95 is actually *very* contextual. So every measure in the paper is reported minus a per-layer random baseline. That subtraction is the methodological contribution.

**Second: words are not being sorted into a small menu of senses.** After the anisotropy correction, take all the vectors of one word across all its contexts and ask how much of their variance the top principal direction explains. Answer: **under 5%, in every layer of every model.** If BERT were secretly assigning "bank(river)" or "bank(money)" from a finite list, one direction would explain a lot. It does not. Contextualisation is continuous, not discrete.

The kicker on that point: **stopwords are the most context-specific words of all.** The lowest self-similarity in ELMo belongs to `and`, `of`, `'s`, `the`, `to` — words with essentially no polysemy. So what drives vector variation is *the variety of contexts a word shows up in*, not how many dictionary senses it has.

> [!NOTE] Anisotropy
> A set of vectors is anisotropic if it is not directionally uniform — it occupies a narrow cone rather than spreading over the sphere. Diagnostic: the expected cosine similarity between two randomly chosen vectors is far from zero. ^anisotropy

> [!NOTE] Self-similarity
> For a word $w$ appearing in $n$ different sentences, the average cosine similarity between its $n$ contextual vectors in layer $\ell$. Equals 1 if the layer does no contextualisation; drops toward the random baseline as contextualisation increases. ^self-similarity

## The Methodology

No training. This is pure measurement on frozen pre-trained models.

**Models.** ELMo (2 biLSTM layers), BERT base cased (12 layers), GPT-2 (12 layers). BERT base was chosen so the depth and width match GPT-2. Layer 0 = the input embedding layer, included deliberately as a non-contextual control.

**Data.** Sentences from SemEval Semantic Textual Similarity 2012–2016. Chosen because the same word recurs in many different sentences — e.g. `dog` in *"A panda dog is running on the road"* and *"A dog is trying to get bacon off his back"*. Words appearing in fewer than 5 unique contexts are dropped.

**Three measures.** Let $f_\ell(s,i)$ be the layer-$\ell$ vector of the token at index $i$ of sentence $s$.

1. Self-similarity — how much a word's vector moves across contexts:
$$\text{SelfSim}_\ell(w)=\frac{1}{n^2-n}\sum_j\sum_{k\neq j}\cos\big(f_\ell(s_j,i_j),f_\ell(s_k,i_k)\big)$$

2. Intra-sentence similarity — how tightly the words *inside one sentence* cluster. With $\vec{s_\ell}=\frac{1}{n}\sum_i f_\ell(s,i)$ the mean of the sentence's word vectors:
$$\text{IntraSim}_\ell(s)=\frac{1}{n}\sum_i \cos(\vec{s_\ell}, f_\ell(s,i))$$

3. Maximum explainable variance — the ceiling on how well one static vector could stand in for all of a word's contextual vectors. Stack the $n$ occurrence vectors into a matrix, take its singular values $\sigma_1 \dots \sigma_m$:
$$\text{MEV}_\ell(w)=\frac{\sigma_1^2}{\sum_i \sigma_i^2}$$

**The anisotropy correction.** For each layer, estimate a baseline from 1K uniformly random word occurrences: for the similarity measures, the mean pairwise cosine; for MEV, the variance share of the first principal component of a *random assortment* of words. Subtract:
$$\text{SelfSim}^*_\ell(w)=\text{SelfSim}_\ell(w)-\mathbb{E}_{x,y\sim U(\mathcal{O})}\big[\cos(f_\ell(x),f_\ell(y))\big]$$
Every number quoted in the paper is this adjusted version unless stated otherwise. Baselines are computed *per layer*, because anisotropy varies wildly by depth.

## Ablation Studies and Experiments

**Anisotropy by layer (Fig. 1).** Non-zero in every layer of every model except one: ELMo's input layer, which is a context-free character-level embedding, and is genuinely isotropic. GPT-2 sits near 0.6 for layers 2–8, then rises exponentially through layer 12 to near-perfect similarity between random words. BERT's penultimate layer is *more* anisotropic than its final layer — a local exception to the monotone trend. Every contextualised hidden layer is more anisotropic than its own input layer, which is the evidence for anisotropy being a by-product of contextualisation itself, not of the data.

**Self-similarity by layer (Fig. 2).** Falls monotonically with depth in all three models. GPT-2's last layer is close to maximally context-specific. Direct analogy the author draws: lower CNN layers see edges, upper layers see class-specific parts (Yosinski 2014); lower LSTM layers are generic, upper ones task-specific (Liu 2019a).

**Intra-sentence similarity (Fig. 3) — the one measure where the three models sharply disagree.**

- **ELMo:** goes *up* with depth. Words in a sentence converge toward each other. A crude contextualisation — Firth's distributional hypothesis applied at the sentence level.
- **BERT:** goes *down* with depth (layer 12 excepted), but stays above the random baseline in every layer, above 0.20 for all but one. Words are informed by their sentence without being collapsed into it.
- **GPT-2:** essentially **0** after adjustment in most layers. Two words in the same sentence are no more alike than two random words. The highest intra-sentence similarity in GPT-2 is at the *input* layer, which does no contextualisation at all.

So high intra-sentence similarity is *not* required for good contextualisation — GPT-2 works fine without it. High anisotropy, by contrast, accompanies context-specificity everywhere. The author is honest that he cannot attribute the ELMo/BERT/GPT-2 difference to any specific architectural choice and leaves it open.

**MEV (Fig. 4).** Under 5% adjusted in every layer of every model. Many words have *raw* MEV below the anisotropy baseline — one vector explains more of the variance across all words than across all occurrences of a single word. Raw MEV is under 5% even unadjusted for ELMo and BERT; only GPT-2 shows a non-trivial raw ~30% for layers 2–11, and that is purely the anisotropy inflating it.

**The counter-intuitive positive result: PC static embeddings (Table 1).** Take a word's first principal component across contexts, call it that word's static vector, and run the classic benchmarks. Best row is **BERT layer 1**:

| Embedding | SimLex999 | MEN | WS353 | RW | Google | MSR |
|---|---|---|---|---|---|---|
| GloVe | 0.194 | 0.216 | 0.339 | 0.127 | 0.189 | 0.312 |
| FastText | 0.239 | 0.239 | 0.432 | 0.176 | 0.203 | 0.289 |
| BERT L1 | **0.315** | 0.200 | 0.394 | **0.208** | **0.236** | **0.389** |
| BERT L12 | 0.233 | 0.082 | 0.325 | 0.144 | 0.184 | 0.307 |
| GPT-2 L12 | 0.140 | −0.009 | 0.113 | 0.163 | 0.020 | 0.021 |

**What does not work:** upper layers, and GPT-2 at any layer. GPT-2 layer 12 scores −0.009 on MEN and 0.021 on MSR analogies — worse than useless. The pattern is exact and inverse to context-specificity: the more context-specific a layer's representations, the worse its principal component works as a static word vector. Which makes sense — averaging over contexts is only meaningful if the vector was mostly about the word to begin with. MEN is the one benchmark where FastText still beats everything.

## Worth Remembering

- **The 5% number is a ceiling, not an estimate of GloVe.** The first PC is the *best possible* single vector for that word. There is no guarantee any actually-trained static embedding is close to it. So "static embeddings are a bad substitute" is understated, not overstated.
- **The methodological lesson generalises far beyond this paper:** never read a raw cosine similarity from a deep model without a random-pair baseline from the *same layer*. This applies to embedding-based retrieval, dedup thresholds, similarity dashboards — anywhere you compare `cos(a,b)` to a fixed cutoff. This paper is the reason "anisotropy correction" became standard hygiene, and it sets up the sentence-embedding fixes that followed (BERT-flow, whitening, SimCSE).
- Connects tightly to [[Representation Degeneration Problem in Training NLMs]] — same narrow-cone phenomenon, but that paper explains *why* it happens from the softmax training objective and the gradients pushing rare-word embeddings in a shared negative direction. Read them as a pair: one measures, one explains.
- **Two open directions the author names.** (1) Add an anisotropy penalty to the language-modelling loss, since Mu et al. showed that just subtracting the mean from static embeddings gives large downstream gains. (2) Ship the PC static embeddings in production as a cheap stand-in for BERT — you get GloVe-shaped deployment cost with better-than-GloVe benchmark numbers. Practical caveat: you must pick a *low* layer, and you must have a corpus of contexts to average over.
- **Caveat on the data.** SemEval STS sentences are short and somewhat unusual. Whether these geometry numbers hold on long documents or code is untested here.
- The "stopwords are most context-specific" result deserves a moment. It kills the naive story that contextual embeddings are doing word-sense disambiguation. `the` has one sense and the wildest vectors.

## Links

Related: [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Representation Degeneration Problem in Training NLMs]] · [[Efficient Estimation of Word Representations (word2vec)]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[A Neural Probabilistic Language Model (JMLR)]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[Attention Is All You Need]] · [[Long Short-Term Memory (Neural Computation)]] · [[LeJEPA- Provable and Scalable Self-Supervised Learning]] · [[Linear Projection]] · [[Foundation Models]]

New topics worth writing: Anisotropy and isotropy in embedding spaces, Principal Component Analysis and singular values, Cosine similarity, ELMo, GloVe, Probing classifiers, All-but-the-top post-processing, Sentence embedding whitening (BERT-flow / SimCSE), Polysemy and word-sense disambiguation
