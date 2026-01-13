---
title: "SimCSE: Simple Contrastive Learning of Sentence Embeddings"
authors: ["Tianyu Gao", "Xingcheng Yao", "Danqi Chen"]
year: 2021
arxiv: "2104.08821"
url: https://arxiv.org/abs/2104.08821
priority: Must-Read
read_on: 2026-08-24
tags: [paper, transformers, self-supervised, vision]
---
## The Core Idea

Take one sentence. Push it through BERT twice. Because [[Regularization|dropout]] randomly zeroes different units on each pass, the two runs give two slightly different vectors. Call those two vectors a **positive pair**. Everything else in the mini-batch is a negative. Train with a contrastive loss. That is the whole unsupervised method.

The trick matters because text augmentation is hard. Images can be cropped, flipped, rotated, and still mean the same thing. Words are discrete — delete one word and you may have changed the meaning. The authors show that even *deleting a single word* costs you 6.6 points on STS-B (82.5 → 75.9). Dropout is an augmentation that lives in the **hidden representations** rather than in the token string, so it never breaks the sentence. It is the smallest possible perturbation that still produces two different vectors.

The second half of the paper is the supervised version: NLI (natural language inference) datasets already contain, for each premise, a hand-written sentence that must be true (*entailment*) and one that must be false (*contradiction*). Use the entailment sentence as the positive, and the contradiction sentence as an extra **hard negative** in the denominator. This is a much more natural use of NLI than the 3-way classification head that [[Sentence-BERT|SBERT]] used.

Result: on 7 semantic-textual-similarity benchmarks, average Spearman correlation with BERT-base goes from 72.05 (previous best unsupervised) to **76.25** unsupervised, and from 79.39 to **81.57** supervised. RoBERTa-large supervised hits **83.76**.

> [!NOTE] Dropout as minimal data augmentation
> Two forward passes of the *same* input with different dropout masks give two embeddings that differ only by noise in the hidden layers. No token is changed, so meaning is preserved perfectly, but the vectors are not identical — which is exactly what a contrastive loss needs. ^dropout-augmentation

---

## The Methodology

**Encoder.** A standard pre-trained BERT or RoBERTa ([[Attention Is All You Need|Transformer]] encoder). The sentence embedding is the `[CLS]` token, optionally passed through an MLP head. All parameters are fine-tuned — nothing is frozen.

**Loss.** The InfoNCE / in-batch-negatives objective (see [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)|CPC]]), which is just [[Cross Entropy|cross-entropy]] over cosine similarities. For a batch of $N$:

$$\ell_i = -\log \frac{e^{\mathrm{sim}(\mathbf{h}_i, \mathbf{h}_i^+)/\tau}}{\sum_{j=1}^{N} e^{\mathrm{sim}(\mathbf{h}_i, \mathbf{h}_j^+)/\tau}}$$

with $\mathrm{sim}$ = cosine similarity and $\tau$ = temperature.

**Unsupervised variant.** $x_i^+ = x_i$. The only difference between $\mathbf{h}_i$ and $\mathbf{h}_i^+$ is the dropout mask $z$:

$$\ell_i = -\log \frac{e^{\mathrm{sim}(\mathbf{h}_i^{z_i}, \mathbf{h}_i^{z_i'})/\tau}}{\sum_{j=1}^{N} e^{\mathrm{sim}(\mathbf{h}_i^{z_i}, \mathbf{h}_j^{z_j'})/\tau}}$$

The dropout is the *default* Transformer dropout ($p = 0.1$ on feed-forward layers and attention probabilities). Nothing extra is added. Data: 1 million random sentences from English Wikipedia, 1 epoch.

**Supervised variant.** Triples $(x_i, x_i^+, x_i^-)$ = (premise, entailment hypothesis, contradiction hypothesis). The contradiction of *every* example in the batch joins the denominator:

$$\ell_i = -\log \frac{e^{\mathrm{sim}(\mathbf{h}_i, \mathbf{h}_i^+)/\tau}}{\sum_{j=1}^{N}\left(e^{\mathrm{sim}(\mathbf{h}_i, \mathbf{h}_j^+)/\tau} + e^{\mathrm{sim}(\mathbf{h}_i, \mathbf{h}_j^-)/\tau}\right)}$$

Data: SNLI + MNLI, 314k triples, 3 epochs.

**Hyperparameters that mattered.**
- $\tau = 0.05$ with cosine similarity. Raw dot product scores 85.9, $\tau=0.05$ cosine scores 86.2, $\tau=1$ collapses to 64.0.
- Batch size 64 (unsup BERT-base) or 512 (supervised). Notably, SimCSE is **not** sensitive to batch size if you retune the learning rate — this contradicts SimCLR's claim that contrastive learning needs huge batches. The authors credit the pre-trained initialisation.
- LR $\in$ {1e-5, 3e-5, 5e-5}, grid-searched.
- One encoder, not two. Shared weights, not a dual encoder.

**Why it works, theoretically.** They lean on the alignment/uniformity framework (see [[Understanding Contrastive Learning through Alignment and Uniformity]]):

$$\ell_{\text{align}} = \mathbb{E}_{(x,x^+)}\|f(x)-f(x^+)\|^2, \quad \ell_{\text{uniform}} = \log \mathbb{E}_{x,y}\, e^{-2\|f(x)-f(y)\|^2}$$

Then a neat spectral argument. With normalised embeddings stacked in matrix $\mathbf{W}$, the negative term of the loss lower-bounds (via Jensen) $\frac{1}{\tau m^2}\mathrm{Sum}(\mathbf{W}\mathbf{W}^\top)$. Since embeddings are normalised, $\mathrm{tr}(\mathbf{W}\mathbf{W}^\top)$ — the sum of eigenvalues — is fixed at $m$. And when all entries of $\mathbf{W}\mathbf{W}^\top$ are positive (empirically true), $\mathrm{Sum}(\mathbf{W}\mathbf{W}^\top)$ upper-bounds the **largest** eigenvalue. So minimising the loss shrinks the top eigenvalue while the total is pinned — it *flattens the singular spectrum*. That is a direct attack on the anisotropy problem from [[How Contextual are Contextualized Word Representations]] and [[Representation Degeneration Problem in Training NLMs]], obtained for free from the contrastive objective rather than from a post-hoc whitening step.

---

## Ablation Studies and Experiments

**Discrete augmentation vs dropout** (STS-B dev, Spearman):

| Augmentation | Score |
|---|---|
| None (dropout only) | **82.5** |
| Crop 10% / 20% / 30% | 77.8 / 71.4 / 63.6 |
| Word deletion 10% / 20% / 30% | 75.9 / 72.2 / 68.2 |
| Delete exactly one word | 75.9 |
| Synonym replacement | 77.4 |
| MLM-replace 15% of words | 62.2 |

Every discrete edit hurts. MLM replacement is catastrophic.

**Dropout rate ablation** — the money table:

| $p$ | 0.0 | 0.01 | 0.05 | 0.1 | 0.15 | 0.2 | 0.5 | fixed 0.1 |
|---|---|---|---|---|---|---|---|---|
| STS-B | 71.1 | 72.6 | 81.1 | **82.5** | 81.4 | 80.5 | 71.0 | **43.6** |

"Fixed 0.1" = use dropout, but apply the *same mask* to both copies, so the two embeddings are identical. It collapses to 43.6. This is the cleanest evidence that the noise, not the dropout regularisation, is doing the work. The alignment/uniformity trace shows why: with identical pairs, uniformity improves but alignment falls off a cliff — a representation collapse. Standard dropout keeps alignment steady while uniformity improves.

**Objective ablation** (STS-B dev, one encoder vs two):

| Objective | 1 encoder | 2 encoders |
|---|---|---|
| Predict next sentence | 67.1 | 68.9 |
| Predict one of next 3 sentences | 67.4 | 68.8 |
| Delete one word | 75.9 | 73.1 |
| SimCSE | **82.5** | 80.7 |

Quick-Thought-style next-sentence prediction is 15 points behind. Sharing one encoder helps.

**Which supervised dataset?** (STS-B dev, full data)

| Source of positives | Score |
|---|---|
| Unsup. SimCSE (1M Wikipedia) | 82.5 |
| QQP (134k) | 81.8 |
| Flickr30k captions (318k) | 81.4 |
| ParaNMT back-translation (5M) | 78.7 |
| NLI **entailment** (314k) | 84.9 |
| NLI neutral (314k) | 82.9 |
| NLI contradiction as *positives* (314k) | 77.6 |
| NLI entailment + contradiction as hard negatives | **86.2** |
| + ANLI (52k) | 85.0 |

Notice: 5M ParaNMT pairs lose to 1M Wikipedia sentences with no labels at all. The reason given is lexical overlap — NLI entailment pairs share only 39% of their words (bag-of-words F1), versus 60% for QQP and 55% for ParaNMT. High overlap makes the positive too easy and the model learns surface matching.

**Hard-negative weighting.** They add a weight $\alpha$ on the hard negative term. $\alpha = 0.5 \to 86.1$, $\alpha = 1.0 \to 86.2$, $\alpha = 2.0 \to 86.2$. Plain $\alpha = 1$ is fine. Adding *neutral* hypotheses as extra hard negatives drops it to 85.3.

**Pooling** (STS-B dev):

| Pooler | Unsup | Sup |
|---|---|---|
| `[CLS]` w/ MLP | 81.7 | **86.2** |
| `[CLS]` w/ MLP at train, dropped at test | **82.5** | 85.8 |
| `[CLS]` w/o MLP | 80.9 | 86.2 |
| First-last layer average | 81.2 | 86.1 |

For supervised, pooling barely matters. For unsupervised, the quirk of training with the MLP head and throwing it away at test time buys ~0.8. (Same "projection head is for training only" pattern as SimCLR.)

**What did not work:**
- Dual encoder for supervised SimCSE: 86.2 → 84.2.
- Adding ANLI: 86.2 → 85.0.
- Combining supervised with the unsupervised dropout objective: no meaningful gain.
- Adding an auxiliary MLM loss $\ell + \lambda \ell^{\text{mlm}}$: helps **transfer** tasks (85.8 → 86.2 avg with $\lambda = 0.1$) but consistently *hurts* STS (86.2 → 85.7). The two goals pull in opposite directions.
- BERT-flow and BERT-whitening applied to SimCSE: hurt on transfer tasks. Good uniformity alone does not give good embeddings.

**Alignment/uniformity picture (Figure 3).** Raw BERT: good alignment, terrible uniformity (highly anisotropic). BERT-flow / whitening: great uniformity, but alignment degrades. Unsupervised SimCSE: fixes uniformity *while holding alignment*. Supervised SimCSE: also improves alignment. That balance is the whole story.

---

## Worth Remembering

- **The evaluation-protocol complaint is worth its own note.** The authors found that published STS numbers are not comparable because papers differ on (a) whether an extra linear regressor is fitted, (b) Pearson vs Spearman, (c) how the yearly sub-tasks are aggregated ("all" = concatenate everything and compute one correlation; "mean" or "wmean" = per-subset then average). SBERT used "all"; BERT-flow *claimed* to follow SBERT but actually used "wmean". They re-ran everything under one protocol (no regressor, Spearman, "all"). This is the same disease diagnosed in [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]].
- **Starting from a pre-trained checkpoint is essential.** The alignment plot shows the pre-trained model already has good alignment; contrastive training only has to fix uniformity. That is also why SimCSE does not need SimCLR-scale batch sizes.
- Transfer-task results (MR, CR, SUBJ, etc.) are only on par with SBERT, not clearly better. The authors argue STS is the real target for sentence embeddings and transfer classification is a different job. Fair, but note the limitation if you want one embedding for everything.
- Practical caveat: **dropout must be on at training time and the two forward passes must use different masks.** If you implement this by duplicating each sentence within the batch, make sure your framework does not reuse the mask. The "fixed 0.1" number (43.6) is what a silent bug here looks like.
- Retrieval quality (Table 8, Flickr30k) is qualitatively better than SBERT — SimCSE returns sentences that share *meaning*, SBERT returns sentences that share *topic words*. Relevant if you are building a [[Sentence-BERT#|bi-encoder]] retriever.
- Open follow-up: the authors suggest the dropout trick could be folded into language-model pre-training itself, or applied to other continuous representations. That thread runs into modern embedding models (E5, GTE) which all descend from this loss.

---

## Links

Related: [[Sentence-BERT]] · [[Understanding Contrastive Learning through Alignment and Uniformity]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[How Contextual are Contextualized Word Representations]] · [[Representation Degeneration Problem in Training NLMs]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Cross Entropy]] · [[Regularization]] · [[Attention Is All You Need]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[LeJEPA- Provable and Scalable Self-Supervised Learning]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Mode Collapse]]

New topics worth writing: SimCLR and the projection-head trick, Natural Language Inference (SNLI/MNLI), Semantic Textual Similarity benchmarks, Spearman rank correlation, Anisotropy and whitening of embedding spaces, Singular value spectrum as a diagnostic, Hard negative mining in contrastive retrieval, Dense Passage Retrieval, BERT-flow, In-batch negatives and their bias
