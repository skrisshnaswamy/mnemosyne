---
title: "Sparse Readout Prism: Explaining Logit-Lens Scores in Features Instead of Tokens"
authors: ["Matteo He", "William F. Shen", "Xinchi Qiu", "Nicholas D. Lane"]
year: 2026
arxiv: "2609.01936"
url: https://arxiv.org/abs/2609.01936
priority: Good-To-Read
read_on: 2026-09-06
tags: [paper, transformers, llm]
---
## The Core Idea

The **logit lens** is a standard interpretability trick: take a hidden state from the middle of a transformer, push it through the model's output matrix (the unembedding, $W_U$), and read off which tokens score highest. People then say things like "at layer 20 the model is thinking about the word *London*", or "the intermediate tokens are English, so the model thinks in English."

This paper points out that a lens reading is a product of **two** things — the state you decoded, and the *readout* you decoded it through. And it argues the reporting unit, the token, is the wrong unit twice over:

- **Too coarse.** Every sense of *bug* — the insect, the software defect — shares one row of $W_U$. A high logit for `bug` cannot tell you which sense the model means.
- **Too fine.** `London` and 伦敦 are different rows, but they can lean on the same underlying direction. Two different reported tokens can mean the same thing about the model.

The fix: fit a **sparse autoencoder to the rows of the LM head itself** — to the *weights*, not to activations. Every token row $w_v$ gets written as a sparse sum of shared directions. Then any logit, or any difference between two logits, splits exactly into one signed number per direction, plus a residual. You stop reporting "the model says `bug`" and start reporting "this margin is carried by a defect feature, a crash feature, and a debug feature."

> [!NOTE] Readout feature
> One direction $d_i$ in a dictionary learned from the rows of the unembedding matrix. A single token row uses several; a single feature appears in many rows. It describes how the *output space* is organised, independent of any input. ^readout-feature

The second half is the payoff. Fitted lenses (tuned lens, Jacobian lens) learn their transport map **from a corpus**. The authors fit two Jacobian lenses on Qwen3.5-9B that differ *only* in whether the fitting prompts were English or Chinese, then feed both the **same frozen hidden states**. The lenses report different languages. The English-fitted lens says `London`; the Chinese-fitted one says 伦敦. They call this **corpus conditionality**. Under the feature basis, both readings are carried by the *same* readout feature on 77 of 80 prompts. So the surface language was a property of the instrument, not the state.

That is the thing this unlocks: a reference frame that is fixed in the weights *before* any reading is taken, so you can tell a change in the model apart from a change in your measuring device. Nobody had done this because SAEs had only ever been fitted to activations — which are themselves corpus-dependent, so using one to referee a corpus-dependence question would just add a third instrument with the same problem.

## The Methodology

**Setup.** Let $h_\ell$ be the hidden state at layer $\ell$. A lens applies some map $T_\ell$ (for the plain logit lens this is just the final RMSNorm; for a fitted lens it also includes the learned transport) to get the decoded state $\tilde h_\ell = T_\ell(h_\ell)$. The logit for token $v$ is $\tilde h_\ell^\top w_v$, where $w_v$ is row $v$ of $W_U$.

**Step 1 — factorise the head.** Fit a TopK sparse autoencoder to the rows of $W_U$. Rows are centred and norm-normalised first (both inverted afterwards, so all reported numbers are in raw logit units). Each row becomes

$$w_v = \mu + \sum_{i=1}^{D} z_{v,i} d_i + r_v$$

with $\mu$ a shared offset (the decoder bias, deliberately kept *outside* the dictionary so it does not eat one of the $k$ sparse slots on every single row), $d_i$ a decoder direction, $z_{v,i}$ the sparse coefficient, $r_v$ an explicit residual. Operating point: $k=256$ active features per row at $32\times$ expansion, so $D = 65{,}536$ for a 2048-dim model. One dictionary per model, fitted once, ~6–12 GPU-hours on a single A40.

**Step 2 — decompose any score.** Pick a coefficient vector $\alpha$ over the vocabulary. Setting $\alpha_A = 1$ gives a plain token logit; setting $\alpha_A = 1, \alpha_B = -1$ gives a margin. Define the selected direction $q_\alpha = \sum_v \alpha_v w_v$ and the score $s_\alpha(\tilde h_\ell) = \tilde h_\ell^\top q_\alpha$. Substituting the row decomposition and swapping the sums:

$$s_\alpha(\tilde h_\ell) = \Big(\sum_v \alpha_v\Big)\tilde h_\ell^\top \mu \;+\; \sum_{i=1}^{D} \underbrace{\beta_i(\alpha)}_{\sum_v \alpha_v z_{v,i}} \underbrace{\tilde h_\ell^\top d_i}_{p_i(\tilde h_\ell)} \;+\; \tilde h_\ell^\top r_\alpha$$

The middle term is one signed contribution $c_i = \beta_i(\alpha)\,p_i(\tilde h_\ell)$ per feature, in logit units. For any **contrast** (coefficients summing to zero — every margin) the offset term vanishes, and so does anything the two rows share, including tokenizer and weight-tying artefacts.

The key structural fact: $\beta_i(\alpha)$ is fixed by the tokens you chose and the dictionary. Only $p_i(\tilde h_\ell)$ moves. So across contexts, layers, or lenses, *nothing changes but the state*. That is what makes decompositions comparable.

**Step 3 — fidelity diagnostics, reported with every decomposition.**
- *Replacement*: swap $W_U$ for the reconstructed $\widehat{W_U}$ and check held-out argmax agreement and KL.
- *Relative error* $\rho_\delta = |s_{\text{exact}} - s_{\text{recon}}| / (|s_{\text{exact}}| + \delta)$. Aggregates use $\delta = 0.5$ (floored, so near-ties don't blow up the median); case studies and baseline comparisons use the strict unfloored $\rho_0$.
- *Sign agreement*: does the reconstruction support the same side of the contrast?
- *Coverage* = correct sign **and** $\rho_0 < 0.5$. This is the headline metric.

**Evaluation suite.** Eight readouts, 0.8B–9B: Qwen3.5-0.8B/2B/9B, Gemma-4-E2B/E4B, Ministral-3-8B, R1-Distill-Qwen-7B, R1-Distill-Llama-8B. Fidelity measured on decoded states of 10,000 held-out C4 continuations plus ~1,350 hand-built selected scores per model (~850 contrasts, with prompt rewordings clustered together in the bootstrap). Gemma sits behind a nonlinear logit softcap that SRP does not factorise, so the two Gemma readouts are excluded from the baseline and intervention comparisons — a split decided *before* any contrast was scored.

## Ablation Studies and Experiments

**Does the sparse head replace the real head?** Replacing $W_U$ with $\widehat{W_U}$ preserves the held-out argmax on 0.891/0.887/0.900 of decoded states for Qwen3.5-0.8B/2B/9B, 0.904 for Ministral, and 0.75–0.76 for the two reasoning-distilled readouts. Median [[KL Divergence|KL]] between original and reconstructed next-token distributions: 0.09–0.14 bits for the base models, 0.43–0.49 bits for the distilled ones.

**The six baselines built from row geometry.** This is the ablation that carries the paper. If the rows of $W_U$ already sit in nice clusters, you don't need a dictionary. They test: nearest-row ridge regression (top-128 rows), weighted kNN on the rows, k-means centroid dictionaries at two widths, hard cluster assignment, and PCA-256. All scored on the same banks under the same harness.

| Model | SRP coverage | Best alternative | Lead |
|---|---|---|---|
| Qwen3.5-0.8B | 0.754 | 0.665 | +8.9 |
| Qwen3.5-2B | 0.780 | 0.625 | +15.5 |
| Qwen3.5-9B | 0.812 | 0.639 | +17.3 |
| Ministral-8B | 0.777 | 0.648 | +12.9 |
| R1-Q-7B | 0.716 | 0.579 | +13.7 |
| R1-Llama-8B | 0.751 | 0.624 | +12.7 |

Intervals do not overlap. The sharpest test is the k-means dictionary built at SRP's *exact* width and sparsity ($D = 65{,}536$, $k = 256$ centroids per row) — capacity-matched, and it still trails by 18 points on Qwen3.5-2B. SRP also has the lowest mean, p95, and max absolute error of all seven methods, so the ranking survives any choice of summary statistic. Nulls (shuffled codes, random support) sit below 0.10 coverage.

An honest note: an earlier version of the clustering and kNN baselines over-reconstructed through a full-rank projection artefact and had to be redesigned. Nice to see that written down (cf. [[On the Difficulty of Evaluating Baselines]]).

**Do the contributions actually mean anything causally?** Take the top-10 features of a contrast, remove that feature's direction from the decoded state ($\tilde h' = \tilde h - (\tilde h^\top d_i)d_i$), measure the margin change with the *original dense* head, and compare to the predicted $-c_i$. Regression through the origin gives $r^2 = 0.83$–$0.93$ on all six softcap-free readouts, against $\le 0.06$ for random directions. Slopes are ~1 on Ministral and the distilled models (0.96–1.13) but **1.34–1.69 on the three Qwen models** — the margin moves *further* than predicted. The authors float feature interference (overlapping decoder directions sharing the ablated mass) as an explanation but explicitly say they have not isolated it. A harness self-test on a synthetic residual-free factorisation recovers $r^2 = 1.000$, so the pipeline isn't manufacturing agreement.

**Does it survive retraining?** Three seeds per width on Qwen3.5-2B, all at equal replacement fidelity (top-1 0.805–0.848). The *individual decoder directions do not recur*: median cosine to the nearest direction in another seed's dictionary is ~0.33. This is a known property of TopK dictionaries. But the *token groups* do recur. For the same contrast side, two seeds' token lists share mean Jaccard 0.21–0.24, against 0.014–0.018 for unrelated contrasts — 13–15× the null, with all 63 tested contrasts above the null's 90th percentile. 87–91% of features find an above-chance counterpart in another seed, and the median group is 83–90% covered by a union of at most 3 features, so the leftover instability is **feature splitting**, not noise.

**Sense disambiguation (CoarseWSD-20).** Pick 8 features per sense by mean contribution difference on train, assign test contexts by which group sums larger. Balanced accuracy 0.90 / 0.92 / 0.80 on Qwen3.5-2B / 9B / R1-Llama-8B, against a permutation null of ~0.41 that repeats the whole selection. Holds at 1, 2, 4, or 8 features per sense.

**What did not work here:** a nearest-centroid probe on the *full decoded state* reaches 0.96 / 0.97 / 0.92, and the full unselected 256-dim account reaches 0.95 / 0.97 / 0.89. So the eight features cost you 5–12 points against just using the raw state. The paper is upfront that the sparse account buys readability, not accuracy.

**A second thing that did not work:** in a per-word classifier framing, the *unweighted* projections $p_i(\tilde h_\ell)$ beat the signed contributions on all three models (0.87 vs 0.84, 0.90 vs 0.76, 0.73 vs 0.67). The explanation is clean — for a fixed token the coefficients $z_{v,i}$ are frozen, so weighting is just a constant per-coordinate rescale that a supervised classifier gains nothing from. Signed contributions name which directions support a *score*; they are not a sense representation.

**The cross-lens study.** Two Jacobian lenses on Qwen3.5-9B, 100 English vs 100 Chinese C4 prompts, identical seeds and procedure, applied to the *same frozen states* from 80 prompts across seven task families. Layers 21/24/26/29. A prompt counts as agreement if the same feature dominates under both lenses at ≥2 of 4 layers.

- Tokens reported in different scripts on **39/80** prompts (9/10 on factual recall). Same dominant feature on **77/80** (96%, CI [0.90, 0.99]).
- Both nulls empty: 0/159 for an unrelated token under the same lens, 0/240 after shuffling prompt pairings.
- On the antonym prompt from the Jacobian-lens paper itself, the English-fitted lens reads `large`/`big` and the Chinese-fitted lens reads 大的/大 — carried by the same feature f112, whose top rows are the 大 family.

Extensions that rule out the obvious confounds:
- **Different lens construction.** Ridge translators in the tuned-lens style, same corpora: 67/80 feature agreement. Holding the corpus fixed at English and varying only construction: 76/80.
- **Shared script.** English vs German (cognates excluded at edit distance ≤2). Tokens differ on 73/90, same feature dominant on 84/90. So it is not a script artefact.
- **Corpus size.** Triple to 300 prompts. Feature agreement is *identical prompt-for-prompt* on EN–ZH (still 77/80), and 84→85/90 on EN–DE. Token disagreement moves in **opposite directions** — falls 39→33 on EN–ZH, rises 73→76 on EN–DE. Estimation noise would shrink both. Eleven EN–ZH prompts change their reported language across fitting scales; none changes its dominant feature.

## Worth Remembering

- **Scope of the corpus-conditionality claim.** It applies only to *fitted* lenses. Studies that decode straight through $W_U$ (Wendler et al., Schut et al. on "do LLMs think in English?") use no fitting corpus and are not touched by this. The paper flags a weaker, separate confound for them: which of several surface realisations wins a ranking is fixed by the row codes of $W_U$ alone, and one feature carries both realisations on ~61/80 of within-lens comparisons.

- **Where the error lives.** SRP's *absolute* error is nearly flat across margin sizes (0.64–0.71 logits). The relative-error tail is a denominator effect, concentrated in the 14% of contrasts where the two tokens are near-tied — coverage there is 0.227. On the 61% of contrasts with margin ≥2, sign agreement is 0.996 and coverage 0.966. So the method is reliable exactly where a reading is meaningful, and the diagnostics make the near-tie regime visible rather than hiding it.

- **Row reconstruction does not predict replacement.** The two Gemma readouts nearly match the others on explained variance over centred rows while replacing far worse. Reasoning-distilled models show the same split: R1-Llama-8B matches Ministral on rowEV to three decimals (0.888) while top-1 lags (0.754 vs 0.904). The likely cause is a heavier row-norm tail on distilled checkpoints (max/median 1.7–2.0 vs 1.5–1.6), and the recipe normalises rows uniformly in $L_2$ while the softmax is dominated by the heaviest rows. Norm-aware recipes are the obvious follow-up.

- **Sparsity budgets do not go lower.** At fixed width, dropping $k$ from 256 to 64 kills half the dictionary (dead-feature rate 0.49–0.51) and degrades everything. At $k=32$, two thirds is dead. What matters is $k/D$, and what a reader actually inspects is sparser than $k$ anyway — a median of 65–107 features carries 80% of a score's contribution mass.

- **Stability is scoped to a fixed recipe.** Within a recipe, seeds reproduce grouping structure well. *Across* recipes it falls apart: the released paper dictionary (earlier preprocessing) shares only 36% of the stable core and is beaten by row-kNN on that comparison. Every stability claim is conditioned on holding the recipe fixed. If you use this, do not mix dictionary families.

- **The edit test loses on two of four models.** SRP directions selected from 5 profanity tokens transfer to 10 held-out ones, matching an oracle that knows the whole lexicon while a token-id bias on the discovery tokens transfers zero. But at *matched* KL cost, SRP beats the simple mean-of-discovery-rows direction decisively on Qwen3.5-2B and **loses** on Qwen3.5-0.8B and Qwen3.5-9B. Two explanations were tested and rejected (replacement fidelity; dictionary width — the latter a registered prediction that failed). Against capacity-matched PCA rank-4 it wins nearly everywhere. Honest scoping, unusual to see.

- **Label provenance.** The text on the bars ("mosquito, bee, beetle") is drafted by an LLM from a feature's top unembedding rows. Three blinded rater runs agree unanimously on 93.4% of items (Fleiss' $\kappa = 0.87$); consensus is 76 coherent / 4 mixed / 19 token-form out of 99. All *quantitative* claims rest on feature ids, signed contributions, and residuals — not labels. 14/99 labels are pure surface artefacts (letter casing, leading-space variants).

- **It composes with direct logit attribution.** DLA splits a margin across residual-stream components; SRP splits it across readout features. Together you get one term per (component, feature) pair — which layer supplied the verification feature, in logit units. The two decompositions answer orthogonal questions: *who wrote the state* vs *what reads it*.

- **Practical caveat.** Everything here is on the readout side. Ablations act linearly on $\tilde h_\ell$ with no re-forward pass, so the validated claim is about *local* score changes. Circuit-level necessity and generation-level effects need different interventions.

- **A stray methodological gem.** The nearest-row reading of $w_{\texttt{bug}} - w_{\texttt{insect}}$ — the zero-fitting baseline — returns almost entirely casing and morphological variants (`Bug`, `BUG`, `bugs`, `_bug`) for the top 12 ranks, with the first sense-bearing rows at ranks 13–17 and the orthographic neighbour `blog` by rank 30. Morphology dominates semantics in raw row cosine.

## Links

Related: [[Attention Is All You Need]] · [[KL Divergence]] · [[Cross Entropy]] · [[Linear Projection]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Layer Normalization]] · [[Representation Degeneration Problem in Training NLMs]] · [[How Contextual are Contextualized Word Representations]] · [[Whitening Sentence Representations]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[A Unified Approach to Interpreting Model Predictions (SHAP)]] · [[Saliency]] · [[Thinking in a Low-Resource Language- What SFT Builds, What RL Fixes, What Accuracy Cannot See]] · [[Chain-of-Thought Faithfulness of Reasoning Models Varies with Where and How Preference Cues Are Delivered]] · [[On the Difficulty of Evaluating Baselines]] · [[The Lottery Ticket Hypothesis]] · [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)]]

New topics worth writing: Logit lens, Tuned lens, Sparse autoencoders for interpretability, TopK sparse autoencoders, Dictionary learning, Superposition hypothesis, Direct logit attribution, Weight tying in language models, Mechanistic interpretability, Feature splitting, Word sense disambiguation, Activation patching, Cluster bootstrap
