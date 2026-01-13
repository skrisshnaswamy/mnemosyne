---
title: "AutoInt: Automatic Feature Interaction Learning via Self-Attention"
authors: ["Weiping Song", "Chence Shi", "Zhiping Xiao", "Zhijian Duan", "Yewen Xu", "Ming Zhang", "Jian Tang"]
year: 2018
arxiv: "1810.11921"
url: https://arxiv.org/abs/1810.11921
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, transformers]
---
## The Core Idea

CTR prediction (predicting if a user clicks) lives or dies on **feature crosses** — combinations like `<Gender=Male, Age=10, Category=VideoGame>`. Before this paper you had two bad options. Either hand-write the crosses (Wide part of [[Wide & Deep Learning for Recommender Systems]]), which does not scale and needs domain experts. Or throw a plain MLP at the concatenated embeddings and hope it discovers crosses on its own — which is what [[Deep Learning Recommendation Model (DLRM)]]-style models and DeepCrossing do. The second option is **implicit**: you cannot look inside and say "the model decided that Age and Genre matter together". And fully-connected layers are known to be bad at learning multiplication.

The trick here is embarrassingly simple in hindsight: **treat the feature fields of one row like the tokens of one sentence, and run multi-head self-attention over them.** A row of Criteo data has 39 fields. Embed each field into a $d$-dimensional vector. Now you have a "sequence" of length 39 with no order. Feed it to the encoder block from [[Attention Is All You Need]]. Each field attends to every other field, and the attention weight *is* the answer to "which two features should be crossed".

Two things fall out of this for free:

1. **Order of interaction = depth.** One interacting layer mixes pairs, so field $m$'s new vector encodes second-order crosses involving $m$. Stack a second layer and $m$'s vector (which now holds $g(x_1,x_2)$) attends to field 3's vector (which still holds raw $x_3$ thanks to the residual), giving third-order crosses. The maximum order grows roughly like $2^L$ with $L$ layers.
2. **Explainability.** The attention matrix is a $M \times M$ heat map you can literally plot. Averaged over MovieLens-1M it says `<Gender, Genre>`, `<Age, Genre>`, `<RequestTime, ReleaseTime>` are the strongly coupled field pairs. [[Deep & Cross Network for Ad Click Predictions]] and xDeepFM also cross features explicitly, but their outer products give you no readable "which pair mattered" signal.

The cost side is the real selling point. On Criteo, AutoInt uses $3.9 \times 10^4$ non-embedding parameters against CIN's $1.9 \times 10^6$ — **48× smaller** — and still beats it.

> [!NOTE] Interacting layer
> One multi-head self-attention block applied across *feature fields* of a single training row, not across time or tokens. Each stacked layer raises the order of feature crosses the model can express. ^interacting-layer

## The Methodology

**Input.** $\mathbf{x} = [\mathbf{x_1}; \dots; \mathbf{x_M}]$, the concatenation of $M$ fields. Categorical fields are one-hot (or multi-hot), numerical fields are a single scalar.

**Embedding layer.** Everything lands in the same $\mathbb{R}^d$ so that categorical and numerical features can actually interact via dot products.

- Categorical: $\mathbf{e_i} = \mathbf{V_i}\mathbf{x_i}$, a lookup.
- Multi-valued categorical (e.g. movie Genre = {Drama, Romance}): $\mathbf{e_i} = \frac{1}{q}\mathbf{V_i}\mathbf{x_i}$, the mean of the $q$ active embeddings.
- Numerical: $\mathbf{e_m} = \mathbf{v_m}\, x_m$ — one learned vector per field, **scaled** by the scalar value. Neat: a numerical field gets a direction from learning and a magnitude from the data.

**Interacting layer.** Standard scaled-free dot-product attention ([[Query, Key, and Value (QKV)]]), per head $h$:

$$\alpha^{(h)}_{m,k} = \frac{\exp\!\big(\langle \mathbf{W}^{(h)}_{Q}\mathbf{e_m},\, \mathbf{W}^{(h)}_{K}\mathbf{e_k}\rangle\big)}{\sum_{l=1}^{M}\exp\!\big(\langle \mathbf{W}^{(h)}_{Q}\mathbf{e_m},\, \mathbf{W}^{(h)}_{K}\mathbf{e_l}\rangle\big)}$$

$$\tilde{\mathbf{e}}^{(h)}_m = \sum_{k=1}^{M}\alpha^{(h)}_{m,k}\,\big(\mathbf{W}^{(h)}_{V}\mathbf{e_k}\big)$$

Note there is **no $1/\sqrt{d}$ scaling** — they use the raw inner product. Heads are concatenated, $\tilde{\mathbf{e}}_m = \tilde{\mathbf{e}}^{(1)}_m \oplus \cdots \oplus \tilde{\mathbf{e}}^{(H)}_m$, then a residual connection with a projection to fix the dimension mismatch ([[Deep Residual Learning for Image Recognition (ResNet)]]):

$$\mathbf{e}^{Res}_m = \text{ReLU}\big(\tilde{\mathbf{e}}_m + \mathbf{W}_{Res}\,\mathbf{e_m}\big)$$

The ReLU is doing double duty — it is the non-linearity that makes the combination function $g(\cdot)$ *non-additive*, which is the formal requirement for something to count as a feature cross.

What is **missing** compared to a real Transformer block: no [[Layer Normalization]], no position encoding (fields have no order — position is identity, which is already in the embedding table), no feed-forward sublayer.

**Output.** Concatenate all $M$ field vectors, one linear projection, sigmoid:

$$\hat{y} = \sigma\big(\mathbf{w}^\top(\mathbf{e}^{Res}_1 \oplus \cdots \oplus \mathbf{e}^{Res}_M) + b\big)$$

**Loss.** Plain binary [[Cross Entropy]] (they call it Logloss), optimised with [[Adam- A Method for Stochastic Optimization|Adam]].

**Hyperparameters that mattered.** $d = 16$, batch 1024, $L = 3$ interacting layers, $d' = 32$, $H = 2$ heads. [[Dropout- A Simple Way to Prevent Overfitting|Dropout]] was needed *only* on MovieLens-1M (739k rows); the three big datasets did not overfit without it. Numerical features are preprocessed with the Criteo-competition trick: $z \mapsto \log^2(z)$ when $z > 2$.

**Complexity.** Parameters in interacting layers: $O(Ldd'H)$ — **independent of the number of fields $M$**. Time: $O(MHd'(M+d))$, quadratic in $M$ as attention always is, but $M \le 39$ so nobody cares.

## Ablation Studies and Experiments

Four datasets: Criteo (45.8M rows, 39 fields), Avazu (40.4M, 23), KDD12 (149.6M, 13), MovieLens-1M (739k, 7). Metrics AUC and Logloss; in CTR a 0.001 move is considered real.

**Main table (AUC, best baseline → AutoInt):**

| Dataset | Best baseline | AutoInt |
|---|---|---|
| Criteo | 0.8009 (CIN / DeepCrossing) | **0.8061** |
| Avazu | **0.7758** (CIN) | 0.7752 |
| KDD12 | 0.7799 (CIN) | **0.7883** |
| MovieLens-1M | 0.8448 (DeepCrossing) | **0.8456** |

Avazu is the honest loss: CIN wins on AUC, AutoInt wins on Logloss (0.3824 vs 0.3829).

**What the baselines reveal.** [[Factorization Machines (ICDM)|FM]] and AFM (second-order only) crush LR everywhere — individual features are simply not enough. But the *high-order implicit* models (DeepCrossing, NFM) do **not** reliably beat the second-order ones. NFM gets 0.7515 AUC on KDD12, worse than FM's 0.7759. Stacking an MLP on top does not guarantee you learn crosses. The explicit models (CIN, CrossNet, HOFM, AutoInt) are consistently above the low-order ones. This is the paper's real argument: explicit > implicit.

The cleanest comparison is **AutoInt vs DeepCrossing** — same embedding layer, same residual structure, same output. The *only* difference is attention-based interacting layers instead of fully-connected layers. That gap (0.8061 vs 0.8009 on Criteo) is attributable to the attention mechanism alone.

**Residual ablation (Table 4).** Remove the residual, keep everything else:

| Dataset | with | without | ΔAUC |
|---|---|---|---|
| Criteo | 0.8061 | 0.8033 | −0.0028 |
| Avazu | 0.7752 | 0.7729 | −0.0023 |
| KDD12 | 0.7888 | 0.7831 | −0.0057 |
| MovieLens-1M | 0.8460 | 0.8299 | **−0.0161** |

Big drops, especially MovieLens. The residual is not a training-stability nicety here — it is *load-bearing for the mechanism*. Without it, layer $\ell$'s output has no copy of the raw first-order feature, so layer $\ell+1$ cannot build order-$(p{+}1)$ crosses by pairing a cross with a raw feature. Killing the residual literally removes the ladder.

**Depth ablation.** 0 layers (weighted sum of raw features) → 1 layer is a huge jump. 1 → 2 → 3 keeps improving. At 3 layers it **flattens**. Very high-order crosses are not informative. Three layers is the sweet spot, which is why the default is 3.

**Embedding dimension.** Diverges by dataset. On KDD12 (149M rows) AUC keeps climbing with $d$. On MovieLens-1M it **peaks at $d=24$ then falls** — classic overfitting on a small dataset. Not a universal setting; tune per dataset.

**AutoInt+ (adding an implicit MLP).** Joint-train with a two-layer feed-forward net, the [[Wide & Deep Learning for Recommender Systems|Wide&Deep]] recipe. It helps everywhere and gives new SOTA (Criteo 0.8083, MovieLens 0.8488). But the interesting number is the *magnitude of the gain*:

| Model | Avg. ΔAUC from adding the MLP |
|---|---|
| Wide&Deep (over LR) | +0.0292 |
| Deep&Cross (over CrossNet) | +0.0200 |
| DeepFM (over FM) | +0.0142 |
| xDeepFM (over CIN) | +0.0068 |
| **AutoInt+ (over AutoInt)** | **+0.0023** |

The stronger your explicit component already is, the less the implicit MLP adds. AutoInt has already captured most of what the MLP would find.

**Efficiency.** Runtime comparable to DeepCrossing and NFM; CIN — the strongest baseline — is far slower because of its crossing layers, and HOFM would not even fit on one GPU for KDD12.

## Worth Remembering

- **The "sequence" here has no order and length ~10–40.** That is why the recipe is stripped down (no positional encoding, no LayerNorm, no FFN) and why the $O(M^2)$ cost of attention is irrelevant. This is a completely different regime from [[Self-Attentive Sequential Recommendation (SASRec)]], which attends over a *user's history*; AutoInt attends over *fields of one row*. The two are orthogonal and composable.
- **No scaling factor on the dot product.** With $d' = 32$ this is probably fine, but if you scale $d'$ up you should reintroduce $1/\sqrt{d'}$ or expect saturated softmaxes.
- **Explainability is claimed, not measured.** The heat maps in Figure 7 are eyeballed, and the "young men like action/thriller" reading is exactly the kind of story you can always tell after seeing a plot. Attention weights are a correlation, not a causal attribution — treat them as a hypothesis generator, not an explanation. If you want something with axioms behind it, [[A Unified Approach to Interpreting Model Predictions (SHAP)]] is the alternative.
- **Numerical field embedding $\mathbf{v_m} x_m$ is a genuinely reusable idea.** One direction, scaled by value. It also means the embedding for a numerical field is a straight line through the origin — the model cannot represent a non-monotone effect of that feature *within the embedding*, only through the downstream non-linearity. [[Revisiting Deep Learning Models for Tabular Data (FT-Transformer)|FT-Transformer]] later does almost this exact thing for tabular data, with a bias term added.
- **Gains are ~0.005 AUC over a strong baseline.** Real for CTR, but small enough that the field's replication problems apply — see [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] and [[On the Difficulty of Evaluating Baselines]]. They did average over 10 runs and report unpaired t-tests, which is better than most papers of this era. They did **not** run an online A/B test.
- **No public-data result is an industrial result.** All four datasets are multi-epoch offline benchmarks; production CTR systems train streaming, single-pass, on shifting distributions. [[Actions Speak Louder than Words- Generative Recommenders (HSTU)]] is worth reading as a counterpoint on what changes at that scale.
- Open question the paper leaves: does the field-attention still pay for itself when the model already has a rich user-history tower, as in [[Deep Interest Network for CTR Prediction (DIN)]]? The two mechanisms are attacking different sources of signal, and nobody in this paper checked whether the gains stack.

## Links

Related: [[Attention Is All You Need]] · [[Factorization Machines (ICDM)]] · [[Deep & Cross Network for Ad Click Predictions]] · [[DCN V2- Improved Deep & Cross Network]] · [[Wide & Deep Learning for Recommender Systems]] · [[Deep Learning Recommendation Model (DLRM)]] · [[Deep Interest Network for CTR Prediction (DIN)]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Query, Key, and Value (QKV)]] · [[Revisiting Deep Learning Models for Tabular Data (FT-Transformer)]] · [[Self-Attentive Sequential Recommendation (SASRec)]] · [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] · [[Ad Click Prediction- a View from the Trenches (KDD)]] · [[Recommender Systems - Evolution]] · [[Cross Entropy]] · [[Dropout- A Simple Way to Prevent Overfitting]] · [[Adam- A Method for Stochastic Optimization]]

New topics worth writing: xDeepFM / Compressed Interaction Network, DeepFM, Neural Factorization Machines, Attentional Factorization Machines, Field-aware Factorization Machines, Higher-Order Factorization Machines, DeepCrossing, Explicit vs implicit feature interaction, AUC vs Logloss for CTR evaluation, Multi-hot field embedding pooling
