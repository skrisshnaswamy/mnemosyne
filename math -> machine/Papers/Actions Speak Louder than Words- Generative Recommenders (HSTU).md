---
title: "Actions Speak Louder than Words: Generative Recommenders (HSTU)"
authors: ["Jiaqi Zhai", "Lucy Liao", "Xing Liu", "Yueming Wang", "Rui Li", "Xuan Cao", "Leon Gao", "Zhaojie Gong", "Fangda Gu", "Michael He", "Yinghai Lu", "Yu Shi"]
year: 2024
arxiv: "2402.17152"
url: https://arxiv.org/abs/2402.17152
priority: Must-Read
read_on: 2026-08-22
tags: [paper, transformers, llm, vision, scaling]
---
## The Core Idea

Industrial recommenders (DLRMs — Deep Learning Recommendation Models) are built out of thousands of hand-made features: counters, ratios, "user's CTR on outdoor videos in the last 7 days", plus embedding tables and a zoo of interaction modules. They work, but they stop improving when you give them more compute. Meta measured this: their production DLRM saturates at about 200 billion parameters. More FLOPs, no more quality.

The claim here is that this is a *formulation* problem, not a compute problem. If you throw away the hand-made features and instead treat the user's raw action history as one long sequence of tokens — like text — then recommendation becomes a next-token problem, and the scaling behaviour of language models comes back.

> [!NOTE] Sequential transduction
> Take an ordered list of input tokens $x_0,\dots,x_{n-1}$ and map it to an output list $y_0,\dots,y_{n-1}$, where $y_i$ may be "undefined" ($\varnothing$). Both ranking and retrieval get cast into this shape. ^sequential-transduction

Two things had to be invented to make this work at billion-user scale.

**A trick to keep ranking "target-aware".** In a normal [[Auto-regressive models|autoregressive]] setup the candidate item only meets the user history at the very end, via a softmax. Ranking needs the candidate to interact with history *early* — that is the whole point of [[Deep Interest Network for CTR Prediction (DIN)|DIN]]-style attention. The fix is to **interleave** items and actions in one sequence: $\Phi_0, a_0, \Phi_1, a_1, \dots$. Now predicting $a_{i+1}$ given everything up to $\Phi_{i+1}$ *is* target-aware cross-attention, and you get it for all $n_c$ engagements in a single [[Causal Attention|causal]] pass.

**An attention variant for a vocabulary that never stops growing.** Softmax normalises over the whole sequence, which destroys the *count* of related past items — and "how many times did this user engage with this topic" is exactly the signal you need to predict *how much* time they will spend. HSTU drops the softmax for pointwise SiLU attention. On synthetic streaming data this is worth 44.7% relative HR@10.

The payoff: models 285× more computationally complex than the production DLRM, deployed, with **12.4%** improvement in the main engagement metric in online A/B, and quality that follows a [[Scaling Laws for Neural Language Models#^power-law|power law]] in training compute across three orders of magnitude, up to GPT-3 / LLaMA-2 scale of total FLOPs.

## The Methodology

### Turning DLRM features into one sequence

- **Categorical features.** Pick the longest series (items the user engaged with) as the *main* time series. Other categorical series (language, followed creators, city) change slowly, so compress them — keep only the first entry of each constant run — and merge them into the main series. Sequence length barely grows.
- **Numerical features.** Delete them. The argument: features like "past CTR on topic X" are aggregations *over* the categorical events that are already in the sequence. A long enough sequence plus target-aware attention should be able to recompute them. This is the boldest bet in the paper and the experiments below test it directly.

### Generative training (the compute trick)

DLRMs train one example per *impression*. With self-attention that costs $\sum_i n_i(n_i^2 d + n_i d^2) = O(N^3 d + N^2 d^2)$ — hopeless. Instead, sample user $i$ at rate $s_u(n_i) = 1/n_i$ (in practice: emit one training example at the end of a user session, not per impression) and predict *all* targets in that user's sequence in one pass. Cost drops by a factor of $N$ to $O(N^2 d + N d^2)$. Encoder work is amortised over many targets. GRs see 1–2 orders of magnitude fewer "examples" than DLRMs for the same data.

### The HSTU layer

A stack of identical residual layers ([[Deep Residual Learning for Image Recognition (ResNet)|ResNet]]-style). Three sub-layers:

$$U(X), V(X), Q(X), K(X) = \text{Split}(\phi_1(f_1(X)))$$
$$A(X)V(X) = \phi_2\!\left(Q(X)K(X)^T + \text{rab}^{p,t}\right)V(X)$$
$$Y(X) = f_2\!\left(\text{Norm}(A(X)V(X)) \odot U(X)\right)$$

- $f_1, f_2$ are **single linear layers**, not MLPs. $f_1$ produces [[Query, Key, and Value (QKV)|Q, K, V]] *and* a gate $U$ in one fused kernel.
- $\phi_1 = \phi_2 = \text{SiLU}$. Note $\phi_2$ replaces softmax — attention weights are not normalised over the sequence. LayerNorm after the pooling is **required** or training blows up.
- $\text{rab}^{p,t}$ is a relative attention bias (T5 style) carrying both *position* and *time gap* $t_j - t_i$, bucketised. Shared across heads.
- $\text{Norm}(AV) \odot U$ is a [[Gated Activation|gate]] — effectively SwiGLU. It replaces the whole DLRM "feature interaction" stage (DCN, factorization machines), because MLPs are bad at approximating dot products. It also stands in for Mixture-of-Experts routing: elementwise gating is MoE-like conditional computation up to a normalisation.

There is **no feedforward block**. Linear layers outside attention go from six (Transformer) to two. Activation memory per layer in bf16 drops from $33d$ to $14d$, which is what lets them stack >2× more layers than a [[Attention Is All You Need|Transformer]] at the same batch size. Batch size matters a lot in recsys, so activation memory — not parameter memory — is the binding constraint, the opposite of LLM training.

Embedding tables: 10B vocabulary × 512d with Adam in fp32 would be 60TB. They use rowwise AdamW with optimiser state on DRAM, cutting HBM from 12 bytes/float to 2.

### Stochastic Length (SL)

Attention costs $\Theta(\sum_i n_i^2)$. User behaviour is temporally repetitive, so you can throw away most of the sequence most of the time. With $N_c$ the max history length and $\alpha \in (1,2]$:

- if $n_{c,j} \le N_c^{\alpha/2}$: keep the whole sequence;
- else with probability $N_c^{\alpha}/n_{c,j}^2$: keep the whole sequence;
- else: keep a subsequence of length $N_c^{\alpha/2}$.

This drops attention complexity to $O(N^\alpha d)$. At $\alpha = 1.6$ a length-4096 sequence becomes length-776 most of the time — over 80% of tokens deleted. Which subsequence you keep matters: sampling weighted by recency ($f_i = t_n - t_i$) beat greedy-most-recent and uniform-random (NE 0.789 vs 0.792 / 0.791 on the consumption task).

### M-FALCON (inference)

Ranking scores up to tens of thousands of candidates. Naively, target-aware means one forward pass per candidate: $O(m n^2 d)$. Instead, append $b_m$ candidates to the sequence at once and edit the causal mask so candidate $i$ cannot attend to candidate $j$ — the outputs are provably identical to $b_m$ separate passes. Cost becomes $O((n+b_m)^2 d) = O(n^2 d)$. Then split $m$ candidates into $\lceil m/b_m \rceil$ microbatches and KV-cache the user-history part of $K(X), V(X)$ across microbatches *and across requests*. A cached pass costs $O(b_m d^2 + b_m n d)$.

### Training scale

100B DLRM-equivalent examples, 64–256 H100s per job. Ablation configs: ranking $l=3$, $n=2048$, $d=512$; retrieval $l=6$, $n=512$, $d=256$. Largest scaling-law model: $n=8192$, $d=1024$, 24 HSTU layers, 1.5T parameters.

## Ablation Studies and Experiments

### Public datasets (multi-epoch, full shuffle)

Against SASRec with the best-known 2023 recipe. [[NDCG]]@10 and Hit Rate over the full corpus (no negative sampling in eval):

| Dataset | Model | HR@10 | NDCG@10 |
|---|---|---|---|
| ML-1M | SASRec | .2853 | .1603 |
| | HSTU (same config) | .3097 (+8.6%) | .1720 (+7.3%) |
| | HSTU-large (4× layers, 2× heads) | .3294 (+15.5%) | .1893 (+18.1%) |
| ML-20M | SASRec | .2906 | .1621 |
| | HSTU-large | .3567 (+22.8%) | .2106 (+30.0%) |
| Books | SASRec | .0292 | .0156 |
| | HSTU-large | .0469 (+60.6%) | .0257 (**+65.8%**) |

[[BERT4Rec- Sequential Recommendation with Bidirectional Transformer|BERT4Rec]] and GRU4Rec are *worse than SASRec* on HR@10 here (ML-20M: .2816 and .2813 vs .2906) once sampled-softmax training is done properly — consistent with [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|the reproducibility literature]]. The bidirectional [[BERT- Pre-training of Deep Bidirectional Transformers#^masked-language-model|cloze objective]] does not buy anything here.

### Architecture ablation, industrial streaming (one pass)

Metric is [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)#^normalized-entropy|Normalized Entropy]]; a 0.001 drop is considered significant (≈0.5% topline).

| Architecture | Retrieval log-pplx | Rank NE (E-Task) | Rank NE (C-Task) |
|---|---|---|---|
| Transformer | 4.069 | **NaN** | **NaN** |
| HSTU (−rab, Softmax) | 4.024 | .5067 | .7931 |
| HSTU (−rab) | 4.021 | .4980 | .7860 |
| Transformer++ (RoPE, SwiGLU) | 4.015 | .4945 | .7822 |
| HSTU (original T5 rab) | 4.029 | .4941 | .7817 |
| HSTU (full) | **3.978** | **.4937** | **.7805** |

What this reveals:

- **Plain Transformers do not train** on ranking in this setting. Loss explodes even with pre-norm and a 10× lower learning rate. Same for softmax-HSTU — it needs ~10× lower LR than pointwise HSTU.
- Swapping softmax → pointwise SiLU is worth 0.0087 NE on E-Task. That is the single biggest architectural win.
- The *temporal* part of the relative attention bias matters: original T5 positional rab gives .4941, adding time gives .4937 and a much bigger retrieval gain (4.029 → 3.978).
- HSTU beats Transformer++ (the LLaMA recipe) while running 1.5–2× faster with 50% less HBM.

### The feature bet

| Ranking method | NE E-Task | NE C-Task |
|---|---|---|
| Production DLRM (~1000 dense + 50 sparse features) | .4982 | .7842 |
| DLRM (DIN+DCN+MMoE, published components only) | .5053 | .7899 |
| DLRM given only GR's raw features | .5053 | .7925 |
| GR (interactions only, SASRec-style) | .4851 | .7903 |
| **GR (full)** | **.4845** | **.7645** |

Read this carefully. Strip DLRM down to GR's raw features and it loses 0.007 NE — so those hand-made features *are* carrying real signal for a DLRM. GR on the same raw features beats the full-featured DLRM. That is the evidence that HSTU reconstructs the numerical features internally.

And "interactions only" — the classic sequential-recommender setup that keeps only items the user touched, discarding the merged categorical series — costs 2.6% NE on the consumption task. So merging demographics/languages/followed-creators into the sequence is doing real work; this is not just SASRec with more layers.

Retrieval, offline HR@100: DLRM 29.0%, GR 36.9%. Online, adding GR as a new retrieval source: +6.2% E-Task. Replacing the main DLRM source: +5.1%.

**Content-based GR (item content features only, the "use an LLM" flavour) gets 11.6% HR@100 vs DLRM's 29.0%.** A crushing result for text-only recommendation at this scale.

### Efficiency

- HSTU vs FlashAttention-2 Transformer, same $d=512$, $h=8$, $d_{qk}=64$, H100, bf16, 8192-length: **15.2× faster training, 5.6× faster inference** (5.3× is the paper's conservative headline). Sources: fully-ragged grouped GEMMs exploiting sequence-length skew (2–5× alone), fused rab construction, and SL.
- SL at 64–84% sparsity costs ≤0.002 NE. Compare against length-extrapolation baselines at matched sparsity (train at 1024, eval at 4096): NTK-aware RoPE zero-shot loses 11.27%, fine-tuned RoPE loses 2.19%, clamped-rab fine-tune loses 2.21%, **SL loses 0.64%**. The authors think zero-shot/fine-tune extrapolation fails because with a billion-scale vocabulary the model never learns good embeddings for older ids.
- End-to-end: GR at 285× DLRM FLOPs still gets **1.50× the QPS at 1024 candidates and 2.99× at 16384**. Microbatching + caching alone gives 1.99× over a single $b_m = m = 1024$ pass.

### Scaling

GRs follow a power law in compute for HR@100, HR@500 and NE across three orders of magnitude. DLRMs plateau — scaled with Transformers, DHEN, and residual-DCN, all saturate around 200B params. **In the low-compute regime DLRMs actually win**, because hand-crafted features are a good prior when you cannot afford to learn them.

Unlike [[Scaling Laws for Neural Language Models|Kaplan et al.]], **sequence length is a first-class scaling knob here** and must be grown in tandem with width and depth. Within reason, the exact hyperparameter split matters less than total compute — same qualitative finding as [[Training Compute-Optimal Large Language Models (Chinchilla)|Chinchilla]], different variables.

## Worth Remembering

- **The "no numerical features" claim is only supported indirectly.** They show DLRM-with-GR-features is bad and GR-with-GR-features is good. They never show GR + hand-made numerical features to check there is nothing left on the table. Plausibly there is.
- **Instability is a real result, not a footnote.** Vanilla Transformers produced NaN on industrial ranking. If you are porting an LLM block into a streaming recsys with a non-stationary vocabulary, expect the softmax to be the problem, and expect to need LayerNorm after unnormalised attention pooling.
- **Pointwise attention breaks a nice property.** Without softmax normalisation, attention output magnitude grows with how many relevant past items exist. That is deliberate — it encodes *intensity* of preference, needed for regression-flavoured targets like watch time — but it means the layer is not scale-invariant and the LayerNorm is load-bearing.
- The self-attention kernel is described as **memory-bound**, scaling as $\Theta(\sum_i n_i^2 d_{qk}^2 R^{-1})$ in memory accesses, where $R$ is register size. This is a [[Flash Attention]] descendant with ragged batching, not a new attention maths.
- **M-FALCON is independent of HSTU.** The masked-batched-candidates trick works for any target-aware causal self-attention ranker, including plain Transformers. If you take one deployable idea from this paper without rewriting your model, take this one.
- Sequence lengths: 8192 here vs 100 in TransAct (Pinterest), 1000 in DIN, 20 in BST. Two to three orders of magnitude. Compare [[PinnerFormer- Sequence Modeling for User Representation at Pinterest|PinnerFormer]], which amortises cost by predicting many future actions from one daily-computed embedding — a different answer to the same "encoder cost per target" problem.
- Retrieval uses sampled softmax; number of negatives is one of the things they scale. No mention of a [[PinnerFormer- Sequence Modeling for User Representation at Pinterest#^logq-correction|logQ correction]] in the industrial setup.
- Open question the authors flag: because the model learns the joint $p(\Phi_0, a_0, \dots, \Phi_{n_c-1}, a_{n_c-1})$, you could **beam-search a whole slate** of items to show next, replacing listwise heuristics like DPP or RL re-ranking. Not evaluated.
- Practical caveat: none of this is reproducible outside a company with 100B examples/day and 256 H100s. The public-dataset numbers use multi-epoch full-shuffle training, which the authors themselves say does not reflect streaming reality.
- The framing — user actions as a *modality*, unified feature space across recs/search/ads — is an explicit bid for [[Foundation Models]] in recommendation.

## Links

Related: [[Attention Is All You Need]] · [[Flash Attention]] · [[Causal Attention]] · [[Query, Key, and Value (QKV)]] · [[Gated Activation]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[Deep Interest Network for CTR Prediction (DIN)]] · [[BERT4Rec- Sequential Recommendation with Bidirectional Transformer]] · [[PinnerFormer- Sequence Modeling for User Representation at Pinterest]] · [[Recommender Systems - Evolution]] · [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[NDCG]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Mixed Precision training]] · [[Auto-regressive models]] · [[Foundation Models]]

New topics worth writing: SASRec (self-attentive sequential recommendation), GRU4Rec, Relative attention bias / T5 position bias, RoPE and length extrapolation (NTK-aware, YaRN), SwiGLU and GLU variants, Deep & Cross Network (DCN v2), Mixture-of-Experts routing (MMoE, PLE), Sampled softmax and in-batch negatives, KV caching, Grouped GEMM / ragged attention kernels, Dirichlet process, Stochastic depth, Rowwise Adagrad/AdamW for embedding tables, Normalized Entropy as a ranking metric, Two-tower retrieval
