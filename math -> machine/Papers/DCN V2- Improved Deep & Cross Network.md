---
title: "DCN V2: Improved Deep & Cross Network"
authors: ["Ruoxi Wang", "Rakesh Shivanna", "Derek Z. Cheng", "Sagar Jain", "Dong Lin", "Lichan Hong", "Ed H. Chi"]
year: 2020
arxiv: "2008.13535"
url: https://arxiv.org/abs/2008.13535
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, vision, theory]
---
## The Core Idea

The original [[Deep & Cross Network for Ad Click Predictions|DCN]] had a clever trick: a "cross layer" that multiplies the input by itself, layer after layer, so that after $l$ layers you have all feature products up to degree $l+1$. Cheap, explicit, no manual feature engineering.

The problem: DCN's cross layer used a **vector** of weights, not a matrix. One cross layer had only $d$ parameters, where $d$ is the size of the embedding input. That is tiny. If your input embedding is 1000-dimensional, the entire "learn every pairwise interaction" job is done by 1000 numbers. On Google-scale data the cross network was so small next to the deep network that nearly all the model capacity went to the DNN, which learns crosses *implicitly* and badly.

DCN-V2's change is almost embarrassingly simple: **replace the weight vector $\mathbf{w}$ with a full weight matrix $W \in \mathbb{R}^{d\times d}$.**

$$\mathbf{x}_{l+1} = \mathbf{x}_0 \odot (W_l \mathbf{x}_l + \mathbf{b}_l) + \mathbf{x}_l$$

That is it. $\odot$ is elementwise (Hadamard) product. DCN is the special case $W = \mathbf{1}\mathbf{w}^\top$ (every row identical), so DCN-V2's function class is a strict superset. The cross network goes from $O(d)$ to $O(d^2)$ parameters per layer, which rebalances capacity against the DNN and lets the model represent *arbitrary* interaction patterns instead of only the rank-one ones DCN could express.

> [!NOTE] Explicit vs implicit feature cross
> **Explicit** means the multiplication $x_i x_j$ appears in a formula you wrote down, with a controllable maximum degree. **Implicit** means you hope a stack of [[ImageNet Classification with Deep CNNs (AlexNet)#^relu|ReLU]] layers approximates it. ReLU nets are universal approximators but are provably bad at cheaply representing products. ^explicit-cross

The second half of the paper is the practical fix for the cost this creates. $d^2$ per layer is too expensive when $d$ is a few thousand. They looked at the singular values of the learned $W$ from a production model and found it decays fast — the matrix is *numerically low rank*. So factor it: $W \approx UV^\top$ with $U,V \in \mathbb{R}^{d\times r}$, $r \ll d$. Then go further and use several small experts with a gate, borrowing from Mixture-of-Experts. That gives **DCN-Mix**.

What this unlocks: an explicit-cross module that is expressive enough to beat a large well-tuned DNN, cheap enough to serve at high QPS, simple enough to drop between an embedding layer and an MLP, and *readable* — the block $W_{i,j}$ literally tells you how much the model cares about the cross of feature $i$ and feature $j$.

## The Methodology

**Embedding layer.** Categorical features get $\mathbf{x}_{\text{embed},i} = W_{\text{embed},i}\mathbf{e}_i$, dense features get normalised, everything is concatenated into $\mathbf{x}_0 \in \mathbb{R}^d$. Multi-valued features are averaged. Important practical detail: unlike DeepFM, xDeepFM, AutoInt, PNN, [[Deep Learning Recommendation Model (DLRM)|DLRM]], **DCN-V2 does not require all embeddings to be the same size.** In production, vocabularies range from 10 to millions; forcing one embedding size is wasteful. The Hadamard structure works because $W_{i,j}$ blocks are rectangular and handle the dimension alignment.

**Cross network.** Stack $L_c$ copies of Eq. (1). Two things to notice:

- The $\mathbf{x}_0 \odot$ term is where the degree grows — each layer multiplies by the *original* input again.
- The $+\mathbf{x}_l$ [[Deep Residual Learning for Image Recognition (ResNet)#^residual-block|residual]] term and the bias $\mathbf{b}_l$ mean layer $l$ contains **all** degrees from 1 to $l+1$, not only degree $l+1$. This turns out to be the thing that makes it trainable (see ablations).

**Deep network.** Plain MLP, $\mathbf{h}_{l+1} = f(W_l\mathbf{h}_l + \mathbf{b}_l)$, $f = $ ReLU.

**Combining the two.** Both variants are offered because which one wins is data-dependent:
- *Stacked*: $\mathbf{x}_0 \to$ cross net $\to$ deep net. Models $f_{\text{deep}} \circ f_{\text{cross}}$. Won on Criteo.
- *Parallel*: both take $\mathbf{x}_0$, outputs concatenated. Models $f_{\text{cross}} + f_{\text{deep}}$. Won on MovieLens-1M.

**Loss.** Logistic output $\hat{y} = \sigma(\mathbf{w}_{\text{logit}}^\top\mathbf{x}_{\text{final}})$, [[Cross Entropy|log loss]] with $L_2$:

$$\text{loss} = -\frac{1}{N}\sum_i y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i) + \lambda\sum_l\|W_l\|_2^2$$

**DCN-Mix, in three escalating steps.**

1. Low rank: $\mathbf{x}_{l+1} = \mathbf{x}_0 \odot (U_l(V_l^\top\mathbf{x}_l) + \mathbf{b}_l) + \mathbf{x}_l$, with $U_l, V_l \in \mathbb{R}^{d\times r}$. Two readings: you learn crosses *in a subspace*, or you project down to $\mathbb{R}^r$ and back up.
2. Mixture of experts, from reading one:
$$\mathbf{x}_{l+1} = \sum_{i=1}^{K} G_i(\mathbf{x}_l)E_i(\mathbf{x}_l) + \mathbf{x}_l, \quad E_i(\mathbf{x}_l) = \mathbf{x}_0 \odot\big(U_l^i(V_l^{i\top}\mathbf{x}_l) + \mathbf{b}_l\big)$$
$G_i$ is a sigmoid or softmax gate on the input. Each expert learns crosses in a different subspace.
3. Nonlinearity in the squeezed space, from reading two: $E_i(\mathbf{x}_l) = \mathbf{x}_0 \odot(U_l^i \cdot g(C_l^i \cdot g(V_l^{i\top}\mathbf{x}_l)) + \mathbf{b}_l)$.

Each formula is a strictly bigger function class at fixed parameter count. Note this is **not** post-training compression — the low-rank structure is imposed before training and learned jointly. Cost: cross net is $O(d^2 L_c)$, DCN-Mix is $O(2drKL_c)$, cheaper when $rK \ll d$.

**Theory.** Theorem 4.1 (bitwise) and 4.2 (feature-wise) prove an $l$-layer cross network reproduces every monomial $x_1^{\alpha_1}\cdots x_d^{\alpha_d}$ with $|\boldsymbol\alpha| \le l+1$, with coefficients that are sums of products of the $W$ entries. DCN reproduces the same *polynomial class* but with far fewer free coefficients, so it cannot hit arbitrary coefficient patterns. DCN-V2 also admits a *feature-wise* view (each embedding block as a unit); DCN is bitwise only.

**Connections.** [[Factorization Machines (ICDM)|FM]] / DeepFM / DLRM = a 1-layer DCN-V2 without residual, with $W$ constrained to $W_{i,j} = w_{ij}I$. xDeepFM's first feature map = same constraint plus a summing projection. [[Attention Is All You Need#^self-attention|Self-attention]] in AutoInt has the same *shape* — each feature attends to a weighted combination of others — but the mixing is `softmax` + weighted sum, whereas DCN-V2 uses $\mathbf{x}_i \odot (W_{i,j}\mathbf{x}_j)$, i.e. an actual multiplication.

**Training setup.** TensorFlow v1, [[Adam- A Method for Stochastic Optimization|Adam]], batch 512 (128 for MovieLens), [[Delving Deep into Rectifiers (He init, PReLU)#^he-init|He Normal]] init, biases 0, gradient clip norm 10, EMA decay 0.9999. Embedding size fixed to $6\cdot(\text{cardinality})^{1/4}$ averaged over vocabs: 39 for Criteo, 30 for MovieLens. Hyperparameters were grid-searched coarse-then-fine for *every* baseline, and every reported number is the mean $\pm$ std over 5 seeds — an explicit response to the reproducibility complaints in [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|Dacrema et al.]] and [[On the Difficulty of Evaluating Baselines|Rendle et al.]]

## Ablation Studies and Experiments

**Synthetic polynomials (RQ1) — the cleanest result in the paper.** They fit known ground-truth polynomials, so there is no noise to hide behind. RMSE over 5 runs:

| target | DCN (1 layer) | DCN-V2 (1 layer) | DNN (1 layer) | DNN (large) |
|---|---|---|---|---|
| $f_1$ (simple, 4 terms) | 8.9e-13 (12 params) | 5.1e-13 (24) | 2.7e-2 (24) | 4.7e-3 (41K) |
| $f_2$ (uneven coefficients) | 1.0e-1 (9) | 4.5e-15 (15) | 3.0e-2 (15) | 1.4e-3 (41K) |
| $f_3$ (100 random crosses in $\mathbb{R}^{100}$) | 2.6e+0 (300) | 6.7e-7 (10K) | 2.7e-1 (10K) | 7.8e-2 (758K) |

Read $f_2$ carefully. DCN's error is $10^{-1}$ — it *cannot fit four terms* when the coefficients are $\{1, 0.1, 1, 0.1\}$, because one shared weight vector forces a rank-one coefficient pattern. And a 758K-parameter DNN cannot get below $7.8\times10^{-2}$ on $f_3$ where DCN-V2 hits $10^{-7}$ with 10K. ReLU stacks are genuinely bad at multiplication.

**Which piece of the cross layer matters.** Fitting a homogeneous degree-3 polynomial: the bare $\mathbf{x}_0 \odot (W\mathbf{x}_l)$ term is best at exactly layer 2 (degree 3 needs 2 layers) and **degrades sharply at every other depth**. Adding bias and residual flattens that curve, because those terms keep all lower-degree crosses alive. Combined-order fitting confirms: DCN-V2 RMSE goes 1.43e-1 → 2.89e-2 → 9.82e-3 for layers 1–3 and then stays flat (9.87e-3, 9.92e-3) instead of collapsing when redundant crosses appear. DNN sits at ~1.0e-1 no matter how deep. **The residual and bias are not decoration — they are what makes depth safe when you do not know the true degree.**

**Cross component alone, no DNN, Criteo, categorical features only (RQ2).** LogLoss: FM 0.4736, PNN(OPNN) 0.4715, CIN 0.4719, AutoInt 0.4711, DNN 0.4704, CrossNet 0.4702, CrossNet-Mix 0.4694. Higher-order beats 2nd-order; cross network beats everything, including a bare DNN.

**Full models (RQ3), Criteo (0.001 LogLoss is the significance threshold here):**

| model | LogLoss | AUC | params | FLOPS |
|---|---|---|---|---|
| PNN | 0.4421 (5.8e-4) | 0.8099 | 3.1M | 6.1M |
| DeepFM | 0.4420 | 0.8099 | 1.4M | 2.8M |
| DLRM | 0.4427 | 0.8092 | 1.1M | 2.2M |
| xDeepFM | 0.4421 | 0.8099 | 3.7M | **32M** |
| AutoInt+ | 0.4420 | 0.8101 | 4.2M | 8.7M |
| DCN | 0.4420 | 0.8099 | 2.1M | 4.2M |
| **DNN** | **0.4421** | **0.8098** | 3.2M | 6.3M |
| DCN-V2 | **0.4406** | **0.8115** | 3.5M | 7.0M |
| DCN-Mix | 0.4408 | 0.8112 | 2.4M | 4.8M |
| CrossNet alone | 0.4413 | 0.8107 | 2.1M | 4.2M |

**The most damning finding in the paper is not about DCN-V2.** Once every baseline is tuned properly and paired with a well-sized DNN, *every single prior method collapses onto the plain DNN*: 0.4420–0.4427 against DNN's 0.4421. Five years of "explicit interaction" papers, and a tuned MLP matches all of them. (Their tuned numbers are also better than what the original papers reported — see appendix Table 9: AutoInt reported DCN at 0.4447, here it is 0.4420.) DCN-V2's 0.4406 is the only real gap.

Their explanation: prior methods fail on *expressiveness* (too few parameters, so a big DNN matches them) or on *trainability* (unstable). xDeepFM has each feature map encoding all pairwise crosses but relies on a **single scalar** $w_{ij}^{kh}$ per cross to learn its importance — hard to learn when jointly trained with millions of other parameters, hence its 4.3e-3 std on MovieLens. PNN's best single run beats DNN on Criteo but its 1.4e-3 std drags the mean to parity. xDeepFM's FLOPS are ~10× its parameter count, which rules it out of production entirely.

**Can cross layers replace ReLU layers?** At matched parameter budgets on Criteo, CrossNet-only beats DNN-only at every budget: at 7.9e5 params 0.4424 vs 0.4427; at 2.6e6, 0.4415 vs 0.4423. A 5-layer cross net was the overall best. The authors flag this as speculative but interesting.

**Hyperparameters (RQ4).**
- *Depth*: steady improvement with more cross layers, decelerating. At $\le 2$ layers a same-sized DNN wins; past that the cross net catches up and overtakes.
- *Rank*: LogLoss falls almost linearly as $r$ goes 4 → 64, then flattens. They call 64 the "threshold rank" and hypothesise it is $O(k)$ where $k$ = number of features (39 on Criteo). Reasoning: split $W_{i,j} = W^L_{i,j} + W^H_{i,j}$; if the dominant part is $W^L_{i,j} = c_{ij}\mathbf{1}\mathbf{1}^\top$ then $W^L$ has rank exactly $k$. Unverified on other datasets.
- *Number of experts — this did not work.* For a 2-layer cross net at total rank 256, LogLoss for $K=$ 1, 4, 8, 16, 32 was 0.4418, 0.4416, 0.4416, 0.4422, 0.4420. **More small experts is no better than one big one, and past 8 it is worse.** They blame naive gating and plain joint optimisation, and suggest Gumbel-softmax, alternating training, temperature tuning as future work. So the MoE part of DCN-Mix is, on these benchmarks, not doing real work — the low-rank factorisation is.

**Interpretability (RQ5).** Because Eq. (7) blocks $W$ by feature, $\|W_{i,j}\|_F$ is an importance score for the cross of features $i$ and $j$. On MovieLens the heatmap surfaces Gender × UserId and MovieId × UserId. Off-diagonal blocks match known-important crosses; diagonal blocks are self-interactions ($x^2$).

**Production at Google.** >100B training examples, vocab sizes 2 to millions, baseline is a plain ReLU MLP. DCN-V2 gave **0.6% AUCLoss** improvement where 0.1% is considered significant, plus online business metric gains. Controlled swap against equal-sized ReLU layers (relative AUCLoss, lower better): 1-layer ReLU 0%, 2-layer ReLU −0.15%, 1-layer DCN-V2 −0.19%, 2-layer DCN-V2 −0.45%. One cross layer beats two ReLU layers.

## Worth Remembering

**Production lessons, which are the most transferable part:**
- Put cross layers **right after the input embeddings**, before the hidden layers. Feature representations lose physical meaning the further you get from the input, so crossing them later is less useful.
- Gains from stacking cross layers **plateau after 2**. Do not go deep.
- Stacking and concatenating both work, but do different things: stacking gets you higher order, concatenating (like multi-head attention) gets you complementary interactions at the same order.
- **Rank $= d/4$ consistently preserved full-rank accuracy.** That is the number to try first.

**Caveats.**
- The MoE gating is the weakest link; $K>1$ bought essentially nothing. If you implement DCN-Mix, start with $K=1$ (plain low-rank) and only add experts if you also improve the gate.
- The threshold-rank $O(k)$ hypothesis rests on one dataset.
- Imposing low rank *before* training makes the cross layer part of the nonlinear composition $f_k(W_k)\circ\cdots\circ f_1(W_1)$, so training dynamics change in ways the authors explicitly say they did not study (they raise Jacobian/Hessian structure as open work — see [[Derivative#Jacobian|Jacobian]], [[Derivative#Hessian|Hessian]]).
- Criteo gains are 0.0015 LogLoss. Real, by this field's convention, but small. The production 0.6% AUCLoss is the more convincing evidence.

**The uncomfortable meta-result.** Table 6 is a quiet replication crisis. Once you tune the baselines honestly and run 5 seeds, DeepFM, DLRM, xDeepFM, AutoInt and DCN are all statistically indistinguishable from a plain MLP on Criteo. This belongs next to [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|Dacrema et al.]] and [[On the Difficulty of Evaluating Baselines|Rendle et al.]] as evidence that reported gains in recommender papers are frequently baseline-tuning artefacts. The authors did the right thing by reporting it against their own favour.

**Open questions.** Does the "cross layers can replace ReLU layers" claim hold outside CTR data? Interaction between DCN-V2 and second-order optimisers? Relationship between embedding size and the rank of the learned $W$?

## Links

Related: [[Deep & Cross Network for Ad Click Predictions]] · [[Deep Learning Recommendation Model (DLRM)]] · [[Factorization Machines (ICDM)]] · [[Wide & Deep Learning for Recommender Systems]] · [[Deep Interest Network for CTR Prediction (DIN)]] · [[Attention Is All You Need]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Cross Entropy]] · [[Adam- A Method for Stochastic Optimization]] · [[Delving Deep into Rectifiers (He init, PReLU)]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[On the Difficulty of Evaluating Baselines]] · [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] · [[Recommender Systems - Evolution]] · [[Linear Projection]] · [[Recommending What Video to Watch Next- A Multitask Ranking System (RecSys)]]

New topics worth writing: Mixture-of-Experts layers and gating functions, singular value decomposition and numerical rank, xDeepFM / Compressed Interaction Network, AutoInt, Product-based Neural Networks, Hadamard product, Criteo CTR benchmark, low-rank matrix factorisation for model compression, Frobenius norm
