---
title: "Deep Interest Network for CTR Prediction (DIN)"
authors: ["Guorui Zhou", "Chengru Song", "Xiaoqiang Zhu", "Ying Fan", "Han Zhu", "Xiao Ma", "Yanghui Yan", "Junqi Jin", "Han Li", "Kun Gai"]
year: 2018
arxiv: "1706.06978"
url: https://arxiv.org/abs/1706.06978
priority: Must-Read
read_on: 2026-08-22
tags: [paper, transformers]
---
## The Core Idea

Before this paper, nearly every deep click-prediction model looked the same: turn sparse IDs into embeddings, squash the user's whole history into **one fixed vector** by summing or averaging it, concatenate everything, feed an MLP. That one vector is the same no matter which ad you are scoring. If a user has browsed woolen coats, T-shirts, earrings, tote bags, leather handbags and children's coats, the model has to cram all six interests into, say, 12 numbers — and then use the exact same 12 numbers whether it is scoring a handbag or a phone charger.

The fix: **make the user vector depend on the ad**. For each candidate ad, look back over the user's behaviour list, score how relevant each past item is to *this* ad, and take a weighted sum. A handbag ad pulls out the tote-bag and leather-handbag behaviours and mostly ignores the shoes. This is [[Attention Is All You Need#^self-attention|attention]], borrowed from machine translation, but pointed at a user history instead of a sentence.

Two things made it not exist before. One is conceptual: people treated the history as a bag of features to pool, not as something to query. The other is practical — the scoring must happen at serving time for *hundreds of ads per user in under 10ms*, so the idea only becomes deployable with real engineering (request batching, GPU kernel tricks).

> [!NOTE] Local activation unit
> A tiny feed-forward net $a(\cdot)$ that takes one past behaviour embedding and the candidate ad embedding, and outputs a scalar weight. The user representation is a weighted sum of behaviours using those weights. Different ad ⇒ different weights ⇒ different user vector. ^local-activation-unit

The other half of the paper is unglamorous but arguably more valuable: two tricks for training a network with ~600 million distinct `goods_id` features without it overfitting after one epoch or grinding to a halt computing an L2 norm over billions of parameters.

## The Methodology

**Features.** Four groups, all categorical, all one-hot or multi-hot. No hand-crafted cross features — the MLP is supposed to find interactions itself.

| Category | Example | Dim | Nonzero per instance |
|---|---|---|---|
| User profile | gender, age_level | ~10 | 1 |
| User behaviour | visited_goods_ids | ~$10^9$ | ~$10^3$ |
| Ad | goods_id, shop_id, cate_id | ~$10^7$ | 1 |
| Context | pid, time | ~10 | 1 |

**BaseModel (the thing being beaten).** Embedding lookup per feature group, sum/average pooling over the variable-length behaviour list to get a fixed vector, concat, MLP, sigmoid. Loss is plain binary [[Cross Entropy|cross entropy]]:

$$L=-\frac{1}{N}\sum_{(\bm{x},y)}\big(y\log p(\bm{x})+(1-y)\log(1-p(\bm{x}))\big)$$

**DIN.** Identical, except the pooling over behaviours becomes:

$$\bm{v}_U(A)=\sum_{j=1}^{H}a(\bm{e}_j,\bm{v}_A)\,\bm{e}_j$$

where $\bm{e}_j$ is the embedding of the $j$-th past item, $\bm{v}_A$ the ad embedding, $H$ the history length. Inside $a(\cdot)$: the two embeddings *and their element-wise outer/cross product* are fed to a small MLP. The product is handed in explicitly because relevance is fundamentally a similarity, and making the net rediscover multiplication from concatenation is wasteful.

**No softmax.** This is the deliberate deviation from standard attention. The weights are *not* normalised to sum to 1. The reasoning: $\sum_j w_j$ is treated as a proxy for how *intensely* the ad matches the user at all. If 90% of a user's history is clothes, a T-shirt ad should produce a large-magnitude $\bm{v}_U$ and a phone ad a small one. Softmax would erase that scale difference.

**Mini-batch aware (MBA) regularization.** Standard $\ell_2$ touches every embedding row every step — impossible at $10^9$ rows. Rewrite the norm as a per-sample sum, then approximate: let $\alpha_{mj}=1$ if feature $j$ appears anywhere in mini-batch $m$, and $n_j$ be that feature's total count across the dataset. Then

$$L_2(\bm{\mathrm{W}})\approx\sum_{j=1}^{K}\sum_{m=1}^{B}\frac{\alpha_{mj}}{n_j}\|\bm{w}_j\|_2^2$$

giving the update

$$\bm{w}_j\leftarrow\bm{w}_j-\eta\Big[\tfrac{1}{|\mathcal{B}_m|}\textstyle\sum_{\mathcal{B}_m}\tfrac{\partial L}{\partial \bm{w}_j}+\lambda\tfrac{\alpha_{mj}}{n_j}\bm{w}_j\Big]$$

Only rows present in the batch get penalised — same sparsity pattern as the [[Backpropagation|gradient]] itself, so it's nearly free. Note the $1/n_j$: **rare features get penalised harder**, frequent ones lightly. That is the opposite of DiFacto, and it matters (see below). $\lambda=0.01$.

**Dice activation.** PReLU switches channels at a hard threshold of 0. Dice moves the threshold to the data's own mean and softens the switch:

$$f(s)=p(s)\cdot s+(1-p(s))\cdot\alpha s,\qquad p(s)=\frac{1}{1+e^{-\frac{s-E[s]}{\sqrt{Var[s]+\epsilon}}}}$$

with batch statistics at train time and moving averages at test time — i.e. it folds a batch-norm-style standardisation into the [[Gated Activation|gate]]. $\epsilon=10^{-8}$. If $E[s]=0$ and $Var[s]=0$, Dice collapses back to PReLU.

**Training setup.** Amazon/MovieLens: SGD, LR starts at 1 with decay 0.1, batch 32. Alibaba: 2B train / 0.14B test samples, embedding dim 12 across all 16 feature groups, MLP $192\times200\times80\times2$, batch 5000, Adam with LR 0.001 and decay 0.9.

**Metric.** Not plain AUC — **user-weighted AUC**, computed per user then averaged by impression count:

$$\text{AUC}=\frac{\sum_i \#\text{impression}_i\times\text{AUC}_i}{\sum_i \#\text{impression}_i}$$

This measures ranking quality *within* a user, which is what actually matters when you rank ads for one visitor. They also report RelaImpr $=\left(\frac{\text{AUC}-0.5}{\text{AUC}_{base}-0.5}-1\right)\times100\%$, which rescales relative to a random guesser.

## Ablation Studies and Experiments

**Public datasets** (5 runs averaged, init noise < 0.0002 AUC):

| Model | MovieLens AUC | Amazon AUC | Amazon RelaImpr |
|---|---|---|---|
| LR | 0.7263 | 0.7742 | −24.34% |
| BaseModel | 0.7300 | 0.8624 | 0.00% |
| Wide&Deep | 0.7304 | 0.8637 | +0.36% |
| PNN | 0.7321 | 0.8679 | +1.52% |
| DeepFM | 0.7324 | 0.8683 | +1.63% |
| **DIN** | 0.7337 | **0.8818** | **+5.35%** |
| DIN + Dice | 0.7348 | **0.8871** | **+6.82%** |

The gap is much bigger on Amazon (rich behaviour, 801 categories) than MovieLens (21 categories). Attention only helps when there is genuine diversity to attend over.

**Alibaba, full feature set:**

| Model | AUC | RelaImpr |
|---|---|---|
| LR | 0.5738 | −23.92% |
| BaseModel | 0.5970 | 0.00% |
| Wide&Deep | 0.5977 | +0.72% |
| PNN | 0.5983 | +1.34% |
| DeepFM | 0.5993 | +2.37% |
| DIN | 0.6029 | +6.08% |
| DIN + MBA | 0.6060 | +9.28% |
| DIN + Dice | 0.6044 | +7.63% |
| **DIN + MBA + Dice** | **0.6083** | **+11.65%** |

Decomposing: attention alone buys +0.0059 AUC. MBA over dropout buys +0.0031. Dice over PReLU buys +0.0015. So the architecture is the biggest single lever, but the regularizer is over half its size — it is not a footnote.

Absolute AUC near 0.60 looks terrible until you remember this is per-user AUC on hard, already-filtered candidates. The authors state that **0.001 absolute AUC is worth deploying** in their system.

**The regularization ablation is the most instructive part.** Adding fine-grained `goods_ids` (0.6B dims) with *no* regularization: test AUC spikes in epoch 1, then collapses.

| Setting | AUC | RelaImpr |
|---|---|---|
| No goods_ids, no reg | 0.5940 | 0.00% |
| goods_ids, no reg | 0.5959 | +2.02% |
| goods_ids + Dropout (50% of IDs) | 0.5970 | +3.19% |
| goods_ids + Filter (top 20M only) | 0.5983 | +4.57% |
| goods_ids + DiFacto reg | 0.5954 | +1.49% |
| goods_ids + **MBA** | **0.6031** | **+9.68%** |

What did **not** work:
- **DiFacto's regularizer performed worse than a dumb frequency filter** and barely beat no regularization at all. DiFacto penalises *frequent* features less; DIN's MBA penalises *rare* features more via $1/n_j$. The lesson is that in this regime the overfitting comes from long-tail IDs whose embeddings are fit from a handful of impressions.
- **Dropout** stops the collapse but slows convergence noticeably.
- **Frequency filtering** works okay but throws away the tail entirely — you lose the very information the fine-grained features were added for.
- **LSTM over the behaviour sequence gave no improvement at all.** This is a striking negative result. Their explanation: unlike text, a browsing history has several concurrent interests interleaved, with abrupt jumps and endings, so it looks like noise to a model expecting grammar-like sequential structure. Order helps less than relevance. (Later work — DIEN, BERT4Rec-style models — pushed back on this; see [[BERT4Rec- Sequential Recommendation with Bidirectional Transformer]].)

**Online A/B test**, Alibaba display ads, 2017-05 to 2017-06, roughly a month: **+10.0% CTR and +3.8% RPM** versus the deployed BaseModel. DIN then took main traffic.

**Visualisations.** Attention weights line up with human intuition (behaviours similar to the ad get high weight). t-SNE of learned good embeddings shows clean per-category clusters, and colouring points by predicted CTR reveals a *multimodal* density — the user has several separate high-interest regions, which is exactly the "diversity" claim made visually.

## Worth Remembering

- **Dropping softmax is unusual and load-bearing.** The unnormalised weight sum encodes interest *intensity*, not just distribution. If you reimplement DIN with a softmax you have changed the model in a way the authors specifically argue against. That said, they never ablate softmax-vs-no-softmax numerically — it's an argued design choice, not a measured one.
- **The serving cost is real.** BaseModel computes the user vector once and reuses it for all candidates. DIN must recompute it per (user, ad) pair. With hundreds of ads per request and a 10ms budget, they needed request batching, GPU memory access tuning, and concurrent CUDA kernels — which doubled QPS per machine. Any adoption of ad-conditioned user representations inherits this cost.
- **MBA is an approximation, not exact $\ell_2$.** The step from Eq. 5 to Eq. 6 replaces "how many samples in this batch contain feature $j$" with "does any sample contain it" ($\alpha_{mj}\in\{0,1\}$). With batch size 5000 and long-tail IDs this is nearly exact; with dense features it is not.
- **Dice is basically batch norm inside the activation.** It carries the same train/test asymmetry (batch stats vs. moving averages) and the same batch-size sensitivity. It bought +0.0015 AUC — worth having, not worth agonising over.
- The gain concentrates where histories are long and diverse. On MovieLens (21 categories) DIN beat BaseModel by only 1.61% RelaImpr. Do not expect this to help on a domain with few item types.
- Connection worth holding: this is [[Attention Is All You Need|attention]] applied a year *before* Transformers dominated recsys, with the query being an item rather than a token, and with intensity preserved. Compare with how [[PinnerFormer- Sequence Modeling for User Representation at Pinterest]] goes the other way — one ad-independent user embedding, computed offline, precisely because per-candidate scoring is too expensive at Pinterest scale. The two papers make opposite engineering trade-offs on the same axis.
- Follow-up question: how much of the +6.08% would survive if BaseModel were given a larger embedding dim? The paper argues expanding dimensionality is impractical (parameters, overfitting, serving), but never reports the curve.

## Links

Related: [[Attention Is All You Need]] · [[BERT4Rec- Sequential Recommendation with Bidirectional Transformer]] · [[PinnerFormer- Sequence Modeling for User Representation at Pinterest]] · [[Recommender Systems - Evolution]] · [[Regularization]] · [[Cross Entropy]] · [[Gated Activation]] · [[Query, Key, and Value (QKV)]] · [[Linear Projection]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Foundational_RecSys_Ranking_Reading_Plan]] · [[Momentum]] · [[Backpropagation]]

New topics worth writing: AUC and user-weighted AUC, Wide & Deep, DeepFM and factorization machines, PReLU and parametric activations, Batch Normalization, DIEN and sequential interest evolution, embedding table sparsity and serving at scale, t-SNE
`�
