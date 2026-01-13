---
title: "Behavior Sequence Transformer for E-commerce (BST)"
authors: ["Qiwei Chen", "Huan Zhao", "Wei Li", "Pipei Huang", "Wenwu Ou"]
year: 2019
arxiv: "1905.06874"
url: https://arxiv.org/abs/1905.06874
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, transformers]
---
## The Core Idea

Taobao's ranking model was a [[Wide & Deep Learning for Recommender Systems|Wide & Deep]] network: throw every feature into an embedding table, concatenate, run through an MLP, sigmoid out a click probability. That pipeline throws away **order**. It knows you clicked a phone, a case, and a pair of trousers. It does not know you clicked the phone *first*.

Order matters commercially. You buy a phone, then you want a case. You buy trousers, then you want shoes. A bag-of-clicks model cannot represent "then".

[[Deep Interest Network for CTR Prediction (DIN)|DIN]] had already improved on Wide & Deep by weighting past clicks by how similar they are to the candidate item. But DIN's attention is between *target and each past item independently* — the past items never talk to each other, so still no sequence.

The move here is small and obvious in hindsight: drop one [[Attention Is All You Need|Transformer]] encoder block on top of the user's last 20 clicked items, let the items attend to each other, take the output vector for the target item, and concatenate that into the same old MLP. Everything else in the production stack stays the same.

Two things make this a paper rather than a footnote:

1. **It shipped.** Online CTR at Taobao went up **+7.57%** over the Wide & Deep control, versus +4.55% for DIN. Serving latency went from 13 ms to 20 ms, which was acceptable.
2. **One block is best.** $b=1$ beat $b=2$ and $b=3$ on offline AUC. Deeper hurt. User click sequences are not sentences — the dependencies are shallow.

> [!NOTE] Behavior sequence
> The ordered list of items a user clicked, $S(u) = \{v_1, \dots, v_n\}$, treated as a token sequence for a Transformer rather than as an unordered set of features. ^behavior-sequence

## The Methodology

The task is CTR prediction: given user $u$'s click history $S(u)$ and a candidate item $v_t$, predict $P(\text{click})$.

**Input, split in two.**

*Left branch — "Other Features."* User profile (gender, age, city), item features (category_id, shop_id, tag), context (match_type, display position, page number), and hand-crafted cross features (age × item_id, gender × category_id). All embedded through one matrix $\mathbf{W}_o \in \mathbb{R}^{|D| \times d_o}$ and concatenated. This is the classic [[Wide & Deep Learning for Recommender Systems|Wide & Deep]] feature soup, untouched.

*Right branch — the sequence.* The last 20 clicked items, **plus the target item appended to the end of the sequence**. That last part is the whole trick: the target sits inside the sequence so self-attention lets it read every past click and every past click reads it.

Each item in the sequence gets only **two** sparse features: `item_id` and `category_id`. Not the hundreds of features available. A prior Alibaba paper found these two are enough, and embedding more per position is too expensive at 47 billion training samples.

**Positional embedding, done differently.** Instead of the sine/cosine functions from the original Transformer, position is a *feature* fed in at the bottom:

$$\text{pos}(v_i) = t(v_t) - t(v_i)$$

where $t(v_t)$ is the current recommendation time and $t(v_i)$ is when the click happened. So position is **time elapsed since that click**, not rank in the list. That value is bucketed and embedded, concatenated onto the item embedding. They report this beat sin/cos in their setting. It carries real information sin/cos cannot: two clicks one second apart are a different signal from two clicks a week apart.

**The Transformer block.** Standard scaled dot-product [[Causal Attention|attention]] (non-causal — every item sees every item):

$$\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}$$

with 8 heads, $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V \in \mathbb{R}^{d \times d}$ derived from the item embedding matrix $\mathbf{E}$ ([[Query, Key, and Value (QKV)|QKV]]). Then residual + [[Layer Normalization|LayerNorm]] + [[Dropout- A Simple Way to Prevent Overfitting|dropout]]:

$$\mathbf{S}' = \text{LayerNorm}\big(\mathbf{S} + \text{Dropout}(\text{MH}(\mathbf{S}))\big)$$
$$\mathbf{F} = \text{LayerNorm}\big(\mathbf{S}' + \text{Dropout}(\text{LeakyReLU}(\mathbf{S}'\mathbf{W}^{(1)} + b^{(1)})\mathbf{W}^{(2)} + b^{(2)})\big)$$

Note **LeakyReLU**, not ReLU, in the feed-forward layer. Small deviation from the original.

**Head.** Take only the output row corresponding to the target item. Concatenate with the Other Features embeddings. Three fully connected layers, shape $1024 \times 512 \times 256$, sigmoid. [[Cross Entropy|Cross-entropy]] loss:

$$\mathcal{L} = -\frac{1}{N}\sum_{(x,y)\in\mathcal{D}} \big(y \log p(x) + (1-y)\log(1-p(x))\big)$$

**Training config.** TensorFlow 1.4, Python 2.7, **Adagrad** optimiser (not [[Adam- A Method for Stochastic Optimization|Adam]]), learning rate 0.01, batch size 256, dropout 0.2, embedding sizes 4–64 depending on feature cardinality, sequence length 20, 8 heads, **1 transformer block**, and — notably — **1 epoch**.

**Data.** 8 days of Taobao App logs. First 7 days train, last day test. 298 million users, 12 million items, **47.5 billion samples**. One epoch over 47 billion samples is plenty.

## Ablation Studies and Experiments

Everything on the same Taobao dataset. Offline metric is AUC; online is CTR gain over the production control group and average response time (RT) per request.

| Method | Offline AUC | Online CTR gain | Avg RT (ms) |
|---|---|---|---|
| WDL | 0.7734 | — (control) | 13 |
| WDL(+Seq) | 0.7846 | +3.03% | 14 |
| DIN | 0.7866 | +4.55% | 16 |
| **BST ($b=1$)** | **0.7894** | **+7.57%** | 20 |
| BST ($b=2$) | 0.7885 | — | — |
| BST ($b=3$) | 0.7823 | — | — |

**WDL(+Seq) is the ablation that matters most.** It is Wide & Deep with the past item embeddings simply **averaged** and concatenated in. No attention, no order. It jumps +0.0112 AUC over plain WDL and +3.03% online — that is roughly **60% of the total gain from just having the item history present at all**, in the crudest possible form. Self-attention adds the remaining 0.0048 AUC, but that translated to a disproportionately large +4.5 points of extra online CTR.

Read cynically: the biggest win is "use the behaviour sequence." The Transformer is the refinement, not the revelation.

**What did not work: stacking blocks.** $b=2$ is slightly worse than $b=1$; $b=3$ is clearly worse (0.7823, below DIN). The authors' explanation is that dependencies among clicked items are simply not as deep as dependencies among words in a sentence. [[Self-Attentive Sequential Recommendation (SASRec)|SASRec]] reported the same shallow-is-better finding. Given only one epoch and no per-position regularisation beyond dropout 0.2, overfitting or optimisation difficulty in deeper stacks is also plausible — they did not disentangle these.

**What did not work: sin/cos positional encoding.** Replaced with the time-gap feature. No number is given for the sin/cos version — this is a claim, not a measured ablation.

**Latency.** 20 ms vs 13 ms baseline. A 54% increase, but absolutely small enough to ship. This is the practical contribution: a Transformer *can* live inside a rank-stage service handling hundreds of millions of users.

## Worth Remembering

- **Putting the target item inside the sequence** is the design choice to steal. It turns "score this candidate against the history" into a single self-attention pass, and gives you the target-aware behaviour that [[Deep Interest Network for CTR Prediction (DIN)|DIN]] gets explicitly, plus item-to-item interaction for free.

- **Position as elapsed time** generalises well beyond this paper. Recency decay is real in recommendation and pure ordinal position cannot express it. Compare with [[Deep Neural Networks for YouTube Recommendations (RecSys)#^example-age|example age]] in the YouTube paper — same instinct, different mechanism.

- **The paper is thin.** Four pages, no public dataset, no released code, no error bars, no reproduction. The AUC deltas (0.7866 → 0.7894) are tiny and the authors defend them with "from our practical experience, even the small gain of offline AUC can lead to huge gain in online CTR." That is an assertion, not evidence. The offline/online correlation here is 3-point-something percent CTR per 0.002 AUC in one case and larger in another — not a stable relationship. Treat this the way [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|Dacrema et al.]] and [[On the Difficulty of Evaluating Baselines|the baselines paper]] would.

- **Sequence length 20 is short.** Long-history modelling (the direction [[Actions Speak Louder than Words- Generative Recommenders (HSTU)|HSTU]] took years later, with thousands of tokens) was out of reach at this latency budget. BST is the "Transformers work here" proof; HSTU is the "now scale them" follow-up.

- **Only 2 features per item.** If you want to reproduce this, resist adding more item-side features into the sequence — the authors deliberately kept it to `item_id` + `category_id` and put everything else in the flat Other Features branch. Sequence positions are expensive; wide flat features are cheap.

- **Adagrad, one epoch, LeakyReLU.** Small unexplained deviations from standard practice. Adagrad over Adam is common in sparse-embedding recsys (per-parameter learning rates suit rarely-updated embedding rows), but they never say why.

- **Open question:** how much of the online +7.57% is the Transformer versus the fact that BST simply had a longer, better-preprocessed history than the control? The paper does not run BST with a shuffled sequence, which is the one ablation that would prove order — not just presence — is what the attention layer is using.

## Links

Related: [[Attention Is All You Need]] · [[Deep Interest Network for CTR Prediction (DIN)]] · [[Wide & Deep Learning for Recommender Systems]] · [[Self-Attentive Sequential Recommendation (SASRec)]] · [[BERT4Rec- Sequential Recommendation with Bidirectional Transformer]] · [[Actions Speak Louder than Words- Generative Recommenders (HSTU)]] · [[Deep Interest Evolution Network (DIEN)]] · [[Layer Normalization]] · [[Cross Entropy]] · [[Query, Key, and Value (QKV)]] · [[Dropout- A Simple Way to Prevent Overfitting]] · [[RoFormer- Enhanced Transformer with Rotary Position Embedding]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Recommender Systems - Evolution]] · [[Deep Learning Recommendation Model (DLRM)]] · [[AutoInt- Automatic Feature Interaction Learning via Self-Attention]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]]

New topics worth writing: Adagrad, LeakyReLU, AUC vs online CTR correlation in ranking systems, target-aware attention, DeepFM, xDeepFM, latency budgets in rank-stage serving
```
