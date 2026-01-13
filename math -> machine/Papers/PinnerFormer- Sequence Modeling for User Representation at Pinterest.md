---
title: "PinnerFormer: Sequence Modeling for User Representation at Pinterest"
authors: ["Nikil Pancha", "Andrew Zhai", "Jure Leskovec", "Charles Rosenberg"]
year: 2022
arxiv: "2205.04507"
url: https://arxiv.org/abs/2205.04507
priority: Must-Read
read_on: 2026-08-22
tags: [paper, transformers, vision]
---
## The Core Idea

Most sequence recommenders ask: *what will this user click next?* That question forces you to run the model in real time. Every time a user taps something, their embedding is stale, so you must recompute it — or keep a mutable hidden state per user and pray your streaming pipeline never corrupts it. At Pinterest scale (500M+ users, tens of actions per user per day) both options are expensive and fragile.

PinnerFormer changes the question to: *what will this user positively engage with over the next 14 days?* That is a much smoother target. A user's two-week interests barely move overnight, so an embedding computed once a day is almost as good as one computed live. The training objective is engineered to make offline batch inference viable, not to win a next-item leaderboard.

The concrete trick is the **dense all-action loss**. Take a [[Causal Attention|causally masked]] transformer over the user's last $M$ actions. It emits one embedding per position. Pick random positions, and for each one, ask it to retrieve a randomly chosen *future* positive action drawn from the next $K$ days — not the immediate next action.

> [!NOTE] Dense all-action loss
> Train every (sampled) position of a causal sequence model to predict a randomly sampled positive engagement from a multi-day future window, rather than the literal next item. "Dense" = many positions supervised; "all-action" = the label pool is the whole future window. ^dense-all-action

Two payoffs. First, the gap between daily and realtime inference shrinks: Recall@10 drops 13.9% for a next-action ([[BERT4Rec- Sequential Recommendation with Bidirectional Transformer|SASRec-style]]) objective when you go from realtime to once-only inference, but only 8.3% for dense all-action. Second — and this surprised the authors — dense all-action *also beats* next-action prediction even in the realtime setting (0.264 vs 0.251 R@10).

The second design choice: **one embedding, not many**. The predecessor, PinnerSage, gave each user ~20 cluster embeddings. Great for retrieval, terrible as a feature — 20 × 256 float16 values per row across billions of training rows in dozens of ranking models. PinnerFormer produces a single 256-d vector that drops straight into any downstream ranker.

## The Methodology

**Inputs.** A user's action sequence $\mathcal{A}_U = \{A_1,\dots,A_S\}$ (Pin saves, clicks, close-ups, comments over the past year), truncated to the most recent $M = 256$. Each action becomes a vector by concatenating:

- **PinSage embedding** (256-d, a graph-convolutional Pin representation fusing image, text, and engagement — a pre-existing feature, frozen here).
- Action type and surface — small learnable embedding tables. Out-of-vocab rows are dropped.
- $\log(\text{duration})$ — a single scalar.
- Time features. Three time values (absolute timestamp, time since the user's latest action, gap to the previous action), each encoded [[Fourier Series Decomposition|Time2vec-style]] with **fixed** periods and a log transform:
$$r(t)_{2i-1} = \cos\!\left(\frac{2\pi t}{p_i} + \phi_{2i-1}\right),\quad r(t)_{2i} = \sin\!\left(\frac{2\pi t}{p_i} + \phi_{2i}\right),\quad r(t)_{2P+1} = \log t$$
with $\phi$ learned. Absolute time uses human-meaningful periods (0.25h, 0.5h, 1h, 2h, 4h, 8h, 16h, 1d, 7d, 28d, 365d). Relative time uses $P_{\text{rel}} = 32$ periods log-spaced from 1 second to 4 weeks — deliberately finer resolution at the short end, because "10 seconds vs 1 minute" matters more than "10 days vs 11 days".

**User tower.** Project the concatenated features to hidden dim $H$, add a fully learnable positional encoding, then a standard [[Attention Is All You Need|transformer]] with **PreNorm** residuals (LayerNorm before each sublayer — more stable than the original PostNorm):
$$U^{(l)} = V^{(l-1)} + \operatorname{MHSA}(\operatorname{LayerNorm}(V^{(l-1)}))$$
$$V^{(l)} = U^{(l)} + \operatorname{FFN}(\operatorname{LayerNorm}(U^{(l)}))$$
FFN hidden = $4H$, 8 attention heads, attention **causally masked**. Final model: 6 layers, $H = 768$. Output goes through a 2-layer GELU MLP and is $L_2$ normalised to $D = 256$. The row $e_1$ (most recent position) is the served user embedding.

**Pin tower.** Just an MLP on top of PinSage, also $L_2$ normalised. Both towers normalised was the most stable configuration and cost nothing in offline metrics.

**Loss.** Sampled softmax with a logQ correction. With $s(u,p) = \langle u,p\rangle/\tau$ and a learned temperature $\tau \in [0.01,\infty)$ (lower-bounded for stability):
$$\mathcal{L}(u_i,p_i) = -\log\frac{e^{s(u_i,p_i) - \log Q_i(p_i)}}{e^{s(u_i,p_i) - \log Q_i(p_i)} + \sum_{j=1}^{N} e^{s(u_i,n_j) - \log Q_i(n_j)}}$$

> [!NOTE] Sample probability correction (logQ)
> When negatives are drawn from the batch, popular items appear as negatives far more often than rare ones, so the model unfairly punishes them. Subtract $\log Q_i(v)$ — the probability item $v$ appears in the batch — from each logit to undo that bias. $Q$ is estimated with a count-min sketch. ^logq-correction

**Negatives.** Mixed: up to 5000 in-batch negatives (gathered across all GPUs, with a user's own other positives masked out) plus 8192 uniformly random Pins from the Homefeed-eligible corpus. Each user in a GPU's batch gets equal weight in the averaged loss.

**Positives.** Repin, close-up ≥10s, or long click (>10s), and only on Homefeed — on Search and Related Pins the query supplies the context, so the engagement says less about the user. No explicit negative signals (hides, short clicks) enter the loss. Multi-task by construction: all three action types share one embedding, no task-specific heads, no per-type weights.

**Data layout.** Store one row per user containing their whole timeline; sample sequences and labels on the fly. This trades away some shuffling for the ability to retune sampling knobs (max sequence length, fraction of possible windows sampled, max sequences per user, max labels per sequence) without regenerating terabytes of data. Up to 32 future actions sampled per window; the dense variant then predicts one of them per sampled position.

**Serving.** Daily incremental job: recompute embeddings only for users who engaged in the last 24 hours, merge with yesterday's table, fall back to the old vector if the new one is missing, push to a key-value store. Pin embeddings are regenerated from scratch daily (cheap — one MLP) and compiled into an HNSW index. Because there is no latency budget offline, they can afford a bigger model than realtime serving would permit. Feature corruption is fixed by rerunning inference for affected users.

## Ablation Studies and Experiments

Metric: **Recall@10** — from a random index of 1M Pins, does the embedding at time $t$ retrieve the Pins this user actually engaged with in $(t, t+14d]$? Plus two diversity metrics: **Interest Entropy@50** (per-user variety across ~350 topics) and **P90 Coverage@10** (what fraction of the 1M index accounts for 90% of retrieved results across users — a collapse detector).

**vs PinnerSage** (Table 1). PinnerSage is given an *oracle* advantage: for each positive, use whichever of its top-$c$ cluster embeddings is closest.

| Model | R@10 | Interest Entropy@50 | P90 Cov@10 |
|---|---|---|---|
| PinnerSage (5 clusters, oracle) | 0.026 | 1.69 | 0.130 |
| PinnerSage (20 clusters, oracle) | 0.046 | 2.10 | 0.133 |
| PinnerFormer | **0.229** | 1.97 | 0.042 |

5× the recall of an oracle-boosted 20-embedding baseline, in 1/20th the storage. But note the coverage: 0.042 vs 0.133. PinnerFormer retrieves a narrower slice of the corpus overall.

**Training objective** (Table 3):

| Objective | R@10 | P90 Cov@10 |
|---|---|---|
| Next Action | 0.186 | 0.050 |
| SASRec (softmax) | 0.198 | 0.048 |
| All Action (28d) | 0.224 | 0.028 |
| Dense All Action (14d) | 0.223 | 0.043 |
| **Dense All Action (28d)** | **0.229** | 0.042 |

Two things worth noticing. A **28-day** training window beats a 14-day one *even though evaluation is fixed at 14 days* — more labels per sequence, better sample efficiency. And dense all-action beats plain all-action mostly on diversity (0.042 vs 0.028): in plain all-action, every positive's [[Backpropagation|gradient]] flows through the same single embedding $e_1$, so they average out into a blurry mean interest. Dense spreads them across different transformer outputs, and averaging only happens *inside* the transformer.

**What did not work:** summing losses from multiple objectives together never beat any single-objective model.

**Negatives and logQ** (Table 4):

| logQ | Negatives | P90 Cov@10 | R@10 |
|---|---|---|---|
| N | random | 0.002 | 0.136 |
| N | in-batch | 0.163 | 0.071 |
| N | mixed | 0.083 | 0.138 |
| Y | random | 0.001 | 0.139 |
| Y | in-batch | 0.119 | 0.167 |
| **Y** | **mixed** | 0.042 | **0.229** |

Random negatives alone collapse the model: 1000 Pins out of 1M cover 90% of results for 100k users — it has learnt popularity, not people. In-batch negatives alone preserve diversity but give poor recall. logQ correction does nothing for random negatives (correctly — uniform sampling is already unbiased) and is a large win for in-batch (0.071 → 0.167) and mixed (0.138 → 0.229). Every recall gain here costs global diversity.

**Multi-task** (Table 5). Train on one action type, evaluate on each:

| Train ↓ / Eval → | 10s Closeup | 10s Click | Repin | All |
|---|---|---|---|---|
| 10s Closeup | **0.27** | 0.02 | 0.09 | 0.17 |
| 10s Click | 0.01 | **0.49** | 0.01 | 0.12 |
| Repin | 0.15 | 0.03 | **0.17** | 0.13 |
| Multi-task | 0.23 | 0.28 | 0.13 | **0.23** |

Single-task always wins its own task. Multi-task is second-best everywhere and best overall — the deliberate compromise for a shared feature. Note how specialised each signal is: a click-trained model gets 0.01 on close-ups.

**Feature ablation** (Table 6, drop one at a time from 0.229 baseline): PinSage → 0.142 and coverage collapses to 0.0005 (without content, there is nothing to distinguish users). Timestamp → 0.210. Surface → 0.224. Action type → 0.226. Duration → 0.226. Positional encoding → 0.228 (nearly free — the time features already carry order). Everything hurts, so everything stays.

**Sequence length.** Steady gains doubling up to ~32, then diminishing. They stop at 256 not because 512/1024 are worse but because those need 16–32 GPUs, and changing batch size changes the in-batch negative pool, which breaks comparability.

**Embedding dimension.** Diminishing returns past 128. 256 chosen to match existing Pinterest feature sizes; 1024 was not worth 4× storage. Small dimensions collapse toward popularity memorisation.

**Model capacity** (Table 9). Monotone improvement: 2L/256 → 0.2189, 6L/768 → 0.2293. Head count (8) made no difference.

**SASRec fixes** (Table 10). Their SASRec baseline is not vanilla. Binary cross-entropy: 0.138. Swap to sampled softmax: 0.181. Also weight the loss on $e_1$ equally with other positions: 0.198. So the "SASRec is worse" comparison is against a *strengthened* SASRec.

**Sparsifying sequences** (dropping weak engagements from heavy users' histories) gave no significant gain — they tried "thoroughly".

**Online A/B.** Homefeed ranking, replacing the weighted PinnerSage aggregate with PinnerFormer: Homefeed Repins +7.5%, Close-ups +6%, Clickthroughs +1%, Time Spent +1%, DAU +0.4%, WAU +0.12%. No regression over several months. Ads ranking (adding PinnerFormer *alongside* PinnerSage): CTR +7.1% / +7.3% / +10.0% on Related Pins / Search / Homefeed; long-click CTR +6.9% / +5.2% / +10.1% — evidence the embedding transfers past the surface it was trained on.

## Worth Remembering

- **The recall/diversity tension is everywhere in this paper.** Every change that raises Recall@10 lowers P90 Coverage@10 — logQ correction, random negatives, larger embedding dimensions, longer prediction windows. PinnerFormer's coverage (0.042) is a third of PinnerSage's (0.133). If you deploy this as a *candidate generator* rather than a ranking feature, that narrowing may bite. The authors list candidate generation as future work, not a shipped result.
- **Recall@10 ≈ 0.23 sounds low but is not a probability of a good recommendation.** It is "did the true engaged Pin beat all but 9 of 1M random Pins", and the target is *every* Pin engaged over 14 days. Only compare within the table.
- **The offline evaluation is somewhat aligned with the training objective by construction.** Dense all-action wins partly because "retrieve all 14-day positives" is what it was trained for and next-action prediction was not. The realtime comparison (Table 2), where dense all-action still wins, is the more honest evidence.
- **Popularity is the failure mode to watch.** Random-negatives-only, tiny embeddings, and no-PinSage all fail the same way: retrieving the same globally engaging Pins for everyone while scoring plausibly on per-user diversity. P90 Coverage is the diagnostic that catches it — Interest Entropy alone does not.
- **No negative signals at all.** Hides and short clicks exist in the log and are simply excluded. Both a limitation and, presumably, a next iteration.
- Only Pin engagements are in the sequence — no searches, no board creations, no impressions without engagement. The authors flag broadening this.
- **The dataset trick generalises.** Storing one row per user and sampling windows at training time is why they could sweep sequence length, sampling fraction, and label count without regenerating data. Worth stealing for any sequential-recommendation project.
- Practical caveat on scaling: they could not test sequences beyond 256 because batch size and negative-pool size are entangled with GPU count. If you change one, your loss changes and cross-run comparison breaks. Fix your negative pool before you sweep anything else.
- The daily-inference framing is a systems argument dressed as a modelling result. The modelling contribution is one sampled loss; the value is that it makes a much cheaper serving architecture nearly as good. That is a good template — [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)|infrastructure cost is a first-class objective]].

## Links

Related: [[Attention Is All You Need]] · [[Causal Attention]] · [[BERT4Rec- Sequential Recommendation with Bidirectional Transformer]] · [[Recommender Systems - Evolution]] · [[Foundational_RecSys_Ranking_Reading_Plan]] · [[Cross Entropy]] · [[Fourier Series Decomposition]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Mode Collapse]] · [[NDCG]] · [[Backpropagation]]

New topics worth writing: Sampled softmax and logQ correction, Two-tower retrieval models, In-batch vs mixed negative sampling, Count-min sketch, PinSage and GraphSAGE, HNSW approximate nearest neighbour search, Time2vec time encoding, PreNorm vs PostNorm transformers, Recall@k for retrieval evaluation, SASRec, Multi-task learning with shared embeddings
