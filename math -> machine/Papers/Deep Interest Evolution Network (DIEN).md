---
title: "Deep Interest Evolution Network (DIEN)"
authors: ["Guorui Zhou", "Na Mou", "Ying Fan", "Qi Pi", "Weijie Bian", "Chang Zhou", "Xiaoqiang Zhu", "Kun Gai"]
year: 2019
arxiv: "1809.03672"
url: https://arxiv.org/abs/1809.03672
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, transformers, vision]
---
## The Core Idea

[[Deep Interest Network for CTR Prediction (DIN)|DIN]] treats each past click as an interest and uses attention to pick the ones relevant to the ad. Two things are missing there. First, a click is not an interest — it is a *symptom* of one. Second, DIN throws away order: it pools the attended behaviours, so "bought a phone then bought a case" looks the same as the reverse.

DIEN fixes both with two stacked [[Long Short-Term Memory (Neural Computation)|GRU]] layers and one extra loss.

1. **Interest extractor layer.** A GRU runs over the click sequence. The hidden state $\mathbf{h}_t$ is meant to be the user's *interest* right after click $t$. But nothing forces it to be — the only training signal is the final click/no-click label at the end, so intermediate states get almost no supervision. So they add an **auxiliary loss**: hidden state $\mathbf{h}_t$ must predict the *next* real click $\mathbf{b}_{t+1}$ against a randomly sampled item the user did not click. Now every step gets its own gradient.

2. **Interest evolving layer.** A second GRU runs over the interest sequence $[\mathbf{h}_1,\dots,\mathbf{h}_T]$, but its **update gate is multiplied by an attention score** computed against the target ad. Interests unrelated to the ad barely move the state; the relevant ones drive it. Different ads get different evolution paths through the same history.

> [!NOTE] Interest drifting
> A user's intentions jump around. She reads books for two weeks, then wants clothes. A plain RNN has one fixed hidden trajectory, so the clothes phase pollutes the book prediction. Attention inside the recurrence lets each target item carve out its own path. ^interest-drifting

Result: AUC 0.6350 → 0.6541 on Alibaba's internal data, and **+20.7% CTR / +17.1% eCPM** in a live Taobao A/B test against the base model (DIN got +8.9% / +6.7%).

## The Methodology

**Features.** Four groups: User Profile (gender, age), User Behavior (list of visited goods ids), Ad (ad_id, shop_id), Context (time). All one-hot, all pushed through per-field embedding tables into dense vectors. Standard embedding + MLP skeleton, same as [[Deep Learning Recommendation Model (DLRM)|DLRM]]-era models.

**BaseModel** = sum-pool the behaviour embeddings, concat with everything else, feed an MLP, train with [[Cross Entropy|log loss]]:

$$L_{target}=-\frac{1}{N}\sum_{(\mathbf{x},y)}\left(y\log p(\mathbf{x})+(1-y)\log(1-p(\mathbf{x}))\right)$$

**Layer 1 — interest extractor.** A vanilla GRU over the ordered behaviour embeddings:

$$\mathbf{u}_t=\sigma(W^u\mathbf{i}_t+U^u\mathbf{h}_{t-1}+\mathbf{b}^u)$$
$$\mathbf{r}_t=\sigma(W^r\mathbf{i}_t+U^r\mathbf{h}_{t-1}+\mathbf{b}^r)$$
$$\tilde{\mathbf{h}}_t=\tanh(W^h\mathbf{i}_t+\mathbf{r}_t\circ U^h\mathbf{h}_{t-1}+\mathbf{b}^h)$$
$$\mathbf{h}_t=(1-\mathbf{u}_t)\circ\mathbf{h}_{t-1}+\mathbf{u}_t\circ\tilde{\mathbf{h}}_t$$

GRU, not LSTM: cheaper, and behaviour sequences at Alibaba are long even over two weeks.

**The auxiliary loss.** For each user $i$, take the real click sequence $\mathbf{e}_b^i$ and a sampled negative sequence $\hat{\mathbf{e}}_b^i$ (items drawn from the catalogue minus what they actually clicked at that step). Then:

$$L_{aux}=-\frac{1}{N}\sum_{i=1}^{N}\sum_t\left[\log\sigma(\mathbf{h}_t^i,\mathbf{e}_b^i[t+1])+\log\left(1-\sigma(\mathbf{h}_t^i,\hat{\mathbf{e}}_b^i[t+1])\right)\right]$$

where $\sigma(\mathbf{x}_1,\mathbf{x}_2)$ is a small network / sigmoid over the concatenation of the two vectors. This is basically [[Distributed Representations of Words and Phrases (negative sampling)|negative sampling]] applied at every timestep of an RNN. Total objective:

$$L = L_{target} + \alpha \cdot L_{aux}, \qquad \alpha = 1$$

Three claimed benefits: hidden states actually mean something; [[Backpropagation|backprop]] through a long GRU gets easier because gradient enters at every step, not just the end; and the embedding table gets far more supervision signal.

**Layer 2 — attention.** Score each interest state against the ad embedding $\mathbf{e}_a$ with a bilinear form, softmaxed over the sequence:

$$a_t=\frac{\exp(\mathbf{h}_t W \mathbf{e}_a)}{\sum_{j=1}^{T}\exp(\mathbf{h}_j W \mathbf{e}_a)}$$

**AUGRU.** Second GRU takes $\mathbf{i}'_t = \mathbf{h}_t$. Everything is a normal GRU except the update gate is scaled by the attention score:

$$\tilde{\mathbf{u}}'_t = a_t \cdot \mathbf{u}'_t$$
$$\mathbf{h}'_t=(1-\tilde{\mathbf{u}}'_t)\circ\mathbf{h}'_{t-1}+\tilde{\mathbf{u}}'_t\circ\tilde{\mathbf{h}}'_t$$

Small $a_t$ → gate near zero → state barely changes → that behaviour is skipped. The final state $\mathbf{h}'_T$ is concatenated with ad, profile and context embeddings and fed to the MLP.

**Industrial setup.** 6 fully-connected layers: 600, 400, 300, 200, 80, 2. Max history length 50. Training set: ads clicked in the last 49 days as targets, with the user's previous 14 days of behaviour as history; test targets from the following day.

**Serving.** This is a recurrent model in an ad system serving >1M users/sec, so latency was the real fight. Three tricks took p-latency from **38.2 ms to 6.6 ms**, 360 QPS per worker:
- element-parallel GRU + kernel fusion (compute each hidden-state element in parallel, merge kernels);
- batching adjacent user requests to keep the GPU fed;
- distilling with "Rocket Launching" ([[Distilling the Knowledge in a Neural Network|distillation]] cousin), shrinking GRU hidden size **108 → 32** at near-equal accuracy.

## Ablation Studies and Experiments

**Data.** Amazon Books (604k users, 368k goods) and Electronics (192k users, 63k goods); reviews treated as behaviours, sorted by time, predict the $T$-th from the first $T-1$. Industrial: 0.8B users, 0.82B goods, 7.0B samples. Public results averaged over 5 runs.

**Public AUC (Electronics / Books):**

| Model | Electronics | Books |
|---|---|---|
| BaseModel | 0.7435 | 0.7686 |
| [[Wide & Deep Learning for Recommender Systems|Wide&Deep]] | 0.7456 | 0.7735 |
| PNN | 0.7543 | 0.7799 |
| DIN | 0.7603 | 0.7880 |
| Two-layer GRU + attention | 0.7605 | 0.7890 |
| **DIEN** | **0.7792** | **0.8453** |

**Industrial AUC:** BaseModel 0.6350, Wide&Deep 0.6362, PNN 0.6353, DIN 0.6428, two-layer GRU+attn 0.6457, GRU+AUGRU (no aux loss) 0.6493, DIEN 0.6541.

**The interest-evolving ablation — this is the interesting one.** Three ways to inject attention into the second GRU, all on top of the same extractor:

| Variant | Electronics | Books |
|---|---|---|
| Two-layer GRU attention | 0.7605 | 0.7890 |
| **AIGRU** (scale the *input*: $\mathbf{i}'_t=\mathbf{h}_t \cdot a_t$) | 0.7606 | 0.7892 |
| **AGRU** (replace update gate with scalar $a_t$) | 0.7628 | 0.7890 |
| **AUGRU** (scale the update gate vector) | 0.7640 | 0.7911 |

**What did not work:** AIGRU is essentially a no-op (+0.0001 on Electronics). Zeroing the *input* of a GRU still changes the hidden state — the recurrence keeps churning — so irrelevant interests still corrupt the trajectory. You must intervene on the gate, not the input. AGRU (borrowed from the DMN+ QA paper) helps a bit on Electronics but does nothing on Books, because collapsing the whole update-gate *vector* into a single *scalar* throws away the per-dimension information about which parts of the state should update. AUGRU keeps both: the gate says *which dimensions*, attention says *how much overall*.

Note the size of these gaps: AUGRU over plain two-layer GRU attention is only +0.0035 / +0.0021 AUC.

**The auxiliary loss is doing most of the work on public data.** GRU+AUGRU alone: 0.7640 / 0.7911. Adding the auxiliary loss (= full DIEN): 0.7792 / 0.8453. That is +0.015 on Electronics and **+0.054 on Books** — an order of magnitude bigger than the AUGRU contribution.

**But it flips industrially.** On the Alibaba data, AUGRU takes 0.6457 → 0.6493 and the auxiliary loss only adds 0.6493 → 0.6541. The authors give two honest reasons: (a) with 7B samples the embedding table is already well-trained, so the extra supervision buys less; (b) the behaviour log spans *all* categories and *all* surfaces of Taobao, while the target ad lives in one scene, so the next-click signal is heterogeneous with respect to the prediction task.

**Visualisation.** They PCA the AUGRU hidden states down to 2D for a history of Computer Speakers → Headphones → Vehicle GPS → SD Cards → Micro SD → External Hard Drives → Headphones → Cases. With target = *Cases* (matches the last behaviour), the trajectory makes a large final jump and the last behaviour gets a high attention score. With target = *Screen Protectors* (unrelated), the path is nearly identical to the "None" baseline where all $a_t$ are equal. So the mechanism visibly does what it claims.

## Worth Remembering

- **Read the ablation asymmetry carefully.** The paper's headline contribution is AUGRU, but on Amazon the auxiliary loss is 3-15× more valuable, and on the industrial data the ranking reverses. If you are copying one idea into your own sequential recommender, the per-step negative-sampling supervision is the cheaper, higher-leverage half — it needs no architecture change, just a second loss term.
- **Amazon Books AUC of 0.8453 vs DIN's 0.7880 is a suspiciously large jump.** These Amazon "review as click" setups are known to be fragile; see [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] and [[On the Difficulty of Evaluating Baselines]]. The industrial numbers (0.6428 → 0.6541) are more believable and still translate to +20.7% CTR live — a reminder that small offline AUC deltas can be big money.
- **Negative sampling detail differs by dataset.** Public: uniformly random items not clicked at that step. Industrial: *impressed but not clicked* ads. The second is a much harder and more honest negative — it conditions on being shown, so it teaches the model preference rather than popularity.
- **Recurrence in a serving path is expensive.** A GRU cannot be evaluated in parallel over time, unlike [[Attention Is All You Need|self-attention]]. The 38.2 ms → 6.6 ms engineering effort is a real cost of this design; [[Self-Attentive Sequential Recommendation (SASRec)|SASRec]]-style attention-only models sidestep it entirely, and [[Actions Speak Louder than Words- Generative Recommenders (HSTU)|HSTU]] later showed you can go much further with pure sequence transduction.
- **Two GRUs stacked and the second one's output is only the last state $\mathbf{h}'_T$.** All the sequence information is squeezed through one vector before the MLP. That is a bottleneck the paper does not question.
- **The attention softmax normalises over the whole history**, so $\sum_t a_t = 1$. With max history 50, average $a_t = 0.02$, meaning most update gates get shrunk heavily and the second GRU's state moves slowly by default. Worth knowing if you re-implement and wonder why $\mathbf{h}'_T$ looks flat.
- **Open question:** the auxiliary loss supervises $\mathbf{h}_t$ to predict the next click. That makes the extractor an [[Auto-regressive models|autoregressive]] next-item model, which is exactly [[Session-based Recommendations with RNNs (GRU4Rec)|GRU4Rec]]. So DIEN is GRU4Rec as a feature extractor, plus target-aware attention on top. Framing it that way makes the design obvious in hindsight.

## Links
Related: [[Deep Interest Network for CTR Prediction (DIN)]] · [[Self-Attentive Sequential Recommendation (SASRec)]] · [[Session-based Recommendations with RNNs (GRU4Rec)]] · [[Long Short-Term Memory (Neural Computation)]] · [[Attention Is All You Need]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[Wide & Deep Learning for Recommender Systems]] · [[Deep Learning Recommendation Model (DLRM)]] · [[Actions Speak Louder than Words- Generative Recommenders (HSTU)]] · [[Distilling the Knowledge in a Neural Network]] · [[Cross Entropy]] · [[Recommender Systems - Evolution]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Foundational_RecSys_Ranking_Reading_Plan]]

New topics worth writing: Gated Recurrent Unit (GRU) mechanics, auxiliary losses for intermediate supervision, PNN (Product-based Neural Networks), interest drifting, kernel fusion for RNN inference, Rocket Launching (light-net co-training), eCPM / PPC / CTR as an ad-system metric triple, ATRank
