---
title: "Recommending What Video to Watch Next: A Multitask Ranking System (RecSys)"
authors: ["Zhao et al."]
year: 2019
url: https://daiwk.github.io/assets/youtube-multitask.pdf
priority: Must-Read
read_on: 2026-08-25
tags: [paper]
---
## The Core Idea

A ranking model for "what video to watch next" has to answer two questions at once, and they fight each other.

**Question one: what do users actually want?** A click is not the same as satisfaction. You can click a video, watch ten seconds, and hate it. So the system predicts *many* user behaviours at the same time — click, watch time, like, dismiss, survey rating — and mixes them into one score. The problem: some of these tasks want the model to learn opposite things. Clickbait maximises clicks and destroys satisfaction. A normal multi-task network with one shared trunk forces all tasks through the same features, so the conflicting tasks drag each other down.

The fix is **Multi-gate Mixture-of-Experts (MMoE)**: replace the shared trunk with a small bank of parallel sub-networks ("experts"), and give each task its own tiny gate that decides how much of each expert it wants. Sharing becomes *soft* and learned, not hard-wired. Cost stays about the same as the shared trunk.

**Question two: was that click real, or just position?** The training logs come from the current ranker. Users click the top slot because it is the top slot. Train on that, and the model learns "the old system's ordering is good" — a feedback loop.

The fix here is the actually interesting bit. Add a tiny **shallow tower** whose only inputs are the bias-causing features (position on screen, crossed with device type). Its scalar output is added to the main model's logit *before* the sigmoid. So the training label gets split into two additive pieces in logit space:

$$\text{logit}(\text{click}) = \underbrace{f_{\text{main}}(\text{user, video, context})}_{\text{true utility}} + \underbrace{f_{\text{shallow}}(\text{position, device})}_{\text{position bias}}$$

At serving time, drop the shallow tower. You are left with utility only. No random-traffic experiment needed to estimate propensities (contrast [[Unbiased Learning-to-Rank with Biased Feedback]], which needs an intervention or a counterfactual estimator), and the bias term re-learns itself continuously as the model trains on fresh logs. That last property matters on YouTube, where popularity distributions shift daily.

The whole thing is a [[Deep Neural Networks for YouTube Recommendations (RecSys)|Wide & Deep]]-style architecture where the "wide" part is the bias tower.

> [!NOTE] Soft parameter sharing
> Tasks share a pool of sub-networks but each task learns *its own weighting* over that pool, instead of being forced through one identical trunk. ^soft-parameter-sharing

> [!NOTE] Position bias
> Items shown higher get clicked more regardless of quality. Any model trained on raw click logs will confuse "was on top" with "was good". ^position-bias

## The Methodology

**Where this sits.** Stage two of the classic [[Deep Neural Networks for YouTube Recommendations (RecSys)#^two-stage-funnel|two-stage funnel]]. Several candidate generators (topic match, co-watch counts, a sequence model, plus the Gramian-estimation retrieval of Krichene et al.) each throw in results; the pool of a few hundred candidates gets scored here.

**Point-wise, on purpose.** Each candidate is scored independently. Pair-wise or list-wise scoring would help diversity but needs multiple passes over the candidate set at serving time. Latency wins.

**The objectives.** Two families:
- *Engagement* — clicks (binary, [[Cross Entropy]] loss), time spent (regression, squared loss).
- *Satisfaction* — likes and dismissals (binary), survey ratings (regression, squared loss).

Each objective is one head. The final ranking score is a **weighted multiplication** of the head outputs, with weights **hand-tuned** by humans to balance engagement against satisfaction. This is not learned. It is the knob product managers turn.

**The MMoE layer.** For task $k$ with $n$ experts:

$$y_k = h_k(f^k(x)), \qquad f^k(x) = \sum_{i=1}^{n} g^k_{(i)}(x)\, f_i(x)$$
$$g^k(x) = \mathrm{softmax}(W_{g^k} x)$$

- $x$ is a **shared bottom hidden layer**, not the raw input. Feeding experts straight off the input layer would be better for modularising the multi-modal features, but the input layer is far wider than a hidden layer, so the multiply count explodes. Efficiency wins again.
- Each expert $f_i$ is a plain MLP with [[ImageNet Classification with Deep CNNs (AlexNet)#^relu|ReLU]] — see [[Deep Learning#^mlp|MLP]].
- Each gate $g^k$ is a single linear map plus softmax. Cheap.
- $h_k$ is the task-specific tower on top.
- **Few experts (4 or 8), all dense.** Unlike Shazeer's sparsely-gated MoE with thousands of experts and top-$k$ routing, every expert runs for every example. The point is *sharing structure*, not conditional compute.

**The shallow tower.** Input: position feature crossed with device type (different devices show different bias curves). Trained jointly with the main model, output added to the final logit. Two details that matter:
- **10% dropout on the position feature during training**, so the model cannot lean entirely on position.
- **At serving, position is fed as missing.** This is what removes the bias.

**Training.** TensorFlow, TPUs, TFX Servo for serving. Models are trained **sequentially in time order** — walk forward through days of logs and keep consuming new data as it arrives, rather than shuffling a fixed dataset. Distributed mini-batch SGD.

**Metrics.** Offline: AUC for classification heads, squared error for regression heads. Online: A/B tests on time spent (engagement) and survey rating scores (satisfaction).

## Ablation Studies and Experiments

**MMoE vs shared-bottom (live A/B on YouTube).** Model complexity is measured in multiplications, held equal across arms.

| Architecture | Multiplications | Engagement | Satisfaction |
|---|---|---|---|
| Shared-Bottom (baseline) | 3.7M | — | — |
| Shared-Bottom | 6.1M | +0.10% | +1.89% |
| MMoE, 4 experts | 3.7M | +0.20% | +1.22% |
| MMoE, 8 experts | 6.1M | **+0.45%** | **+3.07%** |

Read the middle row carefully: just making the shared-bottom model bigger already buys +1.89% satisfaction. MMoE at the *same* size buys +3.07%. So roughly half the satisfaction gain is capacity and half is the architecture; on engagement the architecture is doing 4× more than raw capacity.

**Position bias (live A/B), engagement metric:**

| Method | Engagement |
|---|---|
| Position as a plain input feature | **−0.07%** |
| Adversarial (gradient-reversal on a position-prediction head) | +0.01% |
| Shallow tower | **+0.24%** |

Two things did **not** work. Shoving position in as an ordinary input feature — the standard industry hack — actually *hurt*. And adversarial de-biasing, where an auxiliary head predicts position and its gradient is negated on the way back into the main model, was a wash. Only the additive factorisation, which gives the bias its own explicit place to live, helped.

**What the learned bias looks like.** Raw CTR falls steeply from position 1 to 9 (that is bias *plus* the fact that better items really are at the top). The shallow tower's learned bias also falls monotonically with position, and is smaller for lower positions — it has separated out the propensity component from the utility component.

**Gate input ablation (negative result).** They tried feeding the gating networks the raw input layer instead of the shared hidden layer, hoping the gates would pick experts based on raw features directly. **No substantial live difference.** The hidden layer already carries enough signal for routing.

**Gating polarisation — a real failure mode.** With distributed training, about **20% of runs** collapsed into a state where the softmax gates put near-zero weight on most experts. Tasks stuck with a polarised gate perform badly. Fix: **dropout on the gates** — zero out each expert's utilisation with probability 10%, then renormalise the softmax. This eliminated polarisation across all gates. (Analogous to ReLU-death and to the load-balancing problem in sparse MoE.)

**Expert utilisation plot.** Engagement tasks spread across several experts and share with each other. Satisfaction tasks concentrate on a small, heavily-used subset. That is the conflict the architecture was built to handle, made visible.

## Worth Remembering

- **The gains are tiny in absolute terms and that is the point.** +0.45% engagement at YouTube scale is enormous. Do not read these deltas as weak.
- **The combination weights are hand-tuned.** For all the machinery, the final trade-off between "watched" and "liked" is a human decision baked into a multiplicative formula. Nothing in the paper learns it.
- **Offline metrics did not track online metrics.** They say it plainly: per-task AUC improvements often failed to transfer to live results, which pushed them toward simpler models that generalise better online. Echoes of [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]].
- **Why not a Transformer.** They explicitly rejected importing architectures from NLP/vision: multi-modal feature spaces (sparse IDs + text + thumbnails), power-law sparse features with high feedback variance on tail items, and the fact that architectures tuned for one signal (feature crosses, sequences) tend to improve one objective and hurt others. Compare with [[Actions Speak Louder than Words- Generative Recommenders (HSTU)]], which five years later took exactly the opposite bet and won.
- **Bias handling is one bias only.** Position bias is modelled; everything else — presentation bias, trust bias, biases from how training data is even extracted — is acknowledged as unsolved. Automatically discovering unknown biases is listed as future work.
- **Practical caveat if you build this:** the whole de-biasing trick depends on the *additive-in-logit* factorisation and on setting position to missing at serving. If your bias interacts multiplicatively with utility, or if the missing-value embedding is untrained, it silently breaks. The device cross exists because they measured different bias curves per device — check yours.
- **Related follow-ups they flag:** SNR (Sub-Network Routing) for more stable parameter sharing, and ranking [[Distillation|distillation]] for cutting serving cost.

## Links

Related: [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Unbiased Learning-to-Rank with Biased Feedback]] · [[Entire Space Multi-Task Model (ESMM)]] · [[Deep Interest Network for CTR Prediction (DIN)]] · [[Counterfactual Reasoning and Learning Systems]] · [[Cross Entropy]] · [[Loss, Objectives, and Business Alignment]] · [[Recommender Systems - Evolution]] · [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Calibrated Recommendations (RecSys)]] · [[Actions Speak Louder than Words- Generative Recommenders (HSTU)]] · [[Regularization]]

New topics worth writing: Multi-gate Mixture-of-Experts (MMoE), Sparsely-Gated Mixture-of-Experts layer, Wide & Deep Learning, Multi-task learning and negative transfer, Gradient reversal / adversarial de-biasing, Inverse propensity scoring, Learning-to-rank (point-wise vs pair-wise vs list-wise), Feedback loops in recommender systems, SNR: Sub-Network Routing
