---
title: "Entire Space Multi-Task Model (ESMM)"
authors: ["Xiao Ma", "Liqin Zhao", "Guan Huang", "Zhi Wang", "Zelin Hu", "Xiaoqiang Zhu", "Kun Gai"]
year: 2018
arxiv: "1804.07931"
url: https://arxiv.org/abs/1804.07931
priority: Must-Read
read_on: 2026-08-22
tags: [paper, optimization, vision, theory]
---
## The Core Idea

You want to predict: if a user clicks this item, will they buy it? That is **post-click conversion rate**, or CVR.

The obvious way is broken. You only *observe* whether someone bought after they clicked. So you train only on clicked impressions. But at serving time you score *every* impression, clicked or not. You trained on a small, weird slice of the world and then asked the model about the whole world.

Two problems fall out of that:

1. **Sample selection bias.** Train space (clicks) ≠ inference space (all impressions). The clicked set is shaped by the randomness of clicking, which itself varies across the feature space. The model generalises badly.
2. **Data sparsity.** Clicks are rare, conversions rarer. In Taobao's product data: 8,950M impressions → 324M clicks → 1.77M conversions. The CVR training set is ~4% the size of the CTR training set.

The trick: never train CVR directly. Use the chain rule of probability on the user's action sequence $\text{impression} \to \text{click} \to \text{conversion}$:

$$\underbrace{p(y=1,z=1\mid \bm{x})}_{\text{pCTCVR}} = \underbrace{p(y=1\mid \bm{x})}_{\text{pCTR}} \times \underbrace{p(z=1\mid y=1,\bm{x})}_{\text{pCVR}}$$

where $y$ is click and $z$ is conversion. Both pCTR and pCTCVR have labels on **every** impression — you always know if it was clicked, and you always know if it was clicked-and-bought. So train those two over the full space. pCVR becomes a hidden intermediate quantity, never given a loss of its own, learned only because it has to make the product come out right.

That is the whole insight. The label you cannot observe everywhere is factored out of two labels you *can* observe everywhere.

> [!NOTE] Entire-space modelling
> Train only on tasks whose labels are defined over the full inference space, and recover the biased-sample quantity as a factor inside those tasks. The unobservable conditional is never fit directly; it is squeezed out by the constraint. ^entire-space

Why not just divide? $\text{pCVR} = \text{pCTCVR}/\text{pCTR}$ with two separately trained models. The paper calls this `DIVISION` and it is a real baseline. It fails badly in practice because pCTR is a tiny number, and dividing by a tiny noisy number is numerically unstable — the resulting pCVR can even exceed 1. The multiplication form keeps everything bounded in $[0,1]$ and lets the three estimators co-train and share gradient signal.

## The Methodology

Two sub-networks, both the standard Embedding & MLP shape you'd see in [[Deep Interest Network for CTR Prediction (DIN)|DIN]] or Wide & Deep.

**Shared bottom.** Sparse multi-field input $\bm{x}$ (user field, item field, etc.) → embedding lookup → pooled into a dense vector. **The embedding dictionary is shared between the CTR network and the CVR network.** This is the second half of the paper's contribution.

**Two towers on top.** CVR tower on the left, CTR tower on the right. Each is an MLP of dimensions $360 \times 200 \times 80 \times 2$, ReLU activations, embedding dimension 18. Each ends in a sigmoid-ish head giving $f(\bm{x};\theta_{cvr})$ and $f(\bm{x};\theta_{ctr})$.

**Third output, no parameters.** pCTCVR is just the product of the two tower outputs. No extra weights.

**Loss.** Two [[Cross Entropy|cross-entropy]] terms, both summed over *all* $N$ impressions:

$$L(\theta_{cvr},\theta_{ctr}) = \sum_{i=1}^{N} l\big(y_i,\; f(\bm{x}_i;\theta_{ctr})\big) + \sum_{i=1}^{N} l\big(y_i \& z_i,\; f(\bm{x}_i;\theta_{ctr}) \times f(\bm{x}_i;\theta_{cvr})\big)$$

Note what is missing: **there is no CVR loss term.** $\theta_{cvr}$ only receives [[Backpropagation|gradient]] through the product in the second term. On an un-clicked impression the label $y\&z = 0$, and the gradient still flows into $\theta_{cvr}$ (scaled by pCTR) and into the shared embeddings. That is how the CVR network "learns from un-clicked data".

**Optimiser.** Adam, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. Same across all baselines for fairness.

**Two mechanisms, two problems.** Worth keeping them separate in your head:
- The multiplication factorisation kills sample selection bias.
- The shared embedding table kills data sparsity — the CVR side inherits representations trained on 100× more samples. This is plain transfer learning, in the same family as pre-training in [[BERT- Pre-training of Deep Bidirectional Transformers|BERT]].

**Data.** Taobao recommender traffic logs. Product Dataset: 48M users, 23.5M items, 8,950M impressions, 324M clicks, 1,774k conversions. They released a 1% sample (Public Dataset, 38GB uncompressed): 0.4M users, 4.3M items, 84M impressions, 3.4M clicks, 18k conversions. First public dataset with sequential click→conversion labels. Split is temporal: first half of the time sequence trains, second half tests.

## Ablation Studies and Experiments

Two evaluation tasks, both AUC, averaged over 10 runs:
1. **CVR task** — score pCVR on clicked impressions only (the conventional evaluation).
2. **CTCVR task** — score pCTCVR on *all* impressions. This is the one that exposes sample selection bias, because it forces every model to be judged over the full space. For all methods, pCTCVR is formed as $\text{pCTR} \times \text{pCVR}$ using the *same* independently trained CTR network, so only the pCVR part differs.

**Public Dataset (AUC %):**

| Model | CVR task | CTCVR task |
|---|---|---|
| BASE (train on clicks only) | 66.00 ± 0.37 | 62.07 ± 0.45 |
| AMAN (sample un-clicked as negatives) | 65.21 ± 0.59 | 63.53 ± 0.57 |
| OVERSAMPLING (duplicate positives) | 67.18 ± 0.32 | 63.05 ± 0.48 |
| UNBIAS (rejection sampling, pCTR as reject prob.) | 66.65 ± 0.28 | 63.56 ± 0.70 |
| DIVISION (pCTCVR / pCTR, separate models) | 67.56 ± 0.48 | 63.62 ± 0.09 |
| ESMM-NS (no shared embeddings) | 68.25 ± 0.44 | 64.44 ± 0.62 |
| **ESMM** | **68.56 ± 0.37** | **65.32 ± 0.49** |

**What did not work.** AMAN — randomly sampling un-clicked impressions and calling them conversion-negatives — is the only method that goes *backwards* on the CVR task (65.21 vs 66.00 BASE). It is very sensitive to the sampling rate, and it systematically under-predicts because it labels "no click" as "no conversion", which is not the same statement. Oversampling and UNBIAS help, but only by ~1 AUC point, and UNBIAS carries the numerical instability of dividing by rejection probabilities.

**What the ablation actually reveals.** The clean ablation is the DIVISION → ESMM-NS → ESMM ladder, and it separates the two contributions:

- DIVISION vs BASE (+1.56 CVR): entire-space modelling alone is worth a lot.
- ESMM-NS vs DIVISION (+0.69 CVR, +0.82 CTCVR): switching from divide-at-inference to multiply-and-co-train. This is not cosmetic — it removes the instability *and* lets the three estimators share gradients during training.
- ESMM vs ESMM-NS (+0.31 CVR, **+0.88 CTCVR**): sharing embeddings. Notice the gain is much bigger on the full-space CTCVR task than on the clicks-only CVR task. That is exactly what you'd expect if the shared table is teaching the CVR tower about regions of feature space it never sees in clicked data.

So: the factorisation does most of the work on the CVR metric, the embedding sharing does disproportionate work on the entire-space metric.

**Product Dataset.** ESMM over BASE: **+2.18% AUC on CVR, +2.32% on CTCVR**. BASE here is not a straw man — it was the model serving Alibaba's live traffic. The authors note 0.1% AUC is considered meaningful in production, so 2% is large.

**Sampling-rate sweep.** They retrained everything at varying fractions of the 8.9B-sample product data. Every method improves as data grows, which directly confirms data sparsity is a live constraint. ESMM and ESMM-NS beat all competitors at *every* sampling rate. The one exception where BASE survives is AMAN at 1% sampling on the CVR task.

## Worth Remembering

- **The CVR head is never supervised.** It is a latent factor. If pCTR is near zero for some impression, the gradient into pCVR is near zero too — those impressions teach the CVR tower almost nothing, which is a subtle re-appearance of the same imbalance, just softened rather than eliminated.
- **No calibration guarantee on the intermediate.** ESMM is trained so the *product* matches CTCVR. Many (pCTR, pCVR) pairs give the same product. Nothing in the loss pins pCVR to its true value except the shared-parameter inductive bias and the fact that pCTR is separately supervised. In practice the CTR loss anchors one factor, which anchors the other — but it is an argument, not a proof.
- **Delayed feedback is explicitly out of scope.** Conversions can arrive days after the click, so your "negative" labels are partly just "not yet". The authors say delay is tolerable in their system and that Chapelle (2014) can be bolted on.
- **The towers are swappable.** They say plainly that the sub-networks can be replaced with DIN, Wide & Deep, or anything newer. The contribution is the factorisation and the sharing, not the MLP.
- **The dataset is the underrated part.** Ali-CCP, the released 1% sample, became a standard benchmark. It is the first public data with sequential click→conversion labels.
- **Generalises to longer chains.** The stated follow-up is $\text{request} \to \text{impression} \to \text{click} \to \text{conversion}$ — chain four probabilities instead of two. The whole family of ESM², MMoE-style multi-stage models grew from here.
- **Practical caveat.** If you copy this, resist the urge to add an auxiliary CVR loss on clicked samples "just to help it converge". That reintroduces exactly the biased-sample gradient the paper removed. Later work (ESM²) does add auxiliary tasks, but on entire-space-observable labels like add-to-cart, not on the biased CVR set.
- **Connection worth holding.** The sample selection bias here is the same object as in off-policy evaluation and [[Counterfactual Reasoning and Learning Systems|counterfactual learning]] — you observe outcomes only on the actions the system took. ESMM's answer is not importance weighting (which is what UNBIAS attempts, and which is numerically fragile); it is a structural factorisation that sidesteps the need to reweight at all.

## Links

Related: [[Deep Interest Network for CTR Prediction (DIN)]] · [[Counterfactual Reasoning and Learning Systems]] · [[Cross Entropy]] · [[Recommender Systems - Evolution]] · [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] · [[Foundational_RecSys_Ranking_Reading_Plan]] · [[Backpropagation]] · [[Momentum]] · [[Uncertainty]]

New topics worth writing: Multi-task learning shared-bottom architectures, Sample selection bias, Delayed feedback modelling in ads, Model calibration for CTR/CVR, Importance weighting and rejection sampling, Ali-CCP dataset, MMoE, ESM² and multi-stage action chains
