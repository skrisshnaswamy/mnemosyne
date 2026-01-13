---
title: "Momentum Contrast (MoCo)"
authors: ["Kaiming He", "Haoqi Fan", "Yuxin Wu", "Saining Xie", "Ross Girshick"]
year: 2019
arxiv: "1911.05722"
url: https://arxiv.org/abs/1911.05722
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, self-supervised, vision]
---
## The Core Idea

Contrastive learning is a dictionary lookup. You have a **query** vector $q$ (an image, encoded). You have a pile of **key** vectors. One key is the "right answer" — it comes from the same image as the query, just cropped and colour-jittered differently. All the other keys are wrong answers. Train the encoder so the query lands close to the right key and far from everything else.

Two things make this dictionary good, and they fight each other:

1. **Big.** More wrong answers means a harder, more informative task. The negatives sample the space of images more densely.
2. **Consistent.** All the keys should be encoded by roughly *the same* network. If key #1 was made by a network from 50 steps ago and key #2 by the current network, comparing their distances to $q$ is comparing apples to oranges.

Before MoCo you had to pick one. The **end-to-end** approach uses the current mini-batch as the dictionary — perfectly consistent, but the dictionary size *is* the batch size, capped by GPU memory at ~1024, and large-batch optimisation is its own nightmare. The **memory bank** approach (Wu et al., InstDisc) stores a vector for every image in the dataset — huge dictionary, but each stored vector was written the last time that image was seen, so the bank contains encoders spread across a whole epoch. Wildly inconsistent.

MoCo gets both with two cheap tricks.

**Trick 1 — the dictionary is a queue.** Keep the last $K$ encoded key vectors in a FIFO buffer. Each step, push the current batch's keys in, drop the oldest batch out. The dictionary size $K$ is now a free hyperparameter, completely decoupled from batch size. They use $K = 65536$ with a batch of 256. Storing 65536 vectors of 128 floats is nothing — a few tens of MB.

**Trick 2 — the key encoder moves slowly.** You cannot backprop into a queue (the gradient would have to reach 65536 old samples). Naively copying the query encoder's weights into the key encoder every step *fails outright* — the loss oscillates and never converges. Instead, the key encoder is an exponential moving average of the query encoder:

$$\theta_k \leftarrow m\,\theta_k + (1-m)\,\theta_q, \qquad m = 0.999$$

Only $\theta_q$ gets gradients. With $m = 0.999$ the key encoder barely budges per step, so the keys sitting in the queue — made over the last ~256 steps — were all produced by nearly identical networks. Consistent *and* large.

> [!NOTE] Momentum encoder ^momentum-encoder
> A second copy of the network whose weights are an exponential moving average of the trained network's weights. It receives no gradient. It exists so that outputs produced at different times stay comparable to each other.

What this unlocks: for the first time, unsupervised pre-training **beat** supervised ImageNet pre-training on downstream detection and segmentation — 7 tasks, sometimes by large margins. And because MoCo uses a plain ResNet-50 with no architectural surgery (no patchifying the input, no custom receptive fields, no two-network combos), those weights drop straight into a Faster R-CNN or Mask R-CNN.

## The Methodology

**The loss** is [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)#^infonce|InfoNCE]]:

$$\mathcal{L}_q = -\log \frac{\exp(q \cdot k_+ / \tau)}{\sum_{i=0}^{K}\exp(q\cdot k_i/\tau)}$$

with $\tau = 0.07$. This is literally a $(K{+}1)$-way [[Cross Entropy|cross-entropy]] where the correct class is index 0. In code you concatenate one positive logit with $K$ negative logits and call `CrossEntropyLoss(logits/t, zeros(N))`.

**The pretext task** is instance discrimination. Take image $x$, apply random augmentation twice to get $x^q$ and $x^k$. These are a positive pair. Everything in the queue is negative. Augmentation: 224×224 random resized crop, colour jitter, horizontal flip, random grayscale. (Note how thin this is compared to [[A Simple Framework for Contrastive Learning (SimCLR)|SimCLR]]'s recipe — that gap is most of why MoCo v2 later jumped to 71.1%.)

**The encoders.** Standard [[Deep Residual Learning for Image Recognition (ResNet)|ResNet-50]]. The final FC layer after global average pooling outputs 128-D, which is then L2-normalised. Because everything is unit-norm, the dot product is cosine similarity.

**The training loop**, per batch:

1. `q = f_q(aug(x))`, `k = f_k(aug(x))`, then `k.detach()` — no gradient to keys.
2. Positive logits: batched dot product $q \cdot k$, shape $N{\times}1$.
3. Negative logits: $q$ times the queue matrix, shape $N{\times}K$.
4. Cross-entropy, backprop, SGD update on $f_q$ only.
5. Momentum update $f_k$.
6. Enqueue `k`, dequeue the oldest batch.

**Shuffling BN — the detail without which nothing works.** With ordinary [[Batch Normalization|BatchNorm]], the model cheats. The per-GPU batch statistics act as a fingerprint: the network can figure out which sub-batch a positive key came from by reading the normalisation statistics, and solve the task without learning anything about images. The symptom is unmistakable — pretext-task training accuracy shoots past 99.9% while the kNN validation accuracy on ImageNet *drops*.

The fix: before distributing the mini-batch across the 8 GPUs for the **key** encoder, shuffle the sample order (and unshuffle after encoding). The query batch is left alone. Now a query and its positive key are computed with statistics from two different subsets, and the fingerprint is destroyed. This is applied to MoCo and to the end-to-end baseline; the memory bank doesn't need it because its positives come from past batches anyway.

**Training config.** SGD, weight decay 0.0001, momentum 0.9. ImageNet-1M: batch 256 across 8 GPUs, LR 0.03, 200 epochs, ÷10 at epochs 120 and 160 — about 53 hours for a ResNet-50. Instagram-1B (940M uncurated public images, ~1500 hashtags): batch 1024 across 64 GPUs, LR 0.12 decayed ×0.9 every 62.5k iterations, 1.25M iterations (~1.4 epochs), ~6 days.

**Linear evaluation protocol.** Freeze the backbone, train one FC layer + softmax on the pooled features for 100 epochs. Grid search found the optimal LR is **30** with weight decay **0** — an absurd-looking number that tells you the feature magnitudes are on a completely different scale from supervised features. This matters later for fine-tuning.

## Ablation Studies and Experiments

**The three mechanisms, same pretext task, same loss, ResNet-50, ImageNet linear probe (Figure 3).** All three improve as $K$ grows — that alone validates "bigger dictionary is better." But:

- End-to-end matches MoCo at small $K$, then hits a wall at $K = 1024$ (the largest batch 8× Volta 32GB can hold). It also needed the linear LR scaling rule; without it accuracy fell ~2% at batch 1024.
- Memory bank tops out **2.6% below** MoCo at the same $K$ (58.0% vs 60.6%). Same size, worse consistency. This is the cleanest evidence that consistency, not just size, is doing real work.

**Momentum $m$** ($K = 4096$):

| $m$ | 0 | 0.9 | 0.99 | 0.999 | 0.9999 |
|---|---|---|---|---|---|
| acc (%) | **fail** | 55.2 | 57.8 | 59.0 | 58.9 |

$m = 0$ (just copy the weights each step) does not converge at all — the loss oscillates. That is the headline negative result. The useful range is wide, 0.99–0.9999, so it is not a fragile knob, but it must be *large*.

**ImageNet linear probe vs prior work (Table 1).** MoCo R50 (24M params): **60.6%**, best in its parameter class — InstDisc 54.0, LocalAgg 58.8, CPC v1 48.7, BigBiGAN R50 56.6. Scaling the backbone: RX50 63.9, R50w2× 65.4, R50w4× **68.6%**. Competitors like CMC (68.4) and AMDIM (68.1) get there only with far more parameters *and*, per the paper's footnote, with FastAutoAugment that was itself supervised by ImageNet labels.

**Transfer — the part that actually mattered.** All fine-tuning uses the *same* hyperparameters and schedule as the supervised baseline, which disadvantages MoCo, plus BN is tuned and synced rather than frozen (needed because MoCo's feature magnitudes differ so much).

VOC detection, `trainval07+12`, Faster R-CNN R50-C4: supervised 53.5 AP → MoCo IN-1M 55.9 (+2.4) → MoCo IG-1B **57.2** (+3.7). AP$_{75}$ +4.9. On the R50-dilated-C5 backbone the gains shrink to +0.6 / +1.5 — **transfer advantage depends on the detector architecture**, which the authors flag as a previously hidden confound in this literature.

VOC `trainval2007` (only 5k images, Table 4): every prior self-supervised method loses to supervised on AP$_{50}$ — RelPos −7.4, Jigsaw −9.1, LocalAgg −5.5. MoCo is +0.5, and +5.2 AP / +9.0 AP$_{75}$ with IG-1B.

The mechanism ablation carried into detection (Table 3, VOC, C4): end-to-end 54.6 AP, memory bank 54.9, MoCo **55.9**. The ranking holds downstream, not just on the linear probe.

COCO Mask R-CNN: at the 1× schedule everything is undertrained and MoCo IN-1M is slightly *behind* supervised on FPN (38.5 vs 38.9 box AP). At 2× MoCo wins everywhere. At 6× (Appendix A.8) the gap **widens**: supervised 41.9, from-scratch 41.4, MoCo IG-1B 42.8 (+0.9 vs +0.5 at 2×). Longer fine-tuning favours MoCo.

Also winning: COCO keypoints (+1.0 AP, where supervised pre-training beats random init by *nothing*), COCO DensePose (+1.8 AP, +3.7 AP$_{75}$), LVIS with IG-1B (+0.5), Cityscapes semantic seg (+0.7).

**What did not work:**

- **No shuffling BN → collapse.** Pretext accuracy >99.9%, kNN validation accuracy falls. Pure [[Shortcut Learning in Deep Neural Networks#^shortcut|shortcut]] learning off batch statistics.
- **$m = 0$ → divergence.**
- **VOC semantic segmentation is a genuine loss**: MoCo IN-1M 72.5 mIoU vs supervised 74.4 (−1.9). IG-1B only gets to 73.6. The authors report it plainly.
- **iNaturalist fine-grained classification is a wash**: random init 61.8, supervised 66.1, MoCo IN-1M 65.6, MoCo IG-1B 65.8. Comparable, not better.
- **Cityscapes instance seg**: on par, not ahead.
- **LVIS quirk**: for the supervised baseline, *freezing* BN gives 24.4 AP$^{mk}$ while tuning it gives 23.2 and overfits — opposite of COCO/VOC behaviour.
- **1B images bought surprisingly little.** IG-1B is 730× more data than IN-1M and buys roughly 1–2 AP points. The authors admit the large-scale data "may not be fully exploited" and blame the simple instance-discrimination pretext task.

## Worth Remembering

- The queue is a *free* dictionary. The keys are computed anyway for the current batch; you are simply not throwing them away. The extra cost is a small buffer of 128-D vectors. This is the cheapest good idea in self-supervised vision.
- Discarding the **oldest** batch is not arbitrary — those keys came from the most out-of-date encoder and are the least consistent with the current query encoder.
- The momentum-encoder idea generalised far beyond MoCo. BYOL, DINO, and the EMA target encoder in [[Self-Supervised Learning from Images with I-JEPA|I-JEPA]] all use the same construction. Note the different role from [[Momentum|momentum in optimisation]] — here it smooths *weights over time to keep outputs comparable*, not to accelerate descent.
- MoCo v2 (a four-page follow-up) took R50 from 60.6% → 71.1% by borrowing SimCLR's MLP projection head and stronger blur augmentation, keeping the queue. Interpret this carefully: **the queue mechanism and the augmentation/head recipe are orthogonal axes**, and in 2019 MoCo had a weak recipe on the second axis. The 60.6 number understates the mechanism.
- Linear-probe accuracy and transfer accuracy do not track each other. CPC v2 hits 65.9% linear with 303M params; MoCo's 60.6% with 24M transfers better. Do not rank pre-training methods by linear probe alone.
- The LR-30 / weight-decay-0 linear probe setting is a warning sign in disguise: self-supervised features live at a different scale, so any downstream system whose hyperparameters were tuned for supervised features needs re-normalising (hence the tuned, synced BN during fine-tuning).
- Practical caveat: shuffling BN is a *multi-GPU* trick. On a single GPU you have no sub-batches to shuffle. Modern reimplementations avoid the whole problem by using [[Layer Normalization|LayerNorm]]-style normalisation or SyncBN carefully — but if you write MoCo from scratch on one device and it collapses, this is why.
- Open question the authors raise: swap instance discrimination for a masked-autoencoding pretext task, as in [[BERT- Pre-training of Deep Bidirectional Transformers|BERT]]. Kaiming He answered his own question three years later with MAE.

## Links

Related: [[A Simple Framework for Contrastive Learning (SimCLR)]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[Self-Supervised Learning from Images with I-JEPA]] · [[Understanding Contrastive Learning through Alignment and Uniformity]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Batch Normalization]] · [[Cross Entropy]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[Momentum]] · [[Shortcut Learning in Deep Neural Networks]] · [[LeJEPA- Provable and Scalable Self-Supervised Learning]] · [[SimCSE- Simple Contrastive Learning of Sentence Embeddings]] · [[The Bitter Lesson (essay)]]

New topics worth writing: Instance discrimination / InstDisc memory bank, Noise-Contrastive Estimation (NCE), Faster R-CNN and Mask R-CNN, Feature Pyramid Networks, MoCo v2 and BYOL, Masked Autoencoders (MAE), Exponential moving average of weights (EMA / Polyak averaging), Linear probing as an evaluation protocol, Linear learning-rate scaling rule for large-batch SGD
