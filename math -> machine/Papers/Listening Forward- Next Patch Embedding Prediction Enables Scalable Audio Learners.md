---
title: "Listening Forward: Next Patch Embedding Prediction Enables Scalable Audio Learners"
authors: ["Umberto Cappellazzo", "Xubo Liu", "Stavros Petridis", "Maja Pantic"]
year: 2026
arxiv: "2608.19863"
url: https://arxiv.org/abs/2608.19863
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, transformers, llm, self-supervised, vision, theory]
---
## The Core Idea

Audio self-supervised learning had drifted into complicated recipes. To pre-train an audio encoder without labels you were expected to bolt on a reconstruction decoder, or a separately trained acoustic tokenizer, or an EMA teacher copy of your own network, or a variance-covariance regulariser to stop everything collapsing. NAPE throws all of that out and asks: what if audio just does what language does — predict the next thing?

The observation that makes this reasonable is simple. Images have no natural reading order; a 2D grid of pixels is roughly the same in every direction. Audio does have one. A spectrogram unfolds along time. So chopping a spectrogram into patches, laying them out as a sequence, and training a causal Transformer to guess the next patch is not an arbitrary imposition — it matches how sound actually happens.

The one twist: you do not predict a discrete token, because audio has no vocabulary. You predict a **continuous embedding** — the vector that the patch-embedding layer itself produces for the next patch. The target comes from your own shallow patchifier, frozen on the target side by a stop-gradient.

> [!NOTE] Next-patch-embedding prediction
> At position $t$, the model outputs $\hat z_{t+1}$ and is scored on how close its direction is to the real patch embedding $z_{t+1}$. No decoder, no tokenizer, no teacher network. ^next-patch-embedding

The whole learning signal is three cheap ingredients: a causal mask, a shift by one position, and a stop-gradient. Pull out any one and the thing either diverges or learns nothing. That is the paper's real contribution — showing that this minimal triple is *sufficient* for audio, where everyone else assumed you needed scaffolding.

What it unlocks: an audio encoder with a pre-training recipe you can write in twelve lines of PyTorch, that ties the strongest published baseline (SSLAM) on AudioSet-2M at 50.2 mAP and beats every baseline on speech emotion recognition by 3.5–4.3 points.

## The Methodology

**Input.** Raw waveform → mono 16 kHz → log-mel spectrogram with 128 mel bands, 25 ms Hanning window, 10 ms hop. A 10-second clip becomes a tensor of shape $1 \times 128 \times 1008$ (channel × frequency × time). Cut into non-overlapping $16\times16$ patches, that is $8 \times 63 = 504$ patches per clip.

**Patchifier $f$.** A single strided Conv2d, kernel and stride $16\times16$, mapping each patch to a $d$-dimensional vector. Same as standard [[An Image is Worth 16x16 Words (ViT)|ViT]]. This module does double duty — it produces both the model's *inputs* and its *targets*.

**Scanning order.** Here is the design axis unique to this setting. A Transformer eats a 1D sequence, so the 2D patch grid must be flattened. Under normal bidirectional attention this choice is irrelevant — self-attention is permutation-equivariant once you add position embeddings, so any consistent order gives identical representations. But [[Causal Attention|causal masking]] breaks that. The order now decides which patches count as "past" and which as "future". Four options tested:

- **Raster** — left to right, then up a frequency row. Sweeps all of time before advancing in frequency.
- **Time-major** — bottom to top within one time column, then advance in time. Sweeps all of frequency first.
- **Zigzag** — raster but rows alternate direction, so consecutive patches stay spatially adjacent at row boundaries.
- **Diagonal** — sweep along anti-diagonals, mixing time and frequency progress at every step.

**Encoder $h$.** A ViT with pre-norm blocks and a causal attention mask. Three stability details borrowed from modern LLM practice: [[RoFormer- Enhanced Transformer with Rotary Position Embedding|RoPE]] applied *independently along the frequency and time axes*, LayerScale on residual branches, and parameter-free query–key normalisation on the per-head Q and K projections. Three sizes:

| | $d$ | layers | heads | params |
|---|---|---|---|---|
| Small | 384 | 12 | 6 | 19M |
| Base | 768 | 12 | 12 | 85M |
| Large | 1024 | 24 | 16 | 303M |

**Predictor $g$.** A SimSiam-style 3-layer MLP with intermediate LayerNorms: $\text{Linear}(d,d) \to \text{LN} \to \text{GELU} \to \text{Linear}(d,d) \to \text{LN} \to \text{GELU} \to \text{Linear}(d,d)$. 1.8M parameters. Its job is to *decouple* the space the encoder lives in from the space predictions are made in — if you predict straight from the encoder output, the encoder is forced to sit in the same space as its targets, which limits how rich its features can get.

**Objective.** Negative cosine similarity, averaged over all $N-1$ valid positions:

$$\mathcal{L} = \frac{1}{N-1}\sum_{t=1}^{N-1} \mathcal{D}\big(\texttt{stopgrad}(z_{t+1}),\, \hat z_{t+1}\big), \qquad \hat z_{t+1} = g(h(z_{\le t}))$$

$$\mathcal{D}(z, \hat z) = -\frac{z}{\lVert z \rVert_2} \cdot \frac{\hat z}{\lVert \hat z \rVert_2}$$

Cosine is magnitude-invariant, which is load-bearing: a distance loss can be trivially driven to zero by shrinking both vectors' norms. Cosine cannot be cheated that way.

**The three anti-collapse mechanisms.** Each blocks a different shortcut.

1. **Causality** — the mask stops position $t$ from ever seeing patch $t+1$, so the encoder cannot just copy the answer.
2. **Prediction shift** — position $t$ predicts $t+1$, not itself. Without the shift, causality alone does not help: the model would just pass its own input through.
3. **Stop-gradient** — gradients flow only through the prediction branch. Otherwise the encoder happily maps *everything* to one shared constant vector, which perfectly satisfies the loss. Same logic as SimSiam, and worth comparing to [[Understanding Dimensional Collapse in Contrastive Learning|dimensional collapse]].

**Pre-training.** AudioSet unbalanced + balanced, unlabelled: 1,964,222 + 20,961 clips. [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], base LR $5\times10^{-3}$ with cosine decay, weight decay 0.05, $\beta_1=0.9$, $\beta_2=0.95$, 10% warmup. Batch 256 (128 for Large). 25 epochs (S, L), 30 (B). NVIDIA L40s, 8 GPUs, plain DDP. No exotic parallelism.

**Fine-tuning.** Load the encoder, **turn the causal mask off** so attention is bidirectional, mean-pool the patch tokens, attach a linear head. AdamW with $\beta_2 = 0.999$, layer-wise LR decay 0.7–0.9, LR $1.25\times10^{-3}$. Full augmentation stack (SpecAugment, Mixup, CutMix, DropPath, temporal roll, noise, label smoothing), plus a weight EMA for the two AudioSet tasks. BCE loss for multi-label, soft-target [[Cross Entropy|cross-entropy]] for single-label.

## Ablation Studies and Experiments

Benchmarks: AudioSet-2M and AudioSet-20K (mAP), ESC-50 (50-class environmental sound, 5-fold), Speech Commands V1/V2 (KS1/KS2, keyword spotting), IEMOCAP (4-class emotion, 5-fold). All ablations on NAPE-Base with raster order unless noted.

**The three core components (Table 1).** The most informative table in the paper.

| shift | stopgrad | causal | AS-2M | AS-20K | ESC-50 | KS1 | KS2 | ER |
|---|---|---|---|---|---|---|---|---|
| ✘ | ✓ | ✓ | \multicolumn{6}{c}{**diverges**} |
| ✓ | ✘ | ✓ | \multicolumn{6}{c}{**diverges**} |
| ✓ | ✓ | ✘ | 41.8 | 24.8 | 68.9 | 96.1 | 97.3 | 57.0 |
| ✓ | ✓ | ✓ | **49.6** | **39.1** | **94.2** | 97.9 | 98.8 | 64.9 |

Dropping the shift or the stop-gradient blows up training outright. Dropping causality is more insidious — training looks *fine*, the loss saturates near $-1$ within a few thousand steps, and the representations are garbage. ESC-50 falls 25.4 points. This is the paper's cleanest warning: a smoothly descending SSL loss tells you almost nothing.

**Similarity function (Table 5).** $\ell_1$ and $\ell_2$ both diverge — the loss curve dips to ~0.05 within a few hundred steps as norms shrink toward zero, then climbs back as the problem becomes ill-conditioned. Softmax cross-entropy over the $d$ channels trains stably (softmax bounds the magnitude implicitly) but lands at 48.8 / 37.4 versus cosine's 49.6 / 39.1, and overfits after ~70k steps.

**Prediction target (Table 4).** Predicting the output of the *first encoder layer*, JEPA-style, diverges — the target now depends on the network being trained, and stop-gradient alone is not enough. This is exactly why [[Self-Supervised Learning from Images with I-JEPA|I-JEPA]] needs an EMA teacher and [[LeJEPA- Provable and Scalable Self-Supervised Learning|LeJEPA]] needs SIGReg. Raw mel pixels work fine (49.7 / 38.0 / 94.8) and are essentially tied with patch embeddings (49.6 / 39.1 / 94.2). The shallow, non-recurrent patchifier is what makes the target stable enough to skip the teacher.

**Patch embedding layer (Table 2).** Plain Conv2d beats both alternatives. A deep 4-layer convstem with BatchNorm (the "early convolutions help ViTs" trick): 46.7 / 34.3 / 89.1, losing 4.8 mAP on AS-20K. A Gemma-style speechstem that folds mel into channels and does temporal-only convolutions: 47.6 / 33.1 / 88.4, losing 6.0. Treating time and frequency as symmetric 2D axes *at patchification* matters; the temporal bias should come from the scan order, not the stem.

**Predictor (Table 3).** No predictor at all: 48.7 / 37.8. 2-layer MLP (1.2M): 49.4 / 38.5. 2-layer causal Transformer (14.2M): 49.2 / 38.4. SimSiam MLP (1.8M): 49.6 / 39.1. The 1.8M MLP beats the 14.2M Transformer. Predictor capacity is not the bottleneck.

**Scan order (Figure 3).** Raster, diagonal, and zigzag are all close, with raster and diagonal slightly ahead. **Time-major loses on every one of the six tasks.** Reason: it exhausts a whole frequency column before moving forward in time, so early predictions have a rich instantaneous spectrum but almost no temporal history — backwards for audio.

**Random masking (Table 13, Small model).** Adding masked-modelling-style input masking *hurts*: 0% → 47.6 / 36.2 / 92.9, 20% → 47.3 / 36.2 / 92.2, 50% → 47.2 / 35.2 / 91.7. The causal mask already hides the target, so extra masking just deletes useful context without making the task any harder.

**RoPE vs absolute positions.** +1.8 mAP on AS-2M, +2.4 on AS-20K. One of the larger single-knob effects, and it also lets you feed clips of different lengths without interpolating position tables.

**Non-effects.** Freezing the patch-embedding layer during fine-tuning: no difference (49.61 vs 49.58 on AS-2M) — contradicting the vision paper NEPA, where it helped a lot. LayerNorm vs RMSNorm: a wash. Bidirectional CLS / last-token / mean pooling: all within 0.2 mAP; keeping causal attention with last-token pooling at fine-tune time costs only 0.2.

**Headline comparison (Table 6).**

| Model | Params | AS-2M | AS-20K | ESC-50 | KS1 | KS2 | ER |
|---|---|---|---|---|---|---|---|
| Audio-MAE | 86M | 47.3 | 37.1 | 94.1 | 96.9 | 98.3 | — |
| BEATs iter3 | 90M | 48.0 | 38.3 | 95.6 | 97.7 | 98.3 | 64.5 |
| A-JEPA | 86M | 48.6 | 38.4 | 96.3 | 97.7 | 98.5 | — |
| EAT | 88M | 48.6 | 40.2 | 95.9 | — | 98.3 | — |
| SSLAM | 88M | **50.2** | **40.9** | 96.2 | **98.8** | 98.1 | — |
| SPEAR Large | 327M | 49.7 | 39.3 | — | — | — | — |
| **NAPE-B diagonal** | 85M | 49.7 | 39.2 | 94.8 | 97.9 | 98.6 | 67.1 |
| **NAPE-L raster** | 303M | **50.2** | 40.5 | 96.0 | 97.9 | 98.8 | 68.0 |
| **NAPE-L diagonal** | 303M | 50.0 | 40.4 | 96.2 | 98.2 | **98.9** | **68.8** |

NAPE-L ties SSLAM on AS-2M and comes within 0.4 mAP on AS-20K — while SSLAM needs audio-mixture supervision, a student–teacher setup, *and* a reconstruction decoder. The standout is IEMOCAP: 68.8% vs BEATs' 64.5%, a +4.3 gap on a speech task none of these audio-event models were built for.

**Scaling.** Monotone improvement S → B → L on all six benchmarks, though gains taper from Base to Large. Against Audio-MAE at matched scale, NAPE wins everywhere, with the biggest margin at Small (+2.6 AS-2M, +4.1 AS-20K). Budget ablation: NAPE-Large after only **6 epochs** already hits 49.45 mAP on AS-2M, beating most published baselines. 30 epochs gets to 49.89; AS-20K goes 37.15 → 39.94.

**Linear probing.** Freeze the encoder, train a LayerNorm plus linear head, no augmentation. The best layer is *mid-network* at every scale: layer 2 of 12 (Small), layer 6 of 12 (Base), layer 11 of 24 (Large). Beyond that, probe accuracy falls 3–5 mAP toward the output. Best-layer numbers: AS-2M 23.2 / 25.0 / 27.1, AS-20K 18.9 / 19.7 / 20.4, ESC-50 79.8 / 81.7 / 83.5 for S / B / L. Scaling holds under freezing, but absolute numbers are far below fine-tuning — predicting the next embedding is simply not the same objective as being linearly separable.

**Qualitative.** On held-out clips, cosine similarity between $\hat z_{t+1}$ and the true $z_{t+1}$ is near 1.0 almost everywhere. The exceptions have obvious causes: the very first patch (no context), the first mel row (little context plus low-frequency content), and the rightmost columns (zero-padded frames). The averaged attention map for a query patch shows two clean structures with no supervision: strong attention to the **current time column** (the full spectral snapshot of now) and to **same-frequency patches earlier in the clip** (how this frequency band has been evolving).

## Worth Remembering

- **The causal-mask ablation is the lesson to carry away.** Removing it does not destabilise anything. The loss goes to $-1$ faster and stays there, and the encoder learns nothing transferable. If you build a predictive SSL objective, a beautiful loss curve is evidence of a shortcut at least as often as it is evidence of learning. Cross-reference [[Shortcut Learning in Deep Neural Networks|shortcut learning]].
- **Why NAPE can skip the EMA teacher:** its targets come from a shallow non-recurrent conv layer, not from a deep encoder. As soon as the paper tries a deeper target (first encoder layer), training collapses and you are right back in JEPA territory needing an EMA copy or SIGReg. So "minimalist" is conditional on a *shallow* target.
- **Causal at pre-train, bidirectional at fine-tune.** The mask is a training-time device, dropped entirely for downstream use. It shapes what the encoder is forced to learn; it is not part of the deployed model.
- The mid-layer linear probing peak echoes findings in language models — top layers specialise for the pre-training objective, middle layers hold the general-purpose features. If you plan to use NAPE frozen, probe every layer; the output layer is the wrong default.
- **Limitations the authors are honest about.** Linear probing lags fine-tuning by a wide margin. Gains taper Base → Large. Keyword spotting is saturated (98.8–98.9%), so those columns carry little signal. Baselines are quoted from original papers, not re-run — cf. [[On the Difficulty of Evaluating Baselines]].
- Practical caveats: the diagonal scan is a bit better than raster on most tasks but raster wins AS-2M at Large; the difference is small enough that it may be noise. RoPE on both axes is worth ~2 mAP for free. Do not add masking. Do not use $\ell_2$.
- Open question: nothing here uses NAPE *generatively*. It is a causal model over continuous embeddings — you could in principle roll it forward to synthesise or forecast spectrograms, or use it as a world model for audio. Not attempted.
- All of this was trained on 8× L40s (46GB) with plain DDP, batch 256. That is genuinely reproducible in an academic lab, which is part of the argument.

## Links

Related: [[Attention Is All You Need]] · [[Causal Attention]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Self-Supervised Learning from Images with I-JEPA]] · [[LeJEPA- Provable and Scalable Self-Supervised Learning]] · [[RoFormer- Enhanced Transformer with Rotary Position Embedding]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[A Simple Framework for Contrastive Learning (SimCLR)]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[Auto-regressive models]] · [[Layer Normalization]] · [[Cross Entropy]] · [[Mode Collapse]] · [[Shortcut Learning in Deep Neural Networks]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]]

New topics worth writing: SimSiam and stop-gradient collapse prevention, Masked Autoencoders (MAE), Audio-MAE, BEATs and acoustic tokenizers, log-mel spectrograms, LayerScale, query-key normalisation, layer-wise learning rate decay, SpecAugment, AudioSet benchmark, mean average precision for multi-label audio tagging, RMSNorm, convolutional stems for ViTs
```
