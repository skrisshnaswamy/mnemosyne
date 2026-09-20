---
title: "Studying Image Tokenizers as Visual Languages in Unified Multimodal Models"
authors: ["Siting Li", "Zhengyang Wang", "Simon Shaolei Du", "Xi Chen", "Yang Liu"]
year: 2026
arxiv: "2609.09143"
url: https://arxiv.org/abs/2609.09143
priority: Good-To-Read
read_on: 2026-09-17
tags: [paper, transformers, llm, self-supervised, diffusion, vision]
---
## The Core Idea

A unified multimodal model turns pictures into a sequence of discrete integers, then predicts those integers the same way it predicts words. The thing that does the turning — the **image tokenizer** — is usually judged on its own, off to one side: how well can it rebuild the picture from its integers (reconstruction FID, or rFID), or how well can you classify ImageNet from its features. This paper's claim is that those numbers measure the wrong thing, because they never let the tokenizer meet the text.

The reframe: an image tokenizer is not a preprocessing step, it is a **language**. It has a vocabulary (the codebook), a grammar (which token follows which), and it has to be learned by a language model that is simultaneously trying to keep speaking English. So judge it by how easily a transformer learns to speak it *alongside* text.

> [!NOTE] Multimodal learnability
> How well the image tokens and the text tokens are jointly modelled under one shared next-token objective — read off the per-task validation losses. Distinct from reconstruction fidelity, and the two can point in opposite directions. ^multimodal-learnability

The headline result is that reconstruction fidelity and learnability genuinely diverge. Swapping GigaTok's discriminator for a DINO-based one cuts rFID from $0.81$ to $0.51$ — a big win by the standard metric — and changes downstream generation not at all while *hurting* VQAv2 ($52.25 \to 51.31$). Meanwhile adding a semantic (CLIP-contrastive) loss to UniTok makes reconstruction *worse* (rFID $1.86 \to 2.23$) and makes everything else better: lower text loss, lower I2T loss, lower T2I loss, GenAI $0.670 \to 0.690$, VQAv2 $57.21 \to 61.28$.

The second, stranger result: **the choice of image tokenizer changes how hard plain text is to model**, even though the text data, the text tokenizer and the language backbone are all identical. That is a cross-modal interference effect that no generation-only or understanding-only evaluation could ever see.

Why this did not exist before: people either reported isolated tokenizer metrics, or trained one big unified model and reported benchmark scores. Nobody built a controlled testbed where you hold everything fixed, swap only the visual language, and watch four separate losses move.

## The Methodology

**The testbed.** Take a pretrained Qwen3 text model (0.6B, 1.7B, 4B; plus an 8B run for sanity). Bolt $B$ new learnable embeddings onto the vocabulary, where $B$ is the image codebook size, initialised from a multivariate normal matched to the mean and covariance of the existing embeddings. Extend the LM head to predict them too. Add `⟨boi⟩` / `⟨eoi⟩` markers around image-token runs. Then continually pretrain on mixed text + image data with ordinary [[Cross Entropy|cross-entropy]] next-token prediction — a pure [[Auto-regressive models|autoregressive]] setup, no diffusion head, no separate vision encoder.

Every image is $256 \times 256$ and becomes exactly $K = 16 \times 16 = 256$ tokens. Fixing $K$ and the resolution is deliberate: varying either changes the sequence length and the image-token budget, which would confound everything.

**Sequence formats** (loss is computed only on the bold part; the conditioning side is context but unscored):

- T2I: `{text}{prompt}⟨boi⟩`**`{image tokens}⟨eoi⟩⟨eos⟩`**
- I2T: `⟨boi⟩{image tokens}⟨eoi⟩{prompt}`**`{text}⟨eos⟩`**

Of image-text samples, 80% go to T2I and 20% to I2T. 10% of T2I samples have the text dropped and are marked `⟨unconditional⟩`, so [[Classifier-Free Diffusion Guidance|classifier-free guidance]] works at sampling time (they use CFG scale 7.0).

**Data.** 60M samples at the largest scale: 6.6M pure text from DataComp-LM, 42M LAION-Aesthetics (aesthetic score $\geq 5.5$, recaptioned by InternVL3-1B), 6.5M JourneyDB, 4.8M BLIP3o short captions. The text : image-text ratio is 1:8. That number was chosen not for elegance but to hold the *total number of image tokens seen* roughly equal to Liquid's setup, which used a 1024-token Chameleon tokenizer at a 1:2 ratio.

**Hyperparameters.** Warmup-Stable-Decay schedule, 0.03 warmup ratio, linear decay over the last 20% of steps. They swept lr $\in \{3\text{e-}5, 1\text{e-}4\}$ × batch size $\in \{512, 1024, 2048\}$ at 0.6B. No setting wins everywhere, so the main runs use lr $=3\text{e-}5$, bs $=512$. Pushing lr to 3e-4 (0.6B) or 1e-4 (4B) gave loss spikes.

**SFT.** 4.9M instruction samples, 2 epochs, cosine schedule, peak lr 5e-5, bs 1024.

**The measurement.** Hold out 50k text and 50k image-text samples from the training distribution. Report mean per-token negative log-likelihood over supervised positions only:

$$\mathcal{L} = -\frac{1}{T(N)}\sum_{t \in N}\sum_{i \in S(t)} \ln p(t_i \mid t_{<i})$$

where $S(t)$ is the set of scored positions. This gives four numbers: **text**, **image** (unconditional generation), **T2I**, **I2T**.

An important caveat they flag: unlike bits-per-byte in language modelling, this loss is over the tokenizer's *latent* codes, and tokenization is lossy. It is not a description length of the pixels. It only measures how well the model fits one particular visual language.

**Recipe validation.** An 8B model with the Chameleon tokenizer on 60M samples: GenAI-Bench 0.73 (Liquid-7B: 0.72), WISE 0.38 (0.41), VQA mean 60.83 (61.40), MJHQ-30K gFID 10.55 (5.47). Close enough on three of four to trust the testbed; noticeably behind on image quality.

**Tokenizers compared** (all single-codebook, so there is no ambiguity about how to flatten multiple indices into a sequence):

| Tokenizer | $B$ | semantic loss | discriminator | rFID |
|---|---|---|---|---|
| IBQ-1024 / 8192 / 16384 | 1024 / 8192 / 16384 | no | PatchGAN | 2.24 / 1.87 / 1.37 |
| GigaTok | 16384 | yes | PatchGAN | 0.81 |
| GigaTok-DINO | 16384 | yes | DINO | 0.51 |
| UniTok | 16384 | no | DINOv2-S | 1.86 |
| UniTok-sem | 16384 | yes | DINOv2-S | 2.23 |

## Ablation Studies and Experiments

**The four losses scale, but not the same way.** All four follow rough power laws against pretraining FLOPs, under both data scaling and model scaling — the familiar straight line on a log-log plot from [[Scaling Laws for Neural Language Models|Kaplan et al.]]. But:

- *Text loss barely moves with data.* It is inherited from the pretrained backbone. It does drop with model size, but that is because bigger models start from stronger backbones, not because of multimodal training.
- *T2I loss tracks unconditional image loss almost exactly.* Most of the difficulty in text-to-image is just predicting image tokens at all; the text conditioning is a small part.
- *I2T loss behaves unlike text loss*, even though both are scored over text tokens. Conditioning on image tokens is a separate signal.
- Quadratic fits to the data-scaling curves show text is slightly **concave** (coefficient $a \approx -2\text{e-}3$ for most tokenizers) while image-side losses are **convex** ($a \approx +1\text{e-}2$). Gradient-norm traces explain it: the T2I gradient dominates early and saturates; the text gradient decays more slowly and catches up.

**Tokenizer rankings flip between tasks.** UniTok has the *highest* text loss and the *lowest* T2I loss. GigaTok has the lowest text-side losses and the highest T2I loss. So an averaged "multimodal loss" is useless for this analysis — it hides exactly the trade-off you want to see.

**Loss vs benchmark, tokenizer fixed.** With GigaTok held constant and only data scale and hyperparameters varying, T2I loss correlates strongly with GenAI-Bench VQAScore and MJHQ-30K gFID. Surprisingly, I2T loss also correlates with *generation* quality, though more weakly — captioning loss is picking up general multimodal training progress.

**Loss vs benchmark, across tokenizers — this is where it breaks.** The T2I loss-performance line shifts per tokenizer, because each tokenizer defines a different prediction space. The fix: divide by the maximum entropy of the codebook,

$$\mathcal{L}^{*} = \frac{\mathcal{L}}{\log_2 B}$$

On the three IBQ variants at 0.6B, relating T2I loss to GenAI-all:

| Normalisation | Pearson $r$ | $R^2$ from loss | extra $R^2$ from tokenizer identity |
|---|---|---|---|
| none | $+0.016$ | 0.000 | 0.972 |
| $/\log_2 B$ | $-0.957$ | 0.915 | 0.048 |
| $/H_1$ (empirical unigram entropy) | $-0.953$ | 0.909 | 0.055 |

Unnormalised, the loss explains *nothing* and tokenizer identity explains everything. Normalised, it flips. And $\log_2 B$ is basically as good as measuring the real code distribution — $H_1 / \log_2 B$ is within 1.3% for all seven tokenizers, even though GigaTok's head codes are used ~40× more than its tail codes.

**Residual gap at fixed $B$.** Among the four tokenizers with $B = 16384$, normalisation does not close the gap. At fixed GenAI score, T2I loss orders *inversely* to rFID: a tokenizer that reconstructs worse hits the same generation quality at a lower T2I loss. So a low T2I loss can mean "easy language" rather than "good model".

**I2T loss is the more portable signal.** It is computed over the shared text vocabulary, so it is comparable across tokenizers with no normalisation. Measured *before* SFT, it moderately correlates with post-SFT VQAv2 and GQA, and with post-SFT generation quality.

**What did not work as a signal:** TextVQA. I2T loss is negatively correlated with TextVQA at 0.6B and *positively* correlated at 4B — the sign flips. And the hyperparameter preference inverts too: aggressive settings win on VQAv2 and GQA, mild settings win on TextVQA. OCR is a capability that caption-fit does not touch.

**What did not work as a benchmark:** POPE and MME were dropped entirely. Re-running SFT with a different data-shuffling seed (42 vs 37) moved POPE by 7.07% relative and MME-P by 4.98%, while VQAv2 moved 0.08%, GQA 0.93%, GenAI 1.42%, MJHQ 1.90%. A useful reminder in the spirit of [[On the Difficulty of Evaluating Baselines]] — if your benchmark's seed noise is larger than your effect, you do not have a result.

**The DINO discriminator ablation (Table 3, 0.6B).** This is the cleanest negative result in the paper:

| | rFID↓ | text↓ | I2T↓ | T2I↓ | GenAI↑ | VQAv2↑ |
|---|---|---|---|---|---|---|
| GigaTok-DINO | 0.51 | 2.949 | 1.660 | 7.437 | 0.720 | 51.31 |
| GigaTok | 0.81 | 2.937 | 1.661 | 7.416 | 0.720 | 52.25 |
| UniTok | 1.86 | 2.883 | 1.670 | 6.264 | 0.670 | 57.21 |
| UniTok-sem | 2.23 | 2.873 | 1.655 | 5.848 | 0.690 | 61.28 |

A 37% rFID improvement buys nothing. Meanwhile UniTok-sem, with the *worst* rFID in the table, has the best text loss, best I2T loss, best T2I loss and best VQAv2.

**The cross-modal interference ablation.** UniTok-sem gets a lower *text* loss than UniTok under joint training; IBQ-1024 gets a lower text loss than IBQ-8192. To find out why, they retrained with the image-generation objective removed — reformat all T2I samples into I2T order and train on Text+I2T only. The text-loss gap vanished for both pairs. So the effect comes from image-token *prediction* interfering with text prediction, not from captioning. Interestingly, the I2T gap *persisted*, meaning the tokenizer also changes how informative image tokens are for captioning through a separate route.

The mirror ablation (Text+T2I only, I2T removed) is the nicer finding: the text gap stayed, **and text loss got worse for both tokenizers**. Captioning helps text modelling; image generation hurts it.

**Where semantic supervision actually helps.** Two candidate stories: "better visual grammar" (image-token sequences become easier to predict) or "better visual words" (individual tokens carry more meaning). They tested both.

- *Words:* measure the I2T loss improvement of UniTok-sem over UniTok, split by word type. The gain concentrates heavily on COCO object-category words, far above stopwords, CSS colour names, or general content words. Pointwise mutual information between image tokens and object words is also higher for UniTok-sem.
- *Grammar:* empirical $n$-gram entropy of the token streams. UniTok-sem is slightly lower at 1- and 2-grams but **higher** at 3- and 7-grams. No consistent simplification.

So: better visual words, not simpler visual grammar.

**Vocabulary size is non-monotonic.** Across IBQ-1024 / 8192 / 16384: IBQ-16384 has the best I2T loss, but IBQ-8192 has the lowest normalised T2I and image losses. Text loss runs in the reverse (also non-monotonic) direction. Downstream, IBQ-16384 wins anyway — plausibly because its rFID is best (1.37). So learnability and fidelity pull in different directions and fidelity wins here, which is the opposite of the GigaTok-DINO result. They do not resolve this.

## Worth Remembering

- The single most transferable idea for your work is the measurement discipline, not the vision content: **when you change the representation, the loss you were tracking stops being comparable, and you need an explicit normalisation before you can read it as a signal.** Dividing by $\log_2 B$ took $R^2$ from 0.000 to 0.915. The same problem shows up anywhere you compare models over different label spaces — different item catalogues, different candidate-set sizes, different [[Recommender Systems with Generative Retrieval (TIGER)|semantic ID]] schemes.
- The TIGER connection is direct. TIGER also invents a discrete "language" for items via RQ-VAE and feeds the IDs to a sequence model. This paper's warning applies: a codebook that reconstructs item embeddings best is not necessarily the codebook a transformer learns most easily, and neither is necessarily the one that ranks best.
- **Reconstruction metrics are a proxy that has drifted from its target.** rFID is the vision-tokenizer equivalent of tuning a retriever on recall@1000 and never checking the ranker. Same failure pattern documented in [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|the RecSys replication paper]] and [[Do ImageNet Classifiers Generalize to ImageNet]].
- The quantizer itself is a vector-quantisation codebook, the same machinery as in [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)|product quantisation]] — nearest-entry lookup in a learned codebook — trained with a straight-through estimator, LPIPS perceptual loss, and a [[Generative Adversarial Networks|GAN]] discriminator.
- **Practical caveat on codebook collapse.** The $\log_2 B$ normalisation assumes the codebook is roughly uniformly used. All seven tokenizers here have ~100% utilisation. Under real [[Mode Collapse|collapse]], the nominal $B$ stops describing the effective prediction space and the normalisation fails. The authors say so explicitly.
- Annealing (the decay phase of the WSD schedule) mainly drops text loss and leaves image losses nearly unchanged. Their hypothesis: image-token gradients are noisier or smaller in the decay stage.
- Limitations the authors own: seven tokenizers only, all single-codebook, all fixed at $K = 256$ tokens and $256 \times 256$ resolution. They note it is *mathematically impossible* to hold both "images seen" and "image tokens seen" constant while varying $K$, so that axis is confounded by construction. Everything is on Qwen3 backbones; they do not claim the rankings survive a different model family. No from-scratch scaling-law study, so no compute-optimal frontier in the [[Training Compute-Optimal Large Language Models (Chinchilla)|Chinchilla]] sense.
- Open question worth chasing: they never explain *why* image-token prediction raises text loss. Is it embedding-table crowding, head competition, or gradient conflict? The gradient-norm traces hint at the third but do not test it.

## Links
Related: [[Recommender Systems with Generative Retrieval (TIGER)]] · [[Scaling Laws for Neural Language Models]] · [[Cross Entropy]] · [[Auto-regressive models]] · [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)]] · [[Classifier-Free Diffusion Guidance]] · [[Generative Adversarial Networks]] · [[Mode Collapse]] · [[On the Difficulty of Evaluating Baselines]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Do ImageNet Classifiers Generalize to ImageNet]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[An Image is Worth 16x16 Words (ViT)]]

New topics worth writing: VQGAN and vector-quantised autoencoders, reconstruction FID (rFID), LPIPS perceptual loss, Warmup-Stable-Decay learning-rate schedules, pointwise mutual information as a diagnostic, VQAScore, Chameleon early-fusion multimodal models
