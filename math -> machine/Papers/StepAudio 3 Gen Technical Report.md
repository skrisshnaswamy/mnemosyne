---
title: "StepAudio 3 Gen Technical Report"
authors: ["Bin Lin", "Bo Zhao", "Boyang Wang", "Boyang Zhang", "Boyong Wu", "Chao Yan", "Chen Geng", "Chen Wu", "Cheng Yi", "Chengli Feng", "Chenglin Zhu", "DanNi Wan", "Daxin Jiang", "Dongqing Pang", "Fei Tian", "Feng Tian", "Future Li", "Gang Yu", "Guanglong Yang", "Jia Peng"]
year: 2026
arxiv: "2609.12945"
url: https://arxiv.org/abs/2609.12945
priority: Good-To-Read
read_on: 2026-09-17
tags: [paper, transformers, llm, self-supervised, diffusion]
---
## The Core Idea

One model, one token space, for every kind of sound: speech, singing, music, sound effects, and mixtures of all four — and it makes them the same way a language model makes words. No diffusion decoder, no flow-matching renderer anywhere in the pipeline. Audio is just more tokens in the vocabulary, and generation is next-token prediction.

The obstacle that made this hard before is compression. To get 24 kHz audio that sounds good, you need a rich discrete code. Residual vector quantization gives you that: each 80 ms frame is described by 16 integers, each from a codebook of 2048 entries, where each codebook corrects the error left by the one before it.

> [!NOTE] Residual vector quantization (RVQ)
> Quantize a vector to the nearest entry in codebook 1. Take the leftover error, quantize *that* with codebook 2. Repeat. After 16 rounds you have 16 IDs that together reconstruct the vector far better than any single ID could. Same family of trick as [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)|product quantization]] and the RQ-VAE behind [[Recommender Systems with Generative Retrieval (TIGER)|semantic IDs]]. ^rvq

Now the bind. If you lay all 16 codes out along time, one second of audio becomes $16 \times 12.5 = 200$ tokens and the sequence explodes. If you keep only the first code, you throw away the acoustic detail and the voice sounds thin. StepAudio 3 Gen resolves this by splitting the job across two axes. The big pretrained language model predicts only **codebook 0** along the **time** axis — that is the semantic, prosodic, long-range planning job it is already good at. A small 4-layer causal Transformer then predicts codebooks 1–15 along the **depth** axis, inside a single frame. Time is the LLM's problem; acoustic residue is the little model's problem.

The second idea is the one worth stealing even if you never touch audio: **bolting a new modality onto a pretrained LLM damages it, and the damage comes through two specific channels, which you can block separately.**

- *Channel one, representation.* You sum 16 freshly initialised embeddings to make one audio frame vector. That sum has no reason to look statistically like the pretrained text embeddings the backbone expects. Fix: pass it through a **RVQ Adaptor**, a small residual block whose output projection is initialised to zero, so at step 0 it contributes exactly nothing and the model is unchanged.
- *Channel two, optimization.* The residual-codebook loss sums over 15 predictions for every one temporal decision, so it is an order of magnitude bigger than the token loss. A randomly initialised predictor sending that gradient into the backbone will scramble representations that took two stages to build. Fix: **detach** the predictor's conditioning input from the backbone until the predictor has converged, then reconnect it at a low learning rate with the loss weight dropped from $\lambda = 1.0$ to $\lambda = 0.1$.

What it unlocks in practice: audio understanding, audio generation, and text reasoning live in one causal stream, so a single instruction can say "these two characters, this scene, this background music, these sound effects in this order" and the model plans and renders it all at once.

## The Methodology

**The tokenizer.** A frozen self-supervised encoder gives semantic features. A convolutional encoder with SnakeBeta activations reads the raw waveform at 50 Hz for acoustic features. The two are concatenated on the channel axis, squeezed to 12.5 Hz by a strided convolution, and passed through *one shared* quantizer — so every code layer carries both meaning and sound, rather than layer 0 being "the semantic one".

Two tricks shape where information lands:

- **Quantizer dropout at 0.5** — half the time, training randomly truncates the residual stack, which forces the early codebooks to carry the coarse, must-have content. Raising the rate strengthens the coarse layers further but degrades full-depth reconstruction, so 0.5 is a deliberate middle.
- **Semantic [[Distillation|distillation]]** — a branch regresses the frozen SSL teacher's features from the quantized latent. Applied *under* dropout, this concentrates semantics in codebook 0, which is exactly the codebook the LLM will own.

The decoder is a Vocos-style Transformer with [[RoFormer- Enhanced Transformer with Rotary Position Embedding|rotary embeddings]], a 25-frame sliding attention window, and an inverse-STFT head at 24 kHz. Fully causal, no look-ahead, so you can stream. Training is a [[Generative Adversarial Networks|GAN]] setup: multi-period waveform discriminator plus multi-resolution spectral discriminator, with feature-matching, multi-scale mel, and commitment losses. 700k hours of speech/music/sound with a modality-balanced sampler, then a second stage where encoder, quantizer and semantic branch are **frozen** and only the decoder and discriminators keep training — the token space stops moving, so anything already trained on these tokens stays compatible.

**The backbone.** A standard decoder-only Transformer, extended in two places.

*Output side:* codebook 0's 2048 entries are appended to the text vocabulary. One softmax, one LM head, text and audio freely interleaved:
$$c_{t,0} \in \{0,\dots,2047\} \iff \mathrm{id}(c_{t,0}) \in [V,\, V+2048)$$

*Input side:* each of the 16 codebooks has its own embedding table. The 16 looked-up vectors are **summed** into one frame embedding, pushed through the RVQ Adaptor (pre-norm residual blocks with [[Gated Activation|SwiGLU]], zero-initialised down projections), and the result is added element-wise to the codebook-0 token embedding — **at audio positions only**. Text positions get nothing from this path. Note codebook 0 appears twice on input: once as the vocabulary token, once inside the sum.

*The depth predictor:* at each audio frame it receives two prefixes — the backbone hidden state $\mathbf{z}_t$ (linearly projected to its width) and $c_{t,0}$ — then autoregressively emits $c_{t,1} \ldots c_{t,15}$, feeding each back in. The generation path factorizes as
$$p(c_{t,0},\ldots,c_{t,15}\mid \mathbf{z}_{<t},\mathbf{c}_{<t}) = \underbrace{p(c_{t,0}\mid \mathbf{z}_{<t},\mathbf{c}_{<t})}_{\text{LM head}} \prod_{k=1}^{15} \underbrace{p(c_{t,k}\mid \mathbf{z}_t, c_{t,0},\ldots,c_{t,k-1})}_{\text{RVQ predictor}}$$

The authors are careful that this is **exact**, not an approximation: past frames' residual codes do reach the backbone, because all 16 embeddings were summed into the input. What is given up is the reverse direction — the predictor's output never feeds back into the backbone — which is precisely why the two halves are only joined once the predictor has converged.

**The loss:**
$$\mathcal{L} = \mathcal{L}_{\text{tok}} + \lambda \mathcal{L}_{\text{sp}},\quad \mathcal{L}_{\text{tok}} = -\sum \log p(c_{t,0}, x_t),\quad \mathcal{L}_{\text{sp}} = -\sum_t \sum_{k=1}^{15} \log p(c_{t,k}\mid\cdot)$$
Both are ordinary [[Cross Entropy|cross-entropy]]. $\mathcal{L}_{\text{sp}}$ **sums** over depth rather than averaging — this is the whole reason $\lambda$ and the detach toggle exist.

**Four pretraining stages**, ~2.7T LLM tokens total (one temporal step = one token, so audio costs 12.5 tokens/second even though each carries 16 IDs):

| Stage | Focus | Batch (tok) | LR | Schedule |
|---|---|---|---|---|
| 1 | modality alignment | 4.19M | $2\text{e-}4 \to 2\text{e-}5$ | cosine |
| 2 | audio understanding | 12.58M | $2\text{e-}5$ | constant |
| 3 | generation, detached | 12.58M | $2\text{e-}5$ | constant |
| 4 | long-context cool-down | 25.17M | $2\text{e-}5 \to 1.5\text{e-}5$ | cosine |

- **Stage 1** — ASR and speech-to-text translation, text output supervised only. Backbone, LM head and predictor at **zero** learning rate; only the audio embeddings and adaptor train; token embedding gets a $0.1\times$ multiplier so the new audio rows can move. Because the adaptor is an exact identity at init and the backbone is frozen, text drift is *strictly zero* — no replay needed.
- **Stage 2** — everything unfrozen, audio understanding mixed **1:1 with the original text corpus**. Still no audio targets supervised. LR drops to a constant $2\times10^{-5}$ plateau that all later stages continue from without re-warming. From-scratch modules keep a $10\times$ LR multiplier and are excluded from [[Regularization|weight decay]].
- **Stage 3** — the main run. Generation arrives. Mixture is text : TTS : interleaved dialogue $= 3:1:2$, so text stays at 50% and TTS is one-sixth of every micro-batch. $\lambda = 1.0$, but the predictor's hidden-state prefix is **detached** ([[Backpropagation|stop-gradient]]), so the 15-codebook loss trains the predictor alone. Interleaved samples put 1 text token per 2 audio frames.
- **Stage 4** — detachment off, so the acoustic loss finally shapes the hidden states into good conditioning vectors. $\lambda$ drops $1.0 \to 0.1$ to bring the two terms to comparable size. Context extended $16{,}384 \to 32{,}768$ with context parallelism — at 12.5 Hz that is ~44 minutes of audio.

**Post-training.** SFT on ~5,000 hours across music, sound, voice design, singing and speech; sequence 16,384, batch 64, cosine $1\text{e-}5 \to 1\text{e-}6$, $\lambda$ held at 0.1, new modules still at $10\times$. Then [[Proximal Policy Optimization Algorithms|GRPO]] for instruction following. The reward is a product of two things:
$$R_i = \frac{\bar{s}_i}{100}\begin{cases}\exp(-\tau e_i), & e_i \le 0.5\\ 0, & e_i > 0.5\end{cases},\qquad \tau = 3$$
$\bar{s}_i$ is an instruction-consistency score 0–100: an audio-understanding model captions the generated audio, a text LLM compares that caption to the instruction, and they score it **four times and average** to cut judge variance. $e_i$ is CER or WER from an ASR model against the target transcript. Multiplying means gorgeous-sounding audio saying the wrong words scores zero. $G=16$ samples per instruction; groups with reward std $< 0.02$ are dropped (the DAPO dynamic-sampling trick) because they carry no preference signal; importance ratios are per-codebook, normalised by $1/(16|y_i|)$; a small [[KL Divergence|KL]] coefficient $\beta$ with clipping doing most of the constraining; LR $10^{-6} \to 10^{-7}$.

**The instruction format** has three fields: `ROLE` (who is speaking, what they sound like), `DIRECTOR` (the acoustic scene and intent), `SCRIPT` (speech and sound events laid out in order, each spoken line prefixed by a speaker tag with optional `(description)`, and non-speech events written as `[description]`).

## Ablation Studies and Experiments

Only two controlled comparisons, both after pretraining. They are the load-bearing evidence for the paper's two claims.

**Does the RVQ Adaptor matter?** With vs. without, on audio benchmarks:

| System | AISHELL-1 CER↓ | LibriSpeech WER↓ | MMAU↑ | CoVoST En→Zh↑ | Zh→En↑ | SpeechMMLU↑ |
|---|---|---|---|---|---|---|
| w/o adaptor | 5.25 | 6.00 | 40.70 | 12.05 | 5.99 | 12.13 |
| w/ adaptor | **3.00** | **3.41** | **51.70** | **30.56** | **18.59** | **58.76** |

These are not small deltas. SpeechMMLU goes from 12.13% (near chance for 4-way multiple choice) to 58.76%, and Zh→En BLEU triples. The reading: without a learned transformation, the summed 16-codebook vector is simply *out of distribution* for the backbone's input space, and everything downstream degrades.

**Does the four-stage recipe protect the text ability?** The baseline is a three-stage run (ASR → audio understanding+generation → cool-down) with no adaptor:

| System | FinEval | C-Eval | MMLU | CMMLU | MATH | GSM8K | BBH | HumanEval |
|---|---|---|---|---|---|---|---|---|
| Baseline | 65.86 | 66.34 | 64.99 | 66.69 | 36.62 | 67.94 | 59.61 | 47.56 |
| Interference-aware | **71.68** | **71.92** | **69.20** | **71.73** | **45.07** | **74.05** | **67.03** | **54.27** |

Uniform gains, largest on the hardest reasoning tasks — MATH +8.45, BBH +7.42, HumanEval +6.71. Reasoning is the first thing to go when you overwrite a backbone.

**TTS.** Deliberately judged by humans, not by CER or speaker similarity. The authors argue explicitly that those metrics are blind to what they optimised: a flat, lifeless read can score a perfect CER, and speaker similarity measures timbre, not delivery. Arena-style pairwise comparisons on a Chinese human-likeness set built from transcripts of real speech, 1,500 comparisons, ties allowed. StepAudio 3 Gen tops the Elo table at **1755.33**, and in head-to-head (100 comparisons per opponent) wins **82.0%** overall, between 73% and 90% against each of Qwen-Audio-3.0-TTS-Plus, Doubao-App, MiniMax-Speech-2.8-HD, Inworld-TTS-2, and its own predecessor StepAudio 2.5.

**Voice design.** On the full InstructTTSEval (1,000 ZH + 1,000 EN texts × 3 conditions = 6,000 utterances per system), judged binary-consistent by Gemini 3.1 Pro:

| System | EN AVG | ZH AVG |
|---|---|---|
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | 73.5 | 74.0 |
| Ming-omni-tts-0.5B | 65.1 | 71.0 |
| MOSS-VoiceGenerator | 64.9 | 65.1 |
| VoiceSculptor | — | 52.5 |
| **StepAudio 3 Gen** | **77.7** | **85.2** |

The Role-Play condition is where everyone struggles and the gap is biggest in Chinese: 75.5 vs 56.2 for the next best. Blind human Elo over 490 head-to-head comparisons: 1668.5, first place, 75.5% aggregate win rate (65.3–85.7% per opponent).

**What the ablations do *not* tell you.** This is the honest gap. The pretraining ablation changes **two things at once** — the baseline has no adaptor *and* only three stages — so you cannot separate "the adaptor helps text retention" from "the staged curriculum helps text retention". Given how large the adaptor's effect is in Table 2, a good chunk of Table 3 may just be the adaptor again. There is no ablation isolating the detach toggle, none on $\lambda$, none on the 50% text replay ratio, and none on the time-depth split versus MusicGen's delayed-pattern alternative — which is the paper's single biggest architectural fork. Nothing failed, because nothing was tested alone.

**Things reported as tried and rejected, in passing:** a higher quantizer dropout rate than 0.5 strengthens the coarse codebooks but hurts full-depth reconstruction. The delayed-pattern codebook scheme was rejected by argument, not experiment: it would force a multi-stream output interface and make the pretrained LLM absorb every residual layer's loss directly — exactly the interference the paper is built to avoid.

## Worth Remembering

- **The transferable lesson is the gradient bookkeeping, not the audio.** When you attach a new head whose loss sums over $k$ predictions per backbone step, that loss is roughly $k\times$ the size of the original objective. Detach until the new head converges, then reattach with a weight that equalises the two. That pattern applies to any multi-task or multi-head bolt-on, including a ranker with an auxiliary head, not just audio.
- **Zero-initialised residual adaptors are cheap insurance.** Same idea as the zero-init of a [[Deep Residual Learning for Image Recognition (ResNet)|residual branch]] and the $BA=0$ init of [[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]]: the module starts as an exact identity, so day-one behaviour is provably unchanged and you only pay for what it learns.
- **Freeze the token space once and you can iterate the decoder forever.** Stage-2 tokenizer refinement updates only the acoustic decoder and discriminators. Every model already trained on those codes stays valid. That is a real infrastructure win and worth copying anywhere embeddings are a shared substrate.
- **Evaluation is honest about what metrics cannot see.** The refusal to headline CER/speaker-similarity for TTS, with the reason stated plainly, is the right instinct — same instinct as refusing to headline [[NDCG|nDCG]] when the thing you changed is diversity. But the replacement is Elo over 1,500 human comparisons and an LLM-as-judge on a binary consistency call, with no confidence intervals, no inter-rater agreement, and no report of tie rates. An 82% win rate over 100 comparisons has a standard error around 3.8pp; the paper never says so.
- **The judge overlaps the training signal.** GRPO's reward uses an audio-understanding model to caption and a text LLM to score; InstructTTSEval is scored by Gemini judging style consistency. Different models, but the same *mechanism*, and RL against a caption-based judge plausibly teaches the model to produce audio that captions well. Treat the voice-design numbers with that in mind.
- **Everything is a technical report claim.** No weights, no code, no tokenizer release mentioned, no backbone size disclosed anywhere — you cannot tell whether it is a 7B or a 70B, which makes the comparison against a 0.5B and a 1.7B baseline unreadable as a fair fight.
- **Frames are 80 ms.** 12.5 Hz is coarse for speech, and the model relies entirely on the 16-deep residual stack to recover detail within a frame. This is what makes 44-minute contexts affordable at 32,768 tokens, and it is the bet the whole design rests on.
- **Open questions:** does the detach schedule still matter if the predictor is warm-started rather than random? Does 50% text replay beat 25% or 75%, and does the answer depend on how much the backbone is worth? How much of the depth predictor's 4 layers is actually used — is there a rank-like collapse in the residual codebooks the way [[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]] found for $\Delta W$?

## Links
Related: [[Auto-regressive models]] · [[Causal Attention]] · [[Cross Entropy]] · [[Recommender Systems with Generative Retrieval (TIGER)]] · [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)]] · [[Distillation]] · [[Generative Adversarial Networks]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Proximal Policy Optimization Algorithms]] · [[Training language models to follow instructions with human feedback]] · [[KL Divergence]] · [[Backpropagation]] · [[Gated Activation]] · [[RoFormer- Enhanced Transformer with Rotary Position Embedding]] · [[Regularization]] · [[Denoising Diffusion Probabilistic Models]]

New topics worth writing: GRPO (Group Relative Policy Optimization), DAPO dynamic sampling, neural audio codecs and SoundStream/EnCodec, quantizer dropout, Elo rating for pairwise model evaluation, LLM-as-judge reliability, catastrophic forgetting in modality grafting, zero-initialised adapters as a general pattern, MusicGen delayed codebook patterns, context parallelism

Tags: `llm` `transformers` `evaluation`
