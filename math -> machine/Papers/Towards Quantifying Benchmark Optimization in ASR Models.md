---
title: "Towards Quantifying Benchmark Optimization in ASR Models"
authors: ["Theo Lebryk", "David Ayllon", "Alice Baird", "Jakub Piotr Cłapa", "Jens Madsen", "Panagiotis Tzirakis"]
year: 2026
arxiv: "2608.19936"
url: https://arxiv.org/abs/2608.19936
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, llm]
---
## The Core Idea

Speech recognition models are ranked by word error rate (WER) on public test sets. A low WER is supposed to mean "this model hears speech well". This paper shows that for the top open-source models, a chunk of that low WER comes from something else: the model recognises *which benchmark it is listening to* from acoustic cues, and then reproduces that benchmark's transcript conventions — including the benchmark's **mistakes** — instead of writing down what was actually said.

> [!NOTE] Benchmark optimization
> Gains in a reported score that come from leaning on artifacts specific to that benchmark, not from a real improvement in the underlying skill. Colloquially "benchmaxxing". ^benchmark-optimization

The clever move is *how* they measure it. You cannot just say "the model overfit". Instead they hunt for cases where **the audio underdetermines the reference transcript** — situations where a genuinely faithful listener could not possibly produce the exact reference string. Three such situations:

1. The reference transcript is simply **wrong** (VoxPopuli is full of errors). A faithful model should transcribe what was said, not the typo.
2. The word's audio has been **silenced**. A faithful model cannot know a masked number.
3. Two spellings sound **identical** (`Mr` vs `Mister`). A faithful model has no acoustic basis to pick the one this corpus happens to use.

If a model matches the reference far above chance in these cases, it is reading a cue that is not the speech.

The finding: the six models with the best VoxPopuli WER (5.4–5.8%) are exactly the six with the highest rate of reproducing the reference's errors (0.18–0.30). Every model at 6.5% WER or worse sits at or below 0.10. Whisper-Large-v3 is at 0.02.

And the behaviour is **narrow** and **switchable**. It fires on real benchmark audio and on TTS voice-clones of benchmark speakers. It mostly does *not* fire on generic TTS voices, or on fresh European Parliament recordings scraped in June 2026 (after every model's cutoff) that are otherwise in-distribution. You can turn it on by gluing 8 seconds of VoxPopuli audio onto the end of a clip. You can turn it off by gluing on 8 seconds of conversational audio. You can turn it off by subtracting a single direction from one encoder layer's activations. That is the punchline: the models have learned a *conditional policy* — "if this smells like VoxPopuli, emit VoxPopuli-style text" — and they route into it based on a compact, nearly linear acoustic signal. This is [[Shortcut Learning in Deep Neural Networks|shortcut learning]] with a switch on it, which is worse than plain overfitting, because the model's general ability looks undamaged while its benchmark number is inflated.

## The Methodology

**Models (11).** Encoder–decoder / transducer: Whisper-Large-v3, Cohere-Transcribe, Parakeet-TDT-0.6B-v2, Moonshine-Streaming. Speech-LLMs (audio encoder feeding a text LLM decoder): Canary-Qwen-2.5B, Granite-Speech-4.1-2B, Higgs-Audio-v3-8B, Kimi-Audio-7B, Phi-4-Multimodal, Qwen3-ASR-0.6B, Voxtral-Mini-3B. Parakeet is excluded from likelihood readouts because its token-and-duration transducer makes reading logits at an arbitrary position awkward.

**Data.** VoxPopuli English (European Parliament) is the main target — widely used, and known to be riddled with reference errors. LibriSpeech clean/other secondary. Controls: *DaiKon*, 450 private conversational clips postdating all training cutoffs; *ep-fresh*, freshly scraped June-2026 Parliament recordings following the original VoxPopuli collection recipe; *libri-fresh*, 2026 LibriVox recordings from 14 readers whose catalogues start after every cutoff. Synthetic stimuli come from Qwen3-TTS, either cloning a specific dataset speaker or using a stock generic voice.

**The two readouts.** Each probe marks positions where audio underdetermines the text, and defines a *reference rendering* $r$ (what the benchmark says) against an *audio-true rendering* $a$ (what was actually said).

- `accept-ref` — black box. The fraction of those positions where greedy decoding emits $r$ rather than $a$.
- `audio lift` — white box. Teacher-force the reference span with real audio, then again with the waveform zeroed out ($x_\emptyset$), and take the difference, normalised by character count so different tokenizers stay comparable:

$$\lambda(r) = \frac{\log p_M(r \mid x) - \log p_M(r \mid x_\emptyset)}{|r|_{char}}$$

$\lambda(r) > 0$ means the *audio* — not the language-model prior — pushed the model towards the reference. Since these are cases where the audio should not support the reference, positive lift is the smoking gun. This is the same "subtract the prior to isolate the signal" logic as any contrastive readout; it is essentially a per-character log-likelihood-ratio.

A wrinkle they handle: some models dump huge probability on the EOS token when fed pure silence (Moonshine 0.76, Granite 0.61, Qwen3-ASR 0.25). EOS is removed from the softmax denominator and the distribution renormalised, symmetrically on both terms. This *shrinks* the measured lift, so it is the conservative choice.

**Probe 1 — reference disagreement.** Find spans where the reference transcript is wrong. Scaling human annotation is expensive, so they use a **consensus panel**: four accurate transcribers (chosen by phoneme error rate — Kimi-Audio, Qwen3-ASR, Voxtral-Mini, Moonshine), align each hypothesis to the reference, and keep only per-word edits all four flag unanimously. Result: 1,113 edits over 745 VoxPopuli test clips (586 substitutions, 441 deletions, 86 insertions). 40% of clips carry a flagged edit; ~3% of all reference words. Validated against a human-annotated VoxPopuli set: 93% of consensus-flagged edits also appear in the human annotations. Panel members are scored leave-one-out against the other three, otherwise their own `accept-ref` would be zero by construction.

**Probe 2 — masked-number recovery.** Force-align words with `mms-fa`, then silence every sample over a number's interval (padded 120 ms for aligner slop; every occurrence of that number in the clip is masked so it cannot leak from a repeat mention). Numbers are picked because the language prior gives you little — surrounding pitch, room, and speaker say nothing about whether it is 40 or 50. "One" is excluded as too guessable. Clips under 4 s are skipped so the mask is not a large fraction of the audio.

**Probe 3 — orthographic switching.** Pairs that sound identical: `Mr`/`Mister`, `any one`/`anyone`. Raw `accept-ref` is uninformative here — a model that always writes "mister" scores high whenever the corpus agrees. So they report the **switch rate** $s$ = the smaller of the two conditional `accept-ref` values. A fixed-form transcriber gets $s = 0$; a coin-flipper gets $s \le 0.5$; $s$ reliably above $0.5$ means the model is reading acoustic cues to match each clip's convention. The honorific pair requires knowing *which dataset* a clip came from (VoxPopuli always `Mr`, LibriSpeech always `Mister`). Archaic spacing is harder — both forms appear *inside* LibriSpeech, so switching requires tracking sub-populations or individual samples.

**Mechanism work.** Four levers:

- *Truncation.* Cut the audio to a tight window around the edit ($\pm 1$ aligned word $\pm 0.25$ s). If the model then transcribes faithfully, the earlier failure was gated by context, not perception.
- *Splicing.* Stimulus is $[\text{base}][0.15\text{s silence}][\text{8s donor}]$, donor RMS-matched. Reported number is a difference-in-differences: flip rate with a VoxPopuli donor minus flip rate with a duration-matched conversational donor, so generic suffix effects cancel.
- *Activation patching.* Take cases where truncation produced the audio-true output, then swap the target span's encoder frames for the context-free encoding of that span. If the faithful output returns, the override lived in the encoder. If nothing changes, the decoder is ignoring a faithful representation.
- *Linear steering.* Learn a diff-in-means direction. Train on 22 *ep-fresh* courtesy-opener clips, each rendered twice — once with a real VoxPopuli 5 s suffix, once with an audiobook suffix. Hook every encoder layer, mean-pool activations **over the base-clip frames only** (the suffix's effect leaks backwards through bidirectional attention), and set $d_L = \text{mean}(\text{vox}) - \text{mean}(\text{control})$, unit-normalised. Induction adds $\alpha d_L$ at one layer; ablation projects it out. Specificity floor is a norm-matched random direction. 21 clips held out.

**Case study anchor.** VoxPopuli references systematically drop the audible opening courtesy ("thank you, Mr President"). The high-`accept-ref` models reproduce that omission. This gives a fixed, position-anchored target for the attention and translation levers.

## Ablation Studies and Experiments

**Headline correlation (VoxPopuli, Figure 2 + Table 1).** `accept-ref` on consensus-flagged edits vs the same on human-annotated edits:

| Model | consensus | human-annot. | 95% CI |
|---|---|---|---|
| Cohere-Transcribe | 0.30 | 0.52 | [0.47, 0.58] |
| Canary-Qwen-2.5B | 0.23 | 0.41 | [0.36, 0.47] |
| Granite-Speech-4.1-2B | 0.21 | 0.42 | [0.36, 0.47] |
| Higgs-Audio-v3-8B | 0.21 | 0.39 | [0.33, 0.44] |
| Phi-4-Multimodal | 0.19 | 0.38 | [0.33, 0.44] |
| Parakeet-TDT-0.6B-v2 | 0.18 | 0.38 | [0.32, 0.43] |
| Qwen3-ASR-0.6B | 0.09 | 0.19 | [0.15, 0.24] |
| Whisper-Large-v3 | 0.02 | 0.08 | [0.05, 0.11] |

The two rankings are nearly identical (only Granite and Canary swap), so the cheap automated panel is a decent stand-in for humans.

**Audio lift on masked numbers (Table 4, 115 paired sentences, nats/char).** Cohere: $+1.52$ real, $+1.26$ vox-clone, $+0.92$ ep-fresh clone, $+0.54$ generic. Real minus generic $= +0.98$, CI $[+0.66, +1.33]$. Canary $+1.22 \to +0.77$. That gradient is the whole argument in one row: same sentence, same masked number, only the voice changes, and the model's ability to "hear" a silenced number falls off a cliff.

Two control models make this interpretable. **Whisper's lift *rises* on clean TTS** ($+0.70$ real $\to +1.01$ generic, real$-$generic $= -0.31$), so a drop in lift is not a synthesis artifact. **Qwen3-ASR's lift is negative in every condition**, so its positive diffs do not indicate recovery.

**Hard-cell check (A.4).** Worry: masked-number recovery is just language-model guessing. So they re-score only *hard cells* — spans whose silenced-audio prior NLL/char is $\ge 3.5$ nats (62 of 92 spans). Recovery **rises** for the elevated models: Cohere $0.27 \to 0.36$, Canary $0.14 \to 0.24$, Higgs $0.15 \to 0.23$, Granite $0.10 \to 0.19$. Near-zero models stay at the floor. Recovery concentrates on exactly the spans the prior *cannot* supply.

**Orthographic switching.** 6/11 models significantly beat the 0.5 switch-rate baseline on `Mr`/`Mister`. 8/11 beat it on archaic spacing — meaning models discriminate not just *between* corpora but between sub-populations or individual samples *within* LibriSpeech.

**Trigger battery, `accept-ref` on consensus edits (Table 2).**

| Model | full | truncated | conv. donor appended | activation ablated |
|---|---|---|---|---|
| Cohere | 0.30 | 0.13 | 0.06 | 0.04 |
| Canary-Qwen | 0.23 | 0.12 | 0.05 | 0.02 |
| Granite | 0.21 | 0.12 | 0.20 | 0.14 |
| Higgs | 0.21 | 0.12 | 0.03 | – |
| Phi-4 | 0.19 | 0.09 | 0.05 | 0.20 |
| Parakeet | 0.18 | 0.08 | 0.06 | 0.01 |
| Whisper | 0.02 | 0.05 | 0.02 | – |

Note the low-scoring models go *up* slightly under truncation — the effect is specific to the benchmark-optimized group, not a generic artifact of shortening audio.

**Induction (splicing a VoxPopuli suffix onto ep-fresh clones), vox minus control donor:** Phi-4 $+.096$ [.074,.120], Canary $+.093$, Higgs $+.069$, Cohere $+.066$, Parakeet $+.044$. On *generic*-voice bases the effect is much smaller (Phi-4 $+.018$, Cohere $+.015$), consistent with the trigger needing both a plausible register and a benchmark cue.

**Real vs ep-fresh on each corpus's own content (Table 3).** Cohere: 0.304 vs 0.122 consensus, 0.185 vs 0.074 masked. Fresh, genuinely in-distribution Parliament audio — same domain, same reference-error style, down to the courtesy omission — does not trigger the behaviour nearly as much. That is the clearest evidence that this is not "the model learned Parliament speech", it is "the model learned *these recordings*".

**Steering results.** The learned direction bidirectionally flips Cohere, Parakeet, Canary and partially Granite. Ablation is the strong direction: `accept-ref` falls 82–92%, flipping 88% (Cohere, 303 flips), 95% (Parakeet, 201), 93% (Canary, 252), 45% (Granite, 115) of reference reproductions to audio-true, with $\le 11$ regressions each. WER against the *original erroneous* reference rises; WER against the consensus-*corrected* transcript falls 25–40%. Induction on generic clones is real but weaker: Cohere $0.016 \to 0.072$, Canary $0.011 \to 0.105$, Parakeet $0.013 \to 0.047$, Granite $0.017 \to 0.041$. Random norm-matched direction is flat (0.011–0.015). The direction is genuinely **low-rank**: restricting $d_L$ to the top-1 PCA component recovers 65–80% of the induction effect (Cohere 0.71, Parakeet 0.76, Canary 0.67); $k=4$ gets 85–100%.

Cross-probe transfer, which is the strongest single result: ablating Cohere's *reference-disagreement* direction halves *masked-number* recovery ($0.102 \to 0.051$, $n=157$, 10 lost vs 2 gained flips, exact McNemar $p = .039$), random arm exactly flat. One direction, two probes.

**Where the override lives (Figure 12, Table 5).** Patching a context-free encoding into the target frames removes reference **insertions** (so those were written into the encoder state) but never restores reference **omissions** — for deletions and substitutions the decoder overrides a locally faithful representation. The courtesy case study nails it: Cohere emits the courtesy 0.00 of the time on the full clip, 0.94 when truncated, 0.26 when the decoder's attention is restricted to the opener's frames. Granite emits it 0.00 on transcription but **0.39 when asked to translate the same audio to Spanish** — the information is there, the transcription policy just refuses to use it. Phi-4: 0.00 full, 0.89 attention-isolated, 0.61 translated.

**What did not work / negative results.**

- *The leakage hypothesis.* The HF version of VoxPopuli leaks 40% of test speakers into train. But no model showed a leaked-speaker advantage: pooled elevated-model `accept-ref` was **0.22 on leaked vs 0.26 on unleaked** speakers, masked recovery indistinguishable. Official train/test speaker overlap does not explain the behaviour.
- *Steering Phi-4.* Its direction cleanly separates the spliced conditions but is **causally inert**: ablation $0.200 \to 0.201$, induction flat. Its policy appears to live in decoder parameters, not the encoder. (Yet the input-level donor lever moves it fine, $0.19 \to 0.05$.)
- *Steering Higgs.* No clean operating point. At $\alpha=4$ ablation destroys decoding wholesale (WER vs corrected transcript $0.06 \to 0.27$, 36 garbage clips); at $\alpha \le 2$ it is inert ($0.186 \to 0.174$). No activation cell reported.
- *Granite under TTS.* It keeps orthographic conventions on real audio but loses them under *any* TTS clone, making its clone cells uninformative. Whisper, Phi-4 and Voxtral never emit "Mister" at all, so the honorific probe is vacuous for them.
- *Acoustic perturbations dissociate the models.* Additive noise at 10 dB collapses Canary ($.22 \to .03$), Parakeet ($.17\to.04$) and Higgs ($.20\to.07$), but Cohere, Granite and Phi-4 keep the behaviour under both noise and measured room reverberation (Cohere $.29 \to .29/.26$). Different models key off different subsets of cues — and crucially, this means benchmarks built by *augmenting* existing sets with noise/reverb inherit the same contamination.
- *The masked-under-reverb readout is weak evidence*, because reverberation smears a masked word's energy past the mask boundary.

## Worth Remembering

- **The measurement problem vs the coverage problem.** Most work on the benchmark-reality gap says "our benchmarks miss conditions, add more benchmarks". This paper says the existing benchmarks are *mis-measuring*, and adding derived benchmarks (noise/reverb augmentations of the same audio) does not fix it — several models' benchmark policy survives exactly those perturbations.
- **VoxPopuli has so many reference errors that any model below 3% WER must be transcribing errors.** If you see a sub-3% VoxPopuli number, that is the ceiling of faithfulness, not evidence of skill.
- **The data-scale correlation, offered tentatively.** The high-`accept-ref` models (Phi-4, Cohere, Granite, Canary) trained on under ~1M hours by public accounts. Qwen3-ASR trained on 40M hours of weak supervision and shows only mild inclinations; Whisper is lowest of all. This is correlational and rests on incomplete disclosures, but it suggests scale of weakly-supervised data is a defence — a [[The Bitter Lesson (essay)|bitter-lesson]]-flavoured conclusion.
- **They explicitly do not know how the behaviour arises during training.** All the work is at inference time. Whether it comes from model selection on the benchmark, data leakage, memorisation, or something else is open. Hence their first recommendation: transparent training data mixtures.
- **Using acoustic context is not inherently bad.** Prosody, speaker, room legitimately inform transcription. The probes are specifically constructed so that no acoustic cue *justifies* preferring the reference.
- **Why audio is a worse case than text.** ASR decoders attend over the whole audio encoding, which carries speaker, channel and environment information alongside content. That gives far more degrees of freedom for a "which dataset am I in" classifier than a causally-masked text LM has. Expect this to get worse as RL enters speech modelling — this is a textbook reward-hacking channel.
- **Practical caveats.** The consensus panel is a proxy: it catches 93% of human-flagged edits but misses formatting changes, and panel members must be scored leave-one-out. The TTS stimuli pass an intelligibility gate (at least one of 11 models transcribes the sentence at zero WER; pass rates 0.84–0.93), which mildly biases the sample. The masked-number sample drops ~20% of targets where inter-word gaps are under 80 ms, which biases toward longer, harder numbers — favourable for their claim, but worth knowing.
- **Actionable takeaway for an ML engineer:** if you are picking an ASR model, do not read WER off a leaderboard. Run the reference-disagreement probe on a held-out slice, or at minimum evaluate on data collected after every candidate's training cutoff. Their recommendation against i.i.d. test splits — use temporal or speaker stratification, and keep the headline eval set private — generalises well past ASR.
- **Open question worth chasing:** the steering direction is rank-1-ish and transfers across probes. Could you regularise it away during training — penalise the model for having a linearly decodable "which corpus is this" direction in its encoder? That is a concrete follow-up the paper gestures at but does not attempt.

## Links

Related: [[Shortcut Learning in Deep Neural Networks]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[On the Difficulty of Evaluating Baselines]] · [[Understanding Deep Learning Requires Rethinking Generalization]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Training language models to follow instructions with human feedback]] · [[Attention Is All You Need]] · [[Causal Attention]] · [[Scaling Laws for Neural Language Models]] · [[The Bitter Lesson (essay)]] · [[Cross Entropy]] · [[Distillation]]

New topics worth writing: Word Error Rate and phoneme error rate, Activation patching / causal tracing, Activation steering and diff-in-means directions, Test-set contamination in LLMs, Forced alignment (CTC and mms-fa), Speech-LLM architectures, RNN-Transducer and TDT decoding, McNemar's test, Wilson confidence intervals, Reward hacking in RL
Ge
