---
title: "Evaluating Music Context Preservation: A Multi-facet Framework for Music Editing Systems"
authors: ["Yash Vishe", "Eric Xue", "Xunyi Jiang", "Zachary Novack", "Junda Wu", "Julian McAuley", "Xin Xu"]
year: 2026
arxiv: "2512.14629"
url: https://arxiv.org/abs/2512.14629
priority: Low-Priority
read_on: 2026-10-01
tags: [paper, llm]
---
## The Core Idea

Music editing systems take an audio clip plus an instruction ("make this a jazz piece", "remove the bass") and return edited audio. Everyone measures whether the *requested* change happened. Almost nobody measures whether everything else stayed put.

That "everything else" is the paper's target. They name it **Music Context Preservation (MuseCP)**: the musical attributes you did *not* ask to change should survive the edit. If you ask for a timbre swap and the tempo drifts 20 BPM, the system technically obeyed you and still ruined the track.

> [!NOTE] Music Context Preservation (MuseCP)
> How well an edited output $x'$ retains the musical attributes of the source $x$ that the instruction never asked to change. It is a *preservation* score, not a quality score. ^musecp

The gap is real and documented. Table 1 of the paper walks through eight recent editing systems (2023–2026). Two of them — AUDIT and Instruct-MusicGen — report no preservation evaluation at all. ZETA checks only structure. Audio Prompt Adapter checks only harmony. SteerMusic checks only melody. Each paper picked one facet, usually the one its method was good at.

So this is a measurement paper, not a modelling paper. The contribution is **MuseCPEval**: ten metrics grouped into four musical facets (harmony, rhythm & meter, structure, melody & motif), plus two more for timbre in the appendix. Nothing here is a new algorithm — every metric is assembled from existing MIR (music information retrieval) tools: `librosa`, `mir_eval`, `msaf`. The work is in the *selection*, the *normalisation choices*, and then proving the metrics actually respond to the thing they claim to measure.

Why it did not exist before: the field was busy making editing work at all. Preservation is the boring second half of the problem, and it needs a per-facet decomposition to be useful. A single scalar "similarity to source" tells you nothing actionable — a system could be scoring 0.9 because harmony is perfect and rhythm is broken, or because everything is mildly smeared.

What it unlocks is diagnosis. Section 4 runs the framework across four systems and reads their architectures off the score profile. Adapter-based conditioning preserves global harmony but loses beat alignment. DDPM inversion preserves both harmony and beat because the latent trajectory anchors the output. Frozen-backbone LoRA fusion keeps coarse tonality and loses time-locked rhythm. The metric profile becomes a fingerprint of the method.

This is the same move as [[Evals]] in the LLM world, or [[On Sampled Metrics for Item Recommendation (KDD)|the sampled-metrics paper]] in recommenders: stop trusting one aggregate number, and check that your instrument actually measures what its name says.

## The Methodology

Ten metrics. Let $x$ be the source audio and $x'$ the edited output. Every metric compares the two directly — there is no separate ground-truth reference.

### Harmony (3 metrics)

**Circle-of-fifths distance** — does the key change? The tonic $k \in \{0,\ldots,11\}$ and mode (major/minor) are estimated for both clips with the Krumhansl–Schmuckler algorithm, a classic key-detection method that correlates the piece's pitch-class histogram against learned key profiles. A minor key is folded onto its relative major with $\tilde{k} = (k+3) \bmod 12$, because C minor and E♭ major share a key signature and listeners hear them as close. Then map each tonic to its position $\phi(\tilde{k})$ on the circle of fifths and take the shorter way round:

$$\text{CoF}(x,x') = \frac{1}{6}\min\big(|\phi(\tilde{k}) - \phi(\tilde{k}')|,\ 12 - |\phi(\tilde{k}) - \phi(\tilde{k}')|\big)$$

$0$ = same key, $1$ = maximally distant (C major vs F♯ major). Lower is better.

**Chroma similarity** — does the *pitch-class mix* survive? Both clips become chroma matrices $\mathbf{H} \in \mathbb{R}^{T \times 12}$ via Constant-Q Transform (a spectral transform with logarithmically spaced bins, so bins line up with musical semitones). Average over time to get one 12-vector $\bar{\mathbf{h}}$ per clip, then cosine similarity:

$$\text{ChromaSim}(x,x') = \frac{\bar{\mathbf{h}} \cdot \bar{\mathbf{h}}'}{|\bar{\mathbf{h}}||\bar{\mathbf{h}}'|}$$

This throws away all time information. It is a bag-of-pitch-classes measure.

**Chroma DTW similarity** — the same content, but time-aware. Build the frame-by-frame cosine *distance* matrix $D_{ij} = 1 - \frac{\mathbf{h}_i \cdot \mathbf{h}'_j}{|\mathbf{h}_i||\mathbf{h}'_j|}$, run Dynamic Time Warping to find the cheapest monotone alignment path $P^*$ from $(1,1)$ to $(T,T')$, then average similarity along that path:

$$\text{ChromaDTWS}(x,x') = \frac{1}{|P^*|}\sum_{(i,j)\in P^*}(1 - D_{ij})$$

DTW is the [[Dynamic Programming|dynamic-programming]] alignment that lets one sequence stretch to match the other, so a tempo change does not automatically destroy the score.

> [!NOTE] Dynamic Time Warping
> Finds the cheapest monotone, continuous alignment between two sequences of different lengths, by dynamic programming over a distance matrix. Used here so that a metric measuring *pitch content* is not confounded by *timing*. ^dtw

### Rhythm & meter (3 metrics)

**Folded beat difference.** Estimate BPM for both clips with the `librosa` beat tracker. Naively subtracting is fragile: beat trackers suffer *octave errors*, locking onto half or double the true pulse. So fold:

$$\Delta_{\text{BPM}}(x,x') = \min\Big(|\text{BPM} - \text{BPM}'|,\ |\text{BPM} - 2\,\text{BPM}'|,\ \big|\text{BPM} - \tfrac{\text{BPM}'}{2}\big|\Big)$$

**Beat F-measure.** Extract beat timestamps from both. A beat in $x'$ counts as a true positive if it lands within $\pm 70$ ms of a reference beat in $x$. Standard F1 over precision and recall, via `mir_eval`. This is *phase-sensitive* — shift every beat by 150 ms and the score collapses even though the tempo is identical.

**Information gain.** For each reference beat, take the signed timing error to the nearest estimated beat, normalised by the local beat period. Repeat in the reverse direction so both missed and spurious beats are penalised. Bin the errors into a circular histogram with $K=41$ bins, giving distribution $p$, and measure how far $p$ is from uniform:

$$\text{IG}(x,x') = \frac{\log_2 K - H(p)}{\log_2 K}$$

where $H(p) = -\sum_k p_k \log_2 p_k$. This is [[KL Divergence|KL divergence]] to the uniform distribution, normalised. High IG means errors pile up at one consistent phase offset — the pulse is coherent, just displaced. Low IG means the timing errors are scattered, so there is no metrical relationship at all. That distinction is exactly what Beat F-measure cannot make.

### Structure (2 metrics)

Segments are extracted from both clips with `msaf` (a music structure analysis toolkit), giving each time point a section label like A, B, C.

**Structural pairwise F-measure.** Look at every *pair* of time points and ask: do both labellings put them in the same section? Agreement on togetherness is a true positive, $x'$-only is a false positive, $x$-only is a false negative. F1 over the resulting $P_{\text{pair}}$, $R_{\text{pair}}$.

**Adjusted Rand Index.** Same clustering-comparison idea, but it also credits agreement on *separation*, and it subtracts off the agreement you would get from random labelling:

$$\text{ARI}(x,x') = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}$$

$1$ = perfect agreement, $0$ = chance level, negative = worse than chance.

### Melody & motif (2 metrics)

**Contour DTW similarity.** Mechanically identical to Chroma DTW — 12-dim CQT chromagrams, cosine distance matrix, DTW path, mean similarity along it. The stated difference is framing: this facet "first isolates melodic content from key-level variation". (The paper is thin here; the formula printed is the same shape, and it is even mis-numbered as Equation 7, duplicating the structure equation.)

**Motif 3-gram recall.** Tokenise each track into the sequence of pitch-class *intervals* between successive notes. Let $\mathcal{G}_n(x)$ be the set of contiguous length-$n$ subsequences. With $n=3$:

$$\text{MotifRecall}(x,x') = \frac{|\mathcal{G}_n(x) \cap \mathcal{G}_n(x')|}{|\mathcal{G}_n(x)|}$$

The fraction of the source's short melodic figures that reappear in the output. Intervals rather than absolute pitches, so a transposed motif still counts — but an *inverted* one does not, since flipping interval signs produces different tokens.

### Timbre (appendix, added after ISMIR review)

**SKL similarity.** Compute MFCCs (Mel-frequency cepstral coefficients — a compact description of spectral shape). Drop coefficient 0, which encodes loudness rather than timbre; keep 12, stack with first and second time derivatives for $F=36$ dims. Fit one multivariate Gaussian per clip and take the symmetrised KL divergence, which has a closed form:

$$\text{SKL} = \tfrac{1}{2}\Big[\text{tr}(\bm{\Sigma}'^{-1}\bm{\Sigma}) + \text{tr}(\bm{\Sigma}^{-1}\bm{\Sigma}') + \bm{\delta}^\top(\bm{\Sigma}^{-1} + \bm{\Sigma}'^{-1})\bm{\delta} - 2F\Big]$$

with $\bm{\delta} = \bm{\mu}' - \bm{\mu}$ (the log-determinant terms cancel in the symmetric sum). SKL is unbounded and heavy-tailed, so it is squashed: $\text{SKLSim} = 1/(1 + \text{SKL}/50)$.

**Mean MFCC cosine.** Cosine similarity of the time-averaged MFCC vectors. Range $[-1,1]$.

### Validation setup

50 MIDI files from the Lakh MIDI Dataset, chosen for a clear main melody, stable metre, and recognisable sectional form. Ten facet-targeted edits per file → 500 source/edited pairs. Both versions rendered to audio with FluidSynth and the FluidR3_GM soundfont, so the instrument mapping is identical and any measured difference comes from the edit, not from synthesis.

Editing in the symbolic (MIDI) domain is the key design choice. It lets each operation change *exactly one* facet by construction, which is the controlled condition you need to test whether a metric responds only to its own facet. The "main melody" is defined as the monophonic track with the highest mean pitch among tracks with continuous note activity; sections come from each source's existing annotation. Both are read off the score, not recovered from audio.

## Ablation Studies and Experiments

### Objective validation: does each metric fire only for its own facet?

Table 2's design is the interesting part. Every row is an edit with a *derived* expected value, not one chosen after seeing the result.

**Harmony edits behave correctly.**

| Edit | CoF ↓ | Expected | ChromaSim ↑ | $\Delta$BPM ↓ | BeatF ↑ |
|---|---|---|---|---|---|
| +7 semitones | 0.18 ± 0.09 | ≈0.167 | 0.94 ± 0.05 | 1.26 ± 8.79 | 0.90 ± 0.19 |
| −3 semitones | 0.50 ± 0.11 | 0.5–0.67 | 0.82 ± 0.11 | 0.00 ± 0.00 | 0.92 ± 0.16 |

A perfect fifth moves the tonic one step round the circle, so $\frac{1}{6}\min(1,11) \approx 0.167$ — measured 0.18. A minor third moves it three steps, so $3/6 = 0.5$ — measured 0.50, exactly. And rhythm is untouched, as it should be.

**Rhythm edits behave correctly, and the two rhythm metrics split apart usefully.**

| Edit | $\Delta$BPM ↓ | BeatF ↑ | IG ↑ | ChromaSim ↑ |
|---|---|---|---|---|
| Tempo +50% | 26.10 ± 12.96 | 0.24 ± 0.08 | 0.19 ± 0.11 | 0.99 ± 0.00 |
| All beats shifted 150 ms | 1.35 ± 8.80 | 0.03 ± 0.07 | 0.63 ± 0.13 | 0.99 ± 0.00 |

This is the sharpest result in the paper. A constant 150 ms displacement gives $\Delta$BPM ≈ 0 (tempo unchanged), BeatF ≈ 0.03 (150 ms blows past the ±70 ms tolerance, so nothing matches), but IG stays at 0.63 — because all the timing errors sit at *one* phase offset, so the histogram is concentrated, not uniform. Tempo scaling, by contrast, crushes IG to 0.19: errors scatter, no coherent metrical relation survives. The three metrics are measuring genuinely different things, and the ablation proves it rather than asserting it.

Note also $\Delta$BPM's standard deviation of 8.80 on an edit that should read 0. That is the beat tracker failing on a handful of clips, and it means $\Delta$BPM is noisy per-sample even when its mean is right.

**Structural metrics are the honest failure.** For edits that touch nothing structural, StructPairF and ARI should read ≈1. They do not:

| Edit (structure-preserving) | StructPairF ↑ | ARI ↑ |
|---|---|---|
| +7 semitones | 0.67 ± 0.14 | 0.30 ± 0.29 |
| Tempo +50% | 0.66 ± 0.16 | 0.25 ± 0.29 |
| Beats shifted 150 ms | 0.74 ± 0.16 | 0.48 ± 0.30 |

| Edit (structural) | StructPairF ↑ | ARI ↑ |
|---|---|---|
| ABC → AAA | 0.58 ± 0.17 | 0.19 ± 0.24 |
| ABC → ABA | 0.65 ± 0.16 | 0.31 ± 0.27 |

ARI reads 0.30 for a pure transposition that changed no boundary whatsoever. The cause, which the authors state plainly: the `msaf` segmentation pipeline is sensitive to spectral changes beyond structural boundaries. Transpose the audio and the segmenter draws different lines.

The gap between structure-preserving and structural edits still exists — ABC→AAA (0.19) sits below transposition (0.30) — but it is small relative to the ±0.29 standard deviation. The honest reading, which the paper gives: these are **relative indicators only**. Never read an absolute structure score; only compare two systems on the same data. ARI in particular has standard deviations comparable to its own mean, making single-run comparisons untrustworthy.

**Melodic metrics move in the right direction but are compressed.** The global interval flip gives ContourDTWS 0.78 and MotifRec 0.61. The localised climax inversion gives 0.95 and 0.91 — a smaller perturbation producing a smaller drop, which is the right ordering. But note that the *harmony* edits also move these: +7 semitones gives ContourDTWS 0.78 and MotifRec 0.61, numerically identical to the melody edit. Because contour inversion touches only the main melody (a median of 9% of notes in the mix) while transposition touches everything, the melodic metrics cannot cleanly separate "you changed the melody" from "you changed the key". The facets leak.

### Human listening study

Four pairwise questions per facet. Each presents the original plus two edited versions at *different edit strengths* — the strength ordering is the gold label. Participants pick which is farther from the original with respect to the named facet. 33 responses collected, 11 retained after sanity checks.

| Facet | Human–Gold | Metric–Human | Metric–Gold |
|---|---|---|---|
| Harmony | 72.7 | 65.9 | 93.2 |
| Rhythm & Meter | 100.0 | 100.0 | 100.0 |
| Structure | 72.7 | 65.9 | 72.7 |
| Melody & Motif | 72.7 | 45.5 | 72.7 |

Read this carefully, because the headline "validated" claim rests on it.

Rhythm is perfect across the board — humans, metrics and gold labels all agree 100%. Timing differences are unambiguous to hear.

Harmony metrics beat the humans. Metric–Gold is 93.2% while Human–Gold is only 72.7%. The metric tracks the designed edit strength more reliably than listeners do.

**Melody is where it breaks.** Metric–Human is 45.5% — below the 50% you would get by coin flip on a two-way choice. The metric and the listener disagree more often than they agree. The paper's framing ("this facet is more subjective and harder to quantify") is fair but soft. With 11 participants and 4 questions per facet, that is 44 judgements per facet; the confidence interval on 45.5% is wide enough that you cannot distinguish it from chance in either direction. Combined with the facet leakage above, treat the melodic metrics as the weakest part of the framework.

Also worth flagging: Human–Gold is 72.7% for three of four facets, meaning humans disagree with the intended strength ordering roughly a quarter of the time. That caps how high Metric–Human can go even for a perfect metric.

### Case studies: four systems, four fingerprints

Applied to MusicMagus (zero-shot diffusion with latent embedding steering and [[Cross Attention|cross-attention]] consistency), ZETA (zero-shot DDPM inversion), Audio Prompt Adapter (AudioMAE features injected into AudioLDM2 through decoupled cross-attention), and Instruct-MusicGen (MusicGen plus audio/text fusion modules and [[LoRA]] on text cross-attention, backbone frozen).

**Critical caveat, stated by the authors: these systems cannot be compared to each other.** Each follows its own paper's evaluation protocol, on different data, with different edit tasks. MusicMagus uses 60 AudioLDM2-generated samples; ZETA uses 34 MedleyDB excerpts with 324 instructions; AP-Adapter uses 40 samples × 3 instrument edits; Instruct-MusicGen uses 150 pairs per task from Slakh. The numbers are within-system diagnostics only.

The one metric reported as a clean table:

| System | $\Delta$BPM ↓ |
|---|---|
| MusicMagus | 5.573 ± 10.315 |
| ZETA | 4.253 ± 12.234 |
| Audio Prompt Adapter | 20.856 ± 20.584 |
| Instruct-MusicGen | 20.012 ± 14.310 |

A clear split: the two methods that anchor to the source's latent trajectory or embedding hold tempo to ~5 BPM; the two that inject a *global semantic summary* of the source drift ~20 BPM. Standard deviations exceed or rival the means throughout, so read this as "two clusters", not "ZETA beats MusicMagus".

The architectural readings:

- **MusicMagus** — near-perfect ChromaSim and ChromaDTWS, high StructPairF, good ContourDTWS. Cross-attention consistency during sampling appears to be doing real preservation work, and inference-time intervention with no [[Fine-Tuning|fine-tuning]] avoids distorting the source.
- **ZETA** — strong on both harmony *and* beat. DDPM inversion reuses the source's inverted latent trajectory and only swaps the conditioning text, which anchors low- and mid-level content by construction. Weaker on higher-level organisation under stronger edits.
- **Audio Prompt Adapter** — ChromaSim 0.95, StructPairF 0.88, relatively high ARI, but $\Delta$BPM 20.9 and only moderate ContourDTWS. The diagnosis: global semantic audio embeddings constrain *what* the music is about but say nothing about frame-level timing.
- **Instruct-MusicGen** — harmony and structure hold, rhythm and melody degrade. Nothing in the objective constrains beat position or melodic trajectory, so [[Auto-regressive models|autoregressive]] generation accumulates timing drift.

**The interpretation problem the authors flag honestly.** MusicMagus has low $\Delta$BPM but near-zero BeatF and IG, which they suggest reflects "musically meaningful rhythmic transformations induced by style-transfer edits rather than unintended context loss". Similarly for Instruct-MusicGen: in add/remove/extract-instrument tasks, the target stem may *carry the main melody*, so a semantically correct edit necessarily changes melodic contour and motif statistics.

This is the framework's central limitation and it is unsolved. A low preservation score can mean the system broke something, or it can mean the instruction legitimately required that thing to change. MuseCPEval has no way to tell the difference, because it never knows which facets the instruction targeted. You need a human, or a per-instruction facet mask, to disambiguate.

## Worth Remembering

**What this actually is.** A curated, validated metric suite with the derivations written out. Ten metrics, all from off-the-shelf MIR tooling. The value is in facet decomposition, the folding/normalisation choices (BPM metric-octave folding, minor→relative-major normalisation, the $1/(1+\text{SKL}/50)$ squash), and the controlled MIDI validation protocol. No new modelling.

**The validation protocol is the transferable part.** Edit in the symbolic domain so exactly one facet changes by construction; derive the expected metric value from the metric's definition *before* measuring; render source and edited audio with the identical soundfont so synthesis is not a confound. That is a template for validating any metric, not just musical ones. Compare the discipline to [[Do ImageNet Classifiers Generalize to ImageNet]] or [[On Sampled Metrics for Item Recommendation (KDD)]] — same instinct: interrogate the instrument, do not just use it.

**Practical caveats if you want to use this:**

1. **Structure scores are relative only.** ARI reads 0.30 on an edit that changed no structure. Never quote an absolute value; only compare systems on identical data with identical segmentation settings.
2. **Melodic metrics are the weakest link.** Metric–Human agreement of 45.5% is below chance, and the melodic metrics respond to transposition as strongly as to melodic inversion. Do not use them as a primary criterion.
3. **Cross-system comparison is invalid as reported.** Every case study used its own paper's data and task set. A fair leaderboard would need one shared source corpus and one shared instruction set — which the paper does not provide.
4. **$\Delta$BPM is per-sample noisy.** ±8.8 BPM standard deviation on an edit with true value 0, from beat-tracker failures. Use the median, or report the full distribution.
5. **Low preservation ≠ failure.** Without knowing which facets the instruction targeted, you cannot distinguish context loss from correct editing.

**Sample sizes are small.** 50 MIDI files, 11 human participants, 4 questions per facet. 33 responses collected and 22 discarded on sanity checks — a 67% rejection rate, which is either very strict screening or a hard task. The human study is directional evidence, not a power-adequate result.

**Cross-metric structure worth internalising.** The Beat-F / IG pair is the nicest bit of metric design here. Beat F-measure is a *hard threshold* on alignment (±70 ms hit or miss); IG is a *distributional* measure of whether errors are structured. A constant offset destroys the first and survives the second. Whenever you build a matching metric with a tolerance window, ask what a systematic-but-uniform error does to it — and whether you need a second, distributional companion. This generalises well beyond music.

**The timbre appendix was reviewer-driven** ("Thanks to ISMIR reviewers' suggestions"). SKLSim separates the two instrument swaps cleanly — Grand Piano → Electric Piano reads 0.16, → Acoustic Guitar reads 0.35 — while staying ≥0.72 on all non-timbre edits. Mean MFCC cosine is much less discriminative (0.78 and 0.90 on the swaps, ≥0.97 elsewhere); the Gaussian/KL version is doing the work.

**Follow-up questions:**
- Could the structure metric be fixed by segmenting the *source-aligned* chromagram rather than raw audio, removing the spectral sensitivity?
- Would a per-instruction facet mask (parse the instruction, mark which facets are licensed to change) turn this from a diagnostic into a usable single score?
- The melodic facet needs real melody extraction, not chromagram DTW. Does a monophonic F0 tracker plus interval-level comparison fix the below-chance human agreement?
- Every metric is source-vs-output. Is there a version that catches *degradation* — artefacts, smearing — that preservation metrics are blind to because they only measure difference?

**Connections.** The metric-validation instinct here is [[Troubling Trends in Machine Learning Scholarship]] applied to MIR: the field accumulated metrics without checking whether they measure what their names claim. The per-facet decomposition mirrors how [[Evals]] argues against single aggregate scores. The IG metric is [[KL Divergence|KL to uniform]] used as a structure detector, the same trick that shows up in diversity and calibration measurement.

## Links

Related: [[Evals]] · [[KL Divergence]] · [[Dynamic Programming]] · [[Cross Attention]] · [[LoRA]] · [[Diffusion Models]] · [[Conditional Generation]] · [[Auto-regressive models]] · [[Fine-Tuning]] · [[Evaluating Generative Models]] · [[Troubling Trends in Machine Learning Scholarship]] · [[On Sampled Metrics for Item Recommendation (KDD)]] · [[Do ImageNet Classifiers Generalize to ImageNet]] · [[Controlling Diffusion]] · [[Cross Entropy]] · [[Denoising Diffusion Probabilistic Models]]

New topics worth writing: Music information retrieval metrics, Dynamic Time Warping, Constant-Q Transform and chromagrams, MFCC features, Krumhansl–Schmuckler key detection, Adjusted Rand Index, DDPM inversion for editing, Symmetrised KL between Gaussians, Music structure segmentation, Beat tracking and octave errors, Human listening study design, Held-constant-facet validation protocols
