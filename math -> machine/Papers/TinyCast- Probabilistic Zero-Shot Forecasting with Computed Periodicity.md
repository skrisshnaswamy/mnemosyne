---
title: "TinyCast: Probabilistic Zero-Shot Forecasting with Computed Periodicity"
authors: ["Armin Steinhauser"]
year: 2026
arxiv: "2608.15767"
url: https://arxiv.org/abs/2608.15767
priority: Low-Priority
read_on: 2026-10-05
tags: [paper, transformers, vision, scaling]
---
## The Core Idea

TinyCast is a time-series forecaster with **146,505 parameters** that works on series it has never seen, and emits a *distribution* (nine quantiles) rather than one number per step. It runs on a microcontroller.

The trick is a trade. At this size, every parameter you spend on *learning* that a signal repeats daily is a parameter you cannot spend on anything else. So the seasonality is **computed, not learned**: one Fourier transform per forward pass finds the dominant periods, and those periods are then used to fold the context onto its own phase. Zero parameters. The learned weights handle whatever is left.

> [!NOTE] Zero-shot forecasting
> One model is pretrained on a large corpus of many series, then forecasts a *new* series with no gradient updates and no fitting to that series. The opposite of the classical one-model-per-dataset regime. ^zero-shot-forecast

Why this did not exist before: the ingredients are old. Fisher's 1929 test for a significant spectral peak is old. Folding on a detected period is TimesNet (2023). Learning on a *known* period at tiny scale is SparseTSF (~1k parameters) and FITS (~10k). But those small models are trained on the very series they forecast, and the period is handed to them as a per-dataset hyperparameter. FlowState reads the period from *dataset metadata*. TinyCast measures the period from the series in front of it, which is the only version that works on a signal nobody labelled.

What it unlocks, concretely:

- **Probabilistic accuracy at 146K parameters.** On GIFT-Eval it scores 0.545 normalised weighted quantile loss. Every model that scores better carries at least 1.4M parameters. FlowState spends 9.1M (62×) to reach 0.502.
- **One firmware image, any signal.** 138.1 KiB of INT8 weights on an STM32H753 (Cortex-M7, no accelerator, no off-chip RAM). No host, no network, no per-signal fitting.
- A second, more general claim the author makes explicitly: *any* structure a fixed computation can supply is capacity returned to the learned parameters, and the smaller the budget, the bigger that return. Seasonality just happens to be the cheapest case.

There is a quieter architectural consequence worth keeping. The parameter budget ruled out attention (quadratic, and softmax is non-affine), FFT mixing (needs the whole window) and state-space scans (need a scan kernel integer runtimes do not ship). What is left is dilated causal convolution. That is *also* exactly the set of operators an INT8 embedded runtime can execute. The architecture forced by the budget and the architecture that runs on the device are the same architecture — no distillation step, no redesign.

## The Methodology

### Shape of one forward pass

Context $y = (y_0, \dots, y_{L-1})$ with $L = 2048$. One pass emits nine quantiles for a fixed block of $p = 48$ future steps:

$$Q^{(b)} = f_\theta(z^{(b)}) \in \mathbb{R}^{9 \times 48}, \qquad \tau \in \{0.1, 0.2, \dots, 0.9\}$$

Longer horizons chain blocks. The **median row** of a finished block is appended to the context, the last $L$ values are kept, and the model re-encodes. Horizons up to 720 steps = 15 blocks. Inside a block all 48 positions are produced in parallel; nothing is fed back within a block.

This is [[Auto-regressive models|autoregressive]] at the block level only. It also explains a calibration flaw later on — feeding the *median* forward means later blocks never learn that uncertainty should have grown.

### Normalisation

Per-window min-max, statistics detached from the gradient:

$$\hat{x}_t = \frac{y_t - y_{\min}}{\max\{y_{\max} - y_{\min},\, 10^{-5}\}}$$

Non-constant contexts land in $[0,1]$, which is a convenient range for INT8. The inverse is applied to the output. A single outlier wrecks the scale — the authors list this as a limitation.

### Periodicity detection (zero parameters)

Remove the mean, zero-pad to a power of two, take one real FFT, form the normalised periodogram:

$$I[k] = \frac{|X[k]|^2}{\sum_{k' \geq 1} |X[k']|^2}, \qquad X = \mathrm{rFFT}(\tilde{y})$$

Keep local maxima above Fisher's threshold at $\alpha = 0.05$, Bonferroni-corrected across bins:

$$t_\alpha = \frac{\ln(N_{\text{bins}}/\alpha)}{N_{\text{bins}}}$$

Take the $K = 4$ largest survivors and round their bin indices to integer periods $p_1, \dots, p_4$. A slot that fails the test is set to zero, and its phase channels go to zero too — its fold collapses to the whole-context average.

One $O(L \log L)$ transform per pass. Honest caveat the paper states itself: Fisher's test screens out *white noise*, not aperiodicity. On AR(1) surrogates with coefficient 0.5 it declares a period 98.9% of the time. So a slot can be filled from a series with no real cycle.

### Positional encoding (zero parameters)

13 channels, evaluated at *any* $t \in [0, L+H]$ — a function, not a lookup table, which is what lets future positions get coordinates.

Per period slot $k$, a phase pair:
$$\psi_k(t) = \big(\sin(2\pi t / p_k),\ \cos(2\pi t / p_k)\big)$$
(zero if the slot failed). Four slots → 8 channels.

Plus five recency channels on $\Delta(t) = (t - (L-1))/L$, the signed normalised distance from the last observation:
$$\rho(\Delta) = \big(\Delta,\ \operatorname{sign}(\Delta)\log_2(1+|\Delta|),\ e^{-|\Delta|/2},\ e^{-2|\Delta|},\ e^{-8|\Delta|}\big)$$

Same spirit as the sinusoidal encoding in [[Attention Is All You Need|the Transformer]], with one difference that is the whole point: the frequencies are set **per series** by the detected periods, not fixed globally.

### Encoder

Value channel + 13 positional channels = 14 inputs, projected to $D = 64$. Ten blocks. Block $i$ does:

1. Depthwise-separable causal conv, kernel $K_c = 3$, dilation $d_i = 2^{i-1}$ (so 1 → 512)
2. Residual add, RMSNorm
3. SwiGLU feed-forward, residual add, RMSNorm

Receptive field $1 + (K_c - 1)\sum_i d_i = 2047$, one sample short of the full context, with no downsampling. This is the WaveNet primitive in TCN residual form.

Two budget tricks remove 191,680 parameters — the same architecture without them is 338,185, more than twice the shipped size:

- time mixing is **depthwise-separable** ([[Mixed Precision training|MobileNet]]-style factorisation)
- **one** SwiGLU is ALBERT-tied across all ten blocks

Causal padding is not an accuracy choice; it is what lets the device advance the encoder one position at a time against a bounded working set.

### Decoder — three readouts

The decoder reads the encoder output $h \in \mathbb{R}^{L \times D}$ **once** and builds a query for each of the 48 future positions.

**1. Pooled summary.** $c = [\bar{h} \,\|\, h_{L-1}] \in \mathbb{R}^{128}$ — mean state and last state. Static across the horizon.

**2. Phase binning.** This is where the computed period gets spent, and the ablations say it is the biggest single lever. Assign every position to one of $n_b = 16$ bins of period $p_k$:

$$\phi_k(t) = \Big\lfloor n_b \frac{t \bmod p_k^+}{p_k^+} \Big\rfloor, \qquad p_k^+ = \max(p_k, 1)$$

Average the encoder states sharing a bin into a cycle template:

$$P^{(k)}(t) = \operatorname{mean}\{h_\tau : \tau < L,\ \phi_k(\tau) = \phi_k(t)\} \in \mathbb{R}^{16 \times 64}$$

Because $\phi_k$ depends only on $t$, a **future** position gets a bin by the same rule and reads the template at its own phase. The four slot readouts are mixed by a learned $W_{\text{phase}} \in \mathbb{R}^{64 \times 256}$ (16,448 parameters) into $s_h$.

**3. Future-conv correction.** The other two readouts hand every future position the same view of the context. This one adds local dynamics. Start with a draft in *value* space — the phase average of past normalised values at the dominant period:

$$d_h = \operatorname{mean}\{\hat{x}_\tau : \tau < L,\ \phi_1(\tau) = \phi_1(L+h)\}$$

Pair each draft with its positional encoding, project to width 64, append those 48 vectors to the last **128** encoder states, and run a six-block causal depthwise-separable net (dilations 1–32) over the joined sequence. Last 48 outputs are the correction $u_h$. Note the draft is an *input* to this network, not something it emits — a distinction the negative results make important.

The query:
$$q_h = W_q [\mathrm{PE}(L+h) \,\|\, c \,\|\, s_h] + W_{\text{fc-out}} u_h \in \mathbb{R}^{64}$$

Then residual SwiGLU + RMSNorm + a $64 \to 9$ linear head. $W_{\text{fc-out}}$ is zero-initialised, so the correction starts off and the model begins as the uncorrected map.

Budget split: encoder 57,920 (40%), future-conv 44,864 (31%), phase mix 16,448, query 13,184, decoder FFN 12,544, head 585.

### Objective

Nine-quantile pinball loss, plus one custom term. The **gated committing loss** fires only when naively repeating the last cycle would have beaten the model's median on that window:

$$g = \mathbf{1}\Big\{\textstyle\sum_h |a_h - y_h| < \sum_h |\hat{y}_{h,0.5} - y_h|\Big\}$$
$$\mathcal{L}_{\text{commit}} = \frac{g}{|\mathcal{O}|} \sum_{h} \big[|\hat{y}_{h,0.5} - y_h| - |a_h - y_h|\big]_+$$

Weight $\lambda = 0.3$. Hinged, so it goes silent the moment the median catches the copy. It only supervises the median.

### Training

| | |
|---|---|
| Optimiser | AdamW, wd 0.01, clip 1.0 |
| Peak LR | $3 \times 10^{-3}$, warmup-stable-decay (5% / 60% / 35%) |
| Samples | 150M, effective batch 4096 |
| Precision | bf16 mixed, `torch.compile` max-autotune |
| Hardware | 8× RTX 3090, DDP, 7.8 h wall clock (~62 GPU-hours) |
| Final weights | average of last eight checkpoints |

Rollout is **four blocks** under [[Imitation Learning|scheduled sampling]] — the median-feedback probability ramps 0 → 0.5, so the model trains under the conditions it will be deployed in rather than on clean teacher-forced targets.

Corpus: GIFT-Eval-Pretrain with 20 benchmark-overlapping and 66 gridded-weather directories removed, plus Chronos KernelSynth, plus four synthetic shards (GP over a 38-kernel bank, trapezoid pulse trains, trend-seasonal-impulse; mixed 70/15/15). Windows are frequency-band balanced to match GIFT-Eval's own distribution of configurations. Augmentations at $p=0.5$ each: temporal flip, sign flip, downsample by stride ∈ {2,3,4}, [[Dropout- A Simple Way to Prevent Overfitting|mixup]] with $\mathrm{Beta}(0.2,0.2)$. The sign flip is what later licenses the inference-time symmetrisation.

### Two inference strategies (host only, not on device)

- **Sign symmetrisation.** Average the forecast with the quantile-reversed forecast of the negated input, since the $\tau$-quantile of $-y$ is minus the $(1-\tau)$-quantile of $y$. Doubles the core calls.
- **Canonical-period alignment.** If a configuration's dominant peak sits at an integer multiple $k$ of the canonical samples-per-day cycle while the canonical cycle itself is absent, decimate by $k$, forecast coarse, interpolate back. Seven guard conditions.

### Quantisation

Static W8A8: symmetric per-output-channel weights, per-tensor activations, all scales frozen before compile. Not integer-only — min-max normalisation, every RMSNorm (20 in the encoder, 13 in the decoder), the SiLU gates and the FFT detector stay FP32 islands. The 32 calibration series are chosen by SHA-256 of the series identifier, reading no sample value.

## Ablation Studies and Experiments

### Benchmark setting

GIFT-Eval: 97 configurations (dataset × frequency × forecast term), 7 domains, 10 frequencies. Every score is a ratio to **seasonal naive** (repeat the value one season ago), and the 97 ratios are combined by geometric mean. So 1.0 = parity with seasonal naive, lower is better. Metrics: $\mathrm{nGMASE}$ (point), $\mathrm{nWQL}$ (probabilistic), $\mathrm{nMSIS}$ (interval).

Comparators are **re-aggregated from published per-configuration files** at a pinned benchmark commit, not copied from papers. That is worth noting — it puts everyone on the same reference and makes paired bootstraps possible.

### Headline table

| Model | Params | nGMASE | nWQL | nMSIS |
|---|---|---|---|---|
| **TinyCast** | **146 K** | 0.774 | **0.545** | 0.554 |
| Reverso-Nano | 200 K | 0.760 | (0.661) | (2.035) |
| Reverso-Small | 550 K | 0.726 | (0.626) | (1.945) |
| TTM-R3 | 1.4 M | 0.724 | 0.520 | 0.501 |
| Reverso | 2.6 M | **0.711** | (0.610) | (1.905) |
| Toto-2.0-4m | 4.1 M | 0.757 | 0.524 | **0.455** |
| YingLong-6m | 7.3 M | 0.880 | 0.609 | 0.534 |
| FlowState | 9.1 M | 0.726 | 0.502 | 0.563 |
| Kairos-10m | 9.9 M | 0.753 | 0.554 | 0.776 |
| AutoARIMA | 0 | 1.074 | 0.912 | 0.948 |
| FLAIR | 0 | 0.838 | 0.587 | 0.538 |

Parenthesised values are *point* errors — Reverso emits no distribution, so those three rows are not really probabilistic comparisons. Note that size does not order the field: 2.6M Reverso beats 7.3M YingLong by 0.169 on point accuracy.

**The honest reading.** On point accuracy TinyCast is *behind* its nearest size neighbour (0.774 vs Reverso-Nano's 0.760), and the paired bootstrap for that gap spans zero under cluster resampling. On probabilistic accuracy it is clearly ahead of both Reverso-Nano (−0.116) and Reverso-Small (−0.081), because those models do not emit a distribution at all. The real claim is narrow and the paper states it precisely: *among zero-shot entries declaring no leakage, the only one below 1.4M that emits a predictive distribution.*

Beyond the 10M census, 33 entries score better on point and 31 on probabilistic; the smallest sized one is TabPFN-TS at 11.1M (~75×).

Secondary benchmarks:

- **Chronos-ZS** (27 tasks): relative MASE 0.880, WQL 0.722. This is the *weakest* result — AutoARIMA (0.869) and AutoTheta (0.859) beat it on point accuracy, the only place across three benchmarks where a statistical method leads. Why: 14 of 27 tasks declare a period of ≤4 steps and 10 declare none, so the phase fold has nothing to fold. Split it and the story inverts — on those 14 we score 0.913 vs AutoTheta's 0.851; on the other 13 we lead at 0.846 vs 0.867.
- **fev-bench** (100 tasks): relative MASE 0.819, WQL 0.658, ahead of every statistical baseline. Every neural model ahead carries 28–62× the parameters. Disjointness is *not* established — 12 tasks name corpus subsets used in training, and we score 0.934 on those vs 0.805 on the rest.

### The detector ablation — the central experiment

Retrain at the *shipped* 146,505 budget, full 36,621 steps, detector disabled and its readout left inert:

| | nGMASE | nWQL |
|---|---|---|
| Detector off (retrained) | 0.7814 | 0.5483 |
| Shipped recipe, mean of 3 seeds | 0.7743 | 0.5441 |
| **Detector worth** | **0.0071** | **0.0042** |

Three seeds span 0.0009 nGMASE, so the point-accuracy effect is **about eight times the training noise**. On nWQL the seed spread is 0.0022, so about twice the noise. Over *configurations*, though, neither contrast separates from zero — the CI is $[-0.0004, +0.0148]$.

The authors are careful about what this does and does not measure. The retrained arm keeps its 16,448-parameter phase readout running on a degenerate input instead of reallocating those weights elsewhere. So 0.0071 is an **upper bound** on what computing the period buys relative to a model free to spend that capacity differently.

### Is it the *period*, or just the shape of the output?

Three inference-time substitutions on the trained checkpoint, all against the same control (output suppressed, which alone costs 0.0751 nGMASE):

| Intervention | nGMASE | vs suppression |
|---|---|---|
| Suppress output (control) | 0.8489 | — |
| Fixed data-blind period set | 0.8597 | **+0.0108 (worse)** |
| Detector's own outputs, wrong series | 0.8414 | −0.0074 |
| Metadata canonical period | 0.8509 | +0.0020 |

This is the sharpest result in the paper. **A well-occupied period the series does not have is worse than supplying nothing at all.** An empty slot falls back cleanly; a wrong-but-plausible slot is active harm. And the detector's own distribution applied to the *wrong* series recovers nine tenths of the value, which says a lot of the benefit is corpus-level period statistics rather than per-series correspondence — though the remaining 0.018 gap to the fixed set is the part that is genuinely about *this* series.

The canonical-period substitution is also instructive. Substituting the *true, metadata-declared* period costs 0.136 at sub-hourly/hourly but only 0.005 at daily-or-coarser (contrast 0.131, CI $[0.084, 0.180]$). Correctness cannot explain that — the canonical period is right in both groups. **Window occupancy** can: at 5-minute sampling, period 288 folds a 2048-window onto seven cycles; at daily sampling, weekly period 7 packs nearly 300 cycles per bin. What the encoder responds to is how many cycles land in each bin. The detector earns its place by picking periods that both occupy the window well *and* belong to the series.

### Architecture family (340K base, 7,500 steps)

| Arm | Params | nGMASE | Δ |
|---|---|---|---|
| Detector off, recency path substituted | 410 K | 1.1537 | +0.1312 |
| Control (dilated-conv base) | 340 K | 1.0225 | — |
| + causal padding | 340 K | 1.0161 | −0.0064 † |
| **+ phase binning** | 361 K | 0.9248 | **−0.0977** |
| + phase binning + recency gate | 393 K | 0.9092 | −0.1133 |

† interval spans zero.

Phase binning is the single largest contributor, and capacity does not explain it. The recency gate adds 33K parameters to the same base and returns only 0.0156 — **about a tenth as much per parameter**. Widening $D$ from 64 to 76 (41% more parameters) moves nGMASE by −0.0018, interval containing zero.

Phase binning's benefit also tracks how much periodic structure a configuration has: median 0.021 on the 24 configurations with no declared seasonality, 0.219 on the quarter with the most.

**Causal padding is free.** +0.0001 nGMASE with phase binning present, interval spanning zero. The streaming mode needs it, and it costs nothing.

### Component family (394K, 30,000 steps, nine quantiles)

| Arm | nGMASE | Δ | nWQL | Δ |
|---|---|---|---|---|
| Control | 0.8123 | — | 0.5699 | — |
| + future-conv | 0.7972 | −0.0151 | 0.5614 | −0.0085 |
| + synthetic blend | 0.8060 | −0.0062 † | 0.5669 | −0.0030 † |
| + gated committing loss | 0.8100 | −0.0023 † | 0.5690 | −0.0009 † |
| **+ future-conv + synthetic** | **0.7860** | **−0.0263** | **0.5529** | **−0.0170** |
| 8 AR chunks (vs 4) | 0.8132 | +0.0010 † | 0.5648 | −0.0051 † |
| Backtest-selected period | 0.8167 | +0.0044 † | 0.5749 | +0.0050 † |
| MASE-weighted loss | 0.8146 | +0.0023 † | 0.5719 | +0.0020 † |

Future-conv and synthetic data are close to additive. The committing loss ships only because it is free and its interval centres on a gain.

**A metadata-informed selection rule does not beat the raw Fisher pick.** Choosing among detected + declared canonical + weekly periods by in-context backtest error moves *both* metrics the wrong way.

**A hidden finding.** The longer rollout (8 chunks) is null on both metrics the selection used — but on the interval score it improves by 0.0297, CI $[0.0171, 0.0433]$, the only component arm resolved on nMSIS under clustering. The authors did not ship it because the selection ran on the other two metrics. That is an admitted miss.

### Optimisation

| Arm | nGMASE | Δ |
|---|---|---|
| LR $1\times10^{-3}$ | 0.8957 | +0.0395 |
| LR $2\times10^{-3}$ | 0.8665 | +0.0103 |
| **Control: $3\times10^{-3}$, 100M** | 0.8562 | — |
| LR $4\times10^{-3}$ | 0.8745 | +0.0183 |
| 50M samples | 0.8933 | +0.0371 |
| 150M samples | 0.8533 | −0.0029 † |

The chosen peak beats both neighbours — flat near its optimum. 50M → 100M buys 0.037; 100M → 150M (what shipped) is inside the noise.

### Single-setting sweeps

| Override | Δ nGMASE |
|---|---|
| Phase bins 16 → 32 | **−0.0045** |
| Cross-horizon conv width 5 | −0.0007 |
| Encoder kernel 3 → 5 | +0.0008 |
| Gated encoder conv | +0.0086 |
| 2 harmonics per period | +0.0086 |
| **Period cap 4 → 8** | **+0.0141** |
| Recency-weighted phase fold | +0.0150 |

Raising $K$ from 4 to 8 is the second-worst change in the table. The reason is in the detector statistics: a multi-slot emission carries only 2.2 *distinct* (non-harmonic) cycles on average, 79% carry a second, and only 5% reach four. There is nothing left for slots 5–8 to find.

Doubling bins to 32 is the only improvement, and the authors decline it — the run postdates the freeze and it doubles the on-device cycle template.

### What did not work

Six families of rejected intervention, and the pattern across them is the paper's most transferable lesson.

**Group 1 — supplying computed *values* instead of computed *coordinates*.** On a 6-config probe, the control (plain dilated-conv base) scored 0.8760. Feeding the seasonal-naive draft in as an input value: 0.8825. Adding all significant periods: 0.8930. Adding a linear trend term: 0.9178. Trend-seasonal decomposition as channels, on all 97: 0.9177. **The regression grows monotonically with the amount of prior supplied.**

Read with the detector result, this locates the mechanism precisely. A computed **period** is useful as a coordinate the encoder is *indexed by*. A computed **forecast** is not useful as a value the encoder must *correct*, because the convolutional stack models level and seasonality in value space better than the hand-computed baselines do. The distinction is not "give the model priors" — it is *where* in the computation the prior enters.

**Group 2 — learning the reliability rule.** The hard rule rejects periods with too few cycles in the window. Replacing it with a learned per-period weight: 0.9114 and 0.9190, both worse than the hard rule (0.9002) *and* worse than no rule at all (0.9092). Neither is in the shipped model.

There is a budget interaction here worth filing. In a separate probe, the hard rule plus a wider cross-horizon conv *helped* at reduced budget (each worth 0.009 at 30,000 steps) and *hurt* at full budget (joint cost 0.0161 at 150,000 steps). The expected direction if a hard-coded bias helps an undertrained model and constrains a well-trained one — the attribution is joint, since both flags moved together.

**Group 3 — per-instance adaptation.** Four methods tried. The best, an *oracle* gate, reaches 0.818 against the amortised model's 0.774. So the apparent [[In Context Learning|in-context-learning]] headroom is largely a measurement artefact: a per-series Bayesian fit that looks strong under a balanced interior-window probe falls behind under the exact GIFT-Eval protocol, and the oracle's per-series advantage does not correlate with any tested identifiability statistic.

**Group 4 — tail-robust objective.** Worst-$\alpha$ per-sample CVaR was worse at every $\alpha$. Diagnosis: sample-level CVaR concentrates on the noisiest, most *aleatoric* samples, not the under-served configurations. The training-sample tail does not transfer to the evaluation-configuration tail the metric scores. A clean confusion between [[Uncertainty#^aleatoric|aleatoric]] and [[Uncertainty#^epistemic|epistemic]] difficulty.

**Groups 5 and 6** — linear horizon up-weighting: no net change. Up-weighting the median term: no gain at convergence.

**Kalman-smoothed decoder** — released with the other arms but excluded from the table: at *four times* the family's sample budget it still did not reach the control.

### Deployment numbers

| Configuration | nGMASE | nWQL | nMSIS |
|---|---|---|---|
| Host (bf16, both strategies) | 0.774 | 0.545 | 0.554 |
| Quantised host (W8A8, both) | 0.790 | 0.553 | 0.563 |
| **Firmware (W8A8, neither)** | **0.833** | **0.581** | **0.624** |

Quantiser cost at the host profile: 2.14% / 1.26%. At the firmware profile, against its own matched single-pass reference: **5.18% / 3.29%** — more than double. The degradations are not additive. Composing the three penalties predicts 0.810 / 0.570; the actual is 0.833 / 0.581, leaving an **interaction residual of 0.023** $[0.015, 0.030]$ nGMASE — two fifths of the total deployment cost attributable to no single effect.

Spread matters as much as the mean. At firmware, 87 of 97 configurations degrade, median −3.21%, P90 −15.9%, worst −37.0% (`m4_hourly`, which already crosses parity at host). **Nineteen configurations that beat seasonal naive at host do not at firmware.** The device beats seasonal naive on 72 of 97.

Inference strategies, measured separately:

- Sign symmetrisation: 0.0079 nGMASE / 0.0061 nWQL. Changes all 97 configurations, resolved under cluster resampling.
- Canonical alignment: 0.0120 / 0.0112 — but only **2** of 97 configurations change, both from one base dataset (`bizitobs_l2c`). Cluster resampling cannot separate it from zero; 36% of draws return exactly zero.

### On the board

STM32H753, Cortex-M7 at 480 MHz, 2 MB flash.

| | |
|---|---|
| Core call (full 2048 re-encode + 48-step decode) | **4.08 s** |
| Per-position encoder step | 1.86 ms |
| Decoder alone | 0.20 s |
| INT8 coefficients | 138.1 KiB |
| Full image (incl. 8 KiB test context) | 365.5 KiB (17.8% of flash) |
| Peak RAM | 730.7 KiB |
| Activation arena | 310.1 KiB |
| Throughput | ~89M MACs/s, one per 5.4 cycles |
| Energy (datasheet-derived) | 1.5–2.0 J per forecast |

Latency is essentially input-independent — 9.9 ms spread across 32 contexts on a 4.06 s mean, with the variation tracking active period slots at 1.7 ms each. That is what lets an embedded scheduler reserve a slot.

Three independent INT8 backends (CMSIS-NN 4.08 s, SMLAD intrinsics 4.49 s, portable scalar 6.05 s — a 1.49× span) produce **identical** outputs. Period detection agrees with the host on every one of the 32 fidelity contexts. 55.4% of the 13,824 outputs are bit-identical to host.

Code layout alone moved latency by 1.9% between builds, which is why they pin 32-byte function alignment.

## Worth Remembering

**The calibration problem is real and the authors say so.** Pooled empirical coverage of the nominal 80% interval is **68.0%** — over-confident. And it degrades with horizon: 83.0% short-term, 67.4% medium, 60.0% long. This is exactly what block-autoregressive median feedback predicts — every block after the first conditions on its predecessors' medians, so the intervals never widen with accumulated uncertainty. Restricting to the 14 base datasets appearing at all three terms *widens* the gradient (23.0 → 25.9 points), so it is horizon, not composition.

The obvious fix — feed a *sampled* quantile forward instead of the median, free — recovers under a third of the gap. Pooled coverage 68.0% → 71.7%, long-horizon 60.0% → 66.3%, intervals 16% wider. It improves nMSIS by 0.087 on the 44 configurations that actually recurse (53 have horizons ≤ one block and are bit-identical). Two thirds of the deficit survives it. Whether the residual is a too-narrow quantile head or the rollout itself is *not separated* — doing so needs $S$ independent sampled rollouts, which costs $S\times$ the inference and leaves the device envelope.

**Nine quantiles caps you at 80%.** No 95% or 99% alarm threshold is available directly. For a model whose selling point is feeding control loops and alarm thresholds, that is a real constraint. The nMSIS column is also scored at $\alpha = 0.05$, so the 2.5/97.5 levels come from GluonTS's tail extrapolation — no decile-emitting model actually predicts them.

**Quantile crossing is much worse after quantisation.** On the fixed test input, 21 of 48 horizons contain an adjacent-quantile inversion unquantised; **42 of 48** under exact W8A8, because quantisation can collapse neighbouring quantiles onto the same integer level. Both paths apply a monotone sort at the end (Chernozhukov's rearrangement), so the scored object matches the emitted one — but the raw head is unconstrained, and the *median* fed to the next block is taken before sorting.

**Cold start degrades gracefully, which is the operationally important fact.**

| Observed samples | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| Relative MAE | 1.013 | 0.975 | 0.911 | 0.845 | 0.786 | 0.759 |

A fresh unit sits at parity with seasonal naive, not at failure. By 512 samples it has recovered two thirds of the distance to full-context accuracy. Detector firing rate goes 25% at 64 samples → 81% at 512 → flat at ~84%. But accuracy flattens one step *later* (512→1024 still worth 0.058), so the last of the gain is not detector availability.

**Where the losses cluster.** 90 of 97 GIFT-Eval wins, but: weekly/monthly/annual score 0.873 vs 0.758 elsewhere; 10-second data at long horizons scores 1.083. On Chronos-ZS the losses are near-random-walk (`exchange_rate` 1.085) and intermittent retail (`dominick` 1.506). Wins need roughly 500 regular samples *and* a cycle the detector retains.

**A result that cuts against the paper's own narrative, stated plainly.** Within the architecture family, phase binning's benefit tracks how seasonal a configuration is — as the premise predicts. *Between* models it does not. Against Reverso-Nano, TinyCast wins a **larger** share of configurations the benchmark declares aseasonal (11 of 24) than seasonal ones (19 of 73). The author's reading: the premise is about **capacity**, not about seasonality. A returned parameter budget gets spent on whatever is actually in front of the model.

**Firing rate vs accuracy, with the control applied.** 0.753 nGMASE on the 71 configurations firing >90%, 0.841 on the 10 firing <50%. Seven of those ten have no declared seasonality, so the split might just mean "aseasonal is hard" — except declared seasonality has rank correlation only +0.089 with error, and the 24 no-cycle configurations score 0.766 vs 0.776 for the other 73. Restricting to the 73 declared-cycle configurations leaves the gap mostly intact at +0.074 $[+0.024, +0.125]$. But that low bucket rests on **three** configurations and admits only ten distinct bootstrap resamples. Across all 97 the correlation is −0.110 $[-0.290, +0.083]$ — the direction the premise predicts, living at the tail rather than across the distribution.

### Statistical hygiene worth copying

The paper is unusually careful, and the methodology is more reusable than the model.

- **Two bootstrap schemes.** Resampling the 97 configurations, and *clustering* on the 28 base datasets — because the three forecast terms of one dataset-frequency are the same series at different horizons. Clustering widens intervals by a median factor of 1.41.
- **Benjamini–Hochberg** at $q = 0.05$ over the 36-delta ablation family. 18 of 36 intervals span zero before correction. At the strictest reading (cluster interval *and* cluster correction), **14 deltas survive** — seven arms on both metrics. Detector removal and phase binning are among them; **future-conv is not**, losing one metric or the other under every combination.
- The correction covers the ablation tables and *nothing else*. Comparator differences, substitution arms and subgroup splits each carry one uncorrected 95% interval.
- Of the 84 comparator deltas with per-configuration records, **15 span zero under clustering** — including 6 of the 33 differences in the headline table.
- Three seeds (42/43/44) measure training variance directly. Autotuned kernels and distributed reduction order leave reruns free to differ; a strict-FP32 CPU rerun differs from the bf16 record by ~$10^{-3}$ on two configurations. The firmware is the exception — integer arithmetic is exact across backends and boots.

### Honest limitations the authors list

- Univariate. Reads no covariates (46 of 100 fev-bench tasks supply them) and no cross-series structure (35 are multivariate). Removing both handicaps on the 28 univariate covariate-free fev-bench tasks *widens* the gap to CITRAS-FM (0.120 vs 0.113 overall), which is evidence against the missing-inputs explanation.
- **Silent degradation out of distribution.** No signal when the input leaves the pretraining regime.
- The period is a rounded FFT bin, so resolution falls with the window/period ratio.
- Per-window min-max is sensitive to one extreme value.
- Full-window re-encode on every call — 4.08 s. A streaming variant is projected at 1.86 ms per sample plus 0.27 s at emit (~15× cheaper), but **nothing was linked or timed**, and in a reduced-budget ablation the streaming variant *cost* accuracy, concentrated in short-context configurations.
- Ablation families run at 340K–445K parameters, not the deployed 146,505. Their deltas **bound** what a component buys at the shipped configuration rather than measuring it.
- GIFT-Eval probes informed the architecture, objective and inference choices. Chronos-ZS is the only untouched test of the process — and it is the weakest result.
- Chronos-ZS calibration contamination: Dominick is both a calibration source for the quantiser *and* a Chronos-ZS task, where it is the worst result at 1.506.

### Connections and follow-ups

The per-dataset supervised baselines are worth a glance, since the whole premise is to displace them. TinyCast (0.774 / 0.545 / 0.554) leads all eight in the pinned snapshot, including [[Attention Is All You Need|Transformer]]-based PatchTST (0.849 / 0.587) and iTransformer, and DLinear at 1.061 — worse than seasonal naive. One trained-per-series model, xLSTM-Mixer, reports 0.510 probabilistic, ahead of us.

Concurrent work adds a quantile head to the 2.6M Reverso backbone and reaches 0.499 nWQL at ~3M parameters. That is 20× TinyCast's budget and above the 1.4M frontier line, so the claim survives — but it shows how thin the moat is. The frontier claim is a statement about a *gap in the released field*, not a law.

Questions the paper leaves open:

- Does the "compute what a fixed rule can supply" trade generalise beyond periodicity? The author expects it to and offers no evidence.
- Can the quantile head be trained wide enough to fix calibration without the $S\times$ sampling cost?
- The 8-chunk rollout improves nMSIS by 0.0297 and was discarded on the wrong metrics. Cheapest available win.
- 32 phase bins is the only positive single-setting override and was declined on freeze timing plus device memory. Worth re-running.

### Practical caveats if you wanted to use this

1. **Score the configuration you will deploy, not the one in the headline table.** The 0.774/0.545 figures carry two host-side strategies the device does not run. The device gets 0.833/0.581, and the three penalties interact rather than add.
2. The float simulation in the released `quant.py` reads each tensor's own range and **does not reproduce** the frozen-scale W8A8 rows. Do not use it to predict device accuracy.
3. Calibration series are selected by hash of identifier, which reads no values — a good pattern, but check for overlap with your evaluation set anyway, as Dominick shows.
4. Alignment fires on one base dataset out of 28. Treat the 0.012 it contributes as not generalising.
5. Flash headroom is comfortable (17.8%) but RAM is not: 730.7 KiB peak against the device's 1 MB. The encoder alone ports to a 150 MHz Cortex-M33 at 7.73 ms/position in ~230 KiB; the full model does **not** fit that part's 520 KB SRAM.
6. The header claim is "146,505 parameters", but MACs tell a different story: $3.61 \times 10^8$ per call, **2,465 MACs per parameter per call**, because two tied SwiGLUs are evaluated at every one of 2048 positions. Parameter count sets storage, not arithmetic. For this model they diverge sharply.

## Links

Related: [[Auto-regressive models]] · [[Attention Is All You Need]] · [[Uncertainty]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Dropout- A Simple Way to Prevent Overfitting]] · [[Fourier Series Decomposition]] · [[Quantization]] · [[Mixed Precision Training]] · [[Layer Normalization]] · [[Gated Activation]] · [[In Context Learning]] · [[Imitation Learning]] · [[Regularization]] · [[Distillation]] · [[Evaluating Generative Models]] · [[On the Difficulty of Evaluating Baselines]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Linear Attention]] · [[State-Space Model]] · [[Group Normalization]] · [[Cyclical Learning Rates for Training Neural Networks]]

New topics worth writing: Time series foundation models, Pinball / quantile loss, Fisher's test for harmonic analysis, Periodogram and spectral peak detection, Dilated causal convolution (WaveNet / TCN), Depthwise-separable convolution, Reversible instance normalisation (RevIN), MASE and scaled forecast error metrics, Weighted quantile loss (WQL/CRPS), Mean scaled interval score (MSIS), Forecast calibration and empirical coverage, Quantile crossing and monotone rearrangement, Seasonal naive baseline, STL decomposition, Static post-training quantisation (W8A8), TinyML and embedded inference, Benjamini–Hochberg multiple-testing correction, Cluster bootstrap, Scheduled sampling, ALBERT-style parameter tying, CVaR objectives, GIFT-Eval benchmark
