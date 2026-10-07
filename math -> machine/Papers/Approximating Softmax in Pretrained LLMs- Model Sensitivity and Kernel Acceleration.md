---
title: "Approximating Softmax in Pretrained LLMs: Model Sensitivity and Kernel Acceleration"
authors: ["Shangzhen Zhu", "Muyan Hu", "Tomasz Kozlowski"]
year: 2026
arxiv: "2609.33586"
url: https://arxiv.org/abs/2609.33586
priority: Good-To-Read
read_on: 2026-09-30
tags: [paper, transformers, llm, vision]
---
## The Core Idea

Attention needs an exponential for every score. On an NVIDIA B200 GPU, the matrix-multiply hardware does 8192 operations per clock per streaming multiprocessor, but the special-function unit that computes $\exp$ does only **16**. That is a gap of more than 500×. Matrix multiplies got fast; the exponential did not. So in a fused kernel like FlashAttention-4, the exponential is now sitting on the critical path.

The obvious reaction is "make $\exp$ cheaper but still accurate". The usual bar for that is: keep the error below the storage precision, so nobody can tell. FA4 already does this — it swaps some `exp2` calls for a cubic polynomial with max relative error $8.8\times10^{-5}$.

This paper asks a different question: **how accurate does the exponential actually need to be for a frozen pretrained model to keep working?** Not "how close can we get cheaply", but "how far can we go before the model's loss moves".

The answer is: very far. You can round each attention weight to one of two values per octave — powers of two and 1.5× powers of two — and perplexity goes up by about 0.1–0.5%. Per-element error is about a third of an octave, thousands of times worse than a BF16 rounding error. The model does not care.

But the paper's real contribution is the *shape* of what the model cares about, found by a set of controlled interventions on ten frozen models (0.5B–72B). Three findings, and they are not obvious:

1. **How many positions get weight, and how finely, can be cut a lot. But the *relative* ordering of those weights is sacred.** Keep the top half of each attention row with its softmax weights: perplexity +0.131%. Keep the *same* positions but give them all equal weight: perplexity ×4155. Support is cheap. Relative weighting is not.

2. **Where you spend your resolution matters as much as how much you have.** Give the same number of quantisation bins, but make them narrow near the row maximum and wide in the tail, and loss drops — *even though the average approximation error goes up*. In Qwen2.5-1.5B this raised unweighted RMS log-weight error by 27.9% while cutting $\Delta\mathrm{NLL}$ by 71.7%.

3. **A scalar distortion number does not tell you the damage.** Two perturbations with identical mean Jensen–Shannon divergence on the attention distribution produce different model losses — and the *sign* of the difference flips across models. Flattening the attention hurts more than sharpening it in seven models, less in two.

Why did this not exist before? Because everyone approximating softmax justified it with a pointwise error budget, so nobody went coarse enough to find the edge. And the quantisation work that did go coarse (FQ-ViT, RepQ-ViT) came from vision transformers, where the grid is anchored to an absolute scale ($p=1$), not to each row's maximum — so its phase relative to the row max drifts row to row, wasting resolution exactly where the model needs it.

What it unlocks: **Rowmax-H15**, a coarse exponential that is exactly 3 instructions instead of 8, anchored on the running row maximum that online softmax is already tracking, with *no* calibrated parameter. On B200 the FP8 attention forward gets 12.4% faster at causal 8K, 25.8% at non-causal 8K, and uses 8.4% less board energy per forward.

> [!NOTE] Row maximum anchor
> Softmax is computed as $p_j = \exp(s_j - s_{\max}) / \sum_k \exp(s_k - s_{\max})$ for numerical stability, so the fused kernel already tracks a running row maximum. Anchoring a quantisation lattice to *that* value — rather than an absolute scale — means the largest weight in every row always lands exactly on a lattice point, for free. ^rowmax-anchor

---

## The Methodology

### The thing being replaced

For one query row with valid keys $\mathcal{V}$ and scores $s_j$:

$$\Delta_j = \max_{k\in\mathcal{V}} s_k - s_j, \qquad p_j = \frac{\exp(-\Delta_j)}{\sum_{k\in\mathcal{V}}\exp(-\Delta_k)}$$

Everything else in the model stays frozen. Only the map from a row of scores to normalised weights changes, in every head and layer.

A useful piece of algebra sets up the whole paper. If the approximate weight is $\hat w_j$ with log error $\epsilon_j = \log \hat w_j - \log w_j$, then

$$\hat p_j - p_j \approx p_j\left(\epsilon_j - \mathbb{E}_p[\epsilon]\right)$$

to leading order. A *constant* multiplicative error cancels under normalisation. Only the *variation* of $\epsilon$ across a row matters, and it is weighted by $p_j$. This is why probability-weighted error is the right summary, and why error in the tail is nearly free.

### The six intervention axes

Each experiment changes one and freezes the rest:

| Axis | Intervention |
|---|---|
| Support | top-$k$, or keep scores above the row mean |
| Within-support weighting | softmax weights vs uniform weights, same positions |
| Number of bins $K$ | full-range grid, exponential vs linear weight map |
| Allocation $R$ | where the bins go within the row |
| Reconstruction | upper edge / nearest boundary / interpolation |
| Exponential evaluation | Rowmax-PoT, Rowmax-H15 |

### The grid

Take the row's scores, shift so the minimum is zero: $u_j = s_j - \min_k s_k \in [0, C]$ with $C = \max s - \min s$. Cut $[0,C]$ into $K$ intervals with boundaries

$$e_a = C\,\frac{1-\rho^a}{1-\rho^K}, \qquad \rho = R^{-1/(K-1)}$$

$R$ is the ratio of the lowest-score interval width to the width at the row maximum. $R=1$ gives uniform bins. $R>1$ gives fine bins near the row max and coarse bins in the tail. The grid is recomputed **per row**.

### The same-support control

This is the cleanest experiment in the paper, and worth copying. Comparing softmax weights vs uniform weights in *every* layer is confounded: after layer 1 the two runs have different activations, so they no longer see the same support.

The fix: for each target layer $\ell$ of Qwen2.5-1.5B, run softmax below $\ell$. Both conditions therefore arrive at $\ell$ with the identical hidden state, identical scores, identical support. At $\ell$ only, one uses softmax weights on that support and the other uses uniform. Above $\ell$, both return to softmax.

$$D_W(\ell) = \mathbb{E}_b\left[\Delta\mathrm{NLL}_{b,\text{unif},\ell} - \Delta\mathrm{NLL}_{b,\text{rel},\ell}\right]$$

### Matched-distortion probes

To test whether a scalar distortion measure predicts damage, they apply a temperature to the scores: $p_{\alpha,j} \propto \exp(\alpha s_j)$. Flattening is $\alpha<1$, sharpening is $\alpha>1$. Each $\alpha$ is found by bisection so that the mean row-wise JSD against softmax hits a target — **without ever looking at NLL**. Targets are $J_0$ (the JSD produced by a plain octave power-of-two rounding in that model) and $10J_0$.

$$D_T(J) = \mathbb{E}_b\left[\Delta\mathrm{NLL}_{b,A^-(J)} - \Delta\mathrm{NLL}_{b,A^+(J)}\right]$$

$D_T > 0$ means flattening is worse. If a scalar JSD were sufficient, $D_T$ would be zero.

### Rowmax-PoT, then Rowmax-H15

**Rowmax-PoT** is the simple version: round the base-two log weight, measured from the row max, to the nearest integer.

$$\hat p_j \propto 2^{-d_j}, \qquad d_j = \min\left\{K_{\max},\ \left\lfloor \frac{\Delta_j}{\ln 2} + \tfrac{1}{2}\right\rfloor\right\}$$

One value per octave, tail clamped at $K_{\max}=20$.

**Rowmax-H15** halves that spacing and is what goes in the kernel. Given $x$, the base-two exponent FA4's online path already computes:

$$n = \operatorname{round}_{\mathrm{RNE}}(2x), \qquad \hat w(x) = \left(1 + \tfrac{1}{2}(n \bmod 2)\right)2^{\lfloor n/2\rfloor}$$

Even $n$ gives $2^k$; odd $n$ gives $1.5\cdot 2^k$. Two values per octave. Note the asymmetry: the *input* is rounded in half-octave steps, but the two *outputs* per octave are 0.585 and 0.415 octaves apart, because $\log_2 1.5 \approx 0.585$.

> [!NOTE] Rowmax-H15
> Approximate $2^x$ by the nearest member of $\{1, 1.5\}\times 2^k$. Anchored at the attention row's running maximum, so no scale is ever calibrated. Per-element $\log_2$ error lies in $[-0.25, 0.335]$ octaves — vastly larger than one BF16 ulp, which is why it must be validated by model loss rather than pointwise error. ^rowmax-h15

**The implementation is three instructions.** Add `0x4B400000` (that is $2^{23}+2^{22}$, FA4's own floor-construction constant, but with round-to-nearest-even) to the doubled FP32 input — this parks $n$ in the low mantissa bits. Shift left 22, which moves bit 0 of $n$ into the leading mantissa position and the rest into the exponent field. Add the bit pattern of 1.0 for the bias. Done.

A static SASS audit on isolated `sm_86` micro-kernels counts **3 instructions, dependency depth 3** for Rowmax-H15 against **8 instructions, depth 8** for FA4's cubic emulation.

This turns out to be a coarse specialisation of Schraudolph's 1999 bit-hack for $\exp$. Writing $S(x) = 2^{\lfloor x\rfloor}(1 + x - \lfloor x\rfloor)$ and $Q_{1/2}(x) = \tfrac{1}{2}\operatorname{round}_{\mathrm{RNE}}(2x)$, the ideal operator is exactly $H(x) = S(Q_{1/2}(x))$.

### Everything else in the kernel stays

Running maximum, rescaling, log-sum-exp, pipeline structure, warp roles, FP8 scaling of $P$ — all unchanged. The row sum and the $PV$ accumulator consume the *same* approximated weights, which matters: numerator and denominator must agree or the normalisation breaks.

One subtlety. The separate exponential-path constants ($2\cdot\texttt{scale\_log2}$, $2\cdot\texttt{max\_offset}$) absorb the doubled input. Doubling the *shared* constants instead would square the rescale factor, halve the rescale threshold, corrupt the log-sum-exp and trip the FP8 range assertion.

### Evaluation

WikiText-103 test split, 97 aligned blocks of 2048 tokens (198,559 predictions). $\Delta\mathrm{NLL}$ against each model's own softmax baseline. Perplexity change is $\exp(\Delta\mathrm{NLL})-1$. All intervals are 5,000-replicate paired percentile bootstraps over blocks, seed 0. A contrast is **resolved** when its 95% interval excludes zero.

Ten decoder-only models: Qwen2.5-0.5B/1.5B/3B/72B, Llama-3.2-1B/3B, Llama-3.1-8B/70B, Gemma-2-2B, Mistral-7B-v0.3. BF16 weights, FP32 scores and score-to-probability. Qwen2.5-1.5B gets the detailed sweeps.

Kernel work: five separate rented B200 instances, FA4 commit `0dc2cb48`. Latency is 50 warm-up calls, then CUDA-event-timed windows of 100 back-to-back calls; speed-up is the ratio of pooled medians.

---

## Ablation Studies and Experiments

### Support can go, weighting cannot

Qwen2.5-1.5B, keeping top-$k$ per row with softmax weights:

| Fraction kept | PPL increase |
|---|---|
| 50% | 0.131% |
| 25% | 0.828% |
| 15% | 3.092% |
| 10% | 7.821% |
| 6.5% | 15.185% |
| 2.6% | 37.861% |
| 1% | 90.478% |
| 0.65% | 144.670% |

Graceful until about 15%, then it falls off a cliff.

Now the same supports with **uniform** weights: perplexity multiplied by **5268** (half kept) and **4155** (mean threshold). That is not a degradation, that is destruction.

The strict single-layer control confirms it is the weighting and not a trajectory artefact. $D_W(\ell)$ is positive at **all 28 layers**, all 28 pointwise 95% intervals excluding zero. Values run 0.0181 to 0.2000 for layers 1–27, with layer 0 at **8.1733** — the first layer is catastrophically sensitive.

The linear level-to-weight map (weight proportional to bin index rather than $\exp$ of bin edge) is also a disaster: paired linear-minus-exponential contrast of $8.33$ $[8.15, 8.51]$ at $K=32$. Across six models the linear damage at $K=16$ ranges 4.39 to 8.86.

**Read:** the model needs approximately-exponential relative weights on the positions it attends to. It does not need many positions, and it does not need them precise.

### Resolution is cheap

Full support, $R=1$, upper-edge reconstruction, Qwen2.5-1.5B: $\Delta\mathrm{NLL}$ falls $0.0617 \to 0.00981 \to 0.00156$ as $K$ goes $16 \to 32 \to 64$.

With nearest-boundary reconstruction and $R=4$, the common $K=32$ point stays **below 1% perplexity increase in all ten models**. Only Gemma-2-2B's interval spans 1% (point estimate 0.852%). Best is Mistral-7B-v0.3 at 0.052%.

### Allocation: the counter-intuitive one

Qwen2.5-1.5B, $K=21$ fixed, sweeping $R$:

| $R$ | $\Delta\mathrm{NLL}$ |
|---|---|
| 0.25 | 0.0161 |
| 0.5 | 0.00997 |
| 1 | 0.00489 |
| 2 | 0.00251 |
| 4 | 0.00138 |
| 8 | 0.00100 |

Monotone. Anti-allocation ($R<1$, coarse near the max) is worse than uniform; fine-near-the-max is better.

Now the interesting bit. From $R=1$ to $R=4$:

- unweighted RMS log-weight error: $0.314 \to 0.401$, **up 27.9%**
- probability-weighted RMS error: $0.178 \to 0.126$, **down 29.2%**
- $\Delta\mathrm{NLL}$: **down 71.7%**

The representation got *less accurate on average* and *better for the model*. Which follows directly from $\hat p_j - p_j \approx p_j(\epsilon_j - \mathbb{E}_p[\epsilon])$ — error in low-$p$ entries is multiplied by a small $p_j$.

Why: attention mass is extremely concentrated. On Qwen2.5-1.5B's softmax trajectory, entries with $p\geq 0.1$ are **0.151%** of valid entries and carry **53.8%** of the mass. Entries with $p < 0.0001$ are **75.1%** of entries and carry **1.2%**.

$D_R(21) = \Delta\mathrm{NLL}_{R=1} - \Delta\mathrm{NLL}_{R=4}$ is positive in **all ten** point estimates, with nine of ten intervals excluding zero. Gemma-2-2B is the exception: $+0.00527$ $[-0.00369, +0.0132]$.

The interaction $I_R = D_R(16) - D_R(32)$ is resolved positive in all four Qwen2.5 sizes and Llama-3.1-70B — allocation matters more when you have fewer bins, which is what you would expect.

### Equal distortion, unequal damage

Over 56 interventions on Qwen2.5-1.5B, $\log\Delta\mathrm{NLL}$ against $\log$ mean row-wise JSD fits with slope 1.05 and $R^2 = 0.945$. So JSD *tracks* damage well.

But it does not *determine* it. At matched $J_0$:

- $D_T$ resolved **positive** (flattening worse) in seven models
- resolved **negative** (sharpening worse) in Llama-3.2-1B, $-0.00313$ $[-0.00428, -0.00202]$, and Qwen2.5-72B, $-0.00119$ $[-0.00208, -0.000375]$
- unresolved in Llama-3.1-70B

The Llama-3.2-1B reversal survives matching on total variation instead of JSD. The pattern does not follow model size — 1B and 72B flip the same way, 3B and 8B do not.

### Three negative controls that rule out the easy explanations

**Leakage.** In all ten models, matched flattening moves mass *out* of the set carrying 90% of softmax mass and matched sharpening moves it *in*. Including Llama-3.2-1B, whose loss contrast is reversed. So the direction of probability redistribution is shared and cannot explain the split.

**Local attention output.** The RMS change in $pV$ is almost identical for the two directions (Qwen2.5-1.5B 0.0733 vs 0.0732; Llama-3.2-1B 0.0790 vs 0.0798). Median absolute difference between directions is 0.94% of the smaller value. And the direction with the larger RMS is sharpening in five models, flattening in five — while $D_T$ is positive in seven and negative in three. **The local output norm does not predict the sign.**

**Mass restoration.** Rescale the high-mass set and its complement back to their softmax masses, keeping the perturbed conditional shapes. $G_{\mathrm{gap}} = D_{\mathrm{orig}} - D_{\mathrm{rescue}}$ is above zero in all ten models at $10J_0$. But a positive signed change is not a smaller $|D|$: Qwen2.5-1.5B moves from $+0.0133$ to a **resolved negative** $-0.00278$ $[-0.00510, -0.000565]$, and Llama-3.2-1B and Qwen2.5-72B go from unresolved to resolved negative. Restoring the mass split does not fix things; it just shuffles which way the contrast points.

### Anchor vs resolution, separated

The cleanest justification for the row-max anchor. Qwen2.5-1.5B:

| Change | Reduction in $\Delta\mathrm{NLL}$ |
|---|---|
| anchor: absolute → rowmax, octave spacing | 0.00355 [0.00268, 0.00448] |
| anchor: absolute → rowmax, half-octave | 0.00122 [0.000708, 0.00171] |
| spacing: octave → half-octave, absolute anchor | 0.00443 [0.00367, 0.00517] |
| spacing: octave → half-octave, rowmax anchor | 0.00210 [0.00132, 0.00286] |

The two effects are real, comparable, and **not additive** — the anchor gain shrinks once spacing is already fine. An 8-bit $\log_2$-probability grid (FQ-ViT's structure) costs $2.17\times$ what rowmax-anchored octave PoT costs.

The mechanism story: leakage correlates with lattice phase under the absolute lattices (Pearson $r = -0.775$, $-0.777$) but **not** under the rowmax octave lattice ($r = -0.005$). Consistent with a phase effect, though the authors are careful to say it does not demonstrate one.

### Reconstruction

At $K=21, R=4$:

| Rule | PPL increase |
|---|---|
| upper edge | 0.726% |
| nearest boundary | 0.138% |
| weight-domain interpolation | 0.00808% |

Interpolation is 90× better than nearest — but it needs the position *within* the interval, so it does not produce finitely many outputs and is useless as a quantiser. It is a diagnostic only.

### The kernel: fidelity

Direct measurement on the BF16 path, 97 blocks, five models:

| Model | $\Delta$NLL [95% CI] | PPL change |
|---|---|---|
| Mistral-7B-v0.3 | 0.000914 [0.000600, 0.00125] | 0.091% |
| Qwen2.5-72B | 0.00101 [0.000665, 0.00139] | 0.101% |
| Qwen2.5-1.5B | 0.00121 [0.000862, 0.00156] | 0.121% |
| Llama-3.1-8B | 0.00246 [0.00206, 0.00289] | 0.246% |
| Llama-3.1-70B | 0.00491 [0.00402, 0.00582] | 0.492% |

All five intervals above zero — the loss is small but **real and measurable**, not noise. No trend with size.

### The kernel: speed

FP8 causal, head dim 128, call latency:

| Sequence | Speed-up |
|---|---|
| 1K | −3.6% (retest −2.2%) |
| 2K | −2.6% (retest −3.7%) |
| 4K | +7.1% |
| 8K | +12.4% |
| 16K | +12.5% |
| 32K | +9.9% |

dtype × mask at 8K causal: BF16 **+5.4%**, FP16 **+5.3%**, FP8 **+12.4%**. BF16 and FP16 share a tensor-core rate and give the same gain; FP8 has twice the rate and roughly twice the gain. That pattern is the strongest available evidence for the mechanism — the exponential becomes exposed as the matmul gets faster.

Drop the causal mask at FP8 8K and the gain roughly doubles again to **+25.8%**. Consistent with softmax work no longer being halved by masking.

Energy: 742.5 → 680.3 mJ per forward (−8.4%) at essentially unchanged mean board power (982.3 vs 982.9 W). **The saving is entirely runtime, not low-power arithmetic.**

### What did not work

**The FP8 fidelity measurement.** A direct FP8 run was unusable — per-tensor E4M3 conversion of the BF16 activations added about **0.49 nats** of unrelated error, swamping the ~0.001 nat effect they were trying to measure. The FP8 number in the paper is a PyTorch semantic simulation on one model, and the authors flag this repeatedly.

**The pre-registered BF16 prediction.** They sealed six SHA-256 predictions before any B200 run. Prediction 5 said the BF16 whole-forward gain would stay below 3%. It came in at +3.8%, +5.4%, +7.1% at 4K–16K. Reported as failed.

**Counter-level profiling.** `ncu` was refused with `ERR_NVGPUCTRPERM` on all three instances where it was tried, across two driver versions. Predictions 1–3 and 6 (MUFU utilisation near zero, FP-pipeline utilisation up, softmax-stage time down 35–50%, registers per thread not increased) need counters and **remain unverified**. Every mechanism claim in the paper is therefore inferential.

**Short sequences.** Call latency is *negative* at 2K in both sessions, and the 1K sign flips between sessions (−2.2% vs +7.2%). The traced kernel is faster at every length, so the kernel itself is never slower — but at 1K–2K the non-kernel share of the timed interval is 60.9% and 28.5%, and launch/dispatch overhead eats the win. At 2K, kernel-only is +9.6% while call latency is −4.7%.

**Head dimension 64.** Gains collapse to +2.0% to +3.7% at FP8. But this is not a fair comparison — FA4's tuning table has no head-dim-64 entry, so stock FA4 falls back to a default emulation setting.

**Half-octave input rounding is where the loss actually lives.** The offline ablation is the sharpest result in the appendix. Compared against unquantised-input Schraudolph $S(x)$:

| Offline operator | $\Delta$NLL vs softmax | vs Rowmax-H15 |
|---|---|---|
| Rowmax-H15, $k=1$ | +0.00105 | reference |
| $S(Q_{2^{-2}}), k=2$ | +0.000329 | −0.000721 [−0.00106, −0.000393] |
| $S(Q_{2^{-3}}), k=3$ | +0.0000796 | −0.000970 [−0.00129, −0.000649] |
| $S(x)$ unquantised | −0.0000231 | −0.00107 [−0.00140, −0.000747] |

$S(x)$ shows **no resolved change against softmax at all** and beats Rowmax-H15 on 70 of 97 blocks. The entire model-level cost of Rowmax-H15 is the half-octave *input* rounding: $0.00107$ $[0.000747, 0.00140]$. Finer input quantisation ($k=2,3$) recovers most of it — but those variants were never built into FA4, so their speed is unknown.

**Conditional rescaling moves the anchor.** FA4 skips the online rescale while the row maximum grows by less than $\tau=8.0$. So the row's true largest weight can drift off the lattice — in 39.5% of synthetic rows under BF16 semantics. The point estimate is $1.50\times$ larger under those semantics, though the paired difference includes zero.

**FA4's own tuning sweep has a non-monotone surprise.** Sweeping $\beta$, the fraction of elements using polynomial emulation instead of the hardware exponential:

| Setting | $\beta$ | vs stock |
|---|---|---|
| freq 0 (hardware only) | 0% | +1.4% |
| freq 16/start 1 | 12.5% | +1.6% |
| freq 8/start 1 (stock) | 25% | 0.0% |
| freq 8/start 0 | 37.5% | **+2.7%** |
| freq 2/start 1 | 50% | +0.2% |
| freq 4/start 1 | 50% | −1.2% |
| freq 2/start 0 | 75% | −23.7% |
| freq 4/start 0 | 75% | −24.5% |

At 75% emulation performance falls off a cliff — likely register spilling, though unconfirmed without counters. And the 12.5% setting beats stock 25%, which runs opposite to FA4's own source comments.

Rowmax-H15 is +12.3% against stock default and **+9.3% against the fastest stock setting tested**. But note: that comparison crosses sessions.

### Zero-shot and long context

$K=32, R=4$ on the two 70B-class models across six lm-eval-harness tasks: mean absolute change **0.114 percentage points**, max 0.256.

The negative control is stark. Mean-threshold support raises LAMBADA perplexity by **6.727%** (Qwen2.5-72B) and **11.649%** (Llama-3.1-70B), and Llama-3.1-70B drops 3.11 points on HellaSwag and 4.18 on WinoGrande.

Long context: $D_R$ is resolved positive at 2K, 8K and 16K in both large models, and $R=4$ removes **64.9–78.1%** of the $R=1$ degradation. $D_T$ is far weaker — resolved only in Qwen2.5-72B at 2K and 8K, and **negative** there. Allocation transfers to long context; the directional effect does not.

---

## Worth Remembering

**The transferable lesson is methodological.** If you are approximating a component inside a frozen model, a pointwise error budget is the wrong validation. Rowmax-H15's per-element error is in $[-0.25, 0.335]$ octaves — thousands of BF16 ulps — and it costs 0.1% perplexity. Meanwhile uniform weights on the correct support have *zero* support error and multiply perplexity by 4000. The error metric and the model's sensitivity are not aligned, and you have to measure the model.

**The concentration of attention mass is the whole mechanism.** 0.151% of entries carry 53.8% of the mass; 75.1% of entries carry 1.2%. Everything follows: support can be cut, tail precision is free, resolution belongs near the row maximum, and probability-weighted error is the summary that predicts damage. If you work on [[Sparse Attention]] or [[Quantization]], this distribution is the number to have in your head.

**The row-max anchor is free real estate.** Online softmax already tracks a running maximum ([[Flash Attention]], [[FlashAttention- Fast and Memory-Efficient Exact Attention]]). Anchoring your lattice to it costs nothing, removes all calibration, and buys $\Delta\mathrm{NLL}$ reduction of 0.00355 over an absolute anchor at the same spacing. The vision-transformer quantisation literature (FQ-ViT, RepQ-ViT) anchors absolutely and pays a $2.17\times$ penalty for it.

**Limitations the authors are unusually honest about:**

- Kernel fidelity was measured on BF16 at 2K only — **not at the 8K and 16K shapes where all the speed numbers come from**. Fidelity and speed are measured under different dtypes and different lengths. That is a real gap.
- The FP8 fidelity number is a simulation on one model. The direct run was drowned by unrelated quantisation error.
- Timing covers the attention **forward** only. Not backward, not decode, not full-model. The paper does the arithmetic: causal attention is 21% of Qwen2.5-1.5B prefill FLOPs at 8K and 35% at 16K — and then says these are workload shares, not runtime shares, and declines to claim a prefill speed-up.
- No counters. Every mechanism claim is inferential.
- B200 only. FA4 reports B300/GB300 doubles native exponential throughput to 32 ops/clk/SM, and disables polynomial emulation for `sm_103`. **This paper's advantage may largely evaporate on the next generation.** The authors say explicitly they make no claim about B300.
- Five models is not a scaling trend, and they say so.

**Practical caveats if you wanted to use this:**

1. Below 4K sequence length, don't. Launch overhead eats the gain and call latency goes negative.
2. Head dim 64 is not covered by FA4's tuning table, so the comparison is meaningless there.
3. The gain scales with tensor-core rate relative to SFU rate. FP8 gets 12.4%, BF16 gets 5.4%. If you are not on FP8, the case is much weaker.
4. There is measurable quality loss. 0.09–0.49% perplexity. Small, but resolved above zero in all five models. Budget for it.
5. Gemma-2-2B is consistently the outlier — worst $K=32$ result (0.852%), only unresolved $D_R$, lowest top-1 agreement (92.1%) at only 0.353% perplexity increase. Something about its attention-logit softcap makes it different. Do not assume your model behaves like the median.

**A surprising detail worth keeping:** top-1 next-token agreement can be low while perplexity looks fine. Gemma-2-2B under Rowmax-PoT has 92.1% top-1 agreement at 0.353% perplexity increase. A small average NLL change does **not** mean the model makes the same predictions. If you care about output stability rather than average loss, measure agreement separately.

**Connections.** The $\hat p_j - p_j \approx p_j(\epsilon_j - \mathbb{E}_p[\epsilon])$ result is the same "constant shift cancels under normalisation" fact that makes the max-subtraction trick in softmax safe in the first place. The concurrent EFQ-Softmax arrives at the identical $\{1, 1.5\}\times 2^k$ code set from a completely different direction — it needs E2M1 because its downstream $PV$ consumes MXFP4 operands, whereas here the set falls out of the FP32 bit construction. Two different constraints, same lattice. That is mild evidence the lattice is genuinely well-matched to the problem.

**Open questions:**

- $k=2$ and $k=3$ input quantisation give better fidelity offline and were never put in the kernel. What do they cost in instructions? The bit-level pattern is the same shape (shift by $23-k$), so possibly nothing.
- Rowmax-H15 changes the nonlinearity between the two matmuls; SageAttention and friends change the matmul operands. The authors say these compose in principle but never test it.
- Why does the sign of $D_T$ flip across models, when leakage, local $pV$ change and mass restoration all fail to explain it? Three negative controls, no positive one. This is the most interesting unanswered question in the paper.

---

## Links

Related: [[Attention]] · [[Flash Attention]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Quantization]] · [[Sparse Attention]] · [[Perplexity]] · [[Cross Entropy]] · [[KL Divergence]] · [[Mixed Precision training]] · [[Mixed Precision Training]] · [[GPU processing]] · [[Prefill and Decode]] · [[Multi-Head Attention]] · [[Query, Key, and Value (QKV)]] · [[Causal Attention]] · [[Grouped Query Attention]] · [[GQA- Training Generalized Multi-Query Transformer Models]] · [[Cost and Latency]] · [[Long Context]] · [[QLoRA- Efficient Finetuning of Quantized LLMs]] · [[Linear Attention]] · [[Attention Is All You Need]] · [[Distillation]]

New topics worth writing: Schraudolph exponential bit-hack, online softmax and the running-maximum rescale, special-function units vs tensor cores on GPUs, post-training quantization of attention probabilities (FQ-ViT / RepQ-ViT / PTQ4ViT), paired block bootstrap for model-comparison endpoints, pre-registration in systems ML papers, Jensen-Shannon divergence as a distortion metric, FP8 E4M3 and microscaling formats, probability-weighted vs unweighted error metrics, GPU energy measurement methodology
