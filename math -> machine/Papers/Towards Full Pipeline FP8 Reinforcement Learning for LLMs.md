---
title: "Towards Full Pipeline FP8 Reinforcement Learning for LLMs"
authors: ["Chen et al."]
year: 2026
arxiv: "2609.22870"
url: https://arxiv.org/abs/2609.22870
priority: Good-To-Read
read_on: 2026-09-27
tags: [paper, llm, rl, optimization, theory]
---
## The Core Idea

Run reinforcement learning on a language model, but do the arithmetic in FP8 — 8-bit floating point — instead of BF16. It is about 1.5× faster. It also breaks, and the reason it breaks is not the reason anyone expected.

Everyone assumed the problem was the **train–rollout mismatch**. You generate text with a fast FP8 inference engine, then compute gradients in a different backend. The two disagree about token probabilities, so the data is no longer on-policy. Fixes like truncated importance sampling patch that up.

This paper shows the real killer is somewhere else: **FP8 noise silently disables the part of [[PPO]] that punishes bad output.**

The chain is short and worth memorising:

1. In full-FP8 training, both $\pi_\theta$ and $\pi_{\text{old}}$ come from quantized forward passes. Each probability is slightly wrong.
2. The importance ratio $r_t = \pi_\theta / \pi_{\text{old}}$ is a *quotient* of two noisy numbers, so the errors compound. Measured: probability error ~1–2%, ratio error 1.65× to 2.9× larger.
3. PPO zeroes the gradient of a negative-advantage token once $r_t \le 1-\epsilon$. Extra noise pushes more tokens across that line by accident.
4. Those tokens were garbage — garbled, fragmented text. The optimiser wanted to suppress them. Instead their gradient vanishes.
5. Garbage is never unlearned, so it multiplies. Entropy explodes mid-training, around step 100.

> [!NOTE] Over-clipping
> A token is **lower-bound over-clipped** when $A_t < 0$, the true BF16 ratio is inside the trust region ($r^{\text{BF16}}_t > 1-\epsilon$), but the FP8 ratio has fallen outside it ($r^{\text{FP8}}_t \le 1-\epsilon$). The clip fires for a numerical reason, not a policy reason. ^over-clipping

The insight: the clipping bound $1\pm\epsilon$ is defined in *exact* arithmetic. Once your ratios live in quantized space, $0.8$ no longer means what it meant. The fix — **Calibrated Clipping** — stops treating $0.8$ as sacred and instead asks *what threshold clips the same fraction of tokens that BF16 would have clipped?* Move the bound, not the maths.

This could not have existed earlier because it only shows up when *training* is FP8 too. FP8-rollout-with-BF16-training is completely stable (see Figure 1). The bug needs the noise to be inside the ratio itself.

## The Methodology

### The setup being broken

Standard [[PPO]] / [[GRPO]] surrogate, with clipping:

$$J = \mathbb{E}\left[\sum_t \min\big(r_t(\theta)A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t\big)\right]$$

The paper's key rewrite splits the gradient into two jobs:

$$\underbrace{\sum_{t: A_t>0} \pi_{\text{old}} r_t A_t \mathbb{1}\{r_t < 1+\epsilon\}}_{\text{positive updates: raise good tokens}} + \underbrace{\sum_{t: A_t<0} \pi_{\text{old}} r_t A_t \mathbb{1}\{r_t > 1-\epsilon\}}_{\text{negative updates: suppress bad tokens}}$$

The lower bound $1-\epsilon$ is the **off switch for the second term**. That is the whole paper.

On top of this sits TIS, correcting for the FP8 sampler:

$$w^{\text{TIS}}_t = \min\!\left(\frac{\pi_{\text{old}}(y_t|h_t)}{\pi^{\text{FP8}}_{\text{sampler}}(y_t|h_t)},\ C\right), \quad C = 2$$

TIS is on in every experiment. It does not save you.

### FP8 quantization, concretely

For a tensor $X$, with $F_{\max}$ the largest FP8 value:

$$S_X = \frac{\max(|X|)}{F_{\max}}, \quad \hat{X} = Q_{\text{FP8}}\!\left(\frac{X}{S_X}\right), \quad \tilde{X} = S_X \hat{X}$$

The only design choice is the **granularity** of that $\max$:

| Granularity | One scale per | Accuracy | Speed |
|---|---|---|---|
| Tensorwise | whole tensor | worst (outlier-fragile) | 1.5× BF16 |
| Rowwise | each row | middle | 1.2–1.3× |
| Blockwise | $128\times128$ block | best | 1.1–1.2× |

Note the ordering: coarser is faster *and* noisier. Blockwise is what DeepSeek-V3 pretrained with.

### Calibrated Clipping, step by step

Two stages, run every 20 steps.

**Stage 1 — align the lower bound by quantile.**

Do two forward-only BF16 passes over the current batch, using the master weights, to get reference ratios $r^{\text{BF16}}_t$. Restrict to negative-advantage tokens. Read off what fraction BF16 would have clipped:

$$\alpha_{\text{low}} = F_{\text{BF16},-}(1-\epsilon)$$

Then find the FP8 threshold that clips *the same fraction*:

$$L = F^{-1}_{\text{FP8},-}(\alpha_{\text{low}})$$

Same number of tokens silenced, so no over-clipping. In practice $L$ lands around $0.6$–$0.75$.

**Stage 2 — rebalance the upper bound.**

Relaxing $L$ alone over-corrects. You now allow *too much* negative update; the model becomes timid. So match the positive-to-negative balance to BF16's. Define

$$\rho = \frac{\left|\sum_{A_t>0} w^{\text{TIS}}_t \pi_{\text{sampler}} r_t A_t \mathbb{1}\{r_t < c_{\text{high}}\}\right|}{\left|\sum_{A_t<0} w^{\text{TIS}}_t \pi_{\text{sampler}} r_t A_t \mathbb{1}\{r_t > c_{\text{low}}\}\right|}$$

Compute $\rho_{\text{BF16}}$ from the reference. Then, holding $L$ fixed, grid-search $H$ so $\rho_{\text{FP8}}(L,H) = \rho_{\text{BF16}}$.

**The search and the smoothing.** $L \in [0.5, 0.9]$, $H \in [1.2, 2.0]$, step $\Delta = 0.02$. Updates are damped so the trust region cannot jerk:

$$b_{\text{new}} = b + \text{clip}\!\left(\frac{b^\star - b}{2},\ -\delta,\ \delta\right)$$

with $\delta_L = 0.05$, $\delta_H = 0.1$. Initialise at $[0.6, 1.8]$.

**Why every-20-steps is enough.** Figure 6: the calibrated bound is flat once training settles. It only jumps during an entropy surge. So the two extra BF16 passes cost ~1/20th of a step's forward compute — amortised to nothing.

### Training config

- **GRPO:** Qwen3-8B-Base and Qwen2.5-32B, DeepScaleR, 16K context, batch 256 / mini-batch 64, 8 samples per prompt, LR $10^{-6}$, 500 steps, **no KL term**. Reference bounds $c^{\text{ref}} = [0.8, 1.24]$.
- **DAPO:** Qwen3-14B-Base, DAPO-Math-17K, 20K context, 16 samples per prompt, 200 steps. Reference bounds $[0.8, 1.28]$.
- Stack: VeRL + vLLM (rollout) + TorchAO (FP8 training) + FlashRL's FP8 rollout patch.

## Ablation Studies and Experiments

### The diagnosis, which is better than the fix

The case-study chain is the most reusable part of the note.

**Entropy bins (Figure 2).** Entropy is decomposed into $[0,0.2)$, $[0.2,0.5)$, $[0.5,1.0)$, $>1.0$. The global surge is not a general drift — most responses stay healthy. A small population with entropy $>1.0$ explodes in count after step 100.

**What those responses are.** Garbled, fragmented, nonsensical text. A decoding collapse.

**The smoking gun.** Only **6.64%** of those garbled responses have positive advantage. 93% *should* be being punished. They are growing anyway.

**Where the clipping lands (Figure 4).** Late in training, ~**70%** of over-clipping is at the lower bound. Nearly **90%** of those lower-bound over-clipped tokens come from the high-entropy garbage responses. The noise is not spread evenly — it concentrates exactly where suppression was needed.

**Clip fraction (Figure 1, right).** Even blockwise FP8 clips **3.94×** more often than BF16; rowwise 5.48×, tensorwise 6.68×.

### The ablation that proves the mechanism

Rerun rowwise FP8 with the lower bound hand-set from $0.8 \to 0.6$. Nothing else changed.

- Entropy surge: **gone**.
- But reward saturates low, and response length stays well below BF16 throughout.

So the diagnosis is right *and* the naive fix is wrong. Too much negative update makes the model conservative — short answers, early entropy collapse. This is exactly why Stage 2 exists. Two-sided problem, two-sided fix.

### Main results — GRPO

Qwen3-8B-Base, average over 8 reasoning benchmarks (AIME24/25, AMC23/24, MATH-500, Gaokao, Minerva, OlympiadBench):

| Setting | Average |
|---|---|
| BF16 / BF16 | 57.6 |
| BF16 train / FP8 rollout | 58.2 |
| Tensorwise FP8 | 46.1 |
| **+ Calibrated Clipping** | **55.9** (+9.8) |
| Rowwise FP8 | 47.0 |
| **+ Calibrated Clipping** | **56.5** (+9.5) |
| Blockwise FP8 | 54.1 |
| **+ Calibrated Clipping** | **58.6** (+4.5, beats BF16) |

Qwen2.5-32B, same protocol: BF16 51.9. Vanilla FP8 49.1 / 49.0 / 50.8 → with calibration 51.1 / 51.7 / 53.4. Smaller gaps, and the paper says why: this model emits much shorter responses (400–1200 tokens vs 1000–5000), so there is less room for entropy pathology to grow.

Note the pattern across both tables: **the noisier the format, the more calibration buys you.** Tensorwise goes from unusable to nearly-BF16.

### DAPO, Qwen3-14B-Base, AIME24 Avg@32

| Setting | Score |
|---|---|
| BF16 | 50.9 |
| BF16 train / FP8 rollout | 47.4 |
| Tensorwise FP8 | 35.7 → **47.9** (+12.2) |
| Rowwise FP8 | 38.1 → **46.5** (+8.4) |
| Blockwise FP8 | 41.6 → **47.4** (+5.8) |

DAPO is *more* vulnerable. Its higher upper bound (1.28) means entropy climbs even in BF16 by design, so there is no natural ceiling to stop the surge.

### Coding, Eurus-2-RL (Appendix A.4)

Qwen3-8B, average over TACO/APPS/Codeforces. BF16 49.61. Vanilla FP8 45.77 / 46.00 / 46.38 → 47.88 / 48.58 / 48.75. Consistent, smaller. Not a maths-only artefact.

### What did not work

This section is the reason to read the paper.

**Relaxing the lower bound alone.** Fixes entropy, breaks reward and length. Covered above.

**CISPO (Appendix A.2).** CISPO's whole pitch is: don't zero gradients on clipped tokens, clip the sampling weight instead. That should fix a gradient-vanishing problem by construction. Applied to rowwise FP8 GRPO it **collapses at ~300 steps** — entropy spikes to 6, reward falls to zero. Keeping the gradient is not enough if the *boundary itself* is in the wrong place.

**BAPO (Appendix A.3).** BAPO also moves clipping bounds dynamically, but targets a fixed positive-contribution value $\rho_0 = 0.4$. Under rowwise FP8, $\rho$ falls below target at ~90 steps, BAPO chases it by pushing bounds to $[0.9, 3.0]$, and reward collapses to zero. A *fixed* target is the failure. Calibrated Clipping's target is read off BF16 each time, so it moves with the run.

**Throughput honesty (Figure 7).** Tensorwise 1.4–1.5×, rowwise 1.2–1.3×, blockwise 1.1–1.2×. And blockwise 32B at 16K **OOMs**. The speedup is inverse to the accuracy — and blockwise, the only granularity that reliably matches BF16, is the one that barely helps.

### Hyperparameter sensitivity (Appendix A.5)

Recalibration interval 10 / 20 / 40 steps → 58.31 / 56.51 / 57.91. Initialisation $[0.6,1.8]$ vs $[0.8,1.2]$ → 56.51 / 56.20. Non-monotonic in the interval, which smells like run-to-run noise rather than a real trend. All configurations recover most of the gap. The authors state they did not tune these.

## Worth Remembering

**The transferable lesson has nothing to do with FP8.** Any threshold defined in exact arithmetic becomes a *different* threshold once you add noise to the quantity it tests. Clipping bounds, gradient-norm clip values, early-stopping tolerances, trust-region radii — all of them. If you change precision anywhere upstream, your constants changed meaning.

**A division is a noise amplifier.** 1–2% error on two probabilities becomes up to 2.9% on their ratio. Anything in your pipeline shaped like $a/b$ with both terms estimated deserves suspicion.

**Asymmetric failure is the interesting part.** Noise is symmetric; its *consequences* are not. Over-clipping a positive-advantage token just skips one reward. Over-clipping a negative-advantage token lets a pathology survive and reproduce. There is a feedback loop on one side only. Compare [[Mode Collapse]] and [[Reward Hacking]] — same flavour of "the optimiser found the gap in the incentive".

**Read the entropy bins, not the entropy.** The global average looked fine until step 100. The 4-bin decomposition showed the bad population growing all along. Aggregate stability metrics hide sub-population collapse. This generalises well past RL.

**Practical verdict.** If you want FP8 RL today: blockwise + Calibrated Clipping matches or beats BF16, for 10–20% speed. Tensorwise + calibration gets you 1.5× but sits ~1.7 points below BF16 on 8B. The "free" configuration does not exist.

**Limitations the authors admit.** Everything is token-level clipping (GRPO, DAPO). [[GRPO|GSPO]]-style sequence-level clipping is untested — the paper speculates aggregation may dampen per-token ratio error, but the aggregated sequence ratio could still be distorted. The method also needs BF16 master weights available for shadow passes, which is fine in TorchAO/FSDP but rules out a genuinely BF16-free pipeline. And the whole thing is calibration against a reference you must keep paying for — it reduces the precision benefit rather than removing the dependency.

**Open questions.** Does the effect show up in FP8 *pretraining* with any thresholded objective? Would per-token adaptive bounds beat one global pair? Is there a cheaper reference than a full BF16 pass — e.g. calibrating on a subsample of the batch?

## Links

Related: [[PPO]] · [[GRPO]] · [[Proximal Policy Optimization Algorithms]] · [[Mixed Precision Training]] · [[Quantization]] · [[Mixed Precision training]] · [[On-Policy vs Off-Policy]] · [[Off-Policy Evaluation]] · [[Reward Hacking]] · [[Mode Collapse]] · [[Policy Gradient]] · [[RLHF]] · [[Distributed Training]] · [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]] · [[Training language models to follow instructions with human feedback]] · [[KL Divergence]] · [[Credit Assignment]]

New topics worth writing: FP8 formats (E4M3 vs E5M2), truncated importance sampling (TIS) and rollout–train mismatch, FP8 scaling granularity (tensorwise / rowwise / blockwise), DAPO, GSPO and sequence-level clipping, CISPO, BAPO, TorchAO, entropy collapse and entropy surges in RLVR, quantile matching as a calibration technique
