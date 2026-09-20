---
title: "Verification-Aware Training for Speculative Decoding"
authors: ["Geonmo Gu", "Byeongho Heo", "HeeJae Jun", "Yoohoon Kang", "Sangmin Lee", "Sangdoo Yun", "Dongyoon Han"]
year: 2026
arxiv: "2608.30135"
url: https://arxiv.org/abs/2608.30135
priority: Good-To-Read
read_on: 2026-09-05
tags: [paper, llm, diffusion, vision]
---
## The Core Idea

Speculative decoding makes a big language model generate text faster. A small "draft" model guesses the next $K$ tokens cheaply. The big "target" model then checks all $K$ guesses in **one** forward pass. Checking is sequential: you walk left to right, and the moment one guess is rejected, every guess after it is thrown away — even if some of them happened to be correct. Then drafting restarts from the last accepted token. The output is exactly what the target would have produced on its own, so the speedup is free of quality loss.

The number that decides your speedup is the **average acceptance length** $\tau$: how many draft tokens survive per check cycle.

Here is the mismatch this paper attacks. Every draft model — EAGLE-3, DFlash, all of them — is trained by plain imitation: at each of the $K$ positions, match the target's next-token distribution, with a fixed decay weight like $w_k = 0.8^{k-1}$ that says "earlier positions matter more". Nothing in that objective knows about the two facts that actually govern $\tau$:

1. A token at position $k$ is worth something only if **all** positions before it were accepted. Correctness at position 7 is worthless if position 3 died.
2. The decay schedule is written down before training and is the same for every training example. But the place where a sample actually breaks — its first rejection point $k^*$ — is different for every sample.

Verification-Aware Training (VAT) fixes both by **simulating the verification step at every training step**. You already have both models' distributions during training, so you can just run the acceptance rule and see where the draft would have died. That gives you a per-sample first-rejection index $k^*$ and a per-position accept/reject label. VAT turns those into supervision two ways: an auxiliary head that predicts survival, and a loss weighting anchored at $k^*$ instead of at $k=1$.

Why it did not exist before: the field poured its effort into draft *architectures* (parallel heads → feature-level autoregressive drafters → block-diffusion drafters) and left the objective as boilerplate cross-entropy. VAT touches only the loss. Architecture, target model, and inference path are untouched, so it bolts onto anything. Gains: up to **+11.4% acceptance length** and **+8.7% wall-clock speedup**.

> [!NOTE] Acceptance length
> The expected number of draft tokens the target accepts per verification cycle, written $\tau$. Each cycle costs one target forward pass, so speedup rises roughly with $\tau$. It is the single metric that matters for speculative decoding. ^acceptance-length

## The Methodology

**Simulating verification.** At training step, draft and target give distributions $\hat p_k$ and $p_k$ at every draft position $k$. Standard speculative sampling accepts a drafted token $x$ with probability $\min(1, p_k(x)/\hat p_k(x))$. Define the per-position indicator

$$m_k = \mathbb{1}[\text{token at }k\text{ accepted}]$$

the first rejection point

$$k^* = \min\{k : m_k = 0\}$$

(with $k^* = K+1$ if everything is accepted), and the **cumulative** acceptance label

$$v_k = \mathbb{1}[k < k^*] = \prod_{j\le k} m_j.$$

The difference between $m_k$ and $v_k$ is the whole paper. $m_k$ is local — did this token match. $v_k$ is what inference actually cares about — did this token *and everything before it* match.

**Component 1: the verification head.** A single dense layer sitting on the draft model's last hidden states, mapping each position to a probability $\hat v_k$ that the position survives. Trained with binary [[Cross Entropy|cross-entropy]]:

$$\mathcal{L}_{\text{VH}} = -\frac{1}{K}\sum_{k=1}^{K}\big[v_k\log\hat v_k + (1-v_k)\log(1-\hat v_k)\big]$$

Because it is trained jointly, [[Backpropagation|gradients]] from this loss flow back into the draft body and push its hidden states toward features that predict *agreement with the target*. Cost at inference: zero, because you can simply not use it (though §5.3 shows you can).

**Component 2: verification-adaptive weighting.** Replace the fixed per-position weight $w_k$ with

$$\hat w_k = \begin{cases} 1 & k < k^* \\ w_{k-k^*+1} & k \ge k^*\end{cases}$$

In words: everything before the first rejection gets **full** weight, because those tokens genuinely contributed to $\tau$ on this sample. The base decay curve is not deleted — it is *slid over* so its start lines up with $k^*$. The rejection position itself gets weight 1 (since $w_1=1$ in both baselines), which is deliberate: $k^*$ is the nearest fixable failure, and fixing it directly extends the accepted prefix. Since $k^*$ varies per sample, no fixed schedule can imitate this.

**Full objective.**

$$\mathcal{L} = \sum_{k=1}^{K}\hat w_k\big(\ell_k^{\text{soft}} + \ell_k^{\text{hard}}\big) + \beta\,\mathcal{L}_{\text{VH}}, \qquad \beta = 1.0$$

$\ell^{\text{soft}}$ is cross-entropy against the target's full output distribution (classic [[Distilling the Knowledge in a Neural Network|distillation]]), $\ell^{\text{hard}}$ against the target's sampled token. EAGLE-3 originally used soft only; DFlash used hard only. VAT uses both.

**Setup.** Baselines: EAGLE-3 (autoregressive drafter reusing target hidden states) and DFlash (block-diffusion drafter that emits all 16 draft tokens in one parallel pass — see [[Denoising Diffusion Probabilistic Models]] for the diffusion machinery). Targets: Qwen3-4B, Qwen3-8B, LLaMA-3.1-8B. Training corpus: Perfectblend prompts with responses regenerated by the target model itself (greedy), 3 epochs, A100 80GB, bf16. Base schedules kept as-is: $0.8^{k-1}$ for EAGLE-3, $\exp(-(k-1)/\gamma)$ with $\gamma=7$ for DFlash.

## Ablation Studies and Experiments

**Main table** (average over GSM8K, MATH-500, AIME25, HumanEval, MBPP, LiveCodeBench, MT-Bench, Alpaca; temperature 0):

| Target | Method | Speedup | $\tau$ |
|---|---|---|---|
| Qwen3-4B | EAGLE-3 | 4.07× | 6.28 |
| | + VAT | **4.39×** (+7.9%) | **6.78** (+8.0%) |
| | DFlash | 4.54× | 5.73 |
| | + VAT | **4.81×** (+5.9%) | **6.08** (+6.1%) |
| Qwen3-8B | DFlash | 4.47× | 5.51 |
| | + VAT | **4.86×** (+8.7%) | **6.14** (+11.4%) |
| LLaMA-3.1-8B | EAGLE-3 | 4.17× | 6.08 |
| | + VAT | 4.33× (+3.8%) | 6.23 (+2.5%) |

Gains are smallest on LLaMA-3.1-8B and largest on Qwen3-8B + DFlash. Holds at temperature 1 too.

**Component ablation** (DFlash + Qwen3-4B, average $\tau$, baseline 5.73):

| Verif. head | Adaptive weight | Soft+hard | Speedup | $\tau$ |
|---|---|---|---|---|
| | | | 4.54× | 5.73 |
| ✓ | | | 4.62× | 5.87 |
| | ✓ | | 4.65× | 5.91 |
| | | ✓ | 4.58× | 5.82 |
| ✓ | ✓ | | 4.76× | 5.99 |
| ✓ | ✓ | ✓ | **4.81×** | **6.08** |

Each piece helps alone; they compound. The head and the weighting are addressing different aspects (representation vs. gradient allocation), which is why stacking works.

**Which weight schedule matters.** Uniform weighting ($\tau=5.72$) and EAGLE-3's $0.8^{k-1}$ ($\tau=5.73$) are indistinguishable — the classic decay does essentially nothing. But make *either* base schedule verification-adaptive and you land at $\tau = 6.09$ (EAGLE-3 base) or $6.08$ (DFlash base). **The functional form of the decay is irrelevant; the anchoring at $k^*$ is what works.**

**What did not work** (Appendix A, 1-epoch budget, DFlash + Qwen3-4B, baseline 4.27× / 5.54):

- **Prefix-only** — zero weight from $k^*$ onward: catastrophic collapse to **2.44× / 3.11**, far below baseline. Early in training $k^*$ is small, so most samples produce almost no gradient at all. Killing the post-rejection signal starves the model.
- **Hard cutoff** — weight 1 up to and including $k^*$, then 0: 4.44× / 5.75. Recovers most of the gap but not all. So post-rejection tokens *do* carry useful signal, as long as it is decayed rather than deleted.
- **Unshifted decay** — full weight before $k^*$, base schedule applied from $k=1$: 4.46× / 5.75. Re-anchoring is worth roughly the same as not-zeroing.
- **Marginal contribution** — weight each position by the exact derivative of expected accepted length: 4.38× / 5.74. Worse, because the "probability verification reaches here" factor discounts $k^*$ itself whenever the prefix is uncertain.
- **GRIFFIN-style masking** — zero the loss wherever the drafted token is outside the target's top-3: 4.32× / 5.64, barely above baseline. Per-position criteria miss the sequential structure.
- **D-PACE-style confidence weights** (concurrent work): 4.52× / 5.89 — the closest competitor, but still below VAT's 4.61× / 6.03. Conditioning on the *observed* rejection beats a confidence proxy.

**Training dynamics.** With the head attached, $k^*$ drifts to later positions over training — the draft genuinely survives longer. Surprising second finding: the count of post-rejection tokens that still match the target *declines* over training for the baseline but stays flat with the head, even though those positions start deeper (harder context). The head does not fight the token-level loss; it improves representations generally.

**The head at inference (optional).** Threshold $\hat v_k$ at $p$ to guess the first rejection. Best thresholds: $p=0.5$ for EAGLE-3 (MAE 1.18 tokens), $p=0.6$ for DFlash (MAE 1.76). Use it to truncate the draft before sending it to the target. On DFlash Code: 4.83× → **4.97×** (oracle upper bound 5.20×), at the cost of a small $\tau$ drop (7.67 → 7.35 on DFlash Math) from occasional false rejections. Gains are larger on DFlash than EAGLE-3, because EAGLE-3's autoregressive drafting already dominates compute so there is less verification cost to save.

## Worth Remembering

- **Training overhead is asymmetric.** EAGLE-3 + VAT costs only **+1.2%** per step (18.5 → 18.6 GB), because EAGLE-3 already computes the target's LM head distribution for its soft labels and VAT just reuses it. DFlash + VAT costs **+6.1%** time and **23.8 → 31.5 GB** peak memory, because DFlash never applies the target LM head during training, so simulating verification adds a whole LM-head pass. If you are on a memory-tight setup with a diffusion drafter, budget for that.
- **VAT is insensitive to the verification rule and corpus temperature.** Greedy verification (top-1 agreement) and full stochastic acceptance produce accepted lengths correlated above **0.92** throughout training, and the four combinations of {corpus $T=0,1$} × {greedy, sampling verification} differ by at most 0.02 in speedup and $\tau$. So use greedy — it is cheaper and identical.
- **The real lesson is not the head, it is the anchor.** Uniform vs. exponential decay is a wash. What buys the gain is conditioning the weight on a per-sample event. That is a general pattern worth stealing: if your inference procedure has a sample-dependent break point, do not use a sample-agnostic schedule.
- Limitation the authors state: nothing above 8B parameters was tested. Draft/target ratios and acceptance behaviour could change a lot at 70B+.
- The verification head is a strictly cheaper form of the "predict your own correctness" idea. It is a binary classifier on hidden states, one dense layer, $\beta=1$, no tuning reported. Very low risk to try.
- Open question: the post-rejection stability result (Fig. 2b) suggests the head acts partly as a [[Regularization|regulariser]] on the draft's representations rather than purely as verification supervision. Nobody isolated that.
- Practical caveat: all numbers are HuggingFace Transformers on A100. Under a serving stack like [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)|vLLM]] with batching, the compute balance between drafting and verification shifts, and the early-exit gains in particular may not transfer.

## Links

Related: [[Distilling the Knowledge in a Neural Network]] · [[Cross Entropy]] · [[Auto-regressive models]] · [[Denoising Diffusion Probabilistic Models]] · [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]] · [[Prefix Sliding for efficient test-time scaling]] · [[Fast Transformer Decoding- One Write-Head is All You Need (MQA)]] · [[Backpropagation]] · [[Regularization]]

New topics worth writing: Speculative decoding, EAGLE-3, DFlash and block diffusion language models, Medusa decoding heads, rejection sampling for lossless acceleration, early-exit drafting
