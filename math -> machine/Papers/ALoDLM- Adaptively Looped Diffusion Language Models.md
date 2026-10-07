---
title: "ALoDLM: Adaptively Looped Diffusion Language Models"
authors: ["Liancheng Fang", "Zhuowei Li", "Youngeun Kim", "Tianchen Zhao", "Rajat Koner", "Jiaye Wu", "Linghan Xu", "Xuanbai Chen", "Xiang Xu", "Zheng Zhang", "Jakub Zablocki", "Nishant Sankaran", "Yifan Xing"]
year: 2026
arxiv: "2610.04198"
url: https://arxiv.org/abs/2610.04198
priority: Good-To-Read
read_on: 2026-10-06
tags: [paper, llm, diffusion, theory]
---
## The Core Idea

Diffusion language models (DLMs) write text by filling in blanks. Start with a row of `[MASK]` slots, then guess several of them at once, repeat. Because many slots get filled per forward pass, generation is fast. But the text has always been a bit worse than a plain [[auto-regressive models|autoregressive]] model of the same size.

This paper names the reason: **computation–difficulty mismatch**.

Inside one denoising step, the blanks are not equally hard. Some are trivial — the word after "New" is probably "York". Some are hard — a number in the middle of a sum that depends on blanks you have not filled yet. A standard DLM runs the *same* fixed stack of layers on every blank. Easy slots get more compute than they need; hard slots get starved.

> [!NOTE] Computation–difficulty mismatch
> Within a single parallel denoising step, masked positions differ wildly in how hard they are to predict, but a fixed-depth denoiser spends identical compute on all of them. ^compute-difficulty-mismatch

An autoregressive model never has this problem. It predicts one token, with a fully written prefix behind it. All its compute goes to one decision.

The existing partial fix is confidence-based decoding: if a slot's prediction is unsure, do not commit it, leave it masked, try again next step. But that throws away the work. Next step the token starts again from its bare `[MASK]` embedding. The thinking it did is gone.

ALoDLM's trick: **keep the latent state and loop on it, per token.**

Inside each denoising step there is now an inner loop. After each inner pass:
- Tokens that look confident **commit** — they get sampled, and their *token embedding* is written into the hidden state. They are now context.
- Tokens that are still unsure keep their **hidden vector** and feed it back for another pass through the same layers.

So a hard token gets 2, 3 or 4 passes through the recurrent block, and each extra pass sees more committed neighbours. Easy tokens get one pass and become scaffolding for the hard ones.

This is a [[Mixture of Experts|conditional-compute]] idea applied at token granularity inside the diffusion step, not across steps. It did not exist before because the training problem is nasty: the decision "when does token *i* stop looping?" is discrete, and because committing token *i* changes the context for every other token, you cannot just sum over all possible stopping patterns like [PonderNet] does. The number of joint schedules is $K^{|\mathcal{M}|}$ and each one is a *different* forward trajectory.

What it unlocks, concretely: ALoDLM-8B averages **80.3** across 11 benchmarks, beating Qwen3-8B (78.5, autoregressive) and the best DLM baseline WeDLM-8B (75.1) — while running at about **2.7× the throughput** of vLLM-served Qwen3-8B on GSM8K. It is, as far as the authors know, the largest looped DLM trained.

## The Methodology

### Architecture

Take Qwen3. Cut the layer stack in three:

| Part | Runs | 1.7B | 8B |
|---|---|---|---|
| `Prelude` (embeddings + prefix blocks) | once per denoising step | none (just embeddings) | layers $[0,10)$ |
| `Recurrent Core` | once per inner pass | all 28 layers | layers $[10,26)$ |
| `Coda` (suffix blocks) | once per inner pass, to read out | none | layers $[26,36)$ |

Max recurrent depth $K=4$.

### One denoising step, mechanically

Input is the corrupted sequence $\mathbf{x}_t$. $\mathcal{U}$ is the set of still-masked positions.

$$h^{(0)} = \mathsf{Prelude}(\mathbf{x}_t) \in \mathbb{R}^{N\times d}$$

Then for inner pass $s = 1 \ldots K$:

$$\tilde h^{(s)} = \mathsf{RecurrentCore}(h^{(s-1)}), \qquad r^{(s)} = \mathsf{Coda}(\tilde h^{(s)})$$

The readout $r^{(s)}$ goes to **two heads**:

$$p_i^{(s)} = \operatorname{softmax}\big(\mathsf{LMHead}(r_i^{(s)})\big), \qquad \lambda_i^{(s)} = \sigma\big(\mathsf{ExitGate}(\operatorname{sg}[r_i^{(s)}])\big)$$

with $\lambda_i^{(K)} = 1$ forced, so everything exits by depth $K$. The `sg` is a stop-gradient: the exit gate reads the backbone but does not push gradient back into it.

The two heads do different jobs, and this split matters:
- **`LMHead` confidence decides *which tokens commit*.** A token commits if its predictive entropy $\le \tau$.
- **`ExitGate` decides *whether the inner loop keeps going*.** Hazard rates accumulate as $a_i^{(s)} = a_i^{(s-1)} + (1-a_i^{(s-1)})\lambda_i^{(s)}$. The loop stops when $\mathcal{U}$ empties, or when the mean $a_i$ over remaining positions crosses $q$.

### The feedback rule — the heart of it

$$h_i^{(s)} = \begin{cases}\operatorname{Emb}(\hat y_i), & i \in \mathcal{C}^{(s)} \text{ (committed)}\\[2pt] \tilde h_i^{(s)}, & \text{otherwise}\end{cases}$$

Committed positions are replaced by their **discrete token embedding**. Unresolved positions carry their **continuous hidden state** forward. That asymmetry is the whole architecture: discrete for what you know, latent for what you are still working out.

This differs from [self-conditioning], which feeds back a *detached* prediction across denoising steps. Here the hidden states persist *within* a step and gradients flow back through the loop, so earlier passes are explicitly trained to set up later predictions.

If no token commits in a whole inner loop, the lowest-entropy position is committed anyway, to guarantee progress.

### The training objective

Treat the stopping pattern as a [[Latent Variable Models|latent variable]].

> [!NOTE] Exit schedule
> A discrete vector $\mathbf{z} = (z_i)_{i\in\mathcal{M}_t} \in \{1,\dots,K\}^{|\mathcal{M}_t|}$, where $z_i = s$ means token $i$ committed after inner pass $s$. One $\mathbf{z}$ plus one corrupted input fully determines one inner-loop trajectory. ^exit-schedule

You would like to marginalise:

$$p_\theta(\mathbf{y}_{\mathcal{M}_t}\mid\mathbf{x}_t) = \sum_{\mathbf{z}} \pi(\mathbf{z})\, p_\theta(\mathbf{y}_{\mathcal{M}_t}\mid\mathbf{x}_t,\mathbf{z})$$

Impossible — exponentially many schedules, and each needs its own forward pass because commitments change everyone's context. So introduce a variational posterior $q_\phi(\mathbf{z}\mid\mathbf{x}_t,\mathbf{y})$ parameterised by the exit gate and write a negative ELBO (see [[Auto-Encoding Variational Bayes (VAE)#^elbo|ELBO]]):

$$J(\theta,\phi) := \mathbb{E}_{\mathbf{z}\sim q_\phi}\Big[\underbrace{-\textstyle\sum_{i\in\mathcal{M}_t}\log p_{\theta,i}^{(z_i)}(y_i\mid\mathbf{x}_t;\mathbf{z})}_{\mathcal{L}_{\mathrm{traj}}(\mathbf{z})}\Big] + D_{\mathrm{KL}}(q_\phi\|\pi) \;\ge\; -\log p_\theta(\mathbf{y}_{\mathcal{M}_t}\mid\mathbf{x}_t)$$

Full objective, with $1/t$ reweighting standard for masked diffusion:

$$\mathcal{L}_{\mathrm{train}} = \mathbb{E}_{\mathbf{y}\sim p_{\text{data}},\, t\sim U[0,1],\, \mathbf{x}_t\sim q_t(\cdot\mid\mathbf{y})}\Big[\tfrac1t J(\theta,\phi;\mathbf{x}_t,\mathbf{y})\Big]$$

The prior $\pi(\mathbf{z})$ is a product of **truncated geometric** distributions over $\{1,\dots,4\}$ with rate $c=0.4$, giving $\approx(0.413, 0.277, 0.186, 0.124)$ and a prior mean depth of $\approx 2.02$. It is a pressure toward shallow exits.

### Getting a gradient through a discrete decision

$\theta$ (the denoiser) backprops normally. $\phi$ (the gate) cannot — $\mathbf{z}$ is discrete. They use a [[Simple Statistical Gradient-Following Algorithms (REINFORCE)|score-function estimator]]:

$$\widehat{\mathcal{L}}_{\mathrm{sur}}(\mathbf{z}) = \frac1t\Big\{\mathcal{L}_{\mathrm{traj}}(\mathbf{z}) + \operatorname{sg}\Big[\mathcal{L}_{\mathrm{traj}}(\mathbf{z}) + \log\tfrac{q_\phi(\mathbf{z})}{\pi(\mathbf{z})}\Big]\log q_\phi(\mathbf{z})\Big\}$$

and prove it unbiased: $\mathbb{E}[\nabla\widehat{\mathcal{L}}_{\mathrm{sur}}] = \nabla\mathcal{L}_{\mathrm{train}}$.

Read the second term as [[Policy Gradient|REINFORCE]]. The detached bracket is the **cost of the sampled schedule** — how badly the sequence got predicted, plus how far the schedule strayed from the geometric prior. Every halting decision in that rollout gets credit (or blame) for the whole-sequence outcome. This is sequence-level [[Credit Assignment|credit assignment]] for compute decisions.

The schedule log-prob factorises over *hazards along the realised history*:

$$u_i(\mathbf{z}) = \sum_{s<z_i}\log(1-\lambda_i^{(s)}) + \mathbf{1}\{z_i<K\}\log\lambda_i^{(z_i)},\qquad \log q_\phi(\mathbf{z}) = \sum_i u_i(\mathbf{z})$$

Not an independence assumption — the $\lambda$'s are evaluated on the actual trajectory, where earlier commits already changed later hazards.

### Two variance tricks

Score-function estimators are noisy (see [[Policy Gradient#^baseline|baselines]]).

**1. Control variate.** Subtract the detached first-pass cost $B_b = \sum_i \ell_i^{(1)}$ per sequence. The first readout happens *before* any commitment, so it is independent of $\mathbf{z}$ — it adds nothing in expectation ($\mathbb{E}[B_b\nabla_\phi\log q_\phi]=0$) but tracks per-input difficulty.

$$\widehat{\mathcal{L}}_{\mathrm{gate}} = \frac1M\sum_b \operatorname{sg}[R_b - B_b]\log q_\phi(\mathbf{z}_b)$$

**2. Intermediate supervision.** The rollout already computed predictions at every pass before the exit. Rather than supervising only at $z_i$, average over all passes up to it, weighted by the gate's own exit-depth mass:

$$\widehat{\mathcal{L}}_{\mathrm{den}} = \frac1t\sum_{i}\sum_{s\le z_i}\operatorname{sg}[\omega_i^{(s)}]\,\ell_i^{(s)},\qquad \sum_{s\le z_i}\omega_i^{(s)}=1$$

Costs zero extra forward passes.

**Relaxed KL.** The exact joint KL would push every token toward the same depth profile, killing adaptivity. They decompose it:

$$R_{\alpha,\beta} = \frac{\alpha}{M}\sum_i D_{\mathrm{KL}}(r_i\|\bar r) + \beta\, D_{\mathrm{KL}}(\bar r\|\pi_{0,c})$$

and set $(\alpha,\beta)=(0.1,1)$ — strongly match the *batch-average* depth to the prior, barely penalise *per-token* variation. Theorem 3 shows $\alpha=\beta=1$ recovers the exact joint KL. This relaxation is what lets different token types learn different depths.

### Training setup

- Convert Qwen3-1.7B and Qwen3-8B using **WeDLM's** streaming block-diffusion framework (full-sequence conditioning with a plain [[Causal Attention|causal mask]]).
- **No continued pretraining.** Straight [[Instruction Tuning|SFT]] on a 5B-token corpus — SDAR and WeDLM both need an expensive CPT stage.
- [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], lr $10^{-5}$, cosine, 3% warmup, weight decay 0.01, grad clip 1.0, bfloat16, DeepSpeed ZeRO-2.
- 4,096-token packed sequences, 32-token training blocks each with its own independently sampled mask probability in $[0,1]$. Global batch 256. 8B runs 8 epochs = 34,512 steps.
- An auxiliary next-token [[Auto-regressive models|AR]] loss at equal weight, which prior AR→DLM conversion work found helpful. Total: $\widehat{\mathcal{L}}_{\mathrm{den}} + \widehat{\mathcal{L}}_{\mathrm{gate}} + \widehat{\mathcal{L}}_{\mathrm{AR}}$.

### Serving

Weight sharing does **not** mean shared [[KV Cache|KV cache]]. The same physical layer sees different hidden states at different depths, so they keep **depth-specific** KV buffers $\{K_\ell^{(d)}, V_\ell^{(d)}\}$ for the recurrent core. A depth-$d$ attention reads prefix entries at depth $\min(d, D_{\ell,i})$ — matching depth where it exists, deepest available otherwise. At 8B: $L + (K-1)m = 36 + 3\cdot16 = 84$ logical buffer pairs. Served through [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)|vLLM]].

Layer cost per position for $S$ executed passes: $C(S) = L_{\text{start}} + S(L_{\text{end}}-L_{\text{start}}) + S(L - L_{\text{end}})$. For the 8B partition, a full 4-pass path is $10 + 4\cdot16 + 4\cdot10 = 114$ layers — cheaper than 4 full-stack passes (144).

## Ablation Studies and Experiments

### Main table — 11 benchmarks, one token committed per model pass

All models forced to commit exactly one token per step under their own default rule, OpenCompass protocol, greedy, 4,096-token cap.

| | Qwen3-8B (AR) | LLaDA-8B | Dream-7B | Fast-dLLM-v2-7B | SDAR-8B | WeDLM-8B | **ALoDLM-8B** |
|---|---|---|---|---|---|---|---|
| ARC-C | 93.9 | 85.6 | 84.3 | 77.2 | 90.0 | 91.9 | **94.4** |
| ARC-E | 96.1 | 92.6 | 93.0 | 83.4 | 93.4 | 97.5 | **98.1** |
| MMLU | 76.6 | 62.4 | 68.4 | 66.7 | **78.5** | 78.0 | 76.6 |
| MMLU-Pro | 56.8 | 35.6 | 42.0 | 40.6 | 56.3 | 58.3 | **63.9** |
| GSM8K | 93.6 | 73.8 | 82.0 | 85.1 | 91.4 | 93.3 | **94.2** |
| MATH-500 | 81.8 | 42.2 | 42.0 | 58.2 | 77.0 | 77.8 | **80.8** |
| GPQA-Diamond | 47.0 | 22.7 | 23.7 | 21.2 | 38.4 | 37.4 | **49.5** |
| MBPP | 79.0 | 46.0 | 65.5 | 61.9 | 72.0 | 74.3 | **81.5** |
| MBPP+ | **74.5** | 43.4 | 60.6 | 50.0 | 67.9 | 63.5 | 72.5 |
| HumanEval | 85.4 | 43.3 | 53.7 | 65.9 | 78.0 | 79.9 | **87.8** |
| HumanEval+ | 79.3 | 38.4 | 50.0 | 61.0 | 73.2 | 74.4 | **84.2** |
| **Average** | 78.5 | 53.3 | 60.5 | 61.0 | 74.2 | 75.1 | **80.3** |

Wins 10 of 11. Loses MMLU to SDAR (76.6 vs 78.5) and MBPP+ to the AR baseline. At 1.7B: ALoDLM 65.5 vs Qwen3 63.8 vs SDAR 61.0.

The code gains are the biggest jump — +7.2 on MBPP and +7.9 on HumanEval over WeDLM. The authors attribute this to looping, citing compute-matched looped-transformer scaling results. **Plausible but not isolated here** — no looped-vs-unlooped-at-matched-FLOPs control on code specifically.

### The two decoding knobs do different things

ALoDLM-8B, GSM8K, single-stream, one B200:

**Entropy threshold $\tau$ (how eagerly to commit), $q=0.5$ fixed:**

| $\tau$ | throughput | accuracy |
|---|---|---|
| 0.1 | 278.7 tok/s | 93.8% |
| 0.6 | 508.3 tok/s | 92.3% |
| 0.9 | — | 89.9% |

**Halt threshold $q$ (how long to keep refining), $\tau=0.2$ fixed:**

| $q$ | throughput | accuracy |
|---|---|---|
| 0.1 | 455.3 tok/s | 93.3% |
| 0.9 | 309.0 tok/s | 93.8% |

Two separate dials: $\tau$ trades accuracy for parallelism, $q$ trades speed for depth. Note $\tau$ is the harsher one — pushing it to 0.9 costs 3.9 points.

### The quality–efficiency frontier, and where it loses

94 $(q,\tau)$ settings for ALoDLM vs 21 entropy settings for WeDLM.

- The frontiers **cross near 650 tok/s**. ALoDLM wins in the high-accuracy regime; **WeDLM wins at the fastest operating points**. This is an honest negative and worth holding on to.
- At matched 93.25% accuracy: 612.4 tok/s vs 564.2 — a 8.5% throughput gain.
- On estimated GFLOPs per generated token, ALoDLM is above WeDLM across the *entire* range: 133.5 vs 154.6 GFLOPs/token at 93.25%, a 13.6% reduction.

The gap between the 8.5% wall-clock gain and the 13.6% arithmetic gain is the engineering tax: kernel efficiency, memory traffic, CUDA-graph padding.

### Does depth actually learn to follow difficulty?

Mean first-pass halting probability by token type, averaged over GSM8K, MATH-500, MBPP, HumanEval:

| token type | mean $\lambda^{(1)}$ |
|---|---|
| **numerical** | **0.369** |
| cross-dataset mean | 0.417 |
| word | 0.428 |

Numbers are 11.5% *less* likely to halt early than average — the gate learned to spend more passes on digits. **No difficulty labels were given.** It falls out of jointly optimising prediction and halting. This is the paper's cleanest evidence that the mismatch story is the right diagnosis, not a post-hoc rationalisation.

Also: after the early training phase, deeper readouts show *lower* training loss on their active tokens — refinement is genuinely progressive, not just extra churn.

### Recurrent depth as a test-time dial

Turning $q$ up raises mean loops/token from 1.6 → 2.34, and the 11-benchmark average from 77.9% → 79.1%. So depth is a usable [[Test-Time Compute|test-time compute]] axis, +1.2 points for ~46% more loops. Diminishing, but real.

### Maximum depth $K$ — more is not better

$K \in \{2,4,8\}$ at 8B, matched settings, 6-benchmark mean:
- $K=2$ and $K=4$ improve faster early; $K=8$ lags.
- $K=2$ **plateaus lower**.
- $K=4$ catches $K=8$ later in training.

So $K=4$ at half the max depth. $K=8$ buys nothing — the extra depth is harder to train, not more capable.

### Loop placement matters, and it is not where you'd guess

Both cores are 16 layers of Qwen3-8B:
- **middle** $[10,26)$ **beats last** $[20,36)$ later in training.

Where you put the recurrence changes the result even at identical layer count. Middle-layer recurrence won, which is why the 8B uses a 10-layer prelude and a 10-layer coda rather than looping the tail. (A full-stack $[0,36)$ curve appears but on a different benchmark subset — not a clean comparison.)

### Variance reduction, measured

Removing intermediate supervision raises conditional gradient variance (4,096 denoiser normalisation params, paired exit trajectories, 8 fixed input packs):

| step | variance vs full method |
|---|---|
| 1,000 | 1.76× |
| 6,500 | 1.49× |
| 17,000 | 1.42× |

So the trick matters most early and stays worth ~1.4× throughout. Note what is measured: variance across *exit-trajectory sampling*, not across training examples. Also note the weighted denoiser loss is **not** claimed to have the same expected gradient as sampled-exit supervision — it is a deliberately biased variance trade, justified empirically rather than proved.

## Worth Remembering

**Limitations the authors admit:**

1. **Time to first token can be worse than AR.** During prefill the KV cache is built at *full* recurrent depth. For short answers or latency-sensitive use, the parallel-decode win can be eaten entirely by prefill.
2. **Speed is input-dependent, even greedy.** How much recurrence runs and how many tokens commit together depend on confidence and gate decisions. Throughput swings by prompt, dataset and domain. On domains underrepresented in the 5B-token SFT corpus, the model loops more and commits less — and the speed advantage over an optimised AR baseline can **vanish or reverse**. That is a serious production caveat: you cannot quote a single tokens/sec number.
3. **WeDLM still wins at the very fastest operating points.** ALoDLM is the better choice when you care about quality per FLOP, not when you want maximum raw speed.

**Things to notice:**

- The **5B-token SFT, no CPT** result is remarkable on its own. SDAR and WeDLM both pay for continued pretraining; ALoDLM skips it and still wins. Either looping is a strong inductive prior for the mask-recovery task, or the Qwen3 initialisation carries more than people assumed.
- The depth-aware KV cache has **approximations baked in**. The deepest-available fallback and the shared coda buffer mean served outputs need not match recomputing every prefix token at every depth. There is no reported measurement of how much this drifts from exact. Something to check before trusting serving numbers as the model's true quality.
- **Reported layer counts are not FLOPs per token.** The appendix is explicit: a window can be processed several times, a pass can commit several tokens, already-committed positions still cost compute as context, and with batching the whole batch runs to the deepest active stopping depth. Do not infer speed from the commitment-depth histogram.
- The **$\operatorname{sg}$ on the exit gate's input** is load-bearing and easy to miss. The gate reads the backbone but does not train it. The backbone is trained only by prediction loss; the gate only by the REINFORCE term. This keeps the two learning problems from fighting.
- The paper is honest that the relaxed KL $(\alpha,\beta)=(0.1,1)$ is a **rollout-based relaxation whose expectation need not equal $R_{\alpha,\beta}$**, because the depth profiles depend on the sampled history. The unbiasedness proof covers the original joint-KL objective, not what they actually run. Nice to see that stated rather than buried.

**Open questions:**

- Does the token-adaptive depth story hold with [[RLHF|RL]] post-training, or does the gate collapse to a constant once a reward model is in the loop?
- The gate uses the *mean* cumulative halt probability over unresolved positions to stop the inner loop. That is a collective decision — one very hard token cannot keep the loop alive alone. Would a max or quantile rule help on long-chain arithmetic?
- Middle-layer recurrence beating last-layer is unexplained. Is it that early layers must build features before refinement is meaningful, and late layers must still exist un-looped to decode? That would predict an optimal prelude/core/coda ratio worth measuring.
- Everything is on top of WeDLM's block-diffusion framing with causal attention. How much of the result is the looping and how much is that substrate? There is no ALoDLM-on-plain-masked-diffusion control.

**Connections:** the inner loop with discrete-commit-or-keep-latent is structurally close to [[Diffusion Sampling|iterative refinement]] but per-token; the halting policy is a one-armed stopping problem trained by [[Simple Statistical Gradient-Following Algorithms (REINFORCE)|REINFORCE]] with a baseline, so everything you know about [[Policy Gradient|policy-gradient]] variance applies directly; and the latent variable + variational bound + score-function gradient stack is the [[Auto-Encoding Variational Bayes (VAE)|VAE]] recipe applied to a discrete schedule instead of a continuous code.

## Links

Related: [[Discrete Diffusion]] · [[Diffusion Models]] · [[Denoising Objective]] · [[Forward Diffusion Process]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Latent Variable Models]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[Policy Gradient]] · [[KL Divergence]] · [[Credit Assignment]] · [[Mixture of Experts]] · [[KV Cache]] · [[Test-Time Compute]] · [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Causal Attention]] · [[Auto-regressive models]] · [[Speculative Decoding]] · [[Looping Beyond Twice- A Scalable Recipe for Looped Mixture-of-Experts]] · [[Sparsely-Gated Mixture-of-Experts Layer]] · [[Register Tokens for Bounded-State Reasoning in Diffusion Language Models]] · [[Perplexity]]

New topics worth writing: Looped Transformers, Adaptive Computation Time, PonderNet, Universal Transformers, Block Diffusion, Self-Conditioning in Diffusion, Score-Function Estimator vs Reparameterisation, Control Variates for Policy Gradients, Mixture-of-Recursions, Depth-Adaptive Transformers, Early-Exit Networks, Masked vs Uniform Discrete Diffusion, WeDLM, SDAR, Confidence-Based Parallel Decoding, Diffusion-Draft AR-Verify Decoding, Predictive Entropy as a Commitment Signal
