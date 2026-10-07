---
title: "MemFold: Learning Compact Soft Memory for Long-Context Personalization via On-Policy Optimization"
authors: ["Jingxuan Wu", "Yuzhe Yang", "Yiqiao Huang", "Chengzhi Liu", "Qingni Wang", "Chengxuan Qian", "Shutong Wu", "Jiawei Zhang", "Xin Eric Wang"]
year: 2026
arxiv: "2609.36435"
url: https://arxiv.org/abs/2609.36435
priority: Good-To-Read
read_on: 2026-10-05
tags: [paper, transformers, rl, vision, theory]
---
## The Core Idea

A long-running assistant has to answer *this* user, using what *this* user said earlier. The hard part is not storing the history. The hard part is acting on it — remembering that the user said "I hate fitness trackers" three months ago and *not* recommending a Fitbit today.

Two ways to carry that history forward:

1. **Keep it as text.** Readable, editable. But the text grows with the history, and the model re-reads it every single turn.
2. **Squash it into a fixed number of vectors.** The reader always sees exactly $K$ vectors, no matter how long the history got.

> [!NOTE] Soft memory
> A fixed-size block of $K$ continuous vectors, fed into the model at the same place word embeddings go. It is not text — you cannot read it. It is a learned summary that lives directly in the model's input space. ^soft-memory

Option 2 bounds the cost. The problem is **how everyone trains it**. Compressors are trained to either (a) reconstruct the original text from the vectors, or (b) match a reference answer written by someone else. Both of these score the model on sequences *the model never wrote*. So they never touch the actual failure: the reader, given only the squashed memory, drifts away from what the text would have told it.

MemFold's move is to judge the compressed memory **by the behaviour it supports**, not by the text it can rebuild. Train the reader on its own samples, with two signals:

- **A task reward** (did you get the answer right?) via [[GRPO|group-relative advantages]].
- **A per-token comparison**: take the exact response the compressed-memory reader just wrote, and ask a frozen copy of the same model — one that still has the *full text* memory — how likely each of those tokens was. Where the text-reader was more confident than the compressed-reader, push the compressed-reader up.

That second signal is the clever bit, and it is clever because of what it is *not*. It is not standard distillation. The teacher is never sampled from. The teacher is not smarter — it is literally the same weights, the only difference being that it reads text where the student reads vectors. So the signal is not "copy a better model"; it is **"here is exactly where compression cost you confidence"**, measured on the student's own trajectory.

Why this did not exist before: a scalar reward tells you a 300-token answer was wrong, but not *which span* was wrong. The characteristic failure here is a fluent, on-topic answer where one bullet point out of five violates a preference. A sequence-level reward charges the whole answer for that one bullet. The token-level confidence gap localises it.

What it unlocks: a fixed $K$-vector interface that holds up as the history grows from 32K to 128K tokens — accuracy actually goes *up* at the longer length, while the reader's interface stays the same size.

## The Methodology

### The three roles, one adapter

Everything shares one frozen backbone $\theta_0$ and **one** [[LoRA]] adapter $\theta$. The adapter plays three parts:

| Role | Input | Output |
|---|---|---|
| Writer | history $C$ + query $x$ | textual memory $M$ |
| Reader | query $x$ + soft memory $Z$ | answer $y$ |
| Teacher (frozen snapshot) | query $x$ + textual memory $M$ | log-probs only |

### Step 1 — write the textual memory

$M = e_\theta(C, x)$ — a query-conditioned structured note with three fields:

- **Evidence**: grounded facts from the history.
- **Temporal relations**: "first preferred X, later shifted to Y."
- **Derived facts**: "currently prioritises child-led activities."

Initially these notes are extracted by GLM-5.2 from history + question (never the answer). The adapter is then trained to write them itself, so the external model is gone at inference.

### Step 2 — compress to $K$ vectors

$$Z = \mathcal{C}_\phi(E(M)) = P_\phi\big(\mathrm{Comp}_\phi(Q_K, E(M))\big) \in \mathbb{R}^{K \times d}$$

- $E$ = the **first four transformer blocks** of the frozen backbone. Not the last layer.
- Sequence processed in 2048-token chunks, every 32 consecutive hidden states mean-pooled.
- $\mathrm{Comp}_\phi$ = a 2-layer Perceiver resampler, $K$ learned latent queries, latent dim 768, 12 heads.
- $P_\phi$ projects into the reader's embedding space.
- Default $K = 256$.

**Why layer 4?** They ran a probe-free retrieval test on 3,347 memory records: can this layer's states tell apart competing attribute values and temporal updates? Layer 4 won on all three backbones (AUC 100.0 for Qwen2.5-3B vs 51.76 at the final layer). The final layer is *worse than chance* at distinguishing preference updates. Late layers have already thrown away the distinctions and specialised for next-token prediction.

### Step 3 — four separate initialisation stages

The interface has to be *readable* before on-policy training can do anything. Four stages, each a different objective:

**(a) Reconstruction.** Frozen decoder rebuilds the textual memory from a compressed *raw history* prefix:
$$\mathcal{L}_{\mathrm{rec}} = -\frac{1}{|M|}\sum_{t} \log p_{\theta_0}(m_t \mid x, Z_h, m_{<t})$$
Plus a separation hinge pushing apart mean-pooled representations from different users: $\mathcal{L}_{\mathrm{sep}} = [\sqrt{2-2\kappa} - \|u_i - u_j\|_2]_+$ with $\kappa = 0.8$.

**(b) Representation warmup.** No decoder at all. Two cached views per context, and three terms: cross-context separation + same-context alignment (cosine) + a Gram-matrix decorrelation penalty, the last two weighted 0.1. This is [[VICReg|variance-invariance-covariance]]-flavoured — it is explicitly guarding against [[Mode Collapse|collapse]] of the memory vectors.

**(c) Auxiliary reasoning adaptation.** Now with a reasoning target $s_i$ and a *ranking* loss: the right memory should make the right reasoning cheaper than someone else's memory does.
$$\mathcal{L}_{\mathrm{rank}}(i,j) = [\delta + \ell_i(Z_i) - \ell_i(Z_j)]_+$$
Question, options and target are held fixed; only the memory is swapped. Uses a throwaway rank-8 LoRA that is discarded afterwards.

**(d) Reader initialisation.** Train the reader to answer from its own self-written memories, with an anchoring term to stop the compressor drifting:
$$\mathcal{L}_{\mathrm{anchor}} = \frac{\frac{1}{Kd}\|Z - Z^0\|_F^2}{\max(\frac{1}{Kd}\|Z^0\|_F^2, 10^{-8})}$$
weighted 0.1. The result is $\theta_{\mathrm{init}}$ — which is *also* frozen and reused as the teacher $\pi_T$.

### Step 4 — on-policy optimisation

Sample $G$ responses per query from $\pi_\theta(\cdot \mid x, Z)$. Two losses on the same rollouts.

**The reward half** — standard [[GRPO]]:
$$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}, \qquad \rho_{i,t} = \frac{\pi_\theta(y_{i,t}\mid x,Z,y_{i,<t})}{\pi_{\mathrm{old}}(y_{i,t}\mid x,Z,y_{i,<t})}$$
$$\mathcal{L}_{\mathrm{GRPO}} = -\mathbb{E}\left[\frac{1}{G}\sum_i \frac{1}{|y_i|}\sum_t \min(\rho_{i,t}\hat{A}_i,\ \mathrm{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_i)\right]$$

PersonaMem reward is exact-match on the chosen option. LoCoMo uses $r = 0.75 \cdot \frac{2O}{|S_y|+|S_a|} + 0.25 \cdot \mathbb{1}\{S_y = S_a\}$ — token-F1 plus an exact-match bonus.

**The distillation half** — the new part. Score the student's own tokens twice:
$$\ell_{T,i,t} = \log \pi_T(y_{i,t}\mid x, M, y_{i,<t}), \qquad \ell_{\theta,i,t} = \log \pi_\theta(y_{i,t}\mid x, Z, y_{i,<t})$$

Build a detached sigmoid gate on the gap, $\beta = 5.0$:
$$\bar{g}_{i,t} = \mathrm{sg}\big[\sigma(\beta(\ell_{T,i,t} - \ell_{\theta,i,t}))\big]$$
$$\mathcal{L}_{\mathrm{OPD}} = -\mathbb{E}\left[\frac{1}{G}\sum_i \frac{1}{|y_i|}\sum_t \bar{g}_{i,t}\,\ell_{\theta,i,t}\right]$$

> [!NOTE] Confidence-gated on-policy distillation
> Weight the student's own sampled tokens by how much *more* confident a text-reading copy of itself was about those same tokens. The gate is in $(0,1)$: near 1 where the teacher was much more confident, exactly $1/2$ where they agree, near 0 where the student was more confident. Nothing the student did not sample is ever promoted. ^confidence-gate

Joint: $\mathcal{L} = \lambda_{\mathrm{GRPO}}\mathcal{L}_{\mathrm{GRPO}} + \lambda_{\mathrm{OPD}}\mathcal{L}_{\mathrm{OPD}}$. No extra KL term. Teacher and compressor frozen throughout; only the adapter moves. At inference the teacher is deleted entirely.

### Why the gate, and not the raw log-ratio

The gradient of the surrogate is simply
$$\nabla_\theta \hat{\mathcal{L}}_{\mathrm{OPD}} = -\hat{\mathbb{E}}_\mathcal{B}[\bar{g}_{i,t}\nabla_\theta \ell_{\theta,i,t}]$$
Gate-weighted negative log-likelihood on the student's own tokens. Nothing flows through the teacher.

Subtract the constant $1/2$ (free, by the score-function identity $\mathbb{E}_{a\sim p_\theta}[\nabla_\theta \log p_\theta(a)] = 0$) and the expected update becomes
$$\mathcal{G}(\theta;h) = -\tfrac{1}{2}\mathbb{E}_{a\sim p_\theta}\left[\tanh\!\left(\tfrac{\beta\Delta(a)}{2}\right)\nabla_\theta \log p_\theta(a)\right]$$

Compare with reverse [[KL Divergence|KL]], whose gradient is $-\mathbb{E}[\Delta(a)\nabla_\theta \log p_\theta(a)]$. **Same shape, but $\Delta$ replaced by $\tanh(\beta\Delta/2)$.** For small gaps it *is* reverse KL scaled by $\beta/4$ (error bounded by $\frac{\beta^3}{48}\mathbb{E}[|\Delta|^3\|\nabla\log p_\theta\|]$). For large gaps it saturates, capping any single token's influence at $1/2$.

That cap is deliberate. $\pi_T$ is a *reference reader*, not an accuracy oracle — its own task accuracy is well below the final student's. A raw log-ratio would let one token where the frozen teacher is wildly over-confident dominate the step.

There is a third property worth holding onto. For a softmax policy, the descent direction over logits is
$$-\nabla_z \hat{\mathcal{L}}_{\mathrm{OPD}} = \frac{1}{n}\sum_j \bar{g}(a_j)(e_{a_j} - p_\theta)$$
Any token *not sampled* has its logit decrease. Compare forward-KL distillation, whose direction is $q - p_\theta$: that promotes tokens the teacher likes even when $p_\theta(b)\approx 0$. MemFold only **re-ranks the student's own candidates**; it never imports the teacher's choices from outside the student's support. In expectation each logit change is scaled by $p_\theta(b)$ itself.

### Hyperparameters that mattered

- LoRA rank 16, $\alpha$ 32, dropout 0.05, on Q/K/V/O/gate/up/down.
- On-policy learning rate $3\times10^{-7}$ — three orders of magnitude below the init stages at $1\times10^{-5}$.
- PersonaMem: $G = 8$, $T = 1.0$, top-$p$ 0.98, **max 5 response tokens** (multiple choice), $\lambda_{\mathrm{OPD}} = 0.02$, $\lambda_{\mathrm{GRPO}} = 0.3$.
- LoCoMo: $G = 4$, $T = 0.8$, top-$p$ 0.95, 64 tokens, both $\lambda = 1.0$.
- Textual memories are generated **once** by $\theta_{\mathrm{init}}$, cached, and never refreshed during RL.
- BF16 backbone, FP32 for LoRA and log-prob computation. 4×H200 for the main stages.

## Ablation Studies and Experiments

### Main numbers

Three backbones, four benchmarks. Trained on PersonaMem-32K (and LoCoMo separately), evaluated zero-shot on PrefEval and LongMemEval.

| Backbone | Method | PM-32K | PM-128K | PrefEval | LongMemEval |
|---|---|---|---|---|---|
| Qwen2.5-3B | Full Text | 46.0 | 21.9 | 12.9 | 26.6 |
| | xRAG | 36.0 | 55.8 | 8.5 | 10.2 |
| | MemGen | 54.0 | 66.1 | 13.3 | 3.8 |
| | GRPO | 68.0 | 58.4 | 11.3 | 26.0 |
| | OPSD | 54.0 | 29.6 | 12.8 | 26.2 |
| | **MemFold** | **70.0** | **88.4** | **19.9** | **32.4** |
| Qwen2.5-7B | Full Text | 60.0 | 24.0 | 14.0 | 25.4 |
| | MemGen | 76.0 | 78.5 | 14.1 | 11.0 |
| | GRPO | 70.0 | 62.2 | 14.1 | 25.0 |
| | **MemFold** | **88.0** | **94.4** | 14.1 (tie) | **36.8** |
| Qwen3-4B | Full Text | 56.0 | 4.3 | 13.6 | 27.8 |
| | GRPO | 62.0 | 65.7 | 13.8 | 27.4 |
| | OPSD | 74.0 | 39.5 | 13.8 | 26.0 |
| | **MemFold** | **84.0** | **89.4** | **15.2** | **38.6** |

Two things to notice, both more interesting than the wins themselves.

**The margin widens with history length.** At 32K, MemFold beats the best baseline by 2–12 points. At 128K, by 16–24. Several baselines *collapse* at 128K (OPSD: 74.0 → 39.5). Full Text on Qwen3-4B goes to 4.3% — the model simply cannot find the relevant turn in 124K tokens. MemFold's reader sees the same 256 vectors in both settings; only the history behind them grew.

**It is not buying accuracy with inference tokens.** End-to-end token-equivalent counts are *lower* in aggregate than full-context inference for all three backbones (52,662 vs 58,803 weighted average on 3B). Exception: PrefEval, where histories are short enough that memory-construction cost is not amortised.

### The component ablation

Qwen3-4B, mean over 16 samples plus Pass@16:

| Variant | PM-32K Mean | PM-128K Mean |
|---|---|---|
| **MemFold (full)** | **75.4** | **87.9** |
| w/o OPD | 71.9 | 86.9 |
| w/o GRPO | 58.6 | 70.2 |
| w/o writer init | 70.8 | 80.0 |
| w/o reader init | **47.9** | **46.2** |
| Init only (no RL) | 60.5 | 70.8 |
| Text-space control | 69.3 | 85.5 |

**Reader initialisation is the single biggest lever** — removing it drops 32K accuracy from 75.4 to 47.9, worse than no RL at all. The model must learn to *consume* the soft interface before any policy optimisation can help. This is the practical lesson: a soft-memory interface you have not taught the reader to read is not a weak interface, it is a broken one.

**GRPO does most of the task work** (75.4 → 58.6 without it). **OPD adds a smaller, consistent gain** (75.4 → 71.9 at 32K; 87.9 → 86.9 at 128K). The authors are honest that OPD is the second-order term. It is not the thing that makes this work; it is the thing that sharpens it.

**The negative result that matters most:** the *text-space control* — same recipe, no compression, memory stays as text — scores 69.3 / 85.5, i.e. **worse than the soft version**. So fixed-budget compression is not the bottleneck here. You might expect squashing text into 256 vectors to cost you something; under this recipe it does not, and may act as a useful filter.

### Budget sweep

$K \in \{64, 128, 256, 512\}$, Qwen3-4B. **Not monotonic.** Single-peaked at $K=256$, nearly flat from 128 up, and it *declines* at 512. Only $K=64$ is clearly under-provisioned. The 128K curve is flatter than the 32K one — consistent with the writer already discarding most of the history before compression, so extra vectors change *how much* survives more than *what* survives.

The authors read the decline at 512 as a property of their recipe (the compressor and reader were initialised at a fixed budget and not retuned per $K$) rather than evidence that more vectors carry less. Fair, but it does mean $K$ is not a free dial.

**Cost barely moves with $K$.** Under 1% variation on the longer benchmark, and not monotonic. Going 64 → 512 adds 448 vectors — about a third of the measured token difference on 32K; most of the rest is generated tokens. Every configuration still reads the full history *once* to write the textual memory, and on 128K that pass is all but a fraction of a percent of the per-instance cost.

That is the honest limitation of the whole approach: a compact memory makes the **reader's** interface history-independent, but does not remove the cost of reading the history to build the memory.

### Memory interventions — is the reader using it?

Replace the matched soft memory with (a) a shuffled memory from another user, or (b) null memory. Substantial accuracy drops on both datasets. So the reader depends on the *instance-specific content*, not task priors.

The authors are careful about what this does **not** show: it does not establish *which parts* of the memory are used, and specifically does not show the reader resolves the correct time-dependent version of a preference when several exist in the history. That is the actual claim of the paper and the intervention does not test it.

Also measured here: **the textual-memory teacher's own task accuracy is well below the final student's.** A student beating its teacher is only coherent because of how OPD uses it — bounded, gated, re-ranking only sampled tokens. If they had used forward KL or sampled from the teacher, this would not have worked.

### Training efficiency

Qwen2.5-3B on PersonaMem-32K, 369 optimiser updates, five configurations (GRPO, OPSD, SDPO, GRPO+OPD on full history, MemFold). MemFold climbs faster per update and reaches comparable accuracy with fewer student rollouts. The reward and distillation terms **reuse the same rollouts**, and the teacher only *scores* tokens — no autoregressive decoding.

Caveat the authors flag themselves: these are complete method configurations with different inputs and initialisations, not a controlled loss-only ablation. The curves also exclude initialisation and teacher forward passes from the cost. One training run per method; the error bars come from bootstrapping test questions and 12 answer-option permutations, **not** training seeds. 369 updates on 4 GPUs took ~12.9 minutes.

### What did not work in the baselines

- **AutoCompressor produces no valid output** on PrefEval (returns a single prediction for everything) and malformed output on LongMemEval with the 3B backbone.
- **MemGen collapses out of domain**: 66.1 on PM-128K, then 3.8 on LongMemEval with the 3B backbone. Worse than doing nothing.
- **GRPO and OPSD sometimes underperform the untrained Full Text baseline on PrefEval**, suggesting they overfit to source-domain input format rather than learning transferable preference-following.
- **xRAG is the cheapest but pays heavily**: 8.5 on PrefEval, 10.2 on LongMemEval with 3B.

### The qualitative example

PrefEval, implicit persona. User history implies "I dislike wearable technology and fitness trackers." Query asks for ways to monitor fitness progress.

- **GRPO**: lists sensible things, then "Use a Fitness Tracker: Wearables like Fitbit or Apple Watch…" — 300 tokens, violates the preference.
- **OPSD**: same failure, different wording.
- **MemFold**: journaling, body measurements, progress photos — 96 tokens, respects it.

This is the failure mode the whole method is built around: fluent, topical, with the violation confined to one span. A sequence-level reward charges all 300 tokens for it. Worth noting the authors explicitly say this illustrates a *pattern*, not its frequency.

## Worth Remembering

**The teacher is a *reference*, not an oracle.** This is the conceptual inversion. Every distillation paper you have read uses a stronger teacher and a signed gap pulling the student toward it. Here the teacher is the student's own initialisation, differing *only* in its memory input, and it is measurably worse at the task. The signal is "what did compression cost you", not "copy someone better." The closest precedent is context distillation (Snell et al. 2022), but with a non-negative gate replacing the signed gap.

**Three things the gate buys you, which are easy to conflate:**
1. Bounded per-token influence (the $\tanh$ saturation).
2. Support preservation — unsampled tokens are never promoted (the softmax logit argument).
3. Automatic vanishing — when soft and text readings agree, $\Delta = 0$, gate $= 1/2$, and the zero-mean baseline kills the expected update.

**Limitations the authors admit:**
- The writer is bootstrapped from GLM-5.2 extractions. Not needed at inference, but the quality of the initial textual memory depends on a stronger external model. Learning the writer unsupervised is future work.
- Qwen only, up to 7B. No other families, no larger scale.
- The theory is *local and conditional*. It does not guarantee training reaches $p_\theta \approx q$, that attenuation is monotone, or that $\lambda_{\mathrm{OPD}}$ needs no tuning.

**Caveats the paper buries in appendices:**
- The decomposition $\nabla\hat{\mathcal{L}}_{\mathrm{OPD}} = -\hat{\mathbb{E}}[(\bar{g}-\tfrac12)\nabla\ell_\theta] - \tfrac12\hat{\mathbb{E}}[\nabla\ell_\theta]$ has a baseline term that is zero-mean **only under exact sampling**. Under nucleus sampling it becomes a self-imitation term toward the *truncated* distribution, which sharpens $p_\theta$. PersonaMem rollouts ($T=1.0$, top-$p$ 0.98) are close to exact; LoCoMo ($T=0.8$, top-$p$ 0.95) is not, and the effect is larger there. Per-sequence $1/|y_i|$ normalisation also prevents exact cancellation.
- PersonaMem rollouts cap responses at **five tokens**. Multiple choice. So the headline in-domain result is a *classification* result, and the token-level localisation story — the one motivating the whole method — is only really exercised on LoCoMo and the open-ended OOD benchmarks.
- Test sets are small: 50 questions on PM-32K, 233 on PM-128K. A 2-point gap on 50 items is one question.
- PrefEval and LongMemEval are scored by a Gemini API judge.

**Practical caveats if you wanted to use this:**
- The four-stage initialisation is most of the engineering. Reader init is non-negotiable (the ablation is brutal). Expect to spend your time there, not on the RL.
- The layer-4 result generalises beyond this paper: if you need hidden states that distinguish *competing versions of a fact*, read early, not late. Final-layer states scored below chance on update discrimination. Worth remembering whenever you build [[Embeddings|embeddings]] off an LLM for anything update-sensitive.
- Textual memories are cached once from $\theta_{\mathrm{init}}$ and never refreshed during RL, but regenerated by the final $\theta$ at eval. There is a train/test mismatch baked in here that the paper does not probe.
- You still pay a full history read per instance. If your bottleneck is that read, this does not help you.

**Connections worth chasing:** the structure is a [[Policy Gradient|policy gradient]] with a dense auxiliary term, and the $\tanh$-saturated reverse KL is the same bounded-influence trick as [[PPO|PPO's]] clipping — cap how far one sample can move you. The "optimise the compressed representation by downstream behaviour, not reconstruction" framing is close in spirit to [[JEPA|JEPA's]] predict-in-representation-space argument, and the collapse-guarding init stages are borrowed straight from the SSL playbook.

**Open question:** the text-space control *losing* to the soft version is the most intriguing result in the paper and gets one sentence. If compression is acting as a useful filter rather than a lossy bottleneck, that reframes the whole fixed-budget argument — the budget would be a regulariser, not a cost.

## Links

Related: [[LoRA]] · [[GRPO]] · [[KL Divergence]] · [[Policy Gradient]] · [[PPO]] · [[Distillation]] · [[Distilling the Knowledge in a Neural Network]] · [[On-Policy or Off-Policy Learning- A Systematic Study of Distillation Dynamics]] · [[Do We Really Need KL Divergence for On-Policy Distillation of Large Language Models]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[What Does Privileged Information Add to On-Policy Self-Distillation]] · [[Learning from Teacher Continuations at Student States]] · [[1% of Tokens Can Be Enough- On Gradient Estimation in On-Policy Distillation]] · [[LatentPress- Context Compression Beyond Text and Vision]] · [[Memory]] · [[Context Engineering]] · [[Long Context]] · [[Lost in the Middle]] · [[Credit Assignment]] · [[PACT- From Credit Assignment to Critic Alignment]] · [[Embeddings]] · [[JEPA]] · [[VICReg]] · [[Mode Collapse]] · [[RAG]] · [[Matryoshka Representation Learning]] · [[Prompt Caching]]

New topics worth writing: Perceiver resampler / latent-query cross-attention, prompt and prefix tuning, soft prompts as a learned interface, context distillation, gist tokens, PersonaMem and preference-following benchmarks, layer selection for probing hidden states, reverse-KL saturation and bounded-influence surrogates, score-function identity as a baseline trick, nucleus sampling and self-imitation bias in on-policy losses, token-equivalent cost accounting for compressed-context systems
