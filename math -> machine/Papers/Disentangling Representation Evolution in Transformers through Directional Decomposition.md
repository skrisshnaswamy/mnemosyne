---
title: "Disentangling Representation Evolution in Transformers through Directional Decomposition"
authors: ["He et al."]
year: 2026
arxiv: "2609.15975"
url: https://arxiv.org/abs/2609.15975
priority: Good-To-Read
read_on: 2026-09-19
tags: [paper, transformers]
---
## The Core Idea

Every transformer layer works by addition. You have a hidden state $\mathbf{z}$, a sub-layer (attention or MLP) computes an update $\Delta$, and you write $\mathbf{z}^{l+1} = \mathbf{z}^l + \Delta^l$. That is the residual stream.

The trick here is to split $\Delta$ into two pieces by direction, relative to the state it is being added to:

$$\Delta_{\parallel} = \frac{\Delta \cdot \mathbf{z}}{\|\mathbf{z}\|^2}\mathbf{z} = \alpha\mathbf{z}, \qquad \Delta_{\perp} = \Delta - \Delta_{\parallel}$$

Substituting back gives the whole paper in one line:

$$\mathbf{z}^{l+1}_t = (1 + \alpha^l_t)\,\mathbf{z}^l_t + \Delta^l_{t,\perp}$$

So a layer does exactly two things. It **rescales** the incoming vector by a scalar $(1+\alpha)$, and it **rotates** it by adding something at right angles. Nothing else is possible.

> [!NOTE] Directional decomposition
> Split a layer's update into the part pointing along the current hidden state (parallel, pure magnitude change) and the part at right angles to it (perpendicular, direction change). The parallel part changes *how loud*; the perpendicular part changes *what*. ^directional-decomposition

The puzzle: measuring real models (Qwen3-4B-Instruct, Qwen3-30B-A3B), the parallel part is big. The ratio $\|\Delta_\parallel\|/\|\Delta_\perp\|$ often sits above 1 across depth. That is strange, because a pure rescale is free — the residual connection already carries $\mathbf{z}$ forward at zero parameter cost, and any scalar gain could be done by a LayerScale-style knob. Why would a 4096×4096 attention output projection spend capacity producing something a single scalar could produce?

The answer, from surgery on frozen models: **the parallel part is largely redundant, the perpendicular part is not**. Scaling $\Delta_\perp$ by anything other than 1 wrecks the model immediately. Scaling $\Delta_\parallel$ — even to zero — is nearly free in the right place.

The "right place" is the second contribution, and it is the subtle bit. Doing this inside **attention value space** rather than in the residual stream matters enormously. And within value space, you must first pull out the token's message to itself. That unlocks three things: an interpretability result (direction is what carries meaning), a compression diagnostic (only perpendicular error predicts quality loss), and a training intervention (banning parallel attention updates during pretraining makes models better).

## The Methodology

### Two places to cut

**Residual space.** $\Delta$ is the sub-layer output (attention out, MLP out, or the whole block), $\mathbf{z}$ is the state just before the addition. Straightforward.

**Attention value space.** Look *inside* attention, before the output projection $W_O$. Each source token $s$ gives a value vector $\mathbf{v}_s = W_V\mathbf{x}_s$, and the aggregate for query token $t$ is

$$\mathbf{o}_t = \sum_{s \le t} \mathcal{A}_{ts}\mathbf{v}_s$$

Here the natural reference is not the hidden state but $\mathbf{v}_t$ — the token's **own** value. That is the direction $\mathbf{o}_t$ would have if attention pulled in nothing from anywhere else. Residual space cannot see this structure, because $W_O$ smears the aggregate into the stream and destroys the alignment.

### The self-message problem, and exclude-self scaling

Naively decomposing $\mathbf{o}_t$ against $\mathbf{v}_t$ has a bug. The diagonal term $\mathbf{d}_t = \mathcal{A}_{tt}\mathbf{v}_t$ — the token attending to itself — is *exactly colinear* with $\mathbf{v}_t$ by construction. So it lands entirely inside $\mathbf{o}_{t,\parallel}$. Zeroing the parallel part therefore deletes the token's own identity along with everything else. That is not a test of redundancy, it is a lobotomy.

The fix is to split the sum first:

$$\mathbf{o}_t = \underbrace{\mathcal{A}_{tt}\mathbf{v}_t}_{\mathbf{d}_t \text{ (self)}} + \underbrace{\sum_{s<t}\mathcal{A}_{ts}\mathbf{v}_s}_{\mathbf{c}_t \text{ (context)}}$$

decompose only $\mathbf{c}_t$ against $\mathbf{v}_t$, scale those pieces, then put $\mathbf{d}_t$ back untouched:

$$\widetilde{\mathbf{o}}^{\mathrm{excl}}_t = \mathbf{d}_t + s^{(\parallel)}\mathbf{c}_{t,\parallel} + s^{(\perp)}\mathbf{c}_{t,\perp}$$

This is **exclude-self scaling**. It asks a precise question: is the part of *other tokens'* contribution that merely amplifies *this token's own* direction doing any work?

> [!NOTE] Exclude-self scaling
> Preserve the attention diagonal's direct self-message, and only edit the cross-token aggregate's components. It neither masks the self-attention edge nor renormalises the attention row — which is what makes it different from simply setting $A_{tt}=0$. ^exclude-self

### The unified intervention

Everything is one operation:

$$\widetilde{\mathbf{y}}_t = s^{(\parallel)}\mathbf{y}_{t,\parallel} + s^{(\perp)}\mathbf{y}_{t,\perp}$$

with $s^{(\parallel)}=s^{(\perp)}=1$ as the no-op. All edits are forward-pass hooks. Weights are never touched.

### The attention-map view

Because the self-message is the only thing on the diagonal, parallel-only scaling can be rewritten as a change to $A_{tt}$ alone, holding off-diagonals fixed. For full-aggregate value space the form is exact:

$$\Delta A_{tt} = (s^{(\parallel)}-1)\frac{\mathbf{o}_t^\top \mathbf{v}_t}{\|\mathbf{v}_t\|^2}$$

So "remove the parallel component" is literally "adjust how much each token attends to itself". This connects the geometry to the [[Query, Key, and Value (QKV)|attention]] literature on attention sinks and self-directed mass.

### Setup

- Models: Qwen3 (0.6B to 30B-A3B [[Mixture of Experts|MoE]]), Llama-3.2-3B, Gemma-3.
- Benchmarks: WikiText-2 perplexity, 7 zero-shot commonsense tasks, RULER long-context (13 tasks, 4k–12k).
- Compression: 4-bit AWQ, Wanda pruning at 50% (unstructured, 4:8, 2:4).
- Pretraining: GPT-style, from scratch, OpenWebText (296M/436M/528M) and FineWeb100BT (1.4B/2.7B). ~104.9B tokens, 200K steps, batch 256 × 2048, AdamW, lr $4\times10^{-4}$ / $3\times10^{-4}$, 2K [[GPU processing#Fix 2: warmup (this is the non-negotiable one)|warmup]], cosine decay, 8×H100 BF16, GQA.

Overhead is trivial: <2.5% prefill, <1.0% decode, 3.39% training.

## Ablation Studies and Experiments

### The scaling sweeps (Qwen3-1.7B, WikiText-2, $\Delta$PPL vs no-op)

Two sweeps: vary $s^{(\parallel)}$ with $s^{(\perp)}=1$, then the reverse.

**Perpendicular is hyper-fragile.** Any move away from 1 spikes perplexity. Setting $s^{(\perp)}=0$ is catastrophic — tens of thousands of PPL points in value space and residual attention, *millions* in the residual MLP.

**Parallel has a wide tolerance basin**, but how wide depends entirely on the site:

| Site | Behaviour under parallel scaling |
|---|---|
| Value space (exclude-self) | Flat. <1 PPL point over $s^{(\parallel)}\in[0,1]$; at most 1.3 points out to scale 3 |
| Residual (Attn) | Intermediate degradation |
| Residual (MLP) | Most fragile, collapses outside $[0,1]$ |

The lesson is sharp: robustness is *not* a property of "being parallel". It depends on which space you cut in and what you keep.

### Zero-shot tasks (Table 1, 7 benchmarks)

| Setting | Qwen3-1.7B Avg | Qwen3-30B-A3B Avg |
|---|---|---|
| Baseline | 61.61 | 73.42 |
| Attn Para-Rem. (residual) | 59.16 | 70.70 |
| **V-Para Rem. (naive full aggregate)** | **53.41** | **63.42** |
| V-Excl.-self | 60.10 | 73.31 |
| **Diag. Rem. ($A_{tt}=0$, renormalise)** | **39.17** | **40.68** |

Two failures worth internalising:

1. **Naive value-space removal loses 8–10 points.** This is the self-message bug in action — it deletes $\mathcal{A}_{tt}\mathbf{v}_t$ along with the cross-token parallel stuff.
2. **Hard diagonal removal is near-destruction** (39.17, close to chance on several tasks). Zeroing $A_{tt}$ forces the softmax row to renormalise over off-diagonal positions, which is a completely different and far more violent edit than the geometric one.

Exclude-self, meanwhile, costs 1.5 points at 1.7B and **0.1 points at 30B-A3B**, with slight gains on some individual tasks (WinoGrande 70.24 → 72.45, HellaSwag 77.79 → 78.13, RTE 81.95 → 83.39).

### Long context (RULER, Llama-3.2-3B)

| Method | $s_\parallel$ | 4k | 8k | 12k |
|---|---|---|---|---|
| Baseline | 1.0 | 86.91 | 82.09 | 79.30 |
| V-Full | 0.5 | 84.81 | 78.80 | 74.76 |
| V-Excl.-self | 0.5 | 86.69 | 80.80 | 76.88 |
| V-Full | 0.0 | 60.99 | 48.61 | 39.17 |
| V-Excl.-self | 0.0 | 75.64 | 67.97 | 62.02 |

Full removal costs more as context grows — the gap between the two value-space variants widens from 15 points at 4k to 23 at 12k. Preserving the self-message matters *more* when there is more context to aggregate.

Cross-family at $s_\parallel = 0.5$ on RULER-4k: Gemma-3-12B −0.2 (actually +0.1), Llama-3.2-3B −0.2, Qwen3-4B −0.2, Qwen3-8B −0.5. But small models break: Gemma-3-1B **−5.3**, Qwen3-0.6B **−4.8**, Qwen3-1.7B −3.2. The redundancy is a property of scale — small models apparently cannot spare the capacity.

### The counter-intuitive audit

You would assume value-space editing is safe because it is a *smaller* perturbation. Wrong. On Qwen3-0.6B, measured at the post-$W_O$ output:

| Geometry | Perturbation norm | Perturbation energy | Retained norm |
|---|---|---|---|
| Residual-space removal | 25.15% | 9.32% | 95.01% |
| Exclude-self value-space | **49.10%** | **25.89%** | 83.36% |

Value-space editing changes the attention output **nearly twice as much** — and preserves capabilities, while residual removal changes it half as much and hurts. Stability is about *what structure you preserve*, not about minimising Euclidean distance. That is the single most reusable finding in the paper.

The attention-map views agree: residual-space parallel removal implies wild signed diagonal swings from $-1.00$ to $+0.76$; exclude-self implies bounded, non-positive shifts.

### MLP-internal geometry (Appendix D)

The residual MLP looked fragile — but that was an artefact of *where* they cut. Decomposing the post-gating carrier $h = \phi(W_{\text{gate}}x)\odot W_{\text{up}}x$ instead of the final output, on Qwen3-0.6B, C4 $\Delta$loss:

| Geometry | No-Para | No-Perp |
|---|---|---|
| Residual MLP ($y$ rel. $x$) | +0.1165 | +15.96 |
| Grouped $h$ rel. $x$ | +0.0301 | +13.42 |
| Grouped $y_g$ rel. $h_g$ | **+0.0037** | +13.33 |

Cut in the right internal space and parallel removal is essentially free inside the MLP too. Downstream confirms it: Qwen3-1.7B Core-MC 55.19 → 55.28, MMLU 60.21 → 60.39, GSM8K-strict 68.76 → 68.08. No-Perp collapses MMLU to 24.40 — below random.

### Compression diagnostics

Let $e = \Delta_{\text{comp}} - \Delta_{\text{base}}$, decomposed against $\Delta_{\text{base}}$. On Qwen3-4B:

- **Perpendicular error cleanly ranks the methods** in exactly the order of downstream quality: AWQ < unstructured Wanda < 4:8 < 2:4.
- **Parallel error curves interleave** and rank nothing.
- Correlation of total error with perpendicular error: $r \ge 0.97$, $\rho \ge 0.90$. Perpendicular accounts for 87.5%–98.0% of layer distortion.
- Spearman of total with *parallel* error: 0.596 (quantization), 0.810 (unstructured), **−0.135** (4:8), **−0.432** (2:4). Negative.

This explains why plain isotropic $L_2$ reconstruction — what GPTQ, AWQ and friends actually minimise — is a leaky proxy. It charges you the same for a benign gain change as for a semantic rotation.

### Training-time intervention

Enforce $s^{(\parallel)}=0$ during both training and eval (no train/test mismatch). Validation loss on OpenWebText drops below baseline from *early* training and stays down at 296M, 436M, 528M.

| Model | Setting | Avg (6 tasks) | $\Delta$ |
|---|---|---|---|
| 1.4B | Baseline | 58.5 | — |
| | Attn Para-Rem. | 58.8 | +0.3 |
| | V-Para Rem. | 59.2 | **+0.7** |
| 2.7B | Baseline | 60.2 | — |
| | Attn Para-Rem. | 60.9 | +0.7 |
| | V-Para Rem. | 61.7 | **+1.5** |

Gains are uniform across ARC-E, BoolQ, HellaSwag, OBQA, PIQA, WinoGrande — not one task carrying the average. The gain also *grows* with scale (+0.7 → +1.5).

### What did not work

- **Naive full-aggregate value removal at inference** — 8–10 point drop. The self-message must be preserved.
- **Hard diagonal removal** ($A_{tt}=0$ + renormalise) — near-chance performance. Softmax renormalisation is the culprit, not the geometry.
- **Learned gating instead of hard removal.** They tried $s_\parallel = \sigma(w^\top x)$, a tokenwise learned gate, at 1.4B. It converged to *higher* loss than fixed $s_\parallel=0$, and the fixed version beat both it and baseline throughout. The parallel degree of freedom is not something the model should be allowed to choose — it is just extra freedom to waste.
- **Editing small models.** Gemma-3-1B and Qwen3-0.6B lose ~5 points on RULER even at $s_\parallel = 0.5$.
- **Parallel error as a compression signal** — uninformative, sometimes anti-correlated.

### Head-level structure (Appendix C)

Single-head parallel removal produces signed shifts in both directions — some heads' parallel components *hurt*, removing them helps. Joint all-head removal is **sub-additive**: $\Delta_{\text{joint}} < \sum_h |\Delta_h|$, staying inside the single-head min–max envelope. Opposing head pulls cancel. Only layer 0 spikes under joint removal, suggesting early heads collectively anchor the parallel coordinates.

## Worth Remembering

**Preserved structure beats small perturbation.** The audit table is the headline. A 49% output perturbation that keeps the self-identity routing intact is safer than a 25% perturbation that disturbs residual propagation. Any time you are choosing an edit, pruning criterion, or quantization objective by "how much does this change the activations", this is the counterexample.

**Why parallel components exist at all, unexplained.** The paper shows they are redundant and that removing them helps, but does not say *why* optimisation produces them. Plausibly attention has no mechanism to avoid producing them — token mixing and self-amplification are structurally entangled through the same $\mathcal{A}\mathbf{v}$ sum. Suppressing them is described as freeing capacity, which is a story, not a measurement.

**The cosine-similarity connection.** For $\|\Delta\| \ll \|x\|$ (which they verify — baseline updates are smaller than the residual stream at almost every layer),

$$\cos(x, x+\Delta) \approx 1 - 0.5\left(\frac{\|\Delta_\perp\|}{\|x+\Delta_\parallel\|}\right)^2$$

So the cosine-similarity heuristic used for layer-dropping is *already* a perpendicular-steering measure in disguise. A layer is safe to prune when its perpendicular part vanishes. This retroactively justifies a heuristic that was previously just empirical.

**Practical caveats if you wanted to use this:**
- Inference-time editing is essentially free (hooks, <2.5% overhead) but only reliably safe above ~3B parameters.
- The pretraining intervention is the genuinely useful piece: +1.5 points at 2.7B for 3.39% training overhead, and the effect is visible from early steps so it is cheap to validate.
- Everything is decoder-only. Encoders, [[BERT- Pre-training of Deep Bidirectional Transformers|bidirectional]] models and non-language modalities are untested — the authors say so.
- They are honest that these are analysis tools first: "turning them into practical training or inference methods requires further validation".

**Open questions.** Does perpendicular error as a compression objective actually beat $L_2$ when you optimise for it directly, rather than just measuring it post-hoc? Does the parallel-suppression gain persist at 10B+, or does it saturate? And why does the benefit of parallel suppression *grow* with scale (+0.7 → +1.5) while the tolerance to parallel removal *also* grows with scale — are those the same phenomenon?

**Connections.** This sits next to gated attention and QK-norm as evidence for decoupling contextual routing from magnitude modulation. It is also a geometric account of why [[Deep Residual Learning for Image Recognition (ResNet)|residual connections]] work: they already give you the scalar-gain path for free, so the learned branch should only be doing rotation.

## Links

Related: [[Attention Is All You Need]] · [[Query, Key, and Value (QKV)]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Layer Normalization]] · [[Quantization]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Representation Degeneration Problem in Training NLMs]] · [[The Lottery Ticket Hypothesis]] · [[Gated Activation]] · [[Embeddings]] · [[Grouped Query Attention]] · [[Perplexity]] · [[Mixture of Experts]] · [[Long Context]] · [[Whitening Sentence Representations]] · [[How Contextual are Contextualized Word Representations]]

New topics worth writing: AWQ activation-aware weight quantization, Wanda pruning, residual stream as shared workspace (transformer circuits), attention sinks, LayerScale and ReZero branch scaling, RULER long-context benchmark, rank collapse in pure attention, semi-structured 2:4 sparsity
