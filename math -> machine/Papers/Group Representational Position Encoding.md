---
title: "Group Representational Position Encoding"
authors: ["Yifan Zhang", "Zixiang Chen", "Yifeng Liu", "Zhen Qin", "Huizhuo Yuan", "Kangping Xu", "Yang Yuan", "Quanquan Gu", "Andrew Chi-Chih Yao"]
year: 2025
arxiv: "2512.07805"
url: https://arxiv.org/abs/2512.07805
priority: Good-To-Read
read_on: 2026-09-05
tags: [paper, transformers]
---
## The Core Idea

Every positional encoding in a Transformer is secretly a **group action**. That is the whole paper.

A group action means: you have a set of transformations, you can compose them, and composing "move by $n$" with "move by $m$" gives "move by $n+m$". If your position encoding is built this way, then relative position falls out for free — the attention score between token $i$ and token $j$ can only depend on $j-i$, because

$$\mathbf{G}(i)^{-1}\mathbf{G}(j) = \mathbf{G}(j-i).$$

GRAPE writes every position map as a matrix exponential of a single generator:

$$\mathbf{G}(n) = \exp(n\,\omega\,\mathbf{L}).$$

Change what kind of matrix $\mathbf{L}$ is, and you get a different published method:

- $\mathbf{L}$ **skew-symmetric** ($\mathbf{L}^\top = -\mathbf{L}$) → the exponential is a rotation. Pick $d/2$ skew generators acting on disjoint coordinate pairs, with log-spaced frequencies → you have written down [[RoFormer- Enhanced Transformer with Rotary Position Embedding|RoPE]], exactly.
- $\mathbf{L}$ **nilpotent** ($\mathbf{L}^2 = \mathbf{0}$) in a space with two extra "dummy" dimensions → the exponential is $\mathbf{I} + n\omega\mathbf{L}$, and the attention logit picks up a term linear in $j-i$. That is [[Train Short, Test Long (ALiBi)|ALiBi]], exactly. Make the per-step amount data-dependent and you get the Forgetting Transformer's forget gate, exactly.

So the rotary family and the additive-bias family, which look like completely different ideas, are two choices of Lie algebra in one formula. That is the contribution: a **map of the design space**, plus a few new points on that map that no one had written down.

> [!NOTE] One-parameter subgroup
> A family of matrices $\mathbf{G}(n)$ indexed by a number, satisfying $\mathbf{G}(n+m)=\mathbf{G}(n)\mathbf{G}(m)$ and $\mathbf{G}(0)=\mathbf{I}$. Any such family is $\exp(n\mathbf{L})$ for some fixed $\mathbf{L}$, called the generator. This single property is what makes a positional encoding *relative* and *streaming-cacheable*. ^one-parameter-subgroup

What it unlocks in practice is modest but real: cheap **learned** rotation planes (RoPE's planes are hard-coded coordinate pairs; GRAPE lets you learn them at $O(d)$ cost instead of the $O(d^3)$ a general matrix exponential would need), and a new additive bias where the decay rate between two tokens depends on the *querying* token, not just on the distance.

## The Methodology

**Multiplicative GRAPE (GRAPE-M).** Take two vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^d$ and build the rank-2 skew generator

$$\mathbf{L} = \mathbf{a}\mathbf{b}^\top - \mathbf{b}\mathbf{a}^\top.$$

Let $\alpha = \|\mathbf{a}\|^2$, $\beta = \|\mathbf{b}\|^2$, $\gamma = \mathbf{a}^\top\mathbf{b}$, and $s = \sqrt{\alpha\beta - \gamma^2}$. Because $\mathbf{L}^2 = -s^2\mathbf{P}_\mathcal{U}$ (a scaled projector onto the plane spanned by $\mathbf{a},\mathbf{b}$), the exponential truncates after two terms — a Rodrigues formula:

$$\exp(\mathbf{L}) = \mathbf{I} + \frac{\sin s}{s}\mathbf{L} + \frac{1-\cos s}{s^2}\mathbf{L}^2 = \mathbf{I} - (1-\cos s)\mathbf{P}_\mathcal{U} + \frac{\sin s}{s}\mathbf{L}.$$

Read that second form: **rotate by angle $s$ inside the plane $\mathcal{U}$, do nothing everywhere else.**

You never build the $d\times d$ matrix. To apply it to a vector $\mathbf{x}$ you take two dot products $u=\langle\mathbf{a},\mathbf{x}\rangle$, $v=\langle\mathbf{b},\mathbf{x}\rangle$, then

$$\mathbf{y} = \mathbf{x} + f_1(n)(\mathbf{a}v - \mathbf{b}u) + f_2(n)\big[\gamma(\mathbf{a}v+\mathbf{b}u) - \beta\mathbf{a}u - \alpha\mathbf{b}v\big]$$

with $f_1 = \sin(n\omega s)/s$, $f_2 = (1-\cos(n\omega s))/s^2$ (use Taylor series when $s\to 0$ or you get $0/0$). Cost: $O(d)$ per head. Sum $d/2$ of these on orthogonal planes and you get RoPE; use a learned orthogonal basis $\mathbf{B}$ instead of coordinate pairs and you get "RoPE in a learned frame".

**Additive GRAPE (GRAPE-A).** Rotations cannot produce an additive logit bias, because rotations preserve the inner product's structure. The trick is a **homogeneous lift** — the same trick graphics uses to write a translation as a matrix multiply. Append constants:

$$\widehat{\mathbf{q}}_i = [\mathbf{q}_i;\,1;\,0], \qquad \widehat{\mathbf{k}}_j = [\mathbf{k}_j;\,0;\,1] \in \mathbb{R}^{d+2}.$$

Note $\widehat{\mathbf{q}}_i^\top\widehat{\mathbf{k}}_j = \mathbf{q}_i^\top\mathbf{k}_j$ — the padding adds nothing on its own. Now take $\mathbf{A}_0 = \mathbf{e}_{d+2}\mathbf{e}_{d+1}^\top$, which satisfies $\mathbf{A}_0^2=\mathbf{0}$, so $\exp(n\mathbf{A}_0) = \mathbf{I}+n\mathbf{A}_0$. Score with the **paired inverse-transpose**: transform the query by $\mathbf{G}(i)$ and the key by $\mathbf{G}(j)^{-\top}$. The multiplicative parts cancel and you are left with exactly

$$\widehat{\mathbf{q}}_i^\top \mathbf{G}(j-i)^{-\top}\widehat{\mathbf{k}}_j = \mathbf{q}_i^\top\mathbf{k}_j + (j-i)\beta_h,$$

which is ALiBi. Two extra dimensions are needed, not one: a nilpotent generator has zero trace, so you cannot make a scalar bias appear by having a coordinate feed back into itself.

**GRAPE-A-QK** (their new variant) makes the slope content-dependent while keeping the group law, by gating the same nilpotent basis:

$$\widetilde{\mathbf{q}}_i^\top\widetilde{\mathbf{k}}_j = \mathbf{q}_i^\top\mathbf{k}_j + (j-i)\,\omega\big[\mathrm{softplus}(\mathbf{v}^\top\mathbf{q}_i/\sqrt{d}) + \mathrm{softplus}(\mathbf{u}^\top\mathbf{k}_j/\sqrt{d})\big].$$

Softplus keeps the slope non-negative, and $j-i \le 0$ under [[Causal Attention|causal masking]], so the bias is always a penalty that grows with distance. Because the two gates scale the *same* generator, they commute, so the group law survives.

**FoX is this too.** The Forgetting Transformer computes a forget gate $f_t = \sigma(\mathbf{w}_f^\top\mathbf{x}_t + b_f)$ and adds $D_{ij} = \sum_{\ell=j+1}^{i}\log f_\ell$ to the logits — a per-token learned decay, the [[Long Short-Term Memory (Neural Computation)|LSTM]] forget gate transplanted into attention. Set the per-step nilpotent amount to $\log f_\ell$ and the path product of unipotent matrices collapses (since $\mathbf{E}^2=0$) to exactly $\mathbf{I}+D_{ij}\mathbf{E}$. Constant $f = e^{-\beta_h}$ recovers ALiBi.

**Path-Integral Additive GRAPE (GRAPE-AP)** — the genuinely new mechanism, and the one that wins. FoX's per-step penalty depends only on the token *at* step $\ell$. GRAPE-AP lets it also depend on the **querying** token $t$:

$$\psi_h(t,\ell) = \alpha_h\, g\!\left(\tfrac{1}{d}\langle \mathbf{p}_{t,h},\, \mathbf{R}_\ell\, \mathbf{p}_{\ell,h}\rangle\right) \le 0, \qquad b_h(t,j) = \sum_{\ell=j+1}^{t}\psi_h(t,\ell),$$

where $\mathbf{p}_{u,h}$ is a small positional probe (linear projection + RMSNorm of the token features), $\mathbf{R}_\ell$ a fixed rotation, and $g$ a monotone 1-Lipschitz link into $(-\infty,0)$. Cost per decode step is $O(t)$: one similarity sweep over cached probes, apply $g$, then a prefix sum. This *breaks* the global one-parameter group law — the factors now depend on the endpoint $t$ — but keeps the row-wise path composition, which is all you need for causal attention. Set $\psi$ independent of $t$ and you fall back to FoX; set it constant and you fall back to ALiBi.

**Training setup.** nanoGPT + Llama architecture, only the position encoding swapped. FineWeb-Edu, 50B tokens sampled from the 100B set. Medium = 353M (24 layers, 8 heads, $d_\text{model}=1024$); large = 770M (36 layers, 10 heads, $d_\text{model}=1280$, head dim 128). Context 4096, batch 480. QK RMSNorm for stability. [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], LR $2\times10^{-3}$ (medium) / $1\times10^{-3}$ (large), $(\beta_1,\beta_2)=(0.9,0.95)$, weight decay 0.01, cosine schedule, 2000 warmup steps, grad clip 1.0. RoPE base 10,000. FoX run without FoX-Pro and with its KV-shift module disabled, for a fair comparison.

## Ablation Studies and Experiments

Zero-shot lm-evaluation-harness, average over ARC-E, ARC-C, HellaSwag, OBQA, PIQA, WinoGrande, SciQ.

**353M (medium), average score:**

| Method | Avg |
|---|---|
| RoPE | 51.73 |
| GRAPE-M-nonctx | 51.79 |
| GRAPE-M-ctx | 51.78 |
| ALiBi | 52.87 |
| FoX | 52.96 |
| GRAPE-A-K | 52.43 |
| GRAPE-A-Q | 52.81 |
| GRAPE-A-QK | 53.00 |
| **GRAPE-AP** | **53.25** |

**770M (large), average score:**

| Method | Avg |
|---|---|
| RoPE | 55.76 |
| GRAPE-M-ctx | 54.73 |
| GRAPE-M-nonctx | 54.81 |
| FoX | 56.30 |
| ALiBi | 56.44 |
| **GRAPE-AP** | **56.91** |
| RoPE + KV-shift | 55.32 |
| ALiBi + KV-shift | 56.92 |
| **FoX + KV-shift** | **57.09** |
| GRAPE-AP + KV-shift | 56.86 |

**What did not work — and this is the honest headline.** The multiplicative half of the framework, which is where all the elegant Lie-group machinery lives, **does not beat RoPE**. At 353M it is a wash (51.79 vs 51.73, noise). At 770M it is clearly *worse*: 54.81 and 54.73 versus RoPE's 55.76, about one full point down. Learning the rotation planes and making the phase contextual bought nothing and cost something. The authors do not comment on this in the text; they say "GRAPE can keep a persistent edge over other mechanisms", which is true only of the additive variants.

**Every real gain is additive, and mostly not new.** ALiBi alone beats RoPE by 1.14 points at 353M and 0.68 at 770M. That gap is not GRAPE's; it is ALiBi's, from 2021. GRAPE-AP's contribution on top of the best prior additive baseline is +0.29 (over FoX at 353M) and +0.47 (over ALiBi at 770M) — one to two tenths of a percent per task, on 7 noisy benchmarks, single seed.

**The QK gating ablation is clean and small.** GRAPE-A-K (key-gated only) 52.43 < GRAPE-A-Q (query-gated only) 52.81 < GRAPE-A-QK (both) 53.00. So query-side gating matters more than key-side, and the two compose. All three sit near or below FoX (52.96) except QK. The interpretation: *how much the current query wants to forget* is more informative than *how forgettable this key is*.

**Endpoint dependence is where the win comes from.** GRAPE-AP (53.25) > GRAPE-A-QK (53.00). The one thing GRAPE-AP adds over the gated additive forms is that the per-step penalty $\psi_h(t,\ell)$ depends on the query position $t$. That is what buys the extra 0.25.

**With KV-shift, the win evaporates.** At 770M with the KV-shift module enabled, FoX takes the lead at 57.09 and GRAPE-AP drops to 56.86 — below both FoX and ALiBi. So GRAPE-AP's advantage overlaps with whatever KV-shift is already providing.

**Stability claim.** Figure 3(a) shows the 770M RoPE run has a training-loss spike; GRAPE's curve is smooth. This is one run each, no seeds, so treat it as an anecdote rather than a finding.

**Theory-only comparisons (no experiments run).** Against LieRE, which learns a dense skew generator per head: LieRE needs $\Theta(d^3)$ per matrix exponential and $\Theta(d^2)$ parameters; GRAPE-M's rank-2 decomposition is $O(d)$ both. Real, but LieRE is never actually trained head-to-head here. Against PaTH Attention, which multiplies Householder-like factors $\mathbf{H}_t = \mathbf{I}-\beta_t\mathbf{w}_t\mathbf{w}_t^\top$: the authors show $\det(\mathbf{P}_{j\to i}) = \prod(1-\beta_s)$ decays exponentially in distance unless $\beta_s \in \{0,2\}$, so PaTH contracts and can go near-singular over long paths. Plausible argument, again untested empirically.

## Worth Remembering

- **The unification is the paper's real product, not the method.** If you want one mental model for position encoding: it is a Lie group generator. Skew → rotation → RoPE. Nilpotent → translation in a lifted space → additive bias → ALiBi/FoX. Everything else is a choice of generator.
- **The elegant half loses.** The rank-2 rotation machinery — Rodrigues formula, learned planes, closed-form derivatives, all of Section 2 and Appendix I — underperforms plain RoPE at 770M. This is a nice reminder that a cleaner algebraic story does not imply better loss. It also makes you wonder whether RoPE's *fixed* log-uniform frequency spectrum is a useful [[An Image is Worth 16x16 Words (ViT)#The Core Idea|inductive bias]] rather than an arbitrary restriction — the same "structure beats capacity" pattern as in [[Neural Collaborative Filtering vs. Matrix Factorization Revisited]].
- **The paired inverse-transpose is the load-bearing trick on the additive side.** The lifted matrix $\mathbf{I}+s\mathbf{A}$ is *not* an isometry — Lemma J.4 gives its two non-unit singular values as $\sqrt{1+s^2/2 \pm |s|\sqrt{1+s^2/4}}$, with condition number $\approx 1+|s|$. Transforming the key by $\mathbf{G}(j)^{-\top}$ instead of $\mathbf{G}(j)$ makes all that distortion cancel algebraically, leaving a pure scalar bias. Without it, the additive family would stretch and squash the query/key vectors.
- **Streaming cost is unchanged for GRAPE-M and GRAPE-A** — you rotate a key once when it arrives, cache it, never touch it again, exactly like [[RoFormer- Enhanced Transformer with Rotary Position Embedding|RoPE]]. GRAPE-AP is different: because the potential depends on the current query, you must re-sweep all cached probes every decode step, $O(t)$ extra work and $O(L)$ extra memory per head. That is a real serving cost that the paper mentions but does not benchmark against a [[FlashAttention- Fast and Memory-Efficient Exact Attention|FlashAttention]]-style kernel. Any additive bias also has to be materialised inside the attention kernel, which is a fused-kernel engineering problem in itself.
- **Nothing here tests length extrapolation.** ALiBi's whole reason for existing is training on short contexts and running on long ones. Every experiment here is at 4096 train and 4096 eval. Given that the paper's biggest wins are inherited from ALiBi, the absence of a "train short, test long" table is the most conspicuous gap.
- **Practical caveat:** single seed, one dataset, 7 zero-shot benchmarks with gaps under 1 point. The medium and large rankings disagree about GRAPE-M vs RoPE. Do not swap out RoPE in a production model on the basis of this table.
- **Open question the framework invites:** since additive and multiplicative compose cleanly as a block-diagonal generator in $\mathfrak{gl}(d+2)$ (Appendix E) — rotation block on top, unipotent block on the bottom — the obvious experiment is RoPE **plus** GRAPE-AP. The paper defines it and never runs it.

## Links

Related: [[RoFormer- Enhanced Transformer with Rotary Position Embedding]] · [[Train Short, Test Long (ALiBi)]] · [[Attention Is All You Need]] · [[Causal Attention]] · [[Query, Key, and Value (QKV)]] · [[Long Short-Term Memory (Neural Computation)]] · [[Layer Normalization]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]] · [[Fundamentals]] · [[Linear Projection]] · [[On the Difficulty of Evaluating Baselines]]

New topics worth writing: Lie groups and Lie algebras for ML, matrix exponential and Rodrigues' formula, nilpotent and unipotent matrices, homogeneous coordinates, SO(d) and the special orthogonal group, forget gates in attention (FoX), PaTH Attention and Householder position encoding, LieRE, condition number and singular value analysis of layers, length extrapolation benchmarks
