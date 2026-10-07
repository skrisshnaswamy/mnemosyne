---
title: "ViRDM: Taming Representation Distribution Matching for Few-Step Causal Video Generation"
authors: ["Meng et al."]
year: 2026
arxiv: "2609.28923"
url: https://arxiv.org/abs/2609.28923
priority: Good-To-Read
read_on: 2026-09-28
tags: [paper, self-supervised, diffusion, vision]
---
## The Core Idea

Making a video model fast usually means **distillation**: take a big slow teacher diffusion model, train a small student to do in 4 steps what the teacher does in 50. The dominant recipe for video is Distribution Matching Distillation (DMD), and it is expensive in a specific, annoying way: at every training step you must hold **three** video diffusion networks in GPU memory at once — a frozen teacher (often 14B), an online "critic" that is retrained on the fly to model the student's own output distribution, and the student itself (1.3B).

Why three? Because DMD never compares generated videos to real videos directly. It adds noise back to a generated video, asks "which way does the teacher's score point?" and "which way does the critic's score point?", and pushes the student along the difference. The distance between distributions is inferred *indirectly*, through noise-conditioned [[Score Function|score]] estimates.

ViRDM asks: what if you just compare the two distributions directly, in a frozen feature space, against a set of reference features you computed **once, offline**?

That is the whole idea. Encode 6,505 real video–text pairs with a frozen video encoder ([[JEPA|V-JEPA]] 2.1) and a frozen text encoder (SigLIP2). Store those features. During training, generate a batch of videos, encode them the same way, and minimise the **Maximum Mean Discrepancy (MMD)** between the generated feature cloud and the stored reference feature cloud. No teacher. No critic. Only the generator gets gradients.

> [!NOTE] Representation Distribution Matching (RDM)
> Train a generator by matching the *distribution of frozen encoder features* of its samples to the distribution of features of real data, using a kernel two-sample distance (MMD). The reference side is precomputed and never changes. ^rdm-def

This already existed for one-step **image** generation. The contribution here is that a literal port to video does not run — it OOMs before finishing a single update — and the paper diagnoses exactly three reasons why, then fixes each one.

The payoff is blunt: peak memory 77.1 GB → 48.3 GB per GPU, training time 22 hours → 2 hours (176 → 16 A100 GPU-hours), and VBench Total 84.51 → **84.87**. The whole post-training run is **20 generator updates**.

---

## The Methodology

### The objective

For a video $v$ and its text prompt $c$, build a joint feature by concatenation:

$$h(v,c) = [\,\phi_v(v)\;;\;\beta\,\tau(c)\,], \qquad \tau(c) = \frac{\phi_t(c)}{\lVert \phi_t(c)\rVert_2}$$

- $\phi_v$ = frozen V-JEPA 2.1 ViT-L/16, final-LayerNorm tokens **globally average-pooled over space and time** → 1,024 dims. One vector per whole video.
- $\phi_t$ = frozen SigLIP2 text tower → 1,152 dims, $\ell_2$-normalised.
- Joint dim = 2,176. $\beta = \sigma_v/\sigma_t$ puts both halves on the same scale.

Similarity is a Gaussian RBF kernel, which factorises neatly into a visual term times a text term:

$$k(h,h') = \exp\!\left(-\frac{\lVert\phi_v(v)-\phi_v(v')\rVert^2}{2\sigma_v^2}\right)\exp\!\left(-\frac{\lVert\tau(c)-\tau(c')\rVert^2}{2\sigma_t^2}\right)$$

Two samples count as close only if **both** the video and the prompt are close. Bandwidths $\sigma_v,\sigma_t$ come from the median heuristic on the reference set.

The loss is the empirical squared MMD between $B$ fresh generated samples $\widehat{\mathcal H}$ and $N{=}6{,}505$ reference samples $\mathcal H^\star$:

$$\mathcal L_{\mathrm{RDM}} = \frac{1}{B^2}\sum_{i,i'} k(\hat h_i,\hat h_{i'}) \;-\; \frac{2}{BN}\sum_{i,j} k(\hat h_i, h_j^\star) \;+\; \underbrace{\frac{1}{N^2}\sum_{j,j'} k(h_j^\star,h_{j'}^\star)}_{\text{constant}}$$

Term 1 pushes generated samples **apart** from each other (a built-in anti-[[Mode Collapse|collapse]] force). Term 2 pulls them **towards** the reference cloud. Term 3 has no gradient.

### Barrier 1 — the gradient path does not fit in memory

The loss sits behind three stacked things: a 4-step [[Auto-regressive models|autoregressive]] rollout over 7 temporal chunks, a heavy Wan video [[Variational Autoencoder|VAE]] decoder, and the V-JEPA encoder. All frozen except the generator — but freezing removes *parameter* gradients, not the *activations* needed to pass gradients backwards. Three fixes, applied in order:

**(a) Consistency-style sampling with a stochastic clean exit.** Normal flow-matching sampling only gives you a clean video after all 4 evaluations. But RDM only cares about *clean endpoint predictions* — so switch to [[Consistency Models|consistency-style]] sampling, where **every** step predicts a clean $\hat z_0$ and then re-noises it for the next step. Now every step is a valid supervision point.

Each update then draws one exit $S \sim \mathcal U\{1,\dots,4\}$, shared across all data-parallel ranks and all temporal chunks. Steps $1\ldots S-1$ run under `no_grad`. Only step $S$ builds a graph. Over many updates all four exits get supervised; on any single update only one is differentiated.

> [!NOTE] Stochastic clean exit
> Roll out the few-step sampler without gradients, then differentiate only through one randomly chosen denoising step's clean prediction. Borrowed from Self Forcing's DMD training. ^stochastic-exit

**(b) Staged vector–Jacobian products.** Even with one exit, naively calling `.backward()` keeps the generator, decoder and encoder graphs alive simultaneously. Instead, do the [[Vector Jacobian Product|VJP]] chain in four separate stages, releasing each graph as soon as it has done its job:

$$g_h = \nabla_{\hat h}\mathcal L_{\mathrm{RDM}}, \quad g_v = J_\Phi^\top g_h, \quad g_z = J_D^\top g_v, \quad \nabla_\theta \mathcal L = \Big(\tfrac{\partial \hat z_0}{\partial\theta}\Big)^{\!\top} g_z$$

Stage 1 generates everything with no gradients at all, caching $\hat z_0$, $\hat h$, and the RNG trace. Stage 2 computes $g_h$ from the loss. Stage 3 replays the encoder to get $g_v$, releases it, replays the decoder to get $g_z$, releases it. Stage 4 **replays the generator rollout with gradients on**, using the saved RNG trace so the rollout is bit-identical, and backprops $g_z$. Classic compute-for-memory trade (see [[Pytorch Autograd]] and [[Flash Attention]] for the same instinct).

**(c) A lightweight decoder.** Even alone, differentiating through the Wan VAE decoder OOMs. But the decoder here is not an optimisation target — it is just a differentiable bridge from latents to pixels so the encoder can see them. Swap it for TAEW2.1, a tiny video autoencoder.

Cumulative memory table (81 frames, 832×480, 80 GB A100s, 1 video per GPU):

| Rollout | Decoder | Backward | GPUs | Peak/GPU |
|---|---|---|---|---|
| Full | Wan VAE | end-to-end | 8 | OOM |
| Stochastic exit | Wan VAE | end-to-end | 8 | OOM |
| Stochastic exit | Wan VAE | staged VJP | 8 | OOM |
| Stochastic exit | TAEW2.1 | staged VJP | 1 | 68.5 GB |
| Stochastic exit | TAEW2.1 | staged VJP | 8 | **48.3 GB** |

### Barrier 2 — the image recipe's hyperparameters are wrong for video

Two things transferred badly: how many fresh samples you need, and what you start from. Both are covered in the ablations below.

### Barrier 3 — features underconstrain motion

Globally pooled video features happily accept a pretty but nearly-still clip. Fix: a one-sided hinge on optical flow, using a frozen RAFT network on sampled frame pairs to produce a video-level dynamics score $s(\hat v)$:

$$\mathcal L_{\mathrm{dyn}}(\hat v) = \big[\tau - s(\hat v)\big]_+, \qquad \mathcal L = \mathcal L_{\mathrm{RDM}} + \lambda_{\mathrm{dyn}}\mathcal L_{\mathrm{dyn}}$$

The penalty switches **off** once a clip has enough motion, so it corrects the floor rather than continuously rewarding more movement. $\lambda_{\mathrm{dyn}} = 5\times10^{-4}$.

### Training setup

Wan2.1-T2V-1.3B backbone. 81 frames (21 latent frames) at 832×480, 3 latent frames per chunk. 4 denoising steps, timestep shift 5, normalised timesteps $\{1, 0.9375, 0.8333, 0.625\}$. [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], lr $2\times10^{-6}$, $(\beta_1,\beta_2)=(0.9,0.95)$, **no weight decay**. 20 updates, $B=64$, ~6 min per update, 8×A100.

---

## Ablation Studies and Experiments

### How many fresh generated videos? Far fewer than images need

Image RDM reports a broad optimum above 2,048 samples and degrades noticeably at 512. Video does not behave that way.

| $B$ | Accum/GPU | Total | Quality | Semantic |
|---|---|---|---|---|
| 8 | 1 | 83.19 | 83.89 | 80.39 |
| **64** | 8 | **83.41** | **84.11** | **80.61** |
| 256 | 32 | 83.33 | 84.02 | 80.57 |
| 1024 | 128 | 83.57 | 84.32 | 80.59 |

Not monotonic: 256 is *worse* than 64. And $B=1024$ buys +0.16 Total for 16× the accumulation (≈96 minutes per update instead of 6). $B=64$ is the chosen operating point. Even $B=8$ works, which is the surprising part.

### Initialisation: causal compatibility is mandatory, few-step specialisation is a bonus

Same 20 updates of RDM, only the starting checkpoint differs:

| Init | Causal-prefix? | Few-step? | Total | Dynamic Degree |
|---|---|---|---|---|
| Bidirectional (none) | ✗ | ✗ | **65.80** | 51.39 |
| Teacher Forcing | ✓ | ✗ | 83.21 | 43.18 |
| Causal Consistency Distill. | ✓ | ✓ | 82.99 | 47.44 |
| **Causal ODE** | ✓ | ✓ | **83.41** | 48.61 |

This is the sharpest negative result in the paper. Starting from the plain bidirectional Wan model and just running it with a causal mask collapses to 65.80 — the videos drift badly. Its high Dynamic Degree (51.39) is *incoherent drift*, not motion, which is a nice reminder that Dynamic Degree alone is gameable.

The failure is **not** about multi-step vs few-step: Teacher Forcing is also multi-step and scores 83.21. It is about causality. **RDM can refine an existing causal transport; it cannot create one in 20 updates.** Same conclusion DMD-based methods reached independently.

Few-step init isn't required but improves motion: Teacher Forcing gets stuck at 43.18 Dynamic Degree, Causal ODE reaches 48.61.

### The dynamics diagnostic — why image features are not enough

Controlled experiment: take real 81-frame clips, keep 0/25/50/75/100% of uniformly spaced frames and repeat them to restore length. Measure the RDM loss against the fixed reference.

| Encoder | 0% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|
| DINOv2 (per-frame) | 0.006830 | 0.006098 | 0.006099 | 0.006102 | 0.006101 |
| V-JEPA 2.1 (video) | 0.100708 | 0.027522 | 0.015179 | 0.008694 | 0.006240 |

Read the top row: from 25% to 100% dynamics, the image-feature loss **does not move at all** (four decimal places identical). An independently-applied image encoder cannot see frame order or spacing. It only reacts at the degenerate 0% endpoint, and only because the feature cloud loses diversity there.

Video features do respond — monotonically — but **nonlinearly**: 72.7% of the drop happens in the first quarter of the range, and the curve flattens. Still true under $\sqrt{\mathrm{RDM}}$, so it is not just the squaring. So the objective can lift a model out of "near-frozen" but cannot push it from moderate to high motion.

### Representation choice + flow regulariser

| Representation | Flow reg. | Total | Quality | Semantic | Dyn. Degree | Total w/o DD |
|---|---|---|---|---|---|---|
| DINOv2 (image) | ✗ | 82.70 | 83.16 | 80.88 | 18.06 | **87.04** |
| V-JEPA 2.1 (video) | ✗ | 83.41 | 84.11 | 80.61 | 48.61 | 85.77 |
| V-JEPA 2.1 | ✓ | **84.87** | **85.82** | **81.09** | **72.02** | 85.79 |

Image features give the **best** frame-level quality (87.04 excluding motion) and a nearly static video (18.06). That is a clean statement of the trade.

Honest accounting from the authors: Dynamic Degree is itself a component of VBench Quality, so the +1.71 Quality gain from the regulariser is roughly explained (~1.80) by the Dynamic Degree jump alone. Excluding it, Quality moves by $-0.10$ and Total by $+0.02$. The regulariser fixes motion **without touching anything else**, which is what you want but is also less impressive than the headline.

### Is the flow regulariser just gaming an optical-flow metric?

Fair worry, since Dynamic Degree *is* optical-flow-based. The weight sweep, with out-of-objective diagnostics:

| $\lambda_{\mathrm{dyn}}$ | Total | Dyn. Degree | Motion Smooth. | Temporal Flicker | Subject Consist. | Background Consist. |
|---|---|---|---|---|---|---|
| 0 | 83.41 | 48.61 | 98.11 | 99.05 | 96.56 | 96.25 |
| 1e-4 | 83.62 | 54.17 | 98.31 | 98.98 | 96.50 | 96.19 |
| **5e-4** | 84.87 | 72.02 | 98.29 | 99.13 | 96.41 | 96.24 |
| 1e-3 | **85.64** | **88.89** | 98.56 | 98.14 | 96.15 | 95.36 |

$\lambda=10^{-3}$ gives the best *Total* (85.64) but visibly damages Temporal Flickering ($-0.91$), Subject ($-0.41$) and Background Consistency ($-0.89$) — that is metric-gaming starting. They ship $5\times10^{-4}$, where all four held-out metrics stay within 0.18 of baseline. Good discipline: they deliberately did **not** take the highest headline number.

Human study (25 participants) backs it: 85.6% prefer the regularised model on dynamics, 76.4% on visual quality, 60.8% on text alignment.

### Headline comparison, four-step causal, same Wan2.1-1.3B backbone

| Method | Total | Quality | Semantic |
|---|---|---|---|
| CausVid | 81.15 | 83.99 | 69.79 |
| Self Forcing | 84.20 | 84.90 | **81.39** |
| Causal Forcing | 84.51 | 85.36 | 81.09 |
| **ViRDM** | **84.87** | **85.82** | 81.09 |

All four run at 17.0 FPS with 0.69 s latency. Human study on 20 prompts: ViRDM wins 43.0% on visual quality and 40.4% on alignment; Self Forcing second at 32.6% on both.

### Extensions

- **Fewer causal steps** (first block 4 steps, rest 1–2): at 2 steps, best Quality 85.44, Total 84.42 (0.01 behind Causal Forcing++). At 1 step, best Total 84.27.
- **Bidirectional**: same recipe on bidirectional Wan gives 84.56 / 84.53 / 83.12 Total at 4 / 2 / 1 steps. So the method is not tied to a causal mask.

---

## Worth Remembering

**The efficiency claim is the real result, not the +0.36 VBench.** 84.87 vs 84.51 is small. 16 GPU-hours vs 176, and 48.3 GB vs 77.1 GB, is not. If you replicate one thing, replicate the cost table.

**Twenty updates.** The entire post-training is 20 gradient steps at lr $2\times10^{-6}$. This is refinement of an already-good checkpoint, not training. Read the "84.87" with that framing: it is a cheap final polish on top of Causal Forcing's causal-ODE initialisation, which someone else trained.

**They still depend on the DMD lineage.** The causal-ODE init they start from comes from Causal Forcing, which used a teacher. ViRDM removes the teacher from the *last stage only*. The paper is upfront about this in the Limitations, but a casual reading of "teacher- and critic-free" oversells it.

**The staged-VJP pattern is reusable well beyond video.** Any time your loss sits behind a chain of frozen modules — a frozen decoder, a frozen [[CLIP]]-style scorer, a frozen reward model — you can compute the VJP one module at a time and free each graph, rather than holding one giant tape. Pair it with the RNG-trace replay trick (run forward without gradients, save the random draws, re-run with gradients) and you trade roughly 2× forward compute for a large memory cut.

**The frozen-encoder-defines-the-target problem.** Your generator can only be as good as what V-JEPA's pooled features can distinguish. The freezing diagnostic (Table 5) is essentially a probe of what the encoder is blind to, and finding a nonlinear blind spot led directly to the flow regulariser. That diagnostic — sweep a property, measure how much the loss moves — is a good habit for anyone using a frozen encoder as a learning signal. It is the same failure mode as [[Evaluating Generative Models|FID]] blind spots, and it is why the paper also reports metrics the objective does not touch.

**Global average pooling over space *and* time is doing a lot of damage.** One 1,024-dim vector per 81-frame clip is a very coarse summary. The obvious follow-up is a distribution over patch- or segment-level features rather than one pooled vector, which might make the flow regulariser unnecessary.

**Caveats for use:** 1.3B backbone, 81 frames, 832×480 only. Untested at larger scale, higher resolution, or long horizons. Reference set is fixed at 6,505 pairs — the model can only be pulled towards *that* distribution, so its biases are inherited wholesale.

---

## Links

Related: [[Video Diffusion]] · [[Diffusion Models]] · [[Consistency Models]] · [[Distillation]] · [[Vector Jacobian Product]] · [[Pytorch Autograd]] · [[JEPA]] · [[Self-Supervised Learning from Images with I-JEPA]] · [[Flow Matching]] · [[Auto-regressive models]] · [[Causal Attention]] · [[Evaluating Generative Models]] · [[Mode Collapse]] · [[Variational Autoencoder]] · [[Latent Diffusion]] · [[Score Function]] · [[Diffusion Sampling]] · [[Distilling the Knowledge in a Neural Network]] · [[Decoupled Weight Decay Regularization (AdamW)]]

New topics worth writing: Maximum Mean Discrepancy and kernel two-sample tests, Distribution Matching Distillation, VBench and video generation benchmarks, Gradient checkpointing and activation recomputation, Optical flow and RAFT, Self Forcing and train/test gap in autoregressive video, Median heuristic for kernel bandwidth
