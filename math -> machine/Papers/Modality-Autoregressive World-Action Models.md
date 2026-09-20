---
title: "Modality-Autoregressive World-Action Models"
authors: ["Adam Hung", "Bardienus P. Duisterhof", "Deva Ramanan", "Jeffrey Ichnowski"]
year: 2026
arxiv: "2609.17524"
url: https://arxiv.org/abs/2609.17524
priority: Must-Read
read_on: 2026-09-16
tags: [paper, transformers, diffusion, vision]
---
## The Core Idea

A **world-action model (WAM)** is a robot policy that predicts *what the world will look like next* and *what actions to take*, in the same network. The usual recipe predicts the future as RGB pixels (or video-VAE latents), then reads actions off that. This paper's claim: RGB is the *worst* of the available futures, and the order in which you generate futures matters a lot.

> [!NOTE] World-action model ^world-action-model-modar
> A model that jointly learns $p(\text{future observations}, \text{actions} \mid \text{current observation})$. The point is data: action-labelled robot demos are rare and expensive, but *actionless* video (humans doing chores, other robots) is abundant, and you can still supervise the future-observation half on it.

ModAR's trick is **modality-autoregressive denoising**. Instead of denoising all the future streams at once, it denoises them one at a time, in a fixed order, each conditioned on the finished versions of the earlier ones:

$$p_\theta(\mathbf{Y}_t, \mathbf{A}_t \mid \mathbf{c}_t) = p_\theta(\mathbf{A}_t \mid \mathbf{c}_t, \mathbf{Y}_t)\prod_{k=1}^{K} p_\theta(\mathbf{Y}_t^{m_k}\mid \mathbf{c}_t, \mathbf{Y}_t^{<k})$$

The order is **point tracks → DINO features → depth → RGB → actions**. Compact, structured, easy-to-predict things first; detailed, high-variance pixels last; actions dead last, so the action head is effectively an inverse-dynamics model reading a *fully clean* imagined future.

Why this did not exist: nearly every WAM is initialised from a big pretrained video generator (Wan, Cosmos), which locks you into RGB latents and makes controlled comparisons impossible — you cannot tell whether a gain came from the formulation or from the pretraining. ModAR is 30.1M parameters trained **from scratch**, which is what makes the ablations trustworthy.

What it unlocks, concretely: a 30.1M model beats a 6B video-pretrained WAM (Flex-$\pi$) 75% vs 72% average success, at roughly $20\times$ fewer training FLOPs and no pretraining.

> [!NOTE] Modality ^modality-as-future
> Here "modality" means *a representation of the future*, not a sensor. Depth, DINOv2 patch features, and 2D point tracks are all derived from the same RGB video — they are different lenses on the same signal, each with a different inductive bias (geometry, semantics, motion).

## The Methodology

**Setup.** One $168\times224$ camera frame. Conditioning is $\mathbf{c}_t = (\mathbf{o}_t, \mathbf{q}_t, g)$: the multimodal observation, 14-D dual-arm joint configuration, and a learned embedding of a discrete task label (not language). Prediction horizon $H=16$, dynamics stride $\Delta=8$, so $J=2$ sparse visual targets at $t{+}8$ and $t{+}16$, plus a dense 16-step action chunk. Replan every 16 steps.

**Tokenisation.**
- RGB and depth: patchify into a $12\times16$ grid of $14\times14$ patches.
- DINO: spatial patch tokens from a frozen DINOv2 ViT-S/14.
- Tracks: seed a 2D grid of queries at patch centres, run CoTracker3, and encode each point-time pair as displacement-from-start plus a visibility flag.
- Everything gets a linear projection to a shared token width, a learned modality embedding, and axial [[RoFormer- Enhanced Transformer with Rotary Position Embedding|RoPE]] over (time, height, width) for visual tokens, over time for action tokens.

**Backbone.** A diffusion transformer. Six shared width-384 blocks with 6 heads (ViT-S sized) where all modalities attend to each other, then small modality-specific expert stacks whose attention is restricted to their own stream: two width-384 blocks for DINO/depth/RGB, one width-128 block for tracks, two width-128 blocks for actions, each with a linear output head. Robot config, task embedding, and per-modality flow timesteps enter via adaLN conditioning at every layer. So: fuse across modalities first, specialise second.

**Block-causal mask.** Training needs all stages supervised in one forward pass, so each target modality exists twice: a *clean context copy* and a *noisy prediction copy*. A [[Causal Attention|block-causal mask]] lets the prediction copy of $m_k$ see only the observation and the context copies of $m_1 \dots m_{k-1}$. No leakage, one pass. At inference you actually denoise sequentially, re-embedding each finished block as context, and KV-cache the observation plus completed modalities so you do not recompute them at every Euler step.

**Context noise — the load-bearing detail.** Sequential generation compounds errors: a bad track prediction poisons everything downstream. Fix (borrowed from Latent Forcing): during training, corrupt the context copies. For each context block $m_j$, sample $\epsilon_j^{\text{ctx}}\sim\mathcal{N}(0,I)$ and $\tau_j^{\text{ctx}}\sim\mathcal{U}(1-\beta,1)$ with $\beta = 0.5$, and feed
$$\widetilde{\mathbf{Y}}^{m_j}_{t,\text{ctx}} = \tau_j^{\text{ctx}}\mathbf{Y}^{m_j}_t + (1-\tau_j^{\text{ctx}})\epsilon_j^{\text{ctx}}$$
Training only. Inference uses clean context.

**Objective.** Linear flow-matching interpolant $\widetilde{\mathbf{Y}}^m_t = \tau_m \mathbf{Y}^m_t + (1-\tau_m)\epsilon^m$, but the head predicts the **clean sample** $\widehat{\mathbf{Y}}$, not the velocity ($x$-prediction, JiT style):
$$\mathcal{L}_m(\theta) = \mathbb{E}\left[\frac{\lVert \widehat{\mathbf{Y}}^m_t - \mathbf{Y}^m_t\rVert_2^2}{\max(1-\tau_m,\delta_m)^2}\right], \qquad \delta_m = 0.05$$
Total loss is $\sum_k \mathcal{L}_{m_k} + \mathcal{L}_{\text{act}}$, with the action term dropped on actionless examples. The denominator upweights predictions made near the clean end of the trajectory, where the residual is naturally small.

They report that swapping $x$-prediction for **$v$-prediction is often unstable and can diverge** — the argument is that predicting the clean sample is better conditioned when data lies on a low-dimensional manifold, as visual data does.

**Sampling.** 8 Euler steps per stream, $\tau: 0 \to 1$, with the velocity recovered from the clean prediction as $\mathbf{v}_\theta = (\widehat{\mathbf{Y}} - \widetilde{\mathbf{Y}})/(1-\tau)$. Flow timesteps are sampled logit-normal with per-modality $(\mu,\sigma)$: $(-2,1)$ DINO, $(0,1)$ tracks, $(-1,1.6)$ depth, $(-1,1)$ RGB and actions.

**Optimisation.** [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], lr $10^{-4}$, betas $(0.9, 0.95)$, weight decay 0.1, grad clip 1.0, global batch 48, [[Mixed Precision training|bfloat16]], [[Momentum#^ema-weights|EMA]] decay 0.999, 48k-sample linear warmup then constant LR, 1.2M optimizer steps. Each step draws equal-size batches from the action-labelled and actionless pools, losses weighted equally. Checkpoints evaluated every 100k steps; best checkpoint reported.

**Data.** Simulation: 6 RoboTwin tasks, $D \in \{50, 250, 1250\}$ total demos per task, of which always exactly **50 are action-labelled** — the rest are actionless. One multitask model per method per scale, 50 held-out initial conditions per task. Real world: bimanual YAM arms, 3 tasks (stack cups, fold crumpled towel, place object in drawer + close it), 100 teleoperated robot demos + 200 in-domain actionless human demos + 1,000 EgoDex demos, fixed third-person ZED stereo camera for RGB and depth, 30 rollouts per task.

## Ablation Studies and Experiments

**Formulation comparison** (all four modalities predicted, average success over 6 tasks):

| $D$ | Action-only | Independent-noise | Disjoint | Unified | **ModAR** |
|---|---|---|---|---|---|
| 50 | 0.46 | 0.37 | 0.55 | 0.63 | **0.66** |
| 250 | — | 0.34 | 0.55 | 0.67 | **0.75** |
| 1250 | — | 0.33 | 0.48 | 0.64 | **0.76** |

The baselines are the real published families: *Unified* = co-denoise futures and actions with one shared flow timestep (DreamZero, Cosmos Policy). *Independent-noise* = same but sample each stream's timestep separately during training (Unified World Models, Flex-$\pi$). *Disjoint* = no attention between future and action targets, future prediction is a training-time auxiliary only, dropped at inference (Fast-WAM). *Action-only* = plain flow-matching behaviour cloning.

Two things stand out:
- **ModAR is the only formulation that scales with actionless data.** 66% → 76% as demos go 50 → 1250. Unified gains 1 point. **Disjoint gets *worse*** (55% → 48%) — plausible reading: with more actionless data the shared trunk gets dominated by the future-prediction objective, and since the action head cannot attend to the futures, that is pure negative transfer.
- **Independent-noise is a disaster from scratch** (33–37%, below the no-world-model baseline). Independently sampled noise levels almost never match the synchronised test-time schedule, so the model gets little training signal near the regime it actually runs in. This works fine when you start from a pretrained video model; it does not survive training from scratch.

**Which modalities matter** (average success, ModAR, following the generation order):

| $D$ | Action-only | RGB | Depth | Tracks | DINO | T+D | T+D+Dep | T+D+Dep+RGB |
|---|---|---|---|---|---|---|---|---|
| 50 | 0.46 | 0.43 | 0.46 | 0.41 | 0.60 | 0.63 | 0.63 | 0.66 |
| 250 | — | 0.54 | 0.63 | 0.53 | 0.65 | 0.72 | 0.75 | 0.75 |
| 1250 | — | 0.58 | 0.56 | 0.61 | 0.68 | 0.72 | 0.77 | 0.76 |

RGB-only — the standard WAM setting — is the *weakest* single modality and at $D=50$ is actually worse than no world model at all (0.43 vs 0.46). DINO alone beats it by 6–17 points. Adding modalities is roughly additive up to tracks+DINO+depth; **adding RGB on top buys nothing** (0.75 → 0.75, 0.77 → 0.76).

**Leave-one-out ablations** (at $D=250$, full model = 75%):

| Change | Avg success |
|---|---|
| Full ModAR | 0.75 |
| No context noise | **0.63** |
| Reverse order (RGB→depth→DINO→tracks) | 0.65 |
| w/o tracks | 0.61 |
| w/o DINO | 0.65 |
| w/o depth | 0.70 |
| w/o RGB | 0.75 |

Context noise is the single biggest lever — remove it and you lose 12 points to cascading error. Order is worth 10 points, which is direct evidence for the "structured modalities are a scratchpad" hypothesis. Tracks are the most valuable modality; RGB is worth exactly zero.

**Two honest controls the authors ran against themselves:**

1. *Is ModAR just winning because its action head sees cleaner futures?* They discarded both ModAR's and Unified's native action heads and piped both models' predicted futures into the **same separately trained inverse-dynamics model**. ModAR still wins — so its *predicted futures are genuinely better*, not just better-consumed.
2. *Is ModAR just winning because it takes more sampling steps?* ModAR uses 8 Euler steps × 5 streams = 40; the others use 8 total. Giving the baselines 40 steps at $D=250$ did not close the gap: Unified **dropped** 67% → 59%, Disjoint 55% → 54%, Independent-noise 34% → 37%, versus ModAR's 75%.

**Against a 6B video-pretrained model.** Flex-$\pi$ initialised from Wan2.2-TI2V-5B, full fine-tune (VAE, text encoder, DINOv3 frozen), global batch 288, 30k steps: **72%** average success. ModAR (tracks+DINO+depth, 30.1M params): **75%**. $\sim200\times$ fewer parameters, $\sim20\times$ fewer training FLOPs. The authors are careful to call this system-level, not a controlled architectural comparison.

**Real world.** With tracks+DINO+depth only (RGB dropped, since it bought nothing and costs compute): ModAR **83.3%**, Unified 66.7%, Action-only 52.2%, 30 trials per task. And the actionless-data story holds on real hardware: 100 robot demos alone → 70.0%; +200 in-domain human videos → 81.1%; +1,000 out-of-domain EgoDex videos → 83.3%. Human video with no action labels, from a different embodiment, still helps.

**Latency:** 147.9 ms end-to-end (6.76 Hz) on an RTX 5090 generating all four future modalities plus actions.

## Worth Remembering

- The headline practical takeaway is cheap to apply: **if you already predict a visual future, try replacing RGB with point tracks and DINO features.** Tracks alone are not enough (they are the weakest single modality at $D=50$ and $250$), but tracks+DINO is a strong, small pair.
- The "scratchpad" story generalises beyond robotics. Latent Forcing generates latents before pixels; Modality Forcing generates depth alongside images. ModAR's contribution is applying the ordering principle to a *policy*, and confirming the order matters by reversing it.
- **$v$-prediction diverging** is a real, reported failure and worth filing away. When your targets are visual and live on a low-dimensional manifold, predict $x$ and derive the velocity, not the other way round.
- Note what the sampling-step control implies: giving Unified *more* denoising steps made it **worse**. That is a strange result the paper does not dwell on. Something about the shared-timestep joint denoising is badly behaved when integrated finely.
- Limitations the authors own: only 6 sim + 3 real tasks, **discrete task labels rather than language instructions**, so nothing here demonstrates generalisation to new objects, scenes, or task descriptions. Sequential generation adds latency versus joint or action-only. And they explicitly say they did not systematically search the modality ordering — tracks→DINO→depth→RGB versus its exact reverse is the only comparison run.
- The comparison to Flex-$\pi$ deserves a caveat the authors flag: these are in-distribution RoboTwin tasks. Video pretraining is supposed to buy *out-of-distribution* generalisation, which this benchmark cannot measure. Do not read "30M beats 6B" as a general law.
- Practical caveat for reuse: the pipeline depends on two frozen off-the-shelf models at *data-preparation* time — CoTracker3 for tracks and DINOv2 ViT-S/14 for features. Your track targets inherit CoTracker's failure modes.

## Links

Related: [[GameWAM- A World Action Model for Video Games]] · [[Hydra-0- Action Flow for Generalist World Modeling and Control]] · [[Latent Action as Intention Enables Efficient Future Imagination for World Action Models]] · [[Zero-WAM- In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization]] · [[GigaBrain-0.7- Scaling Embodied Foundation Models to Emergent Capabilities with a Three-System Architecture]] · [[τ_0-VLA- a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Denoising Diffusion Probabilistic Models]] · [[Score-Based Generative Modeling through SDEs]] · [[Auto-regressive models]] · [[Causal Attention]] · [[RoFormer- Enhanced Transformer with Rotary Position Embedding]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Self-Supervised Learning from Images with I-JEPA]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Mixed Precision training]] · [[Query, Key, and Value (QKV)]] · [[Momentum]]

New topics worth writing: Flow matching and linear interpolants, Diffusion Transformer (DiT) and adaLN conditioning, x-prediction vs v-prediction parameterisation, DINOv2 and self-distilled visual features, Point tracking (CoTracker / TAP), Inverse dynamics models, Learning from actionless human video (EgoDex), RoboTwin benchmark
