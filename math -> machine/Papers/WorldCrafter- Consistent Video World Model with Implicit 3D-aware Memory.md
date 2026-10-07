---
title: "WorldCrafter: Consistent Video World Model with Implicit 3D-aware Memory"
authors: ["Yu et al."]
year: 2026
arxiv: "2609.24984"
url: https://arxiv.org/abs/2609.24984
priority: Good-To-Read
read_on: 2026-09-27
tags: [paper, transformers, diffusion, vision]
---
## The Core Idea

A video world model generates the next few frames of a scene as you move a camera around it. The hard part is not making pretty frames. It is **remembering**. Walk left, look at a sofa, spin around for thirty seconds, then look back at the sofa — most models invent a different sofa.

WorldCrafter's trick: **let the camera you are about to point decide how the past gets compressed.**

The model has a fixed, small budget of tokens it can spend on memory — here, the token count of exactly 4 video frames. Everything ever observed has to be squeezed into that budget. Previous work either:

- kept raw past frames and attended over them (cheap to build, but 4 frames of history is 4 frames of coverage, and picking which 4 is fragile), or
- built an **explicit 3D** memory — estimate depth, warp old pixels into the new viewpoint, feed that in. Accurate, but it needs good geometry and it freezes the scene, so moving objects break.

WorldCrafter goes in between. A learned encoder eats 9 past *latent* frames plus their camera poses and produces an abstract representation — no point cloud, no depth map. Then a **readout** module is handed the poses of the frames about to be generated and pulls out just 4 frames' worth of tokens aimed at those viewpoints. Query-driven compression, not fixed compression.

> [!NOTE] Pose-guided readout
> Compressing history *conditional on the viewpoint you are asked for*, so a fixed token budget is spent on what the next chunk actually needs, rather than on a generic summary the generator has to search through. ^pose-guided-readout

Two things make this work that would not have worked two years ago.

First, the encoder is **initialised from LagerNVS**, a novel-view-synthesis model. That matters more than it sounds. The obvious choice would be a geometry model like VGGT, which is what the nearby papers (CineScene, GIM-World) do. But a geometry model is trained to predict *where things are*, not *what they look like*. Reproducing a previously seen sofa is an appearance problem. A novel-view-synthesis encoder had to keep both geometry and appearance to do its original job, so its features carry both.

Second, the memory encoder is **trained jointly with the video [[Diffusion Transformer|DiT]]**, not bolted on. The memory space and the generator's token space co-adapt.

The payoff: 47.6% better revisit consistency (LPIPS) than the strongest baseline, better camera control than all 8 baselines, and memory processing that is **21.7× cheaper** than the depth-and-warp approach — 0.062 s per chunk versus 1.346 s. Distilled, it streams at 16 fps on 4 GPUs.

## The Methodology

### The base generator

Standard latent [[Video Diffusion]]. A VAE maps a video $\mathbf{x} \in \mathbb{R}^{3\times F\times H\times W}$ to a latent $\mathbf{z} = \mathcal{E}_{\mathrm{VAE}}(\mathbf{x}) \in \mathbb{R}^{c\times f\times h\times w}$, and a DiT denoises in that space.

Training is [[Flow Matching]]. For $t \sim \mathcal{U}(0,1)$ and noise $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$:

$$\mathbf{z}_t = (1-t)\mathbf{z} + t\boldsymbol{\epsilon}, \qquad \mathbf{v}^*_t = \boldsymbol{\epsilon} - \mathbf{z}$$

$$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{\mathbf{z},t,\boldsymbol{\epsilon}}\left[\left\|\mathbf{v}_\theta(\mathbf{z}_t, t \mid \mathbf{y}) - \mathbf{v}^*_t\right\|_2^2\right]$$

Generation is chunk-wise [[Auto-regressive models|autoregressive]]. A plain version conditions each chunk only on a sliding window of recent clean frames $\mathbf{z}^{\mathrm{r}}$:

$$\frac{\mathrm{d}\mathbf{z}_t}{\mathrm{d}t} = \mathbf{v}_\theta(\mathbf{z}_t, t \mid \mathbf{z}^{\mathrm{r}}, \mathbf{y})$$

WorldCrafter adds memory $\mathbf{M}$ and a target camera trajectory $\mathbf{C}$:

$$\frac{\mathrm{d}\mathbf{z}_t}{\mathrm{d}t} = \mathbf{v}_\theta(\mathbf{z}_t, t \mid \mathbf{M}, \mathbf{z}^{\mathrm{r}}, \mathbf{C}, \mathbf{y})$$

At every denoising step the DiT sees one concatenated sequence $[\mathbf{M}; \mathbf{z}^{\mathrm{r}}; \mathbf{z}_t]$ and runs ordinary [[Attention|self-attention]] over it. Memory is just extra tokens in the input. Nothing exotic in the architecture.

Division of labour worth holding onto: **memory supplies what the scene looked like; recent context supplies what is currently moving.** Both are needed, and they answer different questions.

### Writing memory

After chunk one, the encoder $\Phi$ turns selected history latents and their poses into a representation:

$$\mathbf{R} = \Phi(\mathbf{z}^{\mathrm{h}}, \mathbf{C}^{\mathrm{h}}) \in \mathbb{R}^{|\mathbf{z}^{\mathrm{h}}|L \times d}$$

$L$ tokens per history latent frame. $|\mathbf{R}|$ grows linearly with how much history you feed in, so the input is capped at $k = 9$ latent frames.

Implementation details that matter:

- Architecture and weights come from the **LagerNVS encoder**, with its shallow DINO image layers thrown away and a new patch-embedding layer bolted on so it reads VAE latents directly, not pixels.
- History poses are expressed **relative to the most recent latent frame** and injected as camera tokens.
- No explicit 3D reconstruction is ever materialised.

### Choosing which 9 frames

Always keep the latest latent frame. Then greedily pick $k-1$ more whose **joint field of view maximises coverage** of the region the upcoming trajectory will look at.

This is a different objective from what prior work does. WorldMem, Context-as-Memory and HY-WorldPlay rank each candidate frame by how similar its FoV is to the target pose, independently. That tends to pick 4 near-duplicates of the same angle. Max-coverage explicitly wants **complementary** views.

### Reading memory out

Two options were tried, same token budget both ways.

**Pose-free** — compress everything, let the DiT figure out what it needs:

$$\mathbf{M} = \operatorname{Readout}(\Phi(\mathbf{z}^{\mathrm{s}}, \mathbf{C}^{\mathrm{s}}))$$

**Pose-guided** — query with a fixed-size set of poses $\mathbf{C}^{\mathrm{q}} \subset \mathbf{C}$ sampled from the trajectory about to be generated:

$$\mathbf{M} = \operatorname{Readout}(\Phi(\mathbf{z}^{\mathrm{s}}, \mathbf{C}^{\mathrm{s}}), \mathbf{C}^{\mathrm{q}})$$

Pose-guided wins and is what ships. The readout module is initialised from **LagerNVS's shallow decoder layers**, plus projections into DiT token space.

Note what it does *not* do: it never renders a target-view image. LagerNVS's decoder was built to synthesise views; here its early layers are repurposed to produce conditioning tokens and the RGB step is skipped entirely. That is the whole saving over the warp-based methods.

### Camera conditioning

Relative camera geometry enters as a positional transformation inside self-attention, following PRoPE. Concretely they use **UCPE's parallel camera-attention branch**: separate Q/K/V projections, output added back to the main self-attention through a **zero-initialised** projection (so at init the branch is a no-op and the pretrained model is untouched).

Applied **only to the noisy part** $\mathbf{z}_t$. Memory and recent context get no camera injection.

### Training — four stages

Backbone is **Helios-base**. Its original window held a 9-frame noise chunk plus FramePack-style clean history (a compressed 16-frame segment, a 2-frame segment, the latest latent, an attention-sink frame). The compressed 16-frame segment is **removed** and replaced with memory tokens equal in count to 4 uncompressed frames. So memory does not inflate the sequence — it displaces the old crude compression.

| Stage | What trains | Data | Compute |
|---|---|---|---|
| 1 | Fine-tune DiT to the new window; 4 random history frames fill the memory slots | 760k OSP videos, 5k iters | 32 GPUs, batch 32 |
| 2 | UCPE camera branch only, **DiT frozen** | 40k filtered OSP + 6k DL3DV | 32 GPUs, batch 128 |
| 3 | Memory encoder warmup, adapting to VAE latents, 9 frames in | DL3DV + filtered OSP, 5k iters | 16 GPUs, batch 16 |
| 4 | Readout + encoder + DiT + camera branch, **all jointly** | DL3DV + OSP 8k iters, then +MIND synthetic 1k iters | 32 GPUs, batch 32 |

Data: Open-Sora-Plan, DL3DV, and synthetic MIND videos. Poses come from **Depth Anything 3** at metric scale; captions from **Qwen2.5-VL**. They then filter OSP down to clips where the camera follows a moving subject — this is what teaches coordinated camera-plus-subject motion.

### Distillation to real time

Coarse-to-fine pyramid denoising (3 spatial resolutions × 2 steps = 6 steps) with **distribution matching distillation**. Spatial coordinates are rescaled per pyramid level while poses and FoV stay fixed, so camera conditioning survives the resolution changes.

The hybrid trick is worth stealing. Synthetic MIND data improves subject-following but **smears textures**. So they distil two models with different data mixes:

- **low-noise model** — from the pre-synthetic base, on OSP + DL3DV, for natural detail
- **high-noise model** — from the post-synthetic base, on OSP + DL3DV + MIND, for subject-following

At inference the high-noise model does every step except the last; the low-noise model does the final one. Coarse structure gets subject-following, final detail gets natural texture. 16 fps on 4 GPUs.

## Ablation Studies and Experiments

### Benchmark

Built for this paper: 145 images (83 dynamic object-centric, 62 static) from HappyOyster, Project Genie, the web, and GPT-Image2. Each pairs with 5 metric camera trajectories → **725 videos per method**. Trajectories run **528–1,648 frames** and contain deliberate **closed-loop revisits**. All videos resized to $640\times384$ before scoring.

Memory is measured by pairing each revisit frame with the matching first-visit frame and comparing them. Camera accuracy uses VGGT-$\Omega$ to recover the trajectory, then Umeyama Sim(3) alignment against the target.

### Revisit consistency

| Method | MEt3R ↓ | LPIPS ↓ | PSNR ↑ | SSIM ↑ |
|---|---|---|---|---|
| LingBot-World 2 | 0.492 | 0.633 | 10.449 | 0.219 |
| DreamX-World | 0.548 | 0.627 | 12.898 | 0.243 |
| Echo-WM | 0.449 | 0.582 | 12.592 | 0.239 |
| Alaya-EVOKE | 0.414 | 0.565 | 12.332 | 0.290 |
| SANA-WM | 0.397 | 0.553 | 13.142 | 0.246 |
| Matrix-Game 3.5 | 0.405 | 0.549 | 12.976 | 0.224 |
| HY-WorldPlay | 0.394 | 0.515 | 12.983 | 0.252 |
| Lyra 2.0 | 0.334 | 0.487 | 14.050 | 0.390 |
| **WorldCrafter** | 0.166 | 0.255 | 18.016 | 0.517 |
| **WorldCrafter-fast** | **0.129** | **0.186** | **20.868** | **0.616** |

LPIPS 0.487 → 0.255 against Lyra 2.0, PSNR 14.05 → 18.02 dB. The 47.6% headline is the LPIPS drop.

The oddity: **the distilled model beats the full model on every memory metric.** 0.186 vs 0.255 LPIPS. Fewer denoising steps normally cost quality. A plausible read is that 6-step pyramid sampling drifts less from the memory conditioning — fewer chances to hallucinate away from it. The paper does not investigate this, which is a shame, because it is the most interesting number in the table.

### Camera control

| Method | RotErr ↓ | TransErr ↓ | CamMC ↓ |
|---|---|---|---|
| DreamX-World | 54.116 | 2.759 | 3.138 |
| HY-WorldPlay | 34.051 | 2.146 | 2.359 |
| LingBot-World 2 | 30.615 | 2.004 | 2.192 |
| Alaya-EVOKE | 26.042 | 2.042 | 2.199 |
| SANA-WM | 23.531 | 1.740 | 1.887 |
| Echo-WM | 21.455 | 2.072 | 2.189 |
| Matrix-Game 3.5 | 19.881 | 1.920 | 2.028 |
| Lyra 2.0 | 16.145 | 1.538 | 1.624 |
| **WorldCrafter** | **13.536** | **1.475** | **1.546** |
| WorldCrafter-fast | 18.251 | 1.638 | 1.737 |

Here the ordering flips — the full model wins, fast drops to third. So distillation trades camera precision for recall consistency.

### Visual quality (VBench)

Best in 5 of 8 dimensions, overall 81.910 (next: Alaya-EVOKE 81.406). But look at **Dynamic Degree**: WorldCrafter scores 96.893 and WorldCrafter-fast 96.505, the **two lowest in the table** — LingBot and Matrix-Game hit 100.000. The model moves less. Some of the consistency is bought by being calmer, and VBench's aggregate hides that because consistency and dynamism are both in the average.

Aesthetic Quality (61.4) and Imaging Quality (69.1) are also mid-table, below SANA-WM and Alaya-EVOKE. The wins are consistency-shaped, not beauty-shaped.

### The ablations — four variants, one change each, budget held fixed

| Variant | MEt3R ↓ | LPIPS ↓ | PSNR ↑ | SSIM ↑ | RotErr ↓ | TransErr ↓ | CamMC ↓ |
|---|---|---|---|---|---|---|---|
| Context memory | 0.382 | 0.497 | 13.907 | 0.315 | 26.522 | 2.083 | 2.150 |
| Frozen memory encoder | 0.227 | 0.305 | 16.873 | 0.472 | 15.428 | 1.793 | 1.886 |
| Pose-free readout | 0.251 | 0.333 | 16.486 | 0.467 | 18.307 | 1.701 | 1.828 |
| Similarity-based retrieval | 0.213 | 0.296 | 17.125 | 0.485 | 14.657 | 1.579 | 1.664 |
| **Full model** | **0.166** | **0.255** | **18.016** | **0.517** | **13.536** | **1.475** | **1.546** |

Ranked by how much each component contributes:

**1. Implicit memory vs raw context frames — by far the biggest.** Swap the encoded memory for 4 retrieved history latents and LPIPS goes 0.255 → 0.497, essentially back to baseline territory (Lyra sits at 0.487). This is the paper. And the time-resolved plot shows the gap *widens* with revisit interval — raw context degrades faster, so this is not a constant offset but a difference in decay rate.

**2. Joint optimisation.** Freezing the encoder in stage 4 costs 0.255 → 0.305. The validation curve shows joint training reaching lower revisit error *earlier at matched iterations* and keeping the lead, so this is not just extra capacity. Learning to consume a fixed representation space is worse than co-adapting it.

**3. Pose-guided readout.** 0.255 → 0.333 if you drop the pose queries. Note the interesting split: pose-free hurts camera control badly (RotErr 13.5 → 18.3) but its TransErr (1.701) is actually *better* than similarity-retrieval's (1.579 — no, worse; but better than frozen-encoder's 1.793). The rotation error is where pose-guidance earns its keep — which makes sense, since knowing where you will be pointing is exactly rotational information.

**4. Max-coverage retrieval — smallest effect.** Similarity ranking gets 0.296 vs 0.255. Real but modest, and note this comparison feeds *both* variants 9 frames. So the retrieval rule is a refinement; the encoder is what does the work.

### What did not work

- **Pose-free readout.** The clean hypothesis "give the DiT everything, self-attention will find what it needs" loses on all seven metrics. Under a tight budget, the compressor has to know the question.
- **Similarity-based frame ranking.** Independently scoring each candidate against the target pose is worse than jointly maximising coverage. Ranking each view on its own merit picks redundant views.
- **Synthetic data in distillation, naively.** MIND data buys subject-following and costs smeared textures. There was no single data mix that gave both — hence the two-model hybrid. That is a workaround, not a solution.
- **Geometry-pretrained encoders.** Not directly ablated in-house, but the design rationale is explicitly a rejection of the VGGT-features route (CineScene, GIM-World): geometry pretraining optimises for geometric prediction, not the appearance fidelity recall needs.

### Efficiency

At $640\times384$, 9-latent-frame chunk, 4 history frames warped, VAE decode and denoising excluded:

| Spatial memory (depth + warp) | WorldCrafter |
|---|---|
| Depth estimation + alignment: 0.409 s | Memory encoding: 0.049 s |
| Batched warping: 0.937 s | Readout: 0.013 s |
| **Total 1.346 s** | **Total 0.062 s** |

21.7×. The warping, not the depth estimation, is the bigger cost — worth knowing if you were planning to optimise the depth model.

## Worth Remembering

**Limitations the authors state.** Consistency still breaks on particularly complex or extended trajectories — this buys you minutes, not unbounded exploration. And history is **re-encoded from scratch at every chunk**, which is wasted work; they name an autoregressive streaming memory encoder that folds each new chunk into a running state as the obvious fix. That is essentially asking for a [[Bayes Filter]]-shaped update instead of a full recompute — predict, then update, rather than re-derive.

**The transfer-learning choice is the reusable lesson.** Picking a novel-view-synthesis encoder over a geometry encoder because the downstream task needs appearance, not just structure, is a good instance of matching pretraining objective to actual requirement. Related to the general warning in [[Shortcut Learning in Deep Neural Networks]]: a model optimised for one proxy will be good at that proxy and not necessarily at what you wanted.

**The fixed budget is the honest framing.** Everything here follows from "you have 4 frames' worth of tokens and an unbounded past." Raw frames spend the budget on 4 viewpoints. Depth-and-warp spends it on a geometric projection that assumes the world is static. Pose-guided implicit memory spends it on a query-specific summary. Same budget, three different allocation policies — this is a compression problem wearing a video-generation costume.

**Practical caveats if you wanted to use it.**

- 32 GPUs for 4 stages, and stage 1 alone is 760k videos. Not a fine-tune you do on a weekend.
- 4 GPUs for 16 fps inference.
- Every training source needed camera poses from Depth Anything 3. Pose annotation is a hard dependency even though the *runtime* system needs no depth. The geometry cost moved from inference to data prep; it did not disappear.
- Camera poses are required at inference too — this is continuous-pose control, not discrete WASD actions.
- Dynamic Degree is the lowest in the comparison table. If your application needs lively motion, check this before adopting.

**Connections.** The memory tokens sitting in the same sequence as the video tokens, distinguished only by which ones get camera injection, is a plain instance of [[Cross Attention|conditioning by concatenation]] — the same move as [[Latent Diffusion]] putting text tokens into the same attention as image tokens. The zero-initialised camera branch is the [[LoRA]] / ControlNet safety pattern: start as identity, so pretrained behaviour is untouched at step zero. And the two-model hybrid distillation resembles the coarse/fine split in [[Diffusion Sampling]], except the split is drawn over *training data* rather than step count.

**Open questions worth chasing.**

- Why does the distilled model remember better? 0.186 vs 0.255 LPIPS is a large gap in the wrong direction and it goes unexplained. If fewer steps really means less drift from conditioning, that is a general result about [[Consistency Models|few-step distillation]], not a WorldCrafter fact.
- How much does WorldCrafter improve when you scale $k$ past 9 input frames? The encoder is linear in $|\mathbf{z}^{\mathrm{h}}|$, so this is a knob with a known cost and an unreported benefit.
- Memory is rewritten but never *edited*. If the scene genuinely changes — a door is opened, an object is moved — the archive holds both states and nothing arbitrates. The benchmark tests revisit consistency, which rewards remembering and never penalises failing to forget.
- Max-coverage retrieval is greedy over FoV overlap. It knows nothing about content salience: a blank wall and a crowded bookshelf count the same.

## Links
Related: [[Video Diffusion]] · [[Diffusion Transformer]] · [[Flow Matching]] · [[Auto-regressive models]] · [[Attention]] · [[Cross Attention]] · [[Memory]] · [[Latent Diffusion]] · [[Consistency Models]] · [[Distillation]] · [[Bayes Filter]] · [[State-Space Model]] · [[Shortcut Learning in Deep Neural Networks]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Linear Attention]]

New topics worth writing: Novel view synthesis, Distribution matching distillation, Camera positional encoding (PRoPE / UCPE / Plücker rays), LPIPS and perceptual metrics, MEt3R multi-view consistency, VBench, Depth Anything and monocular depth estimation, VGGT and feed-forward 3D reconstruction, Umeyama Sim(3) alignment, FramePack context packing, Interactive world models, Pyramid denoising
