---
title: "Honeycomb: Constant-Size Scene Memory Representation for Video World Models"
authors: ["Jack Wei Lun Shi", "Kaichen Zhou", "Haoyu Chen", "Yufeng Weng", "Keane Ong", "Ruojin Cai", "Hang Hua", "Justin K. W. Yeoh", "Mengyu Wang"]
year: 2026
arxiv: "2609.37690"
url: https://arxiv.org/abs/2609.37690
priority: Good-To-Read
read_on: 2026-10-05
tags: [paper, optimization, diffusion, vision, theory]
---
## The Core Idea

A **video world model** is a video generator you steer with a camera path. You give it one photo and a trajectory, and it paints what the camera would see. The hard part is not making pretty frames. It is *coming back*. Walk the camera down a corridor and turn around — the sofa should still be the same sofa.

It forgets because the generator only sees a few recent frames (its [[Context Window]]). Anything older is gone.

The usual fix is a **spatial memory**: store what you saw, tagged with where in 3D you saw it, and re-project it into the next view. Spatia stores RGB pixels in a point cloud. LSM-World stores diffusion latents attached to 3D points. Both work. Both grow forever — every new chunk of video adds more points.

Honeycomb's claim: you do not need a growing store. Put the whole scene into **six fixed-size 2D feature grids** and never let them get bigger.

> [!NOTE] HexMemory
> Six 2D planes of features that together stand in for a 4D volume (3 space axes $x,y,z$ plus write-time $\tau$). Three *spatial* planes $S_{XY}, S_{XZ}, S_{YZ}$, and three *spatiotemporal* planes $T_{ZT}, T_{YT}, T_{XT}$. Each pair covers all four axes exactly once. Grid sizes are fixed for the whole rollout. ^hexmemory

This is a **low-rank factorisation**: instead of a dense 4D grid (which would be enormous), you store 2D slices and multiply them together to get a feature at any point. The trick is borrowed from HexPlane, a 2023 dynamic-scene reconstruction method — but HexPlane fit its planes *per scene* with gradient descent, taking seconds per update. Honeycomb replaces that with one small feed-forward network shared across all scenes, so a write costs 13 ms.

What it unlocks: memory cost stops depending on how long you generate. Write cost stops depending on how much you have already seen. Both become constants. 73.9 MB of feature storage, chunk 1 or chunk 100.

The price is that fixed grids covering a growing region get coarser. Honeycomb accepts that trade and shows it costs surprisingly little.

## The Methodology

### Reading a feature out of the six planes

Given a 3D world point $\bm{p} = (x,y,z)$ and a write time $\tau$:

$$\phi(\bm{p},\tau) = \big[\, S_{XY}(x,y) \odot T_{ZT}(z,\tau)\;;\; S_{XZ}(x,z) \odot T_{YT}(y,\tau)\;;\; S_{YZ}(y,z) \odot T_{XT}(x,\tau) \,\big] \in \mathbb{R}^{3R}$$

Each plane is sampled by bilinear interpolation. $\odot$ is elementwise multiply. $R = 48$ is the feature rank, so $\phi$ is 144-dimensional.

Read it in words: each of the three terms pairs a *space-only* lookup with a *space-and-time* lookup, and multiplying them gates the spatial content by when it was written. The spatial planes carry content shared across all write times; the products let it vary with time.

The memory also stores:
- a bounding box $B$ over world space and write time, used to normalise all coordinates into $[-1,1]$
- a **confidence map** $N$ per plane — the total bilinear weight each cell has received so far

### The writer: how observations become plane features

After a chunk is generated, decode its frames, run a monocular depth estimator, and backproject every valid latent cell into a 3D world point. Each point $i$ carries its latent token $\mathbf{f}_i$, camera centre $\bm{o}_i$, ray direction $\bm{d}_i$, position $p_i$, and write time $\tau_i$.

A small network maps each point's attributes to one **contribution vector** $\bm{c}_i^P$ per plane $P$. That contribution gets *splatted* onto the plane at the point's two relevant coordinates — e.g. at $(x_i,y_i)$ on $S_{XY}$ and at $(z_i,\tau_i)$ on $T_{ZT}$ — spread over the four surrounding cells with bilinear weights $w_{im}$.

Each cell averages what lands on it:

$$\bar{\mathbf{C}}^P_m = \frac{\sum_i w_{im}\,\bm{c}_i^P}{\sum_i w_{im}}, \qquad N_m = \sum_i w_{im}$$

Empty cells are zero-filled. Lightweight networks then turn the averaged contributions plus accumulated weights into the six final feature planes. The weights double as the confidence maps for later fusion.

This is splatting, not optimisation — one forward pass, no per-scene fitting.

### The recurrent update: fusing new into old

Each chunk, only the *new* points go through the writer. Then:

1. **Expand the bounds.** Grow $B_{t-1} \to B_t$ to cover the new observations; extend the time range in whole-chunk steps.
2. **Warp the old planes.** Bilinear-resample the previous planes and confidence maps into the new coordinate range, *keeping grid dimensions fixed*. Newly covered cells get zero confidence; spatial planes get zero features, spatiotemporal planes get one (so the multiply is a no-op).
3. **Confidence-weighted pool.** With $P^o$ the warped old plane and $P^n$ the new one:

$$\bar{P} = \frac{N^o P^o + N^n P^n}{N^o + N^n}$$

Cells with zero total confidence keep the old value.

4. **Learned residual correction.**

$$P_t = \bar{P} + h_P\big(P^o, P^n, \bar{P}, N^o, N^n\big), \qquad N_t = N^o + N^n$$

$h_P$'s output layer is **zero-initialised**, so training starts at pure confidence-weighted averaging and learns a correction on top. Same safety trick as [[LoRA#^lora-init|LoRA's zero-init B]] and [[Diffusion Transformer#^adaln-zero|adaLN-zero]] — the new module begins as identity, so it cannot wreck the model on step one.

Note what step 2 does to resolution: as the camera explores, the same number of grid cells now covers more space. The memory gets blurrier rather than bigger.

### The reader: turning memory back into conditioning

Before generating a chunk, project all stored 3D points into the target view at latent resolution ($44 \times 80$). For each latent cell, keep the nearest projected point in front of the camera. A **visibility mask** $m^t$ marks which cells got a point.

For an occupied cell $(u,v)$ with selected point $i$:

$$\hat{\bm{z}}^t(u,v) = g\big(\phi(\bm{p}_i, \tau_i),\; \bm{d}^t_{uv},\; \bm{o}^t\big)$$

$g$ is a shared reader network; $\bm{d}^t_{uv}$ and $\bm{o}^t$ are the *target* view's ray direction and camera centre. Empty cells get zeros. The reconstructed latent map $\hat{\bm{z}}^t$ (48 channels) and mask $m^t$ feed the denoiser.

### Training the memory model

Writer, fusion networks $h_P$ and reader are trained **jointly** on a latent reconstruction objective — mean squared error in normalised latent space between the reconstructed latents and the original tokens at visible points.

Each clip: write the input frame, then two chunks, matching the inference rollout schedule. After each write, query from the clip's camera views and compute the loss. This joint training is what forces the planes to *keep* information rather than just summarising it — the reader has to be able to get the latent back out.

### The generator

- **Backbone:** Wan2.2, 5B parameters, camera-controllable, frozen initially.
- **Conditioning path:** a VACE/[[Classifier-Free Guidance|ControlNet]]-style side branch, 8 blocks, attached to every 4th backbone block starting from the first, each initialised from the backbone block it attaches to.
- **Chunks:** 9 latent frames = 33 RGB frames at $704 \times 1280$, overlapping. Each chunk's first latent is pinned to the encoded input frame (chunk 1) or the shared boundary latent from the previous chunk.
- **Data:** RealEstate10K. Poses, intrinsics and depth from ViPE with Depth Anything 3 — the same estimator used at inference for write-back.

Two training stages:

| Stage | What trains | Iterations | LR |
|---|---|---|---|
| 1 | ControlNet branch, backbone frozen | 10,000 | $10^{-5}$ |
| 2 | Backbone via [[LoRA]] rank 64 on attention + FFN, branch frozen | 5,000 | $10^{-4}$ |

AdamW, effective batch 64, H200 GPUs. Inference: UniPC scheduler, 40 steps.

### Noise augmentation on the preceding latents

Classic [[Imitation Learning#^errors-compound|exposure bias]] fix. In training, the preceding frames are ground truth; at inference they are the model's own output. To close that gap, stage 2 noises the preceding latents:

$$\widetilde{z}_P = (1-\sigma_{\text{aug}}) z_P + \sigma_{\text{aug}} \epsilon, \qquad \sigma_{\text{aug}} = \frac{t_{\text{aug}}}{1000},\quad t_{\text{aug}} \sim \mathcal{U}(0,50),\; \epsilon \sim \mathcal{N}(0,I)$$

One noise level shared across the sample's 8 preceding frames. Stage 1 uses clean latents. Borrowed directly from Spatia.

One more detail worth noting: **Honeycomb writes everything to memory.** Spatia and LSM-World filter out dynamic objects and sky before writing, because a moving person stored as static geometry corrupts the scene. Honeycomb does not filter, and still wins — so the filtering is doing less work than you would guess, or the learned fusion absorbs the damage.

## Ablation Studies and Experiments

Three evaluations: generation quality (WorldScore, 3,000 image-to-video samples), novel-view synthesis (100 RealEstate10K test videos, ground-truth trajectory), and **closed-loop** revisit — 100 WorldScore scenes, camera leaves and returns to the start, compare final frame to the input image.

### WorldScore

| Method | Average | Static | Dynamic | 3D Const | Photo Const | Style Const |
|---|---|---|---|---|---|---|
| WonderJourney | 54.19 | 63.75 | 44.63 | 80.60 | 79.03 | 62.82 |
| WonderWorld | 61.79 | 72.69 | 50.88 | **86.87** | **85.56** | 70.57 |
| Spatia | 63.21 | 64.88 | 61.54 | 83.26 | **89.09** | 83.33 |
| LSM-World | 61.20 | 62.69 | 59.70 | 80.88 | 76.10 | – |
| Wan2.1 (no memory) | 55.21 | 57.56 | 52.85 | 78.74 | 78.36 | 77.18 |
| **Honeycomb** | **65.52** | 68.01 | **63.03** | 82.29 | 85.76 | **84.21** |

Best average, best dynamic score. *Not* best on 3D consistency (WonderWorld's 86.87) or photometric consistency (Spatia's 89.09) — those methods keep explicit geometry or raw pixels, which should win on exactly those axes. Honeycomb's subject quality (46.28) is the lowest in the table; Wan2.1 with no memory scores 59.38. Memory conditioning appears to cost per-frame crispness.

### Novel-view synthesis and closed-loop

| Method | RE10K PSNR↑ | SSIM↑ | LPIPS↓ | Closed PSNR↑ | SSIM↑ | LPIPS↓ | Flow err↓ |
|---|---|---|---|---|---|---|---|
| ViewCrafter | 12.28 | 0.512 | 0.571 | 12.32 | 0.369 | 0.574 | 30.78 |
| FlexWorld | 13.17 | 0.567 | 0.544 | 12.86 | 0.430 | 0.602 | 55.77 |
| Voyager | 14.67 | 0.577 | 0.493 | 15.99 | 0.459 | 0.423 | 7.11 |
| Spatia | 15.58 | 0.616 | 0.390 | 15.67 | 0.488 | 0.353 | 6.64 |
| LSM-World | 17.46 | 0.636 | 0.452 | 15.12 | 0.460 | 0.463 | 27.05 |
| **Honeycomb** | **18.45** | **0.674** | **0.274** | **17.22** | **0.504** | **0.311** | **3.00** |

Clean sweep. +0.99 dB NVS over LSM-World, +1.23 dB closed-loop over Voyager (the next best there). Flow error — RAFT optical-flow magnitude in pixels between input and final frame — drops from 6.64 to 3.00. That is the most direct revisit measurement: how far did the scene physically drift?

Note LSM-World's inconsistency: good at NVS (17.46) but bad at revisits (15.12 PSNR, 27.05 flow error). Accumulating latent points helps nearby views and does not survive a round trip.

### Resolution ablation — the memory can be much smaller

| Plane resolution | HexMemory (MB) | Closed PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|
| 512 (default) | 73.9 | 17.22 | 0.504 | 0.311 |
| 384 | 42.6 | 17.14 | 0.503 | 0.313 |
| 256 | 19.8 | 17.10 | 0.500 | 0.319 |
| 128 | 5.8 | 16.77 | 0.490 | 0.338 |

Cutting storage by 73% (73.9 → 19.8 MB) costs 0.12 dB. Halving again to 5.8 MB costs 0.45 dB — the knee is between 256 and 128. The scene information needed for a consistent revisit is genuinely low-dimensional.

### Writer ablation — the one that justifies the paper

| Writer | Write time, chunk 2 | chunk 5 | chunk 9 | Closed PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|---|---|
| Direct optimisation (HexPlane-style) | 3,217 ms | 3,228 ms | 3,130 ms | **17.44** | **0.510** | **0.299** |
| Replacement (rebuild from all points) | 11.2 ms | 23.3 ms | 40.1 ms | 17.16 | 0.502 | 0.313 |
| **Recurrent (theirs)** | 13.1 ms | 13.2 ms | 13.2 ms | 17.22 | 0.504 | 0.311 |

Three readings:

1. **Per-scene gradient descent is still the best quality** — 17.44 dB, 0.22 dB above recurrent — and roughly 240× slower per write. The feed-forward writer is a speed/quality trade, not a free win. Honest of them to report it.
2. **Recurrent slightly beats replacement** (17.22 vs 17.16). You lose nothing by never looking at the old points again. Confidence-weighted pooling plus the residual correction is as good as re-reading history.
3. **Replacement's cost is visibly linear** — 11 → 23 → 40 ms over chunks 2, 5, 9. Recurrent is flat at 13 ms. At chunk 50 the gap would be large.

### Retention probe — does early content survive?

129-frame sequence. Write the input frame, then four ground-truth chunks. After each write, reconstruct chunk 1 and measure PSNR (excluding the pinned input frame).

| Memory state | Chunk-1 reconstruction PSNR |
|---|---|
| Memory disabled | 12.29 dB |
| After write 2 (chunk 1 just stored) | 16.71 dB |
| After write 3 | 16.67 dB |
| After write 4 | 16.35 dB |
| After write 5 | 16.30 dB |

Total decay 0.41 dB over three extra writes, ~0.14 dB per write. Still 4.01 dB above no-memory. Forgetting is real but slow in this test.

**Important caveat:** this is four writes on one sequence. It tells you nothing about chunk 50. A 0.14 dB/write linear decay would hit the no-memory baseline around write 30 — and the compounding here is the same shape as [[Agentic Workflows#^compounding-error|compounding error in long agent rollouts]].

### What they did not test

- No rollout longer than ~9 chunks reported anywhere.
- No ablation separating the learned residual $h_P$ from plain confidence-weighted pooling. Since $h_P$ is zero-initialised, that is the natural control — and it is missing.
- No ablation on rank $R=48$.
- No ablation on removing the spatiotemporal planes, which would answer whether the time axis earns its three planes.
- Nothing on what happens when depth estimation fails.

## Worth Remembering

**The low-rank factorisation is the whole mechanism.** A dense 4D feature grid at resolution 512 with $R=48$ would be astronomically large. Six 2D planes at $512^2 \times 48$ is 73.9 MB. The multiplicative read $\phi$ is what reconstitutes 4D from 2D slices. Same structural move as [[LoRA#^intrinsic-rank|LoRA's low-rank $\Delta W$]] or [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)#^latent-factor|matrix factorisation in recsys]]: assume the thing you want is low-rank and store the factors.

**"Fixed-size" is fixed-*storage*, not fixed-fidelity.** As the camera explores, bounds expand and the same grid covers more space at coarser resolution. The memory degrades gracefully rather than growing. Whether that degradation stays graceful over hundreds of chunks is untested — and this is the main thing you would want to know before shipping it.

**The confidence maps are doing real work.** $N$ is just accumulated bilinear weight, but it is the mechanism that makes fusion sensible: well-observed cells resist being overwritten by a thin new observation. A free, non-learned [[Uncertainty#^epistemic|epistemic uncertainty]] signal, used as a fusion weight.

**Depth estimation is an unaudited dependency.** Every write backprojects using monocular depth from Depth Anything 3. Bad depth puts features at the wrong world coordinates, permanently. The paper never ablates depth quality. In a reflective hallway or on a textureless wall, this is where it would break first.

**The subject-quality regression is the honest cost.** 46.28 vs 59.38 for Wan2.1 with no memory. Conditioning on blurry retrieved latents pulls the generator toward the memory's fidelity ceiling — the same ceiling argument as [[Latent Diffusion#^vae-is-the-ceiling|the VAE being the ceiling in latent diffusion]].

**Comparisons are self-run with baselines at default settings.** Standard practice, and standard risk — see [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] and [[On the Difficulty of Evaluating Baselines]] for why under-tuned baselines are the most common way a result inflates.

**The write is a learned [[Bayes Filter|filter update]].** Warp the prior into the new coordinate frame, pool with the new measurement weighted by confidence, correct. That is predict-then-update with a learned correction term bolted on. Honeycomb never calls it that, but the shape is identical.

**Open questions worth chasing:**
- Does the 0.14 dB/write decay stay linear, or does it plateau? The whole long-horizon claim turns on this.
- How much of the win is HexMemory versus the joint writer–reader reconstruction training? A fixed-size baseline trained the same way would settle it.
- Can the time axis be dropped? For a purely static scene the three spatiotemporal planes look like overhead.
- The planes are differentiable and fixed-size, so they could be a state for a policy, not just conditioning for a generator. That is the bridge to [[Diffusion Policy]] and [[Model Predictive Control]] — a constant-size scene state is exactly what a planner wants.

## Links
Related: [[Diffusion Models]] · [[Video Diffusion]] · [[Latent Diffusion]] · [[Conditional Generation]] · [[Classifier-Free Guidance]] · [[LoRA]] · [[Diffusion Transformer]] · [[Context Window]] · [[Memory]] · [[Bayes Filter]] · [[State-Space Model]] · [[Linear Attention]] · [[Auto-regressive models]] · [[Imitation Learning]] · [[Flow Matching]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[WorldCrafter- Consistent Video World Model with Implicit 3D-aware Memory]] · [[The Past Frames the Future- Memory for Autoregressive Video Generation]] · [[Training Object Permanence in World Models]] · [[U-Net]] · [[Evaluating Generative Models]] · [[Agentic Workflows]] · [[Uncertainty]] · [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)]]

New topics worth writing: HexPlane and K-Planes plane factorisation, Neural radiance fields, 3D Gaussian splatting, Bilinear splatting as a differentiable scatter op, Monocular depth estimation, Novel-view synthesis metrics (PSNR/SSIM/LPIPS), RAFT optical flow, WorldScore benchmark, RealEstate10K, Point cloud scene memory, Camera-conditioned video generation, Closed-loop revisit evaluation, UniPC sampler, Wan video diffusion backbone
