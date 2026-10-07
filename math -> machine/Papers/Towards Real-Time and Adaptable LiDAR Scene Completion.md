---
title: "Towards Real-Time and Adaptable LiDAR Scene Completion"
authors: ["Azhar Hussian", "Martin Vossiek", "Vasileios Belagiannis"]
year: 2026
arxiv: "2608.16490"
url: https://arxiv.org/abs/2608.16490
priority: Low-Priority
read_on: 2026-10-01
tags: [paper, transformers, diffusion, vision]
---
## The Core Idea

LiDAR sensors give you a sparse, holey picture of the world. Far away, points thin out. Behind a parked van, there is nothing at all. **Scene completion** is the job of filling the holes: take a partial scan of ~18,000 points, output a dense scene of ~180,000 points that looks like the real geometry.

Nearly every method does this in two steps: **initialize, then refine**. Make a rough guess at where the 180,000 points go, then push each one to a better place.

The interesting claim of this paper is that *the rough guess is where all the speed and all the coverage comes from*, and everybody has been picking it badly.

Two bad choices existed before:

1. **Start from pure Gaussian noise** (LiDiff, ScoreLiDAR, LiDPM). The starting point knows nothing about the scene, so you need hundreds of denoising steps to walk it back to something real. 30 seconds per scan for LiDiff. Even the distilled version takes 7.1 s. Useless for a car.
2. **Start from the input points, jittered** (LiNeXt). Copy each observed point 10 times, add noise with a fixed standard deviation, refine once. Fast — 0.23 s — but every point stays within a small ball of something you already saw. If a whole region is occluded, no point ever lands there, so the refiner has nothing to work with. And the noise scale is a hand-tuned number that depends on the sensor's density, so it must be re-tuned per dataset.

RapidLiDAR's move: **let a network predict the displacement**. Each copied point gets its own learned offset, conditioned on the local scene structure. Points near a big empty region get pushed far, out into the gap. Points in an already-dense area barely move. No noise hyperparameter to tune.

> [!NOTE] Adaptive initialization
> Instead of scattering the starting points with fixed random noise, a small network reads scene features at each point and outputs a 3D displacement. The spread becomes data-dependent: big where the scene is missing, small where it is already covered. ^adaptive-initialization

The second idea is a plumbing one, and it buys the speed. Point-cloud architectures traditionally use **farthest point sampling** (repeatedly pick the point furthest from everything picked so far) and **$k$-nearest-neighbour search** to build local neighbourhoods. Both cost time that grows badly with point count, and outdoor scenes have hundreds of thousands of points. RapidLiDAR throws them out entirely. All features come from **voxel grids** (a regular 3D lattice of cells) and **BEV maps** (bird's-eye view — the 3D grid squashed down onto a flat top-down image). Looking up a feature for a point is then just an interpolation into a grid, which is $O(1)$ per point and perfectly parallel.

Result: a full scene in **0.1 s**, $2.3\times$ faster than LiNeXt, matching the 10 Hz spin rate of a real automotive LiDAR. And because grid lookup does not care how many points you feed it, the same trained model handles different input resolutions without retraining.

## The Methodology

Input $X \in \mathbb{R}^{M\times 3}$ with $M = 18{,}000$. Output $P \in \mathbb{R}^{N\times 3}$ with $N = 180{,}000$. One forward pass.

**Step 1 — multi-scale feature extraction.**

Voxelize $X$ at $\eta = 0.3$ m into an occupancy grid of shape $[1, 20, 333, 333]$ (one channel: occupied or not). Run 3D convolutions, each halving the resolution, giving four feature volumes $F_1 \dots F_4$ with channel widths $[32, 64, 128, 256]$.

These volumes are mostly empty — LiDAR is sparse, so almost no voxel is occupied. That is a problem, because the whole point is to reason about the *empty* regions. So a **dense BEV head** is added on top of $F_4$:

- Merge the channel and depth axes: $(C_4, d_4, h_4, w_4) \to (C_4 \cdot d_4, h_4, w_4)$.
- One 2D convolution projects to $C_{\text{out}} = 512$ channels.
- Treat each of the $h_4 \times w_4$ spatial cells as a token, run **multi-head self-attention** over them (4 heads). This is the only global-context step in the model — it lets a cell that sees nothing borrow meaning from cells that do.
- Reshape back, two more convs with a skip connection. Output: $B_{\text{dense}}$.

So the scene is represented twice: sparse-but-sharp 3D voxels, and dense-but-flat BEV.

**Step 2 — adaptive initialization.**

Build the starting set $\tilde{P}$: repeat each input point $\lfloor N/M \rfloor = 10$ times, jitter each copy with tiny noise $\sigma_{\text{init}} = 0.1$ m (only to break exact duplicates), concatenate with $X$. Now $|\tilde{P}| = N$.

For each point $p_i \in \tilde{P}$, gather a feature by **interpolating from the grids**:

$$f_i = \text{concat}\big(f_i^{(1)}, f_i^{(2)}, f_i^{(3)}, f_i^{(4)}, b_i\big) \in \mathbb{R}^{992}$$

Trilinear interpolation for the four 3D volumes (weights spread over the 8 neighbouring voxels), bilinear for the BEV map (4 neighbouring cells). Interpolation rather than hard indexing, because indexing is not differentiable — see [[Backpropagation]]; gradients need to flow back through the lookup into the convolutional features.

An MLP maps $f_i$ to a displacement $\Delta \in \mathbb{R}^{N \times 3}$, and:

$$P_{\text{init}} = \tilde{P} + \Delta \cdot S_{\max}$$

$S_{\max} = 50$ m is a fixed scale factor so the MLP's bounded output covers the whole scene.

**Step 3 — multi-scale reconstruction.**

Re-interpolate features at the *new* positions $P_{\text{init}}$, giving queries $\mathcal{F} \in \mathbb{R}^{N \times d}$, projected to width 512.

Now refine using [[Attention|attention]] — but full [[Cross Attention|cross-attention]] from 180,000 queries onto every grid cell is $O(N \cdot K_{\text{ctx}})$ and hopeless. So they use **multi-scale deformable attention** (from Deformable DETR): each query predicts a handful of 2D offsets and only samples the feature map at those few locations.

> [!NOTE] Deformable attention
> Instead of every query attending to every position, each query *predicts where to look*: a small set of sampling coordinates, read out by bilinear interpolation, then weighted and summed. Cost per query is fixed ($K$ samples), not proportional to the size of the feature map. ^deformable-attention

Because the deformable-attention CUDA kernels are written for 2D, each voxel volume $F_i$ is also flattened to a BEV map $B_i$ (same channel-and-depth merge trick, 512 channels out). Five levels total: $\{B_1, B_2, B_3, B_4, B_{\text{dense}}\}$.

$$F_{\text{ref}} = \text{MS-DeformAttn}\big(\mathcal{F},\ \pi(P_{\text{init}}),\ \{B_1,\dots,B_4,B_{\text{dense}}\}\big)$$

where $\pi(\cdot)$ drops each 3D point onto the BEV plane to give its reference coordinate. Two deformable stages, 8 heads, $K = 4$ sampling points per head. An MLP then predicts a residual:

$$P = P_{\text{init}} + \Delta P_{\text{ref}}$$

**Loss.** Nothing exotic — symmetric **Chamfer Distance** between prediction and ground truth:

$$\mathcal{L}_{\text{CD}}(P, P_{\text{gt}}) = \frac{1}{|P|}\sum_{p \in P} \min_{q \in P_{\text{gt}}} \|p - q\|_2^2 \;+\; \frac{1}{|P_{\text{gt}}|}\sum_{q \in P_{\text{gt}}} \min_{p \in P} \|q - p\|_2^2$$

The first term punishes predicted points that sit far from any real surface (hallucination). The second punishes real surfaces with no predicted point nearby (missing coverage). Both stages trained end-to-end on this single objective — there is no separate supervision on $P_{\text{init}}$.

**Training.** [[Adam- A Method for Stochastic Optimization|Adam]], base LR $1\times10^{-4}$, cosine schedule, batch size 4, one RTX 6000 Ada, converges in 30 epochs. Train on SemanticKITTI sequences 00–10 minus 08 (08 is validation).

**Refinement network.** Prior work adds a second network that upsamples the completed scene; to compare fairly, they add one too. It freezes the completion model, reuses its multi-scale features, runs four more deformable cross-attention stages, and predicts $\kappa = 6$ offsets per point ($6\times$ upsampling). Trained separately for 5 epochs on the same Chamfer loss.

## Ablation Studies and Experiments

**Main results.** SemanticKITTI validation (seq 08), and KITTI-360 seq 00 evaluated **zero-shot** — the model trained on SemanticKITTI, no fine-tuning. Metrics: Chamfer Distance (lower better), and Jensen–Shannon divergence between predicted and true point distributions in 3D and in BEV.

| Method | SemKITTI CD | JSD 3D | JSD BEV | KITTI-360 CD |
|---|---|---|---|---|
| LiDiff (diffusion) | 0.434 | 0.564 | 0.444 | 0.564 |
| LiDPM | 0.446 | 0.532 | 0.440 | — |
| ScoreLiDAR (distilled) | 0.406 | — | 0.425 | 0.472 |
| LiFlow (flow matching) | 0.309 | — | 0.416 | — |
| LiNeXt (single-pass) | 0.214 | 0.494 | 0.336 | 0.217 |
| **RapidLiDAR** | **0.206** | **0.475** | **0.332** | **0.211** |

With the refinement network bolted on: LiNeXt$^\dagger$ 0.149, RapidLiDAR$^\dagger$ **0.138**.

Read honestly: the quality gap over LiNeXt is small. 0.214 → 0.206 CD is ~4%. This is not a quality paper.

**The speed table is the paper.**

| Method | CD | Params (M) | Time (s) |
|---|---|---|---|
| LiDiff | 0.434 | 32.67 | 30.1 |
| ScoreLiDAR | 0.406 | 32.67 | 7.1 |
| LiNeXt | 0.214 | 1.99 | 0.23 |
| **RapidLiDAR** | **0.206** | 11.8 | **0.10** |

Note the trade they took: **6× the parameters of LiNeXt, but 2.3× faster**. That is the point. LiNeXt is tiny but spends its time in FPS and $k$-NN, which are serial, memory-bound neighbourhood searches — even with hand-written CUDA kernels. RapidLiDAR is bigger but every operation is a dense conv or a grid interpolation, which is what a [[GPU processing|GPU]] is actually good at. And the authors note their implementation uses only standard library ops — no custom kernels.

**Component ablation** (SemanticKITTI val):

| Variant | CD | JSD 3D | JSD BEV |
|---|---|---|---|
| Full | **0.206** | **0.475** | **0.332** |
| w/o adaptive init (fixed $\sigma = 1.0$, LiNeXt-style) | 0.218 | 0.488 | 0.345 |
| w/o multi-scale reconstruction (predict directly from $P_{\text{init}}$) | 0.215 | 0.494 | 0.342 |

Both components contribute, and roughly equally — removing either costs about 0.01 CD. Which is to say: **neither module is individually dramatic.** The headline claim that "initialization is what matters" is supported qualitatively (the figures show real coverage of occluded regions) far more than numerically. If you only cared about CD, the adaptive initialization buys 5.5%.

**$S_{\max}$ sweep** — the displacement scale:

| $S_{\max}$ | CD |
|---|---|
| 50 | 0.2594 |
| 70 | 0.2592 |
| 100 | 0.2589 |

Flat. This is the most useful negative-shaped result in the paper: the one remaining hand-set constant **does not matter**, which is exactly what you want if the claim is "we removed the hand-tuned noise scale". The network learns to scale its own output regardless of the cap.

**Voxel resolution $\eta$:**

| $\eta$ (m) | CD | Params (M) | Time (s) |
|---|---|---|---|
| 0.5 | 0.214 | 10.0 | 0.07 |
| 0.4 | 0.210 | 11.6 | 0.09 |
| 0.3 | **0.206** | 11.8 | 0.10 |
| 0.2 | 0.208 | 11.9 | 0.14 |

**0.2 m is worse than 0.3 m.** Finer is not monotonically better — the text claims "finer resolutions consistently improve CD" but their own table contradicts it at the finest setting. Likely the grid gets emptier and the convolutions have less signal per cell. 0.3 m is the pick; 0.5 m is available if you want 0.07 s and can spend 4% quality.

**What is missing from the experiments.** There is no ablation on the dense BEV head (the self-attention step), no ablation on the number of deformable stages or sampling points $K$, and — most conspicuously — **no experiment at varying input resolution**, despite "handles different input resolutions by design" being a headline contribution. The claim is architectural, not measured.

## Worth Remembering

**The generalisable lesson.** For any initialize-and-refine pipeline, the initialization is a design decision that people treat as a default. Gaussian noise is the maximally uninformative choice and you pay for it in iterations. Jittering the input is the maximally conservative choice and you pay for it in coverage. Learning the initialization from the same features the refiner uses costs one MLP and removes a hyperparameter. That pattern transfers well beyond LiDAR.

**Why this is the same story as the [[FlashAttention- Fast and Memory-Efficient Exact Attention|FlashAttention]] story, in a different costume.** LiNeXt has 1.99M parameters and is slower than an 11.8M-parameter model. The bottleneck was never FLOPs — it was irregular memory access in FPS and $k$-NN. Replacing an "algorithmically clever, hardware-hostile" operator with a "dumber, hardware-friendly" one is a recurring way to win in practice. See also [[Flash Attention]].

**Chamfer Distance is a weak objective, and everyone knows it.** It is symmetric nearest-neighbour matching, so it rewards *covering* the ground truth and punishes strays, but it has no notion of surface, normal, or structure. The authors themselves observe that diffusion methods look visually plausible while scoring worse on CD, and that LiNeXt directly optimises CD and scores well but leaves regions under-completed. A method that wins on CD is winning on "points are near points", which is not the same as "geometry is right". Compare the [[Evaluating Generative Models|FID]] situation in image generation — the metric and the thing you want come apart.

**Zero-shot to KITTI-360 is less impressive than it sounds.** Both datasets are Velodyne HDL-64E scans of German streets recorded by the same lab. It is a different sequence, not a different sensor. The authors are explicit in the conclusion that generalising across sensor beam geometries is *future work* — which is a quiet admission that the "no manual recalibration for each new sensor" claim is argued architecturally, not demonstrated.

**Practical caveats if you wanted to use it:**
- 0.1 s is measured on an RTX 6000 Ada. That is a workstation card, not automotive-grade embedded silicon. Real-time-on-a-desktop ≠ real-time-in-a-car.
- The fixed $N = 180{,}000$ output and $\lfloor N/M \rfloor = 10$ expansion ratio are baked into the architecture. Change $M$ and the repeat factor changes with it.
- Timing excludes the refinement network. The $^\dagger$ numbers (CD 0.138) are a second forward pass on top of the 0.1 s.
- Deformable attention operates on **BEV-projected** features. Everything is squashed to top-down for the refinement. Vertical structure survives only through the depth-merged channels, which is a real information bottleneck for things like overpasses or multi-storey geometry.

**Open question worth chasing.** The adaptive initialization is supervised only indirectly, through the final Chamfer loss on $P$. There is no direct loss on $P_{\text{init}}$. So what stops the initializer from collapsing to "do nothing" and letting the refiner do all the work? The ablation shows it does not collapse, but nothing in the objective prevents it. An intermediate Chamfer term on $P_{\text{init}}$ seems like an obvious thing to try.

## Links

Related: [[Attention]] · [[Cross Attention]] · [[Flash Attention]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Denoising Diffusion Probabilistic Models]] · [[Diffusion Models]] · [[Flow Matching]] · [[Diffusion Sampling]] · [[Distillation]] · [[U-Net]] · [[Backpropagation]] · [[Adam- A Method for Stochastic Optimization]] · [[GPU processing]] · [[Evaluating Generative Models]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Multi-Head Attention]] · [[Attention Is All You Need]] · [[An Image is Worth 16x16 Words (ViT)]]

New topics worth writing: Chamfer Distance, Deformable attention / Deformable DETR, Bird's-eye-view representations, Farthest point sampling, Voxelization and sparse 3D convolution, Point cloud completion, Jensen–Shannon divergence, LiDAR perception for autonomous driving, Trilinear interpolation as a differentiable lookup, SemanticKITTI
