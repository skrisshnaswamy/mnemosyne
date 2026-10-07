---
title: "4DAnyone: Create Anyone in 4D from a Casual Monocular Video"
authors: ["Yudong Jin", "Tao Xie", "Qihang Zhang", "Zehong Shen", "Zhen Xu", "Yujun Shen", "Hujun Bao", "Xiaowei Zhou", "Yinghao Xu"]
year: 2026
arxiv: "2608.20335"
url: https://arxiv.org/abs/2608.20335
priority: Low-Priority
read_on: 2026-09-30
tags: [paper, transformers, diffusion, theory]
---
## The Core Idea

You film someone with your phone. One camera, mild handheld wobble, unknown lens and unknown camera path. 4DAnyone turns that into a **4D human** — a moving 3D model you can orbit around and render from any angle, at any moment in the clip.

The route is: *generate the missing cameras, then reconstruct*. First a video diffusion model invents 16–48 synchronised videos of the same performance from viewpoints that never existed. Then a standard 4D Gaussian Splatting (4DGS) reconstruction is fitted to those fake videos as if they were a real camera rig.

> [!NOTE] 4D Gaussian Splatting
> A scene stored as a cloud of fuzzy 3D blobs, each with a position, colour, opacity and a time window. Rendering is just sorting and blending the blobs, so it is real-time. Fitting it normally needs a calibrated rig of dozens of synchronised cameras. ^4dgs

The actual contribution is not "camera control in video generation" — that already existed. It is **consistency at reconstruction scale**. Prior camera-controlled video models produce one or two plausible new views. Ask for sixteen and they fall apart: the jacket changes colour between view 3 and view 11, the body drifts sideways, and the 4DGS fit turns to soup because the views disagree about geometry.

The paper names the reason cleanly, and this is the part worth keeping. It is an **attention-context budget problem**, not a modelling problem.

A [[Diffusion Transformer|DiT]] can only fit so many tokens in one forward pass. Sixteen videos of 121 frames each do not fit. So you must split the target views into groups, and that split creates two separate wounds:

1. **Reference side.** Each group should be able to look at every view generated so far, to copy appearance. But that reference context grows as $O(N)$ in the number of views. It blows the budget almost immediately, so people truncate it — and truncating is exactly what causes the jacket to change colour.
2. **Target side.** Views in different groups cannot attend to each other at all. Group A and group B are denoised in isolation, so they independently decide where the hips are. Global structure drifts between groups.

Two fixes, one for each wound:

- **Reference Context Packing (RCP)** — squash all reference views into a *fixed* number of tokens by giving different views different resolutions. $O(N) \to O(1)$.
- **Target Context Routing (TCR)** — during the noisy early [[Diffusion Sampling|sampling]] steps, reshuffle which views sit in which group at *every* step, so information leaks across groups over time. During the clean late steps, freeze the groups so neighbouring views can polish details together.

There is a third design choice that matters as much: the geometric signal is a **3D skeleton**, not a depth map. The principle is *accuracy over density*. Dense metric depth is what Gen3C and TrajectoryCrafter use, and it is unreliable on casual video — a wrong depth map is an actively harmful constraint that makes the generation diverge. A 3D skeleton is sparse but you can get it right from monocular video with off-the-shelf human mesh recovery.

## The Methodology

### The pipeline, end to end

1. Run GVHMR on the source video → a ground-aligned SMPL-X mesh sequence.
2. A sparse regressor turns each mesh into 70 3D keypoints in the Goliath vocabulary. Keep 40 of them (see below).
3. Render those 3D keypoints into a **skeleton video** for each of the $v$ target viewpoints.
4. A skeleton encoder turns each skeleton video into a residual added to the noisy latents.
5. The DiT denoises all target views jointly (in groups), conditioned on the source video, the skeletons, and the RCP reference context.
6. FreeTimeGS fits the 4DGS model to the generated videos.

Backbone is **Wan2.2-TI2V-5B**, a 5B-parameter DiT video diffusion model. Chosen because its VAE compresses hard spatially, which is what you want when you are stacking sixteen videos in one attention window.

### Depth-buffered skeleton rendering

A plain 2D stick figure is ambiguous: an arm in front of the torso and an arm behind the torso draw the same picture. The fix is to rasterise the 3D skeleton with a per-pixel **z-buffer**, so nearer limbs correctly cover farther ones. No extra input channels — the occlusion information lives in which line got drawn on top.

Each keypoint's camera-space depth drives the buffer, and only the *ordering* matters, so it is invariant to absolute depth scale and shift. That is convenient, because monocular depth is only ever correct up to scale.

### Keypoint selection

40 keypoints from a 308-keypoint vocabulary: 17 body, 6 foot, 10 palm-level hand (5 knuckles per hand), 7 auxiliary neck/shoulder/elbow. Deliberately **dropped**: all 238 facial keypoints and all 30 finger joints. The reasoning is that noisy fine keypoint detections produce artefacts, so faces and fingers are left to be copied from the source video instead.

### The skeleton encoder

$$\tilde{\mathbf{z}}^{t}_{i} = \mathbf{z}^{t}_{i} + g_{\phi}(\mathbf{S}_{i})$$

$\mathbf{S}_i$ is the depth-buffered skeleton video for target view $i$, $\mathbf{z}^t_i$ the noisy latent at step $t$. $g_\phi$ is 10 Conv3d layers with SiLU (channels $3 \to 16 \to 32 \to 64 \to 128 \to 256 \to d$), giving $32\times$ spatial and $4\times$ temporal downsampling, then a $1{\times}1{\times}1$ projection that is **zero-initialised** — so at step zero of training the residual is zero and the pretrained DiT behaves exactly as before. The standard [[Controlling Diffusion|ControlNet]]-style trick.

### Attention layout

Two attention modules, same architecture and *same initial weights*, differing only in how tokens are reshaped:

- **Video (temporal) attention** — tokens grouped so frames talk to frames.
- **Multiview attention** — tokens rearranged to $(f, v\cdot h \cdot w, d)$, so all views at the same timestep talk to each other.

Both are initialised from the base model's temporal [[Attention Is All You Need|self-attention]] layers. The bet is that whatever machinery already makes frames coherent in time is a good starting point for making views coherent in space. Cheap and it works.

### Reference Context Packing, concretely

The standard Wan2.2 patchify layer is a Conv3d with kernel and stride $(1,2,2)$. RCP adds two more:

| Layer | Kernel / stride | Tokens produced |
|---|---|---|
| $\mathcal{P}_1$ (standard) | $(1,2,2)$ | $1\times$ |
| $\mathcal{P}_2$ | $(1,4,4)$ | $\tfrac{1}{4}\times$ |
| $\mathcal{P}_4$ | $(1,8,8)$ | $\tfrac{1}{16}\times$ |

The fixed reference context is then

$$\mathcal{C}_{\text{R}}=\big[\mathcal{P}_{1}(\mathbf{V}_{\text{src}}),\ \{\mathcal{P}_{2}(\mathbf{V}_{a_{j}})\}_{j=1}^{3},\ \{\mathcal{P}_{4}(\mathbf{V}_{b_{j}})\}_{j=1}^{4}\big]$$

So: the source video at full token resolution, three earlier generated views at quarter cost, four more at one-sixteenth cost. Eight views for the token price of roughly $1 + 3/4 + 4/16 = 2$ views. The budget is fixed by construction, so adding more references later costs nothing extra — that is the $O(1)$.

The new patchify layers are initialised by **tiling the pretrained $(1,2,2)$ kernel spatially and dividing by the area ratio** (4 for $2\times$, 16 for $4\times$), which preserves activation variance. Borrowed from FramePack.

The insight underneath: cross-view appearance is massively redundant. A blurry view still tells you "the jacket is red and the collar is open"; you do not need it at full resolution.

> [!NOTE] Mixed-resolution context
> Rather than *choosing* which reference views to keep (CAT3D's anchor selection, which throws information away), keep all of them at varying fidelity. Nothing is discarded, only blurred. ^mixed-resolution-context

During training, $\mathbf{V}_{\text{src}}$, $\{\mathbf{V}_{a_j}\}$ and $\{\mathbf{V}_{b_j}\}$ are sampled at random from the sequence, and the context is dropped out with probability 0.1 each for "zero out the $\mathcal{P}_4$ block" and "zero out both blocks". That dropout is what makes progressive inference possible — the model has to cope with a half-empty reference context.

**Inference order:**
- Round 1: generate 4 reference videos conditioned only on $\mathcal{P}_1(\mathbf{V}_{\text{src}})$.
- Round 2: generate 4 more, now conditioned on the source plus three Round-1 views at $2\times$.
- Then: build the fixed $\mathcal{C}_{\text{R}}$ from all of these, and generate every remaining target view in four-view groups against it.

Which viewpoints go first is chosen by **farthest-point sampling** on angular distance — greedily pick the candidate whose nearest already-selected view is furthest away. Maximises coverage of the body early.

### Target Context Routing, concretely

Observation: in [[Diffusion Models|diffusion]], high-noise steps decide global structure and low-noise steps refine texture. So group drift is born early. Split inference at a switching timestep $t_s$:

- **$t > t_s$ (high noise):** at each step $n$, cyclically shift the azimuth-ordered view indices by the step index, then re-partition into four-view groups. View 3 shares a group with view 4 this step, with view 5 next step. Structure propagates across the whole ring over the sequence of steps, without ever exceeding the per-group memory budget.
- **$t \le t_s$ (low noise):** freeze the groups to be adjacent views, so neighbours polish details together and cross-view transitions stabilise.

Default $t_s/T = 0.2$ with 20 denoising steps, so 16 sliding steps then 4 fixed. Groups within a step can be run sequentially on one GPU or in parallel across several.

### Loss

$$\mathcal{L} = \mathcal{L}_{\text{latent}} + \lambda \mathcal{L}_{\text{LPIPS}}, \qquad \lambda = 0.25$$

$\mathcal{L}_{\text{latent}}$ is the standard [[Flow Matching|flow-matching]] MSE in latent space. The LPIPS term is a perceptual loss on *decoded* frames, added because the Wan2.2 VAE compresses so aggressively that latent-space loss alone leaves artefacts. Full-resolution LPIPS on 121 frames does not fit in memory, so they use **body-part-aware crops**: one $256{\times}256$ crop per clip, sampled as full body 0.2, face 0.2, left hand 0.1, right hand 0.1, uniform 0.4. Face and hand crops are box-centred; the crop is shared across all frames of the clip.

### Training curriculum

| Stage | Data added | Background | Skeleton | Time |
|---|---|---|---|---|
| 1 | DNA-Rendering only | masked out | body+hands+feet+fingers | ~0.5 d |
| 2 | + MVGameHuman, SynCamVideo | kept | body+hands+feet+fingers | ~1 d |
| 3 | + Pexels, TedTalk (monocular) | kept | fingers **removed** | ~1.5 d |

Stage 1 has a nice detail: with probability 0.2, the source clip and target clip are sampled from **different temporal windows** of the same sequence. This forcibly decouples pose from appearance — the model cannot just copy pixels, it must follow the target skeleton while taking appearance from elsewhere.

Stage 2 deliberately *stops* masking backgrounds, against prior practice (Diffuman4D). Two reasons given: mask boundaries create edge noise and cross-view inconsistency, and with no background the model never learns lighting and shadow.

Stage 3 drops finger keypoints because monocular finger detection is unreliable. Hands come from the reference instead.

704×1280 resolution, learning rate $1\times10^{-5}$, 128 H20-3E GPUs (they note it converges fine on 32+; 128 is just for speed).

### Data

| Dataset | Videos | Cameras | Actors | Type |
|---|---|---|---|---|
| MVGameHuman (theirs) | 38k | 24 | 318 | multi-view |
| SynCamVideo | 34k | 10 | 66 | multi-view |
| DNA-Rendering | 51k | 48 | 548 | multi-view |
| TedTalk | 42k | 1 | 413 | monocular |
| Pexels | 20k | 1 | 1,411 | monocular |

MVGameHuman is rendered in an in-house game engine at 2560×1440. 2D keypoints come from Sapiens2-1B; for multi-view data 3D keypoints come from cross-view triangulation, for monocular data from sampling Sapiens2's pointmap prediction, with large-camera-motion sequences filtered out.

## Ablation Studies and Experiments

### Evaluation protocol

This is well designed and worth copying. Three separate measurements, because "the generated video looks nice" and "the generated videos agree with each other" are different things:

1. **4DGS reconstruction** — fit 4DGS to all 16 generated views, render, compare to ground truth.
2. **Generated video consistency** — hold out 4 evenly spaced views, fit 4DGS to the other 12, then compare the 4DGS render at the held-out views to the *generated* video there. This measures whether the generated views are mutually 3D-consistent, independent of whether they are correct.
3. **Generated video reconstruction** — compare the 16 generated videos directly to ground truth.

10 held-out DNA-Rendering scenes, 3 DyMVHumans scenes (out of distribution for everyone), 16 cameras, 98 frames.

### Headline numbers

DNA-Rendering / DyMVHumans, PSNR↑:

| Method | Consistency | 4DGS recon | Video recon |
|---|---|---|---|
| TrajectoryCrafter (depth-warped, zero-shot) | 13.56 / 15.19 | 14.81 / 15.11 | 13.68 / 14.11 |
| MV-Performer (geometry-conditioned, zero-shot) | 21.25 / 19.98 | 20.38 / 18.69 | 19.33 / 14.36 |
| ReCamMaster† (implicit camera, *fine-tuned with RCP+TCR*) | 21.47 / 21.94 | 20.55 / 19.86 | 20.74 / 19.18 |
| **4DAnyone** | **24.33 / 24.48** | **24.15 / 23.28** | **23.69 / 21.03** |

LPIPS on DNA-Rendering 4DGS: 0.159 vs 0.214 (ReCamMaster†) and 0.191 (MV-Performer).

The ReCamMaster† comparison is the honest one: same data, same training protocol, *same RCP module and TCR strategy*. So the ~3.6 dB gap there is attributable to **explicit skeleton conditioning vs implicit camera-parameter conditioning**, not to the two named contributions. ReCamMaster produces plausible-looking videos but its camera control is imprecise enough that 4DGS renders come out noisy.

TrajectoryCrafter's collapse to 13–15 dB is instructive: accumulated depth error, and it essentially fails outright on front-to-back viewpoint changes. Exactly the failure mode the "accuracy over density" argument predicts.

### Ablations (8 DNA-Rendering scenes, consistency setting)

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| w/o TCR & RCP | 21.09 | 0.766 | 0.216 |
| w/o RCP | 22.03 | 0.780 | 0.203 |
| w/o TCR | 22.21 | 0.788 | 0.196 |
| Full (Random routing) | 22.20 | 0.788 | 0.197 |
| Full (Strided routing) | 22.06 | 0.786 | 0.198 |
| **Full (Sliding routing)** | **22.63** | **0.796** | **0.191** |

Read this carefully, because it says something the abstract does not.

**Each component alone is worth about 0.4–0.6 dB.** RCP adds 0.42 over "w/o RCP"→full... actually: removing RCP costs 0.60 dB, removing TCR costs 0.42 dB. Removing both costs 1.54 dB — more than the sum, so they are genuinely complementary, but the absolute magnitudes are modest next to the 3.6 dB gap to ReCamMaster†.

**The negative results are the interesting half.** With everything else on:

- **Random regrouping gives nothing.** 22.20 vs 22.21 for fixed grouping. Zero gain.
- **Strided regrouping actively hurts.** 22.06, worse than not routing at all.
- Only **sliding** (a cumulative one-position circular shift of the azimuth-ordered views) helps.

So it is not "mixing groups helps". It is specifically *mixing while preserving local view adjacency*. A group of four views that are near each other in angle can still cooperate on detail; a group of four scattered views cannot, and forcing them together destroys more than the information-sharing gains.

### The switching-time sweep

Varying only $t_s/T$, with 20 steps:

| $t_s/T$ | sliding steps | PSNR↑ | LPIPS↓ |
|---|---|---|---|
| 1.00 | 0 | 22.208 | 0.1964 |
| 0.50 | 10 | 22.458 | 0.1933 |
| 0.25 | 15 | 22.609 | 0.1912 |
| **0.20** | 16 | **22.629** | 0.1906 |
| 0.10 | 18 | 22.622 | 0.1905 |
| 0.00 | 20 | 22.641 | 0.1903 |

Monotone improvement up to $t_s/T \approx 0.2$, then flat. The authors are candid that the wiggles past that point are "comparable to variation from 4DGS optimization" — i.e. noise. Note $t_s/T = 0$ (slide the whole way) is statistically indistinguishable from 0.2. So the two-phase story is *half* confirmed: sliding at high noise clearly helps, but the claimed benefit of freezing groups at low noise does not show up in the numbers. That part of the design is theory, not measurement.

### 3D-aware skeleton conditioning

Only shown qualitatively (Fig. 5). Depth buffering resolves front-back ambiguity for overlapping limbs. No PSNR number given — the one ablation that is probably doing the most work is the one without a number attached.

### Failure cases

- **Loose garments.** A skeleton says nothing about a large flowing skirt. Different views invent different fabric, and the 4DGS fit degrades.
- **Wrong pose estimate propagates faithfully.** A dancer standing *en pointe* is read by HMR as flat feet, and every one of the 16 generated views dutifully renders flat feet. The generation is perfectly consistent — consistently wrong.

The authors note an upside of the skeleton route: under occlusion or motion blur, HMR usually still returns a *complete and plausible* skeleton, so you get a slightly shifted but coherent human rather than a shattered one. Failure is graceful.

## Worth Remembering

**The transferable idea has nothing to do with humans.** It is a recipe for "I need $N$ mutually consistent outputs and only $k < N$ fit in one attention window":

- Compress the accumulated context at *mixed* resolution rather than selecting a subset. Nothing is thrown away, only blurred, and the token budget stays flat.
- Rotate group membership across denoising steps to route information between groups you cannot afford to co-attend. But preserve locality when you rotate.

Both are cheap, architecture-agnostic, and inference-time-ish (RCP needs training; TCR is pure inference scheduling and costs nothing).

**"Accuracy over density" is the sharpest line in the paper.** A sparse-but-correct geometric constraint beats a dense-but-wrong one, because a wrong constraint is not merely uninformative — it actively fights the generative prior and makes multiview generation diverge. TrajectoryCrafter at 13.6 dB is the evidence.

**The evaluation split (consistency vs reconstruction) is the reusable methodological bit.** "Generated video consistency" — fit the reconstruction on a subset of generated views and test it against generated views you held out — measures mutual agreement without needing ground truth for the generation itself. That is a general trick for any generate-then-reconstruct pipeline, and it separates two failure modes that PSNR-against-ground-truth conflates.

**Honest attribution.** The headline 3.6 dB over the controlled baseline comes mostly from skeleton conditioning plus the training mixture, not from RCP and TCR, which are worth ~1.5 dB together. The paper's title contributions and its largest effect are not the same thing. Anyone reproducing this should get the conditioning right first.

**Practical timings** (worth knowing before you plan a demo):

| Stage | Time | Hardware |
|---|---|---|
| HMR + skeleton rendering | ~2 min | 1× RTX 4090 |
| 4 target videos, 121 frames, 20 steps | ~7 min | 1× H20 |
| FreeTimeGS 4DGS fit (16 cams, 121 frames, 50k iters) | ~30 min | 1× RTX 4090 |

They note 10 denoising steps works nearly as well as 20, because the conditioning is strong. Inference configurations scale from 16 cameras in one elevation ring (4 GPUs) to 48 cameras in three rings (8 GPUs) — the denser rings are needed for complex clothing or extreme motion.

**Open questions worth chasing:**
- The low-noise freeze phase has no measured benefit. Is it doing anything, or is the whole win from high-noise sliding?
- Why does strided routing *hurt*? "Locality matters" is a description, not a mechanism. Something about neighbouring views sharing high-frequency structure that scattered views cannot.
- The skeleton is the whole geometric prior, so anything that leaves the body — hair, skirts, held objects — is unconstrained. Adding a coarse mesh or a garment prior is the obvious next move, at the cost of reintroducing the "wrong dense signal" problem.
- Chaining Wan-Animate in front of this gives single-image → 4D animatable avatar, which they demo but do not evaluate. Error compounds through two generative stages.

**Ethics, as the authors raise it:** this is a photoreal human video synthesiser conditioned on a single clip. Deepfake, identity and copyright risk are direct, not hypothetical.

## Links

Related: [[Diffusion Models]] · [[Diffusion Transformer]] · [[Video Diffusion]] · [[Flow Matching]] · [[Conditional Generation]] · [[Controlling Diffusion]] · [[Diffusion Sampling]] · [[Attention]] · [[Attention Is All You Need]] · [[Cross Attention]] · [[Sparse Attention]] · [[Latent Diffusion]] · [[Denoising Objective]] · [[Evaluating Generative Models]] · [[Context Window]] · [[U-Net]]

New topics worth writing: Gaussian Splatting (3DGS and 4DGS), FreeTimeGS, human mesh recovery and SMPL-X, LPIPS as a training loss, FramePack and frame context packing, novel view synthesis evaluation protocols, patchify layers and token budgets in video DiTs
Tags: `diffusion` `vision` `evaluation`
