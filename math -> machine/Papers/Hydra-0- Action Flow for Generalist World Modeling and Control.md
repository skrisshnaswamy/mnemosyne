---
title: "Hydra-0: Action Flow for Generalist World Modeling and Control"
authors: ["Hongyu Li", "Bowen Wen", "Xinghao Zhu", "Yixuan Wang", "Yilun Du", "Yunzhu Li", "George Konidaris", "Stan Birchfield", "Soha Pouya", "Chenran Li", "Yan Chang"]
year: 2026
arxiv: "2608.18077"
url: https://arxiv.org/abs/2608.18077
priority: Good-To-Read
read_on: 2026-08-31
tags: [paper, vision]
---
## The Core Idea

A world model is a model that predicts what a video of the world will look like after the robot does something. The hard part has always been *how you tell it what the robot did*.

The usual answer is to feed it the robot's own command — joint angles, or a 6-DoF end-effector delta. That answer is quietly broken for anything general. A joint command only means something if you already know that robot's arm. The *same* end-effector command produces a different-looking arm on a Franka than on a humanoid. So a video model conditioned on native commands has to secretly learn "command → what pixels move" separately for every robot it ever sees. That kills cross-embodiment training, and it means human demonstration video (which has no joint angles at all) cannot be mixed in.

**Action flow** replaces the command with the thing the command *causes in the image*: a set of sparse 2D point trajectories on the visible surface of the robot, in pixel coordinates, with a visibility flag per point per frame.

$$\mathcal{F} = \{\boldsymbol\tau_n\}_{n=1}^N, \qquad \boldsymbol\tau_n = \{(\mathbf{x}_{n,t}, m_{n,t})\}_{t=0}^{H}, \quad \mathbf{x}_{n,t}=(u_{n,t},v_{n,t})$$

That is it. A robot arm, a two-finger gripper, and a human hand all become "these dots move along these paths." The video model never sees an action space.

> [!NOTE] Action flow ^action-flow
> Robot actions expressed as image-plane trajectories of points on the visible acting body, derived from an *executable* command by rolling that command through a physics simulator and projecting the resulting link poses through a calibrated camera. It is a rendering of the command, not a re-parameterisation of it.

The crucial word is **kinematically grounded**. Anyone can draw arrows on an image and ask a video model to follow them (that is ATI, Wan-Move, Motion Prompting). Those arrows may be physically impossible. Here the arrows are produced by actually executing a candidate motor command in Isaac Lab and projecting where the links end up. So every action flow corresponds to a command a real robot could run, and the constraints of the robot's kinematics are baked into the shape of the trajectories.

What this unlocks, in order of increasing interest:

1. **One model over seven heterogeneous datasets**, including human hands from a Vision Pro and handheld UMI grippers, because they all reduce to the same pixel-trajectory format.
2. **Open-loop policy evaluation.** Feed the recorded end-effector trajectory of a policy, generate the video, ask a judge if the task succeeded. Generated vs. real success rates correlate at $r = 0.96$.
3. **The interface runs backwards.** Give the model the desired motion of the *object* and no robot flow at all. To make the video coherent, the model must hallucinate a robot that would produce that object motion. A small head reads the internal DiT features and decodes executable joint commands. So you can transfer a human demonstration to a real robot with no expert robot demonstration of that task.

## The Methodology

**Setup.** Encoder $e_\phi$ maps the first RGB frame to a spatial latent $\mathbf{s}_0$. Dynamics model predicts $\hat{\mathbf{s}}_{1:H} = g_\theta(\mathbf{s}_0, \mathcal{F})$. Decoder $d_\psi$ produces pixels. $H$ = 81 frames at 16 fps ≈ 5 seconds, at 480×832.

### Two ways to build the flow

**Geometry-aware** (used at deployment, and for data with robot URDFs + calibration). Sample points on the robot surface visible in frame 0. Roll the candidate command $\mathbf{a}_{0:H-1}$ through the controller and physics in Isaac Lab to get configurations $\mathbf{q}_t$. Then project:

$$\mathbf{x}_{n,t} = \pi\!\left(\mathbf{K}\begin{bmatrix}\mathbf{I}_3 & \mathbf{0}\end{bmatrix}\mathbf{T}_{CW}\,\mathbf{T}_{\ell(n)}(\mathbf{q}_t)\,\bar{\mathbf{X}}_n\right)$$

$\mathbf{X}_n$ is the point in link $\ell(n)$'s frame, $\mathbf{T}_{\ell(n)}$ the link transform, $\mathbf{T}_{CW}$ extrinsics, $\mathbf{K}$ intrinsics, $\pi$ perspective divide. Visibility $m_{n,t}=1$ only if depth is positive, the point is inside the image, and it agrees with the rendered depth buffer within **1.2 cm** over a $3\times3$ pixel neighbourhood (the neighbourhood stops silhouette edges being wrongly rejected).

**Video-only** (most of the training corpus, which has no URDF or calibration). Run AllTracker on a $128\times128$ query grid to get dense tracks + visibility, then split the tracks into "embodiment" and "manipulated object" using SAM 3 masks. Same $\mathcal{F}$ format, no privileged metadata. Note the asymmetry: tracked-future flow is only ever a *training* condition; at deployment flow is always computed causally from a simulator rollout of the command.

### Training-time sampling: four modes

Each step, one of four pools populates the motion tensor. There is no learned mode token — the mode just decides which tracks get in.

| Mode | Tracks | Probability |
|---|---|---|
| **Embodiment** | arm/gripper/hand tracks — the main action condition | 0.40 |
| **Object** | manipulated-object tracks — task intent, and the inverse mode at test time | 0.40 |
| **All** | embodiment ∪ object ∪ unassigned scene tracks, 256–1024 of them | 0.15 |
| **None** | nothing — conditioning dropout, text+image only | 0.05 |

Embodiment and Object sample 1–128 tracks. If a pool is missing for a sample (DROID and EgoDex have no object masks), that mode is dropped and the rest renormalised.

### Turning trajectories into a conditioning tensor

Straight from ATI and Wan-Move: **carry the first frame's appearance features along the trajectories**. Pool each track into latent-frame windows (visible if visible anywhere in the window; position = mean of visible positions). Bilinearly sample a source feature $\mathbf{h}_n = \mathbf{s}_0(\widetilde{\mathbf{x}}_{n,0})$. Then splat it forward with a Gaussian:

$$\widetilde w_{n,k}(\widetilde{\mathbf p}) = \widetilde m_{n,0}\,\widetilde m_{n,k}\exp\!\left(-\beta\lVert \widetilde{\mathbf p}-\widetilde{\mathbf x}_{n,k}\rVert_2^2\right), \qquad \beta = 220$$

$$M_k(\widetilde{\mathbf p}) = \sum_{n \in \mathrm{TopK}_2(\widetilde{\mathbf p},k)} \widetilde w_{n,k}(\widetilde{\mathbf p})\,\mathbf h_n$$

Only the **top 2** trajectories per destination cell contribute, and there is **no softmax and no normalisation** — the raw weighted sum. Requiring both $\widetilde m_{n,0}$ and $\widetilde m_{n,k}$ means a track with no valid source *or* no valid destination contributes nothing.

Alongside it, a **presence gate**, which is just the same weight sum clipped:

$$g_k(\widetilde{\mathbf p}) = \mathrm{clip}_{[0,1]}\!\left(\textstyle\sum_{n\in\mathrm{TopK}_2} \widetilde w_{n,k}(\widetilde{\mathbf p})\right)$$

This tells the backbone "here is where I put motion-propagated appearance, everywhere else is untouched context." Without it, the model cannot tell a genuinely dark region from an unconditioned one. $C_{\mathrm{motion}} = (M, g)$.

### Loss

Standard flow matching on the backbone, with the motion condition supplied:

$$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{o}_{0:H},\mathcal F,t,\epsilon}\left[\lVert v_\theta(Z_t,t,c,C_{\mathrm{motion}}) - v_t^\star\rVert_2^2\right]$$

Everything in the pretrained backbone is **frozen** except the DiT patch embedding, plus **rank-64 [[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]]** on $q,k,v,o$ and the two FFN projections.

### Three backbones, three injection points

The point of the paper is that the interface is portable, so they wire it into three different video models:

- **Cosmos 2.5 (2B)** — 16-channel latent, so a 17-channel side input (16 propagated + 1 gate) goes through a *zero-initialised* patch projection and is **added** to the noisy-video tokens at every denoising call.
- **Wan2.2 I2V-A14B** — reuses the native I2V visual-condition tensor (16 VAE + 4 mask channels). Latent frame 0 keeps the native condition; later frames get raw propagated mass $+$ pristine feature $\times (1-g)$; gate broadcast into the 4 mask channels; concatenated with the noisy latent. Because Wan2.2 is a two-expert [[Sparsely-Gated Mixture-of-Experts Layer|MoE]] over noise level, motion conditioning goes only to the fine-tuned **high-noise** expert; at the transition the pristine condition is restored and the untouched low-noise expert finishes. Motion is decided early, detail is left alone.
- **Wan2.2 TI2V-5B** — no I2V condition tensor exists, so they widen the DiT input projection and concatenate a 49-channel side input (48 feature + 1 gate). Constant across steps, never noised.

### Making rollout fast

Bidirectional denoising over the whole 81-frame window assumes you already know the full action horizon and pays full cost every time. Two fixes, both from LongLive-2.0:

1. **Autoregressive conversion.** Chop the latent sequence into 7-latent chunks and factorise
$$p_\theta(\mathbf s_{1:H}\mid \mathbf s_0, C_{\mathrm{motion}}) = \prod_{j=1}^{J} p_\theta(\mathbf s_{\mathcal C_j}\mid \mathbf s_{\mathcal C_{<j}}, \mathbf s_0, C_{\mathrm{motion},\mathcal C_j})$$
Generated chunks replace clean history and are reused through a KV cache. The important detail: $C_{\mathrm{motion}}$ is computed **once in full-window coordinates** and sliced at *absolute* chunk offsets, so trajectories are not re-anchored at chunk boundaries. Trained with clean-context teacher forcing and a block-causal mask; no ODE init, no intermediate distillation step.
2. **DMD2 four-step distillation.** Student initialised from the causal teacher, generates each chunk in 4 steps, online fake-score critic, 5 critic updates per generator update, all three networks kept causal. CFG coefficient set to 0 on both score branches, since no unconditional branch was ever adapted.

### The inverse mode: world action model

Condition on object flow $\mathcal F^{\mathrm{obj}}$ only, withhold embodiment flow. The model must invent robot motion consistent with that object motion. Then a readout head, per latent frame: concatenate mean-pooled and single-query attention-pooled spatial tokens → LayerNorm → 2-layer GELU MLP, width 1024 → predicted action and predicted robot state. The state branch is auxiliary supervision only and is never fed back.

$$\mathcal{L}_{\mathrm{WAM}} = \mathcal{L}_{\mathrm{flow}} + \lambda_h\left(\mathcal{L}_{\mathrm{act}} + \mathcal{L}_{\mathrm{state}} + \lambda_v \mathcal{L}_{\mathrm{vel}}\right), \quad \lambda_h = \lambda_v = 0.1$$

$\mathcal{L}_{\mathrm{act}},\mathcal{L}_{\mathrm{state}}$ are masked [[Loss, Objectives, and Business Alignment#Huber & Pseudo-Huber: The "Best of Both Worlds"|Huber]] losses on per-dimension normalised targets; $\mathcal L_{\mathrm{vel}}$ is a masked L1 on first differences (a smoothness term). Post-training touches only rank-32 self-attention LoRA, the motion input projection, and the heads. At deployment the head reads token embeddings directly — **no pixel decoding needed**.

The training data does *not* have to be successful. Every rollout, success or failure, pairs achieved motion with the actions that produced it. That is the whole reason this needs no expert demonstrations.

### Data

7 sources, 178,187 episodes, 1.88M windows before filtering, **1.57M after**, 2,201.7 hours. DROID (single Franka, 313.7 h) is the main robot source; ABC-130k is the bulk (1,474.7 h bimanual); EgoDex is human hands; Deform360 is handheld UMI grippers. Heavily subset toward **deformable objects** (cloth, cable, rope, bag, paper) since that is where explicit-state simulation fails.

Three window-level filters, thresholds picked off motion-score histograms: drop windows whose 90th-percentile track path length is under **50 px at 480p** (static), drop windows whose *embodiment* tracks are under the same floor (frozen gripper), drop DROID episodes whose language annotation says nothing actionable.

Training: Wan2.2 I2V-A14B for **5 days, 40,000 steps, 32 H100s**.

## Ablation Studies and Experiments

### The controlled comparison that carries the paper

Same backbone (Cosmos 2.5 2B), same data mixture, same trajectories — only the conditioning representation changes: native relative 6D end-effector action vs. action flow. Averaged over five held-out sets (100 clips each):

| Model | PSNR ↑ | SSIM ↑ | Obj. EPE ↓ | Grip. EPE ↓ | FID ↓ | FVD ↓ | VLM ↑ |
|---|---|---|---|---|---|---|---|
| ATI (zero-shot) | 17.01 | 0.700 | 23.19 | 4.62 | 36.4 | 444.2 | 3.14 |
| Wan-Move (zero-shot) | 16.35 | 0.688 | 21.53 | 4.67 | 34.4 | 408.3 | 3.71 |
| **Cosmos 2.5, native 6D** | 15.62 | 0.668 | 13.23 | **34.28** | 39.1 | 405.8 | 3.88 |
| Ours (Cosmos 2.5 2B) | 18.41 | 0.725 | 6.27 | 13.80 | 32.4 | 277.4 | 3.83 |
| Ours (Wan2.2 5B) | 19.64 | 0.770 | 6.61 | 3.88 | 24.1 | 248.8 | 3.90 |
| Ours (Wan2.2 A14B) | 20.76 | 0.805 | 6.00 | 3.83 | 20.7 | 193.7 | 3.98 |
| **Ours (A14B, 4-step)** | **21.84** | **0.830** | **5.27** | **3.29** | **18.7** | **155.9** | **4.23** |

Headline numbers: $34.28 \to 3.29$ px gripper EPE is **90.4% lower**; $13.23 \to 5.27$ px object EPE is **60.2% lower**. The gripper EPE column is the tell — the native-action baseline is at 34 px, roughly "the arm goes somewhere plausible but not where you asked." Swapping in flow on the *same* backbone drops it to 13.8. That is the representation doing the work, not scale.

Within-dataset, swapping native 6D → action flow improved PSNR, SSIM, gripper EPE, FID and FVD on **all five** datasets and object EPE on all four where it is measurable.

### Things that are mixed or backwards

- **VLM judge scores are not clean wins.** On XVLA-Soft-Fold, the native-action Cosmos 2.5 scores 4.56 and action-flow Cosmos scores 4.42. On Deform360, native Cosmos 2.90 vs. flow-Cosmos 2.67. The authors' explanation is honest: the Gemma-4-31B judge rates physical plausibility, temporal consistency, object permanence and motion realism — none of which measure whether the arm went *where you told it*. A model can produce a beautiful, plausible video of the wrong motion.
- **The 4-step distilled student beats its 50-step teacher on nearly every metric.** PSNR 21.84 vs 20.76, FVD 155.9 vs 193.7. Distillation is supposed to trade quality for speed. It did not. Unexplained.
- **The small Wan2.2 5B beats the big A14B on object EPE for XVLA-Soft-Fold** (3.40 vs 3.42, a tie) and the tiny Cosmos 2.5 2B beats Wan 5B on Deform360 object EPE (8.47 vs 9.29). Object-motion accuracy is not monotone in backbone size.
- **Zero-shot ATI and Wan-Move are not bad.** ATI hits 4.62 avg gripper EPE with no robot training at all. Generic trajectory-conditioned video models already transfer. The gap is in *object* EPE (23.19 and 21.53 vs 5.27) — they move the drawn dots but do not get the consequences right.

### Data efficiency (IWS, 6 tasks, held out of mid-training)

Nested seeded subsets at 0/1/2/4/20/40/60/80/100% of task data, identical subsets for every method. Two variants: **PT** starts from raw Wan2.2, **MT** starts from the multi-embodiment mid-trained checkpoint.

- At **0%**, MT beats PT on LPIPS, object EPE and FVD across all six tasks. The from-scratch IWS model and the freshly-added Cosmos 2.5 action layers are far worse at 0% — untrained action interfaces do not transfer, unsurprisingly.
- At **100%**, MT has the best LPIPS and FVD on all six, best flow EPE on four. It loses Bimanual box to PT by 0.014 px and Bimanual rope to IWS by 0.15 px — i.e. ties.
- **The plateau is the real result:** from 20% to 100%, MT's numbers move by at most 3.4% (LPIPS), 6.7% (EPE), 6.8% (FVD). Mid-training buys you roughly a 5× reduction in task data. The authors flag that FVD comes from only 40 clips per point and is noisy, so they will not call it convergence.

### Speed (1× H100, bf16, batch 1, 81 frames, no guidance, VAE decode excluded)

| Stage | s/clip | FPS | Speedup |
|---|---|---|---|
| Bidirectional teacher (50 steps) | 20.92 | 3.87 | 1.0× |
| Autoregressive teacher (50 steps) | 12.48 | 6.49 | 1.68× |
| Few-step student (4 steps) | **1.31** | **61.98** | **16.0×** |

Note the AR conversion is 1.68× faster *despite going from 50 to 153 network evaluations*, because each one only touches a 7-latent chunk (including one recache per chunk).

### Policy evaluation (RoboLab)

5 policies ($\pi_0$, $\pi_{0.5}$, GR00T N1.7, Cosmos-3 Edge, Cosmos-3 Nano) × 6 tasks × 10 rollouts = 300 episodes. Open-loop replay: start from the true first frame, feed causal action flow from the *recorded* end-effector trajectory, never query the policy on generated frames.

**Pearson $r = 0.96$, Spearman $\rho = 0.93$, MAE = 5.7 percentage points.** Averaged across tasks, the generated success rates reproduce the correct ranking of all five policies.

### Inverse control

One task, qualitative: flexible-pipe bending. Object flow extracted from a held-out human demo, embodiment flow withheld, generated rollout decoded by the action head into real robot actions, execution succeeds. A proof of concept, presented as such.

## Worth Remembering

**What is conspicuously not ablated.** The four-mode sampling mixture $(0.05, 0.40, 0.40, 0.15)$, the Gaussian locality $\beta = 220$, $K=2$, the choice to skip softmax normalisation, and the presence gate itself all arrive as stated facts with no sweep. The presence gate in particular is argued for on intuition ("lets the backbone distinguish motion-conditioned locations from unmodified context") and never tested by removal. If you are reimplementing, these are the knobs you will have to find yourself.

**The calibration failure mode is graceful, up to a point.** Moderate calibration error perturbs the 2D condition without changing the target video — so training absorbs it. Large projection error breaks spatial correspondence outright. There is no quantification of where "moderate" ends.

**Occlusion is only half-handled.** The depth-buffer visibility check uses static scene geometry and solid-body proxies. Unknown or unmodelled objects occluding the robot are *not* removed, so those tracks will lie about visibility.

**Limitations the authors own:**
- Centimetre-scale grasp imprecision in the world action model. Their hypothesis is weak depth awareness. Fix would be conditioning on depth, tactile, force.
- Whether an object is actually *secured* in a generated rollout is often ambiguous. That is a real problem if you want to use these videos as a success oracle.
- Wrist-camera egomotion is a single qualitative DROID figure, nothing more.
- Everything is **open-loop**. The policy is never asked to act on a generated frame. Closed-loop is where compounding error lives, and it is untested.
- The world action model runs on the **50-step autoregressive teacher**, not the 62-FPS student. The speed result and the control result do not compose yet.

**Connections.** The forward mode is the [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)|Dreamer]] / [[Mastering Diverse Domains through World Models (DreamerV3)|DreamerV3]] idea — imagine the consequence of a candidate action — but in pixels rather than a compact latent, and with the action space swapped for a rendering. The refusal to reconstruct pixels is what [[Self-Supervised Learning from Images with I-JEPA|I-JEPA]]-style latent world models argue for; Hydra-0 goes the other way deliberately, because a human-inspectable video is what makes the policy-evaluation application possible. The trajectory-splatting mechanism is inherited wholesale from ATI and Wan-Move; the contribution is grounding the trajectories in a physics rollout rather than a user's mouse drag.

**Practical caveat.** The forward deployment path is not lightweight. To evaluate one candidate command you need a URDF, camera calibration, an Isaac Lab controller-and-physics rollout, and a depth render — *before* you touch the video model. This is a simulator wrapped around a video model, not a drop-in.

**Open question.** They frame policy evaluation as evidence that action flow is a good interface. But open-loop replay conditions on the *achieved* trajectory, which already contains most of the outcome. How much of $r=0.96$ is the world model understanding physics, versus the flow condition being a near-complete description of what happened? A control conditioning on the *commanded* rather than achieved trajectory would separate these.

## Links

Related: [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Self-Supervised Learning from Images with I-JEPA]] · [[Denoising Diffusion Probabilistic Models]] · [[Score-Based Generative Modeling through SDEs]] · [[Classifier-Free Diffusion Guidance]] · [[Distilling the Knowledge in a Neural Network]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Attention Is All You Need]] · [[Sparsely-Gated Mixture-of-Experts Layer]] · [[τ_0-VLA- a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation]] · [[EXIMO- VLM Guided Exploration of VLA Policies]] · [[GOAG- Generative and Object-Agnostic Grasp Planner for Dexterous Robotic Manipulation]] · [[Loss, Objectives, and Business Alignment]] · [[Mixed Precision Training]] · [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]]

New topics worth writing: Flow matching objectives, Rectified flow, Distribution Matching Distillation (DMD/DMD2), Diffusion Transformer (DiT), Vision-Language-Action models, Point tracking (TAP-Vid / CoTracker / AllTracker), Segment Anything and promptable segmentation, FVD and FID as video metrics, LPIPS, End-point error for optical flow, Isaac Lab and GPU-accelerated robot simulation, Camera intrinsics and extrinsics, Cross-embodiment robot learning, Neural simulators for policy evaluation, KV caching in autoregressive video diffusion, VLM-as-a-judge
