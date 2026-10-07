---
title: "InternW0: A Foundational Physical World Model for Efficient Real-World Interactions"
authors: ["Cai et al."]
year: 2026
arxiv: "2609.27656"
url: https://arxiv.org/abs/2609.27656
priority: Must-Read
read_on: 2026-09-24
tags: [paper, transformers, diffusion, vision]
---
## The Core Idea

A "world action model" (WAM) is a robot policy that does two jobs at once: it predicts what the camera will see next, and it outputs the motor commands. The prediction is meant to help the commands — if you can imagine the next two seconds of video, you have a plan to act against.

The problem is speed. Video prediction is a big diffusion-style model. Running it costs tens or hundreds of milliseconds. But a robot arm wants a new command every ~60 ms, especially when it is touching something. So you get a bad choice:

1. Re-predict the video on every control step → far too slow.
2. Predict once, then act against that frozen prediction → the robot is steering by a picture of a world that has already moved on.

InternW0's trick is a third option. Keep the video prediction as a **cached key/value tensor** (the same K and V matrices from [[Attention]] that a [[KV Cache]] stores), and on every action step, *edit that cache* using the fresh camera frame before the action network reads it. The edit is cheap — a small attention pass plus a projection. The heavy video model runs on its own slow clock in the background.

> [!NOTE] Chunk K/V editor ^chunk-kv-editor
> A small learned module that takes the cached K/V of a video prediction and adds an observation-dependent residual $(\Delta K, \Delta V)$. Same underlying plan, a different *view* of it for each action chunk. Zero-initialised, so it starts as the identity.

What this unlocks: a 60.73 ms critical path (16.47 Hz policy updates) on an RTX 5090D, $3.13\times$ faster than Fast-WAM, while keeping the accuracy of a big video predictor. And because the action side is decoupled, they could bolt force and tactile sensing onto it later without retraining the video backbone.

Second, smaller idea worth taking: they added 275 hours of **egocentric human video from real wet labs** (EgoLab) with no action labels at all. In a limited-data ablation, that video-only data lifted out-of-distribution success by 8.24 percentage points — more than it lifted in-distribution success.

## The Methodology

### Architecture: two experts, separate parameters

A **mixture-of-transformers** (MoT) backbone — not [[Mixture of Experts|MoE]] routing, but a fixed split: each joint layer has two blocks with *separate weights* and separate token streams, one per modality.

- **Video expert (Video-DiT).** Large. Reads video latents + language. Does *not* see actions or which robot it is. This keeps physical prediction shared across everything.
- **Action expert (Action-DiT).** Light. Reads the edited video context, the current frame, proprioception, language, and robot-specific interfaces.

Frozen encoders do the perception: **Wan VAE** for video latents, **DINOv3** for the current chunk's RGB frame, precomputed text embeddings for language. Only the projections and the two experts train.

### The editing mechanism, step by step

For action chunk $n$ at layer $\ell$, with cached video keys/values $(K_\ell, V_\ell)$:

1. Encode the chunk's current RGB frame with DINOv3, compress it to a few **routing queries** $q_n$.
2. **Retrieve** — let the observation query the plan:
$$R_{\ell,n} = \mathrm{Attn}(W^q_\ell q_n,\; K_\ell,\; V_\ell)$$
3. **Route back** — put the retrieved information back onto the original video-token positions, using the keys themselves as queries:
$$D_{\ell,n} = \mathrm{Attn}(K_\ell,\; R_{\ell,n},\; R_{\ell,n})$$
4. Project $D_{\ell,n}$ to residuals and gate them:
$$\widetilde K_{\ell,n} = K_\ell + \sigma(\gamma_\ell)\,\Delta K_{\ell,n}, \qquad \widetilde V_{\ell,n} = V_\ell + \sigma(\gamma_\ell)\,\Delta V_{\ell,n}$$

$\gamma_\ell$ is a learned per-layer gate; the final projection is zero-initialised. So at step zero the editor does nothing, and it learns corrections gradually. Note step 3 is the unusual part — an attention where the *keys* act as queries, which is how the retrieved summary gets smeared back over all the video positions.

$K_\ell, V_\ell$ themselves never change. Each chunk builds its own edited copy.

### Objective: [[Flow Matching]] on both streams

Same loss shape for video latents and action chunks. Given clean target $x_0$, noise $\epsilon$, flow time $\tau \in [0,1]$:

$$x_\tau = (1-\tau)x_0 + \tau\epsilon, \qquad v^* = \epsilon - x_0$$

The network regresses $v^*$. Flow times are drawn from a shifted schedule — one per video example, independent ones per action chunk.

Pretraining loss:
$$\mathcal{L}_{\mathrm{pre}} = \mathbb{E}_{\xi \sim \mathcal{D}_{\mathrm{mix}}}\big[\lambda_v \mathcal{L}_{\mathrm{video}}(\xi) + \lambda_a \mathbb{I}_{\mathrm{act}}(\xi)\, \mathcal{L}_{\mathrm{joint}}(\xi)\big]$$

$\mathbb{I}_{\mathrm{act}}$ is 1 when action labels exist. Egocentric human video sets it to 0 and contributes video supervision only. Both terms are masked squared velocity errors — padding, the clean first frame, and missing action dimensions are excluded.

Video attention uses a **first-frame-causal mask**: the observed first latent frame cannot look forward; future frames can look at it and at each other bidirectionally. The first frame is a clean temporal anchor and is not in the loss.

### IDM conditioning — the detail that makes the coupling train

At training time there are **two video streams** sharing all Video-DiT weights:

- The **primary** stream gets the flow-matching loss.
- The **condition** stream uses an *independently sampled* noise and flow time. With probability 0.5 it is kept clean; otherwise perturbed. It gets **no reconstruction loss**. Its layerwise K/V is what the editor and action expert consume.

Crucially the condition K/V is **not detached**. Gradients from the action loss flow back through the editor into the video expert. So the predictive representation is shaped by *usefulness for control*, not just by pixel reconstruction. At inference, the condition stream is replaced by an actually generated video, cached, and served through the same interface.

### Cross-embodiment plumbing

- **Unified 37-dim action/state vector.** 17 per arm (7 joints, 3 EE translation, 6 rotation-6D, 1 gripper) + 3 for a mobile base. Missing channels zero-filled and masked out of the loss.
- **Soft prompts** per training domain (25 domains), prepended to every action chunk — borrowed from X-VLA. Plus per-domain input/output projections. Every GPU batch is domain-homogeneous, because the interfaces are per-domain.
- Joint actions are absolute next-waypoint targets; EE actions are deltas from the current step.

### Contact-aware post-training

For contact-rich tasks, the action target grows:
$$\mathbf{y}_i = [\mathbf{a}_i^\top,\ \mathbf{f}_i^\top,\ \mathbf{h}_i^\top]^\top$$
action, force/torque, tactile. Force history *before* a chunk is conditioning; future force channels are noised and denoised jointly with actions. Actions get executed; the force channels are predicted feedback. The video backbone is untouched.

### Data

7,233.5 hours, 811,969 episodes, 25 domains:

| Dataset | Type | Hours |
|---|---|---|
| InternData-A1 | sim | 3,494.3 |
| AgibotWorld | real | 2,620.0 |
| RoboCOIN | real | 438.1 |
| Galaxea | real | 314.9 |
| **EgoLab** | human ego video | 275.4 |
| MolmoAct | real | 70.3 |
| RoboDojo | sim | 20.5 |

Domain mixing uses square-root weights, $p_d = \sqrt{N_d} / \sum_j \sqrt{N_j}$, where $N_d$ counts effective anchors. Same family as RDT-1B (sqrt) and $\pi_0$ ($m^{0.43}$) — big datasets get more weight without swamping everything.

**EgoLab**: 275 h of head-mounted 2.5K/50fps video of humans doing wet-lab work (pipetting, weighing, titration, grinding, stirring). Hand and camera motion reconstructed at 50 fps with HaWoR; a VLM does open-vocabulary subtask splitting from 0.5 fps frames in 10 s clips.

**Sampling**: everything resampled to a 15 fps model clock. Each robot sample = 13 multi-view frames (stitched into one $384\times320$ canvas) + 64 action steps split into four 16-step chunks. Frames are 8 control steps apart, so 13 frames span 96 control steps. No history images.

Cleaning is unusually explicit and worth copying: drop static episodes; drop intervals with head/waist motion (and any window touching them); detect gripper open-close-open pulses that *subsampling would erase* and invalidate those chunks per-phase; validity-mask missing channels so a genuine zero stays supervised.

### Training stack

Ray for the execution plane. FSDP2, sharded per joint layer and per expert stage, with narrow per-domain interfaces replicated. FP32 master weights and optimiser state, BF16 compute, FP32 gradient reduction. Each joint layer compiled as a fixed-shape full graph *before* activation-checkpoint wrappers. Metadata-only shuffling with on-demand payload materialisation, plus node-shared video decode cache. Sample-level randomness seeded from sample identity, so resumption reproduces the exact stream.

## Ablation Studies and Experiments

### LIBERO (40 tasks, 50 rollouts each)

| Method | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| $\pi_0$ | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| $\pi_{0.5}$ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| Fast-WAM | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| LingBot-VA | 98.5 | 99.6 | 97.2 | **98.5** | 98.5 |
| **InternW0** | **99.4** | 99.4 | **98.6** | 97.0 | **98.6** |

Saturated benchmark. 98.6 vs 98.5 is noise. Don't read much into this.

### RoboTwin 2.0 Full (50 bimanual tasks)

| Method | Clean | Randomized | Avg |
|---|---|---|---|
| $\pi_{0.5}$ | 82.74 | 76.76 | 79.75 |
| Fast-WAM | 91.88 | 91.78 | 91.83 |
| AHA-WAM | 93.40 | 92.20 | 92.80 |
| OpenWAM-$\alpha$ | 93.74 | 93.46 | 93.60 |
| ABot-M0.5 | **94.00** | **94.20** | **94.10** |
| InternW0 | 93.20 | 93.04 | 93.12 |

**InternW0 loses here.** ABot-M0.5 is ~1 point ahead. The authors say so plainly. The clean/randomized gap of 0.16 pts is the notable bit — randomization was in training, so it should be small.

### RoboTwin 2.0 Clean2Random — the real result

Fine-tune on clean scenes only; evaluate on clean *and* on unseen randomized scenes.

| Method | Clean | Randomized | Avg |
|---|---|---|---|
| Fast-WAM | 77.8 | **1.9** | 39.9 |
| X-VLA | 68.0 | 20.9 | 44.5 |
| Spatial Forcing | 77.2 | 26.7 | 52.0 |
| $\pi_{0.5}$ | 70.7 | 46.0 | 58.4 |
| 4D-WAM | 81.5 | 41.8 | 61.6 |
| GigaBrain-0.7 | 66.8 | 67.9 | 67.3 |
| **InternW0** | **83.20** | **68.0** | **75.60** |

+8.30 pts average over the best baseline. Look at Fast-WAM: 77.8 clean, **1.9** randomized. Total collapse under distribution shift, from a method that scores 91.8 when randomization is in its training set. That contrast is the strongest evidence in the paper that pretraining scale, not architecture, is carrying generalisation. The 15.2-pt clean→random gap for InternW0 means the problem is far from solved.

### The EgoLab ablation

Restricted pretraining: ~50 h robot data, optionally +~50 h EgoLab video. Then fine-tune on clean, evaluate both.

| Pretraining | Clean | Randomized |
|---|---|---|
| Robot only | 72.13 | 13.73 |
| Robot + EgoLab | 76.40 | 21.97 |
| Δ | +4.27 | **+8.24** |

The gain is roughly twice as large out-of-distribution. Action-free human video buys generalisation, not in-domain skill. This is the cleanest controlled experiment in the report.

Caveat: this is a 50 h regime. The full model uses 7,200 h. No evidence the effect survives at scale.

### Real robots — 5 tasks, 15 trials each

| Method | Sandwich | Industrial Parts | Sort Tubes | MoF (progress) | Pipetting (progress) |
|---|---|---|---|---|---|
| $\pi_{0.5}$ | 73.3 | 53.1 | 86.7 | 50.2 | 46.7 |
| Fast-WAM | 40.0 | 50.6 | 66.7 | 10.7 | 18.7 |
| InternW0 | 73.3 | **82.7** | **88.9** | **68.4** | **65.3** |

Three metrics, matched to task shape: episode success (Sandwich), object-level recall (Parts, Tubes), progress rate = fraction of subtasks done correctly *and in order* (MoF, Pipetting).

**Where the gap is, and where it isn't:**

- Sandwich: tie at 73.3. Simple pick-and-stack is already in pretraining. No headroom to show.
- Industrial Parts: 53.1 → 82.7. $\pi_{0.5}$'s failures are *recognition* errors — right grasp, wrong box. Credited to the visual pretraining.
- Sort Tubes: near-tie. Fast-WAM (66.7) closes the gripper *above* the thin tubes — from-scratch visual features can't localise the grasp point.
- MoF 15-stage synthesis: $\pi_{0.5}$ falls off a cliff from subtask 8 onward (46.7 → 33.3 → 6.7) on small transparent glassware. InternW0 holds 66.7 through subtask 12, then drops to 26.7 at the stirrer buttons, which need millimetre precision. Fast-WAM never gets past subtask 3 — it confuses the visually near-identical stages and re-runs or skips steps.
- Pipetting (20-DoF hand + 7-DoF arm, hybrid force-position): InternW0 wins 4/5 subtasks, 65.3 vs 46.7. $\pi_{0.5}$ actually **beats it on subtask 1** (93.3 vs 86.7, pickup/reorientation) — the non-contact one. The gains are concentrated in tip attachment, aspiration, dispensing, ejection: exactly the force-sensitive steps.

Qualitative failure modes (Fig. 6) are the most informative page. On tip attachment, $\pi_{0.5}$ keeps pushing down until the pipette slips out of its grip; Fast-WAM bends the pipette body. InternW0 yields compliantly to the contact force and re-aligns. That is what the force channels buy.

### Efficiency

60.73 ms critical path → 16.47 Hz, $3.13\times$ over Fast-WAM, RTX 5090D, same precision. Critical path = DINOv3 encode + K/V edit + Action-DiT denoise. Video generation is excluded because it is scheduled asynchronously. Extra wins from graph compilation, hoisting denoise-invariant computation out of the sampling loop, and removing redundant tensor transfers — none of which change the maths.

## Worth Remembering

**The measurement choice is doing work.** "60.73 ms" excludes Video-DiT inference by construction. That is legitimate *if* the plan is genuinely still useful when it is a few hundred milliseconds old — and the K/V editor is the bet that it is. But there is no ablation showing how performance degrades as plan staleness grows, and no reported Video-DiT refresh rate. That is the number I would want before deploying this.

**No ablation of the K/V editor itself.** The central architectural contribution is never removed and re-measured. Given that RoboTwin-Full has InternW0 *behind* ABot-M0.5, and the LIBERO numbers are saturated, the evidence for the editor is mostly the efficiency figure plus inheritance from AHA-WAM.

**The IDM non-detach is the subtle bit.** Letting action gradients flow into the video expert means the video model is partly a control-representation learner, not a pixel predictor. Related in spirit to [[JEPA#predict-in-representation-space|predicting in representation space]] and to TD-MPC2's argument that world models should be control-centric, not reconstruction-centric.

**Fast-WAM's 1.9% on Clean2Random is the number to remember.** Same architecture family, no large-scale pretraining, total collapse under visual shift. It is a stark reminder that WAM benchmark rankings under in-distribution randomization tell you almost nothing about robustness. Compare [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|the RecSys baseline problem]] — a strong in-domain number hiding a brittle method.

**Progress rate is a nicer metric than success rate for long-horizon tasks**, and the per-subtask table is nicer still. MoF's breakdown localises failure to specific manipulations (transparent glassware, button pressing). An episode-level success rate would have reported "6.7% vs 26.7%" and told you nothing about why.

**Admitted limitations.** Dexterous-hand data was *excluded* from pretraining, yet pipetting is evaluated with a 20-DoF hand — so that task runs on post-training only. EgoLab is used video-only; the 3D hand reconstructions are just for filtering. Head and waist DoF are outside the supervision space entirely. 15.2-pt clean→random gap. Target size for future models is ~4B then <1B params, which implies InternW0 is larger and not embedded-deployable.

**Practical caveats if you wanted to use this.** Domain-homogeneous batching is forced by the per-domain interfaces — that constrains your data loader design and probably increases gradient variance across steps. The 37-dim unified action space with validity masks is a clean pattern worth stealing for any multi-embodiment setup. The gripper-pulse filter (detecting open-close-open round trips that subsampling would silently erase, per phase) is the kind of data bug that would quietly poison training and never show up in a loss curve.

**Open question.** Does the K/V editing actually encode *physics* corrections, or is it functioning as a learned attention re-weighting that happens to help? An intervention study — feed a deliberately wrong cached plan and see whether the editor recovers — would settle it.

## Links

Related: [[KV Cache]] · [[Attention]] · [[Flow Matching]] · [[Mixture of Experts]] · [[Cross Attention]] · [[Diffusion Policy]] · [[Imitation Learning]] · [[JEPA]] · [[GigaBrain-0.7- Scaling Embodied Foundation Models to Emergent Capabilities with a Three-System Architecture]] · [[Think Like a World Model, Act Like a VLA- Distilling World-Model Representations into Compact Robot Policies]] · [[Latent Action as Intention Enables Efficient Future Imagination for World Action Models]] · [[Modality-Autoregressive World-Action Models]] · [[Hydra-0- Action Flow for Generalist World Modeling and Control]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[POMDP]] · [[Distributed Training]] · [[Mixed Precision Training]] · [[Video Diffusion]] · [[Diffusion Transformer]] · [[Cost and Latency]]

New topics worth writing: Mixture-of-Transformers (modality-specific parameterisation), asynchronous planner–executor decoupling in robot policies, square-root domain mixing weights for heterogeneous corpora, hybrid position–force control, rotation-6D action representation, action-free video pretraining for control, progress rate as a long-horizon evaluation metric, FSDP2 composable sharding
