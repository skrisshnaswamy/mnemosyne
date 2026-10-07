---
title: "Grounded Action Model: 3D Grounding as a Foundation for Robotics"
authors: ["Zhang et al."]
year: 2026
arxiv: "2609.23863"
url: https://arxiv.org/abs/2609.23863
priority: Good-To-Read
read_on: 2026-09-27
tags: [paper, transformers, rl, vision]
---
## The Core Idea

Robot policies need to know two things before they can move: **which object matters**, and **where exactly it is in metres**. Today's robot foundation models get neither for free.

A vision-language-action model (VLA) starts from a vision-language backbone. That backbone was trained to produce text. Text prediction rewards saying "the sponge is next to the bowl" — it never forces the model to commit to the sponge's 3D coordinates. A world-action model (WAM) starts from a video generator. Video prediction rewards getting pixels right, including background pixels that have nothing to do with the task.

So in both cases the *metric grounding* — object identity plus object geometry in real-world units — has to be learned from scratch, out of a few hundred robot demonstrations. Demonstrations are collected in a handful of layouts. What the policy learns instead is a shortcut: a fixed visual pattern → a fixed arm trajectory. Move the target, swap the background, or point at a different object, and the shortcut is wrong.

The fix here is to change the pretrained backbone. Not language, not video — **a promptable 3D object detector**.

> [!NOTE] Grounded Action Model (GAM) ^grounded-action-model
> A manipulation policy whose frozen backbone is a pretrained 3D grounding model. The backbone turns a prompt (words, a click, or a box) into per-object 2D boxes, metric 3D boxes and a depth map. The policy sees *only* the grounded objects plus the robot arm, and predicts joint targets from that.

Two consequences follow, and both are the point of the paper.

**First: the observation is filtered by task, not by camera.** Everything outside the selected objects and the arm is zeroed out. A randomised background cannot confuse the policy because the policy never receives it.

**Second: task specification becomes one interface with three doorways.** Language, a 2D point, or a 2D box all resolve into the same object-centric representation. So a human can click the target. Or a vision-language-model planner can click it — which means a high-level planner with memory can drive a low-level controller that itself has no memory and no semantics.

The headline evidence: on RoboTwin 2.0's randomised ("Hard") setting, where *no method trains on randomised scenes*, GAM gets 47.6% versus 30.4% for the next best. On a real bimanual robot under visual shift, GAM keeps 17/20 successes where $\pi_{0.5}$ drops to 4/20.

Why this did not exist before: promptable 3D detectors that work on in-the-wild images are recent. GAM uses WildDet3D (2026). Without a backbone that reliably outputs metric 3D boxes from a single RGB image given an arbitrary prompt, this design has no foundation to stand on.

---

## The Methodology

Three stages: prompt → detections, detections → observation, observation → actions. Only the last stage is trained.

### Stage 1 — prompt to detections

Inputs: one RGB image $o_t$, and a task spec $\ell$ that is words, 2D points, or 2D boxes.

If the spec is language, a span-tagging head sits on a frozen Flan-T5 encoder and pulls out the object phrases. "Pick up the sponge and place it in the bowl" → two separate queries, `sponge` and `bowl`. Points and boxes skip this — they name the object directly.

The queries go to the frozen grounding backbone $\mathcal{G}$ (WildDet3D). For each of the $N$ queried objects it returns:

- a 2D box $u_i$,
- an oriented metric 3D box $b_i$ (centre, extents, rotation),

and, once per frame:

- a dense metric depth map $\hat{D}_t$,
- the dense visual features $F_t = \mathrm{Enc}_{\mathcal{G}}(o_t)$ from its own image encoder.

### Stage 2 — detections to observation

This is where the object-centric filtering happens. Two token streams plus one state token.

**Image tokens.** Average-pool $F_t$ down to a $16 \times 16$ grid. Keep a cell only if:

- some object box $u_i$ covers ≥ 30% of the cell's area, **or**
- the robot arm silhouette covers > 10% of it.

The arm silhouette is not learned — they pose the robot's URDF meshes with forward kinematics and project them into the camera using the known calibration. Every other cell's features are **set to zero**. A shared linear adapter turns the grid into $16^2 = 256$ image tokens.

So appearance survives, but only around the objects that matter and the gripper reaching for them.

**Detection tokens.** Back-project the whole depth map with the camera intrinsics, transform into the robot base frame → a scene point cloud. Crop the points inside a slightly inflated $b_i$ → that object's points. Sample $P = 512$ of them as $P_i$.

The point encoder $\phi$ is deliberately simple: a shared per-point MLP, then pooling — global max, global mean, **and** a max-pool inside each of eight octants around the centroid. Out comes a 512-dim shape feature. (The octant pooling is what gives it any sense of orientation; global pooling alone would throw that away.)

Each object becomes one descriptor:

$$z_i = \big[\,\tilde{u}_i,\; c_i,\; \gamma(c_i),\; \phi(P_i),\; e_i,\; r_i \,\big]$$

where $\tilde{u}_i$ is the normalised 2D box, $c_i$ the 3D centre, $\gamma(c_i)$ Fourier features of the centre at eight frequencies, $e_i$ the extents, and $r_i$ the box rotation in 6D form. A linear adapter maps $z_i$ into $M = 8$ tokens. Objects the detector missed get a **learned null token** in all eight slots, so the token count never changes.

**State history token.** Joint positions at the current frame and the previous frame, concatenated, projected to one token. Current configuration plus recent motion.

All three streams get learned modality embeddings and pass through one shared self-attention layer. Result: $C \in \mathbb{R}^{L \times d}$ with $L = 256 + 8N + 1$.

### Stage 3 — observation to actions

A 12-block **multi-stream transformer (MM-DiT)**, the same architecture idea as Stable Diffusion 3. Four streams:

1. image tokens,
2. detection tokens,
3. state-history token,
4. the noisy action chunk $A_t^\tau$, embedded as $H$ action tokens.

Each stream has its **own** Q, K, V, output projection, feed-forward and adaptive-layernorm parameters. Attention is joint over all four concatenated — so they exchange information, but each keeps its own weights for reading and writing. An RMSNorm plus a small MLP reads the action tokens out as a predicted velocity.

**Language conditioning adds no tokens.** The 768-dim sentence embedding $e_s$ is projected and added to the flow-timestep embedding:

$$h_{\tau,s} = \mathrm{TimeEmbed}(\tau) + W_\ell e_s + b_\ell, \quad W_\ell \in \mathbb{R}^{576 \times 768}$$

$W_\ell$ and $b_\ell$ are **initialised to zero**, so at step 0 the model is exactly the unconditioned one. This vector drives adaptive layernorm in every block. The division of labour is clean: *detections say which object and where; language says what operation to perform.*

**The loss is [[Flow Matching|flow matching]].** For a demonstration chunk $A_t$ (absolute joint-position targets plus gripper commands), draw $\epsilon \sim \mathcal{N}(0, I)$ and $\tau \in [0,1]$, build $A_t^\tau = \tau A_t + (1-\tau)\epsilon$, and regress the straight-line velocity:

$$\mathcal{L} = \mathbb{E}_{A_t, \epsilon, \tau}\Big[\big\|v_\theta(A_t^\tau \mid \tau, C, e_s) - (A_t - \epsilon)\big\|_2^2\Big]$$

At test time, integrate $v_\theta$ from $\tau = 0$ to $1$ with $K$ Euler steps starting from noise.

**What trains, and what does not.** Trained: image adapter, detection adapter and $\phi$, the fusion layer, the language projection, the MM-DiT. **Frozen: the grounding backbone $\mathcal{G}$.** Grounding capability is acquired once; acting on grounded objects is learned separately.

### Deployment

At episode start, $\mathcal{G}$ grounds the phrases, or takes points/boxes from a human or a planner. Then **CoTracker3** tracks points inside the initial detection boxes and feeds them back as prompts to $\mathcal{G}$ on later frames — that is how object identity is kept consistent across time rather than re-detected fresh each step.

For long-horizon work, a Molmo2 planner watches the current image plus episode history and picks new targets at sub-task boundaries, reinitialising trackers with new point prompts. The action policy itself is unchanged across sub-tasks — it has no idea it is in step 3 of 4.

---

## Ablation Studies and Experiments

### RoboTwin 2.0 — 50 bimanual tasks, clean (Easy) and randomised (Hard)

One policy per task, trained on the 50 clean demonstrations the benchmark provides, 100 rollouts each. **No randomised scenes are seen during policy training by anyone.**

One preparation step matters: the backbone was pretrained on real photos, so they fine-tune it alone on the simulator's rendered domain, supervised by 2D boxes, 3D boxes and depth exported straight from simulator state — no hand annotation. Then frozen.

| Method | Family | Easy | Hard | Avg |
|---|---|---|---|---|
| DP | action policy | 28.0 | 0.6 | 14.3 |
| DP3 (point cloud) | action policy | 55.2 | 5.0 | 30.1 |
| $\pi_0$ | VLA | 46.4 | 16.3 | 31.4 |
| $\pi_{0.5}$ | VLA | 64.0 | 25.9 | 45.0 |
| Abot-M0 | VLA | 57.4 | **30.4** | 43.9 |
| Spatial Forcing | VLA | **77.2** | 26.7 | 52.0 |
| FastWAM | WAM | 77.8 | 1.9 | 39.9 |
| X-WAM | WAM | 70.0 | 25.8 | 47.9 |
| **GAM** | grounded | 63.0 | **47.6** | **55.3** |

Read the Easy column first, because it is the honest part: **GAM is not the best in clean scenes.** Spatial Forcing hits 77.2% and FastWAM 77.8%, both well above GAM's 63.0%. GAM wins on the average purely by not falling over.

Retention tells the story. GAM keeps 76% of its clean performance under randomisation (63.0 → 47.6). FastWAM keeps 2.4% (77.8 → 1.9). Spatial Forcing keeps 35%.

**DP3 is the most instructive baseline.** It consumes the raw scene point cloud, so it has all the 3D information GAM has. Clean: 55.2%. Randomised: 5.0%. *3D sensing alone buys you nothing.* What matters is that the 3D input is filtered down to the task objects.

### LIBERO-PRO — 16 perturbation settings

Train on standard LIBERO demonstrations (each task has one fixed layout), then perturb at test time along four axes:

- **Obj** — object appearance and size change; layout and goal intact.
- **Pos** — object placement changes; layout broken.
- **Sem** — instruction rephrased; layout and goal intact.
- **Task** — the instruction now designates a *different* object in the same scene.

Obj and Sem leave the memorised trajectory valid. Pos and Task do not. That split is the whole diagnostic.

| Model | Spatial Pos | Spatial Task | Object Pos | Object Task | Goal Pos | Avg (16) |
|---|---|---|---|---|---|---|
| OpenVLA | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.48 |
| $\pi_0$ | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.45 |
| Flex-$\pi$ | 0.26 | 0.00 | 0.05 | 0.00 | 0.11 | 0.51 |
| $\pi_{0.5}$ | 0.20 | 0.01 | 0.17 | 0.01 | **0.38** | 0.53 |
| **GAM** | **0.60** | **0.88** | **0.47** | **0.50** | 0.19 | **0.61** |

The baselines score 0.90–0.99 on Obj and Sem, then hit **exactly 0.00** across most of Pos and Task. That is not degradation, it is a different failure: the visuomotor mapping is fixed and the scene no longer matches it.

**Task is the sharpest result.** The instruction now names an object that was only background in training — but every object and every action involved *did* appear somewhere in the training set. So the capability exists; only the pairing is new. Every baseline stays below 0.11 on all four suites. GAM gets 0.88 on Spatial. Because the new target changes the detection tokens, and the policy acts on whatever is in those tokens.

**Where GAM loses.** Obj on LIBERO-Goal (0.69) and LIBERO-10 (0.50), behind $\pi_{0.5}$'s 0.97 and 0.92. And Goal-Pos (0.19 vs 0.38). The authors' reading of the Obj gap: changing object *size* moves the right grasp point. The point clouds do carry the new geometry, but the policy has never seen a demonstration grasping an object that size, so it cannot adjust the motion. Geometry in the observation is not the same as geometry in the motor skill.

### Real robots

**Bimanual YAM.** Three same-coloured objects per side; each arm must place one designated object into a shared container. Four target combinations, 25 demonstrations each, both methods fine-tuned on the same data. GAM gets the target as a click or box; $\pi_{0.5}$ gets it as language. Neither uses a planner or memory.

| Method | ID | OOD (visual shift) | Retention |
|---|---|---|---|
| $\pi_{0.5}$ | 19/20 | 4/20 | 21% |
| **GAM** | 19/20 | **17/20** | **89%** |

Identical in distribution. The gap only opens under shift.

**Single-arm Franka, step completion rate.** Four tasks: two long-horizon (*Sort objects into three bowls*, *Clear the table*) and two memory-dependent (*Uncover the R, G, B blocks in order*, *Restore original layout*). 50 demonstrations per task for everyone. GAM is driven by a Molmo2 planner through the point interface; baselines use their own language interfaces with no external planner.

| Method | ID | OOD | Retention |
|---|---|---|---|
| MolmoAct2 | 35.3% | 17.1% | 48% |
| $\pi_{0.5}$ | 42.4% | 24.0% | 57% |
| **GAM + Molmo2** | **64.7%** | **49.8%** | **77%** |

Caveat worth naming: this is not a controlled comparison of backbones. GAM has a planner and the baselines do not. What it does show is that the point interface works as a channel between a reasoning model and a controller — the controller never learns anything about ordering or memory, and still completes multi-step memory-dependent tasks.

Attention maps (Fig. 4) back this up qualitatively: under shift, baseline attention spreads over the perturbed background and distractors; GAM's stays on the targets. Expected, since the background is literally zeroed out of its input.

### The ablations — 10 RoboTwin tasks, 20 rollouts each

Backbone, action head, data and schedule all held fixed. Whole-scene point clouds keep the same 512-point budget.

| Variant | Image stream | Detection stream | Easy | Hard | Avg |
|---|---|---|---|---|---|
| GAM (full) | grounded | grounded | 51.5 | 42.0 | **46.8** |
| (a) no image masking | full scene | grounded | 30.5 | 14.0 | 22.3 |
| (b) no point cropping | grounded | full scene | 18.5 | 7.0 | 12.8 |
| (c) neither filter | full scene | full scene | 17.0 | 3.5 | 10.3 |
| (d) detection only | — | grounded | 19.5 | 12.5 | 16.0 |
| (e) image only | grounded | — | 21.5 | 19.0 | 20.3 |

**The filtering is doing the work, not the backbone.** Every row shares the same frozen $\mathcal{G}$. Turn off both filters and average success goes 46.8 → 10.3, with Hard collapsing 42.0 → 3.5. Same perception, same data, same architecture — the gains come entirely from *which* backbone outputs reach the policy and how they are represented.

**Point cropping matters more than image masking.** Removing it costs more (12.8) than removing masking (22.3). Plausible reason: an uncropped point cloud spends its 512-point budget on background and may contain very few points on the target at all. A masked-out image region at least degrades gracefully.

**Both streams are needed, and neither is close alone.** Detection only: 16.0. Image only: 20.3. Together: 46.8. That is far more than additive. Detection tokens give metric geometry but no texture or fine visual cue; image tokens give appearance around the target and the gripper but no metric extent. Appearance tells you what you are grasping, geometry tells you where to close the fingers.

---

## Worth Remembering

**Limitations the authors state plainly.**

- **Grounding errors propagate with no recovery path.** If $\mathcal{G}$ mislocalises, the policy acts on a wrong box and nothing downstream can notice. There is no confidence gate, no fallback.
- **Object-centric filtering can delete things you needed.** An unselected obstacle in the path is simply not in the observation. This is the direct cost of the same mechanism that buys the robustness.
- **3D boxes are a coarse grounding.** They suggest dense 3D instance segmentation as a finer alternative, and explicitly invite others to swap in different grounding backbones — WildDet3D is an instance of the paradigm, not the paradigm.

**Things worth flagging that the paper does not emphasise.**

- The RoboTwin protocol required fine-tuning the backbone on rendered simulator images first. That is cheap (labels come free from simulator state) but it *is* a domain-adaptation step that language-backbone methods did not perform. Whether the frozen real-image backbone would have worked untouched is not reported.
- The Franka comparison gives GAM a planner and the baselines none. The ID gap (64.7 vs 42.4) is probably more about the planner than about grounding.
- GAM is the *worst* of the good methods in clean, in-distribution scenes on RoboTwin (63.0 vs 77.2). If your deployment is a fixed cell with fixed lighting and fixed layout, this design costs you accuracy.

**The general lesson, separable from the robots.** Two of the paper's results are about shortcut learning, not manipulation. Baselines score ~0.95 on the original benchmark and **0.00** when the target is relocated. The benchmark was measuring trajectory memorisation and everyone read it as instruction following. LIBERO-PRO's contribution is partly a critique of LIBERO, and it rhymes with [[Shortcut Learning in Deep Neural Networks]] and with the replication concerns in [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]]. Related: [[Do ImageNet Classifiers Generalize to ImageNet]].

**Connections.** The MM-DiT with per-stream weights and joint attention is lifted straight from [[Diffusion Transformer]] / SD3. The training objective is plain [[Flow Matching]], not diffusion. The zero-initialised language projection is the same safety trick as adaLN-zero and [[LoRA]]'s zero-initialised $B$ — start as the identity, let the model earn the deviation. The frozen-backbone-plus-small-trained-head shape is [[Fine-Tuning]]'s cheapest rung, and separating "acquire grounding" from "learn to act" is the same factorisation as [[Foundation Models|pretrain-then-adapt]].

**Follow-up questions.**

1. What happens when grounding fails mid-episode? No numbers on detection recall in randomised scenes, and detection recall is the actual ceiling on the whole system.
2. CoTracker3 does the frame-to-frame identity work. How much of the temporal stability is the tracker rather than the detector? No ablation of the tracker.
3. Does the Obj weakness go away with demonstrations at varied object sizes, or is it an architectural limit in how $\phi$ pools the point cloud?
4. Occlusion. Everything here is single-view RGB with predicted metric depth. A target behind another object has a wrong 3D box and the policy has no way to know.

**Practical caveats for building on this.** You need a promptable 3D detector that works in your domain — that is the hard dependency, and the paper spent a fine-tuning stage on it for simulation. You need known camera calibration and the robot's URDF, since the arm silhouette is projected geometrically rather than learned. The 30% / 10% overlap thresholds for keeping grid cells are not ablated and are likely worth tuning for your camera resolution and object scale.

---

## Links

Related: [[Flow Matching]] · [[Diffusion Transformer]] · [[Shortcut Learning in Deep Neural Networks]] · [[Think Like a World Model, Act Like a VLA- Distilling World-Model Representations into Compact Robot Policies]] · [[Breaking the Vision-Action Shortcut- Latent Interface Training for Generalizable Robotics Foundation Models]] · [[Latent Action as Intention Enables Efficient Future Imagination for World Action Models]] · [[Imitation Learning]] · [[Fine-Tuning]] · [[Foundation Models]] · [[Diffusion Policy]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Do ImageNet Classifiers Generalize to ImageNet]] · [[Rectified Flow]] · [[Cross Attention]]

New topics worth writing: promptable 3D object detection, metric depth estimation from single RGB, object-centric policy observations, point cloud encoders and permutation-invariant pooling, point tracking (CoTracker), URDF and forward kinematics for policy inputs, hierarchical planner-controller interfaces in robotics, LIBERO and LIBERO-PRO benchmark design, RoboTwin 2.0 domain randomisation, action chunking
