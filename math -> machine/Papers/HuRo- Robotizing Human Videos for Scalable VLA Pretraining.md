---
title: "HuRo: Robotizing Human Videos for Scalable VLA Pretraining"
authors: ["Jeong et al."]
year: 2026
arxiv: "2609.10706"
url: https://arxiv.org/abs/2609.10706
priority: Good-To-Read
read_on: 2026-09-27
tags: [paper, diffusion, vision]
---
## The Core Idea

Robot training data is expensive. A human has to teleoperate a real robot, one episode at a time. Human videos — people cooking, cleaning, picking things up — are almost free and cover far more objects, rooms and camera angles. The problem is that a video of a human hand is not a video of a robot: the pixels show skin, and there are no robot joint angles anywhere.

The idea here is to **fake the robot into the video, and fake the robot's actions out of the hand**. Take an egocentric (head-mounted camera) human video. Cut the arms out, paint over the hole, render a real robot's arms in exactly the place the human's arms were, and solve for the joint angles that put the robot's fingertips where the human's fingertips were. Now you have an episode that *looks* like robot data and *is labelled* like robot data: image, robot state, robot action, language instruction. Pretrain on that.

What is new is not any single piece — visual overlay and motion retargeting both existed. The new thing is doing both together, **at scale, on messy videos that were never collected for the downstream task**. Earlier work did joint observation+action robotization only in "task-matched" settings, where someone filmed a human doing the exact task the robot would later do. Other work took actions from human video but kept the human pixels. The open question was whether you could just point this at Ego4D and EPIC-Kitchens and treat it as a generic pretraining corpus.

The answer: yes. The resulting **HuRo** dataset is ~630K episodes, 142M frames, ~1,317 hours at 30 fps, from five sources — over 10× larger than prior robotized-video pretraining sets. A [[Diffusion Policy|VLA]] policy pretrained on all of it, then fine-tuned on 16–43 real demos per task, goes from **51.5% → 80.3%** overall completion on four real bimanual tasks. Under distribution shift the gap is bigger: **34.9% → 72.2%**.

> [!NOTE] Robotization
> Editing a human video so that both the pixels and the labels look like they came from a robot: human arms removed and replaced with a rendered robot, human hand motion converted into robot joint/wrist targets. ^robotization

> [!NOTE] Embodiment gap
> The mismatch between the demonstrator and the robot. It has two halves that can be fixed independently — the **observation** gap (the image shows a hand, not a gripper) and the **action** gap (a MANO hand pose is not a 7-DoF arm command). Most prior work fixes one. ^embodiment-gap

The two headline ablations tell you which half matters. Keeping human pixels but using retargeted actions ("no-overlay") matches full HuRo in-distribution (89.4% vs 88.4%) but collapses out-of-distribution (55.7% vs 72.2%) — worse than using only **10%** of the data with overlay. And keeping the pretrained vision but throwing away the pretrained action head gives almost nothing. So: visual robotization buys robustness, action supervision buys the actual skill.

## The Methodology

Three stages per video clip. Videos are first standardised to 30 fps and resized.

### Stage 1 — Human video annotation

This stage guesses everything the source dataset did not hand you. Different sources arrive with different annotations (EPIC-Kitchens and Ego4D are raw RGB; Ego10K gives intrinsics; EgoDex and EgoVerse give camera geometry and hand pose), so the pipeline fills in only the gaps.

- **Camera intrinsics**: `droidcalib` self-calibration from camera motion; falls back to `AnyCalib` for near-static clips. Frames are then rectified to a pinhole image.
- **Hand detection**: `100DoH` per frame for per-side hand boxes, with `BOT-SORT` tracking to keep left/right assignment stable over time (a per-frame detector flips hands constantly).
- **Hand pose**: `HAWOR` gives a MANO hand pose $P_t^h$ for $h \in \{\mathsf{L},\mathsf{R}\}$ in the rectified camera frame. From this they read wrist pose, fingertip positions and local hand-structure directions.
- **Camera trajectory**: masked `DROID-SLAM`, with the hands masked out so the moving foreground does not poison the [[System Identification|SLAM]] solve. Monocular SLAM has no absolute scale, so `MoGe-2` supplies metric scale, and `GeoCalib` rotates the world frame so gravity points down. Output: $T_t^{W \leftarrow C_{\mathrm{src}}}$.
- **Chunking + captions**: frames with valid hands become manipulation segments, split into bounded-length chunks. For each chunk they draw the projected wrist trajectory onto sampled RGB frames and ask Qwen3.5 for one instruction $l_i$. A second verification pass checks the caption against the frames and drops chunks with no real hand–object manipulation.

An annotated chunk is

$$\mathcal{H}_i = \Big(\{I_\tau\},\, K,\, \{T_\tau^{W \leftarrow C_{\mathrm{src}}}\},\, \{P_\tau^h\},\, l_i\Big)$$

### Stage 2 — Action conversion

Fingertip positions and hand-structure directions get pushed into the world frame $W$ using the camera poses. But $W$ is the *video's* world, not the robot's base frame $B$. So they fit a per-chunk alignment $T_{\xi_i}^{B \leftarrow W}$ — only a 3D translation plus a yaw rotation, which is the right restriction because gravity is already aligned.

Inverse kinematics is solved with `PyRoKi`, in two passes:

1. **Sparse pass.** On a subset of timesteps, jointly optimise the alignment $\xi_i$ *and* the joint configs $q_\tau$. Objective terms: fingertip position tracking, local hand-structure directions (keeps relative finger geometry), ego-view consistency (the robot's head camera link, via forward kinematics, should sit where the recovered human camera sits), plus joint-limit and rest-pose penalties. The sparse $q_\tau$ are thrown away — this pass exists only to find a *reachable* place to put the hand motion.
2. **Dense pass.** Freeze $\xi_i$, re-solve $\{q_\tau\}$ for every frame with the same terms plus a temporal smoothness penalty.

Joints become policy states $s_\tau$ by forward kinematics. There are no real robot actions anywhere, so the action label is just the next state:

$$a_\tau = s_{\tau+1}, \qquad a_{T_i-1} = s_{T_i-1}$$

The same alignment maps the human camera into the robot base frame, which is what the renderer needs:

$$T_\tau^{B \leftarrow C_{\mathrm{obs}}} = T_{\xi_i}^{B \leftarrow W} \, T_\tau^{W \leftarrow C_{\mathrm{src}}}$$

### Stage 3 — Visual conversion

- Segment visible human arms with `SAM2`, prompted by `Detectron2` person regions where needed.
- Inpaint the holes with `ProPainter` → cleaned frames $\bar I_\tau$.
- Render the robot in Isaac Sim using $K$, $q_\tau$ and $T_\tau^{B \leftarrow C_{\mathrm{obs}}}$, composite onto $\bar I_\tau$ → robotized observation $\tilde I_\tau$.

Final episode: $\mathcal{E}_i = \{(\tilde I_\tau,\, s_\tau,\, a_\tau,\, l_i)\}$.

### Dataset composition

Target robot for the main dataset is **ALLEX**: two 7-DoF arms, two 15-DoF dexterous hands, 2-DoF neck, 2-DoF waist (neck and waist frozen during data collection and evaluation). By frame count: 56% EgoDex, 27% EgoVerse, 10% Ego4D, 6% Ego10K, 2% EPIC-Kitchens.

### The policy

GR00T-N1.6-3B, end-effector action interface. The [[CLIP|VLM]] backbone encodes $\tilde I_t$ and $l_i$ into $\phi_t$; conditioned on $\phi_t$ and $s_t$, the action head emits an $H=40$ step action chunk.

Action layout per step: left/right wrist targets + left/right hand-joint targets. Wrist targets are **relative to the current wrist pose**, in the observation camera frame at time $t$ — a 3D translation plus the continuous 6D rotation representation (Zhou et al., which avoids the discontinuities of Euler angles or quaternions). Hand joints are absolute.

Training: backbone from the released GR00T checkpoint, action head from scratch, [[Flow Matching|flow-matching]] objective, visual encoder jointly fine-tuned.

| | steps | batch | LR | schedule |
|---|---|---|---|---|
| Pretrain | 80k | 2048 | $1{\times}10^{-4}$ | constant + linear [[GPU processing#Fix 2: warmup (this is the non-negotiable one)\|warmup]] |
| Finetune | 30k | 128 | $1{\times}10^{-4}$ | cosine decay |

[[Decoupled Weight Decay Regularization (AdamW)|AdamW]], weight decay $1{\times}10^{-5}$.

## Ablation Studies and Experiments

Four real ALLEX tasks. Demo counts are tiny, which is the point — pretraining is supposed to cover for that.

| Task | Demos | ID / OOD rollouts | Scoring |
|---|---|---|---|
| Apple Pick-and-Place | 43 | 12 / 12 | binary |
| Cup Stacking | 40 | 12 / 24 | 3 subgoals |
| Cup-Noodle Handover | 16 | 12 / 12 | 3 subgoals |
| Microwave Loading | 20 | – / 10 | 3 subgoals |

Subgoal scores are normalised by the max before averaging. OOD means spatial shifts (moved objects, moved microwave) and visual shifts (a checkered tablecloth). Microwave Loading is OOD-only.

### Main table (ID / OOD / overall)

| Model | ID | OOD | Overall |
|---|---|---|---|
| $\pi_{0.5}$ | 68.5 | 28.0 | 48.2 |
| GR00T N1.6 | 66.7 | 37.4 | 52.0 |
| 0% PT (no pretraining) | 68.1 | 34.9 | 51.5 |
| 10% PT | 76.9 | 59.5 | 68.2 |
| 50% PT | 78.2 | 69.8 | 74.0 |
| no-overlay (100% data) | **89.4** | 55.7 | 72.5 |
| **100% PT** | 88.4 | **72.2** | **80.3** |

Pretraining steps and batch size are held fixed across the data-scale variants, so this is a data-quantity curve, not a compute curve.

Two things jump out. First, ID saturates fast (76.9 → 88.4 from 10% to 100%) while OOD keeps climbing (59.5 → 72.2). Second, **no-overlay beats full HuRo on ID** and loses badly on OOD — it is even below the 10% overlay model out of distribution. More human-domain data makes you better at the exact fine-tuning distribution and does not make you robust.

Cup Stacking as a case study: OOD completion 60.4% (10% PT) → 79.2% (100% PT). Qualitatively, small-subset models approach cups off-centre, slip grasps, topple stacks; the failures get worse under the checkered cloth.

### Does action supervision matter, or is it just a better visual encoder?

New task, *Diverse Pick-and-Place*: grab two objects with different grasp affordances, put both in a basket. 4 object types, one point per object placed with an *appropriate* grasp, max 2 per rollout. 36 ID trials; 18 OOD trials with unseen cup instances at unseen locations.

| Setting | ID | OOD |
|---|---|---|
| No PT | — | — |
| PT (Visual Only) | modest gain | modest gain |
| PT (Visual + Action) | **61.1** | **50.0** |

Visual-only reinitialises the action head before fine-tuning and barely improves on no pretraining. Full pretraining is much better, consistently across every object pair. Qualitatively: visual-only grabs the scrub brush from above instead of by the handle, and can reach the unseen cup but cannot get fingers through the handle. This is the paper's cleanest result — **the action head is where the transfer lives**, which is exactly the thing H2R and Masquerade-style visual pretraining leave on the table.

### Versus generated video

Baseline: `I2V + IDM` (following DreamGen / RoboCurate) — an image-to-video model adapted to ALLEX generates manipulation videos, and an inverse dynamics model labels pseudo-actions. Matched frame budgets of 0.7M / 3.5M / 7.0M.

HuRo wins at every budget, and **HuRo at 0.7M beats I2V+IDM at 7.0M** on both ID and OOD. I2V+IDM's OOD flatlines from 3.5M to 7.0M; HuRo keeps improving. Generated video inherits the generator's distribution; real video does not.

### Versus keeping the human pixels (Appendix C)

Two human-domain baselines on the same clips and scale: **Human-HRDT** (9D wrist + 5 fingertip positions in wrist frame, 48 dims both hands) and **Human-VITRA** (9D wrist + 15 MANO joint parent-relative Euler rotations, 108 dims both hands). Evaluated OOD in a fresh environment with an unseen background.

| Method | Cup Stacking | Cup-Noodle Handover |
|---|---|---|
| No PT | 31.9 | 16.7 |
| Human-HRDT | 5.6 | 37.5 |
| Human-VITRA | 0.0 | 9.7 |
| 100% HuRo | **70.8** | **66.7** |

Human-VITRA is **worse than no pretraining on both tasks**. Human-HRDT helps on handover and destroys Cup Stacking (5.6 vs 31.9). Human-domain pretraining is not reliably positive transfer. Failures concentrate at alignment/placement and at the bimanual transfer, where relative pose between the two hands matters.

### Masquerade-style comparison (Appendix B)

Controlled: all variants share HuRo annotations on a 2.4M-frame EPIC-Kitchens subset, same architecture, same recipe. Only robotization differs.

| Method | ID | OOD |
|---|---|---|
| No PT | 63.9 | 25.0 |
| Fixed-EEF (Masquerade-style) | 75.0 | 50.0 |
| Fixed-EEF + Hand | 80.6 | 61.1 |
| HuRo-EEF + Hand | **86.1** | **63.9** |

Ladder of three findings: (a) giving Masquerade action supervision already helps a lot; (b) adding hand-joint targets on top of end-effector targets helps more; (c) camera-motion-aware alignment beats a fixed camera→robot extrinsic. The egocentric camera *moves*, so pretending it does not costs you real points.

### Other embodiments (Appendix E)

Annotations, arm masks and inpainted backgrounds are all embodiment-independent — only retargeting and overlay must be recomputed, which is 10.6% (EPIC) / 5.5% (Ego4D) of the pipeline cost.

Transfer to **OpenArm** (XHand1 hands, 42-dim state/action) on a Fruit Pick-and-Place task, ALLEX-pretrained: HuRo PT gets 70.8 ID / 75.0 OOD / 72.9 overall; Human-HRDT 75.0 / 56.3 / 65.6; visual-only 62.5 / 25.0 / 43.8. Visual-only is **worse overall than no pretraining** here (43.8 vs 49.0) — a sharper version of the same finding.

Joint ALLEX+OpenArm pretraining on the same underlying videos (2× data, same updates): 91.7 / 77.8 versus ALLEX-only 86.1 / 63.9.

### Coverage diagnostics (Appendix D)

Visual coverage against 20K OpenImages references using DINOv3 features:

$$\mathrm{Cov}_{\mathrm{OI}}(D) = \frac{1}{|Q_{\mathrm{OI}}|}\sum_{q \in Q_{\mathrm{OI}}} \max_{x \in D} \cos\big(f(q), f(x)\big)$$

| Variant | OI cov. | Unique verbs | Unique objects | Unique verb-object |
|---|---|---|---|---|
| Mixed 10% | 0.664 | 521 | 1,300 | 11,787 |
| Mixed 50% | 0.678 | 778 | 1,980 | 25,656 |
| Mixed 100% | 0.687 | 959 | 2,344 | 35,358 |
| EgoDex-only | 0.616 | 136 | 268 | 938 |

EgoDex-only has the same sampled frame count as Mixed 50% and **27× fewer verb-object pairs**. Downstream: Mixed 50% (71.1M frames) beats EgoDex-only (78.9M frames) at 78.2/69.8 vs 63.9/54.2 — fewer frames, more diversity, better policy. And 0.7M mixed frames match 2.4M EPIC-only frames (83.3/63.9 vs 86.1/63.9) at 3.4× less data.

## Worth Remembering

**The retargeted trajectories are not executable.** This is the limitation to internalise. In a 288-trajectory audit across five sources, only **55.2%** had no detected non-grasp self-contact, and only **62.5%** had no URDF joint-limit violation above $1°$. The IK has no self-collision term at all. Exact projection onto joint limits fixes all of them, and for 95.6% of samples moves fingertips by under 1 mm — but the self-collision problem stands. These are *pretraining labels*, not demonstrations you could replay on hardware.

**No force or tactile channel.** Purely visual and kinematic. Contact-rich manipulation is not covered.

**The overlay does not model occlusion.** The rendered robot is composited on top; scene geometry does not occlude it. Plus residual inpainting artefacts. How robotization *fidelity* trades against downstream performance is completely unexplored — which is the most interesting open question the paper leaves.

**Pipeline cost: 8–10× real time on one RTX 5090.** 143 min for a 15-min EPIC clip, 120 min for Ego4D. Annotation is 63–65% of that; action conversion is only 4–5%. So the cheap part is the robot-specific part, which is why multi-embodiment is nearly free.

**Retention is brutal on in-the-wild footage.** Only 17.3% of EPIC-Kitchens frames and **6.9%** of Ego4D frames survive manipulation-segment selection. After selection, 97–98% make it through. Across all five sources: 3,298.6 hours in → 1,316.8 hours out.

**Reconstruction quality numbers** (EgoDex reruns vs released annotations): camera rotation error 0.218° median, position ATE 4.47 mm median, root-relative hand keypoint error 20.4 mm median, post-IK fingertip residual 21.2 mm median, inpainting 24.8 dB, person-pixel reduction 93.3%. The ~20 mm hand error and ~21 mm IK residual compound — which is probably why grasp precision is the dominant failure mode at small data scales.

**The evaluation is small.** 10–24 rollouts per task per condition. A single rollout is ~4–8 percentage points on a 12-trial binary task, so differences under ~10 points between neighbouring variants should be read as noise. The 34.9 → 72.2 OOD jump is far outside that; the 89.4 vs 88.4 ID tie certainly is not a real difference.

**Questions to follow up.** Does the $a_\tau = s_{\tau+1}$ next-state action label limit anything, given real robot controllers see commanded rather than achieved states — is there a train/test action-space mismatch at fine-tuning time? Does the 56% EgoDex share mean the corpus is effectively a lab-quality dataset with a diverse tail, and what happens at 10% EgoDex? Given the joint two-embodiment result, is there a "robot-agnostic" pretraining regime where you deliberately robotize to many robots to force the policy to learn task structure rather than kinematics?

**Connections.** The overlay-vs-no-overlay result is the same shape as a [[Shortcut Learning in Deep Neural Networks|shortcut learning]] story: with human pixels the policy can key on skin-coloured cues that do not survive a tablecloth change. The 6D rotation representation choice matters more than it looks — discontinuous parameterisations make the regression target badly behaved. And the I2V+IDM comparison is worth filing next to any argument about synthetic data scaling: it improved then stopped, exactly where you would expect a generator's support to run out.

## Links

Related: [[Flow Matching]] · [[Diffusion Policy]] · [[Imitation Learning]] · [[Shortcut Learning in Deep Neural Networks]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Fine-Tuning]] · [[Distillation]] · [[Self-Supervised Learning from Images with I-JEPA]] · [[Think Like a World Model, Act Like a VLA- Distilling World-Model Representations into Compact Robot Policies]] · [[Latent Action as Intention Enables Efficient Future Imagination for World Action Models]] · [[Breaking the Vision-Action Shortcut- Latent Interface Training for Generalizable Robotics Foundation Models]] · [[Zero-WAM- In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization]] · [[GPU processing]]

New topics worth writing: Vision-Language-Action models, motion retargeting and inverse kinematics for manipulation, MANO hand model, monocular SLAM and metric scale recovery, video inpainting, egocentric video datasets (Ego4D / EPIC-Kitchens / EgoDex), 6D continuous rotation representation, inverse dynamics models for pseudo-action labelling, sim-to-real and embodiment transfer, dataset coverage metrics for pretraining corpora
