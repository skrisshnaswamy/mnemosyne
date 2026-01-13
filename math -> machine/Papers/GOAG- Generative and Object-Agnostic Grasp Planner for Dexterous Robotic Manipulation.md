---
title: "GOAG: Generative and Object-Agnostic Grasp Planner for Dexterous Robotic Manipulation"
authors: ["Julien Merand", "Boris Meden", "Mathieu Grossard", "Liming Chen"]
year: 2026
arxiv: "2608.19759"
url: https://arxiv.org/abs/2608.19759
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper]
---
## The Core Idea

When a hand grips a mug, the patch of skin touching the mug and the patch of mug touching the skin are **the same surface**. Two descriptions of one contact region. Every dexterous grasp planner before this learned that region from the object's side: collect thousands of objects, run an expensive analytical planner to find stable grasps, train a network to map object shape → contact map. That means the model only ever sees contacts that happen to occur on the objects in your dataset, and it inherits that bias.

GOAG flips the frame. Take the object, apply the *inverse* of the grasp pose $[R,T]^{-1}$, and now you are sitting inside the gripper's own coordinate frame. From here, contact is a property of the **gripper**: which parts of the palm and finger pads are active, in what combination. That set of "which pads touch together" patterns is finite and it is decided entirely by the hand's kinematics — **no object is required to enumerate it**.

So they train a generative model on gripper geometry alone. Sample random joint angles, pick a grasp type from a human grasp taxonomy, sprinkle contact points inside the regions that grasp type allows, and learn the distribution. The object shows up only at inference: you feed in the object's distance field instead of the gripper's, and the decoder — which has only ever learned "what contact patterns are reachable for me" — reads out the parts of the object that are compatible with the hand.

What this unlocks: dataset generation drops from **1,400 A100-GPU-hours** (GenDexGrasp) to **~1 hour on one RTX 4090**, and the model generalises zero-shot to any object shape, because it never overfit to any object shape in the first place. On MultiDex it gets **86.93%** average success across three hands, beating every baseline including ones trained directly on that data.

> [!NOTE] Object-agnostic training
> Training uses only the robot hand's mesh and joint limits. Object point clouds enter the pipeline exclusively at inference time, through the same encoder. ^object-agnostic

## The Methodology

**Notation.** A grasp is a palm pose $[R,T]$ plus joint values $Q$. The *handprint* $\mathcal{H}=\{h_i\}$ is the set of surface points on the hand's active grasping faces — palm, inward pads and immediate finger sides. These are pulled out of the full mesh by thresholding the dot product of each vertex normal with the palm's facing direction.

**The reframing.** Standard contact-map methods define contacts on the object:

$$\mathcal{C}(\mathcal{O}) = \{o_i \in \mathcal{O} \;:\; \exists h_j \in \mathcal{H}(R,T,Q),\; \mathcal{D}_{aligned}(o_i,h_j) < \epsilon\}$$

GOAG defines them on the hand, with the object moved into the hand's canonical frame:

$$\mathcal{C}(\mathcal{H}(Q)) = \{h_i \in \mathcal{H}(Q) \;:\; \exists o_j \in \mathcal{O}([R,T]^{-1}),\; \mathcal{D}_{aligned}(h_i,o_j) < \epsilon\}$$

The distance is not plain Euclidean — it penalises misalignment of surface normals, so a point "in front of" a surface counts as closer than one grazing sideways:

$$\mathcal{D}_{aligned}(x,y) = e^{\gamma(1 - \langle x-y,\, n_x\rangle)}\sqrt{\|x-y\|_2}, \qquad \gamma = 2.0$$

**Data generation (the whole trick, and it is cheap).** Uniformly sample $10{,}000$ kinematically valid joint configurations $Q$. For each, pick one of **6 grasp types** taken from a human grasp taxonomy adapted to the robot hand (these 6 cover 92.5% of a machinist's working time and 96.2% of a housemaid's, per prior ergonomics studies). Each grasp type paints an *admissible contact region* on the hand surface — e.g. for a pinch, only the fingertip pads. Then randomly sample 50 sets of active contact points strictly inside those regions. Result: 3,000,000 labelled point clouds, one GPU hour.

Why the taxonomy and not pure random sampling? Random contacts would be arbitrary and independent — thumb tip plus ring-finger base plus palm edge, a pattern no real grasp uses. The taxonomy enforces **kinematic harmony**: the sampled contacts co-occur the way fingers actually co-occur.

> [!NOTE] Basis Point Set (BPS)
> A fixed set of $M$ reference points in space. Any variable-size point cloud gets encoded as the vector of distances from each basis point to the cloud — a fixed-length $\mathbb{R}^M$ vector a plain MLP can eat. ^basis-point-set

**The encoding.** Rather than a bounding box, the basis set $W \in \mathbb{R}^{M\times 3}$ is the discretised **kinematic workspace** of the hand — the volume the fingers can actually reach relative to the palm frame, $M = 8{,}192$ points. This concentrates resolution exactly where grasps happen and, critically, is the *same* frame at train and test time, which is what makes the domain swap (hand cloud → object cloud) coherent.

For any input cloud $P$ (hand at train, object at test): $\mathcal{F}_P \in \mathbb{R}^M$ with $\mathcal{D}(P, w_i) = \min_{p_j \in P} \mathcal{D}_{aligned}(p_j, w_i)$.

The supervision target is a soft contact likelihood per basis point:

$$c_i = 1 - 2\big(\mathrm{Sigmoid}(\mathcal{D}(\mathcal{C}(P), w_i)) - 0.5\big) \in [0,1]$$

**The CVAE.** A [[Auto-Encoding Variational Bayes (VAE)|conditional VAE]]. Encoder eats the concatenation $[\mathcal{F}_P, \tilde{\mathcal{C}}(P)]$ of shape $M \times 2$ and outputs $q_\theta(z \mid \mathcal{F}_P, \tilde{\mathcal{C}}(P))$; latent dim $\psi = 128$. Decoder is two fully-connected ResBlocks on $[z, \mathcal{F}_P]$, predicting $\hat{\mathcal{C}}(P)$.

$$L = L_{\text{recon}} + \beta\, D_{\text{KL}}\big(q_\theta(z \mid \mathcal{F}_P, \tilde{\mathcal{C}}(P)) \,\|\, \mathcal{N}(0,I)\big), \qquad \beta = 0.01$$

The reconstruction term is an attention-weighted $L_2$ — because most of the 8,192 basis points have $c_i \approx 0$, and an unweighted loss would just learn "predict zero everywhere":

$$L_{\text{recon}} = \sqrt{\frac{\sum_{i=1}^{M}(c_i - \hat{c}_i)^2 e^{\alpha c_i}}{\sum_{k=1}^{M} e^{\alpha c_i}}}, \qquad \alpha = 3.0$$

**Links Mapper.** A PointNet++ trained in parallel, on the raw sampled point clouds, mapping each contact point $p_i$ to the index $l_i$ of the phalanx that should touch it. Without this the predicted contact cloud is geometrically plausible but kinematically meaningless — you know *where* to touch but not *with what*.

Training: 100 epochs CVAE, 50 epochs PointNet++, batch 128, multiple A100s. The [[KL Divergence|KL]] term is deliberately weak ($\beta = 0.01$) so the latent stays expressive.

**Inference.**
1. Put object $\mathcal{O}$ at pose $[R,T]^{-1}$ in the hand's frame, compute $\mathcal{F}_\mathcal{O}$ against the same $W$.
2. Skip the encoder. Draw $z \sim \mathcal{N}(0,I)$, decode $\hat{\mathcal{C}}(\mathcal{O})$.
3. Threshold at $\hat{c}_i > \tau = 0.8$ → discrete contact set $\mathcal{C}(W)$. Note this stays in workspace coordinates, never projected back onto the object surface.
4. Links Mapper labels each point with a phalanx.
5. **Force-closure check.** Group points by phalanx, take each cluster's barycenter, project onto the object surface → contacts $b_i$. Coulomb friction $\mu = 0.3$, no gravity. Build the grasp wrench space $d = \sum_i G_i f_i$ (convex hull of unit friction-cone edge forces) and test whether the origin lies strictly inside. Fail → resample $z$, up to **20 tries**, then give up (the pose is probably near a workspace boundary).
6. **Joint optimisation.** Solve for $Q^*$:

$$Q^* = \arg\min_Q \big(\lambda_d E_{dist} + \lambda_p E_{pen} + \lambda_s E_{spen} + \lambda_j E_j\big)$$

with $\lambda_d = 1.0,\ \lambda_p = 50.0,\ \lambda_s = 0.1,\ \lambda_j = 1.0$. Terms: $E_{dist} = \sum_{p \in \mathcal{C}(W)} \min_{h \in \mathcal{H}_{l(p)}(Q)} \|p-h\|_2^2$ (pull the *assigned* link to each point); $E_{pen} = \sum_h \max(0, -\Phi_\mathcal{O}(h))$ (object SDF penetration, weighted 50× — collisions are the thing that kills you); $E_{spen}$ (self-collision, safety margin $\epsilon = 0.025$ m between disjoint link bounding-box centroids); $E_j$ (joint limits).

**Grasp pose sampling.** Palm poses are sampled uniformly on the object's convex hull dilated to 110%, palm oriented anti-parallel to the hull normal.

## Ablation Studies and Experiments

**MultiDex test set, 10 objects from ContactDB + YCB, three hands, 100 inference runs each.** Success = object moves < 2 cm after 1 second of external force along each of $\pm x, \pm y, \pm z$ in Isaac Gym.

| Method | Obj-agnostic | Barrett | Allegro | Shadow | **Avg** | sec/grasp (Barrett/Allegro/Shadow) |
|---|---|---|---|---|---|---|
| DFC (analytical) | ✓ | 83.10 | 82.71 | 72.15 | 79.32 | >1800 each |
| GenDexGrasp | ✗ | 70.26 | 71.48 | 71.15 | 70.96 | 9.78 / 16.45 / 14.65 |
| DRO-Grasp (pretrain, no controller) | ✗ | 78.30 | 75.80 | 63.30 | 72.47 | 0.88 / 0.42 / 1.72 |
| **GOAG (no force-closure)** | ✓ | 86.30 | 91.20 | 74.70 | **84.07** | 0.09 / 0.13 / 0.15 |
| **GOAG (full)** | ✓ | 87.40 | 93.20 | 77.90 | **86.93** | 0.18 / 0.19 / 0.20 |

**The one real ablation — dropping force closure — costs 2.86 points (86.93 → 84.07) and halves the time.** The authors read this as the headline: even without any explicit stability test, the CVAE already generates mostly-stable contact patterns, meaning grasp mechanics were learned *implicitly* from taxonomy-structured gripper geometry alone. That is the strongest evidence the object-agnostic premise holds. It also means force closure here is a cheap rejection filter, not a load-bearing component.

**Speed.** DRO-Grasp looks competitive at 0.42 s on Allegro, but it optimises each grasp independently, so cost is linear in grasp count. GOAG minimises one energy function over all candidates in a vectorised batch — higher fixed overhead, far cheaper marginally. The reported number is for 100 grasps amortised. The network is also smaller in parameter count than DRO-Grasp's.

**Diversity** (std dev of successful grasps): GOAG T = 0.0479 m, R = 1.401 rad, Q = 0.3170 rad — the *lowest* translational diversity of any method, and below DFC on all three. The authors' honest framing: because the model is gripper-centric, pose diversity is a property of *how you sample object poses*, not of the network. Only $Q$-diversity comes from the latent.

**Generalisation across 5 datasets, 3,438 objects, Shadow Hand only.** All competitors retrained per dataset; GOAG trained once, ever.

| Method | DexGraspNet | UniDexGrasp | MultiDex | RealDex | DexGRAB | Avg |
|---|---|---|---|---|---|---|
| UniDexGrasp | 33.9 | 23.7 | 21.6 | 27.1 | 20.8 | 25.42 |
| GraspTTA | 18.6 | 21.0 | 30.3 | 13.3 | 14.4 | 19.52 |
| SceneDiffuser | 26.6 | 28.3 | 69.8 | 21.7 | 39.1 | 37.10 |
| UGG | 46.9 | 46.0 | 55.3 | 32.7 | 42.7 | 44.72 |
| DexGrasp Anything | **57.5** | **53.1** | **79.1** | **44.8** | 57.9 | **58.48** |
| GOAG | 43.07 | 49.51 | 77.90 | 37.37 | **62.13** | 53.97 |

Second place overall, first on DexGRAB (human-hand grasps retargeted). It loses clearly on DexGraspNet (43.07 vs 57.5) and RealDex (37.37 vs 44.8).

**What did not work / what the authors admit failed:** the convex-hull pose sampling strategy breaks on **large objects** — it assumes the object volume fits inside the hand's workspace. DexGraspNet and RealDex contain bigger objects, and that is exactly where the gap to DexGrasp Anything opens. The failure is in the pose *proposal*, not the contact model; a better sampler would likely close much of it.

**Real robot:** Allegro Left Hand on a 7-DoF arm, 11 YCB objects grasped successfully. No failure-rate number is reported for the physical trials.

## Worth Remembering

- **The intrinsic domain shift is unaddressed and slightly alarming.** The CVAE is trained on distance fields of *hand surfaces* and tested on distance fields of *arbitrary objects*. Those are very different geometry distributions in the same $\mathbb{R}^{8192}$ space. The paper acknowledges this in one sentence and punts to a companion paper (CoToGrasp, ECCV 2026). That it works at all is the surprising result; *why* it works is not explained here.

- **Force closure is computed in the gripper frame with no gravity and no object mass.** It is a purely geometric feasibility check. A contact set that passes may still be kinematically unreachable once the full object mesh and joint limits are involved — which is exactly why Isaac Gym validation is still required.

- **Data arithmetic doesn't quite close.** 10,000 configurations × 50 contact-point sets = 500,000, but they report 3,000,000 samples. The 6 grasp types presumably multiply in somewhere. Worth checking the code if you replicate.

- **The optimiser weights tell you the priorities.** $\lambda_p = 50$ against $\lambda_d = 1$: penetration is punished 50× harder than missing the target. Sensible — a grasp that misses is a failure, a grasp that clips through the object is physically nonsense.

- **Practical upside if you have a new gripper.** The retraining cost for a novel hand is one GPU-hour of data plus 100 epochs, with *zero* grasp annotation. Compare to building an object-grasp database per embodiment. This is the strongest argument for the approach, more than the raw success rates.

- **It uniquely supports palm full-pose conditioning** (Table I) — you can impose $[R,T]$ from task or environment constraints and only generate grasps that are actually feasible there. Methods that learn pose end-to-end cannot do this cleanly.

- The taxonomy prior is doing quiet, heavy lifting. It is the only place human grasp knowledge enters, and it is what stops the model from learning arbitrary uncoordinated contacts. Nobody ablated removing it, which is the ablation I most want to see.

## Links

Related: [[Auto-Encoding Variational Bayes (VAE)]] · [[KL Divergence]] · [[EXIMO- VLM Guided Exploration of VLA Policies]] · [[τ_0-VLA- a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation]] · [[Denoising Diffusion Probabilistic Models]] · [[Shortcut Learning in Deep Neural Networks]] · [[Understanding Deep Learning Requires Rethinking Generalization]] · [[Regularization]] · [[Loss, Objectives, and Business Alignment]]

New topics worth writing: Basis Point Sets for point cloud encoding, PointNet++ and hierarchical point set learning, Force closure and grasp wrench space, Signed distance fields, Conditional VAE vs unconditional VAE, Grasp taxonomies (Cutkosky / Feix), Isaac Gym and GPU physics simulation, Inverse kinematics as energy minimisation, Cross-embodiment robot learning
