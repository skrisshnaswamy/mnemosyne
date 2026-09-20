---
title: "RoboTok: An Internet-Scale Data Engine for Human Demonstration Retrieval and Dexterous Manipulation Learning"
authors: ["Howard Qian", "Yiting Chen", "Yunfei Xie", "Kejia Ren", "Podshara Chanrungmaneekul", "Gaotian Wang", "Bowen Wen", "Chen Wei", "Kaiyu Hang"]
year: 2026
arxiv: "2609.03199"
url: https://arxiv.org/abs/2609.03199
priority: Good-To-Read
read_on: 2026-09-16
tags: [paper, vision]
---
## The Core Idea

Robot demonstration data is expensive. The internet already has billions of videos of humans manipulating things. The blocker is not collection, it is **search**: given a video of the motion you care about, which of the millions of web clips show the *same kind of motion*?

The usual answer — match on what the video looks like, or on its caption — is wrong for manipulation. Two clips of a kitchen can contain totally different actions. Two clips shot from different angles, in different rooms, with different objects, can contain the same wrist-and-finger motion. Appearance similarity and behaviour similarity are different things.

RoboTok's bet: throw away the pixels entirely and index clips by **the 3D path the hands took, expressed in a coordinate frame attached to the demonstrator's torso**. Camera angle, lighting, background, and whether the person's body is even in shot all stop mattering, because the hand trajectory is re-expressed relative to the person, not relative to the camera.

Two pieces make this work:

1. A **torso-frame estimator that only needs the hands**. Most manipulation footage is a close-up — you see hands and an object, no body. They train a small model that takes only the wrist poses over time and predicts where the demonstrator's torso must be. That is what lets them canonicalise clips where the actor is occluded.
2. **Dynamic Time Warping as a teacher, not as the search method.** DTW gives a good, kinematically meaningful notion of "these two hand trajectories are the same move at different speeds", but comparing a query against 100,000 clips with DTW is hopeless at web scale. So DTW is used offline to generate training labels for a small encoder, and at query time you do plain cosine nearest-neighbour search over precomputed vectors. New clips join the index with one forward pass — no retraining.

> [!NOTE] Dynamic Time Warping (DTW)
> A way to measure distance between two sequences that may run at different speeds. It finds the best "stretching" alignment between them and sums the per-pair distances along that alignment. Costs $O(L_i L_j)$ per pair, which is why it cannot be the online search operator. ^dynamic-time-warping

> [!NOTE] Actor-relative (egocentric) representation
> Expressing motion in a frame glued to the demonstrator's torso rather than to the camera. Makes the same physical action produce the same numbers regardless of where the camera was standing. ^actor-relative-frame

The unlock is that web video becomes a *continuously growing* demonstration source instead of a fixed dataset you have to curate by hand.

## The Methodology

**Problem setup.** Given a query clip $q$ and a corpus $\mathcal{D} = \{x_1, \dots, x_N\}$, return the $K$ clips with the most similar hand motion. Ground-truth similarity is defined by DTW over 21-joint hand poses:

$$\mathrm{DTW}(x_i, x_j) = \min_{\pi \in \Pi} \sum_{(t,u) \in \pi} \|x_i^t - x_j^u\|_2$$

$$s(i,j) = -\frac{\mathrm{DTW}(x_i, x_j)}{\tfrac{1}{2}(L_i + L_j)}$$

The length normalisation stops long clips from being penalised for being long. The goal is an encoder $\Gamma: \mathcal{X} \to \mathbb{S}^{d-1}$ (onto the unit sphere, so inner product = cosine) such that the ranking under $\langle \Gamma(x_i), \Gamma(x_j)\rangle$ matches the ranking under $s$.

**Data pipeline** (this is most of the paper's engineering):

- Source: Action100M, a large web video corpus already pre-filtered for human actions.
- Clip filter: 4–8 seconds long, **near-static camera** (Lucas–Kanade optical flow check), at most one left and one right hand visible. Overlapping segments greedily resolved in favour of the longer clip.
- Hand pose: WiLoR at 5 fps, with handedness linked across frames.
- Metric grounding: WiLoR uses a weak-perspective camera, so poses have no real scale. MoGe-2 supplies metric depth, converting hand poses into metres in the camera frame.
- Gap-filling: HaWoR infills missing frames.
- Canonicalisation: the torso-frame estimator (trained following EgoInfinity, adapted to the SMPL-H body model) predicts a static torso frame from the wrist trajectories alone; all poses are re-expressed in it.

Final corpus: $N = 100{,}000$ clips, with $10{,}000$ held out as evaluation queries.

**Encoder.** Deliberately tiny. Each frame's pose is positionally encoded, then a lightweight cross-attention block pools the sequence into one $\ell_2$-normalised $d$-dim vector. The argument for keeping it small: the input is *already* canonicalised 3D motion, so the encoder does not need to learn viewpoint or appearance invariance. Its only job is to compress the DTW neighbourhood structure.

**Batch construction** (the part that actually matters). With $N = 100{,}000$ and batch $b = 196$, a random batch almost never contains a true DTW neighbour, so a naive [[A Simple Framework for Contrastive Learning (SimCLR)|contrastive]] batch would have no signal. Instead, batches are built from **anchor-centred groups**. For anchor $a$, the relevant set is its top $K = 20$ DTW neighbours $\mathcal{R}(a)$. Each group is
$$\mathcal{G}(a) = \{a,\ p_1,\ p_2,\ n\}$$
with $p_1, p_2$ sampled from $\mathcal{R}(a)$ and $n$ sampled *immediately outside* the top-20 — a **boundary negative**, hard by construction, sitting right on the relevance threshold. 49 groups × 4 = 196 trajectories per batch. Same spirit as [[Approximate Nearest Neighbor Negative Contrastive Learning (ANCE)|ANCE]]-style hard negative mining: random negatives are free but useless.

**Loss.**
$$\mathcal{L} = \mathcal{L}_{\text{set}} + \lambda \mathcal{L}_{\text{rank}}$$
The *set* term pushes the oracle's top-20 above the boundary and random negatives — it decides **who gets into** the retrieved neighbourhood. The *rank* term forces the ordering of the sampled positives to match the DTW ordering — it decides **the order within** the neighbourhood. Drawn from the deep-metric-learning-to-rank line (Wang et al. 2019, Cakir et al. 2019).

**Inference.** Offline: encode every clip once, store in an inner-product index. Online: encode the query, cosine top-$k$. All DTW cost is paid at training time.

**Downstream policy use.** Retrieved clips are turned into robot supervision *only through the reward* — no imitation loss, no action labels. Define $\Phi(s)$ as the negative weighted $k$-NN distance from the current robot hand state to the bank of retargeted demonstration states. Two reward terms are added to the task reward:

- a **standing term** $\beta\Phi(s)$, pulling the policy toward the demonstration manifold;
- a **potential-based shaping term** $w(\gamma\Phi(s') - \Phi(s))$, which by Ng et al. 1999 provably does not change the optimal policy of the underlying task.

Guidance is assigned per parallel environment: environment $i$ is paired with retrieved clip $i \bmod K$ for all of training, so a $K$-clip bank gives $K$ distinct targets in parallel rather than one averaged blur. Policies are [[Proximal Policy Optimization Algorithms|PPO]] from scratch, input = proprioception + fingertip forces, 3-step rolling observation window.

## Ablation Studies and Experiments

**Retrieval, RoboTok corpus** (10,000 held-out queries against ~90,000 candidates, relevant set = top-20 by DTW):

| Method | mAP@20 ↑ | nDCG@20 ↑ | Kendall $\tau$ ↑ | DTW cost@20 ↓ |
|---|---|---|---|---|
| Random | 0.0000 | 0.0001 | 0.0000 | 4.776 m |
| FlowRetrieval | 0.0004 | 0.0017 | 0.0031 | 6.612 m |
| HAND | 0.0009 | 0.0040 | 0.0118 | 4.528 m |
| STRAP | 0.0071 | 0.0257 | 0.0137 | 4.044 m |
| **RoboTok** | **0.3531** | **0.5836** | **0.4867** | **1.333 m** |
| Oracle | 1.000 | 1.000 | 1.000 | 1.145 m |

Recall@20 is 0.996 for RoboTok against 0.12 for STRAP. FlowRetrieval is *worse than random* on DTW cost (6.61 vs 4.78) — optical flow similarity actively anti-correlates with hand-motion similarity here.

**Cross-dataset transfer, AssemblyHands** (831 two-hand assembly clips, sensor-grade 3D hand labels, relevant set = top-5): ordering unchanged. RoboTok mAP@5 = 0.261 vs STRAP 0.133; DTW cost 1.095 m vs oracle 0.966 m (13% above optimal) vs random 1.911 m. Margins narrow, which is the honest signal — some of the home-corpus gap is home-field advantage.

**Generalisation beyond the training neighbourhood.** The encoder was trained only to reproduce the top-20 DTW neighbourhood, but the Recall@$k$ / CKNNA@$k$ curves stay strong out to $k = 500$. The embedding organised globally without being asked to.

**Downstream, original VTDexManip** (6 dexterous sim tasks): RoboTok-guided PPO beats the benchmark's best pretrained baseline (VT-JointPretrain) on 5 of 6 tasks, averaging +7.45% on seen objects and +5.83% on unseen. Example: Lever Sliding unseen goes 2.2% (vanilla PPO) → 92.0%. The authors call this formulation saturated — even *randomly* retrieved demos take Lever Sliding from 2.2% to 57.3%, so the benchmark mostly rewards "having any hand-motion prior at all".

**Downstream, harder VTDexManip** (they restore full 3D wrist motion and delete the hand-designed dense reward shaping, so exploration is genuinely hard; 8 seeds, 100 attempts per object):

| Task / split | Base | Random | Flow | HAND | STRAP | RoboTok |
|---|---|---|---|---|---|---|
| BottleCap, seen | 0.1 | 0.2 | 1.9 | 59.5 | 59.0 | **77.3** |
| Faucet, seen | 1.0 | 5.3 | 0.0 | 6.8 | 0.0 | **44.8** |
| Faucet, unseen | 1.7 | 1.0 | 0.3 | 2.6 | 0.0 | **10.9** |
| Lever, seen | 0.8 | 17.2 | 14.4 | 19.5 | 8.4 | **79.3** |

This is the experiment that carries the paper. Once the dense reward crutch is removed, retrieval quality becomes the difference between 0% and 45%. STRAP scores **0.0** on Faucet Screwing — guidance from the wrong demos is worse than no guidance, because the shaping term actively pulls the policy toward a manifold that does not solve the task.

**What did not work:**
- **Bimanual Hand-over, unseen objects**: RoboTok 34.8% loses to *Random* at 42.1%. Two-hand coordination is where trajectory-only matching is weakest, and generalising a retrieved two-hand motion to a new object shape fails.
- **Table Reorientation**: VT-JointPretrain (85.0 seen / 84.6 unseen) beats RoboTok (82.6 / 76.2). Visual-tactile pretraining still wins on tasks whose success depends on contact detail rather than gross wrist path.
- Optical-flow retrieval (FlowRetrieval) is a dead end for this: image-plane motion conflates camera motion, object motion, and hand motion.

## Worth Remembering

- **The evaluation has a structural circularity.** The "ground truth" is DTW over canonicalised 3D hand trajectories — exactly the objective RoboTok's encoder was distilled from. The baselines (flow-based, feature-based, 2D-hand-path-based) were never optimising this target, so a 50× mAP gap is partly the gap between "was trained on this metric" and "was not". The AssemblyHands transfer and, much more so, the harder VTDexManip policy results are the load-bearing evidence, because policy success is an external judge.
- **The near-static camera filter is a big hidden restriction.** Every clip in the index must have a roughly still camera, because the whole metric-3D reconstruction chain assumes it. That rules out most egocentric footage and most edited web video. The authors flag moving-camera support as future work. "Internet-scale" is really "the static-camera slice of internet-scale".
- **Objects are invisible to the representation.** Retrieval sees 21 hand joints per frame and nothing else. A motion of turning a bottle cap and a motion of turning a dial with the same kinematics are neighbours. For reward shaping this is fine — you want the motion prior. For anything object-conditioned it is a real gap.
- **The pipeline is a stack of five off-the-shelf models** (WiLoR → MoGe-2 → HaWoR → torso estimator → encoder) and errors compound. Metric depth from a single image is the shakiest link, and it sets the scale of every DTW distance in metres.
- **Cheap continual indexing is the actual product claim.** Adding a clip is one forward pass through a tiny encoder plus one vector insert — the same economics as any [[Sentence-BERT|bi-encoder]] retrieval system, with [[Efficient and robust approximate nearest neighbor search using HNSW|ANN]] search underneath. That is why they call it a "data engine" rather than a dataset.
- **Building the training labels is the expensive part nobody costs out.** You need top-20 DTW neighbours for 100,000 variable-length trajectories. Done naively that is $5 \times 10^9$ DTW computations. The paper does not say how they made this tractable.
- **Potential-based shaping is the right choice and worth internalising.** Because $w(\gamma\Phi(s') - \Phi(s))$ leaves the optimal policy of the base [[Markov Decision Process|MDP]] unchanged, bad retrievals can slow learning but cannot, in the limit, make the agent optimise the wrong thing. The Faucet-Screwing 0.0% for STRAP shows "in the limit" is doing heavy lifting in practice.
- Open question: the DTW oracle is a *choice*, not a truth. Euclidean distance over 21 joints weights every finger equally and ignores contact forces and object state. A contact-aware or wrench-aware similarity (cf. the CHORD line of work) might define a different, better neighbourhood.

## Links

Related: [[Proximal Policy Optimization Algorithms]] · [[Markov Decision Process]] · [[Efficient and robust approximate nearest neighbor search using HNSW]] · [[Approximate Nearest Neighbor Negative Contrastive Learning (ANCE)]] · [[Dense Passage Retrieval (DPR)]] · [[Sentence-BERT]] · [[A Simple Framework for Contrastive Learning (SimCLR)]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[NDCG]] · [[GOAG- Generative and Object-Agnostic Grasp Planner for Dexterous Robotic Manipulation]] · [[EXIMO- VLM Guided Exploration of VLA Policies]] · [[GigaBrain-0.7- Scaling Embodied Foundation Models to Emergent Capabilities with a Three-System Architecture]] · [[Zero-WAM- In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization]] · [[Latent Action as Intention Enables Efficient Future Imagination for World Action Models]] · [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)]]

New topics worth writing: Dynamic Time Warping, potential-based reward shaping, deep metric learning to rank (set + rank losses), SMPL-H and parametric body models, monocular metric depth estimation, learning-to-rank distillation from an expensive oracle, cross-embodiment retargeting, CKNNA as a representation-alignment metric
