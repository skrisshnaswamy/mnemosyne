---
title: "TacForcing: Streaming Action Generation with Execution-Time Tactile Feedback"
authors: ["Jianbo Zhou", "Boyuan Zhao", "Yuzheng Zhang", "Yiyang Chen", "Wenxin Chen", "Qiuyue Li", "Xiangyang Gu", "Yuhan Cao", "Xiao Xia", "Yanzhe Hu", "Zhijie Deng"]
year: 2026
arxiv: "2608.25798"
url: https://arxiv.org/abs/2608.25798
priority: Good-To-Read
read_on: 2026-09-05
tags: [paper, transformers, diffusion, vision]
---
## The Core Idea

A robot policy that predicts a whole chunk of future actions at once is blind to what happens *while* it executes them. That is fine for vision — the camera image barely changes in one second. It is fatal for touch. The authors measured this on a dropper-squeezing episode: over a 40-action horizon (35 control steps ≈ 1.17 s), the cosine distance of the visual features from their starting value grew to about $0.005$, while the tactile features moved $0.55$. Touch changes a hundred times faster than sight. So a policy that conditions all 40 actions on one tactile reading taken before execution is using a stale signal for 39 of them.

The usual fix is to bolt on a second, fast controller that runs at high frequency and reacts to force — a "slow-fast" hierarchy (RDP does this). That means two policies, two training pipelines, two sets of failure modes.

TacForcing removes the second controller. The trick is to make the action generator itself *streaming*. Instead of denoising the whole chunk to completion and then executing, it splits the chunk into blocks and gives each block its own position in the denoising schedule. Block 1 finishes first and gets sent to the robot. Blocks 2…K are still half-noise, held in memory. The robot executes block 1, a fresh tactile reading arrives, and generation resumes from the retained partial states — now conditioned on the new touch. Generation and execution interleave.

> [!NOTE] Streaming action expert
> An action-chunk generator where different time positions in the chunk sit at different points along the flow-matching path, so early actions can be finalised and executed while later ones are still being refined. ^streaming-action-expert

The second idea is smaller but does most of the work in the ablations. Once you have fresh touch, the naive thing is to let every unfinished action attend to it. That is wrong for the same reason the original problem was wrong: block 7 will not be executed for another second, and by then this tactile reading will be stale too. **Execution-Aware Tactile Attention (EATA)** masks tactile attention so that the newest tactile tokens condition *only the block that executes next*. Later blocks wait for their own, fresher reading.

> [!NOTE] Execution-Aware Tactile Attention
> An additive attention mask that lets tactile tokens acquired at stage $k$ be visible only to action tokens belonging to block $k$, aligning each sensor reading with the actions it is temporally valid for. ^eata

## The Methodology

**Setup.** At decision step $t$ the robot has a visual observation $V_t$, proprioception $s_t$, tactile deformation maps $T_t$ from $M$ fingertips, and a language instruction $\ell$. Task context $c_t = (V_t, s_t, \ell)$. The policy outputs a chunk $A_t = (a_t, \dots, a_{t+H-1})$.

**Base generative model — flow matching.** Standard [[Score-Based Generative Modeling through SDEs|continuous-time]] flow matching: interpolate between noise and data,

$$x^\tau = (1-\tau)x_0 + \tau x_1, \qquad u^\star = x_1 - x_0,$$

$$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}\left[\|v_\theta(x^\tau, \tau, y) - u^\star\|_2^2\right].$$

In a normal action expert, all $H$ positions share one $\tau$, so the whole chunk is born at the same instant.

**Block-wise flow scheduling.** Replace the single flow time with a vector $\bm{\tau}^{(n)} = (\tau_1^{(n)},\dots,\tau_H^{(n)})$. Assume $H = KB$ and $N = KS$: $K$ blocks of $B$ actions, $S$ sampling steps between consecutive block completions. Position $i$ belongs to block $b(i) = \lfloor (i-1)/B \rfloor + 1$. Block $k$ finishes at sampling step $n_k = kS$, and its flow time follows

$$\lambda_k^{(n)} = \min\!\left(\frac{n}{n_k},\, 1\right).$$

So at any step $n$, block 1 is furthest along, block $K$ is closest to noise. Since $n_1 < n_2 < \dots < n_K$, blocks become executable in order. Completed blocks freeze; unfinished ones keep advancing.

Why blocks and not per-action scheduling? Per-action would let you refresh touch every control step, but you would pay for tactile acquisition, encoding, and a model forward pass at every step. Blocks are the compute/responsiveness knob. They use $B = 5$ everywhere.

**Conditioning.** The VLM encodes $c_t$ once into $C_t = f_{\mathrm{ctx}}(c_t)$ and reuses it for the whole chunk — vision is slow, so no need to redo it. Tactile is refreshed per block: after $k$ blocks execute, $T_t^{(k)}$ is read, each of the $M$ fingertip deformation maps is passed through a shared encoder $f_{\mathrm{tac}}$, giving tokens $Z_t^{(k)} = (z_{t,1}^{(k)},\dots,z_{t,M}^{(k)})$. During steps $(k-1)S < n \le kS$, $Z_t^{(k-1)}$ is the freshest tactile state.

**The EATA mask.** With $i$ indexing action queries and $m$ indexing tactile keys:

$$\mathcal{M}_{i,m}^{(k)} = \begin{cases} 0, & b(i) = k, \\ -\infty, & \text{otherwise.} \end{cases}$$

Additive, applied before softmax — the standard masking trick from [[Causal Attention|causal attention]]. Crucially, the same mask is applied at *training* time, so the model never learns to depend on visibility it will not have at inference.

**Training.** Sample noise $\epsilon \sim \mathcal{N}(0,I)$ and a normalised generation progress $p \sim \mathcal{U}([0,1))$, which stands in for $n = pN$. Block $k$'s flow time and interpolated state:

$$\lambda_k(p) = \min\!\left(\frac{Kp}{k}, 1\right), \qquad \widetilde{A}_t^{(k)} = (1-\lambda_k(p))\,\epsilon^{(k)} + \lambda_k(p)\,A_t^{(k)}.$$

The block about to execute is $k^\star(p) = \lfloor Kp \rfloor + 1$; use tactile $Z_t^{(k^\star - 1)}$ and mask $\mathcal{M}^{(k^\star)}$. Let $\mathcal{U}(p)$ be positions with $\lambda_{b(i)}(p) < 1$ (not yet finished). The loss is velocity regression on unfinished positions only:

$$\mathcal{L}_{\mathrm{train}} = \mathbb{E}_{A_t,\epsilon,p}\left[\frac{1}{|\mathcal{U}(p)|}\sum_{i\in\mathcal{U}(p)} \left\|\widehat{v}_{\theta,i}(p) - (a_{t+i-1} - \epsilon_i)\right\|_2^2\right].$$

Masking out finished positions matters: at inference those positions are frozen and never updated, so training on them would be a train/test mismatch.

**Initialisation and scale.** Not trained from scratch. In simulation, initialised from $\pi_{0.5}$; in the real world, from GR00T N1.7. The tactile encoder is initialised from FTP-1's pretrained tactile encoder in both cases.

**Hyperparameters.** Sim: $H=50$, $K=10$, $B=5$; 50 demos per task, 15k steps, batch 256, [[Decoupled Weight Decay Regularization (AdamW)|AdamW]] at peak LR $5\times10^{-5}$ cosine-decayed to $5\times10^{-6}$, 2k warmup, weight decay $10^{-10}$, grad clip 1.0. Real: $H=40$, $K=8$, $B=5$; 100 demos per task, 30k steps, LR $6\times10^{-5} \to 1\times10^{-6}$, weight decay $10^{-5}$.

**Real hardware.** Two 7-DoF RealMan RM75 arms, 22-DoF Sharpa Wave dexterous hands with fingertip tactile sensors giving deformation maps, top + wrist RealSense cameras, Manus Pro data gloves for demo collection.

## Ablation Studies and Experiments

**Benchmarks.** Six UniVTAC simulation tasks (Lift Bottle, Pull-out Key, Lift Can, Put Bottle in Shelf, Insert Hole, Insert Tube), 100 rollouts each. Three real tasks (Stand Bottle, Transfer Liquid, Wipe Board), 16 trials each.

**Baselines span the design space:** $\pi_{0.5}$ and GR00T N1.7 (vision only), UniVTAC-ACT (ACT + pretrained tactile encoder), FTP-1 (unified tactile tokens, shared tactile expert), RDP (slow-fast hierarchy with a separate high-frequency reactive controller).

**Simulation, success rate %:**

| Method | Lift Bottle | Pull-out Key | Lift Can | Bottle in Shelf | Insert Hole | Insert Tube | Avg |
|---|---|---|---|---|---|---|---|
| $\pi_{0.5}$ | 88 | 43 | 46 | 43 | 39 | 48 | 51 |
| UniVTAC-ACT | 59 | 41 | 24 | 6 | 36 | 58 | 37 |
| RDP | 84 | 18 | 12 | 41 | 23 | 75 | 42 |
| FTP-1 | 89 | 35 | 66 | 23 | 62 | 76 | 59 |
| **TacForcing** | **90** | **48** | 63 | 43 | **69** | **79** | **65** |

Two things stand out. First, TacForcing beats the vision-only $\pi_{0.5}$ by 14 points and the tactile-reactive RDP by 23. Second — and this is uncomfortable for the field — **RDP, the slow-fast reactive baseline, scores 42, *below* the vision-only $\pi_{0.5}$ at 51.** UniVTAC-ACT at 37 is worse still. Adding touch badly is worse than no touch at all.

**Real world, average success rate %:** TacForcing 69, FTP-1 52, GR00T N1.7 42, $\pi_{0.5}$ 27. The biggest gap is Transfer Liquid: TacForcing 50%, every baseline $\le$ 19%. That is the task with heavy visual occlusion (transparent dropper, liquid) and continuous force regulation — exactly where stale touch should hurt most.

**The four-way ablation** (three sim tasks, three real tasks) is the important table:

| Config | Sim avg | Real avg |
|---|---|---|
| Base (no tactile) | 43 | 42 |
| Fixed Tactile (one reading, whole chunk) | 42 | 31 |
| TacForcing w/o EATA (streaming + refresh) | 51 | 48 |
| **TacForcing (+ EATA)** | **60** | **69** |

**What did not work: naive tactile conditioning.** Adding a single pre-execution tactile observation to the chunk-based policy made things *worse* — 43 → 42 in sim, and a brutal 42 → 31 in the real world. Task-level: it helped only on Lift Can, and hurt on Pull-out Key, Insert Hole, and all three real tasks. The stale tactile signal is not neutral noise; the policy actively trusts it and gets misled. This is the paper's cleanest result, and it retroactively explains why RDP and UniVTAC-ACT underperform the vision-only baseline.

**Streaming alone recovers and then some**: +9 sim, +17 real over Fixed Tactile. So refreshing the signal is where the recovery comes from.

**EATA is not a garnish.** It adds +9 sim and +21 real on top of streaming, improving all six evaluated tasks. In the real world it is *larger* than the streaming gain itself. The interpretation: giving fresh touch to actions that will not run for another second is nearly as harmful as giving stale touch to everything. Restricting who can see the sensor reading matters as much as refreshing it.

Note the honest arithmetic: in the real world, streaming-without-EATA (48) is still above Base (42) but only by 6 points, whereas full TacForcing is +27. Most of the real-world win is EATA.

## Worth Remembering

- **The measurement is the contribution as much as the method.** $0.005$ visual drift vs $0.55$ tactile drift over 1.17 s is the single number to carry away. Any modality whose representation moves that fast inside one action horizon cannot be handled by chunk-level conditioning. The same argument would apply to force/torque, audio contact events, or slip detection.

- **Everything is initialised from a big pretrained VLA.** TacForcing is a surgery on $\pi_{0.5}$ / GR00T N1.7 — the standard action expert is swapped for the streaming one. There is no from-scratch result, so it is unknown whether block-wise scheduling helps a small policy or only unlocks something in an already-strong backbone.

- **Cost is not reported.** Streaming means $K$ forward passes per chunk instead of one full denoising loop, plus $K$ tactile encodings. With $K=8$–$10$ that is a real latency budget. The paper says block-level updates "balance responsiveness and computational efficiency" but gives no wall-clock or control-frequency numbers, and no comparison of inference cost against RDP's separate fast controller. For anyone deploying, this is the first thing to measure.

- **$B=5$ is never swept.** There is no ablation on block size or on $K$, which is the central design knob. $B=1$ (refresh every action) and $B=H$ (degenerate to Fixed Tactile) are the two endpoints, and we only see the latter.

- **Total sampling steps $N = KS$ couples generation quality to block count.** You cannot change how many blocks you have without changing how many denoising steps each block gets. That coupling is not discussed.

- **The design generalises past touch.** Nothing in EATA is tactile-specific. It is a rule: *a sensor reading may only condition the actions that will execute before the next reading*. That is a general principle for interleaving perception and generation, and it plugs into any streaming-diffusion policy.

- **Small evaluation.** 16 real trials per task means each success is worth 6.25 percentage points. The Transfer Liquid result (50% vs 19%) is a gap of about 5 trials. Treat real-world differences under ~12 points as noise.

- **RDP scoring below vision-only in simulation** deserves a follow-up. Is it a training/implementation issue, a sim-tactile-fidelity issue, or a genuine statement that slow-fast hierarchies are fragile? The paper trained RDP with official code but does not investigate.

## Links

Related: [[GameWAM- A World Action Model for Video Games]] · [[Hydra-0- Action Flow for Generalist World Modeling and Control]] · [[GigaBrain-0.7- Scaling Embodied Foundation Models to Emergent Capabilities with a Three-System Architecture]] · [[Latent Action as Intention Enables Efficient Future Imagination for World Action Models]] · [[EXIMO- VLM Guided Exploration of VLA Policies]] · [[τ_0-VLA- a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation]] · [[Zero-WAM- In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization]] · [[Score-Based Generative Modeling through SDEs]] · [[Denoising Diffusion Probabilistic Models]] · [[Causal Attention]] · [[Attention Is All You Need]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[GOAG- Generative and Object-Agnostic Grasp Planner for Dexterous Robotic Manipulation]]

New topics worth writing: Flow Matching, Diffusion Forcing, Action Chunking Transformer (ACT), Reactive Diffusion Policy (slow-fast tactile control), visuo-tactile representation learning, GelSight-style deformation-map tactile sensors, receding-horizon control, UniVTAC benchmark
