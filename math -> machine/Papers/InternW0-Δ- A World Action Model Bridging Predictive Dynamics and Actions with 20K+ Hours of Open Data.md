---
title: "InternW0-Δ: A World Action Model Bridging Predictive Dynamics and Actions with 20K+ Hours of Open Data"
authors: ["Xingyu Miao", "Zizun Li", "Baole Fang", "Kaiwen Song", "Tenghui Wang", "Hanxue Zhang", "Yating Wang", "Xudong Li", "Yuping He", "Xueyuan Wei", "Chao Gao", "Xijie Yang", "Yingxiang Xu", "Kerui Ren", "Wenqi Guo", "Jianjun Zhou", "Xinzhe Wang", "Weiguang Zhao", "Ni Yang", "Zetao Cai"]
year: 2026
arxiv: "2609.31394"
url: https://arxiv.org/abs/2609.31394
priority: Good-To-Read
read_on: 2026-09-29
tags: [paper, transformers, diffusion, vision]
---
## The Core Idea

A **World Action Model** (WAM) is a robot policy that is trained to do two things at once: predict what the camera will see next, and predict what the robot should do next. The hope is that a big pretrained video model already "knows" how objects move, fall, and get pushed around, and that this knowledge transfers into better motor commands.

The problem is that being able to *draw* the future is not the same as knowing *what to do*. And if a policy has to actually generate a future video before it can act, it is far too slow for a 30 Hz control loop.

InternW0-$\Delta$'s central trick fixes exactly that gap. It is called **Causal Imprint**.

> [!NOTE] Causal Imprint
> A small set of learnable tokens that sit inside the video model and are *trained* to encode how the scene is about to change. Supervision comes from real future frames, but the tokens themselves are never allowed to look at those frames. At test time they still produce a "what's about to happen" summary from present observations alone. ^causal-imprint

The mechanism is an attention mask, not a new architecture. Future-frame tokens may read the present. The Causal Imprint tokens and the action tokens may **not** read the future. So future information flows in only as a training target — there is no forward activation path from a realised future frame to an action. Result: at inference you run the video half once, cache it, and denoise actions. No video sampling, no VAE decoding.

Two ideas make this a *directed* design rather than just a mask:

1. Future frames become a **loss**, not an **input**. The paper calls this separating future supervision from the inference path.
2. Geometry knowledge arrives the same way — a frozen 3D-tracking teacher is distilled into the video expert during training and thrown away afterwards.

Why did this not exist before? Earlier WAMs either generated future video at test time (slow) or dropped the future-prediction loss entirely (losing the video prior). The precedent here is Fast-WAM, which first showed you can keep future-video supervision while cutting it out of inference. InternW0-$\Delta$ pushes further: it does not just remove future rollout, it builds a *named representation* whose whole job is to carry the predictive signal across into the action stream.

What it unlocks, concretely: 92.8% on LIBERO-Plus (perturbed LIBERO) versus 84.8% for the best prior WAM, and 152.8 ms round-trip latency on a single RTX 5090, which is fast enough to actually run a dexterous hand.

## The Methodology

### The two experts

Think of two transformers running side by side, 30 blocks deep, sharing attention but not weights. This is a **Mixture-of-Transformers** — the same idea as [[Mixture of Experts]], except the routing is by *modality* (video vs. action), not learned per token.

- **Video expert**: initialised from Wan2.2-TI2V-5B, a pretrained text-and-image-to-video [[Diffusion Models|diffusion]] transformer. 3072-dim stream.
- **Action expert**: "ActionDiT", randomly initialised. 1024-dim stream.

At every block, both compute their own $Q, K, V$, the six matrices get concatenated, one masked [[Attention|attention]] call runs, then the outputs split back to their own streams for separate [[Cross Attention|cross-attention]] and feed-forward layers:

$$Y=\operatorname{Attn}\!\left(\begin{bmatrix}Q^{v}\\ Q^{a}\end{bmatrix},\begin{bmatrix}K^{v}\\ K^{a}\end{bmatrix},\begin{bmatrix}V^{v}\\ V^{a}\end{bmatrix}; M\right),\qquad (Y^v, Y^a)=\operatorname{Split}(Y)$$

The mask $M$ is the whole design. Token groups: **A** (anchor frame), **R** (recent frame), **C** (current frame), **F** (future frames), **$\Delta$** (Causal Imprint), **Act** (action). F may attend to A/R/C. $\Delta$ and Act may **not** attend to F.

### Two language paths, on purpose

- **T5** ([[Exploring the Limits of Transfer Learning (T5)|T5]]) embeds the instruction and conditions the *video* expert. This preserves the language↔video interface Wan already learned.
- A **frozen** VLM (RynnBrain1.1-2B) sees the instruction *and* all current camera views, and its hidden states condition the *action* expert.

The reasoning: T5 reads words but never sees the room. "Pick up the cup" maps to different actions depending on where the cup is. The VLM supplies that grounding.

### Sparse memory

Instead of a dense frame history, only three composed observations:

$$\mathcal{X}_t=\{x_a,\; x_{t-H_c},\; x_t\}$$

$x_a$ is the first frame of the episode (coarse task context), $x_{t-H_c}$ is the frame before the previous action chunk (recent dynamics), $x_t$ is now. Each is a multi-view canvas — all available cameras resized and tiled into one $384\times256$ image, then encoded by the frozen Wan VAE.

### Causal Imprint, precisely

Two losses, and the ablations show they are not redundant.

**Loss 1 — regress the change.** Take clean video latents *before* noise is added, and form adjacent differences along the future:

$$\Delta z_{t,i}=z_{t+i\rho_v}-z_{t+(i-1)\rho_v},\qquad i=1,\dots,T_v$$

Stack them into $\Delta Z_t$. The final-layer Causal Imprint features regress it:

$$\mathcal{L}_{\Delta}=\|h_t^{\Delta}-\Delta Z_t\|_2^2$$

**Loss 2 — align to the video model's own future features.** The Causal Imprint token at block 8 is pushed towards the spatially matching feature of the *last* future slice at block 20, with a stop-gradient:

$$\mathcal{L}_{\mathrm{align}}=1-\cos\!\left(h_{b,p}^{\Delta,\ell_s},\;\operatorname{sg}\!\left[h_{b,p}^{F,\ell_t}\right]\right)$$

Loss 1 teaches low-level pixel-latent motion. Loss 2 copies the *semantic* future representation the diffusion transformer builds internally. The second one turns out to matter much more.

This is structurally the same move as [[JEPA]] / [[Self-Supervised Learning from Images with I-JEPA|I-JEPA]]: predict a future *representation*, not future pixels, and use stop-gradient to stop the target collapsing.

### 4D-aware distillation

A frozen Track4World model (a 3D point tracker) processes the ground-truth video window and is pooled into a 1430-dim clip descriptor $r_t^{\mathrm{T}}$ — geometry, 2D/3D motion, camera motion, visibility, each block $\ell_2$-normalised then concatenated. **Cached offline**, so the teacher never runs during policy training.

The student reads clean A/R/C hidden tokens from VideoDiT block 15, runs 16 learnable queries through a 2-layer transformer decoder, mean-pools, projects, and matches:

$$\mathcal{L}_{\mathrm{4D}}=\tfrac{1}{1430}\big\|r_t^{\mathrm{S}}-\operatorname{sg}[r_t^{\mathrm{T}}]\big\|_2^2$$

Gradients flow backwards into the video expert. Teacher and student branch both vanish at inference — so this is pure [[Distillation]] with zero deployment cost, in the spirit of [[Distilling the Knowledge in a Neural Network]] but aligning features rather than logits.

### Objective

Everything is [[Flow Matching]]. For a target $y$: sample $\epsilon\sim\mathcal{N}(0,I)$, $\sigma\in(0,1)$, set $y^\sigma=(1-\sigma)y+\sigma\epsilon$, and regress the velocity:

$$\mathcal{L}_{\mathrm{FM}}(y)=\mathbb{E}_{\epsilon,\sigma}\big[\|f_\theta(y^\sigma,\sigma)-(\epsilon-y)\|_2^2\big]$$

Applied to both future latents and the action chunk. Total:

$$\mathcal{L}=\lambda_v\mathcal{L}_{\mathrm{video}}+\lambda_a\mathcal{L}_{\mathrm{action}}+\lambda_\Delta\mathcal{L}_\Delta+\lambda_{\mathrm{align}}\mathcal{L}_{\mathrm{align}}+\lambda_{\mathrm{4D}}\mathcal{L}_{\mathrm{4D}}$$

with $0.5 / 1.0 / 0.5 / 0.1 / 0.1$. The action weight is **per-data-source**: 1.0 robot, 0.5 UMI, 0.1 Ego2Robot, 0.1 raw ego. They trust human-derived action labels ten times less than robot ones.

### The canonical action space

Every dataset gets remapped into one fixed 80-dim vector with hard-coded semantic slots. Arm joints at `[0,7)` and `[40,47)`, EEF pose at `[7,16)` and `[47,56)`, gripper at `[16,17)`, dexterous hand at `[17,29)`, torso, head, mobile base, plus reserved slots. Dimensions a given robot does not have are masked out of the loss.

Convention that matters: **joint/gripper/hand actions are absolute targets; EEF actions are relative** (3D translation delta + 3D rotation vector in the current EEF frame). State stores 9D absolute EEF pose (position + 6D rotation), action stores 6D relative motion.

### Data

Over 20K hours. Robot: 15 datasets, 13,867 h → **11,302 h / 1.25M episodes / 1.10B frames** after filtering. UMI (Hy-UMI-10K): 2,162 → 2,075 h. Egocentric human (EgoDex + EgoVerse): 4,640 → 4,061 h.

Filtering pipeline, in order:
1. **Signal anomaly + state–action consistency.** After cross-correlation alignment, directional agreement between state deltas and actions must exceed 0.65 per dimension. Episodes where state changes *precede* their actions are rejected (a sign of mislabelled timing).
2. **Static boundary trimming** — cut dead time at episode ends, keep pauses in the middle.
3. **Visual quality** — black frames, Laplacian blur measured *only on Sobel edge pixels* (so a textureless table does not read as blurry), JPEG blockiness at $8\times8$ boundaries.
4. **Action magnitude guard** — drop samples with single-step translation $>0.2$ m or rotation $>0.5$ rad. These survive the smoothness checks because robot reset routines are smooth; they just wreck the normalisation percentiles.
5. **LLM instruction screening**, then **VLM video–instruction consistency** (split episodes at gripper open/close events, caption each clip, check against the instruction).
6. **Manual replay** of sampled episodes per dataset–embodiment pair against the URDF, to catch gripper-sign and coordinate-frame mistakes.

**Ego2Robot** converts human video into fake robot video. Wrist pose → EEF pose; gripper openness from thumb-to-(index+middle)/2 distance, clipped so 5 cm = closed, 7 cm = open. Then a coupled base-placement search + trajectory IK (MuJoCo, coarse→medium→full frames), SAM3 to segment the human, ProPainter to inpaint them out, and the rendered robot composited in with depth-aware occlusion. Human trajectories are slowed **2×** to match robot speed. 1,101 h of source video yields 5,634 robot-hours across 3.73M training segments.

### Training

256 A800s, [[Mixed Precision Training|bf16]], batch 16/GPU → global 4096, [[Decoupled Weight Decay Regularization (AdamW)|AdamW]] at $5\times10^{-5}$, weight decay $10^{-2}$, 5% warmup then cosine. 33 video frames, 32-step action chunk, video:action frequency ratio 4. Mix: 80% robot, 10% Ego2Robot, 8% UMI, 2% raw ego. **235K steps without** 4D distillation, then **10K more with** it — and only 1 sample in each 16-sample microbatch carries the 4D loss. ~14 days.

Post-training is per-benchmark from the *same* checkpoint, 10–15 epochs, same LR.

### Inference

One video-expert prefill per new observation, caching self-attention $K/V$ per layer ([[KV Cache]], same logic as [[Prompt Caching]] — the cache is valid because the mask forbids video tokens attending to action tokens, so nothing on the video side changes as actions denoise). Then $N$ flow steps updating only the action stream: 32 steps × 80 dims.

For asynchronous execution they use **training-time RTC**: during post-training, sample a committed prefix length $d\sim\mathcal{U}\{0,\dots,16\}$, keep those ground-truth actions clean with flow timestep 0, noise the rest, and only supervise the suffix. At deployment the previous plan supplies the prefix. Replan every 16 of 32 steps; at 30 Hz the budget is 533 ms.

## Ablation Studies and Experiments

### Headline numbers

| Benchmark | InternW0-$\Delta$ | Best prior WAM | Best prior VLA |
|---|---|---|---|
| LIBERO-Plus overall SR | **92.8%** | 84.8 (Being-H0.7) | 91.4 (Qwen-RobotManip) |
| RoboTwin 2.0 Clean2Clean | **90.0%** | 89.4 (OpenWAM-$\alpha$) | 84.7 |
| RoboTwin 2.0 Clean2Random | **71.9%** | 48.7 | 69.4 |
| EBench overall score | **66.0** | 64.7 | 60.0 |
| RoboDojo avg SR / score | **23.91 / 30.77** | 11.92 / 17.18 | 19.34 / 24.90 |

The Clean2Random number is the interesting one: trained on clean demos only, evaluated under randomised background, lighting, clutter and table height. 48.7 → 71.9 against the best prior WAM, *without* losing anything on the clean split.

On LIBERO-Plus the biggest per-axis gain is **robot perturbation**: 91.1% vs. 87.4% previous best. Camera 90.6%, language 92.9%.

RoboDojo is where everyone is still bad. 23.91% average. It leads on Gen-Std (33.78%) and Precision (23.25%), but GPT-6 Astra as a policy beats it on Gen-Rand (28.33 vs 11.78) and Open-vocabulary (31.00 vs 10.17).

### The cumulative ablation — this is the real content

Every row is an **independent training run** on the same data and recipe, evaluated on LIBERO-Plus.

| Configuration | SR (%) | Δ |
|---|---|---|
| Baseline | 49.59 | — |
| + Sparse Memory Context | 53.47 | +3.88 |
| + Qwen3.5-2B as VLM | 60.37 | +6.90 |
| + RynnBrain1.1-2B instead | 69.08 | +8.71 |
| + Causal Imprint ($\mathcal{L}_\Delta$ only) | 70.80 | +1.72 |
| + $\mathcal{L}_{\mathrm{align}}$ | 76.45 | **+5.65** |
| + $\mathcal{L}_{\mathrm{4D}}$ | 78.37 | +1.92 |

Read that honestly: **the single biggest lever is which frozen VLM you bolt on.** Swapping Qwen3.5-2B for RynnBrain1.1-2B — same everything else — is worth 8.71 points, more than the paper's two named contributions combined. From the SMC baseline, the VLM pathway alone is worth 15.6 points; Causal Imprint plus alignment is worth 7.4; 4D distillation is worth 1.9.

And within Causal Imprint, the *named* objective (regress latent differences) buys 1.72 points. The **feature-alignment** objective buys 5.65. The thing that works is copying the video model's internal future representation, not regressing raw latent deltas.

### What did not work

**Pi3X as a distillation teacher made things worse than no distillation at all.**

| Teacher | Action injection | SR (%) |
|---|---|---|
| None | — | 76.45 |
| CoWTracker | No | 76.15 |
| Pi3X | No | **73.63** |
| Track4World | Yes | 77.78 |
| Track4World | No | **78.37** |

Two negatives here. Pi3X (a per-frame geometry model) costs 2.8 points versus not distilling; CoWTracker is a wash. Only Track4World — which carries *world-centric temporal* tracking, not per-frame geometry — actually helps. So "add a 3D prior" is not the finding; "add a *motion-over-time* prior" is.

Second: feeding the student descriptor into the action expert is **worse** (77.78 vs 78.37). The geometry is better used as pressure on the video representation than as an extra input to the policy.

**Raw egocentric human video barely does anything.** Pretraining on one source only:

| Pretrain source | LIBERO-Plus | RoboTwin Clean2Random |
|---|---|---|
| None | 78.59 | 4.38 (joint) / 2.80 (EEF) |
| Robot | **83.73** | **32.34** |
| Raw Ego | 80.45 | 3.13 |
| Ego2Robot | 81.86 | 6.66 |
| UMI | 83.24 | 13.90 |

Robot data dominates, especially for robustness (4.38 → 32.34). Raw human video moves RoboTwin by 0.33 points — effectively nothing. Ego2Robot's expensive inpaint-and-render pipeline gets you to 6.66. UMI — handheld gripper data, so the *action space already matches a robot* — gets 13.90 with no rendering at all.

The lesson is blunt: **the closer the action and observation space is to the target robot, the more the data is worth.** Pixel realism is secondary. Compare [[HuRo- Robotizing Human Videos for Scalable VLA Pretraining]], which reaches a similar conclusion from the other direction, and [[Breaking the Vision-Action Shortcut- Latent Interface Training for Generalizable Robotics Foundation Models]] on why vision→action shortcuts are fragile.

Authors' own caveat, and it is fair: both benchmarks are gripper tasks, and success rate may not see representation gains. Human hand data plausibly pays off for dexterous contact, which neither benchmark tests.

**CUDA Graph capture broke gradients** when combined with AOTAutograd and non-reentrant activation checkpointing. They disabled it on the checkpointed path. Rare, valuable, honestly reported negative result.

### Latency ablation

Cumulative, dexterous-hand deployment, one RTX 5090, 50 timed requests:

| Step | RTT (ms) | Speedup | LIBERO-Plus SR |
|---|---|---|---|
| Standard runtime | 780.5 | 1.00× | 92.78 |
| + process isolation (no rclpy) | 374.1 | 2.09× | 92.67 |
| + feature caching & compilation | 249.8 | 3.12× | 92.46 |
| + context caching | 217.3 | 3.59× | 92.41 |
| + grouped action execution | 186.2 | 4.19× | 92.33 |
| + CUDA-graph replay | 152.8 | **5.11×** | 92.23 |

Note the first row: **moving inference out of the ROS process halved latency.** ROS callbacks, timers and the Python GIL were eating 400 ms of CUDA launch scheduling. That is a larger win than every kernel-level optimisation combined.

Accuracy cost of the whole stack: 0.55 points.

### Training throughput

VAE/VLM feature caching + layerwise compilation: **3.02×** end-to-end on LIBERO, **2.11×** on RoboTwin. Compilation alone (when inputs are stochastic and caching is impossible): 20–30%.

### Real robots

Same pretrained checkpoint, four platforms, four different action adapters (AC-One 14D bimanual joints; Arx5 6D EEF + 1D gripper; Franka+XHand 6D EEF + 12D hand per side; TianJi Marvin 7D arm joints + 20D hand).

Pretraining ablation, 20 trials each:

| Task | No pretraining | With pretraining |
|---|---|---|
| Toast bread | 4/20 (20%) | **19/20 (95%)** |
| Luminol reaction | 0/20 (0%) | **19/20 (95%)** |

0% → 95% on a four-reagent chemistry sequence. This is the strongest evidence in the paper that the 20K-hour corpus is doing real work.

**RTC continuity.** Ratio of action change at the plan boundary to ordinary within-plan change (1.0 = no visible seam):

- VJP (inference-time guidance): **8.94**
- Hard prefix: **7.13**
- Training-time prefix conditioning: **1.12**

Hard-prefixing preserves the committed actions but produces a "jetting" jerk at the prefix→suffix seam, because the model switches abruptly from copying to generating. Training it to continue a prefix removes the seam.

### GPT-guided policy (exploratory)

Five RoboDojo tasks where the policy is weak, GPT-6 Astra watching and issuing EEF corrections: average SR **10.80% → 47.20%**, score 13.92 → 52.00. "Classify objects" 14% → 80%. Tasks at 0% become partially solvable. Also: Astra's reasoning budget matters — `xhigh` completed all 5 language-classification episodes, `medium` completed none. Five episodes, so treat as anecdote.

## Worth Remembering

**The honest summary of the contribution.** The paper's title concepts (Causal Imprint, 4D distillation) are worth 7.4 and 1.9 points respectively. The frozen VLM choice is worth 8.7 on its own, and robot-data pretraining is worth 28 points of RoboTwin robustness. If you were ranking levers by return on engineering effort: data scale and filtering > which VLM you pick > future-feature alignment > latent-difference regression > geometry distillation.

**Causal Imprint generalises beyond robotics.** The pattern is: *use a privileged signal you only have at training time to shape a representation computed from non-privileged inputs, enforced by an attention mask rather than a separate network.* That is the same shape as privileged-critic and privileged-distillation work ([[Best Practice Critic Optimization]], [[What Does Privileged Information Add to On-Policy Self-Distillation]]). For an evaluation person: it is a clean way to encode "I have the counterfactual outcome in my logs but not at serve time."

**Stop-gradient everywhere.** Both $\mathcal{L}_{\mathrm{align}}$ and $\mathcal{L}_{\mathrm{4D}}$ use `sg[·]` on the target. Without it, the cheapest solution is for the target to move towards the student and both collapse — the standard failure discussed in [[JEPA#The catch: collapse|JEPA's collapse]] and [[Mode Collapse]].

**"20K hours" is a load-bearing but slippery number.** 11.3K robot + 2.1K UMI + 4.1K ego ≈ 17.5K hours of *source*. Ego2Robot's 5,634 "robot-hours" come from only 1,101 hours of unique source coverage, expanded across embodiments and reported *before* the 2× slowdown. So the headline number double-counts human video across robot bodies. The paper is transparent about this in §4.5; a casual reader would not notice.

**Filtering thresholds worth copying.** The 0.65 state–action directional-agreement cut and the state-changes-precede-actions rejection are cheap tests that catch timing bugs in third-party datasets. The blur detector measured only over Sobel edge pixels is a nice fix for the false-positive that plagues textureless tabletop scenes.

**Limitations the authors state.** No systematic study of how egocentric data should be scaled, converted or weighted — they admit the 2% mixture weight and 0.1 loss weight are unjustified. Agent-assisted control is one corrective setting only; no study of when to invoke the agent, hierarchical planning, or tighter integration.

**Limitations they don't state.** Every baseline number is copied from other papers rather than rerun. The component ablations are at a much smaller scale than the real pretraining run, so it is unclear whether the VLM-vs-Causal-Imprint ranking survives at 245K steps on 20K hours. And the 4D distillation was bolted on for the last 10K steps of a 245K-step run with only 1/16 of each batch carrying the loss — a 1.9-point gain from that budget is either remarkably efficient or noise.

**Practical caveats if you wanted to use this.** The video expert is a 5B Wan checkpoint and the frozen VLM is another 2B — you are serving ~7B of parameters to emit 32×80 numbers. The action expert is only 1024-dim, so most of the compute is the video half you never sample from. Faster-WAM ("do WAMs need deep action modules?") is cited but not compared against, which is the obvious question. Also: the compiled deployment path pads the VLM context to exactly 640 tokens and *rejects* longer contexts rather than truncating, so changing camera count or prompt length means re-tuning that constant.

**Follow-up questions.**
- Does Causal Imprint still help if you drop the video flow-matching loss entirely? The ablation never tests the video expert *without* its generative objective, so we do not know whether video generation is doing work or is just the thing that produced the good initialisation.
- Why does block 8 → block 20 work for alignment? Both are unmotivated constants and unswept.
- UMI beats Ego2Robot at a fraction of the pipeline cost. Is the inpaint-and-render stage worth anything at all, or would action-only retargeting with the original human pixels do as well? [[HuRo- Robotizing Human Videos for Scalable VLA Pretraining]] ran exactly that control.

## Links

Related: [[InternW0- A Foundational Physical World Model for Efficient Real-World Interactions]] · [[Hydra-0- Action Flow for Generalist World Modeling and Control]] · [[GigaBrain-0.7- Scaling Embodied Foundation Models to Emergent Capabilities with a Three-System Architecture]] · [[Think Like a World Model, Act Like a VLA- Distilling World-Model Representations into Compact Robot Policies]] · [[HuRo- Robotizing Human Videos for Scalable VLA Pretraining]] · [[Breaking the Vision-Action Shortcut- Latent Interface Training for Generalizable Robotics Foundation Models]] · [[Latent Action as Intention Enables Efficient Future Imagination for World Action Models]] · [[MemBodied- Recurrent Associative Memory for Vision-Language-Action Models]] · [[Modality-Autoregressive World-Action Models]] · [[Zero-WAM- In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization]] · [[GameWAM- A World Action Model for Video Games]] · [[Flow Matching]] · [[Mixture of Experts]] · [[JEPA]] · [[Self-Supervised Learning from Images with I-JEPA]] · [[Distillation]] · [[Distilling the Knowledge in a Neural Network]] · [[Video Diffusion]] · [[Diffusion Models]] · [[Attention]] · [[Cross Attention]] · [[KV Cache]] · [[Prompt Caching]] · [[Imitation Learning]] · [[Exploring the Limits of Transfer Learning (T5)]] · [[Mixed Precision Training]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Denoising Objective]] · [[Mode Collapse]] · [[Best Practice Critic Optimization]] · [[What Does Privileged Information Add to On-Policy Self-Distillation]]

New topics worth writing: World Action Models, Mixture-of-Transformers, Vision-Language-Action models, Real-Time Chunking and action-chunk continuity, Wan video diffusion backbone, Track4World and world-centric 3D tracking, LIBERO and LIBERO-Plus, RoboTwin Clean2Random protocol, Universal Manipulation Interface (UMI), human-to-robot retargeting and inverse kinematics, cross-embodiment canonical action spaces, robot demonstration data filtering, CUDA Graphs and torch.compile interaction with activation checkpointing
