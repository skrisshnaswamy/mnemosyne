---
title: "GigaBrain-0.7: Scaling Embodied Foundation Models to Emergent Capabilities with a Three-System Architecture"
authors: ["Team et al."]
year: 2026
arxiv: "2608.15875"
url: https://arxiv.org/abs/2608.15875
priority: Good-To-Read
read_on: 2026-08-31
tags: [paper, llm, vision]
---
## The Core Idea

A robot policy that only maps "what I see now" → "what I do next" is blind in two directions. It cannot tell whether the current moment is *early* or *late* in a task when the picture looks the same either way, and it has no opinion about whether the last few seconds helped or hurt. GigaBrain-0.7 splits an embodied foundation model into three cooperating pieces to fix both blind spots, and then scales the whole thing on 37,257 hours of robot and human demonstration data across 16 robot bodies.

The three systems:

- **System 2 — understanding and planning.** A vision-language model (VLM) that looks at the scene, writes a short chain-of-thought, and turns "hang the item on the hanger" into the immediate subtask "pick up the red hat".
- **System 3 — prediction and evaluation.** A 5B *world value model*. It (a) generates a short video of what it thinks will happen and hands the **last frame** to the policy as a "subgoal image", and (b) outputs a scalar $V_t$ = how far along the subtask we are.
- **System 1 — action and control.** The actual policy. A 3B VLM backbone plus a 0.5B action expert that emits continuous action chunks by flow matching, conditioned on the current image, a short visual memory, the subtask text from System 2, the subgoal image and progress signal from System 3, the robot's joint state, and a Robot ID.

Two ideas here are worth carrying forward regardless of what you think of the rest.

**One-stage pretraining.** Most VLAs (including $\pi_{0.5}$) train in phases: first make the VLM understand robots using *discrete* action tokens, then bolt on a continuous flow-matching head later. GigaBrain-0.7 trains the language loss and the continuous-action loss together from the start, in one pass, on everything.

**Soft Knowledge Insulation.** The known problem: gradients from the action head wreck the VLM's language and spatial understanding. The prior fix (Knowledge Insulation) is to *block* those gradients entirely. Here they instead scale them by a coefficient $\alpha_{\mathrm{KI}} \in (0,1)$ — a leaky valve rather than a wall — so the backbone can drift toward embodied-relevant features without forgetting how to read a scene.

> [!NOTE] Vision-Language-Action model (VLA)
> A pretrained vision-language model repurposed to output robot commands. The VLM supplies semantics ("that is a red hat, it is to the left"); a separate head turns those semantics into joint angles or end-effector poses. ^vla-model

> [!NOTE] World value model
> A video-generation model given a second head that scores task progress. It answers both "what will the scene look like in two seconds?" and "am I getting closer to done?" — the second answer is what turns a video model into something a policy can learn from. ^world-value-model

## The Methodology

### The data pyramid

37,256.98 hours of trajectories after cleaning:

| Source | Hours | Share |
|---|---|---|
| Real robot | 20,535.65 | 55.1% |
| UMI (handheld gripper demos) | 8,251.83 | 22.2% |
| EGO (first-person human video) | 2,862.36 | 7.7% |
| Simulation | 1,453.92 | 3.9% |
| World-model generated | 4,153.22 | 11.2% |

Plus **271,976,674** image–text / VQA samples (captioning, grounding, spatial relations, affordance, point prediction) so the VLM does not forget how to see. Real-robot data covers 16 robot types, 1.81M episodes, 2.08B frames; Maker H01 (43.8% of frames) and Maker M01 (20.1%) dominate.

Cleaning steps that matter:

- Everything converted to LeRobot v3.0 format.
- **Unified state/action vector** ordered left arm → right arm → head → waist → base/legs, with a per-embodiment *validity mask* so "this robot has no waist" is distinguishable from "waist angle is zero". Invalid dimensions are excluded from both the flow-matching loss and the noise injection. Rotations use the continuous 6D representation.
- **q01/q99 filtering.** Training normalises by quantiles, so a single outlier stretches the normalisation range and squashes all real signal into a sliver. They drop frames outside the 1st–99th percentile per robot type per dimension.
- Long stationary segments removed (normalised inter-frame $L_2$ change below a threshold).
- Language instructions rewritten into a canonical form by GLM-5.1, then spot-checked by humans; subtasks segmented by a multimodal model.
- **Loss as a cleaning signal.** During early pretraining they log per-sample loss; trajectories that persistently spike get traced back to the raw video and usually turn out to be misaligned, corrupt, or mislabelled. Those go on a blocklist.

### The System 1 block

Mixture-of-Transformers: two parameter streams, one shared attention. At layer $l$, with $\mathbf{H}^{\mathrm{VL}}$ the vision-language stream and $\mathbf{H}^{\mathrm{AE}}$ the action-expert stream:

$$\mathbf{O}^{\mathrm{VL}}_{l} = \mathrm{CausalAttn}_{l}(\mathbf{H}^{\mathrm{VL}}_{l};\, \mathbf{H}^{\mathrm{VL}}_{l})$$
$$\mathbf{O}^{\mathrm{AE}}_{l} = \mathrm{Attn}_{l}(\mathbf{H}^{\mathrm{AE}}_{l};\, [\mathbf{H}^{\mathrm{VL}}_{l}, \mathbf{H}^{\mathrm{AE}}_{l}])$$

then separate feed-forward blocks $\mathrm{FFN}^{\mathrm{VL}}_l$ and $\mathrm{FFN}^{\mathrm{AE}}_l$. Read it as: the language side keeps its normal [[Causal Attention|causal]] self-attention and never sees action tokens (so the pretrained [[Attention Is All You Need|transformer]] behaviour is untouched), while the action side attends bidirectionally over *both* streams. Same trick as $\pi_0$.

**Short-term visual memory.** Rather than pushing $k$ frames of visual tokens into the backbone (which multiplies context length by $k$), *Temporal-Spatial Blocks* sit inside the visual encoder. Temporal aggregation runs causally across recent frames, spatial aggregation runs within the current frame, then **past-frame tokens are thrown away** and only the enriched current-frame tokens go forward. Token count stays roughly single-frame. History frames are randomly dropped during training so the model tolerates any history length, including none.

### The two losses

Next-token prediction, over subtask text, discrete action tokens, and general VL targets:

$$\mathcal{L}_{\mathrm{NTP}} = -\mathbb{E}\Big[\sum_{j=1}^{M} m_j \log p_\theta(y_j \mid c, y_{<j})\Big]$$

Flow matching for the continuous chunk. Interpolate ground-truth actions with noise, $a^\tau_{t:t+H} = \tau a_{t:t+H} + (1-\tau)\epsilon$, and regress the velocity field:

$$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{a,\epsilon,\tau}\big[\|v_\theta(a^\tau_{t:t+H}, c, \tau) - (a_{t:t+H} - \epsilon)\|_2^2\big]$$

Total: $\mathcal{L}_{\mathrm{VLA}} = \mathcal{L}_{\mathrm{NTP}} + \mathcal{L}_{\mathrm{FM}}$. Soft KI modifies only what reaches the backbone:

$$\nabla_{\theta_{\mathrm{VL}}}\mathcal{L}_{\mathrm{VLA}} = \nabla_{\theta_{\mathrm{VL}}}\mathcal{L}_{\mathrm{NTP}} + \alpha_{\mathrm{KI}}\nabla_{\theta_{\mathrm{VL}}}\mathcal{L}_{\mathrm{FM}}$$

Inside the action expert the FM gradient flows at full strength. (They never report the value of $\alpha_{\mathrm{KI}}$ — annoying.)

**Hierarchical supervision.** Half the time the model is given both the task instruction $\ell$ and the aligned subtask $\hat{\ell}_t$ and just predicts actions; half the time only $\ell$, so System 2 must infer the subtask autoregressively before System 1 acts. Attention masks stop subtask targets and future actions leaking backwards.

### System 3, trained separately

Stage I: continue video pretraining of GigaWorld-1 on robot manipulation footage, so the horizon of the generated clip matches System 1's action chunk. Last frame = subgoal image $g_t$.

Stage II: extend it into an MoT world *value* model; a second pathway attending mostly to current and past images regresses task progress, supervised from trajectory-level completion annotations.

Then **freeze it**. During task-specific post-training it produces fresh $g_t$ and $V_t$ per sample but receives no gradients.

### How System 3's output is consumed

The scalar value is thrown away and only its *sign of change* is kept:

$$A_t = \mathbb{1}[V_{t+\delta_t} - V_t > 0] \in \{0, 1\}$$

$A_t$ becomes a single conditioning token in the prompt. So during training the policy learns "this action segment was labelled as making progress" versus "this one was not", and **at inference you simply always set $A_t = 1$** and it generates progress-y behaviour. This is the same advantage-conditioning trick as RAMP / $\pi^*_{0.6}$'s RECAP, but the advantage comes from a world model rather than a human or a VLM judge.

Post-training context: $c_t^{\mathrm{post}} = (c_t^{\mathrm{base}}, g_t, A_t)$, optimised with the same $\mathcal{L}_{\mathrm{FM}}$. **Condition dropout** drops the subgoal image with probability 0.50 and the value token with probability 0.15, *independently*, so the model sees all four regimes and does not become dependent on either.

### Experience reinforcement

Three stages after supervised fine-tuning, all refining the *same* System 1 policy:

1. **Offline RL.** Deploy, collect rollouts with successes and failures. Reward from the progress discriminator plus hand rules; advantage $\tilde{A}_t = G_t - \tilde{V}_t$ (return minus value estimate); Advantage-Weighted Regression re-weights segments so higher-progress rollouts count more.
2. **Online RL.** Redeploy, collect on-policy trajectories, a human operator intervenes at hard states and the correction is recorded as labelled supervision *at exactly that state*. Actor–critic, rewards = chunk-level progress difference + terminal sparse signal. Iterate $\pi^{(k)} \to \mathcal{D}^{(k)} \to \pi^{(k+1)}$.

## Ablation Studies and Experiments

### Backbone: bigger is not better

Three VLM backbones, dual-stream architecture fixed, real-robot success rates:

| Backbone | Size | Clean Desk | Fruit Picking | Shirt Folding |
|---|---|---|---|---|
| PaliGemma2 | 3.5B | 50% | 88% | **30%** |
| Qwen3.5 | 5B | 60% | 12% | 0% |
| Gemma 4 | 8.5B | 100%* | 92% | 0% |

Gemma 4 wins the structured tasks (its Clean Desk number is asterisked — easier position range) but scores **zero** on shirt folding. PaliGemma2, the smallest, is the only one that folds a shirt at all. They pick it. The obvious caveat, which the authors state: sizes and image resolutions differ, so this is a selection experiment, not a clean scaling law.

### Coupling: dense beats cheap

| Architecture | Clean Desk | Fruit | Shirt | Train s/step | Infer s |
|---|---|---|---|---|---|
| Dual Stream | **50%** | **88%** | **30%** | 4.93 | 0.221 |
| Last-Layer Cross-Attn | 20% | 40% | 0% | 4.65 | **0.073** |
| Multi-Layer Cross-Attn | 20% | 36% | 0% | 6.59 | 0.108 |

**This is the clearest negative result in the paper.** Cross-attending only to the VLM's last layer is 3× faster at inference and collapses on every task. Multi-layer cross-attention — a per-layer interface, so *more* information — is both slower to train and no better. Only full interleaved joint attention works. The information pathway between semantics and motor control apparently needs to be very wide.

### Data scaling

- Validation loss drops monotonically as pretraining data grows (fixed model).
- Real-robot success rises with robot-data hours, but **unevenly**: fruit manipulation improves smoothly; chaotic clothes folding is nearly flat then rises sharply. Deformable, long-horizon skills need far more coverage before they appear at all. The authors call this emergence; it is at least a strong non-linearity.
- Starting from 3,000 hours of robot data, adding an equal amount of UMI *or* EGO both beat robot-only; adding both is best. The gap **survives task-specific post-training**, which is the meaningful part — the human data improves the prior, not just the base policy's immediate score.

### Temporal context

Qualitative only, no numbers. Without history, the single-frame policy gets stuck in a loop: it returns to a visually identical state and repeats the same action forever. With temporal context it exits the loop. This is exactly the failure mode a [[Markov Property|Markov]] assumption on raw pixels predicts.

### System 3 ablation (the one worth reading)

Four settings: Base, +SubImage, +Value, +SubImage+Value.

**Clothes folding (AgileX PiPER):** all four score **100% success**. The interesting signal is elsewhere — task score 68.3 → 81.7 → 85.0 → **88.3**, completion time 107 s → 83 → 79 → **75 s**. When success saturates, the conditioning still buys quality and speed.

**Gift wrapping (PiPER-X):** base **fails completely (0%)**. +SubImage 20%, +Value 60%, +SubImage+Value **80%**. Score 61.1 → 71.4 → 93.3 → 96.7. Value conditioning is clearly doing more work than the subgoal image here.

**Cube sorting (Maker H01):** messier. 40% → 50% (image) → 45% (value) → 55% (both). Value alone is *worse* than image alone on success, though it gives the shortest completion time (80 s). Combined score jumps 55.0 → 87.5.

Read across the three: the cheap scalar progress token often outperforms the expensive generated subgoal image, and the combination is usually but not always best.

### Post-training, real robots

Language following, average over six tasks (colour / target / direction grounding):

| Model | AgileX PiPER | Maker H01 |
|---|---|---|
| $\pi_{0.5}$ | 88.8 | 75.2 |
| GigaBrain-0.1 | 76.1 | 69.6 |
| Galaxea G0.5 | 81.4 | 25.7 |
| Xiaomi-Robotics-1 | 72.3 | 58.7 |
| **GigaBrain-0.7** | **91.5** | **84.2** |

Complex manipulation averages: PiPER **84.9** vs $\pi_{0.5}$ 76.6; H01 **74.1** vs $\pi_{0.5}$ 45.2. The H01 gap is much larger than the PiPER gap — unsurprising, since H01 is their own platform and supplies 43.8% of the real-robot frames. Rice sweeping stays hard for everyone (40% best).

### Benchmarks

- **RoboTwin 2.0** (one policy co-trained on 50 tasks, 50 demos each): Easy 66.8 / Hard 67.9 / overall **67.35**. $\pi_{0.5}$ gets Easy **70.7** but Hard only 46.0 (58.35 overall). GigaBrain-0.7 *loses* on the clean setting and barely degrades under domain randomisation — that flatness is the actual result.
- **EBench** (mobile + long-horizon): SR **33.30%**, score **46.1**, vs $\pi_{0.5}$ 28.08 / 42.
- **RoboColiseum** (Real2Sim2Real reconstruction): best on all four axes — instruction following .8166, spatial reasoning .4729, robustness .6800, general manipulation .6092.
- **MiMo-Embodied** VLM benchmark: overall .4621 vs best baseline G0.5-base .3916, and this is *before* MiMo data enters the mixture. After adding it, .5704 — which they honestly flag as a contaminated diagnostic, not a held-out number. The released checkpoint is the contaminated one.

### Experience-driven RL

| Task | SFT | Offline RL | Online RL |
|---|---|---|---|
| Link Installation (PiPER) | 20% | 40% | 100% |
| Gift Box Packing (PiPER-X) | 80% | 90% | 100% |
| Cable Tie Insertion (PiPER-X) | 0% | 40% | 100% |
| Bearing Installation (H01) | 20% | 60% | 100% |

Average 30.0% → 57.5% → 100%. Offline RL helps most where SFT was failing a lot (cable tie 0→40, bearing 20→60) and least where SFT was already strong (gift box 80→90) — sensible, since AWR can only re-weight experience you already have.

## Worth Remembering

**The 100% column should make you uneasy.** Four for four, every task, after online RL with human correction. No trial counts are given for that table (elsewhere they say 10–20 per config). The paper explicitly defers "a detailed treatment of the human-intervention protocol and the full online RL formulation" to a future report — so the headline result of the last section is the least reproducible part of the paper. Treat it as a demo, not a measurement.

**What the paper admits.** Out-of-distribution performance degrades on both platforms; generalisation holds when the *interaction structure* is preserved and breaks when object identity, scene, and task composition all shift together. Several contact-rich tasks (rice sweeping at 40–46%, cube sorting on H01 at 41.7%) are nowhere near saturated. And the MiMo score of the released checkpoint is train-on-test.

**The architecture result generalises beyond robotics.** Last-layer cross-attention as a way to bolt a decoder onto a frozen encoder is a very common design (it is cheap, and it is what a lot of adapter work assumes). Here it loses 30 points of success rate versus per-layer joint attention. If you are attaching a specialised head to a pretrained backbone and the task is *hard*, a narrow interface may be silently costing you far more than the FLOPs you saved.

**Advantage as a prompt token is a cheap trick worth stealing.** No policy gradient, no critic at training time, no [[Proximal Policy Optimization Algorithms|PPO]] machinery. You label each training segment "good" or "bad", condition on the label, and at inference always ask for "good". It is the same shape as [[Classifier-Free Diffusion Guidance|classifier-free guidance]] — train with the condition sometimes dropped, exploit the condition at sampling time. The condition-dropout probabilities (0.50 image, 0.15 value) exist for exactly the same reason.

**Loss-based data cleaning** is underrated and easy: log per-sample loss for a few thousand steps, sort descending, look at the top of the list. In their case it surfaced temporal misalignment, discontinuous actions, wrong instructions, and corrupt video.

**Open questions.** What is $\alpha_{\mathrm{KI}}$, and how sensitive is anything to it? There is no ablation of Soft KI versus hard KI versus no insulation, which is odd given it is named as a contribution. How much of the H01 dominance is method versus 910M frames of home-turf data? Does the frozen System 3 become a bottleneck as System 1 improves past it — the world model never sees the post-trained policy's own state distribution.

## Links

Related: [[Hydra-0- Action Flow for Generalist World Modeling and Control]] · [[τ_0-VLA- a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation]] · [[EXIMO- VLM Guided Exploration of VLA Policies]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)]] · [[WorldMind- Decoupled Game World Model for State-Aware NPC Behavior]] · [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] · [[Conservative Q-Learning for Offline RL]] · [[Training language models to follow instructions with human feedback]] · [[Classifier-Free Diffusion Guidance]] · [[Score-Based Generative Modeling through SDEs]] · [[Denoising Diffusion Probabilistic Models]] · [[Attention Is All You Need]] · [[Causal Attention]] · [[Sparsely-Gated Mixture-of-Experts Layer]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Scaling Laws for Neural Language Models]] · [[The Bitter Lesson (essay)]] · [[High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)]] · [[Markov Property]]

New topics worth writing: Flow matching for generative modelling (Lipman et al.), Mixture-of-Transformers architectures, Knowledge Insulation for VLAs, Advantage-Weighted Regression, $\pi_0$ / $\pi_{0.5}$ / $\pi_{0.7}$ VLA line, PaliGemma2, Universal Manipulation Interface (UMI) data collection, Egocentric video for robot learning, Real2Sim2Real evaluation, 6D continuous rotation representation, RoboTwin 2.0 and EBench benchmarks, Human-in-the-loop corrective RL (DAgger lineage)
