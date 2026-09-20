---
title: "GameWAM: A World Action Model for Video Games"
authors: ["Yuncheng Guo", "Zhanqiu Zhang", "Yiwen Guo", "Weijia Li"]
year: 2026
arxiv: "2608.26200"
url: https://arxiv.org/abs/2608.26200
priority: Good-To-Read
read_on: 2026-09-05
tags: [paper, transformers, diffusion, vision]
---
## The Core Idea

Two families of models exist for video games, and neither does the other's job.

**Game agents** look at the screen and output button presses. They do not model what the screen will look like next. **Interactive world models** (Genie, MineWorld, GameGen-X) predict what the screen will look like next *given* actions — but a human or an external controller has to supply those actions. One picks behaviour blind to consequences; the other predicts consequences but cannot pick behaviour.

A **World–Action Model (WAM)** generates both at once: the future frames *and* the future keyboard/mouse actions, from the same model, in the same forward pass. Predicting pixels acts as free supervision that forces the network to understand game dynamics; that understanding then feeds the action head. WAMs already existed for robot tabletop manipulation (DreamZero, Fast-WAM). GameWAM is the first one built for native closed-loop *video game* control — first-person view, fast camera motion, and a real keyboard and mouse.

> [!NOTE] World–Action Model
> A generative model that jointly produces future visual observations and the executable action trajectory that leads to them, so visual prediction supervises control. ^world-action-model

Three problems had to be solved to make this work in games, and they are the actual content of the paper.

1. **Games have two control regimes on one physical device.** During gameplay, the mouse is a camera. Inside the inventory GUI, the same mouse is a cursor. Identical numbers, totally different statistics. Modelling them as one distribution mixes incompatible scales.
2. **Frames must be sampled densely or you miss things**, but dense sampling burns the token/KV budget, which shortens both how far you can look ahead and how much past you can remember.
3. **A new failure mode nobody had named.** In flow/diffusion action generation you sample a Gaussian "source" noise tensor and integrate it into an action chunk. GameWAM found that the *low-frequency* part of that noise, along the time axis of the action chunk, systematically steers the coarse camera motion — regardless of what the model is looking at. Reuse the same source across replanning steps and the agent slowly spins in place forever. They call this **Low-Frequency Action Source Imprinting (LASI)** and prove it causally with frequency-domain surgery.

The headline result: on the MCU Minecraft benchmark, average success 46.6% vs 42.5% for Game-TARS, using **138 environment steps per successful embodied task instead of ~290–380** for every baseline. It uses roughly 2.79B training tokens against Game-TARS's 566B.

## The Methodology

### The two towers

Two parallel Diffusion Transformers ([[An Image is Worth 16x16 Words (ViT)|DiT]]-style, [[Attention Is All You Need|transformer]] blocks with time-conditioned modulation):

- **Video DiT** — initialised from Wan2.2-TI2V-5B. 30 layers, 24 heads, width 3072. The video VAE is frozen.
- **Action DiT** — same 30/24 layout but width 1024 (~1B params). Initialised by adapting the Wan2.2 video weights down to the narrower width, so it inherits the video prior.

Both are trained with **flow matching**. For modality $m \in \{v, a\}$, you linearly interpolate the clean target with Gaussian noise and regress the velocity:

$$X^m_{\sigma_m} = (1-\sigma_m)X_0^m + \sigma_m \epsilon^m, \qquad U^m = \epsilon^m - X_0^m$$

The network predicts $\widehat{U}^m_\theta$; at inference you start from $X_1 \sim \mathcal{N}(0,I)$ and integrate the ODE from $\sigma=1$ to $0$. Ten Euler steps for closed-loop play. The two flow times $\sigma_v$ and $\sigma_a$ are sampled *independently*.

> [!NOTE] Flow matching
> Instead of a noise-prediction diffusion chain, you learn a velocity field that transports Gaussian noise to data along a straight line, then integrate it with an ODE solver. Fewer steps, simpler loss than [[Denoising Diffusion Probabilistic Models|DDPM]]. ^flow-matching

### The attention mask — the design that carries the paper

At block $j$, let $\mathcal{P}_{c,j}$ be the *clean* visual prefix (real frames that were actually observed), $\mathcal{V}_j$ the noisy future video latents, $\mathcal{A}_j$ the noisy future actions. The default **modality-decoupled** mask is:

$$\operatorname{Vis}(\mathcal{A}_j) = \mathcal{P}_{c,j} \cup \mathcal{A}_j, \qquad \operatorname{Vis}(\mathcal{V}_j) = \mathcal{P}_{c,j} \cup \mathcal{V}_j$$

So the action stream **never reads the noisy future video**, and vice versa. They only meet through the shared clean prefix, whose layer-wise K/V states are produced by the Video DiT and consumed by both. Gradients from the action loss still flow back through that Video-DiT path, so video supervision still shapes the world representation used for control.

Why this matters practically: at deployment you can **skip future-video denoising entirely** and just run the Action DiT off the cached prefix. That is 12.51 Hz instead of 8.12 Hz. The [[Causal Attention|block-causal]] structure over time is unchanged — later blocks are still masked.

### Predict long, execute short

Each plan is $P = 16$ native actions. Only the first $E = 8$ get committed to the environment. Then a real observation arrives and you replan the overlapping suffix. The unexecuted tail is thrown away — **only frames reached through executed actions ever become clean context**.

$$\widehat{\mathbf{A}}^{\mathrm{plan}}_{c,k} \in \mathbb{R}^{P \times d_a}, \quad E < P, \quad \widehat{\mathbf{A}}^{\mathrm{exec}}_{c,k} = \widehat{\mathbf{A}}^{\mathrm{plan}}_{c,k}[1{:}E]$$

During training this costs nothing extra in sequence length: anchors are placed every $E$ steps on the ground-truth trajectory and teacher-forced in parallel. Each anchor sees only its own causally valid prefix. Longer supervision horizon, same decision cadence, no longer autoregressive chain.

### Gameplay vs GUI routing

A per-timestep router emits a logit $\rho_\tau$; at rollout,

$$\hat{r}_\tau = \mathbf{1}[\operatorname{sigmoid}(\rho_\tau) > \tfrac12], \qquad \widehat{U}^a_\tau = (1-\hat{r}_\tau)\widehat{U}^{\mathrm{game}}_\tau + \hat{r}_\tau \widehat{U}^{\mathrm{gui}}_\tau$$

Both branches predict the *same* 22-dimensional Minecraft action vector (2 continuous camera deltas, 20 binary keys — WASD, jump, sneak, sprint, attack, use, drop, inventory, 9 hotbar slots). The difference is the **normalisation statistics for the continuous coordinates** — camera stats vs cursor stats. Binary coords share normalisation. Routing is supervised with masked BCE. A trajectory can flip mode mid-plan without switching models.

ViZDoom uses a 9-D interface; the two Battle maps have a continuous horizontal turn delta normalised to $[-1,1]$ (up to $10°$ per step), the two Defend maps use binary turn-left/turn-right.

### Memory: bounded cache plus hierarchical history

Within a cycle (3 execution blocks = 24 actions), realized frames extend a transient layer-wise K/V cache. At the cycle boundary that cache is **thrown away**. What survives is $H_c = [M_c; R_c]$:

- **Recent** — the executed segment is VAE-encoded and squashed by a Conv3D tokeniser to 4 latent times × 4 spatial tokens = 16 tokens per cycle. A FIFO of $K_R = 2$ such segments.
- **Long-term** — when a segment is evicted from the FIFO, it writes into fixed memory slots via cross-attention plus a gated update:
$$\widetilde{M}_c = \operatorname{Attn}(M_c, \bar{S}_c, \bar{S}_c), \qquad M_{c+1} = \operatorname{LN}[M_c + g_c \odot (\widetilde{M}_c - M_c)]$$
Two timescales, 4 slots each, half-lives of 2 and 4 cycles. The gate has a *time prior* $\alpha_\ell(\Delta) = 1 - 2^{-\Delta/h_\ell}$ added in logit space to a learned content correction, so short-timescale levels update fast and long ones update slowly. Total history ≤ 40 tokens, independent of episode length.

An auxiliary loss keeps the memory from going blank — predict the current clean visual feature from pooled history, with stop-gradient on the target:
$$\mathcal{L}_{\mathrm{hist}} = D_{\mathrm{pred}}(F(\operatorname{Pool}(H_c)), \operatorname{sg}(q_c))$$

### Full objective

$$\mathcal{L} = \lambda_v \mathcal{L}_v + \lambda_a \mathcal{L}_a + \lambda_m \mathcal{L}_{\mathrm{mode}} + \lambda_h \mathcal{L}_{\mathrm{hist}}$$

with $\lambda_v = \lambda_a = 1.0$, $\lambda_m = 0.05$, $\lambda_h = 0.5$. The action loss splits continuous and discrete coordinates and *normalises each group separately by its own valid-mask count* before summing with $\lambda_{\mathrm{cont}} = \lambda_{\mathrm{disc}} = 1.0$ — otherwise 20 binary dims drown out 2 camera dims.

### Data

Three Minecraft streams, all on a common observation–state–action timeline where the state paired with an action is the state *just before* it fires:

- **Regular VPT** (5% of samples) — Baker et al.'s YouTube-derived trajectories, 20 FPS, long context.
- **Event-Anchored VPT** (80%) — MineStudio-style events detected from state deltas, 96-frame windows $[t-87, t+8]$ around each event. Dual filtering (clip starts ≥64 frames apart, same-type anchors >96 frames apart) plus **max–min fair allocation** across event types so common events do not swamp rare ones. 200k windows.
- **Scripted GUI** (15%) — rule-driven MineStudio inventory/crafting demos.

On top of dataset construction there is **event-anchored clip sampling**: within the event stream, training windows near the anchor (radius 8) are drawn with stride 2; far regions use stride 16. Regular VPT uses stride 32, GUI stride 8.

Training: 2 epochs, 21,900 steps, [[Decoupled Weight Decay Regularization (AdamW)|AdamW]] fused ($\beta_1{=}0.9$, $\beta_2{=}0.95$, wd 0.01, clip 1.0), 5% linear warmup to $4\times10^{-5}$ then cosine to $4\times10^{-7}$, BF16, [[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models|ZeRO]]-2, 8× H200, global batch 352. **22 hours total.** Observations 224×224, one frame per two native actions.

## Ablation Studies and Experiments

### MCU (Minecraft, 800+ tasks)

| Model | Embodied steps ↓ | ASR All ↑ (avg) |
|---|---|---|
| VPT | 377 | 3.5 |
| STEVE-1 | 384 | 5.0 |
| JARVIS-VLA | 305 | 24.5 |
| OpenHA | 287 | 31.5 |
| Game-TARS (566B tokens) | 373 | 42.5 |
| **GameWAM** | **138** | **46.6** |

Per category on ASR All: embodied 47.5 (Game-TARS 50.4 — the one place it loses), GUI **60.0** vs 39.1, combat 32.2 vs 38.1. On the Mini subset average it is 50.7 vs OpenHA's 36.8. The step counts are the striking part: 138/155/203 versus 287–406 everywhere else. Fewer than half the actions to finish the same task.

On ViZDoom (4 maps, 50 episodes each) it beats Game-TARS on all four and is competitive with or above the multimodal agents.

### The ablation table (MCU Mini average, full model 50.7)

| Variant | Avg | Δ |
|---|---|---|
| Full GameWAM | **50.7** | — |
| Action-only supervision (no video loss) | 35.7 | **−15.0** |
| Coarser temporal sampling | 36.7 | −14.0 |
| No event-anchored clip sampling | 38.0 | −12.7 |
| Unified gameplay/GUI action distribution | 38.3 | −12.4 |
| Matched horizons ($P = E$) | 41.3 | −9.4 |
| No cross-cycle history | 46.7 | −4.0 |

Read this carefully. **The single biggest contributor is the video prediction loss itself.** Drop it and combat collapses from 39.0 to 13.0. That is the whole thesis of the WAM framing, isolated: predicting pixels is what teaches the action head about dynamics. Second is dense frame sampling — coarse sampling loses 14 points, which justifies the whole block-cycle apparatus that exists to afford dense sampling under a budget.

Cross-cycle history is the *weakest* component and it is not uniformly good: removing it **improves** embodied Mini from 70.0 to 75.0, while GUI drops 43→31 and combat 39→34. Memory helps tasks that span replanning cycles and mildly hurts ones that do not.

### Modality-decoupled vs joint attention

They tried letting noisy video and noisy action attend to each other within a block. It is worse *and* slower:

| Mask | Mini avg | All avg | Hz |
|---|---|---|---|
| Modality-decoupled | **50.7** | **46.6** | **12.51** |
| Joint video–action | 46.3 | 39.6 | 8.12 |

Combat All falls 32.2 → 19.6. The interesting detail: during training the **joint model fits the action loss faster** but its video branch optimises slower and its validation frame predictions are visibly blurrier. Faster action fitting, worse control. The authors hypothesise asymmetric optimisation interference — coupling action learning to a hard visual prediction problem makes the action objective easy to cheat and starves the visual one. They explicitly flag this as a hypothesis, not a mechanism. GUI barely degrades (45.0 vs 43.0 Mini) because GUI frames are visually static and easy to predict; combat, which is full of ego-motion, degrades hardest. That pattern is consistent with the interference story.

### LASI — the failure mode

Origin: reusing one sampled source $Z$ across all replanning steps in an episode made *some* seeds spin the camera in place forever, sometimes completing zero tasks. Resampling per step fixed it. So they went looking for why.

Take an 8-step continuous camera chunk, apply an orthonormal DCT along time (see [[Fourier Series Decomposition]] for the intuition — decompose a signal into frequency components). Modes 0–2 are "low frequency"; 3–7 high. Fix all conditioning (frames, history, proprioception) and vary only $Z$. Three tests, 24 conditions × 16 sources:

1. **Association.** Yaw DCT0 source coefficient vs yaw DCT0 output coefficient: $r = 0.890$ (CI [0.835, 0.936]). All six low-frequency modes are 0.61–0.89.
2. **Donor swap.** Replace only source modes 0–2 with another sample's. The output follows the donor in **94.8%** of trials; change-vs-change correlation $r = 0.921$.
3. **Zeroing.** Zero source modes 0–2. **99.25%** of the source-induced output variance in yaw DCT0 disappears.

Frequency selectivity: unit-norm perturbations in the low-frequency subspace produce **3.5–4.6×** the low-frequency response of equal-norm high-frequency perturbations. Not a magnitude artefact.

It is not just the ODE path. Constructing $X_\sigma^{\mathrm{ana}} = (1-\sigma)X_0 + \sigma Z$ and running **one** denoiser call already shows a source→output regression gain of 0.709 at $\sigma = 0.68$. But the iterative path *amplifies* it: at step 19 ($\sigma = 0.208$) the analytic gain is 0.162 while the model-path gain is 1.117 — a **6.9× ratio**. The gap is zero early in denoising and opens up late.

Closed-loop: 300 traces, 16,707 replanning steps. Episode-mean-adjusted source↔executed-action alignment is 0.33763, ranks **1st of all 75 circular shifts**, and beats all 999 permutations (max null 0.06466, $p = 0.001$).

The accumulation argument is simple. Write $\Delta_c = \mu(\mathcal{C}_c) + b(Z_c) + \varepsilon_c$. Reuse one source for $K$ cycles and the bias term contributes $K \cdot b(Z)$ — a coherent, growing turn. Resample i.i.d. and $\mathbb{E}[\sum_c b(Z_c)] \approx 0$ with variance only $K\operatorname{Var}[b]$.

**What they tried and abandoned:** frequency-selective losses penalising low-frequency source–action coupling; consistency losses making output invariant to the source realisation; extra supervision strengthening conditioning dependence. Every version either damaged normal action learning and closed-loop performance, or preserved performance and left the source dependence intact. No clean fix exists. They ship per-step resampling as an evaluation-time band-aid and say so.

### Zero-shot transfer

The unmodified Minecraft checkpoint dropped into VoxeLibre (a Minetest clone) with only a deterministic key-mapping: 71/120 episodes, 59.2%. Chop Tree 85%, Place Block 80%, Mine Stone 70%, Kill Zombie 60%, Mine Iron Ore 50%, **Kill Cow 10%**. GUI tasks were excluded because the interfaces differ too much.

## Worth Remembering

- **The efficiency number is the most quotable result.** 138 steps vs 287–398 for everyone else. Fewer actions per success means less flailing — the model commits to a direction and follows through. That is what "predict long, execute short" buys.
- **The step metric is only computed over successful episodes**, which makes it soft. A model that fails every hard task and succeeds only on trivially short ones would look great. Read it next to ASR, always.
- Table 1 is a **system-level comparison, not a controlled one**. The baselines differ in training data, model class, action representation, and available inputs. GameWAM is not pretrained across many games (the "Game PT." column is ✗); Game-TARS is, at 200× the token budget.
- **The video loss is doing the work.** If you take one engineering lesson: adding a future-frame prediction head to a policy is worth ~15 points here, more than any architectural trick in the paper. Compare the world-model line — [[Mastering Diverse Domains through World Models (DreamerV3)|DreamerV3]], [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)|Dreamer]] — where dynamics prediction is also the core supervision, though there it drives imagined rollouts rather than direct action generation.
- **Decoupling the noisy modalities is both better and cheaper.** That is unusual. Normally cross-modal attention is the thing you add for quality. Here it is a 4-point ASR loss and a 1.54× slowdown. Worth checking whether the same asymmetry shows up in robot WAMs.
- **LASI is a general warning for any diffusion/flow policy.** If you generate action chunks by integrating a noise tensor, the noise has structure along the time axis, and that structure leaks into the coarse shape of your trajectory. Do not cache or reuse the seed across control steps. The effect concentrated in camera motion, probably because camera trajectories in the training data are dominated by smooth low-frequency motion — so the model learned to route smooth source structure straight into smooth output. Anything with a naturally smooth continuous action (a robot arm base, a steering wheel) could show the same thing.
- **Admitted limits:** no symbolic planner, no recipe representation, no inventory model — GUI crafting is done purely as low-level cursor control learned from scripted demos. The history is purely visual; nothing supervises *which* memory to retrieve for the current task. All evaluation is in simulated games. MCU tasks are atomic; nothing here tests genuinely long chains of dependent decisions.
- Open question: they never ablate the memory *design* (two timescales, half-lives 2 and 4, 4 slots). Given history contributes only 4 points and hurts embodied tasks, it is unclear this hierarchy is earning its complexity.

## Links

Related: [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)]] · [[Hydra-0- Action Flow for Generalist World Modeling and Control]] · [[Latent Action as Intention Enables Efficient Future Imagination for World Action Models]] · [[GigaBrain-0.7- Scaling Embodied Foundation Models to Emergent Capabilities with a Three-System Architecture]] · [[τ_0-VLA- a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation]] · [[Game2World Engine- Unlocking In-the-Wild Gameplay Videos for World Model Training]] · [[WorldMind- Decoupled Game World Model for State-Aware NPC Behavior]] · [[PAWBench- How Far Are We from Probabilistically Aligned World Modeling]] · [[Denoising Diffusion Probabilistic Models]] · [[Score-Based Generative Modeling through SDEs]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Attention Is All You Need]] · [[Causal Attention]] · [[Fourier Series Decomposition]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models]] · [[Mixed Precision Training]] · [[EXIMO- VLM Guided Exploration of VLA Policies]]

New topics worth writing: Flow matching (Lipman et al.), Diffusion Transformer (DiT), Diffusion Policy and action chunking, Genie / latent action models, VPT and behavioural cloning from unlabelled video, MineDojo & MineStudio, ViZDoom, Discrete Cosine Transform, Model Predictive Control and receding-horizon planning, KV cache management for long-context inference
