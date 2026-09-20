---
title: "Zero-WAM: In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization"
authors: ["Jiaming Zhou", "Qihang Zhang", "Gangwei Xu", "Cunxin Fan", "Yujie Zhao", "Ruilin Wang", "Yiming Luo", "Shuai Yang", "Xing Zhu", "Yujun Shen", "Junwei Liang", "Yinghao Xu"]
year: 2026
arxiv: "2608.26103"
url: https://arxiv.org/abs/2608.26103
priority: Good-To-Read
read_on: 2026-09-05
tags: [paper, transformers, llm, vision]
---
## The Core Idea

A robot policy trained on 43 tasks usually fails on task 44. The fix people reach for is more data or fine-tuning. This paper borrows the trick that made large language models useful instead: [[In Context Learning|in-context learning]] — you tell the model the new task at inference time and change no weights.

The twist is what counts as "telling". For text, the context is a prompt. For manipulation, language is a bad interface: "put the seal on the pad, stamp it, then move it back" underspecifies where things go, what the intermediate states look like, and how fast. **The natural prompt for a manipulation task is a video of a human doing it.** The video literally shows the desired sequence of world states.

Two things blocked this before:

1. **No paired data.** To teach a policy "watch this human video, then act", you need human videos paired with robot trajectories that have real executable joint commands. Hand-collected paired datasets top out around 100–350 task categories (RH20T: 147 tasks; EgoScale: 344).
2. **A shortcut.** If you train a model to predict the next robot video chunk given (human video, robot history, text), the next chunk is almost always guessable from the robot history alone. The model learns to ignore the human video during training, then has nothing to fall back on at test time when the video is the *only* source of task info. This is textbook [[Shortcut Learning in Deep Neural Networks|shortcut learning]].

Zero-WAM attacks both. For data, it runs the pairing pipeline **backwards**: take an existing robot trajectory (which already has actions), and *generate* a matching human video from it using a VLM + image editor + video generator. That gives **HumanGen**: 74.2K human–robot in-context pairs over **8.6K tasks**, ~25× the task coverage of anything hand-collected, across 45+ robot embodiments. For the shortcut, it adds an auxiliary loss (IFP) that forces the current representation to also predict video chunks *far* in the future — which robot history cannot supply, so the model must read the human video.

Result: on 7 held-out RoboTwin 2.0 tasks, **46.95% success vs 17.45%** for the strongest video-action baseline, with zero task-specific robot data and zero parameter updates at deployment.

> [!NOTE] In-context task specification
> Instead of a language string, the task is specified by a demonstration video placed in the model's context window. The policy must map "what the human's hands did to the objects" onto "what my arms should do", across a gap in embodiment, viewpoint, and background. ^in-context-task-spec

## The Methodology

**Backbone.** Start from `Wan-2.2-TI2V-5B`, an off-the-shelf bidirectional text/image-to-video generator, and convert it into a causal (left-to-right) video-action model. Video transformer: hidden dim 3072, 30 layers. This follows the LingBot-VA recipe.

**The chunked prediction problem.** A trajectory is cut into aligned pairs $\tau = \{(\mathbf{x}^i, \mathbf{a}^i)\}$ — a short video chunk and the action chunk that produced it. At every step, predict the next of both:

$$p_\theta(\mathbf{x}^{i+1}, \mathbf{a}^{i+1} \mid \mathbf{x}^{\leq i}, \mathbf{a}^{\leq i}, c)$$

Factorised into: predict the future video first, then decode actions from it.

$$p_\theta^{\mathrm{vid}}(\mathbf{x}^{i+1} \mid \mathbf{x}^{\leq i}, \mathbf{a}^{\leq i}, c) \cdot p_\theta^{\mathrm{act}}(\mathbf{a}^{i+1} \mid \mathbf{x}^{\leq i}, \mathbf{a}^{\leq i}, \mathbf{x}^{i+1}, c)$$

The action head is basically an **inverse dynamics model**: "I know where the scene should be one chunk from now; what joint commands get me there?" This is the whole bet of world-action models — generalising to an unseen task becomes generalising *video generation*, which video models are already good at, rather than generalising actions, which policies are bad at.

**Mixture-of-Transformers.** Video and action each get their own QKV projections, FFNs, and output heads, but live in one token sequence and talk only through the shared [[Attention Is All You Need|attention]] layers. Action tokens sit *after* the future-video tokens so they can attend to the predicted future. Action transformer is dim 3072, initialised from the video branch.

**Both losses are [[Denoising Diffusion Probabilistic Models|flow matching]].** With clean target $\mathbf{x}_0$, noise $\bm\epsilon$, time $t \in [0,1]$:
$$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\bm\epsilon, \qquad \mathbf{v}_t^\star = \bm\epsilon - \mathbf{x}_0$$
$$\mathcal{L}_{\mathrm{fm}} = \mathbb{E}\big[\|\mathbf{v}_\theta(\mathbf{x}_t, t, c) - \mathbf{v}_t^\star\|_2^2\big]$$
The action chunk gets an identical objective in action space with its own flow time $r$.

**Where the human video goes.** The human video $\mathbf{h}$ is encoded by the *same* Wan-2.2 VAE as the robot video and prepended as prefix memory. Video prediction conditions on $[\mathbf{h}, \mathbf{x}^{\leq i}]$. The action head does **not** see $\mathbf{h}$ — the reasoning being that all task semantics have already been absorbed into the predicted next robot frame, so action decoding stays plain inverse dynamics.

**Telling human from robot tokens.** Both live in one latent space, so how does the model know which is which? A [[RoFormer- Enhanced Transformer with Rotary Position Embedding|RoPE]] offset. Robot latents keep coordinates $(q, y, x)$; human latents get $(q, y + \Delta_H, x)$ with $\Delta_H = 32 > H_{\mathrm{mv}}$, pushing them outside the robot coordinate range entirely. Cheap and clean.

### In-context future chunk prediction (IFP)

The anti-shortcut loss, and the single biggest win in the paper.

Add $K=4$ auxiliary modules $\{G_k\}$, each architecturally a copy of one video transformer layer, initialised from the last one. Each predicts a *strided* future chunk at index
$$j_k = (i+1) + 1 + (k-1)s, \qquad s = 2$$
so targets sit 2, 4, 6, 8 chunks ahead. Loss weights $(w_1,\dots,w_4) = (0.5, 0.25, 0.15, 0.15)$.

The critical design detail: **the IFP modules are never conditioned on $\mathbf{h}$ directly.** Their only input is $\bm\phi^{i+1}$, a fused feature made by concatenating the hidden states of the current chunk from $M$ intermediate layers of the *main* video transformer and projecting them down with a small MLP:
$$\bm\phi^{i+1} = P_{\mathrm{fuse}}\big(\mathrm{Concat}(\{\mathbf{r}_m^{i+1}\}_{m=1}^M)\big)$$
$$\mathcal{L}_{\mathrm{ifp}} = \sum_{k=1}^{K} w_k \,\mathcal{L}_{\mathrm{fm}}\big(\mathbf{x}^{j_k}; \bm\phi^{i+1}, \mathbf{x}^{\leq i}, \mathbf{a}^{\leq i}, \ell\big)$$

If $G_k$ could see the human video, it would just build its own private video-conditioned predictor, and since IFP is deleted at inference, the shipped policy would gain nothing. Routing through $\bm\phi^{i+1}$ means the only way to lower $\mathcal{L}_{\mathrm{ifp}}$ is to make the **main branch** stuff long-horizon task information into its current representation — and that information can only come from $\mathbf{h}$.

> [!NOTE] IFP
> A training-only auxiliary head that predicts strided far-future video chunks from the main branch's current hidden state. Removed at inference. Its job is not accuracy — it is to make ignoring the prompt expensive. ^ifp

Full ICL loss: $\mathcal{L}_{\mathrm{ICL}} = \mathbb{E}[\mathcal{L}_{\mathrm{fm}}^{i+1}(c) + \lambda_a \mathcal{L}_a^{i+1}(\ell) + \lambda_{\mathrm{ifp}} \mathcal{L}_{\mathrm{ifp}}]$, with $c = \{\mathbf{h}, \ell\}$.

### The data pipeline

**Task-diverse VA.** Public robot corpora (AgiBot, InternData-A1, Open-X-Embodiment, RoboCOIN, RoboMIND) are dominated by repeated teleoperation of the *same* task. So they re-partition each dataset by task (task = action + object, from metadata or parsed from trajectories) and sample a bounded number of trajectories per task. Yields >6,000 tasks, ~400K trajectories per epoch.

**HumanGen generation**, per robot video:
1. VLM (Gemini 3.1 Pro / Qwen3.6-Plus) reads the robot video → task name, initial object states, state changes, final states, plus an image-editing prompt to turn frame 1 into a *human* scene.
2. Image editor (Nano Banana 2 / Qwen-Image-2.0) makes the initial human observation image.
3. VLM writes a video-generation prompt describing how the hands should move.
4. Video model (Wan 2.7 / Kling AI 3.0) synthesises the human video.
5. VLM grades it for task-semantic preservation and physical plausibility; failures are dropped.

Deliberate variation is injected at step 1 — background, viewpoint (ego vs third-person), environment style, object instance, object placement — so the model cannot succeed by copying pixel motion and must learn *task-level* correspondence.

Subsets: Pre-train External (5,062 tasks, 41,188 pairs, heavy visual variation), Pre-train In-house (3,522 tasks, 30,247 pairs, *less* variation — more visually aligned), Simulation ICL (50 RoboTwin tasks × 50 = 2,500), Real-world ICL (252 pairs on bimanual Franka).

**Training.** [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], peak LR $10^{-4}$, weight decay 0.01. Pre-training samples Task-diverse VA : HumanGen at **1:5**. Language dropout 0.1 for non-ICL samples; for ICL samples the human-video latent is dropped 10% of the time and **language dropout is raised to 0.4** to stop the model leaning on text. Robot video chunk size randomly sampled 1–4. Sequence packing to 160K tokens/GPU. Total: **15,360 GPU hours**. RoboTwin post-training: 64 GPUs, 4,000 steps, sampling ratio VA : HumanGen : RoboTwin = 2:10:3.

**Inference.** IFP modules removed. Human video encoded once, tokens cached as prefix. Two modes: language-only (video [[Classifier-Free Diffusion Guidance|CFG]] scale 5) or ICL mode (guidance scale 5 on the video condition, language disabled). Inference chunk size 2, action CFG 1.0.

## Ablation Studies and Experiments

**Setup.** RoboTwin 2.0, 50 bimanual tasks split **43 seen / 7 unseen** at the *task* level. 3 seeds × 100 closed-loop rollouts per task. Baselines: **WAN-Action** (same Wan-2.2 init, same MoT causal framework, trained only on the 43 seen tasks, language-conditioned) and **LingBot-VA** (has its own robotic VA pre-training, post-trained on the same 43 tasks, language-conditioned).

| Unseen task | WAN-Action | LingBot-VA | Zero-WAM |
|---|---|---|---|
| Place object on scale | 3.00 | 6.17 | **24.67** |
| Stamp seal | 7.33 | 3.67 | **47.00** |
| Open microwave | 2.26 | 29.33 | **59.00** |
| Move stapler to pad | 10.67 | 23.33 | **69.14** |
| Place bread in basket | 15.26 | 17.33 | **35.00** |
| Place empty cup | 38.33 | 42.33 | **84.87** |
| Stack blocks three | 0.00 | 0.00 | **9.00** |
| **Average** | **10.98** | **17.45** | **46.95** |

Wins on all seven, so it is not one lucky task. `stack blocks three` (long-horizon) is the only place anyone is near zero, and Zero-WAM is the only method above zero.

**Ablation 1 — does the human video itself help?** Strip away all pre-training; train everything only on the 43 seen RoboTwin tasks from Wan-2.2 weights. Text-only (= WAN-Action): **10.98%**. Same setup but with human video prompts: **36.36%**. So the ICL interface alone is worth ~25 points, and beats LingBot-VA (17.45%) *despite* LingBot-VA having full robotic pre-training. But all three of these small-scale variants score **0 on `stack blocks three`** — the long-horizon task only cracks with large-scale ICL pre-training.

**Ablation 2 — does IFP matter?** Full Zero-WAM minus the IFP loss: **28.55% → 46.95%** with it. Nearly **18 points from an auxiliary loss that is deleted before deployment.** Biggest gains on `open microwave` (unseen articulated dynamics) and `stamp seal` (unseen object relocation), and it is what turns `stack blocks three` from 0.00% into 9.00%. This is the strongest evidence that the shortcut is real and that killing it is where the generalisation comes from.

**Ablation 3 — is it just the rebalanced data?** Build a text-only Zero-WAM: keep the ICL samples in pre-training but **mask the human-video condition**, so each pair degrades to an ordinary text-conditioned VA sample. This isolates the effect of task-balanced sampling. Result: **39.44%** vs LingBot-VA's 17.45% — **+21.99 points from re-partitioning and task-level sampling alone**, on largely the same underlying source corpora LingBot-VA used. Then human-video ICL + IFP adds the remaining ~7.5 points to reach 46.95%.

Read together, the credit split is roughly: task-balanced data ≈ 22 points, ICL + IFP ≈ 7.5 points on top. The data curation is doing more heavy lifting than the headline framing suggests — though the ICL interface is what makes real-world fine-grained specification possible at all.

**Real world**, bimanual Franka, 30 trials each. LingBot-VA gets detailed *text* instructions; Zero-WAM gets *only* a human video, no language.

| Task family | Train combos / demos | LingBot-VA | Zero-WAM |
|---|---|---|---|
| Object-to-container placement | 30 / 120 | 43.3 | **53.3** |
| Three-object sequential manipulation | 16 / 96 | 10.0 | **33.3** |
| Two-table-leg insertion | – / 36 | 0.0 | **16.7** |

The middle row is the interesting one: the human video specifies an *arbitrary order* over three objects. Language would need a careful sentence; the video just shows it. The insertion task is where language breaks completely — "which coloured leg goes in which hole" — and LingBot-VA scores exactly zero.

## Worth Remembering

- **The pipeline direction is the trick.** Everyone else collects human videos and struggles to get matching robot actions. Here you start from robot trajectories that *already have* actions and synthesise the human video. The actions are ground truth; only the prompt is fake. That is the right thing to fake.
- **Two different visual-alignment budgets on purpose.** External pre-train pairs get *maximum* mismatch (scene, background, objects, placement, viewpoint all changed) to force embodiment-invariant task correspondence. In-house pairs get *less* mismatch for easier learning. That is a curriculum hidden in the data spec.
- **IFP is a general anti-shortcut pattern**, not a robotics idea. Whenever a context signal is redundant with recent history on training data, add an auxiliary head predicting something history *cannot* supply, and route it strictly through the main model's hidden state so the main model has to do the work. Delete the head at inference. Reusable.
- **The action head never sees the prompt.** Slightly surprising. All the task information has to survive the bottleneck of the generated future video frame. If video generation degrades, actions degrade with no fallback.
- **Cost.** 15,360 GPU hours for pre-training, plus the API bill for VLM + image edit + video gen across 74.2K samples using frontier models (Gemini 3.1 Pro, Nano Banana 2, Kling 3.0). This is not reproducible on a small budget.
- **Absolute numbers are still low.** 47% average, 9% on the long-horizon task, 16.7% on insertion. This is "sometimes works", not "deployable".
- **Circularity risk the authors do not discuss.** The human videos are generated by a video model, and Zero-WAM is fine-tuned from a video model of the same family (Wan). Are the prompts easy to follow partly because they are in-distribution for a video generator? Evaluation with *real* human demonstration videos would settle it — the real-world experiments seem to use collected pairs, but the failure mode is worth flagging.
- **Distinct from test-time training approaches** (WAM-TTT, RoboTTT), which also handle unseen tasks but need memory or fast-weight updates at deployment. Zero-WAM's contribution is pushing that cost entirely into pre-training so inference is a single forward pass over a cached prefix.
- Scope: stationary tabletop manipulation only. No mobile manipulation, no truly long horizons.

## Links

Related: [[In Context Learning]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[Shortcut Learning in Deep Neural Networks]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[GameWAM- A World Action Model for Video Games]] · [[Hydra-0- Action Flow for Generalist World Modeling and Control]] · [[Latent Action as Intention Enables Efficient Future Imagination for World Action Models]] · [[GigaBrain-0.7- Scaling Embodied Foundation Models to Emergent Capabilities with a Three-System Architecture]] · [[τ_0-VLA- a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation]] · [[EXIMO- VLM Guided Exploration of VLA Policies]] · [[Denoising Diffusion Probabilistic Models]] · [[Classifier-Free Diffusion Guidance]] · [[RoFormer- Enhanced Transformer with Rotary Position Embedding]] · [[Attention Is All You Need]] · [[Causal Attention]] · [[Auto-regressive models]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Exploring the Limits of Transfer Learning (T5)]] · [[Sparsely-Gated Mixture-of-Experts Layer]]

New topics worth writing: Flow matching, Inverse dynamics models, Mixture-of-Transformers (per-modality parameters, shared attention), Vision-Language-Action models, RoboTwin 2.0 benchmark, Teacher forcing and exposure bias, Synthetic data generation pipelines with VLM verification, Auxiliary losses as shortcut suppression, One-shot imitation learning
