---
title: "ProAR: Learning Prospective Reasoning with Autoregressive Video Models"
authors: ["Linghui Shen", "Tinghui Zhu", "Sheng Zhang", "Muhao Chen"]
year: 2026
arxiv: "2610.03664"
url: https://arxiv.org/abs/2610.03664
priority: Good-To-Read
read_on: 2026-10-05
tags: [paper, transformers, diffusion, vision]
---
## The Core Idea

Autoregressive video models generate a video chunk by chunk. Each new chunk is predicted from the chunks already written. That works well for making pretty video. It works badly when the video *is* the answer to a problem — solve a maze, slide puzzle tiles into order, grasp an object with a robot arm.

Why? Because next-chunk prediction only asks "does this look like a plausible continuation?" It never asks "does this get me to the goal?" Each chunk is frozen once written — there is no going back. So a small wrong turn at chunk 2 survives into chunk 3, and the rollout drifts somewhere locally plausible and globally wrong.

The paper calls the missing ability **prospective reasoning**: use a guess about the future to steer what you generate now. The claim is that this needs supervision at two different time scales, and that the two are not substitutes for each other.

> [!NOTE] Prospective reasoning
> Anticipating both the final outcome and the next state transition from the history so far, and using those anticipations to constrain the current generation step. ^prospective-reasoning

Two mechanisms deliver it:

1. **Outcome guidance** — at *every* autoregressive step, the model also predicts the final frame of the video. An asymmetric attention mask lets this predicted goal inform the current chunk, but blocks the noisy current chunk from contaminating the goal. Sparse, explicit, long-range.
2. **Transition guidance** — the hidden states of the current (noisy) chunk are pushed to look like the hidden states of the *clean next* chunk. Dense, implicit, short-range.

The trick that makes (2) nearly free is a property of teacher forcing. During AR training with teacher forcing, every chunk is denoised in parallel, each conditioned on clean ground-truth history. So the clean representation of chunk $i{+}1$ is *already sitting in the same forward pass* — it is in the history stream used to condition later targets. No extra encoder like [[DINOv2- Learning Robust Visual Features|DINOv2]], no second backbone pass. The closest prior work (Video-Mirai) needed an extra forward pass; this does not.

What it unlocks: on a 10-task visual reasoning suite, mean score goes 0.663 → 0.801, and the method beats the fully-trained standard AR baseline using **25% of the training steps**.

## The Methodology

### The base model

Backbone is Wan2.2-TI2V-5B, a 30-block DiT ([[Diffusion Transformer|diffusion transformer]]), converted to autoregressive generation with a causal mask and teacher forcing. The video is $N$ latent chunks $\mathbf{z}^{1:N}$, factorised the usual [[Auto-regressive models|autoregressive]] way:

$$p_\theta(\mathbf{z}^{1:N}\mid \mathbf{c}) = \prod_{i=1}^{N} p_\theta(\mathbf{z}^{i}\mid \mathbf{z}^{<i}, \mathbf{c})$$

Each conditional is trained with [[Flow Matching|flow matching]]. Noisy state is a straight-line interpolation $\mathbf{z}^i_t = (1-t)\mathbf{z}^i_0 + t\bm{\epsilon}$, and the velocity net is regressed onto the target velocity:

$$\mathcal{L}_{\mathrm{AR}} = \mathbb{E}_{i,t,\bm{\epsilon}}\big[\|\mathbf{v}_\theta(\mathbf{z}^i_t, t \mid \mathbf{z}^{<i}_0, \mathbf{c}) - (\bm{\epsilon} - \mathbf{z}^i_0)\|_2^2\big]$$

Note the train/test asymmetry, which is the whole problem: training conditions on **clean ground-truth** history; inference conditions on **its own generated** history, unrevisable. This is [[Imitation Learning#^errors-compound|exposure bias]] wearing video clothes.

### Outcome: the goal belief

At step $i$, two streams get noised at the *same* timestep $t$ with independent noise:

$$\mathbf{z}^t_i = (1-t)\mathbf{z}^0_i + t\bm{\epsilon}_{z,i}, \qquad \mathbf{g}^t_i = (1-t)\mathbf{g}^0 + t\bm{\epsilon}_{g,i}$$

where $\mathbf{g}^0$ is the clean ground-truth **final frame** of the video. One shared backbone predicts velocities for both in one pass. The goal loss is the same flow-matching regression:

$$\mathcal{L}_{\mathrm{goal}} = \mathbb{E}\big[\|\mathbf{v}^g_\theta(\mathbf{g}^t_i, t \mid \mathbf{z}^{<i}_0, \mathbf{c}) - (\bm{\epsilon}_{g,i} - \mathbf{g}^0)\|_2^2\big]$$

Every AR step is supervised by the *same* final frame. So the goal prediction is a belief that gets re-estimated as the history grows — early on it is a guess from one input image, later it is a near-certainty.

The attention mask is the load-bearing part:

- **Goal tokens** attend to: clean history + themselves. **Not** the noisy current chunk.
- **Current tokens** attend to: clean history + themselves + goal tokens.

> [!NOTE] Asymmetric goal mask
> A one-directional attention edge. The uncertain present cannot corrupt the goal belief, but the goal belief can guide the present. Also strictly sparser than full attention, so it is cheaper. ^asymmetric-goal-mask

At inference, both streams are denoised, but only the current chunk is appended to the history. The goal frame is thrown away and re-predicted next step.

### Transition: future representation self-alignment

Pick backbone layer $\ell = 15$ (of 30). Two sets of hidden states at that layer:

- **Teacher** $\mathbf{H}^\ell_{i+1}$ — hidden states of the *clean* chunk $i{+}1$, already computed because it sits in the history stream conditioning later chunks.
- **Student** $\mathbf{C}^{t,\ell}_i$ — hidden states of the *noisy* current chunk $i$.

A training-only predictor $\mathcal{P}_\phi$ (3 DiT blocks, 236M params, 4.7% of the backbone) maps student → teacher, with the teacher stop-gradiented:

$$\overline{\mathbf{H}}^\ell_{i+1} = \mathrm{sg}(\mathbf{H}^\ell_{i+1}), \qquad \widehat{\mathbf{H}}^{t,\ell}_{i+1} = \mathcal{P}_\phi(\mathbf{C}^{t,\ell}_i)$$

Loss is token-wise cosine distance over the $M$ tokens in a chunk:

$$\mathcal{L}_{\mathrm{align}} = \mathbb{E}_{i<N,t}\Big[\tfrac{1}{M}\sum_{m=1}^{M}\big(1 - \cos(\widehat{\mathbf{H}}^{t,\ell}_{i+1,m}, \overline{\mathbf{H}}^\ell_{i+1,m})\big)\Big]$$

The predictor preserves the token grid and width — no compression — so alignment is spatial, position by position. The stop-gradient-plus-predictor shape is exactly the [[Bootstrap Your Own Latent (BYOL)|BYOL]] / [[Exploring Simple Siamese Representation Learning (SimSiam)|SimSiam]] asymmetry, and the "predict a future representation, not future pixels" framing is [[JEPA]] in spirit — except the teacher here is the model's own clean-stream features, not a momentum copy.

### Total objective

$$\mathcal{L} = \mathcal{L}_{\mathrm{current}} + \lambda_{\mathrm{goal}}\mathcal{L}_{\mathrm{goal}} + \lambda_{\mathrm{align}}\mathcal{L}_{\mathrm{align}}$$

$\lambda_{\mathrm{goal}} = 0.25$, $\lambda_{\mathrm{align}} = 0.01$. The raw cosine loss is much bigger than the generation losses, so even at 0.01 the alignment term is 3–10% of the total.

### Hyperparameters that mattered

| Setting | Value |
|---|---|
| Optimiser | AdamW, LR $5\times10^{-6}$, global batch 16 |
| Chunk size | 4 frames (VBVR, VideoRLVR); 8 (WorldArena) |
| Alignment layer | block 15 of 30 |
| Predictor depth | 3 DiT blocks |
| Transition start step | 7,500 / 10,000 (VBVR); 3,000 / 5,000 (VideoRLVR) |
| Denoising steps | 20 per chunk |
| Hardware | 2–4 NVIDIA B200 |

Cost: training +12–14% total. Inference +12% at chunk size 4 (one extra goal frame per chunk), +~8% at chunk size 8. The predictor is deleted entirely at inference — zero cost.

## Ablation Studies and Experiments

### VBVR — 10 perceptual/spatial reasoning tasks, score in $[0,1]$

| Model | Mean |
|---|---|
| Seedance 2.0 (closed-source) | 0.627 |
| Wan2.2 5B (no fine-tune, bidirectional) | 0.285 |
| Wan2.2 5B SFT (bidirectional) | 0.736 |
| **Standard AR** | **0.663** |
| w/ Outcome only | 0.786 |
| w/ Transition only | 0.683 |
| **ProAR** | **0.801** |

Biggest per-task jumps over standard AR: Slide Puzzle +0.543 (0.388 → 0.931), Stable Sort +0.330, Grid Shift +0.050, Pipe Puzzle +0.122. Wins on 8 of 10 tasks. It *loses* slightly on Domino Chain (0.658 → 0.628) and Balancing Vessels (0.880 → 0.863) — both are physics-propagation tasks where the final frame may be less informative than the process.

Note the baseline ordering: standard AR (0.663) is **worse** than the bidirectional SFT model (0.736). Causal generation starts behind on these tasks, and the two guidance terms more than close the gap.

### VideoRLVR — 3 abstract games

Here standard AR already beats the bidirectional baselines, including VideoRLVR's own RL-trained variant. Causal generation suits discrete sequential state transitions.

| Model | Avg SR | Avg Precision |
|---|---|---|
| Wan2.2 SFT (bidirectional) | 24.73 | 52.10 |
| VideoRLVR (bidir. + RL) | 28.73 | 53.47 |
| Standard AR | 50.97 | 61.56 |
| w/ Outcome | 49.93 | 64.82 |
| w/ Transition | 52.40 | 63.50 |
| **ProAR** | **52.97** | **67.17** |

Precision rises on all 3 games; Sokoban precision 40.18 → 47.62. Success rate rises on all 3 (Maze 79.50 → 82.70, FlowFree 29.40 → 30.20, Sokoban 44.00 → 46.00).

**The interesting failure:** Outcome-only *drops* FlowFree success rate from 29.40 to 23.20, despite raising precision. FlowFree requires filling the whole grid with non-overlapping colour paths — so the final frame is a dense, hard-to-predict target, and anchoring to it apparently costs you valid intermediate structure. Transition-only is the better single component on Sokoban (SR 49.60, the best number in the whole table for that game — higher than full ProAR's 46.00).

That last point is worth sitting with: **the full method is not the per-task best everywhere.** The two components trade off.

### Training efficiency

ProAR at 2,500 steps scores 0.708 — already above standard AR's 0.663 at 10,000 steps. So ~25% of the budget. After Transition turns on at 7,500, the curve jumps 0.7666 → 0.8010 by step 10,000.

### When to switch Transition on

| Start step | Mean |
|---|---|
| 0 | 0.799 |
| 5,000 | 0.782 |
| 7,500 (default) | 0.801 |

Essentially a wash between 0 and 7,500, and 5,000 is *worse* than both — a non-monotonic result they do not explain. Their hypothesis for the late start: early in training the backbone has no stable task-relevant representations, so its internal teacher targets are uninformative. The evidence for that story is thin; 0.799 vs 0.801 is noise-sized. **Treat "late start is necessary" as unsupported.**

### Asymmetric mask

| | Mean |
|---|---|
| No asymmetric mask (full attention) | 0.799 |
| Asymmetric | 0.801 |

Also a wash on score. It wins on 6 of 10 tasks and is cheaper because it is sparser. The paper's information-flow argument is plausible but **not demonstrated by this number**. The honest read: the mask is free, not load-bearing.

### Alignment layer and predictor depth

| Layer | Mean | | Predictor | Mean |
|---|---|---|---|---|
| 10 | 0.788 | | 2 DiT blocks | 0.792 |
| 15 | 0.801 | | 3 DiT blocks | 0.801 |
| 20 | 0.791 | | | |

All three layers beat Outcome-alone (0.786), so Transition is robust across the middle of the network. Layer choice is worth ~0.013.

### Embodied extension — WorldArena, 50 bimanual RoboTwin 2.0 tasks

EWMScore 60.43 → 60.95, improving 13 of 16 base metrics. Gains concentrate where you would want: Depth Accuracy 88.42 → 91.11, Instruction Following 80.88 → 81.96, Interaction Quality 73.12 → 73.86, Trajectory Accuracy 41.74 → 42.34, Background Consistency 87.10 → 88.11. Image/aesthetic quality roughly flat. Second overall behind FlowWAM (63.71), which uses dense optical flow as its action representation.

A +0.52 EWMScore is small, and EWMScore is a 16-metric arithmetic mean — so it dilutes real gains in the task-relevant dimensions. Read the component metrics, not the aggregate.

### Representation visualisation

PCA of layer-15 features at $\sigma=0.5$, projected with a shared basis. The noisy current-chunk features show the *current* spatial layout; after passing through the trained predictor, they shift toward the clean next-chunk features — recovering next-chunk object positions. Evidence that the predictor is not just denoising in place.

## Worth Remembering

**The limitation the authors name, and it is the real one.** Outcome guidance hard-codes "goal = final frame." When the last frame is uninformative — idle state after task completion, or a purely process-driven dynamic like liquid settling — the supervision is weak. This matches the two VBVR tasks where ProAR loses (Domino Chain, Balancing Vessels: both process tasks). Their proposed fix is adaptive milestone/keyframe goals, which is unimplemented.

**The cheap idea worth stealing.** Teacher forcing gives you clean future representations for free, inside the same forward pass. If your architecture trains with clean ground-truth conditioning, any future-prediction auxiliary loss can be had without a second pass or an external encoder. This generalises well beyond video.

**Which component does the work.** Outcome is clearly the bigger lever on VBVR (+0.123 alone, vs +0.020 for Transition). But Transition wins alone on Sokoban and never hurts. If you had to pick one, pick Outcome; the combination adds ~0.015 more.

**The ablations that came out flat are more informative than the ones that worked.** Asymmetric mask: +0.002. Transition start step: ±0.002 across 0 and 7,500. Both are presented as design choices justified by mechanism, but the numbers do not separate them from the alternatives. The two components themselves are well separated; the internal design details are not.

**Caveats for using this.**
- No seeds, no error bars, anywhere. The 0.663 → 0.801 jump is large enough to survive that; the 0.799 vs 0.801 comparisons are not.
- VBVR scores are *continuous task-completion scores*, not success rates. Do not read 0.801 as "80% solved."
- VideoRLVR precision/F1 measure overlap with a reference solution; SR measures strict completion. High F1 with low SR happens (FlowWAM-style: FlowFree F1 59.52 but SR 30.20). Always read SR.
- Test sets for VBVR were expanded by the authors from 5 → 50 examples per task using the official generator. Reasonable, but not the official split, so the numbers are not directly comparable to published VBVR results.
- Inference overhead shrinks as chunk size grows, since it is one goal frame per chunk regardless. At chunk 4 it is 12%; at chunk 8, ~8%.

**Open question.** Outcome guidance supervises every step against the same final frame, which means early steps are asked to predict a goal they cannot possibly know from one input image. Why does that not just inject noise? Likely the model learns to output a broad, low-confidence goal estimate early — but nobody measured the quality of the goal prediction as a function of step index. That measurement would tell you whether the goal belief is actually refining or whether the whole gain comes from an auxiliary-task regularisation effect.

## Links

Related: [[Auto-regressive models]] · [[Flow Matching]] · [[Diffusion Transformer]] · [[JEPA]] · [[Bootstrap Your Own Latent (BYOL)]] · [[Exploring Simple Siamese Representation Learning (SimSiam)]] · [[Imitation Learning]] · [[Video Diffusion]] · [[Diffusion Policy]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Causal Attention]] · [[Credit Assignment]] · [[Planning and Decomposition]] · [[Reward Function]] · [[Self-Supervised Learning from Images with I-JEPA]]

New topics worth writing: video-native reasoning, teacher forcing in diffusion training, goal-conditioned generation, representation alignment as an auxiliary loss (REPA family), VBVR benchmark, WorldArena / embodied world-model evaluation, keyframe and milestone goal selection, Wan2.2 backbone, chunk-wise autoregressive video rollout
