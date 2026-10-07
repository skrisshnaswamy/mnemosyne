---
title: "ForgeWM: Progressive Causal Training for Few-Step Action-Conditioned Video World Models"
authors: ["Xinye Li", "Lingshuai Lin", "Lei Wang", "Liuzhou Zhang", "Jialin Cui", "Qingshan Li", "Guanchu Wang", "Qingbin Liu", "Xi Chen", "Jiang Bian", "Wai Lam"]
year: 2026
arxiv: "2608.14022"
url: https://arxiv.org/abs/2608.14022
priority: Low-Priority
read_on: 2026-09-30
tags: [paper, transformers, diffusion, vision]
---
## The Core Idea

A **video world model** is a video generator you can drive. You give it a first frame plus a stream of controls — WASD keys, mouse movement, gamepad sticks — and it paints what happens next. If it is fast enough, you can play it like a game with no game engine underneath.

The problem is speed. Diffusion models paint by denoising, and a normal one needs 20–50 denoising steps per chunk of frames. That is far too slow to close a control loop. People already know how to squeeze a diffusion model down to 1–4 steps (consistency distillation, distribution matching). And people already know how to make video generation **causal** — each chunk only looks at past chunks, so you can stream instead of generating a whole clip at once.

What ForgeWM is about is doing both at once *without losing the controls*.

Why that is hard, concretely. Two things drift apart during this conversion:

1. **The history changes.** During training the model sees clean, real past frames. At play time it sees its own imperfect output. Fewer denoising steps means more error per chunk, and that error feeds the next chunk.
2. **The controls must stay glued to the latents.** The video autoencoder squashes 4 video frames into 1 latent frame, and a causal chunk is 3 latents = 12 video frames. But keyboard states and mouse deltas arrive at *frame* rate, 12 per second. Every step of adaptation, distillation and rollout has to keep the right 12-frame action window pointing at the right chunk, and has to keep three separate key–value caches in sync (visual, keyboard, mouse).

The answer is a **four-stage ladder** where each stage changes exactly one thing: first the game domain, then the attention pattern (bidirectional → causal), then the step count (many → few), then the history distribution (clean → self-generated). Same action interface throughout. Out come three separate checkpoints specialised for 1, 2 and 4 denoising steps.

The second, smaller idea is nicer and more portable. Once you have a fast student, you get a free quality upgrade at **replay** time. The player interacts at 1 step, and the rollout is saved. Afterwards, the *same* one-step model adds a bit of noise back to its own saved draft ($r=0.3$) and re-denoises it with 4 steps. This lands at four-step quality (LPIPS 0.6155 vs 0.6168) while staying ~3× closer to what the player actually saw ($D_{\text{draft}}$ 0.197 vs 0.619 for regenerating from fresh noise). No second model, no online cost.

> [!NOTE] Steady-state denoising budget
> "1-step" here means 1 denoising pass per chunk *after the first chunk*. The 1- and 2-step students still spend 4 passes on the very first generated chunk, because errors there poison the whole rollout. So the name describes the sustained cost, not the total. ^steady-state-budget

## The Methodology

### The base model and objective

Start from Matrix-Game 2.0's image-to-video lineage — a Wan2.1-T2V-1.3B backbone. A frozen VAE turns the video into latents $z$. The first frame enters twice: channel-wise concatenation onto the latents, and a separate visual-context branch.

Training is [[Flow Matching]]. Noise level $\sigma \in [0,1]$ is the path coordinate:

$$z_\sigma = (1-\sigma)z + \sigma\epsilon, \qquad v^*(z_\sigma,\sigma) = \epsilon - z$$

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{z,\epsilon,\sigma}\!\left[w(\sigma)\,\|v_\theta(z_\sigma,\sigma;c) - v^*(z_\sigma,\sigma)\|_2^2\right]$$

The network predicts a velocity, but every later stage compares **clean predictions** instead:

$$\hat z_\theta(z_\sigma,\sigma) = z_\sigma - \sigma\, v_\theta(z_\sigma,\sigma)$$

This single reparameterisation is what lets consistency distillation and distribution matching both plug into the same network — they are both statements about $\hat z$.

### The action interface

Two pathways, deliberately kept separate and never collapsed into a camera pose:

- **Discrete keys** ($k$, a 6-dim vector) become cross-attention keys and values.
- **Continuous mouse** ($m = (\Delta u, \Delta v)$) is windowed, concatenated with the visual hidden state, pushed through an MLP, then handled by temporal attention.

The input width of that mouse MLP is $1536 + \texttt{mouse\_dim\_in} \times 4 \times 3 = 1560$ — that is 1536 visual hidden units, plus (control channels × VAE temporal compression × window). This one number is the whole action-to-latent alignment made explicit.

During rollout the model carries three caches: visual KV, keyboard, mouse. Same alignment and same cache-update protocol in training, distillation, and inference. That consistency is the paper's main engineering discipline.

### Stage 0 — bidirectional domain adaptation

Plain $\mathcal{L}_{\text{FM}}$, full-clip bidirectional temporal attention over 21-latent clips (= 84 frames). 4k updates, generator LR $2\text{e}{-6}$. This checkpoint is **frozen and kept** — it becomes the "real denoiser" in Stage 3.

### Stage 1 — teacher-forced causal training

Branch from the *same base*, not from Stage 0. Swap full temporal attention for block-wise causal attention: inside a 3-latent chunk frames see each other; across chunks, chunk $i$ only sees $j < i$.

The trick is the mask. Concatenate the clean and noisy token streams, then mask so noisy chunk $i$ attends to **clean** chunks $j<i$ plus itself. One noise level per chunk:

$$\mathcal{L}_1 = \mathbb{E}\!\left[w(\sigma_i)\,\big\|v_\theta(z_i^{(\sigma_i)},\sigma_i; c_i, z_{<i}) - (\epsilon_i - z_i)\big\|_2^2\right]$$

So the model learns causal *execution* while its history is still exact. 20k updates, LR $2\text{e}{-5}$ (10× the other stages — this is the stage doing real representation surgery).

### Stage 2 — causal consistency distillation, online

Three copies of the Stage 1 checkpoint: trainable generator, its EMA copy (decay 0.99, starting at step 200), and a frozen teacher.

Take a discrete grid of $N=48$ noise levels $\sigma_0 > \dots > \sigma_{47}$. Sample one adjacent pair. The frozen teacher takes one classifier-free-guided Euler step ($\omega = 3.0$) toward the data end:

$$\tilde z^{(\sigma_{i+1})} = z^{(\sigma_i)} + (\sigma_{i+1}-\sigma_i)\left[v^\emptyset_{\text{tch}} + \omega(v^c_{\text{tch}} - v^\emptyset_{\text{tch}})\right]$$

Then force the student's clean prediction at the *original* level to agree with the EMA's clean prediction at the *advanced* level:

$$\mathcal{L}_2 = \mathbb{E}\!\left[\big\|\hat z_\theta(z^{(\sigma_i)},\sigma_i) - \text{sg}\!\left[\hat z_{\bar\theta}(\tilde z^{(\sigma_{i+1})},\sigma_{i+1})\right]\big\|_2^2\right]$$

Key practical point: **all three networks still condition on clean causal history here.** That is exactly why the teacher's Euler step costs one forward pass instead of an autoregressive unroll, and why no offline ODE-trajectory dataset is needed. 6k updates, LR $2\text{e}{-6}$.

### Stage 3 — on-policy distribution matching

Now let the student roll out on itself. Each chunk conditions on its own generated past. Initialise from Stage 2; the frozen Stage 0 checkpoint is the real denoiser.

Let $\hat z$ be the clean latent from a $K$-step self-rollout, $\hat z^{(\sigma)}$ a re-noised copy. The gradient direction is the disagreement between a frozen real denoiser and a trainable fake one:

$$g = \frac{\hat z_{\text{fake}}(\hat z^{(\sigma)},\sigma) - \hat z_{\text{real}}(\hat z^{(\sigma)},\sigma)}{\text{mean}\big(|\hat z - \hat z_{\text{real}}(\hat z^{(\sigma)},\sigma)|\big)}$$

The denominator is an adaptive normaliser — mean absolute deviation over all elements. Apply $g$ through a surrogate whose gradient w.r.t. $\hat z$ is exactly $g$:

$$\mathcal{L}_3 = \mathbb{E}\!\left[\tfrac12\|\hat z - \text{sg}[\hat z - g]\|_2^2\right]$$

The fake denoiser trains concurrently with its own flow-matching loss on the student's samples. Critic LR $4\text{e}{-7}$; **critic updates every iteration, generator every fifth**. Local attention window of 6 latents (2 chunks). 4k trainer iterations per student, run three times separately for $K \in \{1,2,4\}$.

> [!NOTE] Why Stage 3 exists at all
> Stages 1–2 train on clean history; deployment has dirty history. That gap is exposure bias, exactly the thing [[Imitation Learning]] calls compounding error. Stage 3 fixes it by sampling from the model's own rollout distribution — the same move as DAgger, and the same move [[Auto-regressive models#Exposure bias — the flaw worth knowing|exposure bias]] mitigations make in language models. ^clean-history-mismatch

Everything: AdamW with $\beta = (0.0, 0.999)$ — note $\beta_1 = 0$, i.e. no first-[[Momentum|moment]] memory at all. bf16, gradient checkpointing, FSDP over 8 GPUs, global batch 8, 40k GF-Minecraft clips at $640\times352$, 12 fps, fixed seed 0.

### Replay-time refinement

Given a saved rollout $\hat z_{1:B}$, keep its first frame, actions, and latents. Re-noise each chunk:

$$z_i^{(r_i)} = (1-r_i)\hat z_i + r_i \epsilon_i, \qquad \epsilon_i \sim \mathcal{N}(0,I)$$

then denoise under the recorded action window and the *already-refined* prefix:

$$z_i^{\text{ref}} = \mathcal{R}_\phi^{\mathcal{S}(r_i)}\!\left(z_i^{(r_i)}; x_0, a_{\mathcal{W}_i}, z_{<i}^{\text{ref}}\right)$$

Default: $\mathcal{R}_\phi$ is the frozen ForgeWM-1, $\mathcal{S}(r_i)$ is 4 updates from $r_i = 0.3$ to 0. Chunks are committed one at a time. This is SDEdit applied to a model's own draft, with the twist that the conditioning prefix is progressively upgraded as you go.

## Ablation Studies and Experiments

### Main comparison — 1,000 paired Minecraft trajectories, 77 frames

| Model | IQ↑ | LPIPS↓ | AQ↑ | Subj.Cons↑ | Flow Prof↑ | KCtrl↑ | Mouse Acc↑ | Latency (ms)↓ | FPS↑ |
|---|---|---|---|---|---|---|---|---|---|
| Matrix-Game 2.0 | 0.6282 | 0.6443 | 0.4583 | 0.7349 | 0.9343 | 0.9156 | 0.7061 | 370.9 | 32.35 |
| HY-WorldPlay | 0.6133 | 0.6172 | **0.4855** | **0.9466** | 0.8288 | 0.9286 | 0.5818 | 2164.3 | 7.54 |
| ForgeWM-1 | 0.6776 | 0.6529 | 0.4807 | 0.8279 | 0.9403 | 0.9545 | 0.7848 | **168.2** | **72.10** |
| ForgeWM-2 | **0.6865** | 0.6171 | 0.4814 | 0.8349 | **0.9429** | **0.9740** | **0.8268** | 239.7 | 50.31 |
| ForgeWM-4 | 0.6788 | **0.6168** | **0.4860** | 0.7613 | 0.9420 | **0.9740** | 0.8102 | 369.6 | 32.47 |

Read this carefully, because the headline hides things.

**Quality is not monotone in the step budget.** ForgeWM-2 beats ForgeWM-4 on IQ, Flow Profile and Mouse Accuracy. ForgeWM-1 beats ForgeWM-4 on Subject Consistency. These are *separately trained checkpoints*, so there is no reason for a clean ordering — but it is a warning against assuming "more steps = better" when each budget got its own 4k-iteration run.

**HY-WorldPlay's Subject Consistency (0.9466) is a trap.** SC rewards frame-to-frame DINO similarity plus similarity to frame 0. A video that barely moves scores brilliantly. HY-WorldPlay's Flow Profile is the worst in the table (0.8288) and its Mouse Accuracy is 0.5818 — it is the most *static* model, not the most consistent one. The authors say this out loud. The reference-aligned Flow Profile metric is the honest one.

**Latency.** ForgeWM-1 at 168 ms/chunk, 12 frames per chunk, is 72 FPS of generated content. HY-WorldPlay at 2164 ms is 7.5 FPS — not playable.

### The stage ablation — the most informative table in the paper

All rows forced to a shared 4-step schedule, same 1,000 trajectories, same seeds. Only the weights change.

| Stage | Regime | LPIPS↓ | IQ↑ | AQ↑ | SC↑ |
|---|---|---|---|---|---|
| 0 | bidirectional teacher (ref.) | 0.814 | 0.455 | 0.463 | 0.677 |
| 1 | teacher-forced causal | 0.806 | 0.508 | 0.454 | 0.700 |
| 2 | causal consistency | **0.605** | 0.659 | 0.483 | **0.760** |
| 3 | distribution matching | 0.617 | **0.716** | **0.489** | **0.760** |

**Stage 2 is doing almost all the work.** LPIPS falls 0.806 → 0.605, confidence intervals non-overlapping. Causal consistency distillation, not causalisation, is what makes few-step sampling possible. Stages 0 and 1 are full-trajectory flow-matching models and simply do not function at 4 steps — this is expected, but it is worth seeing the size of the cliff.

**Stage 3 does not improve paired LPIPS at 4 steps. It slightly hurts it.** 0.605 → 0.617, with Stage 2 holding a paired advantage of $-0.012$ ($[-0.015,-0.008]$). What Stage 3 buys is per-frame sharpness: IQ 0.659 → 0.716. SC is unchanged, AQ barely moves. So the on-policy stage trades reconstruction fidelity for appearance. The authors admit the ablation **does not isolate Stage 3's effect at 1 or 2 steps**, which is precisely where you would expect it to matter most, since that is where the clean-history mismatch is largest. That is the biggest hole in the evaluation.

### Test-time step scaling — frozen ForgeWM-1

Freeze the one-step student, change only the solver schedule (1/2/4/8/16/32 steps). IQ peaks at **2** steps. SC peaks at **4**. LPIPS keeps improving through 4–8 steps then slightly regresses. Flow Profile and KCtrl are flat throughout.

The reading: extra test-time denoising mostly increases **motion magnitude**, latency and FLOPs. It does not improve *directional* control. Controllability is baked in by training, not bought at inference.

### Replay refinement

| Method | Replay LPIPS↓ | $D_{\text{draft}}$↓ |
|---|---|---|
| One-step draft (no refinement) | 0.6532 | — |
| **Replay refinement (ours)** | **0.6155** | **0.1970** |
| w/ four-step companion refiner | 0.6157 | 0.1960 |
| w/ four-step draft and refiner† | 0.6099 | 0.2361 |
| Direct ForgeWM-4 from noise | 0.6168 | 0.6187 |

The comparison that matters is row 2 vs row 5. Both land at ~0.616 reference LPIPS. But regenerating from noise gives you a $D_{\text{draft}}$ of 0.619 — a *different* rollout, different viewpoint, different object layout. Refining the draft gives 0.197. Same quality, ~3× closer to what the player experienced.

Row 3 is the useful negative-ish result: using the separate 4-step checkpoint as the refiner gives essentially identical numbers (0.6157 / 0.1960). **You do not need a second model.** The deployed 1-step student refines its own work just as well. The † row uses a different draft and so is not comparable to the others.

### Human study

41 participants, 615 forced-choice selections (no tie option), blind, randomised left-right, matched initial states and control traces. ForgeWM-4 wins 68.8% on visual quality, 57.6% on action accuracy, 55.6% on spatiotemporal consistency; pooled 60.7%. Note the margin shrinks as the criterion gets closer to *control*, and 55.6% of a three-way choice is not far above the 33% floor for consistency.

### Recipe transfer to FPS

Same four stages, gamepad controls (two analog sticks + six buttons), no architectural change beyond widening `mouse_dim_in` from 2 to 4. 65,246 clips across seven titles, Stage 0 run for 12k updates. Macro-average LPIPS 0.656, but **flow ratio 1.45** — the model consistently moves *more* than the reference. Per-game the ratio spans 1.16 (Call of Duty) to 1.78 (Warzone).

### What did not work — the grafting failures

This is the most transferable engineering lesson in the paper and it is buried in Appendix 8.2.

Widening the continuous-control input from 2 to 4 channels broke warm-starting in two distinct ways:

1. **Shape-filtered checkpoint loading discarded the entire input projection**, including the 1536 columns reading the visual hidden state. The resulting model collapsed into "use the hidden state, ignore the control" and **never recovered controllability during training**. A silent, permanent failure.
2. **Zero-initialising the two new channels also failed.** No input variation reaches a zeroed column, so at Stage 0's LR of $2\text{e}{-6}$ the symmetry never breaks and the two new channels stay dead.

The fix: copy the 1536 visual columns and the 2 original control columns verbatim, and initialise each *new* channel by **copying a pretrained control channel** — trained scale, symmetry already broken. The "contributes nothing at init" property comes from elsewhere: the action module's *output* projections ship zeroed in the base checkpoint, so the widened branch adds no residual on step one and the video prior survives.

### Metric they had to replace

They dropped GameWorld Score's Keyboard Accuracy and substituted **KCtrl**, a camera-trajectory sign test. Reason: GameWorld's keyboard evaluator is an inverse-dynamics model trained on *real* frames, so it conflates "did the model respond to the key" with "does the model's output look like real Minecraft." KCtrl instead runs opposite-action pairs (forward/back, left/right) from the same scene and gives credit only if **both** opposite commands produce the correct net translation sign:

$$\text{KCtrl} = \frac{1}{2S}\sum_{s=1}^{S}\left(C_{s,\text{f}}C_{s,\text{b}} + C_{s,\text{l}}C_{s,\text{r}}\right)$$

No magnitude threshold — sign only. Mouse Accuracy is kept, but it is a 9-way direction classification from the VPT inverse-dynamics model, not a regression error.

## Worth Remembering

**Limitations the authors state plainly.**

- **Long-horizon drift is unsolved.** Quantitative evaluation is a 77-frame window (~6.4 s). The 22 s qualitative rollouts show gradual loss of block structure and slowly spreading colour artifacts. Causalisation and distillation *reduce* autoregressive error accumulation; they do not stop it. No claim of indefinite stability.
- **Motion over-response** on FPS, macro-ratio 1.45. Directional and persistence fidelity are better than magnitude fidelity.
- **Stage 3's value is budget-dependent and untested where it should matter most** — no ablation at 1 or 2 steps.
- **Evaluation scope is narrow on purpose.** Quantitative comparison is Minecraft only. HY-WorldPlay uses a different control parameterisation, so its effective motion scale depends on the hand-written adapter — which means part of its poor Flow Profile could be adapter tuning rather than model quality. CrossFPS is presented separately, drawn from a *combined* validation+test split (the archive does not label them separately), so those numbers are not comparable to published benchmark results.
- **Equal-per-game sampling was necessary** because the CrossFPS split is wildly unbalanced — the largest title has ~65× the clips of the smallest. An unweighted mean over clips would silently report one game's score under a cross-game label. Good instinct, worth stealing.

**Surprising results.**

1. Consistency distillation, not causalisation, is the load-bearing stage. If you only had budget for one trick, it is Stage 2.
2. The on-policy stage improves sharpness but not paired fidelity at 4 steps. On-policy training being neutral-to-negative on reconstruction is a real result worth holding.
3. The replay refiner does not need to be a bigger model. The 1-step student refining its own draft equals the 4-step checkpoint as refiner.
4. Quality is non-monotone across the three budget-specialised students. Each got the same 4k iterations; presumably the 4-step student's optimisation is harder, or its 4k iterations were not enough.

**Practical caveats for anyone building on this.**

- Three caches to keep in sync (visual, keyboard, mouse) and one alignment arithmetic (4 frames/latent × 3 latents/chunk = 12 frames/chunk). Get that wrong and your controls silently lag by a chunk.
- $\beta_1 = 0$ in AdamW throughout. Unusual, and worth checking whether it matters — flow-matching video training often uses this.
- Critic:generator update ratio of 5:1 in Stage 3. The adversarial-ish balance matters.
- Latency numbers exclude VAE decoding and file writing. The 72 FPS figure is generation only, not end-to-end playable frame rate.
- Reported FPS mixes chunk sizes (12 frames for ForgeWM/Matrix-Game, 16 for HY-WorldPlay), so latency-per-chunk and FPS tell slightly different stories.

**Connections.**

The replay mechanism is exactly SDEdit's "noise it, denoise it" move, but applied to the model's own output rather than a user sketch, and with a progressively-refined conditioning prefix. The clean-history-to-generated-history progression is the same disease and the same cure as exposure bias in autoregressive sequence models, and the same structure as DAgger in imitation learning. The three-network Stage 2 setup (student, EMA, frozen teacher) is architecturally the same as [[Bootstrap Your Own Latent (BYOL)]]'s target network, doing a different job.

**Open questions.**

- Does Stage 3 actually earn its place at 1 step? The paper cannot say.
- Would training one flexible-budget student (à la AnyFlow's flow-map transitions) beat three specialised ones? Cheaper to ship, and the non-monotonicity above suggests the specialised students are not individually converged.
- The replay trick generalises beyond games: any streaming generative system with a latency-constrained online path and a tolerant offline path could do this. Interactive video editing, live captioning, real-time rendering.
- Motion over-response at 1.45× — is that a distillation artefact (few-step samplers overshooting), a data artefact, or an action-scaling bug in the adapter?

## Links

Related: [[Flow Matching]] · [[Video Diffusion]] · [[Diffusion Models]] · [[Consistency Models]] · [[Distillation]] · [[Diffusion Sampling]] · [[Score-Based Generative Modeling through SDEs]] · [[Classifier-Free Guidance]] · [[Auto-regressive models]] · [[Causal Attention]] · [[Imitation Learning]] · [[Controlling Diffusion]] · [[Denoising Objective]] · [[Latent Diffusion]] · [[VQ-VAE]] · [[Bootstrap Your Own Latent (BYOL)]] · [[Momentum]] · [[KV Cache]] · [[Distilling the Knowledge in a Neural Network]] · [[Evaluating Generative Models]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Diffusion Policy]] · [[Cost and Latency]] · [[Decoupled Weight Decay Regularization (AdamW)]]

New topics worth writing: SDEdit and noise-strength editing, Distribution Matching Distillation (DMD), Self-Forcing and causal video distillation, Diffusion Forcing, exposure bias in autoregressive video, LPIPS as a perceptual metric, VBench and video generation benchmarks, optical flow profile metrics, inverse dynamics models for control evaluation, FSDP and sharded data parallelism, checkpoint grafting when input widths change, action-conditioned world models for games, test-time solver scaling
