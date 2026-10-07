---
title: "World Action Learning via Interaction-Centric Spectral Latent Guidance"
authors: ["Zhiming Liu", "Yikun Miao", "Ying Chen", "Hongrui Yin", "Fangqi Zhu", "Xiaoyi Pang", "Quanxin Shou", "Zhengyang Yan", "Haodong Wang", "Song Guo"]
year: 2026
arxiv: "2610.03607"
url: https://arxiv.org/abs/2610.03607
priority: Good-To-Read
read_on: 2026-10-07
tags: [paper]
---
## The Core Idea

Robot demonstrations are expensive. Videos of people doing things with their hands are cheap and there are millions of hours of them. So: can you learn *action* knowledge from human video and pour it into a robot policy?

The standard trick is a **latent action model** (LAM). You take two video frames, shove them through an encoder, and ask it to produce a small vector that lets you predict the second frame from the first. No action labels needed. That vector becomes a pseudo-action.

> [!NOTE] Latent action model
> A model trained to compress the change between two frames into a short vector, such that the vector plus the first frame predicts the second. The vector is treated as "what action happened". ^latent-action-model

WING's claim is that this breaks on **egocentric** (head-mounted camera) video for two separate reasons.

**Problem one: the camera moves.** A robot's head camera is bolted to the robot and sits still. A human's head camera swings around constantly. When you train a LAM to reconstruct the next frame, it will happily spend its whole latent budget on "the camera rotated 3 degrees left", because that is the single biggest pixel change in the frame. The hand reaching for the cup — the part you actually want — is a small local wiggle by comparison. The latent ends up encoding the observer, not the interaction.

The paper proves a small version of this in Appendix A. For a linear LAM with a $d_z$-dimensional bottleneck, the optimal reconstruction projects the frame difference onto the top $d_z$ eigenvectors of $\Sigma = \text{Cov}(\Delta \mathbf{o})$, and the retained camera energy is exactly

$$\mathbb{E}\left[\|\mathbf{P}\Delta\mathbf{o}_{\text{obs}}\|_2^2\right] = \operatorname{tr}(\mathbf{P}\,\Sigma_{\text{obs}})$$

This is nonzero whenever camera motion has variance in the chosen directions. Reconstruction ranks directions by *total* variance. It has no way to know that some variance is the thing you want and some is noise. That is the whole argument in one line.

**Problem two: humans and robots move differently.** Even if you perfectly capture "the hand closed around the handle", a human does it in 0.4 s with a jittery, dexterous motion; the robot does it slowly with a two-finger gripper. The *fine* temporal detail does not transfer. WING's observation is that the *slow* temporal detail does. They decompose latent action trajectories with a discrete cosine transform and measure cosine similarity between semantically matched human/robot clip pairs:

| Representation | matched − mismatched similarity gap |
|---|---|
| Low frequency ($q_{0:3}$) | **0.51** (95% CI 0.48–0.53) |
| Full spectrum ($q_{0:7}$) | 0.26 (0.22–0.27) |
| High frequency ($q_{4:7}$) | 0.05 (0.03–0.10) |

So: throw away the high frequencies and what is left is shared across bodies.

> [!NOTE] Spectral latent guidance
> Encode $H$ frame transitions into latent actions, run a DCT along time, keep only the lowest $K$ coefficients, and use that as a *conditioning hint* for a robot policy — not as a prediction target. ^spectral-latent-guidance

The unlock: human video becomes a usable pretraining signal for robot control without anyone ever mapping a human hand pose onto a robot gripper. Results: 99.20% on LIBERO, 93.80% on RoboTwin 2.0, 57.7% on RoboCasa–GR1, 75.0% average on four real bimanual tasks.

---

## The Methodology

Two pieces: a latent action model (WING-LAM) and a guidance module bolted onto a world-action model.

### WING-LAM — separating camera from hands

Trained as **teacher → student**, where the teacher gets extra information the student never sees.

**Step 1: build motion labels offline.** For each frame pair $(\mathbf{o}_t, \mathbf{o}_{t+\delta})$:

- Run AllTracker to get point correspondences $(\mathbf{p}_{t,i}, \mathbf{p}_{t+\delta,i})$ with confidence weights $\rho_i$.
- Run EgoHOS to get a rough hand mask.
- Fit an **affine warp** $W_{t\to t+\delta}$ with RANSAC to the *background* points only. That warp is your camera-motion label.
- Subtract it off. What is left is the interaction label:

$$\mathbf{r}_{t,i} = W_{t\to t+\delta}^{-1}(\mathbf{p}_{t+\delta,i}) - \mathbf{p}_{t,i}$$

- Grow the hand mask outward to any nearby point whose residual $\mathbf{r}_{t,i}$ is large. That gives the interaction set $\mathcal{I}_t$ and background set $\mathcal{B}_t$.

The key move is that camera motion is *global and coherent* while hand motion is *local*. Fit the global thing, call the remainder the local thing.

**Step 2: the two-branch teacher.** A spatiotemporal Transformer (DreamDojo architecture) encodes both frames. For each tracked point, sample visual features at its location in both frames, concatenate with normalised coordinates, displacement and confidence → track descriptor $\mathbf{d}_{t,i}$. Two gated set-pooling heads:

$$\mathbf{z}^T_{t,\text{cam}} = P_{\text{cam}}(\{\mathbf{d}_{t,i}\}_{i \in \mathcal{B}_t}) \in \mathbb{R}^{16}, \qquad \mathbf{z}^T_{t,\text{int}} = P_{\text{int}}(\{\mathbf{d}_{t,i}\}_{i \in \mathcal{I}_t}) \in \mathbb{R}^{32}$$

Each decodes its own thing — camera latent → the warp, interaction latent → the residuals — and they compose to reconstruct the endpoints:

$$\widehat{\mathbf{p}}_{t+\delta,i} = \widehat{W}_{t\to t+\delta}(\mathbf{p}_{t,i} + m_i \widehat{\mathbf{r}}_{t,i})$$

Note $m_i$: background points get *only* the warp. The interaction branch is structurally forbidden from explaining them.

Six loss terms, all Smooth-$L_1$ ($\beta = 0.01$) or regularisers:

$$\mathcal{L}_{\text{teacher}} = \lambda_{\text{trk}}\mathcal{L}_{\text{trk}} + \lambda_{\text{cam}}\mathcal{L}_{\text{cam}} + \lambda_{\text{int}}\mathcal{L}_{\text{int}} + \lambda_{\text{cons}}\mathcal{L}_{\text{cons}} + \lambda_{\text{var}}\mathcal{L}_{\text{var}} + \lambda_{\text{cov}}\mathcal{L}_{\text{cov}}$$

with $\lambda_{\text{trk}} = \lambda_{\text{cam}} = \lambda_{\text{int}} = \lambda_{\text{var}} = 1$, $\lambda_{\text{cons}} = 0.1$, $\lambda_{\text{cov}} = 0.01$. The consistency term $\mathcal{L}_{\text{cons}}$ warps the second frame with a fake camera motion and demands the interaction latent not change. The variance term ($\gamma = 0.1$ floor on per-dim std) and covariance term (off-diagonal squared correlations) are there to stop [[Bootstrap Your Own Latent (BYOL)#^collapse|collapse]] — same family of fix as [[VICReg]] and [[Barlow Twins]]. AdamW, 60k steps, lr $5\times10^{-6}$.

**Step 3: distil into an RGB-only student.** At deployment there are no point tracks. So freeze the teacher, initialise a student $S_\phi$ from its visual encoder, feed it *only* the two RGB frames, mean-pool the tokens, project to 32 dims. Train with:

$$\mathcal{L}_{\text{distill}} = \frac{1}{2D_i}\sum_{v \in \{\text{nat},\text{aug}\}}\left\|\mathbf{z}_t^v - \mathbf{z}^T_{t,\text{int}}\right\|^2_2 + \frac{\lambda_{\text{view}}}{D_i}\left\|\mathbf{z}_t^{\text{nat}} - \mathbf{z}_t^{\text{aug}}\right\|^2_2$$

where `aug` is the camera-warped version of the second frame and $\lambda_{\text{view}} = 1$. Both views chase the same target, and they must agree with each other. 60k steps, lr $10^{-5}$, weight decay 0.05.

The teacher and all the tracking machinery are then thrown away. What ships is a frozen two-frame RGB encoder. This is the same privileged-information pattern as in [[Mind2Dialogue- Training Human-Aware Language Models by Simulating User Mental States|privileged distillation]] and [[Best Practice Critic Optimization|privileged critics]] — [[Distillation]] where the teacher's advantage is *inputs*, not capacity.

### The spectral guidance module

Sample observations every 0.2 s (every 6th frame at 30 FPS). Nine endpoints → $H = 8$ transitions → a 1.6 s window:

$$\mathbf{Z}_t = [\mathbf{z}_t, \ldots, \mathbf{z}_{t+7}]^\top \in \mathbb{R}^{8 \times 32}$$

Apply the orthonormal DCT-II matrix $\mathbf{C}$ along time, keep the lowest $K = 4$ rows:

$$\mathbf{g}_t^{\text{low}} = [\operatorname{DCT}(\mathbf{Z}_t)]_{0:3} \in \mathbb{R}^{4 \times 32}$$

Two properties of DCT-II worth knowing (Appendix A.3). It is orthonormal, so $\|\mathbf{Z}\|_F^2 = \|\mathbf{Q}\|_F^2$ — no energy lost by the *full* transform. And it diagonalises the first-difference operator $\mathbf{D}^\top\mathbf{D}$ with eigenvalues $\lambda_k = 4\sin^2(\pi k / 2H)$, so

$$\sum_{h}\|\mathbf{z}_{h+1} - \mathbf{z}_h\|_2^2 = \sum_k \lambda_k \|\mathbf{q}_k\|_2^2$$

Since $0 = \lambda_0 < \lambda_1 < \cdots < \lambda_{H-1}$, mode index *is* speed of variation. Keeping low $k$ keeps the slow stuff. This is just [[Fourier Series Decomposition#^change-of-basis|change of basis]] used as a filter.

**The predictor.** $\mathbf{g}_t^{\text{low}}$ needs future frames, which you do not have at inference. So train a tiny predictor $P_\psi$ from present context only:

$$\widehat{\mathbf{g}}_t^{\text{low}} = P_\psi(\mathbf{l}, \mathbf{o}_t, \mathbf{s}_t), \qquad \mathcal{L}_{\text{pred}} = \frac{1}{KD_i}\|\widehat{\mathbf{g}}_t^{\text{low}} - \mathbf{g}_t^{\text{low}}\|_F^2$$

Architecture: 4 learned queries [[Cross Attention|cross-attending]] into concatenated current-frame tokens + language tokens. One Transformer block, hidden 1024, 8 heads, 4× FFN, output projected to 32 dims. That is it — tiny.

**Injection.** The 4 coefficient vectors get projected to $D_{\text{hid}}$, [[Layer Normalization|LayerNormed]], given a learned per-mode embedding, and the action model reads them through gated cross-attention:

$$\mathbf{H}_a^{(l)} \leftarrow \mathbf{H}_a^{(l)} + \gamma_l \operatorname{Attn}(Q = \operatorname{LayerNorm}(\mathbf{H}_a^{(l)}), K = \mathbf{U}_t, V = \mathbf{U}_t)$$

$\gamma_l$ is learned per layer, so the model can turn guidance off if it is useless. Same gated-residual pattern as [[LoRA]]-style adapters or [[Controlling Diffusion|ControlNet]].

### Training

**Stage 1 — egocentric pretraining.** 200 h of video: ~20 h Ego4D, ~85 h EgoDex, ~95 h EgoVerse. Backbone is Wan2.2-TI2V-5B, a video diffusion model. Train the video expert for future-frame generation **and** $P_\psi$ for guidance prediction. 40 epochs, 32×H100, global batch 512, lr $1\times10^{-4}$. VAE latents and WING-LAM latents are precomputed and cached.

**Stage 2 — robot policy.** Transfer visual backbone + $P_\psi$. WING-LAM stays frozen. Joint loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{video}} + \mathcal{L}_{\text{action}} + 0.1\,\mathcal{L}_{\text{pred}}$$

8×H100, batch 128, AdamW, peak lr $2\times10^{-4}$, [[GPU processing#Fix 2: warmup (this is the non-negotiable one)|linear warmup]] then cosine decay.

Note the shape of this: the robot policy still predicts robot actions from robot demonstrations. Human video only supplies a *hint*. Nobody ever retargets a hand to a gripper.

---

## Ablation Studies and Experiments

### Does WING-LAM actually ignore the camera?

Train frozen-feature probes (ridge, MLP, random Fourier features) to decode camera motion and interaction motion, report held-out $R^2$. On EgoDex the camera target is *ground-truth 6-DoF head pose*, not the affine proxy used in training — so this is a real generalisation test of the suppression.

EgoDex, MLP probe:

| Representation | $R^2_{\text{cam}}$ ↓ | $R^2_{\text{int}}$ ↑ |
|---|---|---|
| DreamDojo | 0.70 | 0.25 |
| CD-LAM | 0.71 | 0.21 |
| LAPA | −0.27 | −0.03 |
| UniVLA | −0.03 | 0.01 |
| WING-LAM teacher (int) | **−0.03** | **0.74** |
| WING-LAM student (int) | **0.00** | **0.48** |
| WING-LAM teacher (cam) | 0.93 | −0.10 |

Read the last two rows together: the camera branch decodes head pose at 0.93 and interaction at −0.10; the interaction branch does the exact opposite. The routing worked.

The CD-LAM result is the most interesting line in the paper. CD-LAM is *explicitly designed* to debias latent actions, and it still leaks camera motion at 0.71 — as much as the undebiased DreamDojo. Reducing "action-irrelevant variation" in general is not the same as separating observer motion specifically.

Note LAPA and UniVLA get near-zero or negative $R^2$ on *both* targets. They are not debiased; they just do not encode much motion at all. Which shows up next.

**Direct perturbation test.** Apply matched camera warps — dynamic ($AB$, $BA$: different warps per frame, so inter-frame motion) vs static ($AA$, $BB$: same warp both frames, no inter-frame motion) — and report standardised dynamic-minus-static latent drift across five magnitudes from (0.25 px, 0.05°) to (4 px, 0.8°). WING-LAM's student is lowest at every level.

**And it keeps the semantics.** LARYBench action classification, frozen features:

| | WING-LAM | DreamDojo | CD-LAM | LAPA | UniVLA | VILLA-X |
|---|---|---|---|---|---|---|
| Ego4D | **64.13** | 60.92 | 60.67 | 21.02 | 15.22 | 14.80 |
| EPIC-KITCHENS | 40.57 | 40.65 | **41.19** | 17.58 | 9.95 | 10.35 |
| HoloAssist | 48.41 | 46.23 | **48.95** | 22.40 | 15.84 | 20.69 |
| EgoDex | **68.77** | 65.92 | 62.95 | 22.56 | 17.43 | 17.20 |
| **Avg** | **55.47** | 53.43 | 53.44 | 20.89 | 14.61 | 15.76 |

The gain over CD-LAM is 2 points — real but modest. The point is not that it wins big; it is that suppressing camera sensitivity did not cost semantic content, which is the thing you would worry about.

### Policy results

LIBERO (50 rollouts × 10 tasks × 4 suites = 2000 rollouts, one checkpoint):

| Method | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| $\pi_0$ | 98.00 | 96.80 | 94.40 | 88.40 | 94.40 |
| $\pi_{0.5}$ | 98.80 | 98.20 | 98.00 | 92.40 | 96.90 |
| LAPA | 73.80 | 74.60 | 58.80 | 55.40 | 65.70 |
| UniVLA | 96.50 | 96.80 | 95.60 | 92.00 | 95.20 |
| LaWAM | 99.40 | 99.60 | 98.40 | 97.00 | 98.60 |
| LingBot-VA | 98.50 | 99.60 | 97.20 | 98.50 | 98.50 |
| **WING** | 99.00 | **100.00** | **98.80** | **99.00** | **99.20** |

LIBERO is saturated — 98.60 → 99.20 is roughly 12 rollouts out of 2000. Do not read much into it. The Long suite (97.00 → 99.00) is slightly more informative since that is where methods usually fall over.

RoboTwin 2.0 (50 bimanual tasks, 100 rollouts each, clean + randomised):

| Method | Clean | Rand. | Avg |
|---|---|---|---|
| $\pi_0$ | 65.92 | 58.40 | 62.16 |
| $\pi_{0.5}$ | 82.74 | 76.76 | 79.75 |
| LingBot-VLA | 88.56 | 86.68 | 87.62 |
| LaWAM | 92.64 | 89.80 | 91.22 |
| Fast-WAM | 91.88 | 91.78 | 91.83 |
| LingBot-VA | 92.90 | 91.50 | 92.20 |
| **WING** | **94.56** | **93.04** | **93.80** |

RoboCasa–GR1 (24 humanoid tabletop tasks, 50 rollouts each): 57.7 vs LDA-1B 55.4, DiT4DiT 50.8, StarVLA-OFT 48.8, GR00T-N1.6 47.6, FastWAM 51.9 (their reproduction).

Real world, Agilex Cobot Magic dual-arm, 3 cameras, 14-D action space, 20 trials per setting, four tasks (Pack Objects, Stack Cups, Battery Insertion, Long-Horizon Battery Assembly). WING gets 75.0% average success under the standard setting — best of the four methods. Under generalisation shifts it is "comparable to $\pi_{0.5}$ with a slightly higher progress score", which is an honest way of saying *tied*.

Look at the per-setting breakdown and it is messier than the headline:

| Task | Setting | $\pi_{0.5}$ SR | WING SR |
|---|---|---|---|
| Battery Insertion | avg | **55.0** | 50.0 |
| Battery Assembly | avg | **33.3** | 31.7 |
| Pack Objects | avg | **80.0** | 78.8 |
| Stack Cups | avg | 32.5 | **37.5** |

$\pi_{0.5}$ wins three of four on success rate. WING wins on the stage-wise progress score. With 20 trials per cell, a 5-point gap is one trial.

### What the ablations actually show

Controlled 2×2, **with ego-pretraining switched off** so only the guidance mechanism is tested:

| WING-LAM | DCT | Spatial | Object | Goal | Long | Real base | Real gen |
|---|---|---|---|---|---|---|---|
| ✗ | ✗ | 99.2 | 97.2 | 96.2 | 92.6 | 65.0 | 36.0 |
| ✓ | ✗ | 98.0 | 99.2 | 95.6 | 94.2 | 67.5 | 40.3 |
| ✗ | ✓ | 99.2 | 98.2 | 98.0 | 96.2 | 71.3 | 43.0 |
| ✓ | ✓ | 99.6 | 100.0 | 99.2 | 96.2 | 72.5 | 45.0 |

**The DCT is doing most of the work.** Row 3 (DCT alone, DreamDojo's LAM) gets you from 36.0 → 43.0 on real-world generalisation; adding WING-LAM on top gets 43.0 → 45.0. All the camera-decomposition machinery — point trackers, RANSAC, masks, teacher-student distillation — buys about 2 points once you already have the frequency filter. Row 2 (WING-LAM without DCT) is barely better than nothing and actually *hurts* on LIBERO Spatial (99.2 → 98.0).

That is the honest reading: the cheap idea works and the expensive idea is a small bonus. If you were reimplementing this, build the DCT filter first on any off-the-shelf LAM.

**How many coefficients?** $K \in \{2, 4, 8\}$ with $K = 8$ being the full spectrum. $K = 4$ wins on every suite. $K = 2$ throws away useful dynamics; $K = 8$ adds high-frequency junk that does not help control. There is a sweet spot in bandwidth, not a monotone "more information is better".

**Pretraining strategy comparison** — all at the same 200 h budget, which is the right control:

- `Vid-P.` video-only pretraining → consistently worse, biggest gaps on real-world. So the win is not just "the model saw egocentric video".
- `HP-P.` hand-pose supervision → worse.
- `LA-P.` LAPA-style, latent actions as *direct prediction targets* → worse. The paper's reading: there is a gap between "representation of a visual transition" and "executable robot command", and forcing the policy to output the former damages it.
- `w/o OD.` swap WING-LAM for DreamDojo → worse in most settings.
- **Ours**, latent actions as *auxiliary conditioning* while the policy keeps predicting real actions → best.

The `LA-P.` vs `Ours` contrast is the conceptual result worth keeping. Same signal, two different ways to use it, and the "hint" framing beats the "target" framing. Compare the LAPA number on LIBERO (65.70 avg) against WING (99.20) — latent actions as targets is actively harmful at this scale.

**Denoising steps** (RoboTwin, chunk size 32): $N = 1$ collapses to 80.46% average. $N = 2$ jumps to 93.20%. After that it is flat — $N = 12$ is best at 94.56/93.04 but $N = 10$ gives 93.55. So you need *some* iterative refinement and then it stops mattering.

**Data efficiency.** With 100%, 50%, 25% of robot demos, pretraining helps at every level and the gap *widens* as data shrinks, biggest at 25% real-world. And validation loss drops monotonically with pretraining length (15k → 30k → 45k → 60k steps). This is the most useful practical result: the pretraining is buying robot-data efficiency, which is the whole point.

### What did not work

- **CD-LAM's debiasing.** Designed for this problem, measured at 0.71 camera $R^2$ — no better than the undebiased baseline on this probe.
- **Latent actions as prediction targets** (LAPA): worse in nearly every setting, and catastrophically so on LIBERO.
- **Hand-pose supervision**: loses to latent guidance at matched budget. Explicit kinematic labels are tied to human morphology; they do not transfer as cleanly as an abstract latent.
- **$K = 8$** (full spectrum): worse than $K = 4$. More information, worse control.
- **$N = 1$** denoising step: 13-point drop. No single-step shortcut here.
- **WING-LAM without the DCT**: on LIBERO Spatial it goes *backwards* (99.2 → 98.0).
- **Latency**: 500.6 ms median, 517.1 ms P95 on a 4090 at chunk 32 / 10 steps, versus $\pi_{0.5}$ at 90.6 ms and GR00T at 63.4 ms. 5–8× slower than lightweight VLAs. Better than Motus (1532.8 ms) and Cosmos-Policy (845.6 ms), but this is not a fast policy.

---

## Worth Remembering

**The camera-motion argument is the transferable insight, not the architecture.** Reconstruction-based objectives select directions by total variance in the thing being reconstructed. They cannot distinguish signal from nuisance. If your biggest pixel change is something you do not care about, your bottleneck will encode it. This is a general statement about [[Autoencoder|autoencoding]] with a bottleneck, and the same logic shows up as [[Shortcut Learning in Deep Neural Networks|shortcut learning]] and in [[Breaking the Vision-Action Shortcut- Latent Interface Training for Generalizable Robotics Foundation Models|vision-action shortcuts]]. The fix is always the same shape: give the model a reason to prefer one direction, by supervision or by architecture.

**"Keep the low frequencies" is a cheap, general transfer heuristic.** Two systems that share semantics but differ in execution dynamics will agree more in slow structure than in fast structure. The DCT is just a change of basis plus a truncation — maybe forty lines of code. It bought more on the real-world generalisation ablation (36.0 → 43.0) than the entire teacher-student camera-decomposition pipeline did (43.0 → 45.0). Try the cheap thing first.

**Hint, don't dictate.** The same latent signal used as a conditioning input beats using it as a prediction target by a wide margin. The policy keeps its own objective and treats the human-derived signal as extra context it can ignore via the learned gate $\gamma_l$. If you are transferring a noisy, out-of-domain signal into a policy, this is the safer interface.

**Limitations the authors state plainly.** The camera decomposition approximates observer motion with a *single global affine warp*. That is a 2D model of a 3D effect. Under strong parallax, large depth variation, or real 3D camera motion it will break and leak camera motion into the "interaction" residual as noise. It also depends on two external models (AllTracker, EgoHOS) at label-construction time, and the whole pipeline — track → RANSAC → mask grow → teacher → distil → DCT → predictor — is a lot of machinery for the measured gain.

**Caveats for anyone reading the numbers.**

- LIBERO is saturated. 98.60 → 99.20 is ~12 rollouts out of 2000. Treat it as a sanity check, not evidence.
- Real-world has 20 trials per cell. One trial = 5 points. $\pi_{0.5}$ beats WING on raw success in 3 of 4 generalisation tasks; the win is on the progress rubric.
- The FastWAM RoboCasa number (51.9) is the authors' own reproduction, not a published figure.
- The cross-embodiment similarity study has 100 matched and 100 mismatched pairs over 25 tasks, with matching done by human annotators. The labels were fixed before any latents were computed, and two extra annotators cross-checked — good practice, but it is still a hand-curated 200-pair set.

**Nice methodological touches worth stealing.** The camera-perturbation test uses *matched* static controls ($AA$, $BB$) against the dynamic condition ($AB$, $BA$), which isolates sensitivity to inter-frame motion from interpolation and boundary artefacts. The EgoDex camera probe uses recorded 6-DoF head pose rather than the affine proxy the model was trained against, so the suppression result is not circular. Latents are standardised per-dimension before comparing across models, since latent scales differ. And the pretraining comparison fixes the 200 h budget across all five strategies — that is the control that makes the `Vid-P.` result meaningful.

**Open questions.** Does $K = 4$ survive a change of sampling rate? The window is fixed at 1.6 s with 0.2 s steps; mode index maps to absolute frequency only through that choice, so $K$ is probably not a transferable constant. Does the low-frequency correspondence hold for contact-rich tasks where the informative part *is* fast (insertion, in-hand manipulation)? Note Battery Insertion is where WING loses to $\pi_{0.5}$. And would a depth-aware or full 3D camera model actually close the remaining gap, or is the affine approximation already good enough that the bottleneck is elsewhere?

**Connections.** The spectral bandwidth analysis (Appendix E.2.2) finds robot trajectories are far more low-frequency-concentrated than egocentric ones — consistent with the central claim and a nice independent check. The frozen-encoder-plus-probe evaluation protocol is the same one used throughout self-supervised learning (frozen features, [[Self-Supervised Learning from Images with I-JEPA|I-JEPA]]). The variance/covariance regularisers are lifted from [[VICReg]]. And the whole idea of predicting in a learned representation space rather than pixel space is the [[JEPA]] argument, applied to action rather than perception.

---

## Links

Related: [[Latent Action as Intention Enables Efficient Future Imagination for World Action Models]] · [[HuRo- Robotizing Human Videos for Scalable VLA Pretraining]] · [[Breaking the Vision-Action Shortcut- Latent Interface Training for Generalizable Robotics Foundation Models]] · [[Think Like a World Model, Act Like a VLA- Distilling World-Model Representations into Compact Robot Policies]] · [[JEPA]] · [[Distillation]] · [[Fourier Series Decomposition]] · [[Shortcut Learning in Deep Neural Networks]] · [[Cross Attention]] · [[Autoencoder]] · [[VICReg]] · [[Barlow Twins]] · [[Bootstrap Your Own Latent (BYOL)]] · [[Flow Matching]] · [[Layer Normalization]] · [[Imitation Learning]] · [[GigaBrain-0.7- Scaling Embodied Foundation Models to Emergent Capabilities with a Three-System Architecture]] · [[Zero-WAM- In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization]] · [[Mind2Dialogue- Training Human-Aware Language Models by Simulating User Mental States]] · [[What Does Privileged Information Add to On-Policy Self-Distillation]]

New topics worth writing: Discrete Cosine Transform, Egocentric video datasets (Ego4D / EgoDex / EgoVerse), Point tracking and optical flow, RANSAC, Affine warp estimation, Vision-Language-Action models, World-action models, Cross-embodiment transfer, Observer motion and active perception, Spectral filtering as inductive bias, Action chunking, Gated residual conditioning, LIBERO benchmark saturation, RoboTwin 2.0, Frozen-feature probing protocols
