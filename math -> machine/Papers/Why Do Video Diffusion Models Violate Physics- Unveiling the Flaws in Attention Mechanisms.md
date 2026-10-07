---
title: "Why Do Video Diffusion Models Violate Physics? Unveiling the Flaws in Attention Mechanisms"
authors: ["Li et al."]
year: 2026
arxiv: "2609.23658"
url: https://arxiv.org/abs/2609.23658
priority: Must-Read
read_on: 2026-09-22
tags: [paper, transformers, llm, diffusion, vision]
---
## The Core Idea

A text-to-video diffusion model decides *where the object will be in every frame* in the first ~5 denoising steps out of 50. Everything after that is filling in texture. So if a basketball is going to hover in mid-air instead of falling, that mistake is already baked in by step 5.

This paper opens up [[Video Diffusion|video diffusion]] models (specifically Wan2.1-T2V-1.3B) and asks *how* that early decision is made — and finds a concrete architectural culprit for physics violations.

The mechanism, in one paragraph:

1. Early in denoising, each frame does not have one object position. It has several **candidate regions** — blobs where the ball might end up.
2. Self-attention is what picks the winner. Regions in different frames "vote" for each other. Once a few frames lock in, the rest follow.
3. [[RoPE]] makes that voting spatially biased. Because RoPE's attention score decays with distance, a region at pixel location $k$ in frame 5 mostly attends to *the same location $k$* in every other frame. The authors call this the **spatial anchoring effect**.
4. So if one frame locks in early at a physically wrong height, RoPE drags all the other frames toward that same height — and it *suppresses* the physically correct candidate in the neighbouring frame, because that one is further away in pixel space.

The result is exactly the failure modes you see: a ball that bounces in mid-air, floats, or freezes.

> [!NOTE] Spatial anchoring effect
> RoPE's long-range decay means a token attends most strongly to tokens at the *same spatial position* in other frames, regardless of where the object actually moved. Attention becomes "stay put" rather than "follow the motion". ^spatial-anchoring

The fix is one line of code: shrink the RoPE rotation angle along height and width by a factor $\lambda < 1$, but **only during the first 5 denoising steps**. Slower decay means candidate regions further away can still compete, so the model explores more positions before committing.

Why this did not exist before: prior work on physics in video generation either bolted on a physics simulator, rewrote the prompt with an LLM, or fine-tuned on physics-heavy data. Nobody had asked which *component* was producing the wrong trajectory. Earlier interpretability work established "first shape, then details" as an observation but stopped there.

## The Methodology

**The model.** Wan2.1-T2V-1.3B, a [[Diffusion Transformer]] trained with [[Flow Matching|flow matching]] ([[Rectified Flow|rectified flow]]). 30 layers, 12 heads, $D=1536$, head dim 128. A 3D VAE encodes an $81\times480\times832$ video into a token grid of $21\times30\times52 = 32760$ tokens. Each block is: self-attention (with 3D RoPE) → [[Cross Attention|cross-attention]] to the text → FFN. 50 denoising steps, [[Classifier-Free Guidance|CFG]] scale 5.0.

Training objective is the standard flow-matching regression:

$$\mathcal{L} = \mathbb{E}\left\|u(x_t,t,c;\theta) - v_t\right\|^2,\quad v_t = x_1 - x_0$$

The main case study is one prompt — "a basketball falls vertically from mid-air onto a wooden floor and bounces up several times" — with seed 26.

### Part 1: cross-attention says *what* goes where

Cross-attention is the only path for text meaning to enter the video tokens, so start there. Take the attention map from every video token to the word "basketball": $\bm{A}^{\mathrm{CA}} \in \mathbb{R}^{f\times h\times w}$. Watch it across denoising steps.

At step 1 it is noise. At step 3 there are several bright blobs per frame. By step 5 they converge. By step 7 you can see a clean parabola.

Two metrics quantify this. Normalise the map into a probability $\bm{P}^{\mathrm{vid}}$ over all (frame, y, x) positions, then:

- **Attention entropy** — $H^{\mathrm{vid}}/\log(fhw)$. How spread out is the attention?
- **Support quality** — $Q^{\mathrm{vid}} = \sum_p \bm{P}^{\mathrm{vid}}(p)\,\bm{M}^{\mathrm{obj}}(p)$, where $\bm{M}^{\mathrm{obj}}$ is a binary mask of where the object *ends up* in the finished video. How much attention mass sits on the true final trajectory?

Both jump sharply at step 5. That is the "shape" being decided.

### Part 2: which heads actually cause it

Having a nice trajectory pattern is not the same as *driving* the trajectory. Two more measurements separate correlation from cause.

- **Convergence speed** — mean support quality over the first 10 steps. Purely observational.
- **Head contribution** — causal. Ablate a head, see how much the predicted velocity inside the object region changes.

The patching metric is:

$$\mathcal{L}_m = \sum_{p} \bm{M}^{\mathrm{obj}}(p)\left[u_t^{\mathrm{cond}}(p)^\top \cdot \operatorname{stopgrad}\left(\Delta u_t^{\mathrm{clean}}(p)\right)\right]$$

where $\Delta u_t^{\mathrm{clean}} = u_t^{\mathrm{cond,clean}} - u_t^{\mathrm{uncond,clean}}$ is the CFG velocity difference *before* any ablation. Subtracting the unconditional branch strips out generic "make it look like a video" signal and leaves the prompt-specific part. The dot product asks: after ablating this head, does the velocity still point the same way the condition wanted?

They use **attribution patching** — a first-order Taylor approximation of the true ablation effect — so all heads are scored in one forward and one backward pass instead of $O(|\mathcal{V}|)$ separate runs:

$$c(n) \approx \left[n(x^{\text{noise}}) - n(x^{\text{clean}})\right]^\top \cdot \nabla_n \mathcal{L}_m$$

Then they verify with real **zero ablation** — setting a head's write into the residual stream to zero across all 50 steps — and just looking at the video.

> [!NOTE] Per-head write
> The thing you actually zero out is $U_{t,\ell,k} = g_{t,\ell}\odot Z_{t,\ell,k}W_{O,\ell,k}$ — the head's slice of the output projection, after the residual gate. Not the attention weights, not the pre-projection output. ^per-head-write

### Part 3: self-attention says *where*

Cross-attention reflects an outcome; self-attention produces it, because it is the only module where video tokens talk to each other.

Extract candidate regions $\{\Omega_{i,k}\}$ per frame from the head-averaged cross-attention map (a clustering pipeline: winsorise, despike, background-subtract, multi-level peak seeding, weighted k-means, prune). Then score how strongly two candidates in different frames pick each other.

Normalised attention from region $\Omega_{i,k_i}$ to $\Omega_{j,k_j}$:

$$\widetilde{C} = \frac{C(\Omega_{i,k_i}\to\Omega_{j,k_j})}{M(\Omega_{i,k_i}\to f_j)}$$

(numerator = mean attention into that region; denominator = total attention into that whole frame). **Mutual consistency** is the product both ways, averaged over heads:

$$\mathrm{MC}(\Omega_{i,k_i},\Omega_{j,k_j}) = \overline{C}(\Omega_{i,k_i}\!\to\!\Omega_{j,k_j})\,\overline{C}(\Omega_{j,k_j}\!\to\!\Omega_{i,k_i})$$

and a region's overall score is the mean over frames of its best partner: $\mathrm{MC}(\Omega_{i,k_i}) = \frac{1}{f}\sum_j \max_{k_j}\mathrm{MC}(\Omega_{i,k_i},\Omega_{j,k_j})$.

High mutual consistency = likely winner. Validated by plotting it against **anchor-distance** (distance to where the object really ends up): at step 3 the clouds overlap, by step 10 winners and losers are two clean clusters.

### Part 4: the RoPE fix

3D RoPE splits the head dimension into thirds for frame / height / width. The query–key score becomes:

$$\operatorname{Re}\left[\sum_{a\in\{f,h,w\}} q^a\, {k^a}^{*}\, e^{i\Delta p^a\theta}\right]$$

Small $\Delta p^h, \Delta p^w$ → large score. That is the anchoring.

The modification just multiplies the angle on the two spatial axes:

$$f^h(q,p)=q^h e^{ip^h\lambda^h\theta},\qquad f^w(q,p)=q^w e^{ip^w\lambda^w\theta},\qquad \lambda^{h},\lambda^{w} < 1$$

Same for keys. Nothing else changes. No new parameters.

**Training-free:** apply it only during the first 5 of 50 inference steps.

**Training-based:** [[LoRA]] fine-tune on WISA (80K physics videos, filtered down to ~48K), $r=64$, $\alpha=32$, lr 1e-4, batch 32, 4×A800, 800 steps. Only attention $W_Q,W_K,W_V,W_O$ are trainable — not the FFN. $\lambda^{h/w}$ is ramped from 0 up to 0.75 during training; 0.70 is used at test time. A **custom timestep sampler** draws from the first 10% of denoising steps with probability $p^{\mathrm{early}} = 0.9$.

## Ablation Studies and Experiments

**Head taxonomy (zero ablation on the basketball case).** Scatter convergence speed against contribution, four quadrants:

| Type | Convergence | Contribution | What breaks when you zero it |
|---|---|---|---|
| 1 | low | low | Nothing moves — *unless* you include layers 0–1, which kill motion entirely (they initialise it) |
| 2 | low | high | Nothing. Object appearance and background change; trajectory identical |
| 1+2 together | — | — | Object size shifts. Motion basically unaffected |
| 3 | high | high | **Trajectory collapses.** Few heads, huge effect |
| 4 | high | low | Slight shift in starting position. Trajectory unchanged |

The lesson: a head showing a clean trajectory pattern is *not* sufficient for it to cause the trajectory (Type 4). And a high contribution score is *not* sufficient either (Type 2) — that metric picks up appearance edits too. You need both, and you need to confirm by watching the video. Replicated on other prompts (wooden block on a slope) and seeds (8, 20).

**Candidate competition, measured.** Figure 19 shows the winner-minus-strongest-loser gap in mutual consistency oscillating **around zero** for the first several steps in every layer. The correct region is often *behind*. The concrete failure trace, seed 20, layer 16: in frame 14, region K1 leads at step 2 (0.21 vs K3's lower score). Frame 10's K1 stabilises at step 3 with confidence 0.63. Because K3 in frame 14 is spatially close to frame 10's winner, anchoring pulls it up — by step 3 it is 0.33 vs K1's 0.21. K3 wins, and the ball stops in mid-air.

**VideoPhy (343 prompts, human-scored, seed 42).** SA = semantic adherence, PC = physical commonsense, both binary.

| Model | SA | PC | Solid-Solid PC |
|---|---|---|---|
| Wan2.1-T2V-1.3B | 53.64 | 29.45 | 24.52 |
| + prompt refinement (PhyT2V-style) | 80.47 | 44.31 | 28.88 |
| + LoRA | 55.69 | 35.57 | 27.95 |
| + VideoREPA | 56.56 | 37.90 | 30.61 |
| + modified RoPE (training-free) | 57.43 | 39.94 | 32.60 |
| + modified RoPE + PR | 86.30 | 58.89 | 45.32 |
| + LoRA + modified RoPE | 64.72 | 41.98 | 35.39 |
| + LoRA + modified RoPE + PR | **87.46** | **62.68** | **50.61** |

Training-free RoPE modification alone (57.43 / 39.94) beats VideoREPA, which needs an external video foundation model during fine-tuning. The gains concentrate in the `solid-*` subsets, which contain large-magnitude motion — exactly what the method targets. On `fluid-fluid` it actually *loses* to plain LoRA (43.74 vs 45.61 PC).

**What did not work:**

- **Learnable $\lambda$.** They tried letting the model pick $\lambda^{h/w}_{\ell,k}(\tau)$ per head and per timestep via $\exp(\mu_{\ell,k}^a + g_\psi(e_\tau))$, initialised to 1. During training about half the heads drove $\lambda$ below 1 and half above; the mean sat at 1.0 and physical commonsense did not improve. The authors' explanation is sharp: inter-frame motion is a tiny fraction of the pixels the flow-matching loss is regressing, so the gradient is dominated by everything *except* motion. The loss cannot find this by itself — you have to impose it.
- **Training the FFN in LoRA.** Adding $W_{in}, W_{out}$ to the trainable set did not improve trajectories and *degraded* aesthetics — the cork in "cork twisted out of a bottle" came out distorted while still rotating correctly. Consistent with FFNs storing static appearance knowledge (Geva et al.) and attention handling inter-token dynamics. This doubles as evidence that the interpretability story (it's attention, not FFN) is right.
- **$p^{\mathrm{early}}$ swept over 0.3 / 0.5 / 0.7 / 0.9 / 1.0.** Too high (→1.0) and the model forgets how to denoise mid-to-late steps, so aesthetics rot. Too low (≤0.5) and trajectories distort — not enough modified-RoPE samples, so the modification acts like injected noise rather than a learned condition. 0.9 is the sweet spot.
- **$\lambda^{h/w}$ swept over 0.55–0.90.** 0.75 for training, 0.70 for inference. Training-free mode needs *per-case manual tuning* of $\lambda$ (0.50 for seed 29, 0.85 for seed 23) — which is exactly why they moved to fine-tuning.
- **The bigger model is not better at physics.** They chose 1.3B over 14B because 14B "does not perform better in physics". Scale does not fix this.
- **The benchmark's own automatic grader.** VideoPhy's fine-tuned VLM judges hallucinated badly and correlated poorly with humans, so all numbers in Table 1 are human-scored, blind to model version.

## Worth Remembering

**Generalisation is better than you'd expect from a single-seed fix.** Trained entirely at seed 42, but the model improves trajectories at seeds 8, 20, 23 and 29 with the same $\lambda$. The mechanism is not seed-specific.

**Scope.** Solid dynamics with trackable trajectories. Fluids get worse, not better. Text-to-video only — image-to-video (where the first frame is already pinned) and few-step autoregressive diffusion are left for future work. One base model family.

**The metrics are the reusable part.** Support quality, convergence speed, mutual consistency, and the anchor-distance validation form a small toolkit for asking "when did this model decide, and what made it decide that?" of any diffusion transformer. The two-step discipline — score causally with attribution patching, then *confirm with real zero ablation and your eyes* — is the right habit, because the cheap metric put Type 2 heads (appearance editors) in the same bucket as real motion heads.

**The honest limitation of the fix.** $\lambda$ is a hand-tuned constant. The authors tried to make it adaptive and it failed. Right now this is "a magic number that happens to be about 0.7", justified by mechanism rather than derived from it.

**Connection worth drawing.** The RoPE decay that hurts here is the *same* property that helps in language models: locality bias is a useful prior when nearby tokens are related. In video, "nearby in pixel space across frames" is a *wrong* prior whenever anything moves fast. Compare [[Train Short, Test Long (ALiBi)|ALiBi]], where the decay is deliberate and beneficial. The same inductive bias, different sign.

**Open question the paper raises but does not close.** If the flow-matching loss genuinely cannot see motion (because motion is a small fraction of the pixel budget), then physical plausibility may need an objective that weights inter-frame differences explicitly — not just an architectural nudge during the first five steps.

## Links

Related: [[RoPE]] · [[RoFormer- Enhanced Transformer with Rotary Position Embedding]] · [[Video Diffusion]] · [[Diffusion Transformer]] · [[Cross Attention]] · [[Attention]] · [[Multi-Head Attention]] · [[Flow Matching]] · [[Rectified Flow]] · [[Diffusion Models]] · [[Diffusion Sampling]] · [[Classifier-Free Guidance]] · [[LoRA]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Query, Key, and Value (QKV)]] · [[Evaluating Generative Models]] · [[Train Short, Test Long (ALiBi)]] · [[Latent Diffusion]]

New topics worth writing: attribution patching and activation patching, mechanistic interpretability for diffusion models, VideoPhy and physical-commonsense benchmarks, spatial anchoring in 3D positional embeddings, timestep-weighted sampling in diffusion fine-tuning, world models and physical plausibility
