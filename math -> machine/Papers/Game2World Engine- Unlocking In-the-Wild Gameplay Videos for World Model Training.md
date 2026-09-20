---
title: "Game2World Engine: Unlocking In-the-Wild Gameplay Videos for World Model Training"
authors: ["Shen et al."]
year: 2026
arxiv: "2608.24680"
url: https://arxiv.org/abs/2608.24680
priority: Good-To-Read
read_on: 2026-09-04
tags: [paper, llm, rl, vision]
---
## The Core Idea

Gameplay videos on the internet are a huge, free source of "how does a 3D world change over time" data. That is exactly what a video world model wants. But a recorded gameplay frame is **not** a picture of the game world. It is a picture of the game world *plus* a health bar, a minimap, a crosshair, a kill feed, a subtitle, a streamer's facecam and a Twitch watermark painted on top in screen space.

The claim here is that these overlays are not a cosmetic nuisance — they actively poison training. The paper's pilot study is the part that justifies everything else: take 5,442 *clean* gameplay clips, make a second copy with synthetic UI pasted on, caption both with the same captioner, fine-tune two otherwise identical copies of Wan2.1-T2V-1.3B. The clean-trained model wins on overall VideoReward by **6.83%**, and on motion quality by **18.8%**. The UI is a [[Shortcut Learning in Deep Neural Networks#^shortcut|shortcut]]: overlays are sharp, high-contrast, stuck to the camera, and follow rules that have nothing to do with physics. A generative model happily burns capacity learning them.

So the task becomes **interface–world disentanglement**: figure out which pixels are screen-space furniture, delete them, and hallucinate back the game content they were hiding — consistently across time.

> [!NOTE] Interface–world disentanglement
> Separating a rendered gameplay frame into (a) the underlying simulated 3D world and (b) the 2D overlay drawn on top of it in screen coordinates. Unlike removing a physical object from a video, the thing being removed never obeyed the scene's geometry or lighting in the first place. ^interface-world

The trick that makes this trainable at scale is a **paired-data generator running backwards**. You cannot get ground truth by finding a clean version of a UI-covered video — it does not exist. So instead they harvest real UI elements out of real gameplay footage, cut them out as transparent sprites, and then paste them back onto *separately collected clean* gameplay clips. Now you have 96K exact (dirty, clean) pairs with pixel masks and boxes for free. Train a mask-free removal model on those; test it on real footage.

The unlock: gameplay UI removal stops being a per-game hand-tuned editing hack and becomes a scalable data-processing step, the way deduplication or aesthetic filtering already is in text and image pipelines.

## The Methodology

Four pieces: a taxonomy, a data engine, a dataset, and a model.

**GameUI-Taxonomy.** 21 labels, built by eyeballing screenshots from 500+ games and merging categories until they stabilised. Grouped by *function and rendering behaviour*, not appearance: `map_radar`, `navigation_compass`, `player_status`, `weapon_ammo`, `inventory`, `ability_status`, `crosshair`, `interaction_prompt`, `screen_notice`, `subtitle`, `chat_comms`, `match_status`, `watermark`, `stream_overlay`, … plus an `other_hud` fallback. The taxonomy matters because it drives *where* an element gets pasted and *how long it stays* — a crosshair is centre-screen and permanent, a kill feed is top-right and lasts two seconds.

**G2WEngine**, in four stages:

1. *Asset extraction.* Sample representative frames from internet gameplay. GPT-5.6 Terra proposes boxes + labels; humans verify. Accepted regions are cut out into transparent sprites, keeping metadata (category, source, original normalised position, scale).
2. *Clean corpus curation.* Chop gameplay into non-overlapping 5s windows, embed the centre frame with CLIP ViT-B/16, cluster with spherical $k$-means where
   $$K=\min\!\left(K_{\max},\,N,\,\max(2,\lfloor 4\sqrt{N}\rfloor)\right),\quad K_{\max}=2048,$$
   then sample **round-robin across clusters** rather than proportional to cluster size, so one popular game cannot dominate. Then drop temporally adjacent clips whose embeddings have cosine similarity $>0.94$. Standardise to 5s / 720p / 30fps.
3. *UI synthesis.* This is where the engineering is. Persistent widgets are drawn per-category with independent probabilities (e.g. `player_status` 0.72, `map_radar` 0.68, `crosshair` 0.64), max one each, expected 4.00 per clip. Transient pop-ups get a long-tailed count: $P(M=0..5)=(0.14,0.43,0.28,0.10,0.04,0.01)$, expected 1.5 per clip. Recorded original coordinates are reused when available; otherwise category-specific anchors from one of three layout "formations" (adventure / tactical / RPG). Positions get seeded jitter $\Delta x \sim \mathcal{U}(-0.018W, 0.018W)$, then collision resolution: for boxes $A,B$, overlap is $\rho(A,B)=|A\cap B| / \min(|A|,|B|)$ and anything above $0.14$ triggers repositioning, with lower-priority widgets suppressed if nothing fits. Pop-ups get one of six animations (fade, instant, slide, wipe_trail, pop, border_arc) with sampled durations and drift. Health/stamina bars have their fill level animated by a sum of sinusoids. Inventory bars are decomposed into an empty background bar plus item icons, so new inventory states can be recomposed.
4. *Output.* Every sample ships aligned clean video, UI-overlaid video, per-frame binary mask, per-frame boxes, taxonomy labels, and a full random-seed record so any example can be reconstructed deterministically.

**Game2World** = **-S**: 96K synthetic pairs, assets from 1,010 keyframes across 303 games, 5,132 verified assets in 21 categories. **-W**: 1,079 real 5s clips from 303 games, no clean reference possible, so every frame is hand-annotated with boxes and labels for judging.

**GameCleaner.** Architecture copies Kiwi-Edit: a multimodal LLM encoder bolted onto a video diffusion transformer. Given source video $\mathbf{x}_{\mathrm{src}}$ and instruction $\mathbf{y}$ ("Remove the gameplay UI from the video."), learnable latent queries $\mathbf{Q}$ pull task-relevant features out of the MLLM by [[Attention Is All You Need#^self-attention|cross-attention]], and a connector $\mathcal{P}$ projects them into DiT conditioning space:

$$\mathbf{c}=\mathcal{P}\!\left(\operatorname{CrossAttn}\!\left(\mathbf{Q},\mathcal{E}_{\mathrm{MLLM}}(\mathbf{x}_{\mathrm{src}},\mathbf{y})\right)\right)$$

Swapping a T5 text encoder for an MLLM is the load-bearing choice: the model has to *look* at the frame and decide "that bar is HUD, that bar is a painted wall". Text alone cannot say which pixels.

To keep the underlying scene from drifting, the source video's VAE latents are added into the noisy stream through a timestep-gated residual:

$$\mathbf{h}_{t}=\operatorname{PE}(\mathbf{z}_{t})+\gamma(t)\,\operatorname{PE}_{\mathrm{src}}\!\left(\operatorname{VAE}(\mathbf{x}_{\mathrm{src}})\right)$$

with $\gamma(t)$ a learned scalar per timestep. Training is plain flow matching:

$$\mathcal{L}_{\mathrm{flow}}=\mathbb{E}_{t,\mathbf{z}_{0},\mathbf{z}_{1},\mathbf{c}}\left[\left\|\mathbf{v}_{\theta}(\mathbf{z}_{t},t,\mathbf{c})-(\mathbf{z}_{1}-\mathbf{z}_{0})\right\|_{2}^{2}\right]$$

$\mathbf{z}_1$ = clean target latent, $\mathbf{z}_0$ = Gaussian noise. MLLM and connector frozen; [[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]] rank 64 on the DiT only. 8×H100, global batch 32, lr $1\times10^{-4}$, 3,000 steps, 720p. During post-training they randomly drop the clean reference image 20% of the time so the model also works reference-free — the same conditioning-dropout move as [[Classifier-Free Diffusion Guidance#^classifier-free-guidance|classifier-free guidance]], used here for graceful degradation rather than for a guidance scale.

**The evaluation protocol** is unusually careful and is arguably a contribution in itself. Videos are sampled at 2 FPS *by timestamp, not by frame index*, so a 16-fps 81-frame output is compared against the right physical moments of a 30-fps 150-frame source. Both sides are Lanczos-resized to the same canonical 1280×720 canvas before any cropping, so crops share pixel scale. An MLLM judge gets a full-frame overview pair plus one detail sheet per annotated element, and returns for each element: `removed / present / uncertain`, and if removed, artifact `none / minor / major`. Metrics:

- **UI** — fraction removed (uncertain counts as failure).
- **Clean** — removed *and* zero visible artifact.
- **AAR** (Artifact-Adjusted Removal) — removed with weight 1.0 / 0.5 / 0.0 for none / minor / major artifacts.
- **BG** — preservation *outside* the annotated boxes: 1.0 / 0.5 / 0.0 for preserved / minor / major change.

Aggregation is frame-macro inside a video, then video-macro across the dataset. Overall $=(\mathrm{AAR}_S+\mathrm{BG}_S+\mathrm{AAR}_W+\mathrm{BG}_W)/4$.

## Ablation Studies and Experiments

**Pilot (does UI actually hurt?).** Same 5,442 clips, two versions.

| | VQ | MQ | TA | Overall |
|---|---|---|---|---|
| Wan-1.3B ft. on clean | −0.643 | −0.298 | 1.008 | **0.068** |
| Wan-1.3B ft. on UI-overlaid | −0.661 | −0.367 | 1.018 | −0.009 |
| Δ | +2.7% | +18.8% | −1.0% | +6.83% |

On the raw training data, clean clips score LAION aesthetic 6.109 vs 5.626 (+8.59%) and RAFT motion 16.392 vs 15.706 (+4.36%). Notably, clean clips score *worse* on MUSIQ clarity (69.254 vs 70.260, −1.43%) — because crisp UI text is exactly what a no-reference sharpness metric loves. **A clarity filter in your data pipeline will actively select for UI-contaminated footage.** That is the most immediately useful finding in the paper.

**Main comparison (Table 3).** Nine baselines, mask-assisted and mask-free.

| Model | Mask | Overall | Synth AAR / BG | Wild AAR / BG |
|---|---|---|---|---|
| GameCleaner (w. Ref) | ✘ | **93.34** | 94.64 / 98.85 | **80.05 / 99.80** |
| GameCleaner (w.o. Ref) | ✘ | 85.97 | 93.17 / 99.50 | 51.60 / 99.60 |
| GameCleaner (no ref pretrain) | ✘ | 85.63 | **95.36** / 99.00 | 48.56 / 99.60 |
| VACE 1.3B | ✔ | 69.37 | 59.11 / 96.50 | 30.40 / 91.45 |
| Kiwi-Edit (the base model!) | ✘ | 60.58 | 34.68 / 97.30 | 13.54 / 96.80 |
| EffectErase | ✔ | 59.68 | 56.17 / 84.45 | 23.55 / 74.55 |
| VACE 14B | ✔ | 56.87 | 27.86 / 99.60 | 5.92 / 94.10 |
| Aurora | ✘ | 37.57 | 60.62 / 24.45 | 51.19 / **14.00** |
| Lucy-Edit | ✘ | 42.62 | 1.30 / 83.30 | 2.86 / 83.00 |
| OmniWeaving | ✘ | 35.31 | 0.95 / 86.70 | 5.27 / 48.30 |

**The interesting failure mode is not "cannot remove" — it is "removes by vandalising".** Aurora gets 51.19 wild AAR, second-best removal in the table, with a background preservation score of **14.00**. It is repainting the whole frame. EffectErase gets raw UI removal of 92.00 in the wild — better than GameCleaner's 84.75 — but its Clean score collapses to 20.42 and BG to 74.55: it touches the right region and leaves a smear. This is exactly why raw removal rate is a bad metric and AAR exists.

The fine-grained table makes it sharper. On wild, EffectErase gets 92.93 Track-All (removed the element on every frame of its track) but **16.12 Track-Clean** (removed it cleanly on every frame). GameCleaner w. Ref: 76.81 / **68.12**. Lower raw hit rate, four times more often actually clean.

**Reference-drop ablation.** With a clean reference at inference, going from 0% → 20% drop during training pushes wild AAR from **70.64 → 80.05**, at a cost of only 96.39 → 94.64 on synthetic. Pushing further breaks it: 50% drop → 56.23 wild, 80% drop → 54.64. So a *little* conditioning dropout regularises against overfitting to the synthetic reference; a lot destroys the reference pathway.

**Data-scaling ablation — the most informative one.** From 60% to 100% of the training data:
- synthetic AAR: 92.58 → 94.64 (**+2.06**)
- wild AAR: 54.96 → 80.05 (**+25.09**)

Synthetic performance saturates almost immediately; real-world generalisation does not, and has not converged at 96K pairs. The intermediate points are non-monotonic, which the authors read as *diversity* of HUD compositions mattering more than gradient steps. This is the strongest argument in the paper for the data-engine framing over the model framing — the model is a fine-tune, the data is the product.

**What did not work / what is fragile:**

- **The judge is resolution-sensitive.** Same videos, same judge, same day, three inference resolutions: Set-1 synthetic AAR of 31.03 (800×448), 28.12 (1280×720), 32.02 (1920×1080). No consistent direction, but ~4 points of swing from resizing alone. Evaluation resolution must be frozen and reported.
- **Individual judge ratings are only okay.** ICC(A,1) for AAR is 0.713 and for the `Uncertain` metric only 0.582. Aggregating five ratings rescues it (ICC(A,5) = 0.926 / 0.874). BG has the best absolute agreement (ICC(A,1)=0.834) but the worst *ranking* agreement (Kendall $W$ = 0.566), i.e. raters agree on the label but not on which model is better.
- Bigger is not better among baselines: **VACE-14B scores worse than VACE-1.3B** (56.87 vs 69.37 Overall), at 5.92 wild AAR. It preserves background beautifully and refuses to remove anything.
- Acknowledged model failures: large opaque panels, rapidly changing overlays, and UI visually entangled with the scene behind it.

## Worth Remembering

- **The reusable idea, stripped of games:** when you cannot obtain (corrupted, clean) pairs from the wild, harvest the *corruption* from the wild as reusable assets, and synthesise pairs by applying it to separately-sourced clean data. The taxonomy exists so the synthetic corruption has realistic spatial and temporal statistics rather than random pasting. This is a template for watermark removal, subtitle removal, dashcam overlay removal, UI removal in screen-recording agent data.
- **The data-scaling result is the headline for a practitioner**, not the leaderboard: +2 points synthetic vs +25 points wild for the same extra data. If you are building anything with a synth-to-real gap, measure both — synthetic saturation tells you nothing.
- **Clarity/sharpness metrics reward the artefact you want to remove.** Concrete instance of a filter that is anti-correlated with the thing it is proxying for.
- **Paper hygiene caveat, in the spirit of [[Troubling Trends in Machine Learning Scholarship]].** The prose does not match the tables. The intro claims "an average score of 0.5697, outperforming LoomVideo by 40.3%" and "only 0.0233 below mask-assisted EffectErase", numbers that appear nowhere in Table 3, where GameCleaner beats EffectErase by 33+ points. The abstract calls Aurora a "temporal mask baseline" when it is listed mask-free. The main text says 8×H100 / 3,000 steps; the appendix says 8×H200 / 1,400 steps. Trust the tables, not the sentences.
- Everything is evaluated by an MLLM judge. Human validation is decent (removed-F1 87.78%, Cohen's $\kappa$ 0.820 over 6,150 element labels) but there is no pixel-level PSNR/LPIPS against the synthetic ground truth reported, even though it exists for Game2World-S. That would have been a cheap, non-circular sanity check.
- **The loop is not closed.** The downstream proof is a text-to-video fine-tune, not an action-conditioned world model like [[Mastering Diverse Domains through World Models (DreamerV3)]] or Genie. Turning cleaned gameplay into real world-model data still needs inverse dynamics or latent actions to recover what the player pressed, which the authors admit does not exist reliably across heterogeneous games. Cleaning is necessary, not sufficient.
- Practical cost: GameCleaner runs a full diffusion trajectory per clip. To clean a million videos you need the few-step distilled version, which is explicitly left as future work — see [[Distilling the Knowledge in a Neural Network]] for the general shape of that fix.
- Open question worth chasing: is the +6.83% VideoReward gain *from removing UI*, or partly from the fact that the UI-overlaid set was synthetically composited and therefore has compositing seams a clean set does not? The pilot's control is "clean vs. clean + synthetic UI", so the model may partly be learning to avoid alpha-blending artefacts rather than HUD semantics.

## Links

Related: [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)]] · [[Hydra-0- Action Flow for Generalist World Modeling and Control]] · [[WorldMind- Decoupled Game World Model for State-Aware NPC Behavior]] · [[GigaBrain-0.7- Scaling Embodied Foundation Models to Emergent Capabilities with a Three-System Architecture]] · [[Shortcut Learning in Deep Neural Networks]] · [[Denoising Diffusion Probabilistic Models]] · [[Score-Based Generative Modeling through SDEs]] · [[Classifier-Free Diffusion Guidance]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Attention Is All You Need]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Distilling the Knowledge in a Neural Network]] · [[Troubling Trends in Machine Learning Scholarship]] · [[Playing Atari with Deep Reinforcement Learning (DQN)]]

New topics worth writing: Flow matching / rectified flow objectives, CLIP contrastive image-text pretraining, MLLM-as-a-judge evaluation protocols, Krippendorff's alpha and intraclass correlation for rater reliability, Video inpainting (STTN / E2FGVI / ProPainter), Genie and latent action models, Inverse dynamics models, Spherical k-means for diversity sampling, VideoReward and VBench for video generation evaluation, Synthetic-to-real gap measurement
