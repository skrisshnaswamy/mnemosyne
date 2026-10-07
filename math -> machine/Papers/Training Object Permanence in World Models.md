---
title: "Training Object Permanence in World Models"
authors: ["Zhang et al."]
year: 2026
arxiv: "2609.28654"
url: https://arxiv.org/abs/2609.28654
priority: Good-To-Read
read_on: 2026-09-28
tags: [paper, llm, vision]
---
## The Core Idea

Video generators make beautiful clips that break basic physics. A ball rolls behind a screen and three balls come out. A ball hits a wall with a hole half its size and sails through anyway. A support is pulled away and the object hangs in the air.

Developmental psychology has names for the two things being broken:

> [!NOTE] Object permanence (OP)
> Things keep existing, keep their identity, and keep their position while you cannot see them. Human infants show this by about 3.5 months. ^object-permanence

> [!NOTE] Object solidity (OS)
> Two solid things cannot occupy the same space. Objects do not pass through barriers, merge on contact, or hang unsupported. Infants register violations within the first half-year. ^object-solidity

The bet of this paper: these two are *upstream* of everything else. A model that lets things interpenetrate cannot produce a correct collision, and every higher-level physical story it tells inherits the error. So instead of scoring video models on general "physics realism", build a dataset whose every sample is a controlled cognitive-science trial, and check whether the failure is *trainable*.

What is new is not a model or a loss. It is a **data infrastructure**: 150 hand-written Blender scene generators, each one a parameterised experiment, split into six families (three OP, three OS). Each generator emits 10,000 samples, giving a 1.5M-sample training corpus, plus a frozen 300-question exam (2 per generator).

The clever structural choice is the **split**. Every 120-frame clip is cut at frame 60, exactly at the onset of the key physical event. The model gets the first 60 frames as context and must generate the second 60 — the occlusion, the collision, the fall, and its consequence. That turns physical reasoning into a video-to-video continuation task rather than a multiple-choice question, which means no [[Hallucination|VLM judge]] is needed; the paper cites evidence that vision-language models themselves lack core knowledge and so cannot grade it.

They then fine-tune one 16B model (PWM-WROP) on the corpus and put 14 models in front of 20 human raters in a blind pairwise study. PWM-WROP lands third overall (Elo 1679.5) and first among true continuation models, beating the next continuation system by 224 Elo — at a native output of $320\times192$ against competitors running 720p and 1080p.

## The Methodology

**Generator design.** Each of the 150 generators is a self-contained parameterised 3D scene. Parameters are split in two on purpose:

- **Structural** — object count, track shape, aperture size, occluder opacity, contact timing, occlusion duration. These *are* the difficulty. Varied across samples so a model cannot memorise one fixed outcome.
- **Surface** — colour, material, lighting, camera viewpoint. Randomised independently. These stop a model passing by matching pixels instead of reasoning.

This is nuisance-parameter randomisation in the causal-inference sense: hold the cognitive structure fixed, shuffle everything that should not matter.

**No physics engine.** Every trajectory is hand-authored as Blender keyframes. The authors decided contact frames, bounce directions and fall onsets analytically, and revised scenes whenever inspection caught interpenetration. So "ground truth physics" here means *author-verified* physics, not solver output. Rendered with EEVEE Next at $1280\times720$, 24 fps.

**The six families.**

| ID | Name | What it demands |
|---|---|---|
| OP-1 | Baillargeonian occlusion | Object goes behind an occluder, must re-emerge with identity, size, lane and direction intact |
| OP-2 | Static-scene occlusion | A moving screen covers a fixed arrangement; on reveal, count/identity/layout unchanged |
| OP-3 | Container permanence | Object hidden in a container that moves, rotates or swaps — location must be recomputed relative to the *container*, not the world |
| OS-1 | Baillargeonian obstruction | Aperture smaller than object ⇒ blocked; larger ⇒ passes |
| OS-2 | Object drop | Support removed ⇒ fall starts *immediately*, then a size filter decides pass-through or rest |
| OS-3 | Object collision | Post-impact trajectories diverge; nothing merges, vanishes or is created |

Split 90 OP generators / 60 OS generators.

**Every sample is a five-tuple:** input video (60 frames), target video (60 frames), natural-language prompt, per-frame object-pose trajectory, metadata (generator ID, sample index, seed, render config). Automated validation rejects any sample with a missing component, mismatched frame rate, or a gap/overlap at frame 60; failures are re-rendered.

**PWM-WROP.** Fine-tuned from Cosmos3-Nano (16B), architecture and tokenizer untouched — only the training signal changes. It is trained on exactly the evaluation contract: 117-frame packed clip at $320\times192$, 57 conditioning frames then 60 predicted frames, prompt as text conditioning ([[Conditional Generation]]). One epoch over 1.5M samples. No task labels, no family names, no extra supervision. Sampling at eval: UniPC, 35 steps, [[Classifier-Free Guidance|guidance]] 6.0, shift 10.0, fixed seed.

**The training stack (PWM).** Native PyTorch on one `trn2.48xlarge` — 16 Trainium2 chips as 64 logical NeuronCores. The 36-layer model is sharded on a 2D mesh: tensor parallel degree 4 × [[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models|FSDP2]] degree 16. fp32 master weights, bf16 compute ([[Mixed Precision training]]). Each transformer block is compiled once with static shapes; prompts are padded to a fixed text length so everything compiles to one shape. Inference needs only the tensor-parallel group — 29 GB of bf16 weights, 7.3 GB per core, fits a 4-core box.

**The three interface classes**, which turn out to matter more than anything else:

1. **True continuation** (4 models incl. PWM-WROP, MAGI-1 24B, LTX-2.3 Extend, Grok Imagine extend) — treat the clip as a prefix and synthesise what follows. This is the only class that actually does the task as specified.
2. **Reference-to-video** (3: Seedance 2.5, Wan 3.0 Prime, MiniMax H3) — treat the clip as a visual *reference* and regenerate the whole event on their own timeline.
3. **Edit / transfer** (7: Wan-VACE 14B, Kling O3 Pro, Runway Aleph 2, Gemini Omni Flash 1.1, Cosmos3 Super, LTX-2.3 Dev, HY-OmniWeaving) — repaint the source span frame by frame. Structurally these **cannot** depict anything after the occlusion boundary, because their output occupies the same time interval as their input.

**Inference harness.** Same two inputs to everyone (input video + verbatim prompt), target never exposed, no prompt engineering, server-side prompt expansion disabled, output admitted only after `ffprobe` confirms geometry. Models that stitch the source onto their output get trimmed (LTX-2.3 Extend drops 3.25 s, Grok drops 60 frames, Kling drops a 0.5 s pad). Short inputs are front-padded by repeating frame 1, which preserves event timing.

**Human judging.** All clips normalised to $1280\times720$, 24 fps, same bitrate. Raters see input + prompt, then two anonymised randomly-ordered continuations, and pick A / B / "about the same". Three criteria judged jointly: prompt alignment, motion plausibility, and permanence (nothing vanishes, appears, passes through, or changes colour/shape/count). Aggregated with a [[Preference Learning#The optometrist 👓|Bradley–Terry]] model, ties as half-wins, rescaled to Elo mean 1500, with 1,000 bootstrap replicates **resampled over raters** rather than over items.

## Ablation Studies and Experiments

**Rater quality control** (worth copying):

- 20 raters, qualification screen at 8/10, median 9/10
- 90.0% correct on embedded attention checks
- 93.5% self-agreement on repeated items, Cohen's $\kappa = 0.891$
- No position bias: left clip preferred in 51.4% of non-tie judgments, $p = 0.615$
- 476 of 480 judgments completed; **361 between distinct models**, 50–52 games per model

**Overall leaderboard** (Elo, mean 1500):

| Rank | Model | Class | Elo | Score rate |
|---|---|---|---|---|
| 1= | Wan 3.0 Prime | ref-to-video | 1723.6 | 77.9% |
| 1= | MiniMax H3 | ref-to-video | 1723.6 | 77.9% |
| 3 | **PWM-WROP** | continuation | 1679.5 | 73.1% |
| 4 | Seedance 2.5 | ref-to-video | 1649.6 | 69.6% |
| 5 | Runway Aleph 2 | edit | 1518.3 | 52.9% |
| 9 | Grok Imagine extend | continuation | 1457.0 | 44.2% |
| 10 | LTX-2.3 Extend | continuation | 1453.4 | 43.1% |
| 14 | MAGI-1 24B | continuation | 1248.0 | 19.2% |

Bootstrap top-1 probability: Wan 3.0 Prime 46.8%, MiniMax H3 36.4%, PWM-WROP 12.3%, Seedance 2.5 4.5%, **everyone else exactly zero**.

Three findings fall out.

**1. Interface class explains the ranking better than scale.** All three reference-to-video systems are in the top four. Ranks 5–11 are almost entirely edit/transfer models separated by only 109 Elo with overlapping intervals — statistically a single blob. The explanation offered: frame-wise repainting structurally cannot show a post-boundary event, so the edit models all fail the same way and pile up in the middle. Wan-VACE 14B, with a small disclosed parameter count, sits right there with Kling O3 Pro and Gemini. Scale is not the variable.

**2. Fine-tuning on the corpus moves a continuation model 224 Elo.** PWM-WROP 1679.5 vs next continuation system 1457.0. Its base model, Cosmos3 Super (as an edit interface), sits at 1409.2. The authors are honest that architecture also differs across these systems, so the gap is not purely attributable to training.

**3. OP and OS come apart.** Within-family ranks for PWM-WROP:

| Family | Rank | Elo |
|---|---|---|
| OP-1 occlusion | 3 | 1616 |
| OP-2 static occlusion | **1** | 1785 |
| OP-3 container | 3 | 1692 |
| OS-1 obstruction | 5 | 1604 |
| OS-2 drop | 2 | 1825 |
| OS-3 collision | **8** | 1394 |

Fine-tuning bought occlusion tracking, not contact dynamics. Meanwhile MiniMax H3 wins **100% of its games** in both OS-2 and OS-3 but is only middling on OP. The freedom to regenerate a scene from scratch helps solidity (you can author a clean collision) and hurts permanence (you no longer have to preserve what was there).

**Qualitative failures**, from same-sample comparisons on six generators. Two named failure modes:

- **Representation dropout** — a plausible scene that has lost track of what was there. Gemini Omni Flash on G43 (`three_balls_parallel_tunnels`) collapses three balls into a cluster at the tunnel exit and emits a fourth ball. Seedance 2.5 on G19 renders an oversized screen over an *empty* region while the real objects stay visible beside it, then pops all three back into existence. Seedance on G66 (`rotating_carousel_cups`) animates the 180° turntable correctly but then lifts the cup now at the front — it tracked the ball by *turntable position*, not by *which cup*.
- **Causal decoupling** — individually plausible events with no physical link between them. Seedance on G27 deletes the whole support structure instead of just the sliding plate, then leaves the ball hovering for several frames before it drifts down. Gemini on G137 (`pool_rack_break`) hallucinates a rack of 12+ balls before impact and then deletes the purple front ball on contact.

**The most common solidity failure across models**, including both reference-to-video leaders: on G10 (`size_gate`), the ball passes straight through a barrier whose hole is half its diameter. The barrier is being treated as decoration, not as a constraint.

**Automatic metrics** (secondary, all resampled to 60 frames and resized to $320\times192$ so scales match):

| Model | LPIPS ↓ | MS-SSIM ↑ | SSIM ↑ | PSNR ↑ | MSE ×10⁻³ ↓ | FID ↓ |
|---|---|---|---|---|---|---|
| **PWM-WROP** | **0.081** | **0.921** | 0.917 | 26.45 | **2.97** | 20.4 |
| MiniMax H3 | 0.105 | 0.877 | 0.938 | 27.51 | 9.09 | 14.8 |
| Wan 3.0 Prime | 0.115 | 0.861 | 0.942 | 27.15 | 4.48 | 15.1 |
| Grok extend | 0.125 | 0.852 | 0.932 | 26.98 | 5.52 | **13.6** |
| Seedance 2.5 | 0.282 | 0.616 | 0.770 | 18.37 | 28.55 | 24.6 |
| Cosmos3 Super | 0.340 | 0.594 | 0.768 | 15.03 | 40.59 | 46.3 |

PWM-WROP wins LPIPS, MS-SSIM, MSE, MAE and final-frame MSE, but loses PSNR, SSIM, and everything computed at 720p ([[Evaluating Generative Models|FID]] 20.4, final-frame LPIPS 0.166) because its $320\times192$ output is upsampled 4× first.

The authors are explicit that these metrics **do not measure the thing**. An edit model that faithfully repaints the static pre-event scene — texture, lighting, geometry — scores well on SSIM while never depicting the reappearance at all. And a continuation model that tracks the object correctly but with slightly different timing gets punished in pixel space. Note the ordering disagreement: Seedance 2.5 ranks 4th by human Elo and *second-worst* by LPIPS.

**Negative results from the systems appendix**, which are the most transferable part of the paper:

- **Load-then-shard fragments memory.** Materialising the full 14.6 GB fp32 tensor-parallel shard before FSDP2 partitions it leaves no room for the root unit's 2.5 GB reduce buffer; the first backward pass dies. Fix: interleave load and shard per module.
- **FSDP2's reduce-scatter copy-in was the bottleneck.** Default chunk-concatenation copy-in ate 7.9 s of a 15.9 s step. A row-concatenation rewrite, verified bit-identical, gave $2.25\times$.
- **Replicated gradients drift under tensor parallelism.** Parameters replicated across the TP group diverged by $5.9\times10^{-5}$ relative after 100 steps. Now explicitly synced every step.
- **Single-tensor AdamW is launch-bound.** ~10 kernel launches and 22 host syncs *per parameter*. Multi-tensor with one sync per step: 1,324 ms → 126 ms on a rank holding 662 tensors, bitwise-equal moments over 5 steps.
- **Collectives are latency-bound, not bandwidth-bound** — 200 MB in 24 ms. Lowering the FSDP degree does not help; only removing collectives does.
- **Open item, unsolved:** keeping blocks unsharded between forward and backward fails on the 2.49 GB fp32 reduce buffer for the $151{,}936\times4{,}096$ vocabulary embedding, which is replicated across the TP group. Fixes would be sharding the vocab or freezing the text embedding. Neither shipped.
- **The fused flash-attention kernel was written but not used.** Its backward pass caps at 8,192 tokens, which excludes 720p geometries, and it was never A/B'd end-to-end. Default ships the compiled [[Attention|scaled-dot-product]] path.

Net effect on throughput ($288\times512$, 30 latent frames, 4,448 tokens/sample, batch 16, medians over 10 steps after 3 warmups, loss trajectory identical step-for-step across all rows):

| Config | s/step | tokens/s |
|---|---|---|
| First working 64-core run | 15.1 | 4,705 |
| + reduce-scatter copy-in rewrite | 6.74 | 10,564 |
| + multi-tensor AdamW, prefetch depth 2 | **5.71** | **12,454** |

**Correctness gates passed before any timing was measured:** bitwise forward parity with the reference on CPU; two-layer real-weight parity of $7\times10^{-7}$; 64-rank bit-exact resume; overfit checks (continuation-half cosine 0.993–0.999 vs 0.85–0.995 for the base model, conditioning half locked at 1.000); same-seed agreement with an independent earlier port at latent cosine 0.88–0.97 over 35 sampler steps.

## Worth Remembering

**The train/test overlap is the thing to be suspicious about.** PWM-WROP was fine-tuned on 1.5M samples from *the same 150 generators* that produce the 300-question exam ("an earlier render of the same 150 generators"). Samples differ; generators do not. Every other model is judged zero-shot. So "first among continuation models" is, at least partly, a within-distribution result. What the paper actually demonstrates is that this family of failure is *learnable*, not that the learning generalises off-generator. There is no held-out-generator split.

**The interface confound is not controlled, only acknowledged.** Three structurally different tasks are being compared on one leaderboard. Reference-to-video models were never asked to continue anything — they regenerate. Edit models are physically incapable of the task. The honest reading is three separate leaderboards, and within the continuation class PWM-WROP's win is over three models that none of us would call frontier.

**Statistical power is thin.** 361 judgments across 14 models, 50–52 games each, and the per-family fits use only 36–96 games. The authors say so plainly: adjacent ranks are rarely distinguishable, and the 1,723.6 tie at the top is a real tie. Bootstrapping over *raters* rather than items is the right conservative choice, and it widens everything.

**"Physically consistent" means hand-keyframed.** No rigid-body solver anywhere. Contact timings, bounce angles and fall onsets are authorial decisions, revised by eye when interpenetration was spotted. That is arguably the right call for a cognitive-science benchmark — you want *exactly* the trial you designed — but it means the ground truth encodes the authors' physics intuitions, and errors in them are invisible.

**The two failure modes are a genuinely useful diagnostic vocabulary.** *Representation dropout* (lost track of what exists and where) versus *causal decoupling* (each event is fine, the links between them are not). They predict the OP/OS dissociation in the Elo table, and they suggest the fixes are different: the first wants persistent object state, the second wants the generator to condition later frames on the *mechanism* of earlier ones.

**The OP/OS split is the most interesting scientific result.** Fine-tuning helped occlusion a lot and collision not at all — PWM-WROP is 1st on OP-2 and 8th on OS-3. The paper connects this to Falck et al. (2020), who found continuity and solidity dissociate in *adult* human vision too. If they really need different representations, then one uniform corpus and one uniform loss will not fix both, and a note on [[JEPA]]-style latent prediction versus pixel prediction becomes relevant.

**Practical caveat if you want to use PWM-WROP:** $320\times192$, 60 predicted frames, 57-frame conditioning window. For the two 90-frame exam items the first 33 input frames fall outside that window and were simply never seen. Anything beyond the window is invisible to the model.

**Hardware caveat:** the whole stack targets AWS Trainium2 in native PyTorch without XLA tracing. The engineering findings about FSDP2 copy-in, multi-tensor AdamW and latency-bound [[Collective Communication|collectives]] transfer to any backend; the NeuronCore mesh layout does not.

**Open questions.** What does a held-out-generator split do to the 224-Elo gap? Would training on OS families specifically close the collision gap, or does contact need a different objective? Does the reference-to-video advantage survive when you force those systems to actually continue rather than regenerate? And could the per-frame trajectory arrays — released but unused in the loss here — serve as auxiliary supervision, giving the model explicit object state instead of hoping it emerges from pixels?

## Links

Related: [[Video Diffusion]] · [[Diffusion Models]] · [[Diffusion Transformer]] · [[Conditional Generation]] · [[Evaluating Generative Models]] · [[Preference Learning]] · [[Fine-Tuning]] · [[Evals]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[JEPA]] · [[Distributed Training]] · ZeRO- Memory Optimizations Toward Living Trillion Parameter Models · [[Mixed Precision training]] · [[Collective Communication]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Classifier-Free Guidance]] · [[Diffusion Sampling]]

New topics worth writing: Core knowledge (Spelke), violation-of-expectation paradigm, LPIPS as a perceptual metric, Bradley–Terry with rater-clustered bootstrap, held-out-generator splits for synthetic benchmarks, video-to-video continuation as an evaluation protocol, FSDP2 internals and copy-in cost, AWS Trainium2 / NeuronCore programming, keyframed vs solver-generated ground-truth physics
