---
title: "WithEveryone: Unified Planning and Identity Grounding for Group Image Generation"
authors: ["Hengyuan Xu", "Qixun Wang", "Yiji Cheng", "Miles Yang", "Zhao Zhong", "Wei Cheng", "Xingjun Ma", "Yu-gang Jiang"]
year: 2026
arxiv: "2608.20336"
url: https://arxiv.org/abs/2608.20336
priority: Low-Priority
read_on: 2026-09-30
tags: [paper, transformers, llm, vision, theory]
---
## The Core Idea

Ask an image model to put ten specific people in one photo and two things go wrong. It forgets who some of them are, and it duplicates faces or fuses two people into one. Existing methods top out at about four or five references; even Nano Banana Pro caps at seven.

The interesting part of this paper is not the architecture. It is a fix to a **training loss**.

Everyone agrees that just *conditioning* on a reference face is not enough. You also need a loss on the generated face: crop the face the model drew, run it through a face recogniser, and push its embedding towards the reference. PuLID does this for one person. WithAnyone does it for at most two.

That loss has a hidden prerequisite. Before you can compare face embeddings, you must know **which generated face corresponds to which reference**. With one person, there is no question. With two, you can guess by matching embeddings (Hungarian assignment). With ten, this collapses — and not gradually. During training you are looking at a noisy one-step guess of the image, where all ten faces look like the same blurry face. So the matching is close to random. Every mispairing drags identity A towards identity B. The supervision that should sharpen individuals instead cancels itself out.

The insight: **you already know the correspondence — it is in your annotation.** The model is also being trained to predict a layout, and that layout comes from the real photo, where each labelled face box belongs to a known person. So crop the predicted image *and* the target image at the same annotated box. Every crop pair is correct by construction. No matching, no matcher to get wrong, and the cost does not grow with the number of people.

> [!NOTE] Layout-Grounded ID Loss (LG-ID Loss)
> Identity supervision on generated faces where the reference↔face correspondence is read off the layout annotation (a known face bounding box per person), not inferred by comparing face embeddings in the generated image. This is what makes output-side identity loss usable at 5–10 people. ^lg-id-loss

This is the single biggest win in the paper: it alone takes reference similarity from 0.339 to 0.506.

The second half is a plumbing idea. The same unified model that generates the image first **writes down a plan in text** — who participates, where each person's face and body goes, what pose they take, which reference each planned person is bound to. A deterministic renderer draws that plan as a stick-figure canvas, which is fed back in as a conditioning image. So "who" and "where" are decided in one context instead of by two separate modules.

## The Methodology

**Backbone.** A transfusion-style mixture-of-transformers: text and structured reasoning are predicted autoregressively, target-image latents are learned with [[Flow Matching]]. 60B parameters on the understanding side, 60B on the generation side, initialised from an internal HunyuanImage 3.5-preview. One shared causal context, so anything decided before the image tokens conditions them directly.

**The training sequence**, in causal order — this is the whole system on a whiteboard:

1. Reference images + user prompt
2. Identity selection, and one ID token loaded per selected person
3. Face boxes + identity bindings, then body boxes + pose keypoints (the Layout CoT)
4. The rendered layout canvas, inserted as a condition image
5. A summary and a detailed recaption
6. One predicted identity representation per person
7. The target image

**ID tokens.** For each selected reference, take a 512-dimensional ArcFace embedding, push it through a small MLP to the model hidden size, and drop it in at one token position. The framing they use is worth keeping: reference-image patches give low-level appearance, the ArcFace vector gives high-level identity — the same split as [[An Image is Worth 16x16 Words (ViT)|ViT]] features versus [[Variational Autoencoder|VAE]] latents. References become an *addressed set* rather than an unordered pool.

**Layout CoT.** Coordinates are discretised into a 2,002-token vocabulary — 1,001 positions per axis. Stages are emitted in fixed order, so every box is conditioned on the boxes already committed. Short natural-language connectors carry scene state between stages while the spatial fields stay machine-parseable. Then a renderer draws face boxes, body boxes and pose skeletons on a blank canvas at the target aspect ratio.

> [!NOTE] Layout Chain of Thought
> A structured, autoregressively predicted plan (indices, boxes, poses, identity bindings) emitted *inside the same model* before image synthesis, then rendered deterministically into a visual condition. Unlike a separate planner + renderer, the plan and the image share one hidden state. ^layout-cot

**ID Representation Forcing.** A single continuous token per person, sitting inside a 10–20K-token sequence, is easy for the generator to ignore. So before the image tokens, the model must emit one *representation token* per identity. Its hidden state $\mathbf{h}^{\mathrm{rep}}_i$ is projected into ArcFace space and trained with a cosine loss:

$$\hat{\mathbf{e}}_i = g_{\mathrm{out}}(\mathbf{h}^{\mathrm{rep}}_i), \qquad \mathcal{L}_{\mathrm{RF}} = \frac{1}{M}\sum_{i=1}^{M}\left(1 - \cos(\hat{\mathbf{e}}_i, \mathbf{e}^{\mathrm{tgt}}_i)\right)$$

Those hidden states stay causally visible to the later image tokens — an identity scaffold. Target embeddings are training targets only, never inference-time inputs.

**The LG-ID Loss, mechanically.** Flow matching gives a velocity $\mathbf{v}_\theta(\mathbf{x}_t, t)$; the one-step clean estimate under the reverse-flow convention is

$$\hat{\mathbf{x}}_{\mathrm{clean}} = \mathbf{x}_t - t\,\mathbf{v}_\theta(\mathbf{x}_t, t)$$

Decode that with the VAE, crop at the annotated face landmarks, encode with frozen ArcFace, and take mean cosine distance $1 - \cos(\mathbf{e}^{\mathrm{pred}}, \mathbf{e}^{\mathrm{tgt}})$ over referenced identities. Bystanders with no reference are skipped.

Two details that matter:

- **Timestep gate at $t \le 0.85$.** Above that, the one-step estimate holds no recognisable face at all, so the loss is noise.
- **The circular-looking assumption.** Cropping the *predicted* image at an annotated box assumes the generated face actually lands there. That holds because plan adherence converges much earlier in training than identity fidelity — Plan IoU is near 0.79 well before identity scores stabilise.

**Joint objective.**

$$\mathcal{L} = \mathcal{L}_{\mathrm{NTP}} + \lambda_{\mathrm{FM}}\mathcal{L}_{\mathrm{FM}} + \lambda_{\mathrm{RF}}\mathcal{L}_{\mathrm{RF}} + \lambda_{\mathrm{ID}}\mathcal{L}_{\mathrm{ID}}$$

with $\lambda_{\mathrm{FM}} = 1.0$, $\lambda_{\mathrm{RF}} = 1.0$, $\lambda_{\mathrm{ID}} = 0.5$. Connector text, layout tokens and recaptions get [[Cross Entropy|next-token]] loss; latents get flow matching; identity predictions get cosine.

**Training.** 400K in-house group photos. [[Old Optimizer, New Norm- An Anthology (Muon)|Muon]], lr $1\times10^{-5}$ on the generation side and $3\times10^{-6}$ on the understanding side. Packed sequences of 72K tokens. 128 H20 GPUs, 1,600 iterations.

## Ablation Studies and Experiments

**Benchmark.** 210 real group photos, 5–10 references each, identity-disjoint: every identity in the benchmark is held out, and any training example containing them is deleted. Stratified 60/50/40/30/20/10 across group sizes 5–10.

**Metrics worth knowing.** `Sim(Ref)` and `Sim(Tgt)` are face similarity to the reference photo and to the person *as they appear in the target scene*, averaged over ArcFace, FaceNet and AdaFace. `Coverage` is the fraction of references realised as a recognisable distinct face. `Dup` is the fraction that collapse onto a face another identity already claimed.

> [!NOTE] Copy-Paste score
> $\mathrm{Copy\text{-}Paste} = (\theta_{gt} - \theta_{gr}) / \theta_{tr}$, where $\theta$ is angular distance in face-embedding space between generated, target and reference. High means the generated face is closer to the *reference crop* than to how that person actually looks in the new scene — i.e. it was pasted, not rendered. Critical, because Sim(Ref) alone is trivially gamed by literal copying. ^copy-paste-score

**Main table (210 examples):**

| Method | Sim(Tgt)↑ | Sim(Ref)↑ | Copy-Paste↓ | Coverage↑ | Dup↓ | CLIP-I↑ |
|---|---|---|---|---|---|---|
| WithEveryone | **0.499** | 0.540 | 0.055 | **0.973** | **0.028** | 0.861 |
| GPT-Image 2 | 0.462 | **0.583** | 0.169 | 0.905 | 0.075 | 0.853 |
| Nano Banana 2 | 0.451 | 0.480 | 0.045 | 0.884 | 0.099 | 0.860 |
| Seedream 5.0 Pro | 0.436 | 0.522 | 0.114 | 0.913 | 0.065 | 0.850 |
| WithAnyone | 0.405 | 0.483 | 0.096 | 0.957 | 0.045 | 0.807 |
| HiDream-O1 | 0.353 | 0.376 | 0.026 | 0.780 | 0.190 | 0.806 |
| FLUX.2 Klein | 0.264 | 0.265 | 0.002 | 0.314 | 0.265 | 0.787 |

Read the GPT-Image 2 row properly. It *beats* WithEveryone on Sim(Ref) — 0.583 vs 0.540 — while scoring worse on Sim(Tgt) and having 3× the Copy-Paste. That pattern is the signature of pasting the reference crop in rather than re-rendering the person under the new pose and lighting. The paper's claim is a better balance, not a clean sweep.

Note the open-source general-purpose models: Coverage mostly below 0.42. They are not failing at identity so much as failing to compose a group at all — they edit the two or three people the prompt implies and stop.

**The ablation ladder (all at 1K, ArcFace only):**

| | Variant | Sim(Ref)↑ | Sim(Tgt)↑ | Count↑ | Coverage↑ |
|---|---|---|---|---|---|
| P1 | Default (no layout, no ID token) | 0.339 | 0.304 | 0.771 | 0.741 |
| P2 | + Layout CoT (model's own plan) | 0.364 | 0.316 | 0.828 | 0.813 |
| P3 | + Layout CoT (ground-truth plan) | 0.412 | 0.367 | 0.958 | 0.891 |
| P4 | + ID token | 0.351 | 0.313 | 0.827 | 0.761 |
| P5 | + ID token + Rep. Forcing | 0.364 | 0.328 | 0.817 | 0.782 |
| P6 | + LG-ID Loss only | **0.506** | **0.435** | 0.845 | 0.947 |
| P7 | Full model | 0.555 | 0.461 | 0.869 | 0.960 |

What this actually says:

- **The loss is doing nearly all the work.** P1 → P6 is +0.167 Sim(Ref). Everything else combined is worth a fraction of that.
- **ID Representation Forcing is nearly free of effect.** +0.013 Sim(Ref), +0.015 Sim(Tgt) over P4. The authors are honest about it: "consistent with improved identity addressability rather than a large direct source of identity gain." The attention plot (diagonal 0.378–0.574 vs 0.104–0.121 off-diagonal at layer 24) shows routing happens, but with no no-RF control and no measurement from *generated face patches* to ID tokens, it is suggestive only.
- **P3 is an oracle, not an achievable number.** The ground-truth layout is derived from the target image. Its value is to prove that a good plan improves identity as well as composition — and to size the gap that better planning could close.
- **The layout gain is a lower bound.** The LG-ID Loss also improved `Count` (0.771 → 0.845) and `Coverage` (0.741 → 0.947) *without any layout conditioning*, because the only way to reduce a region-grounded identity loss is to put the right person in the right place. So the loss carries spatial signal, and the credit assigned to Layout CoT is understated.
- **P7 is not a clean attribution.** It adds several components *and* an extra text-to-layout corpus at once. The paper says so.

**What did not work, or was left on the table:**

- **CLIP-I is a tie, not a win.** 0.861 vs Nano Banana 2's 0.860: bootstrap interval $[-0.006, +0.009]$, Wilcoxon $p = 0.98$, higher on 48% of examples. The bolding in the table is misleading and they flag it themselves. The Sim(Tgt) lead, by contrast, is real: $+0.038$, interval $[+0.027, +0.048]$, $p = 3.8\times10^{-11}$, wins on 73% of examples.
- **$\lambda_{\mathrm{ID}} = 0.5$ is under-tuned.** At $\lambda_{\mathrm{ID}} = 1.0$, Sim(Ref) reaches 0.559–0.561 versus 0.537–0.548 at 0.5 — with no measurable cost in layout quality. They shipped the weaker setting.
- **Representation Forcing does not actually reconstruct identity.** Training loss falls from ~1.0 to 0.02, but at inference the predicted embedding's cosine to the target identity only reaches 0.212 by step 1,800, against a reference↔target baseline of ~0.40. It learns a direction, not the person.
- **Text alignment is not best.** CLIP-T 0.273, behind ID-Patch (0.331) and UniPortrait (0.301) — both of which are far worse on every identity metric. Prompt-following gets easier when you are not required to match anybody in particular.

**The face-size analysis, which is the best diagnostic in the paper.** Obvious hypothesis: identity degrades with group size simply because faces get smaller. They test it. Baselines gain 0.016–0.019 similarity per percentage point of relative face side; WithEveryone's slope is $-0.002$ — essentially flat. Going from 5 to 10 references only moves the mean relative face side from 11.9% to 9.4%, and across that narrow band Nano Banana 2 loses 0.040 while WithEveryone gains 0.006. Regressing similarity on reference count with and without face size as a covariate: the group-size slope mostly vanishes for GPT-Image 2 and Seedream, but WithEveryone's stays at $-0.014$. So *their* remaining degradation is something else — cross-identity interference, occlusion, compositional load. They do not claim to know which.

In absolute pixels the advantage is concentrated on small faces: 112–150 px gives 0.611 vs GPT-Image 2's 0.561; GPT-Image 2 only overtakes in the 200–260 px band, where just 8% of its faces live.

**Planning vs execution.** Plan IoU with the model's own plan is 0.773 (0.814 when trained at 2K) — the image genuinely realises what was planned. Yet swapping in the ground-truth plan still lifts identity and coverage a lot. Plus plan execution converges *before* identity in training. Conclusion: the residual error is in **proposing** the plan, not in **following** it. That is a useful thing to know before spending compute on the generator.

## Worth Remembering

**The transferable lesson has nothing to do with faces.** If a loss requires a correspondence between predicted and target items, and you are recovering that correspondence by similarity matching on noisy predictions, the loss degrades as the number of items grows — and it degrades by actively pulling items towards each other, not by going quiet. If the correspondence exists anywhere in your annotation, read it from there. This generalises to any set-prediction objective supervised under noise.

**The self-consistency trick.** LG-ID Loss crops the *prediction* at the *ground-truth* box, which only works if the prediction already obeys the box. They get away with it because two objectives converge at different speeds — layout adherence first, identity fidelity second. That staggering is load-bearing and is not guaranteed in another setup. Worth checking before copying the pattern.

**Two similarity metrics, and you need both.** `Sim(Ref)` rewards copying the reference crop. `Sim(Tgt)` rewards rendering the same person under new lighting and pose. A method can win the first by doing something you do not want. `Copy-Paste` is the disambiguator, and Nano Banana 2's low 0.045 shows that a low score can also just mean low similarity overall.

**Coverage and Dup deserve more attention than similarity.** Facebook-style "which of the requested people actually appeared, as separate humans" is a different failure mode from "does the face look right", and most open models fail the first one outright. The 0.20 similarity threshold for coverage is permissive, but the authors sweep it: raising it to 0.35 drops P7's coverage by only 0.027, and only 3.6% of matched references sit in $[0.15, 0.30)$.

**Caveats for anyone wanting to use this.**
- 210 examples on one benchmark, with the *largest* group sizes having the *smallest* sample (10 examples at ten people). Per-group numbers are trends.
- All identity metrics inherit demographic bias from the face detectors and recognisers underneath. The paper says this plainly.
- 120B parameters total on 128 H20s. This is not reproducible in an academic lab.
- Layout evaluation is fundamentally awkward when prompts are underspecified — many arrangements are equally valid and there is no single reference to score against. They leave this open.
- Weights: a code link is promised, but the backbone is an unreleased internal HunyuanImage variant.

**Follow-up questions I'd want answered.** Does the layout-grounded correspondence trick survive when the annotation is itself a model output (detector boxes on messy web photos) rather than clean ground truth? And since the bottleneck is plan *proposal*, what happens if you train the understanding side with RL against the Relative Layout Score instead of only next-token prediction?

## Links

Related: [[Flow Matching]] · [[Denoising Objective]] · [[Diffusion Models]] · [[Conditional Generation]] · [[Controlling Diffusion]] · [[Latent Diffusion]] · [[Diffusion Transformer]] · [[Chain of Thought]] · [[Planning and Decomposition]] · [[Structured Output]] · [[Embeddings]] · [[CLIP]] · [[Variational Autoencoder]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Old Optimizer, New Norm- An Anthology (Muon)]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[JEPA]] · [[Evaluating Generative Models]] · [[Auto-regressive models]] · [[Cross Entropy]] · [[Cross Attention]]

New topics worth writing: ArcFace and additive angular margin loss, identity-preserving image generation, Hungarian assignment for set-prediction losses, Mixture-of-Transformers architectures, Representation Forcing, ControlNet-style pose-keypoint conditioning, copy-paste artifact metrics for personalisation
