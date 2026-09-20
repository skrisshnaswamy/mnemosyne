---
title: "LLaDA-Image: Building Strong Image Generators with Fully Open Training Recipes"
authors: ["Chuyan Chen", "Haoxing Chen", "Kun Chen", "Zhenglin Cheng", "Long Cui", "Ruishan Fang", "Zhangxuan Gu", "Zhicheng Huang", "Zhenzhong Lan", "Yuanting Lei", "Haoquan Li", "Jianguo Li", "Rongchuan Li", "Sidu Li", "Tao Lin", "Deyuan Liu", "Jiacheng Liu", "Lin Liu", "Yuxuan Lou", "Zhisheng Lu"]
year: 2026
arxiv: "2609.03796"
url: https://arxiv.org/abs/2609.03796
priority: Good-To-Read
read_on: 2026-09-06
tags: [paper, transformers, llm, optimization, diffusion, vision]
---
## The Core Idea

Most text-to-image models learn "what things look like" and "what words mean" at the same time. That forces you to have a caption for every training image, right from step one. Captions are expensive, they are lossy, and they create a subtle bug: at low training resolution (say $256\times256$) you must shrink the *whole* image so it still matches the caption, and the fine details the caption mentions get destroyed by the downsampling.

LLaDA-Image splits those two jobs apart. First, teach the generator to make images using **only images**. No captions at all. The trick is that an image already contains its own description — you just need something that can read it. A frozen vision-language model looks at the image, produces a semantic summary, and that summary becomes the conditioning signal for the diffusion transformer that must reconstruct the same image. Condition and target come from the same pixels, so they can never disagree.

Because the condition is derived from the crop itself, you are free to take a *crop* of a high-resolution photo rather than squashing the whole thing. The model sees real local detail instead of mush.

Only after this visual prior is built do you introduce captions, and only for a small slice of training. Of 220M generation-training samples, **more than 90% are image-only** and **98% are real photographs** (not synthetic). Language alignment is a late, cheap finishing stage rather than the expensive foundation.

> [!NOTE] Image-only pre-training ^image-only-pretraining
> Train a conditional image generator without any paired text, by using a frozen VLM's reading of the image as the condition for regenerating that same image. Mask most of the VLM's image tokens so the task is not a trivial copy.

The second thing the paper unlocks is a genuinely single checkpoint that does understanding, text-to-image, and instruction editing, plus a distilled 2–4 step version. Result: **53.53 / 53.38** overall on the English/Chinese tracks of Qwen-Image-Bench, the best open-source score on both, beating Z-Image Turbo by 1.87 and 0.67 points.

## The Methodology

Three parts: a frozen VLM, a connector, and a 6B [[An Image is Worth 16x16 Words (ViT)|transformer]] diffusion model.

**The understanding side.** A vision-language model built on LLaDA 2.0 Mini — a *diffusion* language model, not an autoregressive one — with a SigLIP-VQ vision encoder. It is frozen during all generation training. Before generation training begins it gets one supervised fine-tuning pass with chain-of-thought data (2.6M packed 16,384-token sequences, 4 epochs, [[Decoupled Weight Decay Regularization (AdamW)|AdamW]] at $1\times10^{-5}$, mixture generation : understanding : text $= 9{:}9{:}2$). Its training loss is a [[BERT- Pre-training of Deep Bidirectional Transformers|masked-token]] objective in blocks of 32: sample a mask ratio $\rho = \cos(r\pi/2)$, $r\sim\mathcal{U}(0,1)$, mask that fraction of the answer, and take [[Cross Entropy|cross-entropy]] on masked positions only. Each batch is used twice — once with the mask, once with its complement — so every response token gets supervised.

**Getting information out of a frozen model.** A frozen VLM will not hand you the fine visual details a generator needs. Two small modules fix this:

1. **Residual Query Adapter (RQA).** A set of learnable query tokens $\mathbf{q}_0$ cross-attends over the input sequence $\mathbf{c}$: $\mathbf{q}_{\mathrm{res}} = \bm{q}_\psi(\mathbf{q}_0, \mathbf{c})$. These are *appended* to the input, not substituted, and the whole thing goes through the frozen backbone in one prefill: $\mathbf{h}_{\mathrm{vlm}} = \bm{g}(\mathrm{concat}(\mathbf{c}, \mathbf{q}_{\mathrm{res}}))$. So the queries act as a learned prompt that coaxes generation-relevant activations out of a model nobody is allowed to touch.
2. **Connector.** A shallow stack of transformer blocks maps $\mathbf{h}_{\mathrm{vlm}}$ into the DiT's conditioning space: $\mathbf{h}_{\mathrm{cond}} = \bm{c}_\phi(\mathbf{h}_{\mathrm{vlm}})$.

**The generator.** A pure single-stream DiT: condition tokens and noisy image tokens are put in one sequence and go through the same blocks with joint [[Attention Is All You Need|self-attention]]. No separate text branch. Two choices they call out explicitly:

- **Every normalisation layer is parameter-free RMSNorm** — no learned scale, no bias. Their claim is that this alone substantially improves stability over long training runs. (Compare [[Layer Normalization]].)
- **[[Old Optimizer, New Norm- An Anthology (Muon)|Muon]]** everywhere in generation training, with weight decay set to 0.

**Objective.** Standard flow matching. With $\mathbf{z}\sim\mathcal{N}(0,I)$, $t\sim\mathcal{U}(0,1)$, and $\mathbf{x}_t = (1-t)\mathbf{x} + t\mathbf{z}$:

$$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}\big[\|\bm{F}_\theta(\mathbf{x}_t, t, \mathbf{h}_{\mathrm{cond}}) - (\mathbf{z}-\mathbf{x})\|_2^2\big]$$

Note the convention: **larger $t$ means more noise**, and the model predicts $\mathbf{z}-\mathbf{x}$, so a one-step jump from noise to data is $\hat{\mathbf{x}} = \mathbf{z} - \bm{F}_\theta(\mathbf{z}, 1, \cdot)$. Related to [[Denoising Diffusion Probabilistic Models]] and [[Score-Based Generative Modeling through SDEs]].

**Masked self-conditioning (the pre-training stage).** Encode the image into $N$ patch features $\mathbf{c}_{\mathrm{img}} = \bm{v}(\mathbf{y})$. Pair with a fixed auxiliary prompt like *"Generate an image identical to the reference image."* Then throw most of the patches away: sample a binary mask $\mathbf{m}\in\{0,1\}^N$ with keep probability $1-\rho_{\mathrm{img}}$ and use $\tilde{\mathbf{c}}_{\mathrm{img}} = \mathbf{c}_{\mathrm{img}}\odot\mathbf{m}$. Without masking the model would learn an identity map — the condition would *be* the answer. With masking it becomes a sparse-to-dense prediction problem, and the model must learn compositional structure. This is the same intuition as [[Self-Supervised Learning from Images with I-JEPA|I-JEPA]], but the prediction target is pixels through a flow-matching head. Only $\bm{F}_\theta$, RQA, and connector get gradients; $\bm{v}$ and $\bm{g}$ stay frozen.

**The training curriculum**, in order:

| Stage | Resolution | Supervision | LR | Batch |
|---|---|---|---|---|
| Pre-training | $256^2$ square crops | image-only | $4\times10^{-4}$ | 24,576 |
| Mid-training | $512^2$ buckets | image-only | $2\times10^{-4}$ | 6,400 |
| SFT align | $512^2$ buckets | image–text | $5\times10^{-5}$ | 4,608 |
| SFT scale | $1024^2$ buckets | image–text | $3\times10^{-5}$ | 2,048 |
| Refine | $1024^2$ | text-rich + portraits | — | 2,880 |
| Editing | $1024^2$ | T2I : I2I = 1:1 | — | 2,688 |
| Distill | — | TwinFlow | $5\times10^{-6}$ | 256 |

Mid-training exists purely to avoid changing two things at once. Going straight from $256^2$-image-only to $1024^2$-with-captions changes both the resolution *and* the source of supervision in one jump. Mid-training moves the resolution first, keeping the familiar image-derived condition.

**Aspect-ratio buckets.** From mid-training onward, a bucket list of $(W,H)$ pairs all with roughly the same pixel count (from $256\times1024$ up to $1024\times256$ at the $512^2$ budget). Each image goes to the bucket with the closest aspect ratio $k^* = \arg\min_k |W/H - W_k/H_k|$, gets resized to *cover* that bucket, and the few overflow pixels are cropped. Three wins: every data-parallel rank sees roughly the same token count (no memory spikes, no stragglers — see [[GPU processing]]), no geometric distortion from square-squashing, and the model learns to output many aspect ratios.

**Logit-normal timestep sampling** during SFT: $t = \mathrm{sigmoid}(P_{\mathrm{mean}} + P_{\mathrm{std}}\epsilon)$ with $P_{\mathrm{mean}}=P_{\mathrm{std}}=0.8$, biasing training toward high noise. The reasoning: errors made early in the denoising trajectory propagate through everything after, whereas at low $t$ the image is already mostly resolved.

**Editing.** The reference image *never enters the VLM*. Only the instruction text goes through the RQA–VLM–connector path. The reference gets two separate injections straight into the DiT:

- *Semantic:* $\mathbf{f}_{\mathrm{ref}} = \bm{v}(\mathbf{y}_{\mathrm{ref}})$, then a small branch (one embedder + two transformer layers) $\mathbf{h}_{\mathrm{ref}} = \bm{b}_\omega(\mathbf{f}_{\mathrm{ref}})$, concatenated with $\mathbf{h}_{\mathrm{cond}}$.
- *Pixel:* the **clean** reference encoded by the FLUX.2 [[Auto-Encoding Variational Bayes (VAE)|VAE]] and concatenated with the noisy target latent before the input embedder: $\mathbf{x}_t^{\mathrm{edit}} = \mathrm{concat}(\mathbf{x}_t, \mathbf{x}_{\mathrm{ref}})$.

Semantic features tell you *what is there*; the clean latent gives literal pixel evidence for the regions that should not change. Editing is trained mixed 1:1 with plain text-to-image, as capability replay — editing data alone is a narrow distribution and continued training on it erodes the open-ended generation prior.

**Checkpoint merging.** Near convergence, benchmark scores of adjacent checkpoints oscillate even though real capability is flat, because the data mixture the model sees varies step to step. They average the weights of several converged checkpoints, in the spirit of stochastic weight averaging, which smooths this out.

**TwinFlow distillation.** This is the clever bit of the fast variant. Distribution-matching distillation (DMD2) minimises a reverse [[KL Divergence]] between the student's output distribution and the target, and needs *two* score models: a frozen "real" score and an online "fake" score that chases the student's moving distribution. TwinFlow puts both roles in **one** backbone by using the **sign of the time input as a role flag**. Positive $+t$ = generator; negative $-t$ = fake-score estimator.

- One-step sample: $\hat{\mathbf{x}} = \mathbf{z} - \bm{F}_\theta(\mathbf{z}, +1, \mathbf{h}_{\mathrm{cond}})$.
- Fake-score loss (sample detached): $\mathcal{L}_{\mathrm{fake}} = \mathbb{E}\|\bm{F}_\theta(\tilde{\mathbf{x}}_t, -t, \cdot) - (\mathbf{z}' - \mathrm{sg}(\hat{\mathbf{x}}))\|_2^2$.
- DMD loss (gradients kept through $\hat{\mathbf{x}}$): $\mathcal{L}_{\mathrm{DMD}} = \mathbb{E}[w(t)\langle \mathrm{sg}(\mathbf{s}_{\mathrm{fake}} - \mathbf{s}_{\mathrm{real}}), \hat{\mathbf{x}}_t\rangle]$, where velocity is converted to score by $\mathbf{s}_t = -[\hat{\mathbf{x}}_t + (1-t)\hat{\mathbf{v}}_t]/t$.

Two output heads share the backbone; at inference the fake-score head is thrown away, so there is zero serving overhead. Generator and fake-score alternate at a **1:2** ratio (DMD2 used 1:5 — fewer fake-score updates, cheaper training). Four-step backward simulation aligns training inputs to the student's actual inference path. Compare [[Distilling the Knowledge in a Neural Network]] and the adversarial framing of [[Generative Adversarial Networks]] — the fake-score model plays a discriminator-like role, but the objective is a KL, not a min-max game.

## Ablation Studies and Experiments

This is a system report, not an ablation paper. There is no controlled table isolating each component. What exists is a set of asserted "recipes" plus benchmark comparisons. Take the recipes as engineering claims backed by internal experience, not evidence.

**Qwen-Image-Bench** (1,000 stratified prompts, five axes; no prompt rewriting, no test-time thinking):

| Model | EN overall | CN overall |
|---|---|---|
| GPT-Image 2 (closed) | 65.23 | 64.69 |
| Nano-Banana 2.0 (closed) | 59.59 | 59.82 |
| **LLaDA-Image** | **53.53** | **53.38** |
| Z-Image Turbo | 51.66 | 52.71 |
| Qwen-Image 2512 | 51.32 | 52.06 |
| LLaDA-Image Turbo (4 steps) | 50.98 | 50.27 |

Best open-source on both tracks, first in Quality, Aesthetics and Alignment among open models. But the gap to GPT-Image 2 is ~12 points, and the weakest sub-scores are Real-world Fidelity (43.90 EN, vs 59.40 for GPT-Image 2) and Creative Generation (51.09 vs 75.34). Distillation to 4 steps costs about 2.5 points overall, and the loss is concentrated exactly in those two weak dimensions — Creative Generation drops 51.09 → 45.75.

**Text rendering.** LongText-Bench: 0.923 EN / 0.913 ZH. That is *below* Qwen-Image 2512 (0.956 / 0.965) and Boogu-Image. The interesting property is balance — most models are lopsided (GPT-Image 1 High: 0.956 EN but 0.619 ZH; FLUX.1 [Dev]: 0.607 EN, **0.005** ZH). On CVTG-2K, 0.875 average word accuracy (second overall), and it degrades gracefully with more text regions: 0.892 (2 regions) → 0.857 (5 regions).

**GenEval, 0.85 overall.** Perfect 1.00 on single object, 0.98 on two objects, tied-best 0.84 on attribute binding — and **0.53 on counting**, dragging the aggregate down. Qwen-Image gets 0.89 on counting. This is the clearest weakness the numbers expose. DPG-Bench: 87.48 overall, balanced across entity/attribute/relation. The authors explicitly say they do not treat GenEval or DPG as primary evidence — the scores are saturated and rank-order diverges from human preference — and include them only for continuity with prior literature. That framing is honest, though it also conveniently sits next to their weakest counting number.

**Editing, GEdit-Bench:** overall 7.336 EN / 7.294 CN, well behind FireRed-Image-Edit (7.943) and Qwen-Image-Edit 2511 (7.877). The decomposition is the informative part: semantic consistency $G_{\mathrm{SC}} = 8.043$ is competitive, but perceptual quality $G_{\mathrm{PQ}} = 7.182$ is the lowest of every model in the table except its own Turbo variant. **The model understands what you asked for and does it; the resulting image just does not look as good.** Turbo drops further to 7.024 / 6.898.

**CoT SFT on the understanding backbone** is the only stage with a proper before/after table, and it is mixed. Big wins: MMBench-EN 81.5 → 86.1, MMBench-CN 81.2 → 86.2, VL-RewardBench 47.8 → 55.7, ChartQA 80.1 → 82.8, OCRBench 75.7 → 77.6. But **six benchmarks got worse**: RealWorldQA 66.7 → 64.1, SimpleVQA 44.0 → 42.1, MathVision-mini 26.7 → 24.3, DocVQA 89.5 → 88.2, AI2D 82.0 → 80.4, CountBenchQA 86.0 → 84.2. That last one is worth pausing on — the understanding backbone got *worse at counting*, and the generator's worst GenEval category is counting. Suggestive, not proven, but it is the kind of connection worth chasing.

**What they claim did not work, without showing it.** Three assertions:

- Synthetic-heavy data mixtures converge *faster* on early benchmarks but plateau lower and produce less realistic images. Hence >70% real images throughout SFT. No curve is shown.
- Training on editing data alone causes overfitting to the edit objective and forgetting of the general visual prior. Hence the 1:1 T2I replay.
- Conditioning on *all* image tokens during pre-training collapses into a trivial identity map. Hence masking. The masking ratio $\rho_{\mathrm{img}}$ is never given a value or swept.

## Worth Remembering

- **The core reframing is worth internalising even if you never train an image model.** "Paired data is expensive" is usually treated as a data-collection problem. Here it is treated as a curriculum problem: identify which capability actually needs the expensive supervision (language grounding), and build everything else first with cheap supervision. Over 90% of the compute-heavy training needed zero captions.
- Deriving the condition from the same crop you are asking the model to reconstruct is a **guarantee**, not a heuristic. There is no caption–image mismatch possible. That is what buys the freedom to crop instead of downsample.
- The editing design — semantic tokens *and* raw clean VAE latent, concatenated at different points — reflects a real lesson: high-level features tell you what the object is, but they cannot tell you the exact texture of the wall you were not supposed to change. You need literal pixels for preservation.
- Sign-of-time as a role indicator in TwinFlow is a neat parameter-saving trick. One backbone plays both the student and the moving discriminator; one head is discarded at deploy.
- Parameter-free RMSNorm is stated as the number-one recipe and never ablated. If you were reproducing this, that is the first thing I would test, because it is cheap to test and the claim is strong.
- **Admitted limitations:** weak world knowledge for culturally specific or specialised subjects; text rendering degrades with more regions and longer strings; training tops out at a $1024^2$ pixel budget, so 2K is an *extrapolation*, not an evaluated capability.
- Practical caveat: this is a 6B DiT *plus* a frozen VLM *plus* SigLIP-VQ *plus* the FLUX.2 VAE. The 6B figure is the generator only. FP8 variants of both Base and Turbo are released.
- The checkpoint-merging observation is a good general reminder: **benchmark oscillation near convergence is often data-order noise, not capability change.** Averaging a few adjacent checkpoints is nearly free and removes a source of false confidence in model selection.
- Open question I would want answered: does the image-only prior actually beat a caption-trained prior at equal compute, or is it just *cheaper* at equal quality? The paper only shows the endpoint. The absence of that head-to-head is the biggest hole in the story.

## Links

Related: [[Denoising Diffusion Probabilistic Models]] · [[Score-Based Generative Modeling through SDEs]] · [[Attention Is All You Need]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Old Optimizer, New Norm- An Anthology (Muon)]] · [[Layer Normalization]] · [[Distilling the Knowledge in a Neural Network]] · [[KL Divergence]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Self-Supervised Learning from Images with I-JEPA]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Generative Adversarial Networks]] · [[Classifier-Free Diffusion Guidance]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Cross Entropy]] · [[GameWAM- A World Action Model for Video Games]] · [[UniSpace- Unified Visual Representation and Scalable Multimodal Modeling]] · [[Keep-or-Drop- Adaptive Tokenizer for Compact Video Representation]] · [[GPU processing]]

New topics worth writing: Flow matching and rectified flow, Distribution Matching Distillation (DMD/DMD2), Diffusion language models (LLaDA / masked block diffusion), Stochastic Weight Averaging and checkpoint merging, RMSNorm, Aspect-ratio bucketing for variable-resolution training, Logit-normal timestep sampling, SigLIP, Vector-quantised image tokenizers, Single-stream vs dual-stream DiT conditioning
