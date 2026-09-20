---
aliases:
  - CFG
  - Classifier Free Guidance
  - Guidance
  - Guidance Scale
  - CFG Scale
  - Classifier Guidance
  - Negative Prompt
  - Negative Prompts
  - Guided Diffusion
  - Guided Sampling
tags:
  - generative-models
  - diffusion
  - inference
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** At every sampling step, run the denoiser **twice** — with the prompt and without — and **exaggerate the difference**: $\hat\varepsilon = \varepsilon_\text{uncond} + w\,(\varepsilon_\text{cond} - \varepsilon_\text{uncond})$. The scale $w$ trades **fidelity to the prompt** against **diversity**.
> **Metaphor:** A caricaturist. Work out how this face differs from an average face — then push that difference further.
> **Where it bites:** The "CFG scale" slider (7.5 by default in Stable Diffusion), negative prompts, oversaturated "deep-fried" images at high settings — and why guided sampling costs **two** network passes per step.

---
A text-conditioned diffusion model ([[Conditional Generation]]), sampled the honest way, is a let-down. Ask for *"a corgi wearing a top hat"* and you get… a dog, probably, in a vaguely formal setting. Plausible. Only loosely on-brief.

It isn't undertrained. It's doing its job *correctly*: sampling from $p(x \mid c)$ — the full, honest variety of images people have captioned that way. And the internet's captions are loose, so that distribution is broad.

But you didn't want a fair sample of everything that caption has ever been attached to. You wanted the image that is **unmistakably** that.

~={blue}How do you ask a model for something *more typical of the prompt than the data itself*?=~

---
# The caricaturist ✏️

![[cfg_two_predictions_extrapolate.png]]

A caricaturist doesn't draw your face. They work out **how your face differs from the average face** — slightly bigger nose, heavier brow — and then **exaggerate the difference**.

Do that to the denoiser. At each step, ask it twice:

- $\varepsilon_\text{cond} = \varepsilon_\theta(x_t, t, c)$ — *"denoise this, knowing it's a corgi in a top hat"*
- $\varepsilon_\text{uncond} = \varepsilon_\theta(x_t, t, \varnothing)$ — *"denoise this, knowing nothing"*

The difference between them is precisely **what the prompt contributes**. Push along it:

$$\hat\varepsilon = \varepsilon_\text{uncond} + w \cdot (\varepsilon_\text{cond} - \varepsilon_\text{uncond})$$

| $w$ | Meaning |
|---|---|
| 0 | ignore the prompt |
| 1 | plain conditional sampling — the honest $p(x \mid c)$ |
| 3–8 | **extrapolate**: more corgi-in-a-top-hat than the data |
| 15+ | caricature: burnt colours, harsh contrast, artefacts |

> [!NOTE] Classifier-free guidance
> A sampling technique that combines conditional and unconditional predictions from **one** network to sharpen conditioning. In score terms it samples from $\tilde{p}(x \mid c) \propto p(x)\,p(c \mid x)^{w}$ — the conditional, with the *"does this match the prompt?"* factor raised to a power. (Ho & Salimans, 2021 → [[Classifier-Free Diffusion Guidance]].) ^cfg-def

> [!SUCCESS] Core idea
> ~={pink}Guidance isn't a better sample from the model's distribution — it's a sample from a **deliberately sharpened** one.=~ You give up coverage of everything the prompt *could* mean to concentrate on what it *most clearly* means. It's the same dial as lowering the [[Sampling Parameters|temperature]] of an LLM: less variety, more typicality. ^sharpened-distribution

---
# Where the unconditional model comes from

You don't train two networks. During training, **randomly blank the caption ~10% of the time** (replace it with an empty/null token). The one network learns both jobs. It's a single line in the data loader, and everything above depends on it.

**Why "classifier-free"?** Because the 2021 predecessor, *classifier guidance*, needed a separate image classifier, trained on **noisy** images, whose gradient $\nabla_x \log p(c \mid x_t)$ was added to the score. It worked (*"Diffusion Models Beat GANs"*), but needed an extra model and only handled class labels. CFG gets that same gradient for free from Bayes' rule — $\nabla \log p(c \mid x) = \nabla \log p(x \mid c) - \nabla \log p(x)$ — i.e. **cond minus uncond**.

---
# What the dial really does

Exact scores, no neural network — a target where the "prompt" selects the two blue clusters:

![[cfg_guidance_scale.png]]
> [!TIP] Reading the chart
> $w = 0$: the prompt is ignored — only **52%** of samples land on-prompt. $w = 1$: **96%**, with an honest spread of 0.82. $w = 3$: **100%**, spread 0.72. $w = 8$: still 100% — but the spread has fallen to **0.63** and the samples have been pushed *past* the clusters, **away from the grey ones**: the share actually inside the target circles *drops* from 89% to 85%. ~={blue}Guidance doesn't pull you toward the prompt so much as push you away from everything else — and at high settings it overshoots.=~

| Turn $w$ up and you get | |
|---|---|
| ✅ much stronger prompt adherence | |
| ✅ higher perceived quality (it favours typical, clean images) | |
| ❌ **less diversity** — four seeds, four near-identical pictures → [[Mode Collapse]] | |
| ❌ oversaturation, blown contrast, "deep-fried" look — the prediction leaves the range the model was trained on | |
| ❌ FID gets *worse* even as each image looks better → [[Evaluating Generative Models]] | |

---
# Negative prompts — the same formula, abused cleverly

Nothing says the second pass has to be *empty*. Put **"blurry, extra fingers, watermark"** there instead:

$$\hat\varepsilon = \varepsilon_\text{neg} + w\,(\varepsilon_\text{pos} - \varepsilon_\text{neg})$$

Now the sampler extrapolates *away from* the negative prompt and toward the positive one. No retraining, no new model — a different argument to the same two-pass trick.

---
# Practicalities 🛠️

> [!WARNING] It doubles your inference bill
> Two forward passes per step (usually batched together). A 30-step guided sample is **60** network evaluations → [[Diffusion Sampling#^more-steps-isnt-better|the cost formula]]. This is a big reason distilled models are attractive: they typically **bake a fixed guidance scale in**, needing one pass and ignoring the slider → [[Consistency Models#^distillation-tradeoffs|the trade-off]]. ^cfg-doubles-cost

> [!TIP] Settings and fixes
> - **Typical $w$:** 5–9 for SD 1.x/SDXL; 3–5 for flow-matching models (SD3, Flux); ~1–2 for distilled/Turbo models.
> - **Overexposed at high $w$?** Use *CFG rescale* / dynamic thresholding, or apply guidance only during the **middle** of sampling — early steps set layout (guidance matters), late steps polish detail (guidance mostly does damage).
> - **Guidance is a *score-space* operation**, so it works identically for diffusion and [[Flow Matching]] (on velocities), and for any condition — text, class, image, audio.
> - **It composes:** several conditions, each with its own weight. That's how multi-ControlNet and regional prompting work → [[Controlling Diffusion]].

> [!WARNING] High CFG ≠ better model
> Guidance makes a *mediocre* conditional model look good by trading away diversity you may not notice in a handful of samples. When comparing models, compare them **at their own best guidance scale**, and look at a *grid* of seeds, not one image. A model that only looks good at $w = 12$ is leaning on the crutch.

---
> [!SUCCESS] If you remember one thing
> **Predict with and without the prompt; exaggerate the difference.** ~={pink}It's a fidelity-for-diversity dial — indispensable, doubles the cost, and overshoots when pushed.=~

---
# ⁉️
The denoiser can't read English. Something has to turn *"a corgi wearing a top hat"* into vectors that mean the same thing to an image model as they do to you — and that something was trained on 400 million captioned pictures.

→ [[CLIP]]
