---
aliases:
  - Generative Model Evaluation
  - FID
  - Fréchet Inception Distance
  - Frechet Inception Distance
  - Inception Score
  - Precision and Recall for Generative Models
  - FVD
  - Human Preference Score
  - GenEval
  - Memorisation in Generative Models
tags:
  - generative-models
  - evaluation
  - metrics
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** There's no ground truth to compare a generated image with — so you compare **distributions**: do generated images, as a *set*, look like real images as a set? **FID** does that with a mean and a covariance, which is why it's the standard and why it's easy to fool.
> **Metaphor:** Judging a forger by their whole portfolio, not one canvas. *Realistic?* **and** *varied?* — either alone is easy.
> **Where it bites:** Reading any generative-modelling paper; comparing checkpoints; noticing that "FID went down" and "it looks better" are different claims.

---
A classifier is easy to grade: it said *cat*, the label says *dog*, it's wrong.

Now a generator produces an image of a cat that has never existed. **Wrong compared with what?** There's no label. There's no target image. There's not even a meaningful likelihood ([[Maximum Likelihood#^likelihood-is-not-quality-images|likelihood isn't quality]]).

And "does it look good?" hides a trap. Imagine a model that outputs the *same* flawless photograph every single time.

~={blue}Every sample is perfect. Is it a good generative model?=~

---
# The forger's portfolio 🖼️

Obviously not — and that's the structure of the whole problem. A generator can fail in **two independent ways**:

| | Question | Failing it looks like |
|---|---|---|
| **Fidelity** (precision) | is each sample *realistic*? | blur, artefacts, six fingers |
| **Diversity** (recall) | do the samples cover *everything* real data covers? | the same face over and over → [[Mode Collapse]] |

You can't see diversity in one image. So evaluation has to look at **sets**.

> [!NOTE] Fréchet Inception Distance (FID)
> Embed N real and N generated images with a pretrained Inception network. Fit a Gaussian (mean $\mu$, covariance $\Sigma$) to each set of features. Report the Fréchet distance between the two Gaussians:
> $$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\big(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\big)$$
> Lower is better; 0 means the two feature distributions have identical mean and covariance. (Heusel et al., 2017.) ^fid-def

> [!SUCCESS] Core idea
> ~={pink}Compare distributions, not images — in a feature space, not pixel space.=~ Pixels don't measure what people see (shift an image one pixel and the pixel error is huge). A pretrained network's features do, roughly. And a distance between *distributions* catches both failure modes at once: blur moves the mean, collapse shrinks the covariance. ^compare-distributions

---
# What FID can and can't see

Here it is on a toy: real data is eight clusters in a ring; five "generators" fail in different ways.

![[fid_failure_modes.png]]
> [!TIP] Reading the chart
> A good model scores **0.00**. Memorising one mode: **15.5** — caught. Losing four *neighbouring* modes: **5.5** — caught. But ~={red}losing **every other** mode scores **0.01**=~ — statistically indistinguishable from perfect. Half the modes are gone, and because they were dropped *symmetrically*, the mean and the covariance didn't move. FID only ever looks at those two things.

> [!WARNING] FID's blind spots
> - **It's two moments.** Anything that preserves the mean and covariance of the features is invisible — including substantial mode dropping, as above.
> - **The ruler is an ImageNet classifier from 2015.** It cares about ImageNet-ish object texture; it's poor on faces, art, text, medical images — anything far from its training set.
> - **It's biased by sample size.** FID at 5k samples and at 50k are *different numbers*. Never compare across papers unless N, the reference set, the resizing code and even the JPEG settings match.
> - **It ignores the prompt entirely.** A beautiful, diverse set of images of the *wrong thing* scores well.
> - **It disagrees with people** at the top end: [[Classifier-Free Guidance|raising guidance]] makes images look better to humans and makes FID *worse* (diversity falls). ^fid-blind-spots

---
# The rest of the toolbox

| Metric | Measures | Note |
|---|---|---|
| **Inception Score** | are samples confidently classifiable *and* spread across classes? | never looks at real data; largely retired |
| **Precision / Recall** (Kynkäänniemi et al.) | fidelity and diversity **separately**, via nearest-neighbour manifolds | tells you *which* of the two failed — what FID can't |
| **CLIP score** | does the image match its prompt? → [[CLIP]] | the standard for text alignment; inherits [[CLIP#^clip-bag-of-concepts\|CLIP's blindness]] to counting and composition |
| **FD-DINOv2 / CMMD** | FID with a modern encoder / without the Gaussian assumption | better agreement with humans |
| **FVD** | FID for video, using a video network | → [[Video Diffusion]] |
| **Compositional benchmarks** (GenEval, T2I-CompBench, DPG) | object counts, colours, positions, relations — checked by detectors | what CLIP score misses |
| **Learned preference models** (HPS, PickScore, ImageReward) | a [[Preference Learning\|reward model]] trained on human "which is better?" votes | good proxy; [[Reward Hacking\|gameable]] if you optimise against it |
| **Human side-by-side / Elo arenas** | what people actually prefer | the ground truth; slow and expensive |
| **Nearest-neighbour / memorisation checks** | is the model *copying* training images? | a legal question as much as a technical one |

---
# How to actually evaluate 🛠️

> [!TIP] A sane protocol
> 1. **Fix everything**: sampler, steps, guidance scale, seeds, N. Report them.
> 2. **Sweep guidance** and plot **FID against CLIP score** — a curve, not a point. One model beats another only if its *curve* is better; any single operating point can be cherry-picked.
> 3. **Report precision and recall** alongside FID, so a gain can be attributed.
> 4. **Look at grids.** Fixed seeds, fixed prompts, every checkpoint, side by side. Never one hero image. Training loss is [[Denoising Objective#^loss-curve-lies|useless here]].
> 5. **Use a compositional benchmark** for prompt following.
> 6. **Finish with humans** on the decision that matters.

> [!WARNING] The field has a Goodhart problem
> FID has been *the* target for years, and models are now tuned to it: architectural choices, guidance settings and even resizing kernels get selected because they move FID by 0.1. Improvements at that scale mostly measure fit to a 2015 classifier's feature statistics. It's [[Reward Hacking#^goodharts-law|Goodhart's law]], and the same disease as benchmark-chasing in [[Evals|LLM evals]]. ~={red}When a measure becomes a target…=~ ^fid-goodhart

---
> [!SUCCESS] If you remember one thing
> **Fidelity *and* diversity, measured on sets, in a feature space.** ~={pink}FID is a useful smoke alarm and a poor judge — it sees only a mean and a covariance, through the eyes of an old classifier, and knows nothing about your prompt.=~

---
# ⁉️
Everything in this chain so far has been about *continuous* data — pixels and latents, where "add a bit of Gaussian noise" means something. Words aren't like that: there's no such thing as "cat, plus a small amount of noise". Can the diffusion idea survive the jump to discrete data?

→ [[Discrete Diffusion]]
