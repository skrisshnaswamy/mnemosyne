---
aliases:
  - Saliency Map
  - Attribution
tags:
  - explainability
  - computer-vision
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Which parts of the input actually mattered to this prediction? Answer it with a **gradient** — how much would the output change if this pixel changed?
> **Metaphor:** Highlighting the words in a document that made you reach your conclusion.
> **Where it bites:** Explainable AI, debugging shortcut learning. **Caveat:** raw saliency maps are noisy and easy to over-trust.

---
It is a way to identify the importance of stuff.
It gets used in computer vision to identify what parts of the image are of high importance.
It is also used in Explainable AI, to show what parts of the inputs did the model give more importance / attention to.
We use something called [[Saliency#Saliency Maps|saliency maps]] to visualize the model's thinking process

---
# Saliency Maps

It's a way to visualize the importance. Used in Explainable AI.

The trick is to reuse machinery you already have. During [[Backpropagation|backprop]] you compute $\frac{\partial L}{\partial w}$ — how the loss changes with each **weight**. For saliency you compute:

$$\text{saliency} = \left| \frac{\partial \, \text{score}_c}{\partial \, \text{input}} \right|$$

The gradient with respect to the **input** instead. ~={blue}"If I nudged this pixel slightly, how much would the 'cat' score move?"=~ Big gradient → that pixel mattered. Reshape it back to image dimensions and you have a heatmap. 🔥

> [!SUCCESS] Core idea
> Same [[Pytorch Autograd|autograd]] machinery, one backward pass, different question. Training asks *"how should the **weights** change?"*; saliency asks *"how much did the **input** matter?"* ^saliency-is-input-gradient

---
# Why the raw version isn't good enough

> [!WARNING] Vanilla saliency maps are noisy and misleading
> - **Gradient saturation.** If the model is already confident, the gradient is near **zero** — so a pixel that's overwhelmingly responsible for the prediction can show up as *unimportant*. The gradient measures local sensitivity, not contribution. ⚠️
> - **Visual noise.** Neighbouring pixels get wildly different gradients, producing speckle that the eye happily interprets as structure.
> - **They can be independent of the model.** Some popular methods produce near-identical maps for a *trained* network and a *randomly initialised* one — meaning they're partly showing edge detection, not the model's reasoning. ^saliency-caveats

Hence the improvements, each fixing one of those:

| Method | Fix |
|---|---|
| **SmoothGrad** | Average saliency over many noisy copies of the input — cancels the speckle |
| **Integrated Gradients** | Integrate gradients along a path from a blank baseline to the real input — solves saturation, and satisfies an axiom that attributions should sum to the prediction |
| **Grad-CAM** | Use gradients at the last *conv* layer, not the input — coarser but far more stable and semantically meaningful |
| **SHAP / LIME** | Model-agnostic; perturb inputs and observe. No gradients needed, much slower |

---
# What it's actually good for

The honest use isn't explaining a *correct* prediction — it's catching a **wrong reason** for a right answer.

The canonical example: a classifier that separates huskies from wolves with high accuracy. The saliency map shows it's looking at the **snow in the background**, not the animal. Every wolf photo in the dataset had snow. The model learned a shortcut and would fail the moment a husky appeared on grass. 🐺❄️

> [!TIP] The right framing
> Saliency is a **debugging tool**, not an explanation. It's good at answering *"is the model looking somewhere absurd?"* and bad at answering *"why did the model decide this?"* Treat a clean-looking heatmap as a hypothesis to test, not a finding. ^saliency-is-debugging

Worth noting: **[[Query, Key, and Value (QKV)|attention weights]] are not explanations either**, and this is a well-litigated point. Attention shows where information was *read from*, not what drove the decision — you can often perturb attention substantially without changing the output. "The model attended to this token" is a weaker claim than it sounds.

---
# ⁉️
Saliency asks which inputs mattered. The related question — how much the model *should be trusted* on this input at all — is [[Uncertainty]].
