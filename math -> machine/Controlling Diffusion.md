---
aliases:
  - Controllable Generation
  - ControlNet
  - Inpainting
  - Outpainting
  - Image-to-Image
  - img2img
  - SDEdit
  - DreamBooth
  - Textual Inversion
  - IP-Adapter
  - T2I-Adapter
  - Diffusion LoRA
  - Image Editing with Diffusion
  - DDIM Inversion
tags:
  - generative-models
  - diffusion
  - adaptation
  - engineering
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A toolbox for steering a pretrained diffusion model **beyond the prompt** — start from an existing image (img2img), repaint a region (inpainting), impose structure (ControlNet), teach it a new subject or style (LoRA, DreamBooth), or follow a reference picture (IP-Adapter).
> **Metaphor:** Tracing paper and stencils. The artist still paints — but over your outline, inside your mask, in the style of your reference.
> **Where it bites:** Every real product built on image generation. Nobody ships "type a prompt, take what you get".

---
*"A woman in a red coat, standing on the left, looking over her right shoulder, arm raised, in front of my company's actual storefront, in the style of our brand illustrations."*

Try getting that from a prompt. You'll get a woman, and a red coat. The pose, the side of the frame, the specific building, the house style — words are a terrible way to specify any of them.

You *could* sketch the pose in ten seconds. You *have* a photo of the storefront. You *have* twenty brand illustrations.

~={blue}The model was trained on text. How do you hand it a sketch, a photo and a style — without retraining a multi-billion-parameter network?=~

---
# Tracing paper and stencils ✂️

Every technique here exploits a specific property of diffusion. Worth knowing which.

## 1 · Start part-way — **img2img** (SDEdit)

Sampling normally starts from pure noise. Instead: take **your** image, add noise up to, say, 60% ([[Forward Diffusion Process#^jump-to-any-t|one line]]), and denoise from there.

| Strength | Noise added | What survives |
|---|---|---|
| 0.2 | a little | nearly everything — a light retouch |
| 0.5 | half | composition and colours; details re-imagined |
| 0.8 | most | only the broad layout |

It works because sampling is [[Forward Diffusion Process#The schedule — where the model spends its effort|coarse-to-fine]]: the early, high-noise steps decide layout. Skip them, and the layout stays *yours*.

## 2 · Repaint a region — **inpainting**

At every step: **inside the mask**, keep what the model produces; **outside it**, overwrite with the original image noised to the current level. The model never gets to disagree with the known pixels, so what it paints inside has to be consistent with them. (Dedicated inpainting models add the mask as extra input channels and blend better at the seam → [[Conditional Generation#Three ways to get the condition in|concatenation]].) **Outpainting** is the same thing with the mask outside the frame.

## 3 · Impose structure — **ControlNet**

![[controlnet_conditioning.png]]

You have an edge map, a depth map, a pose skeleton, a segmentation map — *spatial* intent.

ControlNet **freezes** the pretrained model, makes a **trainable copy of its [[U-Net]] encoder**, feeds your control image into the copy, and adds the copy's outputs into the frozen decoder through its skip connections — via **zero-initialised** convolutions.

> [!TIP] Why zero-init is the clever bit
> At step zero of training, the added branch contributes *exactly nothing* — the system **is** the original model, still working perfectly. Control "fades in" from there, so you never damage what the base model knows, and it trains on a few thousand pairs on one GPU. Same trick as [[Diffusion Transformer#^adaln-zero|adaLN-Zero]], and as [[LoRA]] initialising one of its matrices to zero.

## 4 · Teach it something new — **personalisation**

| Method | What's trained | Size | Good for |
|---|---|---|---|
| **Textual inversion** | one new *word embedding* | ~KB | a concept the model can nearly draw already |
| **DreamBooth** | the **whole model**, on 3–5 images of a subject + a "prior preservation" loss so it doesn't forget what a generic dog looks like | GB | a specific person / pet / product, with high fidelity |
| **[[LoRA]]** | low-rank adapters on the attention weights | **~10–200 MB** | styles, characters, concepts — the community standard. Stackable and swappable |
| **IP-Adapter** | a small extra cross-attention path that takes **image** embeddings ([[CLIP]]) | ~100 MB | *"like this reference picture"* — no per-subject training at all |

## 5 · Edit a real photo — **inversion**

To edit a real image with prompts, first find the noise that *would have generated it*. Because the deterministic sampler is an invertible ODE ([[Diffusion Sampling#Two families of sampler|DDIM]] / [[Score SDE|probability flow]]), you can run it **backwards**: image → noise. Then sample forwards again with an edited prompt — keeping the attention maps of unchanged words fixed (*prompt-to-prompt*) so everything else stays put.

---
> [!SUCCESS] Core idea
> ~={pink}A pretrained diffusion model is a *prior over images*; every control method adds a constraint and lets the prior fill in the rest.=~ They're cheap because they never touch the hard part — the base model already knows what the world looks like. You're only telling it *which* plausible image you want. That's the [[Foundation Models#The adaptation ladder 🪜|foundation-model adaptation ladder]], in pictures. ^prior-plus-constraint

| You want to… | Reach for |
|---|---|
| vary an existing image | img2img, low strength |
| change one region | inpainting |
| fix pose / layout / edges / depth | ControlNet (or T2I-Adapter — lighter) |
| a consistent character or product | LoRA / DreamBooth |
| "in the style of *these*" | style LoRA, or IP-Adapter |
| edit a real photo by prompt | inversion + prompt-to-prompt, or an instruction-tuned editor (InstructPix2Pix) |
| all of the above at once | they **compose** — each is one more term in the score → [[Classifier-Free Guidance]] |

> [!WARNING] What goes wrong
> - **Controls fight.** A pose that contradicts the prompt, two ControlNets that disagree — you get mush, or one wins. Weights need tuning per combination.
> - **Personalisation overfits.** Five photos, too many steps → the model can *only* draw that dog, in that pose, on that sofa. It's [[Fine-Tuning]]'s usual failure, with a tiny dataset.
> - **Adapters are tied to their base model.** An SD 1.5 LoRA does nothing useful on SDXL or Flux.
> - **Distilled fast models** often respond poorly to controls built for the original → [[Consistency Models#^distillation-tradeoffs|distillation trade-offs]].
> - **Identity and consent.** A handful of photos is enough to generate a real person convincingly. The technical barrier is gone; the guardrail has to be policy → [[Guardrails]]. ^control-failure-modes

---
> [!SUCCESS] If you remember one thing
> **Prompts say *what*; these tools say *where*, *which one*, and *like this*.** ~={pink}Each works by constraining a frozen model from the outside — which is why they're small, cheap, stackable, and tied to the base model they were built for.=~

---
# ⁉️
You can now generate, steer and edit. One uncomfortable question has been dodged for the whole chain: **how do you know whether any of it is good?** There's no "right answer" to compare a generated image with.

→ [[Evaluating Generative Models]]
