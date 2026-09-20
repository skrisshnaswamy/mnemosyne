---
aliases:
  - Contrastive Language-Image Pre-training
  - Contrastive Language–Image Pretraining
  - CLIP Model
  - CLIP Score
  - CLIP Embeddings
  - SigLIP
  - OpenCLIP
  - Zero-Shot Classification
  - InfoNCE
tags:
  - generative-models
  - multimodal
  - retrieval
  - deep-learning
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Two encoders — one for images, one for text — trained on 400M (image, caption) pairs so that **matching pairs land close together** in one shared vector space and mismatched ones land far apart.
> **Metaphor:** A shared map for pictures and sentences. A photo of a dog and the words "a photo of a dog" are pinned to the same spot.
> **Where it bites:** The text encoder of Stable Diffusion, zero-shot classification, image search, dataset filtering (LAION), and the *CLIP score* used to evaluate text-to-image models.

---
You want to classify images of 1,000 kinds of object. The classic recipe: collect a labelled dataset, train a network with a 1,000-way output.

Tomorrow someone asks for a 1,001st class. **You retrain.**

Meanwhile the internet contains billions of images that already come with a natural-language description — alt text, captions, titles. Noisy, free, and about *everything*.

~={blue}Could you learn from the captions directly — and end up with a classifier for categories nobody ever labelled?=~

---
# One map for words and pictures 🗺️

![[clip_contrastive_matrix.png]]

Take a batch of $N$ image–caption pairs. Encode every image to a vector; encode every caption to a vector. Compute all $N \times N$ similarities (dot products of normalised vectors).

The **diagonal** of that grid holds the true pairs. Everything off the diagonal is a mismatch. Train both encoders so the diagonal is large and the rest small — a [[Cross Entropy|cross-entropy]] across each row *and* each column:

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j} \exp(\text{sim}(I_i, T_j)/\tau)} \qquad \text{(+ the same with images and texts swapped)}$$

With a batch of 32,768, every image has one right caption and 32,767 wrong ones to be distinguished from. That's the **InfoNCE** loss → [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]], and the same contrastive recipe as [[A Simple Framework for Contrastive Learning (SimCLR)|SimCLR]] — except the two "views" are different *modalities*.

> [!NOTE] CLIP
> **C**ontrastive **L**anguage–**I**mage **P**re-training (Radford et al., OpenAI, 2021): an image encoder (ViT or ResNet) and a text encoder (transformer) trained jointly with a symmetric contrastive loss on ~400M web image–text pairs, producing **aligned** [[Embeddings|embeddings]]. ^clip-def

> [!SUCCESS] Core idea
> ~={pink}Natural language as the label space.=~ Instead of 1,000 fixed classes, the "label" is any sentence. That's why it generalises to categories it never saw named — and why its text tower became the way to tell an image model what you want. It's also an [[Energy-Based Models#You already use energy-based models 🔗|energy-based model]]: similarity is a negative energy, and the rest of the batch stands in for the intractable "everything else". ^language-as-labels

---
# What falls out for free

**Zero-shot classification.** Write one prompt per class — *"a photo of a dog"*, *"a photo of a cat"* — embed them, embed the image, pick the nearest. No training. CLIP matched a fully supervised ResNet-50 on ImageNet **without using a single ImageNet label** — and, more importantly, held up far better under distribution shift.

**Cross-modal search.** Text → image, image → text, image → image — all nearest-neighbour lookups in one space → [[Vector Database]].

**A judge.** How well does this generated image match its prompt? Cosine similarity between the two embeddings: the **CLIP score** → [[Evaluating Generative Models]].

**A filter.** LAION-5B — the dataset behind Stable Diffusion — was built by keeping web pairs whose CLIP similarity exceeded a threshold.

---
# Its role in image generation

| Where | What CLIP does |
|---|---|
| **Stable Diffusion 1.x / SDXL** | the frozen CLIP **text encoder** turns the prompt into 77 token vectors; the denoiser cross-attends to them → [[Conditional Generation]] |
| **DALL·E 2 ("unCLIP")** | text → CLIP *image* embedding (a "prior") → a diffusion decoder that paints an image with that embedding |
| **SD3, Flux** | two CLIP text encoders **plus** T5 — CLIP for visual concepts, T5 for language |
| **Classifier guidance, 2021-style** | push the image toward higher CLIP similarity with the prompt at every step (CLIP-guided diffusion — the pre-Stable-Diffusion art scene) |
| **Evaluation** | CLIP score |

---
# What it can't do — and why

> [!WARNING] CLIP reads prompts like a bag of concepts
> The contrastive objective only has to tell the right caption from the *other captions in the batch*. For that, **"which objects and styles are present"** is nearly always enough; *who is doing what to whom* rarely matters. So CLIP is weak at:
> - **composition and binding** — "a red cube on a blue sphere" ≈ "a blue cube on a red sphere"
> - **counting** — "three dogs" ≈ "dogs"
> - **negation** — "a street with **no** cars" pulls *toward* cars
> - **spatial relations**, and **text rendering**
> - anything past **77 tokens** — silently truncated
>
> When a Stable Diffusion image swaps attributes between two objects, that's usually *this*, not the diffusion model → [[Conditional Generation#^text-encoder-is-the-bottleneck|the text encoder is the bottleneck]]. ^clip-bag-of-concepts

Also: it inherits the web's biases wholesale; it's vulnerable to *typographic attacks* (an apple with a paper label saying "iPod" gets classified as an iPod); and "zero-shot" performance depends heavily on prompt wording — hence prompt ensembling.

> [!TIP] Descendants worth knowing
> **SigLIP** swaps the softmax for an independent sigmoid per pair — no need for enormous batches, and currently the usual choice of vision encoder for vision-language models. **OpenCLIP** is the open reproduction trained on LAION. And the vision tower of most multimodal LLMs is a CLIP-family encoder bolted onto a language model.

> [!TIP] Contrastive vs generative — two ways to learn from images
> CLIP learns by *discriminating* (which caption goes with this?). Diffusion learns by *generating* (what pixels go here?). A third camp says neither: predict in **representation space** → [[JEPA]]. They're complementary: nearly every text-to-image system has a contrastive model steering a generative one.

---
> [!SUCCESS] If you remember one thing
> **Matching images and captions land in the same spot of one shared space.** ~={pink}That gives you open-vocabulary recognition, cross-modal search, and a way to tell an image model what you want — but it understands *what's in the picture* far better than *how it's arranged*.=~

---
# ⁉️
You now have a prompt encoded into vectors. The remaining obstacle is brutally practical: a 512×512 image is 786,432 numbers, the denoiser has to process all of them, dozens of times, per image. In 2021 that meant a rack of GPUs. In 2022 it ran on a gaming laptop. What changed?

→ [[Latent Diffusion]]
