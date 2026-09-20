---
aliases:
  - Conditioning
  - Conditional Generative Models
  - Conditional Diffusion
  - Class-Conditional Generation
  - Text-to-Image
  - Text to Image
  - Cross-Attention Conditioning
  - AdaLN
  - Adaptive Layer Norm
  - FiLM
  - Timestep Embedding
tags:
  - generative-models
  - diffusion
  - architecture
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Model $p(x \mid c)$ instead of $p(x)$ — hand the network **extra information $c$** (a class, a prompt, another image) as an additional input. The interesting part is *how* it's injected: concatenation, modulation, or cross-attention.
> **Metaphor:** Commissioning an artist. The brief doesn't paint the picture — it steers every brushstroke.
> **Where it bites:** Text-to-image, inpainting, super-resolution, image-to-image, ControlNet. Pick the wrong injection mechanism and the model simply ignores your condition.

---
An unconditional [[Diffusion Models|diffusion model]] trained on ImageNet produces… *an ImageNet image*. A dog, a volcano, a teapot — whatever the dice say.

That's $p(x)$: the distribution of all images. What you want is $p(x \mid \text{"a corgi in a top hat"})$ — a much narrower one.

The training recipe barely changes: show the network the noisy image **and the caption that went with it**, and ask for the noise. Nothing else. No new loss.

~={blue}Why would that work? Why would the network bother to *use* the caption?=~

---
# The brief 🎨

Because the caption makes the job **easier**, and gradient descent uses whatever lowers the loss.

Picture a badly noised image — a vague orange-brown smudge. Denoising it blind, the best you can do is hedge across everything it *might* be. But told *"corgi"*, you know where the ears should go. The caption is information about the answer, so the network learns to lean on it.

$$\mathcal{L} = \mathbb{E}\,\big\|\,\varepsilon - \varepsilon_\theta(x_t,\, t,\, c)\,\big\|^2$$

> [!NOTE] Conditional generation
> Learning $p(x \mid c)$ by providing the condition $c$ as an additional input to the generator or denoiser. For diffusion the objective is unchanged — the network simply becomes $\varepsilon_\theta(x_t, t, c)$ — and it learns the **conditional score** $\nabla_x \log p_t(x \mid c)$. ^conditional-def

> [!SUCCESS] Core idea
> ~={pink}Conditioning is just another input — the model uses it because it helps.=~ Which also tells you when it *won't*: if the condition is only weakly informative, or arrives through a pathway that's easy to ignore, the network happily ignores it. The two things that fix that are a **good injection mechanism** (below) and **guidance** at sampling time → [[Classifier-Free Guidance]]. ^conditioning-is-an-input

Note that the network already had a condition before you added one: **the timestep $t$**. It must know the noise level, because the same patch of pixels means something different at $t = 900$ than at $t = 50$. Everything below applies to $t$ too.

---
# Three ways to get the condition in

| Mechanism | How | Best for | Used by |
|---|---|---|---|
| **Concatenation** | stack $c$ onto the input as extra channels | conditions that are **spatially aligned** with the output — a low-res image, a mask, a depth map, an edge map | super-resolution, inpainting, [[Controlling Diffusion\|ControlNet]]-style inputs |
| **Modulation** (FiLM, **AdaLN**) | a small MLP turns $c$ into a per-channel *scale and shift* applied inside every normalisation layer | **global**, low-dimensional conditions — timestep, class label, a pooled text vector, style | every diffusion model (for $t$); [[Diffusion Transformer\|DiT]]'s main mechanism |
| **Cross-attention** | image features form the **queries**; the condition's tokens form the **keys and values** → [[Query, Key, and Value (QKV)#Self-attention vs cross-attention\|cross-attention]] | **sequences** — a text prompt, where different words matter for different regions | Stable Diffusion 1–2, SDXL, Imagen |
| **Joint attention** (MM-DiT) | put text tokens and image tokens in **one** sequence and let everything attend to everything | text — and letting the text representation be *updated* by the image | SD3, Flux |

> [!TIP] Why cross-attention fits text so well
> *"A red cube on top of a blue sphere."* The pixels where the cube is should listen to **"red"** and **"cube"**; the pixels below should listen to **"blue"** and **"sphere"**. Cross-attention gives every spatial position its own weighting over the words. You can even visualise it — the attention map for the token "cube" lights up where the cube is — and *editing* those maps is how prompt-to-prompt image editing works.

---
# Where does $c$ come from?

The denoiser can't read. A prompt has to become vectors first, and that's done by a **separate, usually frozen, pretrained text encoder**:

| Encoder | Trained by | Character |
|---|---|---|
| **[[CLIP]]** text tower | matching captions to images | knows what things *look like*; weak at syntax, counting, long prompts |
| **T5** (an LLM encoder) | language modelling on text alone | understands *language* — composition, negation, spelling. Imagen's finding: scaling the text encoder helps more than scaling the image model |
| **Both** | — | SD3 and Flux concatenate two CLIPs **and** a T5 |

> [!WARNING] A lot of "the model ignored my prompt" is the text encoder
> Wrong count of objects, attributes swapped between objects, garbled text: often the image model never *received* that information in a usable form. CLIP's text tower truncates at 77 tokens and largely treats a prompt as a bag of concepts. ~={red}The image generator is only as literate as its text encoder.=~ ^text-encoder-is-the-bottleneck

---
# The spectrum of conditions

| Condition $c$ | Task |
|---|---|
| nothing | unconditional generation |
| a class label | class-conditional ImageNet — the benchmark setting |
| a text prompt | **text-to-image** |
| a low-res image | super-resolution (cascaded diffusion) |
| an image + a mask | inpainting / outpainting |
| an edge map / pose / depth map | structure control → [[Controlling Diffusion]] |
| a reference image | style transfer, subject-driven generation |
| previous frames | video prediction → [[Video Diffusion]] |
| a robot's camera view | action generation → [[Diffusion Policy]] |

Same objective, same sampler, every row. That uniformity is a large part of why diffusion displaced purpose-built architectures for each task.

> [!TIP] The same split as everywhere else
> "A frozen pretrained encoder produces a representation; a task model consumes it" is the [[Foundation Models]] pattern — and the division of labour in [[RAG]]: a good [[Embeddings|embedding]] upstream sets the ceiling for everything downstream.

---
> [!SUCCESS] If you remember one thing
> **Conditioning = an extra input, injected by concatenation, modulation or attention.** ~={pink}The model uses it because it makes denoising easier — and if it's easy to ignore, it will be, which is why guidance exists.=~

---
# ⁉️
Trained like this, a text-to-image model follows the prompt… *loosely*. The images are plausible and only vaguely on-brief. The thing that turned "vaguely related" into "exactly what I asked for" is a one-line trick at sampling time — and it's probably the single most important practical idea in the whole area.

→ [[Classifier-Free Guidance]]
