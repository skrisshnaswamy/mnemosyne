---
aliases:
  - Modal Collapse
tags:
  - generative-models
  - failure-mode
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A generative model produces only a **narrow slice** of what it was trained on — high quality, near-zero variety.
> **Metaphor:** A forger who painted one flawless Monet and now paints only that one, forever. Technically excellent. Useless as a body of work.
> **Where it bites:** GANs classically, but also RLHF'd LLMs that answer everything in the same voice. Watch for it whenever a loss rewards *quality* without rewarding *coverage*.

---
This is a common failure reasons typically seen in [[Generative Models]] esp. in [[Generative Adverserial Network]]. Well, even if it's mostly known in GANs, it can also be seen in any generative models.

Typically happens when your model learns to produce only a limited subset of output rather than the rich varied set of output it is trained on.
Like, the model is trained to generate dogs, cats and hamsters, but then it only generate cats.

---
# Why a model would do this

It seems irrational until you look at the incentive. The model isn't broken — it's **winning**.

Take a [[Generative Adverserial Network|GAN]]. The generator's only job is to fool the discriminator. Nobody ever told it to be *diverse*. So it explores, discovers that one particular kind of cat reliably fools the discriminator, and does the entirely rational thing: ~={blue}produces that cat every single time.=~ 🐱

Loss: excellent. Diversity: zero.

> [!SUCCESS] Core idea
> Mode collapse is what you get when your objective rewards **quality** but never rewards **coverage**. The model is optimising exactly what you asked for. You asked for the wrong thing. ^collapse-is-an-incentive-problem

A "mode" here is a peak in the data distribution — a distinct cluster the data actually contains (cats, dogs, hamsters; or the digits 0–9). Collapse means covering some peaks and abandoning others.

---
# The KL connection — it's a predictable consequence

This is the framing that makes mode collapse feel inevitable rather than mysterious.

From [[KL Divergence#Forward KL vs Reverse KL|forward vs reverse KL]]:

| | **Forward** $D_{KL}(P \parallel Q)$ | **Reverse** $D_{KL}(Q \parallel P)$ |
|---|---|---|
| Penalises | missing something real | inventing something fake |
| Behaviour | mode-**covering** (blurry) | mode-**seeking** (collapses) ⚠️ |

Reverse KL charges you nothing for ignoring a mode — you assigned it ~zero probability, so it barely appears in the sum. Covering one peak perfectly and pretending the others don't exist is a **genuinely optimal** strategy under that objective.

> [!TIP] The diagnostic question
> When a generative model collapses, ask: *does my loss actually punish "you never produced this thing that exists in the data"?* If not, expect collapse. It isn't a bug you'll patch — it's the objective being satisfied. 🎯

And the reverse failure has a name too: **mode averaging**, where a mode-covering model smears mass across the empty space *between* peaks and produces blurry nonsense that belongs to no mode at all. Classic VAE output. Sharpness and coverage are in genuine tension.

---
# How you spot it

> [!WARNING] It hides from your loss curve
> A collapsed GAN often has *perfectly healthy-looking* losses. Both networks are training, neither is diverging. You have to look at the **samples**. 👀
>
> - Generate a large batch and just look — do 200 samples show 5 distinct things?
> - **Precision vs recall for generative models** — precision = are samples realistic, recall = do they cover the data. Collapse is **high precision, low recall**
> - **FID** captures it partially (it compares distributions, so missing modes hurt), but it's not sensitive enough on its own
> - Walk a straight line through latent space — healthy models transition smoothly, collapsed ones jump between a few fixed outputs

---
# The fixes

Almost all of them are variations on *"make the objective care about diversity"*:

| Fix | How |
|---|---|
| **Minibatch discrimination** | Let the discriminator see a whole batch at once, so "all your samples look identical" becomes detectable |
| **Unrolled GANs** | Generator optimises against where the discriminator will be in a few steps — so hopping to a new mode stops working |
| **WGAN / gradient penalty** | Replace the divergence with **Wasserstein** distance, which (unlike KL) accounts for *how far apart* the outcomes are and gives useful gradients even where distributions don't overlap |
| **Experience replay** | Show the discriminator old generator samples so it doesn't forget previously abandoned modes |
| **Just use diffusion** | Diffusion models are trained with a likelihood-style, mode-**covering** objective. Sidestepping mode collapse is a large part of why they displaced GANs |

---
# It's not only a GAN problem

Worth carrying, because this is where you'll actually meet it now:

- **RLHF'd LLMs.** Optimising a reward model collapses output diversity — every answer arrives in the same structure, same hedging, same bulleted voice. It's mode collapse in style space, and it's exactly why RLHF objectives carry a [[KL Divergence#1. As a leash — RLHF and PPO 🦮|KL penalty against the base model]]: an explicit leash to stop the policy collapsing onto whatever the reward model likes most.
- **RL policies** that find one adequate strategy and stop exploring — see [[Uncertainty#Exploration vs exploitation|exploration vs exploitation]].
- **Recommenders** that converge on recommending the same popular items to everyone. Same disease, called *popularity bias* or *filter bubbles*. See [[Recommender Systems - Evolution]].
- **Self-training loops** that narrow with each round — the reason [[Distillation#Self-distillation — how a model teaches itself|self-distillation]] needs an external filter.

> [!SUCCESS] If you remember one thing
> Mode collapse is not a training instability. It is a **correctly optimised wrong objective**. Fix the objective, not the optimizer.

---
# ⁉️
The architecture where this failure was first understood — and where the adversarial incentive makes it almost unavoidable — is the [[Generative Adverserial Network]].

Reading the [[Generative Models]] chain? GANs dropped likelihood and paid for it in stability. The next family keeps a principled objective but stops trying to learn the *height* of the probability landscape — only its **slope** → [[Score Function]]
