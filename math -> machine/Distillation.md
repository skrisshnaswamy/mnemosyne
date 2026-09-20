---
aliases:
  - Knowledge Distillation
  - Teacher-Student
tags:
  - training
  - compression
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Train a **small** model to copy a **big** model's *full output distribution* — not just its final answer.
> **Metaphor:** A tutor who explains "it's a dog, but I can see why you'd say wolf — definitely not a car." The near-misses carry the lesson.
> **Where it bites:** Shipping something cheap enough to serve. Also the standard way to make a reasoning model's output affordable.

---
Its a technique to distill knowledge from 1 model to another - typically from a _big_ **teacher** model to a _small_ **student** model; and in rare cases to itself (via techniues like rejection sampling (best of N) - where you sample outcomes from the previous training loop).

---
# The insight — the wrong answers are the lesson

The obvious approach to shrinking a model is: run the big model over lots of data, collect its answers, train the small model on those labels.

That works, but it throws away almost everything valuable.

Show the teacher a photo of a dog. It doesn't just output "dog". It outputs a **distribution**:

```
dog     0.90
wolf    0.08     ← this is the interesting part
cat     0.019
car     0.000001
```

The hard label `dog` tells the student one fact. But that full distribution tells it something much richer: ~={blue}dogs look quite a lot like wolves, somewhat like cats, and nothing whatsoever like cars.=~

That's a similarity structure over the entire label space — a compressed summary of everything the teacher learned about how the world is organised. Hinton called it the **dark knowledge**: the information hiding in the probabilities the model *didn't* pick.

> [!SUCCESS] Core idea
> Hard labels give one bit of signal per example. **Soft targets give a whole distribution per example.** That's why a student trained on teacher distributions beats an identical student trained on ground-truth labels — vastly more information per sample. ^dark-knowledge

---
# Temperature — turning up the quiet signal

There's a problem. That `car: 0.000001` is where the interesting structure lives, and it's numerically invisible. Softmax has squashed it into nothing.

So you re-run the softmax with a **temperature** $T$:

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- $T = 1$ → normal, peaked
- $T = 3$ or $5$ → **softened**, so the small probabilities become large enough to actually carry gradient

You apply the *same* temperature to teacher and student, and match them with [[KL Divergence]].

> [!NOTE] Why the loss has two terms
> $$\mathcal{L} = \alpha \underbrace{T^2 \cdot D_{KL}(\text{teacher}_T \parallel \text{student}_T)}_{\text{imitate the teacher}} + (1-\alpha)\underbrace{\text{CE}(\text{student}, \text{true label})}_{\text{stay correct}}$$
> The second term is the safety net — the teacher is not infallible, and the ground truth keeps the student anchored when the teacher is wrong. The $T^2$ factor is bookkeeping: softmax gradients scale as $1/T^2$, so this keeps the two terms comparable as you change $T$. ^distill-loss

And note it's **forward KL** — mode-covering. You *want* the student to cover everything the teacher considers plausible, not to collapse onto the teacher's single favourite answer. See [[KL Divergence#Forward KL vs Reverse KL|forward vs reverse KL]]. 🎯

---
# The flavours

| Type | What's matched | Notes |
|---|---|---|
| **Response-based** | final output distribution | The classic. Simple, works |
| **Feature-based** | intermediate hidden layers | More signal, but you must align mismatched dimensions |
| **Relation-based** | *relationships between* examples | "these two inputs should be similarly far apart" |
| **Self-distillation** | model teaches **itself** | ↓ below |
| **Sequence-level** | full generated sequences | Standard for LLMs — the student trains on teacher-generated text |

---
# Self-distillation — how a model teaches itself

> [!TIP] Answering the open question in this note
> It looks circular, but it isn't, and the reason is worth understanding.

**Rejection sampling / Best-of-N.** Generate $N$ answers from the *current* model at high temperature. Score them with something **external to the model itself** — a verifier, unit tests, a maths checker, a reward model, or simply "does it match the known answer". Keep only the winners. Fine-tune the model on those winners. Repeat. 🔁

The circularity is broken by the **scorer**, and by an asymmetry that shows up constantly:

> [!SUCCESS] Why it isn't a free lunch
> ~={pink}**Generating a correct answer is hard; recognising one is easy.**=~
>
> A model that solves a problem 30% of the time will, over 20 samples, produce a correct solution almost every time. Filtering keeps only those. Training on them raises the *first-try* rate toward what the model could already achieve *with 20 tries*.
>
> You're not creating knowledge. You're **converting compute into reliability** — moving capability that existed at high sampling cost into the weights, where it's free. ^self-distillation-mechanism

That's the engine behind STaR, ReST, and the rejection-sampling stage in most modern post-training pipelines. And it has a hard limit: if the model can *never* produce a correct answer, there's nothing to filter, and the loop stalls.

Related: **model collapse** — repeatedly training on your own unfiltered output degrades the model, because errors compound and diversity shrinks. The filter is what separates self-improvement from self-poisoning. Related in spirit to [[Mode Collapse]].

---
# Why it matters commercially

Distillation is the main reason capable models are cheap to use. The teacher is trained once, at enormous cost; the student is trained once against it, and then serves millions of requests at a fraction of the cost and latency.

Compare it against the alternatives:
- **Quantization** — same model, fewer bits per weight. Cheap, no retraining. See [[Mixed Precision training]]
- **Pruning** — delete unimportant weights
- **Distillation** — a genuinely *different, smaller* architecture, retrained. The most work, usually the best quality-per-parameter

They compose — production models are often distilled *and* quantized.

> [!WARNING] The catch
> A student can beat models of its own size trained from scratch. It will **not** match its teacher. Distillation transfers a large fraction of the capability, not all of it — and the gap widens on the hardest examples, which are exactly the ones you're likely to care about. 📉

### Papers to refer
- Seminal work on [**Knowledge distillation** by Hinton 2015](https://arxiv.org/pdf/1503.02531) → [[Distilling the Knowledge in a Neural Network]]

---
# ⁉️
Distillation copies a distribution faithfully. The opposite failure — a generative model that collapses onto one narrow slice of what it should produce — is [[Mode Collapse]].
