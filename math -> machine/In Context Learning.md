---
aliases:
  - ICL
  - few shot learning
  - few-shot
excalidraw-plugin: parsed
tags:
  - llm
  - foundation-models
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** The model learns a new task from examples **in the prompt**, with **zero weight updates**.
> **Metaphor:** A session musician handed a few bars of sheet music. They didn't practise for weeks — they read the pattern and played along.
> **Where it bites:** It's why prompting works at all, and the reason "just fine-tune it" is usually the wrong first move.

---
This is a new concept from the LLMs. It's where the model essentially **learns** without it's traditional sense of learning.
Well, a typical ML model **learns** by backpropagating and updating it's weights and parameters. And this happens during training (and also during fine tuning) - but not during inference.
But ICL is different, it is learning without updating the weights and biases.

"few-shot", "one-shot", "zero-shot" are just different variants of it - indicating how many examples we provide.

The idea is, it recognises patterns and then captures them and tries to find the same patterns in the new query.

---
# Why this is genuinely strange

It's worth pausing on how odd this is, because familiarity hides it.

Every model you've built before had a hard boundary: **train time** (weights change) and **inference time** (weights frozen, model applies what it knows). Learning happened on one side of that line, only.

ICL puts learning on the **wrong side of the line**.

```
Translate English to French:
sea otter -> loutre de mer
peppermint -> menthe poivrée
cheese ->              ← it produces "fromage"
```

Not remarkable for translation — it knows French. But the same thing works for a task **invented five seconds ago**:

```
apple -> elppa
train -> niart
cheese ->              ← "eseehc"
```

Nobody trained it to reverse strings. It inferred the rule from two examples, held it in its context, applied it. And the instant the conversation ends, that ability is **gone** — nothing persisted, because nothing changed. ^icl-is-ephemeral

> [!SUCCESS] Core idea
> Weight updates are the *slow* learning: absorbing the structure of language over trillions of tokens. ICL is the *fast* learning that the slow learning made possible. ~={blue}The model didn't learn the task — it learned how to pick up tasks.=~ ^two-timescales

---
# The shots

| | What you give | When |
|---|---|---|
| **Zero-shot** | just the instruction | The task is common; instruction-tuned models are strong here |
| **One-shot** | instruction + 1 example | Mainly to pin down the output **format** |
| **Few-shot** | instruction + several | Unusual tasks, strict formats, subtle label boundaries |

> [!TIP] What examples are really doing
> A well-known finding: replacing the labels in your few-shot examples with **random** ones often barely hurts performance.
>
> Which suggests the examples are less about teaching the *mapping* and more about specifying the **task, the format, and the label space** — "here's the shape of question I'm asking, here are the allowed answers, here's how to lay out your reply." That reframes prompt design: your examples are mostly a **specification**, not a training set. ^random-labels-finding

---
# What's actually happening in there

Nobody has a complete answer, but the two leading accounts are both worth being able to state:

**1. Task location.** Pretraining saw enormous numbers of implicit tasks. The model has effectively learned thousands of them and mixed them together; the prompt doesn't teach a new skill, it **locates** an existing one. The examples are an address, not a lesson. This explains why genuinely alien tasks work far less well than familiar-shaped ones.

**2. Implicit gradient descent.** More surprising, and there's real evidence for it: a transformer's forward pass over in-context examples can *simulate* a learning algorithm. Attention layers can be constructed to perform something equivalent to a gradient step on a small internal model. On this view ICL is a real optimisation loop — just one running inside the activations of a single forward pass rather than in the weights. 🤯

Both are probably partly right. It's an open research area, and saying so is the honest answer.

> [!NOTE] Emergence
> ICL doesn't fade in smoothly with scale — small models are essentially incapable of it, and beyond some size it works well. This is the headline example of an "emergent" ability, with the caveat from [[Foundation Models#^emergence-caveat|Foundation Models]]: some of that sharpness is an artefact of pass/fail metrics on a smoothly improving skill.

---
# The practical limits

> [!WARNING] Where ICL runs out
> - **Context is finite and costs money.** Every example is re-sent and re-processed on every single call. 50 examples in a prompt is fine; 5,000 is not — that's when fine-tuning starts to win on unit economics.
> - **Order matters, sometimes a lot.** The same examples in a different order can change accuracy noticeably. Recency bias is real.
> - **It shifts behaviour, not knowledge.** ICL can't teach facts the model never saw. If the model doesn't know your internal product codes, examples won't install them — that's a **retrieval** problem. See the [[Foundation Models#The adaptation ladder 🪜|adaptation ladder]].
> - **It's not persistent.** Nothing carries over between calls. ^icl-limits

**Chain-of-thought** is the close cousin: rather than showing input→output pairs, you show input→*reasoning*→output, and the model imitates the reasoning. It works because the intermediate tokens give the model somewhere to do computation it cannot do in a single forward pass — more steps, more compute, and a chance to correct itself mid-answer.

---
# ⁉️
ICL is a capability you get for free from a big model. But big models are expensive to serve. Can we take what a large model knows and compress it into a small one?

→ [[Distillation]]
