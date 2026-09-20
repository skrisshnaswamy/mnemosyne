---
aliases:
  - Foundation Model
  - Pretrained Model
  - Base Model
tags:
  - fundamentals
  - llm
  - paradigm
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Train **one** enormous model **once** on everything, then adapt it cheaply to hundreds of tasks — instead of training a fresh model per task.
> **Metaphor:** Hiring a university graduate and giving them a week of onboarding, vs. raising a specialist from birth for one job.
> **Where it bites:** It's why "should we fine-tune or just prompt?" is now the default first question in any ML design conversation.

---
# The world before

For most of ML's history, the recipe was fixed. New problem → new model.

Want sentiment classification? Collect 50,000 labelled reviews, design an architecture, train from random weights. Want named-entity recognition? **Start over.** Collect a different labelled dataset, train a different model from scratch. Want it in German? Start over again.

Every task paid the full price. And the bottleneck was always the same thing: ~={red}labelled data.=~ Labels need humans. Humans are slow and expensive. So every task was capped by how much labelling budget you had.

Meanwhile there was an ocean of **unlabelled** text and images sitting right there, completely unusable, because supervised learning had no way to consume it.

---
# The unlock: get the labels for free

The trick that changed everything is almost embarrassingly simple.

**Hide part of the data and make the model predict it.**

The text already contains the answer. So you don't need a human:
- Cover the next word and predict it → that's [[Auto-regressive models|autoregressive]] pretraining ([[Language Models are Few-Shot Learners (GPT-3)|GPT]])
- Blank out random words in the middle and fill them in → [[BERT- Pre-training of Deep Bidirectional Transformers|BERT]]
- Show two crops of the same image and make their representations match → [[A Simple Framework for Contrastive Learning (SimCLR)|SimCLR]]

> [!NOTE] Self-supervised learning
> Creating the supervision signal **from the data's own structure**, rather than from human annotation. It turns "unlabelled data" into an effectively unlimited labelled dataset. ^self-supervised

And now look what happened to the bottleneck. It's gone. Your training set is *the internet*. The constraint moved from **labels** to **compute** — and compute, unlike human annotators, you can just buy more of.

> [!SUCCESS] Core idea
> Foundation models exist because self-supervision removed the label bottleneck. Everything else — the scale, the emergence, the whole industry — is downstream of that one move. ^why-foundation-models-exist

---
# Why does predicting the next word teach you anything?

This is the question worth being able to answer, because it sounds like a party trick and it isn't.

To predict the next word *well*, across all of human text, you are forced to learn:

> "The trophy didn't fit in the suitcase because **it** was too ___"

To fill that in, you need to know what "it" refers to. Which requires knowing that trophies are rigid and suitcases have capacity. That's not grammar — that's a **model of objects in the physical world**, learned as a side effect of a spelling exercise.

Push that across trillions of tokens and the pressure to compress everything into finite weights forces the model to discover syntax, facts, arithmetic, translation, code structure, and a rough model of how people reason. ~={blue}Not because you asked for any of them, but because each one lowers the prediction loss.=~

> [!TIP] The compression framing
> Next-token prediction is **lossy compression of the corpus**, and the best compression of text *is* understanding it. Memorising a physics textbook takes far more bits than learning the physics. So the training objective quietly rewards understanding over memorisation.

---
# Scale is not a detail — it's the mechanism

The other half of the story is that **bigger genuinely works**, and predictably so.

[[Scaling Laws for Neural Language Models|Kaplan et al.]] found loss falls as a smooth **power law** in model size, data, and compute — straight lines on a log-log plot, over many orders of magnitude. That's remarkable: it means you can *forecast* how good a model you haven't trained yet will be, and justify the spend before spending it.

[[Training Compute-Optimal Large Language Models (Chinchilla)|Chinchilla]] then corrected the recipe. Everyone had been building models that were too **big** and too **undertrained**. For a fixed compute budget, you should scale parameters and tokens *together* — roughly **20 tokens per parameter**. A smaller model fed more data beats a bigger model fed less, at equal cost.

> [!WARNING] The thing people get wrong about scaling laws
> They describe **training loss**, not capability. Loss curves are boringly smooth. But *downstream abilities* often appear to arrive suddenly — a model can't do multi-digit arithmetic at all, then at some scale it can. Whether that's true "emergence" or an artefact of using pass/fail metrics on a smoothly improving underlying ability is genuinely contested. Have the caveat ready; it's a common gotcha in discussion. ^emergence-caveat

The most consequential emergent behaviour is [[In Context Learning|in-context learning]] — the model learning a *new task* from examples in the prompt, with **no weight updates at all**. That's the capability that turned a text predictor into a general-purpose interface.

---
# The adaptation ladder 🪜

Once you have a base model, "adapting it" is a spectrum, not a binary. This is the practical decision you'll actually argue about in meetings — cheapest first:

| Rung | What it is | Cost | Use when |
|---|---|---|---|
| **Prompting** | Just ask well | ~free | Almost always start here |
| **Few-shot / [[In Context Learning\|ICL]]** | Put examples in the prompt | Token cost per call | You have a handful of examples |
| **RAG** | Retrieve relevant docs, stuff them in the context | Retrieval infra | Knowledge is private, large, or changes often |
| **Fine-tuning (full)** | Update all the weights | 💸💸💸 + serving a whole new model | Rarely worth it now |
| **[[LoRA- Low-Rank Adaptation of Large Language Models\|LoRA]] / PEFT** | Train a tiny low-rank adapter, freeze the base | ~1% of the params | You need behaviour/format change, not new facts |
| **[[Distillation]]** | Train a small model to copy a big one | Big upfront, cheap forever | Latency or unit-cost is the binding constraint |

> [!TIP] The rule of thumb worth having memorised
> **Fine-tuning teaches a model a new *behaviour*. Retrieval gives it new *knowledge*.**
> Most people reach for fine-tuning when their actual problem is that the model doesn't know a fact — and fine-tuning is a terrible, expensive, unreliable way to insert facts. ^finetune-vs-rag

There's also the **alignment** step that turns a raw base model into something usable — [[Training language models to follow instructions with human feedback|instruction tuning and RLHF]]. Worth keeping distinct in your head: a *base* model completes text; an *instruct* model answers you. They are meaningfully different objects with different failure modes.

---
# What actually changed, structurally

1. **The economics inverted.** Pretraining is a colossal fixed cost paid by a handful of labs; adaptation is a marginal cost paid by everyone else. ML shifted from a research activity to an infrastructure one — which is why [[ML Infrastructure]] and [[GPU processing]] became core ML skills rather than ops trivia.
2. **The interface became language.** You describe the task instead of engineering features for it. That's why "prompt engineering" briefly became a job title.
3. **One model, many modalities.** Once [[An Image is Worth 16x16 Words (ViT)|ViT]] showed images could be tokens, the same [[Attention Is All You Need|transformer]] recipe absorbed vision, audio and video. Convergence onto one architecture is a big part of why progress got so fast.

> [!WARNING] Homogenisation is the real risk
> The term "foundation model" was coined partly *as a warning*. If everyone builds on the same handful of base models, everyone inherits the same blind spots, the same biases, and the same failure modes — simultaneously, across every downstream application. A defect in the foundation is a defect in every building on it. 🏗️

---
> [!SUCCESS] If you remember one thing
> A foundation model is **general capability bought once with compute, and rented out cheaply to every task afterwards.** Self-supervision made it affordable; scaling laws made it predictable.

---
# ⁉️
The strangest thing on that list is a model learning a brand-new task from three examples in the prompt, with its weights completely frozen. That shouldn't work. Nothing is being trained.

So what *is* going on? → [[In Context Learning]]
