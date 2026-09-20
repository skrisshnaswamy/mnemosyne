---
aliases:
  - Autoregressive
  - AR models
tags:
  - fundamentals
  - llm
  - generation
---

> [!ABSTRACT] 🧠 Recall
> **In one line:** Generate **one token at a time**, feeding each output back in as the next input.
> **Metaphor:** Writing a sentence word by word, where you can re-read everything you've written but can't see what's coming.
> **Where it bites:** Why inference is sequential and latency-bound, why errors compound, and why sampling temperature exists.

---
So the term ~={pink}**auto-regressive**=~ simply means that the output of a model is sequential (one token at a time). The way I see it, it is technically a property of the outcome or rather the use case. But then again, in order to generate the outcome sequentially, the models need to be architected in a way that it can support this. Such models are typically called **Auto regressive models**.

Model architectures based on the transformer arch. **and** which has a decoder network typically are capable of **auto regression** output generation. But that said, there are some subtleties.
Decoder-only models are by design auto-regressive - They are typically designed for solving use cases like next token prediction / generation (Sentence completion task). And they do this by a technique called [[Masking]]. You see they **mask** (hide) the future tokens during training to mimic that when the model is training, it actually cannot see what comes next, but only see the past tokens. This is also an [[Attention]] mechanism and is called the [[Causal Attention]].

---
# Where the name comes from

Break the word apart — it's descriptive, not jargon.

**Auto** = self. **Regressive** = regressing on (predicting from). So: ~={blue}**regressing on yourself.**=~ Predicting the next value of a sequence from the previous values *of that same sequence*.

It's an old idea from time series — an AR($p$) model predicts tomorrow's value from the last $p$ days. See [[Auto-regressive lags]]. An LLM is the same shape, with tokens instead of days and a transformer instead of a linear equation.

Formally, it's the chain rule of probability. The probability of a whole sentence factorises into a product of one-token-at-a-time conditionals:

$$P(x_1, x_2, \ldots, x_n) = \prod_{t=1}^{n} P(x_t \mid x_{<t})$$

> [!SUCCESS] Core idea
> This is why the training objective works. Modelling a whole sentence's joint probability is hopeless — there are more possible sentences than atoms. But the factorisation says it's **exactly equal** to a product of next-token predictions, and *those* are just a classification problem over the vocabulary, trained with [[Cross Entropy]]. ^ar-factorisation

---
# The asymmetry that shapes everything

| | **Training** | **Inference** |
|---|---|---|
| The full sequence | already known | doesn't exist yet |
| Processing | all positions **in parallel** | strictly **one at a time** |
| Bottleneck | compute (throughput) | memory bandwidth (latency) |
| Cost of 1000 tokens | one forward pass | **1000** forward passes |

[[Causal Attention|Masking]] is what allows the left column. Every position is trained simultaneously, each with a correctly restricted view.

But at inference the future genuinely doesn't exist, so you must go one step at a time. And each step re-reads the entire [[Query, Key, and Value (QKV)#The KV cache — where this becomes an infra problem|KV cache]] to produce a single token — which means you're moving a lot of memory to do very little arithmetic.

> [!WARNING] Why generation feels slow even on a fast GPU
> Token generation is **memory-bandwidth-bound**, not compute-bound. The GPU is mostly idle, waiting on weights and cache to arrive. This is why batching helps so much (amortise the same memory reads over many sequences), and why speculative decoding exists (have a small model guess several tokens, then verify them all in one pass of the big model). See [[GPU processing]]. ^ar-is-bandwidth-bound

---
# Exposure bias — the flaw worth knowing

During training the model always sees **ground-truth** history (this is called *teacher forcing*). At inference it sees **its own** previous outputs — which may contain mistakes it has never been trained to recover from.

So a small early error puts the model in a state slightly off the training distribution, which makes the next error a bit more likely, and so on. That's **exposure bias**, and it's the mechanism behind degenerate repetition loops and answers that drift off after a strong start. 🌀

> [!TIP] The practical consequences
> - Errors **compound** — quality degrades with length, they don't stay locally wrong
> - There's no going back. The model cannot revise token 5 after seeing token 90. (This is a real argument for **diffusion** language models, which refine the whole sequence.)
> - It's part of why chain-of-thought works: giving the model room to write intermediate steps means a wrong turn can be *corrected in later text* rather than being locked in.

---
# Decoding — the choice at every step

The model outputs a distribution over the vocabulary. Turning that into one token is a separate decision, and it matters more than people expect:

| Strategy | What it does | Failure mode |
|---|---|---|
| **Greedy** | always the highest-probability token | repetitive, flat, loops |
| **Beam search** | keep $k$ candidate sequences, pick the best overall | good for translation, bland for open text |
| **Temperature** | sharpen ($<1$) or flatten ($>1$) the distribution before sampling | too high → incoherent; too low → greedy |
| **Top-k / Top-p** | sample only from the most likely $k$ tokens, or the smallest set summing to $p$ | the practical default |

> [!NOTE] Why the most likely sequence is not the best sequence
> Greedy decoding maximises probability at each step, and beam search approximately maximises it overall — yet both produce noticeably **worse** open-ended text than sampling.
>
> Because real human text isn't maximally probable. It's *surprising* in places. Always picking the safest word produces something that reads correct and dead. This is why we deliberately inject randomness into a model we spent millions training to be accurate. 🎲 ^likelihood-is-not-quality

---
# Not just language

The pattern generalises to anything you can put in a sequence:
- **PixelRNN/PixelCNN** — images, pixel by pixel (slow, but was state of the art before diffusion)
- **WaveNet** — raw audio, sample by sample
- **Generative recommenders** — the next *item* as a token; see [[Recommender Systems - Evolution#Part 5 — Generative recommendation: where everything converges|Part 5]]
- Classic **AR($p$)** time series — the original meaning

The contrast is with **non-autoregressive** generation: diffusion models start from noise and refine the whole output at once. Slower per sample, but parallel and revisable — the opposite trade.

---
# ⁉️
Autoregressive models are trained only to predict the next token. Yet a large one can be handed three examples of a brand-new task in its prompt and just... do it. With frozen weights. Nothing is being trained.

→ [[In Context Learning]]
