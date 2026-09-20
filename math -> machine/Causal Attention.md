---
aliases:
  - Masked Attention
  - Causal Masking
  - Masking
tags:
  - transformers
  - attention
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Stop each token from seeing the future by setting its attention scores to $-\infty$ **before** the softmax.
> **Metaphor:** An exam where a card slides down the page — you can read everything above your current line, nothing below it.
> **Where it bites:** It's what makes a model decoder-only / [[Auto-regressive models|autoregressive]]. It's also why GPT can cache K/V and BERT can't.

---
# The cheating problem

[[Query, Key, and Value (QKV)|Self-attention]] lets every token look at every other token. Beautiful — and completely fatal if your job is **predicting the next word**.

Take the sentence *"The cat sat on the mat."* You're training the model to predict "mat" given "The cat sat on the". But plain self-attention lets the token at position 5 attend to position 6, which **is** "mat".

The model learns instantly: ~={red}"to predict the next word, just look at the next word."=~ Training loss drops to near zero. Then you run it at inference, where the future genuinely doesn't exist, and it produces nonsense. 💀

It didn't learn language. It learned to copy from a spot that won't be there later.

> [!NOTE] Causal attention
> Attention restricted so that a token at position $i$ can only attend to positions $\leq i$. "Causal" because information may only flow forwards in time — the way causality actually works. ^causal-def

---
# The fix, and why it's done the way it is

The obvious instinct is to zero out the illegal weights after the softmax. **Don't.** Softmax rows sum to 1; if you zero some entries afterwards they no longer sum to 1, and you've quietly rescaled everything.

Instead you sabotage the scores **before** the softmax by setting them to $-\infty$:

$$\text{masked score}_{ij} = \begin{cases} \dfrac{q_i \cdot k_j}{\sqrt{d_k}} & j \leq i \quad \text{(past and present — allowed)} \\[8pt] -\infty & j > i \quad \text{(future — forbidden)} \end{cases}$$

Then softmax does the rest for free, because $e^{-\infty} = 0$. The forbidden positions get **exactly** zero weight, and the remaining weights still sum to 1 all by themselves.

> [!SUCCESS] Core idea
> Mask **before** softmax with $-\infty$, never after with zeros. The normalisation then handles itself. ^mask-before-softmax

In practice it's not literally $-\infty$ (that gives you `NaN`) — it's a large negative number like `-1e9`, or `torch.finfo(dtype).min`. And that constant is dtype-dependent, which is a real gotcha with [[Mixed Precision training|fp16]]: a mask value that's fine in fp32 can overflow to `-inf` in fp16 and poison the softmax.

The mask itself is just a lower-triangular matrix of ones — `torch.tril` — so people often call it a **triangular mask** or **look-ahead mask**.

```
        The  cat  sat  on   the  mat
The     ✅   ❌   ❌   ❌   ❌   ❌
cat     ✅   ✅   ❌   ❌   ❌   ❌
sat     ✅   ✅   ✅   ❌   ❌   ❌
on      ✅   ✅   ✅   ✅   ❌   ❌
the     ✅   ✅   ✅   ✅   ✅   ❌
mat     ✅   ✅   ✅   ✅   ✅   ✅
```

---
# The payoff — parallel training

Here's what makes this genuinely clever, rather than just a safety measure.

You might think "one token at a time" means training has to be sequential, like an RNN. It doesn't. With the mask in place you can push the **whole sentence** through in a single forward pass, and every position simultaneously acts as a training example with the correct restricted view.

One pass over a 1,000-token document gives you 1,000 training examples. ~={blue}That parallelism during training is the single biggest reason transformers displaced RNNs=~ — same objective, but you can saturate a GPU with it.

> [!WARNING] Training is parallel, inference is not
> At **inference** you genuinely must go one token at a time, because token 6 doesn't exist until you've generated it. This asymmetry — parallel training, sequential generation — is the root of why inference is latency-bound while training is throughput-bound. See [[GPU processing]]. ^train-parallel-infer-sequential

---
# Causal vs bidirectional — the architectural fork

| | **Causal** (masked) | **Bidirectional** (unmasked) |
|---|---|---|
| Sees | past only | past **and** future |
| Trained by | next-token prediction | masked-token filling |
| Family | GPT, LLaMA, Claude — decoder-only | BERT, encoder-only |
| Good at | **generating** | **understanding / embedding** |
| Can cache K/V? | ✅ yes — past never changes | ❌ no — everything shifts when context changes |

That last row matters more than it looks. In a causal model, token 3's representation depends only on tokens 1–3, so once computed it is **final** — which is exactly what makes the [[Query, Key, and Value (QKV)#The KV cache — where this becomes an infra problem|KV cache]] possible. In BERT, adding a token at the end changes *every* representation, so there's nothing stable to cache.

> [!TIP] The trade
> Bidirectional models get richer representations (they see both sides) but can't generate. Causal models can generate but each token sees strictly less. It's a real trade-off, not a case of one being better — which is why encoder models are still the right tool for embeddings and retrieval.

Other masks exist too, and they're all the same trick with a different triangle: **padding masks** (ignore filler tokens in a batch), **prefix masks** (bidirectional over the prompt, causal over the generation), and **sliding-window masks** (only the last $k$ tokens, as in Mistral).

---
# ⁉️
Masking is what makes generation *possible*. Generating one token at a time, feeding each output back as the next input, is what makes a model → [[Auto-regressive models|autoregressive]].
