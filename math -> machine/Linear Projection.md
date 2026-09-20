---
aliases:
  - Projection
  - Linear Layer
tags:
  - fundamentals
  - deep-learning
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Multiply your data by a **learnable matrix** to re-express it in a new space — same information, more useful shape.
> **Metaphor:** Converting a list of ingredients into a nutritional summary. Nothing was added; it was re-expressed along axes you actually care about.
> **Where it bites:** It's `nn.Linear`. It's the Q, K and V in [[Query, Key, and Value (QKV)|attention]]. It's every embedding table and every output head. It is the single most common operation in deep learning.

---
A simple way to transform data by multiplying it with a matrix (a learnable set of weights). It's like converting a list of ingredients into a nutritional summary; the information is the same, but it's in a new, more useful format.

It's exactly as the name says. You see when you multiply two vector i.e. dot product - it essentially is trying to project 1 vector onto the other's axis.

---
# Why "projection"?

Go back to the third perspective on matrices in [[Fundamentals#Matrix|Fundamentals]] — **matrices as transformation**. A matrix isn't a box of numbers, it's a *machine* that takes a vector and moves, stretches, squishes or rotates it.

A linear projection is that machine applied with intent. You have a vector living in one space, and you want it expressed in a **different** space:

$$y = Wx + b$$

- $x$ is your input vector, size $d_{\text{in}}$
- $W$ is the learnable matrix, shape $d_{\text{out}} \times d_{\text{in}}$
- $y$ comes out in a new space of size $d_{\text{out}}$

Each **row** of $W$ is a direction, and each number in the output is the dot product of your input with that direction — i.e. ~={blue}"how much of this input points along this direction?"=~ That's the projection. You're asking your data $d_{\text{out}}$ different questions, and the answers are the new coordinates.

> [!TIP] The one-line version
> A linear projection asks your data a fixed list of questions and writes down the answers. **Training is the process of learning which questions are worth asking.** ^questions-metaphor

---
# What the shape change means

The output dimension is a design decision with real meaning:

| | What you're doing | Example |
|---|---|---|
| $d_{\text{out}} < d_{\text{in}}$ | **Compressing** — forcing the model to keep only what matters | The bottleneck in an autoencoder; latent factors in [[Recommender Systems - Evolution\|matrix factorization]] |
| $d_{\text{out}} > d_{\text{in}}$ | **Expanding** — giving the model more room to separate things | The 4× expansion inside a transformer's feed-forward block |
| $d_{\text{out}} = d_{\text{in}}$ | **Re-orienting** — same capacity, better axes | The Q/K/V projections in attention |

That third row is the subtle one. If the size doesn't change, what was the point? **The axes changed.** The information is the same but now it's laid out so that the *next* operation — a dot product, a comparison, a threshold — becomes easy. That's the whole job of a Q/K/V projection: rotate the embedding so that "does this token want that token?" becomes a simple dot product.

---
# The catch — linear alone is not enough

Here's the thing that makes deep learning *deep*.

Stack two linear projections back to back:
$$y = W_2(W_1 x) = (W_2 W_1)x = W_{\text{combined}} \, x$$

Two matrices multiplied together are... **just another matrix**. So a hundred stacked linear layers collapse into exactly one linear layer. You gained nothing. You're back to the [[Deep Learning#^perceptron|perceptron]] and the XOR problem.

> [!WARNING] This is why activation functions exist
> A non-linearity (ReLU, GELU, a [[Gated Activation|gate]]) between the projections is what stops the collapse. **Linear layer = re-express. Non-linearity = decide.** Alternating them is what lets a network build up genuinely complex functions instead of one big linear one. ^why-nonlinearity

---
# Where you'll meet it

- **Embedding layers** — a projection from a one-hot vocabulary vector into a dense space. (In practice implemented as a lookup, because multiplying by a one-hot vector *is* just selecting a row.)
- **Q, K, V** — three separate projections of the same input, so the token can play three different roles. See [[Query, Key, and Value (QKV)]].
- **The output head** — projecting a hidden state back up to vocabulary size to produce logits for [[Cross Entropy]].
- **Two-tower retrieval** — projecting users and items into a *shared* space so a dot product means "match".
- **LoRA** — the observation that a fine-tuning update $\Delta W$ is usually **low-rank**, so you can approximate it with two skinny projections instead of a full matrix.

> [!NOTE] The bias term
> The $+b$ shifts the output away from the origin. Without it, a zero input must produce a zero output — the same reason the [[Deep Learning#^weights|perceptron]] needed a bias. Many modern architectures drop it anyway when a normalisation layer follows, since the norm removes the shift regardless.

---
### Context from Kidan
The "Vanilla Encoder" uses linear projections to create the [[Query, Key, and Value (QKV)]] from the input. This is the standard, efficient way of doing it as described in the original "[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)" paper.

> [!TIP] Why that matters
> Projecting into separate Q, K and V lets a token ask one thing, advertise another, and deliver a third. Kidan's CM2 encoder reused the *same* tensor for all three, which forces "what I'm looking for" and "what I offer" to be identical — and collapses a lot of what makes attention expressive. See [[Query, Key, and Value (QKV)]].

---
# ⁉️
So a projection re-expresses data along learned directions. The most important use of that trick is letting every token in a sequence produce three different views of itself — → [[Query, Key, and Value (QKV)]].
