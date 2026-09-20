---
aliases:
  - DL
  - Neural Networks
tags:
  - fundamentals
  - deep-learning
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Stack simple units into layers, let each layer learn a different level of abstraction, and train the whole thing by pushing the error backwards.
> **Metaphor:** Learning to recognise a face — edges, then shapes, then eyes and noses, then people. Each layer builds on the last.
> **Where it bites:** This note is the **spine**. It's the historical thread that every other note in the vault hangs off.

---
So, way back in the 50s Rosenblatt came up with an idea of [[#^perceptron|perceptron]].

> [!NOTE] Perceptron
> A perceptron is basically supposed to be a single unit of neuron which is capable of learning and recognizing patterns. And it does that by assuming that different patterns are all linearly separable. So if it can learn this linear plane of separation, it can basically recognize and classify which category the sample belongs to. ^perceptron

The preceptron brought in the idea of [[#^weights|weights and biases]]. 

Think of **weights** and **biases** as the scalar coefficients of the features. So imagine we're trying to teach the perceptron how to determine whether or not a person is overweight. Now we use the person's _weight_ ($x_1$) and _height_ ($x_2$) as features. Now if we were to plot the height and weight of every person (sample) in a xy-plane, and then we find a line that separates the individuals who are normal weight and overweight, then this line can be defined by some linear combination of $x_1$ and $x_2$, right? Say for example the line is $3.2x_1 + 2.5x_2 + 6$ now $3.2$ and $2.5$ are the **weights** the perceptron assigns to these features (basically the perceptron tells that the weight of the individual is more important than the height of the individual when we're trying to determine whether the person is overweight or not). And the **bias** term is like a default value when $x_1$ and $x_2$ are 0, so we still have a non-zero outcome. And it is these **weights** and **biases** that the perceptron adjusts when it is trying to learn. ^weights

But then what happens when the boundaries aren't linear? That's where the famous **XOR** problem popped up and showed that when the perceptron is given a _really simple XOR_ problem it fails to learn!

So then decades later, folks realized, **What if we stack perceptrons?** This stack of perceptron is called the [[#^mlp|multi-layer preceptron]].

> [!NOTE] Multi-layer perceptron / Feedforward neural net
> An **MLP** or a **Multi-layer perceptron** or alternatively called a **Feed forward neural network**, is basically a stack of preceptrons. And the idea here was may be each layer could learn a different level of abstraction. Like in image recognition, some layers are responsible to detect edges, while some recognize shapes and others recognize objects. Same concept! ^mlp

But then the question came, ~={blue}How do you teach the layers in the middle?=~ You can see what the output layer does (because it gives you a prediction), but how do you tell the hidden layers how wrong they were?

That’s where [[#^backprop|backpropagation]] came in.

>[!NOTE] [[Backpropagation]]
> Around 1986, **Rumelhart, Hinton, and Williams** rediscovered and popularized this technique (it was mathematically known earlier). Backpropagation (or “backprop”) means exactly what it sounds like:  you take the **error** (the difference between the model’s prediction and the true answer) and **propagate it backward** through the network to adjust every layer’s weights. ^backprop

---
### **Backprop, Gradients, and the Vector-Jacobian Trick**

Here’s the intuition:

The **loss** is a single number — like ~={red}“how wrong am I?”=~
The **gradient** tells you: ~={blue}“how should each weight change to reduce this loss?”=~

For the output layer, you can compute this easily.  
But for earlier layers, you have to use the **chain rule** — the fundamental rule of calculus that tells you how changes in one part affect another indirectly.

If we write it out carefully:

- The loss depends on the output of the network.
- The output depends on each layer’s weights.
- So, the loss depends on the weights _through_ all those layers.

That’s a mess to differentiate directly.  
So we use a clever trick: **the Vector-Jacobian Product (VJP)**.

Instead of writing full Jacobian matrices (which are huge), we multiply the “vector” of derivatives of the loss (∂L/∂output) by each layer’s Jacobian (∂output/∂weights).  
This efficiently computes ∂L/∂weights — the actual gradient needed to update the weights.

So in essence: the VJP simplifies how we compute gradients layer by layer.

That's the [[Vector Jacobian Product|VJP]], and it's what makes training a deep network computationally possible at all.

---
# But it still didn't work — the long winter ❄️

Here's the part of the story people skip. We had MLPs in the 80s. We had [[Backpropagation|backprop]] in 1986. And deep learning still didn't work for **another twenty-five years**.

Why? Three things were missing, and it's worth knowing which:

1. **Data.** Backprop-trained networks are enormously data-hungry, and there simply weren't labelled datasets of the right scale.
2. **Compute.** Nobody had realised that a graphics card — built to shade millions of pixels in parallel — is exactly the machine for multiplying large matrices. See [[GPU processing]].
3. **The gradient wouldn't flow.** With sigmoid activations, whose derivative maxes out at $0.25$, a deep stack **guarantees** the gradient shrinks at every layer. Deep networks were trainable in theory and untrainable in practice — [[Backpropagation#^vanishing-gradients|vanishing gradients]].

So the field went quiet, twice, and neural networks became slightly embarrassing to work on.

---
# The thaw — 2012

**AlexNet** wins ImageNet by a margin so large it ends the argument. → [[ImageNet Classification with Deep CNNs (AlexNet)]]

It wasn't a new idea. It was the three missing pieces arriving at once: **ImageNet** (the data), **two GPUs** (the compute), and **ReLU** (whose derivative is exactly $1$ for positive inputs, so the gradient passes through undamped — the fix for problem 3).

> [!SUCCESS] The lesson people keep re-learning
> The breakthrough wasn't a cleverer algorithm. It was ~={blue}an old algorithm finally meeting enough data and enough compute=~ — plus one small change to stop the gradient dying. That pattern repeats throughout this history. ^scale-not-cleverness

---
# Then everything became an architecture problem

With gradients flowing, the question became: **what shape should the network be?** Each answer is really a statement about what structure the data has.

### CNNs — exploit locality
A pixel relates to its neighbours, and a cat is a cat wherever it appears. So share the same small filter across the whole image. Massively fewer parameters, and translation invariance built in rather than learned.

**[[Deep Residual Learning for Image Recognition (ResNet)|ResNet]]** (2015) then made *depth* work by adding a `+x` shortcut, giving the gradient a path with local derivative exactly $1$ — a motorway straight back to the early layers. This is the single most important architectural idea for training depth, and it is in essentially everything now, transformers included.

### RNNs — exploit sequence
Feed the hidden state back in, so the network has memory. But repeated multiplication brings vanishing gradients back with a vengeance — solved with [[Gated Activation|gates]] in the LSTM. Sequential by construction, so you cannot parallelise across time, which eventually became the fatal limitation.

### Transformers — drop recurrence entirely
[[Attention Is All You Need|Attention is all you need]] (2017). Instead of passing a state along a chain, let every token look directly at every other token via [[Query, Key, and Value (QKV)|Q, K and V]]. The path between any two positions is now length **one**, not length $n$ — no vanishing gradient across time, and (crucially) the whole sequence trains in **parallel**.

> [!TIP] Why transformers actually won
> Not because attention is a smarter mechanism than recurrence. Because it's **parallelisable**, which means it can absorb far more data and compute per unit of wall-clock time. It won on scalability, and [[Foundation Models#Scale is not a detail — it's the mechanism|scale was the mechanism]]. 🏎️

---
# And then scale became the whole story

Once one architecture handled text, images, audio and video, progress stopped being about invention and became about **magnitude**.

[[Scaling Laws for Neural Language Models|Scaling laws]] showed loss falls as a smooth power law in model size, data and compute — you can *forecast* the performance of a model you haven't trained. [[Training Compute-Optimal Large Language Models (Chinchilla)|Chinchilla]] then corrected the recipe: scale parameters and data **together**, roughly 20 tokens per parameter.

Combine that with self-supervision — getting labels free from the data's own structure — and you arrive at [[Foundation Models]]: train once at enormous cost, adapt cheaply forever.

---
# The thread, in one place

| Era | Problem | Fix |
|---|---|---|
| 1958 | Linear boundaries only | Perceptron → XOR kills it |
| 1986 | Can't train hidden layers | [[Backpropagation]] |
| 1986–2012 | Gradients vanish, no data, no compute | ❄️ the winter |
| 2012 | — | ReLU + GPUs + ImageNet |
| 2015 | Depth still hard | Residual connections |
| 2017 | Sequences can't parallelise | [[Causal Attention\|Attention]] |
| 2020– | What now? | **Scale**, and [[Foundation Models]] |

> [!SUCCESS] If you remember one thing
> Nearly every milestone here is a fix for the same underlying problem: ~={pink}**keeping a useful gradient flowing through a long chain of multiplications.**=~ ReLU, residuals, normalisation, gates, attention — all of them, in different clothes.

---
# ⁉️
The mechanics under all of this: [[Backpropagation]] → [[Vector Jacobian Product]] → [[Pytorch Autograd]].
The maths under *that*: [[Derivative]] and [[Fundamentals]].
