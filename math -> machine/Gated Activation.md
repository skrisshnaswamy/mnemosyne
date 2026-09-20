---
aliases:
  - Gating
  - GLU
  - SwiGLU
tags:
  - deep-learning
  - architecture
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A learned **valve** — one branch of the network decides how much of another branch's signal gets through.
> **Metaphor:** A dimmer switch, not a light switch. The network learns the setting, per input, per dimension.
> **Where it bites:** LSTM gates, SwiGLU in every modern transformer FFN, and the memory cost of that third weight matrix.

---
A "gate" in a neural network is a mechanism that learns to control the flow of information. It acts like a smart valve, deciding how much of a certain signal should be let through.

---
# The mechanism

Structurally it's always the same two-branch shape, and once you see it you'll spot it everywhere:

$$\text{output} = \underbrace{\sigma(W_g x)}_{\text{the gate — how much?}} \;\odot\; \underbrace{(W_v x)}_{\text{the content — what?}}$$

$\odot$ is elementwise multiplication. One branch computes **content**; the other squashes into $[0, 1]$ and computes **how much of that content survives**. Multiply them.

Because it's elementwise, the gate is **per-dimension** — it can pass feature 12 fully, block feature 47 entirely, and let feature 3 through at 30%, all for the same input. And it's **input-dependent**: the same weights produce different gate values for different inputs.

> [!SUCCESS] Core idea
> A plain activation like ReLU applies a *fixed* rule to every input. A gate applies a **learned, input-conditional** rule. That's the difference between a hard-wired filter and a dimmer the network gets to set on the fly. 🎛️ ^gate-vs-activation

---
# Why gates were invented — the memory problem

Gates came from RNNs, and specifically from [[Backpropagation#^vanishing-gradients|vanishing gradients]].

A vanilla RNN multiplies its hidden state by a weight matrix at every timestep. Over 100 steps, that's 100 multiplications — the gradient either vanishes to nothing or explodes. The network cannot learn dependencies more than a handful of steps apart.

**LSTM** (1997) solved it with three gates around a protected **cell state**:

| Gate | Question it answers |
|---|---|
| **Forget** | What should I drop from memory? |
| **Input** | What new information is worth storing? |
| **Output** | What part of memory is relevant right now? |

The crucial bit: the cell state is updated by **addition**, not repeated matrix multiplication. That gives the gradient a near-multiplication-free path back through time — the same escape hatch that residual connections give a deep feedforward network. ~={blue}Gates are how you build a shortcut that the network can learn to open and close.=~

> [!NOTE] GRU
> A simplification — two gates instead of three, no separate cell state. Fewer parameters, usually comparable performance. If someone asks "LSTM or GRU?", the honest answer is that it rarely matters much and GRU is cheaper.

---
# Gates in transformers — SwiGLU

RNNs are largely gone, but gating very much isn't. It moved into the feed-forward block of the transformer.

**Classic FFN** (two matrices):
$$\text{FFN}(x) = W_2 \,\text{ReLU}(W_1 x)$$

**Gated FFN / SwiGLU** (three matrices):
$$\text{FFN}(x) = W_2 \big( \underbrace{\text{Swish}(W_1 x)}_{\text{gate}} \odot \underbrace{W_3 x}_{\text{content}} \big)$$

An extra projection, an extra elementwise multiply — and it consistently outperforms the plain version at equal parameter count. LLaMA, PaLM, Mistral and most modern models use it.

> [!WARNING] The honest caveat
> Nobody has a fully satisfying theory for *why* SwiGLU is better. The original paper ends by attributing the gain to "divine benevolence" — a joke, but a genuine admission that it's an empirical finding. The plausible story is that multiplicative interactions let the layer express things additive ones can't, cheaply. Worth being able to say this rather than inventing a confident explanation. 🤷 ^swiglu-unexplained

> [!TIP] The parameter-count trick
> Three matrices instead of two means 1.5× the parameters at the same hidden width. So implementations **shrink the hidden dimension** — typically to $\frac{2}{3}$ of what it would have been — to keep total parameters constant. When you see a hidden size of $\frac{8}{3}d$ rather than a clean $4d$, that's why. ^swiglu-two-thirds

---
# The cost

Gating is not free, and this is exactly what bit Kidan:

1. **More parameters** — a third weight matrix per layer
2. **More activations to keep alive** — both branches must be retained for the backward pass, so peak memory rises. See [[Pytorch Autograd#3. Backward OOMs even though forward was fine|why backward OOMs]]
3. **More kernel launches** — an extra matmul plus an elementwise multiply, which is memory-bound and adds bandwidth pressure rather than useful arithmetic
4. **Sigmoid saturation** — a gate driven far toward 0 or 1 has near-zero gradient and can get stuck shut

---
### From Kidan
- Kidan's CM2 Encoder used a gate in its feed-forward layer.
- This is often more complex and memory-intensive than a simple [[ReLU Activation]], contributing to its poor performance.

> [!TIP] Consistent with the other CM2 findings
> Point 3 above lines up with what you saw elsewhere: the gate adds **bandwidth-bound** work, not matmul work. That's the same reason CM2 got little from [[Mixed Precision training#Kidan's experiments|BFLOAT16]] — Tensor Cores accelerate matmuls, and an elementwise gate isn't one. Three findings pointing at one bottleneck.

---
# ⁉️
Gates control *how much* signal flows. [[Query, Key, and Value (QKV)|Attention]] decides *where* signal flows from — and a softmax attention weight is itself a kind of learned, normalised gate over positions.
