---
aliases:
  - backprop
tags:
  - fundamentals
  - training
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Backprop is **blame assignment** — take the final error and pass it backwards through the network, working out how much each weight contributed to it.
> **Metaphor:** A company post-mortem. The product failed; walk back up the chain and work out how much each department's decision contributed.
> **Where it bites:** Vanishing/exploding gradients, why residual connections exist, why ReLU beat sigmoid, and every "my loss is NaN" conversation.

---
The backbone of machine learning. Without backprop, machines simply cannot learn. It's the process of finding how far away from truth is your prediction, and using this to tell your model, how to adjust it's weights and biases so that it can try again.

And you can't really learn backpropagation, without learning about [[chain rule]]. You see, in a deep learning network, we've many layers. And in such a network, backpropagation needs to use chain rule to compute the gradient at every single layer.

And on theory, backpropagation is done by computing the [[Derivative#Gradient|gradients]] of the outcome of each layer in the direction of the weights matrix. But unfortunately, a neural network is super complex with many layers - with each layer comprising of linear and non linear computation. So doing this is not really easy. So what Hinton suggested was this brilliant idea of [[Vector Jacobian Product]].

Since computing gradient with respect to the weights matrix directly isn't easy, we can simply compute the [[Vector Jacobian Product|VJP]] instead.

---
# The problem it solves — who do we blame?

Think about what's actually hard here. 🏢

A company ships a product. It flops. The CEO knows *one* number: revenue was £2M below target. That's the loss.

Now — how much of that £2M is **marketing's** fault? How much is **engineering's**? And within engineering, how much is the fault of one specific developer's one specific decision three months ago?

Nobody can answer that by staring at the £2M. You have to walk **backwards** through the chain of decisions.

That's exactly the situation a neural network is in. The loss is one number. The network has a billion weights. And you need a *separate* answer for every single one of them: ~={blue}"if I nudged you slightly, how much would that £2M change?"=~

> [!NOTE] What backprop actually computes
> For every weight $w$ in the network, the partial derivative $\frac{\partial L}{\partial w}$ — "how much does the total loss change per unit change in this one weight." That's it. Backprop is a **gradient-computing** algorithm, nothing more. ^what-backprop-computes

> [!WARNING] Backprop is not learning
> This trips people up constantly, and it's a good one to have straight. Backprop **computes** the gradients. The **optimizer** ([[Momentum|SGD, Adam]]) is what actually changes the weights. They're two separate steps:
> ```python
> loss.backward()      # backprop — compute who's to blame
> optimizer.step()     # optimizer — actually act on it
> ```
> Saying "the model backpropagates to learn" is loose. Backprop assigns blame; the optimizer does something about it. ^backprop-is-not-learning

---
# The walk backwards

Take a tiny chain. Input → layer 1 → layer 2 → loss.

The loss depends on layer 2's output. Layer 2's output depends on layer 1's output. Layer 1's output depends on the input and on layer 1's weights.

The [[chain rule]] says you can chain these together by **multiplication**:

$$\frac{\partial L}{\partial w_1} = \underbrace{\frac{\partial L}{\partial y_2}}_{\text{how much layer 2's output mattered}} \times \underbrace{\frac{\partial y_2}{\partial y_1}}_{\text{how much layer 1 affected layer 2}} \times \underbrace{\frac{\partial y_1}{\partial w_1}}_{\text{how much }w_1\text{ affected layer 1}}$$

Read it as the post-mortem: *"how much did this developer's decision affect their team's output × how much did their team affect the product × how much did the product affect revenue."* Multiply the influences along the path.

And the crucial efficiency insight: **you compute this once, from the right, and reuse it.**

When you're working out the blame for layer 1, you've *already computed* $\frac{\partial L}{\partial y_2}$ while doing layer 2. You don't redo it. Each layer receives one thing from the layer above it — "here's how much your output mattered" — combines it with its own local derivative, and passes a new message down.

> [!SUCCESS] Core idea
> Each layer only needs to know **two** things: the gradient handed down from above (the *upstream* gradient), and its own *local* derivative. Multiply, pass it on. No layer ever needs to understand the whole network. ^local-message-passing

This is why the whole backward pass costs roughly the same as **one** forward pass, regardless of how many parameters you have. That's the entire reason deep learning is computationally feasible — see [[Pytorch Autograd#What's really being multiplied: it's VJPs all the way down|why reverse mode wins]].

---
# Why VJP instead of the actual Jacobian

Look again at that middle term, $\frac{\partial y_2}{\partial y_1}$. If layer 1 outputs 1,000 numbers and layer 2 outputs 1,000 numbers, that term is a $1000 \times 1000$ [[Derivative#Jacobian|Jacobian]] — a million entries. For one layer. In a real network you'd need thousands of these, and they'd never fit in memory.

But notice: ~={blue}you never actually want the Jacobian. You only ever want the Jacobian *multiplied by the upstream gradient vector*.=~

And that product can almost always be computed directly, without ever building the matrix. That's the [[Vector Jacobian Product|VJP]] — $v^T J$, computed as a single cheap operation.

> [!TIP] Concrete example
> For $y = 3x$, the Jacobian is a diagonal matrix of 3s. But the backward function is literally `return 3 * grad_output`. Same answer, zero matrices built.
>
> For a matmul $y = Wx$, the Jacobian is enormous — but the VJP is just another matmul. Which is why the backward pass is roughly as fast as the forward one. ^vjp-avoids-jacobian

---
# What goes wrong — and it all comes from that multiplication

Here's the thing about a chain of multiplications: it is **fragile**.

If each layer's local derivative is around $0.5$, then after 50 layers the gradient reaching layer 1 is scaled by $0.5^{50} \approx 10^{-15}$. It is, for all practical purposes, **zero**.

> [!WARNING] Vanishing gradients
> The early layers receive essentially no signal and stop learning. The network trains its last few layers and leaves the rest at their random initialisation. This is *the* reason deep networks were considered untrainable for about two decades. ^vanishing-gradients

And the reverse:

> [!WARNING] Exploding gradients
> If each local derivative is around $1.5$, then $1.5^{50} \approx 6 \times 10^{8}$. The gradient arriving at the early layers is astronomical, the optimizer takes an enormous step, the weights fly off to infinity, and you get `NaN`. Common in RNNs on long sequences. ^exploding-gradients

Almost every architectural trick you know is a defence against one of these two:

| Fix | What it does |
|---|---|
| **ReLU** instead of sigmoid | Sigmoid's derivative maxes out at $0.25$ — it *guarantees* shrinkage every layer. ReLU's is exactly $1$ for positive inputs, so the gradient passes through undamped |
| **Residual connections** (ResNet) | Adds a `+x` shortcut, so there's a path where the local derivative is exactly $1$. The gradient has a clean motorway straight back to the early layers. This is the single biggest reason we can train 100+ layer networks |
| **Normalisation layers** | Keep activations in a sane range so derivatives don't drift toward 0 or ∞ |
| **Careful initialisation** (Xavier/He) | Sets the initial weight scale so the product stays near $1$ at the start |
| **Gradient clipping** | The blunt instrument — if the gradient norm exceeds a threshold, scale it down. Standard for RNNs and RL |

> [!SUCCESS] The unifying idea
> All of these are answering one question: ~={pink}**how do we stop a long chain of multiplications from collapsing to zero or blowing up to infinity?**=~ Once you see that, the history of deep learning architecture makes a lot more sense.

---
# A bit of history

Around 1986, **Rumelhart, Hinton, and Williams** published the paper that popularised backprop for neural networks. The maths — reverse-mode automatic differentiation — was known earlier (Linnainmaa had it in 1970), but this was the work that connected it to training multi-layer networks and made the field take notice.

It's worth knowing that backprop is **not** how brains learn. Biological neurons have no known mechanism for passing an error signal backwards along the exact same weights used in the forward pass (the "weight transport problem"). Backprop is an engineering solution, not a biological one — a useful thing to be able to say when someone leans too hard on the brain analogy. 🧠

---
> [!SUCCESS] If you remember one thing
> Backprop is the **chain rule applied by message-passing**, right to left, where each layer only needs its upstream gradient and its own local derivative. Every training pathology you'll meet comes from that chain being a long product.

---
# ⁉️
Backprop tells you the *direction* to move each weight. But how far? And should you go straight there, or remember where you were already heading?

→ [[Momentum]], and the machinery that runs all this for you, [[Pytorch Autograd]].
