---
aliases:
  - VJP
tags:
  - fundamentals
  - calculus
  - training
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Multiply the upstream gradient (a vector) by a layer's Jacobian **without ever building the Jacobian** — you get the downstream gradient directly.
> **Metaphor:** A relay race run backwards. Each layer takes the baton — "how much does the loss care about my output?" — and hands back "how much does it care about my input?"
> **Where it bites:** It's why [[Backpropagation]] is affordable at all, why reverse-mode beats forward-mode when there's one scalar loss, and what `backward()` in a custom `autograd.Function` actually has to return.

---
##### Definition
Given a vector $v∈R^m$ and a Jacobian matrix $J∈R^{m×n}$, the VJP is the product $v^{T}J$, resulting in a row vector of shape $1×n$.

##### Intuition
The VJP is the core engine of [[Backpropagation]]. Imagine you already know the gradient of your final loss $L$ with respect to a layer's output $y$ (this is your vector $v$ ). You want to find the gradient of the loss with respect to that layer's input $x$ . The VJP efficiently calculates this for you: 
it uses the upstream gradient $v$ and the local derivative $J$ to find the downstream gradient $∂L/∂x$ .

$VJP = ∇_x ( v·f(x) )$.

Simple Example: 
$$
f(x_1,x_2) = 
\begin{bmatrix}
x_1+2x_2 & 3x_1−x_2
\end{bmatrix}
$$
The jacobian (i.e. first-order derivative) is
$$
J = 
\begin{bmatrix}  
1 & 2 \\  
3 & -1  
\end{bmatrix}
$$


So for a 
$$
v = 
\begin{bmatrix}  
4 & 5
\end{bmatrix}
$$
then
$$
VJP = 
\begin{bmatrix}  
4 & 5
\end{bmatrix} . J
= 
\begin{bmatrix}  
19 & 3
\end{bmatrix}
$$

---
# Why not just compute the Jacobian?

Because you cannot afford it.

Take a modest layer mapping 1,000 inputs to 1,000 outputs. Its [[Derivative#Jacobian|Jacobian]] is $1000 \times 1000$ — a **million** entries, for **one** layer. A real network has hundreds of layers and much wider ones. Materialising these would need more memory than the model itself, by orders of magnitude.

But here's the saving observation:

> [!SUCCESS] Core idea
> ~={blue}You never want the Jacobian. You only ever want the Jacobian **multiplied by the upstream gradient**.=~
>
> And that product can almost always be computed directly, without ever building the matrix. ^never-build-the-jacobian

For $y = 3x$, the Jacobian is a diagonal matrix of 3s — but the VJP is literally `return 3 * grad_output`. For a matmul $y = Wx$, the Jacobian is enormous — but the VJP is just another matmul, roughly as cheap as the forward pass. Every primitive operation in PyTorch ships with a hand-written backward function that computes its VJP directly.

---
# Why *reverse* mode, and not forward

There are two ways to walk the chain rule, and the choice is decided entirely by the shape of your problem.

| | **Forward mode (JVP)** | **Reverse mode (VJP)** ✅ |
|---|---|---|
| Walks | inputs → outputs | outputs → inputs |
| Computes | one **input's** effect on all outputs | one **output's** dependence on all inputs |
| Cost | one pass **per input** | one pass **per output** |
| Good when | few inputs, many outputs | **many inputs, one output** |

Deep learning has a billion parameters and **one scalar loss**. That is the perfect case for reverse mode: a single backward pass gives you the gradient for every parameter at once.

Forward mode would need one pass *per parameter* — a billion passes per step. ~={pink}This asymmetry is the entire reason training large models is computationally possible.=~

> [!TIP] The cost that isn't obvious
> Reverse mode is cheap in **time** but expensive in **memory**: to walk backwards you must keep every intermediate activation from the forward pass alive. That's why peak memory occurs during backward, and why gradient checkpointing exists. See [[Pytorch Autograd#3. Backward OOMs even though forward was fine|autograd memory]]. ^reverse-mode-memory-cost

---
# Where you actually meet it

You don't write VJPs — you use them constantly:

- `loss.backward()` seeds $v = 1.0$ (the loss is a scalar) and chains VJPs down the graph. That's why `.backward()` **requires a scalar** — a vector output gives no obvious seed. See [[Pytorch Autograd#^why-scalar|why scalar]].
- `torch.autograd.grad(y, x, v)` is a VJP with the seed $v$ handed in explicitly.
- Custom `autograd.Function`s: `forward()` computes $f(x)$, `backward()` computes the VJP.

> [!SUCCESS] If you remember one thing
> A VJP is **"the Jacobian's effect, without the Jacobian."** [[Backpropagation]] is nothing more than a chain of these, right to left.

---
# ⁉️
The machinery that records which VJPs to apply, and in what order, is [[Pytorch Autograd]].
