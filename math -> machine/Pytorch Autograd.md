---
aliases:
  - autograd
  - Automatic Differentiation
  - Reverse-mode AD
tags:
  - fundamentals
  - pytorch
  - engineering
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Autograd is a **tape recorder** — it silently records every operation you perform on a tensor, then plays the tape **backwards** to compute gradients.
> **Metaphor:** A receipt for every step of a journey. `.backward()` walks the receipts in reverse and works out who owes what.
> **Where it bites:** `.grad` is `None`, gradients mysteriously accumulate, OOM during backward, `.detach()` vs `no_grad()`, "element 0 of tensors does not require grad".

---
# Nobody writes derivatives any more

Step back and appreciate how odd this is. You define a hundred-layer network **in Python, at runtime, with `if` statements and loops in it**, and then you call one method and get exact gradients for every one of a billion parameters.

Nobody derived that by hand. Nobody could.

The reason it works is that you're not asking for a *formula*. Symbolic differentiation (what Wolfram Alpha does) would give you an expression that explodes into unusable size for a deep net. Numerical differentiation (nudge a weight, see what changes) would need a billion forward passes and would be riddled with floating-point error.

**Automatic differentiation** is the third option, and it's the one that actually works: don't derive anything, just **remember what you did** and apply the [[chain rule]] step by step, backwards. ^three-ways-to-differentiate

---
# The tape 📼

Every time you do an operation on a tensor that has `requires_grad=True`, PyTorch quietly writes a note to itself:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x * 3        # note: "y came from x, via multiply-by-3"
z = y ** 2       # note: "z came from y, via square"
```

You wrote three lines of ordinary arithmetic. Behind your back, PyTorch built a little graph:

```
x ──(mul by 3)──► y ──(square)──► z
```

Each tensor carries a `grad_fn` — a pointer to the operation that made it, and to its inputs. That chain of pointers **is** the tape.

> [!NOTE] Dynamic graph
> The tape is built **as the code runs**, fresh every forward pass. Which is why a PyTorch model can contain a real Python `if` or a `while` loop whose length depends on the data — the graph just records whatever path actually executed this time. That's the "define-by-run" property, and it's the main reason PyTorch felt so much nicer than the older define-then-run frameworks. ^dynamic-graph

Then:
```python
z.backward()
print(x.grad)    # tensor([36.])
```

It walks the tape backwards. $\frac{dz}{dy} = 2y = 12$, then $\frac{dy}{dx} = 3$, multiply: $36$. Chain rule, executed by graph traversal.

> [!SUCCESS] Core idea
> Autograd doesn't *know calculus about your model*. It only knows the derivative of each **primitive operation** (`mul`, `exp`, `matmul`, `relu`…), which someone hardcoded once. Your model's gradient is just those tiny known derivatives, composed along the path you happened to take. ^primitives

---
# What's really being multiplied: it's VJPs all the way down

Here's the part that connects this to what you already wrote.

Each backward step *looks* like it should multiply by a full [[Derivative#Jacobian|Jacobian]] matrix. For a layer mapping 1,000 inputs to 1,000 outputs that's a $1000 \times 1000$ matrix — for a real layer, wildly too big to ever build.

So autograd **never builds one**. Every backward function computes a [[Vector Jacobian Product|VJP]] directly: it takes the upstream gradient vector $v$ and returns $v^T J$ without materialising $J$.

For `y = 3x`, the "Jacobian" is a diagonal matrix of 3s — but the backward function is literally `return 3 * grad_output`. Same answer, no matrix.

> [!TIP] This is the answer to "why does `.backward()` need a scalar?"
> A VJP needs a **vector $v$ to start with**. If your output is a single scalar loss, autograd can seed $v = 1.0$ automatically and go. If your output is a *vector*, there's no obvious seed — so PyTorch refuses and tells you `grad can be implicitly created only for scalar outputs`.
>
> That's also why you write `loss.backward()` and not `predictions.backward()`. And if you genuinely want a vector output's gradient, you supply the seed yourself: `y.backward(torch.ones_like(y))`. ^why-scalar

This also explains the direction. **Reverse mode** (one output → many inputs) costs one backward pass regardless of parameter count — perfect for ML, where you have one loss and a billion weights. **Forward mode** is the opposite trade and would need a billion passes. That asymmetry is the entire reason deep learning is computationally possible.

---
# The four things that actually go wrong

### 1. `.grad` is `None`
Almost always one of:
- **The tensor isn't a leaf.** Only leaf tensors (ones *you* created with `requires_grad=True`, i.e. your parameters) get `.grad` populated. Intermediate tensors have their gradients computed and thrown away to save memory. Want one? `y.retain_grad()`.
- **You broke the chain.** Somewhere you did `.detach()`, `.numpy()`, `.item()`, or a `.data` access, and the tape stops there.
- **You never called backward.**

### 2. Gradients accumulate — this is on purpose
```python
for batch in loader:
    optimizer.zero_grad()      # ← forget this and your gradients silently pile up
    loss = criterion(model(batch.x), batch.y)
    loss.backward()
    optimizer.step()
```

`.backward()` **adds into** `.grad`, it doesn't overwrite. Miss `zero_grad()` and step 100 is being updated by the sum of all 100 batches' gradients — training doesn't crash, it just quietly goes insane.

> [!TIP] Why would you ever want that?
> **Gradient accumulation.** Want an effective batch of 256 but only 64 fits in VRAM? Run four batches of 64, call `.backward()` on each, *then* `step()` and `zero_grad()`. The gradients add up exactly as if it were one big batch. It's the standard trick when you're VRAM-limited — see [[GPU processing]].

### 3. Backward OOMs even though forward was fine
Because the tape holds **every intermediate activation** alive — they're needed to compute the backward. Peak memory is during backward, not forward.

Fixes: **gradient checkpointing** (throw activations away, recompute them during backward — trades ~30% more compute for a large memory saving), smaller batches, or [[Mixed Precision training]].

### 4. You accidentally trained through your metrics
```python
total_loss += loss          # ← keeps the entire graph alive, every iteration 💥
total_loss += loss.item()   # ✅
```
Classic slow memory leak across an epoch.

---
# `no_grad()` vs `detach()` vs `eval()`

Three things constantly confused for each other. They do genuinely different jobs.

| | What it does | Use it for |
|---|---|---|
| `torch.no_grad()` | Context manager. **Stop recording the tape.** No graph built, less memory, faster. | Inference, validation loops, anything inside `optimizer.step()` |
| `.detach()` | Returns a tensor that shares the same data but is **cut off the tape**. Gradient stops flowing back through this point. | Target networks, momentum encoders, stop-gradient tricks |
| `model.eval()` | **Nothing to do with gradients at all.** Flips dropout off and makes BatchNorm use running stats. | Always, before evaluating |

> [!WARNING] `eval()` does not disable gradients
> This one costs people real VRAM. `model.eval()` only changes layer *behaviour*. You need **both**:
> ```python
> model.eval()
> with torch.no_grad():
>     preds = model(x)
> ```
> ^eval-vs-nograd

And the `detach()` case worth recognising: in [[Momentum#⚠️ The other momentum: momentum encoders|momentum encoders]] and target networks, the teacher's output is `.detach()`ed precisely so gradients *can't* flow into it. The stop-gradient isn't an optimisation there — it's load-bearing. Remove it and the model collapses.

---
> [!SUCCESS] If you remember one thing
> Autograd is **a tape of primitive operations, played backwards as a chain of [[Vector Jacobian Product|VJPs]]**. Every bug you'll hit is either "the tape was broken" or "the tape was kept alive too long."

---
# ⁉️
Autograd computes the gradients; the optimizer applies them. But all of it is arithmetic on a GPU, in a chosen numeric format, and *that's* where training speed actually lives.

→ [[Mixed Precision training]] and [[GPU processing]].
