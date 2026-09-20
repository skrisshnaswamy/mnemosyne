---
aliases:
  - Heavy Ball
  - Nesterov Momentum
  - Momentum Encoder
tags:
  - fundamentals
  - optimization
  - training
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Momentum makes the optimizer **remember which way it was already going**, so it barrels through flat regions and stops zig-zagging in narrow valleys.
> **Metaphor:** A heavy bowling ball rolling downhill instead of a hiker who stops and re-checks the map at every single step.
> **Where it bites:** It's inside Adam (and therefore inside almost everything). Also a *second, different* meaning — **momentum encoders** in MoCo/BYOL. Know which one is being discussed. ⚠️

---
> [!INFO] You already wrote the intuition
> The heavy-ball analogy and the Hessian comparison live in [[Derivative#Hessian and Momentum|Derivative → Hessian and Momentum]]. This note assumes that and goes further: the actual mechanics, the variants, and the second meaning of the word.

---
# A one-line recap, then onward

Vanilla [[Gradient Descent]] is a memoryless hiker: look at the slope right here, take one step, forget everything, repeat.

Momentum gives it **inertia**:
$$v_t = \beta v_{t-1} + \nabla L_t \qquad\qquad w_t = w_{t-1} - \eta \, v_t$$

Instead of stepping along the current gradient, you step along a **running blend** of every gradient you've seen — recent ones counting most.

Two things fall out of that, and they're the two reasons anyone uses it:
- **On a long gentle slope**, every gradient points the same way, so they *stack up*. Velocity grows, and you cross boring flat regions far faster than a hiker taking uniform steps. → **acceleration**
- **In a narrow ravine**, the side-to-side gradients keep *flipping sign*, so they cancel each other out in the sum. What survives is the small consistent component pointing down the valley. → **damping** 🎿

---
# What $\beta$ actually means

Everyone sets $\beta = 0.9$ and moves on. But it's worth knowing what number you're choosing, because it has a clean interpretation.

That velocity update is an **Exponentially Weighted Moving Average** of past gradients. Unroll it:
$$v_t = \nabla L_t + \beta \nabla L_{t-1} + \beta^2 \nabla L_{t-2} + \beta^3 \nabla L_{t-3} + \dots$$

Each step further back is multiplied by another $\beta$, so influence decays geometrically. Which means there's an *effective memory length*:

$$\text{roughly how many past gradients matter} \approx \frac{1}{1-\beta}$$

> [!TIP] Read $\beta$ as a window size
> - $\beta = 0.9$ → averaging about the last **10** gradients
> - $\beta = 0.99$ → about the last **100**
> - $\beta = 0.5$ → about the last **2** (barely momentum at all)
>
> So "$\beta = 0.9$" isn't a magic constant, it's the statement ~={blue}"smooth my gradient over roughly the last ten steps."=~ ^beta-window

This immediately explains the failure mode. Crank $\beta$ to $0.999$ and you're averaging over a thousand steps — the ball is now so heavy it sails straight past the minimum and takes forever to turn around. **More momentum is not more better.** It's a memory length, and you want it matched to how fast your landscape actually changes.

> [!WARNING] The $\eta$ / $\beta$ coupling
> Raising $\beta$ effectively raises your step size too — you're now summing ~10 gradients instead of 1. If you increase momentum and keep the same learning rate, you can get instability that looks like "the model diverged for no reason." The rough compensation is to scale $\eta$ down by about $(1-\beta)$. ^lr-momentum-coupling

---
# Nesterov: look before you leap

Classical momentum computes the gradient **where you are**, then adds the velocity. But you already know the velocity is going to carry you forward regardless — so why measure the slope at a spot you're about to leave?

**Nesterov Accelerated Gradient (NAG)** does the sensible thing: jump forward by the velocity *first*, and measure the gradient **there**.

> [!TIP] Driving analogy 🚗
> Classical momentum is braking based on where your car is *now*.
> Nesterov is braking based on where your car is *about to be*.
> If a wall is coming, the second one stops in time.

In practice it gives slightly better damping — it starts correcting a moment earlier, so it overshoots less. It's the `nesterov=True` flag in `torch.optim.SGD`, it costs nothing, and it's usually a small free win.

---
# Momentum inside Adam

This is the connection worth having ready, because "Adam" comes up constantly and most people can't unpack it.

Adam is **two** momentum terms stapled together:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L_t \qquad \text{← momentum on the gradient (}\beta_1 = 0.9\text{)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L_t)^2 \qquad \text{← momentum on the gradient }\textit{squared}\text{ (}\beta_2 = 0.999\text{)}$$
$$w_t = w_{t-1} - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}$$

- $m_t$ is exactly the heavy ball. **Which direction have I been going?** (~10-step memory)
- $v_t$ tracks the *magnitude* of gradients regardless of sign. **How wild has this parameter been?** (~1000-step memory)

Then it divides one by the other. So a parameter with consistently large gradients gets its step **shrunk**, and a parameter with tiny gradients gets its step **grown**. That's the "adaptive" in Adaptive Moment Estimation — every parameter gets its own effective learning rate.

> [!SUCCESS] Core idea
> **Momentum answers "which way?". The $v_t$ term answers "how big a step?". Adam is just both at once.** ^adam-is-two-momentums

> [!NOTE] Why bias correction exists
> $m$ and $v$ both start at zero, so for the first few steps they're biased toward zero and the steps come out far too small. Adam divides by $(1 - \beta_1^t)$ and $(1 - \beta_2^t)$ to undo this. Because $\beta_2 = 0.999$, that correction stays significant for **hundreds** of steps — which is part of why Adam wants a **warmup** period and behaves badly without one.

And remember the [[Regularization#The gotcha: weight decay ≠ L2 in Adam|AdamW gotcha]]: because Adam divides by $\sqrt{v_t}$, an L2 penalty folded into the loss gets divided too. Use **AdamW**.

---
# ⚠️ The other momentum: momentum encoders

Completely different thing, same word. If someone says "momentum" in a self-supervised learning conversation, they probably mean this one.

In **MoCo** and **BYOL**, you have two networks — a *student* (trained normally by [[Backpropagation|backprop]]) and a *teacher* (which is never trained by gradients at all). The teacher's weights are an **EWMA of the student's weights**:

$$\theta_{\text{teacher}} \leftarrow m \cdot \theta_{\text{teacher}} + (1-m) \cdot \theta_{\text{student}}, \qquad m \approx 0.999$$

Notice: this is the *same formula* as velocity, but applied to **weights** rather than gradients. Hence the shared name.

**Why bother?** These methods train by making two views of the same image produce matching representations. If both sides are the same rapidly-changing network, there's a trivial cheat — output a constant for everything and the loss is zero. Total collapse.

The momentum teacher blocks that. It's a **slow-moving, lagged copy** of the student — stable enough to be a meaningful target, but still drifting toward the student so the target improves over time. It's chasing a shadow of itself. 👥

> [!WARNING] Don't mix them up in conversation
> | | **Optimizer momentum** | **Momentum encoder** |
> |---|---|---|
> | Averages | past **gradients** | past **weights** |
> | Typical value | $\beta = 0.9$ (~10 steps) | $m = 0.999$ (~1000 steps) |
> | Purpose | Faster, smoother descent | Stable target, prevents collapse |
> | Lives in | SGD, Adam | MoCo, BYOL, mean-teacher, EMA checkpoints |
>
> The related trick you'll also see: keeping an **EMA copy of your model weights** and evaluating *that* instead of the live weights. Costs one extra copy of the model, often worth a free half-point. ^ema-weights

---
> [!SUCCESS] If you remember one thing
> Momentum is an **EWMA with a memory of about $\frac{1}{1-\beta}$ steps**. Applied to gradients it smooths your descent; applied to weights it gives you a stable teacher. Same formula, two jobs.

---
# ⁉️
Momentum, Adam, weight decay — all of these are things you *configure*, and then the framework quietly computes every gradient for you and applies them. But how does it actually know the gradient of a hundred-layer network you defined on the fly?

That's [[Pytorch Autograd]].
