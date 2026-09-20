---
aliases:
  - Transfer Functions
  - Laplace Transform
  - s-domain
  - Poles and Zeros
  - Poles
  - Frequency Response
  - Bode Plot
  - Block Diagram
  - Impulse Response
  - LTI System
  - Time Constant
  - Low-Pass Filter
tags:
  - control-theory
  - decision-sciences
  - fundamentals
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A compact formula, $G(s) = \dfrac{\text{output}}{\text{input}}$, that says how a linear system responds to *anything* — and whose **poles** tell you at a glance whether it decays, rings, or blows up.
> **Metaphor:** Logarithms. They turned multiplication into addition so people could use a slide rule. Laplace turns **calculus into algebra** so engineers can use block diagrams.
> **Where it bites:** Decoding classical-control jargon — *pole, zero, time constant, Bode plot, low-pass*. And it's the ancestry of the "state space models" (S4, Mamba) in sequence modelling.

---
You press the pedal 10% further and hold it there.

The speed doesn't jump. It climbs — quickly at first, then more and more slowly — and levels off 10 mph higher. After **5 seconds** it's covered about 63% of the distance. After 15 seconds it's basically there.

That single experiment told you almost everything about this car: push by one unit and you *eventually* get one unit of speed, and it takes *about 5 seconds* to mostly happen. Two numbers — a **gain** and a **time constant**.

Now: you want to bolt a PID controller in front of it, wrap a feedback loop around the pair, and predict whether the whole thing will [[Stability|oscillate]] — ~={blue}*before* you build it.=~ How?

---
# The slide rule 📏

The honest description of that car is a differential equation:

$$5\,\frac{dy}{dt} + y = u \qquad \text{("speed changes at a rate proportional to how far it still has to go")}$$

Chain a controller onto it and you're solving *systems* of differential equations. Wrap feedback around that and it gets miserable quickly.

Here's the trick people found. Before calculators, multiplying big numbers was slow — so you took **logarithms**, which turn multiplication into *addition*, did the easy sum, and converted back. The **Laplace transform** does the same thing one level up:

| In the time domain | In the $s$-domain |
|---|---|
| differentiate | **multiply by $s$** |
| integrate | **divide by $s$** |
| a differential equation | a polynomial equation |
| two systems in series (a convolution) | **multiply** their transfer functions |

Apply it to the car: $5s\,Y + Y = U$, so

$$G(s) = \frac{Y}{U} = \frac{1}{5s + 1}$$

> [!NOTE] Transfer function
> For a **linear, time-invariant** (LTI) system starting at rest: the ratio of output to input in the $s$-domain. It is a complete description of the system's input → output behaviour, written as a ratio of two polynomials in $s$. ^transfer-function-def

And now block diagrams become *arithmetic*. A PID controller is $C(s) = K_p + \frac{K_i}{s} + K_d s$ (now, past, future — as gain, divide-by-$s$, multiply-by-$s$). Controller then plant: $C \cdot G$. Close the loop around them:

$$\text{closed loop} = \frac{C\,G}{1 + C\,G}$$

No differential equations were solved. That's the whole appeal.

> [!SUCCESS] Core idea
> ~={pink}A transfer function lets you treat a dynamic system as a number you can multiply.=~ Series = multiply, feedback = one formula. It's a change of representation that makes the hard operation cheap — exactly the instinct behind [[Fourier Series Decomposition#^change-of-basis|the Fourier change of basis]]. ^tf-core

---
# Poles — the part worth remembering

Set the denominator to zero and solve for $s$. Those roots are the **poles**, and each one is a natural motion the system *wants* to make, of the form $e^{pt}$.

For the car: $5s + 1 = 0 \Rightarrow s = -0.2$. So its natural motion is $e^{-0.2t}$ — a decay with a time constant of $1/0.2 = 5$ seconds. There's the "63% in 5 s" again.

![[poles_and_responses.png]]
> [!TIP] Reading the chart
> The pole's **position** is the behaviour. Further left → dies away faster. Off the horizontal axis (a complex pair) → it rings, and the height is the ringing frequency. **On** the vertical axis → rings forever. Anywhere in the **right half** → grows without limit. ~={red}"All poles in the left half-plane" is what *stable* means.=~ ^poles-and-stability

**Zeros** are the roots of the *numerator*. They don't decide stability; they shape the response — a zero in the right half-plane makes a system initially move the **wrong way** (a bicycle does it: to lean into a left turn you first steer fractionally *right*).

---
# Frequency response, in one paragraph

Feed the car's pedal a slow sine wave and the speed follows it faithfully. Wiggle the pedal fast and the speed barely moves — the car can't keep up. It's a **low-pass filter**. Plug $s = j\omega$ into $G(s)$ and you get, for every frequency $\omega$, how much the system **amplifies** and how much it **delays**. Draw those two curves on log axes and that's a **Bode plot**. At $\omega = 1/\tau$ the gain has dropped to 71% and the lag is 45°. *Gain margin* and *phase margin* ([[Stability]]) are read straight off it.

> [!TIP] You've built this filter already
> An **EWMA** — and therefore [[Momentum]]'s velocity term — is a first-order low-pass filter with exactly this transfer function. $\beta$ sets the time constant: $\beta = 0.9$ remembers roughly the last 10 steps.

---
# Where it stops working

> [!WARNING] The fine print
> - **Linear, time-invariant systems only.** Real plants are neither; you *linearise* around an operating point and accept that the model is local.
> - **One input, one output**, comfortably. Many-in-many-out gets clumsy.
> - **It hides the internals.** $G(s)$ describes input → output and says nothing about what's happening *inside* the box.
>
> Those three limits are why modern control, estimation and RL all moved to a different description — one that keeps the internal variables in view. ^tf-limits

That description also gave its name to a family of sequence models. **S4** is literally a linear state-space layer $(A, B, C)$, and the reason it trains fast is that such a layer can equally be computed as a convolution with its **impulse response** — the time-domain twin of the transfer function. (**Mamba** makes those matrices depend on the input, which breaks time-invariance — so it gives up the convolution trick and uses a parallel scan instead.)

---
> [!SUCCESS] If you remember one thing
> **Gain** = where it ends up. **Time constant** = how long it takes. **Poles** = whether it decays, rings or explodes. ~={pink}If someone says "there's a pole in the right half-plane", they mean: this thing is unstable.=~

---
# ⁉️
A transfer function only relates what goes *in* to what comes *out*. But to predict a car you need things you can't read off the output alone — position isn't enough, you need its velocity too. What is the minimum you'd have to know about a system's insides?

→ [[State-Space Model]]
