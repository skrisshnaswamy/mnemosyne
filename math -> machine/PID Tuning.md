---
aliases:
  - Controller Tuning
  - Loop Tuning
  - Ziegler-Nichols
  - Ziegler–Nichols
  - Integral Windup
  - Anti-Windup
  - Derivative Kick
  - Gain Scheduling
  - Adaptive Control
tags:
  - control-theory
  - decision-sciences
  - engineering
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Tune in the order **P → I → D**: P for speed, I to close the last gap, D to calm the overshoot. And whenever an integrator meets an actuator limit, expect **windup**.
> **Metaphor:** The same car as [[PID Controller]] — now with three unlabelled knobs on the dashboard and nobody to tell you the numbers.
> **Where it bites:** Any PID loop you'll ever commission. Also autoscalers stuck at max replicas, ad-budget pacing, and every "it overshoots horribly after being maxed out for a while" bug.

---
The [[PID Controller]] note ends with a formula and three gains: $K_p$, $K_i$, $K_d$.

Nobody tells you what they should *be*.

Set them too low and the car takes half a minute to reach 60. Too high and it [[Stability|oscillates]]. And they interact — touch one and the right value for the others moves. So you're sat in the car with three knobs and a stretch of open road. ~={blue}Where do you even start?=~

---
# One knob at a time 🎛️

Start with everything at zero and bring them in one by one. Here's the same car — setpoint 50 → 60 mph, straight into a headwind — at each stage:

![[pid_terms_p_pi_pid.png]]

**1 · P only** ($K_p = 5$). Quick off the mark, peaks at 59.8… and then settles at **58.0** and stays there. Forever. The error is 2 mph, so the push is $5 \times 2$, which is exactly what the headwind eats. That's the [[PID Controller#**I** for Integral|steady-state error]] — and no amount of waiting fixes it.

**2 · Add I** ($K_i = 0.5$). The accumulated error keeps leaning on the pedal until the gap is gone — it reaches 60.0. But now it overshoots to **62.7 mph (27%)** and takes **27 s** to settle. The integral that closed the gap also built up a surplus on the way there.

**3 · Add D** ($K_d = 6$). D sees the speed climbing fast and eases off early. Peak drops to **61.6 (16%)**, settled in **18 s**.

> [!SUCCESS] Core idea
> Each term fixes the previous one's flaw and introduces a smaller one of its own: ~={pink}P is fast but falls short. I closes the gap but overshoots. D calms the overshoot but hates noise.=~ That's why the order is P → I → D, and why you stop adding terms the moment the response is good enough — plenty of real loops are just PI. ^p-then-i-then-d

| Turn this up | Rise time | Overshoot | Settling time | Steady-state error |
|---|---|---|---|---|
| **$K_p$** | faster | more | roughly the same | smaller, never zero |
| **$K_i$** | faster | **more** | **longer** | **eliminated** ✅ |
| **$K_d$** | roughly the same | **less** ✅ | **shorter** ✅ | no effect |

(Read against the four numbers in [[Step Response]].)

> [!NOTE] Ziegler–Nichols — the 1942 recipe that's still a decent first guess
> Set $K_i = K_d = 0$. Raise $K_p$ until the loop *just* oscillates steadily — that gain is $K_u$, and the oscillation period is $T_u$. Then:
> $$K_p = 0.6\,K_u \qquad K_i = \frac{1.2\,K_u}{T_u} \qquad K_d = 0.075\,K_u T_u$$
> It deliberately gives an aggressive, ~25%-overshoot tuning. Treat it as a starting point and back off. ^ziegler-nichols

---
# Integral windup — the one that bites in production 🪤

Now the car hits a long steep hill. The controller floors the pedal — and it's **still** not enough. Speed sags to 55 and sits there.

Think about what the integral term is doing during those 40 seconds. The error is +5 mph the whole time, so it keeps adding… and adding. The pedal can deliver 14 units. The integral is *asking* for over a hundred.

Then the hill ends.

![[integral_windup.png]]
> [!TIP] Reading the chart
> The pedal is already at its limit, so winding the integral up further achieves **nothing** — except storing a debt. When the hill ends, the plain controller keeps the pedal floored while it pays that debt off: it sails to **64 mph** and takes **70 s** to come back. Freeze the integrator whenever the actuator is saturated (**anti-windup**) and it's **60.4 mph** and **14 s**.

> [!WARNING] The general pattern — worth carrying everywhere
> ~={red}Any integrator + any limit = windup.=~ An autoscaler pinned at max replicas keeps accumulating "need more", then refuses to scale down long after traffic drops. An ad-budget pacer that can't spend (no inventory) floods the auction the moment inventory appears. A rate limiter's backlog. If a loop *remembers* error and can *saturate*, ask what stops the memory growing while it's pinned. ^windup-pattern

---
# D's two bad habits

- **It amplifies noise.** The derivative of a jittery signal is enormous, so a raw D term makes the actuator twitch. Fix: low-pass filter the D term.
- **Derivative kick.** Step the setpoint and, for one instant, the *error* has an infinite slope — the actuator slams. Fix: differentiate the **measurement**, not the error. Same effect on disturbances, no kick on setpoint changes.

---
# When one set of gains isn't enough

A car in first gear and a car in fifth are different plants. **Gain scheduling** keeps a table of gains per operating regime (gear, altitude, load) and switches between them. **Adaptive control** goes further and adjusts the gains online from the observed response.

> [!TIP] You've met both already
> A learning-rate **schedule** is gain scheduling. `ReduceLROnPlateau` is a crude adaptive controller. And tuning three interacting gains with expensive trials is exactly the job [[Bayesian Optimization]] was built for.

Follow that thought one step further — *a controller that improves its own behaviour from experience, with no model of the plant* — and you've described [[Reinforcement Learning]].

---
> [!SUCCESS] If you remember one thing
> P → I → D, one at a time, watching the [[Step Response|four numbers]]. ~={pink}And wherever there's an integrator and a limit in the same loop, find out what stops it winding up — before production finds out for you.=~

---
# ⁉️
Everything so far *waits for an error to appear* and then reacts to it. But you, driving, can **see the hill coming**. Why wait for the speed to drop?

→ [[Feedforward Control]]
