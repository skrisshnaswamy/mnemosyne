---
aliases:
  - Rise Time
  - Overshoot
  - Settling Time
  - Steady-State Error
  - Transient Response
  - Damping
  - Damping Ratio
  - Underdamped
  - Overdamped
  - Critically Damped
tags:
  - control-theory
  - decision-sciences
  - metrics
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Jump the target once and watch what happens. Four numbers describe any controller: **rise time, overshoot, settling time, steady-state error.**
> **Metaphor:** A doctor's reflex hammer. One standard tap on the knee, and the twitch tells you the health of the whole system.
> **Where it bites:** Judging a tuning — and reading *any* curve that reacts to a sudden change: a loss curve after an LR change, an autoscaler after a traffic spike.

---
Three cars, all with cruise control. All three are cruising at 50 mph. You flick the setting to **60**.

- **Car A** creeps up. 55 after ten seconds… 59 after twenty… it's the best part of half a minute before it's there.
- **Car B** leaps. It blows straight through 60, peaks at **65**, drops back to 57, up to 61, down to 59… your passenger is feeling sick.
- **Car C** gets to 60 in about five seconds, touches 60.5, and stays.

Everyone can *see* that C is the good one. But "C feels nicer" won't survive a design review. ~={blue}How do you say it in numbers?=~

---
# The reflex hammer 🔨

A doctor doesn't assess your nervous system by watching you walk around all day. One standard tap below the knee, and the twitch says it all — too weak, too strong, or doesn't stop twitching.

Engineers do exactly the same to a control loop. The standard tap is a **step**: jump the setpoint from one value to another, instantly, and record what the output does. The resulting curve is the **step response**, and four measurements are read off it:

![[step_response_metrics.png]]

| Number | The question it answers | Car A | Car B | Car C |
|---|---|---|---|---|
| **Rise time** (10% → 90% of the way) | how *fast* does it respond? | 13.7 s | **2.0 s** | 3.5 s |
| **Overshoot** (% past the target) | how far does it *overdo* it? | 0% | **53%** (peaks at 65.3) | 4.6% |
| **Settling time** (inside ±2% and staying there) | when is it actually *done*? | 24.8 s | **32.7 s** | **10.0 s** |
| **Steady-state error** | does it end up *where you asked*? | 0 | 0 | 0 |

> [!NOTE] Step response
> The output of a system over time after its input (or setpoint) jumps from one constant value to another. It's the standard test because a step contains a bit of everything — a sudden change *and* a long hold. ^step-response-def

Now look at car B's row. It has the **fastest** rise time of the three and the **slowest** settling time. Sit with that for a second.

> [!SUCCESS] Core idea
> ~={pink}Fast and settled are different things, and they fight each other.=~ Push harder and you arrive sooner but overshoot more and ring for longer. Every tuning is a position on that trade-off — you cannot max out all four numbers at once. ^speed-vs-overshoot

---
# Damping — the one dial underneath

Those three cars differ in a single quantity: how much the system resists its own motion. That's **damping**, written $\zeta$ (zeta).

| | $\zeta$ | Behaviour | Everyday version |
|---|---|---|---|
| **Underdamped** | < 1 (car B: 0.2) | fast, overshoots, rings | a screen door that slams and bounces |
| **Critically damped** | = 1 | fastest possible *without* overshoot | a good door closer |
| **Overdamped** | > 1 (car A: 2.0) | no overshoot, sluggish | a door closer wound far too tight |

Car C sits at $\zeta = 0.7$ — slightly *under*damped on purpose. A sliver of overshoot (4.6%) buys a much faster arrival than perfect critical damping would. That's the usual engineering sweet spot.

And the fourth number? **Steady-state error** is zero for all three here. It isn't always — a purely proportional controller into a headwind settles short and stays short. That's the whole reason the **I** in [[PID Controller#**I** for Integral|PID]] exists.

---
# Why an ML person should care 📉

You read step responses every week. You just don't call them that.

| You did this (the step) | You watched this (the response) | Control reading |
|---|---|---|
| Raised the learning rate | loss jumps around, maybe diverges | gain too high → underdamped → [[Stability\|unstable]] |
| Added [[Momentum]] | loss overshoots the valley, swings back | you *lowered the damping* — a heavy ball rings |
| Traffic spikes 3× | autoscaler adds pods, overshoots, scales back | rise time vs overshoot, literally |
| Deployed a new ranker | metric dips, recovers over days | settling time of the whole product loop |

> [!TIP] LR warmup is "don't step — ramp"
> A step is the harshest input there is. Ramp the setpoint up gradually and the same loop barely overshoots. Learning-rate **warmup** is precisely that trick applied to an optimiser.

> [!WARNING] "The fastest response wins"
> The instinct is to tune for rise time, because that's what you notice first. But the number that usually matters to the user is **settling time** — and chasing rise time makes it *worse*. Car B is the quickest off the mark and the last to be usable.

---
> [!SUCCESS] If you remember one thing
> Poke it with a step and read four numbers: **how fast, how far past, how long until it's done, and where it ends up.** ~={pink}Any system that reacts to change can be judged this way — controllers, optimisers, autoscalers, teams.=~

---
# ⁉️
Car B overshoots wildly, but each swing is smaller than the last — it gets there in the end. Now imagine turning the aggression up a little more, so each swing comes out **bigger** than the one before.

→ [[Stability]]
