---
aliases:
  - Feedforward
  - Feed-forward Control
  - Disturbance Rejection
  - Cascade Control
  - Predictive Scaling
tags:
  - control-theory
  - decision-sciences
  - engineering
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Don't wait for the error — **measure the disturbance and act before it hits.** Fast, but blind to its own mistakes, so you always pair it with feedback.
> **Metaphor:** You can *see* the hill. A good driver leans on the pedal as the slope begins, not after the speed has already dropped.
> **Where it bites:** Predictive autoscaling before the 9am spike, cache pre-warming, planned LR schedules — and the name clash with "feedforward network".

---
Cruise control is holding 60 mph on the flat. Ahead, the road tilts up into a long, steep hill.

A feedback controller does nothing. Why would it? The speed is 60. The error is zero. It has no reason to act until the car has *already* slowed down — and with an engine that takes a few seconds to respond, by then you've lost 4 mph.

You, the human, do something different. You see the hill, and your foot goes down **as the slope begins**.

~={blue}What did you use that the controller didn't have?=~

---
# Seeing the hill ⛰️

Two things. You **measured the disturbance itself** (your eyes saw the slope), rather than waiting for its effect. And you had a rough **model** — *"a hill like that needs about this much more pedal."*

![[cruise_control_sees_the_hill.png]]

Here's the same car and the same hill, three ways:

![[feedforward_vs_feedback.png]]
> [!TIP] Reading the chart
> **Feedback only:** dips to **55.6 mph** before the PID drags it back. **Feedback + feedforward** (sees the hill 3 s early, model only 80% right): dips to **58.1** — and notice it nudges *above* 60 just before the hill, exactly like a human driver. **Feedforward alone:** its model was 20% short, so it settles at **56.0 and stays there forever** — nothing ever tells it that it's wrong.

> [!NOTE] Feedforward control
> Measure a disturbance (or know a setpoint change in advance), compute the action that should cancel it from a model of the plant, and apply it **before any error exists**. It is open-loop — see [[Feedback Loop#^open-closed-def|open vs closed loop]]. ^feedforward-def

> [!SUCCESS] Core idea
> ~={pink}Feedforward does the heavy lifting; feedback mops up whatever the model got wrong.=~ Feedforward is fast but can't see its own errors. Feedback sees every error but only after it's happened. Neither is enough alone, and together they cover each other's blind spot. ^ff-plus-fb

| | **Feedback** | **Feedforward** |
|---|---|---|
| Acts on | the error, after it appears | the disturbance, before its effect |
| Needs a model? | no | **yes** |
| Needs to measure the disturbance? | no | **yes** |
| Handles surprises it never anticipated | ✅ | ❌ |
| Corrects its own mistakes | ✅ | ❌ |
| Can cause [[Stability\|instability]] | yes — it's a loop | no — nothing loops back |
| Speed | limited by the loop's lag | as early as you can see |

That last-but-one row matters more than it looks: because feedforward isn't a loop, you can make it as aggressive as you like without ever risking oscillation. It lets you keep the feedback gains *gentle*.

---
# Cascade control — loops inside loops

A close cousin. Instead of one controller going from *speed error* straight to *engine torque*, use two:

- an **outer, slow** loop: speed error → *desired throttle position*
- an **inner, fast** loop: throttle-position error → motor current

The inner loop swats fast disturbances (a sticky throttle plate) before the outer loop ever notices them. Your heating does this too — the room thermostat doesn't drive the burner, it sets the target *water temperature*, and a second loop holds that.

---
# The ML versions

| Feedforward (planned from a model / forecast) | Feedback (reacts to the measured result) |
|---|---|
| **Predictive autoscaling** — scale up at 08:55 because the forecast says 09:00 spikes | reactive autoscaling on CPU |
| A pre-set **LR warmup + cosine schedule** | `ReduceLROnPlateau` |
| **Cache pre-warming** before a launch | cache eviction on miss rate |
| Capacity planned from the sales forecast | on-call paging when queues grow |

The production-grade answer is always both: forecast what you can, and keep a feedback loop underneath for what you couldn't. And [[Model Predictive Control]] is what you get when the feedforward plan is *recomputed at every step* — which quietly turns it back into feedback.

> [!WARNING] "Feedforward" in a neural network is a different idea — same word
> A **feedforward network** just means *data flows input → output with no cycles* (as opposed to a recurrent net). A **feedforward controller** means *acting on a measured disturbance before it causes error*. They share one property — ~={red}nothing loops back=~ — and nothing else. See [[Deep Learning]]. ^ff-name-clash

---
> [!SUCCESS] If you remember one thing
> If you can **see it coming** and you roughly know what it'll do, act early — and keep a feedback loop running to catch the part you got wrong. ~={pink}Forecast, then correct.=~

---
# ⁉️
Both of those needed a *model of the plant* — "this much pedal gives that much speed, after about this long." We've been waving our hands about it. How do control engineers actually **write down** how a system responds?

→ [[Transfer Function]]
