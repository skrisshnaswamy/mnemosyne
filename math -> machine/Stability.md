---
aliases:
  - Instability
  - Unstable
  - Stable System
  - BIBO Stability
  - Gain Margin
  - Phase Margin
  - Lyapunov Stability
  - Marginally Stable
  - Limit Cycle
  - Oscillation
tags:
  - control-theory
  - decision-sciences
  - failure-mode
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A loop is **stable** if its swings die away and **unstable** if they grow. The recipe for instability is always the same: ~={red}a strong reaction + stale information.=~
> **Metaphor:** The bad shower in an old house. Four seconds of pipe between the tap and your skin — and an impatient hand.
> **Where it bites:** Learning rates that diverge, autoscalers that flap, GANs that oscillate, retraining on delayed labels, and RL's "deadly triad".

---
Old house, bad shower. 🚿

The water's cold, so you turn the hot tap. Nothing happens. You turn it further. Still nothing. Further. Then, all at once, it's **scalding** — so you yank it back towards cold. Nothing… nothing… **freezing**.

You're now swinging between the two, and here's the uncomfortable part: the boiler is fine. The tap is fine. The pipes are fine. Your skin is an excellent thermometer. ~={blue}Every single component works — so what, exactly, is broken?=~

---
# The shower

The pipe between the tap and the shower head holds about **four seconds** of water. So what you feel *now* is the result of where the tap was *four seconds ago*.

While it still feels cold you keep turning — four seconds' worth of turning that you haven't felt yet. By the time the first of it arrives, far too much is already on its way and can't be recalled.

Let's put numbers on the impatience. Say you turn the tap at a rate proportional to how wrong the water feels. The only thing we'll change is *how hard* you react:

![[shower_delay_oscillation.png]]

![[stability_gain_delay.png]]
> [!TIP] Reading the chart
> Same shower, same 4-second pipe, same person. **Patient** (gain × delay = 0.4): glides to 38° and stops. **Impatient** (1.2): overshoots to 50°, rings, settles after a minute. **Very impatient** (1.8): every swing is bigger than the last, until it's slamming between the physical limits — scalding, freezing, forever.

For this loop the tipping point is exact: gain × delay $= \pi/2 \approx 1.57$. Below it the swings shrink. Above it they grow.

> [!NOTE] Stability
> A system is **stable** if a bounded input always produces a bounded output (*BIBO* stability) — disturbances die away. **Marginally stable**: it oscillates forever at constant size. **Unstable**: the response grows until something physical saturates or breaks. ^stability-def

> [!SUCCESS] Core idea
> ~={pink}Delay + gain = instability.=~ Your correction is aimed at the error you *saw*, but lands on the error that exists *four seconds later* — by which time the error may have changed sign. At that point your "negative" feedback is pushing **with** the swing, like shoving a child on a swing at exactly the wrong moment. Nothing is faulty. **The loop itself is the problem.** See [[Decision Sciences#^delay-causes-instability|the story version]]. ^delay-plus-gain

---
# The jargon, decoded

| You'll hear | It means |
|---|---|
| **"Poles in the left half-plane"** | the system's natural motions all decay → stable. See [[Transfer Function]] |
| **Gain margin** | how much more aggressive you could get before it goes unstable. *A safety factor on impatience* |
| **Phase margin** | how much more **delay** the loop could tolerate. *A safety factor on staleness* |
| **Limit cycle** | the saturated scalding-freezing oscillation — unstable, but clipped by physical limits |
| **Lyapunov stability** | find an energy-like quantity that *always decreases*; if one exists, the system must settle. A ball in a bowl loses height until it stops — see the bowl in [[Derivative#Hessian]] |

---
# What you can actually do about it 🛠️

> [!TIP] Four fixes, in the order you should try them
> 1. **Turn the gain down.** Be the patient showerer. Costs you speed ([[Step Response]]).
> 2. **Shorten the delay.** Move the sensor closer, sample faster, cut the pipeline lag. Usually the *best* fix and the least considered.
> 3. **Act on where the error is going, not where it was** — the **D** term in [[PID Controller|PID]], or [[Feedforward Control|feedforward]], or a model that predicts through the delay.
> 4. **Rate-limit the actuator** so no single correction can be huge.

---
# The same disease, in ML clothes

| System | The delay | The gain | The symptom |
|---|---|---|---|
| Gradient descent | one step of stale gradient | learning rate | loss oscillates, then diverges |
| Autoscaler | metrics lag 60 s, pods take 90 s to start | scale-up factor | **flapping** — up, down, up, down |
| Fraud model retrained on labels | chargebacks arrive 30 days late | how hard each retrain reacts | the model lurches between over- and under-blocking |
| [[Generative Adverserial Network\|GAN]] | each network chases the other's *last* move | both learning rates | oscillation instead of convergence → [[Mode Collapse]] |
| Deep RL | the target depends on the weights you're updating | step size | the [[Function Approximation#^deadly-triad\|deadly triad]] |

And notice what the fixes look like over there: [[PPO]]'s clip and the [[RLHF#^kl-leash|KL leash]] are **gain limiters**. [[DQN]]'s frozen target network is a way of **holding the target still**. Same medicine.

> [!WARNING] "Something must be broken" / "react faster"
> Two instincts, both wrong. People hunt for the faulty component — there isn't one. Then they try to fix an oscillation by responding *more* aggressively, which is precisely what causes it. ~={red}If a loop is oscillating, the first move is almost always to react less, or sooner — never harder.=~ ^react-less-not-harder

---
> [!SUCCESS] If you remember one thing
> Instability is not a bug in a part, it's a property of the **loop**: strong reactions to old news. ~={pink}Ask two questions of any oscillating system — how stale is the information, and how hard is the response?=~

---
# ⁉️
So the gain can't simply be cranked up. It has to be *chosen* — and a PID controller has three of them, which interact. How do people actually pick the numbers?

→ [[PID Tuning]]
