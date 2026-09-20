---
aliases:
  - Linear Quadratic Regulator
  - Linear-Quadratic Regulator
  - Optimal Control
  - LQG
  - Linear Quadratic Gaussian
  - Riccati Equation
  - Separation Principle
  - Certainty Equivalence
tags:
  - control-theory
  - planning
  - decision-sciences
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** If the dynamics are **linear** and the cost is **quadratic**, the optimal controller is just $u = -Kx$ — a fixed gain — and $K$ falls out of solving the Bellman equation on paper.
> **Metaphor:** Two fines. One for every metre you drift from the lane centre, one for every jerk of the steering wheel. LQR finds the steering law that minimises the total bill.
> **Where it bites:** The bridge between control and RL — it's *the MDP you can solve with algebra*. Drones, satellites, and the sanity check every policy-gradient paper runs.

---
A drone is hovering. A gust shoves it one metre sideways.

You care about two things, and they fight:

- **being off target** — every moment spent away from the hover point is bad
- **effort** — every burst of thrust costs battery and shakes the camera

Snap back as hard as possible, and you burn energy and overshoot. Drift back gently, and you sit out of position for ages. A [[PID Controller]] would make you guess some gains and [[PID Tuning|tune]] until it looked right.

~={blue}What if you could write down *how much you care about each*, and have the gains handed to you?=~

---
# Two fines 🚓

Write the bill:

$$\text{cost} = \sum_t \big(\; \underbrace{q \cdot x_t^2}_{\text{fine for being off target}} + \underbrace{r \cdot u_t^2}_{\text{fine for effort}} \;\big)$$

Squares, so big errors hurt disproportionately (the same reason as [[Loss, Objectives, and Business Alignment#**L1 vs. L2 Loss**|L2 loss]]). The only thing you choose is the **ratio** $q : r$ — how many units of effort is one unit of error worth to you?

Now the remarkable part. With [[State-Space Model|linear dynamics]] $x_{t+1} = Ax_t + Bu_t$ and that quadratic bill, the [[Bellman Equation]] can be solved *exactly*:

- the value function turns out to be a quadratic bowl, $V(x) = x^\top P x$
- and the best action is **linear in the state**: $\;u = -K\,x$

One matrix $K$. No table, no grid — the state is continuous, *infinitely many states*, and the [[Curse of Dimensionality]] simply never shows up. $P$ comes from running the Bellman backup in matrix form until it stops changing — that's the **Riccati equation**.

![[lqr_q_vs_r.png]]
> [!TIP] Reading the chart
> Same drone, same gust, only $r$ changes. **Effort cheap** ($r = 0.01$): $K = [8.0,\ 4.0]$ — back within 5 cm in **1.0 s**, with a huge thrust spike (effort 11.1). **Balanced** ($r = 1$): $K = [0.93,\ 1.37]$, **3.0 s**, effort 0.35. **Effort expensive** ($r = 100$): $K = [0.10,\ 0.44]$, a lazy **9.3 s**, effort 0.01. Nobody tuned those gains. They fell out.

> [!NOTE] LQR
> **L**inear dynamics, **Q**uadratic cost, **R**egulator (drive the state to zero). The optimal policy is a constant state-feedback gain $u = -Kx$, obtained by solving the Riccati equation — which is the Bellman equation specialised to this case. ^lqr-def

> [!SUCCESS] Core idea
> ~={pink}LQR is a proportional controller whose gains are *derived from what you care about* rather than tuned by hand.=~ Look at $K = [0.93, 1.37]$: a gain on position and a gain on velocity — that's a **PD controller**. PID asks "what gains feel right?"; LQR asks "what do you value?" and computes them. ^gains-from-cost

---
# LQG — when you can't see the state

LQR assumes you know $x$ exactly. Add a noisy sensor and you need an estimate — a [[Kalman Filter]]. The combination is **LQG** (…Gaussian), and it comes with a gift:

> [!NOTE] The separation principle
> For linear-Gaussian systems you can design the two halves **independently**: build the best *estimator* as if you weren't controlling anything, build the best *controller* as if the state were known, then feed one into the other — and the result is optimal overall. Acting as though your best estimate were the truth is called **certainty equivalence**. ^separation-principle

That's the two-box architecture from [[Decision Sciences#How this connects back to Chapter 1|Chapter 2]]: `sensor → [filter] → belief → [controller] → action`. It survives into modern RL as *representation learning* vs *policy learning*.

> [!WARNING] It stops being true the moment things aren't linear-Gaussian
> In general, acting on your best guess as if it were certain is **wrong** — sometimes you should act to *gather information*. The [[POMDP|tiger problem]] is the classic counter-example: at 85% sure, the "certainty-equivalent" move loses money.

---
# Where it sits

| | Model | Cost / reward | Solution | Gains come from |
|---|---|---|---|---|
| [[PID Controller\|PID]] | none needed | implicit | 3 gains | hand tuning |
| **LQR** | linear, **known** | quadratic | closed form | **the cost weights** |
| [[Model Predictive Control\|MPC]] | any, known | any, **plus constraints** | optimise online, every step | the optimiser |
| [[Reinforcement Learning\|RL]] | **unknown** | any | learned from experience | data |

LQR is the one spot on that ladder where everything is exactly solvable — which is why RL researchers use it as a test bench: if your fancy algorithm can't recover the known-optimal $K$, something's broken.

> [!WARNING] "Optimal" means optimal *for your bill*
> LQR will find the perfect controller for the $q$ and $r$ you wrote down. Whether that's the behaviour you *wanted* is entirely on you. ~={red}Choosing $Q$ and $R$ is the real design work=~ — and it's the same job, with the same traps, as designing a [[Reward Function]]. It also has no notion of limits: it will cheerfully command 800% throttle. ^optimal-for-your-cost

---
> [!SUCCESS] If you remember one thing
> Linear world + quadratic bill → **the optimal policy is a constant gain**, and you get it by solving Bellman once, on paper. ~={pink}You don't tune the controller; you state your priorities.=~

---
# ⁉️
That last warning is the catch. Real actuators saturate, real lanes have edges, and LQR has no way to even *express* "never exceed this". To respect a hard limit you have to look ahead and plan around it.

→ [[Model Predictive Control]]
