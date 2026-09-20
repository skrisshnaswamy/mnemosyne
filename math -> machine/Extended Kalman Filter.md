---
aliases:
  - EKF
  - Unscented Kalman Filter
  - UKF
  - Nonlinear Kalman Filter
  - Nonlinear Filtering
  - Linearisation
  - Linearization
  - Sigma Points
tags:
  - estimation
  - decision-sciences
  - probability
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** The [[Kalman Filter]] only works for straight-line maths. The **EKF** copes with curves by pretending they're flat *near your current guess*; the **UKF** instead pushes a handful of test points through the real curve.
> **Metaphor:** A flat street map of a round Earth. Perfect for a walk across town. Useless for London → Tokyo.
> **Where it bites:** GPS + IMU fusion in every phone and drone, radar tracking, SLAM — and the classic failure: a filter that is *confidently* wrong.

---
A radar doesn't tell you where a plane is in $x$ and $y$. It tells you **range** and **bearing** — *"10 km away, due north."*

Range is measured well: ±0.3 km. Bearing is measured badly: ±25°.

So where, in $x$ and $y$, is the plane? The obvious move is to convert the best guess — 10 km due north is $(0, 10)$ — and draw a tidy ellipse of uncertainty around it.

~={blue}But what does "±25° at a fixed distance of 10 km" actually look like on a map?=~

---
# The flat map 🗺️

It looks like an **arc**. Swing a 10 km string through 50° and you trace a curved band — a banana, not an ellipse.

![[ekf_banana.png]]
> [!TIP] Reading the chart
> The grey cloud is where the plane could really be. Its true centre is at $y = $ **9.09 km** — *not* 10 — because every point on the arc except the very middle sits below $y = 10$. The **EKF** (red) insists on $y = 10.00$ and draws a flat pancake that misses most of the banana. The **UKF** (green) lands on 9.09.

That's the whole problem in one picture. A Gaussian pushed through a *straight-line* function stays a Gaussian — that's why the [[Kalman Filter]] is exact. Push it through a **curve** and it stops being a Gaussian at all. The filter needs a bell curve to keep working, and the world just handed it a banana.

![[ekf_flat_map_curved_earth.png]]

Two ways out.

**The EKF — flatten the curve where you're standing.** Take the nonlinear function, and replace it with its tangent at your current best guess — a first-order Taylor expansion, whose slope matrix is the [[Derivative#Jacobian|Jacobian]]. Locally it's a straight line, so the ordinary Kalman equations apply. It's the flat street map: fine, *as long as your uncertainty is small compared with how much the curve bends*.

**The UKF — don't flatten anything.** Pick a few carefully placed test points around your belief (**sigma points** — 5 of them here), push each one through the *real* function, and fit a fresh Gaussian to where they land. No derivatives at all.

> [!NOTE] Extended / Unscented Kalman Filter
> Approximations to the [[Bayes Filter]] for **nonlinear** dynamics or sensors, that keep the belief Gaussian. **EKF:** linearise the functions with a Jacobian at the current estimate. **UKF:** propagate deterministic sample points through the true functions and re-fit the mean and covariance. ^ekf-ukf-def

> [!SUCCESS] Core idea
> ~={pink}Both keep pretending the belief is a bell curve — they differ in *how they cope with the bend*.=~ EKF approximates the **function**. UKF approximates the **distribution**. Approximating the distribution is usually the better trade, and needs no calculus. ^approximate-function-vs-distribution

---
# Side by side

| | **Kalman** | **EKF** | **UKF** | [[Particle Filter]] |
|---|---|---|---|---|
| Handles nonlinearity | ❌ | mild | moderate | anything |
| Belief shape | Gaussian | Gaussian | Gaussian | **any** |
| Needs Jacobians | — | **yes** (error-prone by hand) | no | no |
| Cost | tiny | tiny | ~2–3× EKF | 100–1000× |
| Here: mean $y$ | — | 10.00 ❌ | 9.09 ✅ | 9.09 ✅ |

---
# How it fails — and it fails quietly

> [!WARNING] Confidently wrong
> The EKF linearises **at its own estimate**. If that estimate is off, the tangent is drawn in the wrong place, which produces a worse update, which moves the estimate further off. Meanwhile its reported covariance *shrinks* as usual — it has no idea. ~={red}The filter diverges while telling you it's getting more certain.=~
>
> Defences: start with an honest (large) initial uncertainty, inflate the process noise a little, and monitor the **innovation** — the gap between predicted and actual measurements. If the surprises are consistently bigger than the filter says they should be, it's lying to you. ^ekf-divergence

> [!WARNING] "The EKF is the optimal nonlinear filter"
> It isn't. The Kalman filter's optimality is a theorem about **linear-Gaussian** systems. The EKF is an engineering approximation with no such guarantee — it just happens to work very well whenever the uncertainty is small relative to the curvature, which for a 10 Hz GPS fix on a car, it is.

---
# Where you'll actually meet it

Your phone's blue dot is one — GPS (slow, absolute, noisy) fused with accelerometer and gyro (fast, relative, drifting). So is every drone's attitude estimate, and the tracking layer under a self-driving stack. In robotics it's the engine of **SLAM**: estimate the map and your position in it at the same time.

> [!TIP] The shape that transfers
> "Replace a curve by its tangent at your current guess, solve the easy problem, move, repeat" is **the** workhorse move of applied maths. Newton's method does it. Gradient descent does it — see [[Derivative]]. And [[Backpropagation]] is a Jacobian pushed through a chain of functions. The EKF is that same instinct applied to uncertainty.

---
> [!SUCCESS] If you remember one thing
> Straight lines keep bell curves as bell curves. Curves don't. ~={pink}The EKF flattens the curve; the UKF samples it; and when the belief is too strange for any bell curve, you need particles.=~

---
# ⁉️
Both of these still insist the belief is **one bump**. But picture a robot in a building with three identical corridors: it really could be in any of three places. No single bell curve can say "here, *or* there, *or* there."

→ [[Particle Filter]]
