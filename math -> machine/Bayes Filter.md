---
aliases:
  - Bayesian Filtering
  - Bayesian Filter
  - Recursive Bayesian Estimation
  - Recursive Estimation
  - Predict-Update Cycle
  - Predict and Update
  - Sensor Fusion
  - Inverse-Variance Weighting
tags:
  - estimation
  - decision-sciences
  - probability
  - bayesian
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Carry a **belief**, not a number, and run two beats forever — **predict** (time passes, uncertainty grows) and **update** (evidence arrives, uncertainty shrinks). Kalman, HMM and particle filters are all this one loop.
> **Metaphor:** Breathing. Out — you didn't look, you know less. In — you looked, you know more.
> **Where it bites:** Tracking anything through noise, fusing two sensors, smoothing a jittery metric — and recognising that an EWMA is a filter with its trust dial stuck.

---
You're tracking a car. Your current belief: **100 m, give or take 10**.

One second passes. It was doing about 20 m/s, so you slide your guess forward to **120 m**. But are you still sure to within 10? You didn't look. It might have braked. So it's now **120 ± 14**.

Then the radar reports: **130 m ± 5**.

Two sources. They disagree by ten metres. Neither is exact. ~={blue}What's your best guess now — and how sure are you of it?=~

---
# Breathing 🫁

Most people's instinct is to pick one, or split the difference. The right answer is a weighted average — and the weights are just *how much you trust each source*, which is how **un**certain each one is.

The radar ($\pm 5$) is far tighter than your prediction ($\pm 14$), so it should get most of the say:

$$K = \frac{14^2}{14^2 + 5^2} = \frac{196}{221} = 0.887$$

$$\text{new guess} = 120 + 0.887 \times (130 - 120) = 128.9 \text{ m}$$

And the new uncertainty:

$$\sigma^2 = (1 - 0.887) \times 196 = 22.2 \;\Rightarrow\; \pm 4.7 \text{ m}$$

![[bayes_filter_gaussians.png]]
> [!TIP] Reading the chart
> **Left:** the blue belief is *narrower than both* sources — $\pm 4.7$ beats $\pm 5$ and $\pm 14$. Two blurry photographs, stacked, are sharper than either. **Right:** run it on and the uncertainty saws up and down forever — out on predict, in on update — settling into a rhythm of about $\pm 10.8$ before each look and $\pm 4.5$ after. It never reaches zero, because the world keeps moving between looks.

That number $K$ is the **Kalman gain** — the trust dial from [[Kalman Filter]]. The update has the same shape you'll see again and again:

$$\text{new} = \text{old} + K \times (\underbrace{\text{what I saw} - \text{what I expected}}_{\text{the surprise}})$$

> [!NOTE] Bayes filter
> The general recursion for tracking a hidden [[State-Space Model|state]]. Keep a probability distribution over the state (a [[Beliefs|belief]]). **Predict:** push it through the dynamics — it spreads. **Update:** multiply by the likelihood of the new observation and renormalise — it sharpens. Repeat. ^bayes-filter-def

> [!SUCCESS] Core idea
> ~={pink}You stop tracking the world and start tracking **what you know about the world**=~ — a best guess *and* a width. The width is what makes the whole thing work: it's what tells you how much to trust yourself versus the next measurement. See [[Decision Sciences#^track-what-you-know|the story version]] and [[Beliefs#^kalman-is-bayes|why Kalman is just Bayes]]. ^belief-not-number

---
# One loop, three costumes

The recursion is always the same. What changes is **how you choose to represent the belief**:

| Belief stored as | Works when | Predict step | Update step | Name |
|---|---|---|---|---|
| a **Gaussian** (mean + covariance) | linear dynamics, bell-curve noise | matrix algebra — exact | the gain formula above — exact | [[Kalman Filter]] |
| a Gaussian, with the curve *flattened locally* | mildly nonlinear | linearise, then as above | same | [[Extended Kalman Filter]] |
| a **table** of probabilities | a handful of discrete states | multiply by the transition table | multiply by emission probabilities | [[Hidden Markov Model]] (forward algorithm) |
| a **cloud of samples** | anything — multi-modal, nonlinear | move every sample | re-weight, resample | [[Particle Filter]] |

So "Kalman filter" isn't a separate idea. It's the special case where everything stays Gaussian, so the belief never needs more than two numbers per dimension.

---
# Sensor fusion — the update without the predict

Drop the dynamics and you have the everyday version. Two thermometers on the same pipe: one reads 80.0° ± 1°, the other 82.0° ± 2°. Weight each by $1/\sigma^2$:

$$\frac{80/1 + 82/4}{1/1 + 1/4} = \frac{100.5}{1.25} = 80.4° \qquad \sigma = \sqrt{1/1.25} = 0.89°$$

That's **inverse-variance weighting**. The cheap sensor still *helps* — $\pm 0.89$ beats the good sensor's $\pm 1$ — which is the argument for adding a second, worse sensor rather than throwing it away.

---
# Two things people get wrong

> [!WARNING] "More data drives the uncertainty to zero"
> Not in a moving world. Every predict step **adds** uncertainty back (process noise), so the filter settles at a floor — $\pm 4.5$ m in the example — where what each measurement removes equals what each time step adds. If you want it lower you need a better model or a better sensor, not more patience. ^uncertainty-floor

> [!WARNING] "It's just a moving average"
> An **EWMA** *is* a filter — one whose gain $K$ is frozen at a constant $(1-\beta)$. It trusts new data exactly as much on a calm day as on a chaotic one. A real filter **recomputes $K$ every step** from the two uncertainties. That's the entire upgrade. (Same shape as [[Momentum]]'s $\beta$, and as the [[Temporal Difference Learning|TD]] update — see [[Decision Sciences#^kalman-td-same-shape|the callback]].) ^ewma-is-fixed-gain

---
> [!SUCCESS] If you remember one thing
> **Predict → uncertainty grows. Update → it shrinks.** Weight every source by how certain it is, and ~={pink}two uncertain sources will always beat the better one alone.=~

---
# ⁉️
The update step squeezes information out of a measurement. But that assumes the measurement *contains* some. What if the thing you care about leaves no fingerprint on your sensor whatsoever?

→ [[Observability]]
