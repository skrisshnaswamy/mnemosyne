---
aliases:
  - System ID
  - SysID
  - Model Identification
  - Dynamics Learning
  - Learning the Dynamics
  - Black-Box Model
  - Grey-Box Model
  - Persistent Excitation
tags:
  - control-theory
  - planning
  - decision-sciences
  - engineering
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** **Poke the system, log what happens, fit a model of its dynamics.** It's regression — where the thing you're predicting is *the next state*.
> **Metaphor:** A car with no manual. Before you can tune anything, you take it for a test drive and write down how it responds.
> **Where it bites:** Every model-based method needs this first. And its two traps — *you only learn about what you poke*, and *data logged under a controller is confounded* — are the same traps as exploration and off-policy evaluation.

---
You've been handed a car to fit cruise control to. No manual, no spec sheet.

[[Feedforward Control|Feedforward]], [[LQR]] and [[Model Predictive Control|MPC]] all need to know: *if I press the pedal this much, what does the speed do, and how quickly?* Nobody can tell you.

So you find an empty road, hold 50 mph, and at $t = 0$ push the pedal from 20% to 30% — and hold it there, logging the speed once a second.

~={blue}From thirty noisy numbers, what can you work out about this car?=~

---
# The test drive 🚗

![[sysid_step_fit.png]]

Two things, mostly:

- **Where it ends up.** +10% pedal eventually gave about +10 mph → a **gain** of ≈ 1 mph per %.
- **How long it takes.** It covered 63% of that — reached 56.3 mph — after about 5 s → a **time constant** of ≈ 5 s.

Fit the curve $K \cdot 10 \cdot (1 - e^{-t/\tau})$ properly by least squares and you get **$K = 0.98$, $\tau = 4.7$ s**. The truth (I simulated it) was 1.00 and 5.0. From one poke and thirty points, that's a usable model: $G(s) = \frac{0.98}{4.7s + 1}$ — see [[Transfer Function]].

> [!NOTE] System identification
> Building a mathematical model of a dynamic system from measured input–output data: choose a model structure, design an experiment that excites the system, fit the parameters, and **validate on data you didn't fit to**. ^sysid-def

> [!SUCCESS] Core idea
> ~={pink}It's supervised learning where the label is the next state.=~ Inputs: (current state, action). Target: what happened next. Everything you know from [[Regression Analysis]] — fit, residuals, held-out validation, overfitting — applies unchanged. ^sysid-is-regression

---
# The obvious way to fit it — and why it misled me

The natural regression is one step ahead: $\;x_{t+1} = a\,x_t + b\,u_t$. Line up the log against itself, shifted by one row, and run least squares. Done in two lines.

On this data it returns $\tau = $ **4.4 s** — noticeably worse than the 4.7 s from fitting the whole curve. Same data. Why?

Because $x_t$ is on the **right-hand side** of that regression, and $x_t$ is *noisy*. Noise in a regressor biases the coefficient toward zero (*errors-in-variables*), so $a$ comes out too small and the system looks faster than it is. Fitting the simulated **trajectory** to the data — an *output-error* fit — doesn't have that problem.

> [!WARNING] One-step fit ≠ good multi-step model
> A model can have a tiny one-step prediction error and still drift badly when rolled forward 50 steps — because each step's small error is fed into the next. ~={red}If you're going to *plan* with the model, validate it on multi-step rollouts=~, not just next-step error. This is exactly the compounding-error problem that haunts model-based RL. ^validate-on-rollouts

---
# What kind of model?

| | What you assume | Fit | Good for |
|---|---|---|---|
| **White box** | the physics, fully | nothing — it's derived | rare; things you built |
| **Grey box** | the *structure* from physics, unknown constants | a few parameters (mass, drag, $\tau$) | most engineering. Data-efficient, extrapolates sensibly |
| **Black box** | nothing | everything — ARX, a neural net | complex systems with lots of data. Extrapolates poorly |

A neural network trained to predict the next state is black-box system identification. In RL it's called a **world model** → [[Model-Based vs Model-Free RL]].

---
# The two traps 🪤

> [!WARNING] Trap 1 — you only learn about what you poke
> If the pedal sat at 20% all day, the log contains **zero** information about how speed responds to the pedal — however many hours of it you have. The input has to move, and move richly enough to reveal the dynamics you care about (*persistent excitation*). "We have months of production logs" is often worth less than ten minutes of deliberate experiment.
>
> It's an [[Observability#^plumbing-not-tuning|is-the-information-even-there]] problem, and it's the engineering ancestor of [[Exploration vs Exploitation]]: to learn the world you sometimes have to do something other than the thing you think is best. ^persistent-excitation

> [!WARNING] Trap 2 — data logged under a controller is confounded
> Log a car *with cruise control on*. Every time there's a hill, the controller adds pedal. In the data, "more pedal" now coincides with "speed not rising". Fit that naively and you'll conclude the pedal barely works.
>
> The action was chosen **in response to** something that also affects the outcome — that's confounding, and it's the same disease as [[Regression Analysis#^coefficient-is-not-causal|reading a regression coefficient as a causal effect]]. The cures are the same too: inject your own randomised input on top, or correct for how the actions were chosen → [[Off-Policy Evaluation]]. ^closed-loop-confounding

---
# The bridge

> [!TIP] Put the pieces together
> **System identification** (learn the model from data) **+** [[Dynamic Programming|planning]] (use the model to decide) **=** model-based reinforcement learning. Control engineers were doing "learn a model, then plan with it" decades before it had that name.

---
> [!SUCCESS] If you remember one thing
> Poke it on purpose, fit the simplest model that explains the response, and check it on rollouts you didn't fit to. ~={pink}Passive logs from normal operation tell you far less than they look like they should.=~

---
# ⁉️
We now have every ingredient: a **state**, **actions** that change it, a **model** of how, and a **cost** over time. They arrived from four different traditions. It's time they had one name — and it starts with the assumption that made "state" well-defined in the first place.

→ [[Markov Property]] → [[Markov Decision Process]]
