---
aliases:
  - MPC
  - Receding Horizon Control
  - Receding Horizon
  - Online Planning
  - Trajectory Optimization
  - Trajectory Optimisation
  - Shooting Methods
tags:
  - control-theory
  - planning
  - decision-sciences
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** At every step, **plan the next N moves** with your model (respecting every constraint), **do only the first one**, then look again and re-plan from scratch.
> **Metaphor:** Driving a winding road at night. You plan as far as the headlights reach, steer for the next second, and the headlights move with you.
> **Where it bites:** Anything with hard limits — chemical plants, battery management, autonomous driving, data-centre cooling. And it's the blueprint for model-based RL and for plan-act-observe agents.

---
A winding mountain road, at night. 🌙

Your headlights show about a hundred metres. Beyond that: black.

Think about what you actually do. You don't plan the entire drive home — you can't see it. You don't react only to the bit of tarmac under the bonnet either — you'd be in the ditch by the second bend.

You look at the hundred metres you *can* see, work out a line through it — *brake a touch, turn in, accelerate out* — and then you execute **about one second** of that plan. By which time the headlights have moved on, a new bend has appeared, and ~={blue}you throw the rest of the old plan away and make a new one.=~

---
# The headlights 🔦

![[mpc_headlights_receding_horizon.png]]

That's the entire algorithm. At every time step:

1. **Measure** where you are.
2. **Optimise** a sequence of the next $N$ actions, using your model to predict where each would take you — subject to hard **constraints** (pedal 0–100%, stay in lane, don't exceed 80°C).
3. **Apply only the first action.**
4. Shift the window forward one step. Go to 1.

Here it is steering a vehicle 10 m to a target, when the accelerator is limited to ±1 m/s²:

![[mpc_receding_horizon.png]]
> [!TIP] Reading the chart
> **Orange dashed** is the plan made *right now*, 20 steps (4 s) ahead. **Red dot** is the only piece of it that gets used. Look at the bottom panel: the command sits **exactly on the actuator limit** — full brake, then full throttle — for 32 of the 60 steps. The planner *knows* the limit is there and plans a trajectory that respects it, rather than asking for the impossible and being clipped.

> [!NOTE] Model Predictive Control
> A control method that, at each step, solves a finite-horizon optimisation over future actions using a model of the system, applies the first action, and repeats. Also called **receding-horizon control**. ^mpc-def

> [!SUCCESS] Core idea
> ~={pink}Plan open-loop, execute closed-loop.=~ Each individual plan is made blind to the future — but because you re-plan every step from a fresh measurement, errors in the model and surprises from the world get corrected almost immediately. The re-planning *is* the [[Feedback Loop|feedback]]. ^plan-open-execute-closed

---
# Why not the alternatives?

| | Sees the future? | Handles hard limits? | Survives a wrong model? | Compute at run-time |
|---|---|---|---|---|
| [[PID Controller\|PID]] | no | no (saturates, [[PID Tuning#^windup-pattern\|winds up]]) | ✅ | trivial |
| [[LQR]] | yes, infinitely | ❌ — can't even express one | partly | trivial |
| Plan once, follow the plan | yes | ✅ | ❌ — first gust and you're off the road | once |
| **MPC** | yes, $N$ steps | ✅ **explicitly** | ✅ — it re-plans | **an optimisation every step** |

**Constraints are the killer feature.** Most real money is made running a plant *right up against* a limit — maximum throughput, minimum energy, just inside the safety envelope. MPC can sit on that edge deliberately. Nothing else in the table can.

> [!WARNING] "MPC makes a plan and follows it"
> The most common misreading. It makes a plan and **throws 95% of it away**, every single step. The tail of the plan exists only so the *first* action is chosen with its consequences in mind. If you find yourself executing several steps of one plan to save compute, you've traded away the feedback that made it robust. ^mpc-discards-the-plan

---
# The dials

- **Horizon $N$.** Too short and it's short-sighted — it'll happily drive into a dead end it couldn't see. Too long and the optimisation gets slow and leans on a model that's wrong that far out. Rule of thumb: long enough to cover the system's [[Step Response|settling time]].
- **The model.** It must be good enough *over the horizon*, not forever. Get one from physics, or from data → [[System Identification]].
- **The solve time** has to fit inside one control period. This is why MPC arrived first in slow chemical plants (minutes per step) and only recently in cars and drones (milliseconds).

---
# The same shape, elsewhere 🔗

| Field | The "MPC" | 
|---|---|
| **Model-based RL** | learn the dynamics from data, then plan through it with MPC (PETS, and the planning half of [[Model-Based vs Model-Free RL\|Dreamer]]-style agents) |
| **Games** | [[Monte Carlo Tree Search]] — search a few moves deep from the *current* position, play one move, search again |
| **LLM agents** | think → act → observe → re-plan. [[Agentic Workflows]] is a receding-horizon loop with a language model as the planner |
| **Inference-time reasoning** | spend compute *at decision time* instead of baking everything into a reflex → [[Test-Time Compute]] |

> [!TIP] The trade behind all four
> A **policy** ([[Policy]]) is a reflex: all the thinking was done in advance, and acting is instant. **MPC** is deliberation: almost no thinking in advance, real thinking at every step. One amortises compute into training; the other spends it at run-time. The best systems — AlphaZero is the clean example — do both.

---
> [!SUCCESS] If you remember one thing
> **Plan N, do 1, repeat.** ~={pink}The plan isn't there to be followed — it's there so that the one action you actually take was chosen with its future in mind.=~

---
# ⁉️
PID needed no model. Feedforward, LQR and MPC all quietly assumed you *had* one — "press this much, get that much, after about this long." Nobody ships a car with that written in the manual. So where does the model come from?

→ [[System Identification]]
