---
aliases:
  - Model-Based RL
  - Model-Free RL
  - Model-Based Reinforcement Learning
  - Model-Free Reinforcement Learning
  - Model-Based
  - Model-Free
  - World Model
  - World Models
  - Dyna
  - Dyna-Q
  - Learned Dynamics Model
  - Sample Efficiency
tags:
  - reinforcement-learning
  - planning
  - decision-sciences
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** **Model-based** learns *how the world responds* and then plans with it. **Model-free** skips that and learns *what to do* (or *how good things are*) directly from experience.
> **Metaphor:** Moving to a new city. Learn the **map** — or just learn your **habits**.
> **Where it bites:** Sample efficiency vs simplicity. And the word "model" here means a model *of the environment*, which trips everyone up.

---
You move to a new city. Two ways to get good at the commute.

**The map.** You study the streets. Which roads connect, how long each takes, where it jams at 8:45. Now you can work out a route to *anywhere* — including places you've never been.

**The habits.** You just… commute. After a few weeks you know: *out of the door, left; at the lights, straight on.* You couldn't draw the map if asked. You don't need to — you just *know what to do*.

One Monday, your usual bridge is closed.

~={blue}Which version of you copes — and which version got good faster?=~

---
# Map or habits 🗺️

![[model_based_vs_model_free_map_vs_habit.png]]

Map-you reroutes instantly. Habit-you is stuck — the habit said "cross the bridge", and it has nothing else to offer until it has re-learned by trial and error.

But habit-you had a simpler job. Building an accurate map of a city is *hard*; a slightly wrong map sends you confidently down a dead end.

This is the same split as **Table A vs Table B** in [[Decision Sciences#Two tables 📋|the chapter]]:

| | **Table A — the mechanics** | **Table B — the verdict** |
|---|---|---|
| Contains | "from `Low`, ordering → `Medium` 80%, `High` 20%" | "from `Low`, ordering is worth **40**" |
| This is | a **model** of the environment: $P$ and $R$ | a **[[Value Function\|Q-function]]** |
| To act you must | *plan* through it — [[Dynamic Programming\|DP]], [[Model Predictive Control\|MPC]], [[Monte Carlo Tree Search\|search]] | just read off the biggest number |
| Learned by | [[System Identification\|fitting next-state predictions]] | [[Temporal Difference Learning\|averaging what you actually earned]] |

> [!NOTE] Model-based vs model-free
> **Model-based:** learn (or be given) the transition and reward functions, then derive behaviour by planning. **Model-free:** learn a value function and/or policy directly from sampled experience, never representing the dynamics. ^mb-mf-def

> [!WARNING] "Model-free" does not mean "no neural network"
> *Model* here means **a model of the environment** — a thing that predicts "what happens next if I do this". [[DQN]] and [[PPO]] contain huge networks and are both **model-free**, because those networks predict *value* and *action*, not *the next state*. ^what-model-means

---
# The trade

| | **Model-based** | **Model-free** |
|---|---|---|
| Real experience needed | **little** — each real step can be replayed and re-imagined many times | **a lot** |
| Compute | heavy — planning at decision time and/or in the background | light at decision time: a reflex |
| The world changes | re-plan immediately | re-learn slowly |
| Transfers to a new goal | ✅ same map, new destination | ❌ the values were *for the old reward* |
| Best achievable performance | capped by **model error** | no such cap |
| Typical failure | exploits flaws in its own model | never gets enough data |

> [!SUCCESS] Core idea
> ~={pink}A model lets you learn from experience you never had.=~ That's where the sample efficiency comes from — and also the danger: a planner will happily find the spot where your model is wrong in an *optimistic* direction and drive straight at it. Same disease as [[Reward Hacking]], aimed at your dynamics instead of your reward. ^learn-from-imagined-experience

---
# Dyna — just do both

Sutton's Dyna (1990) is the simplest possible hybrid, and it's worth knowing because it *is* the modern recipe in miniature:

1. Take a **real** step. Update your values from it (model-free).
2. Also record it in a model: *"in this state, that action led there."*
3. Now, before the next real step, replay $n$ **imagined** steps from the model, updating the values from each.

![[dyna_planning_steps.png]]
> [!TIP] Reading the chart
> Same maze, same real experience per episode. With **0** imagined updates (plain [[Q-Learning]]) it takes until episode **25** to get under 20 steps. With **5**: episode **5**. With **50**: episode **3**. The agent isn't seeing more of the world — it's ~={blue}thinking harder about what it has already seen.=~ (If that sounds like [[Experience Replay]], it should: replay is a Dyna whose "model" is just a list of stored transitions.)

---
# Where the model breaks

One-step predictions can be excellent and the rollout still useless, because each step's small error is the *next* step's input:

> 99% accurate per step → after 50 imagined steps: $0.99^{50} = 61\%$. After 200: $13\%$.

Same arithmetic as [[LLM Engineering#^compounding-error|the agent-reliability number]]. Hence the standard defences: keep imagined rollouts **short**, branch them from **real** states, use an **ensemble** of models and distrust states where they disagree, and [[System Identification#^validate-on-rollouts|validate on multi-step rollouts]].

---
# The modern family tree

| | Idea | Note |
|---|---|---|
| **Dyna** | real + imagined updates | above |
| **MPC with a learned model** | fit dynamics, plan with [[Model Predictive Control]] every step | PETS and relatives |
| **AlphaZero** | the rules are *given*; search them with a learned value/policy | [[Monte Carlo Tree Search]] |
| **MuZero** | *learn* a latent model good enough to search in — it never reconstructs the board | same |
| **World models / Dreamer** | learn a compact latent dynamics model, then train the policy almost entirely **inside the dream** | [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)]], [[Mastering Diverse Domains through World Models (DreamerV3)]] |

> [!TIP] Psychology uses the same two words
> *Goal-directed* behaviour (flexible, effortful, model-based) versus *habitual* behaviour (fast, rigid, model-free) — with evidence that the brain runs both and arbitrates between them. It's a useful way to remember which is which.

---
> [!SUCCESS] If you remember one thing
> **Map** = learn the rules, then plan: data-efficient, flexible, only as good as the map. **Habits** = learn what to do: simple, robust, data-hungry. ~={pink}Most strong systems keep a map *and* habits.=~

---
# ⁉️
Start with the habits route. With no model you can't *compute* how good a state is. But you can always do the dumbest possible thing: go there, see what you end up earning, and take the average.

→ [[Monte Carlo Methods]]
