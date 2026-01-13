>[!TIP] **How should an agent act when the world is noisy, the future matters, and the goal itself is unclear?**

---
### 1. The first instinct: feedback before intelligence

**Control theory (early 1900s–1950s)** 🎛️

This all begins with machines, not minds.

Engine governors, thermostats, autopilots. The problem was brutally practical:  
“How do I keep a system stable when the world pushes it around?” or rather "How do we get a machine to "behave" without a human constantly turning the knobs?"

#### 1. The Historical Problem
In the early days of the Industrial Revolution, engineers faced a major issue with steam engines. If you fed the engine more coal, it sped up. If the load changed (like a mill grinding more grain), it slowed down. A human could stand there and adjust the steam valve, but humans are slow, they get tired, and they make mistakes.
#### 2. The Core Intuition: The Feedback Loop 🔄
The solution was to create a **Feedback Loop**.

Imagine you are taking a shower. You want the water to be a perfect $38^\circ\text{C}$.
1. **Sense:** You feel the water on your skin.
2. **Compare:** Your brain compares the current temperature to your "target" (the perfect $38^\circ\text{C}$).
3. **Act:** If it’s too cold, you turn the hot handle up.
4. **Repeat:** You feel the water again and adjust further.

> [!TIP] This loop — **Sense → Think → Act** — is the heartbeat of all control systems.

[[PID Controller]] live here. They don’t “think.” They react.
A PID controller says:
- I see an error
- I push back
- I adjust how hard I push based on past and present error

> [!TIP] No uncertainty modeling. No planning. No notion of _future reward_. Just **feedback**.

This era gave us the most important primitive of all:  
> [!TIP] 👉 _**closed-loop control**_ → _action affects the world, which affects the next action._

Everything later inherits this loop.

> [!WARNING] **But then, why was this idea insufficient?**
Simple controllers like this are amazing for things that **stay the same** — like keeping a room at one temperature. But they struggle when:
>1. **The world is "noisy"** (the thermometer is broken or jittery).
>2. **Things are hidden** (you can't see the temperature, only the steam).
>3. **The future matters more than the now** (if I turn the heater on now, it takes 20 minutes to warm up).

---
### 2. Reality intrudes: noise, partial observation

**State-space models & [[Kalman Filter|Kalman filters]] (1950s–1960s)**

Then engineers hit a wall: **sensors lie**.

You don’t observe the true state of the world. You see noisy shadows of it.

>[!INFO] Self-driving car's cruise control
>Imagine you’re designing a self-driving car using just this PID controller logic.
>It works great for speed, but what happens if the "eye" of the car (the camera 📸) is a bit blurry? Or if the sensor that measures speed is jittery, jumping between 59 and 61 mph every second?

Kalman’s insight was subtle and profound:  
> [!TIP] “Let’s separate **what the world is** from **what we observe**, and reason probabilistically about the gap.”

This is where **state** becomes a formal concept:
- Hidden state (what actually matters)
- Observations (what you can measure)
- Dynamics (how state evolves)

Kalman filters didn’t choose actions.  
They answered a different question:
> [!TIP] “Given uncertainty, what do I _believe_ about the world right now?”

This belief-state idea quietly became foundational later.

---

### 3. Decisions become mathematical

**Operations Research & Dynamic Programming (1940s–1960s)**

In parallel—mostly driven by wartime logistics—another thread emerged.

The question here was not control but choice:  
> [!TIP] “How do I make a sequence of decisions that **minimizes cost** or **maximizes reward** **over time**?”

Richard Bellman introduces **dynamic programming** and the Bellman equation.

This is the first time someone formalizes:
- Long-term consequences
- Recursive value of decisions
- Optimality over trajectories, not steps

No learning yet.
The model is assumed known.  
But the _logic of planning_ is born.

---

### 4. The unifying abstraction

**Markov Decision Processes (1950s–1970s)**

MDPs are where everything clicks together.

They fuse:

- State (from control + filtering)
    
- Actions (from control)
    
- Rewards (from economics & OR)
    
- Dynamics (from physics)
    
- Planning (from dynamic programming)
    

The Markov property is the key simplifying assumption:

> “The future only depends on the present state and action.”

From this one assumption spill out:

- Value functions
    
- Q functions
    
- Policies
    
- Bellman expectation and optimality equations
    

This is the **mathematical skeleton of RL**, even though no learning has happened yet.

At this point, decision science exists—but only if you know the world model.

---

### 5. Reality intrudes again: the model is unknown

**Reinforcement Learning (1980s–1990s)**

Now comes the leap.

What if:

- You don’t know the transition dynamics?
    
- You don’t know the reward function exactly?
    
- You only learn by acting?
    

This is where RL is born—not as a new goal, but as a concession to ignorance.

Temporal Difference learning, Q-learning, policy gradients emerge.

Key shift:

> From _planning with a model_ → _learning through interaction_

Exploration vs exploitation becomes unavoidable.  
Regret becomes a metric.  
Learning replaces inference.

---

### 6. Bandits split off as a special case

**Multi-armed bandits & contextual bandits**

Bandits are not simpler RL. They are **RL with no state transitions**.

They exist because many real problems don’t have long horizons:

- Ads
    
- Experiments
    
- Recommendations
    
- A/B testing
    

Contextual bandits add partial state back in, without dynamics.

Bayesian optimization, Thompson sampling, UCB—all live here.

This is where **decision sciences meets statistics** in a very practical way.

---

### 7. Deep learning bends the curve

**Deep RL (2013–present)**

Everything before assumed small state spaces or handcrafted features.

Neural networks change that.

Now:

- States can be images, text, trajectories
    
- Value functions become approximators
    
- Policies become neural programs
    

This is not a new theory—just more expressive function approximation.  
But it unlocks scale.

---

### 8. Humans re-enter the loop

**RLHF, preference learning, inverse RL**

Here the reward function itself becomes uncertain.

Instead of:  
“Maximize this explicit reward”

We say:  
“Figure out what humans _seem to prefer_, then optimize that.”

This pulls in:

- Inverse reinforcement learning
    
- Preference modeling
    
- Contrastive learning over trajectories
    
- Human feedback as noisy reward signals
    

We are back, philosophically, to Kalman’s original insight:

> You never observe the true objective—only signals about it.

---

### So… is your progression “right”?

Yes in spirit. But the **true backbone** is:

- Feedback loops
    
- State vs observation
    
- Long-term optimization
    
- Uncertainty
    
- Learning under partial knowledge
    

PID → Kalman → MDPs → RL is not a strict lineage.  
It’s multiple traditions converging on the same problem from different angles.

---
