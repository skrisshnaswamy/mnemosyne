---
aliases:
  - MARL
tags:
  - MARL
  - reinforcement-learning
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** RL with **several agents learning at once** — and the hard part is that each agent's environment now includes other agents who are *also* changing.
> **Metaphor:** Learning to play football while your teammates are *also* still learning. The "correct" pass changes every week.
> **Where it bites:** Non-stationarity, credit assignment across agents, and the CTDE pattern that most practical methods use.

---
- **What it is:** This is the general field. Instead of a single agent learning in an environment, MARL deals with coordinating a _team_ of multiple agents to perform a _shared task_.

- **The Challenge:** The environment is no longer stationary from one agent's perspective, because the other agents are also learning and changing their behaviors. This makes learning a stable, optimal _team_ strategy difficult.

---
# Why it's not just "RL, but more robots"

A single-agent [[Markov Decision Process|MDP]] rests on one quiet assumption: the transition function $P(s' \mid s, a)$ is **fixed**. The world may be random, but it's random in a *consistent* way, and that consistency is what lets value estimates converge.

Put a second learning agent in it and that assumption dies. From agent A's point of view, the "environment" includes agent B — and B's policy is changing every episode.

> [!WARNING] Non-stationarity — the central problem
> Agent A learns "when I'm here, passing left works." That was true when B stood on the left. Two thousand episodes later B has learned to run right, and A's hard-won knowledge is now **wrong**.
>
> Both agents are chasing a target the other one keeps moving. Convergence guarantees from single-agent RL simply **do not carry over**. ^non-stationarity

Two more problems come with it:

**Multi-agent credit assignment.** The team wins. Which agent's actions caused it? [[Markov Decision Process#The one hard part — delayed reward|Single-agent RL]] already struggles to assign credit across *time*; now you must also assign it across *agents*. An agent that did nothing useful still gets the team reward — this is **lazy agent** behaviour, and it's a real, commonly observed failure. 😴

**Partial observability.** Each agent usually sees only its own local view, not the global state. That's formalised as a [[Dec-POMDP]].

---
# CTDE — the pattern that makes it work

The dominant practical answer is **Centralised Training, Decentralised Execution**:

> [!SUCCESS] The core trade
> **During training**, be generous — let the learning algorithm see the global state, every agent's observation, everyone's actions. Training happens offline, in a simulator, where you control everything.
>
> **During execution**, be strict — each agent acts on its own local observation alone. Which is the only thing that will be available in the real deployment. 🤖 ^ctde

This works because the extra information is used to *train* a component that is then **thrown away**. A centralised critic learns accurate values by seeing everything; the decentralised policies it produces need only local input.

It fixes non-stationarity too: a critic conditioned on *everyone's* actions sees a stationary problem again, because there's no longer a hidden changing factor — it's all in the input.

> [!TIP] You've seen this trick elsewhere
> Giving a training-only component privileged information it won't have at deployment is exactly the [[Distillation|teacher-student]] pattern, and exactly what a privileged critic does in LLM RL. Valid whenever the component using the extra information doesn't survive to inference. ^privileged-info-pattern

---
# The families

| Approach | Idea | Problem |
|---|---|---|
| **Independent learners (IQL)** | Every agent runs its own DQN, ignores the others | Simple, sometimes works, no convergence guarantee |
| **[[Value Function Factorization\|Value factorization]]** (VDN, QMIX) | Learn team $Q^{tot}$, decompose into per-agent $Q^a$ | The main line — see that note |
| **Centralised critic** (MADDPG, MAPPO) | Shared critic sees everything, per-agent actors don't | Critic input grows with agent count |
| **Communication learning** | Agents learn *what to send* each other | Powerful, but needs a channel at deployment |

And a distinction always worth clarifying early in any MARL conversation:

- **Cooperative** — shared reward, one team. Most applied work.
- **Competitive** — zero-sum. Self-play, AlphaGo, poker.
- **Mixed** — some shared, some conflicting interests. Hardest, and closest to reality.

The competitive case brings its own trap: an agent trained only against its current opponent can **overfit to that opponent**, so self-play systems keep a *population* of past versions to play against. Same instinct as [[Mode Collapse|avoiding collapse]] — preserve diversity or the arms race degenerates.

---
# ⁉️
The dominant cooperative approach is decomposing one team value into per-agent values — and the constraint that makes that decomposition safe is → [[Value Function Factorization]].
