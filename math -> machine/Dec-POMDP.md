---
aliases:
  - Decentralized POMDP
tags:
  - Decentralized-Partially-Observable-Markov-Decision-Process
  - MARL
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** An [[Markov Decision Process|MDP]] where the state is **hidden** *and* there are **several agents**, each seeing only their own slice.
> **Metaphor:** A team of divers in murky water, sharing one mission, no radio. Each sees a metre ahead and must act.
> **Where it bites:** It's the formal setting for cooperative [[Multi-Agent Reinforcement Learning|MARL]]. Provably intractable in general — which is *why* CTDE and approximation exist.

---
- **What it is:** This is the formal mathematical framework for MARL problems.
- **Decentralized:** Multiple agents making decisions.
- **Partially Observable:** Each agent only has a _partial view_ of the environment, not the complete global state.

- **Coach Player MARL paper's Extension:** The paper builds on a "Dec-POMDP with entities", which is a way to represent the world. They further extend it to include agent "characteristics" (like skill-level) to handle _heterogeneous_ agents (agents with different capabilities).

---
# Where it sits

Each word is one step away from the plain MDP, and it's worth seeing them as a ladder:

| | Hidden state? | Multiple agents? |
|---|---|---|
| **[[Markov Decision Process\|MDP]]** | ❌ | one |
| **POMDP** | ✅ | one |
| **Dec-POMDP** | ✅ | **many** ⬅️ |

Formally $(S, \{A_i\}, P, R, \{\Omega_i\}, O, \gamma)$ — the MDP tuple plus a set of observations $\Omega_i$ per agent and an observation function $O$ giving what each agent perceives. Note the **single shared** reward $R$: this is the *cooperative* setting by definition. Competing agents need a different formalism (a stochastic game).

---
# Why it's genuinely hard

> [!WARNING] It's NEXP-complete
> Solving a Dec-POMDP optimally is **NEXP-complete** — worse than NP-hard, and doubly exponential in the horizon. Even tiny instances (two agents, a handful of states) are infeasible to solve exactly.
>
> This isn't a footnote. Everything practical in MARL — CTDE, [[Value Function Factorization|value factorization]], centralised critics — exists **because** the exact problem is hopeless and we need approximations that work anyway. ^nexp-complete

The difficulty is a specific thing, and it's worth being able to name it:

**Nested beliefs.** Each agent holds [[Beliefs|beliefs]] about the hidden state. But its best action also depends on what *the other agents* will do — which depends on **their** beliefs. So agent A must reason about B's beliefs, including B's beliefs about A's beliefs, and so on. That recursion has no natural base case. 🪆

> [!SUCCESS] Core idea
> A POMDP is hard because you can't see the state. A Dec-POMDP is *dramatically* harder because you can't see the state **and** you can't see what your teammates believe about the state — and you must coordinate anyway, without communicating. ^why-decpomdp-hard

The other twist: there's no shared belief to act on. In a single-agent POMDP you convert to an MDP over [[Beliefs#Belief states: when the state itself is hidden|belief states]] and you're done. Here every agent has a *different* belief, and none of them is the belief the team would collectively hold. That escape hatch is closed.

---
# How people actually cope

Since exact solutions are out, applied MARL leans on approximations:

- **[[Multi-Agent Reinforcement Learning#CTDE — the pattern that makes it work|CTDE]]** — cheat during training with global information, stay decentralised at execution. The dominant approach.
- **[[Value Function Factorization]]** — impose a monotonic structure so greedy local action-selection is provably team-optimal. Trades expressiveness for tractability.
- **Learned communication** — let agents transmit messages, partially reconstructing shared state. Requires a channel at deployment.
- **Parameter sharing** — one network for all agents (with an agent ID input). Massively reduces what must be learned; only valid when agents are interchangeable.

That last point is where the **entities and characteristics** extension in the Coach-Player paper matters: plain parameter sharing assumes **homogeneous** agents. Adding an explicit characteristic vector (skill level, capability) lets one shared network condition its behaviour on *what kind of agent it currently is* — keeping the efficiency of sharing while supporting a **heterogeneous** team. The "coach" then acts as the centralised component distributing strategic information periodically, which is CTDE with a limited, deliberate communication budget rather than none at all.

---
# ⁉️
Practical approaches all route through [[Multi-Agent Reinforcement Learning]] and [[Value Function Factorization]]. The single-agent version of the hidden-state problem is in [[Beliefs]].
