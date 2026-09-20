---
aliases:
  - Value Function Approximation
  - Function Approximator
  - Deadly Triad
  - The Deadly Triad
  - Generalisation in RL
  - Generalization in RL
  - Linear Function Approximation
  - Tabular Methods
  - Tabular RL
  - Semi-Gradient
tags:
  - reinforcement-learning
  - deep-rl
  - decision-sciences
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Replace the table — one cell per state — with a **function with shared parameters**, so that learning about one state teaches you about *similar* ones. It's what makes RL scale, and it's what makes it unstable.
> **Metaphor:** A club chess player in a position they've never seen. They don't look it up — they judge it by features: material, king safety, pawn structure.
> **Where it bites:** Every deep RL method. And the **deadly triad** — function approximation + bootstrapping + off-policy — which is why naive deep Q-learning diverges.

---
[[Q-Learning]] keeps one number per (state, action). Chess has about $10^{47}$ positions. The Earth has about $10^{50}$ atoms.

Storage is the *small* problem. To fill in a cell you have to **visit** it, several times. And after ten moves of any game, you're in a position that has never occurred in the history of chess. The row for it is blank. A blank row means: *I have never been here; I have no idea.*

And yet a decent club player, shown that same never-before-seen position, will tell you within seconds who's better.

~={blue}They've never seen it either. So what are they doing?=~

---
# The club player ♟️

Judging by **features**. *White's a pawn up. Black's king is exposed. That bishop is bad.* They've never seen **this** position, but they've seen thousands that share its features, and they carry a rough rule from features to "who's winning".

That's function approximation. Swap the lookup table $Q[s][a]$ for a function $Q(s, a;\, \theta)$ — a small vector of weights $\theta$, shared across **every** state.

![[function_approx_generalisation.png]]
> [!TIP] Reading the chart
> The same 22 visited states, both times. **Left, a table:** **34 of 50 cells are still blank** — it knows nothing about anywhere it hasn't stood. **Right, an approximator with 9 parameters:** a sensible value for *every* state, tracking the truth closely (RMS error 0.03) — including the wide gaps it never visited.

> [!NOTE] Function approximation
> Representing a value function or policy with a parameterised function — linear in hand-made features, or a neural network — instead of a table. Updating the parameters for one state necessarily changes the output for others: **generalisation**. ^function-approx-def

> [!SUCCESS] Core idea
> ~={pink}Generalisation is the blessing *and* the curse.=~ It's how 9 numbers can stand in for a million states — the only real escape from the [[Curse of Dimensionality]]. But it also means you can no longer fix one state's value without disturbing its neighbours. In a table, updates are independent. Here, **everything is coupled** — and that coupling is where the trouble starts. ^generalisation-blessing-and-curse

---
# Why it isn't just regression

Fitting a value function looks like [[Regression Analysis|regression]]: inputs are states, targets are returns. If the targets were real [[Monte Carlo Methods|Monte Carlo]] returns, it *would* be.

But the efficient methods are [[Temporal Difference Learning|TD]] — the target is $r + \gamma\, Q(s', \cdot\,; \theta)$. ~={red}The target contains the very weights you're adjusting.=~ Nudge $\theta$ to fix the prediction at $s$, and you've just moved the target you were aiming at.

It's a dog chasing its own tail. In control terms it's a [[Feedback Loop|feedback loop]] around your own learner, and like any loop it can be [[Stability|unstable]].

> [!WARNING] The deadly triad
> Any **two** of these are fine. All **three** together can make the values blow up to infinity:
> 1. **Function approximation** — updates leak into other states
> 2. **Bootstrapping** — targets built from your own estimates (TD, [[Dynamic Programming|DP]])
> 3. **Off-policy training** — learning about a policy other than the one generating the data
>
> Why the third matters: [[On-Policy vs Off-Policy|on-policy]], an over-estimated state gets *visited*, and visiting it corrects it. Off-policy, it may never be visited, so the error is free to feed on itself. — Sutton & Barto's name for it. ^deadly-triad

And [[Q-Learning]] with a neural network is **all three at once**. That's why "just plug a network into Q-learning" failed for twenty years.

---
# The ladder

| | Parameters | Guarantees | Used by |
|---|---|---|---|
| **Table** | one per (state, action) | converges ✅ | [[Q-Learning]], small [[Dynamic Programming\|DP]] |
| **Linear** in hand-made features | one per feature | converges on-policy; can diverge off-policy | tile coding; TD-Gammon's predecessors |
| **Neural network** | thousands → billions | none in general | [[DQN]], [[PPO]], [[SAC]], AlphaZero |

> [!TIP] It did work once, early — and nobody could repeat it
> **TD-Gammon** (Tesauro, 1992): a small neural net trained by TD self-play reached world-class backgammon. For two decades it stood nearly alone. Backgammon's dice randomise the states so thoroughly that the game explores *for* you, keeping the triad's third leg weak. Atari has no dice — and needed the engineering in the next note.

---
# How the field holds the triad down

| Trick | Which leg it weakens | Where |
|---|---|---|
| Freeze a copy of the network to compute targets | bootstrapping — the target stops moving | [[DQN]] |
| Replay old experience in random order | makes the data look more like a fixed dataset | [[Experience Replay]] |
| Take the **min** of two critics | over-estimation feeding itself | [[SAC]], TD3 |
| Limit how far each update may move the policy | the gain of the loop | [[PPO]] |
| Stay on-policy | removes the third leg entirely | [[Policy Gradient]], A2C |
| Penalise values for actions absent from the data | off-policy error with no way to self-correct | [[Offline RL]] |

> [!WARNING] "Just plug a neural net into Q-learning"
> The network was never the hard part. Everything that made deep RL *work* is a stabilisation trick — a frozen target, a replay buffer, a clip, a leash. ~={red}If a deep RL run diverges, suspect the triad before the architecture.=~ ^network-is-not-the-hard-part

---
> [!SUCCESS] If you remember one thing
> A table knows only where it's been. A function **guesses** everywhere — ~={pink}which is what lets RL scale, and what makes it a moving-target problem instead of ordinary regression.=~

---
# ⁉️
Between 2013 and 2015 a small team found the right pair of tricks, pointed one network at the raw pixels of 49 Atari games with the same settings for all of them, and it learned to play more than half of them at human level or better.

→ [[DQN]]
