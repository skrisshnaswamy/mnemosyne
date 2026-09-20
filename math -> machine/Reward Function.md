---
aliases:
  - Reward
  - Rewards
  - Reward Signal
  - Reward Design
  - Reward Shaping
  - Potential-Based Shaping
  - Sparse Reward
  - Dense Reward
  - Reward Hypothesis
  - Cost Function
tags:
  - reinforcement-learning
  - decision-sciences
  - fundamentals
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A single number per step that says **what you want — never how to get it.** It is the entire specification of the task, so it's where most RL projects quietly go wrong.
> **Metaphor:** Paying a contractor. On completion only (honest, no guidance), by the hour (they'll bill hours), or by milestones that are genuinely on the path.
> **Where it bites:** Sparse rewards that never arrive, "helpful" bonuses that get farmed, and the realisation that *you* are the one who wrote the loophole.

---
A robot in a maze. You give it **+1 for reaching the exit** and nothing otherwise. Perfectly honest — that *is* the task.

It wanders for 150 steps and stumbles out by luck. It wanders for 190 steps and stumbles out again. For the first thirty-odd attempts it's essentially a drunk walk, because until it has reached the exit at least once, **every action it has ever taken scored exactly zero**. There's nothing to learn *from*.

So you help. *"+0.2 every time you take a step that gets you closer to the exit."* Surely that can only speed things up?

~={blue}What do you think it learns to do?=~

---
# Paying the contractor 🔨

It learns to walk **towards** the exit, then **away**, then **towards** again — collecting +0.2 on every approach, forever, and never leaving. You paid by the hour. It bills hours.

![[reward_shaping_learning_curves.png]]
> [!TIP] Reading the chart
> **Grey** — the honest sparse reward: ~35 episodes of flailing before it clicks, then near-optimal (14 steps is the shortest path). **Green** — shaping done properly: near-optimal in about **5** episodes. **Red** — the naive bonus: still averaging ~95 steps at the end. It isn't failing to learn. It learned *exactly what you paid for.*

> [!NOTE] Reward function
> A function $R(s, a, s')$ returning one scalar per step. The agent's objective is the (discounted) **sum** of these — the *return*. In control theory the same object with the sign flipped is a **cost** — the $q$ and $r$ in [[LQR]]. ^reward-def

> [!SUCCESS] Core idea
> ~={pink}The reward says **what**, not **how**.=~ It is the only channel through which you tell the agent what you want, which makes it a *specification* — and every gap between the spec and your intent is a gap a good optimiser will find. That's [[Reward Hacking#^optimiser-as-adversary|reward hacking]], and it's the same disease as [[Loss, Objectives, and Business Alignment|a loss that isn't what the business wanted]]. ^reward-is-the-spec

---
# Sparse, dense, shaped

| | Example | Good | Bad |
|---|---|---|---|
| **Sparse** | +1 at the exit, 0 elsewhere | impossible to misread — it *is* the goal | no signal until you succeed by luck; hopeless in big spaces |
| **Dense (hand-made)** | + for getting closer, − for bumping walls… | learning signal every step | ~={red}every term is a loophole=~ |
| **Potential-based shaping** | $F = \gamma\,\Phi(s') - \Phi(s)$ | dense **and** provably leaves the best policy unchanged | you need a sensible $\Phi$ |

**Why the third one is safe.** $\Phi(s)$ is a "how promising is this place" score — say, higher nearer the exit. You're paid for *gaining* potential and charged for *losing* it. Go towards the exit and back again, and the two cancel. Around any loop, the bonuses sum to zero — so there's nothing to farm. (Ng, Harada & Russell, 1999.)

> [!WARNING] The detail that cost me an experiment
> My first potential was $\Phi = -0.1 \times \text{distance}$ — perfectly valid in theory. It learned **slower than no shaping at all** (45 steps vs 16 by the end).
>
> Why: when the robot bumps a wall and stays put, the bonus is $\gamma\Phi - \Phi = (\gamma - 1)\Phi$. With a *negative* $\Phi$ that's a small **positive** payment — for standing still. Switching to a potential that's zero at the start and **rises** toward the goal fixed it. ~={blue}The optimal policy is protected by the theorem; the *learning speed* is not.=~ ^shaping-sign-gotcha

---
# Design rules worth having 🛠️

> [!TIP] In roughly this order
> 1. **Reward the outcome you want, not the behaviour you expect.** "Win the game", not "take pieces".
> 2. **Charge for time.** A small negative reward per step stops an agent that has found a safe corner from sitting there forever — see [[Markov Decision Process#Pac-Man, framed properly 👾|Pac-Man's living penalty]].
> 3. **If you must shape, use a potential.** Anything else is a loophole with your name on it.
> 4. **Watch the behaviour, not the score.** A rising reward curve proves the agent got better *at the reward*.
> 5. **Mind the scale.** Rewards of 1,000 and rewards of 0.001 both train badly; normalise.

---
# When you *can't* write it down

For "drive comfortably", "write a helpful answer" or "be a good summary", there is no formula. Three escapes:

| | Idea | Note |
|---|---|---|
| Copy an expert instead | skip the reward entirely | [[Imitation Learning]] |
| Infer it from watching experts | behaviour → reward | [[Inverse Reinforcement Learning]] |
| Learn it from human comparisons | "A or B?" → a reward model | [[Preference Learning]] → [[RLHF]] |

…and where a program *can* check the answer — tests pass, the maths is right — use that: a **verifiable** reward → [[GRPO]].

> [!WARNING] The reward hypothesis is a claim, not a fact
> *"Every goal can be expressed as maximising the expected sum of one scalar signal"* (Sutton) is the founding assumption of RL. It's remarkably productive. It also sweeps safety limits, competing objectives and fairness into one number with hand-picked weights — and then the weights *are* the design. Same trap as choosing $Q$ and $R$ in [[LQR#^optimal-for-your-cost|LQR]].

---
> [!SUCCESS] If you remember one thing
> The agent does what you **pay** for, not what you **meant**. ~={pink}If the behaviour is wrong, read the reward function before you touch the algorithm.=~

---
# ⁉️
Rewards arrive over time — some now, some a hundred steps from now. Is a reward tomorrow worth the same as one today? The agent needs a rule for that, and it turns out to be a single number.

→ [[Discount Factor]]
