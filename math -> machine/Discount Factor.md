---
aliases:
  - Discount Rate
  - Discounting
  - Gamma
  - Return
  - Discounted Return
  - Cumulative Reward
  - Horizon
  - Effective Horizon
  - Planning Horizon
  - Episodic Task
  - Continuing Task
tags:
  - reinforcement-learning
  - decision-sciences
  - fundamentals
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** One number, $\gamma$ between 0 and 1, that says how much a reward *next* step is worth compared with one *now*. The agent's effective foresight is about $\dfrac{1}{1-\gamma}$ steps.
> **Metaphor:** The marshmallow test. One now, or two if you can wait? $\gamma$ is the agent's patience.
> **Where it bites:** It isn't a tuning knob — it **changes which problem is being solved**. Also: slow convergence near 1, and why RLHF usually just sets it to 1.

---
I'll give you **£100 today**, or **£100 in a year**. Which?

Today. Everyone says today. Now try to say *why* — there are at least three different reasons hiding in that instinct:

1. **It grows.** £100 today can be £105 by next year.
2. **You might not collect.** I could vanish; you could move abroad; a lot can happen in a year.
3. **The promise is fuzzier.** "A year from now" has more ways to go wrong than "now".

An agent collecting rewards over time has the same three concerns. ~={blue}How do you give a machine that preference — without writing three separate rules?=~

---
# The marshmallow test 🍡

With one number. Every step further into the future, multiply the reward's worth by $\gamma$:

$$G = r_0 + \gamma\, r_1 + \gamma^2 r_2 + \gamma^3 r_3 + \cdots$$

$G$ is the **return** — the thing the agent is actually maximising. Put numbers in:

**A reward of 100 that arrives 50 steps from now** is worth, today:

$$\gamma = 0.9:\;\; 100 \times 0.9^{50} = \mathbf{0.52} \qquad\qquad \gamma = 0.99:\;\; 100 \times 0.99^{50} = \mathbf{60.5}$$

Same reward. One agent barely registers it. The other will cross the room for it.

**A reward of 1 every step, forever:**

$$1 + \gamma + \gamma^2 + \cdots = \frac{1}{1 - \gamma} \qquad \Rightarrow \quad \gamma = 0.9 \rightarrow 10, \qquad \gamma = 0.99 \rightarrow 100$$

That $\frac{1}{1-\gamma}$ is the **effective horizon** — roughly how many steps ahead the agent can "see".

![[discount_curves.png]]
> [!TIP] Reading the chart
> Note the log axis. $\gamma = 0.9$ has forgotten everything beyond ~10 steps. $\gamma = 0.999$ is still listening a thousand steps out. If the thing you care about happens 500 steps after the decision that caused it, an agent with $\gamma = 0.9$ **literally cannot care** — the reward reaches it multiplied by $10^{-23}$.

> [!NOTE] Discount factor and return
> The **return** $G_t = \sum_k \gamma^k r_{t+k}$ is the discounted sum of future rewards. The **discount factor** $\gamma \in [0, 1]$ sets how fast future rewards lose weight. $\gamma = 0$: only the next reward exists (completely myopic). $\gamma \to 1$: all rewards count equally. ^discount-def

> [!SUCCESS] Core idea
> $\gamma$ does **three jobs at once** — see [[Markov Decision Process#^gamma-meaning|the MDP note]]:
> 1. **Preference** — sooner is better.
> 2. **Survival** — read $\gamma$ as *"the chance the episode is still going next step"* and the formula is just an expected total. That's reason 2 above, exactly.
> 3. **Maths and noise** — it keeps an infinite sum finite, and it turns down the volume on far-off rewards, which are the noisiest ones.
>
> ~={pink}It's a preference, a survival probability and a variance control, in one symbol.=~ ^three-jobs

---
# Choosing it

| $\gamma$ | Horizon | Behaviour | Use for |
|---|---|---|---|
| 0 | 1 step | pure greed | it's a [[Multi-Armed Bandit\|bandit]] — nothing carries over |
| 0.9 | ~10 | short-sighted, learns fast | quick reflex tasks |
| 0.99 | ~100 | the common default | games, control |
| 0.999 | ~1,000 | far-sighted, slow, noisy | long-horizon strategy |
| 1 | the whole episode | no discounting | **short, finite episodes only** — e.g. RLHF, where the reward comes once at the end |

> [!TIP] The rule of thumb
> Set the horizon $\frac{1}{1-\gamma}$ to **a bit longer than the delay between a decision and its consequences**. Shorter and the agent can't see why the decision mattered. Much longer and you're only adding noise.

The price of patience is real. The [[Bellman Equation#Why just *iterating* it works|Bellman backup]] shrinks error by a factor of $\gamma$ per sweep — 12 sweeps at $\gamma = 0.5$, 85 at 0.9, around a thousand at 0.99. And long horizons mean high-variance returns, which is what [[Temporal Difference Learning]] and [[Actor-Critic|GAE]] exist to tame.

---
# Episodic vs continuing

- **Episodic** — there's a natural end: a game, a conversation, a delivery. You *can* use $\gamma = 1$.
- **Continuing** — it never ends: a thermostat, a data-centre scheduler. Then $\gamma < 1$ is what stops the return being infinite (or you switch to optimising *average* reward per step).

> [!WARNING] "$\gamma$ is just another hyperparameter"
> Tune the learning rate and you find the same answer faster or slower. Change $\gamma$ and you've changed **what counts as the right answer**. An agent at $\gamma = 0.9$ that ignores a payoff 100 steps away isn't under-trained — it's correctly solving the problem you gave it. ~={red}If behaviour looks short-sighted, check $\gamma$ before anything else.=~ ^gamma-changes-the-problem

> [!TIP] Same maths, different job
> $\gamma^k$ weights are an exponentially-weighted sum — the identical shape to [[Momentum]]'s $\beta$ and to an EWMA filter ([[Bayes Filter#^ewma-is-fixed-gain|here]]). One looks *back* over past gradients; this one looks *forward* over future rewards.

---
> [!SUCCESS] If you remember one thing
> **Horizon ≈ $\frac{1}{1-\gamma}$.** ~={pink}An agent cannot want something it has discounted to nothing=~ — so $\gamma$ isn't how well it plans, it's how far it's *allowed* to.

---
# ⁉️
So "how good is this situation?" means *the discounted sum of everything that follows from it*. That quantity is what every RL algorithm is secretly trying to estimate.

→ [[Value Function]]
