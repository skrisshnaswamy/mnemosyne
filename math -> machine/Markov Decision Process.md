---
aliases:
  - MDP
tags:
  - reinforcement-learning
  - fundamentals
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** The standard grammar for *"decisions, made repeatedly, where your choices change what happens next."* Five pieces: **states, actions, transitions, rewards, discount**.
> **Metaphor:** A board game. You're on a square, you have legal moves, moves take you somewhere new, some squares score points.
> **Where it bites:** It is the formal frame underneath **all** of reinforcement learning. Every RL paper assumes you have this vocabulary.

---
# What it's for

The [[Markov Property]] gave us processes that evolve on their own. An MDP adds the thing that makes it interesting: ~={blue}**you get to intervene.**=~

It's the formalism for sequential decision-making — where a choice now changes the situation you face later, so you can't just optimise each step independently.

> [!NOTE] The five pieces — $(S, A, P, R, \gamma)$
> - **$S$ — States.** Every situation you could be in.
> - **$A$ — Actions.** What you may do in each state.
> - **$P(s' \mid s, a)$ — Transitions.** Take action $a$ in state $s$ → land in $s'$ with some probability. *Probabilistic*, because the world is noisy.
> - **$R(s, a)$ — Reward.** The immediate scalar feedback.
> - **$\gamma$ — Discount factor.** How much future reward is worth compared to reward now. Between 0 and 1. ^mdp-tuple

The name carries the key assumption: transitions obey the [[Markov Property]]. $P$ depends only on the current $(s, a)$, never on how you arrived. **This is what makes the problem solvable** — otherwise every decision would require reasoning over the entire history.

---
# The store — building an MDP from scratch 🏪

You manage a small shop. Let's build this up piece by piece.

**States** — how much stock you have: `Low`, `Medium`, `High`.

**Transitions** — each day customers buy things, so you drift downward. `High` → `Medium` with some probability, and so on. This much is just a [[Markov Property|Markov chain]] — the shop drifting on its own.

But you're not a spectator. You can **do** things.

**Actions** — `Order more inventory`, `Do nothing`, `Offer a discount`.

**Rewards** — profit from sales, minus a penalty for running out of stock (lost customers), minus holding costs.

**Policy** — the rule you end up with. *"If stock is Low, order more. If High, discount."* That's your inventory strategy.

> [!SUCCESS] The moment it stops being a chain
> A Markov chain **describes** a system. Add someone who can act, and you're now **optimising** it. That's the entire difference — and it's why the store, the moment you're managing it rather than watching it, is a **Markov Decision Process**. ^chain-vs-mdp

## The bit I got wrong first time

I assumed actions and transitions were separate things — that ordering stock *influenced* supply somehow, but didn't directly cause a move from one state to another.

Here's the correction, and it's worth sitting with.

**Actions don't determine the next state. They choose which set of odds you're playing with.**

You're in `Low Inventory`:

| Action you take | → Low | → Medium | → High |
|---|---|---|---|
| **Order more** | 0.0 | 0.8 | 0.2 |
| **Do nothing** | 0.9 | 0.1 | 0.0 |
| **Discount** | 0.95 | 0.05 | 0.0 |

Look at what changed. Not the *outcome* — the **whole row of probabilities**.

> [!TIP] The slot machine 🎰
> Your action is like choosing **which lever to pull**. The lever decides *which set of outcomes you're gambling on*. But the actual result is still drawn at random from that set.
>
> ~={blue}You don't pick where you land. You pick the distribution you land from.=~ ^actions-choose-the-distribution

That's why the transition function is written $P(s' \mid s, a)$ — *"probability of $s'$, **given** I was in $s$ **and** chose $a$."* The action is a condition on the odds, not a command.

## Pac-Man, framed properly 👾

| Piece | In Pac-Man |
|---|---|
| **State** | Snapshot of the game — your position, every ghost's position and direction, remaining dots and pellets |
| **Actions** | Up, Down, Left, Right |
| **Transitions** | Mostly predictable, but ghosts move semi-randomly — so it's stochastic |
| **Rewards** | Dot +10 · pellet +50 · ghost +200 · **eaten: large penalty** · small penalty per step to discourage dawdling |
| **Policy** | "If a ghost is two squares right and I'm near a pellet, go left" |

> [!NOTE] Who is the decision-maker?
> I initially guessed **the game developer** — someone tuning ghost difficulty. Reasonable, but no.
>
> The decision-maker is whoever chooses the **actions inside the world**: the player (or an AI agent playing). The developer set up the *rules* — they wrote $P$ and $R$. They're not playing the MDP, they **defined** it. ^who-is-the-decision-maker

That "living penalty" is worth noticing too — a small negative reward each step. Without it, an agent that's found a safe corner may just sit there forever, technically not losing. You have to **pay for time** to make it want to finish. 🕰️

# The one hard part — delayed reward

If reward were immediate, this would be trivial: try each action, keep the best. Sequential decision-making is hard for exactly one reason.

> [!TIP] Chess again ♟️
> You sacrifice your queen. Immediate reward: **terrible.** Twelve moves later: checkmate.
>
> So was the sacrifice a good move? Obviously yes — but the reward that proves it arrives twelve steps after the decision. How does an agent connect them?

That's the **credit assignment problem**, and it's the whole difficulty. (Note the shape is identical to [[Backpropagation|backprop]]'s blame assignment — one across *layers*, one across *time*.)

The machinery below exists to solve it.

---
# Value — the central idea

You can't judge an action by its immediate reward. So instead you ask: **how good is it to be here, accounting for everything that follows?**

$$V^\pi(s) = \mathbb{E}\left[\, r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 r_{t+3} + \cdots \,\right]$$

Each step further into the future is multiplied by another $\gamma$, so influence decays geometrically. (Exactly the EWMA shape as [[Momentum#What $\beta$ actually means|momentum's $\beta$]] — same maths, different job.)

- **$V^\pi(s)$ — state value.** How good is this state, if I follow policy $\pi$ from here?
- **$Q^\pi(s,a)$ — action value.** How good is taking action $a$ here, *then* following $\pi$? ← this is the useful one, because $\arg\max_a Q$ **is** your best action
- **$\pi(a \mid s)$ — the policy.** Your strategy. The thing you're actually solving for.

> [!NOTE] What $\gamma$ actually encodes
> - $\gamma = 0$ → completely myopic; only immediate reward exists
> - $\gamma = 0.99$ → effective horizon of roughly $\frac{1}{1-\gamma} = 100$ steps
> - $\gamma \to 1$ → far-sighted, but the sum may not converge on infinite horizons
>
> It's part discounting-preference, part mathematical necessity (it guarantees the infinite sum is finite), and part **variance control** — distant rewards are noisy, and $\gamma$ quietly stops them dominating. ^gamma-meaning

---
## V, Q and π — three things that are easy to blur

I tangled these up, so here they are side by side.

| | What it takes in | What it gives back | Plain English |
|---|---|---|---|
| **$V(s)$** value function | a state | **one number** | "How good is it to be *here*?" |
| **$Q(s,a)$** action-value | a state **and** an action | **one number** | "How good is it to do *this*, from here?" |
| **$\pi(s)$** policy | a state | **an action** | "What should I actually *do*?" |

Two clarifications that cost me some confusion:

> [!WARNING] $V$ is not action-free
> It looks like $V$ ignores actions — it only takes a state. But its *value* depends entirely on what you'd do next. The value of a chess position isn't a property of the board; it's ~={blue}how good this board is **assuming you go on to play well**.=~
>
> $V$ is always $V^\pi$ — value **under a policy**. Change the policy and every number changes. ^v-is-policy-dependent

> [!WARNING] A policy is not a number
> $V$ and $Q$ each return a single scalar. A **policy is a mapping** — a rule, a lookup table, a function. Ask it about a state and it hands you back an *action*, not a score. Different type of object entirely. ^policy-is-a-mapping

## max vs argmax 🏙️

Worth pinning down, because $V^*$ and $\pi^*$ differ by exactly this.

Take the list `[10, 5, 20, 15]`:
- **max** → `20` — the biggest **value**
- **argmax** → `2` — the **position** that produced it

Or, a city skyline:
- **max**: *"The tallest building is 500 feet."* 📏
- **argmax**: *"The tallest building is at 123 Main Street."* 📍

And now the two definitions read themselves:

$$V^*(s) = \max_a Q^*(s,a) \qquad\qquad \pi^*(s) = \arg\max_a Q^*(s,a)$$

> [!SUCCESS] Same maths, different question
> **max** tells you *how good the best option is*. **argmax** tells you *what to do*.
> That's why the optimal value uses `max` and the optimal policy uses `argmax` — and why ~={pink}once you have $Q$, the policy is free.=~ Just take the argmax. This is the whole reason $Q$-learning is more directly useful than learning $V$. ^max-vs-argmax

# The Bellman equation

This is the single most important equation in RL, and it's really just a bookkeeping observation:

$$V^\pi(s) = \mathbb{E}\big[\, r + \gamma V^\pi(s') \,\big]$$

**The value of here = the reward I get now + the discounted value of wherever I land next.**

Which is obvious once stated — but it's *recursive*, and that's what makes it powerful. You've turned "sum an infinite future" into "one step, plus a value you already have an estimate for." Now you can iterate: guess all the values, apply the rule everywhere, repeat. It provably converges. 🎯

> [!SUCCESS] Core idea
> Bellman turns an infinite-horizon problem into a **local update**. That single move is the foundation of value iteration, Q-learning, DQN, actor-critic — essentially every algorithm in the field. ^bellman-is-everything

---
> [!NOTE] Is an MDP a kind of dynamic programming?
> No — and the distinction is clean:
> - An **MDP** is a **framework**. It's how you *describe* the problem.
> - **[[Decision Sciences#And now the words|Dynamic programming]]** is a family of **algorithms**. It's how you *solve* one.
>
> **Value iteration** and **policy iteration** are the two classic DP algorithms for solving an MDP. Both work by applying the Bellman equation over and over until the numbers stop changing. ~={blue}MDP = the problem. DP = one way to solve it.=~ ^mdp-vs-dp

> [!NOTE] Discrete vs continuous
> - **Discrete** — countable. Chess squares, inventory counts, Up/Down/Left/Right.
> - **Continuous** — real-valued. Room temperature, a robot's exact coordinates, a steering angle, a throttle setting.
>
> This distinction decides which methods are even available to you. You can put discrete states in a **table**. You cannot table a steering angle — there are infinitely many. That single fact is what eventually forces neural networks into the picture. ^discrete-vs-continuous

# The fork in the road — do you know $P$ and $R$?

This is the distinction worth having ready, because it splits the entire field in two.

| | **You know the rules** | **You don't** |
|---|---|---|
| Called | Planning / Dynamic Programming | **Reinforcement Learning** |
| Methods | Value iteration, policy iteration | Q-learning, PPO, SAC |
| How you learn | Compute directly from $P$ and $R$ | **Try things and observe** |
| Example | A board game with published rules | A robot in a real room |

Classic MDP theory assumes you have $P$ and $R$ in hand. Real problems almost never do — nobody hands you the transition probabilities of a warehouse. So RL is ~={pink}"solve the MDP while simultaneously discovering what the MDP even is."=~

And that's precisely where [[Uncertainty#Exploration vs exploitation|exploration vs exploitation]] appears: you must sometimes take an action you *believe* is suboptimal, purely to learn whether that belief is right.

---
# The family tree

MDPs are the middle of a spectrum. Knowing where the neighbours sit is genuinely useful in conversation:

| | States observable? | Actions? | Multiple agents? |
|---|---|---|---|
| **Markov Chain** | ✅ | ❌ none | one |
| **Bandit** | (single state) | ✅ | one |
| **MDP** | ✅ | ✅ | one |
| **POMDP** | ❌ hidden | ✅ | one |
| **[[Dec-POMDP]]** | ❌ hidden | ✅ | **many** |

A **bandit** is an MDP with one state — your action doesn't change the situation, so there's no credit assignment, only exploration. That's why bandits are so much easier and why they split off as their own field; see [[Decision Sciences]].

A **POMDP** is what you get when you can't see the state. The fix is to act on your [[Beliefs#Belief states: when the state itself is hidden|belief state]] instead — which converts it back into an MDP over beliefs.

---
> [!SUCCESS] If you remember one thing
> An MDP is $(S, A, P, R, \gamma)$ plus the Bellman recursion. **Everything in RL is either estimating $V$/$Q$, or improving $\pi$, or both.**

---
# ⁉️
The Bellman equation needs expectations over $P$ — and when the state space is huge or continuous, those expectations are integrals nobody can compute. The general escape hatch is to stop computing and start **sampling**.

→ [[Markov Chain Monte Carlo]]

Next in the [[Control and Reinforcement Learning]] chain — the MDP's vocabulary, one piece at a time, starting with the thing you're actually solving for → [[Policy]]
