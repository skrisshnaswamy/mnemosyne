---
aliases:
  - Memoryless Property
  - Markov Assumption
tags:
  - fundamentals
  - probability
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** The future depends **only on the present state**, not on the path you took to get there.
> **Metaphor:** A chess position. It doesn't matter how the pieces got there — the board *is* the state.
> **Where it bites:** It's the assumption that makes [[Markov Decision Process|MDPs]], [[Markov Chain Monte Carlo|MCMC]] and the [[Kalman Filter]] tractable. And it's usually the first assumption to break in the real world.

---
It's a rule that says the probability of a process to transition to a next state only and only depends on the current state and **not all the previous states before it** (past states)
It's called a **memoryless** property

$$P(S_{t+1} \mid S_t) = P(S_{t+1} \mid S_t, S_{t-1}, S_{t-2}, \ldots, S_0)$$

Read the equation as a claim: ~={blue}"knowing the entire history tells you nothing more than knowing where you are right now."=~

---
# The chess board ♟️

Two grandmasters sit down at a board mid-game. To decide the best next move, does it matter whether the position was reached by a Sicilian Defence or by a series of blunders?

**No.** The board *is* the state. Every legal future follows from the arrangement in front of you. The history is genuinely irrelevant — not ignored, but actually containing no extra predictive information.

That's the Markov property, and it's a very strong claim.

Now contrast: **poker**. The cards on the table are visible, but what your opponent did three hands ago tells you a lot about how they play. History carries information the current table doesn't. Poker is **not** Markov in the visible state.

> [!SUCCESS] Core idea
> Markov isn't a property of the *world* — it's a property of ~={pink}**how you defined the state**.=~ Bad state definition → not Markov. Rich enough state definition → Markov. ^markov-is-about-state-design

---
# Why this is a design decision, not a discovery

That last point is the one worth internalising, because it turns Markov from a restriction into a **tool**.

Say you're tracking a car and your state is *position only*. Is that Markov? No — a car at position $x$ heading north behaves completely differently from one at position $x$ heading south. Position alone doesn't determine the future.

So **redefine the state**: position *and* velocity. Now it is Markov (near enough). You didn't change the car. You changed the bookkeeping.

That's exactly the move the [[Kalman Filter]] makes — its state is *"position **and** velocity"* precisely so that the property holds.

The general trick: **if your process isn't Markov, widen the state until it is.**

| Non-Markov state | Fix |
|---|---|
| Position only | Add velocity |
| Today's price | Add the last $n$ days (a "window") |
| Current word | Add all previous words → **this is what a transformer's context window is** |
| Visible board in poker | Add [[Beliefs\|beliefs]] about the opponent → a POMDP belief state |

> [!TIP] A language model is a Markov process
> With state = *the entire context window*. Next-token prediction depends only on what's in the context, not on anything that scrolled out of it. Which reframes "context length" as ~={blue}"how much history we shoved into the state to keep the Markov assumption honest."=~ ^llm-is-markov

#### Pac-Man 👾

Here's the example I landed on when I first tried to think of one.

In Pac-Man, what do you actually need to decide your next move? Where the ghosts are, where the pellets are, what the path ahead looks like.

Does it matter whether you arrived here from the top-left corner of the maze or the bottom-right? **No.** The current screen tells you everything.

> [!TIP] A near-miss worth keeping
> My first instinct was "people who forget their roots when they come into money" — they act as if their past doesn't exist.
>
> But that's **not** a Markov process, and the reason why is the useful bit: a person's entire life history genuinely *does* shape their behaviour. It's still in there, influencing things, even if they'd rather it weren't. Markov isn't *"the past is forgotten"* — it's ~={blue}"the past has already finished doing all its work, and the result is sitting in front of you."=~ ^markov-near-miss

#### The easy maths — the transition matrix

Before any decision-maker shows up, a Markov chain has surprisingly simple maths.

Build a grid: **rows** are the state you're in now, **columns** are the state you go to next, and each cell is the probability of that move.

|  | → Sunny | → Rainy |
|---|---|---|
| **Sunny** | 0.8 | 0.2 |
| **Rainy** | 0.4 | 0.6 |

Every row must sum to **1** — the system has to end up *somewhere*.

That grid is the **transition matrix**, $T$. And here's the neat part: if you want to know where you'll probably be in **two** steps, you compute $T^2$. Three steps, $T^3$. In general:

$$\text{distribution after } n \text{ steps} = \pi_0 T^n$$

where $\pi_0$ is where you started.

> [!SUCCESS] What this maths can and can't do
> It's pure **prediction**. It tells you where the system will probably drift, all on its own.
>
> It has no notion of anyone *wanting* anything. There's nobody in this picture who can act. Add someone who can — and the whole thing changes character, from **prediction** to **optimisation**. That's the [[Markov Decision Process|MDP]]. ^chain-is-prediction-mdp-is-optimisation

> [!SUCCESS] The clearest way I've found to say it
> It's tempting to describe Markov as *"some processes are memoryless — the past doesn't matter."* That's misleading.
>
> Better: ~={pink}a process is Markov when the detail captured in the state is **self-sufficient** — enough to explain everything, so that how you got here becomes moot.=~
>
> If position alone tells you everything, it's already Markov. It doesn't, so we **make** it Markov by adding whatever the history was quietly telling us. ^markov-is-self-sufficiency

> [!TIP] It's compression 🗜️
> This works exactly like a compression algorithm: keep enough detail to reconstruct the behaviour completely, without storing every byte of what happened.
>
> The formal name for that is a **sufficient statistic** — *sufficient* meaning "once you have this, the raw history adds literally nothing." A Markov state is a sufficient statistic of the entire past. ^sufficient-statistic

> [!WARNING] And it isn't free
> Every time you widen the state to make it Markov, the state gets **bigger**. Position was one number; position + velocity is two; add acceleration and heading and it's four.
>
> Push that far enough and the state becomes so large you can no longer enumerate it — which is precisely the wall that ends tabular methods and forces neural networks into the story. See [[Markov Decision Process#^discrete-vs-continuous|discrete vs continuous]]. ^widening-has-a-cost

**Order** is the formal name for this: a first-order chain looks back one step, an $n$-th order chain looks back $n$. But an $n$-th order chain is just a first-order chain whose state is a tuple of the last $n$ things — so it's the same trick again.

---
# What it buys you, and what it costs

**Buys:** tractability, and lots of it. Without it, predicting step $t$ requires conditioning on all $t-1$ previous steps — a table that grows exponentially and can never be estimated. With it, you need one transition rule, reusable at every step. That single simplification is what makes [[Markov Decision Process|MDPs]], [[Markov Chain Monte Carlo|MCMC]], HMMs and the Kalman filter computable at all.

**Costs:** it's frequently false, and the failure is quiet.

> [!WARNING] Where it breaks in practice
> - **Non-stationarity** — the transition rule itself changes over time (user tastes drift, a market regime shifts)
> - **Hidden state** — something real is driving the process and isn't in your state vector. This is the common one, and it's what POMDPs exist for
> - **Long-range dependencies** — something 500 steps ago genuinely matters and fell out of your window
>
> The symptom is usually a model that validates beautifully offline and degrades in production, because offline your fixed state captured enough and online it doesn't. ^markov-failure-modes

---
# ⁉️
If the future depends only on the current state, then choosing an **action** in that state is a well-posed problem — you don't need to reason about the whole history to decide. That framing is the [[Markov Decision Process]].
