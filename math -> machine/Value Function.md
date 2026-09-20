---
aliases:
  - Value Functions
  - State Value
  - State-Value Function
  - Action Value
  - Action-Value Function
  - Q-Function
  - Q Function
  - Q-Value
  - Q-Values
  - Advantage Function
  - Advantage
  - V-Function
tags:
  - reinforcement-learning
  - decision-sciences
  - fundamentals
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A **prediction of the total future reward** from here. $V(s)$ scores a situation, $Q(s,a)$ scores a move, and the **advantage** $A = Q - V$ says how much better that move is than your usual.
> **Metaphor:** The evaluation bar on a chess stream. *+1.3* isn't the material on the board — it's a forecast of how this ends, assuming good play from here.
> **Where it bites:** The critic in actor-critic, the Q in DQN, the advantage in PPO, the value head in AlphaZero. It's the quantity nearly every RL method estimates.

---
You're watching a chess stream. White has just given up a knight for nothing — a whole piece down.

And the evaluation bar at the side of the screen swings **towards White**. +2.1.

The commentators aren't confused. *"Oh, that's lovely. Black's king has nowhere to go."* Twelve moves later it's checkmate.

So that bar clearly isn't counting pieces. ~={blue}What *is* it measuring?=~

---
# The evaluation bar ♟️

![[chess_eval_bar_value.png]]

It's a **forecast**: *how is this game going to end, from this position, if both sides play well from here?* A knight is a fact about the present. The bar is a statement about the future.

That's a value function. And notice the small print — *"if both sides play well"*. Hand White's position to a beginner and its true value is far lower, because they won't find the mating attack. **The value of a position depends on who's going to be playing it.**

> [!NOTE] Value function
> $V^\pi(s)$ is the expected [[Discount Factor|return]] from state $s$ if you follow policy $\pi$ from then on. It's always [[Markov Decision Process#^v-is-policy-dependent|value *under a policy*]] — change the policy and every number changes. ^value-def

| | You get it from | Timescale | Who decides it |
|---|---|---|---|
| **Reward** $r$ | the environment, this step | immediate | the designer — it's *given* |
| **Return** $G$ | adding up what actually happened | one whole episode | luck + policy — it's *observed* |
| **Value** $V$ | your own estimate | the expected long run | you — it's *predicted* |

> [!SUCCESS] Core idea
> ~={pink}Reward is what you just got. Value is what you expect to get in total.=~ A value function is a **cache of the future** — the whole branching tree of what might happen, compressed to one number per state, so that deciding only needs one step of lookahead. That's the [[Dynamic Programming|signpost]], and [[Bellman Equation]] is the rule that keeps the signposts consistent. ^value-is-a-cache

---
# V, Q and A — three questions

Back to the shop in `Low` stock, with the numbers from [[Bellman Equation]]: doing nothing is worth 12, ordering is worth 40. Add a third option, discounting, worth 8.

| | Asks | In the shop |
|---|---|---|
| **$V(s)$** | how good is it to be **here**? | depends on how you behave — see below |
| **$Q(s,a)$** | how good is doing **this**, from here? | nothing **12** · order **40** · discount **8** |
| **$A(s,a) = Q - V$** | how much better is this than **what I'd normally do**? | ↓ |

If your current habit is to pick at random, $V(\text{Low}) = \frac{12 + 40 + 8}{3} = 20$, and the advantages are:

$$A(\text{order}) = +20 \qquad A(\text{nothing}) = -8 \qquad A(\text{discount}) = -12$$

If you always order, $V(\text{Low}) = 40$ and $A(\text{order}) = 0$ — *no better than usual, because it **is** the usual*.

> [!TIP] Why each one exists
> - **$V$** needs a model to act on: to choose a move you have to ask "where would each action take me?" — which requires knowing $P$.
> - **$Q$** doesn't: just take $\arg\max_a Q(s,a)$. That's why model-free control learns $Q$ → [[Q-Learning]], and why [[Markov Decision Process#^max-vs-argmax|once you have Q, the policy is free]].
> - **$A$** strips out *"this state is just good / bad in general"* and leaves only *"was **this action** a good choice?"* — a much cleaner learning signal. It's what [[Policy Gradient]] methods and [[PPO]] actually train on.

---
# The picture to keep

![[value_heatmap_policy_arrows.png]]
> [!TIP] Reading the chart
> The numbers are $V$; the arrows are the greedy policy. Every arrow just points at **the neighbour with the biggest number**. Values fall off by $\gamma = 0.9$ per step away from the goal, and they dip around the pit without anyone telling the agent "avoid the pit" — the danger is simply priced in. How the numbers got there → [[Value and Policy Iteration]].

---
# How you get one

| Method | Uses | Trade-off |
|---|---|---|
| [[Dynamic Programming]] | the known model | exact; needs $P$, $R$ and a small state space |
| [[Monte Carlo Methods\|Monte Carlo]] | average of complete observed returns | unbiased, noisy, waits for the episode to end |
| [[Temporal Difference Learning\|TD learning]] | one step of reward + your *own* estimate of the next state | learns every step, low variance, a bit biased |
| [[Function Approximation]] | a network instead of a table | scales to huge state spaces; can be unstable |

> [!WARNING] A value function is a *belief*, and it can be wrong
> It's your estimate, learned from limited experience. States you've rarely visited have values that are basically guesses — and [[Q-Learning#^maximisation-bias|taking a max over noisy guesses is biased upward]]. A confident-looking number is not evidence. ^value-is-an-estimate

---
# Where it shows up 🔗

- **[[Actor-Critic]]** — the *critic* is a value function.
- **[[DQN]]** — a network that outputs $Q(s, \cdot)$ for every action.
- **[[PPO]] / [[RLHF]]** — a "value model" predicts the reward-to-go so advantages can be computed.
- **[[Monte Carlo Tree Search|AlphaZero]]** — the value head is *literally* the chess evaluation bar.
- **[[LQR]]** — $V(x) = x^\top P x$, a quadratic bowl.

---
> [!SUCCESS] If you remember one thing
> **Reward** = what just happened. **Value** = everything you expect to happen, summed. ~={pink}Almost all of RL is estimating that second number well enough to act on it.=~

---
# ⁉️
Fine — but where do the numbers *come from*? If you know the rules of the world, there are two classic ways to compute them, and between them they describe the rhythm of nearly every RL algorithm.

→ [[Value and Policy Iteration]]
