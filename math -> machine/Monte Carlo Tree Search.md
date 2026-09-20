---
aliases:
  - MCTS
  - UCT
  - Tree Search
  - AlphaGo
  - AlphaZero
  - AlphaGo Zero
  - MuZero
  - Decision-Time Planning
  - Self-Play
  - Rollout Policy
tags:
  - planning
  - deep-rl
  - reinforcement-learning
  - decision-sciences
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** At each move, **grow a lopsided search tree from the current position** — deeper where it looks promising, shallow elsewhere — using bandit statistics to decide where to look next. Then play the most-explored move.
> **Metaphor:** A strong chess player thinking. Not every line — a few candidate moves, and the promising ones followed much further.
> **Where it bites:** AlphaGo / AlphaZero / MuZero. And the ancestor of "think longer at inference time" for LLMs.

---
Go. About **250** legal moves each turn, and a game lasts around **150** turns.

[[Dynamic Programming]] would need a value for every position — more positions than atoms in the universe, by a ludicrous margin. Brute-force lookahead branches 250 ways per ply. And unlike chess, nobody ever managed to hand-write a decent "who's winning?" function for Go — for decades that was *the* obstacle.

In March 2016, AlphaGo beat Lee Sedol 4–1.

~={blue}It didn't enumerate the game. So how do you search a tree that size?=~

---
# Candidate moves 🌳

Watch a strong player think. They don't examine all 250 moves. They glance at the board, pick **three or four candidates**, and follow those. One starts to look bad — abandoned after two moves. Another looks promising — followed twelve moves deep. The tree in their head is **wildly lopsided**, and that lopsidedness *is* the skill.

MCTS builds exactly that tree, by repeating four steps thousands of times:

![[mcts_four_phases.png]]

| Phase | What happens |
|---|---|
| **1 · Selection** | from the root, walk down the existing tree. At each node pick the child with the best [[Upper Confidence Bound\|UCB]] score — *win rate so far + a bonus for being under-explored* |
| **2 · Expansion** | reach the edge of what's been explored → add one new node |
| **3 · Simulation** | estimate how good that new position is. *Classically:* play random moves to the end (a **rollout**) and see who wins. *AlphaZero:* ask a [[Value Function\|value network]] |
| **4 · Backpropagation** | carry that result back up the path, updating the visit count and win rate of every node on the way. (Nothing to do with [[Backpropagation\|gradient backprop]] — same word, different idea.) |

When the thinking time is up, **play the move you visited most**.

> [!NOTE] Monte Carlo Tree Search
> A best-first search that incrementally builds a tree, using sampled evaluations to estimate node values and a bandit rule (**UCT** — UCB applied to Trees) to balance exploring new branches against deepening good ones. ^mcts-def

> [!SUCCESS] Core idea
> ~={pink}Every node in the tree is a little [[Multi-Armed Bandit|bandit]]=~, and its children are the arms. "Which move should I think about next?" is [[Exploration vs Exploitation]] — so solve it with [[Upper Confidence Bound|UCB]], *recursively*. Thinking effort flows toward what's promising **or** still uncertain, and drains away from what's clearly bad. That's how a search of a few thousand nodes can play well in a tree of $10^{170}$. ^every-node-is-a-bandit

---
# AlphaGo → AlphaZero → MuZero

Plain MCTS with random rollouts plays *amateur* Go. The leap was to plug two networks into it:

- A **policy network** $p(a \mid s)$ → a prior over which moves are worth considering *at all*. It prunes 250 moves down to a handful.
- A **value network** $v(s)$ → *"who's winning here?"* — the evaluation function nobody could hand-write. It replaces the random rollout.

Then the loop that needs no human games at all:

> **Search makes the network's policy better. The network is trained to imitate the search. So next time the search starts from a better network — and gets better again.**

MCTS acts as a **policy-improvement operator**; training the net to match it is the **evaluation/distillation** step. It's [[Value and Policy Iteration#^gpi|generalised policy iteration]] — played out at the scale of a data centre.

| System | Human games? | Knows the rules? |
|---|---|---|
| **AlphaGo** (2016) | yes — bootstrapped from them | yes → [[Mastering the game of Go with deep neural networks (AlphaGo)]] |
| **AlphaZero** (2017) | **no** — pure self-play | yes → [[Mastering Chess and Shogi by Self-Play (AlphaZero)]] |
| **MuZero** (2019) | no | **no** — it *learns* a latent model good enough to search in → [[Model-Based vs Model-Free RL]] |

---
# Reflex vs deliberation

| | A trained policy | MCTS |
|---|---|---|
| Compute spent | at **training** time | at **decision** time |
| Acting | one forward pass — instant | thousands of simulations — slow |
| Novel positions | whatever the net generalises to | **searches this exact position** — robust |
| Needs | lots of experience | a model (or simulator) to search in |
| More compute → | retrain | **just think longer**, right now |

That last row is the important one. It's the same trade as [[Model Predictive Control#The same shape, elsewhere 🔗|MPC]] — plan from *where you are*, act, re-plan — and the direct ancestor of [[Test-Time Compute#^capability-is-a-dial|test-time compute]] for LLMs: best-of-$N$, tree-of-thoughts, search over reasoning steps with a verifier. *Capability as something you can buy per request.* → [[TTPO- Test-Time Policy Optimization]]

> [!WARNING] "MCTS is brute force with a fast computer"
> The opposite. **Deep Blue** (1997) was brute force — ~200 million chess positions a second, with a hand-tuned evaluation. **AlphaZero** examined around 80,000 a second — *thousands of times fewer* — and played better, because a learned prior and value told it *where not to look*. MCTS is **selective sampling**. Its power is in everything it declines to search. ^mcts-is-selective

> [!TIP] Where it doesn't fit
> You need a **simulator or model** to search in, a **modest action set**, and ideally **some way to evaluate a position** short of playing to the end. Games satisfy all three. Most messy real-world problems give you none — which is why MCTS conquered board games years before it touched anything else.

---
> [!SUCCESS] If you remember one thing
> **Treat "what should I think about next?" as a bandit problem, at every node.** ~={pink}Add a network that knows where not to look, train that network on what the search finds — and the loop improves itself.=~

---
# ⁉️
MCTS gets to try anything it likes — in its imagination, for free, as often as it wants. Now take away the simulator *and* the ability to experiment. All you have is a hard drive full of what somebody else once did.

→ [[Offline RL]]
