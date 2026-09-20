---
aliases:
  - Backward Induction
  - Principle of Optimality
  - Optimal Substructure
  - Overlapping Subproblems
  - Memoization
  - Memoisation
  - Tabulation
tags:
  - planning
  - decision-sciences
  - fundamentals
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Solve a long sequence of decisions by **starting at the end and working backwards**, writing down *"the best you can do from here"* at each point — so no sub-problem is ever solved twice.
> **Metaphor:** Signposts. Put a sign at every junction saying *"office: 12 min (best case)"*. Once the signs exist, driving is trivial.
> **Where it bites:** Every "optimise over a sequence" problem — shortest paths, Viterbi, edit distance, inventory planning — and the machinery underneath all of RL.

---
You're leaving home for work. First junction: the motorway slip road, or the quiet back road?

- motorway leg: **5 minutes**
- back road leg: **8 minutes**

Five beats eight. You take the motorway. Thirty-seven minutes later you arrive, late, having sat in a car park that used to be Junction 2. The back road would have taken **twenty**.

Nothing unlucky happened. You compared the wrong thing — *"how long is this next bit?"* when the question was *"how long until I'm at my desk?"* That's acting **greedily**, and [[Decision Sciences#^greedy|it fails exactly when consequences are delayed]].

Fine. So you resolve to check every route properly. ~={blue}How many routes is that?=~

---
# Signposts 🪧

Here's the whole map:

![[dp_commute_graph.png]]

This one's tiny — four routes. But a real commute with 10 junctions and 3 choices at each has $3^{10} = 59{,}049$ routes. A delivery round with 30 stops: forget it. Every extra decision **multiplies** the work.

So don't enumerate. **Forget the start. Go and stand at the end.**

**At C, D and E** there's nothing to decide — one road each into the office. Write the time on a signpost: `C: 22` · `D: 8` · `E: 9`.

**Step back to B.** Two options, and the signposts ahead have already done the hard part:
$$\min(\,4 + \underbrace{8}_{D},\;\; 6 + \underbrace{9}_{E}\,) = 12$$
Signpost at B: `12`.

**And A:** $\min(10 + 22,\; 25 + 8) = 32$. Signpost: `32`.

**Finally home:** $\min(5 + 32,\; 8 + 12) = \mathbf{20}$ → take the back road.

Now notice what you *didn't* do. At B you never looked past D or E. You never traced a single complete route. You read ~={blue}one number=~ off each neighbour. And junction **D** — reachable from both A and B — was solved **once** and used twice.

> [!NOTE] Dynamic programming
> A method for sequential optimisation: break the problem into stages, solve the *last* stage first, and store each sub-problem's best value so every earlier stage can simply look it up. Needs two properties: **optimal substructure** and **overlapping sub-problems**. ^dp-def

> [!SUCCESS] Core idea
> ~={pink}Whatever you did to get here, the best thing to do *from* here is the same.=~ That's Bellman's **principle of optimality**, and it's what lets a signpost exist at all: the number at D doesn't care whether you arrived via A or B. The future is baked into the neighbour's number — so you think **one step**, then read. ^principle-of-optimality

---
# The two ingredients

| Ingredient | In the commute | Without it… |
|---|---|---|
| **Optimal substructure** — the best whole route is made of best sub-routes | the best way home→office via B *contains* the best way B→office | the signposts would depend on how you arrived, and the method collapses |
| **Overlapping sub-problems** — the same sub-problem is reached many ways | D is reached from both A and B | there'd be nothing to reuse; it's just brute force with extra steps |

You've met this in coding interviews wearing different clothes:

| Problem | The "signpost" being cached |
|---|---|
| Fibonacci with memoisation | `fib(n)` |
| Edit distance | cost to align the first $i$ and $j$ characters |
| [[Hidden Markov Model#^sum-vs-max\|Viterbi]] | best path probability ending in each state |
| Shortest path (Bellman–Ford) | distance to the target |
| An [[Markov Decision Process\|MDP]] | the **[[Value Function\|value]]** of each state |

**Memoisation** (top-down: recurse, cache as you go) and **tabulation** (bottom-up: fill the table from the end) are the same idea walked in opposite directions.

---
# Two words worth having ready

The signposts are the **[[Value Function|value function]]**. *"Which way do I turn at this junction?"* is the **[[Policy|policy]]**. And the rule you kept applying — *value of here = cost of one step + value of where it lands you, taking the best option* — is the **[[Bellman Equation]]**.

> [!TIP] About that name
> It has nothing to do with computer programming. In the 1950s "programming" meant *planning* — as in a TV programme, or "linear programming". Bellman picked a name that sounded impressive and hard to object to. ~={blue}Dynamic programming = planning, over time, by working backwards.=~ The full story is in [[Decision Sciences#A name that confused me|the chapter]].

---
# What it costs — and what it needs

> [!WARNING] Two big asks
> 1. **The whole map.** Every road, every travel time, *before you set off*. Fine for a chessboard or a warehouse. Useless for a robot in a building it's never seen, or an ad system meeting a user it's never met. Remove the map and you need [[Reinforcement Learning]].
> 2. **A map small enough to label.** DP visits *every state*. Add a few variables and the number of states explodes — Bellman named this himself: the [[Curse of Dimensionality]]. ^dp-needs

---
> [!SUCCESS] If you remember one thing
> Start at the end. Label each place with *the best you can do from here*. ~={pink}Then the hard question "what's the best sequence?" becomes the easy one "which neighbour has the best number?"=~

---
# ⁉️
That one-line rule — *value of here = one step + value of there* — has been doing all the work, and we haven't looked at it properly. It deserves its own note, because every algorithm in RL is a different way of enforcing it.

→ [[Bellman Equation]]
