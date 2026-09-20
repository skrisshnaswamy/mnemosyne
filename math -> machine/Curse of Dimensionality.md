---
aliases:
  - Curse of Dimensions
  - High-Dimensional Spaces
  - Combinatorial Explosion
  - State Space Explosion
  - State Explosion
tags:
  - fundamentals
  - planning
  - decision-sciences
  - probability
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Every variable you add **multiplies** the space instead of adding to it — so tables become impossible to fill, data becomes hopelessly sparse, and "nearest neighbour" stops meaning anything.
> **Metaphor:** Looking for your keys — along a corridor, then across a field, then through every floor of a tower block.
> **Where it bites:** Tabular RL and DP, grid search, kNN and vector search, particle filters, MCMC. It's the reason function approximation exists.

---
A warehouse robot. You want to plan with [[Dynamic Programming]], so you need a table with one row per state.

- Its position along an aisle, in 10 cells → **10** rows.
- Add the other axis → **100**.
- Add which way it's facing (10 headings) → **1,000**.
- Battery level (10), carrying a box or not (2) → **20,000**.
- Oh — and there are three other robots to avoid, each somewhere on that 100-cell floor → × 100 × 100 × 100 →

$$\mathbf{20{,}000{,}000{,}000} \text{ rows.}$$

Nothing exotic happened. You described a small robot in a small room with eight ordinary variables. ~={blue}At a million table updates per second, how long to visit each row *once*?=~

---
# Looking for your keys 🔑

Five and a half hours. And DP needs *many* sweeps. And you need a row for every **action** too.

Lose your keys in a **corridor** and you walk its length: 10 paces. Lose them in a **field** and it's $10 \times 10$. In a **ten-storey building**, $10 \times 10 \times 10$. Each new dimension doesn't add ten more places to look — it multiplies everything you already had by ten.

![[curse_of_dimensionality.png]]
> [!TIP] Reading the chart
> **Left:** ten levels per variable. By 9 variables you're past a billion rows. **Right:** the second face of the curse. Scatter 500 random points in a $d$-dimensional box and compare the *nearest* one with the *farthest* one. In 2-D the nearest is **50× closer** (ratio 0.02). In 100-D the ratio is **0.71**. In 1,000-D, **0.90** — ~={red}everything is roughly equally far from everything.=~

> [!NOTE] Curse of dimensionality
> The exponential growth of volume — and therefore of states to enumerate, or data needed to cover the space — as the number of dimensions increases. Coined by Richard Bellman in 1957, about exactly the DP problem above. ^curse-def

> [!SUCCESS] Core idea
> It has **two faces**, and they bite different people:
> 1. **Combinatorial** — you can't *enumerate* it. Kills tables, grids, exhaustive search.
> 2. **Geometric** — you can't *cover* it. High-dimensional space is almost entirely empty, your data is a few specks in it, and distance loses its meaning.
>
> ~={pink}Both come from the same fact: volume grows as $\text{side}^{d}$.=~ ^two-faces

---
# The geometric face, made concrete

Take a unit cube and shave off a 5% skin from every face, leaving an inner cube of side 0.9. How much of the volume is still "inside"?

$$0.9^d: \qquad d = 2 \rightarrow 81\% \qquad d = 10 \rightarrow 35\% \qquad d = 100 \rightarrow 0.003\%$$

In 100 dimensions, **99.997%** of the volume is in that thin outer skin. Every point is an outlier along *some* axis. So a model trained on samples from that space is nearly always extrapolating, and any method that relies on *"find me similar examples"* has almost none to find.

---
# Where it bites 🪤

| Victim | How |
|---|---|
| Tabular [[Dynamic Programming\|DP]] / [[Q-Learning]] | one row per state — and most rows are never visited even once |
| Grid search over hyperparameters | 5 values × 8 hyperparameters = 390,625 runs → use random search or [[Bayesian Optimization]] |
| [[Particle Filter]] | particles needed grow exponentially with state dimension |
| [[Markov Chain Monte Carlo]] | see [[Markov Chain Monte Carlo#^curse-of-dimensionality-mcmc\|there]] |
| kNN, [[Vector Database\|vector search]] | distances concentrate; exact search degenerates to a scan |
| Binned statistics / histograms | almost every bin is empty |

---
# The ways out

Nobody beats the curse. They dodge it, by refusing to treat every point as unrelated to every other:

| Escape | The bet it makes |
|---|---|
| **[[Function Approximation]]** | nearby states have similar values, so *generalise* rather than tabulate — the move that made deep RL possible |
| **Sampling** ([[Monte Carlo Methods]]) | a Monte Carlo average's error shrinks as $1/\sqrt{N}$ **regardless of dimension** — you pay for variance, not volume |
| **Exploiting structure** | the other robots barely interact → factor the problem ([[Value Function Factorization]]) |
| **Learned low-dimensional representations** | real data lives near a thin manifold inside the big space → [[Embeddings]] |
| **Plan only from where you are** | don't solve every state, just the ones reachable from here → [[Model Predictive Control]], [[Monte Carlo Tree Search]] |

> [!WARNING] "More features can't hurt"
> They can. Every feature you add thins out your data in the space the model has to cover. With a fixed dataset, there's a point past which extra dimensions make the model *worse* — unless they carry real signal, or something ([[Regularization]], a good prior, an architecture with the right structure) stops the model paying attention to the empty directions. ^more-features-can-hurt

---
> [!SUCCESS] If you remember one thing
> Dimensions **multiply**. ~={pink}Anything that needs to enumerate a space, or fill it with data, stops working after a handful of them=~ — so every method that scales is, underneath, a bet about *structure* that lets it avoid doing so.

---
# ⁉️
That sounds fatal for planning. But there's one famous case where the state is continuous — *infinitely* many states — and DP can still be solved exactly, on paper, in a few lines.

→ [[LQR]]
