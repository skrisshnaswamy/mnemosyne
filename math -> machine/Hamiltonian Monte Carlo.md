---
aliases:
  - HMC
  - Hybrid Monte Carlo
tags:
  - HMC
  - probability
  - bayesian
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** [[Markov Chain Monte Carlo|MCMC]] that uses **gradients** — flick a frictionless puck across the landscape instead of taking blind random steps.
> **Metaphor:** Rolling a ball on a curved surface. Physics carries it a long way along a sensible path, rather than you guessing directions.
> **Where it bites:** The engine inside Stan and PyMC. Needs a differentiable density — which is exactly why it doesn't work on discrete parameters.

---
# The fix — stop walking blind

Plain [[Markov Chain Monte Carlo#^curse-of-dimensionality-mcmc|Metropolis-Hastings]] fails in high dimensions because a randomly chosen direction is almost always a bad one.

But we're not actually blind. For most models we can compute $\nabla \log P(\theta)$ — the [[Derivative#Gradient|gradient]] of the log-density. We know which way is uphill. It seems mad to throw that away and guess.

HMC uses it, via a genuinely elegant idea borrowed from physics.

---
# The physics trick 🎱

Flip the landscape upside down, so high-probability regions become **valleys** rather than peaks. Now imagine a frictionless puck sliding around on that surface.

Give the puck a **random kick** in some direction (this is the "momentum" — an auxiliary variable HMC invents purely to make this work). Then let physics run for a while. The puck:
- accelerates as it slides down into valleys
- decelerates climbing out
- **naturally follows the contours** of the landscape rather than cutting across them

Then stop it, record the position as a sample, kick it randomly again, repeat.

> [!SUCCESS] Core idea
> Because the puck follows the terrain's actual geometry, a single trajectory can travel **very far** while staying in high-probability regions. So the next sample is distant and nearly uncorrelated — instead of the tiny, correlated shuffle a random walk gives you. ^hmc-core

The random kick is what keeps it *sampling* rather than *optimising* — each kick gives it a different energy level, so it explores the whole valley system instead of settling at the bottom.

> [!NOTE] Where "Hamiltonian" comes from
> In physics a Hamiltonian is total energy: **potential** (position — here, the negative log-posterior) plus **kinetic** (momentum). Frictionless motion **conserves** it. That conservation is what makes the accept/reject step almost always accept — the proposal is already well-matched to the distribution by construction. ^hamiltonian-def

---
# What actually goes on in the code

Simulating physics on a computer is approximate, and the details matter because they're the knobs you'll be tuning:

**Leapfrog integrator.** Alternately update momentum and position in small steps. It's used instead of a plain Euler step because it's *symplectic* — it conserves energy over long trajectories rather than drifting, which is exactly the property HMC depends on.

**The Metropolis correction.** Discretisation means energy isn't perfectly conserved, so there's still an accept/reject step at the end. But acceptance is typically **>90%**, versus the ~25% you'd tune a random walk to.

Two knobs:
- **$\epsilon$ (step size)** — too big and the simulation goes unstable and everything is rejected; too small and it's slow.
- **$L$ (number of steps)** — this is the awkward one. Too few and you barely move. Too many and ~={red}the puck curves right back to where it started=~ — you burned a hundred gradient evaluations to go nowhere. A **U-turn.** ↩️

> [!WARNING] What HMC can't do
> It needs $\nabla \log P(\theta)$, so the parameter space must be **continuous and differentiable**. Discrete parameters (a cluster assignment, a model-selection index) have no gradient. Standard practice is to marginalise them out analytically, or fall back to Gibbs sampling for those components. ^hmc-needs-gradients

Also: **divergences**. When the geometry has a region of extreme curvature (a funnel — classic in hierarchical models), the leapfrog simulation blows up. Good implementations report these, and a divergence is a real warning that your posterior has a shape the sampler cannot navigate — not just noise to ignore.

---
# ⁉️
HMC is a huge improvement, but choosing $L$ by hand is miserable — and the *right* $L$ differs at different points in the same posterior. What if the sampler could just notice when it's about to double back, and stop there?

→ [[No U-Turn Sampler]]
