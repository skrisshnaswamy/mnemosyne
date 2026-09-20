---
aliases:
  - NUTS
tags:
  - NUTS
  - probability
  - bayesian
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** [[Hamiltonian Monte Carlo|HMC]] that **tunes its own trajectory length** — it keeps simulating until the path starts doubling back, then stops.
> **Metaphor:** Walking a corridor until you notice you're heading back toward the door, then stopping. No need to be told the length in advance.
> **Where it bites:** It's the default sampler in **Stan** and **PyMC**. When someone says "I ran a Bayesian model", it was almost certainly this.

---
# The one problem it solves

[[Hamiltonian Monte Carlo|HMC]] has two knobs, and $L$ — the number of leapfrog steps — is genuinely painful:

- Too **small** → short trajectories, you're back to a slow correlated walk
- Too **large** → the puck curves round and comes back near its start. All those gradient evaluations, zero distance gained ↩️

Worse: there is no single correct $L$. A posterior with a narrow neck and a broad basin needs *different* trajectory lengths in different regions. Any fixed $L$ is wrong somewhere.

> [!SUCCESS] The insight
> Don't specify a length. **Keep simulating, and stop when the trajectory begins to turn back on itself.** ^nuts-core

---
# How it detects the U-turn

Track the vector from the start of the trajectory to the current position, and compare it against the current momentum (direction of travel).

While their dot product is **positive**, you're still moving away from where you started — keep going. The moment it goes **negative**, you've begun heading back. Stop. ✋

That's the "no U-turn" criterion, and it's checked automatically at every step. No tuning.

> [!NOTE] The doubling trick
> A subtlety: naively stopping the instant you detect a U-turn breaks **reversibility**, which MCMC needs for correctness — the chain would no longer converge to the right distribution.
>
> NUTS fixes this by building the trajectory as a **binary tree**, doubling its length each iteration and expanding randomly forwards or backwards in time. It then samples a point from the whole tree. This keeps the procedure reversible while still adapting the length. It's the fiddly part of the paper, and the reason NUTS is something you *use* rather than implement. ^nuts-doubling

The step size $\epsilon$ is handled too — tuned automatically during warmup (dual averaging) to hit a target acceptance rate, typically ~0.8. So in practice **both** HMC knobs disappear.

---
# What you actually do with it

```python
with pm.Model() as model:
    mu = pm.Normal("mu", 0, 1)
    pm.Normal("obs", mu=mu, sigma=1, observed=data)
    trace = pm.sample(2000)   # ← NUTS, by default
```

You specify the *model*. The sampler tunes itself.

> [!WARNING] The diagnostics still matter
> Self-tuning is not self-validating. Always check:
> - **Divergences** — non-zero means the geometry defeated the sampler somewhere. The usual culprit is a hierarchical funnel, and the usual fix is a **non-centred reparameterisation** (rewrite $\theta \sim N(\mu, \sigma)$ as $\theta = \mu + \sigma z,\; z \sim N(0,1)$), which flattens the funnel into something navigable.
> - **$\hat{R}$** close to 1.00 — chains agree
> - **ESS** in the hundreds at least — see [[Markov Chain Monte Carlo#The practical realities|the MCMC diagnostics]]
>
> Divergences in particular are **not** cosmetic. They mean part of your posterior was never explored, so your credible intervals are wrong. ^nuts-diagnostics

> [!TIP] The lineage in one line
> **Metropolis-Hastings** — blind random walk 🚶
> **[[Hamiltonian Monte Carlo|HMC]]** — use the gradient, follow the geometry 🎱
> **NUTS** — same, but stop when you'd start doubling back ↩️
>
> Each step removed a way for the previous method to waste your compute.

---
# ⁉️
Sampling gives you a full posterior — the honest, complete answer. It's also slow, and doesn't scale to a million parameters. The alternative is to give up on exactness and *approximate* the posterior with a simpler distribution, by minimising [[KL Divergence]] to it. That's variational inference, and it's the trade every large Bayesian model ends up making.
