---
aliases:
  - QMIX
  - VDN
  - Value Decomposition
tags:
  - MARL
  - reinforcement-learning
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Learn **one team value**, then split it into per-agent values so each agent can act alone — with a constraint guaranteeing the split stays honest.
> **Metaphor:** A team bonus divided so that no individual can raise their own share by hurting the team.
> **Where it bites:** VDN, QMIX, and the **monotonicity constraint** — the thing everyone asks about.

---
- **What it is:** A popular [[Multi-Agent Reinforcement Learning#CTDE — the pattern that makes it work|CTDE]] technique. It's the foundation that A-QMIX and COPA build upon.
- **The Idea:** It learns the team's total Q-value (the expected future reward, $Q^{tot}$) by "factorizing" it into _individual_ Q-values ($Q^a$) for each agent.

---
# The dilemma it resolves

You're stuck between two things you both need:

- **Training** wants a **team** value. The reward is a team reward; that's what you actually care about.
- **Execution** needs **individual** values. At deployment each agent is alone with its own observation, and must choose an action without consulting anyone.

So you need to learn one number and act on many. Naively decomposing is dangerous — you could easily end up in a situation where an agent maximising its own $Q^a$ actively **harms** $Q^{tot}$. Then decentralised execution silently produces bad team behaviour, while your training metrics look fine.

---
# The monotonicity constraint

The fix is a structural constraint on how the pieces combine:

$$\frac{\partial Q^{tot}}{\partial Q^{a}} \geq 0 \qquad \text{for every agent } a$$

In plain terms: ~={blue}an agent improving its own Q-value can **never** decrease the team's Q-value.=~

> [!SUCCESS] Why this is exactly the right constraint
> If $Q^{tot}$ is monotonically increasing in every $Q^a$, then the joint action that maximises $Q^{tot}$ is exactly the collection of actions that each maximise their own $Q^a$.
>
> $$\arg\max_{\mathbf{u}} Q^{tot} = \big( \arg\max_{u_1} Q^1, \; \ldots, \; \arg\max_{u_n} Q^n \big)$$
>
> So each agent can act **greedily and independently** and the team still lands on the optimal joint action. This is the **IGM** (Individual-Global-Max) property, and it's the whole point of the exercise. ✅ ^igm-property

That matches the conclusion your group reached: *the optimal action for each individual agent is also the best action for the team.* The monotonicity constraint is precisely what makes that statement true rather than merely hoped-for.

---
# VDN → QMIX

**VDN** takes the simplest possible option — just add them up:
$$Q^{tot} = \sum_a Q^a$$
Monotonic (every partial derivative is exactly 1) and trivially satisfies IGM. But a plain sum is very restrictive: it can't express *"these two agents are only valuable together"*, and it ignores the global state entirely.

**QMIX** generalises it. Instead of a sum, combine the $Q^a$ through a **mixing network** — and force monotonicity by constraining that network's weights to be **non-negative**.

> [!NOTE] The hypernetwork trick
> If the mixing weights must be non-negative, how does the global state influence anything?
>
> QMIX uses a **hypernetwork**: a separate network takes the global state and *generates* the mixing network's weights, passing them through an absolute value to keep them non-negative. So the global state shapes **how** the values combine, while monotonicity is preserved by construction rather than by a penalty term.
>
> Note the state enters the *mixer*, never the individual agent networks — the agents must stay decentralised. ^hypernetwork

---
# What it can't represent

> [!WARNING] The known limitation
> Monotonicity buys you IGM at a real cost: there are genuine coordination problems it **cannot** represent.
>
> The standard counterexample is a payoff matrix where both agents must pick action A *or* both must pick B — but mismatching is heavily punished. Here an agent's best action depends on what the other does, so no monotonic decomposition exists. QMIX will converge to a suboptimal joint policy and there is no amount of training that fixes it. ^qmix-limitation

This is why the follow-up literature exists — **QTRAN**, **QPLEX**, **weighted QMIX** all try to widen the representable class while keeping tractable decentralised execution. It's an active trade-off, not a solved problem.

| | Combining function | Expressiveness |
|---|---|---|
| **VDN** | sum | most restrictive |
| **QMIX** | monotonic mixing net, state-conditioned | broader |
| **QTRAN / QPLEX** | relaxed constraints + corrections | broader still, harder to train |

---
# ⁉️
All of this assumes agents can't see the global state — the formal setting is the [[Dec-POMDP]]. And the belief-tracking each agent must do inside it is [[Beliefs#Belief states: when the state itself is hidden|belief state]] reasoning.
