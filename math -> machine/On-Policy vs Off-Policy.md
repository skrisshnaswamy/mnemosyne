---
aliases:
  - On-Policy
  - Off-Policy
  - On Policy
  - Off Policy
  - On-Policy Learning
  - Off-Policy Learning
  - SARSA
  - Expected SARSA
  - Behaviour Policy
  - Behavior Policy
  - Target Policy
  - Importance Sampling
  - Importance Weights
tags:
  - reinforcement-learning
  - decision-sciences
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** **On-policy** learns about the policy that is *generating the data*. **Off-policy** learns about a *different* one — usually the greedy one — from data produced by something else.
> **Metaphor:** Learning to drive from your own driving, versus from someone else's dashcam footage.
> **Where it bites:** Whether you can reuse old data at all. SARSA vs Q-learning, PPO vs DQN/SAC, and the importance ratio at the heart of PPO.

---
A small gridworld. You start bottom-left, the goal is bottom-right. Between them, along the bottom edge, is a **cliff**: step on it and you get −100 and go back to the start. Every ordinary step costs −1.

The shortest route hugs the cliff edge. The safe route goes up and round the top.

Two agents learn here, both exploring the same way — 10% of the time, a random move. Their update rules differ by **one term**:

- **Q-learning** bootstraps from the *best* action in the next state.
- **SARSA** bootstraps from the action it *actually takes* next.

~={blue}Which route does each end up preferring — and which collects more reward?=~

---
# The cliff 🧗

![[cliff_walking_sarsa_vs_q.png]]
> [!TIP] Reading the chart
> **Q-learning** (orange) learns the shortest path — right along the edge. **SARSA** (blue) learns to go the long way round. And yet SARSA earns about **−28** per episode while Q-learning earns about **−50**. The agent that found the *optimal* path scores **worse**.

Here's why. Q-learning is learning the value of the **perfect, wobble-free** policy — and for that policy the cliff edge really is best. But the agent *executing* it still takes a random step one time in ten, and next to a cliff, that's fatal on a regular basis.

SARSA learns the value of **what it actually does, wobbles included**. Walking the edge while 10% random really is a bad idea — so its values say so, and it moves away.

Neither is wrong. They're answering different questions.

> [!NOTE] Behaviour policy vs target policy
> The **behaviour policy** generates the experience. The **target policy** is the one being evaluated and improved. **On-policy:** they're the same. **Off-policy:** they differ. ^behaviour-vs-target

| | **SARSA** (on-policy) | **[[Q-Learning]]** (off-policy) |
|---|---|---|
| Target | $r + \gamma\, Q(s', a')$ — $a'$ is what I'll **really do** | $r + \gamma \max_{a'} Q(s', a')$ — what the **best** move would be |
| Learns the value of | the exploring policy it's following | the optimal greedy policy |
| While training | safer, better online reward | riskier |
| After you switch exploration off | slightly sub-optimal | optimal |

(The name is just the tuple it uses: $S, A, R, S', A'$.)

> [!SUCCESS] Core idea
> ~={pink}On-policy asks "how good is what I'm doing?". Off-policy asks "how good would something *else* be, judging from what I did?"=~ The second question is far more useful — and far more dangerous, because you're reasoning about actions you may never have taken. ^on-vs-off-core

---
# Why off-policy is worth the trouble

Off-policy is what lets you learn from data **you didn't generate right now, with this policy**:

- a **replay buffer** of your own older experience → [[Experience Replay]]
- a human's **demonstrations**
- years of **production logs** from the previous system → [[Offline RL]], [[Off-Policy Evaluation]]
- one exploratory policy's data used to evaluate *many* candidate policies

An on-policy method must **throw its data away after every update**, because the policy that produced it no longer exists. That's the whole reason on-policy methods are sample-hungry.

| | On-policy | Off-policy |
|---|---|---|
| Reuse old data | ❌ | ✅ |
| Sample efficiency | poor | good |
| Stability | **good** | fragile with neural nets — part of the [[Function Approximation#^deadly-triad\|deadly triad]] |
| Examples | SARSA, [[Policy Gradient\|REINFORCE]], A2C, [[PPO]]* | [[Q-Learning]], [[DQN]], DDPG, [[SAC]], [[DPO]] |

\* PPO re-uses each batch for a few epochs, which makes it *slightly* off-policy — and that's exactly what its clipped ratio polices.

---
# The correction — importance sampling

If the data came from policy $\mu$ but you care about $\pi$, re-weight every sample by how much more (or less) often $\pi$ would have done the same thing:

$$\rho = \frac{\pi(a \mid s)}{\mu(a \mid s)}$$

$\mu$ picked an action 10% of the time and $\pi$ would pick it 50% of the time? Count that sample **5×**. Over a multi-step trajectory the ratios *multiply*, so the variance explodes with length — the central headache of off-policy learning.

> [!TIP] You'll meet this ratio again, twice
> - In [[PPO]] it's $r_t(\theta) = \frac{\pi_\text{new}}{\pi_\text{old}}$ — and the famous **clip** is a guard against it wandering far from 1.
> - In [[Off-Policy Evaluation]] it's called the **inverse propensity weight**, and it's how you answer "what would the new recommender have earned?" from the old one's logs.

---
# The mix-up to avoid

> [!WARNING] Off-policy ≠ offline
> **On/off-policy** is about *whose behaviour* the data reflects. **Online/offline** is about *whether you can collect more*.
>
> | | can still interact | fixed dataset only |
> |---|---|---|
> | data from the current policy | on-policy, online (PPO) | — |
> | data from another policy | off-policy, online (DQN + replay) | **[[Offline RL]]** |
>
> Every offline method is off-policy. Most off-policy methods are *not* built for offline use — they quietly rely on being able to go and try the action they're unsure about. ^off-policy-is-not-offline

And in LLM-land: [[RLHF]] with PPO is (nearly) on-policy — it samples fresh answers from the current model every round. [[DPO#^dpo-off-policy|DPO is off-policy]] — it learns from a fixed set of preference pairs that some *earlier* model wrote. That one difference accounts for most of their practical trade-offs.

---
> [!SUCCESS] If you remember one thing
> **On-policy** = "how good is what *I'm* doing?" — stable, wasteful. **Off-policy** = "how good would *that* be, given what someone did?" — efficient, fragile. ~={pink}The cliff shows the first can be the wiser question even though the second finds the "optimal" answer.=~

---
# ⁉️
Every method so far goes the long way round: learn values, *then* read a policy off them. If the policy is what you actually want, why not adjust it directly?

→ [[Policy Gradient]]
