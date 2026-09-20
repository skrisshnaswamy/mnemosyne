---
aliases:
  - Offline Reinforcement Learning
  - Batch RL
  - Batch Reinforcement Learning
  - Conservative Q-Learning
  - CQL
  - Extrapolation Error
  - Distributional Shift in RL
  - Decision Transformer
  - Pessimism Principle
tags:
  - deep-rl
  - reinforcement-learning
  - decision-sciences
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Learn a policy from a **fixed dataset of someone else's decisions**, with **no ability to try anything**. The enemy is the value function's wild guesses about actions nobody ever took — so the cure is **pessimism**.
> **Metaphor:** Learning to cook from a pile of other people's recipes — and never being allowed to taste.
> **Where it bites:** Healthcare, driving, industrial control, recommender logs — anywhere exploring on real users or patients is unacceptable. DPO is its LLM cousin.

---
A hospital has ten years of ICU records. For every patient: vitals by the hour, what each doctor did, and what happened.

You suspect a better treatment policy is hiding in there. You obviously **cannot** let an agent "explore" on patients to find out.

But [[Q-Learning]] is [[On-Policy vs Off-Policy|off-policy]] — it learns from data produced by *other* policies. And a fixed log of transitions looks a lot like a [[Experience Replay|replay buffer]] that simply never gets added to.

~={blue}So point an off-policy algorithm at the log and train. What goes wrong?=~

---
# Cooking without tasting 🍳

It fails badly — often ending up *worse* than the doctors it learned from. Follow the $\max$ to see why:

$$\text{target} = r + \gamma \max_{a'} Q(s', a')$$

That $\max$ ranges over **all** actions — including ones **no doctor ever took in that situation**. The network has never seen them. Its value there is pure extrapolation: could be anything.

And a $\max$ doesn't pick a *typical* guess. It picks the **most optimistic** one.

![[offline_rl_extrapolation.png]]
> [!TIP] Reading the chart
> Eight fits to the same logged data (black dots, all inside the green band). Inside the band they agree with each other and with the truth. **Outside it they fan out wildly.** The red triangles mark the action each fit's argmax would choose: **7 of 8 pick an action at the far edge — somewhere nobody ever tried** — and the truth there (dashed) is poor. The true best action, 0.2, is barely outside the data and none of them find it.

Online, this cures itself: the agent *tries* the over-rated action, gets slapped, and corrects. Offline, there is no slap. The inflated value becomes a **target** for the state before it, and the fantasy spreads backwards through the whole value function. It's [[Q-Learning#^maximisation-bias|maximisation bias]] with the safety catch off — the [[Function Approximation#^deadly-triad|deadly triad]] with no feedback at all.

> [!NOTE] Offline RL
> Learning a policy from a static dataset of transitions collected by some other (unknown, possibly poor) behaviour policy, **with no further interaction**. Its characteristic failure is **extrapolation error**: value estimates for out-of-distribution actions that are wrong, never corrected, and preferentially selected. ^offline-rl-def

> [!SUCCESS] Core idea
> ~={pink}Online, be optimistic about what you don't know. Offline, be **pessimistic**.=~ The confidence interval is the same; the sign flips — because being wrong is only cheap when you get to find out. Compare [[Upper Confidence Bound#The same idea, in bigger places 🔗|UCB]]: optimism is *how you gather information*. With no way to gather any, doubt has to count **against** an action. ^pessimism-principle

---
# The families of fix

| Family | Idea | Examples |
|---|---|---|
| **Constrain the policy** | only choose actions *like the ones in the data* | BCQ, BRAC, TD3+BC (*"RL, plus a behaviour-cloning term"*) |
| **Be conservative about values** | push $Q$ **down** on unseen actions, so they never win the $\max$ | **CQL** → [[Conservative Q-Learning for Offline RL]] |
| **Pessimistic model-based** | learn a model; penalise imagined rollouts wherever the model is unsure | MOPO, MOReL |
| **Skip RL altogether** | treat it as sequence modelling — *"given that I want a return of 90, what action comes next?"* | **Decision Transformer** |

Survey in the vault → [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]].

---
# Why bother — why not just copy the doctors?

| | [[Imitation Learning\|Behaviour cloning]] | **Offline RL** |
|---|---|---|
| Uses rewards? | no | **yes** |
| Best case | as good as the *average* demonstrator | **better than any single demonstrator** |
| How | copy | **stitch**: the good first half of one patient's care + the good second half of another's — a trajectory no one doctor ever performed |
| Needs from the data | good demonstrations | **coverage** — mediocre, *varied* data beats narrow expert data |
| Risk | compounding drift | extrapolation error |

> [!TIP] The counter-intuitive data requirement
> For cloning you want **expert** data. For offline RL you'd rather have **diverse** data — including mistakes — because the mistakes are what show the algorithm which actions are *bad*. A log from a single, consistent, excellent policy is nearly useless here: there's nothing to compare it to. Same point as [[Off-Policy Evaluation#^ope-requirements|OPE needing an exploring logger]] and [[System Identification#^persistent-excitation|system ID needing excitation]].

---
# The part everyone underestimates

> [!WARNING] You can't evaluate it either
> Training offline is hard. **Knowing whether it worked** is harder. You can't run the new policy to see — that's the constraint you started with. So you're thrown back on [[Off-Policy Evaluation]], whose importance weights *multiply* along a trajectory and blow up with the horizon. In practice: conservative methods, domain experts reading the policy's decisions, and a very cautious staged rollout. ~={red}Be deeply suspicious of any offline RL result that was only ever validated offline.=~ ^offline-eval-is-harder

> [!WARNING] "It's just off-policy RL with a big replay buffer"
> Off-policy algorithms were designed assuming they could **go and check**. Nearly all of them quietly depend on it. Take that away and the same code fails in a qualitatively new way → [[On-Policy vs Off-Policy#^off-policy-is-not-offline|off-policy ≠ offline]].

**The LLM connection.** [[DPO#^dpo-off-policy|DPO]] is offline preference optimisation: a fixed set of preference pairs, written by an *earlier* model, no new sampling. And it inherits the family trait — it can raise the probability of responses far outside anything in its data. The on-policy methods ([[PPO]] in [[RLHF]], [[GRPO]]) sample fresh responses every round, and that's most of what separates them.

---
> [!SUCCESS] If you remember one thing
> With no way to try things, **an action nobody took is an action you know nothing about** — ~={pink}and a value function's guess about it must be treated as a liability, not an opportunity.=~

---
# ⁉️
Every note so far assumed somebody could write down the reward. For "drive comfortably", "treat this patient well" or "give a helpful answer", nobody can. The rest of the chain is about what you do then — starting with the simplest idea of all: never mind the reward, **just copy the expert.**

→ [[Imitation Learning]]
