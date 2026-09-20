---
aliases:
  - Bandit
  - Bandits
  - MAB
  - Multi-Armed Bandits
  - Multi Armed Bandit
  - K-Armed Bandit
  - Stochastic Bandit
  - Bernoulli Bandit
  - One-Armed Bandit
  - Bandit Problem
tags:
  - bandits
  - reinforcement-learning
  - decision-sciences
  - fundamentals
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Choose repeatedly among options with **unknown payoffs**, learning only from the one you picked — in a world where **your choice doesn't change what you'll face next**. It's RL with no state.
> **Metaphor:** A row of slot machines with different hidden payout rates, and a bucket of coins.
> **Where it bites:** Headlines, ads, notifications, promo banners, clinical trials, hyperparameter search — and the question *"is this a bandit, or do I really need RL?"*

---
You walk into a casino with a bucket of 1,000 coins. In front of you, a row of slot machines — *one-armed bandits*, because they have one lever and they take your money.

Each machine pays out at its own fixed rate. One of them is much better than the rest. **Nobody will tell you which.**

Every coin you spend *finding out* about a bad machine is a coin you didn't spend on the good one. Every coin you spend on the machine that *looks* best so far is a coin you didn't use to check whether another is better.

~={blue}And notice one thing about this casino: pulling a lever doesn't change anything. The machines don't react. Round 500 looks exactly like round 1.=~

---
# The row of machines 🎰

![[multi_armed_bandit_slot_machines.png]]

That last observation is the whole definition.

In an [[Markov Decision Process|MDP]], what you do now changes the situation you'll face next — order stock today and you start tomorrow in a different state. That's what makes [[Credit Assignment|credit assignment]] hard.

In a bandit there **is** no "next situation". Each round stands alone. The only difficulty left is [[Exploration vs Exploitation|exploration]]: learning which arm is best without wasting too many pulls finding out.

> [!NOTE] Multi-armed bandit
> A sequential decision problem with $K$ actions ("arms"), each with an unknown reward distribution. Each round you pick one arm and observe **only that arm's** reward. There are no states and no transitions. The aim is to minimise [[Regret|regret]]. ^bandit-def

> [!SUCCESS] Core idea
> ~={pink}A bandit is an MDP with a single state — so it's the exploration problem with everything else stripped away.=~ That's exactly why it's worth studying on its own: the theory is clean, the algorithms are simple, and a huge number of real business problems turn out to be this and nothing more. See [[Markov Decision Process#The family tree|the family tree]]. ^bandit-is-one-state-mdp

---
# The test — bandit or MDP?

> [!TIP] Ask one question
> **"Does what I do now change the situation I'll face next?"**
> No → **bandit**. Partly, through *who the user is* → [[Contextual Bandit]]. Yes → [[Markov Decision Process|MDP]] / [[Reinforcement Learning|RL]].

| Problem | Verdict | Why |
|---|---|---|
| Which of 5 headlines to run | **bandit** | one reader's click doesn't change the next reader |
| Which ad for *this* user | **contextual bandit** | the best arm depends on the user; showing it doesn't change the next user |
| Clinical trial arm allocation | **bandit** — this is where the idea came from (Thompson, 1933) | patients are independent |
| Hyperparameter configurations | **bandit** with expensive pulls | → [[Bayesian Optimization]] |
| Inventory ordering | **MDP** | today's order *is* tomorrow's stock |
| A multi-turn conversation | **MDP** | each reply changes the dialogue state |
| A recommender over a long session | *arguably* an MDP; nearly always treated as a contextual bandit | the simplification is usually worth it |

> [!WARNING] "Bandits are just toy RL"
> The other way round: **using full RL on a bandit problem is an engineering mistake.** You take on credit assignment, long-horizon variance and instability to model dynamics that don't exist. When nothing carries over between rounds, a bandit is the *correct* model — and it will learn from a fraction of the data. ^bandits-are-not-toy-rl

---
# The algorithms, at a glance

Same three machines (30% / 50% / 70%), 10,000 pulls each:

![[bandit_pull_allocation.png]]

| Algorithm | Idea | [[Regret]] | Note |
|---|---|---|---|
| Greedy | always play the current best | linear ❌ | locks in early mistakes |
| [[Exploration vs Exploitation\|ε-greedy]] | mostly best, sometimes random | linear | simple, surprisingly OK at short horizons |
| [[Upper Confidence Bound\|UCB]] | play whatever *could* be best | **log** ✅ | deterministic; needs no prior |
| [[Thompson Sampling]] | sample a plausible world, play its best | **log** ✅ | the usual industry default |
| EXP3 | hedge with exponential weights | $\sqrt{T}$ | for *adversarial* rewards — makes no statistical assumptions |

---
# Variants you'll hear named

| Variant | What changes | Where |
|---|---|---|
| **[[Contextual Bandit\|Contextual]]** | you see features before choosing | personalisation |
| **Non-stationary / restless** | payout rates drift | anything seasonal. Use a sliding window or discount old data |
| **Adversarial** | an opponent sets the rewards | security, auctions |
| **Combinatorial / slate** | choose a *set* or an ordered list | a page of recommendations |
| **Dueling** | you only observe *"A beat B"* | interleaved ranking tests → [[Preference Learning]] |
| **Best-arm identification** | you care about the final pick, not the journey | that's an [[AB Testing\|A/B test]] → simple [[Regret]] |
| **Delayed / batched feedback** | rewards arrive late, in clumps | most production systems — a point for [[Thompson Sampling]] |

> [!TIP] The non-stationarity trap
> Textbook bandits assume a machine's payout rate never changes. Real click-through rates do — novelty wears off, seasons turn. An algorithm that has "converged" and stopped exploring will sail straight past the moment the best arm changes. In production, **never let exploration fall to zero.**

---
> [!SUCCESS] If you remember one thing
> If your action doesn't change tomorrow's situation, it's a **bandit** — ~={pink}and the only problem left is exploring efficiently, which is a solved problem.=~ Don't reach for RL.

---
# ⁉️
So how *should* you explore? ε-greedy explores at random — it wastes pulls on machines it already knows are bad. A smarter rule would spend its curiosity only where there's still real doubt.

→ [[Upper Confidence Bound]]
