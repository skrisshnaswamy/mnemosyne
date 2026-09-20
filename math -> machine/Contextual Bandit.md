---
aliases:
  - Contextual Bandits
  - Contextual Multi-Armed Bandit
  - Contextual MAB
  - LinUCB
  - Associative Search
  - Bandit with Side Information
  - Neural Bandit
tags:
  - bandits
  - reinforcement-learning
  - decision-sciences
  - recsys
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A bandit where you **see some features first** — who the user is, the time of day — and the best arm **depends on them**. Still no state: today's choice doesn't change tomorrow's user.
> **Metaphor:** A good waiter. Doesn't recommend the same dish to every table — reads the table first.
> **Where it bites:** Personalisation: news, ads, notifications, pricing, the ranking layer of a recommender. And RLHF is one, at the level of whole responses.

---
A news site has two headlines for the top slot: **A** (sport) and **B** (finance). You run a [[Multi-Armed Bandit|bandit]] for a week.

Result: A gets **5%** clicks. B gets **5%** clicks. A dead heat — the bandit shrugs and splits traffic evenly forever.

Then someone slices the data by time of day.

~={blue}Morning commuters: A 2%, B 8%. Evening readers: A 8%, B 2%.=~

---
# The waiter who reads the table 🍽️

![[contextual_bandit_ctr.png]]

There was never "no winner". There was a **strong winner in each group, pointing opposite ways** — and averaging over everyone cancelled them out exactly. A plain bandit asks *"which arm is best?"*. That was the wrong question. The right one is *"which arm is best **for this person, right now**?"*

Do the arithmetic. The plain bandit earns 5% whatever it does. A policy that shows B in the morning and A in the evening earns **8%** — a **60% lift**, from the same two headlines.

> [!NOTE] Contextual bandit
> Each round: observe a **context** $x$ → choose an action $a$ → receive a reward for **that action only**. The goal is a policy $\pi(x) \rightarrow a$. Unlike an [[Markov Decision Process|MDP]], the action does not influence the next context. ^contextual-bandit-def

| | Sees features? | Action changes the next situation? |
|---|---|---|
| [[Multi-Armed Bandit]] | ❌ | ❌ |
| **Contextual bandit** | ✅ | ❌ |
| [[Markov Decision Process\|MDP]] / [[Reinforcement Learning\|RL]] | ✅ | ✅ |

> [!SUCCESS] Core idea
> ~={pink}It's supervised learning with one cruel twist: you only ever see the label for the action you took.=~ Show headline B and you'll never know what A would have got *from that reader*. So you can't just fit a classifier to the logs — the logs only contain the choices the old policy liked, which is a biased sample. Handling that is the whole field → [[Off-Policy Evaluation]]. ^partial-feedback

---
# How they're built

All of them are *"a model that predicts reward from (context, action)"* + *"an exploration rule on top"*:

| Approach | Model | Explores by | Note |
|---|---|---|---|
| **ε-greedy on a model** | anything — GBDT, a neural net | random arm with prob. ε | crude, and by far the most common in industry, because the **logged propensities are trivially known** (that matters — see below) |
| **LinUCB** | a linear model per arm | [[Upper Confidence Bound\|UCB]]: prediction + a confidence width that's large in *unfamiliar regions of feature space* | Yahoo! news, 2010 → [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]] |
| **[[Thompson Sampling\|Thompson]]** | Bayesian linear / logistic regression | sample a weight vector, act greedily on it | handles batching well |
| **Neural bandits** | a deep net + bootstrapped heads / a Bayesian last layer | approximate posterior sampling | when features are raw text or images |

> [!TIP] Log the propensity. Always.
> Whatever you deploy, record **the probability with which the chosen action was picked**, alongside the context, the action and the reward. That one extra column is what makes it possible, later, to evaluate a *new* policy on *old* logs. It can't be reconstructed afterwards — see [[Observability#^ask-it-first|asking whether the information is even there]]. ^log-the-propensity

---
# Where you'll meet it

- **News and content** — the founding application.
- **Ads** — which creative, for this user, in this slot.
- **Notifications** — whether, when, and which message.
- **Recommenders** — typically the final re-ranking layer on top of a [[Recommender Systems - Evolution#Step 1: Candidate Generation (The Funnel) 🌪️|retrieve-then-rank]] stack.
- **Pricing and promotions**, **clinical dosing**, **UI layout**.

> [!TIP] RLHF is (almost) a contextual bandit 🤖
> Prompt = **context**. The model's whole response = **one action**. The reward model's score = **reward**. One step and the episode ends; the response doesn't change which prompt arrives next. That's why [[RLHF]] can get away with a [[Discount Factor|discount of 1]] and why [[GRPO]] needs no critic. It stops being a bandit the moment the task is multi-turn or tool-using → [[Agentic Workflows]].

---
# The decision that matters

> [!WARNING] "Our users behave sequentially, so we need full RL"
> Almost everything is sequential if you squint. The question is whether **your action materially changes the future context**. If showing article A barely affects who turns up next or what they want, then modelling those dynamics buys nothing and costs you credit assignment, high variance and instability. ~={red}A contextual bandit learns from a fraction of the data an RL agent needs=~ — reach for RL only when long-term effects are large *and* measurable (churn, habit formation, fatigue). ^cb-before-rl

| Sign you've outgrown a contextual bandit | |
|---|---|
| Optimising clicks hurts next week's retention | the action *does* change the future |
| Showing the same thing repeatedly causes fatigue | the state includes your own past actions |
| The task has several dependent steps | it's an MDP |

And the quiet dependency underneath all of it: the features. A contextual bandit is only as good as the context it's given → [[Embeddings]].

---
> [!SUCCESS] If you remember one thing
> "Which option is best?" is often the wrong question. ~={pink}Ask "best for **whom**?" — and remember you only ever see the outcome of the option you picked.=~

---
# ⁉️
Bandits shift traffic toward the winner *while* they learn. The classical alternative does the opposite: split traffic evenly, touch nothing, and decide at the end. When is that old-fashioned approach actually the right one?

→ [[AB Testing]]
