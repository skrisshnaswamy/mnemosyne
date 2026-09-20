---
aliases:
  - TD Learning
  - TD
  - Temporal Difference
  - Temporal-Difference Learning
  - TD Error
  - TD(0)
  - TD(λ)
  - TD Lambda
  - Bootstrapping
  - n-step Returns
  - n-step TD
  - Reward Prediction Error
tags:
  - reinforcement-learning
  - decision-sciences
  - fundamentals
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Don't wait for the final outcome — **update each prediction toward the very next prediction.** The gap between the two is the **TD error**, and it's the learning signal for most of RL.
> **Metaphor:** Driving home and revising your ETA at every landmark, rather than only once you're on the doorstep.
> **Where it bites:** Q-learning, SARSA, actor-critic, DQN — all TD. Also the bias-vs-variance dial ($n$-step, $\lambda$) and a name clash with the statistical bootstrap.

---
You leave the office and guess: **30 minutes** to get home.

| Where you are | Elapsed | You now predict (total) |
|---|---|---|
| leaving the office | 0 | **30** |
| reach the car — it's raining | 5 | **40** |
| off the motorway, quicker than feared | 20 | **35** |
| stuck behind a lorry | 30 | **40** |
| turning into your street | 40 | **43** |
| home | 43 | **43** |

It took 43. Your first guess of 30 was 13 minutes out.

~={blue}At what moment did you *learn* that 30 was too optimistic?=~

---
# The drive home 🚗

Not on the doorstep. You learned it **five minutes in, when you saw the rain** and revised to 40. By the time you got home, you'd already absorbed most of the lesson.

[[Monte Carlo Methods|Monte Carlo]] ignores that. It waits for the real answer (43) and then drags *every* earlier guess toward it: 30 → 43, 40 → 43, 35 → 43…

**Temporal-difference learning** updates as it goes. Standing at the car, it says: *"Leaving the office, I thought 30. One step later I think 40. So 'leaving the office' should have been nearer 40."* It adjusts that one prediction **right then** — using nothing but the next prediction.

$$V(s) \;\leftarrow\; V(s) + \alpha\,\big[\;\underbrace{r + \gamma V(s')}_{\text{what I now think}} - \underbrace{V(s)}_{\text{what I thought}}\;\big]$$

The bracket is the **TD error**, $\delta$ — *the surprise*. A guess, corrected by a slightly later and slightly better-informed guess.

> [!NOTE] Temporal-difference learning
> Learn a value function by moving each estimate toward a **bootstrapped target** — the observed reward plus the *current estimate* of the next state's value — instead of waiting for the full return. Model-free, online, step by step. ^td-def

> [!SUCCESS] Core idea
> ~={pink}Learn a guess from a guess.=~ It works because the later guess contains one more step of *real* information. And it's the [[Bellman Equation]] in sampled form: the TD error is exactly how badly "value of here = reward + value of there" was violated **on this one step**. Drive the average violation to zero and you've solved for the values. ^learn-a-guess-from-a-guess

> [!TIP] You've seen this equation before 🎯
> **Kalman:** `new = prediction + gain × (measurement − prediction)`
> **Running mean:** `new = old + (1/n) × (sample − old)`
> **TD:** `new = old + α × (target − old)`
> Same shape every time — *old estimate plus a fraction of the surprise*. One corrects a belief about where you are, one about how good things are. See [[Decision Sciences#^kalman-td-same-shape|the callback]] and [[Bayes Filter]].

---
# TD vs Monte Carlo — the honest comparison

![[td_vs_mc_random_walk.png]]
> [!TIP] Reading the chart
> A five-state random walk, the textbook test. After 100 episodes the best TD setting is at an error of **0.035**; the best Monte Carlo setting is at **0.089** — and every TD curve gets there faster. TD wins here because each update borrows strength from the neighbouring state's estimate, instead of relying on one noisy full-episode outcome.

| | **Monte Carlo** | **TD(0)** |
|---|---|---|
| Target | the real return $G$ | $r + \gamma V(s')$ |
| Bias | **none** | some — the target leans on your own (imperfect) estimate |
| Variance | **high** — a return is a sum of many random steps | **low** — one random step |
| Learns | at the end of the episode | **every step** |
| Never-ending tasks | ❌ | ✅ |
| Needs a [[Markov Property\|Markov]] state | no | **yes** — it trusts $V(s')$ to summarise the future |

---
# The dial between them — $n$-step and $\lambda$

Why choose? Look **$n$ real steps** ahead, *then* bootstrap:

$$r_0 + \gamma r_1 + \cdots + \gamma^{n-1} r_{n-1} + \gamma^n V(s_n)$$

$n = 1$ is TD(0). $n = \infty$ is Monte Carlo. **TD($\lambda$)** blends all the $n$-step targets with geometrically decaying weights; $\lambda = 0 \rightarrow$ TD(0), $\lambda = 1 \rightarrow$ Monte Carlo. It's implemented with [[Credit Assignment#How RL moves credit backwards|eligibility traces]], and it's the very same dial as **GAE** in [[Actor-Critic]] and [[PPO]]. In practice the sweet spot is rarely at either end — $\lambda \approx 0.9$–$0.95$.

---
# Two things worth knowing

> [!TIP] Your brain does this 🧠
> In the 1990s Wolfram Schultz recorded dopamine neurons in monkeys learning that a light predicts juice. Early on, the neurons fired at the **juice**. After learning, they fired at the **light** — and stayed silent at the juice, because it was now *expected*. Skip the juice, and they dipped below baseline at the moment it should have come. That's a TD error, cell by cell: dopamine signals **reward prediction error**, not reward. It's one of the cleanest meetings of an algorithm and neuroscience there is.

> [!WARNING] "Bootstrapping" — a name clash
> In statistics, *the bootstrap* means resampling your data with replacement to get error bars. **In RL, bootstrapping means updating an estimate from another estimate** — pulling yourself up by your own bootstraps. Different ideas, same word. And RL's kind has a dark side: combine it with function approximation and off-policy data and learning can diverge → [[Function Approximation#^deadly-triad|the deadly triad]]. ^bootstrapping-name-clash

---
> [!SUCCESS] If you remember one thing
> **TD error = (reward + what I now expect) − (what I expected).** ~={pink}Learning happens at the moment of surprise, not at the end of the story.=~

---
# ⁉️
TD as described *evaluates* — it tells you how good states are under whatever you're currently doing. It doesn't tell you what to do **differently**. To improve a policy without a model, you need to score *actions*, not just states.

→ [[Q-Learning]]
