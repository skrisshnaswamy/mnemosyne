---
aliases:
  - UCB
  - UCB1
  - Upper Confidence Bounds
  - UCB Algorithm
  - Optimism Under Uncertainty
  - Confidence Bound
  - Exploration Bonus
tags:
  - bandits
  - reinforcement-learning
  - decision-sciences
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Score each option by **its average so far + a bonus for how little you've tried it**, and play the highest. *Optimism in the face of uncertainty.*
> **Metaphor:** Two restaurants: 4.2★ from 900 reviews, and 4.0★ from 3 reviews. The second one *could* be a 4.8.
> **Where it bites:** Bandits with no prior, the selection rule inside MCTS (AlphaGo), LinUCB for news recommendation, GP-UCB in Bayesian optimisation.

---
Two restaurants on a map app.

- **A:** 4.2 ★ — from **900** reviews.
- **B:** 4.0 ★ — from **3** reviews.

A has the better score. But how much do you really know about B? Three reviews. Its true quality could be anywhere from a 3 to a 4.8. A's can't be — 900 people have pinned it down.

~={blue}If you're going to be eating in this town for years, which one is worth trying tonight?=~

---
# Benefit of the doubt ⭐

B. Not because it's *probably* better — it probably isn't — but because it's the only one that **could** be, and one visit will tell you a lot.

And look at what happens either way:

- **B turns out great** → you've found a better restaurant. Win.
- **B turns out mediocre** → its average drops, its uncertainty shrinks, and you stop going. You lost one evening.

Being optimistic is *self-correcting*. The only options that keep getting chosen are ones that are genuinely good — or ones you still don't know much about.

So judge every option by **the top of its plausible range**:

$$\text{score}_a = \underbrace{\bar{x}_a}_{\text{average so far}} + \underbrace{\sqrt{\frac{2 \ln t}{n_a}}}_{\text{bonus for ignorance}}$$

where $n_a$ is how many times you've tried arm $a$ and $t$ is the total number of rounds.

Put numbers in. After $t = 1{,}000$ rounds ($\ln t = 6.9$):

| Arm | Average | Tried | Bonus | **Score** |
|---|---|---|---|---|
| A | 0.30 | 900 | $\sqrt{13.8/900} = 0.12$ | 0.42 |
| B | 0.25 | 100 | $\sqrt{13.8/100} = 0.37$ | **0.62** ← play this |

B has the *worse* average and gets played — because you've only tried it 100 times and it might still turn out to be the better one.

> [!NOTE] Upper Confidence Bound (UCB1)
> Play each arm once, then always choose $\arg\max_a \big[\bar{x}_a + \sqrt{2 \ln t / n_a}\big]$. The bonus is a (Hoeffding) confidence-interval half-width: with high probability the arm's true mean is below the score. → [[Finite-time Analysis of the Multiarmed Bandit Problem (UCB1)]] ^ucb-def

> [!SUCCESS] Core idea
> ~={pink}Act as if every option is as good as it plausibly could be.=~ Either you're right and you win, or you're wrong and you *find out quickly* — because trying it is what shrinks its interval. Exploration is no longer random; it's aimed precisely at whatever is both **promising** and **uncertain**. ^optimism-is-self-correcting

---
# Watching it work

![[ucb_intervals.png]]
> [!TIP] Reading the chart
> Dots are averages, bars are the optimism bonus, dotted lines are the (hidden) true rates. **After 30 pulls** every bar is huge and the pulls are spread evenly (10 / 8 / 12). **After 3,000**, the best machine has had **2,668** pulls and its bar has almost vanished. The 30% machine got only **80** — just enough for its *upper* bound to drop below the winner's average. ~={blue}It stopped being tried the moment it could no longer plausibly be best.=~

**Why $\ln t$ on top?** Without it, an arm that got unlucky early would be abandoned for good. The slowly growing numerator keeps every neglected arm's bonus creeping upward, so each gets re-checked occasionally — but only *logarithmically* often. That's exactly what produces [[Regret|logarithmic regret]].

---
# UCB versus the alternatives

| | [[Exploration vs Exploitation\|ε-greedy]] | **UCB** | [[Thompson Sampling]] |
|---|---|---|---|
| Explores | at random | where the **upper bound** is highest | by sampling from the belief |
| Randomised? | yes | **no — deterministic** | yes |
| Needs a prior? | no | **no** | yes |
| Regret | linear | **log** | **log** |
| Delayed / batched feedback | fine | ⚠️ picks the *same arm* for the whole batch | handles it gracefully |
| Tuning | $\varepsilon$ | the constant in the bonus | the prior |

> [!WARNING] The textbook constant is cautious
> UCB1's $\sqrt{2\ln t/n}$ comes from a worst-case bound, and it over-explores in practice — in [[Regret#^shape-not-constant|the regret experiment]] it was still *behind* ε-greedy at 1,000 pulls. Real deployments scale the bonus by a tuned constant $c < 1$, or use tighter variants (UCB-Tuned, KL-UCB). ~={red}The guarantee is about the shape of the curve, not about next Tuesday.=~ ^ucb-constant

---
# The same idea, in bigger places 🔗

| Where | How UCB appears |
|---|---|
| **[[Monte Carlo Tree Search]]** | every node of the search tree is a little bandit, and children are chosen by UCB ("UCT"). This is the selection rule inside AlphaGo |
| **[[Contextual Bandit\|LinUCB]]** | the confidence interval comes from a linear model of the context → [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]] |
| **[[Bayesian Optimization]]** | GP-UCB: pick the hyperparameters maximising *predicted mean + $\beta \times$ predicted std* → [[Gaussian Process Optimization in the Bandit Setting (GP-UCB)]] |
| **Deep RL exploration** | count-based and curiosity bonuses are UCB's $1/\sqrt{n}$ generalised to states you can't count |

> [!TIP] The contrast worth remembering
> In **online** learning you add the uncertainty — *optimism*, because being wrong is cheap and informative. In **[[Offline RL|offline]]** learning you **subtract** it — *pessimism*, because you can't go and check. Same interval; opposite sign; decided entirely by whether you get to try again.

---
> [!SUCCESS] If you remember one thing
> **Score = average + uncertainty bonus.** ~={pink}Give the under-tried option the benefit of the doubt — if you're wrong, you'll know soon, and that's the point.=~

---
# ⁉️
UCB is deterministic and needs a confidence-interval formula for whatever you're modelling. There's an older idea — 1933 — that gets the same behaviour with no formula at all, just by rolling dice in the right way.

→ [[Thompson Sampling]]
