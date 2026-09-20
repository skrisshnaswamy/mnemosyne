---
aliases:
  - Bradley-Terry Model
  - Bradley–Terry Model
  - Bradley-Terry
  - Bradley–Terry
  - Preference Model
  - Preference Modelling
  - Preference Modeling
  - Reward Modelling
  - Reward Modeling
  - Pairwise Preferences
  - Pairwise Comparison
  - Learning from Human Preferences
  - Elo Rating
tags:
  - alignment
  - reinforcement-learning
  - llm
  - decision-sciences
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** People can't *score* quality consistently, but they can **compare**. So collect *"A or B?"* judgements and fit a hidden score per item such that **the gap between two scores predicts who wins**. That fitted scorer is a reward model.
> **Metaphor:** The optometrist. Nobody can state their own prescription. Everybody can answer *"better with lens one… or lens two?"*
> **Where it bites:** The reward model in RLHF, the maths inside DPO, Elo ratings, Chatbot Arena leaderboards.

---
Ask ten people to rate a summary **out of 10**.

You'll get 4, 8, 6, 7, 9, 5… One person's 7 is another's 5. The *same* person gives a 7 on Monday and a 6 on Friday. Try to train a reward function on that and you're mostly fitting the raters' moods.

Now show the same ten people **two** summaries and ask: *which is better?*

Nine say the second. In four seconds each.

~={blue}Why is the second question so much easier — and how do you turn a pile of "this one" answers into a number a machine can optimise?=~

---
# The optometrist 👓

![[preference_learning_optometrist.png]]

Because it needs no **scale**. An absolute score requires you to carry a calibrated ruler in your head. A comparison only needs you to *look*.

An optometrist never asks *"how blurry is that, out of ten?"* — they flip between two lenses: **"one… or two?"** A few dozen comparisons later they've pinned down a number *you could never have told them*.

Here's the model that does the same for any set of items. Assume each has a hidden score $r$, and that the chance A beats B depends only on the **gap**:

$$P(A \succ B) = \sigma(r_A - r_B) = \frac{e^{r_A}}{e^{r_A} + e^{r_B}}$$

![[bradley_terry_sigmoid.png]]
> [!TIP] Reading the chart
> Equal scores → a coin flip. A gap of **1 → 73%**. **2 → 88%**. **3 → 95%**. The top axis shows the same thing in chess terms — this *is* the **Elo** system: a 400-point gap means the stronger player wins ~91% of the time. Human preference is modelled as a *noisy* comparison: the better answer usually wins, not always.

> [!NOTE] Bradley–Terry model
> A model of pairwise comparisons in which every item has a latent scalar score and $P(i \text{ beats } j) = \sigma(r_i - r_j)$. Fitting it to observed comparisons recovers the scores. (1952 — for ranking sports teams.) ^bradley-terry-def

**To make it a reward model**, let a network produce the score — $r_\phi(\text{prompt}, \text{response})$ — and train it so the human-preferred response wins:

$$\mathcal{L} = -\log \sigma\big(\,r_\phi(x, y_\text{chosen}) - r_\phi(x, y_\text{rejected})\,\big)$$

That's **logistic regression on a difference of scores** — ordinary [[Cross Entropy]], where the "label" is simply *which one the human picked*.

> [!SUCCESS] Core idea
> ~={pink}Turn judgements people **can** make (comparisons) into the thing an optimiser **needs** (a scalar).=~ It's [[Inverse Reinforcement Learning]] with a far cheaper signal: instead of inferring the reward from watching an expert *perform*, infer it from watching a person *choose*. ^comparisons-into-scalars

---
# Only the gap is real

Add 100 to every score. Every difference is unchanged; every prediction is identical.

So the absolute value of a reward-model score **means nothing**. "This response scored 4.2" carries no information on its own. Only *"it scored 1.3 higher than that one"* does. The reward is [[Observability|identifiable only up to a constant]] — which is why RLHF pipelines normalise rewards, and why [[GRPO]] can get away with using nothing but each response's score *relative to its group*.

---
# Where it came from, and where it went

**2017 —** *Deep RL from Human Preferences* (Christiano et al.). A simulated robot learned to do a **backflip** from about 900 human comparisons of short video clips — under an hour of someone's time — for a behaviour nobody could write a reward function for. The recipe: *compare → fit a reward model → run RL against it*.

**2022 —** the same three steps, applied to a language model, is [[RLHF]] → [[Training language models to follow instructions with human feedback]].

**2023 —** [[DPO]] noticed you can substitute the Bradley–Terry formula straight into the RL objective, solve for the policy, and train on the comparisons directly. The "reward" becomes $\beta \log \frac{\pi(y \mid x)}{\pi_\text{ref}(y \mid x)}$ — no separate reward model at all → [[DPO#^dpo-core|the DPO core idea]].

And on leaderboards: **Chatbot Arena** ranks models by fitting exactly this model to millions of human "which answer is better?" votes.

---
# What goes wrong 🪤

| Failure | What happens |
|---|---|
| **The raters' biases are the labels** | people prefer longer, more confident, more agreeable answers → the reward model learns *that* → the policy is optimised into it → [[RLHF#^sycophancy\|sycophancy]], length inflation |
| **Disagreement is treated as noise** | two raters who genuinely disagree look, to Bradley–Terry, like one rater being inconsistent. It learns the *average* taste, and minorities vanish |
| **Intransitive preferences** | A > B, B > C, C > A happens with real people. A single score per item can't represent it |
| **Valid only near its training data** | the reward model saw responses from the *SFT model*. Optimise hard and the policy produces text unlike anything it was trained on — where its scores are fiction → [[Reward Hacking#^optimiser-as-adversary\|the optimiser finds the gap]] |
| **It's a lossy copy of a lossy signal** | the policy learns from a model of a sample of people's snap judgements |

That fourth row is the whole reason for the [[RLHF#^kl-leash|KL leash]]: keep the policy close to where the reward model is trustworthy. It's the [[Offline RL#^pessimism-principle|same pessimism]] as offline RL — *don't trust your value estimates outside the data*.

> [!WARNING] "The reward model measures quality"
> It measures **what these raters, shown these pairs, tended to pick** — on an arbitrary scale where only gaps mean anything, and only near the distribution it was trained on. That's a useful thing to have. It isn't "quality", and the difference between the two is exactly what an optimiser will exploit. ^rm-is-not-quality

> [!TIP] Practical levers
> Clear guidelines and rater calibration beat more data. Track **inter-rater agreement** — if humans agree 70% of the time, a reward model at 72% accuracy is at the ceiling, not broken. Keep a held-out preference set to watch for over-optimisation. Refresh the reward model on the *current* policy's outputs. Several reward models (an ensemble) give you an uncertainty estimate for free.

---
> [!SUCCESS] If you remember one thing
> People can't score, but they can **compare** — and $P(A \succ B) = \sigma(r_A - r_B)$ turns comparisons into a reward. ~={pink}Only the gaps mean anything, and only where the comparisons were made.=~

---
# ⁉️
You now have a learned reward. The next step is to optimise a language model against it without letting the model wander off to wherever that reward stops making sense.

→ [[RLHF]] — then [[DPO]], which removes the reward model — and then come back for the newest branch, which removes the *critic* instead: → [[GRPO]]
