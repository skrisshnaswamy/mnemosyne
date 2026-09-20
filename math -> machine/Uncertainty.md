---
aliases:
  - Aleatoric Uncertainty
  - Epistemic Uncertainty
tags:
  - fundamentals
  - probability
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** There are **two** kinds of "I don't know" — noise in the world (*aleatoric*, can't fix it) and gaps in my knowledge (*epistemic*, more data fixes it).
> **Metaphor:** A fair die you've inspected vs. a stranger's die you haven't. Both feel uncertain. Only one gets better if you look closer.
> **Where it bites:** Exploration vs exploitation, active learning, calibration, "should the model abstain?", and every safety conversation you'll ever sit in.

---
# Two dice 🎲

Let me give you two dice.

**Die 1** is a normal, fair, six-sided die. You've inspected it. You know *exactly* what it is. Now — what will it roll?

You have no idea. It's a $1/6$ chance for each face. And here's the thing: **you can roll it a million times and you will still not be able to predict roll number one-million-and-one.** More data does not help you. The unpredictability lives in the die, not in your head.

**Die 2** I pulled out of my pocket. It might be fair. It might be weighted. I'm not telling you. What will it roll?

You also have no idea. *But this is a completely different kind of not-knowing.* If I let you roll it 500 times, you'd learn a lot. You'd figure out it's weighted toward 6, or that it's fine. Your not-knowing would **shrink**.

Same feeling — "I don't know" — two totally different animals.

> [!NOTE] Aleatoric uncertainty
> The randomness **inherent in the world itself**. Irreducible. Collecting more data does not reduce it, it only lets you *measure* it more precisely. (From *alea*, Latin for dice.) ^aleatoric

> [!NOTE] Epistemic uncertainty
> Uncertainty from **your own ignorance** — you haven't seen enough, or your model is wrong. Reducible. This is the one that more data, or a better model, actually shrinks. (From *episteme*, Greek for knowledge.) ^epistemic

> [!SUCCESS] Core idea
> The single question that separates them: ~={blue}**"Would more data make this go away?"**=~
> Yes → epistemic. No → aleatoric. ^the-test

---
# Why anyone cares (this is the practical bit)

Both dice give you the same answer — "1/6-ish, don't know." But you should **behave completely differently** about them.

For **Die 1**, gathering more information is a waste of time. Just make your best decision and accept the variance. Hedge, buy insurance, build a buffer.

For **Die 2**, gathering information is *the highest-value thing you can do*. Roll it. Investigate. The uncertainty is an opportunity, not a cost.

> [!TIP] The one-sentence version for a meeting
> Aleatoric uncertainty says **"plan for variance."** Epistemic uncertainty says **"go find out."** 🔍

And once you see it, it's everywhere:

| Situation | Aleatoric part | Epistemic part |
|---|---|---|
| **Self-driving car** | Sensor noise, rain on the lens — will always be there | "I have never seen a road that looks like this" ← the dangerous one |
| **Medical model** | Two patients, identical charts, different outcomes | "No one in my training set was this age with this condition" |
| **Demand forecast** | Random day-to-day customer variation | "We've never operated in this city" |
| **LLM** | Many valid ways to phrase the same answer | "I genuinely don't know this fact" ← *this* is what hallucination looks like from the inside |

That last row is the important one. A model that can't tell its two uncertainties apart **cannot know when to say "I don't know."** It produces a confident-sounding answer from an epistemic void. That's the whole problem.

---
# The bit that actually matters: it changes what you do next

This is where uncertainty stops being a philosophy topic and starts driving algorithms.

## Exploration vs exploitation
Every bandit and RL algorithm you've read about is fundamentally an **epistemic uncertainty machine**.

Why would you ever pick the arm that *isn't* currently the best? Because your estimate of it is **uncertain**, and uncertainty is reducible, and reducing it might reveal something better. You're not being random — you're paying a small known cost to buy information.

- **UCB** — pick the arm with the highest *optimistic* estimate: $\text{mean} + \text{(uncertainty bonus)}$. Literally adds epistemic uncertainty to the score. ~={blue}"Be optimistic in the face of uncertainty."=~
- **Thompson sampling** — sample a plausible world from your [[Beliefs|belief]] and act as if it's true. Arms you're unsure about get sampled high sometimes, so they get tried.
- **$\epsilon$-greedy** — the crude version: explore at random. It works, but it's uncertainty-blind, so it wastes pulls on arms it's already sure are bad.

See [[Decision Sciences]] for where bandits sit in the wider story.

## Active learning
Which 1,000 samples should you pay a human to label, out of a million unlabelled ones? **The ones the model is most epistemically uncertain about.** Labelling data the model already handles confidently teaches it nothing.

## Knowing when to abstain
A model that outputs a calibrated epistemic uncertainty can route hard cases to a human instead of guessing. This is most of the practical value of uncertainty estimation in production, and it's the thing that's usually missing.

---
# You already know one of these machines

Go back to the [[Kalman Filter]]. Look at what it's actually doing:

> "If the sensor is very **uncertain** (noisy), the Kalman Gain is low, so the filter trusts the **prediction** more. If the sensor is very **certain**, the gain is high, so it trusts the **measurement** more."

That's the Kalman filter weighing two sources **by their uncertainty** — the entire algorithm is uncertainty arithmetic. And notice it needs *both* flavours handed to it up front:
- **Measurement uncertainty** — the sensor's noise. Aleatoric. A cheap radar will always be noisy.
- **Process uncertainty** — how much you doubt your own physics model. Epistemic-flavoured. A better model of the wind would shrink it.

Which is a nice reminder: this isn't new theory. You've already got the intuition, it just didn't have these two names attached. ✅

---
# How you actually get a number out of a model

Standard neural nets are **terrible** at this by default. A softmax output of $0.97$ is not a probability of being right — it's just the largest number that came out of the final layer, and networks are notoriously **overconfident** on inputs unlike anything they trained on. Exactly the case where you needed the warning.

Rough ladder of options, cheapest first:

1. **Softmax entropy** — nearly free, but conflates the two uncertainties and is badly calibrated. Use it knowing that.
2. **Temperature scaling** — fit one scalar on a validation set to squash overconfident logits. Absurdly cheap, fixes *calibration* (not epistemic estimation). Do this before anything fancier.
3. **Deep ensembles** — train 5 models with different seeds. Where they **agree**, the spread is small; where they **disagree**, you've found epistemic uncertainty. Simple, and still the strongest baseline. Expensive: 5× everything.
4. **MC Dropout** — leave [[Regularization#Dropout|dropout]] switched on at *inference*, run it 30 times, look at the variance across runs. A poor person's ensemble from a single trained model.
5. **Bayesian neural nets** — learn a *distribution* over each weight rather than a point value. Principled, and mostly impractical at scale.

> [!TIP] Separating the two in practice
> Run an ensemble. **Average prediction entropy** ≈ total uncertainty. **Disagreement between members** ≈ epistemic. Total minus disagreement ≈ aleatoric. Members agreeing on a coin flip is aleatoric; members confidently contradicting each other is epistemic. ^separating

> [!WARNING] Calibration ≠ accuracy
> A **calibrated** model is one where, of all the things it called "70% likely", about 70% turn out true. A model can be highly accurate and badly calibrated, or poorly accurate and perfectly calibrated. If you're going to *act* on the confidence number, calibration is the property you need — and it's measured separately (reliability diagrams, ECE).

---
> [!SUCCESS] If you remember one thing
> Ask **"would more data fix this?"** Everything else — explore or exploit, label or don't, answer or abstain — follows from the answer.

---
# ⁉️
So we can measure uncertainty. But measuring it is passive. What we actually want is to *carry it around and update it* as evidence arrives — to hold "what I currently think is true" as a living thing that reacts to what I see.

That thing has a name: [[Beliefs]].
