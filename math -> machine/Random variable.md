---
aliases:
  - random process
  - stochastic process
tags:
  - fundamentals
  - probability
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Not an unknown number — a **process whose outcome is decided by chance**, described by the distribution of values it can take.
> **Metaphor:** Your walk to work. The distance is fixed; the *time taken* is a random variable.
> **Where it bites:** Everything probabilistic. Expectation, variance, [[Uncertainty]], [[Beliefs]], and every loss function that's secretly an expectation.

---

So technically in simple / early math, the variable X refers to an unknown thing. But in stats, we've another kind of variable called the random variable 'X'. This is slightly different. It's actually a process or function being represented as a variable.
A good example to understand a random variable or process is -
Say you commute from home to work by walk every day. It's a distance of 1.2km as per google maps. But the time taken to commute can be modeled as a random variable X. The reason it is a random variable or a random process is because the outcome here is not deterministic but probabilistic / stochastic. Meaning there could be `n` number of things or events that can happen in between which changes the time taken to commute. Although the distance travelled is somewhat deterministic (again this also can be a random process given it totally depends on how many signals and crossings there are from Point A to Point B) , but circling back, even if the distance travelled is deterministic, the time taken is affected by the state of the signals when you reach a pedestrian crossing, whether or not, the footpath was crowded with other fellow pedestrians, priority users like old folks, or even cyclists and other unknowns.

---
# The difference that actually matters

Put the two side by side, because this is the distinction people blur:

| | **Algebraic variable** $x$ | **Random variable** $X$ |
|---|---|---|
| What it is | A number you don't know **yet** | A **process** with many possible outcomes |
| "Solving" it | $2x = 6 \Rightarrow x = 3$. Done, one answer | There is no single answer. Ever |
| What you can ask | "What is it?" | "What's the *average*? How *spread out*? What's $P(X > 30)$?" |
| Notation | lowercase $x$ | **uppercase** $X$; a specific observed outcome is lowercase $x$ |

That last row is a genuinely useful convention. $X$ is the process ("my commute time"). $x$ is Tuesday's actual 27 minutes. So $P(X = x)$ reads as *"the probability that the process produces this particular value."*

> [!NOTE] The formal definition
> A random variable is a **function** that maps outcomes of a random experiment to numbers. Roll two dice — the outcome is a pair like (3,4); the random variable "sum" maps that to 7. It's a function, which is why it gets uppercase notation but behaves like a variable. ^rv-is-a-function

> [!WARNING] It's neither random nor a variable
> Genuinely one of the worst-named objects in mathematics. It's a **deterministic function** applied to a random input. Worth knowing so the name stops bothering you. 🙃

---
# Describing one — the distribution

You can't state the value, so you state the **distribution**: which values are possible and how likely each is. Your commute might look like a bump centred at 18 minutes with a long right tail (traffic lights, crowds — things can only make it *slower*, never faster than a sprint).

Two flavours:
- **Discrete** — countable outcomes (dice, clicks, word tokens). Described by a **PMF**: $P(X = 5)$ is a real probability.
- **Continuous** — any value in a range (time, temperature, a model's logit). Described by a **PDF**, and here $P(X = 18.0000\ldots)$ is exactly **zero**. Only intervals have probability: $P(17 < X < 19)$.

> [!TIP] The classic gotcha
> A PDF value can exceed 1 — it's a *density*, not a probability. Only the area under it must integrate to 1. If someone says "the probability is 2.3", they've confused density with probability. ^density-not-probability

And two numbers summarise most of it:
- **Expectation** $E[X]$ — the long-run average. The centre.
- **Variance** $\text{Var}(X)$ — the average squared distance from that centre. The spread. ← *this* is the number that becomes [[Uncertainty|uncertainty]] downstream

> [!SUCCESS] Core idea
> A random variable answers *"what can happen and how often?"* — never *"what is it?"* Every statistic you compute is an attempt to compress that whole distribution into a few usable numbers. ^rv-core

---
# Why this underpins all of ML

Once you see it, the framing is everywhere:

- **Your data** is a sample from an unknown distribution. Training assumes test data comes from the *same* one — and distribution shift is exactly that assumption breaking.
- **Your loss** is an expectation. "Minimise the loss" really means minimise $E[\ell(f(X), Y)]$ over the data distribution — and since you can't see the true distribution, you approximate it with the average over your batch. That's the whole justification for mini-batch training.
- **Model outputs** are often distributions, not values. A softmax *is* a discrete distribution over classes; [[Cross Entropy]] and [[KL Divergence]] are ways of comparing two of them.
- **Weights themselves** can be random variables — that's a Bayesian neural net, and the source of epistemic [[Uncertainty]].

> [!TIP] The mental upgrade
> Beginners see a model as $\text{input} \rightarrow \text{answer}$. It's more accurate to see it as ~={blue}$\text{input} \rightarrow \text{a distribution over answers}$=~, from which we usually take the most likely one and quietly discard the rest. Most of the interesting work — calibration, sampling temperature, abstention — is about *not* discarding it.

Related: a **stochastic process** is a whole *family* of random variables indexed by time — your commute time on every day of the year, rather than one day. That's the object the [[Markov Property]] talks about.

---
# ⁉️
A stochastic process over time raises an immediate question: to predict tomorrow, how much of the past do I need? All of it? Or is today enough?

→ [[Markov Property]]
