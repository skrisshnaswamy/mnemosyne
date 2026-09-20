---
aliases:
  - A-B Testing
  - A-B Test
  - AB Test
  - AB Tests
  - Split Testing
  - Online Experiment
  - Online Experiments
  - Online Controlled Experiment
  - Randomised Controlled Trial
  - Randomized Controlled Trial
  - RCT
  - Statistical Power
  - Peeking
tags:
  - experimentation
  - bandits
  - decision-sciences
  - statistics
  - causal-inference
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Split traffic **at random** between variants, wait, compare. Randomisation is what makes the difference *causal*. It's **pure exploration, then pure exploitation** — the opposite trade to a bandit.
> **Metaphor:** A clinical trial. Because that's literally what it is.
> **Where it bites:** Shipping decisions. Under-powered tests, peeking, and the perennial *"should this be a bandit instead?"*

---
A new checkout button. You send half your visitors to the old one (**A**) and half to the new one (**B**) for two weeks — 20,000 each.

- A converts at **5.0%**
- B converts at **5.5%**

The PM is delighted: a 10% relative lift. Ship it?

~={blue}Two questions, and they're separate. First: is that difference *real*? Second: what did it *cost* you to find out?=~

---
# The clinical trial 💊

**Is it real?** The noise on a conversion rate is surprisingly large. To reliably tell 5.0% from 5.5% — 80% power, 5% false-positive rate — you need roughly

$$n \approx \frac{16\, p(1-p)}{\delta^2} = \frac{16 \times 0.05 \times 0.95}{0.005^2} \approx \mathbf{30{,}000} \text{ per arm}$$

You had 20,000. So this test was **under-powered**: even if B truly is better, you'd miss it a good part of the time — and when an under-powered test *does* come up significant, the measured lift is usually an exaggeration.

> [!TIP] The rule of thumb worth memorising
> $n \approx 16\,\sigma^2 / \delta^2$ per arm. **Halve the effect you want to detect and the sample size quadruples.** Small lifts are *expensive* to see — which is the business case for variance reduction such as [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)|CUPED]].

**What did it cost?** For the whole fortnight, half your visitors got the worse button. If B really is half a point better, that's $20{,}000 \times 0.005 = $ **100 conversions** you gave up in order to know. That's the [[Regret|regret]] of the experiment — a fee you pay *deliberately*, for certainty.

> [!NOTE] A/B test
> A randomised controlled experiment: units are assigned to variants **at random**, in fixed proportions, for a pre-planned duration, and outcomes are compared with a pre-specified statistical test. ^ab-test-def

> [!SUCCESS] Core idea
> ~={pink}Randomisation is the entire point.=~ Because a coin decided who saw B, nothing about the users can be correlated with which variant they got — so any difference in outcome was **caused by** the variant. It's the one tool in this vault that gives you causation rather than correlation, which is exactly what [[Regression Analysis#^coefficient-is-not-causal|a regression coefficient can't]]. ^randomisation-gives-causation

---
# A/B test or bandit?

Same two variants (4% vs 6%), 20,000 visitors, two strategies:

![[ab_vs_bandit_allocation.png]]
> [!TIP] Reading the chart
> The A/B test holds 50/50 to the end and gives up **200** conversions. The bandit ([[Thompson Sampling|Thompson]]) has moved 90% of traffic to B within a couple of thousand visitors and gives up **16**. So why would anyone still run an A/B test?

Because they're optimising different things:

| | **A/B test** | **[[Multi-Armed Bandit\|Bandit]]** |
|---|---|---|
| Goal | **learn a fact**, with a known error rate | **earn the most** while learning |
| Minimises | [[Regret#Two kinds of regret — and they want different algorithms\|simple regret]] — the quality of the final decision | cumulative regret — losses along the way |
| Allocation | fixed | adaptive |
| You get | an effect size, a confidence interval, a p-value | a policy. **No clean estimate** — the losing arm has very little data |
| Several metrics / guardrails | ✅ watch them all | ❌ it optimises exactly one number |
| Novelty effects, weekly cycles | ✅ fixed duration spans them | ⚠️ may converge before they show |
| **Use when** | long-lived decisions · you need to know *how much* · many stakeholders | short-lived choices (headlines, promos) · many variants · showing losers is costly |

> [!WARNING] "A bandit is just a faster A/B test"
> It answers a **different question**. A bandit is great at *"which should I be showing?"* and poor at *"by how much is B better, and what happened to retention and revenue?"* — because it starved the losing arm of data, and adaptive sampling biases the naive averages. If the output you need is a **number for a slide**, run the A/B test. ^bandit-is-not-faster-ab

---
# The ways A/B tests go wrong 🪤

| Trap | What happens | Defence |
|---|---|---|
| **Peeking** | checking daily and stopping at the first "significant" result inflates false positives from 5% to 30%+ | fix the duration up front, or use sequential methods → [[Always Valid Inference- Bringing Sequential Analysis to A-B Testing]] |
| **Under-powering** | "no significant difference" gets read as "no difference" | do the sample-size sum *before* you start |
| **Many metrics, many segments** | slice 20 ways and one will be "significant" by luck | pre-register the primary metric; correct for multiple comparisons |
| **Interference** | drivers, marketplaces, social feeds — treating one unit affects another | cluster or time-based randomisation → [[Design and Analysis of Switchback Experiments]] |
| **Sample-ratio mismatch** | you asked for 50/50 and got 50.8/49.2 | something's broken in assignment or logging. Stop and debug |
| **Novelty / primacy** | a new thing gets clicked *because* it's new | run longer; look at the trend |

The standard reference is [[Controlled experiments on the web- survey and practical guide (DMKD)]].

---
# Where it sits in this story

An A/B test is the two-restaurant problem from [[Exploration vs Exploitation]] played in the most disciplined possible way: **explore with a fixed schedule, then exploit forever**. It's also the cleanest data you will ever own — *uniformly random* action assignment, with known propensities of exactly 0.5. Which makes A/B logs perfect raw material for the next idea.

---
> [!SUCCESS] If you remember one thing
> An A/B test buys **certainty about a fact**; a bandit buys **reward while learning**. ~={pink}Decide which one you're shopping for before you start — and do the sample-size arithmetic first, because small lifts are expensive to see.=~

---
# ⁉️
Running an experiment takes weeks, and you have fifty ideas. Meanwhile there are months of logs from the system that's already live. Could those logs tell you how a *new* policy would have done — without ever deploying it?

→ [[Off-Policy Evaluation]]
