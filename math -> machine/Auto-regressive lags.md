---
aliases:
  - AR lags
  - Lagged Dependent Variable
tags:
  - causal-inference
  - time-series
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Put the outcome's **own past values** on the right-hand side of the regression, because today's outcome depends on yesterday's.
> **Metaphor:** Momentum in a business metric. Yesterday's sales are the single best predictor of today's — and if you ignore that, you'll credit the effect to whatever you *did* measure.
> **Where it bites:** Panel/DiD designs. Including a lag controls for persistence — but it also introduces **Nickell bias** with fixed effects.

---
What are AR lags in causal inference?
ref: https://docs.google.com/presentation/d/1EBGXttOWOrXM-jD3ID0S3nKF8ZkzwrIRw0lQ_Iistyk/edit?usp=sharing

Value of $\mu$ in the previous period.
So it sounds very similar to auto regressive outcome generation (**1 at a time**).
So imagine $\mu$ is the treatment effect. So what this AR lags on outcome really mean is like using the outcome's **own history** as a control.

$$Y_{it} = \alpha + \rho \, Y_{i,t-1} + \beta \, D_{it} + \varepsilon_{it}$$

- $Y_{i,t-1}$ — the **lag**. Last period's outcome for the same unit
- $\rho$ — how **persistent** the outcome is. Near 1 = very sticky, near 0 = no memory
- $D_{it}$ — the treatment
- $\beta$ — the effect you actually care about

---
# Why you'd include it

Suppose you're measuring whether a new feature raised engagement. Engagement is **sticky** — a user who was highly engaged last month is likely to be highly engaged this month, feature or no feature.

If you leave that persistence out of the model, it has to go *somewhere*. And it will get absorbed into whatever correlated variable *is* in the model — very possibly your treatment. ~={red}You'd report an effect that is really just momentum.=~

The lag soaks up the persistence, so $\beta$ is estimated on what's genuinely *new* this period.

> [!SUCCESS] Core idea
> A lagged outcome is a **control for "where this unit already was."** It's the regression equivalent of comparing against yesterday rather than against the population average. ^lag-is-a-control

It's the same [[Markov Property]] assumption in a statistical costume: given last period, earlier periods add nothing. And the same shape as [[Auto-regressive models|autoregression]] in time series — the difference is purpose. Time series uses AR to **forecast**; causal inference uses it to **control**.

---
# The trap — you usually can't have both

> [!WARNING] Nickell bias
> Combining a **lagged dependent variable** with **unit fixed effects** produces a *biased* estimate — and the bias does not disappear with more units, only with more **time periods**.
>
> The mechanism: fixed effects are computed by demeaning each unit over time. But $Y_{i,t-1}$ is part of that mean, so the demeaned lag is mechanically correlated with the demeaned error term. That violates the core regression assumption, and the bias is of order $1/T$ — serious in the short panels typical of product analytics. ^nickell-bias

Which forces a real choice:

| You include | You control for | You risk |
|---|---|---|
| **Lag only** | persistence | unobserved unit-level confounders |
| **Fixed effects only** | stable unit differences | mistaking momentum for effect |
| **Both** | both | **Nickell bias** ⚠️ |

The standard defence is **Arellano-Bond** (difference GMM): difference away the fixed effects, then instrument the differenced lag with deeper lags, which are correlated with it but not with the differenced error.

> [!TIP] The practical framing
> Angrist & Pischke's advice is that lagged-outcome and fixed-effects models **bracket** the truth — one tends to over-control, the other under-control. Running both and reporting the range is more honest than picking whichever gives the answer you hoped for. 🎯

Related: if the outcome is genuinely non-stationary (a trend, a unit root), a lag with $\rho \approx 1$ can produce **spurious regression** — two unrelated trending series appearing strongly related. Check stationarity before trusting anything here.

---
# ⁉️
The setting where all of this lives — many units, many time periods — is [[Panel Regression]].
