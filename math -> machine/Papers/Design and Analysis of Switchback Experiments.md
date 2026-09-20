---
title: "Design and Analysis of Switchback Experiments"
authors: ["Iavor Bojinov", "David Simchi-Levi", "Jinglong Zhao"]
year: 2020
arxiv: "2009.00148"
url: https://arxiv.org/abs/2009.00148
priority: Must-Read
read_on: 2026-09-17
tags: [paper, theory]
---
## The Core Idea

A **switchback experiment** turns one unit — a whole city, a whole catalogue — on and off over time, and compares the on-periods to the off-periods. You use it when you cannot split users into A and B groups, because the groups leak into each other. If Lyft gives half its drivers new prices, the other half feel it too: the same riders, the same roads, the same supply pool. So instead of splitting people, you split *time*.

Everyone in the tech industry already ran these. Nobody knew how to run them *well*. Two knobs were picked by gut feel: how often do you flip the coin, and how biased is the coin? This paper solves both exactly.

The obstacle is the **carryover effect**: a treatment you applied at 2pm is still bending the outcome at 3pm. Surge pricing pulls drivers into a neighbourhood; that supply is still there an hour later. So a period right after a switch is contaminated — you are measuring a mixture, not a clean treatment period.

> [!NOTE] Carryover effect of order $m$
> The outcome at time $t$ depends on the treatments at $t-m, \dots, t$, and on nothing earlier. $m$ is the *number of periods* the past keeps mattering. ^carryover-order

The answer is counter-intuitive and clean. If the carryover lasts $m$ periods and your experiment runs $T = nm$ periods, the best design is:

- flip a **fair** coin, always $1/2$, never anything else;
- flip it at periods $\{1,\ 2m+1,\ 3m+1,\ \dots,\ (n-2)m+1\}$.

Read that as: **a long block of $2m$ at the start, then switch every $m$ periods, then a long block of $2m$ at the end.** Not every period (too much contamination), not every $m+1$ periods (the "obvious" choice, and it loses). The two fat end-blocks are the surprise — they exist because the first and last stretches have no neighbour to contaminate them, so you spend your switches in the middle where they buy more.

The second unlock is *how* they got there. No model of the outcome is assumed at all. The potential outcomes are treated as fixed unknown numbers; the only randomness in the world is the coin. That is the **design-based** view, and it means the causal conclusion cannot be wrong because you mis-specified a regression. The price of that generality is that you must defend against the worst case, so the design is chosen by a minimax rule.

## The Methodology

**Setup.** One unit, $T$ periods. At each $t$ you assign $W_t \in \{0,1\}$. The whole history is $\bm{W}_{1:T}$. For every possible path there is a potential outcome $Y_t(\bm{w}_{1:T})$ — a fixed number, not a random variable.

Two assumptions cut this down:

1. **Non-anticipating.** $Y_t$ does not depend on future treatments. Free, because you control the coin.
2. **$m$-carryover.** $Y_t$ depends only on $\bm{w}_{t-m:t}$. So we write $Y_t(\bm{w}_{t-m:t})$.

**The estimand.** The average lag-$p$ effect — what happens if you deploy the policy *permanently*:

$$\tau_p(\mathbb{Y}) = \frac{1}{T-p}\sum_{t=p+1}^{T}\left[Y_t(\bm{1}_{p+1}) - Y_t(\bm{0}_{p+1})\right]$$

$m$ is the truth; $p$ is what the experimenter *believes*. They coincide in the design section and diverge later.

**The estimator.** Horvitz–Thompson — inverse-propensity weighting, exactly the machinery in [[Unbiased Learning-to-Rank with Biased Feedback|IPS]] and [[Doubly Robust Policy Evaluation and Learning]]:

$$\widehat{\tau}_p = \frac{1}{T-p}\sum_{t=p+1}^{T}\left\{ Y_t^{\mathsf{obs}}\frac{\mathbb{1}\{\bm{w}_{t-p:t}=\bm{1}_{p+1}\}}{\Pr(\bm{W}_{t-p:t}=\bm{1}_{p+1})} - Y_t^{\mathsf{obs}}\frac{\mathbb{1}\{\bm{w}_{t-p:t}=\bm{0}_{p+1}\}}{\Pr(\bm{W}_{t-p:t}=\bm{0}_{p+1})} \right\}$$

In plain words: only periods where the last $p+1$ assignments were *all treatment* or *all control* contribute. Everything else is a contaminated period and drops out. Each surviving period is re-weighted by one over its probability. This is unbiased whenever both probabilities are non-zero.

**Regular switchback experiments.** The class they optimise over. Pick randomisation points $\mathbb{T} = \{t_0=1 < t_1 < \dots < t_K\}$ and probabilities $\mathbb{Q} = (q_0,\dots,q_K)$. At each $t_k$ you flip a coin with bias $q_k$; the result holds for all periods until $t_{k+1}$. Note this is *not* adaptive — the whole path is drawn before the experiment starts, unlike a [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)|contextual bandit]].

**The optimisation.** Loss is squared error; risk is its expectation over the coin, which — because the estimator is unbiased — *is* the variance:

$$r(\eta_{\mathbb{T},\mathbb{Q}}, \mathbb{Y}) = \mathbb{E}\left[(\widehat{\tau}_p - \tau_p)^2\right]$$

They solve
$$\min_{\mathbb{T},\mathbb{Q}}\ \max_{\mathbb{Y}}\ r(\eta_{\mathbb{T},\mathbb{Q}},\mathbb{Y})$$
with one extra assumption: outcomes are bounded, $|Y_t(\bm{w})| \le B$. You do not need to know $B$ — the optimal design does not depend on it.

Three steps solve it.

*Step 1 — find the adversary.* The worst possible world is $Y_t(\bm{1}_{m+1}) = Y_t(\bm{0}_{m+1}) = \pm B$ for all $t$: **the treatment does nothing, and every outcome is pinned to the boundary.** All the risk terms are quadratic with positive coefficients, so the extremes dominate. The minimax collapses into a plain minimisation.

*Step 2 — fair coins win.* The risk is a sum of terms shaped like
$$f(q_1,\dots,q_n) = \frac{1}{\prod q_i} + \frac{1}{\prod (1-q_i)} \ \ge\ 2^{n+1}$$
with equality only at $q_i = 1/2$. Both the "all treated" and "all control" propensities appear in the denominator, and you cannot shrink one without blowing up the other. Symmetry in, symmetry out.

*Step 3 — place the switches.* With $q=1/2$ fixed, the risk has a closed form and the problem becomes a subset selection:

$$\min_{\mathbb{T}\subset[T]}\left\{ 4\sum_{k=0}^{K}(t_{k+1}-t_k)^2 + 8m(t_K - t_1) + 4m^2K - 4m^2 + 4\sum_{k=1}^{K-1}\left[(m - t_{k+1}+t_k)^+\right]^2 \right\}$$

Each term is a force. $\sum (t_{k+1}-t_k)^2$ punishes gaps that are *too long* (too few independent observations). $\sum [(m - t_{k+1}+t_k)^+]^2$ punishes gaps *shorter than $m$* (contamination — the surviving data thins out). $8m(t_K - t_1)$ punishes starting too early and ending too late. Relax to continuous, and the optimum happens to land on integers: end blocks of $2m$, middle blocks of $m$.

**The design table.** $T=12$, $m=2$ gives $\mathbb{T}^* = \{1,5,7,9\}$:

| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ✓ | – | – | – | ✓ | – | ✓ | – | ✓ | – | – | – |

**Inference, route 1 — exact.** Test Fisher's sharp null: *no effect at any time point*. Because the null says every assignment path would have produced the same observed outcomes, you can resample paths from the known design, recompute $\widehat{\tau}^{[i]}$ with the observed $Y$'s held fixed, and get
$$\widehat{p}_{\mathsf{F}} = \frac{1}{I}\sum_{i=1}^{I}\mathbb{1}\left\{|\widehat{\tau}^{[i]}| > |\widehat{\tau}|\right\}$$
Exact for large $I$. No distributional assumption. Confidence intervals need test inversion, which is painful.

**Inference, route 2 — asymptotic.** Test Neyman's weak null $\tau_m = 0$. Group the periods into blocks $\bar{Y}_k = \sum_{t=(k+1)m+1}^{(k+2)m} Y_t$. The exact variance is derivable but contains cross-products of potential outcomes you never observe jointly. So they bound it and estimate the bound:

$$\widehat{\sigma}^2_{\mathsf{U}} = \frac{1}{(T-m)^2}\left\{8(\bar{Y}_0^{\mathsf{obs}})^2 + \sum_{k=1}^{n-3}32(\bar{Y}_k^{\mathsf{obs}})^2\,\mathbb{1}\{W_{km+1}=W_{(k+1)m+1}\} + 8(\bar{Y}_{n-2}^{\mathsf{obs}})^2\right\}$$

This is unbiased for the *bound*, hence conservative for the truth. The CLT is a finite-population one: the block variables are **1-dependent** (each block only touches its neighbour), so a $\phi$-dependent CLT applies and
$$\frac{\widehat{\tau}_m - \tau_m}{\sqrt{\mathsf{Var}(\widehat{\tau}_m)}} \xrightarrow{D} \mathcal{N}(0,1)$$
with one technical condition: the variance must not be negligible, $\mathsf{Var}(\widehat{\tau}_m) \ge \Omega(n^{-1})$ — i.e. no handful of periods carries all the variance.

**Finding $m$ when you do not know it.** Run two experiments on two comparable units (or two far-apart time epochs), one designed for $p_1$, one for $p_2 > p_1$. Under $H_0: m \le p_1$ both estimate the same thing, so
$$z = \frac{|\widehat{\tau}_{p_1} - \widehat{\tau}_{p_2}|}{\sqrt{\widehat{\sigma}^2_{p_1} + \widehat{\sigma}^2_{p_2}}}$$
should be standard normal. Reject and you know $m > p_1$. Wrap a search around it.

## Ablation Studies and Experiments

The simulation model is linear additive carryover:
$$Y_t(\bm{w}_{1:t}) = \mu + \alpha_t + \delta^{(1)}w_t + \delta^{(2)}w_{t-1} + \dots + \epsilon_t$$
with $\mu=0$, $\alpha_t = \log t$, $\epsilon_t \sim \mathcal{N}(0,1)$, $T=120$, $m=2$, 100,000 replications.

**Three designs compared.**
- $\mathbb{T}^* = \{1,5,7,\dots,117\}$ — the optimal design.
- $\mathbb{T}^{\mathsf{H1}} = \{1,2,3,\dots,120\}$ — flip every period. **This is what most practitioners actually do.**
- $\mathbb{T}^{\mathsf{H2}} = \{1,4,7,\dots,118\}$ — the "intuitive" design, blocks of exactly $m+1=3$.

**Worst-case risk.** $r(\mathbb{T}^*) = 26.78$, $r(\mathbb{T}^{\mathsf{H1}}) = 33.67$, $r(\mathbb{T}^{\mathsf{H2}}) = 27.85$. Closed-form predictions: 26.67, 33.96, 27.81. Theory matches simulation.

**Risk under the actual model.** Sweeping $\delta^{(1)},\delta^{(2)},\delta^{(3)} \in \{1,2\}$:

| $\delta^{(1)},\delta^{(2)},\delta^{(3)}$ | $\tau_2$ | $r(\mathbb{T}^*)$ | $r(\mathbb{T}^{\mathsf{H1}})$ | $r(\mathbb{T}^{\mathsf{H2}})$ |
|---|---|---|---|---|
| 1,1,1 | 3 | **7.96** | 10.22 | 8.11 |
| 1,1,2 | 4 | **9.57** | 12.39 | 9.74 |
| 1,2,2 | 5 | **11.34** | 14.81 | 11.52 |
| 2,2,2 | 6 | **13.28** | 17.48 | 13.47 |

The naive every-period design is **28–32% worse**. The intuitive design is only **1–2% worse**. The optimal design wins even though it was derived for a worst case that does not hold here.

That 1–2% gap is the most useful number in the paper. It says: **the expensive mistake is switching too often, not getting the block length exactly right.** And since $\mathbb{T}^{\mathsf{H2}}$ (which uses $p = m$ blocks, i.e. slightly conservative) is nearly optimal while $\mathbb{T}^{\mathsf{H1}}$ (aggressive, $p$ effectively 0) is terrible, when you are unsure of $m$, **guess high**.

**Type I / Type II error.** All three designs sit at the 0.05 nominal level for Type I — no design is cheating. The optimal design has the lowest Type II error almost everywhere; the gaps shrink as $T/m$ grows. So the whole benefit is statistical power, not validity.

**Misspecified $m$.** Three cases, all with true $m=2$:

| | $\tau_p$ | $\mathbb{E}[\widehat{\tau}_p]$ | $\mathsf{Var}$ | $\mathbb{E}[\widehat{\sigma}^2_{\mathsf{U}}]$ |
|---|---|---|---|---|
| $p=2$, $\delta=3$ | 9 | 9.028 | 20.10 | 24.25 |
| $p=3$, $\delta=3$ | 9 | 9.012 | 30.10 | 36.32 |
| $p=1$, $\delta=3$ | (6) | 6.037 | 10.14 | 10.92 |

Over-specifying ($p=3 > m$) keeps the estimand and the unbiasedness — you pay **50% more variance** and nothing else. Under-specifying ($p=1 < m$) is worse: the estimator is now unbiased for a *different quantity*, the "$m$-misspecified lag-$p$ effect", which pads the missing treatments with whatever was actually observed. It is still a meaningful causal contrast, but not the one you asked for. Asymptotic normality survives both cases.

The variance bound is tight: 20.10 vs 24.25, 10.14 vs 10.92. Conservative but not wasteful.

**What did not work.**

*Heavy tails break the CLT at practical horizons.* Swap $\epsilon_t \sim \mathcal{N}(0,1)$ for Student's $t$ with **1 degree of freedom** (infinite variance) and at $T=120$ the randomisation distribution is visibly non-normal. You need $T=1200$ to recover normality. The bounded-outcome assumption is doing real work here, and metrics with fat tails — revenue, session length — will need much longer experiments. The variance bound is also a worse approximation under heavy tails.

*Identifying $m$ is brutally expensive.* With $\delta=3$, $T=120$: testing $H_0: m\le 2$ gives $\widehat{p}=0.902$ (correctly fails to reject), testing $H_0: m\le 1$ gives $\widehat{p}=0.350$ — directionally right but nowhere near significant. Growing $T$: at $T=210$, $\widehat{p}=0.182$; at $T=1020$, $\widehat{p}=0.163$; only at $T=2010$ does it reach $\widehat{p}=0.037$. **Their own recommendation is $T/p > 100$** just to distinguish two candidate values of $m$. Most teams do not have 2000 hours of experiment budget to spend before the real experiment starts.

*The optimal design is wrong for instantaneous effects.* In the appendix they also estimate $\tau_{p,q}$ — treatment applied for the last $q+1$ periods only. At $q=0$ (instantaneous effect), the every-period design $\mathbb{T}^{\mathsf{H1}}$ has risk 2.35–2.56 and the optimal design has 2.66–3.00, while $\mathbb{T}^{\mathsf{H2}}$ is far worse at 3.71–4.20. Because $\mathbb{T}^*$ rarely produces a $(0,\dots,0,1)$ pattern, most periods have zero probability of the needed path and contribute nothing. At $q=1$ and $q=2$ the optimal design wins again. So the design is optimal *for the estimand it was designed for*, and you should choose your estimand before your design.

*Exact vs asymptotic p-values.* Exact is always slightly smaller — e.g. 0.101 vs 0.138, 0.231 vs 0.269. Two reasons stack: Fisher's sharp null is stronger than Neyman's, and the asymptotic test uses a conservative variance. Expected, not a defect.

## Worth Remembering

**Practical recipe, in order.** (1) Pick the metric. (2) Pick the period granularity — as long as one period is *shorter than* the carryover and divides it evenly, the granularity does not matter; the optimal design randomises on the same wall-clock schedule either way. 0.5-hour periods with $m=4$ and 1-hour periods with $m=2$ both randomise every two hours. Setting periods *longer* than the carryover loses data irrecoverably. (3) Pick $p$ from domain knowledge, erring high. (4) Pick $T = nm$ by reading off the rejection-rate curve — this is power analysis in a different costume. (5) Sample the path.

**The $m=0$ vs $m=1$ robustness.** With no carryover, $\mathbb{T}^* = \{1,2,\dots,T\}$. With $m=1$, $\mathbb{T}^* = \{1,3,4,\dots,T-1\}$. Nearly identical. So the naive every-period design is fine *if and only if* the carryover really is nothing.

**Three limitations the authors admit.**
- If $m$ is comparable to $T$, the estimator is still unbiased but the variance is so large that no meaningful inference is possible. Generality has a price; if you have a good outcome model, use it.
- The design is fixed before launch. No adaptive coin-flipping, because adaptivity needs assumptions like time-homogeneity that they refuse to make. This is the wall between this paper and the [[A Tutorial on Thompson Sampling|Thompson sampling]] / [[Finite-time Analysis of the Multiarmed Bandit Problem (UCB1)|UCB]] world — a live switchback with adaptive allocation is an open problem.
- Only one estimand is covered. Want something else? Redo the derivation.

**Connections.** The Horvitz–Thompson machinery is the same IPS skeleton as [[Recommendations as Treatments- Debiasing Learning and Evaluation]] and [[Counterfactual Risk Minimization]], but here the propensities are *known exactly and by design* — you built them — so the usual worry about extreme or estimated weights vanishes. Under the optimal design every propensity is a power of $1/2$. This is the single cleanest IPS setting you will meet. The design-based stance also puts this in the same family as [[Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms|replay evaluation]]: randomness comes from the logging/assignment mechanism, not from a model.

**Practical caveat for the sceptic.** The $m$-carryover assumption is a hard cliff — effects matter fully for $m$ periods then vanish entirely. Real carryover decays. Over-specifying $p$ is the safe read of that cliff, and the 50%-variance cost is the premium you pay for the insurance. The authors also suggest running the optimal design on several units (several cities) and pooling, which is how you buy back the power that switchbacks inherently lack — precision scales with the number of *switches*, not the number of periods.

**The thing that will bite you.** $T/m \ge 4$ is required for the closed-form design; $T/m > 100$ is required for the $m$-identification procedure to distinguish anything. If $T$ is not a multiple of $m$, either solve the integer program numerically (no closed form exists for these "imperfect cases") or throw away a few periods and use the clean formula.

## Links

Related: [[Controlled experiments on the web- survey and practical guide (DMKD)]] · [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)]] · [[Always Valid Inference- Bringing Sequential Analysis to A-B Testing]] · [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[Counterfactual Risk Minimization]] · [[Doubly Robust Policy Evaluation and Learning]] · [[Unbiased Learning-to-Rank with Biased Feedback]] · [[Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms]] · [[Counterfactual Reasoning and Learning Systems]] · [[Double-Debiased Machine Learning for Treatment and Structural Parameters]] · [[Auto-regressive lags]] · [[Panel Regression]] · [[Recursive Partitioning for Heterogeneous Causal Effects]] · [[Markov Property]]

New topics worth writing: Horvitz–Thompson estimator, design-based (finite-population) inference, Fisher sharp null and randomisation tests, minimax experimental design, interference and SUTVA violation, marketplace/two-sided experimentation, finite-population central limit theorems, $\phi$-dependent sequences, cluster and time-based randomisation, Neyman vs Fisher nulls

Tags: `experimentation` `causal` `theory`
