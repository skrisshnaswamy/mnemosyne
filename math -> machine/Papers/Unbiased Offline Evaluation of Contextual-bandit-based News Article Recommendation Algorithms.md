---
title: "Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms"
authors: ["Lihong Li", "Wei Chu", "John Langford", "Xuanhui Wang"]
year: 2010
arxiv: "1003.5956"
url: https://arxiv.org/abs/1003.5956
priority: Must-Read
read_on: 2026-09-17
tags: [paper, theory]
---
## The Core Idea

You have a recommender that must pick one of $K$ news articles per user visit. You only ever learn whether the *shown* article was clicked. If you want to test a new algorithm, the honest way is to put it in front of real users — expensive, risky, and never repeatable because online click rates drift day to day. The common alternative in 2010 was to build a simulator of user behaviour and run the algorithm against it, which bakes in whatever modelling bias the simulator has.

The trick here is stupidly simple and provably exact: **log data from a uniformly random policy, then replay it and throw away every event where your algorithm disagrees with the log.**

Walk through the log one event at a time. Each event is (context $\mathbf{x}$, arm $a$ the random logger showed, payoff $r_a$). Ask your algorithm what it would pick given $\mathbf{x}$ and its history so far. If it says $a$, keep the event — add it to the algorithm's history, add $r_a$ to its score. If it says anything else, drop the event entirely and move on, leaving the algorithm's state untouched.

Why this is exact: because the logger picked arms uniformly, an event survives with probability exactly $1/K$, **independent of the context, the payoff, and the algorithm's state**. That independence is the whole paper. The surviving events are therefore distributed exactly as if they had been drawn fresh from the world. So the replay does not *approximate* an online run — it *is* an online run, on a shorter stream.

> [!NOTE] Replay evaluation
> Evaluate a bandit policy offline by streaming through uniformly-randomised logs and keeping only the events where the policy's choice matches the logged action. Unbiased because the match probability is a constant $1/K$. ^replay-evaluation

This is the estimator that makes bandit algorithms comparable the way UCI datasets made classifiers comparable. It also unlocked the Yahoo! Front Page dataset that half the contextual bandit literature was later benchmarked on. Related note: [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]] uses this evaluator; this paper is the proper analysis of it.

## The Methodology

**The setting.** Contextual bandit: at trial $t$ the world draws $(\mathbf{x}_t, r_{t,1},\dots,r_{t,K})$ i.i.d. from unknown $D$. The algorithm sees $\mathbf{x}_t$, picks $a_t$, observes only $r_{t,a_t}$. Target quantity is the **per-trial payoff**

$$g_{\mathsf A} = \frac{1}{T}\,\mathbf{E}_D\!\left[\sum_{t=1}^{T} r_{t,a_t}\right].$$

For news, payoff = click, so $g_{\mathsf A}$ is exactly click-through rate.

**Algorithm 1 (infinite stream).** Loop until you have collected $T$ valid events: pull the next logged event, keep it iff $\mathsf{A}(h_{t-1}, \mathbf{x}) = a$, append to history, add $r_a$ to $\hat G_{\mathsf A}$. Output $\hat G_{\mathsf A}/T$.

**Theorem 1.** For every possible history $h_T$,
$$\Pr_{\text{Policy\_Evaluator}(\mathsf A, S)}(h_T) = \Pr_{\mathsf A, D}(h_T).$$
Not "the mean matches" — the whole distribution over histories matches. So *any* statistic of the run (mean payoff, variance, regret curve) is unbiased. Cost: the number of logged events $L$ consumed to harvest $T$ valid ones has $\mathbf{E}[L] = KT$, and $L \le 2K(T + \ln(1/\delta))$ with probability $1-\delta$ (multiplicative Chernoff).

**Algorithm 2 (finite stream).** More practical version: sweep all $L$ logged events once, keep the matches, and divide by however many you got. Now $T$ is random with mean $L/K$, so the ratio $\hat G_\pi/T$ is no longer strictly unbiased — but it is tightly concentrated.

**Theorem 2** (for a *fixed* policy $\pi$, no history dependence): with probability $\ge 1-\delta$,
$$\left|\frac{\hat G_\pi}{T} - g_\pi\right| = O\!\left(\sqrt{\frac{K g_\pi}{L}\ln\frac{1}{\delta}}\right).$$
Proof is two Chernoff bounds plus a union bound: one on the denominator ($T \approx L/K$), one on the numerator ($\hat G_\pi \approx Lg_\pi/K$). Error shrinks as $1/\sqrt{L}$. Note $g_\pi$ inside the square root — for a rare event like a click ($g_\pi$ small), the relative error is better than the naive bound suggests. This sharpens Theorem 5 of *Exploration Scavenging* (Langford et al. 2008).

**Relaxing uniform logging.** If the logger is randomised but not uniform, you can rejection-sample down to uniform and the guarantees survive — at the cost of throwing away even more data. This is the seed of the [[Doubly Robust Policy Evaluation and Learning|inverse-propensity]] family: uniform logging just means every importance weight equals the same constant $K$, so it cancels.

**The data.** Yahoo! Front Page "Today Module", story position only. A **random bucket** (users assigned by cookie prefix) where the served article is drawn uniformly from the ~20-article editorial pool. About **40 million events**, 1–10 Nov 2009, for the unbiasedness study; a separate **4,000,000-visit** day (1 May 2009) for the variance study. Payoff $=1$ on story click. All CTRs reported as ratios to a hidden constant.

## Ablation Studies and Experiments

**1. Is it actually unbiased? (per-article and per-policy)**
A separate **serving bucket** ran a spatio-temporal CTR model and always showed the current winner article. They extracted that serving policy (the winner at every 5-minute slot, same 10 days) and replayed it on the random-bucket log. Per-article CTR (only articles with >20,000 online views, so the online number is near ground truth) and per-day aggregate CTR both line up almost exactly with the online bucket.

**The fix that mattered:** raw comparison violates i.i.d. In the serving bucket a winning article stays winning, so users see the *same* article over and over and click less each time; in the random bucket they see different articles. They removed this by counting **distinct views** — consecutive views of the same article by the same user collapse to one visit. Without that correction the comparison is contaminated.

**2. Convergence rate.** Error $|c - \hat c|$ plotted against number of valid events, for individual articles and for the overall policy, tracks the $1/\sqrt{T}$ curve predicted by Theorem 2.

**3. Variance across runs.** Three algorithms, parameters fixed at $\epsilon = 0.4$ and $\alpha = 1$; each event subsampled with probability 0.5; 100 independent runs each.

| algorithm | mean | std | max | min | std/mean |
|---|---|---|---|---|---|
| $\epsilon$-greedy (stochastic, context-free) | 1.2664 | 0.0308 | 1.3079 | 1.1671 | 2.4% |
| UCB (deterministic, context-free) | 1.3278 | 0.0192 | 1.3661 | 1.2812 | 1.4% |
| LinUCB (deterministic, contextual) | 1.3867 | 0.0157 | 1.4268 | 1.3491 | 1.1% |

One run is already reliable at this data scale. The two algorithms with known algorithm-specific deviation bounds ([[Finite-time Analysis of the Multiarmed Bandit Problem (UCB1)|UCB]], LinUCB) are the tighter ones.

**4. Does it rank *learning* algorithms correctly online?** Three $\epsilon$-greedy models, each in its own live cookie bucket (~2–3M views/day each), plus a fourth random bucket feeding both the online models' state updates and the offline replay:
- **EMP** — most popular article globally.
- **SEMP** — 18 user segments by age/gender, most popular within segment.
- **CEMP** — per-article logistic regression on user context.

Unlogged business rules (e.g. an article force-shown in a time window) multiply online CTR by an unknown daily factor, so they report the ratio $\rho_{\mathsf A} = g^{\text{offline}}_{\mathsf A} / g^{\text{online}}_{\mathsf A}$. If replay is faithful, $\rho$ should depend on the day but **not** on the algorithm.
- $\rho_{\text{EMP}}$ vs $\rho_{\text{SEMP}}$, 16 days (May 2009): regression slope **1.019**, residual std **0.0563**.
- $\rho_{\text{EMP}}$ vs $\rho_{\text{CEMP}}$, 18 days (May–Jun 2010): slope **1.113**, residual std **0.075**.

So the systemic distortion hits all models about equally, and offline *ordering* survives to online.

**What does not work — the honest negative result.** Unbiasedness does **not** imply concentration for a general, history-dependent bandit algorithm. Example 3: $K=2$, $r_{t,1}=1$ and $r_{t,2}=0$ always, $x\in\{0,1\}$ by a fair coin. Let $\mathsf A$ commit forever to arm 1 if $x_1 = 1$, else forever to arm 2. Then $g_{\mathsf A}=0.5$, but any single run returns $\hat G_{\mathsf A}/T \in \{0, 1\}$, so $|\hat G_{\mathsf A}/T - g_{\mathsf A}| \equiv 0.5$ no matter how large $T$ is. The estimator is still unbiased — the *average over runs* is right — but a single replay can be arbitrarily wrong for an algorithm whose whole trajectory hinges on its first observation. Fix: run replay many times and average, or restrict to algorithms with their own deviation bounds (epoch-greedy, UCB1, EXP3.P).

## Worth Remembering

- **Data efficiency is the headline cost.** You discard $(K-1)/K$ of the log. With $K=20$ articles you keep 5%. That is fine on 40M events, fatal with a large arm set or expensive data. This is the pressure that later pushed the field towards [[Doubly Robust Policy Evaluation and Learning|doubly-robust]] and self-normalised estimators that reuse every event.
- **You need the randomised bucket.** This method's only real requirement is that someone, at some point, agreed to serve uniformly random articles to a slice of traffic. If your organisation never ran an exploration bucket, you have nothing to replay. The authors flag learning from *non-random* logs as open, pointing to Strehl et al., *Learning from Logged Implicit Exploration Data* — which is where propensity estimation enters.
- **The arm set changes and they knowingly ignore it.** Articles enter and leave the pool hourly, so events are independent but *not* identically distributed, breaking the i.i.d. premise of both theorems. They state the theory does not cover it and lean on the empirical stability instead. Anyone applying replay to a catalogue with churn inherits this gap.
- **The distinct-views correction is a real methodological finding**, not bookkeeping. Repeat exposure suppresses clicks, and a deterministic serving policy generates far more repeat exposure than a random one. Any offline/online reconciliation on a feed product will hit the same thing.
- **Ratios, not absolutes.** They could only validate online agreement after dividing out an unmeasured daily business-rule factor. Replay gives you trustworthy *comparisons* between algorithms; the absolute CTR it reports is in a world without your production guardrails.
- Connection worth holding: this is [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives|off-policy evaluation]] with the temporal credit assignment deleted. One step, no bootstrapping, no distribution shift compounding over a horizon — which is exactly why an estimator this crude can be exactly unbiased here and hopeless in full RL.

## Links
Related: [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]] · [[Doubly Robust Policy Evaluation and Learning]] · [[Counterfactual Risk Minimization]] · [[Counterfactual Reasoning and Learning Systems]] · [[Finite-time Analysis of the Multiarmed Bandit Problem (UCB1)]] · [[A Tutorial on Thompson Sampling]] · [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[Unbiased Learning-to-Rank with Biased Feedback]] · [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Uncertainty]] · [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)]]

New topics worth writing: Exploration scavenging, Inverse propensity scoring, Self-normalised importance sampling, Rejection sampling for off-policy evaluation, Learning from logged implicit exploration data, Multiplicative Chernoff bound, Exploration buckets in production systems, Epoch-greedy algorithm
