---
title: "Trustworthy Online Controlled Experiments: Five Puzzling Outcomes Explained (KDD)"
authors: ["Kohavi et al."]
year: 2012
url: https://ai.stanford.edu/~ronnyk/puzzlingOutcomesInControlledExperiments.pdf
priority: Good-To-Read
read_on: 2026-09-25
tags: [paper, rl]
---
## The Core Idea

Running an A/B test is easy. Believing the answer is hard.

This paper is five war stories from Bing and MSN, each one a result that looked real, got acted on (or nearly did), and turned out to be an artefact. The value is not any single trick — it is the catalogue of ways a correctly-implemented experiment platform still lies to you.

The five, in one line each:

1. **A bug that broke search results made both headline business metrics go up.** Distinct queries per user rose over 10%, revenue per user rose over 30%. If those were your [[Controlled experiments on the web- survey and practical guide (DMKD)#^oec|OEC]], you would ship the bug.
2. **Adding a small delay on click made clicks go up.** Not because users clicked more — because the tracking beacon had more time to reach the server before the browser killed the request.
3. **The first few days of an experiment always look like a trend.** They are not a trend. They are a wide confidence interval shrinking.
4. **Running longer does not always buy statistical power.** For metrics like sessions/user, the confidence interval on the percent change stays roughly the same width as days pass.
5. **A previous experiment can contaminate the next one on the same users** — for up to three months.

Why this did not exist before: the statistics of randomised trials were settled in the 1920s by Fisher. But offline trials recruit a fixed cohort, run once, and stop. Online, users arrive continuously, the same user buckets get reused for experiment after experiment, and the measurement is done by JavaScript running inside a browser that is actively trying to navigate away. None of those three facts are in the textbook.

What it unlocks: a set of named failure modes you can check for, plus the single most useful operational habit in the paper — **run A/A tests continuously**.

> [!NOTE] A/A test
> Same experiment machinery, same randomisation, but both groups get the *identical* experience. The true effect is zero by construction, so any "significant" result is your platform misbehaving. With a 95% confidence level you should reject the null about 5% of the time — no more. ^aa-test

## The Methodology

There is no model and no algorithm here. The method is: notice an anomaly, spend person-weeks on the root cause, generalise.

**Setup.** Users are randomly split, persistently (same user gets the same variant across visits), by hashing a user ID. The experimental unit is the user; metrics are computed per user and averaged. Statistical significance is at 95% throughout. See [[AB Testing]] for the base machinery.

### 1 · The OEC problem, worked out

Bing's executive-level goals were query share and revenue per search. The paper decomposes monthly distinct queries:

$$\text{queries/month} = \frac{\text{users}}{\text{month}} \times \frac{\text{sessions}}{\text{user}} \times \frac{\text{distinct queries}}{\text{session}}$$

Now take each term separately:

- **Users/month** — fixed by the experiment design. In a 50/50 split both arms have the same count by construction. Useless as an OEC. (If it *isn't* roughly equal, that is a bug signal — what is now called sample ratio mismatch.)
- **Queries/session** — should be *minimised*, subject to the task succeeding. More queries per session usually means the user did not find the answer. But pushing it down can also mean abandonment. Ambiguous in both directions.
- **Sessions/user** — the one you actually want to increase. Happy users come back.

So the metric the executives track (total queries) and the metric an experiment should optimise (sessions/user) point in *opposite* directions on one of the three terms. Breaking search quality raises queries/session and raises ad clicks, because degraded organic results make ads relatively more relevant. Short-term revenue up, lifetime value down. This is exactly the shape of [[Reward Hacking#^goodharts-law|Goodhart's law]] — the measure stops being the goal.

Verdict in the paper: **do not use distinct queries or revenue per user as a search OEC without an engagement constraint.** Bing's OEC centres on sessions/user.

### 2 · Click tracking

Most sites log clicks with a **web beacon** — a 1×1 pixel image requested from a logging server when the user clicks.

The problem: Chrome, Firefox and Safari aggressively cancel in-flight requests when you navigate away. Losses on Safari were sometimes **over 50%**. Internet Explorer, for backwards-compatibility reasons, keeps executing image requests after navigation — so IE loses almost nothing.

Three experiments got fooled by this:

- Adding JavaScript on click (to write a session cookie before navigating) → measured clicks went up. It just delayed navigation, so more beacons landed.
- Changing the MSN Hotmail link to open in a new tab → measured clicks went up. No navigation away means no cancelled beacon.
- Making a Bing related-search update the page in place instead of navigating → measured feature usage jumped, but total searches did not rise proportionally.

Until March 2010, several Microsoft sites used a **2-second timeout** waiting for the beacon, adding about **400ms average delay** to every click. Delays of a few hundred milliseconds are known to cost real revenue.

### 3 · Why early results look like trends

The standard error of a mean shrinks like $1/\sqrt{n}$. Early in an experiment $n$ is small, so the daily cumulative delta bounces around wildly, then settles.

Concretely, under a true effect of **zero**:

- Day 1 has a **67%** chance of landing outside the final (day-21) 95% confidence bound.
- Day 2 has a **55%** chance.

And because the series is *cumulative*, consecutive points are autocorrelated — day 4's number contains days 1–3. So a random large negative on day 1 decays smoothly towards zero, which to a human eye is a straight line heading upward. The figure in the paper that "trends" from $-1.0\%$ to $-0.3\%$ over four days is from an **A/A test**. The true effect was zero.

### 4 · Why longer ≠ more power

The width of the 95% CI on the *percent* change is roughly

$$\text{CI width} \;\propto\; \frac{\mathrm{CV}}{\sqrt{n}}, \qquad \mathrm{CV} = \frac{\sigma}{\mu}$$

where CV is the **coefficient of variation** — standard deviation divided by mean.

The textbook assumes i.i.d. samples, so CV is a property of the distribution and does not change with $n$. Online, it does. Over 31 days of real Bing data, the standard deviation of cumulative sessions/user grew **faster** than the mean, so CV rose. The ratio $\mathrm{CV}/\sqrt{n}$ stayed flat to within 10% across the whole month.

Why: modelling counts as Poisson would give $\mathrm{CV} = 1/\sqrt{\lambda}$, which *falls* as the mean rises — the opposite of what they see. Negative Binomial is the better fit (citing Rosset & Borodovsky). On top of that, cookie churn and cookie birth inject extra variance into any user-keyed metric.

Practical consequence: to detect effects on sessions/user, **add more users per day**, do not just wait longer.

### 5 · Carryover

Bing, Google and Yahoo all used a **bucket system**: hash users into buckets once, then assign buckets to experiments. Buckets get recycled between experiments, and the hash is only re-randomised infrequently.

So the users who got hammered by last month's bad experiment are still sitting together in the same bucket when the next experiment starts. Their depressed behaviour becomes a fake treatment effect.

Measured duration:

| Case | Setup | Carryover lasted |
|---|---|---|
| Normal experiment | 7-day A/A, then 47-day A/B, then monitored | ~3 weeks after the experiment ended |
| Bug exposing users to a very bad experience | — | still not recovered after **3 months** |

**Fix: two-level bucketing.** The top level picks which users are in the experiment at all. The second level does a *per-experiment* hash, with a fresh seed each time, to assign treatment. Because the second hash is independent of history, any carryover gets mixed evenly into both control and treatment and cancels.

The cost: you lose the shared control. Every experiment needs its own control arm, so carryover contaminates both arms equally.

The bonus: **retrospective A/A**. Since the second-level split is independent of anything that happened before, you can take the days *before* the experiment started, apply the split retroactively, and check it comes out null. If it shows $p < 0.2$ on a key metric, change the hash seed and retry — all without spending calendar time.

## Ablation Studies and Experiments

There is no benchmark table. The evidence is diagnostic. Still, several of the analyses function as ablations:

**The A/A control is the ablation.** The trending-effect example (§3.3) is only convincing because the same graph, extended to 13 days, flattens at zero — and because they knew it was A/A all along. Without that, the "primacy effect" story is unfalsifiable.

**The browser split is the ablation for click tracking.** IE does not cancel beacons; other browsers do. So: if a click-metric win shows up only in non-IE traffic, it is instrumentation, not behaviour. This is a reusable diagnostic — segment by browser and look for effects that should not be browser-dependent.

**The CV decomposition is the ablation for power.** Plotting mean, standard deviation and $\sqrt{n}$ separately over 31 days shows *which* term breaks the textbook assumption. It is the standard deviation growing faster than the mean, not the sample count failing to grow.

**Re-running at larger scale is what killed the carryover result.** The original experiment showed highly significant movement in metrics *unrelated to the change*. Re-run on a bigger sample, most effects vanished. That "unrelated metrics moved" signature is the tell.

### What did not work

- **Poisson for count metrics.** Explicitly rejected. Poisson forces mean = variance, and its CV falls as the mean grows — contradicted by their data.
- **The primacy-effect hypothesis, almost always.** They could not find **a single experiment** where a statistically significant result in one direction later became statistically significant in the other. The common pattern is the opposite: significant-negative gets *more* negative over time, significant-positive gets *more* positive. So extending an experiment that is significantly negative after two weeks is a waste — fail fast.
- **Linear extrapolation of the first four days.** The instinct that produced the whole §3.3 story. The authors admit they were fooled by this repeatedly themselves, and name [[Recommending What Video to Watch Next- A Multitask Ranking System (RecSys)|confirmation bias]]'s cousin: when you built the feature, you suppress the early negative and build a story about the trend.
- **A/A tests as a carryover fix alone.** They work as *detection*, but when one fails you lose the bucket's capacity until re-randomisation, and re-randomising means stopping every experiment sharing that bucket line.

## Worth Remembering

**Hit rates for ideas, collected in one place.** Only about **one third** of ideas tested at Microsoft improved the metric they targeted. Google ran ~12,000 randomised experiments in 2009 with ~**10%** leading to business changes. Netflix considers **90%** of what they try to be wrong. Avinash Kaushik: "80% of the time you/we are wrong about what a customer wants." This is the economic argument for the whole platform — and for why trustworthiness matters more than throughput. Reversing one wrong decision at Bing can pay for a team of analysts.

**The MSN Real Estate example.** Six widget designs, OEC = average revenue per user, winner gave almost **10%** more revenue via higher clickthrough. Included as the "experiments obviously work" opener, but note that the OEC was clean precisely because the site's only job was referral clicks.

**The generalisable diagnostic rules:**
- Effects that differ by browser → suspect instrumentation, not behaviour.
- Metrics *unrelated* to your change moving significantly → suspect carryover or a bad split.
- Arm sizes deviating from design → suspect a bug.
- A week is the minimum duration, for day-of-week effects — not for power.

**Limitations the authors are honest about.** Sessions/user is a surrogate for tasks/user, which is what they actually want but cannot measure. The whole OEC rests on an unverifiable assumption: that a better experience raises users/month, the one term a controlled experiment structurally cannot observe. And measuring long-term value with short-term experiments remains unsolved — this paper only shows you which short-term proxies are actively harmful.

**Practical caveats if you are building this.** The two-level bucket fix costs you shared controls, which roughly doubles the traffic needed per experiment. The fixed-CV problem means your power calculator is probably wrong for engagement metrics — measure CV empirically per metric, per duration, rather than assuming it is constant. And the $1/\sqrt{n}$ variance story is exactly what [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)|CUPED]] attacks from the other side: if you cannot shrink the CI by waiting, shrink it by removing pre-experiment variance.

**Connections.** The early-peeking problem in §3.3 is the informal version of the argument that makes [[Always Valid Inference- Bringing Sequential Analysis to A-B Testing|always-valid p-values]] necessary — Kohavi's answer here is "don't look", the sequential answer is "make looking safe". Carryover is the temporal sibling of interference, handled by design in [[Design and Analysis of Switchback Experiments]]. The OEC/Goodhart problem is the same structure as reward design in [[Reward Hacking]] and as the logging-policy contamination that [[Counterfactual Reasoning and Learning Systems]] and [[Off-Policy Evaluation]] deal with. And the "instrumentation is not as precise as we would like" theme sits beside [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]].

**Open question worth carrying.** If carryover can persist for three months, what does that mean for any long-running holdback, or for offline replay evaluation on logged data collected while other experiments were live? The paper does not touch it.

The closing line is the one to keep: *generating numbers is easy; generating numbers you should trust is hard.*

## Links
Related: [[AB Testing]] · [[Controlled experiments on the web- survey and practical guide (DMKD)]] · [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)]] · [[Always Valid Inference- Bringing Sequential Analysis to A-B Testing]] · [[Design and Analysis of Switchback Experiments]] · [[Reward Hacking]] · [[Counterfactual Reasoning and Learning Systems]] · [[Off-Policy Evaluation]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Uncertainty]]

New topics worth writing: Sample ratio mismatch, Coefficient of variation and experiment power, Negative binomial modelling of count metrics, Web beacon instrumentation and click loss, Two-level bucketing and localized re-randomisation, Novelty and primacy effects, Surrogate metrics and long-term value
