---
title: "Controlled experiments on the web: survey and practical guide (DMKD)"
authors: ["Kohavi", "Longbotham", "Sommerfield & Henne"]
year: 2009
url: https://ai.stanford.edu/~ronnyk/2009controlledExperimentsOnTheWebSurvey.pdf
priority: Must-Read
read_on: 2026-09-17
tags: [paper]
---
## The Core Idea

The web made a very old idea cheap: split your users at random, show half one thing and half another, and measure what happens. Randomisation is what buys you the word *because*. Every other factor — time of day, user type, marketing campaigns, the weather — gets spread evenly across both groups, so the only systematic difference left is the change you made. That is a causal claim, not a correlation.

The reason this paper matters is not the statistics — a t-test was not new in 2009. It is that this is the first end-to-end field manual for running these tests on a website: how to pick the metric, how big the sample must be, how to randomise users without breaking independence, where to put the assignment code, what the data pipeline looks like, and what organisational habits make the whole thing trustworthy or worthless.

The unifying claim, backed by war stories: **nobody can predict which variant wins.** Not designers, not VPs. The paper's own examples move by an order of magnitude in directions the experts got backwards. So the organisation should stop arguing and start measuring. The authors coined **HiPPO** — Highest Paid Person's Opinion — as the thing experiments replace.

> [!NOTE] Overall Evaluation Criterion (OEC)
> The single number that decides the experiment. Agreed **before** the test runs. It should be one metric, not a scorecard, so the organisation is forced to make the trade-offs once, up front, rather than re-litigating every launch. A good OEC contains long-term terms (repeat visits, lifetime value), not just clicks. ^oec

## The Methodology

**The basic loop.** Randomly assign each user to Control (A, the current site) or Treatment (B). Instrument everything. Compute the OEC per variant. Run a test.

$$t = \frac{O_B - O_A}{\hat{\sigma}_d}$$

$|t| > 1.96$ at 95% confidence and large samples → reject the null.

**Sample size.** This is the part most primers skip and the part the reader should memorise:

$$n = \frac{16\sigma^2}{\Delta^2}$$

$n$ is users *per variant*, $\sigma^2$ the variance of the OEC, $\Delta$ the effect you want to detect. The 16 gives 80% power at 95% confidence; swap 16 for 21 to get 90% power. A more conservative version for $r$ variants: $n = (4r\sigma/\Delta)^2$.

> [!NOTE] Statistical power
> The probability of correctly calling a real difference significant. Confidence controls false positives (Type I); power controls false negatives (Type II). You do not set power directly — it falls out of sample size, variance and effect size. ^statistical-power

**Three ways to shrink the standard error**, which is the whole game because $\text{Std-Err} = \hat\sigma/\sqrt{n}$:

1. **More users.** Linear in time, square-root in effect. Expensive.
2. **A lower-variance OEC.** Worked example: an e-commerce site with 5% conversion, $75 average basket, so $3.75 mean revenue per user and $\sigma = 30$. Detecting a 5% revenue change needs $16 \cdot 30^2/(3.75 \cdot 0.05)^2 \approx 409{,}000$ users per arm. Switch the OEC from revenue to *conversion rate* — a Bernoulli with $\sigma = \sqrt{p(1-p)}$ — and the same 5% detection needs under 122,000. A 3.3× cut, six weeks becomes two.
3. **Event-triggered filtering.** If you changed the checkout page, analyse only users who entered checkout. Everyone else is pure noise. Same example: 10% initiate checkout, 50% complete, so within that segment $p = 0.5$, $\sigma = 0.5$, and you need 6,400 *checkout starters* — 64,000 site visitors. Half of 122,000.

The squared $\Delta$ also explains a nice asymmetry: if you planned to detect a 1% change but a bug tanks the OEC by 20%, you find it in $1/400$ of the planned runtime. A two-week experiment surfaces an egregious bug in the first hour. That is the mathematical justification for **ramp-up**: start at 99.9/0.1, step through 0.5%, 2.5%, 10%, 50%, checking at each step, and auto-abort a treatment that is clearly losing.

**Confidence intervals.** Absolute effect is easy: $O_B - O_A \pm 1.96\hat{\sigma}_d$. Percent effect is not, because you are dividing by a random variable. With $CV_A = \hat\sigma_A/O_A$ and $CV_B = \hat\sigma_B/O_B$, Fieller's interval is

$$(\text{PctDiff}+1)\left[\frac{1 \pm 1.96\sqrt{CV_A^2 + CV_B^2 - 1.96^2 CV_A^2 CV_B^2}}{1 - 1.96\,CV_A^2}\right] - 1$$

Do not compute it if the CI for the denominator contains zero. Why bother: one real experiment had an absolute clickthrough effect of 0.00014, which means nothing to anyone, and a 12.85% relative effect, which means everything.

**Robots.** A bot that accepts cookies and never deletes them is a single "user" with 7,000 clicks an hour. It biases the mean *and* inflates $\sigma$, killing power. The authors saw A/A tests fail far more than 5% of the time because of this. Bots that reject cookies are harmless (they never get a user ID). Filter aggressively on user-agent lists plus heuristics on action counts per ID. JavaScript-triggered assignment naturally excludes most bots.

**Randomisation algorithm.** Three hard requirements: uniform assignment, *consistent* repeat assignment for the same user, and **zero correlation between concurrent experiments**. Two optional ones: monotonic ramp-up and manual override.

- *Pseudorandom + caching.* Works if the RNG is seeded once at server startup. Seed per request and adjacent requests share seeds — the authors found this created real two-way interactions between experiments in a published Visual Basic recipe. Needs state (database or cookie), which is expensive and hard to keep consistent across a server fleet, and makes ramp-up awkward.
- *Hash and partition.* Hash `user_id + experiment_id`, partition the output range. Stateless, but brutally sensitive to the hash. Testing five simulated experiments over one million sequential user IDs with chi-square interaction tests: **only MD5 was clean.** SHA256 needed a five-way interaction before correlating. The .NET string hash failed a two-way test. A "clever" optimisation — hash the experiment name and the user separately and XOR them — is catastrophic: if the top bits of the two experiment hashes match, users get identical assignments in both experiments; if not, exactly opposite ones.

**Assignment method** — four families, and the trade-off is intrusiveness versus power:

| Method | Intrusive | Hardware | Flexibility | Render-time hit |
|---|---|---|---|---|
| Traffic splitting (parallel fleets) | No | High | High | Low |
| Page rewriting (proxy edits HTML) | No | Mod–high | Moderate | High |
| Client-side (JS calls assignment service) | Moderate | Low | Low | High |
| Server-side (API call in the code) | High | Low | Very high | Very low |

Server-side is the only one that can experiment on **backend algorithms** — ranking, recommendations, search — because the decision happens where the logic lives. Its cost is code churn and clean-up risk. The fix Amazon used: push assignment into the content-management system, so the home page's slot metadata can be scheduled against an experiment and non-technical editors configure tests through a GUI. Instrument once, run tests forever.

A specific trap in client-side assignment: skipping the service call when the cookie says "Control" to save render time. This makes latency *correlated with variant*, which confounds the experiment.

**Data path.** Log the variant assignment on every raw observation, aggregate by experiment × variant × dimension, test. Three collection options; the authors declare a clear winner — **service-based collection**, because it centralises backend and client-side JavaScript events in one place. Local log files do not scale to near-real-time on large fleets; bolting assignment IDs onto an external analytics vendor precludes real-time analysis and needs code changes per experiment.

## Ablation Studies and Experiments

This is a survey, so the "experiments" are field results. They are worth keeping because they are the evidence that intuition fails.

**Doctor FootCare checkout.** Nine simultaneous UI changes in an "upgrade". The new page lost **90% of revenue**. The culprit was a single one of the nine: a *discount coupon code box*. Seeing it made people stop and wonder whether they were overpaying, and go hunting for a coupon instead of buying. Removing it lifted conversion 6.5% relative. This is also an argument against bundling changes — with nine at once you cannot attribute the loss.

**Microsoft Office help ratings.** Replacing a two-stage Yes/No widget with a single five-star widget. The stated goals were finer-grained feedback and better usability. Response rate **fell by ~90%**. A follow-up test showed a two-stage widget with the stars labelled "Not helpful → Very helpful" beat the single five-star widget by **2.2×** on response rate. And the finer granularity was a mirage — people picked 1 or 5 almost exclusively, because an article either solved your problem or it did not. They shipped yes/no/I-don't-know: slightly lower response than yes/no, but the extra option was worth it.

**MSN home page ads.** Adding three offer modules below the Shopping module would have earned tens of thousands of dollars per day. Run at 5% of US MSN home page users for 12 days. Clickthrough fell **0.38% relative, $p = 0.02$**. To compare that with ad money they priced a lost click two ways: the value the destination MSN property assigns to it, and the search-engine-marketing cost of re-buying that traffic. SEM was the higher number; the two were close enough to agree on. The lost clicks were worth more than the ad revenue. **Idea killed.**

**Behaviour-Based Search at Amazon.** Extending "people who bought X bought Y" to "people who *searched* for X bought Y", surfaced inside search results with no UI change. Strength: searching "24" stopped returning 24-inch towel bars and 24-month toddler clothing and started returning the DVD box set. Weakness: no semantic understanding, so "Sony HD DVD Player" surfaces Toshiba HD DVDs (Sony made Blu-Ray; searchers ended up buying Toshiba). The experiment settled it — **+3% revenue**, hundreds of millions of dollars.

**What did not work — MultiVariable Testing the classical way.** The authors tested five MSN home page factors and argue against the traditional fractional-factorial / Plackett–Burman designs that vendors were selling:

- The 8-run fractional factorial for five factors (Table 1) estimates all five main effects but **cannot estimate any two-factor interaction**, because every interaction is fully confounded with a main effect or another interaction. "No amount of effort at analysis or data mining will allow you to estimate these interactions." You need 16 runs to get all two-factor interactions.
- The 12-run Plackett–Burman **partially** confounds every two-factor interaction with main effects and other interactions — arguably worse, because it looks like you can estimate them.
- You cannot start until *all* five factors are code-complete. One delay delays everything.
- Some cells are simply bad experiences (enlarge the product image *and* add product detail → the buy box falls below the fold and sales drop).

Their replacement: since web experiments do not cost more per cell the way manufacturing does, **run each factor as an independent single-factor experiment, randomised independently, and you get a full factorial for free** — every interaction estimable, and any single factor can be switched off mid-flight without disturbing the others. If you do not care about interactions at all, run **overlapping experiments**: launch each factor the day it is ready. The authors' position, backed by van Belle, is that strong interactions are rarer than people fear.

The counter-intuitive supporting claim: **power does not fall as you add cells**, as long as you analyse via main effects pooled over all data rather than comparing each cell to Control. For a fixed user count, an 8-run MVT and an A/B test have the same power to detect a main effect. What *does* cost power is more levels per factor, and unequal splits.

**The 50/50 result.** Running Treatment at fraction $p$ multiplies your required runtime by $1/(4p(1-p))$. At 99/1 that is **25×**. A test needing 5 days at 50/50 needs 125 days at 99/1 — long enough that cookie churn starts contaminating the data. Ramp up for safety, then sit at 50/50.

**Speed is a confound.** Amazon: +100 ms → −1% sales. Google: +500 ms on search results → −20% revenue. Microsoft Live Search: +1 s → −1% queries per user and −1.5% ad clicks per user; +2 s → −2.5% and −4.4%. If latency is not in your OEC, check that your losing variant is not losing merely because it is slower.

## Worth Remembering

**Limitations the authors own up to:**

- **No "why".** You get the magnitude and direction, nothing else. Pair with usability labs and qualitative work.
- **Primacy and newness.** Experienced users are slower on new navigation (favours Control); curious users click everything new (favours Treatment). Both argue for multi-week runs. A useful diagnostic: **compute the OEC on new users only**, since neither bias applies to them.
- **Short-term horizon.** A few weeks cannot see lifetime value. The answer is not to abandon metrics but to build long-term terms into the OEC — penalise unclicked ad real estate, track repeat visits and abandonment, look at latent conversions.
- **The feature must actually exist** to a shippable standard. Paper prototypes are for the stage before.
- **Consistency and parallel experiments** — users comparing notes, or the same person on two machines. Rare enough to live with.
- **Launch events.** If it is in the press release, everyone gets it. No experiment.

**When user-level randomisation is wrong** (Appendix, and the most under-cited part of this paper):

1. *Interference.* eBay: give some bidders a $5 first-bid incentive and the final price of an item is affected for every bidder on it. Treatment leaks into Control. **Randomise items, not users.**
2. *Undesirable to randomise users.* Price elasticity. Amazon did this in 2000 and got a PR disaster when customers discovered different prices. Randomise items.
3. *Impossible to randomise users.* SEO and robot behaviour — bots do not hold cookies. Randomise groups of similar pages.

**Practical lessons worth tattooing on the platform:**

- **Run A/A tests continuously, in parallel with real experiments.** Check the split matches the plan, the collected data matches the system of record, and that you get significance roughly 5% of the time. An A/A test also gives you the $\sigma$ you need for sample-size planning.
- **Decide the OEC before the experiment.** Otherwise you fish, and familywise Type I error eats you. The corrections that exist (Bonferroni, Tukey, Dunnett, Scheffé) all amount to raising the confidence bar and losing power.
- **"It didn't hurt anything" is not a launch criterion.** A non-significant result may just be an underpowered experiment. Absence of evidence.
- **A significant win can still be a bad launch** once you price the perpetual maintenance cost of the feature.
- **Run at least a full week, then in multiples of weeks.** Weekend visitors are a different population. The same logic extends to holidays, seasons and geographies — a US win may not be a Germany or Japan win.

**Connections and follow-ups.** The variance-reduction advice here (§3.2, point 3) is the ancestor of [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)|CUPED]], which Kohavi's group published two years later with a far more powerful trick — regress out pre-experiment covariates. The ramp-up and auto-abort machinery is a crude bandit; the paper itself points at multi-armed bandits and Hoeffding Races for automated optimisation, which is the door into [[Finite-time Analysis of the Multiarmed Bandit Problem (UCB1)|UCB1]], [[A Tutorial on Thompson Sampling|Thompson sampling]] and [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)|LinUCB]]. What is *missing* by modern standards: nothing on sequential or always-valid testing (peeking at a running experiment is an unexamined sin here), nothing on sample ratio mismatch as a named diagnostic, nothing on interference-aware or switchback designs beyond the eBay Appendix note, and no treatment of heterogeneous effects beyond "mine the data for segments" — the invitation that [[Recursive Partitioning for Heterogeneous Causal Effects|causal trees]] later answered properly.

**Practical caveat for a platform builder today.** The architecture section is the most dated part and the most useful part. Dated: MD5 is now cheap enough not to worry about, proxy-based page rewriting has largely died. Useful: the three randomisation properties are still the correctness spec for any assignment service, and "server-side assignment integrated into the content management system" is still the right shape — instrument once, then configure experiments as data, never as new code.

## Links

Related: [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)]] · [[Counterfactual Reasoning and Learning Systems]] · [[Recursive Partitioning for Heterogeneous Causal Effects]] · [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]] · [[Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms]] · [[Finite-time Analysis of the Multiarmed Bandit Problem (UCB1)]] · [[A Tutorial on Thompson Sampling]] · [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Amazon.com Recommendations- Item-to-Item Collaborative Filtering (IEEE Internet Computing)]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Panel Regression]] · [[Uncertainty]] · [[Random variable]]

New topics worth writing: sample ratio mismatch, always-valid and sequential testing, Fieller's theorem, switchback and interference-aware experiment designs, novelty and primacy effects, guardrail metrics, experiment metric design and surrogate metrics, Plackett–Burman and fractional factorial designs, HiPPO and experimentation culture
