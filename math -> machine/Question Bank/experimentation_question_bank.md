# Principal Data Scientist (L6) — Experimentation & Causal Inference Question Bank

*Format: Question → Answer. Answers are written to be spoken out loud. Jargon is always followed by a one-line plain-English gloss.*

**How to use this:** Don't memorize. Read each answer twice, then close the doc and say it in your own words. The structure is what you want to internalize — the "here's the tension, here's how I'd resolve it, here's what I'd actually do Monday morning" shape. At L6 they're testing judgment under ambiguity, not recall.

---

## Table of Contents

1. [Foundations & Experiment Design](#1-foundations--experiment-design)
2. [Metrics, OEC & Guardrails](#2-metrics-oec--guardrails)
3. [Statistical Testing Fundamentals](#3-statistical-testing-fundamentals)
4. [Power Analysis, MDE & Test Duration](#4-power-analysis-mde--test-duration)
5. [Variance Reduction](#5-variance-reduction)
6. [SUTVA, Interference & Network Effects](#6-sutva-interference--network-effects)
7. [Switchback & Marketplace Experiments](#7-switchback--marketplace-experiments)
8. [Quasi-Experiments & Observational Causal Inference](#8-quasi-experiments--observational-causal-inference)
9. [Regression, Panel Data, Fixed & Random Effects](#9-regression-panel-data-fixed--random-effects)
10. [Sequential Testing, Peeking & Multiple Comparisons](#10-sequential-testing-peeking--multiple-comparisons)
11. [Bayesian Experimentation](#11-bayesian-experimentation)
12. [Heterogeneous Treatment Effects & Personalization](#12-heterogeneous-treatment-effects--personalization)
13. [Trust, Diagnostics & Failure Modes](#13-trust-diagnostics--failure-modes)
14. [Platform, Scale & Org Design](#14-platform-scale--org-design)
15. [Case Studies & Scenario Questions](#15-case-studies--scenario-questions)
16. [Leadership & Influence (L6-specific)](#16-leadership--influence-l6-specific)

---

# 1. Foundations & Experiment Design

### Q1.1 — Walk me through how you'd design an A/B test from scratch.

I think of it as six decisions, in this order, and I try not to skip ahead because skipping ahead is where most bad tests come from.

**First, the decision.** Not the metric — the *decision*. What action will we take differently depending on the outcome? If nobody can answer that, the experiment isn't worth running. I'll literally ask "if this comes back flat, what do we do?" If the answer is "ship it anyway," we're doing a rollout, not a test, and we should be honest about that and just monitor guardrails.

**Second, the hypothesis with a mechanism.** Not "new checkout button will increase conversion" but "the current button is below the fold on mobile, so users don't see it; moving it up should increase the click-through to payment, which should flow to conversion." The mechanism matters because it tells me which intermediate metrics to instrument. If conversion moves but click-through doesn't, something else caused it and I want to know that.

**Third, the randomization unit.** Usually user, sometimes session, sometimes device, sometimes city or time-slice. The rule I use: randomize at the level where the *interference* happens — interference meaning one unit's treatment affecting another unit's outcome. If users talk to each other or compete for the same supply, user-level randomization is broken and I need cluster or switchback designs.

**Fourth, metrics.** One primary metric — the OEC, or Overall Evaluation Criterion, meaning the single number we're willing to be judged on. Then a handful of secondary metrics that test the mechanism, and a fixed set of guardrails — metrics we're not trying to move but refuse to break, like latency, crash rate, unsubscribes.

**Fifth, power and duration.** I compute the minimum detectable effect — MDE, the smallest true effect the test can reliably catch — given the traffic we can allocate and the time we can afford. Then I sanity-check it against what the feature could plausibly do. If we can only detect a 5% lift and the realistic ceiling is 1%, the test is theatre and I'd say so and propose an alternative — bigger swing, proxy metric, or a quasi-experimental approach.

**Sixth, the analysis plan, written before launch.** Primary metric, test, alpha, one- or two-tailed, segments we'll look at, how we'll handle multiple comparisons, stopping rule. Pre-registering this is the single cheapest thing you can do to prevent p-hacking — p-hacking meaning torturing the data until something looks significant.

Then I run an A/A test or at minimum check the sample ratio before I trust anything.

---

### Q1.2 — What's the difference between the randomization unit and the analysis unit, and why does it matter?

The randomization unit is what you flip the coin on. The analysis unit is what you count rows of.

The classic mistake is randomizing by user but analyzing by session or by pageview. If I randomize 10,000 users but analyze 500,000 sessions, my code will happily compute a standard error as if I had 500,000 independent observations. I don't. Sessions from the same user are correlated — a heavy user contributes many correlated rows. So the standard error is understated, the p-value is too small, and I'll ship things that don't work.

Two fixes. Either aggregate up to the randomization unit — compute a per-user metric, then run the test on users. Or use the delta method or clustered standard errors — clustered standard errors meaning you tell the model "these rows belong together, treat the cluster as the independent unit." The delta method is the standard approach for ratio metrics like clicks-per-pageview where the denominator is itself random.

At scale this matters enormously. I've seen this single bug inflate significance by 2-3x, which means roughly a third of "wins" on that surface were noise. It's the first thing I check when someone shows me a suspiciously clean result.

---

### Q1.3 — When would you *not* run an A/B test?

Several cases, and I think being willing to say this out loud is part of the job.

**When you can't randomize.** Pricing changes across a whole market, brand campaigns, regulatory changes, a rebrand. Then I go to quasi-experimental methods — difference-in-differences, synthetic control, geo-experiments.

**When the effect is too small to detect with available traffic in a reasonable time.** If I need 18 months to power the test, the product will have changed three times. I'd either combine it with other changes into a bigger bet, test on a proxy metric that's closer to the mechanism and has lower variance, or accept a decision based on qualitative evidence plus guardrail monitoring.

**When the cost of the test exceeds the value of the information.** If the change is cheap, reversible, and low-risk, sometimes you just ship it and watch. Experimentation has an opportunity cost — traffic, engineering time, calendar time.

**When it's ethically or contractually off the table.** Deliberately degrading safety features, testing on protected groups differentially, anything that creates a materially worse experience for a vulnerable population.

**When there's severe interference you can't design around.** If treating one user necessarily changes the control experience — a shared marketplace, a social feed, a two-sided network — a naive A/B will give you a biased answer and I'd rather run a cluster-randomized or switchback design, or a time-based holdout.

**When the real question is "how much," not "does it work."** Sometimes you need a dose-response curve or a structural model, not a binary A/B.

---

### Q1.4 — A/B vs. A/B/n vs. multivariate testing — when do you use each?

**A/B** is one change, two arms. Cleanest read, most power per arm.

**A/B/n** is several variants of the same idea — three headline options, four price points. You get to pick a winner from a set, but you pay in two ways: traffic splits across more arms so each arm is smaller, and you now have a multiple comparisons problem — the more arms you test, the more likely one looks good by chance. I'd correct for that, typically with Dunnett's test if everything is compared to a single control, which is more powerful than Bonferroni for this exact case because it accounts for the fact that all comparisons share the same control.

**Multivariate / factorial** is testing multiple *factors* simultaneously — button color × headline × layout — and it's underused. The big win is that in a full factorial design you get the main effect of each factor using the *entire* sample, not a fraction of it. Two factors at two levels each is four cells, but the main effect of factor A compares half the users to the other half. So factorial is often *more* efficient than sequential A/B tests, not less. You only lose power when you care about the interactions — interaction meaning the effect of A depends on the level of B — because interaction effects are typically smaller and need more sample.

My practical rule: if the changes are independent and I want to learn about each, factorial. If they combine into one coherent experience I want to ship as a package, test the package. If I'm choosing among variants of one idea, A/B/n with Dunnett.

---

### Q1.5 — How do you handle a test where the treatment is a "big redesign" bundling many changes?

Honestly, I push back first, then adapt.

The problem with a bundle is you learn one bit of information — better or worse — for a huge amount of traffic and calendar time. If it wins, you don't know which part won, so you can't generalize the learning to the next project. If it loses, you don't know whether the whole idea is bad or one component poisoned it.

But sometimes you genuinely can't decompose — a redesign has coherence, and shipping half of it is worse than shipping neither. So what I do:

I test the bundle as the primary decision, but I instrument the mechanism heavily. Every component has a hypothesis about which intermediate metric it should move. Then post-hoc I can at least say "the nav change did what we expected, the new card layout did the opposite," even without clean causal attribution per component.

I also plan for a **follow-up decomposition**. If the bundle wins, run a "leave-one-out" or ablation set on the biggest components — ablation meaning you remove one piece and see how much of the gain disappears. And I watch out for **novelty effects** — users reacting to the change itself rather than its quality — which are much stronger with visual redesigns. That means running longer and looking at the effect trend over time, not just the pooled average.

Finally, I'd argue for a **long-term holdout**: keep 1-5% of users on the old experience for a quarter or two. Redesigns are exactly the case where the two-week read and the six-month read diverge.

---

### Q1.6 — What is a holdout, and how do you use one well?

A holdout is a group of users deliberately kept on the old experience while everyone else gets the new stuff — either a specific feature or the accumulated set of everything the team shipped that quarter.

Two flavors:

**Feature holdout** — after a test wins, you ship to 95% and hold 5% back for a few months. This is how you catch effects that decay (novelty) or effects that build (learning). A recommendation change might look flat in week one and strongly positive in month three as the model accumulates behavioral signal.

**Team/global holdout** — a small slice of users doesn't get *anything* the team shipped all quarter. This is how you answer "did all our work actually add up?" It's uncomfortable and it's incredibly valuable, because the sum of your individually significant wins is almost always less than the sum of the point estimates. Winner's curse, interaction between features, and metric dilution all eat into it.

The catch with holdouts: they're expensive in opportunity cost, they can create fairness or support issues ("why does my app look different?"), and they suffer from dilution and attrition over time. So I keep them small, pre-register how long they run, and make sure the org has agreed *in advance* that we won't kill the holdout the moment it becomes inconvenient.

---

### Q1.7 — How do you decide the traffic allocation — 50/50, 90/10, ramped?

Statistically, 50/50 maximizes power for a fixed total sample — the variance of the difference is minimized when the arms are balanced. So absent other considerations, 50/50.

But there are good reasons to deviate:

**Risk.** New code, potential for breakage or revenue loss — start at 1%, then 5%, 10%, 50%. The ramp isn't for statistics, it's for safety. I treat the early ramp phases as operational checks — errors, latency, crash rates — not as the experiment.

**Multiple concurrent tests.** If the platform is traffic-constrained, smaller allocations let more tests run. There's a trade-off with power that I'd make explicit.

**Asymmetric cost.** If treatment is expensive to serve — an LLM call, a human review — you might run 90/10 and accept the power loss. Worth noting the power loss isn't linear: 90/10 gives you roughly 60% of the effective sample size of a 50/50 split at the same total traffic, which is a real but often acceptable hit.

One important detail: when you ramp, **don't pool the pre-ramp and post-ramp data naively**. The populations and time periods differ, and if you ramp because things looked good, you've introduced selection. I either restart the clock at the final allocation, or use a proper time-stratified analysis.

---

### Q1.8 — Explain the difference between intent-to-treat and treatment-on-treated. Which do you report?

**Intent-to-treat (ITT)** analyzes everyone based on the group they were *assigned* to, regardless of whether they actually experienced the treatment. **Treatment-on-treated (TOT)** or the complier-average effect analyzes only those who actually got the treatment.

The classic case: you assign 100,000 users to see a new feature, but only 20% ever navigate to the page where it lives. ITT dilutes the effect by 5x. TOT looks bigger and more exciting.

I report ITT as the primary, always. Reason: ITT preserves randomization. The moment you condition on "actually used the feature," you've selected on a post-treatment variable — people who use a feature are systematically different from people who don't, and that difference is not caused by your treatment. So TOT computed naively is just a correlational comparison wearing a lab coat.

That said, ITT answers "what happens if we ship this," which is usually the business question, but it can massively understate the effect *for the people it touches*, which matters for deciding whether to invest in driving more people to the feature. So I'll also report a properly identified local effect using assignment as an instrument — this is a **CACE / LATE** estimate, meaning the complier average causal effect, which is the effect among people who would take up the treatment if offered. Practically it's the ITT effect divided by the take-up rate difference, under assumptions. I present both with clear labels: "shipping this to everyone moves the metric X%; among the 20% who engage, the effect is roughly 5X%."

The framing I use with stakeholders: ITT is the ship decision, LATE is the investment decision.

---

### Q1.9 — What's a pre-registered analysis plan and what goes in it?

It's a document written and locked before the test launches. Mine contains:

- The decision the test informs and the ship criteria
- Primary metric (exactly one), its definition down to the SQL, and the direction of expected movement
- Secondary metrics tied to the mechanism
- Guardrail metrics and their "break" thresholds
- Randomization unit, allocation, targeting/eligibility criteria
- Planned sample size, MDE, alpha, power, and expected duration with the calculation shown
- One- or two-tailed, and why
- Multiple-comparison handling
- Pre-specified segments (and the acknowledgment that anything else is exploratory)
- Stopping rules: sequential boundaries if any, and the guardrail conditions that trigger an emergency stop
- How we'll handle known issues: outlier capping rules, bot filtering, SRM check thresholds

The value isn't bureaucratic. It's that it removes every degree of freedom that a motivated human would otherwise use to manufacture a result. It also makes the post-test conversation dramatically shorter, because the "well what if we look at just mobile users in Tier 1 cities" move is already off the table — or at least clearly labeled as exploratory and hypothesis-generating rather than confirmatory.

At L6 I'd add: the plan should be reviewed by someone who isn't invested in the outcome. That's a cheap organizational control that catches a lot.

---

# 2. Metrics, OEC & Guardrails

### Q2.1 — How do you choose the primary metric (OEC)?

Four properties I look for:

**Sensitive** — it actually moves when the product changes. Revenue is often a terrible primary metric for a UI test because it's dominated by variance from a small number of high spenders.

**Aligned with long-term value** — this is the hard one. Clicks are sensitive but you can juice them with clickbait, which is long-term negative. I want a metric where gaming it is the same as genuinely improving.

**Measurable in the test window** — retention at 90 days is a great metric and useless for a two-week test. So you often need a *surrogate* — a short-term metric validated to predict the long-term one.

**Directionally unambiguous** — everyone agrees which way is good. "Time on page" fails this; more time could mean engagement or confusion.

The technique I use for surrogates: take a corpus of past experiments where you *do* have long-term outcomes, and check which short-term metrics predicted the long-term effect *across experiments*. That's the key subtlety — you need the metric to predict the long-term *treatment effect*, not just correlate with the long-term outcome at the user level. A metric can be strongly correlated with retention cross-sectionally and still be a useless surrogate, because the treatment might move it through a channel that doesn't touch retention. Meta and Netflix have both published versions of this idea, and it's the sort of thing I'd want to build as a platform capability rather than re-derive per test.

If the team genuinely can't collapse to one metric, I'd rather use an explicit weighted composite with the weights argued out in advance than have three "primary" metrics, because three primaries means whoever wants to ship picks the one that moved.

---

### Q2.2 — Talk about guardrail metrics. What belongs there?

Guardrails are things you're not trying to improve but refuse to break. Three categories:

**Trust/quality guardrails** — page load time, error rate, crash rate, API latency. These catch "we won on engagement because we accidentally made the page load slower and users clicked more out of confusion" — which sounds absurd but latency regressions are one of the most common silent killers.

**Business guardrails** — revenue, margin, unsubscribe rate, support ticket volume, churn. The engagement win that costs 2% of revenue is not a win.

**Diagnostic/validity guardrails** — sample ratio mismatch, and metrics that shouldn't be affected at all. If your treatment moves a metric it has no causal pathway to, that's evidence of a bug or of contamination, not a discovery.

Two operational points. First, guardrails need **asymmetric thresholds** — I don't need to prove the guardrail improved, I need to be confident it didn't degrade beyond a tolerance. That's a non-inferiority test — testing that the treatment is not worse than control by more than some margin delta — and it's a different calculation than a standard two-sided test. Second, guardrails should be **automatic and platform-level**, not per-experiment opt-in, because the tests where someone forgets to add the revenue guardrail are exactly the tests where it matters.

---

### Q2.3 — What's metric dilution and why does it bite you?

Dilution is when your metric is computed over a population much larger than the population the treatment can possibly affect, which shrinks the observed effect toward zero and destroys your power.

Example: you change the checkout flow. If your metric is "conversion rate across all site visitors," but only 8% of visitors reach checkout, then a 10% improvement in checkout conversion shows up as a 0.8% improvement in the overall metric — and you need roughly 150x the sample to detect it.

The fix is to define the metric on the **triggered population** — users who actually reached the point where the treatment could act on them. Critically, you must trigger *in both arms* using the same criterion, and the criterion must be based on pre-treatment or treatment-independent behavior. "Users who reached the checkout page" is fine if the treatment doesn't change who reaches checkout. If the treatment *does* change who reaches checkout, you've got a post-treatment selection problem and triggering biases you.

Practically: I implement trigger-based analysis at the platform level, log the trigger event in both arms, and always report both the triggered effect and the diluted overall effect. The triggered effect is the scientific finding; the diluted effect is the business impact of shipping.

---

### Q2.4 — Your test wins on the primary metric but loses on a guardrail. Walk me through the decision.

First I check whether both results are real. Is the guardrail regression significant, or is it noise on a metric we look at 30 of? If we monitor 30 guardrails at alpha 0.05, we expect roughly 1.5 false alarms per test even when nothing's wrong. So I'd look at the magnitude, the confidence interval, whether it's consistent across segments and across time, and whether there's a plausible mechanism.

If it's real, I move to trade-off framing, and I try to convert everything to a common currency — usually annualized dollars or a user-value equivalent. "We gain an estimated $4M in incremental orders and lose an estimated $1.5M in increased support load and elevated churn among a segment." That converts a values argument into an arithmetic argument, which is far easier to resolve.

Then I check for a **fix-it path**: is the regression intrinsic to the idea, or an implementation artifact? Latency regressions are usually fixable. A trust regression usually isn't.

And I'd surface **who** is affected. A small average regression that's concentrated in new users or a specific market is a much bigger deal than a diffuse one, because those users churn and never come back.

The thing I'd say explicitly in the interview: at L6 my job isn't to make this call alone, it's to make the trade-off legible enough that the right decision-maker can make it quickly, and to be the person who insists the loss doesn't get quietly dropped from the readout.

---

### Q2.5 — What are novelty and primacy effects, and how do you detect them?

**Novelty effect** — users engage with something because it's new and different, and that engagement decays. **Primacy effect** (sometimes called change aversion) — the opposite; users are worse off initially because they have to relearn, and performance improves as they adapt.

Detection: plot the treatment effect by *days since a user first entered the experiment*, not by calendar date. That distinction is important — calendar-date plots mix cohorts, so a user who joined on day 1 and a user who joined on day 10 get averaged together, and you can't see the decay. Cohort-time plots show it clearly.

If the effect is decaying toward zero, I'd suspect novelty. If it's rising from negative to positive, primacy. Either way, the two-week average is the wrong number to ship on.

Confirmation approaches: run longer; analyze only users who entered in the first week and follow them out; compare new users (who have no old experience to un-learn, so no primacy) to existing users — if the effect is positive for new users and negative for tenured users initially and converging, that's a strong primacy signature. And a long-term holdout settles it definitively.

The failure mode I've seen most: a redesign shows a 3% lift in week one, the team ships, and six weeks later the metric is back to baseline and nobody connects it, because the shipped feature is no longer being measured. That's an argument for holdouts as standard practice on anything with a UI surface change.

---

### Q2.6 — How would you handle a metric with extreme outliers, like revenue per user?

Heavy-tailed metrics are the main reason revenue tests are underpowered. A handful of whales dominate the variance, and the sample mean isn't close to normal even at large n.

Options, roughly in the order I'd reach for them:

**Winsorize or cap** — replace values above, say, the 99.9th percentile with the value at that percentile. Pre-register the cap and derive it from *pre-experiment* data so the cap isn't influenced by the treatment. This is my default. It biases the estimate slightly but the variance reduction usually swamps it, and you're now estimating a well-defined capped-mean parameter.

**Transform** — log or square root. Problem: you're now testing a different estimand — estimand meaning the actual quantity you're estimating. The effect on log-revenue is not the effect on revenue, and you can't naively back-transform. I use this for exploration, rarely for the headline number.

**Change the metric** — decompose revenue into conversion rate × average order value, and test those separately. Conversion rate is bounded and well-behaved, so it's much more powerful. Often the mechanism only plausibly affects one of them anyway.

**Bootstrap or permutation tests** — don't rely on normality at all. Cheap to run these days, and I'd use them as a cross-check on the parametric result.

**Variance reduction with CUPED** — pre-experiment spend is highly predictive of in-experiment spend for heavy spenders, so covariate adjustment is unusually effective here.

I'd also say: if the decision genuinely hinges on whales, you may not be able to power that test at all, and the honest answer is to run a longer, lower-frequency read or use a quasi-experimental approach on aggregate revenue.

---

### Q2.7 — What's Simpson's paradox in an experiment context, and how does it show up?

Simpson's paradox is when a trend holds in every subgroup but reverses when you pool them, because the subgroups have different sizes and different baseline rates.

In a *properly randomized* experiment with fixed allocation, this shouldn't happen for the primary comparison, because randomization balances the subgroup mix across arms. Where it actually bites:

**During a ramp.** You start at 5% treatment on day 1, go to 50% on day 5. Day-1 users and day-5 users have different composition — weekend vs. weekday, different marketing pushes. Pooling across the ramp mixes different allocation ratios with different populations and can flip the sign. Fix: analyze within ramp periods, or use time-stratified estimates.

**When you pool across experiments or across countries with different allocations.**

**When there's differential attrition** — if treatment causes low-value users to drop out, the surviving treatment group looks better on per-user metrics purely through selection. This one is nasty because it looks like a win.

The general diagnostic: whenever a pooled result and the segment-level results disagree, the first thing I check is whether the *allocation ratio* varies across segments. If it does, the pooled number is a weighted average with the wrong weights, and the segment-stratified estimate is the trustworthy one.

---

# 3. Statistical Testing Fundamentals

### Q3.1 — Chi-square test vs. z-test (or t-test) for proportions — when do you use which?

They're answering closely related questions and for the simple two-group case they're essentially the same test.

**Z-test for two proportions** compares the conversion rate in A vs. B. It's directional-capable — you can do one-tailed — and it gives you a confidence interval on the *difference*, which is what stakeholders actually want. This is my default for a standard two-arm conversion test.

**Chi-square test of independence** asks whether the row variable and column variable are independent in a contingency table. For a 2×2 table, the chi-square statistic is exactly the square of the two-proportion z-statistic, and the p-values are identical. So for two arms and a binary outcome, they're the same test — the chi-square is just the two-sided version.

Where chi-square earns its keep is **larger tables**: three or more variants, or a multi-category outcome. If I have four variants and I want a single omnibus test of "is anything different anywhere," chi-square does that in one shot. But an omnibus result isn't actionable on its own — it tells you something differs, not what — so I'd follow up with pairwise comparisons against control using Dunnett's correction.

**T-test** comes in for continuous metrics — revenue per user, sessions per user. In practice with large samples the t and z converge, so the distinction is academic above a few hundred observations per arm. What actually matters at scale isn't t-vs-z, it's whether the metric is heavy-tailed enough that the Central Limit Theorem hasn't kicked in, and whether your observations are actually independent.

One more: for very small samples or very rare events, chi-square's approximation degrades — the usual rule of thumb is expected cell counts under 5. Then I'd use Fisher's exact test or a permutation test.

---

### Q3.2 — One-tailed vs. two-tailed. What's your position?

My default is two-tailed, and I'll explain why I resist the one-tailed argument.

Mechanically: a two-tailed test at alpha 0.05 puts 2.5% in each tail; a one-tailed test puts all 5% in one direction. So one-tailed gives you more power for the same alpha — it's roughly equivalent to a two-tailed test at alpha 0.10 in the direction you care about.

The argument for one-tailed is "we only care if it's better; if it's worse we won't ship either way, so why spend alpha on the other tail?" That's not crazy in isolation.

Why I still default two-tailed: in product experimentation, **you almost always care about the negative direction**, you just care about it differently. A significant negative result is a genuine finding — it means the mechanism is real and reversed, which is often the most valuable thing you learn. A one-tailed test formally throws that away. Also, one-tailed invites abuse: pick the direction after seeing the data and you've doubled your false positive rate while claiming 5%.

And organizationally, two-tailed is the norm most people assume when they read a p-value, so using one-tailed creates a communication burden and a suspicion burden that isn't worth the power gain. If I need more power, I'd rather get it from variance reduction or a longer run than from the tail convention.

Where I would use one-tailed: non-inferiority testing. "Prove this cheaper model isn't more than 1% worse" is genuinely one-directional and the one-sided framing is correct there.

---

### Q3.3 — Explain a p-value to a VP who doesn't have a stats background.

"If the new version were actually no different from the old one, how often would we see a gap at least this big just from random luck? That's the p-value. A p-value of 0.03 means: about 3 times in 100 you'd see a difference this large purely by chance, even with two identical versions."

Then the thing I make sure to add, because this is where the misunderstanding always is: it is *not* the probability that the treatment doesn't work. It's not "97% chance we're right." It's a statement about how surprising the data is under the assumption of no effect — not a statement about the probability of the hypothesis.

And I'd redirect them to the confidence interval, because that's the number that actually helps them decide. "Our best estimate is a 2.1% lift, and the range consistent with our data runs from 0.3% to 3.9%." That tells them the effect is probably positive *and* how big it might be, which is what they need for an ROI decision. A p-value of 0.04 with an interval of [0.1%, 8%] and a p-value of 0.04 with an interval of [1.9%, 2.3%] should lead to very different conversations, and only the interval reveals that.

If they want the "probability it works" number they're intuitively after — that's Bayesian, and I'd offer to give it to them properly rather than mislabel the p-value.

---

### Q3.4 — Type I vs. Type II error — and how do you set alpha and beta in practice?

Type I is a false positive — you ship something that does nothing. Type II is a false negative — you kill something that would have worked.

The convention is alpha 0.05 and power 80% (so beta 0.20), and I think it's worth being honest that these are conventions, not laws. The right choice depends on the asymmetry of costs.

Where I'd tighten alpha: irreversible or expensive changes, anything touching trust or safety, and situations where I'm running many tests and the base rate of true effects is low. That last point is the one people miss — if only 10% of your ideas actually work, then at alpha 0.05 and 80% power, roughly a third of your "significant wins" are false positives. That's just Bayes: 0.05 × 0.90 false positives versus 0.80 × 0.10 true positives. That ratio, the false discovery rate, is what you actually care about, and it's driven as much by your hit rate as by your alpha.

Where I'd loosen it: cheap, reversible changes with a fast feedback loop, or early exploratory rounds where I'm trying to find directions worth investing in and a false positive just means one more test.

Where I'd raise power instead: guardrail metrics. A Type II error on a guardrail — failing to detect real damage — is often worse than a Type I on the primary. So I'd power guardrails to detect the smallest damage I'd care about, which sometimes means the guardrail, not the primary, determines the sample size.

I'd also say: I'd rather move the conversation from binary significance to effect sizes and intervals, and treat alpha as a decision threshold that we tune to context rather than a natural constant.

---

### Q3.5 — What exactly does a 95% confidence interval mean?

Technically: if we repeated this experiment many times and computed an interval the same way each time, about 95% of those intervals would contain the true effect. It's a statement about the *procedure*, not about this particular interval — this particular interval either contains the truth or it doesn't.

That's the honest definition and I'd say it. But I'd immediately add the practically useful version, because the technical one doesn't help anyone make a decision: the interval is the range of effect sizes that are reasonably consistent with what we observed. If zero is outside it, we have evidence of an effect. If the whole interval is below the level we'd need for the feature to pay for itself, we shouldn't ship even if it's "significant."

The thing I push teams toward is reading intervals for *decision-relevance* rather than for whether they cross zero. Three cases worth naming:

- Interval entirely above the ship threshold → clear ship.
- Interval entirely below → clear no-ship, and you've learned something.
- Interval spanning the threshold → the test was underpowered *for the decision you're trying to make*, regardless of the p-value. That's an inconclusive result and the right response is usually "run longer" or "we can't afford to know, here's the risk of each choice."

That third case is where most real arguments happen and naming it explicitly defuses a lot of them.

---

### Q3.6 — What is statistical power, intuitively, and what drives it?

Power is the probability of detecting an effect if it's really there. Eighty percent power means: if the true lift is what I assumed, I'd catch it 4 times out of 5, and miss it 1 time out of 5.

Four things drive it:

**Effect size.** Bigger true effects are easier to find. This is the one you can't control, only estimate.

**Sample size.** More data, more power — but with diminishing returns. The relationship is that detectable effect scales with 1/√n, so to halve your MDE you need 4x the sample. This is the single most important intuition for planning, because it tells you that "just run it a bit longer" rarely rescues an underpowered test.

**Variance of the metric.** Noisier metric, less power. This is the underrated lever, and it's where CUPED, outlier capping, and metric redefinition pay off. Cutting variance 30% is equivalent to getting 43% more traffic for free.

**Alpha.** Stricter threshold, less power. Real but usually not the lever I'd pull.

The point I'd make at a principal level: teams obsess over sample size because it's the obvious knob, but sample size is usually fixed by business reality. Variance is the knob that's actually under your control, and building variance reduction into the platform multiplies the effective capacity of the entire experimentation program. That's a much higher-leverage investment than arguing about individual test durations.

---

### Q3.7 — What's the difference between statistical significance and practical significance?

Statistical significance says the effect probably isn't zero. Practical significance says the effect is big enough to matter.

With enough traffic, everything becomes statistically significant. At 50 million users you can detect a 0.02% lift, and detecting it doesn't mean it's worth the engineering cost, the added code complexity, or the maintenance burden. The reverse also happens: a genuinely valuable 4% lift that comes back p=0.09 because the test was underpowered gets killed, and that's a real loss.

So the practice I push for is defining a **ship threshold in advance** — the smallest effect that would justify shipping, given cost and complexity. Then evaluate the confidence interval against that threshold, not against zero. This is essentially a minimum-effect or superiority-margin test rather than a nil-hypothesis test, and it maps far better onto how businesses actually decide.

It also kills a bad organizational pattern: "it's significant, so we have to ship" for a 0.03% lift on a feature that adds a permanent maintenance cost.

---

# 4. Power Analysis, MDE & Test Duration

### Q4.1 — Walk me through a power calculation.

For a two-arm test on a continuous metric, the sample size per arm is roughly:

**n ≈ 16 × σ² / Δ²**

where σ² is the variance of the metric and Δ is the effect size you want to detect. The 16 comes from the standard alpha 0.05 / power 80% two-sided setup — it's (1.96 + 0.84)² ≈ 7.85, doubled to 15.7 because you need the sample in both arms. I like carrying the 16 in my head because it makes back-of-envelope estimates fast in a meeting.

For a proportion, σ² = p(1−p), so:

**n ≈ 16 × p(1−p) / Δ²** per arm, with Δ in absolute percentage points.

Quick example I'd actually do out loud: baseline conversion 10%, want to detect a 5% relative lift, so absolute Δ = 0.005. Then n ≈ 16 × 0.10 × 0.90 / 0.000025 = 16 × 0.09 / 0.000025 = 57,600 per arm, so about 115,000 users total. At 20,000 eligible users a day, that's roughly six days — so I'd run two full weeks to cover weekly cycles.

Three things I always check beyond the formula:

**Is my σ estimate right?** I compute it from actual historical data on the actual eligible population, not from a textbook assumption. For ratio metrics I use the delta method to get the variance right.

**Is my Δ realistic?** I'd look at the distribution of effect sizes from past experiments on this surface. If the median win on this surface is 0.5% and I'm powering for 5%, my plan is fiction.

**Am I powering the right metric?** If a guardrail needs more sample than the primary, the guardrail sets the duration.

---

### Q4.2 — How do you decide test duration using data rather than intuition?

This is the question I'd want to answer really well, because "two weeks because that's what we always do" is the L4 answer.

I treat duration as the output of four constraints, and I take the max.

**Constraint 1 — Statistical: sample accumulation.** Compute required n from the power calculation, divide by the daily rate of *eligible, triggered* users. Not total users — the ones who can actually be affected. This gives a floor in days.

**Constraint 2 — Cyclical: business cycle coverage.** Behavior is periodic. Weekday users differ from weekend users; the first of the month differs from the 20th if you're in fintech. I always run in **whole multiples of a week**, minimum one full week, and I'd verify the cycle length empirically rather than assuming — run an autocorrelation or a simple day-of-week decomposition on the metric's history. If there's a strong monthly cycle, one week isn't enough regardless of what the power calc says. Also: exclude or carefully handle holidays and known promo periods, because they change both the level and the variance.

**Constraint 3 — Behavioral: effect stabilization.** How long until the treatment effect stops moving? I derive this empirically from the *corpus* of past experiments on this surface: plot effect-by-days-since-exposure for historical tests and see when curves flatten. If recommendation-model tests on this surface historically take 18 days to stabilize because the model needs to learn, then two weeks is systematically biased and I'd say so with the evidence. This is how you convert "novelty effects are a thing" into an actual number.

**Constraint 4 — User coverage / cohort maturity.** For metrics defined over a window — 7-day retention, 30-day repeat purchase — every user needs to have had the full window inside the experiment. So duration ≥ enrollment period + metric window. People forget this constantly and end up with a metric computed on a biased subset of early-enrolled users.

Then I take the maximum of those four, round up to a whole week, and add a buffer for ramp-up days that I won't include in the analysis.

The other half of the answer is the **upper** bound. Longer isn't free. You get sample-ratio drift, cookie churn and device switching that erode the assignment, more chance of a confounding launch, and opportunity cost on the traffic. So I'd also state a maximum — typically 4-6 weeks for a standard test — and if the power calc demands more than that, that's a signal to change the design rather than extend the clock.

---

### Q4.3 — What's an MDE, and how do you set it honestly?

MDE — minimum detectable effect — is the smallest true effect your test can catch with the specified power. It's an output of your design, not an input from your hopes.

The dishonest version, which I've seen a lot: someone wants a two-week test, so they back into an MDE that makes two weeks work, write down "we're powered to detect a 3% lift," and never mention that the feature realistically does 0.5%. The test then comes back null and everyone concludes "the feature doesn't work," when the correct conclusion is "we learned nothing."

How I set it honestly:

**Bottom-up from the mechanism.** If the change affects 20% of sessions and could plausibly improve their conversion by 10%, the diluted effect is 2%. That's the number to power for.

**Empirical prior from past tests.** Pull the distribution of realized effect sizes from the last two years of tests on this surface. If the 75th percentile is 1.2%, powering for 3% means you'd miss three-quarters of real wins.

**Economic threshold.** What lift makes this worth shipping given engineering cost, maintenance, and complexity? If the break-even is 1.5%, powering to detect 0.3% is over-engineering and powering to detect 4% is useless.

Then I take the *smaller* of "plausible effect" and compare it to "economic threshold." If plausible effect < economic threshold, don't run the test — the feature can't pay for itself even if it works.

And I always report the achieved MDE alongside a null result. "This test was powered to detect 2%; we observed 0.4% with a CI of [−0.8%, 1.6%]" is a completely different sentence from "no significant effect," and it's the one that prevents the org from mis-learning.

---

### Q4.4 — Your PM says "we don't have enough traffic to power this test." What are your options?

I'd walk through the levers in order of how much I like them.

**1. Reduce variance (best option, free power).** CUPED with pre-experiment covariates, stratified assignment, outlier capping, regression adjustment. On a metric with strong pre-period signal this routinely cuts variance 30-50%, which is like doubling your traffic on a good day. Always my first move.

**2. Change the metric to a closer, less noisy proxy.** If the mechanism is "make the button visible," test click-through on the button, not downstream revenue. Higher signal, lower variance, much better powered. You accept that you're testing the mechanism rather than the outcome, and you validate the mechanism-to-outcome link separately.

**3. Trigger properly to kill dilution.** Analyze only users who reached the affected surface. Often this alone is a 5-10x effective sample gain.

**4. Increase the effect size.** Make a bolder change. A timid change tested rigorously teaches you nothing; a bold change tells you whether the direction has any juice at all. I'd rather run one aggressive test than three timid ones.

**5. Change the randomization unit.** If you're randomizing by user but the natural unit could be session (and there's genuinely no cross-session learning or carryover), sessions give you far more units. This has real assumption costs so I'd be careful, but it's a legitimate lever.

**6. Use a factorial design** so multiple questions share the same traffic.

**7. Switch to a time-based or geo design.** Switchbacks, geo-holdouts, and synthetic control let you use aggregate volume rather than user counts. Fewer effective units in some ways, but they get around the interference and dilution problems entirely.

**8. Borrow strength across experiments.** Bayesian hierarchical modeling — pooling information from related past tests to sharpen the estimate for this one. Powerful, and requires an org that trusts the method.

**9. Accept a lower confidence bar deliberately.** For a cheap, reversible change, running at alpha 0.20 and power 60% with eyes open is a legitimate business choice. What's not legitimate is running at nominal 0.05 with 30% actual power and pretending.

**10. Don't run it.** Ship on judgment with guardrail monitoring, and be explicit that we're deciding without evidence.

The meta-point I'd make: "not enough traffic" is almost never a dead end, it's a design constraint that should change the design.

---

### Q4.5 — How do you handle the fact that your MDE assumptions are guesses?

A few things.

I do **sensitivity analysis** rather than a point estimate — show the sample size needed at 1%, 2%, and 3% effect sizes and let the team see the curve. It makes the 1/√n relationship visceral: going from detecting 2% to detecting 1% quadruples the cost.

I use **historical effect-size distributions** as a prior rather than a guess. Most mature experimentation programs have hundreds of past tests sitting in a warehouse; the empirical distribution of effects on a given surface is the single best input to the plan, and almost nobody uses it. Building that as a reusable dataset is a platform investment I'd advocate for.

I distinguish **power for the primary** from **power for the decision**. Sometimes the decision is "is it not-negative," which is a non-inferiority question with a totally different sample requirement.

And I'd be clear that power analysis is a planning tool, not a post-hoc one. **Observed power** — computing power after the fact using the observed effect — is statistically meaningless; it's just a monotone transformation of the p-value and adds no information. If someone asks "was the test powered?" after a null result, the right answer is to report the confidence interval, which contains all the relevant information about what the test could and couldn't rule out.

---

### Q4.6 — What is the "winner's curse" in experimentation?

When you run many tests and only ship the significant winners, the *observed* effect of shipped features is systematically larger than the true effect. Selection on a noisy estimate biases the estimate upward.

Intuition: a feature with a true effect of 1% will sometimes measure 1.8% and sometimes 0.2%. If your ship bar is significance, you disproportionately ship the runs that measured high. So your ship set is enriched with lucky draws.

The size of this matters a lot in practice. If you're running many low-powered tests, the inflation can be large — a test with 30% power that comes back significant has an expected observed effect roughly double the truth. This is exactly why "we shipped 12 wins totaling 8% growth" and "the annual metric moved 1.5%" coexist so often.

What I do about it:

**Shrinkage / empirical Bayes.** Pull the observed effect toward the prior mean from historical experiments. The lower the power and the more extreme the result, the more it shrinks. This gives a much better estimate for forecasting.

**Hold-out validation for big claims.** If a result is going into a board deck, verify it with a long-term holdout or a replication.

**Report shrunk estimates for planning, raw estimates for the ship decision.** The ship decision only needs "probably positive." The financial forecast needs an unbiased magnitude, and those are different jobs.

**Raise power.** The cleanest structural fix. Winner's curse is a symptom of underpowered testing at volume.

---

### Q4.7 — Should you extend a test that's "almost significant"?

Not without a pre-registered rule, no. That's one of the purest forms of p-hacking there is.

The mechanics: if you check at day 14 and extend only when p is between 0.05 and 0.15, then check again at day 21, your actual false positive rate is well above 5%. You've given yourself a second draw conditional on the first draw being close, and you've only allowed the extension to help you.

The legitimate ways to get the same flexibility:

**Pre-register the extension rule.** "If p ∈ (0.05, 0.20) at day 14, extend to day 28" — decided before launch and accounted for in the alpha spending. Now it's a valid two-stage design.

**Use a group sequential design.** Pre-specify the interim looks and use spending boundaries (O'Brien-Fleming, Pocock) so the overall Type I error is controlled across all looks. This is the principled answer and it's what I'd build into the platform.

**Use always-valid inference.** Sequential probability ratio tests or confidence sequences let you look continuously without inflating error. You pay a modest power penalty for the privilege.

**Be Bayesian about it.** Posterior probabilities don't have the same peeking pathology, though decision rules based on them still need calibration checks if you care about frequentist error rates.

The thing I'd add as the senior framing: the real problem with ad-hoc extension isn't the statistics, it's that it's asymmetric. Nobody extends a test that's "almost significantly negative." That asymmetry is what turns a small technical error into a systematic bias toward shipping.

---

# 5. Variance Reduction

### Q5.1 — Explain CUPED. Why does it work?

CUPED stands for Controlled experiment Using Pre-Experiment Data. The idea: a lot of the variation in your metric during the experiment has nothing to do with your treatment — it's just that some users are heavy users and some are light users, and that was true before the experiment started. If you can measure that pre-existing difference, you can subtract it out and be left with less noise.

Mechanically: take the metric Y during the experiment, and a covariate X measured *before* the experiment — usually the same metric in the prior period. Then define an adjusted metric:

**Y_adj = Y − θ(X − X̄)**

where θ = Cov(Y, X) / Var(X). That's just the regression coefficient of Y on X. You then run your normal test on Y_adj.

Why it works: Y_adj has the same expected value as Y — subtracting a mean-centered pre-experiment variable doesn't change the expectation, and crucially X can't have been affected by the treatment because it happened before. So the estimate stays unbiased. But the variance drops by a factor of (1 − ρ²), where ρ is the correlation between Y and X. If pre-period spend correlates 0.7 with in-period spend, you cut variance by about 50%, which is like doubling your sample size.

Practical notes I'd mention:

- Use a pre-period of similar length, typically 1-4 weeks before. Longer can help correlation but risks staleness.
- Multiple covariates work — it generalizes to multivariate regression adjustment, and you can throw in tenure, platform, geography, and other pre-period metrics.
- New users have no pre-period. I handle them as a separate stratum with an indicator, rather than imputing zeros, which would distort θ.
- **The covariate must be strictly pre-treatment.** Using an in-experiment covariate breaks everything — you'd be conditioning on a post-treatment variable and reintroducing bias.
- Estimate θ pooled across arms, not per-arm, to avoid a subtle bias.

CUPED is basically the highest-ROI thing in an experimentation platform. It's a one-time engineering cost that permanently increases the throughput of every test that runs on it.

---

### Q5.2 — What's the relationship between CUPED, regression adjustment, and stratification?

They're three faces of the same idea: use pre-treatment information to remove nuisance variation.

**Stratification** — split users into blocks by a pre-treatment variable (country, platform, usage decile), randomize within each block, and then estimate the effect as a weighted average of within-block effects. Balances the covariate by construction rather than by chance.

**Post-stratification** — same estimation, but you didn't control assignment; you just reweight after the fact. Nearly as good as stratification at large samples, and much easier operationally since you don't need to change the assignment service.

**Regression adjustment / CUPED** — include the covariate in a regression instead of blocking on it. Handles continuous covariates naturally and generalizes to many covariates at once.

At large sample sizes they give similar variance reduction, so I'd choose on operational grounds. Stratified assignment is worth the engineering when you have few large clusters or when imbalance would be embarrassing (e.g. a geo test with 20 cities). Regression adjustment is easier and more flexible for the typical user-level test.

One caution worth knowing: with small samples, naive regression adjustment (Freedman's critique) can introduce bias. Lin's estimator fixes this by including treatment-by-covariate interactions and is what I'd use as the default form. At the sample sizes typical in tech, the issue is negligible, but it's the kind of thing worth knowing exists.

And the modern extension: **ML-based adjustment**. Fit a model predicting Y from all pre-treatment features, use the prediction as the covariate. This is essentially what MLRATE (machine-learning regression-adjusted treatment estimator) does, with cross-fitting — training on one fold and predicting on another — to avoid overfitting bias. It can beat single-covariate CUPED meaningfully when you have rich pre-period features.

---

### Q5.3 — What other variance reduction techniques would you consider?

Beyond CUPED and stratification:

**Trigger-based analysis.** Cutting the population to those actually exposed. Not technically variance reduction — it's dilution removal — but it's usually the biggest single win available.

**Outlier capping / winsorization.** Massive on heavy-tailed metrics. Cap derived from pre-period data.

**Metric transformation or decomposition.** Testing conversion and AOV separately rather than revenue directly.

**Paired / matched designs.** For geo or switchback experiments, matching treated units to similar controls before randomizing.

**Variance-weighted estimators.** Weight observations inversely to their variance — helps when unit-level variance is heterogeneous, e.g. geo experiments where cities differ enormously in size.

**Longer pre-periods and better covariates.** Sometimes the win is just using three pre-period metrics instead of one.

**Reducing measurement noise.** Fixing flaky instrumentation, filtering bots, deduplicating events. Unglamorous and frequently the largest available gain — I've seen bot traffic alone account for a large share of variance on top-of-funnel metrics.

I'd frame this as a portfolio: the platform should apply trigger analysis, capping, and CUPED automatically to every test, and then individual experiments only need bespoke work in unusual cases.

---

# 6. SUTVA, Interference & Network Effects

### Q6.1 — What is SUTVA and why should I care?

SUTVA is the Stable Unit Treatment Value Assumption. It has two parts, and both get violated in real products.

**Part one — no interference.** One unit's treatment doesn't affect another unit's outcome. If I give you a discount and you tell your friend, who's in control, and your friend behaves differently, that's interference. Control is contaminated, so the difference between arms understates (or misstates) the true effect.

**Part two — no hidden variations of treatment.** "Treatment" means the same thing for everyone assigned to it. If half the treatment group got a fast version and half got a slow version due to a rollout bug, you're averaging two different treatments and the estimand is unclear.

Why I care: SUTVA is the assumption that makes the simple difference-of-means a valid causal estimate. Break it and your A/B test is measuring something, but not the thing you think.

Where it breaks most commonly in practice:

- **Social products** — feed changes, sharing, messaging. Treated users produce content that control users see.
- **Marketplaces** — treatment users book more drivers/rooms/inventory, leaving less for control. This creates a *negative* spillover that makes treatment look better than it is at full rollout. This is a huge one because it biases in the direction that makes you ship.
- **Shared models** — a ranking model that learns from treatment-group behavior and serves both arms.
- **Shared budgets** — ad campaigns with a fixed daily budget; treatment spending it means control can't.
- **Household/device sharing** — same person, two devices, two arms.
- **Support/ops capacity** — treatment generating more tickets slows down responses for control.

The direction of the bias matters and I'd always reason it through explicitly. Cannibalization inflates the measured effect; positive spillover deflates it.

---

### Q6.2 — How do you detect interference?

It's genuinely hard because the naive estimate looks fine. Approaches:

**Multi-level / dose-response randomization.** Run the same treatment at multiple allocation levels — 5%, 25%, 50% — either across separate clusters or in different regions. If there's no interference, the estimated effect should be the same at every level. If the effect shrinks as allocation grows, that's the signature of cannibalization: at 5% treatment there's plenty of supply to steal, at 50% there isn't. This is my favorite diagnostic because it directly measures the thing you're worried about.

**Compare designs.** Run a user-randomized test and a cluster-randomized (or geo, or switchback) test of the same treatment. A meaningful gap between the two estimates is evidence of spillover, since the cluster design internalizes it.

**Exposure modeling.** Explicitly measure each control user's "exposure" to treated peers — fraction of their friends in treatment, fraction of nearby drivers treated. Then test whether control outcomes vary with exposure. If control users with many treated friends behave differently from control users with none, you have spillover, and you can even estimate its magnitude.

**Theory first.** Honestly, the most reliable detection is reasoning about the mechanism before you launch. If there's a shared, constrained resource anywhere in the causal chain, assume interference until proven otherwise.

**Sanity check on aggregate.** After rollout, does the total metric move by roughly what the test predicted? Systematic overprediction across many marketplace tests is a strong org-level signal that your designs are leaking.

---

### Q6.3 — How do you design around network effects?

Match the randomization unit to the boundary of the interference.

**Cluster randomization.** Randomize groups instead of individuals — friend clusters found via community detection, geographic regions, schools, companies. Interference is mostly contained inside the cluster, so between-cluster comparison is clean. Cost: you have far fewer effective units, so power drops sharply, governed by the intra-cluster correlation (ICC) — how similar units within a cluster are. High ICC means clusters are nearly redundant and you effectively have as many observations as clusters. Design effect ≈ 1 + (m−1)×ICC, where m is cluster size — that's the factor by which your sample size requirement inflates.

**Geo experiments.** Randomize cities or DMAs. Standard for marketplaces and for anything involving offline effects or advertising. Few units, so I'd pair-match cities on pre-period trends and use synthetic control or a matched-pairs analysis rather than a simple mean difference.

**Switchback / time-based.** Alternate treatment for the whole market in time blocks. Solves interference completely at a point in time since everyone is in the same condition. Discussed in detail in the next section.

**Ego-cluster / graph cluster randomization.** For social graphs where clean clusters don't exist, use balanced graph partitioning to create clusters that minimize cross-cluster edges, then randomize those.

**Two-sided / bipartite randomization.** In a marketplace, randomize on one side (say, listings) and measure on the other, or randomize both sides in a factorial to separate supply-side and demand-side effects. This is how you can distinguish "the feature helps buyers" from "the feature just shifts which seller wins."

**Model the interference explicitly.** Keep individual randomization but estimate an exposure-response model that includes both direct and spillover effects. Cheapest in terms of traffic, most demanding in terms of assumptions.

The trade-off I'd articulate: individual randomization is high power and biased; cluster and switchback designs are unbiased and low power. The right answer depends on the expected size of the bias relative to the size of the effect. If interference is likely to be small relative to the effect, take the power. If interference could plausibly reverse the sign, take the unbiasedness.

---

### Q6.4 — Give me a concrete example of a marketplace test where naive A/B would mislead.

Take a ride-hailing app testing a new feature that shows riders a more accurate ETA, which makes them more likely to complete a booking.

Randomize riders 50/50. Treatment riders book more. Measured effect: +4% bookings. Great, ship it.

But drivers are a fixed pool in any given moment. Every extra ride a treatment rider books is a driver who isn't available for a control rider. So control's booking rate is artificially *depressed* by the treatment. The 4% gap is partly a real gain and partly a transfer from control to treatment.

At full rollout, everyone gets the feature, there's no control group to steal from, and the supply constraint binds for everyone. The realized effect might be +1%, or it could be near zero if supply was the true bottleneck all along.

The tell I'd look for: is the constrained resource actually constrained? If driver utilization is 40%, there's slack and the bias is small. If it's 90% during peak, the bias is large and concentrated in peak hours — which would show up as the effect varying inversely with utilization across time-of-day, a nice diagnostic.

The fix: switchback by city-hour, or geo-randomize with matched cities, or run the user-level test at multiple treatment allocations to trace out how the effect shrinks as you saturate the market and extrapolate to 100%.

I'd also note the reverse case exists: a feature that improves driver earnings might *attract* more supply over time, creating a positive spillover that a short A/B understates. Direction depends on mechanism, which is why the theory step comes first.

---

# 7. Switchback & Marketplace Experiments

### Q7.1 — What is a switchback experiment and when do you use it?

A switchback randomizes *time periods* rather than users. The entire market — say, a city — is on treatment for a block of time, then control for the next block, alternating on a randomized schedule. Everyone in the city at a given moment experiences the same condition.

Use it when:
- There's strong interference through a shared resource — marketplace supply, a shared queue, dynamic pricing, matching algorithms.
- The treatment is inherently market-level and can't be assigned per user — a pricing algorithm, a dispatch policy, an inventory allocation rule.
- You have too few geographic units for a clean geo experiment.

The core advantage: it measures the **equilibrium effect**. At any instant the whole market is under one policy, so supply-demand dynamics play out fully. That's exactly what a user-level test can't give you.

The core costs:
- **Far fewer effective units.** Two weeks of 1-hour blocks in one city is 336 observations, not 336,000. Power is a real constraint.
- **Carryover.** The state of the system at the end of a treatment block bleeds into the next control block — drivers repositioned under the treatment policy are still repositioned. This biases the estimate toward zero.
- **Temporal confounding.** Time of day, weather, events. Randomization handles it in expectation but with few blocks, imbalance is likely, so I'd stratify.

---

### Q7.2 — How do you choose the switchback interval?

This is the central design decision and it's a bias-variance trade-off.

**Short blocks** (say 30 minutes): more blocks, so more effective sample and more power. But more carryover contamination, because each block is a larger fraction "polluted" by the previous condition.

**Long blocks** (say 6 hours): carryover is a small fraction of each block, so less bias. But fewer blocks, so much less power. And you have to cover more calendar time to get enough blocks.

How I'd choose it with data rather than intuition:

**Measure the carryover decay empirically.** From historical data or a pilot, look at the metric's behavior after a known policy switch and see how long it takes to reach a new steady state. If the system equilibrates in 20 minutes, blocks of 2 hours mean only ~17% of each block is contaminated. That decay time is the physical quantity that should drive the choice.

**Burn-in / washout.** Discard the first X minutes of each block from the analysis, where X is the measured equilibration time. This directly removes carryover bias at the cost of some data. It's the cleanest fix and I'd use it as the default. Note that this shifts the optimal block length longer, since you're paying a fixed cost per block.

**Simulate.** Build a simple simulation of the market with a known ground-truth effect, run virtual switchbacks at different intervals, and see which interval recovers the truth with the smallest mean squared error. This is how I'd actually settle it for a specific system, and it's a defensible artifact to show stakeholders.

**Randomize the interval itself.** There's literature on randomized-duration switchbacks that reduce bias from predictable patterns.

Practically, hour-level or few-hour blocks are common in ride-hailing and delivery, but I'd never state that as a default without checking the equilibration time for the specific mechanism being tested. A pricing change equilibrates faster than a driver incentive change.

---

### Q7.3 — How do you analyze a switchback experiment properly?

The main trap is treating each order or each user as an independent observation. They're not — everything in the same time block shares the same treatment assignment and the same market conditions. That's the randomization-unit vs. analysis-unit error again, and it's severe here because block sizes are large.

What I do:

**Aggregate to the block level.** Compute the metric per block, then treat blocks as the observations. Simple, transparent, and correctly conservative.

**Or use clustered standard errors** at the block level if I want to keep unit-level data for covariate adjustment.

**Include time fixed effects.** Day-of-week and hour-of-day are enormous drivers of variance in these systems. Including them as fixed effects — meaning a separate intercept per hour-of-day and per day-of-week — soaks up that variance and dramatically improves precision. This is often the difference between a usable and useless switchback.

**Stratify the randomization** on time-of-day and day-of-week rather than randomizing blocks independently. Ensures balance rather than hoping for it. A common approach: within each day, randomize the assignment of blocks so each condition gets a balanced spread of hours.

**Drop burn-in periods** per the carryover analysis.

**Watch for autocorrelation.** Adjacent blocks are correlated even after removing carryover. Newey-West or block-bootstrap standard errors handle this.

**Sanity check with an A/A switchback.** Run the schedule with no actual treatment difference and verify the false positive rate. Given how many things can go wrong here, I'd want this before trusting the first real result.

---

### Q7.4 — Switchback vs. geo experiment vs. cluster randomization — how do you choose?

I'd frame it as: what's the natural boundary of the interference, and what units do I have enough of?

**Switchback** — when interference is market-wide and instantaneous (supply matching, pricing), and you have few geographies. Trades away power for equilibrium validity. Bad for treatments with long-lasting effects on users, because carryover across blocks is then unbounded — you can't switch a user's learned behavior off.

**Geo experiment** — when interference is contained within a region and you have enough regions (ideally 30+, though synthetic control works with fewer). Handles long-lasting treatments fine since a city stays treated. Good for marketing, pricing, offline effects. Weakness: cities are heterogeneous, so raw randomization gives poor balance; you need matched pairs or synthetic control.

**Cluster randomization on a social graph** — when interference follows the social network, not geography. Requires a graph and a partitioning algorithm. Power depends heavily on ICC.

**Individual randomization** — when interference is genuinely negligible. Highest power. Always my first choice if I can defend the assumption.

The decision I'd walk through out loud: does the treatment persist in the user (learning, habit, memory)? If yes, switchback is out. Does interference cross geographic boundaries? If yes, geo is out. How many independent units do I have at each level? That usually narrows it to one or two options, and then I'd pilot.

A hybrid worth mentioning: **switchback within geo** — different cities on independently randomized switchback schedules. Gives you both more units and market-level validity, and lets you use city fixed effects.

---

### Q7.5 — What's a "budget-split" or "ghost ads" design and when is it useful?

These come from the ads world where standard A/B is badly broken by budget interference.

The problem: if you improve ad targeting for a treatment group, those campaigns spend their budget more efficiently and exhaust it faster, which changes what control sees. Budgets are shared, so arms aren't independent.

**Budget-split** — literally split each campaign's budget into two separate pots, one for treatment and one for control, so each arm has its own budget constraint. Now they can't steal from each other. Costs: you halve the effective budget per arm, which changes pacing dynamics and can itself alter behavior, so it's not a perfect solution.

**Ghost ads / PSA control** — for measuring ad effectiveness, the challenge is that you can't just withhold ads from control, because then you don't know *which* control users would have seen the ad. Ghost ads run the full auction for control users and record who *would have* won, without serving the ad. Now you can compare treated users who saw the ad to control users who would have seen it — an apples-to-apples comparison on the same auction-selected population. It's a much cleaner counterfactual than a PSA (public service announcement) placebo, which distorts the auction.

I'd bring this up as an example of the general principle: when interference comes from a shared constrained resource, the design fix is usually to either partition the resource or simulate the counterfactual assignment without acting on it.

---

# 8. Quasi-Experiments & Observational Causal Inference

### Q8.1 — When can't you randomize, and what's your toolkit?

Cases: pricing across a whole market, brand marketing, regulatory or policy changes, a product already fully launched, competitor actions, anything where withholding is unethical or contractually impossible, and retrospective questions ("what did that launch last year actually do?").

My toolkit, roughly ordered by how much I trust it:

1. **Difference-in-differences (DiD)** — compare before/after change in a treated group to before/after change in an untreated group.
2. **Synthetic control** — build a weighted combination of untreated units that reproduces the treated unit's pre-period, use it as the counterfactual.
3. **Regression discontinuity (RDD)** — exploit a sharp cutoff in a running variable that determines treatment.
4. **Instrumental variables (IV)** — find something that shifts treatment but has no other path to the outcome.
5. **Interrupted time series / Bayesian structural time series (CausalImpact)** — model the counterfactual from the series' own history plus controls.
6. **Propensity score matching / weighting** — match treated and untreated units on observed characteristics.
7. **Double/debiased machine learning (DML)** — flexible ML for the nuisance functions with orthogonalization and cross-fitting.

The ordering roughly tracks how much the method relies on "we observed all the confounders," which is the assumption I trust least. DiD, synthetic control, RDD, and IV all get their credibility from a *design* feature — a discontinuity, a policy timing, an exogenous shifter — rather than from an assumption about covariate completeness. That's why I reach for them first.

The framing I'd give: with quasi-experiments, the question is never "which method is best," it's "what is my source of exogenous variation, and what's the story for why it's exogenous?" The method follows from the answer.

---

### Q8.2 — Explain difference-in-differences and its key assumption.

DiD compares the change over time in a treated group to the change over time in a control group. The logic: the control group's change tells you what would have happened to the treated group anyway — seasonality, macro trends, general growth — so subtracting it isolates the treatment.

Concretely: (Treated_after − Treated_before) − (Control_after − Control_before).

In regression form:

**Y = β₀ + β₁·Treated + β₂·Post + β₃·(Treated × Post) + ε**

and β₃ is the DiD estimate. In practice I'd run the two-way fixed effects version — unit fixed effects and time fixed effects — which absorbs any time-invariant unit differences and any common time shocks.

**The key assumption is parallel trends**: absent the treatment, the treated and control groups would have moved in parallel. Note what this does *not* require — the levels can be completely different. A city with twice the revenue is fine as a control as long as its growth *rate* would have matched.

How I check it, since it's fundamentally untestable in the post-period:

- **Plot the pre-period trends.** This is non-negotiable and is the first chart in any DiD writeup.
- **Event-study / dynamic DiD.** Estimate a separate coefficient for each period relative to treatment. Pre-treatment coefficients should be flat and near zero — that's the falsification test — and the post-treatment ones show how the effect evolves. This one plot does most of the persuasive work.
- **Placebo tests** — pretend treatment happened two periods earlier and check you find nothing.
- **Multiple control groups** — if different plausible controls give the same answer, that's reassuring.

Where DiD breaks: differential shocks hitting one group (a competitor entering only the treated market), anticipation effects (people change behavior before the official start), composition changes, and **staggered adoption**.

---

### Q8.3 — What's the problem with staggered DiD, and how do you fix it?

This is a genuinely important recent development and a good thing to know at a senior level.

When units get treated at different times, the standard two-way fixed effects regression does something bad: it uses already-treated units as controls for later-treated units. If treatment effects vary over time — say the effect grows after adoption — those already-treated units have a *changing* outcome, which contaminates the comparison. The result is that the TWFE coefficient is a weighted average of many 2×2 comparisons with weights that can be **negative**. You can get an estimate with the wrong sign even when every individual treatment effect is positive.

The fixes, from the Goodman-Bacon / Callaway-Sant'Anna / Sun-Abraham / Borusyak line of work:

- **Callaway–Sant'Anna** — estimate group-time average treatment effects using only never-treated or not-yet-treated units as controls, then aggregate with sensible weights.
- **Sun–Abraham** — interaction-weighted event study estimator that avoids the contamination in dynamic specifications.
- **Stacked DiD** — build a separate clean dataset around each adoption cohort with valid controls, then stack and estimate.
- **Goodman-Bacon decomposition** — a diagnostic that shows you exactly which 2×2 comparisons are driving your TWFE estimate and how much weight the "bad" comparisons carry. I'd run this first to know whether the problem is material.

In practice: if all units are treated at the same time, standard TWFE is fine. If adoption is staggered *and* effects plausibly vary over time, use a modern estimator and show that it agrees or disagrees with naive TWFE. Being able to say this in an interview signals you've kept up with the literature rather than learned DiD once in 2015.

---

### Q8.4 — Explain synthetic control. When is it better than DiD?

Synthetic control constructs an artificial control unit as a weighted average of untreated units, with the weights chosen so the synthetic unit closely tracks the treated unit's outcome during the *pre*-treatment period. Then you project it forward and the gap between actual and synthetic is your estimated effect.

Why it's better than DiD in some settings:

- **You don't need a single good control.** DiD needs a comparison group with parallel trends. Synthetic control builds one, so you don't have to find a city that behaves like San Francisco — you make one out of 30% Seattle, 25% Boston, and so on.
- **It's transparent.** The weights are visible and interpretable; you can look at them and judge whether the synthetic unit is sensible.
- **The pre-period fit is a visible diagnostic.** If the synthetic doesn't track the treated unit before treatment, you know immediately not to trust the after.
- **It handles a single treated unit,** which is common — one city, one country, one big customer.

Constraints and cautions:

- Needs a **long, stable pre-period** — the more pre-periods the synthetic has to match, the harder it is to fit well by chance, and the more credible the result.
- Needs a **donor pool** of untreated units that aren't themselves affected by the treatment (no spillover into donors).
- Weights are typically constrained to be non-negative and sum to one, which prevents extrapolation but also means you can't match a treated unit that's extreme on some dimension.
- **Inference is nonstandard.** You don't get a conventional p-value. The usual approach is placebo/permutation inference: apply the method to each donor unit as if it were treated, build the distribution of "effects," and see where the real one falls. With 20 donors your smallest achievable p-value is about 1/21.

Variants worth naming: **augmented synthetic control** (adds a bias-correction term when pre-period fit is imperfect), **generalized synthetic control** (factor-model-based, handles multiple treated units), and **synthetic DiD** (combines both, relaxing the exact-fit requirement). And **CausalImpact** — Google's Bayesian structural time series approach — solves a similar problem with a state-space model and gives you credible intervals directly, which is often easier to communicate.

Where I'd use it in a product context: geo rollouts, marketing spend changes in specific markets, evaluating a launch in one country before global rollout, or measuring the impact of a competitor's action.

---

### Q8.5 — Explain propensity score matching. What are its limitations?

The propensity score is the probability a unit receives treatment given its observed characteristics — usually estimated with a logistic regression or a gradient-boosted model. The key result (Rosenbaum & Rubin) is that if treatment assignment is ignorable given the covariates, then it's also ignorable given just the propensity score. That's a huge dimension reduction: instead of matching on 40 variables, you match on one number.

Then you either match each treated unit to control units with a similar score, or weight the sample by the inverse of the propensity (IPW), or stratify into score bins.

**The limitations, which I'd be very direct about:**

**It only handles what you measured.** This is the big one. PSM adjusts for observed confounders. If motivated users are more likely to adopt a feature and motivation isn't in your data, PSM does nothing about it — and in product data, the unobserved confounder is usually intent, which is exactly the thing that also drives the outcome. So PSM is weakest precisely where product questions are hardest.

**Matching can increase imbalance.** King and Nielsen's critique — as you prune to improve balance, propensity score matching can approximate a random pruning and actually worsen covariate balance relative to alternatives. Their recommendation is coarsened exact matching or Mahalanobis matching for low-dimensional problems.

**Extreme weights.** With IPW, units with propensity near 0 or 1 get enormous weights and dominate the estimate. Needs trimming or stabilized weights, and the trimming changes the estimand.

**Overlap / positivity.** You need treated and untreated units with similar scores. If treatment is nearly deterministic for some group, there's no counterfactual and you should say so rather than extrapolate.

**Post-treatment variables.** Including a variable affected by treatment in the propensity model introduces bias. Also, controlling for a collider — a variable caused by both treatment and outcome — creates bias where none existed. So covariate selection needs a causal graph, not a kitchen sink.

**What I'd actually do:** use PSM as one input, not the answer. Report balance diagnostics (standardized mean differences before/after, ideally under 0.1), do a sensitivity analysis for unobserved confounding (Rosenbaum bounds, or the E-value — the strength an unmeasured confounder would need to explain away the result), and prefer doubly robust methods like AIPW or TMLE, which give you a consistent estimate if *either* the propensity model or the outcome model is right. And if there's any way to get a design-based source of variation instead, I'd take it.

---

### Q8.6 — Explain instrumental variables with a product example.

An instrument is a variable that (1) shifts your treatment, (2) affects the outcome *only* through the treatment, and (3) isn't correlated with unobserved confounders. Those are relevance, exclusion, and independence.

The classic product example is **encouragement design**, which is also the cleanest: you can't force users to adopt a feature, but you *can* randomize who gets a prompt encouraging them to. The randomized prompt is a perfect instrument — it's random by construction (independence holds), it increases adoption (relevance), and if the prompt itself is neutral, it only affects the outcome through adoption (exclusion). Then the effect of *adoption* is the ITT effect divided by the difference in adoption rates. This is the same CACE/LATE machinery from the intent-to-treat question, and it's why encouragement designs are so useful when you can't randomize the treatment directly.

Another: non-compliance in any experiment. Assignment is the instrument for actual exposure.

What I'd flag:

**The exclusion restriction is untestable and usually the weak point.** If the encouragement prompt itself makes users think about the product, it affects the outcome through a second channel and IV is biased. I'd design the prompt to be as neutral as possible and check that it doesn't move unrelated metrics.

**Weak instruments are dangerous.** If the instrument barely shifts treatment — say adoption goes from 8% to 9% — the IV estimate has enormous variance and is biased toward the naive OLS estimate. Rule of thumb: first-stage F-statistic above 10, though recent work suggests that's too lenient and you'd want considerably higher. I'd report the first stage explicitly.

**LATE is local.** IV estimates the effect for *compliers* — people whose behavior the instrument changed. That's not the average effect for everyone, and compliers may be systematically different from always-takers. Worth stating clearly rather than letting people read it as an ATE.

---

### Q8.7 — Explain regression discontinuity and when you'd use it in a product setting.

RDD exploits a sharp threshold. If treatment is assigned based on whether some continuous running variable crosses a cutoff, then units just below and just above the cutoff are essentially identical except for treatment. So comparing them locally gives you a clean causal estimate.

Product examples where this actually comes up:
- Loyalty tiers — spend $500 and you get free shipping. Compare users at $495 to users at $505.
- Credit or risk scores — approval at a cutoff score.
- Ranking positions — items ranked 10th (last on page 1) vs. 11th (first on page 2).
- Free-trial eligibility, discount thresholds, support SLA tiers.
- Any ML model with a decision threshold.

Key assumptions and checks:

**No manipulation of the running variable.** If users can precisely control whether they land above or below (spending an extra $6 to hit free shipping), the units aren't comparable — the ones just above are the strategic ones. The **McCrary density test** checks for bunching at the cutoff. If you see a spike right above the threshold, RDD is dead.

**Covariates should be smooth at the cutoff.** Pre-treatment characteristics shouldn't jump. If they do, something other than treatment changes at the threshold.

**Bandwidth choice matters.** Narrow bandwidth means less bias, more variance. Use a data-driven optimal bandwidth (Imbens-Kalyanaraman or Calonico-Cattaneo-Titiunik) and show robustness across bandwidths.

**Functional form.** Use local linear regression, not high-order global polynomials — global polynomials are known to produce artifacts at the boundary.

**Fuzzy RDD** — if crossing the cutoff only changes the *probability* of treatment rather than guaranteeing it, you use the threshold as an instrument. Very common in practice.

The limitation: the estimate is local to the cutoff. It tells you the effect for users near $500 in spend, not for everyone. That's fine if the cutoff is where the policy decision lives, which it often is.

---

### Q8.8 — What is double machine learning and why would you use it?

DML solves this problem: you want a causal effect, you have lots of covariates, and the relationships are nonlinear. If you just throw everything into a regularized ML model and read off the treatment coefficient, regularization biases that coefficient toward zero, and overfitting the nuisance functions biases it too.

DML fixes it with two ideas:

**Orthogonalization (Frisch-Waugh-Lovell, generalized).** Instead of estimating the effect directly, you (a) predict the outcome from the covariates and take residuals, (b) predict the treatment from the covariates and take residuals, then (c) regress the outcome residuals on the treatment residuals. The resulting moment condition is insensitive to small errors in the nuisance models — first-order errors cancel out. That's what makes it robust.

**Cross-fitting.** Split the data; fit the nuisance models on one fold, predict on the held-out fold, and rotate. This removes the overfitting bias that would otherwise contaminate the estimate.

The payoff is that you can use any flexible ML method (gradient boosting, random forests, neural nets) for the nuisance functions and still get a treatment effect estimate that's √n-consistent with valid confidence intervals.

Where I'd use it: observational analyses with rich user-level features, estimating heterogeneous effects, and — importantly — as a variance-reduction and adjustment layer *inside* randomized experiments where you have many pre-treatment covariates. In an RCT you don't need it for unbiasedness, but the residualization gives you real precision gains.

Caveat I'd state: DML doesn't rescue you from unobserved confounding. It's a better way to handle *observed* covariates. If the confounder isn't in your data, DML gives you a precisely estimated wrong answer. The assumption stack is unchanged; only the estimation is better.

---

### Q8.9 — How do you decide between all these quasi-experimental methods?

I'd ask a chain of questions:

**Do I have a randomized or as-good-as-random source of variation?** If there's a lottery, a random rollout order, an arbitrary cutoff, or a random encouragement — use it. That's IV or RDD, and it's the strongest.

**Is there a sharp threshold?** → RDD.

**Is treatment at the unit level with clear before/after timing, and do I have untreated comparison units?** → DiD, and if adoption is staggered, a modern staggered estimator.

**Is there one (or few) treated units with a long pre-period and a good donor pool?** → Synthetic control or CausalImpact.

**Is treatment self-selected with no timing variation and no instrument?** → Now I'm in matching/weighting/DML territory, and I'd be much more cautious in how I present the result. I'd use it to generate hypotheses and size opportunities, not to make a big irreversible call.

**Can I do more than one?** Triangulation is underrated. If DiD and synthetic control and a matched analysis all point the same direction with similar magnitude, that's far more convincing than any single method's confidence interval. If they disagree, that disagreement is the finding, and I'd investigate rather than pick the one I like.

The thing I'd say at a principal level: my job with observational work is as much about calibrating the organization's confidence as it is about producing a number. I'd always state the identifying assumption in plain English — "this is only valid if nothing else changed in these markets at the same time" — because that's the sentence a decision-maker can actually evaluate with their own domain knowledge, and often they know something that invalidates it.

---

# 9. Regression, Panel Data, Fixed & Random Effects

### Q9.1 — Why would you use regression to analyze an A/B test instead of a t-test?

Several reasons, and they compound.

**Covariate adjustment for precision.** Regressing outcome on treatment plus pre-treatment covariates gives the same unbiased effect estimate with smaller standard errors. That's CUPED in regression clothing.

**Handling imbalance.** If randomization happened to produce a country imbalance, including country as a covariate corrects for it.

**Multiple treatment arms and interactions in one model.** Cleaner than a pile of pairwise tests.

**Heterogeneous effects.** Treatment × segment interactions give you subgroup effects with proper standard errors in one framework.

**Clustered standard errors.** Easy to specify when the analysis unit isn't the randomization unit.

**Flexible outcomes.** Logistic for binary, Poisson or negative binomial for counts.

Caveats: only adjust for **pre-treatment** covariates, never post-treatment ones. Use Lin's estimator (treatment-by-centered-covariate interactions) if you want robustness at smaller samples. And be careful with non-linear models — the coefficient in a logistic regression is a log-odds ratio, which is not the same as the risk difference the business cares about, and it isn't collapsible across covariates. I'd usually report marginal effects, meaning the average change in probability, rather than raw coefficients.

---

### Q9.2 — Explain fixed effects vs. random effects. When do you use each?

Both handle grouped or panel data — repeated observations on the same units over time.

**Fixed effects** gives each unit its own intercept, which absorbs everything about that unit that doesn't change over time — observed or unobserved. The effect is then identified purely from *within-unit* variation: how does this city's outcome change when this city's treatment changes? The huge advantage is that all time-invariant confounders are gone by construction, whether or not you measured them. The cost is that you can't estimate the effect of any variable that doesn't vary within unit (you can't estimate the effect of "being in California"), and you're throwing away all between-unit information, which costs precision.

**Random effects** treats the unit-specific intercepts as draws from a distribution and estimates that distribution's variance. It uses both within- and between-unit variation, so it's more efficient. But it requires the assumption that the unit effects are **uncorrelated with the regressors** — and in most causal settings that's exactly what you can't assume. If unobserved city characteristics drive both treatment and outcome, random effects is biased and fixed effects isn't.

**My rule:** for causal identification, default to fixed effects. The efficiency loss is worth the robustness, and the assumption random effects needs is usually the one under dispute. I'd use random effects when I actually care about the *distribution* of unit effects — hierarchical/multilevel modeling, partial pooling across many small groups, borrowing strength — which is a different goal than identifying a treatment effect.

**Hausman test** formally compares them: it tests whether the RE estimates differ systematically from FE. If they do, RE's assumption fails and you use FE. I'd mention it, but honestly I'd lean on the substantive argument more than the test, since the test has low power in exactly the cases that matter.

**Two-way fixed effects** — unit and time — is the standard workhorse for panel causal work, with the staggered-adoption caveat from earlier.

And a practical must: **cluster the standard errors at the level of treatment assignment**, usually the unit. Not doing so is one of the most common errors in panel work and it dramatically understates uncertainty when outcomes are serially correlated.

---

### Q9.3 — When would you use a hierarchical / mixed-effects model in experimentation?

When you have many small groups and want to share information across them.

The canonical case: you've run an experiment across 50 markets and want the effect *per market*. Estimating each market independently gives you 50 noisy estimates, and the extremes are almost certainly noise — the market with the biggest apparent lift probably just got lucky. A hierarchical model shrinks each market's estimate toward the global mean, with the amount of shrinkage determined by how noisy that market's data is. Small markets shrink a lot, big markets barely move. This is partial pooling, and it gives dramatically better out-of-sample predictions than either "estimate each separately" (no pooling) or "one global number" (complete pooling).

Other uses:

- **Meta-analysis across experiments.** Combine many related tests to estimate the distribution of true effects, which then serves as a prior for new tests and directly addresses the winner's curse.
- **Heterogeneous effects across many segments** without the multiple comparisons explosion — the shrinkage does the multiplicity control for you, in a principled way.
- **Nested structures** — users within sessions within markets.
- **Sequential/adaptive allocation** across many arms.

The reason I like this at a platform level: it converts "we have 200 past experiments sitting in a warehouse" into an actual asset. The empirical distribution of effect sizes becomes a prior that improves every future test's estimate and every future power calculation.

---

### Q9.4 — What are clustered standard errors and when do you need them?

Standard error formulas assume observations are independent. When they're not — multiple sessions from one user, multiple users in one city, multiple time periods for one unit — the effective sample size is smaller than the row count, and the naive standard error is too small.

Clustered standard errors tell the estimator "treat these groups as the independent units," which corrects the inference. The general rule: **cluster at the level at which treatment was assigned.** If you randomized cities, cluster by city, even if you have millions of user rows.

Things worth knowing:

- Clustering makes standard errors *larger*, sometimes dramatically. That's the correct answer, not a problem to be avoided.
- With few clusters (under ~40), the cluster-robust estimator is itself biased downward. Fixes: wild cluster bootstrap, or CR2/CR3 small-sample corrections. This bites in geo experiments constantly, where you might have 20 cities.
- **Two-way clustering** when there are two dimensions of correlation — e.g. clustering by both user and time period.
- If you cluster at too fine a level (user when you randomized by city), you haven't fixed the problem.

The interview-worthy version: the mistake isn't forgetting to cluster, it's clustering at the wrong level, and the wrong level is almost always too granular. When in doubt, cluster more coarsely — you'll be conservative rather than overconfident.

---

# 10. Sequential Testing, Peeking & Multiple Comparisons

### Q10.1 — What's the peeking problem?

If you run a fixed-horizon test but check the p-value every day and stop as soon as it dips below 0.05, your actual false positive rate is way above 5%. With continuous monitoring and no correction, it approaches 100% — given infinite time, a random walk crosses any fixed boundary eventually.

Even modest peeking is bad: checking a handful of times roughly doubles or triples the false positive rate depending on the schedule.

The intuition I use to explain it: the p-value wiggles around randomly over the course of the test. If you only stop when it happens to be low, you're deliberately sampling the minimum of a random process and treating it as a single draw. It's the same error as running 20 tests and reporting the best one.

What makes this hard organizationally is that dashboards make peeking free and irresistible, and stakeholders are watching. So the answer can't be "don't look" — it has to be "look at something that's valid to look at." That means either sequential boundaries or always-valid confidence sequences in the dashboard itself, so the number people see is one they're allowed to act on. I'd treat this as a product problem in the experimentation tool, not a training problem.

---

### Q10.2 — Explain group sequential testing and alpha spending.

Group sequential designs let you look a pre-specified number of times and stop early, while keeping the *overall* Type I error at 5%. The idea is that you have a total alpha budget and you spend a portion at each look.

**O'Brien-Fleming** spends very little alpha early and most at the end. So the early boundaries are very hard to cross — you need overwhelming evidence to stop at look one — but the final look is close to the standard 0.05 threshold. This is my default for most product tests: you get genuine early-stopping capability for large effects, and you barely give up any power for the final analysis.

**Pocock** spends alpha evenly, so it uses a constant, lower threshold at every look (around 0.022 for five looks). More likely to stop early, but the final threshold is meaningfully stricter, so you lose power if the effect is modest.

**Lan-DeMets alpha spending functions** generalize this: you specify a spending function over information time, and you don't have to fix the number or timing of looks in advance — you just need to fix the spending function. Much more practical for a real platform where you don't know exactly when people will look.

You can also add **futility boundaries** — stop early for "this clearly isn't working" — which saves a lot of traffic. Futility stopping doesn't cost Type I error at all (it can only reduce false positives), it costs a bit of power, and it's often the highest-value form of early stopping because most tests are null.

The trade-off to state plainly: sequential designs cost you roughly 5-15% more sample size to reach the same power as a fixed-horizon test *if you run to the end*, but they save enormous amounts of time and traffic on the tests that stop early. Across a portfolio of experiments, that's a clear net win.

---

### Q10.3 — What's always-valid inference / confidence sequences?

This is the "look whenever you want, as often as you want" version.

A confidence sequence is a sequence of intervals such that the probability that *any* of them ever fails to contain the true value is at most alpha. Contrast with a standard confidence interval, where the guarantee holds for one fixed look. So with a confidence sequence, you can watch the dashboard continuously, stop whenever you like — including for business reasons unrelated to the data — and the coverage guarantee still holds.

The main constructions are mixture sequential probability ratio tests (mSPRT) — popularized by Optimizely — and more recent work on time-uniform bounds and e-values/betting-based approaches (Ramdas and coauthors), which tend to be tighter.

Advantages: no need to pre-specify looks or sample size, valid under arbitrary stopping including business-driven stopping, and intervals that never mislead you no matter when you glance at them.

Cost: the intervals are wider than fixed-horizon intervals — you're paying for the flexibility. If you always run to a fixed horizon, a fixed-horizon test is more powerful. The width penalty depends on the construction and how the mixture is tuned; tuning it to be tight around your expected effect size recovers a lot.

When I'd choose which:

- **Fixed horizon** — well-understood test, firm duration, nobody's going to peek. Most powerful.
- **Group sequential** — you want early stopping at a few planned checkpoints. Good default for a mature platform.
- **Always-valid** — dashboards that stakeholders watch continuously, or a culture where you can't control when people look. The honest engineering answer for a self-serve platform used by hundreds of PMs.

I'd argue that for a large self-serve experimentation platform, always-valid inference is the right default precisely *because* you can't control user behavior, and a slightly wider interval that's always correct beats a narrow one that's wrong whenever someone gets impatient.

---

### Q10.4 — Bonferroni, Holm, Benjamini-Hochberg — when do you use each?

**Bonferroni** — divide alpha by the number of tests. Controls the family-wise error rate (FWER), the probability of *any* false positive. Dead simple, works under any dependence structure, and very conservative. Fine for a handful of tests. With 50 metrics it makes you nearly blind.

**Holm-Bonferroni** — a step-down version: sort p-values, compare the smallest to alpha/m, next to alpha/(m−1), and so on until one fails. Controls FWER just as strictly as Bonferroni but is uniformly more powerful. There's essentially no reason to use plain Bonferroni over Holm. I mention Bonferroni because it's what people ask about, but Holm is what I'd use.

**Benjamini-Hochberg** — controls the false discovery rate (FDR): the expected *proportion* of your rejections that are false. Much more powerful when you're testing many hypotheses. Right choice when you're screening — 200 metrics on a dashboard, or a large exploratory segment analysis — because you're happy to accept that some fraction of flagged results are noise, as long as you know the fraction.

**Dunnett's test** — specifically for comparing several treatments against a single control, which is the standard A/B/n setup. It accounts for the correlation induced by the shared control, so it's more powerful than Holm for that structure.

**How I'd actually decide:**

- **One primary metric** → no correction. That's what the pre-registration is for.
- **Multiple variants vs. control** → Dunnett, or Holm if I want simplicity.
- **A small set of secondary metrics** I'm making decisions on → Holm.
- **Large metric dashboard for monitoring** → BH on FDR, and treat flags as investigation triggers rather than conclusions.
- **Guardrails** → this is where I'd argue for *no* correction, or even a looser threshold. Correction reduces false positives at the cost of false negatives, and on a guardrail a false negative means shipping real damage. The asymmetry runs the other way, so I'd keep guardrails sensitive and accept some false alarms.
- **Exploratory segment slicing** → label it exploratory, use BH, and treat anything found as a hypothesis for a confirmatory test, not a result.

The framing I'd give: multiple comparisons correction is a tool for managing a specific error trade-off, and you should apply it where the cost of a false positive is high — not mechanically everywhere.

---

### Q10.5 — How do you stop a test early without wrecking your inference?

Three legitimate mechanisms, all of which require deciding in advance:

**Pre-specified sequential boundaries.** Group sequential with O'Brien-Fleming for efficacy plus a futility boundary. Look at pre-planned points, stop if you cross. Fully valid.

**Always-valid confidence sequences.** Stop whenever, for any reason. Fully valid, slightly wider intervals.

**Bayesian decision rules.** Stop when the posterior probability of a meaningful effect exceeds a threshold, or when the expected loss from stopping is small enough. Valid within the Bayesian framework and doesn't have the peeking pathology, though I'd still simulate the operating characteristics — how often does this rule stop wrongly? — because stakeholders will ask in frequentist terms.

**And the fourth mechanism, which is always valid: guardrail-triggered emergency stops.** If error rates spike or revenue craters, stop. You're not making an inferential claim, you're preventing damage. I'd make this an explicit, always-on rule with automated alerting.

Two things I'd flag about early stopping that people miss:

**Early stops give biased effect estimates.** If you stop because the effect looked large, the observed effect overstates the truth — winner's curse, intensified. There are bias-corrected estimators for sequential designs (median-unbiased estimates, or shrinkage), and I'd apply them before the number goes into a financial forecast.

**Stopping early for success means you have less data on the guardrails and secondary metrics.** The primary might be conclusive at day 5 while revenue is still wide open. So I'd make early stopping conditional on the guardrails also being adequately powered, or commit to continuing a small holdout to keep learning.

---

### Q10.6 — What is p-hacking, and how do you prevent it institutionally?

P-hacking is exploiting researcher degrees of freedom until something crosses 0.05. The specific moves:

- Peeking and stopping when significant
- Testing many metrics, reporting the one that moved
- Slicing many segments, reporting the one that moved
- Trying multiple outcome definitions or filters until one works
- Adding or dropping covariates to see what helps
- Excluding "outliers" post-hoc
- Switching from two-tailed to one-tailed after seeing the direction
- Re-running with a different date range
- Running the same test again after a null and reporting the second one

The important thing I'd say: most of this isn't fraud. It's motivated reasoning by people who genuinely believe in their feature, and every individual step feels defensible in the moment. Which is why the fix has to be structural rather than about individual virtue.

Institutional controls I'd put in place:

**Pre-registration built into the tooling.** You can't launch without declaring the primary metric and analysis plan. The platform then reports the pre-registered metric prominently and marks everything else as exploratory.

**Automated, correction-aware readouts.** The dashboard applies the right correction automatically and labels exploratory findings as exploratory. Don't rely on analysts to remember.

**Sequential or always-valid inference by default**, so peeking is harmless.

**Standardized metric definitions in a central layer,** so nobody can redefine "active user" mid-test.

**Separate the analyst from the outcome.** Whoever reads out the test shouldn't be the person whose promo depends on it. Even light rotation helps.

**Track and publish the org's ship rate and post-launch validation rate.** If 60% of tests "win" but the annual metric doesn't move, that's measurable, and it's the argument that gets you the resources to fix the system.

**Reward good null results.** This is the cultural one and it matters more than any of the technical controls. If a well-designed test that kills a bad idea is celebrated the same as a win, most of the pressure to p-hack evaporates. I'd make "experiments that prevented a bad launch" an explicit thing in performance reviews.

---

# 11. Bayesian Experimentation

### Q11.1 — What's the fundamental difference between frequentist and Bayesian analysis of an experiment?

The frequentist asks: assuming there's no effect, how unusual is my data? Parameters are fixed unknowns; the data is random. You get p-values and confidence intervals, and the probability statements are about the *procedure* over hypothetical repetitions.

The Bayesian asks: given my data and my prior beliefs, what do I now believe about the effect? Parameters are random variables with distributions. You get a posterior distribution, and you can make direct probability statements about the effect itself.

The practical difference that matters to stakeholders: Bayesian gives you the sentence people actually want. "There's a 94% probability the treatment is better, and an 80% probability the lift is above 1%." A frequentist p-value cannot say that, and every attempt to phrase it that way is wrong.

Where Bayesian shines:

- **Direct decision-relevant quantities** — P(effect > threshold), expected loss from choosing wrong.
- **Incorporating prior knowledge** — from past experiments, this is free information you're otherwise discarding.
- **Small samples** — the prior regularizes and prevents wild estimates.
- **No peeking pathology** — the posterior is a valid summary of current evidence regardless of when you look.
- **Natural handling of hierarchy** — many segments, many markets, partial pooling.
- **Multiple arms** — P(each arm is best) falls out directly, no correction needed.

Where frequentist stays useful:

- **Error rate guarantees.** If the org needs "we control false positives at 5% across thousands of tests," that's a frequentist statement and it's a legitimate thing to want.
- **Prior specification is contentious.** "Whose prior?" is a real organizational problem, and a motivated prior is a new p-hacking surface.
- **Familiarity.** Every stakeholder half-knows p-values.

My honest position: for a single test in a mature platform, they usually agree, and the choice matters less than people think. I'd choose based on the decision structure. When the question is "which do we ship and what do we expect to gain," Bayesian expected-loss framing is better. When the question is "can we defend this to a regulator / control error rates across a huge portfolio," frequentist. And I'd want the platform to report both, because they answer different questions and neither is a substitute.

---

### Q11.2 — Walk me through a Bayesian A/B test on conversion rate.

Simplest useful case, and I'd do it out loud:

**Prior.** Conversion rates are proportions, so a Beta distribution is natural — and it's conjugate to the binomial, meaning the posterior is also Beta and you get it in closed form with no sampling at all. If I've run 50 tests on this surface and lifts are rarely above 5%, I'd encode a weakly informative prior — say Beta(30, 270) if the historical base rate is 10%, which is worth about 300 pseudo-observations. Or Beta(1,1), a uniform prior, if I want to stay neutral.

**Data.** Control: 1,200 conversions out of 12,000. Treatment: 1,290 out of 12,000.

**Posterior.** Beta(prior_α + conversions, prior_β + non-conversions). With a uniform prior: control is Beta(1201, 10801), treatment is Beta(1291, 10711).

**Decision quantities.** Draw a few hundred thousand samples from each posterior and compute:
- **P(treatment > control)** — the fraction of draws where treatment wins.
- **P(lift > 1%)** — the fraction where the relative lift exceeds the ship threshold. This is the one I'd actually lead with, because "better than control" is a low bar.
- **Expected loss** — if I ship treatment and I'm wrong, how much do I lose on average? Formally E[max(0, control − treatment)]. Stop when this drops below a tolerance you've defined as negligible. This is a genuinely nice stopping rule because it's stated in units of the thing you care about.
- **Credible interval** — the 95% central range of the posterior lift.

**Report.** "Treatment is better with 93% probability; there's a 71% chance the lift exceeds our 1% ship bar; the expected cost of shipping if we're wrong is 0.04% of conversions, which is below our tolerance. Recommend ship."

That's a much more decision-ready sentence than "p = 0.058, not significant," and it's the same data.

For continuous metrics I'd use a normal-normal conjugate setup or, if the metric is messy (heavy tails, zero-inflation), fit it in Stan or PyMC with a more appropriate likelihood — log-normal hurdle models are common for revenue.

---

### Q11.3 — When do you actually need MCMC, and what is it doing?

MCMC — Markov Chain Monte Carlo — is a way to draw samples from a probability distribution you can't write down in closed form. You construct a Markov chain whose long-run stationary distribution is your posterior, run it, and use the samples to compute anything you want: means, intervals, probabilities of arbitrary events.

You don't need it for simple conjugate cases. Beta-binomial and normal-normal have analytic posteriors — MCMC there is a waste of compute.

You **do** need it when:

- **Hierarchical models.** Effects varying across 50 markets with partial pooling — no closed form.
- **Non-standard likelihoods.** Revenue with a zero-inflated log-normal, counts with a negative binomial, time-to-event models.
- **Many parameters with complex dependence.** Time-varying effects, spline-based dose-response.
- **Bayesian structural time series** — CausalImpact under the hood.
- **Anything where you need the full joint posterior**, not just marginals — e.g., P(arm A is best AND lift > threshold).

Practical notes:

- **NUTS / Hamiltonian Monte Carlo** (Stan, PyMC, NumPyro) is the modern default. It uses gradient information to move efficiently through the posterior, so it's vastly better than old Metropolis-Hastings or Gibbs for correlated, high-dimensional posteriors.
- **Diagnostics matter and I'd always report them**: R-hat near 1.00 (chains have mixed), effective sample size in the thousands, no divergent transitions (which in HMC signal geometry the sampler can't explore — usually fixed with a non-centered parameterization), and trace plots that look like fuzzy caterpillars rather than wandering.
- **Prior predictive checks** before fitting — simulate data from the prior and see if it's plausible. Catches priors that imply conversion rates of 400%.
- **Posterior predictive checks** after — simulate data from the posterior and compare to observed. Catches model misspecification.
- **Variational inference** as a faster approximation when MCMC is too slow, accepting that it typically underestimates posterior variance, so I wouldn't use it for the final uncertainty statement.

The honest framing: MCMC is an inference engine, not a causal method. It doesn't make an observational study causal. I'd use it when the *model* is complex, not to add rigor to a weak design.

---

### Q11.4 — How do you choose a prior, and how do you defend it?

This is the question that decides whether Bayesian methods survive in an organization, so I'd take it seriously.

**Where I get priors:**

**Empirical Bayes from your own experiment corpus.** This is the best answer and the one I'd push for. You've run hundreds of tests; fit a distribution to the historical effect sizes on this surface and use it. It's not subjective — it's a fact about your product. And it automatically handles the winner's curse, because extreme observed effects get shrunk toward what's historically plausible.

**Weakly informative defaults.** Priors that rule out absurdity without pushing toward any answer — "the effect is almost certainly between −20% and +20%." This is the responsible default when you don't have a corpus.

**Skeptical priors for high-stakes claims.** Deliberately center on zero with modest variance. If the data overcomes a skeptical prior, that's persuasive to a skeptical audience. This is a rhetorically strong move: "even assuming this probably doesn't work, the data says it does."

**How I defend it:**

**Sensitivity analysis, always.** Show the posterior under a skeptical prior, a neutral prior, and an optimistic prior. If the conclusion is stable across all three, the prior isn't driving anything and the debate is over. If it isn't stable, that's genuinely important information — it means your data is weak and you should say so rather than hide it behind whichever prior you like.

**Show the prior's implied predictions.** Prior predictive checks make the prior concrete: "this prior says a 10% lift is about as likely as winning a coin flip five times in a row." People can evaluate that.

**Pre-register the prior.** Same logic as pre-registering everything else. A prior chosen after seeing the data is p-hacking with extra steps.

**Report the likelihood-only result too** where it's meaningful, so people can see what the data alone says.

---

### Q11.5 — What is a multi-armed bandit and when should you use one instead of an A/B test?

A bandit adaptively shifts traffic toward arms that are performing well while the test runs, rather than holding a fixed split. Thompson sampling is the standard approach: sample a value from each arm's posterior, serve whichever sample is highest, update. Naturally balances exploring and exploiting.

**Use bandits when:**
- The content is **short-lived** and you'll never reuse the learning — headline selection for a news article, promo creative for a one-week sale. Here, regret minimization (earning as much as possible during the test) matters more than clean estimation.
- **Many arms**, most of which are bad. Bandits stop wasting traffic on losers fast.
- **Continuous optimization** with no ship decision — the algorithm just keeps running.
- **Contextual bandits** for personalization, where the best arm depends on the user.

**Use fixed A/B when:**
- You need a **clean, unbiased estimate** of the effect size for a business case or a forecast. Adaptive allocation makes estimation harder — the sample from each arm is no longer independent of that arm's early performance, so naive estimates are biased. There are corrections (inverse propensity weighting on the time-varying assignment probability, or adaptively-weighted estimators from Hadad/Athey and coauthors), but it's extra machinery.
- **Non-stationarity.** If the best arm changes over time — seasonality, a competitor's promotion — a bandit that converged early is stuck exploiting the wrong arm. Fixes exist (discounting, sliding windows, forced exploration floors) but it's a real failure mode.
- **Delayed feedback.** If the outcome takes days to observe, the bandit is updating on stale information and its advantage largely disappears.
- You care about **secondary metrics and guardrails**, which will be badly underpowered on arms that got starved of traffic.
- The decision is **one-shot and permanent** — you're choosing what to build, and you want the effect size, not just the winner.

The nuance I'd add: these aren't mutually exclusive. A common good pattern is a **batched bandit with a floor** — reallocate weekly rather than continuously, and never let any arm drop below 5% of traffic. That preserves most of the regret savings while keeping every arm estimable. That's usually what I'd actually recommend.

---

# 12. Heterogeneous Treatment Effects & Personalization

### Q12.1 — How do you look for heterogeneous treatment effects without p-hacking?

Heterogeneous treatment effect (HTE) just means the treatment works differently for different people. It's real and important — an average of zero can hide +10% for new users and −10% for power users, and shipping on the average is the wrong call in that case.

The tension is that slicing your data 40 ways guarantees you'll find "significant" subgroups that are pure noise.

How I'd handle it:

**Pre-specify the segments that matter.** Usually a small set with a real theoretical reason: new vs. tenured, platform, market tier, engagement decile. Three to five, declared before launch, corrected with Holm. These are confirmatory.

**Everything else is exploratory and labeled as such.** I'd run it, use BH for FDR control, and treat findings as hypotheses. The discipline is in the label, and in never letting an exploratory finding drive a ship decision on its own.

**Test for heterogeneity globally first.** Before hunting for which subgroup differs, test whether there's *any* heterogeneity — an omnibus test, or comparing a model with interactions to one without. If there's no evidence of heterogeneity at all, subgroup hunting is unlikely to be fruitful and I'd say so.

**Use methods designed for this.** Causal forests (Wager & Athey) and X-learners / T-learners / R-learners estimate conditional average treatment effects (CATE — the effect for a user with given characteristics) with honest inference. Causal forests specifically use "honest" splitting — one subsample decides the tree structure, another estimates the effects in the leaves — which is what makes the confidence intervals valid despite the data-driven partitioning. That's the principled version of subgroup hunting.

**Validate out of sample.** Fit the CATE model on half the data, then check on the other half whether the predicted-high-effect group actually shows a bigger effect. This is the honest test, and it's the one I'd insist on before anyone acts on a heterogeneity finding.

**Replicate before acting.** If new users seem to respond differently, run a follow-up targeted at new users. A confirmatory test is worth far more than a subgroup slice.

The framing I'd give a PM: subgroup analysis is for generating your next experiment, not for winning the current argument.

---

### Q12.2 — Your test is flat overall but strongly positive for one segment. What do you do?

First, I'd figure out how much to believe it.

**Was the segment pre-specified?** If yes, it's a real finding with proper error control. If no, I'd compute how many segments were examined and apply the correction, and I'd usually find the result no longer clears the bar.

**Is there a mechanism?** "Mobile users benefit more because the change fixes a small-screen problem" is a story I can evaluate and it makes the finding much more credible. "Users in Ohio benefit more" with no story is almost certainly noise.

**Is it consistent?** Does it hold across time periods, across adjacent segments (if mobile-iOS is positive, is mobile-Android?), and does the effect size trend sensibly across an ordered segment like engagement deciles? A monotone gradient across deciles is far more convincing than one decile spiking.

**Is the offsetting negative real?** If flat overall and one segment is strongly positive, some other segment is negative. That's often the more important finding and the one people skip past.

Then the decision. If it looks credible: run a **confirmatory experiment on that segment specifically**. That's the clean answer and it usually only takes a couple of weeks since you've narrowed the population. If the business can't wait, I'd frame it as "we have suggestive evidence; shipping to this segment is a reasonable bet with expected value X, but our confidence is much lower than a normal ship decision" — and I'd insist we instrument it so we learn from the rollout.

What I wouldn't do is let "it worked for someone" become the default escape hatch for a null test. If that becomes the org's habit, you've eliminated the possibility of ever learning that something doesn't work, and the experimentation program stops adding value.

---

### Q12.3 — How would you build a system that personalizes which variant a user sees?

This is uplift modeling / policy learning, and I'd break it into four pieces.

**1. Get unbiased training data.** You need a randomized experiment where users got variants at random, because that's what lets you learn the causal effect rather than a correlation. Observational data from a system that already personalized is contaminated by the old policy.

**2. Estimate CATE.** Model the effect as a function of user features. Options: T-learner (fit separate outcome models per arm, take the difference — simple but the difference of two noisy models is noisier), S-learner (one model with treatment as a feature — can under-detect effects because regularization shrinks the treatment coefficient), X-learner (better when arms are imbalanced), R-learner (residualization-based, related to DML), or causal forests. I'd try several and compare, because CATE estimation is genuinely hard and no single method dominates.

**3. Turn the CATE into a policy.** Not just "assign the higher-CATE arm" — you also need to account for cost. If the treatment is expensive to serve, the rule is "assign treatment when CATE exceeds the cost," which is a threshold policy. If there are budget constraints, it becomes a constrained optimization.

**4. Evaluate the policy causally.** This is where people cut corners. Model accuracy metrics like AUC are the wrong evaluation for a causal policy. Use:
- **Qini curves / uplift curves** — how much incremental outcome do you get by treating the top X% by predicted uplift? This is the uplift analogue of a gains chart.
- **Off-policy evaluation** on held-out randomized data — inverse propensity weighting or doubly robust estimators to estimate what the new policy *would* have earned.
- **And finally, an actual A/B test of the personalized policy versus the best fixed policy.** This is the only thing that settles it, and I'd insist on it before full rollout. Surprisingly often, personalization loses to "just ship the better arm to everyone," because the CATE signal is weaker than the noise in the estimation.

**5. Keep a randomized exploration slice** permanently — say 5% — so the model keeps getting unbiased training data and you can detect drift. Without this, the system slowly poisons its own training set.

---

# 13. Trust, Diagnostics & Failure Modes

### Q13.1 — What is a sample ratio mismatch and why is it the first thing you check?

SRM is when the observed split between arms differs significantly from the intended split. You planned 50/50, you got 50.4/49.6 with a million users per arm. That sounds tiny — and it's a five-alarm fire.

Why: with a million users per arm, a 0.8 percentage point deviation has a p-value in the neighborhood of 10⁻⁶. Randomization doesn't do that. Something in your system is treating the arms differently, and if the *assignment* is broken, the *comparison* is broken. The estimate is untrustworthy regardless of how good the p-value on your primary metric looks.

Common causes:
- Treatment code crashes for some users, who then drop out of logging — this is the classic, and it's dangerous because the users who crash are systematically different (older devices, worse connectivity), so you've lost a non-random slice.
- Redirect-based tests where one arm loses users to a slow redirect.
- Bot filtering that catches one arm more than the other.
- Assignment happening at a different point in the funnel per arm.
- Caching or CDN behavior differing by variant.
- Data pipeline joins dropping rows unevenly.
- Users being re-randomized mid-experiment.

How I handle it:
- **Automated chi-square test on the ratio, on every experiment, every day.** Alert at a strict threshold — I'd use something like p < 0.0005 to avoid alert fatigue while still catching real problems.
- **If SRM fires, the result is invalid until explained.** Not "let's discount it slightly" — invalid. I've seen SRM flip the sign of results after being fixed.
- **Debug by segmenting the ratio**: by day, browser, device, country, funnel stage. Where the ratio diverges tells you where the bug is.
- **Check upstream too** — SRM in a triggered subpopulation with a clean overall ratio means the trigger condition is treatment-dependent, which is its own serious problem.

I'd say this in an interview because it signals operational maturity: the most valuable thing an experimentation platform does isn't computing p-values, it's telling you when not to believe them.

---

### Q13.2 — What's an A/A test and what does it catch?

An A/A test runs two identical experiences against each other. There should be no effect. It catches:

**Assignment bugs** — non-random or unbalanced hashing, hash collisions between concurrent experiments, sticky assignment failures.

**Miscalibrated variance estimates.** Run many A/A tests and check that the p-value distribution is uniform and roughly 5% come back "significant" at 0.05. If 15% do, your standard errors are too small — usually the randomization-unit vs. analysis-unit problem, or unhandled clustering, or heavy tails. This is the single most valuable use of A/A and I'd run it continuously as a platform health metric, not once.

**Pipeline problems** — logging differences, join issues, deduplication bugs.

**Pre-existing bias** in how metrics are computed per arm.

Two things people get wrong: first, a *single* A/A test showing p = 0.03 is not evidence of a bug — that happens 5% of the time by design. You need the distribution across many A/As. Second, A/A tests can't catch bugs that only manifest when the treatment code actually runs, so they're necessary but not sufficient.

I'd also mention **continuous A/A monitoring**: keep a permanent A/A running in the background as a canary. When it starts firing, something changed in the platform, and you want to know before it corrupts a hundred real experiments.

---

### Q13.3 — What are the most common ways experiments give wrong answers?

I'd group them:

**Design errors**
- Wrong randomization unit for the interference structure
- Underpowered but presented as conclusive
- Dilution — measuring on a population the treatment can't touch
- Metric that doesn't capture the actual objective

**Analysis errors**
- Analysis unit ≠ randomization unit → understated standard errors
- Peeking without correction
- Multiple metrics/segments without correction
- Ratio metrics without the delta method
- Ignoring heavy tails
- Conditioning on post-treatment variables (including "users who engaged")

**Implementation errors**
- SRM from crashes, redirects, or logging asymmetry
- Assignment leakage — users seeing both variants across devices or sessions
- Treatment not actually firing for part of the treatment group
- Instrumentation added in one arm only (so the arm with better logging shows more events)

**Interpretation errors**
- Novelty/primacy read as a permanent effect
- Winner's curse — overstating shipped effects
- Generalizing from a narrow population or time window
- Treating a null as proof of no effect
- Attributing a bundle's win to your favorite component

**Environmental**
- Concurrent experiments interacting
- A competitor, outage, holiday, or press event landing mid-test
- Seasonality not covered by the window

The meta-point: most of these are *systematic*, not random, which means running more experiments doesn't average them out. That's why platform-level guardrails and automated diagnostics matter so much more than individual analyst care. You fix these once, in the system, or you fix them never.

---

### Q13.4 — How do you handle multiple concurrent experiments interacting?

Most large platforms run hundreds of experiments simultaneously, and the standard assumption is that interactions are rare and small. Mostly true, but worth managing.

**Orthogonal randomization** — use independent hash seeds per experiment so assignment to experiment A is independent of assignment to B. Then any interaction effect is *balanced across arms* and doesn't bias either experiment's estimate; it just adds a little variance. This is the workhorse solution and it handles the vast majority of cases.

**Layers / mutual exclusion** — when two experiments genuinely conflict (both changing the same button, or one being a prerequisite for the other), put them in the same layer so a user can only be in one. Costs traffic, so reserve it for real conflicts.

**Interaction detection** — periodically scan pairs of concurrent experiments for significant interaction effects. With hundreds of experiments you have tens of thousands of pairs, so use FDR control and treat hits as investigation triggers. Microsoft has published on running this as an automated service, and in practice genuine interactions are rare but occasionally severe.

**Post-hoc check for a specific test** — if a result looks strange, check whether treatment users are disproportionately in some other experiment's treatment arm. If randomization is orthogonal they shouldn't be, and if they are, you've found a bug.

The judgment call I'd make: don't over-engineer this. The cost of mutual exclusion everywhere is a massive reduction in experimentation throughput, which costs the org far more than the occasional interaction. Orthogonal hashing plus automated interaction scanning plus manual layers for known conflicts is the right balance.

---

### Q13.5 — How do you know your short-term test result predicts long-term impact?

This is one of the hardest problems in the field and I'd be honest that there's no complete answer.

**Long-term holdouts.** Keep a slice on the old experience for months. Directly measures the long-run effect. Expensive, operationally awkward, suffers from dilution as users churn and as cookie-based assignment degrades. Still the gold standard.

**Surrogate validation across an experiment corpus.** Take past experiments where you have both short-term and long-term measurements. Check which short-term metrics predicted long-term *effects* — and again, the emphasis is on predicting the effect, not the outcome level. A metric that's a good surrogate satisfies (roughly) the condition that the treatment affects the long-term outcome only through the surrogate. You can then build a composite index that's the best available predictor of long-run value, and use *that* as your OEC. This is what Netflix, Meta, and Amazon have all published versions of.

**Time-trend extrapolation within the test.** Fit the effect-by-cohort-day curve and extrapolate. Weak — extrapolation is doing all the work — but it's a useful sanity check on whether you're looking at something decaying or something building.

**Post-launch validation.** After shipping, use quasi-experimental methods (interrupted time series, synthetic control on the aggregate metric) to check whether the predicted gain shows up. Do this systematically across many launches and you learn your org's *calibration factor* — how much of predicted gain typically materializes. That number is enormously useful for planning and I'd want to know it.

**Structural reasoning.** Ask what mechanism would cause the effect to decay or grow. Novelty decays. Habit formation grows. Model-learning effects grow. Trust erosion compounds slowly and might not show up for a year. Reason it through before assuming persistence.

The organizational point I'd make: the gap between "sum of shipped wins" and "actual metric movement" is a measurable quantity, and measuring it is one of the highest-value things a principal DS can do, because it recalibrates the entire org's relationship with its own data.

---

### Q13.6 — How do you handle a test that ran during an unusual period — a holiday, an outage, a competitor event?

First, distinguish two situations.

**If the disruption hit both arms equally**, randomization still holds and the internal validity is fine. What's threatened is **external validity** — will the effect be the same in normal times? A checkout optimization tested during Black Friday might behave differently in March because the user mix and intent differ. So the estimate is unbiased *for that period* and may not generalize.

**If the disruption hit the arms differentially** — an outage that only affected one variant's infrastructure, a bug that broke treatment for two days — internal validity is broken and I'd treat the affected window carefully.

What I'd actually do:

- **Look at the effect over time.** If the effect during the anomalous period is similar to the effect outside it, the concern is mostly moot and I'd say so with the chart.
- **Analyze with and without the affected window**, pre-registering that I'll report both. If they agree, great. If they disagree, that's a finding about heterogeneity across contexts, not a nuisance.
- **Never drop the window silently.** Excluding inconvenient data post-hoc is p-hacking, and the fact that it was justified doesn't help if it wasn't declared. Report both and explain.
- **Check for SRM in the affected window** specifically — outages frequently cause differential dropout.
- **Consider re-running** if the disruption covered a large fraction of the test and the stakes are high.
- **Say the external validity caveat out loud** in the readout. "This was measured during peak season; we'd expect the effect to be somewhat smaller in a normal period, and I'd suggest a holdout to confirm."

---

# 14. Platform, Scale & Org Design

### Q14.1 — What does a good experimentation platform look like?

I'd describe it in layers.

**Assignment layer** — deterministic hashing so assignment is stable and reproducible, orthogonal seeds across experiments, layer/mutual-exclusion support, targeting rules, ramp scheduling, and instant kill-switch. Assignment must be logged at the moment of exposure, not inferred later.

**Instrumentation layer** — standardized event logging, and critically a **central metric definition layer** so "active user" means one thing across the company. The number of arguments this prevents is enormous.

**Analysis layer** — automatic SRM checks, trigger-based analysis, CUPED applied by default, correct variance for ratio metrics via delta method, clustered SEs where needed, sequential or always-valid inference, automated multiple-comparison handling, guardrails applied to every test whether or not the owner asked.

**Presentation layer** — a readout that leads with the pre-registered primary metric and its confidence interval, flags exploratory results as exploratory, shows effect over time by cohort day, and makes it hard to tell a misleading story. Design matters here; the default view shapes behavior more than any training does.

**Knowledge layer** — this is the one most orgs skip and the one I'd advocate hardest for. A searchable repository of every experiment ever run: hypothesis, design, result, decision, and follow-up. It gives you empirical effect-size priors for power calculations, prevents re-running tests that already failed, and lets you do meta-analysis across the portfolio. It's the thing that turns experimentation from a series of one-off decisions into a compounding asset.

**Governance layer** — pre-registration enforcement, review requirements scaled to risk, and automated post-launch validation.

If I could only build three things: SRM detection, CUPED, and the knowledge repository. Those give the largest returns per unit of engineering.

---

### Q14.2 — How would you improve experimentation velocity at a company without lowering quality?

I'd separate the two constraints, because "we need to move faster" and "we need to be right" are usually treated as a trade-off when much of the gain is on the efficiency frontier, not along it.

**Get more power from the same traffic** — CUPED, triggering, outlier capping, better metrics, variance-reduced estimators. Free throughput, no quality cost. Usually the biggest single win and the least politically fraught.

**Reduce time-to-launch** — self-serve tooling, templated experiment types, pre-approved metric sets, feature flags decoupled from deploys. Often the actual bottleneck is that it takes three weeks to get an experiment configured, not that tests run too long.

**Sequential designs with futility stopping.** Most experiments are null. Killing them at day 5 instead of day 21 frees enormous capacity.

**Tiered rigor by risk.** Not every test needs the full treatment. A copy change on a low-traffic page doesn't need a pre-registration review; a pricing change does. Match process to stakes and you free reviewer attention for where it matters.

**Factorial designs** so one traffic allocation answers multiple questions.

**Fix the interaction constraint.** If experiments are mutually exclusive by default, you're artificially serializing. Orthogonal assignment plus automated interaction detection usually unlocks a lot.

**Better prioritization.** Half the velocity problem is running low-value experiments. If you can predict which tests are likely to have large effects — using historical patterns by surface, by change magnitude, by team — you'd run fewer, better tests.

The framing I'd bring to leadership: velocity isn't tests-per-quarter, it's **decisions-per-quarter with adequate confidence**. Optimizing the first number without the second is how you end up with a large volume of underpowered tests and a false discovery rate above 50%, which is worse than not experimenting at all because it manufactures confident wrong beliefs.

---

### Q14.3 — How do you build an experimentation culture in an org that doesn't have one?

Start with a real decision that matters and where the intuition is confidently wrong. Nothing changes minds like a senior leader's favorite idea getting cleanly falsified — or validated, which builds trust in the process too.

Then, in rough order:

**Make the tooling frictionless.** If running a test is a two-week ordeal involving three teams, people will skip it and rationalize. Self-serve is a cultural intervention disguised as an engineering project.

**Make nulls safe.** Publicly celebrate a well-run test that killed a bad idea, and quantify what it saved. If the only rewarded outcome is a win, you've built an incentive to manufacture wins.

**Educate through the readout, not through training.** Nobody remembers a stats course. But if every experiment readout follows the same template with confidence intervals and pre-registered metrics front and center, people absorb the framework by osmosis.

**Publish the meta-numbers.** Ship rate, average power, how many launches validated post-hoc, how much of predicted impact materialized. This is the argument that gets you investment, and it also keeps the program honest.

**Embed, don't gatekeep.** A central team that reviews everything becomes a bottleneck and gets routed around. A central team that builds the platform, sets standards, and consults on hard designs scales.

**Pick the right battles.** You will lose some fights about shipping things that shouldn't ship. Losing gracefully and documenting the prediction is fine — sometimes the best long-term move is to let something ship over your objection and then show the data six months later.

At L6 specifically: my job isn't to be the best analyst, it's to make the *median* analysis in the org good. That's a systems and standards job.

---

### Q14.4 — How do you decide what to experiment on versus just ship?

I'd use a rough expected-value framing across three variables:

**Reversibility.** Easy to roll back, low blast radius → bias toward shipping and monitoring. Hard to reverse (data migrations, pricing announcements, anything users will notice being taken away) → test.

**Uncertainty.** If everyone's confident and the change is a clear bug fix or a well-understood pattern, testing buys little information. If there's genuine disagreement among smart people, that disagreement is exactly the signal that a test is worth running. I'd literally ask the room to write down their predicted effect — spread in predictions is a great, cheap indicator of information value.

**Stakes.** Impact on revenue, trust, or a large user population raises the bar for shipping blind.

Plus two practical filters: **can you even power it?** and **what's the opportunity cost of the traffic?**

So: high uncertainty + high stakes + testable → definitely test. Low uncertainty + low stakes + reversible → ship and monitor. High stakes + can't test → quasi-experimental design, staged geo rollout, or a very careful ramp with pre-committed rollback criteria.

The point I'd make: a mature org needs both modes, and being dogmatic about testing everything is its own failure. Experimentation capacity is a scarce resource and spending it on foregone conclusions means not spending it on genuine uncertainty.

---

# 15. Case Studies & Scenario Questions

### Q15.1 — Your test shows a 5% lift in engagement and a 2% drop in revenue. What do you do?

I'd work through it in this order and say so out loud, because the process is the answer here.

**Step 1 — Are both real?** Check the confidence intervals, SRM, and whether the revenue drop is driven by a few whales. Revenue is heavy-tailed; a 2% drop with an interval of [−6%, +2%] is very different from [−2.5%, −1.5%]. I'd also check whether revenue was even powered — often it isn't, and the "drop" is noise on a metric we can't measure well in this window.

**Step 2 — What's the mechanism?** Engagement up and revenue down usually means one of a few things: users are engaging with cheaper content, the change surfaced free alternatives to paid ones, or it increased browsing at the expense of purchase intent. I'd decompose revenue into conversion × frequency × order value and see which component moved. That decomposition usually tells the story immediately.

**Step 3 — Is this a substitution or a destruction?** If users are shifting from paid to free content but total sessions are up, that might be a long-term win — more engaged users monetize later. If they're just buying less with no offsetting behavior change, it's a loss. This distinction requires the long-term view, which brings me to:

**Step 4 — What does the time trend say?** If the revenue gap is narrowing over the test, it might be a transition cost. If it's widening, it's structural and worse than it looks.

**Step 5 — Convert to a common currency.** Estimate the annualized value of the engagement lift using the org's established engagement-to-revenue relationship (and if we don't have one, that's a gap I'd want to fix). Compare to the annualized revenue loss. Now it's arithmetic.

**Step 6 — Is there a fix?** Often the revenue drop comes from one specific component of the change, and there's a version that keeps the engagement gain without it. I'd look for that before accepting the trade-off as given.

**My recommendation shape:** "Don't ship as-is. The revenue loss is real and larger in dollar terms than the engagement gain is worth under our current conversion assumptions. But the engagement mechanism works — here's evidence — so I'd propose variant B that keeps the surfacing change without the placement change, and I'd want a four-week test with a revenue-powered design."

What I'd emphasize in the interview: at L6 the value I add isn't picking a side, it's making sure the org doesn't quietly drop the inconvenient metric, and making the trade-off explicit enough that the right person can decide fast.

---

### Q15.2 — You launched a feature that tested positive, but the company metric hasn't moved. Explain.

There are a lot of possible explanations and I'd systematically work through them rather than guess.

**Measurement/scale explanations:**
- **Dilution.** The feature affects 5% of users; a 4% lift there is a 0.2% company-level move, which is invisible in the noise of the aggregate metric. Often this is the whole answer and the arithmetic settles it in five minutes.
- **The company metric is noisy.** Weekly aggregate revenue has enormous variance; a 0.2% shift is undetectable without a proper counterfactual. Absence of a visible move in a time series is not evidence of no effect.

**Effect-decay explanations:**
- **Novelty.** Effect was real in week one, gone by week six.
- **Winner's curse.** The true effect was smaller than measured.

**Interference explanations:**
- **Cannibalization.** The feature moved activity from another surface rather than creating new activity. Net zero. Very common and it's the reason user-level tests overstate marketplace and content-recommendation wins.
- **Interference in the test** made the measured effect an overestimate of the full-rollout effect.

**Implementation explanations:**
- The shipped version differs from the tested version — a "small" change during rollout, a different config, a performance regression at full scale.
- Rollout is incomplete; not everyone actually has it.

**Offsetting factors:**
- Something else got worse simultaneously — a competitor, seasonality, another team's launch. The feature might be holding the line, not failing.

**How I'd actually investigate:** start with the arithmetic (is the expected company-level move even detectable?), then check the shipped-vs-tested implementation, then look for cannibalization by examining whether the *source* surface declined, then check the long-term holdout if we have one. And if we don't have a holdout, this is exactly the argument for why we should.

The answer I'd give: "Most likely this is dilution plus some cannibalization, and I can check both in a day. But the honest structural answer is that we can't distinguish these possibilities without a holdout, and I'd want to establish holdouts as standard practice so we're not having this conversation again next quarter."

---

### Q15.3 — Design an experiment to measure the impact of a new pricing algorithm in a two-sided marketplace.

I'd walk through my reasoning rather than jumping to an answer.

**First, why user-level A/B fails here.** Pricing affects supply-demand equilibrium. If some riders get a different price, the drivers they attract or repel change availability for everyone else. Interference is severe and structural, and the bias would likely be in the direction that makes the algorithm look good. So individual randomization is off the table.

**Second, what's the natural unit?** Pricing is set at the market level and equilibrium is market-level. So either geography or time.

**My primary design: switchback by city-hour, stratified.**
- Randomize each city independently on hourly blocks, stratified by day-of-week and hour-of-day so each condition gets balanced coverage of peak and off-peak.
- Measure the equilibration time first — from a pilot or historical data on how long after a pricing change the system reaches steady state — and drop that as burn-in from each block.
- Run across multiple cities to increase effective sample and enable city fixed effects.
- Analyze at block level with day-of-week, hour-of-day, and city fixed effects; cluster standard errors by block; check autocorrelation.

**Backup / complementary design: geo experiment with synthetic control.** Pair-match cities on pre-period trends in the key metrics, treat one of each pair. Slower to read but captures effects that persist beyond a switchback block — driver supply responding to earnings over weeks, rider habit formation. Switchbacks fundamentally can't measure those because the condition flips too fast for slow-moving behavior to respond.

I'd probably argue for running both: the switchback for the fast equilibrium effects and a longer geo test for the slow supply-side effects. They answer different questions and the difference between them is itself informative.

**Metrics:**
- Primary: contribution margin or net revenue per unit time — the actual objective, since pricing trades volume against margin.
- Both sides: rider completion rate, rider price sensitivity, driver earnings per hour, driver utilization, driver session length and retention.
- Guardrails: cancellation rate, ETA, unfulfilled request rate, price complaint volume, and equity checks across neighborhoods — pricing algorithms have real fairness exposure and I'd want that instrumented from day one, not added after a press story.
- Diagnostics: an A/A switchback first to validate the analysis pipeline.

**Duration:** driven by block count needed for power (from historical block-level variance), whole weeks, minimum two, and I'd want a full month for the geo arm to catch supply response.

**Risks I'd flag upfront:** carryover from repositioned drivers, the fact that switchbacks can't capture multi-week supply elasticity, media/regulatory attention on visible price variation within a city, and the possibility that riders notice inconsistent pricing across hours — which is both a trust risk and a potential contaminant if it changes behavior.

---

### Q15.4 — A senior leader wants to launch based on a result you think is unreliable. How do you handle it?

I'd separate the technical issue from the relationship issue, because handling only one fails.

**First, understand what they actually want.** Sometimes "I want to launch" is really "I have a commitment to a customer next month" or "I don't believe the metric captures the value." If the concern is strategic rather than statistical, arguing about p-values misses the point entirely and makes me look like an obstacle.

**Second, be precise about my objection, and quantify it.** Not "the result isn't reliable" — that's vague and easy to dismiss. Instead: "The test had a sample ratio mismatch of 51/49 with p < 10⁻⁵, which means assignment was broken; in the three previous cases where we fixed an SRM, the effect estimate changed by more than half, and one flipped sign. So I'd put maybe 40% confidence on the direction, not 95%." A specific, quantified concern is much harder to wave away and much easier to act on.

**Third, offer a path, not just a veto.** "Give me four days to find the SRM cause and we'll re-read." Or "let's ship to 20% with the guardrails wired to auto-rollback and re-measure in two weeks." Or "let's ship but keep a 5% holdout so we learn either way." My job is to reduce the risk of the decision, not to block the decision.

**Fourth, if they still want to go, make the prediction explicit and write it down.** "My estimate is that this delivers between −1% and +1%, not the +4% in the readout. Let's agree now on how we'll check in Q3." That does two things: it protects the org from mis-learning ("we shipped it and everything's fine" becoming institutional knowledge), and it creates the feedback loop that eventually earns credibility.

**Fifth, escalate only when the stakes genuinely warrant it** — user harm, regulatory exposure, or a number going into external reporting. Escalating over an ordinary product call burns capital I'll need later.

The judgment I'd articulate: I'm not the decision-maker, I'm the person responsible for making sure the decision is made with an accurate picture of the uncertainty. If I've done that clearly and they still choose differently, that's a legitimate outcome — they may be weighing things I don't see. What's not acceptable is letting an unreliable number get laundered into a fact.

---

### Q15.5 — How would you measure the impact of a brand marketing campaign that you can't randomize at the user level?

Brand campaigns are the hardest measurement problem in the field — broad reach, diffuse and delayed effects, no clean control. I'd give a layered answer.

**Best available: geo-based randomized holdout.** Randomize DMAs or cities into treated and untreated, run the campaign only in treated markets. This is a real randomized experiment at the geo level, so it's causal. Design details: pair-match markets on pre-period trends and size, use enough markets (30+ if possible), and analyze with matched-pair or synthetic-control estimators. Power is the binding constraint since you have few units and enormous between-market variance, so I'd expect to detect only fairly large effects and I'd state that MDE upfront.

**If you can't withhold everywhere: synthetic control or CausalImpact.** Pick a few markets to hold out (or use markets where the campaign didn't run for operational reasons), construct a synthetic counterfactual from the donor pool, and measure the gap. Weaker than randomization but often the practical answer.

**Ghost ads / PSA controls** for the digital component, where the platform can identify who *would* have been served.

**Media mix modeling (MMM)** — a time-series regression of outcomes on spend across channels, with adstock (carryover of advertising effect over time) and saturation curves. Its virtue is that it covers all channels including offline and gives you a budget allocation answer. Its weakness is severe: it's observational, spend is endogenous (you spend more when you expect demand), channels are collinear, and results are highly sensitive to specification. I'd use MMM for budget planning, not for causal claims about a specific campaign, and I'd calibrate it against geo experiments wherever I have them. That calibration — using experimental results as priors or constraints in the MMM — is the current best practice and it's what makes MMM defensible.

**Triangulate.** Geo experiment for the causal anchor on a few campaigns, MMM for full-portfolio allocation calibrated to those anchors, brand-lift surveys for the awareness mechanism, and a long observation window because brand effects are slow.

The point I'd make: with brand, the right expectation is "reduce uncertainty from enormous to large," not "get a precise number." Being honest about that upfront is better than delivering a precise-looking MMM coefficient that's actually driven by a specification choice.

---

### Q15.6 — Your model team wants to A/B test a new ML ranking model. What's different about testing models?

Several things that don't come up in UI tests.

**Feedback loops and model learning.** The new model changes what users see, which changes what they click, which changes the training data. If the model retrains during the test, it's improving inside the treatment arm in a way it couldn't at full rollout — or the control model is degrading because its training data now reflects mixed exposure. I'd either freeze both models during the test or explicitly plan for the learning curve and run long enough to see it plateau.

**Shared training data is an interference channel.** If both arms feed one training pipeline, treatment behavior contaminates the control model. This is a genuine SUTVA violation and it's easy to miss because it's invisible in the assignment logic.

**Effects build rather than decay.** Unlike UI novelty, personalization models often get *better* over time as they accumulate signal on each user. So the two-week read understates the steady-state effect. This is a case where I'd argue strongly for a longer run and a holdout, and where a cohort-day effect plot is essential.

**Offline metrics don't predict online outcomes reliably.** NDCG going up doesn't mean engagement goes up. I'd want to know the historical correlation between offline gains and online gains for this team — if it's weak, offline evaluation should be a filter, not a decision.

**Position and presentation bias.** If the new model reorders results, click metrics change mechanically because of position effects, not because relevance improved. I'd want position-debiased metrics or an interleaving experiment.

**Interleaving as an alternative.** For ranking specifically, interleaving — blending results from both rankers into one list and seeing which ranker's items get clicked — is dramatically more sensitive than A/B, often needing 10-100x less traffic. It controls for the user completely since each user sees both. The catch is that it measures relative ranking preference, not the downstream business outcome, so I'd use interleaving to narrow the candidate set quickly and then A/B the finalist for the real decision.

**Guardrails specific to models:** latency (inference cost), coverage (does it fail to return results for some queries?), diversity and concentration (is it collapsing onto a small set of popular items?), and fairness across content creators or user groups. A ranking model that boosts engagement by concentrating traffic on the top 1% of creators can be a long-term ecosystem problem that no engagement metric will catch.

---

# 16. Leadership & Influence (L6-specific)

### Q16.1 — Tell me about a time you changed how an organization made decisions.

*(Prepare your own STAR story here — the structure below is what a strong L6 answer looks like.)*

The shape I'd aim for:

**Situation** — a systemic problem, not a one-off analysis. "The org was shipping 60% of tests as wins, but the annual metric was only moving a fraction of the sum of those wins."

**Task** — I made it my problem even though nobody assigned it.

**Action** — diagnosed the root causes with data (measured actual power across the portfolio, found the median test had 35% power; measured false discovery rate implied by our hit rate; ran a corpus analysis showing effect-size inflation). Then built the fix as a system, not a memo: CUPED in the platform, automated SRM checks, pre-registration in the launch flow. Then got adoption through a pilot with one friendly team, showed the numbers, and let them evangelize.

**Result** — quantified. Effective sample size up X%, false discovery rate down, and — the number that mattered — the gap between predicted and realized annual impact narrowed from A to B.

The things that make this an L6 answer rather than an L5 one: you identified a systemic issue nobody asked you to look at, you quantified it rather than asserting it, you built a durable mechanism rather than a one-time analysis, you got adoption through influence rather than mandate, and you measured whether the fix worked.

---

### Q16.2 — How do you explain a null result to a team that's been building for six months?

I'd think about this in three moves.

**Reframe what a null means.** It's not "your work was wasted," it's "we learned this specific hypothesis about user behavior is wrong, before we spent another six months on it." I'd try to have the language for this established *before* the test reads out, in the pre-registration conversation — "here's what we'll conclude if it's flat" — so the null isn't a surprise verdict but an outcome we already agreed how to interpret.

**Be precise about what we actually learned.** A null is not proof of no effect. "The confidence interval is [−0.4%, +0.9%], so we can rule out anything bigger than about 1%. Given the maintenance cost, that's below our ship bar. What we can't rule out is a small positive effect, and we also can't rule out that it works for new users specifically — the point estimate there is +2.1% with a wide interval." That converts a demoralizing "no" into a specific, actionable map of what's known and unknown.

**Extract the forward value.** What does the mechanism tell us? If the intermediate metrics moved but the outcome didn't, the mechanism works and the link to the outcome is broken — that's a genuinely valuable finding that redirects the next six months. If nothing moved at all, the users may not have noticed the change, which is a different lesson.

And on the human side: I'd deliver it in person before it hits a dashboard, I'd credit the quality of the work independent of the outcome, and I'd make sure the null gets written up and shared as a real contribution. The way the org treats this moment determines whether anyone ever runs an honest test again.

---

### Q16.3 — How do you set standards for experimentation across teams without becoming a bottleneck?

Three mechanisms, in order of leverage.

**Encode standards in tooling.** The highest-leverage version of a standard is one that's a default. If the platform automatically applies CUPED, checks SRM, and requires a declared primary metric before launch, you've enforced three standards with zero review meetings. Every standard I can move from a document into code, I do.

**Tier the review by risk.** Most experiments don't need a human reviewer. Define the criteria that trigger review — pricing, trust and safety, anything going into external reporting, anything with a novel design — and let everything else self-serve. This concentrates scarce senior attention where it changes outcomes.

**Build capability rather than dependency.** Office hours, a design-review forum where teams present to each other rather than to me, a set of worked templates for the five most common experiment shapes, and deliberately mentoring 2-3 strong analysts per org area who become the local experts. The measure of success is that the number of questions routed to me goes *down* while quality goes up.

The failure mode I'd name explicitly: a central team that reviews everything becomes a queue, and queues get routed around. The experiments that skip your review are disproportionately the rushed, high-pressure ones — exactly the ones that most need it. So being a bottleneck isn't just slow, it's actively selective against your own effectiveness.

---

### Q16.4 — How do you prioritize what to work on as a Principal DS?

I'd think about leverage across three axes.

**Breadth of impact.** Does this change one decision, one team's decisions, or how the whole org decides? A platform capability that improves every test beats a brilliant analysis of one test, unless that one test is a very large bet.

**Durability.** Does it decay the moment I stop pushing? Standards encoded in tooling persist. Analyses get read once. I bias toward things that keep paying out.

**Counterfactual.** Would this happen without me? If a strong L5 will handle it well, my time is better spent elsewhere — even if the problem is interesting. This is the discipline that's hardest to actually practice, because the interesting problems are magnetic.

Then two more filters: **the risk-weighted stuff nobody owns** — the measurement problem sitting between two teams, the metric everyone uses but nobody validated — is usually where a principal adds the most, because it's structurally under-owned. And **strategic alignment** — the highest-leverage technical work on a dying initiative is worth less than adequate work on the company's central bet.

Practically I'd try to keep a rough split: a meaningful chunk on systemic/platform work, a chunk on the highest-stakes individual decisions where my judgment actually changes the outcome, and a chunk on developing other people. The exact ratio depends on org maturity — in a young experimentation program the platform work dominates; in a mature one, the marginal value shifts toward hard individual decisions and mentorship.

---

### Q16.5 — Questions to ask your interviewer

Asking good questions is part of the evaluation at this level. Some that signal seniority:

- How do you currently measure whether shipped features deliver the impact the tests predicted? Is there a known gap?
- What fraction of experiments reach a decision-grade level of power? Do you measure that?
- How do you handle interference — is there a standard approach for marketplace/network effects, or is it case-by-case?
- What's the relationship between the central experimentation team and product teams — platform, consulting, gatekeeping, or all three?
- What's the biggest measurement problem you currently can't solve?
- Where has the org gotten burned by an experiment result that turned out to be wrong? What changed after?
- How are long-term effects handled — do you run holdouts?
- What would you want the person in this role to have changed twelve months from now?

That last one is the most useful question you can ask in any senior interview.

---

## Appendix A — Formulas Worth Having Memorized

**Sample size per arm (two-sided, α=0.05, power=80%):**
n ≈ 16σ²/Δ²  → for proportions, n ≈ 16·p(1−p)/Δ²

**General form:** n = 2(z_{1−α/2} + z_{1−β})²σ²/Δ², where (1.96+0.84)² ≈ 7.85

**MDE given n:** Δ ≈ 4σ/√n  (halving MDE requires 4× the sample)

**CUPED adjusted metric:** Y_adj = Y − θ(X − X̄), θ = Cov(Y,X)/Var(X)
Variance reduction factor: (1 − ρ²)

**Design effect for clustering:** DEFF = 1 + (m − 1)·ICC, where m = cluster size
Effective sample size = n / DEFF

**Standard error of a difference in proportions:**
SE = √[p₁(1−p₁)/n₁ + p₂(1−p₂)/n₂]

**Delta method variance for a ratio metric R = X/Y:**
Var(R) ≈ (1/μ_Y²)·Var(X) + (μ_X²/μ_Y⁴)·Var(Y) − 2(μ_X/μ_Y³)·Cov(X,Y)

**DiD estimate:** (T_after − T_before) − (C_after − C_before)

**Bayes for conversion (Beta-Binomial):** posterior = Beta(α + successes, β + failures)

**Relationship between chi-square and z for 2×2:** χ² = z²

---

## Appendix B — Terms & One-Line Definitions (Quick Scan)

| Term | Plain meaning |
|---|---|
| OEC | The single primary metric the experiment is judged on |
| MDE | Smallest true effect the test can reliably detect |
| SRM | Observed traffic split differs from intended — a bug signal |
| SUTVA | Assumption that one unit's treatment doesn't affect another's outcome |
| ITT | Analyze by assigned group, regardless of actual exposure |
| LATE / CACE | Effect among people whose behavior the assignment actually changed |
| CUPED | Using pre-experiment data to cancel out noise and gain power |
| ICC | How similar units within a cluster are; drives cluster-design power loss |
| Design effect | Factor by which clustering inflates required sample size |
| Delta method | Correct way to get variance for ratio metrics |
| Winner's curse | Shipped effects look bigger than they are, because you select on noise |
| Dilution | Measuring on a population bigger than the treatment can affect |
| Triggering | Restricting analysis to users who actually hit the affected surface |
| Novelty effect | Engagement from newness that decays |
| Primacy effect | Initial degradation from having to relearn, which recovers |
| Parallel trends | DiD's key assumption: groups would have moved together absent treatment |
| Synthetic control | A weighted blend of untreated units built to mimic the treated one |
| Propensity score | Probability of being treated given observed characteristics |
| Overlap / positivity | Requirement that comparable treated and untreated units exist |
| Doubly robust | Consistent if either the treatment model or outcome model is correct |
| Fixed effects | Per-unit intercepts that absorb all time-invariant unit differences |
| Random effects | Unit effects drawn from a distribution; efficient but needs exogeneity |
| TWFE | Two-way fixed effects — unit and time |
| Staggered DiD problem | TWFE gives biased estimates when adoption times and effects vary |
| Clustered SEs | Standard errors that respect that rows within a group aren't independent |
| FWER | Probability of any false positive across a family of tests |
| FDR | Expected share of your "discoveries" that are false |
| Alpha spending | Budgeting Type I error across multiple planned looks |
| Confidence sequence | Interval valid at every point in time — safe to peek |
| mSPRT | A common construction for always-valid sequential testing |
| Posterior | Updated belief about the effect after seeing the data |
| Credible interval | Range containing the effect with stated probability |
| Expected loss | Average cost of shipping the wrong arm — a Bayesian stopping rule |
| MCMC / NUTS | Sampling methods for posteriors with no closed form |
| R-hat | MCMC convergence check; should be ~1.00 |
| CATE | Treatment effect for a user with given characteristics |
| Qini curve | Evaluation curve for uplift/personalization models |
| Interleaving | Blending two rankers' results in one list; very sensitive for ranking tests |
| Ghost ads | Recording who would have won an ad auction without serving the ad |
| Switchback | Randomizing time blocks instead of users |
| Carryover | Effects of one time block bleeding into the next |
| Burn-in / washout | Discarding the start of each block to remove carryover |
| Non-inferiority test | Testing that something isn't worse by more than a set margin |
| DML | ML for nuisance functions + orthogonalization + cross-fitting |
| E-value | How strong an unmeasured confounder would need to be to explain a result |

---

## Appendix C — Gaps to Fill (Add As You Go)

Topics worth adding as you keep building this:

- Experiment design for **low-traffic / B2B / enterprise** contexts (few accounts, big variance)
- **Ratio metrics and the delta method** worked through numerically
- **Survey and self-reported metrics** in experiments; brand lift design
- **Trust & safety / integrity experiments** where the harm metric is rare
- **Diff-in-diff-in-differences (DDD)** and triple differences
- **Matrix completion / factor models** for panel counterfactuals
- **Adaptive experimental design** and Bayesian optimization for continuous parameters
- **Experimentation ethics** — informed consent, vulnerable populations, IRB-style review
- **Privacy-constrained measurement** — differential privacy, aggregated APIs, cookie deprecation and its effect on assignment
- **Causal DAGs** and using them to justify covariate selection (backdoor criterion, colliders, mediators)
- **Mediation analysis** — decomposing an effect into paths, and why naive mediation is biased
- **Sensitivity analysis** for unobserved confounding in depth (Rosenbaum bounds, E-values)
- **Forecasting and counterfactual simulation** for decisions that can't be tested at all
- Your own **STAR stories** for the leadership section — at least four, quantified

---

*Last updated: build 1. Keep adding — the goal is that by the interview you can answer anything in here without looking, and you know which three stories you'll tell.*
