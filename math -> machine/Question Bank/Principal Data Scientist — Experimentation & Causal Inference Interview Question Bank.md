
## How to use this bank

These answers are intentionally written as **spoken interview answers** rather than textbook definitions. The goal is to sound like someone who understands not only *how* a method works, but also **when to use it, what can go wrong, and what decision it supports**.

For Principal-level interviews, I would keep returning to four questions:

1. **What exactly is the causal question?**
2. **What assumptions make the design valid?**
3. **What could invalidate the experiment or bias the estimate?**
4. **What decision will we make from the result?**

A strong Principal answer usually moves beyond “the p-value is below 0.05” and discusses the **estimand, design, validity, power, practical significance, uncertainty, and business decision**.

---

# PART I — EXPERIMENTATION FUNDAMENTALS

## Q1. What is an A/B test, and why does it allow us to make a causal statement?

### Answer

An A/B test is a randomized experiment where we assign units, usually users, to a treatment group and a control group and compare their outcomes.

The important part is not actually the “A/B” part. It is the **randomization**.

Because treatment is randomly assigned, before the experiment starts the two groups should be comparable on average. So, after the experiment, a systematic difference in outcomes can be attributed to the treatment, assuming the experiment itself was implemented correctly.

So if treatment users have a 5% higher conversion rate, I can say something like:

> “The treatment caused an estimated 5% increase in conversion, under the assumptions of the experiment.”

That is fundamentally different from observing that people who use a feature have higher conversion. The latter is just correlation because feature users may already be different from non-users.

---

## Q2. What makes a randomized experiment internally valid?

### Answer

I would think about internal validity as asking whether our estimated treatment effect really represents the effect of the intervention we intended to test.

There are several things I would check.

First, ==**randomization** needs to work correctly.==

Second, treatment and control need to actually receive the intended experiences. ==If treatment users are accidentally exposed to control, we have **contamination**.==

Third, we need to consider ==**interference**, where one user's treatment changes another user's outcome.==

Fourth, the ==outcome measurement and logging need to be reliable.==

Fifth, we need to handle things like ==attrition, missing data, noncompliance and sample-ratio mismatch.==

And finally, the ==analysis needs to respect the experiment design.== For example, if we randomized at the cluster level but analyze users as independent observations, we can ==underestimate uncertainty.==

So I don't think of experiment validity as just “we randomized users.” Randomization is the foundation, but implementation and analysis have to preserve that randomization.

---

# PART II — DEFINE THE CAUSAL QUESTION FIRST

## Q3. What is an estimand?

### Answer

An **estimand** is the exact causal quantity we are trying to estimate.

For example:

> “What is the average effect of showing the new checkout page to all eligible users for 30 days?”

That's much more precise than saying:

> “Does the new checkout page work?”

The estimand tells us things like:

- who the target population is,
- what treatment is being compared,
- what outcome we're measuring,
- over what time horizon,
- and what effect we actually care about.

This becomes especially important when people don't comply with treatment, when there are multiple treatment versions, or when treatment effects vary substantially between user groups.

At Principal level, I would define the estimand before choosing the statistical test.

---

## Q4. What is the difference between ATE, ATT, and CATE?

### Answer

**ATE**, or Average Treatment Effect, is the average causal effect across the target population.

**ATT**, or Average Treatment Effect on the Treated, is the effect specifically among the people who actually received treatment.

**CATE**, or Conditional Average Treatment Effect, is the average treatment effect for a particular subgroup defined by characteristics.

For example, suppose we're testing a recommendation system.

The ATE might say:

> “The new model increases orders by 1.2% across all eligible users.”

A CATE analysis might tell us:

> “The effect is substantially larger for new users than existing users.”

That distinction matters because an overall average can hide important heterogeneity.

---

## Q5. What is the fundamental problem of causal inference?

### Answer

For each user, we can observe only one outcome.

For a particular user, we can either observe what happened under treatment or what happened under control. We can't observe both for the same person at the same time.

Those two hypothetical outcomes are called **potential outcomes**.

The causal effect for an individual would be:

$$[
Y_i(1)-Y_i(0)
]$$

But one of those two values is always missing.

An experiment solves this by creating a credible comparison group that approximates the counterfactual outcome.

That's the core idea behind basically every causal inference method.

---

# PART III — SUTVA AND INTERFERENCE

## Q6. What is SUTVA?

### Answer

**SUTVA** stands for Stable Unit Treatment Value Assumption.

In practical terms, it has two important ideas.

First, there shouldn't be hidden versions of the treatment. The treatment should mean the same thing across units.

Second, there should be **no interference** between units.

No interference means my outcome shouldn't depend on whether another user received treatment.

For example, suppose I'm testing a marketplace ranking algorithm. If I change the ranking for one group of users, that could change seller behavior, inventory, prices, or supply available to another group.

Then the treatment assigned to one user affects another user's outcome.

That violates the usual no-interference assumption behind a standard user-level A/B test.

---

## Q7. Why does SUTVA matter so much in experimentation?

### Answer

Because standard A/B analysis implicitly assumes that a user's outcome is determined by their own treatment assignment, not by everyone else's assignment.

If that isn't true, the control group isn't really a clean counterfactual.

Imagine treatment users start consuming more inventory. Control users now face less inventory.

The treatment group has changed the environment experienced by the control group.

So the estimated difference may no longer represent:

> treatment versus an untouched control world.

This is why marketplaces, social networks, advertising ecosystems, ride-hailing and other interconnected systems are particularly challenging.

---

## Q8. What would you do if interference is likely?

### Answer

I would change the **unit of randomization** so that interacting units tend to receive the same treatment.

For example, instead of randomizing individual users, I might randomize:

- geographic regions,
- social networks,
- sellers,
- drivers,
- stores,
- or time blocks.

Another option is a design specifically intended to estimate direct and spillover effects.

The trade-off is that larger randomization units usually mean fewer independent units, which can substantially reduce statistical power.

So I wouldn't simply say “randomize by region.” I'd first quantify the expected interference and then work out the power implications.

---

# PART IV — EXPERIMENT UNIT AND RANDOMIZATION

## Q9. How do you decide what the unit of randomization should be?

### Answer

I start with the mechanism of treatment and ask:

> “What is the smallest unit for which treatment can be independently assigned without contamination?”

If individual users can receive treatment independently, user-level randomization is attractive because it gives us many independent units.

But if users interact strongly, I may need cluster randomization.

For example, in a marketplace, if changing treatment for one user affects inventory available to another, individual-level assignment can create interference.

Then a city, market, or another meaningful cluster may be more appropriate.

So I wouldn't optimize only for statistical power. I would optimize for **causal validity first, and then power within a valid design**.

---

## Q10. What is contamination in an A/B test?

### Answer

Contamination means users assigned to one condition are exposed to part of another condition.

For example, 10% of treatment users accidentally see the old experience.

This makes the treatment and control groups more similar, so the observed effect is usually pulled toward zero.

That's called **effect dilution**.

At experiment launch, I would therefore monitor treatment exposure directly rather than assuming assignment implies exposure.

---

## Q11. What is noncompliance?

### Answer

Noncompliance occurs when users don't follow their assigned treatment.

For example, a user is assigned to treatment but doesn't receive the feature.

The standard primary analysis in a randomized experiment is usually **Intent-to-Treat**, or ITT.

ITT compares users according to their original assignment, regardless of whether they actually complied.

The reason is that assignment was randomized. Once I condition on actual exposure, I can introduce selection bias.

If the business question is specifically about the effect among people who actually use the treatment, then I may need additional methods such as an instrumental-variable framework.

---

## Q12. What is ITT and why is it important?

### Answer

**Intent-to-Treat** means I analyze users according to the group they were randomized into.

So if a user was assigned treatment, they stay in treatment even if they never interacted with the feature.

This preserves the benefit of randomization.

It answers:

> “What happens if we offer or assign this treatment?”

rather than:

> “What happens among people who actually used it?”

That distinction is critical.

---

# PART V — A/B EXPERIMENT DESIGN

## Q13. Walk me through how you would design a new A/B test.

### Answer

I'd start before touching the statistics.

First, I'd clarify the product decision and define the estimand.

Then I'd choose:

1. eligibility criteria,
2. randomization unit,
3. treatment and control,
4. primary metric,
5. guardrail metrics,
6. minimum meaningful effect,
7. significance level,
8. desired power,
9. expected traffic,
10. expected duration.

I'd also think through interference, contamination, novelty effects, seasonality, logging correctness and interactions with other experiments.

Then I'd conduct an A/A test or another validation step if the platform or instrumentation is new.

After launch, I'd monitor experiment health first—especially assignment, exposure, data quality and guardrails—before interpreting treatment effects.

Only after the experiment is healthy would I focus on whether the treatment “won.”

---

## Q14. What is a primary metric versus a guardrail metric?

### Answer

The **primary metric** is the main outcome used to decide whether the experiment succeeded.

A **guardrail metric** is something we don't necessarily want to optimize, but we don't want the treatment to damage it materially.

For example, a recommendation change might aim to improve conversion.

The primary metric could be conversion rate.

Guardrails could include:

- latency,
- cancellation rate,
- customer complaints,
- revenue per user,
- or seller experience.

The important principle is that we should define these before looking at results.

Otherwise teams can unconsciously search for the metric that tells the most favorable story.

---

## Q15. What is an Overall Evaluation Criterion or OEC?

### Answer

An **OEC**, or Overall Evaluation Criterion, is a higher-level measure of whether the experiment produced the desired overall outcome.

It can combine multiple aspects of user and business value.

The reason I like the concept is that optimizing a single local metric can create unintended consequences.

For example, increasing clicks isn't necessarily valuable if those clicks produce fewer completed transactions.

So I'd distinguish:

> “Did the feature move our local metric?”

from:

> “Did it improve the overall outcome we care about?”

---

# PART VI — POWER, MDE AND SAMPLE SIZE

## Q16. What is statistical power?

### Answer

Power is the probability that the experiment detects an effect of a specified size when that effect actually exists.

More formally:

\[
Power = 1-\beta
\]

where β is the probability of a Type II error—failing to detect a real effect.

So if I design an experiment for 80% power to detect a 2% relative improvement, I'm saying:

> “Assuming the true effect is 2%, this design has an 80% chance of detecting it under the chosen testing procedure.”

Power depends on things like effect size, variance, sample size and significance threshold.

---

## Q17. What is MDE?

### Answer

**MDE** means Minimum Detectable Effect.

It's the smallest effect that our experiment is designed to reliably detect at a chosen significance level and power.

Suppose our baseline conversion rate is 10%.

If our MDE is 2% relative, we're trying to detect roughly:

\[
10\% \rightarrow 10.2\%
\]

with our chosen power and alpha.

MDE is therefore not a statement about what effect exists.

It's a statement about what effect our experiment is capable of detecting reliably.

---

## Q18. How do you choose an MDE?

### Answer

I wouldn't choose it purely because the resulting sample size looks convenient.

I'd start with the business decision.

I'd ask:

> “What is the smallest improvement that would actually justify shipping the feature?”

That's effectively the **practically significant effect**.

Then I translate that into an MDE.

There is no point designing an enormous experiment to detect an effect so small that it wouldn't change the business decision.

Conversely, choosing an unrealistically large MDE just to make the experiment cheap means we may miss effects that actually matter.

---

## Q19. How do you calculate the required sample size?

### Answer

At a high level, sample size is determined by:

- baseline variance or baseline conversion rate,
- target effect or MDE,
- significance level α,
- desired power,
- allocation ratio,
- and the statistical test or estimator being used.

For a simple two-group mean comparison, we can solve the power equation for sample size.

For a binary metric we'd generally work with the baseline event rate and the expected difference in proportions.

But in real experimentation, I would be careful about using a textbook formula blindly.

If the metric is clustered, autocorrelated, highly skewed, a ratio metric, or the randomization is at the cluster level, the effective sample size can be very different from the raw number of users.

---

## Q20. What happens if I halve the MDE?

### Answer

The experiment becomes dramatically more expensive.

For many standard settings, required sample size grows approximately with the inverse square of the effect size:

\[
n \propto \frac{1}{MDE^2}
\]

So detecting an effect half as large can require roughly four times the observations, all else equal.

That's why choosing a sensible MDE is one of the most important practical design decisions.

---

## Q21. How do you choose experiment duration?

### Answer

I would separate **sample-size duration** from **behavioral duration**.

Sample-size duration asks:

> “How long does it take to collect enough independent information?”

Behavioral duration asks:

> “How long do users need to experience the treatment before the effect stabilizes?”

I need both.

I would estimate traffic from historical data and calculate how many users or clusters are needed for the target MDE.

Then I'd check whether we have enough calendar time to cover important cycles—such as weekday versus weekend behavior or weekly seasonality.

I'd also look for delayed effects, novelty effects, learning effects, retention effects and operational changes.

So I wouldn't use a rule like “every A/B test should run for two weeks.”

The right duration comes from the **power calculation plus the behavioral dynamics of the system**.

---

## Q22. How would you determine duration from historical data rather than intuition?

### Answer

I'd start with historical traffic and metric distributions.

For example, I could estimate:

- users per day,
- baseline conversion,
- variance,
- intraday and day-of-week patterns,
- user repeat rates,
- and autocorrelation.

Then I'd simulate an experiment with the planned assignment ratio and estimator.

I could inject effects of different magnitudes—for example 0.5%, 1%, 2%—and estimate detection probability over different durations.

That gives me something much more informative than saying:

> “Two weeks sounds reasonable.”

It lets me say:

> “With our current traffic and variance, 10 days gives approximately 80% power for our target effect, but extending to 14 days improves robustness to weekly seasonality.”

For more complex designs, simulation is often preferable to relying entirely on textbook power formulas.

---

# PART VII — P-VALUES AND CONFIDENCE INTERVALS

## Q23. What exactly is a p-value?

### Answer

A p-value is the probability of obtaining data at least as extreme as what we observed, **assuming the null hypothesis and the test assumptions are true**.

It is not:

> “the probability that the null hypothesis is true.”

That's a very common misunderstanding.

So a p-value of 0.03 means that, under the null and the specified testing procedure, data this extreme would occur with probability about 3%.

It doesn't mean there is a 97% probability that the treatment works.

---

## Q24. What does a 95% confidence interval mean?

### Answer

A confidence interval gives us a range of effect sizes compatible with the data under the specified statistical procedure.

For example:

> “Estimated uplift is 2%, with a 95% confidence interval from 0.5% to 3.5%.”

That's much more useful than reporting only p < 0.05 because it tells me both direction and magnitude.

At Principal level, I'd almost always want to discuss the interval.

The practical question is:

> “What effects remain plausible, and which of those effects would change the business decision?”

---

## Q25. Suppose the p-value is 0.06. Did the experiment fail?

### Answer

Not necessarily.

I'd look at the confidence interval and the decision threshold.

Suppose our estimate is +2%, but the confidence interval is -0.1% to +4.1%.

That's different from an estimate of +0.1% with an interval of -1% to +1.2%.

The first result may still be commercially interesting but underpowered for a precise conclusion.

So I wouldn't frame the outcome as simply:

> significant versus insignificant.

I'd ask:

> “What effect sizes are still plausible, and what decision do we want to make?”

---

# PART VIII — ONE-TAILED VS TWO-TAILED TESTS

## Q26. When would you use a one-tailed test?

### Answer

A one-tailed test evaluates an effect in one pre-specified direction.

For example:

> “Does the treatment increase conversion?”

rather than:

> “Does the treatment change conversion in either direction?”

But I would use a one-sided test only when the opposite direction genuinely isn't part of the decision problem.

It should be chosen **before seeing the results**.

You shouldn't run a two-sided test, see a favorable result, and then switch to one-sided testing to obtain a smaller p-value.

That would be a form of post-hoc hypothesis manipulation.

---

## Q27. Why are two-tailed tests usually the default?

### Answer

Because in most product experiments, both directions matter.

A new feature can improve or hurt the metric.

If I say:

> “We're testing whether this feature changes conversion.”

then both positive and negative effects are scientifically relevant.

A two-sided test therefore protects against ignoring an important negative outcome.

---

# PART IX — Z-TEST VS T-TEST VS CHI-SQUARE

## Q28. When would you use a z-test versus a chi-square test?

### Answer

It depends on what we're testing.

For a simple binary outcome with two groups, I might test the difference in proportions using a two-proportion z-test.

A chi-square test can test whether categorical variables are associated—for example whether treatment assignment and conversion status are independent.

For a 2×2 table, these approaches are closely related and can give equivalent conclusions under common conditions.

So I wouldn't choose between them based purely on “which test is more powerful.”

I'd choose based on the estimand and the data structure, and I'd prefer a method that directly gives me an interpretable effect estimate and uncertainty interval.

---

## Q29. What is the difference between a chi-square test and a z-test conceptually?

### Answer

A z-test is often framed around a standardized difference.

For example:

> “Is the difference between treatment and control proportions large relative to its standard error?”

A chi-square test often frames the question as:

> “Are the observed counts inconsistent with the hypothesis that these categorical variables are independent?”

For a 2×2 table, the two are mathematically connected.

The important thing is that neither test magically establishes causality. The **experimental design** establishes the causal identification; the statistical test quantifies uncertainty under that design.

---

# PART X — SEQUENTIAL TESTING AND EARLY STOPPING

## Q30. Why can't I just monitor the p-value every day and stop when it becomes significant?

### Answer

Because repeated peeking changes the probability of false positives.

If I repeatedly check until I see p < 0.05, the actual Type I error can become substantially larger than 5%.

So the nominal p-value is no longer calibrated for the stopping rule I've actually used.

That means the problem isn't looking at the experiment.

The problem is pretending that a fixed-sample test was used when the sample size was actually chosen based on the results.

There are sequential-testing methods that explicitly allow ongoing monitoring while preserving valid inference. **Always-valid inference**, for example, was developed specifically for this kind of setting.

---

## Q31. Can we safely stop an experiment early?

### Answer

Yes, but the stopping rule has to be part of the design.

I distinguish three situations.

**Safety stopping:**  
We see a severe negative guardrail impact. Stopping is appropriate because this is a business and safety decision.

**Futility stopping:**  
Evidence suggests the experiment is unlikely to reach the desired decision threshold.

**Efficacy stopping:**  
Evidence is sufficiently strong that continuing has little decision value.

For formal statistical inference, I'd use a sequential design or another method whose error control explicitly accounts for repeated looks.

For operational safety, I don't think we should continue harming users just because a classical fixed-horizon p-value hasn't crossed 0.05.

---

## Q32. What is alpha spending?

### Answer

Alpha spending is a framework for allocating the overall Type I error budget across multiple interim looks at the data.

So instead of saying:

> “We'll use α = 0.05 every day,”

we might allocate smaller portions of that 0.05 to each interim analysis according to a predefined schedule.

This lets us monitor the experiment while maintaining overall error control.

---

## Q33. Does early stopping change the MDE?

### Answer

Potentially, yes, depending on the sequential design and stopping boundaries.

A fixed-horizon experiment has one operating characteristic.

Once I introduce interim looks and stopping rules, I have changed the statistical procedure.

So I would not say:

> “We'll use the original sample-size calculation and freely stop whenever we want.”

I'd redesign or simulate the full sequential procedure and calculate its power and expected sample size.

The key point is:

> **Stopping rules are part of the experiment design.**

---

# PART XI — MULTIPLE TESTING

## Q34. Why is multiple testing a problem?

### Answer

Suppose I test 20 independent hypotheses at α = 0.05.

Even if every null hypothesis is actually true, the probability of getting at least one false positive can be much larger than 5%.

So if I test enough metrics, segments and outcomes, eventually something will look significant purely by chance.

This is one reason post-hoc “metric shopping” is dangerous.

---

## Q35. What is the Bonferroni correction?

### Answer

Bonferroni controls the **family-wise error rate**, meaning the probability of making at least one false rejection across a family of tests.

The simplest version replaces:

\[
\alpha
\]

with:

\[
\frac{\alpha}{m}
\]

where \(m\) is the number of hypotheses.

So with 10 tests and α = 0.05, each test would use α = 0.005.

It's simple and conservative.

I would use it when controlling even one false discovery is particularly important, but for large exploratory test sets there may be more powerful approaches, such as controlling the false discovery rate.

---

## Q36. What is the difference between FWER and FDR?

### Answer

**FWER**, or Family-Wise Error Rate, controls the probability of making at least one false positive.

**FDR**, or False Discovery Rate, controls the expected proportion of false discoveries among all discoveries.

So FWER is stricter.

I'd generally care about FWER when even one false claim is unacceptable.

FDR can be more appropriate when we're screening many hypotheses and are willing to tolerate some false discoveries in exchange for greater sensitivity.

---

# PART XII — A/A TESTS AND EXPERIMENT HEALTH

## Q37. What is an A/A test?

### Answer

An A/A test assigns users randomly to two groups but gives both groups the same experience.

The expected treatment effect is therefore zero.

A/A tests are useful for validating:

- randomization,
- metric computation,
- logging,
- statistical inference,
- and the experiment platform.

For example, if my A/A tests repeatedly produce strange imbalances or excessive false positives, I'd investigate the platform before trusting an A/B result.

---

## Q38. What is Sample Ratio Mismatch?

### Answer

**SRM**, or Sample Ratio Mismatch, means the observed assignment proportions differ unexpectedly from the proportions we intended.

Suppose we intended 50/50 randomization but observe 55/45.

That doesn't automatically prove that the treatment caused something.

But it's a major experiment-health signal.

Possible causes include:

- broken randomization,
- filtering,
- eligibility bugs,
- logging problems,
- bot handling,
- exposure problems,
- or users being dropped differently across groups.

I would investigate SRM before trusting treatment-effect results. Experimentation platforms specifically treat SRM as a key data-quality diagnostic.

---

# PART XIII — VARIANCE REDUCTION

## Q39. What is CUPED?

### Answer

**CUPED** stands for Controlled Experiments Using Pre-Experiment Data.

The idea is simple:

If I have a pre-treatment variable that's strongly correlated with the outcome, I can use it to explain some of the natural variation between users.

Then I analyze the treatment effect on the residualized outcome.

The treatment assignment remains randomized, but the variance becomes smaller.

Smaller variance means a smaller standard error, which means we can detect smaller effects with the same number of users.

The important condition is that the covariate must be measured before treatment, or otherwise be safely unaffected by treatment.

---

## Q40. Why does CUPED not create causal bias?

### Answer

Because the key covariate is measured before treatment.

We're not adjusting away something the treatment itself caused.

Instead, we're using pre-treatment information to explain predictable differences in the outcome.

Conceptually:

\[
Y_i^{adjusted}=Y_i-\theta(X_i-\bar X)
\]

where \(X\) is a pre-treatment covariate correlated with \(Y\).

We haven't changed who received treatment.

We're making the outcome less noisy.

---

## Q41. What happens if I adjust for a variable affected by treatment?

### Answer

That's dangerous.

If treatment affects the covariate and I condition on that covariate, I can block part of the treatment effect or introduce bias.

This is the general rule:

> **Be extremely careful about conditioning on post-treatment variables.**

For example, suppose treatment increases engagement and engagement increases purchases.

If I control for engagement, I may accidentally remove part of the treatment pathway.

---

# PART XIV — REGRESSION IN EXPERIMENTS

## Q42. Why would you use regression if the experiment is randomized?

### Answer

Randomization already gives me causal identification.

Regression isn't necessary to make the basic causal claim.

But regression can improve precision and answer more detailed questions.

For example, I can use it to adjust for pre-treatment covariates, estimate treatment effects for segments, model interactions, or incorporate stratification.

The key distinction is:

> Randomization gives me identification; regression can give me efficiency or richer estimation.

---

## Q43. What regression would you use for an A/B test?

### Answer

It depends on the outcome.

For a continuous outcome, a simple linear model can be:

\[
Y_i = \alpha + \tau T_i + \beta X_i + \epsilon_i
\]

where \(\tau\) estimates the treatment effect.

For binary outcomes I might use logistic regression, although in randomized experiments I care about choosing an estimator whose effect interpretation matches the business question.

I also pay attention to standard errors, especially with clustering, repeated observations or non-independent observations.

---

## Q44. What assumptions does linear regression need for an A/B test?

### Answer

People often overstate the assumptions.

If we're estimating an average treatment effect in a randomized experiment, we don't necessarily need perfectly normally distributed outcomes.

The important issues are things like:

- correct treatment assignment,
- independence structure,
- appropriate standard errors,
- finite variance,
- and correct handling of clustering or repeated observations.

With sufficiently large samples, the Central Limit Theorem often provides useful asymptotic justification even for non-normal outcomes.

So I care more about whether my estimator and uncertainty calculation respect the actual experiment design than whether the raw metric looks Gaussian.

---

# PART XV — PANEL REGRESSION AND FIXED EFFECTS

## Q45. What is panel data?

### Answer

Panel data means I observe the same units repeatedly over time.

For example:

| User | Week | Outcome |
|---|---|---|
| A | 1 | 5 |
| A | 2 | 7 |
| B | 1 | 3 |
| B | 2 | 4 |

This lets me separate changes within a unit from differences between units.

---

## Q46. What is a fixed-effect model?

### Answer

A fixed-effect model controls for time-invariant characteristics of each unit.

A simple formulation is:

\[
Y_{it}=\alpha_i+\gamma_t+\beta T_{it}+\epsilon_{it}
\]

where:

- \(\alpha_i\) is a unit fixed effect,
- \(\gamma_t\) is a time fixed effect,
- \(T_{it}\) is treatment,
- \(\beta\) is the treatment effect.

The unit fixed effect absorbs stable characteristics of that user, city, store, etc.

The time fixed effect absorbs common shocks affecting everyone in a given period.

So instead of comparing:

> “User A versus User B”

we're often exploiting variation such as:

> “How did the same user change over time?”

---

## Q47. When would you use fixed effects instead of random effects?

### Answer

Fixed effects are attractive when I believe the unit-specific unobserved characteristics may be correlated with the explanatory variables.

For example, high-value users might systematically behave differently and also be more likely to receive a particular treatment.

Random-effects models make stronger assumptions about the relationship between the unit-specific effect and regressors.

So I wouldn't choose randomly based on a software default.

I'd ask:

> “Is it plausible that the unobserved unit characteristic is correlated with treatment or another predictor?”

If yes, fixed effects are often safer.

---

# PART XVI — DIFFERENCE-IN-DIFFERENCES

## Q48. What is Difference-in-Differences?

### Answer

**Difference-in-Differences**, or DiD, compares the change over time in a treated group against the change over time in a comparison group.

Conceptually:

\[
DiD =
(Y_{treated,after}-Y_{treated,before})
-
(Y_{control,after}-Y_{control,before})
\]

The first difference removes baseline differences.

The second difference removes common time changes.

The remaining difference is attributed to treatment under the required assumptions.

---

## Q49. What is the key assumption behind DiD?

### Answer

The central assumption is **parallel trends**.

It means that in the absence of treatment, the treated and control groups would have followed similar outcome trends over time.

We can't directly observe that counterfactual after treatment.

So we look at pre-treatment trends as supporting evidence.

But I'd be careful saying:

> “The pre-trends are parallel, therefore the assumption is proven.”

They aren't proof.

They're evidence that makes the assumption more plausible.

DiD is fundamentally dependent on this identification assumption.

---

## Q50. How would you validate a DiD design?

### Answer

I'd examine:

- pre-treatment trends,
- placebo treatment dates,
- alternative control groups,
- event-study coefficients,
- sensitivity to different pre-treatment windows,
- and possible simultaneous interventions.

I'd also ask whether anything happened at the same time as treatment that affects the treated units differently.

For example, if the treated region introduced our feature while simultaneously launching a marketing campaign, DiD can't automatically distinguish those effects.

---

# PART XVII — SYNTHETIC CONTROL

## Q51. What is synthetic control?

### Answer

Synthetic control is useful when I have a small number of treated units—sometimes even one—and a reasonable pool of untreated units.

Instead of choosing one control, I create a weighted combination of control units that closely reproduces the treated unit's pre-treatment behavior.

That weighted combination becomes a synthetic counterfactual.

After treatment, I compare the actual treated trajectory against the synthetic trajectory.

The method was developed by Abadie and Gardeazabal and subsequently extended substantially.

---

## Q52. When would you choose synthetic control instead of DiD?

### Answer

I'd consider synthetic control when:

- the number of treated units is very small,
- I have a reasonably large donor pool,
- the treated unit has a distinctive pre-treatment trajectory,
- and constructing a good weighted counterfactual is more credible than assuming a simple treated-versus-control trend.

For example, if one city receives a major policy intervention, I might build a synthetic city from several untreated cities.

The method is fundamentally about constructing a data-driven counterfactual rather than relying on one arbitrary comparison unit.

---

# PART XVIII — QUASI-EXPERIMENTAL DESIGN

## Q53. What is a quasi-experiment?

### Answer

A quasi-experiment is a study where treatment isn't randomly assigned, but some external structure creates something resembling randomization or a credible counterfactual.

Examples include:

- Difference-in-Differences,
- Regression Discontinuity,
- Instrumental Variables,
- Synthetic Control,
- interrupted time series.

The challenge is that causal identification now comes from assumptions about the assignment mechanism rather than literal randomization.

So my first question is always:

> “Why should I believe the comparison group represents the counterfactual?”

---

## Q54. What is Regression Discontinuity?

### Answer

**Regression Discontinuity**, or RD, is used when treatment is assigned based on crossing a threshold.

For example:

> users with a score above 80 receive a benefit; users below 80 do not.

If users just above and just below the cutoff are otherwise sufficiently similar, we can compare them near the threshold.

The causal effect is identified from the discontinuity at that threshold.

The key idea is that we're estimating a **local treatment effect around the cutoff**, not necessarily an effect for the whole population.

RD designs require careful attention to the assignment rule, bandwidth, continuity assumptions and potential manipulation around the threshold.

---

# PART XIX — PROPENSITY SCORES

## Q55. What is a propensity score?

### Answer

A **propensity score** is the probability that a unit receives treatment given observed covariates:

\[
e(X)=P(T=1|X)
\]

For example, we might model the probability that a customer receives a promotion based on their historical behavior.

The goal is to make treated and untreated users comparable with respect to observed covariates.

---

## Q56. What is propensity score matching?

### Answer

Propensity score matching pairs treated users with untreated users who have similar estimated probabilities of receiving treatment.

The idea is:

> “Find controls that look like the treated users based on the observed characteristics that influenced treatment.”

This can reduce confounding from measured variables.

But it does not solve unmeasured confounding.

That is a crucial limitation.

---

## Q57. What are the main assumptions for propensity-score methods?

### Answer

I'd think about three major requirements.

First, **conditional exchangeability**: after conditioning on the observed covariates, treatment assignment behaves as though it is independent of the potential outcomes.

Second, **positivity**: for relevant covariate patterns, there's a non-zero probability of receiving either treatment.

Third, **consistency**: the treatment needs to be well-defined.

The first assumption is especially difficult because we can't directly verify that we've measured every relevant confounder.

So I would never describe propensity-score matching as “making observational data equivalent to randomized data.”

It can reduce observed imbalance, but the causal claim still depends on assumptions.

---

## Q58. Is propensity-score matching always the best way to use propensity scores?

### Answer

No.

Propensity scores can be used for:

- matching,
- inverse-probability weighting,
- stratification,
- covariate adjustment,
- and doubly robust estimators.

Also, matching isn't automatically superior simply because it's intuitive.

I'd choose the method based on the estimand, overlap, sample size, treatment assignment process and how well we can model the propensity score.

---

# PART XX — DOUBLY ROBUST ESTIMATION

## Q59. What is a doubly robust estimator?

### Answer

A doubly robust estimator combines two models:

1. a treatment-assignment model, often the propensity score;
2. an outcome model.

The appealing property is that under standard conditions, the estimator can remain consistent if **either one of those two models is correctly specified**, rather than requiring both to be correct.

That's why it can be useful in observational causal inference.

But “doubly robust” does **not** mean “immune to bad data or bad assumptions.”

We still need appropriate identification assumptions, overlap and sensible model specification.

---

# PART XXI — CONFOUNDING AND DAGs

## Q60. How do you distinguish correlation from causation?

### Answer

I start by asking what mechanism could create the observed relationship.

Then I think about possible confounders.

A **confounder** is a variable that affects both treatment and outcome and therefore creates a misleading association.

Directed Acyclic Graphs, or **DAGs**, are useful here because they let us explicitly reason about which variables we should adjust for.

The important lesson is:

> We don't control for every variable we have.

We control for variables needed to block problematic backdoor paths, while avoiding variables that are descendants of treatment or colliders.

---

## Q61. What is a collider?

### Answer

A collider is a variable that is influenced by two other variables.

For example:

\[
Treatment \rightarrow C \leftarrow Outcome
\]

If I condition on C, I can create a spurious association between treatment and outcome.

So a common causal-inference mistake is:

> “Let's control for everything.”

More adjustment isn't automatically better.

The adjustment set should be determined by the causal structure.

---

# PART XXII — NETWORK EFFECTS

## Q62. How would you run an experiment on a social network or marketplace?

### Answer

First I'd ask whether SUTVA is plausible.

If users interact, then individual-level randomization may create spillovers.

For example, treatment users may invite control users, or treated sellers may change the prices or inventory seen by controls.

I'd consider cluster-based randomization or another design that explicitly models interference.

I'd also define whether I care about:

- direct effects,
- indirect spillover effects,
- or total ecosystem effects.

That's important because the answer to “does treatment work?” can be different depending on which estimand we're targeting.

---

# PART XXIII — SWITCHBACK EXPERIMENTS

## Q63. What is a switchback experiment?

### Answer

A switchback experiment is an experiment where the treatment alternates over time rather than assigning users permanently to treatment or control.

For example:

- 9–10 AM: treatment
- 10–11 AM: control
- 11 AM–12 PM: treatment
- 12–1 PM: control

It's useful when treatment affects the environment itself.

Ride-hailing is a classic example. Changing an algorithm can affect supply-demand dynamics, so individual user randomization may not isolate the treatment.

Instead, we randomize treatment at the time-window level.

---

## Q64. What makes switchback experiments statistically difficult?

### Answer

The observations aren't necessarily independent.

Consecutive time periods can be correlated.

There can also be **carryover effects**.

For example, if treatment changes driver positioning at 10:00, switching to control at 10:01 doesn't necessarily reset the system immediately.

I'd therefore think carefully about:

- length of each switchback period,
- washout periods,
- time-of-day effects,
- day-of-week effects,
- autocorrelation,
- number of independent switchback blocks,
- and appropriate standard errors.

A recent study of switchback designs highlights precisely these issues: clustering, temporal dependence, carryover and spillovers can materially affect inference.

---

## Q65. How would you choose the switchback duration?

### Answer

I'd look at system dynamics.

I'd estimate how quickly the system responds when treatment changes.

If the effect stabilizes within five minutes, a 30-minute block might be reasonable.

If the system has a two-hour memory, a five-minute block is not independently switching the system.

I'd also avoid making blocks so long that time-of-day effects dominate the treatment contrast.

In practice I'd use historical data and experimentation or simulation to estimate the correlation and carryover structure, then choose a block size that balances independence, operational stability and statistical power.

---

# PART XXIV — RATIO METRICS AND NON-NORMAL DATA

## Q66. Why are ratio metrics tricky?

### Answer

Metrics such as:

\[
\frac{\text{total revenue}}{\text{number of users}}
\]

look simple, but the numerator and denominator are correlated random variables.

The variance of a ratio isn't simply the variance of the numerator divided by the variance of the denominator.

Also, user-level distributions are often very skewed.

A small number of users may account for a huge fraction of revenue.

So I'd be careful about applying a simple t-test to raw ratios without understanding how the metric is constructed.

Possible approaches include delta-method approximations, bootstrap methods, user-level estimators, or experiment-specific variance estimators.

---

## Q67. Does a non-normal metric invalidate a t-test?

### Answer

Not automatically.

With sufficiently large independent samples and appropriate variance behavior, the Central Limit Theorem can make the sample mean approximately normal even if individual observations aren't.

The bigger concern is usually:

- dependence,
- extreme heavy tails,
- inappropriate aggregation,
- incorrect standard errors,
- or insufficient effective sample size.

So I wouldn't automatically switch statistical tests just because a histogram isn't Gaussian.

---

# PART XXV — HETEROGENEOUS TREATMENT EFFECTS

## Q68. What does treatment-effect heterogeneity mean?

### Answer

It means the treatment doesn't have the same causal effect for everyone.

For example:

- +5% for new users,
- +1% for existing users,
- -2% for power users.

The average treatment effect might still be +1%.

So an aggregate analysis could hide meaningful differences.

The challenge is distinguishing **real heterogeneity** from random subgroup noise.

That means subgroup discovery needs its own statistical discipline.

---

## Q69. Why is post-hoc segmentation dangerous?

### Answer

Because every additional subgroup is another opportunity to find a random significant effect.

Suppose I examine:

- gender,
- age,
- geography,
- device,
- tenure,
- activity level,
- dozens of countries,
- hundreds of behavioral segments.

Something will probably look interesting by chance.

So I distinguish between:

> pre-specified hypothesis-driven heterogeneity

and

> exploratory hypothesis generation.

For exploratory findings, I'd treat them as hypotheses to validate in another experiment rather than automatically shipping based on them.

---

# PART XXVI — MULTIPLE EXPERIMENTS

## Q70. What happens if multiple experiments run simultaneously?

### Answer

The main issue is interaction.

If experiment A changes the same user experience or causal pathway as experiment B, the measured effect of A may depend on whether the user is also exposed to B.

That's an interaction effect.

The simplest solution is experiment exclusion.

Another is factorial experimentation, where we intentionally randomize combinations.

At scale, however, I would also maintain exposure logging and experiment metadata so we can detect overlapping treatments and estimate interactions where necessary.

Microsoft's experimentation guidance explicitly identifies experiment interaction prevention and detection as important components of trustworthy experimentation.

---

# PART XXVII — NOVELTY, LEARNING AND CARRYOVER

## Q71. Why can an experiment look strong initially and then fade?

### Answer

Several mechanisms can cause that.

There can be **novelty effects**: users respond simply because something changed.

There can be learning effects: users initially struggle with a new interface but adapt over time.

There can be selection effects: the people who remain exposed aren't representative of the initial population.

There can also be operational adaptation—for example, sellers, drivers or merchants changing their behavior in response.

So I'd look at treatment effects over time rather than only reporting one aggregate number.

---

# PART XXVIII — EARLY RESULTS AND P-HACKING

## Q72. What is p-hacking?

### Answer

P-hacking is manipulating the analysis process to make statistically significant results more likely.

Examples include:

- repeatedly checking until significance appears,
- changing the primary metric after seeing results,
- dropping inconvenient observations,
- trying many model specifications and reporting only one,
- selectively choosing subgroups,
- or changing the hypothesis after seeing the data.

The defense is primarily **pre-specification, transparency and disciplined experiment design**.

---

## Q73. What is HARKing?

### Answer

**HARKing** means Hypothesizing After the Results are Known.

For example:

> “We observed that the feature works particularly well for new users, and that's exactly what we expected.”

If that subgroup wasn't part of the original hypothesis, it shouldn't be presented as though it were.

Exploration is completely legitimate.

The problem is confusing exploration with confirmation.

---

# PART XXIX — BAYESIAN EXPERIMENT ANALYSIS

## Q74. How would you explain Bayesian A/B testing?

### Answer

In a Bayesian analysis, I start with a **prior distribution**, which expresses what effect sizes were plausible before seeing the current experiment.

Then I combine that with the observed data to get a **posterior distribution**.

The posterior represents our updated uncertainty about the treatment effect after seeing the experiment.

So instead of asking:

> “Would data like this be unusual if the effect were zero?”

I can ask directly:

> “Given the data and my assumptions, what is the probability that the treatment effect is positive?”

That's often a very natural way to communicate with decision-makers.

---

## Q75. What is a credible interval?

### Answer

A Bayesian credible interval is an interval containing a specified posterior probability of the parameter.

For example:

> “There's a 95% posterior probability that the treatment effect lies between -0.4% and +2.5%.”

That's a direct probability statement about the parameter **within the Bayesian model and prior specification**.

That's different from the frequentist interpretation of a confidence interval.

---

## Q76. Would you prefer Bayesian or frequentist testing?

### Answer

I wouldn't make it ideological.

Both can be very useful.

Frequentist methods are well understood, have mature error-control frameworks, and are natural for standardized experimentation platforms.

Bayesian methods can be particularly useful when:

- we have meaningful prior information,
- experiments are naturally hierarchical,
- decisions depend on probability of exceeding a business threshold,
- or we want to model uncertainty continuously.

For example, instead of merely asking whether an uplift is statistically significant, I may want:

> “What's the posterior probability that uplift exceeds the minimum economically worthwhile improvement?”

That's a very decision-oriented question.

---

# PART XXX — MCMC

## Q77. What is MCMC?

### Answer

**MCMC**, or Markov Chain Monte Carlo, is a family of methods for drawing samples from a probability distribution that is difficult to compute directly.

In Bayesian inference, that distribution is often the posterior:

\[
p(\theta|data)
\]

Once I have posterior samples, I can estimate quantities such as:

- posterior means,
- credible intervals,
- probability of positive uplift,
- probability of exceeding a business threshold.

Modern Bayesian tools such as Stan commonly use **Hamiltonian Monte Carlo** and its adaptive NUTS variant for efficient posterior sampling.

---

## Q78. Do you actually need MCMC for Bayesian A/B testing?

### Answer

Not always.

For simple conjugate models, we can sometimes derive the posterior analytically.

For example, a binomial conversion model with an appropriate Beta prior has a Beta posterior.

MCMC becomes valuable when the model is more complicated—for example:

- hierarchical experiments,
- correlated effects,
- time-varying effects,
- random effects,
- latent variables,
- or nonlinear models.

So I'd use the simplest inference method that adequately represents the problem.

---

# PART XXXI — BAYESIAN HIERARCHICAL EXPERIMENTS

## Q79. Why would you use a hierarchical Bayesian model for experiments?

### Answer

Suppose I'm running experiments across 50 countries.

I could estimate each country independently, but small countries will have very noisy estimates.

A hierarchical model allows country-level effects to be related through a common population distribution.

This creates **partial pooling**.

Partial pooling means small groups are allowed to learn from the broader population instead of producing extremely noisy standalone estimates.

It's particularly useful when treatment effects are expected to vary across regions, products or user segments but are still related.

---

# PART XXXII — BOOTSTRAP AND RANDOMIZATION INFERENCE

## Q80. When would you use bootstrap?

### Answer

Bootstrap is useful when I want an empirical approximation to the sampling distribution of an estimator and the analytic variance is difficult to derive.

It's particularly useful for complicated metrics such as:

- ratios,
- quantiles,
- nonlinear statistics,
- or complex estimators.

But I need to bootstrap at the correct unit.

If I randomized by city, bootstrapping individual users as though they're independent can be invalid.

The bootstrap has to respect the dependence structure of the data.

---

## Q81. What is randomization inference?

### Answer

Randomization inference uses the actual randomization mechanism of the experiment to construct the distribution we'd expect under a sharp null hypothesis.

Conceptually:

> “What treatment differences would I see if I repeatedly reassigned treatment according to the same randomization rule but there were actually no treatment effect?”

It's especially attractive in experiments because it relies directly on the assignment mechanism rather than a large-sample approximation.

---

# PART XXXIII — CLUSTERING AND STANDARD ERRORS

## Q82. Why does clustering matter?

### Answer

If observations within a cluster are correlated, they don't provide as much independent information as the raw observation count suggests.

For example, 10,000 users from 10 cities aren't equivalent to 10,000 independent cities.

Ignoring clustering can make standard errors too small and confidence intervals too narrow.

So if randomization happened at the city level, I'd generally perform inference at a level consistent with that assignment and dependence structure.

---

# PART XXXIV — CORRELATION AND CAUSAL ISOLATION

## Q83. Suppose conversion increased by 10% after the feature launched. How do you know the feature caused it?

### Answer

I wouldn't conclude that from the time series alone.

I'd ask:

- What happened to the control group?
- Was the treatment randomized?
- Did anything else launch simultaneously?
- Is there seasonality?
- Did traffic mix change?
- Did marketing change?
- Did competitors change?
- Did measurement or logging change?

The strongest answer would be a randomized experiment where treatment and control were exposed to the same external environment.

In a non-randomized setting, I'd use a credible quasi-experimental design and explicitly state the identification assumptions.

---

# PART XXXV — TIME SERIES AND INTERRUPTED TIME SERIES

## Q84. What is an interrupted time-series design?

### Answer

An interrupted time series compares the outcome trajectory before and after an intervention.

For example, we might model:

- baseline level,
- baseline trend,
- immediate level change,
- post-intervention trend change.

It can be useful when there is no untreated control.

But it's vulnerable to other events happening around the intervention date.

So I'd be cautious about making a causal claim unless I have strong evidence that the intervention was the primary explanation for the change.

---

# PART XXXVI — A MORE ADVANCED DESIGN QUESTION

## Q85. A team says, “We don't have enough traffic for an A/B test. Let's just compare this month to last month.” What do you say?

### Answer

I'd first ask what decision we're trying to make.

A before-and-after comparison isn't inherently useless, but it doesn't give us the same causal protection as randomized treatment and control.

Traffic composition, seasonality, product changes and external events can all move the metric.

If individual randomization isn't feasible, I'd investigate alternatives:

- cluster randomization,
- switchbacks,
- DiD,
- synthetic control,
- interrupted time series,
- or possibly a smaller but better-powered experiment using variance reduction.

I'd rather design a smaller experiment with credible identification than run a giant observational comparison and give ourselves false confidence.

---

# PART XXXVII — PRINCIPAL-LEVEL SCENARIOS

## Q86. Your experiment shows +2% conversion with p < 0.001. Would you ship?

### Answer

Not from that information alone.

I'd ask:

1. Is the experiment healthy?
2. Is there SRM?
3. Is the effect on the primary metric?
4. What is the confidence interval?
5. Is +2% practically meaningful?
6. Did guardrails regress?
7. Is the effect stable over time?
8. Are there meaningful negative subgroups?
9. Could another experiment be interacting with it?
10. Did we use the intended stopping and analysis procedure?

Statistical significance tells me there is evidence against the null under the specified analysis.

It doesn't tell me whether the product change is good.

---

## Q87. The experiment is not statistically significant, but the confidence interval is entirely above zero according to a Bayesian analysis. What do you do?

### Answer

First I'd make sure we're comparing like with like.

A Bayesian credible interval and frequentist confidence interval are not interchangeable.

Then I'd understand the Bayesian model and prior.

More importantly, I'd translate the result into the business decision.

For example:

> “The posterior probability of any positive effect is 94%, but the probability that the effect exceeds our economically meaningful threshold is only 45%.”

That may support waiting, gathering more evidence or launching cautiously rather than declaring victory.

---

## Q88. Your treatment effect is +5% for new users and -1% for existing users. What do you do?

### Answer

I'd first verify that the heterogeneity is real.

I'd check:

- sample size,
- confidence or credible intervals by subgroup,
- whether the subgroup was pre-specified,
- interaction terms,
- consistency across related outcomes,
- and whether there is a plausible mechanism.

Then I'd quantify the business impact.

The right decision might be:

> launch only for new users,

rather than:

> ship globally.

At Principal level, I would emphasize that a heterogeneous treatment effect often turns an experiment from a binary “ship / don't ship” decision into a targeting problem.

---

## Q89. Your experiment shows a large improvement for one country but not globally. What would you investigate?

### Answer

I'd separate three possibilities.

First, **real heterogeneity**: the product actually works differently in that market.

Second, **statistical noise**: the subgroup result is an extreme observation from many comparisons.

Third, **implementation differences**: the feature or user behavior differs by market.

I'd check interaction effects, confidence intervals, exposure rates, baseline differences and operational details.

Then I'd decide whether the finding should be treated as a validated localized effect or a hypothesis for another targeted experiment.

---

# PART XXXVIII — EXPERIMENT FAILURE DIAGNOSIS

## Q90. You discover SRM halfway through the experiment. What do you do?

### Answer

I wouldn't simply ignore the first half and continue.

First I'd determine when the mismatch started and identify its root cause.

I'd investigate:

- assignment logic,
- eligibility,
- logging,
- filtering,
- exposure,
- client versions,
- bots,
- missing data,
- and experiment configuration.

Then I'd assess whether the data collected before the issue can still support a valid analysis.

If the assignment mechanism was compromised, I may need to invalidate that period.

The key is not:

> “How do I get a p-value from this dataset?”

It's:

> “Can I still make a credible causal claim from this experiment?”

---

# PART XXXIX — EXPERIMENTS WITH LOW BASE RATES

## Q91. What happens when the outcome is extremely rare?

### Answer

Rare outcomes increase variance and often require large sample sizes.

For example, detecting a meaningful difference in a 0.1% conversion event can require enormous traffic.

I would consider whether we can use a more sensitive proximal metric while retaining the rare outcome as a guardrail.

I'd also verify that the metric is measured reliably.

Sometimes the answer isn't simply “collect more users.” A better estimator or variance-reduction technique can be much more efficient.

---

# PART XL — METRIC DESIGN

## Q92. How would you choose a good experimentation metric?

### Answer

I'd look for five things.

First, **causal relevance**: does the metric measure something the treatment is supposed to change?

Second, **sensitivity**: can we detect meaningful changes without enormous samples?

Third, **reliability**: is the metric logged consistently?

Fourth, **interpretability**: can product and business stakeholders understand what movement means?

Fifth, **resistance to gaming or unintended optimization**.

I generally want one clear primary metric, supported by diagnostic and guardrail metrics.

---

# PART XLI — TRIGGERED ANALYSIS

## Q93. What is triggered analysis?

### Answer

Triggered analysis means analyzing only users who actually became eligible for or exposed to the relevant treatment experience.

It can sometimes improve sensitivity because we're focusing on the population where treatment could actually have an effect.

But we need to be very careful.

If treatment itself affects whether someone enters the triggered population, conditioning on that trigger can introduce selection bias.

So the trigger needs to be defined independently of treatment, or the analysis needs to account for the resulting causal structure.

---

# PART XLII — PRINCIPAL-LEVEL EXPERIMENT STRATEGY

## Q94. What makes a good experimentation program at company scale?

### Answer

I would think about it as a system, not a collection of individual A/B tests.

The platform should provide:

- reliable randomization,
- exposure logging,
- SRM detection,
- standardized metrics,
- experiment registration,
- pre-analysis planning,
- sequential monitoring,
- guardrails,
- interaction detection,
- variance reduction,
- power calculation,
- and reproducible analysis.

But the organizational side matters just as much.

Teams need a culture where:

> “This result is not significant”

is an acceptable outcome.

Otherwise people start optimizing the analysis process instead of learning from experiments.

The goal of an experimentation organization isn't to produce statistically significant results.

It's to produce **reliable decisions**.

---

# PART XLIII — “TRICK” QUESTIONS

## Q95. Does a statistically significant result mean the effect is important?

### Answer

No.

With enough users, a tiny effect can become statistically significant.

That's why I always separate:

> statistical significance

from

> practical or business significance.

The confidence interval and decision threshold are often more informative than the p-value alone.

---

## Q96. Does a non-significant result prove there is no effect?

### Answer

No.

It means we don't have sufficient evidence to reject the null under the specified test.

An underpowered experiment can easily produce a non-significant result even when a meaningful effect exists.

That's why I look at the confidence interval and the MDE.

If the interval is narrow around zero, that's strong evidence that any effect is probably small.

If the interval is huge, we've learned much less.

---

## Q97. Can randomization guarantee perfectly balanced treatment and control groups?

### Answer

No.

Randomization guarantees balance **in expectation**, not exact equality in every sample.

By chance, treatment could have slightly older users, higher-value users, or more users on iOS.

What matters is whether the observed imbalance is compatible with random assignment and whether it materially affects precision or validity.

And that's another reason not to overreact to every small baseline difference.

---

## Q98. Should you test whether treatment and control have significantly different baselines before analyzing the experiment?

### Answer

I wouldn't use a significance test as a gatekeeper.

By design, randomization makes baseline characteristics comparable in expectation.

With a sufficiently large sample, tiny irrelevant differences can become statistically significant.

I'd instead inspect whether there are **material baseline imbalances** or experiment-health problems.

If randomization is working correctly, I don't want to start selectively adjusting the analysis based on whichever baseline variables happened to have significant differences.

---

# PART XLIV — THE PRINCIPAL-LEVEL “BIG PICTURE” QUESTION

## Q99. What is the biggest mistake companies make with experimentation?

### Answer

I think the biggest mistake is treating experimentation as a statistical testing problem rather than a **decision system**.

People often jump directly to:

> “What's the p-value?”

But the more important questions happen earlier.

Did we define the right causal question?

Did we choose the right randomization unit?

Is interference plausible?

Do we have enough power?

Is the metric actually measuring value?

Are there guardrails?

Did we run the experiment long enough to capture the relevant behavior?

Could the result be caused by an instrumentation problem?

And finally:

> “What decision would we make under each plausible effect size?”

If we answer those questions well, the statistical test becomes one component of a much larger causal decision framework.

---

# PART XLV — HIGH-VALUE QUESTIONS TO EXPECT AT PRINCIPAL LEVEL

These are the questions I would expect to be especially valuable for an L6 interview because they test judgment rather than memorization.

## Q100. “Design an experiment to test dynamic pricing in a marketplace.”

### Answer

I'd immediately ask whether individual-level randomization is valid.

Dynamic pricing changes demand and supply behavior, so there may be substantial interference.

I'd determine whether treatment changes the market equilibrium.

If it does, I might need:

- geographic or marketplace-level randomization,
- switchback experimentation,
- or another design that treats the market or time period as the experimental unit.

Then I'd define the OEC carefully.

Optimizing revenue alone could harm customer conversion, driver earnings, seller participation or long-term retention.

I'd therefore define the primary objective and several guardrails before calculating power.

And because this is an equilibrium system, I'd be particularly interested in treatment effects after the system has had time to adapt.

---

## Q101. “We need an answer in three days, but a properly powered experiment needs three weeks. What do you do?”

### Answer

I wouldn't manufacture certainty just because the deadline is three days.

I'd separate the decision into:

> “What can we safely learn in three days?”

and

> “What requires three weeks of evidence?”

If there is a safety risk, we might use a small ramp with conservative guardrails.

If the expected downside is low, perhaps we can make an interim operational decision while continuing the experiment.

But I'd clearly label the evidence as preliminary.

The important principle is:

> **A business deadline doesn't change the statistical uncertainty.**

We can change the decision threshold or accept more uncertainty, but we shouldn't pretend the uncertainty disappeared.

---

## Q102. “The VP wants us to stop because the experiment already looks positive.”

### Answer

I'd ask what “looks positive” means.

If they're seeing a positive point estimate, that's not enough.

I'd look at the pre-specified stopping rule.

If we need a decision immediately, I'd consider a valid sequential analysis or Bayesian decision framework.

I'd also distinguish between:

> “We need to stop because the treatment is clearly beneficial.”

and

> “We need to stop because management wants an answer.”

Those are very different reasons.

I want the stopping rule to be driven by the decision problem and statistical design, not whichever point in time produces the most convenient result.

---

## Q103. “How do you know when you've run an experiment long enough?”

### Answer

I'd say there are three dimensions.

**Statistical sufficiency:**  
Do we have enough independent information to estimate the target effect with the desired precision?

**Temporal coverage:**  
Have we covered the relevant weekly, seasonal and operational patterns?

**Behavioral stabilization:**  
Have novelty, learning or system adaptation settled enough that we're measuring the effect we actually care about?

So “enough users” and “enough time” aren't always the same thing.

---

## Q104. “What is the difference between a good Data Scientist and a Principal Data Scientist in experimentation?”

### Answer

A good Data Scientist can correctly execute an experiment.

A Principal Data Scientist should be able to ask whether the experiment **should exist in that form in the first place**.

I'd expect a Principal to challenge:

- the causal question,
- the estimand,
- the randomization unit,
- the metric,
- the MDE,
- the duration,
- the interference assumptions,
- the analysis plan,
- and the decision rule.

And after the result, I'd expect them to connect the statistical evidence to product and business strategy.

So the progression is from:

> “Can I analyze this experiment?”

to:

> “Is this the right experiment, and what decision should the organization make from it?”

---

# PART XLVI — TOPIC CHECKLIST FOR FURTHER PREPARATION

The bank above covers the core foundation, but for a truly exhaustive Principal-level preparation set, I would continue into these areas:

### Experimental design

- factorial experiments
- multi-arm experiments
- cluster randomized trials
- stratified/block randomization
- covariate-adaptive randomization
- unequal allocation
- ramp-up strategies
- holdouts
- long-term holdouts
- persistent versus non-persistent assignment
- experiment interaction

### Experiment diagnostics

- Sample Ratio Mismatch
- A/A testing
- logging validation
- exposure validation
- contamination
- attrition
- treatment compliance
- instrumentation breaks
- bot filtering
- missing-not-at-random problems
- novelty effects
- carryover

### Statistical inference

- t-tests
- z-tests
- chi-square
- Fisher's exact test
- ANOVA
- Welch's test
- bootstrap
- permutation tests
- randomization inference
- cluster-robust standard errors
- heteroskedasticity
- autocorrelation
- delta method
- ratio metrics
- heavy-tailed metrics

### Causal inference

- potential outcomes
- SUTVA
- exchangeability
- positivity
- consistency
- DAGs
- backdoor criterion
- colliders
- confounding
- mediation
- instrumental variables
- local average treatment effect
- propensity scores
- matching
- inverse probability weighting
- doubly robust estimation
- DiD
- event studies
- synthetic control
- synthetic DiD
- regression discontinuity
- interrupted time series

### Experimentation with interference

- network effects
- marketplace interference
- geographic experiments
- cluster randomization
- switchbacks
- carryover
- spillover effects
- direct versus indirect treatment effects
- equilibrium effects

### Sequential experimentation

- optional stopping
- sequential probability ratio tests
- alpha spending
- group sequential designs
- always-valid inference
- confidence sequences
- early efficacy stopping
- futility stopping
- safety stopping

Sequential monitoring needs to be designed explicitly rather than treating repeated peeking as an ordinary fixed-sample test.

### Bayesian experimentation

- priors
- likelihood
- posterior
- conjugate models
- credible intervals
- posterior probability of uplift
- probability of exceeding an MDE
- expected loss
- decision thresholds
- Bayesian sequential decisions
- hierarchical models
- partial pooling
- MCMC
- HMC
- NUTS
- posterior predictive checks
- sensitivity to prior choice

### Modern experimentation

- CUPED
- CUPAC
- machine-learning variance reduction
- heterogeneous treatment effects
- causal forests
- meta-learners
- uplift modeling
- treatment-policy learning
- experimentation under interference
- adaptive experiments

### Principal-level judgment

- defining the estimand
- determining whether experimentation is feasible
- selecting the randomization unit
- deciding between an RCT and quasi-experiment
- choosing the MDE from economics rather than convenience
- deriving duration from historical behavior
- deciding whether to stop early
- deciding what evidence is sufficient to ship
- balancing statistical and practical significance
- separating exploration from confirmation
- communicating uncertainty to executives
- designing an experimentation platform and governance model

---

# PRINCIPAL-LEVEL MENTAL MODEL

When faced with almost any experimentation question, I would mentally walk through this sequence:

**1. What is the causal question?**

What exactly are we trying to estimate?

↓

**2. What is the estimand?**

ATE? ATT? CATE? Short-term? Long-term? Direct effect? Total effect?

↓

**3. What is the experimental unit?**

User? Session? Seller? Driver? City? Time block?

↓

**4. Can we randomize?**

If yes, prefer a randomized design unless there is a strong reason not to.

↓

**5. Does SUTVA hold?**

Are there interference, spillovers or multiple versions of treatment?

↓

**6. What could invalidate the experiment?**

SRM, contamination, noncompliance, attrition, instrumentation, concurrent experiments, carryover?

↓

**7. What is the primary outcome?**

And what are the guardrails?

↓

**8. What is the smallest decision-relevant effect?**

That's where MDE should come from.

↓

**9. How much information do we need?**

Sample size, power, variance, clustering and expected traffic.

↓

**10. How long should it run?**

Enough independent information + enough temporal coverage + enough time for behavior to stabilize.

↓

**11. What is the analysis plan?**

Estimator, standard errors, covariate adjustment, multiple testing, sequential procedure.

↓

**12. What decision rule are we using?**

Ship? Don't ship? Continue? Roll back? Target a subgroup?

↓

**13. What would make us distrust the result?**

This is the Principal-level question people often forget.

The strongest experimentation practitioners don't merely calculate significance. They continually ask:

> **“What has to be true for this result to be a credible causal statement?”**

---

## Core references used to ground this framework

The structure above follows established experimentation and causal-inference frameworks, including Microsoft's online experimentation material, which explicitly emphasizes power, A/A tests, SRM, carry-over, experiment interactions, OECs, guardrails and heterogeneity.

For causal inference, the assumptions and designs above are consistent with standard potential-outcomes treatments of SUTVA, exchangeability, interference, DiD, regression discontinuity and synthetic control.

For Bayesian computation, MCMC and modern Hamiltonian Monte Carlo approaches such as NUTS are documented in the Stan reference materials.