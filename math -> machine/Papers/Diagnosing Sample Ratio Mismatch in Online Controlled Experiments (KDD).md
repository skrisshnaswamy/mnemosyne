---
title: "Diagnosing Sample Ratio Mismatch in Online Controlled Experiments (KDD)"
authors: ["Fabijan et al."]
year: 2019
url: https://exp-platform.com/Documents/2019_KDDFabijanGupchupFuptaOmhoverVermeerDmitriev.pdf
priority: Good-To-Read
read_on: 2026-09-25
tags: [paper]
---
## The Core Idea

A **sample ratio mismatch (SRM)** is when you configured a 50/50 split but the analysis shows something like 50.2/49.8. That sounds trivial. It is not. It means the set of users who made it into your analysis was *selected* by something other than pure chance — and once selection is not random, the comparison between the two groups no longer measures the treatment. The whole causal claim collapses.

> [!NOTE] Sample Ratio Mismatch ^srm-def
> The observed ratio of users across experiment variants differs from the configured ratio by more than chance can explain. It is a *symptom*, not a disease: it flags that some process downstream of randomisation treated the two groups differently.

The insight the paper adds is that SRM is the single cheapest, highest-yield alarm you can put on an experiment platform, and that the hard part is never detecting it — a chi-square test does that in one line — but finding *why*. Before this paper, the literature had scattered single causes: LinkedIn wrote about triggered-analysis SRMs, Yahoo about lost telemetry. Nobody had mapped the space.

So the contribution is a **taxonomy: 25 distinct root causes, bucketed by which of the five stages of an experiment they live in** (assignment, execution, log processing, analysis, interference), plus ten rules of thumb that let you bisect towards the cause instead of guessing.

Why this unlocks something: SRM investigations were taking analysts "minutes to months", during which the product team is frozen — they cannot ship or kill the feature. A taxonomy turns an open-ended hunt into a decision tree.

The motivating story, which is the whole paper in miniature. MSN raised the number of rotating cards in a carousel from 12 to 16. Engagement with the cards went *down*, significantly. Ship decision: no. But there was an SRM — fewer users in treatment than configured. Root cause: the most engaged treatment users clicked so much that they tripped the **bot-detection threshold** and were deleted from the analysis. The bot filter ate exactly the people the feature worked on. After fixing it, the feature was a clear win and shipped.

That is the shape of every story here: a post-treatment process silently removed a non-random slice of one arm.

## The Methodology

This is a case study paper, not a modelling paper. Four companies — Microsoft, Booking.com, Outreach.io, Online Dialogue — over 25 products. Four data sources: literature review, internal documentation, **14 semi-structured interviews** with analysts and engineers, and quantitative analysis of over **10,000 historical experiments**.

**The detection test.** Under the null hypothesis that assignment was correct, the counts follow the configured proportions. Use Pearson's chi-square:

$$\chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}$$

where $O_i$ is observed users in variant $i$ and $E_i$ is expected. Run it on the paper's example — 821,588 vs 815,482 users, expected 50/50. Total is 1,637,070, so $E = 818{,}535$ each, and the deviation is 3,053 either way:

$$\chi^2 = 2 \times \frac{3053^2}{818535} = 22.77$$

With 1 degree of freedom that gives $p \approx 1.8 \times 10^{-6}$ — about 1 in 550,000. A 0.2 percentage point wobble is overwhelming evidence at that sample size. This is why eyeballing the split does not work and the test is mandatory.

**The five stages, which are the top level of the taxonomy.** Every experiment goes: (1) *assignment* — split users by a hash of some ID; (2) *execution* — the client actually receives and applies the variant and logs usage; (3) *log processing* — telemetry is uploaded, joined, filtered, aggregated; (4) *analysis* — filtering to triggered users and computing metrics. Plus (5) *interference*, which cuts across everything. An SRM can be born in any of them.

---

**1 · Assignment SRMs.** An A/A test on MSN.com had an SRM. An A/A test cannot have an SRM unless the splitter is broken — and it was. The service hashes users into 1,000 buckets of 0.1% each; a bug gave control one bucket fewer, so "50/50" was actually **49.9/50**.

Three requirements on an assignment service, and violating any one gives SRM:
- Each user equally likely to see each variant.
- Assignment is *sticky* — same user, same variant, every visit. Unstable user IDs break this.
- Assignments across concurrent experiments are independent.

The third is the subtle one. If you pick the hash seed at random from a pool, the **birthday paradox** bites: with 365 seeds you only need about **23 experiments** for a 50% chance that two of them share a seed and are therefore perfectly correlated. Fix: a very large seed pool, and rotate the pool over time so user-behaviour correlations do not accumulate in fixed buckets.

Sometimes you want the *opposite* of independence. If treatment-in-A plus treatment-in-B crashes the app, independent randomisation guarantees some users get both, lose their telemetry, and produce an SRM. Those two experiments must be made *mutually exclusive* — separate populations or separate time windows.

**2 · Execution SRMs.** Skype tested an ML model that adjusts audio buffering per network context against a fixed parameter. Randomisation unit was the call, not the user. Result: worse distortion and delay, and **30% fewer treatment sessions collected**. Root cause: an asynchronous refresh of the experiment config mid-call. By design the new config was not applied mid-session, but a bug updated the in-memory variant ID anyway, so ~30% of sessions were logged as if they had never been in treatment.

The general family here — and this is the part worth internalising — is that **the treatment itself changes the probability that a log arrives**:
- Treatment slows the page → users bail before the log fires → fewer treatment users.
- Treatment speeds the page up → *more* logs arrive → SRM that is actually good news.
- Treatment adds new telemetry → one-time users now emit at least one event → more users counted in that arm.
- Treatment makes users click more → more chances for at least one event to survive a lossy channel → more users counted.
- Treatment crashes → logs never generated.
- Only one variant does a redirect → some redirects fail → that arm loses users.

Telemetry loss is worst on third-party clients: phones pause threads to save battery, networks drop. Recommended fix: a **first signal** — a telemetry event fired before any other product code runs, so presence is registered independently of what the variant does. Use it alongside loss metrics, not instead of them.

**3 · Log-processing SRMs.** The MSN carousel bot story. Generalised: this stage is where data gets joined, filtered, aggregated, and where "missing data" is created. Bad joins silently drop rows. Bot filters use post-treatment signals.

The clean rule Booking.com uses: **freeze every user attribute used for filtering at the moment of first exposure**, and forbid changes after that for the duration of the experiment. Then no post-treatment variable can influence who is in the sample. This is exactly the post-treatment-conditioning sin from causal inference, wearing a data-engineering costume.

Whether you can recover depends on *why* data is missing. Transient cooking failure → re-run the pipeline. Delayed telemetry → re-run later. Never collected at all → re-run the experiment. Stream-processing systems that do not retain raw data may not let you re-cook at all.

**4 · Analysis SRMs.** Microsoft Teams tested skipping the First-Run Experience page. The **triggered** analysis (only users who saw, or would have seen, the FRE) had more treatment users; the standard all-users analysis had no SRM. Cause: the client batches events to save network, the old control design loaded slower, control users quit before the batch flushed, and their "I saw the FRE" event was lost. The trigger condition did not capture all eligible users.

This is the LinkedIn finding too: about **10% of triggered analyses there had an SRM**. The fix is to trigger on an *earlier, broader* event — for Teams, something fired right after login rather than the page-visible event — and to make counterfactual logging (control users logging "I would have seen this") a shared framework rather than per-experiment bespoke code.

**5 · Interference SRMs.** A Microsoft Store homepage redesign had an SRM despite everything being correct. Cause: a **search engine marketing campaign with a misconfigured URL** pointing directly at one variant, force-assigning users into it. They segmented those users out; usually you have to re-run.

Two sub-types: *variant interference* — humans forcing variants via debug URL parameters or config strings, pausing one arm mid-flight, ramping only some arms — and *telemetry interference*, where end users attack the product. One case company saw a user run an **injection attack through a telemetry field**, producing a severe SRM in whichever variant they were in.

## Ablation Studies and Experiments

There is no benchmark here. The measurements are prevalence and the diagnostic heuristics.

| Measurement | Number |
|---|---|
| Microsoft experiments with an SRM (last year of data) | **~6%** |
| LinkedIn triggered analyses with an SRM (prior work) | ~10% |
| Skype sessions lost to the config-refresh bug | 30% |
| Experiments analysed for this study | >10,000 |
| Interviews | 14 |
| Distinct root causes in the taxonomy | 25 |
| Products / companies | 25+ / 4 |

Prevalence varies a lot across the five large Microsoft products charted — it is a per-product property, driven by client type and pipeline design. The practical consequence they state plainly: a product running 10,000 experiments a year should expect **at least one SRM per day**.

**The ten rules of thumb**, which are the real deliverable because each is a cheap test that eliminates a whole branch:

1. **Triggered vs standard scorecard.** SRM in the triggered one only → the trigger/filter condition is wrong. Relax the filter progressively and watch where it appears.
2. **Segment by user attribute.** SRM confined to one segment (say, old browsers) → cause is local to that segment's capability.
3. **Segment by time.** Strongest on day 1 and gone later → caching, or a delayed variant start. The most-cited tool in the interviews was simply plotting hourly assignment counts per arm.
4. **Read the performance metrics.** Big regression in page load or crash rate in the SRM'd arm → that degradation is probably real *and* causing the SRM, because the slowest users never logged.
5. **Read the engagement metrics.** Higher average engagement per user in treatment → the cause is preferentially dropping *less* engaged users. Skype was the reverse: the bug hit *long* sessions.
6. **Count SRMs across experiments.** Many unrelated experiments SRM'ing → systemic cause (the taxonomy marks these with `*`), most likely assignment or pipeline.
7. **Check an A/A test.** SRM in A/A → systemic. But beware: if you added extra telemetry to one arm, it is not an A/A any more, because that arm recovers more users.
8. **Look at severity.** Extreme ratios (one arm near zero) → a wholesale logging or trigger failure, not a subtle bias.
9. **Introspect intermediate pipeline stages.** Compare counts at each hop to localise where users vanish.
10. **Compare two independent pipelines.** Two collection/cooking pipelines will not fail identically. Differences localise the bug. Also read the dead-letter logs of records that failed to join.

**What does not work / what they admit.** There is no method that finds the cause automatically. Chi-square tells you *that*, never *why*. Telemetry-loss metrics catch systemic execution SRMs but are useless against design-level SRMs — a badly chosen trigger condition will pass every data-quality check. And for the Skype case the obvious tools all failed; they had to train a **random forest to classify sessions as invalidated vs valid** and read off the most discriminative features, which turned out to all be session-duration-related, which pointed at the config refresh. That "fit a classifier to separate the two classes and read the top features" trick is a genuinely reusable debugging move when the cause is not visible.

## Worth Remembering

**SRM is a specificity-free alarm with enormous power.** It fires on assignment bugs, client crashes, bot filters, bad joins, wrong triggers, marketing campaigns, and injection attacks. That breadth is why it is worth having, and also why the taxonomy is necessary.

**Not all SRMs are bad news.** If treatment made the site faster, more logs survive, and you get an SRM *because the feature works*. One interviewee: "If the treatment is getting us to recover additional users in the logs it almost always means it made some performance improvements. So, these are wins!" You still cannot analyse that experiment as-is — the samples are not comparable — but the diagnosis is a positive finding.

**The deepest lesson is a causal-inference one.** Nearly every case is *conditioning on a post-treatment variable*. The bot flag, the "saw the FRE" event, the successfully-uploaded log — all of these are downstream of treatment, and filtering on them reintroduces the selection bias randomisation was supposed to destroy. Booking's rule (freeze filtering attributes at first exposure) is the general fix, and it is the same rule as "never control for a mediator".

**Limitations.** Prevalence figures are Microsoft-specific and the taxonomy is derived, not exhaustive — it is grounded theory from four companies, so treat the 25 causes as a checklist rather than a partition. There is no tooling artefact released, and no evaluation of whether the rules of thumb actually shorten investigations. The failure modes skew towards rich-client products (Skype, Teams, mobile); a pure server-side system has a narrower surface.

**Practical caveats if you are building this.**
- Run the chi-square on *every* scorecard, including every triggered and segmented one, not just the headline.
- Beware the power trap: at Microsoft's sample sizes a 0.2pp deviation is a 1-in-550k event. At small sample sizes the test has almost no power, so absence of SRM is weak evidence.
- Many SRM checks themselves get run repeatedly as the experiment accumulates data — the same peeking problem as [[Always Valid Inference- Bringing Sequential Analysis to A-B Testing|always-valid p-values]] applies.
- Blocking humans from touching running variants, and restricting force-assignment URLs to internal networks, prevents an entire branch for near-zero cost.

**Open questions.** Can cause classification be automated — feed segment/time/metric deltas into a model that proposes the taxonomy branch? The Skype random-forest trick is a hand-rolled version of exactly that. And what is the right action when an SRM is *caused by the treatment working*? The paper says re-run; it does not offer an estimator that survives treatment-dependent logging.

## Links

Related: [[AB Testing]] · [[Controlled experiments on the web- survey and practical guide (DMKD)]] · [[Trustworthy Online Controlled Experiments- Five Puzzling Outcomes Explained (KDD)]] · [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)]] · [[Applying the Delta Method in Metric Analytics (KDD)]] · [[Always Valid Inference- Bringing Sequential Analysis to A-B Testing]] · [[Graph cluster randomization- network exposure to multiple universes (KDD)]] · [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[Unbiased Learning-to-Rank with Biased Feedback]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Off-Policy Evaluation]] · [[experimentation_question_bank]]

New topics worth writing: Sample Ratio Mismatch, Chi-square goodness-of-fit test, Triggered analysis and counterfactual logging, Post-treatment conditioning bias, Telemetry loss and missing-data mechanisms (MCAR/MAR/MNAR), Experiment assignment hashing and seed collisions, A/A testing as platform validation, Bot detection in experimentation pipelines
