---
title: "When Does Correction Become Repair? Mechanistic Auditing of Internal Interventions in Tool-Using LLMs"
authors: ["Jiayi Li", "Ruizhe Li"]
year: 2026
arxiv: "2609.36138"
url: https://arxiv.org/abs/2609.36138
priority: Good-To-Read
read_on: 2026-10-05
tags: [paper, llm]
---
## The Core Idea

A tool-using LLM has to make a decision *before* it does anything: call a tool, ask the user a question, answer straight away, or refuse. That decision is a pick from $K$ options. Getting it wrong is a different kind of failure from saying something false — nothing has been generated yet, the model has just chosen the wrong *kind* of move.

People already know you can change that choice by editing the model's internal activations. Add a vector to a hidden layer, and the model switches from "answer directly" to "call the tool". The usual way to report this is net accuracy gain: how many decisions got fixed, minus how many got broken.

The insight here is that **net gain is a lossy summary**, and in a $K$-way choice it hides three separate things.

Earlier steering work almost always used a binary action space — tool or no tool, refuse or comply. In a binary space, leaving the wrong answer *is* arriving at the right one. There is nowhere else to go. So nobody had to ask the question. With four actions, leaving the wrong answer can land you on a *third* wrong answer, and accuracy does not change at all, so the metric never sees it.

> [!NOTE] Directional error channel
> A specific ordered mistake: "the right action was $g$, the model produced $s$". Written $c = (g \rightarrow s)$. Not "the model makes tool errors" — a particular wrong turn, with its own steering vector. ^error-channel

The paper's framework, SAKIKO, splits every intervened decision into five buckets instead of one number. On the error cases: the model kept the wrong answer, reached the right answer, or moved to a *different* wrong answer. On the already-correct cases the intervention touched: still correct, or now broken.

Three facts fall out, and they are the whole paper:

1. **Movement is not arrival.** On Phi-3.5, 153 of 200 errors left the wrong answer. Only 107 reached the right one. 46 moved sideways to another error. Accuracy records those 46 as nothing happening.
2. **Fixing is not preserving.** That same Phi-3.5 run shows net $+55$. It also broke **52 of the 93 already-correct decisions** its detector fired on. The $+55$ is $107 - 52$. More than half of what it touched that was working, it broke.
3. **A good point estimate is not evidence.** Two models hit encouraging target-hit rates ($0.578$ and $0.630$) and the protocol still refuses to certify them, because the confidence intervals straddle the threshold.

What this unlocks is a vocabulary for saying what an intervention actually did, instead of whether a scalar went up.

## The Methodology

### The setup

Frozen model, no weight updates. Action space $\mathcal{Y}$ with four options on the main benchmark (When2Call): `tool_call`, `request_for_info`, `cannot_answer`, `direct`. The decision is read out by scoring all four complete candidate responses by teacher forcing and taking the argmax — deterministic, so baseline and intervened runs differ only through the intervention.

Split: 2,556 train / 548 validation / 548 locked test.

### The populations — this is where the paper is careful

Every rate needs a named denominator or it is ambiguous. Four sets:

$$E_c = \{x : y^\star = g,\ \hat{y}_0 = s\} \quad\text{(channel errors, } N_c = |E_c|)$$
$$C = \{x : \hat{y}_0 = y^\star\} \quad\text{(all baseline-correct)}$$
$$C_{\mathrm{exp}} = \{x \in C : \mathrm{fire}(x)\} \quad\text{(correct decisions the detector touched)}$$

$C_{\mathrm{exp}} \subseteq C$, and that containment is the crux of the collateral story.

### The five outcome buckets

$$N_c = S_c + A_c + O_c, \qquad |C_{\mathrm{exp}}| = R + B$$

- $S_c$ — kept the source error ($\hat{y}_1 = s$)
- $A_c$ — **gold arrival**, reached the right action ($\hat{y}_1 = g$)
- $O_c$ — **other wrong**, moved to a third action
- $R$ — correct, stayed correct
- $B$ — **broken**, was correct, now isn't

Source exits: $X_c = A_c + O_c$.

### The endpoints

$$\mathrm{TH}_c = \frac{A_c}{X_c} \quad\text{target-hit: of the ones that moved, how many landed right}$$
$$\mathrm{TG}_c = \frac{A_c - O_c}{N_c} \quad\text{target gain: normalised over the whole channel, so it carries coverage too}$$
$$\mathrm{E1} = \frac{B}{|C|}, \qquad \mathrm{E2} = \frac{B}{|C_{\mathrm{exp}}|}$$

Net gain in its two flavours (and the paper never mixes them):
$$G_{\mathrm{whole}} = \mathrm{Fixed} - B, \qquad G_{\mathrm{channel}} = \sum_c A_c - B$$

Note what net gain *omits*: $O_c$ appears in neither. A sideways move from one error to another leaves accuracy untouched. So for any fixed $G$ and any target rational $t \in (0,1]$, you can set $O_c = A_c(1-t)/t$ and get target-hit exactly $t$ at unchanged $G$. The map from five classes to one number is many-to-one. That is the formal content of the headline claim.

> [!NOTE] E2 ≥ E1, always
> Breaks happen only inside $C_{\mathrm{exp}}$, and $C_{\mathrm{exp}} \subseteq C$, so $\mathrm{E2} \geq \mathrm{E1}$. A selective detector makes $|C_{\mathrm{exp}}|$ small, which makes $\mathrm{E1}$ look good while $\mathrm{E2}$ stays terrible. Reporting $\mathrm{E1}$ alone understates the risk to a touched decision by the factor $|C|/|C_{\mathrm{exp}}|$. ^two-denominators

### The pipeline, eight stages

**Discover.** Build the baseline confusion matrix on train, treat every off-diagonal cell as a candidate channel. No preselected error taxonomy.

**Support.** A channel advances only with at least 30 channel errors in the evaluation population and at least 30 baseline-correct reference rows per split. Failing this means *unadjudicable*, not uncorrectable — a statement about the sample, not the model. This gate killed two settings before any steering was tried.

**Detect.** Per channel, fit a linear Router on the last-prompt-position hidden state at observation layer $\ell_{\mathrm{obs}}$. Standardised features, $L_2$ logistic regression at $C = 1.0$, liblinear, tol $10^{-4}$, cap 2,000 iterations. Positives are the channel's training errors; negatives are **every** baseline-correct training row, not just the ones sharing a gold label. Fire iff $p_{\hat{c}} \geq \tau_{\hat{c}}$, with $\tau$ picked from $\{0.4, 0.5, 0.6, 0.7, 0.8\}$ as the smallest value reaching precision $0.50$ on development rows whose baseline prediction is the source action. Eligibility needs dev ROC AUC $\geq 0.75$. No fire means the forward pass is untouched and $\hat{y}_1 = \hat{y}_0$.

**Estimate.** The sealed estimator is a gradient / target-axis direction:
$$d_{\mathrm{grad}} = \mathrm{unit}\!\left(\frac{1}{n}\sum_i \mathrm{unit}(G_{i,\text{gold}} - G_{i,\text{source}})\right)$$
where $G_{i,m}$ is the gradient of the mean log-probability of candidate $m$'s tokens, taken with respect to the MLP output at $\ell_{\mathrm{inj}}$ and summed over sequence positions. The inner normalisation makes the outer mean an average of *directions*, so one big-gradient example can't dominate. Train rows only. Nothing inside is searched or tuned. A difference-in-means direction runs as a comparator arm; PCA-1 is historical only.

**Intervene.**
$$h'_{\ell_{\mathrm{inj}}} = h_{\ell_{\mathrm{inj}}} + q_c\, s_c\, d_c$$
Applied at the MLP forward output, before the residual add, at every sequence position. $d_c$ unit-norm, $s_c$ the median Euclidean norm of that MLP output over training rows where the Router fires, $q_c$ a relative dose from a frozen grid $\{0, 0.125, 0.25, 0.5, 1.0, 2.0\}$ under a `min(admissible)` rule. The product $q_c s_c$ is the **absolute perturbation budget** and it is what gets held constant across control arms — not $q_c$.

Injection precedes observation in the stack (Qwen3-8B: inject at 21, observe at 26), so a firing decision requires a second forward pass. Layer sites come from a fixed normalised-depth rule anchored on Qwen2.5-7B's $(20, 16)$ over 28 layers, not a per-model search.

**Verify.** Resolve all five buckets, both denominators.

**Control.** Zero-dose, 59 budget-matched random directions, sign-reversed, wrong-layer, frozen difference-in-means, ungated, and a score-space comparator (a frozen logit shift, testing whether an internal edit is needed at all). Cross-channel substitution is *excluded by construction* from the sealed inventory, along with PCA, pooled directions, new estimators and extra doses — excluding them before sealing is what stops the control battery becoming a search.

**Admit/Decline.** A ten-condition gate, written before the first sealed run, applied unchanged. Ten of ten admits; nine of ten declines.

| # | Condition | Threshold |
|---|---|---|
| 1 | channel support | $N_c \geq 30$ |
| 9 | zero arm reproduces baseline | exact |
| 10 | no structural failure | — |
| 4 | specificity, add-one $p$ over $K=59$ | $\leq 0.05$ |
| 2 | target gain, point | $> 0$ |
| 5 | target-hit, point | $> 0.50$ |
| 3 | target gain, CI lower | $> 0$ |
| 6 | target-hit, CI lower | $> 0.50$ |
| 7 | collateral $\mathrm{E1}$, point | $\leq 0.05$ |
| 8 | collateral $\mathrm{E1}$, CI upper | $\leq 0.05$ |

Significance is an add-one Monte Carlo test against the empirical random null:
$$p = \frac{1 + \#\{\mathrm{TG}_{\mathrm{rand}} \geq \mathrm{TG}_{\mathrm{real}}\}}{K + 1}, \quad K = 59$$
Floor $1/60 = 0.0167$. Destination intervals are percentile bootstrap over **channel errors**, not exits — keeping $N_c$ fixed propagates the uncertainty in how many decisions move at all. 10,000 draws.

The repair claim is a conjunction:
$$\textsc{REPAIR}(\pi) \equiv \textsc{CORRECTABLE}(\pi) \wedge \textsc{PRESERVING}_{\text{pop}}(\pi) \wedge \textsc{LICENSABLE}(\pi)$$

### Models

Seven, 3.8B–9B, in two cohorts. **Historical** (Phi-3.5-mini, Qwen2.5-7B, Llama-3.1-8B, Mistral-7B) — three channels, difference-in-means, evaluated on whole-population net gain, no formal verdict because the gate didn't exist yet. **Sealed** (Qwen3-4B, Qwen3-8B, Gemma-2-9B) — one preselected channel (`cannot_answer` → `tool_call`), everything hashed and frozen before the test split was opened, 65 arms per setting (6 named + 59 random), run exactly once.

## Ablation Studies and Experiments

### Correction works, and it is direction-specific

| Model | Effect | Controls |
|---|---|---|
| Qwen2.5-7B | net $+79$ (92 fixed, 13 broken) | reversed $+16$; 10 randoms mean $+25.2$, max $+50$ |
| Phi-3.5-mini | net $+55$ | random $+14$, reversed $+14$, wrong layer $+3$, **cross-channel $-17$** |
| Qwen3-8B | $\mathrm{TG}_c = 0.276$ | 0/59 randoms reach it, largest $0.023$ |
| Qwen3-4B | $\mathrm{TG}_c = 0.081$ | 0/59, largest $0.000$ |
| Gemma-2-9B | $\mathrm{TG}_c = 0.073$ | 0/59, largest $0.000$ |

The cross-channel number is the sharpest: applying one channel's vector to another channel makes things **worse than doing nothing**. The three Qwen2.5-7B channel directions have pairwise cosines of $0.488$, $0.569$, $0.706$ — related, but not interchangeable. A single monolithic "tool error" vector would conflate them.

The random null is built carefully. Seeds derive from `SHA256(salt ‖ i)`, a PCG64DXSM generator, normalised in float64, hash-pinned before the split opened. Verified unit-norm to $2.7 \times 10^{-9}$, median pairwise $|\cos| = 0.011$, max $|\cos|$ against the real direction $0.050$. No rejection sampling, no orthogonalisation, no cosine filtering, no outcome matching — and the cosines were computed *after* freezing, with nothing removed on their basis.

On MetaTool (binary), steering two **opposite** channels on disjoint populations both improved: $+8.6$ and $+13.0$ mean net gain over five runs. That rules out "the vector just makes the model call tools more". It says nothing about destination, because $O_c = 0$ by construction in a binary space.

### What did not work

**Two models failed specificity outright.** On Llama-3.1-8B, 5 of 20 random directions match or beat the real effect. On Mistral-7B, the calibrated direction ($+12$) is **below** the random mean ($+29$), 17 of 20 randoms beat it, and the sign-reversed vector ($+19$) also beats it. Linear steerability is not universal. Both rest on secondary audit records with $K=20$, so the conclusion is "insufficient evidence under this configuration", not "no direction exists".

**One model never got steered.** Qwen3.5-9B had the *highest* baseline accuracy ($0.4533$) and was disqualified before intervention. Its accuracy gain was unevenly distributed: recall on follow-up questions roughly doubled ($0.326$ vs $0.170$ and $0.132$ in the panel) while recall on refusals fell to the lowest in the panel ($0.087$). Since reference and error rows inside a gold class compete for fixed items, that starved every refusal channel's reference side — 29 rows against the frozen minimum of 30. Protocol halt. (Honest caveat: split-sensitive — eligible in 56.7% of 30 retrospective reallocations.)

**One benchmark never got steered.** ACEBench passed every readout validity check (accuracy $0.756$, macro-F1 $0.687$, 7.25% unparseable, 100% replay agreement). It failed channel discovery: no transition met the per-split support minimum, and a paraphrase-agreement gate returned $0.56$ against a required $0.80$. Retired. The authors flag that their paraphrase carried a weaker format directive than the original, so the $0.56$ is not clean evidence of wording fragility.

### The three dissociations — the actual contribution

**1. Movement ≠ arrival.** Phi-3.5: 200 routed errors → 153 exits → 107 gold arrivals, 46 sideways. $\mathrm{TH}_c = 0.699$. Nearly a third of movement goes nowhere useful and accuracy cannot see it.

On Qwen3-8B, internal steering vs a frozen logit shift:

| | Activation | Score-space |
|---|---|---|
| gold arrivals $A_c$ | 38 | 37 |
| other wrong $O_c$ | 14 | 21 |
| target-hit | $0.731$ | $0.638$ |
| Net (whole) | $+40$ | $+39$ |

Aggregate says these are the same intervention. Destination says seven decisions went to a third wrong action instead. The paired tests are exact McNemar $p = 1.0$ and $p = 0.189$ — not significant, and the comparison wasn't prespecified, so the authors report it descriptively and claim **no ordering**. Across the three sealed settings the two methods don't order consistently anyway: activation leads on Qwen3-8B and Gemma, score-space leads on Qwen3-4B.

**2. Fixing ≠ preserving.** Phi-3.5 breaks 52 of 93 exposed correct decisions. $\mathrm{E2} = 55.9\%$, $\mathrm{E1} = 19.7\%$, tolerance $5\%$. Both blown.

And the Router confidence **cannot** separate broken from intact cases: Cliff's $\delta = -0.094$, $p = 0.44$. Break rates stay above 50% even at $\tau_c = 0.9$. Raising the threshold does not buy safety — that's the uncomfortable finding, because "just be more selective" is the obvious fix and it doesn't work here.

The denominator matters enormously. Qwen3-4B: benign $\mathrm{E1} = 1.3\%$ hiding $\mathrm{E2} = 5.7\%$. Qwen3-8B: zero breaks, but on only **6 exposed decisions**, giving a useless $\mathrm{E2}$ upper bound of $0.393$.

**3. Point estimate ≠ evidence.** Qwen3-8B admits: $\mathrm{TH}_c = 0.731$, CI $[0.604, 0.846]$; $\mathrm{TG}_c = 0.276$, CI $[0.115, 0.425]$; $B = 0$. Qwen3-4B declines on exactly two conditions and nothing else: $\mathrm{TH}_c = 0.578$ with CI $[0.456, 0.697]$ crossing $0.50$, and $\mathrm{TG}_c$ CI $[-0.048, 0.202]$ spanning zero. Gemma-2-9B same pattern, CI $[-0.031, 0.177]$.

The two sealed Qwen settings differ in aggregate net gain by **one decision**: $+40$ against $+39$. Both clear every point estimate. Both clear the $K=59$ specificity null. The only thing separating ADMIT from DECLINE is interval width.

The power table shows the declines are exactly what the sample sizes predict — exits needed to certify a true target-hit above $0.50$ at $\alpha = 0.05$, 80% power:

| true $\mathrm{TH}$ | 0.73 | 0.68 | 0.63 | 0.60 | 0.58 | 0.55 |
|---|---|---|---|---|---|---|
| exits needed | 30 | 49 | 93 | 158 | **245** | 620 |

Qwen3-8B had 52 exits against a requirement of ~30. Qwen3-4B had 64 against 245. Gemma had 27.

### What the ablations reveal about which piece does the work

**Decodability ≠ steerability.** On Qwen2.5-7B layers 16–22, linear probe AUROC plateaus at $\approx 0.96$. Over the same layers, the difference-in-means norm climbs $2.12 \to 5.88 \to 10.41 \to 14.26$ and validation net gain goes from $+13$ at layer 16 to $+46$ at layer 20. Layer 16 fails because $\cos(\text{DiffMean}, \mathrm{PC}_1) = 0.0002$ — the direction is near-orthogonal to the channel's leading principal component. **A probe telling you the state is readable tells you nothing about whether intervening there will help.**

**Gating is not uniformly necessary.** Removing the Router: Phi-3.5 $+62 \to -32$, MetaTool $\to -39$ and $-34$, Qwen3-8B $+38 \to +13$ with collateral $0/6 \to 29/176$ ($16.5\%$). But Qwen2.5-7B goes $+79 \to +94$ (ungated is *better* on net), and Gemma goes $+16 \to +17$. So gating raises net in three of five settings and lowers it in two. What it raises *consistently* is arrivals per broken decision: $5.27 \to 7.08$ on Qwen2.5-7B, $5.25 \to 17.00$ on Gemma, and Qwen3-8B's gated arm breaks nothing at all. The authors explicitly decline to make a necessity claim on net.

**Dose is non-monotonic.** Qwen3-4B target-hit spans $0.525$ to $0.911$ across doses, peaking at intermediate magnitudes and regressing at higher ones.

**Reversed and misallocated arms are genuinely dead, not merely weaker.** On Qwen3-4B the reversed direction moved **zero** decisions off their source. The frozen difference-in-means estimator at the same site reached $\mathrm{TG}_c = -0.016$ on 4 exits; wrong-layer $-0.024$ on 15. Same pattern on Gemma: reversed moved nothing, DiffMean produced one exit and zero arrivals.

**The licensing rule changes conclusions.** Replaying all nine evaluated units against eight alternative reporting rules: net-gain-above-zero accepts all nine; adding specificity removes three; adding point-estimate destination removes one more; bounded collateral removes another; non-vacuous collateral bound removes one more; the full conjunction accepts **one**. The interval layer alone is worth three units — three clear every point estimate and two are still declined.

## Worth Remembering

**The ADMIT is heavily qualified, and the authors say so.** Qwen3-8B passes preservation on $\mathrm{E1}$ with zero breaks over 211 correct decisions. But the Router exposed only **6** of them, so the exposure-conditional bound is $0.393$ — vacuous. The statement supported is "no collateral damage observed in the population", not "safe on the decisions it touches". The gate uses $\mathrm{E1}$ purely because the Qwen3-8B preregistration fixed that denominator before $\mathrm{E2}$ was recorded, and they refused to retrofit a frozen rule.

**No setting in the paper certifies exposure-conditional preservation.** Not even the ADMIT. Bounding $\mathrm{E2}$ at $5\%$ with zero observed breaks needs 59 exposed decisions; with one break, 93; two, 124; three, 153. Observed exposures: Qwen3-8B 6, Gemma 11, Qwen3-4B 50, Phi-3.5 93 with 52 breaks. Under a gate that actually tested the named property, the ADMIT would become a fourth decline.

**The bootstrap bound on collateral is degenerate at zero breaks.** If $B = 0$, every nonparametric resample also has $B = 0$, so the bootstrap 95% upper bound is exactly zero and Condition 8 *cannot fail*. The paper prints Clopper–Pearson bounds instead ($0.0141$ on $0/211$), which are stricter, and notes the verdict doesn't turn on it — but flags that a gate written today should name a rare-event-informative bound up front.

**Two gaps the authors name in their own gate.** First, no coverage condition: $\mathrm{TH}_c$ is computed over exits, so a Router that fires rarely on easy rows scores well. Arrivals over $N_c$ for the three sealed settings are $0.437$, $0.298$, $0.177$ — considerably less flattering. Second, Condition 2 only asks $\mathrm{TG}_c > 0$, licensing an arbitrarily small gain. They decline to add either retrospectively because a threshold set after the evaluation means nothing.

**$\mathrm{TG}_c$ and $\mathrm{TH}_c$ are not independent at the point estimate.** Since $\mathrm{TG}_c = (X_c/N_c)(2\mathrm{TH}_c - 1)$, Conditions 2 and 5 are algebraically equivalent when $X_c > 0$. Kept separate only because the freeze numbered them so. Their *interval* versions (3 and 6) are genuinely distinct.

**Everything rests on one benchmark.** All formal outcomes are When2Call. MetaTool is binary so destination isn't posable. ACEBench was retired. Labels are synthetically generated by the benchmark authors with no re-annotation, so a systematic labelling bias could shift destination distributions without tripping any gate.

**The positive claim is one model, one channel, one dose.** Qwen3-8B on `cannot_answer` → `tool_call` with gradient steering. Not the family, not other transitions, not activation steering in general. ASA and CAST were not re-implemented as baselines because the protocol was locked before sealed execution.

**Provenance is unusually complete and includes the embarrassing bits.** Three pre-access gate repairs, one mechanical VOID with exactly one authorised restart on Qwen3-4B (all arms re-executed from scratch, no resume), historical random-null sizes ranging from 1 to 10 directions on early runs versus 59 sealed, six rows dropped for 24GB OOM (three train, three dev, **none in the sealed split**, with the feasibility boundary a natural gap at sequence length 3,169 vs 4,572). A corrections table lists six superseded numbers, including one layer-sweep result misattributed to Phi-3.5 that actually belongs to Qwen2.5-7B.

**Practical caveats if you wanted to use this.** Artifacts are Git LFS; a checkout without `git lfs pull` gives you 132-byte pointer files that read as *empty rather than failing* — your reproduction will silently produce nothing. Determinism needs batch size 1, single process, single model load, fixed arm and sample order, no checkpointing between arms.

**Open questions.** Does the result survive outside English text prompts — the paper cites work showing tool-calling failures shift across languages and again under speech rendering of the same benchmarks, including When2Call. Can a Router be built that exposes enough correct decisions to make $\mathrm{E2}$ certifiable, given that raising the threshold demonstrably doesn't work? And does pre-execution repair connect to downstream outcome quality at all — the paper audits which action mode was picked, never whether the resulting call was good.

## Links

Related: [[Tool Use]] · [[Guardrails]] · [[Evals]] · [[Agent Evaluation]] · [[Saliency]] · [[Observability and Tracing]] · [[Uncertainty]] · [[Always Valid Inference- Bringing Sequential Analysis to A-B Testing]] · [[Controlled experiments on the web- survey and practical guide (DMKD)]] · [[Trustworthy Online Controlled Experiments- Five Puzzling Outcomes Explained (KDD)]] · [[Troubling Trends in Machine Learning Scholarship]] · [[On the Difficulty of Evaluating Baselines]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Reward Hacking]] · [[Sparse Readout Prism- Explaining Logit-Lens Scores in Features Instead of Tokens]] · [[Planning and Decomposition]] · [[Agentic Workflows]] · [[Human in the Loop]]

New topics worth writing: Activation steering, Representation engineering, Linear probes on hidden states, Preregistration in ML evaluation, Clopper–Pearson intervals, Add-one Monte Carlo permutation tests, Cliff's delta, McNemar's test, Statistical power for proportions, Sparse autoencoders for interpretability, Conditional Activation Steering (CAST), When2Call benchmark, Destination-resolved evaluation
