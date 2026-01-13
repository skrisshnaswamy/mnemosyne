---
title: "Inherent Trade-Offs in the Fair Determination of Risk Scores"
authors: ["Jon Kleinberg", "Sendhil Mullainathan", "Manish Raghavan"]
year: 2016
arxiv: "1609.05807"
url: https://arxiv.org/abs/1609.05807
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, theory]
---
## The Core Idea

Three things people mean by "this risk score is fair" cannot all be true at once. Not "are hard to achieve together" — mathematically impossible, except in two degenerate cases.

The setting: you score people with a number between 0 and 1 meant to be the probability they belong to a "positive class" (will reoffend, will click the ad, carries the disease). People come from two groups (race, gender). The three demands:

- **(A) Calibration within groups.** If you give a set of people the score $0.3$, then about 30% of them should be positive — and this must hold separately inside group 1 and inside group 2.
- **(B) Balance for the negative class.** Among people who are truly negative, the average score should be the same in both groups.
- **(C) Balance for the positive class.** Among people who are truly positive, the average score should be the same in both groups.

(B) and (C) are the continuous-score generalisation of "equal false positive rate" and "equal false negative rate".

**Theorem 1.1:** if a risk assignment satisfies all three, then either

1. **perfect prediction** — every $p_\sigma$ is exactly 0 or 1, you already know the answer for everybody; or
2. **equal base rates** — the fraction of positives is identical in the two groups.

Outside those two cases, every possible scoring rule — algorithm, human judge, coin flip, anything — violates at least one of the three. And this is not a statement about a particular estimator or dataset; the theorem quantifies over *all* functions from feature vectors to scores.

Why it did not exist before: the fairness debate around the COMPAS recidivism tool in 2016 was two camps talking past each other. ProPublica showed COMPAS violates (B) and (C) — Black defendants who did not reoffend got higher scores than white defendants who did not reoffend. Northpointe replied that COMPAS satisfies (A). Both were right. Because recidivism base rates differed between the groups, the theorem says exactly one of these outcomes was available. The paper turns an empirical shouting match into a two-line algebra fact.

> [!NOTE] Base rate
> The fraction of a group that truly belongs to the positive class, $\rho_t = \mu_t / N_t$. If base rates differ between groups — and they nearly always do — the impossibility bites. ^base-rate

## The Methodology

There is no training, no dataset, no model. It is a proof. Here is the whole machine.

**Setup.** Each person has a feature vector $\sigma$ — everything the scorer can see. Let $p_\sigma$ be the fraction of people with vector $\sigma$ who are positive. Crucially, $p_\sigma$ is assumed the same for both groups: once you know $\sigma$, knowing the group tells you nothing extra about the outcome. Group $t$ has $n_{t\sigma}$ people with vector $\sigma$; $N_t$ people total; $\mu_t$ positives.

**Risk assignment.** A set of bins, each bin $b$ carrying a score $v_b$, plus a matrix $X$ where $X_{\sigma b}$ is the fraction of $\sigma$-people sent to bin $b$. Randomised splitting is allowed. The rule does **not** see the group label — and the theorem shows that this does not save you.

In matrix form: $P = \mathrm{diag}(p_\sigma)$, $V = \mathrm{diag}(v_b)$, $n_t$ the count vector.

**Step 1 — calibration in matrix form.** The expected number of true positives from group $t$ landing in bin $b$ is $(n_t^\top P X)_b$. The expected total score handed out to group-$t$ people in bin $b$ is $(n_t^\top X V)_b$. Calibration says these are equal, bin by bin:

$$n_t^\top P X = n_t^\top X V$$

**Step 2 — the conservation law.** Multiply both sides by the all-ones vector $\mathbf{e}$:

$$n_t^\top P X \mathbf{e} = n_t^\top X v = \mu_t$$

Read that in words: **calibration forces the total score you give out to a group to equal the number of positives in that group.** Your scores are a budget of size $\mu_t$, and calibration fixes the budget.

**Step 3 — the two lines.** Let $x$ = average score given to a negative person, $y$ = average score given to a positive person. Conditions (B) and (C) say $x$ and $y$ are the *same numbers* for both groups. The budget from step 2 then says, for each group:

$$(N_t - \mu_t)\,x + \mu_t\, y = \mu_t$$

Divide by $N_t$ and write $\rho_t$ for the base rate:

$$(1-\rho_t)\,x = \rho_t\,(1 - y) \quad\Longleftrightarrow\quad \frac{x}{1-y} = \frac{\rho_t}{1-\rho_t}$$

Two groups give two lines in the $(x,y)$ plane. If $\rho_1 = \rho_2$ the lines coincide and anything on the line works. If $\rho_1 \neq \rho_2$ the lines are distinct and meet at exactly one point: $(x,y) = (0,1)$. Average score 0 for every negative, 1 for every positive. Combined with calibration, that forces $p_\sigma \in \{0,1\}$ everywhere — perfect prediction.

That is the entire proof. Two linear equations in two unknowns.

**The two escape hatches, concretely.** Perfect prediction: put all $p_\sigma = 0$ in a bin scored 0 and all $p_\sigma = 1$ in a bin scored 1. Equal base rates: one bin, everyone gets the same score $\rho$. The second is fair and completely useless — it carries zero information.

**Approximate version (Theorem 1.2).** Define $\varepsilon$-approximate (A), (B), (C) as multiplicative slack $[1-\varepsilon, 1+\varepsilon]$ on each equality. Then any instance satisfying all three approximately must satisfy either $f(\varepsilon)$-approximate perfect prediction ($\gamma_t \geq 1 - f(\varepsilon)$, where $\gamma_t$ is the average score of positives in group $t$) or $f(\varepsilon)$-approximate equal base rates ($|\rho_1 - \rho_2| \le f(\varepsilon)$), with

$$f(\varepsilon) = \sqrt{\varepsilon}\,\max\!\left(1,\ 3\sqrt{\varepsilon} + \tfrac{3}{4}\right)$$

The proof assumes base rates differ by at least $\sqrt{\varepsilon}$ and derives $\gamma_1 \geq 1 - \sqrt{\varepsilon}(2\sqrt{\varepsilon} + 3/4)$, a contradiction with being far from perfect prediction.

## Ablation Studies and Experiments

There are no experiments. There is no benchmark, no baseline, no number to report. What the paper does instead is map the boundary of the impossible.

**Statistical parity is even worse off.** Statistical parity — equal *average score over everybody* in each group — is inconsistent with (A) alone when base rates differ, and separately inconsistent with (B)+(C) together. Sum the coordinates of the calibration equation and divide by $N_t$: the left side is the base rate, which differs; so the right side (the mean score) must differ too. So a scorer cannot be both calibrated and statistically parous under unequal base rates.

**The cost of fairness, measured.** Define individual loss as $v$ for a negative person and $1-v$ for a positive person. Group loss works out to

$$\ell_t(X) = 2\left(\mu_t - n_t^\top P X v\right) = 2\mu_t(1-\gamma_t)$$

The loss-minimising assignment is the **identity assignment**: give every feature vector $\sigma$ its own bin scored $p_\sigma$. It is perfectly calibrated. It usually violates (B) and (C). Every other assignment has strictly larger loss. So unless the identity assignment happens to already be fair, **fairness costs accuracy, always, with no exception**.

**When can you do better than the useless single-bin assignment?** With equal base rates, define the *fairness difference* $d = \gamma_1 - \gamma_2$. Lemma 4.1: the set of achievable fairness differences over non-trivial calibrated assignments is an *interval*. The construction is neat — you do not average scores, you concatenate bins: $X^{(3)} = [\lambda X^{(1)}\ \ (1-\lambda)X^{(2)}]$ with the bin score vectors stacked. Calibration is preserved block by block, and the fairness difference is $\lambda d_1 + (1-\lambda) d_2$. Corollary 4.2: a non-trivial fair assignment exists iff there exist non-trivial calibrated assignments, one weakly favouring each group.

**What they could not do.** Finding the *minimum-loss* fair assignment under equal base rates is left open — no polynomial algorithm, no hardness proof, even for deciding whether a non-trivial fair solution exists at all.

**Where hardness does show up.** Restrict to **integral** assignments (everyone with feature vector $\sigma$ goes to one bin, no randomised splitting). Deciding whether a non-trivial fair integral assignment exists is NP-complete, by reduction from Subset Sum. The gadget: for each input number $w_i$ create a pair $p_{\sigma_{2i-1}} = i/(m+1) - \varepsilon_i$ and $p_{\sigma_{2i}} = i/(m+1) + \varepsilon_i$ with $\varepsilon_i = \sqrt{\hat w_i / 2}$. Merging that pair into one bin contributes exactly $\hat w_i$ to a sum that must hit $1/m^4$; merging any *other* pair contributes at least $1/(16m^3)$, which overshoots. So bin-merging choices are forced to be exactly subset selections.

## Worth Remembering

- **The theorem does not care how the score was made.** Algorithm, logistic regression, deep net, a panel of judges — the impossibility is about the score's statistical properties, not its provenance. "Just use a fairer model" is not a way out.
- **Hiding the group label does not help.** The risk rule in the model never sees $t$. Bias appears anyway, because the two groups have different distributions over $\sigma$.
- **The medical framing is the cleanest statement.** If more women than men carry disease $X$, then any probability test for $X$ must have at least one of: (a) probabilities systematically off for one gender, (b) higher average risk assigned to healthy people of one gender, (c) higher average risk assigned to carriers of one gender. This is a fact about arithmetic, not about medicine.
- **$\sqrt{\varepsilon}$ is a weak rate.** Approximate fairness within 1% ($\varepsilon = 0.01$) only guarantees you are within $\approx 0.1$ of one of the special cases. The approximate theorem is qualitatively reassuring but quantitatively loose.
- **The model assumes $p_\sigma$ is known and group-independent given $\sigma$.** This is a strong idealisation — it means the theorem covers the *post-processing* problem: given a well-calibrated risk tool's output, can you fix it up to also satisfy the balance conditions? Answer: no. It sidesteps measurement bias entirely, where the recorded label (arrest) is not the construct of interest (crime), and where $p_\sigma$ may genuinely differ by group. That is a separate and possibly larger problem.
- **Concurrent work.** Hardt–Price–Srebro (equalized odds) drop calibration and so *can* satisfy (B)+(C) for binary predictors, and give an optimisation procedure. Chouldechova independently proved the binary-classifier version of this impossibility the same month. Three papers, one result, autumn 2016.
- **Practical takeaway.** Before arguing about which fairness metric to ship, check the base rates. If they differ, you are choosing which criticism to accept, not whether to accept one. The paper explicitly refuses to say which you should pick.
- **Open question they flag as useful:** when false positives and false negatives have very different social costs, can you get calibration plus *one* balance condition? Unresolved.

## Links

Related: [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[Unbiased Learning-to-Rank with Biased Feedback]] · [[Counterfactual Reasoning and Learning Systems]] · [[Calibrated Recommendations (RecSys)]] · [[Recursive Partitioning for Heterogeneous Causal Effects]] · [[Shortcut Learning in Deep Neural Networks]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Decision Sciences]] · [[Uncertainty]] · [[Random variable]]

New topics worth writing: Equalized odds and equality of opportunity (Hardt et al.), Calibration and reliability diagrams, Statistical parity and disparate impact, COMPAS and the ProPublica debate, Subset Sum and NP-completeness reductions, Individual fairness (Dwork et al.), Measurement bias vs allocative bias in ML fairness
