---
title: "Recommendations as Treatments: Debiasing Learning and Evaluation"
authors: ["Tobias Schnabel", "Adith Swaminathan", "Ashudeep Singh", "Navin Chandak", "Thorsten Joachims"]
year: 2016
arxiv: "1602.05352"
url: https://arxiv.org/abs/1602.05352
priority: Must-Read
read_on: 2026-08-25
tags: [paper, optimization, theory]
---
## The Core Idea

Ratings you see are not a random sample of ratings that exist. People watch movies they expect to like, and rate those. So the observed data is **MNAR** — Missing Not At Random. The missingness depends on the very thing you are trying to predict.

The trick: treat "user $u$ saw item $i$" exactly like "patient got the drug" in a medical study. Showing an item is an **intervention**. Then all the standard machinery of causal inference applies — in particular [[Unbiased Learning-to-Rank with Biased Feedback|propensity weighting]].

> [!NOTE] Propensity ^propensity-score
> $P_{u,i} = P(O_{u,i}=1)$ — the probability that the rating of user $u$ for item $i$ ever becomes visible to you. Not a probability the user likes it. A probability you get to *see* it.

Why this did not exist before in recsys: prior MNAR work (Marlin, Hernández-Lobato) modelled the missingness process and the rating process **jointly**, in one big generative likelihood. That is complicated and slow. This paper separates them. Estimate propensities with whatever model you like; then just **reweight each observed rating by $1/P_{u,i}$** in your ordinary matrix factorization loss. That is the whole change. One extra multiplication per term.

What it unlocks:
1. **Unbiased offline evaluation.** Naive test-set MSE/MAE/DCG on held-out observed ratings is not just noisy, it *ranks models wrong*. IPS fixes the ranking.
2. **A drop-in fix for training.** Any existing weighted-MF optimiser works unchanged.
3. **Learning theory.** You get a generalisation bound, and the bound tells you exactly how much bad propensity estimates cost you.

The toy example that makes it click: horror lovers rate horror 5 and romance 1, but rarely bother rating the romance films they hate. So 1-star ratings are almost absent from the observed data. A model that predicts "everything is a 4" scores *better* on observed-only MAE than a model that is actually correct, because the 1s it would get right are invisible.

## The Methodology

**Setup.** True rating matrix $Y \in \Re^{U \times I}$, fully defined but mostly unseen. Observation mask $O \in \{0,1\}^{U\times I}$. Propensity matrix $P$ with $P_{u,i} > 0$ everywhere (every cell *could* be seen — this is the positivity/overlap assumption).

**The target.** Every metric is written as an average over the *full* matrix:
$$R(\hat Y) = \frac{1}{U\cdot I}\sum_{u=1}^{U}\sum_{i=1}^{I}\delta_{u,i}(Y,\hat Y)$$
with $\delta$ plugged in per metric: $|Y_{u,i}-\hat Y_{u,i}|$ for MAE, squared for MSE, $(I/\log(\text{rank}(\hat Y_{u,i})))\,Y_{u,i}$ for DCG, $(I/k)\,Y_{u,i}\mathbf{1}\{\text{rank}\le k\}$ for PREC@k. Neat: rating-accuracy metrics and ranking metrics live in the same equation.

**The naive estimator** averages only over observed cells:
$$\hat R_{naive}(\hat Y) = \frac{1}{|\{(u,i):O_{u,i}=1\}|}\sum_{(u,i):O_{u,i}=1}\delta_{u,i}(Y,\hat Y)$$
Biased whenever $\delta$ correlates with whether the cell is observed. Which is always.

**IPS estimator:**
$$\hat R_{IPS}(\hat Y|P) = \frac{1}{U\cdot I}\sum_{(u,i):O_{u,i}=1}\frac{\delta_{u,i}(Y,\hat Y)}{P_{u,i}}$$
Unbiased. The proof is one line: $\mathbb{E}[O_{u,i}/P_{u,i}] = 1$. Note it needs only the **marginal** probabilities — correlations inside $O$ do not break unbiasedness (they do affect variance).

**SNIPS estimator** — divide by the realised sum of weights instead of $U\cdot I$:
$$\hat R_{SNIPS}(\hat Y|P) = \frac{\sum_{O_{u,i}=1}\delta_{u,i}/P_{u,i}}{\sum_{O_{u,i}=1}1/P_{u,i}}$$
This is a control-variate trick: we *know* $\mathbb{E}[\sum 1/P_{u,i}] = U\cdot I$, so use the observed sum as a self-correction. Slightly biased, usually lower variance.

**Variance cost.** Proposition 3.1 gives, with probability $1-\eta$,
$$|\hat R_{IPS} - R| \le \frac{1}{U\cdot I}\sqrt{\tfrac{1}{2}\log\tfrac{2}{\eta}\sum_{u,i}\rho_{u,i}^2},\quad \rho_{u,i}=\delta_{u,i}/P_{u,i}$$
With uniform $P_{u,i}=p$ this is $O(1/(p\sqrt{UI}))$. With very non-uniform propensities it blows up even at the same expected sample size. Small propensities are the enemy.

**MF-IPS — the actual model.** Hypothesis class is bog-standard biased matrix factorization, rank $d$:
$$\hat Y_{u,i} = v_u^\top w_i + a_u + b_i + c$$
Objective:
$$\arg\min_{V,W,A}\left[\sum_{O_{u,i}=1}\frac{\delta_{u,i}(Y, V^\top W + A)}{P_{u,i}} + \lambda(\|V\|_F^2 + \|W\|_F^2)\right]$$
Identical to ordinary incomplete MF except each loss term is divided by its propensity. Conventional MF is the special case where all $P_{u,i}$ are equal (MCAR). Optimised with **L-BFGS**.

**Generalisation bound (Theorem 4.2)**, finite $\mathcal{H}$, loss bounded by $\Delta$:
$$R(\hat Y^{ERM}) \le \hat R_{IPS}(\hat Y^{ERM}|P) + \frac{\Delta}{U\cdot I}\sqrt{\frac{\log(2|\mathcal H|/\eta)}{2}}\sqrt{\sum_{u,i}\frac{1}{P_{u,i}^2}}$$

**Estimating propensities when you did not run the experiment.** Two options:

*Naive Bayes.* Assume $O$ depends only on the rating value: $P(O_{u,i}=1|Y_{u,i}=r) = \frac{P(Y=r|O=1)P(O=1)}{P(Y=r)}$. The numerator you count from your MNAR logs. The denominator $P(Y=r)$ — the *true* rating distribution — needs a **small MCAR sample** (randomly-served items). No way around that.

*Logistic regression.* $P_{u,i} = \sigma(w^\top X_{u,i} + \beta_i + \gamma_u)$ with user and item offsets, $X_{u,i}$ any observable covariates. No MCAR sample needed, but you must assume the observables explain away the dependence on $Y$.

**Robustness theory.** With wrong propensities $\hat P$, the bias is exactly
$$\text{bias}(\hat R_{IPS}(\hat Y|\hat P)) = \sum_{u,i}\frac{\delta_{u,i}}{U\cdot I}\left[1 - \frac{P_{u,i}}{\hat P_{u,i}}\right]$$
and Theorem 5.2 adds a $\frac{\Delta}{U\cdot I}\sum_{u,i}|1 - P_{u,i}/\hat P_{u,i}|$ term to the bound while the variance term now uses $\hat P$. Practical reading: **deliberately overestimating small propensities can help** — you add bias but shrink variance more. A [[Regularization|bias–variance]] knob that plain ERM does not have.

**Model selection.** Cross-validation with $k=4$ folds on the MNAR data, scoring folds with the IPS estimator itself. Subtlety: splitting changes the propensities, so scale training-fold propensities by $\frac{k-1}{k}$ and validation-fold ones by $\frac{1}{k}$.

## Ablation Studies and Experiments

**Semi-synthetic ML100K.** They complete the 100K MNAR ratings with plain MF, then re-bucket the values to match a realistic rating marginal from Marlin & Zemel, giving a *fully known* $Y$ (944 users × 1683 movies). Observation model: propensity $k$ if the true rating is 4 or 5, and $k\alpha^{4-r}$ for $r<4$. $k$ set so 5% of cells are revealed. $\alpha=1$ is MCAR; $\alpha\to 0$ reveals only 4s and 5s. $\alpha=0.25$ reproduces the real ML100K observed rating marginals ($[0.06,0.10,0.25,0.42,0.17]$ vs real $[0.06,0.11,0.27,0.35,0.21]$).

**Estimator accuracy (Table 1, $\alpha=0.25$, 50 draws of $O$).** Five hand-built prediction matrices. The damning pair:

| $\hat Y$ | True MAE | IPS | SNIPS | Naive |
|---|---|---|---|---|
| REC ONES | 0.102 | 0.102 ± 0.007 | 0.102 ± 0.007 | **0.011** ± 0.001 |
| REC FOURS | 0.026 | 0.026 ± 0.000 | 0.026 ± 0.000 | **0.173** ± 0.001 |
| ROTATE | 2.579 | 2.581 ± 0.031 | 2.579 ± 0.012 | 1.168 ± 0.003 |
| SKEWED | 1.306 | 1.304 ± 0.012 | 1.304 ± 0.009 | 0.912 ± 0.002 |
| COARSENED | 1.320 | 1.314 ± 0.015 | 1.318 ± 0.005 | 0.387 ± 0.002 |

Naive says REC ONES (MAE 0.011) beats REC FOURS (0.173). The truth is the reverse — 0.102 vs 0.026. **The naive estimator inverts the ranking of models.** DCG@50 is worse still: true 30.76 for REC ONES, naive says 153.07.

Key number: the standard deviation of IPS (±0.007) is far smaller than the bias Naive incurs (0.09 on that row). Unbiasedness is worth the variance.

**Sweeping $\alpha$ (Figure 2).** IPS and SNIPS are orders of magnitude more accurate than Naive over nearly the whole range. Even at very small $\alpha$ (brutal bias, brutal weight variance) IPS still wins. At $\alpha=1$ (MCAR) SNIPS is algebraically identical to Naive, and IPS pays a small penalty for weighting noise — that is the **only** regime where propensity weighting hurts.

SNIPS beats IPS consistently on MSE; on DCG they tie. So the control variate helps for pointwise losses, not for ranking losses.

**Learning (Figure 3, left).** MF-IPS vs MF-Naive, both $d=20$, $\lambda$ cross-validated separately, 30 trials. MF-IPS wins on MSE across all $\alpha$. Same story for MAE.

**Degraded propensities (Figure 4, Figure 3 right).** They shrink the MCAR sample used for the Naive Bayes marginal, so propensity estimates get progressively worse. Findings:
- IPS/SNIPS **never** do worse than Naive, at any level of propensity degradation.
- Surprising: **IPS-NB with *estimated* propensities sometimes beats IPS-KNOWN with true propensities** on MSE. This is a known effect in causal inference (Hirano et al. 2003) — estimated propensity scores act like stratification and can be *more* efficient than the true ones.
- MF-IPS-NB beats MF-Naive even under severely degraded estimates.

**Real data (Table 2).** Yahoo! R3 (300K self-selected song ratings, 15400 users; MCAR test set from 5400 users rating 10 random songs; propensities via Naive Bayes using 5% of the test set as the MCAR sample). Coat: a new dataset they collected — 290 Turkers shopping for a coat, each rating 24 self-selected + 16 random coats out of 300; propensities via logistic regression on all user×item covariate pairs (gender, age, location, fashion-awareness × gender, coat type, colour, promoted).

| | Yahoo MAE | Yahoo MSE | Coat MAE | Coat MSE |
|---|---|---|---|---|
| **MF-IPS** | **0.810** | **0.989** | **0.860** | **1.093** |
| MF-Naive | 1.154 | 1.891 | 0.920 | 1.202 |
| HL MNAR | 1.177 | 2.175 | 0.884 | 1.214 |
| HL MAR | 1.179 | 2.166 | 0.892 | 1.220 |

All differences significant, paired t-test $p<0.001$. Yahoo MSE of 0.989 beats the best published number at the time (CTP-v at 1.115); MAE 0.810 is close to but not better than 0.770. Hyperparameters: $\lambda \in \{10^{-6},\dots,1\}$, $d \in \{5,10,20,40\}$, cross-validated. Note the baselines got a generous deal — HL's $d$ was picked by **best test performance**, and MF-IPS still won.

Striking: the sophisticated generative joint-likelihood models (HL MNAR) barely beat plain MF-Naive on Yahoo, and lose to it on Coat MSE. The simple discriminative reweighting beats both by a wide margin.

**What did not work / limits found.** Non-differentiable $\delta$ (DCG, PREC@k) cannot be plugged directly into the MF objective — they only train on MSE/MAE and note that structured-SVM or smoothing approaches would be needed. SNIPS gave no variance reduction for DCG. And IPS is strictly worse than Naive under true MCAR data.

## Worth Remembering

- **The headline practical claim is about evaluation, not just training.** If your offline metric is computed on observed held-out ratings, it may be ordering your models backwards. This is the same disease as [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|weak-baseline offline evaluation]], but from a different direction.
- **Positivity is a hard requirement.** $P_{u,i} > 0$ for every cell. If your production system never shows some item to some user, IPS has nothing to say about that cell, and $1/\hat P$ explodes for near-zero ones.
- **The MCAR sample is the hidden cost.** The Naive Bayes route needs randomly-served ratings to get $P(Y=r)$. In production that means a small exploration bucket where you serve random items and eat the revenue hit. The logistic-regression route avoids this but buys a strong assumption: observables fully explain selection. Same tension as in [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)|logged-bandit evaluation]].
- **Authors' own list of unexplored improvements:** propensity clipping (cap $1/\hat P$ at some ceiling), [[Doubly Robust Policy Evaluation and Learning|doubly robust estimation]] (combine a direct rating model with the IPS correction to cut variance), and better propensity estimators (boosted regression). All three are standard now; none were done here.
- **The modularity is the real selling point.** Assignment model and rating model are estimated separately. Swap the propensity estimator for anything that outputs a conditional probability. Swap the MF for a neural model. This is why the idea survived — it is a loss-reweighting recipe, not an architecture.
- **Estimated propensities beating true propensities** is worth internalising. It is not a fluke; it is the Hirano–Imbens–Ridder result. Do not assume knowing $P$ exactly is optimal.
- The Coat dataset (290 users, 300 items, self-selected train + uniform test) was released with this paper and is still a standard debiasing benchmark. Yahoo! R3 likewise. Both are tiny — treat conclusions as directional.
- Open question the paper does not answer: propensities here depend only on user/item, not on **position** in a feed. Real systems have position bias on top of selection bias — see [[Recommending What Video to Watch Next- A Multitask Ranking System (RecSys)|shallow tower position debiasing]] for the industrial hack.
- The bias–variance term $\frac{\Delta}{UI}\sum|1 - P_{u,i}/\hat P_{u,i}|$ is a genuinely useful diagnostic: it says the cost of bad propensities is *linear* in the relative error, while the variance term is $\sqrt{\sum 1/\hat P^2}$. Overestimating tiny propensities trades a linear penalty for a square-root saving.

## Links

Related: [[Unbiased Learning-to-Rank with Biased Feedback]] · [[Doubly Robust Policy Evaluation and Learning]] · [[Counterfactual Reasoning and Learning Systems]] · [[Collaborative Filtering for Implicit Feedback Datasets (ICDM)]] · [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)]] · [[Recommender Systems - Evolution]] · [[Regularization]] · [[NDCG]] · [[Loss, Objectives, and Business Alignment]] · [[Uncertainty]]

New topics worth writing: Inverse Propensity Scoring, Self-Normalized Importance Sampling (SNIPS), Missing Not At Random (MNAR), Empirical Risk Minimization, Positivity / Overlap assumption in causal inference, Propensity clipping, L-BFGS, Matrix Factorization for rating prediction, Covariate shift and importance weighting, Rubin causal model / potential outcomes
