---
title: "Off-policy evaluation for slate recommendation"
authors: ["Swaminathan et al."]
year: 2016
arxiv: "1605.04812"
url: https://arxiv.org/abs/1605.04812
priority: Good-To-Read
read_on: 2026-09-25
tags: [paper, rl]
---
## The Core Idea

You logged a search engine. Each row is: a query, the **ranked list of 5 documents you showed**, and one number for how happy the user seemed (time-to-success, or NDCG, or a satisfaction score). Now someone hands you a new ranker and asks: what would that number have been if we had shipped this instead?

The standard answer is [[Off-Policy Evaluation|inverse propensity scoring]] — reweight each logged row by $\pi(\text{slate}|x) / \mu(\text{slate}|x)$, the ratio of how likely the new policy and the old policy were to show that exact list. It is unbiased and it is useless here. With 8 candidate documents and 5 slots there are $8 \cdot 7 \cdot 6 \cdot 5 \cdot 4 = 6720$ possible lists. The new ranker almost never picks the list you logged, so almost every weight is zero and the rare non-zero ones are enormous. The variance scales with the **number of slates**, which is $m^{\Omega(\ell)}$ — exponential in the list length.

The trick: stop asking about whole lists. Assume the page-level reward splits as a **sum over slots**,

$$V(x,\mathbf{s}) = \sum_{j=1}^{\ell} \phi_x(j, s_j)$$

so document $a$ in slot $j$ contributes some unknown amount $\phi_x(j,a)$, regardless of what else is on the page. Then you can build an estimator whose weights divide by **per-slot** probabilities instead of whole-slate probabilities. The weight magnitude drops from $m^{\Omega(\ell)}$ to $\mathcal{O}(\ell m)$ — in the example above, from 6720 to about 40. Sample complexity goes from $m^{\Omega(\ell)}$ to $\mathcal{O}(\ell m / \varepsilon^2)$.

The subtle and important part is what the assumption does **not** require. $\phi_x$ is allowed to be a completely different vector for every single context $x$. The method never estimates $\phi_x$ and never looks at features of $x$. It is fully agnostic to how context is represented. That is what separates it from the "direct method" — fit a model $\hat r(x,\mathbf{s})$ and average its predictions — which needs the features to actually be predictive and is biased when they are not.

> [!NOTE] Pseudoinverse (PI) estimator ^pi-estimator
> An off-policy estimator for ranked lists. It assumes the page-level reward is a sum of unobserved per-slot, per-action rewards, and then reweights logged rewards by $\mathbf{q}_{\pi,x}^T \bm{\Gamma}_{\mu,x}^{\dagger} \mathbf{1}_{\mathbf{s}}$ — a quantity built from *marginal* slot-action probabilities, not whole-slate probabilities. Unbiased under that linearity assumption, and exponentially lower variance than IPS.

> [!NOTE] Linearity (additive-decomposition) assumption ^slate-linearity
> $V(x,\mathbf{s}) = \mathbf{1}_{\mathbf{s}}^T \bm{\phi}_x$ where $\mathbf{1}_{\mathbf{s}}$ is the one-hot-per-slot indicator of the slate. No interactions between items on the page. [[NDCG]] satisfies it exactly. Diversity, novelty, and cascade-style metrics like ERR do not.

## The Methodology

**The setup.** Combinatorial [[Contextual Bandit]]. A context $x$ arrives (query + user profile). The policy picks a slate $\mathbf{s} = (s_1, \dots, s_\ell)$ — one action per slot, from $m$ candidates per slot. One scalar reward $r \in [-1,1]$ comes back for the whole page. Logged data is $n$ tuples $(x_i, \mathbf{s}_i, r_i)$ collected by a known stochastic **logging policy** $\mu$. Goal: estimate $V(\pi) = \mathbb{E}_\pi[r]$ for a new **target policy** $\pi$.

**Representing a slate as a vector.** Build $\mathbf{1}_{\mathbf{s}} \in \mathbb{R}^{\ell m}$, indexed by (slot, action) pairs. Entry $(j,a)$ is 1 if the slate put action $a$ in slot $j$, else 0. So the vector has exactly $\ell$ ones. Under the linearity assumption the reward is a **linear function of this vector**, with a context-specific weight vector $\bm{\phi}_x$.

**Recovering $\bm{\phi}_x$ as a regression.** Fix one context. The covariates $\mathbf{1}_{\mathbf{s}}$ are drawn from $\mu(\cdot|x)$, the response is $r$, and the model is linear. The minimum-norm least-squares solution is the textbook closed form:

$$\bar{\bm{\phi}}_x = \bigl(\underbrace{\mathbb{E}_\mu[\mathbf{1}_{\mathbf{s}}\mathbf{1}_{\mathbf{s}}^T|x]}_{\bm{\Gamma}_{\mu,x}}\bigr)^{\dagger} \underbrace{\mathbb{E}_\mu[r\,\mathbf{1}_{\mathbf{s}}|x]}_{\bm{\theta}_{\mu,x}}$$

$\bm{\Gamma}_{\mu,x}$ is the $\ell m \times \ell m$ second-moment matrix of slate indicators under the logging policy. It is **singular** — every valid slate has exactly one 1 per slot, so slot blocks are linearly dependent and $\mathbf{1}^T\mathbf{1}_{\mathbf{s}} = \ell$ always. Hence $\dagger$, the Moore–Penrose pseudoinverse, which is where the estimator's name comes from. That is fine: the null space of $\bm{\Gamma}$ is exactly the set of directions that no logged slate can distinguish, so the reward does not depend on them.

**The estimator.** Replace $\bm{\theta}_{\mu,x}$ with its one-sample version $\hat{\bm{\theta}}_i = r_i \mathbf{1}_{\mathbf{s}_i}$, and average the predicted slate value under $\pi$:

$$\hat V_{\text{PI}}(\pi) = \frac{1}{n}\sum_{i=1}^{n} r_i \cdot \mathbf{q}_{\pi,x_i}^T \bm{\Gamma}_{\mu,x_i}^{\dagger} \mathbf{1}_{\mathbf{s}_i}, \qquad \mathbf{q}_{\pi,x} := \mathbb{E}_\pi[\mathbf{1}_{\mathbf{s}}|x]$$

$\mathbf{q}_{\pi,x}$ is just the **marginal probability that $\pi$ puts action $a$ in slot $j$** — an $\ell m$-vector you can get in closed form or by sampling a handful of slates from $\pi$. There is no model of the reward anywhere. Everything is probabilities.

**Unbiasedness.** Under linearity plus absolute continuity ($\mu(\mathbf{s}|x) > 0$ whenever $\pi(\mathbf{s}|x) > 0$), $\mathbb{E}[\hat V_{\text{PI}}] = V(\pi)$. The proof is two lines once you show $\mathbf{1}_{\mathbf{s}}^T \bm{\Gamma}^{\dagger}\bm{\theta} = V(x,\mathbf{s})$ on the support of $\mu$.

A self-normalised variant $\hat V_{\text{wPI}}$ divides by the sum of the weights, exactly like [[The Self-Normalized Estimator for Counterfactual Learning (SNIPS) (NeurIPS)|SNIPS]] does for IPS. Asymptotically zero bias, lower variance, and it is what the experiments actually use.

**What the weights look like concretely.** Two closed forms make the whole thing click.

Cartesian product slates (one article per news section), logging policy independent across slots:

$$\hat V_{\text{PI}}(\pi) = \frac{1}{n}\sum_i r_i \left(\sum_{j=1}^\ell \frac{\pi(s_{ij}|x_i)}{\mu(s_{ij}|x_i)} - \ell + 1\right)$$

Read that carefully. IPS would divide by the probability of the **whole slate**, which is a product of $\ell$ small numbers. PI **sums** $\ell$ per-slot ratios and subtracts $\ell - 1$. Same importance-sampling spirit, but additive rather than multiplicative — that is the entire exponential saving.

Rankings with $\ell = m$ and uniform logging:

$$\hat V_{\text{PI}}(\pi) = \frac{1}{n}\sum_i r_i\left(\sum_{j=1}^\ell (m-1)\,\pi(s_{ij}|x_i) - m + 2\right)$$

**Two sanity checks.** With $\ell = 1$, PI *is* IPS. With $\pi = \mu$, PI collapses exactly to $\frac{1}{n}\sum_i r_i$ — the plain average of logged rewards, which is obviously right.

**The deviation bound.** Define

$$\sigma^2 := \mathbb{E}_{x}\bigl[\mathbf{q}_{\pi,x}^T \bm{\Gamma}_{\mu,x}^{\dagger}\mathbf{q}_{\pi,x}\bigr], \qquad \rho := \sup_{x}\sup_{\mathbf{s}: \mu(\mathbf{s}|x)>0} \bigl|\mathbf{q}_{\pi,x}^T \bm{\Gamma}_{\mu,x}^{\dagger}\mathbf{1}_{\mathbf{s}}\bigr|$$

the average and worst-case mismatch between logging and target. Bernstein gives, with probability $1-\delta$:

$$\bigl|\hat V_{\text{PI}}(\pi) - V(\pi)\bigr| \le \sqrt{\frac{2\sigma^2 \ln(2/\delta)}{n}} + \frac{2(\rho+1)\ln(2/\delta)}{3n}$$

Both $\sigma^2$ and $\rho$ equal 1 when $\pi = \mu$, so the bound is *distribution-dependent* — it tightens automatically when the two policies overlap. That is unlike the regret bounds in the combinatorial-bandit literature this borrows from, which are tied to the learner's own exploration schedule.

For $\varepsilon$-uniform logging, $\mu_\varepsilon = (1-\varepsilon)\mu + \varepsilon\,\text{Uniform}$, the bound becomes $\mathcal{O}\bigl(\sqrt{\varepsilon^{-1}\ell m / n}\bigr)$. The key lemma is $\bar\rho_{\nu,x} = \ell m - \ell + 1$ for the uniform policy — a norm on $\bm{\Gamma}_\nu^{\dagger}$, slightly sharper than Cesa-Bianchi and Lugosi's version.

**Off-policy optimization (PI-OPT).** This is the nice second act. Pointwise [[NDCG|learning-to-rank]] needs a per-document label. You do not have one — you only logged a page-level number. So **impute** the labels: compute $\hat{\bm{\phi}}_i = \bm{\Gamma}_{\mu,x_i}^{\dagger}\hat{\bm{\theta}}_i$, which turns one bandit example into $\ell m$ regression examples with targets $\hat\phi_i(j,a)$. Feature vector is $[\mathbf{f}(x,a); \mathbf{1}_j]$ — the query-document features concatenated with a one-hot position. Fit gradient-boosted trees (1000 trees, ≤70 leaves). At serving time, score every (document, slot) pair and fill the slate greedily by highest score, removing used documents and used slots. When the query set is finite you can average $\hat\phi$ across all logged impressions of a query for a lower-variance, still-unbiased target.

So: **pointwise L2R optimising a whole-page metric, with no relevance judgments.**

## Ablation Studies and Experiments

**Semi-synthetic, MSLR-WEB30K.** 31K queries, up to 1251 judged documents each, relevance in $\{0,\dots,4\}$. Features split into title and body groups. Four base rankers trained: lasso and regression tree on title features (used for logging), lasso and tree on body features (used as targets). They are genuinely different — fewer than 2.75 documents overlap in the top 10 on average, and pairwise Kendall's $\tau$ is *negative* (−0.22 to −0.42) across the title/body divide.

Logging policies sample slot-by-slot without replacement from $p_\alpha(a|x) \propto 2^{-\alpha\lfloor \log_2 \text{rank}(x,a)\rfloor}$. Sweeping $\alpha$ from 0 interpolates uniform → deterministic.

40 conditions: 2 metrics × 2 slate sizes ($(m,\ell) = (10,5)$ and $(100,10)$) × 10 logging/target combinations. Metric is RMSE vs 25+ runs.

| Finding | Detail |
|---|---|
| wPI beats wIPS | In **all 40 conditions**. At $(100,10)$, wIPS's RMSE barely improves with more data at all — it is producing noise. |
| DM wins at small $n$ | Low variance helps when data is scarce. But DM often **plateaus or degrades** as $n$ grows, because it overfits the logging distribution, which is not the target distribution. |
| wPI is the only robust one | Aggregate CDF plots of normalised RMSE at 600k samples: wPI is near-best everywhere; DM is best sometimes and much worse other times. |
| wPI tolerates near-deterministic logging | Insensitive to $\alpha$. DM, by contrast, *improves* when logging and target overlap more — which is the wrong dependence to have. |
| Weighted > unweighted | Both wIPS and wPI beat their unweighted versions. |

**What breaks the assumption.** ERR (expected reciprocal rank) is a cascade metric — $\text{ERR} = \sum_r \frac{1}{r}\prod_{i<r}(1-R(s_i))R(s_r)$ — so item $r$'s contribution depends on everything above it. Linearity is violated. wPI is visibly biased on ERR, as predicted. **It still outperformed the baselines.** That is the most practically useful negative result in the paper: the assumption buys you a favourable bias-variance trade even when it is false.

**Larger slates hurt, but gracefully.** Going from $(10,5)$ to $(100,10)$, wPI degrades somewhat but remains informative. wIPS becomes meaningless.

**Off-policy optimization, MSLR-WEB10K, top-3.**

| Metric | LambdaMART (tuned, listwise) | Uniform random | SUP (relevance labels) | PI-OPT (bandit only) |
|---|---|---|---|---|
| NDCG@3 | 0.457 | 0.152 | 0.438 | 0.421 |
| ERR@3 | — | 0.096 | 0.311 | **0.321** |

SUP used $8\times 10^5$ human relevance judgments, regressing on gains $2^{\text{rel}}-1$ (raw relevances were worse). PI-OPT used $10^7$ bandit samples under uniform logging and **never saw a relevance label**. Competitive on NDCG, better on ERR. The ERR win is the real point: SUP's regression target was hand-designed for NDCG, whereas PI-OPT derives the right pointwise target automatically from whatever page-level metric you logged.

Both trail the highly-tuned LambdaMART on NDCG@3 (1000 trees × 70 leaves listwise), which the authors are honest about.

**Real search logs.** 77 unique queries, 22K impressions, $\ell = 5$, $m \le 8$, logging policy samples non-uniformly from a pre-filtered candidate set. Two metrics that definitely violate linearity: **time-to-success** (seconds to first satisfied click, capped, scaled to $[0,1]$) and **UtilityRate** (a duration-weighted timeline of positive and negative events, in $[-1,1]$). Target policy is a logistic click model, restricted to slates present in the logs so ground truth is known exactly.

PI gives a consistent multiplicative RMSE improvement over IPS throughout, and overtakes DM (regression trees over ~20,000 slate-level features) from moderate sample sizes onward. This is, to the authors' knowledge, the first credible off-policy evaluation of real whole-page satisfaction metrics.

## Worth Remembering

**The assumption is the whole paper, and it is a *metric* assumption, not a model assumption.** That distinction matters. The direct method assumes your features predict reward. PI assumes the *metric* decomposes additively across slots. You can check the second one by inspecting the metric's formula — NDCG passes, ERR fails — whereas you can only check the first one empirically, after the fact.

**Interactions are what you give up.** Anything about the *set* rather than the *items* is invisible: diversity, redundancy, novelty, "don't show two hotels from the same chain". If your business metric rewards a well-composed page, PI will be biased in a direction you cannot easily sign. See [[Calibrated Recommendations (RecSys)]] for what that class of objective looks like.

**You must log propensities.** $\bm{\Gamma}_{\mu,x}$ and $\mathbf{q}_{\pi,x}$ both require knowing the logging distribution, and absolute continuity means $\mu$ must put non-zero mass on every slate $\pi$ might pick. In practice that means running $\varepsilon$-uniform exploration — and the bound degrades as $1/\varepsilon$, so exploration budget converts directly into estimator precision. Same lesson as [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]] and [[Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms]].

**The pseudoinverse is $\ell m \times \ell m$ per context.** For $(m,\ell) = (100,10)$ that is a $1000\times 1000$ matrix. But the closed forms in the appendices (Props 2 and 4) mean you rarely compute it: for product slate spaces with factorised logging, and for uniform logging over rankings, the weight is a scalar you can write down. Derive the closed form for your logging policy before reaching for `np.linalg.pinv`.

**Explicitly listed as missing:** a doubly-robust variant (combine PI with a reward model to cut variance at small $n$ — see [[Doubly Robust Policy Evaluation and Learning]]) and weight clipping (see [[Optimal and Adaptive Off-policy Evaluation in Contextual Bandits (SWITCH)]]). Both would plausibly close the small-sample gap where DM currently wins. The natural relaxation of the assumption is decomposing over *pairs* of slots to capture first-order interactions; the estimator still applies but the sample-complexity analysis is open.

**A practical reading.** The honest summary of Figure 2 is: at small $n$, use the direct method; at moderate-to-large $n$, use PI; never use IPS on slates. If you only ever get to build one thing, build PI, because it is the only estimator that does not fall apart in some region of the space.

**Why this matters for the reader's trajectory.** This is the ranking-shaped analogue of the move from IPS to per-component importance weights. The structural insight — *find a factorisation of the action space such that importance weights become additive rather than multiplicative* — is reusable well beyond slates, and is the same instinct behind per-step weighting in [[Doubly Robust Off-policy Value Evaluation for Reinforcement Learning|sequential DR]] and [[Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning (MAGIC)|MAGIC]].

**Open question worth chasing:** the paper restricts its real-data target policy to slates already in the logs so ground truth is available. That is a friendly setting. How PI behaves when the target proposes genuinely novel slates, and how you would even *validate* it there, is the hard production question — and it connects straight back to [[Off-Policy Evaluation#Where this shows up 🔗|validated on whose logs]].

## Links

Related: [[Off-Policy Evaluation]] · [[Contextual Bandit]] · [[Doubly Robust Policy Evaluation and Learning]] · [[The Self-Normalized Estimator for Counterfactual Learning (SNIPS) (NeurIPS)]] · [[Optimal and Adaptive Off-policy Evaluation in Contextual Bandits (SWITCH)]] · [[Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms]] · [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]] · [[Counterfactual Risk Minimization]] · [[Counterfactual Reasoning and Learning Systems]] · [[NDCG]] · [[Unbiased Learning-to-Rank with Biased Feedback]] · [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[On-Policy vs Off-Policy]] · [[AB Testing]] · [[Multi-Armed Bandit]] · [[Calibrated Recommendations (RecSys)]] · [[Recommending What Video to Watch Next- A Multitask Ranking System (RecSys)]] · [[Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning (MAGIC)]] · [[On Sampled Metrics for Item Recommendation (KDD)]] · [[Open Bandit Dataset and Pipeline- Towards Realistic and Reproducible Off-Policy Evaluation]]

New topics worth writing: Combinatorial bandits, Semi-bandit feedback, Moore–Penrose pseudoinverse, Expected Reciprocal Rank (ERR), Cascade click models, Slate/whole-page metrics, Additive decomposition assumptions in OPE, Pointwise vs pairwise vs listwise learning-to-rank, Bernstein's inequality, Minimax optimality of IPS
