---
title: "Unbiased Learning-to-Rank with Biased Feedback"
authors: ["Thorsten Joachims", "Adith Swaminathan", "Tobias Schnabel"]
year: 2016
arxiv: "1608.04468"
url: https://arxiv.org/abs/1608.04468
priority: Must-Read
read_on: 2026-08-25
tags: [paper]
---
## The Core Idea

Clicks are cheap but lying. A result at rank 1 gets clicked far more than the same result at rank 10, simply because people look at the top of the page. So if you train a ranker to "put clicked documents high", you mostly teach it to reproduce whatever the old ranker already did.

The trick here: stop trying to *decode* clicks into relevance labels, and instead treat the click log as a **biased sample** of a full label set, then fix the sample with importance weights. Each click is divided by the probability that the user would have *looked at* that position at all. A click at rank 10, where the look-probability is $0.1$, counts ten times as much as a click at rank 1.

This is inverse propensity scoring (IPS) borrowed from causal inference, applied per click rather than per query.

> [!NOTE] Propensity (here)
> The marginal probability that the relevance of result $y$ was *revealed* to you, given the ranking you actually showed. Under the position-based model it is just $p_r$, the probability a user examines rank $r$. ^propensity

Why this did not exist before: earlier work used click models (cascade, DBN) to *estimate a relevance score* $\mathrm{rel}(x,y)$ per query-document pair. That needs the same query to appear many times, so you can average out noise. Dead on arrival for tail queries, personal search, intranet search. Here the click model is demoted to a *weighting device* applied in hindsight. Each click is its own training example. Queries never need to repeat.

The other unlock: no randomisation during serving. Bandit-style learning-to-rank (LTR) needs you to shuffle results in production, which costs revenue and user trust. Here randomisation is needed only for a small pilot to fit the $p_r$ vector; the actual training log is ordinary observational traffic from a deterministic ranker.

## The Methodology

**The loss.** Binary relevance $r_i(y) \in \{0,1\}$. The loss of a ranking is the sum of ranks of relevant documents (lower is better):

$$\Delta(\bm{y}\mid \bm{x}_i, r_i) = \sum_{y \in \bm{y}} \mathrm{rank}(y \mid \bm{y}) \cdot r_i(y)$$

Risk is this averaged over the query distribution. You cannot compute it, because you only see clicks.

**The estimator.** Let $o_i(y) \in \{0,1\}$ mark whether the true relevance of $y$ was revealed, with propensity $Q(o_i(y)=1 \mid \bm{x}_i, \bar{\bm{y}}_i, r_i)$, where $\bar{\bm{y}}_i$ is the ranking that was actually shown. Then

$$\hat{\Delta}_{IPS}(\bm{y}\mid \bm{x}_i,\bar{\bm{y}}_i,o_i) = \sum_{\substack{y:\, o_i(y)=1 \\ \wedge\; r_i(y)=1}} \frac{\mathrm{rank}(y\mid \bm{y})}{Q(o_i(y)=1\mid \bm{x}_i,\bar{\bm{y}}_i,r_i)}$$

The proof of unbiasedness is one line of expectation algebra: $\mathbb{E}[o_i(y)] = Q(\cdot)$, so the $Q$ cancels. It needs $Q > 0$ for every *relevant* document — irrelevant ones may have zero propensity and nothing breaks.

The beautiful practical consequence: only documents that were both observed **and** relevant contribute. Under a noise-free click model that set is exactly "the clicked documents". You never need negative labels, and you never need to know whether an unclicked result was examined.

**Position-based propensity model.** Click = examine, then decide. Examination depends only on rank, so it is a vector $p_1, \dots, p_R$. With noise-free clicking, examination *is* observation, and the empirical risk collapses to something you can compute from a log:

$$\hat{R}_{IPS}(S) = \frac{1}{n}\sum_{i=1}^{n} \sum_{y:\, c_i(y)=1} \frac{\mathrm{rank}(y \mid S(\bm{x}_i))}{p_{\mathrm{rank}(y \mid \bar{\bm{y}}_i)}}$$

**Click noise.** Real users click relevant things with probability $\epsilon_+ < 1$ and irrelevant things with probability $\epsilon_- > 0$. This makes $\hat{R}_{IPS}$ biased for the true risk. But the authors show the bias is *order preserving*: comparing two systems, the $\epsilon_-$ term cancels because $\sum_y \delta\mathrm{rank}(y\mid \bm{x}) = 0$ (a permutation of the same candidate set has the same total rank sum), and what survives is $(\epsilon_+-\epsilon_-)$ times the true risk difference. Since $\epsilon_+ > \epsilon_-$, the argmin is unchanged. Noise costs you variance, not correctness.

**Estimating $p_r$ without wrecking the product.** You only need the $p_r$ up to a positive constant, since scaling all propensities does not change the argmin. So: pick a landmark rank $k$, and before serving, swap the document at rank $k$ with the document at rank $r$. The click-through rate of *that same document* at rank $r$ versus rank $k$ gives $p_r/p_k$ directly, because the "would click if examined" term is a property of the document and cancels. Much gentler than uniform shuffling.

**Propensity SVM-Rank.** Take SVM-Rank and weight each clicked example's slack by $1/q_j$:

$$\hat{w} = \arg\min_{w,\xi} \frac{1}{2} w\cdot w + \frac{C}{n}\sum_{j=1}^{n} \frac{1}{q_j}\sum_{y \in Y_j} \xi_{jy}$$
$$\text{s.t. } \forall j,\ \forall y \in Y_j\setminus\{y_j\}:\ w\cdot[\phi(\bm{x}_j,y_j) - \phi(\bm{x}_j,y)] \geq 1 - \xi_{jy},\quad \xi_{jy}\geq 0$$

$y_j$ is the clicked document, $Y_j$ the candidate set (a few hundred documents from a stage-one retriever). Each click is a separate training instance. The hinge sum upper-bounds $\mathrm{rank}(y_j) - 1$, so the objective is a convex upper bound on the IPS risk plus [[Regularization|L2 regularisation]]. Solved with the one-slack cutting-plane formulation, so it stays linear-time.

**Naive SVM-Rank** = the exact same program with all $q_j = 1$. That is the whole baseline, and it makes the comparison clean.

## Ablation Studies and Experiments

**Synthetic (Yahoo LTR Challenge set 1).** Relevance binarised: ratings 3–4 → 1, ratings 0–2 → 0. A Ranking SVM trained on 1% of the labelled data plays the "production ranker" $S_0$ that generates the displayed rankings. Presentation bias is simulated as $p_r = (1/r)^\eta$; defaults $\eta=1$, $\epsilon_+=1$, $\epsilon_-=0.1$, which makes about 33% of clicks land on irrelevant results and gives rank 10 a 10% examination chance.

- **Learning curves.** Propensity SVM-Rank climbs toward the *skyline* (a Ranking SVM trained on all clean full-information labels) as clicks accumulate. Naive SVM-Rank's curve is **flat** — more data does not help it at all. That flatness is the whole story: naive error is dominated by asymptotic bias, which no amount of data removes.
- **What did not work at small $n$.** Propensity SVM-Rank is *worse* than naive on small datasets. The IPS weights $1/p_r$ blow up variance. Fix: propensity clipping, $\max\{\tau, Q(\cdot)\}$ in the denominator, with $\tau$ cross-validated. Clipping helps at small $n$. Note the honest admission: $\tau=1$ *is* Naive SVM-Rank, and the validation set is too small and noisy to reliably pick it every run.
- **Bias severity ($\eta$ sweep, $\epsilon_-=0$).** IPS wins whenever bias is substantial. Going from $n=45K$ to $n=225K$ improves Propensity SVM-Rank and does nothing for naive.
- **Noise sweep ($\epsilon_-$ from 0 to 0.3, i.e. up to 59.8% of clicks on irrelevant documents, $n=170K$ and $850K$).** The gap *widens* with noise. Confirms the order-preservation argument.
- **Misspecified propensities.** Data generated with true $\eta=1$, model trained with a wrong $\eta$. Asymmetric result, and this is the most useful practical finding: **$\eta < 1$ (overestimating small propensities) is safe and still beats naive; $\eta > 1$ (underestimating small propensities) hurts.** Underestimating $p_r$ means a deep click gets a giant weight and the estimator explodes. Clipping is exactly a principled form of "overestimate small propensities". And naive is just the extreme misspecification $\eta=0$.

**Real search engine (scientific article search).** 1000-dimensional $\phi(\bm{x},y)$, hand-tuned production weight vector (Prod). 21 days of logs. Propensities estimated from 7 days of swap interventions with landmark $k=1$, swapping the top document to a uniformly random rank $j \in \{1,\dots,21\}$, then smoothed by interpolating with the raw observed CTR@$j$; smallest $p_r \approx 0.12$, and $p_r = p_{21}$ imputed for $r > 21$. Training data is tiny: **5437 clicks** (16 days) train, 1755 clicks (5 days) validation.

Balanced interleaving, per-query:

| Comparison | wins | loses | ties | sign test |
|---|---|---|---|---|
| Propensity SVM-Rank vs Prod | 87 | 48 | 83 | $p=0.001$, RR $0.71$ |
| Propensity SVM-Rank vs Naive SVM-Rank | 95 | 60 | 102 | $p=0.006$, RR $0.77$ |

Five thousand clicks and a crude one-parameter-per-rank propensity model beat a hand-tuned production ranker.

## Worth Remembering

- **The conceptual move worth stealing.** A click model can play two very different roles: (a) infer relevance, which needs repeated queries; (b) assign a weight in hindsight, which does not. Role (b) is far weaker and far more usable. You only need propensities for the documents that were clicked — never for the ones that were not — which is much easier than building a full generative model of user behaviour.
- **Unbiasedness only needs $Q>0$ on relevant documents.** Irrelevant results with zero examination probability are harmless. This is a genuinely loose requirement.
- **Unconfoundedness is the hidden assumption.** Propensities must depend only on observable things ($\bm{x}_i$, $\bar{\bm{y}}_i$). If some unlogged factor drives both examination and relevance, the estimator is no longer unbiased. Standard causal caveat, same as in [[Counterfactual Reasoning and Learning Systems]].
- **Bias–variance, again.** Naive = low variance, high bias, flat learning curve. IPS = zero bias, high variance, improves forever. Clipping is the dial between them. In production, where clicks are plentiful, the paper argues you want the IPS end.
- **The position-only examination model is very simple.** No cascade effect, no trust bias (users believing top results are better *and clicking them more for that reason*), no saliency bias from thumbnails or bold text. The authors say trust bias was worked out but cut for space, and suggest replacing scalar $p_r$ with a regression function for saliency. In modern systems you would model $p$ as a function of position, device, layout, and above-fold status.
- **Direct ancestor of everything called "unbiased LTR".** The two-tower `logQ` corrections in [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)]] and [[PinnerFormer- Sequence Modeling for User Representation at Pinterest]] are the same reweighting instinct applied to sampling bias rather than position bias. [[Entire Space Multi-Task Model (ESMM)]] attacks the closely related sample-selection problem with a modelling trick instead of a weighting trick.
- **Practical caveat for reproduction.** The estimator is scale-free in $p_r$, so you never need absolute examination probabilities — only ratios. That makes the swap experiment cheap and the whole pipeline much less fragile than it looks.
- **Open follow-ups the authors flag.** Extend propensity ERM to pointwise and listwise LTR (this is pairwise only); use propensity weighting to remove *pooling bias* in editorially judged test collections; make click-DCG correlate better with true DCG.
- **Question to hold onto.** The loss here is sum-of-ranks, not [[NDCG]]. Sum-of-ranks is linear in the per-document contribution, which is exactly why the noise-cancellation proof works ($\sum_y \delta\mathrm{rank} = 0$). A discounted metric breaks that linearity — how much of the theory survives?

## Links
Related: [[Counterfactual Reasoning and Learning Systems]] · [[BPR- Bayesian Personalized Ranking from Implicit Feedback]] · [[Entire Space Multi-Task Model (ESMM)]] · [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)]] · [[PinnerFormer- Sequence Modeling for User Representation at Pinterest]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[NDCG]] · [[Regularization]] · [[Recommender Systems - Evolution]] · [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)]] · [[Uncertainty]]

New topics worth writing: Inverse propensity scoring, Empirical risk minimisation and consistency, Ranking SVM and the one-slack cutting-plane solver, Position bias and examination click models, Cascade click model, Interleaving evaluation (balanced and team-draft), Propensity clipping and self-normalised IPS, Batch learning from bandit feedback, Missing-not-at-random data, Unconfoundedness
