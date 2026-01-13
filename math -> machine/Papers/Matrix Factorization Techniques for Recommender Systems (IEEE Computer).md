---
title: "Matrix Factorization Techniques for Recommender Systems (IEEE Computer)"
authors: ["Koren", "Bell & Volinsky"]
year: 2009
url: https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper]
---
## The Core Idea

Take the big table of "which user gave which movie how many stars". It is mostly holes — a typical user rated a tiny fraction of the catalogue. The bet here is that this table, despite the holes, is *low rank*: there are only a few dozen hidden dimensions of taste (comedy vs drama, male-oriented vs female-oriented, quirky-indie vs mainstream), and every user and every movie is just a point in that small space. Predict a rating as a dot product of the two points.

$$\hat r_{ui} = q_i^\top p_u, \qquad q_i, p_u \in \mathbb{R}^f, \ f \approx 20\text{–}200$$

The idea of a low-rank approximation was old — that is just [[Fundamentals|SVD]]. The new thing is *how to fit it when most of the matrix is missing*. Classical SVD is simply undefined on an incomplete matrix. Earlier work filled the holes in ("imputation"), which blows the data up to dense size and injects whatever bias your filling rule had. The move in this paper: **fit only the observed cells, and control overfitting with a regulariser instead of with imputed data.**

$$\min_{p_*, q_*} \sum_{(u,i) \in \kappa} (r_{ui} - q_i^\top p_u)^2 + \lambda\left(\|q_i\|^2 + \|p_u\|^2\right)$$

where $\kappa$ is only the known ratings. That one change turns an ill-posed linear algebra problem into a plain [[Regularization|regularised]] regression you can run [[Derivative#Gradient|gradient]] descent on.

The second, equally important idea: because it is now just a loss function, you can **bolt extra terms onto the prediction rule and keep the same training loop**. Biases, implicit feedback, user attributes, time-varying parameters, per-observation confidence weights — each is a few extra parameters in the same sum-of-squares. That flexibility, not the raw accuracy, is why matrix factorisation ate the field.

> [!NOTE] Latent factor model
> Explain user–item behaviour by placing both users and items in the same low-dimensional space, where closeness (dot product) means preference. The dimensions are learned from ratings, not hand-labelled like Pandora's Music Genome "genes". ^latent-factor

## The Methodology

**Baseline model.** Most of the variance in ratings has nothing to do with taste. Some users are harsh, some movies are good. Model that separately first:

$$b_{ui} = \mu + b_i + b_u$$

$\mu$ is the global average (3.7 stars on Netflix). If *Titanic* runs $+0.5$ and Joe runs $-0.3$, Joe's predicted rating is 3.9. Only the leftover goes to the factors:

$$\hat r_{ui} = \mu + b_i + b_u + q_i^\top p_u$$

$$\min_{p_*,q_*,b_*}\sum_{(u,i)\in\kappa}(r_{ui}-\mu-b_u-b_i-p_u^\top q_i)^2 + \lambda(\|p_u\|^2+\|q_i\|^2+b_u^2+b_i^2)$$

**Training, option A — SGD (Simon Funk's method).** Loop over ratings one at a time. Compute $e_{ui} = r_{ui} - q_i^\top p_u$, then

- $q_i \leftarrow q_i + \gamma\,(e_{ui}\,p_u - \lambda\,q_i)$
- $p_u \leftarrow p_u + \gamma\,(e_{ui}\,q_i - \lambda\,p_u)$

Simple, fast, one pass touches only observed cells. Note this is exact [[Backpropagation|gradient]] descent on a two-factor bilinear model — no chain rule depth needed.

**Training, option B — Alternating Least Squares.** The objective is *not* convex in $(p, q)$ jointly, because they multiply each other. But fix all $p_u$ and it becomes a quadratic in $q$ with a closed-form solution, and vice versa. Alternate. Each step provably decreases the loss. ALS wins in two situations:

1. **Parallelism.** With $p$ fixed, every $q_i$ solves independently — embarrassingly parallel across machines.
2. **Implicit feedback.** There the "training set" is every user–item pair (absence is a signal), so it is dense and looping one-by-one is hopeless. ALS handles it in closed form — this is exactly the setup of [[Collaborative Filtering for Implicit Feedback Datasets (ICDM)]].

**Extension 1 — extra input sources (helps cold start).** Let $N(u)$ be items user $u$ implicitly touched (browsed, bought), and $A(u)$ their known attributes (age, zip, gender). Give each item a *second* vector $x_i$ and each attribute a vector $y_a$, and enrich the user side:

$$\hat r_{ui} = \mu + b_i + b_u + q_i^\top\!\left[p_u + |N(u)|^{-0.5}\!\!\sum_{i\in N(u)}\!\! x_i + \!\!\sum_{a\in A(u)}\!\! y_a\right]$$

The $|N(u)|^{-0.5}$ normalisation matters — without it, heavy users' profiles have huge norm and dominate.

**Extension 2 — temporal dynamics.** Item popularity drifts, users' rating scales drift, and tastes drift.

$$\hat r_{ui}(t) = \mu + b_i(t) + b_u(t) + q_i^\top p_u(t)$$

Item factors $q_i$ stay **static** — a movie does not change; people's perception of it does, and that is absorbed by $b_i(t)$ and $p_u(t)$.

**Extension 3 — confidence weights.** Not all observations deserve equal weight (ad campaigns, adversarial raters, implicit counts like watch-time). Attach $c_{ui}$:

$$\min \sum_{(u,i)\in\kappa} c_{ui}\,(r_{ui}-\mu-b_u-b_i-p_u^\top q_i)^2 + \lambda(\cdots)$$

$\lambda$ is picked by cross-validation. Salakhutdinov & Mnih's Probabilistic Matrix Factorization gives the Bayesian justification for that L2 term (it is a Gaussian prior on the factors).

## Ablation Studies and Experiments

**Data.** Netflix Prize: >100 million ratings, ~500,000 users, >17,000 movies, 1–5 stars. Test set ~3 million ratings, scored by RMSE. Netflix's own system (Cinematch) sat at **RMSE = 0.9514**; the $1M grand prize needed **0.8563** (a 10% cut). The BellKor team hit **8.43% better** (2007 Progress Prize) and, merged with Big Chaos, **9.46%** (2008).

**The model-complexity ladder (Figure 4).** This is the real ablation — five variants swept over dimensionality $f$, plotted against parameter count:

| Variant | $f$ range swept | Best RMSE (approx.) |
|---|---|---|
| Plain factorisation | 40 → 180 | ~0.905 → 0.902 |
| + biases | 50 → 200 | ~0.895 |
| + implicit feedback | 50 → 200 | ~0.890 |
| + temporal dynamics (v.1) | 100 → 500 | ~0.883 |
| + temporal dynamics (v.2) | 50 → 1,500 | ~0.876 |

Two clean readings. **(a)** More factors always helps, monotonically, within each variant — no overfitting cliff, because $\lambda$ is holding the line. **(b)** More *kinds* of parameter helps more than more of the *same* parameter. Plain factorisation at 180 dims is beaten by a bias model at 50 dims. The authors call out temporal effects as "particularly important" — the biggest single jump on the ladder.

Note what this implies: **plain matrix factorisation alone never got close to the prize.** The gain came from the bias/implicit/time scaffolding around it.

**What did not work.**

- **Imputation.** Filling missing ratings to make the matrix dense is expensive (data explodes) and distorting (a wrong fill rule bakes in wrong signal). Abandoned in favour of fitting observed entries only.
- **Naive SVD.** Mathematically undefined with missing values, and if you naively fit only the few known entries with no regularisation, it overfits badly.
- **Explaining everything with $q_i^\top p_u$.** Called "unwise" — you waste factor capacity re-learning "some people rate high, some movies are good".
- **Nearest-neighbour / CF methods** were the prior state of the art and were beaten on both accuracy and memory footprint.

**Qualitative check (Figure 3).** The first two learned factors on Netflix are readable. Axis 1: *Half Baked*, *Freddy vs. Jason*, *Road Trip* on one end; *Sophie's Choice*, *Moonstruck*, *The Way We Were* on the other. Axis 2: *Punch-Drunk Love*, *I Heart Huckabees* (quirky indie) at the top; *Armageddon*, *Runaway Bride* (formulaic) at the bottom. *The Wizard of Oz* sits at the origin — it appeals to everyone. Two-dimensional slices lie though: *Annie Hall* and *Citizen Kane* land next to each other (both "prestigious classic by a famous director") and only the third factor separates them.

## Worth Remembering

- **The final Netflix winner was an ensemble of >100 predictors**, mostly factorisation variants. This paper is the readable summary of a leaderboard grind, not a single clean result. Treat the Figure 4 numbers as "what one model family gets", not "what wins".
- **Cold start is the admitted weakness.** Collaborative filtering knows nothing about a brand-new user or item. The $\sum x_i$ and $\sum y_a$ terms are a patch, not a cure; content filtering genuinely beats CF here.
- **Non-convexity is real but survivable.** SGD converges to a good local minimum in practice; ALS gives monotone decrease but only to a local optimum too. Initialisation and $\lambda$ matter more than people admit.
- **Practical caveat on the Netflix result:** RMSE on star ratings is a poor proxy for what a recommender does. Netflix famously never shipped the winning ensemble. Modern practice moved to ranking losses ([[BPR- Bayesian Personalized Ranking from Implicit Feedback]]) and implicit signals, not squared error on explicit stars. Also see [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] on how easily this literature fools itself.
- **Direct descendants.** [[Factorization Machines (ICDM)]] generalises the dot-product-of-embeddings idea to arbitrary feature pairs. The two-tower retrieval models in [[Deep Neural Networks for YouTube Recommendations (RecSys)]] are matrix factorisation with the lookup tables replaced by neural encoders. Sequential models like [[Self-Attentive Sequential Recommendation (SASRec)]] and [[BERT4Rec- Sequential Recommendation with Bidirectional Transformer]] replace the static $p_u$ with something computed from the user's history — which is the logical end point of the "$p_u(t)$" temporal idea here.
- **The bias term is the free lunch.** In almost any recommender or CTR system, fitting $\mu + b_u + b_i$ first and modelling only the residual is a cheap, large win. Same instinct as centring features before regression.
- **Open question worth chasing:** the confidence-weighted loss (Eq. 8) is the seed of everything in debiased learning-to-rank — see [[Recommendations as Treatments- Debiasing Learning and Evaluation]] and [[Unbiased Learning-to-Rank with Biased Feedback]], where $c_{ui}$ becomes an inverse propensity score rather than a heuristic count.

## Links

Related: [[Collaborative Filtering for Implicit Feedback Datasets (ICDM)]] · [[Factorization Machines (ICDM)]] · [[BPR- Bayesian Personalized Ranking from Implicit Feedback]] · [[Recommender Systems - Evolution]] · [[Regularization]] · [[Derivative#Gradient|gradient]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Self-Attentive Sequential Recommendation (SASRec)]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[Real-time Personalization using Embeddings for Search Ranking at Airbnb (KDD, best paper)]] · [[Loss, Objectives, and Business Alignment]] · [[Fundamentals]]

New topics worth writing: Singular Value Decomposition, Alternating Least Squares, Probabilistic Matrix Factorization, Low-rank matrix completion, Cold-start problem, RMSE vs ranking metrics, Netflix Prize history, Ensembling / blending for competitions
