---
title: "Ad Click Prediction: a View from the Trenches (KDD)"
authors: ["McMahan et al."]
year: 2013
url: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, optimization]
---
## The Core Idea

Google serves billions of ad predictions a day. The model is a plain [[Regression Analysis|logistic regression]] with billions of coefficients. Two things dominate: how much RAM the model takes when it is copied to every serving datacentre, and how accurate it is. This paper is the field report on getting both.

The central algorithmic contribution is **FTRL-Proximal**, an online update that behaves *exactly* like online gradient descent when you turn regularisation off, but which stores the model in a different coordinate system so that $L_1$ regularisation actually drives weights to **exact zero**. That mattered because of a real dilemma at the time:

- Online gradient descent (OGD) with an $L_1$ subgradient added on: good accuracy, but almost never produces exact zeros, so no sparsity, so no memory saving.
- Regularized Dual Averaging (RDA): great sparsity, worse accuracy on Google's data.

FTRL-Proximal gets RDA's sparsity *and* OGD's accuracy. That is the whole trick, and it is why this algorithm was the default CTR optimiser in industry for about a decade.

The second contribution is the rest of the paper, which is arguably more valuable: memory tricks, evaluation methodology, a cheap confidence score, calibration, feature governance, and an honest list of things that failed. It is the ancestor of [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)|Hidden Technical Debt]] (same lab, overlapping authors) — the argument that the learning algorithm is a small part of a production ML system.

> [!NOTE] FTRL-Proximal
> "Follow The (Proximally) Regularized Leader". Instead of storing the weight $w$, store a running quantity $z$ that accumulates gradients. The weight is recomputed lazily from $z$ at prediction time by a closed-form soft-threshold, which is where the exact zeros come from. ^ftrl-proximal

## The Methodology

**The setup.** On round $t$ you see a sparse feature vector $x_t \in \mathbb{R}^d$ ($d$ in the billions, but only hundreds of non-zeros). Predict $p_t = \sigma(w_t \cdot x_t)$ with $\sigma(a) = 1/(1+e^{-a})$. Observe $y_t \in \{0,1\}$. Suffer logistic loss (this is [[Cross Entropy]] for two classes):

$$\ell_t(w) = -y_t\log p_t - (1-y_t)\log(1-p_t)$$

The [[Derivative#Gradient|gradient]] is beautifully simple:

$$\nabla \ell_t(w) = (p_t - y_t)\,x_t$$

One pass over the data, streamed from disk. Training is Downpour-SGD-style asynchronous, but with a single-layer model instead of a deep net — which is exactly why they can go to billions of coefficients.

**The FTRL objective.** OGD does $w_{t+1} = w_t - \eta_t g_t$. FTRL-Proximal instead solves

$$w_{t+1} = \arg\min_w \left( g_{1:t}\cdot w + \tfrac{1}{2}\sum_{s=1}^{t}\sigma_s\|w - w_s\|_2^2 + \lambda_1\|w\|_1 \right)$$

where $g_{1:t}=\sum_{s=1}^t g_s$ and $\sigma_s$ is defined so that $\sigma_{1:t} = 1/\eta_t$. With $\lambda_1=0$ these two updates produce **identical** weight sequences. With $\lambda_1>0$ the FTRL form induces real sparsity.

**Why it is cheap.** Rewrite the argmin and everything collapses into one stored number per coordinate:

$$z_t = z_{t-1} + g_t + \left(\tfrac{1}{\eta_t} - \tfrac{1}{\eta_{t-1}}\right)w_t$$
$$w_{t+1,i} = \begin{cases} 0 & \text{if } |z_{t,i}| \le \lambda_1 \\ -\eta_t\,(z_{t,i} - \operatorname{sgn}(z_{t,i})\lambda_1) & \text{otherwise}\end{cases}$$

So FTRL stores $z \in \mathbb{R}^d$ where OGD stores $w \in \mathbb{R}^d$. Same memory, exact zeros for free.

**Per-coordinate learning rates.** The other half of the accuracy. Standard theory says use one global $\eta_t = 1/\sqrt{t}$. That is wrong here: think of ten independent coins, each with its own indicator feature. A global schedule shrinks the step size for coin $i$ even on rounds where coin $i$ was never flipped. Instead:

$$\eta_{t,i} = \frac{\alpha}{\beta + \sqrt{\sum_{s=1}^{t} g_{s,i}^2}}$$

This is the same $\sum g^2$ accumulator that [[Adam- A Method for Stochastic Optimization|Adam]] and AdaGrad use, arrived at from an online-regret angle. $\beta=1$ is almost always fine (it just stops early steps being huge); $\alpha$ needs tuning per dataset. Streeter & McMahan show a family of problems where global-rate OGD has regret $\Omega(T^{2/3})$ while independent per-coordinate copies get $O(T^{1/2})$.

**Memory trick 1 — probabilistic feature inclusion.** In some models *half the unique features appear exactly once* across billions of examples. Tracking them is waste. But you cannot pre-filter in an online setting without an extra read/write pass. So admit features probabilistically on first sight:
- *Poisson inclusion*: add a new feature with probability $p$; expected sightings before admission is $1/p$.
- *Bloom filter inclusion*: rolling counting Bloom filters detect the first $n$ occurrences; admit after $n$. False positives (admitting too early) happen, false negatives do not.

**Memory trick 2 — q2.13 fixed point.** Nearly all coefficients live in $(-2, +2)$. So use 16 bits: 1 sign, 2 integer, 13 fractional. Naive truncation biases the accumulated sum, so round randomly:

$$w_{\text{rounded}} = 2^{-13}\left\lfloor 2^{13}w + R \right\rfloor,\quad R \sim \mathrm{Unif}[0,1)$$

The point is that discretisation error now has **zero mean**. Same idea as stochastic rounding in [[Mixed Precision Training]]. Values outside $[-4,4)$ are clipped. For FTRL they store $\eta_t z_t$, which has similar magnitude to $w_t$.

**Memory trick 3 — training many variants together.** One shared hash table across all model variants, so the key (a long string or hash) is stored once. Variants that lack a feature store coefficient 0 and learning rate 0.

**Memory trick 4 — the Single Value Structure.** Even more aggressive: store *one* coefficient per coordinate shared by all variants, plus a bit-field saying which variants use it. Each variant computes its own desired new value; the values are **averaged** and written back. Lossy and completely ad hoc, but variant *rankings* came out nearly identical to exact training, at an order of magnitude less RAM.

**Memory trick 5 — replace $\sum g^2$ with counts.** Assume every event containing feature $i$ has the same click probability $p = P/(N+P)$ and the model has learned it. Then

$$\sum g_{t,i}^2 \approx P\left(1 - \tfrac{P}{N+P}\right)^2 + N\left(\tfrac{P}{N+P}\right)^2 = \frac{PN}{N+P}$$

So store two small integers ($N$ negatives, $P$ positives) instead of a float. The paper calls this a "ruthless approximation" and admits the premise is "a terrible approximation" — but it works, and $N, P$ are shared across all variants, so the cost amortises.

**Data subsampling.** Clicks are rare, so keep every query with at least one click, and only a fraction $r$ of click-free queries. Correct the bias with importance weights $\omega_t = 1$ for clicked queries and $\omega_t = 1/r$ otherwise. The weight scales the loss and hence the gradient, and since $s_t = 1/\omega_t$ is the sampling probability, $\mathbb{E}[\ell_t] = s_t \omega_t \ell_t = \ell_t$ — the subsampled weighted objective is unbiased for the full one. Sampling is done at *query* level, not impression level, because query-level features are computed once per query.

**Progressive validation.** Do not hold out data. Because computing a gradient needs a prediction anyway, log every prediction *before* training on that example, and aggregate hourly. This is exactly the serving situation (predict on fresh data, then learn from it), uses 100% of data for both train and test, and has better statistics than any holdout.

**Uncertainty score.** They want explore/exploit signal (see [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)|LinUCB]]) but proper confidence intervals need inverting an $n\times n$ matrix with $n$ in the billions — a non-starter, and the model is online, regularised, and non-IID so "converged batch model" theory does not apply. Instead: notice the learning rate itself already encodes confidence. Since $|p_t - y_t| \le 1$ and assuming $|x_{t,i}| \le 1$,

$$|x\cdot w_t - x \cdot w_{t+1}| \le \alpha \sum_{i:|x_i|>0}\frac{x_{t,i}}{\sqrt{n_{t,i}}} = \alpha\,\boldsymbol{\eta}\cdot x \equiv u(x)$$

The uncertainty score is that bound. One sparse dot product — same cost as the prediction itself.

**Calibration.** Fit a monotone correction $\tau_d(p)$ per data slice $d$. Simple version: $\tau(p) = \gamma p^\kappa$, fit by Poisson regression on aggregated data. Better version: piecewise-linear isotonic regression (weighted least squares subject to monotonicity), which fixed bias at both the very high and very low ends of the prediction range.

**Feature governance.** A central metadata index of thousands of "signals" (raw semantic sources like *words in the ad*, *country of origin*) consumed by hundreds of models across teams and platforms. Signals are annotated for deprecation, platform availability, domain applicability. Automatic alerts vet new consumption; whitelists gate production; unconsumed signals are auto-earmarked for code and data deletion.

## Ablation Studies and Experiments

Metric is **AucLoss** $= 1 - \text{AUC}$, and they always report *relative* change against a baseline, never absolute. Reason: absolute LogLoss depends on the Bayes risk of the slice — a 2% CTR slice has a much lower achievable LogLoss than a 50% one, and CTR varies by country and query and time of day. A 1% AucLoss reduction is considered large.

**Sparsity vs accuracy (Table 1, FTRL-Proximal = baseline):**

| Method | Non-zeros | AucLoss detriment |
|---|---|---|
| FTRL-Proximal | baseline | baseline |
| RDA | +3% | 0.6% |
| FOBOS | +38% | 0.0% |
| OGD-Count | +216% | 0.0% |

OGD-Count is the obvious straw man: count feature occurrences, hold the coefficient at 0 until the count passes $k$, then run plain OGD. $k$ was tuned to match FTRL's accuracy. It needs **3.16× as many non-zero coefficients**. The paper's line: "In many instances, a simple heuristic works almost as well as the more principled approach, but this is not one of those cases."

**Per-coordinate learning rates:** −11.2% AucLoss versus a tuned global rate. In a domain where 1% is large, this is the single biggest accuracy win in the paper.

**Probabilistic feature inclusion (Table 2):**

| Method | RAM saved | AucLoss detriment |
|---|---|---|
| Bloom ($n=2$) | 66% | 0.008% |
| Bloom ($n=1$) | 55% | 0.003% |
| Poisson ($p=0.03$) | 60% | 0.020% |
| Poisson ($p=0.1$) | 40% | 0.006% |

Bloom filters dominate: at 66% saving, Bloom loses less than Poisson does at 60%.

**q2.13:** no measurable loss versus 64-bit floats, 75% of coefficient RAM saved.

**Uncertainty score validation:** they could not use real labels (you never observe the true CTR). So they trained a "ground truth" model on slightly different features, threw away the real clicks, and resampled labels from its predictions. Then ran FTRL on the relabelled data and plotted log-odds error $|\sigma^{-1}(p_t) - \sigma^{-1}(p_t^*)|$ against $u(x_t)$. Strong correlation, and performance **comparable to a bootstrap of 32 models** trained on random subsamples — at a tiny fraction of the cost.

### What did not work

Four negative results, reported in a dedicated section. This is the most cited part of the paper for practitioners.

1. **Aggressive feature hashing.** Weinberger et al. hashed personalised spam filtering down to $2^{24}$ features; Chapelle did the same for display ads. Google could not go below **several billion** features without observable loss. So hashing bought them nothing, and they kept interpretable non-hashed feature vectors instead.

2. **[[Dropout- A Simple Way to Prevent Overfitting|Dropout]].** Tried rates 0.1 to 0.5, each with a grid search over learning rate and number of passes. Never helped; usually hurt. Their diagnosis is the interesting bit: dropout works in vision because features are **dense and correlated**, so dropping some forces the classifier to stop relying on any single one. Here features are sparse and labels are noisy, so dropout just deletes data you did not have much of to begin with.

3. **Feature bagging.** Train $k$ models on overlapping feature subsets, average the outputs — the [[XGBoost- A Scalable Tree Boosting System|tree-ensemble]] intuition. Hurt by 0.1%–0.6% AucLoss depending on the scheme. They had also hoped it would parallelise training.

4. **Feature vector normalisation.** Number of non-zeros per event varies a lot, so $\|x\|$ varies a lot; they worried this slowed convergence. Trained on $x/\|x\|$ under several norms. Early results showed small gains that never translated into overall positive metrics; net effect looked slightly detrimental. Suspected cause: interaction with the per-coordinate learning rates and the regularisation.

## Worth Remembering

- **Sparsify for serving, not for training.** Methods that stop tracking statistics for zero-coefficient features (truncated gradient) cost unacceptable accuracy. FTRL tracks many features during training and only zeroes them out for the serving copy. The asymmetry is the point — models are replicated to many datacentres, so serving memory is what costs money.

- **The training/serving accuracy relationship is the whole design constraint.** Sparsity is not about generalisation here, it is about RAM.

- **Aggregate metrics lie.** A small aggregate win can be a mix of gains in one country and losses in another. They built **GridViz**, an interactive grid where rows are models and columns are data slices (country, query topic, match type, page layout), column width encodes impression volume, and cell colour encodes metric change versus control. Hundreds of possible slicings, selectable by dropdown or regex.

- **Calibration has no theoretical guarantee here.** The system is a feedback loop — the model's predictions determine which ads are shown, which determines the next batch of training data. Without strong extra assumptions you cannot prove calibration helps. They do it anyway because it lets the auction be designed independently of the ML.

- **Limitations they are honest about.** The $N,P$ count approximation for $\sum g^2$ rests on a premise they themselves call terrible. The Single Value Structure is "lossy and ad hoc". The uncertainty score is a heuristic upper bound, not a confidence interval.

- **Context in 2013.** This is a *linear* model with hand-crafted sparse features. [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)|Facebook's GBDT + LR]] paper is the same year and same problem; [[Wide & Deep Learning for Recommender Systems|Wide & Deep]] (2016) and [[Deep Interest Network for CTR Prediction (DIN)|DIN]] (2018) are what replaced the linear part. But the FTRL update, the per-coordinate rate, progressive validation, and importance-weighted negative subsampling all survived into deep CTR systems.

- **Open question.** The negative dropout result is about *sparse, noisy* inputs. Does it still hold once the sparse features go through an embedding table and the downstream layers are dense? Modern CTR nets do use dropout in the MLP head, which suggests the answer is "the diagnosis was right, and it was a statement about the input layer".

## Links
Related: [[Adam- A Method for Stochastic Optimization]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Regularization]] · [[Cross Entropy]] · [[Dropout- A Simple Way to Prevent Overfitting]] · [[Mixed Precision Training]] · [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Calibrated Recommendations (RecSys)]] · [[Uncertainty]] · [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Wide & Deep Learning for Recommender Systems]] · [[Recommender Systems - Evolution]]

New topics worth writing: FTRL-Proximal and Regularized Dual Averaging, AdaGrad and online regret bounds, Isotonic regression for probability calibration, Counting Bloom filters, The hashing trick for feature spaces, Progressive validation, Importance weighting for biased subsampling, Stochastic rounding and fixed-point q-formats
