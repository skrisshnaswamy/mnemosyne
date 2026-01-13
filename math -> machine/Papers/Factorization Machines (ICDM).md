---
title: "Factorization Machines (ICDM)"
authors: ["Rendle"]
year: 2010
url: https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf
priority: Must-Read
read_on: 2026-08-25
tags: [paper]
---
## The Core Idea

Take a plain feature vector $x \in \mathbb{R}^n$ — the kind you feed a linear model. A degree-2 polynomial model would add a weight $w_{i,j}$ for every pair of features. That is $n^2/2$ free numbers, and each one is learned **only** from rows where feature $i$ and feature $j$ are both non-zero.

In recommender data that is fatal. If you one-hot encode users and items, the pair (Alice, Star Trek) has either one training row or zero. With zero rows, the best fit for $w_{\text{A,ST}}$ is exactly $0$. The model learns nothing about interactions and collapses back to "user bias + item bias".

The fix: stop giving each pair its own weight. Give each **feature** a small vector $v_i \in \mathbb{R}^k$, and define the pair weight as a dot product.

$$\hat{w}_{i,j} := \langle v_i, v_j \rangle = \sum_{f=1}^{k} v_{i,f}\, v_{j,f}$$

Now the parameters are tied together. The row (Bob, Star Trek) teaches you about $v_{\text{ST}}$. The row (Bob, Star Wars) teaches you about $v_{\text{SW}}$, and Bob's shared vector pulls those two movie vectors close. Alice's rating of Star Wars gives you $\langle v_A, v_{SW}\rangle$. Because $v_{ST} \approx v_{SW}$, you now get a sensible $\langle v_A, v_{ST}\rangle$ **for a pair you never observed**. Data for one interaction leaks usefully into related interactions. That is the whole trick.

Two consequences make it more than a nice idea:

1. The model looks like a polynomial SVM but can be trained **in the primal** with SGD, in time linear in $n$ (a rearrangement, shown below, kills the $O(n^2)$ double loop). No dual, no support vectors, no keeping training data around at prediction time.
2. Because the input is just a generic real-valued vector, you get matrix factorisation, PARAFAC, SVD++, PITF and FPMC as **special cases** — you pick which one by choosing how to build $x$, not by writing a new model and a new derivation. This is the part that made FM a workhorse: one implementation, many models.

> [!NOTE] Factorization Machine
> A general predictor $\hat y(x) = w_0 + \sum_i w_i x_i + \sum_{i<j}\langle v_i, v_j\rangle x_i x_j$, where every pairwise weight is factorised into a dot product of per-feature latent vectors, so pair weights share statistical strength instead of being independent. ^factorization-machine

## The Methodology

**The model equation (degree 2).**

$$\hat y(x) := w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle v_i, v_j\rangle x_i x_j$$

Parameters: $w_0 \in \mathbb{R}$ (global bias), $w \in \mathbb{R}^n$ (per-feature strength), $V \in \mathbb{R}^{n\times k}$ (the latent vectors). $k$ is the one hyperparameter that matters.

**Why $k$ should be small.** For any positive-definite $W$ there is a $V$ with $W = VV^\top$ if $k$ is big enough — so FM is fully expressive in the limit. Under sparsity you deliberately keep $k$ small. Restricting $k$ *is* the [[Regularization|regulariser]]; it is what forces the sharing that lets you generalise to unseen pairs.

**The linear-time rearrangement.** Naively the double sum costs $O(kn^2)$. Because no parameter carries an index $(i,j)$, you can complete the square:

$$\sum_{i<j}\langle v_i,v_j\rangle x_i x_j = \frac{1}{2}\sum_{f=1}^{k}\left[\left(\sum_{i=1}^{n} v_{i,f}x_i\right)^{2} - \sum_{i=1}^{n} v_{i,f}^{2}x_i^{2}\right]$$

Cost drops to $O(kn)$, and since only non-zero $x_i$ contribute, to $O(k\,\bar m_D)$ where $\bar m_D$ is the average number of non-zeros. For plain user-item MF, $\bar m_D = 2$ — a prediction is essentially free.

**Gradients.** Trained by SGD with square, logit, or hinge loss, plus $L_2$:

$$\frac{\partial}{\partial \theta}\hat y(x) = \begin{cases} 1 & \theta = w_0 \\ x_i & \theta = w_i \\ x_i \sum_{j=1}^{n} v_{j,f}x_j - v_{i,f}x_i^{2} & \theta = v_{i,f}\end{cases}$$

The inner sum $\sum_j v_{j,f}x_j$ does not depend on $i$, so cache it during the forward pass and every gradient is $O(1)$. A full update for one row is $O(k\,m(x))$. See [[Backpropagation]] for the general pattern — here the "network" is shallow enough to write the [[Derivative#Gradient|gradient]] by hand.

**Tasks.** Regression (use $\hat y$ directly, least squares); binary classification (sign of $\hat y$, hinge or logit); ranking (sort by $\hat y$, train on pairs $(x^{(a)}, x^{(b)})$ with a pairwise loss, as in [[BPR- Bayesian Personalized Ranking from Implicit Feedback]]).

**Higher degree.** A $d$-way FM adds terms for $l$-way interactions with its own factor matrix $V^{(l)} \in \mathbb{R}^{n\times k_l}$ per level, factorised PARAFAC-style. Naive cost $O(k_d n^d)$, again reducible to linear by the same argument. The paper does not experiment with $d>2$.

**The feature-engineering trick — how FM eats other models.** All of these come from choosing $x$:

- **Matrix factorisation.** $x$ = one-hot user + one-hot item. Everything else drops out, leaving $\hat y = w_0 + w_u + w_i + \langle v_u, v_i\rangle$ — biased MF exactly.
- **SVD++.** Add a block of $|L|$ implicit indicators for every movie the user has rated, each set to $1/\sqrt{|N_u|}$. FM then produces the SVD++ equation *plus* extra terms: item-set biases $w_l$, user–$N_u$ interactions $\langle v_u,v_l\rangle$, and pairwise interactions between movies inside $N_u$.
- **PITF (tag recommendation).** One-hot user + item + tag gives $w_0 + w_u + w_i + w_t + \langle v_u,v_i\rangle + \langle v_u,v_t\rangle + \langle v_i,v_t\rangle$. When you optimise pairwise ranking between two tags under the same $(u,i)$, everything not containing $t$ cancels, leaving $w_t + \langle v_u,v_t\rangle + \langle v_i,v_t\rangle$ — PITF, up to (i) the extra bias $w_t$ and (ii) FM sharing one $v_t$ across both interactions where PITF keeps two separate tag vectors.
- **FPMC (next-basket).** One-hot user + item, plus a normalised block for the previous basket $B^u_{t-1}$. Ranking cancellation leaves $w_i + \langle v_u,v_i\rangle + \frac{1}{|B^u_{t-1}|}\sum_{l\in B^u_{t-1}}\langle v_i,v_l\rangle$.

The example feature vector in the paper (Figure 1) is a movie rating row with five blocks: active user one-hot, active item one-hot, normalised set of other rated movies, a real-valued month counter, and a one-hot of the previously rated movie.

## Ablation Studies and Experiments

The paper is mostly theory; there are two figures.

**Netflix rating prediction (100M training instances), RMSE vs $k$ from 0 to ~128.** FM error falls steadily as $k$ grows, reaching roughly 0.90 RMSE. The SVM stays flat around 0.96–0.98 no matter how much capacity you give it. This is the central empirical claim: the polynomial SVM has the *same expressive class* as a 2-way FM, and still cannot use it, because under sparsity the max-margin solution for every test-case interaction weight $w^{(2)}_{u,i}$ is $0$.

Worked out: with only user and item one-hots, $m(x)=2$, and the polynomial SVM reduces to
$$\hat y(x) = w_0 + \sqrt2(w_u + w_i) + w^{(2)}_{u,u} + w^{(2)}_{i,i} + \sqrt2\, w^{(2)}_{u,i}.$$
$w_u$ and $w^{(2)}_{u,u}$ are redundant. And each $w^{(2)}_{u,i}$ sees at most one training observation, zero for test pairs. So the polynomial SVM degenerates into the **linear** SVM — bias-only collaborative filtering. That is exactly the flat line in the figure.

**ECML/PKDD Discovery Challenge 2009, Task 2 (tag recommendation), F1@top-5 vs number of parameters.** FM matches the PITF model that *won* the challenge, across parameter budgets from ~0 to 1.2e7, both reaching around 0.35 F1. Not better — comparable. The point is that a generic model with the right input matches a hand-designed, task-specific one.

**What does not work / what the analysis rules out.**

- **Linear SVM** on this data: parameters estimate fine but quality is low, because it only ever learns $w_0 + w_u + w_i$.
- **Polynomial SVM**: fails not from optimisation trouble but from the *parametrisation* — dense, independent $w_{i,j}$ requires direct observations of the pair $(i,j)$. If either $x_i = 0$ or $x_j = 0$, that row contributes nothing to $w^{(2)}_{i,j}$. Too few co-occurrences, no learning. This generalises beyond CF to any hugely sparse problem, including bag-of-words text.
- **Large $k$ under sparsity**: the paper argues (not measured directly) that big $k$ hurts generalisation, because there is not enough data to estimate a complex $W$. Small $k$ is the whole point.
- **Standard MF / PARAFAC**: not general predictors. They require $x$ to be partitioned into $m$ blocks with exactly one $1$ per block. You cannot hand them a real-valued feature like "months since 2009".

The ablation that matters is implicit: factorised vs dense parametrisation, with expressiveness held constant. Factorisation is doing all the work.

## Worth Remembering

- **FM $\ne$ MF with side features bolted on.** It is a genuinely different weight-tying scheme. The dense-vs-factorised distinction is the reusable idea; it shows up again anywhere you have a huge cross-product of categoricals.
- The FM versions of SVD++ and FPMC are *not identical* to the originals — they carry extra interaction terms (movie–movie inside $N_u$, user–basket-item) that the original models omit. Rendle argues this is harmless or helpful, but it is an approximation, not an equality.
- FM shares one vector $v_t$ per feature across *all* its interactions. PITF gives a tag two vectors, one for the user-interaction and one for the item-interaction. Sharing is more constrained; empirically it did not cost anything on the challenge data. This constraint is exactly what Field-aware FMs later relax.
- **Prediction needs only the parameters.** Unlike a kernel SVM, you never store training data. $O(k\,m(x))$ per prediction makes it deployable in a ranker.
- The linear-time rearrangement is the same "sum of squares minus squares of sums" identity that shows up in variance computation. Worth internalising — it is reused in DeepFM, xDeepFM and the interaction layers of most CTR models.
- Limitation the paper does not dwell on: FM is still a **shallow, degree-2** model. It cannot represent higher-order or non-monotone interactions unless you go to $d$-way, which is never evaluated. Everything after this (Wide & Deep, [[Deep Interest Network for CTR Prediction (DIN)]], DeepFM) is about layering learned non-linearity on top of this base.
- Practical caveat: FM is very sensitive to feature normalisation within a block. Note that in every mimicry construction, set-valued blocks are normalised — $1/\sqrt{|N_u|}$ for SVD++, $1/|B^u_{t-1}|$ for FPMC. Different normalisation gives you a different model.
- Follow-up question worth chasing: FM assumes the interaction matrix is low-rank *and* positive semi-definite in the limit ($W = VV^\top$). Some real interactions are repulsive (negative). $\langle v_i, v_i \rangle \geq 0$ always, so self-interaction is forced non-negative — though the $i<j$ sum excludes it.

## Links

Related: [[Collaborative Filtering for Implicit Feedback Datasets (ICDM)]] · [[BPR- Bayesian Personalized Ranking from Implicit Feedback]] · [[Recommender Systems - Evolution]] · [[Deep Interest Network for CTR Prediction (DIN)]] · [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] · [[Regularization]] · [[Derivative#Gradient|gradient]] · [[Fundamentals#Vectors|vectors]] · [[Foundational_RecSys_Ranking_Reading_Plan]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Distributed Representations of Words and Phrases (negative sampling)]]

New topics worth writing: Field-aware Factorization Machines, DeepFM, Support Vector Machines and the kernel trick, PARAFAC / CP tensor decomposition, SVD++, FPMC and next-basket recommendation, low-rank matrix completion, libFM
