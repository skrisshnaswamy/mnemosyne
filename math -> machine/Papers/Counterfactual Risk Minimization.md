---
title: "Counterfactual Risk Minimization"
authors: ["Adith Swaminathan", "Thorsten Joachims"]
year: 2015
arxiv: "1502.02362"
url: https://arxiv.org/abs/1502.02362
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, rl, optimization, theory]
---
## The Core Idea

You have a log from a live system. For each user, the system picked one action, and you saw one number back — a click, a loss, a reward. You never learn what would have happened with any *other* action. This is **logged bandit feedback**. The question: can you train a *better* policy from this log alone, without running a new A/B test?

The standard answer is [[Unbiased Learning-to-Rank with Biased Feedback|inverse propensity scoring]]: reweight each logged example by how likely your new policy was to take that action, divided by how likely the old policy was. That gives an unbiased estimate of the new policy's risk. Then just pick the policy that minimises the estimate.

The insight of this paper: **unbiasedness is not enough.** The IPS estimator has a *different variance for every candidate policy*. A policy that stays close to the logging policy $h_0$ gets a tight, trustworthy estimate. A policy that puts mass on actions $h_0$ almost never took gets a wild, noisy estimate — and noise looks like good luck about half the time. So plain empirical risk minimisation over IPS is a machine for finding policies whose estimates happen to be optimistically wrong. It systematically picks the far-away, poorly-supported policies.

The fix: penalise the estimator's own empirical variance. Concretely, minimise

$$\hat{h}^{CRM} = \arg\min_{h \in \mathcal{H}}\left\{ \hat{R}^M(h) + \lambda\sqrt{\frac{\bm{Var}_h(u)}{n}} \right\}$$

This is not a heuristic. It falls out of an empirical Bernstein generalisation bound — a bound on true risk whose second term literally *is* the empirical standard deviation of the weighted losses. So minimising this objective is minimising a high-probability upper bound on the true risk. That is exactly the logic of Vapnik's structural risk minimisation, transplanted to the counterfactual setting, which is why they name it **Counterfactual Risk Minimisation**.

> [!NOTE] Counterfactual Risk Minimisation
> Pick the policy with the tightest *conservative upper bound* on true risk, where the bound = propensity-weighted empirical risk + a data-dependent variance penalty. The variance term acts as a regulariser that pulls the learned policy toward regions of action space the logging policy actually explored. ^crm-principle

What it unlocks: batch, offline policy learning over hypothesis classes as rich as those used in supervised learning — infinite, gradient-optimisable, with exponentially large output spaces — instead of exhaustive search over a handful of candidate policies.

## The Methodology

**Setup.** Inputs $x \sim \Pr(\mathcal{X})$. A hypothesis is a *stochastic* policy $h(y \mid x)$ — a distribution over outputs. Deterministic policies are the special case with all mass on one $y$. Risk is

$$R(h) = \mathbb{E}_{x}\mathbb{E}_{y \sim h(x)}[\delta(x,y)]$$

with $\delta$ a loss (small = user happy). The log is $\mathcal{D} = \{(x_i, y_i, \delta_i, p_i)\}$ where $y_i \sim h_0(x_i)$ and $p_i = h_0(y_i \mid x_i)$ is the recorded **propensity**. You must log the propensity at serving time; you cannot recover it later.

**Step 1 — importance sampling.** Change of measure gives

$$R(h) = \mathbb{E}_x \mathbb{E}_{y \sim h_0(x)}\left[\delta(x,y)\frac{h(y\mid x)}{h_0(y \mid x)}\right]$$

**Step 2 — clip the weights.** If some $p_i \approx 0$ the ratio explodes and variance is unbounded. Truncate at $M$:

$$\hat{R}^M(h) = \frac{1}{n}\sum_{i=1}^n \delta_i \min\left\{M, \frac{h(y_i \mid x_i)}{p_i}\right\}$$

$M$ trades bias for variance. Smaller $M$ = more bias.

**Step 3 — scale the loss into $[-1, 0]$.** This is a subtle trap and the paper is emphatic about it. Supervised ERM does not care if you add a constant to the loss. Here it matters completely. If $\delta \geq 0$, a policy that assigns probability $0$ to every logged action gets $\hat{R}^M = 0$ with zero variance — the best possible objective — while being an arbitrarily terrible policy. The objective became a *lower* bound instead of an upper bound. For $\delta \in [\bigtriangledown, \bigtriangleup]$, transform

$$\delta' = \frac{\delta - \bigtriangleup}{\bigtriangleup - \bigtriangledown}$$

so all losses are non-positive and the estimator upper-bounds risk.

**The bound (Theorem 1).** With probability $\geq 1-\gamma$, for all $h$, with $n \geq 16$:

$$R(h) \le \hat{R}^M(h) + \sqrt{\frac{18\,\bm{Var}_h(u)\,\mathcal{Q}_\mathcal{H}(n,\gamma)}{n}} + \frac{15 M \mathcal{Q}(n,\gamma)}{n-1}$$

where $u_h^i = \delta_i \min\{M, h(y_i|x_i)/p_i\}$ and $\mathcal{Q}$ is a log-covering-number capacity term. Proof is Maurer & Pontil's empirical Bernstein bound applied to an auxiliary deterministic function class $f_h(x,y) = 1 + \frac{\delta(x,y)}{M}\min\{M, \frac{h(y|x)}{h_0(y|x)}\} \in [0,1]$, which lets classical capacity notions apply to a class of *stochastic* policies.

**POEM.** The instantiation for structured output prediction. The policy family is an exponential model over the whole output space:

$$h_w(y \mid x) = \frac{\exp(w \cdot \phi(x,y))}{\mathbb{Z}(x)}$$

This is the "soft-max" version of the usual $\arg\max_y w\cdot\phi(x,y)$ rule from structured SVMs or CRFs. For multi-label classification $\phi(x,y) = x \otimes y$. Plug into the CRM objective and you get the POEM training objective in $w$. It is **not convex**, even at $\lambda = 0$.

**Iterated variance majorization** — the trick that makes SGD possible. The $\sqrt{\bm{Var}_w(u)}$ term couples all $n$ examples, so you cannot write it as a sum over minibatches. Because $\sqrt{\cdot}$ and $-(\cdot)^2$ are both concave, a first-order Taylor expansion at the current $w_0$ gives an *upper* bound that decomposes:

$$\sqrt{\bm{Var}_w(u)} \le A_{w_0}\sum_i u_w^i + B_{w_0}\sum_i \{u_w^i\}^2 + C_{w_0} = Q(w; w_0)$$

with $A_{w_0} = -\overline{u_{w_0}}/\{(n-1)\sqrt{\bm{Var}_{w_0}(u)}\}$ and $B_{w_0} = 1/\{2(n-1)\sqrt{\bm{Var}_{w_0}(u)}\}$. Now every term is a sum over examples. Freeze $A, B$ at the start of each epoch, run SGD, refresh at epoch end. This is majorisation–minimisation and it provably decreases the true objective. The per-sample update:

$$w \leftarrow w - \eta\left\{\nabla u_w^i + \lambda\sqrt{n}\left(A_{w^t}\nabla u_w^i + 2B_{w^t}u_w^i \nabla u_w^i\right)\right\}$$

Optimiser is mini-batch AdaGrad, batch size 100, initialised at $w = 0$ (i.e. uniform over $\mathcal{Y}$). See [[Adam- A Method for Stochastic Optimization]] for the adaptive-learning-rate lineage.

**Hyperparameter rules of thumb.** $M \approx$ ratio of the 90th percentile to the 10th percentile propensity in the training set. $\lambda = c\lambda^*$ with $c \in [10^{-6}, \dots, 1]$ in decades, where $\lambda^*$ is calibrated so that $\hat{R}^M(h_0) + \lambda^*\sqrt{\bm{Var}_{h_0}(u)/n} < 0$. The reason for the guardrails: $\hat{R}^M \in [-M, 0]$ and the variance penalty is in $[0, M/(2\sqrt{n})]$, so if $M$ is too small every policy gets the same clipped estimate, and if $\lambda$ is too big the degenerate "avoid all logged data" policy wins with objective $0$. Validation uses the *unclipped* unbiased IPS estimator on a held-out 25%.

## Ablation Studies and Experiments

**Data generation.** Supervised→bandit conversion. Take a labelled multi-label dataset, train a CRF on 5% of it as $h_0$, sample $y_i \sim h_0(x_i)$, record Hamming loss as $\delta_i$ and $h_0(y_i|x_i)$ as $p_i$. Four passes over the training set. Datasets: Scene (294 feat, 6 labels), Yeast (103, 14), TMC (30438, 22), LYRL (47236, 4). Evaluate expected Hamming loss on the true supervised test set. Averaged over 10 runs.

**Main result — Table 3, test Hamming loss, lower is better:**

| | Scene | Yeast | TMC | LYRL |
|---|---|---|---|---|
| $h_0$ | 1.543 | 5.547 | 3.445 | 1.463 |
| IPS (L-BFGS) | 1.193 | 4.635 | 2.808 | 0.921 |
| **POEM (L-BFGS)** | **1.168** | **4.480** | **2.197** | **0.918** |
| IPS (SGD) | 1.519 | 4.614 | 3.023 | 1.118 |
| **POEM (SGD)** | **1.143** | 4.517 | 2.522 | 0.996 |
| CRF (full supervision, skyline) | 0.659 | 2.822 | 1.189 | 0.222 |

POEM beats IPS on every dataset and every optimiser variant, significant at $p < 0.05$ (one-tailed paired t-test, 10 runs). The variance regulariser is doing real work — the gap is largest on TMC (2.808 → 2.197, batch). Note also that IPS(SGD) on Scene barely improves on $h_0$ at all (1.519 vs 1.543) while POEM(SGD) gets 1.143 — variance regularisation rescues the stochastic optimiser.

The CRF skyline is far below everything. That is expected and honest: each full-information label is worth roughly $|\mathcal{Y}| = 2^q$ bandit labels of information.

**Compute — Table 4, seconds per validation run.** Variance regularisation is not free. POEM(batch) on TMC takes 949.95s vs 136.34s for IPS(batch). The majorisation trick recovers most of it: POEM(SGD) on TMC is 276.13s. Runtime is dominated by computing the partition function $\mathbb{Z}(x_i)$, so the gap widens with more labels.

**MAP predictions (Table 5).** Take the learned $w$ and predict $\arg\max_y w\cdot\phi(x,y)$ instead of sampling. This is faster at serve time — no partition function needed. Result: not worse, usually slightly better. Yeast 4.517 → 4.065; TMC 2.522 → 2.299; LYRL 0.996 → 0.880. Practically useful.

**More data (Figure 1, Yeast).** Replay the training set $2^1$ to $2^8$ times, resampling from $h_0$ each time. Risk falls monotonically toward, but nowhere near, the CRF. Convergence is slow because the exponential family has thin tails — $h_0$ rarely explores the interesting corners of $\mathcal{Y}$.

**Quality of $h_0$ (Figure 2, Yeast).** Vary the fraction $f$ of supervised data used to train $h_0$, from 1% to 100%. There is a real tension: a better $h_0$ gives cleaner data but is harder to beat. POEM finds a policy at least as good as $h_0$ at every $f$. Even a very poor $h_0$ ($f \approx 0.2$) yields a useful improvement.

**Stochasticity of $h_0$ — the most interesting ablation (Figure 3, LYRL).** Sharpen the logging policy with a temperature multiplier $w_0 \mapsto \alpha w_0$, $\alpha \in [0.5, 32]$. Theory says counterfactual learning is *impossible* if $h_0$ is deterministic — you can always construct adversarial losses on the unexplored actions, even as $|\mathcal{D}| \to \infty$. So what happens as you approach that wall? POEM improves on both $h_0$ and $h_0^{map}$ as long as some minimum stochasticity remains, with the biggest margins when $h_0$ is most random. Past $\alpha \geq 2^4$, POEM's performance simply **collapses onto $h_0^{map}$** — it does not blow up, it does not pick something absurd, it just reproduces the logging policy's deterministic predictions. That graceful degradation is the strongest evidence that the conservative bound is behaving as designed.

**What did not work / what breaks.**
- Naive IPS-ERM with $\delta \geq 0$: completely degenerate. The optimiser finds a policy avoiding all logged data, scores $0$ risk and $0$ variance. Loss rescaling is mandatory, not cosmetic.
- $\lambda$ too large: same degeneracy returns, because the "avoid everything" policy achieves objective $0$ which beats any negative-but-noisy real policy.
- $M$ too small: every hypothesis gets the same clipped estimate and learning signal vanishes.
- Deterministic or non-full-support $h_0$: no amount of data helps. Impossibility, not slow convergence.
- Adding $\ell_2$ regularisation on $w$ (as CRFs do) changed none of the trends.
- Heavier-tailed $h_0$ is *not* automatically better. What matters is whether $h_0$ explores the regions of $\mathcal{Y}$ with *good* losses.

## Worth Remembering

- **The reframe that makes this work**: expand the hypothesis class from deterministic functions to *stochastic policies*. Then a policy's distance from the logging policy becomes measurable, and the variance of its risk estimate becomes a computable quantity you can regularise on. Without stochastic hypotheses there is no propensity ratio to write down.

- The variance penalty is a **data-dependent regulariser**. Unlike $\ell_2$ or [[Dropout- A Simple Way to Prevent Overfitting|dropout]], it does not encode a prior about weight magnitude. It encodes "do not trust yourself where you have no evidence." Conceptually the same instinct as the pessimism/conservatism family in [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives|offline RL]], arriving several years earlier and from a generalisation-bound direction rather than a Bellman-error one.

- The results are stated for clipped IPS but the authors note they hold equally for [[Doubly Robust Policy Evaluation and Learning|doubly robust]] estimators, which would give lower variance still if you also have a decent model of the feedback.

- **You must log propensities.** This is an engineering commitment made at serving time. No propensity, no CRM. Exploration scavenging and bootstrapping exist as workarounds for deterministic logs, but they are not what this paper does.

- CRM applies more broadly than bandit feedback: the objective never needs $\nabla \delta$, only $\delta$ evaluated at logged points. So it is a route to supervised learning with **non-differentiable losses** (NDCG, BLEU, revenue) as long as you can sample and score.

- Limitation the authors name: noisy $\delta$, ordinal or coactive feedback, and adaptive (non-stationary) $h_0$ are all out of scope. Real production logging policies are usually retrained weekly, which violates the stationarity assumption directly.

- The theory covers global minimisers; the objective is non-convex and POEM finds local optima. It works empirically, but nothing in Theorem 1 protects you at a local minimum.

- A practical caveat: the variance term makes each gradient step depend on epoch-level statistics ($A_{w^t}, B_{w^t}$). Any implementation needs a full pass to refresh them, so this is not a pure streaming algorithm.

## Links

Related: [[Counterfactual Reasoning and Learning Systems]] · [[Doubly Robust Policy Evaluation and Learning]] · [[Unbiased Learning-to-Rank with Biased Feedback]] · [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]] · [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] · [[Adam- A Method for Stochastic Optimization]] · [[Regularization]] · [[Uncertainty]] · [[Random variable]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[Decision Sciences]]

New topics worth writing: Importance sampling and effective sample size, Empirical Bernstein bounds, Structural Risk Minimisation and covering numbers, Majorization-Minimization optimisation, Truncated/clipped importance weights, Conditional Random Fields, Off-policy evaluation, Pessimism in offline learning
