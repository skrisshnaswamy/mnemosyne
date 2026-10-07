---
title: "Optimal and Adaptive Off-policy Evaluation in Contextual Bandits (SWITCH)"
authors: ["Wang", "Agarwal & Dudík"]
year: 2016
arxiv: "1612.01205"
url: https://arxiv.org/abs/1612.01205
priority: Good-To-Read
read_on: 2026-09-25
tags: [paper, rl, theory]
---
## The Core Idea

Off-policy evaluation asks: given logs from an old system, how well would a *new* policy have done? The two standard answers are [[Doubly Robust Policy Evaluation and Learning|doubly robust]] (DR) and plain [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)#^replay-evaluation|importance weighting]] (IPS). This paper does two things.

**First, a hardness result.** Earlier theory said IPS is wasteful — you should always be able to beat it with a reward model. This paper shows that claim quietly assumed something strong: that you can *consistently* learn $\mathbb{E}[r \mid x, a]$ from the features. Drop that assumption and the picture flips. When contexts are rich — high-dimensional, continuous, each user basically unique — IPS and DR are **minimax optimal**. Nothing can beat them in the worst case by more than a constant factor.

That is the sentence worth keeping. There are two regimes, and they behave oppositely:

> [!NOTE] Agnostic contextual regime
> When each context appears roughly once (continuous $x$, high-dimensional features), you cannot estimate per-context rewards by averaging, and no reward model is guaranteed to be right. Here IPS is optimal. When contexts repeat many times (multi-armed bandits, few discrete segments), you *can* average, a reward model is consistent, and IPS is strictly wasteful. Most real recommender logs are the first case. ^agnostic-contextual

**Second, a practical estimator.** Minimax optimal does not mean good. IPS and DR both blow up when a single importance weight $\rho = \pi(a\mid x)/\mu(a\mid x)$ is huge — one log line with $\rho = 5000$ dominates the whole estimate. The **switch** estimator handles this by splitting the problem per *(context, action)* pair: where the weight is small, use IPS or DR; where the weight is big, throw the weight away and just trust the reward model. One threshold $\tau$ decides which.

Why this did not exist before: prior fixes (clipping weights, trimming) all operate on the weight itself, and either keep a bad weight or drop the data point entirely. Switching says the *estimator* should change, not the weight. The region where importance weighting is unreliable is exactly the region where you must fall back on a model, accept some bias, and stop pretending to be unbiased.

What it unlocks: orders-of-magnitude MSE reductions on real data, and a bias–variance dial you can tune from the logs themselves, with no held-out ground truth.

## The Methodology

### Setup

Context $x \sim \lambda$, action $a$ from a finite set, reward $r \sim D(r\mid x,a)$. Logging policy $\mu$ collected the data; target policy $\pi$ is what we want to score. The quantity wanted:

$$v^\pi = \mathbb{E}_{x\sim\lambda}\,\mathbb{E}_{a\sim\pi(\cdot\mid x)}\,\mathbb{E}_{r\sim D}[r]$$

Importance weight: $\rho(x,a) = \pi(a\mid x)/\mu(a\mid x)$. Assumed finite everywhere $\pi$ puts mass — the usual **absolute continuity** condition. If $\mu$ never took an action $\pi$ likes, you are stuck.

The three baseline estimators:

$$\hat{v}_{\text{IPS}} = \frac{1}{n}\sum_i \rho_i r_i$$

$$\hat{v}_{\text{DM}} = \frac{1}{n}\sum_i \sum_a \pi(a\mid x_i)\,\hat{r}(x_i,a)$$

$$\hat{v}_{\text{DR}} = \frac{1}{n}\sum_i \Big[\rho_i\big(r_i - \hat{r}(x_i,a_i)\big) + \sum_a \pi(a\mid x_i)\hat{r}(x_i,a)\Big]$$

IPS is DR with $\hat{r} \equiv 0$. Note the paper's Eq. (2) prints IPS without the $1/n$; that is a typo — everything downstream uses the average.

### The lower bound

Fix $\lambda$, $\mu$, $\pi$. Take the worst case only over reward distributions, constrained by two known functions: $0 \le \mathbb{E}[r\mid x,a] \le R_{\max}(x,a)$ and $\mathrm{Var}[r\mid x,a] \le \sigma^2(x,a)$.

**Theorem 1** (with $\xi_\gamma(x,a) = \mathbf{1}(\mu(x,a) \le \gamma)$, the indicator for "rare" pairs):

$$R_n \;\ge\; \frac{\mathbb{E}_\mu[\rho^2\sigma^2] + \mathbb{E}_\mu[\xi_\gamma \rho^2 R_{\max}^2]\big(1 - 350 n\gamma\log(5/\gamma)\big)}{700\,n}$$

If $\lambda$ has a density (continuous contexts), every pair is rare, $\xi_\gamma \equiv 1$, and it collapses to

$$R_n \;\ge\; \frac{\mathbb{E}_\mu[\rho^2\sigma^2] + \mathbb{E}_\mu[\rho^2 R_{\max}^2]}{700\,n}$$

Compare to the known MSE of DR (Dudík et al. 2014, Lemma 3.3(i)):

$$\mathrm{MSE}(\hat{v}_{\text{DR}}) = \frac{1}{n}\Big(\underbrace{\mathbb{E}_\mu[\rho^2\sigma^2]}_{\text{reward noise}} + \underbrace{\mathrm{Var}_x\,\mathbb{E}_{a\sim\mu}[\rho r^*]}_{\text{context variation}} + \underbrace{\mathbb{E}_x\,\mathrm{Var}_{a\sim\mu}[\rho(\hat{r}-r^*)]}_{\text{model error}}\Big)$$

With $0 \le \hat{r} \le R_{\max}$, this is $\mathcal{O}\big(\frac{1}{n}(\mathbb{E}_\mu[\rho^2\sigma^2] + \mathbb{E}_\mu[\rho^2 R_{\max}^2])\big)$ — the lower bound up to a constant. **Minimax risk is $\Theta$ of that expression, and both IPS and DR hit it.**

The same equation also says *why DR still beats IPS in practice*: the third term shrinks as $\hat{r}$ gets closer to $r^*$, and IPS is the case $\hat{r}\equiv 0$. Minimax-equal, finite-sample-unequal.

**Proof shape.** Both halves are Le Cam two-point arguments — build two problem instances close enough in KL that no test can tell them apart in $n$ samples, far enough apart in $v^\pi$ that any estimator must be wrong about one.

- *Term one* ($\sigma^2$): rewards are Gaussian with mean $\eta(x,a)$; reduces to Gaussian mean estimation. This term matches Li et al. (2015) for multi-armed bandits.
- *Term two* ($R_{\max}^2$): this is the new bit. Set $\sigma \equiv 0$ — rewards are **deterministic**. Naively that should be easy: just read the reward off. The trick is to put a *prior* over reward functions, $\eta(x,a) = R_{\max}$ with probability $\theta$, else $0$, constant on each cell of a discretisation of $\mathcal{X}\times\mathcal{A}$. Then bound $\mathbb{E}_\theta[\mathrm{MSE}_\eta]$ below by $\mathrm{MSE}_{\mathbb{E}_\theta[\eta]}$ using $a^2 \ge (a+b)^2/2 - b^2$. The averaged problem *has* noise even though every individual instance is noiseless, so standard machinery applies. The residual term $\mathcal{T}_2$ — the variance of $v_\eta$ under the prior — is killed with Hoeffding, and that is where the $\gamma\log(5/\gamma)$ correction comes from. This step only works if $\lambda$ spreads mass over many contexts. With a single context it evaporates, which is exactly why multi-armed bandits are easier.

### The switch estimator

Split $v^\pi$ by importance weight:

$$\mathbb{E}_\pi[r] = \underbrace{\mathbb{E}_\mu[\rho\, r\,\mathbf{1}(\rho \le \tau)]}_{\text{estimate by IPS/DR}} + \underbrace{\mathbb{E}_{x}\Big[\sum_a \mathbb{E}[r\mid x,a]\,\pi(a\mid x)\mathbf{1}(\rho(x,a) > \tau)\Big]}_{\text{estimate by DM}}$$

Both halves are exact identities. Plug in sample estimates:

$$\hat{v}_{\textsc{switch}} = \frac{1}{n}\sum_i r_i\rho_i\mathbf{1}(\rho_i \le \tau) \;+\; \frac{1}{n}\sum_i\sum_a \hat{r}(x_i,a)\,\pi(a\mid x_i)\,\mathbf{1}(\rho(x_i,a) > \tau)$$

The second sum runs over **all** actions in each logged context, not just the one taken — you need $\rho(x_i,a)$ for every $a$, so you need the full logging distribution, not just the propensity of the chosen action.

Replace the first term with DR and you get **switch-DR**, the version that actually wins. The $\hat{r}$ inside DR and the $\hat{r}$ used for imputation need not be the same model.

$\tau = 0$ gives pure DM. $\tau \to \infty$ gives pure IPS (or DR).

**Theorem 2** bounds the MSE, with $\epsilon(x,a) = \hat{r}(x,a) - \mathbb{E}[r\mid x,a]$ the model bias:

$$\frac{2}{n}\Big\{\mathbb{E}_\mu\big[(\sigma^2 + R_{\max}^2)\rho^2\mathbf{1}(\rho\le\tau)\big] + \mathbb{E}_\pi\big[R_{\max}^2\mathbf{1}(\rho>\tau)\big]\Big\} + \mathbb{E}_\pi\big[\epsilon\,\mathbf{1}(\rho>\tau)\big]^2$$

Read it as: variance only ever sees weights capped at $\tau$, and bias only ever comes from the switched-off region.

### Picking $\tau$ from the data

This is the part you would actually implement. No ground truth available, so estimate variance and *upper bound* the bias.

Let $Y_i(\tau)$ be the per-sample switch contribution, $\bar Y(\tau)$ its mean. Then

$$\widehat{\mathrm{Var}}_\tau = \frac{1}{n^2}\sum_i \big(Y_i(\tau) - \bar Y(\tau)\big)^2$$

For bias, take the last term of Theorem 2 and replace the unknown $\epsilon$ by its worst case $R_{\max}$ (assumed known — usually it is; here $R_{\max}\equiv 1$):

$$\widehat{\text{Bias}}^2_\tau = \Big[\frac{1}{n}\sum_i \mathbb{E}_\pi\big[R_{\max}\mathbf{1}(\rho>\tau)\,\big|\,x_i\big]\Big]^2$$

$$\hat\tau = \arg\min_\tau\; \widehat{\mathrm{Var}}_\tau + \widehat{\text{Bias}}^2_\tau$$

Searched over 21 thresholds on an exponential grid between the smallest and largest observed importance weight.

The bias bound is deliberately **pessimistic**: it assumes DM is maximally wrong at every switched point. Consequence — switch only abandons the unbiased part when the variance there is enormous. That conservatism is what preserves minimax optimality: any bias incurred is bounded by the estimate, and only paid when IPS would have cost more.

## Ablation Studies and Experiments

10 UCI multiclass datasets turned into contextual bandits the standard way: label = action, reward 1 if correct, else 0. Two reward regimes:

- **deterministic** — reward is exactly the 0/1 correctness,
- **noisy** — with probability 0.5 reveal the true reward, otherwise a coin flip. Raises $\sigma^2$ deliberately.

$\pi$ = deterministic argmax of a logistic regression on the multiclass data. $\mu$ = a logistic model fit on a **covariate-shifted** copy, so the two policies genuinely disagree. $R_{\max}\equiv 1$.

Sizes $100, 200, 500, 1000, \ldots$ up to $n$, bootstrap-sampled, 500 replicates each. Reported metric is MSE **truncated at 1**: $\mathbb{E}[(\hat v - v^\pi)^2 \wedge 1]$. The authors are explicit that this truncation *helps IPS and DR* by capping their catastrophic runs — switch's numbers are unaffected. Results are then normalised: $\text{Rel. MSE} = \text{MSE}(\hat v)/\text{MSE}(\hat v_{\text{IPS}})$.

### Baselines

| Method | What it does |
|---|---|
| IPS | plain importance weighting |
| DM | logistic regression reward model |
| DR | doubly robust, 2-fold cross-fitting of $\hat r$ |
| TrunIPS | cap weights at $\tau$, renormalise to sum to 1 (Bembom & van der Laan) |
| TrimIPS | switch with $\hat r \equiv 0$ — discard large-weight points (Bottou et al.) |
| MAGIC | Thomas & Brunskill's weighted combination over many thresholds |

DR uses two-fold splitting: fit $\hat r$ on one half, evaluate on the other, swap, average. DM does **not** — it fits and evaluates on the same data, which is part of why DM looks optimistic at small $n$.

### Findings

**switch-DR dominates every baseline** on both reward settings, by up to orders of magnitude in relative MSE (log scale in the plots). The margin is *larger* in the noisy-reward setting — bigger $\sigma$ hurts IPS and DR more, and capping weights is worth more.

**The tuning rule is close to oracle.** They also ran $\tau$ chosen in hindsight to minimise test MSE. The data-driven $\hat\tau$ tracks it closely — the conservative bias bound costs little.

**Two regimes visible in the $n$-curves** (optdigits and yeast):

1. DM strong early, overtaken by IPS/DR as $n$ grows. Here switch-DR beats *both* at every $n$; DR alone improves on IPS only modestly.
2. DM poor throughout. switch-DR matches IPS and DR — it does not get dragged down by a bad model.

That second case is the important robustness claim: switch does not inherit DM's failure.

### What did not work

**MAGIC's automatic weighting reverts to DM.** This is the sharpest negative result in the paper. MAGIC and switch are structurally similar — both trade off bias and variance over a threshold family — but MAGIC's **bias estimate is too optimistic**. So whenever IPS or DR show meaningful variance, MAGIC's objective prefers DM, and it collapses onto the reward model. switch's deliberately pessimistic $R_{\max}$-based bound does the opposite. The lesson generalises: *in a bias–variance selection rule, the bias estimator being conservative is a feature, not sloppiness.*

**TrimIPS ($\hat r\equiv 0$) is strictly worse than switch.** Same threshold, same variance reduction, but throwing away the high-weight region entirely instead of imputing it. The gap is the value of the reward model *specifically in the region where weighting fails* — not, as DR uses it, as a control variate everywhere.

**TrunIPS** (cap-and-renormalise) also loses. Capping keeps the point but with a wrong weight; renormalising adds its own bias with no principled control.

**IPS alone is unusable in practice** despite being minimax optimal. The theory and the experiments say opposite-sounding things and both are right: minimax is a worst-case statement over reward distributions, and real problems are not the worst case.

**The constants do not match.** Upper bound $\mathcal{O}(1/n \cdot (\cdot))$, lower bound $(\cdot)/700n$. The paper is honest that the gap is an artefact of finite-sample analysis, where lower-order terms must be bounded explicitly rather than ignored.

## Worth Remembering

**The regime distinction is the transferable idea.** Before deciding which estimator to use, ask: *does any given context repeat in my logs?* If yes (few segments, coarse features), a reward model can be consistent and DM/DR are genuinely better. If no (user IDs, embeddings, continuous features), you are in the agnostic regime, no model is trustworthy, and importance weighting is not something you can cleverly outperform — only stabilise. Most production [[Recommender Systems - Evolution|recommender]] logs are the second case.

**This does not contradict Li, Munos & Szepesvári (2015).** They proved minimax risk $\Theta(\mathbb{E}_\mu[\rho^2\sigma^2]/n)$ for multi-armed bandits, with IPS suboptimal and DM optimal. That is precisely the first term of Theorem 1; the second term vanishes when there is one context. Nor does it contradict the semiparametric-efficiency line (Hahn 1998, Hirano et al. 2003): under consistency the achievable risk is $\frac{1}{n}(\mathbb{E}_\mu[\rho^2\sigma^2] + \mathrm{Var}_x\mathbb{E}_\pi[r^*\mid x])$, which is *strictly below* this paper's lower bound. Consistency assumptions buy real statistical power. The point is that you often cannot pay for them.

**Practical caveats if you want to run this:**

- You need $\mu(a\mid x)$ for **every action**, not just the logged one, to compute the indicators in the DM half. Logging only the chosen propensity is not enough. Log the full distribution.
- You need a credible $R_{\max}(x,a)$. Binary or bounded rewards make this trivial; revenue-style rewards with heavy tails do not.
- Absolute continuity is still required. Deterministic logging policies break everything upstream of this paper.
- The grid over $\tau$ (21 points, exponential, spanning observed weights) is cheap — the whole tuning procedure is one pass per threshold.

**What the paper leaves open, in its own words:** high-probability finite-sample bounds on switch (only MSE in expectation here), and sharper lower bounds under *realistic* — rather than none or full — assumptions on the reward model. That middle ground, "the model is somewhat right", is arguably where every real system lives and is still not well characterised.

**Follow-up questions worth chasing:** switch uses a single hard threshold on $\rho$; a soft weighting (shrinkage between DR and DM per point) might do better and is what later OPE work explores. Also, switch-DR's variance estimate ignores the randomness in fitting $\hat r$ — cross-fitting mitigates this but does not remove it, which is exactly the concern [[Double-Debiased Machine Learning for Treatment and Structural Parameters|double machine learning]] formalises through Neyman orthogonality.

**Connection worth making:** the structure — *unbiased where the data supports it, modelled where it does not* — is the same instinct as pessimism in [[Conservative Q-Learning for Offline RL|offline RL]], and the same instinct as [[Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning (MAGIC)|MAGIC]]'s $j$-step blending, applied one step rather than over a horizon. Reading MAGIC and this paper side by side is the cleanest way to see why the bias estimator matters more than the estimator family.

## Links

Related: [[Doubly Robust Policy Evaluation and Learning]] · [[The Self-Normalized Estimator for Counterfactual Learning (SNIPS) (NeurIPS)]] · [[Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms]] · [[Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning (MAGIC)]] · [[Off-Policy Evaluation]] · [[Counterfactual Risk Minimization]] · [[Contextual Bandit]] · [[Counterfactual Reasoning and Learning Systems]] · [[Open Bandit Dataset and Pipeline- Towards Realistic and Reproducible Off-Policy Evaluation]] · [[On-Policy vs Off-Policy]] · [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[Double-Debiased Machine Learning for Treatment and Structural Parameters]] · [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] · [[Uncertainty]]

New topics worth writing: Minimax risk and Le Cam's two-point method, Semiparametric efficiency bounds, Importance weight truncation and clipping, Horvitz-Thompson estimator, Absolute continuity and overlap in OPE, Bias-variance tuning without ground truth, Atoms and non-atomic measures
