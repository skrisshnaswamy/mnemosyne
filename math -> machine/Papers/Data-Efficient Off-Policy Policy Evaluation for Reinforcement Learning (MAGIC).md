---
title: "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning (MAGIC)"
authors: ["Thomas & Brunskill"]
year: 2016
arxiv: "1604.00923"
url: https://arxiv.org/abs/1604.00923
priority: Good-To-Read
read_on: 2026-09-24
tags: [paper, rl]
---
## The Core Idea

You have logs from an old policy. You want to know how good a *new* policy would be, without running it. That is off-policy policy evaluation ([[Off-Policy Evaluation]]).

Two families of answer existed. Both are bad in different ways.

**Importance sampling.** Reweight the logged rewards by how much more likely the new policy was to take those actions. Unbiased, but the weights are products over time steps, so the variance explodes.

**A learned model.** Fit an MDP from the logs, then compute the new policy's value inside that fitted world. Low variance, but if the model is wrong — function approximation, partial observability — it stays wrong forever. Bias that never shrinks.

The [[Doubly Robust Policy Evaluation and Learning|doubly robust]] estimator already mixed the two: use the model as a [[Policy Gradient#^baseline|baseline]] (control variate) inside the importance-sampling sum. This paper does two more things.

**One: weight the importance weights.** Divide each trajectory's weight by the *sum* of all weights instead of by $n$. That is self-normalisation, the same trick as [[The Self-Normalized Estimator for Counterfactual Learning (SNIPS) (NeurIPS)|SNIPS]]. It adds a little bias, kills a lot of variance. Call it WDR.

**Two, and this is the real idea: the model and importance sampling are not two options, they are the ends of a ruler.** Build a *$j$-step return*: use importance sampling for the first $j$ rewards of a trajectory, then hand over to the model for everything after. $j = -1$ is the pure model. $j = \infty$ is pure WDR. Everything in between is a partial estimator with intermediate bias and variance.

Now you have a whole vector of estimates, one per $j$. Take a weighted average of them, and choose the weights to directly minimise estimated mean squared error. That is MAGIC.

> [!NOTE] MAGIC
> Model And Guided Importance sampling Combining. Blend model-based and importance-sampling estimates by mixing $j$-step returns, with mixing weights chosen to minimise estimated MSE. ^magic-def

Why this did not exist before: people treated the bias–variance choice in OPE as a *selection* problem (pick DR or pick the model). The insight is that the choice is *continuous along the time axis of a trajectory*, and you can let the data pick the point. That matters most when the model is good in some parts of a trajectory and bad in others — e.g. a robot that is confused at the start of an episode and confident later.

## The Methodology

### Setup

Data is $D := \{(H_i, \pi_i)\}_{i=1}^n$: trajectories plus the known [[On-Policy vs Off-Policy|behaviour policies]] that made them. Evaluation policy $\pi_e$ is known. Goal: estimate $v(\pi_e) = \mathbb{E}[\sum_t \gamma^t R_t]$ with low MSE.

The importance weight for the first $t$ steps:

$$\rho_t = \prod_{i=0}^{t}\frac{\pi_e(A_i \mid S_i)}{\pi_b(A_i \mid S_i)}$$

An approximate model gives $\hat r^{\pi_e}(s,a,t)$ — its guess of the reward $t$ steps after taking $a$ in $s$. From those, $\hat q^{\pi_e}$ and $\hat v^{\pi_e}$.

### Doubly robust, written flat

Jiang & Li defined DR recursively and needed a finite, known horizon $L$. The paper re-derives it as a plain [[Monte Carlo Methods|Monte Carlo]] estimator with a control variate, which removes the horizon assumption and gives a readable form:

$$\mathrm{DR}(D) = \sum_{i=1}^n \sum_{t=0}^\infty \gamma^t w_t^i R_t^{H_i} - \sum_{i=1}^n \sum_{t=0}^\infty \gamma^t\Big(w_t^i \hat q^{\pi_e}(S_t^{H_i}, A_t^{H_i}) - w_{t-1}^i \hat v^{\pi_e}(S_t^{H_i})\Big)$$

with $w_t^i = \rho_t^i / n$.

> [!NOTE] Control variate
> If you want $\mathbb{E}[X]$ and you have $Y$ with $Y \approx X$ and known $\mathbb{E}[Y]$, then $X - Y + \mathbb{E}[Y]$ is still unbiased but has variance $\mathrm{Var}(X) + \mathrm{Var}(Y) - 2\mathrm{Cov}(X,Y)$. Subtract something correlated with your noise, add back its mean. ^control-variate

The derivation is a chain of these. Start with per-decision importance sampling. Subtract the model's prediction of each reward (weighted by $\rho_t$) — that is $Y$. Its true mean is unknown, so add back $Z$: the same prediction, but weighted by $\rho_{t-1}$ instead, which is lower variance because it does not include the *current* action. Now $Z$ itself has too much variance, so give *it* a control variate. Repeat until the control variate stops being random. Collapse the nested $\hat r$ sums into $\hat q$ and $\hat v$ and you get the equation above.

### WDR: change one denominator

$$w_t^i := \frac{\rho_t^i}{\sum_{j=1}^n \rho_t^j}$$

That is it. Since $\mathbb{E}[\rho_t] = 1$, the denominator converges to $n$, so the estimator is still strongly consistent — its MSE goes to zero almost surely. It is biased for finite $n$, and the bias has a readable shape: at $n=1$, $w_t^1 = 1$ for every $t$, so WDR is an unbiased estimate of the *behaviour* policy's value. As $n$ grows the estimate slides from $v(\pi_b)$ towards $v(\pi_e)$.

Two consistency theorems, both needing absolute continuity ($\pi_b(a|s) = 0 \Rightarrow \pi_e(a|s) = 0$):
- **Theorem 1**: one behaviour policy + finite horizon → consistent.
- **Theorem 2**: many behaviour policies + bounded importance weights ($\rho_t^i \le \beta < \infty$) → consistent. No bound on how large $\beta$ is.

### Why DR and WDR can still lose to the model

Rewrite WDR as:

$$\underbrace{\frac{1}{n}\sum_i \hat v^{\pi_e}(S_0^{H_i})}_{(a)} + \sum_i\sum_t \gamma^t w_t^i \underbrace{\Big[R_t - \hat q^{\pi_e}(S_t, A_t) + \gamma \hat v^{\pi_e}(S_{t+1})\Big]}_{(b)}$$

If the model is perfect, $(a)$ is a perfect low-variance estimate. And if transitions and rewards are *deterministic*, $(b)$ is exactly zero and WDR equals the model. But if transitions are stochastic, $(b)$ is not zero — and it gets multiplied by high-variance weights. A perfect model plus stochastic dynamics still gives you a noisy estimator.

This is the whole motivation for MAGIC.

### The $j$-step return

$$g^{(j)}(D) := \mathrm{IS}^{(j)}(D) + \mathrm{AM}^{(j+1)}(D)$$

Importance sampling predicts rewards $R_0 \ldots R_j$; the model predicts $R_{j+1}$ onwards. After the same control-variate derivation (Appendix F), it comes out as:

$$g^{(j)}(D) = \sum_{i}\sum_{t=0}^{j}\gamma^t w_t^i R_t^{H_i} + \sum_i \gamma^{j+1} w_j^i \hat v^{\pi_e}(S_{j+1}^{H_i}) - \sum_i \sum_{t=0}^{j}\gamma^t\Big(w_t^i \hat q^{\pi_e}(S_t, A_t) - w_{t-1}^i \hat v^{\pi_e}(S_t)\Big)$$

Read it as: importance-sampled rewards up to $j$, then the model's value of the state you landed in, with the DR control variate covering both halves. $g^{(-1)}$ is the pure model (AM). $g^{(\infty)}$ is WDR.

### Choosing the mixing weights

$\mathrm{BIM}(D) = x^\top g(D)$ where $g$ stacks the $j$-step returns for $j \in J$, a finite chosen set (always include $-1$ and $\infty$). Constrain $x$ to the simplex — non-negative, sums to one — which is both tractable and a form of [[Regularization|regularisation]].

Using the bias–variance decomposition of MSE:

$$\hat x^\star \in \arg\min_{x \in \Delta^{|J|}} x^\top\big[\hat\Omega_n + \hat b_n \hat b_n^\top\big]x$$

A small quadratic program over the simplex. $\hat\Omega_n$ is the covariance matrix of the different-length returns; $\hat b_n$ is their bias vector.

**Estimating $\Omega_n$.** Write $g^{(j)}(D) = \sum_i g_i^{(j)}(D)$ as a sum of per-trajectory pieces. These pieces are *not* independent — they share the weight denominator — but the dependence vanishes as $n \to \infty$. Pretend they are independent and take the sample covariance. (Bootstrap estimates gave similar results at much higher cost.)

**Estimating $b_n$ — the harder half.** Bias is distance from $v(\pi_e)$, which is exactly what you do not know. Using AM as the stand-in assumes AM's bias is zero, which is begging the question. Using WDR conflates WDR's variance with the bias you want.

The fix: build a 50% confidence interval on WDR's mean, and define the bias of return $j$ as its *distance from that interval*:

$$\hat b_n(j) := \mathrm{dist}\big(g^{(J_j)}(D), \mathrm{CI}(g^{(\infty)}(D), 0.5)\big)$$

Zero if the return falls inside the interval. Deliberately conservative at small $n$ — it *underestimates* bias early, which is right, because at small $n$ variance is what is killing you. The interval is the tighter of a percentile bootstrap ($\kappa = 200$ resamples, though 2000 is the usual advice) and Chernoff–Hoeffding. In practice it is almost always the bootstrap; Hoeffding is in there to make the proofs work.

**Theorem 4**: if at least one return in $J$ is consistent, and both $\hat\Omega_n$ and $\hat b_n$ are consistent, MAGIC is consistent.

## Ablation Studies and Experiments

Four domains, $\gamma = 1$, finite horizon. 128 trials per setting, MSE on log–log axes. Baselines: IS, PDIS, WIS, CWPDIS, DR, and AM (the pure model, elsewhere called the *direct method*).

The domains are built to be adversarial in specific ways — this is the useful part of the paper.

**ModelFail.** Three real states, but the agent observes only one. At $t=0$ the action decides whether you go up or down; at $t=1$ either action ends the episode with reward $+1$ (up) or $-1$ (down). To a model that cannot see the state, actions appear to do nothing. The fitted model says every policy is worth $0.38$; the truth ranges from $-0.5$ to $+0.5$. AM's MSE plateaus at a non-zero floor forever. WDR beats AM by **orders of magnitude** and DR by roughly one order.

**ModelWin.** Three *observable* states, stochastic transitions ($0.4/0.6$ splits), $L = 20$. The model converges to the truth fast. But because $\hat v(s_2) = \hat v(s_3)$, the DR control variate cannot predict which reward you got, so term $(b)$ is non-zero, and it is multiplied by 20-step importance weights. **AM beats WDR by about an order of magnitude.**

**Gridworld.** $4\times4$, $L = 100$, deterministic transitions and rewards. Two variants: TH (model told the true horizon) and FH (model told $L = 101$ — an artificial partial observability that ruins predictions near the end of trajectories).

- Gridworld-FH, $\pi_4 \to \pi_5$: WDR beats every other estimator by at least an order of magnitude.
- Gridworld-TH: DR, WDR and MAGIC **collapse exactly onto AM**. With deterministic dynamics and a tabular maximum-likelihood model, term $(b)$ is identically zero — the importance weights do nothing. Not a failure, but a sharp demonstration that the guided estimators only earn their keep under stochasticity or model error.

**Hybrid** — the domain that justifies the paper. ModelFail glued in front of ModelWin: partial observability early in each trajectory, resolving later. This is the shape of real robotics and tutoring problems. Here **MAGIC beats everything, including both AM and WDR**, because it can use importance sampling for the unmodellable prefix and the model for the modellable tail.

### What the ablations show

| Ablation | Finding |
|---|---|
| MAGIC-B ($J = \{-1, \infty\}$) | On Hybrid, MAGIC-B is markedly worse than full MAGIC. **Blending only AM and WDR is not enough** — the intermediate $j$-step returns are doing the work. This is the single most important ablation in the paper. |
| Half-data vs full-data split | Splitting $D$ into model-fitting and estimation halves preserves theoretical guarantees but **hurts empirically**. DR/WDR curves shift upward. Recommendation: reuse all data, accept that the proofs no longer strictly apply. |
| DR-v2 / WDR-v2 (replacing $\hat q(S_t,A_t)$ with $\hat r(S_t,A_t,0) + \gamma\hat v(S_{t+1})$) | **Did not outperform** the originals on these domains. Reported only in the supplementary spreadsheet. |
| Bootstrap vs sample covariance for $\hat\Omega_n$ | Similar performance, much higher cost. Sample covariance kept. |
| Alternative purely-model-based $\mathrm{AM}^{(j)}$ using $\hat d_0$ | Performed similarly; the importance-weighted version was kept. |

### What did not work cleanly

On **ModelWin, MAGIC does not perfectly track AM** — it drifts slightly above as $n$ grows. The authors attribute this to error in $\hat\Omega_n$ and $\hat b_n$. On a log scale the gap is small next to MAGIC's advantage over DR, but it is a real admission: the meta-estimator is only as good as its estimate of its own error.

MAGIC is also weakest at **small $n$**, exactly when you most want it: $\hat\Omega_n$ is noisy and the confidence interval feeding $\hat b_n$ is wide.

## Worth Remembering

**Unbiasedness is not the goal.** $\mathrm{MSE} = \mathrm{Var} + \mathrm{Bias}^2$. The paper argues explicitly that *strong consistency* — MSE converging almost surely to zero — is the right property to demand, and that insisting on zero bias at finite $n$ costs you real accuracy. If you need exact confidence bounds, unbiasedness matters; if you need the best point estimate for deciding whether to ship $\pi_e$, it does not.

**The deterministic-dynamics degeneracy is worth internalising.** If your environment is deterministic and your model is tabular and accurate, DR, WDR and MAGIC all *become* the model estimator. The importance-sampling machinery only pays for itself when there is stochasticity or model misspecification for it to correct.

**The bias-estimation trick generalises.** "I cannot measure bias without knowing the truth, so I will measure distance from a confidence interval around my consistent-but-noisy estimator, which is deliberately too small early and correct late" is a reusable pattern any time you are auto-tuning a bias–variance dial from data.

**Assumptions to check before you use it.**
- Absolute continuity. No deterministic behaviour policy. This is the same requirement as everywhere in OPE — see the logging discipline in [[Contextual Bandit#^log-the-propensity|log the propensity]].
- Behaviour policies known *exactly*. The authors list estimated propensities as open future work.
- Bounded importance weights, or a single behaviour policy plus finite horizon.
- Stationary transitions and rewards. Non-stationarity is listed as unsolved.

**Practical caveats.** $J$ must be finite; with a finite horizon $L$ the paper uses all of $\{-1,\dots,L\}$, which means $O(L)$ returns and an $L \times L$ covariance matrix per estimate — this does not scale gracefully to very long horizons. The quadratic program was solved with Gurobi. Use all your data for both the model and the estimate.

**Open questions the authors raise.** If the sample mean importance weight is near zero, the sample covariance matrix *fails to capture* the resulting variance — a known blind spot in $\hat\Omega_n$. Also: with several candidate models available, is there a way to pick which pairs best with WDR? And does consistency survive fitting the model on the same data used for estimation?

**Connections.** The $j$-step return is structurally the off-policy cousin of the $\lambda$-return in [[Temporal Difference Learning]]; the paper is explicit that the $\lambda$-, $\gamma$- and $\Omega$-returns were all candidate weighting schemes and all were rejected because none guarantee consistency as $n \to \infty$ (they were designed for a single trajectory, on-policy). The DR control variate is the same object as the [[Policy Gradient#^baseline|baseline]] in policy gradients and the advantage-sum estimator from poker agent evaluation. WDR's self-normalisation is the sequential version of [[The Self-Normalized Estimator for Counterfactual Learning (SNIPS) (NeurIPS)|SNIPS]].

## Links

Related: [[Off-Policy Evaluation]] · [[Doubly Robust Policy Evaluation and Learning]] · [[Doubly Robust Off-policy Value Evaluation for Reinforcement Learning]] · [[The Self-Normalized Estimator for Counterfactual Learning (SNIPS) (NeurIPS)]] · [[Eligibility Traces for Off-Policy Policy Evaluation (ICML)]] · [[On-Policy vs Off-Policy]] · [[Offline RL]] · [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] · [[Counterfactual Risk Minimization]] · [[Temporal Difference Learning]] · [[Monte Carlo Methods]] · [[Markov Decision Process]] · [[Policy Gradient]] · [[Value Function]] · [[Contextual Bandit]] · [[POMDP]] · [[Model-Based vs Model-Free RL]] · [[Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms]] · [[Open Bandit Dataset and Pipeline- Towards Realistic and Reproducible Off-Policy Evaluation]] · [[Uncertainty]]

New topics worth writing: Control variates for variance reduction, Self-normalised importance sampling, Strong consistency vs unbiasedness for estimators, The Ω-return and γ-return, Percentile bootstrap confidence intervals, Horizon-free OPE estimators, Partial observability and model misspecification in OPE
