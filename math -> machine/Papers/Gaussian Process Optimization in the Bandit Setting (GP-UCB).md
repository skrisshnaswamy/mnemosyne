---
title: "Gaussian Process Optimization in the Bandit Setting (GP-UCB)"
authors: ["Niranjan Srinivas", "Andreas Krause", "Sham M. Kakade", "Matthias Seeger"]
year: 2010
arxiv: "0912.3995"
url: https://arxiv.org/abs/0912.3995
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, rl, optimization, theory]
---
## The Core Idea

You want to find the highest point of an unknown function $f$. Each measurement is expensive and noisy. You cannot try everything. This is Bayesian optimisation, and by 2010 people had good heuristics for it — Expected Improvement, Most Probable Improvement, EGO — but **nobody had proved any of them converge at a rate**. You could run them, but you could not say "after $T$ samples I am within $\epsilon$ of the best."

This paper closes that gap. It analyses **GP-UCB**: model $f$ with a Gaussian process, and at each step pick the point with the highest optimistic estimate,

$$x_t = \arg\max_{x \in D} \; \mu_{t-1}(x) + \beta_t^{1/2}\,\sigma_{t-1}(x).$$

That rule is the obvious generalisation of [[Finite-time Analysis of the Multiarmed Bandit Problem (UCB1)|UCB1]] and of [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)|LinUCB]] to a continuous, non-linear world. The rule was not new. The **proof** is new, and the key move in the proof is the new idea:

> [!NOTE] Maximum information gain $\gamma_T$
> $\gamma_T = \max_{A \subset D, |A| = T} I(y_A; f_A)$ — the most you could possibly learn about $f$ (in bits, [[KL Divergence|mutual information]] sense) from any $T$ noisy measurements. For a GP this has a closed form: $I(y_A;f_A) = \tfrac12 \log|I + \sigma^{-2} K_A|$, where $K_A$ is the kernel matrix on those points. ^information-gain

The theorem is: cumulative regret is bounded by $O^*(\sqrt{T \beta_T \gamma_T})$. In words: **how badly a bandit algorithm can do is controlled by how much there is to learn.** If the function is smooth, there is little to learn, $\gamma_T$ grows slowly, and regret is sublinear.

Why this did not exist before: earlier bandit theory needed a *finite dimension* $d$ to plug into the bound. A GP is a random function in an infinite-dimensional space, so $d = \infty$ and every bound blew up. Replacing "dimension" with "information gain" is the trick that makes an infinite-dimensional problem have a finite bound. And $\gamma_T$ is not an abstract quantity — the authors compute it explicitly from the kernel's eigenvalue decay.

This paper also quietly connects two fields that had ignored each other: **bandits** (maximise reward) and **Bayesian experimental design** (learn the function fast). The same quantity governs both.

## The Methodology

**The model.** $f: D \to \mathbb{R}$, observations $y_t = f(x_t) + \epsilon_t$ with $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$. Prior $f \sim \mathcal{GP}(0, k)$ with $k(x,x) \le 1$. The posterior after $T$ points has closed form:

$$\mu_T(x) = k_T(x)^\top (K_T + \sigma^2 I)^{-1} y_T, \qquad \sigma_T^2(x) = k(x,x) - k_T(x)^\top (K_T + \sigma^2 I)^{-1} k_T(x).$$

Note $\sigma_T$ does *not* depend on the observed $y$ values — only on *where* you sampled. That fact matters later.

**The algorithm.** One line per round: pick $\arg\max \mu_{t-1}(x) + \sqrt{\beta_t}\,\sigma_{t-1}(x)$, sample, update the posterior. That is the whole thing. Maximising the UCB index is itself a non-convex problem, but the assumption is that evaluating $f$ costs far more than a global search over the cheap surrogate.

**The proof, in four steps.** This is worth internalising because the shape recurs everywhere in bandit theory.

1. *Confidence.* With $\beta_t = 2\log(|D| t^2 \pi^2 / 6\delta)$ and a union bound over the $|D|$ arms and over time, $|f(x) - \mu_{t-1}(x)| \le \beta_t^{1/2}\sigma_{t-1}(x)$ for all $x, t$ with probability $\ge 1-\delta$.
2. *Optimism gives per-step regret.* If the confidence bound holds, then because $x_t$ maximises the index, the instantaneous regret is $r_t \le 2\beta_t^{1/2}\sigma_{t-1}(x_t)$. Regret is bounded by your own uncertainty at the point you chose.
3. *Sum of variances = information gain.* This is the neat identity:
$$I(y_T; f_T) = \tfrac12 \sum_{t=1}^{T} \log\!\left(1 + \sigma^{-2}\sigma_{t-1}^2(x_t)\right).$$
Since $s^2 \le C_2 \log(1+s^2)$ on the relevant range, $\sum_t \sigma_{t-1}^2(x_t)$ is bounded by $\gamma_T$ up to constants.
4. *Cauchy–Schwarz.* $R_T^2 \le T\sum_t r_t^2 \le C_1 T \beta_T \gamma_T$, with $C_1 = 8/\log(1+\sigma^{-2})$.

**Three settings, three theorems.**
- *Theorem 1* — finite $D$, $f$ drawn from the known GP: $R_T \le \sqrt{C_1 T \beta_T \gamma_T}$, i.e. $O^*(\sqrt{T\gamma_T \log|D|})$.
- *Theorem 2* — $D \subset [0,r]^d$ compact and convex. You cannot union-bound over infinitely many points, so they build a **fake discretisation used only in the analysis**: a grid $D_t$ of size $(\tau_t)^d$ with $\tau_t = dt^2br\sqrt{\log(2da/\delta)}$, fine enough that $|f(x^*) - f([x^*]_t)| \le 1/t^2$. The extra error sums to $\sum 1/t^2 = \pi^2/6$, hence the "+2" in the bound. This needs the kernel to have well-behaved sample-path derivatives: $\Pr\{\sup_x |\partial f/\partial x_j| > L\} \le a e^{-(L/b)^2}$, true for stationary kernels four-times differentiable.
- *Theorem 3* — **agnostic case**. $f$ is any fixed function with RKHS norm $\|f\|_k^2 \le B$, and the noise is only a bounded martingale difference sequence, not Gaussian. The algorithm still runs the (now misspecified) GP model. With $\beta_t = 2B + 300\gamma_t \log^3(t/\delta)$ you get $O^*(\sqrt{T}(B\sqrt{\gamma_T} + \gamma_T))$. The proof needs Freedman's Bernstein inequality for martingales and an inductive "escape event" argument — Hoeffding–Azuma alone only gives $T^{3/4}$, too weak.

> [!NOTE] RKHS vs GP sample
> These two settings do not contain each other. If $f \sim \mathcal{GP}(0,k)$ then $\|f\|_k = \infty$ almost surely — GP sample paths are *rougher* than the functions in the corresponding RKHS. ^rkhs-vs-gp

**Bounding $\gamma_T$ — the second half of the paper.** Finding the best $T$ points is NP-hard, but $F(A) = I(y_A;f)$ is **submodular** (diminishing returns) and monotone, so greedy is within $(1-1/e)$ of optimal. The authors use this *backwards*: instead of "greedy is near-optimal," they say "the optimum is at most $(1-1/e)^{-1}$ times greedy," which turns $\gamma_T$ into something computable. Greedy here is the pure-exploration rule $x_t = \arg\max \sigma_{t-1}(x)$ — which, remember, ignores the observed $y$ entirely.

Then relax further: let the greedy step pick any unit vector, not just an indicator of a point. Those picks become the leading eigenvectors of $K_D$. So

$$\gamma_T \le \frac{1/2}{1-e^{-1}} \max_{(m_t)} \sum_t \log(1 + \sigma^{-2} m_t \hat\lambda_t), \quad \sum_t m_t = T.$$

**Everything reduces to how fast the kernel's eigenvalues decay.** Splitting the sum at $T_*$ gives $\gamma_T = O(\sigma^{-2}[B(T_*)T + T_*\log n_T T])$ where $B(T_*)$ is the eigenvalue tail sum. Plugging in known operator spectra:

| Kernel | Spectrum | $\gamma_T$ | Regret $R_T$ |
|---|---|---|---|
| Linear, $d$-dim | finite | $O(d\log T)$ | $O^*(\sqrt{dT})$ |
| Squared Exponential | $\lambda_s \le cB^{s^{1/d}}$, exponential | $O((\log T)^{d+1})$ | $O^*(\sqrt{T}(\log T)^{\frac{d+1}{2}})$ |
| Matérn ($\nu>1$) | $\lambda_s \le cs^{-(2\nu+d)/d}$, power law | $O(T^{\frac{d(d+1)}{2\nu+d(d+1)}}\log T)$ | $O^*(T^{1-\eta})$, $\eta = \frac{\nu}{2\nu+d(d+1)}$ |

The headline: for the Squared Exponential kernel the **dimension $d$ only appears in the exponent of $\log T$**, not of $T$. Extreme smoothness beats the curse of dimensionality. Compare Kleinberg et al.'s Lipschitz-only bound of $\Omega(T^{(d+1)/(d+2)})$, which degrades to nearly linear regret as $d$ grows.

## Ablation Studies and Experiments

Three datasets, five methods. Metric is mean *average* regret $R_T/T$.

**Methods compared:** GP-UCB, Expected Improvement (EI), Most Probable Improvement (MPI), "mean only" (pure exploitation, $\arg\max\mu_{t-1}$), "variance only" (pure exploration, $\arg\max\sigma_{t-1}$ — this is the experimental-design rule).

1. **Synthetic**: functions sampled from an SE kernel, lengthscale $0.2$, $D=[0,1]$ discretised to 1000 points, $\sigma^2 = 0.025$ (5% of signal variance), $T=1000$, $\delta = 0.1$, 30 trials.
2. **Temperature**: 46 sensors at Intel Research Berkeley, 5 days at 1-minute resolution. First two-thirds of the data gives the empirical covariance, used directly as the kernel matrix. Test on the remaining third. $T=46$, $\sigma^2=0.5$, averaged over 2000 objective functions.
3. **Traffic**: 357 sensors on highway I-880 South, weekday 6–11 AM for a month; goal is to *minimise* speed (find the worst congestion). $T=357$, $\sigma^2=4.78$, 900 runs.

**Results.** On temperature data GP-UCB and EI clearly beat the rest and are statistically indistinguishable from each other. On synthetic and traffic data MPI joins them at the top. The two naive baselines — mean-only and variance-only — are consistently worse everywhere. So the honest summary is: **GP-UCB is at least on par with the heuristics, and it is the only one with a regret bound.**

**What did not work / what the ablations reveal:**

- **The theoretical $\beta_t$ is too conservative.** Running GP-UCB with $\beta_t$ exactly as Theorem 1 prescribes is competitive, but cross-validation found that **dividing $\beta_t$ by 5** improves it. The authors did not optimise the constants in the bound. Practically, treat $\beta$ as a tunable exploration knob, not a formula.
- **Pure exploitation gets stuck.** The "mean only" baseline confirms the intuition that $\arg\max\mu_{t-1}$ is "too greedy too soon" and lands in shallow local optima.
- **Pure exploration is wasteful.** The "variance only" rule is provably near-optimal for *learning* $f$ globally (that is the $(1-1/e)$ submodularity guarantee) and yet it is a poor optimiser. Being the best possible experiment designer is not the same as being a good bandit. This is the sharpest conceptual result in the experiments section.
- **Ornstein–Uhlenbeck (Matérn $\nu=1/2$) breaks Theorem 2.** Sample paths are nowhere differentiable with independent increments, the derivative assumption fails, and the authors *conjecture the theorem is simply false* there. Not a proof gap they patched — an admitted limit.
- The spectral-decay figure shows the ordering directly: independent (diagonal) kernel has flat spectrum and the worst $\gamma_T$ bound; SE has the fastest decay and the best. Matérn and linear sit between.

## Worth Remembering

- **The reusable lesson**: whenever you need a bandit bound in a non-parametric setting, look for a quantity that plays the role of "dimension." Here it is $\gamma_T$, and it is computable from kernel eigenvalues. The proof skeleton — confidence bound $\to$ optimism gives $r_t \le 2\beta^{1/2}\sigma_{t-1}$ $\to$ sum of posterior variances is an information gain $\to$ Cauchy–Schwarz — is the standard recipe now.
- **$\gamma_T$ can be computed numerically for your specific problem.** Because the greedy variance-maximising rule needs no observations, you can run it offline on your $D$ and kernel before spending a single real evaluation, and get a problem-specific regret bound. Rarely done in practice but genuinely available.
- **Bounds on $R_T/T$ convert to optimisation convergence rates**: $\max_{t\le T} f(x_t)$ is at least as good as the average, so $f(x^*) - \max_{t\le T}f(x_t) \le R_T/T$. That is the bridge from bandit language to "how close am I to the optimum."
- **Practical caveats.** (i) You need $\|f\|_k \le B$ known in the agnostic case, though guess-and-doubling suffices. (ii) The GP posterior costs $O(t^3)$ to update naively — fine for $T$ in the hundreds, a real problem beyond. (iii) The kernel and its hyperparameters are assumed *known and correct*; the theory says nothing about learning the lengthscale online, which is what everyone actually does and which can break the confidence intervals. (iv) $\beta_t$ grows like $\log|D|$ or $d\log t$, so the bound degrades if your candidate set is huge.
- **Connection to [[Practical Bayesian Optimization of Machine Learning Algorithms]]**: that paper uses EI, a heuristic that here has no bound but empirically ties GP-UCB. The two are near-substitutes in practice; GP-UCB gives you the theory, EI gives you a hyperparameter-free acquisition.
- **Compare with [[A Tutorial on Thompson Sampling]]**: TS is the randomised alternative to this deterministic optimism, and can be far easier when the posterior maximisation is awkward. GP-TS bounds came later, largely by reusing $\gamma_T$ from this paper.
- Open follow-ups: batch/parallel GP-UCB (you rarely evaluate one point at a time), contextual GP-UCB, and the fact that $\gamma_T$ for Matérn was later tightened — the $d(d+1)$ in the exponent here is loose.

## Links

Related: [[Finite-time Analysis of the Multiarmed Bandit Problem (UCB1)]] · [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]] · [[A Tutorial on Thompson Sampling]] · [[An Empirical Evaluation of Thompson Sampling (NeurIPS)]] · [[Practical Bayesian Optimization of Machine Learning Algorithms]] · [[Uncertainty]] · [[KL Divergence]] · [[Random variable]] · [[Decision Sciences]] · [[Beliefs]] · [[Kalman Filter]]

New topics worth writing: Gaussian Processes, Reproducing Kernel Hilbert Space, Mutual Information, Submodularity and greedy maximisation, Mercer's theorem and kernel operator spectra, Expected Improvement, Martingale concentration (Freedman's inequality), Bayesian experimental design
