---
title: "A Tutorial on Thompson Sampling"
authors: ["Russo", "Van Roy", "Kazerouni", "Osband & Wen"]
year: 2018
arxiv: "1707.02038"
url: https://arxiv.org/abs/1707.02038
priority: Must-Read
read_on: 2026-08-25
tags: [paper, rl, optimization, theory]
---
## The Core Idea

Thompson sampling (TS) is one line of code that solves the explore/exploit problem: **sample a model from your posterior, then act as if that sample were the truth.**

That is the whole algorithm. You keep a probability distribution over the unknown parameters $\theta$ of the world. Each round you draw one random $\hat\theta \sim p(\theta \mid \text{history})$, solve the *deterministic* optimisation problem "what is the best action if $\theta = \hat\theta$?", play it, observe the outcome, and update the posterior with Bayes' rule.

Why is that clever? Because the probability TS picks action $k$ is exactly the posterior probability that $k$ is the optimal action. So the algorithm automatically spends exploration effort in proportion to *how plausible it is that an action is best* — no more, no less.

The tutorial's motivating picture: three Bernoulli arms with posteriors after 1000 pulls of arm 1 (600 wins), 1000 of arm 2 (400 wins), and 3 pulls of arm 3 (1 win). Beliefs are $\text{beta}(601,401)$, $\text{beta}(401,601)$, $\text{beta}(2,3)$.

- **Greedy** picks arm 1 forever. It never learns whether arm 3 (mean 0.4 but wildly uncertain) is secretly better.
- **$\epsilon$-greedy** splits its exploration budget equally: arm 2 and arm 3 each get half. But arm 2 has essentially zero chance of being optimal — that half is pure waste. With $K$ arms the waste grows like $K$.
- **Thompson sampling** picks arms 1, 2, 3 with probability $\approx 0.82$, $\approx 0$, $\approx 0.18$. It writes off arm 2 completely and probes arm 3 exactly as much as arm 3 deserves.

> [!NOTE] Thompson sampling
> Choose the action that is optimal for one random draw from the posterior over models. Equivalently: choose each action with probability equal to the posterior probability that it is the best action. Also called *posterior sampling* or *probability matching*. ^thompson-sampling

The idea is from 1933 (Thompson, on clinical trials) and was ignored for eighty years. It came back because (a) the internet made cheap, high-volume, per-user experiments normal, and (b) Chapelle & Li's 2011 empirical paper showed it beats tuned UCB on real ad data.

What it unlocks that [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)|UCB]] does not: **you never have to design a confidence bound.** With complicated action spaces — 184,756 paths through a graph, an assortment of 6 substitutable products, a neural network's weight space — writing down a tight, computable upper confidence bound is hard, and a loose bound directly costs you regret. TS only needs a posterior sample plus an ordinary optimiser. The confidence bound still appears — but only inside the *proof*, never inside the code.

---

## The Methodology

**The general algorithm.** Actions $x_t \in \mathcal{X}$, outcomes $y_t \sim q_\theta(\cdot \mid x_t)$, reward $r_t = r(y_t)$ with $r$ a known function. Prior $p$ over $\theta$.

Greedy and TS differ in **one line**:

| | model used | action |
|---|---|---|
| Greedy | $\hat\theta \leftarrow \mathbb{E}_p[\theta]$ | $\arg\max_x \mathbb{E}_{q_{\hat\theta}}[r(y_t)\mid x_t=x]$ |
| **TS** | $\hat\theta \sim p$ | $\arg\max_x \mathbb{E}_{q_{\hat\theta}}[r(y_t)\mid x_t=x]$ |

Then both update $p \leftarrow \mathbb{P}_{p,q}(\theta \in \cdot \mid x_t, y_t)$, i.e.
$$\mathbb{P}(\theta = u \mid x_t,y_t) = \frac{p(u)\,q_u(y_t\mid x_t)}{\sum_v p(v)\,q_v(y_t\mid x_t)}.$$

A common misreading: TS does **not** sample a plausible *observation* $y$. It samples a plausible *parameter* — a plausible click-through rate, not a plausible click.

**Beta-Bernoulli bandit.** Prior $\theta_k \sim \text{beta}(\alpha_k,\beta_k)$. Draw $\hat\theta_k \sim \text{beta}(\alpha_k,\beta_k)$ for each $k$, play $\arg\max_k \hat\theta_k$, then
$$(\alpha_{x_t},\beta_{x_t}) \leftarrow (\alpha_{x_t}+r_t,\ \beta_{x_t}+1-r_t).$$
Only the played arm updates. $\alpha,\beta$ are "pseudo-counts" of wins and losses; $\alpha=\beta=1$ is the uniform prior. Conjugacy means the whole posterior is two numbers per arm.

**Online shortest path (the structure showcase).** Graph $G=(V,E)$, mean edge delays $\theta_e$, action = a source-to-sink path, cost = sum of realised edge times. Prior: $\ln\theta_e \sim N(\mu_e,\sigma_e^2)$ (log-Gaussian, so delays stay positive), observation $y_{t,e}$ also log-Gaussian with mean $\theta_e$. Update is closed form, a precision-weighted average:
$$(\mu_e,\sigma_e^2)\leftarrow\left(\frac{\tfrac{1}{\sigma_e^2}\mu_e+\tfrac{1}{\tilde\sigma^2}(\ln y_{t,e}+\tfrac{\tilde\sigma^2}{2})}{\tfrac{1}{\sigma_e^2}+\tfrac{1}{\tilde\sigma^2}},\ \frac{1}{\tfrac{1}{\sigma_e^2}+\tfrac{1}{\tilde\sigma^2}}\right).$$
The key point: with $\hat\theta$ in hand, "pick the best path" is just **Dijkstra**. There are 184,756 paths in the 20-stage binomial bridge they simulate, and TS never enumerates them.

A nice modelling touch: the commuter knows the *distance* $d_e$ of each road, so she sets $\mu_e = \ln(d_e)-\sigma_e^2/2$, giving $\mathbb{E}[\theta_e]=d_e$. Prior knowledge goes straight into the algorithm.

**Correlated version.** $y_{t,e} = \zeta_{t,e}\,\eta_t\,\nu_{t,\ell(e)}\,\theta_e$, where $\eta_t$ is a whole-city shock (weather), $\nu_{t,0},\nu_{t,1}$ are upper/lower-half shocks, $\zeta_{t,e}$ is edge noise. Working in $\phi_e=\ln\theta_e$ makes everything jointly Gaussian, and the posterior updates as
$$(\mu,\Sigma)\leftarrow\left((\Sigma^{-1}+\tilde C)^{-1}(\Sigma^{-1}\mu+\tilde C z_t),\ (\Sigma^{-1}+\tilde C)^{-1}\right)$$
with $\tilde C$ the inverse observation-covariance padded with zeros off the traversed path. This is [[Kalman Filter|Kalman-filter]]-shaped algebra.

### When you cannot do exact Bayes

Four approximate-sampling options, all illustrated on a **binary-feedback** path problem ($\theta_e \sim$ Gamma, $y_t=1$ with probability $\sigma(M-\sum_{e\in x_t}\theta_e)$ — a logistic thumbs-up on the route):

1. **Gibbs sampling.** Cycle through coordinates, sample $\theta_k$ from its 1-D conditional. Accurate, general, but too slow to run thousands of simulations. See [[Markov Chain Monte Carlo]].
2. **Laplace approximation.** Find the posterior mode $\bar\theta$, then approximate with $N(\bar\theta, C^{-1})$ where $C=-\nabla^2\ln f_{t-1}(\bar\theta)$ — the negative [[Derivative#Hessian|Hessian]] of the log-posterior. Cheap when the log-posterior is log-concave. This is what Chapelle & Li used for logistic ad CTR.
3. **Langevin Monte Carlo.** Simulate $d\phi_t=\nabla\ln g(\phi_t)dt+\sqrt{2}dB_t$, discretised as
$$\phi_{n+1}=\phi_n+\epsilon A\nabla\ln g(\phi_n)+\sqrt{2\epsilon}A^{1/2}W_n.$$
Gradient ascent on log-density plus injected Gaussian noise. Two practical fixes mattered a lot: **stochastic gradients** from minibatches of 100 observations, and a **preconditioner** $A = -(\nabla^2\ln g)^{-1}$ at the mode, because the log-posterior becomes badly conditioned in later rounds. Chain initialised at the mode.
4. **Bootstrap.** Resample $t-1$ action-outcome pairs with replacement into a fake history $\hat{\mathbb{H}}_{t-1}$, draw $\theta^0$ from the prior, then
$$\hat\theta=\arg\max_\theta\ e^{-(\theta-\theta^0)^\top\Sigma(\theta-\theta^0)}\hat L_{t-1}(\theta).$$
The random prior draw supplies the exploration early (when a plain bootstrap would badly *underestimate* uncertainty); the resampled likelihood dominates later. Nonparametric, but with no theory behind it.

**Incremental versions.** All four have per-round cost growing with $t$, because they touch every past observation. Two fixes:

- *Online Newton / extended Kalman*: keep $H_t = H_{t-1}+\nabla^2 g_t(\bar\theta_{t-1})$, $\bar\theta_t=\bar\theta_{t-1}-H_t^{-1}\nabla g_t(\bar\theta_{t-1})$, sample $\hat\theta\sim N(\bar\theta_{t-1},H_{t-1}^{-1})$. For generalised linear models $\nabla^2 g_t$ is rank one, so Sherman–Morrison updates $H_t^{-1}$ in place.
- *Ensemble sampling*: keep $N$ models, each fit to a differently-perturbed version of the data (weights $z_t^n\sim\text{Poisson}(1)$, which approximates a bootstrap resample). Sample $n$ uniformly each round, act greedily under model $n$.

### Practical variants

- **Contexts.** Observe $z_t$ before acting. Treat $\tilde x_t=(x_t,z_t)$ as the action, constrained to $\mathcal{X}_t=\{(x,z_t):x\in\mathcal{X}\}$. Same algorithm.
- **Constraints / caution.** Just shrink $\mathcal{X}_t$. To guarantee expected reward $\geq \underline r$, set $\mathcal{X}_t=\{x:\mathbb{E}[r_t\mid x_t=x]\geq \underline r\}$. Actions banned today can be re-admitted tomorrow if related experiments raise their estimate.
- **Nonstationarity.** Decay toward a stationary prior each round:
$$(\alpha_k,\beta_k)\leftarrow((1-\gamma)\alpha_k+\gamma\bar\alpha + r_t,\ (1-\gamma)\beta_k+\gamma\bar\beta+1-r_t).$$
$\gamma$ controls how fast uncertainty is re-injected; $\gamma=0$ is standard TS, $\gamma=1$ forgets everything. General form: $p(u)\leftarrow \bar p^\gamma(u)p^{1-\gamma}(u)q_u(y_t\mid x_t)/Z$. The agent then never stops exploring, which is what you want when the world drifts.
- **Concurrency.** $K$ agents acting in the same round? Just draw $K$ independent posterior samples. Regret per agent falls faster in *wall-clock time* but slower per *action taken*, because the batch does not learn from itself.

---

## Ablation Studies and Experiments

**Three-arm Bernoulli, $\theta=(0.9,0.8,0.7)$, uniform priors, 10,000 sims × 1000 periods.** Greedy sometimes locks onto arm 3 forever: two lucky wins early make its estimate 0.7 while the untouched arms sit at 0.5, and the trap self-reinforces. Greedy's per-period regret plateaus at a positive value. TS's per-period regret goes to zero and it picks arm 1 in almost every final period.

**20-stage binomial bridge, independent delays.** TS regret converges to ~0 fast. $\epsilon$-greedy learns far slower at every $\epsilon$ tried, because a random path is an uninformative path — it wastes trials on edges already well understood. Cumulative-travel-time-vs-optimal converges to 1 for TS, not for $\epsilon$-greedy.

**Correlated delays: coherent vs. misspecified TS.** Run TS pretending edges are independent (the Example 4.0.1 model) on data actually generated with shared weather/half-bridge shocks. Coherent TS — which models the correlation — is substantially better. **Modelling the information structure is where the gain is**, not the sampling rule.

**Misspecified prior.** True arm means drawn from $\text{beta}(1,50)$, $\text{beta}(1,100)$, $\text{beta}(1,200)$. Running TS with a uniform prior instead measurably increases regret and visibly delays the separation of the three arms' posterior means. The prior is not cosmetic.

**Approximation sanity checks (Figure 5.2).** On problems where exact TS is computable, Laplace / Langevin / bootstrap all track exact TS closely — Laplace slightly worse on the Bernoulli bandit, bootstrap slightly worse on the shortest path. On the harder binary-feedback problem, **Langevin clearly beats Laplace and bootstrap**. The authors' read: the posterior there is just not Gaussian enough, and "despite serving as a popular approach in practical applications of TS, the Laplace approximation can leave substantial value on the table."

A cute detail: the Laplace approximation is *not* invariant to change of variables. On the log-Gaussian path problem, applying Laplace to $\phi=\ln\theta$ makes it **exact**, because $\phi$ really is Gaussian. Pick your parameterisation.

**News recommendation** ($K=3$ articles, $d=7$ binary user features, logistic model $g(z_t^\top\theta_x)$, $\theta_x\sim N(0,I)$, 2000 instances). Best $\epsilon$-greedy was $\epsilon=0.01$ after a search — and even tuned, it is *substantially* outperformed by both Laplace-TS and Langevin-TS. Related: [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]].

**Product assortment** ($n=6$ products, $\sigma^2=0.04$, profit $p_i=1/6$, log-Gaussian demand $\log d_i \mid \theta,x \sim N((\theta x)_i,\sigma^2)$ with off-diagonal $\theta_{ij}$ = cross-product effects; prior variance 1 on diagonal, 0.2 off-diagonal). Greedy is bad. Fixed $\epsilon$-greedy at its best ($\epsilon=0.07$) is much better. **Annealing $\epsilon=\frac{m}{m+t}$ with $m=9$ is better still** — a useful reminder that annealed dithering is a real baseline. TS still wins.

**Cascading bandits — the one place UCB sometimes wins.** User scans a ranked list of $J$ items, clicks the first attractive one; $\mathbb{E}[r_t\mid x_t,\theta]=1-\prod_{j=1}^J(1-\theta_{x_j})$.
- $K=1000$, $J=100$, $\theta_k\sim\text{Beta}(1,40)$, 20,000 periods: **CascadeTS crushes CascadeUCB1**. Reason: $h(x,U_t)$ assumes *every one of the 100 items simultaneously* sits at its optimistic bound — absurdly optimistic. TS samples each $\theta_k$ independently, so the sampled list value has realistic spread. Even UCB tuned to $c=0.05$ (optimised for this exact horizon) still loses.
- $K=50$, $J=10$, tuned $c=0.1$: **UCB now beats TS.** The explanation is geometric — UCB uses hyper-rectangular confidence sets (a box, one interval per item) while the Bayesian CLT says the right shape is an ellipsoid. Boxes approximate ellipsoids badly, and the mismatch worsens with dimension. So the TS advantage grows with $J$.

**Active learning with a 2-layer ReLU net** ($M=100$ input dim, $D=50$ hidden, $K=100$ actions, prior variance $\lambda=1$, noise $\sigma_z^2=100$; 5 SGD steps per round, lr $10^{-3}$, minibatch 64; leaky ReLU $\max(0.01x,x)$ used during *training* for gradient flow even though the target net is plain ReLU). Fixed $\epsilon$-greedy < annealed $\epsilon=\frac{k}{k+t}$ < **ensemble TS**. As few as **30 ensemble members** suffice. Smaller ensembles act greedier: better on short horizons, prone to premature convergence later. See [[Backpropagation]], [[Regularization]].

**RL in MDPs: sample per *episode*, not per timestep.** In the chain MDP $\{s_{-N},\dots,s_N\}$ where only the far left or far right end pays off, resampling $\hat\theta$ every timestep gives a random walk — expected $2^N$ episodes before the first reward. Sampling **once per episode** and holding the policy fixed learns the optimal policy in a *single* episode with an informed prior, and still clearly wins with a uniform Dirichlet prior over transitions plus Gaussian reward priors. This is PSRL, and the property it preserves is called **deep exploration**.

> [!NOTE] Deep exploration
> Committing to a coherent, temporally-extended plan long enough to find out whether it works. Randomising at every timestep destroys it — the noise cancels out over the episode. ^deep-exploration

---

## Worth Remembering

**Why the theory works.** For *any* function $U_t$ measurable w.r.t. history, TS satisfies $\mathbb{E}[U_t(x_t)]=\mathbb{E}[U_t(x^*)]$ — because $x_t$ and $x^*$ have the same posterior distribution. So
$$\mathbb{E}[\mu(x^*,\theta)-\mu(x_t,\theta)] = \underbrace{\mathbb{E}[\mu(x^*,\theta)-U_t(x^*)]}_{\text{pessimism} \leq 0} + \underbrace{\mathbb{E}[U_t(x_t)-\mu(x_t,\theta)]}_{\text{width}}.$$
Every UCB regret bound converts into a TS Bayesian-regret bound. **But $U_t$ never enters the algorithm** — so TS's regret is governed by the *best possible* confidence bound, while a UCB algorithm's regret is governed by the one you actually implemented. That is the crisp statement of TS's advantage.

**Numbers to keep:**
- Asymptotically optimal for Bernoulli: $\lim_{T\to\infty}\frac{\mathbb{E}[\text{Regret}(T)\mid\theta]}{\log T}=\sum_{k\neq k^*}\frac{\theta_{k^*}-\theta_k}{d_{\text{KL}}(\theta_{k^*}\|\theta_k)}$, matching the Lai–Robbins lower bound. ([[KL Divergence]])
- Worst-case Bernoulli: $O(\sqrt{KT\log T})$, near the $\Omega(\sqrt{KT})$ floor.
- Linear bandits: $\mathbb{E}[\text{Regret}(T)]=O(d\sqrt T\log T)$ — depends on **dimension, not number of actions**. Meaningful even with infinitely many actions.
- General: $\tilde O\big(\sqrt{\dim_E(\mathcal{F},T^{-2})\cdot\log N(\mathcal{F},T^{-2},\|\cdot\|_\infty)\cdot T}\big)$. The **eluder dimension** is a genuinely new complexity measure — VC dimension and covering numbers are *not* enough for online decision problems.
- Information-theoretic: with information ratio $\Gamma_t=\frac{(\mathbb{E}[\mu(x^*,\theta)-\mu(x_t,\theta)])^2}{I(x^*;(x_t,y_t)\mid\mathbb{H}_{t-1})}$ ("regret² per bit"), $\mathbb{E}[\text{Regret}(T)]\leq\sqrt{\bar\Gamma H(x^*)T}$. For the shortest path with $d$ edges, $\bar\Gamma\leq d/2$, so regret $\leq\sqrt{dH(x^*)T/2}$ — depends on edges, not paths, **and on the entropy of your prior over which path is best**. A good prior is worth real regret. If you observe every edge's delay, $\bar\Gamma\leq 1/2$.

**Randomisation is necessary.** For any *stationary deterministic* rule there is a prior making regret linear in $T$ (Example 8.1.1: a known 1/2-reward arm vs. an unknown arm; whichever the rule picks first, it picks forever). You need either randomisation (TS) or nonstationarity (UCB's shrinking bonus).

### Four places TS is the wrong tool

1. **No exploration needed.** Portfolio selection with backtestable public returns. Also contextual bandits with rich *continuous* context — [11] shows random context provides "free" exploration and greedy is fine. Caveat: with **binary** context features or per-action offsets $\bar\theta_x$, greedy can still get permanently stuck. The news-recommendation setup in §7.1 has *both*, so it does need exploration.
2. **Pure exploration / best-arm identification.** A/B tests and simulation ranking-and-selection. Once TS is fairly sure, it exploits nearly always and stops refining knowledge of alternatives. Russo's top-two variant fixes this and is far better. **If you use TS for A/B testing, use the pure-exploration variant.**
3. **Time sensitivity (many-armed deterministic bandit).** $K$ arms, $\theta_x\sim U[0,1]$ deterministic. TS pulls a *new* arm nearly every round while $t\ll K$; as $K\to\infty$ it never repeats. Just scanning arms in order and stopping at the first $\theta_x\geq 1-\epsilon$ finds an $\epsilon$-optimal arm in $1/\epsilon$ rounds, independent of $K$. Satisficing TS is the fix. The entropy $H(x^*)$ in the regret bound is exactly the symptom — it blows up.
4. **Actions valuable only for information.** Example 8.2.3: $k+1$ arms; arm 0 pays $1/2\theta$ and is known never optimal, but *reveals $\theta$ instantly*. TS never plays it, and grinds through arms one at a time. Same failure in a sparse linear bandit (TS needs $d/2$ rounds where bisection needs $\log d$) and in assortment recommendation (offering $m$ *diverse* product types identifies customer type $m\times$ faster than TS's always-homogeneous assortments). Information-directed sampling, which explicitly minimises the information ratio, fixes all three — at higher computational cost.

**Open issues the authors flag:** asymptotic optimality is only proved for specific uninformative priors; the $\tilde O(d^{3/2}\sqrt T)$ worst-case linear bound applies to a *variance-inflated* variant, not vanilla TS; the bootstrap approximation has essentially no theory; and entropy-based bounds go vacuous with infinitely many actions (rate-distortion is the suggested repair).

**Practical checklist for using it:** pick a conjugate model if you can, because posterior updates become two-number bookkeeping. Fit an empirical prior from past similar experiments (the tutorial fits $\text{beta}(1,100)$ to a histogram of past ad CTRs, which rules out CTR > 0.05 — a big head start). Model correlations between actions; that is where the wins live. Prefer a change of variables that makes your posterior close to Gaussian. If you need it incremental, ensemble sampling with ~30 members is a good default. There is an accompanying Python package reproducing every figure.

---

## Links
Related: [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]] · [[Decision Sciences]] · [[Uncertainty]] · [[Beliefs]] · [[Markov Decision Process]] · [[Markov Chain Monte Carlo]] · [[Hamiltonian Monte Carlo]] · [[Kalman Filter]] · [[KL Divergence]] · [[Random variable]] · [[Derivative]] · [[Playing Atari with Deep Reinforcement Learning (DQN)]] · [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)]] · [[Counterfactual Reasoning and Learning Systems]] · [[Doubly Robust Policy Evaluation and Learning]] · [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Recommender Systems - Evolution]]

New topics worth writing: Upper Confidence Bound (UCB1), Regret and Bayesian regret, Eluder dimension, Information-directed sampling, Information ratio, Conjugate priors, Beta-Bernoulli conjugacy, Laplace approximation, Langevin Monte Carlo / SGLD, Gibbs sampling, Statistical bootstrap, Ensemble sampling, PSRL and deep exploration, Gittins index, Best-arm identification, Satisficing bandits, Cascade click models, Multinomial logit choice models, Sherman–Morrison update
