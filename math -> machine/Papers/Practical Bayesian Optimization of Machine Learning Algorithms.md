---
title: "Practical Bayesian Optimization of Machine Learning Algorithms"
authors: ["Jasper Snoek", "Hugo Larochelle", "Ryan P. Adams"]
year: 2012
arxiv: "1206.2944"
url: https://arxiv.org/abs/1206.2944
priority: Must-Read
read_on: 2026-08-25
tags: [paper, optimization]
---
## The Core Idea

Tuning hyperparameters is usually guesswork. You pick a learning rate, a weight decay, a number of epochs, run the whole training job, look at validation error, and repeat. Each run is expensive — hours or days. Grid search burns hundreds of runs to cover a handful of knobs.

Bayesian optimisation treats "validation error as a function of hyperparameters" as an unknown function $f(\mathbf{x})$ you are trying to minimise, and puts a probability distribution over what that function might look like. After each run you update the distribution. Then you pick the next point to try by asking: *where is the expected payoff highest, given everything I know so far?* This is not new — Mockus proposed it in 1978. What this paper does is make it actually work on real machine learning problems, and it does so with three unglamorous engineering choices:

1. **Marginalise over the model's own hyperparameters instead of optimising them.** The Gaussian process that models $f$ itself has knobs (length scales, noise). Everyone before fit them by maximum likelihood — a single point estimate. This paper samples them with MCMC and averages the acquisition function over the samples. This turns out to matter a lot.
2. **Stop using the squared-exponential kernel.** It assumes $f$ is infinitely smooth. Real validation-error surfaces are not. Matérn 5/2 assumes twice-differentiable, and wins.
3. **Optimise expected improvement *per second*, not per evaluation.** You do not care about run count. You care about wall-clock time. Model the log-duration with a second GP and divide.

Plus a Monte-Carlo trick for running experiments in parallel: hallucinate ("fantasise") outcomes for the jobs still running, and average the acquisition function over those hallucinations.

The headline result: nine hyperparameters of Krizhevsky's three-layer CIFAR-10 conv net, tuned automatically, hit **14.98% test error** versus the **18%** the human expert (Krizhevsky himself) had achieved, which was also the published state of the art at the time.

> [!NOTE] Bayesian optimisation
> Optimising an expensive black-box function by (a) fitting a probabilistic model of it from the points you have already evaluated, and (b) using that model's mean *and* uncertainty to choose the next point. You spend real compute deciding where to look, because looking is so expensive. ^bayesian-optimisation

## The Methodology

**The surrogate model.** A Gaussian process prior over $f:\mathcal{X}\to\mathbb{R}$, $\mathcal{X}\subset\mathbb{R}^D$. A GP says: any finite set of points has a joint multivariate Gaussian distribution. Given observations $y_n \sim \mathcal{N}(f(\mathbf{x}_n), \nu)$, the posterior at any new $\mathbf{x}$ is Gaussian in closed form, giving a predictive mean $\mu(\mathbf{x})$ and a predictive variance $\sigma^2(\mathbf{x})$. That is the whole appeal — you get calibrated [[Uncertainty|uncertainty]] for free, no sampling needed for the prediction itself.

**The kernel.** The usual default is the ARD squared exponential,

$$K_{\mathsf{SE}}(\mathbf{x},\mathbf{x}') = \theta_0 \exp\!\left\{-\tfrac{1}{2} r^2(\mathbf{x},\mathbf{x}')\right\},\qquad r^2(\mathbf{x},\mathbf{x}') = \sum_{d=1}^{D}\frac{(x_d - x'_d)^2}{\theta_d^2}$$

where each dimension gets its own length scale $\theta_d$ (that is what ARD, automatic relevance determination, means — a large $\theta_d$ means dimension $d$ barely matters). Snoek et al. swap this for ARD Matérn 5/2:

$$K_{\mathsf{M52}}(\mathbf{x},\mathbf{x}') = \theta_0\left(1 + \sqrt{5r^2} + \tfrac{5}{3}r^2\right)\exp\left\{-\sqrt{5r^2}\right\}$$

Sample functions from this are twice differentiable — the same smoothness assumption quasi-Newton optimisers make — but not infinitely smooth. Less rigid, more realistic.

**The acquisition function.** Three candidates, all closed-form under a GP. Let $\gamma(\mathbf{x}) = \dfrac{f(\mathbf{x}_{\mathsf{best}}) - \mu(\mathbf{x})}{\sigma(\mathbf{x})}$.

- Probability of improvement: $a_{\mathsf{PI}} = \Phi(\gamma(\mathbf{x}))$.
- Expected improvement: $a_{\mathsf{EI}} = \sigma(\mathbf{x})\big(\gamma(\mathbf{x})\Phi(\gamma(\mathbf{x})) + \phi(\gamma(\mathbf{x}))\big)$.
- GP lower confidence bound: $a_{\mathsf{LCB}} = \mu(\mathbf{x}) - \kappa\,\sigma(\mathbf{x})$.

They use EI. PI is badly behaved (it will happily take a 0.001 improvement with 99% probability over a huge improvement with 60% probability). LCB needs its own tuning knob $\kappa$ — and the whole point is to remove knobs. Note how close $a_{\mathsf{LCB}}$ is in spirit to the confidence bound in [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)|LinUCB]]: mean minus a multiple of the standard deviation, the same explore/exploit arithmetic.

**The fully-Bayesian bit (contribution 1).** The GP has $D+3$ of its own hyperparameters: $D$ length scales $\theta_{1:D}$, amplitude $\theta_0$, noise $\nu$, constant mean $m$. The standard move is to maximise the marginal likelihood

$$p(\mathbf{y}\mid\{\mathbf{x}_n\},\theta,\nu,m) = \mathcal{N}(\mathbf{y}\mid m\mathbf{1},\ \boldsymbol{\Sigma}_\theta + \nu\mathbf{I})$$

and use that single $\theta$. Instead they compute the **integrated acquisition function**:

$$\hat{a}(\mathbf{x}\,;\,\{\mathbf{x}_n,y_n\}) = \int a(\mathbf{x}\,;\,\{\mathbf{x}_n,y_n\},\theta)\;p(\theta\mid\{\mathbf{x}_n,y_n\})\;\mathrm{d}\theta$$

estimated by Monte Carlo with slice sampling (Murray & Adams 2010) over $\theta$. Draw ~10 posterior samples of $\theta$, compute EI under each, average, then maximise the average. With five data points and an unknown length scale, single-$\theta$ EI will confidently point at one spot; the integrated version points somewhere else, hedged across the plausible smoothness levels. Related in machinery to [[Markov Chain Monte Carlo]] — you are integrating out nuisance parameters rather than plugging in a point estimate.

Why is this affordable? Both the MCMC and the marginal-likelihood optimisation cost $O(N^3)$ for an $N\times N$ linear solve, and $N$ is tiny (tens to hundreds of experiments). The training runs you are trying to schedule cost hours. The surrogate's compute is free by comparison.

**Cost-aware search (contribution 2).** The duration $c(\mathbf{x})$ is also unknown, so model $\ln c(\mathbf{x})$ with a second, independent GP. Log, because durations are positive and multiplicatively spread. Then maximise

$$\frac{a_{\mathsf{EI}}(\mathbf{x})}{\mathbb{E}[c(\mathbf{x})]}$$

— expected improvement per second. This automatically prefers cheap configurations early, e.g. loose convergence tolerances, fewer epochs, while it maps out the rest of the space.

**Parallelism (contribution 3).** Suppose $N$ runs are done and $J$ runs are pending at known locations $\{\mathbf{x}_j\}$ with unknown outcomes $\{y_j\}$. The right thing to compute is

$$\hat{a}(\mathbf{x}) = \int_{\mathbb{R}^J} a(\mathbf{x}\,;\,\{\mathbf{x}_n,y_n\},\theta,\{\mathbf{x}_j,y_j\})\;p(\{y_j\}\mid\{\mathbf{x}_j\},\{\mathbf{x}_n,y_n\})\;\mathrm{d}y_1\cdots\mathrm{d}y_J$$

The inner distribution is just a $J$-dimensional Gaussian — the GP posterior at the pending points. So: sample $J$ "fantasy" outcomes, condition the GP on them, compute EI, repeat, average. Cheap. Ginsbourger & Riche had mentioned this idea and dismissed it as intractable; here it works fine in practice.

## Ablation Studies and Experiments

**Branin-Hoo** (2D synthetic benchmark, 100 repeats) and **logistic regression on MNIST** (4 hyperparameters: SGD learning rate, $\ell_2$ [[Regularization|regularisation]], minibatch size 20–2000, epochs 5–2000; 10 repeats). Baseline is the Tree Parzen Algorithm of Bergstra et al. (2011). On Branin-Hoo, GP EI MCMC finds the minimum in **fewer than half** the evaluations TPA needs. Integrating over $\theta$ beats the point estimate in both problems.

**Online LDA on 249,560 Wikipedia articles** (200k train / 24,560 val / 25k test, 7,702-word vocabulary, 100 topics, $\eta=\alpha=0.01$). Three hyperparameters: $\tau_0$, $\kappa$ (which set the learning rate $\rho_t = (\tau_0+t)^{-\kappa}$), and minibatch size. Hoffman et al.'s original paper used a $6\times6\times8 = 288$-point grid; each evaluation takes **5–10 hours**, so the grid is **60–120 processor-days**. Restricting the optimisers to the same grid for a fair comparison:

- GP EI MCMC is the most sample-efficient per function evaluation.
- 3× and 5× parallelised GP EI MCMC reach a good perplexity in far less *wall-clock* time, even though they waste evaluations.
- Unshackled from the grid, parallel GP EI MCMC finds a **better minimum than the published grid search**, using a fraction of the runs.

**Motif finding with M3E structured SVMs** (~40,000 protein sequences). Three hyperparameters, gridded: $C$ over 25 log-spaced values from $10^{-1}$ to $10^6$, entropy parameter $\alpha$ over 14 log-spaced values from 0.1 to 5, convergence tolerance $\epsilon \in \{10^{-4},10^{-3},10^{-2},10^{-1}\}$ — 1,400 combinations, each averaged over 5 random 50-50 splits. This is where the cost model earns its keep. GP EI MCMC wins on *evaluation count*, but **GP EI per Second wins on wall-clock** because it learns to run with a sloppy tolerance ($\epsilon = 10^{-1}$) while exploring $C$ and $\alpha$, then tightens later. 3× GP EI per Second is the *worst* per evaluation and the *best* per second — a clean illustration that the two metrics genuinely disagree.

**The kernel ablation** (Figure 5c, same motif problem, 100 repeats each) is the most informative single result: covariance choice significantly changes performance, and **estimating the length scales is critical**. The infinite differentiability of the squared exponential is too strong an assumption for this surface. So the paper's two "boring" choices — Matérn 5/2 and marginalising $\theta$ — are exactly the ones doing the work.

**CIFAR-10 conv net** (Krizhevsky's three-layer network, his own code). Nine hyperparameters: number of epochs, learning rate, four weight costs (one per layer plus the softmax weights), and the width, scale and power of the response normalisation on the pooling layers. Five randomly initialised runs, tuned on a held-out validation set. Human expert baseline: 18% test error — which *was* the state of the art (tied with Coates & Ng 2011). GP EI MCMC: **14.98% test error**. Over 3 points better than a careful expert and than the published SOTA. The knobs being tuned are exactly the fiddly ones in [[ImageNet Classification with Deep CNNs (AlexNet)|AlexNet]]-style networks that nobody has good intuitions for.

**What did not work / was rejected:** probability of improvement (worse behaved than EI); GP-UCB (needs its own $\kappa$, so it just relocates the tuning problem — though the authors note the regret framing is more principled for many settings); optimising GP hyperparameters by marginal likelihood rather than sampling them; and the squared-exponential kernel. They also assume $f$ and $c$ are independent, and openly flag multi-task GPs (Teh et al. 2005; Bonilla et al. 2008) as the better but unexplored option.

## Worth Remembering

- The GP posterior cost is $O(N^3)$ in the number of *experiments*, not data points. Fine at $N \sim 100$, painful past a few thousand. If your objective is cheap, Bayesian optimisation is the wrong tool — the surrogate overhead dominates.
- Everything here assumes continuous, boxed, low-dimensional $\mathcal{X}$ ($D \approx 2$–$9$ in the experiments). Conditional hyperparameters ("only if using dropout, tune the dropout rate") and categorical ones are not handled. This is where the tree-based TPE baseline they beat actually has the structural advantage.
- The parallel version is *greedy sequential with hallucinated pending results*, not a true batch-optimal method. The authors say the principled thing (rolling out the acquisition policy) is intractable, and this approximation is what they could afford.
- Note the shape of the explore/exploit machinery: $\mu - \kappa\sigma$, posterior sampling, expected improvement. This is the same family of ideas as [[A Tutorial on Thompson Sampling|Thompson sampling]] and bandits generally, applied to a continuous arm space with a GP as the [[Beliefs|belief]] model. The difference: bandits care about cumulative regret over every pull; here you only care about the single best point at the end, so pure exploitation of a bad region costs you nothing but time.
- Practical caveat on the CIFAR-10 headline: it beat the expert on *nine already-chosen* hyperparameters of an *already-designed* architecture. The search did not invent anything. It squeezed a fixed design harder than a human was willing to.
- This paper became the Spearmint package, and the direct ancestor of essentially every "automatic hyperparameter tuning" service. Its practical successor for large-scale deep learning is usually Hyperband/ASHA — cheap early stopping beats a clever surrogate when each run is many GPU-hours and you have many workers.
- Open question worth chasing: why *does* marginalising the length scales help so much? Likely because with 5–20 observations the marginal likelihood is nearly flat and the MLE length scale is wildly overconfident, which collapses $\sigma(\mathbf{x})$ and kills exploration. Integration keeps the uncertainty honest early, when it matters most.

## Links

Related: [[Uncertainty]] · [[Beliefs]] · [[Markov Chain Monte Carlo]] · [[Random variable]] · [[A Tutorial on Thompson Sampling]] · [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]] · [[Regularization]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Decision Sciences]] · [[No U-Turn Sampler]] · [[Hamiltonian Monte Carlo]]

New topics worth writing: Gaussian Processes, Kernel functions and the Matérn family, Expected Improvement / acquisition functions, Slice sampling, Automatic Relevance Determination, Marginal likelihood (evidence), Tree-structured Parzen Estimator (TPE), Hyperband and successive halving, Random search for hyperparameter optimisation (Bergstra & Bengio 2012), Multi-task Gaussian processes, Online LDA and variational Bayes
