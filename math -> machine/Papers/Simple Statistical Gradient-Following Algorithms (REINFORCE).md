---
title: "Simple Statistical Gradient-Following Algorithms (REINFORCE)"
authors: ["Williams"]
year: 1992
url: https://link.springer.com/content/pdf/10.1007/BF00992696.pdf
priority: Must-Read
read_on: 2026-08-25
tags: [paper, llm, rl, theory]
---
## The Core Idea

You have a network whose output is **random**. You feed it an input, it samples an action, the world hands back a single number $r$ (the reward). Nobody tells you what the right action was. There is no target, so there is no error to backpropagate.

The insight: you do not need one. If you can write down the *probability* of the action you took, you can get an unbiased estimate of the [[Derivative#Gradient|gradient]] of expected reward with respect to the weights, using only $r$ and the derivative of the log-probability of the action you happened to sample.

The whole family is one line:

$$\Delta w_{ij} = \alpha_{ij}\,(r - b_{ij})\,\frac{\partial \ln g_i}{\partial w_{ij}}$$

Williams calls this **REINFORCE**: **RE**ward **I**ncrement $=$ **N**onnegative **F**actor $\times$ **O**ffset **R**einforcement $\times$ **C**haracteristic **E**ligibility. Three factors: a learning rate, the reward minus a baseline, and the log-probability derivative.

Theorem 1 is the payload: for *any* algorithm of this form, $E\{\Delta W \mid W\} = \alpha \nabla_W E\{r \mid W\}$. On average, you are doing exact gradient ascent on expected reward — even though the system never builds a model of the reward function, never estimates a gradient explicitly, and stores nothing from which a gradient could be reconstructed. That is what "simple" in the title means. A better word would be *non-model-based*.

Why this did not exist before: the dominant learning rule of the era, [[Backpropagation]], needs a differentiable path from weights to loss. A random sampling step breaks that path. Existing reinforcement rules ($L_{R-I}$, $A_{R-P}$) were hand-designed heuristics that *seemed* to work. Williams showed they are special cases of a gradient method, and gave a recipe for deriving a correct rule for **any** random unit whose density you can differentiate — Bernoulli, Gaussian, Poisson, whatever.

> [!NOTE] Characteristic eligibility ^characteristic-eligibility
> $e_{ij} = \partial \ln g_i / \partial w_{ij}$, where $g_i(\xi, w_i, x_i) = \Pr\{y_i = \xi \mid w_i, x_i\}$. It says: "if I nudge this weight up, how much more likely does the action I just took become?" Reward-weight that quantity and you climb the reward. Modern name: the **score function**.

This is the ancestor of every policy-gradient method: [[Proximal Policy Optimization Algorithms]], the RL step in [[Training language models to follow instructions with human feedback]], and the actor half of every actor-critic.

## The Methodology

**Setup.** A feedforward net of stochastic units. Unit $i$ has input vector $x_i$, weights $w_i$, and samples output $y_i$ from $g_i(\cdot, w_i, x_i)$. A single scalar reward $r$ is **broadcast to every unit** — no per-unit target, no credit assignment through the wiring. Objective: maximise $E\{r \mid W\}$, treated as a deterministic but unknown function of the weight matrix.

**Conditions for the theorem.** $\alpha_{ij} \ge 0$ and may depend only on $w_i$ and time (not on the input to the unit — Lemma 2 needs this). The baseline $b_{ij}$ must be **conditionally independent of $y_i$** given $W$ and $x_i$. That is the only real constraint; you cannot peek at the action you took when computing the baseline, or the estimate becomes biased.

**Proof sketch.** Two facts do all the work. Fact 2: $\sum_\xi \partial g_i / \partial w_{ij} = 0$, because probabilities sum to one and differentiating a constant gives zero — this is why *any* action-independent baseline cancels out exactly. Fact 1 plus conditioning on $y_i$ gives $E\{\Delta w_{ij}\} = \alpha_{ij}\,\partial E\{r\}/\partial w_{ij}$.

**Bernoulli-logistic unit.** $p_i = 1/(1+e^{-s_i})$, $s_i = w_i^{\top} x_i$, output $y_i \in \{0,1\}$. Eligibility for $p_i$:

$$\frac{\partial \ln g_i}{\partial p_i} = \frac{y_i - p_i}{p_i(1-p_i)}$$

Chain-rule it through the logistic, whose derivative is exactly $p_i(1-p_i)$, and the denominator cancels:

$$\frac{\partial \ln g_i}{\partial w_{ij}} = (y_i - p_i)\,x_j$$

So the update is $\Delta w_{ij} = \alpha\, r\, (y_i - p_i)\, x_j$ — which is *literally* the $A_{R-I}$ rule Barto's group had been using. Suddenly it has a proof. With reward comparison it becomes $\Delta w_{ij} = \alpha (r - \bar r)(y_i - p_i) x_j$, where $\bar r(t) = \gamma r(t-1) + (1-\gamma)\bar r(t-1)$.

**Gaussian unit (the interesting one).** Output $y \sim \mathcal{N}(\mu, \sigma^2)$, with both $\mu$ and $\sigma$ adaptable:

$$\frac{\partial \ln g}{\partial \mu} = \frac{y - \mu}{\sigma^2}, \qquad \frac{\partial \ln g}{\partial \sigma} = \frac{(y-\mu)^2 - \sigma^2}{\sigma^3}$$

$\sigma$ is a *learnable exploration width*. Single-parameter units cannot separate "where to explore" from "how much"; a two-parameter distribution can. Suggested rates: $\alpha_\mu = \alpha_\sigma = \alpha\sigma^2$. Footnote 2 warns $\sigma$ can go negative — adapt $\ln \sigma$ instead. (Every modern continuous-control policy still does exactly this.)

**Proposition 1.** For any exponential-family density written as $g = \exp[Q(\mu,\theta)y + D(\mu,\theta) + S(y)]$ where $\mu$ is the mean, the mean's eligibility is *always* $(y-\mu)/\sigma^2$. Bernoulli, normal, Poisson, exponential all obey it. So "prediction error over variance" is not a Gaussian coincidence.

**Episodic REINFORCE.** Delayed reward, $k$ steps, one $r$ at the end:

$$\Delta w_{ij} = \alpha_{ij}(r - b_{ij}) \sum_{t=1}^{k} e_{ij}(t)$$

Proved by *unfolding in time*: duplicate the net once per step into an acyclic net $N^*$, apply Theorem 1 there, then note $\partial E\{r\}/\partial w_{ij} = \sum_t \partial E\{r\}/\partial w^t_{ij}$ because all copies share a value. Implementation is one accumulator per weight, ticking along in real time, ignorant of the reward that will eventually arrive.

**Mixing with backprop.** Two distinct tricks:

1. *Stochastic outputs, deterministic hidden layers.* Because randomness is independent across output units, $\ln g = \sum_{k \in O} \ln g_k$, so $\partial \ln g / \partial w_{ij}$ is a sum you can compute with a standard backward pass. You "inject" $(y_k - p_k)/(p_k(1-p_k))$ just after each output unit's squashing function and run [[Backpropagation]] normally. Hidden weights get a proper chain-rule signal instead of a noisy correlation.
2. *Backprop through the random number generator.* You cannot in general get $\partial J/\partial p_i$ from $\partial J/\partial y_i$ through a Bernoulli sampler — if $J$ is nonlinear in $y_i$ there is no relation. But for a Gaussian you can rewrite $y = \mu + \sigma z$ with $z$ a standard normal, giving $\partial y/\partial \mu = 1$ and $\partial y/\partial\sigma = (y-\mu)/\sigma = z$. This is the **reparameterisation trick**, 1992, twenty-one years before [[Auto-Encoding Variational Bayes (VAE)]]. Williams notes it works "whenever the output can be expressed as a function of the parameters together with some auxiliary random variables, as long as the dependence on the parameters is differentiable" — the exact modern statement.

## Ablation Studies and Experiments

This is a theory paper. There are no tables. The experimental content is a survey of simulations, mostly reported elsewhere, and it is unusually honest about failure.

**What worked.** Sutton (1984) compared several rules on single-Bernoulli-unit associative and non-associative tasks; REINFORCE with reinforcement comparison beat everything else tested, including $L_{R-I}$.

**What did not work:**

- **No general convergence theory.** Theorem 1 says the *expected* step points uphill. It says nothing about where you end up. Even $L_{R-I}$, the one case understood analytically, converges to a deterministic action with probability 1 — including, with nonzero probability, the *worse* action. Analogy given: a biased random walk on the integers with absorbing barriers still gets absorbed at the wrong end sometimes. So "unbiased ascent direction" $\ne$ "finds the optimum."
- **Baseline $b=0$ is actively harmful.** With $r$ always positive and a Gaussian unit, $\sigma$ can collapse to $0$ *before* $\mu$ reaches any hill top. Exploration dies, learning stops. The theorem is indifferent to the baseline; empirically it decides whether the thing works at all. With reinforcement comparison, any $b$ between the two possible reward values produces motion always toward the better action.
- **Reward shaping was the bottleneck, not the algorithm.** On multilayer and recurrent nets doing supervised tasks with only reward feedback, "it often required careful selection of the reinforcement function to obtain solutions" — the obvious reward functions have severe false maxima. $A_{R-P}$ succeeded on those same naive rewards where REINFORCE did not. Being a correct gradient method did not save it.
- **REINFORCE is slow.** Episodic REINFORCE "especially slow", because it spreads credit *uniformly over all past time steps*. No temporal discrimination at all. This is precisely the variance problem that TD, advantage estimation and later GAE exist to fix.
- **A non-gradient variant beat it.** Replace $p$ with an exponential moving average $\bar y$ of past outputs: $\Delta w = \alpha (r - \bar r)(y - \bar y)$. This "has been found generally to converge faster and more reliably" than the corresponding REINFORCE rule — despite having no derivation. Williams cannot explain it and says so. He conjectures it is secretly doing a linear regression of $r$ on $y$ at each unit.
- **Dayan's minimum-variance baseline** — chosen to minimise the variance of weight changes rather than to equal mean reward — gives only "a slight improvement", and "a more convincing advantage remains to be demonstrated."
- Gullapalli's alternative, setting $\sigma \propto (1-r)$, is presented as a sensible competitor: explore wide when performance is bad. Not compared head-to-head.

**Ablation-in-spirit:** the entropy-bonus variant (Williams & Peng 1991) added an entropy term to the reward to escape local optima, and helped networks that needed hierarchical search. Entropy regularisation in policy gradients starts here.

## Worth Remembering

- The **baseline is free**. Fact 2 means subtracting anything not depending on the sampled action leaves the gradient unbiased but changes the variance enormously. Every value-function baseline and advantage estimate in modern RL is this one identity, cashed repeatedly.
- REINFORCE "measures the correlation between variations in local behaviour and the resulting variations in global performance." Each unit is a selfish agent guessing at its own contribution from a shared scalar. It ignores everything known about the wiring. That is why it is high-variance and why hybrid schemes — backprop wherever the path is differentiable, score function only at the sampling step — are strictly better when available.
- Williams distinguishes **model-based** (learn $E\{r\mid x,y\}$ and differentiate it, e.g. Munro 1987) from REINFORCE, which is not model-based even locally. He frames [[Playing Atari with Deep Reinforcement Learning (DQN)|Q-learning]] as learning *local, per-state* models of cumulative reward. Useful taxonomy for reading [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] later.
- Practical caveat: the eligibility $\frac{1}{g_i}\frac{\partial g_i}{\partial w_{ij}}$ blows up as $g_i \to 0$. For the logistic case the singularity cancels analytically — but only for the logistic. Pick your squashing function so the cancellation happens, or you will get infinite updates on rare actions. (Same disease importance-sampling ratios have; same reason PPO clips them.)
- Delayed reward is handled here by brute unfolding and uniform credit. The paper explicitly flags the better route — treat a $k$-step episode with per-step rewards as $k$ overlapping episodes when $r$ is causal — then says "we omit further discussion of the details." That omitted paragraph is the reward-to-go trick.
- Convergence to "points that lead to zero variance in network behaviour" is the failure mode worth internalising: the policy becomes deterministic, the score function goes to zero, and learning halts regardless of quality. Premature determinism, not divergence, is how policy gradient dies.

## Links

Related: [[Backpropagation]] · [[Derivative#Gradient|Derivative]] · [[Proximal Policy Optimization Algorithms]] · [[Playing Atari with Deep Reinforcement Learning (DQN)]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Training language models to follow instructions with human feedback]] · [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] · [[Markov Decision Process]] · [[A Distributional Perspective on Reinforcement Learning]] · [[A Tutorial on Thompson Sampling]] · [[Random variable]] · [[Deep Learning]] · [[Uncertainty]] · [[Decision Sciences]]

New topics worth writing: Score function estimator (likelihood-ratio gradient), Variance reduction and control variates in gradient estimation, Actor-critic architectures, Advantage estimation and GAE, Entropy regularisation in policy gradients, Exponential families, Temporal difference learning, Gumbel-Softmax / straight-through estimators for discrete sampling
