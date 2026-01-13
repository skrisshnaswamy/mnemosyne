---
title: "An Empirical Evaluation of Thompson Sampling (NeurIPS)"
authors: ["Chapelle & Li"]
year: 2011
url: https://papers.nips.cc/paper_files/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf
priority: Must-Read
read_on: 2026-08-25
tags: [paper, rl, optimization, theory]
---
## The Core Idea

Thompson sampling is a 1933 idea that almost nobody used in 2011. This paper is the empirical mugging that made people use it.

The exploration/exploitation problem: you have several options (arms), each pays off with some unknown rate, and you must both learn which is best and earn as much as possible while learning. The dominant answer at the time was UCB — build a confidence interval around each arm's estimated payoff, pick the arm with the highest *upper* end. Optimism in the face of uncertainty, deterministic, with proven regret bounds.

Thompson sampling does something different and stranger. Keep a posterior — a probability distribution over what each arm's true payoff rate might be. Each round, **draw one random sample** from each posterior, and act as if that sample were the truth. That single line of code makes you pick each arm exactly as often as the probability that it *is* the best arm. No confidence bound to derive, no tuning knob.

Why was it unpopular? No finite-time regret bound existed. Only asymptotic convergence had been proved. So the theory community ignored it.

What this paper shows, on simulations plus two real Yahoo! products:

1. Thompson sampling beats a *tight* UCB, not a strawman one, and it tracks the Lai–Robbins asymptotic lower bound.
2. It is dramatically more robust to **delayed feedback** — the situation every real system is in, because you batch-update your model every 10 minutes, not every impression. With 1000-step delay in simulation, UCB's regret is 226,220 and Thompson's is 59,256. A 3.8× gap.
3. The reason for (2) is beautifully simple: Thompson sampling is randomised. If it makes a bad call, the next user gets a different draw. UCB is deterministic — until fresh data arrives, it makes the *same* bad call for every single user in the batch.

That last point is the practical payload of the paper. It is why Thompson sampling is now the default bandit in industry.

> [!NOTE] Probability matching ^probability-matching
> Choosing action $a$ with probability equal to the posterior probability that $a$ is optimal:
> $$\int \mathbb{I}\!\left[\mathbb{E}(r|a,x,\theta) = \max_{a'}\mathbb{E}(r|a',x,\theta)\right] P(\theta|\mathcal{D})\, d\theta$$
> You never compute this integral. Drawing one $\theta \sim P(\theta|\mathcal{D})$ and taking the argmax samples from it for free.

## The Methodology

**The general algorithm.** Data $\mathcal{D}$ is a set of triples $(x_i, a_i, r_i)$ — context, action, reward. Model rewards with a likelihood $P(r|a,x,\theta)$ and a prior $P(\theta)$. Bayes gives $P(\theta|\mathcal{D}) \propto \prod_i P(r_i|a_i,x_i,\theta) P(\theta)$.

Each round $t$:
1. Receive context $x_t$.
2. Draw $\theta_t \sim P(\theta|\mathcal{D})$.
3. Play $a_t = \arg\max_a \mathbb{E}_r(r|x_t, a, \theta_t)$.
4. Observe $r_t$, append to $\mathcal{D}$.

**Bernoulli bandit version.** Each arm $i$ pays 1 with unknown probability $\theta_i^*$. Beta is conjugate to Bernoulli, so keep success and failure counts $S_i, F_i$; the posterior is $\text{Beta}(S_i+\alpha, F_i+\beta)$. Draw one $\theta_i$ per arm, play the argmax, increment $S$ or $F$. Prior used: $\text{Beta}(1,1)$, i.e. uniform.

**The UCB they compare against.** Not the loose Auer et al. bound. A tight Chernoff-derived bound, with $m$ = pulls of the arm, $k$ = total reward, $\delta = \sqrt{1/t}$:
$$\frac{k}{m} + \sqrt{\frac{2\frac{k}{m}\log\frac{1}{\delta}}{m}} + \frac{2\log\frac{1}{\delta}}{m}$$
They state plainly that they tried the original Auer bound and it gave *larger* regret, so they used the good one. Fair fight.

**Contextual version (display ads).** The posterior over logistic-regression weights is approximated by a diagonal Gaussian via a Laplace approximation — mean at the mode, per-weight precision from the curvature. Algorithm 3:

- Prior: each weight $w_i \sim \mathcal{N}(m_i, q_i^{-1})$, start $m_i=0$, $q_i=\lambda$.
- Each batch, minimise
$$\frac{1}{2}\sum_{i=1}^d q_i (w_i - m_i)^2 + \sum_{j=1}^n \log\!\left(1 + \exp(-y_j w^\top x_j)\right)$$
- Update $m_i = w_i$ and $q_i \leftarrow q_i + \sum_j x_{ij}^2 p_j(1-p_j)$ where $p_j = \sigma(w^\top x_j)$.

That last term is the diagonal of the [[Derivative#Hessian|Hessian]] of the logistic loss. The Gaussian posterior does double duty: it is the prior for the next batch, and it is what you sample from for Thompson sampling. Features are user × page × ad conjunctions, feature-hashed into a sparse binary vector of dimension $2^{24}$.

**Two knobs they explore.**

*Posterior reshaping.* Replace $\text{Beta}(a,b)$ with $\text{Beta}(a/\alpha, b/\alpha)$. Mean unchanged, variance multiplied by roughly $\alpha^2$. $\alpha < 1$ = sharper posterior = less exploration. In the Gaussian case, multiply the standard deviations $q_i^{-1/2}$ by $\alpha$.

*Optimistic Thompson sampling.* Replace the sampled score with $\max\big(\mathbb{E}_r(r|x_t,a,\theta_t),\ \mathbb{E}_{r,\theta|\mathcal{D}}(r|x_t,a)\big)$ — never let a draw push an arm *below* its posterior mean. The intuition: exploration is helped by inflating uncertain arms, and there is no obvious benefit to deflating them.

**Evaluation setups.**
- *Simulation:* best arm pays 0.5, other $K-1$ arms pay $0.5-\varepsilon$. $K \in \{10,100\}$, $\varepsilon \in \{0.02, 0.1\}$, 100 repeats.
- *Display ads:* real contexts and real ad inventory (66,373 ads, 5,910 to 1 eligible per context, mean 1,364), but **simulated clicks** from a known $w^*$. ~13,000 contexts/hour, model refreshed hourly, 4 days.
- *News (Yahoo! front page):* 34M randomised serving events from 7 days in June 2009, evaluated with the unbiased replay estimator of Li et al. 2011. ~20 articles in the pool. User features: >1000 raw binary dims, PCA'd to 20 components plus a constant bias term = 21. One weight vector per article.

## Ablation Studies and Experiments

**Simulation regret.** All four settings show log-linear regret curves for Thompson sampling, sitting close in slope to the Lai–Robbins lower bound
$$R(T) \geq \log(T)\left[\sum_{i=1}^K \frac{p^* - p_i}{D(p_i \| p^*)} + o(1)\right]$$
where $D$ is the [[KL Divergence|KL divergence]]. Thompson beats tight-UCB across the board.

**Delay simulation** (10 items, item retires with prob $10^{-3}$ per step, true rates $\sim \text{Beta}(4,4)$, $T=10^6$). Regret at feedback delay $\delta$:

| $\delta$ | 1 | 3 | 10 | 32 | 100 | 316 | 1000 |
|---|---|---|---|---|---|---|---|
| UCB | 24,145 | 24,695 | 25,662 | 28,148 | 37,141 | 77,687 | 226,220 |
| TS | 9,105 | 9,199 | 9,049 | 9,451 | 11,550 | 21,594 | 59,256 |
| Ratio | 2.65 | 2.68 | 2.84 | 2.98 | 3.22 | 3.60 | 3.82 |

Note that TS regret is essentially flat out to $\delta = 32$ — it barely notices. UCB degrades monotonically from step one.

**Display advertising, CTR regret (%), lower is better:**

| Method | TS 0.25 | TS 0.5 | TS 1 | LinUCB 0.5 | LinUCB 1 | LinUCB 2 | ε-greedy .005 | .01 | .02 | Exploit | Random |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Regret | 4.45 | **3.72** | 3.81 | 4.99 | 4.22 | 4.14 | 5.05 | 4.98 | 5.22 | 5.00 | 31.95 |

**What did not work:**

- **ε-greedy never beat pure exploitation.** 5.05 / 4.98 / 5.22 versus Exploit's 5.00. The authors' explanation: exploit-only is already far better than random (5.00 vs 31.95), so every random action ε-greedy takes costs a lot of regret and buys little information. Uniform exploration is a bad deal when the greedy policy is good.
- **Exploit-only was surprisingly strong** — better than every ε-greedy setting, and beaten only by TS and LinUCB. With a contextual model, the changing context supplies incidental exploration for free (citing Sarkar 1991). If someone tells you their contextual bandit beats greedy, ask by how much.
- **Optimistic Thompson sampling gave only a marginal gain** in simulation ($K=10$, $\varepsilon=0.02$), despite prior work reporting bigger wins. Proposed reason: with many arms, the arm that wins the argmax has usually been boosted by its own draw anyway, so the max-with-mean clamp rarely binds.
- **The looser Auer et al. UCB bound gave worse regret**, so it was discarded in favour of the Chernoff bound.
- **Replay evaluation could not be used for display ads.** With $K$ in the thousands, matching logged random actions to policy actions discards almost all data — variance explodes. Importance weighting (Strehl et al.) also fails because it assumes a *static* policy, and a bandit changes every round. Hence the semi-synthetic click simulation.

**Posterior reshaping ablation.** $\alpha \in \{0.25, 0.5\}$ gives *lower average regret* in the non-asymptotic regime, but a fatter right tail — some runs blow up badly. Asymptotically $\alpha=1$ wins. This is the classic bias–variance-of-outcomes trade: under-exploration usually works and occasionally locks onto the wrong arm forever. Consistent with the ad result where $\alpha=0.5$ (3.72) beat $\alpha=1$ (3.81).

**News recommendation.** Normalised CTR versus a random baseline, at delays of 10 / 30 / 60 minutes. TS and optimistic TS stay competitive at all three delays. UCB is fine at 10 minutes and drops significantly by 60. At one-hour delay, (optimistic) Thompson sampling is statistically significantly best.

**Calibration check.** The diagonal Gaussian Laplace approximation is crude, but the 95% CTR confidence intervals contained the true CTR 95.1% of the time. The variance estimates were not broken by the diagonal assumption.

## Worth Remembering

- **The delay result is the transferable lesson**, and it is not really about Thompson sampling. It is about *randomised* versus *deterministic* policies under batched updates. Any deterministic argmax policy repeats its mistake across the whole batch. Any policy with sampling noise spreads its bets. If your ranker updates hourly, this applies to you.
- Thompson sampling in its plain form has **zero hyperparameters**. That is the argument for it as a baseline. The moment you add $\alpha$, you are back to tuning, and $\alpha$ buys you mean regret at the cost of tail regret.
- The display-ads clicks were **simulated**, not real. Contexts and inventory were real, and $w^*$ was a perturbed version of a real learned model, but the reward channel was synthetic. This is the paper's weakest experiment; the news result on 34M real randomised events is the strong one.
- No finite-time regret bound is offered. The authors ask for one explicitly. It arrived shortly after — Agrawal & Goyal (2012) and Kaufmann et al. (2012) proved logarithmic and near-optimal bounds, which is a large part of why Thompson sampling became respectable.
- Laplace + diagonal Gaussian is a cheap, streaming-friendly way to get a posterior over logistic-regression weights. The previous batch's posterior becomes the next batch's prior — a Kalman-flavoured online update on a $2^{24}$-dimensional model. Sampling a full weight vector per request is one Gaussian draw per active feature.
- Article features did not help in the news setting. With only ~20 arms and plenty of data per arm, a separate 21-dim weight vector per article beats sharing information across articles. Feature sharing pays off only when data is sparse.
- Practical caveat: sampling a fresh weight vector for every incoming request costs real latency at $2^{24}$ features. In production you sample lazily, only for the features that fire.
- Open question this leaves: reshaping ($\alpha$) helped in both simulation and ads, but with a heavy tail. Is there a principled schedule — anneal $\alpha$ toward 1 over time — that gets the early gains without the catastrophic runs?

## Links

Related: [[A Tutorial on Thompson Sampling]] · [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]] · [[Doubly Robust Policy Evaluation and Learning]] · [[Counterfactual Reasoning and Learning Systems]] · [[KL Divergence]] · [[Uncertainty]] · [[Beliefs]] · [[Regression Analysis]] · [[Derivative]] · [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Decision Sciences]] · [[Markov Decision Process]] · [[Random variable]] · [[Regularization]]

New topics worth writing: Laplace approximation, Conjugate priors and the Beta-Bernoulli pair, UCB1 and Lai-Robbins lower bound, Feature hashing, Batched vs sequential bandit feedback, Agrawal-Goyal finite-time regret bounds for Thompson sampling, Principal Component Analysis
