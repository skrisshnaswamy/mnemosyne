---
title: "Counterfactual Reasoning and Learning Systems"
authors: ["Léon Bottou", "Jonas Peters", "Joaquin Quiñonero-Candela", "Denis X. Charles", "D. Max Chickering", "Elon Portugaly", "Dipankar Ray", "Patrice Simard", "Ed Snelson"]
year: 2013
arxiv: "1209.2355"
url: https://arxiv.org/abs/1209.2355
priority: Must-Read
read_on: 2026-08-22
tags: [paper, rl]
---
## The Core Idea

A big web system — an ad engine, a recommender, a ranker — is a loop, not a function. The model's scores change what gets shown; what gets shown changes what users click; what users click becomes tomorrow's training data. So the usual question "does this new model have better accuracy?" is the wrong question. The right question is "what would total clicks and revenue have been if we had shipped this change?"

The trick here: treat the whole engine as a **structural equation model** (a list of equations saying "this variable is some unknown function of those variables plus noise"), then use **importance sampling** on logs from a *deliberately randomised* production system to answer counterfactual questions — "how would the system have performed last month if parameter $\theta$ had been different?" — without running a new A/B test.

Why this did not exist before, in practice:

- Causal inference had `do`-calculus, which is complete for identifiability but combines estimated densities in ways that blow up variance, and offers no usable error bars.
- Off-policy evaluation in contextual bandits assumed a **finite** action set and a logging policy that puts probability $\geq \epsilon$ on every action. An ad auction has continuous knobs (reserve prices) and infinitely many possible ad slates. Discretising throws away the obvious prior that a slightly higher reserve shows slightly fewer ads.
- Monte-Carlo reinforcement learning normalises the importance weights by $1/\sum_i w_i$ instead of $1/n$, which silently lies to you when part of the space was never explored.

What it unlocks: one randomised logging experiment, run cheaply and continuously in production, answers a *whole family* of counterfactual questions later, each with an honest confidence interval that separates "I need more data" from "I need different data". And the same estimator, maximised over $\theta$, becomes a learning algorithm.

> [!NOTE] Counterfactual question
> "How would the system have performed **during the data collection period** if we had used $M^*$ instead of $M$?" — a statement about a world that did not happen, estimated from the world that did. ^counterfactual-question

## The Methodology

**1. Write the system as a structural equation model.** For Bing ad placement, in causal (time) order:

$$
\begin{aligned}
x &= f_1(u,\varepsilon_1) && \text{query from user intent } u\\
a &= f_2(x,v,\varepsilon_2),\quad b = f_3(x,v,\varepsilon_3) && \text{eligible ads, bids, from inventory } v\\
q &= f_4(x,a,\varepsilon_4) && \text{scores: click probs } q_{i,p}, \text{ reserves } R_p\\
s &= f_5(a,q,b,\varepsilon_5),\quad c = f_6(a,q,b,\varepsilon_6) && \text{ad slate, click prices}\\
y &= f_7(s,u,\varepsilon_7) && \text{clicks}\\
z &= f_8(y,c,\varepsilon_8) && \text{revenue}
\end{aligned}
$$

The engineer *knows* $f_2$–$f_6$ and $f_8$ (he wrote them). He does not know $f_1$ and $f_7$ — "whoever designed the user did not leave sufficient documentation". The graph is acyclic, so algebra on these equations = interventions on the real system. Note $y$ does not depend on $q$ or $c$: users cannot see the scores or the prices. That structural fact is later worth a huge amount.

> [!NOTE] Isolation assumption
> The exogenous variables (here $u,v$ and all noise $\varepsilon_k$) are drawn i.i.d. from a fixed unknown joint distribution, factorising as $P(u,v)P(\varepsilon_1)\cdots P(\varepsilon_8)$. This is what lets you treat millions of page views as repeated independent trials. It is *part of the counterfactual condition*, not a fact about the world — it says "assume users and advertisers do not react". You must therefore also track proxies for user and advertiser happiness. ^isolation-assumption

**2. Turn it into a Markov factorisation.** Simulating the SEM defines a joint distribution
$$P(\omega)=P(u,v)P(x|u)P(a|x,v)P(b|x,v)P(q|x,a)P(s|a,q,b)P(c|a,q,b)P(y|s,u)P(z|y,c).$$
The SEM is an algebraic object you can manipulate; the Bayesian network is a distribution you can sample. They share the same symbols.

**3. Intervene by swapping one factor.** A new scoring model replaces $P(q|x,a)$ with $P^*(q|x,a)$; every other factor is unchanged. So the importance weight collapses to a ratio of just the changed factors:

$$Y^* = \int_\omega \ell(\omega)P^*(\omega) \approx \frac{1}{n}\sum_{i=1}^n \ell(\omega_i)\,w_i,\qquad w_i=\frac{P^*(\omega_i)}{P(\omega_i)}=\frac{\text{factors only in } P^*}{\text{factors only in } P}.$$

You never need to know $f_1$ or $f_7$. You only need to know the factors you changed.

**4. Randomise production so the denominator is nonzero.** Before each auction they draw $\varepsilon\sim\mathcal{N}(0,1)$ and multiply **all mainline reserves** by $m=\rho e^{-\sigma^2/2+\sigma\varepsilon}$ — log-normal with mean $\rho$. Live settings: $\rho=1$, $\sigma=0.3$, so 95% of multipliers land in $[0.52, 1.74]$. **22 million search result pages over five consecutive weeks.** Pages, not users, were assigned to buckets, to match the isolation assumption. Then $w_i = p(m_i;\rho^*,\sigma^*)/p(m_i;\rho,\sigma)$, a ratio of two log-normal densities — no discretisation, continuous knob, done.

**5. Clip the weights and report two intervals.** Raw importance weights are heavy-tailed; the variance can be infinite. Assume $\ell(\omega)\in[0,M]$, pick a ceiling $R$ (in practice: **the fifth largest ratio observed in the data**), and zero out weights above it:
$$\bar w(\omega)=w(\omega)\,\mathbb{1}\{P^*(\omega)<R\,P(\omega)\}.$$
This gives a *clipped* expectation $\bar Y^*\le Y^*$, plus a bound on what you threw away:
$$\bar Y^* \le Y^* \le \bar Y^* + M(1-\bar W^*),\qquad \bar W^*=\mathbb{E}[\bar w(\omega)].$$
Two error bars come out, and they mean different things:

- **Outer interval** $\widehat Y^*\pm\epsilon_R$ — sampling noise. Too wide → *collect more data with the same setup*.
- **Inner interval** $[\bar Y^*,\ \bar Y^*+M(1-\widehat W^*+\xi_R)]$ — the mass of the counterfactual world your logging policy never visited. Too wide → *no amount of extra data helps; change the randomisation*.

Both $\epsilon_R,\xi_R$ come from empirical Bernstein bounds (Maurer & Pontil 2009), $\mathcal{O}(n^{-1/2}\log\delta)$, and hold for the worst-case distribution rather than relying on the central limit theorem to have kicked in.

**6. Two structural tricks that shrink the intervals.**

*(a) Move the reweighting variable.* You do not have to reweight at the point of intervention. Reweight at any variable set that **intercepts every causal path from intervention to measurement**. Since clicks depend on the slate $s$ and not on the scores $q$, marginalise $q$ out and reweight on the *slate*:
$$w_i=\frac{P^*(s_i|x_i,a_i,b_i)}{P(s_i|x_i,a_i,b_i)}=\frac{\Psi(m_i^{\max};\rho^*,\sigma^*)-\Psi(m_i^{\min};\rho^*,\sigma^*)}{\Psi(m_i^{\max};\rho,\sigma)-\Psi(m_i^{\min};\rho,\sigma)},$$
where $[m_i^{\min},m_i^{\max}]$ is the whole *range* of multipliers that would have produced the slate you actually saw, and $\Psi$ is the log-normal CDF. Many different reserves produce the same slate, so the weight is far better behaved. Pages with no eligible ads get $w_i=1$ exactly, because the outcome (zero clicks) was inevitable under every counterfactual.

*(b) Subtract a predictor (doubly robust).* Let $\upsilon$ be the **invariant variables** — observed things that are not downstream of the intervention (time of day, season). Build a predictor $\zeta$ of $\ell$ that depends only on invariants or on known functions. Then
$$Y^*\approx \frac{1}{n}\sum_i \zeta_i^* \;+\; \frac{1}{n}\sum_i(\ell(\omega_i)-\zeta_i)\,w(\omega_i).$$
The first term is simulated by replay (no big weights). The second has smaller magnitude, so smaller variance. This is exactly Dudík et al.'s doubly-robust bandit estimator, re-derived as "put the predictor into the causal graph". For **differences** $Y^+-Y^*$ the weights $\Delta w = (P^+-P^*)/P$ can be negative, so centring matters enormously — *always* use a predictor, even a constant one.

**7. Learning = maximise the lower bound.** Do not maximise the unbiased estimate; it peaks where you have no data. Instead
$$\theta^* = \arg\max_\theta \widehat Y^\theta \quad\text{(the clipped estimate)}$$
and prove it with a **uniform** confidence interval over all $\theta$ simultaneously — union bound for finite $\mathcal F$, or a uniform empirical Bernstein bound with growth function $\mathcal M(n)=10\,\mathcal N(2n,\mathcal F,1/n)$ over the function class $\{\omega\mapsto\ell(\omega)\bar w(\omega),\ \omega\mapsto\bar w(\omega)\}$. Ordinary per-$\theta$ intervals are useless here precisely because the optimiser hunts for the $\theta$ where they break.

**8. Derivatives fall out for free.** With $w_\theta=P^\theta/P$,
$$\frac{\partial Y^\theta}{\partial\theta}\approx\frac{1}{n}\sum_i[\ell(\omega_i)-\zeta(\upsilon_i)]\,w_\theta(\omega_i)\frac{\partial \log P^\theta(\omega_i)}{\partial\theta}.$$
When $P=P^\theta$ the ratio vanishes and this **is** the policy gradient / REINFORCE estimator, with $\zeta$ as the baseline. The optimal constant baseline is $\zeta=\mathbb{E}[\ell w_\theta'^2]/\mathbb{E}[w_\theta'^2]$.

## Ablation Studies and Experiments

**The motivating failure: Simpson's paradox in live ad logs.** Kidney stones first (Charig 1986): open surgery 78% (273/350), lithotripsy 83% (289/350) — but split by stone size, surgery wins both: 93% vs 87% on small stones, 73% vs 69% on large. The confounder is that doctors gave the gentler treatment to the easier cases.

Then the same shape in Bing data — 2000 pages per group, three hours of traffic, click-through rate on the **second** mainline ad:

| | Overall | $q_2$ low | $q_2$ high |
|---|---|---|---|
| $q_1$ low | 6.2% (124/2000) | **5.1%** (92/1823) | **18.1%** (32/176) |
| $q_1$ high | **7.5%** (149/2000) | 4.8% (71/1500) | 15.6% (78/500) |

Overall it looks like a high-scoring top ad *helps* the ad below it. Split by $q_2$ and it flips: it *hurts*. The likely story: commercial queries drive both $q_1$ and clicks everywhere on the page — a common cause. And the authors are honest that they still cannot conclude: a "better" click model would lower $q_2$ on commercial pages, push those ads below reserve into the sidebar, and probably *lose* clicks and money despite being more accurate. "Making sense out of such data is just too complex!" That is the whole argument for the machinery.

**Validation of the counterfactual estimate (Fig. 13).** Estimated mainline-ads-per-page, clicks-per-page and revenue-per-page as a function of $\rho^*$ from the 22M-page randomised bucket. A **separate live bucket ran with mainline reserves cut ~18%**; the measured metrics land on the predicted curves. That is the money result: the counterfactual estimate matched the real A/B test.

**Cost of randomisation.** A third, unrandomised control bucket ran in parallel. Randomisation caused a **small but statistically significant increase in mainline ads per page**; click yield and revenue differences were **not significant**. So the tax for permanent exploration was near zero.

**Ablation: reweighting on scores vs on slates (Fig. 13 vs Fig. 14).** Same data, same estimand. Reweighting on $q$ gives inner intervals that blow up as soon as $\rho^*$ leaves the explored range. Reweighting on the *slate* $s$ keeps them tight over a much wider range of multipliers. Same samples, better answer, purely by exploiting the graph structure ($y \perp q \mid s$). This is the clearest single ablation in the paper: the estimator is not what's doing the work, the causal structure is.

**What did not work / what needed a patch:**

- Revenue could **not** be estimated with $s$ alone, because $z$ depends on scores through prices $c$, and $s$ does not intercept that path. Fix: define a filtered price variable $\tilde c = f(c,y)$ that drops prices for ads that got no click, and reweight on the joint $(s,\tilde c)$.
- **Pointwise** estimates (no randomisation at all, $\sigma\to 0$) give huge intervals. Workaround: since $Y_\nu(\rho)\approx Y_0(\rho)+\nu Y_0''(\rho)/2$, estimate at two variance levels and extrapolate, $Y_0(\rho)\approx 2Y_\nu(\rho)-Y_{2\nu}(\rho)$.
- Advertiser values $V_a\approx(\partial Y_a/\partial b_a)/(\partial Z_a/\partial b_a)$ cannot be recovered for three groups: advertisers whose bids are too low to ever show; advertisers shown too rarely for stable derivatives (partially fixed by pooling similar advertisers); and advertisers who bid absurdly high to grab everything, where both derivatives are zero and only a *lower bound* on $V_a$ is obtainable (capping bids at $b_{\max}$ at least yields data). The authors read these as failures of the rational-advertiser model itself, not of the estimator.
- You cannot randomise bids directly — advertisers expect to pay based on what they bid. Neat workaround: since rank-score is $r_{i,p}=b_i q_{i,p}$, a multiplier on the bid is algebraically a multiplier on the estimated click probability. Show the same ads, charge prices as if the probability moved, log data as if the bid moved.

**Learning experiment: auction tuning (§6.3).** Two knobs per query cluster $k$: a squashing exponent $\alpha_k$ (Lahaie & McAfee, rank-score $r_{ip}=\gamma_p b_i \beta_i(x)^{\alpha}$; $\alpha<1$ trusts bids more and the click model less) and a reserve multiplier $\rho_k$. Randomised $\alpha$ normally, $\rho$ log-normally; **12 million pages over four weeks**. They maximise estimated *advertiser value* (not publisher revenue — "the size of the pie, not the publisher's slice", less likely to just raise prices), subject to a global cap on mainline ads per page. Fig. 21 shows level curves over a $(\rho,\alpha)$ grid for one cluster: mainline ads varying $-6\%$ to $+10\%$, advertiser value from 164 to 169 in arbitrary units. Off-policy counterfactual derivatives drive the optimiser.

The pointed argument against the obvious baseline: the alternative is to replay the auctions and simulate users with a click model. But $\alpha$ exists *to compensate for the uncertainty of the click model*. Using that same click model to tune $\alpha$ is circular. The counterfactual approach uses real user behaviour under mild randomisation instead.

## Worth Remembering

- **The single most transferable lesson**: put randomisation into production permanently, log the random draw, and you buy the ability to answer questions you have not thought of yet. Section 4.7 lists what one reserve-randomisation experiment retroactively answers: different randomisation variances, query-dependent reserve multipliers $\rho^*(x)$ (just reweight with $p(m_i;\rho^*(x_i),\sigma)$), pointwise values. None of these needed a new experiment.
- **Inner vs outer interval is the practical gift.** Most off-policy evaluation gives you one number and one error bar. Here you get a diagnostic: more data, or different data? That distinction is what makes sequential design possible.
- **Limits the authors admit.** Sequential design (§6.4) has no optimality theory here — bandit regret bounds assume the outcome of one action tells you nothing about others, which is "both unrealistic and pessimistic" for ads. They fall back on maximising $\widehat Y^\theta$ each round plus ad-hoc exploration floors, noting that Thompson sampling and even fixed heuristics do fine over the horizon where the problem is stationary: "leveraging the problem structure seems more important in practice than perfecting an otherwise sound exploration strategy."
- **The equilibrium section (§7) is the speculative part.** They assume rational advertisers with utility $U_a^\theta(b_\star)=V_a Y_a - Z_a$, cite Athey & Nekipelov that smooth noise in the auction makes the discrete problem differentiable with a unique Nash equilibrium, estimate $V_a$ from the first-order conditions, then differentiate the equilibrium equations to get $db_a = \Xi_a\,d\theta$ and hence $dY=(\partial Y/\partial\theta + \sum_a \Xi_a \partial Y/\partial b_a)\,d\theta$. Honest caveats: this misses advertisers who did not bid but would have; near-singular systems mean the rational assumption has stopped determining behaviour. The programme was "not yet fully realized" as of publication.
- **Splitting advertisers into treatment/control is basically impossible** because each auction involves many of them at once. Splitting users is easy. Controlling both simultaneously — "probably impossible". This is a structural reason A/B testing is insufficient in marketplace systems, not just a cost argument. Relevant to anyone running two-sided platforms.
- **Why the ratio-of-factors form matters practically.** Replay (§4.1) needs to know *every* function on the causal path from intervention to measurement — fine for a classifier and a loss, impossible when a human is in the path. Reweighting needs only the factors that *differ*. Different knowledge requirements, and you can mix the two (that's what the predictor trick is).
- **Follow-up questions.** How does clipping at "the fifth largest observed ratio" behave as $n$ grows — is it stable, or does it silently tighten with sample size? How much of the slate-reweighting win survives when many scores are randomised independently instead of one global multiplier (they mention it needs integrating a multivariate Gaussian, Genz 1992, "details in a forthcoming publication")? And how would you build the invariant predictor $\zeta$ in a modern stack — presumably a small model trained on invariant features only, which is a nice, cheap engineering job.
- **Lineage.** This is the intellectual ancestor of modern off-policy evaluation and of logged-bandit-feedback training, and the derivative section is literally the policy gradient with a baseline. If you have read the RLHF paper, the connection is: PPO against a reward model is one particular way to optimise a counterfactual estimate; this paper is the general framework, with error bars, for the case where you cannot afford a simulator.

## Links

Related: [[Markov Decision Process]] · [[Markov Property]] · [[Decision Sciences]] · [[Uncertainty]] · [[Random variable]] · [[Markov Chain Monte Carlo]] · [[Derivative]] · [[Training language models to follow instructions with human feedback]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Recommender Systems - Evolution]] · [[Principal Data Scientist — Experimentation & Causal Inference Interview Question Bank]] · [[experimentation_question_bank]] · [[Foundational_Reading_Plan]]

New topics worth writing: Importance sampling, Structural equation models, Simpson's paradox, do-calculus and identifiability, Off-policy evaluation, Doubly robust estimators, Empirical Bernstein bounds, Policy gradient / REINFORCE, Contextual bandits, Generalized second price auction, Covariate shift, Uniform convergence and growth functions, Weight clipping and effective sample size
