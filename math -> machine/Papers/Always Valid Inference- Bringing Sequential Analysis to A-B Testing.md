---
title: "Always Valid Inference: Bringing Sequential Analysis to A/B Testing"
authors: ["Ramesh Johari", "Leo Pekelis", "David J. Walsh"]
year: 2015
arxiv: "1512.04922"
url: https://arxiv.org/abs/1512.04922
priority: Must-Read
read_on: 2026-09-17
tags: [paper, theory]
---
## The Core Idea

A normal p-value is a promise about one specific moment. You pick the sample size $n$ in advance, you collect exactly $n$ users, you look once, and you reject if $p_n \le \alpha$. The Type I error guarantee ("at most 5% false positives") holds *only* for that single look.

Nobody behaves like that. A/B testing dashboards show a live "chance to beat baseline" number, and people watch it. They stop as soon as it crosses 95%. That is called **peeking** or **continuous monitoring**, and it destroys the guarantee. The paper's Figure 1 shows an A/A test — treatment and control are literally identical — where the live p-value dips below 0.05 at some point purely by chance. With 10,000 samples, a common size online, Type I error can go up roughly **fivefold** past the nominal $\alpha$.

The insight: instead of scolding users for peeking, change what the number means. Build a p-value process $(p_n)_{n=1}^\infty$ such that the guarantee holds *at every possible stopping rule at once*.

> [!NOTE] Always valid p-value
> A sequence $(p_n)$ is **always valid** if for *any* stopping time $T$ — any rule for deciding when to stop, including ones that look at the data — $\mathbb{P}_{\theta_0}(p_T \le s) \le s$ for all $s \in [0,1]$. The user can stop whenever they like, for whatever reason, and the false-positive rate is still bounded. ^always-valid-pvalue

Why did this not exist before? Sequential testing is old — Wald did it in 1945, and clinical trials have used it for decades. But a classical sequential test is a **black box**: you hand it $\alpha$ and a power target up front, and it tells you "stop now, reject" or "stop now, accept" at one moment. It cannot tell you anything at an arbitrary time you choose. That is fine when one statistician designs one trial. It is useless for a platform where thousands of customers each have a different $\alpha$, a different tolerance for waiting, and no statistics training.

What always valid p-values unlock is the **interface**. The dashboard still shows a p-value and a confidence interval. The user still applies the same dumb rule: reject when it drops below my $\alpha$. But now that rule is correct. And — this is the elegant part — a user who does exactly that is, without knowing it, *running a sequential test tailored to themselves*. The p-value is a compression of a whole family of sequential tests into one streaming number, and thresholding it at your personal $\alpha$ recovers the member of that family you wanted.

This is the statistical backend that shipped in Optimizely in January 2015 and has since analysed hundreds of thousands of experiments. It connects directly to [[Controlled experiments on the web- survey and practical guide (DMKD)]] — that paper tells you to fix $n$ in advance; this one tells you what to do because nobody does.

## The Methodology

**Setup.** Data $X_1, X_2, \dots \stackrel{iid}{\sim} F_\theta$ from a one-parameter exponential family, $f_\theta(x) = f_0(x)\exp(\theta x - \psi(\theta))$, where $\psi$ is the log-partition function. Test $H_0: \theta = \theta_0$ against $H_1: \theta \ne \theta_0$.

**The duality theorem (Thm 4.2).** This is the structural result. Given any sequential test — a family of stopping-and-decision rules $(T(\alpha), \delta(\alpha))$ indexed by $\alpha$, nested so that smaller $\alpha$ means waiting longer — define

$$p_n = \inf\{\alpha : T(\alpha) \le n,\ \delta(\alpha) = 1\}$$

"the smallest $\alpha$ at which the test would already have rejected by time $n$". That is an always valid p-value. Going the other way, from any always valid p-value you recover a sequential test by $\tilde T(\alpha) = \inf\{n : p_n \le \alpha\}$. The correspondence is exact for tests that **never accept $H_0$ in finite time** — so-called *tests of power one*, where a negative result just means "keep going forever". Those are the ones that translate cleanly into a monotone streaming number.

**The engine: the mSPRT.** The specific always valid p-value they ship comes from the **mixture sequential probability ratio test** (Robbins, 1970).

> [!NOTE] Mixture SPRT
> Pick a **mixing distribution** $H$ over possible effect sizes. With sample mean $s_n$, form
> $$\Lambda_n^H(s_n) = \int_\Theta \left(\frac{f_\theta(s_n)}{f_{\theta_0}(s_n)}\right)^n dH(\theta)$$
> and stop the first time $\Lambda_n^H \ge 1/\alpha$. $\Lambda_n^H$ is a martingale under the null, so Ville's inequality gives $\mathbb{P}_{\theta_0}(\sup_n \Lambda_n^H \ge 1/\alpha) \le \alpha$ — the guarantee holds at *all* $n$ simultaneously, not one. ^msprt

So the always valid p-value is simply $p_n = \min\{1,\ \min_{k \le n} 1/\Lambda_k^H(s_k)\}$. Confidence intervals fall out by inverting: $\mathsf{CI}_n = \{\theta : p_n^\theta > \alpha\}$, the set of effect sizes not yet rejected. The mixture $H$ is what stops this from being a plain likelihood ratio test — you do not know the true effect size, so you average the evidence over a distribution of candidate effects. Think of it as a prior on the alternative, though the validity does not depend on it being correct.

**The user model.** To say anything about efficiency you need a model of what users want. Theirs is deliberately crude: an **$(M, \alpha)$ user** stops at whichever comes first — the first time $p_n \le \alpha$ (reject), or time $M$ (give up). $M$ is patience. Two summary curves:

- **power profile** $\nu(\theta) = \mathbb{P}_\theta(\delta = 1)$
- **relative run-length profile** $\rho(\theta) = \mathbb{E}_\theta(T)/M$

You want $\nu$ near 1 and $\rho$ near 0.

**Three regimes.** The mSPRT's run-time satisfies (Pollak–Siegmund) $T^H(\alpha)/\log(1/\alpha) \to I(\theta,\theta_0)^{-1}$, where $I$ is the [[KL Divergence|KL]]-style information number $(\theta-\theta_0)\psi'(\theta) - (\psi(\theta)-\psi(\theta_0))$. So everything hinges on $M$ versus $\log(1/\alpha)$:

- $M \gg \log(1/\alpha)$ — "aggressive". You get $\nu \to 1$, $\rho \to 0$. Free lunch, nothing to optimise.
- $M \ll \log(1/\alpha)$ — "conservative". $\nu \to 0$ for *every* method. The experiment was hopeless; no design saves you.
- $M \sim \log(1/\alpha)$ — "Goldilocks". The only interesting case.

**Theorem 5.4 (first-order efficiency).** For Goldilocks users, as $\alpha \to 0$ with $M = O(\log(1/\alpha))$, the mSPRT's relative efficiency $\phi(M,\alpha) \to 1$. In plain terms: no other test that is at least as powerful at every effect size can have a shorter run-time at any effect size, to leading order. Crucially **this holds for any mixing distribution $H$** — the first-order result does not care which mixture you pick.

**Theorem 5.5 (choosing $H$).** $H$ matters at second order. In a Bayesian setup with true effects drawn from a prior $G$, the run-time-minimising mixture solves

$$\gamma^* \in \arg\min_\gamma\ -\mathbb{E}_{\theta \sim G}\ \mathbf{1}_{A(M,\alpha)}\, I(\theta,\theta_0)^{-1} \log h_\gamma(\theta)$$

where $A(M,\alpha) = \{\theta : I(\theta,\theta_0) \ge \log(1/\alpha)/M\}$ is the set of effects big enough to be findable within budget $M$. For Gaussian data with a $N(0,\tau^2)$ prior and $N(0,\gamma^2)$ mixture the answer is clean:

$$\gamma^{2*} = \tau^2 \cdot \frac{\Phi(-b)}{\tfrac{1}{b}\phi(b) - \Phi(-b)}, \qquad b = \left(\frac{2\log \alpha^{-1}}{M\tau^2}\right)^{1/2}$$

Match the mixture variance to the prior variance, times a correction for truncation. Impatient users (small $M$, so large $b$) should tilt the mixture towards larger effects — there is no point spending your evidence budget looking for effects you cannot afford to find.

**Two streams.** Real A/B tests have control and treatment, so this is a two-sample problem with a nuisance parameter. Trick: pair up arrivals, $W_n = (X_n, Y_n)$, and reparameterise as $(\theta, \mu)$ with $\theta = \mu_1 - \mu_0$ and $\mu = (\mu_0+\mu_1)/2$. For normal data, holding $\mu$ fixed at any value gives a genuine one-parameter exponential family $f_\theta(w) \propto \phi\!\left(\frac{y-x-\theta}{\sigma\sqrt 2}\right)$ — and since the likelihood ratio only depends on the data through $(-1,1)^\top S_n \sim N(\theta, 2\sigma^2/n)$, the whole thing is free of $\mu$. Exact Type I control for the composite null.

For **binary** data (conversion rates) it does not reduce to an exponential family. They use the CLT limit,
$$\tilde f_\theta(w) = \phi\!\left(\frac{y-x-\theta}{\sqrt{p_0^*(1-p_0^*) + p_1^*(1-p_1^*)}}\right)$$
plugging in sample means for $p_0^*, p_1^*$. This is **approximate**, and the approximation is worst exactly when $\alpha$ is moderate, because then the test fires early, before the normal approximation kicks in.

**Multiple testing.** Because the output is a p-value, you can feed it into existing multiple-testing machinery. They ask when a procedure **commutes with always validity** — whether plugging always valid p-values evaluated at a stopping time $T$ into a fixed-horizon procedure preserves the guarantee.

- **Bonferroni** commutes, always (FWER). Reason: $p_T^1, \dots, p_T^m$ are each marginally super-uniform, and Bonferroni needs nothing more.
- **BH under general dependence (BH-G)** commutes, always (FDR).
- **BH assuming independence (BH-I)** does **not** commute in general, even with genuinely independent experiments. A stopping time that watches all $m$ experiments *induces correlation* among the p-values at the moment you stop. Theorem 7.3 gives a sufficient condition on $T$ under which control is recovered; FDR is otherwise bounded by $\alpha\left(\frac{m_0}{m} + \frac{|I|\sum_{k=2}^m 1/k}{m}\right)$ where $I$ is the set of offending hypotheses.

## Ablation Studies and Experiments

**Simulation grid.** Normal data, $\alpha \in \{10^{-4}, 10^{-2}, 10^{-1}\}$, prior variance $\tau^2 \in \{10^{-4}, 10^{-2}, 10^{-1}\}$, $M \in \{10^1,\dots,10^7\}$, $B = 10^4$ Monte Carlo draws. The $\tau^2$ values are chosen to be realistic: $\tau^2 = 10^{-4}$ ≈ a 10% relative lift on a 1% conversion rate; $10^{-2}$ ≈ a 100% lift; $10^{-1}$ ≈ 1000%. Results are reported against a rescaled patience $\tilde M = M(\log\alpha^{-1}/\tau^2)^{-1}$.

**How much does the mixture $H$ matter?** They compare $\gamma \in \{\gamma^*, 1, 10^{-1}\tau, \tau, 10\tau\}$.

- Getting $\gamma$ wrong by **one order of magnitude**: < 5% drop in average power, ≤ 10% increase in average run-length. Basically fine.
- Wrong by **two orders of magnitude**: up to 20% power drop, 40% longer runs. Not fine.

So the honest reading is: the choice of prior is a *tuning knob, not a foundation*. Type I error is untouched either way — misspecification costs you speed, never correctness. That is a much better failure mode than a Bayesian method where a bad prior corrupts the inference itself.

**Where the theory breaks.** The few cases where a non-optimal $\gamma$ beat $\gamma^*$ all had $\alpha = 0.1$ and $\tilde M < 1.0$ — i.e. impatient users with loose thresholds. The asymptotics behind Theorem 5.5 assume $\alpha \to 0$, and here you can watch them fail.

**Versus fixed-horizon testing.** Proposition 5.6: with $G = N(0,\tau^2)$ and the optimal mixture, if you tune a fixed-horizon UMP test to have *the same average power*, then $\mathbb{E}_G[\rho^*]/\mathbb{E}_G[\rho_f] \to 0$. The mSPRT gets equal power in **sublinear** sample size. Empirically (Figure 3) the benefit of stopping early beats the cost of the wider always-valid boundary whenever $\hat\rho \ge 0.5$, across every parameter setting.

This is the result that should change behaviour: even a disciplined statistician who never peeks is, on average, running *longer experiments than necessary*.

**Versus other sequential tests.** They compare three always valid boundaries:

1. `mSPRT_opt` — theirs, tuned $H$
2. `r70` — Robbins (1970), $\beta(n,\alpha) = \frac{n+1}{n}\log\frac{n+1}{2\alpha}$
3. `k14` — an LIL-based bound from Kaufmann et al. (2014), $\beta(n,\alpha) = \log\alpha^{-1} + 3\log\log\alpha^{-1} + \tfrac32\log\log(en)$

`mSPRT_opt` wins on both average power and average run-length at every finite regime tested, while `r70` and `k14` trade places with each other. The interesting nuance: **`k14` has the better asymptotic rate** — LIL-based confidence sequences shrink faster as $n \to \infty$ — but it pays for that rate with worse finite-sample behaviour at low and moderate $\tilde M$. Even at $\tilde M = 100$ (a user willing to wait 100× the typical run-length), `mSPRT_opt` still holds ~5% better average power and 20–40% better run-length. A better rate is not the same as a better test in the range anyone actually operates in.

**Real deployment data.** 10,000 randomly sampled binary experiments from the platform in early 2015, split by customer subscription tier (Bronze/Silver/Gold/Platinum). Higher tiers = larger, better-optimised companies = smaller true effects. For each tier they fit a centred normal prior $G$ to the observed standardised effects, with **James–Stein shrinkage** applied first to strip out the noise in the observed effect sizes (otherwise you inflate $\tau$ with sampling error).

They then compare the mSPRT's stopping time to the fixed horizon giving 80% average power under $G$. The ratio falls below 1 with high probability, in every tier. More pointed: they also simulate an experimenter who has *private* information about the effect size, beyond what $G$ captures, and picks a fixed $n$ giving 80% power at their lower bound. To beat the mSPRT on run-time, that experimenter needs to know the effect size to a **relative error below 50%** — which, as the authors say, is rarely achievable in practice. Your guess about your own effect size is usually worse than the platform's pooled prior.

**What did not work / negative results.**

- Binary two-stream p-values are *not* exactly valid at moderate $\alpha$. The test stops before the CLT approximation is any good. Only for small $\alpha$ do simulations show approximate always-validity.
- BH-I does not commute (Example C.8): $m = 4$, three null and one non-null, with a stopping time that triggers when any null crosses $\alpha/4$, or two cross $\alpha/2$, or all three cross $3\alpha/4$. Direct calculation gives FDR $= \alpha + \frac{\alpha}{16}(2 - 9\alpha + 45\alpha^2) > \alpha$ for every $\alpha \in (0,1)$. Small violation, but real, and it proves the general claim fails.
- Stopping at "the first time *any* of a given subset of 2 to $m-1$ experiments hits significance" breaks FDR control. Stopping at "the first time exactly $x$ hypotheses are rejected" is fine — condition (14) holds trivially there, since the probability concentrates on $r = x$.

## Worth Remembering

**The conceptual move is the reusable part.** They did not invent a new test. The mSPRT is from 1970. What they invented is the *interface contract*: a streaming quantity whose guarantee is quantified over stopping times rather than over one fixed $n$. That pattern — "make the guarantee uniform over the adversary's choice, then hand the user a simple threshold" — generalises well beyond A/B testing. Anywhere a human watches a live metric and decides when to act, this is the shape of the fix.

**Type I control is robust, efficiency is not.** Worth internalising: the martingale argument gives validity for *any* mixture $H$. Get the prior wrong and you waste samples; you never generate false positives. Compare this to a naive Bayesian dashboard, where a bad prior biases the posterior itself.

**Limitations the authors state plainly:**

- **Heavy tails.** Everything assumes binary or normal observations. Real A/B metrics (revenue per user, session length) are often violently right-skewed. They cite Fithian & Wager and admit no construction exists for those models here.
- **Seasonality.** IID across visitors fails when visitors arriving at similar times are correlated — weekday/weekend, campaign spikes. Their production fix is a *heuristic*: a "reset policy" that detects likely seasonality and applies conservative corrections. Explicitly flagged as unsatisfying; a proper time-dependent mSPRT is future work.
- **Fixed allocation only.** The theory assumes visitors are randomised to arms independently of past data — 50/50, or any fixed split. They note (without proof here) that Type I control survives adaptive allocation, but the efficiency results do not.

**The bandit connection, and why they chose testing anyway.** Section 2.3 is worth reading for anyone heading towards [[Finite-time Analysis of the Multiarmed Bandit Problem (UCB1)|bandits]]. They distinguish pure exploration from regret minimisation, and argue pure exploration is the relevant framing for web experiments — the experimentation window is short relative to how long a winning feature ships for, so in-experiment reward barely matters. But they still chose hypothesis testing, for a reason that is operational rather than statistical: **the experimenter wants a confidence interval on the losing variation too**. If the winner's lift is too small to justify deployment cost, you need to know that. And the size of a null result tells you what to try next. A bandit hands you an arm, not an effect size.

Also note the technical overlap: LIL-type confidence sequences are exactly the machinery behind lil'UCB and [[Gaussian Process Optimization in the Bandit Setting (GP-UCB)|GP-UCB]]-style anytime bounds. The difference is what error you control — PAC bounds ("find the best arm if the gap exceeds $\varepsilon$") versus Type I error ("do not claim an effect when there is none"). PAC is insufficient for a platform, because the threshold $\varepsilon$ cannot be set for heterogeneous users.

**Practical caveats if you wanted to build this:**

- You need a prior on effect sizes, fitted from your own historical experiments, with shrinkage. Segment it — Optimizely found it varied strongly by customer tier, and yours will vary by surface, metric and team.
- Do not use always valid p-values with $\alpha = 0.1$ on binary data and expect the stated guarantee.
- If you layer FDR control on top, restrict stopping rules to ones that satisfy condition (14). "Stop when $x$ experiments are significant" is safe. "Stop when any of these three metrics goes green" is not.
- The always valid CI is *wider* than the fixed-horizon CI at the same $n$ — that is the price. It is a real price; the paper's claim is that the ability to stop early more than repays it once $\hat\rho \ge 0.5$.

**Follow-up questions:**

- The companion paper — Johari, Koomen, Pekelis, Walsh, *Peeking at A/B Tests* (KDD 2017) — is the implementation-focused version and is probably the better first read for the engineering details.
- Howard, Ramdas, McAuliffe & Sekhon's confidence-sequence work (2018 onward) is the modern successor and handles the non-parametric and heavy-tailed cases this paper leaves open.
- How does always validity interact with variance reduction? If [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)|CUPED]] shrinks your metric variance, the mSPRT should fire sooner — but the pre-experiment covariate adjustment changes the martingale, and it is not obvious the guarantee is preserved unmodified.
- Does this compose with switchback or cluster-randomised designs, where the IID-across-visitors assumption is violated by construction rather than by seasonality?

## Links

Related: [[Controlled experiments on the web- survey and practical guide (DMKD)]] · [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)]] · [[Finite-time Analysis of the Multiarmed Bandit Problem (UCB1)]] · [[A Tutorial on Thompson Sampling]] · [[Gaussian Process Optimization in the Bandit Setting (GP-UCB)]] · [[Uncertainty]] · [[Beliefs]] · [[KL Divergence]] · [[Random variable]] · [[Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms]] · [[Counterfactual Reasoning and Learning Systems]]

New topics worth writing: Sequential probability ratio test (Wald), Confidence sequences and the law of the iterated logarithm, Optional stopping and the peeking problem, Martingales and Ville's inequality, Benjamini–Hochberg and false discovery rate, Family-wise error rate and Bonferroni, James–Stein shrinkage, Exponential families and log-partition functions, Alpha-spending and alpha-investing, Statistical power and sample size calculation
