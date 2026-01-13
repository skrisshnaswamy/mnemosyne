---
title: "Double/Debiased Machine Learning for Treatment and Structural Parameters"
authors: ["Victor Chernozhukov", "Denis Chetverikov", "Mert Demirer", "Esther Duflo", "Christian Hansen", "Whitney Newey", "James Robins"]
year: 2016
arxiv: "1608.00060"
url: https://arxiv.org/abs/1608.00060
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper]
---
## The Core Idea

You want one number: the effect of a treatment $D$ on an outcome $Y$, after adjusting for a big pile of controls $X$. The controls matter in some complicated unknown way, so you would like to use a flexible machine learning model — a random forest, a lasso, a boosted tree — to soak them up.

The obvious thing does not work. If you fit an ML model $\widehat g_0(X)$ for the nuisance part and then plug it into the estimating equation for $\theta_0$, your estimate of $\theta_0$ is badly biased. Not slightly biased — biased enough that it does not even converge at the normal $1/\sqrt{N}$ rate, so every confidence interval you build is wrong. The paper shows a histogram of this: a random forest fitting a *smooth* function of a *few* variables still produces an estimate shifted far to the right of the truth.

Two separate things are causing that.

1. **Regularization bias.** ML methods trade bias for variance on purpose. That is why they work. But the bias in $\widehat g_0$ leaks straight into $\widehat\theta_0$.
2. **Overfitting bias.** The residuals of observation $i$ are correlated with the fitted function, because observation $i$ was used to fit the function.

The fix is two ingredients, each aimed at one problem.

> [!NOTE] Neyman orthogonality
> A score (estimating equation) $\psi(W;\theta,\eta)$ is Neyman orthogonal if its expectation is *locally insensitive* to the nuisance parameter $\eta$ at the truth:
> $$\partial_\eta \, \mathrm{E}_P\,\psi(W;\theta_0,\eta_0)[\eta-\eta_0] = 0$$
> Small errors in $\eta$ have no first-order effect on where the moment condition is zero. ^neyman-orthogonality

> [!NOTE] Cross-fitting
> Split the data into $K$ folds. Fit the nuisance functions on the other $K-1$ folds, evaluate the score on the held-out fold, swap roles, average. Every observation is used both ways, so no efficiency is lost, but no observation's own residual is ever contaminated by its own fitted value. ^cross-fitting

Together these give you a $\theta_0$ estimate that is $\sqrt{N}$-consistent, approximately normal, and — the part economists care about — **uniformly** valid over a large class of data-generating processes, so the confidence intervals do not silently break under small perturbations of the truth.

What this unlocks: you can now use *any* sufficiently good ML method for the nuisance functions and still get honest standard errors on a causal parameter. The theory needs only rate conditions, not the classical low-complexity (Donsker) assumptions that high-dimensional ML violates by construction.

## The Methodology

### The lead example: partially linear regression

$$Y = D\theta_0 + g_0(X) + U, \qquad \mathrm{E}[U\mid X,D]=0$$
$$D = m_0(X) + V, \qquad \mathrm{E}[V\mid X]=0$$

$\theta_0$ is the thing you want. $g_0$ and $m_0$ are the nuisance parameters $\eta_0=(g_0,m_0)$, both possibly nasty functions of many variables.

**The naive estimator.** Learn $\widehat g_0$, then
$$\widehat\theta_0 = \Big(\tfrac1n\sum_{i\in I} D_i^2\Big)^{-1}\tfrac1n\sum_{i\in I}D_i\big(Y_i - \widehat g_0(X_i)\big)$$
Decompose $\sqrt n(\widehat\theta_0-\theta_0) = a + b$. Term $a$ is fine, converges to a normal. Term $b$ is the killer:
$$b \approx (\mathrm{E}D^2)^{-1}\frac{1}{\sqrt n}\sum_{i\in I} m_0(X_i)\big(g_0(X_i)-\widehat g_0(X_i)\big)$$
These summands do **not** have mean zero, because $D_i$ is centred at $m_0(X_i)\ne 0$. If $\widehat g_0$ converges at rate $n^{-\varphi_g}$ with $\varphi_g<1/2$ (which regularised methods always do), then $b$ is of order $\sqrt n\,n^{-\varphi_g}\to\infty$. Diverges.

> [!NOTE] Regularization bias
> The bias term is a $\sqrt n$-scaled average of $m_0(X)\times(\text{error in }\widehat g_0)$. It is non-zero precisely because $D$ is *confounded* — because $m_0(X)\neq 0$. If treatment were purely random, this term would vanish. ^regularization-bias

**The orthogonal estimator.** Also learn $\widehat m_0$, form the residualised treatment $\widehat V = D - \widehat m_0(X)$, and solve
$$\check\theta_0 = \Big(\tfrac1n\sum_{i\in I}\widehat V_i D_i\Big)^{-1}\tfrac1n\sum_{i\in I}\widehat V_i\big(Y_i-\widehat g_0(X_i)\big)$$
Now the bias term becomes
$$b^* = (\mathrm{E}V^2)^{-1}\frac{1}{\sqrt n}\sum_{i\in I}\big(\widehat m_0(X_i)-m_0(X_i)\big)\big(\widehat g_0(X_i)-g_0(X_i)\big)$$
It is a **product** of two errors. Bounded by $\sqrt n\, n^{-(\varphi_m+\varphi_g)}$. This vanishes if $\varphi_m+\varphi_g > 1/2$ — so each function needs only the crude rate $o(N^{-1/4})$, or one can be slow if the other is fast. This is the whole "double" in double machine learning: you fit *two* prediction problems, and their errors multiply instead of adding.

The corresponding score is
$$\psi(W;\theta,\eta) = \big(Y - D\theta - g(X)\big)\big(D - m(X)\big)$$
Check: the Gateaux derivative in $\eta$ at $(\theta_0,\eta_0)$ is zero, because $\mathrm{E}[V\mid X]=0$ kills one term and $\mathrm{E}[U\mid X,D]=0$ kills the other. Orthogonal. The non-orthogonal version $\varphi(W;\theta,g)=(Y-D\theta-g(X))D$ fails this unless $m_0(X)=0$.

An equivalent Robinson-style parameterisation uses $\ell_0(X)=\mathrm{E}[Y\mid X]$ instead of $g_0$:
$$\psi = \big(Y-\ell(X)-\theta(D-m(X))\big)\big(D-m(X)\big)$$
Slightly nicer in practice because both nuisances are plain conditional means you can hand to any regressor.

### Why sample splitting, separately

Even with an orthogonal score there is a remainder $c^*$ containing terms like
$$\frac{1}{\sqrt n}\sum_{i\in I} V_i\big(\widehat g_0(X_i)-g_0(X_i)\big)$$
If $\widehat g_0$ was fit on a *different* fold, then conditional on that fold $\widehat g_0$ is a fixed function, $\mathrm{E}[V_i\mid X_i]=0$, so this term has mean zero and variance $\tfrac1n\sum(\widehat g_0-g_0)^2 \to 0$. Chebyshev finishes it. Two lines.

Without splitting, this can explode. The paper's deliberately contrived example: let $\widehat g_0(X_i)=g_0(X_i)+(Y_i-g_0(X_i))/N^{1/2-\epsilon}$ for in-sample $i$. This converges uniformly at the near-parametric rate $N^{-1/2+\epsilon}$ — an *excellent* estimator by any rate criterion — yet the remainder becomes $\propto N^{\epsilon}\to\infty$. The histogram is shifted hard left. Cross-fitting removes the bias entirely and the spread does not widen.

> [!NOTE] Donsker condition
> The classical semiparametric trick for controlling these remainders: assume the nuisance function class has bounded entropy integral. This rules out even a unit-ball linear model with $p_N$ growing regressors, whose log covering number grows like $p_N$. Any high-dimensional ML method breaks it. Cross-fitting replaces it with nothing. ^donsker

### The two algorithms

Both take a $K$-fold partition $(I_k)$, fit $\widehat\eta_{0,k}$ on $I_k^c$.

**DML1** — solve per fold, then average:
$$\mathbb{E}_{n,k}[\psi(W;\check\theta_{0,k},\widehat\eta_{0,k})]=0,\qquad \tilde\theta_0=\frac1K\sum_k \check\theta_{0,k}$$

**DML2** — pool first, solve once:
$$\frac1K\sum_{k=1}^K \mathbb{E}_{n,k}[\psi(W;\tilde\theta_0,\widehat\eta_{0,k})]=0$$

The authors recommend DML2. The pooled empirical Jacobian is more stable than $K$ separate ones. Asymptotically identical.

Variance:
$$\widehat\sigma^2 = \widehat J_0^{-1}\Big(\tfrac1K\sum_k \mathbb{E}_{n,k}[\psi\psi']\Big)(\widehat J_0^{-1})',\qquad \widehat J_0 = \tfrac1K\sum_k \mathbb{E}_{n,k}[\psi^a(W;\widehat\eta_{0,k})]$$
where $\psi=\psi^a\theta+\psi^b$ for linear scores.

### The main theorem

Under Neyman ($\lambda_N$-near-)orthogonality plus rate conditions on $r_N, r_N', \lambda_N'$:
$$\sqrt N\,\sigma^{-1}(\tilde\theta_0-\theta_0)=\frac{1}{\sqrt N}\sum_{i=1}^N \bar\psi(W_i)+O_P(\rho_N)\rightsquigarrow N(0,\mathrm{I})$$
uniformly over $P\in\mathcal{P}_N$, with $\bar\psi=-\sigma^{-1}J_0^{-1}\psi(\cdot,\theta_0,\eta_0)$ and remainder
$$\rho_N := N^{-1/2}+r_N+r_N'+N^{1/2}\lambda_N+N^{1/2}\lambda_N'$$

The crude sufficient condition is $\varepsilon_N = o(N^{-1/4})$ on each nuisance. But when second cross-derivatives vanish ($\lambda_N'=0$) — known propensity score, randomised trials, the optimal-instrument problem — you only need $\varepsilon_N=o(1)$, i.e. bare consistency.

### Constructing orthogonal scores when you do not have one

Four recipes, all in Section 2:

- **Neyman's original (finite-dim nuisance).** From $\ell(W;\theta,\beta)$, use $\psi=\partial_\theta\ell - \mu\,\partial_\beta\ell$ with $\mu_0=J_{\theta\beta}J_{\beta\beta}^{-1}$. For high-dimensional linear regression this gives exactly $\psi=(Y-D\theta-X'\beta)(D-\mu X)$ with $\mu_0=\gamma_0'$ — the debiased-lasso score.
- **Concentrating out (infinite-dim nuisance).** Maximise the objective over $\beta$ for each $\theta$, plug $\beta_\theta$ back in, differentiate: $\psi(W;\theta,\eta)=d\ell(W;\theta,\eta(\theta))/d\theta$. Applied to squared loss this reproduces the PLR score above. Works for any criterion, not just log-likelihood; if it *is* a log-likelihood you also get the efficient score for free.
- **GMM.** $\psi=\mu\, m(W;\theta,\beta)$ with $\mu_0=A'\Omega^{-1}-A'\Omega^{-1}G_\beta(G_\beta'\Omega^{-1}G_\beta)^{-1}G_\beta'\Omega^{-1}$. Setting $A=G_\theta$, $\Omega=\mathrm{Var}(m)$ gives semiparametric efficiency.
- **Influence-function adjustment.** Take any score $\varphi(W;\theta,\beta)$ and add the Newey (1994) correction term $\phi$ for the estimation of $\beta$: $\psi=\varphi+\phi$. For PLR, $\phi = -m_0(X)\{Y-D\theta-\beta(X,\theta)\}$, giving the same orthogonal score again.

If $J_{\beta\beta}$ is ill-conditioned you solve a relaxed problem $\min\|\mu\|$ s.t. $\|J_{\theta\beta}-\mu J_{\beta\beta}\|_q \le r_N$, which buys you **near**-orthogonality ($\lambda_N=o(N^{-1/2})$) instead of exact, trading some efficiency for robustness.

### Treatment effect scores

**ATE**, $\theta_0=\mathrm{E}[g_0(1,X)-g_0(0,X)]$, using the Robins–Rotnitzky doubly-robust influence function:
$$\psi = \big(g(1,X)-g(0,X)\big) + \frac{D(Y-g(1,X))}{m(X)} - \frac{(1-D)(Y-g(0,X))}{1-m(X)} - \theta$$

**ATTE**:
$$\psi = \frac{D(Y-\bar g(X))}{p} - \frac{m(X)(1-D)(Y-\bar g(X))}{p(1-m(X))} - \frac{D\theta}{p}$$
with $\bar g_0(X)=g_0(0,X)$, $p_0=\mathrm{E}[D]$. Note you never need $g_0(1,X)$.

**LATE** with binary instrument $Z$: numerator and denominator each get their own doubly-robust expression, and $\theta$ multiplies the denominator inside a single score. Nuisances are $\mu_0(Z,X)$, $m_0(Z,X)$, $p_0(X)$.

All three are efficient — they hit the Hahn (1998) / Chamberlain (1992) semiparametric bound.

### Handling split randomness

The particular split matters in finite samples. Run the whole thing $S$ times and report
$$\tilde\theta_0^{\text{median}}=\text{median}\{\tilde\theta_0^s\},\qquad \widehat\sigma^{2,\text{median}}=\text{median}\{\widehat\sigma_s^2+(\widehat\theta_s-\tilde\theta^{\text{median}})(\widehat\theta_s-\tilde\theta^{\text{median}})'\}$$
The variance formula adds the between-split spread to the within-split variance. Medians preferred over means for robustness to outlier splits. This is first-order equivalent to a single split, so it changes nothing asymptotically — it is purely a finite-sample honesty device.

## Ablation Studies and Experiments

Three empirical applications, each run with seven nuisance estimators: Lasso, single regression tree (CART, 10-fold CV penalty), Random Forest (1000 trees), Boosting (10-fold CV), Neural Net (2 or 8 neurons, decay 0.01–0.02), an **Ensemble** (weights on Lasso/Boosting/Forest/NN chosen by 5-fold CV to minimise out-of-sample MSE, weights sum to 1), and **Best** (pick the winning method separately for each nuisance function). 100 sample splits, median method throughout.

### Pennsylvania Reemployment Bonus (randomised)

$Y=\log$ unemployment duration, $D=$ assigned to treatment 4 (most generous cash bonus). Since treatment is random, the propensity is just the treated fraction.

ATE, interactive model, 5-fold: $-0.081$ (Lasso), $-0.085$ (Tree), $-0.074$ (Forest), $-0.077$ (Boosting), $-0.073$ (NN), $-0.078$ (Ensemble), $-0.077$ (Best). Standard errors all $0.036$. Partially linear model gives $-0.073$ to $-0.084$.

Every method significant at 5%. **The two standard errors — plain median, and median adjusted for split variation — are identical to three decimals.** In a randomised experiment the split does not matter.

### 401(k) eligibility on net financial assets

$Y=$ net financial assets, $D=$ eligible for 401(k). Nine raw controls (age, income, family size, education, married, two-earner, DB pension, IRA, home ownership). Propensity trimmed at $[0.01, 0.99]$ in the interactive model.

**No controls at all: $\$19{,}559$ (se $1413$).** That is the confounded number.

With flexible controls, ATE 5-fold, interactive model: $\$7{,}170$ (Lasso) through $\$8{,}105$ (Forest), Ensemble $\$7{,}839$. Partially linear model: $\$8{,}187$–$\$9{,}247$. So roughly a 55–60% attenuation once income and job-choice variables are properly controlled.

LATE of 401(k) *participation*, instrumented by eligibility: $\$8{,}944$ (Lasso) to $\$11{,}764$ (Forest), Ensemble $\$11{,}173$. Compare the classic linear IV with the Poterba et al. hand-picked specification: $\$13{,}102$ (se $1922$). Mild attenuation, suggesting the intuitive specification was slightly too simple.

**What the ablations reveal here:**

- **5-fold beats 2-fold on standard errors for every method.** More data in the auxiliary sample $\Rightarrow$ better-learned nuisances $\Rightarrow$ tighter $\theta$. Intuitive, and the authors explicitly warn it does **not** generalise (see the next example).
- **Lasso has visibly larger split-adjusted standard errors.** 401(k) LATE 2-fold: bracket se $2192$, split-adjusted se $3014$. Compare Boosting: $1666 \to 1718$. The authors' explanation: linear models extrapolate badly. When the main fold contains observations outside the auxiliary fold's support, the linear model has to extrapolate and its prediction errors blow up. Supporting evidence: Lasso's se drops from $3014$ (2-fold) to $3307$... actually the 5-fold LATE Lasso se rises, but in the ATE table Lasso's adjusted se falls from $1530$ (2-fold) to $1398$ (5-fold), consistent with less extrapolation when the auxiliary sample is larger.
- **Split-adjusted standard errors are noticeably larger than unadjusted ones here**, but never change a conclusion.

### Institutions on economic growth (Acemoglu–Johnson–Robinson)

Only $N=64$ countries. $Y=\log$ GDP per capita, $D=$ expropriation-risk protection index, $Z=$ settler mortality. Controls: distance from equator plus continent dummies. AJR *assumed* geography enters linearly; DML relaxes that to an unknown learned function.

2-fold: $0.85$ (Lasso), $0.81$ (Tree), $0.84$ (Forest), $0.77$ (Boosting), $0.94$ (NN), $0.80$ (Ensemble), $0.83$ (Best).
5-fold: $0.77$, $0.95$, $0.90$, $0.73$, $1.00$, $0.83$, $0.88$.

AJR's baseline: $1.10$ (se $0.46$). DML estimates are somewhat smaller, all significant at 5%, qualitatively the same conclusion.

**Here 5-fold standard errors are *larger* than 2-fold for everything except Lasso** — the exact opposite of the 401(k) result. The authors flag this openly as an open question: there is no general relationship between $K$ and precision of $\theta$. Neural Net was dropped from the Ensemble because $N=64$ made training unstable.

### What did not work / what the theory rules out

- **The naive plug-in.** Figure 1, left panel. Random forest, smooth $g_0$, few variables — the most favourable possible setting — and the estimator is still badly shifted and non-normal.
- **Full-sample nuisance fitting.** Figure 2, left panel. Even a nuisance estimator converging at $N^{-1/2+\epsilon}$ produces divergent bias.
- **Entropy-growth conditions as an alternative to splitting.** They exist (Belloni et al. 2017) but impose "unnecessarily strong restrictions." In the sparse optimal-IV model, no splitting requires $s^2 \ll n$; splitting requires only $s \ll n$. In PLR/ATE, no splitting needs $(s^g)^2+(s^m)^2 \ll N$; splitting needs only $s^g s^m \ll N$. So a very sparse propensity buys you a very dense regression function, and vice versa.
- **The efficient PLR score requires estimating heteroskedasticity.** The Chamberlain-efficient score involves $\sigma(D,X)^2=\mathrm{E}[U^2\mid D,X]$ in three places. The authors deliberately choose $\Omega(R)=1$ instead, giving the simple $(D-m_0(X))(Y-D\theta-g_0(X))$ score. Efficient only under homoskedasticity — they trade efficiency for not having to smooth a conditional variance.

## Worth Remembering

**The headline practical rule.** Fit $\mathrm{E}[Y\mid X]$ and $\mathrm{E}[D\mid X]$ with whatever ML you like, on held-out folds, residualise both, regress residual-on-residual. That is DML for the partially linear model, and it is five lines of code. The theory is there to tell you it is *valid*, not to make it complicated.

**Orthogonality and cross-fitting fix different things.** Orthogonality kills regularization bias — the first-order effect of $\widehat\eta$ being systematically off. Cross-fitting kills overfitting bias — the correlation between residuals and fitted values. You need both. Neither substitutes for the other.

**The $N^{-1/4}$ rate is the headline but not the binding constraint.** What actually matters is the *product* $\|\widehat m-m_0\|_{P,2}\times\|\widehat g-g_0\|_{P,2} \le \delta_N N^{-1/2}$. Very asymmetric splits of that budget are fine. In a randomised experiment where the propensity is known exactly, you need only consistency for the outcome model — which is why the Pennsylvania example is so stable across methods.

**Uniformity is the point, not just pointwise normality.** The result holds uniformly over an expanding class $\mathcal{P}_N$, meaning it survives any sequence $P_N \in \mathcal{P}_N$. Non-orthogonal methods can be shown to *fail* this. This is what separates "my confidence interval covers 95% asymptotically for this fixed truth" from "my confidence interval covers 95% and does not break if the truth moves a bit."

**Limitations the authors admit.**
- No guidance on choosing $K$. They suggest 4 or 5 over 2 from empirical experience, but the AJR example contradicts the 401(k) example on whether more folds help precision. Genuinely open.
- $S\to\infty$ behaviour of the repeated-split correction is unstudied. They say it "would be interesting to investigate."
- The rate conditions are non-primitive: you must import convergence-rate guarantees for whatever ML method you use, and those are method- and assumption-specific.
- Propensity trimming ($0.01/0.99$ in the 401(k) example) is a hack with no theory attached. Overlap violations remain overlap violations.
- The nonlinear-score theorem additionally requires a bounded parameter space and a uniform covering entropy bound $\log N(\epsilon\|F\|,\mathcal{F},\|\cdot\|)\le v\log(a/\epsilon)$ on the *score* class (not the nuisance class) — a mild condition but it is there.

**A caveat that shows up in the numbers.** Lasso is the one method whose split-adjusted standard errors blow up. If your controls have any support gaps between folds, a linear nuisance model extrapolates and you pay for it. Tree-based methods, which predict a constant outside the training support, are much better behaved here. That is a small, concrete argument for boosting over lasso in the nuisance step, independent of predictive accuracy.

**Connection to the surrounding literature.** The ATE score is exactly the Robins–Rotnitzky doubly-robust influence function — DML is not inventing new estimators for treatment effects so much as explaining *why* the doubly-robust ones were the right ones, and adding the splitting that makes them work with ML. Targeted maximum likelihood (van der Laan) attacks the same problem from the likelihood side with the "super learner" doing the same job as the Ensemble column here. And the debiased-lasso literature (Zhang & Zhang, van de Geer, Javanmard & Montanari) turns out to be Neyman's 1959 construction applied to sparse linear regression.

**Follow-up questions.** How do you pick $K$? How does this interact with clustered or panel data, where the i.i.d. assumption behind the fold-independence argument fails? What happens when the target is a whole function (CATE) rather than a scalar — the theory here is explicitly for low-dimensional $\theta$, and the causal-forest line of work is the answer to that.

## Links

Related: [[Counterfactual Reasoning and Learning Systems]] · [[Doubly Robust Policy Evaluation and Learning]] · [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[Counterfactual Risk Minimization]] · [[Unbiased Learning-to-Rank with Biased Feedback]] · [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)]] · [[Regularization]] · [[Regression Analysis]] · [[Panel Regression]] · [[XGBoost- A Scalable Tree Boosting System]] · [[Why do tree-based models still outperform deep learning on tabular data]] · [[Reconciling Modern ML Practice and the Bias-Variance Trade-off]] · [[Derivative#Jacobian|Jacobian]] · [[Uncertainty]] · [[Principal Data Scientist — Experimentation & Causal Inference Interview Question Bank]] · [[experimentation_question_bank]] · [[Practical Bayesian Optimization of Machine Learning Algorithms]]

New topics worth writing: Neyman orthogonality, Cross-fitting, Semiparametric efficiency bound, Influence function, Donsker class and entropy conditions, Partially linear regression (Robinson), Doubly robust estimation / AIPW, Local Average Treatment Effect (LATE), Propensity score and unconfoundedness, Debiased lasso, Targeted Maximum Likelihood Estimation (TMLE), Super Learner, Gateaux derivative, Generalized Method of Moments (GMM), Approximate sparsity
