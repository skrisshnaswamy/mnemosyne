---
title: "Doubly Robust Off-policy Value Evaluation for Reinforcement Learning"
authors: ["Jiang & Li"]
year: 2015
arxiv: "1511.03722"
url: https://arxiv.org/abs/1511.03722
priority: Good-To-Read
read_on: 2026-09-24
tags: [paper, llm, rl, theory]
---
## The Core Idea

You have logs from one policy (the **behaviour policy** $\pi_0$) and you want to know what a *different* policy $\pi_1$ would have earned. In a one-step problem this is a solved question — see [[Doubly Robust Policy Evaluation and Learning]]. In a **sequential** problem, where a policy makes $H$ decisions in a row before you see the outcome, the two available tools were both bad:

- **Importance sampling** ([[Off-Policy Evaluation]]): reweight each logged trajectory by the product of per-step probability ratios. Unbiased, but the product of $H$ ratios explodes. Variance grows exponentially in the horizon.
- **Regression / model fitting**: learn an MDP from the logs, then compute $\pi_1$'s value inside that model. Low variance, but the bias depends on how wrong your function class is, and **you cannot estimate that bias from data**. So the number is not trustworthy for safety decisions.

The trick here is a rewriting that most people miss. Step-wise IS can be written as a *recursion*:

$$V_{\text{step-IS}}^{H+1-t} = \rho_t\left(r_t + \gamma V_{\text{step-IS}}^{H-t}\right), \qquad \rho_t = \frac{\pi_1(a_t|s_t)}{\pi_0(a_t|s_t)}$$

Once you see that, the whole sequential problem becomes **$H$ stacked contextual bandit problems**. At step $t$: the state $s_t$ is the context, $a_t$ is the arm pulled, and the "reward" you observe is $r_t + \gamma V^{H-t}$ — whose expected value is exactly $Q(s_t,a_t)$. So you can apply the bandit doubly-robust estimator at every level of the recursion.

> [!NOTE] Doubly robust estimator for RL
> Add a learned value function as a *control variate*: predict what you expect, and only importance-weight the **surprise**. $$V_{\text{DR}}^{H+1-t} := \widehat{V}(s_t) + \rho_t\left(r_t + \gamma V_{\text{DR}}^{H-t} - \widehat{Q}(s_t,a_t)\right)$$ ^dr-recursion

Read it in plain words: *start from your model's guess for this state, then correct it by the importance-weighted amount your model got wrong.* If $\widehat{Q}$ is good, the correction term $r_t + \gamma V^{H-t} - \widehat{Q}$ is small, so multiplying it by a huge $\rho_t$ does far less damage. If $\widehat{Q}$ is garbage — even identically zero — you fall back exactly to step-wise IS. **Step-wise IS is the special case $\widehat{Q}\equiv 0$.** That is the whole pitch: you can only gain.

What it unlocks: an estimator that is *still unbiased* (so confidence intervals remain honest, and you can use it to gate policy deployment) but with variance close to the regression approach. The second contribution is a hardness result — in the worst case, DR with a perfect $\widehat{Q}$ **exactly matches the Cramér–Rao lower bound**, so no unbiased estimator can do better.

## The Methodology

**Setup.** Finite-horizon MDP, $H$ steps, discount $\gamma$. Goal is the scalar $v^{\pi_1,H} = \mathbb{E}_{\tau\sim(\mu,\pi_1)}\left[\sum_t \gamma^{t-1}r_t\right]$ — the expected return from the start distribution, not the whole value function. Logs are i.i.d. length-$H$ trajectories from a *known, stochastic* $\pi_0$.

**The estimator.** Base case $V_{\text{DR}}^0 := 0$, then unroll Eqn. above backwards from $t=H$ to $t=1$. Final estimate is $V_{\text{DR}}^H$, averaged over trajectories. Here $\widehat{V}(s_t) = \sum_a \pi_1(a|s_t)\widehat{Q}(s_t,a)$.

**Where $\widehat{Q}$ comes from.** Fit an MDP model from data, run [[Bellman Equation|Bellman backups]] to get $\widehat{Q}$. Critically, $\widehat{Q}$ **must be independent of the trajectories you plug into the estimator**, or the unbiasedness argument breaks. Same requirement as $\pi_1$. It is fine for $\widehat{Q}$ and $\pi_1$ to be fit on the *same* held-out data as each other — they just can't touch the evaluation set.

**$k$-fold DR.** Splitting your evaluation data in half wastes it. So: partition $D_{\text{eval}}$ into $k$ folds, fit $\widehat{Q}$ on $k-1$ folds, apply DR to the held-out fold, average. Each fold's estimate is unbiased, so the average is unbiased, and every trajectory contributes. This is the same [[Double-Debiased Machine Learning for Treatment and Structural Parameters#Cross-fitting|cross-fitting]] idea. They used $k=2$ only because refitting models is slow.

**Variance decomposition (Theorem 1).** This is the part worth memorising:

$$\mathbb{V}_t[V_{\text{DR}}^{H+1-t}] = \underbrace{\mathbb{V}_t[V(s_t)]}_{\text{state transition}} + \underbrace{\mathbb{E}_t\Big[\mathbb{V}_t[\rho_t\Delta(s_t,a_t)\mid s_t]\Big]}_{\text{action stochasticity}} + \underbrace{\mathbb{E}_t[\rho_t^2\,\mathbb{V}_{t+1}[r_t]]}_{\text{reward noise}} + \underbrace{\mathbb{E}_t[\gamma^2\rho_t^2\,\mathbb{V}_{t+1}[V_{\text{DR}}^{H-t}]]}_{\text{future}}$$

with $\Delta(s_t,a_t) := \widehat{Q}(s_t,a_t) - Q(s_t,a_t)$, the model error.

The only term $\widehat{Q}$ touches is the second one, through $\Delta$. So **DR only removes the variance caused by the behaviour policy's action randomness.** Even with $\widehat{Q}=Q$ exactly, variance from noisy transitions and noisy rewards survives, amplified by $\rho_t^2$.

**DR-v2, the extension.** To also kill transition-induced variance you would want to subtract $\gamma \widehat{V}(s_{t+1})\frac{\widehat{P}(s_{t+1}|s_t,a_t)}{P(s_{t+1}|s_t,a_t)}$ — but you don't know $P$. If you're confident your *transition* model is accurate (even if your reward model isn't), assume $\widehat{P}/P \equiv 1$ and the term collapses to just $\gamma\widehat{V}(s_{t+1})$. This buys more variance reduction at the price of a bias bounded by $\epsilon V_{\max}\sum_{t=1}^H \gamma^t$, where $\epsilon = \max_{s,a}\|\widehat{P}(\cdot|s,a) - P(\cdot|s,a)\|_1$. A quantified, controllable bias — unlike the regression estimator's.

**Confidence intervals.** Because DR is unbiased and applied to i.i.d. trajectories, standard concentration applies directly. Hoeffding gives $b\sqrt{\frac{1}{2n}\log\frac{2}{\delta}}$ where $n=|D|$ and $b$ is the range of the estimate. In practice they use normal approximations instead, since strict bounds are far too pessimistic.

## Ablation Studies and Experiments

Five estimators compared everywhere: step-wise IS, step-wise WIS (weighted IS — biased but consistent, considered the best practical IS-family point estimate), REG (regression), DR, and **DR-bsl** (DR with a deliberately stupid $\widehat{Q}$ — a step-dependent constant, no state or action dependence at all).

Target policies are $\pi_1 = (1-\alpha)\pi_{\text{train}} + \alpha\pi_0$ for $\alpha \in \{0, 0.25, 0.5, 0.75\}$. Larger $\alpha$ means $\pi_1$ is closer to $\pi_0$, which makes evaluation easier.

**Mountain Car.** 2D continuous state, 3 actions, deterministic, horizon compressed to 100 steps, $\gamma=0.99$, uniform-random $\pi_0$. Model via state aggregation ($\times 2^6$, $\times 2^8$, round, tabular). $|D_{\text{train}}|=2000$, $|D_{\text{eval}}|=5000$, 4000+ runs.

The x-axis of every figure is the *split* of $D_{\text{eval}}$ between fitting $\widehat{Q}$ and running the estimator. As you give IS/WIS more data they improve; REG gets worse. DR needs both, so it peaks at an intermediate split — and **at that peak it beats IS/WIS using the full dataset, in all four $\alpha$ settings.**

**The ablation that matters: DR-bsl beats IS and WIS most of the time.** A constant $\widehat{Q}$ that knows nothing about the state already reduces variance substantially. That tells you the win is mostly about *centring the thing you importance-weight*, not about having an accurate model. Cheap to get, hard to get wrong.

**Sailing.** Stochastic shortest-path on a $10\times 10$ grid, 4 integer state variables, 8 actions, $R_{\min}=-3-4\sqrt{2}$. Model via Kernel-based RL, bandwidth 0.25. $|D_{\text{train}}|=1000$, $|D_{\text{eval}}|=2500$, 4000 runs. Qualitatively the same, with two differences: WIS matches DR in the middle two panels (so the advantage isn't universal), and in the $\alpha=0.75$ panel DR with a 3:2 split beats *everything including REG* by a large margin, with 2-fold DR better still.

**KDD Cup 1998 donation data.** 5 integer state features, 12 actions, 22 steps, no discount, 3754 real trajectories. No ground truth, so they fit a simulator from the real data and treat that as truth. Deliberately rigged: the transition model is fit the same way as the simulator (near-perfect), while the reward model is fit by linear regression on only **3 of the 5 features** (deliberately wrong). This is exactly DR-v2's target regime — and DR-v2 is the best estimator in every setting: beats WIS when $\pi_1$ is far from $\pi_0$, beats REG when they're close.

**Safe policy improvement.** Pick the policy with the highest lower confidence bound $V_\dagger - C\sigma_\dagger$; hold onto $\pi_0$ if nothing clears it. Candidates generated by sweeping train/test splits $\in\{0.2,0.4,0.6,0.8\}$ and mixing rates $\alpha \in \{0,0.1,\dots,0.9\}$.

- With good candidates: DR's improvement over $\pi_0$ hugely outperforms IS, **because IS's variance is so large it refuses to accept any policy far from $\pi_0$.** DR is willing to accept genuine improvements.
- The obvious worry: is DR just being reckless? They ran the mirror experiment with $\pi_{\text{train}}$ trained to *minimise* value, producing candidates worse than $\pi_0$. At matched $C$, **DR is as safe as IS or safer** (at $|D|=5000$). So the aggressiveness comes from lower variance, not from lost caution.
- Both methods got the best value at $C=0$, which is a reminder that these confidence bounds are conservative.

**The hardness result.** For **discrete tree MDPs** — where state *is* the full history, so no prior structure is assumed beyond discreteness — the Cramér–Rao lower bound on the variance of any unbiased estimator is

$$\sum_{t=1}^{H+1}\mathbb{E}\left[\rho_{1:(t-1)}^2\,\mathbb{V}_t[V(s_t)]\right]$$

Unfold Theorem 1 with $\Delta\equiv 0$ and this is *exactly* DR's variance. So the transition-stochasticity variance DR cannot remove is **intrinsic to the problem**, not a weakness of the estimator. If you want to beat it you must inject prior structure. In the relaxed DAG case (different histories can merge into the same state), $\rho_{1:t-1}$ is replaced by the state-action occupancy ratio $P_1(s_{t-1},a_{t-1})/P_0(s_{t-1},a_{t-1})$, which is $\leq$ the trajectory ratio — DAG MDPs are strictly easier.

## Worth Remembering

- **DR dominates step-wise IS by construction.** $\widehat{Q}\equiv 0$ recovers IS exactly. There is no scenario where a worse $\widehat{Q}$ makes you worse than having no $\widehat{Q}$ at all — the variance change is through $\mathbb{V}[\rho_t\Delta]$, and $\Delta = \widehat{Q}$ when $\widehat{Q}=0$. This is why DR-bsl already wins.
- **DR only fixes one of three variance sources.** Action stochasticity, yes. Transition stochasticity and reward noise, no. In a highly stochastic environment a perfect $\widehat{Q}$ will still leave you with enormous variance at long horizons. Do not expect miracles from a better model.
- **The independence requirement is a real operational constraint**, not a footnote. If you fit $\widehat{Q}$ on the same trajectories you evaluate on, the estimator is biased and your confidence interval lies. Use $k$-fold DR.
- **$\pi_0$ must be known and stochastic.** Every $\rho_t$ needs $\pi_0(a_t|s_t)$. If your production system didn't log propensities, none of this is available to you — see [[Contextual Bandit#^log-the-propensity]].
- **The variance is still exponential in $H$ in general.** The $\rho_t^2$ factors compound through the recursion. DR shrinks the constant; it does not change the horizon dependence. For long-horizon problems you still need something else.
- **WIS is a real competitor.** In Sailing, WIS matched DR in two of four settings. WIS is biased but consistent; DR is unbiased. If you need honest confidence intervals, that difference decides it. If you just want a point estimate, benchmark both.
- **DR-v2 is a conditional tool.** It is only sensible when you genuinely believe your transition model but distrust your reward model. The bias bound $\epsilon V_{\max}\sum\gamma^t$ scales with horizon, so it degrades fast if $\epsilon$ isn't tiny.
- The authors note that **Thomas & Brunskill (2016)** later reframed this DR estimator as a control-variate method, which is the cleaner way to see it, and built more advanced variants (MAGIC) on top. If you're implementing this today, start there.
- Open question they flag: applying this to real-world problems. In 2015 all the evidence was on simulated benchmarks plus one semi-synthetic dataset.

## Links

Related: [[Off-Policy Evaluation]] · [[Doubly Robust Policy Evaluation and Learning]] · [[Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms]] · [[The Self-Normalized Estimator for Counterfactual Learning (SNIPS) (NeurIPS)]] · [[Counterfactual Risk Minimization]] · [[Contextual Bandit]] · [[On-Policy vs Off-Policy]] · [[Offline RL]] · [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] · [[Markov Decision Process]] · [[Bellman Equation]] · [[Value Function]] · [[Eligibility Traces for Off-Policy Policy Evaluation (ICML)]] · [[Double-Debiased Machine Learning for Treatment and Structural Parameters]] · [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[Monte Carlo Methods]] · [[Policy Gradient]] · [[Open Bandit Dataset and Pipeline- Towards Realistic and Reproducible Off-Policy Evaluation]]

New topics worth writing: Control variates for variance reduction, Cramér–Rao lower bound and constrained CCRB, MAGIC estimator (Thomas & Brunskill 2016), High-confidence off-policy evaluation and safe policy improvement, Weighted importance sampling bias–consistency trade-off, Kernel-based reinforcement learning (Ormoneit & Sen)
