---
title: "Finite-time Analysis of the Multiarmed Bandit Problem (UCB1)"
authors: ["Auer", "Cesa-Bianchi & Fischer"]
year: 2002
url: https://link.springer.com/content/pdf/10.1023/A:1013689704352.pdf
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, rl, theory]
---
## The Core Idea

You have $K$ slot machines. Each one pays out from a fixed but unknown distribution. You want to collect the most reward over $n$ pulls. Every pull is a choice: pull the arm that has looked best so far (exploit), or pull one you barely know (explore). This is the cleanest possible version of the exploration–exploitation problem that sits under all of [[Markov Decision Process|RL]].

The score you are graded on is **regret**: how much reward you lost by not always pulling the best arm.

$$\text{Regret}(n) = \mu^* n - \sum_{j=1}^{K} \mu_j \, \mathbb{E}[T_j(n)]$$

where $\mu^* = \max_i \mu_i$ and $T_j(n)$ is the number of times arm $j$ was pulled in $n$ rounds.

Lai and Robbins (1985) had already proved two things. First, no algorithm can do better than $\Theta(\ln n)$ regret. Second, algorithms exist that hit that bound — but only **asymptotically** (as $n \to \infty$), and only for specific parametric families of reward distributions, and computing their index needed the whole reward history.

What is new here: a policy so simple you can write it in five lines, whose regret is bounded by $O(\ln n)$ **for every finite $n$**, not just in the limit, and for **any** reward distribution as long as it lives in $[0,1]$. No knowledge of the distributions is needed. That policy is **UCB1**.

The trick is *optimism in the face of uncertainty*. Do not score an arm by its average reward. Score it by the top of its confidence interval — the best its true mean could plausibly be. An arm gets picked either because it really is good, or because you have not pulled it enough to rule it out. Both are useful reasons to pull it. And the "not pulled enough" reason shrinks automatically as you gather data, so exploration turns itself off.

> [!NOTE] Upper Confidence Bound (UCB)
> Rank each arm by (empirical mean) + (a bonus that grows with total time and shrinks with how often *this* arm was pulled). Pick the top. No random coin flips anywhere. ^upper-confidence-bound

This is the paper that made bandits usable engineering rather than asymptotic statistics. It is the direct ancestor of the UCT selection rule inside [[Mastering the game of Go with deep neural networks (AlphaGo)|Monte Carlo Tree Search]], and it is the deterministic counterpart to [[A Tutorial on Thompson Sampling|Thompson Sampling]] and [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)|LinUCB]] (which is literally UCB with a linear reward model bolted on).

## The Methodology

### UCB1

Pull each arm once. Then, forever after, pull the arm $j$ maximising

$$\bar{x}_j + \sqrt{\frac{2 \ln n}{n_j}}$$

- $\bar{x}_j$ = average reward from arm $j$ so far.
- $n_j$ = times arm $j$ was pulled.
- $n$ = total pulls so far.

That is the entire algorithm. $O(1)$ memory per arm, $O(K)$ per step.

Where does $\sqrt{2\ln n / n_j}$ come from? Hoeffding's inequality. For $n_j$ samples in $[0,1]$ with mean $\mu_j$,

$$\mathbb{P}\left\{\bar{x}_j \geq \mu_j + \sqrt{\tfrac{2\ln t}{n_j}}\right\} \leq e^{-4 \ln t} = t^{-4}.$$

So the bonus is exactly the width of a one-sided confidence interval that fails with probability $t^{-4}$. The $\ln n$ in the numerator is what forces the bound to shrink slowly enough that you keep re-checking every arm forever, and $t^{-4}$ is chosen so that $\sum_t \sum_s \sum_{s_i} t^{-4}$ converges — that convergent sum is where the $\pi^2/3$ in the theorem comes from ($\sum t^{-2} = \pi^2/6$).

**Theorem 1.** For rewards in $[0,1]$, the regret of UCB1 after $n$ plays is at most

$$\left[8 \sum_{i:\mu_i < \mu^*} \frac{\ln n}{\Delta_i}\right] + \left(1 + \frac{\pi^2}{3}\right) \sum_{j=1}^{K} \Delta_j$$

with $\Delta_i = \mu^* - \mu_i$ the gap to the best arm.

The proof shape is worth internalising because every UCB proof since copies it. To pull a bad arm $i$ at time $t$, at least one of three bad things must happen:

1. The optimal arm's confidence bound is *under* its true mean: $\bar{X}^*_s \leq \mu^* - c_{t,s}$. Probability $\leq t^{-4}$.
2. Arm $i$'s bound is *over* its true mean: $\bar{X}_{i,s_i} \geq \mu_i + c_{t,s_i}$. Probability $\leq t^{-4}$.
3. The confidence interval is still genuinely too wide to separate them: $\mu^* < \mu_i + 2c_{t,s_i}$.

Case 3 becomes **impossible** once $s_i \geq 8\ln n / \Delta_i^2$ — the interval has shrunk below the gap. So arm $i$ gets pulled at most $\frac{8 \ln n}{\Delta_i^2}$ times, plus a constant from summing the tail probabilities of cases 1 and 2. Multiply by $\Delta_i$ and sum, per equation (5).

The constant $8/\Delta_i^2$ is worse than Lai–Robbins' $1/D(p_j \| p^*)$, where $D$ is the [[KL Divergence]] between the two arms' reward densities. Pinsker's inequality gives $D(p_j\|p^*) \geq 2\Delta_j^2$, so the best achievable constant is $1/(2\Delta_j^2)$ — UCB1 is a factor of 16 off it.

### UCB2 — closing the constant gap

Same optimism, but plays are grouped into **epochs**. When arm $i$ is chosen for its $r$-th epoch, it is played $\tau(r+1) - \tau(r)$ times in a row, where $\tau(r) = \lceil (1+\alpha)^r \rceil$. The index becomes

$$\bar{x}_i + a_{n,r_i}, \qquad a_{n,r} = \sqrt{\frac{(1+\alpha)\ln(en/\tau(r))}{2\tau(r)}}.$$

Batching plays means you recompute the index far less often, and the geometric epoch lengths let the analysis be tighter. Regret (Theorem 2, valid for $n \geq \max_i 1/(2\Delta_i^2)$):

$$\sum_{i:\mu_i<\mu^*}\left(\frac{(1+\alpha)(1+4\alpha)\ln(2e\Delta_i^2 n)}{2\Delta_i} + \frac{c_\alpha}{\Delta_i}\right)$$

As $\alpha \to 0$ the leading constant approaches the optimal $1/(2\Delta_i^2)$. But $c_\alpha \to \infty$ as $\alpha \to 0$ — you trade the asymptotic slope against the additive constant. The authors suggest letting $\alpha_n$ decay slowly with $n$.

### $\varepsilon_n$-GREEDY — making the obvious heuristic provable

Plain $\varepsilon$-greedy pulls a uniformly random arm with fixed probability $\varepsilon$, else the empirical best. Fixed $\varepsilon$ gives **linear** regret — you keep paying $\varepsilon \cdot \bar\Delta$ forever. The fix: decay it as $1/n$.

$$\varepsilon_n = \min\left\{1, \frac{cK}{d^2 n}\right\}$$

where $d$ is a lower bound on the smallest gap $\min_{i:\mu_i<\mu^*}\Delta_i$, and $c$ is a tuning constant. **Theorem 3** bounds the *instantaneous* probability of picking a suboptimal arm at step $n$ by

$$\frac{c}{d^2 n} + 2\left(\frac{c}{d^2}\ln\frac{(n-1)d^2 e^{1/2}}{cK}\right)\left(\frac{cK}{(n-1)d^2 e^{1/2}}\right)^{c/(5d^2)} + \frac{4e}{d^2}\left(\frac{cK}{(n-1)d^2 e^{1/2}}\right)^{c/2}$$

For $c > 5$ this is $c/(d^2 n) + o(1/n)$, which integrates to logarithmic regret. Note this is a *stronger* statement than Theorems 1–2 (per-step, not cumulative), but it needs the gap $d$ up front — a real practical handicap. The proof uses [[Cross Entropy|Hoeffding]] for the reward deviations and **Bernstein's inequality** for the count of random exploration pulls $T_j^R(n)$, giving $\mathbb{P}\{T_j^R(n) \leq x_0\} \leq e^{-x_0/5}$ with $x_0 = \frac{1}{2K}\sum_t \varepsilon_t$.

### UCB1-NORMAL

If you know rewards are Gaussian with unknown mean *and* unknown variance, use the sample variance in the bonus:

$$\bar{x}_i + \sqrt{16 \cdot \frac{Q_i - n_i \bar{x}_i^2}{n_i - 1} \cdot \frac{\ln n}{n_i}}, \qquad Q_i = \sum_t X_{i,t}^2$$

plus a forcing rule: pull any arm that has been played fewer than $\lceil 8 \ln n \rceil$ times. Regret (Theorem 4):

$$256(\log n)\sum_{i:\mu_i<\mu^*}\frac{\sigma_i^2}{\Delta_i} + \left(1 + \frac{\pi^2}{2} + 8\log n\right)\sum_j \Delta_j$$

The proof leans on Student-$t$ and $\chi^2$ tail bounds the authors **could only verify numerically** (stated as Conjecture 1 and 2). So Theorem 4 is not fully proved.

### UCB1-TUNED — the one people actually run

Not in the theorems, buried in Section 4. Estimate an upper bound on each arm's variance:

$$V_j(s) = \left(\frac{1}{s}\sum_{\tau=1}^{s}X_{j,\tau}^2\right) - \bar{x}_{j,s}^2 + \sqrt{\frac{2\ln t}{s}}$$

and replace the UCB1 bonus with

$$\sqrt{\frac{\ln n}{n_j}\min\{1/4,\ V_j(n_j)\}}$$

where $1/4$ caps it at the max variance of a Bernoulli. Low-variance arms get a much smaller exploration bonus, so you stop wasting pulls on arms you have already pinned down. **No regret bound is proven for this.** It is the best performer in the experiments anyway.

## Ablation Studies and Experiments

All Bernoulli arms, 100,000 pulls, averaged over 100 runs, plotted semi-log. Two metrics: fraction of pulls on the optimal arm, and cumulative regret. Seven reward configurations:

| # | Arms | Difficulty |
|---|---|---|
| 1 | 0.9, 0.6 | easy — big gap, low variance at optimum |
| 2 | 0.9, 0.8 | small gap |
| 3 | 0.55, 0.45 | hard — small gap *and* high variance |
| 11 | 0.9, then 9× 0.6 | easy, 10 arms |
| 12 | 0.9, 0.8×3, 0.7×3, 0.6×3 | many distinct suboptimal arms |
| 13 | 0.9, then 9× 0.8 | small gap, 10 arms |
| 14 | 0.55, then 9× 0.45 | hard, 10 arms |

**Results:**

- A **perfectly tuned $\varepsilon_n$-GREEDY usually wins**. That is the uncomfortable headline. The simple heuristic beats the clever index policy — *if* you already know the answer well enough to tune $c$.
- **$\varepsilon_n$-GREEDY loses on distributions 12 and 14.** Why: it explores *uniformly* over all arms. If there are many suboptimal arms with very different means (dist. 12), it wastes the same budget on an arm at 0.6 as on one at 0.8. UCB1 spends its exploration budget proportional to $1/\Delta_i^2$, i.e. it barely touches obviously-bad arms.
- **$\varepsilon_n$-GREEDY degrades fast when $c$ is off.** Too small $\to$ regret grows *linearly* (a curve in the semi-log plot). Too large $\to$ still logarithmic, but with a steep leading constant. Distribution 13 is the one exception where a wide range of $c$ works. There is **no single $c$ that works across all seven distributions** — the authors had to search per-distribution. This is the practical killer.
- **UCB1-TUNED matches a well-tuned $\varepsilon_n$-GREEDY in most cases, with zero tuning.** And it is **insensitive to reward variance**: it performs about the same on distribution 2 vs 3, and on 13 vs 14, where the mean gap is the same but the variance at the optimum differs a lot. That is the variance term in the bonus earning its keep.
- **UCB2 is consistently slightly worse than UCB1-TUNED.** Its parameter $\alpha$ barely matters as long as it is small; $\alpha = 0.001$ was fixed for everything after the initial sweep (Figure 5). So the better *theoretical* constant did not translate into a better *empirical* result. Worth sitting with.

**What did not work / what was left unproven:**

- The theoretically-optimal-constant policy (UCB2) lost to the theoretically-unanalysed one (UCB1-TUNED). Tighter bounds $\neq$ better behaviour at $n = 10^5$.
- No regret bound for UCB1-TUNED. The variance estimate makes the analysis break.
- Theorem 4 (UCB1-NORMAL) rests on two numerically-checked conjectures about Student and $\chi^2$ tails.
- Fixed $\varepsilon$ gives linear regret, confirming the obvious.

## Worth Remembering

**The bonus term is a design knob, not a law.** $\sqrt{2\ln n / n_j}$ falls straight out of Hoeffding with a $t^{-4}$ failure rate. Change the failure rate and the constant changes. Practitioners routinely shrink the 2 because UCB1 over-explores in practice — the bound is worst-case over all $[0,1]$ distributions, and your actual arms are usually not adversarial.

**Regret is gap-dependent and this cuts both ways.** $\frac{8\ln n}{\Delta_i}$ blows up as $\Delta_i \to 0$. But a tiny gap means you barely lose anything by picking wrong, so the *product* $\Delta_i \cdot T_i(n)$ stays bounded. The worst case over gaps gives $O(\sqrt{Kn\ln n})$ — set $\Delta \approx \sqrt{K\ln n / n}$.

**The independence assumptions are weaker than they look.** Theorems 1–3 do **not** need rewards to be independent across arms ($X_{i,s}$ and $X_{j,t}$ may be dependent), and do not need i.i.d. rewards within an arm. All they need is $\mathbb{E}[X_{i,t} \mid X_{i,1},\dots,X_{i,t-1}] = \mu_i$ — a martingale condition. Useful when arms share a hidden common factor.

**Limitations the authors admit:** the setting is stationary (arm means never drift), context-free (no features), and single-play. They explicitly say they do not know whether their techniques extend to restless/Markovian bandits where playing an arm advances its internal state — Gittins indices solve that but need prior knowledge of the reward processes.

**Practical caveats.** The $\sqrt{\ln n / n_j}$ bonus means an arm never gets permanently dropped; if your environment is non-stationary you need a discounted or sliding-window variant. UCB1 is fully deterministic, which is great for reproducibility and awful for A/B-test-style deployments where you want randomised assignment for later [[Doubly Robust Policy Evaluation and Learning|off-policy evaluation]] — [[A Tutorial on Thompson Sampling|Thompson Sampling]] is the better choice when you need propensities. Also note the paper requires knowing the support is $[0,1]$; you must scale your rewards.

**Follow-up questions.** Why did UCB2's better constant not show up empirically at $n = 10^5$ — is $c_\alpha$ dominating at that horizon? Does the variance-adaptive bonus in UCB1-TUNED have a modern proof (it does: UCB-V, Audibert et al. 2009)? And how does the $8/\Delta^2$ constant compare to KL-UCB, which achieves the actual Lai–Robbins bound?

## Links

Related: [[A Tutorial on Thompson Sampling]] · [[An Empirical Evaluation of Thompson Sampling (NeurIPS)]] · [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]] · [[KL Divergence]] · [[Markov Decision Process]] · [[Mastering the game of Go with deep neural networks (AlphaGo)]] · [[Decision Sciences]] · [[Uncertainty]] · [[Practical Bayesian Optimization of Machine Learning Algorithms]] · [[Doubly Robust Policy Evaluation and Learning]] · [[Random variable]]

New topics worth writing: Hoeffding's inequality, Bernstein's inequality, Lai–Robbins lower bound, KL-UCB, UCB-V, UCT and Monte Carlo Tree Search selection, Gittins index, non-stationary and discounted bandits, minimax vs gap-dependent regret
