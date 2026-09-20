---
title: "Online Learning with LLM Experts from Limited Feedback"
authors: ["Wei et al."]
year: 2026
arxiv: "2609.05820"
url: https://arxiv.org/abs/2609.05820
priority: Good-To-Read
read_on: 2026-09-19
tags: [paper, transformers, llm, rl]
---
## The Core Idea

You have $K$ different LLMs. Each prompt arrives one at a time, and you must pick one model to answer it. Some models are better at code, some at maths, none is best at everything — on the Nectar dataset the win rates of six popular models are $0.182, 0.091, 0.203, 0.319, 0.073, 0.132$, so the best single model wins only a third of the time. Routing is worth doing.

The problem: you almost never find out whether the answer was good. Users do not rate responses. To get a quality score you need a human or a strong LLM judge, and that costs money. So you have a **budget** $m \ll T$: over $T$ rounds you may only buy $m$ quality labels.

The new thing is not "route prompts with a [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)|contextual bandit]]" — that is standard. The new thing is that the agent chooses **what to label and when**. Feedback is not tied to the current round. If at round $t$ you decide to spend one unit of budget, you may go back and buy the label for *any* past prompt you have not labelled yet. That freedom is the whole paper.

Why does that matter? Because the prompt arriving right now may be boring — nearly a duplicate of ten prompts you already know about. Labelling it teaches you nothing. Somewhere in your history there is a weird prompt that points in a direction of feature space you have never measured. Labelling *that* one shrinks your uncertainty far more. The trick is to pick the past prompt that maximises the determinant of the covariance matrix.

> [!NOTE] Look-back feedback selection
> When you spend a scarce label, do not label the current example. Label the past example whose feature vector $x_\ell$ most increases $\det(V + x_\ell x_\ell^\top)$, where $V$ is the design matrix of already-labelled examples. This is greedy D-optimal design run online. ^look-back-feedback

This did not exist before because standard bandit analysis leans hard on the covariance matrix being updated *every* round — that is what makes the sum of confidence widths telescope. Break that and the usual proof collapses. The contribution is an algorithm plus a proof technique that survives the break.

What it unlocks: regret $\tilde{O}(dT/\sqrt{m})$ with full-information feedback, $\tilde{O}(dT\sqrt{K/m})$ with bandit feedback. Set $m = T$ and you recover the classic $\tilde{O}(d\sqrt{T})$ linear-bandit rate. The price of limited feedback is a clean multiplicative $\sqrt{T/m}$.

## The Methodology

**The model.** Each expert $a \in [K]$ has an unknown weight vector $\theta_a \in \mathbb{R}^d$ with $\|\theta_a\|_2 \le 1$. A prompt at round $t$ is an [[Embeddings|embedding]] $x_t$ with $\|x_t\|_2 \le L$. The reward is linear plus noise:

$$r_{t,a} = \langle \theta_a, x_t \rangle + \eta_{a,t}$$

with $\eta$ independent and sub-Gaussian, variance proxy 1. The authors justify linearity by noting this is exactly a linear head on a frozen transformer embedding — standard reward modelling, minus fine-tuning the encoder.

Regret is measured against picking the best expert *per prompt*:

$$\mathsf{Reg}(T) = \mathbb{E}\Big[\sum_t \langle \theta_{a_t^\star}, x_t\rangle - \sum_t \langle \theta_{a_t}, x_t\rangle\Big], \quad a_t^\star = \arg\max_a \langle \theta_a, x_t\rangle$$

**Ordering within a round.** Prompt arrives → pick expert → *then* optionally buy feedback. You cannot peek at similar past prompts before deciding, because that would need a query every round and blow the budget.

---

### Algorithm 1 — `LimFullFeed` (full information)

Full information means one purchased label reveals the reward of **all $K$ experts** on that prompt. So all experts share one covariance matrix $V_t$.

Set $b = m/T$. Initialise $V_0 = \lambda I_d$, $\hat\theta_{0,a} = 0$.

Each round:

1. **Act greedily.** $a_t = \arg\max_a \langle \hat\theta_{t-1,a}, x_t\rangle$. No exploration bonus. Why? Because all experts share the same $V_t$, so the confidence width $\beta\|x_t\|_{V_{t-1}^{-1}}$ is identical across experts and cancels in the argmax. Exploration happens entirely through *which prompt you label*, not which arm you pull.
2. **Should I buy a label?** Yes if $\lfloor b(t-1)\rfloor < \lfloor bt \rfloor$ — i.e. roughly every $T/m$ rounds. This makes the budget constraint hold by construction, no bookkeeping needed.
3. **Which one?** $$s_t = \arg\max_{\ell \in [t]\setminus \mathcal{S}_{t-1}} \det(V_{t-1} + x_\ell x_\ell^\top)$$ over all past unlabelled rounds. Never re-label a round (a repeat observation is not independent).
4. **Update.** $V_t \gets V_{t-1} + x_{s_t}x_{s_t}^\top$, and for every expert $y_{t,a} \gets y_{t-1,a} + r_{s_t,a}x_{s_t}$, $\hat\theta_{t,a} \gets V_t^{-1}y_{t,a}$ — plain [[Regression Analysis|ordinary least squares]] (ridge, really, from the $\lambda I_d$).

Cost: naively $O(d^3)$ per round for the inverse and determinants. Drops to $O(d^2)$ with Sherman–Morrison for the inverse and the matrix determinant lemma for $\det$.

**Why the proof works.** Standard analysis writes the per-round error as $\log\det(V_t + x_tx_t^\top) - \log\det V_t$ and telescopes, because $V_{t+1} = V_t + x_tx_t^\top$. Here $V$ freezes for $T/m$ rounds so there is nothing to telescope. The fix: because $s_t$ was chosen to *maximise* the determinant increase, every prompt $x_r$ seen in the frozen window satisfies $\det(V_{\tau} + x_rx_r^\top) \le \det(V_\tau + x_{s_t}x_{s_t}^\top)$. So you can upper-bound all $T/m$ frozen rounds by the single real update, recovering a telescoping sum at a cost of a factor $T/m$ inside the square root — hence $\sqrt{T/m}$ in the regret.

**Theorem 3.2.** $\mathsf{Reg}(T) = O(dTm^{-1/2}\log(TL))$. No $K$ dependence, because one label covers all experts.

**Theorem 3.3 (lower bound).** $\mathsf{Reg}(T) \ge \frac{T\sqrt{d}e^{-4}}{8\sqrt{m}}$, via Bretagnolle–Huber on prompts at Hamming distance 1. Gap of $\sqrt{d}$ to the upper bound, which comes from needing a self-normalised martingale union bound over all of $\mathbb{R}^d$ because future prompts are unknown. If the prompts were known in advance you would only union-bound over $T$ vectors and the gap closes.

---

### Algorithm 2 — `LimBanFeed` (bandit feedback)

Now one label reveals **only** the expert that actually answered. Each expert gets its own $V_{t,a}$, and experts you use rarely stay poorly estimated. So you now need real exploration on the arm choice.

Set $z = T/m$. Per expert: $V_{0,a} = \lambda I_d$, counter $n_a = 0$.

1. **Act optimistically** — [[Finite-time Analysis of the Multiarmed Bandit Problem (UCB1)|UCB]]-style: $$a_t = \arg\max_a \langle \hat\theta_{t-1,a}, x_t\rangle + \beta\|x_t\|_{V_{t-1,a}^{-1}}$$ with $\beta = \sqrt{\lambda} + \sqrt{6\log T + d\log(1 + TL^2/d)}$. This is exactly LinUCB, just with a stale covariance matrix.
2. Increment $n_{a_t}$.
3. **Buy a label for expert $a_t$ once $n_{a_t} \ge z$**, then reset $n_{a_t} = 0$. Each expert is labelled once per $z$ uses, so labels are allocated in proportion to how often each expert is actually used. Budget check: $\sum_a \lfloor |S_{T,a}|/z\rfloor \le T/z \le m$ whenever $z \ge T/m$. Again satisfied by construction.
4. **Which past prompt?** Same D-optimal rule, but restricted to prompts that *this expert* answered: $$s_t = \arg\max_{s \in \{s \le t : a_s = a_t\}\setminus S_{t-1,a_t}} \det(V_{t-1,a_t} + x_sx_s^\top)$$

**Theorem 4.1.** $\mathsf{Reg}(T) = O(dTK^{1/2}m^{-1/2}\log(TL))$. The extra $\sqrt{K}$ is exactly because a full-information label is worth $K$ bandit labels.

**Theorem 4.2 (lower bound).** $\mathsf{Reg}(T) \ge \frac{TKe^{-4}}{8\sqrt{m}}$. Gap of $\sqrt{d/K}$ remains; the authors say it is open which side is loose.

---

### Variable evaluation costs (Appendix A)

If judging expert $j$ costs $z_j$ (GPT-4 judging a long answer costs more than judging a short one), evaluate expert $j$ after it has been used $z_j \cdot \tilde z$ times, with $\tilde z \ge T/m$. Expensive experts get labelled less often. Regret becomes $O\big(dTm^{-1/2}(\sum_j z_j)^{1/2}\log T\big)$, collapsing to Theorem 4.1 when all $z_j = 1$.

## Ablation Studies and Experiments

**Data.** Two routing benchmarks.

- **RouterBench** — 11 LLMs, ~405k prompts, responses scored and normalised to $[0,1]$. They sample 10k prompts, embed with `bge-small-en-v1.5` (384-d), then [[Fundamentals|PCA]] down to $d=40$, which keeps 50% of feature variance.
- **Nectar** — $K=6$, $d=40$, $T=60{,}000$. Ground-truth $\theta_a$ is fabricated by taking GPT-4's best-model ranking, one-hot encoding it into a $10000\times11$ response matrix, and fitting OLS. So the "true" reward model is itself a linear fit — the linear assumption is true by construction here, which is a real caveat.

**Baselines.**
- `NoLookBack` — same budget, same schedule, but labels the *current* prompt instead of searching history. This isolates the one idea in the paper.
- `AllFeedback` — labels every round, $m = T$. Performance ceiling.

**Headline table (Nectar, cumulative regret, mean ± sd over 5 runs):**

| $m$ | LimFullFeed | NoLookBack (full) | LimBanFeed | NoLookBack (bandit) |
|---|---|---|---|---|
| 1000 | 672.9 ± 37.4 | **631.6 ± 22.1** | **3583.4 ± 68.0** | 3789.6 ± 30.4 |
| 2000 | 441.4 ± 42.6 | **408.0 ± 26.2** | **3084.1 ± 25.6** | 3580.0 ± 73.3 |
| 5000 | **214.1 ± 7.7** | 218.7 ± 10.6 | **2295.7 ± 42.3** | 3020.7 ± 17.2 |
| 10000 | **128.0 ± 8.7** | 129.0 ± 11.0 | **1839.1 ± 29.1** | 2538.4 ± 22.2 |
| 20000 | **73.6 ± 6.0** | 75.8 ± 6.1 | **1567.9 ± 13.4** | 2090.7 ± 21.2 |

**What did not work, and it is the interesting part.** In the full-information setting at small budget, **look-back loses**. At $m=1000$: 672.9 vs 631.6. At $m=2000$: 441.4 vs 408.0. Both gaps are outside one standard deviation. At $m \ge 5000$ the methods are statistically indistinguishable (214.1 vs 218.7; 128.0 vs 129.0; 73.6 vs 75.8 — all overlapping error bars).

So in the full-information setting, the paper's central mechanism buys you **nothing measurable**, and at tight budgets it actively hurts. The authors' text says "`LimFullFeed` consistently outperforms `NoLookBack`", which their own Table 1 does not support. Their explanation for the small-$m$ case — "each feedback reveals all experts" — is a post-hoc rationalisation, not a mechanism. A plausible real reason: greedy determinant maximisation chases outlier embeddings, which are informative about directions nobody ever queries again. With few labels you would rather spend them on the bulk of the prompt distribution.

**Where it does work: bandit feedback.** Here the gap is real and large — 2295.7 vs 3020.7 at $m=5000$ (24% lower regret), 1567.9 vs 2090.7 at $m=20000$ (25%). The gap widens with $m$. The reason is structural: in the bandit setting each expert has its own sparse, non-uniformly collected dataset, so the covariance matrices are badly conditioned in different directions, and choosing *which* past prompt to label per expert genuinely matters.

**Theory check.** Regret does fall roughly as $1/\sqrt{m}$: full-information goes $672.9 \to 73.6$ as $m$ goes $1000 \to 20000$ (a $20\times$ budget increase, $\sqrt{20} \approx 4.5$, observed ratio $9.1$ — better than predicted). Bandit goes $3583 \to 1568$, ratio $2.3$, worse than $\sqrt{20}$ — consistent with the bandit setting being harder to saturate.

**Full-info regret is uniformly below bandit regret** at every budget, roughly $5$–$20\times$ lower. Matches the $\sqrt{K}$ term with $K=6$ plus the extra exploration cost.

**`AllFeedback`** has the lowest regret and visibly lower run-to-run variance, as you would expect from more data.

**Runtime (Table 2, Nectar, seconds).** Both scale linearly in $T$: `LimBanFeed` 0.41s at $T=500$ → 14.35s at $T=15000$. `LimBanFeed` scales linearly in $K$ (2.67s at $K=2$ → 9.97s at $K=20$) because it maintains $K$ covariance matrices; `LimFullFeed` is flat in $K$ (~1.5s throughout) thanks to the shared matrix. Both are nearly flat in $d$ over $8 \to 128$, which suggests the argmax-over-history scan, not the linear algebra, dominates.

**The adversarial argument against `NoLookBack`** is theoretical only: an adversary who serves prompts at the scheduled feedback rounds that are orthogonal to all other prompts makes `NoLookBack` suffer linear regret. No experiment constructs this.

## Worth Remembering

**The honest summary.** Look-back D-optimal feedback selection is worth it under bandit feedback (~25% regret reduction) and is not worth it under full information. If you only ever pay for a full judge sweep across all candidate models, just label whatever arrives. If you pay per-model — which is the realistic deployment, since you only *have* the response from the model you routed to — then the look-back matters.

**Limitations the authors state.** You must store every past prompt and every past response, possibly forever, so you can retroactively buy a judgment. That is a real storage and privacy cost that the regret bound does not price. And the experiments are static benchmarks replayed as a stream — no [[Monolith- Real Time Recommendation System With Collisionless Embedding Table|concept drift]], no non-stationary user mix, no model deprecation.

**The linearity assumption is doing heavy lifting, and the evaluation hides it.** The ground-truth $\theta_a$ in the Nectar experiment *is* an OLS fit on one-hot GPT-4 preferences. The generative model matches the estimator exactly. Real routing quality is not a linear function of a 40-dimensional PCA of a 384-d sentence embedding. Expect worse in production, and expect the misspecification to bite hardest exactly where the D-optimal rule sends you — the outlier directions.

**PCA to 40 dims keeps only 50% of variance.** That is a lot thrown away, and it is unclear whether $d=40$ was chosen for tractability or because it happened to work. No ablation on $d$ for regret (only runtime).

**Connection worth making: this is active learning inside a bandit.** The D-optimal criterion is exactly the classical experimental-design objective, and the paper's own discussion admits that a hindsight-optimal batch chooser would use [[Gaussian Process Optimization in the Bandit Setting (GP-UCB)|G-optimal design]] and get $O(T\sqrt{d/m})$ — which is the lower bound. The periodic-greedy scheme is shown to be near-optimal in $m$, which is the actually satisfying part of the theory.

**What differentiates this from neighbouring literature** (Appendix C is good): partial monitoring — reward is partially observed, agent has no choice. Delayed bandits — delay is exogenous. Lazy updates in LinUCB — model updates infrequently but uses *all* past data. Bandits with knapsacks — budget is spent on actions, not observations. Here the agent controls the delay *and* chooses the subset. The closest work, Tucker et al. (2023) on costly reward observations, maximises reward minus cost; this paper argues you cannot subtract a dollar-cost from a quality score, so you should constrain instead. That framing choice is worth stealing.

**Follow-up questions.** (1) Does a mixed criterion — say, label the current prompt with probability $p$ and the D-optimal past prompt otherwise — beat both at small $m$? The table strongly hints yes. (2) The look-back rule is purely about $x$; it ignores the *disagreement between experts* at that prompt. Labelling a prompt where all $K$ models are predicted equally good teaches you little about routing, however novel the embedding. A gap-weighted selection rule seems obviously better and is untested. (3) [[Doubly Robust Policy Evaluation and Learning|Doubly-robust]] reuse of the unlabelled rounds — you have $T-m$ prompts with a chosen action and no reward; a reward model could impute them.

**Practical caveat.** The determinant argmax scans all unlabelled history each time you buy a label. At $T=60{,}000$ that is fine; at production scale you need a reservoir or a coreset, and nobody has shown the regret bound survives that.

## Links

Related: [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]] · [[Finite-time Analysis of the Multiarmed Bandit Problem (UCB1)]] · [[Gaussian Process Optimization in the Bandit Setting (GP-UCB)]] · [[A Tutorial on Thompson Sampling]] · [[Doubly Robust Policy Evaluation and Learning]] · [[Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms]] · [[Open Bandit Dataset and Pipeline- Towards Realistic and Reproducible Off-Policy Evaluation]] · [[Counterfactual Risk Minimization]] · [[Embeddings]] · [[Evals]] · [[Uncertainty]] · [[Practical Bayesian Optimization of Machine Learning Algorithms]]

New topics worth writing: D-optimal and G-optimal experimental design, LLM routing and cascading, label-efficient online learning, partial monitoring, bandits with knapsacks, self-normalised martingale concentration, Bretagnolle–Huber inequality, active learning under budget constraints
