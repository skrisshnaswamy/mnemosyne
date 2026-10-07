---
title: "1% of Tokens Can Be Enough: On Gradient Estimation in On-Policy Distillation"
authors: ["Sheng et al."]
year: 2026
arxiv: "2609.24432"
url: https://arxiv.org/abs/2609.24432
priority: Good-To-Read
read_on: 2026-09-27
tags: [paper, rl, vision]
---
## The Core Idea

On-policy distillation (OPD) trains a small "student" model on its own generated text, while a bigger "teacher" model scores each next-token position. At every position you nudge the student's next-token distribution towards the teacher's. The loss is reverse KL: $D_{\mathrm{KL}}(p \Vert q) = \mathbb{E}_{a \sim p}[\log p_a / q_a]$, where $p$ is the student and $q$ the teacher.

Here is the problem. That expectation runs over the whole vocabulary (~150k tokens). In practice you estimate it from the **one token the student actually sampled**. So the gradient you apply at that position is a single-sample estimate of an expectation. Sometimes that single sample is a good stand-in for the average. Sometimes it is wildly off.

Recent work on "sparse OPD" picks a small subset of token positions to supervise — 1%, 0.1% — using heuristics for **usefulness**: high student entropy, big teacher–student disagreement, early position in the chain. Every one of those asks *"would fixing this token help?"*

Nobody asked *"can we even estimate the fix reliably from one sample?"*

That is the gap. A token can be very useful to fix and yet give you a gradient estimate whose noise swamps the signal — so you apply an update pointing in roughly a random direction. Conversely a token with a beautifully reliable gradient estimate might be pointing nowhere important.

> [!NOTE] Information-Efficiency Ratio (IER)
> At a fixed prefix, the ratio of squared gradient signal to the minimum achievable single-sample estimation noise, both measured in Fisher geometry. $\mathrm{IER}^{-1}$ is exactly the relative mean-squared error of the single-sample gradient estimate. High IER = this position's gradient direction can be trusted. ^ier-def

The headline empirical claim: combine IER with an existing usefulness score and you can match or beat **full** OPD (supervise every token) while supervising **0.1%–1% of tokens** in the rollout batch. At 0.1% on the medical task that is roughly *one token per trajectory*.

And a striking measurement drops out of the analysis: fewer than 0.1% of token positions have $\mathrm{IER} > 1$. For the overwhelming majority of tokens, the estimated noise exceeds the estimated signal. Full OPD is mostly applying noise.

## The Methodology

### Setting up the gradient at one prefix

Fix a student-generated prefix $s$. Let $z_\theta$ be the student's logits, $p = \mathrm{softmax}(z_\theta)$, and $q$ the teacher's distribution. Define the log-likelihood ratio $\rho(a) = \log \frac{p_a}{q_a}$ and the score $\phi_a = \nabla_{z_\theta} \log p_a = e_a - p$, where $e_a$ is the one-hot vector for token $a$.

Because $\mathbb{E}_p[\phi_a] = 0$, the exact gradient of the reverse KL with respect to the logits is

$$g = \mathbb{E}_p[\rho(a)\,\phi_a].$$

This is a [[Policy Gradient|policy-gradient]]-shaped object: a score function times a scalar weight. The single-sample estimator, with an action-independent baseline $b$ subtracted (a [[Policy Gradient#^baseline|baseline]], exactly as in REINFORCE), is

$$\widehat{g}_b(a) = (\rho(a) - b)\,\phi_a, \qquad a \sim p, \qquad \mathbb{E}_p[\widehat{g}_b] = g.$$

The baseline changes nothing in expectation but changes the variance a lot.

### Why measure noise in Fisher geometry

The obvious noise measure is Euclidean: $\mathbb{E}_p[\Vert \widehat{g}_b - g\Vert_2^2]$. This is what vOPD (a prior variance-reduction paper) uses. The objection: reverse KL is about a distance between *distributions*, so plain Euclidean distance on logits treats probabilities as ordinary coordinates and mismeasures what matters.

Instead, use the local geometry that KL itself induces. For small $\Delta z$,

$$D_{\mathrm{KL}}(p \Vert \mathrm{softmax}(z_\theta + \Delta z)) = \tfrac12 \Delta z^\top F \Delta z + o(\Vert\Delta z\Vert^2), \qquad F = \mathrm{diag}(p) - pp^\top,$$

where $F$ is the Fisher information matrix. Compare the *natural* gradients $F^+ g$ and $F^+\widehat{g}_b$ (with $F^+$ the pseudoinverse), and measure their difference's length under $F$. That works out to $\delta_b^\top F^+ \delta_b = \Vert\delta_b\Vert_{F^+}^2$.

### Theorem 1 — the decomposition

Define the **leverage factor** $L(a) = \frac{1 - p_a}{p_a}$. Then:

$$\text{Signal} = \Vert g\Vert_{F^+}^2 = \mathrm{Var}_p[\rho(a)]$$
$$\text{Noise}_b = \mathbb{E}_p\big[(\rho(a) - b)^2 L(a)\big] - \mathrm{Var}_p[\rho(a)]$$
$$b^\star = \frac{\mathbb{E}_p[\rho(a)L(a)]}{\mathbb{E}_p[L(a)]}$$

Both sides collapse into scalar statistics of the log-ratio $\rho$ — no matrices to invert at runtime. The signal is just the **variance of the log-ratio across the vocabulary**. The noise is a leverage-weighted second moment of the centred log-ratio, minus the signal. The optimal baseline is a leverage-weighted mean of $\rho$. The proof leans on two neat identities: $g = p \odot v_s$ where $v_s = \rho - \bar\rho$, and $\Vert\phi_a\Vert_{F^+}^2 = L(a)$ exactly.

$$\mathrm{IER} = \frac{\text{Signal}}{\text{Noise}_{b^\star}}$$

The leverage factor is the intuition-carrier: for a rare token ($p_a$ small), $L(a)$ is huge, so if you happen to sample it, its contribution explodes. That is the whole source of the noise.

Corollary: average $K$ i.i.d. samples at the same prefix and relative MSE becomes $\frac{1}{K \cdot \mathrm{IER}}$. So to hit relative error $\epsilon$ you need $K \geq \frac{1}{\epsilon\,\mathrm{IER}(s)}$ samples. At $K=1$, $\mathrm{IER}^{-1}$ *is* the relative MSE.

### Making it computable

The exact formulas need full-vocabulary sums at every prefix. Too expensive. The approximation:

1. Build a candidate set $\mathcal{C} = \mathrm{TopK}(z_\theta) \cup \mathrm{TopK}(z_T) \cup \{a\}$ with $K = 16$, so $16 \le |\mathcal{C}| \le 33$.
2. Missing logits (a token in one model's top-K but not the other's) get filled with $-12$.
3. Renormalise both distributions by softmax over $\mathcal{C}$ only.
4. Compute $\widehat\rho(a) = \mathrm{clip}(\log \widehat p(a)/\widehat q(a), -30, 30)$.
5. Plug into the Theorem-1 formulas, with $\epsilon = 10^{-8}$ guarding denominators.

### Combining reliability with usefulness

Rank all tokens in a rollout batch by $\widehat{\mathrm{IER}}$; normalise to $r_j \in [0,1]$. Let $u_j \in [0,1]$ be a normalised usefulness score. Then

$$s_j^{\mathrm{OR}} = 1 - (1-u_j)(1-r_j), \qquad s_j^{\mathrm{AND}} = u_j r_j.$$

**IER-OR** keeps tokens that are *either* useful *or* reliable. **IER-AND** demands both. Note the training objective is untouched — this is purely which positions contribute gradient. The sampled reverse-KL loss stays as-is.

### Training setup

- Rollout batch: 4 prompts × 16 responses. Temperature 1, top-$p$ 1.
- Max prompt / response / context: 2,048 / 8,192 / 16,384.
- 8 minibatches of size 8. [[Adam- A Method for Stochastic Optimization|Adam]], lr $10^{-6}$, weight decay 0.1, $(\beta_1,\beta_2) = (0.9, 0.98)$.
- Scores normalised **within each rollout batch**; one global budget across the batch; at least one token retained per response. So individual responses contribute different fractions.

**Model pairs.** Math: JustRL-Nemotron-1.5B → OpenMath-Nemotron-1.5B (strong-to-weak, same size) and JustRL-Qwen3-4B → Qwen3-1.7B (big-to-small). Prompts from DAPO-Math-17k, 50 rollout rounds. Eval on AIME 2025/2026 and HMMT-Feb 2025/2026, 32 samples/problem, reported as Bayes@32. Medical: ClinAlign-4B → Qwen3-4B on RaR-Medicine, 100 rounds, HealthBench graded by gpt-oss-120B (macro F1 0.6614 on the meta-eval — GPT-4.1 would score 0.709 but blew the budget).

**Baselines.** Prefix (early tokens), Entropy (student next-token entropy), TIP (soft-OR of entropy and teacher–student divergence), TA-OPD (disagreement × teacher mass on student's top-k), CA-SoftOR (a TA-OPD variant), plus Random, Sampled-RKL-max and Sampled-RKL-min from a concurrent paper.

## Ablation Studies and Experiments

### IER alone, at absurd sparsity

Medical, HealthBench overall/hard. Student 38.23/8.78, full OPD 45.77/19.77, teacher 46.37/20.24.

| Budget | IER alone | Prefix alone |
|---|---|---|
| 10% | 45.85 / 19.68 | 42.17 / 14.11 |
| 1% | 44.95 / 18.39 | 39.91 / 10.97 |
| 0.1% | 45.25 / 18.37 | 38.30 / 8.68 |

At 0.1% — about one token per trajectory — IER lands at 45.25 against full OPD's 45.77. Prefix at the same budget delivers 38.30, i.e. **nothing above the untrained student**. IER as a standalone selector is not a gimmick.

Math, Nemotron pair, 0.1% budget: IER hits 58.9 on AIME26 (full OPD 59.9) and 34.1 on HMMT26 (full OPD 34.7). On the Qwen3 pair it *exceeds* full OPD on three of four benchmarks.

### IER + usefulness

Medical at 0.1%: TIP+IER-OR reaches **46.08**/19.61 and TA-OPD+IER-OR 45.69/19.68, against full OPD's 45.77/19.77. Prefix is the dramatic case: 38.30/8.68 → **44.98/19.49** with IER-OR. Adding reliability rescues a selector that was otherwise useless at this budget.

Math at 0.1%, Qwen3 pair: TIP+IER-AND gives 19.4 / 16.3 / 11.7 / 14.4 on AIME25/26, HMMT25/26 versus full OPD's 14.4 / 12.5 / 8.7 / 13.4 — a consistent and large win. TA-OPD+IER-AND at 1% on the Nemotron pair matches full OPD.

### Why the combination isn't redundant

The Jaccard/Spearman analysis is the most informative diagnostic. Across selectors (Prefix excepted), **Spearman rank correlation over all tokens is high** — they broadly agree on ordering. But **Jaccard similarity of the top-10% selected sets is low**, and drops further as the budget shrinks. So the selectors agree in the bulk and disagree exactly at the extreme tail, which is the only part a 0.1% budget ever sees. IER is measuring a genuinely different axis.

### More tokens is not better

Sweeping budgets over [0.1, 1, 5, 10, 20, 50, 80]%: performance is **not monotone** in budget. Gains concentrate at 0.1–1%, and 1–5% with both scores matches or beats full OPD in several settings. Higher budgets sometimes *degrade* results. Consistent with the IER-distribution finding — beyond the tiny reliable tail you are adding noise-dominated updates.

### Thinking-on vs thinking-off

Qwen3 pair, TIP vs TIP+IER-AND. Thinking-off: sparse methods generally match or beat full OPD. **Thinking-on: TIP alone can fall below full OPD and even below the untrained student.** Adding IER-AND recovers it in most cases. Longer reasoning chains mean more positions, and a usefulness-only score picks badly among them.

### What did not work

- **Entropy is the bad partner.** Neither IER-OR nor IER-AND reliably improves it; some Entropy combinations *reduce* math performance. Entropy+IER-OR at 1% on the Nemotron pair: 49.7 on AIME25 versus 53.6 for Entropy alone.
- **No selector wins universally.** Sampled-RKL-max/min beat almost everything on the Nemotron pair at 0.1% and 1%, then collapse on the Qwen3 pair to roughly random-selection level.
- **Prefix is weak** as a standalone method here, which cuts against the prefix-based line of OPD work.
- **No compute saving.** IER at 10% costs +2.2–2.5% mean step time and up to +2.17 GiB peak memory versus full OPD. Sparse supervision here means fewer gradient contributions, not less work — full trajectories are still generated and scored.

## Worth Remembering

**The honest limitation the authors state twice.** High IER says nothing about usefulness, magnitude, or task alignment. Rescale the gradient and transform the baseline accordingly and both signal and noise scale equally — IER is invariant. It is a *relative accuracy* measure and nothing more. That is precisely why it must be combined with a usefulness score rather than replacing one.

**The pattern in where IER helps.** Gains are largest when (a) the budget is extremely small and (b) the base selector is weak. At 10% budget with a strong selector like TIP, IER adds little. Read this as: IER is a tail-selection tool.

**IER declines with token position** (Figure 5, mean/median/max all drop). This independently corroborates the prefix-based line of work — early tokens really do have more trustworthy gradients — but via a mechanism, not a heuristic. The authors suggest a future efficiency win: truncate a rollout once enough high-IER tokens have accumulated, which *would* buy real compute savings.

**Connection to [[Policy Gradient]] and [[GRPO]].** This is the classical variance-reduction story ([[Simple Statistical Gradient-Following Algorithms (REINFORCE)|REINFORCE]], baselines, [[High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)|GAE]]) applied to distillation, with one twist: instead of using the optimal baseline to *reduce* variance in training, they use the variance *at* the optimal baseline as a *selection score*. Nice reuse of an old result.

**Practical caveats if you wanted to implement this.** The $K=16$ candidate set and the $-12$ fill value for missing logits are load-bearing and unablated. Overlap between teacher and student top-16 sets can be small, so a lot of the log-ratio mass is coming from a made-up constant. Clipping $\rho$ to $[-30,30]$ is also unablated. Whether $\widehat{\mathrm{IER}}$ tracks true IER is never checked against a full-vocabulary computation, even on a small model.

**Evaluation caveats.** Standard errors are 0.4–1.0 pp on math and ~0.5/1.0 on HealthBench. Many of the reported "wins" are inside one standard error. The convincing results are the large ones: Prefix+IER-OR's +6.7 pp on HealthBench overall, and TIP+IER-AND's +5 pp on AIME25 for the Qwen3 pair. Treat single-cell comparisons in the big table with caution — with 5 selectors × 2 rules × 2 pairs × 3 budgets × 4 benchmarks there is plenty of room for noise to look like signal.

**Open questions.** Adaptive $K$ per prefix (sample more where IER is low) is flagged as future work and follows directly from the corollary. Whether IER could *weight* rather than *select* tokens — a continuous version of the objective — is untried. And the whole thing is a per-prefix, per-token local analysis; nothing here speaks to whether a sequence of reliable local updates composes into a good trajectory-level update.

## Links

Related: [[Distillation]] · [[Policy Gradient]] · [[KL Divergence]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)|REINFORCE]] · [[High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)|GAE]] · [[Credit Assignment]] · [[Monte Carlo Methods]] · [[On-policy Distillation with Verifiable Reward]] · Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation · [[When EOS Tokens Disagree- Understanding Length Inflation in On-Policy Distillation]] · [[What Does Privileged Information Add to On-Policy Self-Distillation]] · [[PACT- From Credit Assignment to Critic Alignment]] · [[Dynamic Important Example Mining for Reinforcement Finetuning]] · [[Distilling the Knowledge in a Neural Network]] · [[GRPO]] · [[Off-Policy Evaluation]] · [[Uncertainty]]

New topics worth writing: Fisher information matrix, natural gradient descent, information geometry, control variates and variance reduction in gradient estimators, signal-to-noise ratio analysis of stochastic gradients, leverage in importance-weighted estimators, Bayes@k evaluation, HealthBench
