---
title: "Off-Policy Evaluation for Large Action Spaces via Embeddings"
authors: ["Saito & Joachims"]
year: 2022
arxiv: "2202.06317"
url: https://arxiv.org/abs/2202.06317
priority: Good-To-Read
read_on: 2026-09-25
tags: [paper, rl, theory]
---
## The Core Idea

When the action space is large, [[Off-Policy Evaluation|off-policy evaluation]] based on importance weighting falls apart. The weight $w(x,a) = \pi(a|x)/\pi_0(a|x)$ has a range that grows with the number of actions, so variance explodes; and a logging policy spread thinly over 5,000 items will assign probability zero to many of them, so [[Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms|IPS]] is also *biased*. In the synthetic experiments here, going from 10 actions to 5,000 at fixed sample size inflates the MSE of IPS by over 300×.

The trick: stop weighting by the **action**, and weight by an **action embedding** instead.

Movies have genres and directors. Products have categories and prices. If you have any such side information $e$ attached to each action, you can define a different importance weight — one over the marginal distribution of embeddings induced by each policy:

$$w(x,e) := \frac{p(e|x,\pi)}{p(e|x,\pi_0)}, \qquad p(e|x,\pi) = \sum_{a} \pi(a|x)\, p(e|x,a)$$

Many actions collapse onto the same embedding. So the embedding space is smaller than the action space, the weights are smaller, and the support condition is easier to satisfy. That is the whole idea.

> [!NOTE] Marginalized IPS (MIPS)
> $\hat V_{\mathrm{MIPS}}(\pi;\mathcal D) = \frac1n \sum_i w(x_i,e_i)\, r_i$. Same shape as IPS, but the weight is a ratio of *embedding* probabilities, not action probabilities. ^mips

Why did this not exist before? Because IPS and [[Doubly Robust Policy Evaluation and Learning|DR]] are already **minimax optimal** — you provably cannot beat them in the worst case. The only way out is to bring in extra structure that the worst case does not have. Action embeddings are that structure.

What it unlocks: reliable OPE in recommender-scale action spaces, where IPS and DR are numerically useless.

## The Methodology

Setup is the standard [[Contextual Bandit]]: context $x \sim p(x)$, action $a \sim \pi(a|x)$, reward $r \sim p(r|x,a)$, and the policy value is $V(\pi) = \mathbb{E}[r]$. The logged data now carries a fourth field:

$$\mathcal{D} = \{(x_i, a_i, e_i, r_i)\}_{i=1}^n, \quad (x,a,e,r) \sim p(x)\pi_0(a|x)p(e|x,a)p(r|x,a,e)$$

The embedding $e$ may be discrete or continuous, deterministic given $a$ (a product category) or stochastic and context-dependent (a price set by a personalisation algorithm). Nothing is assumed about how it was made.

**The two assumptions.** IPS needs common support over actions. MIPS swaps that for two different conditions.

1. **Common embedding support.** $p(e|x,\pi) > 0 \Rightarrow p(e|x,\pi_0) > 0$. Strictly weaker than action-level common support — many unsupported actions can still map onto supported embeddings.
2. **No direct effect.** $a \perp r \mid x, e$. Every causal path from action to reward runs *through* the embedding. In causal-graph terms, $e$ fully mediates the effect.

> [!NOTE] No Direct Effect
> Once you know the context and the embedding, the identity of the action tells you nothing more about the reward. If genre and director fully determine how much you will like a film, the film's ID is redundant. This is the assumption that makes MIPS unbiased, and it is the one you cannot check from data. ^no-direct-effect

Under these two, $V(\pi) = \mathbb{E}_{p(x)p(e|x,\pi)p(r|x,e)}[r]$ and MIPS is unbiased (Prop 3.4). Note the toy example in Table 1: $\pi_0(a_1|x_1) = 0$ so IPS is biased, and the largest action weight is $4.0$; but every embedding is supported and the largest embedding weight is $1.5$.

**The bias when the assumption fails** (Theorem 3.5):

$$\mathrm{Bias} = \mathbb{E}_{p(x)p(e|x,\pi_0)}\Big[\sum_{a<b} \underbrace{\pi_0(a|x,e)\pi_0(b|x,e)}_{\text{how unpredictable } a \text{ is given } e} \underbrace{(q(x,a,e)-q(x,b,e))}_{\text{direct effect}} \underbrace{(w(x,b)-w(x,a))}_{\text{policy mismatch}}\Big]$$

Three multiplicative factors, and any one of them being small kills the bias:
- If $e$ nearly *identifies* $a$, then $\pi_0(a|x,e)$ is near 0 or 1 and the product term vanishes.
- If actions sharing an embedding have similar rewards, the direct-effect term vanishes.
- If $\pi \approx \pi_0$, the weight-difference term vanishes.

**The variance is never worse** (Theorem 3.6):

$$n\big(\mathbb{V}[\hat V_{\mathrm{IPS}}] - \mathbb{V}[\hat V_{\mathrm{MIPS}}]\big) = \mathbb{E}_{p(x)p(e|x,\pi_0)}\big[\mathbb{E}[r^2|x,e]\cdot \mathbb{V}_{\pi_0(a|x,e)}[w(x,a)]\big] \geq 0$$

A variance of a variance — non-negative by construction, and non-negative *even when the no-direct-effect assumption is violated*. It grows when the reward is noisy and when many different actions hide behind one embedding.

**The tension, stated plainly.** Theorem 3.5 wants embeddings that are *informative* (fine-grained, predictive of the action) to kill bias. Theorem 3.6 wants embeddings that are *coarse* (many actions per embedding) to kill variance. Theorem 3.7 combines them into an MSE gain expression, where the bias term is multiplied by $(1-n)$ — so bias dominates as $n$ grows, and IPS eventually overtakes MIPS.

**Estimating the weights.** You usually do not know $p(e|x,\pi_0)$. Use the identity

$$w(x,e) = \mathbb{E}_{\pi_0(a|x,e)}[w(x,a)]$$

So: fit a classifier that predicts $a$ from $(x,e)$, giving $\hat\pi_0(a|x,e)$, then average the vanilla weights under it. Logistic regression in the synthetic experiments, Categorical Naive Bayes on the real data. This is why MIPS dodges the [[Curse of Dimensionality|curse of dimensionality]] that would sink a kernel-smoothing approach to continuous actions — it is a supervised classification problem, not a density estimate over $\mathcal{E}$.

**Embedding selection.** Since coarser is sometimes better, deliberately *throw away* embedding dimensions. Which ones? Apply SLOPE / SLOPE++ (Su et al. 2020b; Tucker & Lee 2021), an estimator-selection rule from Lepski's principle for bandwidth choice. It picks the largest-bias / smallest-deviation estimator whose estimate still agrees with all lower-bias candidates:

$$\hat m := \max\{m : |\hat V_m - \hat V_j| \leq \mathrm{CNF}(m) + (\sqrt 6 - 1)\mathrm{CNF}(j),\ \forall j<m\}$$

The appeal is that it never needs a bias estimate — estimating the bias is as hard as OPE itself. Deviations $\mathrm{CNF}$ are bounded with a Student's-$t$ confidence interval. With many dimensions the full subset search is intractable, so they use a greedy version.

## Ablation Studies and Experiments

**Synthetic setup.** 10-dim Gaussian contexts. Categorical embeddings sampled from a softmax over per-action parameters (Eq. 4). Reward $q(x,e) = \sum_k \eta_k (x^\top M x_{e_k} + \theta_x^\top x + \theta_e^\top x_{e_k})$ with $\eta$ from a Dirichlet, so each embedding dimension carries a known share of the signal. Logging policy is softmax over $q(x,a)$ with inverse temperature $\beta=-1$ (slightly *worse* than uniform random); target is $\epsilon$-greedy with $\epsilon = 0.05$ (near-deterministic, near-optimal). Reward noise $\sigma = 2.5$. Baselines: DM (random forest + 2-fold cross-fitting), IPS, DR, plus MIPS with true weights as an oracle.

| Sweep | Result |
|---|---|
| Actions $10 \to 5000$, $n=10{,}000$ | $\mathrm{MSE_{IPS}}/\mathrm{MSE_{MIPS}}$ goes $1.38 \to 12.38$ |
| Sample size $800 \to 25{,}600$, $\|\mathcal A\|=1000$ | ratio goes $9.10 \to 4.87$ — the edge shrinks as $n$ grows, exactly as Theorem 3.7 predicts |
| Reward noise $\sigma$ $0.5 \to 4.0$ | ratio goes $2.97 \to 14.98$; IPS MSE climbs $0.55 \to 3.22$ |
| Logging policy $\beta$ $3 \to -3$ | IPS and DR MSE blow up at $\beta=-2,-3$; MIPS stays flat |
| Deficient actions $0 \to 900$ of 1000 | MIPS (true weights) is essentially unaffected; estimated-weight MIPS degrades as the gap to the oracle opens |

**The ablation that carries the paper: deliberately breaking the assumption.** They rebuild the embedding as 20 binary dimensions ($|\mathcal E| = 2^{20}$) so the no-direct-effect assumption holds exactly only when all 20 are observed. Then they hide $\{0,2,4,\dots,18\}$ dimensions.

MSE is **minimised with dimensions missing** — 4 missing for estimated-weight MIPS, 8 missing for true-weight MIPS. Squared bias rises monotonically as expected; variance falls faster. Violating your own identification assumption is the correct move.

**Does SLOPE find that optimum automatically?** Yes, at small $n$. MIPS + SLOPE substantially beats plain MIPS in MSE for small samples, by trading bias for a large variance reduction. The gap closes as $n$ grows, which is what you want.

**Real data.** Open Bandit Dataset, "ALL" campaign, 100,000 sub-sampled rows. Uniform random logging policy, [[Thompson Sampling|Thompson sampling]] target, both collected during a live A/B test — so the ground-truth value is an on-policy Monte Carlo estimate. 80 items × 3 slots = 240 actions (same item in a different slot counts as a different action). 4-dimensional embeddings from hierarchical item categories. 150 bootstrap resamples, squared errors normalised by IPS.

**MIPS w/ SLOPE beats IPS in about 80% of runs. MIPS *without* SLOPE performs roughly the same as IPS.** So on real data with only 4 embedding dimensions and 240 actions, the embedding-selection step is not a refinement — it is the whole win.

### What did not work

- **Switch-DR, DRos, DR-$\lambda$, MRDR — the modern weight-shrinkage estimators — did not help.** With their hyperparameters tuned by SLOPE++, Switch-DR, DRos and DR-$\lambda$ all collapse towards DM's behaviour: they fail to improve with growing sample size and end up *worse than plain IPS and DR* in the large-sample regime. SLOPE++ evidently prefers the low-variance, high-bias end of the shrinkage dial when the weights are huge. MRDR's variance grew with the action count and it ended up indistinguishable from IPS/DR.
- **DM alone is not a substitute.** It is flat across sample sizes (bias does not shrink with data) and its bias worsens when logging and target policies diverge, because of extrapolation error. MIPS beat DM everywhere except the very smallest sample ($n=800$).
- **Estimated weights are visibly worse than true weights.** MIPS (true) beats MIPS in every large-action and high-deficiency setting. The authors flag closing that gap as open work — the weight estimator, not the estimand, is the current bottleneck.
- Interesting wrinkle: **IPS's MSE goes *down* as deficient actions increase**, because fewer supported actions means smaller weights and less variance — even though its bias is rising. A reminder that MSE curves hide which term is moving.

## Worth Remembering

**This is not the marginalized importance sampling from RL.** Liu et al. (2018), Xie et al. (2019) marginalize over the *state* distribution to break the curse of horizon. MIPS marginalizes over the *action embedding* to break the curse of a large action set. Same word, different disease.

**Relation to slate OPE.** The pseudo-inverse estimator of [[Off-policy evaluation for slate recommendation|Swaminathan et al. (2017)]] also fights combinatorial action spaces, but it buys tractability with a **linearity assumption on the reward**. MIPS buys it with an assumption about embedding quality instead. MIPS is the more general tool — it applies to slates, to rankings with slot-level rewards, and to plain contextual bandits, and it does not commit you to a click model. The cascade-model assumptions in ranking OPE only hold for vertical interfaces, which real products often aren't.

**Closest statistical cousin: surrogate outcomes in causal inference.** Athey et al. (2019)'s surrogacy condition — no direct effect of treatment on the long-term outcome except through the surrogate — is formally the same shape as Assumption 3.2. Different motivation (waiting decades for lifetime earnings vs. evaluating a recommender), same identification trick.

**Practical caveats if you want to run this:**

- You need embeddings logged *per interaction*, not just per action, if they are context-dependent. Most systems log the action ID and nothing else.
- Always run the selection step. The real-data result says plain MIPS is worth roughly nothing over IPS at 240 actions with 4 embedding dims; the selection is what buys the 80% win rate.
- The bias term scales with $(1-n)$. If you have a lot of logs and not that many actions, MIPS will eventually lose to IPS. Do not deploy it as a blanket replacement; it is a small-$n$/large-$|\mathcal A|$ tool.
- Estimating $\hat\pi_0(a|x,e)$ is a $|\mathcal{A}|$-class classification problem. At 5,000 actions that is itself nontrivial, and its error propagates directly into the estimate (Theorem B.2 gives the exact form: bias picks up an extra $-\mathbb{E}[\delta(x,e)q(x,\pi_0,e)]$ term where $\delta = 1 - \hat w/w$).
- If embeddings are deficient (some $e$ with $p(e|x,\pi_0)=0$), Theorem B.4 shows you get the same shape of support bias as IPS, just over $\mathcal{E}$ instead of $\mathcal{A}$. The assumption is weaker, not free.

**Open follow-ups the authors name:** learning or optimising action embeddings from the logged data rather than taking them as given; a better marginal-weight estimator; and applying marginal weighting inside DR-style estimators rather than only inside IPS. That last one is the obvious next paper.

**One line to carry away:** when the weights are too big, do not clip them or shrink them — change what you are weighting over.

## Links

Related: [[Off-Policy Evaluation]] · [[Contextual Bandit]] · [[Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms]] · [[Doubly Robust Policy Evaluation and Learning]] · [[Optimal and Adaptive Off-policy Evaluation in Contextual Bandits (SWITCH)]] · [[The Self-Normalized Estimator for Counterfactual Learning (SNIPS) (NeurIPS)]] · [[Off-policy evaluation for slate recommendation]] · [[Open Bandit Dataset and Pipeline- Towards Realistic and Reproducible Off-Policy Evaluation]] · [[Offline A-B testing for Recommender Systems]] · [[Counterfactual Risk Minimization]] · [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[Unbiased Learning-to-Rank with Biased Feedback]] · [[On-Policy vs Off-Policy]] · [[Embeddings]] · [[Curse of Dimensionality]] · [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]] · [[Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning (MAGIC)]]

New topics worth writing: Support deficiency in off-policy bandits (Sachdeva et al. 2020), SLOPE and Lepski's principle for estimator selection, Surrogate outcomes and the surrogacy condition, Minimax optimality of IPS/DR, Off-policy learning with action embeddings (OffCEM / POTEC)
