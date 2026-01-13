---
title: "A Unified Approach to Interpreting Model Predictions (SHAP)"
authors: ["Scott Lundberg", "Su-In Lee"]
year: 2017
arxiv: "1705.07874"
url: https://arxiv.org/abs/1705.07874
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, theory]
---
## The Core Idea

Many methods existed to answer "why did the model say *this* for *this* row?" — LIME, DeepLIFT, layer-wise relevance propagation, Shapley regression, Shapley sampling, quantitative input influence. Nobody knew how they related. Some were heuristics that people trusted because the pictures looked reasonable.

The insight: **all six are the same kind of object**. Each one hands you a small linear model over binary "feature is on / feature is off" switches:

$$g(z') = \phi_0 + \sum_{i=1}^{M}\phi_i z_i', \qquad z' \in \{0,1\}^M$$

Lundberg and Lee call this family **additive feature attribution methods**. Once you see them as one family, you can ask which member is *right*. And there is exactly one.

Three properties you would want any such explanation to have:

1. **Local accuracy** — the attributions add up to the actual prediction: $f(x) = \phi_0 + \sum_i \phi_i x_i'$.
2. **Missingness** — a feature that is switched off gets zero credit.
3. **Consistency** — if you change the model so feature $i$ contributes *more* in every possible context, its attribution must not go *down*.

Theorem 1: only one set of $\phi_i$ satisfies all three, and it is the **Shapley value** from 1953 cooperative game theory. Everything else in the family — including LIME with its default kernel, and the original DeepLIFT — silently breaks local accuracy or consistency somewhere.

> [!NOTE] Shapley value
> Split a payout among players by averaging each player's marginal contribution over every possible order in which players could join the team. Here the "payout" is the prediction and the "players" are the features. ^shapley-value

What this unlocks: a target to aim at. Every existing method becomes "an approximation of SHAP, good or bad". And the paper gives a second, surprising bridge — Shapley values can be computed by **weighted linear regression**, which makes them far cheaper to estimate.

## The Methodology

**The definition.** SHAP values are Shapley values of the *conditional expectation* of the model:

$$f_x(z') = E[f(z)\mid z_S]$$

where $S$ is the set of features switched on in $z'$. Plug that into the Shapley formula:

$$\phi_i(f,x) = \sum_{z' \subseteq x'} \frac{|z'|!\,(M-|z'|-1)!}{M!}\Big[f_x(z') - f_x(z'\setminus i)\Big]$$

Read it as: how much does knowing feature $i$ move the expected prediction, averaged over all subsets of the other features you might already know. $\phi_0 = E[f(z)]$ is the base rate — what you would predict knowing nothing. The $\phi_i$ walk you from that base value to $f(x)$.

**Why conditional expectation.** Models cannot take "missing" as an input. So "feature is off" is simulated by integrating it out. Two optional approximations make this tractable:

$$E[f(z)\mid z_S] \approx E_{z_{\bar S}}[f(z)] \quad\text{(feature independence)} \approx f([z_S, E[z_{\bar S}]]) \quad\text{(model linearity)}$$

The second one — just plug the mean value in — is what DeepLIFT's "reference value" secretly is.

**Kernel SHAP.** The headline estimator. Take LIME's setup — sample binary masks $z'$, run the model on the masked inputs, fit a weighted least-squares linear model — but replace LIME's hand-picked weighting kernel with the one that provably recovers Shapley values (Theorem 2):

$$\Omega(g)=0,\qquad \pi_{x'}(z') = \frac{M-1}{\binom{M}{|z'|}\,|z'|\,(M-|z'|)}$$

$$L(f,g,\pi_{x'}) = \sum_{z'\in Z}\big[f(h_x^{-1}(z'))-g(z')\big]^2 \pi_{x'}(z')$$

The kernel blows up to infinity when $|z'| = 0$ or $M$ — those two points are the constraints $\phi_0 = f_x(\varnothing)$ and $\sum_i \phi_i = f(x)$, and in practice you eliminate two variables analytically instead of using infinite weights. The shape is symmetric and **U-shaped**: coalitions that are almost empty or almost full get the most weight, because those are the ones that isolate a single feature's effect. LIME's exponential-distance kernel looks nothing like this. That is the whole difference. No [[Regularization|regularisation]] term, ordinary weighted [[Regression Analysis|linear regression]], and you get all $M$ attributions from one fit instead of sampling each feature separately.

**Model-specific shortcuts.**

- *Linear SHAP*: for $f(x)=\sum_j w_j x_j + b$, closed form — $\phi_0 = b$ and $\phi_i = w_i(x_i - E[x_i])$. No sampling at all.
- *Low-order SHAP*: the regression costs $O(2^M + M^3)$, fine for small $M$.
- *Max SHAP*: exact Shapley values for a $\max$ over $M$ inputs in $O(M^2)$ instead of $O(M2^M)$, by sorting inputs and computing the probability each one raises the running maximum. This closes an open problem — DeepLIFT had no principled rule for max-pooling.
- *Deep SHAP*: treat the network as many small pieces, solve each piece's Shapley values analytically (linear layers, max pools, single-input activations), then chain the multipliers backwards like [[Backpropagation|backprop]]:
$$m_{x_j f_3} = \frac{\phi_i(f_3,x)}{x_j - E[x_j]},\qquad m_{y_i f_3} = \sum_j m_{y_i f_j} m_{x_j f_3},\qquad \phi_i(f_3,y)\approx m_{y_i f_3}(y_i - E[y_i])$$
This is DeepLIFT's structure with the *heuristic* linearisation rules replaced by ones derived from Shapley values.

## Ablation Studies and Experiments

**Sample efficiency (Figure 3).** Kernel SHAP (with a debiased lasso) vs Shapley sampling vs open-source LIME, estimating one feature's importance on two decision trees as the number of model evaluations grows. Reported as 10th/90th percentiles over 200 replicate runs.

- Dense tree, 10 features: Kernel SHAP converges with far fewer evaluations than Shapley sampling.
- Sparse tree, 3 of 100 features actually used: the gap is bigger, because the regularised regression exploits sparsity — sampling each feature separately cannot.
- LIME does not converge to the SHAP value at all. It converges to something *else*. That is the point: its kernel is wrong, so it violates local accuracy and/or consistency.

**Human agreement (Figure 4), Mechanical Turk.** Premise: for a model simple enough that a human understands it, a good explanation should match what humans say.

- *Sickness score*: output is 5 if exactly one of {fever, cough} is present, 2 if both, 0 if neither. Explaining the output of 2. Humans split credit in a particular way; SHAP matched it, LIME and DeepLIFT did not. 30 respondents.
- *Max allocation*: three men earn profit equal to the maximum questions any of them got right (5, 4, 0 → \$5 profit). Splitting the \$5. 52 respondents. SHAP matched the common human answer; DeepLIFT's heuristic max rule did not.

**MNIST class differences (Figure 5).** A small convnet (2 conv + 2 dense + 10-way softmax). Compare original DeepLIFT, new DeepLIFT (updated after this work to better approximate Shapley), Kernel SHAP, and LIME. SHAP and LIME both used 50k samples; LIME had to be modified to use single-pixel segments rather than superpixels to be competitive. Mask the 20% of pixels each method says should flip the prediction from 8 to 3, then measure the change in log-odds over 20 random images. Methods closer to SHAP produce a larger flip. The new DeepLIFT, which was revised *because of* this paper's analysis, beats the original.

**What did not work.**

- LIME's default choices — the exponential local kernel, the $\Omega$ complexity penalty, the heuristic loss weighting — do not recover Shapley values. The paper is explicit: "the LIME choices for these parameters are made heuristically," and the consequence is unintuitive attributions in the sickness example.
- Original DeepLIFT's back-prop rules were "intuitive but heuristically chosen" and fail on max pooling.
- Layer-wise relevance propagation is shown to be a *special case* of DeepLIFT with all reference activations set to zero — which means it inherits the same consistency problems, and its reference (all-zeros) is usually not a meaningful "uninformative" baseline.
- Exact SHAP is $O(2^M)$. Every practical variant leans on the independence assumption in Eq. 11, which is false whenever features correlate.

## Worth Remembering

- **The unification is the contribution, not the algorithm.** Shapley values for model explanation existed (Štrumbelj & Kononenko 2014, Lipovetsky & Conklin 2001). What is new is the proof that *any* additive attribution method satisfying three mild properties must be Shapley, which retroactively judges every competing method.
- **Third axiom made redundant.** Young (1985) proved uniqueness from three axioms; here, two of them plus the *missingness* property suffice, because the binary simplified-input setting is more constrained than a general cooperative game.
- **Feature independence is the load-bearing lie.** $E[f(z)\mid z_S] \approx E_{z_{\bar S}}[f(z)]$ means you evaluate the model on rows that may never occur in reality — a 6-foot-tall newborn. With correlated features this produces attributions that are mathematically consistent but describe a fantasy data distribution. The paper flags this as an assumption and moves on; a large follow-up literature (interventional vs conditional SHAP) argues about it.
- **"Consistency" is about comparing models, not features.** It says nothing about whether $\phi_i$ tells you *causation*. A SHAP value is a statement about your fitted model, not about the world. If your model learned a [[Shortcut Learning in Deep Neural Networks|shortcut]], SHAP will faithfully explain the shortcut.
- **The regression trick is the practical gift.** Turning combinatorics into weighted least squares is what made SHAP shippable. Later work (TreeSHAP, same first author) computes exact SHAP for tree ensembles in polynomial time, which is why SHAP is now near-universal alongside [[XGBoost- A Scalable Tree Boosting System|XGBoost]], [[LightGBM- A Highly Efficient Gradient Boosting Decision Tree (NeurIPS)|LightGBM]] and [[CatBoost- Unbiased Boosting with Categorical Features|CatBoost]] on tabular problems.
- **Cost caveat.** Kernel SHAP needs thousands of model evaluations per row explained. For a heavy model this is not something you run on every prediction in production.
- **Human-agreement evaluation is weak evidence.** It works only for toy models a human can hold in their head; it says nothing about whether SHAP is the right explanation for a 100-layer network. Worth remembering when reading claims of "matches human intuition" — see [[Troubling Trends in Machine Learning Scholarship]].
- Follow-up questions: what does $\phi_i$ mean when features are one-hot pieces of one categorical variable? How do you aggregate local SHAP values into a global importance without smuggling in an averaging assumption? What replaces the base value $E[f(z)]$ when your background dataset shifts?

## Links

Related: [[Regression Analysis]] · [[Backpropagation]] · [[Deep Learning]] · [[Saliency]] · [[XGBoost- A Scalable Tree Boosting System]] · [[LightGBM- A Highly Efficient Gradient Boosting Decision Tree (NeurIPS)]] · [[CatBoost- Unbiased Boosting with Categorical Features]] · [[Why do tree-based models still outperform deep learning on tabular data]] · [[Shortcut Learning in Deep Neural Networks]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Regularization]] · [[Random variable]] · [[Troubling Trends in Machine Learning Scholarship]]

New topics worth writing: Shapley value and cooperative game theory, LIME, DeepLIFT, Layer-wise relevance propagation, TreeSHAP, Interventional vs conditional SHAP, Integrated Gradients, Feature importance and permutation importance, Model interpretability vs explainability
