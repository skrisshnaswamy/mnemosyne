---
title: "Variational Autoencoders for Collaborative Filtering"
authors: ["Liang et al."]
year: 2018
arxiv: "1802.05814"
url: https://arxiv.org/abs/1802.05814
priority: Good-To-Read
read_on: 2026-09-26
tags: [paper, theory, scaling]
---
## The Core Idea

Take a [[Variational Autoencoder]], feed it one user's whole click history as a single sparse vector, and have it reconstruct that history. The bottleneck vector $\mathbf{z}_u$ becomes the user's latent taste; the decoder's output scores every item at once, and you rank by that score.

Two changes are what actually make it work, and both are small:

**1. Multinomial likelihood instead of Gaussian or logistic.** The decoder ends in a softmax over all $I$ items, so the predicted probabilities must sum to 1. Items compete for a fixed budget of probability mass. That competition is what makes the loss behave like a ranking loss instead of a per-item regression.

**2. Turn the KL term down.** The standard VAE loss ([[Auto-Encoding Variational Bayes (VAE)#^elbo|ELBO]]) weights reconstruction and KL equally. On sparse, high-dimensional click data that over-regularises: the model underfits badly. Multiply the KL by $\beta < 1$ and it works. The best $\beta$ on ML-20M was around 0.2.

> [!NOTE] Partially regularised VAE
> A VAE trained with $\beta < 1$ on the KL term. It is no longer a lower bound on the log likelihood, so you lose the ability to sample new fake users from the prior. For recommendation you never wanted that — you always predict *conditional on* a real user's history. So the loss costs nothing. ^partially-regularised-vae

Why it did not exist before: [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)|matrix factorization]] and friends are linear, and the neural CF papers of the time ([[Neural Collaborative Filtering|NCF]], CDAE) had per-user parameters that overfit on big catalogues. A VAE's parameter count grows with items only, not users, and a held-out user needs no optimisation at all — one forward pass through the encoder gives you $\mathbf{z}$.

The other framing worth keeping: this paper is an argument that recommendation is a **small-data** problem wearing big-data clothes. Millions of users, but each one has ~50 signals out of 20,000 items. That is why the Bayesian treatment helps.

## The Methodology

**Input.** Binarise the click matrix $\mathbf{X} \in \mathbb{N}^{U \times I}$. One training example is one user's row $\mathbf{x}_u$ — a bag-of-items vector of length $I$.

**Generative model.**

$$\mathbf{z}_u \sim \mathcal{N}(0, \mathbf{I}_K), \quad \pi(\mathbf{z}_u) \propto \exp\{f_\theta(\mathbf{z}_u)\}, \quad \mathbf{x}_u \sim \mathrm{Mult}(N_u, \pi(\mathbf{z}_u))$$

$f_\theta$ is an MLP; $N_u = \sum_i x_{ui}$ is that user's click count. The log-likelihood is just cross-entropy against the click counts:

$$\log p_\theta(\mathbf{x}_u \mid \mathbf{z}_u) \overset{c}{=} \sum_i x_{ui} \log \pi_i(\mathbf{z}_u)$$

Set $f_\theta$ linear and swap in a Gaussian likelihood and you recover classical matrix factorization. So this strictly generalises it.

**Inference model (encoder).** Rather than one variational parameter pair per user, an amortised network $g_\phi(\mathbf{x}_u) = [\mu_\phi(\mathbf{x}_u), \sigma_\phi(\mathbf{x}_u)]$ outputs the mean and variance of a diagonal Gaussian. Sample with the [[Auto-Encoding Variational Bayes (VAE)#^reparameterisation-trick|reparameterisation trick]]: $\mathbf{z}_u = \mu_\phi + \epsilon \odot \sigma_\phi$, $\epsilon \sim \mathcal{N}(0, \mathbf{I})$, so gradients flow to $\phi$.

**The objective.**

$$\mathcal{L}_\beta(\mathbf{x}_u) = \mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}_u \mid \mathbf{z}_u)] - \beta \cdot \mathrm{KL}(q_\phi(\mathbf{z}_u \mid \mathbf{x}_u) \,\|\, p(\mathbf{z}_u))$$

$\beta = 1$ is the ordinary ELBO. $\beta \gg 1$ is β-VAE for disentanglement. This paper lives at $\beta \in [0, 1]$.

**Finding $\beta$ cheaply.** Do not grid search. Start at $\beta = 0$ and linearly anneal up towards 1 over 200,000 gradient updates, watching validation NDCG@100. The curve rises then falls. Note the $\beta$ at the peak, retrain with the same schedule but cap $\beta$ there. Cost: roughly two training runs, not twenty. If you are broke, just stop annealing the moment validation dips — one run, zero overhead.

**Architecture.** Symmetric encoder/decoder, $K = 200$ latent dims, hidden layers of 600, tanh between layers. So $[I \to 600 \to 200 \to 600 \to I]$. Zero or one hidden layer won; **two hidden layers did not help**. With zero hidden layers the VAE is effectively a log-linear model.

**Training.** [[Adam- A Method for Stochastic Optimization|Adam]], batch size 500 *users*, 200 epochs on ML-20M and 100 on the others. Dropout $p = 0.5$ on the input layer. No weight decay on the VAE. Because a batch is whole user rows, there is **no negative sampling** and no negative-sample-count hyperparameter to tune — a real simplification versus the (user, item)-entry sampling that CDAE and NCF used.

**Prediction.** Encode the fold-in history, take $\mathbf{z} = \mu_\phi(\mathbf{x})$ (mean, no sampling), decode, rank items by the un-normalised $f_\theta(\mathbf{z})$. Two function evaluations. No per-user optimisation for unseen users — which is the deployment argument.

**The sibling model: Mult-DAE.** Same multinomial decoder, but the encoder outputs a point $\mathbf{z} = g_\phi(\mathbf{x})$ instead of a distribution, and there is no KL term at all. This is a denoising autoencoder (the dropout is the noise). It needs weight decay 0.01 to not overfit. It is the ablation that isolates "does the distribution over $\mathbf{z}$ buy anything?"

**Evaluation protocol — strong generalisation.** Users, not interactions, are split into train/valid/test. Held-out users are never seen during training. For each, 80% of their history is "folded in" to compute $\mathbf{z}$, and metrics are measured on the remaining 20%. This is harder and more realistic than weak generalisation, where the same user appears in both train and test.

**Data.** ML-20M (137k users, 20k items, 10M interactions, 0.36% dense), Netflix (463k / 17.8k / 56.9M, 0.69%), MSD (571k / 41k / 33.6M, 0.14%). Ratings ≥ 4 counted as a click.

## Ablation Studies and Experiments

**Headline, NDCG@100:**

| | ML-20M | Netflix | MSD |
|---|---|---|---|
| Mult-VAE$^{\text{PR}}$ | **0.426** | **0.386** | **0.316** |
| Mult-DAE | 0.419 | 0.380 | 0.313 |
| WMF | 0.386 | 0.351 | 0.257 |
| SLIM | 0.401 | 0.379 | — |
| CDAE | 0.418 | 0.376 | 0.237 |

Standard errors ~0.002 (ML-20M), ~0.001 (others). SLIM could not be run on MSD — the parallel grid search took two weeks on Netflix alone.

**The likelihood ablation** — take the best model, swap only the output distribution, retune its hyperparameters (ML-20M, NDCG@100):

| | VAE$^{\text{PR}}$ | DAE |
|---|---|---|
| Multinomial | **0.426** | **0.419** |
| Logistic | 0.419 | 0.414 |
| Gaussian | 0.415 | 0.409 |

Multinomial wins, but the margin over logistic is small (0.007). That makes sense — a softmax can be approximated by independent binary logistic terms, which is standard practice in language modelling. Gaussian is clearly worst. So the likelihood choice is real but modest; it is not the whole story.

**The $\beta$ ablation is the load-bearing one.** Figure 1: no annealing at all gives poor validation NDCG. Annealing all the way to $\beta = 1$ (reached around epoch 80) rises, peaks, then *falls back* to barely better than no annealing. Capping $\beta$ at the peak keeps the gain. This is the clearest evidence that the standard ELBO is over-regularised for this data — and it matches Krishnan et al. (2017), who found VAEs underfit sparse high-dimensional data.

**VAE vs DAE, broken down by user activity.** Test users split into quintiles by fold-in click count, paired t-tests per bin:

- On ML-20M the VAE wins significantly on the least-active users — exactly where the prior does useful work by shrinking a noisy estimate.
- On the most active ML-20M users, **Mult-DAE actually beats Mult-VAE$^{\text{PR}}$**. The prior becomes a liability when there is plenty of data per user.
- On MSD the least-active bin shows no gap — but MSD was preprocessed to require ≥ 20 songs per user, so its "least active" bin is much more active than ML-20M's ≥ 5-movie bin. The confound is admitted.

**What did not work:**

- **NCF would not train** on these datasets. The authors used the original source code; validation metrics dropped within a few epochs across a wide range of regularisation settings. They fell back to comparing on NCF's own two small datasets (ML-1M, Pinterest), where Mult-DAE with a bare $[I \to 200 \to I]$ beats NCF-without-pretraining on both, and beats *pretrained* NCF on Pinterest (NDCG@10 0.580 vs 0.558).
- **CDAE collapses on MSD** — NDCG@100 0.237, below even WMF's 0.257. Its parameter count grows with users *and* items, and validation metrics fell while training loss kept improving. Classic overfitting.
- **BPR** was not competitive enough to report, consistent with Sedhain et al. (2016).
- **Depth past one hidden layer** gave no improvement.
- **The fast approximation to SLIM** (Levy & Jack 2013) was not competitive.
- **Plain autoencoders** (no noise, no KL) overfit hard — the network dumps all probability mass on the observed non-zeros.

## Worth Remembering

**The honest reading of the results.** Mult-VAE$^{\text{PR}}$ beats Mult-DAE by 0.007 NDCG@100 on ML-20M and 0.003 on MSD. Both crush the linear baselines. So the win is mostly "deep multinomial autoencoder over whole user rows" and only secondarily "Bayesian". The VAE's real selling points are robustness to hyperparameters (no weight decay needed, unlike DAE) and better handling of cold-ish users.

**The $\beta$ trick generalises; the multinomial likelihood may not.** The authors say this explicitly. Likelihood choice is data-dependent. Partial regularisation of a KL term is a technique you can carry elsewhere.

**Information-theoretic framing.** $\beta < 1$ is the deep variational information bottleneck of Alemi et al. (2017), who also found $\beta < 1$ helps for supervised classification. It also resembles maximum-entropy discrimination — $\beta$ is a dial between discriminative and generative behaviour. β-VAE pushes the same knob the *other* way ($\beta \gg 1$) for disentanglement. Same equation, opposite motivations.

**Scaling caveat that matters in production.** The softmax normalises over all $I$ items. Under 50k items this was fine. Above that it becomes the bottleneck, and you need an approximate normaliser (they point at Botev et al. 2017) — or you are back to the sampled-softmax and [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)|logQ-correction]] machinery of two-tower retrieval.

**This is a full-catalogue scorer, not a retriever.** The decoder produces a score for every item in one pass, which is exactly what you cannot afford with millions of items. It slots in as a re-scorer or as a model for medium catalogues, not as candidate generation.

**No sequence, no time.** The input is a bag of items. Order is thrown away. That is the gap [[Self-Attentive Sequential Recommendation (SASRec)|SASRec]] and [[BERT4Rec- Sequential Recommendation with Bidirectional Transformer|BERT4Rec]] fill. No side information either, though the authors note conditioning on it would be the obvious extension.

**Read this alongside the replication critique.** [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|Dacrema et al.]] found many neural recommenders lose to well-tuned simple baselines. Mult-VAE is one of the few that has repeatedly survived that scrutiny — partly because this paper used strong generalisation and tuned its baselines properly (a two-week SLIM grid search is not a strawman). Also read [[On Sampled Metrics for Item Recommendation (KDD)|Krichene & Rendle]] before trusting any NDCG comparison drawn from sampled candidate sets; this paper ranks the full catalogue, which is the right thing.

**Open question the authors leave.** They have no theory for *why* $\beta < 1$ works so well, only the information-bottleneck analogy. Eight years on this is still mostly empirical.

## Links

Related: [[Variational Autoencoder]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[KL Divergence]] · [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)]] · [[Collaborative Filtering for Implicit Feedback Datasets (ICDM)]] · [[Neural Collaborative Filtering]] · [[Neural Collaborative Filtering vs. Matrix Factorization Revisited]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[NDCG]] · [[On Sampled Metrics for Item Recommendation (KDD)]] · [[Autoencoder]] · [[Latent Variable Models]] · [[Cross Entropy]] · [[Dropout- A Simple Way to Prevent Overfitting]] · [[Regularization]] · [[Recommender Systems - Evolution]] · [[BPR- Bayesian Personalized Ranking from Implicit Feedback]] · [[Distributed Representations of Words and Phrases (negative sampling)]]

New topics worth writing: β-VAE and the KL-weight knob, KL annealing schedules, deep variational information bottleneck, strong vs weak generalisation in recsys evaluation, SLIM and sparse linear item-item models, EASE (the closed-form successor that beat Mult-VAE), amortised variational inference, sampled softmax normalisation for large item sets
