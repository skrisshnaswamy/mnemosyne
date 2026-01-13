---
title: "On the Difficulty of Evaluating Baselines"
authors: ["Steffen Rendle", "Li Zhang", "Yehuda Koren"]
year: 2019
arxiv: "1905.01395"
url: https://arxiv.org/abs/1905.01395
priority: Must-Read
read_on: 2026-08-25
tags: [paper, rl]
---
## The Core Idea

Five years of recommender papers reported beating "matrix factorization" on Movielens 10M. The authors took plain matrix factorization — a method from 2007 — tuned it carefully, and beat every single one of those newer methods.

The claim is not "those methods are bad". The claim is sharper and more uncomfortable: **running someone else's baseline properly is a hard skill, and nobody is rewarded for doing it well.** A researcher inventing method X has every incentive to get X right and roughly zero incentive to spend two weeks learning how to make a 2008 Gibbs sampler sing. So the baseline number goes into the table under-tuned, the improvement looks real, it passes peer review, and the next paper copies that same bad number as its own baseline. Numbers propagate. Nobody re-measures.

The killer detail: the ML10M benchmark satisfies every hygiene rule the community has. Public data. Simple, documented 90:10 split protocol. Hyperparameter search reported. Standard deviations reported. Statistical significance claimed. Code often released. And it still produced five years of wrong conclusions. So the usual checklist — reproducible, tuned, significant — is **necessary but not sufficient**. Significance measures variance *within one setup*. It says nothing about whether the setup was any good.

> [!NOTE] Baseline propagation
> A weak number for a well-known method gets published once, then copied from paper to paper as the "known" performance of that method. Nobody re-runs it, because re-running old methods is not publishable. The error compounds silently. ^baseline-propagation

The contrast case is the Netflix Prize. Same difficulty of tuning, but the incentive was *the RMSE itself*, not novelty. So people happily published "I got matrix factorization to 0.8985" as a contribution. Over three years the community converged on well-calibrated numbers for simple methods. And the lessons learned there (implicit feedback helps, time helps) transfer perfectly to ML10M — the authors confirm it.

This connects directly to [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]], which found the same disease in the top-N recommendation literature.

## The Methodology

There is no new model here. The whole method is: take known models, set them up properly, measure.

**Benchmark.** Movielens 10M, 10-fold cross validation, each fold a random 90:10 train:test split, metric RMSE. This matches the protocol used by all the papers being compared against.

**Implementation.** Everything is built as a [[Factorization Machines (ICDM)|Factorization Machine]] using libFM, with the relational-data solver. A factorization machine with the right feature set *is* each of these classic models. Features:

| Symbol | Feature |
|---|---|
| u | user id (categorical) |
| i | item id (categorical) |
| t | day of rating (categorical, one bucket per day) |
| iu | bag-of-words of every movie this user watched |
| ii | bag-of-words of every user who watched this movie |

| Model | Features | Equivalent to |
|---|---|---|
| Matrix Factorization | u, i | Biased MF / RSVD |
| timeSVD | u, i, t | temporal MF |
| SVD++ | u, i, iu | Koren 2008 |
| timeSVD++ | u, i, t, iu | Koren 2009 |
| timeSVD++ flipped | u, i, t, iu, ii | new-ish; "flipped" implicit info |

**Learning — mostly Bayesian.** They lean on Gibbs sampling ([[Markov Chain Monte Carlo|MCMC]]) rather than SGD, and the stated reason is the whole point of the paper: *Bayesian learning has fewer critical hyperparameters, so it is harder to set up badly.* The priors (including regularisation strength) are sampled along with the parameters, so you never tune a regularisation constant. Only three knobs:

1. Number of sampling steps — more is better, up to 512.
2. Embedding dimension — bigger is better, up to 512.
3. Initialisation — Gaussian with $\sigma = 0.1$, the libFM default, unchanged from Netflix Prize practice.

Both "more is better" curves are monotone (Figures 3 and 4). No early stopping trickery. This is the practical takeaway: a method whose quality only increases along its knobs is a method a stranger cannot mis-tune.

**Learning — SGD, for comparability.** L2-regularised MF trained with SGD, 128 epochs fixed. Grid: regularisation $\lambda \in \{0.02, 0.03, 0.04, 0.05\}$, learning rate $\eta \in \{0.001, 0.003\}$. Tuned at $d=64$ on a 5% held-out slice of train, then retrained on full train. Winner was $\lambda = 0.04$, $\eta = 0.003$, stable across all 10 folds. The grid range was chosen from what worked on Netflix — i.e. from *experience*, not from a blind search.

## Ablation Studies and Experiments

**The headline table.** Previously published numbers on ML10M (RMSE, lower is better):

| Method | RMSE | Who reported |
|---|---|---|
| RSVD (=MF, SGD) | 0.8256 | a later paper's baseline |
| U-RBM | 0.823 | baseline |
| BPMF (Bayesian MF) | 0.8197 | baseline |
| Biased MF | 0.803 | baseline |
| ALS-WR | 0.7830 | baseline |
| I-AutoRec | 0.782 | its inventors |
| LLORMA | 0.7815 | its inventors |
| WEMAREC | 0.7769 | its inventors |
| CF-NADE (2 layer) | 0.771 | its inventors |
| GLOMA | 0.7672 | its inventors |
| AdaError | 0.7644 | its inventors |
| **MRMA (best ever published, NeurIPS 2017)** | **0.7634** | its inventors |

Their re-runs:

| Method | RMSE |
|---|---|
| SGD MF, $d=64$ | 0.7756 |
| SGD MF, $d=512$ | 0.7720 |
| Bayesian MF, $d=512$ | **0.7633** |
| Bayesian timeSVD | 0.7587 |
| Bayesian SVD++ | 0.7563 |
| Bayesian timeSVD++ | 0.7523 |
| Bayesian timeSVD++ flipped | **0.7485** |

Read the two tables against each other:

- Plain SGD MF at 0.7720 beats LLORMA, AutoRec, WEMAREC, I-CFN++ — four "novel" methods published as improvements over it. Reported as 0.8256/0.803; actual 0.7720. That is a gap of ~0.05, five times the size of a full year of Netflix Prize progress.
- Bayesian MF, the method reported as *third worst in the table* at 0.8197, actually lands at 0.7633 — dead level with MRMA, the best published result. The reported baseline was wrong by 0.056.
- Decade-old SVD++ and timeSVD++ then blow past everything. Final 0.7485 beats MRMA by 0.0144 — roughly the total improvement the field claimed over several years.

**What the ablation actually reveals.** Two levers do the work, and neither is architectural:

1. *Learning algorithm.* Gibbs sampling beats SGD on the identical model: 0.7633 vs 0.7720 at $d=512$ (Figure 5). Same objective, same parameterisation, different inference — and the Bayesian one is also the one with fewer things to get wrong.
2. *Features, not depth.* Adding time ($-0.0046$), then implicit feedback ($-0.0070$), then flipped implicit item info ($-0.0038$) gives far more than any neural architecture in the table. All three ideas are from 2008–2009.

**Consistency checks (the "did it not work" section, sort of).** The authors are careful about internal contradictions in the published literature, which is what tipped them off:
- Biased MF and RSVD are *literally the same algorithm* — L2 MF trained by SGD — yet are reported at 0.803 and 0.8256. Any difference is setup noise, so at least one is badly tuned.
- ALS-WR and Biased MF are the same *model*, different optimiser; on Netflix they tie when both are tuned. On ML10M they were reported 0.005 apart.
- BPMF was the best MF learner on Netflix, yet appears near the bottom on ML10M. That is a red flag, not a finding.

**What did not fully work / was left on the table.** The SGD numbers are admittedly not maximal. The authors note that on Netflix they got further with separate learning rates and regularisation for biases vs embeddings, and for users vs items, plus a learning-rate decay schedule — none of which they used here. So even the *paper about tuning baselines properly* concedes its own SGD baseline is under-tuned. That is either delicious irony or the strongest form of the argument, depending on your mood.

**Netflix Prize as a control group.** Reported RMSE for *vanilla* matrix factorization over three years: 0.94 (SVD with imputation) → 0.93 (FunkSVD) → 0.9227 → 0.9190 → 0.9094 → 0.9028 → 0.9018 → 0.8998 → 0.8985. Same model class, spread of 0.04 across teams. Convergence took a competition with a $1M prize. For scale: the entire Grand Prize was a 0.095 relative improvement, and a full year of community effort moved 0.8712 → 0.8616.

**On benchmark overfitting.** They pre-empt the obvious objection: after ten years of study, Netflix's public quiz leaderboard and private test leaderboard show near-identical ordering and the same relative gaps. At ML-dataset scale, overfitting the benchmark is a smaller risk than mis-tuning the baseline.

## Worth Remembering

- **Statistical significance is a trap here.** MRMA's paper reports BPMF with a standard deviation of 0.0004 — beautifully tight, and 0.056 away from the truth. Low variance within a bad setup tells you the setup is *consistently* bad. Significance testing should be the *last* check, not the argument.
- **Reproducibility does not save you either.** ML10M is public, the split is trivial, implementations exist, authors release code. All of it reproduced fine. Reproducible ≠ correct. You can perfectly reproduce a wrong number.
- **Hyperparameter search does not save you.** The authors' diagnosis is that grids are chosen without experience: is the optimum at the grid boundary? Should the grid be finer? Does a $d=64$ optimum transfer to $d=512$? And there are knobs nobody thinks to search — centring, shuffling, init scale, when to stop. An expert knows these implicitly and would not think them worth reporting; a non-expert never learns they exist. This is silent, and it is the main failure mode.
- **Practical rule for your own work:** prefer methods with monotone knobs. If "more dimensions, more steps" always helps, a stranger cannot ruin your baseline. That is a real argument for Bayesian/MCMC inference in benchmark settings that has nothing to do with priors or uncertainty.
- **Second practical rule:** if two entries in your results table are the same algorithm under different names and they differ, stop and debug before publishing. The authors found the bug purely by reading the table sceptically.
- **The authors are not dismissive of the newer work.** They explicitly say local low-rank structure, mixtures of matrix approximation, and autoencoders are useful ideas. The complaint is about the empirical claim, not the ideas. Also: the new methods may themselves be under-tuned, which would only strengthen the thesis.
- **The proposed fix is social, not technical.** (1) Standardised benchmarks with fixed splits, and *discourage* authors from running baselines from scratch. (2) Create incentives to publish "I made an old method better" — competitions, Kaggle, KDDCup, or leaderboards where a tuning result counts as a contribution. Novelty-as-sole-criterion is the root cause.
- **Open question this leaves.** If baseline numbers on the most-studied rating dataset in the field were wrong for five years, what is the state of leaderboards where the protocol is *not* standardised — which is most of recommender systems, and arguably most of applied ML? The authors say one-off evaluations are strictly worse. Worth holding this next to [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] and [[Understanding Deep Learning Requires Rethinking Generalization]] as papers about how ML knows less than it thinks.
- Note the historical rhyme: [[Recommendations as Treatments- Debiasing Learning and Evaluation]] and this paper both say the evaluation, not the model, is where the error lives.

## Links

Related: [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Factorization Machines (ICDM)]] · [[Collaborative Filtering for Implicit Feedback Datasets (ICDM)]] · [[Recommender Systems - Evolution]] · [[Markov Chain Monte Carlo]] · [[Hamiltonian Monte Carlo]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Practical Bayesian Optimization of Machine Learning Algorithms]] · [[Regularization]] · [[Foundational_RecSys_Ranking_Reading_Plan]] · [[BPR- Bayesian Personalized Ranking from Implicit Feedback]] · [[The Bitter Lesson (essay)]]

New topics worth writing: Bayesian Probabilistic Matrix Factorization (BPMF), SVD++ and implicit feedback in rating prediction, timeSVD++ and temporal dynamics, Gibbs sampling for latent factor models, RMSE as a recommender metric, the Netflix Prize, benchmark overfitting vs adaptive data analysis, libFM
