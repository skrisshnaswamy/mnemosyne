---
title: "Why do tree-based models still outperform deep learning on tabular data?"
authors: ["Léo Grinsztajn", "Edouard Oyallon", "Gaël Varoquaux"]
year: 2022
arxiv: "2207.08815"
url: https://arxiv.org/abs/2207.08815
priority: Must-Read
read_on: 2026-08-25
tags: [paper, transformers, vision]
---
## The Core Idea

On tables of data — rows of samples, columns of named features like `age`, `weight`, `price` — gradient-boosted trees still beat neural networks. Everyone suspected this. This paper does two things nobody had done properly: it builds a careful benchmark (45 datasets, ~20,000 CPU/GPU hours of random search) that settles the *whether*, and then it runs surgical experiments on the data itself to answer the *why*.

The *why* is three concrete properties of tabular data that clash with three inductive biases of neural nets:

1. **Target functions on tabular data are irregular (bumpy).** Neural nets have a [spectral bias](https://arxiv.org/abs/1806.08734) toward smooth, low-frequency functions. Decision trees fit piecewise-constant staircases, so a sharp jump at `date = 0.37` costs them nothing.
2. **Tabular data is full of useless features.** You can delete half the columns of a typical tabular dataset and a GBT barely notices. MLPs are badly hurt by junk columns; trees and Transformers with feature embeddings are not.
3. **The axes mean something.** An MLP is *rotationally invariant*: rotate every feature vector by the same unitary matrix $R$ and the whole learn-then-predict pipeline gives identical results. That is a bad property here, because the original column basis is the good basis. Ng (2004) proved any rotationally invariant learner needs sample complexity growing at least linearly in the number of irrelevant features — which links (3) straight back to (2).

> [!NOTE] Rotational invariance of a learner
> A learning *procedure* is rotationally invariant if training on $\{(Rx_i, y_i)\}$ and testing on $Rx$ gives the same predictions as training on $\{(x_i,y_i)\}$ and testing on $x$. A plain MLP with a first dense layer is: the layer can absorb $R^{-1}$ into its weights, and standard init/regularisation are isotropic. A decision tree is not — it only ever splits on single axes. ^rotational-invariance

What this unlocks: a checklist for anyone building a tabular architecture. Be robust to junk features. Break rotation invariance. Be able to fit irregular functions. This also retroactively explains *why* per-feature embeddings (FT-Transformer, SAINT, periodic embeddings) help — not because embeddings are magic, but because any embedding destroys rotation invariance.

## The Methodology

**The benchmark.** 45 datasets, mostly from OpenML, filtered hard:

- Heterogeneous columns (no image/signal data where every column is the same sensor).
- $d/n < 1/10$ (not high-dimensional).
- I.I.D. — no time series or streams.
- Real-world, not toy. Simulated is fine if practically relevant (e.g. Higgs).
- $\geq 4$ features, $\geq 3{,}000$ samples.
- **Not too easy**: dropped if default logistic/linear regression lands within 5% relative of *both* a default ResNet and a default HistGradientBoosting. The reasoning: trees beat logistic regression on real tabular data, so if they tie, you are probably at the Bayes rate already.
- **Not deterministic**: no poker, no chess. The target must be noisy.

**Side issues removed on purpose.** Training set truncated to 10,000 samples (the "medium" regime; 50,000 for a smaller "large" study). All missing data deleted (columns first, then rows). Classification binarised to the two biggest classes, then balanced. Categorical features with >20 levels dropped. Numerical features with <10 unique values dropped; with exactly 2, converted to categorical.

**Split.** 70% train, then of the remaining 30%: 30% validation (for picking hyperparameters), 70% test. Folds vary by test-set size — 1 fold above 6,000 test samples, up to 5 folds below 1,000.

**Preprocessing.** Features Gaussianised for NNs with scikit-learn's `QuantileTransformer`. Heavy-tailed regression targets log-transformed; whether to Gaussianise the target is itself a hyperparameter. One-hot encoding for models without native categorical support.

**Models.**

| Family | Models |
|---|---|
| Trees | RandomForest, GradientBoostingTrees (HistGBT when categorical), XGBoost — see [[XGBoost- A Scalable Tree Boosting System]] |
| Deep | MLP, ResNet (MLP + dropout + batch/layer norm + skip connections), FT-Transformer, SAINT |

ResNet and FT-Transformer are from Gorishniy et al. 2021. SAINT (Somepalli et al. 2021) adds *inter-sample* attention — attention across rows, not just across features. The skip connections are the same idea as [[Deep Residual Learning for Image Recognition (ResNet)]]; the attention block is standard [[Attention Is All You Need]] machinery.

**The benchmarking procedure — this is the careful bit.** Hyperparameter tuning itself has variance, and most papers hide it. So: run ~400 random-search iterations per model per dataset. Then, for each budget $n \in \{1, 2, \dots, 400\}$, take the best-on-validation config among the first $n$ iterations and report its test score. Repeat this **15 times with the random-search order reshuffled**, giving a bootstrap-like distribution over "score you'd get with budget $n$". Iteration 1 is always the default hyperparameters. NNs run on GPU (300 epochs, early stopping with patience 40, or 10 for SAINT); trees on CPU.

**Aggregation across datasets.** Affine-renormalise each test score between the best model and the 10% (classification) / 50% (regression) test-error quantile — *not* the worst model, because the worst model is an outlier and tells you nothing about dataset difficulty. Negative regression scores clipped to 0.

**The three probing experiments.** Restricted to numerical features, classification, medium size.

*Smoothing.* Replace each training target with a Gaussian-kernel-smoothed version:
$$\tilde{Y}(X_i) = \frac{\sum_{j=1}^N K(X_i, X_j)\, Y(X_j)}{\sum_{j=1}^N K(X_i, X_j)}, \qquad K(x^*, x) = \exp\!\left(-\tfrac{1}{2}(x^*-x)^\top \Sigma^{-1}(x^*-x)\right)$$
with $\Sigma$ the robust covariance estimate (`MinCovDet`) times a squared lengthscale. Lengthscale 0 = original data. Datasets cut to their 5 most important features first, because kernel smoothing dies in high dimensions. **Only the train set is smoothed; test targets stay original.**

*Feature removal / addition.* Rank features by RandomForest importance. Remove an increasing fraction from the bottom. Separately, *add* features drawn from standard Gaussians, uncorrelated with the target and with each other.

*Rotation.* Gaussianise, then apply a random orthogonal matrix from `scipy.stats.special_ortho_group.rvs` to train and test alike.

## Ablation Studies and Experiments

**Headline result.** Trees win at every random-search budget, on all four settings (classification/regression × numerical-only/mixed). The gap does not close after 400 iterations. And this is measured in *iterations*, which flatters the NNs — plotted against wall-clock time, the gap widens sharply, since tree iterations are much cheaper even though the NNs got GPUs and the trees got CPUs.

**Categorical features are not the main problem.** Common folklore says NNs struggle with tabular data because of categorical columns. Restricting to numerical features only *narrows* the gap — but most of it survives. So categoricals are a real handicap, not the main one.

**Finding 1 — smoothing.** At small lengthscales, smoothing the training target *hurts tree models markedly* but *barely touches NNs*. Read this backwards: the trees were extracting real signal from irregular, high-frequency structure, and the NNs never had that signal in the first place. Concretely, on the `electricity` dataset restricted to its top-2 features, a default RandomForest hits 85% test (100% train) with a decision boundary full of thin vertical stripes along the `date` axis; a default MLP gets 80% and produces a smooth boundary that ignores them. The authors note it is hard, though not impossible, to find MLP hyperparameters that learn those stripes.

**Finding 2 — uninformative features.** A GBT trained *only on the discarded bottom features* scores near chance up to 20% removed and stays poor up to 50% — so those columns are genuinely uninformative, not merely redundant. Then: removing junk features *shrinks* the ResNet-vs-tree gap; adding random Gaussian columns *widens* it. FT-Transformer sits between the two, closer to the trees.

**Finding 3 — rotation.** This is the striking one. Under a random rotation:
- ResNet's accuracy is *unchanged* — confirming it really is rotationally invariant in practice, not just in theory.
- Every other model drops.
- **The ranking flips.** NNs now beat tree-based models, and ResNet beats FT-Transformer.

So it is not that trees are better learners in the abstract. It is that trees exploit the axis-aligned structure of tabular data, and NNs throw that structure away. When you destroy the structure for everyone, the NNs win.

The two experiments then join up (Fig. 6b): remove the least-important half of the features *before* rotating, and every model except ResNet still drops — but by less. The rotation penalty is partly a junk-feature penalty.

**What did not work / what the authors concede.** Hyperparameter tuning is not the fix — the gap persists at 400 iterations. Removing categoricals is not the fix. The one caveat they flag honestly: Kadra et al. (2021) claim a searched "cocktail" of 13 regularisation techniques on a plain MLP matches XGBoost. The authors do not include data augmentation and heavy regularisation cocktails in their search space, so that route stays open. They also note Kadra et al.'s benchmark leans on *deterministic* game-like datasets (poker, chess) where their method excels — exactly the datasets this paper deliberately excludes.

**Hyperparameter importance (from fitting a RandomForest to predict normalised score from hyperparameters).** Learning rate is by far the most important knob for both NNs and GBTs, and its linear coefficient is not consistently large — meaning it needs per-dataset tuning, not a global default. For trees, depth is the second big one, and **deeper is better, even for boosting**. That is Finding 1 again: deep trees can carve irregular patterns.

**Scale.** On the handful of datasets big enough to compare 10k vs 50k training samples, the gap narrows. The authors call this suggestive and leave a proper study to future work.

## Worth Remembering

- The correct mental model: **trees are not stronger, they are better matched.** Rotate the data and the ordering inverts. Inductive bias, not capacity — the same lesson as convolutions on images in [[ImageNet Classification with Deep CNNs (AlexNet)]] and the patch-embedding discussion in [[An Image is Worth 16x16 Words (ViT)]] (see `^inductive-bias`).
- **Practical design rule.** If you build a tabular NN, put a per-feature embedding layer before anything else. Any embedding — learned bins, periodic features, a small per-column MLP — breaks rotation invariance, and the paper's reading is that *breaking the invariance* is most of the benefit, not the specific embedding. They flag "find a cheaper way to break rotation invariance than embeddings" as an open problem worth taking.
- The connection to $L_1$ regularisation is direct and worth holding onto: Ng (2004), the paper they lean on, is literally titled "Feature selection, $L_1$ vs. $L_2$ regularization, and rotational invariance". $L_1$ is not rotationally invariant and that is exactly why it does feature selection. Cf. [[Loss, Objectives, and Business Alignment]] and [[Regularization]].
- **Everything they excluded is a caveat for real use.** No missing values, no high-cardinality categoricals, balanced binary targets, no time series, ≤10k rows. Real tabular problems have all of these. The conclusion "use XGBoost" holds; the specific *why* may shift when missingness or 1M-row tables enter.
- Their benchmarking protocol is reusable beyond this paper: **report score as a function of tuning budget, averaged over shuffled search orders**, with default hyperparameters as iteration 1. That single plot kills most "our model wins" claims that come from unequal tuning effort. Connects to [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] and its `^baseline-propagation`.
- They released the raw CSV of all ~400 × 45 × 7 random-search runs (20,000 compute hours). You can benchmark a new method against a *fixed tuning budget* without re-running any of it. Repo: `github.com/LeoGrin/tabular-benchmark`.
- Open question I'd chase: is the smoothness bias fixable by architecture (Fourier/periodic embeddings, ExU activations) or is it fundamental to gradient descent on smooth parameterisations? [[Fourier Series Decomposition]] is the right lens — a piecewise-constant staircase is all high frequencies, and that is precisely what SGD learns last.

## Links

Related: [[XGBoost- A Scalable Tree Boosting System]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Attention Is All You Need]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Regularization]] · [[Loss, Objectives, and Business Alignment]] · [[Practical Lessons from Predicting Clicks on Ads at Facebook (ADKDD)]] · [[Practical Bayesian Optimization of Machine Learning Algorithms]] · [[The Bitter Lesson (essay)]] · [[Fourier Series Decomposition]] · [[Deep Learning]]

New topics worth writing: Spectral bias of neural networks (Rahaman et al.), Rotational invariance and L1 feature selection (Ng 2004), FT-Transformer and numerical feature embeddings, SAINT and inter-sample attention, Random search vs Bayesian hyperparameter optimisation budgets, Quantile transformation and feature Gaussianisation, Random Forest feature importance as a ranking tool
