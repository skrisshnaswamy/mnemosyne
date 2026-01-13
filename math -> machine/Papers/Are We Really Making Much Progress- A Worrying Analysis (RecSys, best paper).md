---
title: "Are We Really Making Much Progress? A Worrying Analysis (RecSys, best paper)"
authors: ["Maurizio Ferrari Dacrema", "Paolo Cremonesi", "Dietmar Jannach"]
year: 2019
arxiv: "1907.06902"
url: https://arxiv.org/abs/1907.06902
priority: Must-Read
read_on: 2026-08-22
tags: [paper, vision]
---
## The Core Idea

Between 2015 and 2018 the recommender-systems field filled up with neural models. Everyone assumed deep learning was winning here too, the way it won vision and language. This paper checks that assumption and finds it mostly false.

Two claims, both uncomfortable:

1. **Most of the work cannot be reproduced.** Of 18 neural top-n recommendation papers from KDD, SIGIR, WWW and RecSys, only **7 (39%)** could be re-run at all — meaning working code plus at least one usable dataset with recoverable train/test splits.
2. **Of those 7, six are beaten by simple, well-tuned, non-neural baselines.** Not new baselines. Item-based nearest neighbours from 2001. A three-step random walk on a user–item graph. In one case, recommending the single most popular item list to everybody won.

The mechanism behind the illusion is a feedback loop the authors name directly: *weak baselines get published, then become the baselines for the next paper*. If neural model A was only ever compared to an untuned k-NN, and neural model B beats A, nobody has shown B beats a tuned k-NN. Papers compare deep models to other deep models and the whole family drifts away from the actual state of the art. Add that authors report optimal hyper-parameters for **their** method and say nothing about how baselines were tuned, and the comparison is structurally unfair before a single number is computed.

What this unlocks is not an algorithm. It is a methodology bar: tune your baselines with the same budget you tune your model, publish splits, and check whether a popularity ranker already does the job.

> [!NOTE] Baseline propagation
> When a weak baseline is accepted into the literature, later papers inherit it instead of the true best method. Improvements then accumulate on paper without accumulating in reality. ^baseline-propagation

## The Methodology

**Paper selection.** Manual scan of long papers 2015–2018 in KDD, SIGIR, TheWebConf (WWW), RecSys. Kept if the paper (a) proposed a deep learning method and (b) targeted top-n recommendation, evaluated with ranking/classification metrics (Precision, Recall, MAP, [[NDCG]]). Session-based and group recommendation excluded. Result: 18 papers.

**Reproducibility bar.** Code public or obtainable from authors (30-day wait after contacting them), running with only trivial modification; plus at least one public dataset with the original splits available or reconstructable. Code that was a skeleton with the model gutted counted as non-reproducible. Company-internal data counted as non-reproducible.

Per-conference reproducibility: KDD 3/4, WWW 2/4, SIGIR 1/3, RecSys **1/7**.

**Evaluation approach.** Rather than rebuild every method inside one common framework (which would change the evaluation protocol), they *refactored the original code* so that training, hyper-parameter search and prediction stayed as the authors wrote them, but the evaluation step was split out. That same evaluation code was then used for the baselines. The neural methods ran with the authors' own reported optimal hyper-parameters. So the neural side is being given its best published shot.

**The baselines.** All conceptually simple, all shallow.

- **TopPopular** — non-personalised; rank by interaction count.
- **ItemKNN** — cosine similarity between item rating vectors $\mathbf{r}_i, \mathbf{r}_j \in \mathbb{R}^{|U|}$ with a *shrink* term $h$:
  $$s_{ij}=\frac{\mathbf{r}_i\cdot\mathbf{r}_j}{\|\mathbf{r}_i\|\|\mathbf{r}_j\|+h}$$
  $h$ pushes down similarities computed from few co-ratings. Ratings optionally re-weighted with TF-IDF or BM25; normalisation optionally off.
- **UserKNN** — same thing on user–user similarity.
- **ItemKNN-CBF** — content-based: same cosine but over item feature vectors $\mathbf{f}_i \in \mathbb{R}^{|F|}$.
- **ItemKNN-CFCBF** — hybrid: concatenate $[\mathbf{r}_i, w\mathbf{f}_i]$ then take cosine. Extra hyper-parameter $w$ trades content against ratings.
- **$P^3\alpha$** — three-step random walk on the bipartite user–item graph. Jump user→item with $p_{ui}=(r_{ui}/N_u)^\alpha$, item→user with $p_{iu}=(r_{ui}/N_i)^\alpha$. Equivalent to item-KNN with similarity $s_{ij}=\sum_v p_{jv}p_{vi}$.
- **$RP^3\beta$** — $P^3\alpha$ with similarities divided by item popularity raised to $\beta$; $\beta=0$ recovers $P^3\alpha$. This is an explicit popularity de-bias knob.
- **SLIM** (used once, on MovieLens) — sparse linear item-item model, implemented as scikit-learn ElasticNet, i.e. [[Regularization|L1+L2 regularised]] regression of each item's column on the others.

**Baseline tuning — the part that actually does the work.** Bayesian search via scikit-optimize, **35 trials per algorithm per dataset**, first 5 random. Ranges: neighbourhood $k \in [5, 800]$, shrink $h \in [0,1000]$, $\alpha,\beta \in [0,2]$. For SLIM: L1/L2 ratio $10^{-5}$ to $1.0$, regularisation magnitude $10^{-3}$ to $1.0$. Baselines were optimised for whichever metric the original paper optimised (e.g. HR@5 for CMN, Precision for MCRec).

That's it. No new model. The contribution is 35 trials of Bayesian search on a 2001 algorithm.

## Ablation Studies and Experiments

Six of seven methods lose. Method by method:

**CMN (Collaborative Memory Networks, SIGIR '18)** — memory networks + attention + latent factors + neighbourhood.

| CiteULike-a | HR@5 | NDCG@5 | HR@10 | NDCG@10 |
|---|---|---|---|---|
| RP³β | **0.8226** | **0.7114** | **0.8941** | **0.7347** |
| UserKNN | 0.8213 | 0.7033 | 0.8935 | 0.7268 |
| CMN | 0.8069 | 0.6666 | 0.8910 | 0.6942 |

Beaten on every metric, on CiteULike-a and Pinterest, by at least two baselines. On **Epinions**, CMN did beat the KNN baselines (HR@5 0.4195 vs 0.3821) — but **TopPopular scored 0.5429**, crushing everything. Diagnosis: Epinions has a Gini index of 0.69 on item popularity versus 0.37 for CiteULike-a, and CMN's top-n lists contain items 8–25% more popular on average than the baselines'. CMN's "win" is a popularity bias meeting a popularity-skewed dataset.

**MCRec (KDD '18)** — meta-paths over side information (movie genres etc.), co-attention. On MovieLens100k, plain ItemKNN wins on all three metrics: PREC@10 **0.3327** vs 0.3077, REC@10 **0.2199** vs 0.2061, NDCG@10 **0.2603** vs 0.2363. Two additional problems found by reading the code: the NDCG implementation is non-standard and questionable (authors re-ran with a standard NDCG), and **the best epoch was selected per-metric on the test set**. Also, MF and NeuMF baselines were run with hyper-parameters copied from their original papers, not tuned for these datasets.

**CVAE (Collaborative Variational Autoencoder, KDD '17)** — see [[Auto-Encoding Variational Bayes (VAE)]]. On dense CiteULike-a: REC@50 CVAE 0.0772 versus ItemKNN-CFCBF **0.1837**. The hybrid content+CF neighbourhood method wins by more than 2×. CVAE only edges ahead at cutoff 100–300 in some configurations — list lengths of 300, which the paper never justifies.

**CDL (Collaborative Deep Learning, KDD '15)** — stacked denoising autoencoders + CF. REC@50: CDL 0.0543, ItemKNN-CBF **0.2135**. Pure content-based k-NN is roughly 4× better at cutoff 50. Interesting nuance: CVAE *is* consistently better than CDL, so there is real progress **within** the neural line — it just started below the shallow baselines and hasn't caught up.

**NCF / NeuMF (WWW '17)** — replaces the matrix-factorisation inner product with an MLP. This is the mixed case.
- Pinterest: ItemKNN (HR@10 **0.8744**) and RP³β (0.8740) beat NeuMF (0.8719) on all metrics.
- MovieLens1M: NeuMF genuinely wins over the KNN family — HR@10 0.7120 vs RP³β 0.6715. But **SLIM beats it**: HR@5 0.5589 vs 0.5486, NDCG@10 0.4470 vs 0.4369.

So the one method that clearly beat the simple heuristics still loses to a linear model from 2011. Same epoch-chosen-on-test-set problem as MCRec; the authors fixed it to use a validation set. The original ItemKNN baseline in the NCF paper varied only $k$ — no shrink, no TF-IDF/BM25 weighting.

**SpectralCF (RecSys '18)** — graph convolution in the spectral domain, aimed at cold start. This is the most damning finding. On the authors' *provided* MovieLens1M split, SpectralCF beat everything with Recall@20 about 50% above the best baseline. On HetRec and Amazon Instant Video (splits made by the reproducers), **every baseline including TopPopular beat it**.

So they inspected the provided split. Gini index of item popularity: a genuine random split gives ≈0.79 for both train and test. The provided training split matched. The provided **test** split had Gini **0.92** — far more popularity-concentrated than random sampling could produce. Re-splitting MovieLens1M randomly as the paper describes:

| MovieLens1M | REC@20 | MAP@20 |
|---|---|---|
| RP³β | **0.2910** | 0.1088 |
| UserKNN | 0.2881 | **0.1106** |
| TopPopular | 0.1853 | 0.0576 |
| SpectralCF | 0.1843 | 0.0539 |

SpectralCF now loses to TopPopular. The entire result was the split.

**What the ablations actually reveal.** The load-bearing component is not any architecture — it is hyper-parameter budget on the baseline. The shrink term $h$, the BM25/TF-IDF weighting, and the $\beta$ popularity correction in RP³β are exactly the knobs the original papers left untouched in their k-NN baselines. Turn them on with 35 Bayesian trials and most of the reported gains vanish.

## Worth Remembering

- **Only one method survived contact and even that was conditional.** NeuMF beats simple heuristics on MovieLens1M, loses to them on Pinterest, and loses to SLIM on MovieLens1M. The authors are careful not to say neural recommenders are useless — they say the evidence for them is much thinner than the literature suggests.
- **RecSys itself had the worst reproducibility ratio, 1/7.** The reproducibility rate did rise year over year across the sample.
- **Two of seven papers picked the training epoch using the test set.** Found by reading source code, not stated in either paper. Practical lesson: if you are reproducing something, read the training loop before you trust the table.
- **Popularity bias is the dominant confound.** Two of the "wins" (CMN on Epinions, SpectralCF on MovieLens) trace directly to how popularity is distributed in the evaluation data. Always run TopPopular. If it wins, your metric or your split is telling you something.
- **Sanity-check published splits with a Gini index or Shannon entropy** against a split you make yourself. Cheap, and it caught a headline result here.
- Follow-up worth knowing: this triggered a long argument in the field, a 2021 journal extension by the same authors, and rebuttals arguing the baselines were tuned on the test metric and that the neural papers' contribution was never pure accuracy. The counter-counter-argument is that the neural papers claimed accuracy wins.
- Connects naturally to [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] — both are about the gap between what a paper measures and what actually matters, and to [[Counterfactual Reasoning and Learning Systems]] on the difference between an offline number and a real effect.
- Practical caveat if you want to use the finding: offline top-n accuracy on MovieLens is not the objective any production recommender optimises. A tuned ItemKNN winning HR@10 does not mean it wins on revenue, diversity, or cold-start latency. But it does mean you should build the cheap thing first and force the expensive thing to beat it.

## Links

Related: [[Recommender Systems - Evolution]] · [[Foundational_RecSys_Ranking_Reading_Plan]] · [[NDCG]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Regularization]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Counterfactual Reasoning and Learning Systems]] · [[Reading Queue]]

New topics worth writing: SLIM (sparse linear methods for top-n), Item-based collaborative filtering / ItemKNN, Popularity bias in recommendation, Gini index and Shannon entropy as distribution diagnostics, Bayesian hyper-parameter optimisation, Reproducibility crisis in applied ML, Leave-one-out vs random split evaluation protocols, BM25 and TF-IDF weighting
