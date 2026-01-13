---
title: "Calibrated Recommendations (RecSys)"
authors: ["Steck"]
year: 2018
url: https://dl.acm.org/doi/10.1145/3240323.3240372
priority: Good-To-Read
read_on: 2026-08-23
tags: [paper]
---
## The Core Idea

A user watched 70 romance films and 30 action films. A good top-10 list should be about 7 romance and 3 action. In practice a recommender trained for accuracy gives you 10 romance and 0 action. That gap is the whole paper.

The reason is not a bug. It is what accuracy-maximising *means*. If you only know the genre split and nothing else, the best bet for every single slot is the majority class. Predicting "romance" every time is right 70% of the time. Sampling genres in proportion (70/30) is right only $0.7\cdot 0.7 + 0.3 \cdot 0.3 = 58\%$ of the time. So any model chasing a ranking metric will happily wipe out the minority interest. Real data has more signal than "just the genre", but data is always finite and noisy, so the effect survives in weakened form.

The second thought experiment makes it sharper. Say each film has its own probability $p(i\mid g)$ inside a genre, and genres are exclusive, so $p(i\mid u) = p(i\mid g)\,p(g\mid u)$. Compare the 10th-best romance film to the *best* action film:

$$\frac{p(i_{g_r,10}\mid u)}{p(i_{g_a,1}\mid u)} = \underbrace{\frac{p(i_{g_r,10}\mid g_r)}{p(i_{g_a,1}\mid g_a)}}_{\approx 1/2.1} \cdot \underbrace{\frac{p(g_r\mid u)}{p(g_a\mid u)}}_{2.33} \approx 1.11 > 1$$

The number 2.1 is measured on MovieLens-20M. The ratio is above 1, so the 10th romance film still beats the single best action film. Rank by score, get zero action films. Not one.

> [!NOTE] Calibrated recommendation
> A recommendation list is calibrated when the mix of categories in the list matches the mix of categories in the user's own history. If they played genre $g$ 30% of the time, roughly 30% of the list should be $g$. ^calibrated-list

The interesting twist: **sorting is the culprit, not the model.** If you take a trained LDA-style model and *sample* from it (pick a genre from $p(g\mid u)$, then a film from $p(i\mid g)$), the list comes out balanced by construction. Ranking by $p(i\mid u) = \sum_g p(i\mid g) p(g\mid u)$ destroys that balance while raising accuracy. So the generative process is calibrated and the argmax over it is not.

This unlocks a framing: a user's minority interests are a fairness problem *inside a single person*. Standard fairness criteria (equalised odds, equal opportunity) do not apply here, because [16] proved they are incompatible with calibration unless the groups have the same base rate — and 70% vs 30% is exactly a different base rate. Hence calibration is the criterion that fits.

## The Methodology

Three pieces: a target distribution, a metric, a re-ranker.

**The two distributions.** Every film $i$ has a genre distribution $p(g\mid i)$, assumed given. On MovieLens, if a film is tagged with 3 genres, each gets $1/3$, so $\sum_g p(g\mid i) = 1$.

User history distribution, over the set $\mathcal{H}$ of films played:

$$p(g\mid u) = \frac{\sum_{i\in\mathcal{H}} w_{u,i}\, p(g\mid i)}{\sum_{i\in\mathcal{H}} w_{u,i}}$$

where $w_{u,i}$ can weight recent plays higher. Recommendation list distribution, over the list $\mathcal{I}$:

$$q(g\mid u) = \frac{\sum_{i\in\mathcal{I}} w_{r(i)}\, p(g\mid i)}{\sum_{i\in\mathcal{I}} w_{r(i)}}$$

with $w_{r(i)}$ a rank discount — you can borrow MRR or [[NDCG]] weights so that top slots count more.

**The metric.** Use [[KL Divergence]] with the history as the target:

$$C_{KL}(p,q) = KL(p \| \tilde q) = \sum_g p(g\mid u)\log\frac{p(g\mid u)}{\tilde q(g\mid u)}$$

KL blows up when $q(g\mid u)=0$ and $p(g\mid u)>0$ — exactly the case we care about — so smooth it:

$$\tilde q(g\mid u) = (1-\alpha)\, q(g\mid u) + \alpha\, p(g\mid u), \qquad \alpha = 0.01$$

Three properties make KL the right pick here:
1. It is exactly 0 at perfect calibration.
2. It is very sensitive to errors where $p$ is *small*. Playing a genre 2% and recommending 1% is punished harder than playing 50% and recommending 49%.
3. It prefers the flatter side of an error. If the target is 30%, recommending 31% scores better than 29%.

Hellinger distance $C_H = \|\sqrt p - \sqrt q\|_2 / \sqrt 2$ also works and handles zeros natively, but is less sensitive in the small-$p$ regime.

**The re-ranker.** Calibration is a property of the whole list, and most rankers are trained pointwise or pairwise, so you cannot easily bake it into training. Post-process instead, via maximum marginal relevance:

$$\mathcal{I}^* = \arg\max_{|\mathcal{I}|=N}\ (1-\lambda)\, s(\mathcal{I}) - \lambda\, C_{KL}(p, q(\mathcal{I}))$$

with $s(\mathcal{I}) = \sum_{i\in\mathcal{I}} s(i)$ the base model's scores, and $\lambda\in[0,1]$ trading accuracy against calibration.

Picking the best set of $N$ is NP-hard. Go greedy: start empty, add one film at a time, each time picking the film that maximises the objective for $\mathcal{I}_{n-1}\cup\{i\}$.

**Why greedy is safe.** Expand the KL:

$$C_{KL} = \underbrace{-H(p)}_{\text{const}} + \underbrace{\log\textstyle\sum_{r=1}^{|\mathcal{I}|} w_r}_{\text{const at fixed size}} - \sum_g p(g\mid u)\log\sum_{i\in\mathcal{I}} w_{r(i)}\,\tilde q(g\mid i)$$

The first term is the entropy of the user's history — fixed. The second depends only on list size. The third is submodular (a log-sum over a set). Dropping the constants gives

$$\mathcal{I}^* = \arg\max_{|\mathcal{I}|=N}\ (1-\lambda)\, s(\mathcal{I}) + \lambda \sum_g p(g\mid u)\log\sum_{i\in\mathcal{I}} w_{r(i)}\,\tilde q(g\mid i)$$

which is modular + submodular = submodular, so greedy gets the classic $(1-1/e)$ guarantee. At each step both objectives pick the same film, so nothing is lost. Bonus: greedy yields an *ordered* list, and every prefix of length $n < N$ is itself $(1-1/e)$-optimal — which matters when the user only sees the first few slots before scrolling.

**Escaping the filter bubble.** Pure calibration mirrors the past, so it can trap you. Mix in a prior $p_0(g)$ (uniform, or the average genre distribution across all users):

$$\bar p(g\mid u) = \beta\, p_0(g) + (1-\beta)\, p(g\mid u)$$

and use $\bar p$ as the target. $\beta$ dials diversity against calibration. Several categories at once (genre, subgenre, language, film-vs-TV) just means adding one $\lambda^{(cat)} C_{KL}^{(cat)}$ term per category — the sum of submodular functions stays submodular.

**Data and base model.** MovieLens-20M, keeping only ratings $\geq 4$ stars as implicit "plays". After dropping films with no genre tags: ~10M plays, 21k films, 140k users. 99%/1% train/test split (~100k test plays), matching the Netflix Prize ratio. Base recommender: weighted 50-dimensional matrix factorisation, L2 regularisation and missing-play weight tuned to maximise recall@50. No prior $p_0$ and no rank weights $w$ were used in the reported runs, for simplicity.

## Ablation Studies and Experiments

The one sweep that matters is $\lambda$ (Table 2, averaged over test users):

| calibration | recall@10 | recall@50 | $C_{KL}$@10 | $C_{KL}$@50 |
|---|---|---|---|---|
| none ($\lambda=0$) | 0.209 | 0.464 | 0.677 | 0.185 |
| $\lambda=0.2$ | 0.209 | 0.464 | 0.465 | 0.171 |
| $\lambda=0.5$ | 0.199 | 0.464 | 0.274 | 0.141 |
| $\lambda=0.8$ | 0.170 | 0.463 | 0.128 | 0.092 |
| $\lambda=0.9$ | 0.146 | 0.460 | 0.084 | 0.061 |
| $\lambda=0.95$ | 0.121 | 0.453 | 0.065 | 0.037 |
| $\lambda=0.99$ | 0.090 | 0.417 | 0.054 | 0.009 |
| $\lambda=0.999$ | 0.082 | 0.339 | 0.054 | 0.005 |

Read this carefully, because the two columns behave very differently.

- **recall@50 is nearly free until $\lambda \approx 0.95$.** From 0.464 to 0.453 while $C_{KL}$@50 falls 5× (0.185 → 0.037). Calibrating a long list costs almost nothing.
- **recall@10 is expensive.** At $\lambda=0.5$ you have already lost 5% relative; at $\lambda=0.99$ recall@10 has more than halved (0.209 → 0.090). Ten slots is a tight budget and every genre you insert evicts a high-scoring film.
- $C_{KL}$@10 improves fast at small $\lambda$ then plateaus at 0.054; $C_{KL}$@50 needs large $\lambda$ to move. Short lists are structurally harder to calibrate — with 10 slots and 19 genres, you cannot represent a 3% interest at all.

The practical suggestion that falls out: **ramp $\lambda$ from small to large as you walk down the greedy list.** Top slots stay close to the original ranking; lower slots progressively cover the neglected interests.

**Per-user spread (Figure 1).** The histogram of $C_{KL}$@50 at $\lambda=0$ is wide. Different users get wildly different calibration quality from the same model. An average metric hides this — this is a per-user pathology, not a global one.

**What crowding-out looks like (Figures 2, 3).** For a user in the worst-calibrated 10%: Drama and Adventure are over-represented, while Musical, Mystery, Thriller and Western are *essentially absent* from the top-50 despite being in the play history. Aggregated over that worst 10%: users played ~3% documentaries; the baseline top-50 gave them essentially zero. IMAX, Musical, Western likewise flattened to zero. Action, Adventure, Crime, Mystery, Sci-Fi over-represented. After re-ranking with $\lambda=0.99$ the gaps shrink across the board; Drama takes the hit, ending slightly under-represented, which is exactly the space that the small genres needed.

**What does not work — competing metrics.** Table 1 is a small but decisive ablation of the metric itself, not the algorithm.

- **DP** [9], the proportionality metric from search-result diversification, is a modified squared difference. With target 60/40 and $N=10$, it gives the same score DP=1 to a 5:5 split and a 7:3 split. Both are one film off, but 5:5 should be better (property 3). It also gives DP=1 regardless of how extreme the target is, violating property 2.
- **BinomDiv** [26] satisfies properties 2 and 3 but fails property 1: at perfect calibration it does not take a constant value. Target 70/30 with $N=10$ gives $2.04\times10^{-4}$; with $N=100$ the same perfect 70:30 gives $7.91\times10^{-35}$. So the number is meaningless on its own, and since each user has a different genre distribution you **cannot average it across users**. It also underflows numerically past a couple of hundred recommended items — a real problem for the Netflix homepage, where lists are that long.
- KL passes all three and is the only one that averages sensibly.

**Not tested.** No comparison against the DP or BinomDiv *algorithms* on their own metrics — the author explicitly declines, noting that each diversity method wins on the metric it was designed for. No online A/B test. No baseline other than the 50-dim MF model.

## Worth Remembering

- **Accuracy loss is not a bug of the method, it is the definition.** Section 2.1 says it plainly: making recommendations balanced is *expected* to reduce accuracy. The offline recall metric literally rewards crowding out. So if you evaluate a calibration method with recall, you have set it up to fail. This is a case where the offline metric and the product goal point in different directions — same flavour of problem as in [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] and [[Counterfactual Reasoning and Learning Systems]].
- **The problem is not restricted to explicit genre tags.** LDA topics or learned embeddings have the same failure mode, because the argmax-over-a-mixture argument does not care whether the categories are labelled or latent. It also does not care whether films belong to one genre or several.
- **Calibration ≠ diversity.** Diversity means "items in the list are not similar to each other". In a two-genre world the maximally diverse list is 50/50, not 70/30. Diversity may also add genres the user never touched. Neither of these gets you to "reflect the user's interests in proportion". You can get calibration out of a diversity method only by tuning the trade-off per user — which is exactly what nobody does.
- **The account-sharing case.** If two people share a Netflix profile, the less active one's taste is a minority interest and gets crowded out by the same mechanism. Calibration protects them without needing to detect the split.
- **Practical caveats.** You need $p(g\mid i)$ from somewhere; MovieLens has tags, most catalogues do not, and uniform splitting over tags is a crude choice that the paper does not test alternatives to. $\alpha=0.01$ smoothing is arbitrary. The re-rank runs at serving time over the candidate pool, so pool size bounds how much calibration you can actually achieve — you cannot recommend a Western if no Western made the candidate cut.
- **Limitations the author names.** Everything is user-centric; the item-centric view (if an item is shown twice as often, is it consumed twice as often?) is left as future work. Calibration is a list property, so folding it into a listwise learning-to-rank objective is the natural next step rather than post-processing. And the whole evaluation lives in the standard offline train/test setup, which the paper itself has just spent Section 2 criticising.
- **Open question.** The greedy prefix guarantee is nice, but the $\lambda$-ramping trick is suggested and never evaluated. What schedule? Does it break the submodularity argument, since the objective now changes between greedy steps? (It probably does.)

## Links

Related: [[KL Divergence]] · [[Cross Entropy]] · [[NDCG]] · [[Recommender Systems - Evolution]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Counterfactual Reasoning and Learning Systems]] · [[Entire Space Multi-Task Model (ESMM)]] · [[Uncertainty]] · [[Beliefs]] · [[Foundational_RecSys_Ranking_Reading_Plan]]

New topics worth writing: Submodular functions and the (1-1/e) greedy guarantee, Maximum Marginal Relevance, Latent Dirichlet Allocation, Determinantal Point Processes for diversity, Fairness criteria (equalised odds vs calibration) and the Kleinberg impossibility result, Hellinger distance and f-divergences, Filter bubbles and feedback loops in recommenders, Implicit feedback matrix factorisation (Hu-Koren-Volinsky), Diversity metrics in recommender systems
