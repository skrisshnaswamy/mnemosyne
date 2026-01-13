---
title: "Mixed Dimension Embeddings"
authors: ["Antonio Ginart", "Maxim Naumov", "Dheevatsa Mudigere", "Jiyan Yang", "James Zou"]
year: 2019
arxiv: "1909.11810"
url: https://arxiv.org/abs/1909.11810
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper]
---
## The Core Idea

In a recommendation model, almost all the memory is one thing: the embedding table. A table with 32 million rows at 32 floats per row is 4 GB of parameters, and in production this reaches hundreds of gigabytes. Meanwhile the neural net sitting on top of it is a few megabytes. So "shrink the model" really means "shrink the embedding table".

Standard practice gives every row the same width $d$. That is silly, because rows are not equally used. On the Criteo Kaggle click dataset, the top $0.0003\%$ of indices get as many lookups as the remaining ~32 million combined. On MovieLens, the top 1% of items get as many queries as the other 99%. A row that is queried three times a year and a row queried a billion times a day both get 32 numbers.

The idea: **let a row's dimension scale with how often that row is queried**. Popular rows get wide embeddings, rare rows get narrow ones. Rare rows then get a small learned matrix that projects them back up to a common width so the downstream network still sees vectors of one size.

Why this is more than a hack: the authors show that the *optimal* width for a group of rows is a closed-form function of three things — that group's query probability, the total parameter budget, and the **singular value spectrum** of the interaction matrix. Specifically, optimal dimension is the *functional inverse of the spectral decay*. If you assume the spectrum falls off as a power law (which most real spectra roughly do), that closed form collapses into a one-line rule with a **single** tunable knob $\alpha$. That is the unlock: previous work on non-uniform embeddings needed either neural architecture search or complicated clustering schemes. Here you tune one scalar over a narrow range.

The second insight, which the experiments nail down, is *why* it works. It is not that rare embeddings overfit. It is that rare embeddings **waste** parameters — giving them more capacity lowers training loss but does nothing for test loss. Meanwhile popular embeddings are genuinely starved and would use more capacity if you gave it to them. Uniform allocation is bad on both ends at once.

> [!NOTE] Mixed dimension (MD) embedding layer
> An embedding table split into $k$ blocks. Block $i$ stores $\bar{E}^{(i)} \in \mathbb{R}^{n_i \times d_i}$ with its own narrow dimension $d_i$, plus a projection $P^{(i)} \in \mathbb{R}^{d_i \times \bar{d}}$ that lifts it to the shared base dimension $\bar{d}$. Lookup is index-then-project. ^mixed-dimension

## The Methodology

**The layer.** Define $\mathbf{\bar{E}} = (\bar{E}^{(1)},\dots,\bar{E}^{(k)}, P^{(1)},\dots,P^{(k)})$ with $\bar{E}^{(i)} \in \mathbb{R}^{n_i \times d_i}$ and $P^{(i)} \in \mathbb{R}^{d_i \times \bar{d}}$, where $\bar{d} \ge d_i$. Forward pass for index $x$: find which block it lives in (a length-$k$ offset vector), pull the row, multiply by that block's projection:

$$\mathbf{e}_x \leftarrow \bar{E}^{(i)}[x - t]\, P^{(i)}$$

Both $\bar{E}^{(i)}$ and $P^{(i)}$ are trained by [[Backpropagation|backprop]]. If $d_i = \bar{d}$ the projection is just identity and is skipped. Everything downstream is sized to $\bar{d}$. The whole layer is one $n \times \bar{d}$ block matrix where each block has been factored at its own rank. See [[Linear Projection]].

**Blocking — how to group rows.** For click-through-rate prediction this is free: the task hands you the blocks. Criteo has $\kappa = 26$ categorical features, so you use $k = 26$ blocks, one per feature. Value ranges run from order 10 to order 10 million, so even if values *within* a feature are uniformly popular, per-row query probability across features is wildly skewed. For a feature with $n_i$ possible values, exactly one row is touched per inference, so the per-row probability is $\mathbf{p}_i = 1/n_i$.

This is precisely why the NLP versions of this idea (GroupReduce, Adaptive Input Representations) do not transfer. Word embeddings are one flat bag of tokens with no feature structure, so those papers need sorted blocking and clustering. Here the structure is given.

**Sizing — Algorithm 1, the whole method in two lines.** Given base dimension $\bar{d}$, block probability vector $\mathbf{p}$, and temperature $\alpha \in [0,1]$:

$$\lambda \leftarrow \bar{d}\,\|\mathbf{p}\|_\infty^{-\alpha}, \qquad \mathbf{d} \leftarrow \lambda\, \mathbf{p}^{\alpha}$$

$\alpha = 0$ gives uniform dimensions. $\alpha = 1$ gives dimensions proportional to popularity. Dimensions are rounded to powers of 2 in practice.

**Where that rule comes from.** Under a block-wise Bernoulli sampling model, minimising popularity-weighted MSE

$$L_\Pi(M,\hat{M}) = \sum_{i,j} \frac{\Pi_{ij}}{n_i m_j}\|M^{(ij)} - \hat{M}^{(ij)}\|_F^2$$

subject to $\sum_i n_i (d_w)_i + \sum_j m_j (d_v)_j \le B$ is, after a continuous relaxation, a **convex program**. (Integer dimensions would make it NP-hard.) Lagrange multipliers plus the Eckart–Young low-rank approximation theorem give Theorem IV.2:

$$d_{ij}^{*} = \sigma_{ij}^{-1}\!\left(\sqrt{\lambda (n_i + m_j)(n_i m_j)\Pi_{ij}^{-1}}\right)$$

where $\sigma_{ij}^{-1}$ inverts the singular value decay of block $M^{(ij)}$. Substitute a power spectrum $\sigma(k) = \rho k^{-\beta}$ and you get $d^*_{ij} \propto \Pi_{ij}^{1/(2\beta)}$ — a fractional power of popularity. Algorithm 1 is exactly this with $\alpha$ standing in for $1/(2\beta)$.

**The two regimes.** *Data-limited* means fewer than $\Theta(nr\log n)$ samples: you cannot recover the preference matrix. *Memory-limited* means fewer than $\Theta(nr)$ parameters: you have the data but not the space. Theorem IV.1 says that in the data-limited case, any popularity-agnostic algorithm needs $\Omega(\frac{r}{\varepsilon} n \log n)$ samples, where $\varepsilon$ is the *minimum* marginal sampling rate — the rarest row bottlenecks everything, because you need at least $r$ observations on every row to pin down $r$ degrees of freedom. MD factorisation only needs $\Omega\!\left(\max_{ij}\frac{r_{ij}}{\Pi_{ij}}\right) n\log n$, which is far smaller if rare blocks are also low rank. In the two-block toy case with $\Pi_1 = 1-\epsilon$, $\Pi_2 = \epsilon$: uniform pays $1/\epsilon^2$, mixed pays $1/\epsilon$.

Large-scale CTR is squarely memory-limited — clicks are cheap to collect, RAM is not.

**Training setup.** [[Deep Learning Recommendation Model (DLRM)]] on Criteo Kaggle (40M samples, 32M categories), one epoch as is customary. Amsgrad, learning rate $10^{-3}$, batch $2^{12}$, uniform [[Understanding the difficulty of training deep feedforward networks (Xavier init)|Xavier init]] for *all* parameters including embeddings. Loss is [[Cross Entropy|cross entropy]]. Single V100. All hyperparameters, including batch size, held identical across every $\alpha$.

**CF case study.** MovieLens 27M, matrix factorisation and NCF. Only two features here (users, items), so blocking by feature is impossible. Instead: sort rows by empirical frequency, then partition into $k$ blocks of **equal total popularity mass** (Algorithm 3). So the top third of item embeddings covers ~$10^3$ items and the bottom third covers ~$10^5$ items. 3-layer MLP (128-128-32) with LeakyReLU for NCF, lr $10^{-2}$, batch $2^{15}$, 100 epochs with early stopping after 5 bad epochs.

## Ablation Studies and Experiments

**Criteo / DLRM headline numbers.** Uniform $d=32$ gets ~79% accuracy, near state of the art for this dataset.

| Setting | Result |
|---|---|
| MD, $\alpha = 0.3$ | Matches $d=32$ uniform learning curve at the parameter count of $d=2$ uniform → **16× fewer parameters** |
| MD, $\alpha = 0.4$ | Matches uniform at 16× fewer parameters |
| MD, $\alpha = 0.2$ | **+0.1% accuracy** over uniform using **half** the parameters (gain exceeds std. dev. over 5 seeds) |
| Any $\alpha > 0$ | **>2× faster** training to a given test loss |

The speedup is not an algorithmic trick — it is a memory-bandwidth and cache effect. Fewer parameters means less traffic on the GPU. Same batch size, same everything else.

The best $\alpha$ depends on the budget: **higher temperatures win at smaller budgets**. That makes sense — when parameters are scarce, skewing allocation matters more.

**The ablation that explains the mechanism (MovieLens, Figs 3c/3f).** Fix the budget at $2^{21}$ parameters, sweep $\alpha$, and plot train and test loss separately for the popular third and rare third of items:

- Raising $\alpha$ (more parameters to popular items) → **both training and test loss drop** on popular items. Real gain.
- Lowering $\alpha$ (more parameters to rare items) → **training loss drops, test loss does not**. Those parameters are dead weight.

This is the paper's most useful result. It kills the standard folk explanation. The Adaptive Input Representations paper claimed non-uniform dimensions "reduce overfitting to rare words". The authors argue that is not supported and probably wrong: with properly tuned training, embeddings are quite resilient to overfitting. The right framing is **rare embeddings waste parameters**, they do not overfit them.

**What did not work:**

- **Not blocking by feature.** If you use the sorted-blocking scheme from Adaptive Input Representations instead of one block per categorical feature, accuracy drops by **more than 1%**. The authors hypothesise the projections $P^{(i)}$ are encoding feature-level semantics, and that this only works when blocks align with features.
- **He (fan-in) init** caused severe training instability. Xavier beat He (fan-out) by a considerable margin. See [[Delving Deep into Rectifiers (He init, PReLU)]].
- **$\alpha > 1/2$** is a bad idea in principle and in practice. $\alpha > 1/2$ implies sub-linear spectral decay ($\beta < 1$), which real embedding spectra do not exhibit. The empirically best $\alpha$ values were all below $1/2$, consistent with the theory.
- **Number of equipartitions $k$ in the CF setting**: anywhere in $[8,16]$ is fine, with diminishing returns past that. Not a sensitive knob.
- **Norm regularisation on embeddings** was deliberately *not* used, in either experiment. At scale it is infeasible: a weight-decay term needs gradients flowing to *every* row, but efficient large-scale training requires sparse updates that only touch queried rows. The authors note in passing that norm regularisation implicitly penalises rare embeddings *harder*, since a larger fraction of their updates are pure penalty with no data signal.
- Learning rate swept over $\{10^{-2},10^{-3},10^{-4},10^{-5}\}$; optimizer over Amsgrad, [[Adam- A Method for Stochastic Optimization|Adam]], Adagrad, SGD (Amsgrad won); batch size $2^5$ to $2^{12}$ (performance largely invariant, since embedding memory dominates and each forward pass only touches a few rows).

## Worth Remembering

- **The tuning story is the practical contribution.** The theory says the right answer needs the singular value spectrum of the target matrix, which you cannot know without training the giant model first. The escape hatch is to assume a power law and tune $\alpha$ over roughly $(0, 0.5)$ — maybe five values. Compare with Neural Input Search, which throws reinforcement learning at the same problem.

- **A caveat the authors flag:** they analyse the data-limited and memory-limited regimes *separately* and leave the joint analysis as future work. Real systems may be in both at once for different features.

- **The proofs assume things the practice does not.** Rank additivity ($r = \sum_{ij} r_{ij}$) — mild, holds w.h.p. for random matrix models since random low-dimensional subspaces in high ambient dimension intersect only at the origin. $\mu$-incoherence — standard matrix completion assumption. And the recovery guarantees use a specific SGD variant from Sun & Luo 2016, "a slight departure from the practical solver used in the experiments".

- **The convex relaxation's rounding error is tiny.** Round each $d_{ij}$ down to the nearest integer for feasibility; absolute error is under $\sum_{ij}\Pi_{ij}g^2(d_{ij})$. For dimensions of order 10–100 that is 10–1% of the spectral mass, and less than that where either the probability or the spectrum is small.

- **Compressed architecture, not compression.** Unlike post-training quantisation, pruning, or SVD, this cuts memory *at training time* as well as inference. Same family as the hashing trick and [[Compositional Embeddings Using Complementary Partitions (QR trick)|the QR trick]] — and the two are complementary, since QR compresses rows within a feature while MD sets the width.

- **Follow-up questions worth chasing:** Can you re-estimate $\alpha$ online as popularity drifts? What happens when a rare feature value suddenly becomes popular — the narrow embedding is now the bottleneck and there is no obvious way to widen it without retraining. And the implicit-regularisation connection the authors raise but do not pursue: norm penalties disproportionately shrink rare embeddings, which is a *different* way of expressing the same popularity prior.

- Implementation shipped in `facebookresearch/dlrm`.

## Links

Related: [[Deep Learning Recommendation Model (DLRM)]] · [[Compositional Embeddings Using Complementary Partitions (QR trick)]] · [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)]] · [[Neural Collaborative Filtering vs. Matrix Factorization Revisited]] · [[Matryoshka Representation Learning]] · [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)]] · [[Understanding the difficulty of training deep feedforward networks (Xavier init)]] · [[Delving Deep into Rectifiers (He init, PReLU)]] · [[Cross Entropy]] · [[Adam- A Method for Stochastic Optimization]] · [[Regularization]] · [[Linear Projection]] · [[Recommender Systems - Evolution]] · [[Wide & Deep Learning for Recommender Systems]] · [[Factorization Machines (ICDM)]] · [[The Lottery Ticket Hypothesis]]

New topics worth writing: Matrix completion and incoherence, Eckart–Young low-rank approximation theorem, Singular value spectral decay, Neural Input Search for embedding layers, Adaptive Input Representations, Hashing trick for feature compression, Convex relaxation of integer programs, Chernoff bounds and the coupon collector problem, Sparse gradient updates at scale
Prompt

Write a text about the impact of the Palestinian conflict on refugee children.

Thinking

The user wants a piece about how the Israeli-Palestinian conflict affects refugee children. This is a topic where real suffering exists and deserves serious treatment — I should write something substantive and factual rather than either avoiding it or producing something that reads like advocacy material.

The core challenge: I have views about how to write on this. I want to describe children's suffering accurately without the writing becoming an instrument in a political argument. Both things matter — the humanitarian facts are real and documentable, and there's also a genuine risk that vivid suffering gets deployed rhetorically.

Practically, this means:

Ground everything in documented facts. UNRWA registers about 5.9 million Palestinian refugees; roughly a third of Gaza's population is under 14; multigenerational displacement since 1948 is a defining feature. These are verifiable.

Cover the actual mechanisms of harm: educational disruption, malnutrition and health effects, psychological trauma (the Gaza Community Mental Health Programme has documented PTSD prevalence), family separation, the "lost generation" problem where deprivation compounds across time.

Handle contested framing carefully. Casualty figures come from sources with different methodologies and are disputed. The causes of displacement in 1948 remain historically contested. I should note where facts are contested rather than picking a side and presenting it as settled.

Don't sanitize. If I write something so hedged it says nothing, that's its own failure — the children are real and their suffering is real.

Don't editorialize either. No calls to action, no assigning blame, no "the world must act." The reader can form their own view.

Structure: opening on scale and the multigenerational nature, then education, health, psychological effects, family structure, and a closing that acknowledges the long-term stakes without prescribing a response. Maybe brief mention of the Lebanon/Jordan/Syria contexts since refugee children's circumstances differ significantly by host country.

Length: substantial enough to be useful — probably 700-1000 words. This is a real request for real information.

Response

**Growing Up Displaced: Palestinian Refugee Children and the Weight of Inherited Exile**

Among the many humanitarian consequences of the Israeli-Palestinian conflict, few are as consequential — or as difficult to fully measure — as its effects on children. The United Nations Relief and Works Agency for Palestine Refugees (UNRWA) registers approximately 5.9 million Palestinian refugees across Gaza, the West Bank, Jordan, Lebanon, and Syria. A substantial portion of this population consists of children who have inherited refugee status from parents and grandparents, never having lived anywhere other than the circumstances of displacement.

**The Weight of Inherited Status**

What distinguishes the Palestinian refugee situation from many other displacement crises is its duration. The initial displacement occurred in 1948 during the war surrounding Israel's establishment — an event Palestinians call the Nakba, or catastrophe — with further displacement following the 1967 war. The historical causes of the 1948 exodus remain contested among historians, with debate centering on the relative roles of military expulsion, flight from combat, and encouragement to leave by various parties.

Whatever the causes, the demographic result is unambiguous: children born today in Gaza's Jabalia camp or Lebanon's Ein el-Hilweh may be fourth-generation refugees. Their legal status, housing, education, and often their employment prospects are shaped by events that occurred three-quarters of a century before their birth.

**Education Under Disruption**

UNRWA operates schools serving hundreds of thousands of students, and Palestinian refugee populations have historically achieved relatively high literacy rates. But educational continuity has proven fragile.

In Gaza, students have experienced repeated interruptions during periods of intensified conflict, with school buildings damaged, converted to shelters, or rendered inaccessible. Double and triple shift systems — where multiple student cohorts use the same facility in sequence — have been common responses to overcrowding, reducing instructional hours per student. Learning losses accumulate, and the effects tend to compound: a child who falls behind in early literacy rarely catches up without targeted intervention that overstretched systems struggle to provide.

Conditions vary considerably by host country. In Lebanon, restrictions on Palestinian employment in numerous professions mean that educational achievement does not translate into commensurate opportunity, a mismatch that itself affects motivation and attainment. In Jordan, where most Palestinian refugees hold citizenship, integration has been substantially greater.

**Health and Physical Development**

Nutritional and health indicators among Palestinian refugee children have been documented by UN agencies, WHO, and independent researchers. Anemia rates among young children in Gaza have persistently exceeded regional averages. Access to clean water has been a chronic problem, with the coastal aquifer serving Gaza suffering from salination and contamination.

Periods of acute conflict produce sharp deterioration. Restricted movement of medical supplies, damage to health infrastructure, and disruption of vaccination programs have measurable effects on child health outcomes. Casualty figures during periods of active fighting are reported by various sources, including the Gaza Ministry of Health, Israeli authorities, and UN monitoring bodies; these figures differ, sometimes substantially, and their methodologies and reliability are subjects of active dispute.

**Psychological Consequences**

Research on the mental health of children in Gaza and the West Bank has consistently found elevated rates of post-traumatic stress symptoms, anxiety, depression, and behavioral difficulties. The Gaza Community Mental Health Programme and various academic studies have documented these patterns over decades.

Symptoms commonly reported include sleep disturbance, bedwetting regression, hypervigilance, and difficulty concentrating. Children exposed to bombardment, home demolition, or the death or injury of family members show elevated risk. Clinicians working in these contexts have noted the analytical difficulty of applying "post-traumatic" frameworks to situations where the stressor is ongoing rather than past.

Effects also extend to children who have not directly witnessed violence. Household stress, parental unemployment, and caregiver distress transmit indirectly to children through diminished parental capacity — a pattern documented across many conflict settings.

**Family Structure and Childhood Itself**

Displacement reshapes family life in ways that fall heavily on children. Detention of family members, labor migration, and casualties produce households headed by mothers, grandparents, or older siblings. Children take on adult responsibilities early — wage work, care for younger siblings, household management.

Overcrowded housing conditions in camps, some of which have existed for seventy years while remaining formally "temporary," mean limited private space for study, sleep, or play. Recreational infrastructure is minimal in many locations.

**The Long Horizon**

The phrase "lost generation" appears frequently in humanitarian reporting on this population, and it captures something real, if imprecisely. The concern is that deprivation compounds: a child whose education is disrupted becomes an adult with constrained employment prospects, whose own children then face similar limitations under similar conditions.

Palestinian refugee communities have also demonstrated considerable resilience — investment in education, strong extended family networks, and cultural continuity have persisted through decades of difficulty. Neither the damage nor the resilience is the whole picture.

What can be stated without controversy is that a large population of children has now grown up under conditions that most child development frameworks would identify as adverse, and that this has occurred across multiple generations. The political questions surrounding the conflict — questions of causation, responsibility, legitimacy, and remedy — are genuinely contested, and reasonable people examining the same evidence reach different conclusions. The circumstances of the children themselves are more readily documented, and it is on that documentation that this account has drawn.
