---
title: "Neural Collaborative Filtering"
authors: ["He et al."]
year: 2017
arxiv: "1708.05031"
url: https://arxiv.org/abs/1708.05031
priority: Good-To-Read
read_on: 2026-09-26
tags: [paper]
---
## The Core Idea

Matrix factorisation gives every user a vector $\mathbf{p}_u$ and every item a vector $\mathbf{q}_i$, then scores the pair with a dot product: $\hat{y}_{ui} = \mathbf{p}_u^T \mathbf{q}_i$. That dot product is a **fixed, hand-chosen rule**. It multiplies matched dimensions and adds them with equal weight. Nobody ever checked whether that is the right rule — it was inherited from the Netflix Prize era and never questioned.

The claim here: replace the dot product with a neural network, and let the data decide how a user vector and an item vector should combine.

Why it matters is easiest to see in the paper's own counterexample (Figure 1). Take four users and measure their true similarity with the Jaccard coefficient (overlap of items they touched, divided by union). The first three users can be placed in a 2-D latent space so that all their pairwise similarities come out right. Now add a fourth user $u_4$ whose true ordering is $s_{41}(0.6) > s_{43}(0.4) > s_{42}(0.2)$. To get $\mathbf{p}_4$ close to $\mathbf{p}_1$, you have to put it on one side or the other — and either choice makes $\mathbf{p}_4$ closer to $\mathbf{p}_2$ than to $\mathbf{p}_3$. The ordering is impossible to satisfy. Not because the model is undertrained, but because the geometry of a low-dimensional inner product cannot hold it.

You could fix this by cranking $K$ up. More dimensions, more room. But in sparse data that overfits — see [[Curse of Dimensionality]] and [[Regularization#^overfit-def|overfitting]]. So the paper takes the other road: keep $K$ small, make the **combining function** richer.

> [!NOTE] Interaction function
> The part of a recommender that turns a user vector and an item vector into one score. In MF it is a dot product. In NCF it is a learned [[Deep Learning#^mlp|MLP]]. Everything in this paper is about that one function. ^interaction-function

The second, quieter contribution is the **loss**. Implicit feedback (clicks, watches, pins) is binary: 1 if it happened, 0 if it did not. Previous work squashed this into a squared-error regression, which implicitly assumes Gaussian noise on a 0/1 label. Instead: treat it as binary classification and use the log loss. This alone — same MF model, different objective — beats [[BPR- Bayesian Personalized Ranking from Implicit Feedback|BPR]].

What it unlocks: a generic template. Two embedding towers at the bottom, any neural block you like in the middle, one score at the top. MF becomes one special case of it.

## The Methodology

### The framework

The input is two one-hot vectors, $\mathbf{v}^U_u$ and $\mathbf{v}^I_i$ — user ID and item ID, nothing else. No content features, no context. Pure collaborative filtering.

An [[Embeddings|embedding]] layer turns each one-hot into a dense vector. For a one-hot input, a dense layer is just a row lookup, so $\mathbf{p}_u = \mathbf{P}^T\mathbf{v}^U_u$ picks row $u$ out of the user embedding table $\mathbf{P} \in \mathbb{R}^{M \times K}$. Same for items with $\mathbf{Q}$.

Then a stack of layers, and one scalar out:

$$\hat{y}_{ui} = \phi_{out}(\phi_X(\dots\phi_2(\phi_1(\mathbf{p}_u, \mathbf{q}_i))\dots))$$

The last hidden layer's width is called the **predictive factors** — the paper's name for model capacity, chosen so NCF and MF can be compared at matched size.

### The loss

Take the output through a sigmoid so $\hat{y}_{ui} \in [0,1]$, read it as "probability item $i$ is relevant to user $u$", and write the likelihood:

$$p(\mathcal{Y}, \mathcal{Y}^- \mid \Theta) = \prod_{(u,i) \in \mathcal{Y}} \hat{y}_{ui} \prod_{(u,j) \in \mathcal{Y}^-} (1 - \hat{y}_{uj})$$

Negative log of that:

$$L = -\sum_{(u,i) \in \mathcal{Y} \cup \mathcal{Y}^-} y_{ui}\log\hat{y}_{ui} + (1-y_{ui})\log(1-\hat{y}_{ui})$$

Which is exactly binary [[Cross Entropy|cross-entropy]] / log loss. The authors point this out themselves — no new mathematics, just the right standard loss for a binary target.

$\mathcal{Y}^-$ is built by uniformly sampling unobserved $(u,j)$ pairs each epoch. The sampling ratio (negatives per positive) is a free knob, and this is the practical advantage over pairwise losses: [[BPR- Bayesian Personalized Ranking from Implicit Feedback|BPR]]'s triple structure pins you to exactly one negative per positive. Here you can pick any number.

### Three models

**GMF (Generalised MF).** First layer is the element-wise product, then a learned linear read-out and a sigmoid:

$$\hat{y}_{ui} = \sigma\big(\mathbf{h}^T(\mathbf{p}_u \odot \mathbf{q}_i)\big)$$

Set $\mathbf{h} = \mathbf{1}$ and $a_{out} = $ identity and you have plain MF back, exactly. Letting $\mathbf{h}$ be learned means latent dimensions can carry different importance; the sigmoid makes it non-linear. This is the "MF is a special case" argument, and it is a clean one.

**MLP.** Concatenate instead of multiplying, then push through hidden layers:

$$\mathbf{z}_1 = \begin{bmatrix}\mathbf{p}_u \\ \mathbf{q}_i\end{bmatrix}, \quad \phi_l(\mathbf{z}_{l-1}) = a_l(\mathbf{W}_l^T\mathbf{z}_{l-1} + \mathbf{b}_l), \quad \hat{y}_{ui} = \sigma(\mathbf{h}^T\phi_L)$$

Concatenation on its own models no interaction at all — it just stacks two vectors side by side. The hidden layers are what create the interaction. ReLU throughout (see [[ImageNet Classification with Deep CNNs (AlexNet)#^relu|ReLU]]); the paper's reasoning is that sigmoid saturates, tanh is a rescaled sigmoid ($\tanh(x/2) = 2\sigma(x) - 1$) so only half-fixes it, and ReLU is non-saturating and gives sparse activations, which suits sparse data.

Tower structure: each layer half the width of the one below. With 8 predictive factors, the stack is $32 \to 16 \to 8$, so embedding size 16 per tower.

**NeuMF (the fusion).** GMF's element-wise product is a *linear* kernel on the latent dimensions; MLP's stack is a non-linear one. Run both and concatenate their last hidden layers:

$$\phi^{GMF} = \mathbf{p}_u^G \odot \mathbf{q}_i^G, \qquad \phi^{MLP} = a_L(\mathbf{W}_L^T(\dots) + \mathbf{b}_L), \qquad \hat{y}_{ui} = \sigma\!\left(\mathbf{h}^T\begin{bmatrix}\phi^{GMF} \\ \phi^{MLP}\end{bmatrix}\right)$$

The design decision worth noticing: **separate embedding tables for the two branches** ($\mathbf{p}_u^G$ and $\mathbf{p}_u^M$ are different vectors). Sharing one table is simpler and is what the Neural Tensor Network did, but it forces both branches to use the same embedding size — and the two branches may want very different sizes.

### Pre-training

NeuMF's objective is non-convex, so initialisation matters. The recipe: train GMF and MLP separately to convergence with [[Adam- A Method for Stochastic Optimization|Adam]], then load their weights into the matching halves of NeuMF. The output vector is stitched with a mixing weight:

$$\mathbf{h} \leftarrow \begin{bmatrix}\alpha \mathbf{h}^{GMF} \\ (1-\alpha)\mathbf{h}^{MLP}\end{bmatrix}, \qquad \alpha = 0.5$$

Then fine-tune with **vanilla SGD, not Adam**. The stated reason: Adam needs its [[Momentum|momentum]] state to behave, and they carried over only the weights, not the moment estimates. A real engineering detail, easy to get wrong.

### Data and protocol

| | MovieLens 1M | Pinterest |
|---|---|---|
| Interactions | 1,000,209 | 1,500,809 |
| Items | 3,706 | 9,916 |
| Users | 6,040 | 55,187 |
| Sparsity | 95.53% | 99.73% |

MovieLens is explicit ratings binarised to "did they rate it at all". Pinterest is filtered to users with $\geq$ 20 pins.

Evaluation: **leave-one-out**. Hold out each user's most recent interaction, plus **99 randomly sampled items the user never touched**, and rank the true item among those 100. Report HR@10 (was it in the top 10?) and [[NDCG|NDCG@10]] (how high?).

Keras implementation. 4 negatives per positive by default. Gaussian init ($\mu = 0$, $\sigma = 0.01$). Batch sizes swept over $\{128, 256, 512, 1024\}$, learning rates over $\{0.0001, 0.0005, 0.001, 0.005\}$. Predictive factors $\{8, 16, 32, 64\}$. Three MLP hidden layers unless stated.

## Ablation Studies and Experiments

### Main comparison

Baselines: ItemPop (rank by popularity, no personalisation), ItemKNN (item-based CF), [[BPR- Bayesian Personalized Ranking from Implicit Feedback|BPR]], and eALS (weighted squared loss, all unobserved treated as negatives, weighted by item popularity).

NeuMF wins on both datasets at every factor size. Average relative improvement: **+4.5% over eALS, +4.9% over BPR**. On Pinterest, NeuMF at 8 factors beats eALS and BPR at 64 factors — the whole argument, in one sentence. Paired t-tests: $p < 0.01$ at every cut-off $K$ from 1 to 10.

Ordering of the rest: model-based beats ItemKNN, and everything beats ItemPop by a wide margin.

**The cleanest ablation in the paper**: GMF beats BPR consistently. Same MF model, same capacity — only the objective differs (log loss with sampled negatives vs pairwise ranking loss). So a measurable share of the headline win is the loss function, not the architecture.

### Does depth help?

HR@10 for MLP with varying hidden layers:

| Factors | MLP-0 | MLP-1 | MLP-2 | MLP-3 | MLP-4 |
|---|---|---|---|---|---|
| MovieLens, 8 | 0.452 | 0.628 | 0.655 | 0.671 | 0.678 |
| MovieLens, 64 | 0.453 | 0.687 | 0.696 | 0.702 | 0.707 |
| Pinterest, 8 | 0.275 | 0.848 | 0.855 | 0.859 | 0.862 |
| Pinterest, 64 | 0.274 | 0.864 | 0.867 | 0.869 | 0.873 |

Two things fall out.

**MLP-0 is catastrophic.** 0.452 HR@10 on MovieLens — no better than ItemPop. MLP-0 means embeddings concatenated and projected straight to a score, with no hidden layer. This is the paper's own prediction confirmed: concatenation models *zero* interaction. The hidden layers are not a refinement, they are the mechanism.

**Depth gains are real but small.** MLP-1 → MLP-4 is about +0.05 HR@10 on MovieLens, +0.014 on Pinterest. Positive and monotone, but the first hidden layer does nearly all of the work.

**What did not work:** stacking *linear* layers (identity activation) instead of ReLU gave "much worse" performance. The depth benefit comes from non-linearity, not from parameter count. The paper reports this in prose without a table — worth flagging.

### Negative sampling ratio

Sweeping negatives per positive (factors = 16): **one negative is not enough**. Optimal ratio is **3 to 6**. On Pinterest, past 7 negatives performance *drops*. GMF at ratio 1 is roughly level with BPR; GMF at higher ratios clearly beats it. So the pointwise loss's advantage is precisely the freedom to sample more than one negative.

### Pre-training

| | With pre-train HR@10 | Without |
|---|---|---|
| MovieLens, 8 | 0.684 | **0.688** |
| MovieLens, 16 | 0.707 | 0.696 |
| MovieLens, 32 | 0.726 | 0.701 |
| MovieLens, 64 | 0.730 | 0.705 |
| Pinterest, 8 | 0.878 | 0.869 |
| Pinterest, 64 | 0.877 | 0.872 |

Average gain: +2.2% MovieLens, +1.1% Pinterest. Note the failure at MovieLens factors = 8 — pre-training was slightly *worse* there. Small effect overall, and worth knowing it costs you two extra training runs.

### Overfitting

Training loss for all three NCF models drops monotonically, but recommendation quality peaks around **iteration 10** and then degrades. NeuMF has the lowest training loss and also the best HR/NDCG — but only if you stop early. Also: GMF at high factor counts overfits and loses to GMF at lower factor counts.

### Baselines against each other

eALS beats BPR on MovieLens (+5.1%) but *loses* to BPR on Pinterest NDCG. So neither baseline dominates, and which one you would have picked depends on the dataset.

## Worth Remembering

**The 99-sampled-negatives protocol is the paper's biggest vulnerability.** Ranking against 100 items instead of the full catalogue was standard in 2017. [[On Sampled Metrics for Item Recommendation (KDD)|Krichene and Rendle (2020)]] later showed that sampled metrics are *inconsistent* — they do not preserve the ordering you would get from the full ranking, so a sampled comparison can reverse the true winner. Every number above is measured with an instrument now known to be unreliable.

**And the headline result did not survive replication.** [[Neural Collaborative Filtering vs. Matrix Factorization Revisited|Rendle et al. (2020)]] re-ran this on the same data and found that a properly tuned dot-product MF **matches or beats NeuMF** — and that learned similarity is hard to train well. The mechanism they point at is [[Neural Collaborative Filtering vs. Matrix Factorization Revisited#^inductive-bias-over-capacity|inductive bias beating capacity]]: the dot product is a constraint that suits the problem, and an MLP has to learn even simple multiplication from scratch. See also [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|Dacrema et al.]] on weak baselines and [[On the Difficulty of Evaluating Baselines#^baseline-propagation|baseline propagation]].

So read this note as two separate things. The **framework** — two embedding towers, a learned head, log loss, sampled negatives — became the standard shape for neural recommenders, and that has held up. The **specific empirical claim** that MLP-based interaction beats the dot product has not.

**What almost certainly is real:** the log-loss result. GMF beats BPR on the same model with a different objective, and this is a within-paper controlled comparison that does not depend on the architecture claim. Binary cross-entropy with 3–6 sampled negatives is a sensible default for implicit feedback.

**Limitations the authors state:** pairwise learning was left as future work (they only do pointwise). Non-uniform negative sampling (popularity-biased) was left as future work too — worth noting, given that [[Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys)|logQ correction]] and [[Approximate Nearest Neighbor Negative Contrastive Learning (ANCE)|hard negatives]] later turned out to matter a great deal.

**The cold-start opening they flag and never use:** because the input layer takes an arbitrary feature vector, swapping one-hot IDs for content features is a trivial change. That door is the one [[Wide & Deep Learning for Recommender Systems|Wide & Deep]] and [[Deep Learning Recommendation Model (DLRM)|DLRM]] walked through.

**Practical caveats if you ever build this:**
- Early-stop on validation HR, not training loss. The curves diverge by iteration 10.
- Separate embeddings per branch matters. Sharing them constrains both branches to one size.
- If you pre-train, switch to plain SGD for the fusion step, or carry the optimiser moments across too.
- Do not evaluate against 100 sampled items. Rank the full catalogue, or use a metric known to be sampling-consistent.

**Open question:** is there a middle ground between the dot product and a free-form MLP — something with the dot product's inductive bias but more expressive? [[Deep & Cross Network for Ad Click Predictions|DCN]] and [[AutoInt- Automatic Feature Interaction Learning via Self-Attention|AutoInt]] are two answers; both keep an explicit multiplicative structure rather than hoping an MLP discovers one.

## Links

Related: [[Neural Collaborative Filtering vs. Matrix Factorization Revisited]] · [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)]] · [[BPR- Bayesian Personalized Ranking from Implicit Feedback]] · [[Collaborative Filtering for Implicit Feedback Datasets (ICDM)]] · [[On Sampled Metrics for Item Recommendation (KDD)]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Cross Entropy]] · [[Embeddings]] · [[NDCG]] · [[Wide & Deep Learning for Recommender Systems]] · [[Factorization Machines (ICDM)]] · [[Recommender Systems - Evolution]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[Adam- A Method for Stochastic Optimization]] · [[Deep Learning]] · [[Regularization]]

New topics worth writing: Jaccard coefficient as a similarity target, leave-one-out evaluation protocols for recsys, pointwise vs pairwise vs listwise losses compared directly, eALS and weighted ALS, Neural Tensor Network, expressiveness of the dot product vs learned similarity functions
