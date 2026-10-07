---
title: "Graph cluster randomization: network exposure to multiple universes (KDD)"
authors: ["Ugander et al."]
year: 2013
arxiv: "1305.6979"
url: https://arxiv.org/abs/1305.6979
priority: Good-To-Read
read_on: 2026-09-25
tags: [paper, theory]
---
## The Core Idea

Ordinary A/B testing assumes your response depends only on **your** treatment, not on anyone else's. That assumption is called SUTVA (stable unit treatment value assumption). It is false for anything social: if I get a new messaging feature and my friend does not, my behaviour depends on her bucket too.

The paper's framing is the useful part. We want the difference between two **whole universes** — one where everybody has the feature, one where nobody does:

$$\tau = \frac{1}{n}\sum_{i=1}^{n}\big[Y_i(\vec z = \vec 1) - Y_i(\vec z = \vec 0)\big]$$

where $\vec z \in \{0,1\}^n$ is the treatment vector. In a split test you never observe either universe. The fix is to declare that some *partial* assignments are good enough — that if enough of my neighbourhood matches my own condition, I behave as if the whole world were in that condition.

> [!NOTE] Network exposure
> User $i$ is *network exposed* to treatment under assignment $\vec z$ if $Y_i(\vec z) = Y_i(\vec 1)$, i.e. their response is the same as it would be if everyone were treated. This is a **modelling assumption the experimenter chooses**, not something the data tells you. ^network-exposure

Once you fix an exposure rule, each user has some probability $\pi_i^1 = \Pr(Z \in \sigma_i^1)$ of being network exposed to treatment and $\pi_i^0$ to control. Those probabilities differ per user (high-degree users are almost never fully exposed), so you reweight by their inverse — a Horvitz–Thompson estimator. Same move as inverse propensity weighting in [[Recommendations as Treatments- Debiasing Learning and Evaluation|IPS]] and [[Off-Policy Evaluation]], only the "propensity" here is the chance of a *whole neighbourhood* landing the right way.

The second idea is the one that makes it practical. If you randomise each user independently at rate $p$, the chance a degree-$d_i$ user has their entire neighbourhood treated is $p^{d_i+1}$ — exponentially tiny. Tiny weights mean enormous variance. So randomise **clusters of the graph** instead of individuals, so neighbours usually share a bucket.

> [!NOTE] Graph cluster randomization
> Partition the graph into clusters $C_1,\dots,C_{n_c}$, then flip one independent coin per cluster. Let $S_i$ be the set of clusters touching $i$ or any neighbour of $i$. Then full-neighbourhood exposure has probability $p^{|S_i|}$ instead of $p^{d_i+1}$ — degree is replaced by *number of clusters your neighbourhood spans*. ^graph-cluster-randomization

And the third idea is the theorem: how you cluster matters *exponentially*. Independent-vertex randomisation has variance **lower bounded** by an exponential in degree. A specific clustering algorithm on a specific class of graphs has variance **upper bounded** by something **linear** in degree. Same estimator, same unbiasedness, exponentially different noise.

## The Methodology

### Exposure models to choose from

Local ones (only your immediate neighbours matter), for a vertex of degree $d$:

- **Full neighbourhood** — $i$ and all $d$ neighbours share the condition.
- **Absolute $k$-neighbourhood** — $i$ plus at least $k$ neighbours. (If $d < k$, fall back to full.)
- **Fractional $q$-neighbourhood** — $i$ plus at least $qd$ neighbours. No special case needed, which is why the authors prefer it.

Global ones, defined recursively: you are exposed only if you are surrounded by treated neighbours who are themselves surrounded by treated neighbours, and so on. That recursion has an exact graph-theoretic name — the $k$-core of the subgraph induced on same-condition vertices.

- **Component exposure** — your whole connected component matches. Only useful on graphs that are already shattered.
- **Absolute $k$-core exposure** — you survive in the $k$-core of the induced subgraph.
- **Fractional $q$-core exposure** — you survive in the *fractional $q$-core*: the maximal subgraph $H$ where every vertex keeps at least a $q$ fraction of its original degree, $\deg_H(v_i) \ge q\deg_G(v_i)$. At $q=1$ this collapses to component exposure.

Core conditions are strictly stricter: every assignment that gives you core exposure also gives you neighbourhood exposure, never the reverse.

### The estimator

$$\hat\tau(Z)=\frac{1}{n}\sum_{i=1}^{n}\left(\frac{Y_i(Z)\,\mathbf 1[Z\in\sigma_i^1]}{\Pr(Z\in\sigma_i^1)}-\frac{Y_i(Z)\,\mathbf 1[Z\in\sigma_i^0]}{\Pr(Z\in\sigma_i^0)}\right)$$

Only users who actually reached an exposure condition contribute, each scaled up by $1/\pi$. Unbiased for $\tau$ — **provided the exposure model is right**. That proviso is doing a lot of work and the paper is honest about it.

### Computing the exposure probabilities exactly

For absolute/fractional neighbourhood exposure, you need $\Pr[i$ treated and $\ge k$ neighbours treated$]$. Reindex so $i$ sits in cluster $s$, let $w_{ij}$ be how many edges $i$ has into cluster $j$, and let $X_j \sim \text{Bernoulli}(p)$ be that cluster's coin. Then

$$\Pr[Z\in\sigma_i^1]=\Pr[X_s=1]\cdot\Pr\!\left[\textstyle\sum_{j=1}^{s-1}w_{ij}X_j \ge k - w_{is}\right]$$

The sum $\sum_j w_{ij}X_j$ is a **weighted Poisson-binomial**. No closed form, but a dynamic program gets it exactly:

$$f(j,T)=p\,f(j-1,T-w_{ij})+(1-p)\,f(j-1,T)$$

with $T$ bounded by $d_{\max}$, so runtime is $O(d_{\max}\, s)$ per vertex. Because the thresholds are nested, one double for-loop returns the probability of *every* threshold at once — effectively the whole distribution over exposure levels for that user.

For core exposure, exact computation is an open problem. All they offer is Proposition 3.2: $\Pr(\text{core}) \le \Pr(\text{matching neighbourhood condition})$. Useful only as a screen for users whose probability is already dangerously small. [[Monte Carlo Methods|Monte Carlo]] over the randomisation is floated but not tried.

### Variance

Writing $\pi_i^x$ for marginal and $\pi_{ij}^{xy}$ for joint exposure probabilities,

$$\text{Var}[\hat Y^x]=\frac{1}{n^2}\left[\sum_i \frac{1-\pi_i^x}{\pi_i^x}Y_i(\sigma_i^x)^2+\sum_{i}\sum_{j\ne i}\frac{\pi_{ij}^{xx}-\pi_i^x\pi_j^x}{\pi_i^x\pi_j^x}Y_i Y_j\right]$$

and $\text{Var}[\hat\tau]=\text{Var}[\hat Y^1]+\text{Var}[\hat Y^0]-2\text{Cov}[\hat Y^1,\hat Y^0]$.

Two things fall out. Small $\pi_i$ in a denominator is the whole problem. And the cross terms vanish whenever $S_i \cap S_j = \emptyset$ — two users whose neighbourhoods touch no common cluster are independent.

**Proposition 3.3** (easy sufficient condition): if max degree is $O(1)$ and every cluster has size $O(1)$, then $\text{Var}[\hat\tau] = O(1/n)$. Fine asymptotically in $n$, useless as guidance, because the hidden constant can be exponential in degree.

### The negative result

**Proposition 4.4**: under independent *vertex* randomisation with full-neighbourhood exposure on a $d$-regular graph,

$$\text{Var}[\hat\tau] \ge \frac{Y_m^2}{n}\left(p^{-(d+1)}+(1-p)^{-(d+1)}-2\right)$$

For general degrees this becomes a sum over $p^{-(d_i+1)}$ — **one single high-degree vertex can blow up the whole estimate**.

### Restricted-growth graphs and the 3-net clustering

> [!NOTE] Restricted growth
> $G$ is restricted-growth if there is a constant $\kappa$, independent of degree, with $|B_{r+1}(v)| \le \kappa |B_r(v)|$ for all vertices $v$ and all $r>0$, where $B_r(v)$ is the ball of vertices within $r$ hops. The condition is deliberately skipped at $r=0$: going from $\{v\}$ to the neighbourhood is allowed to blow up by $d+1$. After that first hop, growth must be tame. ^restricted-growth

This is weaker than the standard bounded-growth condition $|B_{2r}(v)| \le \kappa|B_r(v)|$ used for nearest-neighbour search. Graphs with a roughly uniform-density embedding in $\mathbb{R}^m$ — lattices, random geometric graphs — satisfy it with $\kappa = 2^m$.

The clustering algorithm is a **3-net** in the shortest-path metric:

1. All vertices unmarked.
2. While unmarked vertices remain: pick any unmarked $v$, call it a centre $v_j$, mark everything in $B_2(v_j)$.
3. Assign every vertex in the graph to its nearest centre, ties broken by index.
4. Each centre's assigned set is a cluster.

**Proposition 4.2**: on a $d$-regular restricted-growth graph, every neighbourhood $B_1(w)$ intersects at most $\kappa^3$ clusters — **a bound with no $d$ in it**. Proof sketch: each $C_j \subseteq B_2(v_j)$; the $B_1$ balls around distinct centres are disjoint (or one centre would have marked the other); if $t$ clusters touched $B_1(w)$ their centres all sit in $B_3(w)$, so $t(d+1)$ disjoint vertices fit inside $B_4(w)$, but restricted growth applied three times gives $|B_4(w)| \le \kappa^3(d+1)$. Contradiction for $t > \kappa^3$. For arbitrary degrees the bound weakens to $\kappa^6$, via the fact that restricted growth forces nearby vertices to have similar degrees.

**Proposition 4.5**: therefore $\pi_i^1 \ge p^{\kappa^3}$, and only vertices within $B_6(i)$ — at most $\kappa^5(d+1)$ of them — can have dependent assignments, giving

$$\text{Var}[\hat Y^1] \le \frac{Y_M^2}{n}\left[(p^{-\kappa^3}-1)+\kappa^5(d+1)(p^{-2\kappa^3-1}-1)\right]$$

plus a covariance contribution bounded by $\frac{2Y_M^2}{n}[\kappa^5(d+1)+1]$. **Linear in $d$, not exponential.** That is the headline: exponential lower bound for naive randomisation, linear upper bound for 3-net cluster randomisation.

## Ablation Studies and Experiments

There is no real experiment. No Facebook A/B test, no real graph, no measured treatment effect — unusual for a KDD paper with two Facebook authors. The evidence is one exact calculation and one simulation.

**The cycle graph, worked exactly.** $n$-vertex cycle, full-neighbourhood exposure, $p=1/2$, everyone responds $\bar Y$ to treatment exposure and $0$ to control. Clusters are contiguous blocks of $c$ vertices.

| clustering | exposure prob | asymptotic variance |
|---|---|---|
| $c=1$ (per-vertex) | $\pi_i = 1/8$ | $(15/2)\,\bar Y^2/n \approx 7.50\,\bar Y^2/n$ |
| block of $c \ge 2$ | — | $\left(\tfrac{c}{2}+2+\tfrac{4}{c}\right)\bar Y^2/n$ |
| $c=2$ | — | $5.00\,\bar Y^2/n$ |
| **$c=3$** | — | $\mathbf{4.83}\,\bar Y^2/n$ |
| $c=4$ | — | $5.00\,\bar Y^2/n$ |
| $c=10$ | — | $7.40\,\bar Y^2/n$ |

The minimum is at $c=3$ — **exactly the size of a neighbourhood on the cycle**. That is the design rule the rest of the paper generalises: make clusters about the size of the thing you need to be internally consistent. Note also that the curve is U-shaped: bigger clusters are not monotonically better, because large clusters make users' exposures correlated and the double-sum term grows.

**Powers of the cycle, simulated.** The $k$-th power of the cycle connects each vertex to its $k$ nearest neighbours on each side, so $d = 2k$. With $n=5000$ and one million sampled randomisations for $k=1..5$: the $k=1$ simulation matches the analytic curve exactly (a good sanity check), the optimal cluster size grows roughly like $c = 2k+1 = d+1$, and the variance at the optimum grows **linearly in $k$**.

**Analytic confirmation for $c = d+1$ on the cycle**: each vertex then touches at most 2 clusters so $1/\pi_i \le p^{-2}$, and at most $3d+1$ other vertices have dependent assignments, giving

$$\text{Var}[\hat\tau] \le Y_M^2(p^{-2}-1)(3d+2)\frac{1}{n}$$

**What did not work / what is left open:**

- Exact probabilities for $k$-core and fractional $q$-core exposure. No tractable algorithm; only an upper bound via the corresponding neighbourhood condition.
- Unbiased **variance estimation**. Following Aronow and Samii, if $\sigma_i^x \cap \sigma_j^y = \emptyset$ for some pair — which cluster randomisation makes *more* likely, since two adjacent users can rarely be in opposite universes — no unbiased variance estimate exists. You can only upper bound it. The point estimate stays unbiased; your confidence interval does not.
- A tractable objective for *choosing* the clustering to minimise variance. Explicitly left as future work; adversarial variance and A/A variance minimisation are floated as candidates.
- The theory covers restricted-growth graphs only. For real social graphs the authors simply say: use any community detection or graph partitioning method, and accept that no variance guarantee applies.

## Worth Remembering

**The bias is not bounded anywhere.** Unbiasedness is conditional on the exposure model being correctly specified. If users respond to something other than what you declared as $\sigma_i^1$, you get bias whose size depends on how far actual observed outcomes sit from the $\vec 1$ / $\vec 0$ outcomes you wanted. The authors say outright that accepting some bias to buy variance may be the right trade, and that quantifying misspecification bias is future work. In practice this is the biggest hole: the exposure model is an untestable assumption sitting under an unbiasedness claim.

**Two knobs, opposite directions.** Bigger clusters raise the exposure probabilities $\pi_i$ (shrinking the $1/\pi$ terms) but induce correlation between users (growing the $\pi_{ij} - \pi_i\pi_j$ terms). The cycle table shows the resulting U shape cleanly. "Cluster more aggressively" is not a free improvement.

**The reusable design heuristic:** cluster at roughly the scale of the exposure condition you need. On the cycle, neighbourhood size 3 → optimal cluster size 3. On $k$-th powers, degree $2k$ → optimal cluster size $2k+1$. This is the same instinct as choosing a switchback interval longer than the carryover window in [[Design and Analysis of Switchback Experiments]] — match the randomisation unit to the interference range.

**Restricted growth probably does not describe your social graph.** It holds for lattices and random geometric graphs. Real social networks are small-world with heavy-tailed degrees — $|B_{r+1}|/|B_r|$ is not bounded by a degree-independent constant in the first few hops. So treat the linear-in-degree result as a proof that *careful clustering can be exponentially better*, not as a guarantee you inherit in production.

**Practical caveats for anyone building this:**

- Fractional $q$-neighbourhood exposure is the one to start with. It needs no per-user special-casing, the DP gives exact probabilities in $O(d_{\max}|S_i|)$, and you get the whole threshold curve for one pass.
- You must log the clustering and the per-cluster coin, not just per-user buckets, or you cannot recompute $\pi_i$ afterwards.
- Screen for users with tiny $\pi_i$ *before* running. A handful of very high degree users touching many clusters will dominate the variance, exactly as in the degree-heterogeneous version of Proposition 4.4. This is the same failure mode as heavy [[The Self-Normalized Estimator for Counterfactual Learning (SNIPS) (NeurIPS)|importance weights]] in off-policy estimation, and the same remedies (clipping, self-normalisation) would trade bias for variance here too.
- On very high-degree social graphs, the paper suggests defining exposure on a *sparsified strong-tie* graph rather than the full friend graph. Cheap and sensible.

**Connections.** The estimator is Horvitz–Thompson, the exposure probability is a propensity — the whole thing is IPS with a graph-shaped propensity, so everything you know about weight variance from [[Doubly Robust Policy Evaluation and Learning]] and [[Counterfactual Reasoning and Learning Systems]] transfers. Nothing here about variance reduction from covariates, so it composes with [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)|CUPED]]-style adjustment in principle, though nobody had done it at the time.

**Open question worth chasing:** the conclusion asks about *continuous* exposure — a response linear in the number of exposed neighbours $k$ rather than a binary in/out condition. That is the version you actually want for a dose-response curve, and the discrete indicator $\mathbf 1[Z \in \sigma_i^x]$ throws away most of the data (everyone who fell short of the threshold contributes nothing).

## Links

Related: [[AB Testing]] · [[Controlled experiments on the web- survey and practical guide (DMKD)]] · [[Trustworthy Online Controlled Experiments- Five Puzzling Outcomes Explained (KDD)]] · [[Design and Analysis of Switchback Experiments]] · [[Recommendations as Treatments- Debiasing Learning and Evaluation]] · [[Off-Policy Evaluation]] · [[The Self-Normalized Estimator for Counterfactual Learning (SNIPS) (NeurIPS)]] · [[Doubly Robust Policy Evaluation and Learning]] · [[Counterfactual Reasoning and Learning Systems]] · [[Improving the Sensitivity of Online Controlled Experiments (CUPED) (WSDM)]] · [[Unbiased Learning-to-Rank with Biased Feedback]] · [[Monte Carlo Methods]]

New topics worth writing: SUTVA and interference, Horvitz–Thompson estimator, k-core decomposition, r-nets and doubling dimension, graph partitioning / balanced label propagation, network bucket testing, ego-cluster randomization, Poisson-binomial distribution
