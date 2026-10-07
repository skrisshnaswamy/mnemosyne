---
title: "The Linear Representation Hypothesis Needs a Group Action"
authors: ["Yao et al."]
year: 2026
arxiv: "2609.27158"
url: https://arxiv.org/abs/2609.27158
priority: Must-Read
read_on: 2026-09-25
tags: [paper, theory]
---
## The Core Idea

"Features are directions in activation space." Everybody says it. Almost nobody says what "the same direction" means.

That is the whole paper. If you want to claim something about *the representation* rather than about *this one trained model's coordinates*, you must say which changes of the numbers count as merely re-describing the same thing, and which count as a genuinely different representation. Rotate the activation space, rescale it, shift it — is that a different representation or the same one in new clothes?

Once you ask this, the Linear Representation Hypothesis stops being one hypothesis and becomes a family of them. Different members place the feature in different mathematical spaces, and each space tolerates a different set of re-descriptions.

Four formulations, all called "linear":

- **Displacement.** $h_b - h_a \approx v$. The king–queen offset from [[Efficient Estimation of Word Representations (word2vec)|word2vec]], and also what difference-in-means steering vectors are. Under $h \mapsto Ah + t$, the offset transforms as $v \mapsto Av$ — the shift $t$ cancels. The feature lives in $V$.
- **Decodability.** A linear probe $w$ reads a property off as $w^\top h$. Under $h \mapsto Ah$, the probe must transform as $w \mapsto A^{-\top} w$ to keep the readout. The feature lives in $V^*$, the **dual** space.
- **Superposition.** $h = b + \sum_i x_i W_i$. Affine-covariant in the activations; separately ambiguous in the latent coordinates.
- **Subspace.** A feature need not be one line. Days-of-the-week live on a circle, a 2-D thing. That object lives in the Grassmannian $\mathrm{Gr}(k, V)$.

> [!NOTE] The primal/dual trap
> A steering vector and a probe weight are both stored as a length-$d$ float array. They are **not** the same kind of object. There is no nonzero map $V^* \to V$ that commutes with all invertible linear changes of coordinates. You can only identify them by *choosing an inner product*, and that choice is itself an assumption. ^primal-dual

So: cosine-similarity between a probe vector and a steering vector is a claim that only survives under *similarity* transformations (rotation + one global scale). Literally adding probe weights to activations as a steering intervention is stronger still — it only survives under *isometries*. Neither is wrong. Both are usually unstated.

What this unlocks: a checklist that lets you tell whether two interpretability papers are testing the same hypothesis or two different ones that happen to produce arrays of the same shape.

## The Methodology

There is no model and no training run. The contribution is a specification format plus two theorems about it.

**The four things you must state.** A representation claim is a tuple $(G, M, F, P)$:

| Symbol | What it is |
|---|---|
| $G$ | the group of transformations you treat as *mere re-description* |
| $M$ | the space the extracted object lives in, **with** how $G$ acts on it |
| $F$ | the procedure that produces the object from activations |
| $P$ | the predicate you finally assert |

The claim is $P(F(\phi))$ where $\phi: \mathcal{X} \to V$ is the representation.

**The admissibility conditions.** For the claim not to be an artefact of coordinates:

$$F(g \cdot \phi) = g \cdot F(\phi), \qquad P(g \cdot m) = P(m)$$

$F$ must be **equivariant** (transform along with the coordinates) and $P$ must be **invariant** (not change at all). Note this is stronger than just requiring the composite $P \circ F$ to be invariant — the authors insist on the factorisation because the intermediate object is itself part of the claim and gets reused downstream.

**The three groups.** Everything sits in a hierarchy:

$$G_{\text{iso}} \subset G_{\text{sim}} \subset G_{\text{aff}}$$

- $G_{\text{aff}}$: any invertible $A$ plus a shift. Preserves collinearity, subspace dimension, rank. Kills angles and lengths.
- $G_{\text{sim}}$: $A = sQ$ with $Q$ orthogonal, $s > 0$. Adds angles, orthogonality, cosine.
- $G_{\text{iso}}$: $s = 1$. Adds absolute norms and distances.

Bigger group = fewer things you're allowed to say = **stronger** claim. The strength of a result is the *largest* group under which it survives.

**The architectural floor.** This is the part with teeth, and it is not a free choice.

Let $\Gamma$ be the set of parameter changes that leave the computed function $f_\theta$ exactly unchanged. If such a change pushes through to the activations at your reading point, your $G$ must contain it. Otherwise your number differs between two parameter settings that compute the *same model*, and it is a property of the parameterisation, not the model.

> [!NOTE] Reading-point dependence
> The admissible group depends on **where in the network you read**. Same architecture, different answer at different sites. ^reading-point

The worked case: in dot-product [[Attention]], $W_Q \mapsto \Lambda^{-\top} W_Q$ and $W_K \mapsto \Lambda W_K$ leaves every logit identical, for *any* invertible $\Lambda \in \mathrm{GL}(d_{\text{head}})$. At the residual stream this does nothing. At a query or key site it means the gauge group contains all of $\mathrm{GL}(d_{\text{head}})$.

And $\mathrm{GL}(d_{\text{head}})$ admits no nonzero invariant bilinear form. Take $\Lambda = cI$: then $b(cv, cw) = c^2 b(v,w) = b(v,w)$ for all $c > 0$ forces $b = 0$. **There is no inner product on per-head key space that the architecture respects.** Cosines between keys, key-cache PCA, $\log\det(K^\top K)$ — none of these are properties of the model.

[[RoPE]] shrinks this gauge but does not rescue it. The surviving $\Lambda$ are exactly those commuting with all $R(\tau)$, which (for distinct rotary frequencies) is

$$\Lambda = \bigoplus_{j=1}^{n} s_j R_2(\alpha_j), \qquad s_j > 0$$

— independently scaled rotations of each rotary plane. The $s_j$ vary *per plane*, so norms change, cosines change, and the eigenvalues of $K^\top K$ move non-uniformly. Only a uniform $s_1 = \cdots = s_n$ would restore $G_{\text{sim}}$, and the architecture does not impose it.

**Composition.** Analyses are pipelines. If every stage is equivariant under a common $G$, the composite is. If not, you must check the whole thing. The practical rule: **the pipeline is capped by its most restrictive stage.** Estimate a direction under $G_{\text{aff}}$, compare by cosine under $G_{\text{sim}}$, calibrate by a norm under $G_{\text{iso}}$ — the conclusion is only guaranteed under $G_{\text{iso}}$.

One subtlety: this is not simple intersection of per-stage groups. Centering changes the action the next stage sees. $h - \bar h \mapsto A(h - \bar h)$, so cosine-after-centering is invariant to translations that would break cosine on raw activations.

## Ablation Studies and Experiments

There are no benchmarks. The empirical content is (a) an audit table classifying standard quantities by their maximal group, and (b) a set of case studies of published work. Both are reusable.

**The audit table**, condensed to the parts worth memorising:

| Survives under | Quantities |
|---|---|
| $G_{\text{aff}}$ (weakest structure, strongest claim) | exact rank, span, subspace containment and intersection dimension, CCA on centered data |
| $G_{\text{sim}}$ | leading principal subspace, effective rank / participation ratio, intrinsic dimension, cosine / angles / orthogonality, principal angles, Grassmann distance, linear CKA, nearest-neighbour agreement |
| $G_{\text{iso}}$ (strongest structure, narrowest claim) | norms, distances, intervention magnitude, absolute-threshold rank, Procrustes distance, reconstruction loss |
| signed permutations of latents only | coordinatewise sparsity penalty |

**Sparse autoencoders get pulled apart into two separate symmetry stories.** Under $h \mapsto Ah + t$, take $D \mapsto AD$, $W_{\text{enc}} \mapsto W_{\text{enc}}A^{-1}$, and absorb $t$ into the biases: every pre-activation and therefore every latent code is *unchanged*. The dictionary is affine-covariant. But the residual transforms as $A(h - \hat h)$, so the Euclidean reconstruction loss is only preserved when $A^\top A = I$. **It is the loss, not the architecture, that drags SAEs down to $G_{\text{iso}}$ on the activation side.** Separately, on the latent side, $W \mapsto WB$, $x \mapsto B^{-1}x$ leaves reconstruction alone; the coordinatewise $\ell_1$ penalty cuts that mixing symmetry down to signed permutations. Two different restrictions on two different spaces, routinely conflated.

**Case studies — what breaks:**

- *Refusal direction* (Arditi et al. 2024). One difference-in-means estimate is used two ways: added to activations **unnormalised** (a displacement in $V$, magnitude matters) and **normalised then projected out** (a projective direction in $\mathbb{P}(V)$, only the line matters). Both work. Both are reported as evidence for one direction. They live in different spaces and need different equivalences.
- *Steering normalisation conventions.* CAA normalises magnitudes across behaviours but not across layers — and residual-stream norms grow through the forward pass, so the same coefficient means different displacements at different depths. ActAdd does the opposite and leaves the vector unnormalised. The same number denotes different interventions under the two conventions.
- *SVCCA* (Raghu et al. 2017). CCA alone is affine-invariant on centered data. SVCCA truncates singular values first — a spectral operation not preserved by general invertible maps. The CCA stage cannot give back what preprocessing destroyed. This is the cleanest "pipeline capped by worst stage" failure in the paper.
- *KeyDiff* (KV-cache eviction by cosine similarity among keys within a head). The eviction rule may work fine empirically; the *geometric explanation* is the problem. Every quantity offered in support — key cosines, $L^2$ norms, key-cache PCA, $\log\det(K^\top K)$ — is gauge-dependent at a key site. Two parameter settings computing the identical function evict different tokens.
- *Cross-head geometry.* Cross-Gram singular values between head projections transform as $G_{hh'} \mapsto B_h^\top G_{hh'} B_{h'}$ — not invariant unless you orthonormalise the spanning matrices first. Cosines between head-space singular vectors, calibrated against a uniform distribution on the Euclidean sphere, inherit the same problem: an anisotropic change of head coordinates moves both the statistic and the reference distribution.

**What does pass the audit:**

- *ITI* (Li et al. 2023) turns out to be fine. They normalise the probe direction and scale the added displacement by the empirical standard deviation along it. Under $A = sQ$: the unit direction goes to $Q\hat w$, the projected std goes to $s\sigma$, so $\sigma\hat w \mapsto A(\sigma \hat w)$. Covariant. The construction is admissible under $G_{\text{sim}}$ — though identifying a $V^*$ probe direction with a $V$ intervention direction still leans on a chosen geometry.
- *Wollschläger et al. 2025* explicitly rescale an optimised refusal direction to match the norm of a difference-in-means direction, and sample unit directions directly inside refusal cones rather than normalising convex combinations (which would bias the induced distribution). The procedure is matched to the projective object the intervention actually consumes.
- *Yamagiwa et al.* compare column spaces of transposed projection matrices — subspaces of the **residual stream**, which the head gauge leaves untouched. Right object, right space.

**The constructive result.** For KeyDiff, the paper supplies a gauge-invariant replacement. What you actually care about is how much substituting one key for another changes attention logits, and that is mediated by $q^\top(k_1 - k_2)$. So let $\Omega = \mathbb{E}[qq^\top]$ over realised queries and use

$$d(k_1, k_2) = (k_1 - k_2)^\top \Omega (k_1 - k_2)$$

Under the gauge $q \mapsto \Lambda^{-\top} q$, $k \mapsto \Lambda k$, we get $\Omega \mapsto \Lambda^{-\top}\Omega\Lambda^{-1}$ and $k_1 - k_2 \mapsto \Lambda(k_1-k_2)$. The two cancel. $d$ is unchanged, and it literally equals the mean squared logit change from the substitution. When $\Omega \propto I$ it reduces to Euclidean distance, which orders pairs like cosine does when key norms match — so the original rule is the special case where the sampled gauge happened to make queries isotropic.

**The Platonic Representation Hypothesis gets the same treatment.** Convergence claims need a group too, and the truth value turns on which one. Kornblith et al. showed that if representation width $\geq$ number of sampled inputs and both activation matrices have full row rank, *any* similarity measure invariant to arbitrary invertible linear maps cannot distinguish them at all. So affine-invariant convergence is near-trivial in that regime. Isometric convergence is impossible across models with different widths and tokenizers. Practice sits near $G_{\text{sim}}$ — but by accident, determined by whichever alignment measure was reached for, not stated in advance.

## Worth Remembering

**The maximality point is the one to keep.** Condition (A1) is trivially satisfiable — just shrink $G$ or weaken $P$ until nothing is at risk. Admissibility alone says nothing. What carries content is the *largest* group under which your claim survives. Reporting under an unnecessarily small group isn't invalid, it's just a loss of information. This is why the audit table is organised by maximal group.

**You cannot just declare $G_{\text{iso}}$ everywhere and move on.** That is the obvious escape hatch, and the architectural floor forbids it whenever the model realises transformations outside $G_{\text{iso}}$ — which it does, at every query and key site.

**The honest objection the authors take seriously:** training returns one specific $\theta$, and the geometry of that $\theta$ is a real fact about the artefact you have. True, and permitted. But it fixes what the statement is *about*. A quantity that varies across the gauge orbit is a property of $\theta$ plus the training procedure, not of $f_\theta$. The empirical obligation that follows — show the geometry is stable across seeds and runs before generalising — is "rarely discharged".

**Connections worth chasing:**

- The [[How Contextual are Contextualized Word Representations|anisotropy]] and [[Representation Degeneration Problem in Training NLMs|representation degeneration]] literature is entirely built on cosine and spectral quantities in the residual stream. Residual-stream reading points are the *safe* case here, so those results survive — but the argument tightens up if you say so explicitly.
- [[Whitening Sentence Representations]] and the near-orthogonality debate get sharper: Golechha et al. argue near-orthogonality arises generically in high dimensions and can be *induced* by whitening. Making the geometry explicit turns "is the structure real?" into the precise question "under which inner product does the orthogonality predicate hold?"
- [[Understanding Dimensional Collapse in Contrastive Learning]] uses covariance spectra — check which of those quantities are ratio-based (survive $G_{\text{sim}}$) versus absolute-threshold (need $G_{\text{iso}}$).
- The framing rhymes with [[Troubling Trends in Machine Learning Scholarship]]: a suggestive but underspecified term ("linear representation", "direction") absorbing several incompatible meanings.

**Practical caveats if you want to use this:**

1. It does not tell you which group is *correct*. It tells you that you have to pick one, and that the architecture sets a lower bound.
2. Adopting it will narrow some of your existing conclusions. That is the intended cost.
3. The immediately actionable piece is the audit table plus the reading-point check. Before reporting a cosine, ask: what function-preserving reparameterisation exists at this site, and does it move my number?
4. The KeyDiff fix is a template. If your quantity is gauge-dependent, look for the bilinear form the architecture actually uses, and build the metric from it.

**Open question the paper leaves:** the architectural floor is computed for attention QK sites and residual streams. Nobody has worked out $\mathrm{Im}\,\rho$ for MoE routers, normalisation layers with learned gains, or gated MLPs — all of which have their own function-preserving rescalings.

## Links

Related: [[Attention]] · [[Query, Key, and Value (QKV)]] · [[Multi-Head Attention]] · [[RoPE]] · [[RoFormer- Enhanced Transformer with Rotary Position Embedding]] · [[Embeddings]] · [[Efficient Estimation of Word Representations (word2vec)]] · [[How Contextual are Contextualized Word Representations]] · [[Representation Degeneration Problem in Training NLMs]] · [[Whitening Sentence Representations]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Autoencoder]] · [[KV Cache]] · [[Linear Projection]] · [[Troubling Trends in Machine Learning Scholarship]] · [[Saliency]] · [[Fourier Series Decomposition]]

New topics worth writing: group actions and equivariance, gauge symmetry in neural network parameters, the Linear Representation Hypothesis, sparse autoencoders and dictionary learning, activation steering and difference-in-means directions, linear probing, the Platonic Representation Hypothesis, CKA and representational similarity measures, dual spaces and covariance/contravariance, Grassmannians and projective space, SVCCA
