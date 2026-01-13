---
title: "A Tutorial on Energy-Based Learning"
authors: ["LeCun et al."]
year: 2006
url: http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, optimization, self-supervised, vision]
---
## The Core Idea

Strip a model down to one scalar function: $E(W, Y, X)$. It takes an input $X$ and a candidate answer $Y$ and returns a number. Low number = "these two go together". High number = "these two do not".

Everything else follows.

**Inference** is minimisation: $Y^* = \arg\min_{Y \in \mathcal{Y}} E(W, Y, X)$. You do not compute a probability, you search for the lowest-energy answer.

**Learning** is shaping that surface: push $E$ down at the correct answers in your training set, push it up somewhere else.

The point of the framework is what it *removes*. A probabilistic model must be normalised — the numbers over all possible $Y$ must sum to 1. That forces you to compute the partition function $\int_{y} e^{-\beta E(W,y,X)}$, which is often intractable (imagine $\mathcal{Y}$ = all English sentences, or all images). An energy function has no such constraint. You can build $E$ out of any architecture you like — a conv net, a dynamic-programming trellis, a sum of arbitrary factors — and you never have to make the pieces normalise.

> [!NOTE] Energy function
> A scalar-valued function $E(W,Y,X)$ measuring incompatibility between input $X$ and answer $Y$. Small = compatible. Energies are in arbitrary units; only *differences* between energies of different $Y$ for the same $X$ carry meaning. ^energy-function

The price you pay: energies are uncalibrated. You cannot combine two separately trained energy models, because there is no guarantee their scales match. If you need calibrated numbers, you convert with the Gibbs distribution

$$P(Y|X) = \frac{e^{-\beta E(Y,X)}}{\int_{y \in \mathcal{Y}} e^{-\beta E(y,X)}}$$

and pay the partition-function cost. LeCun's argument is that most tasks — classification, ranking, detection, control — never need this. So do not pay for it.

The second, deeper idea is the **collapse problem**. If your loss only says "make the energy of the right answer low", a flexible enough architecture will just make *every* energy low — a flat surface at zero. Correct answers have low energy, but so does everything else, so inference is useless. Any working loss must contain a **contrastive term** that pulls energy *up* somewhere. The whole design space of losses is: *which* wrong answers do you pull up, and *how hard*?

That framing swallows a large chunk of ML. [[Cross Entropy]] / negative log-likelihood is "pull up on every answer, weighted by its current probability". Hinge loss is "pull up on the single worst offender". Contrastive learning ([[A Simple Framework for Contrastive Learning (SimCLR)|SimCLR]], [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)|InfoNCE]]) is "pull up on the in-batch negatives". Contrastive divergence is "pull up on a nearby MCMC sample". Same skeleton, different sampling policy.

## The Methodology

Building an EBM means designing four things:

1. **Architecture** — the internal structure of $E(W,Y,X)$.
2. **Inference algorithm** — how you find $\arg\min_Y E$. Exhaustive search if $|\mathcal{Y}|$ is small; [[Derivative#Gradient|gradient]] descent if $Y$ is continuous and $E$ is smooth; Viterbi / dynamic programming if $E$ decomposes over a chain; graph cuts, simulated annealing, linear programming otherwise.
3. **Loss functional** $\mathcal{L}(W, \mathcal{S}) = \frac{1}{P}\sum_i L(Y^i, E(W, \cdot, X^i)) + R(W)$ — note the per-sample loss sees the *whole energy slice* $E(W, Y, X^i)$ over all $Y$, not just the energy at $Y^i$. $R(W)$ is the [[Regularization|regulariser]].
4. **Learning algorithm** — usually stochastic gradient descent, and LeCun argues bluntly that a well-tuned SGD beats "sophisticated" second-order batch methods in practice, because SGD exploits redundancy between samples.

### The three answers that matter

For training sample $(X^i, Y^i)$:
- $Y^i$ — the correct answer.
- $Y^{*i}$ — what the model currently outputs (lowest energy overall).
- $\bar{Y}^i$ — the **most offending incorrect answer**: the lowest-energy answer that is *wrong*. In the continuous case, "wrong" means at least $\epsilon$ away from $Y^i$.

> [!NOTE] Most offending incorrect answer
> $\bar{Y}^i = \arg\min_{Y \neq Y^i} E(W, Y, X^i)$. The wrong answer the model currently likes best. Margin losses target exactly this one point. ^most-offending

### The loss zoo

Write $E_C = E(W, Y^i, X^i)$ (correct) and $E_I = E(W, \bar{Y}^i, X^i)$ (incorrect). Then:

| Loss | Formula | Margin |
|---|---|---|
| Energy | $E_C$ | **none** |
| Perceptron | $E_C - \min_Y E(W,Y,X^i)$ | 0 |
| Hinge | $\max(0, m + E_C - E_I)$ | $m$ |
| Log | $\log(1 + e^{E_C - E_I})$ | $>0$ |
| LVQ2 | $\min(1, \max(0, (E_C - E_I)/\delta E_I))$ | 0 |
| MCE | $\sigma(E_C - E_I)$ | $>0$ |
| Square-square | $E_C^2 + (\max(0, m - E_I))^2$ | $m$ |
| Square-exp | $E_C^2 + \gamma e^{-E_I}$ | $>0$ |
| NLL / MMI | $E_C + \frac{1}{\beta}\log\int_y e^{-\beta E(W,y,X^i)}$ | $>0$ |
| MEE | $1 - e^{-\beta E_C}/\int_y e^{-\beta E}$ | $>0$ |

The NLL gradient is worth memorising, because it *is* the contrastive picture:

$$\frac{\partial L_{\text{nll}}}{\partial W} = \frac{\partial E(W, Y^i, X^i)}{\partial W} - \int_{Y \in \mathcal{Y}} \frac{\partial E(W, Y, X^i)}{\partial W} P(Y|X^i, W)$$

Push down on the correct answer's energy; pull up on every answer with force proportional to how likely the model currently thinks it is. This is the same "positive phase minus negative phase" you see in Boltzmann machines and in [[Generative Adversarial Networks|GAN]] discriminator training.

Two limits worth knowing: NLL $\to$ perceptron loss as $\beta \to \infty$ (zero temperature — only the single lowest-energy answer gets pulled up), and NLL $\to$ log loss when $|\mathcal{Y}| = 2$.

### Classic models as EBMs

- **Regression**: $E = \frac{1}{2}\|G_W(X) - Y\|^2$. Inference is trivial ($Y^* = G_W(X)$). Energy loss, perceptron loss, and NLL are all *identical* here because the contrastive term is constant. The parabola shape is what prevents collapse — pushing the apex down at $Y^i$ automatically raises everything else.
- **Binary classifier**: $E = -Y G_W(X)$, $Y \in \{-1,+1\}$. Perceptron loss gives the perceptron rule. Hinge loss with $G_W(X) = W^\top X$ and $\|W\|^2$ regulariser gives a linear SVM. NLL gives $\frac{1}{P}\sum \log(1 + e^{-2Y^i W^\top \Phi(X^i)})$ — logistic regression.
- **Siamese / implicit regression**: $E = \frac{1}{2}\|G_{W_1}(X) - G_{W_2}(Y)\|^2$. Lets *many* $Y$ have low energy for one $X$ — needed for "the cat ate the ___" or for constraints like $X^2 + Y^2 = 1$. This is the architecture behind signature verification, face verification, and by direct descent [[Sentence-BERT]] and modern contrastive encoders.

### Latent variables

Sometimes the energy depends on a hidden $Z$ you never observe — face pose, word segmentation, character boundaries. Inference minimises over it too:

$$E(Y,X) = \min_{Z \in \mathcal{Z}} E(Z,Y,X), \qquad Y^* = \arg\min_{Y,Z} E(Z,Y,X)$$

Or you marginalise instead: $Y^* = \arg\min_Y -\frac{1}{\beta}\log\int_z e^{-\beta E(z,Y,X)}$, which is the same thing with $E$ replaced by a **free energy**. Minimising is the $\beta \to \infty$ case of marginalising. The only thing distinguishing $Z$ from $Y$ is that you are given labels for $Y$ during training and never for $Z$.

### Factor graphs and efficient inference

If $E$ decomposes as a sum of factors over subsets of variables, you never enumerate all configurations. Their toy example: $E = E_a(X,Z_1) + E_b(X,Z_1,Z_2) + E_c(Z_2,Y_1) + E_d(Y_1,Y_2)$ with three binary and one ternary variable. Naive search evaluates the energy $2\times2\times2\times3 = 24$ times (96 factor evaluations). Building a trellis instead needs $2 + 4 + 4 + 6 = 16$ factor evaluations, and the answer is the shortest path — Viterbi.

For the log partition function you run the same recursion in the soft domain (the **forward algorithm**), with each trellis node computing

$$\alpha_k = -\frac{1}{\beta}\log\sum_j e^{-\beta(E_{kj} + \alpha_j)}$$

which becomes min-sum as $\beta \to \infty$. Gradients flow back through this trellis exactly like [[Backpropagation]] through a feed-forward net.

### Late normalisation

HMMs and directed Bayes nets normalise *internally* — outgoing transition probabilities per state must sum to 1. This creates the **label bias problem** (Bottou, 1991): transitions leaving a state compete only with each other, so paths through low-branching states get inflated probability. Also the **missing probability mass problem**: models must add a fake "background model" to absorb junk inputs. EBMs normalise only at the very end, if at all. Same reason CRFs beat maximum-entropy Markov models.

A second practical win: in an HMM, emission probabilities are Gaussian mixtures in 10–100 dimensions while transition probabilities are discrete over a handful of arcs. The dynamic ranges differ by orders of magnitude, so transitions count for nothing. Practitioners hack this by raising transitions to a power — which breaks normalisation and is embarrassing in a probabilistic framework. In an EBM you just multiply a subset of energies by a coefficient and move on.

### Graph Transformer Networks

For handwriting, the factor graph is too large to write down (every grammatical transcription × every segmentation). GTNs instead treat the *trellis itself* as the data structure being passed between layers. A GTN is a stack of **graph transformers**, each taking graphs in and emitting graphs out, the way a neural net stacks vector→vector layers.

The check-reading pipeline: over-segment the image → **segmentation graph** $Gr_{seg}$ (each path = one way to cut the image into characters, each arc = a piece of ink) → **recognition transformer** applies one copy of a 2D conv net $G_W$ per arc, emitting one arc per character class weighted by its energy → **interpretation graph** $Gr_{int}$ → **path selector** keeps only paths whose label sequence matches $Y$ → $Gr_{sel}$ → **Viterbi transformer** picks one path, indexed by latent $Z$ (the segmentation).

The energy is $E(W,Y,X) = \sum_{kl}\delta_{kl}(Y)G_{kl}(W,X)$ where $\delta_{kl} = 1$ if that arc survived into the final graph. So the gradient is just $\sum_{kl}\delta_{kl}(Y)\,\partial G_{kl}/\partial W$ — you track which arcs were selected and backprop through them. Two named training modes: *discriminative Viterbi training* (= perceptron loss, minimise over $Z$) and *discriminative forward training* (= NLL loss, marginalise over $Z$ with the forward algorithm). The commercially deployed bank-check reader used the NLL version.

## Ablation Studies and Experiments

The empirical core is a deliberately trivial task: learn $Y = X^2$ from 200 samples with $X$ uniform on $[-1, 1]$. The point is not accuracy, it is watching the energy surface deform.

**Setup A — safe architecture.** $E = \|G_W(X) - Y\|_1$, with $G_W$ a 1–20–1 net with sigmoid hidden units. Trained with the plain **energy loss** (no contrastive term at all). Result: works fine. The surface converges to the correct V-shaped valley after ~39 epochs. Why it is safe: the energy as a function of $Y$ is always a V with fixed slopes; only the apex can move. Pushing the apex to $Y = X^2$ *necessarily* raises every other $Y$. NLL and perceptron loss give identical results here because their contrastive terms are constant.

**Setup B — flexible architecture, same loss. This is the failure.** $E = \|G_{W_1}(X) - G_{W_2}(Y)\|_1$, each branch a 1–10–10 net. Trained with energy loss. Within 9 epochs the surface is **flat and zero everywhere**. Both branches learned to ignore their inputs and emit the same constant vector. Loss is minimised, model is worthless. This is the cleanest demonstration in the paper: *the loss was fine for architecture A and fatal for architecture B.* Architecture and loss must be chosen together.

**Setup B + square-square loss.** $L = E_C^2 - (\max(0, m - E_I))^2$. The contrastive term explicitly pushes up any point whose energy threatens to drop below the correct answer's. Surface reaches the right shape by ~34 epochs. No collapse.

**Setup B + NLL loss.** Converges *faster* than square-square, and the minimum is *deeper* — because square-square pins incorrect energies at exactly $m$, while NLL pushes them toward infinity (with exponentially decaying force). But each iteration is far more expensive: the integral over $Y$ was approximated by 20 regularly spaced points in $[-1,1]$, so 20 energy-gradient evaluations per step versus 2 for square-square. (Fair caveat the authors make: square-square still has to *find* $\bar{Y}^i$, which is not free.)

### The formal condition

The theory section gives a sufficient condition. Work in the 2D plane of $(E_C, E_I)$. Let $HP_1$ be the half-plane $E_C + m < E_I$ (correct answer wins by the margin) and $HP_2$ its complement. Let $R$ be the feasible region — all $(E_C, E_I)$ reachable by some $W$.

> [!NOTE] Condition 3 (sufficient condition for a good loss)
> Minimising $L$ drives the model to the margin condition if there exists at least one point $(e_1, e_2) \in R \cap HP_1$ such that $Q(e_1,e_2) < Q(e_1', e_2')$ for **every** $(e_1', e_2') \in R \cap HP_2$. In plain terms: the loss surface, viewed only in terms of the correct and most-offending energies, must have a strictly better point on the good side of the margin than anything on the bad side. ^condition-3

The NLL proof is short and worth keeping: $g_C = \partial L/\partial E_C = 1 - e^{-E_C}/\sum_Y e^{-E_Y} > 0$ and $g_I = \partial L/\partial E_I = -e^{-E_I}/\sum_Y e^{-E_Y} < 0$ *always*. So the negative gradient always points into $HP_1$, and the loss decreases monotonically as you cross the margin. A first-order Taylor step of size $\epsilon$ from the best point on the margin line strictly decreases the loss.

### What does not work, and the caveats

- **Energy loss with a general architecture**: no margin, marked "none" in the table. Collapses. Only safe when the architecture itself bounds things — e.g. an RBF output layer $E = \sum_k \delta(Y-k)\|U_k - G_W(X)\|^2$ with **fixed, distinct** centres $U_k$. Because $\|U_1 - U_2\|^2 = d$ is fixed, $d_1 + d_2$ has a strictly positive lower bound, so the region $E_C + E_I \leq d$ is unreachable and collapse is geometrically impossible. **Let the RBF centres be learned and this guarantee vanishes** — the centres merge and everything collapses. A one-line change from "good loss" to "bad loss".
- **Perceptron loss**: margin exactly 0, so in principle it can collapse. In practice it often survives — the authors' honest reasoning is that the collapsed solutions occupy a small piece of parameter space and nothing actively *drives* the system toward them. It was used successfully for handwriting recognition and POS tagging. But they call the oversight in LeCun et al. 1998a a real one.
- **LVQ2 loss**: also zero margin, and non-convex, and it saturates — it caps how much any single outlier contributes. This makes it robust to label noise (matters in speech) at the cost of convexity. Used successfully for isolated-word recognition despite the zero margin.
- **Convexity is oversold.** The paper argues directly against the CRF/max-margin community's main selling point: "convex loss functions are no guarantee for good performance, and non-convex losses may in fact be easier to optimize than convex ones in practice." Cites Huang & LeCun 2006 and Collobert et al. 2006.
- **The Lafferty et al. claim is rebutted.** CRF paper claimed GTNs do not define a proper probability distribution over label sequences. The paper shows the NLL-trained GTN does: $P(Y|X) = \int_z e^{-\beta E(z,Y,X)} / \int_{y,z} e^{-\beta E(y,z,X)}$.
- **Never tried**: training GTNs with a generalised margin loss. Explicitly flagged as an open gap.

## Worth Remembering

**The taxonomy of partition-function difficulty** — this is the most useful practical chart in the paper:

1. *Trivial* — $|\mathcal{Y}|$ small, or $Z$ does not depend on $W$. Energy loss is safe.
2. *Analytical* — closed form, e.g. quadratic energy in $Y$ → Gaussian integral.
3. *Computable* — exponentially many terms but factorisable. Tree-structured graphical models, chain trellises. Forward algorithm.
4. *Approachable* — loopy graphs, needs loopy belief propagation or variational approximation. **Reframing worth keeping: variational methods are just "a way to choose a subset of energies to pull up."**
5. *Intractable* — must sample. And "a sampling method is a policy for choosing suitable candidate answers whose energy will be pulled up."

Once you see it this way, a non-probabilistic loss like hinge is not an *approximation* to the probabilistic one — it is a different, deliberate sampling policy (always pick the worst offender).

**Contrastive divergence** fits in cleanly as two shortcuts on NLL: approximate the contrastive integral by [[Markov Chain Monte Carlo|MCMC]] samples, and start the chain *at the correct answer* and run only a few steps. You get $\tilde{Y}^i$ near $Y^i$, then $W \leftarrow W - \eta(\partial E(W,Y^i,X^i)/\partial W - \partial E(W,\tilde{Y}^i,X^i)/\partial W)$. This makes the correct answer a *local* minimum but gives no guarantee about distant low-energy wrong answers.

**The efficiency question they leave open**, and which is still open: how many wrong answers must you pull up before the surface is correct? NLL pulls up everything including answers your inference algorithm will never reach — wasteful. Margin losses pull up one point — cheap but maybe too few. They propose (but do not construct) a figure of merit: compute cost of the loss and its gradient, relative to the *volume* of threateningly-low incorrect answers whose energy gets raised. That is basically the question the entire negative-sampling literature ([[Distributed Representations of Words and Phrases (negative sampling)|word2vec negative sampling]], in-batch negatives, hard negative mining) has been answering empirically ever since.

**Learning with approximate inference is quietly fine.** If your inference algorithm can never reach some far-off region of $\mathcal{Y}$, the model may assign low energy there — but you never have to pull it up, because it never surfaces. This is an *advantage* of energy-based losses over NLL, where the contrastive term dutifully pulls up on answers no search will ever find.

**Practical caveat if you want to use this**: the framework is descriptive, not prescriptive. It tells you *whether* a (loss, architecture) pair can collapse. It does not tell you which pair is best for your task. The most transferable rule of thumb is the Setup A / Setup B contrast: if your energy's shape in $Y$ is *fixed* (a parabola, a V) and only its location moves, a contrastive term is optional. The moment $Y$ goes through its own learned encoder, you need one.

**Connections forward.** Almost everything with a "negative" in it is here in embryo: [[A Simple Framework for Contrastive Learning (SimCLR)|SimCLR]]'s NT-Xent is NLL over in-batch negatives; [[Understanding Contrastive Learning through Alignment and Uniformity|alignment and uniformity]] is exactly "push down on positives, pull up elsewhere" decomposed; [[Understanding Dimensional Collapse in Contrastive Learning|dimensional collapse]] and [[Mode Collapse]] are the same failure the flat energy surface shows in Figure 11; [[Self-Supervised Learning from Images with I-JEPA|I-JEPA]] and [[LeJEPA- Provable and Scalable Self-Supervised Learning|LeJEPA]] are LeCun's own descendants, where SIGReg is a modern answer to "what contrastive term prevents collapse without enumerating negatives". [[Score-Based Generative Modeling through SDEs|Score-based models]] and [[Denoising Diffusion Probabilistic Models|diffusion]] are the sidestep: instead of the energy, learn $\nabla_Y \log p$, which kills the partition function by differentiation. And [[Direct Preference Optimization (DPO)|DPO]] is a hinge-flavoured energy on preference pairs.

**Open question for later**: do the variational approximations everyone actually uses (mean field with standard architectures) satisfy Condition 3? The authors say they do not know. Twenty years on I do not think anyone checked.

## Links

Related: [[Cross Entropy]] · [[KL Divergence]] · [[Backpropagation]] · [[Loss, Objectives, and Business Alignment]] · [[Generative Adversarial Networks]] · [[Denoising Diffusion Probabilistic Models]] · [[Score-Based Generative Modeling through SDEs]] · [[A Simple Framework for Contrastive Learning (SimCLR)]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[Understanding Contrastive Learning through Alignment and Uniformity]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Self-Supervised Learning from Images with I-JEPA]] · [[LeJEPA- Provable and Scalable Self-Supervised Learning]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[A Neural Probabilistic Language Model (JMLR)]] · [[Sentence-BERT]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Direct Preference Optimization (DPO)]] · [[Mode Collapse]] · [[Markov Chain Monte Carlo]] · [[Regularization]] · [[Derivative]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Markov Property]]

New topics worth writing: Partition function and the normalisation constraint, Conditional Random Fields, Label bias problem, Contrastive Divergence, Viterbi algorithm and the forward algorithm, Factor graphs and the min-sum algorithm, Graph Transformer Networks, Hinge loss and the max-margin principle, Boltzmann machines, Siamese networks, Time-Delay Neural Networks, Dynamic Time Warping, Hidden Markov Models, Maximum Mutual Information estimation, LVQ2 loss
