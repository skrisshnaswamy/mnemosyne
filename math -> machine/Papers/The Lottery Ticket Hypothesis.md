---
title: "The Lottery Ticket Hypothesis"
authors: ["Jonathan Frankle", "Michael Carbin"]
year: 2018
arxiv: "1803.03635"
url: https://arxiv.org/abs/1803.03635
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, optimization, vision, scaling]
---
## The Core Idea

Pruning a trained network — throwing away 90% of its weights — has been standard since the 1990s. The pruned network still works. But everyone believed the *pruned shape* was only usable as a compression artefact: if you took that same sparse shape, re-randomised the weights, and trained it from scratch, it trained worse than the big dense network. So pruning made inference cheap but never made training cheap.

Frankle and Carbin found the missing ingredient. Do not re-randomise. **Rewind the surviving weights to the exact random values they had at step 0**, before any training happened. That sparse network — same shape, *original* initial values — trains to the same or better accuracy than the full network, in the same or fewer steps.

> [!NOTE] Winning ticket
> A subnetwork $f(x; m \odot \theta_0)$ — a binary mask $m$ applied to the *original* random initialisation $\theta_0$ — that trains in isolation to match the full network's accuracy in at most the same number of iterations, with $\lVert m \rVert_0 \ll |\theta|$. ^winning-ticket

The hypothesis: a big randomly-initialised network is a **lottery** with many tickets in it. Some sparse subnetwork happened to get a lucky combination of connections *and* starting values. Training the big network is really just [SGD](#) finding and training that lucky subnetwork. Overparameterisation helps not because you need all the weights, but because more weights means more lottery tickets, so a better chance one of them is a winner.

Why this did not exist before: the rewind step is one line of code, and nobody thought to try it because the community had already concluded "sparse architectures are hard to train." The conclusion was right; the experiment was just missing the initialisation control.

What it unlocks: a concrete existence proof that sparse trainable networks are hiding inside dense ones, which reframes questions about [[Understanding Deep Learning Requires Rethinking Generalization|why overparameterised nets generalise]] and [[Reconciling Modern ML Practice and the Bias-Variance Trade-off|the role of model size]].

## The Methodology

The whole method is four steps.

1. Randomly initialise $f(x; \theta_0)$, with $\theta_0 \sim \mathcal{D}_\theta$ (Gaussian Glorot, i.e. [[Understanding the difficulty of training deep feedforward networks (Xavier init)|Xavier init]]).
2. Train for $j$ iterations to get $\theta_j$.
3. Prune $p\%$ of the smallest-magnitude weights in $\theta_j$. This gives a mask $m \in \{0,1\}^{|\theta|}$.
4. **Reset** the surviving weights to their values in $\theta_0$. You now hold $f(x; m \odot \theta_0)$.

Notation: $P_m = \lVert m \rVert_0 / |\theta|$ is the fraction of weights still alive. $P_m = 21\%$ means 79% pruned.

**One-shot vs iterative.** Doing steps 2–4 once is *one-shot pruning*. The paper mostly uses *iterative pruning*: repeat train → prune → reset for $n$ rounds, removing 20% of the *surviving* weights each round (10% for conv layers in some nets, half rate for output layers). Iterative finds much smaller winning tickets, at the cost of training the network 15–30 times.

Appendix B tests a variant where you keep training from the trained weights between rounds instead of resetting each round. Resetting every round (Strategy 1) wins everywhere.

**Pruning is unstructured and magnitude-based** — exactly the Han et al. (2015) heuristic. Small $|w|$ dies. Nothing clever.

**Layer-wise vs global.** For small nets, prune each layer separately at its own rate. For VGG-19 and Resnet-18, prune *globally* — rank all conv weights together. Reason: VGG-19's first conv layer has 1,728 parameters and its last has 2.35 million. Pruning both at 20% makes the tiny layer a bottleneck that dies out first. Global pruning gets VGG-19 winning tickets down to $P_m = 1.5\%$ vs $6.9\%$ for layer-wise.

**Architectures and settings** (all vision, all small):

| Net | Data | Params | Optimiser | Iterations |
|---|---|---|---|---|
| Lenet-300-100 | MNIST | 266K | Adam 1.2e-3 | 50K |
| Conv-2 / 4 / 6 (mini-VGG) | CIFAR10 | 4.3M / 2.4M / 1.7M | Adam 2e-4 / 3e-4 / 3e-4 | 20K / 25K / 30K |
| Resnet-18 | CIFAR10 | 271K | SGD 0.1, mom 0.9 | 30K |
| VGG-19 | CIFAR10 | 20M | SGD 0.1, mom 0.9 | 112K |

The deeper two use [[Batch Normalization|batchnorm]], weight decay 1e-4, step learning-rate decay, and data augmentation.

**"Learning speed" is measured by early stopping**: the iteration of minimum validation loss. Test accuracy is then reported *at that iteration*. This is a proxy — validation loss dips, bottoms out, then rises as the net overfits, and they call the bottom "done learning."

**The control experiment that carries the whole paper**: keep the mask $m$, but sample a fresh $\theta_0' \sim \mathcal{D}_\theta$. Train $f(x; m \odot \theta_0')$. If this matched the winning ticket, structure alone would explain everything and initialisation would be irrelevant. It does not match.

## Ablation Studies and Experiments

**Lenet / MNIST, iterative pruning.** Winning tickets get *faster and more accurate* as you prune, up to a point:

- At $P_m = 21.1\%$: early stopping happens **38% earlier** than the full network.
- At $P_m = 13.5\%$: test accuracy is **+0.3 percentage points** over the original.
- At $P_m = 3.6\%$: performance falls back to the original network's level. Below that it degrades.
- At $P_m = 21\%$, winning ticket vs random reinit: **2.51× faster** to minimum validation loss, and **0.5 pp** more accurate.

Random reinit never improves. It gets monotonically slower and less accurate from the first pruning round. Accuracy drops off at $P_m = 21.1\%$ for reinit vs $2.9\%$ for the winning ticket.

**Generalisation, not just optimisation.** At early stopping, training accuracy also rises with pruning, so you might say the tickets just optimise better. But at iteration 50,000 training accuracy is ~100% for essentially all networks with $P_m \geq 2\%$, and winning tickets *still* hold a test accuracy advantage of up to 0.35 pp. Same train accuracy, better test accuracy → smaller generalisation gap. The authors call the shape an **Occam's Hill**: too big overfits, too small underfits, best in the middle.

**Conv-2/4/6 on CIFAR10.** The effect is stronger than on MNIST.

| Net | Best speedup | Best accuracy gain |
|---|---|---|
| Conv-2 | 3.5× ($P_m=8.8\%$) | +3.4 pp ($P_m=4.6\%$) |
| Conv-4 | 3.5× ($P_m=9.2\%$) | +3.5 pp ($P_m=11.1\%$) |
| Conv-6 | 2.5× ($P_m=15.1\%$) | +3.3 pp ($P_m=26.4\%$) |

All three stay above the original test accuracy down to $P_m > 2\%$.

Interesting wrinkle: for Conv-2 and Conv-4, random reinit accuracy *initially holds steady or improves* at moderate sparsity. So at mild pruning the sparse **structure** alone is worth something; only at extreme sparsity does initialisation become necessary.

**Dropout stacks with it.** With [[Dropout- A Simple Way to Prevent Overfitting|dropout]] at 0.5, initial accuracy rises 2.1/3.0/2.4 pp on Conv-2/4/6, and iterative pruning adds a *further* 2.3/4.6/4.7 pp on top. Dropout can be read as training an ensemble of subnetworks; the lottery ticket view says one of those subnetworks is the winner, and the two ideas do not fight.

### What did not work

**Learning rate 0.1 on the deep nets — the biggest failure.** On VGG-19 at lr 0.1, iterative pruning finds *nothing*: the reset tickets do no better than random reinit. At lr 0.01 the lottery pattern returns (within 1 pp of original while $P_m \geq 3.5\%$) but the network's *ceiling* is lower because the learning rate is worse. Resnet-18 behaves the same way.

The fix is **linear warmup**. Ramping lr from 0 to 0.1 over $k=10000$ iterations lets VGG-19 find genuine winning tickets down to $P_m = 1.5\%$, and beats the unpruned baseline by ~1 pp. For Resnet-18, warmup with $k=20000$ at lr 0.03 reaches 90.5% at $P_m = 27.1\%$, matching the unpruned lr-0.1 baseline — but even with warmup they **could not find winning tickets at lr 0.1 for Resnet-18 at all**. This is a real limitation, not a footnote: the method is fragile to the exact optimiser schedule.

**Pruning at iteration 0 fails.** The winning ticket weights tend to have large *initial* magnitude, so you might guess you can just prune the smallest initial weights and skip training entirely. Networks found this way perform *worse than random reinit*. Training data is genuinely needed to find the mask.

**Resampling from the ticket's own distribution fails.** Winning ticket initialisations are bimodal (peaks either side of 0) and asymmetric, very unlike the Glorot Gaussian. Sampling fresh weights from that empirical distribution $\mathcal{D}_m$ performs barely better than plain random reinit. So it is not the *distribution* that matters — it is the *specific weight in the specific place*.

**Are ticket weights already near their final values?** The obvious hypothesis: winning tickets are lucky because they start close to the optimum. The data says the opposite. Winning ticket weights move *further* during training than non-ticket weights, and are more likely to move *away* from zero. So the benefit is about being well-placed in the loss landscape for SGD to work with, not about being pre-solved.

**Noise robustness.** Adding Gaussian noise of $0.5\sigma$ to ticket initialisations barely hurts. Even $3\sigma$ noise still beats random reinit. Tickets are not knife-edge fragile.

**Which layers matter (Conv nets).** Pruning only convolutions raises accuracy and speeds learning; pruning only FC layers hurts. But FC layers hold 99% / 89% / 35% of Conv-2/4/6's parameters, so conv-only pruning barely shrinks the model.

**Random reinit vs random sparsity.** These are nearly identical for all conv nets. Only on Lenet/MNIST does reinit beat a random sparse mask — plausibly because MNIST digits are centred, so edge pixels are useless and the learned mask captures that.

**Hyperparameter sweeps (3,000+ Lenet runs).** The pattern survives SGD, SGD+momentum, Adam, Gaussian inits with various $\sigma$, and network widths. Slower pruning rates (10%, 20% per round) beat fast ones (40%+); 20% was chosen as the compute/quality compromise. Larger initial networks yield higher-accuracy tickets at any given parameter count, but not faster ones, and all sizes return to baseline accuracy at roughly 9,000–15,000 surviving weights.

## Worth Remembering

- **The result is an existence proof, not an algorithm.** Finding the ticket requires training the full network 15–30 times. You cannot use this to make training cheaper *today*. The authors say so plainly.
- **Only MNIST and CIFAR10, only vision classification.** ImageNet was explicitly out of reach on compute. (The follow-up literature found that at ImageNet scale you must rewind to iteration $k$ of training rather than iteration 0 — "late rewinding" — which weakens the pure "lottery at initialisation" story.)
- **Unstructured sparsity.** The tickets are scattered zeros, useless for real hardware speedups without structured pruning or sparse kernels.
- **The Liu et al. (2019) tension.** That paper argued pruned architectures train fine when randomly reinitialised, apparently contradicting this. Frankle and Carbin reproduce their result at ≤80% sparsity on the same VGG-19 setup, and reconcile it: below some sparsity, an overparameterised net is still overparameterised enough that any init works. Beyond that (98.5% pruned) initialisation starts to matter a lot.
- **The untested conjecture** worth holding onto: SGD *itself* seeks out and trains a well-initialised subnetwork, and dense nets are easy to train because they contain many candidate tickets. Nothing in the paper demonstrates this. It is a hypothesis about what optimisation is doing.
- Practical caveat if you try to reproduce: the learning-rate sensitivity on deep nets is severe, warmup is not optional, and global pruning is not optional either. Getting these wrong makes the effect vanish entirely, which is exactly what happened to everyone before 2018.
- Connection worth chasing: the improved-generalisation result sits next to compression-based generalisation bounds (Zhou et al. 2018, Arora et al. 2018) — those say compressible networks generalise better; this says big networks *contain* the compressed thing already.

## Links

Related: [[Understanding Deep Learning Requires Rethinking Generalization]] · [[Reconciling Modern ML Practice and the Bias-Variance Trade-off]] · [[Deep Double Descent- Where Bigger Models and More Data Hurt]] · [[Dropout- A Simple Way to Prevent Overfitting]] · [[Understanding the difficulty of training deep feedforward networks (Xavier init)]] · [[Delving Deep into Rectifiers (He init, PReLU)]] · [[Adam- A Method for Stochastic Optimization]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Gradient-Based Learning Applied to Document Recognition (LeNet)]] · [[Distilling the Knowledge in a Neural Network]] · [[Batch Normalization]] · [[Regularization]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Backpropagation]]

New topics worth writing: Magnitude pruning, Structured vs unstructured sparsity, Learning rate warmup, Weight rewinding and Stabilizing the Lottery Ticket Hypothesis, Optimal Brain Damage, Minimum description length and compression bounds, Occam's Hill, Sparse training methods (RigL, SET), Intrinsic dimension of objective landscapes
