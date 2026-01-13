---
title: "Mastering Diverse Domains through World Models (DreamerV3)"
authors: ["Danijar Hafner", "Jurgis Pasukonis", "Jimmy Ba", "Timothy Lillicrap"]
year: 2023
arxiv: "2301.04104"
url: https://arxiv.org/abs/2301.04104
priority: Must-Read
read_on: 2026-08-23
tags: [paper, rl, vision]
---
## The Core Idea

One reinforcement learning algorithm, one fixed set of hyperparameters, 150+ tasks across 8 very different domains — and it beats the specialised, hand-tuned expert algorithm on almost all of them. That is the claim, and it is the whole point of the paper.

The learning machinery itself is not new. DreamerV3 still does what [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)|Dreamer]] did: learn a **world model** (a neural network that predicts what happens next), then train an actor and critic entirely inside that model's imagination, never touching the real environment during policy learning.

What is new is a set of unglamorous **robustness tricks** that make the same numbers work whether rewards are $0$ or $10^4$, whether they arrive every frame or once per 30 minutes, whether observations are 4 floats or 64×64 images. Before this, moving an RL algorithm from Atari to a robot arm meant weeks of hyperparameter search. The recurring failure was always *scale*: reward magnitude, return magnitude, image reconstruction magnitude, KL magnitude. Each domain needed its own weighting. DreamerV3's answer is to squash every quantity of unknown scale through a symmetric log transform, turn regression into classification over log-spaced bins, and normalise returns by a robust percentile range instead of a standard deviation.

The payoff: applied **out of the box, with default settings**, Dreamer is the first agent to mine a diamond in Minecraft from scratch — no human demonstrations, no curriculum. VPT needed 720 GPUs for 9 days plus contractor-recorded mouse-and-keyboard data. Dreamer used 1 GPU for 9 days.

> [!NOTE] World model
> A learned simulator. It maps (state, action) → (next state, reward, done). Once you have one, you can train a policy on made-up rollouts instead of real experience, which is enormously cheaper. ^world-model

## The Methodology

Three networks, trained at the same time from a replay buffer while the agent acts.

### 1. World model — a Recurrent State-Space Model (RSSM)

$$
\begin{aligned}
\text{Sequence model:} \quad & h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1}) \\
\text{Encoder:} \quad & z_t \sim q_\phi(z_t \mid h_t, x_t) \\
\text{Dynamics predictor:} \quad & \hat z_t \sim p_\phi(\hat z_t \mid h_t) \\
\text{Reward / continue / decoder:} \quad & \hat r_t, \hat c_t, \hat x_t \sim p_\phi(\cdot \mid h_t, z_t)
\end{aligned}
$$

$h_t$ is a deterministic recurrent state (a GRU). $z_t$ is a **stochastic discrete** latent: a vector of 32 categorical variables with 32 classes each, sampled with straight-through gradients (forward pass samples, backward pass pretends the sample was the softmax). The model state is $s_t = \{h_t, z_t\}$ — this is the thing the actor and critic see, and it is designed to be Markov (see [[Markov Property]]), which is why the policy needs no memory of its own.

Encoder/decoder are CNNs for images, MLPs for vectors. Everything else is an MLP.

The loss is three parts, weighted $\beta_{\text{pred}}=1$, $\beta_{\text{dyn}}=1$, $\beta_{\text{rep}}=0.1$:

$$
\begin{aligned}
\mathcal{L}_{\text{pred}} &= -\ln p_\phi(x_t \mid s_t) - \ln p_\phi(r_t \mid s_t) - \ln p_\phi(c_t \mid s_t) \\
\mathcal{L}_{\text{dyn}} &= \max\big(1,\ \mathrm{KL}[\,\mathrm{sg}(q_\phi(z_t \mid h_t, x_t)) \,\|\, p_\phi(z_t \mid h_t)\,]\big) \\
\mathcal{L}_{\text{rep}} &= \max\big(1,\ \mathrm{KL}[\,q_\phi(z_t \mid h_t, x_t) \,\|\, \mathrm{sg}(p_\phi(z_t \mid h_t))\,]\big)
\end{aligned}
$$

This is a sequential [[Auto-Encoding Variational Bayes (VAE)|VAE]]: reconstruct the input, and keep the posterior close to the prior. The two [[KL Divergence|KL]] terms are the *same* KL, split by a stop-gradient $\mathrm{sg}(\cdot)$. $\mathcal{L}_{\text{dyn}}$ pulls the prior toward the posterior (make the predictor better). $\mathcal{L}_{\text{rep}}$ pulls the posterior toward the prior (make the representation easier to predict), and gets only $0.1$ weight so it does not destroy detail.

Two fixes matter here:

- **Free bits.** The $\max(1, \cdot)$ clips each KL at 1 nat ≈ 1.44 bits. Below that the term is switched off and gradients go to reconstruction instead. This is what removes the per-domain tuning: previously, 3D worlds needed a *strong* regulariser (to throw away irrelevant visual clutter) and 2D games with static backgrounds needed a *weak* one (to keep fine pixel detail). Free bits gives you both.
- **1% uniform mixture.** Every categorical distribution is $0.99 \times \text{network output} + 0.01 \times \text{uniform}$. It cannot collapse to deterministic, so the KL cannot spike to infinity.

### 2. Critic

Imagine $T=16$ steps forward from replayed states. Discount $\gamma = 0.997$. Compute bootstrapped $\lambda$-returns:

$$
R^\lambda_t = r_t + \gamma c_t\big((1-\lambda) v_t + \lambda R^\lambda_{t+1}\big), \qquad R^\lambda_T = v_T
$$

The critic is **distributional** — it predicts a whole distribution over returns, not a single number — trained by maximum likelihood $\mathcal{L}(\psi) = -\sum_t \ln p_\psi(R^\lambda_t \mid s_t)$. Extra details:

- Critic loss also applied to *replay* trajectories at weight $\beta_{\text{repval}} = 0.3$ (imagination gets $\beta_{\text{val}} = 1$).
- Regularise the critic toward an EMA of its own weights. Same spirit as the target network in [[Playing Atari with Deep Reinforcement Learning (DQN)|DQN]], but you still bootstrap with the *current* critic.
- **Output weight matrix of the reward predictor and critic initialised to zero.** Random init gives large fake rewards at step 0 and delays learning.

### 3. Actor

Plain REINFORCE, for both discrete and continuous actions:

$$
\mathcal{L}(\theta) = -\sum_{t=1}^{T} \mathrm{sg}\!\left(\frac{R^\lambda_t - v_\psi(s_t)}{\max(1, S)}\right)\log \pi_\theta(a_t \mid s_t) + \eta\,\mathrm{H}[\pi_\theta(a_t \mid s_t)]
$$

with a fixed entropy bonus $\eta = 3\times10^{-4}$ across every domain. The scale $S$ is what makes that fixed $\eta$ possible:

$$
S = \mathrm{EMA}\big(\mathrm{Per}(R^\lambda_t, 95) - \mathrm{Per}(R^\lambda_t, 5),\ 0.99\big)
$$

Two details that look small and are not:

- Percentiles, not min/max, so a single lucky episode does not shrink everything.
- $\max(1, S)$ — **only ever divide, never multiply**. If returns are tiny (sparse reward, nothing found yet), you leave them alone. Dividing by a near-zero standard deviation would blow up pure function-approximation noise, drown the entropy bonus, and kill exploration exactly when you need it most.

### Robust prediction: symlog and twohot

$$
\operatorname{symlog}(x) = \operatorname{sign}(x)\ln(|x|+1), \qquad \operatorname{symexp}(x) = \operatorname{sign}(x)(\exp(|x|)-1)
$$

Symlog is a log that works on negative numbers and is ≈ identity near zero. Vector observations are symlog'd on the way in and as decoder targets. Squared loss is then taken in symlog space, and predictions read out with symexp.

> [!NOTE] symexp twohot loss
> For rewards and returns, instead of regressing a scalar, output logits over 255 bins spaced as $\operatorname{symexp}([-20 \ldots +20])$. Encode the target as a **twohot** vector: all zeros except the two bins straddling the value, weighted linearly by closeness, summing to 1. Train with soft-target [[Cross Entropy|cross entropy]]. Read out as $\hat y = \operatorname{softmax}(f(x))^\top B$, so any continuous value between bins is representable. The key property: gradient magnitude depends only on predicted *probabilities*, never on the size of the target. A reward of $10^4$ produces the same size gradient as a reward of $1$. ^symexp-twohot

This is the answer to the dilemma the authors set up: squared loss on huge targets diverges; absolute or Huber loss (see [[Loss, Objectives, and Business Alignment]]) stalls; running-statistics normalisation as in [[Proximal Policy Optimization Algorithms|PPO]] makes the optimisation non-stationary.

## Ablation Studies and Experiments

**Benchmarks** (fixed hyperparameters throughout, one A100 per agent):

| Benchmark | Budget | Result |
|---|---|---|
| Atari (57 games, sticky actions) | 200M frames | Beats MuZero at a fraction of the compute; beats Rainbow, IQN |
| Atari100k (26 games) | 400K frames | Beats IRIS, TWM, SPR, SimPLe (EfficientZero excluded — it resets levels early) |
| ProcGen (16 games) | 50M frames | Matches tuned PPG, beats Rainbow |
| DMLab (30 tasks) | 100M frames | Beats IMPALA and R2D2+ at **1B** steps → >1000% data-efficiency gain |
| Proprio Control (18 tasks) | 500K steps | New SOTA, beats D4PG, DMPO, MPO |
| Visual Control (20 tasks) | 1M steps | New SOTA, beats DrQ-v2 and CURL (which use data augmentation) |
| BSuite (23 envs, 468 configs) | — | New SOTA, biggest gain in the **reward-scale-robustness** category |
| Minecraft Diamond | 100M steps | **Every** trained agent finds a diamond; no baseline does |

PPO with fixed hyperparameters loses on every single domain. The PPO baseline was not strawmanned — Acme implementation, tuned learning rate / entropy / batch size, IMPALA network, matched $\gamma = 0.997$, and it reproduces the officially published tuned PPO score on ProcGen.

On Minecraft, baselines (tuned IMPALA, Rainbow, PPO) get as far as the iron pickaxe. None reaches a diamond.

**Ablation of robustness techniques** (14 diverse tasks). Ranked by average damage when removed:

1. The **KL objective** of the world model (free bits + the $0.1$ representation weight) — biggest single contributor.
2. **Return normalisation** (percentile range with the $\max(1,S)$ floor).
3. **symexp twohot** regression for reward and value.

The revealing pattern: *each technique is critical on some subset of tasks and irrelevant on others*. That is exactly what you would expect from robustness fixes — they are insurance, not average-case improvements. The average curve understates each one.

**Where the learning signal comes from.** They cut either (a) the reward/value prediction gradients into the representation, or (b) the reconstruction gradients. Dreamer depends **predominantly on the unsupervised reconstruction loss**. This is the opposite of DQN, PPO, and MuZero, which shape their representations almost entirely from task reward. It suggests you could pretrain the world model on reward-free video and bolt an actor on afterwards.

**Scaling.** 6 model sizes from 12M to 400M parameters on Crafter and a DMLab task: performance rises **monotonically** with size, with no hyperparameter changes. Bigger models also need *fewer environment steps* — capacity buys data-efficiency, not just final score. Raising the replay ratio (gradient steps per environment step) does the same. Compare the predictability discussed in [[Scaling Laws for Neural Language Models]].

**What did not work / was rejected:**

- Normalising **advantages** rather than returns (the PPO recipe) — puts a fixed weight on return-maximisation versus entropy no matter whether any reward is reachable, so it amplifies noise under sparse rewards and stalls exploration.
- Normalising by **standard deviation** — collapses when rewards are sparse and the std is near zero, blowing up whatever noise exists.
- **Constrained entropy** targeting a fixed average entropy (SAC-style, MPO-style) — robust, but explores too slowly under sparse rewards and converges lower under dense ones.
- No stable fixed hyperparameters were found for any of these three.
- The **asymmetric** value transform from R2D2 was tried for the critic and was less effective on average.
- Truncating targets (Huber, DQN-style clipping) and PopArt-style weight rescaling both rejected.

## Worth Remembering

- The headline is not "world models are good", it is "the scaling of every loss term was the thing blocking generality all along". Three transforms — symlog, twohot bins, percentile return normalisation — plus free bits, and the same config works from a 4-dimensional cartpole to Minecraft.
- Sparse-reward exploration is protected by one asymmetry: $\max(1, S)$. Only shrink returns, never inflate them. Worth stealing for any policy-gradient method.
- Zero-init on the last layer of the reward head and critic. Cheap, and it removes a real early-training pathology.
- 1% uniform mixture on categoricals to make KL terms behaveable — also cheap, also transferable to any VAE-like model with discrete latents.
- The world model is trained by reconstruction, so it spends capacity on whatever pixels dominate the image, not on what matters for the task. In visually cluttered environments this is a known weakness of reconstruction-based world models, and it is why LeJEPA-style approaches ([[LeJEPA- Provable and Scalable Self-Supervised Learning]]) argue for dropping the decoder. Dreamer's answer is free bits, not removing the decoder.
- Minecraft used the MineRL competition action space with the **block-breaking setting** from prior work, because the raw action space makes it hard for a stochastic policy to hold a key down. That is a small piece of domain accommodation hiding inside "out of the box".
- The actor uses REINFORCE for continuous actions too, not the reparameterised pathwise gradient DreamerV1 used. Slightly higher variance, but one code path for everything.
- Imagination horizon is only $T=16$; everything beyond that is carried by the critic's $\lambda$-return bootstrap. The 30-minute Minecraft episode is never imagined end to end.
- Open follow-up the authors flag: because the representation is shaped mostly by unsupervised reconstruction, you could pretrain a world model on internet video, or train one world model shared across domains.

## Links

Related: [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)]] · [[Proximal Policy Optimization Algorithms]] · [[Playing Atari with Deep Reinforcement Learning (DQN)]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[KL Divergence]] · [[Cross Entropy]] · [[Markov Decision Process]] · [[Markov Property]] · [[Regularization]] · [[Scaling Laws for Neural Language Models]] · [[Loss, Objectives, and Business Alignment]] · [[Uncertainty]] · [[LeJEPA- Provable and Scalable Self-Supervised Learning]]

New topics worth writing: Recurrent State-Space Model (RSSM), Distributional reinforcement learning (C51), Straight-through estimator, Free bits / KL annealing in VAEs, REINFORCE and the policy gradient theorem, Lambda-returns and GAE, Target networks and EMA parameter averaging, MuZero, Model-based vs model-free RL, Entropy regularisation in policy gradients, PopArt normalisation, Replay ratio and sample efficiency
