---
title: "Dream to Control: Learning Behaviors by Latent Imagination (Dreamer)"
authors: ["Danijar Hafner", "Timothy Lillicrap", "Jimmy Ba", "Mohammad Norouzi"]
year: 2019
arxiv: "1912.01603"
url: https://arxiv.org/abs/1912.01603
priority: Must-Read
read_on: 2026-08-23
tags: [paper, rl, vision]
---
## The Core Idea

Dreamer is a reinforcement learning agent that learns to control a robot from camera images without practising much in the real environment. It practises inside its own head.

Two ideas stack together.

**One: shrink the world into a small vector.** A learned "world model" compresses each $64\times64\times3$ image into a 30-dimensional latent state $s_t$ that obeys the [[Markov Property]] — the state contains everything you need to predict the next state. Once you have that, you can roll out thousands of imagined futures in parallel, cheaply, because you never render an image again. You just step a small recurrent network. This part was already done by PlaNet (the authors' earlier paper).

**Two: learn the policy by differentiating through the dream.** This is the new part. The world model is a neural network, so the whole imagined trajectory — action → next state → reward → value — is one big differentiable computation graph. So you can ask: "how should I nudge my policy weights so the imagined return goes up?" and get an exact answer with [[Backpropagation]], instead of estimating it from noisy sampled returns like REINFORCE or [[Proximal Policy Optimization Algorithms|PPO]] do.

Why did this not exist before? Two reasons. Model-based agents mostly used *derivative-free* search (random shooting, CEM) inside their model, because they did not trust the model's gradients — a slightly wrong model has very wrong gradients over long rollouts. And they only optimised rewards inside a fixed imagination window $H$, which makes the agent short-sighted: if the reward for standing up arrives after step $H$, the agent never sees it.

Dreamer fixes the short-sightedness by learning a **value model** in the latent space that summarises "everything after the horizon". Then a horizon of only 15 steps is enough to learn behaviours that pay off far later. That combination — analytic gradients *plus* a bootstrapped value beyond the horizon — is the whole paper.

What it unlocks: on 20 DeepMind Control Suite tasks from pixels, average score 823 after $5\times10^6$ environment steps, versus 786 for D4PG (a strong model-free agent) after $10^8$ steps. That is 20× fewer environment interactions, and 3 hours of GPU time per $10^6$ steps versus 11 hours for PlaNet's online planning and 24 hours for D4PG.

> [!NOTE] Latent imagination
> Rolling the dynamics forward purely in compressed state space $s_\tau$, never decoding back to pixels. Cheap enough to imagine thousands of futures at once, which is what makes gradient-based policy learning inside a model practical. ^latent-imagination

## The Methodology

Three loops run forever, interleaved: learn the model from stored data, learn the behaviour inside the model, act in the world to collect more data.

**The world model.** Three pieces (plus a decoder used only for training signal):

$$\begin{aligned}\text{Representation: } &p_\theta(s_t \mid s_{t-1}, a_{t-1}, o_t)\\ \text{Transition: } &q_\theta(s_t \mid s_{t-1}, a_{t-1})\\ \text{Reward: } &q_\theta(r_t \mid s_t)\\ \text{Observation: } &q_\theta(o_t \mid s_t)\end{aligned}$$

The representation model looks at the image and updates the state; the transition model must guess the next state *without* the image. This is exactly a nonlinear, action-conditioned [[Kalman Filter]] — a filter step and a predict step — with neural networks instead of matrices. Implemented as the RSSM from PlaNet: a deterministic GRU carrier plus a stochastic 30-d diagonal Gaussian per step. Encoder and decoder are the DCGAN-style conv nets from World Models; everything else is 3 dense layers of width 300 with ELU.

Training objective is the ELBO / variational information bottleneck, same shape as in [[Auto-Encoding Variational Bayes (VAE)]]:

$$\mathcal{J}_{\text{REC}} = \mathbb{E}_p\Big[\sum_t \underbrace{\ln q(o_t\mid s_t)}_{\text{image}} + \underbrace{\ln q(r_t\mid s_t)}_{\text{reward}} - \beta\,\underbrace{\mathrm{KL}\big(p(s_t\mid s_{t-1},a_{t-1},o_t)\,\|\,q(s_t\mid s_{t-1},a_{t-1})\big)}_{\text{make the prediction match the filter}}\Big]$$

That [[KL Divergence]] term is doing real work: it punishes the model for needing fresh pixel information at each step, which pushes it to carry information forward and learn long-range dynamics. $\beta = 1$, but the KL is clipped below 3 free nats (do not penalise below that floor) to stop posterior collapse.

**The behaviour.** Two more networks, both dense, both living only in latent space:

- Action model $q_\phi(a_\tau \mid s_\tau)$ — a tanh-squashed Gaussian, $a_\tau = \tanh(\mu_\phi(s_\tau) + \sigma_\phi(s_\tau)\epsilon)$, $\epsilon\sim\mathcal{N}(0,I)$. The tanh mean is scaled by 5 so the policy can saturate at the action limits.
- Value model $v_\psi(s_\tau)$ — predicts discounted imagined return from $s_\tau$.

The [[Auto-Encoding Variational Bayes (VAE)#^reparameterisation-trick|reparameterisation trick]] is why the action sample is differentiable. Discrete actions use straight-through gradients instead.

**The imagination rollout.** Take a batch of real sequences from the replay dataset (50 sequences × 50 steps), encode them to get real states $s_t$. From *each* of those 2500 states, branch off and imagine forward $H=15$ steps using transition model, reward model, and current policy. No images involved.

**The value target.** A $\lambda$-return computed inside the dream. First the $k$-step estimate:

$$\mathrm{V}_N^k(s_\tau) = \mathbb{E}\Big[\sum_{n=\tau}^{h-1}\gamma^{n-\tau}r_n + \gamma^{h-\tau}v_\psi(s_h)\Big],\quad h=\min(\tau+k,\, t+H)$$

then the exponentially weighted mix:

$$\mathrm{V}_\lambda(s_\tau) = (1-\lambda)\sum_{n=1}^{H-1}\lambda^{n-1}\mathrm{V}_N^n(s_\tau) + \lambda^{H-1}\mathrm{V}_N^H(s_\tau)$$

with $\gamma = 0.99$, $\lambda = 0.95$.

**The two losses.**

$$\max_\phi \mathbb{E}\Big[\sum_{\tau=t}^{t+H}\mathrm{V}_\lambda(s_\tau)\Big],\qquad \min_\psi \mathbb{E}\Big[\sum_{\tau=t}^{t+H}\tfrac{1}{2}\big\|v_\psi(s_\tau) - \mathrm{V}_\lambda(s_\tau)\big\|^2\Big]$$

The critic regresses the target with the gradient stopped through the target. The actor gets its gradient *through the dynamics*: $\mathrm{V}_\lambda$ depends on rewards and values, which depend on imagined states, which depend on imagined actions, which depend on $\phi$. One long chain rule, all analytic. Compare to DDPG/SAC which also use reparameterised gradients but only through a one-step Q-function — they never differentiate through a transition. The world model is frozen during this step.

**Hyperparameters that mattered.** Adam, learning rates $6\times10^{-4}$ (world model), $8\times10^{-5}$ (actor and critic). Gradient norm clipped at 100. Action repeat fixed at $R=2$ for every task (PlaNet tuned it per task; Dreamer does not). 5 random seed episodes to start; then alternate 100 gradient steps with collecting 1 episode, acting with the distribution's mode plus $\mathcal{N}(0, 0.3)$ noise. For Atari/DMLab: $H=10$, $\beta=0.1$, tanh-bounded rewards, $\epsilon$-greedy annealed $0.4\to0.1$ over 200k steps, and a learned discount head (binary classifier over $\{0,\gamma\}$) so imagined trajectories get down-weighted where the episode probably ended.

## Ablation Studies and Experiments

**Main benchmark: 20 DMControl tasks from pixels, 5 seeds.**

| | A3C (proprio, $10^8$) | D4PG (pixels, $10^8$) | PlaNet ($5\times10^6$) | Dreamer ($5\times10^6$) |
|---|---|---|---|---|
| Acrobot Swingup | 41.9 | 91.7 | 3.2 | **365.3** |
| Cartpole Swingup Sparse | 179.8 | 482.0 | 0.6 | **812.2** |
| Cheetah Run | 213.9 | 523.8 | 496.1 | **894.6** |
| Hopper Hop | 0.5 | 242.0 | 0.4 | **369.0** |
| Hopper Stand | 27.9 | 929.9 | 6.0 | 923.7 |
| Quadruped Walk | – | – | 238.9 | **931.6** |
| Finger Spin | 129.4 | **985.7** | 495.3 | 498.9 |
| Walker Walk | 311.0 | 968.3 | 944.7 | 961.7 |
| **Average** | 243.7 | 786.3 | 333.0 | **823.4** |

Note Finger Spin — Dreamer loses badly there (499 vs 986). It is not a clean sweep.

**Ablation 1: does the value model matter?** Compare Dreamer against (a) an action model trained on $\mathrm{V}_R$, the plain sum of rewards inside the horizon with no bootstrap, and (b) PlaNet's online planning (CEM in latent space, no policy network). Sweep the imagination horizon.

The result: without the value model, performance depends strongly on $H$ and collapses on long-credit-assignment tasks. Acrobot Swingup and Hopper Hop need the value model; both alternatives fail there. Walker (reactive, dense reward) is solved by all three — that is the tell. **The value model is the component doing the work on hard tasks; the analytic gradient is doing the work on speed.** With $H=20$, Dreamer beats both alternatives on 16 of 20 tasks with 4 ties.

**Ablation 2: which representation objective?** The behaviour learner is agnostic to how the world model is trained, so they swapped in three:

- **Pixel reconstruction** — best on most tasks.
- **Contrastive / InfoNCE** — replace the decoder $q(o_t\mid s_t)$ with an encoder-style $q(s_t\mid o_t)$ and subtract $\ln\sum_{o'}q(s_t\mid o')$ over the minibatch to stop collapse. Solves roughly half the tasks. Avoids pixel prediction but extracts less information (the mutual-information bound is loose, per McAllester & Statos).
- **Reward prediction only** — did **not work at all** in their experiments. Without a signal tying latents to what the image actually shows, the model does not learn a useful state.

That last one is the most useful negative result: MuZero-style pure value/reward-grounded representations need far more data than $5\times10^6$ steps.

**Things they explicitly found unnecessary.** Latent overshooting (a multi-step KL trick from PlaNet), an entropy bonus for the actor, and target networks for the critic. All three are standard stabilisers elsewhere; none were needed here.

**Action repeat sweep (Appendix F).** RL is famously sensitive to control frequency. $R=2$ was best across tasks, so they fixed it globally instead of tuning per environment — a small but honest fairness improvement over PlaNet.

**Discrete actions.** Dreamer learns something on a subset of Atari and on the DMLab object-collection level, but the authors say plainly that world-model-only agents are "not yet competitive" there and flag representation learning as the bottleneck.

## Worth Remembering

- The imagined [[Markov Decision Process]] is *fully observed* even though the real task is a POMDP. That is the whole point of the latent state: the messy partial observability gets absorbed into the filter, and behaviour learning happens in a clean MDP. This is the same move as the [[Kalman Filter]] belief state.
- Every imagined trajectory starts from a *real* encoded state drawn from replay. The model is never asked to imagine from scratch, only to extrapolate 15 steps from somewhere it has actually been. This is a big part of why the model errors do not explode.
- Value gradients through learned dynamics were known to be brittle at scale (Parmas et al., "curse of chaos"). Dreamer sidesteps the chaos partly by keeping $H$ short (15) and letting the critic cover the rest — the gradient path is never longer than 15 transitions.
- **Limitation the authors admit:** visual complexity. Reconstruction-based world models spend capacity on pixels regardless of whether those pixels matter. On Atari, where a lot of the screen is irrelevant clutter, this hurts. The contrastive alternative is the escape hatch but underperforms today. (Later work — DreamerV2/V3, and things like [[LeJEPA- Provable and Scalable Self-Supervised Learning]] — is all about this exact problem.)
- **Exploration is an afterthought.** Fixed $\mathcal{N}(0, 0.3)$ Gaussian noise on the action. Any task genuinely requiring directed exploration is out of scope here.
- Practical caveat if you implement it: the free-nats KL clipping is not optional decoration. Without it the stochastic latent collapses and the transition model learns nothing.
- Nice framing to keep: this is Dyna (Sutton, 1991) with the model made differentiable. The 1991 insight was "learn a model, train the policy on model rollouts". The 2019 addition is "and since the model is a neural net, do not sample — differentiate."
- Compare the credit-assignment story to [[Playing Atari with Deep Reinforcement Learning (DQN)]]: DQN propagates value one bootstrap step at a time through millions of replay samples. Dreamer gets 15 steps of exact multi-step credit in a single backward pass.

## Links

Related: [[Markov Decision Process]] · [[Markov Property]] · [[Kalman Filter]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[KL Divergence]] · [[Backpropagation]] · [[Proximal Policy Optimization Algorithms]] · [[Playing Atari with Deep Reinforcement Learning (DQN)]] · [[Beliefs]] · [[Uncertainty]] · [[Derivative]] · [[Vector Jacobian Product]] · [[LeJEPA- Provable and Scalable Self-Supervised Learning]] · [[Decision Sciences]] · [[Regularization]]

New topics worth writing: World models and Dyna-style model-based RL, Recurrent State Space Model (RSSM), PlaNet and latent online planning (CEM), TD($\lambda$) and $\lambda$-returns, Soft Actor-Critic and reparameterised actor-critics, DDPG/D4PG, Information bottleneck objective, InfoNCE and contrastive predictive coding, Straight-through gradient estimator, Free bits / free nats in VAEs, DeepMind Control Suite, MuZero.
