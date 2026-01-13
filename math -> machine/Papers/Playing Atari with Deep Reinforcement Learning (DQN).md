---
title: "Playing Atari with Deep Reinforcement Learning (DQN)"
authors: ["Volodymyr Mnih", "Koray Kavukcuoglu", "David Silver", "Alex Graves", "Ioannis Antonoglou", "Daan Wierstra", "Martin Riedmiller"]
year: 2013
arxiv: "1312.5602"
url: https://arxiv.org/abs/1312.5602
priority: Must-Read
read_on: 2026-08-23
tags: [paper, llm, rl, vision]
---
## The Core Idea

Before this paper, if you wanted a reinforcement learning agent to play a video game, a human first had to write code that told the agent what was on the screen — "there is a paddle here, a ball there, an enemy at these coordinates". The learning algorithm then worked on those hand-made features with a linear model. The agent was only as good as the person who designed the features.

This paper removes the human. One convolutional network takes raw pixels in and puts out an estimate of future score for each button press. Nothing else. Same network, same hyperparameters, seven different Atari games.

The reason nobody had done this already is not that people lacked the idea. It is that combining [[Deep Learning]] with Q-learning was known to blow up. Three specific problems:

1. **The target moves.** In supervised learning your label is fixed before training. Here the thing you regress towards is computed by the same network you are updating. You are chasing your own tail.
2. **The data is correlated.** Frame $t$ and frame $t+1$ of Breakout look almost identical. Gradient steps on consecutive frames are nearly the same step, repeated. Neural nets assume roughly independent samples.
3. **The data distribution shifts because of you.** If the network currently thinks "go left" is good, all the training data becomes left-side states. Then it thinks "go right", and the whole distribution flips. This feedback loop can drive parameters to diverge — Tsitsiklis & Van Roy had proved that off-policy learning plus non-linear function approximation can diverge.

The fix that makes the whole thing work is **experience replay**: do not learn from the frame you just saw. Throw every transition into a big buffer of the last one million frames, and sample random minibatches of 32 out of it. That single change kills problems 2 and 3 at once. Correlations are broken because your minibatch is drawn from a million frames of history spread across many episodes. Distribution shift is smoothed because you are training against an average of all your past behaviours, not your current one.

What it unlocks: end-to-end control from vision. The network learns its own features and, crucially, learns features that are useful for *discriminating action values*, not features that a human thought looked important.

> [!NOTE] Experience replay — store transitions $(s_t,a_t,r_t,s_{t+1})$ in a fixed-size buffer and train on uniform random samples from it, rather than on the transition you just experienced. Breaks temporal correlation, reuses each transition many times, and forces off-policy learning. ^experience-replay

## The Methodology

**The problem setup.** The agent sees a screen image $x_t$, picks an action $a_t$ from 4 to 18 legal buttons, gets reward $r_t$ (the change in game score). A single screen is not enough to know what is happening — you cannot see which way the ball is moving from one frame — so the state is technically the whole history $s_t = x_1,a_1,x_2,\dots,x_t$. Treating each history as a distinct state makes this a large but finite [[Markov Decision Process]], so standard RL machinery applies. In practice they cheat and use only the last 4 frames (see preprocessing).

**What is being learned.** The optimal action-value function

$$Q^*(s,a) = \max_\pi \mathbb{E}\!\left[\textstyle\sum_{t'=t}^{T}\gamma^{t'-t} r_{t'} \;\middle|\; s_t=s, a_t=a, \pi\right]$$

which obeys the Bellman equation

$$Q^*(s,a) = \mathbb{E}_{s'}\!\left[r + \gamma \max_{a'} Q^*(s',a') \;\middle|\; s,a\right].$$

Read it plainly: the value of doing $a$ now is the reward you get, plus the discounted value of the best thing you can do next.

**The loss.** A network with weights $\theta$ approximates $Q(s,a;\theta)$. Train it with squared error against a bootstrapped target:

$$L_i(\theta_i) = \mathbb{E}_{s,a\sim\rho}\!\left[\left(y_i - Q(s,a;\theta_i)\right)^2\right], \qquad y_i = \mathbb{E}_{s'}\!\left[r + \gamma\max_{a'}Q(s',a';\theta_{i-1})\right]$$

with [[Derivative#Gradient|gradient]]

$$\nabla_{\theta_i}L_i = \mathbb{E}\!\left[\left(r + \gamma\max_{a'}Q(s',a';\theta_{i-1}) - Q(s,a;\theta_i)\right)\nabla_{\theta_i}Q(s,a;\theta_i)\right].$$

The key detail: $\theta_{i-1}$ inside the target is treated as **fixed** — no gradient flows through it. This is a plain L2 regression ([[Loss, Objectives, and Business Alignment|L2 loss]]) onto a label that you recompute each iteration. Note the 2013 paper does *not* yet have a separate frozen target network; that arrives in the 2015 Nature version. Here the target uses the current $\theta$ in the algorithm box, which is why stability came mostly from replay.

**Architecture — the thing you would sketch on the whiteboard.**

- Input: $84\times84\times4$. Grayscale, downsampled to $110\times84$, cropped to $84\times84$ (square only because the GPU conv kernels from [[ImageNet Classification with Deep CNNs (AlexNet)|AlexNet]] wanted square input). The 4 channels are the last 4 frames stacked — this is how velocity gets into the state.
- Conv 1: 16 filters, $8\times8$, stride 4, [[ImageNet Classification with Deep CNNs (AlexNet)#^relu|ReLU]].
- Conv 2: 32 filters, $4\times4$, stride 2, ReLU.
- Fully connected: 256 ReLU units.
- Output: linear, **one unit per action**.

That last choice matters more than it looks. Previous work (neural fitted Q-iteration) fed the action *in* as an input, so scoring 18 actions cost 18 forward passes. One output head per action gives you all Q-values in a single pass, so cost is constant in the number of actions.

**Training loop (Algorithm 1).** Per timestep: pick $a_t$ $\epsilon$-greedily from the current network; execute it; store $(\phi_t,a_t,r_t,\phi_{t+1})$ in the replay buffer $\mathcal{D}$; sample a random minibatch of 32 from $\mathcal{D}$; set

$$y_j = \begin{cases} r_j & \text{if } \phi_{j+1} \text{ terminal} \\ r_j + \gamma\max_{a'}Q(\phi_{j+1},a';\theta) & \text{otherwise}\end{cases}$$

and take one SGD step on $(y_j - Q(\phi_j,a_j;\theta))^2$ via [[Backpropagation]].

**Hyperparameters that mattered.**

- **Reward clipping.** All positive rewards → $+1$, all negative → $-1$, zeros unchanged. Score scales differ wildly between games; clipping bounds the error derivative so one learning rate works everywhere. The cost is real: the agent cannot tell a 10-point pickup from a 1000-point one.
- RMSProp, minibatch 32.
- $\epsilon$ annealed linearly $1 \to 0.1$ over the first million frames, then held at 0.1.
- 10 million frames of training total; replay buffer holds the most recent 1 million frames.
- **Frame skipping** $k=4$: the agent chooses an action every 4th frame and the action is repeated in between. Emulator steps are far cheaper than network forward passes, so this gives ~4× more game experience per unit of compute.

## Ablation Studies and Experiments

This is a 2013 workshop paper, so there is no proper ablation table. What there is:

**Main results (Table 1, average score, $\epsilon$-greedy with $\epsilon=0.05$):**

| | B.Rider | Breakout | Enduro | Pong | Q*bert | Seaquest | S.Invaders |
|---|---|---|---|---|---|---|---|
| Random | 354 | 1.2 | 0 | −20.4 | 157 | 110 | 179 |
| Sarsa (hand features) | 996 | 5.2 | 129 | −19 | 614 | 665 | 271 |
| Contingency | 1743 | 6 | 159 | −17 | 960 | 723 | 268 |
| **DQN** | **4092** | **168** | **470** | **20** | **1952** | **1705** | **581** |
| Human | 7456 | 31 | 368 | −3 | 18900 | 28010 | 3690 |

DQN beats every prior learning method on all seven games, and beats the human expert on Breakout (168 vs 31), Enduro (470 vs 368) and Pong (20 vs −3). Note that Sarsa and Contingency both got significant prior knowledge — background subtraction, and each of the 128 palette colours treated as its own channel, which is close to handing the agent an object-type mask. DQN gets raw pixels.

**The evolutionary comparison, and why it is not a fair fight.** HyperNEAT (HNeat) reaches 52 on Breakout and 1720 on Space Invaders, but only as a *single best episode* using a deterministic policy — it finds one fixed button sequence that exploits the game's determinism. DQN's *average* score under $\epsilon$-greedy still beats HNeat's *best* on six of seven games. Space Invaders is the exception (DQN average 581, best 1075, vs HNeat 1720).

**Where it fails.** Q*bert (1952 vs 18900 human), Seaquest (1705 vs 28010) and Space Invaders (581 vs 3690). The authors' explanation: these need strategies spanning long time horizons. With $\gamma$ discounting and a 4-frame window, the network cannot represent a plan that pays off thousands of steps later.

**The metric problem, which is the most useful practical finding.** Average episode reward is *extremely* noisy during training — small weight changes cause large changes in which states the policy visits, so the reward curve looks like it is not learning at all. Instead they held out a fixed set of states (collected by a random policy before training) and tracked the average of $\max_a Q(s,a)$ over them. That curve is smooth and monotone. If you are debugging a value-based agent, watch predicted Q, not returns.

**Stability.** Despite zero convergence guarantees with a non-linear approximator, they observed no divergence in any run. This was the surprising result at the time — the received wisdom said this should explode.

**Value function sanity check.** On Seaquest, predicted value jumps when an enemy appears, peaks as the fired torpedo is about to connect, and falls back after the enemy dies. The network has learned event structure, not just a score-correlated blur.

**One game-specific tweak, admitted.** Space Invaders uses $k=3$ instead of $k=4$ because at $k=4$ the blinking lasers are invisible in every frame the agent sees. The single crack in the "no game-specific information" claim.

## Worth Remembering

- **The 2013 vs 2015 gap.** This paper is the seven-game workshop version. It has replay but no **target network** — the frozen copy $\theta^-$ updated every $C$ steps. The Nature 2015 paper adds that, extends to 49 games, and gets much better numbers. If someone says "DQN", they usually mean the 2015 one. The idea that the target must be held fixed is present in the maths here ($\theta_{i-1}$) but not cleanly in the implementation.
- **Limitations the authors name themselves:** uniform sampling from the replay buffer treats every transition as equally informative, and the finite buffer always evicts the oldest. They explicitly point at prioritised sweeping as the fix — which became Prioritised Experience Replay three years later.
- **Reward clipping is a silent behaviour change.** The agent optimises "number of good things" not "score". On games where a rare large reward is the whole point, this actively misleads.
- **Partial observability is handled by a hack.** Stacking 4 frames is a crude fixed-window approximation to the true history. Anything requiring memory beyond ~16 raw frames is invisible to the agent. This is the opening that DRQN (recurrent DQN) later walked through.
- **Off-policy learning is not optional here, it is forced.** The moment you sample from a replay buffer, the data was generated by old weights, so the behaviour policy differs from the greedy target policy. That is exactly why Q-learning (off-policy) is used and SARSA (on-policy) is not.
- **Connection to the wider vault.** The "one output head per input, not one forward pass per candidate" trick is the same efficiency argument as scoring all items in one pass in ranking systems. The predicted-Q-as-training-signal trick is the RL analogue of watching a smooth proxy metric because your business metric is too noisy.
- **Practical caveat if you reimplement:** 10M frames with $k=4$ is only ~2.5M agent decisions per game. This is small by modern standards and the results are correspondingly fragile. Seeds matter a lot.

## Links

Related: [[Markov Decision Process]] · [[Markov Property]] · [[Decision Sciences]] · [[Deep Learning]] · [[Backpropagation]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Loss, Objectives, and Business Alignment]] · [[Derivative]] · [[Uncertainty]] · [[Multi-Agent Reinforcement Learning]] · [[Value Function Factorization]] · [[Dec-POMDP]] · [[Foundational_Reading_Plan]]

New topics worth writing: Q-learning and the Bellman optimality equation, Temporal difference learning, Target networks, Prioritised experience replay, Epsilon-greedy exploration, RMSProp, SARSA vs Q-learning (on-policy vs off-policy), Deadly triad (bootstrapping + off-policy + function approximation), Arcade Learning Environment, Double DQN, Frame skipping and action repeat
