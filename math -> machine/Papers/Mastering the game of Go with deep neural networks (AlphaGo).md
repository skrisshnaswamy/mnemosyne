---
title: "Mastering the game of Go with deep neural networks (AlphaGo)"
authors: ["Silver et al."]
year: 2016
url: https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, rl]
---
## The Core Idea

Go was the last board game where computers were bad. The reason is arithmetic: about $b \approx 250$ legal moves per turn, over $d \approx 150$ turns, so roughly $250^{150}$ possible move sequences. You cannot search that. Chess programs like Deep Blue got around a similar (smaller) problem with two tricks — cut the search short and score the position with a hand-written evaluation function, and prune bad branches early. Nobody could write a good evaluation function for Go. Human experts describe positions with words like "thick" and "influence", not with countable material.

The insight: **let two convolutional networks replace the two hand-written heuristics, and bolt them onto Monte Carlo tree search.**

- A **policy network** $p(a|s)$ gives a probability over moves. It narrows the *breadth* of the search — instead of considering 250 moves, the search mostly considers the handful the network thinks a strong human would play.
- A **value network** $v_\theta(s)$ outputs one number: "who is winning from here?". It cuts the *depth* — you stop searching at some position and ask the network, instead of playing the game out.

Neither idea is new on its own. What was new is that the networks are deep convolutional nets (13 layers, ~192 filters) rather than the linear feature combinations everyone had used before, and that they are trained by a four-stage pipeline that ends in self-play reinforcement learning. Deep nets are ~1000× slower to evaluate than a linear model, so the whole search engine had to be rebuilt asynchronously to hide that latency behind GPU batching.

What it unlocked: AlphaGo beat every other Go program 494 games out of 495, and beat Fan Hui, the European champion and a 2-dan professional, 5–0 in formal games. That was expected to take another decade.

> [!NOTE] Monte Carlo tree search (MCTS)
> A search method that builds a lopsided tree. Each "simulation" walks down from the root picking promising moves, hits a leaf, gets an estimate of who is winning there, and pushes that estimate back up the path. Good branches get visited more, so they get searched deeper. It needs no evaluation function — random playouts to the end of the game work — but it needs a lot of them. ^mcts

## The Methodology

Four learned components, trained in order.

**1. SL policy network $p_\sigma$ — copy human moves.**
Input is a $19\times19\times48$ stack of binary feature planes. The features are near-raw: stone colour (3 planes), liberties (8 planes, one-hot for 1,2,…,≥8), how many turns since a stone was played (8), how many stones a move would capture (8), plus one tactical feature (ladder search). Layer 1 is $5\times5$ conv, $k=192$ filters, ReLU. Layers 2–12 are $3\times3$ conv, ReLU. Layer 13 is a $1\times1$ conv with a per-position bias, then softmax over the 361 points. This is just [[Cross Entropy|classification]] over board points:

$$\Delta\sigma \propto \frac{\partial \log p_\sigma(a|s)}{\partial \sigma}$$

Trained on 29.4M positions from 160,000 KGS games by 6–9 dan players, augmented with all 8 rotations/reflections. SGD, step size 0.003 halved every 80M steps, **no momentum**, mini-batch 16, asynchronous on 50 GPUs via DistBelief, 340M steps, ~3 weeks. Result: **57.0% test accuracy** at predicting the human move (55.7% with raw board + move history only). Previous best from other groups was 44.4%.

**2. Rollout policy $p_\pi$ — the cheap one.**
A linear softmax over $3\times3$ stone-colour/liberty patterns around the candidate move, plus 12-point diamond "response" patterns around the opponent's last move. Only **24.2%** accurate, but takes **2 µs** per move instead of 3 ms for the big net. Trained on 8M Tygem positions.

**3. RL policy network $p_\rho$ — play to win, not to imitate.**
Same architecture, weights initialised from $\sigma$. Self-play against a *randomly sampled earlier version of itself* (a fresh snapshot is added to the opponent pool every 500 iterations) — this prevents the net from overfitting to one opponent. Reward is zero everywhere except the end: $z_t = \pm 1$. Update is plain [[Simple Statistical Gradient-Following Algorithms (REINFORCE)|REINFORCE]]:

$$\Delta\rho \propto \sum_t \frac{\partial \log p_\rho(a_t|s_t)}{\partial \rho}\,(z_t - v(s_t))$$

with the baseline $v(s_t)$ set to zero on the first pass and to the value network on the second. 10,000 mini-batches of 128 parallel games, 50 GPUs, one day. Result: beats the SL policy **80%** of the time, and — *sampling moves directly from the softmax, with no search at all* — beats Pachi (which runs 100,000 simulations per move) **85%** of the time. The previous supervised-only nets managed 11%.

**4. Value network $v_\theta$ — who is winning.**
Same conv trunk plus an extra "colour to play" plane, an extra conv layer, a 256-unit fully connected ReLU layer, and a single $\tanh$ output. Regression on outcome:

$$\Delta\theta \propto \frac{\partial v_\theta(s)}{\partial \theta}\,\big(z - v_\theta(s)\big)$$

The data generation is the clever bit. Training on positions from the KGS human games **failed**: successive positions differ by one stone but share the same label, so the net memorised games. Train MSE 0.19, test MSE **0.37** — it learned nothing generalisable. Fix: build 30 million positions where **each one comes from a different self-play game**. Each game is generated in three phases — sample $U \sim \text{unif}\{1,450\}$, play moves $1\ldots U-1$ from the SL policy (noisier, more diverse), play move $U$ **uniformly at random**, then play the rest from the RL policy — and only the single pair $(s_{U+1}, z_{U+1})$ is kept. Train/test MSE became 0.226/0.234. 50M mini-batches of 32, 50 GPUs, one week.

**The search (APV-MCTS).**
Each edge $(s,a)$ stores prior $P$, visit counts $N_v, N_r$, and accumulated values $W_v, W_r$. Selection uses a PUCT rule:

$$a_t = \arg\max_a \big(Q(s,a) + u(s,a)\big), \qquad u(s,a) = c_{\text{puct}} P(s,a)\frac{\sqrt{\sum_b N_r(s,b)}}{1 + N_r(s,a)}$$

Early on this follows the policy network's prior; as visits accumulate it follows $Q$. Same shape as a [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)|UCB]] bonus.

At a leaf $s_L$, evaluation happens **two ways at once** and gets blended:

$$V(s_L) = (1-\lambda)\, v_\theta(s_L) + \lambda\, z_L$$

where $z_L$ is the result of one fast rollout to the end of the game with $p_\pi$. $\lambda = 0.5$. The final action value is the corresponding weighted average of the two running means.

A node is expanded once $N_r(s,a) > n_{\text{thr}} = 40$, where $n_{\text{thr}}$ is tuned **dynamically at runtime** so that positions enter the policy-network queue at exactly the rate the GPUs drain it. Fresh nodes get placeholder priors from a cheap tree policy, replaced atomically when the GPU returns $p_\sigma^\beta(\cdot|s')$ with softmax temperature $\beta = 0.67$. Threads that are mid-simulation apply a **virtual loss** of $n_{vl}=3$ to the edges they are traversing, so other threads go elsewhere; it is removed on backup. All updates lock-free.

Other hyperparameters: $c_{\text{puct}}=5$. Final move = **most visited**, not highest value (less sensitive to a single lucky rollout). Resigns when $\max_a Q < -0.8$. Subtree is reused between moves. Symmetries are exploited at run time by picking one of the 8 dihedral transforms *at random* per evaluation and letting the search average over them.

Hardware: 40 search threads, 48 CPUs, 8 GPUs on one machine; the distributed version used 40 threads, 1202 CPUs, 176 GPUs.

## Ablation Studies and Experiments

**Which component carries the weight (Extended Data Table 7, Elo, 5 s/move, single machine):**

| Variant | Policy net prior | Value net | Rollouts | Elo |
|---|---|---|---|---|
| $\alpha_{rvp}$ full | ✓ | ✓ | ✓ | **2890** |
| $\alpha_{rp}$ rollouts only ($\lambda=1$) | ✓ | — | ✓ | 2416 |
| $\alpha_{vp}$ value only ($\lambda=0$) | ✓ | ✓ | — | 2177 |
| $\alpha_{rv}$ no policy prior | — | ✓ | ✓ | 2077 |
| $\alpha_{v}$ value only, no prior | — | ✓ | — | 1655 |
| $\alpha_{r}$ rollouts only, no prior | — | — | ✓ | 1457 |
| $\alpha_{p}$ policy net alone, **no search** | ✓ | — | — | 1517 |

Read this carefully. Mixing beats either evaluator alone by ~470 Elo, and the mixed version won ≥95% of games against every other variant. The two evaluators are genuinely complementary: the value net approximates games played by the strong-but-slow $p_\rho$; rollouts precisely score out the actual tactics using the weak-but-fast $p_\pi$. Removing the policy prior costs ~800 Elo. And the raw policy network with zero search (1517) is already at the level of Fuego/Pachi-class programs doing 100k simulations.

**Accuracy is not the objective — speed is (Extended Data Table 3).** A 256-filter net with an 8-fold symmetry ensemble hits the best test accuracy, 57.0%, and beats the match network 69% of the time *in raw head-to-head play with no search*. But it takes 55.3 ms per evaluation instead of 4.8 ms, and inside AlphaGo's search it wins only **5%** of games. The 192-filter, single-symmetry net was chosen because it is 11× faster. Similarly the 128-filter net (54.6% acc, 2.8 ms) still wins 53% inside search. Figure 2a shows raw accuracy and playing strength are tightly correlated *when compute is held fixed* — but the search cares about simulations per second.

**Input features matter a lot.** 4 planes → 47.6% test accuracy; 12 → 54.7%; 20 → 54.7%; all 48 → 55.4%.

**Value net vs rollouts as an evaluator (Figure 2b).** MSE against true game outcome, on human expert positions. Ordering, best to worst: RL-policy rollouts ≈ value network < SL-policy rollouts < fast-rollout-policy rollouts < uniform random rollouts. The single forward pass of $v_\theta$ nearly matches 100 rollouts with $p_\rho$ while using **15,000× less computation**.

**Scaling (Table 8, 2 s/move).** 1 GPU/40 threads: 2181 Elo. 2 GPUs: 2738. 4: 2850. 8: 2890. Distributed: 64 GPUs/428 CPUs → 2937; 176 GPUs/1202 CPUs → 3140; 280 GPUs/1920 CPUs → 3168. Clearly saturating — the last near-doubling of hardware bought 28 Elo.

**Tournament.** AlphaGo 2890 Elo, distributed 3140, vs Crazy Stone 1929, Zen 1888, Pachi 1298, Fuego 1148, GnuGo 431. Fan Hui anchored at 2908. With four handicap stones given away, AlphaGo still beat Crazy Stone 77%, Zen 86%, Pachi 99%.

**What did not work:**

- **Training the value net on human games.** Overfit hard (0.19 train / 0.37 test MSE) because consecutive positions share a label. Needed one position per game, 30M distinct games.
- **The RL policy network as the MCTS prior.** Counter-intuitive: $p_\rho$ crushes $p_\sigma$ head to head, but $p_\sigma$ makes the *better* search prior. RL sharpens the distribution onto one best move; search wants a diverse beam of plausible moves to explore. The value net, however, is better when derived from $p_\rho$ than from $p_\sigma$.
- **Rotation/reflection-invariant convolution filters.** Helps small nets, *hurts* large ones — it stops intermediate filters from learning asymmetric patterns. Replaced by run-time random-symmetry averaging.
- **AMAF / RAVE and progressive widening**, the standard MCTS heuristics in every other Go program: "these biased heuristics do not appear to give any additional benefit" once you have a policy network supplying priors. Also no opening book and no dynamic komi.

## Worth Remembering

- The whole system is a **two-model distillation of search into networks, then re-injection of networks into search**. The pipeline is imitation → self-play policy gradient → regression on self-play outcomes. Every stage exists to make the *next* stage's data possible. AlphaGo Zero later showed stage 1 is removable entirely.
- The random move $a_U$ in value-net data generation is not decoration. It de-correlates the state distribution and makes $(s_{U+1}, z_{U+1})$ an unbiased sample of $v^{p_\rho}$ — otherwise you only ever see positions the policy already likes. This is the same coverage problem that [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives|offline RL]] fights constantly.
- **Reported strength is against Fan Hui (2p), not a top pro.** The paper does not claim world-champion level; that came a year later against Lee Sedol. The Elo anchoring here mixes 5-second-per-move computer games with long-time-control human games, which the authors admit is approximate.
- The comparison the authors are proud of: during the Fan Hui match AlphaGo evaluated **thousands of times fewer positions** than Deep Blue did against Kasparov. Better selection (policy net) and better evaluation (value net) beat raw node throughput. Also a data point for the compute-plus-general-methods thesis in [[The Bitter Lesson (essay)]] — though note the 48 hand-designed feature planes and the hand-crafted rollout patterns still in there.
- Practical caveat: the "value network only" variant ($\lambda=0$) at 2177 Elo already beat every prior program. If you want the idea without the engineering, the value net is the transferable half. The rollout half needs a fast simulator and a game with a cheap terminal score.
- The value net is trained to evaluate under $p_\rho$'s play, not under optimal play. It inherits that policy's blind spots — which is roughly how the Lee Sedol game-4 loss happened.
- The self-play opponent pool (sample a random past checkpoint) is a cheap and effective stabiliser for two-player learning; it shows up again in league training and in [[Multi-Agent Reinforcement Learning]] setups generally.

Open question worth chasing: the search uses $p_\sigma$ for priors and $p_\rho$ for the value target, which means two separate networks with two separate purposes. AlphaGo Zero collapses these into one two-headed net and makes MCTS itself the training target (policy iteration on visit counts) — that is the real simplification, and it made the whole pipeline above obsolete within eighteen months.

## Links
Related: [[Playing Atari with Deep Reinforcement Learning (DQN)]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[Markov Decision Process]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Proximal Policy Optimization Algorithms]] · [[A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB)]] · [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[The Bitter Lesson (essay)]] · [[Cross Entropy]] · [[Multi-Agent Reinforcement Learning]] · [[Distilling the Knowledge in a Neural Network]]

New topics worth writing: Monte Carlo Tree Search, PUCT and UCT exploration bonuses, AlphaGo Zero and AlphaZero, self-play policy iteration, Elo and BayesElo rating, virtual loss in parallel tree search, minimax and alpha-beta pruning, TD(λ), value function approximation in two-player zero-sum games
