---
title: "Mastering Chess and Shogi by Self-Play (AlphaZero)"
authors: ["David Silver", "Thomas Hubert", "Julian Schrittwieser", "Ioannis Antonoglou", "Matthew Lai", "Arthur Guez", "Marc Lanctot", "Laurent Sifre", "Dharshan Kumaran", "Thore Graepel", "Timothy Lillicrap", "Karen Simonyan", "Demis Hassabis"]
year: 2017
arxiv: "1712.01815"
url: https://arxiv.org/abs/1712.01815
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, rl, vision]
---
## The Core Idea

One algorithm, no game-specific knowledge, three games it beats the world champion program in.

Before this, the best chess engines (Stockfish, Deep Blue) worked like this: a human-written scoring function made of hundreds of handcrafted features — material values, pawn structure, king safety, bishop pair, piece-square tables — combined linearly with tuned weights, plus alpha-beta search with decades of clever pruning heuristics. Shogi engines like Elmo used the same recipe. Go engines were different but also hand-tuned. Each program only worked in its own game.

AlphaZero throws all of it away. It gets the rules of the game and nothing else. It starts from random weights, plays itself, and learns. The same code, the same network shape, the same hyper-parameters run for chess, shogi and Go. Only three things change per game: the board encoding, the move encoding, and one exploration noise constant.

The trick that makes this work is an old one from [[Mastering the game of Go with deep neural networks (AlphaGo)|AlphaGo]] / AlphaGo Zero, now shown to be general: **search and the network teach each other**. The network guesses a move distribution $\mathbf{p}$ and a position value $v$. Monte-Carlo tree search (MCTS) uses those guesses to explore, and comes out with a *better* move distribution $\boldsymbol{\pi}$ than the network alone. Then the network is trained to imitate $\boldsymbol{\pi}$. Search is a policy improvement operator; the network is the thing that stores the improvement. Repeat 700,000 times.

> [!NOTE] Policy improvement by search
> MCTS with a weak network gives a stronger move distribution than the network's raw output. Training the network on the search output is the entire learning signal — no human games, no labels. ^search-as-improvement

What it unlocks: chess fell to 4 hours of training, shogi to 2 hours, Go to 8 hours. And AlphaZero looks at **80,000 positions per second in chess versus Stockfish's 70 million** — roughly a thousand times fewer — and still wins 28–0 with 72 draws out of 100. Depth of judgement replaced breadth of search. This is [[The Bitter Lesson (essay)|the bitter lesson]] written out as a Nature-grade result.

## The Methodology

**The network.** One deep convolutional net with two heads:

$$(\mathbf{p}, v) = f_\theta(s)$$

$\mathbf{p}$ is a probability over every possible move, $v \in [-1,1]$ is the expected game outcome from position $s$. Same residual architecture as AlphaGo Zero (see [[Deep Residual Learning for Image Recognition (ResNet)|ResNet]]).

**The loss.** One line, and it explains the whole system:

$$l = (z - v)^2 - \boldsymbol{\pi}^\top \log \mathbf{p} + c\|\theta\|^2$$

- $(z-v)^2$: squared error between predicted value and the actual game result $z \in \{-1, 0, +1\}$.
- $-\boldsymbol{\pi}^\top \log \mathbf{p}$: [[Cross Entropy|cross-entropy]] pushing the network's raw policy toward the search's visit-count distribution.
- $c\|\theta\|^2$: L2 [[Regularization|weight decay]].

Note the value head predicts *expected outcome*, not probability of winning. AlphaGo Zero assumed binary win/loss because Go has no draws. Chess is full of draws — the optimal game is believed to be a draw — so the target is a three-valued $z$ and $v \approx \mathbb{E}[z \mid s]$.

**Board encoding.** An $N \times N \times (MT + L)$ stack of planes. $T = 8$ steps of history (needed for repetition rules), $M$ binary planes per step for piece types of each player, plus $L$ constant planes for colour, move count, castling rights, repetition count, no-progress count.

| Game | Input planes |
|---|---|
| Go | 17 |
| Chess | 119 |
| Shogi | 362 |

**Move encoding.** Chess: an $8\times8\times73$ stack = 4,672 moves. The $8\times8$ picks the square to lift a piece from. Of the 73 planes: 56 are "queen moves" (7 distances × 8 compass directions), 8 are knight moves, 9 are underpromotions (knight/bishop/rook × 3 pawn directions). Promotion to queen is the default for any 7th-rank pawn move. Shogi: $9\times9\times139$ = 11,259 moves, with 7 extra planes for dropping a captured piece back on the board. Illegal moves get probability zero and the rest are renormalised.

**Search.** MCTS, 800 simulations per move during training. Each simulation walks from root to leaf picking the move with low visit count, high prior $p_a$, and high average value. Reaching a leaf, the network evaluates it once and the value is backed up along the path. Crucially this **averages** subtree values instead of taking a minimax. The authors argue this is why a neural evaluator finally beats alpha-beta here: a deep net makes occasional wild errors, and averaging cancels them, while minimax propagates the single biggest error straight to the root.

**Exploration.** Dirichlet noise $\text{Dir}(\alpha)$ added to the root priors, scaled inversely to the typical branching factor: $\alpha = 0.3$ (chess), $0.15$ (shogi), $0.03$ (Go). This is the *only* per-game hyper-parameter. During training, moves are sampled proportional to root visit counts; at evaluation, taken greedily.

**Training loop.** Unlike AlphaGo Zero, there is no "best player" gate. AlphaGo Zero froze a champion network, generated self-play with it, and only promoted a new one if it won 55% of a match. AlphaZero keeps **one continuously updated network** and always self-plays with the latest weights. Simpler, and it works.

Also gone: the 8-fold rotation/reflection data augmentation and the random symmetry transform at evaluation time. Go is symmetric; chess is not (pawns only go forward, castling differs by side), so symmetry is dropped everywhere for generality.

And no [[Practical Bayesian Optimization of Machine Learning Algorithms|Bayesian optimisation]] of search hyper-parameters — AlphaGo Zero tuned them, AlphaZero reuses one setting across all three games.

**Scale.** 700k mini-batches of 4,096. Learning rate 0.2, dropped to 0.02, 0.002, 0.0002. 5,000 first-gen TPUs generating self-play games, 64 second-gen TPUs training.

| | Chess | Shogi | Go |
|---|---|---|---|
| Training time | 9h | 12h | 34h |
| Self-play games | 44M | 24M | 21M |
| Thinking time/move | 40ms | 80ms | 200ms |

## Ablation Studies and Experiments

**Time-to-surpass** (Figure 1, Elo from 1-second-per-move games during training):

- Chess: beat Stockfish 8 after **4 hours / 300k steps**.
- Shogi: beat Elmo after **under 2 hours / 110k steps**.
- Go: beat AlphaGo Lee after **8 hours / 165k steps**.

**Head-to-head, 100 games, 1 minute per move.** AlphaZero on 1 machine with 4 TPUs; Stockfish and Elmo on 64 CPU threads, 1GB hash, strongest settings, no pondering.

| Game | Opponent | AZ as White (W/D/L) | AZ as Black (W/D/L) |
|---|---|---|---|
| Chess | Stockfish | 25 / 25 / 0 | 3 / 47 / 0 |
| Shogi | Elmo | 43 / 2 / 5 | 47 / 0 / 3 |
| Go | AlphaGo Zero (3-day) | 31 / – / 19 | 29 / – / 21 |

Zero losses to Stockfish across 100 games. Eight losses to Elmo. The Go margin over AlphaGo Zero is thin (60–40 overall) — AlphaZero is *more general*, not dramatically stronger at Go.

**The scaling ablation (Figure 2) — the most interesting one.** Give each engine more thinking time per move and plot Elo. AlphaZero's MCTS gains Elo *faster* with extra time than Stockfish's or Elmo's alpha-beta does. This directly contradicts the received wisdom that alpha-beta is inherently the right search for chess. Prior MCTS chess programs were much weaker than alpha-beta programs; the difference is that they had no strong learned evaluator to guide them.

**Opening discovery (Table 2).** They took the 12 human openings played more than 100,000 times in an online database and plotted how often AlphaZero played each during self-play training. All 12 are discovered independently, played heavily for a while, then some are discarded as training goes on — e.g. the French Defence rises and falls. Starting matches *from* each of these openings, AlphaZero still beat Stockfish overall: as White 242 wins / 353 draws / 5 losses; as Black 48 / 533 / 19.

**What did not work / what they tried and dropped:**

- A **flat vector policy** over moves for chess and shogi instead of the plane-structured $8\times8\times73$ representation: "the final result was almost identical although training was slightly slower." So the spatial move encoding is a mild convenience, not the source of the strength.
- Symmetry augmentation and evaluation-time symmetry transforms were removed and nothing broke — they were a Go-specific crutch.
- The best-player gating from AlphaGo Zero was removed and nothing broke.
- Bayesian hyper-parameter tuning was removed and nothing broke.

The pattern is that almost every piece of machinery bolted onto AlphaGo Zero turned out to be removable. The load-bearing components are: the two-headed network, MCTS as a policy improver, and the single loss in Equation 1.

## Worth Remembering

**The knowledge AlphaZero *does* get.** The paper is honest about this in the Methods. It is not truly knowledge-free:

1. The input and output are laid out as planes matched to the grid of the board (a convolutional [[An Image is Worth 16x16 Words (ViT)#The Core Idea|inductive bias]]).
2. It has a **perfect simulator** — full knowledge of the rules, used inside MCTS to roll positions forward, detect terminal states, and score them. This is the biggest caveat for anyone hoping to transfer this to the real world. MuZero later removed exactly this.
3. Rules are used to encode castling, repetition, no-progress, promotions, drops.
4. Typical legal-move count sets the Dirichlet $\alpha$.
5. Games past a length cap are declared draws.

**Why averaging beats minimax here.** Worth internalising as a general lesson: when your evaluator is a non-linear function approximator with unpredictable errors, an operator that takes maxima over many evaluations will surface the worst error. An operator that averages will cancel it. Traditional engines used *linear* evaluations (weaker but better-behaved), which minimax could tolerate.

**Compute honesty.** "4 hours" hides 5,000 TPUs generating 44 million self-play games. This is a wall-clock claim, not a compute-efficiency claim. And the comparison is slightly unfair in the other direction too: Stockfish ran on 64 CPU threads while AlphaZero ran on 4 TPUs, and Stockfish 8 with a 1GB hash at one minute per move was later argued to be a handicapped configuration.

**Draws changed the objective.** The move from binary win/loss to expected outcome is small in code and essential in chess, where 47 of 50 games as Black were drawn. If your domain has a "neutral" outcome, do not squash it into a binary label.

**Connections.** The self-play loop is a form of generalised policy iteration over an [[Markov Decision Process|MDP]] where the "policy improvement" step is search rather than a [[Simple Statistical Gradient-Following Algorithms (REINFORCE)|gradient step]]. Compare with the model-based imagination loop in [[Mastering Diverse Domains through World Models (DreamerV3)|DreamerV3]], which must *learn* the dynamics AlphaZero is handed for free. Compare also with [[Playing Atari with Deep Reinforcement Learning (DQN)|DQN]], which has no search at all at action time.

**Open question the paper flags itself:** they never tried adding back the classical tricks (opening books, endgame tablebases, transposition tables, quiescence search). "It is likely that some of these techniques could further improve the performance of AlphaZero." Purity was the point, not maximum strength.

## Links
Related: [[Mastering the game of Go with deep neural networks (AlphaGo)]] · [[The Bitter Lesson (essay)]] · [[Cross Entropy]] · [[Markov Decision Process]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Playing Atari with Deep Reinforcement Learning (DQN)]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Regularization]] · [[Practical Bayesian Optimization of Machine Learning Algorithms]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[Uncertainty]]

New topics worth writing: Monte-Carlo Tree Search and the PUCT selection rule, Alpha-beta pruning and quiescence search, MuZero (learned dynamics, no simulator), Generalised policy iteration, Elo and BayesElo rating, Self-play as curriculum, Dirichlet noise for exploration in tree search, Expert Iteration (ExIt)
