---
title: "Long Short-Term Memory (Neural Computation)"
authors: ["Hochreiter & Schmidhuber"]
year: 1997
url: https://www.bioinf.jku.at/publications/older/2604.pdf
priority: Must-Read
read_on: 2026-08-24
tags: [paper]
---
## The Core Idea

Recurrent nets are supposed to remember. In practice, in 1997, they could not remember anything more than about ten steps back. This paper explains exactly why, and then builds an architecture where the "remembering" part of the gradient is *exactly 1.0* by construction.

The why: when you run [[Backpropagation]] through time, the error signal travelling from step $t$ back to step $t-q$ gets multiplied by one factor per step. Hochreiter's 1991 analysis (reproduced here) gives the scaling factor as

$$\frac{\partial \vartheta_v(t-q)}{\partial \vartheta_u(t)} = \sum_{l_1=1}^{n}\cdots\sum_{l_{q-1}=1}^{n}\;\prod_{m=1}^{q} f'_{l_m}\!\big(\mathrm{net}_{l_m}(t-m)\big)\, w_{l_m l_{m-1}}$$

That inner product is $q$ numbers multiplied together. If each is below 1, the whole thing shrinks like a geometric series — the gradient vanishes and long lags never get learned. If each is above 1, it explodes and the weights oscillate. There is no comfortable middle: it is a knife edge.

And the numbers are bad. For the logistic sigmoid, $\max f' = 0.25$. So the per-step factor $|f'(\mathrm{net})\,w|$ stays below 1 whenever $|w| < 4.0$. Standard small initialisation puts you firmly in the vanishing regime. Making weights bigger does not save you, because as $|w| \to \infty$ the derivative $f'$ goes to zero *faster* than $w$ grows. Raising the learning rate does not save you either — it scales long-range and short-range error flow by the same amount, so the ratio is unchanged. BPTT is, in the authors' words, "too sensitive to recent distractions."

The fix: build a unit where the per-step factor is forced to be exactly $1.0$. Solve $f'_j(\mathrm{net}_j)\,w_{jj} = 1.0$ and you get $f_j(x) = x$ (identity) with $w_{jj} = 1.0$. A linear unit with a self-loop of weight one. Its activation just stays put: $y_j(t+1) = y_j(t)$. They call this the **constant error carrousel** (CEC). Error entering it flows backwards forever without shrinking.

> [!NOTE] Constant Error Carrousel (CEC)
> A linear unit with a fixed self-connection of weight $1.0$. Because $f' \cdot w = 1$, gradient flowing back through it is neither scaled up nor down, no matter how many steps. This is the entire trick behind LSTM. ^constant-error-carrousel

But a bare CEC is useless, because it is always reading and always writing. Two conflicts appear:

- **Input weight conflict.** The same incoming weight $w_{ji}$ must both *store* some inputs and *ignore* others. It receives contradictory update signals and learns nothing clean.
- **Output weight conflict.** The same outgoing weight $w_{kj}$ must both *release* the memory at the right moment and *shield* downstream units from it the rest of the time. Same contradiction.

So: wrap the CEC in two multiplicative **gates**. An input gate decides when writing is allowed. An output gate decides when reading is allowed. The gates are ordinary sigmoid units that see the whole hidden state, so they can be context-sensitive in a way a single fixed weight can never be. That is the memory cell.

What it unlocks: learning across time lags of **1000+ steps** on noisy, incompressible sequences, at $O(1)$ cost per weight per step, with storage that does not grow with sequence length.

## The Methodology

**The memory cell.** For cell $c_j$ with input gate $in_j$ and output gate $out_j$:

$$\mathrm{net}_{c_j}(t) = \sum_u w_{c_j u}\, y_u(t-1), \qquad y^{in_j}(t) = f_{in_j}\!\big(\mathrm{net}_{in_j}(t)\big), \qquad y^{out_j}(t) = f_{out_j}\!\big(\mathrm{net}_{out_j}(t)\big)$$

$$\boxed{\;s_{c_j}(t) = s_{c_j}(t-1) + y^{in_j}(t)\, g\!\big(\mathrm{net}_{c_j}(t)\big), \qquad s_{c_j}(0)=0\;}$$

$$y^{c_j}(t) = y^{out_j}(t)\, h\!\big(s_{c_j}(t)\big)$$

Read the state update carefully. The old state is carried over with coefficient exactly $1$. That is the CEC. The new candidate $g(\mathrm{net}_{c_j})$ is multiplied by the input gate before being added.

**There is no forget gate.** The 1997 cell can only ever *add* to its internal state. Forgetting was bolted on by Gers, Schmidhuber & Cummins in 2000. Everything people now call "LSTM" includes it; this paper does not.

**Squashing functions used in experiments.** Gates are logistic in $[0,1]$. $g$ has range $[-2,2]$: $g(x) = \frac{4}{1+e^{-x}} - 2$. $h$ has range $[-1,1]$: $h(x) = \frac{2}{1+e^{-x}} - 1$. The asymmetry matters — $g$ writes with more dynamic range than $h$ reads back.

**Memory cell blocks.** $S$ cells sharing one input gate and one output gate form a "block of size $S$". A block of size 1 is a plain cell. Blocks let several cells hold a distributed code while costing only two gates total.

**Topology.** One input layer, one hidden layer, one output layer. The hidden layer is fully self-connected — every cell and every gate sees every cell output and every gate output from the previous step. Output units read only from memory cells (in most experiments).

> [!NOTE] Multiplicative gate
> A sigmoid unit whose output multiplies another signal, acting as a soft on/off valve. Learned, context-dependent, differentiable. The ancestor of every gating mechanism since — GRUs, highway nets, [[Gated Activation]] units. ^multiplicative-gate

**Truncated gradient — the part that actually makes it work.** LSTM uses an RTRL-style forward-mode update, but errors arriving at *memory cell net inputs* ($\mathrm{net}_{c_j}$, $\mathrm{net}_{in_j}$, $\mathrm{net}_{out_j}$) are **not propagated further back in time**. They still update the incoming weights, then they are cut. Formally the paper sets

$$\frac{\partial\,\mathrm{net}_{in_j}(t)}{\partial y_u(t-1)} \;\stackrel{tr}{\approx}\; 0, \qquad \frac{\partial\,\mathrm{net}_{out_j}(t)}{\partial y_u(t-1)} \;\stackrel{tr}{\approx}\; 0, \qquad \frac{\partial\,\mathrm{net}_{c_j}(t)}{\partial y_u(t-1)} \;\stackrel{tr}{\approx}\; 0$$

Why cut? Because otherwise error could leave a cell through its input gate, loop around the fully connected hidden layer, and re-enter through the output gate — and that loop is exactly where exponential decay lives again. Truncation guarantees $\partial s_{c_j}(t+1)/\partial s_{c_j}(t) \stackrel{tr}{=} 1$ exactly. Inside the cell, error is propagated back through previous internal states with no scaling at all.

The error signal's full journey: it arrives at the cell output, gets scaled once by $y^{out_j}(t)\,h'(s_{c_j}(t))$, then flows unscaled backwards for as many steps as you like, then on exit gets scaled once by $y^{in_j}(t-q)\,f'_{in_j}$ (if leaving via the input gate) or $y^{in_j}(t-q)\,g'$ (if leaving via the cell input). **Two multiplications total, regardless of $q$.** That is the whole point.

**Complexity.** $O(W)$ per time step, where $W$ is the number of weights — same as BPTT for a fully recurrent net, far better than RTRL's $O(W^2)$ or worse. Only the derivatives $\partial s_{c_j}/\partial w_{lm}$ for $l \in \{c_j, in_j\}$ need storing, which is $2CSI$ numbers. Unlike BPTT, storage does not depend on sequence length — LSTM is **local in space and local in time**.

**Two practical failure modes and their fixes:**

1. *Abuse problem.* Early in training, the net finds it can reduce error without storing anything, so it hijacks memory cells as constant bias units. Freeing them again takes forever. Fix: initialise output gate biases negatively (e.g. $-1, -2, -3$ for three blocks), which pushes cell outputs to zero and staggers when cells become "available". Alternative fix: add cells one at a time when error stops dropping.

2. *Internal state drift.* Since $s_{c_j}$ only accumulates, if inputs are mostly one sign the state wanders off, $h'(s_{c_j})$ saturates, and the gradient vanishes after all. Fix: initialise the *input* gate bias negative (e.g. $-3, -6$), so the cell writes very little at first. The authors report the exact bias value hardly matters.

**Training details.** On-line learning (not batch). Weights initialised in $[-0.1, 0.1]$ (or $[-0.2,0.2]$ for Experiments 1–2). Learning rates from $0.01$ to $1.0$ depending on task. Activations reset to zero between sequences. In most experiments the error signal appears **only at the sequence end**.

## Ablation Studies and Experiments

The authors are unusually careful about *what makes a benchmark valid*. Two criteria: (a) minimal time lags must be long for **all** training sequences — no short-lag examples to bootstrap from; (b) the task must not be solvable by **random weight guessing**. They report having discovered that simple weight guessing beats seven published methods on Bengio et al.'s parity and 2-sequence problems, which means those benchmarks measured nothing.

**Experiment 1 — Embedded Reber grammar.** Not a long-lag task (lags as short as 9), included because it is the standard benchmark and because it shows output gates earning their keep.

| Method | # weights | % success | sequences to success |
|---|---|---|---|
| RTRL (3 hidden) | ~170 | "some fraction" | 173,000 |
| RTRL (12 hidden) | ~494 | "some fraction" | 25,000 |
| Elman net | ~435 | **0** | >200,000 |
| Recurrent Cascade-Correlation | ~119–198 | 50 | 182,000 |
| LSTM (3 blocks × size 2, lr 0.5) | 276 | **100** | **8,440** |

LSTM is 2 failures out of 150 trials, and 3–20× faster even counting only the competitors' successful runs. Crucially: **"Without output gates, we did not achieve fast learning."** The output gate's job here is to stop the hard-to-learn long-lag memory (the first T or P) from disturbing the easy short-lag Reber transitions.

**Experiment 2 — long lags with distractors.**

*2a, noise-free, lag $p=100$:* RTRL 0% (>5M sequences), BPTT 0% (>5M), Neural Sequence Chunker 33% at 32,400, **LSTM 100% at 5,040**. For reference, RTRL manages 78% at lag $p=4$ and drops to **0% at $p=10$**.

*2b, kill the local regularities* by replacing the deterministic filler with a random subsequence. The chunker relies on compressibility, so it now fails entirely. LSTM: always successful, 5,680 sequences. This is a targeted ablation *of the competitor*, and it is the sharpest result in the section.

*2c, lags up to 1000 with hundreds of distractor symbols:*

| $q$ (lag−1) | $p$ (distractors) | $q/p$ | # weights | sequences to success |
|---|---|---|---|---|
| 50 | 50 | 1 | 364 | 30,000 |
| 200 | 200 | 1 | 1264 | 33,000 |
| 1,000 | 1,000 | 1 | 6064 | 49,000 |
| 1,000 | 200 | 5 | 1264 | 75,000 |
| 1,000 | 100 | 10 | 664 | 135,000 |
| 1,000 | 50 | 20 | 364 | 203,000 |

Read the top block: **scale the vocabulary with the lag and learning time barely moves** — 30k → 49k while the lag goes 51 → 1001. Read the bottom block: what actually hurts is *repetition frequency* of individual distractors ($q/p$), because frequently-seen symbols cause weight oscillation. Lag length is nearly free; distractor density is not.

**Experiment 3 — noise and signal on one channel.** 3a is Bengio's 2-sequence problem; LSTM solves it (27,380–58,370 sequences) but the authors state plainly that **random weight guessing beats LSTM here**, and say so in the table caption. 3c fixes the benchmark by making targets noisy real values ($0.2$ and $0.8$ plus Gaussian noise, sd $0.32$), so the net must learn conditional expectations to high precision — guessing is now hopeless. LSTM: 269,650 sequences, 0.00558 misclassified, average deviation from the true expectation 0.014.

**Experiment 4 — Adding Problem.** Two marked real values in $[-1,1]$ somewhere in a sequence; output their scaled sum. Only 93 weights (2 blocks of size 2). To *stress* the drift problem the authors deliberately bias all non-input units.

| $T$ | minimal lag | wrong / 2560 | sequences |
|---|---|---|---|
| 100 | 50 | 1 | 74,000 |
| 500 | 250 | 0 | 209,000 |
| 1000 | 500 | 1 | 853,000 |

**Experiment 5 — Multiplication Problem.** Included as a self-check: the CEC is an additive accumulator, so maybe LSTM is *cheating* on the adding problem by exploiting built-in integration. Swap sum for product. It works, but noticeably worse — at $T=100$, stopping at $n_{seq}=13$ took **1,273,000** sequences with 14/2560 wrong (MSE 0.0139), versus 74,000 for the equivalent adding task. Stopping early at $n_{seq}=140$ gives 139/2560 wrong. So: non-integrative tasks are solvable but roughly 17× slower. Honest, and the ablation reveals the CEC's additive structure really is a task-dependent prior. The authors note a hidden layer above the cells might fix the fine-tuning.

**Experiment 6 — Temporal order.** Classify by the *order* of two (6a) or three (6b) X/Y symbols buried at random positions in a 100–110 length sequence of distractors. 6a: 4 classes, 156 weights, 1/2560 wrong, 31,390 sequences. 6b: 8 classes, 308 weights, 2/2560 wrong, 571,100 sequences. Gaps between relevant inputs are ≥30 steps.

**What did not work:**

- **Full (non-truncated) gradient.** They tried it. "There was no significant difference to truncated LSTM, exactly because outside the CECs error flow tends to vanish quickly." Same reason full BPTT does not beat truncated BPTT. The expensive version buys nothing.
- **Strongly delayed XOR.** Storing one of the two inputs alone does not reduce expected error at all, so there is no gradient path to a partial solution. The task is *non-decomposable*. LSTM fails.
- **500-step parity.** Fails for the same structural reason. Random guessing solves it; LSTM with small initial weights does not.
- **No output gate, on Reber.** Learning is not fast.
- **The chunker on task 2b.** Its whole mechanism depends on compressible subsequences.

## Worth Remembering

- **The vanishing gradient result is the more important half of this paper.** The $|w| < 4.0$ threshold for the logistic sigmoid, and the fact that neither bigger weights nor bigger learning rates help, is a clean argument that should live next to your [[Derivative#Jacobian|Jacobian]] intuitions. The weak upper bound $\left|\frac{\partial\vartheta_v(t-q)}{\partial\vartheta_u(t)}\right| \le n\,(f'_{\max}\|W\|_A)^q$ makes the exponential explicit.
- **No forget gate in the original.** The state is a pure running sum. On a continuous stream with no resets it will saturate. Every experiment here resets activations between sequences, which quietly hides the problem.
- **Explicitly diagnosed as "like a feedforward net that sees the whole input at once."** The authors say this is why LSTM fails on parity and delayed XOR, and also why it wins everywhere else. That framing is remarkably close to how people now talk about [[Attention Is All You Need|attention]] — the difference between "sequential state" and "look at everything" is smaller than it looks.
- **Counting is a stated weakness.** "All gradient-based approaches suffer from practical inability to precisely count discrete time steps." Distinguishing 99 from 100 steps ago needs an extra mechanism. Distinguishing 3 from 11 is fine.
- **Cost of gating.** Each conventional hidden unit becomes at most 3 units, so the weight count rises by at most $3^2 = 9$ in the fully connected case. Their experiments deliberately hold weight counts comparable across methods (e.g. 10,504 for LSTM vs 10,404 for BPTT in 2a), which makes the comparisons fair.
- **Robustness claim.** "There appears to be no need for parameter fine tuning." A large learning rate self-corrects because it pushes output gates towards zero, damping its own effect. Nice bit of accidental self-[[Regularization|regularisation]].
- **The methodology lesson is broader than LSTM.** Discovering that random weight guessing beats seven published algorithms on a standard benchmark, then designing harder benchmarks explicitly to defeat guessing, is exactly the discipline missing from a lot of empirical ML. Pair this with [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]].
- **Follow-ups worth chasing:** Gers et al. 2000 (forget gates), Gers & Schmidhuber 2000 (peephole connections), Graves & Schmidhuber 2005 (bidirectional LSTM), Cho et al. 2014 (GRU — merges input/forget into one gate), and Greff et al. 2015 ("LSTM: A Search Space Odyssey", which ablates every component and finds the forget gate and output activation matter most).
- **Practical caveat if you ever implement this from the paper:** modern LSTM implementations use full BPTT through everything, not the truncated forward-mode rule described here. The two are different algorithms. The paper's own experiments say it does not matter much, which is a useful thing to know when your gradients look strange.

## Links

Related: [[Backpropagation]] · [[Deep Learning]] · [[Derivative]] · [[Attention Is All You Need]] · [[Seq2Seq models]] · [[Auto-regressive models]] · [[Gated Activation]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Kalman Filter]] · [[Markov Property]] · [[Regularization]] · [[Vector Jacobian Product]] · [[BERT4Rec- Sequential Recommendation with Bidirectional Transformer]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]]

New topics worth writing: Vanishing and exploding gradients, Backpropagation Through Time (BPTT), Real-Time Recurrent Learning (RTRL), Forget gates and LSTM variants (Gers 2000), Gated Recurrent Unit (GRU), Truncated backpropagation, Gradient clipping, Recurrent neural networks, Embedded Reber grammar benchmark, Credit assignment over long horizons
