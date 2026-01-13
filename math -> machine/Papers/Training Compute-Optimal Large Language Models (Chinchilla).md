---
title: Training Compute-Optimal Large Language Models (Chinchilla)
authors:
  - Jordan Hoffmann
  - Sebastian Borgeaud
  - Arthur Mensch
  - Elena Buchatskaya
  - Trevor Cai
  - Eliza Rutherford
  - Diego de Las Casas
  - Lisa Anne Hendricks
  - Johannes Welbl
  - Aidan Clark
  - Tom Hennigan
  - Eric Noland
  - Katie Millican
  - George van den Driessche
  - Bogdan Damoc
  - Aurelia Guy
  - Simon Osindero
  - Karen Simonyan
  - Erich Elsen
  - Jack W. Rae
year: 2022
arxiv: "2203.15556"
url: https://arxiv.org/abs/2203.15556
priority: Must-Read
read_on: 2026-08-22
tags:
  - paper
  - transformers
  - llm
  - optimization
  - scaling
  - chinchilla
---
## The Core Idea

If you have a fixed amount of compute — say a cluster for three months — you have to pick two numbers: how big the model is ($N$ parameters) and how much text it reads ($D$ tokens). Compute is roughly the product, $C \approx 6ND$. So you are spending one budget on two things, and the split matters.

The prevailing answer, from [[Scaling Laws for Neural Language Models|Kaplan et al. 2020]], was: spend it mostly on model size. Their fit said a $10\times$ bigger budget should buy a $5.5\times$ bigger model but only $1.8\times$ more data. The whole field followed. GPT-3 (175B), Jurassic-1 (178B), Gopher (280B), MT-NLG (530B) were all trained on roughly **300 billion tokens** — the models grew 3× but the data stayed frozen.

This paper says that split is wrong. Train 400+ models, fit the frontier again more carefully, and the answer comes out **equal scaling**: double the model, double the data. Formally $N_{opt} \propto C^{0.5}$ and $D_{opt} \propto C^{0.5}$, not $C^{0.73}$ and $C^{0.27}$.

The consequence is blunt: every big model of that era was badly undertrained. Gopher used $5.76 \times 10^{23}$ FLOPs on 280B parameters and 300B tokens. The same budget spent optimally is a **70B model on 1.4T tokens**. They built that model — Chinchilla — and it beats Gopher on essentially everything, while being 4× cheaper to run at inference.

> [!NOTE] Compute-optimal training
> Choosing $(N, D)$ to minimise final loss subject to a fixed FLOP budget, rather than picking a model size first and training "until it looks done". Compute-optimal does **not** mean "trained to convergence" — the optimal model is deliberately stopped early relative to what it could absorb. ^compute-optimal

The reason the old answer was wrong is subtle and worth keeping: Kaplan et al. used a **fixed cosine learning-rate schedule length** (130B tokens) for every run, and read off intermediate losses along the curve as if they were the loss of a shorter run. But a model whose learning rate has not finished decaying looks worse than it is. So every short-data point was penalised, which made data look less valuable than it is, which pushed the fit toward "just make the model bigger". A learning-rate-schedule bug, essentially, that cost the field two years of compute.

## The Methodology

Three independent ways to estimate the frontier. They agree, which is the point.

**Approach 1 — envelope of training curves.** Train a family of models from 70M to 10B parameters. For each size, train it **four times** with four different cosine cycle lengths (spanning $16\times$ in token count), each decaying the learning rate by $10\times$ over exactly its own horizon. Smooth each loss curve (Gaussian, window 10 steps), interpolate to get loss-as-a-function-of-FLOPs, then at 1500 log-spaced FLOP values ask: which run has the lowest loss here? That traces the lower envelope. Fit power laws to the winning $(N, D)$ pairs. Result: $a = 0.50$, $b = 0.50$.

**Approach 2 — IsoFLOP profiles.** Pick 9 fixed FLOP budgets from $6\times10^{18}$ to $3\times10^{21}$. For each budget, train many different model sizes, adjusting token count so total FLOPs stay constant (and setting the cosine cycle to match). Plot final loss against $N$ — you get a clear **valley**. Fit a parabola, read off the bottom. Then fit power laws through the 9 valley bottoms. Result: $a = 0.49$, $b = 0.51$.

**Approach 3 — fit a parametric loss surface.** Take every final loss from Approaches 1 and 2 and fit

$$\hat{L}(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

The three terms come from a classical risk decomposition: $E$ is the irreducible entropy of natural text (the Bayes risk); $A/N^\alpha$ is how much a perfectly-trained $N$-parameter transformer falls short because the function class is too small; $B/D^\beta$ is how much you lose from taking finitely many gradient steps on a finite sample. Fitted values:

$$L(N,D) = 1.69 + \frac{406.4}{N^{0.34}} + \frac{410.7}{D^{0.28}}$$

Fitting detail that matters: they minimise **Huber loss** ($\delta = 10^{-3}$) on the log residuals, using L-BFGS from a grid of initialisations. Huber is robust to outliers (see [[Loss, Objectives, and Business Alignment]]) and the low-FLOP runs *are* outliers — they fit the power-law shape badly. A larger $\delta$ makes the fit chase the small-compute regime and predict held-out large runs poorly.

Minimising $\hat{L}$ subject to $6ND = C$ gives closed forms:

$$N_{opt}(C) = G\left(\tfrac{C}{6}\right)^{a}, \quad D_{opt}(C) = G^{-1}\left(\tfrac{C}{6}\right)^{b}, \quad a = \tfrac{\beta}{\alpha+\beta},\; b = \tfrac{\alpha}{\alpha+\beta}$$

With $\alpha = 0.34, \beta = 0.28$ this gives $a = 0.46$, $b = 0.54$ — slightly *more* data-hungry than the other two.

**Chinchilla itself.** 70B parameters, 1.4T tokens, same [[Attention Is All You Need|transformer]] decoder architecture as Gopher, same $5.76\times10^{23}$ FLOP budget.

| | Gopher 280B | Chinchilla 70B |
|---|---|---|
| Layers | 80 | 80 |
| Heads | 128 | 64 |
| Key/value size | 128 | 128 |
| $d_{model}$ | 16,384 | 8,192 |
| Max LR | $4\times10^{-5}$ | $1\times10^{-4}$ |
| Batch (tokens) | 3M → 6M | 1.5M → 3M |

Feed-forward width is always $4 \times d_{model}$. Batch size doubles midway through training for both.

Four non-size differences from Gopher: **AdamW instead of Adam** (decoupled weight decay — see [[Regularization]]); a **float32 master copy of the weights** in the sharded optimiser state while forward/backward run in bfloat16 ([[Mixed Precision training]]); a SentencePiece tokenizer without NFKC normalisation (94.15% token overlap with Gopher's, helps maths and chemistry); and a reweighted mix of the same MassiveText corpus — MassiveWeb 45%, Books 30%, C4 10%, News 10%, GitHub 4%, Wikipedia 1%. At 1.4T tokens Wikipedia is seen 3.4 times and MassiveWeb 1.24 times; everything else is under one epoch.

## Ablation Studies and Experiments

**The three approaches, side by side (Table 2):**

| Approach | $a$ ($N_{opt} \propto C^a$) | $b$ ($D_{opt} \propto C^b$) |
|---|---|---|
| 1. Training-curve envelope | 0.50 (0.488, 0.502) | 0.50 (0.501, 0.512) |
| 2. IsoFLOP profiles | 0.49 (0.462, 0.534) | 0.51 (0.483, 0.529) |
| 3. Parametric loss fit | 0.46 (0.454, 0.455) | 0.54 (0.542, 0.543) |
| Kaplan et al. 2020 | 0.73 | 0.27 |

Parentheses are 10th/90th percentiles from bootstrapping (80% resample, 100 times).

**What the frontier implies (Approach 1, Table 3).** A 175B model should get $3.85\times10^{24}$ FLOPs and **3.7T tokens**. A 280B Gopher-sized model wants **5.9T tokens**. A 1T-parameter model needs 21.2T tokens and 221× Gopher's compute — so a trillion-parameter dense model was simply not the right thing to build in 2022.

**Direct head-to-head at $10^{21}$ FLOPs (Appendix D.4).** This is the cleanest single test. Kaplan's fit says the optimal model is 4.68B parameters; Approach 1 says 2.86B. They trained a 4.74B and a 2.80B transformer with matched depth-to-width ratio, batch size 0.5M tokens, max LR $1.5\times10^{-4}$. The smaller model wins.

**Chinchilla vs Gopher headline results:**

- **MMLU 5-shot: 67.6% vs 60.0%** (+7.6). GPT-3 is 43.9%; average human rater 34.5%; human expert 89.8%. This beat the *June 2023* forecast (63.4%) of 73 competitive forecasters, in March 2022. Better on 51/57 subjects, tied on 2, worse on 4 (`college_mathematics`, `econometrics`, `moral_scenarios`, `formal_logic`).
- **BIG-bench: 65.1% vs 54.4%** (+10.7 average). Worse on only 4/62 tasks.
- **RACE-h few-shot: 82.3% vs 71.6%**. RACE-m: 86.8% vs 75.1%. LAMBADA zero-shot: 77.4% vs 74.5% (MT-NLG 530B gets 76.6%).
- **WikiText-103 perplexity: 7.16 vs 7.75.** Better bits-per-byte on all 19 Pile subsets.
- **Natural Questions 64-shot: 35.5% vs 28.2%** (new closed-book SOTA). TriviaQA unfiltered 5-shot: 73.2% vs 63.6%.
- **TruthfulQA 0-shot: 43.6% vs 29.5%.** Notable because Lin et al. argued bigger models get *less* truthful; here better modelling of the same data helped a lot.
- Common sense: beats MT-NLG 530B on 4 of 5 benchmarks despite being 7.5× smaller.

**Things that did not work / negative findings:**

- **Cosine cycle length must match the run length (Appendix B).** They swept cycle lengths at 1, 1.1, 1.25, 1.5, 2, 5× the actual number of training steps. Overshooting by more than ~25% clearly degrades final loss. This is the exact mechanism that broke the earlier scaling law, and it is the most practically useful ablation in the paper.
- **The frontier is not a clean power law (Appendix E).** Fitting the first, middle, and last third of the envelope points separately gives *different* slopes — there is negative curvature in $\log N_{opt}$. Extrapolating from tiny models overestimates optimal model size. This is also why Approach 3 predicts smaller models than 1 and 2: Huber loss downweights the low-FLOP points, which happen to sit on the steeper part.
- **Language modelling scores are partly confounded.** Chinchilla saw 4× more data, so Pile/WikiText numbers may reflect train-test leakage. The authors say to trust MMLU, BIG-bench and closed-book QA instead.
- **Toxicity did not improve.** Mean PerspectiveAPI toxicity on 25,000 unprompted samples: Gopher 0.081 (median 0.064), Chinchilla 0.087 (median 0.066). 95th percentile 0.230 vs 0.238. A better model of the data is not a less toxic one.
- **Bias improved unevenly.** Winogender pronoun resolution: 78.3% vs 71.4% overall, but male pronouns improved only 3.2% while female improved 8.3% and neutral 9.2%. Largest gain was on female "gotcha" examples (+10%). Better, but the improvement is not uniform across groups, which is itself a form of bias.
- **AdamW > Adam, independent of schedule** (Appendix G, Figures A6/A7, tested at 417M and 1.4B). The higher-precision optimiser-state weights also helped. Both are confounds in the Chinchilla-vs-Gopher comparison, though small ones.
- **Result holds on other datasets** (Appendix C). Re-running the IsoFLOP analysis on C4 gives $a=0.50, b=0.50$; on GitHub code $a=0.53, b=0.47$. So the equal-scaling conclusion is not a MassiveText artefact — as long as you stay under one epoch.
- **FLOP accounting barely matters.** Their exact per-layer FLOP count (including embeddings, attention logits, softmax) differs from the $6ND$ approximation by 0.99× to 1.10× across model sizes. Use $6ND$.

## Worth Remembering

- **The number to memorise: ~20 tokens per parameter.** From Table 3, a 67B model wants 1.5T tokens; a 1B model wants 20.2B. That ratio is roughly constant along the frontier because both exponents are ≈0.5. This became the default rule of thumb for years afterwards (and later models like Llama deliberately overshot it, trading extra training compute for cheaper inference).
- **Inference is the hidden argument.** Chinchilla is 4× smaller, so serving and fine-tuning cost 4× less forever. The paper is careful to note that compute-optimal *training* is not the same as compute-optimal *deployment* — if you serve a model a lot, you should train even smaller and even longer than Chinchilla says.
- **Limitations the authors admit:** only two large runs (Chinchilla and Gopher) actually validate the extrapolation, with nothing at intermediate scale. The power-law form is assumed, and they see curvature that suggests optimal models are even *smaller* than predicted at high budgets. Everything was trained for **less than one epoch** — the multi-epoch regime is unexplored, and it matters enormously once you run out of high-quality web text.
- **The data wall.** Table 3 projects that a 10T-parameter model would need 216T tokens. There is not that much good text. The paper explicitly says dataset scaling and dataset *quality* now matter more than architecture engineering, and flags that bigger corpora make train-test contamination and private-data leakage harder to control.
- **A useful reading of the loss fit:** $\alpha = 0.34$ and $\beta = 0.28$ are both well below the theoretical $1/2$ you would hope for from stochastic approximation bounds. The authors note that future architectures and optimisers should be judged partly on whether they *raise these exponents* — that would be a genuine efficiency gain, not just more compute.
- The $E = 1.69$ term is an estimate of the irreducible [[Cross Entropy|cross-entropy]] of natural text under their tokenizer. No amount of scale gets below it.
- Chinchilla was never released publicly (see the model card: "We will not make this model available publicly"), so the empirical scaling law is the lasting contribution, not the weights.

## Links

Related: [[Scaling Laws for Neural Language Models]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[Attention Is All You Need]] · [[Auto-regressive models]] · [[Cross Entropy]] · [[Loss, Objectives, and Business Alignment]] · [[Mixed Precision training]] · [[Regularization]] · [[Foundation Models]] · [[In Context Learning]] · [[MLE_L4_Question_Bank]]

New topics worth writing: Cosine learning rate schedules and cycle-length calibration, AdamW and decoupled weight decay, L-BFGS and quasi-Newton optimisation, Bias-variance/risk decomposition (Bayes risk, approximation error, estimation error), MMLU and BIG-bench evaluation suites, Bits-per-byte as a language modelling metric, Mixture-of-Experts scaling, Retrieval-augmented language models (RETRO), Inference-optimal vs compute-optimal scaling, Data quality and the token wall
