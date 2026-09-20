---
title: "Continual Learning Mechanisms Compose for Long-Horizon Memorization"
authors: ["Zheyuan Zhang", "Alvin Zhang", "Daniel Khashabi", "Tianmin Shu"]
year: 2026
arxiv: "2609.06986"
url: https://arxiv.org/abs/2609.06986
priority: Good-To-Read
read_on: 2026-09-19
tags: [paper, llm]
---
## The Core Idea

Take a language model. Teach it 100 small sets of facts, one set after another. After the last set, ask it about the first set. It will have forgotten almost everything — 1.2% of the answers survive. This is catastrophic forgetting, and this paper asks a narrow question about it: if no single anti-forgetting trick works at this length, do several tricks stacked together work?

The answer is yes, and the stacking is not merely additive. Two of the mechanisms help each other far more than their separate gains would predict. Together with a third and fourth they push final retention from 1.2% to 34.9% — a 28× gain.

The framing that makes this tractable is the paper's real contribution. Every continual-learning method is sorted into one of two boxes:

- **Anchors** answer *what should this update preserve?* There are three kinds. A **data anchor** preserves examples (here: fake examples the model generates for itself). A **function anchor** preserves the model's input→output behaviour (self-[[Distillation|distillation]] against yesterday's model). A **weight anchor** preserves specific parameter values judged important (EWC, Synaptic Intelligence).
- **Low-rank allocation rules** answer *where do successive updates live?* Either you keep training one [[LoRA]] adapter forever (shared), or you fold each task's adapter into the dense weights and start a fresh one (merged).

> [!NOTE] Long-horizon memorization
> Learn 100 query→answer tasks by sequential fine-tuning. No stored raw examples from earlier tasks. No task ID at inference time. Evaluate on the *exact training queries* — this measures recall, not generalisation. ^long-horizon-memorization

Why did nobody do this before? Continual-learning papers overwhelmingly propose one mechanism and compare it against other single mechanisms. The combinatorics were never searched, partly because 100 sequential fine-tunes × 90 configurations × 3 seeds × 3 datasets is expensive. The paper's second contribution is a search procedure cheap enough to make that affordable.

## The Methodology

**Setup.** Backbone is Qwen3-4B-Base. $T=100$ tasks arrive in a stream. Task $t$ has data $\mathcal{D}_t$ of query–answer pairs. You get $\mathcal{D}_t$ and whatever state you carried from $t-1$; you may not revisit old raw examples. Plain objective is [[Cross Entropy|cross-entropy]] over the whole formatted sequence:

$$\mathcal{L}_{\mathrm{SFT}}^{t}(\Theta)=-\mathbb{E}_{(x,y)\sim\mathcal{D}_{t}}\left[\log p_{\Theta}(x,y)\right]$$

Note they do **not** mask the query tokens — the whole `Question: q \nAnswer: \boxed{a}` string is trained on. Their reason is practical: at deployment time in test-time training you often cannot cleanly separate prompt from answer.

The full objective adds three retention terms:

$$\Theta_{t}=\operatorname*{arg\,min}_{\Theta}\ \mathcal{L}_{\mathrm{SFT}}^{t}(\Theta)+\mathcal{R}_{D}^{t}(\Theta)+\mathcal{R}_{F}^{t}(\Theta)+\mathcal{R}_{W}^{t}(\Theta)$$

### The data anchor — unconditional generative replay

This is the clever one. No stored examples, so the model manufactures its own. They add a single new token `<|replay_token|>` to the vocabulary, initialised to the mean of all existing [[Embeddings|embeddings]] and then frozen, and prepend it to every training sequence. After many tasks, sampling from $p_\Theta(\cdot \mid s)$ with that one token as the entire prompt produces plausible old-looking question-answer pairs.

Before each task $t>1$: freeze the previous model, generate $N_R=300$ sequences from the seed token (nucleus sampling, top-$p=0.9$, generation temperature $\tau_G=1.5$, max 384 new tokens), drop empties. These are thrown away at the end of the task — they are not carried forward, so state stays constant.

The loss on them is **soft**, not hard. The frozen model provides the next-token distribution at every position; the student matches it with forward [[KL Divergence|KL]] at replay temperature $\tau_D=2$:

$$\mathcal{L}_{D}^{t}=\frac{\tau_{D}^{2}}{\sum_{m,\ell}a_{m,\ell}}\sum_{m,\ell}a_{m,\ell}\,\mathrm{KL}\!\left(q_{m,\ell}^{D}\,\|\,p_{m,\ell}^{D}\right)$$

The $\tau_D^2$ keeps [[Backpropagation|gradient]] magnitude comparable across temperatures — same trick as in [[Distilling the Knowledge in a Neural Network|Hinton distillation]]. This is mixed into the fit loss rather than added as a penalty:

$$\mathcal{L}_{\mathrm{fit}}^{t}=(1-w)\mathcal{L}_{\mathrm{SFT}}^{t}+w\,\mathcal{L}_{D}^{t}, \qquad w\in\{0.5,\,0.75\}$$

Every current-task minibatch is paired with exactly one replay minibatch.

### The function anchor — self-distillation on current data

Straight Learning-without-Forgetting. Yesterday's frozen model is the teacher. On *today's* sequences, match its full-vocabulary next-token distribution at $\tau_F=5$, weight $\lambda_F=1$. The difference from the data anchor is only the inputs: data anchor distils on generated sequences, function anchor distils on real current ones.

### The weight anchor — SI or online EWC

A diagonal quadratic pulling parameters back toward their post-task-$t$ values:

$$\mathcal{R}^{t+1}=\sum_{i=1}^{P}\Omega_{t,i}\left(\vartheta_{i}-\vartheta_{t,i}^{\star}\right)^{2}$$

**Synaptic Intelligence** builds $\Omega$ from the optimisation path — how much each coordinate reduced the loss per unit of movement:

$$\omega_{t,i}=-\sum_{k}g_{t,k,i}\left(\vartheta_{t,k+1,i}-\vartheta_{t,k,i}\right), \qquad \Omega_{t,i}=\Omega_{t-1,i}+\frac{\max(0,\omega_{t,i})}{\Delta_{t,i}^{2}+\xi}$$

with $\xi=0.1$, $\lambda_{\mathrm{SI}}=1$. **Online EWC** instead estimates a diagonal Fisher after training (up to 1000 sequences), rescales each task's Fisher to the first task's mean magnitude, accumulates with $\gamma=1$, $\lambda_{\mathrm{EWC}}=1000$. Crucially the tracked coordinates are *LoRA entries*, not dense weights — this matters later.

### Low-rank allocation

$$W_{t}=\begin{cases}W_{0}+\rho B_{t}A_{t} & \text{shared LoRA}\\ W_{t-1}+\rho B_{t}A_{t} & \text{merged LoRA}\end{cases}$$

Shared LoRA keeps optimising one adapter pair forever; cumulative change has rank $\le r$ no matter how many tasks. **Merged LoRA** (borrowed from ReLoRA) folds $\rho B_t^\star A_t^\star$ into the dense matrix after each task, then attaches a fresh adapter with Kaiming-initialised $A$ and $B=0$. Cumulative change can reach rank $tr$. Both keep constant state.

$r=32$, $\alpha=64$, all seven projections targeted, dropout 0.05. AdamW, lr $5\times10^{-4}$, 10 epochs per task, batch 8, 5% linear warmup then constant, optimiser reset each task.

### The three datasets

All 100 tasks, all evaluated on the training queries, all guaranteed one answer per query globally.

- **Symbol-QA** — 10,000 random 6-char key → 4-char value pairs, 100 per task. Zero structure. Pure association memory.
- **LLM-QA** — 10,000 facts about invented entities ("fictional lighthouses and their keepers"), generated by Qwen3-4B-Instruct, validated so the entity name appears in the question, answers ≤ 8 words. 100 per task. Natural language, no real-world knowledge to lean on.
- **Real-QA** — 5,000 items, 500 each from TriviaQA, NQ-Open, PopQA, SQuAD, WebQuestions, OpenBookQA, SciQ, ARC-Easy, ARC-Challenge, MedMCQA. Multiple-choice items converted to free-form by dropping the options. **Filtered per-model**: 5 samples at temperature 0.7; if any contains an accepted answer as a word-boundary substring, discard. So these are facts the base model demonstrably failed on. Pooled, shuffled globally, cut into 100 tasks of 50 — so every task mixes all ten sources, preventing domain-block effects.

### Task-level successive halving

90 configurations = {∅, oEWC, SI} × {∅, SD@1, SD@3} × {∅, four replay settings} × {shared, merged}. Instead of allocating more *iterations* like classic successive halving, TSH allocates more *tasks*: train all 90 for 10 tasks, keep top 45; 20 tasks, keep 23; 50 tasks, keep 10; those 10 run to 100. Promoted configs resume from checkpoint rather than retrain.

Cost per seed per dataset: $90(10)+45(10)+23(30)+10(50)=2540$ task-units versus 9000 exhaustive — a 71.8% saving.

Search uses a *different task order* (seed 1234) from the final evaluation. Selected hyperparameters are then frozen and the winners retrained from scratch on the report order, inside a full $2^4$ factorial (SI × SD × Replay × Merge), 3 seeds each.

## Ablation Studies and Experiments

**Metrics.** From the lower-triangular accuracy matrix $M_{i,j}$ (accuracy on task $j$ after learning through task $i$): **Final** $=\frac{1}{T}\sum_j M_{T,j}$, **Diag** $=\frac{1}{T}\sum_j M_{j,j}$ (immediate acquisition), **Forget** = mean drop from peak to final.

**Headline.** Final retention after 100 tasks:

| | Symbol-QA | LLM-QA | Real-QA |
|---|---|---|---|
| Naive sequential SFT | 1.0 | 1.4 | 1.3 |
| Best *single* mechanism | 4.2 | 7.5 | 12.5 |
| Best composition | 23.2 | 41.8 | 54.8 |
| All three anchors + merged LoRA | 18.5 | 41.8 | 44.3 |

Average of the all-anchors + merge recipe across datasets: **34.9%** vs 1.2% naive. It is the only configuration in the 16-cell factorial that lands top-3 on all three datasets.

**Memory half-life** (age at which retention halves): naive gets 1, 1, 2 tasks. Best single mechanism gets 4, 6, 11. Best composition gets 19, 32, 44. Composition changes the *timescale* of decay, not its shape — every curve still slopes down.

**The factorial is the payload.** Main effects in percentage points:

| | SI | SD | Replay | Merge | Replay × Merge |
|---|---|---|---|---|---|
| Symbol-QA | +0.3 | **+5.7** | **+9.5** | **+5.9** | **+3.6** |
| LLM-QA | **+5.8** | **+5.0** | **+18.5** | **+14.9** | **+9.4** |
| Real-QA | **+6.3** | **+3.7** | **+19.3** | **+20.5** | **+11.7** |

Replay and merged LoRA are the two biggest effects everywhere, and their interaction is positive and significant everywhere. The super-additivity is stark when you look at raw configs: on Real-QA, replay alone gives +11.2 and merge alone +2.7 over naive (sum: 13.9), but together they give **+46.9**. On LLM-QA: 7.7 predicted, 31.0 observed.

Why? Speculation the paper does not quite spell out, but the mechanism is visible: merged LoRA gives each task a *fresh* rank-32 workspace so tasks stop fighting over the same 32 directions, while replay supplies the signal that stops the dense weights drifting. Fresh capacity without a retention signal just overwrites faster; a retention signal with no fresh capacity has nowhere to put the new task.

### What did not work

**Weight anchors are the weak leg.** SI has no detectable main effect on Symbol-QA at all ($p=0.649$). Online EWC as a standalone is *worse than naive* in the sense that it damages learning: Diag drops to $86.0\pm15.1$ on Symbol-QA and $89.2$ on LLM-QA, where everything else sits at 96–100.

**SI × merged LoRA is actively harmful on Symbol-QA** ($-3.0$, $p<0.001$), and the paper gives a clean structural reason. SI stores one importance scalar per LoRA coordinate. Merged LoRA throws the adapter away and Kaiming-initialises a new $A$ with the same shape and name. The implementation then applies the *old* importance value to the *new* coordinate at the same row and column — a coordinate whose functional meaning is now completely different. Preserving the penalty correctly would require $H_{\mathrm{new}}=J_{t}^{\top}H_{\mathrm{old}}J_{t}$, which is dense even when $H_{\mathrm{old}}$ is diagonal, so a diagonal method structurally cannot represent it. The result is loss of *plasticity*, which is why it shows up on the diagonal, not in the decay.

Two configurations fail this way outright: `si_sd_merge` on Symbol-QA gets Final $6.8\pm5.9$ with Diag $95.0\pm8.0$ — on some seeds the diagonal collapses completely, on others training is fine. The mean describes a mixture of two behaviours, not a typical run.

**SD × Replay is negative on the natural-language sets** ($-3.5$ LLM-QA, $-10.3$ Real-QA). Self-distillation on current data buys much less once generative replay is already present — both are distillation against the same frozen teacher, so their signal overlaps.

**Growing state does not buy retention.** O-LoRA (keeps a frozen adapter per task, penalises overlap between $A$ matrices) changes retention by $-3.3$, $+1.0$, $+1.6$ against the matched merged-LoRA composition. Their sequential OSRM adaptation (initialise new $A$ from the smallest right singular vectors of stored past-task feature means) *loses* $-0.8$, $-11.5$, $-11.6$. Paying linear-in-tasks storage gets you nothing on the headline metric.

**But it buys general capability.** All methods catastrophically forget general ability. Base model averages 69.4% over GSM8K/MATH/MGSM/MMLU-Redux. After 100 tasks: O-LoRA 26.8% (LLM-QA) and 28.8% (Real-QA); merged LoRA 13.0% and 13.4%; OSRM 8.2% and 8.0%. After Symbol-QA everything drops below 8% — and MMLU-Redux sits at chance (25%) even when scored by direct letter log-likelihood, so this is not merely output-format corruption. Memorising 10,000 random symbol pairs destroys the model.

**Replay is a variance mechanism.** Seed standard deviations split cleanly: replay-bearing stacks give ±2 to ±6 points; non-replay configs give ±0.0 to ±1.3. A controlled re-run at generation temperature 0.7 versus 1.5 gives $21.2\pm7.9$ versus $20.8\pm2.1$ — identical means, ~4× difference in spread. Sharp low-entropy replay concentrates on few modes, and whether those modes happen to suit a seed's trajectory decides the run.

**Selection noise is real and the authors say so.** TSH picked `si_sd_replay_merge` on Symbol-QA; the final factorial ranks `sd_replay_merge` above it. The winner regressed from 23.4 to 18.5 on the new task order while the eventual leader held at 22.4 → 23.2. Since Symbol-QA tasks are exchangeable by construction, this is pure argmax-over-noisy-candidates regression, not order overfitting. TSH's 10-task rankings nonetheless correlate at Spearman $\rho = 0.93/0.95/0.90$ with the 100-task rankings.

## Worth Remembering

**Memorisation, not generalisation.** The evaluation queries are literally the training queries. A model could hold the association and still fail a paraphrase. Nothing here establishes that the knowledge is usable. For anyone hoping to use parametric memory as a [[RAG]] substitute, that gap is the whole question and this paper does not touch it.

**34.9% is still 65% forgotten.** Every survival curve keeps declining. Composition buys a longer half-life, not a floor.

**The anchors/allocation taxonomy is the durable part.** It is a genuinely useful way to file continual-learning methods: *what to preserve* is orthogonal to *where to put the new stuff*, and the paper shows empirically that the two dimensions interact rather than compose independently.

**The coordinate-identity problem generalises.** Any diagonal parameter-importance penalty ([[Regularization|regularisation]] of the EWC/SI family) is defined *in the basis where the importances were measured*. Reparameterise — merge an adapter, reinitialise, re-rank — and the penalty becomes meaningless numbers applied to unrelated coordinates. This is worth carrying to any setting that mixes weight-space regularisers with structural changes to the parameterisation.

**Task-level successive halving is transferable.** Allocating *task horizon* rather than iterations, with 10-task rankings already at $\rho \approx 0.9$ with 100-task outcomes, is a cheap method for any sequential-training search. The pairing with a clean $2^4$ factorial afterwards — search to narrow, factorial to get unbiased effects — is a good pattern; the search's own winner was demonstrably a winner's-curse pick.

**Practical caveats if you wanted to run this.** Ten epochs per task on 50–100 examples is very aggressive fitting (Diag ≈ 100% everywhere). Generative replay costs 300 generations of up to 384 tokens before every task, plus a frozen teacher forward pass on every minibatch — roughly 2× the compute of naive SFT plus generation. Merged LoRA is not reversible: once folded, you cannot detach an adapter to recover the base model. O-LoRA can, which may matter more in production than its 1-point retention edge.

**Open question the results raise.** Merged LoRA and replay carry the result; SI and SD are conditional. Would a replay + merge stack with *more* replay samples, or replay drawn from several past checkpoints rather than only $t-1$, beat the four-way stack? The replay budget ($N_R=300$) was never swept.

## Links

Related: [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[LoRA]] · [[Distilling the Knowledge in a Neural Network]] · [[Distillation]] · [[KL Divergence]] · [[Cross Entropy]] · [[Fine-Tuning]] · [[Regularization]] · [[Mode Collapse]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[Inject, Align, Recover- Staged Post-Training for Retrieval-Free Document Knowledge Internalization]] · [[Practical Bayesian Optimization of Machine Learning Algorithms]] · [[RAG]] · [[Memory]] · [[The Lottery Ticket Hypothesis]]

New topics worth writing: Catastrophic forgetting, Continual learning (task/class/domain-incremental), Elastic Weight Consolidation, Synaptic Intelligence, Fisher information, Generative replay, Learning without Forgetting, ReLoRA, O-LoRA and orthogonal subspace learning, Successive halving / Hyperband, Factorial experimental design and interaction effects, Sequential model editing, Plasticity–stability tradeoff
