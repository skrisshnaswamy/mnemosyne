---
title: "WHALE: A Simple Recipe for Joint Harness-Weight Optimization"
authors: ["Haechan Kim", "Yoonho Lee", "Gisang Lee", "Chelsea Finn", "Kangwook Lee"]
year: 2026
arxiv: "2609.00196"
url: https://arxiv.org/abs/2609.00196
priority: Good-To-Read
read_on: 2026-09-06
tags: [paper, llm, rl]
---
## The Core Idea

An LLM agent is two things, not one. There is the **model** (the weights) and there is the **harness** — the ordinary code wrapped around the model that decides what goes into the context, how tools are described and called, what happens when a tool errors, how many turns are allowed, and when to stop. Almost all training work treats the harness as a fixed background and only moves the weights. Almost all "agent optimisation" work does the reverse.

The trouble is that each one caps the other. A retrieval agent with perfect weights cannot use evidence a brittle harness never fetched. A perfect retriever cannot help a model that cannot read the passage and commit to an answer. So whichever you freeze becomes the bottleneck, and you cannot even tell which one it is by looking at a single training curve.

WHALE (Weight-Harness Alternating LEarning) is the obvious-in-hindsight fix: **alternate**. Train the weights for a short while with the harness frozen, then search for a better harness with the weights frozen, then repeat. Nothing about that is deep — it is coordinate descent — but the paper is the first careful, budget-matched study of doing it over *executable harness code* rather than just prompt text, and the interesting content is entirely in the scheduling question: **how long should each phase run before you switch?**

> [!NOTE] Harness
> The executable program surrounding the model: system/user prompt templates, tool schemas, input/output formatting, error feedback, context management, retry budgets, turn limits, and the termination rule. Everything except the weights, the data, the verifier, and the environment's own rules. ^harness

Two findings make this more than an engineering note. First, **which component is the bottleneck depends entirely on the domain**. In search QA, harness search alone matched the peak accuracy of full weight training using **5.79%** as many rollouts. In maths, harness search alone got essentially nowhere (0.42% accuracy) until a *small* weight update made the exact same harness search suddenly productive. Second, **one big pass per component loses to many small alternating passes**, in both accuracy and cost, because a long phase over-fits its component to a counterpart that is about to change.

## The Methodology

Let $\theta$ be the weights and $h$ the harness. Together they define a trajectory distribution $\pi_{\theta,h}(\cdot\mid x)$ over agent rollouts. With a task distribution $\mathcal{P}$ and a binary verifier $R$ (1 only if the final answer is right — no intermediate reward):

$$J(\theta,h) \triangleq \mathbb{E}_{x\sim\mathcal{P}}\,\mathbb{E}_{\tau\sim\pi_{\theta,h}(\cdot\mid x)}[R(x,\tau)].$$

The loop is just:

$$\theta_{k+1}=\operatorname{ModelUpdate}(\theta_k;h_k,\mathcal{D}_{\text{weight}}),\qquad h_{k+1}=\operatorname{HarnessSearch}(h_k;\theta_{k+1},\mathcal{D}_{\text{harness}}).$$

Both operators are black boxes; you can swap in anything. Their instantiations:

**Weight phase — online rejection-sampling fine-tuning (RSFT).** Sample $G=8$ rollouts per prompt from a frozen copy $\theta_{\text{old}}$ under the current harness $h_k$. Throw away everything the verifier scores 0. Do plain supervised learning on the survivors, token-normalised, and only over the *model-generated* tokens $\mathcal{I}(\tau)$ (user prompts and tool outputs are masked out):

$$\widehat{J}_{\text{weight}} = \frac{1}{\sum_{(x,\tau)\in\mathcal{S}^+_s}|\mathcal{I}(\tau)|}\sum_{(x,\tau)\in\mathcal{S}^+_s}\sum_{t\in\mathcal{I}(\tau)}\log p_\theta(\tau_t\mid x,h,\tau_{<t}).$$

Then sync the new weights to the rollout workers and repeat. This is STaR/ReST-style self-training — simpler and more stable than a policy gradient, and it needs no critic (contrast with [[Proximal Policy Optimization Algorithms|PPO]]).

> [!NOTE] Rejection-sampling fine-tuning
> Generate many of your own answers, keep only the verified-correct ones, and fine-tune on them with ordinary [[Cross Entropy|cross-entropy]]. A reward-filtered SFT step, not a gradient of the reward. ^rsft

**Harness phase — Meta-Harness (MH).** An LLM proposer (Claude Opus 4.7) is given filesystem access to an *archive* $\mathcal{A}_j$ of past harness source files plus their *artifacts* $\mathcal{E}_j$ (aggregate score, per-example pass/fail, saved trajectory logs). One proposer session writes $M=3$ candidate harnesses. Each candidate is scored by actually running it:

$$\widehat{J}_{\text{harness}}(\theta,h)=\frac{1}{|\mathcal{D}_{\text{harness}}|}\sum_{x\in\mathcal{D}_{\text{harness}}}\widehat{\mathbb{E}}_{\tau\sim\pi_{\theta,h}(\cdot\mid x)}[R(x,\tau)],$$

with $|\mathcal{D}_{\text{harness}}|=256$ and **one rollout per candidate–example pair** — so 768 rollouts per iteration, cheap. The best-scoring harness in the whole archive is marked accepted; later proposals can refine any candidate or revert to an earlier one.

**Budgets.** The default schedule is $(E,I)=(0.6,6)$: 0.6 epochs of weight training then 6 harness-search iterations, per cycle, drawing from a persistent sampler so no data is revisited.

**Adaptive WHALE** removes $(E,I)$ entirely with a per-phase patience rule driven *only by training signal, never a validation set*. The weight phase stops when the sliding-window mean verifier reward has not set a new phase-best for $P_w$ steps (window = min length = patience = 0.2 epochs). The harness phase stops when the archive's best training score has not improved for $P_h=2$ iterations, minimum 6 iterations.

**Setup.** Qwen3.5-2B for SearchQA and Math, Qwen3.5-4B for Chess. Learning rate $10^{-7}$, prompt batch 256, SFT minibatch 64, temperature 1.0, top-$k$ 20. Response limits 8,192 / 8,192 / 16,384 tokens. Evaluation is $\operatorname{mean@8}$.

**Domains.** SearchQA (Search-R1 setup, HotpotQA + NQ training, 7-dataset 700-question test set, FAISS over Wikipedia 2018); Math (DAPO-Math-17K training, AIME 2024+2025 test, Python interpreter as tool); Chess puzzles (Lichess, UCI moves, one wrong legal move ends the rollout).

Crucially, the search space is bounded: the harness may rewrite prompts, tool I/O formatting, error feedback, retry budgets, turn limits, retrieval parameters, query rewriting, board presentation, move parsing. It may **not** touch the model, the data, the verifier, the environment transitions, or the reference answers. In chess it explicitly may not generate or search for a move itself.

## Ablation Studies and Experiments

**Main table** ($\operatorname{mean@8}$ %, WHALE at $(0.6,6)$):

| Method | SearchQA avg | Math (AIME) | Chess |
|---|---|---|---|
| Harness-only | 38.29 | 0.42 | 19.82 |
| Weight-only | 38.27 | 15.42 | 22.17 |
| FST (prompt-only search) | 35.34 | 17.92 | 25.68 |
| **WHALE** | **48.34** | **24.79** | **29.83** |

WHALE wins on all ten individual benchmarks, so the gain is not carried by one subset. The single-component ordering flips across domains — tied in SearchQA, weight-dominant in Math and Chess — and WHALE beats the stronger baseline by 7.67–10.05 pp either way.

**The FST control is the sharpest comparison.** Fast–Slow Training is re-implemented with WHALE's *own* update methods and *own* schedule, but with the harness search restricted to the system and user prompts of $h_0$. So the 4.15–13.00 pp gap measures exactly one thing: the value of searching over executable code rather than text. Notably FST is *worse than both single-component baselines* in SearchQA (35.34 vs ~38.3) — prompt-only adaptation actively hurt there.

**Where the work is actually done (behaviour metrics).**

*SearchQA is harness-dominant.* Retrieval accuracy (fraction of trajectories that fetch at least one document containing the reference answer): weight-only barely moves it; harness-only takes it from 26.88% → 60.61% with far fewer rollouts, because the harness controls query rewriting, top-$k$, and document ranking. WHALE reaches 65.41%. But answer *extraction* reverses: weight-only peaks at 79.93%, harness-only at 57.28%. Format compliance both fix (97.07% vs 99.98%). So each channel owns different capabilities.

*Math is model-dominant.* The bottleneck is truncation. 95.83% of base responses hit the token cap and die mid-sentence with no `\boxed{}`. Weight-only drops that to 30.83% by learning from correct rollouts that terminated in time. Harness-only tries response caps, turn limits and final-answer recovery and cannot beat the base model's habit of reasoning in prose forever — its format accuracy goes 0.00% → 0.63% for **46,080 rollouts**. Inside WHALE, after one small weight update, the *cycle-1 harness search* raises format accuracy 0.83% → 4.38% (+3.54 pp) in **4,608 rollouts**. Same search, ten times cheaper, five times the gain, purely because the model changed first.

**Alternating vs stagewise.** Stagewise = spend the whole weight budget, then the whole harness budget, once. Best stagewise points were $(E^*,I^*)=(2.7,32)$ → 43.02% in SearchQA and $(4.3,33)$ → 15.63% in Math. WHALE at $(0.6,6)$ beats these by 5.32 and 9.16 pp and passes the final stagewise number after only **29%** and **49%** of the rollouts. The diagnosis is *conditional over-optimisation*: $J(\theta,h)$ is not separable, so a long weight phase fits the model to $h_0$'s quirks and then the harness changes underneath it. In Math the 60-iteration stagewise harness search kept improving its 256-example training score while gaining only +0.21 pp on test — textbook over-fitting to $\mathcal{D}_{\text{harness}}$.

**Phase length sweep** (best $\operatorname{mean@8}$ %):

| $(E,I)$ | SearchQA | Math |
|---|---|---|
| (0.2, 2) | 46.63 | 16.67 (run destabilised) |
| **(0.2, 6)** | **50.09** | **28.33** |
| (0.6, 2) | 48.84 | 17.92 |
| (0.6, 6) | 48.34 | 24.79 |
| (1.0, 10) | 45.93 | 24.79 |

**What did not work.** $(0.2,2)$ in Math is the *noisy extreme*: with only two proposal iterations of evidence, a candidate that scored well by chance got accepted, the model then adapted to it, and the run destabilised and was killed. At the other end, scaling budgets up never helped in either domain — accuracy falls monotonically along $(0.2,6)\to(0.6,6)\to(1.0,10)$. And the two budgets do not decouple: raising $E$ from 0.2 to 0.6 *helps* at $I=2$ in Math (16.67→17.92) but *hurts* at $I=6$ (28.33→24.79). You cannot tune them one at a time.

> [!NOTE] Noisy vs over-optimised extremes
> Each phase must run long enough to tell a real improvement from sampling noise, and stop before it fits itself to a frozen counterpart that is about to move. Too short → you accept a lucky harness and train the model into it. Too long → you land in a single-axis local optimum. ^phase-length-tradeoff

Also worth noting: even in the *model-dominant* domain, the schedule that runs harness search most often per training epoch wins. Each small weight update renews the gains a bounded harness search can find.

**Adaptive WHALE.** SearchQA: **52.82%**, the best point in the whole paper — +4.48 pp over $(0.6,6)$ and +2.73 pp over the best hand-tuned schedule, with 23% fewer rollouts. Math: 26.46%, +1.67 pp over $(0.6,6)$ but **1.87 pp below** the best hand-tuned $(0.2,6)$. So it is not free lunch — it beats a reasonable default reliably, and beats an exhaustively tuned schedule sometimes. The realised median phases were 0.24 and 0.29 epochs with $I=7$ in both domains, landing right on the swept optimum, while individual phases stretched to 1.16 epochs and $I=13$ when the signal kept improving.

## Worth Remembering

- **A short harness search is a cheap diagnostic.** Before spending hundreds of thousands of rollouts on weight training, run 6 harness iterations (≈4.6k rollouts). If accuracy jumps, you were harness-bottlenecked and training would have been the expensive way to buy the same thing.
- **Rollout counts exclude proposer compute.** All the "harness search is 20× cheaper" claims count only target-agent rollouts. The Claude Opus proposer sessions are not in the ledger. If your proposer is a frontier API model, the real cost picture differs.
- The trajectory appendices are unusually informative. In SearchQA the searched harness *raised* the turn budget from 2 to 4, forced at least one retrieval, injected a between-turn state line listing already-retrieved titles, and added a verification checklist to the user prompt. In Math the harness left the opening prompt byte-identical and only changed the turn budget and appended a post-execution reminder to emit `\boxed{}` — the prose-to-code switch came from the weights, exactly as the model-dominant story predicts. In Chess the harness added a grouped, annotated legal-move list `(mate)/(check)/(capture)` and a disambiguation rule that bracketed `[uci]` tokens inside `<think>` are scratch and the last `<move>` tag outside it is the commitment. That rule alone kills the base model's dominant failure mode (2,025 of 2,048 base rollouts hit the token cap, mean 8,112 assistant tokens vs 764 under WHALE).
- Some of what "harness search" buys is arguably just fixing a deliberately weak $h_0$ — two-turn budgets, one truncated passage, no system prompt in Math. The baselines are honest in that everyone shares $h_0$, but a strong hand-written harness might close much of the gap in SearchQA.
- **The verifier is fixed and binary throughout**, shared by training acceptance, harness scoring, and test evaluation. This matters: harness search directly optimises the same number used to grade it, on 256 examples, with one rollout each. Reward-hacking pressure is real and the Math stagewise result shows it landing.
- Learning rate is $10^{-7}$ — extremely small. The "small weight update unlocks harness search" claim is at that scale; whether a coarser optimiser behaves the same is untested.
- Only 2B/4B models, three domains, and (as far as reported) single seeds per schedule. The AIME test set is 60 problems, so Math differences of a few points sit near the noise floor.
- The general shape — alternate two coupled updates, each conditioned on a frozen counterpart — is coordinate descent, and shares its failure modes with adversarial alternation in [[Generative Adversarial Networks|GANs]]: whoever runs too long over-fits to a stale opponent.

## Links

Related: [[Training language models to follow instructions with human feedback]] · [[Proximal Policy Optimization Algorithms]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[In Context Learning]] · [[Dynamic Important Example Mining for Reinforcement Finetuning]] · [[TTPO- Test-Time Policy Optimization]] · [[Is Next-Chunk Reasoning RL Really Better than SFT- Revisiting Training Strategies under no-CoT Data]] · [[Chain-of-Experience for Continual LLM Improvement]] · [[What Makes Good Agentic Data- An ACE Lens on Data Generation for LLM Agents]] · [[The Handoff Tax- Continuing Non-Native Trajectories in LLM Agents]] · [[AgentMercury- Your Agent Can Synthesize Verifiable Environments for Business Scenarios at scale]] · [[Autonomous Mathematical Discovery in an Open-World Multi-Agent Environment]] · [[Generative Adversarial Networks]] · [[The Bitter Lesson (essay)]] · [[Cross Entropy]] · [[Regularization]]

New topics worth writing: Meta-Harness, STaR / ReST self-training, rejection-sampling fine-tuning, coordinate descent, Search-R1, ReTool, DAPO, GEPA and reflective prompt evolution, DSPy prompt compilation, automated design of agentic systems, LLM-as-a-judge verifiers, ReAct
`</document>`
