---
title: "Self-Play Search Distillation for Large Language Model Reasoning"
authors: ["Lorenzo Molfetta", "Wai-Chung Kwan", "Giacomo Frisoni", "Luca Ragazzi", "Gianluca Moro", "Pavlos Vougiouklis", "Jeff Z. Pan", "Pasquale Minervini"]
year: 2026
arxiv: "2609.30936"
url: https://arxiv.org/abs/2609.30936
priority: Good-To-Read
read_on: 2026-09-30
tags: [paper, llm, vision, theory]
---
## The Core Idea

Board-game engines like MuZero already play better than any human. When they pick a move, they do not just output the move — the search leaves behind a pile of evidence: which alternatives it considered, how many simulations each got, what each branch was worth, and what actually happens if you play them out. That evidence is normally thrown away.

**Self-Play Search Distillation (SPSD) turns that thrown-away search evidence into text, and trains a language model on it.**

Why this is different from the usual synthetic-data story: when a model generates its own training data, the data is capped by the model. SPSD's teacher is a separate engine that is *superhuman at the thing it teaches* — comparing options and reasoning about consequences. The student never plays against the engine, never sees search output at test time, and the engine is frozen during data generation. So supervision quality is fully decoupled from the student's current ability.

The second half of the idea is that **the game rules are an executable checker**. Every claim in a generated trace ("if I play d4, the opponent can reply a1, and that is not yet a loss") is replayed in the actual game engine. If the replay disagrees, the row is deleted. So this is synthetic data with a verifier attached, not an LLM writing plausible-sounding rationales.

The surprising payoff: training only on Connect4, Domineering, Othello and Tic-Tac-Chess **transfers to maths it never saw**. Qwen3-4B-Base goes from 24.1 → 36.6 mean over six maths benchmarks, while its win rate on 15 held-out games goes 15% → 45%.

> [!NOTE] Self-Play Search Distillation
> Take a frozen search engine (MuZero family). At each state it visits, export the chosen move, the ranked alternatives, their values, and replayable continuations. Verbalise that into a chain of thought, throw away anything the game engine cannot reproduce, and post-train an LLM on it. ^spsd-def

What it unlocks: a knob for generating *unbounded amounts* of process supervision about decision-making, at zero human annotation cost, with rule variation as a free diversity dial.

## The Methodology

Three stages: build the experts, export and verify records, post-train the student.

### Stage 1 — the search experts

For each of four games (Connect4, Domineering, simplified Othello, Tic-Tac-Chess) they train an **EfficientZero** agent by self-play via LightZero. EfficientZero is a MuZero variant, so the search runs over *learned* latent dynamics. Settings: one residual block, 64 channels, Adam at constant lr 0.003, 500K env steps (1M for Othello), unroll 5, replay 100K segments, undiscounted returns ($\gamma = 1$), no reanalysis.

These experts do [[Monte Carlo Tree Search]] with the usual PUCT rule — pick the child maximising backed-up value plus a prior-weighted exploration bonus:

$$a_{\text{tree}} = \arg\max_a \left[ Q_k(s,a) + U_k(s,a) \right], \quad U_k(s,a) = P_k(s,a)\frac{\sqrt{N_{k,s}}}{1+N_k(s,a)}\left( c_1 + \log\frac{N_{k,s}+c_2+1}{c_2} \right)$$

and the search-improved policy is just normalised root visit counts:

$$\pi^E_k(a|s) = \frac{N_k(s,a)^{1/T_{\text{mcts}}}}{\sum_b N_k(s,b)^{1/T_{\text{mcts}}}}$$

This is the [[Mastering Chess and Shogi by Self-Play (AlphaZero)|expert iteration]] loop — search improves a decision at one state, the network generalises it to others. SPSD just taps the search output instead of only feeding it back into the network.

### Stage 2 — records, then replay

Data generation runs **50 MCTS simulations per move**. To spread out over the state space (the expert is deterministic, so it would otherwise produce the same game forever), each trajectory opens with a **uniformly random legal prefix of up to 8 moves**, then the expert controls both seats. Prefix moves are not training rows.

Each decision state exports:

$$m(s) = \left(s,\ A_g(s),\ a^\star(s),\ \pi^E(\cdot|s),\ v^E(s),\ B^E(s)\right)$$

— the state, the legal actions, the selected move, the root visit policy, the root value, and branch evidence $B^E(s)$ (visit count, edge value, replayable prefix per exported action). At most 8 root actions are exported.

**The verification pass.** Restore $s$ in the real game engine. Recompute board, active player, legal actions. If any field disagrees with the serialised record, or $a^\star \notin A(s)$, **throw the whole record away**. If an optional *alternative* branch fails to replay, drop just that branch.

**Picking the contrast.** After excluding $a^\star$, the exported legal action with the most visits becomes $a_{\text{alt}}$ (ties by export order). Its edge value, from the root player's view:

$$Q(a) = r(s,a) + \gamma\,\sigma\,V(T(s,a))$$

with $\gamma = 1$, and $\sigma = -1$ when the child flips the player to move (otherwise $+1$). Values are never rescaled across states or games.

A trace is allowed to *compare* two moves only if either $Q(a^\star) - Q(a_{\text{alt}}) \ge \Delta$ with $\Delta = 0.05$, or the target replays to a strictly better terminal outcome (win > draw > loss). Otherwise it falls back to a **target-only** trace, or a bare **fallback** trace stating only replay-supported facts about the chosen move. Opponent replies are followed to depth $D = 1$. Words like "win" or "loss" appear only when replay actually reached a terminal state.

Critically: **the raw numbers never enter the text.** Visit counts and $Q$ values decide *which evidence to keep*; the trace itself is prose about board effects.

### The two row families

1. **Move choice** (80% of rows). Prompt = rules + board + side to move + legal action handles + answer contract. Target = one boxed legal handle. The trace compares the chosen move against one alternative, narrates a verified opponent reply, then commits.
2. **State questions** (20%, split into six equal quotas). A model can pick a good move while misreading the board, so these probe the reading itself: occupancy of a named cell, legality of a handle, count of opponent immediate wins, legal-action count, full legal-action enumeration, and successor-state occupancy. Every answer is computed from the engine, not written by a model.

Splits are grouped by source trajectory, and each decision state belongs to exactly one family — so a state is never both trained on and probed.

Sizes: 5,000 examples per game, except Tic-Tac-Chess at 600 (small pool of verified states).

### Stage 3 — post-training

Written as $(x, y; e)$: $x$ is the board prompt the student sees, $y$ the target move, $e$ the expert trace — **hidden from the student's input**.

Three paths:

**SFT.** Train directly on $(e, y)$ — the trace is the target sequence. Sequence length 4,096, lr $2\times10^{-5}$, 2 epochs, effective batch 128.

**OPSD (on-policy self-distillation)** — the method that actually works. The problem with SFT: it fits states the *expert* visits, not the states a weaker student reaches. OPSD fixes the state distribution mismatch. The student samples its own continuation $y = (y_1,\dots,y_T)$ from $p_\theta(\cdot|x)$. A frozen teacher sees the *same prefix* but is also given the privileged trace:

$$q^e_{\bar\theta}(\cdot|x, y_{<t}) = p_{\bar\theta}(\cdot|x, e, y_{<t})$$

and the student minimises forward [[KL Divergence|KL]] at every visited prefix:

$$\sum_{u \in V} q^e_{\bar\theta}(u|x,y_{<t}) \log \frac{q^e_{\bar\theta}(u|x,y_{<t})}{p_\theta(u|x,y_{<t})}$$

Forward direction is deliberate — it punishes putting low probability on tokens the grounded teacher likes, keeping the student inside the teacher's support. (Contrast with reverse KL, which would let the student collapse onto one mode.) Settings: lr $5\times10^{-6}$ constant, AdamW, 1,000 steps, effective batch 16, [[LoRA]] $r{=}64,\alpha{=}128$, per-token divergence clip 0.06, gradient norm clip 0.1, rollout sampling $T{=}1.1, p{=}0.95, k{=}20$. Student thinking disabled, teacher thinking enabled. One H200.

**RuleBot-Distill** — the control. Identical OPSD machinery, identical student prompt, identical answer contract. The only change: decision states and selected moves come from a fixed *heuristic* bot, and the completion is a short rule-verified sentence with no ranked alternatives, no values, no replayed branches. This isolates the question: *does the search structure add anything beyond a verified correct action?*

## Ablation Studies and Experiments

Evaluation: **15 held-out Ludii games**, disjoint from training. Primary metric is mean FIDE score (loss=0, draw=0.5, win=1), plus win rate and **legality** (fraction of attempts producing a valid legal action — format failures, parse failures and illegal moves all count as failures). Out-of-domain: MATH500, AIME24, AIME25, AMC23, OlympiadBench, Minerva Math.

### Main table

| Model | Condition | Maths mean | FIDE % | Legal % | Win % |
|---|---|---|---|---|---|
| Qwen3-4B-Base | Baseline | 24.1 | 15.0 | 65.0 | 15.0 |
| | SFT | 30.3 | 19.9 | **34.9** | 20.0 |
| | RuleBot | 33.9 | 35.1 | 70.1 | 35.0 |
| | **OPSD** | **36.6** | **39.9** | **75.1** | **45.0** |
| Qwen3-8B (no think) | Baseline | 48.3 | 42.5 | 95.0 | 45.0 |
| | SFT | 46.8 | 52.5 | 95.0 | 55.0 |
| | RuleBot | 51.0 | 42.5 | 90.0 | 45.0 |
| | OPSD | 51.4 | 52.5 | 85.0 | 55.0 |
| Qwen3-8B (thinking) | Baseline | 74.9 | 51.0 | 100.0 | 53.7 |
| | SFT | 75.4 | **15.6** | **30.6** | **16.4** |
| | OPSD | 75.0 | 59.5 | 100.0 | 62.7 |
| Llama-3.1-8B | Baseline | 19.5 | 42.5 | 90.0 | 45.0 |
| | OPSD | 19.2 | 47.5 | 100.0 | 50.0 |

### What did not work

**SFT is actively destructive on strong models.** On the thinking-enabled Qwen3-8B, SFT drops legality from 100% → 30.6% and FIDE from 51.0 → 15.6. It learned the surface form of the linearised traces and lost the ability to emit a legal, parseable move. This is [[Fine-Tuning#catastrophic-forgetting|catastrophic forgetting]] in its most literal form: the model forgot how to follow the answer contract.

**SFT peaks early then decays.** The training-trajectory plots show SFT's held-out FIDE rising fast, peaking, then falling as training continues — and legality *and* maths fall with it. OPSD improves monotonically instead, because attaching the teacher to student-sampled prefixes keeps the update target on the states the student actually visits. This is the clearest argument in the paper for on-policy over offline supervision.

**Llama-3.1-8B gets essentially nothing on maths** (19.5 → 19.2 under OPSD). Games improve slightly (42.5 → 47.5 FIDE). So the transfer is backbone-dependent, not a general law.

**Thinking-enabled 8B maths is saturated.** 74.9 → 75.0. No headroom for this signal.

### The ablation that carries the paper: search structure vs. verified action

RuleBot-Distill gets the same on-policy machinery and the same verified correct move — it just lacks ranked alternatives and replayed consequences. On Qwen3-4B-Base:

- Win rate: 35% → **45%**
- Maths mean: 33.9 → **36.6**

So roughly 10 points of win rate and 2.7 points of maths come specifically from *the comparison structure*, not from being told the right answer. That is the load-bearing result. (RuleBot did win one benchmark — OlympiadBench 35.7 vs OPSD's 35.1.)

### Rule diversity

They built a second pool: 5 base games × 9 variants each = 50 environments. Each variant changes one local rule (scoring thresholds, legal winning directions, movement rules, opening locks, blocked-state outcome) while **preserving observation shape, board topology and action-space size** — so the interface is identical and the change is attributable.

At step 1,000, the variants curriculum beats the base curriculum on all three metrics:

| Curriculum | FIDE % | Legality % | Maths mean |
|---|---|---|---|
| Base (4 games) | 39.9 | 75.1 | 36.6 |
| + variants (50 envs) | **42.4** | **77.3** | **39.2** |

And it leads at *every* evaluated step, not just the end. Rule variation is a free diversity dial that improves both in-domain and out-of-domain performance under a fixed step budget.

### Did it actually learn MCTS-like reasoning?

The nicest analysis in the paper. Using GPT-5.6 to read each trace and extract which positions the model considered and which move it committed to, they then restore the real position and query the expert. Two numbers per episode:

- $Q_{50}$ — mean expert search value of the committed moves, from the mover's perspective
- **oracle@50** — how often the committed move matches the expert's preferred move

Wins concentrate in the high-$Q_{50}$, high-agreement corner for all four systems tested. But the *shape* differs: for the SPSD model (and GPT-5.5), high-value choices are strongly associated with wins **only when agreement is also high** — the model wins by picking what the search picks. Gemini 3.1 Pro keeps a high win rate at *intermediate* agreement with high-value moves — it finds good moves the search does not prefer. Opus 5 has a lower win rate even at high value and high agreement.

The SPSD model produces these moves from the plain game prompt, with all search values hidden at inference. So the agreement is a genuine **behavioural signature of distillation**: it internalised "rank actions by downstream consequence".

### Scale context

A separate scan across model sizes (Figure 1) shows game competence tracking maths accuracy — but **even frontier models do not saturate board-game reasoning**. This is a live headroom claim, not a solved benchmark.

### OPSD hyperparameter search (Appendix G, Qwen3-4B-Base, maths mean)

| Change | Result |
|---|---|
| Main config (fixed teacher, $\beta{=}0.25$, lr $5\times10^{-6}$, $r{=}64$) | 31.9 |
| Forward KL alone ($\beta{=}0$) | **4.0 — total collapse** |
| Cosine schedule | 31.5 |
| EMA teacher, decay 0.998, $r{=}128$ | **36.6** |
| EMA 0.997, $r{=}128$, 2,000 steps | 33.2 |
| $r{=}256$ | 35.4 |
| Effective batch 32 | 34.7 |

Three things fall out. An **exponential-moving-average teacher beats a fixed teacher** (36.6 vs 31.9) and its decay follows an inverted U peaking at 0.998. Pure forward KL with no generalised-JSD mixing destroys training. And **doubling the step budget hurts** (36.2 → 33.2) — more optimisation is not more transfer.

## Worth Remembering

**The honest framing is conditional.** The authors say so: "search-derived supervision improves performance when the student can convert value-ordered traces into decisions." It worked spectacularly on a 4B base model with lots of headroom, decently on 8B, and not at all for Llama's maths. This is not a universal recipe.

**The 4B result is the whole paper.** Qwen3-4B-Base is the only model where every metric moves together and the ablations separate cleanly. Treat the other rows as robustness checks, not replications.

**Rows count, not tokens.** 5,000 examples per game × 4 games, 80/20 move-choice vs state-question. That is a small corpus by post-training standards — the gains come from *what is in each row*, not volume.

**The verifier is the moat.** Everything checkable is checked: legal actions, transitions, terminal outcomes, and every narrated branch. Failed records are deleted, not repaired. This is what separates it from LLM-generated rationales, and it is also why the method only works in executable domains.

**Numbers stay out of the text.** Search values and visit counts are used for *selection* of what to say, then discarded. The student learns the comparison habit, not the engine's numbers. Compare [[Distilling the Knowledge in a Neural Network|distillation]], where the soft targets themselves are the signal.

**Games and maths move together.** The scale plot showing game competence tracking maths accuracy is a useful independent observation: board-game decision quality may be a cheap proxy for general reasoning, and it is far from saturated even at frontier scale.

**Practical caveats if you wanted to use this:**
- Do not use plain SFT on a strong instruction-tuned model. You will break its ability to follow the output contract. Use OPSD or something on-policy.
- Use an EMA teacher, decay near 0.998. The fixed-teacher gap is 4.7 maths points.
- Mix a divergence anchor into the KL; pure forward KL collapsed to 4.0.
- Stop at ~1,000 steps. 2,000 was worse.
- Build the rule variants. They cost nothing and win on every metric at every step.
- Budget the verification pass. Every record needs a full state restore and branch replay.

**Open questions.** Why does a 4B base model transfer and Llama-3.1-8B-Instruct not — is it the base/instruct distinction, headroom, or tokeniser/family? Does the gain survive the model also being trained on maths, or is it filling a gap that real maths data fills better? Would a continual curriculum of growing environment pools keep paying, as the conclusion suggests? And does the $D=1$ reply depth matter — would deeper replay help, or just add noise?

## Links
Related: [[Monte Carlo Tree Search]] · [[Mastering Chess and Shogi by Self-Play (AlphaZero)]] · [[Mastering the game of Go with deep neural networks (AlphaGo)]] · [[Distilling the Knowledge in a Neural Network]] · [[KL Divergence]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Chain of Thought]] · [[Imitation Learning]] · [[On-Policy vs Off-Policy]] · [[Fine-Tuning]] · [[LoRA]] · [[Model-Based vs Model-Free RL]] · [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Learning from Teacher Continuations at Student States]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[What Does Privileged Information Add to On-Policy Self-Distillation]] · [[Distillation]] · [[Credit Assignment]] · [[Value Function]] · [[Exploration vs Exploitation]] · [[Test-Time Compute]]

New topics worth writing: EfficientZero and sample-efficient MuZero variants, Expert Iteration (ExIt) as a general framework, LightZero benchmark suite, procedure cloning, Stream of Search, generalised Jensen-Shannon divergence as a distillation objective, EMA teachers in self-distillation, Ludii general game system, unsupervised environment design and POET, verifier-grounded synthetic data generation, Absolute Zero / self-proposed task curricula, SPIRAL and game self-play for LLM reasoning
