---
title: "Exploring Collaboration between a language and a non-language agent"
authors: ["Harini S", "Somesh Singh", "Yaman K Singla", "Rajiv Ratn Shah", "David Doermann", "Balaji Krishnamurthy"]
year: 2026
arxiv: "2609.00474"
url: https://arxiv.org/abs/2609.00474
priority: Good-To-Read
read_on: 2026-09-06
tags: [paper, transformers, llm, rl, vision]
---
## The Core Idea

When an LLM uses a specialist tool, the tool has to speak. A chess engine is asked "what do you think of this position?" and it answers with a few lines of text: `Nc2: P=0.34, +0.12; e4: P=0.21, +0.08`. That text is a summary. The engine's actual understanding lives in a 1024-dimensional activation vector that encodes king safety, pawn structure, piece coordination, and learned look-ahead. Squeezing that into a ranked move list throws almost all of it away.

The paper names the cost of that squeeze the **verbalization debt**, and measures it.

The fix is to skip the text. Take the engine's hidden activations, push them through a small learned MLP that maps them into the LLM's embedding space, and paste the result into the LLM's context as 32 continuous "state tokens" — sitting right next to ordinary word tokens. The LLM then attends over words, actions, and engine-state in one single stream. They call this **latent state internalization**, and the resulting model **LLAMIA**.

> [!NOTE] Verbalization debt ^verbalization-debt
> The performance you lose when a non-language agent must compress its internal state into text before an LLM can read it. It is not a constant tax: it is small when text happens to carry a good proxy for the answer, and total when it does not.

The key evidence: a single 14B LLAMIA beats GPT-5.1-with-engine-tool on all six chess tasks, and on one task — ranking chess puzzles by how *interesting* humans find them — every text-mediated system scores Spearman $\rho \leq 0.12$ regardless of size, while LLAMIA reaches $0.52$. Interestingness depends on the shape of the whole policy distribution and the value landscape across candidate moves. No engine text output contains that. The channel, not the model, was the limit.

Why it did not exist before: prior latent-token work (CoCoNut, Token Assorted) moves an LLM's reasoning into its *own* hidden states. Cross-modal injection (PaLM-E, RT-2) projects *raw sensory input*. Nobody had piped a **separate, frozen, pretrained non-language agent's processed policy/value state** into an LLM's reasoning trace.

## The Methodology

**The two agents.**
- Subagent $G_\psi$: Lc0-BT4, the strongest open chess engine — a 15-layer Transformer encoder, 240M params, ~2810 Elo with no search. Frozen throughout. Read at **layer 14 of 15** (penultimate block), giving $\bm{h}_s \in \mathbb{R}^{1024}$.
- Backbone $\pi_\theta$: Qwen3 (4B / 8B / 14B). Chain-of-thought mode off. Tool calls in Hermes format, natively supported.

**LatentBridge $H_\varphi$.** A three-layer MLP with GeLU, copied in spirit from the projector in vision-language models like LLaVA:

$$\bm{z}_t = H_\varphi\big(G_\psi(s_t)\big) \in \mathbb{R}^{k \times e}, \quad k = 32$$

where $e$ is the LLM hidden size. A `<state>` marker anchors the injection site; the 32 positions after it have their embeddings overwritten with $\bm{z}_t$ before the forward pass.

**The trace.** One rollout interleaves three token types:

$$\tau = \big(\bm{z}_0^{1:k},\; w_{1:j_1},\; a_1,\; \bm{z}_1^{1:k},\; \ldots,\; w_{\text{final}}\big)$$

- $w$ = language tokens (the LLM's reasoning, plus the engine's text reply)
- $a$ = action tokens (moves that actually advance the board)
- $\bm{z}$ = 32 latent state tokens

The LLM calls `get_policy` **when it chooses to**. It can call it on the current board, or on a hypothetical board reached by a candidate move — a counterfactual. Every board-mutating call (`make_move`, `undo_move`, `reset_position`) triggers a fresh engine forward pass and a fresh set of 32 tokens. Read-only calls do not. So the token cost is capped at 32 per state change, regardless of how deep the analysis goes.

The ablation `LLAMIA-Verb` is identical in every way except $k = 0$: same backbone, same data, same reward, same DAPO recipe, text only.

**Stage 1 — projector alignment.** LLM frozen, train $H_\varphi$ only. 5M (state, engine-policy) pairs from Lichess, sampled evenly over opening/middlegame/endgame, split by *game ID* not position. Plain [[Cross Entropy|cross-entropy]] on producing the engine's top move:

$$\mathcal{L}_{\text{Stage 1}} = -\mathbb{E}\,\log \pi_\theta\big(\pi(s) \mid \bm{z}_s, \text{prompt}\big)$$

Four prompt types are rotated (evaluate, principal variation, legal moves, describe) so the projector does not overfit one output format. LR 2e-4, batch 256, 2 epochs (~39K steps). Because the LLM is frozen, no risk of forgetting language.

**Stage 2 — RL with DAPO.** Unfreeze both $\pi_\theta$ and $H_\varphi$, train jointly. DAPO is a group-relative [[Proximal Policy Optimization Algorithms|PPO]] variant with asymmetric clipping and no learned critic — advantage is just the group-normalised reward $\hat{A}_t = (\mathcal{R}(\tau) - \mu_G)/\sigma_G$ over $G=8$ rollouts of the same prompt.

$$J(\theta,\varphi) = \mathbb{E}_\tau\Bigg[\frac{1}{|\mathcal{I}_{\mathrm{gen}}|}\sum_{t \in \mathcal{I}_L \cup \mathcal{I}_A} \min\Big(r_t \hat{A}_t,\; \mathrm{clip}(r_t, 1-\varepsilon_l, 1+\varepsilon_h)\hat{A}_t\Big)\Bigg]$$

Gradient is taken only over positions the LLM *generated* — language tokens $\mathcal{I}_L$ and action tokens $\mathcal{I}_A$. State-token positions are agent-injected and masked out. But [[Backpropagation|gradients still flow back through them]] into $H_\varphi$, so the projector learns *what to present* while the LLM learns *how to read it*.

Hyperparameters: LR 1e-5 for $H_\varphi$, 1e-6 for $\pi_\theta$ (10× lower — the LLM is barely nudged), batch 128, 3000 steps, KL coeff 0.01, clip $\varepsilon_l/\varepsilon_h = 0.2/0.28$, rollout temperature 1.0, eval temperature 0.0, 32K context. 32 × A100-80GB, FSDP + vLLM rollouts + Ray, with lc0 co-resident at 3 GiB/GPU.

**LLAMIA-Bench** — six tasks, each unsolvable alone:

| Task | Metric | Notes |
|---|---|---|
| BC-MAIA | move-match % | predict what a human at Elo 1100–1900 plays, 5 buckets |
| BC-Wild (OOD) | move-match % | GM-25, Low-Time (<10% clock), ΔElo (>500 gap) |
| Puzzle Difficulty | Spearman $\rho$ | ground truth = Glicko-2 rating from millions of solve attempts |
| Puzzle Interest | Spearman $\rho$ | community up/downvotes, −100 to +100 |
| Move Annotation | BLEU-2 | Planning + Comparative subcategories |
| Game Commentary | G-eval | new dataset: 1,900 narrated games from Agadmator's YouTube |

FEN-level disjointness is enforced across Stage 1, Stage 2, and test.

## Ablation Studies and Experiments

**Headline table (14B, DAPO, everything else matched):**

| System | BC-MAIA | BC-Wild | Diff. $\rho$ | Interest $\rho$ | Rationale | Comm. |
|---|---|---|---|---|---|---|
| GPT-5 text only | 28 | 22 | 0.30 | 0.12 | 27.0 | 0.23 |
| GPT-5 + Lc0 (text tool) | 45 | 40 | 0.48 | 0.10 | 37.5 | 0.55 |
| Qwen3-14B + Lc0 (untrained) | 39 | 33 | 0.28 | 0.05 | 18.8 | 0.15 |
| **LLAMIA-Verb-14B** | 45 | 39 | 0.45 | 0.08 | 33.2 | 0.40 |
| **LLAMIA-14B** | **53** | **49** | **0.71** | **0.52** | **45.8** | **0.75** |

Dedicated experts for reference: Allie-Adaptive-Search (trained on 93M games) gets 55 on MAIA and 45 on Wild. LLAMIA uses 20K games on a general backbone, lands *inside* the expert band in-distribution, and beats the best expert by +4 pp out-of-distribution.

**The gap is ordered, and the ordering is the argument.** Going text → latent at fixed 14B:

- Interest: 0.08 → 0.52 (**6.5×**). No verbal proxy exists at all.
- Commentary: 0.40 → 0.75. Per-move loss compounds over 30+ moves.
- Difficulty: 0.45 → 0.71. Solution length is a *partial* proxy in the principal variation.
- Rationale: 33.2 → 45.8. PV inference covers much of "Planning".
- BC: 45 → 53. The engine's top-$k$ already covers most human moves.

**It is not scale.** LLAMIA-**4B** scores 38 on Interest; LLAMIA-Verb-**14B** scores 8. The debt widens throughout DAPO training, reaching 2–3× by convergence, at all three backbone sizes.

**What did not work — the controls that rule out the boring explanations:**

| Control | BC-M | Diff. | Int. | Comm. |
|---|---|---|---|---|
| LLM-Only (RL, no engine) | 34 | 0.22 | 0.07 | 0.13 |
| LLM-ChessCLIP (RL, 32 slots, raw board encoder) | 39 | 0.24 | 0.08 | 0.29 |
| Qwen3+Lc0 untrained tool use | 39 | 0.28 | 0.05 | 0.15 |
| LLAMIA-Verb (RL, text) | 45 | 0.45 | 0.08 | 0.40 |
| LLAMIA-SFT (latent, no RL) | 51 | 0.66 | 0.48 | 0.58 |
| LLAMIA (latent, RL) | 53 | 0.71 | 0.52 | 0.75 |

- **RL cannot manufacture the expertise.** LLM-Only lands *below* untrained tool use.
- **It is not extra embedding capacity.** ChessCLIP gets the same 32 slots, filled with a board encoder instead of the engine's state, and recovers almost nothing.
- **Shuffling LLAMIA's own tokens collapses it back toward Verb** while keeping the token count identical: Interest goes 0.52 → 0.46 (shuffle-4) → 0.38 (shuffle-8). Degradation is fastest on Interest, slowest on BC — the same ordering again.
- **Prettier text is not the answer.** A hand-built three-line template from engine stats hits 29.4 BLEU-2 / 0.31 G-eval; rewriting it with GPT-5 gets 32.1 / 0.39. Both below the plain verbal tool (37.5 / 0.55).
- **A linear probe on the frozen engine is worse than an LLM with no engine.** A 0.7M-param MLP on layer-14 activations scores BC 14, Difficulty 0.15, Interest **−0.07**. The engine's latent state is *not* directly decodable into these human-aligned targets; the LLM's language knowledge is doing real work on top of it.

**Where SFT vs RL splits.** On single-step tasks (Interest, Difficulty, BC), latent SFT alone recovers most of the debt and RL adds ~0.04. On multi-step tasks it flips: latent SFT→DAPO gains **+0.17** G-eval on commentary, more than double the verbal SFT→DAPO gain (+0.08). The reading of the state is perceptual and learnable by supervision; the *strategy* of when to query and what to compare needs RL — and only pays off when there is something worth comparing.

**Emergent collaboration strategies.** A GPT-4o judge ($\kappa = 0.78$ vs humans) labelled 500 episodes per task into five patterns. LLAMIA switches strategy by task: engine-follow 65% for gameplay, consult-then-override 48% for BC, counterfactual-query 40% for commentary. LLAMIA-Verb collapses to engine-follow 62–76% on *every* task. Strategy entropy: LLAMIA reaches $H = 1.53$ nats (95% of the $\ln 5 = 1.61$ maximum), Verb plateaus at 0.93. Both prompts are identical and neither reward mentions counterfactuals. The verbal channel returns the same compressed summary however you query it, so there is nothing to learn about *how* to ask.

**Cost is roughly neutral.** LLAMIA invokes 1.9 times per query vs Verb's 2.9. Each latent call costs 182 tokens vs 150. Net: 346 vs 435 tokens per query, 1.4 s vs 2.2 s latency at 14B. Training is within ~6% (22.6 vs 21.4 GPU-hours at 14B).

**Other sweeps.** $k \in \{4,8,16,32,64\}$: average 45.1 → 49.7 → 54.9 → **56.9** → 56.7. Saturates at 32. Layer choice: blocks 12–14 are within noise of each other on probe tasks; they picked 14 by lowest held-out Stage-1 loss, which agrees with interpretability work locating value and look-ahead features there. Engine strength: swapping BT4 for weaker Lc0 nets degrades monotonically (avg 56.9 at ~2810 Elo → 38.7 at ~2000 Elo), so the latent channel is transmitting genuine capability, not just "some vector".

**Two things that broke during development.** Rewarding top-1 exact match on behavior cloning **collapsed training** — signal too sparse, near-zero gradients. They switched to normalised top-3 rank (1.0 / 0.5 / 0.25 / 0). Separately, skipping Stage 1 entirely caused instability through the first 40% of Stage 2.

**Human study** ($n=12$, all ≥1700 Elo, screened by a 20-segment calibration task). LLAMIA-14B is identified as the bot in only **39%** of trials — below chance. LLAMIA-Verb-14B, same backbone and budget, is caught **72%** of the time; raters said its moves "felt too consistent" and it "never played a dubious move". On commentary, LLAMIA is preferred in 80.2% of judged pairs, and the **Insight** gap (1.70 Likert points) is larger than the **Accuracy** gap (1.10). Text preserves *what* is on the board; it loses *why*. Best result: participants read the commentary and then guessed the position evaluation. On simple positions both systems help equally ($\Delta r = 0.03$); on complex positions the gap opens to $\Delta r = 0.41$ (0.69 vs 0.28). Loss from verbalization scales with position complexity.

**Go transfer.** Same recipe, KataGo-b18 as subagent, 361 board intersections as spatial tokens, only the first MLP layer width changed to 384. With 8K training positions, LLAMIA-Go-14B hits 48/50 top-1 human move-match at 5k/5d ranks, matching rank-calibrated KataGo-HumanSL and beating the verbal control by ~10 points at every scale.

## Worth Remembering

- **The interface is a design variable, not plumbing.** Two systems with identical weights, data, reward, and compute differ by 6.5× on one task purely because of how the subagent talks. If you are wiring a specialist model behind an LLM, the API schema is a modelling decision.
- **Latent tokens are read differently depending on the prompt.** On a fixed back-rank mate position, conditioning on "2000 Elo" concentrates attention on the mating geometry; "1100 Elo" disperses it to material. Same 32 vectors. The projection behaves like a perceptual input the LLM interprets in context, not a static feature block.
- **The deployment story is the bridge.** Internalization needs weight access, so it cannot be applied to GPT-5 directly. LLAMIA is meant to sit in between: it internalizes the engine and speaks natural language outward, giving closed models indirect access.
- **Limitations the authors admit.** Nearly all evidence is chess. Go is one task only. You need the subagent's activations, so closed-weight specialists are out. The projector must be retrained per subagent (they retrain from scratch for each of the six Lc0 variants).
- **Confound they flag themselves.** Transformer-family subagents benefit more from a bigger LLM backbone (Interest 3.1×, Commentary 2.7×) than SE-ResNet ones — but architecture, parameter count, and MLP-projector compatibility are all tangled together. Genuinely open question whether subagent *architecture* matters for internalization beyond raw strength.
- **Sloppiness to note.** Table 17 ships with a literal `XX.X` placeholder for GPT-5+Lc0 puzzle-solving. Table 20's BT4 row reports BC 56 / avg 56.9 while the main results table gives BC-MAIA 53 for the same model — different aggregation, but unexplained.
- **The probe result is the most interesting negative.** If the engine's latent state trivially encoded difficulty and interest, a small MLP would read them out. It cannot ($\rho = -0.07$ on Interest). The signal only becomes usable when a language model with world knowledge attends over it. This is *joint* reasoning, not decoding.
- **Follow-ups.** Does the LatentBridge transfer across LLM backbones without retraining? Can you internalize *several* subagents at once, and do their token blocks interfere? What happens if the subagent is not frozen? And a practical one: how much of the 32-token budget is redundant — is there a Matryoshka-style ordering where the first 8 carry most of it? ([[Matryoshka Representation Learning]] suggests the answer might be yes.)

## Links

Related: [[Attention Is All You Need]] · [[Mastering Chess and Shogi by Self-Play (AlphaZero)]] · [[Mastering the game of Go with deep neural networks (AlphaGo)]] · [[Proximal Policy Optimization Algorithms]] · [[Training language models to follow instructions with human feedback]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[In Context Learning]] · [[LatentPress- Context Compression Beyond Text and Vision]] · [[Latent Action as Intention Enables Efficient Future Imagination for World Action Models]] · [[Cross Entropy]] · [[Backpropagation]] · [[Distilling the Knowledge in a Neural Network]] · [[Multi-Agent Reinforcement Learning]] · [[Matryoshka Representation Learning]] · [[EXIMO- VLM Guided Exploration of VLA Policies]]

New topics worth writing: DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization), Leela Chess Zero / BT4 architecture, Maia and human-move behaviour cloning, G-eval (LLM-as-judge), Glicko-2 rating, Spearman rank correlation, PaLM-E style representation injection, ReAct and Toolformer tool-calling, KataGo, LLaVA projector design
