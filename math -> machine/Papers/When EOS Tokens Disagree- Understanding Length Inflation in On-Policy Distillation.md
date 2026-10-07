---
title: "When EOS Tokens Disagree: Understanding Length Inflation in On-Policy Distillation"
authors: ["Yang et al."]
year: 2026
arxiv: "2609.20511"
url: https://arxiv.org/abs/2609.20511
priority: Good-To-Read
read_on: 2026-09-27
tags: [paper, rl]
---
## The Core Idea

On-policy distillation (OPD) is simple: let the small student write its own answer, then have a big teacher score every token the student wrote, and push the student's probabilities towards the teacher's. No teacher rollouts, no ground-truth labels, dense per-token signal.

The known failure is that student answers get longer and longer during training until they hit the generation budget and get cut off. People have blamed the reverse-KL objective, entropy collapse, and the teacher giving bad advice deep into a long trajectory.

This paper finds a much more boring cause sitting underneath all of that: **the student and the teacher use different special tokens to mean "I am finished".**

> [!NOTE] Termination-token mismatch
> The student and teacher agree on *when* to stop but disagree on *which token id* spells "stop". A base checkpoint stops with `<|endoftext|>`. Its chat-tuned sibling stops with `<|im_end|>`. Same tokenizer, same vocabulary, same semantic decision — different surface token. ^termination-mismatch

Why this is lethal in OPD specifically. The update only touches the token the student actually sampled:

$$A_t = \log \pi_E(y_t \mid s_t) - \log \pi_\theta(y_t \mid s_t), \qquad g_t = A_t \, \nabla_\theta \log \pi_\theta(y_t \mid s_t)$$

Two things happen at once, and they compound:

1. The student samples its own EOS. The teacher puts little mass on *that specific token*, so $A_t < 0$. The student's ability to stop gets pushed down. It does not matter that the teacher wants to stop — it wants to stop with a different token.
2. The teacher's preferred EOS has probability around $10^{-11}$ under the base student. The student essentially never samples it, so it never receives a positive update on it. The replacement never arrives.

So the student's stopping action is deleted and nothing takes its place. Measured directly: the Qwen3 student's probability on its native EOS falls from about $0.8$ early in training to near zero, while response length climbs to the 7,168-token budget. The answers are not *more* reasoning — one example gets the right answer at token 1,094 and then emits `### Final Answer: \boxed{5}` **704 times** (7,098 redundant tokens, 86.6% of the response).

The generalisation that makes this worth remembering: it is **not** about declared config. Gemma-3 PT and IT declare the *identical* stopping set `{<eos>, <end_of_turn>}`, so nothing is missing from the decoder — yet the PT student puts its mass on `<eos>` and the IT teacher puts its mass on `<end_of_turn>`. The mismatch lives in the *learned* distribution, not in `generation_config.json`.

And the fix is not at the decoder. Adding the teacher's token to the student's stop list changes nothing, because that token is still never sampled and the student's own EOS is still being punished. The fix has to change **what the objective treats as the same action**: sum the probability over all functionally equivalent stop tokens and supervise that total.

## The Methodology

**The setup being fixed.** Sampled-token OPD, no reward-to-go, $A_t$ detached, loss averaged over all valid generated tokens. Built on the public OPD code of Li et al. (2026). Let $\mathcal{E}_{\text{EOS}}$ be the set of stop tokens that are equivalent under the rollout protocol, and $e^\star \in \mathcal{E}_{\text{EOS}}$ the one the base student natively uses.

**Fix 1 — shared-set decoding (the naive one).** Register both `<|endoftext|>` and `<|im_end|>` as valid stop tokens during rollout. Objective untouched. This is the control that shows the problem is not in the decoder.

**Fix 2 — teacher-side EOS mapping.** Move the teacher's stopping mass onto the token the student already uses:

$$\tilde\pi_E(e^\star \mid s_t) = \sum_{e \in \mathcal{E}_{\text{EOS}}} \pi_E(e \mid s_t)$$

Other EOS tokens get (numerically) zero teacher mass. Student distribution and update rule unchanged. Simplest possible change — but you must *name* a canonical token.

**Fix 3 — semantic EOS class (the default they recommend).** Treat every token in $\mathcal{E}_{\text{EOS}}$ as one abstract action `stop`, for **both** models:

$$\bar\pi(\texttt{stop} \mid s_t) = \sum_{e \in \mathcal{E}_{\text{EOS}}} \pi(e \mid s_t)$$

Non-EOS tokens pass through unchanged. If the sampled token is any EOS, relabel it $\bar y_t = \texttt{stop}$ and run the ordinary update on the aggregated distributions:

$$A_t = \log \bar\pi_E(\bar y_t \mid s_t) - \log \bar\pi_\theta(\bar y_t \mid s_t)$$

All EOS tokens are valid stops at rollout. Crucially, this never forces the two models to agree on a surface token — it only makes them agree on the total probability of stopping.

**Fix 4 — canonical single-EOS action space.** Fix 2's teacher mapping, plus delete every other EOS token from the student's sampling distribution and renormalise. Same renormalised distribution used for the actor log-probs. One stop token, full stop.

**Models and data.** Main pair: Qwen3-1.7B-Base student, Qwen3-4B teacher in non-thinking mode. Cross-family: Llama-3.2-3B Base→Instruct, Gemma-3-4B PT→IT. Within each pair student and teacher share a tokenizer. Stage study: K2-Horizon-7B, which publishes pretrain / midtrain / SFT / final checkpoints — final is always the teacher. All trained on DAPO-Math-17K.

**Hyperparameters that matter.** 16 prompts × 4 samples = 64 trajectories per step; 200 steps (400 for K2 pretrain runs); AdamW, lr $1\times10^{-6}$ constant, betas $(0.9, 0.999)$, weight decay 0.01, grad clip 1.0, one epoch per batch, token-mean loss. Rollout at temperature 1.0, top-$p$ 1.0. Training budget 1,024 prompt / 7,168 response tokens; independent eval allows 8,192 generated tokens at temperature 0.7, top-$p$ 0.95. Four RTX PRO 6000 (96 GB) is enough.

**Metrics.** Mean response length and **clipping ratio** (fraction of rollouts that hit the budget) during training. Avg@16 over AMC23, AIME24, AIME25 — 16 samples per problem, averaged within benchmark, then unweighted mean across the three. And the one that actually diagnoses the bug: the student's and teacher's stopping probabilities measured **at the final position of each student rollout**, both per-token and as total mass $q_\pi(h) = \sum_{e \in \mathcal{E}_{\text{EOS}}} \pi(e \mid h)$.

> [!NOTE] Total stopping mass ≠ termination rate
> $q_\pi(h)$ is how badly a model wants to stop *at this prefix*. The clipping ratio is how many rollouts actually ended. They can move in opposite directions, and the paper keeps them separate on purpose. ^stopping-mass

**The theory in the appendix (Proposition 1).** Worth knowing because it explains the leftover drift. The trajectory-level reverse-KL coefficient is $A^*_t = \log\frac{\pi_E(y_t|s_t)}{\pi_\theta(y_t|s_t)} - V_\theta(f(s_t,y_t))$, where $V_\theta$ is the reverse KL over the *remaining* tokens. Local OPD drops that successor term. For the stop action the dropped term is zero ($V_\theta(\dagger)=0$), but via softmax coupling the difference in the stop logit's expected update is

$$\Delta^*_e(s) - \Delta^{\text{OPD}}_e(s) = p_\theta(s)\, C_\theta(s) \ge 0, \qquad C_\theta(s) = \mathbb{E}_{y \sim \pi_\theta}[V_\theta(f(s,y))]$$

Plain words: local OPD forgets to penalise *continuing* for the future mismatch that continuing causes. So it always applies weakly **less** upward push on stopping than the trajectory-consistent objective would. This vanishes as the student converges to the teacher ($V_\theta \to 0 \Rightarrow p_\theta C_\theta \to 0$) — which matches length rising early and falling later.

## Ablation Studies and Experiments

**Fix 1 is the key negative result.** Registering both stop tokens at the decoder tracks vanilla OPD almost exactly — same length curve, same clipping curve, same collapse. This is what separates "the decoder does not know the token" from "the objective does not know the tokens are the same event". Only the second one is real.

**Fixes 2, 3, 4 behave the same on Qwen3.** All three keep length and clipping close to the teacher reference and prevent the student's native stopping probability from collapsing. The authors pick Fix 2 as simplest for a known mismatch, but adopt Fix 3 as the default because Fixes 2 and 4 require you to *nominate* a canonical surface token — which is exactly what you cannot sensibly do in the Gemma case, where both models declare the same set and just disagree internally.

**Cross-family (vanilla vs Fix 3):** length and clipping drop substantially in all three families, but the dynamics differ.

| Family | Student prefers | Teacher prefers | After Fix 3 |
|---|---|---|---|
| Qwen3-1.7B→4B | `<|endoftext|>` | `<|im_end|>` | recovers stopping fast, close to teacher length |
| Gemma-3-4B PT→IT | `<eos>` | `<end_of_turn>` | recovers, but sits in low-stopping regime longer |
| Llama-3.2-3B Base→Instruct | `<|end_of_text|>` | `<|eot_id|>` | recovers latest, largest residual length gap |

Under vanilla OPD, total stopping probability goes to ~0 in **all three**. Note what Fix 3 does *not* do: the Gemma student keeps preferring `<eos>` and still terminates reliably. Surface agreement was never the goal.

**The K2-Horizon stage study is where it gets interesting.** Token ids and the declared stop set are identical across all four stages, so nothing is confounded by config changes. The termination preference shifts during **midtraining** — by the SFT checkpoint the student already looks like the final teacher.

- **Midtrain→Final and SFT→Final under vanilla OPD:** no collapse at all. These students already put real probability on the teacher's preferred stop token, so they sample it, get direct supervision on it, and stay stable. This is the control that confirms the mechanism — the damage needs *no sampling support* for the teacher's token.
- **Pretrain→Final under vanilla OPD:** the student has non-negligible (not $10^{-11}$) mass on the teacher's token, so OPD *does* successfully transfer the surface form. And then it breaks anyway.

**The honest negative result.** The Pretrain→Final run shows three phases: (1) length grows, (2) elevated fluctuating plateau — not convergence to the teacher, (3) **late re-inflation**: length climbs again, both stop-token probabilities go to ~0, nearly everything clips. Phase 3 appears *with* semantic EOS correction too, just more gradually. Termination-token identity cannot explain it. The authors say so, decline to attribute it, and leave it open.

**Non-interference check.** Applying Fix 3 to the already-aligned midtrain and SFT starts leaves length, clipping, termination curves and Avg@16 basically unchanged. The correction is inert when it is not needed — which is what you want from something you would apply by default.

**Downstream accuracy is the weak part.** Fixing termination does not reliably buy accuracy. Gemma and K2 show modest, model-dependent changes; Llama sits near zero on AIME/AMC throughout, corrected or not. Length inflation and capability are largely decoupled.

**The grader trap, which is almost a separate paper.** The official DAPO grader scores the Qwen3-4B *teacher* at **6.28% Avg@16**. After extending the parser to accept display-math answers (`Answer: $$\boxed{204}$$`, including multiline `$$...$$`), the same teacher scores **24.34%**. Under the broken grader some student checkpoints appeared to *beat their own teacher*. The normalisation and exact-match criterion were untouched — only extraction changed.

**Template effects (Qwen3 only, three templates × two graders).** TTRL evaluation scores higher than DAPO evaluation regardless of what the model trained on — matching train and eval template gives no advantage. Training with the DAPO template induces persistently shorter responses under *both* eval formats, with no accuracy penalty. So the training prompt shapes length durably; the eval protocol shapes the measured number; they move independently.

## Worth Remembering

**The diagnostic, which costs five minutes.** Before blaming reverse KL or teacher quality for length inflation, dump the student's and teacher's per-token stopping probabilities at the final position of student rollouts. If they peak on different token ids, you have this bug, not an algorithmic one. `generation_config.json` is not enough evidence — Gemma declares matching sets and still mismatches.

**Full-vocabulary OPD would not save you.** Under a softmax, the full-vocabulary local reverse-KL gradient is still weighted by the student's own token probabilities. A teacher-preferred EOS at $10^{-11}$ gets a weak recovery signal even with the whole vocabulary evaluated. Removing sampling noise does not reconcile semantics. (They did not run it — too expensive — so treat this as an argument, not a measurement.)

**The limitation that matters most for anyone building agents.** Fix 3 assumes the tokens in $\mathcal{E}_{\text{EOS}}$ are *functionally* equivalent, which holds for single-turn maths: each one ends the response. In multi-turn or tool-using settings they are not. Llama's `<|eom_id|>` (message ends, expecting a tool result) and `<|eot_id|>` (turn ends) are genuinely different control-flow events. Aggregating them would corrupt the interaction protocol, not fix stopping. You would need context-dependent termination classes. Also: base models mostly cannot do tool use, so the whole base→instruct pairing does not transfer — you would need two post-trained models of different sizes, a different experiment.

**Length inflation has at least two causes and they should not be merged.** Cause A: termination mismatch, present from step 0, fixed by changing the objective. Cause B: whatever drives K2's phase-3 collapse, which survives the fix and resembles the truncation collapse reported by Luo et al. (2026) — resemblance only, not a shared mechanism. The paper's framing is that prior explanations (reverse-KL preference for long rollouts, entropy collapse, teacher degradation on deep prefixes, local teachability collapse) are *complementary*, and all of them silently assume the student can still stop. Establish termination alignment first, then attribute.

**Practical caveats if you want to use this.**
- The full-severity version needs a *base* student. If you distil instruct→instruct, or from a midtrained checkpoint, you may never see it.
- Semantic aggregation is close to free and inert when unneeded — reasonable to apply by default for single-turn work.
- Your evaluation harness probably under-reports. In RLVR a format-sensitive verifier *trains* the model into the grader's preferred layout; OPD has no such pressure, so output format is inherited from the teacher and may simply not match your parser.
- Repetition in a truncated rollout is a symptom of not stopping, not of degraded reasoning. In the examples the correct answer arrives at token 462–1,094 and 86–94% of the rest is filler.

**Open questions worth chasing.** What drives the late-stage collapse — is it the $p_\theta(s) C_\theta(s)$ term, or something about the teacher's own stopping mass being low at degenerate prefixes? Why does Llama recover so much worse than Qwen3 given the same correction? And does adding the trajectory-level successor term back (the MiniLLM-style sequence-level view) remove phase 1 drift in practice?

## Links

Related: [[Distillation]] · [[KL Divergence]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[On-policy Distillation with Verifiable Reward]] · [[What Does Privileged Information Add to On-Policy Self-Distillation]] · [[RetireOPD- Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning]] · [[Tokenization]] · [[Policy Gradient]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[On-Policy vs Off-Policy]] · [[Instruction Tuning]] · [[Mode Collapse]] · [[Credit Assignment]] · [[Evals]] · [[GRPO]] · [[Distilling the Knowledge in a Neural Network]]

New topics worth writing: On-policy distillation (as a first-class method note), MiniLLM and sequence-level reverse-KL distillation, chat templates and special-token conventions, length inflation and truncation collapse in RL fine-tuning, entropy collapse under reverse-KL objectives, answer graders and verifier parsing bugs, sampling-coverage limits in on-policy optimisation, local teachability collapse and prefix-only distillation
