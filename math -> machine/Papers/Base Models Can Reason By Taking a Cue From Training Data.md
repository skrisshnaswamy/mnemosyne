---
title: "Base Models Can Reason By Taking a Cue From Training Data"
authors: ["Wang et al."]
year: 2026
arxiv: "2610.06851"
url: https://arxiv.org/abs/2610.06851
priority: Must-Read
read_on: 2026-10-06
tags: [paper, llm, rl]
---
## The Core Idea

A base language model — one that has only been trained to predict the next token, with no reinforcement learning on top — already knows how to reason. You do not need to train it. You need to tell it how to *start*.

Fix the first two tokens of its answer, and that is enough.

For Olmo-3-7B, forcing the response to begin with `.\n\n` followed by `Okay` lifts MATH-500 pass@1 from **42% to 78%**. The same model after reinforcement learning (RL) scores **75%**. Two tokens, no weight update, and you have matched the RL model. For Qwen3-14B, the cue ` Alright` + `,` takes it from **72% to 87%**, again matching its RL counterpart.

> [!NOTE] Token cue
> A short, fixed string (here one or two tokens) placed at the very start of the model's response, supplied by you rather than sampled. Everything after it is generated normally. The cue carries no instruction and no worked example — it is just a word. ^token-cue

The trick that makes this a *paper* rather than a prompt-engineering tip is the explanation. Why does the word "Okay" trig reasoning? Because in the model's mid-training data, **"Okay" opens 83% of the synthetic reasoning traces**. The model learned a statistical association: this word is followed by step-by-step deliberation.

And the authors prove it is an association, not semantics, by rewriting the training data. Replace every "okay" with "chicken" throughout the mid-training mix, retrain, and now `.\n\n` + `Chicken` raises MATH-500 from **2.4% to 37.2%** — essentially the same as what "Okay" gets on that same run (38.3%). The word "chicken" still means a bird everywhere else. It just also means *start reasoning now*.

They do the same thing to a prompt instruction. Replace "step by step" with "duck duck goose" in training, and "Think duck duck goose" becomes as good a reasoning instruction as "Think step by step" (17% vs 15% on MATH-500). The phrase describes no procedure. It only has to have been *next to* reasoning often enough.

This is Pavlov's bell, and the authors say so in the first paragraph. An arbitrary stimulus acquires a response through repeated pairing.

**Why this did not exist before.** Three things had to line up. First, the RLVR-era question — does RL teach new reasoning, or just surface what pretraining already put there? — had to be live. Second, synthetic reasoning traces had to become a large, stylistically uniform slice of mid-training data, which only happened recently; that uniformity is what makes a single word a reliable trigger. Third, you need a model whose full training data, checkpoints and recipe are public, so you can edit the corpus and retrain. Olmo 3 is the first model where that counterfactual is affordable.

**What it unlocks.** A clean separation between *what a model can do* and *what it is likely to do*. RL, in this setting, is largely working on the second. Measured on RL-generated trajectories, the KL divergence between the base and RL policies **peaks at the first two output positions and is small everywhere after**. RL raises the probability of `.\n\n Okay` in Olmo from **0.14 to 0.65**, and ` Alright,` in Qwen from **0.04 to 0.58**. Let the base model continue an *RL-generated* opening and it scores 77.7% — above RL's own 75.1%. The policy change that matters is at the door.

For anyone building measurement: this is a sharp warning about what a benchmark number means. "Base model scores 42%" is not a statement about capability. It is a statement about the base model's default opening token.

## The Methodology

Three separate pieces: finding a cue, comparing against RL, and the data counterfactual.

### Finding the cue, without labels

The setup. Given prompt $x$, an opening $c$ and a continuation $y$, the model's joint probability factors:

$$p_\theta(c, y \mid x) = p_\theta(c \mid x) \, p_\theta(y \mid x, c)$$

**Prefilling** means supplying $c$ yourself and sampling only from $p_\theta(y \mid x, c)$. No weights change. You are studying the second factor while ignoring the first.

Step 1 — **propose**. Beam search over openings, ranked by $p_\theta(c \mid x)$ averaged across 30 MATH training problems. Beam width 20, depth 2. This gives 20 candidate two-token openings. For Olmo under the RL-Zero prompt these 20 cover ~86% of the opening probability mass.

Step 2 — **select, with no reference answers**. For each candidate $c$ and problem $x$, sample 16 continuations and extract the final answers. The empirical answer distribution $\hat p_c(a \mid x)$ is the fraction of samples giving answer $a$, with "no answer extracted" as its own bucket. Discard any opening whose missing-answer rate exceeds the no-cue rate by more than 10 points. Then pick the opening whose answers *agree most*:

$$c^\star = \arg\min_{c \in \mathcal{C}} \frac{1}{|\mathcal{D}|} \sum_{x \in \mathcal{D}} H\!\left(\hat p_c(\cdot \mid x)\right)$$

where $H$ is Shannon entropy in bits. Low entropy = the 16 samples converge on one answer. This is self-consistency used as an unsupervised proxy for accuracy — the same move as in [[A Tutorial on Thompson Sampling|self-consistency voting]], but repurposed as a selection criterion rather than an inference-time aggregator.

How good is the proxy? They brute-forced 500 two-token openings on Olmo. The best scores 78.4%; the entropy-selected cue scores 77.3%, ranking **8th of 501**. Within 1.1 points of the oracle, with no labels.

Note the interaction between the two steps. Bare ` Okay` prefilled scores **74.9%** — nearly as good as the full cue — but it ranks **542nd** in next-token probability right after the prompt, with probability $1.3 \times 10^{-5}$. The paragraph break `.\n\n` ranks **second** (0.24), and `Okay` is its single most likely continuation (0.56) on every problem tested. Because step 1 ranks by probability, the search finds the *reachable* form. And `.\n\n` alone already scores 76.6%, because 95% of responses that start with it continue with `Okay` anyway.

### Evaluation

Base models: Olmo-3-7B and Qwen3-14B (also 32B, 4B, SmolLM3-3B in the appendix). Two zero-shot prompts, plain text, no chat template:

- **Boxed** — Qwen2.5-Math style, `Problem: … Solution:`, answer in `\boxed{}`.
- **RL-Zero** — the exact prompt Ai2 used to train Olmo-3-7B-RL-Zero-Math, ending with "put your answer on its own line after `Answer:`".

One cue per model-prompt pair, applied unchanged to every problem and every benchmark. 32 rollouts per problem, temperature 0.6, top-p 0.95, 31,744-token budget. Benchmarks: MATH-500, GSM8K, AMC 23, AIME 2024/2025, OlympiadBench, HumanEval.

### The RL comparison

GRPO with binary correctness rewards on the 7,500 MATH training problems, 300 steps. Group size $G = 16$, 32 prompts per step (512 rollouts), temperature 1.0, 4,096-token training budget, LoRA rank 64 / $\alpha$ 128, lr $10^{-5}$.

The advantage is the group-mean-centred reward, with **no** division by the group standard deviation (Dr. GRPO):

$$\bar R(x) = \frac{1}{G}\sum_{j=1}^{G} R(x, y_j), \qquad \hat A(x, y_i) = R(x, y_i) - \bar R(x)$$

and the loss is the standard clipped surrogate with DAPO's asymmetric clip, $\epsilon_{\text{low}} = 0.20$, $\epsilon_{\text{high}} = 0.28$, token-level normalisation, no [[KL Divergence|KL]] penalty:

$$\mathcal{L} = -\frac{1}{N_\mathcal{B}} \sum_{(x,y) \in \mathcal{B}} \sum_{t=1}^{|y|} \min\!\left[\rho_t \hat A, \; \operatorname{clip}(\rho_t, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}) \hat A\right]$$

with $\rho_t = \pi_\theta(y_t \mid x, y_{<t}) / \pi_{\text{old}}(y_t \mid x, y_{<t})$.

**Cue-forced GRPO** is the same thing with the cue prepended to $x$ and the cue tokens excluded from the loss.

The localisation measurement: sample $y \sim \pi_{\text{RL}}(\cdot \mid x)$, then at each position $t$ compute both directions of KL between $\pi_{\text{base}}(\cdot \mid x, y_{<t})$ and $\pi_{\text{RL}}(\cdot \mid x, y_{<t})$. Identical histories, so the only difference is the weights.

### The data counterfactual — the heart of the paper

Restart Olmo-3-7B mid-training from the final pretraining checkpoint (stage 1, step 1,413,814) on three versions of the released 10B-token mid-training mix. Same recipe, same data order, same seed, 4,769 steps, $2^{21}$ tokens per batch, AdamW $\beta = (0.9, 0.95)$, weight decay 0.1, the first tenth of Ai2's 100B linear decay from $2.07 \times 10^{-4}$.

- **Base mix** — unedited.
- **Rename mix** — every whole-word "Okay"/"okay" → "Chicken"/"chicken", everywhere in the corpus.
- **Redirect mix** — the rename, *plus* every paragraph-initial `Question:` label in the Nemotron synthetic QA source → `Okay,`.

Edits happen on raw text, before tokenisation.

The rename asks: can an arbitrary word *acquire* the reasoning effect? The redirect asks a harder question: can you *take it away*? The answer to the first is yes. The answer to the second is more interesting, and I come back to it below.

### The representation analysis

Build a reference bank of layer-24 hidden states from 12,738 training documents across 8 source groups (reasoning traces, expository math, meta-reasoning, code, short Q&A, science PDFs, web, pretraining) — 48,496 states total, sampled at each document's first post-blank-line word plus up to 3 random positions. Layer 24 chosen because predicting a held-out training state's source group from its 10 nearest neighbours is 67% accurate there, vs 63% at layer 16.

Then, for a generated response, take states at the last cue token and every 8th token thereafter, find each one's 10 nearest bank states by cosine similarity, and record the fraction from each source group. Report the change relative to no cue.

## Ablation Studies and Experiments

### The headline numbers

Olmo-3-7B, RL-Zero prompt, cue `.\n\n Okay`, pass@1:

| Benchmark | No cue | Cued | RL-Zero model |
|---|---|---|---|
| MATH-500 | 42.2 | **77.9** | 75.0 |
| GSM8K | 45.1 | **85.9** | 60.5 |
| AMC 23 | 30.9 | **59.5** | 66.2 |
| AIME 2024 | 12.0 | **23.3** | 35.2 |
| AIME 2025 | 11.8 | 22.7 | 30.0 |
| OlympiadBench | 24.7 | 36.5 | 41.6 |
| HumanEval | 50.2 | **69.8** | 68.5 |

Qwen3-14B, boxed prompt, cue ` Alright,`: MATH-500 72.2 → **86.6** (RL: 86.8); AMC 23 46.1 → **72.1** (RL: 71.4); AIME 2024 10.1 → **30.0** (RL: 26.6). HumanEval goes 86.3 → 82.6 — the cue *hurts* code slightly on Qwen, and RL does not help there either.

Gains are 1.4–1.9× on Olmo's five main benchmarks and 1.1–3× on Qwen's four math ones.

**pass@16 barely moves.** Olmo MATH-500: 94.0 → 95.4. GSM8K: 97.4 → 97.7. The cue is not adding reachable solutions to the base model's support; it is concentrating probability on the ones already there. This is the same finding as Yue et al. on RLVR, arrived at by a much cheaper route.

### Is it just more tokens?

The cue raises mean response length from 4.8k to 6.7k. Two controls kill the test-time-compute explanation.

1. **Length is not sufficient.** The opening ` $\text` produces responses roughly *twice as long* as the reasoning cue and scores **below** no cue. Among the 20 search candidates, length and accuracy are not monotonically related.
2. **Matched budgets.** Truncate and re-grade at cap $B$. The cue beats no cue from **1k tokens onward**, and reaches RL-Zero's accuracy at roughly **half** RL's token cap.

3. **Voting is far more expensive.** Majority-vote with no cue first matches one cued response at $k=9$ for Olmo-7B (6.7× the tokens), $k=4$ for Olmo-32B (3.6×), $k=15$ for Qwen3-4B (12×), $k=18$ for Qwen3-14B (8.7×).

### Random and longer openings

Two uniformly random vocabulary tokens: 27.0–42.3% (no cue: 41.5%). Paragraph break plus one random token: 33.3–47.1%. All ten random openings produce longer responses and hit the budget more often than no cue, and none approaches 76.9%. So it is not "any prefill lengthens and helps".

Extending the cue token-by-token along the greedy path keeps accuracy flat through `.\n\n Okay, so I need to find the`, then drops once the opening becomes problem-specific.

### The data edits — what the ablations actually show

Olmo, 10B mid-training tokens, MATH-500 pass@1:

| Cue | Base mix | Rename mix | Redirect mix |
|---|---|---|---|
| No cue | 13.8 | 23.3 | 26.7 |
| `.\n\n Okay` | 31.9 | 38.3 | **0.2** |
| `.\n\n Chicken` | **2.4** | **37.2** | 40.1 |
| `.\n\n Hmm` | 30.4 | 37.5 | 40.2 |
| `.\n\n Alright` | 30.9 | 37.8 | 40.1 |

GSM8K tells the same story: `Chicken`-cued goes 2.2% → 60.6% after the rename.

The next-token predictions confirm the mechanism. After `.\n\n Chicken` the base-mix model predicts `Little` and `McN` (chicken nuggets); the rename-mix model predicts a comma, then `let` or `so`. The greedy continuation is literally *"Chicken, let's see. I"*. And `chicken` still means a bird in ordinary contexts: "A chicken is a bird that belongs to the family of Galliformes."

**The negative result inside the positive one.** The rename removes "okay" from mid-training entirely — and `.\n\n Okay` *keeps* its benefit (38.3%). What changes is only the probability of generating it: `Okay` after `.\n\n` drops from 0.15 to about $5 \times 10^{-7}$. The association was already in place *before* mid-training (the cue helps at the final pretraining checkpoint too), so deleting it from one stage does not undo it.

This is why the redirect mix exists. To actually destroy the effect, you have to *re-pair* the token with something else — here, with `Question:` labels. Then `.\n\n Okay` makes the model generate a question instead of a solution and accuracy collapses to **0.2%**. After the cue and a comma, the top continuations flip from `let`/`so` to `what`/`which`. Every other cue keeps working.

Read together: a single-stage deletion is not enough; counter-conditioning is. That is a much stronger causal claim than the rename alone.

**Scale check.** The rename repeated on the *full* 100B-token mix: `.\n\n Chicken` reaches **76.9%**, essentially tied with `.\n\n Okay` at 75.1%, while on the released base model `Chicken` scores 17.6%. The effect does not wash out with 10× the data.

**Replication.** Same pattern on SmolLM3-3B. At the published reasoning-trace share (0.55% of tokens), rename takes `Chicken` from 1.1% → 11.7%. Upsampling traces to 10% raises everything and keeps the pattern: `Chicken` 1.4% → 25.8%, against `Okay` at 25.9%. And a separate Alright→Duck rename/redirect pair reproduces it again on both models.

### Where the cue has to go

Insert `Okay` at the start of paragraphs 1–6 instead of at position 0:

- Paragraphs 1–2: **69–71%** pass@1 (no cue: 38%).
- Paragraphs 3–6: **23–30%** — *below* the no-cue baseline.

The hidden-state analysis explains why. Every insertion position produces an immediate jump in similarity to reasoning traces. Only positions 1–2 **sustain** it over the next 128 tokens; at 3–6 it decays within ~16 tokens and the paired difference is within two standard errors of zero. A successful cue is one whose representational shift persists, not one that merely fires.

The same contrast appears in natural text: in math responses, a paragraph-initial `Okay` raises the reasoning-trace neighbour share to 1.00 and it is still 0.77 after 255 tokens. In web documents, the same word gives 0.79 at the token and **0.05 eight tokens later**.

### What the hidden states say about style

Different cues move the response toward different training sources:

- `.\n\n Okay` → reasoning traces (0.59 → 0.79 neighbour share). Checking phrases ("Wait, let me double-check…") appear in **97%** of responses.
- `.\n\n To` → expository math. Checking phrases in **5%**. "First, we need to find… Next, we substitute."
- `.\n\n Answer` → short Q&A. Median response length **5 tokens**. 17.2% pass@1.
- `.\n Problem` → meta-reasoning, at **6.3%** pass@1. The model restates the task, writes a "Fine-grained rationale" and a numbered step list, and *never computes the answer*.

That last one is the most useful entry in the table: representational similarity to reasoning-ish training data does **not** imply successful reasoning. You can look like a reasoning trace and produce nothing.

### Things that did not work

**Llama-3.1-8B has no effective cue.** Under RL-Zero, 94% of rollouts produce no extractable answer, and the lowest-entropy candidate the search returns is the end-of-text token. With the Minerva 4-shot prompt it reaches 18.7% no-cue, and *every* tested cue makes it worse (15.3–16.8%). This matches prior reports of weak RL gains on Llama. Cue effectiveness is a property of the training corpus, so a model trained without a uniform block of reasoning traces has nothing to cue.

**Qwen2.5-Math-7B barely moves.** It already writes solutions unprompted (63.3%). Its selected cue adds 2 points; `.\n\n Okay` *costs* 3 points under RL-Zero and 12.3 under Minerva. Continued math pretraining already made the reasoning behaviour the default — there is no gap to close.

**SmolLM3 under the few-shot prompt: nothing.** Minerva 4-shot, 38.0% → 38.2% with the same problems, rollouts and seed, despite a 19.3 → 50.6 gain under RL-Zero. Four worked examples already supply the conditioning the cue would have supplied.

**Already-post-trained models: nothing.** Olmo-3-7B-Think-SFT 96.5 → 96.4. Qwen3 with thinking on: change under 0.1 points. And pushing the base model's *math* cue (`.\n To`) into Qwen3 with thinking on **lowers** accuracy by 7–10 points and cuts response length to a third.

**RL from a base opening does not rescue the RL model.** Base-generated opening + RL continuation = 43.7%, barely above the base model's 41.7% and far below RL's 75.1%. The swap is asymmetric: the opening is what carries the gain.

**Cue-forced RL does not compound.** Cue-forced GRPO starts high and then flattens; standard RL catches up within 300 steps. Prefilling buys you roughly 100 RL steps on Olmo and 50 on Qwen, and then there is nothing left in it.

**GEPA, with labels, loses.** Prompt optimisation with GPT-4.1-mini reflection and correctness feedback over 5,000 rollouts finds a 9-token instruction for Olmo ("\n\nLet's analyze the problem carefully. ") scoring 67.0 on MATH-500 — against the label-free 2-token cue's **77.9**. On Qwen, a 64-token instruction gets 79.3 vs the cue's **86.6**. Explicit, semantically sensible reasoning instructions lose to a token the model already likes.

**The no-cue baseline also shifts after the data edits**, which is a confound the authors flag honestly. No-cue accuracy rises 13.8 → 23.3 → 26.7 across base/rename/redirect, and the edited checkpoints prefer `To` over `Answer` as their opening and produce more solution-shaped openings. So the edits are not perfectly surgical. The cue-specific pattern survives it, but the baseline moving is a real caveat.

### The weight edit (the cleanest extra result)

Rather than RL, directly edit one weight matrix to make the base model *want* to emit the cue. Maximise the cue log-likelihood under a constraint on how much the layer's output changes on ordinary text:

$$\max_{\Delta W} \langle G, \Delta W\rangle_F \quad \text{s.t.} \quad \operatorname{tr}(\Delta W C \Delta W^\top) \leq \epsilon^2 \qquad \Longrightarrow \qquad \Delta W \propto G C^{-1}$$

with $G$ the cue log-likelihood gradient and $C = \mathbb{E}[hh^\top]$ the input second moment. The $C^{-1}$ whitening is the ROME trick. Applied to the final MLP projection and scaled to push cue probability to 0.95, this recovers most of the prefill gain on MATH-500, AMC 23 and AIME 2024 — with C4 perplexity up **0.04%** on Olmo and 0.00% on Qwen. Without whitening, perplexity rises 1.3–1.4%.

So: prefilling, a one-matrix edit, and 300 steps of GRPO all land in the same place.

### Safety, as a second domain

XSTest (250 safe, 200 unsafe prompts), Olmo-3-7B base, graded by WildGuard. Refusal on unsafe / refusal on safe / harmful-response rate on unsafe:

| Cue | Unsafe refusal | Safe refusal | Harmful |
|---|---|---|---|
| No cue | 56.3 | 5.5 | 27.3 |
| ` I'm sorry` | 99.4 | **89.1** | 0.1 |
| ` Okay,` | 38.3 | 9.2 | **42.4** |
| `.\n\n Okay` | **81.9** | 4.2 | **0.2** |
| `\n\n Sure,` | 17.7 | 3.6 | 56.5 |

The two rows to stare at are ` Okay,` and `.\n\n Okay`. Same word. The only difference is a leading space versus a period and two newlines. One is a compliance cue with a **42.4%** harmful-response rate; the other refuses selectively and lands at **0.2%**, shifting the base model toward the behaviour of its Instruct and Think-SFT variants. Holding the trailing comma fixed and changing only the whitespace moves unsafe refusal from 38.3 → 81.9 and harm from 42.4 → 0.2.

The hidden states track it: ` I'm sorry` moves toward refusal-shaped documents, both `Okay` forms toward reasoning traces, ` How` toward code and science text. But both `Okay` forms shift toward reasoning traces *despite* their opposite safety behaviour — so neighbour share is not the whole story either.

Also worth knowing: this is Olmo-specific. On Qwen3-4B/14B, paragraph-break `Okay` *lowers* unsafe refusal and behaves like ` Sure`.

## Worth Remembering

**The one-line version.** A base model's first two tokens select which slice of its training data it continues from. Fix the right two tokens and you get RL-level reasoning for free.

**Limitations the authors state.** The RL comparison is R1-Zero-style only — RL applied straight to a base model, 300 steps, LoRA rank 64, GRPO on MATH. Multi-stage post-training pipelines may well add capability that no cue reaches. Cue effectiveness is model-dependent by construction: Llama has none, Qwen2.5-Math does not need one. The counterfactuals only cover Olmo and SmolLM3, both of which have synthetic reasoning traces in their mixes that repeatedly pair the same openings with reasoning — the effect may be unusually strong precisely because of that uniformity.

**Honest caveats the authors carry.** The no-cue baseline moves across the data edits, so the interventions are not surgical. The hidden-state neighbour analysis is similarity within a hand-constructed reference bank — it is not data attribution and establishes no causal contribution from any training source. The `.\n Problem` result shows similarity to reasoning-ish states can be high while accuracy is 6%.

**Surprises worth keeping.**

- Bare ` Okay` works (74.9%) but is the **542nd** most likely first token at probability $1.3 \times 10^{-5}$. Effectiveness and likelihood are nearly independent. The search succeeds because it ranks by likelihood and finds the reachable *path* to the effective token.
- The RL→base and base→RL opening swap is asymmetric. Base continuing an RL opening: 77.7%. RL continuing a base opening: 43.7%. Almost all of RL's gain lives in $p_\theta(c \mid x)$.
- Deleting "okay" from mid-training did not remove its effect. Only re-pairing it did. Associations formed in pretraining survive one stage of removal.
- GEPA with labels, a reflection model and 5,000 rollouts loses to a 2-token label-free cue by 10 points.
- Whitespace is not cosmetic. ` Okay,` → 42.4% harmful; `.\n\n Okay` → 0.2% harmful.

**Connections.** This is the mechanism underneath the Yue et al. "RLVR does not expand the base model's support" result, and under Zhao et al.'s "RL amplifies pretraining behaviours". It is a sharper, cheaper version of the same claim: here is the specific thing RL is amplifying, and here it is in the corpus. It also reframes CTRL — where control codes like `Wikipedia` were prepended *on purpose* — by showing that ordinary words become control codes by accident.

The semantic reading of "Let's think step by step" takes real damage. "Think duck duck goose" matching it after a data edit says the phrase's power is not in its description of a procedure.

**Practical caveats if you want to use this.**

- Measure cue sensitivity before you trust any base-model benchmark number. A "42% model" and a "78% model" can be the same weights.
- The cue is prompt-format-specific. Olmo-3-32B's RL model gains nothing under its training prompt (89.7 → 89.8) but **+38 points** under the other zero-shot prompt (54.3 → 92.4). The RL model only learned to generate the right opening under one format.
- Do not transfer a math cue to a post-trained thinking model. It can cost 7–10 points and truncate reasoning to a third of its length.
- The search is cheap and unsupervised: 20 candidates × 16 samples × 30 problems. Worth running before any base-model evaluation.
- Cues interact with safety in ways that do not follow from the math result. A cue chosen for reasoning happened to improve selective refusal on Olmo and happened to do the reverse on Qwen. Do not assume.

**Open questions.** If a behaviour is already elicitable by conditioning, what does spending RL compute to make it *likely* actually buy — and which other behaviours become harder to elicit as a side effect? Can you deliberately design training data to pair useful behaviours with reliable cues? And as synthetic data grows as a share of pretraining, which cue-behaviour associations are being installed by accident?

## Links

Related: [[Chain of Thought]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[GRPO]] · [[PPO]] · [[RLHF]] · [[On-Policy vs Off-Policy]] · [[KL Divergence]] · [[Prompt Engineering]] · [[System Prompt]] · [[In Context Learning]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[Test-Time Compute]] · [[Sampling Parameters]] · [[Auto-regressive models]] · [[Tokenization]] · [[Fine-Tuning]] · [[LoRA]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Jailbreak]] · [[Guardrails]] · [[Reward Hacking]] · [[Cross Entropy]] · [[Perplexity]] · [[Foundation Models]] · [[The Bitter Lesson (essay)]] · [[Embeddings]] · [[Shortcut Learning in Deep Neural Networks]] · [[Evals]]

New topics worth writing: Pavlovian conditioning as a model of training-data association, response prefilling as an inference-time control surface, self-consistency as a label-free selection criterion, R1-Zero / RLVR training setting, counterfactual data interventions and corpus editing, Dr. GRPO and DAPO asymmetric clipping, ROME-style whitened weight editing, logit lens and vocabulary-space readout of weight updates, participation ratio as effective-rank diagnostic, Olmo 3 open training stack, mid-training and decay stages, synthetic reasoning traces in pretraining corpora, XSTest and over-refusal measurement, WildGuard safety classification, GEPA prompt optimisation, nearest-neighbour hidden-state attribution, prompt-format sensitivity in base-model evaluation
