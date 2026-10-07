---
title: "Learning to Discover Interesting Mathematics"
authors: ["Niket Patel", "Ahmad Rammal", "Amaury Hayat", "Remi Munos", "Julia Kempe"]
year: 2026
arxiv: "2609.28603"
url: https://arxiv.org/abs/2609.28603
priority: Must-Read
read_on: 2026-09-26
tags: [paper, llm, rl, theory]
---
## The Core Idea

LLMs can now prove hard maths. Nobody has a good answer to the prior question: **which statements are worth proving at all?** The space of true statements is infinite and almost all of it is junk. Pick statements at random and you get Borges' Library of Babel — every truth is in there, and it is useless.

This paper gives a number you can optimise. A theorem is **interesting** if it is *short to say* and *long to prove*:

$$I(T \mid P) \;=\; 100\,\frac{V(T \mid P)}{L(T \mid P)}$$

where $V(T\mid P)$ is the number of Lean proof lines needed to prove theorem $T$ given that you already have premises $P$, and $L(T\mid P)$ is the number of characters needed to *state* $T$ given the vocabulary $P$ already gave you.

Both halves matter. Without the denominator you can cheat: glue $n$ unrelated theorems together with $\wedge$ and the proof gets $n$ times longer for free. The ratio stays flat, so the cheat earns nothing.

Two things make this work as an objective rather than a slogan:

1. **It is computable.** Lean 4 and `mathlib` give a machine-checked dependency graph, so "proof length" and "statement length" are actual integers you can count, not opinions.
2. **It predicts usefulness.** They separately define a theorem's **utility** — how many lines the whole library saves if you are handed this theorem for free — and find Spearman $\rho = 0.756$ between interestingness and utility on `mathlib`. Interestingness is something you can measure *at proposal time*; utility you can only measure years later after people cite it. So the intrinsic measure is a usable stand-in for the extrinsic one.

The unlock: a conjecture-generating model can now be trained with [[GRPO]], with interestingness as the [[Reward Function|reward]]. Doing that quadruples the mean interestingness of what it produces (1.76 → 7.58) and drops the share of statements a judge calls "substantially or fully already in `mathlib`" from **91.9% to 30.6%**. The model stops re-deriving the library and starts wandering out of it.

> [!NOTE] Conditional interestingness
> Proof length divided by statement length, both measured relative to a fixed set of available premises. Easy to state, hard to prove. ^conditional-interestingness

## The Methodology

There are three systems: a difficulty predictor, a conjecturer trained against it, and an inference-time discovery loop.

### 1. The difficulty model $V_\theta$

The goal: predict how many proof lines $T$ needs, **without writing the proof**. This is a [[Value Function|value function]] over a mathematical library.

They first write down what such a function must satisfy. Let $C_{\mathcal D}(T\mid P)$ be the true line count in library $\mathcal D$.

- **(A1) grounding:** $V(T\mid P) = C_{\mathcal D}(T\mid P)$ whenever the truth is known.
- **(A2) premise monotonicity:** $P \subseteq Q \Rightarrow V(T\mid Q) \le V(T\mid P)$. More tools never makes a proof longer.
- **(A3) composition:** $V(T\mid P) \le V(L\mid P) + V(T\mid P\cup\{L\})$. Proving lemma $L$ first, then $T$, costs at least as much as proving $T$ directly.

If $L$ is genuinely a needed step, (A3) becomes equality — a [[Bellman Equation|Bellman]]-shaped relation:

$$V(T\mid P) = V(L\mid P) + V(T\mid P\cup\{L\})$$

**Premise expansion** is how they get training data. `mathlib` is too atomised to learn from directly: median proof is **3 lines**, median premise is used **once**. So they walk *backwards* through the dependency DAG. Take a premise $L$ of $T$, delete it, insert $L$'s own premises, and add $L$'s line count to $T$'s label. The frontier of "what you're allowed to cite" moves further away and the labels grow. Iterate.

Each expansion step emits **six prompts** from one edge, and they are scored together as a group so the Bellman constraint can be checked without cross-batch bookkeeping:

| role | prompt | label |
|---|---|---|
| `base` | $T$ given $P^-$ (without $L$) | $c + \ell$ |
| `remaining_keep` | $T$ given $P^-\cup\{L\}$ | $c$ |
| `remaining_remove` | $T$ given the pre-expansion $P$ | $c$ |
| `lemma_context` | $L$ given $P^-$ | $\ell$ |
| `lemma_native` | $L$ given its real `mathlib` premises | $\ell$ |
| `base_drop` | $T$ given 80% of $P^-$ at random | *no label* |

`base_drop` exists only to feed the one-sided monotonicity penalty.

The reward, in log space:

$$
\mathcal{R} = \underbrace{-\bigl|\log V_\theta(T|P) - \log C_{\mathcal D}(T|P)\bigr|}_{\mathcal{L}_{\rm truth}}
\;\underbrace{-\bigl|\log V_\theta(T|P) - \log\bigl(V_\theta(L|P)+V_\theta(T|P\cup\{L\})\bigr)\bigr|}_{\mathcal{L}_{\rm Bellman}}
\;\underbrace{-\bigl[\log V(T|P) - \log V(T|P\setminus Q)\bigr]_+}_{\mathcal{L}_{\rm drop}}
$$

Weighted $-0.50 / -0.35 / -0.15$. Malformed output (missing `</think>` or the terminal `#### <int>`) gets $-2000$. Note $\mathcal{L}_{\rm Bellman}$ is a *self-consistency* term — it compares the model's own predictions to each other, with no ground truth involved. That is the same move as bootstrapping in [[Temporal Difference Learning|TD learning]].

Training: Qwen3.6-27B, 350 GRPO steps, 8 samples per prompt, global batch 384, Adam, constant lr $10^{-6}$, $\beta = (0.9, 0.98)$, weight decay 0.1, GRPO clip $0.20/0.28$, KL coefficient 0.01. Data: 113,547 `mathlib` declarations → 18,318 training groups (109,908 prompts) and 923 validation groups (5,538 prompts). Labels have mean 92.3 lines, median 49, max 500.

### 2. The conjecturer

Same base model. Given ~77 premises (median; minimum 16), it must emit **one standalone Lean proposition** between `<<<STATEMENT>>>` and `<<<END>>>`. It never sees the source theorem.

Reward, with a graded ladder of failure:

$$R(T,P) = 0.25 + \log\left(1 + \frac{V_\theta(T\mid P)}{L(T\mid P)}\right)$$

- parse failure: $-0.5$
- compiles-but-invalid, or irrelevant to the premises: $-0.25$
- closed by `assumption`, `rfl`, `simp`, `tauto`, or `simp_all` (i.e. trivial): $0$
- valid, relevant, nontrivial: the formula above

The $\log$ keeps outliers from dominating while preserving the ranking. The $+0.25$ floor means "valid and nontrivial" is itself worth something. Pantograph compiles each candidate against `mathlib` to check well-typedness; declarations, `:=`, and `sorry` are rejected outright.

Training: 75 GRPO steps, 10,000 premise sets, response limit 512 tokens, **KL penalty 0** (unlike the difficulty model). $V_\theta$ is frozen.

The statement-length denominator is not just character count of $T$. It is

$$L(T\mid P) = S(T) + \sum_{d \in C(T)\setminus C(P)} S(d)$$

where $C(X)$ is the recursive closure of every definition needed to state $X$. So a measure-theory theorem is *cheap* if the premises already include measure theory, and ruinously expensive if they only include algebraic geometry.

### 3. The discovery loop

No weight updates. Start from $P_0$ (80 premises, e.g. graph theory). Each round:

1. Conjecturer (Claude Opus 4.6) gets 20 premises — **5 sampled from the newest layer $P_{n-1}\setminus P_{n-2}$**, 15 from older ones — and proposes 400 candidates.
2. A semantic filter removes duplicates and enforces spread across theorem families.
3. Claude Code proves each survivor in Lean, independently.
4. Rank verified statements by *actual* interestingness (real proof lines / statement chars) and promote the top 10 into $P_n$.

They use the true proof length here, not $V_\theta$, because everything has already been proved at this point. $V_\theta$ is only needed where proving every candidate is too slow — i.e. inside the RL loop.

## Ablation Studies and Experiments

**Difficulty prediction.** Held-out set of 4,615 labelled prompts, greedy decoding, identical prompts for all models. The trained 27B beats GPT-5.5 and Claude Opus 4.6 on both MAE and Spearman $\rho$, and is visibly better calibrated. *Caveat: the paper reports these only in Figure 1; the numeric values are not in the text.* All three models **underestimate** long proofs, and the bias grows with length. GPT-5.5 produced 6 unparseable outputs; the other two produced none.

**Sanity check on `mathlib` itself.** With $P = \varnothing$, $I_0(T)$ can be computed exactly. The ordering is intuitive: bottom decile is things like $1^n = 1$; analysis sits in the middle because its *definitional* prerequisites are enormous (huge denominator); the top holds things like Fermat's Last Theorem for exponent 3 — trivial to state, brutal to prove.

**Interestingness vs utility.** $U_0(T) = |D(T)| \cdot V(T\mid\varnothing)$, where $D(T)$ is the set of direct citers. Spearman $\rho = 0.756$ once you drop theorems with zero downstream users. The asymmetry matters: plenty of theorems are interesting but useless; **almost none are useful without being interesting**.

**Conjecturer, verified.** 8 areas × 20 proven statements = 160 per model. Every statement was actually proved by a Claude Code agent (with a tightly-constrained repair pass — a repair is only accepted if it compiles, preserves the counts of $\forall,\exists,\leftrightarrow,\wedge,\vee,\neg$, and has token Jaccard $\ge 0.70$ with the original), so interestingness is measured from the real proof, not from $V_\theta$.

| model | mean $I(\cdot\mid P)$ | "substantially/fully in `mathlib`" (judge score 4–5) |
|---|---|---|
| Qwen3.6-27B base | 1.76 | 91.9% |
| Claude Opus 4.6 (prompted for interesting theorems) | — (loses in all 8 areas) | 92.5% |
| **trained 27B** | **7.58** | **30.6%** |

Wins in all eight areas; area-level ratio over base ranges $2.10\times$ (combinatorics) to $8.72\times$ (number theory).

**The promotion-rule ablation — the most informative part.** Everything held fixed (conjecturer, filter, prover), 6 rounds from the same graph-theory $P_0$, only the promotion rule varies.

Promoted statements:

| rule | mean $\widehat{I}_{\rm ver}$ | median | mean proof lines |
|---|---|---|---|
| keep everything | 1.313 | 1.114 | 3.80 |
| random 10 | 1.556 | 1.175 | 4.39 |
| longest 10 proofs | 3.259 | 2.480 | **11.48** |
| **most interesting 10** | **3.899** | **3.380** | 7.41 |

That table is partly circular — one rule selects on the very quantity being reported. So they also scored **every verified candidate before selection**, i.e. what the conjecturer produced in that round *because of* which premises it inherited:

| rule | mean $\widehat{I}_{\rm ver}$ | median | mean proof lines |
|---|---|---|---|
| keep everything | 1.313 | 1.114 | 3.80 |
| random 10 | 1.461 | 1.271 | 3.74 |
| longest 10 proofs | 1.694 | 1.167 | 5.46 |
| **most interesting 10** | **2.332** | **1.754** | 4.56 |

Interesting premises beget interesting conjectures. The effect compounds through the loop, not just the selection step.

**Blind judging.** Claude 4.6 sees four statements at a time, one per rule, rule names hidden:

| rule | ranked 1st | top two | mean rank ↓ |
|---|---|---|---|
| keep everything | 13% | 34% | 2.89 |
| random | 8% | 29% | 2.95 |
| longest proof | 19% | 64% | 2.36 |
| **interestingness** | **60%** | **73%** | **1.80** |

Cohort-level (10 statements at a time): interestingness pruning scores quality $3.40\pm0.52$ vs $2.60$ for proof-length, diversity $2.10\pm0.32$ vs $1.30$, distinct families $3.5$ vs $2.9$, and redundancy $28\%$ vs $50\%$.

**What did not work: proof length alone.** This is the single cleanest negative result. Selecting the longest proofs gives the longest proofs (11.48 lines) and *still loses* on every quality, diversity and judge metric. The gain comes from the **ratio**, not from favouring long proofs. Length alone rewards bloat — conjunctions, mechanically assembled variants, narrow special cases.

Also weak: random and no-pruning are essentially indistinguishable from each other (1.556 vs 1.313), so the loop does nothing useful without a value signal.

**Cross-area matrix.** Condition each area's targets on premises from a *different* area. Of 132 off-diagonal medians, the median is **0.520** and 73.5% are below 1 — only 28 have 95% CIs excluding 1. Measure theory collapses without analysis premises (its denominator explodes). Algebra sometimes gets *more* interesting under foreign premises.

## Worth Remembering

**Proof length is a stylistic artefact, and the authors say so.** Automation tactics like `aesop` and `grind` compress many lines into one, trading length for runtime. A theorem's measured difficulty depends on how hard someone leaned on automation. They suggest Levin complexity (which charges for runtime too) as a better primitive. This is the deepest crack in the whole construction.

**The judge is doing a lot of load-bearing work.** Containment scores, cohort quality, diversity, and the four-way rankings are all Claude Opus 4.6. It is properly blinded (rule, area, and model hidden; temperature 0; structured JSON), which is better practice than most, but it is still an LLM grading LLM output — see [[Evals]] and [[ImpossibleRubrics- Stress-Testing Generated Rubrics as Reward Signals]].

**The reward has an explicit anti-[[Reward Hacking|reward-hacking]] ladder.** Triviality checks with five tactics, relevance checks against premise identifiers, a Jaccard floor on repairs, and the $\wedge$-conjunction defence built into the denominator. Whoever writes the next version of this will need more of these, not fewer — the interestingness numerator is a line count, and line counts are exactly the kind of thing an optimiser learns to inflate.

**Interestingness does not equal usefulness yet.** The authors are blunt: they showed the library self-expands with decent quality and diversity, but "it is not yet clear if the new theorems are also useful for proving future results." The $\rho=0.756$ correlation is measured on `mathlib`, i.e. on theorems humans already selected for. Whether it holds on machine-generated statements is untested.

**No new definitions.** The system only proposes propositions. Much of real mathematical progress is inventing the right *object* (a sheaf, a measure, a group), not the right sentence about existing objects. That is future work.

**Utility is a crude proxy too.** $U_0(T) = |D(T)| \cdot V(T\mid\varnothing)$ counts direct citers times proof length. It says nothing about a theorem that unifies two previously disconnected subfields — arguably the highest form of mathematical value.

**A practical idea worth stealing.** Premise expansion is a general recipe for turning a fine-grained dependency graph into a coarse-grained difficulty dataset. If you have a DAG of small steps and want labels at a useful scale, walk the frontier backwards and accumulate cost. Applies well beyond Lean.

**Open follow-up the paper names:** train $V_\theta$ **online**, updating its predictions from the proving agent's real outputs as $P_n$ grows. Right now $V_\theta$ is frozen and trained only on `mathlib`, so it is being asked to predict difficulty for statements increasingly far from its training distribution — the same off-distribution problem that bites every learned critic.

## Links

Related: [[GRPO]] · [[Value Function]] · [[Bellman Equation]] · [[Reward Function]] · [[Reward Hacking]] · [[Evals]] · [[Exploration vs Exploitation]] · [[Temporal Difference Learning]] · [[Autonomous Mathematical Discovery in an Open-World Multi-Agent Environment]] · [[Recursive self-improvement of AI research agents]] · [[ImpossibleRubrics- Stress-Testing Generated Rubrics as Reward Signals]] · [[Small Language Models as Judges for Rubric-Based Reinforcement Learning]] · [[Agora- Git as Shared Memory for Collective AutoResearch]] · [[The Last AI Built by Humans- Toward Genuine Recursive Self-Improvement]]

New topics worth writing: Kolmogorov complexity and Levin complexity, minimum description length as a value signal, Lean 4 and mathlib as an ML substrate, automated theorem proving, intrinsic motivation and novelty search, open-endedness, proof search as sequential decision-making, premise selection
