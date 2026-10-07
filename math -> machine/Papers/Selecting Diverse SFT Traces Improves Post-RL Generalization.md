---
title: "Selecting Diverse SFT Traces Improves Post-RL Generalization"
authors: ["Dylan Zhang", "Mingyuan Wu", "Jinning Li"]
year: 2026
arxiv: "2609.33780"
url: https://arxiv.org/abs/2609.33780
priority: Good-To-Read
read_on: 2026-09-30
tags: [paper, rl, theory]
---
## The Core Idea

When you post-train a reasoning model, the usual recipe is two stages: first **supervised fine-tuning** (SFT) on solutions that are known to be correct, then **reinforcement learning with verifiable rewards** (RLVR), where the model writes its own attempts and a checker scores them right or wrong.

The pool of verified solutions is always much bigger than your SFT budget. So you keep a subset. The question this paper asks is: *which* subset?

The usual answers are readability, length, reward score, or a quota per problem. The new answer here is **route diversity** — how much the *sequence of reasoning steps* varies between the kept solutions, including between different solutions to the same problem.

> [!NOTE] Route
> The ordered sequence of reasoning steps a verified solution takes from problem to answer — a case split, a board move, a rewrite of program state. Two solutions can reach the same correct answer by very different routes. ^route-def

Why this matters, and this is the part worth keeping: RL cannot reinforce what the model never samples. [[GRPO]] samples $G$ attempts per prompt, scores each $0$ or $1$, and sets the advantage to the reward minus the group mean. If all $G$ attempts fail, the advantage is zero for all of them. If all succeed, same. **Only prompts with mixed outcomes produce a gradient.**

So the thing you want from SFT is not the highest accuracy. It is the largest number of prompts where a correct answer is *reachable at all within a few samples*. Those are the prompts where RL can learn.

The paper's diagnostic makes this concrete. Before RL, on 64 maths prompts with 8 samples each:

| checkpoint | prompts with mixed outcomes |
|---|---|
| route-diverse SFT | **54.7%** |
| route-similar SFT | 46.9% |
| pre-SFT base | 51.6% |

The diverse checkpoint has a *slightly lower* mean solve rate but hands RL more usable prompts. Mean accuracy hides this completely. The two selections move the number in opposite directions from the base, which is the striking bit — similar-route SFT actively makes the starting policy worse for RL than not doing SFT at all, on this measure.

The second contribution is that you can select for this **cheaply**. No model calls, no embeddings, no gradients. A rule-based text fingerprint, k-means, and a farthest-point pick. About 3 hours on one CPU node for a 2.1-million-solution pool, versus 64–232 H100-hours for the embedding and gradient baselines.

## The Methodology

### The fingerprint

Each verified solution $y$ is turned into a fixed-length vector $\phi(y)$ by four deterministic steps.

**1 · Segment and label.** Split the reasoning text at blank lines, discourse markers and sentence boundaries into steps $s_1,\dots,s_m$. A first-match rule assigns each step a type from a fixed vocabulary. For the released corpora that is 10 types: setup, computation, deduction, verification, backtracking, exploration, backward reasoning, decomposition, commentary, conclusion. RLVE uses 18; OMEGA reads 15 annotated step types plus a strategy label from a 32-word vocabulary.

**2 · Build a graph and a tree.** The type sequence gives a directed transition graph with edge weights $w(a,b) = \#\{t : \ell(s_t)=a, \ell(s_{t+1})=b, a \neq b\}$. A cursor also builds a tree: an ordinary step attaches sequentially and becomes current; an exploration step opens a branch; a verification step attaches as a leaf without moving the cursor; a backtracking step re-attaches under an earlier node.

**3 · Features.** Concatenate up to five blocks — type frequencies and transition rates (137–159 dims), per-edge-type transition matrices and hashed root-to-leaf paths (430–1,150), binary strategy-pattern indicators (64), conversation-level lengths and entropies (83), and a signed feature-hash bag of route labels (1,024). Released corpora use all five, giving $D = 1{,}738$.

**4 · Standardise and project.** Using pool-wide mean and standard deviation, computed *before* either condition is picked:

$$z_j(y) = \operatorname{clip}\!\left(\frac{f_j(y)-\mu_j}{\sigma_j},\,-5,\,5\right), \qquad \phi(y) = \frac{\hat z(y)P}{\lVert \hat z(y)P \rVert_2}$$

where $P \in \mathbb{R}^{D\times 96}$ is a fixed-seed Gaussian random projection. Distances are plain Euclidean.

### The selection

From one eligible pool, at one budget $n$, build two matched datasets.

**Diverse** ($\mathcal{D}_{\mathrm{div}}$): split the budget across domains as evenly as capacity allows; run mini-batch k-means (40,000 clusters total on the released corpora, seed 42); give each cluster a budget $b_c \propto |C_c|$; inside each cluster, greedy farthest-point selection:

$$S^{(t+1)} = S^{(t)} \cup \Big\{\arg\max_{i \notin S^{(t)}} \min_{j \in S^{(t)}} \lVert \phi_i - \phi_j \rVert_2 \Big\}$$

seeded at the cluster's centroid-nearest point. This is the greedy $k$-center coreset construction.

**Similar** ($\mathcal{D}_{\mathrm{sim}}$): per domain, keep the $n_d$ rows with the smallest $\lVert \phi_i - \mu_d \rVert_2$. One dense blob.

Both sets are the same size. The only random input is the k-means seed.

### Training

SFT is plain token-level negative log-likelihood on the response — see [[Cross Entropy]]. Then GRPO from that checkpoint: $G$ rollouts per prompt, binary verifier reward, group-normalised advantage $\hat A_g = (r_g - \bar r)/\mathrm{std}(r)$, clipped importance ratio, per-token KL penalty $\beta = 0.001$ toward the SFT checkpoint. No entropy bonus. The [[KL Divergence|KL]] leash and clipping are inherited from [[PPO]].

Everything else is matched between conditions: student model, prompt pool, trajectory budget, SFT recipe, RL recipe, evaluation protocol, and **the checkpoint step at which both are read**.

### The headline setups

- **RLVE** (procedurally generated puzzle environments with rule-based verifiers, adjustable difficulty). SFT on difficulty 1–5, RL through 10, evaluation to 15. This lets them test generalisation *above* anything either stage trained on.
- **Single-model condition**: Qwen3-4B-Thinking-2507 writes *every* candidate, so teacher identity cannot explain anything.
- **Three released corpora**: OpenThoughts3, INTELLECT-3, Nemotron-Cascade 2. Select 100k rows from the released solutions, SFT an OLMo3-7B, run the same 64-step GRPO.

### Why coverage is the metric

$$\text{pass@}k = \frac{1}{M}\sum_i \mathbf{1}\big[\textstyle\max_j r_{ij} = 1\big]$$

The fraction of held-out problems solved at least once in $k$ tries. If per-sample success is $p(x)$, then $\text{pass@}k(x) = 1-(1-p(x))^k$. Raising the number of problems with $p(x) > 0$ raises this directly, and those are exactly the problems that can start producing mixed reward groups.

## Ablation Studies and Experiments

### Teacher count as a warm-up proxy (Section 2)

Before selecting routes directly, they use "how many different models wrote the solutions" as a crude stand-in for variety, at a **fixed trajectory budget**.

- Qwen3-4B on Enigmata: 12 teachers beat 1 by ~18 points of pass@64. Growing the *task pool* from 16 to 399 environments adds at most 6.4 points, and under half a point once you already have 12 teachers. Variety of routes mattered more than variety of tasks.
- Qwen3-4B, SFT on puzzles → RL on DAPO-Math-17k → evaluate on maths. MATH-500 pass@1 goes **34.14% → 65.08%** (16-env pool). Twelve teachers lead in 54 of 56 benchmark × pool × budget cells.
- RL on reasoning-gym, evaluate on OMEGA out-of-distribution: neither stage touches OMEGA, and 5 teachers still beat 1 at every budget.

Caveat the authors state plainly: teacher count changes *which* generators, not just how many, so this is suggestive rather than clean.

### The direct route-selection result

**OLMo3-7B on RLVE, same RL for both:** route-diverse SFT leads by **16.9 points of pass@8** on environments held out from SFT. It leads in every difficulty band including 11–15, above anything either stage trained on.

The solved-set overlap is the number I would keep. Diverse retains 95.67% of what Similar solves, and adds **1,133 questions Similar misses**. Similar uniquely solves **53**. This is expansion, not a trade.

**Single-model condition** (one teacher writes everything, Qwen3-4B-Base student, 10 competition-maths benchmarks):

| SFT budget | mean pass@8 gain, Diverse over Similar |
|---|---|
| 10k | +6.17 |
| 25k | ~ |
| 50k | +3.39 (range across budgets 3.39–6.17) |

So the effect survives with teacher identity held perfectly constant. That rules out the obvious "you just mixed in a better model" explanation.

### Released corpora — selection baselines

Against random, a topology baseline (same fingerprints, simpler rule), gradient-diversity (OLMo3-7B forward pass, 4096-dim surrogate), embedding (Qwen3-Embedding-8B), and lexical TF-IDF farthest-point selection: **their method wins every comparison of mean post-RL score**, relative gains 1.2%–10.8% at pass@1 and pass@8.

Against the Similar selection from the same pool: leads on every maths benchmark in all three corpora by 4.9–18.5 points, plus all three OMEGA splits and GPQA-Diamond.

### What the ablations actually reveal

The fingerprint spread is measurably different but *not* because diverse traces use more kinds of reasoning:

| statistic (per prompt) | Diverse | Similar | ratio |
|---|---|---|---|
| response pairwise distance | 11 | 5.0 | 2.3× |
| topology pairwise distance | 23 | 9.8 | 2.3× |
| pattern entropy | 4.9 | 4.9 | 1.0× |
| active-pattern count | 32 | 31 | 1.0× |

Same *vocabulary* of reasoning patterns, same evenness — just further apart in how they are arranged. That is a useful constraint on what "diversity" means here.

After RL, correct completions from diverse checkpoints are also more lexically varied: mean pairwise bigram Jaccard distance +16.68% (Qwen3-4B) and +15.13% (1.7B) on a 128-token prefix. But at a 512-token prefix the gap shrinks to +6.0% and +4.0%. The variety is concentrated in how solutions *open*.

### What did not work

This is the honest half, and the paper does report it.

- **Instruction following got slightly worse.** IFBench accuracy −0.6 points on OpenThoughts3, −1.0 on INTELLECT-3. IFEval ties on OpenThoughts3. Gains are not uniform across capabilities.
- **INTELLECT-3 Enigmata favoured Similar** — by 1.3 points mean accuracy and 4.25 points pass@4. That is a puzzle benchmark losing on a paper about puzzle-ish diversity.
- **Teacher count is non-monotonic.** In the reasoning-gym → maths transfer sweep, the mean over 8 benchmark×pool cells is 30.36%/59.04% (pass@1/pass@64) at *two* teachers and 21.48%/55.32% at *five*. More teachers is not a monotone dial.
- **Minerva reversals**: 3 and 4 teachers fall 1.8 and 3.3 points below 1 teacher at pass@64, starting from pass@8 and pass@4 respectively.
- **AIME 2025 in the 16-env pool** reverses at pass@8 and pass@16 for the Qwen3-4B twelve-teacher comparison.
- **Task-pool size has local reversals**: the 62-task pool trails the 52-task pool at every budget (pass@64 drops 22.10% → 20.70%).
- **The gap shrinks at large $k$.** Their toy model predicts it: the coverage gap $(1-\beta u_S)^k - (1-\beta u_D)^k$ is zero at $k=0$, peaks at a finite $k^\star$, then collapses as both policies saturate. Observed: peak at intermediate difficulty for Qwen3-4B, on the *easiest* band for OLMo3-7B, at the *smallest* capacity on OMEGA. The prediction is "largest near the edge of what the model already solves reliably", which is consistent but loose.

## Worth Remembering

**One training run per condition.** The authors say so repeatedly. Every margin in this paper — including the 16.9-point headline — is a single seed. The bootstrap intervals on the lexical-diversity numbers cover evaluation-set variation only, not training seeds. Treat the direction as well-supported and the magnitudes as soft.

**The unmeasured link.** Their account is: diverse routes → larger repertoire → higher $p(x)$ on more prompts → more mixed reward groups → more for RL to learn. They measure the endpoints (route spread; mixed-reward share; post-RL coverage). They do *not* measure the middle — that route diversity in training causes route diversity at sampling time on a given held-out prompt. They flag this in Appendix H as the obvious follow-up.

**A no-cost intervention.** This is what makes it interesting for practice. Your pipeline already verifies far more solutions than it trains on. No new generation, no teacher ensemble, no model in the loop. The selector is rules plus k-means plus a greedy pick.

**The mixed-reward framing generalises.** The quantity
$$P(\text{mixed} \mid x) = 1 - p(x)^G - (1-p(x))^G$$
peaks at $p(x) = 1/2$. That is the same shape as "reward variance" prompt-selection methods used *inside* RL ([[DRACO- Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training]] and the zero-variance-group filtering line). This paper's claim is that you can buy some of it earlier and more cheaply, by choosing data.

**Connections.** The coreset selection is the same greedy $k$-center used in active learning. "Spread out in representation space" is the same objective as uniformity in contrastive learning — see [[Understanding Contrastive Learning through Alignment and Uniformity]]. The worry about SFT narrowing the policy before RL runs through [[Locked at the Entrance, Open Inside- Where RLVR Narrows the Solution Space]] and [[Mode Collapse]].

**Practical caveats if you wanted to use this.**
- Both selections are drawn from the same pool and *need not be disjoint*. The comparison is spread-vs-concentration, not two disjoint halves.
- The fingerprint is domain-specific. RLVE reads step types from lexical cues in the text; OMEGA reads recorded annotations. You would need to write a step vocabulary for your domain.
- Fingerprint distance is a proxy for procedural difference and *also picks up wording*. The Table 3 calibration shows it separates geometry, not semantically distinct algorithms.
- Normalisation statistics are computed on the eligible pool *before* selection. If you recompute them per-condition you break the comparison.
- Cluster count matters: they used 40,000 clusters for 100k selected rows, so roughly 2–3 rows per cluster. The farthest-point step is doing very local work; the proportional cluster budgeting is doing the global spreading.

**Open question I would want answered.** The IFBench and Enigmata losses suggest the selection trades something away — possibly the consistent, well-formed, centroid-like traces that instruction following benefits from. Whether a mixed selection (mostly centroid, some spread) beats both was not tested.

## Links

Related: [[GRPO]] · [[PPO]] · [[Fine-Tuning]] · [[Instruction Tuning]] · [[Reward Function]] · [[Exploration vs Exploitation]] · [[Mode Collapse]] · [[Chain of Thought]] · [[Cross Entropy]] · [[KL Divergence]] · [[Test-Time Compute]] · [[Credit Assignment]] · [[Locked at the Entrance, Open Inside- Where RLVR Narrows the Solution Space]] · [[Is Next-Chunk Reasoning RL Really Better than SFT- Revisiting Training Strategies under no-CoT Data]] · [[Dynamic Important Example Mining for Reinforcement Finetuning]] · [[DRACO- Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training]] · [[Understanding Contrastive Learning through Alignment and Uniformity]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Training language models to follow instructions with human feedback]]

New topics worth writing: k-center coreset selection, farthest-point sampling, RLVR (reinforcement learning with verifiable rewards), zero-variance group filtering in GRPO, pass@k as a coverage estimator, random projection and Johnson–Lindenstrauss, feature hashing, mini-batch k-means, self-training and STaR, data selection for instruction tuning
