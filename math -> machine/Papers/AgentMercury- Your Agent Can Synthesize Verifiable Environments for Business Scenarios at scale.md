---
title: "AgentMercury: Your Agent Can Synthesize Verifiable Environments for Business Scenarios at scale"
authors: ["Jeong et al."]
year: 2026
arxiv: "2608.20634"
url: https://arxiv.org/abs/2608.20634
priority: Good-To-Read
read_on: 2026-08-31
tags: [paper, rl]
---
## The Core Idea

Agents learn by doing things in an environment. But somebody has to build the environment first. Today that "somebody" is a human engineer, or a script that generates a world *around a task you already picked*. If you want an agent that files support tickets, you build a ticket world. The world is a prop for the task.

This paper flips the order. Start from a **business scenario** — "a mid-size logistics firm in Poland" — and build a whole running world from it: a company, its software services (CRM, email, Slack, billing, ticketing), the database tables behind them, the tools that read and write those tables, a seeded starting state, and a set of rules that should hold across services. *Then* sample many tasks out of that world. One world, many tasks.

The authors name this third role **Planet**, and put it next to the two roles everyone already talks about:

- **policy** $\pi$ — what the agent *does*.
- **world model** $W$ — how the world *responds* (see [[Mastering Diverse Domains through World Models (DreamerV3)]], [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)]]).
- **Planet** — what world *exists at all*.

> [!NOTE] Planet role
> The environment-authoring role. Takes a high-level scenario $\sigma$ and emits a complete, executable world $w$, before any task exists. ^planet-role

Two results make this more than a taxonomy.

**One: training on these business worlds transfers wildly out of domain.** Qwen3.5-4B trained with RL only on synthetic enterprise workflows went from 45.9 → 56.0 on AIME26 (competition maths), 28.5 → 35.4 on HMMT, 36.6 → 44.0 on LiveCodeBench. None of those benchmarks were used to build the environments. Something about "track state, call tools, satisfy constraints" is a general skill.

**Two: world-building itself is learnable.** They saved the intermediate traces of their own construction pipeline and fine-tuned a model on them. Success at authoring a valid executable world from an unseen business brief jumped from **3.3% → 83.3%**.

Why this could not exist before: you need (a) models strong enough to write a coherent multi-service schema, and (b) a *deterministic* way to check whether the resulting world is real. Point (b) is the quiet engineering trick — the world's rules are compiled into SQL checks over the database, so both "is this world valid?" and "did the agent succeed?" are yes/no, not vibes.

## The Methodology

The whole pipeline is one chain:

$$\sigma \xrightarrow{\textsc{Planet}} w \xrightarrow{\textsc{Task}} (u,\rho) \xrightarrow{\pi} \tau \xrightarrow{\textsc{Grade}} r$$

$\sigma$ is the scenario, $w$ the world, $u$ the user instruction, $\rho$ the rubric, $\tau$ the trajectory, $r \in [0,1]$ the reward.

### Building the world

A world is the usual POMDP-ish tuple plus one extra piece (compare [[Markov Decision Process]]):

$$w = \langle \mathcal{S}, \mathcal{A}, \Omega, T, O, s_0, \mathcal{R}\rangle$$

$T$ is a **real** transition function — actual code hitting an actual database, not a learned approximation. $\mathcal{R}$ is the new bit.

> [!NOTE] World-level invariant
> A rule the world *expects* to hold but does **not** enforce. Example: "after an upstream sale is recorded, a downstream billing record must exist." The simulator will not create that record for you. The agent has to. ^world-invariant

This is the crux. If you hard-code invariants into $T$, the world does the agent's job. By pulling them out of $T$ and into a checkable set $\mathcal{R}$, the same world can host many different tasks, and cross-service correctness becomes something the agent must *earn*.

Each invariant has two faces:
- $\mathcal{R}_{\text{vis}}$ — surfaced as in-world documents and policies the agent has to *find* by reading around. Not handed over at task start.
- $\mathcal{R}_{\text{hid}}$ — executable SQL conditions used only by the grader.

Same rules, two views.

Generation is factorised into stages so each step conditions on the last:

$$\textsc{Planet}(w \mid \sigma) = p(C\mid\sigma)\, p(G\mid C)\, p(\Sigma\mid G,C)\, p(s_0\mid\Sigma)\, p(\mathcal{R}\mid G,\Sigma,s_0)$$

$C$ = company identity (anonymised, to stop the model hallucinating real firms), $G$ = service graph, $\Sigma$ = state schema. The state and action spaces fall out of $G$ and $\Sigma$.

### Instantiating a task

$$(\Delta s_0, u, \rho) \sim \textsc{Task}(\cdot \mid s_0, \mathcal{R})$$

$\Delta s_0$ patches the world's initial state for this specific task; $u$ is the natural-language ask; $\rho$ is the task rubric. The world structure never changes — only the starting state and the goal. Ten task seeds per environment.

### Interaction and grading

Standard loop, agent sees observations not state:

$$s_t \xrightarrow{\pi} a_t \xrightarrow{T} s_{t+1} \xrightarrow{O} \omega_{t+1}$$

Because $T$ is deterministic and $s_0$ is seeded, any trajectory can be replayed and re-scored exactly. Grading happens *after* the episode:

$$r(\tau) = \textsc{Grade}(s_H, \tau; \rho, \mathcal{R}_{\text{hid}})$$

Mostly deterministic assertions over the final database — record exists, field equals, field not-equals — with an LLM judge (DeepSeek-V4-Flash, temperature 0) only where needed. Mean 5.4 assertions per task, range 2–10.

### The numbers of the corpus

| | |
|---|---|
| Environments released | 4,783 (4,326 used for RL) |
| Tasks | 43,300 |
| Industries / countries | 14 / 50 |
| Unique tools | 842 |
| State tables | 222 |
| Service combinations | 148 |
| Tools per world | 10–26, mean 16.1 |
| Services per task | 1–5, mean 2.9 |

25,991 of 43,300 tasks are flagged as carrying cross-system action risk.

### RL setup

GRPO (Group Relative Policy Optimization — sample $k$ answers per prompt, use the group mean as the baseline instead of a learned value head; the descendant of [[Proximal Policy Optimization Algorithms]] used in [[Training language models to follow instructions with human feedback|RLHF]] pipelines). 8 samples per prompt, global batch 128, rollout batch 16. Decoupled-PPO surrogate with asymmetric clipping $\epsilon = 0.20$ / $\epsilon_{\text{high}} = 0.28$ (the Dr. GRPO fix). Adam, $\beta_1{=}0.9$, $\beta_2{=}0.98$, constant LR $1\times10^{-6}$, no warmup, no weight decay, no entropy bonus. GRPO std-normalisation **off**.

One detail that matters: groups whose 8 rewards have zero standard deviation are dropped. If every rollout scored the same, there is no relative signal, so the gradient is noise.

Rollouts: up to 20 tool-call turns, 16,384 response tokens, 8,192 tokens per turn, temperature 1.0. Fully asynchronous — 2 GPUs on the actor, 6 on rollout generation, staleness cap of 4 updates, minimum fresh-token ratio 0.75. Environments run in-process (no network) via MCP tool servers.

They also ran **SAO** (single-rollout asynchronous optimization) as a second RL algorithm, to show the environments are not tied to one optimiser.

## Ablation Studies and Experiments

### In-domain: EnterpriseOps-Gym (8 business domains, pass@1, mean of 3 runs)

| Model | Avg. |
|---|---|
| GPT-5 | 30.9 |
| Kimi-K2-Thinking | 22.4 |
| Gemini-2.5-Pro | 19.9 |
| Qwen3.5-4B | 12.3 ± 0.2 |
| **+ GRPO + AgentMercury** | **15.7 ± 0.6** |
| Qwen3.5-35B-A3B | 24.8 ± 0.6 |
| **+ GRPO + AgentMercury** | **28.1 ± 1.4** |
| **+ SAO + AgentMercury** | **28.3 ± 1.5** |

At 4B the biggest jumps are Drive (6.2 → 15.6) and Email (23.9 → 33.3). **CSM got worse: 9.2 → 5.6.** Seven of eight domains improved. At 35B, all eight improved under both optimisers.

### Out of domain (none of these shaped the training environments)

| Benchmark | 4B base | 4B + GRPO |
|---|---|---|
| AIME26 | 45.9 | **56.0** |
| HMMT | 28.5 | **35.4** |
| LiveCodeBench v5–6 | 36.6 | **44.0** |
| SciCode | 22.6 | **25.7** |
| τ³-Airline | 48.8 | **58.7** |
| τ³-Retail | 70.4 | **73.6** |
| τ³-Telecom | 92.5 | 91.9 |
| BFCL | 30.3 | **31.7** |
| GPQA-Diamond | 76.5 | **77.5** |

Telecom, already at 92.5, did not move. The authors read this as evidence the gain is not a uniform reward shift.

At 35B the maths benchmarks are near-saturated (AIME26 91.0 → 92.2) but **BFCL jumps 31.1 → 42.1** — the biggest single gain in the paper, and BFCL is pure structured tool-calling. That is the most honest signal about what the training actually teaches.

### The variance result, which is arguably the real finding

The 35B base model on τ³ is not weak on average, it is *unstable*: Telecom $49.1 \pm 49.8$, Retail $52.7 \pm 22.1$. Standard deviation as large as the mean. After training: Telecom $65.5 \pm 23.8$, Retail $55.7 \pm 10.2$, Airline $39.1{\pm}10.6 \to 50.9{\pm}8.1$. The policy became *reliable*, not just better. For multi-step agents this matters more than mean score.

### Training dynamics (the anti-reward-hacking check)

Reward climbs steadily. Response length rises then falls. Truncation ratio falls from ~0.35 to ~0. Degenerate-response ratio stays near zero. So the reward is not being won by rambling or by malformed output. Out-of-domain gains appear gradually across checkpoints, not in one jump — evidence against evaluation noise.

### Can a model learn to *build* worlds? (30 held-out briefs, 12 structural validators, all-or-nothing)

| Model | Zero-shot | + Recipe |
|---|---|---|
| Claude Opus 4.8 | 83.3% | 80.0% |
| GPT-5.4 | 66.7% | 66.7% |
| DeepSeek-V4-Pro | 83.3% | 86.7% |
| GLM-5.2 | 80.0% | 76.7% |
| Qwen3.5-35B-A3B | **3.3%** | 20.0% |
| Qwen3.5-35B-A3B + fine-tune | **83.3%** | **10.0%** |

Fine-tuning used 29,823 samples of construction traces, mixing four kinds of supervision: brief→world, completing a half-built world, repairing a corrupted world given validator feedback, and intent→diff. Fisher exact test $p = 1.2\times10^{-10}$. The trained model passes 11.5 of 12 validators on average.

### What did **not** work

- **Giving the recipe (invariant digest + trimmed exemplar) does not reliably help.** Across the five API models the mean pass rate *fell* slightly, 80.7% → 78.0%, and McNemar gives $p \ge 0.375$ for every model. It is prompt sensitivity, not a real effect. Sometimes a longer prompt just adds more formatting to get wrong.
- **The recipe actively destroys the fine-tuned model: 83.3% → 10.0%.** 27 of the 30 failures are cross-service validation errors. Once the procedure lives in the weights, extra procedural text fights the learned generation policy. This is also the cleanest argument that the fine-tune taught construction, not instruction-following — if it were just following instructions, more instructions would help.
- **SAO fails at 4B.** A single rollout per prompt gives too weak a signal for a small policy; the run was unstable and excluded from the main table. GRPO's relative-baseline-within-a-group is what carries small models.
- **CSM regressed** at 4B (9.2 → 5.6), unexplained.
- **The dominant zero-shot authoring failure is cross-service invariant collapse** — the model puts the trigger and the target of a rule inside the *same* service. GPT-5.4 does this on 10 of 30 briefs. The spec looks syntactically fine. Only executing it catches the error.

### What the diversity analysis actually shows (Appendix A, and it is honest)

- t-SNE on TF-IDF of task *text* produces pretty clusters that mean **nothing**. $k$-means with $k{=}9$ gives clusters with near-identical metadata composition (CRM ~53–61% of every cluster, Account Manager dominant in 53–63% of every cluster). Text embedding distance is a bad proxy for task diversity here.
- t-SNE on the *tool set* (binary vector over 750 tools) gives clean domain-separated clusters. Same tasks, different representation. **The variation lives in the executable surface, not the language.**
- Mean pairwise text similarity is 0.13 (broad), but the *nearest-neighbour* median similarity is 0.83, and 8.3% of tasks have a near-twin above 0.99. Only 17% of those twins are sibling seeds from the same environment — so the redundancy comes from shared task templates across different companies, not from the 10-seeds-per-world design.
- Tool count takes exactly five values: $\{10, 14, 18, 22, 26\}$. Ten core tools plus four per additional resource. Task "shape" is a low-dimensional lattice, not a continuum.

## Worth Remembering

**Only 7.4% of the corpus received gradients.** The reported 4B run touched 3,200 of 43,300 tasks over 200 steps. Yet that subset covered 53.5% of environments, 62.9% of industries, 75.8% of tools. The lesson: with scenario-first generation, task *count* and world *diversity* are different axes. You do not need to burn the whole corpus.

**The loop is not closed, and the authors say so.** Planet does not learn from the policy's failures. There is no mechanism that says "the agent keeps botching reconciliation across billing and CRM — generate more of those worlds." That would be an adaptive curriculum, and it is the obvious next paper. Right now this is offline data generation with better structure.

**No world model is trained.** They write down $W(s_{t+1}\mid s_t, a_t, \mathcal{R}) \approx T$ and then explicitly decline to learn it, using the real executable $T$ instead. Cheaper and exact, but it means you cannot search over hypothetical worlds before instantiating them.

**GRPO vs SAO is a scale story.** GRPO's group-relative baseline is essential at 4B. SAO needs no repeated sampling per prompt, so one prompt budget covers more distinct environments — plausibly better at large scale where you want breadth over sharpness. This is a genuine trade-off, not a bug.

**Practical caveat if you want to use this:** the benchmark separation is real but soft. Business tool-use tasks and τ³/BFCL tool-use tasks are cousins. The AIME26 and LiveCodeBench gains are the surprising ones and deserve more scrutiny than one paper gives them — a 10-point AIME jump from training on CRM workflows is the kind of claim that invites the [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|baseline-scrutiny]] treatment. Was the 4B base fully tuned before RL? Compare the concerns in [[On the Difficulty of Evaluating Baselines]] and [[Towards Quantifying Benchmark Optimization in ASR Models]].

**The design idea worth stealing regardless:** separating *rules the world checks* from *rules the world enforces*. Most simulators enforce their own consistency, which quietly removes the hardest part of the job from the agent. Moving invariants into a post-hoc verifier makes cross-system correctness a learnable objective rather than a free gift.

**Open questions.** Does the near-twin population (8.3% at >0.99 similarity) inflate the apparent scale? What happens if you strip the tasks down to a single service — is cross-service structure the whole source of the transfer? And does the transfer survive at 100B+, or is it just filling gaps that bigger models already covered? This connects to [[The Bitter Lesson (essay)]]: scaling *environments* rather than *hand-designed tasks* is the same argument one level up.

## Links
Related: [[Proximal Policy Optimization Algorithms]] · [[Training language models to follow instructions with human feedback]] · [[Markov Decision Process]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)]] · [[The Bitter Lesson (essay)]] · [[Autonomous Mathematical Discovery in an Open-World Multi-Agent Environment]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[τ_0-VLA- a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation]] · [[On the Difficulty of Evaluating Baselines]] · [[Towards Quantifying Benchmark Optimization in ASR Models]] · [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]]

New topics worth writing: GRPO (Group Relative Policy Optimization), RLVR — reinforcement learning from verifiable rewards, Model Context Protocol (MCP) tool servers, τ-bench and τ³ agent evaluation, EnterpriseOps-Gym, single-rollout asynchronous optimization (SAO), asynchronous RL rollout infrastructure and staleness, synthetic environment generation for agents, automatic curriculum learning, McNemar and Fisher exact tests for paired model comparison
