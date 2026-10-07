---
title: "Verifiable Hidden Dynamics Play: Generating Agentic RL Environments from Solved Mechanisms"
authors: ["Shen et al."]
year: 2026
arxiv: "2609.27321"
url: https://arxiv.org/abs/2609.27321
priority: Good-To-Read
read_on: 2026-09-28
tags: [paper, llm, rl]
---
## The Core Idea

Training an LLM agent needs two things per task: an environment it can act in, and a trustworthy score for what it did. Most pipelines that *generate* environments build the environment first, then bolt on a grader afterwards — a checklist, a rubric, an LLM judge. The grader is a second guess, and its validity has to be argued separately.

VHD-Play flips the order. Start with a **maths problem you can already solve**. Draw parameters $\theta$ from a family (inventory replenishment, knapsack, routing, linear programming). Run a real solver on it. Now you hold the optimal value $u^*(\theta)$ and the value of a fixed dumb default policy $u_0(\theta)$ **before any environment exists**. Then hand $\theta$ to a frozen LLM, which dresses the same decision process up as a stateful world with tools, a database and a story ("you run a hardware distribution centre in Shenzhen Bay").

The environment's physics and its scoring rule both descend from the same solved object. Nothing is aligned after the fact.

> [!NOTE] Mechanism-first construction ^mechanism-first
> Solve a parameterised mathematical model, then *render* it as an interactive environment. The solution fixes the reward rule; the model fixes the state transitions. Contrast with environment-first: build $E$, then define $\mathcal{R}_E$ or annotate trajectories.

The second idea is **information asymmetry as a dial**. The builder sees all of $\theta$. The agent sees only what the observation map $\Omega_\theta$ publishes. Hidden demand tables, misleading "forecast stickers" on the shelf, and *probe* actions that cost a turn but reveal the truth. So the agent has to gather information and then commit, and both spend the same budget. This is a [[POMDP]] where the latent state is the parameter draw.

What it unlocks: new environments cost a parameter resample plus one LLM call. They report 3,300 admitted environments at **$0.01–$0.03 each**, spanning 28 topical domains. And because [[Dynamic Programming]] solvers scale with the problem, you can crank up items and horizon and still have an exact reference — a training substrate that grows with the policy.

The headline finding is a diagnosis, not the pipeline. The base model solves these problems at **0.962** when written out in the prompt. Put the *same* problem behind a stateful interface *and reveal every parameter up front*, and it drops to **0.231**. The gap is not knowledge. It is operating.

## The Methodology

**The chain.** For a mechanism family $f$:

$$M(\theta) \xrightarrow{\text{solve}} z_\theta \xrightarrow{\text{fix}} \mathcal{R}_\theta, \qquad M(\theta) \xrightarrow{\text{realize}} D(\theta) \xrightarrow{\text{wrap}} E(\theta)$$

- $M(\theta)$ — the mathematical model: decisions, constraints, objective.
- $D(\theta) = (\mathcal{S}, \Lambda_\theta, U_\theta)$ — executable dynamics: state space, transition, utility.
- $E(\theta) = \text{Wrap}(D(\theta); \Omega_\theta, \mathcal{A}, T)$ — what the agent touches: observations, actions, a turn budget.

Standard [[Markov Decision Process]] plumbing: $a_t = \pi(o_{\leq t})$, $s_{t+1} = \Lambda_\theta(s_t,a_t)$, $o_{t+1} = \Omega_\theta(s_{t+1},a_t)$.

**The reward.** A family solver $S_f$ (no LLM involved) computes $z_\theta = (u^*(\theta), u_0(\theta))$ by closed form, enumeration, backward Bellman recursion, Held–Karp, HiGHS or SLSQP. Then:

$$r(\pi;\theta) = \text{clip}_{[0,1]}\!\left(\frac{u(\pi;\theta) - u_0(\theta)}{u^*(\theta) - u_0(\theta)}\right)$$

Plain words: score 0 means "you matched a policy that reads the misleading stickers, splits the budget evenly and never probes." Score 1 means "you matched the full-information optimum." Pure arithmetic over a realised number and two frozen constants. No [[Preference Learning|reward model]], no judge. Compare [[ImpossibleRubrics- Stress-Testing Generated Rubrics as Reward Signals|generated rubrics]], where the grader itself is the thing you have to trust.

Note $u^*$ is an *upper anchor*, not always attainable online — it was computed with full information the agent never gets.

**The setter.** A frozen Qwen3.6-35B-A3B receives (a) the full parameter vector and (b) one independently sampled real-world document passage. It emits an entity–relationship model, a backing relational database, **at least ten executable tool schemas and bodies**, the player instruction, and an executable default policy. The passage supplies entities, relations and vocabulary; $M(\theta)$ supplies the decision process. The setter's job is narrow — *render* a mechanism, not invent one — which is why a mid-size model suffices.

**Admission.** Three gates:
- **C1** executability — dynamics and tools initialise and run; reference policies reach a numeric terminal value.
- **C2** valid default region — $u_0$ must sit in a family-defined admissible band, else regenerate.
- **C3** reward agreement — replay the solver's optimal policy *through the generated tools* and check it reproduces the precomputed $u^*$ within tolerance.

Held-out audit: 10 environments × 8 replay-supported families × 3 policies (oracle / default / idle) × 2 repeats = 480 executions. Zero exceptions, all 240 repeated outcomes agreed, no replayed utility exceeded its stored $u^*$.

Yield: 65.6% admitted on the first setter attempt, mean 1.41 attempts, 31.6% had the tool interface rewritten.

**Training.** [[GRPO]], 64 prompts per step, 16 retained rollouts per prompt (oversampled to 18). Advantage = return minus group mean, **no** standard-deviation normalisation, **no** KL penalty, **no** value model. One inner epoch, lr $2\times10^{-6}$, clip ratio $4\times10^{-3}$. The reported checkpoint is **step 34** — 2,176 environments consumed, 34,816 scored rollouts. Budgets: up to 1,000 turns per episode, 16,384 tokens per turn, 120,000 cumulative generation tokens, 65,536-token managed context. Observed episodes: median 31 assistant turns, mean 59.7, p95 211.

Split: 2,200 train (3 families) / 300 held-out same-family / 800 across 8 unseen families.

## Ablation Studies and Experiments

**The three-presentation experiment — the most useful thing in the paper.** Same problem, same reward scale, three wrappers:

- $F$ **written-out** — full instance in the prompt, Python tool available, return an answer.
- $I$ **informed-agentic** — all parameters revealed up front, but you must act turn by turn through the environment.
- $A$ **agentic** — parameters only obtainable by probing.

| Family | $F$ | $I$ | $A$ |
|---|---|---|---|
| Inventory DP | 0.976 → 0.999 | 0.469 → 0.968 | 0.449 → 0.912 |
| Knapsack | 0.984 → 0.996 | 0.132 → 0.948 | 0.117 → 0.843 |
| Linear programming | 0.978 → 1.000 | 0.161 → 0.732 | 0.143 → 0.661 |
| Quadratic programming | 0.975 → 1.000 | 0.168 → 0.836 | 0.162 → 0.795 |
| Routing | 0.897 → 0.965 | 0.226 → 0.889 | 0.148 → 0.866 |
| **Mean** | **0.962 → 0.992** | **0.231 → 0.875** | **0.204 → 0.815** |

Read the $F \to I$ collapse carefully: **0.962 to 0.231 with nothing hidden**. The whole drop is carrying decisions through changing state, shared constraints and irreversible commitments. Hiding the parameters costs only another 0.027. Training closes 84% of the $I$ gap and 77% of the $F \to A$ gap, and $F$ barely moves (+0.030) — so nothing was traded away.

**Transfer across families** (base → step 34, agentic):

| Band | Families | Mean gain |
|---|---|---|
| Held-out, trained families | inventory DP, routing, negotiation | +0.56 |
| Near-OOD (unseen optimisation) | knapsack, LP, QP | +0.63 |
| Far-OOD, different decision structure | sequential stopping, sequential search | +0.24 |
| Far-OOD, probe-budgeted | facility location, scheduling, tardiness | +0.16 |

Weakest cells: sequential search +0.066, scheduling +0.085. Both are far-OOD and both have tight probe budgets.

**External benchmarks** (nothing shared with training):

| Benchmark | Base | Trained | Qwen3.7-Max |
|---|---|---|---|
| BFCL V4, 10 interaction cells (mean) | 61.25 | 64.08 | — |
| TravelBench plan score | 0.700 | 0.794 | 0.891 |
| E-Commerce Bench, mean ending balance | 54,294 | 182,844 | 165,224 |

The 365-day storefront is the striking one: base went bankrupt in 1 of 5 runs, trained completed all 5 over 1,147–1,825 assistant turns and **beat Qwen3.7-Max**. That is ~20× the 59.7-turn training average. BFCL: 8 of 10 cells up, 2 flat, biggest gains on relevance (+6.25) and missing-function (+6.00).

**What did *not* work / controls:**

- **"It just learned to call the Python tool."** Oaxaca–Blinder decomposition attributes only **+0.030** of a **+0.443** agentic gain to increased code-tool adoption; **+0.413** is improvement *within* fixed tool-use strata. Adoption did rise, 50.4% → 79.2%. Routing's adoption term is *negative* — its code-using base episodes scored below its own mean.
- **Elicitation, not rewriting.** Forward $D_{\mathrm{KL}}(\pi_{\text{trained}} \| \pi_{\text{base}}) = 0.089$ nats per assistant token on held-out states (95% bootstrap $[0.082, 0.097]$), higher early in trajectories (0.150) than mid (0.048). A small policy shift buys a large behavioural one — see [[KL Divergence]].
- **The matched environment-first control is the sharpest negative result in the paper.** They took 2,200 released EnvScaler scenarios (generate environment → then generate checklist validators), same base model, same GRPO config, same 34 steps. Native reward went up and TravelBench improved 0.700 → 0.753. But **BFCL fell to 58.56, below the 61.25 base**, and storefront balance stayed flat at 52,661. EnvScaler's own paper shows big gains from an 8B start (BFCL-v3 multi-turn 28.88 → 41.88); those gains did not reproduce from a stronger 35B start.
- **Setter capacity.** A frozen copy of the *starting* 35B checkpoint, acting as setter, emitted 22/30 requested records and 21/22 survived reference replay — 70% effective admission versus 100% for Qwen3.7-Max. The weaker setter needed 2.50 attempts on average versus 1.03, and rewrote 45.5% of interfaces versus 0%. Those retries erase its per-token price advantage: **both land at $0.01–$0.03 per admitted environment**.
- **Co-scaling.** Four knapsack scales, a fresh checkpoint trained and evaluated at each:

| items × periods | Base | Trained | Gain |
|---|---|---|---|
| 6 × 5 | 0.432 | 0.843 | +0.412 |
| 8 × 7 | 0.321 | 0.872 | +0.551 |
| 9 × 9 | 0.237 | 0.912 | +0.675 |
| 11 × 11 | 0.085 | 0.946 | +0.861 |

The gain *grows* with scale. Same checkpoint can build an environment it cannot yet navigate.

**Trajectory evidence** (paired, illustrative — not prevalence estimates). Base, given all 8 items revealed and a code tool, writes *"For each period, find best combination"*, burns 27 of 29 shared budget units in the first three periods, scores 0. Trained writes *"I need to plan across all 7 periods"*, builds one horizon-wide allocation, scores 1. Second case: base says *"Need to probe items to learn true profits"*, probes 4 of 8, then commits. Third: base identifies the winner after 2 inspections but scouts all 14, paying 134 in inspection cost for net $-40.71$; trained stops at 8 inspections costing 27 — *"Any additional scouting would only reduce my net value"* — for net 66.29. That last one is [[Exploration vs Exploitation]] arithmetic being done explicitly, which is basically Weitzman search.

## Worth Remembering

**The transferable insight, independent of the pipeline.** If you want to know whether your agent's failure is *reasoning* or *operating*, run your task in both forms: fully-specified prompt vs. stateful interface with everything revealed. The $F \to I$ gap is the operating deficit. 0.962 → 0.231 here is enormous and was invisible to any static benchmark.

**Only one reward, at the end.** No process rewards, no turn-level shaping. Yet it fixed a 1,825-turn storefront. That is a notable data point against the assumption that long horizons demand dense [[Credit Assignment]] — compare [[DRACO- Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training]], which argues the opposite. The authors flag this as a limitation, not a claim.

**Why no [[Reward Hacking]] story.** There is nothing to hack. $u^*$ and $u_0$ were computed by HiGHS and backward induction before the environment's code was written. This is the cleanest version of verifiable reward available for interactive tasks — but it only exists because the task came from a family with a solver.

**And that is the binding constraint.** Every family here is a solved operations-research model. Coverage beyond that is an open question. The authors say so plainly: they do not claim OR is the only or a sufficient substrate. Two families (sequential stopping, sequential search) are scored against *hindsight ceilings* no online policy can reach, so their absolute numbers compare arms rather than measure distance to optimum.

**Caveats for anyone wanting to use it:**
- One training run, one evaluation seed on generated families. No variance bars on the headline 0.204 → 0.815.
- 3 training families, 11 evaluated. The far-OOD gains (+0.16, +0.24) are much smaller than the near-OOD ones (+0.63).
- Training uses the whole construction at once. They never ablate the reward rule against the information boundary separately, so you cannot tell which half is doing the work.
- Adding a family is not free: you write a parameter sampler, a reference procedure, and a realisation/replay adapter, once. Cheap thereafter.
- The $F$ score staying at 0.992 is only checked on five optimisation families. No broader regression suite, so the [[Fine-Tuning#The failure modes 🪤|forgetting]] question is open.

**Open questions worth chasing.** Does the mechanism-first order help in domains where the "solved model" is a bandit or an [[Off-Policy Evaluation|off-policy estimator]] rather than a deterministic program — e.g. generate logged-bandit environments where the true propensities and the optimal policy are both known by construction? That would give exact ground truth for OPE estimators, which is precisely what [[Open Bandit Dataset and Pipeline- Towards Realistic and Reproducible Off-Policy Evaluation|OBD]] has to work hard to approximate.

## Links

Related: [[GRPO]] · [[Reward Function]] · [[POMDP]] · [[Markov Decision Process]] · [[Dynamic Programming]] · [[Exploration vs Exploitation]] · [[Credit Assignment]] · [[Agent Evaluation]] · [[Evals]] · [[Tool Use]] · [[Agentic Workflows]] · [[Reward Hacking]] · [[KL Divergence]] · [[AgentMercury- Your Agent Can Synthesize Verifiable Environments for Business Scenarios at scale]] · [[ImpossibleRubrics- Stress-Testing Generated Rubrics as Reward Signals]] · [[Small Language Models as Judges for Rubric-Based Reinforcement Learning]] · [[DRACO- Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training]] · [[Beliefs]] · [[Multi-Armed Bandit]] · [[Off-Policy Evaluation]] · [[Open Bandit Dataset and Pipeline- Towards Realistic and Reproducible Off-Policy Evaluation]]

New topics worth writing: Oaxaca–Blinder decomposition for ablating behavioural change, RLVR (reinforcement learning with verifiable rewards) as a design pattern, Weitzman sequential search and the reservation-price rule, Held–Karp exact TSP recursion, Smith's rule for weighted completion time, procedural environment generation for agent training, information asymmetry as a training-difficulty dial, BFCL V4 benchmark structure, E-Commerce Bench long-horizon evaluation
