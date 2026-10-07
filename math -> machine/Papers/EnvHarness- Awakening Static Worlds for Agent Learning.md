---
title: "EnvHarness: Awakening Static Worlds for Agent Learning"
authors: ["Chengsong Huang", "Zifeng Wang", "Rujun Han", "Jun Yan", "Yanfei Chen", "Zoey CuiZhu", "Ke Jiang", "Peng Xia", "Han Yu", "Yufan Zhuang", "Yifei Ming", "Jiaqi Pan", "Bhavana Dalvi Mishra", "Jiaxin Huang", "Burak Gokturk", "Tomas Pfister", "Chen-Yu Lee"]
year: 2026
arxiv: "2608.19880"
url: https://arxiv.org/abs/2608.19880
priority: Low-Priority
read_on: 2026-09-29
tags: [paper, llm, rl, theory]
---
## The Core Idea

Agents learn by acting inside environments. Someone hand-builds each environment: the tasks, the rules, the checker that says "you passed". Building one is slow, and once built it never changes. The same environment greets a weak agent and a strong one with identical behaviour. That is bad in two ways. It cannot aim at the particular thing *your* agent is bad at, and once your agent solves everything in it, it has nothing left to teach.

The usual fix is to generate new environments with an LLM. Two problems. Each generator only works for one domain — a web-page generator cannot make coding tasks. And the generated checker may be wrong, so you must over-generate and throw most of it away, and still cannot fully trust what survives.

**EnvHarness** takes a different route: do not build a new world, wrap the old one.

The trick is to sit *on top of* the environment's standard two functions — `reset()` (start an episode) and `step(action)` (do one thing). A wrapper can change what the agent sees, what it is allowed to do, and where it starts, without touching a single line of the environment's code. Because the task and the scoring function underneath are untouched, **every reshaped environment keeps the original human-written verifier.** That is the whole safety argument, and it is why this beats generation: you get new training signal for free, and correctness is inherited rather than re-earned.

The authors frame it as a mirror of the "agent harness" idea. An agent harness bolts memory, tools and loops onto a frozen model to make an agent. EnvHarness bolts wrappers onto a frozen environment to make a new environment.

$$\text{Agent} = \text{Model} + \text{Harness} \qquad\longleftrightarrow\qquad \text{Customized Env} = \text{Static Env} + \text{EnvHarness}$$

> [!NOTE] EnvHarness
> A wrapper layer over an existing environment that changes its behaviour purely through the `reset`/`step` interface. Because it never reaches inside, one implementation works for every domain, and the original task and verifier survive intact. ^envharness-def

The second half is **EnvRigger**: an automated loop that watches the agent fail, guesses *why*, writes a wrapper to attack that specific weakness, and tests the wrapper by running the agent again. It treats the agent as a black box — no weights, no logits, just trajectories. This makes the environment conditioned on *this policy right now*, which is what static environments can never be.

What it unlocks: environments and policy can co-evolve. Each round targets the boundary of what the agent can currently do. On SWE-bench Verified the success rate keeps climbing to 54.79 at 300 environments while the original environments flatten at 52.13 and generated ones at 50.37.

## The Methodology

### The formal object

An environment is a tuple $E = (\mathcal{S}, \mathcal{A}, \mathcal{O}, T, R, s_0)$ — states, actions, observations, transition function, reward (the verifier), initial state. A harness component is a transformation

$$E' = w(E), \qquad E' = (\mathcal{S}', \mathcal{A}', \mathcal{O}', T', R', s_0')$$

that only operates at the interface. Crucially $R$ is *never* exposed. Reward stays the benchmark's own verdict.

### Three components

**Stage — change where the episode starts.** Parameterised by a list of actions $\delta = (a_1, \dots, a_k)$ that get replayed through `step()` before the agent takes over:

$$s_0' = T(\cdots T(T(s_0, a_1), a_2)\cdots, a_k)$$

Note the elegance: because the setup is *replayed through the environment's own action vocabulary*, the new starting state is always reachable and always legal. No poking at internal state.

Running example: ALFWorld task "put a clean mug on the desk". The default puts the mug in plain sight. A Stage runs `take mug 1`, `open drawer 1`, `put mug 1 in drawer 1`, `close drawer 1` — now the agent must *search*. Or the reverse: pre-clean the mug to shorten the task for a weak agent.

**Contract — rewrite the interaction.** A triple $r = (f_A, f_T, f_O)$, each defaulting to identity:

$$(\mathcal{A}', \mathcal{O}', T') = \big(f_A(\mathcal{A}),\; f_O(\mathcal{O}),\; f_T(T)\big)$$

- $f_A$ blocks or rewrites actions — e.g. remove teleport-navigation so the agent must walk.
- $f_T$ rewrites the environment's response — e.g. refuse `clean mug` unless the agent is holding it.
- $f_O$ filters what the agent sees — e.g. truncate the room description to two sentences, forcing multi-step spatial exploration.

In code a Contract is a Python subclass overriding up to three hooks. The stored form of the component is literally the source string; it is recompiled per episode inside a subprocess so bad generated code kills an episode, not the framework.

**Chain — join two environments into one episode.** Parameterised by $(E_{\text{ext}}, g)$. Spaces become unions ($\mathcal{A}' = \mathcal{A} \cup \mathcal{A}_{\text{ext}}$) and the composite reward is the **conjunction** $R' = R_A \wedge R_B$ — each half judged by its own original verifier. The agent works in $A$, gets a hand-off observation, continues in $B$, under one shared step budget. Sub-environment terminations are masked so only the composite decides the episode is over. $B$ is reset lazily at the hand-off, so a container never boots if $A$ fails early.

Components compose, and the order matters ($w_1 \circ w_2 \neq w_2 \circ w_1$):

$$E' = w_{\text{chain},\ell}\big(w_{\text{contract},r}(w_{\text{stage},\delta}(E))\big)$$

### The interface that makes it domain-agnostic

Everything derives from one abstract class, `ActionableEnv`: `reset`, `step`, `evaluate`, `observe`, `get_env_state`, `save_state`/`from_state`. A **Bridge** is a benchmark's direct implementation and is the *only* layer that knows about the runtime — a Docker container, a Playwright browser, a TextWorld engine. Seven Bridges cover four runtime classes.

The key restriction: `get_env_state()` returns a **runtime-safe view** — plain data, no container handles, no sockets. Wrapper hooks may only read that. This is exactly why the same wrapper code runs over a browser and over a repository.

`observe()` is deliberately split from `reset()`, because a Stage mutates the world *after* reset but *before* the agent acts, and the outer layer needs to re-read without paying for another reset.

### EnvRigger: the four stages

Given base env $E$, task $t$, policy $\pi$, produce

$$E' = \mathcal{H}(E, t; \pi) = (w_k \circ \cdots \circ w_1)(E)$$

1. **Observe.** Run $\pi$ on $t$ for $K=5$ rollouts. Failures show the weakness; successes bound it.
2. **Diagnose.** Find root causes — repetitive action loops, mis-parsed long observations, misread tool constraints. Also decides direction: struggling policy → scaffold and simplify; 100% success → the environment is too soft, make it harder.
3. **Write.** Emit one or more components targeting the diagnosis. The candidate set is accepted or rejected as a whole.
4. **Validate.** Wrap, run $K=5$ *fresh* rollouts, judge from aggregate statistics — success rate, failure distribution, timeout count — never a single trace. Accept / reject (unsolvable or too easy) / refine. Refinement loops back to Write, at most 5 rounds.

The prompt contains a genuinely useful piece of engineering wisdom: *a mutation that makes the task impossible is not a difficulty increase. $\text{SR}=0$ from impossibility is exactly as useless as $\text{SR}=1$ from triviality.* And when refining: if SR moved toward the target band, the perturbation *type* is right — keep the code verbatim and only adjust magnitude. Do not throw away code that cost 5 rollouts to validate.

The designer LLM is the **same backbone as the policy** in every experiment. No distilling from a stronger model.

### Training setups

**Skill-based learning (main results).** Collect trajectories in the customised environments, extract skills following ReasoningBank, equip the frozen policy, evaluate on held-out original tasks. Models: Gemini-3.1-Flash-Lite (ALFWorld, WebArena), Gemini-3.5-Flash (rest).

**RL.** Qwen3-8B-base trained with [[GRPO]] on ALFWorld and WebShop. 8×H100, vLLM rollouts at temperature 0.4, FSDP with parameter/optimiser offload and gradient checkpointing. Batch 16, PPO mini-batch 256, prompt ≤4096 tokens, response ≤512, 50-step episodes, 0.1 invalid-action penalty, 150 steps.

Chain is excluded from the automated loop — EnvRigger cannot see inside joined environments well enough to diagnose — and is studied separately.

## Ablation Studies and Experiments

### Main tables

ALFWorld and WebArena, mean of three runs:

| Skill source | ALFWorld In-Dist | ALFWorld OOD | ALFWorld Avg | WebArena Avg |
|---|---|---|---|---|
| No Skills | 62.6 | 60.7 | 61.7 | 38.7 |
| Original Envs | 63.3 | 61.4 | 62.4 | 38.5 |
| GenEnv | 63.3 | 61.9 | 62.6 | — |
| VeriEnv | — | — | — | 39.6 |
| **EnvHarness** | **66.2** | **70.4** | **68.3** | **41.6** |

SWE-bench Verified, OfficeQA, SpreadsheetBench:

| Skill source | SWE SR ↑ | SWE steps ↓ | OfficeQA EM | Spreadsheet Pass@1 |
|---|---|---|---|---|
| No Skills | 47.67 | 53.58 | 54.23 | 46.44 |
| Original Envs | 49.88 | 55.01 | 54.40 | 45.88 |
| SWE-smith | 50.12 | 54.72 | — | — |
| **EnvHarness** | **52.58** | **49.61** | **56.20** | **49.15** |

### The negative result that matters most

**Skills from unmodified environments can make things worse.** On SpreadsheetBench, Original Envs scores 45.88 vs 46.44 with no skills at all. On SWE-bench, they *lengthen* episodes from 53.58 to 55.01 steps. The explanation is clean: a static environment only lets the agent practise what it already does, so the extracted skills are redundant restatements of existing behaviour, and retrieving them costs context for nothing.

EnvHarness beats no-skills on every single benchmark, which the authors attribute directly to the write-and-validate loop — nothing is committed until fresh rollouts confirm it.

### Domain-specific generators lose on their own turf

Against GenEnv on ALFWorld: +5.7 average, **+8.5 out-of-distribution**. Against SWE-smith on SWE-bench: +2.46 SR and 5.11 fewer steps. Generating more instances just buys repetitive practice; targeting a diagnosed flaw buys new behaviour.

### RL

| Training set | ALFWorld In-Dist | ALFWorld OOD | WebShop Score | WebShop SR |
|---|---|---|---|---|
| Original Envs | 81.4 | 89.6 | 75.6 | 66.0 |
| EnvHarness Envs | **87.9** | 88.8 | **79.2** | **67.4** |

Wins 3 of 4. The one loss (ALFWorld OOD, −0.8) is small. The point is that reshaped environments are a *standalone* optimisation signal, not just extra data.

### Chain, isolated

| | SR ↑ | Avg Steps ↓ |
|---|---|---|
| No Skills | 47.67 | 53.58 |
| Original Envs | 49.88 | 55.01 |
| Stage/Contract only | 52.58 | 49.61 |
| **Chain only** | 49.63 | **41.96** |
| **Combined** | **54.30** | 43.12 |

Chain alone is *below* the Original Envs baseline on success rate (49.63 vs 49.88) but slashes steps by 11.6. The conjunctive reward $R_A \wedge R_B$ trains goal preservation and budget discipline, not single-task maximisation. Combining both skill banks gets the best SR *and* near-best efficiency — the two are complementary, not overlapping.

### Scaling

Each batch of 50 environments yields one skill bank. Over 300 environments: EnvHarness 47.67 → 54.79 (+7.12) and still rising; Original 52.13; SWE-smith 50.37. The difference in protocol is the whole story — baselines draw batches independently of the learner, while EnvHarness synthesises each batch against the policy *already equipped with the previous skills*.

The appendix traces this concretely, and it is the most convincing evidence in the paper:

- **Round 1** — basic competence. Skills about running `pytest -x` and applying edits via `patch -p1`. The components forced these by rewriting pytest commands and blocking three alternative edit paths.
- **Round 2** — the shell itself breaks. Components fake `pytest: command not found`, fake OOM kills (exit 137) on whole-file runs, fake timeouts. Skills learned: `pytest.main()` from Python, `pytest -k` filters.
- **Round 3** — below the shell. Component prepends the conda env to `PATH`; component makes `sed -i` fail. Skills: absolute interpreter paths, `grep -rn` navigation before committing to an expensive patch.

Also in round 3: on roughly a third of training tasks the policy now succeeds on *every* baseline rollout, EnvRigger tries to make them harder, every candidate fails validation, and the revision budget runs out. Those tasks contribute nothing. Honest reporting of where the loop stalls.

### Cross-model

Four policies on SWE-bench, each using itself as designer:

| Model | No Skills SR / steps | Original SR / steps | EnvHarness SR / steps |
|---|---|---|---|
| Gemini 3.1 Flash-Lite | 30.7 / 36.7 | 36.8 / 50.0 | 40.0 / 50.6 |
| Qwen3.6 27B | 41.0 / 69.8 | 48.4 / 37.1 | 52.1 / 40.8 |
| Gemini 3.5 Flash | 47.7 / 53.6 | 49.9 / 55.0 | 52.6 / 49.6 |
| Claude Sonnet 4.6 | 67.2 / 29.3 | 69.2 / 25.4 | 72.4 / 25.6 |

Gain over Original Envs is 2.7–3.7 points and **independent of policy strength** across a 30.7–67.2 range. The loop neither breaks on the weakest nor saturates on the strongest.

The step counts are the more interesting half, and they show three regimes:
- Qwen flails (69.8 steps bare) — skills nearly halve it to 37–41.
- Flash-Lite *quits early* (36.7 steps bare) — skills make it persist to ~50 steps and solve far more.
- Sonnet is already directed (29.3) — skills barely move it.

**Average steps alone is not a quality metric.** A short episode is either efficiency or surrender, and those sit at nearly the same number. Read it with the success rate or not at all.

### Generalisation, leave-one-out on ALFWorld

Extract from all task types except one, evaluate on the held-out type. EnvHarness wins on 4 of 6, +3.1 average. Biggest gain `clean` +16.4. **One regression: `heat` −8.7.** Reshaped environments push the policy off memorised routines, so the skills encode transferable behaviour rather than per-type recipes — but not uniformly.

### Metric targeting

Ask for a band and the loop hits it. Per-task success rate into $[0.4, 0.6]$: coverage 6.0% → 80.0%, mean SR 0.74 → 0.48. The original tasks were bimodal (always-solved or never-solved) and got compressed to the middle. Average steps into $[25, 35]$ is harder — an exact count, not a rate — and coverage only reaches 53.0% from 18.0%.

### Cost

| | Design tokens | Rollout tokens | Total |
|---|---|---|---|
| ALFWorld GenEnv | 38K | 64.2M | 64.2M |
| ALFWorld EnvHarness | 1.46M | 226.6M | 228.0M |
| WebArena VeriEnv | 20K | 137.7M | 137.8M |
| WebArena EnvHarness | 1.58M | 135.7M | 137.3M |

Design costs ~40× more than one-pass generators, because full trajectories go into the diagnosis prompt. But design is a rounding error next to rollouts. Against VeriEnv, which also executes for real, the totals are the same (137.3M vs 137.8M) — **the gains are not bought with extra compute.** GenEnv is 3.5× cheaper only because its rollouts are simulated by an LLM rather than executed, which is the hallucination trade.

## Worth Remembering

**The design commitment is the contribution.** The paper's real claim is architectural, not empirical: environment construction becomes a *wrapping* problem rather than an *authoring* problem. Everything else follows — domain-agnosticism follows from the interface, verifier trust follows from never touching $R$, black-box operation follows from the runtime-safe state view.

**$R$ is deliberately not exposed.** The system prompt says why: success is the benchmark's own verdict, so reshaping reward cannot move the eval metric. This is a discipline worth stealing for any off-policy or counterfactual setup — if you let the optimiser touch the measure, you have built a [[Reward Hacking|reward-hacking]] machine.

**Validation is aggregate, never anecdotal.** $K=5$ rollouts, and acceptance is decided on success rate, failure distribution and timeout count. Judging a candidate from one trace is explicitly forbidden. This is ordinary [[AB Testing|experimental]] hygiene applied inside an automated loop, and it is the reason EnvHarness beats no-skills everywhere while Original Envs does not.

**The difficulty band is the curriculum.** Neither $\text{SR}=0$ nor $\text{SR}=1$ carries signal. Targeting the middle is the same principle as [[Exploration vs Exploitation|exploring where you are uncertain]] and as prioritised level replay in curriculum RL — the paper's contribution is doing it by wrapping rather than by generating levels.

### Limitations the authors state

- **`reset` is the binding constraint.** A Stage must place the world in a chosen state; a Chain must return it to a known state. This excludes live services (a sent email cannot be un-sent) and physical robots (the room does not tidy itself). Deterministic resets are an assumption, not a nicety.
- **Chain is purely sequential.** Branching and interleaving are implementable in the control flow, but only concatenation admits a composite verifier, because only then do you have two verdicts to conjoin. Chain also has no notion of whether two subtasks are semantically *related* — the pairing in the experiments is random.
- **The design loop is expensive per environment**, and a weaker designer needs more iterations. Paid once per environment, not per episode.
- **Text only.** Visual, GUI and embodied environments would need components that can specify and check states not expressible as text.

### Caveats for someone using this

- Skills from unmodified environments are a real regression risk, not a neutral baseline. Measure against no-skills, always.
- Adding a new benchmark still needs a hand-written Bridge. "Domain-agnostic" means the loop, the components and the prompts transfer — not that integration is free.
- Generated Contract code is compiled from a string in a subprocess. That containment is deliberate and you should keep it; see [[Sandboxing]].
- The ALFWorld In-Dist/OOD split is the benchmark's own seen/unseen split, not one the authors constructed.
- Every evaluation instance is attempted once. Standard deviations come from three independent runs of the whole pipeline.

### Open questions

Does the co-evolution curve actually keep climbing past 300 environments, or is the round-3 stall (a third of tasks producing nothing) the start of saturation? What happens if the designer is *stronger* than the policy — the same-backbone constraint rules out distillation but also leaves the most obvious scaling axis untested. And how would you measure semantic compatibility between two tasks well enough to make Chain non-random?

## Links

Related: [[GRPO]] · [[Reward Hacking]] · [[Exploration vs Exploitation]] · [[AB Testing]] · [[Sandboxing]] · [[Agentic Workflows]] · [[Agent Evaluation]] · [[Evals]] · [[Tool Use]] · [[Offline RL]] · [[Reinforcement Learning]] · [[Markov Decision Process]] · [[Reward Function]] · [[Credit Assignment]] · [[WHALE- A Simple Recipe for Joint Harness-Weight Optimization]] · [[RRSI- Regularized Recursive Self-Improvement of Agent Harnesses]] · [[An Empirical Study of Harness Design for Coding Agents]] · [[AgentMercury- Your Agent Can Synthesize Verifiable Environments for Business Scenarios at scale]] · [[Verifiable Hidden Dynamics Play- Generating Agentic RL Environments from Solved Mechanisms]] · [[Harness-Zero- Harness Distillation via Agent-as-Harness]] · [[Recursive self-improvement of AI research agents]] · [[ScienceIDE- Turning World's Scientific Codebase into Agent Learnable Environments]] · [[Imitation Learning]] · [[Regret]]

New topics worth writing: Unsupervised Environment Design (PAIRED, POET), Prioritized Level Replay, automatic curriculum learning, ReasoningBank and skill extraction from trajectories, Gymnasium environment interface design, decorator pattern for RL environments, SWE-bench and agentic coding benchmarks, environment scaling for agent training
