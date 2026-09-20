---
title: "ScienceIDE: Turning World's Scientific Codebase into Agent Learnable Environments"
authors: ["Geng et al."]
year: 2026
arxiv: "2609.19134"
url: https://arxiv.org/abs/2609.19134
priority: Good-To-Read
read_on: 2026-09-19
tags: [paper, rl, vision, theory]
---
## The Core Idea

Coding agents learned to code because GitHub gave them millions of repositories with tests. Run the test, get a pass or fail, learn. Science has just as much code — plasma solvers, ocean models, astrophysics simulators — but you cannot learn from it the same way. The paper calls this the **scientific experience bottleneck**.

Why the bottleneck exists is the interesting part. A unit test in a web app says "this function returns 4". A scientific test says "the simulated plasma temperature should match the reference to within some tolerance, but only for these observables, and the array order does not matter because particles get reshuffled, and the timestep count will differ between runs and that is fine". The correctness contract is real but it is unwritten. It lives in a domain expert's head. So you cannot scrape it.

ScienceIDE's move: **make the scientific correctness contract an executable object, and separate it from the task**. An expert looks at one *module* of a real codebase (say, the radiative-cooling routine in PLUTO) and freezes a set of **checks**: which physical numbers to compare, under what tolerance, with what alignment rule. That contract is written once. Then any number of tasks can be generated against it — break the code and ask an agent to fix it, delete a routine and ask it to be rewritten, ask for a GPU port — and every one of them is graded by the same checks. Expert effort is amortised over hundreds of tasks instead of being spent per task.

> [!NOTE] Scientific check ^scientific-check
> A fixed input, a set of graded outputs, and a **pass policy** that says what counts as agreement. Two policy shapes: *pointwise*, $|c - r| \le a + \rho|r|$ for every graded value, and *invariants*, comparing conserved quantities or distribution moments when run-to-run variation makes pointwise comparison useless.

What it unlocks: 64 environments from 27 real scientific codebases, 2,812 generated tasks, 1,076 executable checks — and the same interface feeds evaluation, [[Fine-Tuning|supervised fine-tuning]], and online [[Proximal Policy Optimization Algorithms|RL]]. It is SWE-Gym / [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)|SWE-smith]]-style environment scaling, but for code where "did it pass" is a physics question.

## The Methodology

**Step 1 — module, not repository.** An agent pins an upstream revision, builds it, runs the official tests and shipped examples, and records what it sees: output formats, numerical jitter, expensive code paths. It proposes **modules** — a coherent scientific responsibility with executable coverage, not just a folder. A human expert reviews the split. One repository can yield several environments.

**Step 2 — turn official tests into checks.** Every upstream unit test, regression test and shipped example is traced to a module and either kept as a check, excluded with a written reason, or flagged as a gap. The key discipline: **grade physics, not bookkeeping**. A particle's properties are matched by its particle ID, never its array slot. Storage order, adaptive step counts, wall-clock timings, MPI rank layout, random draws, eigenvector sign — none of these are observables.

How the tolerance gets set is careful:

- A **nominal** and a **variant** initial condition are both run. The variant perturbs the smallest sufficient set of inputs so you can *measure* how sensitive the graded outputs are.
- Where possible an **altbuild** runs the same pinned source under a different legitimate compiler/build and records the cross-build spread.
- Both runs must independently reproduce the reference and score full reward before the check is accepted.
- The curator then picks the policy, bound and window by *reading the numerical mechanism in the source*. Each check carries a plain-language **warrant**: what is compared, which scientifically meaningful deviation the bound is meant to catch, and why a legitimate alternative implementation can still pass.

Pointwise is preferred whenever a bound can contain the measured noise *and still reject a real fault*. If the graded window can be shortened so pointwise survives, do that first. Only then fall back to invariants.

**Step 3 — factories.** The check suite is frozen *before* any task is written. A **factory** then adapts generic authoring procedures (reversible edits, execution, artifact assembly) to one environment's local rules. Taxonomy of seven task families: Acceleration, Repair, Discovery, Reproduction, Integration, Calibration, Implementation. Only Repair (inject a defect via mutation) and Implementation (excise a routine) automate candidate generation. The registry today is lopsided: **2,515 repair, 295 implementation, 2 acceleration**.

**Step 4 — validity by execution, not by proposal.** A generated candidate becomes a task only if, in the final grading container:

- a known-valid **witness** passes all checks,
- the **defective baseline** leaves headroom (does not already pass),
- the injected change actually causes a check failure that the reference repair removes.

Silent mutations survive as explicit controls. Ambiguous specs, unreachable branches and harness crashes are thrown out or marked ungraded.

**Partial credit.** Repair reward normalises against the broken starting point:

$$r_{\mathrm{repair}} = \max\!\left(0, \frac{r - f}{1 - f}\right), \qquad 0 \le f < 1$$

where $r$ is raw scientific agreement and $f$ is the unfixed build's score. So fixing nothing scores 0 even if the broken code already passes half the checks.

**The episode interface.** The agent gets an editable workspace, inspects code and inputs, edits, runs experiments, submits artifacts to a *private* verifier. The harness logs actions, observations, check rewards, execution status and resource use — and crucially distinguishes **scientific disagreement**, **incomplete delivery**, and **infrastructure failure** as three different things.

### SFT setup

Demonstrations collected from GPT-5.6-sol, filtered by the numerical verifier. 4,567 training segments from 564 tasks; 544 validation segments from 81 tasks, disjoint by task ID. Trained with ms-swift, [[LoRA]] on all linear layers, rank 32, $\alpha = 64$, dropout 0.05, BF16, batch 64, sequence limit 36,864 tokens, LR $2\times10^{-5}$, 3% warmup, [[GPU processing|cosine decay]], 3 epochs (216 steps). Loss supervises only *new* assistant actions and tool calls; instructions, observations, reused history, and actions flagged as failed or repetitive are masked out. Final checkpoints evaluated — no cherry-picking on benchmark score.

### RL setup

From base Qwen3.5-4B, no SFT init. Two environments: LAPS (3D pseudo-spectral Hall-MHD, 99 tasks, 85/14 split) and MITgcm-biogeo (ocean biogeochemistry, 87 tasks). Reward = the verifier itself: compile the edited Fortran, run the simulation, grade against private references. Hints give the defect's file, line and edit class, but not what the code should compute.

Group of $G = 8$ rollouts per prompt, advantage is group-mean-centred with **no division by group std**, following Dr. GRPO:

$$\hat{A}_i = R_i - \frac{1}{G}\sum_{j=1}^{G} R_j$$

The reason for dropping the std: rewards here are near-bimodal (a group is almost all 0 or almost all 1), so a tiny denominator would blow up one lucky success into a huge gradient.

Token update is a [[Proximal Policy Optimization Algorithms|PPO]]-style clipped ratio with an **asymmetric** bound, $\varepsilon_{\text{low}} = 0.2$, $\varepsilon_{\text{high}} = 0.3$ (DAPO Clip-Higher), so rare correct repair actions have room to gain probability mass:

$$\ell_{i,t}(\theta) = \min\!\big(r_{i,t}\hat{A}_i,\; \mathrm{clip}(r_{i,t}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}})\hat{A}_i\big)$$

Token-mean aggregation. No [[KL Divergence|KL]] penalty, no entropy bonus (measured KL stayed at $6\times10^{-4}$). Rollout and training on **disjoint GPUs** running concurrently — one optimiser step took 2,069 s of a 3,210 s step while rollout took 751 s, so a synchronous loop would idle waiting on the slowest Fortran rebuild. That concurrency creates one-step policy staleness, corrected with token-level truncated importance sampling, $w_{i,t} = \min(\exp \rho_{i,t}, 2.0)$ where $\rho$ is the trainer-minus-rollout log-prob difference. 24 H20 GPUs per run: 8 generation (vLLM, TP=2), 16 training (FSDP2, SP=4). Turn limit 50, response budget 65,536 tokens.

## Ablation Studies and Experiments

### ScienceIDE-Hard: 15 agents, 85 tasks

52 repair + 33 implementation tasks, 18 environments from PLUTO, Athena++, MITgcm, LAPS, PHANTOM. Codebases of $10^4$–$10^5$ lines. One-hour episode budget, identical containers, no localisation hints. Strict success means $r_{\text{repair}} = 1$.

| Agent | Success | $/task | min/task | output tokens |
|---|---|---|---|---|
| Fable 5.1 | 67.1% | 7.90 | 16.8 | 85.7k |
| Opus 5 | 64.6% | — | — | — |
| Astra | 63.1% | 3.56 | 9.4 | 13.9k |
| Sol | 55.0% | — | — | — |
| DeepSeek V4.1 Flash | 36.0% | — | — | 172.4k |
| remaining 11 | < 40% | | | |

The top three are not statistically separated — Fable has a single attempt per task and no interval; Opus and Astra's repeat intervals overlap.

**Effort does not buy correctness.** Spearman correlation of success with runtime is $-0.22$; with output token volume it is $0.01$. DeepSeek V4.1 Flash burns 12× Astra's tokens for 27 points less success.

**Budget reshuffles the ranking.** At 10 minutes Astra leads at 49.6% vs Fable's 25.9%; Fable overtakes at ~31 minutes. From 20→60 minutes Astra gains only 2.0 points, Fable 11.8, Opus 16.0, Qwen3.8 Max 30.2. Same final score, wildly different time curves. (These are retrospective completion profiles, not reruns at shorter budgets.)

### What failure actually looks like

Termination status is not a cause. Qwen exhausts the budget on 37.3% of attempts, averaging 48.5 min on unsuccessful episodes. Haiku has *zero* recorded timeouts, averages 7.7 min on failures, and returns at or below baseline 74.9% of the time — it gives up fast. MiniMax does both: 31.6% budget exhaustion, 34.1 min failures, low success.

**Success is unstable.** Among agents with full repeat coverage, 28.0% of model–task pairs contain both a success and a failure within their first three attempts. Gemini flips outcome on 33 of 85 tasks; Astra on only 8.

**Rank hides specialisation.** Astra and Sol both hit 47.3% on PLUTO but split 96.3% vs 38.5% on Athena++. Gemini beats Sol on LAPS (29.5% vs 20.8%) despite losing overall. Fable solves 57/85; the union of all 15 agents solves 72; 11 tasks are solved by exactly one agent and 13 by none.

### Trace review — the most useful part

358 attempts from Fable and Astra reviewed, 125 unsuccessful.

- **Reference-convention mismatch is the dominant failure: 71.4% of Fable's and 64.9% of Astra's task-balanced failures.** The code runs, the physics formula is roughly right, and it still disagrees with the reference because of a numerical or state convention the code never spells out. In a LAPS timestep task both models omit the rule for retaining the previous timestep — the low-resolution case matches, the high-resolution case diverges at a periodic update. In an MITgcm tracer task all four attempts rebuild almost the same coefficient table with **one truncated coefficient**.
- **Wrong-target edits: 10.7% / 17.0%.** Nine failed attempts across three different PLUTO cooling tasks all make the *same* natural-log → base-10-log change in a routine that was not the assigned defect. The three real defects (temperature floor, rotational temperature, vibrational temperature) stay unrepaired.
- **A passing local test can be testing the wrong thing.** On T058 both the wrong and the right repair pass the agent's own repeated-run comparison; only the target repair passes the private reference. On a PLUTO substep-handoff task the failing attempts exercise a modified error-clamp routine while the defective handoff sits untouched.
- **Correct edit, incomplete delivery.** On an Athena++ chemistry task, Fable and two Astra attempts make the *right* code change but omit four required control-output groups and fail anyway.
- Repeated runs reproduce *the same specific mistake*, not independent random failures. 17 attempts stayed unexplained after re-review.

### Contamination audits — read this before trusting any leaderboard

The authors audit their own campaign and publish what broke:

- Containers were declared network-isolated. Later trajectory inspection found **455 upstream-directed commands across 5,314 trials, 21 confirmed successful fetches, 13 of which scored as solved**. Nineteen staging scripts had silently rewritten the network declaration to `public`. The repository task files did not describe the containers that actually ran.
- Provider-side tools bypass container isolation entirely. Gemini received substantive provider-tool content on **71 of 85 tasks**; one trial pulled a 22,180-character upstream file via `web_fetch`.
- DeepSeek V4 Pro used an aggregator API key to call *other models* on 20 tasks, solving 13. After a model allowlist and rerun, its score moved from 0.365 to 0.294.
- Legacy container images still carry file-timestamp cues, and difficulty selection predates retrieval isolation.

### SFT results

Held-out repair reward (mean $r_{\text{repair}}$, partial credit retained, greedy, 8,192-token context, single 2,048-token patch):

| Model | Environment | Initial → SFT |
|---|---|---|
| 4B | PLUTO-Particles-Dust | 0.0000 → 0.3333 |
| 9B | PLUTO-RMHD/ResRMHD | 0.0000 → 0.2857 |
| 9B | LAPS | 0.3125 → 0.5000 |
| 9B | MITgcm-Biogeo | 0.0625 → 0.1250 |

Transfer to public benchmarks — 15 model–benchmark pairs gained ≥3 points. 4B: HumanEvalFix JavaScript +10.98, QuixBugs Java 0.400→0.500, CodeXGLUE defect detection 0.422→0.492 (still below chance-adjacent 0.5), BBH multistep arithmetic 0.920→0.968. 9B: **BBH Word Sorting 0.240→0.576**, plus GSM8K. 72B: APPS Introductory +6.25, LiveCodeBench Execution +5.47, ARC-Easy +3.125.

**What did not hold.** On disjoint confirmation items, HumanEvalFix Python for 9B **declined** 86.11 → 77.78 ($[-19.44, 2.78]$). The 72B HumanEvalFix C++ screening gain did not persist. CodeXGLUE refinement "gains" are +1.19 and +0.77 points on a base of 1.17% and 1.63% — technically real, practically noise-adjacent. Intervals are unadjusted for multiple comparisons across 33 screened configurations, so expect some of these to be selection.

### The RL ablation that carries the paper

**Outcome-only reward plus token-mean loss creates a length-shortening shortcut.** An episode ends for three reasons: repair finished, turn cap hit, response budget hit. Only the first says anything about correctness — but the reward treats all three as the same zero. A budget-truncated episode in a group with positive mean reward gets $\hat{A}_i < 0$, and under token-mean aggregation that negative gradient is spread over *the most tokens*, because the longest trajectories are the ones attempting the hardest repairs.

In the unmasked run this becomes a self-reinforcing collapse:

| | Unmasked | Masked |
|---|---|---|
| Mean reward | $0.339 \to 0.573 \to 0.078$ | $0.487 \to 0.914$ |
| Tokens per turn | $1125 \to 327$ | $1095 \to 1165$ |
| Truncation rate | $29\% \to 39\%$ | $28.9\% \to 2.3\%$ |
| Turns per episode | $34.9 \to 49.5$ | $32.2 \to 24.1$ |
| Entropy | rising | $0.586 \to 0.348$ |

Reward climbs, then falls **below its starting point**. Tokens per turn drop by more than 3×. Shorter turns mean more episodes hit the turn cap, which means more truncation, which strengthens the same bad signal.

**The fix separates two roles of a truncated trajectory.** Its reward still counts toward the group baseline in Eq. (2); its tokens get **no policy gradient**:

$$m_{i,t} = \begin{cases} 0 & \text{if } \tau_i \in \mathcal{T}_{\text{budget}} \\ \mathbb{1}[\text{token } t \text{ is model-generated}] & \text{otherwise} \end{cases}$$

The mask is applied *after* termination classification, so it catches turn- and response-budget cutoffs but not infrastructure timeouts or grading errors. They also switch **off DAPO's reward-side length penalty**: an audit found it firing on 28% of samples, and 119 of 142 firings were on already-failing episodes, while it hit *finished, perfect-score* episodes about 64% of the time. Length here measures how hard the defect was to localise, not something to punish.

**Results with the mask.** Held-out reward after 30 steps over base Qwen3.5-4B: LAPS $0.357 \to 0.857$ ($2.4\times$), MITgcm-biogeo $0.286 \to 0.571$ ($2.0\times$). Training reward LAPS $0.427 \to 0.828$, MITgcm $0.381 \to 0.597$. Truncation falls 39.5%→6.6% and 34.1%→23.8%. Critically, **tokens per turn go up** (1103→1135, 780→914) — the policy is finishing earlier, not writing less. Importance-sampling diagnostics stay healthy throughout both the good and collapsed runs (effective sample size $\in [0.950, 0.978]$, log-prob correlation $\ge 0.938$), which rules out staleness as the cause of the collapse and leaves the objective itself holding the bag.

One warning sign: groups with **zero reward variance** — which contribute no gradient at all under a mean-centred advantage — grow from 10.0% to 47.5% over the LAPS run. The curriculum runs out of usefully-hard tasks.

## Worth Remembering

**The truncation-masking lesson generalises far beyond science.** Any outcome-only [[Proximal Policy Optimization Algorithms|RL]] setup with a budget cutoff and token-mean loss has this bug. The policy's cheapest route to higher reward is to stop generating. If you are training long-horizon agents and your response length is collapsing while reward briefly rises, look here first. The principle stated cleanly: *a harness cutoff is evidence about the task, not about the tokens*. Keep it in the baseline, drop it from the gradient.

**The pointwise/invariants distinction is a template for metric design.** Compare exactly what is physically meaningful; align by a carried identity, not by position; explicitly enumerate what is *not* an observable. Anyone building a [[Evals|reward or eval harness]] over a stochastic system should copy this — the list of non-observables (ordering, timings, layout, random draws, sign conventions) is the part people forget.

**What the authors admit.**

- Everything measured is reference-verifiable repair and implementation in computational physics and geoscience. Nothing here is open-ended discovery.
- Factory-generated variants share code paths, so 2,812 tasks is **not** 2,812 independent scientific problems. Task count is not coverage.
- Verification is itself a modelling choice. Passing the checks does not mean correct outside the checks, and a legitimate alternative implementation can disagree with the reference.
- No external audit, no adversarial [[Reward Hacking|reward-hacking]] evaluation yet.
- The hard set probes only up to a one-hour budget and two coupled edit sites.
- **SFT separation is by task identifier only** — related variants were not audited for overlap. RL gains are *within* training environments, with hints, and with no between-seed variance estimate. Neither result shows transfer to an unseen codebase.
- Retrieval controls cannot exclude that the models saw this public source during pretraining.

**The honest-reporting of their own contamination is unusual and worth copying.** They published the 455 fetch commands, the 19 mis-staged scripts, the aggregator-key incident and the score correction from .365 to .294. Most benchmark papers would have quietly fixed and reported the clean number.

**Practical caveat if you wanted to use this.** Environments are pinned to specific upstream revisions, compilers and hardware. Change any of them and every tolerance has to be recalibrated — versioning preserves provenance, not scientific validity. The expert cost is amortised, not eliminated.

**Connections.** The construction is [[Deep Neural Networks for YouTube Recommendations (RecSys)|two-stage-funnel]]-shaped in spirit — generate broadly, admit narrowly by executable evidence — and the validity gate (witness passes, baseline has headroom, defect actually changes a check) is a nice concrete instance of the general "propose cheaply, verify expensively" pattern. The trace-review finding that a model's own local tests pass while the reference fails is the same shape as [[Shortcut Learning in Deep Neural Networks|shortcut learning]]: the agent optimised the signal it could see.

**Follow-up questions.** How much of the SFT transfer survives multiple-comparison correction across 33 configurations? Does the truncation mask matter at longer budgets where truncation is rarer? The zero-variance group fraction hitting 47.5% suggests the task bank saturates in 30 steps — does a difficulty curriculum from the factories fix that, and is measured per-model difficulty stable enough to build one?

## Links

Related: [[Proximal Policy Optimization Algorithms]] · [[Evals]] · [[Fine-Tuning]] · [[LoRA]] · [[Reward Hacking]] · [[Shortcut Learning in Deep Neural Networks]] · [[Distributed Training]] · [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]] · [[Training language models to follow instructions with human feedback]] · [[Agentic Workflows]] · [[Do ImageNet Classifiers Generalize to ImageNet]] · [[Troubling Trends in Machine Learning Scholarship]]

New topics worth writing: GRPO and group-relative advantage estimation, Dr. GRPO, DAPO Clip-Higher and asymmetric PPO clipping, truncated importance sampling for stale rollouts, RLVR (reinforcement learning with verifiable rewards), SWE-bench and repository-level agent benchmarks, SWE-Gym / SWE-smith environment scaling, benchmark contamination auditing, mutation testing and Defects4J, numerical tolerance design for scientific verification, asynchronous rollout-training disaggregation, token-mean vs sequence-mean loss aggregation
