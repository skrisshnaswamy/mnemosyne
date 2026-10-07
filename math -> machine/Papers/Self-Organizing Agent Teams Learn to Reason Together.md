---
title: "Self-Organizing Agent Teams Learn to Reason Together"
authors: ["Pappu et al."]
year: 2026
arxiv: "2609.22682"
url: https://arxiv.org/abs/2609.22682
priority: Good-To-Read
read_on: 2026-09-28
tags: [paper, llm, vision]
---
## The Core Idea

Take three different LLMs. Don't fine-tune them. Don't hand them a workflow. Instead, let them **learn how to hold a conversation with each other**, and reuse that conversational structure on problems they have never seen.

That is the whole paper. The learned artefact is not a prompt, not a graph of tool calls, not a task decomposition. It is a *teamwork strategy*: who speaks in which phase, what persistent role each model carries, who sees whose output, and how the final answer is assembled.

Why this did not exist before: nearly every prior multi-agent method fixes the **unit of work** in advance.

- Debate and Mixture-of-Agents take *candidate answers* as the unit. Each agent writes a full solution, then they vote, revise, or get merged. Prior work showed most of debate's apparent gain is recoverable by just voting over the first-round answers — it is [[Chain of Thought|CoT]] plus an ensemble.
- Workflow/topology search (GPTSwarm, AFlow, Conductor) takes *subtasks* as the unit. It assumes the problem splits cleanly and you know how.

Both break when **no member has a solution and nobody knows the right split**. Then the useful division of labour has to emerge *inside* the conversation: one model supplies the right invariant, a second fixes its arithmetic, a third audits the fix.

The paper calls that **collaborative computation**, and its second contribution is the measurement that makes the claim falsifiable.

> [!NOTE] Routing oracle
> A hypothetical perfect per-problem selector over the team members' *independent* answers. If any single member got it right alone, the oracle gets it right. It is the computational version of the "truth-wins" condition from organizational psychology. Beating the best member proves nothing — different members may simply solve different problems. Beating the **routing oracle** is the only way to show interaction produced an answer that was not lying around in somebody's independent output. ^routing-oracle

The math team beats the oracle: **66.7% vs 59.0%** averaged over five benchmarks, and by **13.4 points** on AIME 2026. That is the headline result. The knowledge team does not, and the paper is honest about it — which sets up the third contribution, below.

> [!NOTE] Collaborative computation
> Agents exchanging, challenging, repairing and synthesising *partial* reasoning into a solution none of them produced alone — as opposed to aggregating or selecting among complete candidate answers. ^collaborative-computation

## The Methodology

**A tiny language for teamwork.** A strategy is $P = (S, \tau, \alpha)$:

- $S = [s_1,\dots,s_K]$ — an ordered list of conversational phases.
- $\tau$ — one shared *teamwork prompt*, the team's norms (e.g. "consensus is not sufficient").
- $\alpha = \{\alpha_i\}$ — persistent per-agent *role prompts* that hold across every phase ("you are the final auditor").

Each phase is

$$s_k = (A_k,\; r_k,\; f_k,\; \pi_k,\; \rho_k)$$

where $A_k \subseteq \mathcal{A}$ is who participates, $r_k$ is how many rounds, $f_k \in \{\mathrm{L},\mathrm{S}\}$ is the information-flow mode, $\pi_k$ is a shared step prompt, and $\rho_k$ are optional per-agent step prompts.

The two flow modes are the interesting bit:
- **Local** ($\mathrm{L}$): only the participants of that phase see the turns.
- **Summary** ($\mathrm{S}$): the exchange stays local while it runs, then one randomly chosen participant summarises the key points and current answer position, and *that summary* goes into everyone's context.

So the search can build a phase where two models verify a third model's disputed derivation **with the third model excluded from the room**. That is an organisational move no fixed debate protocol can express. Compare the fixed wiring in [[Multi-Agent LLM Systems]] and [[Agentic Workflows]] — the point here is that the wiring itself is the learned object.

**Learning the bank (all offline).**

1. Seed with $P_{\mathrm{init}}$: independent solutions, two debate-like rounds, majority vote. This is basically standard debate.
2. Per training problem, keep an archive of candidate strategies plus full transcripts and outcomes.
3. The roster member with the highest *training* accuracy does **teamwork reflection**: it reads prior strategies, transcripts, per-member answers, and probe results, picks a parent to build on, and proposes targeted mutations to roles/phases/synthesis rules. 6 mutation rounds per problem, up to 3 candidates per round.
4. Any mutation that solves its source problem is frozen and run on **5 other training problems** — the *validation probes*. This is the transfer signal.
5. A separate instance of the same model runs a **leakage audit** on every field: reject anything encoding an answer value, a problem-specific fact, or a source-derived solution recipe.
6. Coverage-greedy selection down to a bank of **10 complementary strategies**. Frozen. No further learning, ever.

This is the same family of move as [[DSPy]] and GEPA — optimise a non-weight artefact by reflective search — but the artefact is a conversation structure rather than a prompt.

**Test time.** Run all 10 strategies on the new problem. Each produces a *certificate*: a self-contained reasoning trace meant to be checked step by step, not a bare answer. One judge (again the highest-training-accuracy member) sees the problem and **all 10 certificates in one prompt**, is explicitly told *not to solve the problem*, may only downgrade a certificate by naming a concrete local defect, and must fall back to answer frequency among defect-free certificates as a mandatory tie-break. Full prompt is in the appendix and is worth stealing.

Two metrics, and the gap between them is the paper's second half:
- **Team coverage** — fraction of problems where *at least one* of the 10 certificates is correct.
- **Team accuracy** — fraction correct after the judge picks.

**The two setups.**

| | Math & physics | Knowledge & logic |
|---|---|---|
| Members | o3-mini, Claude Sonnet 4, DeepSeek-V3 | Gemini-2.5-Flash, Llama-4-Maverick, GPT-4.1 |
| Reflector + judge | o3-mini | Gemini-2.5-Flash |
| Training set | **15** AIME-2024 problems | **25** GPQA Diamond problems |
| Transfer to | AIME 25/26, HMMT Feb 26, TheoremQA-physics | MMLU-Pro, BBEH (5 logic subtasks) |

Models were deliberately chosen to be mid-tier so the benchmarks are not saturated.

## Ablation Studies and Experiments

**Math & physics (accuracy %, mean of 3 seeds):**

| Method | AIME24 | AIME25 | AIME26 | HMMT26 | TQA-phys | Avg |
|---|---|---|---|---|---|---|
| Best member | 64.4 | 40.0 | 42.2 | 26.3 | 71.1 | 48.8 |
| Self-consistency $K{=}10$ | 75.8 | 44.4 | 53.4 | 31.9 | 71.5 | 55.4 |
| Self-reflection (5 passes) | 71.1 | 41.1 | 52.2 | 29.3 | 72.2 | 53.2 |
| **Linearization** (o3-mini alone runs every role/phase) | 74.0 | 43.3 | 68.5 | 39.4 | 68.4 | 58.7 |
| Debate (3 rounds) | 66.7 | 41.1 | 53.3 | 30.3 | 72.2 | 52.7 |
| Mixture of Agents | 75.6 | 46.7 | 57.8 | 32.3 | 74.0 | 57.3 |
| **Homogeneous team** (3× o3-mini, same strategies) | 66.7 | 46.7 | 60.0 | 36.4 | 70.2 | 56.0 |
| **SAT** | **84.7** | **60.8** | **71.2** | **39.4** | **77.2** | **66.7** |
| *Routing-oracle coverage* | 73.3 | 51.1 | 57.8 | 36.4 | 76.6 | 59.0 |
| *SAT coverage* | 93.3 | 73.3 | 73.3 | 54.5 | 81.6 | 75.2 |

The two controls that actually matter:

- **Linearization** is the compute-matched single-agent control: the strongest member serially plays every role and phase of each learned strategy at roughly the team's total token budget. It reaches 58.7 vs SAT's 66.7. So the gain is not "the structure makes o3-mini think longer". On HMMT26 it **ties** SAT exactly (39.4 both) — on the hardest benchmark the multi-model part buys nothing.
- **Homogeneous team** is three copies of o3-mini running the identical frozen strategies: 56.0. So the *heterogeneity* is load-bearing, not just the structure. Learned organisation + diverse models is the combination; either alone underperforms.
- **The seed** $P_{\mathrm{init}}$ (debate + majority vote) gets 54.1 avg on math, 69.6 on knowledge. Reflection is worth roughly +12.6 and +3.2 points over its own starting point.

**Knowledge & logic (%):**

| Method | GPQA | MMLU-Pro | BBEH | Avg |
|---|---|---|---|---|
| Best member | 72.7 | 79.7 | 45.3 | 65.9 |
| Linearization (Gemini-2.5-Flash) | 79.0 | 84.0 | 53.3 | 72.1 |
| Debate | 73.7 | 79.7 | **57.3** | 70.2 |
| Mixture of Agents | 72.0 | **84.3** | **57.3** | 71.2 |
| Homogeneous team | 78.0 | 76.0 | **58.7** | 70.9 |
| **SAT** | **80.0** | 82.4 | 56.0 | **72.8** |
| *Routing-oracle coverage* | 84.0 | 88.0 | 66.7 | 79.6 |
| *SAT coverage* | 94.0 | 91.0 | 78.7 | 87.9 |

**What did not work — and this is the useful half.**

1. **SAT loses several individual columns.** MoA beats it on MMLU-Pro (84.3 vs 82.4). The homogeneous team beats it on BBEH (58.7 vs 56.0). SAT only wins on the *average*. The 0.7-point margin over linearization on this suite is not a result you should lean on.
2. **Coverage far exceeds accuracy on knowledge & logic**: 87.9% coverage, 72.8% accuracy. The team *generates* a correct certificate on 87.9% of problems, beats the routing oracle's coverage on all three benchmarks, and then **throws it away at selection**. Generating and recognising are two separate problems, and the judge is the bottleneck.
3. **Demonstrability is the whole story of when this helps.** They measure it as *team-certificate discriminability*: pair one correct and one wrong team certificate, strip labels, show both in balanced A/B order to a **ten-model panel drawn from outside both rosters**, and record how often the correct one wins. Across the eight benchmarks this tracks improvement-over-best-member at Spearman $\rho = 0.90$, exact permutation $p = 0.005$, robust to leaving out any single benchmark ($\rho = 0.86$–$0.96$).

> [!NOTE] Demonstrability
> From organizational psychology (Laughlin & Ellis, 1986): whether a team can tell correct reasoning from incorrect reasoning once it is on the table. Not a formal verifier — a soft, continuous, benchmark-level proxy for verifiability. AIME 2026 scores 0.687 and SAT gains +29.0 points there; TheoremQA-physics scores 0.423 and SAT gains +6.1. ^demonstrability

The mechanism it implies: collaboration only pays when a correct insight, once produced, can **survive challenge and redirect the rest of the conversation**. When correct reasoning is hard to recognise, plausible errors crowd it out — the same failure mode as [[Reflection#Where the signal can come from, ranked|reflection without a real signal]].

**The qualitative evidence.** On HMMT 2026 problem 6, all three members answer wrongly alone. o3-mini supplies the gcd–lcm exponent invariant and the factorisation but counts **five** unit-exponent primes instead of six. DeepSeek catches it: $\tau(N) = 5\cdot4\cdot3\cdot2^6 = 3840$. Sonnet independently re-checks the $p=2,3,5$ counts. The team returns 3840, which appeared in none of the three initial answers. On a GPQA chemistry problem, debate turns a wrong majority into *unanimous* wrongness by displacing GPT-4.1's correct answer, MoA propagates a bad pathway identification, and SAT's claim-audit phase repairs the atom tracking and lands on the right answer from three wrong starting points.

Two learned strategies worth knowing by name:
- `problem_adaptive_method_diversification` — members propose distinct solution frames *for this problem*, get assigned different frames, then compare and repair. The structure transfers; the decomposition is emergent.
- `divergence_reconciliation` — encodes a behavioural observation about DeepSeek: it surfaces useful alternative derivations but also introduces errors. So it stays in as a diversity source, and a later phase has o3-mini and Claude verify its disputed steps **with DeepSeek excluded**. This is a learned model-specific comparative advantage, which is the most striking thing in the appendix.

## Worth Remembering

- **15 problems.** That is the entire math training set. The compute goes into search (6 rounds × up to 3 mutations × 15 problems, each executed by the full team, plus 5 validation probes per survivor), not into data. Serving cost is 10 multi-phase conversations per test problem plus one judge call — expensive, and only the linearization baseline is compute-matched.
- **Two guards against leakage, both soft.** A semantic source-dependence audit by a same-model instance, and the validation probes (a strategy that only helps its source problem adds no coverage and dies in greedy selection). Neither is a proof; the judge's prompt telling it not to solve the problem is also just a prompt. Worth treating the AIME-2024 in-distribution number with more suspicion than the transfer numbers.
- **The routing oracle is the reusable idea**, independent of this paper. If you evaluate any ensemble of models, ask: could a perfect per-item selector over the members' independent outputs have done this? Most reported multi-agent gains would not survive that test. Same discipline as [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|baseline scepticism]] and [[On the Difficulty of Evaluating Baselines]].
- **Judge-limited, and the authors say so.** The obvious next move — optimise the strategies *for demonstrability* rather than for accuracy, so correct reasoning becomes easier to recognise — is left as future work. Demonstrability here is measured post hoc and never feeds the search, so the $\rho=0.90$ is correlational across eight points.
- **Certificates are a design choice, not an incidental output.** Forcing each strategy to emit a step-checkable trace is what makes single-judge selection possible at all. The same idea shows up in [[Evals#LLM-as-judge, done properly ⚖️|LLM-as-judge]] and [[Grounding]]: judge the written support, not your own answer.
- The framing echo of [[The Bitter Lesson (essay)]] is inverted here: this is not more compute or more parameters, it is a learned *organisational* prior that survives transfer across competition years and into physics. Whether that survives stronger base models is untested — they picked mid-tier models precisely because strong ones saturate these benchmarks.

## Links
Related: [[Multi-Agent LLM Systems]] · [[Agentic Workflows]] · [[DSPy]] · [[Evals]] · [[Reflection]] · [[Chain of Thought]] · [[Test-Time Compute]] · [[Planning and Decomposition]] · [[Agent Evaluation]] · [[Prompt Engineering]] · [[Context Engineering]] · [[Agora- Git as Shared Memory for Collective AutoResearch]] · [[Autonomous Mathematical Discovery in an Open-World Multi-Agent Environment]] · [[Recursive self-improvement of AI research agents]] · [[WHALE- A Simple Recipe for Joint Harness-Weight Optimization]] · [[An Empirical Study of Harness Design for Coding Agents]] · [[Using Grounded Theory for Agent Behavior Analysis at Scale]] · [[On the Difficulty of Evaluating Baselines]]

New topics worth writing: routing oracle and the truth-wins condition, demonstrability as a task property, coverage-vs-accuracy gap in generate-then-select systems, Mixture-of-Agents, multi-agent debate and its critiques, GEPA reflective prompt evolution, self-consistency sampling, transactive memory in teams
