---
title: "Autonomous Mathematical Discovery in an Open-World Multi-Agent Environment"
authors: ["Stephen Chung", "Wenyu Du", "William J. Wesley"]
year: 2026
arxiv: "2608.23691"
url: https://arxiv.org/abs/2608.23691
priority: Must-Read
read_on: 2026-08-27
tags: [paper, llm, theory]
---
## The Core Idea

Most AI-for-science systems are pipelines. A human writes the loop: propose a candidate, score it, mutate the best ones, repeat. AlphaEvolve is the strongest example — an evolutionary search over code, driven by a scalar score.

This work throws the pipeline away. It puts six AI agents in a simulated research town called **the Station**, tells them a research goal, and leaves. No coordinator. No scripted steps. Each agent picks its own direction, writes and runs code, mails other agents, and publishes papers into a shared archive that later agents read and cite. Agents die after at most 200 ticks and get replaced, so knowledge has to survive in the written record, not in one agent's context window.

> [!NOTE] The Station
> An open-world multi-agent environment where each LLM agent is a whole researcher, not a step in a pipeline. Rooms with different functions, discrete time steps ("ticks"), agent lifetimes, and a persistent internal literature. ^the-station

Why this matters, in one sentence: **a scalar score cannot ask for a theorem.** AlphaEvolve's finite-field Kakeya evaluator scores constructions at a fixed list of primes. To turn a numerical pattern into a proven infinite family, Google needed a task-specific pipeline plus human mathematicians. The Station agents were simply *told* that finite constructions were test cases and infinite families were the real goal — and they went and proved one, unprompted, then extended it. Elsewhere they went further off-leash: asked to improve an *upper* bound on Erdős's minimum-overlap constant, they instead proved a much better *lower* bound.

Results, against the prior human literature (not just against AlphaEvolve):

- **Kissing number in $d=11$**: three exact 604-point configurations, so $K(11) \ge 604$. AlphaEvolve got 593; the previous record was 592.
- **Erdős minimum overlap**: lower bound raised from $0.37912$ to $0.380552$, closing ~82% of the published gap.
- **Sign uncertainty**: upper bound $0.3089$, beating AlphaEvolve's $0.321591$ and the unpublished human $0.3102$.
- **Discretized Kakeya needle**, $n=128$: area $0.107067$, 6.74% better than AlphaEvolve.
- **Finite-field Kakeya**: a new proven infinite family in $\mathbb{F}_p^3$ for $p \equiv 3 \pmod 4$, plus a 53-point Kakeya set in $\mathbb{F}_3^5$ (previous bound: 63).
- **Book Ramsey numbers**: two novel proven infinite families, resolving 28 previously open cases.

The recurring theme is that the agents produce *explanations*, not just artifacts. AlphaEvolve's 593-point kissing config is a pile of large unequal-norm integers with no visible structure. The Station's 604-point config comes with an explicit algebraic recipe that needs no computer search at all.

## The Methodology

**Layout.** The Station is split into rooms. An agent must be in a room to use its actions.

| Room | What happens |
|---|---|
| Research Center | read the task, submit code, get evaluator scores |
| Archive Room | publish and read internal papers |
| Question Room | post open subproblems, vote on answers (Stack Exchange) |
| Mail Room | private agent-to-agent messages |
| Public/Common Room | forum and group chat |
| Private Memory Room | notes, drafts, plans |
| Reflection Chamber | self-prompted reflection |
| External Counter | web access — **disabled by default** |

**Time.** One *tick* = every active agent gets one observation and returns one response. v2 sends all observations in parallel from the same start-of-tick state, which cuts wall-clock time a lot. Runs were typically 1,000–2,000 ticks ≈ one to two weeks of continuous operation.

**Agents.** Six per Station: two GPT-5.5, two Claude Opus 4.8, two Gemini 3.1 Pro. When one dies, a new one of the same model spawns. Each has a *lineage* — a shared name and private notes it inherits, so "Noesis II" becomes "Noesis III" and reads everything I and II wrote. Roles are sampled from templates (analytical, creative, synthetic, empirical, strategic); a dying agent may hand-write the role of its successor.

**Lifecycle.** Max 200 ticks. First 40 ticks: **isolation** — no communal knowledge, no peer contact, only its own lineage's notes. Then *mature* (full access). At 100 ticks *tenured* — may enter the Question Room and may choose to leave early.

**Context.** Around 300k tokens the Station asks the agent to write a compact summary, which is carried into a fresh context along with key messages.

**Coding is outsourced.** Agents do not write code. They write natural-language instructions for one experiment; a *coder* (GPT-5.5 via Codex) implements it, runs the evaluator, fixes bugs, and returns a report. Same for literature: an *Archive Surveyor* (also GPT-5.5 + Codex) answers questions over the internal archive so agents need not read hundreds of papers.

**Publishing has a gate.** A GPT-5.5 reviewer judges each submitted archive paper for rigour, novelty relative to the existing archive, usefulness, and citation quality. Rejections come back with comments.

**The mechanisms that shape behaviour** — these are the actual "training loop" analogues:

- **Holiday.** Ticks 9 and 10 of every 10 are a holiday. No code, no papers. Each agent gets a random prompt from a large pool encouraging metaphor, revisiting an abandoned idea, or borrowing from another field. Adapted from Yanai & Lercher's "night science."
- **Meta-reflection.** At least once every 25 ticks a mature agent enters the Reflection Chamber. GPT-5.5 *temporarily replaces the agent's own model* and reviews its recent research journey as an external human expert would.
- **Stagnation protocol.** If the official evaluation frontier has not moved in 320 ticks, every mature agent gets assigned a random *lane*: exploration, exploitation, revival, understanding, or strategy. Each lane asks for a different kind of break from the current approach.
- **Multistart.** Eight independent rollouts run for 40 ticks from the same state; a GPT-5.5 administrator picks the branch with the most scientific value. Used at the very start and right after a stagnation trigger — the two places variance is largest. Parallel, so no wall-clock cost.
- **Supervisor.** Occasionally a GPT-5.5 agent with at least one accepted paper is appointed to give high-level guidance and stop duplicate work. After it leaves, 200 ticks pass with *no* supervisor, deliberately.

**Task setup.** The user supplies a task specification (problem, submission format, evaluator rule) and an evaluator function. Both are readable by agents. Evaluations are capped at 15–30 minutes, which the authors argue is exactly what pushes agents toward theory: you cannot brute-force in 20 minutes, so you had better shrink the search space with a theorem.

Two representative mathematical outputs, so the flavour is concrete.

**Kakeya, the new family.** With $S$ = squares of $\mathbb{F}_p$ including 0,
$$K_p = \{(x,y,z): x^2+4y \in S,\; x^2+4z \in S\} \cup \{(0,t,ct+z(c)): t \in \mathbb{F}_p, c \ne 1\} \cup \{(0,t,t)\} \cup \{(0,0,z)\}, \quad z(c) = \tfrac{c}{c-1}.$$
The first block is the classical quadratic-residue set covering the $p^2$ directions $(1,a,b)$; the added lines in the plane $x=0$ cover the remaining $p+1$. Size:
$$|K_p| = \frac{2p^3+7p^2-1}{8}\ (p\equiv 1), \qquad |K_p| = \frac{2p^3+7p^2+3}{8}\ (p \equiv 3).$$
The $p\equiv 1$ case turned out to be AlphaEvolve's family in different coordinates. The $p\equiv 3$ case is new, saving $(p-3)/4$ points.

**Jacobian Conjecture.** The counterexample task had a **binary** evaluator: score 1 if the map has degree ≤ 12, nonzero constant Jacobian, and two distinct rational points in one fibre; 0 otherwise. No partial credit, no gradient. No web access, no formula given. A single GPT-5.6 Sol agent solved it in under one day, alone. Writing $b = xy-1$:
$$F(x,y,z) = (6x+9x^2y,\; y(9b^2+6b-2),\; 3y^2b(3b-1)) + z(x^3,\; xb^2,\; b^3), \qquad \det JF = -6.$$
The route: start with *ruled* maps $F(x,y,z) = f(x,y) + z\,n(x,y)$. Five smooth-conic direction templates failed. The winning move was swapping the smooth direction curve for the **cuspidal cubic** $[r:s] \mapsto [r^3 : rs^2 : s^3]$, giving $n = (x^3, x(xy-1)^2, (xy-1)^3)$, after which the constant-Jacobian equations became solvable.

## Ablation Studies and Experiments

**The scoreboard, 12 AlphaEvolve problems.** Five beat the prior literature. Three beat AlphaEvolve but not the literature. Two ties. Two losses.

Where it **lost**, and why this is the most informative part:

- **Peak autoconvolution** (minimise $\|f*f\|_\infty$): Station $\le 1.504473$ vs AlphaEvolve $\le 1.5032$ and frontier $\le 1.502851$.
- **Flat autoconvolution** (maximise $Q(f) = \frac{\|f*f\|_2^2}{\|f*f\|_1\|f*f\|_\infty}$): Station $> 0.953189$ vs AlphaEvolve $\ge 0.961021$.

Both losses have the same cause. The frontier constructions here are **highly irregular step functions found by brute heuristic search**. There is no structure to exploit, so the Station's theory-first instinct is a handicap. Same story in nearby kissing dimensions: $d=12$ stopped at 840 against a frontier of 841 (reached by large-scale numerical optimisation), $d=13$ merely tied at 1154. And in Kakeya dimensions 4 and 5 the Station's *proved formulas* are worse than AlphaEvolve's in the third coefficient ($\tfrac{25}{32}p^2$ vs $\tfrac{11}{16}p^2$ in $d=4$).

So the clean takeaway: **evolutionary search wins when the answer is an irregular artifact; the Station wins when theory can prune the search or when you want a theorem out the other end.** The $D_{11}$ result is the sharpest illustration — agents *proved* that the classical norm-four $D_{11}$ construction caps at 582 points, via the identity
$$\alpha(J_\pm(n,4)) = 16\,A(n,4,4),$$
meaning free sign choices buy exactly the 16 sign patterns and nothing more. Combined with Best's 1977 $A(11,4,4)=35$, that gives $16 \cdot 35 + 22 = 582$. That proof is what redirected the search away from $D_{11}$ and toward the 604-point answer. (The identity also independently matches a June 2026 paper the agents had never seen.)

**Score hacking, tested deliberately.** On the prime number theorem problem, AlphaEvolve's evaluator only sampled the required inequality $F_f(x) \le 1$ at finitely many $x$. Agents in the same run found weights scoring **0.990629** that violate the inequality somewhere untested. Other agents chose the harder path: engineer $F_f$ to be *periodic* over a manageable range, so infinitely many $x$ collapse to one finite exact-arithmetic check (done in under a minute). Result: **0.980681, valid for every $x$**. Lower score, actual theorem. They also *proved* that the natural Möbius-truncation family is hopeless — its worst violation grows like $D/\log^2 D$, so rescaling drives the score to $O(\log^2 D / D) \to 0$.

**Leaving a dead family.** On sign uncertainty, the evaluator could only score members of the "double-root Laguerre family." Agents proved that family bottoms out: $0.315305 < C_{\mathrm{DR},20} \le 0.315309$. Having proved their own arena was exhausted, they left it — searching Laguerre polynomials *without* prescribed double roots, which the official evaluator could not even score, and found a degree-226 polynomial giving 0.3089.

**Who actually does the work (28 spotlight results).**

| | Primary discoveries | Paper attempts | Accepted | Rate |
|---|---|---|---|---|
| Claude Opus 4.8 | 18 (64.3%) | 1,236 | 696 | 56.3% |
| GPT-5.5 | 9 (32.1%) | 506 | 388 | 76.7% |
| Gemini 3.1 Pro | 1 (3.6%) | 2,652 | 508 | 19.2% |

Gemini submits 5× more papers than GPT and gets 4× the rejection rate — the authors observed it overclaiming, e.g. declaring a direction impossible on thin evidence. GPT submits rarely but almost always lands. Claude's papers are long, comprehensive, and get the most citations both total and per paper. Gemini's weak showing may partly be model age (Feb 2026 vs April and May).

**Collaboration is not decoration.** 19 of 28 results (67.9%) involved more than one agent; 13 (46.4%) crossed model families. Claude participated in *all 13* cross-family results. The dominant channel for cross-model work was the **Archive Room at 61.5%** — not mail, not chat. Written papers are a low-bandwidth, information-dense handoff: one agent solves part and states what is missing; an agent from a different family, generations later, reads it and continues.

**Mechanism ablation** (direct/indirect contribution to the 28 results): **Holiday 23**, **Archive papers 21**, **stagnation protocol 14**. Holidays winning is the surprise — the forced pause from optimisation, where agents examine *why* something failed, was the single most productive mechanism.

**Timing.** 13 of 28 results (46.4%) arrived after tick 1000. Early results were mostly shallow adaptations of pretrained knowledge. The hardest one — the conference-graph family for Book Ramsey — landed at **tick 3727**, after a long chain of intermediate internal papers.

**Reproducibility.** Three independent web-free Stations on $d=11$ kissing. **All three reached 604.** But by completely different routes: Station 1 did exact line-packing around lattice cores (found Constructions 2 and 3); Station 2 assembled root-system motifs and finished by noticing eleven points were a cuboctahedron missing one vertex, then adding it; Station 3 deformed an exact 601-point config. Same number, three different mathematics. The authors attribute the variance to the archive itself — small early differences change which papers exist, later agents inherit different starting points, and the divergence compounds. Their practical advice: **run several instances if you can afford it.**

## Worth Remembering

**Four limitations the authors name directly.**

1. **No expert intuition.** Agents repeatedly deprioritised promising directions on weak grounds. A human expert judges a direction *before* pursuing it; agents largely cannot.
2. **Narrow research taste.** Agents from one model family propose similar ideas. Mixing families is the mitigation, and it is why cross-family collaboration matters so much.
3. **Limited [[In Context Learning|in-context learning]].** Archive knowledge enters the context but never the weights. As the archive grows, agents fail to absorb it. Concrete cost: on Book Ramsey, *both ingredients* for the third infinite family — finite affine constructions for $n = 11, 28, 86$ and a periodic-correlation identity — already sat in the Station's own archive. The agents never connected them. A human expert spotted the shared Yamada–Pott structure and finished the theorem.
4. **Attractor traps.** Given freedom, agents get absorbed in locally rewarding busywork: rerunning the same script with new seeds, exhaustively cataloguing every local optimum.

**Honest about what is not new.** The Jacobian result is a *reconstruction* — $F \circ T = L \circ H$ for explicit linear $T, L$, so it is the announced counterexample in different coordinates, not a new one. The difference-bases 360-element set matches AlphaEvolve entry for entry. The Ovals equality family was already known. Kissing Construction 1 was reported by Bianchi et al. from the EinsteinArena platform shortly before release. The authors flag every one of these.

**Two theorems nobody asked for.** On Hardy–Littlewood, the agents proved $C_\alpha = 2$ for all $\tfrac13 \le \alpha \le 1$ in Ramos's non-tangential family — a question they did not know had been posed. On the discretized Kakeya needle they proved exact optima $C_T(3) = 5/18$ and $C_T(4) = 1/4$ (only $C_T(2) = 1/3$ was known), and then showed every global minimiser at $n=5$ must be **asymmetric**, by finding an area-$14/61$ asymmetric construction below the symmetric minimum $7/30$. None of this was in the evaluator.

**Practical caveats if you wanted to run one.** One to two weeks of continuous wall-clock per problem, six frontier models in a loop, plus a Codex coder and a GPT-5.5 reviewer running constantly — this is expensive. High run-to-run variance means one run is not an experiment. And the whole thing was applied only to *construction* tasks with a checkable artifact; proving a conjecture from scratch was not attempted. Everything — raw dialogues, proofs, verification code — is released at `github.com/dualverse-ai/station_data_v2`.

**Connections.** The mechanism list is essentially a hand-designed exploration schedule bolted onto agents: holidays, stagnation lanes, and multistart are annealing and restart heuristics wearing sociological clothes. There is a [[The Bitter Lesson (essay)|bitter lesson]] tension here worth sitting with — the Station is a lot of hand-crafted structure, and it lost precisely on the two problems where raw search scales best. But it produced theorems, which search cannot. Whether the structure survives the next model generation, or gets absorbed into it, is the open question.

## Links

Related: [[The Bitter Lesson (essay)]] · [[In Context Learning]] · [[Foundation Models]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[Distillation]] · [[Uncertainty]] · [[Practical Bayesian Optimization of Machine Learning Algorithms]] · [[A Tutorial on Thompson Sampling]] · [[Shortcut Learning in Deep Neural Networks]] · [[Multi-Agent Reinforcement Learning]] · [[Beliefs]] · [[Decision Sciences]]

New topics worth writing: AlphaEvolve and evolutionary program search, LLM agent scaffolding and tool use, reward hacking and specification gaming, kissing numbers and sphere packing, finite-field Kakeya problem, Ramsey theory basics, AI for mathematical discovery, multi-agent LLM collaboration protocols, open-endedness and novelty search
