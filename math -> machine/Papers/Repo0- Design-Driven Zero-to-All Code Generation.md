---
title: "Repo0: Design-Driven Zero-to-All Code Generation"
authors: ["Silin Chen", "Haoyi Teng", "Xiaodong Gu", "Yuling Shi", "Jiale Huang", "Yongpan Wang", "Hongyu Zhang", "Haibing Guan"]
year: 2026
arxiv: "2608.19854"
url: https://arxiv.org/abs/2608.19854
priority: Low-Priority
read_on: 2026-09-30
tags: [paper, llm, optimization, theory]
---
## The Core Idea

Most coding agents get handed a skeleton. The folders exist, the file names exist, sometimes even the function signatures exist, and the agent's job is to fill in bodies. Repo0 attacks the harder setting the authors call **zero-to-all**: you get a README-style description in plain English and nothing else. No folder layout, no module list, no interfaces. The agent has to decide *what the components are* and *what code goes in them* at the same time.

The insight is small and specific: **the architecture cannot be planned once at the start, because you do not yet know whether your planned components are actually cohesive.** Prior work (notably RPG) builds one planning graph up front, freezes it, and generates code against it. Repo0 keeps the architecture as a mutable *state* that gets edited by explicit structural moves — split, merge, revise, save, add — until a numeric stopping condition says no more edits are warranted. Only then does code generation start.

The second trick is the representation. Repo0 refuses to put requirements and code modules in one graph. It keeps two:

> [!NOTE] Dual-DAG
> Two separate directed acyclic graphs plus a mapping between them. $G^R$ holds requirements and sub-requirements, with edges meaning "these are used together, so their behaviour must line up." $G^C$ holds implementation components, with edges meaning "this one imports/inherits/contains that one." The alignment relation $\mathcal{A} \subseteq V^R \times V^C$ says which component realises which requirement, many-to-many. ^dual-dag

Why does the split matter? Because the *requirement* graph is what lets you **measure** the component graph. A component's quality is judged by looking at the requirements assigned to it and asking: are those requirements actually related to each other? That question is only askable if the two levels are kept apart. Collapse them into one graph and you lose the yardstick.

So: modularity — high cohesion, low coupling, the 1974 Stevens/Myers/Constantine idea — becomes a **computable signal** that drives graph surgery, rather than a vague thing you hope the LLM internalised. That is the unlock.

Concretely, on six real Python repos, this beats the strongest planning baseline by up to **20.08 points of Functionality Coverage** and **29.74 points of Pass Rate**.

## The Methodology

Three phases. Phase I builds the state, Phase II evolves it, Phase III writes code.

### The state

At every step $t$ the system holds $S_t = (G_t^R, G_t^C, \mathcal{A}_t)$.

- $G_t^R = (V_t^R, E_t^R)$ — requirements. Nodes are high-level requirements and their sub-requirements. An edge $(u,v)$ means $v$ logically follows $u$: $v$ depends on the output, data contract or interface $u$ established.
- $G_t^C = (V_t^C, E_t^C)$ — components. A node is a bounded implementation responsibility that will become real files. Edges are implementation dependencies.
- $\mathcal{A}_t$ — the many-to-many map from requirements to the components that realise them.

$G^R$ settles early and is then **frozen**. $G^C$ and $\mathcal{A}$ keep changing. That asymmetry is deliberate: freezing the requirement graph pins the functional scope so the evolution loop cannot quietly drift away from what was asked for.

### Phase I — requirements to initial architecture

1. **Extract.** An LLM pulls capability-level requirement items out of the README. Rules given to the model: one distinct repository-level capability or constraint per item; fold sub-features into the description rather than promoting them; do not split low-level operations into top-level items.
2. **Merge.** Redundant or subsumed items get consolidated. Merely *related* items stay separate. Survivors become high-level requirements — decomposition anchors, not file specs.
3. **Decompose**, in three stages, borrowed from Atom of Thoughts:
   - *Elaborate* — the model writes out the requirement as fully as it can: behaviour, inputs, outputs, constraints, interface expectations, error cases, ambiguous scope.
   - *Identify* — from that enriched text, list sub-requirements.
   - *Label* — add the directed edges between them.

   Reason-first-then-label rather than label-directly is the whole point; the elaboration is scratch space that makes the edge labels less arbitrary.
4. **Ground.** For each high-level requirement, hand the LLM its sub-requirements and ask for bounded components. Each component comes back with a name, a responsibility description, and an explicit list of sub-requirements served. Those lists *are* $\mathcal{A}_0$.
5. **Wire.** Ask the LLM for implementation dependencies between components. Requirement-level edges from $G^R$ are passed in as **soft evidence only** — never copied across. This is where the Dual-DAG earns its keep: "used together" and "imports" are not the same relation.

Worked example from the paper, a mini `requests` clone called HttpEasy: request construction, session persistence, response handling and auth get merged into two high-level requirements. One of them decomposes into build-a-request → send-via-transport → parse-response, which grounds into `RequestBuilder`, `TransportAdapter`, `ResponseParser`, with `SessionClient` depending on the first two.

### Phase II — the evolution loop

Five actions on $G^C$:

- **split** — for a diffuse component $c$, take the subgraph of $G^R$ induced by the sub-requirements $c$ serves, run graph partitioning on it (a minimum-cut objective), and hand the resulting groups to the LLM as evidence. The LLM rewrites $c$ into narrower components, redistributes the alignment pairs, and reconnects the incident dependency edges. The cut is the *evidence*; the LLM does the *instantiation*.
- **merge** — combine two components: concatenate responsibilities, union the alignment pairs, redirect incident edges.
- **revise** — boundary-preserving. Rewrites the responsibility text, interface assumptions, implementation notes, or alignment entries. The component itself stays.
- **save** — mark as stable, carry forward untouched this round unless a neighbour's change makes it eligible again.
- **add** — recover a missing requirement or component.

Now the metrics. For component $c$, responsibility set

$$RS(c) = \{q \in V_t^R \mid (q,c) \in \mathcal{A}_t\}$$

Let $E_{\mathrm{in}}(c)$ count edges of $G^R$ with both endpoints inside $RS(c)$. Cohesion is edge density among the requirements grouped into one component:

$$\mathrm{cohesion}(c) = \begin{cases} 1 & |RS(c)| \le 1 \\[4pt] \dfrac{E_{\mathrm{in}}(c)}{|RS(c)|(|RS(c)|-1)/2} & |RS(c)| > 1 \end{cases}$$

Plain words: of all the pairs of requirements you put in this component, what fraction actually talk to each other? Low means you bundled unrelated jobs together.

Run it: a component serving 4 sub-requirements with 2 internal edges has $\mathrm{cohesion} = 2 / (4 \cdot 3 / 2) = 2/6 = 0.33$. Below the threshold $\gamma_{\mathrm{split}} = 2/3$, so it is a split candidate — provided $|RS(c)|$ also exceeds a round-dependent floor $\tau_{\mathrm{split}}^{(t)}$, which exists purely to stop runaway fragmentation.

Coupling is Jaccard overlap of two components' responsibility sets:

$$\mathrm{coupling}(A,B) = \frac{|RS_A \cap RS_B|}{|RS_A \cup RS_B|}$$

Two components sharing 3 of 5 total sub-requirements give $3/5 = 0.6$ — under $\theta_{\mathrm{merge}} = 0.7$, no merge. A merge candidate needs coupling above $0.7$ **and** connectivity (edges of $G^R$ bridging the two sets) greater than 1. Then an LLM vets it, because high overlap can be legitimate — layered designs, adapters, upstream/downstream pairs all look coupled by this measure and should not be collapsed.

> [!NOTE] Structural convergence
> The loop stops when a full round produces zero eligible split or merge actions. Not a fixed budget, not "the LLM says it's done" — a metric-derived fixed point. After it fires, one final semantic pass applies `revise` where a component's responsibility, coverage or interface assumptions look inconsistent with its neighbours. ^structural-convergence

Thresholds $\gamma_{\mathrm{split}} = 2/3$ and $\theta_{\mathrm{merge}} = 0.7$ were picked on two **held-out** repos from Commit0 Lite (`wcwidth`, small; `sphinx`, large) by eyeballing evolved component boundaries against the real repo architecture. Neither repo appears in the evaluation.

### Phase III — code generation

The converged $G^C$ becomes a concrete plan: package assignments, file paths, exported symbols, and a dependency-aware generation order derived from the graph topology.

Each component's generation context is three things: its own responsibility description, its aligned requirement nodes from $\mathcal{A}_t$, and the upstream components that must already exist.

Then [[Attention Is All You Need|standard]] test-driven development per component:

1. Generate an importable **skeleton** that pins the public API.
2. Synthesise **tests** from the aligned requirement nodes and interface assumptions.
3. Fill in the **implementation** to pass them.
4. Validate: import checks, interface checks, `pytest`.
5. Failures → localised repair patches to implementation, tests, or `__init__.py`. If the failure reveals the *description* was wrong, apply `revise` before the next component.

So the architecture is frozen at convergence, but component descriptions stay patchable from execution feedback.

## Ablation Studies and Experiments

**Benchmark.** RepoCraft — six real Python repos, renamed to blunt pretraining leakage: `scikit-learn`→MLKit-Py (185 files, 66k LOC, 236 tasks), `pandas`→TableKit (217/106k/175), `sympy`→SymbolicMath (699/219k/192), `statsmodels`→StatModeler (271/83k/234), `requests`→HttpEasy (17/2.8k/50), `django`→PyWebEngine (681/109k/165). Two software engineers further rewrote the task descriptions to strip out structural hints, architectural cues and function-level interface details while keeping the functional content.

**Metrics.** Functionality Coverage (fraction of reference functional categories matched by at least one generated functionality), Functionality Novelty (generated functionalities matching *nothing* in the reference), Pass Rate (fraction of tasks whose adapted ground-truth tests pass), Voting Rate (majority-vote semantic judgement that a matching interface exists).

**The evaluation control worth copying:** cross-model grading. DeepSeek V3.2 grades GPT-5-mini's repos and rewrites their tests; GPT-5 mini grades DeepSeek's. Nobody marks their own homework.

**The control that makes the comparison mean something:** every method uses its own planning procedure, but *all of them* then share the identical downstream code-generation, validation and repair scaffold under the same TDD protocol. The only variable is the architecture.

Temperature 0, three runs averaged per setting.

### Main results

| Model | Method | requests Cov / Pass | statsmodels Cov / Pass | django Cov / Pass |
|---|---|---|---|---|
| GPT-5 mini | mini-SWE-agent | 68.18 / 4.11 | 18.18 / 0.00 | 47.92 / 37.04 |
| | Paper2Code | 95.50 / 24.66 | 44.32 / 4.42 | 66.67 / 30.04 |
| | RPG | 90.91 / 31.51 | 70.40 / 77.90 | 60.42 / 47.33 |
| | **Repo0** | **100.00 / 50.98** | **80.68 / 85.51** | **80.50 / 74.36** |
| DeepSeek V3.2 | mini-SWE-agent | 86.36 / 21.92 | 59.09 / 2.65 | 33.33 / 10.70 |
| | Paper2Code | 90.91 / 4.11 | 14.77 / 49.56 | 62.50 / 7.82 |
| | RPG | 95.45 / 61.64 | 64.70 / 39.29 | 68.75 / 46.50 |
| | **Repo0** | **100.00 / 78.08** | **78.41 / 69.03** | **79.17 / 74.07** |
| Human | Gold Project | 100.00 / 94.12 | 100.00 / 94.15 | 100.00 / 96.34 |

Best Coverage and best Voting Rate in all six settings; best Pass Rate in all six. Gaps over RPG under GPT-5 mini: +19.47 Pass on requests, +7.61 on statsmodels, +27.03 on django.

Baseline failure signatures differ usefully. mini-SWE-agent cannot hold repository-wide consistency at scale — 0.00% Pass Rate on statsmodels under GPT-5 mini. Paper2Code posts high Novelty that never converts into correctness: it writes plenty of code, just not the code asked for. RPG is genuinely strong and degrades gracefully, which is what you would expect from real planning that happens to be frozen too early.

### Ablations (GPT-5 mini)

| Variant | requests Cov / Pass | statsmodels Cov / Pass | django Cov / Pass |
|---|---|---|---|
| Repo0 | 100.00 / 50.98 | 80.68 / 85.51 | 87.50 / 74.36 |
| w/o Requirement Context | 95.45 / 45.39 | 67.92 / 85.51 | 78.29 / 64.36 |
| w/o Component-Graph Ordering | 100.00 / 50.98 | 80.68 / **55.51** | 83.56 / 67.70 |
| w/o Dual-DAG | 95.45 / 48.72 | 78.55 / 85.51 | 82.24 / 64.36 |
| w/o Structural Evolution | 94.32 / 42.51 | 75.35 / 73.51 | 81.58 / 61.03 |

Reading it:

- **Structural evolution is the biggest single contributor.** Removing it hurts every repo on every axis. requests loses 5.68 Coverage / 8.47 Pass / 17.86 Voting; django loses 5.92 / 13.33 / 8.34.
- **Generation order is invisible until scale.** On requests, dropping dependency-aware ordering changes *nothing* — 17 files, order does not matter. On statsmodels it costs **30 points of Pass Rate**. This is the ablation that scales worst with repo size and it is the one people would be most tempted to skip.
- **Requirement Context mainly buys correctness, not coverage.** Feeding a component not just its own aligned requirements but the requirements *coordinated with them* via $G^R$ edges is what keeps interacting behaviours consistent. Pass Rate drops 10.00 on django, 5.59 on requests.
- **Dual-DAG earns a consistent but modest keep.** Collapsing to one graph always costs either coverage or accuracy, never nothing.

**The negative result to notice:** several ablated variants score *higher* Functionality Novelty on statsmodels (+4.09 without ordering, +3.18 without Dual-DAG, +2.02 without evolution) while Coverage falls. Novelty rising as Coverage falls is the signature of a system inventing functionality instead of implementing the spec. It is a straightforward [[Reward Hacking|Goodhart]] trap: read Novelty alone and the broken variants look more creative.

### Structural evolution analysis — the sharpest experiment

Repo0's metric-guided convergence is compared on statsmodels against (a) no evolution, and (b) **LLM-decided** evolution with fixed budgets of 1, 3 and 5 rounds. The budgeted variants can take the same structural actions; they just have no cohesion/coupling signal deciding *which* components to touch or *when* to stop.

Repo0 beats every budgeted variant — including the 5-round one, which gets strictly more refinement opportunity. Against no-evolution it goes 75.90→80.68 Coverage, 11.05→11.48 Novelty, 81.90→85.51 Pass, 93.00→98.65 Voting.

Past one round, unconstrained LLM-decided evolution *degrades* Coverage, Pass and Voting while slightly raising Novelty. The stated mechanism: without a convergence criterion the model keeps restructuring past the useful point, fragmenting components and then restructuring again to compensate. **More refinement is not the benefit. Knowing when to stop is the benefit.**

### Action distribution

Aggregated over six repos and both backbones: `split` dominates, `save` second, then `merge`, `revise`, `add` well behind. Evolution is mostly local boundary tuning, and the high `save` share says most initial components were already fine.

GPT-5 mini triggers noticeably more `revise` and `add` than DeepSeek V3.2. The tempting read — GPT-5 mini is better at architectural repair — is wrong, and the authors ran the control: fix the same unoptimised GPT-5-mini initial state and evolve it with both models. DeepSeek V3.2 then produces roughly the same number of `revise` actions. **The action distribution is a property of the initial architecture, not of the evolving model.** `revise` matters most for models whose first-pass component responsibilities are vague.

### Cost

Under DeepSeek V3.2, Repo0 generation costs $11.95 / $28.19 / $27.24 on requests / statsmodels / django. Evaluation dominates for big repos — django evaluation alone is $72.82, total $100.06. RPG's generation is $9.32 and $35.59 *more* on requests and statsmodels, $8.83 less on django. Under GPT-5 mini, Repo0 is cheaper to generate everywhere.

The causal story offered: better cohesion and lower coupling mean less redundant and conflicting code, so fewer expensive TDD repair loops. Modularity pays for itself in retries.

## Worth Remembering

**The metric is only definable because of the representation.** This is the part to keep. Cohesion here is not "does this module feel focused" — it is edge density in the *requirement* graph restricted to one component's responsibility set. You literally cannot compute it without two separate graphs and an alignment map. The Dual-DAG ablation looks like the mildest one in the table, but every other mechanism in the paper is downstream of it.

**LLM proposes, metric decides where and when.** The division of labour is clean. Graph min-cut finds candidate partitions; thresholds gate eligibility; convergence is a fixed point; the LLM does the rewriting and vetoes bad merges on semantic grounds (layers and adapters legitimately overlap). Neither half works alone — that is exactly what the budgeted-LLM comparison shows.

**Limitations the authors state.** Six Python repos, one language, one benchmark. They argue the mechanism is language-agnostic since it operates on architecture rather than syntax, but do not demonstrate it. Structural update quality still depends on the backbone's architectural reasoning — the metrics only *select candidates*; the LLM instantiates them.

**Thresholds are hand-tuned on two repos by manual comparison to golden architectures.** $\gamma_{\mathrm{split}} = 2/3$, $\theta_{\mathrm{merge}} = 0.7$. Held out properly, but that is a small tuning set and the sensitivity is never reported. If you ported this, that is the first thing to sweep.

**Architecture freezes at convergence.** Phase III can `revise` descriptions from validation feedback but cannot split or merge. So the claim "design and implementation are deeply intertwined" is softer than it sounds: structural evolution happens strictly *before* any code exists, informed by requirement-graph topology, never by observed code. The headline motivation — "generated code reveals whether components are cohesive" — is not actually the loop that runs. Real code-informed restructuring is left open.

**Practical caveats if you wanted to build this.** Evaluation cost exceeds generation cost on large repos ($72.82 vs $27.24 on django) — budget for grading, not just building. Dependency-aware generation order is nearly free to implement and worth 30 Pass Rate points at statsmodels scale; do not skip it. And never read Functionality Novelty without Coverage beside it.

**Follow-up questions.** Does cohesion measured on the *dependency graph extracted from generated code* beat cohesion on the requirement graph, now that code exists? What happens if the frozen $G^R$ is simply wrong — there is no mechanism to revise it, so a bad decomposition is permanent. And does convergence actually terminate reliably, or does $\tau_{\mathrm{split}}^{(t)}$ (round-dependent, never specified numerically) quietly do the terminating?

## Links

Related: [[Agentic Workflows]] · [[Planning and Decomposition]] · [[Multi-Agent LLM Systems]] · [[Chain of Thought]] · [[Evals]] · [[Reward Hacking]] · [[Context Engineering]] · [[An Empirical Study of Harness Design for Coding Agents]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Agent Evaluation]] · [[Reflection]] · [[Structured Output]] · [[Tool Use]]

New topics worth writing: Cohesion and coupling (Stevens/Myers/Constantine structured design), Graph partitioning and minimum cut, Jaccard similarity, Test-driven development as an agent scaffold, RepoCraft and Commit0 benchmarks, Repository planning graphs (RPG), Atom of Thoughts decomposition, Requirements traceability
