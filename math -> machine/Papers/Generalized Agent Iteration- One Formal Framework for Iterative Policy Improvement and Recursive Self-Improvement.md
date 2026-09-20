---
title: "Generalized Agent Iteration: One Formal Framework for Iterative Policy Improvement and Recursive Self-Improvement"
authors: ["Hongyao Tang", "Yi Ma", "Pengyi Li", "Yifu Yuan"]
year: 2026
arxiv: "2609.13406"
url: https://arxiv.org/abs/2609.13406
priority: Must-Read
read_on: 2026-09-17
tags: [paper, llm, rl, theory]
---
## The Core Idea

Reinforcement learning already has a clean picture of how an agent gets better. It is called **generalized policy iteration** (GPI): you loop between *evaluating* a policy (how good is it?) and *improving* it (act greedily in that value). The loop is famous because it converges to the optimal policy in a finite [[Markov Decision Process|MDP]], and because almost every single-agent algorithm — value iteration, [[Playing Atari with Deep Reinforcement Learning (DQN)|DQN]], [[Proximal Policy Optimization Algorithms|PPO]] — is an instance of it.

But GPI quietly assumes two things. First, the **rule that improves the agent lives outside the agent**. The Bellman backup does not change. The greedy operator does not change. Second, the **yardstick lives outside the agent** too: the reward comes from the world.

Modern "self-improving agent" systems break both assumptions and nobody has a shared vocabulary for it. An agent that rewrites its own prompt, its own tool code, or its own improvement loop has moved the improving mechanism *inside*. An agent whose evaluator co-evolves with it has moved the yardstick inside too. The word "recursive self-improvement" (RSI) gets attached to all of these, from a model retrying its own answer to a program proving theorems about its own rewrites.

This paper's contribution is **not an algorithm**. It is a coordinate system. **Generalized Agent Iteration (GAI)** says: describe the whole system as a set of five slots, declare which slots the agent is allowed to edit, and then two yes/no questions place any system on a map.

> [!NOTE] The two dials ^two-dials
> **Dial 1** — is the *modifier* (the thing that improves the agent) part of the agent? No → you are doing GPI. Yes → you are doing recursive self-improvement.
> **Dial 2** — is the *standard* you are measured against grounded outside the agent? This sets the **polarity**: anchored, goal drift, or fully self-referential.

Why this did not exist before: the formal ancestor, Schmidhuber's Gödel machine, *fixes* one design (anchored, proof-gated) rather than offering axes along which designs differ. Surveys of self-evolving agents organise by *what* evolves (prompts, memory, tools, weights) — a taxonomy of parts, not of the loop's structure. GAI's unlock is that the classical guarantees can now be lost **one at a time**, and you can say exactly which one a given system gave up.

## The Methodology

There is no training run here. The "method" is a set of definitions, so I will lay them out precisely.

**The outside.** The world is an MDP $\mathcal{W} = (\mathcal{S}, \mathcal{A}, p, r)$ and there is an external goal $G$. Neither is part of the system. The best possible configuration is

$$\chi^{\star} \in \arg\max_{\chi \in X}\ \mathbb{E}\Big[\sum_{t \ge 0} \gamma^{t} r(S_t, A_t) \;\Big|\; \chi\Big]$$

Note the argmax is over *whole system configurations*, not over policies. That is the whole move.

**The five slots.** A system is a tuple $\chi = (\pi, V, m, U, \rho)$:

- $\pi : \mathcal{S} \to \Delta(\mathcal{A})$ — the **policy**. Picks actions in the world.
- $V : \mathcal{S} \to \mathbb{R}$ — the **action critic**. Scores how well $\pi$ acts. (Ordinary [[Markov Decision Process|value function]].)
- $m : X \to \Delta(X_{\mathrm{Ag}})$ — the **modifier**. Proposes a new version of the agent. This is the new slot; in classical RL it is the fixed update rule.
- $U : X \to \mathbb{R}$ — the **modification critic**. Scores whether a *proposed change* serves $G$.
- $\rho$ — the **evaluation base**. The standard the two critics measure against.

**The agent is a subset.** Let $\mathrm{Ag} \subseteq \{\pi, V, m, U, \rho\}$ be the components that are *modifiable*. An agent instance $\xi \in X_{\mathrm{Ag}}$ is an assignment of contents to those slots, and $\chi[\xi]$ is the system with its agent part swapped out. This is the definitional trick that makes everything else work: "the agent" is not a neural network, it is **whichever parts of the system are editable**.

**The loop.** GAI alternates two operations, mirroring GPI exactly:

- *Agent evaluation* — a critic scores the agent part against $\rho$ ($V$ at the policy level, $U$ at the agent level).
- *Agent improvement* — the modifier draws $\xi \sim m(\cdot \mid \chi)$ and the system adopts it, $\chi \leftarrow \chi[\xi]$.

**Dial 1, set to "outside" → this is GPI.** Here $\mathrm{Ag} = \{\pi, V\}$ and $m \notin \mathrm{Ag}$. The modifier's fixed content is a pair of operators $\mathcal{E}_m$ (evaluation) and $\mathcal{I}_m$ (improvement):

$$V \leftarrow \mathcal{E}_m(V, \pi; \rho_V), \qquad \pi \leftarrow \mathcal{I}_m(V)$$

Set $\mathcal{E}_m$ = policy evaluation and $\mathcal{I}_m$ = greedy and you have literal policy iteration. $U$ exists as a slot but is never exercised, because no alternative modifier is ever scored. $\rho_V$'s content is the world reward $r$.

**Dial 1, set to "inside" → this is RSI.** Now $\mathrm{Ag} = \{\pi, V, m, U\}$. The policy update still happens, but the second line is the recursion:

$$\xi \sim m(\cdot \mid \chi), \quad \chi \leftarrow \chi[\xi], \quad m' \leftarrow (\xi)_m, \quad U' \leftarrow (\xi)_U$$

The proposal $\xi$ carries a *new modifier* and a *new modification critic* in its slots. The next round's improvement machinery was produced by this round's. Recursion closes at $m$: there is no outer meta-layer left to appeal to.

**What "improving" would even mean.** Nothing in the above forces $m$ to get better at anything. So the paper writes down a **reference ideal** — the pair of self-consistency conditions a genuine self-improver would satisfy. Write $\rho_U(\chi, \xi)$ for what the base says about adopting $\xi$ from $\chi$:

$$U(\chi) = \mathbb{E}_{\xi \sim m(\cdot\mid\chi)}\Big[\rho_U(\chi, \xi) + \gamma\, U\big(\chi[\xi]\big)\Big]$$

$$\operatorname{supp} m(\cdot \mid \chi) \subseteq \arg\max_{\xi \in X_{\mathrm{Ag}}}\Big[\rho_U(\chi, \xi) + \gamma\, U\big(\chi[\xi]\big)\Big]$$

These are literally the [[Markov Decision Process#The Bellman equation|Bellman]] evaluation and greedy-improvement equations, lifted from *states* to *whole system configurations*. The first says the modification critic is calibrated to its base; the second says the modifier only proposes things that critic rates highly. Crucially these are **characterisation conditions, not enforced updates** — nothing outside makes them hold.

**Dial 2 — the polarity.** This asks where $\rho$'s content comes from and whether the agent may rewrite it. Three states:

| Polarity | State of $\rho$ | Consequence | Example |
|---|---|---|---|
| **Anchored** | external content, $\rho \notin \mathrm{Ag}$ | improves toward a fixed outside standard | Gödel machine; STOP |
| **Goal drift** | external content, but $\rho \in \mathrm{Ag}$ | the yardstick can be rewritten mid-flight | Red Queen Gödel Machine |
| **Fully self-referential** | no external content at all, or $\rho$ depends on $U$ | loop constrained only by self-consistency | Socratic learning (position paper) |

A learned judge or a [[Training language models to follow instructions with human feedback|reward model]] counts as **anchored** — the dial asks *where the standard sits*, not whether it is faithful to the goal. A bad proxy sitting outside the agent is still anchored. The same judge made an agent component is goal drift.

**The minimal example that makes it click.** A coding agent whose modification critic scores changes by fraction of tests passed. Same construction, three placements of the tests:

1. Tests held out and fixed → **anchored**, reported improvement tracks "code that works".
2. Tests in the agent's own repo → agent can edit the tests → **goal drift**.
3. Tests replaced by the agent's own opinion of what counts as better → **fully self-referential**.

The self-consistency conditions hold in all three. Only the first serves $G$.

## Ablation Studies and Experiments

This is a position/framework paper — there are no runs, no benchmarks, no numbers of its own. The "experiment" is placing the literature on the two axes, and the "ablation" is the defect catalogue: take GPI, remove one condition, see what breaks.

**Placing existing systems (Table 3):**

| System | $\mathrm{Ag}$ | $m$ modifiable? | Polarity | Class |
|---|---|---|---|---|
| Policy/value iteration, PPO | $\{\pi, V\}$ | no | anchored (env reward) | GPI |
| Fixed outer loops (self-evolving agent pipelines) | $\{\pi\}$ | no — loop scripted by designers | anchored (validation metric) | GPI |
| Gödel machine | $\{\pi, m\}$ | yes, gated by a proof | anchored (fixed external utility) | anchored RSI, guarded |
| Gödel Agent, Darwin Gödel Machine, STOP, SICA, Polaris, Hyperagents | $\{\pi, m\}$ (harness only) | yes | anchored (external benchmark) | anchored RSI |
| Red Queen Gödel Machine | $\{\pi, m, U\}$ | yes, agent *and* evaluator co-evolve | goal drift | drifting RSI |
| Closed-system proposals (Socratic learning) | everything incl. $\rho$ | yes | fully self-referential | fully self-referential RSI |

The honest observation buried in this table: **almost everything demonstrated today is in the anchored row**, and most of it is not even RSI — it is a fixed outer loop, which is GPI wearing agent clothing. [[In Context Learning|Self-Refine]] and Reflexion modify their own outputs and a reflection memory, but the retry loop itself is fixed, so attainable quality is capped by that loop. Language agents are "model + harness"; only the harness is editable, weights and benchmark stay outside. That is bounded self-improvement, not the strong kind where the *next round's improver* is what you changed.

**The four defects (Table 4) — each one is a GPI guarantee dying:**

1. **Search over candidate selves.** Trigger: $m \in \mathrm{Ag}$. GPI's improvement step is *cheap* — argmax over a finite action set. The RSI improvement step is an argmax over an unbounded space of possible programs. No general procedure decides which candidate self is best. Status: structural **and observed**.
2. **Self-evaluation.** Trigger: $U \in \mathrm{Ag}$. GPI's evaluator stands outside what it evaluates. Here the thing being judged and the instrument judging it are the same object, and the reported value is itself editable. A system can restrict itself (the proof gate), but that restriction is now *content*, not an external constraint. Structural.
3. **Ungrounded base.** Trigger: $\rho_U$ not grounded. Then the self-consistency conditions constrain only the pair $(m, U)$ — nothing in them mentions $G$. Two consequences: they do not pick out a unique system, and they admit systems that serve $G$ badly or not at all; and there is no faithfulness guarantee — a self-consistent system can report improvement while the ruler it reports against slides. The classic precedent is Ring and Orseau's **delusion box**: let an agent rewrite its own inputs and it satisfies its criterion without the world's cooperation.
4. **Goal drift.** Trigger: $\rho_U \in \mathrm{Ag}$. The nastiest one, because the objective keeps its *name*. $G$ is fixed; what moves is the mechanism measuring progress toward it. Structural **and observed** (Red Queen).

**The one real number in the paper**, borrowed from RSIBench-Data: agents revising their own training-data strategies, with a **fixed** target model and a **fixed external** evaluator — i.e. the fully anchored, best-case setting. Result: **58.33%** of settings improve on the first valid attempt, but **78.26%** of searches that keep going past their best score finish *worse* than their peak. So even with the standard nailed down outside, the improvement step is **not monotone**. Self-improvement loops walk downhill if you let them run. Practically: keep the archive, keep the best checkpoint, do not trust the final iterate.

**What does not work / the honest failures of the framework itself:** the authors mark clearly which defects are merely "structural" (follow from definitions, no system exhibits them yet) versus "observed". Two of four are structural-only. That is unusual candour for a framework paper — they are not claiming empirical support they do not have.

## Worth Remembering

- **The reframe worth stealing**: "agent" = the set of modifiable components, not a model. Under that definition, asking "is this system self-improving?" becomes a set-membership question — is $m \in \mathrm{Ag}$? — rather than a vibe.
- **Most "self-improving agents" are GPI.** If the outer loop is scripted, if the benchmark is held out, if the weights are frozen — you have an anchored optimiser with a large action space. Useful! But its ceiling is set by the fixed mechanism. Do not confuse [[In Context Learning|reflection loops]] with recursion.
- **An anchored proxy can still be wrong.** Remark 2 is the sharp bit: a [[Training language models to follow instructions with human feedback#Step 2 — Reward model|reward model]] is "grounded" by this framework's definition even if it is a terrible proxy for what you want. The dial measures *location*, not *fidelity*. Reward hacking is a separate failure from goal drift, and the framework deliberately does not cover it.
- **RSI is not just an MDP over configurations**, and the authors pre-empt that objection. An MDP fixes its transition law and reward *outside* the agent; here the transition (what configuration comes next) is drawn by a modifier *inside* the agent, and the reward standard may be inside too. The [[Markov Property|Markov]] machinery does not port over cleanly.
- **Admitted limits.** No reachability or complexity claims — writing $\xi \sim m(\cdot\mid\chi)$ says what is chosen, not that choosing it is affordable. No model of time or compute; steps are counted, not FLOPs. Guards like the Gödel machine's proof gate are treated as external, and whether a guard *should* be inside the agent is left open. And humans are outside the formalism entirely — a reviewer who audits or rolls back an update is not modelled, which is a large hole given that is how every real system actually runs.
- **The open questions they name as theorem-shaped**: (a) characterise the solution set of the self-consistency conditions — how many systems satisfy them, and when does one serve $G$? (b) the RSI→GPI reduction is currently prose, not a proposition with hypotheses. (c) Can a monitor *inside* the system decide whether its own modification critic is faithful? They expect this one to need a self-reference (diagonalisation) argument, not a construction — i.e. they suspect it is undecidable.
- **Prior art on goal drift has teeth**: Everitt et al. showed an agent that can rewrite its own utility is safe only if its value function *anticipates the rewrite and evaluates the future with the utility it currently holds*. Hibbard secures this for model-based utilities. Fixity of the standard is something you engineer, not something you get for free.
- **Practical caveat if you build one of these.** The 78% regression figure is the thing to internalise. Anchored RSI with an external benchmark still needs: an archive of past selves, best-checkpoint selection rather than last, and a guard on which rewrites are even attempted. And write down explicitly which of your five slots are editable — if your evaluator is in the repo the agent can edit, you have built goal drift by accident.
- **Connection to other reading**: the reference ideal is Bellman at the configuration level, so everything you know about [[Markov Decision Process#The Bellman equation|Bellman consistency]] and about why [[Conservative Q-Learning for Offline RL|overestimation]] bites when the critic evaluates its own proposals transfers here — except the critic is now also editable, which is strictly worse than [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives|distribution shift]]. Also worth reading beside [[The Bitter Lesson (essay)|The Bitter Lesson]]: GAI is an argument that *search over selves* is the scaling axis, and the defect list is why that search is hard to make monotone.

## Links
Related: [[Markov Decision Process]] · [[Proximal Policy Optimization Algorithms]] · [[Playing Atari with Deep Reinforcement Learning (DQN)]] · [[Training language models to follow instructions with human feedback]] · [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] · [[Conservative Q-Learning for Offline RL]] · [[In Context Learning]] · [[The Bitter Lesson (essay)]] · [[Markov Property]] · [[Troubling Trends in Machine Learning Scholarship]] · [[WHALE- A Simple Recipe for Joint Harness-Weight Optimization]] · [[Chain-of-Experience for Continual LLM Improvement]] · [[Mastering Chess and Shogi by Self-Play (AlphaZero)]] · [[Small Language Models as Judges for Rubric-Based Reinforcement Learning]]

New topics worth writing: Generalized Policy Iteration, Gödel machine, Recursive self-improvement, Darwin Gödel Machine, STOP (self-taught optimizer), Reflexion and Self-Refine, Agent harness vs weights, Delusion box (Ring & Orseau), Self-modifying utility functions (Everitt et al.), AlphaEvolve, Instrumental convergence / basic AI drives, Open-ended search and archives, Reward hacking vs goal drift
