---
title: "What Makes Good Agentic Data? An ACE Lens on Data Generation for LLM Agents"
authors: ["Xingshan Zeng", "Zishan Xu", "Boju Zhang", "Yuzhou Wu", "Lingzhi Wang", "Jianghao Lin", "Liangyou Li", "Yasheng Wang", "Lifeng Shang", "Xin Jiang", "Weinan Zhang", "Yong Yu", "Qun Liu", "Weiwen Liu"]
year: 2026
arxiv: "2608.27260"
url: https://arxiv.org/abs/2608.27260
priority: Good-To-Read
read_on: 2026-09-04
tags: [paper, llm, rl]
---
## The Core Idea

This is a survey, not a new model. Its contribution is a vocabulary that makes hundreds of "we synthesised agent data" papers comparable.

Two moves.

**Move 1: write every agentic training example as the same four-part object.**

$$d = (E, q, \tau, v)$$

- $E$ — the environment specification. The tools, the database or repo, the transition rules, permissions, what parts of state are visible.
- $q$ — the task signal. What the agent must achieve, and under what constraints. May be hidden and revealed slowly across user turns.
- $\tau$ — the interaction realisation. The actual recorded rollout $(o_0, a_1, o_1, \ldots, a_T, o_T)$.
- $v$ — an optional verifier. A schema check, a unit test, a terminal-state predicate, a proof assistant, an LLM judge.

Before this, a tool-use paper talked about "API pools", a coding paper about "repositories and tests", a GUI paper about "screenshots and clicks". Same four roles, four vocabularies. The factorisation says: they are the same object, so a mechanism invented for one transfers to another.

**Move 2: treat data generation as *constrained distribution design*, not sample production.** This is the ACE lens — **A**ccuracy, **C**omplexity, div**E**rsity. The key claim is that these three are *not* co-equal. Accuracy is a hard gate; the other two only shape what happens inside that gate.

$$\max_{\phi}\;\mathbb{E}_{\mathcal{B}\sim p_{\phi}}\!\left[\lambda_{C}\frac{1}{|\mathcal{B}_{A}|}\sum_{d\in\mathcal{B}_{A}}g_{z}(C_{z}(d))+\lambda_{D}D(\mathcal{B}_{A})\right]\;\text{s.t.}\;\Pr_{d\sim p_{\phi}}[A(d)=1]\ge\alpha$$

$\mathcal{B}$ is a generated batch, $\mathcal{B}_A$ is the part that passes the validity check $A(d)$, $C_z$ is difficulty *for a named learner* $z$, $D$ is batch-level coverage. If nothing passes the gate, the utility is zero. A broken environment does not become useful by being hard or unusual.

What this unlocks: it separates *how you build the data* from *what you should keep*. Most papers mix the two, so you cannot tell whether the gains came from the clever tool-graph sampler or from the execution filter bolted on at the end.

> [!NOTE] Common data object
> Any agent training sample is $(E, q, \tau, v)$: an actionable world, a grounded goal, a realised interaction, and a way to check it. The four need not be serialised separately — initial state can live inside an executable environment, and $q$ can be implicit in the user turns of $\tau$. ^common-data-object

## The Methodology

**The underlying process model.** Agent interaction is a POMDP $\mathcal{M}=(\mathcal{U},\mathcal{S},\mathcal{A},\mathcal{O},\mathcal{P},\mathcal{R})$ — task space, latent state, actions, observations, transitions, reward. See [[Markov Decision Process]] and [[Dec-POMDP]] for the machinery. The policy sees only history, not state:

$$h_{t}=(o_{0},a_{1},\ldots,o_{t-1}),\quad a_{t}\sim\pi_{\theta}(\cdot\mid h_{t}),\quad (s_{t+1},o_{t+1})\sim\mathcal{P}(\cdot\mid s_{t},a_{t})$$

That gap matters for data: the trajectory records what the policy *saw*, while a real executable environment additionally keeps latent state for transitions and reward. A logged trajectory therefore cannot always be replayed; an executable $E$ can.

Concretely $e=(\mathcal{D},\mathcal{F},\mathcal{P}_{\mathrm{rule}},\Omega,v)$: state carrier (DB / repo / simulator), action set, rules and permissions, observation function, success interface.

**The taxonomy of pipelines.** Organised by which factor is the anchor.

The default order is $p(E)\,p(q\mid E)\,p(\tau\mid E,q)$ — **forward generation**. Build a world, write tasks the world supports, then roll out. Environments come from three places:
1. *Real / curated* — crawled APIs, MCP servers, GitHub repos (ToolLLM, Gorilla, TOUCAN, SWE-Gym). Realistic, but with broken docs and licensing pain.
2. *LLM-synthesised* — evolve an API pool, then generate on it (ToolACE, ToolAlpaca, Seal-Tools). Scalable, but interfaces can be underspecified.
3. *Programmatic* — actually implement a database, transition function, and validator in code (EnvScaler, EnvFactory, ScaleEnv, Agent-World). Supports reset, state checks, RL rollouts. Expensive to build and maintain.

**Reverse generation** flips the order:
- *Task-first* — declare the capability you want, then build tools and environment to support it (AgentInstruct, BUTTON composing atomic tasks into multi-turn ones; the whole "turn a math problem into a tool-integrated trajectory" family — ToRA, MathCoder, ReTool).
- *Trajectory-first* — explore the environment, then write the instruction that the observed path answers (OS-Genesis on GUI traces, Learn-by-interact, Explorer). Executability is free; coverage is capped by whatever the explorer happened to find.
- *Structure-first* — generate a scaffold (tool graph, dialogue skeleton, verified blueprint) and realise $E,q,\tau$ around it (APIGen-MT, Magnet, ToolACE-MT). The scaffold is not a fifth factor; it is a device to keep the other four consistent.

**Accuracy, formally.** A conjunction, deliberately:

$$A(d)=V_{E}(E)\wedge V_{q}(q\mid E)\wedge V_{\tau}(\tau\mid E,q)\wedge V_{v}(v\mid E,q,\tau)$$

A perfect-looking trajectory does not rescue an infeasible task. A correct terminal state does not rescue a verifier that accepts a policy-violating shortcut. Four recurring mechanisms in the literature:

1. **Layered checking.** Cheap deterministic rules first (schema, arg types), then execution, then model critics, then human review only as escalation. APIGen: format → execution → semantic review. ToolMind and WebSTAR add *turn-level* filtering, because in a long trajectory one bad step silently corrupts every later observation.
2. **Constraint-grounded construction.** Verify the blueprint *before* simulating the dialogue (APIGen-MT). Sample from a tool dependency graph so incompatible calls cannot be written.
3. **Execution-grounded verification.** Actually run the call, diff the database, run the tests, ask the proof assistant. This is the big trend line — from "does it look plausible" to "does it run".
4. **Feedback-driven repair.** Localise the failed test, regenerate that tool, retry from the last verified state, and promote successful tasks into seeds for the next round.

**Complexity, formally.** Difficulty is *not* a property of the task. It is a property of task-plus-learner:

$$C_{z}(d)=1-\Pr[v(d,\tau)=1\mid d,z]$$

where $z$ is the whole execution configuration — model, scaffold, tool access, verifier, sampling temperature, inference budget. Horizon, tool count, graph depth are *explanatory variables*, not scores.

And you do not maximise $C_z$. You aim at a band. The paper gives a sharp paired criterion for "useful agentic task": with $p_z(d)$ the verified success probability,

$$p_{z_{0}}(d)<\rho\leq p_{z_{A}}(d)$$

i.e. the base model fails it, the agent-assisted configuration solves it. That separates the genuinely agent-requiring middle from both the already-solved tail and the beyond-frontier tail.

Levers for moving complexity: dependency depth vs width in a subgoal graph; withholding information so it must be recovered by clarification or a tool call; adding coupled constraints; making success stricter (optimal outcome, forbidden-action avoidance, "diagnose that this is infeasible"). The stated condition is **resolvability** — withheld information must have a recovery path, or you have made the task ambiguous, not hard.

**Diversity, formally.**

$$D(\mathcal{B})=\sum_{k\in\{E,Q,I\}}w_{k}H\!\left(Z_{k}\mid d\in\mathcal{B}_{A},\,C_{z}(d)\in\mathcal{I}_{z}\right)-\lambda\,\mathrm{Red}(\mathcal{B}_{A})$$

Entropy over environment / task / interaction factor categories, computed *only over valid samples inside the useful difficulty band*, minus a behavioural redundancy penalty. Plus two reporting statistics:

$$\mathrm{Cov}_{k}=\frac{|\{b:n_{b}>0\}|}{|\mathcal{S}_{k}|},\qquad \widetilde{H}_{k}=-\frac{1}{\log|\mathcal{S}_{k}|}\sum_{b}\widehat{p}_{b}\log \widehat{p}_{b}$$

Coverage says which declared categories are non-empty; normalised entropy says whether mass is spread or piled into one. They are complementary — you can have 100% coverage with 99% of samples in one bucket. The paper also warns about **template locking**: each factor looks diverse marginally, but only occurs in a few fixed *combinations*. So report a few joint distributions (tool family × capability, state regime × required outcome), not just marginals.

## Ablation Studies and Experiments

A survey has no experiments of its own. What it does is collect the controlled comparisons scattered across the field, which is more useful than it sounds because most of them point the same way.

**Quantity loses to structure, repeatedly.**
- EnvFactory: a *smaller* set of robustly verified environments matched or beat simply scaling the number of environments. The bottleneck was verification quality, not environment count.
- DIVE: expanding tool-pool coverage and per-task toolset coverage improved out-of-distribution tool use more efficiently than drawing more samples from the same support.
- "Beyond Quantity" (code agents): at a *fixed* data budget, allocating across distinct trajectory structures beat repeated sampling.
- Recursive synthesis: solver pass rate fell monotonically across rounds of verified workflow extension — evidence that structural growth actually tracked behavioural difficulty, which is exactly the check most difficulty papers skip.

**What does not work.**

- **Structural proxies as difficulty scores.** Adding steps can make a task *easier* if the added steps are parallelisable or if the explicit decomposition reveals the plan. Extra constraints often *prune* the search space. Horizon and tool-call count are unreliable unless they reflect the shortest valid solution.
- **Maximising failure rate.** All-fail candidates give no successful trajectory, no stable reward, and no evidence the task is even solvable under the current budget. They look like frontier data and are indistinguishable from broken environments unless you gate on accuracy first.
- **Optimising repeatedly against one fixed verifier.** Two failure modes. (a) Reward hacking — trajectories that hit the terminal predicate while breaking implicit constraints, a [[Shortcut Learning in Deep Neural Networks|shortcut]] in the exact classic sense. (b) Distribution narrowing — repair loops push the accepted set toward verifier-friendly patterns, discarding legitimate alternative strategies and making the corpus *look* less diverse than the valid solution space actually is.
- **Independent generation of the four factors.** Each is individually plausible; together they drift apart (the Anchor paper on "artifact drift"). This is the argument for blueprint-first and state-first construction.
- **LLM simulators as environments.** They scale cheaply from API specs alone, but responses stay locally coherent while encoding *wrong state dynamics*. The trajectory reads fine; the world it implies is not the real one.
- **Model-name diversity as provenance diversity.** Mixing teacher models is an *intervention*, not an outcome — different models often reproduce the same task and trajectory templates. Judge committees drawn from similar models share systematic biases.
- **Persona variation in social agents.** Weak on its own. Only counts when private information, incentives, or partner policies actually change the transitions or rewards.
- **Screenshot variation in GUI agents.** Very different-looking pages can share one interaction topology, while a one-bit permission change on the *same* screen reverses the correct action. Transfer to unseen real interfaces is the only credible evidence; generated page count is not.

**What the framing reveals about which component does the work.** Trajectory accuracy is singled out as most critical, because $\tau$ is the only factor the model directly imitates — an error in $E$ or $q$ makes the sample useless, but an error in $\tau$ gets *copied*. Verifier accuracy is second, and matters most under RL, where $v$ becomes the reward and determines whether you reinforce competence or exploit a bug (compare the reward-model failures in [[Training language models to follow instructions with human feedback]] and [[Proximal Policy Optimization Algorithms]]).

## Worth Remembering

**The asymmetry is the whole point.** Accuracy is a constraint, not an objective you trade against. Complexity and diversity are objectives, evaluated only on the surviving set. Most "we made harder data" and "we made more diverse data" claims in this literature are measured *before* the gate, which makes them uninterpretable.

**Report your $z$.** Difficulty numbers are meaningless without the model, scaffold, tool access, inference budget, and rollout count. The recommended protocol is paired: fix the instance, change exactly one thing, measure the change in verified success. This is the same discipline problem as [[On the Difficulty of Evaluating Baselines]] and [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]].

**Curriculum drift is the sharpest practical caveat.** The useful band moves as the learner improves, so static generators saturate. But adaptive generators overfit to the current model's transient failures, one verifier's blind spots, one simulator's dynamics. The prescription: hold out fixed anchor sets and real environments, recalibrate periodically. A generator, learner, and verifier that co-adapt will happily converge on the same shared mistake. This is the failure mode lurking under all self-evolving agent work — see [[AgentMercury- Your Agent Can Synthesize Verifiable Environments for Business Scenarios at scale]] and [[Chain-of-Experience for Continual LLM Improvement]].

**Complexity ≠ realism.** Related but distinct. Many real workflows are routine; many synthetic tasks are hard but implausible. Recreated environments should preserve task-relevant action semantics and state transitions, not maximise detail.

**A scaling law over "effective support", not sample count.** The suggested reframing: the scaling variable is the size of the region that is simultaneously valid, learnable, and behaviourally non-redundant. That is a different curve from the token-count curves in [[Scaling Laws for Neural Language Models]] and [[Training Compute-Optimal Large Language Models (Chinchilla)]], and nobody has measured it yet. Open question worth chasing.

**Agentic pre-training changes the unit.** Once you move interaction supervision earlier than post-training, a training example no longer needs to be a complete $(E,q,\tau,v)$ record. Local state transitions, inverse-dynamics pairs, dependency completions, and reachability objectives are enough. Accuracy then means "is this transition locally correct", not "was the task completed".

**Honest limitation the authors state.** ACE is not a governance checklist — cost, latency, safety, and licensing are real constraints it deliberately ignores. It is three *generation-time* properties, chosen because they most directly determine learning value. Treat it as a lens, not a rubric.

**Practical caveat if you want to use this tomorrow.** The cheapest high-value change is ordering your filters by cost: schema → execution → state diff → step-level critic → trajectory critic → human. Reject at the earliest failing stage. For long trajectories this is not just cheaper, it is more correct, because one invalid transition poisons every observation after it.

## Links

Related: [[AgentMercury- Your Agent Can Synthesize Verifiable Environments for Business Scenarios at scale]] · [[Markov Decision Process]] · [[Dec-POMDP]] · [[Proximal Policy Optimization Algorithms]] · [[Training language models to follow instructions with human feedback]] · [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] · [[Shortcut Learning in Deep Neural Networks]] · [[Chain-of-Experience for Continual LLM Improvement]] · [[On-policy Distillation with Verifiable Reward]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[Distilling the Knowledge in a Neural Network]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[SWE Refactor Bench- Can Coding Agents Complete a Long-Horizon, Whole-Repository Stack Migration]] · [[Autonomous Mathematical Discovery in an Open-World Multi-Agent Environment]] · [[Towards Quantifying Benchmark Optimization in ASR Models]] · [[Do ImageNet Classifiers Generalize to ImageNet]] · [[On the Difficulty of Evaluating Baselines]] · [[Troubling Trends in Machine Learning Scholarship]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Uncertainty]] · [[The Bitter Lesson (essay)]]

New topics worth writing: Model Context Protocol (MCP) and tool ecosystems, reward hacking and verifier gaming, zone-of-proximal-development curriculum generation, automatic goal generation in RL, Vendi score and diversity metrics, domain randomisation for sim-to-real, LLM-as-simulator fidelity, self-evolving agent loops, agentic mid-training
