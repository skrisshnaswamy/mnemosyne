---
title: "The Past Frames the Future: Memory for Autoregressive Video Generation"
authors: ["Chen et al."]
year: 2026
arxiv: "2609.28466"
url: https://arxiv.org/abs/2609.28466
priority: Good-To-Read
read_on: 2026-09-28
tags: [paper, transformers, llm, diffusion, vision, theory]
---
## The Core Idea

Video models are getting good at generating a few seconds. The next frontier is generating *forever* — a stream you can keep extending, walk around in, and interact with. The natural way to do that is **autoregressive (AR) generation**: produce the next chunk of video conditioned on everything you already made, then repeat.

The problem is arithmetic. History grows without bound, but attention costs $O(n^2)$ and the [[KV Cache]] costs $O(n)$ memory. So every real system truncates: it only looks at the last $W$ units. That means anything you established 500 frames ago — a character's face, the layout of a room, the fact that you *opened a door* — has physically left the input by the time it matters again.

This survey's actual contribution is not a method. It is a **definition that makes "memory" testable**, plus a taxonomy that lets you compare mechanisms that currently hide behind different words (cache, context, history, state, bank, anchor).

> [!NOTE] Memory (operational definition) ^memory-operational
> A state is *memory* if it persists across autoregressive steps and **deleting or editing it changes later outputs**, at a point where the original evidence is no longer in the local context. Not defined by module name, carrier type, or window size.

That last clause is the sharp bit. It refuses to draw a hard line between "long context" and "memory". Recent frames fed densely to every step are just **active context**. The same frames become **memory** the moment somebody *manages* them — selects, compresses, indexes, retrieves, consolidates, or revises them. Memory is a verb, not a buffer.

The second insight, stated as the paper's thesis: **capacity is not the metric**. A retained state is only useful if it is (a) accurate, (b) accessible when needed, and (c) causally influential on the output. You can hold a million tokens of history and still fail all three.

The survey then organises everything through five questions: **Forms** (what carries history), **Functions** (what must be preserved), **Operations** (the read/write/update lifecycle), **Learning** (how memory behaviour is trained under closed-loop rollout), **Evaluation** (how to prove memory is actually working).

## The Methodology

### The setup: outer rollout vs inner generation

Abstract a video into $N$ ordered **visual units** $\mathbf{y}_{1:N}$. A unit can be a discrete token block, one latent frame, or a multi-frame chunk. The AR factorisation is the usual thing:

$$p_\theta(\mathbf{y}_{1:N}\mid \mathbf{c}) = \prod_{n=1}^{N} p_\theta(\mathbf{y}_n \mid \mathbf{y}_{<n}, \mathbf{c})$$

with $\mathbf{c}$ the external condition (text, reference image, camera path, controller input).

The clean separation the paper insists on: **"autoregressive" describes the outer loop over units, not the architecture that makes each unit.** Two families fill in the inner step differently.

**Discrete token AR** (VideoGPT, VideoPoet, Emu3) quantises frames into tokens via [[VQ-VAE]] and factorises again *inside* the unit:

$$p_\theta(\mathbf{y}_n \mid \mathbf{y}_{<n},\mathbf{c}) = \prod_{\ell=1}^{L} p_\theta(\mathbf{z}^{(n)}_\ell \mid \mathbf{z}^{(n)}_{<\ell}, \mathbf{y}_{<n}, \mathbf{c})$$

**Continuous frame/chunk generation** runs a [[Diffusion Models|diffusion]] or [[Flow Matching|flow]] trajectory for the whole unit:

$$\frac{d\mathbf{y}_n^{(\tau)}}{d\tau} = f_\theta\!\left(\mathbf{y}_n^{(\tau)}, \tau, \mathbf{y}_{<n}, \mathbf{c}\right)$$

Two clocks, and they get confused constantly: $\tau$ is the *inner* denoising time, $n$ is the *outer* AR step. Keep them separate in your head.

### The bottleneck

Practical generators condition on a window:

$$\mathbf{C}_n = \mathbf{y}_{\max(1,\,n-W)\,:\,n-1}$$

so $p_\theta(\mathbf{y}_n \mid \mathbf{y}_{<n},\mathbf{c})$ gets approximated by $p_\theta(\mathbf{y}_n \mid \mathbf{C}_n,\mathbf{c})$. [[Sparse Attention]] and [[Flash Attention]] make this cheaper but do not remove it — the [[KV Cache]] still grows linearly and accelerator memory is finite.

For interactive settings, actions enter:

$$p_\theta(\mathbf{y}_{1:N}\mid\mathbf{c},\mathbf{a}_{1:N}) = \prod_{n=1}^{N} p_\theta(\mathbf{y}_n \mid \mathbf{y}_{<n}, \mathbf{a}_{\le n}, \mathbf{c})$$

A distinction worth stealing: **observation-changing** controls (move the camera in a static room) versus **state-changing interventions** (open the door, move the cup). The first stresses spatial preservation. The second creates a genuine causal debt — the door must *stay open* after the action has scrolled out of the window.

### Memory-conditioned generation

Add a persistent state $\mathbf{M}_n$ alongside the window:

$$p_\theta(\mathbf{y}_{1:N}\mid\mathbf{c}) = \prod_{n=1}^{N} p_\theta(\mathbf{y}_n \mid \mathbf{C}_n, \mathbf{M}_n, \mathbf{c})$$

with the goal being $p_\theta(\mathbf{y}_n \mid \mathbf{C}_n,\mathbf{M}_n,\mathbf{c}) \approx p_\theta(\mathbf{y}_n \mid \mathbf{y}_{<n},\mathbf{c})$.

In the simplest case $\mathbf{M}_n = \mathcal{A}(\mathbf{y}_{<n-W},\mathbf{c})$ — a compression of the truncated prefix. But $\mathbf{M}_n$ can be an attention cache, a retrieval store, a recurrent hidden state, a structured scene graph, or an adapted parameter delta.

> [!NOTE] Evidence vs. world state ^evidence-vs-state
> $\mathbf{M}_n$ can hold two different things. **Evidence**: what was observed or generated in the past. **World state**: what is *currently believed true*, which must be revised when things change. A door you opened is world state; the frame showing you open it is evidence. Systems that only store evidence struggle to represent revision.

That second reading is exactly a **belief state** — compare [[Beliefs#^belief-state-trick|belief states]], the [[Bayes Filter]], and [[POMDP]]. The framing is the same: you cannot see the whole world, so you carry a sufficient statistic and update it. The paper explicitly nods to Neural Turing Machines and Memory Networks as ancestors.

### The lifecycle — six operators

This is the most reusable piece. Per AR step:

**Read side (before generating):**

$$\mathbf{q}_n = Q_\theta(\mathbf{C}_n,\mathbf{c}) \qquad \mathbf{r}_n = \mathcal{R}_\beta(\mathbf{q}_n, \mathbf{M}_n) \qquad \mathbf{h}_n = \mathcal{I}_\eta(\mathbf{C}_n, \mathbf{r}_n, \mathbf{c})$$

Form a query, retrieve, integrate. Then generate from $p_\theta(\mathbf{y}_n \mid \mathbf{h}_n)$.

**Write side (after generating):**

$$\mathbf{w}_n = \mathcal{W}_\alpha(\mathbf{y}_n, \mathbf{C}_n, \mathbf{c}) \qquad \widetilde{\mathbf{M}}_{n+1} = \mathcal{U}_\phi(\mathbf{M}_n, \mathbf{w}_n) \qquad \mathbf{M}_{n+1} = \mathcal{G}_\gamma(\widetilde{\mathbf{M}}_{n+1}; B)$$

Extract a write candidate, update, then **enforce a budget $B$** via eviction, compression, or revision.

Every existing method is one filling-in of these six slots. A FIFO queue of latents is $\mathcal{U}$ = append, $\mathcal{G}$ = drop-oldest, $\mathcal{R}$ = identity. A retrieval bank is a nontrivial $\mathcal{R}$. A recurrent hidden state folds $\mathcal{U}$ and $\mathcal{G}$ into one matrix update. If you have ever built a [[RAG]] pipeline, this is the same skeleton with video units instead of text chunks.

### Forms — the four carriers

> [!NOTE] Memory carrier ^memory-carrier
> The representational object in which history is retained and exposed to later generation. Taxonomy is by *what the object is*, not by what problem it solves.

**1 · Visual memory** — observation-aligned evidence, traceable back to a specific past frame or interval:

$$\mathbf{M}_n^{\mathrm{vis}} = \{\mathbf{m}_i^{\mathrm{vis}} = E_{\mathrm{vis}}(\mathbf{y}_i) \mid i \in \mathcal{I}_n^{\mathrm{vis}}\}$$

Two sub-spaces.

*Pixel space* stores renderable RGB. A unit is $\mathbf{m}^{\mathrm{pix}}_i = \mathbf{x}_{s_i:u_i}$ — a single frame when $s_i = u_i$, a clip when $s_i < u_i$. Frames preserve identity, texture, viewpoint. Clips additionally preserve short-range motion, ordering, and interaction dynamics — at higher cost, and with the risk that old motion is a *stale prior* once the action changed. Three retention strategies: dense recent frames (that is really active context), sparse anchor frames (stable long-range reference, e.g. ConsistI2V), and retrieval by visual or geometric relevance (Context-as-Memory, VMem).

*VAE space* stores latents from a video autoencoder:

$$\mathbf{m}_i^{\mathrm{vae},0} = E_{\mathrm{VAE}}(\mathbf{x}_{s_i:u_i}), \qquad D_{\mathrm{VAE}}(\mathbf{m}_i^{\mathrm{vae},0}) \approx \mathbf{x}_{s_i:u_i}$$

Diffusion systems sometimes keep them *noised* at level $\tau$:

$$\mathbf{m}_i^{\mathrm{vae},\tau} = \alpha_\tau \mathbf{m}_i^{\mathrm{vae},0} + \sigma_\tau \boldsymbol{\epsilon}_i$$

The noise level does not change the category — see [[Forward Diffusion Process]] for where $\alpha_\tau,\sigma_\tau$ come from. Patterns here: sequential carry-over (LVDM), bounded windows/queues (FIFO-Diffusion, PA-VDM), and retrieval of non-local latent blocks (LongLive-RAG). Cheaper than RGB and already in the generator's native space (no decode/re-encode round trip), but lossy and **coupled to a specific autoencoder** — swap the VAE and your stored memory is garbage. Relevant to [[Latent Diffusion#^vae-is-the-ceiling|the VAE ceiling]].

**2 · Implicit state memory** ($\mathbf{M}_n^{\mathrm{imp}}$) — model-native latent state with no imposed semantics: retained/refreshed KV entries, recurrent hidden states ([[Linear Attention|linear-attention]] / [[State-Space Model|SSM]]-style), compressed summary tokens.

**3 · Explicit state memory** ($\mathbf{M}_n^{\mathrm{exp}}$) — structured, interpretable variables: entities, poses, layouts, 3D scene state, relations, events. This is the only form where you can *edit* the world directly (WorldMem, VMem).

**4 · Adaptive parametric memory** ($\mathbf{M}_n^{\mathrm{par}}$, tracked as $\Delta\theta_u$) — history baked into weights or adapters that get updated during the rollout. Effectively [[LoRA]] as a memory device.

The comparison axes the survey tabulates across all four: **storage growth** (fixed / bounded / linear / scene-scaled), **unit structure** (itemized / distributed / structured / entity / hybrid), **access key**, **interpretability**, **editability without retraining**, **model coupling**. Those six columns are the checklist to bring to any new "memory for video" paper.

## Ablation Studies and Experiments

**There are none — this is a survey.** No benchmark table, no new numbers, no controlled comparison. Treat every trade-off below as *the authors' synthesis of other people's results*, not as measured evidence. That is the main epistemic caveat on the whole document.

What the taxonomy does buy is a clean failure-mode catalogue for memoryless rollouts (window-only conditioning):

| Failure | What it looks like |
|---|---|
| **Entity forgetting** | Object leaves frame, comes back wrong / duplicated / re-identified as a similar instance; face and body drift |
| **Appearance drift** | Entity stays visible but texture, colour, lighting, style slowly slide |
| **Spatial inconsistency** | Re-enter a room, get a different layout or geometry |
| **Dynamics degradation** | Locally plausible motion, globally broken kinematics; freezing, phase errors, contact violations |
| **Semantic drift** | Contradicts or repeats prior events; loses narrative progression; never finishes a long action |
| **Causal / state inconsistency** | Opened door closes itself; moved object teleports back; environment resets |

The honest caveat the authors put in themselves, and it is the best sentence in the paper: **these symptoms are not proof of a memory problem.** The same artefacts can come from bad dynamics modelling, bad control execution, or bad rendering. So the symptom list is a set of *candidate* attributions, and the evaluation section exists specifically to separate "this model has no memory" from "this model has memory and is otherwise bad". Compare the general problem in [[Evaluating Generative Models#^fid-blind-spots|FID's blind spots]] — a global quality score cannot tell you which mechanism failed.

**What the survey argues does not work:**

- **Just enlarging the context window.** It increases what is *available*, but not what is *managed*. Quadratic attention plus linear cache means the ceiling arrives regardless. And within a long window, relevance is not flat — see [[Lost in the Middle]].
- **Treating the prompt as memory.** A static text condition is *not* memory under their definition, because nothing evolving is maintained. It only becomes memory if the system derives and updates state from it during the rollout.
- **Naively storing your own generations.** Because written evidence is often *generated* evidence, artefacts get persisted and re-fed. This is [[Auto-regressive models#^ar-factorisation|AR error accumulation]] with a longer memory: now the mistake survives eviction. Same family of problem as [[Imitation Learning#^errors-compound|compounding error]] and the teacher-forcing/self-rollout mismatch.
- **Dense retention as a default.** Redundant and linear in cost. But sparse and retrieved references trade that for *staleness and misalignment* — the anchor no longer describes the current state.
- **Frame memory for motion.** A single frame records a state, not the process that produced it. If you need dynamics, you need clips or explicit dynamic state.

## Worth Remembering

**The definition is the takeaway.** "Does removing this state change later output after its evidence has left the window?" is an intervention test. It is falsifiable, carrier-agnostic, and it immediately disqualifies a lot of papers that relabel a KV cache as memory. Keep it.

**Long context and memory are a spectrum, and the axis is *management*.** Selection, compression, indexing, retrieval, consolidation, revision. Each of those is an operator slot ($\mathcal{W}, \mathcal{R}, \mathcal{U}, \mathcal{G}, \mathcal{I}$) you can specify independently. That is genuinely a good decomposition for reading the literature.

**The text I have is incomplete.** Sections 4–8 — Functions, Operations, Learning, and Evaluation — are signposted in the introduction but the body cuts off part-way through §3 (Forms). So: the learning story (memory-state distributions $q_{\mathrm{TF},n}$ under teacher forcing vs $q_{\theta,n}$ under self-rollout vs $q_{\mathrm{aug},n}$ under perturbation, and the four losses $\mathcal{L}_{\mathrm{pred}}, \mathcal{L}_{\mathrm{mem}}, \mathcal{L}_{\mathrm{dyn}}, \mathcal{L}_{\mathrm{aware}}$) is only visible via the notation table. That table is a useful skeleton on its own: it says out loud that **memory must be trained on the state distribution it will actually see at deployment**, which is the closed-loop / [[On-Policy vs Off-Policy|on-policy]] version of the same distribution-shift argument you already know.

**The distribution-shift angle is the part worth your time.** $q_{\mathrm{TF},n} \ne q_{\theta,n}$ is exactly the train/deploy mismatch that makes [[Offline RL]] and [[Off-Policy Evaluation]] hard, wearing video clothing. If you train the memory operators on ground-truth histories and deploy them on self-generated ones, the operators see states they were never fit on.

**Scope exclusion to note:** they deliberately do *not* cover world-action models comprehensively ([[GameWAM- A World Action Model for Video Games]], LingBot-VA), on the grounds that those optimise action prediction and get memory incidentally through persistent KV. Fair, but it means the survey under-covers the setting where memory is most obviously evaluable — because there you have a task reward.

**Practical caveats if you were building this:**

- Pick your carrier by the six table columns, not by vibes. The ones that bite in production are **storage growth** and **model coupling**. Linear growth kills a streaming service; VAE coupling kills your ability to upgrade the generator without rebuilding every stored memory.
- Only explicit state memory is editable. If your product needs "the user moved the cup and it must stay moved", latent caches will not give you that guarantee.
- The budget operator $\mathcal{G}_\gamma$ is where the real engineering is, and it is the least studied. Eviction policy *is* your memory policy.
- Any benchmark you build must separate memory failure from generation failure. Otherwise you are measuring [[Evaluating Generative Models|video quality]] and calling it memory.

**Open questions the authors flag:** composable and resource-aware memory architectures, *trustworthy state updating* (how do you avoid writing a wrong belief and then defending it?), self-rollout learning, and standardised evaluation. The second one is the interesting one — it is a credit-assignment and calibration problem, not an architecture problem.

## Links

Related: [[Video Diffusion]] · [[Auto-regressive models]] · [[KV Cache]] · [[Long Context]] · [[Context Window]] · [[Memory]] · [[Beliefs]] · [[POMDP]] · [[Bayes Filter]] · [[State-Space Model]] · [[Linear Attention]] · [[Latent Diffusion]] · [[VQ-VAE]] · [[Flow Matching]] · [[RAG]] · [[Lost in the Middle]] · [[Sparse Attention]] · [[Imitation Learning]] · [[Evaluating Generative Models]] · [[WorldCrafter- Consistent Video World Model with Implicit 3D-aware Memory]] · [[Video DeltaNet- A Video-Native Hybrid Attention for Livestream Video Generation]] · [[GameWAM- A World Action Model for Video Games]] · [[Register Tokens for Bounded-State Reasoning in Diffusion Language Models]]

New topics worth writing: memory lifecycle operators (write/read/update/manage/integrate), FIFO-Diffusion and rolling-window video generation, WorldMem and editable explicit scene state, VMem surfel-indexed frame retrieval, teacher-forcing vs self-rollout memory-state distribution shift, interactive world models and state-changing interventions, memory-revealing evaluation protocols for long video
