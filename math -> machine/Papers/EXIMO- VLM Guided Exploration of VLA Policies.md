---
title: "EXIMO: VLM Guided Exploration of VLA Policies"
authors: ["Bhavya Sukhija", "Oliver Groth", "Mohit Shridhar", "Tim Hertweck", "Michael Bloesch", "Markus Wulfmeier", "Abbas Abdolmaleki", "Martin Riedmiller"]
year: 2026
arxiv: "2608.19891"
url: https://arxiv.org/abs/2608.19891
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, llm, rl, diffusion, vision]
---
## The Core Idea

You have a robot policy that already knows how to pick things up and put them down. It was trained by copying humans — hundreds of hours of teleoperation. It works well on the exact phrasings and exact task shapes it saw. It falls over on two kinds of new task:

1. **Long tasks** — "put the plate *and* the bowl on the rack" — which need two or three of its known skills chained together.
2. **Reasoning tasks** — "put the item a monkey can eat into the bowl" — where the target object is described, not named.

The usual fixes are both bad. Collect more teleoperation data: expensive, and you repeat it for every new task. Use [[Markov Decision Process|RL]] from scratch: the reward is "did you succeed", which is almost never 1, so exploration is hopeless on a long task.

The insight is that the two failures above are **semantic**, not **motor**. The robot's hands are fine. Its brain does not know that a monkey eats bananas, or that "plate and bowl" means two sequential pick-and-places. And there is already a model sitting on a shelf that knows exactly those things: a large vision-language model like Gemini.

So: put the VLM in charge of the robot policy at *data collection time only*. The VLM looks at camera images, thinks, and emits one short instruction — "pick up the blue plate with your left hand". The robot policy executes it, because it was trained to follow natural language. The VLM watches, then emits the next instruction. This closed loop solves tasks the robot alone cannot.

Then the trick that makes it more than a demo. Keep only the episodes that succeeded. Train the robot policy on those episodes — but **relabel the language goal**. Instead of conditioning each timestep on the VLM's sub-instruction ("pick up the blue plate"), condition it on the *original whole task* ("put the plate and bowl on the rack"). The multi-step plan gets compiled into the small policy's weights. At deployment you throw the VLM away: no latency, no API calls, and — surprisingly — *higher* success than keeping the VLM in the loop.

Finally, use online RL on top of that, which is now tractable because the policy already succeeds sometimes, so the sparse reward is actually reachable.

Three stages, hence the name: **Ex**plore, **Im**itate, **O**ptimize.

> [!NOTE] VLA (vision-language-action model) ^vla
> A policy that takes camera images plus a natural-language goal and outputs robot actions. Here: Gemini Robotics On-Device (GROD), 3B parameters, a [[An Image is Worth 16x16 Words (ViT)|PaliGemma]] vision-language backbone with a diffusion head that predicts a chunk of future actions.

> [!NOTE] Orchestration ^vlm-orchestration
> A big model steering a small model *in natural language*, with no code API, no fixed skill list, and no grounding module. Possible only because the small model was already trained to take free-form language.

## The Methodology

### Stage 1 — Explore

Base policy $\bm{\pi}^{VLA}(\bm{a} \mid \bm{s}, g)$: state $\bm{s}$ (camera images + proprioception), goal $g$ in text, action $\bm{a}$.

Orchestrator $\bm{\pi}^{VLM}(g_t \mid \bm{s}_{\leq t}, g)$: given the *history* of states and the overall goal, emit the next sub-goal.

The prompt gives the VLM images from several timesteps (0, 50, 100, 150, 200) and several camera viewpoints, plus the interaction history. The VLM must reason inside `<think></think>` and emit exactly one instruction inside `<answer><instruct></instruct></answer>` — very close to [[Chain-of-Thought Prompting Elicits Reasoning in LLMs|chain-of-thought]] with a structured output slot. Rules in the prompt: one action per instruction, no chaining with "and"/"then", always give colour and object type, disambiguate with spatial language, do not refer to robot parts. If the task is done, emit `<score>0</score>`.

Rollouts run closed-loop — the VLM can change its mind mid-episode. At episode end, a **ground-truth environment success detector** decides whether the rollout goes in the buffer. Failures are discarded. This filtering is doing real work; see the ablations.

### Stage 2 — Imitate

Take the filtered buffer of $(\bm{s}, g, \bm{a})$ tuples. Train with the **same behaviour-cloning objective the VLA was originally trained with** — predict the action chunk $\bm{a}_{0:K-1}$ from $(\bm{s}, g)$, gradient descent on the existing weights. No new loss, no new head.

The only non-obvious bit: $g$ in the training tuple is the *task* goal, not the VLM's sub-goal. That is what turns a two-model system into a one-model system. It is a form of [[Distilling the Knowledge in a Neural Network|distillation]] where the teacher is "small model + planner + filter" and the student is the small model alone.

### Stage 3 — Optimize

Doing RL directly on a 3B diffusion-head policy is painful: huge, and the action distribution is implicit (you cannot cheaply write $\log \pi(\bm{a}|\bm{s})$ for a [[Denoising Diffusion Probabilistic Models|diffusion]] sampler). So freeze the VLA and learn a small **residual policy** instead:

$$\bm{a} = \bm{a}^{\text{VLA}} + \Delta\bm{a}, \qquad \Delta\bm{a} \sim \bm{\pi}^{\text{ref}}(\cdot \mid \bm{s}, g, \bm{a}^{\text{VLA}})$$

The residual policy sees the VLA's proposed action as part of its input. The residual MDP has state $\bm{x} = (\bm{s}, \bm{a}^{\text{VLA}})$ and transitions

$$\bm{s}' \sim T(\cdot \mid \bm{s}, \bm{a}^{\text{VLA}} + \Delta\bm{a}), \qquad \bm{a}'^{\text{VLA}} \sim \bm{\pi}^{VLA}(\cdot \mid \bm{s}', g)$$

Objective is discounted probability of success — a pure sparse indicator reward:

$$J(\bm{\pi}^{\text{ref}}, \rho) = \mathbb{E}_{\bm{\pi}^{\text{ref}}, \bm{s}_0 \sim \rho}\left[\sum_{t=0}^{\infty} \gamma^t \mathbb{1}_{\bm{s}_t \in \text{Success}(g)}\right]$$

Optimised with **MPO** (Maximum a posteriori Policy Optimisation) — an off-policy actor-critic that does policy improvement as a [[KL Divergence|KL]]-constrained reweighting of sampled actions, so it is in the same family as [[Soft Actor-Critic]] and the trust-region idea behind [[Proximal Policy Optimization Algorithms|PPO]].

### Setup

ALOHA bimanual robot, in simulation. 22 tasks (T2–T23), four families: dish-placement chains (bowl + glass on rack), reasoning ("the object you pour coffee in"), spatial (left vs right compartment of a caddy), and multi-object chains (scissors + screwdriver in caddy). Evaluation: 1000 episodes per task, terminated on success. RL curves averaged over 5 seeds, mean ± 2 standard errors.

## Ablation Studies and Experiments

The whole paper is an ablation of its own three stages. Numbers are reported as figures, not tables, so the qualitative deltas are what survive.

**Does orchestration help? (Explore)** VLM-orchestrated GROD beats bare GROD on success rate across the 22 tasks, with the biggest gaps on chained tasks (PlateBowlOnRack) and reasoning tasks (BananaInBowl-Reasoning). Time-to-success stays about the same. Crucially, **episode length drops** — the orchestrated agent finishes or fails faster, so each unit of wall-clock buys more usable data.

**Does distilling help? (Imitate)** GROD + SFT beats *both* bare GROD and orchestrated GROD. That the student beats its own teacher is the headline surprise. Two reasons offered: the SFT set contains only successful, filtered episodes (high-quality supervision, a bit like reject-sampling in [[Training language models to follow instructions with human feedback|RLHF]] pipelines), and distillation collapses a multi-step plan into one goal-conditioned policy that no longer has to survive VLM latency or a wrong sub-goal mid-episode.

**Does RL on top help? (Optimize)** GROD + SFT + residual RL starts higher, learns faster, and *converges* higher than residual RL on the base GROD — even though the base model was given **extra environment steps** to compensate for the episodes spent in the Explore phase. So the gain is not just "more data"; the SFT policy is in a better place in policy space. On several tasks, SFT alone beats base-GROD-plus-RL despite the latter collecting far more data.

**Free-form vs grounded instructions.** They restricted the VLM to only `pick the X` / `put the X in the Y` and compared to unrestricted natural language. Performance was **comparable** on the five tasks tested. Reading: GROD genuinely follows free-form language, so the classic LLM-planner grounding machinery (masked-LM projection onto a skill set, affordance scoring as in SayCan, a second LLM for API translation) is unnecessary. That is a real simplification, not a small one.

### What did not work

Two negative results, both in the appendix, both about the same failure mode.

1. **Distilling the orchestrator into the residual policy instead of the VLA.** They collected orchestrated data with the base VLA, defined $\Delta\bm{a} = \bm{a}^{VLA-VLM} - \bm{a}^{VLA}$, and pretrained the residual with advantage-weighted BC (AWBC). Offline, the residual policy improved with dataset size. Then online RL started — and it learned **more slowly than pure online RL with no pretraining at all**. Cause: distribution shift between the offline orchestrated data and the online rollouts.

2. **Keeping the VLM in the loop during online RL.** They used the VLM on a fraction $p$ of training episodes to get the best of both. During data collection the orchestrated episodes did succeed more often (bottom row of Fig. 7). But at evaluation — where there is no VLM — the $p=0$ agent won. The residual policy trained to correct $\bm{a}^{VLA-VLM}$ learns nothing useful about correcting $\bm{a}^{VLA}$; those $p$ fraction of transitions are near-useless, or worse.

The lesson is sharp: **the VLM's guidance must be absorbed by the model that gets conditioned on language (the VLA), not by a downstream correction module.** A residual controller sits below the level at which the VLM's contribution is meaningful. This is a clean instance of the [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives|action distribution shift]] problem — the offline data is from a different action-generating process than the one you will deploy.

## Worth Remembering

- **The success detector is ground truth and hand-provided.** This is the load-bearing assumption. The filtering step is what makes the SFT data good. The authors flag replacing it with a VLM judge as future work; until then, "no teleoperation needed" quietly becomes "no teleoperation, but you still hand-write a success check per task."
- **Everything is in simulation.** ALOHA sim, 22 tasks. No real-robot numbers.
- **Environment resets are assumed free.** Also flagged as future work — orchestrate the VLM to "undo the task" and close the loop fully. For reversible tasks this would give a genuinely autonomous learn-forever system: VLM as planner, reward model, and reset controller.
- **Sparse reward, no shaping.** The reward is literally $\mathbb{1}_{\bm{s}_t \in \text{Success}(g)}$. RL works only because the SFT policy already reaches success sometimes. This is the paper's real argument for stage ordering: exploration is not a search problem here, it is a *prior* problem, and the VLM supplies the prior.
- **The framing the authors give at the end is the useful one:** Explore + Imitate is a *self-distillation post-training recipe*. You take extra context available at training time (the VLM's task decomposition) and fold it into a model that will not have that context at test time. Same shape as chain-of-thought distillation, same shape as context distillation. Their setup is off-policy (collect once, then SFT); on-policy variants are the obvious next step.
- **Practical caveat if you want to copy this:** the VLM sees images from several past timesteps at once (0, 50, 100...) plus interaction history. That is expensive per call, and the VLM is queried repeatedly per episode. The economics only work because you pay it once, during collection, and never at deployment.
- **Open question.** The student beating the teacher is attributed to filtering plus compilation. It would be worth separating those: what happens if you SFT the base VLA on filtered *non-orchestrated* successful rollouts? That ablation is missing, and it is the one that isolates how much of the gain is the VLM versus how much is just rejection sampling on your own policy.

## Links

Related: [[Distilling the Knowledge in a Neural Network]] · [[Distillation]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] · [[Soft Actor-Critic]] · [[Proximal Policy Optimization Algorithms]] · [[Markov Decision Process]] · [[Denoising Diffusion Probabilistic Models]] · [[Foundation Models]] · [[An Image is Worth 16x16 Words (ViT)]] · [[KL Divergence]] · [[Training language models to follow instructions with human feedback]] · [[The Bitter Lesson (essay)]] · [[In Context Learning]]

New topics worth writing: MPO (Maximum a posteriori Policy Optimisation), residual policy learning, vision-language-action models, behaviour cloning, action chunking, SayCan and LLM grounding for robotics, hierarchical policies, advantage-weighted regression (AWR/AWBC), on-policy distillation, ALOHA bimanual manipulation
