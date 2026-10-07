---
title: "Transferring the Intelligence of VLMs to Robotic Control"
authors: ["Meng-Hao Guo", "Zhe-Han Mo", "Jia-Jun Wang", "Yi Zhang", "Kejin Wang", "Yi-Xuan Deng", "Jia-Peng Zhang", "Yongming Rao", "Shi-Min Hu"]
year: 2026
arxiv: "2609.22966"
url: https://arxiv.org/abs/2609.22966
priority: Good-To-Read
read_on: 2026-09-27
tags: [paper, llm, vision]
---
## The Core Idea

Most robot models today are built by *training* a big model on robot data. You collect thousands of demonstrations, then fine-tune a vision-language model until it outputs motor commands. That is expensive, and there is growing evidence it damages the model: turning a VLM into an action predictor degrades its general reasoning and instruction following.

RoboDawn asks a different question. What if the reasoning needed for manipulation is *already in the VLM*, and the only missing piece is a way to speak to the robot?

So they do not train anything. They freeze the VLM and give it a tiny game-controller vocabulary: move 5 cm along x, rotate 30° about yaw, close the gripper, go home, done. The model looks at a camera image, says a command, the robot executes it, the model looks again. A `while` loop with a frozen model inside it — the same shape as [[Agentic Workflows#The junior researcher|ReAct]].

> [!NOTE] Intelligence transfer
> Reusing general perception/reasoning ability across changes in body, environment and task, without retraining for the new body. The claim is that a VLM trained only on web images and text already holds most of what manipulation needs. ^intelligence-transfer

The second half of the idea is the one that actually pays. A written list of commands is ambiguous — the model does not know how big "move y 10" looks on screen, or what "rotate pitch" does to a gripper. So they paste in **one worked example** of a whole episode: images, states, commands issued, what physically changed, and a short written reason for each step. No gradient update. Just context.

That single example takes RoboTwin 2.0 C2R from 53.2% to 73.6% success — beating every model that was trained on 50 demonstrations per task for all 50 tasks.

Why this could not exist before: the VLMs were not good enough. The paper's own ablation makes this brutally clear — the same harness scores 14.4% with a weak model and 73.6% with a strong one. This is a [[The Bitter Lesson (essay)|bitter-lesson]] result wearing an interface-design costume.

## The Methodology

### The loop

At round $t$ the frozen VLM $\pi_\theta$ sees:

$$(y_t, \mathbf{a}_t) = \pi_\theta(L, E, D;\ I_t, x_t, F_{t-1}, M_t)$$

In words: given the task sentence $L$, a fixed profile $E$ describing the workspace and camera, and the demonstrations $D$ — plus the current annotated images $I_t$, the measured robot state $x_t$, what happened last round $F_{t-1}$, and a running memory $M_t$ — produce a structured response $y_t$ and a list of commands $\mathbf{a}_t$.

Then the environment runs them:

$$(s_{t+1}, F_t) = \mathcal{E}_P(s_t, \mathbf{a}_t), \qquad (I_{t+1}, x_{t+1}) = \mathcal{O}_P(s_{t+1})$$

$$M_{t+1} = \mathcal{U}(M_t, \mathbf{a}_t, y_t, F_t, x_{t+1})$$

$\theta$, $E$ and $D$ never change during an episode. All adaptation comes from images, feedback and memory. The true state $s_t$ is never shown to the model — no privileged object poses.

$y_t$ is not free text. It holds an estimate of task progress, the current plan, and a small scratchpad — a structured [[Chain of Thought|chain of thought]] that is carried forward. The ablation shows this is load-bearing.

### The action vocabulary

Everything is defined relative to the **gripper interaction point (GIP)** — the midpoint between the two fingertips. Not the wrist pose the low-level controller uses. This one choice removes a lot of confusion: the thing the model reasons about is the thing that touches the object.

$$\mathcal{A} = \{\texttt{<arm> move <axis> <d>},\ \texttt{<arm> rotate <rot> <}\theta\texttt{>},\ \texttt{<arm> point <pose>},$$
$$\texttt{<arm> gripper <g>},\ \texttt{<arm> home},\ \texttt{wait},\ \texttt{done}\}$$

- `arm` ∈ {left, right}
- translation axes x, y, z; rotation axes roll, pitch, yaw — **world frame**, not gripper frame
- `point` gives orientation presets: down, forward, down45
- `gripper` takes open, close, or a number in $[0,1]$
- translations clipped at 20 cm, rotations at 90° per command

`move` shifts the GIP and keeps orientation. `rotate` turns it and keeps position. Each command becomes a full planned motion to a target GIP pose, run until the robot stops moving. Trajectory generation and control are hidden from the model entirely.

Images are annotated with a **grid** for spatial localisation. This turns out to matter more than the reasoning trace.

### The demonstration context

$$D = D_{\mathrm{prim}} \oplus D_{\mathrm{task}}, \qquad D_{\mathrm{task}} = \{\mathcal{D}^{(m)}\}_{m=1}^{N_D}$$

Two levels:

- $D_{\mathrm{prim}}$ — a **command primer**, shared across tasks, showing what each primitive does visually. Here is what "rotate yaw 45" looks like.
- $D_{\mathrm{task}}$ — full episodes. $N_D = 0, 1, {>}1$ gives zero-, one-, few-shot.

Each episode is

$$\mathcal{D}^{(m)} = \{(I_j^{(m)}, x_j^{(m)}, r_j^{(m)}, \mathbf{a}_j^{(m)}, f_j^{(m)})\}_{j=1}^{N_m}$$

image, robot state, rationale, commands, physical effect.

Three details that make this work, and they are the practical meat:

**1. Retargeting.** Raw expert trajectories are continuous low-level actions — a different language from the one the VLM speaks. Each trajectory is first reduced to end-effector waypoints plus gripper states, then each waypoint is re-expressed as a short run of move/rotate/gripper commands. The demonstration becomes something the model *could have issued itself*. Without this, the example teaches nothing about the interface.

**2. Image budgeting.** Cap of 16 images per round in context. When a long RoboDojo trajectory blows the budget, they keep images only for semantically interesting rounds — grasp, rotation, completion — and drop the boring transitions, while keeping the full text trajectory.

**3. Rationales are synthetic.** $r_j^{(m)}$ is not written by a human. A VLM reviews the recorded episode afterwards with a task-agnostic prompt and writes the reason for each round, formatted exactly like the online model's own responses. The example looks like the model's own output.

Demonstrations come from scenes disjoint from evaluation. In simulation they are the *same* trajectories used to train every baseline — a fair comparison where one side sees 2,500 demos and the other sees one.

## Ablation Studies and Experiments

### RoboTwin 2.0 C2R — 50 bimanual tasks, 10 runs each

Baselines are post-trained on 50 clean demos per task and evaluated with domain randomisation (the "C2R" = clean-to-randomised generalisation test).

| Method | Shots | Success % |
|---|---|---|
| FastWAM | Full set | 1.9 |
| GR00T-1.7 | Full set | 20.7 |
| X-WAM | Full set | 25.8 |
| $\pi_{0.5}$ | Full set | 46.0 |
| LingBot-VLA | Full set | 50.4 |
| HarnessVLA (Claude Code) | Full set | 58.4 |
| **RoboDawn (Gemini-3.8-Flash)** | 0 | 47.0 |
| **RoboDawn (Gemini-3.8-Flash)** | 1 | 62.2 |
| **RoboDawn (GPT-6 Astra)** | 0 | 53.2 |
| **RoboDawn (GPT-6 Astra)** | 1 | **73.6** |

Zero-shot already beats $\pi_{0.5}$ and LingBot-VLA. One shot beats the previous agentic SOTA by 15.2 points.

### Number of shots (Gemini-3.8-Flash)

| Shots | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| SR % | 47.0 | 62.2 | 63.6 | 65.4 | 62.7 |

**The first demo does 15 points of work; the next seven do 0.5, then go negative.** Eight shots is *worse* than four. The authors attribute this to long-context degradation — the familiar [[Lost in the Middle|lost-in-the-middle]] problem. If you were going to use this, one demo is the whole recipe.

### Which model (1-shot)

| GPT-5.6-Luna | GPT-5.6-Sol | Seed-2.1-Pro | Gemini-3.8-Flash | GPT-6 Astra |
|---|---|---|---|---|
| 14.4 | 43.2 | 45.0 | 62.2 | **73.6** |

A 59-point spread from the harness held constant. The interface is not the intelligence; the model is.

### Harness components (Gemini-3.8-Flash, zero-shot)

| Variant | SR % | Drop |
|---|---|---|
| Full | 47.0 | — |
| w/o grids | 32.4 | **−14.6** |
| w/o reasoning | 34.8 | −12.2 |
| w/o command primer | 44.0 | −3.0 |

**Grid-based spatial localisation on the images matters more than the reasoning trace.** That is the ablation worth remembering. The model's weak point is knowing *where things are in metric space*, not deciding what to do. And the command primer — the part that looks like the clever contribution — buys 3 points. The demonstration-level context is where the value is, not the primitive-level primer.

### RoboDojo — 42 tasks, 5 runs

| Method | Shots | Score | SR % |
|---|---|---|---|
| $\pi_{0.5}$ | Full set | 11.41 | 6.91 |
| GalaxeaVLA (G0.5) | Full set | 20.23 | 14.88 |
| DM0.5 | Full set | 24.90 | 19.34 |
| GPT-6 Astra (bare) | 0 | 28.97 | 22.58 |
| **RoboDawn (GPT-6 Astra)** | 0 | 39.92 | 35.67 |
| **RoboDawn (GPT-6 Astra)** | 1 | 54.63 | **47.17** |

Note row 4: the bare model with no harness gets 22.58%. The harness adds 13 points on top, the demo adds another 11.5. Both halves contribute.

### Test-time scaling

Success rises with the per-episode command budget. One-shot: 31.2% at 60 commands → 47.2% at 240. Zero-shot: 23.7% → 35.7%. More interaction converts into more success — attributed to failure recovery and memory. This is the [[Test-Time Compute|test-time compute]] dial appearing in robotics.

### Efficiency — the honest weak spot

| Model | Inference | Actions | Motion | Inference/motion |
|---|---|---|---|---|
| $\pi_{0.5}$ | 101 ms | 45.0 steps | 2.70 s | 0.037 |
| X-VLA | 143 ms | 28.6 steps | 1.71 s | 0.084 |
| LingBot-VA | 8.89 s | 22.2 steps | 1.33 s | 6.67 |
| **RoboDawn** | 9.74 s | 3.4 cmds | 2.09 s | **4.65** |

Ratio below 0.5 means you could hide inference behind motion. RoboDawn is at 4.65 — the robot spends most of its life waiting for the model to think. It issues only 3.4 commands per round though, so it is *terse*, just slow.

### Real robots, zero-shot, Gemini-3.8-Flash

| Block in basket (Franka) | Block stacking (Franka) | Cloth folding (Piper) |
|---|---|---|
| 9/10 | 5/10 | **0/10** |

### What did not work

- **Cloth folding: 0/10.** The authors' hypothesis is that folding needs heavy end-effector rotation, and rotations are under-represented on the web. Several near-misses — folds were not neat enough to count.
- **Eight demonstrations hurt** relative to four.
- **Rotations generally.** Stated as a named limitation: VLMs handle translation far more reliably than rotation and 3D orientation. ICL narrows this but does not close it.
- **Three named failure modes on RoboDojo:** (a) insufficient precision — right intent, fails at insertion into a small target; (b) IK/collision — semantically sound plan, the realised motion knocks the scene over, a gap between high-level plan and physically safe execution; (c) wrong success judgment — pours liquid, declares `done` before enough transferred. That last one is a [[Reward Hacking|Goodhart]]-adjacent problem: the model's notion of "finished" and the benchmark's do not match.

## Worth Remembering

**The comparison is generous to RoboDawn in one way and harsh in another.** Harsh: baselines get 2,500 demonstrations, RoboDawn gets one. Generous: RoboDawn is a frontier closed model doing multi-second reasoning per step, while $\pi_{0.5}$ runs at 101 ms. These are not the same product. One is a research result about where intelligence lives; the other is something you could put on a factory line.

**The strongest evidence for the paper's thesis is the model ablation, not the demo ablation.** 14.4% → 73.6% by swapping the VLM with everything else fixed means the manipulation ability is inside the pretrained weights. The interface is a straw for drinking it.

**But the grid ablation complicates the story.** If web-scale pretraining had truly given the model embodied spatial ability, drawing a grid on the image should not be worth 14.6 points. What the grid does is give the model a *coordinate system it can read off pixels*. The intelligence transfers; the metric grounding has to be handed over explicitly.

**One demo is the whole ICL story.** 15 points from the first, 3 from the next seven, negative from the eighth. If you build anything like this, budget for one very carefully constructed example, not a growing library.

**The retargeting step is the reusable engineering idea.** An expert trajectory in the wrong action space is useless as context. Rewriting it into commands the model could itself have emitted — plus VLM-written rationales in the model's own response format — is what makes the example legible. Same principle as [[In Context Learning|in-context learning]] generally: the demonstration must match the output distribution you want.

**Connections.** The "training on actions degrades general ability" claim (Hancock et al., Yang et al.) is [[Fine-Tuning#The failure modes 🪤|catastrophic forgetting]] in a robotics costume, and it is the load-bearing motivation for the whole paper — worth checking independently, since it is cited rather than demonstrated here. The closed-loop structure is [[Agentic Workflows|ReAct]] with a physical environment; the memory update $\mathcal{U}$ is [[Memory|working memory]]; the "observe, act, observe again" rhythm is a [[Feedback Loop|closed loop]] with a very slow controller. The `done` command and its misfiring is an [[Agent Evaluation|agent evaluation]] problem.

**Concurrent work.** Show-Harness (Chen et al. 2026b) independently shows frontier VLMs doing closed-loop control through a semantic action interface. RoboDawn's differentiator is the ICL design. Two groups landing on the same harness idea at once is itself evidence that the capability threshold was just crossed.

**Open questions.** Does the one-shot gain survive when the demo scene differs a lot from the test scene — here demos are from clean scenes and tests are domain-randomised, which is a real gap but a controlled one. Could the grid be replaced by depth or a learned localiser, and would that close the precision failures? And: is the frozen-VLM route actually *complementary* to VLAs, as the authors claim — a high-level planner issuing semantic commands to a trained low-level policy — or does the 4.65 inference/motion ratio make that hybrid unworkable in practice?

## Links

Related: [[In Context Learning]] · [[Agentic Workflows]] · [[Chain of Thought]] · [[Test-Time Compute]] · [[Fine-Tuning]] · [[Lost in the Middle]] · [[The Bitter Lesson (essay)]] · [[Imitation Learning]] · [[Memory]] · [[Feedback Loop]] · [[Agent Evaluation]] · [[Reward Hacking]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[Think Like a World Model, Act Like a VLA- Distilling World-Model Representations into Compact Robot Policies]] · [[EXIMO- VLM Guided Exploration of VLA Policies]] · [[In-Context Robot Learning with VLM Agents]]

New topics worth writing: Vision-Language-Action models, Semantic action interfaces for robots, RoboTwin 2.0 and RoboDojo benchmarks, Gripper interaction point, Trajectory retargeting for in-context demonstrations, Inverse kinematics failure in agentic control, Inference-to-motion ratio as a robotics efficiency metric
