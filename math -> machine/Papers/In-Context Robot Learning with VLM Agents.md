---
title: "In-Context Robot Learning with VLM Agents"
authors: ["Cheng et al."]
year: 2026
arxiv: "2609.19138"
url: https://arxiv.org/abs/2609.19138
priority: Good-To-Read
read_on: 2026-09-26
tags: [paper, llm, vision]
---
## The Core Idea

Take a general-purpose vision-language model off the shelf. Do not fine-tune it on robot data. Give it a text instruction, live camera images, and some **context** — a video of a human doing the task, a goal photo, or a log of what it already tried. Then let it pick robot moves by calling tools, one move at a time, with real execution feedback coming back after each call.

That is `GPT-Policy`. The claim is narrow and worth stating precisely: a model with **no robot-specific training** and **no gradient updates at test time** can still use a demonstration to do better at a physical task.

> [!NOTE] Robotic in-context learning
> Adapting robot behaviour from demonstrations, examples, or interaction experience supplied at test time — with frozen weights and no persistent parameter change. The model must decide *which* bit of context matters for *this* decision, then turn it into motion. ^robotic-icl

Why this did not exist before: until recently VLMs could not reliably emit structured tool calls in a long closed loop, and robot demonstration data was assumed to need a trained policy to consume. The interesting bit is the **cross-embodiment transfer**. A human video contains no robot joint angles, no end-effector poses, nothing numeric. The model watches a person block a towel with one hand and slide the other underneath, then invents its own robot poses to do the same thing. Success goes 0/3 → 2/3.

The second finding is the honest one, and it is the part to remember: **context fixes understanding, not execution**. The robot now knows what to do and still fails at making contact, still cannot verify its own outcome, and still crashes its two arms into each other. The authors report the collisions rather than hiding them.

This sits alongside [[In Context Learning]] — the same "learn from the prompt, change no weights" phenomenon — but moved into a physical loop where the environment answers back.

## The Methodology

Three parts: a context compiler, a frozen VLM, and a constrained controller.

**The decision rule.** At step $t$ the model sees the instruction $T$, the context $c_t$, the current observation $o_t = (I_t, s_t)$ (images plus robot state), and the result of the last tool call $f_{t-1}$. It emits one tool request $a_t = (u_t, v_t)$ — a tool name and an arguments object:

$$a_t \sim \pi_\theta(\cdot \mid T, c_t, o_t, f_{t-1}), \qquad (o_{t+1}, f_t) = \mathcal{E}(a_t, o_t)$$

$\theta$ never changes. $\mathcal{E}$ is the robot interface. Exactly one tool call per turn, returned as JSON — no prose outside the object.

**The tools.**
- `move_to` — one absolute target pose for the tool centre point (TCP), given as `pose_xyzquat` = $[x,y,z,q_x,q_y,q_z,q_w]$ in that arm's base frame.
- `move_eef_chunk` — an ordered list of poses, for a contact-free path that needs no intermediate look.
- `set_gripper` — normalised opening in $[0,1]$. Deliberately a *separate* decision from motion; Cartesian moves hold the gripper fixed.
- `check_path` — dry-run the IK without moving.
- `locate_point` — turn a pixel into geometry using calibrated intrinsics/extrinsics. One view gives a ray, not depth; two wrist views with parallax give a triangulated point.
- `done` / `give_up`.

**What the controller does with a request.** This is the "constrained" part, and it is where the paper stops being a prompt-engineering exercise.

1. Interpolate a geometric path between poses: straight line for position, SLERP (spherical linear interpolation, the shortest-arc path between two rotations) for orientation.
$$p_j(s) = (1-s)p_j + s\,p_{j+1}, \qquad R_j(s) = R_j \exp\!\left(s \operatorname{Log}(R_j^\top R_{j+1})\right)$$
2. Sample that path, then solve inverse kinematics at each sample, seeding each solve from the previous solution: $q_k = \operatorname{IK}(\hat p_k, \hat R_k; q_{k-1})$.
3. Reject the whole request unless every sample meets the residual tolerances $\|e_{p,k}\|_2 \le \epsilon_p$ and $\|e_{R,k}\|_2 \le \epsilon_R$ — **2 mm and about 1°** in execution, with tighter numerical stopping criteria of $10^{-4}$ m and $5\times10^{-4}$ rad.
4. Time-stamp the joint references with Ruckig, then stretch time by $\alpha_0 = \max\{1, r_v, \sqrt{r_a}, \sqrt[3]{r_j}\}$ where $r_v, r_a, r_j$ are the worst velocity/acceleration/jerk-to-limit ratios. This enforces the limits on the *sampled reference*, not on real continuous-time jerk.
5. Return measured state, endpoint error, and settling status as $f_t$.

A rejection is itself feedback. The prompt tells the model not to tilt the grasp just to make IK pass.

**The five context families.**

| Context | What it supplies |
|---|---|
| Goal image $G$ | The desired end arrangement. No procedure. |
| Human video $V$ | Procedure and grasp strategy. **No action labels at all.** |
| Robot video $V$ | Same, from a teleoperated robot. |
| Robot video + actions $A$ | Adds time-aligned TCP poses, gripper states, issued commands. |
| Interaction history $H_t$ | Its own past observations, calls, outcomes; plus live human moves and gestures. |

**Turning a video into tokens.** A vision model picks keyframes from overlapping windows (prompt P0a), then a global review pass deduplicates holds while forcing retention of first frame, last frame, contact, release, and arm-role changes (P0b). Cap: 24 keyframes, 48 images, 1280 px per side. Frames are interleaved with relative timestamps, camera labels, and stage annotations, wrapped in explicit `HISTORICAL DEMONSTRATION` / `END HISTORICAL DEMONSTRATION` markers so the model does not mistake a past episode for the live scene.

**Action alignment.** Each keyframe is matched to the nearest measured state within 0.1 s — timestamp matching, not hardware-synced exposure. Between keyframe $t_i$ and $t_{i+1}$ the recorded command segment is attached. Sampling keeps segment endpoints, roughly one sample per second, and both sides of every gripper change. In the row encoding, `"="` means "same as the row above" and `null` means **missing, never zero**.

**Hardware.** Two bimanual platforms (YAM, ARX X5: 6 joints/arm, top + two wrist RGB at 640×480) plus Morphi Kino (7 joints/arm, head/chest/two wrists at 1280×720, mobile base). TCP speed 0.08 m/s, 0.5 rad/s. Joint limits 0.6 rad/s, 2 rad/s², 12 rad/s³.

**Protocol.** Episode ends on success, budget exhaustion, or safety stop. Success requires the final scene to meet task-specific geometric and semantic criteria — the model saying `done` is explicitly *not* the label. Three trials per condition. Backbone is GPT-6 Astra.

## Ablation Studies and Experiments

The core ablation is the cleanest one available: same instruction, same observations, same success criteria, context added or removed.

**Human video, no action labels (cross-embodiment):**

| Task | Context | S/T | Decisions | Time (min) |
|---|---|---|---|---|
| Pick Red Towel | None | 0/3 | 96.3 | 24.6 |
| Pick Red Towel | Human Video | **2/3** | 76.7 | 18.9 |
| Pick Up Notebook | None | 0/3 | 94.0 | 24.6 |
| Pick Up Notebook | Human Video | **2/3** | 66.7 | 16.1 |

Success goes up *and* cost goes down. That combination matters — it rules out "context just made it try harder for longer". Notebook pickup drops 29% of decisions and 35% of wall time.

**Robot video vs robot video + actions (contact-sensitive):**

| Task | Context | S/T | Decisions | Time (min) |
|---|---|---|---|---|
| Unscrew Bottle Cap | None | 0/3 | 71.0 | 16.1 |
| Unscrew Bottle Cap | Robot Video | 2/3 | 74.3 | 15.2 |
| Unscrew Bottle Cap | Video + Action | **3/3** | 54.7 | 17.9 |
| Remove and Reinsert Plug | None | 0/3 | 24.0 | 5.3 |
| Remove and Reinsert Plug | Robot Video | **0/3** | 33.7 | 7.9 |
| Remove and Reinsert Plug | Video + Action | **2/3** | 48.3 | 10.8 |

**The negative result is the plug.** Video alone: 0/3, same as no context, but with 40% more decisions and 49% more time. Pure cost, zero benefit. Numeric action references are what rescue it — and they cost *more* again (48.3 decisions, 10.8 min) to get 2/3. So on a task that demands actual mating and seating, watching is not enough; you need the intermediate commanded poses.

The stated mechanism: keyframes are sparse, so the motion *between* them has to be guessed. Actions fill that in. Bottle opening has 205 retained action samples for 13 keyframes; the plug has 131 samples for 14 keyframes (down from 1,405 before sampling). Figure 7 measures this directly — supporting-gripper tilt and target-to-demonstration orientation difference at specific keyframes are both closer to the demonstration under Video + Action.

Note the cost story is not clean: action references raised decision count on both tasks relative to video-only, and raised time on the bottle. Averages include failures, so a failing run that burns the whole budget drags the mean. Treat the efficiency numbers as weaker evidence than the success numbers.

**Goal images:** 3/3 on both Arrange T Shape (66.7 decisions, 15.8 min) and Arrange Fruit (49.0, 12.4). No None baseline was run — only "preliminary qualitative comparisons". **This is not an ablation.** Take the 3/3 as a capability demo, not as evidence that the goal image caused it.

**Self-history:** 3/3 on both. Lemon to Pink Plate: 35.3 decisions, 8.1 min. Movable Exploration: 40.33 decisions but **25.53 min** — near-identical decision count, triple the wall time, because a mobile base moving is slow. Decisions and time are not interchangeable. Two emergent behaviours: the agent removes a towel to uncover a hidden plate before placing the lemon, and it routes around obstacles while searching. Again, no context-free control.

**Human interaction:** 3/3 on both. Tic-Tac-Toe (69.7 decisions, 13.6 min) — wins and draws both count as success, and the agent plays optimally from the board state. Pointed Fruit Pickup (67.3, 15.0) reads pointing gestures and stops on an OK sign.

**Model comparison (single runs, red towel):**

| Model | Context | Progress | Time (min) | Tokens |
|---|---|---|---|---|
| GPT-6 Astra | None | 55% | 24.63 | 12.05 M |
| GPT-6 Astra | Human Video | 100% | 15.85 | 4.96 M |
| Fable 5.1 | Human Video | 30% | 13.27 | 2.39 M |
| Kimi K3 | Human Video | 20% | 11.73 | 1.74 M |

The token number is the quiet headline: **12 million tokens for one failed towel pickup**, dropping to 5 million with the video. Context is not just better, it is 59% cheaper, because flailing is expensive. But: single runs, "task progress" is not success rate, and the authors explicitly refuse to call this a model ranking. Fable and Kimi being cheap is confounded with them giving up earlier.

**What did not work, gathered in one place:**
- Robot video alone on plug reinsertion — no gain, real cost.
- Precise contact. Knowing the strategy does not produce the millimetre.
- Outcome verification. The model cannot reliably tell whether a plug is seated or merely resting on the socket.
- Physical safety. **Inter-arm collisions were observed repeatedly.** The Cartesian planner does not check collisions at all; IK acceptance certifies nothing about clearance.
- Even the best condition is 2/3 or 3/3 on $n=3$. A single run is a third of the reported rate.

## Worth Remembering

**The prompt is doing an enormous amount of the work.** Appendix D is the real artifact. P1 defines the frames and the one-tool-call contract. P2 draws the historical/live boundary. P3 is a catalogue of hard-won distinctions: a submitted target is not an arrival; settled means joint stability, not TCP accuracy; joint torque includes gravity and is not contact force; a fully-open gripper command does not prove a wide object detached. P4 governs recovery — do not tilt the grasp just to pass IK, do not `give_up` while safe alternatives remain, do not declare done on an unconfirmed insertion. If you copy one thing from this paper, copy the discipline of separating *requested*, *submitted*, and *measured* everywhere in the feedback schema.

**Evaluation caveats to hold firmly.** $n=3$ per cell. Goal-image and interaction families have no no-context control, so they cannot support a causal claim. The authors say so themselves: "incomplete ablations and unobserved pretraining limit causal and novel-skill claims." GPT-6 Astra's pretraining is unknown; nobody can rule out that these objects and tasks resemble something it saw. **Current best understanding, on thin evidence** — not an established result.

**The System 1 / System 2 reading.** The most useful frame the authors offer. VLM agents are slow deliberators — minutes per task, millions of tokens. A trained vision-language-action policy is the fast reflex. The proposed split gives reasoning and replanning to the VLM and pose refinement to a VLA controller. The open question they name: *when should local feedback trigger replanning?* That is the interesting problem, and it is unsolved here. Compare with [[Model Predictive Control]] — plan open-loop, execute closed-loop, throw the plan away — which is the same shape at a different timescale.

**Where this connects.** The `move_eef_chunk` tool is an action-chunk interface — propose several poses, execute them without stopping to look — which is the same bandwidth-vs-reactivity trade [[Diffusion Policy]] makes. The whole loop is ReAct with a robot arm instead of a search API; see [[Agentic Workflows]] for the control-flow view and [[Tool Use]] for the wire format. Context-length pressure is managed exactly as in [[Context Engineering]]: pin the reference inputs, drop older live images, summarise stale exchanges. And [[Grounding]] is the live issue — the model must answer from the current frame, not from the demonstration it was shown.

**Practical caveats if you wanted to build this.**
- Budget for 5–12 M tokens and 15–25 minutes *per task attempt*. This is not a product.
- Build the collision layer first. The paper's own conclusion is that existing safeguards are insufficient for autonomous deployment.
- Outcome verification is the missing primitive. Everything downstream — knowing when to stop, when to retry, whether to trust `done` — depends on it, and neither the harness nor the model provides it.
- The `null`-means-missing convention in the action rows is a real design choice. Zero-filling a missing pose would silently command the robot to the origin.

**Follow-up questions.**
- What happens with 30 trials instead of 3? The gap between 2/3 and 3/3 is currently noise.
- Does a goal image beat no goal image? Nobody ran it.
- Can the agent *compose* subskills from two separate demonstrations into a new order? The authors flag this as untested, and it is the actual test of in-context learning versus sophisticated replay.
- Where is the failure — does the model propose a bad pose, or a good pose the controller tracks badly? The residual tolerances are logged; that decomposition should be extractable.

## Links
Related: [[In Context Learning]] · [[Agentic Workflows]] · [[Tool Use]] · [[Context Engineering]] · [[Diffusion Policy]] · [[Imitation Learning]] · [[Model Predictive Control]] · [[Grounding]] · [[Planning and Decomposition]] · [[Human in the Loop]] · [[Guardrails]] · [[Reflection]] · [[Structured Output]] · [[Chain of Thought]] · [[Zero-WAM- In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization]] · [[Think Like a World Model, Act Like a VLA- Distilling World-Model Representations into Compact Robot Policies]] · [[EXIMO- VLM Guided Exploration of VLA Policies]] · [[Computer Use]] · [[Observability and Tracing]]

New topics worth writing: inverse kinematics and residual tolerances, SLERP and quaternion interpolation, tool centre point and robot frame conventions, jerk-limited trajectory generation (Ruckig), cross-embodiment transfer, one-shot imitation learning, keyframe selection from demonstration video, action chunking, hierarchical System 1 / System 2 robot control, collision-aware motion planning, tactile and force feedback for grasp verification, mobile manipulation and active perception
