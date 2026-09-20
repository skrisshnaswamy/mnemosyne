---
aliases:
  - Diffusion Policies
  - Diffusion for Control
  - Diffusion for Decision Making
  - Diffuser
  - Action Diffusion
  - Action Chunking
  - Vision-Language-Action Model
  - VLA
  - DDPO
  - Diffusion-DPO
  - RL Fine-Tuning of Diffusion Models
tags:
  - generative-models
  - diffusion
  - reinforcement-learning
  - robotics
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Use a diffusion model as a robot's [[Policy|policy]]: conditioned on what the camera sees, **denoise a short sequence of future actions** out of random noise. It can represent *"go left **or** go right"* — which an ordinary regression policy averages into *"crash into the middle"*.
> **Metaphor:** A driver approaching a bollard. Left is fine. Right is fine. The average of left and right is the bollard.
> **Where it bites:** Modern robot imitation learning (Diffusion Policy, π₀, the VLA family) — and, in the other direction, **RL applied to diffusion models** to align image generators.

---
You're teaching a robot arm to push a block around an obstacle, by [[Imitation Learning|imitation]]. You collect 200 human demonstrations.

Half the demonstrators went round the **left** of the obstacle. Half went round the **right**. Both are perfectly good.

You train the standard way: a network maps the camera image to an action, with a squared-error loss.

~={blue}Faced with the obstacle dead ahead, what action minimises squared error against a dataset that says "left" half the time and "right" the other half?=~

---
# The bollard 🚧

**Straight ahead.** Into the obstacle.

Squared-error regression outputs the **mean** of the targets. When the data has two good answers, the mean is a third answer — and often the *worst* one. It's exactly the [[Maximum Likelihood#The consequence that shapes the whole field|orange curve]]: a model too simple for two-humped data, putting its mass in the empty middle.

Human demonstrations are *full* of this. People hesitate, take different routes, grasp from different sides. A policy is a **distribution** over actions ([[Policy#Why would you ever want a *random* policy?|and often needs to be]]) — so represent it with something built for multi-modal distributions.

> [!NOTE] Diffusion policy
> A visuomotor policy that represents $\pi(a_{t:t+H} \mid o_t)$ as a **conditional diffusion model**: starting from Gaussian noise, iteratively denoise a *sequence* of future actions, conditioned on recent observations. (Chi et al., 2023.) ^diffusion-policy-def

| | Regression (behaviour cloning) | Discretised / mixture | **Diffusion policy** |
|---|---|---|---|
| Two valid answers | averages them ❌ | handles a few | **samples one, commits to it** ✅ |
| High-dimensional action *sequences* | fine | bins explode ([[Curse of Dimensionality]]) | ✅ |
| Training stability | ✅ | fiddly | ✅ — it's [[Denoising Objective\|plain regression on noise]] |
| Inference cost | 1 pass | 1 pass | **10–100 denoising steps** |

> [!SUCCESS] Core idea
> ~={pink}Actions are just another thing to generate.=~ Swap "image" for "the next 16 actions" and "prompt" for "camera view", and every tool in this chain carries over: [[Conditional Generation|conditioning]], [[Classifier-Free Guidance|guidance]], few-step [[Diffusion Sampling|samplers]], [[Flow Matching]]. The payoff is a policy that can say "either — but *one* of them", where regression can only say "somewhere in between". ^actions-are-generated

---
# Two details that matter as much as the diffusion

**Action chunking.** Predict a *chunk* — the next ~16 actions — execute the first ~8, then re-plan. One sample commits to **one** mode for the whole chunk, so the arm doesn't dither left-right-left between timesteps. And re-planning halfway is closed-loop feedback: it's [[Model Predictive Control#^plan-open-execute-closed|receding-horizon control]] with a generative model as the planner.

**Position, not velocity, targets** — and a CNN or transformer denoiser conditioned on the image via [[Conditional Generation#Three ways to get the condition in|FiLM]]. Unglamorous, and responsible for a lot of the reported gains.

> [!WARNING] The latency problem is real
> A control loop wants 10–50 Hz. A diffusion policy needs tens of network passes per decision. Chunking amortises it (one expensive plan per ~8 actions); DDIM-style samplers cut the steps; [[Consistency Models|consistency]] and [[Flow Matching|flow-matching]] policies (π₀ uses flow matching) get it to a handful. But it's still slower than a reflex — the same [[Monte Carlo Tree Search#Reflex vs deliberation|reflex-vs-deliberation]] trade as everywhere else. ^policy-latency

---
# The wider family

| Idea | What's diffused | Note |
|---|---|---|
| **Diffusion Policy** | action sequences | imitation learning |
| **Diffuser** (Janner et al.) | entire **state–action trajectories** | *planning as sampling*: generate a whole plan, and **guide** it toward high reward with a learned value function — [[Classifier-Free Guidance#Where the unconditional model comes from\|classifier guidance]], with the [[Value Function\|value]] as the classifier |
| **Decision Diffuser** | trajectories, conditioned on the desired return | [[Offline RL]] recast as conditional generation |
| **VLA models** (π₀, RT-2 lineage) | actions, from a vision-language backbone | a VLM reads the scene and the instruction; a diffusion/flow "action head" emits motion |
| **Generative world models** | future video | → [[Video Diffusion#From video to world model 🌍\|video as simulator]] |

Staying near the data is a feature here, not a bug: a diffusion model trained on logged trajectories only generates trajectories *like the logged ones* — exactly the conservatism [[Offline RL#^pessimism-principle|offline RL]] needs.

---
# The arrow pointing the other way: RL *on* diffusion models

A sampling run is a sequence of decisions: state = $(x_t, t)$, action = the denoising step, episode = noise → image, reward = given at the very end. **That's an [[Markov Decision Process|MDP]].** So:

| Method | Idea |
|---|---|
| **DDPO / DPOK** | treat denoising as a multi-step policy; apply [[Policy Gradient\|policy gradients]] / [[PPO]] with a reward for the final image (aesthetics, prompt alignment, compressibility…) |
| **Reward-gradient methods** (DRaFT, AlignProp) | if the reward is differentiable, backprop it straight through the sampler |
| **Diffusion-DPO** | [[DPO]] for images: learn from *pairs* of images people preferred — no reward model, no RL loop |

> [!WARNING] …and it reward-hacks, exactly as you'd predict
> Optimise an image model against an "aesthetic score" and you get garish, oversaturated, samey pictures that the scorer loves. Optimise for a CLIP-based alignment reward and text starts appearing *written in the image*. The reward is a proxy; the optimiser finds the gap → [[Reward Hacking]]. The cure is the familiar one: a KL leash to the base model, and watching held-out metrics → [[RLHF#^kl-leash|the KL leash]]. ^diffusion-reward-hacking

---
> [!SUCCESS] If you remember one thing
> **When the data has several right answers, regression returns their average — which may be wrong.** ~={pink}A diffusion policy samples one good answer and commits to it; and the same maths run backwards lets RL fine-tune image generators.=~

---
# ⁉️
Every model in this chain *generates* — pixels, tokens, actions. There's a serious argument that for **understanding** the world, generating is the wrong goal: most of what's in a frame is unpredictable detail, and a model forced to predict it wastes itself. What's the alternative?

→ [[JEPA]]
