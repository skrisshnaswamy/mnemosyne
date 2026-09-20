---
title: "Agentic Game Development as a Verifiable Trajectory Data Engine for Scaling World Models"
authors: ["Pengfei Zhou", "Hexin Wang", "Zhengfeiyang Zhang", "Yixing Ma", "Zhenglin Wan", "Kaipeng Zhang", "Wangbo Zhao", "Yang You"]
year: 2026
arxiv: "2608.25518"
url: https://arxiv.org/abs/2608.25518
priority: Good-To-Read
read_on: 2026-09-05
tags: [paper, llm, rl, vision, theory]
---
## The Core Idea

Code agents got good fast because code can be *run*. A compiler, a test suite, a runtime — these give you a cheap, dense, hard-to-fake reward signal, so you can do reinforcement learning after pretraining. Spatial models (video, 3D, "world models") have no such thing. Their quality is scored by fuzzy proxies: FVD, CLIP similarity, an MLLM asked to be a judge, or human raters comparing final outputs. These are noisy, biased, and gameable, so the only reliable training signal left is "predict the next frame of scraped video". The authors call the resulting cost the **unverifiability tax**: you pay for progress with more data and more compute because you cannot pay for it with better feedback.

Their claim: the missing verifier already exists, and it is called a **game engine**. A Unity/Unreal/Godot scene is not a picture — it is an executable specification. The engine can cheaply answer: does it load? do colliders intersect? is the physics rollout stable? is the navmesh connected? does a scripted probe agent reach the goal? None of that is fooled by a pretty texture. What the engine *cannot* answer is whether the scene is any good — whether it matches the brief, reads clearly, fits the product. That is what a human developer decides.

So the reward is split by **authority**: the engine gives dense, reproducible, local structural checks; the human gives a sparse, global accept/reject. They call training against this pair **RLHEV** — Reinforcement Learning with Human-Engine Verification.

The second, and I think more interesting, half of the idea: the *artifact* is not the valuable data, the *trace* is. A finished scene tells you what worked. A development trace tells you the intent, the failed collision check, the specific object that caused it, the repair edit, the re-check, and the reviewer's verdict. That is structurally the same shape as a code trajectory with tests and patches, and it is a byproduct of work people are already paid to do.

> [!NOTE] Unverifiability tax
> When a domain has no cheap automatic checker, all progress must be bought with more data, more compute, and more human annotation. The bottleneck is the feedback channel, not the architecture. ^unverifiability-tax

> [!NOTE] RLHEV
> Post-training where a hard gate from engine checks must pass, then the scalar reward is a weighted blend of human acceptance and normalised engine diagnostics. ^rlhev

This is also a re-reading of [[The Bitter Lesson (essay)]]. The usual summary is "general methods that scale with compute win". The authors point out that every domain where that visibly happened — Go, chess, code, maths — already had a *free evaluator* built in: the game rules, the compiler, the numeric checker. Compute searches; the verifier decides. Spatial intelligence has the compute and not the verifier.

## The Methodology

**AWoMo (Agentic World Model)** is not a network, it is a loop with four interfaces:

- *intent* — task brief, reference images, design constraints
- *action* — scene programs, asset edits, tool calls, repair actions
- *verification* — engine checks: load, collision, physics stability, navmesh reachability, script errors, bounded playability probes
- *review* — human accept / reject / critique / residual-risk note

Loop: propose → render → verify → repair → review. Every pass is stored.

**UWDP (Unified World-Development Protocol)** is the storage format. One step is

$$u_t = (b, o_t, s_t, a_t, g_t, v_t, h_t, \rho_t)$$

where $b$ is the design intent ("arrange a playable storage room with a clear path to the exit"), $o_t$ is a *stable object id* (`crate_07`, or a relation `near(chair_03)`), $s_t$ is the executable state (transform, collider bounds, material tags, navmesh status), $a_t$ is the edit ("move `crate_07` to $(2.4, 0.0, 2.6)$, regenerate collider"), $g_t$ is the engine verdict (collision fail before, navmesh pass after), $v_t$ is rendered evidence, $h_t$ is the reviewer decision, and $\rho_t$ links the failure to the repair plus cost and residual risk.

The stable ids are the load-bearing part. They let you connect *this check failed* → *this object caused it* → *this edit fixed it*. That is credit assignment you cannot get from an image-caption pair.

**Compiling traces into training data.** Three products from one trace: (1) accepted terminal states → supervised generation targets conditioned on $b$; (2) failed check + its repair → next-edit / repair-prediction pairs; (3) $g_t$ and $h_t$ → the fused RL reward. Rejected traces are kept — they are the negative half of the acceptance signal that final-artifact datasets throw away.

**The reward.** With $h \in [0,1]$ the human reward, $e \in [0,1]$ the normalised engine reward, and $\mathbf{g} \in \{0,1\}^m$ binary engine gates:

$$r_{\alpha,\beta}(\mathbf{x},\mathbf{a}) = \mathbb{I}\{\mathbf{g}(\mathbf{x},\mathbf{a}) = \mathbf{1}\}\big(\alpha h + \beta e\big) - \lambda_c\, c(\mathbf{x},\mathbf{a})$$

Gates first (all must pass or reward is zero), then the blend. Main runs use $\alpha = 0.65$, $\beta = 0.35$, $\lambda_c = 0$.

**The objective** is offline, reward-weighted log-likelihood with a KL leash to the supervised reference policy $\pi_0$ — the same skeleton as [[Training language models to follow instructions with human feedback|RLHF]] and [[Simple Statistical Gradient-Following Algorithms (REINFORCE)|REINFORCE with a baseline]]:

$$\max_\theta \; \mathbb{E}_{(\mathbf{x},\mathbf{a})\sim\mathcal{D}}\big[w(\mathbf{x},\mathbf{a})(r_{\alpha,\beta} - b(\mathbf{x}))\log \pi_\theta(\mathbf{a}\mid\mathbf{x})\big] - \lambda_{\mathrm{KL}}\,\mathbb{E}_\mathbf{x}\big[D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_0)\big]$$

See [[KL Divergence]] for the leash term and [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] for why offline RL needs one.

**The model.** `UnifiedGameAssetModel`: 2.890B parameters, hidden width 3584, only **8 transformer layers**, 4 attention heads, 4 gated experts (a Mixture-of-Transformers, cousin of the [[Sparsely-Gated Mixture-of-Experts Layer]]). Modality-typed token streams with heads for text, rendered images, 3D Gaussians, meshes, and Unity/Godot/Unreal/MuJoCo representations. Initialised from Cosmos 3, then on-policy distilled (see [[On-policy Distillation with Verifiable Reward]]), then continued pretraining: 8×A100, global batch 4096, grad accumulation 2, 79,130 training samples, best checkpoint at step 589 by validation loss.

**Pretraining manifest:** 87,745 accepted samples, 504 rejected. By output modality: text 87,745, images 53,426, Unreal 9,678, Unity 7,423, 3D assets 3,194, meshes 3,194, MuJoCo 2,946, Godot 2,685, 3D Gaussians 12. All developer-authorised.

## Ablation Studies and Experiments

**UnitySceneBench.** Binary classification: given an edit prompt, Unity asset context, reference-image features and layout payload, is this candidate edit accepted or rejected? 720 train / 80 val / 200 test (100 accept, 100 reject). Primary score is a fixed blend:

$$\text{Primary} = 0.45\,\text{bal.acc} + 0.25\,\text{acc} + 0.20\,\text{F1} + 0.10\,\text{AUC}$$

Baselines: zero-shot CLIP threshold; Fuzzy Proxies (trained on CLIP-similarity reward); SFT (supervised on the labels); Offline RLHF ($\alpha,\beta = 1,0$); Engine-based RLVR ($0,1$); Full RLHEV ($0.65, 0.35$).

Full RLHEV wins: primary $0.681$, accuracy and balanced accuracy $0.665$, F1 $0.733$, AUC $0.690$ — $+0.098$ primary and $+0.120$ accuracy over the best non-full baseline. **But** Figure 4 is explicitly *best-of-eight seeds*, not a mean. The seed-averaged view is a separate figure. Reporting a max over 8 runs as the headline number is exactly the practice [[On the Difficulty of Evaluating Baselines]] warns about.

**Generation quality vs training budget** (mean over 8 seeds, so this table is the trustworthy one):

| Method | 40 | 160 | 640 | 720 |
|---|---|---|---|---|
| Fuzzy Proxies | 0.7478 | 0.7154 | 0.7127 | 0.7174 |
| SFT | 0.7651 | 0.7270 | 0.7486 | 0.7441 |
| Offline RLHF | 0.7735 | 0.7449 | 0.7589 | 0.7652 |
| Engine RLVR | 0.7430 | 0.7137 | 0.7904 | 0.7934 |
| Full RLHEV | **0.8002** | **0.7821** | **0.8106** | **0.8197** |

Two things stand out. Engine-only reward is the *worst* method at small budgets (0.743 at n=40, below even the CLIP-proxy baseline) and only overtakes SFT and RLHF past 320 examples — dense structural checks need volume before they pay. And every method *dips* from 40 to 160 examples before recovering, which nobody explains.

**Transfer.** Judged by a normalised MLLM-as-judge score in $[0,1]$ (Qwen3.6-35B-A3B, same rubric as the human channel, every score human-verified, never used as training reward).

- Unity → held-out Unity: scratch $0.25$, target-adapted from source checkpoint $\mathbf{0.75}$
- Unity → Unreal: scratch $0.25$ → adapted $0.35$
- Unity → Godot: scratch $0.15$ → adapted $0.35$

**What did not work.** The source-only checkpoint, evaluated on a new engine without any target adaptation, *collapses*: judge $0.15$, proxy score $0.005$, loss $16.69$ — far worse than the untouched zero-shot base model (judge $0.25$, loss $12.38$). Training hard on Unity actively destroys the ability to emit valid Unreal or Godot output. Engine-specific traces are useful *initialisation*, not transferable skill; you must recalibrate on the target runtime.

**The ablation that actually carries the paper** (Appendix A.3, and it is buried). They train a ridge-regression probe to rank target-engine quality, with only **8** labelled target examples, and compare two feature sets:

- *snapshot-only*: format tags, prompt length, asset count, output size
- *protocol-trace*: adds Unity accept/reject label, deviation type, target-engine id, used-asset identity

Held-out Spearman correlation, 8 seeds, two target engines:

| Source instances | Snapshot-only | Protocol-trace |
|---|---|---|
| 0 | $0.159 \pm 0.168$ | $0.719 \pm 0.094$ |
| 720 | $0.141 \pm 0.084$ | $0.758 \pm 0.044$ |

The gap of ~0.6 is entirely from the *representation*. Adding 720 source examples moves the trace probe by 0.04 and moves the snapshot probe by nothing at all. Read plainly: recording process metadata is worth vastly more than scaling the amount of final-artifact data. That is the paper's thesis in one table, and it is stronger evidence than the RL results.

**Embodied diagnostics.** AWoMo here is not a policy — it is a data-augmentation filter. For R2R it scores existing PREVALENT augmentation trajectories, rebalances to the target path distribution, and fine-tunes a SAME policy. For MuJoCo it synthesises state-action pairs and filters them with acceptance checks.

| Benchmark | Original | Naive aug. | AWoMo aug. | $\Delta$ |
|---|---|---|---|---|
| R2R success rate | $76.006 \pm 0.133$ | $76.176 \pm 0.175$ | $76.607 \pm 0.094$ | $+0.79\%$ |
| Gym MuJoCo return | $1568 \pm 1757$ | $1648 \pm 1689$ | $1725 \pm 1643$ | $+9.96\%$ |
| D4RL normalised | $18.30 \pm 14.66$ | $25.56 \pm 12.45$ | $27.16 \pm 9.49$ | $+48.43\%$ |

The D4RL headline of $+48\%$ is mostly naive augmentation ($18.3 \to 25.6$); AWoMo adds $25.6 \to 27.2$ on top, well inside a standard deviation of $\pm 9.5$. The MuJoCo standard deviation *exceeds its mean*. R2R moves 0.6 points. Treat all three as directional at best.

## Worth Remembering

**The circularity in the main benchmark.** On UnitySceneBench the "human reward" $h$ *is* the accept/reject label — which is also the classification target. So Offline RLHF and Full RLHEV receive the ground-truth label as reward, while the SFT baseline receives the same label as a supervised target. The comparison is not "reward vs no reward"; it is "two ways of consuming the same label". The genuinely independent signal in Full RLHEV is only the $\beta = 0.35$ engine component. This weakens the P2 claim considerably.

**Resolution of the judge scores.** Values of $0.15$, $0.25$, $0.35$, $0.75$ on a $[0,1]$ scale, in steps of $0.05$, strongly suggest ~20 judged items. Unity→Unreal going $0.25 \to 0.35$ is a difference of two examples. The authors are honest that no engine-native metric compares across engines, but that honesty does not make ten-percentage-point differences on twenty items meaningful.

**Scale.** 8 transformer layers and 2.89B parameters, 720 training examples, 200 test examples. This is a position paper with pilot studies attached, and the authors say so ("current experiment results are positive but still diagnostic"). Judge it as an argument, not as a result.

**The objections the authors raise on themselves** (Appendix C.1) are the best part of the paper:
- *Games are not reality.* No scans, no robots, no real-to-sim-to-real loop in any experiment. They name the right next test — take a real environment, fine-tune lightly, measure which executable checks still bind.
- *The engine reward is gameable too.* A loose collider check will be optimised around. Their answer is verifier ensembles, randomised probes, held-out checks, and human review on top. The saving grace over CLIP-style proxies is that engine failures are *localisable* — you can point at the object — which makes reward hacking measurable rather than only perceptual.
- *Cheaper 3D scans do not help.* A scan tells you what one instance looked like. It cannot tell you whether a *new* synthesised output is correct. Capture is not verification.

**Reward-estimation bound.** They note the standard result: if the implemented reward is within $\epsilon$ of the intended reward everywhere, an $\eta$-optimal policy under the implemented reward loses at most $2\epsilon + \eta$ under the intended one. It is stated, not used, and does not constrain anything in the experiments — mild [[Troubling Trends in Machine Learning Scholarship#|mathiness]].

**Practical takeaway if you build systems.** The instrumentation argument stands on its own, independent of whether the RL numbers hold. If you are running any workflow where a machine can partially check the output and a human makes the final call — game dev, CAD, data pipelines, analytics dashboards — logging the *trajectory* (intent, failed check, the specific object that failed, the repair, the verdict) rather than the accepted artifact is nearly free and, by the probe ablation, worth more than an order of magnitude more artifacts. Rejected attempts especially: they are the negative half of the signal and normally get deleted.

**Open question.** The whole loop's endgame is recursive: a world model builds a level, game agents playtest it and find soft-locks and unreachable goals, those failures train the next world model. That is [[Mastering Chess and Shogi by Self-Play (AlphaZero)|AlphaZero]]-shaped self-play, and none of it is implemented here. The hard part is making playtest agents difficult to fool — which is where [[AgentMercury- Your Agent Can Synthesize Verifiable Environments for Business Scenarios at scale]] and [[Autonomous Mathematical Discovery in an Open-World Multi-Agent Environment]] are attacking the same problem in non-spatial domains.

## Links

Related: [[The Bitter Lesson (essay)]] · [[Training language models to follow instructions with human feedback]] · [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] · [[AgentMercury- Your Agent Can Synthesize Verifiable Environments for Business Scenarios at scale]] · [[Game2World Engine- Unlocking In-the-Wild Gameplay Videos for World Model Training]] · [[WorldMind- Decoupled Game World Model for State-Aware NPC Behavior]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[On-policy Distillation with Verifiable Reward]] · [[Sparsely-Gated Mixture-of-Experts Layer]] · [[KL Divergence]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[On the Difficulty of Evaluating Baselines]] · [[Troubling Trends in Machine Learning Scholarship]] · [[Mastering Chess and Shogi by Self-Play (AlphaZero)]] · [[Autonomous Mathematical Discovery in an Open-World Multi-Agent Environment]]

New topics worth writing: RLVR (Reinforcement Learning from Verifiable Reward), Reward hacking and reward-model overoptimisation, Fréchet Video Distance, Genie and interactive video world models, Unsupervised Environment Design (PAIRED, POET), Sim-to-real transfer and domain randomisation, Vision-and-Language Navigation / R2R, D4RL and offline RL benchmarks, Mixture-of-Transformers, Procedural content generation via machine learning
