---
title: "PAWBench: How Far Are We from Probabilistically Aligned World Modeling?"
authors: ["Yuandong Pu", "Le Zhuo", "Sayak Paul", "Gabriel Jorge Menezes", "Avram Đorđević", "Shiyang Li", "Yifan Zhou", "Bin Fu", "Wenlong Zhang", "Junjun He", "Yu Qiao", "Yihao Liu", "Jinbo Xing", "Xi Chen"]
year: 2026
arxiv: "2608.27345"
url: https://arxiv.org/abs/2608.27345
priority: Good-To-Read
read_on: 2026-09-05
tags: [paper, llm, vision]
---
## The Core Idea

A video generator is being sold as a "world model" — a thing that can predict what happens next when you take an action. Current benchmarks check one video at a time: does it look real, is it temporally smooth, does it follow the prompt. This paper says that test is the wrong test.

The reason: many physical events have **more than one valid future**. Flip a coin and it lands heads or tails. Both are correct. So a world model should not be judged on whether one rollout looks plausible. It should be judged on whether **repeated rollouts from the same starting image and same action recover the right set of outcomes, in the right proportions**.

The authors name this requirement **probabilistic alignment** and build a benchmark for it.

> [!NOTE] Probabilistic alignment
> Given a fixed initial observation $x$ and action $a$, a world model $M$ induces a distribution $P_M(\tau \mid x,a)$ over future trajectories $\tau$. Map each trajectory to its terminal outcome with $g(\tau) \in \mathcal{Y}$, giving
> $$p_M(y \mid x,a) = \Pr_{\tau \sim P_M(\cdot\mid x,a)}[g(\tau) = y].$$
> **Support alignment** requires $\operatorname{supp}(p_M) = \mathcal{Y}$ — every valid outcome shows up at least once. **Probability-mass alignment** is stronger and requires $p_M(\cdot \mid x,a) = q$, the true reference distribution. ^probabilistic-alignment

The distinction matters. A model can be diverse without being calibrated (it shows heads and tails, but heads 90% of the time). It can also be calibrated on the outcomes it shows while never producing a third valid outcome at all. Single-sample evaluation cannot see either failure.

What this unlocks: a diagnostic that tells you whether a video model is actually a simulator you could plan against. If you want to use a world model for decision-making, you need to know both *what could happen* and *how likely each thing is* — a distributional object, not one pretty video. See [[Mastering Diverse Domains through World Models (DreamerV3)]] for the classic latent world-model framing this is critiquing from outside.

Headline finding: across **eleven** current video generators, **none** achieves accurate probabilities, broad coverage, and reliability across scenes at the same time. Average total variation distance from the reference distribution is **31.2** (out of a max of 100) where finite-sampling noise alone would only give **8.33**.

## The Methodology

**PAWBench** — 50 hand-built scenarios, 8 mechanism groups, split into two suites of 25:

- **PAW-Calibration** — scenarios where you can *derive* the reference distribution $q$ from analysis or symmetry. Coin toss, spinner with equal sectors, one-peg Galton board, blind ball draw. Groups: Tossing, Rotation, Routing, Draw.
- **PAW-Coverage** — scenarios where you can *list* the valid outcomes but cannot honestly assign probabilities. Bowling roll, bottle flip, dog choosing a route around an obstacle. Groups: Collision, Stability, Agent interaction, Material transition.

Each scenario is one source image (generated with Nano Banana Pro / GPT Image 2, then hand-picked), one action prompt, a finite outcome set $\mathcal{Y}$, and readout criteria. Three curation rules: (1) the randomness must come from a *visible physical mechanism*, not from a vague prompt or hidden state; (2) the action must be one atomic intervention; (3) outcomes must be visually distinguishable.

**PAWEval** — the readout. For each model–scene pair, sample $K=50$ rollouts with the image and prompt held fixed. Gemini 3.5 Flash applies a frozen per-scene rubric to each video and returns either a label $y \in \mathcal{Y}$ or a readout-failure $\bot$.

Metrics, computed only over readable in-schema rollouts ($n_{\text{readout}} = K - m$ where $m$ is the number of $\bot$):

$$\hat p_{M}(y) = \frac{1}{n_{\text{readout}}}\sum_{i=1}^{K}\mathbb{1}[y_i = y]$$

Calibration uses **total variation distance**, reported $\times 100$:
$$\mathrm{TVD}(\hat p_M, q) = \tfrac12 \sum_{y \in \mathcal{Y}} |\hat p_M(y) - q(y)|$$

Coverage uses **valid-support recovery**:
$$\mathrm{Cov} = \frac{|\{y \in \mathcal{Y} : \hat p_M(y) > 0\}|}{|\mathcal{Y}|}$$

**The gate.** A scene "passes" if at most 30 of its 50 rollouts fail readout (i.e. at least 20 are readable). **Scene Pass Rate (SPR)** is the fraction of the 25 scenes that pass. Conditional TVD/Coverage averages are only over passing scenes — so you must read the average *together with* SPR, or a model that only answers on 6 easy scenes looks great.

A separate **trustworthiness audit** checks whether the action was executed, whether object identity held, and whether the dynamics were physically sane. Deliberately, this does **not** affect any score. A spinner that ends on blue after a physically impossible rotation still counts as "blue". The authors keep readability and physical honesty as separate axes.

## Ablation Studies and Experiments

**Main table.** Eleven systems: HappyHorse, Veo3.1 Fast, Kling 3 Std., Seedance 2, Wan2.7, Wan2.2, LTX-2.3, LTX-2.5, Cosmos 3 Super I2V, LingBot-Video-MoE, MiniMax H3.

| Model | Calib. TVD ↓ | Calib. SPR | Cov. % ↑ | Cov. SPR |
|---|---|---|---|---|
| Cosmos 3 Super I2V | **20.5** | 80.0% | 55.2 | 92.0% |
| MiniMax H3 | 24.2 | 68.0% | 48.7 | 92.0% |
| Wan2.2 | 26.3 | 64.0% | 63.4 | 92.0% |
| Wan2.7 | 26.3 | 92.0% | 50.0 | 96.0% |
| LTX-2.3 | 30.1 | **24.0%** | **71.7** | 72.0% |
| Seedance 2 | 30.5 | **100.0%** | 50.9 | 84.0% |
| Veo3.1 Fast | 35.4 | 88.0% | 41.8 | 100.0% |
| HappyHorse | 43.1 | 92.0% | 47.1 | 100.0% |

The pattern is the whole paper. Cosmos wins calibration but fails the gate on 20% of scenes. LTX-2.3 wins coverage but only 24% of its calibration scenes are even scoreable. Seedance 2 scores on everything but leads nothing. **Coverage and calibration are not correlated** — being diverse and being right about probabilities are different skills.

**Is it just sampling noise?** No. They ran $B = 50{,}000$ Monte Carlo replicates, drawing from the *true* reference distributions while matching each model's exact passing scenes and readable-sample counts. Null mean TVD = 8.33, 99th percentile = 9.22. Observed = 31.2. Every single generator exceeds the 99th percentile of its own matched null. Holds under pooled aggregation and a stricter $n_{\text{readout}} \geq 30$ threshold.

**Is it the judge?** Partly, but not enough to explain the gap. 1,500 videos, 7 independent human votes each. On the 888 videos where both PAWEval and a decisive human panel gave an in-schema label, they agreed 81.3%. Agreement scales with human consensus: 58.7% (3–4 votes agree), 77.3% (5), 91.7% (6–7).

**More samples?** Going from $K=1$ to $K=100$: coverage keeps rising for 3 of 4 models, **calibration barely moves**. More rollouts find more outcomes but do not fix the proportions. This is why $K=50$ is used as a budget, not claimed as a convergence point.

**Causal vs non-causal interventions — the sharpest result.** Paired scenes. A *causal* change alters the physics (tilt the pencil left, so the reference goes from 50/50 to 100/0). A *non-causal* change alters something irrelevant (put outcome-suggestive text next to a Galton board, or recolour a distractor ball). Models **underreact to causal changes** — the distribution shifts too little, or in the wrong direction — and **overreact to non-causal cues**, moving probability mass when nothing physical changed. This is the clearest sign that these models are pattern-matching pixels and text rather than tracking a transition function. Compare the mechanism in [[Shortcut Learning in Deep Neural Networks]].

### Three attempts to fix it, in increasing order of invasiveness

**1. Language (prompt engineering).** Can you steer the distribution by naming futures in prompts?

First, do VLMs even know the distribution? Query five VLMs (Qwen3.5 Plus, GPT-5.5, GLM-5V Turbo, Kimi K2.6, Gemini 3.5 Flash) 50 times each with the same image + action, project answers into $\mathcal{Y}$. Best calibration TVD is GLM-5V Turbo at **34.8**; best coverage is Gemini 3.5 Flash at **46.6%**. **The language models are already misaligned before any video is made.**

Then chain it: GPT-5.5 predicts an outcome and writes a generator prompt requesting it (**PE**). Its own selections score TVD 44.3 / Coverage 35.0%. Passing them to generators **raised SPR for all four models but made calibration TVD worse in every case** (Cosmos 20.5 → 31.6, MiniMax 24.2 → 38.9, LTX-2.5 30.2 → 36.4) and improved coverage for only two of four. **This did not work.**

**Oracle PE** — the authors hand-schedule targets across the 50 rollouts to match $q$ exactly (largest-remainder allocation for Calibration, even split for Coverage). Now it works: Cosmos 20.5 → **12.8** TVD, coverage 55.2 → **87.3%**. MiniMax 24.2 → 10.8, coverage 48.7 → 82.9%. But even with a target handed over explicitly, generators **hit the requested outcome only 37.6–58.1% of the time**. So there are two independent failures: the controller picks the wrong distribution, *and* the generator cannot reliably render a named outcome.

**2. Initial noise coupling (C2C).** Instead of drawing $K$ independent noise tensors, draw them in groups of $m=5$ and centre them:
$$\tilde\epsilon_{g,i} = \sqrt{\tfrac{m}{m-1}}\Big(\epsilon_{g,i} - \tfrac1m\textstyle\sum_j \epsilon_{g,j}\Big)$$
Each $\tilde\epsilon_{g,i}$ is still marginally $\mathcal{N}(0,I)$, but within a group they sum to zero and have covariance $-I/(m-1)$ — they repel each other. $G=10$ groups, $m=5$, so still $K=50$.

Results: coverage rises for all three tested generators (Wan2.2 63.4 → 69.2, LTX-2.3 71.7 → 74.8, Cosmos 55.2 → 63.9) and mean TVD drops slightly. **But this does not change the model's learned law at all** — the marginal is preserved. It only proves that an i.i.d. gallery of 50 was under-exploring futures the model could already produce. SPR does not consistently improve.

**3. Fine-tuning.** Five LoRA adapters (rank 32) on Wan2.2 I2V-A14B, trained on 2,000-example datasets of falling pencils with left-fall shares of 0/20/50/80/100%. Same 100 left + 100 right source videos, repeated to fill; only the mixture ratio varies. $832\times480$, 49 frames, lr $10^{-4}$, 2,000 steps, paired adapters for the high- and low-noise experts. See [[LoRA- Low-Rank Adaptation of Large Language Models]].

Evaluated on two scenes with a **direction-neutral** prompt: upright pencil (reference 50/50) and left-leaning pencil (reference 100/0).

| $P(\text{left})$ in training | Upright TVD | Left-leaning TVD |
|---|---|---|
| Base | 17.3 | 41.3 |
| 0% | 50.0 | 97.2 |
| 20% | **17.6** | 50.0 |
| 50% | 23.5 | 18.9 |
| 80% | 48.0 | **0.0** |
| 100% | 50.0 | **0.0** |

Training mixture clearly moves the output distribution, and the relationship is **nonlinear** — 20% left-fall training gives ~33% left-fall generation on the upright scene. But **no single mixture gets both scenes right**. 20% nails the upright pencil while the left-leaning scene sits at 50/50; 80% nails the left-leaning scene while the upright pencil falls left almost every time. **Both scenes move in the same direction under the same adjustment.** The fine-tuning writes a global directional prior, not a state-conditioned distribution. That is the negative result the paper cares most about.

## Worth Remembering

- **The "one plausible video" fallacy is the whole point.** If you evaluate a generative simulator by sampling once and eyeballing it, you cannot detect mode collapse or mass misallocation. This is the video-model analogue of [[Mode Collapse]] in [[Generative Adverserial Network]], but measured at the level of *semantic outcomes* rather than pixel diversity.
- **Read averages next to SPR.** LTX-2.3 has the best coverage average over the 72% of scenes that pass and only 24% of calibration scenes passing. Conditional metrics over self-selected subsets are always suspicious — same disease as [[On the Difficulty of Evaluating Baselines]].
- **Separating scoring from auditing is a nice design choice.** A physically impossible rotation that ends on blue still counts as blue. This keeps the distributional measurement clean and puts physics violations in a separate, non-scoring diagnostic. Physical-process errors are the largest failure category overall.
- **The Monte Carlo null is the right move**, and worth stealing. Whenever you estimate a distribution from finite samples and compare it to a reference, simulate the null with *matched* sample counts before claiming a gap.
- **The causal/non-causal probe is cheap and devastating.** Distractor text moves the model's outcome distribution; tilting the object does not move it enough. Any world model claim should be stress-tested this way.
- **C2C is free and worth using in production** if you sample multiple videos and want diverse candidates: it preserves the marginal, so it cannot break your model, and it broadens what a finite gallery covers. But do not confuse it with fixing calibration.
- **Admitted limits.** Only terminal outcomes are evaluated, not trajectory-level dynamics or intermediate states. Finite budget ($K=50$). Scenarios are clean, short-horizon, non-interactive — no embodied or long-horizon environments. And the judge is an LLM at 81.3% human agreement on clear cases, which is decent but not free of systematic bias.
- **Open question the paper leaves.** What training objective would actually produce a scene-conditioned calibrated distribution? Data reweighting gives only a global knob. You would presumably need a loss that penalises distributional mismatch across repeated rollouts per condition — which is expensive, since it requires sampling $K$ futures per training example.

## Links

Related: [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[KL Divergence]] · [[Mode Collapse]] · [[Shortcut Learning in Deep Neural Networks]] · [[Denoising Diffusion Probabilistic Models]] · [[Uncertainty]] · [[Random variable]] · [[Do ImageNet Classifiers Generalize to ImageNet]] · [[Towards Quantifying Benchmark Optimization in ASR Models]] · [[Agentic Game Development as a Verifiable Trajectory Data Engine for Scaling World Models]] · [[Game2World Engine- Unlocking In-the-Wild Gameplay Videos for World Model Training]] · [[Classifier-Free Diffusion Guidance]]

New topics worth writing: Total variation distance, Calibration of generative models, Precision and recall for generative models, Antithetic and coupled sampling variance reduction, Distributional alignment in LLM sampling, Stochastic video prediction (SV2P / SVG-LP), LLM-as-judge evaluation reliability
