---
title: "Every Coin Has Two Sides: On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models"
authors: ["Zhaoyi Li", "Deyang Kong", "Yuan Wei", "Evan Yang", "Ranran Shen", "Mahardika Krisna Ihsani", "Ming Yang", "Wei Zhang", "Chuan Hao", "Jian Yang", "Ran Tao", "Bryan Dai", "Shikun Zhang", "Wei Ye", "Ying Wei", "Defu Lian"]
year: 2026
arxiv: "2608.16647"
url: https://arxiv.org/abs/2608.16647
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, llm, vision]
---
## The Core Idea

On-policy distillation (OPD) is this: the student model writes its own answers, and the teacher model grades every token of those answers by saying how likely *it* would have been to write that token. The student is nudged toward tokens the teacher likes. Nobody had carefully asked how far this training actually reaches — most papers train on math and then test on math.

> [!NOTE] On-policy distillation ^opd
> Train the student on text the student itself generated, with dense per-token supervision from a teacher. Contrast with offline [[Distilling the Knowledge in a Neural Network|distillation]], where the student is trained on the teacher's own text and never sees its own mistakes.

Three findings, and they fit together.

**1. OPD copies reasoning style, not answers.** If you filter the training set to only problems the teacher gets wrong every single time, the student ends up at the *same* accuracy as if you had used only problems the teacher always solves. Grade-school GSM8K problems recover over 80% of the gain you get from hard competition math. The teacher's per-token probabilities carry useful information about *how to think* even when the final answer is wrong.

**2. Whether it generalises depends on lineage, not on teacher strength.** If the teacher was made by post-training the *same* base checkpoint the student came from ("same-origin"), the student climbs to roughly the teacher's level on everything — English math, Chinese math, long-horizon math, code, science — even though you only trained on English math prompts. If the teacher comes from a different base model ("cross-origin"), the student mostly just fits the training distribution. A *weaker* same-origin teacher beats a *stronger* cross-origin teacher.

> [!NOTE] Same-origin vs cross-origin ^model-origin
> Two models are same-origin if they share a concrete initialisation checkpoint — e.g. DS-distill-1.5B (student) and JustRL-1.5B (teacher, made by RL post-training on that exact checkpoint). Cross-origin means different roots.

The mechanism they measure: the **top-16 overlap ratio** between teacher and student next-token distributions. For same-origin pairs it starts high and *rises* over training. For cross-origin pairs it starts low and stays flat or *falls*. Same objective, same loss going down — but only same-origin OPD moves the whole policy toward the teacher. Cross-origin OPD just lowers the KL on the prompts you fed it.

**3. This breaks multi-teacher OPD.** The standard recipe (MOPD) sends math prompts to a math expert and science prompts to a science expert, assuming each teacher's influence stays in its lane. It does not. Because each teacher pulls the student's *whole* policy, changing the prompt mixture ratio moves *every* benchmark toward whichever teacher got the bigger share — regardless of which domain that teacher was assigned. They call this the **seesaw**. The generalisation that makes single-teacher OPD good is exactly what makes multi-teacher OPD uncontrollable. Same coin, two sides.

## The Methodology

The objective is reverse KL between student and teacher, on student-sampled trajectories:

$$\mathcal{L}_{\mathrm{OPD}}(\theta)=\mathbb{E}_{x\sim\mathcal{D},\, y\sim\pi_\theta(\cdot|x)}\left[\frac{1}{T}\sum_{t=1}^{T} D_{\mathrm{KL}}\big(\pi_\theta(\cdot|h_t)\,\|\,\pi_\phi(\cdot|h_t)\big)\right]$$

where $h_t = (x, y_{<t})$ is the context at step $t$, $\pi_\theta$ is the student, $\pi_\phi$ the teacher. Note the direction: **reverse** KL (student first), which is mode-seeking — see [[KL Divergence]].

In practice you estimate this with a single-sample $k_1$ estimator, which turns it into a policy-gradient loss:

$$\mathcal{L}^{\mathrm{PG}}_{\mathrm{OPD}}(\theta)=-\mathbb{E}_{x,y}\left[\frac{1}{T}\sum_{t=1}^{T}\hat{A}^{\mathrm{OPD}}_{t}\log\pi_\theta(y_t|h_t)\right],\qquad \hat{A}^{\mathrm{OPD}}_{t}=\operatorname{sg}\big[\log\pi_\phi(y_t|h_t)-\log\pi_\theta(y_t|h_t)\big]$$

`sg` is stop-gradient. Read it as [[Simple Statistical Gradient-Following Algorithms (REINFORCE)|REINFORCE]] where the per-token "reward" is the teacher's log-prob minus the student's. A token gets reinforced when the teacher likes it more than the student does, and suppressed otherwise. No verifier, no reward model, no [[Proximal Policy Optimization Algorithms|PPO]] clipping — just a dense token-level advantage.

**MOPD** is the same loss, but each prompt is routed to the domain teacher matching its domain.

**Setup.** Students: DS-distill-1.5B/7B/14B, Qwen3-8B-SFT, Qwen3-4B. Teachers: Qwen3-32B, Light-R1-14B/7B, Polaris-7B/4B, JustRL-1.5B, Nemotron-1.5B, DeepScaleR-1.5B, OpenMath-Nemotron, VibeThinker. Max 200 steps, prompt batch 128 (so ≤ 25.6K prompts total), 4 rollouts per prompt, lr $1e{-}5$, temperature 1.0, top-p 1.0, max sequence length 40K–96K depending on the pair.

**Data.** Math = Big-Math-RL-Verified. Code = DeepCoder-Preview. Science = TextbookReasoning / SCP-116K. IF = Llama-Nemotron-Post-Training-IF.

**Evaluation.** All Avg@K (average over K independent samples, *not* pass@K — no best-of selection). English math = mean of AMC2023, MATH-500, AIME2025, AIME2026, BeyondAIME, OlymMATH-Hard. Chinese math = OlymMATH-ZH + LiveMathBench-ZH. Long-horizon = R-HORIZON's AIME24-Horizon-2 and AMC23-Horizon-4, where several problems are chained so the answer to one feeds the next. Code = LiveCodeBench v5. Science = GPQA-Diamond. IF = IFEval.

## Ablation Studies and Experiments

**Teacher-side difficulty filtering — does nothing.** Sample 4 teacher rollouts per BigMath problem, build three 25K subsets: *easy* (teacher pass-rate = 1), *hard* (pass-rate = 0), *random*. All three converge to nearly identical final accuracy across Qwen3-32B→Qwen3-8B-SFT, Polaris-7B→DS-distill-1.5B, and Polaris-7B→DS-distill-7B. Extending to absolute extremes: GSM8K (grade school) and the hardest slice of DeepMath-103K both recover >80% of the BigMath-random gain, with final averages within 2 points. Diverse difficulty is still slightly better than either extreme, but the effect is small.

**Student-side dynamic sampling — small but real.** Table 1. Baseline (no filtering), Polaris-7B→DS-distill-1.5B: 41.4% average. Keeping only pass-rate = 0 problems: 41.4%. Only pass-rate = 1: 41.4%. Discarding only problems the student already fully solves (keep pass-rate $\in [0,1)$): **42.0%** (+0.6 pp). Same pattern for Light-R1-14B→DS-distill-7B: 52.4% baseline → **52.8%** (+0.4 pp), while both extremes were slightly *worse* (52.1%, 52.2%). Lesson: stop re-teaching what the student has mastered; do not throw away what it partly knows.

**Teacher's own RL prompts give no advantage.** Polaris-4B→Qwen3-4B trained on Polaris-53K (the actual data used to make Polaris) vs. random BigMath: same final performance. You do not need the teacher's training data.

**Origin beats strength.** For DS-distill-7B: cross-origin Light-R1-14B has clearly higher standalone accuracy than same-origin Polaris-7B, yet produces a much weaker student. On long-horizon math the cross-origin student shows essentially *no* gain over its starting point. Same-origin students land close to the teacher's line on English, Chinese, *and* long-horizon math.

**Cross-domain, both directions.** Train on math only → LiveCodeBench improves for both 1.5B and 7B students. Train on code, science, or even instruction-following only → math improves, including Chinese and long-horizon math. But the split is stark: for same-origin teachers the math-prompt curve and the code-prompt curve on LiveCodeBench both approach the teacher; for cross-origin teachers the code-prompt curve sits **consistently and substantially above** the math-prompt curve. Cross-origin OPD needs you to train on the target domain.

**What did not work / went backwards.** Training DS-distill-1.5B with JustRL-1.5B (a math-focused teacher whose *science* ability is worse than the student's) degrades GPQA-Diamond below the starting point — regardless of whether you feed it math or science prompts. The teacher's weaknesses transfer too.

**The seesaw.** Two teachers, JustRL-1.5B (better math) and Nemotron-1.5B (much better science/IF). Setting 1: JustRL = math teacher, Nemotron = science/IF teacher. As JustRL's share grows (Nemotron/JustRL from 1/1 to 2/25), GPQA-Diamond, LiveCodeBench, and IFEval **all fall** toward JustRL's lower levels; GPQA drops ~5% early in training. Setting 2 swaps the roles — Nemotron is now the math teacher — and those same three benchmarks now *rise* as Nemotron's share grows. The direction of movement is set by which teacher has more prompts, not by which domain that teacher was assigned.

Math side, Table 6, step 200 (DS-distill-1.5B starts at 8.8%): with JustRL as math teacher, JustRL/Nemotron = 25/8 gives 19.6%; 1/1 gives 19.1%; 2/25 gives 17.4%; 0/25 gives 14.9%. With roles reversed (Nemotron is the math teacher), JustRL/Nemotron = 25/0 still gives the best math at **17.5%** versus 15.8% at 1/1 — even though JustRL supervises *no math prompts at all* in that setting.

**Tug-of-war over time.** On AIME24-Horizon-2, MOPD students first track the JustRL-only curve and later drift down toward the Nemotron-only curve. A cascaded run makes it cleanest: train with Nemotron + science prompts, GPQA rises sharply; switch to JustRL + math prompts, GPQA falls back down. Nothing is locked in.

**The mixture ratio can invert.** DS-distill-7B with cross-origin Light-R1-14B as math expert and same-origin Polaris-7B as science/IF expert. Raising the *math* share (8/25 → 25/8) makes math accuracy **drop** on all four math benchmarks. Reason: Polaris-7B is itself a strong math model (95.2% AMC2023, 62.5% AIME2026) and, being same-origin, transfers that math ability for free through science prompts. Single-teacher OPD from the cross-origin math expert alone is the *worst* run on every benchmark. More math data displaced the signal that was actually teaching math.

## Worth Remembering

- The top-16 overlap ratio is the diagnostic to steal. Both settings drive the same KL down, but overlap rising vs. flat tells you whether you are getting whole-policy alignment or local curve-fitting. Cheap to log during training.
- **Check the lineage before you pick a teacher.** A same-origin 7B can beat a cross-origin 14B. Table 3 in the appendix is a nice worked example of what "same lineage" means: DS-R1-Distill-Qwen-1.5B is the root for JustRL, DeepScaleR, Nemotron-Research-Reasoning, and Dev-1.5B.
- Prompt routing in MOPD is **not a safety boundary**. The authors flag this explicitly in the ethics section: undesirable teacher behaviours will leak into domains you never routed to that teacher. Evaluate across all domains, not just the assigned one.
- Diagnostic corollary: if the student is weak on science, the science teacher may not be the problem. Look at every teacher's off-domain profile, not just its headline number in its own domain.
- A student can exceed its teacher. Light-R1-14B → DS-distill-14B ends up clearly above the standalone teacher on long-horizon math. The teacher's benchmark score is not a ceiling.
- Connects to the [[Chain-of-Thought Prompting Elicits Reasoning in LLMs|chain-of-thought]] literature nicely: "reasoning behaviour transfers, answers do not" is the distillation-side version of the same claim.
- Limits the authors own: reasoning-only models, four text domains, exactly two teachers per MOPD run, fixed domain routing. No multimodal, no tool use, no agents. No adaptive router. Larger expert pools untested.
- Open question the paper does not answer: *why* does shared initialisation make whole-policy alignment easy? "Distributional gap" is a description, not a mechanism. Is there a measurable threshold of initial overlap above which OPD generalises?
- Practical caveat: the runs are short (≤200 steps, ≤25.6K prompts). Long-run behaviour, especially the tug-of-war dynamics, may not have settled.

## Links

Related: [[Distilling the Knowledge in a Neural Network]] · [[KL Divergence]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[Proximal Policy Optimization Algorithms]] · [[Training language models to follow instructions with human feedback]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Distillation]] · [[Direct Preference Optimization (DPO)]] · [[Foundation Models]] · [[Sparsely-Gated Mixture-of-Experts Layer]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Inject, Align, Recover- Staged Post-Training for Retrieval-Free Document Knowledge Internalization]]

New topics worth writing: On-policy distillation, Reverse KL vs forward KL in sequence models, Exposure bias, k1/k2/k3 KL estimators, Multi-teacher distillation and capability interference, Model lineage and post-training compatibility, Top-K distribution overlap as an alignment metric, Long-horizon / compositional reasoning benchmarks (R-HORIZON), Avg@K vs Pass@K evaluation, GRPO and verifier-free RL post-training, Catastrophic forgetting in multi-domain fine-tuning
