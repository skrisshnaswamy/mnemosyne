---
title: "VA-Judger: Reward Modeling from Human Preference Feedback for Joint Video-Audio Generation"
authors: ["Yinming Huang", "Shuyuan Tu", "Xi Yan", "Zihan Yang", "Jianhua Han", "Hang Xu", "Kaihang Pan", "Yu-Gang Jiang", "Zuxuan Wu"]
year: 2026
arxiv: "2608.18607"
url: https://arxiv.org/abs/2608.18607
priority: Low-Priority
read_on: 2026-09-30
tags: [paper, llm, rl, vision]
---
## The Core Idea

Video generators now make sound and picture together. To improve one with reinforcement learning, you need a number that says "this clip is good". Today that number is a bag of separate scores glued together: one model rates the picture, another rates the audio, another checks the lips match the voice, another checks the sound matches the text.

That bag is the problem. Each score looks at one slice. None of them looks at the whole thing as one event. A clip can score high on lip-sync while showing the wrong action. It can look beautiful while the sound is emotionally wrong for the scene. Add the scores up and you get a target the generator can game — the classic [[Reward Hacking|Goodhart]] failure, where the measure stops being the goal.

The numbers back this up. On 1,150 human-labelled pairs, the seven standard metrics predict which clip a human prefers with 50.35%–57.04% accuracy. Coin-flip is 50%. Ensembling all seven only reaches 58.70%, and on clips from unseen generators it drops to 52.40%.

So: build a reward model from human preferences instead, the way [[Training language models to follow instructions with human feedback|RLHF]] did for text and [[Direct Preference Optimization (DPO)|preference learning]] did for images. Nobody had done this for joint video+audio, and you cannot borrow an existing one — a text-to-video reward model never hears the audio, and an audio scorer never sees the picture. Human taste here is not the sum of two separate opinions.

**VA-Judger** is a single model that watches the video, hears the audio, reads the prompt, writes out a scored comparison on five dimensions, then picks a winner. It hits 68.43% agreement with humans overall — about 10 points over an untuned omni-model, and roughly 10 points over the best metric ensemble.

> [!NOTE] Omni-reward model
> One model that takes text, video and audio at once and returns a preference over two candidate clips, instead of several single-modality scorers whose outputs get summed. ^omni-reward-model

## The Methodology

### The task shape

Input: a prompt $p$, two clips $x_1 = (v_1, a_1)$ and $x_2 = (v_2, a_2)$, plus a fixed system prompt holding the rubric. Output: a [[Chain-of-Thought Prompting Elicits Reasoning in LLMs|chain-of-thought]] comparison — for each clip, a score from 1 to 10 with a short justification on five dimensions:

- **A** prompt alignment
- **B** video-audio consistency
- **C** audio quality
- **D** video quality
- **E** completeness and coherence

Then a total, then `<answer>video k is better</answer>`.

Forcing both clips into one input is the whole point. The model cannot produce a video score and an audio score in isolation; it has to look at them together.

### The data: VAPref-10K

Prompts come from real clips, not synthesised taxonomies. 10,173 real video-audio segments from YouTube, Bilibili, films and TV, trimmed to 5–10 seconds, across six categories (multi-person dialogue, voice-over narration, music, cinematic ambience, environmental events, talks).

Pipeline: caption each real clip with Qwen3.5-Omni → the caption is full of timestamps and fragmented events, useless as a generation prompt → rewrite with Qwen into a flowing prompt that keeps the visuals, the sound events, the speech and the cross-modal relations. That gives 10,083 candidates; filtering leaves ~9K usable prompts.

Then generate candidate clips and pair them. Pairing is split by difficulty on purpose:

| Pool | Size | How built | Why |
|---|---|---|---|
| Easy | 4,390 | Ovi vs LTX-2 (and Ovi vs daVinci-MagiHuman for speech categories) | LTX-2 is clearly better, so the winner is obvious |
| Hard | 9,146 | Same prompt, same model, different random seed | Winner depends on subtle differences |

All hard pairs are labelled by humans: pick the winner, then mark *which rubric dimensions justify it*. Annotators never assign absolute scores — absolute scoring drifts badly between people. Only comparisons.

**VA-Judger-Bench** is held out: 400 easy, 250 in-domain hard, 500 out-of-domain. The out-of-domain half pairs outputs from Kling 2.6, Wan 2.6, Veo 3.1 Fast, Veo 3.1 Quality and Sora 2 — none of which appear in training.

### Stage 1 — easy cold start

Start from Qwen3-Omni-30B-A3B-Instruct. Gemini 3.1 Pro writes full rubric responses for the easy pairs; train on those for 400 updates. Gemini agrees with humans on 80% of easy pairs in a 200-pair pilot, which is good enough when the goal is just teaching format and coarse discrimination. This is [[Fine-Tuning|supervised fine-tuning]] as a format lesson: learn to emit five dimensions, aggregate them, close the `<answer>` tag.

Classic curriculum learning — easy before hard. Language backbone trains; visual encoder and aligner frozen. 8 GPUs, BF16, ZeRO-3, peak LR $5\times10^{-6}$, warmup ratio 0.05, seq length 24,576, effective batch 128.

### Stage 2 — hard pairs, filtered by humans

On hard pairs Gemini's agreement with humans falls to **60%** in a 200-pair pilot. So its verdicts cannot be trusted as labels.

The fix is rejection sampling with a three-part filter. Gemini still writes the reasoning; a response survives only if:

1. Its final pick matches the human pick.
2. It parses cleanly against a regex format check.
3. For **every** dimension $d$ the human marked as supporting, the winning clip scores strictly higher: $s_{y^\star, d} > s_{3-y^\star, d}$ for all $d \in \mathcal{D}_{\mathrm{h}}$.

From 9,146 raw hard pairs, 4,436 responses survive. Train 1,000 more updates on those. The model gets well-formed reasoning traces that are *pinned to human judgment at the dimension level*, not just at the verdict.

### Stage 3 — dimension-wise GRPO

[[Fine-Tuning|SFT]] only teaches imitation. The likelihood loss weighs every token equally and never asks whether the model's *own sampled* judgments agree with a human. A model can reach low SFT loss while writing dimension scores that contradict its own final answer.

So switch to [[GRPO]] with a verifiable reward. Sample $G = 8$ responses per pair. A single binary correct/incorrect reward would be too sparse — the model can guess the right winner while scoring the dimensions incoherently, which is a shortcut, not learning. Instead, two components:

$$r_{\mathrm{ans}}(o) = \mathbf{1}[\hat{y} = y^\star], \qquad r_{\mathrm{dim}}(o) = \frac{1}{|\mathcal{D}_{\mathrm{h}}|}\sum_{d \in \mathcal{D}_{\mathrm{h}}} \mathbf{1}\!\left[s_{y^\star, d} > s_{3-y^\star, d}\right]$$

$$R_i = \tfrac{1}{2}\left(r_{\mathrm{ans}}(o_i) + r_{\mathrm{dim}}(o_i)\right)$$

In words: half your marks for picking the clip the human picked; half for scoring *each dimension the human cited* in the direction the human implied. Say the human preferred clip 2 and cited dimensions B and D. A response that says "video 2 is better" but rates clip 1 higher on D gets $r_{\mathrm{ans}} = 1$, $r_{\mathrm{dim}} = 0.5$, so $R = 0.75$. Full agreement gets 1.0.

> [!NOTE] Dimension-wise reward
> Grade the reasoning, not only the verdict. Every human-cited dimension becomes its own check, turning one bit of supervision into several. ^dimension-wise-reward

Advantages are group-normalised, the [[Policy Gradient|policy gradient]] is advantage-weighted log-likelihood:

$$A_i = \frac{R_i - \bar{R}}{\sigma_R + \epsilon}, \qquad \mathcal{J}_{\mathrm{GRPO}}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G} \frac{A_i}{|o_i|}\sum_{t=1}^{|o_i|} \log \pi_\theta(o_{i,t} \mid x, o_{i,<t})\right]$$

Two things deliberately absent: no [[PPO]] ratio clipping, no [[KL Divergence|KL]] penalty against a reference model. The authors say this keeps the objective consistent with the sampled group-relative advantages and avoids holding a second copy of the model in memory. 600 updates on 9,136 pairs (10 "Other-only" annotations dropped, since $|\mathcal{D}_{\mathrm{h}}| > 0$ must hold). Encoders frozen, thinker trained, 6 training GPUs + 2 rollout servers, temperature 1.0, Adafactor at $1\times10^{-6}$.

### Using it as a reward

Policy is LTX-2 (19B), backbone frozen, [[LoRA]] adapters at rank 32, scaling 64, inserted into video attention, audio attention, feed-forwards and the cross-modal attention layers. Current and old policies get separate adapters; disabling adapters recovers the reference policy.

Per prompt: old policy makes 8 candidates → 28 unique pairs → VA-Judger scores every pair → each candidate averages its scores over its 7 comparisons. Then the routing, which is the neat bit:

- dimension **C** → audio reward → audio branch
- dimension **D** → video reward → video branch
- mean of **A, B, E** → shared cross-modal reward → both branches

Each reward normalised within the group of 8. Training: OmniNFT's 20K-prompt VGGSound split, 400 updates, 121 frames at $512\times768$, 24 FPS, 20 denoising steps, AdamW at $3\times10^{-5}$, BF16, FSDP on 7 GPUs with 1 GPU hosting the judge as an inference server.

## Ablation Studies and Experiments

### Reward model accuracy (1,150 pairs)

| Model | Easy | In-domain | Out-of-domain | Overall |
|---|---|---|---|---|
| VideoAlign (video quality) | 62.25 | 53.20 | 48.40 | 54.26 |
| Audiobox Aesthetics (audio) | 58.50 | 50.00 | **44.00** | 50.35 |
| CLIP Score (text-video) | 55.75 | 54.80 | 53.80 | 54.70 |
| ImageBind A-V | 61.25 | 55.60 | 51.80 | 55.91 |
| SynchFormer offset | 61.50 | 46.80 | 48.20 | 52.52 |
| Javis Score (aggregate) | 60.00 | 55.60 | 55.40 | 57.04 |
| Ensemble: hard voting | 64.50 | 52.00 | 50.00 | 55.48 |
| Ensemble: Z-score soft voting | 67.00 | 58.00 | 52.40 | 58.70 |
| Qwen3-Omni Instruct, no CoT | 60.75 | 54.00 | 54.60 | 56.61 |
| Qwen3-Omni Instruct, CoT | 63.25 | 54.80 | 55.00 | 57.83 |
| **VA-Judger** | **76.25** | **66.00** | **63.40** | **68.43** |

Read the out-of-domain column first. Audiobox Aesthetics at 44.00% is *worse than a coin flip* — on unseen generators it actively disagrees with people. VideoAlign, the strongest video metric, drops to 48.40%. These are the metrics OmniNFT optimises against.

The ensemble result is the sharper finding. Combining all seven metrics buys 1.66 points overall against the best single metric, but *loses* 3.00 points out-of-domain. The metrics do hold complementary information, but stacking them label-free does not survive a change in which generator produced the clips.

### What the training stages are worth

Two clean comparisons exist:

- **CoT vs no CoT on the untuned base.** Qwen3-Omni Instruct goes 56.61 → 57.83 by reasoning first. Just under 1.2 points. Reasoning format alone is nearly worthless.
- **Untuned CoT vs VA-Judger.** 57.83 → 68.43. Everything that matters comes from the three training stages, not from the prompt.

A caveat worth flagging: the paper does **not** report a stage-by-stage ablation of the reward model. There is no number for Stage 1 only, or Stages 1+2 without GRPO. The 10.6-point gain is attributed to easy-cold-start + human-filtered hard SFT + dimension-wise GRPO as a bundle, and the relative contribution of each is unmeasured. For a paper whose central methodological claim is that a three-stage curriculum is the right recipe, that is the missing table.

### What does not work: the captioner

Qwen3-Omni Captioner **with** CoT scores 49.30 Total Acc against 56.00 without it. Adding reasoning made it worse. The gap between Total Acc (49.30) and Parsed Acc (56.76) explains why: CoT made the captioner produce responses the regex could not parse into a valid `<answer>`, and unparseable counts as wrong. Reasoning does not help a model that has not been taught the output contract.

### Post-training LTX-2 (200 JavisBench prompts)

VA-Judger leads 17 of 20 metrics. Headline moves from base LTX-2:

| | LTX-2 | +OmniNFT | +VA-Judger |
|---|---|---|---|
| Visual quality | 2.248 | 3.727 | **3.942** |
| Motion quality | 0.697 | 0.947 | **1.183** |
| Audio quality | 4.767 | 5.399 | **5.610** |
| CLAP (text-audio) | 0.304 | 0.394 | **0.425** |
| AV-ImageBind | 0.091 | 0.154 | **0.265** |
| AVHScore | 0.091 | 0.146 | **0.261** |
| JavisScore | 0.074 | 0.122 | **0.230** |
| DeSync ↓ | 0.430 | **0.226** | 0.592 |

### The DeSync result is the most honest thing in the paper

OmniNFT wins DeSync (0.226 vs 0.592) — and DeSync is in OmniNFT's own reward. Meanwhile VA-Judger's DeSync gets *worse* than the untrained LTX-2 base (0.430 → 0.592) while every human-facing measure improves and 62.30% of human votes go to it.

Two readings, and the paper only offers one. Their reading: OmniNFT's DeSync lead is metric-specific [[Reward Hacking|reward hacking]], direct optimisation of the thing being measured. The reading they skip: their own model may genuinely have drifted on audio-visual timing, and DeSync is catching something real that human raters watching 5-second clips did not weight heavily. The human vote is the stronger evidence, but a 38% relative degradation on a synchronisation metric, in a paper about joint video-audio generation, deserved more than one sentence.

### Ablating the reward model itself

The `LTX-2 + Qwen3-Omni-RL` row is the useful control: same rubric, same pairwise aggregation, same LoRA config, same prompts — but the *untuned* Qwen3-Omni as judge. It improves on base LTX-2 (JavisScore 0.074 → 0.122) but lands roughly at OmniNFT's level and well below VA-Judger (0.230). So the reward-model **training** is doing the work, not merely the switch to an omni-model judge or the rubric prompt.

### Human evaluation

20 adults, 200 three-way comparisons, 4,000 selections: VA-Judger 62.30%, OmniNFT 27.63%, base LTX-2 10.08%.

## Worth Remembering

**The transferable trick is the dimension-wise reward.** A binary preference label is one bit. Asking the annotator *which dimensions justify the choice* turns it into several bits, and those bits become independent verifiable checks during [[GRPO]]. This generalises far beyond video: anywhere you collect pairwise preferences and the annotator could plausibly tell you *why*, you are leaving supervision on the floor by storing only the winner. Relevant to anyone building [[Preference Learning|preference]] pipelines for ranking or recommendation.

**Human annotators compare, they do not score.** The paper is explicit: absolute scalar ratings drift across experts because of subjective calibration. Same reason [[BPR- Bayesian Personalized Ranking from Implicit Feedback|BPR]] uses pairs and [[NDCG]] evaluation needs relevance grades to be consistent. Collect comparisons; let the model produce scalars.

**Rejection sampling against human labels is a cheap way to get reasoning traces you can trust.** Gemini writes the prose; humans supply the verdict and the dimensions; the filter keeps only traces where the prose agrees with the verdict *and* the dimension ordering. 9,146 → 4,436 survival rate is 48.5%, which is the price. Compare [[Distilling the Knowledge in a Neural Network|distillation]], where you take whatever the teacher says.

**Out-of-domain is where every automatic metric dies.** In-domain, the metric ensemble hits 58.00%; out-of-domain, 52.40%. VA-Judger also drops (66.00 → 63.40), but less. Any evaluation of a reward model that only tests on the generators that produced the training data is telling you nothing about deployment. This is the same lesson as [[Off-Policy Evaluation|OPE]]'s "validated on whose logs?" question.

**Omissions to hold in mind.**
- No per-stage ablation of the reward model, as noted.
- No ablation of the reward split during post-training — is the A/B/E → both branches, C → audio, D → video routing actually better than one scalar? Unmeasured.
- No [[KL Divergence|KL]] leash on the GRPO stage and no ratio clipping. The stated motivation is memory and objective consistency; the usual risk is the judge drifting from its SFT behaviour in ways the verifiable reward cannot detect.
- The generator evaluation is 200 prompts and 5–10 second clips. Short.
- VA-Judger is a 30B [[Mixture of Experts|MoE]] omni-model doing 28 pairwise comparisons per prompt during training. That is a full extra GPU serving the judge. The reward is not cheap.

**Connection worth chasing.** The failure mode here — ensemble of proxy metrics, optimised against, produces content that scores well and looks wrong — is exactly the story of offline recommender metrics in [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|the RecSys reproducibility work]] and [[On Sampled Metrics for Item Recommendation (KDD)|sampled metrics]]. Different domain, same disease: the measurement layer was never validated against the thing it stands in for.

## Links

Related: [[Reward Hacking]] · [[GRPO]] · [[Training language models to follow instructions with human feedback]] · [[Preference Learning]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[LoRA]] · [[Fine-Tuning]] · [[Evaluating Generative Models]] · [[Video Diffusion]] · [[Off-Policy Evaluation]] · [[Policy Gradient]] · [[KL Divergence]] · [[PPO]] · [[Evals]] · [[Distilling the Knowledge in a Neural Network]]

New topics worth writing: Rejection sampling for distillation data, Curriculum learning, Rubric-based reward models, Joint audio-video generation, Verifiable rewards (RLVR), Metric ensembling and its failure under distribution shift, Annotator calibration in preference collection
