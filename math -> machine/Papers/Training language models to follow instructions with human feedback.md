---
title: "Training language models to follow instructions with human feedback"
authors: ["Long Ouyang", "Jeff Wu", "Xu Jiang", "Diogo Almeida", "Carroll L. Wainwright", "Pamela Mishkin", "Chong Zhang", "Sandhini Agarwal", "Katarina Slama", "Alex Ray", "John Schulman", "Jacob Hilton", "Fraser Kelton", "Luke Miller", "Maddie Simens", "Amanda Askell", "Peter Welinder", "Paul Christiano", "Jan Leike", "Ryan Lowe"]
year: 2022
arxiv: "2203.02155"
url: https://arxiv.org/abs/2203.02155
priority: Must-Read
read_on: 2026-08-22
tags: [paper, llm, rl]
---
## The Core Idea

A language model trained to predict the next word on the internet is good at *continuing text*. It is not good at *doing what you asked*. Those are two different jobs. GPT-3 will answer "Explain the moon landing to a six year old" by writing more questions in the same style, because that is what a web page would do next. Making the model bigger does not fix this — the objective itself is pointed at the wrong target. The paper calls this **misalignment**.

The fix is not a better prompt or a bigger model. It is to change what the model is optimised for, using humans as the judge. Three steps: hire people to write good answers and fine-tune on them; hire people to *rank* several model answers and train a small model to predict those rankings; then use that predictor as a reward and run reinforcement learning on the big model.

The headline result is the reason this paper mattered: a **1.3B parameter InstructGPT is preferred by human labelers over the 175B GPT-3** — a model 100x larger, same architecture, only difference is the human-feedback fine-tuning. And the whole alignment procedure cost 60 petaflop/s-days versus 3,640 for pretraining GPT-3. So for the money, buying alignment beat buying scale by a huge margin. This is the direct ancestor of ChatGPT and of every "chat" model since.

The thing that unlocks it is the **reward model**. You cannot write down a loss function for "helpful, honest, harmless". But you *can* ask a person which of two answers is better, and you can fit a model to those answers. That turns a fuzzy human judgement into a differentiable scalar you can hill-climb on.

> [!NOTE] RLHF (Reinforcement Learning from Human Feedback)
> Train a model to predict human preferences, then use that model's output as the reward signal for reinforcement learning on the policy you actually care about. Humans only ever compare outputs; they never write the reward number themselves. ^rlhf

## The Methodology

Everything starts from pretrained GPT-3 ([[Language Models are Few-Shot Learners (GPT-3)]]), same architecture, context length 2048, fp16 weights with fp32 master copies (see [[Mixed Precision training]]), Adam with $\beta_1 = 0.9, \beta_2 = 0.95$. Three sizes: 1.3B, 6B, 175B.

### The data

Prompts came from two places. To bootstrap, ~40 hired contractors wrote prompts themselves (plain tasks, few-shot examples, and tasks copied from API waitlist applications). After an early InstructGPT was deployed, real prompts submitted to the OpenAI API Playground became the main source. Splits are by user ID, so the test set has no users from training.

Three datasets:

| Dataset | Prompts | What's in it |
|---|---|---|
| SFT | 13k | human-written ideal answers |
| RM | 33k | human rankings of model outputs |
| PPO | 31k | prompts only, no labels |

The mix is mostly open-ended: 45.6% generation, 12.4% open QA, 11.2% brainstorming, 8.4% chat. Classification and QA together are only ~18%. Over 96% English.

Labelers were screened on a test — agreement with researchers on flagging sensitive content, agreement on rankings, and quality of demonstrations on delicate prompts. They agree with each other 72.6 ± 1.5% of the time. For scale, researcher-researcher agreement in the earlier summarisation work was 73 ± 4%. So this is roughly as consistent as experts.

### Step 1 — Supervised fine-tuning (SFT)

Plain [[Cross Entropy]] next-token loss on the human demonstrations. 16 epochs, residual dropout 0.2, cosine LR decay to 10%, no warmup. LR 9.65e-6 / batch 32 for 1.3B and 6B; 5.03e-6 / batch 8 for 175B.

Odd detail worth keeping: **validation loss overfits after 1 epoch, but human preference and reward-model score keep improving for 16 epochs.** They selected checkpoints by reward-model score, not loss. Classic case where the proxy metric ([[Regularization|held-out loss]]) lies about the thing you care about.

### Step 2 — Reward model

Take the SFT model, throw away the unembedding layer, bolt on a [[Linear Projection|linear projection]] to a single scalar. Input is (prompt, response), output is $r_\theta(x,y)$.

Labelers see $K$ responses at once, $K \in [4,9]$, and rank them. This gives $\binom{K}{2}$ pairwise comparisons per prompt. The loss is pairwise logistic — a two-outcome [[Cross Entropy]] on the difference in scores:

$$\text{loss}(\theta) = -\frac{1}{\binom{K}{2}}\,\mathbb{E}_{(x,y_w,y_l)\sim D}\Big[\log \sigma\big(r_\theta(x,y_w) - r_\theta(x,y_l)\big)\Big]$$

where $y_w$ beat $y_l$. The score *difference* is the log-odds that a human prefers $w$ over $l$.

Two implementation choices that mattered a lot:

1. **All $\binom{K}{2}$ comparisons from one prompt go in the same batch element.** If you shuffle them into a flat dataset, the model overfits within a single epoch, because comparisons from the same prompt are heavily correlated — you'd see the same completion many times in different pairs. Batching them together also means one forward pass per completion instead of $\binom{K}{2}$.
2. **Only 6B reward models.** 175B RMs could reach lower validation loss but training was unstable, and using a 175B value function would blow up the PPO compute.

Trained 1 epoch, LR 9e-6, batch 64 *prompts* (so up to $64\times 36 = 2304$ comparisons). Insensitive to LR (±50% is fine), extremely sensitive to epochs — a second epoch overfits visibly. Finally, since the loss only cares about differences, they add a bias so that labeler demonstrations score a mean of 0.

> [!NOTE] Reward model
> A network that eats (prompt, answer) and outputs one number. Trained only on which of two answers a human liked more, so its absolute scale is meaningless — only differences carry information. ^reward-model

### Step 3 — PPO against the reward model

The RL setup is a **bandit**, not a long-horizon [[Markov Decision Process]]: sample one prompt, emit one full response, get one reward, episode over. The objective:

$$\text{objective}(\phi) = \mathbb{E}_{(x,y)\sim D_{\pi^{RL}_\phi}}\Big[r_\theta(x,y) - \beta \log \frac{\pi^{RL}_\phi(y\mid x)}{\pi^{SFT}(y\mid x)}\Big] + \gamma\, \mathbb{E}_{x\sim D_{\text{pretrain}}}\big[\log \pi^{RL}_\phi(x)\big]$$

Three terms:

- $r_\theta(x,y)$ — the learned reward.
- The per-token **KL penalty** against the frozen SFT model ([[KL Divergence]]), $\beta = 0.02$. Without it the policy drifts into weird text that scores high on the reward model but is garbage — the reward model is only accurate near the distribution it was trained on. This is the leash.
- The **pretraining mix**, $\gamma = 27.8$: gradients from ordinary next-token prediction on the original GPT-3 data, mixed in to stop the model forgetting how to do normal NLP. Setting $\gamma = 0$ gives the plain "PPO" model; $\gamma > 0$ gives "PPO-ptx", which is what "InstructGPT" means in the paper.

Concretely, for each minibatch they compute the PPO gradient and the pretraining gradient in consecutive steps and accumulate both into the buffer, with 8x more pretraining examples than RL episodes.

Hyperparameters: 256k episodes, batch 512, minibatch 64 (8 inner steps, single inner epoch), PPO clip ratio 0.2, sampling temperature 1, no discount in generalised advantage estimation, EMA of weights with decay 0.992, constant LR with 10-iteration warmup. The value function is a 6B model initialised from the reward model, LR 9e-6. The policy is initialised from an SFT model trained 2 epochs with 10% pretraining mix.

## Ablation Studies and Experiments

Main metric: how often labelers prefer a model's output to the 175B SFT model's output, on held-out API prompts from customers never seen in training.

**Preference results (175B, vs 175B GPT-3 head to head)**
- InstructGPT preferred **85 ± 3%** of the time over plain GPT-3.
- Preferred **71 ± 4%** over GPT-3 with a hand-crafted few-shot instruction prefix.
- Ladder of improvement: GPT-3 < GPT-3 prompted < SFT < PPO ≈ PPO-ptx.
- **1.3B PPO-ptx beats 175B GPT-3.**

**Held-out labelers.** People who produced none of the training data prefer InstructGPT at about the same rate. A 5-fold cross-validation over labeler groups shows the RM predicts held-out labelers' preferences at 69.6 ± 0.9% versus 72.4 ± 0.4% for in-group. Small drop — it is not just memorising 40 people's quirks.

**Truthfulness.** On TruthfulQA, InstructGPT gives truthful-and-informative answers about twice as often as GPT-3, without being told to be truthful. On closed-domain tasks (summarise this, answer from this passage), hallucination rate falls from **41% to 21%**.

**Public NLP instruction datasets are the wrong target.** They fine-tuned 175B GPT-3 on ~1M examples of FLAN and of T0++. Both come out *worse than the SFT baseline*. InstructGPT beats FLAN 78 ± 4% and T0 79 ± 4% head-to-head. Winrates against the SFT baseline: InstructGPT 73.4 ± 2%, T0 26.8 ± 2%, FLAN 29.8 ± 2%. The reason: those datasets are built from tasks that automatic metrics can score — classification, QA — which is 18% of real usage, while generation and brainstorming are 57%.

### What did *not* work

- **Turning up the KL coefficient instead of mixing pretraining data.** They swept $\beta$ up to 2.0 (100x the default) with $\gamma = 0$. It never fully recovers DROP and SQuADv2, and it tanks the validation reward. The pretraining *distribution* is what matters, not just staying close to the SFT policy. Also, $\beta = 0$ and $\beta = 2$ both give bad Likert scores; the sweet spot is 0.01–0.02.
- **Training longer.** Running PPO-ptx for 512k episodes instead of 256k: DROP and SQuADv2 start *above* GPT-3 and then slide back below it. The alignment tax comes back if you keep going.
- **175B reward models.** Lower validation loss, but unstable training and unsuitable as a value-function init.
- **More than one RM epoch.** Immediate overfit.
- **Bias benchmarks.** No improvement on Winogender or CrowS-Pairs. Worse: adding "respond respectfully" *lowers* the entropy of the model's choice between the two sentences in a pair, i.e. makes it *more* confident and therefore more biased by their metric — regardless of which way the bias points.
- **Toxicity is conditional, not intrinsic.** With a "be respectful" instruction, InstructGPT is ~25% less toxic than GPT-3 on RealToxicityPrompts. With no instruction, the advantage vanishes. When *told* to be maximally biased and offensive, InstructGPT is **much more toxic than GPT-3** — it follows the instruction better, including the bad ones.
- **Pretraining data ratio.** At a ratio of 4, the pretraining log-loss climbs during training. Ratio 32 gives better Likert scores but multiplies training time; they settled on 8. Minibatch 32 was slightly better than 64, but they used 64 for GPU utilisation.
- **SFT for PPO init.** They tried 1 and 2 epochs of demonstrations with 0%, 10%, 50% pretraining mix. Only 10% stood out, and even then PPO was not very sensitive.

### The alignment tax

Plain PPO regresses on SQuADv2, DROP, HellaSwag, and WMT15 Fr→En. PPO-ptx removes most of this and even beats GPT-3 on HellaSwag, at no cost to labeler preference. It still lags GPT-3 on DROP, SQuADv2, and translation.

> [!NOTE] Alignment tax
> The capability you lose on other tasks as the price of making a model follow instructions. If the tax is high, people will just deploy the unaligned model, so low-tax alignment methods matter for adoption, not just for niceness. ^alignment-tax

## Worth Remembering

**Failures they admit.** InstructGPT accepts false premises ("Why is it important to eat socks after meditating?" gets a long, earnest answer about theories). It hedges too much — they suspect this is because labelers were told to reward epistemic humility, and the reward model learned "hedge = good". It degrades with multiple explicit constraints ("list 10 movies from the 1930s set in France"). And it will follow harmful instructions, because during *training* labelers were told to prioritise helpfulness over harmlessness (in final *evaluation* the order was reversed).

**Who is "human" in human feedback.** ~40 contractors, 75% under 35, mostly US or Southeast Asia, hired through Upwork and Scale AI. Most comparisons were labeled by exactly one person, for cost. The researchers wrote the labeling instructions. The prompts came from OpenAI API customers, who came off a waitlist seeded by OpenAI employees. Section 5.2 is unusually honest about this: it is alignment to a specific small group's stated preferences, not to "human values".

**Generalisation is the surprising bit.** The fine-tuning data is >96% English and contains almost no code, yet InstructGPT follows French instructions and answers questions about Python. GPT-3 can be prompted into these behaviours but does not do them by default. "Follow the instruction" seems to be learned as an abstract mode, not memorised per task — related to how [[In Context Learning]] emerges without being trained for.

**Practical caveats if you were to build this.**
- The reward model is the bottleneck and the failure mode. Everything downstream inherits its blind spots, and PPO will actively hunt for them. The KL penalty is the only thing stopping it.
- Pick SFT checkpoints by reward-model score, not validation loss.
- One epoch on the reward model. Group comparisons from the same prompt into one batch element.
- Budget: alignment cost ~1.6% of pretraining compute here.

**Connections.** The reward model loss is the Bradley-Terry pairwise preference model, which later gets folded directly into the policy loss by DPO, removing the RL loop entirely. The KL-to-reference term is the same idea as trust regions in policy optimisation and the same object as in [[KL Divergence]]. The over-optimisation problem — policy exploits reward model, output quality collapses while reward climbs — is a cousin of [[Mode Collapse]] in [[Generative Adverserial Network]]s: an optimiser winning against a learned critic rather than against reality.

**Open question worth chasing.** Comparisons are a low-bandwidth signal. The authors flag natural-language critiques and labeler edits as richer alternatives — both of which show up in later work.

## Links

Related: [[Language Models are Few-Shot Learners (GPT-3)]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[Attention Is All You Need]] · [[KL Divergence]] · [[Cross Entropy]] · [[Markov Decision Process]] · [[In Context Learning]] · [[Loss, Objectives, and Business Alignment]] · [[Mixed Precision training]] · [[Foundation Models]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[Mode Collapse]] · [[Decision Sciences]]

New topics worth writing: Proximal Policy Optimization (PPO), Bradley-Terry preference model, Generalized Advantage Estimation, Direct Preference Optimization (DPO), reward hacking / over-optimisation of learned rewards, TruthfulQA, FLAN and T0 instruction tuning, Constitutional AI, alignment tax
