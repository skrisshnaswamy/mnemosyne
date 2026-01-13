---
title: "τ_0-VLA: a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation"
authors: ["Xiaowei Cai", "Yunuo Cai", "Bingao Chen", "Jingxiao Chen", "Zhi Chen", "Siyuan Feng", "Tengyu Hou", "Jingshun Huang", "Han Jiang", "Runkun Ju", "Dong Li", "Mingxiang Li", "Shaowei Li", "Xinchen Li", "Yifan Li", "Yi Liu", "Zhongyuan Liu", "Jianlan Luo", "Junwen Miao", "Ruiqi Ni"]
year: 2026
arxiv: "2608.16885"
url: https://arxiv.org/abs/2608.16885
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, llm, diffusion, vision]
---
## The Core Idea

A robot that cleans a room has to make two different kinds of decision. **What should I do next?** (put the shirt in the basket, then hang the bag) and **how do I move my arm to do it?** Most vision-language-action models answer the first question with one forward pass of a language model and then move on. If that answer is wrong, the arm executes the wrong thing perfectly, the world changes, and no amount of motor precision can undo it.

$\tau_0$-VLA's claim: the "what next" decision should get a **variable amount of thinking time**, the same way a reasoning LLM spends more tokens on a hard maths question. This is test-time compute, but for subtasks instead of text.

The trick that makes it work is *how* candidate subtasks get compared. Language-only reasoning can argue about "should I add salt now?" but has no grip on physical reality. Action-level search (sampling motor trajectories) sees physics but only over half a second. Subtasks sit in the sweet spot: rare enough that spending a second of GPU time on each is affordable, long enough that each one visibly changes the scene. So you can **imagine the picture that each candidate subtask would produce**, and score that picture.

> [!NOTE] Consequence-aware subtask search
> Before committing to a subtask, generate $N$ candidates in plain language, use an image-editing world model to predict the head-camera photo at the *end* of each candidate, and let a value model grade those imagined photos. Keep the best $B$ branches, expand to depth $D$, then let a separate "reflective" model write the final subtask. ^consequence-search

Two more pieces make this practical. First, an **adaptive router**: search is expensive, so the system only pays for it when the fast single-pass proposal looks uncertain, measured from token logits it already computed. Second, a **correctable execution memory** — a text scratchpad of what has been done — that is explicitly trained to fix itself when it drifts out of sync with what the cameras show. That memory is what lets the robot know it already salted the eggs, a fact no camera image reveals.

The paper is also a system paper: the low-level controller is trained on **40,115 hours** of real robot data across a shared 40-dimensional action space, so one policy drives a wheeled humanoid, a bimanual ARX, and two Franka arms.

## The Methodology

### The hierarchy

At logical step $t$, the high-level policy $\mu$ builds a context
$$h_t=(\ell,\;\mathcal{M}_{t-1},\;z^\star_{t-1},\;o_t)$$
from the task instruction $\ell$, carried memory $\mathcal{M}_{t-1}$, the last subtask, and the current multi-view observation. It returns an updated memory and a subtask, $(\mathcal{M}_t,z^\star_t)=\mu(h_t)$. The low-level policy then produces an $H$-step action chunk
$$\mathbf{a}_{t:t+H-1}=\pi_\theta(o_t,\mathbf{s}_t,z^\star_t,\eta)$$
where $\mathbf{s}_t$ is proprioception and $\eta$ is text metadata naming the embodiment and control mode. If $z^\star_t=z^\star_{t-1}$, execution just continues.

### Four models in the head

All three language models are fine-tuned from the same robot-pretrained Qwen3.5-9B checkpoint; the world model starts from **Step1X-Edit** (an image editing diffusion model).

1. **Proposal model $P$.** One forward pass gives `<think>`, `<memory>`, `<subtask>`. Produces the direct proposal $z^{\mathrm{dir}}_t$ and the refreshed memory.
2. **World model $\mathcal{W}$.** Takes *one* head-camera RGB image plus a candidate subtask string, returns the predicted image at completion: $\hat o=\mathcal{W}(\tilde o,z)$.
3. **Value model $V$.** A five-way multiple-choice VQA head — from *clearly wrong* to *clearly correct*, mapped to scalars in $[0.05,0.95]$. It sees $(\ell, z, \hat o)$ only.
4. **Reflective model $F$.** Sees the real, observation-aligned context $\bar h_t$ plus the surviving branches $\mathcal{C}_t$, and **generates** the final subtask. It is not a selector — its output need not be one of the candidates.

### The search

Beam search over language. Root branch = current real observation, score 0. At each depth $d\le D$, every surviving branch $b$ is expanded by sampling the proposal model $N$ times. For each child,
$$\hat o(b\oplus z)=\mathcal{W}(\tilde o(b),z),\qquad S(b\oplus z)=S(b)+V(\ell,z,\hat o(b\oplus z))$$
and the child's *imagined* image becomes the visual input for its own expansion. Children are ranked **globally** by cumulative score and pruned to top-$B$. Memories created inside search are branch-local and never touch $\mathcal{M}_t$. Routing decisions from proposal calls inside search are ignored. Compute scales through $N$, $B$, $D$.

### The router

Let $p_i$ be the probability of generated token $i$, and $m_i=\lambda^{(1)}_i-\lambda^{(2)}_i$ the top-two logit margin. Then
$$g_t=\mathbf{1}\!\left[\;\underbrace{\tfrac{1}{|\mathcal{I}_{\text{all}}|}\textstyle\sum_i p_i}_{\text{mean token prob}}\le\delta_{\text{all}}\;\lor\;\underbrace{\tfrac{1}{|\mathcal{I}_{\text{mem}}|}\textstyle\sum_{i\in\text{mem}} m_i}_{\text{mean margin in }\texttt{<memory>}}\le\delta_{\text{mem}}\right]$$
$g_t=1$ triggers search. Free — the logits already exist. Thresholds are per-task, calibrated on held-out data. Nice detail: the memory field gets its own confidence check, because uncertainty about *what has already happened* is a different failure than uncertainty about *what to do*.

### Teaching memory to repair itself

No new labels. Take an annotated demonstration, **corrupt only the input memory**, and read the correct target off the demo. Five families:

| Family | Input → target memory | Failure it fixes | Mix |
|---|---|---|---|
| within-subtask | $\mathcal{M}_n\to\mathcal{M}_n$ | none (aligned) | 58% |
| transition | $\mathcal{M}_n\to\mathcal{M}_{n+1}$ | starting the next step | 15% |
| catch-up | $\mathcal{M}_{n-1}\to\mathcal{M}_n$ | memory lags reality | 10% |
| rollback | $\mathcal{M}_{n+1\ldots n+3}\to\mathcal{M}_n$ | memory ran ahead | 12% |
| error-think | annotated failure frame | unnoticed failure | 5% |

Rollback is deliberately capped at 10–15% so the model does not learn to distrust memory that is actually correct.

### The low-level policy

A Qwen3.5-2B vision-language backbone plus a **Mixture-of-Transformers** action expert: action tokens and backbone tokens share joint [[Attention Is All You Need#Multi-head attention|attention]] at every full-attention layer but run through separately parameterised streams.

Everything lives in a canonical $\mathbb{R}^{40}$ layout: left/right EEF position (3+3), 6-D rotation (6+6), two grippers, two waist channels, two planar base velocities, and $2\times 8$ arm joints. End-effector and joint actions are **relative** to the current state. A binary mask $\mathbf{M}$ (diagonal, $d_a\times d_a$) zeroes channels the embodiment does not have.

Training is masked conditional flow matching. With $\tau=1$ noise and $\tau=0$ clean,
$$\mathbf{a}^\tau_{t+j}=\tau\mathbf{M}\boldsymbol\epsilon_{t+j}+(1-\tau)\mathbf{M}\mathbf{a}_{t+j},\qquad \mathbf{u}_{t+j}=\mathbf{M}(\boldsymbol\epsilon_{t+j}-\mathbf{a}_{t+j})$$
$$\mathcal{L}_{\mathrm{FM}}=\mathbb{E}\left[\frac{1}{H\operatorname{tr}(\mathbf{M})}\sum_{j=0}^{H-1}\left\|\mathbf{M}\mathbf{v}^\tau_{\theta,t+j}-\mathbf{u}_{t+j}\right\|_2^2\right]$$
$\tau=0.001+0.999x$ with $x\sim\mathrm{Beta}(1.5,1)$ (so more mass near clean actions). Per-sample loss capped at 100 before averaging. $H=30$, ten uniform Euler steps at inference, mask reapplied before every velocity evaluation. Same family of objective as [[Score-Based Generative Modeling through SDEs|score-based generative models]] and [[Denoising Diffusion Probabilistic Models|diffusion]], but a straight-line probability path.

Three training stages: **(1) knowledge-isolated co-training** — action gradients are blocked at the backbone interface so the expert learns control without wrecking the pretrained VLM; **(2) end-to-end** — remove the block, keep multimodal data as auxiliary loss; **(3) task-specific fine-tune** on a small demo set per deployment task.

### Data

23.4K hours internal (21.9K on AGIBOT G1, 585h G2, 578h ARX AC One, 347h Franka) + 16.7K hours public (9.25K of UMI handheld-gripper data). Interleaved with multimodal VQA, grounding, and depth-reasoning examples throughout.

High-level supervision was pre-labelled by Gemma4-31B-it, then filtered programmatically: 11.74% of episodes discarded, **40.4M clean samples (88.26%) retained**.

### Serving

The high-level policy is far slower than one control tick, so it runs as a **background worker** that refreshes a per-episode subtask cache roughly every 1 s. The control loop reads the cache and returns a chunk immediately at ~30 Hz. A slow or crashed high-level call just delays the next command; the arm keeps executing the cached subtask.

## Ablation Studies and Experiments

### Long-horizon tasks, 10 physical trials each, no search

Progress is a milestone score with a prerequisite DAG: 1.0 first-attempt, 0.5 after a failed retry, 0 skipped, and a milestone only counts if all its ancestors are done.

| Method | Clean Room SR | Prep Ingredients | Stir Fry | Milk Tea | Avg SR | Avg Progress |
|---|---|---|---|---|---|---|
| GR00T N1.7 | 0/10 | 1/10 | 0/10 | 0/10 | 2.5% | 45.29% |
| LingBot-VLA | 0/10 | 0/10 | 0/10 | 0/10 | 0.0% | 44.43% |
| $\pi_{0.5}$ | 4/10 | 2/10 | 0/10 | 3/10 | 22.5% | 73.05% |
| $\tau_0$-VLA (flat) | 4/10 | 2/10 | 0/10 | 5/10 | 27.5% | 80.10% |
| $\tau_0$-VLA hierarchical, Plan Once | **5/10** | **4/10** | **4/10** | **5/10** | **45.0%** | **87.85%** |

The flat-vs-hierarchical row is the clean ablation: same low-level policy, same sensors, only difference is whether the language command is the whole task or a bounded subtask with memory. Average SR 27.5% → 45.0%.

**The most diagnostic single number is Stir Fry: 0/10 → 4/10.** The bottleneck there is salt. Adding salt changes the image almost not at all, so a policy conditioned only on the current frame either salts twice or never. Written memory resolves an ambiguity that vision structurally cannot. Repeated salting is scored as a *prohibited action*, not a retry — it kills SR outright.

Milk Tea is the opposite regime: both $\tau_0$ variants already sit at 5/10, and remaining failures are lid-attachment and straw-insertion, i.e. contact-rich motor problems the hierarchy cannot touch.

### Cross-embodiment, direct execution (no high-level policy at all)

| Method | Collect Laundry | Cotton Pad | Eyelash Curler | Makeup Puff |
|---|---|---|---|---|
| GR00T N1.7 | 4/10 | 10/10 | 8/10 | 7/10 |
| LingBot-VLA | 2/10 | 9/10 | 3/10 | 3/10 |
| $\pi_{0.5}$ | 9/10 | 9/10 | 8/10 | 7/10 |
| $\tau_0$-VLA | **10/10** | **10/10** | **9/10** | **10/10** |

Tidy Makeup Table is built so the *image is the same* and only the instruction changes the target object, arm, order, and destination — a pure language-grounding test. LingBot-VLA gets the arm right but grabs whatever object is visually nearby; that is a grounding failure, not a control failure.

### Test-time compute: open-loop next-subtask accuracy

Judged by GPT-5.4 with a three-label rubric (`equivalent` / `adjacent` / `wrong`), two temperature-0 calls per tuple, third on disagreement, majority vote. Only `equivalent` counts. `adjacent` (a valid but not-the-immediate step) is kept as a diagnostic — a good idea worth stealing, since exact string match badly underrates paraphrases.

On **Book Organization (OOD)** — four books, four slots, sort tallest-to-shortest by pairwise swaps, initial arrangement never seen in training:

- Plan Once: **50.0%**
- Best-of-$N$ (same world model, same value model, one-step select): **57.5%**
- TTC: **74.0%**

Best-of-$N$ is the ablation that matters. It has *all* the same components; it lacks only multi-step expansion and reflective commitment. It recovers 7.5 points of the 24. So roughly **two-thirds of the gain comes from depth plus reflection, not from having a world model at all.**

### Closed-loop, with routing on

| Method | Milk Tea SR / Prog | Book Org SR / Prog | Clean Room SR / Prog |
|---|---|---|---|
| Plan Once | 5/10, 91.92% | 6/10, 66.67% | 5/10, 94.80% |
| TTC | **7/10, 95.38%** | **9/10, 93.33%** | **7/10, 97.60%** |

Book Organization gains most (6→9). It is the task with no fixed script — the right swap depends entirely on the current arrangement, so there is nothing for a single-pass policy to memorise.

### Compute vs accuracy

Accuracy against inference cost is fit with a **saturating exponential** (least squares over all runs). Sharp rise from a small budget, then a plateau. The honest reading: TTC buys a lot at moderate budgets and then stops paying. There is no log-linear scaling story here.

### What did not work / was tuned away

- **Naive cross-task contamination filter** (annotator hallucinating milk-tea constraints under a fruit-shelving task): a plain disjointness test discarded only 0.33% and left contamination behind; a strict noun-overlap ratio discarded 2.32% and threw away valid verbose constraints. The shipped two-rule filter with an expanded stop-word list discards 0.42% and leaves **zero** residual contamination on a 3.26M-sample validation set.
- **Rollback training instances capped at 10–15%.** Too many and the policy learns to distrust memory that is correct.
- `restorable=false` failure samples are dropped entirely — you cannot teach recovery from an unrecoverable state.

## Worth Remembering

- **The value model never sees the real scene.** It scores $(\ell, z, \hat o)$ where $\hat o$ is a *generated* image. Every error in the world model propagates straight into the ranking. And the world model consumes exactly one head-camera view — wrist cameras, depth, and proprioception are all invisible to the search.
- **The reflective model is a generator, not an argmax.** Search output is used as *context* for writing a fresh subtask. This is the difference between $\tau_0$-VLA and Best-of-$N$, and the ablation says it is where most of the value is. Compare [[Chain-of-Thought Prompting Elicits Reasoning in LLMs|chain-of-thought]]: the intermediate steps are scaffolding for the final answer, not the answer.
- **Compared to [[Mastering the game of Go with deep neural networks (AlphaGo)|AlphaGo]]-style [[Mastering the game of Go with deep neural networks (AlphaGo)#^mcts|MCTS]]**: no simulator, no rollouts to terminal, no backup. It is one-shot beam search with a learned value at every node and a learned generative transition. Closer in spirit to [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)|Dreamer]]'s latent imagination, except the "latent" is a literal RGB image plus a sentence, which makes it inspectable but far more expensive per step.
- **Nothing is trained with RL.** The value model is supervised classification into five ordinal buckets, graded against the ground-truth next step from offline rollouts of $P$ and $\mathcal{W}$. So "value" here means *does this subtask look like the demonstrator's next move*, not expected return.
- **Per-task threshold calibration is a real deployment cost.** The routing rule is shared, but $\delta_{\text{all}},\delta_{\text{mem}}$ are fitted on held-out data for each task. A new task needs new calibration data before the router is trustworthy.
- **No compute budget is reported in wall-clock for the closed-loop runs.** We know the cache refreshes about every 1 s and control runs at ~30 Hz, but the paper does not say what $N,B,D$ were used in Table III or how often the router fired.
- **The baselines are not equally tuned.** GR00T N1.7 and LingBot-VLA score 0/10 on every long-horizon task; that spread suggests they were not adapted to these embodiments with the same care as stage-3 fine-tuning gave $\tau_0$. Read Table I as "our system vs $\pi_{0.5}$" and treat the rest as context. See [[On the Difficulty of Evaluating Baselines]].
- **The knowledge-isolation stage is borrowed** from Driess et al. and is a nice general recipe: when bolting a new head onto a pretrained backbone, block the new gradients at the interface until the head is competent, *then* unfreeze. Related in spirit to how [[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]] protects pretrained weights, but by gradient masking rather than parameterisation.
- **Ten trials per cell.** 5/10 vs 7/10 is roughly one standard error apart. The direction of every result is consistent, but no individual number is tight.
- Open question: the world model predicts the *terminal* frame of a subtask, so imagined images at depth $\ge 2$ are edits of edits. How fast does that compound? The saturation curve may be the world model degrading rather than search genuinely running out of headroom.

## Links

Related: [[Attention Is All You Need]] · [[Denoising Diffusion Probabilistic Models]] · [[Score-Based Generative Modeling through SDEs]] · [[Mastering the game of Go with deep neural networks (AlphaGo)]] · [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Foundation Models]] · [[Uncertainty]] · [[Markov Decision Process]] · [[On the Difficulty of Evaluating Baselines]] · [[The Bitter Lesson (essay)]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Language Models are Few-Shot Learners (GPT-3)]]

New topics worth writing: Flow matching and rectified flow, Vision-Language-Action models, Beam search, Mixture-of-Transformers, Test-time compute scaling, LLM-as-a-judge evaluation, Action chunking, 6D rotation representation, Knowledge insulation in multimodal fine-tuning, Process reward models
