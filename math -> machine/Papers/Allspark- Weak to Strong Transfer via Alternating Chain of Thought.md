---
title: "Allspark: Weak to Strong Transfer via Alternating Chain of Thought"
authors: ["Kaizhao Liang", "Junxiong Wang", "Chen Liang", "Zhendong Wang", "Qiang Liu"]
year: 2026
arxiv: "2609.32913"
url: https://arxiv.org/abs/2609.32913
priority: Good-To-Read
read_on: 2026-09-30
tags: [paper, transformers, rl]
---
## The Core Idea

Training a frontier model with reinforcement learning is expensive mostly because of **rollouts** — you have to make the big model generate thousands of long reasoning traces, over and over, to get a gradient signal. The bill scales with the size of the model you want to improve.

Allspark asks: what if the model you want to improve never generates a single training rollout?

The trick is to split the reasoning trace between two writers. Take a small model. Make two copies. Freeze one copy (call it the **student**), train the other with RL (call it the **teacher**). They take turns writing chunks of one shared chain of thought. The frozen student writes the final answer. The reward — did the final answer come out right — only updates the teacher's tokens.

So the teacher is not being trained to solve problems. It is being trained to write text that makes *another model* solve problems.

Then at inference you swap the frozen small student for a big frozen model. The teacher is unchanged. Both models are frozen. They alternate chunks, the big model writes the answer.

> [!NOTE] Weak-to-strong transfer via text
> The interface between teacher and student is plain text, not logits. The teacher's reasoning is decoded to a string, stripped of control markers, and re-encoded with the student's own tokenizer. So teacher and student need not share a vocabulary, a tokenizer, an architecture, or a company. ^text-interface

That last point is what makes this different from everything nearby. [[Distilling the Knowledge in a Neural Network|Knowledge distillation]] and on-policy distillation match token distributions, which needs aligned tokenizers and, crucially, needs the student's parameters. The weak-to-strong distillation line (Direct-OPD, W2S-OPD, OPRD — all 2026) still samples from the strong student and still updates it. Allspark does neither. Trained a teacher once on a 276B model, reused it to steer a 975B Inkling, a 1T Kimi and a 550B Nemotron, with no retraining for any of them.

The conceptual point worth keeping: **a useful reasoning chunk does not have to be a correct or complete solution.** The paper writes this as a continuation value,

$$Q(h, z) = \mathbb{E}[R(x,y) \mid h, z]$$

where $h$ is the shared prefix so far, $z$ is the chunk the teacher just wrote, and $R \in \{0,1\}$ is whether the final answer was right. The value of $z$ is entirely about what the student does next with it. A weaker model can write a chunk with high $Q$ — reframe the question, name the right substitution, rule out a wrong option — without being able to finish the job. This is why "weak teacher, strong student" is not a contradiction.

## The Methodology

**The alternating loop.** One shared reasoning prefix $h$, starting as the problem $x$. The student goes first. Then turns alternate:

$$z \sim \begin{cases}\rho(\cdot \mid h), & \text{student turn} \\ \pi(\cdot \mid h), & \text{teacher turn}\end{cases}$$

Each chunk is appended to $h$. Reasoning ends when someone emits `</think>` naturally, or when a shared token ceiling is hit (then the evaluator inserts `</think>` itself). The student then writes the final answer $y \sim \rho(\cdot \mid h)$.

Chunk sizes: first student chunk is short (128 tokens in Qwen, 256 in Inkling), later chunks are 256 (Qwen) or 2048 (Inkling). The reasoning budget is **shared across all chunks from both writers** — not handed out fresh per turn.

**The objective.** Maximise expected final correctness over teacher parameters only:

$$\max_\pi J(\pi), \qquad J(\pi) = \mathbb{E}_{\pi,\rho}[R(x,y)], \qquad \rho \text{ fixed}$$

**The loss mask is the whole design.** Only sampled teacher reasoning tokens get gradient. Student reasoning: masked. The final answer: masked (the student wrote it). Inserted `</think>` and separators: masked. Student tokens still sit in the [[Causal Attention|causal]] context of the teacher's later predictions — the teacher conditions on them, it just isn't trained to produce them.

### Qwen setup (the controlled study)

Both arms start from Qwen3-1.7B, seed 0, [[LoRA]] only ($r=16$, $\alpha=32$, dropout 0, on $q,k,v,o$ + gate/up/down). Base weights frozen. Tiny data: 142 MATH prompts and 162 LogiQA 2.0 prompts in the RL pool after screening, 128 dev questions per domain.

**Prompt screening.** A prompt is kept only if the initial policy gets 1–7 out of 8 samples right. Too easy or too hard gives no advantage signal. Online, each actor re-screens with 8 fresh samples (same 1–7 rule), then draws $G=32$ trajectories; groups with 1–31 successes are kept.

**The loss.** Group-standardised advantage, exactly as in [[GRPO]]:

$$A_i = \frac{r_i - \mu}{\sigma} \text{ if } \sigma > 0, \text{ else } 0$$

with $\mu, \sigma$ the mean and population std of the 32 binary rewards. No epsilon on $\sigma$. The same $A_i$ is used at every trainable position in that trajectory.

Then per-token, a head-only **score centering** loss (after Marek & Ryabinin, 2026) rather than a [[PPO|clipped ratio]]:

$$\ell(\theta) = -A\left[\mathbf{1}\{y \in H\}\log p_\theta(y \mid h) - \sum_{v \in H} q(v \mid h)\log p_\theta(v \mid h)\right]$$

Read it plainly. $H$ is the top $K=128$ tokens from the checkpoint that actually sampled this trajectory, and $q$ is that old checkpoint's probabilities, frozen. The first term pushes up the token that got sampled. The second term is a baseline: the old sampler's own expected log-probability under the *current* learner. Positive advantage means "make the sampled token more likely than the sampler's average"; negative advantage flips it. If the sampled token wasn't in the top 128, the first term vanishes and only the centering term acts. The $q$ values are **not renormalised** to sum to 1 — they keep their original mass, so the centering term is implicitly weaker when the sampler was uncertain.

Batch loss: sum over tokens, average over trajectories only.

$$\mathcal{L}(\theta) = \frac{1}{MG}\sum_{\tau \in \mathcal{B}} \sum_{t \in \mathcal{T}(\tau)} \ell_{\tau,t}(\theta)$$

$M = 6$ prompt groups (2 per domain) × $G = 32$ = 192 trajectories per update. **No per-sequence length normalisation** — a long trajectory contributes more terms than a short one.

**Asynchrony.** Three rollout actors, one learner. Each group's version lag must satisfy $0 \le v_\text{learner} - v_\text{sampler} \le 2$; older queued groups are thrown away. No importance-sampling ratio, no clipping, no reference-[[KL Divergence|KL]] term, no entropy bonus. The authors are explicit that the sampler-head centering term is *not* a staleness correction.

AdamW, constant LR $10^{-5}$, grad-norm clip 1, weight decay 0, temperature 1 / top-p 1 / no top-k. Training ceilings 4096 reasoning + 1536 answer; evaluation ceilings doubled to 8192 + 4096.

### Inkling setup (the scale study)

Teacher is Inkling-Small (276B total, 12B active — a [[Mixture of Experts]]), paired with a frozen copy of itself. Rank-32 LoRA, 72 rollout steps. 16 tasks per step × 3 thinking-effort levels $\{0.4, 0.55, 0.7\}$ × 32 rollouts = nominally 1536 rollouts per step. LR $2\times 10^{-5}$ to step 24, then $5\times 10^{-5}$.

Different loss here — Tinker's built-in importance-sampling [[Policy Gradient|policy gradient]]:

$$\ell(\pi) = -\frac{\bar{n}}{n}A\sum_{t \in T}\frac{\pi(y_t \mid h_t)}{\pi_\text{old}(y_t \mid h_t)}, \qquad A = r - \bar{r}$$

$T$ is teacher reasoning positions only, $n = |T|$, $\bar{n}$ the batch mean teacher-token count — so this *does* length-normalise, unlike the Qwen arm. Baseline $\bar{r}$ is the mean within the task–effort group; **no division by group std**, unlike the Qwen arm.

**The reward is shaped, not binary.** On ARC-AGI-2:

$$r = ws + (1-w)c - \min\left\{p_\max,\ \lambda\max\left(0, \frac{L}{L_0} - 1\right)\right\}$$

$s$ = exact task success, $c$ = per-cell grid accuracy, $L$ = retained reasoning tokens from *both* writers. $w = 0.85$, $L_0 = 5000$, $p_\max = 0.30$. Length penalty is off ($\lambda = 0$) for steps 0–35, then $\lambda = 0.15$ for steps 36–71 on the same adapter. So the first half learns to be right, the second half learns to be right *and* brief. Hard reasoning ceiling 24,576 tokens.

Data: the 1000 public ARC-AGI-2 training tasks, split by deterministic task-ID hash into 904 train / 96 dev.

**Cross-family handoff plumbing.** At each turn boundary: decode the chunk with the writer's tokenizer, strip control markers, append to a shared plain-text trace, re-encode the *whole* trace with the receiver's tokenizer, drop it into the receiver's native hidden-reasoning channel. Chunks are trimmed to the last paragraph boundary if at least half the sampled chunk survives. Token IDs never cross the boundary. The length guard is recomputed in the receiving tokenizer each time.

## Ablation Studies and Experiments

### Qwen3, 128 dev questions per domain, one sample each

| Student | Teacher | Math | Reasoning |
|---|---|---|---|
| Qwen3-1.7B | none | 81.3 | 73.4 |
| Qwen3-1.7B | ordinary RL | 88.2 | 71.8 |
| Qwen3-1.7B | **Allspark** | **89.8** | **75.0** |
| Qwen3-1.7B (RL-tuned, solo) | none | 89.8 | 70.3 |
| Qwen3-4B | none | **96.9** | 81.2 |
| Qwen3-4B | untrained 1.7B | 93.0 | 79.7 |
| Qwen3-4B | ordinary-RL 1.7B | 89.8 | 79.6 |
| Qwen3-4B | **Allspark 1.7B** | 96.1 | **82.8** |

**This is the most informative table in the paper, and mostly because of the negatives.** Bolt an *untrained* 1.7B onto a 4B student and you lose 3.9 points of math and 1.5 of reasoning. Bolt an *ordinary-RL-trained* 1.7B on and you lose **7.1** points of math — worse than the untrained one. A model trained to solve problems for itself is actively bad at helping a bigger model. That is the whole thesis, stated as a failure: the ability to be a useful collaborator is a separate skill from the ability to be right, and you only get it if you train for it under the collaborative rollout.

The Allspark teacher gives roughly neutral math (96.1 vs 96.9 — inside sampling noise at $n=128$, one sample per question) and +1.6 reasoning. Honest reading: on the Qwen scale this is a sign of life, not a win. The authors say as much.

Note also that the Allspark teacher paired with its *own* frozen twin beats the solo RL-tuned model on reasoning (75.0 vs 70.3) while matching it on math (89.8 vs 89.8). Ordinary RL bought math and *lost* reasoning relative to the base model (71.8 / 70.3 vs 73.4) — a small [[Fine-Tuning#The failure modes 🪤|forgetting]] signature that the Allspark arm doesn't show.

### Inkling, 96 ARC-AGI-2 dev tasks, 3 attempts each (288 attempts), mean pass@1

Points swept over thinking efforts 0.40 / 0.55 / 0.70 / 0.85.

- Inkling alone: best observed **78.1%** at 12.3k retained tokens.
- Allspark (Inkling-Small teacher + Inkling student): peak **81.6%**; and 79.2% at only **10.8k** tokens.

So at a chosen operating point it beats the strong model's *best* accuracy with 12.1% fewer retained output tokens. That is a real shift in the accuracy–token curve, not just a higher ceiling.

### Earlier within-family control (checkpoint 35, effort 0.7, one attempt each)

| Student | Teacher | Accuracy |
|---|---|---|
| Inkling-Small | none | 60.4 |
| Inkling-Small | untrained | 60.4 |
| Inkling-Small | **Allspark** | **70.8** |
| Inkling | none | 75.0 |
| Inkling | untrained | **66.7** |
| Inkling | **Allspark** | **76.0** |

The untrained-teacher row is the clean control and it is damning in the right direction: an untrained partner does *nothing* for the weak student (60.4 → 60.4) and costs the strong student **8.3 points** (75.0 → 66.7). Interleaving a second voice into a chain of thought is harmful by default. The training is what makes it helpful.

### Cross-family transfer (same frozen Inkling-Small teacher)

| Student | Δ accuracy | Δ retained tokens |
|---|---|---|
| Kimi-K2.6 (128K) | **+11.5 pp** | −7.7% |
| Nemotron-3-Ultra, medium | **+17.7 pp** | increased |
| Nemotron-3-Ultra, full | +5.2 pp | increased |

95% intervals from 5000 paired task-level bootstrap resamples, seed 17. Task–attempt seeds paired against the student-alone baseline.

The pattern: the **bigger the gain, the weaker the student's native thinking setting.** Nemotron-medium gains 17.7 points, Nemotron-full only 5.2. Read charitably, the teacher supplies structure the student would otherwise have to discover with more of its own reasoning. Read sceptically, some of the gain may be the alternating harness rather than the teacher's content — see the caveat below.

### What did not work

- **Ordinary RL teachers make strong students worse.** −7.1 math on Qwen3-4B. The single most useful negative result here.
- **Untrained teachers make strong students worse.** −8.3 on Inkling, −3.9 math on Qwen3-4B.
- **Math on Qwen3-4B never actually improved.** 96.1 vs 96.9 solo. Only reasoning moved.
- **Token savings do not become dollar savings.** Modelled cost per attempt with Allspark is 1.42–1.82× the student alone at matched effort. The best Inkling operating point uses 12.1% fewer tokens but only 2.7% less money ($0.0606 vs $0.0623). Kimi's 7.7% token reduction comes with a **21.1% cost increase** ($0.1466 → $0.1774). Prefill and cached rereads eat the saving.
- **Coding case C4** is included as a genuine regression: correct → incorrect, from a shared indexing error that both models propagated.
- The R2 case study notes the teacher **overstated the passage** and the student still landed on the right answer. Helpful ≠ faithful — a relative of [[Chain of Thought#The uncomfortable part|unfaithful CoT]].

## Worth Remembering

**The baseline comparison is not perfectly clean, and the paper says so.** The alternating evaluator explicitly closes the reasoning channel and *requests a separate final answer*, including after hitting the reasoning ceiling. The student-alone baseline must transition from thinking to answering by itself within one completion. So the comparison is between two whole inference procedures, one of which has a forced "now answer" nudge. Some fraction of the cross-family gain could be that nudge rescuing runs that would otherwise ramble past their budget. The Qwen P1 case is exactly this shape: the solo run hit the answer-token limit without committing to a number; the alternating run said "1".

**Two different RL losses, two different papers inside one paper.** The Qwen arm uses group-standardised advantages + head-only score centering + no length normalisation + binary reward. The Inkling arm uses mean-centred (not std-divided) advantages + an importance-sampling ratio + length normalisation + a shaped dense reward with a length penalty. You cannot read the Qwen ablations as evidence about the Inkling result or vice versa. Treat them as two separate experiments that agree in direction.

**The evaluation is thin.** 96 dev tasks, one seed per training run, dev questions were also used for [[Fine-Tuning|checkpoint]] selection (checkpoint 94 chosen by argmax over the same panel later reported as results — Eq. 7). No held-out test. Qwen: 304 total training prompts. This is *preliminary evidence under limited compute*, in the authors' own words, and the right epistemic tier is **open research question**, not established result.

**Serving cost is the real limitation.** You now run two models per query, with handoffs. Latency goes up by more than the token count suggests, because each handoff is a round trip and each receiver re-prefills newly arrived text. The archive's cost model assumes a separate prefix cache per writer — see [[Prompt Caching]] for why that assumption matters so much: the shared trace grows, and the fraction of it that is cacheable for a given writer depends on who spoke last.

**It needs an interruptible reasoning stream.** Non-thinking models, or APIs that don't expose a hidden reasoning channel you can open, close and refill, are untested. Some providers do not let you inject arbitrary text into the thinking channel at all.

**Connections.** The "teacher steers a frozen student at inference" family is older than this: weak-to-strong search (Zhou 2024) steers with likelihood differences, Co-LLM learns when to defer, CoWeST has the strong model refine weak drafts. Allspark's distinction is that the steering is *learned by RL against the collaborative outcome*, and the medium is text rather than logits.

**Why this is interesting from a measurement angle.** The teacher's objective is the continuation value $Q(h,z)$ — the value of an action is defined purely by what a *different, fixed* policy does afterwards. That is a two-agent [[Credit Assignment|credit assignment]] problem where one agent is frozen, and at transfer time the frozen agent is swapped for a different one. Nobody here checks whether $Q$ estimated against the weak student transfers to $Q$ against the strong student — that is precisely an [[Off-Policy Evaluation|off-policy]] question, and it is the gap that would explain why math didn't move and why gains shrink as the student's native effort rises.

**Follow-up questions worth chasing.**
1. Add a harness-only control: alternate the strong student *with itself* under the same chunking, trimming and forced-answer rules. That isolates the teacher's content from the scaffold.
2. Does teacher quality degrade as the student gets stronger — i.e. is there a weak-teacher ceiling, and where is it?
3. The length-penalty phase change (steps 0–35 at $\lambda=0$, 36–71 at $\lambda=0.15$) is never ablated. How much of the accuracy–token improvement is that curriculum, and how much is the alternating structure?
4. Could you train one teacher against *several* frozen partners to make $Q$ more transferable?

## Links

Related: [[GRPO]] · [[Policy Gradient]] · [[Distilling the Knowledge in a Neural Network]] · [[Chain of Thought]] · [[LoRA]] · [[Credit Assignment]] · [[Off-Policy Evaluation]] · [[On-Policy vs Off-Policy]] · [[PPO]] · [[Test-Time Compute]] · [[Prompt Caching]] · [[Mixture of Experts]] · [[Reward Function]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[Learning from Teacher Continuations at Student States]] · [[What Does Privileged Information Add to On-Policy Self-Distillation]] · [[Multi-Agent LLM Systems]] · [[The Handoff Tax- Continuing Non-Native Trajectories in LLM Agents]] · [[Sampling Parameters]]

New topics worth writing: weak-to-strong generalization, score centering for off-policy RL, ARC-AGI-2, cross-tokenizer distillation, thinking-effort control, continuation value in multi-writer rollouts, prompt screening for RL curricula, accuracy–token tradeoff curves, asynchronous RL with bounded version lag
