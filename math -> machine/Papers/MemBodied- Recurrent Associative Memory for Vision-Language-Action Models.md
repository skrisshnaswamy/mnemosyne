---
title: "MemBodied: Recurrent Associative Memory for Vision-Language-Action Models"
authors: ["Pala et al."]
year: 2026
arxiv: "2609.28256"
url: https://arxiv.org/abs/2609.28256
priority: Good-To-Read
read_on: 2026-09-28
tags: [paper, transformers, llm, vision]
---
## The Core Idea

A robot policy that only sees the current camera frame cannot solve tasks where the right action depends on something that already happened. Move a block from the left pad to the middle, press a button, now put it back — where does it go? The picture in front of the robot no longer says. Two different situations look identical but need different actions. The paper calls this **temporal state aliasing**.

> [!NOTE] Temporal state aliasing
> Two moments in an episode produce the same (or near-identical) observation, but the correct action differs because of what happened earlier. It is exactly a violation of the [[Markov Property]] at the level of the observation — the task is a [[POMDP]] dressed up as an [[Markov Decision Process|MDP]]. ^state-aliasing

The obvious fix — stuff past frames into the model's context — costs you. Keep everything and the [[Context Window]], memory and latency grow with episode length. Keep a fixed window and you throw away the one frame that mattered.

**MemBodied** keeps a fixed-size memory instead. Two parts, doing two different jobs:

1. **An associative state.** A small matrix per layer, size $128 \times 128$, that gets *overwritten* each policy call using a gated delta rule. It stores *what I did and what happened as a result*.
2. **An episode anchor.** A compressed snapshot of the very first frame of the episode, frozen for the whole episode. It stores *what the scene looked like before I touched anything*.

The second exists because of a flaw in the first. Repeated delta-rule writes erase fine detail — early information decays. So the initial scene gets a separate, write-once pathway that nothing can overwrite.

Both are read at every policy call, and their readouts are fed into the action network. Cost does not grow with episode length. The whole memory is 40M parameters, 1.26% of $\pi_0$.

The result: on five memory-dependent RMBench tasks, mean success 50.0% versus 6.4% for the stateless $\pi_0$ policy ($7.81\times$), and inference latency 129.2 ms versus 1593.7 ms for the compressed-history baseline NativeMEM — a 91.9% cut.

## The Methodology

### The setup

The backbone is $\pi_0$: a PaliGemma vision-language model feeding a [[Flow Matching|flow-matching]] action expert that emits a chunk of $H = 50$ actions at once.

At step $t$ the policy sees images $I_t$, robot state $s_t$, instruction $\ell$, and the memory $\mathcal{E}_{t-1} = (\mathcal{M}_{t-1}, A)$:

$$\mathbf{a}_t = [a_{t,1},\dots,a_{t,H}] \sim p_\theta(\cdot \mid I_t, s_t, \ell, \mathcal{E}_{t-1}).$$

### The associative state

One matrix per action-network layer:

$$\mathcal{M}_t = \{S_t^{(l)}\}_{l=1}^{L}, \qquad S_t^{(l)} \in \mathbb{R}^{B \times r \times r}, \quad r = 128.$$

This is a **fast-weight memory** — the lineage runs Schmidhuber 1992 → linear-attention-as-fast-weights → DeltaNet → Gated DeltaNet.

> [!NOTE] Fast weights
> A second set of weights that changes *during* an episode, written by the network itself, separate from the slow weights that [[Backpropagation|backprop]] learns during training. Here the "weights" are a key→value lookup matrix. Reading is one matrix–vector product. ^fast-weights

**What gets written.** Not an observation. An *interaction*: the action chunk plus the visual consequence of that action.

For each camera $c$, the next robot state $s_{t+1}$ queries the next frame's patch tokens by [[Cross Attention|cross-attention]]:

$$e_{t+1}^{(c)} = \operatorname{softmax}\!\left(\frac{(W_{pq}\phi_s(s_{t+1}))(W_{pk}X_{t+1}^{(c)})^\top}{\sqrt{D}}\right) W_{pv}X_{t+1}^{(c)}$$

Gradients are **stopped** at the visual patch tokens, so the memory path cannot corrupt the vision encoder. The camera encodings are averaged to $e_{t+1}^{\text{vision}}$, then concatenated with the summed action chunk:

$$y_t = \left[e_{t+1}^{\text{vision}}\; ;\; \sum_{h=1}^{H} a_{t,h}\right], \qquad v_t^{(l)} = W_v^{(l)} y_t.$$

**How it gets written.** Key from the layer output, squashed and normalised:

$$k_t^{(l)} = \mathrm{L_2Norm}\left(\tanh(W_k^{(l)} h_{t,\text{out}}^{(l)})\right)$$

Two sigmoid gates from the layer output and the write value — $\beta_t^{(l)}$ (write strength) and $\alpha_t^{(l)}$ (retention). Then the gated delta rule:

$$S_t^{(l)} = \operatorname{Diag}(\alpha_t^{(l)}) S_{t-1}^{(l)} + \operatorname{Diag}(\beta_t^{(l)})\left(v_t^{(l)} - \operatorname{Diag}(\alpha_t^{(l)}) S_{t-1}^{(l)} k_t^{(l)}\right)(k_t^{(l)})^\top$$

Read the bracket in plain words: *what I want to store*, minus *what this key already retrieves*. That difference is the correction. So a repeated key **revises** its stored value rather than piling another copy on top. Plain additive writes would just accumulate and saturate.

**How it gets read.** Query from the representation entering the layer:

$$q_t^{(l)} = \mathrm{L_2Norm}\left(\tanh(W_q^{(l)} h_{t,\text{in}}^{(l)})\right), \qquad r_t^{(l)} = S_{t-1}^{(l)} q_t^{(l)}.$$

**Where the readout goes.** The action-expert suffix is arranged as

$$[\text{state token},\ \text{memory token},\ H \text{ noisy action tokens}].$$

The memory token is a learned vector, re-created fresh every call — it carries nothing. Only the matrices persist. At each layer the readout is scaled and gated into it:

$$x_{\text{mem},t}^{(l)} \leftarrow x_{\text{mem},t}^{(l)} + g_t^{(l)} m_t^{(l)}, \quad m_t^{(l)} = \frac{\alpha_{\text{mem}}}{r} W_m^{(l)} r_t^{(l)}, \quad g_t^{(l)} = \sigma(W_g^{(l)} h_{t,\text{in}}^{(l)})$$

With $\alpha_{\text{mem}} = 256$ and $r = 128$, the multiplier is 2. This is a [[LoRA]]-style $\alpha/r$ scaling. The modified token then joins [[Multi-Head Attention|self-attention]], so action tokens can read it as ordinary content.

### The episode anchor

Take the first frame's frozen vision tokens, average-pool each camera's $16 \times 16$ patch grid to $4 \times 4$, concatenate across cameras → $A \in \mathbb{R}^{N_A \times D}$. A 16× spatial compression, one write, no updates.

Reading it: pool the current frame the same way as in the write-value construction to get $z_t$, then cross-attend at rank $r_A = 64$:

$$c_t = z_t + W_o^A\left[\operatorname{softmax}\!\left(\frac{(W_q^A z_t)(W_k^A A)^\top}{\sqrt{r_A}}\right) W_v^A A\right]$$

$b_t = W_A c_t$ is broadcast across the horizon and concatenated into each action representation. Crucially, the anchor is **not** appended to the prefix — prefix length, token positions and attention mask are untouched.

### The causal read/write schedule

$$\mathcal{M}_{t-1} \xrightarrow{\text{read with }(I_t, s_t, \ell)} \mathbf{a}_t \xrightarrow{\text{execute}} I_{t+1} \xrightarrow{\text{write}} \mathcal{M}_t$$

The write is **delayed**: it happens only once the consequence frame arrives. That way the stored value contains the outcome, but the action that caused it never got to peek at it. At inference the policy caches the layer states and the final-denoising-step action chunk, and writes when the next observation lands.

### Training

The delayed write forces sequence-level training. Each sample is $N$ observation–action pairs, spaced 50 environment steps apart (one action horizon). Image encoding is parallel over batch and sequence; the memory state is propagated sequentially. Gradients flow through the whole sequence, so a late action loss trains early memory writes.

**There is no separate memory loss.** What to write, keep and retrieve is learned entirely from the action objective.

Key settings:
- Sequence length 8 (three $M(1)$ tasks), 14 (Battery Try), 18 (Block Ranking) — chosen from mean episode length.
- RMBench: [[LoRA]] rank 16 / $\alpha$ 16 on PaliGemma, rank 32 / $\alpha$ 32 on the action expert; 10,000 steps, batch 8.
- LIBERO: full fine-tuning to match the published $\pi_0$ baseline; 30,000 steps, batch 32, sequence length 6.
- [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], $\beta = (0.9, 0.95)$, peak LR $2.5\times10^{-5}$, cosine decay, 1,000 warmup steps, EMA 0.99, grad clip 1, bf16.
- 3 cameras at $224\times224$, 10 flow-matching denoising steps.

One nice engineering detail on LIBERO: evaluation replans every 5 steps, but memory was trained on a 50-step cadence. They keep **ten memory slots in round-robin** — each call reads one slot, which is revisited 50 environment steps later.

## Ablation Studies and Experiments

### Main table — RMBench, 5 tasks, 50 rollouts each

| Model | PutBack | Rearrange | Swap | Battery | Ranking | Mean |
|---|---|---|---|---|---|---|
| Diffusion Policy | 0.0 | 0.0 | 11.0 | 10.0 | 10.0 | 6.2 |
| ACT | 0.0 | 29.0 | 2.0 | 19.0 | 0.0 | 10.0 |
| X-VLA | 18.0 | 13.0 | 16.0 | 26.0 | 1.0 | 14.8 |
| $\pi_{0.5}$ (published) | 11.0 | 13.0 | 16.0 | 16.0 | 6.0 | 12.4 |
| $\pi_0$-Stateless | 6.0 | 2.0 | 8.0 | 16.0 | 0.0 | 6.4 |
| $\pi_0$-Hint (privileged) | 10.0 | 0.0 | 0.0 | 0.0 | 0.0 | **2.0** |
| $\pi_0$-$\mu$VLA | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| $\pi_0$-FrameStack | 14.0 | 30.0 | 0.0 | 4.0 | 26.0 | 14.8 |
| $\pi_0$-Vanilla Recurrent | 18.0 | 22.0 | 14.0 | 22.0 | 8.0 | 16.8 |
| MemBodied-AS | 32.0 | 26.0 | 12.0 | 20.0 | 14.0 | 20.8 |
| MemBodied-H | 16.0 | 50.0 | 14.0 | 28.0 | 8.0 | 23.2 |
| MemBodied w/o anchor | 34.0 | 76.0 | 16.0 | 30.0 | 32.0 | 37.6 |
| **MemBodied** | 40.0 | 92.0 | 56.0 | 40.0 | 22.0 | **50.0** |

### What failed, and this is the interesting half

**$\pi_0$-$\mu$VLA got 0.0% on all five tasks.** This is the recurrent-memory-token approach from $\mu$VLA, adapted to $\pi_0$ with 64 carried tokens, sequence length 8, gradients truncated every two recurrent steps. Complete collapse. The authors are honest: this says their adaptation failed under a 10,000-step budget, not that $\mu$VLA is bad in its own setting. Their guess — learning recurrence through carried *tokens* needs far longer training than learning it through an explicit matrix. Plausible and worth noting: carried tokens must re-encode everything through repeated transformations, whereas a matrix has a direct read/write interface.

**$\pi_0$-Hint scored 2.0% — worse than doing nothing (6.4%).** They fed the stateless policy simulator-derived text describing task progress, at inference only, with no training on it. Giving a model privileged information it was never trained to consume actively hurts. Off-distribution prompt text degraded the policy. A clean warning about "just tell the model what it needs to know" thinking.

**FrameStack (4 frames, 50 steps apart) got 14.8%** — better than stateless but scored **0.0% on Swap Blocks and 4.0% on Battery Try**, both below the stateless policy on Battery. In-context history is not automatically useful; it can crowd the context and hurt.

**Adding the first frame directly to the prefix** got 22.0% on Put Back Block versus 34.0% for the anchor-free model and 40.0% with the anchor. Adding the raw frame is *worse than no initial-scene information at all*. The compressed, cross-attended anchor is doing real work; the raw tokens are distraction. Compare [[Lost in the Middle]].

### What the ablations actually reveal

**The readout interface matters more than the recurrence.** All three variants share the same associative state. Only how the readout reaches the action network differs:

| Variant | Mean |
|---|---|
| Attention steering (additive corrections to attention query/output) | 20.8 |
| Hierarchical (extra LSTM-like slow matrix with forget gate) | 23.2 |
| Memory token (readout added to a dedicated suffix token) | 37.6 |

Feeding the readout as **content** beats using it to *perturb* attention by 16.8 points. Note that MemBodied-AS here is basically the $\delta$-mem interface, and MemBodied-Vanilla-Recurrent (16.8) uses that same steering mechanism — consistent story.

**The anchor is worth 12.4 points but is not free.** 37.6 → 50.0 overall. Biggest single win is Swap Blocks, 16.0 → 56.0. But **Block Ranking drops from 32.0 to 22.0**. That task is about tracking which orderings you already tried, not about the initial scene, so a fixed first-frame reference is dead weight competing for capacity.

**Both memory channels contribute.** Write-value composition, 3 tasks:

| Write value | Mean |
|---|---|
| Action summary only | 34.7 |
| Vision (next frame) only | 42.7 |
| Both | 46.7 |

On Battery Try the combination reaches 30.0 versus 24.0 (vision) and 22.0 (action) — removing either channel costs 6–8 points. Vision dominates on the pick-and-place tasks.

**Memory rank scales monotonically** (anchor-free, 2 tasks): $r=32 \to 27.0$, $r=64 \to 42.0$, $r=128 \to 55.0$. No sign of saturation at 128, so they likely left performance on the table.

**Carrying memory across episodes hurts.** An exploratory test (anchor still reset each rollout): 50.0 → 36.8, worse on four of five tasks. Put Back Block 40.0 → 18.0, Swap Blocks 56.0 → 36.0. Stale episode information actively interferes.

### Versus NativeMEM (compressed history)

Standalone MemBodied 50.0% vs NativeMEM 38.4%. But **NativeMEM wins on Put Back Block and Swap Blocks**; MemBodied wins the other three. Combining them gets 45.2% — *worse* than MemBodied alone. Their hypothesis: NativeMEM's history tokens pull attention away from the associative readout.

Efficiency, profiled on A100-80GB, ~2,550 policy cycles per method over 4 tasks:

| | Latency | Peak GPU | Added params |
|---|---|---|---|
| NativeMEM | 1593.7 ms | 20.57 GiB | 415M (12.81% of $\pi_0$) |
| MemBodied | 129.2 ms | 18.61 GiB | 40M (1.26% of $\pi_0$) |

### $\pi_{0.5}$ backbone and real robots

$\pi_{0.5}$: 12.4% → 48.0% ($3.87\times$), improving on all five. Rearrange Blocks 13.0 → 94.0.

$\pi_{0.5}$ has no state token in the suffix (state lives in the language prefix), so the memory token's own hidden representation serves as both read query and write key.

Real robots (two AgileX PiPER arms, 3 cameras, ~50 demos/task, 20 rollouts/task): 3.33% → 26.67%. Small absolute numbers; the confidence intervals will be wide.

### LIBERO — does memory hurt when you don't need it?

| Method | Spatial | Object | Goal | Long | Mean |
|---|---|---|---|---|---|
| $\pi_0$ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| MemBodied | 96.2 | 97.8 | 95.8 | **90.6** | 95.1 |

Spatial −0.6, Object −1.0, Goal flat, **Long +5.4**. The gain is concentrated exactly where episodes are long. Small regressions elsewhere, arguably noise at 500 rollouts per suite.

### The qualitative analysis is the best part of the paper

They hand-annotated 50 seed-matched rollout pairs per task. This isolates *where* memory helps:

- **Put Back Block.** Both policies move the block and press the button in **every** rollout. They differ only on where to return it. Stateless: wrong pad 37/50 — its 13 correct choices match the 13.8 expected from its own pad preference, i.e. it is guessing. MemBodied: wrong pad 13/50, corrects 24 stateless errors, introduces zero reverse errors. $p < 10^{-4}$.
- **Rearrange Blocks.** After the first placement the two mats look identical. MemBodied picks the correct block **50/50**. Stateless picks the left block 37/40 times it moves anything. But of its 20 correct picks, only **1** succeeds — it fails to press the button properly. Selection and execution errors compound.
- **Swap Blocks.** The hidden variable is *progress*, not location. Stateless presses "done" early in 41/50; MemBodied in 19/50. MemBodied completes all three moves 31/50 vs 8/50. Of MemBodied's 22 residual failures, 12 press after two moves — the last transition is the sticking point.
- **Battery Try.** Both seat both batteries at similar rates (23/50 vs 20/50) — grasping is not the difference. After a failed orientation test, stateless takes a median of four more attempts and solves 8/20; MemBodied usually solves in one more, 20/23 ($p = 0.028$ after Holm correction). 25 of MemBodied's 30 failures are dropped batteries, not memory.
- **Block Ranking.** Both complete an arrangement at similar rates (38 vs 40), but MemBodied then *tests* it 35/38 times vs 14/40. On four seeds the stateless policy builds the winning arrangement and dismantles it untested. But after a rejection, MemBodied builds and tests a new order in only **10/28** — long multi-attempt search is not fixed.

## Worth Remembering

**The delayed write is the architectural crux.** Writing "action + consequence" instead of "observation" is what makes the memory represent a *transition*. It is also what forces backprop-through-time training and forces the inference-time state cache. If you only remember one design decision, remember this one.

**No auxiliary memory loss.** The memory is trained purely by the action objective flowing backwards through the sequence. Simpler than memory-bank or retrieval approaches, which usually need a retrieval objective or a separate language channel. It also means the memory can only ever learn to store what helps the *current training distribution* of actions.

**Two failure classes remain, and they are cleanly separated.** Execution errors (stalling before release, missing the button, dropping batteries) — nothing to do with memory. And long-horizon accumulation errors — repeating a rejected block ordering, failing the third swap move. The single-event $M(1)$ recall problem is largely solved; the multi-event $M(n)$ search problem is not. Block Ranking at 22% is the honest tell.

**Limitations the authors state.** Cross-episode carryover hurts (the state is episodic, full stop). The anchor is task-dependent and actively harmful when the task is about progress rather than the starting scene. NativeMEM beats them on two of five tasks, and combining the two approaches makes things worse. The $\mu$VLA baseline result is theirs, not $\mu$VLA's.

**Practical caveats if you wanted to use this.**
- Sequence length is a per-task hyperparameter set from mean episode length (8 to 18). You need to know roughly how long your episodes are.
- The read/write cadence must match between training and deployment. The 10-slot round-robin on LIBERO is a workaround for a 5-step replan against 50-step memory training — and that write uses the *predicted* 50-step chunk, most of which was never executed. A small correctness compromise.
- Rank 128 had not saturated. Try 256.
- Gradient stopping at the visual patch tokens for the write path is not optional decoration — without it the memory objective would flow back into the vision encoder.

**Open questions.** Why does content injection beat attention steering by so much? The paper observes it but does not explain it — my guess is that a token participating in self-attention lets every action token read it with its own query, whereas steering applies one global correction. Does the gated delta rule actually forget selectively, or does $\alpha$ just saturate near 1? Nobody inspected the gates. And the combined NativeMEM + MemBodied regression deserves an attention-map analysis rather than a hypothesis.

**Connections.** The gated delta rule is exactly the [[Linear Attention]] / [[Why Gated DeltaNet Survives 4-Bit Quantization- NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM|gated delta rule]] machinery from language models, transplanted to control. The whole design is a [[Bayes Filter]] in spirit — carry a fixed-size summary of the past forward, update it with each new observation — but the update rule is learned rather than derived. And the read/write split is the same move as [[KV Cache]]: pay a fixed memory cost to avoid recomputation over history, except here the cost does not grow at all.

## Links

Related: [[POMDP]] · [[Markov Property]] · [[Linear Attention]] · [[Why Gated DeltaNet Survives 4-Bit Quantization- NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM]] · [[Flow Matching]] · [[LoRA]] · [[Cross Attention]] · [[Multi-Head Attention]] · [[Context Window]] · [[Lost in the Middle]] · [[Bayes Filter]] · [[State-Space Model]] · [[Memory]] · [[KV Cache]] · [[Imitation Learning]] · [[Diffusion Policy]] · [[Think Like a World Model, Act Like a VLA- Distilling World-Model Representations into Compact Robot Policies]] · [[Latent Action as Intention Enables Efficient Future Imagination for World Action Models]] · [[The Past Frames the Future- Memory for Autoregressive Video Generation]] · [[Long Short-Term Memory (Neural Computation)]] · [[Decoupled Weight Decay Regularization (AdamW)]]

New topics worth writing: Fast weight programmers, DeltaNet and the delta rule as memory write, Gated DeltaNet, RMBench and memory-complexity task taxonomies, $\pi_0$ flow-matching action experts, action chunking, LIBERO benchmark suites, Holm correction for multiple comparisons, Wilson binomial confidence intervals, truncated backpropagation through time
