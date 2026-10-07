---
title: "Register Tokens for Bounded-State Reasoning in Diffusion Language Models"
authors: ["Ge et al."]
year: 2026
arxiv: "2609.16372"
url: https://arxiv.org/abs/2609.16372
priority: Good-To-Read
read_on: 2026-09-22
tags: [paper, transformers, llm, rl, diffusion, theory]
---
## The Core Idea

A masked diffusion language model (dLLM) writes text by starting from a block of `[mask]` tokens and un-masking them over several passes, all positions at once. Because attention is bidirectional, every position can see every other position — unlike a [[Causal Attention|causal]] decoder, where a token can only look left.

That bidirectionality has a consequence nobody had cashed in: **a fixed position in the prompt can be written to, not just read from.** In a causal model, early positions are frozen once generated. In a dLLM, the hidden state at position 3 gets recomputed on every forward pass, and it can absorb information from tokens that come after it.

The trick here: reserve $R=4$ slots at the front of the prompt, call them **register tokens**, and train the model to stash its reasoning progress there. Then throw away all the generated text, keep only those four vectors, and carry on thinking.

> [!NOTE] Register token
> A fixed-position slot in a dLLM's prompt whose last-layer hidden state is read out after a chunk finishes and re-injected as the *input embedding* at the same position for the next chunk. It is a continuous vector in $\mathbb{R}^d$, not a token id. ^register-token

The setting is **bounded-state multi-chunk reasoning**. Generate $C$ tokens. Clear them. Generate $C$ more, seeing only the original question plus four carried vectors. Repeat. The active window never grows, no matter how long the reasoning runs. Compare that with the normal thing, where you append a fresh masked block to a growing context and attention cost goes quadratic in total length — the [[Long Context]] wall.

Why this did not exist before: everyone doing "keep thinking past the window" in autoregressive land carries **text**. Markovian Thinking keeps the last few tokens; auto-compaction writes a summary paragraph. Text is a narrow pipe. Four continuous vectors are a wider one — and in a dLLM they come for free from a mechanism the architecture already has.

What it unlocks, concretely: on code generation with a 64-token window, registers beat carrying four text tokens by up to **19.5 points** of pass@1, because a correct program almost never fits in one window and a four-token excerpt cannot tell the next window where the function was up to.

## The Methodology

**Background — the dLLM loss.** Corrupt the target $y_0$ by replacing each token with `[mask]` independently with probability $t \sim \mathcal{U}[0,1]$. Predict the originals:

$$\mathcal{L}_{\text{mask}}(\theta) = -\mathbb{E}\left[\frac{1}{t}\sum_{k \in M_t} \log p_\theta(y_0^k \mid c \oplus y_t)\right]$$

where $M_t$ is the set of masked positions and $c$ is the clean prompt. It is [[Cross Entropy|cross-entropy]] on the masked slots, up-weighted by $1/t$. See [[Discrete Diffusion]].

**Inference (five lines).**

1. Start with the prompt, registers initialised to the `[mask]` input embedding.
2. Denoise a fully masked chunk of $C$ tokens to completion.
3. Run **one extra clean forward pass** over prompt + finished chunk.
4. Read the last-layer hidden states at the four register positions → an $R \times d$ tensor.
5. Clear the chunk. Inject those vectors as input embeddings at the same four positions. Go to 2.

The same four slots are overwritten every time, so the model has to learn what to keep and what to bin.

**Training — chunked SFT.** Split a long reasoning trace into chunks $\mathbf{t}_0, \dots, \mathbf{t}_K$ of at most $C$ tokens. Train each with the masked objective above. For chunk $k>0$, first run the register-write pass over the *clean* chunk $k-1$, then feed those vectors in. That write pass stays in the computation graph, so gradient from chunk $k$'s loss flows back through exactly one boundary. The state handed to chunk $k+1$ is detached — training memory stays bounded.

**The two shortcuts, and how they are blocked.** This is the part that actually carries the paper. Left alone, the model ignores the registers entirely, because it has two easier routes to the answer.

*Shortcut 1 — just re-read the prompt.* Fix: an attention mask where completion positions **and** register positions cannot attend to prompt keys, at every layer. (Blocking registers too matters — otherwise the completion reaches the prompt indirectly, through the registers.) Applied per-trace with probability $p_{\text{prompt}} = 0.3$ for math, $0.7$ for the code continuation.

*Shortcut 2 — infer from the tokens already un-masked in this chunk.* Fix: take $M=4$ denoising passes per chunk, with the first forced to $t=1$ (everything masked) and the rest at $t \sim \mathcal{U}[10^{-3}, 1]$.

When both bite at once — and they do on $p_{\text{prompt}}/M = 7.5\%$ of continuation updates — the registers are the only wire to anything.

> [!NOTE] Why the two masks are the whole method
> On a prompt-masked, fully-masked pass, a lemma by induction over layers shows every hidden state in the "state subgraph" (registers + completion) is a function of $(\mathbf{r}, \ell)$ alone, where $\ell = (P, C)$ is just the two lengths. So
> $$\mathbb{E}[-\log p_\theta(Y_j \mid \mathbf{r}, \ell)] \geq H(Y_j \mid \ell) - I(Y_j; \mathbf{r} \mid \ell)$$
> The best predictor that ignores the registers pays $H(Y_j \mid \ell)$. Beating it *requires* the registers to carry mutual information about the target. The mask converts "please use the registers" into "you cannot do better without them." ^register-pressure

**Setup.** LLaDA-8B-Base and Dream-7B-Base. 60K traces: 30K OpenMathInstruct-2, 30K OpenCodeInstruct. One epoch, LR $2\times10^{-5}$, batch size 1. Math chunks $C=128$ (up to 8 chunks); code chunks $C=64$ (up to 16). Both get the same 1024-token total budget. Greedy decoding, zero-shot chat prompts.

**Three baselines, all trained on the same data.**

| Baseline | What it carries |
|---|---|
| **Discrete text** | The last four generated token *ids*, pasted into four text slots at the front of the next chunk. Trained with $p_{\text{prompt}}=0$. |
| **Full-sequence SFT** | Nothing. Ordinary post-training on whole traces, then evaluated under the same bounded protocol. |
| **Memory tokens** | Same four slots, same read/write machinery, but trained to *reconstruct* the previous chunk: $\mathcal{L} = \mathcal{L}_{\text{next}}(\text{sg}(\mathbf{r}_k)) + 0.05 \cdot \mathcal{L}_{\text{recon}}(\mathbf{t}_{k-1}, \mathbf{r}_k)$. An ICAE-style compressor. Note the stop-gradient: the task loss trains the reader, reconstruction trains the writer. |

## Ablation Studies and Experiments

**The headline table** (percentages, first-answer accuracy for math, pass@1 for code):

| Benchmark | Full-seq SFT | Discrete text | Memory tokens | **Registers** |
|---|---|---|---|---|
| *LLaDA-8B* | | | | |
| GSM8K | **57.4** | 40.6 | 48.4 | 49.1 |
| GSM-Hard | **21.7** | 13.0 | 18.6 | 16.9 |
| MATH500 | **22.4** | 14.6 | 15.6 | 17.6 |
| HumanEval | 14.0 | 18.3 | 12.8 | **26.2** |
| MBPP | 25.7 | 18.3 | 10.5 | **29.2** |
| *Dream-7B* | | | | |
| GSM8K | **49.6** | 35.0 | 37.3 | 42.9 |
| MATH500 | **24.2** | 10.8 | 12.6 | 12.4 |
| HumanEval | 15.9 | 25.6 | 2.4 | **30.5** |
| MBPP | 30.0 | 21.4 | 4.7 | **40.9** |

Registers beat discrete text in all 12 rows and beat both carry baselines in 10 of 12.

**The result the authors did not want.** Full-sequence SFT — which carries *no state at all* — wins every single math row. And every correct answer it gives arrives in **chunk 1**, even though only 3.7% of its training completions fit in 128 tokens. Training on whole traces apparently teaches it to write short. The honest reading: on GSM8K-style problems, a carried state is not needed, because the answer fits in one window.

**Where carry actually earns its keep.** On code at $C=64$, only 3.7–6.6% of Dream register generations finish in chunk 1. Of the registers' 35.7 accuracy points on Dream code, **32.4 come from programs that finish after a reset**. Carry helps exactly when the required output does not fit in one window — no more, no less.

The mirror-image control makes the same point. Run the register model (trained at $C=128$) at $C=64$ on math: it still answers in chunk 1 on 95–98% of examples, so carrying versus resetting buys only **0.3 points**, and it stays below plain SFT. Discrete text, which habitually runs long, scores 30.2 with carry and **0.3 with reset** on GSM8K — totally dependent on its channel.

**Do later chunks actually read the state?** Two tests.

- Carry vs reset on the retrained LLaDA register model: 49.1 → 44.4 on GSM8K, mean drop 3.8 points across four math sets. Reset accuracy *exactly equals* carry's chunk-1 accuracy on every benchmark, which is the sanity check you want.
- Intervention (historical checkpoint, GSM8K): intact registers **46.4**; replace with Gaussian noise rescaled to each slot's original norm **32.7**; reset to the initial embeddings **21.5**. So direction carries 13.7 points of real content, and mere presence-at-the-right-norm carries another 11.2. It is not just extra positions or extra capacity.

**What is written in there?** On a LongArithmetic checkpoint (running sums, where the needed state is obvious), linear ridge probes with a problem-level train/test split:

| Target | Metric | Registers | Majority | Shuffled |
|---|---|---|---|---|
| Final answer | Pearson $R$ | 0.85 | – | −0.06 |
| Running total | Pearson $R$ | 0.84 | – | −0.01 |
| Final-answer sign | Acc | 94.1% | 52.9% | 48.7% |
| **Next operation** | Acc | **80.0%** | 56.7% | 50.7% |

The last row is the interesting one: *before* any answer is emitted, the registers linearly encode which operation comes next. They hold both "where I got to" and "what to do next."

**Slot count.** At a fixed 30K-trace budget, registers beat discrete text by +1.3 / +4.7 / +1.2 points at $N=1/4/8$. At $N=16$ registers *lose* (31.8 vs 36.1) — but this turns out to be an optimisation artefact, not a capacity ceiling: at 64K traces they reach parity (38.1 vs 38.9), and with a warmup–stable–decay cooldown at 80K they win (44.9 vs 42.7). Training loss shows the same pattern — discrete text drops faster early, registers cross below it late. Bigger register banks need more steps to learn a stable write/read protocol.

**A genuine instability.** Continuing that 80K run at constant LR $2\times10^{-5}$ *destroyed* the register model: 38.1 → 27.4, the fraction of GSM8K completions with no boxed answer went 0% → 31%, mean chunks consumed 1.9 → 4.0. The discrete-text twin was fine. One seed, so treat as a warning not a law — but a continuous carry channel appears more fragile to a hot learning rate than a discrete one.

**RL on top.** *Chunked diffu-GRPO* extends [[GRPO]]-style group-relative advantage to the carry setting. For rollout $g$, find $\tau_g$ = the first chunk whose accumulated prefix is correct; reward the prefix there; advantage $A_g = \bar r_g - \frac{1}{G}\sum \bar r_{g'}$; mask out chunks after $\tau_g$ with $w_k^g = \mathbf{1}[k \le \tau_g]$; clipped ratio objective with a reference-[[KL Divergence|KL]] term, same shape as [[Proximal Policy Optimization Algorithms|PPO]]. At the chunk-$k{+}1$ update the preceding state write is recomputed with gradients on, so reward credit reaches the writer.

| Task | Discrete text | Registers | Gain |
|---|---|---|---|
| Countdown | 19.7 | 22.3 | +2.6 |
| LongArithmetic | 31.6 | 39.7 | +8.1 |

Note that discrete text has **no differentiable state path** — sampled token ids — so reward only reaches the policy through next-chunk likelihoods. That asymmetry may be most of the gap.

**The synthetic study, which is the most instructive failure.** Four-layer width-256 dLLM trained from scratch on modular running sums, four 16-token chunks, chance = 10%.

| Condition | Chunk 1 | Chunk 2 | Chunk 3 | Chunk 4 |
|---|---|---|---|---|
| Reset (no carry) | 100.0 | 9.3 | 10.7 | 9.6 |
| Discrete text (oracle here) | 100.0 | 99.9 | 99.8 | 99.8 |
| Registers, task loss only, 100K steps | 100.0 | 10.3 | 10.8 | 10.7 |
| Registers, full BPTT, 24K steps | 100.0 | 10.0 | 9.0 | 10.0 |
| Registers, + state supervision, seed 0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Registers, + state supervision, **seed 1** | **21.4** | 10.1 | 10.1 | 10.1 |

The write/read protocol **does not emerge from the task loss**, not at 100K steps (~13M traces), and not with full backprop through every boundary. Adding a direct linear head that supervises register 1 against the true boundary total unlocks it — and in one seed the supervision can be annealed to zero halfway and the protocol survives. But seed 1 collapses entirely. The authors call this an existence proof, not a recipe.

The plausible story: a coordination failure. The reader has no reason to attend to slots that are currently noise; the writer gets no gradient until the reader attends. Formally, the writer path is $\nabla_\theta \mathcal{L}|_{\text{writer}} = \frac{\partial \mathcal{L}}{\partial \mathbf{r}} \frac{\partial \mathbf{r}}{\partial \theta}$, which is exactly zero if the reader ignores $\mathbf{r}$.

**Other things that did not work:**
- **Memory tokens collapse on code.** Dream HumanEval 2.4, MBPP 4.7 — worse than the raw base model. Reconstruction-style compression ([[Auto-Encoding Variational Bayes (VAE)|autoencoder]]-flavoured) is the wrong objective when the next chunk needs *what to do next*, not *what was just said*.
- **Self-summarisation control.** Let the discrete-text model generate its own four-token summary instead of using its trained slots: GSM8K 37.5 vs 45.7 for the trained channel (and 23.0 for a fresh reset). Explicit summarising is worse than a learned last-$N$ carry.
- **Instruction-tuned pilot.** Applying the unmodified base recipe to LLaDA-8B-Instruct drops it from 46.6 to 37.1 average. The recipe does not transfer without retuning.

**Cost and accuracy trade.** At a 1024-token horizon, just keeping the whole trace is much better — 63.2 vs 48.9 on GSM8K. Registers buy speed, not accuracy: the extra write pass is ~1.6% overhead (1 pass on top of 65 denoising passes), and with early stopping off, carry stays flat at ~4.7 s/chunk while full context grows — **3.4×** faster at 2048 tokens, **5.6×** at 3584.

## Worth Remembering

**The comparison is not clean, and the authors say so.** Registers train with $p_{\text{prompt}}=0.3$; discrete text trains with $p_{\text{prompt}}=0$. So every reported gap is *channel plus recipe*, not channel alone. They never ablate the prompt-mask probability or $M$ separately. Most SFT comparisons are one seed.

**Four slots is a slot-count match, not a capacity match.** Four $d$-dimensional float vectors hold vastly more than four token ids. That is the point of the method, but it means "registers beat discrete text at equal slot count" is a weaker statement than it sounds.

**There was a scoring bug, handled unusually honestly.** The original evaluator ignored plain `<answer>` tags; a later analysis accepted them but also credited *later* correct answers after an earlier wrong one. Everything in the main table is rescored with one strict first-answer rule, replayed over saved chunks, and every number that could not be rescored is explicitly labelled "historical" and kept separate. On Dream GSM8K this flipped the apparent result: stored online scores said 33.3 (registers) vs 35.6 (discrete); corrected scores say 42.9 vs 35.0. Worth copying the practice.

**The real lesson may be about objectives, not registers.** Three ways to train the same four slots: task loss alone (fails on synthetic, works on pretrained 8B), reconstruction (works on math, collapses on code), next-chunk task loss with shortcut-blocking (works). The slot mechanism is easy; making gradient flow into it is the hard part.

**Connection to things you already have.** This is the bidirectional-attention cousin of the `[CLS]` token in [[BERT- Pre-training of Deep Bidirectional Transformers]] and the register tokens in [[An Image is Worth 16x16 Words (ViT)|ViT]] — except those aggregate *within* one forward pass and are read-only downstream. Here the slot is written, cleared, and re-read across resets, making it closer to a learned recurrent state. It is also a direct alternative to [[KV Cache]] truncation: instead of dropping old keys and values, compress the whole history into four persistent positions.

**Open questions worth chasing.** How much state fits in $R$ slots before it saturates? The paper only tests $R \in \{1,4,8,16\}$ on one task family. Can the same write/read protocol compress a long *input* rather than a long *output* — a trained alternative to [[RAG]] or cache eviction? And does the [[Lost in the Middle]] failure show up for carried state the way it does for long context?

**Practical caveats if you wanted to build this.** (1) You need a bidirectional model; this does not port to a causal decoder. (2) Budget for the write pass — cheap, but it is a full forward pass per boundary. (3) Expect the register channel to need more steps and a decaying learning rate; constant LR blew up the $N=16$ run. (4) Do not expect accuracy gains over full-context decoding — the sell is bounded memory and flat per-chunk latency.

## Links

Related: [[Discrete Diffusion]] · [[Diffusion Models]] · [[Attention]] · [[Causal Attention]] · [[Context Window]] · [[Long Context]] · [[KV Cache]] · [[Memory]] · [[Context Engineering]] · [[GRPO]] · [[Proximal Policy Optimization Algorithms]] · [[Cross Entropy]] · [[Chain of Thought]] · [[An Image is Worth 16x16 Words (ViT)]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Lost in the Middle]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Sparse Attention]]

New topics worth writing: LLaDA and Dream (open masked-diffusion LLMs), ICAE and gist-token context compression, Recurrent Memory Transformer, Coconut and continuous latent reasoning, Markovian Thinking and bounded-state reasoning, diffu-GRPO, linear probing as an interpretability tool, warmup-stable-decay learning rate schedules
