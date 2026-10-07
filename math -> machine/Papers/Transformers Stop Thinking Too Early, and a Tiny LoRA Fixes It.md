---
title: "Transformers Stop Thinking Too Early, and a Tiny LoRA Fixes It"
authors: ["Zehao Jin", "Ruixuan Deng", "Junran Wang"]
year: 2026
arxiv: "2609.36585"
url: https://arxiv.org/abs/2609.36585
priority: Good-To-Read
read_on: 2026-10-03
tags: [paper, transformers, llm]
---
## The Core Idea

Give a language model this:

```
K = apple
B = K
D = B
print(D)
```

and it answers `apple`. Add a few more lines to the chain and it breaks. Across thirteen pretrained base models (Qwen3, Llama, OLMo-3, Gemma-3), the longest chain followed at 80% accuracy is **1.4 to 3.6 lines**, median 2.2. A 292B mixture-of-experts model (DeepSeek-V4-Flash) reaches 4.0 and then sits at chance. OLMo-3-32B has 64 layers and reaches 2.6 — the same as its 32-layer sibling. Depth is not being spent on following references.

The surprise is how little it takes to fix. Train a **rank-8 [[LoRA]] on the residual stream at one early layer**, freeze every original weight, and Qwen3-8B goes from **15.5% to 99% exact accuracy on 24-line chains**. That is 65,537 trained parameters, under 0.01% of the model. A LoRA trained on longer programs reaches 50 lines in a single forward pass.

So why does a tiny edit do so much? Because the edit does not do the work. The frozen layers do. The paper traces what changes:

> [!NOTE] The relay ^relay-def
> By default, each program line only figures out *which chain it belongs to* for the first two or three lines, and the query token resolves one or two more pointers before a late layer copies the answer value. With the LoRA, the program's own lines keep passing chain identity up the chain — line 5 learns its root, then line 8, then line 13, then line 16 — inside a short band of middle layers. The paper calls this a **relay**. The LoRA only *starts* it; frozen [[Attention]] heads carry it.

This reframes what a wrong answer means. A model failing a 20-hop in-context lookup is not evidence that the weights cannot do 20 hops. The computation is latent in the frozen weights and reachable with a 65K-parameter edit. Evaluation of "default behaviour" systematically understates accessible computation.

The second payoff is that **loops now pay off**. Pretrained recurrent-depth models (Ouro, Huginn) gain almost nothing from extra loops when frozen — Ouro-1.4B's reach after one through five loops is 0, 1.6, 2.3, 2.2, 2.2 lines. With the LoRA, extra loops extend the chain: **60 lines after four loops, ~146 after six, at least 160 after eight** (87% accuracy at 160 lines, four times the longest training chain). The loops were always able to repeat the relay; nothing had started one.

## The Methodology

**The task.** A program has $c$ chains of $d$ assignments each, shuffled together, then a query. The root of a chain stores a single-token noun; every later line names the previous variable. A line's **pointer** is the variable on its right-hand side; its **parent line** is where that variable was defined. Two orderings: *level order* (grouped by depth, shuffled within depth) and *interleaved* (randomly merged, definition before use). Names, nouns and queried chain are randomised.

Two scores. *Choice accuracy* compares logits only among the chains' root values (chance $1/c$). *Exact accuracy* requires the right root to beat the whole vocabulary. **Reach** = longest chain answered at ≥80%, linearly interpolated at the first downward crossing.

**The intervention.** At one layer's input, replace the residual state $h$ with

$$h \leftarrow M(h) = s\,h + BA\,h$$

with $A \in \mathbb{R}^{8 \times n}$, $B \in \mathbb{R}^{n \times 8}$, $s$ a learned scalar, no bias. This is the DiReFT form of a representation intervention. Two things matter about its shape:

- It acts **independently at each token**. It moves no information between positions. All token-to-token communication is still done by frozen attention and MLPs.
- The learned scale stays essentially at identity: $s = 1.0006$ for the Qwen LoRA, $1.004$ for Ouro. So it is not a global rescaling trick.

**Training.** AdamW, lr $10^{-3}$, 1,200 steps, batches of 16 two-chain level-order programs up to 20 lines. Loss is answer [[Cross Entropy|cross-entropy]] plus $\mathrm{KL}(p_0 \| p_M)$ on WikiText-103 passages (weight 1) to keep general text behaviour from drifting — a measured control, not a guarantee. [[Perplexity]] moves 10.14 → 10.148 in Qwen, 13.294 → 13.302 in Ouro. The standard Qwen LoRA enters **layer 14 of 36**; Ouro's enters layer 6 and is applied in every loop.

**The models.** Thirteen standard base models, 16–64 layers. Two looped families: Ouro-1.4B and Ouro-2.6B (24 and 48 layers per loop, trained with four loops) and Huginn-0125 (a four-layer recurrent core trained with ~32 recurrences on average).

**How the computation is located.** Three instruments, used together:

1. **Causal tracing.** Change one root noun or redirect one pointer, then patch the clean residual state at position $p$, layer $\ell$ into the counterfactual run. Score the recovered fraction of the answer-logit gap:
$$S(p,\ell)=\frac{g(x_{\rm cf};h^{\rm clean}_{p,\ell})-g(x_{\rm cf})}{g(x_{\rm clean})-g(x_{\rm cf})}$$
where $g$ is the clean-minus-counterfactual final-token logit difference.
2. **Linear read-outs.** A ridge probe predicts, from a line's pointer state, *which chain's root* that line ultimately refers to. A line counts as readable at 75% (two chains). Relay progress = longest prefix where every line passes. Note this probe targets *chain membership*, not the noun — it separates "where to read" from "copy the value".
3. **Attention knockouts.** Measure attention from a line's pointer to the pointer $j$ lines earlier **on its own chain**, minus attention to the matching line of the other chain. Heads are selected on one half of the programs and scored on the other half (an honest split that most interpretability papers skip).

## Ablation Studies and Experiments

**The default pass, measured.** In Qwen3-8B the root value stays at its source token until the query takes it over near **layer 32 of 36**, almost regardless of chain length. Pointer effects arrive at the query earlier, nearest pointer first. The interval over which one pointer's effect falls from 90% to 10% is $7.1 \pm 1.4$ layers across models with 16–64 layers — **roughly constant width, not proportional to depth**. Blocking query attention to pointer lines hurts in the middle layers but costs at most two points between the end of that window and the late value copy. So: short context relay, a couple of query-side hops, a late copy, and then layers that no longer extend the chain.

**The main result.** Layer-14 LoRA, two chains, level order, exact accuracy:

| lines | 4 | 8 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|
| frozen | 57.5 | 44.5 | 25.5 | 15.5 | 14.5 | 15.5 |
| LoRA | 100 | 100 | 99.5 | 98.5 | 98.0 | 99.0 |

24-line is 198/200 (Wilson 96.4–99.7%); three training seeds give 94.7–97.5%. Harder distributions degrade but stay far above frozen: interleaved order drops 24-line to 76.5%; three chains drop 16-line to 58.0%. A separately trained three-chain LoRA transfers from Python-style code to JavaScript (`const B = A;`, 41% → 99%) and to English (`B means the same as A.`, 0% → 94%).

**Where the relay lives.** With the LoRA, chain identity becomes readable at lines 5, 6, 6, 8, 9, 13, 16 at the outputs of **layers 16–22**. Frozen, it reaches line 4 at layer 17 and stops. Ouro's relay advances almost entirely in **layers 7–15 of every loop**, reaching lines 4, 9, 25, 40 across the first four loops. Ouro-2.6B, with 48 layers per loop, still uses a band of ~11 layers and leaves 37 contributing at most two lines. The relay band is narrow and its width does not scale with depth.

**Attention reaches further, step by step.** With the LoRA, Qwen's chain-selective attention reaches two lines up at layers 16–18, then 3, 4, 6, 7 at layers 19–22. Each layer advances the relay by **at most** that distance. This is *not* pointer doubling — distances grow by one or two and attention spreads over several earlier lines (effective number 1.6–5.0) rather than making discrete power-of-two jumps.

**The heads already existed.** In Ouro, heads that previously read no further than the parent line take over the long reads (attention beyond the parent rises from ≤0.02 frozen to 0.78–1.06), while the strongest frozen parent-readers read *less* far. In Huginn the same head (layer 3, head 34) takes over directly: 0.01 → 0.74. Ablations confirm these are load-bearing, against layer-matched random controls:

| cut | accuracy | matched random |
|---|---|---|
| 10 parent-reading heads (Qwen, 16-line interleaved) | 53% | 83% |
| 8 longer-reading heads | 65% | 90% |
| 7 longest-reading heads (Ouro, 4 loops) | reach 12.0 | median 24 |

**Parent attention is necessary, and only in the relay band.** Removing each pointer's attention to its parent line in **layers 14–22** returns 6-, 8- and 12-line chains to chance (53%, 48%, 55%). The *same cut in layers 23–29* leaves 100%, 100%, 98%. Removing attention two-or-more lines up (but keeping the parent) drops Ouro's reach from 17.8 to 9.4. Three-chain controls show the useful attention is along the line's **own** chain; cuts to other chains or random lines barely matter. The effect survives both zeroing-after-softmax and renormalising masks.

**What is being passed.** Linear probes decode the *names* defined three to seven lines up, appearing in distance order at layers 17–23 (0.41 down to 0.14, chance 0.03); frozen never exceeds 0.19 beyond two lines. Names arrive *before* chain membership is readable. Blocking parent attention in Ouro's loop 2 drops decoding of names four lines up from 0.29 to 0.18, near loop 1's 0.13. Once a line's chain is resolved, the names fade. Reading: parent attention carries names down the chain, longer attention uses those names to find the chain's earlier lines.

**The relay is not in the LoRA's own subspace.** Chain membership is no more readable in the 8-dimensional span of $B$ than in a random 8-dimensional subspace; deleting that span from later states leaves read-out near 95%. Applying the LoRA only to program tokens keeps the gain; only at the query adds 2–4 lines. In Ouro a **first-loop-only** LoRA lets the three unmodified later loops carry on to ~25 lines. All of this places the long computation in the frozen weights.

**Re-running the useful layers.** Re-entering Qwen3-8B's layers 14–22 (with the LoRA at each re-entry) raises 64-line exact accuracy from 34% (one pass) to 66% (one re-entry) to 92% (two). Frozen: 10%. This is [[Test-Time Compute]] spent on exactly the band that carries the relay.

**Other edits work at the same place — and fail at the same place.** Qwen3-8B, 24-line:

| intervention | params | layer 14 | layer 26 |
|---|---|---|---|
| residual LoRA | 66K | 96.5 | 54.5 |
| projection LoRA (all 7 projections) | 606K | 90.0 | 52.5 |
| FLAS flow, 3 steps | 66K | 94.0 | 54.0 |
| FLAS block, 1 step | 168M | 95.5 | 56.0 |

Chance is 50%. A 168M-parameter block at layer 26 does no better than a 66K one. Placement dominates parameter count.

**The placement limit — and the honest caveat.** Holding rank, data and optimisation fixed, moving the Qwen LoRA from **layer 20 to layer 21** drops reach from 20.5 to 5.2 lines. Analogous cliffs at layers 12–15 (OLMo-3-7B) and 13–15 (Llama-3.1-8B), reproduced with second seeds. At layer 20 the relay still starts (attention reaches 3–5 lines up by layer 24); at layer 21 attention stays at the parent (≤0.11 beyond) and nothing starts. Late layers can *continue* a relay but cannot *begin* one.

They tried to predict this limit from the frozen model: the **cutoff layer** = first layer where restoring a pointer's own token recovers less than half its effect (averaged over lines 2 and 3 of three-line programs). Qwen's is 20.5. They **preregistered** the prediction (timestamped before the held-out runs) against two alternatives fitted on five development models, with success = within the bracket ±1 layer in ≥3 of 4 held-out models.

| held-out | bracket | cutoff |
|---|---|---|
| Llama-3.2-1B | 7–8 | 8.0 ✓ |
| Qwen3-4B | 20–22 | 22.0 ✓ |
| Gemma-3-12B | 24–27 | 23.0 ✓ |
| OLMo-3-32B | 22–26 | 19.0 ✗ |

Criterion met: MAE 2.25 layers vs 3.45 for "45% of depth". **But** across all nine models the cutoff's error is 2.22 layers against **2.27 for a post-hoc "48% of depth" rule**. The authors say so plainly: the cutoff is a starting point for a local sweep, with no demonstrated advantage over relative depth. That is the paper's cleanest negative result.

**Loop timing resolves an apparent contradiction.** An Ouro LoRA at layer 20 — *after* the relay band — still works if applied every loop, because it precedes the *next* loop's layers 7–15; it just lags by about a loop. Apply it only in the **last** loop and it collapses: layer 6 reaches 21.6 lines, layer 20 reaches 2.6 (frozen 2.5). Ouro-2.6B separates this from "unused depth": a last-loop LoRA at layer 24 has 24 layers after it but no middle band, and reaches 4.7; at layer 12, inside the band, 9.6. What matters is whether *useful middle-layer computation* still follows the edit, not how many layers remain.

**Things that did not work or reversed.**
- Loops are not free: Ouro loses reach past ~2× its training loop count (at 12 loops, 4-line accuracy drops to 91%, 64-line to 68%). Huginn peaks at 16 recurrences (58 lines) then falls to 30 at 32 and 13 at 64. Classic overthinking.
- Few-shot demonstrations fix output *format* and nothing else: Ouro's one-line exact accuracy goes 3% → 100% with four solved examples, while reach stays ≤3.3. Pure [[In Context Learning]] does not buy the computation.
- Training with fewer loops makes the relay faster per loop but caps it: LoRAs trained with 1/2/4 loops reach 12.5, 4.4, 1.9 lines in the *first* loop, and the one-loop LoRA actually gets *worse* with a second loop (reach ~2).
- A LoRA trained on single-hop SQuAD does not improve MuSiQue ($-2.3$ EM), so the multi-hop gain is not just answer formatting.
- Dropping the time embedding from the FLAS block costs a lot (24-line 95.5% → 70.0%).

**MuSiQue.** This tests the *placement* claim on a real benchmark (not transfer of a program-trained LoRA — each is trained separately on MuSiQue, answer tokens only, shuffled gold paragraphs, 900 dev questions, paired bootstrap intervals, identical prompts).

| model | frozen EM | best early LoRA | late LoRA |
|---|---|---|---|
| Qwen3-8B | 52.9 | +11.4 (layer 6) | $-1.6$ (layer 30) |
| OLMo-3-7B | 54.4 | +9.4 (layer 4) | $-3.8$ (layer 24) |
| Llama-3.1-8B | 46.8 | +17.9 (layer 4) | +1.3 (layer 20) |

Early projection LoRA keeps most of the all-layer gain: 11.0 vs 12.8 (Qwen), 10.6 vs 10.8 (OLMo), 20.1 vs 20.9 (Llama). Late projection LoRA gains 0.4, 0.9, **10.7** — Llama's useful region runs past its program-measured limit, which the authors flag rather than hide. Equal-width nine-layer quarters in Qwen give +9.0, +9.4, +4.9, $-3.1$: parameter count is not the story. A 4,097-parameter steering vector at layer 14 gets +7.8, only 1.5 below the 65K LoRA.

On fictional-fact multi-hop questions (cleaner than MuSiQue, still compositional), Qwen3-8B goes 41.2% → 97.4% EM on 2–5 hops and 23% → 88% on unseen 6-hop. Ouro extends reliable answers from 2 hops to 6 at four loops, and to 8 hops (91–92%) at six or eight loops — twice the training hops.

**Was the ability learned, or installed?** Two checks. (1) The identical LoRA recipe on OLMo-3 pretraining checkpoints unlocks 1.5 / 10.2 / 23.7 lines at ~21B / 84B / 336B tokens, while frozen reach stays below 2. The jump between 21B and 84B is specifically *following one link* (one-line lookup already 99% at 21B, two-line only 58.5%). (2) Small looped transformers trained from scratch on the programs learn the same relation between attention distance and relay progress — the relay advances by exactly its attention distance in 108 of 165 layer steps, more in 8, less in 49.

## Worth Remembering

**The transferable lesson is about where to intervene, not about this task.** If you are placing a [[LoRA]], an adapter, or a steering vector, the question is not "how many layers are left" but "is there still *useful computation* after this point". A 168M block placed after the relay band lost to a 66K edit placed inside it. Sweep a few early-to-middle layers; the frozen cutoff measurement gives you a prior worth about two layers of precision, which is roughly what "48% of depth" gives you for free.

**"The model got it wrong" and "the model cannot do it" are different claims.** This is the cleanest statement of that distinction I have seen with a mechanism attached. Default-pass benchmark numbers measure a *default* computation: a short relay, a couple of query-side hops, a late value copy. The paper's phrasing is worth keeping — *evaluation should distinguish default behaviour from computation accessible under a specified intervention*.

**Loops are a multiplier on something that must already be running.** Frozen Ouro gains nothing after loop 3. The same weights, with a 32K-parameter edit, go to 160 lines. This cuts against reading flat loop-scaling curves as evidence that recurrent depth does not work.

**Honest limits the authors state.**
- The mechanistic tasks are synthetic and fully in-context. MuSiQue supports the *placement* claim, with gold paragraphs in the main setting; it is not evidence that the same relay runs there.
- Detailed traces are Qwen3-8B and Ouro-1.4B only; two looped families, and Ouro-2.6B is grown from Ouro-1.4B (not independent).
- A LoRA is trained per task format. Nothing here is a general capability unlock.
- The WikiText KL is a *measured* control on one distribution, not a guarantee about broad behaviour. [[Fine-Tuning#catastrophic-forgetting|Forgetting]] elsewhere is untested.
- First-loop-only sufficiency is only shown through ~25 lines.
- The learned routing directions and the minimum sufficient rank are unidentified. The causal tests constrain the algorithm without pinning it down.

**Methodological habits worth stealing.** Preregistering a prediction with a timestamp and reporting that it ties with a trivial baseline. Selecting attention heads on one half of the programs and scoring on the other. Random-head ablation controls matched by layer *and* count. Reporting that the original paragraph shuffle used Python's process-salted string hash, invalidating cross-process comparisons, then re-running everything with a fixed seed and paired bootstrap intervals. Flagging "these are also the best tested layers on the same dev set, with no held-out layer-selection split."

**Open questions.** Does the relay exist in models trained on natural multi-hop text, or only under this intervention? Can the placement band be predicted without any gradient steps? Why does Llama's useful MuSiQue region extend past its program-measured limit — is that a second mechanism? And if the heads already exist frozen, what exactly is the 8-dimensional edit writing that re-routes them?

## Links

Related: [[LoRA]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Attention]] · [[Multi-Head Attention]] · [[Query, Key, and Value (QKV)]] · [[Causal Attention]] · [[Sparse Attention]] · [[Fine-Tuning]] · [[In Context Learning]] · [[Chain of Thought]] · [[Test-Time Compute]] · [[Gated Recurrent Transformers- Expressive Depth through Recurrent Modulation]] · [[KL Divergence]] · [[Cross Entropy]] · [[Perplexity]] · [[Attention Is All You Need]] · [[Sparse Readout Prism- Explaining Logit-Lens Scores in Features Instead of Tokens]] · [[Long Context]] · [[Lost in the Middle]] · [[Register Tokens for Bounded-State Reasoning in Diffusion Language Models]]

New topics worth writing: causal tracing and activation patching, induction heads, variable binding and entity tracking, representation fine-tuning (ReFT/DiReFT), recurrent-depth and looped transformers, steering vectors and activation addition, multi-hop question answering (MuSiQue), preregistration in interpretability research, overthinking in recurrent networks
