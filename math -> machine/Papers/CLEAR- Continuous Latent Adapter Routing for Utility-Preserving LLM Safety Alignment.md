---
title: "CLEAR: Continuous Latent Adapter Routing for Utility-Preserving LLM Safety Alignment"
authors: ["Wang et al."]
year: 2026
arxiv: "2608.21278"
url: https://arxiv.org/abs/2608.21278
priority: Good-To-Read
read_on: 2026-08-31
tags: [paper, llm]
---
## The Core Idea

Safety training makes a model refuse bad requests. It also makes the model dumber on good requests. That second part is the **alignment tax** — the same weight update applies to every input, so a prompt about chemistry homework gets pushed toward refusal just as hard as a prompt about making nerve gas. On Llama-3-8B-Instruct, plain safety SFT drops GSM8K math accuracy from 75.06% to 66.34%. That is eight points of reasoning burnt to buy safety.

CLEAR's move: **do not bake safety into the weights. Bake it into a switch that is off by default.**

Freeze the whole backbone. Add one small [[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]] adapter trained only to produce refusals. Then add a tiny neural network — a "gate" — that reads the model's own hidden states for the prompt and outputs one number $g(x) \in [0,1]$. Every adapted layer computes

$$h' = (W + g(x)\,\Delta W_\theta)\,h.$$

If the gate says $0$, you are running the original frozen model, bit for bit. If it says $1$, you are running the fully safety-tuned model. Anything in between is a blend.

Why this is not just "put a classifier in front of the model": the gate is **not** an external filter that blocks the request. It is trained *jointly* with the adapter, and it modulates the forward pass. The adapter learns "what safe behaviour looks like *when I am switched on*", and the gate learns "how far on should I be". Neither is trained in isolation, so the adapter never has to work at half strength on a prompt the gate is unsure about — the two co-adapt.

What it unlocks: on Llama-3-8B-Instruct, HarmBench attack success rate falls from 32.25% to 0.50% while GSM8K stays at 73.46% — about 7 points above both SFT (66.34%) and always-on LoRA (66.72%). The safety is nearly free.

> [!NOTE] Conditional latent routing
> Instead of one globally modified policy, you keep the frozen policy and a safety policy, and interpolate between them per-input using a score computed from the model's own internal activations. Safety becomes a *dial*, not a *rewrite*. ^conditional-routing

---

## The Methodology

**The gate.** For prompt $x_i$, run the frozen backbone on the prompt tokens only (no assistant tokens — the score must exist before decoding starts). Grab hidden states from three chosen layers, average across layers and token positions into one vector $z_i$, feed to a 2-layer MLP with sigmoid output:

$$z_i = \operatorname{Agg}\big(\{H_i^{(\ell)}\}_{\ell \in \mathcal{L}}\big), \qquad g_i = G_\psi(z_i).$$

Hidden width is $1/8$ of the backbone hidden size. Total: **664,129 parameters**. The layers used are intermediate-to-late: `[15, 20, 25]` for Llama-3-8B, `[12, 16, 20]` for Gemma-2-2B.

One scalar $g_i$ is shared across *all* adapted modules and *all* decoding steps. It is computed once, in a prompt-only prefill pass with the adapter disabled, then held fixed. No per-token, per-layer, or per-head routing.

**The adapter.** LoRA rank $r=8$, scaling $\alpha=16$, dropout 0, injected into `q_proj` and `v_proj` only. Backbone frozen.

**Three losses, summed.**

1. *Subtype-weighted [[Cross Entropy|BCE]] on the gate.* Training data is WildJailbreak (261,559 examples; 128,781 safe, 132,778 unsafe) with four subtype labels: `vanilla_benign`, `adversarial_benign`, `vanilla_harmful`, `adversarial_harmful`.
$$L_{\mathrm{gate}} = \frac{1}{N}\sum_i w_{\mathrm{type}}(i)\,\mathrm{BCE}(g_i, y_i)$$
The two *adversarial* subtypes get bigger weights, because those are the confusing ones — a jailbreak that hides its intent, or a harmless question dressed up to look like an attack.

2. *Hard pairwise margin.* BCE alone gets the labels right but leaves the score *distributions* overlapping, which matters here because $g$ is a continuous knob, not a decision. So for every unsafe–safe pair $(u,s)$ inside a mini-batch:
$$L_{\mathrm{pair}}(u,s) = \max\{0,\; m - (g_u - g_s)\}$$
and the pairs are reweighted by a softmax over how *wrong* they currently look:
$$a_{u,s} = \operatorname{softmax}_{u,s}\big(\beta(g_s - g_u)\big), \qquad L_{\mathrm{hard}} = \sum_{u,s} a_{u,s} L_{\mathrm{pair}}(u,s).$$
A safe prompt that scored high, or an unsafe prompt that scored low, dominates the gradient. This is hard-negative mining, in the same spirit as contrastive training.

3. *Unsafe-only language modelling.* The adapter sees the next-token loss **only on unsafe examples** with safe target responses (refusal, redirection, high-level safety guidance):
$$L_{\mathrm{LoRA}} = \frac{1}{|\mathcal{B}_{\mathrm{unsafe}}|}\sum_{i \in \mathcal{B}_{\mathrm{unsafe}}} \ell_i^{\mathrm{LM}} + \lambda_{L2}\|\Delta W_\theta\|_2^2.$$
This is the clean bit of the design: the adapter is *never* asked to reproduce good behaviour on benign prompts, because on benign prompts the gate is supposed to shut it off and let the frozen backbone do its job.

Total: $L = \lambda_{\mathrm{BCE}} L_{\mathrm{gate}} + \lambda_{\mathrm{pair}} L_{\mathrm{hard}} + L_{\mathrm{LoRA}}$, optimised with [[Decoupled Weight Decay Regularization (AdamW)|AdamW]].

**Training config.** 1 epoch, effective batch 64, LR $3\times10^{-4}$, warmup ratio 0.05, weight decay $10^{-4}$ on LoRA and **0 on the gate**. Single GH200. Llama-3-8B: 5h09m, 5.51M trainable params (vs 8.03B for SFT), 40.5 GB peak vs 92.3 GB for SFT.

Note that CLEAR trains *slower* than plain LoRA (5:09 vs 3:18) because of the extra prompt-only gate forward pass.

---

## Ablation Studies and Experiments

**Main table — Llama-3-8B-Instruct** (HB ASR = HarmBench attack success, lower better; Safe OR = over-refusal on harmless prompts, lower better):

| Method | HB ASR ↓ | Unsafe Ref ↑ | Safe OR ↓ | GSM8K ↑ | MMLU ↑ | TQA MC2 ↑ |
|---|---|---|---|---|---|---|
| Base | 32.25 | 88.50 | 2.80 | 75.06 | 66.76 | 52.47 |
| SFT | 2.00 | 94.00 | 10.80 | 66.34 | 63.86 | 48.28 |
| LoRA | 0.00 | 82.50 | 6.40 | 66.72 | 64.18 | 50.46 |
| **CLEAR** | 0.50 | 92.50 | 4.80 | **73.46** | 63.62 | **52.63** |

Always-on LoRA actually wins on raw ASR (0.00 vs 0.50). CLEAR wins on everything about *not breaking the model*. TruthfulQA MC2 is fully recovered (52.63 vs base 52.47) where SFT lost 4 points. MMLU is the one place CLEAR does *not* recover — 63.62, slightly below both SFT and LoRA. The authors do not explain this; my guess is MMLU prompts sometimes trip the gate.

**Gemma-2-2B-it:** ASR 9.50 → 0.00, unsafe refusal 48.50 → 96.00, GSM8K 46.47 → 41.62 (vs 38.06 SFT / 38.21 LoRA). But **safe over-refusal doubles, 4.80 → 9.60** — worse than both baselines. On the small model the gate fires on harmless prompts too often. This is the honest weak spot.

**Against real aligned models (Alpaca family).** Llama-3-8B-RR (representation rerouting) only gets ASR to 15.50%. SACPO 21.75%, P-SACPO 22.50%, Beaver-7B 24.00%. Stacking CLEAR on top of P-SACPO gets 0.50% while keeping GSM8K at 6.14% (P-SACPO base: 5.99%). So CLEAR composes with existing aligned checkpoints rather than replacing them.

**Gate quality vs external guards.** At threshold 0.5 on XSTest, the 664K-param internal gate beats external guardrails on the routing trade-off. Llama Guard 3 is 8.03B params (12,000× bigger). PromptGuard (279M) routes *nearly everything* as unsafe — but the authors are careful to say this is an objective mismatch, not stupidity: PromptGuard detects prompt *injection*, and XSTest's safe-but-spicy prompts light up the INJECTION head.

**Scaling (Qwen2.5-Instruct, 0.5B → 7B).** ASR ≤2% everywhere. Utility is basically untouched at 7B: GSM8K 82.18 → 82.18 exactly, MMLU 74.15 → 74.19. The gate diagnostics explain why:

| Size | Gate ROC-AUC | PR-AUC | Mean margin |
|---|---|---|---|
| 0.5B | 0.7795 | 0.7475 | 0.0875 |
| 1.5B | 0.8800 | 0.8286 | 0.0749 |
| 3B | 0.9155 | 0.8707 | 0.0927 |
| 7B | 0.9566 | 0.9363 | 0.2362 |

**Bigger backbones have more linearly separable harm signal in their hidden states.** The margin jumps 2.5× from 3B to 7B. This is the most useful finding in the paper: CLEAR gets *better* as models scale, because the free lunch it depends on (the model already internally knows the prompt is harmful) gets richer. On Qwen-0.5B the gate is barely better than a coin flip on the hard cases, and safe over-refusal is a terrible 51.60%.

### What did not work

- **BCE-only gate** (drop the hard pairwise margin, drop L2). ASR *rises* from 0.00 → 3.25%, unsafe refusal falls 96.00 → 83.50%, and safe over-refusal blows up to 14.40%. A plain classifier loss gives you correct labels with mushy, overlapping scores — and mushy scores mean half-activated adapters on inputs that need a decision. **The margin loss is doing real work, not decoration.**
- **Removing the LoRA $L_2$ term.** Small but consistent degradation: ASR 0.00 → 0.50, Safe OR 9.60 → 10.40, GSM8K 41.62 → 40.26. The [[Regularization|regulariser]] keeps $\Delta W$ small so a mid-range gate score does not swing the model too far.
- **Transformer gate instead of MLP.** Transformer-16 gives ASR 1.50% and GSM8K 33.13%; the multi-layer MLP gives 0.00% and 41.93%. The heavier gate is *worse*. Aggregated hidden states are already good features; you do not need attention over them.
- **Single-layer vs multi-layer gate input.** MLP-16 (one layer) → ASR 0.25%, GSM8K 34.42%. MLP-12/16/20 (three layers) → ASR 0.00%, GSM8K **41.93%**. Reading three depths is worth 7.5 GSM8K points. Different layers carry different parts of the harm signal.

**Latent adaptive attack (exploratory).** White-box PGD on hidden states, $\epsilon = 0.001$ in $\ell_\infty$, 10 steps, targeting the "Sure, here is" prefix, injected at embedding / middle / final layers, on 100 HarmBench behaviours. With a gate-adaptive PCA defence (project activations back onto a benign 128-dim subspace, projection strength $\gamma$ set by how anomalous the gate score is), CLEAR gets ASR 0.0 / 7.1 / 18.4%, versus LoRA 54.0 / 48.0 / 31.0% and SFT 47.0 / 35.0 / 22.0%. Caveat: this compares CLEAR-*with-a-defence* against undefended baselines, so it is not a clean ablation of routing itself.

---

## Worth Remembering

- **The failure mode is entirely in the gate, and the authors say so.** A benign prompt with a high score gets unnecessary refusal; a harmful prompt with a low score walks straight past a switched-off adapter. There is no fallback. Always-on LoRA at least fails gracefully; CLEAR fails open.
- The gate score is **computed once and frozen for the whole generation**. A prompt that looks benign but turns harmful mid-conversation, or a multi-turn escalation, is out of scope. Single-turn text only.
- **Zero inference-time weight merging.** Because $g$ varies per request, you cannot fold $\Delta W$ into $W$. You pay LoRA's runtime cost plus one extra prompt-only prefill for the gate. For serving, that is a real cost — think about whether it batches well alongside [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)|paged KV cache]] scheduling.
- Connection to [[Sparsely-Gated Mixture-of-Experts Layer|mixture-of-experts]]: same routing skeleton, but the router output is used as a **continuous scale**, not a top-$k$ selection, and there is exactly one expert. No load balancing loss needed, no capacity factor. It is MoE with the discreteness removed, which is why it trains stably end-to-end.
- The hard-pair softmax weighting $a_{u,s} = \mathrm{softmax}(\beta(g_s - g_u))$ is structurally the same idea as hard negative mining in [[Approximate Nearest Neighbor Negative Contrastive Learning (ANCE)|ANCE]] or [[Dense Passage Retrieval (DPR)|DPR]]: the informative training signal lives in the pairs the model currently gets backwards.
- **The scaling result is the transferable insight.** Harmfulness is close to linearly readable from mid-layer hidden states, and gets more readable with scale (ROC-AUC 0.78 → 0.96). This is consistent with the "refusal is mediated by a single direction" line of work. If you want a cheap safety classifier, you probably do not need a separate 8B guard model — you need three layers of your own activations and a 600K MLP.
- Open questions: (1) does the gate calibrate under distribution shift to jailbreak styles absent from WildJailbreak? Nothing here tests that. (2) Could you train the gate to output *calibrated* probability and set a deployment threshold per use case — chatbot vs coding assistant? (3) Two adapters, one for refusal and one for "answer but carefully", routed by a 2-way gate, would attack the over-refusal problem directly.
- Practical caveat: all backbones are ≤8B. The authors admit frontier-scale and multimodal behaviour is unverified. Given the scaling trend, it probably works *better* at 70B — but "probably" is doing work.

---

## Links
Related: [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Sparsely-Gated Mixture-of-Experts Layer]] · [[Training language models to follow instructions with human feedback]] · [[Direct Preference Optimization (DPO)]] · [[Constitutional AI- Harmlessness from AI Feedback]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Cross Entropy]] · [[QLoRA- Efficient Finetuning of Quantized LLMs]] · [[Regularization]] · [[Dense Passage Retrieval (DPR)]] · [[Approximate Nearest Neighbor Negative Contrastive Learning (ANCE)]] · [[Inject, Align, Recover- Staged Post-Training for Retrieval-Free Document Knowledge Internalization]]

New topics worth writing: Alignment tax and over-refusal, WildJailbreak dataset, HarmBench and XSTest evaluation protocols, Llama Guard and external guardrail models, Representation engineering / refusal directions, Projected gradient descent attacks on hidden states, Hard negative mining in margin losses, Conditional computation beyond MoE
