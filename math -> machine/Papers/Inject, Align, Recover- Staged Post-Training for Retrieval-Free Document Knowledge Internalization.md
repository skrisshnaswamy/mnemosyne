---
title: "Inject, Align, Recover: Staged Post-Training for Retrieval-Free Document Knowledge Internalization"
authors: ["Qian Kou", "Xiaofeng Shi", "Xiaosong Qiu", "Hua Zhou"]
year: 2026
arxiv: "2608.20281"
url: https://arxiv.org/abs/2608.20281
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, llm, vision, theory]
---
## The Core Idea

Normal practice for "answer questions about my documents" is retrieval: find the right passage, paste it into the prompt, let the model read it. This paper asks the opposite question. Suppose retrieval is not allowed — too slow, too private, or you simply want to test whether training actually changed the model's memory. Can you bake a fixed pile of documents *into the weights* so the model answers from memory alone?

They call this **document knowledge internalization**, and the honest finding is that it is a two-objective problem, not one. You can always push domain accuracy up by fine-tuning hard on document-derived question–answer pairs. But the model gets dumber at everything else. Llama-3.2-3B fine-tuned on the CC corpus goes from 11.2% → 35.5% domain accuracy, while MMLU collapses from 50.8% → 11.2%. That is not a deployable model. It answers your document questions and has forgotten how to be an assistant.

> [!NOTE] Document knowledge internalization ^internalization
> Turning a bounded, fixed document collection into *parametric* knowledge — knowledge stored in the weights — so that held-out questions can be answered with **no source text in the prompt at inference time**.

The contribution is a clean split of the problem into three separately-measurable jobs, called **IAR**:

1. **Inject** — show the model the documents densely, but as supervised instruction→document tasks, not as raw next-token soup.
2. **Align** — teach it to expose that knowledge through a question-answer interface.
3. **Recover** — after training, *average the damaged model back towards the original instruct model in weight space* to claw back general ability.

None of the three pieces is new on its own. Weight merging existed. Reconstruction objectives existed. What did not exist was the controlled decomposition: measuring how much each stage contributes, and treating "which merged checkpoint do I ship" as an explicit, rule-based choice on a domain-vs-general frontier rather than a hidden hyperparameter. The authors are unusually blunt that IAR is "a decomposition of adaptation budgets, not a recipe that dominates every baseline."

## The Methodology

**Setup.** You have documents $D$. From them a generator produces training QA pairs $Q_{\text{train}}$ and held-out test pairs $Q_{\text{test}}$. Start from an already instruction-tuned model $M_0$. At test time the model sees **only the question** — no document, no retrieved passage.

Two corpora: **CC** (from Common Corpus, English-ish, 14,258 train / 750 test QA) and **CCI** (Chinese, 10,926 train / 575 test). Four model families at 3–4B: Llama-3.2-3B, Phi-4-mini, Qwen3-4B, SmolLM3-3B. Plus Qwen3-8B/14B/32B on CC only.

### Stage 1 — Inject

Three ways of turning a document into an (instruction, target) pair, all trained with ordinary [[Cross Entropy]] on the **assistant target only** (system and user tokens are masked out of the loss):

- **Continuation** — user prompt = instruction + first half of document; target = the rest.
- **Rewrite** — user prompt = a generated summary, outline, or "knowledge skeleton"; target = the full cleaned document. The model has to expand a compressed sketch back into the original.
- **Instruction-formatted reconstruction** — user prompt = a short generic reading instruction; target = the full cleaned document. Dense exposure, nothing given away.

The mixed objective is a plain sampling mixture:

$$\mathcal{L}_{\mathrm{inj}}=\sum_{m\in\mathcal{M}}\pi_m\,\mathbb{E}_{(u,y)\sim\mathcal{D}_m}\big[\ell_\theta(u,y)\big],\qquad \ell_\theta(u,y)=-\frac{1}{|y|}\sum_{t=1}^{|y|}\log p_\theta(y_t\mid u,y_{<t})$$

Important detail: $\pi_m = n_m/\sum_k n_k$ is the *realised row share* after mixture sampling and length filtering, not a tunable loss weight. Recipes are written as ratios like `1:1:2` (continuation : rewrite : reconstruction), roughly 19,000 rows.

Contrast with plain continued pretraining, which has no prompt/target boundary at all:

$$\mathcal{L}_{\mathrm{CPT}}=-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})$$

The recipe labelled `1:0:0` (reconstruction only) is *not* CPT — it still uses a chat template and masks the prompt.

### Stage 2 — Align

Standard answer-only supervised fine-tuning on the document-derived QA pairs:

$$\mathcal{L}_{\mathrm{align}}=-\frac{1}{|a|}\sum_{t=1}^{|a|}\log p_\theta(a_t\mid q,a_{<t})$$

The only difference from the baseline is where you start:
$$\theta_{\mathrm{SFT}}=\mathrm{Align}(\theta_0),\qquad \theta_{\mathrm{IA}}=\mathrm{Align}(\theta_I)$$
$\theta_{\mathrm{IA}}$ (the "IA checkpoint") is what goes into stage 3.

### Stage 3 — Recover

Take the **task vector** — the difference between the adapted model and the original — and only walk part of the way along it:

$$\Delta=\theta_{\mathrm{IA}}-\theta_0,\qquad \theta_R=\theta_0+\lambda\Delta$$

More generally they run a **fixed 12-candidate grid** of merge operators (via `mergekit`), decided *before* looking at any validation number:

| Operator | Grid |
|---|---|
| SLERP (spherical interpolation) | $t\in\{0.2,0.3,0.4\}$ |
| Task arithmetic | $w\in\{0.3,0.5,0.7\}$ |
| TIES (drop small entries, resolve sign conflicts) | density $d\in\{0.3,0.5,0.7\}$ |
| DARE (random drop + rescale) | drop rate $d_r\in\{0.1,0.3,0.5\}$ |

> [!NOTE] Task vector ^task-vector
> $\Delta = \theta_{\text{finetuned}} - \theta_{\text{base}}$. A direction in weight space that "contains" what fine-tuning taught. Scaling it by $\lambda<1$ gives you a dial between the two models.

**The selection rule** (run on a validation split only, before the held-out test is ever touched). Let $D(c)$ be domain accuracy and $G(c)=\frac{I+M+B}{3}$ the mean of IFEval, MMLU and MSBench. With tolerance $\tau=1.0$ percentage point and Vanilla SFT as reference $v$:

1. keep candidates with $D(c)\ge D(v)-\tau$;
2. keep those with $G(c)\ge G(v)$ **and** at least two of the three general metrics no more than $\tau$ below Vanilla SFT;
3. among non-dominated points in the $(D,G)$ plane, domain accuracy is the primary key; anything within $\tau$ of the best domain score is one "tier";
4. inside that tier pick the largest $G$; ties break on largest minimum general improvement, then smaller merge hyperparameter.

**Training hyperparameters that matter.** AdamW ([[Decoupled Weight Decay Regularization (AdamW)]]), LR $5\times10^{-5}$, cosine schedule with 5% warmup, weight decay 0.01, grad clip 1.0, BF16 ([[Mixed Precision training]]), max length 4096, per-GPU batch 1 × 8 accumulation × 8 A100-40GB = effective batch 64, DeepSpeed ZeRO-2 no offload ([[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models]]). 3 epochs for Inject, 3 for Align. Single run per checkpoint, no seed averaging.

**Evaluation.** Domain answers are graded by an adaptive LLM judge panel: two judges score $\{0, 0.5, 1\}$, a third is called only on disagreement, final score is the median. Over 242,255 judged answers: exact agreement 0.707, binary agreement 0.848, Cohen's $\kappa=0.691$, third judge fired 29.7% of the time. Domain accuracy = fraction scored correct or partially correct. Guardrails are IFEval, MMLU, and a fixed 200-example MSBench subset.

## Ablation Studies and Experiments

### Main table (RQ1)

| Model | Method | CC Dom. | CC IFEval | CC MMLU | CC MSB. | CCI Dom. | CCI IFEval | CCI MMLU | CCI MSB. |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-3B | Base Instruct | 11.2 | 77.1 | 50.8 | 54.5 | 29.2 | 77.1 | 50.8 | 54.5 |
| | Vanilla SFT | 35.5 | 54.2 | 11.2 | 21.5 | 53.0 | 61.2 | 22.5 | 31.5 |
| | CPT+SFT | 38.3 | 26.4 | 3.7 | 13.5 | 53.7 | 24.2 | 4.3 | 17.5 |
| | **IAR** | 36.5 | 60.2 | 35.0 | 30.5 | 55.3 | 61.3 | 33.2 | 36.5 |
| Qwen3-4B | Base Instruct | 34.3 | 84.8 | 65.8 | 77.0 | 70.6 | 84.8 | 65.8 | 77.0 |
| | Vanilla SFT | 42.4 | 51.1 | 8.8 | 51.0 | 75.1 | 45.6 | 26.3 | 49.5 |
| | CPT+SFT | 49.6 | 31.8 | 18.8 | 60.5 | 69.0 | 29.6 | 12.8 | 58.0 |
| | **IAR** | **50.5** | 59.8 | 19.5 | 63.0 | **76.3** | 76.1 | 64.5 | 70.0 |
| Phi-4-mini | Vanilla SFT | 24.4 | 47.8 | 51.0 | 32.5 | 40.2 | 47.8 | 53.8 | 31.5 |
| | **IAR** | 34.1 | 49.0 | 57.0 | 43.0 | 39.7 | 51.6 | 50.2 | 44.0 |
| SmolLM3-3B | Vanilla SFT | 32.1 | 35.6 | 10.5 | 25.0 | 52.3 | 41.7 | 16.7 | 31.5 |
| | **IAR** | 37.5 | 40.3 | 25.7 | 29.0 | 53.9 | 57.4 | 46.8 | 47.0 |

IAR beats Vanilla SFT on all four metrics in 7 of 8 settings. Headline averages: +3.6 pp domain, +12.1 pp mean general. The Qwen3-4B CCI row is the most dramatic recovery — MMLU 26.3 → 64.5 while domain still goes up.

**CPT+SFT is a disaster on general ability.** Llama CC: IFEval 26.4, MMLU 3.7. Raw next-token training on documents wrecks instruction following. Caveat the authors flag themselves: those runs start from the *Base* checkpoint, not Instruct, so it is not a clean isolation of the objective.

### The confound they took seriously: is it just more tokens? (RQ2)

**BudgetMatch** repeats plain QA-only SFT for as many epochs as needed to match the token count of Inject+Align — 14, 17, 21 and 11 epochs for CC-Llama, CC-Qwen, CCI-Llama, CCI-Qwen respectively (matched to within 0.1–2.8% of realised non-padding tokens, e.g. 60.280M vs 58.653M for CC Llama).

This is a *strong* control and it partly works. On CC it lifts domain accuracy by +4.9 (Llama) and +4.4 (Qwen) over 3-epoch Vanilla SFT. On CCI it does nothing (Llama unchanged at 53.0) or hurts (Qwen −2.9). So more tokens explains some of the CC gain but does not transfer.

IAR beats BudgetMatch on 14 of 16 metrics. The two losses: **CC Llama domain accuracy** (36.5 vs 40.4 — a real trade, giving up 3.9 domain points for +11.0 mean general) and **CC Qwen MMLU** (19.5 vs 31.8).

### Does Inject help *before* merging? (RQ3)

Yes, in all eight settings, though unevenly. Best IA over Vanilla SFT, in domain points:

- CC: Llama +2.8, Phi +7.7, Qwen +5.3, SmolLM +4.7
- CCI: Llama +5.6, Phi +6.1, Qwen **+0.4**, SmolLM +2.3

**No Inject recipe wins everywhere.** Full grids (pre-recovery domain accuracy):

| CC | Vanilla | 1:1:1 | Continue | 1:0:0 | Rewrite | 1:1:2 |
|---|---|---|---|---|---|---|
| Llama | 35.5 | 36.1 | 35.1 | 36.8 | 34.4 | **38.3** |
| Phi | 24.4 | 27.5 | 27.2 | 29.9 | 27.6 | **32.1** |
| Qwen3-4B | 42.4 | **47.7** | 40.9 | 44.8 | 43.5 | 46.9 |
| SmolLM3 | 32.1 | **36.8** | 32.7 | 33.6 | 31.6 | 36.1 |

On CCI, `1:1:2` wins for Llama/Phi/SmolLM but reconstruction-only `1:0:0` (75.5) edges out mixtures for Qwen. **Continuation-only is the weakest single objective almost everywhere**, and rewrite-only is often *worse than Vanilla SFT* (CCI Phi: 37.7 vs 40.2). The mixture is doing the work, not any one task.

### Scaling within Qwen3 on CC (RQ4)

| Model | Method | Domain | IFEval | MMLU | MSBench |
|---|---|---|---|---|---|
| 8B | Vanilla SFT | 48.7 | 56.4 | 14.0 | 52.5 |
| | Best IA | **57.5** | 50.6 | 18.5 | 48.5 |
| | IAR (TIES $d{=}0.3$) | 56.8 | 62.2 | 26.7 | 73.5 |
| 14B | Vanilla SFT | 54.8 | 62.9 | 54.5 | 57.0 |
| | Best IA | **60.5** | 53.5 | 40.3 | 42.0 |
| | IAR (TIES $d{=}0.3$) | 59.6 | 67.5 | 67.2 | 73.5 |
| 32B | Vanilla SFT | 56.4 | 58.5 | 44.0 | 56.5 |
| | Best IA | **63.9** | 53.0 | 63.0 | 44.5 |
| | IAR (TIES $d{=}0.3$) | 62.8 | 67.0 | 74.5 | 72.5 |

This is the cleanest picture in the paper. Merging costs only **0.7 / 0.9 / 1.1** domain points and buys **14.9–24.1** points of mean general performance, gained on *every* benchmark, not one metric compensating. And the merge is not just undoing the training: IA adds +8.8/+5.7/+7.5 domain over Vanilla SFT, and after merging IAR still keeps +8.1/+4.8/+6.4. Adaptation and recovery are genuinely separable.

### What did not work

- **[[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]] under-internalises badly.** On CC it keeps general ability beautifully (Llama MMLU 53.2 vs base 50.8) but domain accuracy *drops below the pre-training baseline* — Llama 22.5, SmolLM3 15.5 vs Vanilla SFT's 32.1. Rank-16 adapters are not enough capacity to memorise a corpus. If your goal is stuffing facts into weights, parameter-efficient tuning is the wrong tool.
- **FAPM (post-hoc pruning of the task vector, sparsity 0.9)** is the best *general* recovery method in the paper — Qwen3-4B CC hits IFEval 83.8 / MMLU 61.3 / MSBench 83.5, nearly the untouched instruct model. But it throws the documents away: Llama CC domain drops 35.5 → 22.3, CCI 53.0 → 38.6. Keeping only the top 10% of task-vector entries deletes exactly the knowledge you paid to install. Interestingly, on Qwen3-4B CC it *does* work (IA-FAPM 44.1 domain, beating Vanilla SFT's 42.4) — model-dependent.
- **Rewrite-only Inject** is often net negative versus doing nothing.
- **Context-aware SFT** on Qwen3-4B CCI scores 65.0, well below the 70.6% untrained base — training with context present actively hurts when context is absent at test time.
- **Replay** (25% of QA rows *replaced* by general instructions) helps general metrics somewhat but is inconsistent on domain (CCI Qwen: 66.4, far below 75.1).
- **The Qwen3-4B CCI boundary case.** The untrained instruct model already scores 70.6% — much higher than Llama (29.2), Phi (27.3), SmolLM (34.1). Almost no headroom. The authors ran a nice diagnostic: score every model on the *same* CCI source-document continuations using **conditional bits per byte** (tokenizer-independent, unlike perplexity), $\mathrm{BPB}=\frac{\sum_t -\log p(x_t\mid x_{<t})}{\ln 2 \cdot B}$. Qwen3-4B gets 0.744 vs Llama 1.021, Phi 0.968, SmolLM 0.838 — lower on 100% of documents vs Llama, 86.8% vs SmolLM. They explicitly refuse to call this contamination; it is "consistent with a stronger fit to the evaluated text distribution", nothing more. Also honest: Phi has lower BPB than Llama but *worse* CCI QA, so likelihood does not straightforwardly cause QA ability.
- **Qwen3-1.7B on CCI gets worse from any training**: base 60.0 → Vanilla SFT 57.0 → Inject+Align 56.7. Sometimes the prior is already the best you have.

## Worth Remembering

**The honest framing is the point.** This paper reads like a careful engineering report, not a victory lap. Section after section says "this does not support a universal best recipe", "this is a repeated within-family pattern rather than a scaling law", "coverage is limited to completed archived runs and is not extrapolated". Compare [[On the Difficulty of Evaluating Baselines]] and [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] — this is what a paper looks like when the authors have internalised that lesson.

**The practical takeaway if you actually want to do this.** Merging back towards the base instruct model is absurdly cheap and gets you most of your general ability back for ~1 point of domain accuracy. TIES with density 0.3 was selected in 6 of 8 main settings and all 3 scaling settings. If you fine-tune an assistant on a narrow corpus and it becomes stupid, try $\theta_0 + \lambda(\theta_{\text{ft}}-\theta_0)$ before you try anything clever.

**Limitations they admit.**
- **Single seed per checkpoint.** Every reported number is one training run. Seeds are fixed (42) but the domain generation uses temperature 0.7 with *no sampler seed*, and judge APIs get no seed either. They say plainly: fixed training seeds "should not be read as repeated-run control". Bootstrap intervals (2,000 resamples) cover *evaluation-sample* noise only.
- **LLM-as-judge is the primary metric.** $\kappa=0.691$ is moderate-to-good, not gold. Per-judge means differ a lot: gpt-oss-120b gives $P(s{=}1)=0.321$, deepseek-v3.2 gives 0.127. They store raw votes for audit, which is the right move, but domain accuracy is a soft number.
- **The scaling runs have missing provenance.** For Qwen3-8B/14B/32B the Inject/Align training args, logs and node manifests are gone from the archive. They refuse to infer them from current defaults. Admirable, and also a hole.
- **Recovery is partial, not complete.** Selected checkpoints keep 15.6–19.2 domain points over the base instruct model, but never return to its general scores.
- **CPT+SFT baseline is initialisation-confounded** (Base vs Instruct start). The three Instruct-initialised diagnostics they do have are mixed: CC Llama 38.7 (beats Best IA's 38.3) but CCI Llama 52.9 (loses to 58.6).

**Follow-up questions.** Does the Inject mixture ratio matter more than the fact of mixing at all? Would a learned merge coefficient per layer beat a single global $\lambda$? The QA-generation pipeline itself (anchor extraction → type selection → question → validation → answer, with a fail-open validator) is a whole sub-system and only 44.6% of CC candidate QA pairs survive to the dataset — how much does IAR's ranking depend on that generator?

**Connections.** The Recover stage is [[Distillation|distillation]]-adjacent in spirit but operates purely in weight space with no forward passes. The catastrophic-forgetting trade-off here is the same shape as the alignment tax in [[Training language models to follow instructions with human feedback]]. The token-budget control is the right instinct from [[Scaling Laws for Neural Language Models]] and [[Training Compute-Optimal Large Language Models (Chinchilla)]].

## Links

Related: [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Training language models to follow instructions with human feedback]] · [[Cross Entropy]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[Scaling Laws for Neural Language Models]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models]] · [[Mixed Precision training]] · [[Distillation]] · [[On the Difficulty of Evaluating Baselines]] · [[Foundation Models]]

New topics worth writing: Retrieval-Augmented Generation (RAG), Catastrophic forgetting and Elastic Weight Consolidation, Model merging in weight space (task arithmetic, TIES, DARE, SLERP), Model soups, LLM-as-a-judge evaluation and inter-judge agreement, Continued pretraining / domain-adaptive pretraining, Knowledge editing (ROME, MEND), Bits-per-byte as a tokenizer-independent likelihood metric, IFEval, MMLU
