---
title: "LLMs Get Smarter from Targeted Synthetic Multilingual Data"
authors: ["Ishika Agarwal", "Arkajyoti Charaborty", "Tanner Sorensen", "Neha Gupta", "Andreas Stolcke"]
year: 2026
arxiv: "2608.15964"
url: https://arxiv.org/abs/2608.15964
priority: Low-Priority
read_on: 2026-10-01
tags: [paper, llm, rl]
---
## The Core Idea

An LLM answers the same question differently depending on which language you ask it in. Ask "who invented X" in English and you get one answer; ask in Arabic and you get another. This is called **language-specific competency** (LSC). The cause is not mysterious: pretraining data is overwhelmingly English, so the model internally routes non-English text through English-shaped representations in its middle layers, then routes back out to the target language at the end (Wendler et al., 2024). English becomes a ceiling on what the model can represent.

The field had two bad options:

1. **Route everything through English.** Prompt the model to "reason in English". Performance goes up, but you have given up on the other languages as thinking media.
2. **Balance the pretraining data.** Aya-23-8B did this across 23 languages. Performance became consistent across languages — and *lower everywhere*, including below Qwen's English score. This is the "Curse of Multilinguality": add languages, the average flattens out.

HOTFIXR takes a third route, and it is entirely about **data**. Instead of asking "what data should I train on?", it asks "what data would most expose *this specific student model's* multilingual cracks?" — and then it trains a separate model whose only job is to write those questions.

> [!NOTE] Lingual deficit ^lingual-deficit
> A score for a generated question, measuring how badly a *particular* student model handles it. It has two halves: the student is unsure of the answer even when reasoning in English (a general weakness), **and** the student's internal representation of its reasoning differs a lot depending on which language it reasoned in (a language-specific weakness).

The trick that makes this new: the second half of that score is not a task metric at all. It is a **cosine distance between two hidden states** — the model's last reasoning token when told to reason in English, versus when told to reason in Spanish. The answer to a question does not depend on which language you think in, so those two internal states *should* sit near each other. When they do not, you have found a misalignment. That is a cheap, label-free, task-agnostic signal, and it is what lets the question generator hunt for weaknesses without needing a benchmark to score against.

What it unlocks: a **one-time** 8-GPU-hour cost buys a question generator that can then produce unlimited targeted training data. Result: +6.2% average over training baselines in-distribution, while losing only 0.9% on held-out tasks (other fine-tuning baselines lose 4–9%), and only 1.4% on held-out *languages* (baselines lose 3–12.5%).

## The Methodology

Three models, all involved at different points:

- **Question generator** — the thing being trained with [[GRPO]]. Same base model family as the student.
- **Student** — the model you actually want to improve. Qwen2.5-7B, Qwen2.5-14B, or Llama-3.1-8B.
- **Label generator** — Qwen2.5-32B-Instruct, used once, to answer the generated questions.

### The loop

1. The question generator is given a real Nemotron sample (question + reasoning + answer) as an in-context difficulty calibrator, plus a target language, and asked to write a *new, original* question of comparable difficulty. It emits a `<question>/<reasoning>/<answer>` triple. The prompts are in the paper's Appendix E and are worth copying — they explicitly say "do NOT copy, paraphrase, or reuse".
2. The student is shown that question, twice. Once prompted "answer the following question, reason using English". Once prompted "answer the following question, reason using $L$", where $L$ cycles through French, Spanish, Arabic, Portuguese, Italian.
3. The lingual deficit score is computed from those two runs.
4. That score is the GRPO reward for the question generator.

### The reward, exactly

**Language-agnostic incompetency (LAI)** — is the student unsure, even in its best language? Take the English-reasoning response $R$ under prompt $P$:

$$U(P,R) = 1 - \frac{1}{|R|}\sum_t p_\theta\!\left(R_t^{(1)} \mid P, R_{<t}\right)$$

That is one minus the average probability of the top-1 token, averaged over the whole response. High $U$ = the student was hedging the whole way through. Reward the generator for high $U$.

**Language-specific incompetency (LSI)** — do the two reasoning paths end up in different places internally? Find the last reasoning token in $R_{EN}$ and in $R_L$ (the token right before the answer delimiter), pull the final-layer hidden state for each, and take the cosine distance. Reward the generator for *large* distance.

Plus a **format reward** for emitting the three tags correctly. All three are needed — see the ablation below.

### Training numbers that mattered

- $|\mathcal{D}_{QG}| = 500$ in-context seed samples (167 each from Nemotron-PTDv2 STEM, MATH, CHAT). **Deliberately tiny.**
- ~41 GRPO steps. **Deliberately few.**
- Both limits exist to stop [[Reward Hacking|reward hacking]]. With more data or more steps, the generator finds one question sitting on a local maximum of the lingual-deficit score and emits *that same question forever*, regardless of the in-context example. You then get 5,000 identical rows and the student fails catastrophically.
- 6× A100, 0.96 min per training sample → ~8 GPU-hours total, one time.

### Then

The trained generator produces $\mathcal{D}_S$ = 5,000 samples, balanced across the six in-distribution languages. Qwen-32B writes the labels. The student is fine-tuned on $\mathcal{D}_S$ with plain SFT. No contrastive term, no architecture change, no [[KL Divergence|KL]] penalty — all the cleverness lives upstream in the data.

> [!NOTE] Why a bigger model writes the labels ^label-generator-32b
> The authors tried having each student label its own generated questions. It failed for a boring reason: the 7B/8B students could not reliably *follow the instruction* to answer in the prompted language. Instruction-following, not knowledge, was the bottleneck. So this method currently needs access to a stronger model for labels.

## Ablation Studies and Experiments

### Setup

Six in-distribution languages (English, French, Spanish, Arabic, Portuguese, Italian). Out-of-distribution languages: German + Japanese for factual/translation, Russian + Chinese for RAG.

Four task families:

| Task | Benchmark | Metric | ID/OOD |
|---|---|---|---|
| Agentic reasoning | Nemotron STEM/MATH/CHAT | Accuracy / LLM-judge | ID |
| Factual knowledge | MMMLU | Accuracy | OOD |
| RAG comprehension | mHotPotQA | ROUGE-L | OOD |
| Translation | OPUS-100 | LLM-judge (Prometheus-7B-v2) | OOD |

Everything is discretised to binary correct/incorrect (≥80% ROUGE, or ≥4/5 judge score) and reported as % correct, so numbers are comparable across tasks. All results averaged over 3 runs.

Seven baselines, grouped by what they isolate:

- `Base`, `EngReason` — no training. `EngReason` is the "just reason in English" strategy.
- `SelectionGT` — 5,000 real Nemotron samples with real labels. Tests: is synthesis better than just using the data you have?
- `SelectionGEN` — same 5,000 questions, labels from Qwen-32B. Tests: how much do generated labels cost you?
- `Filtered` — score 10,000 real samples with the lingual deficit metric, keep the top 5,000. Tests: is the *score* enough, or do you need generation?
- `Untrained` — use the base model as question generator, no GRPO. Tests: is the GRPO step earning its keep?
- `DataEnvGym` — prior art, generates data from student mistakes on a specific dataset, adapted here with a language instruction.

### Headline numbers

HOTFIXR vs each baseline, averaged over 9 settings (3 students × 3 task groups):

| Baseline | ID $\Delta$ | ID wins | OOD $\Delta$ | OOD wins |
|---|---|---|---|---|
| Base | +4.3 | 9/9 | **−0.9** | 3/9 |
| EngReason | +5.4 | 9/9 | −0.7 | 4/9 |
| SelectionGT | +7.0 | 9/9 | +3.8 | 9/9 |
| SelectionGEN | +6.0 | 9/9 | +8.9 | 8/9 |
| Filtered | +7.3 | 9/9 | +6.6 | 8/9 |
| Untrained | +5.8 | 9/9 | +1.4 | 7/9 |
| DataEnvGym | +7.1 | 9/9 | +7.3 | 9/9 |
| **Average** | **+6.2** | **9/9** | **+3.7** | **7/9** |

Read the OOD column carefully. HOTFIXR *still loses* 0.9% against the untouched base model on held-out tasks — fine-tuning on 5,000 samples always costs you something. The claim is narrower and more honest: among methods that train, it forgets least, by 5.6% on average.

### The negative results, which are the interesting half

**`Filtered` is barely better than random selection.** Scoring 10,000 existing samples by lingual deficit and keeping the top half gives 48.9 ID — *worse than the base model's 51.9*. Meanwhile generating from a trained generator gives 56.2. So the lingual-deficit score is not, on its own, a useful data filter. Its value is as a **reward for search**, not a ranking function. Existing data simply does not contain the questions that maximally expose a given student.

**`DataEnvGym` is the worst OOD performer** (57.4 vs base 65.6). It targets student mistakes on a *specific dataset*, so it overfits to that dataset's shape. HOTFIXR's reward never looks at a dataset — uncertainty and hidden-state distance are properties of the student alone.

**`EngReason` gives nothing.** 50.8 ID vs base 51.9, 65.4 OOD vs 65.6. The "just think in English" trick, measured properly here, does not help. Language spread does drop (8.8 vs 9.6), which is the one thing it buys.

**`SelectionGEN` collapses on RAG.** Qwen-7B: 79.0 → 41.4. Qwen-14B: 79.9 → 42.8. Generated labels on generic questions are actively destructive for reading comprehension. That is a −37 point hole and the paper does not dwell on it.

### The reward ablation — the cleanest result in the paper

Nemotron (ID) scores, all rows include the format reward:

| Model | Format | +LAI | +LSI | +LAI+LSI |
|---|---|---|---|---|
| Qwen 7B | 52.0 | 50.9 | 50.8 | **57.8** |
| Llama 8B | 48.2 | 48.5 | 47.5 | **52.5** |
| Qwen 14B | 49.3 | 50.9 | 48.8 | **58.3** |

Neither piece works alone. LAI alone is within noise. LSI alone is *worse* than format-only on every model. Together: +5 to +9 points. The interpretation: LAI alone finds questions the model is unsure about but which may be genuinely unanswerable or malformed; LSI alone finds questions where representations diverge but which may be trivially easy. The conjunction — hard *and* cross-lingually inconsistent — is the useful region.

### Is it just distillation from the 32B?

Appendix C answers this properly. `Distillation (32B)` uses Qwen-32B to write *both* questions and labels:

| Model | Distillation (32B) Nemotron | HOTFIXR Nemotron |
|---|---|---|
| Qwen 7B | 49.6 | 57.8 |
| Qwen 14B | 53.3 | 58.3 |
| Llama 8B | 47.7 | 52.5 |

A big model's generic questions get you ~49.6. A small, GRPO-trained generator aimed at the student's specific cracks gets you 57.8. The gain is in the **targeting**, not the teacher's knowledge.

### OOD language forgetting

Delta vs base, averaged over models:

| Method | De | Ja | Ru | Zh | Avg |
|---|---|---|---|---|---|
| SelectionGT | −3.3 | −4.7 | −9.5 | −10.5 | −7.0 |
| SelectionGEN | −1.9 | −4.0 | −22.5 | −21.5 | −12.5 |
| Filtered | −3.8 | −4.8 | −14.7 | −16.3 | −9.9 |
| Untrained | −2.9 | −2.0 | −2.3 | −5.3 | −3.1 |
| DataEnvGym | −6.8 | −6.0 | −12.2 | −15.4 | −10.1 |
| **HOTFIXR** | **−2.6** | **−0.7** | **−0.9** | **−1.3** | **−1.4** |

Russian and Chinese — the two OOD languages furthest from the six trained ones — are where every other method haemorrhages. SelectionGEN drops 22.5 points on Russian. HOTFIXR drops 0.9.

### The consistency claim is weak, and they say so

| Method | Range | Trimmed Range | Std | IQR | CV |
|---|---|---|---|---|---|
| Base | 20.2 | 16.0 | 7.2 | 9.6 | 0.12 |
| HOTFIXR | **19.4** | **14.6** | **6.9** | **9.4** | **0.11** |

Every dispersion measure improves — marginally. Std 7.2 → 6.9. The authors' own wording: "there is no SOTA data curation method for ensuring multilingual consistency." So HOTFIXR raises the average and holds the spread roughly flat. It does *not* solve LSC. It buys back some of the performance that consistency-focused methods give away.

### Continual learning (Appendix D)

Round 1: fresh question generator trained from the base model, but the student continues from its Round-0 checkpoint. Qwen-7B only. Gains: +7% Nemotron, +8% OOD tasks, +8% seen languages, +7% unseen. Round 2 helps more than Round 1 did — which suggests the generator finds *new* cracks once the old ones are patched, rather than saturating. Single model, single setting, so treat as **suggestive, not established**.

## Worth Remembering

**The reward is the contribution.** Strip everything else and the reusable idea is: *the cosine distance between a model's hidden states when it reasons in two languages is a trainable, label-free signal for cross-lingual misalignment.* That generalises well past this paper. Anywhere you have two paths that should converge on the same internal state, the gap between them is a reward.

**Reward hacking is the binding constraint, not compute.** They capped the seed set at 500 and GRPO at ~41 steps not for budget reasons but because the generator otherwise degenerates to emitting one single question forever. This is a sharp, concrete instance of the [[Reward Hacking|optimiser-as-adversary]] problem: the reward has a local maximum, and GRPO finds it. Anyone reproducing this should log **output diversity** of the generator, not just reward.

**Limitations the authors admit:**
- All high-resource languages. Arabic is the hardest one tested. Low-resource languages are future work.
- Verifiable tasks only — clear right/wrong answers. No claim about open-ended generation.
- Needs a stronger model for labels, because small students cannot follow "answer in language $L$" reliably.
- Question generator and student are instantiations of the *same* model family throughout. Cross-family transfer untested.

**Practical caveats if you wanted to use this:**
- You need access to the student's **hidden states**, not just its logits. No API-only version of this exists.
- You need to identify "the last reasoning token" — i.e. the model must have a clean answer delimiter. Fiddly for models without structured reasoning output.
- The 8 GPU-hours is the generator training. You still pay for 5,000 label generations from a 32B model, plus SFT.
- Fine-tuning on 5,000 samples costs ~1% on held-out tasks *no matter what you do*. Budget for it.

**Surprising bits.** `Filtered` performing *below base* is the most instructive number here — the score works as a reward and fails as a filter. And `EngReason` being flat-to-negative on ID undercuts a widely-repeated multilingual prompting heuristic, at least for Qwen and Llama at this scale.

**Open questions.** Does the hidden-state distance shrink after training, or does the student just get better at the questions while the misalignment persists? The paper never measures its own LSI metric post-training, which would be the direct test of whether it fixes representations or just patches behaviour. Also: the consistency numbers barely move, so which is it?

## Links

Related: [[GRPO]] · [[Reward Hacking]] · [[Fine-Tuning]] · [[Distillation]] · [[Instruction Tuning]] · [[Embeddings]] · [[Uncertainty]] · [[Evals]] · [[Perplexity]] · [[Chain of Thought]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[What Does Privileged Information Add to On-Policy Self-Distillation]] · [[Thinking in a Low-Resource Language- What SFT Builds, What RL Fixes, What Accuracy Cannot See]] · [[Selecting Diverse SFT Traces Improves Post-RL Generalization]] · [[Continual Learning Mechanisms Compose for Long-Horizon Memorization]] · [[Dynamic Important Example Mining for Reinforcement Finetuning]] · [[Understanding Contrastive Learning through Alignment and Uniformity]] · [[Distilling the Knowledge in a Neural Network]] · [[Exploring the Limits of Transfer Learning (T5)]]

New topics worth writing: Language-specific competency (LSC), Curse of Multilinguality, latent English anchoring in multilingual transformers, acquisition functions for synthetic data generation, hidden-state cosine distance as a training signal, catastrophic forgetting from SFT on synthetic data, MMMLU, OPUS-100, multilingual HotPotQA, Prometheus LLM-judge models, cross-lingual representation alignment, multilingual data curation and filtering pipelines
