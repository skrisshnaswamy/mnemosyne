---
title: "Using Grounded Theory for Agent Behavior Analysis at Scale"
authors: ["Zhuoran Lu", "Yangyang Yu", "Zhuoyan Li", "Yibo Meng", "Nan Jiang", "Chengxi Zang", "Jie Gao", "Ziang Xiao"]
year: 2026
arxiv: "2608.30391"
url: https://arxiv.org/abs/2608.30391
priority: Good-To-Read
read_on: 2026-09-16
tags: [paper, llm, vision]
---
## The Core Idea

We can tell when an agent failed. We almost never know *what it did* on the way to failing. Benchmarks like SWE-bench report a pass/fail bit; logs report step counts and token counts. Neither answers "what kind of behaviour was that?"

The two usual escapes both break. Reading trajectories by hand is honest but a 200-step software-engineering trace takes a person maybe eight minutes, so 2,000 traces is a month of work. Writing a fixed classifier for a list of failure modes only works if you already knew the list — and on a new task or a new agent you do not.

The move here is to borrow a method from sociology rather than invent a new one. **Grounded theory** is a 60-year-old qualitative technique for building categories *out of* data instead of testing categories *against* it. Its appeal for agent analysis is three specific properties:

1. It is inductive, so it can name behaviours nobody has a word for yet.
2. It has a built-in stopping rule — **theoretical saturation**, meaning "keep sampling more data until new data stops producing new categories".
3. Every category stays chained to the exact quotes that produced it, so a claim can be audited back to raw text.

> [!NOTE] Grounded theory
> A qualitative method where you label concrete incidents in raw data, group the labels into categories, and keep refining the categories against freshly sampled data until nothing new appears. The theory is "grounded" because every claim points back at incidents. ^grounded-theory

**AutoTraceGT** is the automation of that loop: four LLM agents that read agent trajectories, invent labels, merge labels into a codebook, and stop when the codebook stops growing. The output is a task-specific taxonomy of agent behaviour, plus a narrative theory of why runs fail, plus — usefully — a binary feature vector you can feed a classifier.

What it unlocks: the same analytic rigour a human qualitative researcher would apply to 150 traces, applied to 7,500, for about a dollar per corpus.

## The Methodology

Four role-specialised agents at three scales.

**`OpenCode` — one trajectory at a time.** Maps a trajectory to a set of records $(\text{code}, \text{span}, \text{quote})$. The code is a 2–5 word conceptual label in verb+noun form: *"persists without adjusting strategy"*, *"re-examines same ground without advancing"*. The span says which steps. The quote is verbatim evidence.

The prompt design is doing a lot of work here. It bans descriptive captions ("if your code could serve as a caption for the log excerpt, it is too descriptive") and it bans a vocabulary list — *hallucination, context window, token, prompt, LLM, model, agent* — to stop the coder from importing received wisdom about how agents fail instead of reading what is in front of it. Long trajectories are split into 50-message chunks; a "segment memo" is carried forward between chunks so analytic continuity survives the context limit. Parallel across trajectories, sequential within one.

**`AxialCode` — one batch (B = 30 trajectories) at a time.** Groups the round's open codes into categories plus typed relations between them. Each category carries a definition, its member code records, and a status tag $\sigma \in \{\textsc{succ}, \textsc{fail}, \textsc{both}\}$ aggregated from the outcome of the source trajectories. Categories with a single member get flagged `thin`. Codes that fit nowhere go to an explicit `UNDERDEVELOPED` list rather than being forced into a bucket.

**`Manage` — across rounds.** This is the engine. It holds a versioned codebook $K_t = (\mathcal{C}_t, \mathcal{R}_t, L_t)$ where $L_t$ is a revision log. For each new category it picks one of $\{\texttt{add}, \texttt{merge}, \texttt{split}, \texttt{flag}\}$; for each new relation one of $\{\texttt{confirm}, \texttt{extend}, \texttt{add}, \texttt{contradict}\}$. Policy is **merge-first**: align to existing structure whenever a compatible match exists.

Count the new categories added in round $t$:

$$a_t = |\{\ell \in L_t \setminus L_{t-1} : \ell.\text{action} = \texttt{add}\}|$$

Stop when $a_t < \epsilon$ for $W$ consecutive rounds. They use $\epsilon = 0.2$ (as a rate) and $W = 2$.

> [!NOTE] Theoretical saturation
> The point at which sampling more data no longer yields new categories. Operationalised here as the add-rate falling below a threshold for two rounds running. This is the paper's main methodological upgrade over earlier LLM qualitative pipelines, which just stopped after a fixed number of iterations. ^theoretical-saturation

**`TheoreticalCode` — once, at the end.** Takes only $K_T$ and $L_T$ (no new raw evidence) and produces $(c^*, \mathcal{N})$: a *core category* and a narrative that arranges every other category around it as precondition, consequence, modifier, interrupter, or alternative path. It is also asked to output unresolved tensions rather than paper over them.

**Data.** Six corpora. Three with outcome labels only: Tau-Bench (1,980 traces, 1,183 success), Go-Browse (2,000, 728 success), SWE-Agent (2,000, only 167 success). Three with expert-written failure reasoning from prior work: ALFWorld (100), GAIA (50), WebShop (50).

**Backbones.** GPT-4.1-mini, GPT-5, GPT-5-mini, GPT-OSS-120B. Temperature 1 throughout. Sampling policy: plain uniform random over the remaining pool.

**Downstream use.** The codebook is flipped from inductive to deductive. Each trajectory becomes a fixed-length binary vector: one *presence* bit per category, one *co-occurrence* bit per confirmed axial relation. That vector goes either into an interpretive GLM (failure = 1) or into FLAML AutoML for failure prediction. For the prediction task, every trajectory is truncated to its **first 50% of steps**, the codebook is induced on the training split only, and test trajectories never touch a coding agent — a fairly careful anti-leakage setup.

## Ablation Studies and Experiments

**Does it actually saturate?** Four diagnostics over normalised coding progress: the `add` fraction tapers, the `merge`/`confirm` fraction rises, category count plateaus, and cosine similarity to the terminal codebook approaches 1. The authors are honest that panel (d) approaches 1 *by construction* — it is measured against the endpoint — so the shape of the curve, not its endpoint, is the evidence.

**Is it reproducible?** Three runs per dataset–model cell (12 cells), each on a *disjoint* trajectory subset. In-cell codebook cosine similarity: median 0.929, mean 0.923. The 594 cross-cell pairs: median 0.791, 95th percentile 0.901. So 0.901 becomes a data-derived reproducibility floor, and all 12 cell means clear it. Mann–Whitney $U$ gives $p < 10^{-20}$.

**Is it reading the data or the model's priors?** Fixing the dataset and varying the backbone LLM, cross-LLM coverage beats three nulls (cross-dataset any-model, cross-dataset same-model, and a 1,000-shuffle dataset-label permutation), $p < 0.001$ every time.

**Coverage against human taxonomies** (GPT-5-mini codebook, GPT-5 as judge, human taxonomy from prior expert coding of ~500 traces per benchmark):

| Dataset | $|\mathcal{H}|$ | $|\mathcal{C}|$ | Recall | Precision | Match |
|---|---|---|---|---|---|
| ALFWorld | 15 | 20 | 75.0 | 60.0 | 82.4 |
| GAIA | 11 | 17 | 73.7 | 63.2 | 58.0 |
| WebShop | 10 | 13 | 90.9 | 88.9 | 87.9 |

Recall exceeding precision is the interesting direction: the machine finds categories the humans did not enumerate. Three examples of what the human schema structurally cannot express:

- ALFWorld **noncompliant inaction** — producing *no* admissible action when one is required. The human labels `invalid_action` and `impossible_action` both presuppose an attempt was made.
- GAIA **signaled-but-unrealized shifts** — announcing a change of plan that never happens. The failure spans the boundary between planning and execution, so no single cognitive-module label fits.
- WebShop **mode oscillation without synthesis** — flipping between paging and resets without integrating feedback. Collapsed into a generic `inefficient_plan` bucket by the human taxonomy.

**Does iteration matter?** A one-pass baseline (same GPT-5-mini, code as many trajectories as fit in one context window). AutoTraceGT wins recall and match on all three: ALFWorld 75.0 vs 60.0 recall, GAIA 73.7 vs 54.5, WebShop 90.9 vs 63.6. Note the one-pass baseline has *higher precision* on ALFWorld (70.0 vs 60.0) — it produces a smaller, tighter, blander codebook.

**Theoretical convergence.** Independently of the prior expert analysis, the core category on all three benchmarks names the same mechanism: an action stream that has stopped absorbing environmental feedback. *feedback-decoupled control* (ALFWorld), *persistent repetition without adaptation* (GAIA), *ritualized non-diagnostic search* (WebShop). Prior work called it a cascade of errors and attributed it to upstream cognitive modules; this account stays at the observable-behaviour level and posits no modules.

**GLM sanity check.** In Go-Browse, "clicks visible UI affordance" has $\beta = -2.59$ (associated with *success*). "Clicks UI, closes without verification" has $\beta = +3.28$. So the clicking is not the problem — skipping verification before declaring done is. Fit diagnostics: Nagelkerke $R^2$ of 0.618 / 0.431 / 0.483 on the three datasets, Pearson $\chi^2/\text{df}$ near 1 for two of four rows (1.809 on the small Tau-Bench subset is a worse fit).

**Failure prediction from a 50% prefix** (MCC / ROC AUC, four backbones × three datasets). The headline is *mixed*, and the authors say so. With GPT-5-mini on Tau-Bench: few-shot direct prediction 0.061 MCC, few-shot codebook features 0.257, AutoTraceGT codebook features 0.151 (**worse**), union of both 0.306. On Go-Browse with GPT-5: few-shot 0.358, AutoTraceGT 0.498, union 0.499. The pattern across the table: AutoTraceGT features alone beat baselines on Go-Browse and SWE-Agent but lose on Tau-Bench in three of four backbones; the **union** of AutoTraceGT and few-shot codebook features is best or joint-best almost everywhere. The reading is that the two codebooks carry complementary high- and low-level signal, not that one dominates.

**Things that did not work or did not help:**

- **Fancier sampling policies.** Stratified-by-semantic-cluster, distance-weighted, and codebook-conditioned sampling were all tried. None consistently dominates. Uniform random saturates in the *fewest* batches with comparable final codebook size. Codebook-conditioned sampling reaches ~0.99 Chao1 coverage with fewer rare categories but converges slower.
- **Semantic clustering as a predictive baseline.** Embed the prefix with `text-embedding-3-large`, $K$-means with $K = 10$, one-hot the cluster, same AutoML. It beats AutoTraceGT features on Tau-Bench (0.268 vs 0.151 MCC) but loses on SWE-Agent and Go-Browse, and never beats the complementary feature set.
- **The merge-first worry.** Reviewers would ask whether "merge first" mechanically fakes saturation by suppressing `add`. Rerun on SWE-Agent with a neutral prompt: mean add-rate moves 0.125 → 0.129, still satisfies $\epsilon = 0.20$. Codebook grows from 20 to 26 categories but has cosine 0.957 to the original — above the 0.923 reseed reference. So the bias is real but small.
- **Batch size and chunk size barely matter.** $B \in \{10, 20, 30\}$ gives the same saturated codebooks; smaller batches just take more rounds. Chunk size 50 is a context-budget choice, not a methodological one.
- **Cross-family generalisation is thin.** All four main backbones are GPT. A single Gemini-3-Flash run on SWE-Agent converges in a median of 4 batches with only **11** final categories versus ~20 for GPT — noticeably coarser. The authors call this an extension of evidence, not a proof of robustness.

**Judge validation.** Two ML grad students independently labelled 100 category pairs from definitions alone. Inter-annotator Cohen's $\kappa = 0.74$ (87% raw agreement); annotator-vs-judge $\kappa = 0.66$ and $0.76$. Disagreements are lopsided: annotators flipped 13 and 9 judge-*non*-matches to matches, but only 4 and 3 in the other direction — the judge is conservative, so reported coverage is likely a floor. Separately, both annotators reviewed all 17 categories unmatched to the human taxonomy and called **all 17 valid**, with 14 unanimously failure-related. That is a strong result: the extra categories are not noise.

## Worth Remembering

**Cost.** Open coding runs under $0.04 per trajectory on every OpenAI model tested, usually under $0.01. A complete GPT-5-mini codebook costs $0.74 (Tau-Bench, 3 batches, 91 calls) to $1.91 (SWE-Agent, 6 batches, 274 calls). Compare to $3.33 per trajectory for a human at eight minutes and $25/hour — though that human estimate is doing heavy lifting and the authors flag it.

**The audit trail is the real artefact.** Appendix Table 13 traces one SWE-Agent category, *edit application errors*, across ten codebook versions: 9 members at v1, 84 at v10, with each of the ten merge decisions logged alongside its justification ("wrong-file and wrong-scope edits are already covered; the proposal and existing category have no clear behavioral distinction"). If you want to argue with the taxonomy, you can argue with a specific logged decision. That is rare in LLM-generated analysis and is arguably the paper's biggest practical contribution over prompting a model to "list failure modes".

**Limitations the authors own.** Every coding stage is an LLM, so backbone blind spots propagate; cross-LLM stability rules out *single-model* dominance but not biases shared across frontier models — and the Gemini result hints the family effect is real. Coverage numbers are conditional on an LLM judge. It runs to saturation, not a fixed budget, with many calls per trajectory, so it is an offline corpus tool, not something you put in a serving loop. Only single-agent English trajectories with binary outcomes were tested.

**The honest gap.** The algorithmic stopping rule — add-rate below a threshold — is *not* the methodological notion of saturation, which also cares about the density of examples inside each category and the stability of relations between them. A codebook can stop adding categories while its relational structure is still churning. Future work flagged by the authors.

**Practical caveat if you want to use this.** The predictive story is the weakest part. On Tau-Bench a naive few-shot-induced codebook beat AutoTraceGT's in three of four backbones, and $K$-means on embeddings beat it too. Use this to *understand* behaviour and generate hypotheses; do not assume the codebook is automatically a better feature space than something cheap.

**Follow-up question worth chasing.** The obvious next step, which the authors name: turn these categories into process-level reward signals. If "clicks UI then closes without verification" has $\beta = +3.28$ for failure, that is a step-level penalty you could put into RL training rather than a post-hoc label. The link to fine-grained credit assignment work is direct.

## Links

Related: [[SWE Refactor Bench- Can Coding Agents Complete a Long-Horizon, Whole-Repository Stack Migration]] · [[DRACO- Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training]] · [[What Makes Good Agentic Data- An ACE Lens on Data Generation for LLM Agents]] · [[The Handoff Tax- Continuing Non-Native Trajectories in LLM Agents]] · [[Small Language Models as Judges for Rubric-Based Reinforcement Learning]] · [[AgentMercury- Your Agent Can Synthesize Verifiable Environments for Business Scenarios at scale]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[In Context Learning]] · [[Troubling Trends in Machine Learning Scholarship]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Shortcut Learning in Deep Neural Networks]] · [[Chain-of-Thought Faithfulness of Reasoning Models Varies with Where and How Preference Cues Are Delivered]]

New topics worth writing: Grounded theory and qualitative coding, Theoretical saturation, Matthews correlation coefficient (MCC), Cohen's kappa and inter-annotator agreement, LLM-as-judge validation protocols, Chao1 species-richness estimator, FLAML and AutoML, Permutation tests as null baselines, Nagelkerke pseudo-R², Agent failure taxonomies (MAST, AgentErrorTaxonomy), Process reward models
