---
title: "The Router Within: Eliciting Native Skill Routing from a Frozen LLM"
authors: ["Chen et al."]
year: 2026
arxiv: "2609.15982"
url: https://arxiv.org/abs/2609.15982
priority: Good-To-Read
read_on: 2026-09-19
tags: [paper, transformers, llm, self-supervised]
---
## The Core Idea

An "agent skill" is a folder with a `SKILL.md` file inside: instructions, scripts, reference files that teach an agent to do one job well. The hard part is not writing skills. It is picking the right one from a library that may hold tens of thousands.

Today there are two ways to pick, and both are bad in a different way.

**Progressive disclosure** (what Claude Code and Codex do): paste every installed skill's name and one-line description into the system prompt, and let the agent choose. The agent is smart, so the choice is smart. But the menu eats the context window in proportion to the library. Codex caps skill metadata at 2% of the window and throws the rest away. And even within the cap, accuracy decays as the library grows, because the model's attention is spread over every entry, and a one-line description omits most of what the choice depends on. See [[Lost in the Middle]] and [[Context Window]] for why a long menu is not free.

**Retrieve-and-rerank**: keep the context clean, hand selection to an external embedding model plus a [[Reranking|reranker]]. Now the context is fine, but the decision is made by a model that never saw the rollout, does not know what the agent is in the middle of, and does not get better when the agent gets better.

So: picking *inside* the context taxes the agent; picking *outside* it loses the agent's judgment. The paper's question is whether you can have both — the agent's understanding, but outside its context.

The answer is yes, and the trick is that **the routing signal is already sitting in the frozen agent's hidden states**. You do not need to fine-tune it, add a model beside it, or put any skill text in its prompt. You need two linear matrices — 7.9M parameters total, `768 × 5120` each — to read the signal out.

**Gavel** (Glance And Verdict from a frozen LLM) does this in two steps:

1. **Glance.** Bolt a brand-new [[Query, Key, and Value (QKV)|attention head]] onto the frozen model at one middle layer. A query map $W_q$ reads the task's token states; a key map $W_s$ read each skill's token states, computed once at install time. Score the entire library by max-similarity per token. Cheap enough to sweep everything.
2. **Verdict.** For the ~9 survivors, resume the skill's forward pass with the task appended and read two things straight off the model's own output distribution: how well the skill *predicts* the task text, and the model's own yes/no log-odds on "does this skill serve this task?"

The elegance is in the fusion. All three scores estimate the *same* quantity — $\log p(s \mid x)$, the log posterior of skill given task — just read three different ways: contrastively, generatively, discriminatively. So they combine as a plain sum (a product of experts), and any one of them can veto.

> [!NOTE] Native read-out
> Getting a useful signal out of a frozen model's intermediate activations, without changing a single weight of it, by training a tiny map on top. Contrast with [[LoRA]], which changes behaviour; a read-out only *observes*. ^native-read-out

> [!NOTE] Progressive disclosure
> The deployed convention for skills: the agent sees a menu of every skill's name and description in its prompt, and reads the full body of one only when it decides to. Cost scales with library size and is paid on every single model call. ^progressive-disclosure

What it unlocks: routing accuracy that **climbs as the backbone climbs**, with no retraining of the router. A 0.6B backbone with Gavel beats a 32B backbone under progressive disclosure. The frozen 32B triggers the right skill 90.9% of the time in a live bash harness, versus 1.1% under the standard prompt.

## The Methodology

### Setup

Library $\mathcal{S} = \{s_1,\dots,s_N\}$ of documents (metadata header + body). Frozen LLM $\mathcal{M}$, Qwen3-32B in the main runs, 64 blocks. At a routing point the context $x$ holds the task — either a written user request, or whatever state the rollout has reached. $h_\ell(c)$ = layer-$\ell$ hidden states of $\mathcal{M}$ on input $c$.

Three hard constraints they set themselves: no skill text in the context until one is chosen; no standalone model beside the backbone; no training run when a new skill is installed.

### Which layer to read

They pick the layer where **matrix entropy of the token states bottoms out**. Stack a skill's layer-$\ell$ token states as $Z$, take the normalised spectrum $p = \lambda(ZZ^\top)/\operatorname{tr}(ZZ^\top)$, and compute $S_1 = -\sum_k p_k \log p_k$. This measures how many directions the states actually occupy. Entropy climbs for two-thirds of the depth, then **collapses from 4.01 to 1.83 across a single block**, floors at layer 44 of 64 (~70% depth), and recovers.

One catch they handle: a massive-activation "sink" token dominates the spectrum wherever it appears, so the entropy would be measuring the sink, not the representation. They drop the top-norm token before taking the spectrum.

Training the read-out at six candidate depths (32, 44, 55, 58, 60, 63) confirms layer 44 wins on R@20. **But** — and this is a lovely detail — if you skip the trained projections and just use raw cosine similarity over pooled states, layer 44 is the *second worst* of 34 depths (R@20 of .503 vs .916 at layer 60), a 44-point spread. The compressed mid-layer holds the best information and is the least readable without a learned map.

### The glance

A new head with a query path and a key path and **no value path** — it only scores, it never mixes.

At install, render skill $s$ in a fixed prompt $r(s)$ ("Here is an agent skill (SKILL.md): name… description… body… Here is a user task:"), one forward pass, store a unit key per token of header and body:

$$d^s_j \propto W_s\, h_{\ell^*}(r(s))_j$$

At routing time, each task token becomes a unit query $q_i \propto W_q\, h_{\ell^*}(x)_i$ — free, because the rollout computed those states anyway. Per-token score is hard attention, exactly [[ColBERT- Efficient and Effective Passage Search via Late Interaction|ColBERT]]-style [[ColBERT- Efficient and Effective Passage Search via Late Interaction#late-interaction|late interaction]]:

$$m_i(s) = \max_j \langle q_i, d^s_j\rangle$$

The head attends across *sequences* — a live context into documents encoded elsewhere — so it carries **no positional encoding** at all. No [[RoPE]], nothing.

**Training.** Multi-positive [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)|InfoNCE]] on the span mean $\bar m(s) = \operatorname{mean}_i m_i(s)$:

$$\mathcal{L} = -\log \frac{\sum_{s \in \mathcal{S}^+} e^{\tau \bar m(s)}}{\sum_{s \in \mathcal{S}^+ \cup \mathcal{S}^-} e^{\tau \bar m(s)}}$$

with $\tau = 40$. Gradients stop at $h_{\ell^*}$, so only $W_q, W_s$ move. 51,104 SkillRet training queries over 9,084 skills. [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], constant LR $10^{-3}$, weight decay 0.01, fp32, 24,000 steps. Backbone states are computed once and cached, so training is cheap. A step packs tasks until gold + negatives hit a 200K-token budget, max 256 tasks; [[Distributed Representations of Words and Phrases (negative sampling)|negatives]] resampled from the library each step.

**Inference uses a different aggregation than training.** This matters. The mean was fine for training but fails at routing: only a handful of task tokens point at the right skill, and as spans grow the rest — each vaguely similar to everything — bury them by sheer count. So each token *votes*, contributing only to its top-$k$ skills:

$$g(s|x) = \frac{1}{\sum_i w_i}\sum_{i\,:\,s\in\mathcal{T}_i} w_i\, m_i(s), \qquad \mathcal{T}_i = \text{top-}k\text{ under } m_i$$

$k = \min(10, \max(3, \mathrm{round}(0.1N)))$, so 10 for any library of 100+. $w_i$ is a decay favouring recent tokens; uniform on written tasks, $w_i = 2^{-a_i/64}$ mid-rollout where $a_i$ is the token distance back from the decision point. The normaliser keeps $g$ on the scale the head was trained at.

### Compressing the key banks

One key per token means the bank grows with total library length. But a document that dwells on one capability leaves a cluster of near-identical keys, and since everything passes through a $\max$, one representative serves the cluster. So keep an $\varepsilon$-**cover**: a subset where every discarded key is within $\varepsilon$ of a retained one, built by **farthest-first traversal** (Gonzalez 1985) — repeatedly keep the key farthest from those already kept, until nothing is left uncovered.

**Proposition 1 (distortion).** For any unit $q$:
$$\max_{d \in D_s}\langle q,d\rangle - \varepsilon \;\le\; \max_{c \in C_s}\langle q,c\rangle \;\le\; \max_{d \in D_s}\langle q,d\rangle$$

So compression can only lower a score, never raise one, and by at most $\varepsilon$. The proof is one line of Cauchy–Schwarz. A corollary carries the bound through the voting rule for any skill clearing every token's top-$k$ boundary by more than $\varepsilon$.

The bound is loose in practice. Cauchy–Schwarz is tight only when $q$ is collinear with $d^* - c$, which for unit vectors forces $\langle q, d^*\rangle \le \varepsilon/2$ — a key that barely matched anyway. On the tokens that actually decide a routing, $q$ sits close to $d^*$, and the real loss is $\approx \|d^*-c\|^2/2 \le \varepsilon^2/2 = 0.34$ rather than the linear $0.83$.

**Proposition 2 (bank size).** The traversal's output is simultaneously an $\varepsilon$-cover *and* $\varepsilon$-separated, so $\mathcal{N}_\varepsilon(D_s) \le |C_s| \le \mathcal{P}_\varepsilon(D_s)$. A skill's bank is sized by the number of $\varepsilon$-distinguishable directions its tokens span, not by its length.

At $\varepsilon = 0.83$: banks shrink **8.5×** at a cost of at most 1.6 points. Because compression coarsens the geometry $W_q$ was trained against, $W_q$ alone gets a 3,000-step fine-tune at LR $10^{-4}$ against the compressed banks, with $W_s$ and the keys frozen.

### The verdict

A factorised head cannot do subtle joint inference. Full [[Causal Attention|causal attention]] over skill and task together can, but costs one forward per candidate — affordable only on a shortlist. Only candidates within margin $\Delta = 0.133$ of the glance's best get a verdict, ~9 on average.

Continue the render $r(s)$ with the task — classical query-likelihood order (Ponte & Croft 1998), which also leaves the installation pass a reusable [[KV Cache|KV-cacheable]] prefix. Read two signals from the *same* forward pass:

**Generative** — mean log-likelihood of the task given the skill:
$$L(s|x) = \frac{1}{|x|}\sum_{i=1}^{|x|} \log p_\mathcal{M}(x_i \mid r(s), x_{<i})$$

**Discriminative** — append a fixed question ("Does this skill provide what that task needs? Answer yes or no:") and take log-odds at the final position:
$$V(s|x) = \log\!\sum_{t\in\mathcal{Y}} p_\mathcal{M}(t \mid r(s),x,u) - \log\!\sum_{t\in\mathcal{N}} p_\mathcal{M}(t \mid r(s),x,u)$$

with $\mathcal{Y} = \{$yes, Yes, YES$\}$, $\mathcal{N} = \{$no, No, NO$\}$, each with and without leading space, all single tokens. Because $u$ comes *after* the task, the causal mask leaves every task position untouched — one pass, both read-outs.

### The ruling

$$S(s|x) = g(s|x) + \alpha L(s|x) + \gamma V(s|x)$$

$(\alpha,\gamma) = (1.0, 0.025)$ on Qwen. $e^S$ is a product of experts (Hinton 2002), so any one expert can veto a candidate the other two tolerate.

The justification is the nice bit. All three are estimates of $\log p(s|x)$ up to a per-task shift:
- $g$ — the minimiser of multi-positive InfoNCE is the log ratio of posterior to negative-sampling distribution, up to a shift shared by every skill under a task.
- $L$ — summing over positions gives $\log p_\mathcal{M}(x \mid r(s))$, which [[Beliefs|Bayes]] turns into the log posterior under a uniform prior, up to another shift. Dividing by $|x|$ moves no ranking within a task; it just puts tasks of different lengths on one scale.
- $V$ — the model's stated posterior, discriminative.

So $\alpha, \gamma$ are exchange rates converting nats onto the glance's scale, and simultaneously tempering exponents that discount an overconfident expert.

Everything after training is four scalars ($\varepsilon, \alpha, \gamma, \Delta$) calibrated once on SkillRet validation. Every other benchmark is zero-shot.

## Ablation Studies and Experiments

### Scoring

Hit@1 over the full library, **adjudicated**. Gold labels systematically miss adequate skills — the libraries are scraped from public repos where the same capability exists in several documents. So GPT-5.6 Sol compares the committed skill against every gold in both orders, blind to which is which; the skill is credited if it wins at least as often as it loses. Examples of the problem: SkillRet's library contains two skills both literally named `mcp-builder`, only one annotated. Eval-Core holds a verbatim copy of the gold `rdkit` skill (4-gram Jaccard .92). SRA-Bench's "distractor" pool of 25,626 crawled documents contains real, adequate skills.

**Adjudication helps the baselines more than Gavel** — progressive disclosure gains 21.1 points on SkillRet against Gavel's 10.0 — and the raw Hit@1 of full Gavel is still above every other row in every column. So the ranking is not an artefact of the judge. Human check on 200 stratified pairs: 86% agreement with the judge; on the 132 pairs where the judge picked a side, humans flipped it 8 times.

### Written tasks

Three benchmarks, chosen to break in different directions from the training corpus:
- **SkillRet test** — 4,997 Claude-written conversational queries, 6,660 held-out skills. Closest to training.
- **SRA-Bench** — 861 tasks from reasoning/coding benchmarks, 26,262 skills mostly crawled from the web. Different genre on *both* sides.
- **Eval-Core** — 75 real `task.md` project files, 78K documents, two pools (hard adds 780 adversarial distractors).

Adjudicated Hit@1, full pipelines:

| System | SkillRet | SRA-Bench | Eval-Core easy | hard |
|---|---|---|---|---|
| BM25 | .501 | .523 | .667 | .640 |
| SKILLRET-Emb-0.6B | .880 | .486 | .800 | .733 |
| Qwen3-Emb-8B → prog. disclosure (20) | .864 | .643 | .907 | .893 |
| SkillRouter 0.6B + 0.6B reranker | .908 | .487 | .827 | .787 |
| Qwen3-Emb-8B + Qwen3-Reranker-8B | .865 | .740 | .920 | .920 |
| **Gavel** | **.946** | **.873** | **.947** | **.933** |
| *glance alone* | *.873* | *.696* | *.893* | *.867* |

Gavel beats the strongest pipeline by 3.8 on SkillRet, **13.4 on SRA-Bench**, 1.3–2.7 on Eval-Core — against pipelines adding 1.2B to 16B external parameters, versus 7.9M. The glance *alone* is the best retrieval stage on SRA-Bench and Eval-Core, and R@20 is .9706 / .9675 / .8800 / .8533 — best or tied-best everywhere.

The baselines fail in an instructive pattern. The two *trained* small embedders match the glance on SkillRet (their training distribution) but collapse on SRA-Bench — **below BM25**, the classic out-of-domain dense-retrieval failure from [[Approximate Nearest Neighbor Negative Contrastive Learning (ANCE)|BEIR-style]] evaluation. Untrained Qwen3-Emb-8B does the opposite: fine where the task looks like web retrieval, but trails the far smaller trained embedders on SkillRet's conversational requests, which share little vocabulary with skill bodies. Neither failure mode touches the glance, because it reads a general backbone.

### Mid-rollout: SkillTraj

The authors' new benchmark, and the more honest test. 372 GPT-5.6-Sol-written multi-turn trajectories with tool calls, rendered in the backbone's chat template, each marking the exact *token position* where a skill becomes needed. Four scenarios by where the need originates: **user request** (116), **tool evidence** (106), **agent plan** (105, the need appears inside the agent's own drafted plan), **wrong-skill recovery** (45, the agent already loaded a wrong skill whose full body sits in the context).

Admission is blind and strict: a Claude judge sees only prefixes truncated at the point under test, never the continuation, and the before/after views are judged independently in random order. The need must be absent before, present after, solved by the gold, and *not* solved by any of the gold's nine nearest library neighbours (or that neighbour is recorded as a second gold). Leakage screens ban the gold's name and flag long verbatim overlap. Failing trajectories are regenerated from scratch, never patched.

Adjudicated Hit@1:

| System | user req | tool ev | agent plan | recovery |
|---|---|---|---|---|
| SkillRouter (full ctx) | .483 | .123 | .210 | .200 |
| SkillRouter (last msg) | .871 | .340 | .438 | .667 |
| Qwen3 8B+8B (full ctx) | .707 | .443 | .495 | .511 |
| Qwen3 8B+8B (last msg) | .879 | .509 | .686 | .667 |
| Glance + prog. disclosure | .759 | .528 | **.152** | .778 |
| **Gavel** | **.966** | **.726** | **.905** | **.889** |

Gavel leads by 8.6 to 21.9 points. Two findings worth keeping:

**Every pipeline is better off with just the last message than with the full context that actually holds the evidence.** In all four scenarios. An embedding is a single vector; feeding it a long noisy trajectory blends every topic the rollout touched into mush. Gavel does not have to choose, because it reads per-token states with a recency decay.

**Progressive disclosure collapses to .152 on `agent plan`** — worse than anything. The freshly drafted plan turns on details only a skill *body* can confirm, and the menu has only descriptions. It does best on `wrong-skill recovery` (.778), where the agent has just watched a skill fail and summarises the missing capability in one clean sentence that a name and description match fine.

### Ablations — what did not work

**Glance → Jina-ColBERT-v2.** The obvious objection: maybe the glance wins just because it scores at token granularity, not because of what it reads. So swap in a retriever *trained* for exactly this late-interaction scoring, keep the verdict behind it, recalibrate $(\alpha,\gamma)$ the same way. It loses badly everywhere: .780 / .710 / .853 / .840. The signal is the agent LLM's compression, not the comparison scheme.

**Product of experts → pointwise rerank.** Restore the classical division of labour — retrieval nominates, reranker alone decides — by committing to $V$ and discarding $g$ and $L$. Worst variant of all: raw Hit@1 falls to **.407** on SkillRet (adjudicated .717), because self-reported relevance judgments are badly calibrated and the ambiguous queries scramble.

**Verdict → skill body in generation.** Reverse the order: task first, skill second, and score the mean log-likelihood of the *body*. This fails for a clean reason — a body's likelihood measures how much the skill resembles the model's own writing, a per-skill bias that does *not* cancel across candidates, whereas query-likelihood's bias is shared across skills under one task and so leaves the ordering intact. Calibration drives $\alpha \to 0$, discarding the signal outright.

**Verdict → progressive disclosure over the glance's 20.** Closest on Eval-Core (curated skills with good descriptions), far behind on SRA-Bench (.695 vs .873), where telling skills apart needs the procedural body.

**Full bodies in the menu don't fix it either.** With Qwen3-32B's window extended to 131K by YaRN, all twenty complete `SKILL.md`s fit (longest prompt 124K). Result: +7 points on SRA-Bench, **−2.2 on SkillRet**, noise on Eval-Core. Gavel stays 16 adjudicated points ahead on SRA-Bench. The diagnostic: on SkillRet the gold's complete body is in the prompt for **92% of queries, and the picker chooses something else in 29% of them**. Missing information is not the problem — dispersed attention is. Cost: median prompt 1.3K → 28K–52K tokens, 26–124 seconds of prefill per pick.

### Dropping one verdict signal

| Variant | SkillRet | SRA-Bench | Eval-Core easy | hard | SkillTraj |
|---|---|---|---|---|---|
| glance alone | .873 | .696 | .893 | .867 | .737 |
| $g + \alpha L$ | .927 | .815 | .840 | .813 | .817 |
| $g + \gamma V$ | .898 | .835 | .907 | .893 | .825 |
| full | **.946** | **.873** | **.947** | **.933** | **.871** |

The full ruling wins everywhere, but the two signals are *not* interchangeable. $V$ helps on every library. $L$ helps hugely on SkillRet (+5.4) and SRA-Bench (+12.0) but *hurts* Eval-Core by four queries per pool. The likelihood carries corpora whose queries spell the task out in enough detail for the right skill to anticipate the wording; the judgment carries corpora where the question is whether a procedural page or a half-formed rollout state calls for the skill — which the model answers better when *asked* than when made to predict wording. They miss different queries, and the product keeps the union.

### Native read-outs that failed (Appendix A — the best part of the paper)

Three attempts to lean *harder* on machinery the model already owns. All three fail, and each failure pins down one design choice.

**1. Use the backbone's own attention keys as the matching space.** The maximally native glance: index skills by the pre-rotary keys the model's real heads assign, match the task's real queries against them. Untrained, the best of 24 configurations of head selection, low-rank whitening and aggregation reaches Hit@1 = **.001** on a 10,123-skill library, against .918 for the trained head. Exact zeros on SRA-Bench, with R@20 below random. Most of the loss is calibration — native scores are not comparable across documents encoded in separate sequences. Restricted to a 20-candidate shortlist it reaches .099, twice in-pool chance. Training a mixer over per-head scores makes a usable shortlist reranker but drops 7.6 points on the full library, and the weights are **library-bound**: −1.2 points on the library they were fitted on, **−24 on another**. Equal-weight and non-negative mixtures collapse further, ruling out negative weights as the cause. The signal *is* there — the single best head reaches .084 alone, and head ranking is stable across libraries — but one to two orders of magnitude too faint, and no a priori criterion finds the good heads (the head placing the most attention mass on the gold ranks **29th** by retrieval).

**2. Initialise from native heads.** Stack six strong native heads' query/key maps up to the glance's dimension. Result: 0.7 points *below* random init; the multi-head shape 0.6 below one wide head; both together −4.6. And head selection contributes nothing — six *randomly chosen* heads beat the six the mixer weighted highest by 1.3 points. Hence: one wide map, randomly initialised.

**3. Learned aggregation via trained retrieval tokens.** Replace the hand-crafted decay $w_i$ with 64 trained retrieval tokens appended to the context, aggregating via the backbone's own attention. They learn **recency instead of content** — attention settles on the most recent stretch. The authors blame the data, not the idea: the plentiful supervision is clean single-turn requests where recency is a sufficient policy, so nothing forces selectivity, yet the router must survive contexts whose trigger sits buried mid-history. The hand-crafted decay stays.

### End-to-end in a live harness

Integrated into `mini-swe-agent`. Every benchmark above hands the router its moment; a live rollout needs a **gate** deciding *when* to try. They train a linear classifier over (a) accumulated glance scores over the library and (b) the final-layer state the model is predicting from, using SkillTraj decision points as positives and matched no-skill trajectories as negatives. It fires after staying on for two consecutive tokens. If the verdict then judges even the winner unfit ($V < 0$), load nothing.

On **Skill-Use** (177 executable tasks, 79 skills), trigger rate = fraction of tasks loading the correct skill:

| System | Trigger |
|---|---|
| MiniMax-M3 in Codex | .864 |
| GLM-5.1 in Codex | .706 |
| Qwen3.6-Max in Codex | .684 |
| DeepSeek-V4-Pro in Codex | .650 |
| **Qwen3-32B, progressive disclosure, mini-swe-agent** | **.011** |
| Qwen3-Emb-8B + Qwen3-Reranker-8B | .897 |
| **Qwen3-32B + Gavel** | **.909** |

That .011 is the headline. Under the standard prompt Qwen3-32B reads a skill on 2 of 175 tasks that call for one. But if you *force* its first action to be reading the one skill it picks from that same listing, the score is **.834** — so the listing is fine; the failure is the unprompted decision to consult documentation at all. That is a trained harness habit. Frontier models swing up to 38 points in trigger rate between Codex and Claude Code; Qwen3-32B predates the skills ecosystem entirely. Gavel supplies the trigger from outside the model's habits.

The retrieve-and-rerank pipelines score well but **load a skill on every trajectory whether one is needed or not**. Gavel holds fire on ~5% of tasks.

### Scaling with backbone

On SRA-Bench, both the glance and the full pipeline climb with backbone size and generation. **Gavel on a 0.6B model — one that all but fails at picking from a metadata menu — already beats the 32B under progressive disclosure.** On Qwen3.8-27B, metadata-menu picking still trails Gavel on the same backbone by 12.2 points. See [[Scaling Laws for Neural Language Models]] for the general shape.

### Backbone transfer

Rebuilt end-to-end on `gemma-4-31b-it` (different family, chat template, interleaved sliding-window attention, logit soft-capping). Nothing transferred. Layer selected anew → block 34 of 60. $\varepsilon = 0.77$. $(\alpha,\gamma) = (0.15, 0.003)$ — a third the Qwen values, and they explain why: both verdict read-outs spread ~3× wider on Gemma (median within-query std 14.8 vs 4.5 nats for $V$; 0.75 vs 0.23 nats/token for $L$) while the glance spreads about the same (0.061 vs 0.052), so a coefficient a third the size carries the same weight. Shape survives; Gavel again tops all three benchmarks.

## Worth Remembering

**The cost analysis is the practical decision rule.** Gavel pays only for verdicts — the gate and glance read states the rollout computes anyway, and banks are built offline:
$$C_{\text{Gavel}} = m \times 9 \times (100 + 1895 + 239) \times \$4/10^6 \approx \$0.080\,m$$
independent of library size $n$ and call count $T$. Progressive disclosure re-sends its $100n$-token menu on every call (hosted APIs are stateless; all previous input tokens are billed each time), cached after the first:
$$C_{\text{PD}} \approx \$4\times10^{-5}\, nT$$
**Gavel is cheaper once $n > 2000\,m/T$.** Three firings = \$0.24; a 100-skill menu over 50 calls = \$0.25. Close, at realistic scale. The real cost of the menu is not dollars — it is $100n$ tokens of finite window held for the whole session, and every decoded token attending over them.

**Latency shape.** Glance is one matmul, milliseconds. Verdict is nine ~2.2K-token prefills, batched, ~half a second per firing, less with per-skill prefixes [[KV Cache|KV-cached]] — and **off-path**: they never enter the rollout's context, and the skill prefix is query-independent so it caches per skill.

**The gate is deliberately unpolished.** A linear classifier fired on two consecutive tokens, trained once and not engineered. The authors flag two better designs they did not build: a sequence model over the *history* of glance scores (watching a need build across positions rather than judging each alone), and a backbone fine-tuned to emit a dedicated skill-call token, folding the decision into decoding. Both leave glance and verdict untouched.

**The failure mode of the gate is visible in the transcripts.** It is read at every decoded token, so a firing interrupts mid-sentence — mid-word, even, in one case mid-email-address. The half-written message stays in context as far as it got, the `SKILL.md` is appended as a `[skill loaded: …]` message, and the turn restarts. Most fires land in the opening sentence; seventeen of 181 land in later turns after command output arrives. One task fires twice, the second time loading a *non*-gold skill (`api-security-best-practices`) that the task genuinely needed.

**The honest headline is `.011` → `.909`, not a modelling win.** Most of that gap is the trigger decision, not routing quality — the forced-read control already gets .834. The routing improvement over strong pipelines is real but smaller (2–13 points on written tasks, 9–22 mid-rollout). Do not read the Skill-Use table as "Gavel routes 80 points better".

**Limitations the authors admit.** The gate is trained on their own synthetic SkillTraj and evaluated on Skill-Use — related construction, different distribution, but the gate is the least validated component. SkillTraj itself is LLM-written and LLM-judged, however carefully screened. Adjudication is LLM-judged with an 86% human agreement rate and a measured lean toward generosity (19 pairs credited that humans would not, 10 withheld that humans would, out of 200). Four scalars are tuned on SkillRet validation and reused zero-shot — reasonable, but it is still four numbers someone chose.

**Connections.** The glance is [[ColBERT- Efficient and Effective Passage Search via Late Interaction|ColBERT]] late interaction with the backbone's own frozen states as the encoder, trained with [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)|InfoNCE]] and read out via voting rather than pooling. The verdict is the query-likelihood reranker (Sachan et al.) plus the monoT5-style yes/no relevance head (Nogueira et al.), both on the frozen agent. ToolkenGPT is the closest prior — frozen LLM emitting a tool as a token — but trains one embedding per tool, so a new tool costs a training run; Gavel's install cost is one forward pass. Contrast with [[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]] and GRIT, which modify weights; the read-out only observes. The layer choice rests on Skean et al.'s compression-valley result and Queipo-de-Llano et al.'s attention-sink correction. Compare the two-stage funnel to [[Deep Neural Networks for YouTube Recommendations (RecSys)#two-stage-funnel|candidate generation + ranking]] — but with a product of experts rather than the reranker superseding the retriever, and the ablation shows that difference is worth 3–6 points.

**If you wanted to use this.** You need local access to the backbone's mid-layer hidden states, so it does not work behind a hosted API today — Appendix K sketches how a provider would ship it (projections shipped with the model like a tokenizer, banks built at skill upload like a prompt cache, glance running inside the prefill the provider already does). You must recalibrate $(\alpha,\gamma)$ per backbone family, and the reason is measurable: read-out spread differs ~3× between Qwen and Gemma. Budget one forward pass per skill at install, and ~$8.5\times$-compressed key storage per skill.

**Open questions.** Does the same read-out route tools, memories, and [[Model Context Protocol|MCP]] servers? The authors leave it open, and it is the obvious next paper. Can the glance alone eventually settle routing, with no verdict, once the backbone is capable enough — the whole library ranked in passing by the model already working on the task? Appendix F.3 raises this explicitly. And can learned aggregation replace the hand-crafted decay if you train it on data where recency is *not* a sufficient policy?

**One transferable idea for evaluation work.** The adjudication protocol is a template worth copying: judge blind to which side is gold, run both orders, credit on win-rate not identity, then report the raw score alongside and *check that your ranking survives on the raw numbers too*. And report which systems adjudication helps most — if it helps your baselines more than your method, as here, your comparison is not resting on the judge.

## Links
Related: [[ColBERT- Efficient and Effective Passage Search via Late Interaction]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[Reranking]] · [[Lost in the Middle]] · [[Context Window]] · [[Query, Key, and Value (QKV)]] · [[KV Cache]] · [[Deep Neural Networks for YouTube Recommendations (RecSys)]] · [[Embeddings]] · [[Vector Database]] · [[Sentence-BERT]] · [[Dense Passage Retrieval (DPR)]] · [[Approximate Nearest Neighbor Negative Contrastive Learning (ANCE)]] · [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Tool Use]] · [[Agentic Workflows]] · [[Model Context Protocol]] · [[Scaling Laws for Neural Language Models]] · [[Evals]] · [[NDCG]] · [[Causal Attention]] · [[RoPE]]

New topics worth writing: agent skills and SKILL.md routing, product of experts, epsilon-covers and farthest-first traversal, covering and packing numbers, matrix-based entropy of hidden states, compression valleys and attention sinks, query-likelihood retrieval models, linear probing of hidden states, monoT5 and pointwise LLM reranking, ToolkenGPT and tool-as-token, LLM-as-judge adjudicated benchmark scoring, tool retrieval benchmarks, YaRN context extension, prompt caching economics
