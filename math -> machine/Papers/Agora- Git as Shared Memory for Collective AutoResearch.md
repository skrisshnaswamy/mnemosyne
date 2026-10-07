---
title: "Agora: Git as Shared Memory for Collective AutoResearch"
authors: ["Zhang et al."]
year: 2026
arxiv: "2609.18094"
url: https://arxiv.org/abs/2609.18094
priority: Good-To-Read
read_on: 2026-09-22
tags: [paper, transformers, llm, theory]
---
## The Core Idea

Many agent sessions work on the same research problem at different times. Each one finishes, and its findings die with the transcript. The next session does not know which learning rate blew up, which branch was abandoned, or which claim anyone actually reproduced. Existing multi-agent systems fix this by putting agents in one conversation with one manager — see [[Multi-Agent LLM Systems]] — but that only works inside a single episode.

Agora throws away the conversation and keeps the **record**. Every contribution an agent makes is a Git commit. Every parent edge means "builds on this". The result is an append-only directed acyclic graph (DAG — a set of nodes with one-way links and no cycles) of the whole research history: results, failures, hypotheses, reproductions.

> [!NOTE] Contribution DAG
> The shared state is not a chat log or a workspace. It is a commit graph where the node is the claim *and* the artifact that produced it, and the edge is "I extended this." ^contribution-dag

Two things fall out of this that a chat-based system cannot give you:

1. **Provenance is free.** The commit hash names the exact code. You can re-run any claim from a clean checkout.
2. **You can compute on the graph.** Which branches are leaves? Which results nobody reproduced? Which cluster of ideas is eating all the attention? Those are just queries.

That second point is where the interesting bit lives. A leaderboard tells 13 agents to all pile onto the current best score. Agora adds **diversity views** — explicitly surfacing neglected branches — and the paper's most convincing single observation is that turning those views on changed what the agents did within a day.

Workers share no filesystem, no model, no manager, and no prompt beyond a two-page brief. Coordination happens entirely through publishing and reading.

## The Methodology

### The graph

A project is $G = (V, E)$. Each node is

$$v = (h, a, T, d, x, m, P, \tau)$$

commit hash $h$, author account $a$, tags $T$, description $d$, structured metadata $x$, an optional project metric $m$, parent set $P$, server timestamp $\tau$.

Tags are typed and carry weights:

| Tag | Weight | Meaning |
|---|---|---|
| `setup` | +5 | commit zero, the brief |
| `result` | +5 | an experiment, success **or** failure |
| `insight` | +5 | interpretation of evidence |
| `hypothesis` | +5 | untested proposal, may not claim a metric |
| `report` | +5 | synthesis across nodes |
| `verification` | +20 / +10 / −20 | confirmed / partial / failed reproduction |
| `endorsed` | 0 | acknowledgement, deliberately worthless |
| `wip` | 0 | in-flight, to stop duplication |

### Credit comes from other people's work, not votes

$$S(u) = \sum_{v:(u,v)\in E} \mathbb{1}[a(u) \neq a(v)]\, w(v)$$

Your score is the weighted count of what **other accounts** built on top of you. The indicator function kills self-citation, so you cannot farm impact by extending your own branch. Endorsements are weight 0 on purpose — they are cheap, so they are excluded from fitness. A verifier who changes their verdict has the new verdict replace the old one's effect, while both commits stay in history.

> [!NOTE] Evidence score
> Impact is measured by *downstream reuse by strangers*, not by upvotes. This is the same instinct as a citation count, with self-citation mechanically banned. ^evidence-score

### Telling agents where to look

`agora analyze` returns metric leaders, most-built-on nodes, leaves, unverified results, contested verifications, open hypotheses, recent activity, tags, contributors.

Once ≥50% of contributions have [[Embeddings|embeddings]], it also builds single-link clusters over descriptions (cosine threshold 0.90, capped at the 5,000 most recent) and reports cluster sizes, top-cluster share, and an entropy-based effective cluster count.

Candidates are then ranked by a diversity-aware [[Upper Confidence Bound|UCB]]:

$$U(v) = 100\,Q(v) + C\sqrt{\frac{\log(N+1)}{n(v)+1}} + \frac{100D}{\sqrt{1+\rho(v)}}$$

- $Q(v)$ — quality percentile.
- Middle term — the standard [[Multi-Armed Bandit|bandit]] optimism bonus: $n(v)$ is follow-on work on $v$, $N$ is total. Rarely-extended nodes get a boost. Same shape as [[Monte Carlo Tree Search|UCT]].
- Third term — $\rho(v)$ counts near-duplicate descriptions, so a node in a crowded semantic neighbourhood gets penalised.
- $C$ grows when scores are bunched near the best, i.e. when the community is stuck on a plateau, [[Exploration vs Exploitation|explore harder]].

Results are shown in three named slots: **exploit** (refine the leader), **explore known** (extend promising work in a thin cluster), **explore novel** (untouched nodes in singleton clusters).

### Implementation

Go service, Next.js web UI, CLI. Per-project bare Git repo. SQLite holds eight tables (agents, projects, contributions, parents, tags, cross-project refs, embeddings, rate limits) and is **fully rebuildable from Git**. 26 HTTP routes, 15 CLI command groups. Two publication paths: metadata-only work posts JSON and the server makes the commit; code-bearing work uploads a Git bundle and the server validates then re-commits with a server timestamp.

### The task they ran it on

A weight-transfer problem, chosen because it is hard and cheaply scored.

- **Target**: frozen 14-layer hybrid, alternating [[Multi-Head Attention|multi-head attention]] blocks with simplified Mamba-style [[State-Space Model|selective SSM]] blocks. Hidden size 672, 7 heads, untied embeddings, 119,572,320 params. Dimensions chosen so **no donor matches any of them**.
- **Donors**: 141 open-weight models, 534 GB, 32 families (GPT-2, LLaMA, Mistral, Qwen, Gemma, Pythia, RWKV, Mamba).
- **Rules**: write `transfer(model, config)`. No training data, no gradient update on the target, no touching the evaluator. The FineWeb-Edu loader raises an error if called from inside `transfer()`.
- **Metric**: bits per byte (bpb) — summed next-token [[Cross Entropy|cross-entropy]] over 200 FineWeb-Edu texts in 512-token chunks, divided by UTF-8 byte count. Related to [[Perplexity]] but normalised by bytes so tokenizers are comparable. Seed 42, bit-identical on the same hardware, third-decimal drift across GPU types.
- **Anchors**: random init = 3.3923 bpb. Trained GPT-2 124M ≈ 1.0.

### The workers

13 coding-agent sessions — Claude Code (Opus 4.7) and Codex (GPT-5.5) — each in a container with one 80GB GPU, one Agora credential, and a one-line prompt: *read `program.md`, run `agora analyze`*. When a session ended, the launcher started a fresh one on a free credential. **Nothing named a method, assigned a role, or ranked participants.**

### What they found

**Stage A — transfer behaviour, not parameters.** Six donors sharing the GPT-2 vocabulary (GPT-2 small/large, Cerebras-GPT 111M–1.3B) are queried on every vocabulary token under 28 single-token contexts. Their next-token log-softmaxes are clipped to ±25, weighted by donor (0.725 on GPT-2 small) and by per-row context weights, and summed into a $50257 \times 50257$ context-averaged bigram log-prob table $M$. Column mean $u$ is split off as a unigram anchor; the centered table $C = M - \mathbf{1}u^\top$ is factorised to rank 671 by randomized SVD (oversample 32, one power iteration, fixed seed). $U$ becomes the input embedding, $VS$ the output head, dimension 0 carries $u$, two temperatures rescale. All sublayers zeroed. You now have a factorised bigram model wearing a 14-layer network as a costume — the same latent-factor move as in [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)|matrix factorisation]].

**Stage B — add a little context back.** Sparse hand-set edits on 96-dimensional bands $B_k = [1+96k, 97+96k)$ of the hidden state. Every attention layer has $W_q = W_k = 0$, which makes [[Attention|attention]] a uniform causal mean-pool over one band, written back at a small scale. Layer-0 SwiGLU gets SVD-projected slices of GPT-2 small's first MLP at scale 0.009. In each SSM block the selective path is switched off, reducing it to a gated depthwise causal convolution; layer 1 uses a sign-alternating kernel $(1.85, 1.65, 0.20, -2.70)$ that emphasises recent positions.

Every constant in Stage B was introduced as one change on the then-current best, and kept because the evaluator went down.

## Ablation Studies and Experiments

**Headline**: 3.3923 → **1.899044 bpb**, closing 62% of the gap to a trained GPT-2 124M, with zero gradient steps on the target.

**The milestone trace is the most instructive table in the paper:**

| When | Account | bpb | Change |
|---|---|---|---|
| — | — | 3.3923 | random init |
| Apr 27 00:24 | worker1 | **4.6784** | slice-copy GPT-2 + Mamba weights — **worse than random** |
| Apr 27 00:57 | worker1 | 2.5151 | unigram prior from GPT-2's *predictions* |
| Apr 27 01:50 | worker1 | 2.1284 | bigram matrix → randomized SVD into embedding + head |
| Apr 27 06:55 | worker2 | 1.9319 | 24 prefixes, geometric-mean aggregation |
| Apr 28 04:31 | slurm_worker_4 | 1.9228 | second donor + variance/naturalness weights |
| Apr 29 11:04 | slurm_worker_2 | 1.9136 | six donors, 28 contexts |
| May 1 05:10 | worker2 | 1.9062 | one power iteration in the SVD |
| May 1 10:28 | slurm_worker_3 | 1.9043 | layer-0 attention as uniform causal mean-pool |
| May 3 00:13 | slurm_worker_3 | 1.9028 | **first SSM edit, chosen after reading the new diversity views** |
| May 5 17:34 | slurm_worker_6 | 1.8995 | layer-0 feed-forward from GPT-2 small |
| May 8 13:24 | slurm_worker_1 | 1.8990 | cross-band SSM output writes |

**The first 18 scored contributions account for ~98% of the total reduction.** The remaining 1,106 results bought the final 0.03 bpb. The first *eight* improvements are ~70% of the descent. This is a sharply diminishing-returns curve, and the paper says so plainly.

**What did not work** (53 contributions explicitly tagged negative):

- **Copying parameters directly** — the very first attempt, at 4.68, worse than random. The whole run's insight is that you transfer *behaviour* (next-token statistics), not weights.
- Doubling the prefix set from 28 to 48 made things worse, documented with four controlled variants.
- Flattening the singular-value spectrum.
- Transplanting native Mamba blocks from hybrid donors.
- Copying GPT-2's embedding matrix directly.
- Building the prior from Pythia, which has a different tokenizer.

**Coordination measurements:**

- Final graph: 1,703 nodes, 1,894 edges, 149 multi-parent nodes, one component holding 98.9% of nodes.
- Winner's ancestry: 145 commits, 15 of 17 accounts, **115 of 144 parent edges cross account boundaries**. No single worker assembled the recipe.
- 165 verification contributions across 95 distinct targets, verifier ≠ author, **zero reported failures**. Same-hardware reproductions bit-identical; A100 vs H100 differ by up to $1.3\times10^{-3}$ bpb.
- **Parallel rediscovery**: 696 pairs of different accounts posted identical scores. 63% within an hour of each other, 80% within six hours. That is a lot of wasted GPU time, and the shared graph did not prevent it.
- **Narrow spine**: one lineage collects nearly all follow-on work; side branches are short and abandoned fast.

**The one intervention, which is also the one quasi-experiment.** By May 2, over a third of all activity sat in one semantic cluster and the leaderboard had stalled around 1.904. The authors deployed clustering, diversity summaries, and the diversity-aware UCB. Workers started using the views immediately; on May 3 at 00:13 a worker exploring the sparsely-populated state-space cluster published the first SSM edit at 1.9028, and the trace continued to 1.8995 by May 5.

That is suggestive, not proof. It is one arm, one timepoint, no control. The authors say this outright.

## Worth Remembering

**The honest limitation, stated by the authors.** There is no matched comparison. Appendix C *proposes* one: four arms (isolated / flat chronological log / central planner / Agora), matched on agents, models, compute, evaluator and wall-clock, with the whole community run — not the individual commit — as the unit of analysis. Until that runs, "does shared memory improve discovery per unit of compute?" is **an open question**, not a result.

**The last improvement is noise.** $9\times10^{-6}$ bpb, against cross-hardware variation of $1.3\times10^{-3}$. The final ~0.01 bpb of the run is below the instrument's resolution. Classic overfitting-to-the-dev-evaluator territory — every component was selected on the *same* 200 texts, so there is no held-out check at all.

**"No reported failures" across 165 verifications should make you uncomfortable.** Either the reproductions were genuinely easy (plausible: fixed seeds, bit-identical hardware, committed code), or verifiers have no incentive to report failure. The scoring rule gives +20 for a confirmed reproduction and −20 for a failed one — the −20 lands on the *target's* score, but a verifier still has to do the work of proving a negative. Watch this if you build something similar: the $-20$ weight is the only thing standing between you and a rubber-stamp culture.

**The self-citation exclusion is the cheap trick worth stealing.** $\mathbb{1}[a(u) \neq a(v)]$ is one line, and it removes the single most obvious way to game a reuse-based score. Compare the incentive failures in [[Reward Hacking]] — any scalar the agents can see becomes a target.

**Emergent conventions nobody asked for.** The brief required reproducible contributions. It did *not* ask for prediction bands. From April 28 onward, 400+ descriptions declare a predicted outcome band *before* the measured result, and later workers explicitly close follow-ups named by earlier ones. That is pre-registration arising from imitation in a shared corpus — a nice observation, and one the authors did not engineer.

**The science result is separately interesting.** "Transfer the donor's *output distribution*, not its weights" is a [[Distillation|distillation]]-flavoured move done without any gradients: query donors on every vocabulary token, build a bigram table, low-rank factorise it into the target's embedding and head. You get 62% of a trained GPT-2 with no training data. The architectures did not need to match because only the tokenizer had to.

**Practical caveats if you wanted to run this.** Thirteen 80GB GPUs for 12 days for a 0.03 bpb improvement after day one is a terrible compute-to-discovery ratio, and the 696 duplicate-score pairs say a big chunk went to rediscovering the same thing in parallel. The infrastructure is real and reusable; the efficiency claim is not established. Also note how much of the design leans on the task having a **cheap, deterministic, seconds-long evaluator** — see [[Evals]]. Without that, verification collapses and the `+20` weight has nothing to attach to.

**Connections.** The three-slot exploit/explore-known/explore-novel UI is quality-diversity thinking (MAP-Elites, novelty search) rendered as a menu for an LLM. The score-then-diversify arc is the same failure mode [[Regret|regret]] analysis warns about: a greedy leaderboard is a policy, and it was measurably the wrong one for five days. The lineage-as-commits idea sits next to [[Agent State and Checkpointing]] and [[Observability and Tracing]] — the difference is that a trace is for debugging one run, whereas this graph is meant to be read by the *next* run.

**Follow-up questions.** Does the UCB formula's $D$ and $C$ tuning actually matter, or would any nudge toward neglected clusters have worked? What happens with 100 workers instead of 13 — does the spine get narrower or does the graph fragment? And would a flat chronological log have done just as well, which is precisely the Appendix C arm nobody ran.

## Links
Related: [[Multi-Agent LLM Systems]] · [[Agentic Workflows]] · [[Upper Confidence Bound]] · [[Multi-Armed Bandit]] · [[Exploration vs Exploitation]] · [[Monte Carlo Tree Search]] · [[Memory]] · [[Evals]] · [[Agent Evaluation]] · [[Observability and Tracing]] · [[Agent State and Checkpointing]] · [[State-Space Model]] · [[Attention]] · [[Embeddings]] · [[Perplexity]] · [[Cross Entropy]] · [[Matrix Factorization Techniques for Recommender Systems (IEEE Computer)]] · [[Distillation]] · [[Reward Hacking]] · [[Regret]] · [[The Bitter Lesson (essay)]]

New topics worth writing: quality-diversity and MAP-Elites, novelty search, blackboard architectures, provenance and reproducible research artifacts (DataLad, RO-Crate, MLflow), bits-per-byte as an evaluation metric, randomized SVD, weight transfer without gradients, pre-registration and prediction bands as an agent convention, science-of-science and multiple discovery
