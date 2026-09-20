---
title: "SWE Refactor Bench: Can Coding Agents Complete a Long-Horizon, Whole-Repository Stack Migration?"
authors: ["Hong et al."]
year: 2026
arxiv: "2608.23564"
url: https://arxiv.org/abs/2608.23564
priority: Good-To-Read
read_on: 2026-09-04
tags: [paper, vision]
---
## The Core Idea

Normal coding benchmarks work because of one signal: **red to green**. A test fails before the patch, passes after. That jump *is* the proof the work happened.

Whole-repository migration destroys that signal. You start with SQLite that builds and passes everything. You ask the agent to port it from POSIX to WASI. Now the repository is already green before the agent touches anything. So the "do nothing" submission — hand the repo back untouched — scores **100% on any behavioural test suite you can write**, while doing zero work.

The authors name this failure **Blindness**.

> [!NOTE] Blindness
> A behaviour-only evaluator gives full credit without ever establishing that the migration happened. It can detect *broken* behaviour, but never *absent* work. ^blindness

The key argument is that this is not a hole you can plug with more tests. Formally: a test suite $T$ is a finite set of observations drawn from the observable interface $\mathcal{O}$. The preservation condition is defined *relative to* the original repository $R_A$, so $\mathrm{rate}(R_A; T) = 1$ for **every** $T$, by construction. Growing $T$ makes the empty-diff submission look *better*, not worse. Every case the migrated repo must pass is a case the original already passes.

So the fix must live outside behaviour. **SWE Refactor Bench** adds a second, non-behavioural instrument — a judge that reads the *source* and asks "is the old stack gone?" — and gives it a **veto**. Then it adds a third instrument: six other coding agents that hunt, after submission, for behavioural differences the fixed suite was never written to catch.

This separates two abilities that everyone had been measuring as one thing:

- **Did the migration happen?** (a claim about repository text and the build closure)
- **Did behaviour survive?** (a claim about a running artifact)

Agents fail these in *opposite directions*. Of 520 runs: 30 preserved behaviour by skipping the migration, 252 did the migration and broke behaviour. Only **28 of 520 (5.4%)** passed all three stages. 13 of the 20 tasks were never solved by anyone. Best model, `claude-opus-5` at xhigh effort: **47.0/100**.

## The Methodology

### The task, formally

A task is a tuple

$$\tau=(R_A,\; \Sigma_A \to \Sigma_B,\; \mathcal{O},\; \mathcal{I},\; E,\; B)$$

- $R_A$ — "State A", a real open-source repo at one commit, buildable and working.
- $\Sigma_A \to \Sigma_B$ — the stack swap (language, framework, host platform, or build toolchain).
- $\mathcal{O}$ — the **observable interface**: process output and exit status, exported symbols of an installed library, the manifest of an install tree, HTTP responses.
- $\mathcal{I}$ — the instruction, $E$ — an offline container image, $B$ — a time budget (6–30 hours).

A submission $R_S$ (the working tree when the run ends) solves $\tau$ iff both hold:

$$\Sigma_B \text{ builds the artifact, and } \Sigma_A \text{ is absent from repo and build closure} \tag{migration}$$
$$\mathcal{O}(R_S) = \mathcal{O}(R_A) \tag{preservation}$$

No new functionality is designed anywhere. State A already does everything State B must do. The whole difficulty is reproducing behaviour exactly on a stack that expresses it differently.

### The task set

20 tasks, 867k lines of source, drawn from load-bearing infrastructure: cmark, zlib, SQLite, libsodium, GraphHopper, QuickJS, Acorn, go-yaml, PyCryptodome, Gson.

| Class | n | Examples | Budget (h) | Fixed checks |
|---|---|---|---|---|
| Language rewrite | 7 | C→Rust, C→Java, Go→Zig, JS→TS | 12–30 | 59,771 |
| Framework rewrite | 7 | Flask→Starlette, Gin→chi, Vue→React | 6–16 | 55,852 |
| Platform port | 3 | POSIX→wasm32-wasi, CommonJS→V8 realm | 6–10 | 6,725 |
| Build toolchain | 3 | Autotools→CMake, Maven→Gradle, setuptools→Meson | 6 | 7,770 |

Selection rule: **pick the debt first, the repository second**. Three admission requirements — (1) the old stack must be *load-bearing*, so removing it reaches the design and not just the imports (cmark's parser is a hand-written C state machine; moving it to Rust means redesigning ownership); (2) the observable interface must have real downstream consumers (a C ABI, an HTTP API, an install tree); (3) State A must be repeatedly runnable, so differential testing is possible.

Repo history is **squashed to a single commit** — `git log` is the first thing a good agent reads, and an upstream log may already describe a past migration.

### Stage I — Migration Audit (a veto)

Criteria are written as prompt-form questions about *this specific repository* (136 criteria total, 5–10 per task). Judge model is `gpt-5.6-sol`. Each criterion is judged **three independent times, majority wins**. Every failing verdict must cite re-checkable evidence. **Every criterion must pass**; one failure zeroes the run.

For cmark the eight criteria are: `no-c-sources`, `no-foreign-headers`, `no-c-in-build`, `rust-present`, `rust-is-primary`, `no-embedded-reference`, `no-verifier-awareness`, `default-path`.

When the workspace is collected, an exclusion list strips `build-*`, `target/`, object files, install prefixes, `.git` — so "the old stack is gone" is a fact about *source*, not about what the agent left lying around.

### Stage II — Behavioural Tests

**130,118** fixed checks in 264 modules, averaging >6,000 per task. Not hand-written: the same calls are run against State A inside the evaluation image, and the recorded output becomes the expected answer.

**All-or-nothing.** One wrong check scores zero. Justification: these tasks ask whether the result is a *drop-in* replacement, and a library wrong once in a thousand calls is not one.

Sanity gate: at image build time, State A is scored against itself; unless every module reaches $1.0$, the image build is refused.

### Stage III — Agentic Verification

Six independent coding agents, **one hour each**, both source trees in hand, looking for something the original does that the submission does not. Five get one assigned direction (a C ABI, a route set, an install tree); the sixth is unrestricted.

A verifier cannot submit a report. Only an **executable counterexample**: it must pass on State A, fail on the submission, and reproduce three times — so a broken test and a flaky test both fail to win. Each task ships an allow-list and deny-list of directions (cmark: 15 allowed, 11 forbidden). Forbidden includes internal struct layout, behaviour the original also gets wrong, and — explicitly — *detecting which tree you are running against*, which would satisfy every mechanical condition for a break while proving nothing.

This is classical differential testing (McKeeman 1998) in agentic form.

### The score

$$S(\tau)=\underbrace{\mathbf{1}[g=\textsf{pass}]}_{\text{Stage I: veto}}\cdot \underbrace{\mathbf{1}[r_i = 1\;\forall i]}_{\text{Stage II: all-or-nothing}}\cdot \Bigl(0.4 + \underbrace{0.6\cdot\tfrac{s}{6}}_{\text{Stage III}}\Bigr) \in \{0\}\cup[0.4,1]$$

with $g$ the audit verdict, $r_i$ the pass rate of module $i$, $s$ the number of verifiers that failed to break the submission.

Stage I **multiplies** rather than adds, because without the migration the task was not done and no amount of correctness should buy points. Stage III is scored *linearly* rather than all-or-nothing, because "not broken by six verifiers" is stronger evidence than "not broken by one", but neither is a proof of equivalence — the linear term records degree of evidence, honestly.

### Evaluation setup

8 frontier models, 26 model–effort configurations, each run once on all 20 tasks = **520 scored runs**. GPT-series use the Codex harness; the rest use Claude Code. Fresh container per run, no network beyond the model endpoint. Stage I/II/III materials live in a separate image never mounted into the agent's container — the tests do not merely go unread, they **do not exist** for the agent.

## Ablation Studies and Experiments

### The funnel

Of 520 runs: **340 (65.4%)** passed Stage I, **118 (22.7%)** passed every fixed check, but only **88** did both and reached Stage III, of which **28 (5.4%)** survived all six verifiers.

Mean score over all 520 runs: **13.44/100**. Mean over the 88 that reached Stage III: **79.43**. The difficulty is entirely in getting to Stage III.

Crucially the three gates stop **different** submissions. Overlapping stages would be redundant; disjoint stages measure three different things. With the fixed suite as the only instrument, **118 submissions would tie for first place** — including ones that migrated nothing and ones a verifier breaks within an hour.

### Leaderboard (best config per model)

| Model | Effort | Behavioural pass % | Score /100 | $/task |
|---|---|---|---|---|
| claude-opus-5 | xhigh | 92.8 | **47.0** | 74.9 |
| gpt-5.6-sol | max | 84.1 | 28.5 | 143.5 |
| kimi-k3 | max | 93.9 | 19.5 | 28.9 |
| claude-sonnet-5 | medium | 73.9 | 15.0 | 11.9 |
| gpt-5.6-luna | max | 89.1 | 10.5 | 2.8 |
| qwen3.8-max | max | 74.7 | 10.0 | 14.5 |
| dsv4-flash | max | 90.7 | 7.0 | 4.3 |
| glm-5.2 | max | 85.2 | 6.5 | 17.5 |

The rows worth studying are the three with **zero acceptances** — `gpt-5.6-luna`, `dsv4-flash`, `glm-5.2` — each of which produced runs (4, 3, 3) that passed *every* fixed check. Judged by behaviour alone all three would show "perfect scores" on the leaderboard. Under the full protocol they solved nothing.

Note also that behavioural pass rate is a badly misleading headline: `kimi-k3` has the *highest* behavioural pass (93.9%) and a score of 19.5.

### The two abilities are genuinely separate

- **30 runs** never migrated and kept every fixed check (blindness), spread over 7 of the 8 models. Stage II gives all 30 full marks; only Stage I stops them.
- **252 runs** completed the migration and broke behaviour. Only Stage II stops those.

Per-task, both extremes appear:
- `lang04` (Acorn, JS→Rust): 20/26 runs passed Stage I, **not one** passed every fixed check.
- `lang01` (cmark, C→Rust): only 6/26 passed Stage I, but 5 passed every fixed check — and **all five are blindness**. Each cleared 7 of 8 criteria and failed only `rust-is-primary`: the Rust reproduces the C control flow statement for statement. A transliteration into `unsafe` Rust, the same hand-managed pointer stack, same names, same order. Ownership was never redesigned; the memory safety the migration exists to buy was not bought. **No behavioural test can express that**, because behaviour was never where the problem was.

Four submission shapes Stage I must separate: nothing rewritten (C copied, suffix changed); wrapped (`extern "C"` shim, C still does the work); half done; rewritten wrongly. The first two pass every behavioural test.

### The last 1% is where agents die

Among the 340 runs that passed Stage I: **91%** get past half the fixed suite, **58%** reach 99%, **36%** reach 99.9%, but only **26%** make no error at all. That final step alone eliminates 35 of the 123 runs that had reached 99.9%. 140 runs land in $[99\%,100\%)$, missing a median of 12.5 checks; **18 miss exactly one**.

Those 18 are not random. They cluster:
- `fw03` (Conduit, Vue→React): four different models all ended at 21768/21769, failing the same check — the original uses hash routing so `/` settles at `/#/`, the React version stays at `/`. That breaks every bookmark and shared link.
- `build03` (PyCryptodome, setuptools→Meson): five models all ended at 380/381 — the `METADATA` long description in the built wheel is 0 characters, so the published PyPI page is blank.

Neither is nitpicking. Both are production regressions, and the original passes them.

### Stage III takes back two thirds

Of the 88 submissions that missed no fixed check, only **28** survived all six verifiers. The other **60 (68.2%)** were broken within the hour; the average submission held off only **3.94 of 6** verifiers. Median time to a counterexample: **17.0 minutes**, versus 32.8 minutes for a survival.

Examples of what a fixed suite could never have anticipated:
- `fw04` (ChartMuseum, Gin→chi): the original truncates `Content-Type` at the first space **or** semicolon; the migration truncates only at the semicolon. One extra space before the multipart boundary routes the request to a different handler. This is an implementation detail of Gin the task author did not know existed.
- `lang05` (go-yaml, Go→Zig): Go's `time.Parse` also accepts a comma as decimal separator, so `2001-12-14T21:59:43,10Z` is a `!!timestamp` on the original and a `!!str` in Zig.
- `pf01` (SQLite, POSIX→WASI): after checkpointing a WAL database and switching back to DELETE mode, native SQLite removes both `-wal` and `-shm`; the WASI version removes only `-wal`.

### Category profiles — the bottleneck moves

| Category | Stage I pass | Stage II (of I) | Stage III (of II) | Accepted | Score |
|---|---|---|---|---|---|
| Build toolchain | 80.8% | 54.0% | **17.6%** | 6 | 31.4 |
| Platform port | 57.7% | 37.8% | 23.5% | 4 | 17.2 |
| Framework | 72.5% | **18.9%** | 56.0% | 14 | 12.0 |
| Language | 54.9% | **12.0%** | 33.3% | 4 | 5.6 |

Agents are *not* best at one class throughout. Build toolchain has the highest Stage I and II rates and the **lowest** Stage III survival; framework rewrites are the reverse. The reading: in toolchain and platform work the product code is largely untouched, so the fixed suite passes often but Stage III finds differences outside it. In framework and language work the product code itself changes, so the fixed suite kills submissions early. Language rewrites show the sharpest attrition — only **12 of 182 runs** ever reached the verifiers.

### Is the instrument itself sound? (the good part)

**Judge stability.** Of 3,536 criterion verdicts, 3,405 (96.3%) were unanimous across three samples; 131 split 2:1. That looks small, but Stage I requires *every* criterion, so one disagreement decides a whole run: **35 of the 340** passing runs had at least one criterion passing only 2:1. Without majority voting (letting any single dissent count), 305 rather than 340 would pass. Majority-of-3 is load-bearing, not ceremonial.

**Judge vs humans.** Two independent SE researchers labelled all 156 runs of 6 tasks. Agreement **89.7%** (140/156, $\kappa = 0.795$). Direction matters more than count: of 16 disagreements, **14 are the judge being too strict** and only 2 too lenient. Following all 16 downstream: 12 of the 14 too-strict would have failed Stage II anyway; 1 of the 2 too-lenient likewise; the remaining one reached Stage III and four verifiers broke it. So Stage I error let **no** human-recognised fake into the accepted set.

**Self-preference?** `gpt-5.6-sol` is both judge and contestant. It passed 57.1% of the 240 GPT-authored submissions vs 72.5% of the other 280 — the wrong direction for collusion.

**Verifier ablation — model choice beats configuration by an order of magnitude.** All six verifiers face the same 88 submissions. The two `claude-opus-5` verifiers break **55.7%** and **53.4%**; the other four break **21.6%–26.1%**. The two opus verifiers differ in prompt *and* effort *and* scope (one directed, one unrestricted) and yet differ by only **2.3 points**. Changing prompt/effort within a model moves the break rate ~2 points; changing the model moves it **~30**.

Consequence the authors state plainly: retire the two strongest verifiers and the remaining four would accept **46** submissions instead of 28. An accepted submission is *not proven correct*; it survived the strongest adversaries currently fieldable.

**But the six are not redundant.** Counting *exclusive* breaks (only this verifier found it): the two opus verifiers got 5 and 2, the four weaker ones 4 between them. Even the weakest verifier rejected a submission the other five let through. Strength and coverage are different axes.

**Verifier self-preference?** Pooled, verifiers broke 33.9% of same-family submissions vs 34.3% of others — no effect. The apparent counter-evidence (opus verifiers break 40.8% of opus submissions vs 65.0% of others) fails its own control: the four non-opus verifiers show 17.1% vs 29.1% on the same submissions, **the same factor**. Opus submissions are simply harder to break, for everyone.

### What did not work / what was never solved

13 of 20 tasks got no acceptance, and 7 never even had a submission reach the verifiers — in two distinct ways:

- `lang03` (sqlparse Python→Go), `lang04` (Acorn JS→Rust), `pf02` (Stylus CommonJS→V8 realm): **no run ever passed every fixed check**.
- `lang01` (cmark), `fw01` (httpbin Flask→Starlette), `fw02` (json-server Express→Fastify), `fw07` (GraphHopper Dropwizard→Spring Boot): some runs passed every fixed check, but **every such run was rejected at Stage I as blindness**. A behavioural instrument reports these four tasks as *solved*; the three-stage protocol reports them as unsolved.

Easiest tasks by score: `build03` (64.6), `pf03` (49.2), `fw06` (43.1). Hardest with score exactly 0.00: `lang01`, `lang03`, `lang04`, `fw01`, `fw02`, `fw07`, `pf02`.

## Worth Remembering

- **The clean statement of the problem.** Any evaluation defined *relative to* the starting state cannot detect that work happened. This generalises far past code: refactoring, dependency upgrades, any "make it different but behave the same" task. If your reward is $\mathrm{rate}(R;T)$ and $\mathrm{rate}(R_A;T)=1$, the maximum of the reward sits on a submission with zero work in it.
- **Not reward hacking, exactly.** The authors are careful here. A repo handed back as-is circumvents no check — it earns full marks *by the rules*. The rules simply cannot see the migration. So the remedy is not tighter tests but a second instrument outside behaviour, holding a veto. Compare with reward-model gaming in [[Training language models to follow instructions with human feedback]] — that is exploiting a loophole; this is a specification that is silent.
- **Why not formal verification.** Translation validation, differential symbolic execution and verified compilers all need two sides whose semantics can be related. A cross-language whole-repository rewrite does not offer that. Agentic verifiers are an explicit *approximation* of the same effect.
- **The benchmark tightens itself over time.** Because Stage III uses frontier coding agents as adversaries, the same fixed 20 tasks get scored more strictly as models improve. That is unusual and rather elegant — but it also means scores from different years are not comparable, and the paper's numbers are a property of *this verifier panel*.
- **Cost matters and is wild.** `gpt-5.6-sol` at max effort costs $143.5/task for score 28.5; `claude-opus-5` at xhigh costs $74.9/task for 47.0. `gpt-5.6-luna` costs $2.8/task for 10.5.
- **The cmark result is the sharpest thing in the paper.** All 15 modules at $1.00$, all 4,184 checks pass, 7 of 8 audit criteria pass — and final score **0**, because the Rust is the C transliterated. Every behavioural instrument in the stack calls this complete; the one instrument that reads code calls it a transliteration.
- **The instruction never enumerates the tests.** Stated as a design principle: "an instruction that enumerates the tests is an instruction to satisfy the tests." Worth stealing for any agentic eval you build.
- **Limitation the authors admit.** These results establish a capability gap on *these 20 tasks*; they do not rank the intrinsic difficulty of migration classes in general. Sample sizes per class are small (3 tasks for platform and toolchain).
- **Open question.** Stage I is an LLM judge answering narrow, evidence-citing questions. It works here ($\kappa=0.795$, biased strict) but it is still a model grading a model. Scaling this to thousands of tasks means scaling the human-written criteria too — nobody has shown how to generate those automatically without reintroducing blindness.

## Links

Related: [[Towards Quantifying Benchmark Optimization in ASR Models]] · [[On the Difficulty of Evaluating Baselines]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Troubling Trends in Machine Learning Scholarship]] · [[Shortcut Learning in Deep Neural Networks]] · [[Do ImageNet Classifiers Generalize to ImageNet]] · [[Training language models to follow instructions with human feedback]] · [[Constitutional AI- Harmlessness from AI Feedback]] · [[AgentMercury- Your Agent Can Synthesize Verifiable Environments for Business Scenarios at scale]] · [[Autonomous Mathematical Discovery in an Open-World Multi-Agent Environment]] · [[Best Practice Critic Optimization]] · [[Thinking in a Low-Resource Language- What SFT Builds, What RL Fixes, What Accuracy Cannot See]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]]

New topics worth writing: SWE-bench and repository-level coding benchmarks, Differential testing, LLM-as-a-judge and its failure modes, Reward hacking and specification gaming, Technical debt (Cunningham 1992), Property-based testing (QuickCheck), Metamorphic testing, Delta debugging, Translation validation and verified compilers, WebAssembly and WASI, Long-horizon agent harnesses (Codex, Claude Code), Cohen's kappa
