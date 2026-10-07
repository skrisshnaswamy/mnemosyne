---
title: "Learning from Teacher Continuations at Student States"
authors: ["Wang et al."]
year: 2026
arxiv: "2609.36246"
url: https://arxiv.org/abs/2609.36246
priority: Good-To-Read
read_on: 2026-09-30
tags: [paper, llm, rl, vision]
---
## The Core Idea

Distillation usually means one of two things. Either you take text the teacher wrote and train the student on it (offline SFT), or you let the student write text and have the teacher score every token of it (on-policy distillation, OPD). OLIVE does a third thing: **the student starts the answer, then the teacher finishes it, and the student trains only on the teacher's half.**

That is the whole trick. The student writes a prefix. The prefix is handed to the teacher as context. The teacher continues autoregressively from that exact point. The student is then trained with plain [[Cross Entropy|cross-entropy]] on the teacher's continuation, with the prefix masked out of the loss.

Why this matters — each piece fixes a specific named failure:

**1. Offline SFT trains on states the student never visits.** You train on teacher trajectories, but at test time the student conditions on its own words. That mismatch is sequential covariate shift, and errors compound over a long answer. This is the classic behaviour-cloning problem from [[Imitation Learning]], and the classic fix is DAgger: collect the expert's label *at states the learner reaches*. OLIVE is DAgger with an LLM teacher.

**2. OPD grades tokens but never demonstrates the recovery.** OPD computes a reverse-KL target at every token of the student's own rollout. If the teacher says "token 400 should have been different", tokens 401 onward are still conditioned on the student's original token 400. So the supervision says *that was wrong* but never shows *what follows from being right*. Teacher takeover fixes exactly this: every token after the handoff is conditioned on the teacher's own earlier choices.

**3. OPD needs teacher logits.** Reverse KL needs the teacher's per-token probability distribution. An API model gives you text only. Because OLIVE's loss is cross-entropy on text, you can re-tokenise the teacher's output with the *student's* tokenizer and train. GPT-5.4-mini becomes a usable online teacher.

> [!NOTE] Online intervention
> Supervision is placed at a state the *current* student actually reaches, and the demonstration continues forward from there under the teacher's own control. Prefixes are regenerated every step, so the supervision tracks the student as it changes. ^online-intervention

The unlock: offline distillation plateaus after ~2 epochs because the fixed data stops matching where the student now is. OLIVE keeps climbing, and ends 13 points higher on ScienceWorld after 5 epochs.

## The Methodology

### Single-turn (reasoning)

Per training step:

1. Sample prompt $x$ from the prompt pool.
2. **Roll in:** student samples a prefix, $y_{1:k} \sim \pi_\theta(\cdot \mid x)$, with $k = 4096$ tokens. No filtering — every prefix is kept.
3. **Continue:** teacher generates $\tilde{y} \sim \pi_T(\cdot \mid x, y_{1:k})$, capped at $M = 1024$ tokens. Note this is a *partial* continuation. The teacher does not have to finish the problem or be verified correct.
4. **Update:** re-tokenise $\tilde{y}$ with the student tokenizer to get $\tilde{y}_{1:\ell}$ (length differs from the teacher's token count — that is fine), feed the whole stitched sequence $(x, y_{1:k}, \tilde{y}_{1:\ell})$ through the student, and minimise

$$\mathcal{L}(\theta) = -\sum_{j=1}^{\ell} \log \pi_\theta\!\left(\tilde{y}_j \mid x,\, y_{1:k},\, \tilde{y}_{<j}\right)$$

The prompt and the student prefix stay in the context window but contribute nothing to the loss. The sampled text is frozen during the update — no gradient flows back through the sampling.

That is it. No KL term, no reward, no value function. It is [[Fine-Tuning|SFT]] with the data regenerated every step.

### Multi-turn (agentic)

Same split, but $k$ and $M$ count **turns**, not tokens.

- Student acts in the environment for $k$ turns (ReAct format: Thought, then Action), accumulating actions and real observations.
- Teacher takes over for up to $M = 5$ turns. Each teacher action is conditioned on the full history and is *actually executed*, so the subsequent observations come from the teacher's choices.
- Loss: cross-entropy on the teacher's actions only. Student turns masked. **Environment observations masked too** — you do not want the student learning to predict the environment.

$k = 10$ for long-horizon environments (ALFWorld, ScienceWorld), $k = 5$ for shorter ones.

### Asynchronous implementation

The obvious cost is GPU idle time while the teacher generates. Borrowing from asynchronous RL (A3C, IMPALA), they overlap: while the teacher continues batch $n$, the student samples prefixes for batch $n+1$.

This means a prefix may come from a slightly older student than the one being updated. The staleness is bounded by an **asynchronous depth** $d$ — the max number of student updates between generating a prefix and training on it. They use $d = 3$.

### Hyperparameters that mattered

| | OLIVE | OPD baseline |
|---|---|---|
| Loss | token CE on continuation | reverse KL, top-16 logprobs |
| Learning rate | $1 \times 10^{-5}$ | $1 \times 10^{-6}$ |
| Rollouts per prompt | 4 | 4 |
| Batch size | 64 | 64 |
| Epochs | 1 | 1 |
| Token budget | 4096 prefix + 1024 continuation | 7168 total |

Note the LR is 10× higher for OLIVE. CE on text and reverse KL on distributions are different-magnitude signals.

### Setup

- **Reasoning:** RLVE — synthetic verifiable environments with tunable difficulty knobs. They deliberately dial difficulty *above* the student's ability, since distillation is meant to add capability the student lacks. 18 environments × 500 problems = 9K hard problems, 10 test problems per environment. Students: Qwen3-1.7B and Qwen3-4B, thinking mode on. Teacher: Qwen3-4B-Thinking-2507.
- **Agentic:** 5 AgentGym environments — ALFWorld, ScienceWorld, TextCraft, BabyAI, SearchQA. Student Qwen3-1.7B, teacher Qwen3-32B. Turn caps: 30 (ALFWorld/TextCraft/ScienceWorld), 20 (BabyAI), 16 (SearchQA).
- 8× H200 for the efficiency study.

## Ablation Studies and Experiments

### Reasoning (RLVE)

| Method | 1.7B pass@8 | 1.7B avg@8 | 4B pass@8 | 4B avg@8 |
|---|---|---|---|---|
| Original student | 11.1 | 3.3 | 46.1 | 18.1 |
| Offline teacher-gen SFT | 15.0 | 5.6 | **53.3** | 19.9 |
| OPD | 14.4 | 4.6 | 45.6 | 21.3 |
| **OLIVE** | **19.4** | **7.8** | 52.2 | **23.5** |

OLIVE wins avg@8 on both students (+4.5 and +5.4 over base). Offline SFT edges it on 4B pass@8 (53.3 vs 52.2) — worth noting, since pass@8 measures whether *any* of 8 samples works, and offline SFT saw far more complete teacher solutions.

**OPD actively hurt the 4B student's pass@8** (46.1 → 45.6). That is the headline negative result for the baseline.

### Why OPD stalls — the diagnostic

They track top-$K$ overlap between student and teacher next-token distributions on a validation set through OPD training. It goes from **0.707 to 0.713**. Essentially flat. The student and teacher have different *thinking styles*, and reverse-KL on the student's own rollouts does not close that gap. If the distributions barely overlap to begin with, grading tokens gives you almost nothing to move toward.

### Agentic (avg@4 success rate, %)

| Method | ALFWorld | ScienceWorld | TextCraft | BabyAI | SearchQA |
|---|---|---|---|---|---|
| Student | 19.4 | 0.1 | 23.0 | 38.3 | 30.5 |
| Teacher (32B) | 52.1 | 15.6 | 85.5 | 83.3 | 55.1 |
| OPD | 22.3 | **0.0** | 29.5 | 43.1 | **29.6** |
| TCoD-B2F | 37.1 | 0.8 | 39.5 | 66.3 | 37.7 |
| TCoD-F2B | 28.0 | 0.5 | 45.5 | 62.5 | 37.8 |
| Guided OPD | 27.1 | 0.8 | 45.5 | **67.8** | 37.5 |
| **OLIVE** | **40.0** | **7.5** | **55.3** | 67.5 | **39.1** |

ScienceWorld is the cleanest evidence for the whole thesis. The student scores 0.12% — essentially never succeeds. OPD takes it to **0.00%**. There is nothing to grade: the student's early actions are broken, every later turn in the episode inherits the wreckage, and the teacher's distribution conditioned on those doomed states carries no signal toward completion. OLIVE gets 7.5% by *intervening* — actually taking over and steering the episode back.

OLIVE also uses **fewer turns** per success in every environment, so it is not just brute-forcing.

BabyAI is the one place OLIVE loses (67.5 vs Guided OPD's 67.8) — a tie, really, and BabyAI is the environment the student was already best at.

### The case study is worth reading

Ten student turns on ScienceWorld: the task says the animals are `outside`, and the student loops `go to outside` → `open outside` → `look around`, over and over. It is trying to open the *location* instead of the *door*. Score stays 0.

Teacher's first turn: `open door to the outside`. Done in four more turns, final score 1.00.

A teacher-only trajectory starting from the initial observation would never contain that correction, because the teacher would never have got stuck that way. This is exactly what "supervision at states the student reaches" buys.

### Online vs offline prefixes (§5.1)

Compared against **OEC** — the offline version of OLIVE: sample prefixes once from the initial student, get teacher continuations, filter by verifier, train. Plus a plain SFT baseline on filtered full teacher trajectories. For fairness, this experiment lets OLIVE generate *full* filtered continuations too, so the only difference is whether prefixes are refreshed.

OLIVE beats both. Refreshing the prefix as the policy moves is doing real work, independent of the takeover mechanism.

### Forgetting

Measured on AIME25 (avg@16), LiveCodeBench v6 (avg@8), IFEval, GPQA-Diamond. OLIVE causes only a **0.9% average drop** on general benchmarks while gaining the most on task. Offline distillation loses more. Consistent with the "RL's Razor" line of work — staying near your own distribution forgets less.

### The plateau (§5.2) — the result I would quote

Student Qwen3-1.7B, teacher **GPT-5.4-mini** (text only, so OPD is simply unavailable). ScienceWorld, 5 epochs:

- Offline distillation **plateaus after epoch 2**. The static data has stopped describing where the student is.
- OLIVE is *worse* for the first two epochs, then keeps climbing and ends **13% above** offline.

Be honest about that shape: if you only train two epochs, offline wins. The advantage is a long-run one.

Also: OLIVE with more rollouts per prompt beats the matched offline checkpoint at equal update count — so the gain is diversity of supervision, not just more gradient steps.

### Plasticity under sequential training (§5.3, Fig. 8)

Train on the 5 environments one after another on Qwen3-4B, then test all of them. The gap is **largest on ScienceWorld, which is last in the sequence** — +13.0 points for OLIVE. Fixed offline data mismatches worse and worse as the policy drifts through earlier stages. OLIVE re-elicits from whatever policy currently exists, so stage 5 is as learnable as stage 1.

Generalisation check: train sequentially on BabyAI → TextCraft → SearchQA → ScienceWorld with GPT-5.4-mini, hold out ALFWorld. OLIVE is +2.8 over offline on the unseen environment.

### Cost (§4.3)

8× H200, Qwen3-4B student, 9K RLVE problems.

- Synchronous OLIVE ≈ OPD GPU-hours, better performance. Because OLIVE only needs 1024 teacher tokens and does not compute reverse KL over 7168 student tokens.
- Async OLIVE: **−28% total time vs OPD**, **−23.8% vs synchronous OLIVE**, with only minimal performance loss.

### Prefix length ablation (ALFWorld, teacher turns fixed at 5)

Success rate with varying student prefix turns: 35.6–40.0%. Any prefix beats none. But it is **not monotonic** — 10 turns is best (40.0%), 15 turns drops to 35.6%, 20 turns recovers to 39.0%. There is no "longer prefix is better" rule here; you have to tune it per environment.

### Terminal agents (Appendix C.3, preliminary)

Qwen3.5-2B student, Qwen3.5-9B teacher, 500 TMax tasks, eval on 50 TBLite tasks with pass@4. Budget matched at 20 turns (OPD: 20 student turns graded; OLIVE: 10 student + 10 teacher).

- Base: 10.0%
- OPD: **10.0%** — zero improvement
- OLIVE: 12.5%

Small eval set, so treat as directional. But the same pattern: OPD does nothing when the student is too weak to reach recoverable states.

## Worth Remembering

**This is DAgger, renamed for LLMs.** Learner roll-in, expert roll-out, refreshed every iteration. The lineage is Ross & Bagnell 2011 via Ross & Bagnell 2014 (interactive no-regret). If you already understand [[Imitation Learning]] and why behaviour cloning compounds errors, you already understand why OLIVE works. What is new is applying it where the "expert" is an API you can only sample text from.

**The paper says "ongoing work" on the front page.** Some numbers are single-run, the terminal-agent result is explicitly preliminary, and there is no seed variance reported anywhere. Treat the direction as solid and the magnitudes as soft.

**No filtering, no verifier, no reward.** The main reasoning experiments keep every prefix and every partial teacher continuation, unverified. The teacher does not even finish the problem. This is remarkable — and it means OLIVE is cheap to set up relative to anything RLVR-shaped. The §5.1 analysis *does* use verifier filtering, but only to make the comparison to OEC fair.

**Partial continuations are enough.** The reasoning case study shows why: the teacher spends its first few hundred characters completing the student's half-written thought, then corrects the error. The correction arrives early. You do not need to pay for a full solution.

**Honest limitations:**

- **You must host the teacher online.** That is the real cost, and async only hides part of it. A 32B teacher serving generations alongside training is a serious infrastructure commitment. The async depth $d$ is a knob you now own.
- **Slower to start.** Two epochs of losing to offline SFT is a real cost if your budget is short.
- **$k$ needs tuning and the response is non-monotonic.** No principled way given to pick it.
- **The prefix must be recoverable.** The paper says the continuation works "when the student prefix remains recoverable". If the student has made an irreversible mistake — deleted a file, broken the environment — no continuation saves it. They do not measure how often this happens.
- **Off-policy by construction.** The loss puts gradient only on teacher tokens, but conditions on student tokens. Nobody corrects for the mismatch. It works empirically; there is no theory here.

**Practical caveat on tokenizers.** Re-tokenising the teacher's text with the student's tokenizer is what makes black-box teaching possible, but it means the teacher's own token boundaries are lost. For a teacher with a very different tokenizer this could matter more than the paper's Qwen-to-Qwen and GPT-to-Qwen settings reveal.

**The framing I would carry forward.** Distillation design has two axes, not one: *what form* the supervision takes (logits vs text vs preferences), and *where it is placed and whether it refreshes*. Most of the field argues about the first. This paper argues the second is at least as important — and the flat 0.707 → 0.713 overlap curve is the evidence that a better loss on the wrong states buys nothing.

**Open question.** OPD and OLIVE are not exclusive. Grade the student's prefix with reverse KL *and* train on the teacher's continuation. Nobody tried it here.

## Links

Related: [[Imitation Learning]] · [[Distillation]] · [[Distilling the Knowledge in a Neural Network]] · [[Cross Entropy]] · [[KL Divergence]] · [[On-Policy vs Off-Policy]] · [[Fine-Tuning]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[On-policy Distillation with Verifiable Reward]] · [[1% of Tokens Can Be Enough- On Gradient Estimation in On-Policy Distillation]] · [[When EOS Tokens Disagree- Understanding Length Inflation in On-Policy Distillation]] · [[RetireOPD- Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning]] · [[What Does Privileged Information Add to On-Policy Self-Distillation]] · [[Continual Learning Mechanisms Compose for Long-Horizon Memorization]] · [[Agentic Workflows]] · [[Credit Assignment]] · [[GRPO]] · [[Chain of Thought]] · [[Tokenization]]

New topics worth writing: DAgger and learner roll-in / expert roll-out, sequential covariate shift, asynchronous RL (A3C and IMPALA), AgentGym and ScienceWorld as agentic benchmarks, RLVE tunable-difficulty environments, ReAct interaction protocol, plasticity loss in continual post-training, black-box distillation from API teachers, staleness bounds in async training
---
name: "AgentSpace Registry Kubernetes Executor"
description: "Container-native execution runtime for Claude Skills in Kubernetes clusters — CRDs, resource contracts, network policy, and operational envelope."
version: "1.0.0"
license: "Apache-2.0"
authors:
  - "AgentSpace Platform Engineering"
tags: ["kubernetes", "operator", "execution", "sandboxing", "infrastructure"]
allowed-tools: []
runtime:
  kind: "kubernetes"
  min_cluster_version: "1.27"
  crd_group: "agentspace.dev"
  crd_version: "v1"
---

# AgentSpace Registry: Kubernetes Executor

## Purpose

This skill defines how AgentSpace Registry schedules skill invocations as short-lived Kubernetes workloads. It exists so that skill authors can reason about the **execution envelope** — what resources a skill gets, what it can reach on the network, how long it may run, and how failures surface — without reading operator source code.

It is a reference document. Reading it does not grant any capability, and it is not itself an executable skill.

---

## 1. Resource model

Every invocation becomes a `SkillRun` custom resource, which the operator reconciles into a single-container `Pod`.

### 1.1 The `SkillRun` CRD

```yaml
apiVersion: agentspace.dev/v1
kind: SkillRun
metadata:
  name: pdf-extract-7f3a9b
  namespace: agentspace-runs
spec:
  skillRef:
    name: pdf-extract
    version: "2.4.1"
    digest: "sha256:9c1f…"     # required; resolved at admission
  invocation:
    argsSecretRef: run-7f3a9b-args   # arguments never inline in spec
    timeoutSeconds: 300
  resources:
    tier: standard                    # see §1.2
  network:
    profile: egress-none              # see §3
status:
  phase: Succeeded                    # Pending|Running|Succeeded|Failed|TimedOut|Evicted
  startedAt: "2025-11-04T09:12:33Z"
  finishedAt: "2025-11-04T09:12:51Z"
  exitCode: 0
  resultRef: run-7f3a9b-result
  conditions: [...]
```

Two properties matter for authors:

- **`spec.skillRef.digest` is mandatory.** Tags are resolved to digests at admission time and pinned. A skill cannot be swapped underneath a scheduled run.
- **Arguments live in a `Secret`, not in the spec.** This keeps potentially sensitive invocation payloads out of `kubectl get -o yaml` output and out of audit logs that record spec diffs.

### 1.2 Resource tiers

Authors select a tier rather than raw requests/limits. This keeps scheduling predictable and prevents a single skill from claiming a whole node.

| Tier | CPU request | CPU limit | Memory request | Memory limit | Ephemeral storage | Max wall clock |
|---|---|---|---|---|---|---|
| `micro` | 100m | 500m | 128Mi | 256Mi | 256Mi | 30s |
| `standard` | 500m | 2 | 512Mi | 1Gi | 2Gi | 300s |
| `heavy` | 2 | 4 | 2Gi | 8Gi | 16Gi | 1800s |
| `gpu-small` | 2 | 4 | 4Gi | 16Gi | 32Gi | 3600s |

Limits are hard. Memory overruns produce `OOMKilled` with `exitCode: 137`; wall-clock overruns produce `phase: TimedOut`. Neither is retried automatically — see §5.

Requesting a tier above `standard` requires an annotation on the skill manifest reviewed during registry onboarding. There is no runtime escalation path.

---

## 2. Pod construction

The operator builds each pod from a fixed template. Authors cannot inject arbitrary pod spec fields; the template is the contract.

### 2.1 Security context

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 65532
  runAsGroup: 65532
  fsGroup: 65532
  seccompProfile:
    type: RuntimeDefault
containers:
  - name: skill
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop: ["ALL"]
```

Consequences worth internalising:

- **Root filesystem is read-only.** Writable space is a single `emptyDir` mounted at `/workspace`, sized by tier. Skills that assume they can write to `/tmp` should use `/workspace/tmp`.
- **All capabilities dropped.** No `CAP_NET_RAW` means no raw sockets, no ICMP ping. No `CAP_SYS_PTRACE` means no attaching to other processes.
- **No privilege escalation.** `setuid` binaries in the image will not gain privileges.

### 2.2 Service account and token projection

Each run gets a dedicated `ServiceAccount` with no RBAC bindings. The default API token mount is disabled:

```yaml
automountServiceAccountToken: false
```

A skill that needs to talk to the Kubernetes API must be granted a scoped role during registry onboarding, which produces an explicit projected token with a short audience-bound lifetime. The absence of an ambient token is deliberate: it removes the most common lateral-movement path out of a compromised workload.

### 2.3 Scheduling constraints

```yaml
nodeSelector:
  agentspace.dev/pool: skill-runners
tolerations:
  - key: agentspace.dev/skill-runner
    operator: Exists
    effect: NoSchedule
topologySpreadConstraints:
  - maxSkew: 2
    topologyKey: kubernetes.io/hostname
    whenUnsatisfiable: ScheduleAnyway
    labelSelector:
      matchLabels:
        agentspace.dev/component: skill-run
```

Skill pods land only on a tainted, dedicated node pool. They never co-schedule with control-plane components, the registry API, or the operator itself. The spread constraint prevents a burst of runs from concentrating on one node and triggering correlated evictions.

---

## 3. Network policy

Network access is the single most consequential part of the execution envelope, so it is expressed as a small set of named profiles rather than per-run rules.

### 3.1 Profiles

| Profile | DNS | Cluster-internal | Internet egress | Typical use |
|---|---|---|---|---|
| `egress-none` | denied | denied | denied | Pure computation: parsing, formatting, transformation |
| `egress-dns-only` | allowed | denied | denied | Skills that resolve names for validation but do not connect |
| `egress-allowlist` | allowed | denied | allowlisted FQDNs, 443 only | Skills calling a specific known API |
| `egress-internal` | allowed | named services only | denied | Skills querying a first-party internal service |

`egress-none` is the default and should remain the default for the overwhelming majority of skills.

### 3.2 Generated policy

For `egress-none`:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: run-7f3a9b
  namespace: agentspace-runs
spec:
  podSelector:
    matchLabels:
      agentspace.dev/run-id: 7f3a9b
  policyTypes: ["Ingress", "Egress"]
  ingress: []
  egress: []
```

Empty `ingress` and `egress` arrays with both policy types declared is a **default-deny**, not an allow-all. This distinction is a recurring source of misconfiguration in hand-written policy; the operator generates it so authors never have to get it right by hand.

For `egress-allowlist`, the operator additionally programs the service mesh sidecar with the FQDN set, because `NetworkPolicy` alone cannot express hostname-based rules. The allowlist is declared in the skill manifest and frozen at the pinned digest — a skill cannot widen its own allowlist at runtime.

### 3.3 Ingress is always denied

No skill pod accepts inbound connections. Results are written to a result `Secret` and read by the operator; there is no listening socket, no `Service`, no `Ingress`. A skill that wants to expose an endpoint is not a skill — it is a service, and belongs in a different part of the platform.

---

## 4. Lifecycle and result handling

### 4.1 Phases

```
Pending ──> Running ──> Succeeded
   │            │
   │            ├──> Failed      (non-zero exit)
   │            ├──> TimedOut    (wall clock exceeded)
   │            └──> Evicted     (node pressure, preemption)
   │
   └──> Failed  (admission rejected, image pull failure, quota exhausted)
```

### 4.2 Result contract

The skill writes a single JSON document to `/workspace/result.json` before exiting zero. The operator reads it, validates it against the skill's declared output schema, and stores it in the result `Secret`.

Three failure modes are distinguished, because they call for different responses:

- **Exit zero, no `result.json`** → `Failed`, reason `ResultMissing`. Almost always a skill bug.
- **Exit zero, `result.json` fails schema validation** → `Failed`, reason `ResultInvalid`. The raw document is retained for 1 hour in a quarantine secret for debugging, then deleted.
- **Non-zero exit** → `Failed`, reason `NonZeroExit`, with `exitCode` recorded. The last 64KiB of stderr is captured into status; stdout is not, because skills sometimes write large payloads there.

### 4.3 Cleanup

Pods are deleted 60 seconds after reaching a terminal phase. `SkillRun` objects are garbage-collected after 24 hours. Result secrets expire on a TTL declared by the caller, capped at 7 days.

The 60-second grace window exists so that log shippers can drain. It is not a debugging window — for debugging, set `spec.invocation.retainPodSeconds` (max 900) explicitly, which is audited.

---

## 5. Failure handling and retries

**The operator does not retry.** This is a deliberate choice and worth stating plainly, because it inverts the default expectation for Kubernetes controllers.

The reasoning: a skill invocation may have side effects the operator cannot see. A skill with `egress-allowlist` that POSTs to an external API has already had an effect by the time it times out. Automatically re-running it risks duplicate writes. Since the operator cannot distinguish idempotent from non-idempotent skills — and self-declared idempotency is not trustworthy enough to build retry logic on — the safe default is to surface the failure and let the caller decide.

Callers that know their skill is idempotent implement retry themselves, with their own backoff, and their own duplicate-detection if that assumption turns out to be wrong.

Two exceptions, both pre-execution:

- **Image pull backoff** is retried by the kubelet as normal. Nothing has executed yet.
- **`Evicted` before the container starts** is retried once by the operator, because again, no code has run. If eviction happens after `Running` is reached, it is terminal.

---

## 6. Observability

### 6.1 Structured events

Every phase transition emits a Kubernetes `Event` and a structured log line:

```json
{
  "ts": "2025-11-04T09:12:51.204Z",
  "level": "info",
  "msg": "skillrun phase transition",
  "run_id": "7f3a9b",
  "skill": "pdf-extract",
  "version": "2.4.1",
  "digest": "sha256:9c1f…",
  "from": "Running",
  "to": "Succeeded",
  "duration_ms": 18104,
  "exit_code": 0,
  "peak_memory_bytes": 412663808,
  "network_profile": "egress-none"
}
```

Arguments and results are never logged. Only their secret references and sizes.

### 6.2 Metrics

```
agentspace_skillrun_duration_seconds{skill,version,phase}      histogram
agentspace_skillrun_total{skill,version,phase,reason}          counter
agentspace_skillrun_peak_memory_bytes{skill,version,tier}      histogram
agentspace_skillrun_queue_wait_seconds{tier}                   histogram
agentspace_skillrun_active{tier}                               gauge
agentspace_netpol_denied_total{skill,version,profile}          counter
```

`agentspace_netpol_denied_total` is the one to watch closely. A non-zero and growing value for a skill declared `egress-none` means the skill is attempting network access it was never granted. That is not necessarily malicious — a dependency may have added telemetry — but it always warrants investigation, because it means the skill's declared envelope and its actual behaviour have diverged.

### 6.3 Audit trail

The following are recorded immutably for 400 days:

- Admission decisions, including rejections and their reasons
- Digest resolution: which tag mapped to which digest at which time
- Any use of `retainPodSeconds`
- Any grant of a scoped Kubernetes API role to a skill
- Every `egress-allowlist` FQDN set, at the version it was frozen

---

## 7. Operator deployment

The operator itself runs under tighter constraints than the workloads it manages, on the principle that the thing holding the keys should be the most constrained component.

```yaml
replicas: 2                      # leader-elected, active/passive
resources:
  requests: {cpu: 200m, memory: 256Mi}
  limits:   {cpu: 1,    memory: 512Mi}
securityContext:
  runAsNonRoot: true
  readOnlyRootFilesystem: true
  capabilities: {drop: ["ALL"]}
```

RBAC is scoped to exactly what reconciliation requires:

| Resource | Verbs | Namespace |
|---|---|---|
| `skillruns` | get, list, watch, update, patch | `agentspace-runs` |
| `skillruns/status` | update, patch | `agentspace-runs` |
| `pods` | get, list, watch, create, delete | `agentspace-runs` |
| `secrets` | get, create, delete | `agentspace-runs` |
| `networkpolicies` | get, create, delete | `agentspace-runs` |
| `serviceaccounts` | get, create, delete | `agentspace-runs` |
| `events` | create, patch | `agentspace-runs` |

Notably absent: `secrets` `list` and `watch`. The operator reads secrets only by name, derived from the `SkillRun` it is reconciling. It cannot enumerate the namespace's secrets. This costs a little efficiency and removes a broad read primitive from a component that handles invocation arguments.

Also absent: any cluster-scoped permission beyond reading the CRD definitions themselves.

---

## 8. Constraints and non-goals

Stated explicitly so authors do not design against capabilities that do not exist:

- **No persistent storage.** `/workspace` is destroyed with the pod. Skills needing durable state write to an external store via `egress-allowlist`.
- **No inter-skill communication.** Skills cannot discover or address each other. Composition happens at the caller layer.
- **No inbound network.** See §3.3.
- **No privileged operations.** No host mounts, no host network, no host PID, no privileged containers, no arbitrary sysctls. There is no exception process for these.
- **No runtime capability escalation.** Every permission — tier, network profile, API role, allowlist — is fixed at the pinned digest. A running skill cannot request more.
- **Single container per run.** No sidecars from the author's manifest. The mesh sidecar, where present, is operator-injected and not author-configurable.

---

## 9. Skill manifest reference

The fields this runtime reads:

```yaml
runtime:
  kind: kubernetes
  tier: standard                  # §1.2; default: standard
  network:
    profile: egress-none          # §3.1; default: egress-none
    allowlist: []                 # required iff profile is egress-allowlist
  timeoutSeconds: 300             # capped by tier max
  outputSchema: ./schema/result.json   # required
  kubernetesApi:                  # optional; requires onboarding review
    role: null
```

Admission rejects a manifest that:

- omits `outputSchema`
- sets `timeoutSeconds` above the tier maximum
- declares `egress-allowlist` with an empty or wildcard allowlist
- requests a tier above `standard` without the onboarding annotation
- requests `kubernetesApi.role` without a corresponding approved grant

---

## 10. Worked example

A skill that extracts text from PDFs. No network, modest memory, short timeout.

```yaml
---
name: "pdf-extract"
description: "Extract plain text and structural outline from a PDF document."
version: "2.4.1"
runtime:
  kind: kubernetes
  tier: standard
  network:
    profile: egress-none
  timeoutSeconds: 120
  outputSchema: ./schema/result.json
---
```

Reconciled envelope:

- 500m CPU requested, 2 CPU ceiling; 512Mi requested, 1Gi ceiling
- Read-only root, writable `/workspace` at 2Gi
- Default-deny `NetworkPolicy`, both directions
- No service account token
- Killed at 120s → `TimedOut`
- Result read from `/workspace/result.json`, schema-validated, stored in a TTL'd secret
- Pod deleted 60s after terminal phase

The skill's own code needs no awareness of any of this. It reads its arguments, writes `result.json`, exits zero. The envelope is the operator's job.

---

## Appendix A: `exitCode` reference

| Code | Meaning | Phase |
|---|---|---|
| 0 | Success (requires valid `result.json`) | `Succeeded` |
| 1–125 | Skill-defined error | `Failed` |
| 126 | Command not executable | `Failed` |
| 127 | Command not found | `Failed` |
| 137 | SIGKILL — almost always OOM | `Failed` |
| 139 | SIGSEGV | `Failed` |
| 143 | SIGTERM — graceful timeout shutdown | `TimedOut` |

## Appendix B: Version compatibility

| Runtime | Cluster | CRD | Status |
|---|---|---|---|
| 1.0.x | 1.27–1.31 | `v1` | Supported |
| 0.9.x | 1.26–1.29 | `v1beta1` | Deprecated; EOL 2026-03 |
| 0.8.x | 1.24–1.27 | `v1alpha1` | Unsupported |

Migrating `v1beta1` → `v1`: the `spec.resources.cpu`/`spec.resources.memory` pair is replaced by `spec.resources.tier`. The operator translates existing objects on first reconcile and records the mapping in an event. Manifests should be updated before the EOL date; there is no translation shim in 1.1.
