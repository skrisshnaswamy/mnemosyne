"""Diagrams for the LLM agents / frameworks / production notes.

Same house style as llm_diagrams.py. Outputs PNGs into ../References/.
Every function print()s the numbers the notes quote, so the prose and the
picture can never disagree. Run with Homebrew python3 from inside sourcecode/.
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "../References/"
BG, FG, GRID = "#ffffff", "#2b2b2b", "#d8d8d8"
BLUE, ORANGE, GREEN, RED, GREY = "#3b7dd8", "#e08a1e", "#2ea36c", "#d64545", "#9aa0a6"
PURPLE = "#8e6fd8"
rng = np.random.default_rng(7)


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, color=FG, pad=12)
    ax.set_xlabel(xlabel, fontsize=11, color=FG)
    ax.set_ylabel(ylabel, fontsize=11, color=FG)
    ax.grid(True, linestyle='--', alpha=.5, color=GRID)
    ax.tick_params(colors=FG, labelsize=9)
    for s in ax.spines.values(): s.set_color(GRID)


def save(fig, name):
    fig.savefig(OUT + name, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig); print("  ✓", name)


# ───────────────────────────────────────────────────────────── 1
def compounding_reliability():
    """Per-step reliability over a horizon, and what a verifier buys you."""
    print("\n[1] compounding reliability")
    steps = np.arange(1, 41)

    def simulate(p, n, k, detect, retries, trials=40000):
        """k = verify every k steps; detect = P(verifier notices a bad block)."""
        ok = 0
        for _ in range(trials):
            good = True
            for start in range(0, n, k):
                block = min(k, n - start)
                for _attempt in range(retries + 1):
                    failed = rng.random(block) > p
                    if not failed.any():
                        break                      # block clean
                    if rng.random() > detect:
                        good = False; break        # slipped past the verifier
                else:
                    good = False                   # out of retries
                if not good: break
            ok += good
        return ok / trials

    naive95 = 0.95 ** steps
    naive99 = 0.99 ** steps
    checked = np.array([simulate(0.95, int(n), 5, 0.80, 1, 6000) for n in steps])

    fig, ax = plt.subplots(figsize=(7.4, 4.4), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(steps, naive95 * 100, color=RED, lw=2.4, label="95% per step, no checks")
    ax.plot(steps, naive99 * 100, color=ORANGE, lw=2.2, label="99% per step, no checks")
    ax.plot(steps, checked * 100, color=GREEN, lw=2.4,
            label="95% per step + verify every 5 (catches 80%, one retry)")
    ax.axvline(20, color=GREY, ls=':', lw=1.4)
    ax.annotate(f"20 steps: {naive95[19]*100:.0f}%", xy=(20, naive95[19]*100),
                xytext=(22, 18), color=RED, fontsize=9.5,
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))
    ax.annotate(f"same model, checked: {checked[19]*100:.0f}%", xy=(20, checked[19]*100),
                xytext=(21, 88), color=GREEN, fontsize=9.5,
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2))
    style(ax, "A reliable step is not a reliable run", "Steps in the run", "P(whole run correct) %")
    ax.set_ylim(0, 105); ax.legend(fontsize=8.5, loc='lower left')
    save(fig, "agent_compounding_reliability.png")
    print(f"    20 steps @95%: {naive95[19]*100:.1f}%   @99%: {naive99[19]*100:.1f}%"
          f"   @95%+verifier: {checked[19]*100:.1f}%")
    print(f"    40 steps @95%: {naive95[39]*100:.1f}%   @95%+verifier: {checked[39]*100:.1f}%")


# ───────────────────────────────────────────────────────────── 2
def agent_token_spend():
    """Token accounting for a 20-step agent run. Re-sent context dominates."""
    print("\n[2] where an agent run's tokens go")
    SYS, TOOLS, QUERY = 1200, 2800, 150
    THOUGHT, RESULT = 180, 700           # output per step, tool result per step
    steps = np.arange(1, 21)

    fresh, resent, out = [], [], []
    history = 0
    for _ in steps:
        ctx = SYS + TOOLS + QUERY + history          # everything re-sent this step
        new = THOUGHT + RESULT                       # what this step adds
        resent.append(ctx - 0)
        fresh.append(new)
        out.append(THOUGHT)
        history += new
    resent = np.array(resent); fresh = np.array(fresh); out = np.array(out)

    total_in = resent.sum()
    total_out = out.sum()
    # what you'd pay if nothing were re-sent (an impossible lower bound)
    floor_in = SYS + TOOLS + QUERY + (RESULT * 20)
    PIN, POUT = 3.0 / 1e6, 15.0 / 1e6
    cost = total_in * PIN + total_out * POUT

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.2, 4.4), facecolor=BG,
                                 gridspec_kw={'width_ratios': [1.35, 1]})
    a1.set_facecolor(BG)
    a1.bar(steps, resent / 1000, color=BLUE, label="context re-sent this step")
    a1.bar(steps, out / 1000, bottom=resent / 1000, color=ORANGE, label="tokens generated this step")
    a1.axhline(20, color=GREY, ls='--', lw=1.2)
    a1.text(0.6, 20.6, "20k tokens", color=GREY, fontsize=9)
    style(a1, "Every step pays for every step before it", "Agent step", "Tokens billed (thousands)")
    a1.legend(fontsize=9, loc='upper left'); a1.set_ylim(0, 26)

    a2.set_facecolor(BG)
    labels = ["Unavoidable\n(prompt + tool\nresults, once)", "Re-sent\ncontext", "Generated\ntokens"]
    vals = [floor_in / 1000, (total_in - floor_in) / 1000, total_out / 1000]
    cols = [GREEN, RED, ORANGE]
    bars = a2.bar(labels, vals, color=cols)
    for b, v in zip(bars, vals):
        a2.text(b.get_x() + b.get_width() / 2, v + 4, f"{v:.0f}k", ha='center',
                color=FG, fontsize=10)
    style(a2, "The bill is mostly the same words, again", "", "Tokens (thousands)")
    a2.tick_params(labelsize=8.5); a2.set_ylim(0, max(vals) * 1.18)
    save(fig, "agent_run_token_spend.png")
    print(f"    input tokens billed: {total_in:,}   output: {total_out:,}")
    print(f"    unavoidable floor:   {floor_in:,}  → re-sent share = "
          f"{(total_in-floor_in)/total_in*100:.0f}% of input")
    print(f"    cost @ $3/$15 per M: ${cost:.3f} per run; final context = "
          f"{SYS+TOOLS+QUERY+20*(THOUGHT+RESULT):,} tokens")
    return total_in, total_out


# ───────────────────────────────────────────────────────────── 3
def prompt_cache_savings():
    """Same 20-step run with and without a stable cacheable prefix."""
    print("\n[3] prompt caching")
    SYS, TOOLS, QUERY, THOUGHT, RESULT = 1200, 2800, 150, 180, 700
    PIN, PCW, PCR, POUT = 3.0 / 1e6, 3.75 / 1e6, 0.30 / 1e6, 15.0 / 1e6
    steps = np.arange(1, 21)

    def run(cacheable_prefix):
        """cacheable_prefix = how many leading tokens are byte-identical each step."""
        hist, costs, written = 0, [], 0
        for i in steps:
            ctx = SYS + TOOLS + QUERY + hist
            cached = min(cacheable_prefix, ctx)
            fresh = ctx - cached
            c = fresh * PIN + cached * (PCW if i == 1 else PCR) + THOUGHT * POUT
            costs.append(c); hist += THOUGHT + RESULT; written = cached
        return np.cumsum(costs)

    nocache = run(0)
    static = run(SYS + TOOLS)                       # system + tools stable
    # growing prefix: cache breakpoint moves with the transcript
    hist, growing = 0, []
    for i in steps:
        ctx = SYS + TOOLS + QUERY + hist
        cached = SYS + TOOLS + QUERY + hist          # everything but this step's new text
        cached = ctx if i > 1 else 0
        fresh = ctx - cached
        new_written = (THOUGHT + RESULT) if i > 1 else ctx
        growing.append(fresh * PIN + (cached - new_written) * PCR +
                       new_written * PCW + THOUGHT * POUT)
        hist += THOUGHT + RESULT
    growing = np.cumsum(growing)

    fig, ax = plt.subplots(figsize=(7.6, 4.4), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(steps, nocache, color=RED, lw=2.4, label="no cache (or a timestamp in the system prompt)")
    ax.plot(steps, static, color=ORANGE, lw=2.2, label="cache the system prompt + tool defs")
    ax.plot(steps, growing, color=GREEN, lw=2.4, label="cache breakpoint moved down the transcript")
    ax.annotate(f"${nocache[-1]:.3f}", xy=(20, nocache[-1]), xytext=(15.2, nocache[-1]*1.06),
                color=RED, fontsize=10)
    ax.annotate(f"${growing[-1]:.3f}", xy=(20, growing[-1]), xytext=(15.2, growing[-1]*0.55),
                color=GREEN, fontsize=10)
    style(ax, "A stable prefix is worth more than a smaller prompt",
          "Agent step", "Cumulative cost of the run ($)")
    ax.legend(fontsize=8.5, loc='upper left')
    save(fig, "prompt_cache_savings.png")
    print(f"    no cache ${nocache[-1]:.4f} | static prefix ${static[-1]:.4f} "
          f"({(1-static[-1]/nocache[-1])*100:.0f}% off) | rolling breakpoint ${growing[-1]:.4f} "
          f"({(1-growing[-1]/nocache[-1])*100:.0f}% off)")


# ───────────────────────────────────────────────────────────── 4
def retry_storm():
    """What a naive retry buys (fewer failures) and what it costs (p99, bill)."""
    print("\n[4] retry storms")
    rates = np.linspace(0.01, 0.30, 15)
    TIMEOUT, TRIALS = 30.0, 15000

    def sim(err, max_attempts):
        lat = np.zeros(TRIALS); attempts = np.zeros(TRIALS); failed = np.zeros(TRIALS)
        for i in range(TRIALS):
            t, n, ok = 0.0, 0, False
            for a in range(max_attempts):
                n += 1
                if rng.random() < err:
                    t += TIMEOUT                       # a failure is a hung request
                else:
                    t += float(rng.lognormal(np.log(2.0), 0.45)); ok = True; break
            lat[i] = t; attempts[i] = n; failed[i] = not ok
        return np.percentile(lat, 50), np.percentile(lat, 99), attempts.mean(), failed.mean()

    one = np.array([sim(e, 1) for e in rates])
    three = np.array([sim(e, 3) for e in rates])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.2, 4.3), facecolor=BG)
    a1.set_facecolor(BG)
    a1.plot(rates * 100, three[:, 1], color=RED, lw=2.4, label="p99 — retry up to 3×")
    a1.plot(rates * 100, one[:, 1], color=ORANGE, lw=2.2, ls='--', label="p99 — no retry")
    a1.plot(rates * 100, three[:, 0], color=BLUE, lw=2.2, label="p50 — retry up to 3×")
    a1.axhline(TIMEOUT, color=GREY, ls=':', lw=1.4)
    a1.text(1.5, TIMEOUT + 1.5, "one 30s timeout", color=GREY, fontsize=9)
    style(a1, "Retries turn a failure into three timeouts", "Upstream error rate (%)", "Latency (s)")
    a1.legend(fontsize=8.5, loc='center left')

    a2.set_facecolor(BG)
    a2.plot(rates * 100, one[:, 3] * 100, color=GREY, lw=2.2, ls=':',
            label="requests that fail — no retry")
    a2.plot(rates * 100, three[:, 3] * 100, color=GREEN, lw=2.4,
            label="requests that fail — retry up to 3×")
    a2.plot(rates * 100, (three[:, 2] - 1) * 100, color=RED, lw=2.4,
            label="extra calls billed (% over one per request)")
    i20 = np.argmin(abs(rates - 0.20))
    a2.annotate(f"at 20% errors: failures {one[i20,3]*100:.0f}% → {three[i20,3]*100:.1f}%,\n"
                f"but you pay {three[i20,2]:.2f} calls per request",
                xy=(20, (three[i20, 2] - 1) * 100), xytext=(6, 26), color=FG, fontsize=9,
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1.1))
    style(a2, "Retries buy reliability with money and tail latency",
          "Upstream error rate (%)", "Percent")
    a2.legend(fontsize=8.5, loc='upper left')
    save(fig, "retry_storm_cost.png")
    for e in (0.05, 0.20):
        i = np.argmin(abs(rates - e))
        print(f"    err {e*100:.0f}%: no-retry p99 {one[i,1]:.1f}s fail {one[i,3]*100:.1f}%  |  "
              f"retry3 p50 {three[i,0]:.1f}s p99 {three[i,1]:.1f}s "
              f"attempts {three[i,2]:.2f} fail {three[i,3]*100:.2f}%")


# ───────────────────────────────────────────────────────────── 5
def parallel_tools():
    """Sequential vs parallel tool calls on wall-clock."""
    print("\n[5] parallel tool calls")
    ns = np.arange(1, 9); TRIALS = 20000
    MODEL_TURN = 1.4                      # a round trip to the model between batches
    seq_p50, par_p50 = [], []
    for n in ns:
        d = rng.lognormal(np.log(0.8), 0.55, size=(TRIALS, n))
        seq = d.sum(axis=1) + n * MODEL_TURN          # one model turn per call
        par = d.max(axis=1) + MODEL_TURN              # one turn, n calls at once
        seq_p50.append(np.percentile(seq, 50)); par_p50.append(np.percentile(par, 50))
    seq_p50 = np.array(seq_p50); par_p50 = np.array(par_p50)

    fig, ax = plt.subplots(figsize=(7.2, 4.3), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(ns, seq_p50, color=RED, lw=2.4, marker='o', ms=5, label="one call per turn (sequential)")
    ax.plot(ns, par_p50, color=GREEN, lw=2.4, marker='o', ms=5, label="all independent calls in one turn")
    ax.annotate(f"{seq_p50[3]:.1f}s vs {par_p50[3]:.1f}s at 4 tools",
                xy=(4, seq_p50[3]), xytext=(4.2, seq_p50[3] + 2.5), color=FG, fontsize=9.5,
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1.1))
    style(ax, "Parallel tool calls are the cheapest latency win in an agent",
          "Independent tool calls needed", "Median wall-clock (s)")
    ax.legend(fontsize=9)
    save(fig, "parallel_vs_sequential_tools.png")
    print(f"    4 tools: sequential {seq_p50[3]:.2f}s vs parallel {par_p50[3]:.2f}s "
          f"({seq_p50[3]/par_p50[3]:.1f}×)")
    print(f"    8 tools: sequential {seq_p50[7]:.2f}s vs parallel {par_p50[7]:.2f}s "
          f"({seq_p50[7]/par_p50[7]:.1f}×)")


# ───────────────────────────────────────────────────────────── 6
def chunk_size_recall():
    """Synthetic doc + answer span. Small chunks split it; big chunks dilute it."""
    print("\n[6] chunk size vs answering recall")
    DOC, K, TRIALS = 20000, 5, 4000
    sizes = [64, 128, 256, 512, 1024, 2048]

    def trial(S, ov_frac):
        stride = max(1, int(S * (1 - ov_frac)))
        starts = np.arange(0, DOC - S + 1, stride)
        L = rng.integers(60, 260)                     # the answer span, in words
        a = rng.integers(0, DOC - L)
        b = a + L
        lo = np.maximum(starts, a); hi = np.minimum(starts + S, b)
        overlap = np.maximum(0, hi - lo)              # relevant words per chunk
        # a bi-encoder scores a chunk by how much of it is on-topic, plus noise
        score = overlap / S + rng.normal(0, 0.11, size=len(starts))
        top = np.argsort(-score)[:K]
        covered = np.zeros(L, dtype=bool)
        for t in top:
            s0, s1 = starts[t], starts[t] + S
            lo2, hi2 = max(s0, a), min(s1, b)
            if hi2 > lo2: covered[lo2 - a: hi2 - a] = True
        return covered.mean() >= 0.99, (overlap[top] > 0).any()

    res = {}
    for ov in (0.0, 0.25):
        full, anyhit = [], []
        for S in sizes:
            f = np.mean([trial(S, ov)[0] for _ in range(TRIALS)])
            h = np.mean([trial(S, ov)[1] for _ in range(TRIALS)])
            full.append(f * 100); anyhit.append(h * 100)
        res[ov] = (np.array(full), np.array(anyhit))

    fig, ax = plt.subplots(figsize=(7.6, 4.4), facecolor=BG); ax.set_facecolor(BG)
    x = np.arange(len(sizes))
    ax.plot(x, res[0.0][1], color=GREY, lw=2.0, ls='--', marker='s', ms=4,
            label="retrieved *something* relevant (no overlap)")
    ax.plot(x, res[0.0][0], color=RED, lw=2.4, marker='o', ms=5,
            label="retrieved the **whole** answer (no overlap)")
    ax.plot(x, res[0.25][0], color=GREEN, lw=2.4, marker='o', ms=5,
            label="retrieved the whole answer (25% overlap)")
    best = int(np.argmax(res[0.25][0]))
    ax.axvline(best, color=GREY, ls=':', lw=1.4)
    ax.annotate(f"best: {sizes[best]} words", xy=(best, res[0.25][0][best]),
                xytext=(best + 0.15, res[0.25][0][best] - 22), color=FG, fontsize=9.5,
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1.1))
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in sizes])
    style(ax, "The chunk that scores best is often not the chunk that answers",
          "Chunk size (words), top-5 retrieved", "% of questions fully answerable")
    ax.legend(fontsize=8.5, loc='lower center'); ax.set_ylim(0, 105)
    save(fig, "chunk_size_vs_recall.png")
    for i, S in enumerate(sizes):
        print(f"    {S:>5} words: whole-answer {res[0.25][0][i]:5.1f}%   "
              f"any-hit {res[0.25][1][i]:5.1f}%  (no-overlap whole {res[0.0][0][i]:5.1f}%)")


# ───────────────────────────────────────────────────────────── 7
def context_growth():
    """Four ways of handling a context that grows every turn."""
    print("\n[7] context growth and compaction")
    turns = np.arange(0, 61)
    BASE, PER = 4150, 2400                   # system+tools+query, and per-turn growth
    LIMIT = 128000
    naive = BASE + PER * turns

    def compacted(every, ratio):
        out, cur = [], BASE
        for t in turns:
            cur += PER
            if t > 0 and t % every == 0:
                cur = BASE + (cur - BASE) * ratio
            out.append(cur)
        return np.array(out)

    summarise = compacted(10, 0.25)
    trunc_tools = BASE + (PER * 0.38) * turns          # truncate tool results to ~2k chars
    window = np.minimum(naive, BASE + PER * 12)        # keep last 12 turns only

    fig, ax = plt.subplots(figsize=(7.8, 4.5), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(turns, naive / 1000, color=RED, lw=2.4, label="append everything")
    ax.plot(turns, trunc_tools / 1000, color=ORANGE, lw=2.2, label="truncate tool results")
    ax.plot(turns, summarise / 1000, color=GREEN, lw=2.4, label="compact every 10 turns")
    ax.plot(turns, window / 1000, color=BLUE, lw=2.2, ls='--', label="sliding window of 12 turns")
    ax.axhline(LIMIT / 1000, color=GREY, ls='--', lw=1.5)
    ax.text(1, LIMIT / 1000 + 1.5, "128k window", color=GREY, fontsize=9)
    hit = int(np.argmax(naive > LIMIT)) if (naive > LIMIT).any() else -1
    style(ax, "An agent that appends everything has a deadline",
          "Turn in the run", "Tokens in the window (thousands)")
    ax.legend(fontsize=8.5, loc='upper left'); ax.set_ylim(0, 165)
    save(fig, "context_growth_compaction.png")
    print(f"    append-everything hits 128k at turn {hit if hit>0 else '>60'}; "
          f"at turn 40 it is {naive[40]/1000:.0f}k vs compacted {summarise[40]/1000:.0f}k")
    print(f"    cost of turn 40 at $3/M input: append ${naive[40]*3/1e6:.4f} vs "
          f"compact ${summarise[40]*3/1e6:.4f}")


# ───────────────────────────────────────────────────────────── 8
def eval_sample_size():
    """Wilson interval width vs number of eval cases."""
    print("\n[8] eval sample size")
    ns = np.arange(10, 1001)
    z, p = 1.96, 0.80

    def wilson(p, n):
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return c - h, c + h

    lo, hi = wilson(p, ns)
    width = (hi - lo) * 100

    fig, ax = plt.subplots(figsize=(7.4, 4.3), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(ns, width, color=BLUE, lw=2.4)
    for n, c in [(30, RED), (100, ORANGE), (400, GREEN)]:
        w = (wilson(p, n)[1] - wilson(p, n)[0]) * 100
        ax.plot([n], [w], 'o', color=c, ms=7)
        ax.annotate(f"n={n}: ±{w/2:.1f} pts", xy=(n, w), xytext=(n * 1.25, w + 1.4),
                    color=c, fontsize=9.5)
    ax.axhline(10, color=GREY, ls='--', lw=1.4)
    ax.text(11, 10.6, "±5 pts — the smallest change worth shipping on", color=GREY, fontsize=9)
    style(ax, "30 cases cannot tell you whether you improved anything",
          "Eval cases (pass rate 80%)", "95% confidence interval width (pts)")
    ax.set_xscale('log'); ax.set_ylim(0, 35)
    save(fig, "eval_sample_size_ci.png")
    for n in (30, 50, 100, 200, 400, 1000):
        l, h = wilson(p, n); print(f"    n={n:>4}: 80% ± {(h-l)/2*100:.1f} pts  → [{l*100:.0f}, {h*100:.0f}]")


# ───────────────────────────────────────────────────────────── 9
def model_routing():
    """Cheap-first with escalation vs always sending the big model."""
    print("\n[9] model routing")
    N = 60000
    diff = rng.beta(2.2, 3.0, N)                    # 0 = easy, 1 = hard
    small_ok = rng.random(N) < np.clip(0.99 - 1.05 * diff ** 2, 0.05, 0.99)
    big_ok = rng.random(N) < np.clip(0.995 - 0.16 * diff, 0.40, 0.995)
    conf = np.clip(1 - diff + rng.normal(0, 0.07, N), 0, 1)   # small model's noisy self-report
    C_S, C_B = 0.0012, 0.0180                       # $ per request

    ths = np.linspace(0, 1, 101)
    costs, accs, escs = [], [], []
    for t in ths:
        esc = conf < t
        acc = np.where(esc, big_ok, small_ok).mean()
        cost = (C_S + esc * C_B).mean()
        costs.append(cost * 1000); accs.append(acc * 100); escs.append(esc.mean())
    costs, accs, escs = np.array(costs), np.array(accs), np.array(escs)

    fig, ax = plt.subplots(figsize=(7.4, 4.4), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(costs, accs, color=BLUE, lw=2.4, label="cheap model first, escalate when unsure")
    ax.plot([C_B * 1000], [big_ok.mean() * 100], 'o', color=RED, ms=9)
    ax.annotate(f"always big\n${C_B*1000:.1f} / 1k · {big_ok.mean()*100:.1f}%",
                xy=(C_B * 1000, big_ok.mean() * 100), xytext=(C_B * 1000 - 7.5, big_ok.mean()*100 - 4),
                color=RED, fontsize=9.5)
    ax.plot([C_S * 1000], [small_ok.mean() * 100], 'o', color=GREEN, ms=9)
    ax.annotate(f"always cheap\n${C_S*1000:.1f} / 1k · {small_ok.mean()*100:.1f}%",
                xy=(C_S * 1000, small_ok.mean() * 100), xytext=(C_S * 1000 + 0.8, small_ok.mean()*100 + 1),
                color=GREEN, fontsize=9.5)
    # the operating point where you escalate 30% of traffic
    knee = int(np.argmin(abs(escs - 0.30)))
    ax.plot([costs[knee]], [accs[knee]], '*', color=ORANGE, ms=17)
    ax.annotate(f"escalate the hardest 30%\n${costs[knee]:.1f} / 1k · {accs[knee]:.1f}%",
                xy=(costs[knee], accs[knee]), xytext=(costs[knee] + 0.6, accs[knee] - 9),
                color=ORANGE, fontsize=9.5,
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.1))
    style(ax, "Most requests do not need your best model",
          "Cost per 1,000 requests ($)", "Accuracy (%)")
    ax.legend(fontsize=9, loc='lower right')
    save(fig, "model_routing_cost_curve.png")
    print(f"    always cheap: ${C_S*1000:.2f}/1k at {small_ok.mean()*100:.1f}%")
    print(f"    always big:   ${C_B*1000:.2f}/1k at {big_ok.mean()*100:.1f}%")
    print(f"    escalate 30%: ${costs[knee]:.2f}/1k at {accs[knee]:.1f}% "
          f"({C_B*1000/costs[knee]:.1f}× cheaper than always-big, "
          f"{big_ok.mean()*100-accs[knee]:.1f} pts down)")
    for frac in (0.1, 0.2, 0.3, 0.5, 0.7):
        j = int(np.argmin(abs(escs - frac)))
        print(f"      escalate {frac*100:>3.0f}%: ${costs[j]:5.2f}/1k  acc {accs[j]:.1f}%")


# ───────────────────────────────────────────────────────────── 10
def streaming_wait():
    """Perceived wait with and without streaming."""
    print("\n[10] streaming")
    n = np.arange(0, 1201, 10)
    TTFT, TPS, READ = 0.65, 55.0, 4.5        # s, tokens/s generated, tokens/s read
    blocking = TTFT + n / TPS
    streamed = np.full_like(n, TTFT, dtype=float)
    read_time = n / READ

    fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(n, blocking, color=RED, lw=2.4, label="wait before anything appears (no streaming)")
    ax.plot(n, streamed, color=GREEN, lw=2.4, label="wait before the first word (streaming)")
    ax.axvline(600, color=GREY, ls=':', lw=1.4)
    ax.annotate(f"600-token answer:\n{blocking[60]:.1f}s of blank screen\nvs {TTFT:.2f}s",
                xy=(600, blocking[60]), xytext=(640, blocking[60] - 3.2), color=FG, fontsize=9.5,
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1.1))
    style(ax, "Streaming does not make it faster — it makes the wait disappear",
          "Answer length (tokens)", "Seconds")
    ax.text(30, 20.5, f"a reader gets through {READ:.1f} tokens/s.\n"
            f"the model writes {TPS:.0f}. once the first word is on screen,\n"
            f"the reader never catches up.", color=GREY, fontsize=9.5)
    ax.legend(fontsize=8.5, loc='center left'); ax.set_ylim(0, 25)
    save(fig, "streaming_perceived_wait.png")
    print(f"    600 tokens: blocking wait {blocking[60]:.1f}s vs TTFT {TTFT}s; "
          f"generation {600/TPS:.1f}s, human reading {600/READ:.0f}s")
    print(f"    generation outruns reading whenever TPS ({TPS}) > reading ({READ}) — "
          f"{TPS/READ:.0f}× headroom")


# ───────────────────────────────────────────────────────────── 11
def latency_waterfall():
    """Where the seconds go in one agent step."""
    print("\n[11] latency waterfall")
    stages = ["Network + queue", "Prefill (18k ctx)", "Decode (180 tok)",
              "Tool execution", "Re-prefill next step"]
    secs = [0.12, 0.64, 3.27, 0.81, 0.68]
    cols = [GREY, BLUE, RED, ORANGE, BLUE]

    fig, ax = plt.subplots(figsize=(8.4, 3.2), facecolor=BG); ax.set_facecolor(BG)
    left = 0
    for s, v, c in zip(stages, secs, cols):
        ax.barh([0], [v], left=left, color=c, height=.5)
        if v > 0.5:
            ax.text(left + v / 2, 0, f"{s}\n{v:.2f}s", ha='center', va='center',
                    fontsize=8.5, color='white')
        left += v
    ax.set_xlim(0, sum(secs) * 1.02); ax.set_yticks([]); ax.set_ylim(-0.75, 0.4)
    ax.set_xlabel("seconds — one step of an agent loop", color=FG, fontsize=10)
    ax.set_title("Decode owns the clock; the tool call almost never does",
                 fontsize=12.5, color=FG, pad=12)
    ax.tick_params(colors=FG, labelsize=9)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(sum(secs) * 0.5, -0.52,
            f"one step = {sum(secs):.2f}s  →  a 20-step run = {sum(secs)*20/60:.1f} minutes",
            color=RED, fontsize=10.5, ha='center')
    save(fig, "agent_step_latency_waterfall.png")
    print(f"    one step {sum(secs):.2f}s; decode is {secs[2]/sum(secs)*100:.0f}% of it; "
          f"20 steps = {sum(secs)*20:.0f}s")


# ───────────────────────────────────────────────────────────── 12
def handoff_loss():
    """Facts surviving a chain of summarising handoffs vs shared state."""
    print("\n[12] multi-agent handoffs")
    hs = np.arange(0, 7)
    FACTS = 8
    TRIALS = 40000
    for_plot = {}
    for r in (0.98, 0.94, 0.88):
        succ = []
        for h in hs:
            alive = np.ones((TRIALS, FACTS), dtype=bool)
            for _ in range(h):
                alive &= rng.random((TRIALS, FACTS)) < r
            succ.append(alive.all(axis=1).mean() * 100)
        for_plot[r] = np.array(succ)

    fig, ax = plt.subplots(figsize=(7.4, 4.3), facecolor=BG); ax.set_facecolor(BG)
    for r, c in zip((0.98, 0.94, 0.88), (GREEN, ORANGE, RED)):
        ax.plot(hs, for_plot[r], color=c, lw=2.4, marker='o', ms=5,
                label=f"each handoff keeps a fact with p = {r}")
    ax.axhline(100, color=GREY, ls='--', lw=1.5)
    ax.text(0.1, 101, "shared state — nothing is re-summarised", color=GREY, fontsize=9)
    ax.annotate(f"3 handoffs @0.94: {for_plot[0.94][3]:.0f}%",
                xy=(3, for_plot[0.94][3]), xytext=(3.2, for_plot[0.94][3] + 14),
                color=ORANGE, fontsize=9.5, arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.1))
    style(ax, "Every handoff is a lossy compression of the task",
          "Handoffs between agents", "% of runs where all 8 facts survive")
    ax.legend(fontsize=8.5); ax.set_ylim(0, 112)
    save(fig, "multi_agent_handoff_loss.png")
    for r in (0.98, 0.94, 0.88):
        print(f"    p={r}: 1 handoff {for_plot[r][1]:.0f}%  3 handoffs {for_plot[r][3]:.0f}%  "
              f"6 handoffs {for_plot[r][6]:.0f}%")


# ───────────────────────────────────────────────────────────── 13
def reflection_vs_verifier():
    """Self-critique with no external signal vs an actual verifier."""
    print("\n[13] reflection")
    rounds = np.arange(0, 6); TRIALS = 60000

    def sim(detect, false_alarm, fix_rate):
        acc = []
        correct = rng.random(TRIALS) < 0.55
        acc.append(correct.mean() * 100)
        cur = correct.copy()
        for _ in rounds[1:]:
            flagged = np.where(cur, rng.random(TRIALS) < false_alarm,
                               rng.random(TRIALS) < detect)
            revised = flagged & (rng.random(TRIALS) < fix_rate)
            # a flagged-and-revised item becomes correct if it was wrong, and
            # can break if it was right
            cur = np.where(flagged & revised, ~cur, cur)
            acc.append(cur.mean() * 100)
        return np.array(acc)

    selfcrit = sim(detect=0.33, false_alarm=0.22, fix_rate=0.70)
    verifier = sim(detect=0.96, false_alarm=0.02, fix_rate=0.70)

    fig, ax = plt.subplots(figsize=(7.4, 4.3), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(rounds, selfcrit, color=RED, lw=2.4, marker='o', ms=5,
            label="self-critique (the same model grades itself)")
    ax.plot(rounds, verifier, color=GREEN, lw=2.4, marker='o', ms=5,
            label="external verifier (tests, schema, compiler)")
    ax.axhline(selfcrit[0], color=GREY, ls='--', lw=1.4)
    ax.text(0.08, selfcrit[0] + 1.2, "no reflection at all", color=GREY, fontsize=9)
    style(ax, "Reflection is only as good as the signal behind it",
          "Rounds of critique-and-revise", "Accuracy (%)")
    ax.legend(fontsize=8.5, loc='center right'); ax.set_ylim(40, 100)
    save(fig, "reflection_vs_verifier.png")
    print(f"    baseline {selfcrit[0]:.1f}%")
    print(f"    self-critique after 1/3/5 rounds: {selfcrit[1]:.1f}% / {selfcrit[3]:.1f}% / {selfcrit[5]:.1f}%")
    print(f"    verifier     after 1/3/5 rounds: {verifier[1]:.1f}% / {verifier[3]:.1f}% / {verifier[5]:.1f}%")


# ───────────────────────────────────────────────────────────── 14
def checkpoint_interval():
    """Expected re-executed work after a crash, vs how often you persist state."""
    print("\n[14] checkpointing")
    N = 20
    intervals = np.arange(1, 21)
    TRIALS = 200000
    crash_at = rng.integers(1, N + 1, TRIALS)          # crash just before this step
    redo = []
    for k in intervals:
        last_ckpt = ((crash_at - 1) // k) * k
        redo.append((crash_at - 1 - last_ckpt).mean())
    redo = np.array(redo)
    STEP_COST = 0.041                                   # $ per step, from chart 2's run

    fig, ax = plt.subplots(figsize=(7.4, 4.3), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(intervals, redo, color=BLUE, lw=2.4, marker='o', ms=4.5,
            label="steps re-executed after a crash (mean)")
    ax2 = ax.twinx()
    ax2.plot(intervals, redo * STEP_COST, color=RED, lw=0, marker='')
    ax2.set_ylabel("wasted $ per crash", color=RED, fontsize=11)
    ax2.set_ylim(ax.get_ylim()[0] * STEP_COST, ax.get_ylim()[1] * STEP_COST)
    ax2.tick_params(colors=RED, labelsize=9)
    ax.axvline(1, color=GREEN, ls=':', lw=1.6)
    ax.annotate(f"persist every step:\n{redo[0]:.1f} steps lost",
                xy=(1, redo[0]), xytext=(2.2, 1.2), color=GREEN, fontsize=9.5,
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.1))
    ax.annotate(f"no checkpoints at all:\n{redo[-1]:.1f} steps lost, "
                f"${redo[-1]*STEP_COST:.2f}",
                xy=(20, redo[-1]), xytext=(11.5, redo[-1] - 2.8), color=RED, fontsize=9.5,
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.1))
    style(ax, "The crash is not the problem — the redo is",
          "Steps between saved checkpoints (run of 20)", "Steps re-executed")
    ax.legend(fontsize=9, loc='upper left')
    save(fig, "checkpoint_interval_redo.png")
    for k in (1, 2, 5, 10, 20):
        print(f"    every {k:>2} steps: {redo[k-1]:.2f} steps redone "
              f"(${redo[k-1]*STEP_COST:.3f} wasted per crash)")


def contact_sheet(names):
    from matplotlib.image import imread
    cols = 3; rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.4 * rows), facecolor=BG)
    for ax, n in zip(axes.ravel(), names):
        ax.imshow(imread(OUT + n)); ax.set_title(n, fontsize=8); ax.axis('off')
    for ax in axes.ravel()[len(names):]: ax.axis('off')
    fig.tight_layout()
    fig.savefig("/tmp/llm_agents_contact_sheet.png", dpi=70, facecolor=BG)
    plt.close(fig); print("\n  contact sheet → /tmp/llm_agents_contact_sheet.png")


if __name__ == "__main__":
    compounding_reliability()
    agent_token_spend()
    prompt_cache_savings()
    retry_storm()
    parallel_tools()
    chunk_size_recall()
    context_growth()
    eval_sample_size()
    model_routing()
    streaming_wait()
    latency_waterfall()
    handoff_loss()
    reflection_vs_verifier()
    checkpoint_interval()
    contact_sheet([
        "agent_compounding_reliability.png", "agent_run_token_spend.png",
        "prompt_cache_savings.png", "retry_storm_cost.png",
        "parallel_vs_sequential_tools.png", "chunk_size_vs_recall.png",
        "context_growth_compaction.png", "eval_sample_size_ci.png",
        "model_routing_cost_curve.png", "streaming_perceived_wait.png",
        "agent_step_latency_waterfall.png", "multi_agent_handoff_loss.png",
        "reflection_vs_verifier.png", "checkpoint_interval_redo.png",
    ])
