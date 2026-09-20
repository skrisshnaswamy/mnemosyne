"""Bandits, experimentation, deep-RL and preference-learning diagrams. Simulated / computed.
Run:  cd sourcecode && python3 ds_bandits_deeprl.py"""
from ds_style import *
from math import lgamma, erf

ARMS = np.array([0.3, 0.5, 0.7]); ARMCOL = [RED, ORANGE, GREEN]

def bandit(alg, T, runs, seed=0, arms=ARMS, c=2.0):
    """Vectorised over runs. Returns cumulative expected regret (mean over runs) and pull counts."""
    rng = np.random.default_rng(seed); K = len(arms); S = np.zeros((runs, K)); N = np.zeros((runs, K)); reg = np.zeros(T); tot = np.zeros(runs); idx = np.arange(runs)
    for t in range(T):
        if alg == "thompson": a = rng.beta(S + 1, N - S + 1).argmax(1)
        elif t < K: a = np.full(runs, t)
        elif alg == "greedy": a = (S / N).argmax(1)
        elif alg == "eps": a = np.where(rng.random(runs) < 0.1, rng.integers(0, K, runs), (S / N).argmax(1))
        elif alg == "ucb": a = (S / N + np.sqrt(c * np.log(t + 1) / N)).argmax(1)
        r = rng.random(runs) < arms[a]; S[idx, a] += r; N[idx, a] += 1; tot += arms.max() - arms[a]; reg[t] = tot.mean()
    return reg, N.mean(0)

# 31 + 32 — regret curves, and where the pulls went
def regret_and_allocation():
    T, runs = 10000, 300; algs = [("greedy", "pure greed", GREY), ("eps", "ε-greedy (ε = 0.1)", RED), ("ucb", "UCB1", BLUE), ("thompson", "Thompson sampling", GREEN)]; res = {}
    fig, ax = new(figsize=(7.4, 4.2))
    for k, lbl, col in algs:
        reg, N = bandit(k, T, runs); res[k] = (reg, N); ax.plot(reg, color=col, lw=2.2, label=f"{lbl}  →  {reg[-1]:.0f}")
    style(ax, "Good exploration makes regret flatten out. Bad exploration pays forever.", "pulls  (three machines paying out 30%, 50%, 70% of the time)", "cumulative regret (wins lost vs always playing the best)"); ax.legend(fontsize=9, title="regret after 10,000 pulls", title_fontsize=8.5)
    save(fig, "regret_curves.png"); print("    regret@T:", {k: round(float(v[0][-1]), 1) for k, v in res.items()}, " regret@1000:", {k: round(float(v[0][999]), 1) for k, v in res.items()})
    fig, ax = new(figsize=(7.4, 3.3)); y = np.arange(len(algs))[::-1]
    for (k, lbl, _), yy in zip(algs, y):
        left = 0
        for j in range(3):
            w = res[k][1][j] / T * 100; ax.barh(yy, w, left=left, color=ARMCOL[j], alpha=.9)
            if w > 4: ax.text(left + w / 2, yy, f"{w:.0f}%", ha='center', va='center', color='white', fontsize=9)
            left += w
    ax.set_yticks(y); ax.set_yticklabels([a[1] for a in algs]); style(ax, "Where each strategy spent its 10,000 pulls", "share of pulls  (red = 30% machine, orange = 50%, green = the 70% one)", None, grid=False); ax.set_xlim(0, 100)
    save(fig, "bandit_pull_allocation.png"); print("    share on best arm:", {k: round(float(v[1][2] / T * 100), 1) for k, v in res.items()})

# 33 — UCB intervals closing in
def ucb_intervals():
    rng = np.random.default_rng(3); S = np.zeros(3); N = np.zeros(3); snaps = {}
    for t in range(3000):
        a = t if t < 3 else int((S / N + np.sqrt(2 * np.log(t + 1) / N)).argmax()); S[a] += rng.random() < ARMS[a]; N[a] += 1
        if t + 1 in (30, 300, 3000): snaps[t + 1] = (S / N, np.sqrt(2 * np.log(t + 1) / N), N.copy())
    fig, axes = new(1, 3, figsize=(12, 3.7), sharey=True)
    for ax, (t, (m, b, n)) in zip(axes, snaps.items()):
        for j in range(3):
            ax.errorbar([j], [m[j]], yerr=[[0], [b[j]]], fmt='o', color=ARMCOL[j], lw=3, ms=8, capsize=7); ax.plot([j - .3, j + .3], [ARMS[j]] * 2, color=FG, ls=':', lw=1.3); ax.text(j, 0.02, f"{int(n[j])} pulls", ha='center', fontsize=8.5, color=FG)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["30% machine", "50% machine", "70% machine"]); style(ax, f"after {t} pulls", None, "average so far  +  optimism bonus" if t == 30 else None); ax.set_ylim(0, 1.6)
    suptitle(fig, "UCB: play whatever *could* be best — and watch the doubt shrink where it matters", y=1.05)
    save(fig, "ucb_intervals.png"); print("    snapshot pulls:", {t: v[2].astype(int).tolist() for t, v in snaps.items()})

# 34 — Thompson posteriors
def thompson_posteriors():
    rng = np.random.default_rng(5); S = np.zeros(3); N = np.zeros(3); snaps = {}; x = np.linspace(0.001, 0.999, 600)
    for t in range(1000):
        a = int(rng.beta(S + 1, N - S + 1).argmax()); S[a] += rng.random() < ARMS[a]; N[a] += 1
        if t + 1 in (10, 100, 1000): snaps[t + 1] = (S.copy(), N.copy())
    fig, axes = new(1, 3, figsize=(12, 3.6))
    for ax, (t, (s, n)) in zip(axes, snaps.items()):
        for j in range(3):
            a_, b_ = s[j] + 1, n[j] - s[j] + 1; pdf = np.exp((a_ - 1) * np.log(x) + (b_ - 1) * np.log(1 - x) - (lgamma(a_) + lgamma(b_) - lgamma(a_ + b_)))
            ax.plot(x, pdf, color=ARMCOL[j], lw=2.2, label=f"{int(n[j])} pulls"); ax.fill_between(x, pdf, color=ARMCOL[j], alpha=.15); ax.axvline(ARMS[j], color=ARMCOL[j], ls=':', lw=1.2)
        style(ax, f"after {t} pulls", "payout rate", "belief" if t == 10 else None); ax.set_yticks([]); ax.legend(fontsize=8.5)
    suptitle(fig, "Thompson sampling: wide beliefs get tried, narrow ones settle", y=1.05)
    save(fig, "thompson_posteriors.png"); print("    pulls:", {t: v[1].astype(int).tolist() for t, v in snaps.items()})

# 35 — context flips the winner
def contextual():
    fig, ax = new(figsize=(7.2, 3.9)); x = np.arange(3); w = 0.36; A, B = [5, 2, 8], [5, 8, 2]
    ax.bar(x - w / 2, A, w, color=BLUE, label="headline A (sport)"); ax.bar(x + w / 2, B, w, color=ORANGE, label="headline B (finance)")
    for i in range(3): ax.text(x[i] - w / 2, A[i] + .15, f"{A[i]}%", ha='center', fontsize=9.5); ax.text(x[i] + w / 2, B[i] + .15, f"{B[i]}%", ha='center', fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels(["everyone\n(what a plain bandit sees)", "morning commuters", "evening readers"]); style(ax, "No winner overall — a clear winner in every context", None, "click-through rate", grid=False); ax.legend(fontsize=9); ax.set_ylim(0, 10)
    save(fig, "contextual_bandit_ctr.png")

# 36 — A/B test vs bandit
def ab_vs_bandit():
    pA, pB, T, runs = 0.04, 0.06, 20000, 300; rng = np.random.default_rng(2); S = np.zeros((runs, 2)); N = np.zeros((runs, 2)); share = np.zeros(T); idx = np.arange(runs); p = np.array([pA, pB])
    for t in range(T):
        a = rng.beta(S + 1, N - S + 1).argmax(1); S[idx, a] += rng.random(runs) < p[a]; N[idx, a] += 1; share[t] = a.mean()
    fig, axes = new(1, 2, figsize=(12, 3.9)); ax = axes[0]
    ax.plot(smooth(share, 200) * 100, color=GREEN, lw=2.2, label="bandit (Thompson sampling)"); ax.axhline(50, color=GREY, lw=2.2, label="A/B test: 50/50 until the end")
    style(ax, "share of visitors sent to the better variant (B)", "visitors so far", "%"); ax.legend(fontsize=9, loc='lower right'); ax.set_ylim(0, 100)
    lost_ab, lost_b = 0.5 * (pB - pA) * np.arange(1, T + 1), np.cumsum((1 - share) * (pB - pA)); ax = axes[1]
    ax.plot(lost_ab, color=GREY, lw=2.2, label=f"A/B test  →  {lost_ab[-1]:.0f} conversions"); ax.plot(lost_b, color=GREEN, lw=2.2, label=f"bandit  →  {lost_b[-1]:.0f} conversions")
    style(ax, "conversions lost by showing the worse variant", "visitors so far", None); ax.legend(fontsize=9, loc='upper left')
    suptitle(fig, "An A/B test pays full price to learn. A bandit starts cashing in early.", y=1.04)
    save(fig, "ab_vs_bandit_allocation.png"); print(f"    lost: A/B {lost_ab[-1]:.0f}, bandit {lost_b[-1]:.0f}; share to B over last 2000: {share[-2000:].mean() * 100:.1f}%")

# 37 — off-policy evaluation: naive vs IPS vs SNIPS
def ope():
    rng = np.random.default_rng(9); click = np.array([[0.30, 0.05], [0.05, 0.20]]); n, reps = 5000, 600; est = {"naive": [], "IPS": [], "SNIPS": []}
    for _ in range(reps):
        x = rng.integers(0, 2, n); a = np.where(rng.random(n) < 0.97, x, 1 - x); prop = np.where(a == x, 0.97, 0.03); r = rng.random(n) < click[x, a]; m = (a == 1); w = m / prop
        est["naive"].append(r[m].mean()); est["IPS"].append((w * r).mean()); est["SNIPS"].append((w * r).sum() / w.sum())
    truth = click[:, 1].mean(); fig, ax = new(figsize=(7.4, 4.0)); bins = np.linspace(0.04, 0.24, 90)
    for k, col in [("naive", RED), ("IPS", BLUE), ("SNIPS", GREEN)]:
        ax.hist(est[k], bins=bins, color=col, alpha=.55, label=f"{k}:  {np.mean(est[k]):.3f} ± {np.std(est[k]):.3f}")
    ax.axvline(truth, color=FG, lw=2, ls='--'); ax.text(truth + 0.002, ax.get_ylim()[1] * 0.92, f"truth = {truth:.3f}", fontsize=9, color=FG)
    style(ax, "The naive estimate is confidently wrong. IPS is noisily right.", "estimated click rate of the new policy 'show finance to everyone'", "600 simulated logs of 5,000 visits"); ax.legend(fontsize=9, loc='upper left'); ax.set_yticks([])
    save(fig, "ope_ips_vs_naive.png"); print(f"    truth {truth:.4f}", {k: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4)) for k, v in est.items()})

# 38 — Bayesian optimisation: GP + expected improvement
def bayesopt():
    f = lambda x: 0.15 * (x + 3) ** 2 + 0.1 * np.sin(3 * x) + 0.3; X = np.array([-4.8, -4.0, -2.4, -1.7, -1.1]); y = f(X); xs = np.linspace(-5, -1, 400)
    k = lambda a, b: 0.12 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / 0.55 ** 2); mu0 = y.mean(); Kinv = np.linalg.inv(k(X, X) + 1e-6 * np.eye(len(X))); Ks = k(xs, X)
    mu = mu0 + Ks @ Kinv @ (y - mu0); sd = np.sqrt(np.clip(0.12 - np.einsum('ij,jk,ik->i', Ks, Kinv, Ks), 1e-12, None)); best = y.min(); z = (best - mu) / sd
    Phi = 0.5 * (1 + np.vectorize(erf)(z / np.sqrt(2))); phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi); ei = (best - mu) * Phi + sd * phi; nx = xs[ei.argmax()]
    fig, axes = new(2, 1, figsize=(7.6, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1])); ax = axes[0]
    ax.plot(xs, f(xs), color=GREY, ls='--', lw=1.5, label="the truth (unknown to the optimiser)"); ax.plot(xs, mu, color=BLUE, lw=2.2, label="surrogate model: best guess"); ax.fill_between(xs, mu - 2 * sd, mu + 2 * sd, color=BLUE, alpha=.15, label="…and how unsure it is")
    ax.plot(X, y, 'o', color=FG, ms=7, label="5 training runs so far"); ax.axvline(nx, color=GREEN, ls=':', lw=1.5); style(ax, "Bayesian optimisation: model what you know, then try where it's promising *or* unknown", None, "validation loss"); ax.legend(fontsize=8.3, loc='upper center', ncol=2); ax.set_ylim(-0.25, 1.35)
    axes[1].plot(xs, ei, color=GREEN, lw=2.2); axes[1].fill_between(xs, ei, color=GREEN, alpha=.2); axes[1].plot([nx], [ei.max()], 'v', color=GREEN, ms=10); axes[1].text(nx + 0.75, ei.max() * 0.6, f"← run this next: lr = 10^{nx:.2f}", fontsize=9, color=GREEN)
    style(axes[1], None, "log10(learning rate)", "expected\nimprovement"); axes[1].set_yticks([])
    save(fig, "bayesopt_gp_ei.png"); print(f"    next x={nx:.3f}; true argmin={xs[f(xs).argmin()]:.3f}; best so far {best:.3f} at x={X[y.argmin()]}")

# 39 — table vs function approximator
def func_approx():
    rng = np.random.default_rng(12); V = lambda s: 1 / (1 + np.exp(-10 * (s - 0.55))) + 0.15 * np.sin(9 * s); s_vis = np.sort(rng.uniform(0, 1, 22)); y = V(s_vis) + rng.normal(0, 0.05, 22); xs = np.linspace(0, 1, 400)
    bins = np.linspace(0, 1, 51); tab = np.full(50, np.nan); ib = np.clip(np.digitize(s_vis, bins) - 1, 0, 49)
    for b in np.unique(ib): tab[b] = y[ib == b].mean()
    cen = np.linspace(0, 1, 9); Fm = lambda s: np.exp(-0.5 * (s[:, None] - cen[None]) ** 2 / 0.12 ** 2); w = np.linalg.solve(Fm(s_vis).T @ Fm(s_vis) + 1e-2 * np.eye(9), Fm(s_vis).T @ y)
    fig, axes = new(1, 2, figsize=(12, 3.8), sharey=True)
    for ax in axes: ax.plot(xs, V(xs), color=GREY, ls='--', lw=1.5, label="true value"); ax.plot(s_vis, y, 'o', color=FG, ms=5, label="states actually visited")
    axes[0].bar(bins[:-1] + 0.01, np.nan_to_num(tab), width=0.02, color=RED, alpha=.75, label="table: one cell per state"); style(axes[0], f"A table: {np.isnan(tab).sum()} of 50 cells still blank", "state", "value"); axes[0].legend(fontsize=8.5, loc='upper left')
    axes[1].plot(xs, Fm(xs) @ w, color=BLUE, lw=2.4, label="approximator: 9 shared parameters"); style(axes[1], "An approximator: every state gets an answer", "state", None); axes[1].legend(fontsize=8.5, loc='upper left')
    suptitle(fig, "A table knows only where it's been. An approximator guesses the rest.", y=1.04)
    save(fig, "function_approx_generalisation.png"); print(f"    blank cells {np.isnan(tab).sum()}/50; approximator RMS error {np.sqrt(np.mean((Fm(xs) @ w - V(xs)) ** 2)):.3f}")

# 40 — why replay: decorrelation
def replay():
    rng = np.random.default_rng(6); n = 20000; s = np.zeros(n)
    for t in range(1, n): s[t] = 0.995 * s[t - 1] + rng.normal(0, 0.1)
    fig, axes = new(1, 2, figsize=(12, 3.8)); ax = axes[0]; ax.plot(s[:6000], color=GREY, lw=1, alpha=.8); i0 = 3000
    ax.plot(np.arange(i0, i0 + 32), s[i0:i0 + 32], 'o', color=RED, ms=5, label="32 consecutive steps — one tiny patch"); pick = rng.integers(0, 6000, 32); ax.plot(pick, s[pick], 'o', color=GREEN, ms=5, label="32 draws from the replay buffer — the whole range")
    style(ax, "what goes into one minibatch", "time step", "state"); ax.legend(fontsize=8.5, loc='upper left')
    lags = np.arange(0, 301, 5); ac = lambda z: [1.0 if l == 0 else np.corrcoef(z[:-l], z[l:])[0, 1] for l in lags]; sh = rng.permutation(s); ax = axes[1]
    ax.plot(lags, ac(s), color=RED, lw=2.2, label="raw experience stream"); ax.plot(lags, ac(sh), color=GREEN, lw=2.2, label="sampled from the buffer"); style(ax, "how similar is a sample to the one k steps later?", "k", "correlation"); ax.legend(fontsize=9)
    suptitle(fig, "Consecutive experience is nearly one sample repeated. Replay turns it back into a dataset.", y=1.04)
    save(fig, "replay_decorrelation.png"); print(f"    lag-1 corr: stream {np.corrcoef(s[:-1], s[1:])[0, 1]:.3f}, lag-100: {np.corrcoef(s[:-100], s[100:])[0, 1]:.3f}, shuffled {np.corrcoef(sh[:-1], sh[1:])[0, 1]:.3f}")

# 41 — PPO clipped objective
def ppo_clip():
    r = np.linspace(0, 2, 400); eps = 0.2; fig, axes = new(1, 2, figsize=(11, 3.7))
    for ax, A, ttl in [(axes[0], 1.0, "the action turned out GOOD  (advantage > 0)"), (axes[1], -1.0, "the action turned out BAD  (advantage < 0)")]:
        L = np.minimum(r * A, np.clip(r, 1 - eps, 1 + eps) * A); ax.plot(r, r * A, color=GREY, ls='--', lw=1.5, label="unclipped: push as far as you like"); ax.plot(r, L, color=BLUE, lw=2.6, label="PPO: clipped")
        ax.axvspan(1 - eps, 1 + eps, color=GREEN, alpha=.12); ax.plot([1], [A], 'o', color=FG, ms=7); ax.text(1.02, A - 0.22 * np.sign(A) - 0.05, "start\n(new = old)", fontsize=8, color=FG)
        style(ax, ttl, "probability ratio   new policy ÷ old policy", "objective" if A > 0 else None); ax.legend(fontsize=8.5, loc='upper left' if A > 0 else 'lower left')
    axes[0].text(1.23, 0.45, "flat: no reward for\npushing past +20%", fontsize=8.5, color=BLUE); axes[1].text(0.03, -1.18, "flat: no reward for\npushing past −20%", fontsize=8.5, color=BLUE)
    suptitle(fig, "PPO's clip: beyond ±20%, there is nothing more to gain — so the policy stops moving", y=1.05)
    save(fig, "ppo_clip.png")

# 42 — entropy temperature
def sac_temperature():
    Q = np.array([1.0, 0.9, 0.2, -0.5, -1.0]); names = ["a₁", "a₂", "a₃", "a₄", "a₅"]; fig, axes = new(1, 4, figsize=(12, 3.2), sharey=True)
    for ax, al, lbl in zip(axes, [0.05, 0.3, 1.0, 5.0], ["α = 0.05\nalmost greedy", "α = 0.3", "α = 1", "α = 5\nnearly uniform"]):
        p = np.exp((Q - Q.max()) / al); p /= p.sum(); ax.bar(names, p, color=[BLUE, BLUE, GREY, GREY, GREY]); style(ax, lbl, None, "probability of choosing it" if al == 0.05 else None); ax.set_ylim(0, 1)
        print(f"    alpha={al}: p={np.round(p, 3)}")
    suptitle(fig, "Entropy temperature α: a₁ and a₂ are nearly as good — a max-entropy policy keeps both alive", y=1.12)
    save(fig, "sac_temperature.png")

# 43 — offline RL: extrapolation outside the data
def offline_extrapolation():
    true = lambda a: 1.0 - 1.6 * (a - 0.2) ** 2; xs = np.linspace(-1, 1, 400); fig, ax = new(figsize=(7.4, 4.2)); arg = []
    for sd in range(8):
        rng = np.random.default_rng(sd); a = rng.uniform(-0.6, 0.1, 40); y = true(a) + rng.normal(0, 0.05, 40); co = np.polyfit(a, y, 5); fit = np.polyval(co, xs); arg.append(xs[fit.argmax()])
        ax.plot(xs, fit, color=BLUE, lw=1.3, alpha=.75, label="8 equally good fits to the logged data" if sd == 0 else None); ax.plot([xs[fit.argmax()]], [np.clip(fit.max(), -2.9, 3.6)], 'v', color=RED, ms=8, label="…and the action each one's argmax picks" if sd == 0 else None)
    ax.plot(a, y, 'o', color=FG, ms=3.5, alpha=.7, label="logged (action, return) pairs"); ax.plot(xs, true(xs), color=GREEN, lw=2.4, ls='--', label="true value of each action"); ax.axvspan(-0.6, 0.1, color=GREEN, alpha=.1); ax.text(-0.58, -2.75, "actions the old policy\nactually tried", fontsize=8.5, color=GREEN)
    style(ax, "The max operator goes looking for exactly the actions nobody ever tried", "action", "estimated value"); ax.set_ylim(-3, 5.4); ax.legend(fontsize=8, loc='upper center', ncol=2)
    save(fig, "offline_rl_extrapolation.png"); print("    argmax actions of the 8 fits:", np.round(arg, 2), " (true best = 0.20, data covers -0.6..0.1)")

# 44 — behaviour cloning drifts off the demonstrated road
def bc_drift():
    rng = np.random.default_rng(8); T, runs = 400, 40; fig, ax = new(figsize=(7.6, 4.2)); off = {}
    for name, cover, col in [("behaviour cloning: only ever saw the expert near the centre line", 0.3, RED), ("DAgger: expert also labelled the states the learner wandered into", 3.0, GREEN)]:
        X = np.zeros((runs, T))
        for k in range(1, T):
            x = X[:, k - 1]; known = np.abs(x) <= cover; act = np.where(known, -0.5 * x, rng.normal(0, 0.1, runs)) + rng.normal(0, 0.03, runs); X[:, k] = x + act + rng.normal(0, 0.12, runs)
        for i in range(runs): ax.plot(X[i], color=col, lw=.8, alpha=.45)
        ax.plot([], [], color=col, lw=2, label=name); off[name[:6]] = (np.abs(X[:, -1]) > 2).mean() * 100
    ax.axhspan(-0.3, 0.3, color=GREY, alpha=.3); ax.text(3, 0.45, "where the demonstrations live", fontsize=8.5, color=FG, bbox=dict(facecolor=BG, edgecolor='none', alpha=.85, pad=1)); ax.axhline(2, color=FG, ls='--', lw=1); ax.axhline(-2, color=FG, ls='--', lw=1); ax.text(3, 2.12, "edge of the road", fontsize=8.5, color=FG)
    style(ax, "Behaviour cloning only knows the road the expert drove", "time step", "distance from the lane centre (m)"); ax.set_ylim(-5, 5); ax.legend(fontsize=8.3, loc='lower left')
    save(fig, "bc_compounding_error.png"); print("    % of runs off the road at the end:", {k: round(float(v), 1) for k, v in off.items()})

# 45 — Bradley-Terry
def bradley_terry():
    d = np.linspace(-4, 4, 400); p = 1 / (1 + np.exp(-d)); fig, ax = new(figsize=(7.2, 4.1)); ax.plot(d, p * 100, color=BLUE, lw=2.5); ax.axhline(50, color=GREY, ls=':', lw=1.1)
    for g in (0, 1, 2, 3): ax.plot([g], [100 / (1 + np.exp(-g))], 'o', color=ORANGE, ms=7); ax.text(g + 0.12, 100 / (1 + np.exp(-g)) - 6, f"gap {g} → {100 / (1 + np.exp(-g)):.0f}%", fontsize=9, color=ORANGE)
    ax2 = ax.secondary_xaxis('top', functions=(lambda x: x * 400 / np.log(10), lambda e: e * np.log(10) / 400)); ax2.set_xlabel("the same gap in Elo points", fontsize=9, color=FG); ax2.tick_params(labelsize=8.5, colors=FG)
    style(ax, None, "reward-model score of A minus score of B", "P(a human prefers A)  %"); fig.suptitle("Only the gap matters: a score difference becomes a win probability", fontsize=12, color=FG, y=1.04)
    save(fig, "bradley_terry_sigmoid.png"); print("    sigmoid(1,2,3) =", [round(100 / (1 + np.exp(-g)), 1) for g in (1, 2, 3)], " 400 Elo ->", round(100 / (1 + 10 ** (-1)), 1))

# 46 — GRPO group-relative advantage
def grpo():
    fig, axes = new(1, 2, figsize=(11.5, 3.8), sharey=True)
    for ax, r, ttl in [(axes[0], np.array([1, 0, 0, 1, 0, 0, 0, 1.0]), "a useful prompt: 3 of 8 answers correct"), (axes[1], np.ones(8), "a too-easy prompt: 8 of 8 correct")]:
        adv = (r - r.mean()) / (r.std() + 1e-8); ax.bar(np.arange(1, 9), adv, color=[GREEN if v > 0 else (RED if v < 0 else GREY) for v in adv]); ax.axhline(0, color=FG, lw=1)
        for i, (rv, av) in enumerate(zip(r, adv)): ax.text(i + 1, av + (0.08 if av >= 0 else -0.2), "✓" if rv else "✗", ha='center', fontsize=12, color=FG)
        style(ax, ttl, "sampled answer #", "advantage = (reward − group mean) ÷ group std" if ax is axes[0] else None); ax.set_ylim(-1.4, 1.8); ax.set_xticks(np.arange(1, 9)); print(f"    rewards {r.astype(int)} -> adv {np.round(adv, 2)}")
    axes[1].text(4.5, 0.6, "everyone scored the same\n→ every advantage is 0\n→ nothing to learn from", ha='center', fontsize=9.5, color=GREY)
    suptitle(fig, "GRPO: grade each answer against its own group — no critic network needed", y=1.04)
    save(fig, "grpo_group_advantage.png")

if __name__ == "__main__":
    for f in (regret_and_allocation, ucb_intervals, thompson_posteriors, contextual, ab_vs_bandit, ope, bayesopt, func_approx, replay, ppo_clip, sac_temperature, offline_extrapolation, bc_drift, bradley_terry, grpo): f()
