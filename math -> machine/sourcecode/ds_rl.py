"""Reinforcement-learning diagrams — classic textbook experiments, re-run here so the numbers in the notes are real.
Run:  cd sourcecode && python3 ds_rl.py"""
from ds_style import *
import random

# 24 — Dyna-Q: planning steps between real steps (Sutton & Barto's maze)
def dyna():
    R, C, start, goal = 6, 9, (2, 0), (0, 8); walls = {(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)}; mv = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    def nxt(s, a):
        r, c = s[0] + mv[a][0], s[1] + mv[a][1]
        return s if not (0 <= r < R and 0 <= c < C) or (r, c) in walls else (r, c)
    def run(n, seed, episodes=50):
        rnd = random.Random(seed); Q = {}; model = {}; keys = []; out = []
        q = lambda s: Q.setdefault(s, [0.0] * 4)
        for ep in range(episodes):
            s, steps = start, 0
            while s != goal:
                qs = q(s); a = rnd.randrange(4) if rnd.random() < 0.1 else rnd.choice([i for i in range(4) if qs[i] == max(qs)])
                s2 = nxt(s, a); r = 1.0 if s2 == goal else 0.0; qs[a] += 0.1 * (r + 0.95 * max(q(s2)) - qs[a])
                if (s, a) not in model: keys.append((s, a))
                model[(s, a)] = (r, s2); s = s2; steps += 1
                for _ in range(n):                                   # imagined experience, replayed from the learned model
                    ps, pa = keys[rnd.randrange(len(keys))]; pr, ps2 = model[(ps, pa)]; pq = q(ps); pq[pa] += 0.1 * (pr + 0.95 * max(q(ps2)) - pq[pa])
            out.append(steps)
        return out
    fig, ax = new(figsize=(7.4, 4.2))
    for n, col in [(0, RED), (5, ORANGE), (50, GREEN)]:
        m = np.mean([run(n, sd) for sd in range(10)], 0); ax.plot(np.arange(2, 51), m[1:], color=col, lw=2.2, label=f"{n} imagined updates per real step" if n else "0 — plain Q-learning (model-free)")
        print(f"    n={n}: first episode {m[0]:.0f} steps; episode 2: {m[1]:.0f}; episode 5: {m[4]:.0f}; first episode under 20 steps: {next((i + 1 for i, v in enumerate(m) if v < 20), None)}")
    ax.axhline(14, color=GREY, ls=':', lw=1.2); ax.text(36, 28, "shortest path: 14 steps", fontsize=8.5, color=GREY)
    style(ax, "Imagined practice between real steps: same experience, far fewer episodes", "episode (the first one is a blind search for everyone — not shown)", "real steps to reach the goal"); ax.legend(fontsize=8.5); ax.set_ylim(0, 800)
    save(fig, "dyna_planning_steps.png")

# 25 — Monte Carlo estimate of pi
def mc_pi():
    rng = np.random.default_rng(4); fig, axes = new(1, 2, figsize=(11.5, 4.1)); P = rng.uniform(-1, 1, (2000, 2)); ins = (P ** 2).sum(1) <= 1
    ax = axes[0]; ax.scatter(P[ins, 0], P[ins, 1], s=5, color=BLUE, alpha=.7); ax.scatter(P[~ins, 0], P[~ins, 1], s=5, color=RED, alpha=.7); ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=FG, lw=1.5))
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); style(ax, f"2,000 darts: {ins.sum()} land inside → π ≈ 4 × {ins.mean():.3f} = {4 * ins.mean():.3f}", grid=False)
    ax = axes[1]; N = np.unique(np.logspace(1, 6, 200).astype(int))
    for k in range(5):
        hits = np.cumsum((rng.uniform(-1, 1, (N[-1], 2)) ** 2).sum(1) <= 1); ax.semilogx(N, 4 * hits[N - 1] / N, color=BLUE, lw=1, alpha=.7)
    sd = 4 * np.sqrt((np.pi / 4) * (1 - np.pi / 4) / N); ax.fill_between(N, np.pi - 2 * sd, np.pi + 2 * sd, color=ORANGE, alpha=.2, label="±2 standard errors  ∝ 1/√N"); ax.axhline(np.pi, color=FG, ls=':', lw=1.2)
    style(ax, "100× more darts buys only 10× less error", "number of darts N (log scale)", "estimate of π"); ax.legend(fontsize=9); ax.set_ylim(2.6, 3.7)
    suptitle(fig, "Monte Carlo: can't compute it? Sample it, and average", y=1.03)
    save(fig, "monte_carlo_pi.png"); print(f"    2000 darts -> {4 * ins.mean():.4f};  standard error at N=1e3: {4 * np.sqrt(.785 * .215 / 1e3):.3f}, at 1e5: {4 * np.sqrt(.785 * .215 / 1e5):.4f}")

# 26 — TD(0) vs constant-alpha Monte Carlo on the 5-state random walk
def td_vs_mc():
    true = np.arange(1, 6) / 6.0
    def run(method, alpha, seed, episodes=100):
        rnd = random.Random(seed); V = [0.0] + [0.5] * 5 + [0.0]; err = []
        for ep in range(episodes):
            err.append(np.sqrt(np.mean((np.array(V[1:6]) - true) ** 2))); s, traj = 3, [3]
            while 0 < s < 6:
                s2 = s + (1 if rnd.random() < 0.5 else -1); r = 1.0 if s2 == 6 else 0.0
                if method == "td": V[s] += alpha * (r + V[s2] - V[s])
                s = s2; traj.append(s)
            if method == "mc":
                G = 1.0 if s == 6 else 0.0
                for st in traj[:-1]: V[st] += alpha * (G - V[st])
        return err
    fig, ax = new(figsize=(7.4, 4.2)); res = {}
    for method, alphas, col in [("td", (0.05, 0.10, 0.15), BLUE), ("mc", (0.01, 0.02, 0.04), RED)]:
        for a, ls in zip(alphas, ['-', '--', ':']):
            m = np.mean([run(method, a, sd) for sd in range(100)], 0); res[(method, a)] = m[-1]
            ax.plot(m, color=col, lw=2, ls=ls, label=f"{'TD(0)' if method == 'td' else 'Monte Carlo'}, α = {a}")
    style(ax, "TD learns from every step; Monte Carlo waits for the ending", "episodes of a 5-state random walk", "RMS error of the value estimates"); ax.legend(fontsize=8.5, ncol=2); ax.set_ylim(0, 0.26)
    save(fig, "td_vs_mc_random_walk.png"); print("    final RMS:", {k: round(float(v), 3) for k, v in res.items()})

# 27 — cliff walking: SARSA vs Q-learning
def cliff():
    R, C, start, goal = 4, 12, (3, 0), (3, 11); mv = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    def env(s, a):
        r, c = min(max(s[0] + mv[a][0], 0), R - 1), min(max(s[1] + mv[a][1], 0), C - 1)
        return (start, -100.0) if (r == 3 and 1 <= c <= 10) else ((r, c), -1.0)
    def run(method, seed, episodes=500):
        rnd = random.Random(seed); Q = np.zeros((R, C, 4)); rew = []
        def pick(s): return rnd.randrange(4) if rnd.random() < 0.1 else int(rnd.choice(np.flatnonzero(Q[s] == Q[s].max())))
        for ep in range(episodes):
            s, a, tot = start, None, 0.0; a = pick(s)
            while s != goal:
                s2, r = env(s, a); a2 = pick(s2); tot += r
                target = r + (0 if s2 == goal else (Q[s2][a2] if method == "sarsa" else Q[s2].max()))
                Q[s][a] += 0.5 * (target - Q[s][a]); s, a = s2, a2
            rew.append(tot)
        return rew, Q
    def greedy_path(Q):
        s, path = start, [start]
        while s != goal and len(path) < 60: s, _ = env(s, int(Q[s].argmax())); path.append(s)
        return path
    fig, axes = new(2, 1, figsize=(7.8, 6.0), gridspec_kw=dict(height_ratios=[1, 1.35])); ax = axes[0]
    ax.add_patch(plt.Rectangle((0.5, 2.5), 10, 1, color=RED, alpha=.25)); ax.text(5.5, 3, "the cliff   (−100, back to the start)", ha='center', va='center', color=RED, fontsize=9.5)
    ax.text(0, 3, "S", ha='center', va='center', fontsize=12, fontweight='bold'); ax.text(11, 3, "G", ha='center', va='center', fontsize=12, fontweight='bold')
    stats = {}
    for method, col, off, lbl in [("sarsa", BLUE, -0.08, "SARSA — learns the value of what it actually does (wobbles included)"), ("q", ORANGE, 0.08, "Q-learning — learns the value of the perfect, wobble-free policy")]:
        runs = [run(method, sd) for sd in range(30)]; m = np.mean([r for r, _ in runs], 0); stats[method] = m[-100:].mean()
        Qm = np.mean([q for _, q in runs], 0); p = np.array(greedy_path(Qm)); ax.plot(p[:, 1] + off, p[:, 0] + off, color=col, lw=3, alpha=.9)
        axes[1].plot(smooth(m, 10), color=col, lw=2, label=lbl)
    ax.set_xlim(-0.5, 11.5); ax.set_ylim(3.5, -0.5); ax.set_xticks(np.arange(-0.5, 12, 1)); ax.set_yticks(np.arange(-0.5, 4, 1)); ax.set_xticklabels([]); ax.set_yticklabels([]); style(ax, "The path each one ends up preferring")
    style(axes[1], "…and what that costs while it's still exploring 10% of the time", "episode", "reward per episode"); axes[1].set_ylim(-100, 0); axes[1].legend(fontsize=8.3, loc='lower right')
    fig.tight_layout(); save(fig, "cliff_walking_sarsa_vs_q.png"); print("    mean reward per episode, last 100:", {k: round(float(v), 1) for k, v in stats.items()})

# 28 — maximisation bias
def max_bias():
    rng = np.random.default_rng(0); X = rng.normal(0, 1, (200000, 10)); Y = rng.normal(0, 1, (200000, 10))
    single, mx, dbl = X[:, 0], X.max(1), Y[np.arange(len(X)), X.argmax(1)]
    fig, ax = new(figsize=(7.4, 4.1)); bins = np.linspace(-4, 5, 90)
    for d, col, lbl in [(single, GREY, f"one honest estimate  (mean {single.mean():+.2f})"), (mx, RED, f"max over 10 estimates  (mean {mx.mean():+.2f})"), (dbl, GREEN, f"double estimator: one set picks, another scores  (mean {dbl.mean():+.2f})")]:
        ax.hist(d, bins=bins, density=True, histtype='step', lw=2.2, color=col, label=lbl)
    ax.axvline(0, color=FG, ls=':', lw=1.2); ax.text(-3.9, 0.30, "true value of\nevery action: 0 →", fontsize=8.5, color=FG)
    style(ax, "The max of noisy guesses is biased upward — even when every true value is zero", "estimated value", "how often"); ax.legend(fontsize=8.5, loc='upper left'); ax.set_yticks([]); ax.set_ylim(0, 0.95)
    save(fig, "maximisation_bias.png"); print(f"    E[max of 10 N(0,1)] = {mx.mean():.3f};  double estimator = {dbl.mean():.3f}")

# 29 — REINFORCE with and without a baseline
def reinforce():
    mu, runs, T, alpha = np.array([10.0, 10.5, 11.0]), 500, 3000, 0.05; fig, ax = new(figsize=(7.4, 4.1)); out = {}
    for base, col, lbl in [(False, RED, "no baseline: every reward (≈ +10) says 'do that more'"), (True, GREEN, "baseline = running average reward: only 'better than usual' counts")]:
        rng = np.random.default_rng(7); H = np.zeros((runs, 3)); b = np.zeros(runs); pbest = np.zeros(T)
        for t in range(T):
            pi = np.exp(H - H.max(1, keepdims=True)); pi /= pi.sum(1, keepdims=True); a = (rng.random((runs, 1)) > np.cumsum(pi, 1)).sum(1).clip(0, 2)
            r = mu[a] + rng.normal(0, 1, runs); onehot = np.eye(3)[a]; H += alpha * ((r - (b if base else 0.0))[:, None]) * (onehot - pi); b += (r - b) / (t + 1); pbest[t] = pi[:, 2].mean()
        ax.plot(pbest, color=col, lw=2.2, label=lbl); out[base] = pbest[-1]
    ax.axhline(1 / 3, color=GREY, ls=':', lw=1.2); ax.text(60, 0.27, "chance", fontsize=8.5, color=GREY)
    style(ax, "Same gradient, same data — subtracting a baseline is the whole difference", "pulls", "probability of choosing the best arm"); ax.legend(fontsize=8.5, loc='lower right'); ax.set_ylim(0, 1)
    save(fig, "reinforce_baseline.png"); print("    final P(best):", {("baseline" if k else "none"): round(float(v), 3) for k, v in out.items()})

# 30 — epsilon-greedy on the 10-armed testbed
def eps_testbed():
    runs, T = 2000, 1000; fig, axes = new(1, 2, figsize=(11.5, 3.9)); out = {}
    for eps, col in [(0.0, RED), (0.01, ORANGE), (0.1, BLUE)]:
        rng = np.random.default_rng(1); q = rng.normal(0, 1, (runs, 10)); best = q.argmax(1); Q = np.zeros((runs, 10)); Nn = np.zeros((runs, 10)); rew, opt = np.zeros(T), np.zeros(T); idx = np.arange(runs)
        for t in range(T):
            a = np.where(rng.random(runs) < eps, rng.integers(0, 10, runs), (Q + rng.random((runs, 10)) * 1e-9).argmax(1)); r = q[idx, a] + rng.normal(0, 1, runs)
            Nn[idx, a] += 1; Q[idx, a] += (r - Q[idx, a]) / Nn[idx, a]; rew[t], opt[t] = r.mean(), (a == best).mean() * 100
        lbl = f"ε = {eps}" + ("  (pure greed)" if eps == 0 else ""); axes[0].plot(rew, color=col, lw=1.8, label=lbl); axes[1].plot(opt, color=col, lw=1.8, label=lbl); out[eps] = (rew[-100:].mean(), opt[-100:].mean())
    style(axes[0], "average reward", "steps", None); style(axes[1], "% of pulls on the truly best arm", "steps", None); axes[1].legend(fontsize=9, loc='lower right'); axes[1].set_ylim(0, 100)
    suptitle(fig, "Pure greed gets stuck early. A little randomness keeps learning.", y=1.04)
    save(fig, "epsilon_greedy_testbed.png"); print("    last-100 (reward, %optimal):", {k: (round(float(a), 2), round(float(b), 1)) for k, (a, b) in out.items()})

if __name__ == "__main__":
    for f in (dyna, mc_pi, td_vs_mc, cliff, max_bias, reinforce, eps_testbed): f()
