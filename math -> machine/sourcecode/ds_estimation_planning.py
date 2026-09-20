"""Estimation, planning and MDP diagrams. Simulated / computed, not sketched.
Run:  cd sourcecode && python3 ds_estimation_planning.py"""
from ds_style import *
from matplotlib.patches import Ellipse, FancyArrowPatch, Circle

gauss = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

# 8 — the Bayes / Kalman update with the numbers from Decision Sciences ch.2
def bayes_filter():
    m_pred, s_pred, z, s_z = 120.0, 14.0, 130.0, 5.0
    K = s_pred ** 2 / (s_pred ** 2 + s_z ** 2); m = m_pred + K * (z - m_pred); s = np.sqrt((1 - K) * s_pred ** 2)
    x = np.linspace(70, 170, 800); fig, axes = new(1, 2, figsize=(12, 3.9))
    ax = axes[0]
    for mu, sd, col, lbl in [(m_pred, s_pred, ORANGE, f"prediction  {m_pred:.0f} ± {s_pred:.0f} m"), (z, s_z, GREEN, f"radar  {z:.0f} ± {s_z:.0f} m"), (m, s, BLUE, f"new belief  {m:.1f} ± {s:.1f} m")]:
        ax.plot(x, gauss(x, mu, sd), color=col, lw=2.2, label=lbl); ax.fill_between(x, gauss(x, mu, sd), color=col, alpha=.15)
    style(ax, "Two uncertain sources make one estimate that beats both", "position (m)", "belief"); ax.legend(fontsize=9, loc='upper left'); ax.set_yticks([])
    P, q, r = 100.0, 96.0, 25.0; xs, ys = [0], [np.sqrt(P)]
    for k in range(1, 9):
        P = P + q; xs.append(k - 0.5); ys.append(np.sqrt(P)); P = P * r / (P + r); xs.append(k); ys.append(np.sqrt(P))
    ax = axes[1]; ax.plot(xs, ys, color=BLUE, lw=2, marker='o', ms=4)
    ax.annotate("predict:\ntime passes,\nyou know less", xy=(0.5, 14), xytext=(1.5, 12.3), fontsize=8.5, color=ORANGE, arrowprops=dict(arrowstyle='->', color=ORANGE))
    ax.annotate("update:\nyou looked,\nyou know more", xy=(1, ys[2]), xytext=(2.1, 6.3), fontsize=8.5, color=GREEN, arrowprops=dict(arrowstyle='->', color=GREEN))
    style(ax, "Uncertainty breathes: out on predict, in on update", "time step", "uncertainty σ (m)"); ax.set_ylim(3, 15)
    save(fig, "bayes_filter_gaussians.png"); print(f"    K={K:.4f}  mean={m:.2f}  sigma={s:.2f}   steady-state sigma after update={ys[-1]:.2f}, after predict={ys[-2]:.2f}")

# 9 — a Gaussian pushed through a nonlinearity (range/bearing -> x/y)
def ekf_banana():
    rng = np.random.default_rng(5); r0, th0, sr, sth = 10.0, np.pi / 2, 0.3, np.deg2rad(25)
    r = rng.normal(r0, sr, 5000); th = rng.normal(th0, sth, 5000); X = np.column_stack([r * np.cos(th), r * np.sin(th)])
    mc_mean, S = X.mean(0), np.diag([sr ** 2, sth ** 2])
    J = np.array([[np.cos(th0), -r0 * np.sin(th0)], [np.sin(th0), r0 * np.cos(th0)]]); ekf_mean = np.array([r0 * np.cos(th0), r0 * np.sin(th0)]); ekf_cov = J @ S @ J.T
    n, lam = 2, 1.0; L = np.linalg.cholesky((n + lam) * S); pts = [np.array([r0, th0])] + [np.array([r0, th0]) + sg * L[:, i] for i in range(n) for sg in (1, -1)]
    W = np.array([lam / (n + lam)] + [1 / (2 * (n + lam))] * (2 * n)); Y = np.array([[p[0] * np.cos(p[1]), p[0] * np.sin(p[1])] for p in pts])
    ukf_mean = W @ Y; ukf_cov = sum(w * np.outer(y - ukf_mean, y - ukf_mean) for w, y in zip(W, Y))
    def ell(ax, mean, cov, col, ls):
        vals, vecs = np.linalg.eigh(cov); ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        ax.add_patch(Ellipse(mean, 4 * np.sqrt(vals[1]), 4 * np.sqrt(vals[0]), angle=ang, fill=False, edgecolor=col, lw=2.2, ls=ls))
    fig, ax = new(figsize=(7.4, 4.6)); ax.scatter(X[:, 0], X[:, 1], s=2, color=GREY, alpha=.35, label="where the target really could be (5,000 samples)")
    ell(ax, ekf_mean, ekf_cov, RED, '-'); ax.plot(*ekf_mean, 'o', color=RED, ms=7, label=f"EKF: linearise, pretend it's flat  (mean y = {ekf_mean[1]:.2f})")
    ell(ax, ukf_mean, ukf_cov, GREEN, '--'); ax.plot(*ukf_mean, 's', color=GREEN, ms=7, label=f"UKF: push 5 sigma points through  (mean y = {ukf_mean[1]:.2f})")
    ax.plot(*mc_mean, 'x', color=FG, ms=10, mew=2.5, label=f"true mean  (y = {mc_mean[1]:.2f})")
    style(ax, "Push a Gaussian through a curve and it stops being a Gaussian", "x (m)", "y (m)"); ax.legend(fontsize=8, loc='lower center'); ax.set_xlim(-11, 11); ax.set_ylim(3.5, 11.5)
    save(fig, "ekf_banana.png"); print(f"    true mean y={mc_mean[1]:.3f}  EKF={ekf_mean[1]:.3f}  UKF={ukf_mean[1]:.3f}  analytic={r0 * np.exp(-sth ** 2 / 2):.3f}")

# 10 — particle filter in a corridor with three identical doors
def particle_filter():
    rng = np.random.default_rng(2); N, doors = 2000, [15, 35, 80]
    like = lambda x: 0.03 + sum(np.exp(-0.5 * ((x - d) / 2.0) ** 2) for d in doors)
    def resample(p, w): return p[rng.choice(N, N, p=w / w.sum())] + rng.normal(0, 0.4, N)
    p0 = rng.uniform(0, 100, N); p1 = resample(p0, like(p0)); p2 = p1 + 20 + rng.normal(0, 1.5, N); p3 = resample(p2, like(p2))
    stages = [(p0, 15, "1 · no idea: 2,000 guesses spread evenly"), (p1, 15, "2 · sensor says DOOR → guesses pile up at all three doors"),
              (p2, 35, "3 · robot drives 20 m → every guess moves too (and blurs a little)"), (p3, 35, "4 · sensor says DOOR again → only one pile is still standing at a door")]
    fig, axes = new(4, 1, figsize=(7.6, 6.6), sharex=True)
    for ax, (p, truth, lbl) in zip(axes, stages):
        ax.hist(p, bins=np.arange(0, 121, 1), color=BLUE, alpha=.85)
        for d in doors: ax.axvline(d, color=ORANGE, lw=5, alpha=.35)
        ax.plot([truth], [0], marker='^', color=RED, ms=11, clip_on=False, zorder=5); style(ax, None, None, None); ax.set_yticks([]); ax.set_xlim(0, 120)
        ax.text(0.995, 0.86, lbl, transform=ax.transAxes, ha='right', fontsize=8.8, color=FG, bbox=dict(facecolor=BG, edgecolor='none', alpha=.8, pad=1))
        print(f"    stage: {np.mean(np.abs(p - truth) < 3) * 100:.0f}% of particles within 3 m of the truth")
    axes[-1].set_xlabel("position along the corridor (m)   ▲ = where the robot really is   orange = doors", fontsize=9.5, color=FG)
    suptitle(fig, "Particle filter: thousands of guesses, pruned by evidence", y=0.94)
    save(fig, "particle_filter_corridor.png")

# 11 — HMM filtering: the umbrella world
def hmm_umbrella():
    Tm = np.array([[0.7, 0.3], [0.3, 0.7]]); E = {1: np.array([0.9, 0.2]), 0: np.array([0.1, 0.8])}   # state 0 = rain
    obs = [1, 1, 0, 1, 1, 1, 0, 0, 0, 1]; b = np.array([0.5, 0.5]); out = []
    for o in obs:
        b = (Tm.T @ b) * E[o]; b /= b.sum(); out.append(b[0])
    fig, ax = new(figsize=(7.4, 4.0)); days = np.arange(1, len(obs) + 1)
    ax.plot(days, out, color=BLUE, lw=2.2, marker='o', ms=6); ax.axhline(0.5, color=GREY, ls=':', lw=1.2)
    for d, o, p in zip(days, obs, out): ax.text(d, 1.04, "☂" if o else "—", ha='center', fontsize=15 if o else 11, color=ORANGE if o else GREY)
    ax.text(0.35, 1.045, "seen:", fontsize=8.5, color=GREY, ha='right')
    style(ax, "You never see the weather — only the umbrella", "day", "P(it is raining | umbrellas seen so far)"); ax.set_ylim(0, 1.15); ax.set_xlim(0.2, 10.6); ax.set_xticks(days)
    save(fig, "hmm_umbrella_filtering.png"); print("    P(rain):", " ".join(f"{p:.3f}" for p in out))

# 12 — the commute, solved backwards
def dp_commute():
    pos = dict(Home=(0, 0), A=(1.4, 1.1), B=(1.4, -1.1), C=(2.9, 1.8), D=(2.9, 0), E=(2.9, -1.8), Office=(4.4, 0))
    edges = [("Home", "A", 5), ("Home", "B", 8), ("A", "C", 10), ("A", "D", 25), ("B", "D", 4), ("B", "E", 6), ("C", "Office", 22), ("D", "Office", 8), ("E", "Office", 9)]
    V = dict(Office=0)
    for nd in ["C", "D", "E", "A", "B", "Home"]: V[nd] = min(w + V[v] for u, v, w in edges if u == nd)
    best, greedy = {("Home", "B"), ("B", "D"), ("D", "Office")}, {("Home", "A"), ("A", "C"), ("C", "Office")}
    fig, ax = new(figsize=(8.2, 4.7)); ax.axis('off'); ax.set_facecolor(BG)
    for u, v, w in edges:
        col, lw, ls = (GREEN, 3.2, '-') if (u, v) in best else ((RED, 2.2, '--') if (u, v) in greedy else (GREY, 1.4, '-'))
        ax.add_patch(FancyArrowPatch(pos[u], pos[v], arrowstyle='-|>', mutation_scale=16, color=col, lw=lw, ls=ls, shrinkA=19, shrinkB=19))
        mx, my = (pos[u][0] * .55 + pos[v][0] * .45), (pos[u][1] * .55 + pos[v][1] * .45)
        ax.text(mx, my + 0.17, f"{w} min", fontsize=9, color=col if col != GREY else FG, ha='center', bbox=dict(facecolor=BG, edgecolor='none', pad=.6, alpha=.9))
    for nd, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.3, facecolor="#eaf1fb", edgecolor=BLUE, lw=1.8, zorder=3)); ax.text(x, y + 0.04, nd if len(nd) == 1 else nd, ha='center', va='center', fontsize=8.5 if len(nd) > 1 else 11, color=FG, zorder=4)
        ax.text(x, y + (0.5 if nd in ("A", "C") else -0.52), f"best from here: {V[nd]}", ha='center', fontsize=8.3, color=BLUE, zorder=4, bbox=dict(facecolor=BG, edgecolor='none', pad=.4, alpha=.9))
    ax.text(-0.45, -2.2, "green = what working backwards finds (8 + 4 + 8 = 20 min)", color=GREEN, fontsize=9.5)
    ax.text(-0.45, -2.5, "red dashed = always take the quickest next leg (5 + 10 + 22 = 37 min)", color=RED, fontsize=9.5)
    ax.set_xlim(-0.5, 4.9); ax.set_ylim(-2.7, 2.6); ax.set_title("Work backwards: every junction gets one number, and the numbers do the planning", fontsize=11.5, color=FG, pad=8)
    save(fig, "dp_commute_graph.png"); print("    values:", V)

# ---------------------------------------------------------------- gridworld shared by the value-iteration charts
ROWS, COLS, GOAL, PIT, WALLS = 6, 6, (0, 5), (2, 4), {(1, 2), (2, 2), (3, 2)}
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]; ARROW = ["↑", "→", "↓", "←"]
def step(s, a):
    r, c = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
    return s if not (0 <= r < ROWS and 0 <= c < COLS) or (r, c) in WALLS else (r, c)
def reward(s2): return 10.0 if s2 == GOAL else (-10.0 if s2 == PIT else 0.0)
def sweep(V, gamma, slip=0.0, continuing=False):
    """continuing=True: entering GOAL/PIT pays out and teleports you back to the start, so the task never ends."""
    V2, pol = V.copy(), np.zeros((ROWS, COLS), int); nxt = lambda s2: V[(5, 0)] if continuing and s2 in (GOAL, PIT) else V[s2]
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) in WALLS or (r, c) in (GOAL, PIT): continue
            q = []
            for a in range(4):
                outs = [(1 - slip, step((r, c), a))] + ([(slip / 3, step((r, c), b)) for b in range(4) if b != a] if slip else [])
                q.append(sum(p * (reward(s2) + gamma * nxt(s2)) for p, s2 in outs))
            V2[r, c], pol[r, c] = max(q), int(np.argmax(q))
    return V2, pol
def draw_grid(ax, V, pol=None, numbers=False, vmax=10):
    M = np.ma.masked_where(np.isin(np.arange(ROWS * COLS).reshape(ROWS, COLS), [r * COLS + c for r, c in WALLS]), V)
    ax.imshow(M, cmap='RdYlGn', vmin=-vmax, vmax=vmax); ax.set_xticks([]); ax.set_yticks([])
    for (r, c) in WALLS: ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="#555555"))
    ax.text(GOAL[1], GOAL[0], "GOAL\n+10", ha='center', va='center', fontsize=7.5 if not numbers else 9, color='white', fontweight='bold')
    ax.text(PIT[1], PIT[0], "PIT\n−10", ha='center', va='center', fontsize=7.5 if not numbers else 9, color='white', fontweight='bold')
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) in WALLS or (r, c) in (GOAL, PIT): continue
            if numbers: ax.text(c, r + 0.27, f"{V[r, c]:.1f}", ha='center', va='center', fontsize=8, color=FG)
            if pol is not None and abs(V[r, c]) > 1e-9: ax.text(c, r - (0.12 if numbers else 0), ARROW[pol[r, c]], ha='center', va='center', fontsize=13 if numbers else 10, color=FG)

# 20 + 21 — value iteration ripples, and the final signposts
def value_iteration():
    V = np.zeros((ROWS, COLS)); snaps = {0: (V.copy(), None)}
    for k in range(1, 40):
        V, pol = sweep(V, 0.9);
        if k in (1, 2, 4, 39): snaps[k] = (V.copy(), pol.copy())
    V[GOAL], V[PIT] = 10, -10
    fig, axes = new(1, 5, figsize=(13, 3.0))
    for ax, k in zip(axes, [0, 1, 2, 4, 39]):
        Vk = snaps[k][0].copy(); Vk[GOAL], Vk[PIT] = 10, -10
        draw_grid(ax, Vk, snaps[k][1] if k == 39 else None); ax.set_title(f"sweep {k}" if k < 39 else "converged (+ best action)", fontsize=10, color=FG)
    suptitle(fig, "Value iteration: the goal's value ripples outward, one ring per sweep", y=1.05)
    save(fig, "value_iteration_ripples.png")
    fig, ax = new(figsize=(5.6, 5.2)); Vf = snaps[39][0].copy(); Vf[GOAL], Vf[PIT] = 10, -10; draw_grid(ax, Vf, snaps[39][1], numbers=True)
    ax.set_title("Values are the signposts; the policy just walks uphill", fontsize=11.5, color=FG, pad=10)
    save(fig, "value_heatmap_policy_arrows.png"); print("    V(start bottom-left) =", round(float(snaps[39][0][5, 0]), 3))

# 13 — Bellman backup is a contraction
def bellman_contraction():
    fig, ax = new(figsize=(7.2, 4.2))
    for g, col in [(0.5, GREEN), (0.9, BLUE), (0.99, RED)]:
        V = np.zeros((ROWS, COLS))
        for _ in range(6000): V, _p = sweep(V, g, slip=0.2, continuing=True)
        Vstar, V, err = V, np.zeros((ROWS, COLS)), []
        for k in range(120): err.append(np.abs(V - Vstar).max()); V, _p = sweep(V, g, slip=0.2, continuing=True)
        ax.semilogy(err, color=col, lw=2.2, label=f"γ = {g}"); ax.semilogy(err[0] * g ** np.arange(120), color=col, lw=1, ls=':')
        hit = next((i for i, e in enumerate(err) if e < 1e-3), None); print(f"    gamma={g}: sweeps to error<1e-3: {hit}")
    style(ax, "Every sweep shrinks the error by at least γ — that's why iterating works", "sweeps of value iteration", "largest error in any state (log scale)")
    ax.legend(fontsize=9, title="solid = measured, dotted = γᵏ bound", title_fontsize=8); ax.set_ylim(1e-6, 400)
    save(fig, "bellman_contraction.png")

# 14 — curse of dimensionality, both faces
def curse():
    rng = np.random.default_rng(1); fig, axes = new(1, 2, figsize=(12, 3.9)); d = np.arange(1, 13)
    ax = axes[0]; ax.semilogy(d, 10.0 ** d, color=RED, lw=2.3, marker='o', ms=5); ax.axhline(1e9, color=GREY, ls='--', lw=1.2); ax.text(1.1, 2e9, "a billion table rows", fontsize=8.5, color=GREY)
    style(ax, "Every new dimension multiplies the table", "number of state variables (10 levels each)", "states to fill in")
    dims = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]; ratio = []
    for D in dims:
        pts = rng.uniform(0, 1, (500, D)); q = rng.uniform(0, 1, (20, D)); dist = np.sqrt(((pts[None] - q[:, None]) ** 2).sum(-1)); ratio.append((dist.min(1) / dist.max(1)).mean())
    ax = axes[1]; ax.semilogx(dims, ratio, color=BLUE, lw=2.3, marker='o', ms=5); ax.axhline(1, color=GREY, ls=':', lw=1.2)
    style(ax, "…and 'nearest neighbour' stops meaning anything", "dimensions", "nearest distance ÷ farthest distance"); ax.set_ylim(0, 1.05)
    save(fig, "curse_of_dimensionality.png"); print("    nearest/farthest:", {D: round(r, 3) for D, r in zip(dims, ratio)})

# 18 — three reward designs on the same maze
def reward_shaping():
    n, goal, cap, eps, alpha, g, EP, SEEDS = 8, (7, 7), 200, 0.1, 0.5, 0.97, 150, 20
    dist = lambda s: abs(s[0] - goal[0]) + abs(s[1] - goal[1])
    phi = lambda s: 0.0 if s == goal else 1.0 - dist(s) / 14.0          # rises toward the goal, zero at the terminal state
    def run(kind, seed):
        rng = np.random.default_rng(seed); Q = np.zeros((n, n, 4)); out = []
        for ep in range(EP):
            s, steps = (0, 0), 0
            while s != goal and steps < cap:
                a = int(rng.integers(4)) if rng.random() < eps else int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
                s2 = (min(max(s[0] + MOVES[a][0], 0), n - 1), min(max(s[1] + MOVES[a][1], 0), n - 1))
                r = 1.0 if s2 == goal else 0.0
                if kind == "potential": r += g * phi(s2) - phi(s)
                if kind == "negpot": r += g * (-0.1 * dist(s2)) - (-0.1 * dist(s))
                if kind == "naive" and dist(s2) < dist(s): r += 0.2
                Q[s][a] += alpha * (r + (0 if s2 == goal else g * Q[s2].max()) - Q[s][a]); s = s2; steps += 1
            out.append(steps)
        return out
    fig, ax = new(figsize=(7.4, 4.2))
    for kind, col, lbl in [("sparse", GREY, "sparse: +1 at the goal, nothing else"), ("potential", GREEN, "potential-based shaping"), ("naive", RED, "naive bonus: +0.2 for every step closer")]:
        m = np.mean([run(kind, sd) for sd in range(SEEDS)], 0); ax.plot(smooth(m, 5), color=col, lw=2.2, label=lbl); print(f"    {kind}: mean steps over last 20 episodes = {m[-20:].mean():.1f}")
    m = np.mean([run("negpot", sd) for sd in range(SEEDS)], 0); print(f"    (for the note) potential = -0.1*distance: mean steps over last 20 episodes = {m[-20:].mean():.1f}")
    ax.axhline(14, color=GREY, ls=':', lw=1.2); ax.text(100, 24, "shortest possible: 14 steps", fontsize=8.5, color=GREY); ax.text(100, 190, "200 = gave up (episode cap)", fontsize=8.5, color=GREY)
    style(ax, "Same maze, three reward designs", "episode", "steps to reach the goal"); ax.legend(fontsize=8.5, loc='right', bbox_to_anchor=(0.99, 0.33)); ax.set_ylim(0, 210)
    save(fig, "reward_shaping_learning_curves.png")

# 19 — discounting
def discount():
    t = np.arange(0, 3000); fig, ax = new(figsize=(7.4, 4.1))
    for g, col in [(0.5, GREEN), (0.9, BLUE), (0.99, ORANGE), (0.999, RED)]:
        ax.semilogx(t + 1, g ** t, color=col, lw=2.2, label=f"γ = {g}   → horizon ≈ {1 / (1 - g):.0f} steps"); ax.plot([1 / (1 - g)], [g ** (1 / (1 - g) - 1)], 'o', color=col, ms=7)
    ax.axhline(np.exp(-1), color=GREY, ls=':', lw=1.2); ax.text(1.1, 0.40, "weight ≈ 0.37 — the 'effective horizon'", fontsize=8.5, color=GREY)
    style(ax, "γ decides how far ahead the agent can see", "steps into the future (log scale)", "weight on a reward at that step"); ax.legend(fontsize=8.5, loc='upper right')
    save(fig, "discount_curves.png"); print("    100 at 50 steps:", {g: round(100 * g ** 50, 2) for g in (0.9, 0.99)})

# 22 — eligibility traces
def traces():
    k = np.arange(0, 20); fig, axes = new(1, 4, figsize=(12, 2.9), sharey=True)
    for ax, lam, col in zip(axes, [0.0, 0.5, 0.9, 1.0], [GREY, BLUE, ORANGE, RED]):
        ax.bar(-k, (0.99 * lam) ** k, color=col, alpha=.9); style(ax, f"λ = {lam}" + ("  (pure TD)" if lam == 0 else "  (Monte Carlo)" if lam == 1 else ""), "steps before the surprise", "share of the credit" if lam == 0 else None)
    suptitle(fig, "λ decides how far back the credit (or blame) reaches", y=1.08)
    save(fig, "eligibility_traces.png")

# 23 — the tiger problem
def tiger():
    nn = np.arange(0, 6); b = 0.85 ** nn / (0.85 ** nn + 0.15 ** nn); fig, axes = new(1, 2, figsize=(11.5, 3.8))
    ax = axes[0]; ax.bar(nn, b, color=BLUE, alpha=.9); ax.axhline(100 / 110, color=RED, ls='--', lw=1.4); ax.text(-0.45, 0.918, "break-even 0.909", fontsize=8.5, color=RED, bbox=dict(facecolor=BG, edgecolor='none', pad=.5, alpha=.9))
    for i, v in zip(nn, b): ax.text(i, v - 0.07, f"{v:.3f}", ha='center', fontsize=8.5, color='white')
    style(ax, "Each consistent growl sharpens the belief", "times you've heard the tiger on the left", "belief: tiger is on the left"); ax.set_ylim(0.4, 1.02)
    bb = np.linspace(0.5, 1, 200); ax = axes[1]; ax.plot(bb, 110 * bb - 100, color=BLUE, lw=2.3); ax.axhline(0, color=GREY, lw=1); ax.axhline(-1, color=ORANGE, ls=':', lw=1.4); ax.text(0.505, 2.2, "cost of listening once more: −1", fontsize=8.5, color=ORANGE)
    for bv, col in [(0.85, RED), (b[2], GREEN)]:
        ax.plot([bv], [110 * bv - 100], 'o', color=col, ms=8); ax.annotate(f"belief {bv:.2f} → {110 * bv - 100:+.1f}", xy=(bv, 110 * bv - 100), xytext=((0.55, -24) if bv < 0.9 else (0.66, 12.5)), fontsize=9, color=col, arrowprops=dict(arrowstyle='->', color=col))
    style(ax, "At 85% sure, opening the door still loses money", "belief that the tiger is behind the left door", "expected payoff of opening the right door"); ax.set_ylim(-48, 18)
    save(fig, "tiger_belief_value.png"); print("    beliefs:", np.round(b, 4), " EV@0.85:", 110 * 0.85 - 100, " EV@2 growls:", round(110 * b[2] - 100, 2))

if __name__ == "__main__":
    for f in (bayes_filter, ekf_banana, particle_filter, hmm_umbrella, dp_commute, value_iteration, bellman_contraction, curse, reward_shaping, discount, traces, tiger): f()
