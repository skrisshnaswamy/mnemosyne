"""Diffusion, samplers, flow matching, rectified flow, guidance, latents, FID.
No neural networks: for a Gaussian-mixture target the score and the flow-matching velocity are EXACT, so every sampler here is the real algorithm.
Run:  cd sourcecode && python3 gm_diffusion_flows.py"""
from ds_style import *

BMIN, BMAX = 0.1, 20.0
beta = lambda t: BMIN + t * (BMAX - BMIN)
abar = lambda t: np.exp(-(BMIN * t + 0.5 * (BMAX - BMIN) * t ** 2))          # signal kept at time t (VP-SDE)

def score(X, t, mu, w, s):
    """Exact score of the diffused mixture at time t. X: (n,d); mu: (k,d)."""
    a = abar(t); v = a * s ** 2 + (1 - a); d = X[:, None, :] - np.sqrt(a) * mu[None]
    logp = np.log(w)[None] - 0.5 * (d ** 2).sum(-1) / v; r = np.exp(logp - logp.max(1, keepdims=True)); r /= r.sum(1, keepdims=True)
    return -(r[:, :, None] * d).sum(1) / v

def reverse(X, mu, w, s, n, rng=None, t0=1.0, t1=1e-3, sfun=None, keep=None):
    """rng given -> reverse SDE (stochastic); rng None -> probability-flow ODE (deterministic). Euler."""
    ts = np.linspace(t0, t1, n + 1); out = {}
    for i in range(n):
        t, dt = ts[i], ts[i] - ts[i + 1]; b = beta(t); sc = (sfun or score)(X, t, mu, w, s) if sfun is None else sfun(X, t)
        if rng is None: X = X + (0.5 * b * X + 0.5 * b * sc) * dt
        else: X = X + (0.5 * b * X + b * sc) * dt + np.sqrt(b * dt) * rng.normal(0, 1, X.shape)
        if keep and i + 1 in keep: out[i + 1] = X.copy()
    return (X, out) if keep else X

M1, W1, S1 = np.array([[-2.0], [2.0]]), np.array([0.4, 0.6]), 0.35
def true_1d(n, rng): k = rng.random(n) < W1[1]; return np.where(k, 2.0, -2.0) + rng.normal(0, S1, n)
def w1(a, b): return np.mean(np.abs(np.sort(a) - np.sort(b)))

# 8 — the forward process
def forward_density():
    rng = np.random.default_rng(2); x = np.linspace(-4.5, 4.5, 400); ts = np.linspace(0, 1, 300); D = np.zeros((len(x), len(ts)))
    for j, t in enumerate(ts):
        a = abar(t); v = a * S1 ** 2 + 1 - a
        D[:, j] = sum(wk * np.exp(-0.5 * (x - np.sqrt(a) * m[0]) ** 2 / v) / np.sqrt(2 * np.pi * v) for wk, m in zip(W1, M1))
    fig, ax = new(figsize=(8, 4.3)); ax.imshow(D ** 0.6, origin='lower', aspect='auto', extent=[0, 1, -4.5, 4.5], cmap='Blues')
    for x0 in true_1d(7, rng):
        xs = [x0]
        for t in ts[:-1]: xs.append(xs[-1] - 0.5 * beta(t) * xs[-1] * (ts[1] - ts[0]) + np.sqrt(beta(t) * (ts[1] - ts[0])) * rng.normal())
        ax.plot(ts, xs, color=ORANGE, lw=1, alpha=.9)
    ax.text(0.01, 3.7, "data: two sharp bumps", fontsize=9, color=FG); ax.text(0.7, 3.7, "pure noise: one N(0,1) blob", fontsize=9, color=FG)
    style(ax, "The forward process: stir in noise until nothing of the data is left", "diffusion time t", "x", grid=False)
    save(fig, "forward_diffusion_density.png"); print(f"    signal kept: t=.25 {np.sqrt(abar(.25)):.2f}  t=.5 {np.sqrt(abar(.5)):.2f}  t=.75 {np.sqrt(abar(.75)):.3f}  t=1 {np.sqrt(abar(1.)):.4f}")

# 9 — noise schedules
def schedules():
    T = 1000; t = np.arange(1, T + 1); lin = np.cumprod(1 - np.linspace(1e-4, 0.02, T)); f = np.cos((t / T + 0.008) / 1.008 * np.pi / 2) ** 2; cos = f / f[0]
    fig, axes = new(1, 2, figsize=(11.5, 3.8))
    for ab, col, lbl in [(lin, RED, "linear β  (DDPM, 2020)"), (cos, BLUE, "cosine  (Improved DDPM, 2021)")]:
        axes[0].plot(t, ab, color=col, lw=2.3, label=lbl); axes[1].plot(t, np.log(ab / (1 - ab)), color=col, lw=2.3)
    axes[0].axhline(0.05, color=GREY, ls=':'); axes[0].text(20, 0.08, "below here the image is essentially gone", fontsize=8.3, color=GREY)
    style(axes[0], "ᾱ_t: how much of the original image survives", "step t of 1000", None); axes[0].legend(fontsize=9); style(axes[1], "log signal-to-noise ratio", "step t of 1000", None); axes[1].set_ylim(-12, 10)
    suptitle(fig, "The noise schedule decides where the model spends its effort", y=1.04)
    save(fig, "noise_schedules.png"); print(f"    step where alpha_bar<0.05: linear {np.argmax(lin < .05) + 1}, cosine {np.argmax(cos < .05) + 1}; alpha_bar@500: lin {lin[499]:.3f} cos {cos[499]:.3f}")

# 10 — reverse diffusion in 2-D
def reverse_2d():
    rng = np.random.default_rng(5); ang = np.arange(6) * np.pi / 3; mu = 2.2 * np.column_stack([np.cos(ang), np.sin(ang)]); w = np.ones(6) / 6; s = 0.12
    X, snaps = reverse(rng.normal(0, 1, (2500, 2)), mu, w, s, 500, rng=rng, keep={200, 350, 450, 500}); snaps = {0: None, **snaps}
    fig, axes = new(1, 5, figsize=(14.5, 3.2)); first = np.random.default_rng(5).normal(0, 1, (2500, 2))
    for ax, (k, P), lbl in zip(axes, snaps.items(), ["t = 1.0  pure noise", "t = 0.6", "t = 0.3", "t = 0.1", "t = 0  data"]):
        P = first if P is None else P; ax.scatter(P[:, 0], P[:, 1], s=2, color=BLUE, alpha=.6); ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.6, 3.6); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); style(ax, lbl, grid=False)
    d = np.linalg.norm(X[:, None] - mu[None], axis=2); hit = (d.min(1) < 3 * s * 1.5).mean(); share = np.bincount(d.argmin(1), minlength=6) / len(X)
    suptitle(fig, "Reverse diffusion: 500 small denoising steps turn a blob of noise into six sharp modes", y=1.06)
    save(fig, "reverse_diffusion_2d.png"); print(f"    {hit * 100:.1f}% of samples land on a mode; share per mode {np.round(share, 3)} (true 0.167 each)")

# 11 — stochastic vs deterministic paths
def sde_vs_ode():
    n = 400; ts = np.linspace(1, 1e-3, n + 1); starts = np.linspace(-2.4, 2.4, 22)[:, None]; fig, axes = new(1, 2, figsize=(12, 3.9), sharey=True)
    for ax, stoch, ttl in [(axes[0], True, "reverse SDE (DDPM-style)\nfresh noise at every step"), (axes[1], False, "probability-flow ODE (DDIM-style)\nsame start → same result, always")]:
        rng = np.random.default_rng(7); X = starts.copy(); path = [X[:, 0].copy()]
        for i in range(n):
            t, dt = ts[i], ts[i] - ts[i + 1]; b = beta(t); sc = score(X, t, M1, W1, S1)
            X = X + (0.5 * b * X + (b if stoch else 0.5 * b) * sc) * dt + (np.sqrt(b * dt) * rng.normal(0, 1, X.shape) if stoch else 0); path.append(X[:, 0].copy())
        P = np.array(path)
        for j in range(P.shape[1]): ax.plot(1 - ts, P[:, j], color=BLUE if P[-1, j] > 0 else ORANGE, lw=1, alpha=.85)
        style(ax, ttl, "noise  →  data", "x" if stoch else None)
    suptitle(fig, "Two ways back from noise — and they produce the same distribution", y=1.1)
    save(fig, "sde_vs_ode_paths.png")

# flow-matching velocity for x_t = (1-t) x0 + t x1, x0 ~ N(0,1), x1 ~ mixture  (exact)
def fm_velocity(X, t):
    var = (1 - t) ** 2 + t ** 2 * S1 ** 2; d = X[:, None, :] - t * M1[None]; logp = np.log(W1)[None] - 0.5 * (d ** 2).sum(-1) / var
    r = np.exp(logp - logp.max(1, keepdims=True)); r /= r.sum(1, keepdims=True); cov = t * S1 ** 2 - (1 - t)
    return (r[:, :, None] * (M1[None] + cov / var * d)).sum(1)
def fm_sample(X, n):
    ts = np.linspace(0, 1, n + 1); path = [X[:, 0].copy()]
    for i in range(n): X = X + fm_velocity(X, ts[i]) * (ts[i + 1] - ts[i]); path.append(X[:, 0].copy())
    return X, np.array(path), ts
def monotone_map(x0):                                   # the 1-D optimal-transport map N(0,1) -> mixture: match quantiles
    from math import erf
    g = np.linspace(-7, 7, 20001); cdf = sum(wk * 0.5 * (1 + np.vectorize(erf)((g - m[0]) / (S1 * np.sqrt(2)))) for wk, m in zip(W1, M1))
    u = 0.5 * (1 + np.vectorize(erf)(x0 / np.sqrt(2))); return np.interp(u, cdf, g)

# 13 — curved vs straight paths
def fm_paths():
    starts = np.linspace(-2.4, 2.4, 22)[:, None]; n = 400; ts = np.linspace(1, 1e-3, n + 1); X = starts.copy(); path = [X[:, 0].copy()]
    for i in range(n):
        t, dt = ts[i], ts[i] - ts[i + 1]; b = beta(t); X = X + (0.5 * b * X + 0.5 * b * score(X, t, M1, W1, S1)) * dt; path.append(X[:, 0].copy())
    Pd = np.array(path); _, Pf, tf = fm_sample(starts.copy(), 400); T = monotone_map(starts[:, 0]); fig, axes = new(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, (tt, P), ttl in zip(axes, [(1 - ts, Pd), (tf, Pf), (np.array([0, 1]), np.vstack([starts[:, 0], T]))], ["diffusion (probability-flow ODE)\nnothing… nothing… then a sharp swerve", "flow matching\ngentler, earlier, more even", "rectified flow (after 'reflow')\nperfectly straight — one step is exact"]):
        for j in range(P.shape[1]): ax.plot(tt, P[:, j], color=BLUE if P[-1, j] > 0 else ORANGE, lw=1.1, alpha=.9)
        style(ax, ttl, "noise  →  data", "x" if ax is axes[0] else None)
    suptitle(fig, "Same noise, same data, three different roads — and straighter roads need fewer steps", y=1.1)
    save(fig, "flow_matching_paths.png")

# 14 — why independent pairing gives crossing paths, and reflow un-crosses them
def reflow():
    rng = np.random.default_rng(3); x0 = rng.normal(0, 1, 60); x1 = true_1d(60, rng); fig, axes = new(1, 2, figsize=(11.5, 3.9), sharey=True)
    for a, b in zip(x0, x1): axes[0].plot([0, 1], [a, b], color=BLUE if b > 0 else ORANGE, lw=.9, alpha=.8)
    for a, b in zip(np.sort(x0), monotone_map(np.sort(x0))): axes[1].plot([0, 1], [a, b], color=BLUE if b > 0 else ORANGE, lw=.9, alpha=.8)
    style(axes[0], "noise paired with RANDOM data points\n→ lines cross; the averaged flow must curve", "noise  →  data", "x")
    style(axes[1], "after reflow: pairs from the model's own map\n→ nobody crosses; the flow can be straight", "noise  →  data", None)
    suptitle(fig, "Rectified flow: re-pair noise and data so the lines stop crossing", y=1.08)
    save(fig, "rectified_flow_reflow.png")

# 12 — how many steps does each sampler need?
def steps_vs_error():
    rng = np.random.default_rng(11); N = 20000; ref = true_1d(N, rng); floor = np.mean([w1(true_1d(N, rng), ref) for _ in range(5)]); steps = [1, 2, 4, 8, 16, 32, 64, 128, 256]; res = {k: [] for k in ("sde", "ode", "fm", "rf")}
    for n in steps:
        z = rng.normal(0, 1, (N, 1)); res["sde"].append(w1(reverse(z.copy(), M1, W1, S1, n, rng=rng)[:, 0], ref)); res["ode"].append(w1(reverse(z.copy(), M1, W1, S1, n)[:, 0], ref))
        res["fm"].append(w1(fm_sample(z.copy(), n)[0][:, 0], ref)); res["rf"].append(w1(monotone_map(z[:, 0]), ref))
    fig, ax = new(figsize=(7.8, 4.3))
    for k, col, lbl in [("sde", RED, "diffusion, stochastic sampler (DDPM-style)"), ("ode", ORANGE, "diffusion, deterministic ODE (DDIM-style)"), ("fm", BLUE, "flow matching"), ("rf", GREEN, "rectified flow (straight paths)")]:
        ax.loglog(steps, res[k], color=col, lw=2.2, marker='o', ms=5, label=lbl)
    ax.axhline(floor, color=GREY, ls=':', lw=1.3); ax.text(1.05, floor * 1.18, "as good as fresh real samples", fontsize=8.5, color=GREY)
    style(ax, "Straighter paths, fewer steps", "number of sampling steps (network evaluations)", "distance from the true distribution (Wasserstein-1)"); ax.legend(fontsize=8.5, loc='upper right')
    save(fig, "sampling_steps_vs_error.png"); print("    W1 by steps", steps); [print(f"      {k}: {np.round(v, 3)}") for k, v in res.items()]; print(f"      floor {floor:.3f}")

# 15 — classifier-free guidance
def cfg():
    rng = np.random.default_rng(9); muA = np.array([[-1.6, 0.9], [-0.2, -0.3]]); muB = np.array([[1.6, -0.6], [0.6, 1.4]]); mu = np.vstack([muA, muB]); s = 0.5; wA = np.array([.5, .5]); wU = np.ones(4) / 4
    fig, axes = new(1, 4, figsize=(14, 3.7)); out = {}
    for ax, g in zip(axes, [0.0, 1.0, 3.0, 8.0]):
        sf = lambda X, t, g=g: score(X, t, mu, wU, s) + g * (score(X, t, muA, wA, s) - score(X, t, mu, wU, s))
        X = reverse(rng.normal(0, 1, (1500, 2)), None, None, None, 400, rng=rng, sfun=sf)
        for m, col in [(muA, BLUE), (muB, GREY)]:
            for c in m: ax.add_patch(plt.Circle(c, 2 * s, fill=False, color=col, lw=1.6, ls='--'))
        ax.scatter(X[:, 0], X[:, 1], s=3, color=RED, alpha=.5); ax.set_xlim(-3.6, 3.6); ax.set_ylim(-2.6, 3.2); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        dA = np.linalg.norm(X[:, None] - muA[None], axis=2).min(1); dB = np.linalg.norm(X[:, None] - muB[None], axis=2).min(1); inA = (dA < dB).mean(); sd = X.std(0).mean(); out[g] = (inA, sd, (dA < 2 * s).mean())
        style(ax, f"guidance w = {g:g}" + ("  (prompt ignored)" if g == 0 else "  (plain conditional)" if g == 1 else ""), grid=False)
    axes[0].text(-3.5, 2.85, "blue = what the prompt asked for", fontsize=8, color=BLUE); axes[0].text(-3.5, -2.45, "grey = everything else", fontsize=8, color=GREY)
    suptitle(fig, "Classifier-free guidance: turn it up and samples obey the prompt — then overshoot and lose variety", y=1.05)
    save(fig, "cfg_guidance_scale.png"); [print(f"    w={g:g}: on-prompt {a * 100:.0f}%  inside the target circles {c * 100:.0f}%  spread {b:.2f}") for g, (a, b, c) in out.items()]

# 16 — pixels vs latents
def latent_cost():
    fig, ax = new(figsize=(7.4, 3.7)); vals = [512 * 512 * 3, 64 * 64 * 4, 1024 * 1024 * 3, 128 * 128 * 4]; names = ["512² pixels", "its 64×64×4 latent", "1024² pixels", "its 128×128×4 latent"]
    b = ax.barh(names[::-1], vals[::-1], color=[GREEN, RED, GREEN, RED])
    for r, v in zip(b, vals[::-1]): ax.text(v * 1.05, r.get_y() + r.get_height() / 2, f"{v:,}", va='center', fontsize=9)
    ax.set_xscale('log'); ax.set_xlim(5e3, 3e7); style(ax, "Latent diffusion: denoise 48× fewer numbers, then decode once", "numbers the denoiser must process at every one of its steps (log scale)", None, grid=False)
    save(fig, "latent_vs_pixel_cost.png"); print(f"    ratio {vals[0] / vals[1]:.0f}x; attention cost ratio if one token per position: {(vals[0] / 3 / (vals[1] / 4)) ** 2:.0f}x")

# 17 — what FID can and can't see
def fid():
    rng = np.random.default_rng(4); ang = np.arange(8) * np.pi / 4; C = 3 * np.column_stack([np.cos(ang), np.sin(ang)])
    gen = lambda idx, s, n=4000: C[rng.choice(idx, n)] + rng.normal(0, s, (n, 2)); real = gen(np.arange(8), 0.3)
    def fd(A, B):
        m1, m2, c1, c2 = A.mean(0), B.mean(0), np.cov(A.T), np.cov(B.T); ev = np.linalg.eigvals(c1 @ c2).real.clip(0); return float(((m1 - m2) ** 2).sum() + np.trace(c1) + np.trace(c2) - 2 * np.sqrt(ev).sum())
    sets = [("a good model", gen(np.arange(8), 0.3)), ("blurry: right places,\ntoo much noise", gen(np.arange(8), 0.9)), ("lost one SIDE\n(4 neighbouring modes)", gen(np.arange(4), 0.3)), ("lost every OTHER mode\n(4 of 8 missing!)", gen(np.arange(0, 8, 2), 0.3)), ("memorised a\nsingle mode", gen(np.array([0]), 0.3))]
    fig, axes = new(1, 5, figsize=(15, 3.6))
    for ax, (lbl, G) in zip(axes, sets):
        ax.scatter(real[::4, 0], real[::4, 1], s=3, color=GREY, alpha=.5); ax.scatter(G[::4, 0], G[::4, 1], s=3, color=RED, alpha=.5); ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        v = fd(real, G); style(ax, f"{lbl}\nFréchet distance = {v:.2f}", grid=False); print(f"    {lbl.splitlines()[0]}: {v:.3f}")
    suptitle(fig, "FID only compares a mean and a covariance — so it can be blind to half the modes going missing   (grey = real, red = generated)", y=1.1)
    save(fig, "fid_failure_modes.png")

if __name__ == "__main__":
    for f in (forward_density, schedules, reverse_2d, sde_vs_ode, fm_paths, reflow, steps_vs_error, cfg, latent_cost, fid): f()
