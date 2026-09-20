"""Generative-model foundations: likelihood, KL, flows, energy, latents, scores, Langevin. Computed / simulated.
Run:  cd sourcecode && python3 gm_foundations.py"""
from ds_style import *

gauss = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

# 1 — maximum likelihood on a Gaussian mean
def mle():
    rng = np.random.default_rng(3); data = rng.normal(3.0, 1.0, 30); mus = np.linspace(0, 6, 300)
    ll = np.array([np.sum(np.log(gauss(data, m, 1.0))) for m in mus]); best = mus[ll.argmax()]; x = np.linspace(-1, 7, 400)
    fig, axes = new(1, 2, figsize=(12, 3.9)); ax = axes[0]; ax.plot(data, np.zeros_like(data), '|', color=FG, ms=18, mew=1.5, label="30 observed data points")
    for m, col, lbl in [(1.0, RED, "guess μ = 1  → the data would be very surprising"), (best, GREEN, f"μ = {best:.2f}  → the data is least surprising"), (5.0, ORANGE, "guess μ = 5")]:
        ax.plot(x, gauss(x, m, 1), color=col, lw=2.2, label=lbl)
    style(ax, "slide the curve until the data looks as unsurprising as possible", "x", "density"); ax.legend(fontsize=8.3, loc='upper left'); ax.set_ylim(-0.03, 0.62)
    ax = axes[1]; ax.plot(mus, ll, color=BLUE, lw=2.4); ax.axvline(data.mean(), color=GREEN, ls=':', lw=1.5); ax.plot([best], [ll.max()], 'o', color=GREEN, ms=8)
    ax.text(data.mean() + 0.1, ll.min() + 5, f"peak = the sample mean, {data.mean():.2f}", fontsize=9, color=GREEN); style(ax, "log-likelihood of the data, for every possible μ", "μ", "log-likelihood")
    suptitle(fig, "Maximum likelihood: choose the parameters that make your data least surprising", y=1.04)
    save(fig, "mle_gaussian_fit.png"); print(f"    sample mean {data.mean():.3f}; argmax {best:.3f}; LL(1)={ll[np.abs(mus - 1).argmin()]:.1f} LL(best)={ll.max():.1f}")

# 2 — forward vs reverse KL fitting one Gaussian to two bumps
def kl_modes():
    x = np.linspace(-8, 8, 4000); dx = x[1] - x[0]; p = 0.5 * gauss(x, -2.5, 0.6) + 0.5 * gauss(x, 2.5, 0.6)
    m_f = np.sum(x * p) * dx; s_f = np.sqrt(np.sum((x - m_f) ** 2 * p) * dx); best = (1e9, 0, 0)
    for m in np.linspace(-4, 4, 161):
        for s in np.linspace(0.2, 4, 96):
            q = gauss(x, m, s); kl = np.sum(q * (np.log(q + 1e-300) - np.log(p + 1e-300))) * dx
            if kl < best[0] - 1e-9: best = (kl, m, s)
    _, m_r, s_r = best; fig, ax = new(figsize=(7.6, 4.1)); ax.fill_between(x, p, color=GREY, alpha=.35, label="the real data: two bumps")
    ax.plot(x, gauss(x, m_f, s_f), color=ORANGE, lw=2.4, label=f"forward KL (maximum likelihood): cover everything  (σ = {s_f:.1f})")
    ax.plot(x, gauss(x, m_r, s_r), color=BLUE, lw=2.4, label=f"reverse KL: pick one bump and nail it  (σ = {s_r:.1f})")
    ax.annotate("puts mass where there\nis NO data → blurry samples", xy=(0, gauss(0, m_f, s_f)), xytext=(-1.35, 0.43), fontsize=8.5, color=ORANGE, arrowprops=dict(arrowstyle='->', color=ORANGE))
    ax.annotate("ignores half the data\n→ mode collapse", xy=(-2.5, 0.02), xytext=(-7.7, 0.36), fontsize=8.5, color=BLUE, arrowprops=dict(arrowstyle='->', color=BLUE))
    style(ax, "A model too simple for the data must choose how to be wrong", "x", "density"); ax.legend(fontsize=8.3, loc='upper right'); ax.set_xlim(-8, 8); ax.set_ylim(0, 0.75)
    save(fig, "kl_mode_covering_vs_seeking.png"); print(f"    forward: m={m_f:.2f} s={s_f:.2f} | reverse: m={m_r:.2f} s={s_r:.2f}")

# 3 — change of variables
def flow():
    rng = np.random.default_rng(1); f = lambda z: z + 1.6 * np.tanh(2.0 * z); df = lambda z: 1 + 3.2 / np.cosh(2.0 * z) ** 2
    z = np.linspace(-3.5, 3.5, 2000); xs = f(z); px = gauss(z, 0, 1) / df(z); samp = f(rng.normal(0, 1, 40000))
    fig, axes = new(1, 3, figsize=(12.5, 3.7)); axes[0].fill_between(z, gauss(z, 0, 1), color=GREY, alpha=.5); style(axes[0], "1 · simple noise  z ~ N(0,1)", "z", "density")
    axes[1].plot(z, xs, color=BLUE, lw=2.4); axes[1].plot(z, z, color=GREY, ls=':', lw=1); style(axes[1], "2 · an invertible warp  x = f(z)", "z", "x")
    axes[1].annotate("steep here → points get\nspread apart → density DROPS", xy=(0, 0), xytext=(-3.4, 2.6), fontsize=8.3, color=RED, arrowprops=dict(arrowstyle='->', color=RED))
    axes[2].hist(samp, bins=120, density=True, color=GREY, alpha=.45, label="40,000 samples pushed through f"); axes[2].plot(xs, px, color=GREEN, lw=2.4, label="exact density = p(z) ÷ |f′(z)|")
    style(axes[2], "3 · a two-bump distribution — with an exact density", "x", None); axes[2].legend(fontsize=8.3, loc='upper center'); axes[2].set_ylim(0, 0.5)
    suptitle(fig, "Normalizing flow: warp simple noise into data, and keep the books on how much you stretched", y=1.05)
    save(fig, "normalizing_flow_change_of_variables.png"); print(f"    f'(0)={df(0):.2f}  p_x(0)={gauss(0, 0, 1) / df(0):.3f} vs p_z(0)={gauss(0, 0, 1):.3f}; integral p_x = {np.sum(0.5 * (px[1:] + px[:-1]) * np.diff(xs)):.4f}")

# 4 — energy vs density
def ebm():
    x = np.linspace(-3.6, 3.6, 3000); E = (x ** 2 - 4) ** 2 / 6 + 0.35 * x; un = np.exp(-E); Z = np.sum(un) * (x[1] - x[0]); dE = np.gradient(E, x)
    fig, axes = new(3, 1, figsize=(7.4, 6.6), sharex=True)
    axes[0].plot(x, E, color=RED, lw=2.4); style(axes[0], "energy E(x): any function you like — low = plausible", None, "energy"); axes[0].set_ylim(-1.5, 8)
    axes[1].fill_between(x, un / Z, color=BLUE, alpha=.35); axes[1].plot(x, un / Z, color=BLUE, lw=2); style(axes[1], f"probability = exp(−E) ÷ Z      (here Z = {Z:.2f}; in 1-D you can integrate it. In a million dimensions you can't.)", None, "density")
    axes[2].plot(x, -dE, color=GREEN, lw=2.4); axes[2].axhline(0, color=GREY, lw=1); style(axes[2], "score = −dE/dx: 'which way is more plausible?'  —  Z has vanished", "x", "score"); axes[2].set_ylim(-8, 8)
    fig.tight_layout(); save(fig, "ebm_energy_vs_density.png"); print(f"    Z={Z:.3f}; minima near x={x[E.argmin()]:.2f} and {x[x > 0][E[x > 0].argmin()]:.2f}")

# 5 — AE vs VAE latent space (illustrative)
def vae_latent():
    rng = np.random.default_rng(4); cents = np.array([[-3.2, 2.4], [3.4, 1.6], [0.2, -3.3]]); cols = [BLUE, ORANGE, GREEN]; fig, axes = new(1, 2, figsize=(11.5, 4.6))
    for k, (c, col) in enumerate(zip(cents, cols)): axes[0].scatter(*(c + rng.normal(0, 0.22, (70, 2))).T, s=9, color=col)
    axes[0].plot([0.3], [0.4], 'X', color=RED, ms=14); axes[0].annotate("sample a latent here →\nthe decoder has never seen\nanything like it: garbage", xy=(0.3, 0.4), xytext=(-4.6, -1.9), fontsize=8.8, color=RED, arrowprops=dict(arrowstyle='->', color=RED))
    style(axes[0], "plain autoencoder: tight islands, empty sea", "latent dim 1", "latent dim 2")
    for k, (c, col) in enumerate(zip(cents * 0.42, cols)):
        pts = c + rng.normal(0, 0.16, (70, 2)); axes[1].scatter(*pts.T, s=9, color=col)
        for p in pts[:22]: axes[1].add_patch(plt.Circle(p, 0.55, color=col, alpha=.045))
    for r in (1, 2): axes[1].add_patch(plt.Circle((0, 0), r, fill=False, color=GREY, ls='--', lw=1.2))
    axes[1].text(1.5, 1.75, "the N(0, I) prior\nyou will sample from", fontsize=8.5, color=GREY); style(axes[1], "VAE: every point is a fuzzy blob, all pulled toward N(0, I)", "latent dim 1", None)
    for ax in axes: ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect('equal')
    suptitle(fig, "Why you can sample from a VAE but not from an autoencoder  (illustrative)", y=1.0)
    save(fig, "vae_latent_holes.png")

# shared 2-D mixture
MU = np.array([[-2.2, -1.0], [2.0, -1.4], [0.2, 2.2]]); W = np.array([0.2, 0.3, 0.5]); SIG = 0.55
def mix_score(X, mu=MU, w=W, s=SIG):
    d = X[:, None, :] - mu[None]; logp = np.log(w)[None] - 0.5 * (d ** 2).sum(-1) / s ** 2; r = np.exp(logp - logp.max(1, keepdims=True)); r /= r.sum(1, keepdims=True)
    return -(r[:, :, None] * d).sum(1) / s ** 2, r

# 6 — the score field
def score_field():
    g = np.linspace(-4.5, 4.5, 200); XX, YY = np.meshgrid(g, g); P = np.zeros_like(XX)
    for m, w in zip(MU, W): P += w * np.exp(-0.5 * ((XX - m[0]) ** 2 + (YY - m[1]) ** 2) / SIG ** 2)
    q = np.linspace(-4.2, 4.2, 19); QX, QY = np.meshgrid(q, q); S, _ = mix_score(np.column_stack([QX.ravel(), QY.ravel()])); n = np.linalg.norm(S, axis=1, keepdims=True); S = S / np.maximum(n, 1e-9) * np.minimum(n, 6) / 6
    fig, ax = new(figsize=(6.4, 5.8)); ax.contourf(XX, YY, P, levels=12, cmap='Blues', alpha=.85); ax.quiver(QX, QY, S[:, 0].reshape(QX.shape), S[:, 1].reshape(QX.shape), color=FG, scale=26, width=0.004)
    ax.set_aspect('equal'); style(ax, "The score: at every point, an arrow toward 'more like the data'", None, None, grid=False); ax.set_xticks([]); ax.set_yticks([])
    save(fig, "score_field_mixture.png")

# 7 — Langevin dynamics following that field
def langevin():
    rng = np.random.default_rng(0); X = rng.uniform(-4.3, 4.3, (1500, 2)); eps = 0.01; snaps = {0: X.copy()}
    for k in range(1, 2001):
        S, _ = mix_score(X); X = X + eps * S + np.sqrt(2 * eps) * rng.normal(0, 1, X.shape)
        if k in (10, 100, 2000): snaps[k] = X.copy()
    fig, axes = new(1, 4, figsize=(13, 3.5))
    for ax, (k, P) in zip(axes, snaps.items()):
        ax.scatter(P[:, 0], P[:, 1], s=3, color=BLUE, alpha=.6); ax.plot(MU[:, 0], MU[:, 1], '+', color=RED, ms=11, mew=2); ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        style(ax, "start: pure noise" if k == 0 else f"after {k} steps", grid=False)
    _, r = mix_score(X); frac = np.bincount(r.argmax(1), minlength=3) / len(X)
    suptitle(fig, "Langevin dynamics: step along the arrows, add a little noise, repeat — and noise becomes data", y=1.04)
    save(fig, "langevin_sampling_steps.png"); print(f"    share per mode after 2000 steps: {np.round(frac, 3)}  vs true weights {W}")

if __name__ == "__main__":
    for f in (mle, kl_modes, flow, ebm, vae_latent, score_field, langevin): f()
