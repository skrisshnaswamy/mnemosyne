"""Control-theory diagrams. Every curve here is a simulation, not a sketch.
Run:  cd sourcecode && python3 ds_control.py"""
from ds_style import *

# ---------------------------------------------------------------- the cruise-control plant used by several charts
def sim_car(T=60.0, dt=0.01, Kp=5.0, Ki=0.0, Kd=0.0, target=lambda t: 10.0, dist=lambda t: 0.0,
            ulim=None, antiwindup=False, ff=lambda t: 0.0, tau1=1.5, tau2=8.0, x0=0.0, f0=0.0, i0=0.0):
    """Speed deviation x (mph above 50). Engine lag tau1, vehicle lag tau2. u = throttle, in mph-equivalents."""
    n = int(T / dt); t = np.arange(n) * dt
    f, x, integ, prev = f0, x0, i0, x0
    X, U, I = np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(n):
        e = target(t[k]) - x
        dmeas = (x - prev) / dt if k else 0.0            # derivative on the measurement, not the error
        u_raw = Kp * e + Ki * integ - Kd * dmeas + ff(t[k])
        u = u_raw if ulim is None else min(max(u_raw, ulim[0]), ulim[1])
        saturated = (u != u_raw)
        if not (antiwindup and saturated and np.sign(e) == np.sign(u_raw - u)):
            integ += e * dt
        prev = x
        f += dt * (-f + u) / tau1
        x += dt * (-x + f - dist(t[k])) / tau2
        X[k], U[k], I[k] = x, u, Ki * integ
    return t, X, U, I

def step_metrics(t, y, y0, y1, t0=0.0):
    yn = (y - y0) / (y1 - y0); m = t >= t0
    tt, yn = t[m] - t0, yn[m]
    rise = tt[np.argmax(yn >= 0.9)] - tt[np.argmax(yn >= 0.1)]
    over = max(0.0, yn.max() - 1) * 100
    out = np.where(np.abs(yn - 1) > 0.02)[0]
    settle = tt[out[-1] + 1] if len(out) and out[-1] + 1 < len(tt) else float('nan')
    return rise, over, settle, (1 - yn[-1]) * (y1 - y0)

# 1 — bang-bang thermostat with and without hysteresis
def bangbang():
    rng = np.random.default_rng(3); dt, T = 0.1, 90.0; n = int(T / dt); t = np.arange(n) * dt
    def run(band):
        temp, on, sw = 19.0, True, 0; Ts, On = np.zeros(n), np.zeros(n)
        for k in range(n):
            meas = temp + rng.normal(0, 0.08)
            new_on = on
            if meas < 20 - band: new_on = True
            elif meas > 20 + band: new_on = False
            sw += (new_on != on); on = new_on
            temp += dt * (-(temp - 5) + 30 * on) / 40.0
            Ts[k], On[k] = temp, on
        return Ts, On, sw
    Th, Oh, sh = run(0.5); Tn, On_, sn = run(0.0)
    fig, axes = new(2, 1, figsize=(7.4, 4.8), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax = axes[0]; ax.axhspan(19.5, 20.5, color=GREEN, alpha=.12); ax.axhline(20, color=GREY, ls=':', lw=1.2)
    ax.plot(t, Th, color=BLUE, lw=2, label=f"with a ±0.5° band — {sh} switches in 90 min")
    ax.plot(t, Tn, color=RED, lw=1.2, alpha=.85, label=f"no band — {sn} switches (relay chatter)")
    ax.text(1, 20.62, "hysteresis band", color=GREEN, fontsize=8.5)
    style(ax, "A thermostat never settles — it bounces inside a band, on purpose", None, "room temperature (°C)")
    ax.legend(fontsize=8.5, loc='lower right'); ax.set_ylim(18.8, 21.2)
    axes[1].fill_between(t, 0, Oh, step='pre', color=ORANGE, alpha=.7); axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["off", "on"])
    style(axes[1], None, "minutes", "heater")
    save(fig, "bangbang_hysteresis.png"); print(f"    switches: band={sh}, no band={sn}")

# 2 — step response and its four numbers
def step_response():
    dt, T, w = 0.002, 40.0, 0.6; n = int(T / dt); t = np.arange(n) * dt
    fig, ax = new(figsize=(7.6, 4.5)); out = {}
    for zeta, col, name in [(2.0, GREY, "A — sluggish (overdamped)"), (0.2, RED, "B — ringing (underdamped)"), (0.7, BLUE, "C — well tuned")]:
        x = v = 0.0; y = np.zeros(n)
        for k in range(n):
            a = w * w * (1.0 - x) - 2 * zeta * w * v; v += a * dt; x += v * dt; y[k] = x
        mph = 50 + 10 * y; out[name[0]] = step_metrics(t, mph, 50, 60)
        ax.plot(t, mph, color=col, lw=2.2 if zeta != 2.0 else 1.8, label=name)
    ax.axhspan(59.8, 60.2, color=GREEN, alpha=.18); ax.axhline(60, color=GREY, ls=':', lw=1)
    rb, ob, sb, _ = out["B"]
    ax.annotate(f"overshoot {ob:.0f}%", xy=(5.4, 50 + 10 * (1 + ob / 100)), xytext=(9, 66.2), color=RED, fontsize=9, arrowprops=dict(arrowstyle='->', color=RED))
    ax.annotate("±2% band — 'settled'\nonce you stay inside", xy=(30, 60.2), xytext=(24.5, 63.3), color=GREEN, fontsize=8.5, arrowprops=dict(arrowstyle='->', color=GREEN))
    ax.annotate("", xy=(out["C"][0] + 0.9, 51), xytext=(0.9, 51), arrowprops=dict(arrowstyle='<->', color=BLUE))
    ax.text(5.0, 50.75, "rise time (10% → 90%)", color=BLUE, fontsize=8.5)
    style(ax, "One poke, four numbers: how to read a step response", "seconds after the setpoint jumps 50 → 60 mph", "speed (mph)")
    ax.legend(fontsize=8.5, loc='lower right'); ax.set_ylim(49, 67)
    save(fig, "step_response_metrics.png")
    for k2, (r, o, s, e) in out.items(): print(f"    car {k2}: rise {r:.1f}s  overshoot {o:.1f}%  settle {s:.1f}s  sse {e:.2f}")

# 3 — stability: same loop, three levels of impatience, 4 s of pipe delay
def stability():
    dt, T, L = 0.05, 90.0, 4.0; n = int(T / dt); t = np.arange(n) * dt; d = int(L / dt)
    fig, axes = new(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, gK, lbl, col in zip(axes, [0.10, 0.30, 0.45], ["patient", "impatient", "very impatient"], [GREEN, ORANGE, RED]):
        u = np.zeros(n); Tm = np.full(n, 20.0)
        for k in range(1, n):
            Tm[k] = np.clip(20.0 + (u[k - d] if k >= d else 0.0), 10, 60)      # what reaches your skin is 4 s old
            u[k] = u[k - 1] + gK * (38.0 - Tm[k]) * dt                          # you keep turning the tap while it feels wrong
        ax.plot(t, Tm, color=col, lw=2); ax.axhline(38, color=GREY, ls=':', lw=1.2)
        style(ax, f"{lbl}: gain × delay = {gK * L:.1f}", "seconds", "water temperature (°C)" if ax is axes[0] else None)
        ax.set_ylim(8, 62)
    bb = dict(facecolor=BG, edgecolor='none', alpha=.85, pad=1.5)
    axes[2].text(62, 56.5, "scalding", color=RED, fontsize=9, bbox=bb); axes[2].text(62, 11.5, "freezing", color=BLUE, fontsize=9, bbox=bb)
    suptitle(fig, "Same shower, same 4-second pipe — only the impatience changes", y=1.06)
    save(fig, "stability_gain_delay.png"); print(f"    critical gain×delay = pi/2 = {np.pi / 2:.2f}")

# 4 — P, PI, PID on the same car
def pid_terms():
    tgt = lambda t: 10.0 if t >= 2 else 0.0; wind = lambda t: 2.0 if t >= 2 else 0.0
    fig, ax = new(figsize=(7.6, 4.4)); res = {}
    for name, g, col in [("P only", (5, 0, 0), GREY), ("P + I", (5, 0.5, 0), ORANGE), ("P + I + D", (5, 0.5, 6), BLUE)]:
        t, X, U, I = sim_car(T=60, Kp=g[0], Ki=g[1], Kd=g[2], target=tgt, dist=wind, tau1=4.0)
        ax.plot(t, 50 + X, color=col, lw=2.1, label=name); res[name] = (X[-1], X.max(), step_metrics(t, 50 + X, 50, 60, 2.0))
    ax.axhline(60, color=GREY, ls=':', lw=1.2)
    ax.annotate("steady-state error:\nstuck at 58", xy=(52, 58), xytext=(40, 54.3), color=FG, fontsize=9, arrowprops=dict(arrowstyle='->', color=GREY))
    style(ax, "P gets close, I closes the gap, D calms it down", "seconds (setpoint 50 → 60 mph into a headwind at t = 2)", "speed (mph)")
    ax.legend(fontsize=9, loc='lower right'); ax.set_ylim(49, 66)
    save(fig, "pid_terms_p_pi_pid.png")
    for k2, (fin, mx, m) in res.items(): print(f"    {k2}: final {50 + fin:.2f}  peak {50 + mx:.2f}  overshoot {m[1]:.1f}%  settle {m[2]:.1f}s")

# 5 — integral windup
def windup():
    hill = lambda t: 9.0 if 40 <= t < 80 else 0.0; tgt = lambda t: 10.0
    fig, axes = new(2, 1, figsize=(7.6, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[2, 1])); pk = {}
    for name, aw, col in [("plain PID", False, RED), ("PID with anti-windup (integrator frozen while saturated)", True, GREEN)]:
        t, X, U, I = sim_car(T=150, Kp=5, Ki=0.5, Kd=6, target=tgt, dist=hill, ulim=(-6, 14), antiwindup=aw, x0=10, f0=10, i0=10 / 0.5, tau1=4.0)
        axes[0].plot(t, 50 + X, color=col, lw=2, label=name); axes[1].plot(t, I, color=col, lw=2); pk[name] = (50 + X[t > 80].max(), t[(t > 80) & (np.abs(X - 10) > 0.2)].max() - 80)
    for ax in axes: ax.axvspan(40, 80, color=GREY, alpha=.15)
    axes[0].axhline(60, color=GREY, ls=':', lw=1.2); axes[0].text(41, 63.6, "steep hill — pedal is floored,\nspeed still sags", fontsize=8.5, color=FG)
    style(axes[0], "Integral windup: the hill ends, but the integrator hasn't noticed", None, "speed (mph)"); axes[0].legend(fontsize=8.5, loc='lower left'); axes[0].set_ylim(53, 66)
    style(axes[1], None, "seconds", "integral term"); axes[1].axhline(14, color=GREY, ls='--', lw=1); axes[1].text(100, 20, "pedal limit", fontsize=8, color=GREY)
    save(fig, "integral_windup.png")
    for k2, (p, d) in pk.items(): print(f"    {k2}: peak after hill {p:.1f} mph, back within 0.2 mph after {d:.0f}s")

# 6 — feedforward + feedback
def feedforward():
    hill = lambda t: 20.0 if t >= 40 else 0.0; tgt = lambda t: 10.0; fig, ax = new(figsize=(7.4, 4.2)); dips = {}; lead = 3.0
    for name, ffk, col in [("feedback only (PID)", 0.0, RED), ("feedback + feedforward (sees the hill 3 s early, model 80% right)", 0.8, BLUE), ("feedforward alone, no feedback", None, GREY)]:
        if ffk is None: t, X, U, I = sim_car(T=110, Kp=0, Ki=0, Kd=0, target=tgt, dist=hill, ff=lambda t: 10 + 0.8 * hill(t + lead), x0=10, f0=10, tau1=4.0)
        else: t, X, U, I = sim_car(T=110, Kp=5, Ki=0.5, Kd=6, target=tgt, dist=hill, ff=lambda t, k=ffk: k * hill(t + lead), x0=10, f0=10, i0=10 / 0.5, tau1=4.0)
        ax.plot(t, 50 + X, color=col, lw=2, ls='--' if ffk is None else '-', label=name); dips[name] = (50 + X[t > 40].min(), 50 + X[-1])
    ax.axvline(40, color=GREY, ls=':', lw=1.2); ax.text(30.6, 61.35, "hill starts at t = 40 →", fontsize=8.5, color=GREY)
    style(ax, "Feedback waits for the error. Feedforward doesn't have to.", "seconds", "speed (mph)"); ax.legend(fontsize=8.5, loc='center right'); ax.set_xlim(30, 110); ax.set_ylim(55.2, 61.8)
    save(fig, "feedforward_vs_feedback.png")
    for k2, (mn, fin) in dips.items(): print(f"    {k2}: lowest {mn:.2f}, final {fin:.2f}")

# 7 — pole locations and what the system does
def poles():
    t = np.linspace(0, 12, 600); fig, axes = new(2, 4, figsize=(12, 4.8), gridspec_kw=dict(height_ratios=[1, 1.15]))
    cases = [(-0.6, 0, "real, negative\n→ dies away", GREEN), (-0.3, 2.2, "complex, left half\n→ rings, then settles", BLUE),
             (0.0, 2.2, "on the axis\n→ rings forever", ORANGE), (0.25, 2.2, "right half\n→ grows without limit", RED)]
    for j, (s, w, lbl, col) in enumerate(cases):
        a = axes[0, j]; a.axvspan(0, 1.2, color=RED, alpha=.08); a.axhline(0, color=GREY, lw=.8); a.axvline(0, color=GREY, lw=.8)
        a.plot([s, s] if w else [s], [w, -w] if w else [0], 'x', color=col, ms=11, mew=3)
        a.set_xlim(-1.2, 1.2); a.set_ylim(-3, 3); style(a, lbl, "Re(s)" if j == 0 else None, "Im(s)" if j == 0 else None, grid=False); a.set_xticks([]); a.set_yticks([])
        b = axes[1, j]; y = np.exp(s * t) * (np.cos(w * t) if w else 1.0); b.plot(t, y, color=col, lw=2); b.axhline(0, color=GREY, lw=.8)
        style(b, None, "time", "response to a kick" if j == 0 else None); b.set_ylim(-3.2, 3.2 if j == 3 else 1.3) if j == 3 else b.set_ylim(-1.2, 1.2)
    axes[0, 3].text(0.5, -0.55, "unstable\nside", color=RED, fontsize=8)
    suptitle(fig, "Where the poles sit tells you how the system behaves", y=1.06)
    save(fig, "poles_and_responses.png")

# 15 — LQR: one dial between accuracy and effort
def lqr():
    dt = 0.1; A = np.array([[1, dt], [0, 1.0]]); B = np.array([[0.5 * dt * dt], [dt]]); Q = np.diag([1.0, 0.0])
    fig, axes = new(1, 2, figsize=(11, 3.9))
    for r, col, lbl in [(0.01, RED, "effort is cheap  (r = 0.01)"), (1.0, BLUE, "balanced  (r = 1)"), (100.0, GREEN, "effort is expensive  (r = 100)")]:
        P = Q.copy()
        for _ in range(20000):
            P = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(r + B.T @ P @ B) @ B.T @ P @ A
        K = np.linalg.inv(r + B.T @ P @ B) @ B.T @ P @ A
        x = np.array([[1.0], [0.0]]); xs, us = [], []
        for _ in range(150):
            u = -(K @ x); xs.append(x[0, 0]); us.append(u[0, 0]); x = A @ x + B @ u
        tt = np.arange(150) * dt; axes[0].plot(tt, xs, color=col, lw=2, label=lbl); axes[1].plot(tt, us, color=col, lw=2)
        print(f"    r={r}: K = [{K[0, 0]:.2f}, {K[0, 1]:.2f}]   sum u^2 = {np.sum(np.square(us)) * dt:.2f}   time to |x|<0.05: {tt[np.argmax(np.abs(np.array(xs)) < 0.05)]:.1f}s")
    axes[0].axhline(0, color=GREY, ls=':', lw=1); style(axes[0], "distance from the hover point (m)", "seconds", None); axes[0].legend(fontsize=8.5)
    axes[1].axhline(0, color=GREY, ls=':', lw=1); style(axes[1], "thrust command", "seconds", None)
    suptitle(fig, "LQR: one dial trades accuracy against effort — and the gains fall out of the maths", y=1.04)
    save(fig, "lqr_q_vs_r.png")

# 16 — MPC: plan N, apply one
def mpc():
    dt, N = 0.2, 20; A = np.array([[1, dt], [0, 1.0]]); B = np.array([0.5 * dt * dt, dt])
    F = np.zeros((2 * N, 2)); G = np.zeros((2 * N, N)); Ak = np.eye(2)
    for k in range(N):
        Ak = A @ Ak; F[2 * k:2 * k + 2] = Ak
        for j in range(k + 1): G[2 * k:2 * k + 2, j] = np.linalg.matrix_power(A, k - j) @ B
    Qb = np.diag(np.tile([1.0, 0.3], N)); H = G.T @ Qb @ G + 0.05 * np.eye(N); Lc = np.linalg.eigvalsh(H).max()
    def plan(x0, U0):
        g = G.T @ Qb @ F @ x0; U = U0.copy()
        for _ in range(3000): U = np.clip(U - (H @ U + g) / Lc, -1, 1)
        return U
    x = np.array([10.0, 0.0]); U = np.zeros(N); xs, us, snap = [x[0]], [], None
    for k in range(60):
        U = plan(x, np.append(U[1:], 0.0))
        if k == 10: snap = (k, x.copy(), U.copy(), (F @ x + G @ U)[0::2])
        us.append(U[0]); x = A @ x + B * U[0]; xs.append(x[0])
    k0, x0, U0, pred = snap; tt = np.arange(61) * dt
    fig, axes = new(2, 1, figsize=(7.6, 5.3), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1]))
    axes[0].plot(tt[:k0 + 1], xs[:k0 + 1], color=BLUE, lw=2.4, label="what actually happened"); axes[0].plot(tt[k0:], xs[k0:], color=BLUE, lw=1, alpha=.35)
    axes[0].plot((k0 + 1 + np.arange(N)) * dt, pred, color=ORANGE, lw=2, ls='--', label="the plan made right now (20 steps ahead)")
    axes[0].axvspan(k0 * dt, (k0 + N) * dt, color=ORANGE, alpha=.08); axes[0].axvline(k0 * dt, color=GREY, ls=':'); axes[0].axhline(0, color=GREY, ls=':', lw=1)
    axes[0].text(k0 * dt + .1, 9.0, "now", fontsize=9, color=GREY); axes[0].text(k0 * dt + 1.0, 6.7, "prediction horizon", fontsize=9, color=ORANGE)
    style(axes[0], "MPC plans 20 steps, uses one, and throws the rest away", None, "distance to target (m)"); axes[0].legend(fontsize=8.5, loc='upper right')
    axes[1].step(tt[:k0 + 1], us[:k0 + 1], where='post', color=BLUE, lw=2.2); axes[1].step((k0 + np.arange(N)) * dt, U0, where='post', color=ORANGE, lw=1.8, ls='--')
    axes[1].plot([k0 * dt + dt / 2], [U0[0]], 'o', color=RED, ms=8); axes[1].annotate("only this one\nis applied", xy=(k0 * dt + dt / 2, U0[0]), xytext=(k0 * dt + 1.3, -0.35), fontsize=8.5, color=RED, arrowprops=dict(arrowstyle='->', color=RED))
    for yl in (-1, 1): axes[1].axhline(yl, color=RED, ls='--', lw=1, alpha=.6)
    axes[1].text(9.2, 0.78, "actuator limit", fontsize=8, color=RED); axes[1].step(tt[k0:60], us[k0:], where='post', color=BLUE, lw=1, alpha=.35)
    style(axes[1], None, "seconds", "acceleration command"); axes[1].set_ylim(-1.25, 1.25)
    save(fig, "mpc_receding_horizon.png"); print(f"    steps at the limit: {int(np.sum(np.abs(np.array(us)) > 0.999))} of 60; final distance {xs[-1]:.3f}")

# 17 — system identification from one step test
def sysid():
    rng = np.random.default_rng(11); dt, K, tau = 1.0, 1.0, 5.0; n = 31; t = np.arange(n) * dt; u = np.full(n, 10.0)
    a = np.exp(-dt / tau); x = np.zeros(n)
    for k in range(n - 1): x[k + 1] = a * x[k] + K * (1 - a) * u[k]
    y = x + rng.normal(0, 0.35, n); Phi = np.column_stack([y[:-1], u[:-1]]); ah, bh = np.linalg.lstsq(Phi, y[1:], rcond=None)[0]
    tau_arx, K_arx = -dt / np.log(ah), bh / (1 - ah); tf = np.linspace(0, 30, 300)
    best = (1e18, 0, 0)
    for tc in np.linspace(1.0, 12.0, 1101):                       # for each candidate tau the best gain is a 1-D least squares
        basis = 10 * (1 - np.exp(-t / tc)); Kc = (basis @ y) / (basis @ basis); sse = np.sum((y - Kc * basis) ** 2)
        if sse < best[0]: best = (sse, Kc, tc)
    _, K_h, tau_h = best
    fig, ax = new(figsize=(7.4, 4.2)); ax.plot(t, 50 + y, 'o', color=GREY, ms=5, label="logged speed (1 Hz, noisy)")
    ax.plot(tf, 50 + K_h * 10 * (1 - np.exp(-tf / tau_h)), color=BLUE, lw=2.3, label=f"fitted model: gain {K_h:.2f} mph per %, time constant {tau_h:.1f} s")
    ax.plot(tf, 50 + 10 * (1 - np.exp(-tf / tau)), color=GREEN, lw=1.3, ls='--', label="the truth (gain 1.00, τ = 5.0 s)")
    ax.axhline(56.32, color=ORANGE, ls=':', lw=1.2); ax.axvline(tau_h, color=ORANGE, ls=':', lw=1.2); ax.text(tau_h + .4, 53.4, "63% of the way\n= one time constant", fontsize=8.5, color=ORANGE)
    style(ax, "Poke it once, fit two numbers: gain and time constant", "seconds after the pedal steps from 20% to 30%", "speed (mph)"); ax.legend(fontsize=8.5, loc='lower right')
    save(fig, "sysid_step_fit.png"); print(f"    curve fit: K={K_h:.3f}, tau={tau_h:.2f}s   |   naive one-step regression on noisy logs: K={K_arx:.3f}, tau={tau_arx:.2f}s")

if __name__ == "__main__":
    for f in (bangbang, step_response, stability, pid_terms, windup, feedforward, poles, lqr, mpc, sysid): f()
