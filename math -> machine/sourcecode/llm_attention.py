"""Diagrams for the Attention hub and its concept notes. Outputs PNGs into ../References/.

Reuses the house style from llm_diagrams.py. Every function prints the numbers the
notes quote, so prose and picture can never disagree.

Run: cd "math -> machine/sourcecode" && python3 llm_attention.py
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from llm_diagrams import style, save, BG, FG, GRID, BLUE, ORANGE, GREEN, RED, GREY

rng = np.random.default_rng(0)


# 1 — why you divide by sqrt(d_k): a real simulation of softmax saturation
def softmax_scaling():
    dims = np.array([4, 8, 16, 32, 64, 128, 256, 512])
    trials, n_keys = 4000, 40
    unscaled_max, scaled_max, unscaled_H, scaled_H = [], [], [], []

    for d in dims:
        q = rng.standard_normal((trials, d))
        k = rng.standard_normal((trials, n_keys, d))
        logits = np.einsum('td,tkd->tk', q, k)          # one query against 40 keys
        for arr_max, arr_H, z in ((unscaled_max, unscaled_H, logits),
                                  (scaled_max, scaled_H, logits / np.sqrt(d))):
            z = z - z.max(axis=1, keepdims=True)
            p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
            arr_max.append(p.max(axis=1).mean())
            arr_H.append((-(p * np.log(p + 1e-12)).sum(axis=1) / np.log(n_keys)).mean())

    i128 = list(dims).index(128)
    print("  softmax_scaling — 40 keys, random Q/K")
    print(f"    d_k=128 unscaled: top weight {unscaled_max[i128]*100:.1f}%, "
          f"normalised entropy {unscaled_H[i128]:.2f}")
    print(f"    d_k=128 scaled  : top weight {scaled_max[i128]*100:.1f}%, "
          f"normalised entropy {scaled_H[i128]:.2f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(dims, np.array(unscaled_max)*100, 'o-', color=RED, lw=2.2,
            label=r"raw $QK^T$ — one token wins")
    ax.plot(dims, np.array(scaled_max)*100, 'o-', color=GREEN, lw=2.2,
            label=r"divided by $\sqrt{d_k}$ — stays soft")
    ax.axhline(100/40, color=GREY, ls=':', lw=1.4)
    ax.text(5, 100/40 + 1.5, "2.5% — perfectly flat over 40 keys", color=GREY, fontsize=9)
    ax.axvline(128, color=GREY, ls='--', lw=1.2)
    ax.text(140, 60, "head dim 128\n(a typical head)", color=GREY, fontsize=9)
    style(ax, r"Without $\sqrt{d_k}$, attention collapses onto one token",
          "Head dimension $d_k$", "Weight on the top-scoring key (%)")
    ax.set_xscale('log', base=2); ax.set_ylim(0, 100); ax.legend(fontsize=9, loc='upper left')
    save(fig, "attn_softmax_scaling.png")


# 2 — KV cache bytes per token: share it (GQA/MQA) vs compress it (MLA)
def kv_bytes_per_token():
    L, H, d_head, b = 80, 64, 128, 2          # 80 layers, 64 heads, dim 128, fp16
    d_c, d_rope = 512, 64                     # DeepSeek-V2-style latent + decoupled RoPE key

    mha = 2 * L * H * d_head * b              # K and V, every head
    gqa = 2 * L * 8 * d_head * b              # 8 KV heads
    mqa = 2 * L * 1 * d_head * b              # 1 KV head
    mla = L * (d_c + d_rope) * b              # one latent vector per layer, no separate V

    budget = 20 * 1024**3                     # 20 GB left for cache on an 80 GB card
    names = ["MHA\n64 KV heads", "GQA\n8 KV heads", "MQA\n1 KV head", "MLA\nlatent 512"]
    vals = np.array([mha, gqa, mqa, mla]) / 1024**2   # MB per token
    cols = [RED, ORANGE, GREY, GREEN]

    print("  kv_bytes_per_token — 80 layers, 64 heads, head dim 128, fp16")
    for n, v in zip([x.split('\n')[0] for x in names], vals):
        print(f"    {n:4s}: {v:.3f} MB/token → {budget/(v*1024**2):,.0f} tokens in 20 GB")
    print(f"    MLA is {vals[0]/vals[3]:.1f}x smaller than MHA, {vals[1]/vals[3]:.1f}x smaller than GQA")

    fig, ax = plt.subplots(figsize=(7.4, 4.2), facecolor=BG); ax.set_facecolor(BG)
    bars = ax.bar(names, vals, color=cols, width=.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + .05, f"{v:.2f} MB\n{budget/(v*1024**2)/1000:,.0f}k tokens",
                ha='center', fontsize=9, color=FG)
    ax.set_ylim(0, vals.max()*1.32)
    style(ax, "Sharing K/V helps. Compressing it helps more.",
          "same model shape, four cache designs — second line is how many tokens fit in 20 GB",
          "KV cache per token (MB)")
    save(fig, "attn_kv_bytes_per_token.png")


# 3 — sliding window: how little you compute, and how far a token can still see
def sparse_receptive_field():
    n, w = 32768, 512
    full = n * (n + 1) / 2                       # causal pairs
    win = n * w - w * (w - 1) / 2                # causal sliding window of w
    layers = np.arange(1, 65)
    reach = layers * w

    print("  sparse_receptive_field — 32,768 tokens, window 512")
    print(f"    full causal pairs: {full:,.0f}")
    print(f"    sliding-window pairs: {win:,.0f}  ({win/full*100:.1f}% of full)")
    print(f"    reach after 32 layers: {32*w:,} tokens; layers to span 32,768: {int(np.ceil(n/w))}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.2), facecolor=BG)
    for ax in (a1, a2): ax.set_facecolor(BG)

    a1.bar(["full causal", "window 512"], [full/1e9, win/1e9], color=[RED, GREEN], width=.55)
    a1.text(0, full/1e9*1.02, f"{full/1e9:.2f}B pairs", ha='center', fontsize=9.5, color=FG)
    a1.text(1, win/1e9 + full/1e9*.03, f"{win/1e9:.3f}B — {win/full*100:.1f}%",
            ha='center', fontsize=9.5, color=FG)
    style(a1, "You skip 96.9% of the pairs", "", "Attention pairs (billions)")
    a1.set_ylim(0, full/1e9*1.18)

    a2.plot(layers, reach, color=BLUE, lw=2.4, label="reach = layers × window")
    a2.axhline(n, color=RED, ls='--', lw=1.5)
    a2.text(2, n*1.06, "32,768 — the whole document", color=RED, fontsize=9)
    a2.axvline(64, color=GREY, ls=':', lw=1.3)
    a2.text(46, n*.35, "64 layers\nto span it", color=GREY, fontsize=9)
    style(a2, "…and depth buys the reach back", "Layer", "Tokens a position can reach")
    a2.legend(fontsize=9, loc='upper left'); a2.set_ylim(0, n*1.25)
    fig.tight_layout()
    save(fig, "sparse_attention_receptive_field.png")


# 4 — softmax O(n^2 d) vs linear O(n d^2): where the crossover actually is
def linear_attention_crossover():
    d = 4096
    n = np.logspace(np.log10(64), np.log10(1_000_000), 400)
    soft = 4 * n**2 * d        # QK^T and AV
    lin = 4 * n * d**2         # (K^T V) once, then Q times it

    for probe in (1024, 4096, 32768, 262144):
        s, l = 4*probe**2*d, 4*probe*d**2
        print(f"    n={probe:>7,}: softmax {s/1e12:9.2f} TFLOP · linear {l/1e12:6.2f} TFLOP "
              f"· softmax is {s/l:5.2f}x linear")

    fig, ax = plt.subplots(figsize=(7.4, 4.3), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(n, soft/1e12, color=RED, lw=2.3, label=r"softmax attention — $O(n^2 d)$")
    ax.plot(n, lin/1e12, color=GREEN, lw=2.3, label=r"linear attention — $O(n d^2)$")
    ax.axvline(d, color=GREY, ls='--', lw=1.4)
    ax.annotate(f"crossover at n = d = {d:,}\nbelow this, softmax is the cheap one",
                xy=(d, 4*d**2*d/1e12), xytext=(d*1.6, 4*d**2*d/1e12*0.06),
                fontsize=9, color=FG,
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1.2))
    ax.axvspan(64, d, color=BLUE, alpha=.06)
    ax.text(150, 1e-3, "where almost every\nrequest actually lives", fontsize=9, color=BLUE)
    style(ax, "Linear attention only pays off past the model dimension",
          "Sequence length (tokens)", "FLOPs per layer (trillions)")
    ax.set_xscale('log'); ax.set_yscale('log'); ax.legend(fontsize=9, loc='upper left')
    save(fig, "linear_attention_crossover.png")


if __name__ == "__main__":
    softmax_scaling()
    kv_bytes_per_token()
    sparse_receptive_field()
    linear_attention_crossover()
