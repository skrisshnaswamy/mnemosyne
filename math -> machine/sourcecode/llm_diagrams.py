"""Diagrams for the LLM Engineering notes. Outputs PNGs into ../References/."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "../References/"
BG, FG, GRID = "#ffffff", "#2b2b2b", "#d8d8d8"
BLUE, ORANGE, GREEN, RED, GREY = "#3b7dd8", "#e08a1e", "#2ea36c", "#d64545", "#9aa0a6"

def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, color=FG, pad=12)
    ax.set_xlabel(xlabel, fontsize=11, color=FG)
    ax.set_ylabel(ylabel, fontsize=11, color=FG)
    ax.grid(True, linestyle='--', alpha=.5, color=GRID)
    ax.tick_params(colors=FG, labelsize=9)
    for s in ax.spines.values(): s.set_color(GRID)

def save(fig, name):
    fig.savefig(OUT+name, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig); print("  ✓", name)

# 1 — attention cost: quadratic vs linear memory
def attention_cost():
    n = np.arange(0, 33000, 250)
    fig, ax = plt.subplots(figsize=(7,4.2), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(n, (n**2)*2*4/1e9, color=RED, lw=2.2, label="Standard attention — $O(n^2)$ matrix")
    ax.plot(n, n*2*4*128/1e9, color=GREEN, lw=2.2, label="FlashAttention — $O(n)$, never materialised")
    ax.axvline(8192, color=GREY, ls=':', lw=1.4)
    ax.text(8600, 1.9, "8k context", color=GREY, fontsize=9)
    style(ax, "Why long context was unaffordable", "Sequence length (tokens)", "Memory for attention (GB, per head)")
    ax.legend(fontsize=9, framealpha=.95); ax.set_ylim(0,4.5)
    save(fig, "attn_cost_quadratic.png")

# 2 — KV cache growth
def kv_growth():
    n = np.arange(0, 132000, 1000)
    per_tok = 2*32*8*128*2/1e9   # 2(K,V) * layers * kv_heads * head_dim * bytes
    fig, ax = plt.subplots(figsize=(7,4.2), facecolor=BG); ax.set_facecolor(BG)
    for b,c in [(1,BLUE),(8,ORANGE),(32,RED)]:
        ax.plot(n, n*per_tok*b, color=c, lw=2.2, label=f"batch = {b}")
    ax.axhline(80, color=GREY, ls='--', lw=1.5)
    ax.text(2000, 82, "80 GB — one H100", color=GREY, fontsize=9)
    style(ax, "The KV cache is what limits your concurrency", "Context length (tokens)", "KV cache size (GB)")
    ax.legend(fontsize=9); ax.set_ylim(0,120)
    save(fig, "kv_cache_growth.png")

# 3 — temperature reshaping a distribution
def temperature():
    logits = np.array([3.2, 2.9, 2.1, 1.4, 0.9, 0.3, -0.4, -1.1])
    toks = ["the","a","this","my","our","its","his","an"]
    fig, axes = plt.subplots(1,4, figsize=(12,3.4), facecolor=BG, sharey=True)
    for ax,T,lbl in zip(axes,[0.2,0.7,1.0,1.8],
                        ["T = 0.2\nnear-greedy","T = 0.7\ntypical","T = 1.0\nraw model","T = 1.8\nchaotic"]):
        p = np.exp(logits/T); p/=p.sum()
        ax.bar(toks, p, color=[BLUE if i==0 else GREY for i in range(len(toks))])
        ax.set_facecolor(BG); ax.set_title(lbl, fontsize=10, color=FG)
        ax.tick_params(colors=FG, labelsize=8); ax.set_ylim(0,1)
        for s in ax.spines.values(): s.set_color(GRID)
        ax.grid(axis='y', linestyle='--', alpha=.4, color=GRID)
    axes[0].set_ylabel("probability", fontsize=10, color=FG)
    fig.suptitle("Temperature reshapes the distribution — it does not change what the model believes",
                 fontsize=12, color=FG, y=1.04)
    save(fig, "sampling_temperature.png")

# 4 — lost in the middle
def lost_middle():
    pos = np.linspace(0,1,21)
    acc = 0.78 - 0.34*np.sin(np.pi*pos)**1.6 + 0.06*pos
    fig, ax = plt.subplots(figsize=(7,4.2), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(pos*100, acc*100, color=BLUE, lw=2.4, marker='o', ms=4)
    ax.fill_between(pos*100, acc*100, 40, color=BLUE, alpha=.08)
    ax.annotate("buried here → worst recall", xy=(50, acc[10]*100), xytext=(52, 72),
                arrowprops=dict(arrowstyle='->', color=RED), color=RED, fontsize=9.5)
    style(ax, "Lost in the middle", "Position of the relevant fact in the context (%)", "Retrieval accuracy (%)")
    ax.set_ylim(40,85)
    save(fig, "lost_in_the_middle.png")

# 5 — quantization levels
def quant():
    x = np.linspace(-1,1,600)
    fig, axes = plt.subplots(1,3, figsize=(11,3.2), facecolor=BG, sharey=True)
    for ax,(bits,name) in zip(axes,[(16,"BF16 — ~65k levels"),(8,"INT8 — 256 levels"),(4,"INT4 — 16 levels")]):
        lv = 2**bits
        q = np.round(x*(lv/2-1))/(lv/2-1)
        ax.plot(x, x, color=GREY, lw=1.2, ls='--', label="true value")
        ax.step(x, q, color=BLUE if bits>4 else RED, lw=1.8, where='mid', label="stored value")
        ax.set_facecolor(BG); ax.set_title(name, fontsize=10.5, color=FG)
        ax.tick_params(colors=FG, labelsize=8)
        for s in ax.spines.values(): s.set_color(GRID)
        ax.grid(True, linestyle='--', alpha=.4, color=GRID)
    axes[0].set_ylabel("stored", fontsize=10, color=FG); axes[2].legend(fontsize=8)
    fig.suptitle("Quantization — fewer bits means coarser steps, not smaller numbers",
                 fontsize=12, color=FG, y=1.05)
    save(fig, "quantization_levels.png")

# 6 — prefill vs decode
def prefill_decode():
    fig, axes = plt.subplots(1,2, figsize=(11,3.8), facecolor=BG)
    ax=axes[0]; ax.set_facecolor(BG)
    ax.bar(["Prefill\n(1 pass, 2000 tokens)","Decode\n(per token)"], [98, 3],
           color=[GREEN, RED], width=.55)
    style(ax, "GPU utilisation", "", "% of peak FLOPs used"); ax.set_ylim(0,105)
    ax.text(1, 6, "GPU mostly idle,\nwaiting on memory", ha='center', fontsize=9, color=RED)
    ax=axes[1]; ax.set_facecolor(BG)
    t=np.arange(1,9)
    ax.bar(t-0.2, [1.0]+[0]*7, width=.4, color=GREEN, label="prefill — parallel")
    ax.bar(t+0.2, [0]+[0.35]*7, width=.4, color=RED, label="decode — sequential")
    style(ax, "Where the wall-clock goes", "step", "time (arbitrary)")
    ax.legend(fontsize=9)
    fig.suptitle("Two phases, two completely different bottlenecks", fontsize=12.5, color=FG, y=1.04)
    save(fig, "prefill_vs_decode.png")

# 7 — speculative decoding
def spec_decode():
    a=np.linspace(0,1,200); k=4
    speed=(1-a**(k+1))/((1-a)*(1+k*0.15))
    fig, ax = plt.subplots(figsize=(7,4.2), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(a*100, speed, color=BLUE, lw=2.4)
    ax.axhline(1, color=GREY, ls='--', lw=1.4); ax.text(2,1.05,"no speed-up", color=GREY, fontsize=9)
    ax.axvspan(70,90, color=GREEN, alpha=.10)
    ax.text(80, 0.4, "typical\nacceptance", ha='center', fontsize=9, color=GREEN)
    style(ax, "Speculative decoding — it lives or dies on the acceptance rate",
          "Draft-token acceptance rate (%)", "Speed-up vs plain decoding (×)")
    ax.set_ylim(0,3.2)
    save(fig, "speculative_decoding.png")

# 8 — MoE
def moe():
    fig, ax = plt.subplots(figsize=(7.5,4), facecolor=BG); ax.set_facecolor(BG)
    names=["Dense 8B","Dense 70B","MoE 8×7B\n(47B total)"]
    total=[8,70,47]; active=[8,70,13]
    x=np.arange(3)
    ax.bar(x-.2, total, .4, color=GREY, label="total parameters (memory you must hold)")
    ax.bar(x+.2, active, .4, color=BLUE, label="active per token (compute you pay)")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10, color=FG)
    style(ax, "Mixture of Experts — pay for memory once, pay for compute per token", "", "billions of parameters")
    ax.legend(fontsize=9)
    ax.annotate("knows a lot,\nthinks cheaply", xy=(2.2,13), xytext=(2.35,40),
                arrowprops=dict(arrowstyle='->', color=GREEN), color=GREEN, fontsize=9.5, ha='center')
    save(fig, "moe_active_params.png")

# 9 — test-time compute
def test_time():
    n=np.arange(1,65)
    fig, ax = plt.subplots(figsize=(7,4.2), facecolor=BG); ax.set_facecolor(BG)
    for p,c,l in [(0.30,BLUE,"per-attempt success 30%"),(0.55,GREEN,"55%")]:
        ax.plot(n, (1-(1-p)**n)*100, color=c, lw=2.3, label="pass@k — "+l)
    ax.plot(n, (30+14*np.log1p(n)), color=ORANGE, lw=2.3, ls='--',
            label="realistic: needs a verifier to pick")
    style(ax, "Spending compute at inference instead of training",
          "Samples drawn (k)", "Accuracy (%)")
    ax.legend(fontsize=9, loc='lower right'); ax.set_ylim(20,102); ax.set_xscale('log', base=2)
    save(fig, "test_time_compute.png")

# 10 — context budget
def ctx_budget():
    fig, ax = plt.subplots(figsize=(8,2.6), facecolor=BG); ax.set_facecolor(BG)
    parts=[("System prompt",1.2,BLUE),("Tool definitions",3.5,ORANGE),
           ("Retrieved docs",14,GREEN),("Conversation history",9,"#8e6fd8"),
           ("Room left",4.3,GREY)]
    left=0
    for name,w,c in parts:
        ax.barh([0], [w], left=left, color=c, height=.55)
        if w>2.5: ax.text(left+w/2, 0, f"{name}\n{w}k", ha='center', va='center',
                        fontsize=8.5, color='white' if c!=GREY else FG)
        left+=w
    ax.set_xlim(0,32); ax.set_yticks([]); ax.set_xlabel("tokens (thousands) — 32k window", color=FG, fontsize=10)
    ax.set_title("A context window is a budget, and most of it is already spent", fontsize=12.5, color=FG, pad=12)
    ax.tick_params(colors=FG, labelsize=9)
    for s in ax.spines.values(): s.set_visible(False)
    save(fig, "context_budget.png")
