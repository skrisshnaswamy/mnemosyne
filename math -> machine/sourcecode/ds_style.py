"""Shared style for the control / decision-sciences diagrams. PNGs go into ../References/.
Same palette and look as llm_diagrams.py so the whole vault reads as one set."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "../References/"
BG, FG, GRID = "#ffffff", "#2b2b2b", "#d8d8d8"
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#3b7dd8", "#e08a1e", "#2ea36c", "#d64545", "#9aa0a6", "#8e5bd0"

def style(ax, title=None, xlabel=None, ylabel=None, grid=True):
    ax.set_facecolor(BG)
    if title:  ax.set_title(title, fontsize=11.5, color=FG, pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10, color=FG)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10, color=FG)
    if grid:   ax.grid(True, linestyle='--', alpha=.5, color=GRID)
    ax.tick_params(colors=FG, labelsize=9)
    for s in ax.spines.values(): s.set_color(GRID)

def new(nrows=1, ncols=1, figsize=(7, 4.2), **kw):
    return plt.subplots(nrows, ncols, figsize=figsize, facecolor=BG, **kw)

def suptitle(fig, text, y=1.02, size=12.5):
    fig.suptitle(text, fontsize=size, color=FG, y=y)

def save(fig, name):
    fig.savefig(OUT + name, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig); print("  ✓", name)

def smooth(y, k):
    """Trailing moving average, same length as y."""
    y = np.asarray(y, float); c = np.cumsum(np.insert(y, 0, 0.0))
    out = np.empty_like(y)
    for i in range(len(y)):
        j = max(0, i - k + 1); out[i] = (c[i + 1] - c[j]) / (i + 1 - j)
    return out
