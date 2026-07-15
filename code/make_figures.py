#!/usr/bin/env python3
"""Generate publication-quality PDF figures for the paper."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/mnt/e/Project/MDPI/symmetry-paper-2/paper/figures"

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "figure.dpi": 300,
})

# Data from experiments (CIFAR-100, ResNet-50 from scratch)
fractions = [5, 10, 25, 50, 100]
none     = [14.3, 19.5, 34.0, 46.5, 60.0]
hflip    = [16.7, 24.6, 41.8, 55.9, 68.0]
standard = [22.4, 33.3, 51.9, 63.6, 73.3]


def fig_efficiency():
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(fractions, none,     "o--", color="#888888", label="None (baseline)")
    ax.plot(fractions, hflip,    "s-",  color="#1f77b4", label="hflip (mirror symmetry)")
    ax.plot(fractions, standard, "^-",  color="#d62728", label="standard (mirror + photometric)")
    ax.set_xlabel("Training data fraction (%)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_xticks(fractions)
    ax.set_ylim(0, 80)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    # annotate the growing gap
    ax.annotate("", xy=(25, 51.9), xytext=(25, 34.0),
                arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.2))
    ax.text(26, 42.5, "+17.9", color="#d62728", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/efficiency.pdf", bbox_inches="tight")
    print(f"saved {OUT}/efficiency.pdf")


def fig_ablation():
    """Bar chart: which symmetry direction helps (CIFAR-100, single transform)."""
    labels = ["none", "vflip", "rot90", "rot30", "rot15", "hflip"]
    acc    = [60.0,  61.1,    64.8,    67.7,    68.3,    67.2]
    kind   = ["base", "violating", "violating", "preserving", "preserving", "preserving"]
    colors = {"base": "#888888", "violating": "#d62728", "preserving": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bars = ax.bar(labels, acc, color=[colors[k] for k in kind], edgecolor="black", linewidth=0.5)
    ax.axhline(60.0, color="#888888", linestyle=":", lw=1)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_ylim(55, 72)
    for b, a in zip(bars, acc):
        ax.text(b.get_x() + b.get_width()/2, a + 0.3, f"{a:.1f}", ha="center", fontsize=9)
    # legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors["preserving"], label="structure-preserving"),
               Patch(facecolor=colors["violating"], label="structure-violating"),
               Patch(facecolor=colors["base"], label="baseline")]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/ablation.pdf", bbox_inches="tight")
    print(f"saved {OUT}/ablation.pdf")


if __name__ == "__main__":
    fig_efficiency()
    fig_ablation()
    print("DONE")
