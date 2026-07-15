#!/usr/bin/env python3
"""Generate publication-quality PDF figures for the paper (revised)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/mnt/e/Project/MDPI/symmetry-paper-2/paper/figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "figure.dpi": 300,
})


def fig_efficiency():
    """CIFAR-100 control: 4 conditions across data fractions.
    Shows (a) flip-only gain over none grows as data shrink,
    (b) marginal flip contribution (standard - standard_noflip) is stable."""
    fractions = [25, 50, 100]
    none     = [32.9, 47.5, 60.2]
    hflip    = [42.2, 56.0, 66.7]
    std_nf   = [50.6, 62.2, 70.7]
    standard = [53.1, 64.7, 73.6]

    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    ax.plot(fractions, none,     "o--", color="#888888", label="None")
    ax.plot(fractions, hflip,    "s-",  color="#1f77b4", label="hflip (mirror only)")
    ax.plot(fractions, std_nf,   "D-",  color="#2ca02c", label="standard$_{\\mathrm{noflip}}$ (crop+jitter)")
    ax.plot(fractions, standard, "^-",  color="#d62728", label="standard (+flip)")
    ax.set_xlabel("Training data (%)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_xticks(fractions)
    ax.set_ylim(25, 80)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    # annotate stable flip margin (standard - standard_noflip) at 25%
    ax.annotate("", xy=(25, 53.1), xytext=(25, 50.6),
                arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.1))
    ax.text(26, 51.4, "flip +2.5", color="#444444", fontsize=8.5)
    # annotate growing flip-only gain (hflip - none)
    ax.annotate("", xy=(25, 42.2), xytext=(25, 32.9),
                arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1.1))
    ax.text(11, 35.5, "flip-only +9.3", color="#1f77b4", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/efficiency.pdf", bbox_inches="tight")
    print(f"saved {OUT}/efficiency.pdf")


def fig_ablation():
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
