#!/usr/bin/env python3
"""Generate publication-quality PDF figures for the paper (revised, 5 figures)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, glob, os
import numpy as np

OUT = "/mnt/e/Project/MDPI/symmetry-paper-2/paper/figures"
RES = "/mnt/e/Project/MDPI/symmetry-paper-2/results"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "figure.dpi": 300,
})

C_NONE="#888888"; C_HF="#1f77b4"; C_NF="#2ca02c"; C_STD="#d62728"


def fig_efficiency():
    fractions = [25, 50, 100]
    none     = [32.9, 47.5, 60.2]
    hflip    = [42.2, 56.0, 66.7]
    std_nf   = [50.6, 62.2, 70.7]
    standard = [53.1, 64.7, 73.6]
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    ax.plot(fractions, none,     "o--", color=C_NONE, label="None")
    ax.plot(fractions, hflip,    "s-",  color=C_HF,   label="hflip (mirror only)")
    ax.plot(fractions, std_nf,   "D-",  color=C_NF,   label=r"standard$_{\rm noflip}$ (crop+jitter)")
    ax.plot(fractions, standard, "^-",  color=C_STD,  label="standard (+flip)")
    ax.set_xlabel("Training data (%)"); ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_xticks(fractions); ax.set_ylim(25, 80)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.annotate("", xy=(25, 53.1), xytext=(25, 50.6),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=1.1))
    ax.text(26, 51.4, "flip +2.5", color="#444", fontsize=8.5)
    ax.annotate("", xy=(25, 42.2), xytext=(25, 32.9),
                arrowprops=dict(arrowstyle="<->", color=C_HF, lw=1.1))
    ax.text(11, 35.5, "flip-only +9.3", color=C_HF, fontsize=8.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/efficiency.pdf", bbox_inches="tight")
    print("saved efficiency.pdf")


def fig_flip_margin():
    """Marginal contribution decomposition: crop+jitter vs flip, across data fractions."""
    fracs = ["25%", "50%", "100%"]
    cropjitter = [50.6-32.9, 62.2-47.5, 70.7-60.2]   # standard_noflip - none
    flip       = [53.1-50.6, 64.7-62.2, 73.6-70.7]   # standard - standard_noflip
    x = np.arange(len(fracs)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    b1 = ax.bar(x-w/2, cropjitter, w, color=C_NF,  label="crop + color jitter")
    b2 = ax.bar(x+w/2, flip,       w, color=C_STD, label="mirror flip (symmetry)")
    ax.set_xticks(x); ax.set_xticklabels(fracs)
    ax.set_xlabel("Training data"); ax.set_ylabel("Marginal accuracy gain (pts)")
    ax.set_ylim(0, 12); ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    for b in list(b1)+list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.15, f"+{b.get_height():.1f}",
                ha="center", fontsize=8.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{OUT}/flip_margin.pdf", bbox_inches="tight")
    print("saved flip_margin.pdf")


def fig_ablation():
    labels = ["none", "vflip", "rot90", "rot30", "rot15", "hflip"]
    acc    = [60.0,  61.1,    64.8,    67.7,    68.3,    67.2]
    kind   = ["base", "violating", "violating", "preserving", "preserving", "preserving"]
    colors = {"base": C_NONE, "violating": C_STD, "preserving": C_NF}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bars = ax.bar(labels, acc, color=[colors[k] for k in kind], edgecolor="black", linewidth=0.5)
    ax.axhline(60.0, color=C_NONE, linestyle=":", lw=1)
    ax.set_ylabel("Top-1 accuracy (%)"); ax.set_ylim(55, 72)
    for b, a in zip(bars, acc):
        ax.text(b.get_x()+b.get_width()/2, a+0.3, f"{a:.1f}", ha="center", fontsize=9)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors["preserving"], label="structure-preserving"),
               Patch(facecolor=colors["violating"], label="structure-violating"),
               Patch(facecolor=colors["base"], label="baseline")]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/ablation.pdf", bbox_inches="tight")
    print("saved ablation.pdf")


def fig_robustness_per_corr():
    """Per-corruption mCA for none / hflip / standard_noflip / standard."""
    rb = json.load(open(f"{RES}/exp4_robustness.json"))
    corrs = list(rb["corruptions"].keys())
    none = [rb["corruptions"][c]["none"]["avg"]            for c in corrs]
    hf   = [rb["corruptions"][c]["hflip"]["avg"]           for c in corrs]
    nf   = [rb["corruptions"][c]["standard_noflip"]["avg"] for c in corrs]
    std  = [rb["corruptions"][c]["standard"]["avg"]        for c in corrs]
    order = np.argsort(none)
    corrs = [corrs[i] for i in order]
    x = np.arange(len(corrs)); w = 0.21
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    ax.bar(x-1.5*w, [none[i] for i in order], w, color=C_NONE, label="None")
    ax.bar(x-0.5*w, [hf[i]   for i in order], w, color=C_HF,   label="hflip")
    ax.bar(x+0.5*w, [nf[i]   for i in order], w, color=C_NF,   label=r"std$_{\rm noflip}$")
    ax.bar(x+1.5*w, [std[i]  for i in order], w, color=C_STD,  label="standard")
    ax.set_xticks(x); ax.set_xticklabels([c.replace("_","\n") for c in corrs], fontsize=7, rotation=0)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
    ax.axhline(69.4, color=C_NONE, ls=":", lw=0.8)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/robustness.pdf", bbox_inches="tight")
    print("saved robustness.pdf")


def fig_training_curves():
    """Test accuracy vs epoch for CIFAR-100 full, 4 conditions (from saved history)."""
    augs = {"none":C_NONE, "hflip":C_HF, "standard_noflip":C_NF, "standard":C_STD}
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    for aug, col in augs.items():
        f = glob.glob(f"{RES}/control__cifar100__resnet50__{aug}.json")
        if not f: continue
        r = json.load(open(f[0]))
        h = r["history"]["test_top1"]
        ax.plot(range(1, len(h)+1), h, color=col, label=aug)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test top-1 accuracy (%)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/training_curves.pdf", bbox_inches="tight")
    print("saved training_curves.pdf")


if __name__ == "__main__":
    fig_efficiency()
    fig_flip_margin()
    fig_ablation()
    fig_robustness_per_corr()
    fig_training_curves()
    print("DONE")
