"""Figures for the paper/poster (matplotlib, validated categorical palette)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

_FONT_DIR = Path(__file__).resolve().parent.parent / "poster" / "fonts"
_FAMILY = "DejaVu Sans"
if (_FONT_DIR / "Lato-Regular.ttf").exists():
    for f in _FONT_DIR.glob("Lato-*.ttf"):
        font_manager.fontManager.addfont(str(f))
    _FAMILY = "Lato"

BLUE, ORANGE, AQUA, VIOLET = "#6aaaf0", "#1f4fa8", "#1baf7a", "#a070e0"  # ORANGE slot is deep blue
INK, INK2, MUTED, GRID, SURF = "#16181d", "#4a4d55", "#8a8d94", "#e6e8ec", "#ffffff"
LEVEL_NAME = {"L0": "Answer only", "L1": "+ citations", "L2": "+ scores"}

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 18, "axes.labelsize": 15, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "legend.frameon": False, "font.family": [_FAMILY, "DejaVu Sans"],
})


def _save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=220)
    plt.close(fig)


def fig_leak(M, out):
    pols = ["L0", "L1", "L2"]
    auc = [M["membership"][p]["auc"] for p in pols]
    ci = np.array([M["membership"][p]["auc_ci"] for p in pols])
    tpr = [M["membership"][p]["tpr@5%fpr"] for p in pols]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    x = np.arange(3)
    ax.bar(x, auc, width=0.55, color=[BLUE, ORANGE, AQUA], edgecolor=SURF, linewidth=2)
    ax.errorbar(x, auc, yerr=[np.array(auc) - ci[:, 0], ci[:, 1] - np.array(auc)], fmt="none",
                ecolor=INK2, elinewidth=1.6, capsize=6)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1.4)
    ax.text(0.5, 0.507, "random", color=MUTED, fontsize=11.5, ha="center", va="bottom")
    for i, (a, t) in enumerate(zip(auc, tpr)):
        ax.text(i, ci[i, 1] + 0.02, f"AUC {a:.2f}\nTPR@5%FPR {t:.0%}", ha="center", va="bottom",
                color=INK, fontsize=12.5)
    ax.set_xticks(x, [LEVEL_NAME[p] for p in pols])
    ax.set_ylim(0.3, max(1.05, ci[:, 1].max() + 0.2))
    ax.set_yticks([0.3, 0.5, 0.7, 0.9, 1.0])
    ax.set_ylabel("Membership attack AUC")
    ax.set_title("Membership attack success by disclosure", color=INK, loc="left")
    _save(fig, out, "fig_leak")


def fig_roc(M, out):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    names = {"L0": "Answer only", "L1": "+ citations", "L2": "+ scores", "shield(1.0)": "Privacy Shield (ε=1)"}
    cols = {"L0": BLUE, "L1": ORANGE, "L2": AQUA, "shield(1.0)": VIOLET}
    for p, (fpr, tpr) in M["rocs"].items():
        ax.plot(fpr, tpr, color=cols[p], lw=2.4,
                label=f"{names[p]} (AUC {M['membership'][p]['auc']:.2f})")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1.2)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC of the membership attack", color=INK, loc="left")
    ax.legend(loc="lower right", fontsize=12.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    _save(fig, out, "fig_roc")


def fig_frontier(M, out):
    from .disclosure import EPS_GRID
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    F, Mi = M["faithfulness"], M["membership"]
    for fam, col, lab in [("shield", BLUE, "Privacy Shield (unlinkable ids + DP noise)"),
                          ("dp", ORANGE, "DP noise only (stable ids)")]:
        xs = [Mi[f"{fam}({e})"]["auc"] for e in EPS_GRID]
        ys = [F[f"{fam}({e})"]["top1_agreement"] for e in EPS_GRID]
        ax.plot(xs, ys, "-o", color=col, lw=2.2, ms=9, mec=SURF, mew=2, label=lab)
        if fam == "shield":
            for e, xx, yy in zip(EPS_GRID, xs, ys):
                if e in (0.5, 2.0, 8.0, 32.0):
                    ax.annotate(f"ε={e:g}", (xx, yy), textcoords="offset points", xytext=(8, -16),
                                fontsize=11.5, color=INK2)
    refs = [("L2", "Status quo (ids + scores)", "D"), ("L1", "Citations only (ids)", "s"),
            ("coarse", "High / Med / Low", "^"), ("unlink", "Unlinkable ids only", "o")]
    for p, lab, mk in refs:
        ax.plot(Mi[p]["auc"], F[p]["top1_agreement"], mk, color=INK2, mfc="none", ms=11, mew=2.2,
                ls="none", label=lab)
    ax.axvline(0.5, color=MUTED, ls="--", lw=1.2)
    ax.axvline(Mi["L0"]["auc"], color=MUTED, ls=":", lw=1.4)
    ax.text(0.49, 0.03, "random", color=MUTED, fontsize=11, rotation=90, ha="right", va="bottom")
    ax.text(Mi["L0"]["auc"] + 0.01, 0.03, "answer-only floor", color=MUTED, fontsize=11,
            rotation=90, ha="left", va="bottom")
    ax.set_xlabel("Membership attack AUC  (← more private)")
    ax.set_ylabel("Sources-panel faithfulness\n(shown top source = true top source)")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0.45, max(1.0, max(Mi[p]["auc"] for p in Mi) + 0.05))
    ax.set_title("Privacy vs. faithfulness of the sources panel", color=INK, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=11.5, ncol=2,
              columnspacing=1.2, handletextpad=0.5)
    _save(fig, out, "fig_frontier")


def fig_attr(M, out):
    pols = ["L0", "L1", "L2", "shield(1.0)"]
    lab = ["Answer\nonly", "+ citations", "+ scores", "Shield\nε=1"]
    acc = [M["attribute"][p]["accuracy"] for p in pols]
    ci = np.array([M["attribute"][p]["ci"] for p in pols])
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    x = np.arange(len(pols))
    ax.bar(x, acc, width=0.55, color=[BLUE, ORANGE, AQUA, VIOLET], edgecolor=SURF, linewidth=2)
    ax.errorbar(x, acc, yerr=[np.array(acc) - ci[:, 0], ci[:, 1] - np.array(acc)], fmt="none",
                ecolor=INK2, elinewidth=1.6, capsize=6)
    ax.axhline(0.2, color=MUTED, ls="--", lw=1.4)
    ax.text(0.5, 0.212, "chance", color=MUTED, fontsize=11.5, ha="center", va="bottom")
    for i, a in enumerate(acc):
        ax.text(i, ci[i, 1] + 0.02, f"{a:.0%}", ha="center", color=INK, fontsize=13)
    ax.set_xticks(x, lab)
    ax.set_ylim(0, max(1.05, ci[:, 1].max() + 0.12))
    ax.set_ylabel("Diagnosis guessed correctly")
    ax.set_title("Which condition does the target have?", color=INK, loc="left")
    _save(fig, out, "fig_attr")


def fig_hist(M, out):
    h = M["hist_top_patient_share"]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    bins = np.linspace(0, 1, 21)
    ax.hist(h["nonmember"], bins=bins, color=ORANGE, alpha=0.55, label="Not a patient", edgecolor=SURF)
    ax.hist(h["member"], bins=bins, color=BLUE, alpha=0.55, label="Is a patient", edgecolor=SURF)
    ax.set_xlabel("Top patient-record attribution share (avg. of 3 questions)")
    ax.set_ylabel("Number of people")
    ax.set_title("Top patient-record share: patients vs. not", color=INK, loc="left")
    ax.legend(fontsize=13)
    _save(fig, out, "fig_hist")


def make_all(M, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_leak, fig_roc, fig_frontier, fig_attr, fig_hist):
        f(M, out)
