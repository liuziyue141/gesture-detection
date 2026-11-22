"""Figure 1 for the milestone's Partial Results (setting A: 5 registered gestures, test subjects).
Run:  .venv/bin/python dtw_baseline/fig_partial_results.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import CLASS_NAMES, Standardizer, frame_features, load_landmarks, load_split
from evaluate import prf
from exp_full import FIVE, TAUS, evaluate, run_all, totals
from sanity_check import OUT

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3 (validated set)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})


def main():
    res = json.load(open(OUT / "full_results.json"))["A"]
    split = load_split()
    fit = [v[0] for _, v in sorted(split["train"].items())]
    F = [frame_features(load_landmarks(v)) for v in fit]
    std = Standardizer().fit([f for f, _ in F], [p for _, p in F])
    test = [r for r in run_all(split, std, FIVE) if r["set"] == "test"]
    P, R, F1 = map(np.array, zip(*[prf(*totals(evaluate(test, t, FIVE))) for t in TAUS]))

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.35), gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) threshold sweep
    for y, col, name, lw in [(P, BLUE, "Precision", 2), (R, ORANGE, "Recall", 2), (F1, AQUA, "F1", 2.6)]:
        a.plot(TAUS, y, color=col, lw=lw, label=name)
    tau = res["tau"]
    a.axvline(tau, color=INK2, ls=(0, (3, 2)), lw=0.9)
    a.text(tau + 0.01, 0.03, f"τ chosen on\ntrain subjects ({tau:.2f})", color=INK2, fontsize=7.5, va="bottom")
    a.set(xlabel="Detection threshold τ (calibrated DTW score)", ylabel="Event-level metric",
          ylim=(0, 1.02), xlim=(TAUS[0], 0.6))
    a.grid(axis="y", color=GRID, lw=0.6)
    a.legend(frameon=False, loc="center right", fontsize=7.5)
    a.set_title("(a) Threshold sweep, 5 gestures", fontsize=9, loc="left", color=INK)

    # (b) per-gesture precision / recall at the chosen tau
    names = [CLASS_NAMES[c] for c in FIVE]
    pc = [res["per_class"][n]["P"] for n in names]
    rc = [res["per_class"][n]["R"] for n in names]
    x = np.arange(len(names))
    w = 0.36
    b.bar(x - w / 2 - 0.01, pc, w, color=BLUE, label="Precision")
    b.bar(x + w / 2 + 0.01, rc, w, color=ORANGE, label="Recall")
    for xi, p_, r_ in zip(x, pc, rc):
        b.text(xi - w / 2 - 0.01, p_ + 0.02, f"{p_:.2f}", ha="center", fontsize=6.5, color=INK2)
        b.text(xi + w / 2 + 0.01, r_ + 0.02, f"{r_:.2f}", ha="center", fontsize=6.5, color=INK2)
    b.set(xticks=x, xticklabels=names, ylim=(0, 1.3), yticks=np.arange(0, 1.01, 0.2), ylabel="At chosen τ")
    b.grid(axis="y", color=GRID, lw=0.6)
    b.set_axisbelow(True)
    b.legend(frameon=False, ncol=2, loc="upper center", fontsize=7.5)
    b.set_title("(b) Per-gesture results", fontsize=9, loc="left", color=INK)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig1_partial_results.png", dpi=300)
    print("saved", OUT / "fig1_partial_results.png", "| F1 at tau:", F1[np.argmin(abs(TAUS - tau))])


if __name__ == "__main__":
    main()
