"""Small-scale check: can DTW tell all 14 gestures apart (static + dynamic mixed)?

  - every subject registers all 14 classes (demos from video 1, gaps filled from later videos)
  - tau picked on 3 training subjects, evaluated on 3 test subjects (videos 2-4 each)
Reports
  (a) oracle classification: for each labeled instance, is its own template the lowest score near it?
      -> 14x14 confusion matrix (no threshold, location known)
  (b) detection P/R/F1 per class, and where the false positives land
Run:  .venv/bin/python dtw_baseline/exp_14class.py [--shape-vel]
"""
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

import data
from data import CLASS_NAMES, Standardizer, frame_features, load_landmarks, load_split
from evaluate import detect, match, prf
from sanity_check import DELTA, OUT, run_subject

CLASSES = list(range(1, 15))
N_SUBJ = 3


def label_at(inst, t):
    """Class whose [s, e+delta] window contains t (0 = no gesture)."""
    for c, s, e in inst:
        if s <= t <= e + DELTA:
            return c
    return 0


def oracle_confusion(runs):
    M = np.zeros((15, 15), int)
    for _, scores, inst in runs:
        for c, s, e in inst:
            if c in scores:
                mins = {k: scores[k][s:e + DELTA + 1].min() for k in scores}
                M[c, min(mins, key=mins.get)] += 1
    return M


def detection_report(runs, tau):
    per = {k: [0, 0, 0] for k in CLASSES}            # tp, fp, fn
    fp_where = Counter()                              # (detected class, true label at that time)
    for _, scores, inst in runs:
        reg = [x for x in inst if x[0] in scores]
        dets = detect(scores, tau)
        for k in CLASSES:
            dk = [d for d in dets if d[0] == k]
            ik = [x for x in reg if x[0] == k]
            tp, fp = match(dk, ik, DELTA)
            per[k][0] += tp
            per[k][1] += fp
            per[k][2] += len(ik) - tp
        # attribute FPs: re-run the joint matching and look up what was happening at each FP
        used = [False] * len(reg)
        for k, t in dets:
            hit = False
            for j, (c, s, e) in enumerate(reg):
                if not used[j] and c == k and s <= t <= e + DELTA:
                    used[j] = hit = True
                    break
            if not hit:
                fp_where[(k, label_at(reg, t))] += 1
    return per, fp_where


def totals(per):
    tp, fp, fn = (sum(v[i] for v in per.values()) for i in range(3))
    return tp, fp, fn


def main():
    split = load_split()
    train_subj, test_subj = sorted(split["train"]), sorted(split["test"])
    fit_videos = [split["train"][s][0] for s in train_subj[::5]][:8]
    feats = [frame_features(load_landmarks(v)) for v in fit_videos]
    std = Standardizer().fit([f for f, _ in feats], [p for _, p in feats])

    tune = [run_subject(split["train"][s], std, CLASSES, True) for s in train_subj[:N_SUBJ]]
    test = [run_subject(split["test"][s], std, CLASSES, True) for s in test_subj[:N_SUBJ]]
    runs_tr = [r for _, rs in tune for r in rs]
    runs_te = [r for _, rs in test for r in rs]
    minutes_te = sum(len(next(iter(s.values()))) for _, s, _ in runs_te) / 15 / 60
    print(f"tune subjects {train_subj[:N_SUBJ]}, test subjects {test_subj[:N_SUBJ]}  ({minutes_te:.1f} min of test video)")
    for (demos, _), s in zip(test, test_subj):
        print(f"  {s} demo lengths: " + " ".join(f"{CLASS_NAMES[k]}={len(d[0])}" for k, d in sorted(demos.items())))

    # (a) oracle classification on the test subjects
    M = oracle_confusion(runs_te)
    acc = np.trace(M) / M.sum()
    print(f"\n(a) oracle 14-way classification on test subjects: {np.trace(M)}/{M.sum()} = {acc:.2f}")
    print("    per class (correct/total, most common confusion):")
    for c in CLASSES:
        n = M[c].sum()
        if n == 0:
            continue
        row = M[c].copy()
        row[c] = 0
        conf = f"-> {CLASS_NAMES[row.argmax()]} x{row.max()}" if row.max() else ""
        print(f"      {CLASS_NAMES[c]:4s} {M[c, c]:2d}/{n:2d}  {conf}")

    # (b) detection: tau on tuning subjects
    print("\n(b) tau sweep on tuning subjects:")
    best = None
    for tau in np.arange(0.3, 0.81, 0.05):
        tp, fp, fn = totals(detection_report(runs_tr, tau)[0])
        p, r, f = prf(tp, fp, fn)
        print(f"    tau={tau:.2f}  TP={tp:3d} FP={fp:4d} FN={fn:3d}  P={p:.2f} R={r:.2f} F1={f:.2f}")
        if best is None or f > best[1]:
            best = (tau, f)
    tau = best[0]
    per, fp_where = detection_report(runs_te, tau)
    tp, fp, fn = totals(per)
    p, r, f = prf(tp, fp, fn)
    print(f"\n    test subjects at tau={tau:.2f}: TP={tp} FP={fp} FN={fn}  P={p:.2f} R={r:.2f} F1={f:.2f}  "
          f"FP/min={fp / minutes_te:.2f}")
    print("    per class:  TP  FP  FN    P    R")
    for k in CLASSES:
        a, b, c = per[k]
        pk, rk, _ = prf(a, b, c)
        print(f"      {CLASS_NAMES[k]:4s}    {a:3d} {b:3d} {c:3d}  {pk:.2f} {rk:.2f}")
    print("    false positives by (detected class <- what was really happening):")
    for (k, c), n in fp_where.most_common(12):
        print(f"      {CLASS_NAMES[k]:4s} <- {CLASS_NAMES[c]:4s}  x{n}")

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    Mn = M[1:, 1:] / np.maximum(M[1:, 1:].sum(1, keepdims=True), 1)
    ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    for i in range(14):
        for j in range(14):
            if M[i + 1, j + 1]:
                ax.text(j, i, M[i + 1, j + 1], ha="center", va="center", fontsize=7,
                        color="white" if Mn[i, j] > 0.5 else "black")
    names = CLASS_NAMES[1:]
    ax.set(xticks=range(14), yticks=range(14), xticklabels=names, yticklabels=names,
           xlabel="lowest-scoring template", ylabel="true gesture",
           title=f"Oracle 14-way DTW classification, {N_SUBJ} test subjects (acc {acc:.2f})")
    plt.setp(ax.get_xticklabels(), rotation=60)
    fig.tight_layout()
    name = f"confusion_14class{'_shapevel' if data.USE_SHAPE_VEL else ''}.png"
    fig.savefig(OUT / name, dpi=120)
    print(f"\nsaved {OUT / name}")


if __name__ == "__main__":
    data.USE_SHAPE_VEL = "--shape-vel" in sys.argv
    print(f"features: {'shape + wrist motion + finger velocity (87-d)' if data.USE_SHAPE_VEL else 'shape + wrist motion (45-d)'}")
    main()
