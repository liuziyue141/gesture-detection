"""Controlled comparison: does registering more templates hurt the same 5 gestures?

Same tuning subjects (3 train) and test subjects (3 test). Two settings:
  A) register only the 5 throw-group gestures
  B) register all 14 gestures
Both are scored ONLY on the 5 throw-group gestures: in B, detections of the other 9 classes are
ignored, so the difference isolates "more competing templates" for these 5 gestures.
tau is tuned separately for each setting on the tuning subjects.
Run:  .venv/bin/python dtw_baseline/exp_5_vs_14.py
"""
import numpy as np

from data import CLASS_NAMES, TEST_CLASSES, Standardizer, frame_features, load_landmarks, load_split
from evaluate import detect, match, prf
from sanity_check import DELTA, run_subject

N_SUBJ = 3
FOCUS = TEST_CLASSES           # TUP TDN TLT TRT OP2


def counts(runs, tau, focus):
    tp = fp = fn = 0
    for _, scores, inst in runs:
        dets = [d for d in detect(scores, tau) if d[0] in focus]
        ins = [x for x in inst if x[0] in focus and x[0] in scores]
        a, b = match(dets, ins, DELTA)
        tp, fp, fn = tp + a, fp + b, fn + len(ins) - a
    return tp, fp, fn


def oracle(runs, focus):
    ok = n = 0
    for _, scores, inst in runs:
        for c, s, e in inst:
            if c in focus and c in scores:
                mins = {k: scores[k][s:e + DELTA + 1].min() for k in scores}
                ok += min(mins, key=mins.get) == c
                n += 1
    return ok, n


def main():
    split = load_split()
    tr, te = sorted(split["train"]), sorted(split["test"])
    fit = [split["train"][s][0] for s in tr[::5]][:8]
    F = [frame_features(load_landmarks(v)) for v in fit]
    std = Standardizer().fit([f for f, _ in F], [p for _, p in F])
    for name, classes in [("A: register 5 ", FOCUS), ("B: register 14", list(range(1, 15)))]:
        runs_tr = [r for s in tr[:N_SUBJ] for r in run_subject(split["train"][s], std, classes, True)[1]]
        runs_te = [r for s in te[:N_SUBJ] for r in run_subject(split["test"][s], std, classes, True)[1]]
        tau = max(np.arange(0.2, 0.81, 0.025), key=lambda t: prf(*counts(runs_tr, t, FOCUS))[2])
        tp, fp, fn = counts(runs_te, tau, FOCUS)
        p, r, f = prf(tp, fp, fn)
        ok, n = oracle(runs_te, FOCUS)
        print(f"{name}: oracle {ok}/{n}={ok / n:.2f} | tau={tau:.3f} TP={tp} FP={fp} FN={fn} "
              f"P={p:.2f} R={r:.2f} F1={f:.2f}")


if __name__ == "__main__":
    print("scored on", [CLASS_NAMES[c] for c in FOCUS], f"| {N_SUBJ} tuning + {N_SUBJ} test subjects")
    main()
