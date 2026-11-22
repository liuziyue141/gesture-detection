"""Full-scale DTW baseline evaluation, 5-gesture and 14-gesture settings.

Protocol
  - subjects: official IPN Hand split, 37 train / 13 test (subject-disjoint)
  - standardizer: fitted on the first video of every train subject
  - calibration: each template's score is divided by its median score on 2 reference videos from
    other (train) subjects, excluding its own class there (see sanity_check.background_level)
  - demos: each subject's 4 videos are rotated; in rotation r, demos come from video r (classes missing
    there are filled from the next videos, and that instance is masked out), detection runs on the
    other 3 videos. Each video is therefore evaluated 3 times, with 3 different demo sets.
  - tau: one global threshold per setting, chosen to maximize F1 on the 37 train subjects
    (DTW has no trainable parameters, so the train subjects act as the validation set)
  - metrics on the 13 test subjects: event-level P/R/F1 (delta = 0.5 s), FP per minute,
    95% CI by bootstrapping test subjects, oracle classification accuracy, detection latency
Run:  .venv/bin/python dtw_baseline/exp_full.py
"""
import json
import time
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import (CLASS_NAMES, FPS, TEST_CLASSES, Standardizer, load_landmarks, frame_features,
                  load_split)
from evaluate import detect, match_pairs, prf
from sanity_check import DELTA, OUT, run_subject

ALL = list(range(1, 15))
FIVE = TEST_CLASSES                                   # TUP TDN TLT TRT OP2
DYNAMIC = [5, 6, 7, 8, 9, 12, 13, 14]                  # throws, OP2, zooms, grab
POINTING = [1, 2, 3, 4, 10, 11]                        # pointing + clicks performed from pointing
TAUS = np.round(np.arange(0.20, 0.801, 0.025), 3)
rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------- running
def run_all(split, std, classes):
    """List of run records over all subjects x 4 demo rotations x 3 evaluation videos."""
    recs = []
    for set_name in ["train", "test"]:
        for subj, vids in sorted(split[set_name].items()):
            for r in range(len(vids)):
                rot = vids[r:] + vids[:r]
                _, runs = run_subject(rot, std, classes, calibrate=True)
                for v, scores, inst in runs:
                    recs.append(dict(set=set_name, subject=subj, rot=r, video=v, scores=scores,
                                     inst=inst, frames=len(next(iter(scores.values())))))
    return recs


def label_at(inst, t):
    for c, s, e in inst:
        if s <= t <= e + DELTA:
            return c
    return 0


def evaluate(recs, tau, focus):
    """Counts restricted to `focus` classes (detections of other classes are ignored, instances of
    other classes are not counted). Returns per-record list of dicts."""
    out = []
    for rec in recs:
        dets = [d for d in detect(rec["scores"], tau) if d[0] in focus]
        inst = [x for x in rec["inst"] if x[0] in focus and x[0] in rec["scores"]]
        pairs, fps = match_pairs(dets, inst, DELTA)
        out.append(dict(rec=rec, pairs=pairs, fps=fps, inst=inst))
    return out


def totals(ev, classes=None):
    tp = fp = fn = 0
    for e in ev:
        keep = (lambda c: True) if classes is None else (lambda c: c in classes)
        a = sum(keep(i[0]) for _, i in e["pairs"])
        tp += a
        fp += sum(keep(k) for k, _ in e["fps"])
        fn += sum(keep(i[0]) for i in e["inst"]) - a
    return tp, fp, fn


def tune(recs_train, focus):
    f1 = [prf(*totals(evaluate(recs_train, t, focus)))[2] for t in TAUS]
    return TAUS[int(np.argmax(f1))], f1


def bootstrap_f1(ev, n=2000):
    by_subj = defaultdict(lambda: np.zeros(3))
    for e in ev:
        by_subj[e["rec"]["subject"]] += totals([e])
    arr = np.array(list(by_subj.values()))
    f1s = []
    for _ in range(n):
        s = arr[rng.integers(0, len(arr), len(arr))].sum(0)
        f1s.append(prf(*s)[2])
    per_subject = [prf(*a)[2] for a in arr]
    return np.percentile(f1s, [2.5, 97.5]), per_subject


def oracle(recs, focus):
    M = np.zeros((15, 15), int)
    for rec in recs:
        sc = rec["scores"]
        for c, s, e in rec["inst"]:
            if c in focus and c in sc:
                mins = {k: sc[k][s:e + DELTA + 1].min() for k in sc}
                M[c, min(mins, key=mins.get)] += 1
    return M


def summarize(name, recs, focus, tau_focus=None):
    train = [r for r in recs if r["set"] == "train"]
    test = [r for r in recs if r["set"] == "test"]
    tau, f1_curve = tune(train, focus) if tau_focus is None else (tau_focus, None)
    ev = evaluate(test, tau, focus)
    tp, fp, fn = totals(ev)
    p, r, f = prf(tp, fp, fn)
    minutes = sum(rec["frames"] for rec in test) / FPS / 60
    ci, per_subj = bootstrap_f1(ev)
    M = oracle(test, focus)
    lat = [(t - i[2]) / FPS for e in ev for (_, t), i in e["pairs"]]
    res = dict(name=name, tau=float(tau), tp=tp, fp=fp, fn=fn, P=p, R=r, F1=f, fp_per_min=fp / minutes,
               f1_ci=[float(x) for x in ci], f1_subject_mean=float(np.mean(per_subj)),
               f1_subject_std=float(np.std(per_subj)), oracle_acc=float(np.trace(M) / M.sum()),
               latency_median_s=float(np.median(lat)) if lat else None,
               latency_iqr_s=[float(x) for x in np.percentile(lat, [25, 75])] if lat else None,
               test_minutes=minutes, n_instances=tp + fn)
    print(f"\n=== {name}  (tau={tau:.3f}, chosen on train)")
    print(f"  P={p:.3f} R={r:.3f} F1={f:.3f}  [95% CI {ci[0]:.3f}-{ci[1]:.3f}]  FP/min={fp / minutes:.2f}  "
          f"(TP={tp} FP={fp} FN={fn}, {minutes:.0f} min evaluated)")
    print(f"  per-subject F1 {np.mean(per_subj):.3f} +- {np.std(per_subj):.3f} "
          f"(min {min(per_subj):.2f}, max {max(per_subj):.2f})")
    print(f"  oracle classification acc {res['oracle_acc']:.3f};  latency after gesture end: median "
          f"{res['latency_median_s']:.2f}s, IQR {res['latency_iqr_s'][0]:.2f}-{res['latency_iqr_s'][1]:.2f}s")
    per_class = {}
    print("  class   TP   FP   FN     P     R    F1   oracle")
    for c in focus:
        a, b, d = totals(ev, [c])
        pc, rc, fc = prf(a, b, d)
        oc = M[c, c] / max(M[c].sum(), 1)
        per_class[CLASS_NAMES[c]] = dict(tp=a, fp=b, fn=d, P=pc, R=rc, F1=fc, oracle=oc)
        print(f"  {CLASS_NAMES[c]:5s} {a:4d} {b:4d} {d:4d}  {pc:.2f}  {rc:.2f}  {fc:.2f}   {oc:.2f}")
    res["per_class"] = per_class
    return res, ev, M, f1_curve


def main():
    t0 = time.time()
    split = load_split()
    fit = [v[0] for _, v in sorted(split["train"].items())]
    F = [frame_features(load_landmarks(v)) for v in fit]
    std = Standardizer().fit([f for f, _ in F], [p for _, p in F])
    print(f"standardizer on {len(fit)} train videos ({time.time() - t0:.0f}s)")

    results = {}
    t1 = time.time()
    recs5 = run_all(split, std, FIVE)
    t2 = time.time()
    recs14 = run_all(split, std, ALL)
    t3 = time.time()
    frames14 = sum(r["frames"] for r in recs14)
    print(f"DTW runs: 5-gesture {t2 - t1:.0f}s, 14-gesture {t3 - t2:.0f}s "
          f"({frames14} frames x 14 templates; {1e6 * (t3 - t2) / frames14 / 14:.1f} us per frame-template incl. I/O)")

    r5, ev5, M5, c5 = summarize("A. register 5, score 5", recs5, FIVE)
    r14, ev14, M14, c14 = summarize("B. register 14, score 14", recs14, ALL)
    r14on5, _, _, _ = summarize("C. register 14, score the same 5 (tau re-tuned on train)", recs14, FIVE)
    results.update(A=r5, B=r14, C=r14on5)

    # groups inside setting B (same detections, split by class)
    print("\n=== B by group")
    for gname, g in [("dynamic", DYNAMIC), ("pointing-family", POINTING)]:
        a, b, d = totals(ev14, g)
        p, r, f = prf(a, b, d)
        results[f"B_{gname}"] = dict(tp=a, fp=b, fn=d, P=p, R=r, F1=f)
        print(f"  {gname:16s} TP={a:4d} FP={b:4d} FN={d:4d}  P={p:.3f} R={r:.3f} F1={f:.3f}")

    # where do B's false positives land?
    where = Counter()
    for e in ev14:
        for k, t in e["fps"]:
            where[(CLASS_NAMES[k], CLASS_NAMES[label_at(e["rec"]["inst"], t)])] += 1
    nfp = sum(where.values())
    by_truth = Counter()
    for (k, c), n in where.items():
        by_truth[c] += n
    print(f"\n=== B false positives ({nfp}) by what was really happening:")
    for c, n in by_truth.most_common():
        print(f"  during {c:4s}: {n:5d} ({n / nfp:.0%})")
    print("  top (detected <- truth):", ", ".join(f"{k}<-{c} x{n}" for (k, c), n in where.most_common(8)))
    results["B_fp_by_truth"] = {c: n for c, n in by_truth.items()}

    # left- vs right-hand videos (setting B)
    print("\n=== B by recording hand")
    for hand in ["_R_", "_L_"]:
        sub = [e for e in ev14 if hand in e["rec"]["video"]]
        if sub:
            a, b, d = totals(sub)
            p, r, f = prf(a, b, d)
            results[f"B_hand{hand.strip('_')}"] = dict(videos=len({e['rec']['video'] for e in sub}), F1=f, P=p, R=r)
            print(f"  {hand.strip('_')}: {len({e['rec']['video'] for e in sub})} videos  P={p:.3f} R={r:.3f} F1={f:.3f}")

    # ----- figures
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    Mn = M14[1:, 1:] / np.maximum(M14[1:, 1:].sum(1, keepdims=True), 1)
    ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    for i in range(14):
        for j in range(14):
            if Mn[i, j] >= 0.05:
                ax.text(j, i, f"{Mn[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if Mn[i, j] > 0.5 else "black")
    names = CLASS_NAMES[1:]
    ax.set(xticks=range(14), yticks=range(14), xticklabels=names, yticklabels=names,
           xlabel="lowest-scoring template near the instance", ylabel="true gesture",
           title=f"DTW oracle classification, 14 gestures, test subjects (acc {r14['oracle_acc']:.2f})")
    plt.setp(ax.get_xticklabels(), rotation=60)
    fig.tight_layout()
    fig.savefig(OUT / "full_confusion_14.png", dpi=130)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    test5 = [r for r in recs5 if r["set"] == "test"]
    test14 = [r for r in recs14 if r["set"] == "test"]
    for ax, recs, focus, res, title in [(axes[0], test5, FIVE, r5, "A: register 5"),
                                        (axes[1], test14, ALL, r14, "B: register 14")]:
        P, R, F1 = zip(*[prf(*totals(evaluate(recs, t, focus))) for t in TAUS])
        ax.plot(TAUS, P, label="precision")
        ax.plot(TAUS, R, label="recall")
        ax.plot(TAUS, F1, label="F1", lw=2)
        ax.axvline(res["tau"], color="k", ls="--", lw=0.8, label="tau chosen on train")
        ax.set(xlabel="threshold tau (calibrated score)", title=f"{title} (test subjects)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("event-level metric")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "full_pr_vs_tau.png", dpi=130)

    with open(OUT / "full_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nsaved figures + {OUT / 'full_results.json'}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
