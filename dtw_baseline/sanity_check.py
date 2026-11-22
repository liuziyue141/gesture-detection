"""Small end-to-end run on real data (NOT the final evaluation).

  - standardizer fitted on 8 training videos
  - threshold tau picked on ONE training subject, then applied to ONE test subject
  - for each subject: demos cut from video 1, detection on videos 2-4
Run:  .venv/bin/python dtw_baseline/sanity_check.py
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import (CLASS_NAMES, FPS, ROOT, TEST_CLASSES, Standardizer,
                  frame_features, load_instances, load_landmarks, load_split, load_video,
                  pick_demos, subject_of)
from dtw import match_template
from evaluate import detect, prf, score

DELTA = int(0.5 * FPS)
def c_miss(X):
    """Expected distance between two independent z-scored frames of this dimension."""
    return np.sqrt(2 * X.shape[1])
OUT = ROOT / "dtw_baseline" / "figures"


POOL_SUBJECTS = 2          # reference videos used for calibration, from the END of the training list
_pool_cache = {}


def reference_pool(std, exclude_subject):
    """First video of the last POOL_SUBJECTS training subjects (never the subject being evaluated)."""
    train = sorted(load_split()["train"].items())
    vids = [v[0] for s, v in reversed(train) if s != exclude_subject][:POOL_SUBJECTS]
    for v in vids:
        if (v, id(std)) not in _pool_cache:
            _pool_cache[(v, id(std))] = (*load_video(v, std), load_instances(v))
    return [_pool_cache[(v, id(std))] for v in vids]


def background_level(demos, std, subject):
    """b_k = median score of template k on OTHER people's videos, over tracked frames that are not
    an instance of class k. Dividing by b_k puts all templates on a comparable scale.

    Why not the user's own non-gesture frames: in IPN Hand the hand is almost never visible outside
    a gesture (pointing fills the time between commands; otherwise the hand is out of view), so that
    set is empty or a handful of frames. Why exclude class k: pointing fills a large share of every
    video, so including it would give P1F/P2F an artificially small b_k. This uses training labels
    only, never the evaluated subject's."""
    pool = reference_pool(std, subject)
    b = {}
    for k, (f, p, _, _) in demos.items():
        vals = []
        for X, xp, inst in pool:
            d = match_template(f, p, X, xp, c_miss(X))[0]
            keep = xp.copy()
            for c, s, e in inst:
                if c == k:
                    keep[max(0, s - DELTA):e + DELTA + 1] = False
            vals.append(d[keep])
        vals = np.concatenate(vals)
        if len(vals) == 0:
            raise ValueError(f"empty calibration pool for class {k}")
        b[k] = np.median(vals)
    return b


def run_subject(videos, std, classes, calibrate):
    """Demos from videos[0] (gaps filled from later videos); returns per-video (scores, instances)
    for videos[1:]. If a demo was taken from an evaluation video, that instance is removed from the
    labels and no detection is allowed on it."""
    demos = pick_demos(videos, classes, std)
    b = background_level(demos, std, subject_of(videos[0])) if calibrate else {k: 1.0 for k in demos}
    out = []
    for v in videos[1:]:
        X, xp = load_video(v, std)
        scores = {k: match_template(f, p, X, xp, c_miss(X))[0] / b[k] for k, (f, p, _, _) in demos.items()}
        inst = load_instances(v)
        for k, (_, _, dv, (a, e)) in demos.items():
            if dv == v:
                inst = [x for x in inst if (x[1], x[2]) != (a, e)]
                for s in scores.values():
                    s[a:e + DELTA + 1] = np.inf
        out.append((v, scores, inst))
    return demos, out


def evaluate_at(runs, tau, classes):
    tp = fp = fn = 0
    minutes = 0.0
    for _, scores, inst in runs:
        a, b, c = score(detect(scores, tau), inst, classes, DELTA)
        tp, fp, fn = tp + a, fp + b, fn + c
        minutes += len(next(iter(scores.values()))) / FPS / 60
    return tp, fp, fn, fp / minutes


def per_instance_report(runs, classes):
    """For every labeled instance of a registered class: lowest score of each template in [s, e+delta]."""
    rows = []
    for v, scores, inst in runs:
        for c, s, e in inst:
            if c not in classes:
                continue
            mins = {k: scores[k][s:e + DELTA + 1].min() for k in classes}
            rank = sorted(mins, key=mins.get).index(c) + 1
            rows.append((v, CLASS_NAMES[c], mins[c], rank))
    return rows


def main():
    split = load_split()
    train_subj = sorted(split["train"])
    test_subj = sorted(split["test"])
    classes = TEST_CLASSES

    t0 = time.time()
    fit_videos = [split["train"][s][0] for s in train_subj[::5]][:8]
    feats = [frame_features(load_landmarks(v)) for v in fit_videos]
    std = Standardizer().fit([f for f, _ in feats], [p for _, p in feats])
    print(f"standardizer fitted on {len(fit_videos)} videos ({time.time() - t0:.1f}s)")

    tune_s, eval_s = train_subj[0], test_subj[0]
    for calibrate, grid in [(False, np.arange(1.0, 4.01, 0.25)), (True, np.arange(0.3, 1.01, 0.05))]:
        t0 = time.time()
        demos_tr, runs_tr = run_subject(split["train"][tune_s], std, classes, calibrate)
        demos_te, runs_te = run_subject(split["test"][eval_s], std, classes, calibrate)
        print(f"\n===== calibrate={calibrate}  ({time.time() - t0:.1f}s incl. JSON loading)")
        for name, demos in [(tune_s, demos_tr), (eval_s, demos_te)]:
            print(f"  demos for {name}: " + ", ".join(
                f"{CLASS_NAMES[k]}={len(f)}fr" for k, (f, _, v, _) in sorted(demos.items())))
        print(f"  tau sweep on train subject {tune_s} (videos 2-4):")
        best = None
        for tau in grid:
            tp, fp, fn, fpm = evaluate_at(runs_tr, tau, classes)
            p, r, f = prf(tp, fp, fn)
            print(f"    tau={tau:4.2f}  TP={tp:2d} FP={fp:3d} FN={fn:2d}  P={p:.2f} R={r:.2f} F1={f:.2f}  FP/min={fpm:.2f}")
            if best is None or f > best[1]:
                best = (tau, f)
        tau = best[0]
        tp, fp, fn, fpm = evaluate_at(runs_te, tau, classes)
        p, r, f = prf(tp, fp, fn)
        print(f"  -> test subject {eval_s} at tau={tau:.2f}: TP={tp} FP={fp} FN={fn}  "
              f"P={p:.2f} R={r:.2f} F1={f:.2f}  FP/min={fpm:.2f}")

    print("\nper-instance (test subject): own-template min score near the instance, and its rank among 5")
    rows = per_instance_report(runs_te, classes)
    for v, c, d, rank in rows:
        print(f"  {v:16s} {c}  d={d:.2f}  rank={rank}")
    print(f"  rank-1 rate: {np.mean([r == 1 for *_, r in rows]):.2f}")

    # plot score curves for one evaluation video
    OUT.mkdir(exist_ok=True)
    v, scores, inst = runs_te[0]
    fig, ax = plt.subplots(figsize=(14, 4))
    t = np.arange(len(next(iter(scores.values())))) / FPS
    colors = plt.cm.tab10(np.arange(len(classes)))
    for col, k in zip(colors, classes):
        ax.plot(t, scores[k], color=col, lw=0.9, label=CLASS_NAMES[k])
    for c, s, e in inst:
        if c in classes:
            ax.axvspan(s / FPS, e / FPS, color=colors[classes.index(c)], alpha=0.25)
        else:
            ax.axvspan(s / FPS, e / FPS, color="0.85", alpha=0.5, lw=0)
    for k, td in detect(scores, tau):
        ax.plot(td / FPS, tau, "v", color=colors[classes.index(k)], ms=8)
    ax.axhline(tau, color="k", ls="--", lw=0.8)
    ax.set(xlabel="time (s)", ylabel="calibrated score d_k(t) / b_k",
           title=f"{v}  (shaded: labeled gestures, grey: other classes; v: detections; dashed: tau)",
           ylim=(0, 4))
    ax.legend(ncol=5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "score_curves.png", dpi=120)
    print(f"\nsaved {OUT / 'score_curves.png'}")


if __name__ == "__main__":
    main()
