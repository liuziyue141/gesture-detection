"""Turning per-frame DTW scores into detections, and scoring detections against labels."""
import numpy as np

PERSIST = 3     # a gesture fires after this many consecutive matching frames


def detect(scores, tau, persist=PERSIST):
    """scores: {class: d_k (T,)}. A frame 'matches' k if d_k(t) is the smallest score and below tau.

    Gesture k fires when its run of matching frames reaches `persist`. It re-arms only once its own
    score rises back above tau, so a long hold yields one detection even if, during the hold, the
    lowest-scoring template keeps switching between similar gestures.
    Returns a time-sorted list of (class, t).

    Vectorized form of `detect_reference`: within each maximal stretch where d_k < tau, fire at the
    first frame that ends `persist` consecutive matching frames.
    """
    classes = sorted(scores)
    D = np.stack([scores[k] for k in classes])          # (K, T)
    best = D.argmin(0)
    taus = tau if isinstance(tau, dict) else {k: tau for k in classes}
    dets = []
    for r, k in enumerate(classes):
        below = D[r] < taus[k]
        hit = (best == r) & below
        run_ok = np.convolve(hit, np.ones(persist, int), "full")[:len(hit)] == persist
        idx = np.flatnonzero(run_ok)
        if len(idx):
            stretch = np.cumsum(~below)[idx]               # id of the below-tau stretch
            _, first = np.unique(stretch, return_index=True)
            dets += [(k, int(t)) for t in idx[first]]
    return sorted(dets, key=lambda d: (d[1], d[0]))


def detect_reference(scores, tau, persist=PERSIST):
    """Plain loop version of `detect`, kept as the readable reference (tests check they agree)."""
    classes = sorted(scores)
    D = np.stack([scores[k] for k in classes])
    best = D.argmin(0)
    dets = []
    for r, k in enumerate(classes):
        below = D[r] < tau
        hit = (best == r) & below
        run, armed = 0, True
        for t in range(len(hit)):
            if not below[t]:
                armed = True
            run = run + 1 if hit[t] else 0
            if run == persist and armed:
                dets.append((k, t))
                armed = False
    return sorted(dets, key=lambda d: (d[1], d[0]))


def match_pairs(dets, instances, delta):
    """One-to-one matching. A detection (k, t) is correct if an unmatched instance (k, s, e) has
    s <= t <= e + delta; among several candidates the earliest instance is used.
    Returns (pairs, false_positives): pairs are (detection, instance) tuples.
    """
    used = [False] * len(instances)
    pairs, fps = [], []
    for k, t in dets:
        for j, (c, s, e) in enumerate(instances):
            if not used[j] and c == k and s <= t <= e + delta:
                used[j] = True
                pairs.append(((k, t), instances[j]))
                break
        else:
            fps.append((k, t))
    return pairs, fps


def match(dets, instances, delta):
    """Returns (tp, fp); every detection is either a hit or a false positive."""
    pairs, fps = match_pairs(dets, instances, delta)
    return len(pairs), len(fps)


def score(dets, instances, classes, delta):
    """(tp, fp, fn) counting only instances of the registered `classes`."""
    inst = [x for x in instances if x[0] in classes]
    tp, fp = match(dets, inst, delta)
    return tp, fp, len(inst) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f
