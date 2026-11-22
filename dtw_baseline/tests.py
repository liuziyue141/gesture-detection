"""Correctness checks. Run:  .venv/bin/python dtw_baseline/tests.py"""
import itertools

import numpy as np

from data import frame_features, interpolate_gaps
from dtw import local_costs, subsequence_dtw
from evaluate import detect, match

rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------- DTW vs brute force
def brute_force(C):
    """Enumerate every allowed path explicitly (independent of the DP)."""
    m, T = C.shape
    d = np.full(T, np.inf)
    start = np.full(T, -1)
    for steps in itertools.product([0, 1, 2], repeat=m - 1):
        if any(a == 0 and b == 0 for a, b in zip(steps, steps[1:])):
            continue                                  # no two 'reuse' steps in a row
        offs = np.concatenate([[0], np.cumsum(steps, dtype=int)]).astype(int)
        for s in range(T - offs[-1]):
            cost = C[np.arange(m), s + offs].sum() / m
            t = s + offs[-1]
            if cost < d[t]:
                d[t], start[t] = cost, s
    return d, start


def test_matches_brute_force():
    for trial in range(30):
        m, T = rng.integers(1, 6), rng.integers(1, 14)
        C = rng.random((m, T))
        d, s = subsequence_dtw(C)
        d_ref, s_ref = brute_force(C)
        assert np.allclose(d, d_ref), (trial, d, d_ref)
        ok = np.isfinite(d_ref)
        assert (s[ok] == s_ref[ok]).all(), (trial, s, s_ref)


# ----------------------------------------------------------------------------- warped gestures
def resample(x, n):
    src = np.linspace(0, len(x) - 1, n)
    return np.stack([np.interp(src, np.arange(len(x)), x[:, j]) for j in range(x.shape[1])], 1)


def test_finds_warped_copies():
    dim, m = 6, 15
    Q = np.cumsum(rng.normal(size=(m, dim)), 0)
    Q -= Q.mean(0)
    stream = [rng.normal(scale=3, size=(20, dim))]
    truth = []
    for n in [15, 24, 9, 45]:                 # same speed, 1.6x slower, 1.7x faster, 3x slower
        seg = resample(Q, n) + rng.normal(scale=0.1, size=(n, dim))
        s0 = sum(len(x) for x in stream)
        stream += [seg, rng.normal(scale=3, size=(20, dim))]
        truth.append((s0, s0 + n - 1, n))
    X = np.concatenate(stream)
    d, start = subsequence_dtw(local_costs(Q, np.ones(m, bool), X, np.ones(len(X), bool), 5.0))
    bg = np.ones(len(X), bool)
    for s0, e0, _ in truth:
        bg[s0:e0 + 8] = False
    print(f"  background: min d={d[bg].min():.2f}")
    for s0, e0, n in truth:
        t = s0 + np.argmin(d[s0:e0 + 8])
        print(f"  copy of length {n:2d} (true span {s0}-{e0}): best d={d[t]:.2f} at t={t}, start={start[t]}")
        if n <= 2 * m - 1:   # inside the allowed warp: found at the right place, well below background
            assert abs(t - e0) <= 1 and abs(start[t] - s0) <= 1 and d[t] < 0.5 * d[bg].min()
    # the 3x-slower copy is outside the allowed warp and must score clearly worse than the in-range ones
    assert d[truth[3][0]:truth[3][1] + 8].min() > 2 * max(d[e0] for _, e0, _ in truth[:3])


def test_missing_frames():
    Q, X = rng.normal(size=(4, 3)), rng.normal(size=(6, 3))
    qp, xp = np.array([1, 1, 0, 1], bool), np.array([1, 0, 1, 1, 0, 1], bool)
    C = local_costs(Q, qp, X, xp, c_miss=7.0)
    assert C[0, 1] == 7.0 and C[2, 0] == 7.0 and C[2, 1] == 7.0
    assert np.isclose(C[0, 0], np.linalg.norm(Q[0] - X[0]))


# ----------------------------------------------------------------------------- features
def test_features_invariant_to_position_and_scale():
    pts = rng.random((10, 21, 2))
    f1, p1 = frame_features(pts)
    f2, p2 = frame_features(pts * 2.5 + np.array([0.3, -0.1]))
    assert p1.all() and np.allclose(f1, f2)
    assert np.allclose(f1[0, 42:], 0)                 # no motion on the first frame


def test_interpolation():
    pts = np.zeros((8, 21, 2))
    pts[:, 0, 0] = np.arange(8.0)
    pts[2:4] = np.nan                                 # gap of 2 -> filled
    pts[5:] = np.nan                                  # trailing gap -> stays missing
    out = interpolate_gaps(pts)
    assert np.allclose(out[:5, 0, 0], np.arange(5.0)) and np.isnan(out[5:, 0, 0]).all()


# ----------------------------------------------------------------------------- trigger + matching
def test_trigger_rule():
    T = 20
    a, b = np.full(T, 5.0), np.full(T, 5.0)
    a[2:9] = 0.5                                      # 7 matching frames -> one detection at t=4
    a[10:12] = 0.5                                    # only 2 frames -> nothing
    b[13:17] = 0.4                                    # b fires at 15
    a[13:17] = 0.6                                    # a is below tau but not the best -> nothing
    assert detect({1: a, 2: b}, tau=1.0) == [(1, 4), (2, 15)]
    # long hold where the best template flickers between two similar gestures: still one detection
    a, b = np.full(T, 5.0), np.full(T, 5.0)
    a[2:18], b[2:18] = 0.5, 0.6
    b[8:10] = 0.4                                     # b briefly lowest (only 2 frames: no b detection)
    assert detect({1: a, 2: b}, tau=1.0) == [(1, 4)]


def test_vectorized_detect_matches_loop():
    from evaluate import detect_reference
    for _ in range(200):
        T, K = rng.integers(1, 60), rng.integers(1, 5)
        # piecewise-constant random scores so runs and ties actually occur
        scores = {k: np.repeat(rng.random(T // 3 + 1), 3)[:T] for k in range(1, K + 1)}
        tau = rng.uniform(0.2, 0.9)
        assert detect(scores, tau) == detect_reference(scores, tau)


def test_matching():
    inst = [(1, 10, 20), (1, 40, 50), (2, 60, 70)]
    dets = [(1, 12), (1, 15), (1, 54), (2, 90), (2, 75)]
    # (1,12) hits #0; (1,15) duplicate -> FP; (1,54) within delta of #1; (2,75) hits #2; (2,90) late -> FP
    assert match(sorted(dets, key=lambda d: d[1]), inst, delta=5) == (3, 2)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("all tests passed")
