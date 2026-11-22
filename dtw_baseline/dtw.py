"""Subsequence DTW (open-begin) with a slope constraint and exact length normalization.

For a template Q = (q_1..q_m) and stream X = (x_1..x_T), the score at every frame t is

    d(t) = min over start s and warping path P from (1, s) to (m, t) of  (1/m) * sum_{(i,j) in P} c(i, j)

Every step of a path advances the template by exactly one frame and the stream by 0, 1 or 2 frames:

    (i-1, t-1) -> (i, t)   same speed
    (i-1, t-2) -> (i, t)   stream frame t-1 skipped : performance SLOWER than the demo
    (i-1, t  ) -> (i, t)   stream frame t reused    : performance FASTER than the demo
                           (not allowed twice in a row)

So every path has exactly m cells (dividing by m is exact), and a match spans between about m/2 and
2m-1 stream frames, i.e. the user may be up to ~2x faster or slower than the demo.

Rows of the DP only depend on the previous row, so we loop over the m template frames and vectorize
over all T stream frames. The result is causal: d(t) uses only x_1..x_t, so running it offline on a
whole video gives exactly what a streaming implementation would output frame by frame.
"""
import numpy as np

INF = np.inf


def local_costs(Q, q_present, X, x_present, c_miss):
    """c(i, t) = ||q_i - x_t|| if both frames are tracked, else c_miss. Shape (m, T).

    'Both missing' is deliberately NOT free: otherwise any template with untracked frames is
    rewarded for matching video where the hand is out of view.
    """
    sq = (Q ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None, :] - 2 * Q @ X.T
    C = np.sqrt(np.maximum(sq, 0.0))
    C[~(q_present[:, None] & x_present[None, :])] = c_miss
    return C


def _shift(a, k, fill):
    """a shifted right by k along the last axis: out[t] = a[t-k]."""
    out = np.full_like(a, fill)
    out[k:] = a[:-k]
    return out


def subsequence_dtw(C):
    """C: (m, T) local costs. Returns d (T,) normalized score and start (T,) index of the best match ending at t.

    Two DP states per cell so 'reuse stream frame' cannot repeat:
        N[i,t] : best cost of a path ending at (i,t) whose last step moved in the stream (or i = 0)
        A[i,t] : best cost of a path ending at (i,t) whose last step reused stream frame t
    """
    m, T = C.shape
    N = C[0].copy()                       # free start: a match may begin at any frame
    A = np.full(T, INF)
    sN = np.arange(T)
    sA = np.zeros(T, dtype=int)
    for i in range(1, m):
        best = np.minimum(N, A)
        s_best = np.where(N <= A, sN, sA)
        diag, skip = _shift(best, 1, INF), _shift(best, 2, INF)
        s_diag, s_skip = _shift(s_best, 1, 0), _shift(s_best, 2, 0)
        use_diag = diag <= skip
        N_new = C[i] + np.minimum(diag, skip)
        sN_new = np.where(use_diag, s_diag, s_skip)
        A_new = C[i] + N                  # reuse frame t: only after a step that moved
        sA_new = sN
        N, A, sN, sA = N_new, A_new, sN_new, sA_new
    S = np.minimum(N, A)
    start = np.where(N <= A, sN, sA)
    return S / m, start


def match_template(template, t_present, X, x_present, c_miss):
    return subsequence_dtw(local_costs(template, t_present, X, x_present, c_miss))
