"""Loading IPN HandS landmarks/annotations and turning them into per-frame features.

Everything downstream runs at 15 fps: raw frame f (1-based, 30 fps) maps to index (f - 1) // 2.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LANDMARK_DIR = ROOT / "landmarks_mp_hands"
ANNOT_DIR = ROOT / "IPN_HandS_annotations"

CLASS_NAMES = ["NOG", "P1F", "P2F", "C1F", "C2F", "TUP", "TDN", "TLT", "TRT",
               "OP2", "2C1", "2C2", "ZIN", "ZOT", "GRB"]
TEST_CLASSES = [5, 6, 7, 8, 9]          # TUP TDN TLT TRT OP2
TRAIN_CLASSES = [c for c in range(1, 15) if c not in TEST_CLASSES]

SRC_FPS, FPS = 30, 15
STEP = SRC_FPS // FPS
ASPECT = 640 / 480                       # MediaPipe x is /width, y is /height
MAX_GAP = 3                              # interpolate tracking gaps up to this many frames (15 fps)


# ----------------------------------------------------------------------------- metadata
def subject_of(video):
    """'1CM1_1_R_#217' -> '1CM1_1' (recording scene + subject number)."""
    return "_".join(video.split("_")[:2])


def load_split():
    """Returns {subject: sorted list of its 4 videos} for 'train' and 'test' (official, subject-disjoint)."""
    out = {"train": defaultdict(list), "test": defaultdict(list)}
    with open(ANNOT_DIR / "frame_counts.csv") as f:
        for r in csv.DictReader(f):
            out[r["set"]][subject_of(r["video"])].append(r["video"])
    return {k: {s: sorted(v, key=lambda x: int(x.split("#")[1])) for s, v in d.items()}
            for k, d in out.items()}


def load_instances(video):
    """Gesture instances as (class, start, end) in 15-fps indices, inclusive. Non-gesture rows dropped."""
    inst = []
    with open(ANNOT_DIR / "temporal_annotations" / f"{video}.txt") as f:
        for line in f:
            c, a, b = (int(x) for x in line.split(","))
            if c:
                inst.append((c, (a - 1) // STEP, (b - 1) // STEP))
    return inst


# ----------------------------------------------------------------------------- landmarks
def load_landmarks(video):
    """(T, 21, 2) array of one tracked hand per 15-fps frame (NaN where no hand), aspect-corrected.

    MediaPipe's Left/Right label is unreliable on this data, so we ignore it and keep a single
    'active hand' slot: when several hands are detected, take the largest one.
    """
    with open(LANDMARK_DIR / f"{video}.json") as f:
        d = json.load(f)
    frames = sorted(d, key=lambda k: int(k.rsplit("_", 1)[1].split(".")[0]))[::STEP]
    out = np.full((len(frames), 21, 2), np.nan)
    for t, k in enumerate(frames):
        best, best_size = None, -1.0
        for h in d[k]["hand_landmarks"]:
            p = np.array([h["landmarks"][str(j)][:2] for j in range(21)])
            p[:, 0] *= ASPECT
            size = np.linalg.norm(p[9] - p[0])
            if size > best_size:
                best, best_size = p, size
        if best is not None:
            out[t] = best
    return out


def interpolate_gaps(pts, max_gap=MAX_GAP):
    """Linearly fill runs of missing frames of length <= max_gap that have tracked frames on both sides."""
    pts = pts.copy()
    ok = ~np.isnan(pts[:, 0, 0])
    idx = np.flatnonzero(ok)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            w = (np.arange(a + 1, b) - a) / (b - a)
            pts[a + 1:b] = (1 - w)[:, None, None] * pts[a] + w[:, None, None] * pts[b]
    return pts


# ----------------------------------------------------------------------------- features
USE_SHAPE_VEL = False   # also append the frame-to-frame change of the 42 shape values (finger motion)


def frame_features(pts, shape_vel=None):
    """pts (T,21,2) with NaN for missing -> feats (T,45 or 87), present (T,) bool.

    shape      : 21 points minus wrist, divided by hand size |p9 - p0|            (42)
    motion     : wrist velocity / size, and d(log size) between consecutive frames  (3)
    shape_vel  : (optional) shape[t] - shape[t-1], i.e. finger motion              (42)
    Motion is 0 when the previous frame is missing; features are 0 where the hand is missing.
    The presence flag is returned separately and used as a mask by the DTW local cost.
    """
    shape_vel = USE_SHAPE_VEL if shape_vel is None else shape_vel
    pts = interpolate_gaps(pts)
    present = ~np.isnan(pts[:, 0, 0])
    T = len(pts)
    feats = np.zeros((T, 45))
    wrist = pts[:, 0]
    size = np.linalg.norm(pts[:, 9] - pts[:, 0], axis=1)
    p = present
    feats[p, :42] = ((pts[p] - wrist[p, None]) / size[p, None, None]).reshape(-1, 42)
    both = p & np.roll(p, 1)
    both[0] = False
    prev = np.roll(np.arange(T), 1)
    feats[both, 42:44] = (wrist[both] - wrist[prev[both]]) / size[both, None]
    feats[both, 44] = np.log(size[both]) - np.log(size[prev[both]])
    if shape_vel:
        dv = np.zeros((T, 42))
        dv[both] = feats[both, :42] - feats[prev[both], :42]
        feats = np.concatenate([feats, dv], 1)
    return feats, present


class Standardizer:
    """Per-dimension z-score using statistics of tracked frames from training subjects."""

    def fit(self, feat_list, present_list):
        X = np.concatenate([f[p] for f, p in zip(feat_list, present_list)])
        self.mean, self.std = X.mean(0), X.std(0) + 1e-6
        return self

    def __call__(self, feats, present):
        z = (feats - self.mean) / self.std
        z[~present] = 0.0
        return z


_landmark_cache = {}


def load_video(video, standardizer=None, cache=True):
    """Standardized features of one video. Raw landmarks are cached (parsing JSON dominates runtime);
    returned arrays are fresh, so callers may modify them."""
    if video not in _landmark_cache or not cache:
        pts = load_landmarks(video)
        if not cache:
            return _finish(pts, standardizer)
        _landmark_cache[video] = pts
    return _finish(_landmark_cache[video], standardizer)


def _finish(pts, standardizer):
    feats, present = frame_features(pts)
    if standardizer is not None:
        feats = standardizer(feats, present)
    return feats, present


# ----------------------------------------------------------------------------- demos
STILL_MAX = 3 * FPS      # demos longer than 3 s are treated as held poses...
STILL_KEEP = int(1.5 * FPS)  # ...and cropped to their central 1.5 s


def cut_demo(feats, present, start, end):
    """Template = the annotated segment; long (held) gestures keep only their central 1.5 s.

    Untracked frames are dropped from the template: an 'empty' template frame carries no information
    about the gesture, and keeping it lets the template match periods where the hand is out of view.
    """
    if end - start + 1 > STILL_MAX:
        mid = (start + end) // 2
        start, end = mid - STILL_KEEP // 2, mid - STILL_KEEP // 2 + STILL_KEEP - 1
    f, p = feats[start:end + 1], present[start:end + 1]
    return f[p], p[p]


MIN_TRACKED_FRAC = 0.5   # a demo candidate needs at least half its frames tracked...
MIN_TEMPLATE = 4         # ...and at least this many tracked frames, else the next instance is used


def pick_demos(videos, classes, standardizer):
    """For each class, the first usable instance in the earliest of `videos` that contains it.
    An instance is unusable if MediaPipe lost the hand for most of it (a real system would ask the
    user to record the demo again).

    Returns {class: (template_feats, template_present, source_video, (start, end))}.
    """
    demos = {}
    for v in videos:
        need = [c for c in classes if c not in demos]
        if not need:
            break
        inst = load_instances(v)
        if not any(c in need for c, _, _ in inst):
            continue
        feats, present = load_video(v, standardizer)
        for c, a, b in inst:
            if c in need and c not in demos:
                f, p = cut_demo(feats, present, a, b)
                if len(f) >= MIN_TEMPLATE and present[a:b + 1].mean() >= MIN_TRACKED_FRAC:
                    demos[c] = (f, p, v, (a, b))
    return demos
