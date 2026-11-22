# DTW Nearest-Template Baseline: Full-Scale Results

All numbers come from `exp_full.py` (log: `figures/full_log.txt`, raw numbers: `figures/full_results.json`).
Reproduce with `.venv/bin/python dtw_baseline/exp_full.py` (about 25 s on a laptop).

## 1. Setup

**Task.** The input is a stream of per-frame MediaPipe hand landmarks plus one demonstration per
registered gesture. The output is a set of detections (k, t), each meaning "gesture k occurred,
reported at frame t". A detection is correct if an unmatched labeled instance of gesture k spans
[t_start, t_end] and t_start ≤ t ≤ t_end + δ, with δ = 0.5 s. Each instance can be matched at most once.
Every other detection is a false positive (FP), and every unmatched instance is a miss.
We report event-level precision, recall, F1, and FP per minute.

**Data.** IPN Hand / IPN HandS: 200 videos from 50 subjects (4 videos each), using the raw
MediaPipe Hands landmarks, downsampled to 15 fps. We use the official subject-disjoint split:
37 train and 13 test subjects. DTW has no trainable parameters, so the train subjects act purely
as the validation set for the single detection threshold τ.

**Features (45-d per frame, plus a presence mask).** We keep one hand slot per frame (the largest
detected hand). MediaPipe's left/right label is unreliable on this data: 43% of detections in
left-hand videos are labeled "Right". The features are:
- the 21 (x, y) points relative to the wrist, divided by hand size (wrist to middle-finger MCP);
- the wrist velocity divided by hand size;
- the frame-to-frame change in log hand size.

Gaps of up to 3 frames are linearly interpolated. Features are z-scored with train-subject statistics.

**Demonstrations.** For each subject, one demo video provides the templates: the first usable
annotated instance of each registered gesture. Gestures longer than 3 s keep only their central
1.5 s, and untracked frames are dropped. Detection runs on the subject's other 3 videos. All 4 videos
take turns as the demo video, so every video is evaluated 3 times with 3 different demo sets.
Over the 13 test subjects this gives 355 minutes of evaluated video.

**Matching: open-begin subsequence DTW.** At every frame t, each template is aligned to the
best-matching stream segment that ends at t. The local cost is the Euclidean distance between
frames, or a fixed penalty when either frame is untracked. Each step advances the template by one
frame and the stream by 0, 1, or 2 frames, and a 0-frame step can never follow another. This
slope constraint lets a performance be up to about 2× faster or slower than the demo. Every path
has exactly m cells, so dividing by the template length m is an exact length normalization.

The algorithm runs in O(m) per frame per template and is causal, so it can be computed frame by frame on a live stream.
It is verified against brute-force path enumeration in `tests.py`. Throughput is about 0.3 µs per
frame-template, including I/O.

**Calibration, threshold, and trigger.** Each template's score is divided by its median score on
reference videos from two *other* (train) subjects, excluding its own class there.

A frame matches gesture k if k's calibrated score is the lowest of all templates and below τ.
Gesture k fires after 3 consecutive matching frames. It re-arms only after its own score rises
back above τ.

**Settings.**
- **A:** register and score the 5 gestures TUP, TDN, TLT, TRT, OP2. These are the proposal's
  held-out classes for the learned encoder.
- **B:** register and score all 14 gestures.
- **C:** register all 14, but score only the same 5 as in A, with τ re-tuned on train for those 5.
  This isolates the effect of having more competing templates.

## 2. Main results (13 test subjects)

| Setting | τ | Precision | Recall | **F1** [95% CI] | FP / min | Oracle acc. |
|---|---|---|---|---|---|---|
| A: 5 registered, 5 scored | 0.300 | 0.737 | 0.565 | **0.639** [0.574, 0.698] | 0.44 | 0.80 |
| B: 14 registered, 14 scored | 0.300 | 0.367 | 0.458 | **0.408** [0.352, 0.459] | 9.20 | 0.41 |
| C: 14 registered, same 5 scored | 0.350 | 0.695 | 0.661 | **0.678** [0.600, 0.748] | 0.63 | 0.54 |

- The 95% CIs come from bootstrapping test subjects.
- Per-subject F1 is 0.64 ± 0.11 for A and 0.41 ± 0.09 for B. The worst subject in B scores 0.23.
- *Oracle acc.* is a diagnostic, not a detection metric. Given the true location of each labeled
  instance, it asks whether the instance's own template has the lowest score within
  [t_start, t_end + δ].
- *Latency:* the median detection fires 0.2 s **before** the annotated end of the gesture in A
  and C, because annotations include the trailing part of the movement.

**Per-gesture F1 in setting B, grouped.**

| Group | Gestures | P | R | F1 |
|---|---|---|---|---|
| Dynamic, distinct motion | TUP TDN TLT TRT OP2 ZIN ZOT GRB | 0.51 | 0.57 | **0.54** |
| Pointing family | P1F P2F C1F C2F 2C1 2C2 | 0.30 | 0.39 | **0.34** |

Best gestures in B: OP2 0.81, TLT 0.73, TUP 0.72. Worst: C1F 0.20, C2F 0.22, 2C1 0.29, 2C2 0.32.

## 3. Findings

1. **The number of registered gestures is not what hurts DTW.** Going from 5 to 14 templates does
   not degrade the original 5 gestures. With τ re-tuned, their F1 is 0.678 (C) vs. 0.639 (A). The
   confidence intervals overlap, so we read this as "no degradation" rather than a gain. The extra
   templates seem to act partly as negatives: they absorb motions that would otherwise trigger a
   throw, which allows a looser threshold. Prior work reported accuracy dropping as more custom
   gestures were added; at the scale of 5 to 14 gestures, we do not observe that for DTW.

2. **The drop from F1 0.64 to 0.41 comes from *which* gestures are added.** 71% of all false
   positives in B happen while the user is pointing (P1F 32%, P2F 39%). The most common errors are
   C1F during P1F (416), C2F during P2F (401), 2C2 during P2F (310), and 2C1 during P1F (285).
   In IPN Hand, pointing is effectively the idle state: it fills the time between commands, and
   clicks are performed *from* the pointing pose. A click template is "pointing pose plus a brief
   finger flex". Because DTW averages cost over the whole template, the flex contributes little, so
   long pointing segments contain many sub-segments that look like a click. Single vs. double clicks
   are also nested: 2C1 is classified as C1F 37% of the time, and 2C2 as C2F 35% of the time.

3. **Short templates are over-matched.** GRB (median 0.6 s) is the most common wrong answer for
   throws: TDN→GRB 0.42, TRT→GRB 0.36, TUP→GRB 0.24 in the oracle confusion. GRB also fires 120
   times during non-gesture frames, typically as the hand enters or leaves the view. An open-begin
   matcher lets a short template find a matching piece inside almost any fast motion. This, rather
   than hand tracking, explains the low recall of TDN (0.32) and TRT (0.36): both gestures are
   tracked in 98% and 90% of their frames.

4. **The threshold is fragile.** On test subjects, F1 peaks sharply near τ = 0.3, and precision
   collapses above τ ≈ 0.35 (`figures/full_pr_vs_tau.png`). The τ chosen on train subjects lands
   at the test optimum, so the calibration transfers across people. However, a single global τ
   leaves recall at only about 0.56.

5. **The results vary a lot across users.** Per-subject F1 ranges from 0.38 to 0.81 in A, and
   from 0.23 to 0.56 in B. Demo quality matters: one demonstration per gesture is a noisy reference.
   There are too few left-hand videos in the test set (4) to draw any conclusion about handedness.

## 4. Implications for the learned encoder

The DTW baseline is reasonable for gestures with distinctive motion (F1 0.64–0.68 on the 5 held-out
gestures, at under 1 FP per minute). It fails where gestures share a hand pose and differ in a
small, brief motion, and it is biased toward short templates. These are the cases an encoder
trained with a background or margin loss should improve. The comparison points for the encoder are:
- setting A, 5 held-out gestures: F1 0.639;
- setting B, all 14 gestures: F1 0.408, with a pointing-family F1 of 0.34.

Changes worth testing for DTW itself:
- a per-gesture threshold instead of a global τ;
- a length prior or a minimum-duration rule against short-template bias;
- delayed decisions for nested gestures: wait about 0.5 s and prefer a longer template that covers
  a shorter one;
- treating P1F/P2F as background rather than as gestures to detect.

## 5. Deviations from the proposal

- **Subsequence DTW instead of fixed windows with a Sakoe-Chiba band.** A fixed window of template
  length cuts off slower performances, and a band only makes sense for equal-length sequences.
  The slope constraint (0.5×–2× speed) replaces the band.
- **One hand slot (45-d) instead of left/right slots (92-d).** MediaPipe handedness is unreliable
  here, and two hands appear in only about 1% of frames.
- **Motion features are frame-to-frame changes**, not displacement relative to the window's
  first frame, because there is no fixed window start anymore.
- **Added per-template score calibration and the "re-arm only when the score rises above τ"
  rule.** Without the latter, a single long pointing hold produced many detections.
- **Split.** We use the official 37/13 subject split, with the train subjects used as validation,
  instead of 35/5/10.
