# Which tracker? A measured comparison

The counting pipeline shipped with ByteTrack because it is the Ultralytics
default - an untested choice. This study tests it against four alternatives
on the pipeline's own footage and ground truth, with hypotheses written
down before the runs.

## 1. Why this needs measuring at all

Public MOT benchmarks (MOT17 pedestrians, DanceTrack, SportsMOT) do not
cover this domain: tiny vehicles, top-down drone view, an OOD detector.
Literature expectations going in: BoT-SORT leads accuracy on MOT17
(HOTA 63.7 vs ByteTrack's 60.1 - figures from the Roboflow Trackers
benchmark at trackers.roboflow.com, after per-tracker grid search; the
original papers' MOT17 numbers differ slightly), ByteTrack leads speed,
OC-SORT is strong on erratic motion. The question is whether that ordering
survives contact with our domain.

## 2. Pre-registered hypotheses (written before any run)

- **H1** Static arena: all trackers count within 68-72 (the arena is easy).
- **H2** BoT-SORT produces fewer near-line fragments (less dedup work).
- **H3** ReID adds nothing here (no long occlusions) - an expected null.
- **H4** BoT-SORT's global motion compensation (GMC) collapses the id churn
  on the zoom-out clip.
- **H5** The GMC ablation (gmc_method: none) restores the churn, isolating
  GMC as the cause.
- **H6** ByteTrack is fastest; ReID pays the largest speed cost.

## 3. Setup

Five configs (bytetrack, botsort_gmc, botsort_reid, botsort_nogmc, ocsort -
all stock yamls with only the ablation field changed; ultralytics 8.4.137
ships six trackers, deepocsort/fasttrack/tracktrack left for future work),
two arenas (traffic3: static highway, GT = 72 human-verified crossings;
traffic2: continuous zoom-out, the stress case), fresh model per run (no
tracker state leakage), CPU-only torch, FPS measured on the tracking pass
under one protocol. Counted tables come from an exact replay of the
verified counting logic on each run's logged trajectories; the
bytetrack/traffic3 run reproduced the ground-truth study bit-for-bit
(70 counted, 339 ids, 14,039 boxes) as a determinism check.

## 4. Results

| config | t3 counted (GT 72) | t3 ids | t3 med len | t3 FPS | t2 counted | t2 ids | t2 med len | t2 FPS |
|---|---|---|---|---|---|---|---|---|
| bytetrack | 70 | 339 | 6 | **14.9** | 18 | 1,818 | 8 | 9.9 |
| botsort_gmc | 70 | 323 | 6 | 10.1 | 21 | **1,658** | **11** | 7.3 |
| botsort_reid | 69 | 554 | 4 | 10.4 | 23 | 2,616 | 7 | 7.5 |
| botsort_nogmc | 70 | 323 | 6 | 13.7 | 19 | 1,722 | 9 | 10.2 |
| ocsort | 70 | **302** | **8** | 12.2 | 19 | 1,904 | 9 | 8.7 |

Cross-analysis highlights (full details: `tracker_study_cross.md`):

- **Near-line structure is tracker-invariant on static footage:** every
  config yields the identical 41-cluster / 24-multi crossing structure;
  net raw crossings above the physical 72 range only +14 (ocsort) to +18
  (non-reid botsorts); every config except botsort_reid produces the same
  final counted table (70, 38L/32R).
- **Metric correction:** the "16K+ ids" previously quoted for the zoom-out
  clip was the on-screen id counter. ByteTrack allocates an id to every
  one-frame candidate (17,309 allocations) but only **1,818 distinct ids
  ever appear in output**. The honest churn figure is 1,818 in 23.5s
  (median track 8 frames); the 17.3K is id allocations, ~10x the visible
  churn.
- **ReID diagnosis:** the extra fragmentation is size-concentrated - the
  smallest-area quartile holds +126 of the +231 extra tracks, median track
  length collapsing to 2 frames (vs 9 for botsort_gmc in the same bucket).
  Mechanism: ~35x17px low-texture top-down crops give unstable embeddings,
  so the appearance_thresh 0.8 check vetoes matches IoU alone would make.
  ReID actively works against itself at this object scale, and it is the
  only config that breaks the counted table (69).
- **Churn attribution:** of 1,402 ByteTrack track deaths on the zoom-out
  clip, 42% die at or shrink to the detector's size floor under a strict
  cut (<=200px^2; median box area at death 240px^2 vs 330px^2 overall),
  and a generous 400px^2 floor reclassifies 74% as detector-driven. The
  split is threshold-sensitive and reported as such.

## 5. Hypotheses vs reality

| # | Verdict | The data |
|---|---|---|
| H1 | **Confirmed** | 70/70/69/70/70 - the static arena does not separate trackers |
| H2 | **Refuted (null)** | identical cluster structure across configs; BoT-SORT actually had slightly *more* fragment excess (+18 vs +15); spread of 4 crossings |
| H3 | **Strengthened** | ReID is actively harmful here: 554 vs 302-339 static ids and it breaks the table (69). Speed-wise the two GMC-enabled configs are the two slowest; ReID itself (model: auto) is nearly free |
| H4 | **Refuted - the headline finding** | GMC buys only 4-9% id reduction (1,658 vs 1,722/1,818) at 26-28% speed cost |
| H5 | Consistent with H4's refutation | the ablation shows the same near-null in both directions |
| H6 | **Half-confirmed** | "ByteTrack is fastest" holds (14.9 FPS). "ReID pays the largest cost" is contradicted: botsort_gmc is the slowest config in both arenas (10.1/7.3 vs ReID's 10.4/7.5) - the big cost is GMC's sparse optical flow, not ReID |

**Why H4's refutation matters more than a confirmation would have:** we
assumed the zoom-out churn was camera-motion-driven, so motion compensation
should fix it. It barely helped - because the churn is mostly
detector-driven: as the camera pulls back, vehicles shrink toward the
detector's size floor, detections flicker, and every flicker kills a track.
No association logic can track what the detector cannot see. This is the
same conclusion as the detector project's resolution study, arriving from a
third direction: **object size in pixels is the first-order constraint of
this entire stack.**

## 6. Which tracker when (this domain's answer + literature)

| Situation | Pick | Why |
|---|---|---|
| Static drone footage, throughput matters | **ByteTrack** | counting accuracy identical to alternatives, fastest (+22% over OC-SORT, +48% over stock BoT-SORT) |
| Identity stability matters more than speed | **OC-SORT** | best id economy on static footage (302 ids, longest median tracks there) at mid speed |
| Moving camera | BoT-SORT+GMC, but fix the detector first | GMC's gain (4-9%) is real but second-order next to the detector's size floor, and it costs 26-28% speed |
| Tiny, low-texture objects | **avoid ReID** | unstable embeddings veto correct IoU matches; measured 1.6-1.7x fragmentation |
| Pedestrian-scale MOT (literature) | BoT-SORT(+ReID) | MOT17 HOTA 63.7 vs 60.1 (Roboflow benchmark, grid-searched) - large textured objects are where appearance helps |

## 7. Decision

**ByteTrack stays the default.** On static footage every tracker delivers
the same verified counts, so the fastest one wins; on moving footage the
bottleneck is the detector, which no tracker choice fixes. The default is
now a measured choice instead of an inherited one. OC-SORT is the
documented alternative when id stability is worth ~18% speed.

## 8. Threats to validity

One clip per arena; CPU timing (relative ordering should hold on GPU, the
GMC/ReID overheads are CPU-side compute); class-label GT is partial
(crossing counts are fully verified, per-class attribution inherits the
detector's twin confusion); OC-SORT/newer trackers tested at stock
parameters only - per-tracker grid search (as the literature does) could
shift accuracy rankings, though not the counting-invariance finding.

> Note on evidence: the per-run trajectory CSVs live in
> `analysis/tracker_runs/`, generated locally by
> `scripts/run_tracker_matrix.py` and deliberately not committed; re-run
> the matrix to regenerate them.