# Final pre-publication review - drone-traffic-analytics

Report-only review, 2026-09-01. Scope: README fact-check against the repo's
own files, robustness scan of `track_count.py` (no fixes applied), and repo
hygiene. Severity scale: **MEDIUM** (fix before publishing), **LOW** (worth
fixing, not blocking), **INFO** (be aware; no action required).

**Verdict: publishable after two MEDIUM items** - one stale number in the
README's headline table, and the decision of what to do about verification
evidence (crop images) that the shipped docs cite but git ignores. No HIGH
findings; working tree is clean; all external links resolve; no debug
leftovers.

## 1. README fact-check

### Verified correct

| Claim (README) | Checked against | Status |
|---|---|---|
| Ground-truth table totals 70 / 41 / 70 | crossing_verification.md sections 1-2, fix_verification.md section 2 | OK |
| Ground truth "72 crossings, 39 left / 33 right" | crossing_verification.md section 6 (post-audit) | OK |
| Final counter "38 left / 32 right" | fix_verification.md sections 1-2 | OK |
| "97% of ground truth" | 70/72 = 97.2% | OK |
| "-17 crossings silently dropped" (zero-product) | 17 recovered tracks, fix_verification.md section 3 | OK |
| "two residual misses ... analyzed in the fix report" | c31, c32 in fix_verification.md section 5/7 | OK |
| "every multi-track event" human-reviewed | all 24 multi-track crops reviewed, crossing_verification.md section 6 | OK |
| "23s" zoom-out clip | videos/traffic2.mp4 probes at 23.5 s | OK (duration only, see below) |
| Setup: weights filename and path | `WEIGHTS = "weights/yolo11n_visdrone_1024.pt"` in track_count.py; release v1.0 contains exactly `yolo11n_visdrone_1024.pt` (checked live) | OK |
| Setup: requirements | requirements.txt (ultralytics>=8.4, opencv-python>=4.10) covers every non-stdlib import in the repo; installed 8.4.137 / 5.0.0 satisfy | OK |
| CLI flags in examples (`--axis v --line 0.5 --show`) | argparse in track_count.py | OK |
| imgsz 1024, conf 0.25, "Release v1.0" | constants in track_count.py; release tag exists (HTTP 200) | OK |
| Links: detector repo, release tag, profile, `analysis/*.md`, `scripts/verify_crossings.py`, `assets/demo.gif` | HTTP 200 / tracked in git | OK |
| Counted classes "(car, truck, van, bus, motorcycle, pedestrian)" | COUNT_CLASSES = {pedestrian, car, van, truck, bus, motor} | OK |

### Findings

* **MEDIUM - stale number in the headline table.** README line 35 says the
  naive counter had "+16 duplicate crossings". After the c03 audit correction,
  crossing_verification.md section 6 says **15** fragment duplicates, and only
  15 reconciles the row's own arithmetic: 72 ground truth + 15 dupes - 17
  dropped = 70 counted (with 16 it gives 71). Change "+16" to "+15".
* **LOW - "logging all 14K trajectories" conflates points with tracks.** The
  CSV holds 14,039 trajectory *points* (tracked boxes) across **339** distinct
  tracks. Suggest "14K trajectory points" or "339 trajectories (14K points)".
* **LOW - the "Split long vehicles" limitation bullet undersells the residual
  errors.** It claims the dedup's "one blind spot" is the nose-to-tail convoy
  (c32) with "one such miss". fix_verification.md documents a second miss of a
  different kind (c31: a car absorbed by a motorcycle's anchor at 23 px, just
  inside the 25 px lane gate) plus a self-cancelling pair in c35 (fragment
  escaped the 15-frame window / distinct truck wrongly merged). The README's
  own table row says "two residual misses", so the bullet is also internally
  inconsistent with it.
* **LOW - "~11-13 FPS ... on a laptop RTX 3070" is not reproducible from this
  environment.** The installed torch is CPU-only (2.13.0+cpu, CUDA
  unavailable), and this review's verification runs measured 11.1-12.3 FPS on
  CPU (inference + CSV logging, no drawing/encode). The range is plausibly a
  CPU number misattributed to the GPU; if it truly came from a CUDA setup,
  that environment is not the one checked in.
* **INFO - "16K+ ids in 23s" is unverifiable from repo artifacts.** The 23 s
  duration matches traffic2.mp4, but no log or CSV in the repo backs the id
  count; verifying it requires re-running the tracker on that clip.
* **INFO - Pexels link returns HTTP 403 to curl.** Bot-blocking, not a broken
  link; loads fine in a browser.

## 2. Robustness scan of track_count.py (report-only)

* **INFO - missing weights file:** fails loud and clear before any side effect
  (`FileNotFoundError: ... 'weights\\...pt'` from ultralytics; verified live).
* **LOW - missing/unreadable video path:** the cv2 probe silently yields
  w=h=0 and fps 0->30, `output/` is created and a broken 0x0 VideoWriter
  opened, before ultralytics finally raises a (clear) FileNotFoundError - no
  early input validation, and side effects precede the error.
* **INFO - `boxes.id` None for the whole video (no tracks):** graceful; the
  annotated video and HUD still render, the counts table and counts.csv are
  written empty, no crash.
* **LOW - `--line 0` or `--line 1` (and anything outside (0,1)):** counts are
  silently zero forever - a center coordinate can never be on both strict
  sides of the frame edge, so the sign-change product is never negative; no
  argparse bounds check or runtime warning tells the user why.
* **LOW - portrait / non-1280px video with `--axis v`:** runs correctly, but
  DEDUP_ACROSS=25 / DEDUP_ALONG=120 / DEDUP_WINDOW=15 are absolute pixels and
  frames tuned to 1280x720 @ ~30 fps (~60 px lanes); at other resolutions,
  aspect ratios, or frame rates the dedup gates silently mis-scale
  (over- or under-merging). Not stated in README or docstring.
* **INFO - relative weights path:** the script must be run from the repo root;
  from any other cwd it dies with FileNotFoundError on the weights.
* **INFO - `--show` on a headless machine:** cv2.imshow raises; without
  `--show` headless operation is fine.
* **INFO - fps fallback:** a source reporting fps=0 gets written at 30 fps,
  changing playback speed of the annotated output.

## 3. Repo hygiene

### Clean

* `git status` is clean - no uncommitted or untracked-unignored files.
* Tracked set is minimal (9 files); `__pycache__/`, venv, weights, videos,
  output, logs, CSV, and crops are all correctly ignored.
* No TODO / FIXME / XXX / HACK / breakpoint / pdb anywhere in tracked files.

### Findings

* **MEDIUM - shipped docs cite evidence that git ignores.** Both tracked
  analysis reports rest on artifacts a public visitor will not have:
  `analysis/trajectories_traffic3.csv` (cited by both reports as the replay
  input) and `analysis/crossing_crops/` - including 41 specific crop filenames
  in crossing_verification.md's tables - are gitignored. The crop images are
  the *human ground-truth evidence* behind the README's headline
  "counts you can trust" table. Options: track the 41 crops (small JPEGs) and
  optionally the ~800 KB CSV, or add a note to the reports that these
  artifacts are generated locally by `scripts/verify_crossings.py`.
  Related: the exact source clip (traffic3.mp4) is not identified in the
  README credits (only the intersection footage is attributed), so the study
  is not reproducible by a visitor even with the scripts.
* **LOW - `assets/demo.gif` is 12.9 MB.** Tracked forever in history and
  downloaded on every clone; GitHub renders it but it is heavy for a README
  asset. Consider compressing (or a linked MP4, which GitHub also embeds).
* **LOW - `scripts/verify_fixes.py` cannot run for a visitor as shipped.** It
  requires the gitignored CSV and deliberately hard-asserts this study's
  exact state (41 clusters, exact BEFORE/AFTER tables), so on any regenerated
  data from a different clip/tracker version it fails loud by design - but
  nothing tells a visitor that. A one-line note in the script docstring or
  README would prevent confusion. (Re-running it also regenerates
  fix_verification.md sections 1-6 and drops the hand-written section 7.)
* **INFO - unused ignore pattern.** `.gitignore` ends with `*.lock`, which
  matches nothing in the repo; harmless.

## Summary

| Severity | Count | Items |
|---|---:|---|
| MEDIUM | 2 | README "+16" -> "+15"; gitignored evidence cited by shipped docs |
| LOW | 8 | 14K wording; limitation bullet vs two misses; RTX 3070 attribution; no video-path validation; degenerate `--line` silent zeros; px-tuned dedup constants; 12.9 MB gif; verify_fixes.py visitor note |
| INFO | 8 | clean failures and environment caveats listed above |

No finding blocks publication outright; the two MEDIUM items are a
one-character-class edit and a decision about shipping (or disclaiming) the
verification evidence.
